#!/usr/bin/env python
"""Deterministic code-smell checks of relax/, against a checked-in baseline (a warning step of the refactor gate).

    python scripts/dev/smell_check.py [--write-baseline] [--format text|json] [--strict] [--root DIR]

Each check is one function ``check_<name>(repo)`` returning findings; its docstring holds the rule sentence
it enforces. The source is read with ``ast`` only: nothing is imported. A finding's key is the check id, the
module, the qualified function and a detail, never a line number, so the baseline
(``scripts/dev/smell_baseline.json``) survives unrelated edits. A normal run prints one summary line per check
(findings, new against the baseline) and the new findings in detail; it exits 0 unless ``--strict`` is given
and there are new findings. ``--write-baseline`` records every current finding.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import json
import re
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path("scripts/dev/smell_baseline.json")

# Check 6: modules allowed to read the environment (paths relative to the repository root; a trailing "/"
# allows a whole directory).
ENV_READ_ALLOWED = (
    "relax/helpers/env_flags.py",
    "relax/command_line.py",
    "relax/commands/",
    "relax/refinement/command_options.py",
)
# Check 7: the command layer, which alone may import relax.diagnostics and relax.parity.
DIAGNOSTICS_IMPORT_ALLOWED = (
    "relax/command_line.py",
    "relax/commands/",
    "relax/refinement/command_options.py",
    "relax/diagnostics/",
    "relax/parity/",
)
DIAGNOSTICS_PACKAGES = ("relax.diagnostics", "relax.parity")
# Check 1: casts whose result rebinding the same name is a self-cast.
SELF_CASTS = frozenset({"int", "float", "bool", "str", "np.asarray", "numpy.asarray"})
# Check 5: thresholds of the near-duplicate scan (as in the deep2 prototype).
EXACT_MIN_STATEMENTS = 3
NEAR_MIN_STATEMENTS = 5
NEAR_JACCARD = 0.8
NEAR_SHINGLE = 5
NEAR_COMMON_SHINGLE = 200
# Check 11: a module is reported when more than this many of its from-imported names are used once.
SINGLE_USE_IMPORT_THRESHOLD = 20

FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


@dataclass(frozen=True)
class Finding:
    check: str
    key: str
    path: str
    line: int
    message: str


class Repo:
    """The parsed Python files of relax/, scripts/ and tests/ under ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.trees: dict[str, ast.Module] = {}
        self.sources: dict[str, str] = {}
        for top in ("relax", "scripts", "tests"):
            for path in sorted((self.root / top).rglob("*.py")):
                rel = path.relative_to(self.root).as_posix()
                text = path.read_text(errors="replace")
                try:
                    self.trees[rel] = ast.parse(text, rel)
                except SyntaxError:
                    continue
                self.sources[rel] = text
        self.modules = {module_name(rel): rel for rel in self.trees if rel.startswith("relax/")}

    def files(self, *tops: str) -> list[str]:
        return [rel for rel in self.trees if rel.split("/", 1)[0] in tops]


def module_name(rel: str) -> str:
    parts = rel.removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def allowed(rel: str, allowlist) -> bool:
    return any(rel.startswith(a) if a.endswith("/") else rel == a for a in allowlist)


def scopes(tree: ast.Module):
    """Yield (qualified name, scope node) for the module ("<module>") and every function, nested included."""
    yield "<module>", tree

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FUNCTION_NODES + (ast.ClassDef,)):
                qual = f"{prefix}{child.name}"
                if isinstance(child, FUNCTION_NODES):
                    yield qual, child
                yield from visit(child, qual + ".")
            else:
                yield from visit(child, prefix)

    yield from visit(tree, "")


def own_nodes(scope):
    """The nodes of one scope, not descending into nested functions, lambdas or classes."""
    stack = list(scope.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, SCOPE_NODES):
            continue
        stack.extend(ast.iter_child_nodes(node))


def unique(findings: list[Finding]) -> list[Finding]:
    """Suffix repeated keys with #2, #3 in line order so every key is distinct and stable."""
    seen = collections.Counter()
    out = []
    for f in sorted(findings, key=lambda f: (f.key, f.path, f.line)):
        seen[f.key] += 1
        key = f.key if seen[f.key] == 1 else f"{f.key}#{seen[f.key]}"
        out.append(Finding(f.check, key, f.path, f.line, f.message))
    return out


def resolve_import(rel: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    base = module_name(rel).split(".")
    if not rel.endswith("__init__.py"):
        base = base[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def imported_modules(rel: str, tree: ast.Module) -> set[str]:
    """Every relax module (or package) a file imports, at any depth, including importlib string targets."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = resolve_import(rel, node)
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
        elif (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) in ("importlib.import_module", "import_module", "__import__")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            out.add(node.args[0].value)
    return {m for m in out if m == "relax" or m.startswith("relax.")}


def with_parents(mods, known) -> set[str]:
    out = set()
    for m in mods:
        parts = m.split(".")
        out.update(".".join(parts[:i]) for i in range(1, len(parts) + 1) if ".".join(parts[:i]) in known)
    return out


MODULE_RE = re.compile(r"\brelax(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def string_module_refs(tree: ast.Module, known) -> set[str]:
    """Relax modules named in string constants (monkeypatch targets, ``-m`` invocations)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in MODULE_RE.findall(node.value):
                parts = m.split(".")
                for i in range(len(parts), 1, -1):
                    if ".".join(parts[:i]) in known:
                        out.add(".".join(parts[:i]))
                        break
    return out


class Graph:
    """Import graph of relax/ and reachability from the entry points (pyproject scripts, command_line, commands)."""

    def __init__(self, repo: Repo):
        known = repo.modules
        self.edges = {m: with_parents(imported_modules(rel, repo.trees[rel]), known) - {m} for m, rel in known.items()}
        self.file_imports = {}
        for rel, tree in repo.trees.items():
            direct = with_parents(imported_modules(rel, tree), known)
            self.file_imports[rel] = direct | string_module_refs(tree, known)
        seeds = {m for m in known if m.startswith("relax.commands.") or m.split(".")[-1].startswith("command_line")}
        pyproject = repo.root / "pyproject.toml"
        if pyproject.exists():
            scripts = tomllib.loads(pyproject.read_text()).get("project", {}).get("scripts", {})
            seeds |= {target.split(":")[0] for target in scripts.values()}
        self.reachable = self.reach(seeds | ({"relax"} if "relax" in known else set()))

    def reach(self, seeds) -> set[str]:
        seen, stack = set(), list(seeds)
        while stack:
            m = stack.pop()
            if m in seen or m not in self.edges:
                continue
            seen.add(m)
            stack.extend(self.edges[m])
        return seen


def check_self_cast(repo: Repo) -> list[Finding]:
    """Convert a value's type once, where it enters relax: never rebind a name to a cast of itself (``x = int(x)``)
    or of the field of the same name (``x = int(options.x)``); the producer stores the right type. [rule 10]"""
    out = []
    for rel in repo.files("relax"):
        mod = module_name(rel)
        for qual, scope in scopes(repo.trees[rel]):
            for node in own_nodes(scope):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target, value = node.targets[0], node.value
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    target, value = node.target, node.value
                else:
                    continue
                if not (isinstance(target, ast.Name) and isinstance(value, ast.Call) and len(value.args) == 1):
                    continue
                cast, arg = ast.unparse(value.func), value.args[0]
                if cast not in SELF_CASTS:
                    continue
                if (isinstance(arg, ast.Name) and arg.id == target.id) or (
                    isinstance(arg, ast.Attribute) and arg.attr == target.id
                ):
                    out.append(
                        Finding(
                            "self-cast",
                            f"self-cast:{mod}:{qual}:{target.id}",
                            rel,
                            node.lineno,
                            f"{ast.unparse(node)[:100]}",
                        )
                    )
    return out


def is_record(cls: ast.ClassDef) -> bool:
    for deco in cls.decorator_list:
        name = ast.unparse(deco.func if isinstance(deco, ast.Call) else deco)
        if name.split(".")[-1] == "dataclass":
            return True
    return any(ast.unparse(b).split(".")[-1] == "NamedTuple" for b in cls.bases)


def field_reads(tree: ast.Module) -> set[str]:
    """Names this file reads as fields: ``.name`` loads, getattr/hasattr strings, keywords of a ``replace`` call,
    and whole string constants (dict keys of asdict, field-name filters)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            out.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
            out.add(node.value)
        elif isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in ("replace", "_replace"):
            out.update(k.arg for k in node.keywords if k.arg)
    return out


def check_record_fields(repo: Repo) -> list[Finding]:
    """A record field is read by relax code; a field nobody reads is deleted, and a field only tests or scripts
    read is deleted with the tests' reads or moved to them. [rule 41]"""
    production, other = set(), set()
    for rel, tree in repo.trees.items():
        (production if rel.startswith("relax/") else other).update(field_reads(tree))
    out = []
    for rel in repo.files("relax"):
        mod = module_name(rel)
        for cls in ast.walk(repo.trees[rel]):
            if not (isinstance(cls, ast.ClassDef) and is_record(cls)):
                continue
            for stmt in cls.body:
                if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                    continue
                if "ClassVar" in ast.unparse(stmt.annotation):
                    continue
                name = stmt.target.id
                if name in production:
                    continue
                check = "field-test-only" if name in other else "field-unread"
                where = "only tests or scripts read it" if name in other else "nothing reads it"
                out.append(
                    Finding(
                        check, f"{check}:{mod}:{cls.name}.{name}", rel, stmt.lineno, f"field {cls.name}.{name}: {where}"
                    )
                )
    return out


def check_unreachable_modules(repo: Repo, graph: Graph | None = None) -> list[Finding]:
    """Every module under relax/ is reachable by imports from an entry point (the console script, command_line,
    relax.commands); a module only tests or scripts import moves to tests/ or scripts/, one nothing imports is
    deleted. [rule 37]"""
    graph = graph or Graph(repo)
    users = collections.defaultdict(lambda: collections.Counter())
    for rel, mods in graph.file_imports.items():
        for m in mods:
            users[m][rel.split("/", 1)[0]] += 1
    out = []
    for m, rel in sorted(repo.modules.items()):
        if m in graph.reachable:
            continue
        u = users[m]
        out.append(
            Finding(
                "unreachable-module",
                f"unreachable-module:{m}",
                rel,
                1,
                f"not reachable from an entry point (importers: relax {u['relax']}, "
                f"scripts {u['scripts']}, tests {u['tests']})",
            )
        )
    return out


def name_counts(node) -> collections.Counter:
    c = collections.Counter()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            c[n.id] += 1
        elif isinstance(n, ast.Attribute):
            c[n.attr] += 1
        elif isinstance(n, ast.alias):
            c[n.name.split(".")[-1]] += 1
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            c.update(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", n.value)))
    return c


def check_dead_defs(repo: Repo, graph: Graph | None = None) -> list[Finding]:
    """Every top-level function and class in relax/ is used by reachable relax code; one used by nothing is
    deleted, one used only by tests, scripts or unreachable modules moves there. [rule 37]"""
    graph = graph or Graph(repo)
    counts = {rel: name_counts(tree) for rel, tree in repo.trees.items()}
    out = []
    for m, rel in repo.modules.items():
        tree = repo.trees[rel]
        for node in tree.body:
            if not isinstance(node, FUNCTION_NODES + (ast.ClassDef,)):
                continue
            name = node.name
            own = counts[rel][name] - name_counts(node)[name]
            uses = collections.Counter()
            for other, mods in graph.file_imports.items():
                if other == rel or m not in mods or not counts[other][name]:
                    continue
                if other.startswith("relax/"):
                    uses["relax" if module_name(other) in graph.reachable else "unreachable"] += 1
                else:
                    uses[other.split("/", 1)[0]] += 1
            if own or uses["relax"]:
                continue
            if not uses:
                check, what = "dead-def", "used by nothing"
            else:
                check = "def-outside-only"
                what = "used only by " + ", ".join(f"{k} ({v} files)" for k, v in sorted(uses.items()))
            out.append(Finding(check, f"{check}:{m}:{name}", rel, node.lineno, f"{name}: {what}"))
    return out


class _Normalise(ast.NodeTransformer):
    def __init__(self):
        self.names = {}

    def _rename(self, name):
        return self.names.setdefault(name, f"v{len(self.names)}")

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self._rename(node.id), ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg, node.annotation = self._rename(node.arg), None
        return node

    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value="S"), node) if isinstance(node.value, str) else node


def _strip_docstring(body):
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return body[1:]
    return body


def check_near_duplicates(repo: Repo) -> list[Finding]:
    """One computation, one function: no two functions in relax/ have the same body after renaming locals, or
    bodies nearly the same (5-token shingle Jaccard >= 0.8); keep one and call it. A labelled oracle is
    exempt. [rule 20]"""
    funcs = []
    for rel in repo.files("relax"):
        mod = module_name(rel)
        for qual, node in scopes(repo.trees[rel]):
            if qual == "<module>":
                continue
            body = _strip_docstring(node.body)
            if not body:
                continue
            fn = ast.FunctionDef(
                name="f",
                args=copy.deepcopy(node.args),
                body=copy.deepcopy(body),
                decorator_list=[],
                returns=None,
                type_params=[],
            )
            tree = _Normalise().visit(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])))
            src = ast.unparse(ast.fix_missing_locations(tree))
            nstmt = sum(isinstance(n, ast.stmt) for b in body for n in ast.walk(b))
            funcs.append(
                dict(
                    id=f"{mod}:{qual}",
                    rel=rel,
                    line=node.lineno,
                    nstmt=nstmt,
                    h=hashlib.md5(src.encode()).hexdigest(),
                    toks=re.findall(r"\w+|[^\s\w]", src),
                )
            )
    out = []
    groups = collections.defaultdict(list)
    for f in funcs:
        groups[f["h"]].append(f)
    for group in groups.values():
        if len(group) < 2 or group[0]["nstmt"] < EXACT_MIN_STATEMENTS:
            continue
        group = sorted(group, key=lambda f: f["id"])
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                out.append(
                    Finding(
                        "near-duplicate",
                        f"near-duplicate:exact:{a['id']}~{b['id']}",
                        a["rel"],
                        a["line"],
                        f"same normalised body ({a['nstmt']} statements) as {b['id']} ({b['rel']}:{b['line']})",
                    )
                )
    cand = [f for f in funcs if f["nstmt"] >= NEAR_MIN_STATEMENTS]
    for f in cand:
        t = f["toks"]
        f["S"] = {tuple(t[i : i + NEAR_SHINGLE]) for i in range(max(1, len(t) - NEAR_SHINGLE + 1))}
    inverted = collections.defaultdict(set)
    for i, f in enumerate(cand):
        for s in f["S"]:
            inverted[s].add(i)
    for i, f in enumerate(cand):
        shared = collections.Counter()
        for s in f["S"]:
            if len(inverted[s]) <= NEAR_COMMON_SHINGLE:
                shared.update(j for j in inverted[s] if j > i)
        for j, c in shared.items():
            g = cand[j]
            jac = c / (len(f["S"]) + len(g["S"]) - c)
            if jac >= NEAR_JACCARD and f["h"] != g["h"]:
                a, b = sorted((f, g), key=lambda x: x["id"])
                out.append(
                    Finding(
                        "near-duplicate",
                        f"near-duplicate:near:{a['id']}~{b['id']}",
                        a["rel"],
                        a["line"],
                        f"Jaccard {jac:.2f} with {b['id']} ({b['rel']}:{b['line']})",
                    )
                )
    return out


def _is_environ(node) -> bool:
    return ast.unparse(node) in ("os.environ", "environ")


def check_env_reads(repo: Repo) -> list[Finding]:
    """The environment is read only in relax/helpers/env_flags.py and the command boundary (``ENV_READ_ALLOWED``);
    code below receives the value as an argument or an options field. [rule 7]"""
    out = []
    for rel in repo.files("relax"):
        if allowed(rel, ENV_READ_ALLOWED):
            continue
        mod = module_name(rel)
        for qual, scope in scopes(repo.trees[rel]):
            for node in own_nodes(scope):
                arg = None
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and _is_environ(node.value):
                    arg = node.slice
                elif (
                    isinstance(node, ast.Call)
                    and node.args
                    and (
                        ast.unparse(node.func) in ("os.getenv", "getenv")
                        or (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "get"
                            and _is_environ(node.func.value)
                        )
                    )
                ):
                    arg = node.args[0]
                elif (
                    isinstance(node, ast.Compare)
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], (ast.In, ast.NotIn))
                    and _is_environ(node.comparators[0])
                ):
                    arg = node.left
                if arg is None:
                    continue
                var = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else ast.unparse(arg)
                out.append(
                    Finding(
                        "env-read",
                        f"env-read:{mod}:{qual}:{var}",
                        rel,
                        node.lineno,
                        f"reads the environment variable {var} in {qual}",
                    )
                )
    return out


def check_diagnostics_imports(repo: Repo) -> list[Finding]:
    """Only the command layer imports relax.diagnostics or relax.parity (``DIAGNOSTICS_IMPORT_ALLOWED``); an
    algorithm module receives what they provide through a port or an argument. [rule 51]"""
    out = []
    for rel in repo.files("relax"):
        if allowed(rel, DIAGNOSTICS_IMPORT_ALLOWED):
            continue
        mod = module_name(rel)
        lines = {}
        for node in ast.walk(repo.trees[rel]):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = resolve_import(rel, node)
                names = [base] + [f"{base}.{a.name}" for a in node.names] if base == "relax" else [base]
            else:
                continue
            for name in names:
                if any(name == p or name.startswith(p + ".") for p in DIAGNOSTICS_PACKAGES):
                    lines.setdefault(name, node.lineno)
        for name, line in lines.items():
            out.append(
                Finding(
                    "diagnostics-import",
                    f"diagnostics-import:{mod}:{name}",
                    rel,
                    line,
                    f"algorithm module imports {name}",
                )
            )
    return out


def check_strip_literal(repo: Repo) -> list[Finding]:
    """Remove a prefix or suffix with ``removeprefix``/``removesuffix``; ``strip``/``rstrip``/``lstrip`` take a set
    of characters, so a literal argument has one character. [rule 46]"""
    out = []
    for rel in repo.files("relax", "scripts"):
        mod = module_name(rel)
        for qual, scope in scopes(repo.trees[rel]):
            for node in own_nodes(scope):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("strip", "rstrip", "lstrip")
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and len(node.args[0].value) > 1
                ):
                    lit = node.args[0].value
                    out.append(
                        Finding(
                            "strip-literal",
                            f"strip-literal:{mod}:{qual}:{node.func.attr}({lit!r})",
                            rel,
                            node.lineno,
                            f".{node.func.attr}({lit!r}) strips characters, not a string",
                        )
                    )
    return out


def _stores(scope) -> collections.Counter:
    """How often each name is bound in a function, nested scopes included (a closure may rebind via nonlocal)."""
    c = collections.Counter()
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            c[node.id] += 1
        elif isinstance(node, (ast.Nonlocal, ast.Global)):
            c.update(dict.fromkeys(node.names, 2))
    return c


def check_alias_locals(repo: Repo) -> list[Finding]:
    """Read a record field where it is used (``options.x``); do not copy a parameter's field into a local that is
    only read. [rules 18, 55]

    Conservative: the record is a parameter other than self/cls and is never rebound, its field is never
    assigned in the function, the local is bound once and is not read inside a nested function, lambda or
    comprehension (hoisting into a closure can be deliberate)."""
    out = []
    for rel in repo.files("relax"):
        mod = module_name(rel)
        for qual, fn in scopes(repo.trees[rel]):
            if qual == "<module>":
                continue
            params = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs} - {"self", "cls"}
            stores = _stores(fn)
            field_stores = {
                ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
            }
            nested_reads = set()
            for node in ast.walk(fn):
                if node is not fn and isinstance(
                    node, SCOPE_NODES + (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
                ):
                    nested_reads.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
            reads = collections.Counter(
                n.id for n in own_nodes(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            )
            aliases = []
            for node in own_nodes(fn):
                if not (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                ):
                    continue
                local, obj = node.targets[0].id, node.value.value.id
                if (
                    obj in params
                    and stores[obj] == 0
                    and stores[local] == 1
                    and reads[local]
                    and local not in nested_reads
                    and ast.unparse(node.value) not in field_stores
                    and local not in params
                ):
                    aliases.append((node, local))
            for node, local in aliases:
                out.append(
                    Finding(
                        "alias-local",
                        f"alias-local:{mod}:{qual}:{local}={ast.unparse(node.value)}",
                        rel,
                        node.lineno,
                        f"{local} = {ast.unparse(node.value)} is only read ({len(aliases)} such aliases in {qual})",
                    )
                )
    return out


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def check_private_imports(repo: Repo) -> list[Finding]:
    """A ``_private`` name stays in its module: no relax module imports one from another relax module
    (``from relax.x import _y``) or reads one through a module it imported (``import relax.x as m``,
    ``from relax import x``, ``import relax.x``, then ``m._y``, ``x._y``, ``relax.x._y``); make the name public
    or move its user.

    A module alias is matched by name over the whole file, so a local that shadows it could be misread; relax
    code does not rebind module aliases."""
    known = repo.modules
    out = []
    for rel in repo.files("relax"):
        mod, tree = module_name(rel), repo.trees[rel]
        found = {}
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                src = resolve_import(rel, node)
                if not (src == "relax" or src.startswith("relax.")):
                    continue
                for a in node.names:
                    if f"{src}.{a.name}" in known:
                        aliases[a.asname or a.name] = f"{src}.{a.name}"
                    if src != mod and _is_private(a.name):
                        found.setdefault((src, a.name), node.lineno)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname and a.name in known:
                        aliases[a.asname] = a.name
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and _is_private(node.attr)):
                continue
            base = ast.unparse(node.value)
            src = aliases.get(base, base if base in known else None)
            if src is not None and src != mod:
                found.setdefault((src, node.attr), node.lineno)
        for (src, name), line in found.items():
            out.append(Finding("private-import", f"private-import:{mod}:{src}:{name}", rel, line, f"uses {src}.{name}"))
    return out


def check_single_use_imports(repo: Repo) -> list[Finding]:
    """Import a name where it pays: a relax module with more than ``SINGLE_USE_IMPORT_THRESHOLD`` names it
    from-imports (relax or third-party; not the standard library or ``__future__``) and uses exactly once has
    grown a wide, shallow interface; split the module or call through the owning module.

    One finding per (module, name), only for modules over the threshold, so a key does not change when an
    unrelated import is added or removed; the count is in each finding's message. A use is a ``Name`` load
    anywhere in the file; a name imported only to re-export (zero uses) is not single-use."""
    out = []
    for rel in repo.files("relax"):
        mod, tree = module_name(rel), repo.trees[rel]
        imported = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            src = resolve_import(rel, node)
            if src.split(".")[0] in sys.stdlib_module_names or src == "__future__":
                continue
            for a in node.names:
                if a.name != "*":
                    imported.setdefault(a.asname or a.name, (src, node.lineno))
        uses = collections.Counter(
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        )
        single = sorted(name for name in imported if uses[name] == 1)
        if len(single) <= SINGLE_USE_IMPORT_THRESHOLD:
            continue
        for name in single:
            src, line = imported[name]
            out.append(
                Finding(
                    "single-use-import",
                    f"single-use-import:{mod}:{name}",
                    rel,
                    line,
                    f"{name} (from {src}) is used once; {len(single)} single-use imports in {mod}",
                )
            )
    return out


CHECKS = {
    "self-cast": check_self_cast,
    "record-fields": check_record_fields,
    "unreachable-module": check_unreachable_modules,
    "dead-defs": check_dead_defs,
    "near-duplicate": check_near_duplicates,
    "env-read": check_env_reads,
    "diagnostics-import": check_diagnostics_imports,
    "strip-literal": check_strip_literal,
    "alias-local": check_alias_locals,
    "private-import": check_private_imports,
    "single-use-import": check_single_use_imports,
}
# The finding ids each check may emit, in report order.
CHECK_IDS = (
    "self-cast",
    "field-unread",
    "field-test-only",
    "unreachable-module",
    "dead-def",
    "def-outside-only",
    "near-duplicate",
    "env-read",
    "diagnostics-import",
    "strip-literal",
    "alias-local",
    "private-import",
    "single-use-import",
)


def run_checks(root: Path) -> list[Finding]:
    repo = Repo(root)
    graph = Graph(repo)
    findings = []
    for check in CHECKS.values():
        if check in (check_unreachable_modules, check_dead_defs):
            findings += check(repo, graph)
        else:
            findings += check(repo)
    return unique(findings)


def load_baseline(path: Path) -> set[str]:
    return set(json.loads(path.read_text())["findings"]) if path.exists() else set()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--baseline", type=Path, default=None, help="default: <root>/scripts/dev/smell_baseline.json")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true", help="exit 1 when there are new findings")
    args = parser.parse_args(argv)
    baseline_path = args.baseline or args.root / BASELINE
    start = time.monotonic()
    findings = run_checks(args.root)
    if args.write_baseline:
        baseline_path.write_text(json.dumps({"findings": sorted(f.key for f in findings)}, indent=1) + "\n")
    baseline = load_baseline(baseline_path)
    new = [f for f in findings if f.key not in baseline]
    by_check = collections.Counter(f.check for f in findings)
    new_by_check = collections.Counter(f.check for f in new)
    if args.format == "json":
        json.dump(
            {
                "summary": {c: {"findings": by_check[c], "new": new_by_check[c]} for c in CHECK_IDS},
                "new": [vars(f) for f in new],
                "findings": [vars(f) for f in findings],
            },
            sys.stdout,
            indent=1,
        )
        print()
    else:
        for c in CHECK_IDS:
            print(f"{c}: {by_check[c]} findings, {new_by_check[c]} new vs baseline")
        for f in sorted(new, key=lambda f: (f.check, f.path, f.line)):
            print(f"NEW {f.path}:{f.line} [{f.check}] {f.message}  ({f.key})")
        print(f"smell check: {len(findings)} findings, {len(new)} new vs baseline ({time.monotonic() - start:.1f} s)")
    return 1 if args.strict and new else 0


if __name__ == "__main__":
    sys.exit(main())
