#!/usr/bin/env python
"""Static call-arity check: calls into relax that omit a required argument or pass a wrong one.

    python scripts/dev/call_arity_check.py [ROOT]     # findings of the tree at ROOT (default: this repository)

The CPU fingerprint chain runs stand-in engines, so a call that only a GPU path executes can omit a keyword
made required on its callee (``engine_tuning``, ``scoring_overrides``) and still pass the chain. This reads
every ``relax/``, ``scripts/`` and ``tests/`` file with ``ast`` (it imports nothing) and checks each call whose
callee it resolves without doubt to a function or class defined in ``relax/`` or ``scripts/lib/``:

* the callee is a module-level function, a class (its ``__init__``, or the fields of a dataclass or a
  ``NamedTuple``) or a static or class method called through its class;
* the call names it through a from-import, a module attribute (``mod.f`` with ``mod`` imported from the
  repository), a dotted ``import relax.a.b`` path or a definition of the same module, in a scope where no
  other binding shadows the name;
* the callee carries no decorator that may change its signature (``jax.jit``, ``functools.partial(jax.jit,
  ...)``, ``lru_cache``, ``contextmanager`` and the like keep it).

It reports (a) a required argument that the call does not pass, (b) a keyword the callee does not accept and
(c) more positional arguments than the callee takes. A ``**mapping`` splat counts with its keys when they
are known: a dict display, ``dict(...)``, a local name assigned from those (with its ``name["key"] =``,
``name.update(...)`` and ``name |= ...`` stores before the call, and the stores of a resolved function it is
passed to) or a call of a resolved function that returns one. For (a) a splat counts with every key it may
carry, and a splat of unknown keys turns (a) off; for (b) it counts with the keys it surely carries. A
``*args`` splat turns off (c) and the positional part of (a). Calls inside ``with pytest.raises(TypeError)``
are skipped. Each finding is ``path: callee: problem`` without a line number, sorted, so that the gate diffs
a head report against its base report. Exit 0.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCAN_DIRS = ("relax", "scripts", "tests")
CALLEE_DIRS = ("relax/", "scripts/lib/")
INF = float("inf")
# Builtins and mapping methods that read a mapping without storing keys into it.
READ_ONLY_CALLS = frozenset(
    "dict len sorted list set tuple frozenset print repr str bool any all iter enumerate zip isinstance id "
    "type hash next min max sum format".split()
)
READ_ONLY_METHODS = frozenset({"get", "items", "keys", "values", "copy", "format", "join", "info", "debug", "warning"})
# Decorators that keep the signature of the function they wrap.
SIGNATURE_KEEPING = frozenset(
    "jit jax.jit lru_cache functools.lru_cache cache functools.cache contextmanager contextlib.contextmanager "
    "staticmethod classmethod full_float32 residual_statistics_precision".split()
)
PARTIAL = frozenset({"partial", "functools.partial"})
NAMEDTUPLE = frozenset({"NamedTuple", "typing.NamedTuple"})
DATACLASS = frozenset({"dataclass", "dataclasses.dataclass"})
FIELD = frozenset({"field", "dataclasses.field"})
LEAVES = (ast.Name, ast.Constant, ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop, ast.Pass)
CONDITIONAL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.IfExp, ast.BoolOp, ast.match_case)


def dotted(node) -> str | None:
    """``a.b.c`` of a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def decorator_name(deco) -> str | None:
    return dotted(deco.func if isinstance(deco, ast.Call) else deco)


def const_str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


@dataclass
class Signature:
    positional: list  # parameter names, positional-only ones first
    posonly: set
    defaults: set  # positional names that carry a default
    kwonly: dict  # name -> required
    varargs: bool
    varkw: bool


def function_signature(node, drop_first: bool) -> Signature:
    a = node.args
    pos = [p.arg for p in a.posonlyargs + a.args]
    defaults = set(pos[len(pos) - len(a.defaults) :]) if a.defaults else set()
    posonly = {p.arg for p in a.posonlyargs}
    if drop_first and pos:
        posonly.discard(pos[0])
        defaults.discard(pos[0])
        pos = pos[1:]
    kwonly = {p.arg: d is None for p, d in zip(a.kwonlyargs, a.kw_defaults)}
    return Signature(pos, posonly, defaults, kwonly, a.vararg is not None, a.kwarg is not None)


@dataclass
class Binding:
    lineno: float
    kind: str  # import | def | class | param | assign | other | global
    payload: object = None


@dataclass
class Event:
    """A use of a local name that may change the keys of the mapping it holds."""

    lineno: float
    kind: str  # setitem | update | ior | remove | alias | escape
    payload: object
    conditional: bool


@dataclass
class Scope:
    node: object
    parent: Scope | None
    module: Module
    kind: str  # module | function | class | comprehension
    bindings: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)  # name -> [Event]
    children: list = field(default_factory=list)

    def bind(self, name, lineno, kind, payload=None):
        self.bindings.setdefault(name, []).append(Binding(lineno, kind, payload))


@dataclass
class Module:
    name: str
    rel: str
    tree: ast.Module | None
    checkable: bool
    scope: Scope | None = None
    scopes: dict = field(default_factory=dict)  # id(function node) -> Scope
    calls: list = field(default_factory=list)  # (call, scope)


def raises_type_error(item: ast.withitem) -> bool:
    call = item.context_expr
    if not (isinstance(call, ast.Call) and (dotted(call.func) or "").endswith("raises") and call.args):
        return False
    first = call.args[0]
    names = [dotted(e) for e in (first.elts if isinstance(first, ast.Tuple) else [first])]
    return any(n in ("TypeError", "Exception", "BaseException") for n in names)


def bind_params(scope: Scope, args: ast.arguments, lineno):
    for p in args.posonlyargs + args.args + args.kwonlyargs + [x for x in (args.vararg, args.kwarg) if x]:
        scope.bind(p.arg, lineno, "param")


class ScopeBuilder:
    """Collects the bindings, mapping events and calls of every scope of one module."""

    def __init__(self, module: Module):
        self.module = module
        self.skip = False  # inside ``with pytest.raises(TypeError)``
        self.conditional = False  # inside a branch or loop of the current scope

    def build(self):
        self.module.scope = Scope(self.module.tree, None, self.module, "module")
        for stmt in self.module.tree.body:
            self.visit(stmt, self.module.scope)

    def new_scope(self, node, parent, kind):
        scope = Scope(node, parent, self.module, kind)
        self.module.scopes[id(node)] = scope
        parent.children.append(scope)
        return scope

    def event(self, scope, name, lineno, kind, payload=None):
        scope.events.setdefault(name, []).append(Event(lineno, kind, payload, self.conditional))

    def bind_target(self, target, scope, lineno, value=None):
        if isinstance(target, ast.Name):
            scope.bind(target.id, lineno, "assign" if value is not None else "other", value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self.bind_target(elt, scope, lineno)
        elif isinstance(target, ast.Starred):
            self.bind_target(target.value, scope, lineno)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            self.event(scope, target.value.id, lineno, "setitem", const_str(target.slice))

    def in_new_scope(self, inner, nodes):
        saved = self.conditional
        self.conditional = False
        for node in nodes:
            self.visit(node, inner)
        self.conditional = saved

    def visit(self, node, scope: Scope):
        if isinstance(node, LEAVES):
            return
        lineno = getattr(node, "lineno", 0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            is_class = isinstance(node, ast.ClassDef)
            scope.bind(node.name, lineno, "class" if is_class else "def", node)
            outer = list(node.decorator_list)
            if is_class:
                outer += node.bases + [k.value for k in node.keywords]
            else:
                outer += node.args.defaults + [d for d in node.args.kw_defaults if d]
            for child in outer:
                self.visit(child, scope)
            inner = self.new_scope(node, scope, "class" if is_class else "function")
            if not is_class:
                bind_params(inner, node.args, lineno)
            self.in_new_scope(inner, node.body)
            return
        if isinstance(node, ast.Lambda):
            inner = self.new_scope(node, scope, "function")
            bind_params(inner, node.args, lineno)
            self.in_new_scope(inner, [node.body])
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            inner = self.new_scope(node, scope, "comprehension")
            for gen in node.generators:
                self.bind_target(gen.target, inner, lineno)
            parts = [g.iter for g in node.generators] + [c for g in node.generators for c in g.ifs]
            parts += [getattr(node, p) for p in ("elt", "key", "value") if hasattr(node, p)]
            self.in_new_scope(inner, parts)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    scope.bind(alias.asname, lineno, "import", ("module", alias.name))
                else:
                    top = alias.name.split(".")[0]
                    scope.bind(top, lineno, "import", ("module", top))
            return
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    target = ("attr", node.module or "", node.level, alias.name)
                    scope.bind(alias.asname or alias.name, lineno, "import", target)
            return
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                scope.bind(name, lineno, "global")
            return
        if isinstance(node, ast.Assign):
            single = len(node.targets) == 1
            for target in node.targets:
                self.bind_target(target, scope, lineno, node.value if single else None)
            if isinstance(node.value, ast.Name):
                self.event(scope, node.value.id, lineno, "alias")
            for child in [node.value] + [t for t in node.targets if not isinstance(t, ast.Name)]:
                self.visit(child, scope)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.bind_target(node.target, scope, lineno, node.value)
                self.visit(node.value, scope)
            return
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                if isinstance(node.op, ast.BitOr):
                    self.event(scope, node.target.id, lineno, "ior", node.value)
                else:
                    scope.bind(node.target.id, lineno, "other")
            self.visit(node.value, scope)
            return
        if isinstance(node, ast.NamedExpr):
            self.bind_target(node.target, scope, lineno)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self.bind_target(node.target, scope, lineno)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    self.bind_target(item.optional_vars, scope, lineno)
                self.visit(item.context_expr, scope)
            saved = self.skip
            self.skip = saved or any(raises_type_error(item) for item in node.items)
            for stmt in node.body:
                self.visit(stmt, scope)
            self.skip = saved
            return
        elif isinstance(node, ast.ExceptHandler) and node.name:
            scope.bind(node.name, lineno, "other")
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scope.bind(target.id, lineno, "other")
                elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    self.event(scope, target.value.id, lineno, "remove")
        elif isinstance(node, ast.Call):
            self.note_call(node, scope, lineno)
            if not self.skip:
                self.module.calls.append((node, scope))
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            scope.bind(node.name, lineno, "other")
        elif isinstance(node, ast.MatchMapping) and node.rest:
            scope.bind(node.rest, lineno, "other")
        saved = self.conditional
        if isinstance(node, CONDITIONAL):
            self.conditional = True
        for child in ast.iter_child_nodes(node):
            self.visit(child, scope)
        self.conditional = saved

    def note_call(self, call: ast.Call, scope: Scope, lineno):
        func = call.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            name, attr = func.value.id, func.attr
            if attr == "update":
                self.event(scope, name, lineno, "update", call)
            elif attr == "setdefault":
                self.event(scope, name, lineno, "setitem", const_str(call.args[0]) if call.args else None)
            elif attr in ("pop", "popitem", "clear", "__delitem__"):
                self.event(scope, name, lineno, "remove")
            elif attr == "__setitem__":
                self.event(scope, name, lineno, "setitem", None)
        for index, arg in enumerate(call.args):
            if isinstance(arg, ast.Name):
                self.event(scope, arg.id, lineno, "escape", (call, index))
            elif isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name):
                self.event(scope, arg.value.id, lineno, "escape", (call, None))
        for kw in call.keywords:
            if kw.arg is not None and isinstance(kw.value, ast.Name):
                self.event(scope, kw.value.id, lineno, "escape", (call, kw.arg))


@dataclass
class Symbol:
    kind: str  # module | function | class | method
    module: Module
    node: object = None
    owner: object = None  # the class of a method


@dataclass
class Keys:
    """The keys a mapping may carry (None: unknown) and the keys it surely carries."""

    possible: set | None
    certain: set


UNKNOWN = Keys(None, set())


def union(parts) -> Keys:
    possible, certain = set(), set()
    for part in parts:
        possible = None if possible is None or part.possible is None else possible | part.possible
        certain |= part.certain
    return Keys(possible, certain)


def either(parts) -> Keys:
    """Keys of one of several alternatives."""
    joined = union(parts)
    certain = set.intersection(*(p.certain for p in parts)) if parts else set()
    return Keys(joined.possible, certain)


class Index:
    def __init__(self, root: Path):
        self.root = root
        self.modules: dict[str, Module] = {}
        for top in SCAN_DIRS:
            base = root / top
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.py")):
                rel = path.relative_to(root).as_posix()
                parts = rel[:-3].split("/")
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                try:
                    tree = ast.parse(path.read_text(), filename=rel)
                except (SyntaxError, UnicodeDecodeError, ValueError):
                    continue
                for i in range(1, len(parts)):  # packages without __init__.py
                    self.modules.setdefault(".".join(parts[:i]), Module(".".join(parts[:i]), "", None, False))
                self.modules[".".join(parts)] = Module(".".join(parts), rel, tree, rel.startswith(CALLEE_DIRS))
        for module in self.modules.values():
            if module.tree is None:
                module.scope = Scope(None, None, module, "module")
            else:
                ScopeBuilder(module).build()

    # ---- name resolution -------------------------------------------------------------------------------
    @staticmethod
    def package_of(module: Module) -> str:
        return module.name if module.rel.endswith("__init__.py") else module.name.rpartition(".")[0]

    def find_module(self, name: str, importer: Module) -> Module | None:
        """An absolute import as the repository's entry points see it (tests/ and the test's folder on sys.path)."""
        for candidate in (name, f"{self.package_of(importer)}.{name}", f"tests.{name}"):
            if candidate in self.modules:
                return self.modules[candidate]
        return None

    def import_target(self, target, importer: Module, depth: int):
        if target[0] == "module":
            module = self.find_module(target[1], importer)
            return Symbol("module", module) if module else None
        _, name, level, attr = target
        if level:
            package = self.package_of(importer)
            for _ in range(level - 1):
                package = package.rpartition(".")[0]
            module = self.modules.get(f"{package}.{name}" if name else package)
        else:
            module = self.find_module(name, importer)
        return self.module_attr(module, attr, depth + 1) if module else None

    def from_bindings(self, bindings, scope: Scope, depth: int):
        """The symbol a name binds when every binding agrees, else None."""
        kinds = {b.kind for b in bindings}
        if kinds == {"import"}:
            targets = {b.payload for b in bindings}
            return self.import_target(next(iter(targets)), scope.module, depth) if len(targets) == 1 else None
        if len(bindings) == 1 and kinds <= {"def", "class"} and scope.kind == "module":
            return Symbol("function" if kinds == {"def"} else "class", scope.module, bindings[0].payload)
        return None

    def module_attr(self, module: Module, name: str, depth: int = 0):
        if depth > 12 or module is None:
            return None
        bindings = module.scope.bindings.get(name)
        if bindings:
            return self.from_bindings(bindings, module.scope, depth)
        sub = self.modules.get(f"{module.name}.{name}")
        return Symbol("module", sub) if sub else None

    def lookup(self, scope: Scope, name: str):
        """(binding scope, bindings) of a name read in a scope, Python's rules for class scopes included."""
        first = scope
        while scope is not None:
            if scope.kind != "class" or scope is first:
                bindings = scope.bindings.get(name)
                if bindings:
                    if any(b.kind == "global" for b in bindings):
                        module_scope = scope.module.scope
                        return module_scope, module_scope.bindings.get(name, [])
                    return scope, bindings
            scope = scope.parent
        return None, None

    def builtin(self, scope: Scope, name: str) -> bool:
        return self.lookup(scope, name)[0] is None

    def resolve(self, expr, scope: Scope):
        if isinstance(expr, ast.Name):
            where, bindings = self.lookup(scope, expr.id)
            return self.from_bindings(bindings, where, 0) if where is not None and bindings else None
        if isinstance(expr, ast.Attribute):
            base = self.resolve(expr.value, scope)
            if base is None:
                return None
            if base.kind == "module":
                return self.module_attr(base.module, expr.attr)
            if base.kind == "class":
                for stmt in base.node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == expr.attr:
                        return Symbol("method", base.module, stmt, base.node)
        return None

    # ---- signatures ------------------------------------------------------------------------------------
    @staticmethod
    def keeps_signature(node) -> bool:
        for deco in node.decorator_list:
            name = decorator_name(deco)
            if (
                name in PARTIAL
                and isinstance(deco, ast.Call)
                and deco.args
                and dotted(deco.args[0]) in ("jit", "jax.jit")
            ):
                continue
            if name not in SIGNATURE_KEEPING:
                return False
        return True

    def signature(self, symbol: Symbol | None):
        """(qualified name, Signature) of a checkable callee, else None."""
        if symbol is None or not symbol.module.checkable:
            return None
        node = symbol.node
        if symbol.kind == "function":
            return (
                (f"{symbol.module.name}.{node.name}", function_signature(node, False))
                if self.keeps_signature(node)
                else None
            )
        if symbol.kind == "method":
            names = {dotted(d) for d in node.decorator_list}
            if not self.keeps_signature(node) or not names & {"staticmethod", "classmethod"}:
                return None
            return f"{symbol.module.name}.{symbol.owner.name}.{node.name}", function_signature(
                node, "classmethod" in names
            )
        if symbol.kind == "class":
            sig = self.class_signature(symbol, 0)
            return (f"{symbol.module.name}.{node.name}", sig) if sig else None
        return None

    def class_signature(self, symbol: Symbol, depth: int):
        node = symbol.node
        if depth > 8 or any(k.arg == "metaclass" for k in node.keywords):
            return None
        if any(decorator_name(d) not in DATACLASS for d in node.decorator_list):
            return None
        members = {s.name: s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if "__new__" in members:
            return None
        if "__init__" in members:
            init = members["__init__"]
            return function_signature(init, True) if self.keeps_signature(init) else None
        if node.decorator_list:
            fields = self.dataclass_fields(symbol, depth)
            return None if fields is None else self.fields_signature(fields)
        bases = [dotted(b) for b in node.bases]
        if bases and set(bases) <= NAMEDTUPLE:
            fields = self.own_fields(node, kw_only=False)
            return None if fields is None else self.fields_signature(fields)
        if not bases:
            return Signature([], set(), set(), {}, False, False)
        if len(bases) == 1:
            base = self.resolve(node.bases[0], symbol.module.scope)
            if base is not None and base.kind == "class" and base.module.checkable:
                return self.class_signature(base, depth + 1)
        return None

    def dataclass_fields(self, symbol: Symbol, depth: int):
        """[(name, has default, keyword-only)] of a dataclass's generated ``__init__``, else None."""
        node = symbol.node
        if depth > 8 or any(isinstance(s, ast.FunctionDef) and s.name in ("__init__", "__new__") for s in node.body):
            return None
        decos = node.decorator_list
        if len(decos) != 1 or decorator_name(decos[0]) not in DATACLASS:
            return None
        options = {k.arg: k.value for k in decos[0].keywords} if isinstance(decos[0], ast.Call) else {}
        if (
            any(not isinstance(v, ast.Constant) for v in options.values())
            or "init" in options
            and not options["init"].value
        ):
            return None
        merged = {}
        for base in node.bases:
            if dotted(base) == "object":
                continue
            base_symbol = self.resolve(base, symbol.module.scope)
            if base_symbol is None or base_symbol.kind != "class":
                return None
            base_fields = self.dataclass_fields(base_symbol, depth + 1)
            if base_fields is None:
                return None
            merged.update({f[0]: f for f in base_fields})
        own = self.own_fields(node, "kw_only" in options and options["kw_only"].value is True)
        if own is None:
            return None
        merged.update({f[0]: f for f in own})
        return list(merged.values())

    @staticmethod
    def own_fields(node, kw_only: bool):
        """[(name, has default, keyword-only)] of the annotated class-body fields, else None."""
        fields = []
        for stmt in node.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            ann = stmt.annotation
            ann_name = (dotted(ann.value if isinstance(ann, ast.Subscript) else ann) or "").split(".")[-1]
            if ann_name == "ClassVar":
                continue
            if ann_name == "KW_ONLY":
                kw_only = True
                continue
            value = stmt.value
            has_default, field_kw_only = value is not None, kw_only
            if isinstance(value, ast.Call) and dotted(value.func) in FIELD:
                options = {k.arg: k.value for k in value.keywords}
                if None in options or any(
                    not isinstance(options[k], ast.Constant) for k in ("init", "kw_only") if k in options
                ):
                    return None
                if "init" in options and options["init"].value is False:
                    continue
                has_default = "default" in options or "default_factory" in options
                if "kw_only" in options:
                    field_kw_only = options["kw_only"].value is True
            fields.append((stmt.target.id, has_default, field_kw_only))
        return fields

    @staticmethod
    def fields_signature(fields) -> Signature:
        positional = [f[0] for f in fields if not f[2]]
        defaults = {f[0] for f in fields if not f[2] and f[1]}
        kwonly = {f[0]: not f[1] for f in fields if f[2]}
        return Signature(positional, set(), defaults, kwonly, False, False)

    # ---- mapping keys ----------------------------------------------------------------------------------
    def dict_keys(self, expr, scope: Scope, lineno, depth: int = 0) -> Keys:
        if depth > 8:
            return UNKNOWN
        if isinstance(expr, ast.Dict):
            parts = []
            for key, value in zip(expr.keys, expr.values):
                if key is None:
                    parts.append(self.dict_keys(value, scope, lineno, depth + 1))
                elif const_str(key) is not None:
                    parts.append(Keys({key.value}, {key.value}))
                else:
                    parts.append(UNKNOWN)
            return union(parts)
        if isinstance(expr, ast.IfExp):
            return either([self.dict_keys(e, scope, lineno, depth + 1) for e in (expr.body, expr.orelse)])
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            return union([self.dict_keys(e, scope, lineno, depth + 1) for e in (expr.left, expr.right)])
        if isinstance(expr, ast.Name):
            return self.name_keys(expr.id, scope, lineno, depth)
        if not isinstance(expr, ast.Call):
            return UNKNOWN
        func = expr.func
        if isinstance(func, ast.Name) and func.id == "dict" and self.builtin(scope, "dict"):
            if len(expr.args) > 1 or any(isinstance(a, ast.Starred) for a in expr.args):
                return UNKNOWN
            parts = [self.dict_keys(a, scope, lineno, depth + 1) for a in expr.args]
            for kw in expr.keywords:
                parts.append(
                    self.dict_keys(kw.value, scope, lineno, depth + 1) if kw.arg is None else Keys({kw.arg}, {kw.arg})
                )
            return union(parts)
        if isinstance(func, ast.Attribute) and func.attr == "copy" and not expr.args and not expr.keywords:
            return self.dict_keys(func.value, scope, lineno, depth + 1)
        symbol = self.resolve(func, scope)
        if symbol is not None and symbol.kind == "function" and symbol.module.tree is not None:
            return self.returned_keys(symbol, depth + 1)
        return UNKNOWN

    def returned_keys(self, symbol: Symbol, depth: int) -> Keys:
        inner = symbol.module.scopes.get(id(symbol.node))
        returns, stack = [], list(symbol.node.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(node, (ast.Yield, ast.YieldFrom)) or isinstance(node, ast.Return) and node.value is None:
                return UNKNOWN
            if isinstance(node, ast.Return):
                returns.append(node)
            stack.extend(ast.iter_child_nodes(node))
        if not returns or inner is None:
            return UNKNOWN
        return either([self.dict_keys(r.value, inner, r.lineno, depth + 1) for r in returns])

    @staticmethod
    def name_events(where: Scope, name: str, limit):
        """The events of a name in its binding scope and in the nested scopes that do not rebind it."""
        events = [e for e in where.events.get(name, []) if e.lineno < limit]
        stack = list(where.children)
        while stack:
            scope = stack.pop()
            if name in scope.bindings:
                continue
            # A comprehension runs where it stands; a nested function may run at any time.
            cutoff = limit if scope.kind == "comprehension" else INF
            events += [
                Event(e.lineno, e.kind, e.payload, True) for e in scope.events.get(name, []) if e.lineno < cutoff
            ]
            stack.extend(scope.children)
        return events

    def name_keys(self, name: str, scope: Scope, lineno, depth: int) -> Keys:
        where, bindings = self.lookup(scope, name)
        if where is None:
            return UNKNOWN
        limit = lineno if where is scope or scope.kind == "comprehension" else INF
        before = [b for b in bindings if b.lineno < limit]
        if not before or any(b.kind != "assign" for b in before):
            return UNKNOWN
        keys = either([self.dict_keys(b.payload, where, b.lineno, depth + 1) for b in before])
        possible, certain = keys.possible, set(keys.certain)
        for event in sorted(self.name_events(where, name, limit), key=lambda e: e.lineno):
            if event.kind == "remove":
                certain = set()
                continue
            added = self.event_keys(event, where, depth)
            if added.possible is None:
                possible = None
            elif possible is not None:
                possible |= added.possible
            if added.certain is None:
                certain = set()
            elif not event.conditional:
                certain |= added.certain
        return Keys(possible, certain)

    def event_keys(self, event: Event, scope: Scope, depth: int) -> Keys:
        """The keys an event may add to a local mapping (certain None: it may remove keys)."""
        kind, payload = event.kind, event.payload
        if kind == "setitem":
            return Keys(None, set()) if payload is None else Keys({payload}, {payload})
        if kind == "ior":
            return self.dict_keys(payload, scope, event.lineno, depth + 1)
        if kind == "update":
            parts = [
                UNKNOWN if isinstance(a, ast.Starred) else self.dict_keys(a, scope, event.lineno, depth + 1)
                for a in payload.args
            ]
            for kw in payload.keywords:
                parts.append(
                    self.dict_keys(kw.value, scope, event.lineno, depth + 1)
                    if kw.arg is None
                    else Keys({kw.arg}, {kw.arg})
                )
            return union(parts)
        if kind == "escape":
            return self.escape_keys(payload, scope, depth)
        return Keys(None, None)  # alias: anything may happen to it

    def escape_keys(self, payload, scope: Scope, depth: int) -> Keys:
        """Keys a call that receives the mapping as an argument may store into it."""
        call, slot = payload
        func = call.func
        if isinstance(func, ast.Name) and func.id in READ_ONLY_CALLS and self.builtin(scope, func.id):
            return Keys(set(), set())
        if isinstance(func, ast.Attribute) and func.attr in READ_ONLY_METHODS:
            return Keys(set(), set())
        symbol = self.resolve(func, scope)
        if symbol is None or symbol.kind != "function" or slot is None or depth > 6:
            return Keys(None, set())
        node = symbol.node
        params = [p.arg for p in node.args.posonlyargs + node.args.args]
        if isinstance(slot, int):
            if slot >= len(params):
                return Keys(None, set())
            slot = params[slot]
        inner = symbol.module.scopes.get(id(node))
        if inner is None:
            return Keys(None, set())
        possible, removes = set(), False
        for event in self.name_events(inner, slot, INF):
            if event.kind == "remove":
                removes = True
                continue
            added = self.event_keys(event, inner, depth + 1)
            if added.possible is None:
                return Keys(None, None if removes else set())
            possible |= added.possible
            removes |= added.certain is None
        return Keys(possible, None if removes else set())

    # ---- the check -------------------------------------------------------------------------------------
    def check_call(self, call: ast.Call, scope: Scope) -> list[str]:
        resolved = self.signature(self.resolve(call.func, scope))
        if resolved is None:
            return []
        callee, sig = resolved
        star = any(isinstance(a, ast.Starred) for a in call.args)
        n_pos = len(call.args)
        explicit = [kw.arg for kw in call.keywords if kw.arg is not None]
        splat = union([self.dict_keys(kw.value, scope, call.lineno) for kw in call.keywords if kw.arg is None])
        problems = []
        if not star and not sig.varargs and n_pos > len(sig.positional):
            problems.append(f"too many positional arguments ({n_pos} > {len(sig.positional)})")
        accepted = {p for p in sig.positional if p not in sig.posonly} | set(sig.kwonly)
        filled = set() if star else set(sig.positional[:n_pos])
        for name in explicit:
            if name not in accepted and not sig.varkw:
                problems.append(f"unexpected keyword '{name}'")
            elif name in filled:
                problems.append(f"multiple values for argument '{name}'")
        if not sig.varkw:
            for name in sorted(splat.certain - accepted - set(explicit)):
                problems.append(f"unexpected keyword '{name}' (from a ** mapping)")
        if splat.possible is not None:
            provided = filled | set(explicit) | splat.possible
            for name, required in sig.kwonly.items():
                if required and name not in provided:
                    problems.append(f"missing required keyword-only argument '{name}'")
            if not star:
                for name in sig.positional[n_pos:]:
                    if name not in sig.defaults and name not in provided:
                        problems.append(f"missing required argument '{name}'")
        return [f"{scope.module.rel}: {callee}: {p}" for p in problems]

    def findings(self) -> list[str]:
        return sorted(
            line
            for module in self.modules.values()
            for call, scope in module.calls
            for line in self.check_call(call, scope)
        )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[2]
    for line in Index(root).findings():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
