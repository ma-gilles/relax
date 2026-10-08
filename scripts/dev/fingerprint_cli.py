"""The command line shared by the fingerprint harnesses (``fingerprint.py``, ``vdam_fingerprint.py``).

Each harness builds one ``Harness`` record (its cases, mutations, worker, what it does not cover, and the few
names and environment rules where the two differ) and calls ``main(HARNESS, argv)``. The commands ``run``,
``diff``, ``check``, ``cases`` and ``selftest``, their options, outputs and exit codes are defined here once.
This module imports neither harness: the comparison functions travel in the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Harness:
    """What one fingerprint harness gives the shared command line; fixed at the harness's import."""

    script: Path  # the harness file; ``run_tree`` runs it again as ``<script> _worker ...`` in a child process
    description: str  # the argparse description (the first paragraph of the harness's docstring)
    cases: dict  # {name: (description, keywords)}, in run order
    mutations: tuple  # (name, old, new, description, expected detected) per selftest mutation
    not_covered: tuple[str, ...]
    worker: Callable[[str, str, str, list[str]], None]  # (source, out, tmp_root, case names)
    diff_fingerprints: Callable  # (a, b) -> (counts, lines)
    accepted: Callable  # counts -> bool
    case_errors: Callable  # fingerprint -> lines for cases whose error status is not the case's (refusal or run)
    differing_cases: Callable  # (a, b) -> case names
    mutated_tree: Callable  # (source, target, old, new) -> {file: places replaced}
    file_prefix: str  # prefix of the scratch names under --work-dir (``tmp_``, ``fp_``, ``mutant_``)
    temp_prefix: str  # mkdtemp prefix of the default --work-dir
    case_width: int  # column width of the case names in ``cases``
    cases_note: str  # text after "<N> cases" in ``cases``
    dropped_env_prefixes: tuple[str, ...]  # parent environment variables the child does not inherit
    jax_compilation_cache: bool  # True: a JAX compilation cache per child under --work-dir; False: none


def _print_not_covered(harness: Harness) -> None:
    print("NOT covered by these fingerprints:")
    for item in harness.not_covered:
        print(f"  - {item}")


def export_rev(rev: str, work_dir: Path) -> Path:
    """A ``git archive`` export of ``rev`` (``relax`` and ``tests``) under ``work_dir``."""
    sha = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", f"{rev}^{{commit}}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    target = work_dir / f"src_{sha[:12]}"
    if not target.exists():
        staging = Path(tempfile.mkdtemp(dir=work_dir, prefix="export_"))
        archive = subprocess.Popen(["git", "-C", str(REPO_ROOT), "archive", sha, "relax", "tests"], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(staging)], stdin=archive.stdout, check=True)
        if archive.wait() != 0:
            raise RuntimeError(f"git archive {sha} failed")
        staging.rename(target)
    return target


def run_tree(harness: Harness, source: Path, out: Path, work_dir: Path, names: list[str], *,
             threads: int = 4, quiet: bool = False) -> int:
    """Fingerprint ``source`` in a child process with a CPU-only environment that imports from ``source``."""
    label = hashlib.sha256(f"{source}|{out}".encode()).hexdigest()[:12]
    tmp_root = work_dir / f"{harness.file_prefix}tmp_{label}"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV") and not k.startswith(harness.dropped_env_prefixes)}
    env.update(
        PYTHONPATH=str(source), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
        RECOVAR_JAX_CACHE_DIR=str(work_dir / f"recovar_jax_cache_{label}"),
    )
    if harness.jax_compilation_cache:
        env["JAX_COMPILATION_CACHE_DIR"] = str(work_dir / f"jax_cache_{label}")
    else:
        env.pop("JAX_COMPILATION_CACHE_DIR", None)
    command = [sys.executable, str(harness.script), "_worker", str(source), str(out), str(tmp_root), *names]
    log_path = out.with_suffix(out.suffix + ".log")
    with open(log_path, "w") as log:
        code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if not quiet or code != 0:
        tail = log_path.read_text().splitlines()
        for line in tail if code != 0 else tail[-1:]:
            print(line)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return code


def _resolve_source(args, work_dir: Path) -> Path:
    if getattr(args, "rev", None):
        return export_rev(args.rev, work_dir)
    return Path(args.source).resolve() if getattr(args, "source", None) else REPO_ROOT


def _work_dir(harness: Harness, args) -> Path:
    if not args.work_dir:
        return Path(tempfile.mkdtemp(prefix=harness.temp_prefix))
    path = Path(args.work_dir).resolve()
    if REPO_ROOT in path.parents or path == REPO_ROOT:
        raise SystemExit(f"--work-dir {path} is inside the checkout; choose a directory outside it")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _case_names(harness: Harness, requested: list[str]) -> list[str]:
    unknown = [name for name in requested if name not in harness.cases]
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}; `{harness.script.name} cases` lists them")
    return requested or list(harness.cases)


def command_run(harness: Harness, args) -> int:
    work_dir = _work_dir(harness, args)
    code = run_tree(harness, _resolve_source(args, work_dir), Path(args.out).resolve(), work_dir,
                    _case_names(harness, args.cases))
    _print_not_covered(harness)
    return code


def command_diff(harness: Harness, args) -> int:
    a, b = (json.loads(Path(path).read_text()) for path in (args.a, args.b))
    counts, lines = harness.diff_fingerprints(a, b)
    print(f"A: {args.a} ({a['source']})\nB: {args.b} ({b['source']})")
    print("\n".join(lines))
    errors = [f"ERROR {side} {line}" for side, fp in (("A", a), ("B", b)) for line in harness.case_errors(fp)]
    if errors:
        print("\n".join(errors))
        print(f"FAILED: {len(errors)} case runs ended in an error that is not the case's; their status, not the run, "
              "was compared")
    _print_not_covered(harness)
    return 0 if harness.accepted(counts) and not errors else 1


def command_check(harness: Harness, args) -> int:
    work_dir = _work_dir(harness, args)
    names = _case_names(harness, args.cases)
    fp = f"{harness.file_prefix}fp_"
    base_source = export_rev(args.base, work_dir)
    base_out = work_dir / f"{fp}{base_source.name[4:]}.json"
    if names != list(harness.cases) or not base_out.exists():
        if run_tree(harness, base_source, base_out, work_dir, names) != 0:
            return 2
    if args.head:
        head_source = export_rev(args.head, work_dir)
        head_out = work_dir / f"{fp}{head_source.name[4:]}.json"
    else:
        head_source, head_out = REPO_ROOT, work_dir / f"{fp}worktree.json"
    if run_tree(harness, head_source, head_out, work_dir, names) != 0:
        return 2
    return command_diff(harness, argparse.Namespace(a=str(base_out), b=str(head_out)))


def command_cases(harness: Harness, args) -> int:
    for name, (description, _) in harness.cases.items():
        print(f"{name:{harness.case_width}s} {description}")
    print(f"{len(harness.cases)} cases{harness.cases_note}")
    _print_not_covered(harness)
    return 0


def command_selftest(harness: Harness, args) -> int:
    work_dir = _work_dir(harness, args)
    source = _resolve_source(args, work_dir)
    all_cases = list(harness.cases)
    known = {mutation[0]: mutation for mutation in harness.mutations}
    unknown = [name for name in args.mutations if name not in known]
    if unknown:
        raise SystemExit(f"unknown mutation(s): {', '.join(unknown)}")
    selected = [known[name] for name in args.mutations] if args.mutations else list(harness.mutations)
    clean_out = work_dir / f"{harness.file_prefix}fp_selftest_clean.json"
    if run_tree(harness, source, clean_out, work_dir, all_cases, quiet=True) != 0:
        print("the unmutated tree failed to run")
        return 2
    clean = json.loads(clean_out.read_text())

    def one(mutation):
        name, old, new, _, _ = mutation
        target = work_dir / f"{harness.file_prefix}mutant_{name}"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir()
        changed = harness.mutated_tree(source, target, old, new)
        if sum(changed.values()) != 1:
            return name, changed, []
        out = work_dir / f"{harness.file_prefix}fp_mutant_{name}.json"
        if run_tree(harness, target, out, work_dir, all_cases, threads=2, quiet=True) != 0:
            return name, changed, None
        cases = harness.differing_cases(clean, json.loads(out.read_text()))
        shutil.rmtree(target, ignore_errors=True)
        return name, changed, cases

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        outcomes = list(pool.map(one, selected))
    failures = 0
    for (name, _, _, description, expected), (_, changed, cases) in zip(selected, outcomes, strict=True):
        if sum(changed.values()) != 1:
            verdict, detail = "FAIL", (
                f"the mutation's target text is in {sum(changed.values())} places of this tree, not one; update MUTATIONS"
            )
        elif cases is None:
            verdict, detail = "FAIL", "the mutated tree did not run to a fingerprint"
        elif expected and not cases:
            verdict, detail = "FAIL", "not detected"
        elif not expected and not cases:
            verdict, detail = "BLIND", "not detected (known blind spot of the cases)"
        elif not expected:
            verdict, detail = "OK", f"detected in {len(cases)} cases (was a known blind spot; mark it detected)"
        else:
            verdict, detail = "OK", f"detected in {len(cases)} cases, e.g. {cases[0]}"
        failures += verdict == "FAIL"
        print(f"{verdict:5s} {name}: {description} [{', '.join(changed)}] -> {detail}")
    print(f"{len(selected)} mutations, {failures} failed")
    _print_not_covered(harness)
    return 1 if failures else 0


def _add_tree_options(sub) -> None:
    tree = sub.add_mutually_exclusive_group()
    tree.add_argument("--source", help="a source tree with relax/ and tests/ (default: this checkout as it is)")
    tree.add_argument("--rev", help="a commit to export with git archive and fingerprint")
    sub.add_argument("--work-dir", help="scratch directory outside the checkout (default: a new temp directory)")


def main(harness: Harness, argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_worker"]:
        harness.worker(argv[1], argv[2], argv[3], argv[4:])
        return 0
    parser = argparse.ArgumentParser(description=harness.description, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="fingerprint one source tree to a JSON file")
    run.add_argument("out")
    run.add_argument("cases", nargs="*")
    _add_tree_options(run)
    run.set_defaults(function=command_run)
    diff = commands.add_parser("diff", help="compare two fingerprint files; exit 1 unless only log rows differ")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.set_defaults(function=command_diff)
    check = commands.add_parser("check", help="fingerprint BASE and HEAD (default: the worktree) and diff them")
    check.add_argument("base")
    check.add_argument("head", nargs="?")
    check.add_argument("--cases", nargs="*", default=[])
    check.add_argument("--work-dir")
    check.set_defaults(function=command_check)
    cases = commands.add_parser("cases", help="list the cases")
    cases.set_defaults(function=command_cases)
    selftest = commands.add_parser("selftest", help="check that each deliberately perturbed controller is detected")
    selftest.add_argument("mutations", nargs="*")
    selftest.add_argument("--jobs", type=int, default=4)
    _add_tree_options(selftest)
    selftest.set_defaults(function=command_selftest)
    args = parser.parse_args(argv)
    return args.function(harness, args)
