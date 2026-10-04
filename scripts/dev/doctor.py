#!/usr/bin/env python
"""Check that this checkout, its environment and the shared directories are ready for work.

    pixi run doctor                    # or: <any python> scripts/dev/doctor.py [--json]

It only checks; it changes nothing (no install, no build, no fetch, no file written). One line per
check, ``OK``, ``WARN`` or ``FAIL``, and under each line that is not ``OK`` the exact fix. The exit
status is 1 only when a check is ``FAIL``: the Python it runs under is not the environment
``pixi.toml`` and ``pixi.lock`` describe, ``relax`` would import from another checkout, or a fixture
root is unreadable. A ``WARN`` is something a task may need (natives, the RELION binding, a writable
run root, an idle GPU) and names the command or override for it.

It uses the standard library only and runs under whatever Python starts it, so it also reports in a
fresh worktree that has no ``.pixi`` yet.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
OK, WARN, FAIL = "OK", "WARN", "FAIL"
SAME_FIX = "the environment fix under `recovar pin`"
BINDING_GLOB = "_relion_bind_core*.so"
# Physical GPU 0 of the shared development machine belongs to other users and is never offered.
RESERVED_GPU = 0
IDLE_MEMORY_MIB = 100
NATIVE_PACKS_SCANNED = 400


class Result(NamedTuple):
    status: str
    name: str
    detail: str
    fix: str = ""


# ----------------------------------------------------------------------------- pure logic


def recovar_pin(pixi_toml: str) -> str | None:
    """The RECOVAR commit the default environment pins (``[feature.pinned.pypi-dependencies]``)."""
    match = re.search(r'^recovar\s*=\s*\{[^}\n]*\brev\s*=\s*"([0-9a-f]{7,40})"', pixi_toml, re.MULTILINE)
    return match.group(1) if match else None


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_packages(lock: str, environment: str = "default") -> dict:
    """What ``pixi.lock`` holds for one environment.

    ``conda``: the ``name-version-build`` of every conda package; ``pypi``: ``{name: version}`` of
    every wheel or sdist; ``git``: ``{name: commit}`` of every git dependency. Local path
    dependencies (this checkout, installed editable) are not listed.
    """
    conda: set[str] = set()
    pypi: dict[str, str] = {}
    git: dict[str, str] = {}
    inside = False
    for line in lock.splitlines():
        if line.startswith("packages:"):
            break
        header = re.match(r"^  (\S+):\s*$", line)
        if header:
            inside = header.group(1) == environment
            continue
        entry = re.match(r"^\s+- (conda|pypi): (\S+)\s*$", line)
        if not inside or not entry:
            continue
        kind, url = entry.groups()
        base = unquote(url.split("#")[0].rstrip("/").rsplit("/", 1)[-1])
        if kind == "conda":
            conda.add(re.sub(r"\.(conda|tar\.bz2)$", "", base))
        elif url.startswith("git+"):
            name = re.sub(r"\.git$", "", base.split("?")[0].split("@")[0])
            git[normalized(name)] = url.rsplit("#", 1)[-1] if "#" in url else ""
        elif url.startswith(("http://", "https://")):
            if base.endswith(".whl"):
                name, version = base.split("-")[:2]
            else:
                name, version = re.sub(r"\.(tar\.gz|zip|tar\.bz2)$", "", base).rsplit("-", 1)
            pypi[normalized(name)] = version
    return {"conda": conda, "pypi": pypi, "git": git}


def lock_differences(locked: dict, conda_installed: set, pypi_installed: dict) -> list[str]:
    """Packages of the lock that the environment lacks or holds at another version."""
    differences = [f"conda {name} is not installed" for name in sorted(locked["conda"] - set(conda_installed))]
    for name, version in sorted(locked["pypi"].items()):
        installed = pypi_installed.get(name)
        if installed is None:
            differences.append(f"pypi {name} {version} is not installed")
        elif installed.lower() != version.lower():
            differences.append(f"pypi {name} is {installed}, the lock has {version}")
    return differences


def idle_gpus(gpu_csv: str, compute_apps_csv: str = "") -> tuple[list[tuple[int, str]], list[int]]:
    """``(idle, busy)`` from ``nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits``.

    ``idle`` lists ``(index, UUID)`` of the GPUs that hold no compute process and almost no memory;
    GPU 0 appears in neither list, whatever its state.
    """
    in_use = {line.split(",")[0].strip() for line in compute_apps_csv.splitlines() if line.strip()}
    idle, busy = [], []
    for line in gpu_csv.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        index, uuid = int(parts[0]), parts[1]
        if index == RESERVED_GPU:
            continue
        used = float(parts[2]) if re.fullmatch(r"[0-9.]+", parts[2]) else float("inf")
        if uuid in in_use or used > IDLE_MEMORY_MIB:
            busy.append(index)
        else:
            idle.append((index, uuid))
    return idle, busy


def static_root(default: str) -> str:
    """The fixed directory of a launcher default such as ``/scratch/x/codex/${RUN_ID}``."""
    return default.split("$", 1)[0].rstrip("/")


def launcher_roots(script: str) -> list[tuple[str, str]]:
    """``(variable, root)`` for each ``${VARIABLE:-/absolute/default}`` output root in a launcher."""
    found = []
    for variable, default in re.findall(r"\$\{([A-Z0-9_]+):-(/[^}\"']*)\}", script):
        if re.search(r"(SCRATCH_DIR|RUNTIME_ROOT|OUTPUT_ROOT|OUTPUT_BASE|RESULTS_ROOT)$", variable):
            found.append((variable, static_root(default)))
    return found


def exit_status(results: list[Result]) -> int:
    return 1 if any(result.status == FAIL for result in results) else 0


def render(results: list[Result]) -> str:
    lines = []
    for result in results:
        lines.append(f"{result.status:<4} {result.name}: {result.detail}")
        if result.fix and result.status != OK:
            lines.append(f"     fix: {result.fix}")
    counts = {status: sum(result.status == status for result in results) for status in (OK, WARN, FAIL)}
    lines.append(f"{counts[OK]} OK, {counts[WARN]} WARN, {counts[FAIL]} FAIL")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- checks


def _run(*argv: str, cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _under(path: str | os.PathLike | None, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _origin(module: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    return getattr(spec, "origin", None) if spec else None


def _shared_pixi(prefix: Path) -> Path | None:
    """The ``.pixi`` directory a prefix ``<checkout>/.pixi/envs/default`` belongs to."""
    resolved = prefix.resolve()
    return resolved.parents[1] if resolved.parent.name == "envs" and resolved.parents[1].name == ".pixi" else None


def _environment_fix(root: Path) -> str:
    return (
        f"run it with the checkout's environment: `cd {root} && pixi install --frozen && pixi run doctor` "
        "(about 12 GB), or link an environment built from the same pixi.lock "
        f"(`ln -s <its checkout>/.pixi {root}/.pixi`) and rerun `{root}/.pixi/envs/default/bin/python "
        "scripts/dev/doctor.py`"
    )


def check_environment(root: Path) -> list[Result]:
    prefix = Path(sys.prefix)
    version = ".".join(map(str, sys.version_info[:3]))
    results = []
    pin = recovar_pin((root / "pixi.toml").read_text())
    try:
        direct_url = json.loads(importlib.metadata.distribution("recovar").read_text("direct_url.json") or "{}")
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    if direct_url is None:
        results.append(
            Result(
                FAIL, "recovar pin", f"recovar is not installed in {prefix} (Python {version})", _environment_fix(root)
            )
        )
    else:
        commit = direct_url.get("vcs_info", {}).get("commit_id")
        if pin and commit == pin:
            results.append(Result(OK, "recovar pin", f"{commit[:9]} installed in {prefix}, as pixi.toml pins"))
        elif commit is None:
            results.append(
                Result(
                    WARN,
                    "recovar pin",
                    f"recovar comes from {direct_url.get('url')} (no commit: the dev environment), pixi.toml pins "
                    f"{str(pin)[:9]}",
                    "use the default environment for anything compared with main: `pixi run doctor`",
                )
            )
        else:
            results.append(
                Result(
                    FAIL,
                    "recovar pin",
                    f"{prefix} holds recovar {commit[:9]}, pixi.toml pins {str(pin)[:9]}",
                    _environment_fix(root),
                )
            )

    conda_meta = prefix / "conda-meta"
    if not conda_meta.is_dir():
        results.append(
            Result(
                FAIL,
                "pixi.lock",
                f"{sys.executable} is not a pixi environment (no {conda_meta}), so it was not built from pixi.lock",
                SAME_FIX,
            )
        )
    else:
        locked = locked_packages((root / "pixi.lock").read_text())
        conda_installed = {path.name[: -len(".json")] for path in conda_meta.glob("*.json")}
        pypi_installed = {}
        for distribution in importlib.metadata.distributions(path=[str(p) for p in sys.path if _under(p, prefix)]):
            name = distribution.metadata["Name"]
            if name:
                pypi_installed[normalized(name)] = distribution.version
        differences = lock_differences(locked, conda_installed, pypi_installed)
        total = len(locked["conda"]) + len(locked["pypi"])
        if differences:
            shown = "; ".join(differences[:4]) + (f"; and {len(differences) - 4} more" if len(differences) > 4 else "")
            results.append(
                Result(
                    FAIL,
                    "pixi.lock",
                    f"{len(differences)} of {total} locked packages differ in {prefix}: {shown}",
                    _environment_fix(root),
                )
            )
        else:
            results.append(Result(OK, "pixi.lock", f"all {total} locked packages are installed at the locked builds"))

    jax = _origin("jax")
    if jax is None:
        results.append(Result(FAIL, "jax", f"jax is not importable from {sys.executable}", SAME_FIX))
    elif not _under(jax, prefix):
        results.append(
            Result(
                FAIL,
                "jax",
                f"jax resolves to {jax}, outside this environment ({prefix})",
                "unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV; export PYTHONNOUSERSITE=1; then "
                f"export PYTHONPATH={root}",
            )
        )
    else:
        results.append(Result(OK, "jax", f"{importlib.metadata.version('jax')} from {prefix} (Python {version})"))

    relax = _origin("relax")
    if relax is None:
        results.append(Result(FAIL, "relax import", "relax is not importable", f"export PYTHONPATH={root}"))
    elif not _under(relax, root):
        results.append(
            Result(
                FAIL,
                "relax import",
                f"relax resolves to {relax}, not to this checkout (a shared environment's editable install "
                "points at the checkout it was built in)",
                f"export PYTHONPATH={root}",
            )
        )
    else:
        results.append(Result(OK, "relax import", f"from this checkout ({relax})"))
    return results


def check_pixi_directory(root: Path, environment_ok: bool) -> Result:
    pixi = root / ".pixi"
    default = pixi / "envs" / "default"
    shared = _shared_pixi(Path(sys.prefix))
    if not os.path.lexists(pixi):
        fix = f"cd {root} && pixi install --frozen (about 12 GB)"
        if environment_ok and shared is not None:
            fix = f"ln -s {shared} {pixi}  (the environment this ran under, which matches the pin and the lock)"
        return Result(
            WARN,
            ".pixi",
            "absent: scripts/run_em_fast_guard.sh and the tier runner use <checkout>/.pixi/envs/default",
            fix,
        )
    kind = f"symlink to {os.path.realpath(pixi)}" if pixi.is_symlink() else "directory"
    if not default.is_dir():
        return Result(FAIL, ".pixi", f"{kind}, but {default} does not exist", f"cd {root} && pixi install --frozen")
    if default.resolve() != Path(sys.prefix).resolve():
        return Result(
            WARN,
            ".pixi",
            f"{kind}; this run used {sys.prefix} instead, so the checks above describe that environment",
            f"{default}/bin/python scripts/dev/doctor.py",
        )
    return Result(OK, ".pixi", kind)


def _run_base(root: Path) -> Path | None:
    match = re.search(r'^RUN_BASE = Path\("([^"]+)"\)', (root / "scripts" / "run_test_tier.py").read_text(), re.M)
    return Path(match.group(1)) if match else None


def _native_packs(root: Path) -> list[Path]:
    """Directories that may hold built natives: the ones the environment names, then recent tier runs."""
    packs = []
    for variable in ("RELAX_CUDA_LIB", "RECOVAR_CUDA_LIB", "RECOVAR_RELION_BIND_BUILD_DIR"):
        if os.environ.get(variable):
            packs.append(Path(os.environ[variable]).parent)
    base = _run_base(root)
    try:
        records = list(base.glob("*/natives/NATIVE.json")) if base else []
    except OSError:
        records = []

    def modified(record: Path) -> float:
        try:
            return record.stat().st_mtime
        except OSError:
            return 0.0

    packs += [record.parent for record in sorted(records, key=modified, reverse=True)[:NATIVE_PACKS_SCANNED]]
    return packs


def check_natives(root: Path) -> tuple[Result, Path | None]:
    sys.path.insert(0, str(root / "scripts"))
    try:
        import native_sources
    finally:
        sys.path.pop(0)
    sources = native_sources.native_sources(root)
    digest = sources["sha256"]
    build = "bash scripts/build_test_natives.sh <new directory>  (the tier commands do this for their own run)"
    if sources["recovar_commit"] in ("not installed", "unknown"):
        return (
            Result(
                WARN,
                "natives",
                "the native-source digest includes the installed recovar commit, and this Python has none",
                SAME_FIX,
            ),
            None,
        )
    reasons = []
    for pack in _native_packs(root):
        try:
            record = json.loads((pack / "NATIVE.json").read_text()).get("native_sources") or {}
        except (OSError, ValueError):
            continue
        if record.get("sha256") != digest:
            continue
        try:
            reason = native_sources.check(pack, root)
        except Exception as error:  # the full check starts this Python and imports recovar
            reason = f"scripts/native_sources.py check could not run: {type(error).__name__}: {error}"
        if reason is None:
            return Result(OK, "natives", f"source digest {digest[:12]}; matching pack {pack}"), pack
        reasons.append(reason)
    if reasons:
        return Result(
            WARN, "natives", f"source digest {digest[:12]}; a pack has it but fails the check: {reasons[0]}", build
        ), None
    return (
        Result(WARN, "natives", f"source digest {digest[:12]}; no readable pack built from it was found", build),
        None,
    )


def check_relion_binding(root: Path, pack: Path | None) -> Result:
    directories = [root / "relax" / "relion_bind" / "build", root / "relax" / "relion_bind"]
    if os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR"):
        directories.insert(0, Path(os.environ["RECOVAR_RELION_BIND_BUILD_DIR"]).expanduser())
    for directory in directories:
        try:
            built = sorted(directory.glob(BINDING_GLOB))
        except OSError:
            built = []
        if built and os.access(built[0], os.R_OK):
            return Result(OK, "RELION binding", str(built[0]))
    fix = "pixi run build-relion-bind  (needs RELION_SRC_DIR, the RELION 5.0.1 src)"
    if pack is not None and sorted((pack / "relion_bind").glob(BINDING_GLOB)):
        fix = (
            f"export RECOVAR_RELION_BIND_BUILD_DIR={pack / 'relion_bind'}  (built from this checkout's native sources)"
        )
    return Result(
        WARN,
        "RELION binding",
        "not built: tests marked requires_relion_bind skip, and pytest --require-natives stops",
        fix,
    )


def check_fixtures(root: Path) -> Result:
    manifest = json.loads((root / "tests" / "fixtures" / "em_fixture_manifest.json").read_text())
    sets = manifest.get("sets", manifest)
    roots = sorted({entry["root"] for entry in sets.values() if isinstance(entry, dict) and "root" in entry})
    unreadable = [path for path in roots if not (os.path.isdir(path) and os.access(path, os.R_OK | os.X_OK))]
    if unreadable:
        shown = ", ".join(unreadable[:3]) + (f" and {len(unreadable) - 3} more" if len(unreadable) > 3 else "")
        return Result(
            FAIL,
            "fixtures",
            f"{len(unreadable)} of {len(roots)} fixture roots are missing or unreadable for {getpass.getuser()}: {shown}",
            f"their owner grants access (`setfacl -R -m u:{getpass.getuser()}:rX <root>`); a missing root is "
            "regenerated only on the user's request (tests/fixtures/em_fixture_manifest.json records how)",
        )
    return Result(
        OK,
        "fixtures",
        f"all {len(roots)} fixture roots of the manifest are readable (contents: python tests/helpers/em_fixtures.py)",
    )


def _writable(path: Path) -> bool:
    """Whether the account can create entries in ``path``, or in its nearest existing parent."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def check_write_roots(root: Path) -> list[Result]:
    user = getpass.getuser()
    grant = f"its owner runs `setfacl -m u:{user}:rwx,d:u:{user}:rwx <directory>`"
    results = []
    base = _run_base(root)
    if base is None:
        results.append(Result(WARN, "tier run root", "RUN_BASE not found in scripts/run_test_tier.py"))
    elif _writable(base):
        results.append(Result(OK, "tier run root", f"{base} is writable"))
    else:
        results.append(
            Result(
                WARN,
                "tier run root",
                f"{base} is not writable for {user}",
                f"pass a run root you can write: `pixi run test-smoke --run-root <new directory>`; or {grant}",
            )
        )
    by_root: dict[str, list[str]] = {}
    for script in sorted((root / "scripts").glob("*.sh")):
        for variable, directory in launcher_roots(script.read_text()):
            by_root.setdefault(directory, []).append(variable)
    guard = re.search(
        r'^DEFAULT_OUTPUT_ROOT = Path\("([^"]+)"\)',
        (root / "scripts" / "run_vdam_abinitio_merge_guard.py").read_text(),
        re.M,
    )
    if guard:
        by_root.setdefault(guard.group(1), []).append("VDAM_ABINITIO_GUARD_OUTPUT_ROOT")
    blocked = {directory: names for directory, names in by_root.items() if not _writable(Path(directory))}
    if blocked:
        detail = "; ".join(
            f"{directory} ({', '.join(sorted(set(names)))})" for directory, names in sorted(blocked.items())
        )
        results.append(
            Result(
                WARN,
                "launcher roots",
                f"{len(blocked)} of {len(by_root)} default output roots are not writable for {user}: {detail}",
                f"export the variable named in parentheses to a directory you can write before using that launcher; or {grant}",
            )
        )
    else:
        results.append(Result(OK, "launcher roots", f"all {len(by_root)} default output roots are writable"))
    return results


def check_git(root: Path) -> Result:
    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    if branch is None:
        return Result(WARN, "git", f"{root} is not a git checkout (an export or snapshot)")
    head = (_run("git", "rev-parse", "--short", "HEAD", cwd=root) or "").strip()
    dirty = len((_run("git", "status", "--porcelain", cwd=root) or "").splitlines())
    state = f"branch {branch.strip()} at {head}, {dirty} uncommitted path(s)"
    counts = _run("git", "rev-list", "--left-right", "--count", "origin/main...HEAD", cwd=root)
    if counts is None:
        return Result(WARN, "git", f"{state}; origin/main is unknown here", "git fetch origin main")
    behind, ahead = (int(value) for value in counts.split())
    state += f"; {ahead} ahead of and {behind} behind origin/main as last fetched"
    if behind:
        return Result(WARN, "git", state, "git fetch origin && git rebase origin/main  (then rerun the CPU checks)")
    return Result(OK, "git", state)


def check_gpus() -> Result:
    if os.environ.get("SLURM_JOB_ID"):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return Result(
            OK, "GPUs", f"inside Slurm job {os.environ['SLURM_JOB_ID']}: use the assigned devices ({visible or 'none'})"
        )
    gpus = _run("nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits")
    if gpus is None:
        return Result(OK, "GPUs", "no local GPU here (nvidia-smi is unavailable): GPU work goes through Slurm")
    apps = _run("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader") or ""
    idle, busy = idle_gpus(gpus, apps)
    if not idle:
        return Result(
            WARN,
            "GPUs",
            f"no idle local GPU among {sorted(busy) or 'none'}; GPU {RESERVED_GPU} is reserved for other users",
            "wait, or use Slurm (--partition=cryoem); never GPU 0",
        )
    listed = ", ".join(f"GPU {index} ({uuid})" for index, uuid in idle)
    return Result(
        OK,
        "GPUs",
        f"idle now: {listed}; GPU {RESERVED_GPU} is reserved. Recheck nvidia-smi, then "
        f"export CUDA_VISIBLE_DEVICES={idle[0][1]}",
    )


def run_checks(root: Path = REPO_ROOT) -> list[Result]:
    results = check_environment(root)
    environment_ok = all(result.status == OK for result in results if result.name != "relax import")
    results.append(check_pixi_directory(root, environment_ok))
    natives, pack = check_natives(root)
    results += [natives, check_relion_binding(root, pack), check_fixtures(root), *check_write_roots(root)]
    results += [check_git(root), check_gpus()]
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(argv)
    results = run_checks()
    if args.json:
        print(json.dumps([result._asdict() for result in results], indent=1))
    else:
        print(f"relax doctor: {REPO_ROOT} as {getpass.getuser()} with {sys.executable}")
        print(render(results))
    return exit_status(results)


if __name__ == "__main__":
    raise SystemExit(main())
