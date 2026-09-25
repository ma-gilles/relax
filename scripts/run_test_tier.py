#!/usr/bin/env python
"""Run a relax test tier: smoke, medium or long (the tier policy is in CONTRIBUTING.md).

    pixi run test-smoke     # python scripts/run_test_tier.py submit smoke
    pixi run test-medium    # python scripts/run_test_tier.py submit medium
    pixi run test-long      # python scripts/run_test_tier.py submit long

``submit`` freezes the checkout (HEAD plus any uncommitted diff) into
``<run-root>/src``, verifies the tier's fixture sets against the manifest, builds the
native libraries once (``scripts/build_test_natives.sh``) and starts the tier:

* smoke runs as a one-GPU cryoem job when it would start within ``--queue-wait-minutes``,
  otherwise on one idle local GPU 1-3 (never GPU 0), selected by UUID after an nvidia-smi
  check; medium and long never run locally;
* medium runs as ONE Slurm job on three GPUs (fast parity cases, GPU unit shards, VDAM and
  the end-to-end run share it);
* long runs as ONE Slurm job on four GPUs: the four EM long-tier arms (K1 50k standalone and
  seeded, native VDAM, K4 50k) and the K1/K4 100k/256 completion arms, then the band scoring.

Related GPU work is packed into one job (della-cryoem allows 16 running jobs but 32 GPUs per
user). Every GPU job goes to the cryoem partition; ``--queue general`` (the general GPU
partition, A100 80GB) is only for when the user explicitly allows another partition.

Inside the allocation ``run`` executes the tier's items: CPU items alongside the GPU
items, GPU items on one worker process per visible GPU (longest first, after any items
they depend on). Each item keeps its log,
junit XML and pytest basetemp under ``<run-root>/items/<name>``. A tier passes when
every item exits 0 and no required item skipped a test. FSC summaries of the parity
cases are written to ``<run-root>/fsc.json``; they are reported, not gated, until the
thresholds in ``tests/tiers/fsc_thresholds.json`` are approved. Every run ends with a
receipt (``scripts/write_test_receipt.py``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.native_sources import check as check_native_sources  # noqa: E402
TIERS_DIR = REPO_ROOT / "tests" / "tiers"
RUN_BASE = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_test_tiers")
BUDGET_S = {"smoke": 5 * 60, "medium": 2 * 3600, "long": 8 * 3600}
LOCAL_GPUS = ("1", "2", "3")  # physical GPU 0 of the development machine stays free
FAST = "tests/integration/test_em_parity_fast.py"
LONG = "tests/long_test/test_em_parity_long.py"
E2E = "tests/integration/test_em_tier_e2e.py"
PARITY_FLAGS = ["--run-slow", "--run-integration", "--run-gpu"]
FAST_CASES = {  # item name: pytest node id
    "k1_replay": "test_em_parity_fast_k1_replay",
    "k1_local_replay": "test_em_parity_fast_k1_local_replay",
    "k1_adaptive_replay": "test_em_parity_fast_k1_adaptive_replay",
    "kclass_replay": "test_em_parity_fast_kclass_replay",
    "k1_coldstart_standalone": "test_em_parity_fast_k1_coldstart[standalone]",
    "k1_coldstart_relion_seeded_debug": "test_em_parity_fast_k1_coldstart[relion_seeded_debug]",
    "k1_perturbreplay": "test_em_parity_fast_k1_perturbreplay",
    "k1_os1_coldstart_standalone": "test_em_parity_fast_k1_os1_coldstart_standalone",
    "k1_gui60_coldstart_standalone": "test_em_parity_fast_k1_gui60_coldstart_standalone",
    "k1_multioptics_coldstart": "test_em_parity_fast_k1_multioptics_coldstart",
    "kclass_coldstart": "test_em_parity_fast_kclass_coldstart",
    "kclass_nonadaptive_replay": "test_em_parity_fast_kclass_nonadaptive_replay",
    "kclass_strict_oversample_coldstart": "test_em_parity_fast_kclass_strict_oversample_coldstart",
}
# The 5-minute smoke budget holds on an A100 (local fallback): 254 s of the 363 s for all four
# fixed-state replays. k1_local_replay stays because smoke must cover the local-search regime
# (a past merge broke it while the global checks stayed green); the global K1 path is still
# exercised by k1_adaptive_replay (global hp3, os1). k1_replay runs in medium.
SMOKE_REPLAYS = ("k1_local_replay", "k1_adaptive_replay", "kclass_replay")
FAST_CASE_SECONDS = {  # H100 walls of the last full fast tier (Q 14320189) and relax coverage runs
    "k1_replay": 72,
    "k1_local_replay": 100,
    "k1_adaptive_replay": 80,
    "kclass_replay": 42,
    "k1_coldstart_standalone": 600,
    "k1_coldstart_relion_seeded_debug": 600,
    "k1_perturbreplay": 210,
    "k1_os1_coldstart_standalone": 600,
    "k1_gui60_coldstart_standalone": 850,  # H100 relax wall 848 s (Slurm 14402553)
    "k1_multioptics_coldstart": 110,  # H100 relax wall 106 s cold (Slurm 14419909)
    "kclass_coldstart": 180,
    "kclass_nonadaptive_replay": 90,
    "kclass_strict_oversample_coldstart": 110,
}
SWEEP_DIRS = ("tests/unit", "tests/integration", "tests/ppca_abinitio")
CPU_MERGE_UNITS = (
    "tests/unit/initial_model/test_dense_adapter.py",
    "tests/unit/initial_model/test_iteration_loop.py",
    "tests/unit/initial_model/test_init_and_estep.py",
    "tests/unit/initial_model/test_native_driver.py",
    "tests/unit/test_k_class_joint_semantics.py",
    "tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_relion_sigma_offset_prior_center_matches_store_weighted_sums_units",
    "tests/unit/test_refine_relion_mode.py::test_relion_mode_writes_absolute_translations_from_previous_offset",
    "tests/unit/test_refine_relion_mode.py::test_relion_mode_dense_k_class_writes_absolute_translations_from_previous_offset",
)
SWEEP_EXCLUDE = {FAST, E2E}
# Files that change process-wide JAX state at import and must not share a pytest process.
ISOLATE = {"tests/unit/test_em_deterministic_reductions.py"}
SHARD_TARGET_S = 600
# One job per tier: GPUs, wall limit and memory. 8 CPUs per GPU; memory is 128G per GPU
# except long, whose concurrent arms peak at 180G (K1 100k, 135 GB measured), 160G (K4 100k,
# 98 GB) and 128G (the 50k arms, <= 63 GB), i.e. 600G for its four GPUs.
TIER_JOB = {
    "smoke": {"gpus": 1, "time": "01:00:00", "mem_gb": 128},
    "medium": {"gpus": 3, "time": "03:00:00", "mem_gb": 384},
    "long": {"gpus": 4, "time": "16:00:00", "mem_gb": 600},
}
LONG_ARM_SECONDS = {  # H100 walls: long tier (relax 14309851/14287533, Q 14320191), completions (Q 14320204/18)
    "long_k1_standalone": 4700,
    "long_k1_relion_seeded_debug": 4700,
    "long_k1_native_vdam": 250,
    "long_kclass": 6700,
    "completion_k1": 9000,
    "completion_k4": 19300,
    # EMPIAR-10097 it13 -> 14 (hp3, current size 136): RELION 556 s for the iteration on H100.
    "long_realdata_hp3_default": 1200,
    "long_realdata_hp3_resident": 1200,
    # Class3D K4 5k/128 at HEALPix 4, 2 iterations (relax 7381f84, Slurm 14410096): 1860 s on H100.
    "long_class3d_hp4": 2000,
}
FIXTURE_SETS = {
    "smoke": ["k1_5k128_data", "k1_5k128_relion_os0", "k1_5k128_relion_os1", "k2_5k128_data", "k2_5k128_relion_os0"],
    "medium": [
        "k1_5k128_data",
        "k1_5k128_relion_os0",
        "k1_5k128_relion_os1",
        "k1_5k128_relion_os1_repeats",
        "k1_5k128_relion_gui60",
        "k2_5k128_data",
        "k2_5k128_relion_os0",
        "k4_5k128_data",
        "k4_5k128_oracle_h2_os1",
        "k4_5k128_oracle_h1_os1",
        "k1_50k256_data",
        "k1_50k256_relion_initialmodel_it008",
    ],
    "long": [
        "k1_50k256_data",
        "k1_50k256_relion_os0",
        "k1_50k256_relion_initialmodel_it008",
        "k1_50k256_relion_repeats",
        "k4_50k256_data",
        "k4_50k256_relion_os0",
        "k4_50k256_relion_repeats",
        "k1_100k256_data",
        "k1_100k256_relion",
        "k1_100k256_relion_repeats",
        "k1_100k256_mask",
        "k4_100k256_data",
        "k4_100k256_dispatch_oracle",
        "k4_100k256_mask",
        "empiar_10097_hp3_state",
        "empiar_10097_particle_stack",
        "k4_5k128_data",
        "k4_5k128_class3d_hp4_relion",
    ],
}


@dataclass
class Item:
    name: str
    argv: list[str]
    gpu: bool
    seconds: float
    required: bool = True  # a skipped test fails a required item
    env: dict[str, str] = field(default_factory=dict)
    after: list[str] = field(default_factory=list)  # items that must pass first


# ----------------------------------------------------------------------------- plans


def _pytest(py: str, *args: str, flags=PARITY_FLAGS) -> list[str]:
    return [py, "-m", "pytest", "-p", "no:cacheprovider", "-v", "-s", *flags, *args]


def changed_paths(src: Path, base: str) -> list[str]:
    """Files changed against ``base``: committed on this branch and uncommitted."""
    merge_base = subprocess.check_output(["git", "-C", str(src), "merge-base", base, "HEAD"], text=True).strip()
    out = subprocess.check_output(["git", "-C", str(src), "diff", "--name-only", merge_base], text=True)
    return sorted(set(out.split()))


# A GPU test is marked ``gpu``, or gated some other way: a skipif on GPU availability (for
# example the resident tests' ``requires_resident_gpu``), a runtime backend check, or the
# ``gpu_device`` / ``custom_cuda_lib`` fixtures. Those skip silently on CPU, so smoke must
# select them by these signs as well.
GPU_TEST_SIGNS = re.compile(
    r"pytest\.mark\.gpu|mark\.gpu\b|_gpu_available\(|@requires_\w*gpu\b"
    r"|default_backend\(\)\s*[!=]=\s*[\"']gpu[\"']|\bgpu_device\b|\bcustom_cuda_lib\b"
)


def gpu_test_files(src: Path) -> list[str]:
    """Test files of the GPU sweep selection that run on a GPU (GPU_TEST_SIGNS)."""
    files = []
    for d in SWEEP_DIRS:
        for path in sorted((src / d).rglob("test_*.py")):
            rel = path.relative_to(src).as_posix()
            if rel not in SWEEP_EXCLUDE and GPU_TEST_SIGNS.search(path.read_text(errors="ignore")):
                files.append(rel)
    return files


def touched_gpu_tests(src: Path, changed: list[str]) -> list[str]:
    """GPU test files for the changed paths: changed GPU tests, and GPU tests importing a changed module.

    A change under relax/cuda/ or to a native build input selects every GPU test importing
    relax.cuda; ``tests/tiers/gpu_path_map.json`` adds explicit glob -> test mappings.
    """
    import fnmatch

    gpu_files = gpu_test_files(src)
    texts = {f: (src / f).read_text(errors="ignore") for f in gpu_files}
    selected = {f for f in changed if f in texts}
    modules = []
    for path in changed:
        if path.startswith("relax/") and path.endswith(".py"):
            mod = path[:-3].replace("/", ".")
            modules.append(mod[: -len(".__init__")] if mod.endswith(".__init__") else mod)
        elif path.startswith("relax/cuda/") or path.startswith("relax/relion_bind/"):
            modules.append("relax.cuda" if path.startswith("relax/cuda/") else "relax.relion_bind")
        elif path.startswith("scripts/") and path.endswith(".py"):
            modules.append(path[:-3].replace("/", "."))
    for f, text in texts.items():
        for mod in modules:
            parent, _, leaf = mod.rpartition(".")
            if re.search(rf"\b{re.escape(mod)}\b", text) or re.search(
                rf"from\s+{re.escape(parent)}\s+import\s+[^\n]*\b{re.escape(leaf)}\b", text
            ):
                selected.add(f)
    mapping = json.loads((TIERS_DIR / "gpu_path_map.json").read_text())["map"]
    for pattern, tests in mapping.items():
        if any(fnmatch.fnmatch(p, pattern) for p in changed):
            selected.update(tests)
    return sorted(selected)


NEW_CASE_SECONDS = 300  # estimate for a fast case without a recorded wall


def fast_cases(src: Path, py: str) -> dict[str, str]:
    """Every test collected from the fast parity file, named as in FAST_CASES when known.

    Collecting (instead of listing) means a case added to the file joins the medium tier
    without editing this runner; FAST_CASES only names cases and records their walls.
    """
    out = subprocess.run(
        [py, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q", *PARITY_FLAGS, FAST],
        cwd=src, capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTHONPATH=str(src)),
    )
    nodes = [line.split("::", 1)[1] for line in out.stdout.splitlines() if line.startswith(f"{FAST}::")]
    if out.returncode or not nodes:
        raise SystemExit(f"cannot collect {FAST}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    names = {node: name for name, node in FAST_CASES.items()}
    cases = {}
    for node in nodes:
        cases[names.get(node) or re.sub(r"[^A-Za-z0-9]+", "_", node.removeprefix("test_em_parity_fast_")).strip("_")] = node
    return cases


def _durations() -> dict[str, int]:
    return json.loads((TIERS_DIR / "gpu_file_seconds.json").read_text())["seconds"]


def smoke_touched_split(touched: list[str], replay_seconds: float) -> tuple[list[str], list[str]]:
    """Touched GPU files that fit smoke's budget beside the replays, and the rest (for medium).

    Files are taken shortest first by their recorded wall (tests/tiers/gpu_file_seconds.json,
    10 s when unrecorded) while the smoke GPU total stays within BUDGET_S["smoke"]; the medium
    tier's unit sweep runs every GPU file, so a deferred file is still tested there.
    """
    seconds = _durations()
    room = BUDGET_S["smoke"] - replay_seconds
    kept, deferred = [], []
    for f in sorted(touched, key=lambda f: (seconds.get(f, 10), f)):
        cost = seconds.get(f, 10)
        if cost <= room:
            kept.append(f)
            room -= cost
        else:
            deferred.append(f)
    return sorted(kept), sorted(deferred)


def sweep_shards(src: Path, py: str) -> list[Item]:
    """The GPU unit sweep, packed into pytest processes of about SHARD_TARGET_S each."""
    seconds = _durations()
    files = []
    for d in SWEEP_DIRS:
        files += [p.relative_to(src).as_posix() for p in sorted((src / d).rglob("test_*.py"))]
    files = [f for f in files if f not in SWEEP_EXCLUDE]
    files.sort(key=lambda f: -seconds.get(f, 10))
    shards: list[list[str]] = []
    loads: list[float] = []
    for f in files:
        cost = seconds.get(f, 10)
        if cost >= SHARD_TARGET_S or f in ISOLATE:
            shards.append([f])
            loads.append(cost)
            continue
        open_shards = [i for i, s in enumerate(shards) if loads[i] + cost <= SHARD_TARGET_S and s[0] not in ISOLATE]
        if open_shards:
            i = min(open_shards, key=lambda i: loads[i])
            shards[i].append(f)
            loads[i] += cost
        else:
            shards.append([f])
            loads.append(cost)
    flags = ["--run-slow", "--run-integration", "--run-gpu"]
    env = sweep_opt_in_env(src)
    return [
        Item(f"unit_{i:02d}", _pytest(py, *s, flags=flags), True, loads[i] + 30, required=False, env=dict(env))
        for i, s in enumerate(shards)
    ]


def sweep_opt_in_env(src: Path) -> dict[str, str]:
    """Environment that turns on the GPU tests which otherwise skip on an opt-in variable.

    Every test that needs CUDA must run in the GPU sweep, so the opt-ins are set here:
    the CUDA x-half compact-pair guard, and the P4-J resident operand check, which reads a
    RELION particles STAR (the K1 5k fixture, resolved through the manifest).
    """
    manifest = json.loads((src / "tests" / "fixtures" / "em_fixture_manifest.json").read_text())
    sets = manifest.get("sets", manifest)
    star = Path(sets["k1_5k128_data"]["root"]) / "particles.star"
    if not star.is_file():
        raise SystemExit(f"GPU sweep fixture missing: {star}")
    return {"RELAX_RUN_CUDA_XHALF_TEST": "1", "RELAX_P4J_STAR_FIXTURE": str(star)}


def plan(tier: str, src: Path, base: str, run_root: Path | None = None) -> list[Item]:
    py = str(src / ".pixi" / "envs" / "default" / "bin" / "python")
    guard = Item("cpu_fast_guard", ["bash", "scripts/run_em_fast_guard.sh"], False, 90)
    # The InitialModel and K-class unit contracts the retired EM merge guard ran after merges.
    merge_units = Item(
        "cpu_merge_units",
        _pytest(py, *CPU_MERGE_UNITS, flags=[]),
        False,
        120,
    )
    if tier == "smoke":
        replays = Item(
            "replays",
            _pytest(py, *[f"{FAST}::{FAST_CASES[c]}" for c in SMOKE_REPLAYS]),
            True,
            sum(FAST_CASE_SECONDS[c] for c in SMOKE_REPLAYS),
        )
        items = [guard, merge_units, replays]
        touched, deferred = smoke_touched_split(touched_gpu_tests(src, changed_paths(src, base)), replays.seconds)
        if deferred:
            print(f"smoke budget: {len(deferred)} touched GPU file(s) deferred to the medium tier: {', '.join(deferred)}")
        if touched:
            seconds = _durations()
            items.append(
                Item(
                    "touched_gpu_units",
                    _pytest(py, *touched),
                    True,
                    sum(seconds.get(f, 10) for f in touched),
                    required=False,
                    env=sweep_opt_in_env(src),
                )
            )
        return items
    if tier == "medium":
        items = [guard, merge_units]
        items += [
            Item(name, _pytest(py, f"{FAST}::{node}"), True, FAST_CASE_SECONDS.get(name, NEW_CASE_SECONDS))
            for name, node in fast_cases(src, py).items()
        ]
        items.append(
            Item(
                "vdam_k1_50k",
                _pytest(py, "--em-parity-long", f"{LONG}::test_em_parity_long_k1_native_initialmodel_quality"),
                True,
                400,
            )
        )
        items.append(
            Item("e2e_k1_5k_standalone", _pytest(py, f"{E2E}::test_k1_5k128_standalone_autorefine"), True, 1500)
        )
        items += sweep_shards(src, py)
        return items
    if tier == "long":
        return long_plan(src, py, run_root)
    raise ValueError(f"unknown tier {tier!r}")


def long_plan(src: Path, py: str, run_root: Path | None) -> list[Item]:
    """The long tier's arms in one allocation; the completion arms run the job scripts the
    completion launcher wrote with --dry-run under <run-root>/completion/jobs (its setup script
    already ran at submission, so every arm is ready when the job starts)."""
    root = run_root or Path("<run-root>")
    jobs = root / "completion" / "jobs"
    long_flags = ["--em-parity-long"]
    items = [
        Item("long_k1_standalone", _pytest(py, *long_flags, f"{LONG}::test_em_parity_long_k1_full[standalone]"), True,
             LONG_ARM_SECONDS["long_k1_standalone"]),
        Item("long_k1_relion_seeded_debug",
             _pytest(py, *long_flags, f"{LONG}::test_em_parity_long_k1_full[relion_seeded_debug]"), True,
             LONG_ARM_SECONDS["long_k1_relion_seeded_debug"]),
        Item("long_k1_native_vdam",
             _pytest(py, *long_flags, f"{LONG}::test_em_parity_long_k1_native_initialmodel_quality"), True,
             LONG_ARM_SECONDS["long_k1_native_vdam"]),
        Item("long_kclass", _pytest(py, *long_flags, f"{LONG}::test_em_parity_long_kclass_full"), True,
             LONG_ARM_SECONDS["long_kclass"]),
        *[Item(f"long_realdata_hp3_{arm}",
               _pytest(py, *long_flags, f"{LONG}::test_em_parity_long_realdata_hp3_replay[{arm}]"), True,
               LONG_ARM_SECONDS[f"long_realdata_hp3_{arm}"]) for arm in ("default", "resident")],
        Item("long_class3d_hp4", _pytest(py, *long_flags, f"{LONG}::test_em_parity_long_class3d_hp4_global"), True,
             LONG_ARM_SECONDS["long_class3d_hp4"]),
        Item("completion_k1", ["bash", str(jobs / "em_completion_k1_100k256.sh")], True,
             LONG_ARM_SECONDS["completion_k1"]),
        Item("completion_k4", ["bash", str(jobs / "em_completion_k4_100k256.sh")], True,
             LONG_ARM_SECONDS["completion_k4"]),
        Item("completion_summary", ["bash", str(root / "completion" / "summarize.sh")], False, 600,
             after=["completion_k1", "completion_k4"]),
        Item("long_tables",
             [py, "scripts/extract_em_parity_tables.py", "--tier", "long", "--ledger-root", str(root / "items"),
              "--require-case", "k1_long", "k1_native_initialmodel", "kclass_long"], False, 30,
             after=["long_k1_standalone", "long_k1_native_vdam", "long_kclass"]),
        # FSC against the RELION repeat bands, unmasked and masked: reported until the thresholds are approved.
        Item("bands", [py, "scripts/em_tier_bands.py", "--run-root", str(root), "--output",
                       str(root / "bands.json")], False, 900,
             after=["long_k1_standalone", "long_kclass", "completion_k1", "completion_k4"]),
    ]
    return items


# ----------------------------------------------------------------------------- execution


def _junit_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    keys = ("tests", "failures", "errors", "skipped")
    return {k: sum(int(s.get(k, 0)) for s in suites) for k in keys}


def run_item(item: Item, run_root: Path, src: Path, gpu: str | None) -> dict:
    out = run_root / "items" / item.name
    out.mkdir(parents=True, exist_ok=True)
    argv = list(item.argv)
    if "pytest" in argv:
        argv += ["--basetemp", str(out / "bt"), f"--junitxml={out / 'junit.xml'}"]
    env = dict(os.environ, **item.env)
    env["TMPDIR"] = str(out / "tmp")
    if env.get("RELAX_TIER_DIAGNOSE_TIMING") == "1":  # per-iteration phase timings of the refinement
        env["RELAX_PARITY_TIMING_DIR"] = str(out / "timing")
    (out / "tmp").mkdir(exist_ok=True)
    if gpu is None:
        env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        env.update(CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS="cuda,cpu")
    (out / "command.txt").write_text(shlex.join(argv) + "\n")
    t0 = time.time()
    with open(out / "log.txt", "w") as log:
        rc = subprocess.run(argv, cwd=src, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    counts = _junit_counts(out / "junit.xml")
    status = "pass" if rc == 0 else "fail"
    if status == "pass" and item.required and counts.get("skipped", 0):
        status = "fail"  # a required case that did not execute is not a pass
    if status == "pass" and "pytest" in argv and not counts.get("tests"):
        status = "fail"
    return {
        "name": item.name,
        "gpu": gpu,
        "rc": rc,
        "status": status,
        "wall_s": round(time.time() - t0, 1),
        "estimate_s": item.seconds,
        "required": item.required,
        "junit": counts,
    }


def execute(items: list[Item], run_root: Path, src: Path, gpus: list[str]) -> list[dict]:
    """One worker process per GPU takes the longest ready GPU item; CPU items run in list order on
    their own worker. An item waits for the items in its ``after`` list and is not run (fails)
    when one of them failed."""
    results: dict[str, dict] = {}
    lock = threading.Condition()
    pending = sorted((i for i in items if i.gpu), key=lambda i: -i.seconds)

    def state(item: Item) -> str:
        done = [results.get(d) for d in item.after]
        if any(r is not None and r["status"] != "pass" for r in done):
            return "blocked"
        return "ready" if all(r is not None for r in done) else "waiting"

    def finish(item: Item, res: dict) -> None:
        with lock:
            results[item.name] = res
            lock.notify_all()
        print(json.dumps(res), flush=True)

    def blocked(item: Item, gpu: str | None) -> dict:
        return {"name": item.name, "gpu": gpu, "rc": None, "status": "fail", "wall_s": 0.0, "estimate_s": item.seconds,
                "required": item.required, "junit": {}, "note": f"not run: a prerequisite in {item.after} failed"}

    def gpu_worker(gpu: str) -> None:
        while True:
            with lock:
                while True:
                    if not pending:
                        return
                    choice = next((i for i in pending if state(i) != "waiting"), None)
                    if choice is not None:
                        pending.remove(choice)
                        break
                    lock.wait(timeout=30)
            finish(choice, blocked(choice, gpu) if state(choice) == "blocked" else run_item(choice, run_root, src, gpu))

    def cpu_worker() -> None:
        for item in (i for i in items if not i.gpu):
            with lock:
                while state(item) == "waiting":
                    lock.wait(timeout=30)
            finish(item, blocked(item, None) if state(item) == "blocked" else run_item(item, run_root, src, None))

    threads = [threading.Thread(target=cpu_worker)] + [threading.Thread(target=gpu_worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(results.values(), key=lambda r: r["name"])


def score_fsc(run_root: Path, src: Path) -> dict:
    """FSC of the fast parity cases present under items/ against their RELION oracles (reported)."""
    py = str(src / ".pixi" / "envs" / "default" / "bin" / "python")
    out = run_root / "fsc.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    proc = subprocess.run(
        [py, "scripts/em_tier_fsc.py", "relax", "--run-root", str(run_root / "items"), "--output", str(out)],
        cwd=src,
        env=env,
        capture_output=True,
        text=True,
    )
    (run_root / "fsc.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode or not out.exists():
        return {"status": "error", "log": str(run_root / "fsc.log")}
    return {"status": "reported", "path": str(out)}


def compare_pinned(run_root: Path, src: Path, tier: str, gpu_model: str) -> dict:
    """Compare fsc.json with the relax summaries pinned on the same GPU model; enforced only once approved."""
    py = str(src / ".pixi" / "envs" / "default" / "bin" / "python")
    proc = subprocess.run(
        [
            py,
            "scripts/em_tier_pinned.py",
            "compare",
            "--tier",
            tier,
            "--fsc",
            str(run_root / "fsc.json"),
            "--output",
            str(run_root / "pinned_compare.json"),
            "--gpu-model",
            gpu_model,
        ],
        cwd=src,
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"),
    )
    (run_root / "pinned_compare.log").write_text(proc.stdout + proc.stderr)
    try:
        return json.loads((run_root / "pinned_compare.json").read_text())["verdict"]
    except (OSError, KeyError, json.JSONDecodeError):
        return {"status": "error", "log": str(run_root / "pinned_compare.log")}


def gpu_models(run_root: Path, gpus: list[str]) -> dict[str, str]:
    """GPU model of every visible device id (index or UUID), from nvidia-smi inside the allocation."""
    table = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"], text=True)
    (run_root / "GPU.csv").write_text("index, name, uuid\n" + table)
    rows = [[x.strip() for x in line.split(",")] for line in table.strip().splitlines()]
    models = {}
    for g in gpus:
        match = [r for r in rows if g in (r[0], r[2])]
        if len(match) != 1:
            raise SystemExit(f"cannot identify visible GPU {g!r} in nvidia-smi output")
        models[g] = match[0][1]
    return models


def cmd_run(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    src = run_root / "src"
    spec = json.loads((run_root / "PLAN.json").read_text())
    items = [Item(**i) for i in spec["items"]]
    # The run must load natives built from this candidate's own native sources.
    stale = check_native_sources(run_root / "natives", src)
    if stale:
        raise SystemExit(f"refusing to run the {spec['tier']} tier: {stale}")
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    if not gpus:
        raise SystemExit("no visible GPU: the tier must run inside a GPU allocation")
    models = gpu_models(run_root, gpus)
    t0 = time.time()
    results = execute(items, run_root, src, gpus)
    wall = time.time() - t0
    for r in results:
        r["gpu_model"] = models.get(r["gpu"]) if r["gpu"] else None
    model = ",".join(sorted(set(models.values())))
    fsc = score_fsc(run_root, src) if spec["tier"] in ("smoke", "medium") else {}
    pinned = compare_pinned(run_root, src, spec["tier"], model) if fsc.get("status") == "reported" else {}
    failed = [r["name"] for r in results if r["status"] != "pass"]
    if pinned.get("status") == "fail" and pinned.get("enforced"):
        failed.append("pinned_fsc")
    summary = {
        "tier": spec["tier"],
        "status": "fail" if failed else "pass",
        "failed_items": failed,
        "wall_s": round(wall, 1),
        "budget_s": BUDGET_S[spec["tier"]],
        "gpus": gpus,
        "gpu_models": models,
        "items": results,
        "fsc": fsc,
        "pinned": pinned,
    }
    (run_root / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_receipt(run_root, spec, summary["status"], wall, model)
    print(f"tier {spec['tier']}: {summary['status']} in {wall / 60:.1f} min; failed: {failed}", flush=True)
    return 0 if not failed else 1


def write_receipt(run_root: Path, spec: dict, status: str, wall: float | None, gpu_model: str) -> None:
    job = os.environ.get("SLURM_JOB_ID") or spec.get("where", "local")
    subprocess.run(
        [
            sys.executable,
            str(run_root / "src" / "scripts" / "write_test_receipt.py"),
            "--run-root",
            str(run_root),
            "--tier",
            spec["tier"],
            "--status",
            status,
            "--job",
            str(job),
            "--gpu-model",
            gpu_model,
        ]
        + (["--wall-s", f"{wall:.0f}"] if wall is not None else []),
        check=False,
    )


# ----------------------------------------------------------------------------- submission


def _git(src: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(src), *args], text=True)


def freeze_source(run_root: Path) -> dict:
    """Detached worktree of HEAD plus the uncommitted diff; the queued job never sees later edits."""
    src = run_root / "src"
    head = _git(REPO_ROOT, "rev-parse", "HEAD").strip()
    diff = _git(REPO_ROOT, "diff", "HEAD", "--binary")
    untracked = _git(REPO_ROOT, "ls-files", "--others", "--exclude-standard").split()
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(src), head],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if diff:
        (run_root / "uncommitted.patch").write_text(diff)
        subprocess.run(["git", "-C", str(src), "apply", "--binary", str(run_root / "uncommitted.patch")], check=True)
    for rel in untracked:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, src / rel)
    # The frozen source shares this checkout's pixi environment (a full copy is ~12 GB). Its
    # editable relax install points here, so every tier process runs with PYTHONPATH (or its
    # working directory) at the frozen source first; the fast guard asserts that provenance.
    (src / ".pixi").symlink_to((REPO_ROOT / ".pixi").resolve())
    import hashlib

    return {
        "head": head,
        "dirty": bool(diff or untracked),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "untracked": {rel: hashlib.sha256((src / rel).read_bytes()).hexdigest() for rel in untracked},
        "src": str(src),
    }


def verify_fixtures(tier: str, src: Path) -> None:
    py = str(src / ".pixi" / "envs" / "default" / "bin" / "python")
    proc = subprocess.run([py, "tests/helpers/em_fixtures.py", *FIXTURE_SETS[tier]], cwd=src)
    if proc.returncode:
        raise SystemExit("fixture verification failed; the tier cannot run (missing or changed data)")


def build_natives(run_root: Path, src: Path) -> Path:
    natives = run_root / "natives"
    subprocess.run(["bash", str(src / "scripts" / "build_test_natives.sh"), str(natives)], check=True)
    return natives


def _env_lines(run_root: Path, natives: Path) -> str:
    src = run_root / "src"
    exports = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
        "RECOVAR_JAX_CACHE_DIR": f"{run_root}/jax",
        "JAX_COMPILATION_CACHE_DIR": f"{run_root}/jax",
        "RECOVAR_CUDA_CACHE_DIR": f"{run_root}/cuda_cache",
        "RECOVAR_CUDA_LIB": f"{natives}/libcuda_backproject.so",
        "RELAX_CUDA_LIB": f"{natives}/librelax_cuda.so",
        "RECOVAR_RELION_BIND_BUILD_DIR": f"{natives}/relion_bind",
        "RELAX_REQUIRE_CUSTOM_CUDA_FOR_TESTS": "1",
        "RELION_SRC_DIR": "/scratch/gpfs/GILLES/mg6942/relion/src",
        "PYTHONPATH": str(src),
    }
    for name in ("RELAX_TEST_RECEIPTS", "RELAX_TIER_DIAGNOSE_TIMING"):  # sbatch --export=NONE drops them otherwise
        if os.environ.get(name):
            exports[name] = os.environ[name]
    lines = ["unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH"]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in exports.items()]
    lines.append(f"mkdir -p {run_root}/jax {run_root}/cuda_cache")
    return "\n".join(lines)


QUEUES = {
    # della-cryoem QOS: 16 running jobs and 32 GPUs per user; A100 and H100 nodes.
    "cryoem": ["#SBATCH --partition=cryoem", "#SBATCH --qos=della-cryoem"],
    # General GPU partition: Slurm routes by the time limit (gpu-test, gpu-short: 44 jobs / 44 GPUs,
    # 1 day); A100 only, and gpu80 keeps the jobs off the 40 GB cards.
    "general": ["#SBATCH --constraint=gpu80"],
}


def write_sbatch(run_root: Path, tier: str, natives: Path, queue: str, gpu_model: str) -> Path:
    """One shared-node job per tier with one worker process per GPU (TIER_JOB sizes it)."""
    src = run_root / "src"
    py = src / ".pixi" / "envs" / "default" / "bin" / "python"
    job = TIER_JOB[tier]
    lines = list(QUEUES[queue])
    if gpu_model != "any":
        if queue == "general" and gpu_model != "a100":
            raise SystemExit(f"the general GPU partition has no {gpu_model}")
        lines = [line for line in lines if "--constraint" not in line] + [
            f"#SBATCH --constraint={gpu_model}" + (",gpu80" if queue == "general" else "")
        ]
    directives = "\n".join(lines)
    script = run_root / f"{tier}_{queue}.sbatch"
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name=relax-{tier}
#SBATCH --account=gilles
{directives}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={8 * job["gpus"]}
#SBATCH --mem={job["mem_gb"]}G
#SBATCH --gres=gpu:{job["gpus"]}
#SBATCH --time={job["time"]}
#SBATCH --export=NONE
#SBATCH --output={run_root}/job-%j.log
set -uo pipefail
{_env_lines(run_root, natives)}
# One worker process per GPU shares the job's CPUs: 8 host threads each.
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# Node load and GPU use every 30 s, to tell a slow tier from a busy node.
( while true; do date -Is; uptime; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; sleep 30; done ) > {run_root}/node_monitor.log 2>&1 &
cd {src}
{py} {src}/scripts/run_test_tier.py run {tier} --run-root {run_root}
""")
    return script


def expected_start(script: Path) -> dt.datetime | None:
    proc = subprocess.run(["sbatch", "--test-only", str(script)], capture_output=True, text=True)
    m = re.search(r"to start at (\S+)", proc.stdout + proc.stderr)
    return dt.datetime.fromisoformat(m.group(1)) if m else None


def idle_local_gpu(gpu_model: str = "any") -> str | None:
    """UUID of an idle physical GPU among 1-3 (no compute process, < 1 GiB used) of the requested model."""
    try:
        gpus = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,name", "--format=csv,noheader,nounits"], text=True
        )
        busy = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        ).split()
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in gpus.strip().splitlines():
        index, uuid, used, name = (x.strip() for x in line.split(","))
        if gpu_model != "any" and gpu_model.lower() not in name.lower():
            continue
        if index in LOCAL_GPUS and uuid not in busy and int(used) < 1024:
            return uuid
    return None


def prepare_long(run_root: Path, src: Path) -> None:
    """Write the completion arms' job scripts (launcher --dry-run) and their summary command."""
    from importlib.util import module_from_spec, spec_from_file_location

    loader = spec_from_file_location("em_fixtures", src / "tests" / "helpers" / "em_fixtures.py")
    fx = module_from_spec(loader)
    loader.loader.exec_module(fx)
    repeats = fx.fixture_root("k1_100k256_relion_repeats")
    oracle = fx.fixture_root("k4_100k256_dispatch_oracle")
    completion = run_root / "completion"
    env = dict(
        os.environ,
        EM_COMPLETION_SCRATCH_DIR=str(completion),
        EM_COMPLETION_RUNTIME_ROOT=str(completion / "runtime"),
        K1_TRAJECTORY_MODE="standalone",
        K1_MAX_ITER="25",  # headroom past RELION's convergence (it016 + final pass), as the Q run used
        K1_RELION_PARTICLE_SHUFFLE="mt19937",
        K1_RELION_REPEAT_DIRS=f"{repeats / 'rep1'} {repeats / 'rep2'}",
        K4_RELION_DIR=str(oracle / "run"),
        K4_RELION_DISPATCH_SCHEDULE=str(oracle / "schedule" / "dispatch_schedule.npz"),
    )
    proc = subprocess.run(["bash", "scripts/run_em_completion_bench_slurm.sh", "--dry-run"], cwd=src, env=env,
                          capture_output=True, text=True)
    (run_root / "completion_launch.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise SystemExit(f"completion launcher dry run failed; see {run_root / 'completion_launch.log'}")
    # The setup script (venv with the frozen source installed, RELION binding) is CPU work: run it
    # here so no GPU of the allocation waits on it.
    with open(completion / "em_completion_setup.out", "w") as log:
        if subprocess.run(["bash", str(completion / "jobs" / "em_completion_setup.sh")], cwd=src,
                          env=env | {"SLURM_JOB_ID": "submission"},
                          stdout=log, stderr=subprocess.STDOUT).returncode:
            raise SystemExit(f"completion setup failed; see {completion / 'em_completion_setup.out'}")
    k1_relion = fx.fixture_root("k1_100k256_relion")
    k4_relion = oracle / "run"
    py = src / ".pixi" / "envs" / "default" / "bin" / "python"
    (completion / "summarize.sh").write_text(f"""#!/bin/bash
set -uo pipefail
export JAX_PLATFORMS=cpu RECOVAR_DISABLE_CUDA=1
cd {src}
{py} -m scripts.summarize_em_completion_bench --require-k1 --require-k4 \\
  --k1-recovar-dir {completion}/k1_100k256_recovar --k1-relion-dir {k1_relion} \\
  --k1-fixture-dir {fx.fixture_root("k1_100k256_data")} --k1-relion-repeat-dirs {repeats / 'rep1'} {repeats / 'rep2'} \\
  --k4-recovar-dir {completion}/k4_100k256_recovar --k4-relion-dir {k4_relion} \\
  --k4-fixture-dir {fx.fixture_root("k4_100k256_data")} \\
  --output-json {completion}/summary_metrics.json --output-markdown {completion}/summary.md
""")


def cmd_submit(args: argparse.Namespace) -> int:
    tier = args.tier
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    head = _git(REPO_ROOT, "rev-parse", "--short", "HEAD").strip()
    run_root = (args.run_root or RUN_BASE / f"{tier}_{head}_{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root.parent / "SAFE_TO_DELETE").touch()
    source = freeze_source(run_root)
    src = Path(source["src"])
    verify_fixtures(tier, src)
    if tier == "long":
        prepare_long(run_root, src)
    items = plan(tier, src, args.base, run_root)
    spec = {"tier": tier, "source": source, "base": args.base, "created": stamp, "items": [asdict(i) for i in items]}
    (run_root / "PLAN.json").write_text(json.dumps(spec, indent=1) + "\n")
    for i in items:
        print(f"  {'GPU' if i.gpu else 'CPU'} {i.name:36s} ~{i.seconds / 60:6.1f} min", flush=True)
    if args.dry_run:
        print(f"dry run: plan at {run_root}/PLAN.json")
        return 0
    natives = build_natives(run_root, src)
    if args.gpu_model == "h100" and args.queue != "cryoem":
        raise SystemExit("H100 nodes are in the cryoem partition only")
    queue = args.queue
    start = expected_start(write_sbatch(run_root, tier, natives, queue, args.gpu_model))
    print(f"expected start on {queue}: {start}", flush=True)
    busy = start is None or (start - dt.datetime.now()).total_seconds() > args.queue_wait_minutes * 60
    if tier == "smoke" and args.where != "slurm" and (args.where == "local" or busy):
        # Only smoke may run locally: when the cryoem queue cannot start it promptly, on one idle
        # physical GPU 1-3 (never GPU 0), chosen by UUID after an nvidia-smi check.
        uuid = idle_local_gpu(args.gpu_model)
        if uuid is None:
            if args.where == "local":
                raise SystemExit("no idle local GPU among 1-3")
            print("the cryoem queue is busy and no local GPU 1-3 is idle; submitting to Slurm anyway")
        else:
            spec["where"] = f"local:{uuid}"
            (run_root / "PLAN.json").write_text(json.dumps(spec, indent=1) + "\n")
            print(f"running smoke on local GPU {uuid}")
            env_script = run_root / "smoke_local.sh"
            env_script.write_text(
                f"#!/bin/bash\nset -uo pipefail\n{_env_lines(run_root, natives)}\nexport CUDA_VISIBLE_DEVICES={uuid}\n"
                "export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8\n"
                f"cd {src}\n{src}/.pixi/envs/default/bin/python {src}/scripts/run_test_tier.py run smoke --run-root {run_root}\n"
            )
            return subprocess.run(["bash", str(env_script)]).returncode
    job = subprocess.check_output(["sbatch", "--parsable", str(run_root / f"{tier}_{queue}.sbatch")], text=True).strip()
    (run_root / "JOB.txt").write_text(job + "\n")
    print(f"submitted {tier} tier: Slurm {job} ({queue}); run root {run_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit", help="freeze the checkout and start a tier")
    p.add_argument("tier", choices=["smoke", "medium", "long"])
    p.add_argument("--run-root", type=Path)
    p.add_argument("--base", default="origin/main", help="changes against this ref select smoke's GPU unit files")
    p.add_argument("--where", choices=["auto", "slurm", "local"], default="auto",
                   help="smoke only: auto = cryoem if it starts within --queue-wait-minutes, else one idle local GPU 1-3")
    p.add_argument("--queue-wait-minutes", type=int, default=15)
    p.add_argument(
        "--gpu-model",
        choices=["any", "a100", "h100"],
        default="any",
        help="pin the GPU model (Slurm constraint); required when the run is compared numerically with a control",
    )
    p.add_argument("--queue", choices=["cryoem", "general"], default="cryoem",
                   help="cryoem (all GPU jobs); general only when the user explicitly allows another partition")
    p.add_argument("--dry-run", action="store_true", help="write the plan, build and submit nothing")
    p = sub.add_parser("run", help="execute a planned tier inside its allocation")
    p.add_argument("tier", choices=["smoke", "medium", "long"])
    p.add_argument("--run-root", type=Path, required=True)
    p = sub.add_parser("plan", help="print a tier's items for this checkout")
    p.add_argument("tier", choices=["smoke", "medium", "long"])
    p.add_argument("--base", default="origin/main")
    args = parser.parse_args(argv)
    if args.cmd == "submit":
        return cmd_submit(args)
    if args.cmd == "run":
        return cmd_run(args)
    for i in plan(args.tier, REPO_ROOT, args.base):
        print(f"{'GPU' if i.gpu else 'CPU'} {i.name:36s} ~{i.seconds / 60:5.1f} min  {shlex.join(i.argv[3:])[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
