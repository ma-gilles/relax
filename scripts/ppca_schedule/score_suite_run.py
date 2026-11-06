"""Score a finished PPCA run with the curated suite's scorer and the direct pose fit, and write one JSON.

The run trained on the RELION-sign STAR of a suite dataset (scripts/prepare_vdam_ppca_star.py), so each scored checkpoint is
copied, with its mean negated, into a work directory named <dataset>__<arm>_s<seed> (scripts/ppca_suite_score_star.py; the
direct fit names the dataset from that folder). There the suite's scorer infers latents and poses on the evaluation
subsets (GPU), then the direct fit (CPU) gives the pose metric of record and the per-state maps in that frame. A
dataset of true rank 0 (H1) has no PCs to stack: the scorer stops after writing its inference and h1_score.py scores it.

The suite's scorer, its environment files and the direct pose fit are not part of this repository, so every location is an
argument and none has a default:
  --suite-tools   the suite's tools directory: suite_score.py, and gpu_env.sh and cpu_env.sh when it has them
  --fit-tools     the directory holding direct_fit.py and h1_score.py
  --gpu-env, --cpu-env   shell files sourced before the scorer and before the fit (default: <suite-tools>/gpu_env.sh and
                  cpu_env.sh when they exist, else the caller's environment); they may define PY, the interpreter
  --scorer-python, --fit-python   the interpreter of the scorer and of the fit (default: PY from the sourced file)
Every tool is checked before any scoring starts.

Run (on a GPU node, with the pixi python of this checkout):
  python scripts/ppca_schedule/score_suite_run.py RUN_DIR --dataset C__realistic__noise1__b128crop64 --arm vdam_local \
      --seed 11 --datasets-root <suite>/datasets --suite-tools <suite>/tools --fit-tools <dir of direct_fit.py> \
      --checkpoints 1000 3000 -o RUN_DIR/suite_scores.json
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

# Run as python scripts/ppca_schedule/score_suite_run.py: the repository root must be on the path.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ppca_suite_score_star import write_mean_negated_copy  # noqa: E402

# The scorer's summary entries that are not kept: its pose is the map-frame one, the direct fit's being the metric of record.
NOT_SCORES = ("dataset", "checkpoint", "q", "pose_median_deg", "pose_frac_below_10deg")


def run_in_environment(env_script: Path | None, commands: str, log: Path) -> int:
    """Run ``commands`` in bash after sourcing ``env_script`` (if any), writing stdout and stderr to ``log``; return the status."""
    source = f"source {shlex.quote(str(env_script))}\n" if env_script else ""
    script = f"set -uo pipefail\n{source}{commands}\n"
    with open(log, "w") as stream:
        return subprocess.run(["bash", "-c", script], stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def environment_file(given: Path | None, tools: Path, name: str) -> Path | None:
    """The shell file to source: ``given``, else ``tools``/``name`` when it exists, else None (the caller's environment)."""
    if given is not None:
        return given
    default = tools / name
    return default if default.exists() else None


def set_python(python: Path | None) -> str:
    """The shell line that makes ``python`` the interpreter ``$PY``; empty to keep the sourced file's ``$PY``."""
    return f"export PY={shlex.quote(str(python))}\n" if python else ""


def check_inputs(needed: dict[str, Path | None]) -> None:
    """Raise FileNotFoundError naming every entry of ``needed`` (label -> path) that does not exist; a None path is skipped."""
    missing = [f"{label}: {path}" for label, path in needed.items() if path is not None and not path.exists()]
    if missing:
        raise FileNotFoundError("Not found, so nothing was scored:\n  " + "\n  ".join(missing))


def pc_over_mean_power(checkpoint: Path) -> float:
    """Total power of the loading columns over that of the mean column of a checkpoint's model."""
    with np.load(checkpoint, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"])
    return float(np.sum(np.abs(theta[:, 1:]) ** 2) / np.sum(np.abs(theta[:, 0]) ** 2))


def per_state_fsc(work: Path, checkpoint: int) -> list[float]:
    """The direct fit's FSC of each state's map at one checkpoint (empty without per-state maps)."""
    path = work / f"suite_score_checkpoint_{checkpoint:04d}" / "direct_fit.json"
    states = json.loads(path.read_text()).get("states") or [] if path.exists() else []
    return [float(state["fsc_direct_frame"]) for state in states]


def collect_scores(work: Path, checkpoint: int) -> dict:
    """Every numeric score of the scorer's summary, the direct fit's scalars (``pose_*``, ``state_fsc_direct*``) and the PC power of one checkpoint."""
    scored = work / f"suite_score_checkpoint_{checkpoint:04d}"
    row: dict = {"pc_over_mean_power": pc_over_mean_power(work / f"checkpoint_{checkpoint:04d}.npz")}
    if (scored / "summary.json").exists():
        summary = json.loads((scored / "summary.json").read_text())
        row.update(
            {
                name: value
                for name, value in summary.items()
                if name not in NOT_SCORES and isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
    if (scored / "direct_fit.json").exists():
        fit = json.loads((scored / "direct_fit.json").read_text())
        row["pose_direct_median_deg"] = fit["pose_median_deg"]
        row["pose_direct_within_10deg"] = fit["frac_below_10"]
        if fit.get("state_fsc_mean_direct") is not None:
            row["state_fsc_direct"] = fit["state_fsc_mean_direct"]
            row["state_fsc_direct_min"] = min(per_state_fsc(work, checkpoint))
    if (scored / "h1_metrics.json").exists():
        h1 = json.loads((scored / "h1_metrics.json").read_text())
        row["mean_fsc_to_true_mean"] = h1["mean_fsc_raw_1_15_box64"]
        row["first_shell_below_0p5"] = h1["first_shell_below_0p5"]
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the run directory (checkpoint_NNNN.npz, iterations.jsonl)")
    parser.add_argument("--dataset", required=True, help="the suite dataset's folder name")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True, help="the suite's datasets/ directory")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 3000])
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="score the checkpoints without their loading columns (a run that never activated its PCs)",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--suite-tools", type=Path, required=True, help="the suite's tools directory (suite_score.py)")
    parser.add_argument("--fit-tools", type=Path, required=True, help="the directory of direct_fit.py and h1_score.py")
    parser.add_argument(
        "--gpu-env", type=Path, help="shell file sourced before the scorer (default: <suite-tools>/gpu_env.sh)"
    )
    parser.add_argument(
        "--cpu-env", type=Path, help="shell file sourced before the fit (default: <suite-tools>/cpu_env.sh)"
    )
    parser.add_argument(
        "--scorer-python", type=Path, help="interpreter of the scorer (default: PY of the GPU environment file)"
    )
    parser.add_argument(
        "--fit-python", type=Path, help="interpreter of the direct fit (default: PY of the CPU environment file)"
    )
    args = parser.parse_args(argv)

    evaluation = args.datasets_root / args.dataset / "evaluation"
    check_inputs({"--datasets-root/--dataset evaluation manifest": evaluation / "manifest.json"})
    rank_zero = json.loads((evaluation / "manifest.json").read_text())["true_rank"] == 0
    suite_score = args.suite_tools / "suite_score.py"
    fit = args.fit_tools / ("h1_score.py" if rank_zero else "direct_fit.py")
    gpu_env = environment_file(args.gpu_env, args.suite_tools, "gpu_env.sh")
    cpu_env = environment_file(args.cpu_env, args.suite_tools, "cpu_env.sh")
    check_inputs(
        {
            "--suite-tools suite_score.py": suite_score,
            f"--fit-tools {fit.name}": fit,
            "--gpu-env": gpu_env,
            "--cpu-env": cpu_env,
            "--scorer-python": args.scorer_python,
            "--fit-python": args.fit_python,
            **{f"checkpoint {c}": args.run / f"checkpoint_{c:04d}.npz" for c in args.checkpoints},
        }
    )
    work = args.run.resolve() / "suite_work" / f"{args.dataset}__{args.arm}_s{args.seed}"
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    if rank_zero:
        (work / "iterations.jsonl").unlink(missing_ok=True)
        (work / "iterations.jsonl").symlink_to(args.run.resolve() / "iterations.jsonl")
    scratch = f"export SLURM_TMPDIR={shlex.quote(str(work / 'tmp'))} TMPDIR={shlex.quote(str(work / 'tmp'))} RECOVAR_CACHE_DIR=\n"
    for checkpoint in args.checkpoints:
        name = f"checkpoint_{checkpoint:04d}"
        scored = work / f"suite_score_{name}"
        write_mean_negated_copy(args.run / f"{name}.npz", work, mean_only=args.mean_only)
        # A rank-0 dataset has no true PCs to stack, so the scorer ends nonzero after writing its inference.
        done = scored / ("inference.npz" if rank_zero else "summary.json")
        if not done.exists():
            status = run_in_environment(
                gpu_env,
                f"{scratch}{set_python(args.scorer_python)}"
                f'exec "${{PY:?no interpreter: pass --scorer-python or set PY in the GPU environment file}}" '
                f"{shlex.quote(str(suite_score))} --run {shlex.quote(str(work))} "
                f"--eval {shlex.quote(str(evaluation))} --checkpoint {shlex.quote(str(work / (name + '.npz')))}",
                work / f"{name}.score.log",
            )
            if not done.exists():
                raise RuntimeError(f"The suite scorer failed on {name} (status {status}); see {work}/{name}.score.log")
    # PYTHONPATH: the environment file's SOURCE (a pinned checkout) when it defines one, else the caller's.
    status = run_in_environment(
        cpu_env,
        f"{scratch}{set_python(args.fit_python)}"
        f'export PYTHONPATH="${{SOURCE:-${{PYTHONPATH:-}}}}" JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 OMP_NUM_THREADS=4 '
        f"SUITE_DATASETS={shlex.quote(str(args.datasets_root))}\n"
        f"cd {shlex.quote(str(work / 'tmp'))} && "
        f'exec "${{PY:?no interpreter: pass --fit-python or set PY in the CPU environment file}}" '
        f"{shlex.quote(str(fit))} {shlex.quote(str(work))}",
        work / "direct_fit.log",
    )
    if status != 0:
        raise RuntimeError(f"{fit.name} failed (status {status}); see {work}/direct_fit.log")
    scores = {str(c): collect_scores(work, c) for c in args.checkpoints}
    states = {str(c): per_state_fsc(work, c) for c in args.checkpoints}
    args.output.write_text(
        json.dumps(
            {"dataset": args.dataset, "arm": args.arm, "seed": args.seed, "checkpoints": scores, "per_state_fsc_direct": states},
            indent=1,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
