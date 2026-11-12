#!/usr/bin/env python3
"""Pose error of scored PPCA runs against the particles' true contrast.

The suite scorer's direct fit (``direct_fit.json`` in each ``suite_score_checkpoint_NNNN``) stores one global rotation and a hand; the pose error of a
particle is the angle between its inferred pose (``inference.npz``, the grid index of the best pose) and its true pose after that rotation. This script
recomputes the per-particle errors from those files (the median is checked against the stored one), then prints, for one dataset and a list of arms:

* a per-run table (median error and the share of particles below 10 deg), so that a failed seed is visible;
* a table pooled over the runs, by bin of the particle's true contrast (``evaluation/contrast.npy``).

``--contrast-dataset`` takes the true contrast from another dataset of the same particles, so that the runs on a unit-contrast control (same particles, poses and noise level,
no contrast) are binned by the contrast the particles have in the dataset that has it.

Run (CPU):
    JAX_PLATFORMS=cpu python scripts/ppca_contrast/pose_by_contrast.py ROUND [ROUND ...] --datasets-root <dir> \
        --dataset C__contrastc30__noise3__b128crop64 --arms s3_vdam_uniform co_vdam [--seeds 11 12] \
        [--contrast-dataset C__contrastc30__noise3__b128crop64] [--edges 0.7 0.9 1.1 1.3] [--checkpoint 3000] [-o table.md]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relax.sampling import grids  # noqa: E402

MIRROR_LEFT, MIRROR_RIGHT = np.diag([1.0, 1.0, -1.0]), np.diag([-1.0, 1.0, 1.0])


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between the rotation matrices of ``a`` and ``b``, both (n, 3, 3)."""
    return np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", a, b) - 1) / 2, -1, 1)))


def find_runs(rounds: list[Path], dataset: str, arm: str, seeds: list[str] | None, checkpoint: str) -> dict[str, Path]:
    """The scoring directory of each seed of ``arm`` on ``dataset``, later rounds winning over earlier ones."""
    found: dict[str, Path] = {}
    for root in rounds:
        for run in sorted((root / dataset / arm).glob("seed*")):
            seed = run.name.removeprefix("seed")
            scored = sorted(run.glob(f"suite_work/*/suite_score_checkpoint_{int(checkpoint):04d}"))
            if scored and (scored[0] / "direct_fit.json").exists() and (seeds is None or seed in seeds):
                found[seed] = scored[0]
    return found


def run_errors(scored: Path, truth_rotations: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-particle pose errors (deg) and particle ids of one scored checkpoint, from the stored global rotation and hand."""
    fit = json.loads((scored / "direct_fit.json").read_text())
    with np.load(scored / "inference.npz") as inference:
        order, ids, index = int(inference["healpix_order"]), inference["particle_ids"], inference["rotation_idx"]
    if order not in cache:
        cache[order] = grids.get_relion_hidden_rotation_grid(order, matrices=True).astype(np.float64)
    inferred = cache[order][index]
    if fit["mirror"]:
        inferred = MIRROR_LEFT @ inferred @ MIRROR_RIGHT
    errors = angle_deg(inferred, truth_rotations[ids] @ np.asarray(fit["rotation"]))
    if abs(float(np.median(errors)) - fit["pose_median_deg"]) > 0.01:
        raise ValueError(
            f"{scored}: recomputed median {np.median(errors):.3f} differs from the stored {fit['pose_median_deg']:.3f}"
        )
    return errors, ids


def cell(errors: np.ndarray) -> str:
    """``median / share below 10 deg`` of an error array."""
    return f"{np.median(errors):.2f} / {100 * np.mean(errors < 10):.1f}%" if len(errors) else ""


def tables(
    rounds: list[Path],
    datasets_root: Path,
    dataset: str,
    contrast_dataset: str,
    arms: list[str],
    seeds: list[str] | None,
    edges: list[float],
    checkpoint: str,
) -> str:
    """The per-run table and the pooled table by true-contrast bin, as Markdown."""
    with np.load(datasets_root / dataset / "evaluation" / "truth.npz") as truth:
        rotations = truth["rotations"].astype(np.float64)
    contrast = np.load(datasets_root / contrast_dataset / "evaluation" / "contrast.npy")
    bounds = [-np.inf, *edges, np.inf]
    names = [f"<{edges[0]}", *(f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])), f">{edges[-1]}"]
    cache: dict = {}
    per_arm: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for arm in arms:
        per_arm[arm] = {
            seed: run_errors(path, rotations, cache)
            for seed, path in find_runs(rounds, dataset, arm, seeds, checkpoint).items()
        }
    lines = [
        f"\n{dataset} (contrast of {contrast_dataset}); median error in degrees / share below 10 degrees\n",
        "Per run:\n",
        "| arm | seed | median / <10 |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {arm} | {seed} | {cell(errors)} |"
        for arm, runs in per_arm.items()
        for seed, (errors, _) in sorted(runs.items())
    ]
    lines += [
        "",
        "Pooled over the runs, by the true contrast of the particle:\n",
        "| arm (runs) | all | " + " | ".join(names) + " |",
        "|" + " --- |" * (2 + len(names)),
    ]
    for arm, runs in per_arm.items():
        if not runs:
            continue
        errors = np.concatenate([e for e, _ in runs.values()])
        true_c = np.concatenate([contrast[i] for _, i in runs.values()])
        bins = [cell(errors[(true_c >= lo) & (true_c < hi)]) for lo, hi in zip(bounds[:-1], bounds[1:])]
        lines.append(f"| {arm} ({len(runs)}) | {cell(errors)} | " + " | ".join(bins) + " |")
    first = next((runs for runs in per_arm.values() if runs), {})
    if first:
        ids = next(iter(first.values()))[1]
        counts = [int(((contrast[ids] >= lo) & (contrast[ids] < hi)).sum()) for lo, hi in zip(bounds[:-1], bounds[1:])]
        lines.append(f"| particles | {len(ids)} | " + " | ".join(map(str, counts)) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Print (or write) the pose tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rounds", type=Path, nargs="+")
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--contrast-dataset", help="dataset whose evaluation/contrast.npy bins the particles (default: --dataset)"
    )
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--edges", type=float, nargs="+", default=[0.7, 0.9, 1.1, 1.3])
    parser.add_argument("--checkpoint", default="3000")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    text = tables(
        args.rounds,
        args.datasets_root,
        args.dataset,
        args.contrast_dataset or args.dataset,
        args.arms,
        args.seeds,
        args.edges,
        args.checkpoint,
    )
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
