#!/usr/bin/env python
"""Reconstruct a benchmark dataset with given poses and score the map against the ground truth.

The poses are a refinement's final ones (``--run-dir``: its last numbered ``run_itNNN_data.star`` holds each
particle's best pose, origin and the dataset's CTF) or the simulator's true ones (``--true-poses``: the dataset's
``particles.star``). RELION's ``relion_reconstruct`` backprojects every particle once with its CTF; it has no
half-set split and no FSC-based regularization, so the map shows what the poses alone allow. Its unmasked FSC to the
ground-truth map is written with it. ``--zero-origins`` reconstructs with the estimated angles and zero origins:
the simulated particles are centred, so this leaves out the offsets the refinement estimated.

Run::

    python scripts/pose_prior_benchmark/reconstruct_with_poses.py --dataset-dir DATASET --run-dir REFINE \\
        --out-dir OUT --relion-module relion/5.0.1/gcc-14.2.1-cpu [--zero-origins]
    python scripts/pose_prior_benchmark/reconstruct_with_poses.py --dataset-dir DATASET --true-poses \\
        --out-dir OUT --relion-module relion/5.0.1/gcc-14.2.1-cpu

``OUT`` receives ``reconstruction.mrc``, ``metrics.json`` (the poses used and the FSC summary), ``fsc_curve.csv`` and
``relion_reconstruct.log``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fsc_metrics import centered_corr, shell_fsc  # noqa: E402
from scripts.pose_prior_benchmark.evaluate_run import (  # noqa: E402
    GT_FSC_THRESHOLDS,
    GT_MAP_NAME,
    fsc_report,
    jsonable,
    list_iterations,
    read_dataset_geometry,
    require_unseeded_run,
)

MAP_NAME = "reconstruction.mrc"
PADDING = 2  # relion_reconstruct's default, stated so a change of the default cannot change the benchmark


def pose_star(dataset_dir: Path, run_dir: Path | None) -> tuple[Path, int | None]:
    """The STAR holding the poses to reconstruct with, and its iteration (None for the true poses).

    ``run_dir`` None selects the dataset's ``particles.star``; otherwise the run's last numbered iteration.
    """
    if run_dir is None:
        return dataset_dir / "particles.star", None
    require_unseeded_run(run_dir)
    iteration = list_iterations(run_dir)[-1]
    return run_dir / f"run_it{iteration:03d}_data.star", iteration


def write_zero_origin_star(star: Path, out: Path) -> None:
    """Write a copy of ``star`` in which every particle's origin is zero."""
    import starfile

    blocks = starfile.read(star, always_dict=True)
    blocks["particles"][["rlnOriginXAngst", "rlnOriginYAngst"]] = 0.0
    starfile.write(blocks, out, overwrite=True)


def run_relion_reconstruct(star: Path, dataset_dir: Path, out_map: Path, module: str | None, log: Path) -> None:
    """Run ``relion_reconstruct`` on ``star`` (image names relative to ``dataset_dir``), with CTF correction.

    ``module`` is the environment module that puts the binary and its libraries in a login shell; None: the
    binary is already on the PATH.
    """
    command = f"relion_reconstruct --i {shlex.quote(str(star))} --o {shlex.quote(str(out_map))} --ctf --pad {PADDING} --verb 0"
    if module:
        command = f"module load {shlex.quote(module)} && {command}"
    with open(log, "w") as handle:
        subprocess.run(["bash", "-lc", command], cwd=dataset_dir, stdout=handle, stderr=subprocess.STDOUT, check=True)


def score_reconstruction(reconstruction: np.ndarray, gt: np.ndarray, voxel: float) -> tuple[dict[str, Any], np.ndarray]:
    """The unmasked FSC of ``reconstruction`` to the ground-truth map ``gt``: its summary and the curve."""
    if reconstruction.shape != gt.shape:
        raise ValueError(f"the reconstruction has shape {reconstruction.shape}, the ground truth {gt.shape}")
    correlation = centered_corr(reconstruction, gt)
    if not correlation > 0:
        raise ValueError(f"reconstruction and ground truth correlate at {correlation:.3f}: the poses or frame differ")
    curve = shell_fsc(reconstruction, gt)
    box = gt.shape[0]
    return {
        "voxel_size_A": voxel,
        "box": box,
        "gt_correlation": correlation,
        "gt_reconstruction": fsc_report(curve, GT_FSC_THRESHOLDS, voxel, box),
    }, curve


def write_outputs(metrics: dict[str, Any], curve: np.ndarray, out_dir: Path) -> None:
    """Write ``metrics.json`` and ``fsc_curve.csv`` (shell, resolution in A, FSC to the ground truth)."""
    (out_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2) + "\n")
    resolution = metrics["resolution"]
    with open(out_dir / "fsc_curve.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shell", "resolution_A", "gt_reconstruction"])
        for shell, fsc in enumerate(curve):
            writer.writerow([shell, resolution["box"] * resolution["voxel_size_A"] / shell if shell else math.inf, fsc])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path, required=True, help="simulated dataset (simulate_dataset.py)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="relax refine output directory; its final poses are used")
    source.add_argument("--true-poses", action="store_true", help="use the simulator's true poses")
    parser.add_argument("--zero-origins", action="store_true", help="set every particle's origin to zero")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--relion-module", default=None, help="environment module that provides relion_reconstruct")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    from recovar.utils.helpers import load_relion_volume

    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    star, iteration = pose_star(args.dataset_dir, args.run_dir)
    used_star = star
    if args.zero_origins:
        used_star = args.out_dir / "poses.star"
        write_zero_origin_star(star, used_star)
    out_map = args.out_dir / MAP_NAME
    run_relion_reconstruct(
        used_star, args.dataset_dir, out_map, args.relion_module, args.out_dir / "relion_reconstruct.log"
    )
    if used_star != star:
        used_star.unlink()
    voxel, box = read_dataset_geometry(args.dataset_dir)
    gt = np.asarray(load_relion_volume(str(args.dataset_dir / GT_MAP_NAME)), dtype=np.float64)
    if gt.shape[0] != box:
        raise ValueError(f"the ground truth has box {gt.shape[0]}, the particles {box}")
    reconstruction = np.asarray(load_relion_volume(str(out_map)), dtype=np.float64)
    resolution, curve = score_reconstruction(reconstruction, gt, voxel)
    metrics = {
        "poses": {
            "source": "true" if args.true_poses else "estimated",
            "star": str(star),
            "iteration": iteration,
            "zero_origins": args.zero_origins,
        },
        "resolution": resolution,
    }
    write_outputs(metrics, curve, args.out_dir)
    summary = resolution["gt_reconstruction"]
    print(
        f"{metrics['poses']['source']} poses: GT-FSC area {summary['auc']:.3f}, 0.143 at {summary['at_0.143']['first_A']} A"
    )


if __name__ == "__main__":
    main()
