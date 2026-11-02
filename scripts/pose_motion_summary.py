#!/usr/bin/env python
"""Each body's fitted motion and pose noise in pose-neighborhood fixtures.

    python -m scripts.pose_motion_summary --root data/pose_neighborhoods --json summary.json <dataset> ...

For a dataset's RELION multi-body run: the rotation share of the rank 1 fitted motion and the noise it leaves (no truth
used), and the same poses against the simulated true poses: the rotation error about each axis, and the error of the
projected body position in each image direction, after removing their means.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def against_truth(fitted, true) -> dict:
    """Per-axis rotation error (deg) and per-direction image error (angstroms) of ``PoseFeatures`` of one body."""
    from scipy.spatial.transform import Rotation

    # F = Q C^(1/2), so F_fitted F_true^(-1) is the rotation between the two poses.
    error = Rotation.from_matrix(fitted.shape_factors @ np.linalg.inv(true.shape_factors)).as_rotvec()
    delta = fitted.points - true.points
    delta = delta - np.sum(delta * fitted.normals, axis=1, keepdims=True) * fitted.normals
    delta = delta - delta.mean(axis=0)
    error = error - np.median(error, axis=0)
    in_plane = np.einsum("nij,nj->ni", np.linalg.svd(fitted.normals[:, None], full_matrices=True)[2][:, 1:], delta)
    return {
        "rotation_error_robust_deg_per_axis": float(np.rad2deg(1.4826 * np.median(np.abs(error)))),
        "translation_error_robust_angstrom_per_direction": float(1.4826 * np.median(np.abs(in_plane))),
        "fraction_rotation_error_over_5_deg": float(np.mean(np.linalg.norm(error, axis=1) > np.deg2rad(5.0))),
        "rotation_error_deg_per_axis": float(np.rad2deg(np.sqrt(np.mean(np.sum((error - error.mean(axis=0)) ** 2, axis=1)) / 3.0))),
        "translation_error_angstrom_per_direction": float(np.sqrt(np.mean(np.sum(delta**2, axis=1)) / 2.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    from relax.commands.multibody import run_options
    from relax.multibody import subsets
    from relax.multibody.pose_neighborhoods import fit_low_rank_motion, mask_moments, motion_summary
    from relax.multibody.refine import MultibodyCommand, start_from_consensus
    from scripts.multibody_adaptive_subsets import oracle_result

    report = {}
    for name in args.datasets:
        data = args.root / name
        consensus = sorted(glob.glob(str(data / "relion_consensus/run_it*_optimiser.star")))[-1]
        command = MultibodyCommand(body_star=data / "bodies.star", healpix_order=4, auto_local_healpix_order=4, offset_range_px=3.0,
                                   offset_step_px=1.5, oversampling=1, reconstruct_subtracted_bodies=True, max_iter=0)
        run, _ = start_from_consensus(consensus, command, run_options(), output=data / "unused_motion_summary")
        result = subsets.read_multibody_result(data / "relion_multibody", run)
        truth = oracle_result(run, result, data)
        for b, body in enumerate(run.bodies):
            features = subsets.body_pose_features(run, result, b)
            covariance = mask_moments(body.mask, body.com, run.pixel_size).covariance
            motion = fit_low_rank_motion(features, 1)
            report[f"{name}_body{b + 1}"] = {
                **motion_summary(features, motion, covariance),
                **against_truth(features, subsets.body_pose_features(run, truth, b)),
            }
            print(f"{name} body {b + 1}", report[f"{name}_body{b + 1}"], flush=True)
    args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
