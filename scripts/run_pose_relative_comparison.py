#!/usr/bin/env python
"""Does comparing body 2 to body 1 remove the consensus coupling? A diagnostic on a fixture's poses. No reconstruction.

    python -m scripts.run_pose_relative_comparison --root data/pose_neighborhoods --out <dir> <dataset>

A consensus error ``E`` enters every body's rotation as ``E Q``, so ``Q1^T Q2`` cancels it, but it keeps body 1's own
motion. Three feature sets of body 2 go through the same orderings as ``scripts.run_pose_ordering_comparison``: its
own pose, the relative pose, and its rotation with the regression on body 1's fitted rotation removed. True cameras give
the relative pose's reference. Writes ``relative_<dataset>.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from relax.multibody.pose_neighborhoods import BodyMoments, PoseFeatures, pose_features
from scripts.run_pose_ordering_comparison import KEPT, TARGETS, TARGETS_10K, curve_sizes, score_target, true_camera_features


def rotations_of(features: PoseFeatures, moments: BodyMoments) -> np.ndarray:
    values, vectors = np.linalg.eigh(moments.covariance)
    return features.shape_factors @ (vectors / np.sqrt(values) @ vectors.T)


def relative_features(first: PoseFeatures, second: PoseFeatures, moments1: BodyMoments, moments2: BodyMoments) -> PoseFeatures:
    """Body 2 in body 1's frame: ``Q1^T Q2``, the centroid difference, and the depth direction, all turned by ``Q1^T``."""
    q1, q2 = rotations_of(first, moments1), rotations_of(second, moments2)
    turn = np.swapaxes(q1, 1, 2)
    rotations = turn @ q2
    points = np.einsum("nij,nj->ni", turn, second.points - first.points)
    return pose_features(rotations, points - rotations @ moments2.centroid, np.einsum("nij,nj->ni", turn, second.normals), moments2)


def regressed_features(first: PoseFeatures, second: PoseFeatures, moments1: BodyMoments, moments2: BodyMoments) -> PoseFeatures:
    """Body 2 with the part of its rotation vector that body 1's fitted rotation vector predicts removed."""
    r1 = Rotation.from_matrix(rotations_of(first, moments1)).as_rotvec()
    r2 = Rotation.from_matrix(rotations_of(second, moments2)).as_rotvec()
    centred = r1 - r1.mean(axis=0)
    rotations = Rotation.from_rotvec(r2 - centred @ np.linalg.lstsq(centred, r2 - r2.mean(axis=0), rcond=None)[0]).as_matrix()
    return pose_features(rotations, second.points - rotations @ moments2.centroid, second.normals, moments2)


def twist_share(rotations: np.ndarray, twist: np.ndarray) -> float:
    """The share of the rotation vectors' variance explained by a linear fit on ``twist``."""
    vectors = Rotation.from_matrix(rotations).as_rotvec()
    vectors = vectors - vectors.mean(axis=0)
    design = twist[:, None] - twist.mean()
    fitted = design @ np.linalg.lstsq(design, vectors, rcond=None)[0]
    return float(np.sum(fitted**2) / np.sum(vectors**2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("dataset")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from relax.commands.multibody import run_options
    from relax.multibody import subsets
    from relax.multibody.pose_neighborhoods import compatibility_distance, fit_low_rank_motion, low_rank_distance, mask_moments, rotation_distance
    from relax.multibody.refine import MultibodyCommand, start_from_consensus
    from scripts.multibody_adaptive_subsets import state_rows

    data = args.root / args.dataset
    record = json.loads((data / "body_motion.json").read_text())
    consensus = sorted(glob.glob(str(data / "relion_consensus/run_it*_optimiser.star")))[-1]
    command = MultibodyCommand(body_star=data / "bodies.star", healpix_order=4, auto_local_healpix_order=4, offset_range_px=3.0,
                               offset_step_px=1.5, oversampling=1, reconstruct_subtracted_bodies=True, max_iter=0)
    run, _ = start_from_consensus(consensus, command, run_options(), output=args.out / f"unused_{args.dataset}")
    result = subsets.read_multibody_result(data / "relion_multibody", run)
    names = np.asarray(run.checkpoint.particles["rlnImageName"])
    states = state_rows(data, names)
    table = record["states"]
    labels = np.unique(np.asarray([(s["body2_twist"], s["body2_shift"], s["deformation"]) for s in table]), axis=0, return_inverse=True)[1].ravel()[states]
    twist = np.asarray([s["body1_twist"] for s in table])[states]
    moments = [mask_moments(b.mask, b.com, run.pixel_size) for b in run.bodies]
    fitted = [subsets.body_pose_features(run, result, b) for b in range(2)]
    true = [true_camera_features(run, data, record, b, states, names) for b in range(2)]
    sets = {
        "own pose": fitted[1],
        "relative": relative_features(*fitted, *moments),
        "body 1 rotation regressed out": regressed_features(*fitted, *moments),
        "relative, true cameras": relative_features(*true, *moments),
    }
    bound = float(record["depth_bounds_angstrom"][1])
    report = dict(dataset=args.dataset, sets={})
    for name, features in sets.items():
        ranks = [fit_low_rank_motion(features, k) for k in (1, 2)]
        coupling = dict(
            rank1_correlation_with_body1_twist=float(abs(np.corrcoef(ranks[0].scores[:, 0], twist)[0, 1])),
            rotation_variance_share_of_body1_twist=twist_share(rotations_of(features, moments[1]), twist),
        )
        entry = dict(coupling=coupling, targets={})
        for state, image in TARGETS.get(args.dataset, TARGETS_10K):
            row = subsets.target_row(names, f"{image}@particles.128.mrcs")
            size = int(np.sum(labels == labels[row]))
            kept = tuple(sorted({*KEPT, size}))
            sizes = curve_sizes(len(names))
            tables, curves = [], []
            for r in np.flatnonzero(states == states[row])[:20]:
                r = int(r)
                scores = {
                    "bounded": compatibility_distance(features, r, bound), "rotation only": rotation_distance(features, r),
                    "rank 1": low_rank_distance(ranks[0], r), "rank 2": low_rank_distance(ranks[1], r),
                }
                tables.append(score_target(scores, labels, r, kept))
                curves.append(score_target(scores, labels, r, tuple(sizes.tolist())))
            entry["targets"][str(state)] = dict(
                state_size=size, kept=list(kept), curve_sizes=sizes.tolist(),
                same_state_mean_first_20={k: np.mean([t[k] for t in tables], axis=0).round(3).tolist() for k in tables[0]},
                curves_mean_first_20={k: np.mean([c[k] for c in curves], axis=0).round(3).tolist() for k in curves[0]},
            )
        report["sets"][name] = entry
    (args.out / f"relative_{args.dataset}.json").write_text(json.dumps(report, indent=1))
    print(args.dataset, {k: v["coupling"] for k, v in report["sets"].items()})


if __name__ == "__main__":
    main()
