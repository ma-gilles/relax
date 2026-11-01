#!/usr/bin/env python
"""Stage A geometry audit of ``relax.multibody.pose_neighborhoods``: Figure 1 and its arrays.

    python -m scripts.plot_pose_neighborhood_geometry --out <dir>

Writes ``figure1.png``, ``figure1.pdf``, ``figure1_arrays.npz`` and ``stage_a_errors.json``. Errors are normalised
squared scores (by the body's mean squared radius) and describe the algebra, not a pose accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from relax.multibody.pose_neighborhoods import BodyMoments, compatibility_distance, pose_features, zero_depth_distance
from scripts.lib.pose_reference import direct_free_squared_score_high_precision, direct_squared_score


def moments_of(points, weights):
    weights = weights / weights.sum()
    centroid = weights @ points
    centred = points - centroid
    return BodyMoments(centroid, (centred * weights[:, None]).T @ centred)


def body(rng, n=60, offset=8.0):
    """An asymmetric off-centre body, angstrom-like units."""
    points = rng.normal(size=(n, 3)) * [12.0, 6.0, 3.0] + offset * np.array([1.0, -0.5, 0.25])
    return points, rng.uniform(0.0, 1.0, size=n) + 1e-3


def score(points, weights, rotations, lifts, normals, bound, *, centroid=True):
    moments = moments_of(points, weights)
    if not centroid:
        moments = BodyMoments(np.zeros(3), moments.covariance)
    return compatibility_distance(pose_features(rotations, lifts, normals, moments), 1, bound)[0] ** 2


def random_cases(n, seed=0):
    rng = np.random.default_rng(seed)
    fast, reference = np.zeros(n), np.zeros(n)
    for k in range(n):
        points, weights = body(rng, n=int(rng.integers(4, 60)), offset=float(rng.choice([0.0, 8.0, 25.0])))
        rotations = Rotation.from_rotvec(rng.normal(size=(2, 3)) * 0.5).as_matrix()
        lifts = rng.normal(size=(2, 3)) * 8.0
        normals = Rotation.random(2, random_state=rng).as_matrix()[:, :, 2]
        bound = float(rng.choice([1.0, 5.0, 10.0, 30.0, 100.0]))
        scale = float((weights / weights.sum()) @ np.sum(points * points, axis=1))
        fast[k] = score(points, weights, rotations, lifts, normals, bound) / scale
        reference[k] = direct_squared_score(points, weights, *rotations, *lifts, *normals, bound) / scale
    return fast, reference


def same_pose_views(n, seed=1):
    rng = np.random.default_rng(seed)
    points, weights = body(rng)
    translation = np.array([6.0, -3.0, 9.0])
    rotation = Rotation.from_rotvec([0.2, 0.4, -0.1]).as_matrix()
    normals = Rotation.random(n, random_state=rng).as_matrix()[:, 2, :]
    lifts = translation - (normals @ translation)[:, None] * normals
    features = pose_features(np.tile(rotation, (n, 1, 1)), lifts, normals, moments_of(points, weights))
    bound = 1.25 * np.linalg.norm(translation)
    return compatibility_distance(features, 0, bound), zero_depth_distance(features, 0), normals[:, 2] * 0 + (normals @ normals[0])


def off_centre_sweep(offsets, seed=2):
    rng = np.random.default_rng(seed)
    base, weights = body(rng, offset=0.0)
    base -= (weights / weights.sum()) @ base
    rotations = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]).as_matrix()
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    lifts = np.array([[1.0, 2.0, 0.0], [-1.0, 0.0, 3.0]])
    rows = []
    for offset in offsets:
        points = base + offset * np.array([1.0, 0.0, 0.0])
        rows.append((direct_squared_score(points, weights, *rotations, *lifts, *normals, 10.0),
                     score(points, weights, rotations, lifts, normals, 10.0),
                     score(points, weights, rotations, lifts, normals, 10.0, centroid=False)))
    return np.asarray(rows)


def parallel_sweep(angles, seed=3):
    rng = np.random.default_rng(seed)
    points, weights = body(rng)
    rotations = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.1, 0.0, 0.05]]).as_matrix()
    lifts = np.array([[0.0, 0.0, 0.0], [5.0, 2.0, 4.0]])
    moments = moments_of(points, weights)
    delta = lifts[0] - lifts[1] + (rotations[0] - rotations[1]) @ moments.centroid
    rows = []
    for angle in angles:
        normals = np.array([[0.0, 0.0, 1.0], [np.sin(angle), 0.0, np.cos(angle)]])
        cross = np.cross(*normals)
        depth = np.nan if angle == 0 else float(np.max(np.abs(
            [np.cross(delta, normals[1]) @ cross, np.cross(delta, normals[0]) @ cross])) / (cross @ cross))
        rows.append((score(points, weights, rotations, lifts, normals, 10.0), score(points, weights, rotations, lifts, normals, None),
                     direct_squared_score(points, weights, *rotations, *lifts, *normals, 10.0),
                     direct_free_squared_score_high_precision(points, weights, *rotations, *lifts, *normals), depth))
    return np.asarray(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-cases", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fast, reference = random_cases(args.n_cases)
    error = np.abs(fast - reference) / np.maximum(1.0, reference)
    accepted, naive, cosines = same_pose_views(500)
    offsets = np.linspace(0.0, 40.0, 21)
    centres = off_centre_sweep(offsets)
    angles = np.concatenate([[0.0], np.logspace(-12, 0, 49)])
    parallel = parallel_sweep(angles)
    bounded_error = np.abs(parallel[:, 0] - parallel[:, 2]) / np.maximum(1.0, parallel[:, 2])
    free_error = np.abs(parallel[:, 1] - parallel[:, 3]) / np.maximum(1.0, parallel[:, 3])
    report = {
        "tolerance": "abs(fast-reference) <= 1e-10*max(1,reference), squared scores over the body's mean squared radius; an algebra check",
        "random_cases": args.n_cases, "error_max": float(error.max()),
        "error_percentiles_50_99": [float(v) for v in np.percentile(error, [50, 99])],
        "same_pose_max_score_angstrom": float(accepted.max()), "same_pose_naive_median_angstrom": float(np.median(naive)),
        "parallel_bounded_error_max": float(bounded_error.max()), "parallel_free_error_max": float(free_error.max()),
        "centroid_omission_max_relative_error": float(np.max(np.abs(centres[:, 2] - centres[:, 0]) / centres[:, 0])),
        "seeds": [0, 1, 2, 3], "float": "float64",
    }
    (args.out / "stage_a_errors.json").write_text(json.dumps(report, indent=1))
    np.savez(args.out / "figure1_arrays.npz", fast=fast, reference=reference, error=error, accepted=accepted, naive=naive,
             cosines=cosines, offsets=offsets, centres=centres, angles=angles, parallel=parallel,
             bounded_error=bounded_error, free_error=free_error)
    print(json.dumps(report, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    ax = axes[0, 0]
    ax.loglog(reference, fast, ".", ms=2, color="#2a6f97")
    limits = [max(reference.min(), 1e-6), reference.max()]
    ax.plot(limits, limits, color="#555555", lw=0.8, label="identity")
    ax.set(xlabel="direct point-residual score (normalised)", ylabel="fast score (normalised)",
           title=f"Fast against direct, {args.n_cases} random cases")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.hist(np.log10(np.maximum(error, 1e-18)), bins=40, color="#2a6f97")
    ax.axvline(-10, color="#b23a48", ls="--", label="gate 1e-10")
    ax.set(xlabel="log10 error (floored at 1e-18)", ylabel="cases", title=f"Error, max {error.max():.1e}")
    ax.legend(frameon=False)
    ax = axes[0, 2]
    order = np.argsort(cosines)
    ax.plot(cosines[order], naive[order], ".", ms=3, color="#b23a48", label="zero-depth vector distance (naive)")
    ax.plot(cosines[order], np.maximum(accepted[order], 1e-16), ".", ms=3, color="#2a6f97", label="bounded score")
    ax.set_yscale("log")
    ax.set(xlabel="cosine between the two viewing directions", ylabel="distance (A)", title="One physical pose, 500 views")
    ax.legend(frameon=False, loc="center right")
    ax = axes[1, 0]
    ax.plot(offsets, centres[:, 0], color="#555555", lw=3, label="direct")
    ax.plot(offsets, centres[:, 1], color="#2a6f97", ls="--", label="corrected formula")
    ax.plot(offsets, centres[:, 2], color="#b23a48", label="centroid term omitted (wrong)")
    ax.set(xlabel="centroid offset from pivot (A)", ylabel="squared score (A^2)", title="Off-centre body")
    ax.legend(frameon=False)
    ax = axes[1, 1]
    shown = angles.copy()
    shown[0] = 1e-13
    ax.semilogx(shown, parallel[:, 0], color="#2a6f97", marker=".", label="bounded, L = 10 A")
    ax.semilogx(shown, parallel[:, 1], color="#e0a030", marker=".", label="free depth")
    ax.plot(shown[0], parallel[0, 0], "s", color="#2a6f97", ms=6, label="exactly parallel (drawn at 1e-13)")
    ax.set(xlabel="angle between viewing directions (rad)", ylabel="squared score (A^2)", title="Nearly parallel views")
    ax.legend(frameon=False)
    ax = axes[1, 2]
    ax.loglog(shown[1:], parallel[1:, 4], color="#e0a030", label="depth the free solution needs")
    ax.axhline(10.0, color="#2a6f97", ls="--", label="bound L = 10 A")
    ax.set(xlabel="angle between viewing directions (rad)", ylabel="depth (A)", title="Depth used by the free solution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "figure1.png", dpi=150)
    fig.savefig(args.out / "figure1.pdf")


if __name__ == "__main__":
    main()
