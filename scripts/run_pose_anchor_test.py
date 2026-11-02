#!/usr/bin/env python
"""Pose-only test of anchoring the target: estimate its 3D position from its neighbours' lines. No images.

    python -m scripts.run_pose_anchor_test --json <out.json>

The anchor is the least-squares point of the K nearest particles' depth lines. A candidate is then scored by the
rotation term plus its squared distance from that point to its own line, and the K nearest are taken again, for three
rounds. The seed ranking is the bounded score, the rotation term, or the rank 1 or rank 2 fitted motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from relax.multibody.pose_neighborhoods import compatibility_distance, fit_low_rank_motion, low_rank_distance, rotation_distance
from scripts.run_pose_motion_shapes import KEPT, NOISE, simulate, targets

FAMILIES = ("translation", "mixed", "arc", "branch")
NEIGHBOURS = (30, 100)
ROUNDS = 3


def lines_point(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """The point nearest in least squares to the lines through ``points`` along ``normals``."""
    projection = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    return np.linalg.lstsq(projection.sum(axis=0), np.einsum("nij,nj->i", projection, points), rcond=1e-6)[0]


def anchored(features, target: int, seed_score: np.ndarray, neighbours: int) -> tuple[np.ndarray, np.ndarray]:
    """The final score and the anchor, after ``ROUNDS`` rounds from ``seed_score``."""
    points, normals = features.points, features.normals
    rotation = rotation_distance(features, target) ** 2
    near = np.argsort(seed_score, kind="stable")[:neighbours]
    for _ in range(ROUNDS):
        anchor = lines_point(points[near], normals[near])
        offset = anchor - points
        perpendicular = offset - np.sum(offset * normals, axis=1, keepdims=True) * normals
        score = rotation + np.sum(perpendicular**2, axis=1)
        near = np.argsort(score, kind="stable")[:neighbours]
    return score, anchor


def fractions(score: np.ndarray, state: np.ndarray, target: int) -> list[float]:
    order = np.argsort(score, kind="stable")
    return [float(np.mean(state[order[:n]] == state[target])) for n in KEPT]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    report = []
    for family in FAMILIES:
        for shift_noise, angle_noise in NOISE:
            table, errors, seed_purity = {}, {}, {}
            for seed in range(args.seeds):
                sim = simulate(family, shift_noise, angle_noise, seed)
                features, state = sim["features"], sim["state"]
                motions = {"rank1": fit_low_rank_motion(features, 1), "rank2": fit_low_rank_motion(features, 2)}
                for target in targets(sim):
                    seeds = {
                        "bounded": compatibility_distance(features, target, sim["bound"]),
                        "rotation": rotation_distance(features, target),
                        **{name: low_rank_distance(motion, target) for name, motion in motions.items()},
                    }
                    for name, score in seeds.items():
                        table.setdefault(name, []).append(fractions(score, state, target))
                        for k in NEIGHBOURS:
                            final, anchor = anchored(features, target, score, k)
                            key = f"anchor K{k} from {name}"
                            table.setdefault(key, []).append(fractions(final, state, target))
                            errors.setdefault(key, []).append(float(np.linalg.norm(anchor - sim["translation"][target])))
                            near = np.argsort(score, kind="stable")[:k]
                            seed_purity.setdefault(key, []).append(float(np.mean(state[near] == state[target])))
            entry = dict(
                family=family, shift_noise=shift_noise, angle_noise=angle_noise,
                fractions={k: np.mean(v, axis=0).round(3).tolist() for k, v in table.items()},
                anchor_error_median_angstrom={k: round(float(np.median(v)), 2) for k, v in errors.items()},
                seed_purity={k: round(float(np.mean(v)), 3) for k, v in seed_purity.items()},
            )
            report.append(entry)
            print(json.dumps(entry), flush=True)
    args.json.write_text(json.dumps(dict(kept=KEPT, seeds=args.seeds, results=report), indent=1))


if __name__ == "__main__":
    main()
