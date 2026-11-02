#!/usr/bin/env python
"""Pose-only test of the low-rank motion on motion that is not a line. No images.

    python -m scripts.run_pose_motion_shapes --json <out.json>

Five families of one body's motion, from a level in [-1, 1] in 9 bins: three lines (controls), an arc and a two-path
branch (a stem that splits at level 0; a state is a bin and, past the split, a path). Views are random and independent
of state. Reported: the same-state fraction of the 100, 300 and 1000 kept particles for the bounded score, the rank 1
and chosen-rank motion, bounded as a filter then motion ordering, and the true level (oracle). The rank comes from a
truth-free check: the visible residual of each rank against the noise estimate of rank 4. That predeclared check
fails (it picks rank 3-4 on lines too, since high ranks absorb noise), so a second check, chosen after seeing it and so
exploratory, is also reported: rank 2 when the rotation noise estimate falls by more than 5% from rank 1 to 2, else 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from relax.multibody.pose_neighborhoods import (
    BodyMoments,
    compatibility_distance,
    fit_low_rank_motion,
    low_rank_distance,
    motion_summary,
    rotation_distance,
    pose_features,
)

N, BINS, MAX_RANK = 6000, 9, 4
KEPT = (100, 300, 1000)
MOMENTS = BodyMoments(np.zeros(3), np.diag([900.0, 400.0, 100.0]))
ROTATION, TRANSLATION = np.deg2rad(15.0), 10.0
AXIS = np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])
SECOND = np.cross(AXIS, [1.0, 0.0, 0.0]) / np.linalg.norm(np.cross(AXIS, [1.0, 0.0, 0.0]))
NOISE = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (4.0, 3.0))
FAMILIES = ("rotation", "translation", "mixed", "arc", "branch")
X, Y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


def motion(family: str, level: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation vectors (radians) and translations (angstroms) of a family, ``[N, 3]`` each."""
    if family == "arc":
        angle = level * np.deg2rad(60.0)
        radius = TRANSLATION / np.sin(np.deg2rad(60.0))
        translation = radius * (np.outer(np.sin(angle), X) + np.outer(1.0 - np.cos(angle), Y))
        return ROTATION * level[:, None] * (np.outer(np.cos(angle), AXIS) + np.outer(np.sin(angle), SECOND)), translation
    if family == "branch":
        sign = np.where(path == 0, 1.0, -1.0)
        split = np.deg2rad(40.0)
        direction = np.where(level[:, None] < 0, X[None], np.cos(split) * X[None] + (sign * np.sin(split))[:, None] * Y[None])
        axis = np.where(level[:, None] < 0, AXIS[None], np.cos(split) * AXIS[None] + (sign * np.sin(split))[:, None] * SECOND[None])
        return ROTATION * level[:, None] * axis, TRANSLATION * level[:, None] * direction
    rotation = np.outer(level * ROTATION * (family != "translation"), AXIS)
    return rotation, np.outer(level * TRANSLATION * (family != "rotation"), X)


def simulate(family: str, shift_noise: float, angle_noise: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    level = rng.uniform(-1.0, 1.0, N)
    path = rng.integers(0, 2, N) if family == "branch" else np.zeros(N, dtype=np.int64)
    bin_index = np.argmin(np.abs(level[:, None] - np.linspace(-1.0, 1.0, BINS)), axis=1)
    state = bin_index + BINS * (path * (level >= 0))
    rotvec, translation = motion(family, level, path)
    cameras = Rotation.random(N, random_state=rng).as_matrix()
    in_plane = np.einsum("nij,nj->ni", cameras[:, :2], translation) + rng.normal(size=(N, 2)) * shift_noise
    lifts = np.einsum("nji,nj->ni", cameras[:, :2], in_plane)
    rotations = (Rotation.from_rotvec(np.deg2rad(angle_noise) * rng.normal(size=(N, 3))) * Rotation.from_rotvec(rotvec)).as_matrix()
    bound = 1.25 * max(float(np.max(np.linalg.norm(translation, axis=1))), 4.25)
    features = pose_features(rotations, lifts, cameras[:, 2], MOMENTS)
    return dict(features=features, level=level, path=path, state=state, bound=bound, translation=translation)


def choose_rank(noises: list[dict]) -> int:
    """The smallest rank whose visible residual reaches the highest rank's, in rotation and in translation."""
    floor = noises[-1]
    for k, values in enumerate(noises, start=1):
        if (values["rotation_noise_deg_per_axis"] <= 1.1 * floor["rotation_noise_deg_per_axis"] + 0.25
                and values["translation_noise_angstrom_per_direction"] <= 1.1 * floor["translation_noise_angstrom_per_direction"] + 0.25):
            return k
    return len(noises)


def exploratory_rank(noises: list[dict]) -> int:
    """Rank 2 when a second component takes more than 5% off the rotation residual: a line's rotation residual is
    noise, which a second component barely reduces. A curve made of translation alone would not show here."""
    first, second = noises[0]["rotation_noise_deg_per_axis"], noises[1]["rotation_noise_deg_per_axis"]
    return 2 if second < 0.95 * first else 1


def same_state(order: np.ndarray, state: np.ndarray, target: int, kept: int) -> float:
    return float(np.mean(state[order[:kept]] == state[target]))


def targets(sim: dict, per_state: int = 4) -> list[int]:
    """The first particles of the central and one end state, and for a branch of the tip of the other path."""
    state = sim["state"]
    picks = [BINS // 2, BINS - 1, 0]
    if sim["path"].any():
        picks.append(2 * BINS - 1)
    return [int(i) for s in picks for i in np.flatnonzero(state == s)[:per_state]]


def fractions(sim: dict, motions: list, rank: int, second_rank: int) -> dict:
    state, features, bound = sim["state"], sim["features"], sim["bound"]
    out = {name: [] for name in ("bounded", "rotation only", "rank1", "rank2", "chosen rank", "filter then motion",
                                                    "exploratory rank", "filter then exploratory rank", "oracle")}
    oracle = np.stack([sim["level"], sim["path"] * 2.0 * (sim["level"] >= 0)], axis=1)
    for target in targets(sim):
        bounded = np.argsort(compatibility_distance(features, target, bound), kind="stable")
        chosen = low_rank_distance(motions[rank - 1], target)
        out["bounded"].append([same_state(bounded, state, target, n) for n in KEPT])
        out["rank1"].append([same_state(np.argsort(low_rank_distance(motions[0], target), kind="stable"), state, target, n) for n in KEPT])
        out["chosen rank"].append([same_state(np.argsort(chosen, kind="stable"), state, target, n) for n in KEPT])
        out["rotation only"].append([same_state(np.argsort(rotation_distance(features, target), kind="stable"), state, target, n) for n in KEPT])
        out["rank2"].append([same_state(np.argsort(low_rank_distance(motions[1], target), kind="stable"), state, target, n) for n in KEPT])
        second = low_rank_distance(motions[second_rank - 1], target)
        out["exploratory rank"].append([same_state(np.argsort(second, kind="stable"), state, target, n) for n in KEPT])

        def filter_then(distance):
            return [same_state(bounded[: 3 * n][np.argsort(distance[bounded[: 3 * n]], kind="stable")], state, target, n) for n in KEPT]

        out["filter then motion"].append(filter_then(chosen))
        out["filter then exploratory rank"].append(filter_then(second))
        truth = np.linalg.norm(oracle - oracle[target], axis=1)
        out["oracle"].append([same_state(np.argsort(truth, kind="stable"), state, target, n) for n in KEPT])
    return {name: np.mean(values, axis=0) for name, values in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    report = []
    for family in FAMILIES:
        for shift_noise, angle_noise in NOISE:
            ranks, second_ranks, noises_by_seed, tables = [], [], [], []
            for seed in range(args.seeds):
                sim = simulate(family, shift_noise, angle_noise, seed)
                motions = [fit_low_rank_motion(sim["features"], k) for k in range(1, MAX_RANK + 1)]
                noises = [motion_summary(sim["features"], m, MOMENTS.covariance) for m in motions]
                ranks.append(choose_rank(noises))
                second_ranks.append(exploratory_rank(noises))
                noises_by_seed.append([[n["rotation_noise_deg_per_axis"], n["translation_noise_angstrom_per_direction"]] for n in noises])
                tables.append(fractions(sim, motions, ranks[-1], second_ranks[-1]))
            mean = {name: np.mean([t[name] for t in tables], axis=0).round(3).tolist() for name in tables[0]}
            entry = dict(family=family, shift_noise=shift_noise, angle_noise=angle_noise, ranks=ranks, exploratory_ranks=second_ranks,
                         noise_by_rank_deg_angstrom=np.mean(noises_by_seed, axis=0).round(3).tolist(), fractions=mean)
            report.append(entry)
            print(json.dumps(entry), flush=True)
    args.json.write_text(json.dumps(dict(kept=KEPT, n=N, seeds=args.seeds, results=report), indent=1))


if __name__ == "__main__":
    main()
