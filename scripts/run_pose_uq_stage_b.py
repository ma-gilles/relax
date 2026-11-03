#!/usr/bin/env python
"""Stage B of the pose-UQ study: the projected PPCA on simulated poses with known correlated 5D errors. No images.

    python -m scripts.run_pose_uq_stage_b --out <dir>

One body, the moments and motion families of ``run_pose_motion_shapes`` (three lines, an arc, a two-path branch),
6000 particles, plus two view failures of the mixed line: views in a 30 degree cone (independent of state), and the
cone's axis turning with the state (state-dependent views, which the shared population prior assumes away). Each
particle gets a heteroscedastic, correlated local covariance ``S_i`` (rotation about 0.5 deg, shift about 0.3 A, a
lognormal spread, random correlations), one error draw from it through the exact transform, and ``R_i = J S_i J^T``.
States, views, targets and particle order are shared by every method of a setting. Truth is used for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from relax.multibody import pose_ppca as pp
from relax.multibody import pose_uq as uq
from relax.multibody.pose_neighborhoods import compatibility_distance, fit_low_rank_motion, low_rank_distance, pose_features, rotation_distance
from scripts.run_pose_motion_shapes import BINS, KEPT, MOMENTS, N, motion, same_state, targets

FAMILIES = ("rotation", "translation", "mixed", "arc", "branch", "mixed, view cone", "mixed, state-dependent views")
ROTATION_SD_DEG, SHIFT_SD = 0.5, 0.3
RANKS = (2, 4, 10)
FLOOR = 1e-3


def cameras_for(family: str, level: np.ndarray, rng) -> np.ndarray:
    n = len(level)
    if "view cone" in family or "state-dependent" in family:
        tilt = np.deg2rad(30.0) * np.sqrt(rng.uniform(0, 1, n))
        azimuth = rng.uniform(0, 2 * np.pi, n)
        axis = np.stack([np.cos(azimuth), np.sin(azimuth), np.zeros(n)], axis=1)
        cone = Rotation.from_rotvec(axis * tilt[:, None])
        if "state-dependent" in family:
            cone = Rotation.from_rotvec(np.outer(np.deg2rad(60.0) * level, [1.0, 0.0, 0.0])) * cone
        return (Rotation.from_rotvec(np.outer(rng.uniform(0, 2 * np.pi, n), [0, 0, 1.0])) * cone.inv()).as_matrix()
    return Rotation.random(n, random_state=rng).as_matrix()


def local_covariances(rng, n: int) -> np.ndarray:
    scale = np.exp(rng.normal(0.0, 0.4, n))
    sd = np.concatenate([np.full(3, np.deg2rad(ROTATION_SD_DEG)), np.full(2, SHIFT_SD)])
    mixing = np.eye(5) + 0.3 * rng.standard_normal((n, 5, 5))
    correlation = mixing @ np.swapaxes(mixing, 1, 2)
    d = np.sqrt(np.diagonal(correlation, axis1=1, axis2=2))
    correlation = correlation / d[:, :, None] / d[:, None, :]
    return correlation * (sd[None, :, None] * sd[None, None, :]) * (scale**2)[:, None, None]


def simulate(family: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    level = rng.uniform(-1.0, 1.0, N)
    shape = "branch" if family == "branch" else "arc" if family == "arc" else family.split(",")[0]
    path = rng.integers(0, 2, N) if shape == "branch" else np.zeros(N, dtype=np.int64)
    state = np.argmin(np.abs(level[:, None] - np.linspace(-1.0, 1.0, BINS)), axis=1) + BINS * (path * (level >= 0))
    rotvec, translation = motion(shape, level, path)
    Q = Rotation.from_rotvec(rotvec).as_matrix()
    cameras = cameras_for(family, level, rng)
    P = uq.camera_projection(cameras)
    S = local_covariances(rng, N)
    draw = np.einsum("nij,nj->ni", np.linalg.cholesky(S), rng.standard_normal((N, 5)))
    Q_obs = np.asarray(uq.exp_so3(draw[:, :3])) @ Q
    s_obs = np.einsum("nij,nj->ni", P, translation) + draw[:, 3:]
    root = uq.covariance_root(MOMENTS.covariance)
    y = uq.observed_features(Q_obs, s_obs, P, root, MOMENTS.centroid)
    R = uq.propagate(uq.feature_jacobian(Q_obs, P, root, MOMENTS.centroid), S)
    truth = np.concatenate([(Q @ root).reshape(N, 9), translation + Q @ MOMENTS.centroid], axis=1)
    lifts = np.einsum("nji,nj->ni", P, s_obs)
    features = pose_features(Q_obs, lifts, cameras[:, 2], MOMENTS)
    bound = 1.25 * max(float(np.max(np.linalg.norm(translation, axis=1))), 4.25)
    return dict(level=level, path=path, state=state, y=y, P=P, R=R, S=S, truth=truth, normals=cameras[:, 2], features=features, bound=bound)


def zero_depth_pca(sim, rank):
    lifted = np.concatenate([sim["y"][:, :9], np.einsum("nji,nj->ni", sim["P"], sim["y"][:, 9:])], axis=1)
    centred = lifted - lifted.mean(axis=0)
    return centred @ np.linalg.svd(centred, full_matrices=False)[2][:rank].T


def r2(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(1.0 - np.sum((prediction - truth) ** 2) / np.sum((truth - truth.mean(axis=0)) ** 2))


def evaluate(sim: dict, coordinates: dict) -> dict:
    out = {}
    for name, value in coordinates.items():
        rows = []
        for target in targets(sim):
            d = value(target) if callable(value) else np.linalg.norm(value - value[target], axis=1)
            order = np.argsort(d, kind="stable")
            rows.append([same_state(order, sim["state"], target, k) for k in KEPT])
        out[name] = np.mean(rows, axis=0).round(3).tolist()
    return out


def setting(family: str, seed: int) -> dict:
    sim = simulate(family, seed)
    y, P, R = sim["y"], sim["P"], sim["R"]
    started = time.time()
    coordinates, report = {}, dict(family=family, seed=seed, fits={})
    depth_truth = np.sum(sim["truth"][:, 9:] * sim["normals"], axis=1)
    for label, scale in (("known R", 1.0), ("R x 0.25", 0.25), ("R x 4", 4.0), ("R = 0", 0.0)):
        for rank in RANKS:
            if label != "known R" and rank != 4:
                continue
            fit = pp.fit_projected_ppca(y, P, scale * R, rank, floor=FLOOR, tolerance=1e-6)
            post = pp.latent_posterior(y, P, scale * R, fit)
            name = f"PPCA q{rank}, {label}"
            coordinates[name] = post.principal(fit)[0]
            depth = np.sum(post.feature_mean[:, 9:] * sim["normals"], axis=1)
            standard = (post.feature_mean - sim["truth"])
            report["fits"][name] = dict(tau2=fit.tau2, iterations=len(fit.log_likelihood), amplitudes=fit.amplitudes.round(3).tolist(),
                                        feature_r2=r2(post.feature_mean, sim["truth"]), depth_r2=r2(depth, depth_truth),
                                        rms_feature_error=float(np.sqrt(np.mean(np.sum(standard**2, axis=1)))))
    for rank in RANKS:
        coordinates[f"zero-depth PCA {rank}"] = zero_depth_pca(sim, rank)
    features = sim["features"]
    rank1, rank2 = fit_low_rank_motion(features, 1), fit_low_rank_motion(features, 2)
    coordinates["rank 1 (existing)"] = lambda t: low_rank_distance(rank1, t)
    coordinates["rank 2 (existing)"] = lambda t: low_rank_distance(rank2, t)
    coordinates["rotation only (existing)"] = lambda t: rotation_distance(features, t)
    coordinates["bounded (existing)"] = lambda t: compatibility_distance(features, t, sim["bound"])
    oracle = np.stack([sim["level"], sim["path"] * 2.0 * (sim["level"] >= 0)], axis=1)
    coordinates["oracle"] = oracle
    report["same_state"] = evaluate(sim, coordinates)
    report["seconds"] = time.time() - started
    return report, sim, coordinates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results, arrays = [], {}
    for family in FAMILIES:
        for seed in range(args.seeds):
            report, sim, coordinates = setting(family, seed)
            results.append(report)
            print(json.dumps({k: report[k] for k in ("family", "same_state")}), flush=True)
            if seed == 0:
                key = family.replace(",", "").replace(" ", "_")
                arrays[f"{key}_level"], arrays[f"{key}_state"] = sim["level"], sim["state"]
                arrays[f"{key}_ppca"], arrays[f"{key}_pca"] = coordinates["PPCA q4, known R"], coordinates["zero-depth PCA 4"]
    (args.out / "stage_b.json").write_text(json.dumps(dict(kept=KEPT, n=N, floor=FLOOR, results=results), indent=1))
    np.savez_compressed(args.out / "stage_b_arrays.npz", **arrays)


if __name__ == "__main__":
    main()
