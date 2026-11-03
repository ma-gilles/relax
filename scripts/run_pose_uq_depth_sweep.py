#!/usr/bin/env python
"""Depth recovery of projected PPCA against rank, joint vs per-body, true vs fitted poses. CPU EM, no images.

    python -m scripts.run_pose_uq_depth_sweep --results <pose_uq dir>/full --root <pose_neighborhoods> --out <dir> <case>

For q = 1, 2, 3, 4, 6, 10: the per-body model of body 2 and the joint model (bodies 1 and 2), fitted on
- fitted poses: the 5k tables of the embed stage (y, P, R from the raw Hessian), at the floor the 5k run chose for that
  arm and rank (q = 1 and 3 take the choice of q = 2 and 4, the next swept rank), and
- true poses: the simulator's state motions seen through the true cameras, R = 0, floor 1e-4 (the 5k true-pose control).
Depth = n_i . (posterior-mean position - its mean), as in ``run_pose_uq_depth``. Truth, the alignment of the
simulation frame into the run's (fitted poses only) and the edge-on thirds are as there; the true-pose fits live in
the simulation frame, so they need no alignment. The relative arm at q = 10 is read from that script's output.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from relax.multibody import pose_ppca as pp
from scripts.run_pose_uq_depth import projected, summary

RANKS = (1, 2, 3, 4, 6, 10)
TRUE_FLOOR = 1e-4


def floor_for(choices: dict, rank: int) -> float:
    swept = sorted(int(k) for k in choices)
    return float(choices[str(next(r for r in swept if r >= rank))])


def scores(predicted, truth, normal, P, direction) -> dict:
    keep = np.all(np.isfinite(predicted), axis=1)
    truth_dev = truth[keep] - truth[keep].mean(axis=0)
    pred_dev = predicted[keep] - predicted[keep].mean(axis=0)
    depth_true, depth_pred = projected(normal[keep], truth_dev), projected(normal[keep], pred_dev)
    plane_true = np.einsum("nij,nj->ni", P[keep], truth_dev)
    plane_pred = np.einsum("nij,nj->ni", P[keep], pred_dev)
    alignment = np.abs(normal[keep] @ direction)
    edges = np.quantile(alignment, [1 / 3, 2 / 3])
    thirds = {}
    for label, mask in (("low |n.u| (motion in plane)", alignment <= edges[0]), ("middle", (alignment > edges[0]) & (alignment <= edges[1])),
                        ("high |n.u| (motion along view)", alignment > edges[1])):
        thirds[label] = summary(depth_pred[mask], depth_true[mask])
    plane = np.sqrt(np.mean(np.sum((plane_pred - plane_true) ** 2, axis=1)) / 2.0)
    return dict(depth=summary(depth_pred, depth_true), in_plane_rmse_per_direction=float(plane),
                in_plane_true_sd_per_direction=float(np.sqrt(np.mean(np.sum(plane_true**2, axis=1)) / 2.0)), thirds=thirds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("case")
    args = parser.parse_args()
    from relax.multibody import pose_uq as uq
    from relax.multibody.bodies import read_bodies
    from relax.multibody.pose_neighborhoods import mask_moments
    from scripts.multibody_adaptive_subsets import true_body_motions

    name = f"case{args.case}_5k"
    data = args.root / name
    embed = np.load(args.results / f"embed_{name}.npz", allow_pickle=True)
    choices = json.loads((args.results / f"embed_{name}.json").read_text())
    record = json.loads((data / "body_motion.json").read_text())
    rows = np.asarray([int(s.split("@")[0]) - 1 for s in embed["image_names"]])
    with open(data / "simulation_info.pkl", "rb") as handle:
        states = np.asarray(pickle.load(handle)["image_assignment"])[rows]
    with open(data / "poses.pkl", "rb") as handle:
        cameras = np.asarray(pickle.load(handle)[0])[rows]
    pixel = 4.25
    bodies = read_bodies(data / "bodies.star", pixel_size=pixel)
    rotations, translations, pivot = true_body_motions(record, len(bodies))
    moments = [mask_moments(b.mask, b.com, pixel) for b in bodies]
    roots = [uq.covariance_root(m.covariance) for m in moments]

    # Truth: each body's centroid position (simulation frame) and its rotation.
    centroids, true_Q, true_t = [], [], []
    for b, body in enumerate(bodies):
        M = rotations[states, b]
        rest = body.com * pixel
        moved = np.einsum("nij,j->ni", M, rest - pivot) + pivot + translations[states, b]
        centroids.append(moved + np.einsum("nij,j->ni", M, moments[b].centroid))
        true_Q.append(M)
        true_t.append(moved - rest)
    true_P = uq.camera_projection(cameras)
    true_normal = cameras[:, 2, :]
    true_y = np.concatenate([np.concatenate([(true_Q[b] @ roots[b]).reshape(-1, 9),
                                             np.einsum("nij,nj->ni", true_P, true_t[b] + np.einsum("nij,j->ni", true_Q[b], moments[b].centroid))], axis=1)
                             for b in range(2)], axis=1)

    # Fitted: the embed stage's table, and the simulation frame carried into the run's frame.
    P, y, R = embed["P"], embed["y"], embed["R"]
    normal = np.cross(P[:, 0], P[:, 1])
    consensus = np.concatenate([P, normal[:, None]], axis=1)
    u, _, vt = np.linalg.svd(np.einsum("nji,njk->ik", cameras, consensus))
    G = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    usable_body = [(embed[f"body{b + 1}_status"] == 0) & np.all(np.isfinite(R[:, 11 * b : 11 * b + 11, 11 * b : 11 * b + 11].reshape(len(R), -1)), axis=1) for b in range(2)]
    usable_joint = usable_body[0] & usable_body[1]

    def aligned(b, keep):
        """True centroid of body b in the run's frame: G, then a constant fitted on the observed in-plane coordinates."""
        truth = centroids[b] @ G
        observed = y[:, 11 * b + 9 : 11 * b + 11]
        offset = np.linalg.lstsq(P[keep].reshape(-1, 3), (observed[keep] - np.einsum("nij,nj->ni", P[keep], truth[keep])).reshape(-1), rcond=None)[0]
        return truth + offset

    def direction(points):
        centred = points - points.mean(axis=0)
        return np.linalg.svd(centred, full_matrices=False)[2][0]

    report = dict(case=args.case, ranks=list(RANKS), true_floor=TRUE_FLOOR, rows=[])
    zero = lambda n, d: np.zeros((n, d, d))
    for poses in ("true", "fitted"):
        for arm in ("per-body", "joint"):
            for rank in RANKS:
                started = time.time()
                if poses == "true":
                    keep = np.ones(len(true_y), dtype=bool)
                    floor = TRUE_FLOOR
                    obs_y = true_y if arm == "joint" else true_y[:, 11:]
                    obs_R = zero(len(true_y), obs_y.shape[1])
                    obs_P = true_P
                else:
                    keep = usable_joint if arm == "joint" else usable_body[1]
                    source = choices["joint"]["best_floor"] if arm == "joint" else choices["per_body"]["2"]["best_floor"]
                    floor = floor_for(source, rank)
                    obs_y = y if arm == "joint" else y[:, 11:]
                    obs_R = R if arm == "joint" else R[:, 11:, 11:]
                    obs_P = P
                fit = pp.fit_projected_ppca(obs_y[keep], obs_P[keep], obs_R[keep], rank, floor=floor, tolerance=1e-6, max_iterations=3000)
                mean = np.full((len(keep), len(fit.mean)), np.nan)
                mean[keep] = pp.latent_posterior(obs_y[keep], obs_P[keep], obs_R[keep], fit).feature_mean
                bodies_scored = (0, 1) if arm == "joint" else (1,)
                for b in bodies_scored:
                    column = 12 * b + 9 if arm == "joint" else 9
                    predicted = mean[:, column : column + 3]
                    if poses == "true":
                        truth, view, proj = centroids[b], true_normal, true_P
                    else:
                        truth, view, proj = aligned(b, keep), normal, P
                    entry = dict(poses=poses, arm=arm, body=b + 1, rank=rank, floor=floor, tau2=fit.tau2, iterations=len(fit.log_likelihood),
                                 converged=fit.converged, seconds=time.time() - started,
                                 **scores(predicted, truth, view, proj, direction(truth[keep])))
                    report["rows"].append(entry)
                    print(poses, arm, b + 1, rank, floor, round(entry["depth"]["correlation"], 3), round(entry["depth"]["rmse"], 3),
                          round(entry["depth"]["true_sd"], 3), entry["iterations"], round(entry["seconds"], 1), flush=True)
    relative = args.results / f"depth_{name}.json"
    if relative.exists():
        report["relative_q10_fitted"] = json.loads(relative.read_text())["arms"]["relative"]
    (args.out / f"depth_sweep_{name}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
