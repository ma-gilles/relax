#!/usr/bin/env python
"""How well does the projected PPCA recover each particle's unobserved depth? Body 2, cases A-D, from the 5k arrays.

    python -m scripts.run_pose_uq_depth --results <pose_uq dir>/full --root <pose_neighborhoods> --out <dir>

No new fits: the complete-feature posterior mean ``b_i = mu + Sigma A_i^T T_i^-1 (y_i - A_i mu)`` of the fitted models,
for the per-body model (rank 10, chosen floor), the joint model (``joint_feature_mean``) and the relative model.
Their translation part is a 3D position: body 2's centroid (per body, joint) or body 2's centroid relative to body 1's
moved pivot (relative, turned back to the molecular frame by the fitted ``Q1``). Its depth is ``n_i . b``, with
``n_i`` the fitted consensus viewing direction.

Truth (evaluation only): the simulator's state motions and the true cameras. The simulation frame is carried into
the run's frame by the rotation ``G`` that best maps the true cameras onto the consensus cameras (``A_fit ~ A_true G``,
Procrustes over all particles) and a constant translation fitted on the OBSERVED in-plane coordinates alone, so no
predicted depth enters the alignment. True depth is then ``n_i . (G^T X_i + c)``. The consensus's per-particle error
stays in ``n_i``, so true and predicted depth are along the same, fitted, direction.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from relax.multibody import pose_ppca as pp


def projected(n: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sum(n * x, axis=1)


def summary(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Both are deviations from their own population mean position, so a view-dependent constant (the relative arm's
    fixed body-1-to-body-2 offset seen through each camera) is removed; the zero-depth guess predicts no deviation."""
    error = predicted - truth
    return dict(n=int(len(truth)), correlation=float(np.corrcoef(predicted, truth)[0, 1]), rmse=float(np.sqrt(np.mean(error**2))),
                robust_rmse=float(1.4826 * np.median(np.abs(error - np.median(error)))), true_sd=float(np.std(truth)),
                zero_guess_rmse=float(np.sqrt(np.mean(truth**2))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    from relax.multibody.bodies import read_bodies
    from relax.multibody.pose_neighborhoods import mask_moments
    from scripts.multibody_adaptive_subsets import true_body_motions

    figure = {}
    for case in "ABCD":
        name = f"case{case}_5k"
        data = args.root / name
        embed = np.load(args.results / f"embed_{name}.npz", allow_pickle=True)
        relative = np.load(args.results / f"relative_{name}.npz")
        record = json.loads((data / "body_motion.json").read_text())
        names = embed["image_names"]
        rows = np.asarray([int(s.split("@")[0]) - 1 for s in names])
        with open(data / "simulation_info.pkl", "rb") as handle:
            states = np.asarray(pickle.load(handle)["image_assignment"])[rows]
        with open(data / "poses.pkl", "rb") as handle:
            cameras = np.asarray(pickle.load(handle)[0])[rows]
        pixel = 4.25
        bodies = read_bodies(data / "bodies.star", pixel_size=pixel)
        rotations, translations, pivot = true_body_motions(record, len(bodies))
        P = embed["P"]
        normal = np.cross(P[:, 0], P[:, 1])
        consensus = np.concatenate([P, normal[:, None]], axis=1)
        u, _, vt = np.linalg.svd(np.einsum("nji,njk->ik", cameras, consensus))
        G = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt  # A_fit ~ A_true G
        moments = [mask_moments(b.mask, b.com, pixel) for b in bodies]
        centre = [b.com * pixel for b in bodies]

        def true_position(b):
            """The simulation-frame position of body b's centroid, and of its moved pivot."""
            M = rotations[states, b]
            moved_pivot = np.einsum("nij,j->ni", M, centre[b] - pivot) + pivot + translations[states, b]
            return moved_pivot + np.einsum("nij,j->ni", M, moments[b].centroid), moved_pivot

        body2, _ = true_position(1)
        _, pivot1 = true_position(0)
        truths = {"per-body": body2, "joint": body2, "relative": body2 - pivot1}

        # Predicted complete-feature positions, molecular frame.
        y, R = embed["y"], embed["R"]
        usable2 = (embed["body2_status"] == 0) & np.all(np.isfinite(R[:, 11:, 11:].reshape(len(R), -1)), axis=1)
        fit = pp.PPCAFit(embed["body2_mean"], embed["body2_W"], float(embed["body2_tau2"]), 1e-6, np.zeros(1), True)
        per_body = np.full((len(y), 3), np.nan)
        per_body[usable2] = pp.latent_posterior(y[usable2, 11:], P[usable2], R[usable2, 11:, 11:], fit).feature_mean[:, 9:]
        joint = np.full((len(y), 3), np.nan)
        joint[embed["joint_usable"]] = embed["joint_feature_mean"][:, 21:24]
        rel_fit = pp.PPCAFit(relative["mean"], relative["W"], float(relative["tau2"]), 1e-6, np.zeros(1), True)
        rel_usable = relative["usable"]
        local = pp.latent_posterior(relative["y"][rel_usable], relative["P"][rel_usable], relative["R"][rel_usable], rel_fit).feature_mean[:, 9:]
        # Back to the molecular frame: P_rel = P Q1, so Q1 = P^+ P_rel on the image rows; the depth row is n Q1.
        Q1 = np.linalg.solve(consensus[rel_usable], np.concatenate([relative["P"][rel_usable], np.cross(relative["P"][rel_usable, 0], relative["P"][rel_usable, 1])[:, None]], axis=1))
        rel = np.full((len(y), 3), np.nan)
        rel[rel_usable] = np.einsum("nij,nj->ni", Q1, local)
        predictions = {"per-body": per_body, "joint": joint, "relative": rel}
        observed = {"per-body": y[:, 20:22], "joint": y[:, 20:22], "relative": relative["y"][:, 9:11]}

        report = dict(case=case, alignment=dict(G=G.tolist()), arms={})
        for arm, predicted in predictions.items():
            keep = np.all(np.isfinite(predicted), axis=1)
            truth = truths[arm] @ G  # G^T X, row form
            # The constant offset from the observed in-plane coordinates alone (least squares over all particles).
            design = P[keep].reshape(-1, 3)
            offset = np.linalg.lstsq(design, (observed[arm][keep] - np.einsum("nij,nj->ni", P[keep], truth[keep])).reshape(-1), rcond=None)[0]
            truth = truth + offset
            truth_dev = truth[keep] - truth[keep].mean(axis=0)
            pred_dev = predicted[keep] - predicted[keep].mean(axis=0)
            depth_true, depth_pred = projected(normal[keep], truth_dev), projected(normal[keep], pred_dev)
            plane_true = np.einsum("nij,nj->ni", P[keep], truth_dev)
            plane_pred = np.einsum("nij,nj->ni", P[keep], pred_dev)
            # Edge-on-ness: the view direction against body 2's main motion direction (truth, its first principal axis).
            centred = truth[keep] - truth[keep].mean(axis=0)
            direction = np.linalg.svd(centred, full_matrices=False)[2][0]
            alignment = np.abs(normal[keep] @ direction)
            thirds = np.quantile(alignment, [1 / 3, 2 / 3])
            split = {}
            for label, mask in (("motion most in-plane (low |n.u|)", alignment <= thirds[0]), ("middle", (alignment > thirds[0]) & (alignment <= thirds[1])),
                                ("motion most along view (high |n.u|)", alignment > thirds[1])):
                split[label] = dict(depth=summary(depth_pred[mask], depth_true[mask]), abs_n_dot_u_range=[float(alignment[mask].min()), float(alignment[mask].max())])
            report["arms"][arm] = dict(offset_angstrom=offset.tolist(), motion_direction=direction.tolist(), depth=summary(depth_pred, depth_true),
                                       in_plane_x=summary(plane_pred[:, 0], plane_true[:, 0]), in_plane_y=summary(plane_pred[:, 1], plane_true[:, 1]), by_view=split)
            if arm == "relative":
                figure[case] = (depth_true, depth_pred, alignment)
        (args.out / f"depth_{name}.json").write_text(json.dumps(report, indent=1))
        print(case, json.dumps({a: {k: {kk: round(vv, 3) for kk, vv in v.items() if kk != "n"} for k, v in r.items() if k in ("depth", "in_plane_x")} for a, r in report["arms"].items()}), flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d3a73"])
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for i, case in enumerate("ABCD"):
        t, p, a = figure[case]
        sc = ax[i].scatter(t, p, c=a, cmap=cmap, s=3, linewidths=0, vmin=0, vmax=1)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax[i].plot(lim, lim, color="#52514e", ls="--", lw=1)
        r = json.loads((args.out / f"depth_case{case}_5k.json").read_text())["arms"]["relative"]["depth"]
        ax[i].set(title=f"case {case}: r = {r['correlation']:.2f}, RMSE {r['rmse']:.2f} A\ntrue sd {r['true_sd']:.2f} A", xlabel="true depth (A, truth)", ylabel="predicted depth n . b (A)")
        ax[i].grid(color="#e4e3df", lw=0.6)
        fig.colorbar(sc, ax=ax[i], shrink=0.8, label="|n . motion direction|")
    fig.suptitle("Relative-pose PPCA: body 2's predicted depth (posterior-mean position along the view) against the truth", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "fig6_depth_relative.png", dpi=150)


if __name__ == "__main__":
    main()
