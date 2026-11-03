#!/usr/bin/env python
"""Figures of the pose-UQ and projected-PPCA study, from the arrays ``run_pose_uq_ppca`` wrote. CPU, no GPU.

    python -m scripts.plot_pose_uq_ppca --results <pose_uq dir> --root <pose_neighborhoods> --out <figures dir>

1 likelihood audit, 2 UQ calibration, 3 PPCA diagnostics, 4 embeddings (latent scatters in PC coordinates and the raw
PC directions with their scale), 5 neighbourhood curves against the perfect ordering. Every coordinate is named on its
axis: "principal coordinate" is amplitude times standardized z, angstroms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d3a73"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("orange", ["#fde0d2", "#f5a47f", "#eb6834", "#b5461b", "#7a2c0d"])
CASES = "ABCD"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "legend.frameon": False, "lines.linewidth": 2.0})


def load(path: Path):
    return json.loads(path.read_text())


def audit(results: Path, out: Path) -> None:
    d = np.load(results / "C_small/audit_caseC_5k_body2.npz")
    r = load(results / "C_small/audit_caseC_5k_body2.json")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    rel = d["native"] / d["continuous"] - 1.0
    ax[0].plot(d["continuous"], rel, "o", color=BLUE, ms=4)
    ax[0].set(xlabel="continuous float64 score", ylabel="native float32 / continuous - 1",
              title=f"native fine diff2 vs continuous score\nmax |rel| {np.abs(rel).max():.1e}, 64 particles")
    steps, H, g = d["steps"], d["hessians"], d["gradients"]
    ref = list(steps).index(1.0)
    dev = np.abs(H - H[ref]).max(axis=(2, 3))
    gdev = np.abs(g - g[ref]).max(axis=2)
    for values, colour, label in ((dev, BLUE, "Hessian (whitened)"), (gdev, ORANGE, "gradient (whitened)")):
        med = np.median(values, axis=1)
        lo, hi = np.quantile(values, [0.1, 0.9], axis=1)
        ax[1].plot(steps, med, "o-", color=colour, label=label, ms=5)
        ax[1].fill_between(steps, lo, hi, color=colour, alpha=0.15, lw=0)
    ax[1].set(xscale="log", yscale="symlog", xlabel="central-difference step (expected-Fisher sd)", ylabel="max |entry - entry at 1 sd|",
              title="finite-difference convergence\nmedian and 10-90% of particles")
    ax[1].legend()
    grid, sections = d["grid"], d["sections"]
    rise = sections - sections[:, list(grid).index(0)][:, None]
    ax[2].plot(grid, 0.5 * grid**2, color=MUTED, ls="--", lw=1.5, label="local quadratic 0.5 u^2 (F frame)")
    ax[2].plot(grid, np.median(rise, axis=0), "o-", color=BLUE, label="score, median over particles", ms=5)
    lo, hi = np.quantile(rise, [0.1, 0.9], axis=0)
    ax[2].fill_between(grid, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax[2].set(xlabel="distance along a random whitened direction (sd)", ylabel="score rise", title="likelihood sections vs local quadratic")
    ax[2].legend()
    fig.suptitle(f"Figure 1. Likelihood and derivative audit, case C body 2, box {r['box']}, pixel {r['pixel_size']} A, {r['window_pixels']} Fourier pixels", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig1_audit.png", dpi=150)


def calibration(results: Path, out: Path) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    rows = []
    for c in CASES:
        for b in (1, 2):
            r = load(results / f"full/uq_case{c}_5k_body{b}_estimated.json")
            rows.append((f"{c}{b}", np.mean(r["error_analysis"]["observed_hessian_qualified"]["z_robust_sd"])))
    sandwich = {}
    for name, path in (("C2", "C_small/sandwich_caseC_5k_body2.json"), ("A1", "full/sandwich_caseA_5k_body1.json"), ("A2", "full/sandwich_caseA_5k_body2.json")):
        s = load(results / path)
        sandwich[name] = (np.mean(s["raw_hessian"]["z_robust_sd"]), np.mean(s["sandwich"]["z_robust_sd"]))
    synthetic = load(results / "C_small/synthetic_caseC_5k_body2_signal1.json")
    component = load(results / "C_small/synthetic_caseC_5k_body2_component.json")
    x = np.arange(len(rows))
    ax[0].bar(x, [v for _, v in rows], color=BLUE, width=0.6, label="raw Hessian, real images (5k)")
    for name, (raw, corrected) in sandwich.items():
        i = [k for k, _ in rows].index(name)
        ax[0].plot([i], [corrected], "D", color=ORANGE, ms=8, label="mask-corrected (sandwich), 300 particles" if name == "C2" else None)
    ax[0].axhline(1.0, color=INK, lw=1)
    ax[0].axhline(np.mean(synthetic["observed_hessian"]["z_robust_sd"]), color=AQUA, lw=1.5, ls="--", label="perfect-model synthetic")
    ax[0].axhline(np.mean(component["observed_hessian"]["z_robust_sd"]), color=VIOLET, lw=1.5, ls=":", label="true body + Gaussian noise")
    ax[0].set_xticks(x, [k for k, _ in rows])
    ax[0].set(xlabel="case and body", ylabel="robust sd of error / predicted sd", title="calibration of the image-level pose\n(1 = calibrated)")
    ax[0].set_ylim(0, 2.9)
    ax[0].legend(fontsize=7, loc="upper left", ncol=2)
    m = np.load(results / "full/uq_caseC_5k_body2_estimated.npz")
    error = m["error_analysis"]
    error = error - np.median(error, axis=0)
    predicted = np.sqrt(np.diagonal(np.linalg.inv(m["hessian"]), axis1=1, axis2=2))
    for k, colour, label in ((0, BLUE, "rotation x"), (3, ORANGE, "shift x")):
        scale = np.rad2deg(1) if k < 3 else 1.0
        p, e = predicted[:, k] * scale, np.abs(error[:, k]) * scale
        bins = np.quantile(p, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        centre = [np.median(p[idx == i]) for i in range(10)]
        observed = [1.4826 * np.median(e[idx == i]) for i in range(10)]
        ax[1].plot(centre, observed, "o-", color=colour, label=f"{label} ({'deg' if k < 3 else 'A'})", ms=5)
    lim = ax[1].get_xlim()
    ax[1].plot(lim, lim, color=MUTED, ls="--", lw=1, label="calibrated")
    ax[1].plot(lim, 2 * np.asarray(lim), color=MUTED, ls=":", lw=1, label="2x")
    ax[1].set(xlabel="predicted sd (raw Hessian), decile median", ylabel="observed robust |error|", title="error vs predicted uncertainty\ncase C body 2, 5k, gauge removed")
    ax[1].legend(fontsize=7)
    levels = np.array([0.5, 0.9, 0.95, 0.99])
    for name, path, colour in (("C2", "C_small/sandwich_caseC_5k_body2.json", BLUE), ("A1", "full/sandwich_caseA_5k_body1.json", AQUA), ("A2", "full/sandwich_caseA_5k_body2.json", VIOLET)):
        s = load(results / path)
        ax[2].plot(levels, [s["raw_hessian"]["ellipsoid_coverage"][str(v)] for v in levels], "o--", color=colour, ms=4, lw=1.2, label=f"{name} raw")
        ax[2].plot(levels, [s["sandwich"]["ellipsoid_coverage"][str(v)] for v in levels], "s-", color=colour, ms=4, label=f"{name} mask-corrected")
    ax[2].plot(levels, [synthetic["observed_hessian"]["ellipsoid_coverage"][str(v)] for v in levels], "^-", color=ORANGE, label="perfect-model synthetic")
    ax[2].plot([0.5, 1], [0.5, 1], color=MUTED, lw=1)
    ax[2].set(xlabel="nominal coverage (5D ellipsoid)", ylabel="observed coverage", title="ellipsoid coverage")
    ax[2].legend(fontsize=7)
    tangent, normal, bias = [], [], []
    for c in CASES:
        e = load(results / f"full/embed_case{c}_5k.json")
        for b in ("1", "2"):
            f = e["per_body"][b]["finite_angle"]
            tangent.append(f["tangent_variance_ratio_median"])
            normal.append(f["normal_variance_median"] / f["tangent_variance_median"])
            bias.append(f["bias_norm_median"])
    labels = [f"{c}{b}" for c in CASES for b in (1, 2)]
    ax[3].bar(np.arange(8) - 0.2, tangent, width=0.4, color=BLUE, label="tangent variance, finite / Jacobian")
    ax[3].bar(np.arange(8) + 0.2, np.asarray(normal) * 1e4, width=0.4, color=ORANGE, label="normal / tangent variance x 1e4")
    ax[3].set_xticks(np.arange(8), labels)
    ax[3].set(xlabel="case and body", title=f"feature covariance: Jacobian vs finite angles\nmedian mean bias {np.median(bias):.4f} A")
    ax[3].legend(fontsize=7)
    fig.suptitle("Figure 2. UQ calibration against the simulated truth (image-level camera pose; no consensus error enters)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_calibration.png", dpi=150)

    # Consensus error: molecular-frame error by body-1 twist.
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for i, c in enumerate(CASES):
        t = np.load(results / f"full/twist_case{c}_5k.npz")
        r = load(results / f"full/twist_case{c}_5k.json")
        for b, colour in ((1, BLUE), (2, ORANGE)):
            usable = t[f"body{b}_usable"]
            twist = t["body1_twist"][usable]
            err = t[f"body{b}_error_deg"][:, 1]
            jitter = np.random.default_rng(0).normal(0, 0.35, len(twist)) + (b - 1.5) * 1.2
            ax[i].plot(twist + jitter, err, ".", color=colour, ms=1.5, alpha=0.4, label=f"body {b}, residual sd {r[str(b)]['robust_residual_deg'][1]:.2f} deg")
        ax[i].set(title=f"case {c}", xlabel="true body-1 twist (deg), jittered")
        ax[i].legend(fontsize=7, markerscale=6)
    ax[0].set_ylabel("molecular-frame error, axis 2 (deg)")
    fig.suptitle("Figure 2b. Molecular-frame rotation error is the consensus following body 1's twist (r = -0.99); not in any conditional UQ", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig2b_consensus.png", dpi=150)


def ppca_diagnostics(results: Path, out: Path) -> None:
    fig, ax = plt.subplots(2, 4, figsize=(17, 7))
    for i, c in enumerate(CASES):
        e = load(results / f"full/embed_case{c}_5k.json")
        a = np.load(results / f"full/embed_case{c}_5k.npz")
        for b, colour in (("1", BLUE), ("2", ORANGE)):
            sweep = e["per_body"][b]["sweep"]
            ranks = sorted({v["rank"] for v in sweep.values()})
            for floor, style in ((1e-3, "o-"), (1e-2, "s--"), (1e-1, "^:")):
                ax[0, i].plot(ranks, [np.mean([x["robust_rmse"] for k, x in sweep[f"q{q}_floor{floor:g}"]["validation"].items() if "shift" in k]) for q in ranks],
                              style, color=colour, ms=4, lw=1.3, label=f"body {b}, floor {floor:g} A^2")
            ax[0, i].axhline(np.mean([x["mean_only_robust_rmse"] for k, x in sweep["q10_floor0.01"]["validation"].items() if "shift" in k]), color=colour, lw=1, alpha=0.6)
        joint = e["joint"]["sweep"]
        ranks = sorted({v["rank"] for v in joint.values()})
        ax[0, i].plot(ranks, [np.mean([x["robust_rmse"] for k, x in joint[f"q{q}_floor{e['joint']['best_floor'][str(q)]:g}"]["validation"].items() if "shift" in k]) for q in ranks],
                      "D-", color=AQUA, ms=5, label="joint, chosen floor")
        ax[0, i].set(title=f"case {c}: withheld-shift prediction", xlabel="rank q", ylabel="validation robust RMSE (A)")
        if i == 0:
            ax[0, i].legend(fontsize=6)
        amplitudes = [("joint", a["joint_amplitudes"], AQUA)] + [(f"body {b}", np.linalg.norm(a[f"body{b}_W"], axis=0), colour) for b, colour in ((1, BLUE), (2, ORANGE))]
        for name, values, colour in amplitudes:
            ax[1, i].plot(np.arange(1, len(values) + 1), values, "o-", color=colour, ms=4, label=name)
        tau = float(a["joint_tau2"])
        ax[1, i].axhline(np.sqrt(tau), color=MUTED, ls="--", lw=1, label=f"joint tau = {np.sqrt(tau):.2f} A (floor-set)")
        ax[1, i].set(yscale="log", title=f"case {c}: covariance spectrum (q = 10)", xlabel="component", ylabel="amplitude ||w_k|| (A)")
        ax[1, i].legend(fontsize=6)
    fig.suptitle("Figure 3. PPCA diagnostics: withheld-shift prediction against rank and floor (flat lines: mean only); amplitude spectra", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig3_ppca.png", dpi=150)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for i, c in enumerate(CASES):
        e = load(results / f"full/embed_case{c}_5k.json")
        for b, colour in (("1", BLUE), ("2", ORANGE)):
            sweep = e["per_body"][b]["sweep"]
            floors = sorted({v["floor"] for v in sweep.values()})
            ax[i].plot(floors, [sweep[f"q10_floor{f:g}"]["validation"]["log_likelihood_per_particle"] for f in floors], "o-", color=colour, label=f"body {b}")
        ax[i].set(xscale="log", xlabel="tau^2 floor (A^2)", ylabel="validation log likelihood / particle", title=f"case {c}, q = 10")
        ax[i].legend()
    fig.suptitle("Figure 3b. The likelihood rises without limit as the floor falls (rotation features lie on SO(3)): tau is set by the floor, chosen by withheld shifts", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig3b_floor.png", dpi=150)


def direction_parts(column: np.ndarray, mean: np.ndarray, root: np.ndarray):
    """A 12-feature PC direction as a rotation-vector rate (deg per amplitude unit) and a translation (A)."""
    Q0 = mean[:9].reshape(3, 3) @ np.linalg.inv(root)
    u, _, vt = np.linalg.svd(Q0)
    Q0 = u @ vt
    dQ = column[:9].reshape(3, 3) @ np.linalg.inv(root)
    skew = dQ @ Q0.T
    omega = 0.5 * np.array([skew[2, 1] - skew[1, 2], skew[0, 2] - skew[2, 0], skew[1, 0] - skew[0, 1]])
    return np.rad2deg(omega), column[9:]


def embeddings(results: Path, root: Path, out: Path) -> None:
    from relax.multibody.bodies import read_bodies
    from relax.multibody.pose_neighborhoods import mask_moments

    for c in CASES:
        a = np.load(results / f"full/embed_case{c}_5k.npz")
        usable = a["joint_usable"]
        principal = a["joint_principal"]
        cov = a["joint_principal_covariance"]
        level, twist = a["level"], a["body1_twist"]
        P = a["P"]
        normal = np.cross(P[:, 0], P[:, 1])
        tilt = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))
        uncertainty = np.full(len(usable), np.nan)
        uncertainty[usable] = np.sqrt(np.trace(cov[:, :2, :2], axis1=1, axis2=2))
        fig, ax = plt.subplots(2, 4, figsize=(17, 8))
        colourings = (("body-2 state level (truth)", level, SEQ_BLUE), ("body-1 twist, deg (truth)", twist, SEQ_ORANGE),
                      ("view tilt from z, deg", tilt, SEQ_BLUE), ("posterior sd in PC1-2, A", uncertainty, SEQ_ORANGE))
        for j, (name, values, cmap) in enumerate(colourings):
            for row, (x, y) in enumerate(((0, 1), (2, 3))):
                pts = principal[usable]
                sc = ax[row, j].scatter(pts[:, x], pts[:, y], c=values[usable], cmap=cmap, s=3, linewidths=0)
                ax[row, j].set(xlabel=f"joint principal coordinate {x + 1} (A)", ylabel=f"joint principal coordinate {y + 1} (A)")
                if row == 0:
                    ax[row, j].set_title(name)
                fig.colorbar(sc, ax=ax[row, j], shrink=0.8)
        fig.suptitle(f"Figure 4{c}a. Case {c}: joint PPCA (q = 10) MAP principal coordinates, PCs 1-2 (top) and 3-4 (bottom); {int(usable.sum())} qualified particles. "
                     "A compact embedding is not evidence of accurate states.", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"fig4{c}_latent_joint.png", dpi=130)

        # Per-body body-2 and the true-pose joint embedding, same coordinates.
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
        panels = (("per-body PPCA, body 2", a["body2_principal"], level), ("per-body PPCA, body 1", a["body1_principal"], twist),
                  ("joint PPCA, true poses", a["true_joint_principal"], level), ("zero-depth PCA, body 2", a["body2_pca"], level))
        for j, (name, pts, values) in enumerate(panels):
            keep = np.all(np.isfinite(pts[:, :2]), axis=1)
            sc = ax[j].scatter(pts[keep, 0], pts[keep, 1], c=values[keep], cmap=SEQ_BLUE if values is level else SEQ_ORANGE, s=3, linewidths=0)
            ax[j].set(title=f"{name}\ncolour: {'body-2 level' if values is level else 'body-1 twist'}", xlabel="coordinate 1 (A)", ylabel="coordinate 2 (A)")
            fig.colorbar(sc, ax=ax[j], shrink=0.8)
        fig.suptitle(f"Figure 4{c}b. Case {c}: comparable embeddings, PCs 1-2", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"fig4{c}_latent_compare.png", dpi=130)

        # Raw PC directions of the joint fit, per body, with their scale.
        bodies = read_bodies(root / f"case{c}_5k/bodies.star", pixel_size=4.25)
        roots = [np.linalg.cholesky(mask_moments(b.mask, b.com, 4.25).covariance) for b in bodies]
        W, mean = a["joint_W"], a["joint_mean"]
        amplitudes = a["joint_amplitudes"]
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
        for k in range(4):
            parts = [direction_parts(W[12 * b : 12 * (b + 1), k], mean[12 * b : 12 * (b + 1)], roots[b]) for b in range(2)]
            x = np.arange(6)
            for b, colour in ((0, BLUE), (1, ORANGE)):
                omega, shift = parts[b]
                ax[k].bar(x + (b - 0.5) * 0.38, np.concatenate([omega, shift]), width=0.38, color=colour, label=f"body {b + 1}")
            ax[k].set_xticks(x, ["rot x", "rot y", "rot z", "t x", "t y", "t z"])
            ax[k].set(title=f"joint PC {k + 1}: amplitude {amplitudes[k]:.2f} A\n(one sd of z)", ylabel="deg (rotation) / A (translation)")
            ax[k].legend(fontsize=7)
            ax[k].axhline(0, color=INK, lw=0.8)
        fig.suptitle(f"Figure 4{c}c. Case {c}: raw joint PC directions, one standard deviation of each latent, molecular frame", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"fig4{c}_pc_directions.png", dpi=130)


def curves(results: Path, root: Path, out: Path) -> None:
    orderings = root / "quality/orderings"
    styles = {"per-body PPCA q10": (BLUE, "-"), "joint q10 whole latent": (AQUA, "-"), "joint q10 decoded body": (GREEN, "--"),
              "zero-depth PCA 10": (MUTED, ":"), "relative PPCA q10": (VIOLET, "-"), "existing best (m3)": (ORANGE, "-."),
              "joint PPCA, true poses": (INK, "--"), "perfect": (INK, ":")}
    for body in ("2", "1"):
        fig, ax = plt.subplots(2, 4, figsize=(17, 7), sharey=True)
        for i, c in enumerate(CASES):
            e = load(results / f"full/embed_case{c}_5k.json")
            rel = results / f"full/relative_case{c}_5k.json"
            rel = load(rel) if rel.exists() else None
            m3 = load(orderings / f"orderings_case{c}_5k.json")["bodies"].get(body, {}).get("fitted", {}).get("targets", {})
            for row, (state, target) in enumerate(e["curves"][body].items()):
                sizes = target["curve_sizes"]
                cm = target["curves_mean_first_20"]
                for key, label in (("per-body PPCA q10", "per-body PPCA q10"), ("joint q10 whole latent", "joint q10 whole latent"),
                                   (f"joint q10 decoded body {body}", "joint q10 decoded body"), ("zero-depth PCA 10", "zero-depth PCA 10")):
                    colour, ls = styles[label]
                    ax[row, i].plot(sizes, cm[key], color=colour, ls=ls, label=label)
                truth = e["curves"][f"{body}_true"][state]["curves_mean_first_20"]
                ax[row, i].plot(sizes, truth["joint q10 whole latent, true poses"], color=INK, ls="--", lw=1.4, label="joint PPCA whole latent, true poses")
                ax[row, i].plot(sizes, truth["joint q10 decoded body, true poses"], color=MUTED, ls="-", lw=1.2, label="joint PPCA decoded body, true poses")
                ax[row, i].plot(sizes, cm["perfect"], color=INK, ls=":", lw=1.4, label="perfect")
                if state in m3:
                    existing = {k: v for k, v in m3[state]["curves_mean_first_20"].items() if k != "perfect"}
                    i100 = int(np.argmin(np.abs(np.asarray(m3[state]["curve_sizes"]) - 100)))
                    best = max(existing, key=lambda k: existing[k][i100])
                    ax[row, i].plot(m3[state]["curve_sizes"], existing[best], color=ORANGE, ls="-.", label=f"existing best (m3): {best}")
                if rel is not None and body == "2":
                    ax[row, i].plot(sizes, rel["curves"][state]["curves_mean_first_20"]["relative PPCA q10"], color=VIOLET, label="relative PPCA q10")
                    ax[row, i].plot(sizes, rel["curves_true"][state]["curves_mean_first_20"]["relative PPCA q10, true poses"], color=VIOLET, ls="--", lw=1.2, label="relative PPCA, true poses")
                ax[row, i].set(xscale="log", title=f"case {c}, body {body}, target state {state}", xlabel="particles kept", ylabel="same-state fraction")
                ax[row, i].legend(fontsize=5.5)
                if c == "D" and body == "2":
                    ax[row, i].text(0.02, 0.03, "unrankable by pose: true poses reach .35", transform=ax[row, i].transAxes, color=RED, fontsize=8)
        fig.suptitle(f"Figure 5. Body {body}: same-state fraction of the nearest particles, mean over 20 particles of each target state", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"fig5_curves_body{body}.png", dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    audit(args.results, args.out)
    calibration(args.results, args.out)
    ppca_diagnostics(args.results, args.out)
    embeddings(args.results, args.root, args.out)
    curves(args.results, args.root, args.out)


if __name__ == "__main__":
    main()
