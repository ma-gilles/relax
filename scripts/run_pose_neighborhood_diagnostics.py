#!/usr/bin/env python
"""Stage B of the pose-neighborhood study: controlled poses, no images. Figures 2 and 3 and their arrays.

    python -m scripts.run_pose_neighborhood_diagnostics --out <dir>

A two-body simulator draws viewing directions independently of structural state. The score sees true rotations and
projected shifts, never a depth. Compared: the bounded score, the free-depth score (diagnostic), zero-depth Euclidean
distance (negative control), pooled PCA of both bodies with its first two components (predeclared), and the true
full-pose distance (labelled oracle). All numbers are angstroms and degrees.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from relax.multibody.pose_neighborhoods import (
    BodyMoments,
    compatibility_distance,
    compatibility_distance_and_bound_hits,
    pose_features,
    snap_round_off,
    zero_depth_distance,
)
from relax.multibody.subsets import nested_thresholds

PIXEL = 4.25
MAX_ROTATION_DEG = 15.0
MAX_TRANSLATION = 10.0
N_STATES = 9
N_PARTICLES = 6000
N_CANDIDATES = 20
N_MIN_PER_HALF = 50
FAMILIES = ("rotation", "translation", "mixed", "independent")
METHODS = ("bounded", "free", "zero-depth", "pooled PCA", "oracle")
COLORS = {"bounded": "#2a6f97", "free": "#e0a030", "zero-depth": "#b23a48", "pooled PCA": "#6a4c93", "oracle": "#333333"}
AXIS_ROT = np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])
AXIS_TRANS = np.array([1.0, 0.0, 0.0])
BODY = BodyMoments(np.array([8.0, -4.0, 3.0]), np.diag([900.0, 400.0, 100.0]))
OTHER = BodyMoments(np.array([-5.0, 6.0, -2.0]), np.diag([300.0, 500.0, 200.0]))


def depth_bound(family: str, body: int = 0) -> float:
    """``1.25 max(T_b, one pixel)`` from the generator's declared largest pivot displacement, before any result."""
    moved = family != "rotation"
    return 1.25 * max(MAX_TRANSLATION if moved else 0.0, PIXEL)


def physical_poses(level: np.ndarray, family: str):
    """Rotations and translations of ``level`` in [-1, 1]."""
    rotate = family != "translation"
    translate = family != "rotation"
    Q = Rotation.from_rotvec(np.outer(level * np.deg2rad(MAX_ROTATION_DEG) * rotate, AXIS_ROT)).as_matrix()
    return Q, np.outer(level * MAX_TRANSLATION * translate, AXIS_TRANS)


def simulate(family: str, seed: int, n: int = N_PARTICLES, *, continuous: bool = True) -> dict:
    """Particles with a structural level in [-1, 1]: continuous, ``state`` being its bin of ``N_STATES``, or exactly
    the bin centres, where one state's particles tie at zero score."""
    rng = np.random.default_rng(seed)
    centres = np.linspace(-1.0, 1.0, N_STATES)
    draw = (lambda: rng.uniform(-1.0, 1.0, size=n)) if continuous else (lambda: centres[rng.integers(0, N_STATES, size=n)])
    level = draw()
    other_level = draw() if family == "independent" else level
    state = np.argmin(np.abs(level[:, None] - centres), axis=1)
    motion = "mixed" if family == "independent" else family
    Q, t = physical_poses(level, motion)
    Q2, t2 = physical_poses(other_level, motion)
    cameras = Rotation.random(n, random_state=rng).as_matrix()
    half = rng.integers(0, 2, size=n)
    return {"state": state, "Q": Q, "t": t, "Q2": Q2, "t2": t2, "cameras": cameras, "half": half, "family": family}


def observe(sim, *, shift_noise=0.0, rotation_noise_deg=0.0, seed=0):
    """Rotations, zero-depth lifts and depth directions as the score receives them, with optional fitted-pose noise."""
    rng = np.random.default_rng(seed)
    cameras = sim["cameras"]
    normals = cameras[:, 2, :]
    in_plane = np.einsum("nij,nj->ni", cameras, sim["t"])[:, :2] + rng.normal(size=(len(cameras), 2)) * shift_noise
    lifts = np.einsum("nij,nj->ni", np.swapaxes(cameras, 1, 2)[:, :, :2], in_plane)
    Q = sim["Q"]
    if rotation_noise_deg:
        Q = Rotation.from_rotvec(np.deg2rad(rotation_noise_deg) * rng.normal(size=(len(Q), 3))).as_matrix() @ Q
    return Q, lifts, normals


def oracle_distance(sim, target: int) -> np.ndarray:
    """The true weighted RMS displacement of body 1's points between each particle and the target."""
    root = np.diag(np.sqrt(np.diag(BODY.covariance)))
    dQ = sim["Q"] - sim["Q"][target]
    shape = np.sum((dQ @ root) ** 2, axis=(1, 2))
    centre = sim["t"] - sim["t"][target] + dQ @ BODY.centroid
    return np.sqrt(shape + np.sum(centre * centre, axis=1))


def pooled_pca(sim, observed) -> np.ndarray:
    """First two unwhitened components of both bodies' residual rotations and in-plane shifts (RELION's flex analysis
    has this form: every body's motion in one table)."""
    Q, _, _ = observed
    cameras = sim["cameras"]
    table = [Rotation.from_matrix(Q).as_rotvec(degrees=True), np.einsum("nij,nj->ni", cameras, sim["t"])[:, :2],
             Rotation.from_matrix(sim["Q2"]).as_rotvec(degrees=True), np.einsum("nij,nj->ni", cameras, sim["t2"])[:, :2]]
    table = np.concatenate(table, axis=1)
    table = table - table.mean(axis=0)
    _, _, vt = np.linalg.svd(table, full_matrices=False)
    return table @ vt[:2].T


def method_distances(sim, observed, features, scores, target, bound) -> dict:
    d, hits = compatibility_distance_and_bound_hits(features, target, bound)
    length = max(bound, np.sqrt(np.trace(BODY.covariance) + BODY.centroid @ BODY.centroid))
    return {
        "bounded": snap_round_off(d, length),
        "free": snap_round_off(compatibility_distance(features, target, None), length),
        "zero-depth": zero_depth_distance(features, target),
        "pooled PCA": np.linalg.norm(scores - scores[target], axis=1),
        "oracle": snap_round_off(oracle_distance(sim, target), length),
        "_hits": hits,
    }


def view_bins(normals: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Nearest of ``n_bins`` Fibonacci directions on the half sphere, a line of sight having no sign."""
    k = np.arange(n_bins) + 0.5
    z = 1.0 - k / n_bins
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    r = np.sqrt(1.0 - z * z)
    grid = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    return cKDTree(np.concatenate([grid, -grid])).query(normals)[1] % n_bins


def subset_curve(distance, sim, target, oracle, bins, hits, halves):
    thresholds = nested_thresholds(distance, n_candidates=N_CANDIDATES, n_min_per_half=N_MIN_PER_HALF, halves=halves)
    rows = []
    for h in thresholds:
        chosen = distance <= h
        counts = [int(np.sum(chosen[r])) for r in halves]
        rows.append((chosen.sum(), np.mean(sim["state"][chosen] == sim["state"][target]), np.median(oracle[chosen]),
                     np.quantile(oracle[chosen], 0.9), np.unique(bins[chosen]).size / 64.0, min(counts) / max(counts),
                     np.mean(hits[chosen]), np.mean(np.abs(sim["state"][chosen] - sim["state"][target]))))
    return np.asarray(rows)


GRID = np.unique(np.round(np.logspace(np.log10(100), np.log10(N_PARTICLES), 24)))


def on_grid(curve: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(GRID, curve[:, 0], curve[:, c], left=np.nan) for c in range(1, curve.shape[1])], axis=1)


def targets_of(sim, seed=0):
    """Six particles of the central state and six of the first end state, chosen before any score, by row."""
    return [int(i) for s in (N_STATES // 2, 0) for i in np.flatnonzero(sim["state"] == s)[:6]]


def study(family, *, bound_scale=1.0, shift_noise=0.0, rotation_noise_deg=0.0, seed=11, continuous=True) -> dict:
    sim = simulate(family, seed, continuous=continuous)
    observed = observe(sim, shift_noise=shift_noise, rotation_noise_deg=rotation_noise_deg, seed=seed + 1)
    features = pose_features(*observed, BODY)
    scores = pooled_pca(sim, observed)
    bound = bound_scale * depth_bound(family)
    halves = [np.flatnonzero(sim["half"] == h) for h in range(2)]
    bins = view_bins(observed[2])
    curves = {m: [] for m in METHODS}
    random_coverage = []
    rng = np.random.default_rng(seed + 2)
    for target in targets_of(sim):
        oracle = oracle_distance(sim, target)
        distances = method_distances(sim, observed, features, scores, target, bound)
        for m in METHODS:
            curves[m].append(on_grid(subset_curve(distances[m], sim, target, oracle, bins, distances["_hits"], halves)))
    for count in GRID.astype(int):
        random_coverage.append(np.unique(bins[rng.choice(N_PARTICLES, count, replace=False)]).size / 64.0)
    target = targets_of(sim)[6]
    scatter = method_distances(sim, observed, features, scores, target, bound)
    return {
        "curves": {m: np.nanmean(np.stack(c), axis=0) for m, c in curves.items()},
        "random_coverage": np.asarray(random_coverage), "bound": bound,
        "scatter": {"bounded": scatter["bounded"], "oracle": scatter["oracle"], "view_cos": np.abs(observed[2] @ observed[2][target]),
                    "same_state": sim["state"] == sim["state"][target]},
        "bound_hit_all": float(np.mean(scatter["_hits"])), "zero_ties": float(np.mean(scatter["bounded"] == 0.0)),
    }


def permutation_check(family, seed=11) -> float:
    """Largest change of same-state fraction when the particles of a discrete, tied fixture are relabelled."""
    sim = simulate(family, seed, continuous=False)
    observed = observe(sim)
    features = pose_features(*observed, BODY)
    bound = depth_bound(family)
    target = targets_of(sim)[6]
    order = np.random.default_rng(seed).permutation(N_PARTICLES)
    d = snap_round_off(compatibility_distance(features, target, bound), bound)
    shuffled = pose_features(observed[0][order], observed[1][order], observed[2][order], BODY)
    d2 = snap_round_off(compatibility_distance(shuffled, int(np.argsort(order)[target]), bound), bound)
    halves = [np.flatnonzero(sim["half"] == h) for h in range(2)]
    halves2 = [np.flatnonzero(sim["half"][order] == h) for h in range(2)]
    t1 = nested_thresholds(d, n_candidates=N_CANDIDATES, n_min_per_half=N_MIN_PER_HALF, halves=halves)
    t2 = nested_thresholds(d2, n_candidates=N_CANDIDATES, n_min_per_half=N_MIN_PER_HALF, halves=halves2)
    worst = 0.0
    for a, b in zip(t1, t2):
        same = sim["state"] == sim["state"][target]
        worst = max(worst, abs(np.mean(same[d <= a]) - np.mean(same[order][d2 <= b])))
    return float(worst)


def target_view_study(family, seed=11, n_views=64, keep=600) -> dict:
    """The target's physical pose and every candidate fixed; only the target's camera changes."""
    sim = simulate(family, seed)
    Q, lifts, normals = observe(sim)
    bound = depth_bound(family)
    target = targets_of(sim)[6]
    oracle = oracle_distance(sim, target)
    cameras = Rotation.random(n_views, random_state=seed).as_matrix()
    neighbours, purity, median, tie = [], [], [], []
    for camera in cameras:
        moved_lifts, moved_normals = lifts.copy(), normals.copy()
        moved_normals[target] = camera[2]
        moved_lifts[target] = camera[:2].T @ (camera @ sim["t"][target])[:2]
        d = snap_round_off(compatibility_distance(pose_features(Q, moved_lifts, moved_normals, BODY), target, bound), bound)
        h = np.sort(d)[keep - 1]
        chosen = d <= h
        neighbours.append(chosen)
        purity.append(np.mean(sim["state"][chosen] == sim["state"][target]))
        median.append(np.median(oracle[chosen]))
        tie.append(np.mean(d == 0.0))
    neighbours = np.asarray(neighbours)
    a, b = np.triu_indices(n_views, 1)
    overlap = np.sum(neighbours[a] & neighbours[b], axis=1) / np.sum(neighbours[a] | neighbours[b], axis=1)
    along = np.abs(cameras[:, 2] @ AXIS_TRANS)
    return {"overlap": overlap, "purity": np.asarray(purity), "median_oracle": np.asarray(median), "tie": np.asarray(tie), "view_along_motion": along}


def ambiguity_example(a=10.0, n=4000, seed=5) -> dict:
    """Identical rotations, translations 0 and ``a e_x``, the target at state 0 seen along ``e_x``, then across it."""
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, size=n)
    cameras = Rotation.random(n, random_state=rng).as_matrix()
    t = np.outer(state * a, [1.0, 0.0, 0.0])
    normals = cameras[:, 2, :]
    lifts = t - np.sum(t * normals, axis=1, keepdims=True) * normals
    bound = 1.25 * max(a, PIXEL)
    Q = np.tile(np.eye(3), (n, 1, 1))
    out = {}
    for name, view in (("along e_x", [1.0, 0.0, 0.0]), ("along e_y", [0.0, 1.0, 0.0])):
        normal = np.array(view)
        target = int(np.flatnonzero(state == 0)[0])
        moved_n, moved_l = normals.copy(), lifts.copy()
        moved_n[target], moved_l[target] = normal, 0.0
        d = snap_round_off(compatibility_distance(pose_features(Q, moved_l, moved_n, BODY), target, bound), bound)
        out[name] = {"d_state0": d[state == 0], "d_state1": d[state == 1],
                     "zero_fraction_state0": float(np.mean(d[state == 0] == 0.0)), "zero_fraction_state1": float(np.mean(d[state == 1] == 0.0))}
    return out


def timing(sizes=(1000, 10000, 100000), repeats=30, seed=3) -> dict:
    rows = {}
    for n in sizes:
        rng = np.random.default_rng(seed)
        Q = Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.1).as_matrix()
        cameras = Rotation.random(n, random_state=rng).as_matrix()
        normals = cameras[:, 2, :]
        lifts = rng.normal(size=(n, 3)) * 5.0
        lifts -= np.sum(lifts * normals, axis=1, keepdims=True) * normals
        halves = [np.arange(0, n, 2), np.arange(1, n, 2)]
        spans = {"prepare": [], "distance": [], "sort_select": []}
        for r in range(repeats):
            t0 = time.perf_counter()
            features = pose_features(Q, lifts, normals, BODY)
            t1 = time.perf_counter()
            d = compatibility_distance(features, r, 12.5)
            t2 = time.perf_counter()
            thresholds = nested_thresholds(d, n_candidates=50, n_min_per_half=50, halves=halves)
            _ = d <= thresholds[len(thresholds) // 2]
            t3 = time.perf_counter()
            spans["prepare"].append(t1 - t0); spans["distance"].append(t2 - t1); spans["sort_select"].append(t3 - t2)
        rows[str(n)] = {k: {"cold_s": v[0], "warm_median_s": float(np.median(v[1:])), "warm_p90_s": float(np.quantile(v[1:], 0.9))} for k, v in spans.items()}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {f: study(f) for f in FAMILIES}
    discrete = {f: study(f, continuous=False) for f in FAMILIES}
    wide = {f: study(f, bound_scale=2.0) for f in FAMILIES}
    noise_levels = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (4.0, 3.0)]
    noisy = {f: [study(f, shift_noise=s, rotation_noise_deg=r) for s, r in noise_levels] for f in ("mixed", "translation")}
    views = {f: target_view_study(f) for f in FAMILIES}
    ambiguity = ambiguity_example()
    permutation = {f: permutation_check(f) for f in FAMILIES}
    cost = timing()
    summary = {
        "families": list(FAMILIES), "methods": list(METHODS), "n_particles": N_PARTICLES, "n_states": N_STATES,
        "max_rotation_deg": MAX_ROTATION_DEG, "max_translation_A": MAX_TRANSLATION, "pixel_A": PIXEL,
        "depth_bounds_A": {f: results[f]["bound"] for f in FAMILIES},
        "bound_hit_fraction_all_particles": {f: results[f]["bound_hit_all"] for f in FAMILIES},
        "zero_tie_fraction_continuous": {f: results[f]["zero_ties"] for f in FAMILIES},
        "zero_tie_fraction_discrete": {f: discrete[f]["zero_ties"] for f in FAMILIES},
        "discrete_same_state_near_1000": {f: {m: float(np.interp(1000, GRID, discrete[f]["curves"][m][:, 0])) for m in METHODS} for f in FAMILIES},
        "permutation_worst_same_state_change": permutation,
        "ambiguity": {k: {q: v[q] for q in ("zero_fraction_state0", "zero_fraction_state1")} for k, v in ambiguity.items()},
        "same_state_fraction_near_1000": {
            f: {m: float(np.interp(1000, GRID, results[f]["curves"][m][:, 0])) for m in METHODS} for f in FAMILIES},
        "median_oracle_near_1000_A": {
            f: {m: float(np.interp(1000, GRID, results[f]["curves"][m][:, 1])) for m in METHODS} for f in FAMILIES},
        "two_L_same_state_near_1000": {f: float(np.interp(1000, GRID, wide[f]["curves"]["bounded"][:, 0])) for f in FAMILIES},
        "noise_levels_A_deg": noise_levels,
        "noise_same_state_near_1000": {f: {m: [float(np.interp(1000, GRID, x["curves"][m][:, 0])) for x in noisy[f]] for m in METHODS} for f in noisy},
        "target_view_overlap_median": {f: float(np.median(views[f]["overlap"])) for f in FAMILIES},
        "target_view_overlap_p10": {f: float(np.quantile(views[f]["overlap"], 0.1)) for f in FAMILIES},
        "target_view_same_state_range": {f: [float(views[f]["purity"].min()), float(views[f]["purity"].max())] for f in FAMILIES},
        "target_view_median_oracle_range_A": {f: [float(views[f]["median_oracle"].min()), float(views[f]["median_oracle"].max())] for f in FAMILIES},
        "timing": cost,
        "timing_conditions": {"dtype": "float64", "machine": next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.machine()), "threads": "OMP_NUM_THREADS from the job",
                              "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                              "includes": "NumPy arrays in memory; no transfers or file I/O"},
    }
    (args.out / "stage_b_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        args.out / "stage_b_arrays.npz", grid=GRID,
        **{f"curve_{f}_{m}".replace(" ", "_"): results[f]["curves"][m] for f in FAMILIES for m in METHODS},
        **{f"curve2L_{f}".replace(" ", "_"): wide[f]["curves"]["bounded"] for f in FAMILIES},
        **{f"scatter_{f}_{k}": v for f in FAMILIES for k, v in results[f]["scatter"].items()},
        **{f"view_{f}_{k}": v for f in FAMILIES for k, v in views[f].items()},
        **{f"ambiguity_{k.replace(' ', '_')}_{q}": np.asarray(v[q]) for k, v in ambiguity.items() for q in ("d_state0", "d_state1")},
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "timing"}, indent=1))
    make_figures(args.out, results, wide, noisy, noise_levels, views, ambiguity, cost)


def make_figures(out, results, wide, noisy, noise_levels, views, ambiguity, cost) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 5, figsize=(21, 14.5), squeeze=False)
    for row, family in enumerate(FAMILIES):
        result = results[family]
        ax = axes[row, 0]
        s = result["scatter"]
        ax.scatter(s["oracle"], s["bounded"], c=s["view_cos"], s=2, cmap="viridis", vmin=0, vmax=1, rasterized=True)
        limit = max(s["oracle"].max(), s["bounded"].max())
        ax.plot([0, limit], [0, limit], color="#555555", lw=0.8)
        ax.set(xlabel="oracle full-pose distance (A)", ylabel="bounded score (A)", title=f"{family}: score against oracle (colour |cos view|)")
        for column, (index, ylabel, title) in enumerate(((0, "same-state fraction", "true state in subset"), (2, "q90 oracle distance (A)", "upper-quantile oracle distance")), start=1):
            ax = axes[row, column]
            for m in METHODS:
                ax.semilogx(GRID, result["curves"][m][:, index], color=COLORS[m], label=m)
            ax.set(xlabel="retained particles", ylabel=ylabel, title=title)
        axes[row, 1].axhline(1 / N_STATES, color="#999999", ls=":", lw=0.8)
        ax = axes[row, 3]
        curve = result["curves"]["bounded"]
        ax.semilogx(GRID, curve[:, 3], color=COLORS["bounded"], label="view coverage (64 bins)")
        ax.semilogx(GRID, result["random_coverage"], color="#999999", ls=":", label="coverage, random subset")
        ax.semilogx(GRID, curve[:, 4], color="#3a7d44", label="smaller / larger half")
        ax.semilogx(GRID, curve[:, 5], color="#e0a030", label="depth-bound-hit fraction")
        ax.set(xlabel="retained particles", ylabel="fraction", title="views, halves, depth bound (bounded score)", ylim=(-0.03, 1.05))
        ax = axes[row, 4]
        v = views[family]
        ax.plot(v["view_along_motion"], v["purity"], ".", color=COLORS["bounded"], label="same-state fraction, 600 kept")
        ax.plot(v["view_along_motion"], v["tie"], ".", color="#b23a48", label="exact-zero fraction")
        ax.plot(v["view_along_motion"], v["median_oracle"] / 20.0, ".", color="#333333", label="median oracle distance / 20 A")
        ax.set(xlabel="|cos| target view with the motion axis", ylabel="fraction",
               title=f"target camera varied: median overlap {np.median(v['overlap']):.2f}", ylim=(-0.03, 1.05))
        ax.legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure2.png", dpi=120)
    fig.savefig(out / "figure2.pdf", dpi=120)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    ax = axes[0]
    for name, ls in (("along e_x", "-"), ("along e_y", "--")):
        values = ambiguity[name]
        for key, color in (("d_state0", "#2a6f97"), ("d_state1", "#b23a48")):
            ax.hist(values[key], bins=40, range=(0, 11), histtype="step", ls=ls, color=color, label=f"view {name}, {'same' if key == 'd_state0' else 'other'} state")
    ax.set_yscale("log")
    ax.set(xlabel="bounded score (A)", ylabel="particles", title="Translation-only ambiguity, two states 10 A apart")
    ax.legend(frameon=False, fontsize=8)
    for ax, family in zip(axes[1:3], noisy):
        for (s, r), result, alpha in zip(noise_levels, noisy[family], (1.0, 0.75, 0.5, 0.3)):
            ax.semilogx(GRID, result["curves"]["bounded"][:, 0], color=COLORS["bounded"], alpha=alpha, label=f"{s:g} A, {r:g} deg")
        ax.axhline(1 / N_STATES, color="#999999", ls=":", lw=0.8)
        ax.set(xlabel="retained particles", ylabel="same-state fraction", title=f"Pose noise, {family}")
        ax.legend(frameon=False, fontsize=8)
    ax = axes[3]
    for family in FAMILIES:
        ax.semilogx(GRID, results[family]["curves"]["bounded"][:, 0], label=family)
        ax.semilogx(GRID, wide[family]["curves"]["bounded"][:, 0], ls="--", color=ax.lines[-1].get_color())
    ax.set(xlabel="retained particles", ylabel="same-state fraction", title="Depth bound L (solid) against 2L (dashed)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure2b.png", dpi=120)
    fig.savefig(out / "figure2b.pdf", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 4.4))
    sizes = sorted(int(k) for k in cost)
    for key, color in (("prepare", "#6a4c93"), ("distance", "#2a6f97"), ("sort_select", "#e0a030")):
        warm = [cost[str(n)][key]["warm_median_s"] for n in sizes]
        ax.loglog(sizes, warm, marker="o", color=color, label=f"{key}, warm median")
        ax.loglog(sizes, [cost[str(n)][key]["cold_s"] for n in sizes], marker="x", ls=":", color=color, label=f"{key}, cold")
    ax.set(xlabel="particles", ylabel="seconds", title="One target against N particles, float64 CPU")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure3.png", dpi=150)
    fig.savefig(out / "figure3.pdf")


if __name__ == "__main__":
    main()
