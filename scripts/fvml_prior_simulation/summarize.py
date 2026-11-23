"""Collect the simulation's result rows into a CSV and draw the comparison figures.

Writes, under ``--out-dir``:
  * ``summary.csv``: one row per (target, fit setting, particle count, replicate, fit order, method), the fitted
    parameters dropped;
  * ``js_fit_order.png`` and ``js_eval_order.png``: the Jensen-Shannon distance (bits) of each estimate to the target,
    on the fit grid and on the target's evaluation grid, against the particle count (median over replicates, band
    from the smallest to the largest), one panel per target and fit order; the histogram is RELION's estimate;
  * ``relative_fit_order.png`` / ``.csv``: each fit's distance on the fit grid over the histogram's for the same draw
    (median over replicates), one panel per target;
  * ``kappa_recovery.png``: for the mixture targets, each lobe's fitted concentration over its true one;
  * ``maps_<target>.png``: the target and the three estimates at one fit order as rot x tilt density maps;
  * ``by_lobes_c<c>.png`` and ``by_c.png``: the cell fit's distance per lobe count (one figure per c with several
    counts) and per cap width c (base lobe count), beside the histogram;
  * ``selection.csv``: per target, order, particle count and FvML method, which lobe count (and which c) the held-out
    log-likelihood picks across replicates, and the picked fit's distance over the best of the candidates.
The figures before ``by_lobes_c<c>.png`` use each target's base setting: c = 0.25 and 5 lobes (the true count for a
known mixture), or the first configured setting when that one was not run.

Run (Snakemake calls it as rule summarize):
    python scripts/fvml_prior_simulation/summarize.py --config scripts/fvml_prior_simulation/config.yaml \
        --results-dir <output_root>/results --targets-dir <output_root>/targets --out-dir <output_root>/summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.fvml_prior_simulation.simulation import (  # noqa: E402
    METHODS,
    Target,
    load_target,
    mixture_cell_probabilities,
)
from relax.sampling.fvml_mixture import FvmlMixture  # noqa: E402

COLORS = {"histogram": "#52514e", "point": "#2a78d6", "cell": "#eb6834"}
LABELS = {"histogram": "histogram (RELION)", "point": "FvML, point atoms", "cell": "FvML, cell masses"}
INF_POSITION = 1e7  # where the "inf" particle count is drawn on the log axis
MAP_ORDER = 4
MAP_COUNTS = (1000, 10000, 100000)
BASE_C, BASE_LOBES = 0.25, 5
SETTING_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def read_rows(results_dir: Path) -> pd.DataFrame:
    """All result rows of ``results_dir/<target>/<setting>/n*.jsonl``, with the setting and lobe count as columns."""
    rows = []
    for path in sorted(results_dir.glob("*/*/n*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                rows.append({**json.loads(line), "setting": path.parent.name})
    if not rows:
        raise FileNotFoundError(f"no result rows under {results_dir}")
    frame = pd.DataFrame(rows)
    frame["n_value"] = [INF_POSITION if n == "inf" else float(n) for n in frame["n_particles"]]
    frame["lobes"] = frame["n_components"] - 1
    for column in ("heldout_log_likelihood", "heldout_log_likelihood_truth"):
        frame[column] = [float(v) for v in frame[column]]
    return frame


def base_setting(target: Target, c: float) -> str:
    """The setting the per-target figures use: cap width ``c`` and 5 lobes, or the true count for a known mixture."""
    return f"c{c:g}_lobes" + ("truth" if target.mixture is not None else str(BASE_LOBES))


def _band(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Median, minimum and maximum of ``column`` over replicates per particle count."""
    return frame.groupby("n_value")[column].agg(["median", "min", "max"]).sort_index()


def _format_count_axis(ax: plt.Axes, counts: list[float]) -> None:
    """Log x axis with the particle counts as ticks and "inf" for the floor."""
    ax.set_xscale("log")
    ax.set_xticks(counts)
    ax.set_xticklabels(["inf" if c == INF_POSITION else f"{int(c):.0e}".replace("e+0", "e") for c in counts])
    ax.minorticks_off()


def plot_distance(frame: pd.DataFrame, column: str, title: str, path: Path) -> None:
    """One panel per target (rows) and fit order (columns): ``column`` against the particle count per method."""
    targets = list(dict.fromkeys(frame["target"]))
    orders = sorted(frame["order"].unique())
    counts = sorted(frame["n_value"].unique())
    fig, axes = plt.subplots(len(targets), len(orders), figsize=(4.2 * len(orders), 2.8 * len(targets)), squeeze=False)
    for i, target in enumerate(targets):
        for k, order in enumerate(orders):
            ax = axes[i, k]
            for method in METHODS:
                sub = frame[(frame["target"] == target) & (frame["order"] == order) & (frame["method"] == method)]
                if method == "histogram":
                    sub = sub[sub["n_value"] != INF_POSITION]  # without draws the histogram is the target itself
                band = _band(sub, column)
                ax.plot(band.index, band["median"], color=COLORS[method], lw=2, marker="o", ms=5, label=LABELS[method])
                ax.fill_between(band.index, band["min"], band["max"], color=COLORS[method], alpha=0.18, lw=0)
            ax.set_yscale("log")
            _format_count_axis(ax, counts)
            ax.grid(True, which="major", color="#e4e3dc", lw=0.6)
            ax.set_title(f"{target}, order {order}", fontsize=9)
            if k == 0:
                ax.set_ylabel("JS distance (bits)")
            if i == len(targets) - 1:
                ax.set_xlabel("particles")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def relative_to_histogram(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per target, fit order, particle count and FvML method: median over replicates of ``column`` over the histogram's.

    Each replicate's fit is divided by the histogram of the same draw; below 1 the fit is closer to the target.
    """
    keys = ["target", "order", "n_particles", "replicate"]
    finite = frame[frame["n_particles"] != "inf"]
    histogram = finite[finite["method"] == "histogram"].set_index(keys)[column]
    rows = []
    for method in ("point", "cell"):
        fits = finite[finite["method"] == method].set_index(keys)[column]
        ratio = (fits / histogram.reindex(fits.index)).rename("ratio").reset_index()
        ratio["method"] = method
        rows.append(ratio)
    ratios = pd.concat(rows)
    ratios["n_value"] = ratios["n_particles"].astype(float)
    return ratios.groupby(["target", "order", "n_value", "method"])["ratio"].median().reset_index()


def plot_relative(frame: pd.DataFrame, column: str, title: str, path: Path) -> None:
    """One panel per target: the fits' distance over the histogram's against the particle count, a line per order."""
    ratios = relative_to_histogram(frame, column)
    targets = list(dict.fromkeys(frame["target"]))
    columns = 3
    rows = math.ceil(len(targets) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.0 * rows), squeeze=False)
    styles = {3: ":", 4: "--", 5: "-"}
    for ax, target in zip(axes.flat, targets):
        for method in ("point", "cell"):
            for order in sorted(ratios["order"].unique()):
                sub = ratios[(ratios["target"] == target) & (ratios["method"] == method) & (ratios["order"] == order)]
                sub = sub.sort_values("n_value")
                ax.plot(
                    sub["n_value"],
                    sub["ratio"],
                    styles.get(int(order), "-"),
                    color=COLORS[method],
                    lw=1.8,
                    marker="o",
                    ms=4,
                    label=f"{method}, order {order}",
                )
        ax.axhline(1.0, color="#0b0b0b", lw=0.8)
        ax.set_yscale("log")
        _format_count_axis(ax, sorted(ratios["n_value"].unique()))
        ax.grid(True, which="major", color="#e4e3dc", lw=0.6)
        ax.set_title(target, fontsize=9)
        ax.set_ylabel("fit / histogram")
        ax.set_xlabel("particles")
    for ax in list(axes.flat)[len(targets) :]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_kappa_recovery(frame: pd.DataFrame, targets: dict[str, Target], path: Path) -> None:
    """Fitted over true concentration per lobe for the mixture targets, point and cell fits, per fit order."""
    names = [n for n, t in targets.items() if t.mixture is not None and n in set(frame["target"])]
    orders = sorted(frame["order"].unique())
    counts = sorted(frame["n_value"].unique())
    fig, axes = plt.subplots(len(names), len(orders), figsize=(4.2 * len(orders), 2.8 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        truth = targets[name].mixture
        for k, order in enumerate(orders):
            ax = axes[i, k]
            sub = frame[(frame["target"] == name) & (frame["order"] == order)]
            kappa_max = float(sub["kappa_max"].iloc[0])
            for method, style in (("point", "--"), ("cell", "-")):
                rows = sub[sub["method"] == method].dropna(subset=["kappa_ratio"])
                ratios = np.array(rows["kappa_ratio"].tolist())
                for j, kappa in enumerate(truth.kappas):
                    per_n = pd.DataFrame({"n_value": rows["n_value"].to_numpy(), "r": ratios[:, j]})
                    med = per_n.groupby("n_value")["r"].median().sort_index()
                    shade = 0.35 + 0.65 * j / max(1, truth.kappas.size - 1)
                    ax.plot(
                        med.index,
                        med.values,
                        style,
                        color=COLORS[method],
                        alpha=shade,
                        lw=1.6,
                        marker="o",
                        ms=4,
                        label=f"{method}, kappa {kappa:g}" if k == 0 else None,
                    )
            for kappa in truth.kappas:
                if kappa > kappa_max:
                    ax.axhline(kappa_max / kappa, color="#9b9a93", lw=0.8, ls=":")
            ax.axhline(1.0, color="#0b0b0b", lw=0.8)
            _format_count_axis(ax, counts)
            ax.set_ylim(0.0, 1.6)
            ax.grid(True, color="#e4e3dc", lw=0.6)
            ax.set_title(f"{name}, order {order} (cap {kappa_max:.0f})", fontsize=9)
            if k == 0:
                ax.set_ylabel("fitted / true kappa")
                ax.legend(fontsize=6, frameon=False, ncol=2)
            if i == len(names) - 1:
                ax.set_xlabel("particles")
    fig.suptitle("Concentration recovery (dotted: the cap over a lobe's true kappa)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _rot_tilt_image(pmf: np.ndarray, order: int, shape: tuple[int, int] = (180, 360)) -> np.ndarray:
    """A cell probability vector as density per steradian on a regular (tilt, rot) grid in degrees."""
    tilt = np.radians((np.arange(shape[0]) + 0.5) * 180.0 / shape[0])
    rot = np.radians((np.arange(shape[1]) + 0.5) * 360.0 / shape[1])
    pixels = hp.ang2pix(1 << order, tilt[:, None] + 0 * rot[None], 0 * tilt[:, None] + rot[None], nest=True)
    return pmf[pixels] / (4.0 * math.pi / pmf.size)


def _fitted(row: pd.Series, symmetry: np.ndarray) -> FvmlMixture:
    """The mixture a result row stores."""
    fit: dict[str, Any] = row["fit"]
    return FvmlMixture(fit["weights"], fit["means"], fit["kappas"], symmetry)


def plot_maps(frame: pd.DataFrame, target: Target, results_dir: Path, c: float, path: Path) -> None:
    """Target and estimates (replicate 0, fit order ``MAP_ORDER`` or the finest fitted, base setting at cap ``c``)
    as rot x tilt maps.

    One column per particle count of ``MAP_COUNTS`` that was run.
    """
    fitted_orders = sorted(frame["order"].unique())
    map_order = MAP_ORDER if MAP_ORDER in fitted_orders else int(fitted_orders[-1])
    setting_dir = results_dir / target.name / base_setting(target, c)
    counts = [n for n in MAP_COUNTS if (setting_dir / f"n{n}_masses.npz").is_file()]
    if not counts:
        return
    truth = target.truth(map_order)
    images: list[list[tuple[str, np.ndarray]]] = []
    for n in counts:
        masses = np.load(setting_dir / f"n{n}_masses.npz")[f"order{map_order}"]
        column = [("target", truth), (f"histogram, {n} particles", masses / masses.sum())]
        for method in ("point", "cell"):
            rows = frame[
                (frame["target"] == target.name)
                & (frame["n_particles"].astype(str) == str(n))
                & (frame["replicate"] == 0)
                & (frame["order"] == map_order)
                & (frame["method"] == method)
            ]
            mixture = _fitted(rows.iloc[0], target.symmetry)
            column.append((f"{LABELS[method]}, {n}", mixture_cell_probabilities(mixture, map_order)))
        images.append(column)
    density = [_rot_tilt_image(p, map_order) for col in images for _, p in col]
    positive = np.concatenate([d[d > 0] for d in density])
    vmin, vmax = np.percentile(positive, 1), positive.max()
    fig, axes = plt.subplots(4, len(counts), figsize=(4.4 * len(counts), 9.0), squeeze=False)
    for c, column in enumerate(images):
        for r, (label, pmf) in enumerate(column):
            ax = axes[r, c]
            image = _rot_tilt_image(pmf, map_order)
            shown = ax.imshow(
                np.clip(image, vmin, None),
                extent=(0, 360, 180, 0),
                aspect="auto",
                cmap="Blues",
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            )
            ax.set_title(label, fontsize=8)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_yticks([0, 45, 90, 135, 180])
            if c == 0:
                ax.set_ylabel("tilt (deg)")
            if r == 3:
                ax.set_xlabel("rot (deg)")
    fig.subplots_adjust(hspace=0.45, wspace=0.25, right=0.86)
    fig.colorbar(shown, ax=axes, shrink=0.6, label="density per steradian (log)")
    fig.suptitle(f"{target.name}: HEALPix order {map_order}, replicate 0", fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_settings(frame: pd.DataFrame, settings: dict[str, list[str]], label: str, title: str, path: Path) -> None:
    """Cell fit's distance on the fit grid per setting (``settings[target]``, in order), rows targets, columns orders.

    The histogram (the same draws in every setting) is drawn in grey from the first setting.
    """
    targets = [t for t in settings if len(settings[t]) > 1]
    orders = sorted(frame["order"].unique())
    counts = sorted(frame["n_value"].unique())
    fig, axes = plt.subplots(len(targets), len(orders), figsize=(4.2 * len(orders), 2.8 * len(targets)), squeeze=False)
    for i, target in enumerate(targets):
        for k, order in enumerate(orders):
            ax = axes[i, k]
            here = frame[(frame["target"] == target) & (frame["order"] == order)]
            hist = here[(here["setting"] == settings[target][0]) & (here["method"] == "histogram")]
            band = _band(hist[hist["n_value"] != INF_POSITION], "js_bits_fit")
            ax.plot(
                band.index,
                band["median"],
                color=COLORS["histogram"],
                lw=1.5,
                ls="--",
                marker="s",
                ms=4,
                label="histogram",
            )
            for j, setting in enumerate(settings[target]):
                sub = here[(here["setting"] == setting) & (here["method"] == "cell")]
                band = _band(sub, "js_bits_fit")
                color = SETTING_COLORS[j % len(SETTING_COLORS)]
                value = sub[label].iloc[0]
                ax.plot(band.index, band["median"], color=color, lw=2, marker="o", ms=4, label=f"{label} {value:g}")
                ax.fill_between(band.index, band["min"], band["max"], color=color, alpha=0.15, lw=0)
            ax.set_yscale("log")
            _format_count_axis(ax, counts)
            ax.grid(True, which="major", color="#e4e3dc", lw=0.6)
            ax.set_title(f"{target}, order {order}", fontsize=9)
            if k == 0:
                ax.set_ylabel("JS distance (bits)")
            if i == len(targets) - 1:
                ax.set_xlabel("particles")
        axes[i, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def heldout_selection(frame: pd.DataFrame, settings: dict[str, list[str]], family: str) -> pd.DataFrame:
    """Per target, order, finite particle count and FvML method: the setting the held-out log-likelihood picks.

    ``settings[target]`` are the candidates. Per replicate the candidate with the largest held-out log-likelihood is
    picked; reported are how often each candidate is picked and the median over replicates of the picked fit's
    distance over the smallest distance among the candidates (1: the pick is the best).
    """
    rows = []
    finite = frame[(frame["n_value"] != INF_POSITION) & (frame["method"] != "histogram")]
    for target, candidates in settings.items():
        if len(candidates) < 2:
            continue
        sub = finite[(finite["target"] == target) & (finite["setting"].isin(candidates))]
        for (order, n, method), group in sub.groupby(["order", "n_value", "method"]):
            picks, regrets = [], []
            for _, replicate in group.groupby("replicate"):
                picked = replicate.loc[replicate["heldout_log_likelihood"].idxmax()]
                picks.append(picked["setting"])
                regrets.append(picked["js_bits_fit"] / replicate["js_bits_fit"].min())
            best = group.groupby("setting")["js_bits_fit"].median().idxmin()
            rows.append(
                {
                    "family": family,
                    "target": target,
                    "order": order,
                    "n_particles": int(n),
                    "method": method,
                    "picked": ";".join(f"{c}:{picks.count(c)}" for c in candidates),
                    "best_by_median_js": best,
                    "median_picked_over_best_js": float(np.median(regrets)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Parse arguments, write the CSV and the figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    everything = read_rows(args.results_dir)
    targets = {name: load_target(args.targets_dir / f"{name}.npz") for name in config["targets"]}
    everything.drop(columns=["fit"]).to_csv(args.out_dir / "summary.csv", index=False)
    present = {t: set(everything.loc[everything["target"] == t, "setting"]) for t in targets}
    lobe_settings = {
        c: {
            t: [s for s in (f"c{c:g}_lobes{x['lobes']}" for x in config["settings"] if x["c"] == c) if s in present[t]]
            for t, target in targets.items()
            if target.mixture is None
        }
        for c in dict.fromkeys(x["c"] for x in config["settings"])
    }
    c_settings = {
        t: list(
            dict.fromkeys(
                s
                for s in (
                    f"c{x['c']:g}_lobes" + ("truth" if target.mixture is not None else str(x["lobes"]))
                    for x in config["settings"]
                    if target.mixture is not None or x["lobes"] == BASE_LOBES
                )
                if s in present[t]
            )
        )
        for t, target in targets.items()
    }
    for c, settings in lobe_settings.items():
        if any(len(v) > 1 for v in settings.values()):
            title = f"Cell fit by lobe count (c = {c:g})"
            plot_settings(everything, settings, "lobes", title, args.out_dir / f"by_lobes_c{c:g}.png")
    plot_settings(
        everything, c_settings, "c", "Cell fit by cap width c (5 lobes or the true count)", args.out_dir / "by_c.png"
    )
    pd.concat(
        [heldout_selection(everything, v, f"lobes_c{c:g}") for c, v in lobe_settings.items()]
        + [heldout_selection(everything, c_settings, "c")]
    ).to_csv(args.out_dir / "selection.csv", index=False)
    base_c = BASE_C if BASE_C in lobe_settings else float(config["settings"][0]["c"])
    frame = everything[
        [s == base_setting(targets[t], base_c) for t, s in zip(everything["target"], everything["setting"])]
    ]
    plot_distance(frame, "js_bits_fit", "Distance to the target on the fit grid", args.out_dir / "js_fit_order.png")
    plot_distance(
        frame, "js_bits_eval", "Distance to the target on its evaluation grid", args.out_dir / "js_eval_order.png"
    )
    plot_relative(
        frame,
        "js_bits_fit",
        "JS distance of the FvML fit over that of the histogram, same draw (below 1: the fit is closer)",
        args.out_dir / "relative_fit_order.png",
    )
    relative_to_histogram(frame, "js_bits_fit").to_csv(args.out_dir / "relative_fit_order.csv", index=False)
    plot_kappa_recovery(frame, targets, args.out_dir / "kappa_recovery.png")
    for target in targets.values():
        plot_maps(frame, target, args.results_dir, base_c, args.out_dir / f"maps_{target.name}.png")


if __name__ == "__main__":
    main()
