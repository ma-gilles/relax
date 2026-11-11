#!/usr/bin/env python3
"""Run-internal report of PPCA runs on a real data set with and without the per-particle contrast latent (no ground truth).

Reads, per run directory, ``iterations.jsonl`` and ``embeddings.npz`` (all particles) and writes a Markdown table and figures:

* the final state of each run: radius, HEALPix order, singular values of the loadings, mean and loading power, mean gate of the
  last update, wall time;
* for a run with the contrast latent: the trajectory of the prior sd, of the population gain and of the estimates' mean and
  spread, the histogram of the contrast estimates, the share of particles below the contrast floor;
* ``--star`` (a RELION STAR) and ``--group-column`` (default ``_rlnMicrographName``; cahra_01's micrograph names are all one, its
  ``_rlnDefocusAngle`` takes 4308 values of about 330 particles each, one per micrograph): the intraclass correlation of the contrast
  estimate and of every latent coordinate over the groups, ``(between - within) / (between + (n - 1) within)`` by one-way ANOVA. Ice thickness is shared by the
  particles of a micrograph, so a contrast estimate that tracks it has a high ICC; a latent coordinate that carries contrast in
  a run without the latent has one too;
* for every pair (run without the latent, run with it, same label suffix): how well the contrast estimate of the second is
  predicted linearly by the latents of the first (R^2), and the canonical correlations of the two latent spaces.

Run (CPU; a few GB of memory for 1.6 million particles):
    python scripts/ppca_contrast/cahra_contrast_report.py RUN [RUN ...] --star particles.star [--group-column _rlnDefocusAngle] -o report_dir
Runs are named by their directory; a pair is ``none_X`` and ``latent_X``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def read_groups(star: Path, column: str) -> np.ndarray:
    """The ``column`` (a STAR label such as ``_rlnMicrographName``) of every row of the STAR's particle table, in row order, as integer
    codes of its distinct values."""
    columns, skip = [], 0
    with open(star) as stream:
        in_particles = False
        for number, line in enumerate(stream):
            text = line.strip()
            if text.startswith("data_"):
                in_particles = text == "data_particles"
                columns = []
            elif in_particles and text.startswith("_rln"):
                columns.append(text.split()[0])
                skip = number + 1
            elif in_particles and columns and text and not text.startswith(("#", "loop_")):
                break
    index = columns.index(column)
    names = pd.read_csv(star, sep=r"\s+", header=None, skiprows=skip, usecols=[index], dtype=str, comment="#").iloc[:, 0]
    return pd.factorize(names)[0]


def intraclass_correlation(values: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1) of ``values`` over ``groups`` (integer codes), by one-way ANOVA with unequal group sizes."""
    counts = np.bincount(groups)
    keep = counts > 0
    sums = np.bincount(groups, weights=values)
    means = sums[keep] / counts[keep]
    grand = values.mean()
    n_groups, n_total = int(keep.sum()), len(values)
    between = float(np.sum(counts[keep] * (means - grand) ** 2)) / (n_groups - 1)
    within = float(np.sum((values - (sums / np.maximum(counts, 1))[groups]) ** 2)) / (n_total - n_groups)
    n0 = (n_total - np.sum(counts[keep] ** 2) / n_total) / (n_groups - 1)
    return (between - within) / (between + (n0 - 1) * within)


def linear_r2(predictors: np.ndarray, target: np.ndarray) -> float:
    """R^2 of the least-squares fit of ``target`` (n,) on ``predictors`` (n, k) with an intercept."""
    design = np.column_stack([predictors, np.ones(len(predictors))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return 1.0 - float(np.sum(residual**2) / np.sum((target - target.mean()) ** 2))


def canonical_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Canonical correlations of two coordinate sets ``(n, p)`` and ``(n, r)`` (cosines of the principal angles)."""
    qa, _ = np.linalg.qr(a - a.mean(axis=0))
    qb, _ = np.linalg.qr(b - b.mean(axis=0))
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)


def load_run(path: Path) -> dict:
    """One run's log records and embeddings."""
    records = [json.loads(line) for line in (path / "iterations.jsonl").read_text().splitlines()]
    with np.load(path / "embeddings.npz", allow_pickle=False) as saved:
        embeddings = {key: saved[key] for key in saved.files}
    return {"name": path.name, "records": records, "embeddings": embeddings}


def final_state_row(run: dict) -> dict:
    """The run's end state from its last log record."""
    last = run["records"][-1]
    gates = np.asarray(last["gates"], float)
    row = {
        "run": run["name"], "iterations": last["iteration"], "radius": last["radius"], "order": last["healpix_order"],
        "mean power": last["mean_power"], "loading power": last["loading_power"],
        "loading sv": ", ".join(f"{v:.0f}" for v in last["loading_singular_values"]),
        "mean gate min / loading gate min": f"{gates[:, 0][gates[:, 0] > 0].min():.2f} / {gates[:, 1][gates[:, 1] > 0].min():.2f}",
        "wall (h)": sum(r["elapsed_seconds"] for r in run["records"]) / 3600,
    }
    contrast = last.get("contrast")
    if contrast:
        row.update({"contrast sd": contrast["next_sd"], "estimate mean": contrast["estimate_mean"], "estimate sd": contrast["estimate_sd"]})
        row["gain"] = contrast["gain"]
    embeddings = run["embeddings"]
    if "contrast" in embeddings:
        row["below floor"] = int(embeddings["n_contrast_below_floor"])
    return row


def contrast_trajectory_figure(runs: list[dict], path: Path) -> None:
    """Prior sd, gain, and the estimates' mean and spread over the iterations of every run with the latent."""
    figure, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for run in runs:
        steps = [(r["iteration"], r["contrast"]) for r in run["records"] if "contrast" in r]
        if not steps:
            continue
        x = [s[0] for s in steps]
        for axis, key in zip(axes, ("sd", "gain", "estimate_mean", "estimate_sd")):
            axis.plot(x, [s[1][key] for s in steps], label=run["name"], lw=0.9)
            axis.set_title(key)
            axis.set_xlabel("iteration")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def contrast_histogram_figure(runs: list[dict], path: Path) -> None:
    """Histogram of the final contrast estimates of every run with the latent."""
    latent_runs = [r for r in runs if "contrast" in r["embeddings"]]
    if not latent_runs:
        return
    figure, axis = plt.subplots(figsize=(5, 3.6))
    for run in latent_runs:
        axis.hist(run["embeddings"]["contrast"], bins=200, histtype="step", label=run["name"], density=True)
    axis.set_xlabel("contrast estimate")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    """Write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--star", type=Path, help="particle STAR in the order the runs read it")
    parser.add_argument("--group-column", default="_rlnMicrographName", help="STAR label whose values are the groups of the ICC")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(path) for path in args.runs]
    groups = read_groups(args.star, args.group_column) if args.star else None
    lines = ["## Final state", ""]
    rows = [final_state_row(run) for run in runs]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines += ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(c) is None else (f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c])) for c in columns) + " |")
    if groups is not None:
        lines += ["", f"## Intraclass correlation over {args.group_column}", "", "| run | quantity | ICC |", "| --- | --- | --- |"]
        for run in runs:
            ids = run["embeddings"]["particle_ids"]
            if "contrast" in run["embeddings"]:
                lines.append(f"| {run['name']} | contrast estimate | {intraclass_correlation(run['embeddings']['contrast'].astype(float), groups[ids]):.3f} |")
            for k in range(run["embeddings"]["z"].shape[1]):
                lines.append(f"| {run['name']} | z{k + 1} | {intraclass_correlation(run['embeddings']['z'][:, k].astype(float), groups[ids]):.3f} |")
    by_name = {run["name"]: run for run in runs}
    pairs = [(by_name[n], by_name["latent_" + n.removeprefix("none_")]) for n in by_name if n.startswith("none_") and "latent_" + n.removeprefix("none_") in by_name]
    if pairs:
        lines += ["", "## Does a run without the latent carry the contrast in its latents?", "",
                  "| pair | R^2 of the contrast estimate on the latents without the latent | canonical correlations of the two latent spaces |", "| --- | --- | --- |"]
        for plain, latent in pairs:
            order_plain, order_latent = np.argsort(plain["embeddings"]["particle_ids"]), np.argsort(latent["embeddings"]["particle_ids"])
            z_plain, z_latent = plain["embeddings"]["z"][order_plain].astype(float), latent["embeddings"]["z"][order_latent].astype(float)
            estimate = latent["embeddings"]["contrast"][order_latent].astype(float)
            cc = canonical_correlations(z_plain, z_latent)
            lines.append(f"| {plain['name']} vs {latent['name']} | {linear_r2(z_plain, estimate):.3f} | {', '.join(f'{c:.2f}' for c in cc)} |")
    contrast_trajectory_figure(runs, args.output / "contrast_trajectory.png")
    contrast_histogram_figure(runs, args.output / "contrast_histogram.png")
    text = "\n".join(lines) + "\n"
    (args.output / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
