"""Collect the refinement replays (``replay_refinement.py``) into tables and figures.

Writes, under ``--out-dir``:
  * ``replay.csv``: every row of every replay, the fitted parameters dropped;
  * ``strategies.csv``: per run, c, held-out share and start mode, over both halves: the final fit's Jensen-Shannon
    distance (bits) to the population and RELION's learned prior's, medians over the last ``LAST`` iterations; how
    often the held-out choice is the candidate closest to the population and how much farther it is otherwise; the
    seconds per iteration (all candidates plus the final refit), the iterations of a candidate fit and each candidate
    lobe count's distance (``js_bits_candidate_<lobes>_last``, fitted on the training share);
  * ``js_by_iteration.png``: distance to the population per iteration, one panel per run and half, with the
    ``FIXED_LOBES`` candidate of the first curve as a fixed lobe count;
  * ``lobes_by_iteration.png``: the chosen lobe count per iteration;
  * ``seconds_by_iteration.png``: seconds per iteration of the warm and cold modes.

Run (Snakemake calls it as rule summarize_replay):
    python scripts/fvml_prior_simulation/summarize_replay.py --replay-dir <output_root>/replay \
        --out-dir <output_root>/replay/summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LAST = 3  # final iterations the strategy table summarizes
# (c, held-out share, mode) drawn in the figures, with colour and label
CURVES = [
    (0.25, 0.2, "warm", "#eb6834", "FvML warm, c 0.25"),
    (0.25, 0.2, "cold", "#2a78d6", "FvML cold, c 0.25"),
    (1.0, 0.2, "warm", "#1baf7a", "FvML warm, c 1"),
]
GREY = "#52514e"
FIXED_LOBES = 5  # candidate drawn as a fixed lobe count (its fit on the training share)


def read_rows(replay_dir: Path) -> pd.DataFrame:
    """All rows of ``replay_dir/<run>/*.jsonl``."""
    rows = [
        json.loads(line)
        for path in sorted(replay_dir.glob("*/c*_f*_half*.jsonl"))
        for line in path.read_text().splitlines()
        if line
    ]
    if not rows:
        raise FileNotFoundError(f"no replay rows under {replay_dir}")
    frame = pd.DataFrame(rows)
    frame["heldout_log_likelihood"] = [float(v) for v in frame["heldout_log_likelihood"]]
    return frame


def iteration_cost(frame: pd.DataFrame) -> pd.DataFrame:
    """Seconds per iteration (every candidate and the final refit) per run, half, c, share and mode."""
    fits = frame[frame["kind"].isin(["candidate", "final"])]
    keys = ["run", "half", "c", "heldout_fraction", "mode", "iteration"]
    return fits.groupby(keys, as_index=False)["seconds"].sum()


def selection_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Per iteration: the chosen candidate's distance over the smallest among the candidates (1: chosen is closest)."""
    candidates = frame[frame["kind"] == "candidate"]
    keys = ["run", "half", "c", "heldout_fraction", "mode", "iteration"]
    rows = []
    for key, group in candidates.groupby(keys):
        chosen = group.loc[group["heldout_log_likelihood"].idxmax()]
        rows.append({**dict(zip(keys, key)), "chosen_over_best_js": chosen["js_bits"] / group["js_bits"].min()})
    return pd.DataFrame(rows)


def strategies(frame: pd.DataFrame) -> pd.DataFrame:
    """The strategy table described in the module docstring."""
    final = frame[frame["kind"] == "final"]
    baseline = frame[frame["kind"] == "baseline"]
    candidates = frame[frame["kind"] == "candidate"]
    cost = iteration_cost(frame)
    quality = selection_quality(frame)
    rows = []
    for (run, c, share, mode), group in final.groupby(["run", "c", "heldout_fraction", "mode"]):
        last = sorted(group["iteration"].unique())[-LAST:]
        tail = group[group["iteration"].isin(last)]
        base = baseline[
            (baseline["run"] == run)
            & (baseline["c"] == c)
            & (baseline["heldout_fraction"] == share)
            & baseline["iteration"].isin(last)
        ]
        here = {"run": run, "c": c, "heldout_fraction": share, "mode": mode}
        mask = np.logical_and.reduce([cost[k] == v for k, v in here.items()])
        q = quality[np.logical_and.reduce([quality[k] == v for k, v in here.items()])]
        cand = candidates[np.logical_and.reduce([candidates[k] == v for k, v in here.items()])]
        rows.append(
            {
                **here,
                "js_bits_final_last": float(tail["js_bits"].median()),
                "js_bits_learned_prior_last": float(base["js_bits_learned_prior"].median()),
                "js_bits_histogram_last": float(base["js_bits_histogram"].median()),
                "lobes_chosen_last": ";".join(str(int(v)) for v in tail.sort_values(["half", "iteration"])["lobes"]),
                "chosen_is_closest": float(np.mean(q["chosen_over_best_js"] <= 1.0 + 1e-9)),
                "chosen_over_best_js_median": float(q["chosen_over_best_js"].median()),
                "chosen_over_best_js_max": float(q["chosen_over_best_js"].max()),
                "seconds_per_iteration_median": float(cost.loc[mask, "seconds"].median()),
                "seconds_per_iteration_max": float(cost.loc[mask, "seconds"].max()),
                "seconds_total_per_half": float(cost.loc[mask, "seconds"].sum() / group["half"].nunique()),
                "candidate_iterations_median": float(cand["n_iter"].median()),
                "candidate_lobes_at_cap_median": float(cand["lobes_at_cap"].median()),
                **{
                    f"js_bits_candidate_{int(lobes)}_last": float(group_c["js_bits"].median())
                    for lobes, group_c in cand[cand["iteration"].isin(last)].groupby("lobes")
                },
            }
        )
    return pd.DataFrame(rows)


def _curve(frame: pd.DataFrame, c: float, share: float, mode: str) -> pd.DataFrame:
    """The final rows of one (c, share, mode) combination."""
    return frame[
        (frame["kind"] == "final") & (frame["c"] == c) & (frame["heldout_fraction"] == share) & (frame["mode"] == mode)
    ]


def plot_panels(frame: pd.DataFrame, draw, ylabel: str, title: str, path: Path, log: bool = True) -> None:
    """One panel per run (rows) and half (columns); ``draw(ax, rows of that run and half)`` fills a panel."""
    runs = sorted(frame["run"].unique())
    halves = sorted(frame["half"].unique())
    fig, axes = plt.subplots(len(runs), len(halves), figsize=(6 * len(halves), 3.2 * len(runs)), squeeze=False)
    for i, run in enumerate(runs):
        for j, half in enumerate(halves):
            ax = axes[i, j]
            draw(ax, frame[(frame["run"] == run) & (frame["half"] == half)])
            if log:
                ax.set_yscale("log")
            ax.grid(True, which="major", color="#e4e3dc", lw=0.6)
            ax.set_title(f"{run}, half {half}", fontsize=9)
            ax.set_xlabel("iteration")
            if j == 0:
                ax.set_ylabel(ylabel)
        axes[i, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_js(ax: plt.Axes, rows: pd.DataFrame) -> None:
    """Distance to the population per iteration: RELION's learned prior, the histogram and the FvML curves."""
    base = rows[(rows["kind"] == "baseline") & (rows["c"] == CURVES[0][0]) & (rows["heldout_fraction"] == CURVES[0][1])]
    base = base.sort_values("iteration")
    ax.plot(
        base["iteration"], base["js_bits_learned_prior"], color=GREY, ls="--", marker="s", ms=3, label="RELION prior"
    )
    ax.plot(base["iteration"], base["js_bits_histogram"], color=GREY, ls=":", lw=1, label="histogram of poses")
    for c, share, mode, color, label in CURVES:
        sub = _curve(rows, c, share, mode).sort_values("iteration")
        if not sub.empty:
            ax.plot(sub["iteration"], sub["js_bits"], color=color, lw=1.8, marker="o", ms=3, label=label)
    c, share, mode, color, label = CURVES[0]
    fixed = rows[
        (rows["kind"] == "candidate")
        & (rows["c"] == c)
        & (rows["heldout_fraction"] == share)
        & (rows["mode"] == mode)
        & (rows["lobes"] == FIXED_LOBES)
    ].sort_values("iteration")
    if not fixed.empty:
        ax.plot(
            fixed["iteration"],
            fixed["js_bits"],
            color=color,
            ls="--",
            lw=1.2,
            label=f"{label}, {FIXED_LOBES} lobes fixed",
        )


def draw_lobes(ax: plt.Axes, rows: pd.DataFrame) -> None:
    """The chosen lobe count per iteration, for the FvML curves and the 10 % share."""
    for c, share, mode, color, label in CURVES:
        for s, style in ((share, "-"), (0.1, ":")):
            sub = _curve(rows, c, s, mode).sort_values("iteration")
            if not sub.empty:
                ax.plot(
                    sub["iteration"], sub["lobes"], color=color, ls=style, marker="o", ms=3, label=f"{label}, {s:g}"
                )


def draw_seconds(ax: plt.Axes, rows: pd.DataFrame) -> None:
    """Seconds per iteration (all candidates and the final refit) for the FvML curves."""
    cost = iteration_cost(rows)
    for c, share, mode, color, label in CURVES:
        sub = cost[(cost["c"] == c) & (cost["heldout_fraction"] == share) & (cost["mode"] == mode)]
        if not sub.empty:
            ax.plot(sub["iteration"], sub["seconds"], color=color, lw=1.8, marker="o", ms=3, label=label)


def main() -> None:
    """Parse arguments, write the tables and the figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = read_rows(args.replay_dir)
    frame.drop(columns=["fit"]).to_csv(args.out_dir / "replay.csv", index=False)
    strategies(frame).to_csv(args.out_dir / "strategies.csv", index=False)
    plot_panels(
        frame,
        draw_js,
        "JS distance (bits)",
        "Distance to the simulated population, per refinement iteration (20 % held out)",
        args.out_dir / "js_by_iteration.png",
    )
    plot_panels(
        frame,
        draw_lobes,
        "lobes chosen",
        "Lobe count chosen on the held-out share (solid 20 %, dotted 10 %)",
        args.out_dir / "lobes_by_iteration.png",
        log=False,
    )
    plot_panels(
        frame,
        draw_seconds,
        "seconds",
        "Seconds per iteration: every candidate plus the final refit (one core)",
        args.out_dir / "seconds_by_iteration.png",
    )


if __name__ == "__main__":
    main()
