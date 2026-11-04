"""The expected rotation accuracy along a `relax ppca_initial_model` run that sets its HEALPix order from it.

A run with --order-rule accuracy estimates, every --accuracy-interval iterations and in the last, the expected rotation
accuracy of its model at the poses the E-step just found (docs/math/vdam_ppca_algorithm.md, 18.10): `acc_rot` under the
divergence of the PPCA model and `acc_rot_mean_only` under the mean image alone. The next iteration runs one order
finer while the angular step of the order in use is at least 0.75 of the one the run reads (--accuracy-signal). Reads
RUN_DIR/run.json and iterations.jsonl and writes to OUT:
  pose_accuracy.json  one record per estimate: iteration, order, next order, radius, both accuracies, the divergence terms
  pose_accuracy.png   the accuracies against the angular step of the order and the rule's 0.75 threshold; the divergence
                      terms at the crossing; the order in use and the one planned; the radius
A run with the stages table (--order-rule stages) makes no estimate and is refused.

Run (CPU; iterations.jsonl is read once, about a minute for the 13 GB of a run that reaches order 4):
  python scripts/plot_ppca_pose_accuracy.py RUN_DIR -o RUN_DIR/plots
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_ppca_schedule import last_trajectory  # noqa: E402
from plot_ppca_schedule_results import AQUA, BLUE, GRAY, INK, INK_2, ORANGE, style  # noqa: E402

from relax.ppca_initial_model.order_rule import ACCURACY_FRACTION, angular_step_deg  # noqa: E402

KEPT = ("iteration", "healpix_order", "next_order", "radius", "pose_accuracy")
ITERATION = re.compile(r'"iteration": (\d+)')
TERMS = ("mean", "log_det", "trace")


def read_signal(run_dir: Path) -> str:
    """The accuracy the run's order rule reads (``divergence`` or ``mean_only``); refuses a run without the rule."""
    config = json.loads((run_dir / "run.json").read_text())["config"]
    if config["order_rule"] != "accuracy":
        raise ValueError(
            f"{run_dir} uses the order rule {config['order_rule']!r}: only --order-rule accuracy estimates the accuracy"
        )
    return config["accuracy_signal"]


def read_estimates(path: Path) -> list[dict]:
    """The records with a ``pose_accuracy`` estimate of the last run in an iterations.jsonl, in iteration order.

    Only those lines are parsed, and only ``KEPT`` of them is kept; every other line gives its iteration, which is
    what a resume (a repeated iteration) needs to supersede earlier lines.
    """
    rows = []
    with open(path) as stream:
        for number, line in enumerate(stream, 1):
            if '"pose_accuracy"' in line:
                record = json.loads(line)
                rows.append({key: record[key] for key in KEPT if key in record})
                continue
            found = ITERATION.search(line)
            if found is None:
                raise ValueError(f"{path}, line {number}: no iteration")
            rows.append({"iteration": int(found.group(1))})
    estimates = [row for row in last_trajectory(rows) if "pose_accuracy" in row]
    if not estimates:
        raise ValueError(f"{path} holds no pose_accuracy estimate")
    return estimates


def tabulate(estimates: list[dict]) -> list[dict]:
    """One flat record per estimate: the iteration, the order in use and planned, the radius, both accuracies in
    degrees, the angular step of the order in degrees and the divergence terms at the crossing."""
    return [
        {
            "iteration": row["iteration"],
            "order": row["healpix_order"],
            "next_order": row["next_order"],
            "radius": row["radius"],
            "step_deg": angular_step_deg(row["healpix_order"]),
            "acc_rot": row["pose_accuracy"]["acc_rot"],
            "acc_rot_mean_only": row["pose_accuracy"]["acc_rot_mean_only"],
            **{term: row["pose_accuracy"]["terms_at_error"][term] for term in TERMS},
            "n_trials": row["pose_accuracy"]["n_trials"],
        }
        for row in estimates
    ]


def series(table: list[dict], key: str) -> np.ndarray:
    """The ``key`` column of the table as floats."""
    return np.array([row[key] for row in table], float)


def plot(table: list[dict], signal: str, title: str, out: Path) -> None:
    """Four panels over the iterations: the accuracies against the order's step and the rule's threshold, the
    divergence terms, the order in use and planned, the radius."""
    style()
    iterations = series(table, "iteration")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(iterations, series(table, "acc_rot"), color=BLUE, lw=1.5, label="acc_rot (divergence)")
    ax.plot(iterations, series(table, "acc_rot_mean_only"), color=ORANGE, lw=1.5, label="acc_rot_mean_only")
    ax.step(iterations, series(table, "step_deg"), where="post", color=GRAY, lw=1.5, label="angular step of the order")
    ax.plot(
        iterations,
        ACCURACY_FRACTION * series(table, "acc_rot" if signal == "divergence" else "acc_rot_mean_only"),
        color=BLUE if signal == "divergence" else ORANGE,
        lw=1.0,
        ls="--",
        label=f"{ACCURACY_FRACTION} x {signal} (refine while the step is above)",
    )
    ax.set_yscale("log")
    ax.set_ylim(top=3.5 * max(series(table, "acc_rot").max(), series(table, "acc_rot_mean_only").max()))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 4, 5, 7)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("degrees")
    ax.legend(fontsize=8, labelcolor=INK_2, loc="upper right")
    for key, color in zip(TERMS, (BLUE, ORANGE, AQUA)):
        axes[0, 1].plot(iterations, series(table, key), color=color, lw=1.4, label=key)
    axes[0, 1].set_ylabel("divergence term at the crossing")
    axes[0, 1].legend(fontsize=8, labelcolor=INK_2)
    axes[1, 0].step(iterations, series(table, "order"), where="post", color=BLUE, lw=1.5, label="order in use")
    axes[1, 0].step(
        iterations, series(table, "next_order"), where="post", color=ORANGE, lw=1.0, ls="--", label="planned"
    )
    axes[1, 0].set_ylabel("HEALPix order")
    axes[1, 0].set_yticks(range(int(series(table, "order").min()), int(series(table, "next_order").max()) + 1))
    axes[1, 0].legend(fontsize=8, labelcolor=INK_2)
    axes[1, 1].plot(iterations, series(table, "radius"), color=BLUE, lw=1.5)
    axes[1, 1].set_ylim(0, 1.1 * series(table, "radius").max())
    axes[1, 1].set_ylabel("Fourier radius (shells)")
    for ax in axes[1]:
        ax.set_xlabel("iteration")
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Read the run's estimates and write the table and the figure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    signal = read_signal(args.run_dir)
    table = tabulate(read_estimates(args.run_dir / "iterations.jsonl"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pose_accuracy.json").write_text(json.dumps(table) + "\n")
    plot(
        table,
        signal,
        f"{args.run_dir.parent.name} {args.run_dir.name}: expected rotation accuracy (rule reads {signal})",
        args.out / "pose_accuracy.png",
    )
    first, last = table[0], table[-1]
    print(
        f"{len(table)} estimates, iterations {first['iteration']} to {last['iteration']}: acc_rot {first['acc_rot']:.2f} "
        f"to {last['acc_rot']:.2f} deg, mean-only {first['acc_rot_mean_only']:.2f} to {last['acc_rot_mean_only']:.2f}, "
        f"order {first['order']} to {last['next_order']}"
    )


if __name__ == "__main__":
    main()
