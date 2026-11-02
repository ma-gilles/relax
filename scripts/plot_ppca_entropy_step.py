"""Where the pose-entropy step falls, against the VDAM schedule, the fudge and the HEALPix order of the run.

Reads the output roots of scripts/ppca_schedule (rounds 3 to 7), prints for every run of the chosen panel set the
iteration of the first pose entropy below 8, 3 and 2, the batch, the VDAM step and the radius at the last of them, the
iteration the radius first reaches 15 and first exceeds it, its final value, and the final score, and writes
entropy_<set>.png: pose entropy over the iterations, one row per seed and one column per panel. The panel sets:

  step    the table and data_vs_prior at 1200 against 2400 iterations, and the data_vs_prior increment 5, 10, 20
          (docs/math/vdam_ppca_algorithm.md, 18.7);
  fudge   a constant fudge of 1 against InitialModel's own fudge schedule;
  order   data_vs_prior at HEALPix order 3 throughout against order 4 from iteration 700;
  acc     those two against the accuracy order rule started at order 3 and at order 1 (18.10); triangles on the
          axis mark the iterations where the order of a run changes.

The schedule of VDAM (batch ramp, step) follows --iterations, so a longer run is a stretched run; the dotted lines
mark where the batch starts and stops ramping. Colors: the baseline gray, the others the first slots of the
reference categorical palette.

Run (CPU, a few minutes):
  python scripts/plot_ppca_entropy_step.py --panels fudge --root ROOT/round3 --root ROOT/round4 --root ROOT/round6 \
      -o OUT_DIR
"""

import argparse
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ppca_schedule import last_trajectory  # noqa: E402
from plot_ppca_schedule_results import AQUA, BLUE, GRAY, INK_2, ORANGE, SEEDS, style  # noqa: E402

from relax.steps.sgd_schedule import compute_phase_lengths  # noqa: E402


@dataclass(frozen=True)
class Arm:
    """One arm of the figure: its run directory name, its legend entry and its line color."""

    name: str
    label: str
    color: str


PANEL_SETS = {
    "step": (
        (
            "table, q = 1",
            (Arm("long_q1_control", "1200 iterations", GRAY), Arm("table_q1_2400", "2400 iterations", INK_2)),
        ),
        (
            "table, q = 3",
            (Arm("long_q3_control", "1200 iterations", GRAY), Arm("table_q3_2400", "2400 iterations", INK_2)),
        ),
        (
            "data_vs_prior, q = 1",
            (Arm("dvp_1200", "1200 iterations", GRAY), Arm("dvp_2400", "2400 iterations", ORANGE)),
        ),
        (
            "data_vs_prior, 1200 iterations",
            (
                Arm("dvp_1200_incr5", "increment 5", BLUE),
                Arm("dvp_1200", "increment 10", GRAY),
                Arm("dvp_1200_incr20", "increment 20", AQUA),
            ),
        ),
    ),
    "fudge": (
        (
            "table, q = 1",
            (Arm("long_q1_control", "fudge 1", GRAY), Arm("long_q1_default_fudge", "default fudge", BLUE)),
        ),
        (
            "data_vs_prior, q = 1",
            (Arm("dvp_1200", "fudge 1", GRAY), Arm("dvp_1200_default_fudge", "default fudge", BLUE)),
        ),
        (
            "data_vs_prior, q = 3",
            (Arm("dvp_q3_1200", "fudge 1", GRAY), Arm("dvp_q3_1200_default_fudge", "default fudge", BLUE)),
        ),
    ),
    "order": (
        (
            "data_vs_prior, q = 1",
            (Arm("dvp_1200", "order 3", GRAY), Arm("dvp_1200_hp4_at_700", "order 4 from iteration 700", BLUE)),
        ),
    ),
    "acc": (
        (
            "data_vs_prior, q = 1",
            (
                Arm("dvp_1200", "order 3", GRAY),
                Arm("dvp_1200_hp4_at_700", "order 4 from iteration 700", BLUE),
                Arm("acc_1200_start3", "accuracy rule, start order 3", ORANGE),
                Arm("acc_1200_start1", "accuracy rule, start order 1", AQUA),
            ),
        ),
    ),
}


def find_run(roots: list[Path], arm: str, seed: int) -> Path | None:
    """The directory of ``arm``'s run for ``seed`` in the first root that has it (None if no root does)."""
    return next((d for r in roots if (d := r / arm / f"seed{seed}").is_dir()), None)


@functools.cache
def read_run(directory: Path) -> dict[str, np.ndarray]:
    """Per iteration: the mean pose entropy, the batch (images of both halves), the VDAM step, the radius and the order."""
    rows = last_trajectory([json.loads(line) for line in open(directory / "iterations.jsonl")])
    step = [r["step"][0] if isinstance(r["step"], list) else r["step"] for r in rows]
    return {
        "iteration": np.array([r["iteration"] for r in rows]),
        "entropy": np.array([np.mean([p["pose_entropy_mean"] for p in r["posterior"]]) for r in rows]),
        "batch": np.array([sum(r["half_counts"]) for r in rows]),
        "step": np.array(step, float),
        "radius": np.array([r["radius"] for r in rows]),
        "order": np.array([r["healpix_order"] for r in rows]),
    }


def first_below(run: dict[str, np.ndarray], threshold: float) -> int | None:
    """The first iteration with a mean pose entropy below ``threshold`` (None if there is none)."""
    below = run["entropy"] < threshold
    return int(run["iteration"][np.argmax(below)]) if below.any() else None


def order_changes(run: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """The (iteration, order) at which the HEALPix order of the run first takes each of its values."""
    changed = np.flatnonzero(np.diff(run["order"]) != 0) + 1
    return [(int(run["iteration"][0]), int(run["order"][0]))] + [
        (int(run["iteration"][i]), int(run["order"][i])) for i in changed
    ]


def print_table(roots: list[Path], panels: tuple) -> None:
    """One line per run: the entropy crossings, the schedule at the last one, the radius milestones, the final score."""
    print(
        f"{'arm':26s} seed  iters  ramp          <8     <3     <2   <2/iters  batch  step  radius  r>=15  r>15  "
        f"r_end  final  R2  orders (iteration: order)\n"
        f"{'':26s}                (start, end)                                    at <2          at    at"
    )
    for panel in panels:
        for arm in panel[1]:
            for seed in SEEDS:
                directory = find_run(roots, arm.name, seed)
                if directory is None:
                    continue
                run = read_run(directory)
                total = int(run["iteration"][-1])
                phases = compute_phase_lengths(total)
                ramp = (phases.grad_ini_iter, phases.grad_ini_iter + phases.grad_inbetween_iter)
                t8, t3, t2 = (first_below(run, x) for x in (8, 3, 2))
                at = np.flatnonzero(run["iteration"] == t2)[0] if t2 else None
                wide = run["iteration"][run["radius"] >= 15]
                opened = run["iteration"][run["radius"] > 15]
                scores = directory / "latent_scores.json"
                r2 = json.loads(scores.read_text())["between_state_r2"] if scores.exists() else float("nan")
                print(
                    f"{arm.name:26s} {seed}  {total:5d}  {ramp!s:13s} {t8!s:>6s} {t3!s:>6s} {t2!s:>6s}  "
                    f"{(t2 / total if t2 else float('nan')):7.2f}  "
                    f"{run['batch'][at] if at is not None else '-':>5}  "
                    f"{run['step'][at] if at is not None else float('nan'):4.2f}  "
                    f"{run['radius'][at] if at is not None else '-':>6}  {int(wide[0]) if wide.size else None!s:>5s}  "
                    f"{int(opened[0]) if opened.size else None!s:>4s}  {int(run['radius'][-1]):>5d}  {run['entropy'][-1]:5.2f}  {r2:.3f}  "
                    + " ".join(f"{i}:{o}" for i, o in order_changes(run))
                )


def figure(roots: list[Path], panels: tuple, out: Path) -> None:
    """Pose entropy against the iteration: a row per seed, a column per panel, the batch ramp dotted."""
    fig, axes = plt.subplots(
        len(SEEDS), len(panels), figsize=(max(6.0, 4.0 * len(panels)), 8.6), sharey=True, squeeze=False
    )
    for row, seed in enumerate(SEEDS):
        for col, (title, arms) in enumerate(panels):
            ax = axes[row, col]
            for arm in arms:
                directory = find_run(roots, arm.name, seed)
                if directory is None:
                    continue
                run = read_run(directory)
                total = int(run["iteration"][-1])
                phases = compute_phase_lengths(total)
                ax.plot(run["iteration"], run["entropy"], color=arm.color, lw=1.5)
                for x in (phases.grad_ini_iter, phases.grad_ini_iter + phases.grad_inbetween_iter):
                    ax.axvline(x, color=arm.color, lw=0.9, ls=(0, (1, 2)))
                t2 = first_below(run, 2)
                if t2:
                    ax.plot([t2], [2], "o", color=arm.color, ms=6, mec="white", mew=0.8)
                changes = order_changes(run)[1:]
                if changes:
                    ax.plot([i for i, _ in changes], [0.25] * len(changes), "^", color=arm.color, ms=5)
            ax.set_ylim(0, 13)
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(title, loc="left", color=INK_2, fontsize=10)
                ax.legend(
                    handles=[plt.Line2D([], [], color=a.color, lw=2, label=a.label) for a in arms],
                    loc="upper right",
                    fontsize=8,
                )
            if col == 0:
                ax.set_ylabel(f"seed {seed}\npose entropy", color=INK_2)
            if row == len(SEEDS) - 1:
                ax.set_xlabel("iteration")
    fig.suptitle(
        "Pose entropy against the iteration (dots: first entropy below 2; triangles: the order changes; dotted lines: the "
        "batch ramp of each run)",
        x=0.06,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Print the table and draw the figure into the output directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", choices=sorted(PANEL_SETS), default="step", help="the panel set to print and draw")
    parser.add_argument("--root", type=Path, action="append", required=True, help="an output_root; repeat per round")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    panels = PANEL_SETS[args.panels]
    print_table(args.root, panels)
    style()
    figure(args.root, panels, args.out / f"entropy_{args.panels}.png")
    print(f"\nwrote entropy_{args.panels}.png to {args.out}")


if __name__ == "__main__":
    main()
