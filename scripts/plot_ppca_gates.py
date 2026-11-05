"""Spectrogram of the VDAM shell gates of a `relax ppca_initial_model` run, from its iterations.jsonl.

Two panels share the iteration axis: the mean's gate and the loadings' gate (one gate per shell, shared by
every PC), each a (Fourier shell x iteration) image coloured by the gate value in [0, 1]. A gate near 1 takes
the update fully (the two pseudo-halves agree relative to the signal); near 0 it shrinks the shell towards
zero. Shells beyond the update's Fourier radius have no gate and are grey; a real zero (a shell whose halves
disagree with no signal, or zero disagreement) is dark. White dashed lines mark stage changes (radius or
HEALPix order). A top panel shows the fudge and step schedules that scale the gates and the step.

iterations.jsonl is appended to, so it can hold several runs; iteration 1 starts a run, and a later repeat of
an iteration (a resume) supersedes the earlier lines from there on. The last run is plotted unless --run says
otherwise. Only --optimizer vdam runs have gates. For a run that is still going, --in-progress ignores an
incomplete last line (and plots nothing before the first complete iteration).

Run (CPU, a few seconds):
  python scripts/plot_ppca_gates.py RUN_DIR_OR_JSONL \
      [--run -1] [--in-progress] [-o OUT.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KEYS = ("iteration", "radius", "healpix_order", "gates", "fudge", "step")


def read_records(path: Path, allow_partial_last_line: bool = False) -> list[dict]:
    """The lines of an iterations.jsonl reduced to the fields this script plots.

    A run still writing the file can leave an incomplete last line; ``allow_partial_last_line`` drops it.
    Any other line that does not parse is an error.
    """
    records, unparsed = [], None
    with open(path) as stream:
        for line in stream:
            if unparsed is not None:
                raise unparsed  # the line before this one did not parse and was not the last
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                if not allow_partial_last_line:
                    raise
                unparsed = error
                continue
            if "gates" not in row:
                raise ValueError(f"{path} has no shell gates (only --optimizer vdam records them)")
            records.append({key: row[key] for key in KEYS})
    return records


def split_runs(records: list[dict]) -> list[list[dict]]:
    """Group records into runs: iteration 1 starts a run; a repeated later iteration supersedes its successors."""
    runs: list[list[dict]] = []
    for record in records:
        k = record["iteration"]
        if k == 1 or not runs:
            runs.append([record])
        else:
            runs[-1] = [r for r in runs[-1] if r["iteration"] < k] + [record]
    return runs


def gate_images(run: list[dict]) -> np.ndarray:
    """Gates as ``(2, shells, iterations)`` (mean, loadings); shells beyond each update's radius are NaN."""
    gates = np.stack([np.asarray(r["gates"], np.float64) for r in run], axis=-1).transpose(1, 0, 2)
    radius = np.array([r["radius"] for r in run])
    active = np.arange(gates.shape[1])[:, None] <= radius[None, :]
    return np.where(active[None], gates, np.nan)


def stage_starts(run: list[dict]) -> list[tuple[int, str]]:
    """``(iteration, label)`` of each update whose radius or HEALPix order differs from the previous one's."""
    return [
        (b["iteration"], f"r={b['radius']}, HP{b['healpix_order']}")
        for a, b in zip(run, run[1:])
        if (a["radius"], a["healpix_order"]) != (b["radius"], b["healpix_order"])
    ]


def plot_gates(run: list[dict], title: str, path: Path) -> None:
    """Write the mean and loadings gate spectrograms, with the fudge and step schedules above them."""
    iterations = np.array([r["iteration"] for r in run])
    if not np.array_equal(iterations, np.arange(iterations[0], iterations[-1] + 1)):
        raise ValueError("The run's iterations are not contiguous")
    images = gate_images(run)
    top = max(r["radius"] for r in run)
    starts = stage_starts(run)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")  # shells beyond the update's radius

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1, 3, 3]}, layout="constrained"
    )
    schedule = axes[0]
    schedule.plot(iterations, [r["fudge"] for r in run], color="#c44e52", label="fudge")
    schedule.set_ylabel("fudge", color="#c44e52")
    step = schedule.twinx()
    step.plot(iterations, [r["step"] for r in run], color="#4c72b0", label="step")
    step.set_ylabel("step", color="#4c72b0")
    for iteration, label in starts:
        schedule.text(iteration - 0.5, 1.02, label, transform=schedule.get_xaxis_transform(), fontsize=8, ha="left")
    for ax, image, name in zip(axes[1:], images, ("mean", "loadings")):
        shown = ax.imshow(
            np.ma.masked_invalid(image[: top + 1]), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
            interpolation="nearest", extent=(iterations[0] - 0.5, iterations[-1] + 0.5, -0.5, top + 0.5),
        )
        ax.set_ylabel(f"{name}: Fourier shell")
    for ax in axes:
        for iteration, _ in starts:
            ax.axvline(iteration - 0.5, color="white" if ax is not schedule else "gray", lw=0.9, ls="--")
    axes[-1].set_xlabel("iteration")
    fig.colorbar(shown, ax=axes[1:], label="gate (0: shrink to zero, 1: take the update fully)", shrink=0.9)
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_run_gates(
    jsonl: Path, output: Path, run_index: int = -1, last_iteration: int | None = None, in_progress: bool = False
) -> None:
    """Plot the gate spectrogram of run ``run_index`` of ``jsonl`` to the PNG ``output``, printing which run it took.

    ``last_iteration`` (for example the final checkpoint's) must be the chosen run's last iteration, so the
    figure belongs to the model it is shown with; otherwise a ValueError is raised.
    ``in_progress`` is for a run that is still writing ``jsonl``: an incomplete last line is ignored, and with
    no complete iteration yet a message is printed and nothing is written.
    """
    records = read_records(jsonl, allow_partial_last_line=in_progress) if jsonl.exists() or not in_progress else []
    if in_progress and not records:
        print(f"{jsonl.name}: no complete iteration logged yet; gates skipped")
        return
    runs = split_runs(records)
    ranges = [f"{r[0]['iteration']}-{r[-1]['iteration']}" for r in runs]
    index = run_index % len(runs)
    print(f"{jsonl.name}: {len(runs)} run(s), iterations {ranges}; plotting run {index} ({ranges[index]})")
    ends = runs[index][-1]["iteration"]
    if last_iteration is not None and ends != last_iteration:
        raise ValueError(f"The chosen run of {jsonl} ends at iteration {ends}, but the checkpoint is iteration {last_iteration}")
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_gates(runs[index], f"VDAM shell gates | {jsonl.parent.name} | iterations {ranges[index]}", output)
    print(f"Wrote {output}")


def main() -> None:
    """Parse the command line and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="run directory (with iterations.jsonl) or an iterations.jsonl")
    parser.add_argument("--run", type=int, default=-1, help="which run of the file to plot (default -1: the last)")
    parser.add_argument("-o", "--output", type=Path, help="output PNG (default: <run dir>/plots/gates_spectrogram.png)")
    parser.add_argument("--in-progress", action="store_true", help="the run is still writing the file: ignore an incomplete last line")
    args = parser.parse_args()

    jsonl = args.source / "iterations.jsonl" if args.source.is_dir() else args.source
    plot_run_gates(jsonl, args.output or jsonl.parent / "plots" / "gates_spectrogram.png", args.run, in_progress=args.in_progress)


if __name__ == "__main__":
    main()
