"""Compare PPCA runs of one suite dataset with a baseline run, on every metric the evaluation gives, as a markdown table.

Per run (<root>/<arm>/seed<seed>, written by scripts/ppca_schedule/Snakefile): the suite scorer's and direct fit's scores
at one checkpoint (suite_scores.json), the latent scores (latent_scores.json), the training wall time (the run's log) and,
with the dataset's evaluation directory, the power of the mean and of the loadings in three bands of shells against the
truth. Replicates of the baseline (other seeds) set the noise of each metric: the larger of their range and a floor of the
metric's resolution. A run's value is flagged ``+`` or ``-`` when it is better or worse than the baseline by more than
twice that noise, and unflagged otherwise: the heuristic is as good as the number of replicates (two or three here).

Run:
  python scripts/ppca_schedule/compare_runs.py ROOT/s1_ord_int100/seed11 ROOT/s1_lr1p2/seed11 \
      --baseline ROOT/sgd_new_recipe/seed11 --replicate SEEDS/sgd_new_recipe/seed12 --replicate SEEDS/sgd_new_recipe/seed13 \
      --reference ROOT15/vdam_recipe/seed11 --evaluation <suite>/datasets/<dataset>/evaluation \
      --power-reference ROOT15/vdam_recipe/seed11 --checkpoint 3000 -o table.md
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# (column key, header, direction of better: 1 higher, -1 lower, 0 none, scale shown, decimals, noise floor)
METRICS = (
    ("pose_direct_median_deg", "pose med (deg)", -1, 1, 2, 0.2),
    ("pose_direct_within_10deg", "<10 deg (%)", 1, 100, 1, 0.01),
    ("state_fsc_direct", "state FSC", 1, 1, 3, 0.01),
    ("state_fsc_direct_min", "state FSC min", 1, 1, 3, 0.02),
    ("mean_fsc_to_true_mean", "mean FSC", 1, 1, 3, 0.005),
    ("true_energy_captured", "energy", 1, 1, 3, 0.01),
    ("subspace_max_angle_deg", "subspace (deg)", -1, 1, 1, 3.0),
    ("kmeans_ari", "ARI", 1, 1, 3, 0.02),
    ("kmeans_nmi", "NMI", 1, 1, 3, 0.02),
    ("between_state_r2", "R2", 1, 1, 3, 0.02),
    ("pc_max_fraction_outside_true_span", "PC outside span", -1, 1, 3, 0.05),
    ("within_over_between", "within/between", -1, 1, 3, 0.02),
    ("pc_rigid_share_1_15", "rigid share", 0, 1, 3, 0.0),
    ("pc_over_mean_power", "PC/mean", 0, 1, 3, 0.0),
    ("wall_min", "train min", 0, 1, 0, 0.0),
)
# The bands of shells of the power ratios, and the shells over which the model's scale against the truth is set.
BANDS = {"1-4": (1, 4), "5-15": (5, 15), "16-31": (16, 31)}
SCALE_SHELLS = (3, 15)
BAND_METRICS = tuple(
    (f"{kind}_pow_{band}", f"{kind} {band}", 0, 1, 2, 0.0) for kind in ("mean", "pc") for band in BANDS
)


def shell_profile(values: np.ndarray, grid: int) -> np.ndarray:
    """Average of ``values`` (a half volume of the model or a full centred volume) over the shells 0 to 31 of a ``grid`` cube."""
    import jax.numpy as jnp

    from relax.reconstruction.regularization import compute_relion_weight_shell_stats

    stats = compute_relion_weight_shell_stats(
        jnp.asarray(values), (grid,) * 3, r_max=grid // 2 - 1, shell_rounding="round"
    )
    return np.asarray(stats["avg_weight_shells"], np.float64)


def truth_profiles(evaluation: Path, grid: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """The truth's power by shell: the mean's, and the states' heterogeneity (their weighted variance about the mean)."""
    with np.load(evaluation / "truth.npz") as truth:
        volumes, counts = truth["volumes_fourier"], truth["counts"].astype(np.float64)
    weights = counts / counts.sum()
    mean = np.tensordot(weights, volumes, axes=1)
    heterogeneity = sum(w * shell_profile(np.abs(v - mean) ** 2, grid) for w, v in zip(weights, volumes, strict=True))
    return shell_profile(np.abs(mean) ** 2, grid), heterogeneity


def model_profiles(checkpoint: Path, grid: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """The power by shell of a checkpoint's mean column and of its loading columns together (the model's own units)."""
    with np.load(checkpoint, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"])
    power = np.abs(theta) ** 2 / grid**4
    return shell_profile(power[:, 0], grid), shell_profile(power[:, 1:].sum(axis=1), grid)


def model_scale(reference_mean: np.ndarray, truth_mean: np.ndarray) -> float:
    """The model's power per unit of the truth's, from a reference run's mean over ``SCALE_SHELLS``."""
    low, high = SCALE_SHELLS
    return float(np.mean(reference_mean[low : high + 1] / truth_mean[low : high + 1]))


def band_ratios(model: np.ndarray, truth: np.ndarray, scale: float, kind: str) -> dict[str, float]:
    """The model's power over the truth's (after ``scale``) in each band of shells, keyed ``<kind>_pow_<band>``."""
    return {
        f"{kind}_pow_{band}": float(model[lo : hi + 1].sum() / (scale * truth[lo : hi + 1].sum()))
        for band, (lo, hi) in BANDS.items()
    }


def wall_minutes(log: Path) -> float | None:
    """Minutes between the first and the last timestamped line of a training log; None if there is none."""
    stamps = []
    for line in log.read_text(errors="replace").splitlines():
        match = re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", line)
        if match:
            stamps.append(datetime.fromisoformat(match.group()))
    return (stamps[-1] - stamps[0]).total_seconds() / 60 if len(stamps) > 1 else None


def load_run(run: Path, checkpoint: int, profiles=None, latent: bool = True) -> dict[str, float]:
    """The metrics of one run directory ``<root>/<arm>/seed<seed>`` at ``checkpoint``.

    ``profiles`` is ``(truth mean, truth heterogeneity, scale)`` for the power ratios, or None to leave them out.
    ``latent`` keeps the latent scores, which are those of the run's final embeddings (so only meaningful at its last checkpoint).
    """
    row: dict[str, float] = {}
    suite = run / "suite_scores.json"
    if suite.exists():
        row.update(json.loads(suite.read_text())["checkpoints"][str(checkpoint)])
    latent_scores = run / "latent_scores.json"
    if latent and latent_scores.exists():
        row.update(json.loads(latent_scores.read_text()))
    log = run.parent.parent / "logs" / f"{run.parent.name}.{run.name}.train.log"
    if log.exists() and (minutes := wall_minutes(log)) is not None:
        row["wall_min"] = minutes
    if profiles is not None:
        mean, loadings = model_profiles(run / f"checkpoint_{checkpoint:04d}.npz")
        truth_mean, truth_heterogeneity, scale = profiles
        row.update(band_ratios(mean, truth_mean, scale, "mean"))
        row.update(band_ratios(loadings, truth_heterogeneity, scale, "pc"))
    return row


def run_labels(runs: list[Path]) -> dict[Path, str]:
    """Short unique labels for run directories: the shortest trailing part of each path (at least arm/seed) no other run shares."""
    labels = {}
    for run in runs:
        parts = run.parts
        for count in range(2, len(parts) + 1):
            label = "/".join(parts[-count:])
            if sum("/".join(other.parts[-count:]) == label for other in runs) == 1:
                break
        labels[run] = label
    return labels


def noise(replicates: list[dict[str, float]], key: str, floor: float) -> float:
    """The range of ``key`` over the baseline and its replicates, at least ``floor``."""
    values = [row[key] for row in replicates if key in row]
    return max(floor, max(values) - min(values)) if len(values) > 1 else floor


def flag(value: float, baseline: float, noise_level: float, direction: int) -> str:
    """``+`` / ``-`` when ``value`` is better / worse than ``baseline`` by more than twice the noise, else a space."""
    if direction == 0 or abs(value - baseline) <= 2 * noise_level:
        return " "
    return "+" if (value - baseline) * direction > 0 else "-"


def format_table(
    runs: dict[str, dict[str, float]],
    baseline: str,
    replicate_rows: list[dict[str, float]],
    metrics: tuple,
    unflagged: frozenset = frozenset(),
) -> str:
    """A markdown table of ``runs`` (label -> metrics), flagged against ``runs[baseline]`` with the replicates' noise.

    The baseline and the labels in ``unflagged`` (references) carry no flag.
    """
    head = ["run", *(header for _, header, *_ in metrics)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    group = [runs[baseline], *replicate_rows]
    for label, row in runs.items():
        cells = [label]
        for key, _, direction, scale, decimals, floor in metrics:
            if key not in row:
                cells.append("n/a")
                continue
            flagged = label != baseline and label not in unflagged and key in runs[baseline]
            mark = flag(row[key], runs[baseline][key], noise(group, key, floor), direction) if flagged else " "
            cells.append(f"{row[key] * scale:.{decimals}f}{mark}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories <root>/<arm>/seed<seed> to compare")
    parser.add_argument("--baseline", type=Path, required=True, help="the run the others are flagged against")
    parser.add_argument("--replicate", type=Path, action="append", default=[], help="another seed of the baseline arm")
    parser.add_argument("--reference", type=Path, action="append", default=[], help="a run shown without flags")
    parser.add_argument("--checkpoint", type=int, default=3000)
    parser.add_argument("--final-checkpoint", type=int, default=3000, help="the runs' last checkpoint (latent scores belong to it)")
    parser.add_argument("--evaluation", type=Path, help="the dataset's evaluation directory (for the power ratios)")
    parser.add_argument("--power-reference", type=Path, help="a run whose final mean sets the model's scale against the truth")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)

    profiles = None
    metrics = METRICS
    if args.evaluation:
        if not args.power_reference:
            parser.error("--evaluation needs --power-reference")
        truth_mean, truth_heterogeneity = truth_profiles(args.evaluation)
        reference_mean, _ = model_profiles(args.power_reference / f"checkpoint_{args.checkpoint:04d}.npz")
        profiles = (truth_mean, truth_heterogeneity, model_scale(reference_mean, truth_mean))
        metrics = METRICS + BAND_METRICS

    shown = [*args.reference, args.baseline, *args.runs]
    labels = run_labels(shown)
    runs = {labels[run]: load_run(run, args.checkpoint, profiles, args.checkpoint == args.final_checkpoint) for run in shown}
    replicates = [load_run(run, args.checkpoint, profiles, args.checkpoint == args.final_checkpoint) for run in args.replicate]
    baseline = labels[args.baseline]
    text = format_table(runs, baseline, replicates, metrics, frozenset(labels[run] for run in args.reference))
    group = [runs[baseline], *replicates]
    text += "\nNoise (range of the baseline and its replicates, at least the floor): " + ", ".join(
        f"{header} {noise(group, key, floor) * scale:.3g}" for key, header, direction, scale, _, floor in metrics if direction
    )
    text += "\n"
    print(text)
    if args.output:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
