"""How close the mean of a `relax ppca_initial_model` run stays to a given truth mean, along its checkpoints.

The run must share the truth map's frame (it started from it with --init-mean-map, or its poses are known). For every
checkpoint the half-Fourier mean is compared with the truth map's spectrum, masked as the model is (the support of
initialization.with_mean_volume): the correlation in each shell (FSC), and over the shells the mean occupies (its
active radius) the correlation, the relative error |mean - truth| / |truth| and the power ratio; and in each band of
shells BANDS the mean's power ratio and correlation and the loadings' total power relative to the truth mean's. Writes
to OUT:
  mean_vs_truth.json  those numbers per checkpoint
  mean_vs_truth.png   FSC curves along the run; in-band FSC, relative error and power ratio; active radius and the
                      shell where the FSC falls below 0.5; the power ratio, the FSC and the loading power by band

Run (CPU, seconds per run):
  python scripts/plot_ppca_mean_vs_truth.py RUN_DIR --truth-map truth_mean.mrc -o RUN_DIR/plots [--every 50]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recovar.core import fourier_transform_utils as ftu  # noqa: E402
from recovar.utils.helpers import load_relion_volume  # noqa: E402

from relax.ppca_initial_model.initialization import support_mask, with_mean_volume  # noqa: E402

INK = "#1a1a1a"
INK_2 = "#555555"
GRID = "#e4e4e4"
SERIES = ("#2a6fbb", "#c4572a", "#3a8f5c")  # in-band FSC, relative error, power ratio
FSC_LEVEL = 0.5
BANDS = ((1, 4), (5, 8), (9, 12), (13, 15), (16, 22), (23, 31))  # shell ranges of the per-band numbers


def shell_correlation(a: np.ndarray, b: np.ndarray, shells: np.ndarray, n_shells: int) -> np.ndarray:
    """Real correlation of two half-Fourier volumes in each shell ``0..n_shells - 1`` (nan where a shell is empty)."""
    out = np.full(n_shells, np.nan)
    for shell in range(n_shells):
        keep = shells == shell
        norm = np.sqrt(np.vdot(a[keep], a[keep]).real * np.vdot(b[keep], b[keep]).real)
        if keep.any() and norm > 0:
            out[shell] = np.real(np.vdot(b[keep], a[keep])) / norm
    return out


def active_radius(mean: np.ndarray, shells: np.ndarray) -> int:
    """The outermost shell in which the mean column has a nonzero coefficient (the model is band-limited to the radius)."""
    return int(shells[np.abs(mean) > 0].max())


def in_band_numbers(mean: np.ndarray, truth: np.ndarray, shells: np.ndarray, radius: int) -> dict[str, float]:
    """Correlation, relative error and power ratio of ``mean`` against ``truth`` over the shells ``0..radius``."""
    keep = shells <= radius
    a, b = mean[keep], truth[keep]
    return {
        "fsc": float(np.real(np.vdot(b, a)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)),
        "relative_error": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "power_ratio": float(np.vdot(a, a).real / np.vdot(b, b).real),
    }


def band_numbers(
    mean: np.ndarray,
    loadings: np.ndarray,
    truth: np.ndarray,
    shells: np.ndarray,
    radius: int,
    bands: tuple[tuple[int, int], ...] = BANDS,
) -> dict[str, list[float]]:
    """Per band: the mean's power relative to the truth's, its correlation with the truth, and the loadings' total power
    relative to the truth mean's. A band wholly above ``radius``, or where the truth has no power, is nan."""
    power_ratio, fsc, loading_ratio = [], [], []
    for first, last in bands:
        keep = (shells >= first) & (shells <= last)
        truth_power = np.vdot(truth[keep], truth[keep]).real
        mean_power = np.vdot(mean[keep], mean[keep]).real
        if first > radius or truth_power == 0:
            power_ratio.append(np.nan)
            fsc.append(np.nan)
            loading_ratio.append(np.nan)
            continue
        power_ratio.append(float(mean_power / truth_power))
        fsc.append(
            float(np.real(np.vdot(truth[keep], mean[keep])) / np.sqrt(mean_power * truth_power))
            if mean_power
            else np.nan
        )
        loading_ratio.append(float(np.sum(np.abs(loadings[keep]) ** 2) / truth_power))
    return {"band_power_ratio": power_ratio, "band_fsc": fsc, "band_loading_ratio": loading_ratio}


def checkpoint_iterations(run_dir: Path, every: int) -> list[int]:
    """The iterations of the run's checkpoints that are multiples of ``every`` (and the first), in order."""
    found = sorted(
        int(re.fullmatch(r"checkpoint_(\d+)\.npz", p.name).group(1)) for p in run_dir.glob("checkpoint_*.npz")
    )
    return [k for k in found if k % every == 0 or k == found[0]]


def measure(run_dir: Path, truth_map: Path, every: int) -> list[dict]:
    """One row per chosen checkpoint: its iteration, active radius, in-band numbers and per-shell FSC."""
    run = json.loads((run_dir / "run.json").read_text())
    volume = load_relion_volume(str(truth_map))
    shape = volume.shape
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(shape), np.int32).reshape(-1)
    half = int(np.prod(ftu.volume_shape_to_half_volume_shape(shape)))
    mask = support_mask(shape[0], run["input"]["particle_diameter_ang"] / voxel_size(truth_map))
    truth = np.asarray(with_mean_volume(jnp.zeros((half, 1), jnp.complex64), volume, volume_shape=shape, mask=mask))[
        :, 0
    ]
    rows = []
    for k in checkpoint_iterations(run_dir, every):
        with np.load(run_dir / f"checkpoint_{k:04d}.npz", allow_pickle=False) as saved:
            theta = np.asarray(saved["theta"])
        mean, loadings = theta[:, 0], theta[:, 1:]
        radius = active_radius(mean, shells)
        rows.append(
            {
                "iteration": k,
                "active_radius": radius,
                **in_band_numbers(mean, truth, shells, radius),
                **band_numbers(mean, loadings, truth, shells, radius),
                "fsc_by_shell": shell_correlation(mean, truth, shells, shape[0] // 2).tolist(),
            }
        )
    return rows


def voxel_size(path: Path) -> float:
    """The voxel size in Angstrom of an MRC map."""
    with mrcfile.open(str(path), header_only=True, permissive=True) as handle:
        return float(handle.voxel_size.x)


def half_fsc_shell(fsc_by_shell: list[float]) -> int:
    """The first shell (from 1) where the correlation falls below ``FSC_LEVEL`` (the last shell if it never does)."""
    values = np.asarray(fsc_by_shell, float)
    below = np.flatnonzero(~(values[1:] >= FSC_LEVEL))
    return int(below[0] + 1) if len(below) else len(values) - 1


def style(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    """Recessive grid and axes, secondary ink for labels."""
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def plot(rows: list[dict], title: str, out: Path) -> None:
    """Six panels: FSC by shell along the run, in-band numbers by iteration, active radius and the FSC 0.5 shell; below
    them the power ratio, the FSC and the loadings' power relative to the truth mean's, by band of shells."""
    iterations = np.array([r["iteration"] for r in rows])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), constrained_layout=True)
    axes, bottom = axes[0], axes[1]
    picks = np.unique(np.linspace(0, len(rows) - 1, min(7, len(rows))).round().astype(int))
    ramp = plt.cm.Blues(np.linspace(0.3, 0.95, len(picks)))
    for color, index in zip(ramp, picks):
        axes[0].plot(rows[index]["fsc_by_shell"], color=color, lw=1.3, label=f"it {rows[index]['iteration']}")
    axes[0].axhline(FSC_LEVEL, color=INK_2, lw=0.6, ls=":")
    axes[0].set_ylim(-0.2, 1.02)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK_2, ncol=2)
    style(axes[0], "shell", "FSC of the mean with the truth mean")
    for color, key, label in zip(
        SERIES, ("fsc", "relative_error", "power_ratio"), ("FSC in band", "relative error", "power ratio")
    ):
        axes[1].plot(iterations, [r[key] for r in rows], color=color, lw=1.4, label=label)
    axes[1].axhline(1.0, color=INK_2, lw=0.6, ls=":")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    style(axes[1], "iteration", "over the shells the mean occupies")
    axes[2].step(
        iterations, [r["active_radius"] for r in rows], where="post", color=SERIES[0], lw=1.4, label="active radius"
    )
    axes[2].plot(
        iterations,
        [half_fsc_shell(r["fsc_by_shell"]) for r in rows],
        color=SERIES[1],
        lw=1.4,
        label="FSC < 0.5 from shell",
    )
    axes[2].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    style(axes[2], "iteration", "shell")
    ramp = plt.cm.Blues(np.linspace(0.3, 0.95, len(BANDS)))
    for ax, key, ylabel in zip(
        bottom,
        ("band_power_ratio", "band_fsc", "band_loading_ratio"),
        ("mean power / truth mean power", "FSC with the truth mean", "loading power / truth mean power"),
    ):
        for color, (first, last), values in zip(ramp, BANDS, zip(*[r[key] for r in rows])):
            ax.plot(iterations, values, color=color, lw=1.3, label=f"shells {first}-{last}")
        style(ax, "iteration", ylabel)
    bottom[0].axhline(1.0, color=INK_2, lw=0.6, ls=":")
    bottom[1].axhline(FSC_LEVEL, color=INK_2, lw=0.6, ls=":")
    bottom[1].set_ylim(-0.2, 1.02)
    bottom[0].legend(frameon=False, fontsize=7, labelcolor=INK_2, ncol=2)
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Measure the run's mean against the truth map and write the JSON and the figure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--truth-map", type=Path, required=True, help="the truth mean as a relax map (--init-mean-map)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=50, help="checkpoint spacing in iterations")
    args = parser.parse_args()
    rows = measure(args.run_dir, args.truth_map, args.every)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mean_vs_truth.json").write_text(json.dumps(rows) + "\n")
    plot(
        rows,
        f"{args.run_dir.parent.name} {args.run_dir.name}: mean against the truth mean",
        args.out / "mean_vs_truth.png",
    )
    last = rows[-1]
    print(
        f"iteration {last['iteration']}: radius {last['active_radius']}, in-band FSC {last['fsc']:.3f}, "
        f"relative error {last['relative_error']:.3f}, power ratio {last['power_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
