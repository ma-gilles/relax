#!/usr/bin/env python
"""Reconstruction quality of the pose-neighborhood experiments: Figures 4 and 5, their arrays and the signed table.

    python -m scripts.plot_pose_neighborhood_quality --root data/pose_neighborhoods --out <dir>

Reads each run's ``eval.json`` (``scripts.evaluate_adaptive_subsets``) and, for Figure 5, its maps. Every map is an
unsharpened half-map average in the body frame; nothing is aligned or scaled to the truth. The signed differences are
selected minus best single candidate and selected minus all particles, in masked MSE (negative: selected is better).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.fsc_metrics import normalized_fsc_auc

METHODS = {"pose": ("C0", "pose"), "pca": ("C1", "pooled PCA")}
ROWS_5K = [("caseA_5k", 13), ("caseA_5k", 1), ("caseB_5k", 13), ("caseB_5k", 1), ("caseC_5k", 13), ("caseC_5k", 1),
           ("caseD_5k", 40), ("caseD_5k", 37)]
LADDER = {
    "true poses, exact nuisance, known noise": "ladder1_truepose_exact_known",
    "true poses, estimated nuisance, known noise": "ladder2_truepose_estnuis_known",
    "true poses, estimated nuisance and noise": "ladder3_truepose_estnuis_estnoise",
    "fitted poses, estimated nuisance and noise": "pose_s1",
}
CHANGING = {"caseB": {1}, "caseC": {1}, "caseD": {37}}


def load(run: Path) -> dict | None:
    path = run / "eval.json"
    return json.loads(path.read_text()) if path.exists() else None


def summary(report: dict, name: str) -> dict:
    """Masked MSE and FSC-AUC of the four maps in one region, with the signed differences."""
    mse = np.asarray(report["masked_mse"][name]["candidates"])
    selected = report["masked_mse"][name]["selected"]
    best = int(np.argmin(mse))
    return {
        "selected": selected, "best": float(mse[best]), "all": float(mse[-1]), "narrowest": float(mse[0]),
        "best_candidate": best, "best_count": int(np.sum(report["counts_per_half"], axis=0)[best]),
        "selected_minus_best": selected - float(mse[best]), "selected_minus_all": selected - float(mse[-1]),
        "slope_all": report["amplitude_slope"][name]["candidates"][-1], "slope_selected": report["amplitude_slope"][name]["selected"],
        **{f"fsc_auc_{key}": float(normalized_fsc_auc(np.asarray(curve))) for key, curve in report["fsc_curves"][name].items()},
    }


def table(root: Path, datasets: list[str]) -> list[dict]:
    rows = []
    for dataset in datasets:
        for run in sorted((root / dataset / "subsets").iterdir()):
            report = load(run)
            if report is None:
                continue
            for name in ("body", "region"):
                rows.append({"dataset": dataset, "run": run.name, "region": name, **summary(report, name)})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(rows[0])
    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in keys) + "\n")


def draw_mse(ax, report, name, color, label):
    counts = np.sum(report["counts_per_half"], axis=0)
    mse = np.asarray(report["masked_mse"][name]["candidates"])
    ax.plot(counts, mse, c=color, marker="o", ms=2, lw=1, label=f"{label} candidates")
    ax.axhline(report["masked_mse"][name]["selected"], c=color, ls="--", lw=1, label=f"{label} selected")


def figure4(root: Path, rows, out: Path, stem: str, runs_of) -> dict:
    """One row per target: MSE in the body, MSE in the changing region where there is one, FSC to truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.5 * len(rows)), squeeze=False)
    arrays = {}
    for r, (dataset, state) in enumerate(rows):
        case = dataset[:5]
        name = "region" if state in CHANGING.get(case, ()) else "body"
        runs = {m: load(root / dataset / "subsets" / run) for m, run in runs_of(dataset, state).items()}
        title = f"{dataset} state {state}"
        for m, report in runs.items():
            if report is None:
                continue
            color, label = METHODS[m]
            draw_mse(axes[r, 0], report, "body", color, label)
            draw_mse(axes[r, 1], report, name, color, label)
            for key, style in (("narrowest", ":"), ("all", "-"), ("best", "--"), ("selected", "-.")):
                curve = report["fsc_curves"][name][key]
                axes[r, 2].plot(np.arange(len(curve)), curve, c=color, ls=style, lw=1, label=f"{label} {key}")
                arrays[f"{dataset}_s{state}_{m}_{name}_fsc_{key}"] = np.asarray(curve)
            arrays[f"{dataset}_s{state}_{m}_counts"] = np.sum(report["counts_per_half"], axis=0)
            arrays[f"{dataset}_s{state}_{m}_mse_body"] = np.asarray(report["masked_mse"]["body"]["candidates"])
            arrays[f"{dataset}_s{state}_{m}_mse_{name}"] = np.asarray(report["masked_mse"][name]["candidates"])
        axes[r, 0].set_title(f"{title}: body mask", fontsize=9)
        axes[r, 1].set_title(f"{title}: {'changing region' if name == 'region' else 'no changing region, body mask'}", fontsize=9)
        axes[r, 2].set_title("FSC to truth, " + ("region" if name == "region" else "body"), fontsize=9)
        axes[r, 0].set_ylabel("masked MSE")
        axes[r, 2].set_ylabel("FSC")
        axes[r, 2].set_ylim(-0.1, 1.02)
    for ax in axes[-1, :2]:
        ax.set_xlabel("particles in subset (both halves)")
    axes[-1, 2].set_xlabel("Fourier shell")
    axes[0, 0].legend(fontsize=6)
    axes[0, 2].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=130)
    plt.close(fig)
    np.savez_compressed(out / f"{stem}_arrays.npz", **arrays)
    return arrays


def sampling_labels(diag: Path, mask: np.ndarray, z: int) -> tuple[np.ndarray, np.ndarray]:
    """Each voxel of slice ``z``'s local-selection point (-1 outside), and which points lie in the body mask. RECOVAR's
    ``sampling.mrc`` stores labels as float16, which cannot hold labels above 2048 exactly, so the points are taken
    from their definition."""
    import pickle

    from recovar.heterogeneity.locres import get_sampling_points

    box = mask.shape[0]
    params = pickle.load(open(diag / "params.pkl", "rb"))
    sampling, voxel = float(params["locres_sampling"]), float(params["voxel_size"])
    points = np.asarray(get_sampling_points(box, sampling, float(params["locres_maskrad"]), voxel)) + box // 2
    half = int(np.round(sampling / voxel)) // 2
    labels = -np.ones((box, box), dtype=np.int64)
    for k in np.flatnonzero(np.abs(points[:, 0] - z) <= half):
        labels[points[k, 1] - half : points[k, 1] + half, points[k, 2] - half : points[k, 2] + half] = k
    return labels, mask[points[:, 0], points[:, 1], points[:, 2]] > 0


def figure5_panel(run: Path, dataset: Path, state: int) -> dict:
    """The maps of one run in a common frame, the slice through the changing region and the selector's choice."""
    import pickle

    from scripts.evaluate_adaptive_subsets import _read, truth_and_masks

    frame = truth_and_masks(run, dataset, state)
    report = load(run)
    name = "region" if frame["region"] is not frame["mask"] else "body"
    best = report["best_mse_candidate"][name]
    n = len(report["thresholds"])
    diag = run / "diagnostics" / "subset000"

    def average(pattern):
        return 0.5 * sum(_read(pattern.format(h=h)) for h in (1, 2))

    maps = {
        "truth": frame["truth"],
        "all": average(str(diag / f"estimates_half{{h}}_unfil/{n - 1:04d}.mrc")),
        f"best ({int(np.sum(report['counts_per_half'], axis=0)[best])})": average(str(diag / f"estimates_half{{h}}_unfil/{best:04d}.mrc")),
        "selected": average(str(run / "subset000_half{h}_unfil.mrc")),
    }
    weight = frame["region"]
    z = int(np.argmax(np.sum(weight * np.abs(frame["truth"]), axis=(1, 2))))
    counts = np.sum(report["counts_per_half"], axis=0)
    choice = np.asarray(pickle.load(open(diag / "split_choice.pkl", "rb"))["split_choice"])
    labels, inside = sampling_labels(diag, frame["mask"], z)
    rows, columns = np.nonzero(frame["mask"][z] > 0)
    crop = (slice(max(rows.min() - 2, 0), rows.max() + 3), slice(max(columns.min() - 2, 0), columns.max() + 3))
    return {
        "slices": {key: (value[z] * frame["mask"][z])[crop] for key, value in maps.items()}, "z": z, "name": name,
        "shell_choice": counts[choice[inside]], "all_points": counts[choice], "counts": counts, "sampling": labels[crop],
    }


def figure5(root: Path, entries: list[tuple], out: Path, stem: str) -> None:
    """Truth, all, best, selected, differences to truth, and where and at which shells the selector chose what."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(label, figure5_panel(root / dataset / "subsets" / run, root / dataset, state)) for label, dataset, run, state in entries]
    fig, axes = plt.subplots(len(panels), 9, figsize=(20, 2.5 * len(panels)), squeeze=False)
    arrays = {}
    limit = max(np.abs(p["slices"]["truth"]).max() for _, p in panels)
    worst = max(np.quantile(np.abs(v - p["slices"]["truth"]), 0.999) for _, p in panels for k, v in p["slices"].items() if k != "truth")
    for row, (label, panel) in zip(axes, panels):
        truth = panel["slices"]["truth"]
        for ax, (key, value) in zip(row[:4], panel["slices"].items()):
            ax.imshow(value, cmap="gray", vmin=-limit, vmax=limit)
            ax.set_title(f"{key}" if key != "truth" else f"{label}\ntruth, z={panel['z']}", fontsize=8)
            arrays[f"{label}_{key}"] = value
        for ax, (key, value) in zip(row[4:7], list(panel["slices"].items())[1:]):
            ax.imshow(value - truth, cmap="RdBu_r", vmin=-worst, vmax=worst)
            ax.set_title(f"{key} - truth", fontsize=8)
        points = panel["sampling"].astype(int)
        lowest = np.median(panel["all_points"][:, :6], axis=1)
        where = np.where(points >= 0, lowest[np.clip(points, 0, None)], np.nan)
        image = row[7].imshow(where, cmap="viridis", vmin=0, vmax=panel["counts"][-1])
        row[7].set_title("median retained count,\nlowest 6 shells", fontsize=8)
        fig.colorbar(image, ax=row[7], fraction=0.046)
        row[8].plot(np.median(panel["shell_choice"], axis=0), marker="o", ms=2)
        row[8].fill_between(np.arange(panel["shell_choice"].shape[1]), *np.quantile(panel["shell_choice"], [0.1, 0.9], axis=0), alpha=0.3)
        row[8].set_title("retained count by shell\n(median, 10-90%, points in body)", fontsize=8)
        for ax in row[:8]:
            ax.axis("off")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=130)
    plt.close(fig)
    np.savez_compressed(out / f"{stem}_arrays.npz", **{k.replace(" ", "_"): v for k, v in arrays.items()})


def noise_rungs(root: Path) -> dict:
    """What estimated noise changes: the known white level against the converted RELION spectrum, and the maps."""
    import pickle

    from relax.multibody.subsets import _noise
    from scripts.evaluate_adaptive_subsets import _read

    runs = root / "caseB_5k" / "subsets"
    known, estimated = runs / "ladder2_truepose_estnuis_known", runs / "ladder3_truepose_estnuis_estnoise"
    a, b = (np.mean([_read(r / f"subset000_half{h}_unfil.mrc") for h in (1, 2)], axis=0) for r in (known, estimated))
    choice = [pickle.load(open(r / "diagnostics/subset000/split_choice.pkl", "rb"))["split_choice"] for r in (known, estimated)]
    record = json.loads((known / "subset_subsets.json").read_text())
    box = 128
    spectrum = 0.5 * sum(_noise(root / "caseB_5k" / "relion_multibody", record["maps_iteration"], box))
    ratio = record["white_noise_per_fourier_pixel"] / spectrum[1 : box // 2]
    return {
        "relative_rms_change_of_selected_map": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "fraction_of_selector_choices_changed": float(np.mean(choice[0] != choice[1])),
        "white_noise_per_fourier_pixel": record["white_noise_per_fourier_pixel"],
        "true_over_converted_noise_by_shell": {"median": float(np.median(ratio)), "min": float(ratio.min()), "max": float(ratio.max())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--large", action="store_true", help="also the 10,000-particle fixtures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.root

    rows = table(root, ["caseA_5k", "caseB_5k", "caseC_5k", "caseD_5k"])
    write_csv(rows, args.out / "signed_differences_5k.csv")
    figure4(root, ROWS_5K, args.out, "figure4_matrix", lambda d, s: {"pose": f"pose_s{s}", "pca": f"pca_s{s}"})
    figure5(root, [
        ("B end pose", "caseB_5k", "pose_s1", 1), ("B end PCA", "caseB_5k", "pca_s1", 1),
        ("C end pose", "caseC_5k", "pose_s1", 1), ("C end PCA", "caseC_5k", "pca_s1", 1),
        ("A central pose", "caseA_5k", "pose_s13", 13), ("D end pose", "caseD_5k", "pose_s37", 37),
    ], args.out, "figure5_matrix")
    (args.out / "noise_rungs.json").write_text(json.dumps(noise_rungs(root), indent=1))
    ladder = []
    for label, run in LADDER.items():
        ladder.append({"rung": label, **summary(load(root / "caseB_5k" / "subsets" / run), "region")})
    write_csv(ladder, args.out / "ladder.csv")

    if args.large:
        datasets = [f"case{c}_10k_n{n}_s{s}" for c in "AB" for n in ("0.1", "0.3") for s in (0, 1)]
        write_csv(table(root, datasets), args.out / "signed_differences_10k.csv")
        for case, state in (("A", 13), ("A", 1), ("B", 13), ("B", 1)):
            figure4(root, [(d, state) for d in datasets if d.startswith(f"case{case}")], args.out, f"figure4_large_{case}_s{state}",
                    lambda d, s: {"pose": f"pose_s{s}", "pca": f"pca_s{s}"})
        figure5(root, [
            (f"B end {n} pose", f"caseB_10k_n{n}_s0", "pose_s1", 1) for n in ("0.1", "0.3")
        ] + [(f"B end {n} PCA", f"caseB_10k_n{n}_s0", "pca_s1", 1) for n in ("0.1", "0.3")], args.out, "figure5_large")


if __name__ == "__main__":
    main()
