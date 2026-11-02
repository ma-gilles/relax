#!/usr/bin/env python
"""Score an adaptive-subset run (``scripts.multibody_adaptive_subsets``) against the synthetic truth.

    python -m scripts.evaluate_adaptive_subsets --run <out dir> --data <dataset dir> --truth-state 3 \\
        [--multibody <MultiBody dir>] --json report.json --plot curve.png

For every candidate bandwidth (``diagnostics/<prefix>000/estimates_half<h>_unfil/NNNN.mrc``) and for RECOVAR's
selected maps (``<prefix>000_half<h>_unfil.mrc``): the FSC area and the mean squared error, in the target body's mask,
of each half map against the true target-body map (``body2_states/volNNNN.mrc`` of ``--truth-state``, else the body
alone at the dataset's mean pose, ``bodies_mean/``, or at the neutral pose, ``bodies/``), moved to the body frame the
candidates are in. The truth is the simulator's own: ``synthetic_dataset.load_ground_truth_volumes`` applies the
recorded volume scale and solvent/B-factor operator to the file, so the error compares matched amplitudes. RECOVAR's own criterion per candidate is its split-shell
opposite-half error summed over every region and shell (``split_choice.pkl``), and its argmin is the single
bandwidth it would choose for the whole body. With ``--multibody``, the all-particle candidate is also compared with
that run's soft-posterior body map, a frame check: the two estimators differ but must agree in geometry.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from scripts.fsc_metrics import normalized_fsc_auc, shell_fsc


def _read(path) -> np.ndarray:
    """A volume RECOVAR wrote (``recovar.utils.write_mrc``), in the recovar frame."""
    from recovar.utils.helpers import load_mrc

    return np.asarray(load_mrc(str(path)), dtype=np.float64)


def calibrated_truth(sim_info: dict, path_root: str, index: int, box: int) -> np.ndarray:
    """Real volume ``index`` of the files ``<path_root>NNNN.mrc`` as the simulator projected it."""
    import jax.numpy as jnp
    from recovar.core import fourier_transform_utils as ftu
    from recovar.simulation import synthetic_dataset

    volumes = synthetic_dataset.load_ground_truth_volumes(sim_info, path_root)
    return np.asarray(ftu.get_idft3(jnp.asarray(volumes[index]).reshape((box, box, box))).real, dtype=np.float64)


def masked_mse(volume, truth, weight) -> float:
    """``sum w^2 (v - t)^2 / sum w^2`` in the units of the maps."""
    return float(np.sum((weight * (volume - truth)) ** 2) / np.sum(weight**2))


def inside_nyquist_sphere(volume) -> np.ndarray:
    """``volume`` with the Fourier corners beyond the Nyquist radius removed. The reconstruction stops at that radius,
    while a truth map from atoms has power in the corners (about 11% here), which no reconstruction can match."""
    n = volume.shape[0]
    grid = np.meshgrid(*[np.fft.fftfreq(n)] * 2, np.fft.rfftfreq(n), indexing="ij")
    radius = np.sqrt(sum(g**2 for g in grid)) * n
    return np.fft.irfftn(np.fft.rfftn(volume) * (radius < n // 2), s=volume.shape)


def amplitude_slope(volume, truth, weight) -> float:
    """Least-squares slope of the map on the truth in the mask: below one when the map's amplitude is low."""
    return float(np.sum(weight**2 * volume * truth) / np.sum(weight**2 * truth**2))


def truth_and_masks(run_dir: Path, data_dir: Path, truth_state: int | None, prefix: str = "subset") -> dict:
    """The calibrated truth, body mask and changing region, all in the candidates' body frame, and the run's record."""
    from recovar import utils

    from relax.multibody.bodies import read_bodies
    from relax.multibody.subtraction import recovar_mask, relion_to_recovar_axes, translate_zero_fill

    record = json.loads((run_dir / f"{prefix}_subsets.json").read_text())
    target = int(record["target_body"]) - 1
    motion = json.loads((data_dir / "body_motion.json").read_text())
    body = read_bodies(data_dir / "bodies.star", pixel_size=float(motion["voxel_size"]))[target]
    to_body = -relion_to_recovar_axes(body.com)
    mask = translate_zero_fill(recovar_mask(body), to_body)
    # The true body maps of a moving dataset are at its mean pose; bodies/ of the spliceosome sets is the unswung one.
    folder = "bodies_mean" if (data_dir / "bodies_mean").is_dir() else "bodies"
    sim_info = utils.pickle_load(data_dir / "simulation_info.pkl")
    box = int(sim_info["grid_size"])
    if truth_state is None:
        truth_root, truth_index = str(data_dir / folder / "vol"), target
    else:
        truth_root, truth_index = str(data_dir / "body2_states" / "vol"), truth_state
    truth_path = Path(f"{truth_root}{truth_index:04d}.mrc")
    truth = translate_zero_fill(calibrated_truth(sim_info, truth_root, truth_index, box), to_body)
    # The region that changes between states: where the target state differs from the undeformed body (the
    # deformation benchmark), dilated and softened; the body mask when there is no deformation.
    region = mask
    neutral_path = data_dir / folder / f"vol{target:04d}.mrc"
    if truth_state is not None and neutral_path.exists():
        from scipy import ndimage

        diff = np.abs(_read(truth_path) - _read(neutral_path))
        if diff.max() > 0:
            core = ndimage.binary_dilation(diff > 0.1 * diff.max(), iterations=3)
            region = translate_zero_fill(ndimage.gaussian_filter(core.astype(np.float64), 1.5), to_body) * (mask > 0)
    return dict(record=record, target=target, to_body=to_body, truth=truth, truth_path=truth_path, mask=mask, region=region)


def evaluate(run_dir: Path, data_dir: Path, truth_state: int | None, multibody: Path | None, prefix: str = "subset") -> dict:
    from recovar.utils.helpers import load_relion_volume

    from relax.multibody.subtraction import translate_zero_fill

    frame = truth_and_masks(run_dir, data_dir, truth_state, prefix)
    record, target, to_body, truth, truth_path, mask, region = (
        frame[k] for k in ("record", "target", "to_body", "truth", "truth_path", "mask", "region"))

    def score(volume, weight=None):
        weight = mask if weight is None else weight
        fsc = shell_fsc(volume * weight, truth * weight)
        return float(normalized_fsc_auc(fsc))

    diag = run_dir / "diagnostics" / f"{prefix}000"
    n = len(record["thresholds"])
    candidate_maps = [[_read(diag / f"estimates_half{h + 1}_unfil" / f"{k:04d}.mrc") for k in range(n)] for h in range(2)]
    selected_maps = [_read(run_dir / f"{prefix}000_half{h + 1}_unfil.mrc") for h in range(2)]
    candidates = [[score(m) for m in maps] for maps in candidate_maps]
    selected = [score(m) for m in selected_maps]
    region_candidates = np.mean([[score(m, region) for m in maps] for maps in candidate_maps], axis=0)
    region_selected = float(np.mean([score(m, region) for m in selected_maps]))
    regions = (("body", mask), ("region", region))
    mse = {
        name: {
            "candidates": np.mean([[masked_mse(m, truth, w) for m in maps] for maps in candidate_maps], axis=0).tolist(),
            "selected": float(np.mean([masked_mse(m, truth, w) for m in selected_maps])),
        }
        for name, w in regions
    }
    errors = np.asarray(pickle.load(open(diag / "split_choice.pkl", "rb"))["ml_errors"], dtype=np.float64)
    criterion = errors.reshape(errors.shape[0], -1).sum(axis=1)
    mean_auc = np.mean(candidates, axis=0)
    best = int(np.argmax(mean_auc))
    chosen = int(np.argmin(criterion))
    report = {
        "run": str(run_dir), "truth": str(truth_path), "target_body": target + 1,
        "thresholds": record["thresholds"], "counts_per_half": record["counts_per_half"],
        "candidate_masked_auc": candidates, "selected_masked_auc": selected,
        "criterion_total": criterion.tolist(),
        "global_choice": chosen, "best_candidate": best,
        "all_particle_auc": float(mean_auc[-1]), "narrow_core_auc": float(mean_auc[0]),
        "best_auc": float(mean_auc[best]), "global_choice_auc": float(mean_auc[chosen]),
        "local_selected_auc": float(np.mean(selected)),
        "excess_error_global": float(mean_auc[best] - mean_auc[chosen]),
        "excess_error_local": float(mean_auc[best] - np.mean(selected)),
        "region_candidate_auc": region_candidates.tolist(), "region_selected_auc": region_selected,
        "region_best_candidate": int(np.argmax(region_candidates)),
        "region_global_choice_auc": float(region_candidates[chosen]),
        "region_all_particle_auc": float(region_candidates[-1]), "region_narrow_core_auc": float(region_candidates[0]),
        "region_best_auc": float(region_candidates.max()),
        "masked_mse": mse, "truth_rms_in_body_mask": float(np.sqrt(masked_mse(truth, 0.0, mask))),
        "fsc_region_selected_half1": np.asarray(shell_fsc(selected_maps[0] * region, truth * region)).tolist(),
        "fsc_body_selected_half1": np.asarray(shell_fsc(selected_maps[0] * mask, truth * mask)).tolist(),
    }
    def half_fsc(maps, weight):
        return np.mean([shell_fsc(m * weight, truth * weight) for m in maps], axis=0).tolist()

    # FSC curves and amplitude slopes of the four maps every comparison reports; the best candidate is by masked MSE.
    picks = {name: {"narrowest": 0, "all": n - 1, "best": int(np.argmin(mse[name]["candidates"]))} for name, _ in regions}
    report["best_mse_candidate"] = {name: picks[name]["best"] for name in picks}
    report["fsc_curves"] = {
        name: {**{key: half_fsc([maps[k] for maps in candidate_maps], w) for key, k in picks[name].items()},
               "selected": half_fsc(selected_maps, w)}
        for name, w in regions
    }
    report["amplitude_slope"] = {
        name: {"candidates": np.mean([[amplitude_slope(m, truth, w) for m in maps] for maps in candidate_maps], axis=0).tolist(),
               "selected": float(np.mean([amplitude_slope(m, truth, w) for m in selected_maps]))}
        for name, w in regions
    }
    # The same measures against the truth the reconstruction can reach; relative MSE divides by the truth's power in
    # the mask, which removes the volume scale that differs between noise levels.
    limited = inside_nyquist_sphere(truth)
    report["nyquist_truth"] = {
        name: {"relative_mse_candidates": (np.mean([[masked_mse(m, limited, w) for m in maps] for maps in candidate_maps], axis=0) / masked_mse(limited, 0.0, w)).tolist(),
               "relative_mse_selected": float(np.mean([masked_mse(m, limited, w) for m in selected_maps]) / masked_mse(limited, 0.0, w)),
               "slope_candidates": np.mean([[amplitude_slope(m, limited, w) for m in maps] for maps in candidate_maps], axis=0).tolist(),
               "slope_selected": float(np.mean([amplitude_slope(m, limited, w) for m in selected_maps]))}
        for name, w in regions
    }
    if multibody is not None:
        posterior = translate_zero_fill(load_relion_volume(str(multibody / f"run_body{target + 1:03d}.mrc")), to_body)
        endpoint = 0.5 * sum(_read(diag / f"estimates_half{h + 1}_unfil" / f"{n - 1:04d}.mrc") for h in range(2))
        report["all_particle_vs_multibody_map_masked_auc"] = float(normalized_fsc_auc(shell_fsc(endpoint * mask, posterior * mask)))
    return report


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.sum(report["counts_per_half"], axis=0)
    auc = np.mean(report["candidate_masked_auc"], axis=0)
    criterion = np.asarray(report["criterion_total"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    top.plot(counts, 1.0 - auc, marker="o", ms=3, label="body mask")
    if "region_candidate_auc" in report:
        top.plot(counts, 1.0 - np.asarray(report["region_candidate_auc"]), marker="s", ms=3, label="changing region")
    top.axvline(counts[report["best_candidate"]], ls=":", c="k", label="best in body (truth)")
    top.axvline(counts[report["global_choice"]], ls="--", c="C3", label="RECOVAR global choice")
    top.axhline(1.0 - report["local_selected_auc"], ls="-.", c="C2", label="RECOVAR local selection (body)")
    if "region_candidate_auc" in report:
        top.axvline(counts[report["region_best_candidate"]], ls=":", c="C1", label="best in region (truth)")
        top.axhline(1.0 - report["region_selected_auc"], ls="-.", c="C1", label="RECOVAR local selection (region)")
    top.set_ylabel("1 - masked FSC AUC vs truth")
    top.legend(fontsize=8)
    bottom.plot(counts, criterion - criterion.min(), marker="o", ms=3, c="C3")
    bottom.set_ylabel("RECOVAR criterion (relative)")
    bottom.set_xlabel("particles in subset (both halves)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--truth-state", type=int, default=None)
    parser.add_argument("--multibody", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.run, args.data, args.truth_state, args.multibody)
    print({k: v for k, v in report.items() if not isinstance(v, list)})
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))
    if args.plot is not None:
        plot(report, args.plot)


if __name__ == "__main__":
    main()
