"""Evaluation-only per-state FSC report for the three-state VDAM/PPCA pilot.

See docs/math/vdam_ppca_algorithm.md, section 14, for the state reconstruction.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.utils.helpers import load_mrc, load_relion_volume, write_mrc
from scipy.optimize import linear_sum_assignment

from relax.diagnostics.gt_metrics import relion_alignment_rotations
from relax.diagnostics.gt_registration import RigidVolumeTransform, align_volume_rigid_to_reference
from relax.ppca_initial_model.checkpoint import file_hash
from scripts.fsc_metrics import first_shell_below, normalized_fsc_auc, shell_fsc


def _bandlimit(volume, radius):
    n = volume.shape[0]
    shells = ftu.get_grid_of_radial_distances(volume.shape, rounded=False)
    return np.asarray(ftu.get_idft3(ftu.get_dft3(volume) * (shells <= radius)).real, np.float32)


def _active_band(curve, box, voxel_size, radius):
    band = np.asarray(curve[: radius + 1], np.float64)
    auc = normalized_fsc_auc(band)
    thresholds = {}
    for threshold in (0.143, 0.5):
        shell = first_shell_below(band, threshold)
        thresholds[str(threshold)] = {
            "first_shell_below": shell,
            "resolution_ang": float(box * voxel_size / (shell if shell else radius)),
            "censored_at_cutoff": shell is None,
        }
    return {
        "shell_min": 1,
        "shell_max": radius,
        "nominal_cutoff_ang": float(box * voxel_size / radius),
        "fsc_auc": float(auc) if np.isfinite(auc) else None,
        "resolution": thresholds,
    }


def _fsc_report(candidate, reference, voxel_size, mask, active_radius=None):
    report = {}
    for name, weight in (("raw", 1.0), ("spherical_mask", mask)):
        curve = shell_fsc(candidate * weight, reference * weight)
        auc = normalized_fsc_auc(curve)
        thresholds = {}
        for threshold in (0.143, 0.5):
            shell = first_shell_below(curve, threshold)
            thresholds[str(threshold)] = {
                "first_shell_below": shell,
                "resolution_ang": None if shell is None else float(candidate.shape[0] * voxel_size / shell),
            }
        report[name] = {
            "curve": [float(value) if np.isfinite(value) else None for value in curve],
            "fsc_auc": float(auc) if np.isfinite(auc) else None,
            "resolution": thresholds,
        }
        if active_radius is not None:
            active_curve = shell_fsc(_bandlimit(candidate, active_radius) * weight, _bandlimit(reference, active_radius) * weight)
            report[name]["active_band"] = _active_band(active_curve, candidate.shape[0], voxel_size, active_radius)
            report[name]["above_trained_support_from_shell"] = active_radius + 1
    return report


def _fit(moving, reference, rotations, voxel_size, reference_hash):
    alignment = align_volume_rigid_to_reference(moving, reference, rotations)
    transform = RigidVolumeTransform.from_alignment(
        alignment, volume_shape=moving.shape, voxel_size=voxel_size, gt_sha256=reference_hash
    )
    return transform, dataclasses.asdict(alignment.receipt)


def _class_mean_coordinates(ids, z, labels):
    if (
        ids.shape != (len(labels),)
        or not np.array_equal(np.sort(ids), np.arange(len(labels)))
        or z.shape != (len(labels), 2)
        or not np.isfinite(z).all()
        or not np.array_equal(np.unique(labels), np.arange(3))
    ):
        raise ValueError("Embedding IDs, latent coordinates or evaluation labels are invalid")
    z_by_id = np.empty_like(z)
    z_by_id[ids] = z
    return np.stack([z_by_id[labels == k].mean(axis=0) for k in range(3)]).astype(np.float32)


def _states_from_model(theta, means, n):
    if theta.shape != (n * n * (n // 2 + 1), 3) or theta.dtype != np.complex64 or means.shape != (3, 2):
        raise ValueError("Expected a q2 complex64 half-Fourier model and three mean coordinates")
    states_ft = theta[:, 0][None] + np.einsum("fq,kq->kf", theta[:, 1:], means, optimize=True)
    states = np.asarray(ftu.get_idft3_real(states_ft.reshape(3, n, n, n // 2 + 1), (n, n, n)), np.float32)
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite reconstructed PPCA state")
    return states


def _check_effective_gt_maps(gt_fourier, gt):
    """Fail if saved real-space GT is not the exact inverse DFT of projected GT."""
    n = gt.shape[1]
    expected = np.asarray(ftu.get_idft3(gt_fourier.reshape(3, n, n, n))).real.astype(np.float32)
    if not np.array_equal(gt, expected):
        raise ValueError("Corrected GT maps differ from the simulated Fourier target")


def evaluate(checkpoint, embeddings, training_manifest, evaluation_dir, fixture_verification, vdam_maps, output, *, active_radius=None):
    checkpoint = Path(checkpoint).resolve()
    embeddings = Path(embeddings).resolve()
    training_manifest = Path(training_manifest).resolve()
    evaluation_dir = Path(evaluation_dir).resolve()
    fixture_verification = Path(fixture_verification).resolve()
    output = Path(output).resolve()
    training = json.loads(training_manifest.read_text())
    n = int(training["box"])
    voxel_size = float(training["voxel_size"])
    with np.load(checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        theta = saved["theta"].copy()
    if metadata["iteration"] != metadata["config"]["iterations"]:
        raise ValueError("Evaluation requires a final PPCA checkpoint")
    if metadata["identity"]["manifest_sha256"] != file_hash(training_manifest):
        raise ValueError("PPCA checkpoint training identity mismatch")
    with np.load(embeddings, allow_pickle=False) as values:
        ids = values["particle_ids"].copy()
        z = values["z"].copy()
    verification = json.loads(fixture_verification.read_text())
    if verification["identity"]["manifest_sha256"] != file_hash(training_manifest):
        raise ValueError("Fixture verification training identity mismatch")
    for name in ("truth.npz", "manifest.json", "state0.mrc", "state1.mrc", "state2.mrc"):
        if file_hash(evaluation_dir / name) != verification["files"][f"evaluation/{name}"]:
            raise ValueError(f"Evaluation input identity mismatch: {name}")
    with np.load(evaluation_dir / "truth.npz", allow_pickle=False) as truth:
        labels = truth["labels"].copy()
        gt_fourier = truth["volumes_fourier"].copy()
    evaluation_manifest = json.loads((evaluation_dir / "manifest.json").read_text())
    correction = evaluation_manifest.get("atomic_solvent_correction", {"enabled": False})
    if correction.get("enabled"):
        if correction.get("ground_truth_representation") != "corrected_effective":
            raise ValueError("Corrected fixture must store already transformed ground truth")
        if gt_fourier.dtype != np.complex64 or gt_fourier.shape != (3, n**3):
            raise ValueError("Corrected fixture Fourier target has wrong dtype or shape")
        if hashlib.sha256(np.ascontiguousarray(gt_fourier).tobytes()).hexdigest() != evaluation_manifest.get(
            "ground_truth_fourier_sha256"
        ):
            raise ValueError("Corrected fixture Fourier target hash mismatch")
        for state in range(3):
            name = f"state{state}.mrc"
            if file_hash(evaluation_dir / name) != evaluation_manifest.get("downsampled_maps", {}).get(name):
                raise ValueError(f"Corrected fixture effective map hash mismatch: {name}")
    means = _class_mean_coordinates(ids, z, labels)
    ppca = _states_from_model(theta, means, n)
    gt_paths = [evaluation_dir / f"state{k}.mrc" for k in range(3)]
    gt = np.stack([load_mrc(str(path)) for path in gt_paths]).astype(np.float32)
    if correction.get("enabled"):
        _check_effective_gt_maps(gt_fourier, gt)
    vdam_paths = [Path(path).resolve() for path in vdam_maps]
    if len(vdam_paths) != 3:
        raise ValueError("Exactly three VDAM class maps are required")
    vdam = np.stack([load_relion_volume(str(path)) for path in vdam_paths]).astype(np.float32)
    if gt.shape != ppca.shape or vdam.shape != ppca.shape:
        raise ValueError("All maps must share the training box")
    if active_radius is not None and not 1 <= active_radius < n // 2 - 1:
        raise ValueError("Active radius is outside reported FSC shells")
    coords = np.arange(n, dtype=np.float32) - n // 2
    radius = np.sqrt(sum(axis * axis for axis in np.meshgrid(coords, coords, coords, indexing="ij")))
    edge = np.clip((radius - training["particle_diameter_ang"] / (2 * voxel_size)) / 5, 0, 1)
    mask = ((1 + np.cos(np.pi * edge)) / 2).astype(np.float32)
    rotations = relion_alignment_rotations(2)
    gt_mean = gt.mean(axis=0)
    gt_mean_hash = hashlib.sha256(gt_mean.tobytes()).hexdigest()
    fit_filter = (lambda volume: _bandlimit(volume, active_radius)) if active_radius is not None else (lambda volume: volume)
    ppca_frame, ppca_receipt = _fit(fit_filter(ppca.mean(axis=0)), fit_filter(gt_mean), rotations, voxel_size, gt_mean_hash)
    vdam_frame, vdam_receipt = _fit(fit_filter(vdam.mean(axis=0)), fit_filter(gt_mean), rotations, voxel_size, gt_mean_hash)
    ppca_shared = np.stack([ppca_frame.apply(vol, voxel_size=voxel_size, gt_sha256=gt_mean_hash) for vol in ppca])
    vdam_shared = np.stack([vdam_frame.apply(vol, voxel_size=voxel_size, gt_sha256=gt_mean_hash) for vol in vdam])
    pairing_auc_values = [
        [_fsc_report(vdam_shared[j], gt[k], voxel_size, mask, active_radius)["raw"]["active_band"]["fsc_auc"] if active_radius is not None else _fsc_report(vdam_shared[j], gt[k], voxel_size, mask)["raw"]["fsc_auc"] for k in range(3)]
        for j in range(3)
    ]
    if any(value is None for row in pairing_auc_values for value in row):
        raise ValueError("VDAM/GT matching has no finite non-DC FSC")
    pairing_auc = np.asarray(pairing_auc_values, dtype=np.float64)
    if not np.isfinite(pairing_auc).all():
        raise ValueError("Nonfinite VDAM/GT matching score")
    vdam_rows, state_cols = linear_sum_assignment(-pairing_auc)
    matched = {int(state): int(row) for row, state in zip(vdam_rows, state_cols)}
    output.mkdir(parents=True, exist_ok=False)
    write_mrc(str(output / "common_mask.mrc"), mask, voxel_size=voxel_size)
    states = []
    for k in range(3):
        j = matched[k]
        gt_hash = file_hash(gt_paths[k])
        ppca_local, ppca_local_receipt = _fit(fit_filter(ppca[k]), fit_filter(gt[k]), rotations, voxel_size, gt_hash)
        vdam_local, vdam_local_receipt = _fit(fit_filter(vdam[j]), fit_filter(gt[k]), rotations, voxel_size, gt_hash)
        ppca_individual = ppca_local.apply(ppca[k], voxel_size=voxel_size, gt_sha256=gt_hash)
        vdam_individual = vdam_local.apply(vdam[j], voxel_size=voxel_size, gt_sha256=gt_hash)
        write_mrc(str(output / f"ppca_state{k}_shared.mrc"), ppca_shared[k], voxel_size=voxel_size)
        states.append({
            "true_state": k,
            "particle_count": int(np.count_nonzero(labels == k)),
            "z_bar": means[k].tolist(),
            "matched_vdam_class_zero_based": j,
            "ppca_shared_vs_gt": _fsc_report(ppca_shared[k], gt[k], voxel_size, mask, active_radius),
            "vdam_shared_vs_gt": _fsc_report(vdam_shared[j], gt[k], voxel_size, mask, active_radius),
            "ppca_shared_vs_matched_vdam": _fsc_report(ppca_shared[k], vdam_shared[j], voxel_size, mask, active_radius),
            "ppca_individual_vs_gt_diagnostic": _fsc_report(ppca_individual, gt[k], voxel_size, mask, active_radius),
            "vdam_individual_vs_gt_diagnostic": _fsc_report(vdam_individual, gt[k], voxel_size, mask, active_radius),
            "ppca_individual_transform": ppca_local.to_dict(),
            "vdam_individual_transform": vdam_local.to_dict(),
            "ppca_individual_fit_receipt": ppca_local_receipt,
            "vdam_individual_fit_receipt": vdam_local_receipt,
        })
    report = {
        "schema": "recovar-vdam-ppca-pilot-evaluation-v1",
        "scientific_acceptance": "not threshold-qualified",
        "training_manifest_sha256": file_hash(training_manifest),
        "fixture_verification_sha256": file_hash(fixture_verification),
        "evaluation_manifest_sha256": file_hash(evaluation_dir / "manifest.json"),
        "checkpoint_sha256": file_hash(checkpoint),
        "embeddings_sha256": file_hash(embeddings),
        "vdam_map_sha256": [file_hash(path) for path in vdam_paths],
        "common_mask_sha256": file_hash(output / "common_mask.mrc"),
        "ppca_shared_transform": ppca_frame.to_dict(),
        "vdam_shared_transform": vdam_frame.to_dict(),
        "ppca_shared_fit_receipt": ppca_receipt,
        "vdam_shared_fit_receipt": vdam_receipt,
        "vdam_pairing_auc": pairing_auc.tolist(),
        "active_radius": active_radius,
        "states": states,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("embeddings")
    parser.add_argument("training_manifest")
    parser.add_argument("evaluation_dir")
    parser.add_argument("fixture_verification")
    parser.add_argument("--vdam-maps", nargs=3, required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--active-radius", type=int)
    args = parser.parse_args()
    report = evaluate(
        args.checkpoint,
        args.embeddings,
        args.training_manifest,
        args.evaluation_dir,
        args.fixture_verification,
        args.vdam_maps,
        args.output,
        active_radius=args.active_radius,
    )
    print(json.dumps({"states": len(report["states"]), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
