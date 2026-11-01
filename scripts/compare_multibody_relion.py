#!/usr/bin/env python
"""Compare a relax multi-body iteration with RELION's: body poses, body half maps and model curves.

    python -m scripts.compare_multibody_relion --relion <RELION MultiBody dir> --relax <relax dir> \\
        --iteration 1 [--final] --json out.json

For each body: the angle between the two programs' residual rotations (the composed body matrices share the
consensus pose and body frame, so this is the angle between their body poses) and the offset difference, per
particle, summarised; for each half map the FSC between relax's and RELION's map (shellwise and its normalized
area) and their correlation; the gold-standard FSC and data-vs-prior curves of both. Map files are read in
RELION's convention, as both programs write them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np
import starfile

from relax.sampling.healpix import euler_angles_to_matrix
from scripts.fsc_metrics import centered_corr, normalized_fsc_auc, shell_fsc


def _stem(directory: Path, iteration: int | None) -> Path:
    return directory / ("run" if iteration is None else f"run_it{iteration:03d}")


def _body_tables(data_star: Path) -> list:
    data = starfile.read(data_star, always_dict=True)
    return [data[k] for k in sorted((k for k in data if k.startswith("images_body_")), key=lambda k: int(k.rsplit("_", 1)[1]))]


def _rotation_angles(eulers_a, eulers_b) -> np.ndarray:
    a, b = euler_angles_to_matrix(eulers_a), euler_angles_to_matrix(eulers_b)
    cos = np.clip((np.einsum("nij,nij->n", a, b) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos))


def _summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "max": float(values.max()), "n": int(values.size),
    }


def _model_curves(model_star: Path, body: int) -> dict:
    blocks = starfile.read(model_star, always_dict=True)
    table = blocks[f"model_body_{body + 1}"]
    return {
        "fsc": np.asarray(table["rlnGoldStandardFsc"], dtype=np.float64),
        "data_vs_prior": np.asarray(table["rlnSsnrMap"], dtype=np.float64),
    }


def compare(relion_dir: Path, relax_dir: Path, iteration: int | None) -> dict:
    relion_stem, relax_stem = _stem(relion_dir, iteration), _stem(relax_dir, iteration)
    relion_bodies = _body_tables(Path(f"{relion_stem}_data.star"))
    relax_bodies = _body_tables(Path(f"{relax_stem}_data.star"))
    if len(relion_bodies) != len(relax_bodies):
        raise ValueError(f"{len(relion_bodies)} RELION bodies, {len(relax_bodies)} relax bodies")
    report = {"iteration": iteration, "relion": str(relion_stem), "relax": str(relax_stem), "bodies": []}
    halves = [None] if iteration is None else [1, 2]
    for b, (rel, rlx) in enumerate(zip(relion_bodies, relax_bodies)):
        euler_cols = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
        angles = _rotation_angles(rel[euler_cols].to_numpy(float), rlx[euler_cols].to_numpy(float))
        offsets = np.hypot(
            rel["rlnOriginXAngst"].to_numpy(float) - rlx["rlnOriginXAngst"].to_numpy(float),
            rel["rlnOriginYAngst"].to_numpy(float) - rlx["rlnOriginYAngst"].to_numpy(float),
        )
        identity = _rotation_angles(np.tile([0.0, 90.0, 0.0], (len(rel), 1)), rel[euler_cols].to_numpy(float))
        body = {
            "body": b + 1,
            "pose_angle_deg": _summary(angles),
            "pose_offset_angstrom": _summary(offsets),
            "fraction_same_pose": float(np.mean((angles < 1e-3) & (offsets < 1e-3))),
            "relion_residual_angle_from_identity_deg": _summary(identity),
            "maps": [],
        }
        for h in halves:
            suffix = f"_body{b + 1:03d}.mrc" if h is None else f"_half{h}_body{b + 1:03d}.mrc"
            a = mrcfile.read(f"{relax_stem}{suffix}").astype(np.float64)
            r = mrcfile.read(f"{relion_stem}{suffix}").astype(np.float64)
            fsc = shell_fsc(a, r)
            body["maps"].append({
                "half": h, "fsc_auc": float(normalized_fsc_auc(fsc)), "fsc": fsc.tolist(),
                "corr": float(centered_corr(a, r)),
                "relative_l2": float(np.linalg.norm(a - r) / np.linalg.norm(r)),
            })
        if iteration is not None:
            relion_model = _model_curves(Path(f"{relion_stem}_half1_model.star"), b)
            relax_model = _model_curves(Path(f"{relax_stem}_model.star"), b)
            body["gold_standard_fsc"] = {"relion": relion_model["fsc"].tolist(), "relax": relax_model["fsc"].tolist()}
            body["data_vs_prior"] = {
                "relion": relion_model["data_vs_prior"].tolist(), "relax": relax_model["data_vs_prior"].tolist()
            }
        report["bodies"].append(body)
    return report


def _scalar(star: Path, block: str, label: str):
    table = starfile.read(star, always_dict=True)[block]
    value = table[label]
    return float(value.iloc[0]) if hasattr(value, "iloc") else float(value)


def trajectory(relion_dir: Path, relax_dir: Path) -> list[dict]:
    """Each numbered iteration's scheduling values in both programs: image size, resolution, accuracy, changes,
    stall counters, sampling and fixed bodies."""

    rows = []
    iteration = 0
    while Path(f"{_stem(relion_dir, iteration)}_optimiser.star").exists() or Path(f"{_stem(relax_dir, iteration)}_optimiser.star").exists():
        row = {"iteration": iteration}
        for name, directory in (("relion", relion_dir), ("relax", relax_dir)):
            stem = _stem(directory, iteration)
            if not Path(f"{stem}_optimiser.star").exists():
                continue
            # Either directory may hold RELION's files (a same-seed repeat) or relax's.
            relion_files = Path(f"{stem}_half1_model.star").exists()
            model = Path(f"{stem}_half1_model.star") if relion_files else Path(f"{stem}_model.star")
            sampling_star = Path(f"{stem}_sampling.star") if relion_files else Path(f"{stem}_optimiser.star")
            sampling_block = "sampling_general" if relion_files else "optimiser_general"
            row[name] = {
                "current_size": _scalar(model, "model_general", "rlnCurrentImageSize"),
                "resolution": _scalar(model, "model_general", "rlnCurrentResolution"),
                "ave_pmax": _scalar(model, "model_general", "rlnAveragePmax"),
                "acc_rot": _scalar(Path(f"{stem}_optimiser.star"), "optimiser_general", "rlnOverallAccuracyRotations"),
                "changes_rot": _scalar(Path(f"{stem}_optimiser.star"), "optimiser_general", "rlnChangesOptimalOrientations"),
                "changes_off": _scalar(Path(f"{stem}_optimiser.star"), "optimiser_general", "rlnChangesOptimalOffsets"),
                "wo_resol_gain": _scalar(Path(f"{stem}_optimiser.star"), "optimiser_general", "rlnNumberOfIterWithoutResolutionGain"),
                "wo_changes": _scalar(Path(f"{stem}_optimiser.star"), "optimiser_general", "rlnNumberOfIterWithoutChangingAssignments"),
                "incr_size": _scalar(Path(f"{stem}_optimiser.star"), "optimiser_general", "rlnIncrementImageSize"),
                "healpix_order": _scalar(sampling_star, sampling_block, "rlnHealpixOrder"),
                "offset_range": _scalar(sampling_star, sampling_block, "rlnOffsetRange"),
                "perturbation": _scalar(sampling_star, sampling_block, "rlnSamplingPerturbInstance"),
                "keep_fixed": starfile.read(model, always_dict=True)["model_bodies"]["rlnBodyKeepFixed"].astype(int).tolist(),
            }
        rows.append(row)
        iteration += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--relion", required=True, type=Path)
    parser.add_argument("--relax", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=None, help="numbered iteration; omit for the final files")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--trajectory", action="store_true", help="print every iteration's scheduling values")
    args = parser.parse_args()
    if args.trajectory:
        for row in trajectory(args.relion, args.relax):
            for name in ("relion", "relax"):
                if name in row:
                    values = " ".join(f"{k}={v}" for k, v in row[name].items())
                    print(f"it{row['iteration']:03d} {name:6s} {values}")
        return
    report = compare(args.relion, args.relax, args.iteration)
    for body in report["bodies"]:
        maps = " ".join(f"h{m['half']}:auc={m['fsc_auc']:.4f},corr={m['corr']:.4f}" for m in body["maps"])
        print(
            f"body {body['body']}: pose angle median {body['pose_angle_deg']['median']:.3f} deg "
            f"(p90 {body['pose_angle_deg']['p90']:.3f}), offset median {body['pose_offset_angstrom']['median']:.3f} A, "
            f"same pose {body['fraction_same_pose']:.3f}; RELION residual angle median "
            f"{body['relion_residual_angle_from_identity_deg']['median']:.3f} deg; maps {maps}"
        )
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
