#!/usr/bin/env python
"""One benchmark row of relax multibody against same-seed RELION runs (the RELION repeats give the band).

    python -m scripts.multibody_parity_row --data <dataset> --relion <MultiBody dir> [<repeat dir> ...] \\
        --relax <relax dir> --json row.json

Real data has no truth: pass ``--body_star`` (mask paths relative to ``--data``) and ``--pixel_size`` instead of a
synthetic dataset's ``body_motion.json``, and the ground-truth columns are left out.

Per body: the masked map FSC-AUC of relax's final ``run_body<BBB>.mrc`` against the first RELION run's, and of
each RELION repeat against the first (the band); the masked ground-truth FSC-AUC and the median motion error of
every run; the fraction of identical final body poses. Per run: the iterations, and the first iteration whose
schedule (image size, resolution, HEALPix order, offset range, fixed bodies) differs from the first RELION run's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from recovar.utils.helpers import load_relion_volume

from scripts.compare_multibody_relion import compare, trajectory
from scripts.evaluate_multibody_gt import evaluate
from scripts.fsc_metrics import normalized_fsc_auc, shell_fsc

SCHEDULE_KEYS = ("current_size", "resolution", "healpix_order", "offset_range", "keep_fixed")


def _schedule(relion_dir: Path, other: Path) -> dict:
    rows = trajectory(relion_dir, other)
    first = None
    for row in rows:
        if "relion" in row and "relax" in row:
            a, b = row["relion"], row["relax"]
            same = all(
                (abs(a[k] - b[k]) <= 1e-4 * max(1.0, abs(a[k])) if not isinstance(a[k], list) else a[k] == b[k])
                for k in SCHEDULE_KEYS
            )
            if not same and first is None:
                first = row["iteration"]
        elif first is None:
            first = row["iteration"]
    n = {name: sum(name in row for row in rows) for name in ("relion", "relax")}
    return {"iterations_reference": n["relion"], "iterations": n["relax"], "first_schedule_difference": first}


def _masked_auc(a_path: Path, b_path: Path, mask: np.ndarray) -> float:
    a, b = load_relion_volume(str(a_path)), load_relion_volume(str(b_path))
    return float(normalized_fsc_auc(shell_fsc(a * mask, b * mask)))


def row(data: Path, relion_dirs: list[Path], relax_dir: Path, *, body_star: Path = None, pixel_size: float = None) -> dict:
    from relax.multibody.bodies import read_bodies

    if body_star is None:
        pixel_size = float(json.loads((data / "body_motion.json").read_text())["voxel_size"])
        bodies = read_bodies(data / "bodies.star", pixel_size=pixel_size)
    else:
        bodies = read_bodies(body_star, pixel_size=pixel_size, paths_from=data)
    masks = [np.transpose(body.mask, (2, 1, 0)) for body in bodies]
    reference = relion_dirs[0]
    out = {"data": str(data), "relion": [str(d) for d in relion_dirs], "relax": str(relax_dir), "bodies": []}
    gt = {} if body_star is not None else {str(d): evaluate(data, d, None) for d in [*relion_dirs, relax_dir]}
    final = compare(reference, relax_dir, None)
    repeats = [compare(reference, d, None) for d in relion_dirs[1:]]
    for b, mask in enumerate(masks):
        name = f"run_body{b + 1:03d}.mrc"
        out["bodies"].append({
            "body": b + 1,
            "relax_vs_relion_masked_auc": _masked_auc(relax_dir / name, reference / name, mask),
            "relion_repeat_vs_relion_masked_auc": [_masked_auc(d / name, reference / name, mask) for d in relion_dirs[1:]],
            "identical_final_poses": final["bodies"][b]["fraction_same_pose"],
            "relion_repeat_identical_final_poses": [r["bodies"][b]["fraction_same_pose"] for r in repeats],
            "identical_final_poses_01": final["bodies"][b]["fraction_same_pose_01"],
            "relion_repeat_identical_final_poses_01": [r["bodies"][b]["fraction_same_pose_01"] for r in repeats],
            "pose_angle_median_deg": final["bodies"][b]["pose_angle_deg"]["median"],
            "relion_repeat_pose_angle_median_deg": [r["bodies"][b]["pose_angle_deg"]["median"] for r in repeats],
            "gt_masked_auc": {k: v["bodies"][b]["maps"][0]["masked"]["auc"] for k, v in gt.items()},
            "gt_motion_error_median_deg": {k: v["bodies"][b]["motion_error_deg"]["median"] for k, v in gt.items()},
        })
    out["schedule"] = {str(d): _schedule(reference, d) for d in [*relion_dirs[1:], relax_dir]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--relion", required=True, type=Path, nargs="+")
    parser.add_argument("--relax", required=True, type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--body_star", type=Path, default=None, help="real data: the body STAR file")
    parser.add_argument("--pixel_size", type=float, default=None, help="real data: Angstroms per pixel")
    args = parser.parse_args()
    report = row(args.data, args.relion, args.relax, body_star=args.body_star, pixel_size=args.pixel_size)
    for body in report["bodies"]:
        print(f"body {body['body']}: relax-vs-RELION masked AUC {body['relax_vs_relion_masked_auc']:.5f}; "
              f"RELION repeats {[round(v, 5) for v in body['relion_repeat_vs_relion_masked_auc']]}; "
              f"identical poses {body['identical_final_poses']:.4f}, repeats "
              f"{[round(v, 4) for v in body['relion_repeat_identical_final_poses']]}; at .01 "
              f"{body['identical_final_poses_01']:.4f}, repeats "
              f"{[round(v, 4) for v in body['relion_repeat_identical_final_poses_01']]}; median pose angle "
              f"{body['pose_angle_median_deg']:.3f} deg, repeats "
              f"{[round(v, 3) for v in body['relion_repeat_pose_angle_median_deg']]}; GT masked AUC "
              f"{[round(v, 4) for v in body['gt_masked_auc'].values()]}")
    for k, v in report["schedule"].items():
        print(f"schedule {Path(k).name}: {v}")
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
