#!/usr/bin/env python
"""One benchmark row of relax multibody against same-seed RELION runs (the RELION repeats give the band).

    python -m scripts.multibody_parity_row --data <dataset> --relion <MultiBody dir> [<repeat dir> ...] \\
        --relax <relax dir> --json row.json

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


def row(data: Path, relion_dirs: list[Path], relax_dir: Path) -> dict:
    from relax.multibody.bodies import read_bodies

    motion = json.loads((data / "body_motion.json").read_text())
    bodies = read_bodies(data / "bodies.star", pixel_size=float(motion["voxel_size"]))
    masks = [np.transpose(body.mask, (2, 1, 0)) for body in bodies]
    reference = relion_dirs[0]
    out = {"data": str(data), "relion": [str(d) for d in relion_dirs], "relax": str(relax_dir), "bodies": []}
    gt = {str(d): evaluate(data, d, None) for d in [*relion_dirs, relax_dir]}
    final = compare(reference, relax_dir, None)
    for b, mask in enumerate(masks):
        name = f"run_body{b + 1:03d}.mrc"
        out["bodies"].append({
            "body": b + 1,
            "relax_vs_relion_masked_auc": _masked_auc(relax_dir / name, reference / name, mask),
            "relion_repeat_vs_relion_masked_auc": [_masked_auc(d / name, reference / name, mask) for d in relion_dirs[1:]],
            "identical_final_poses": final["bodies"][b]["fraction_same_pose"],
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
    args = parser.parse_args()
    report = row(args.data, args.relion, args.relax)
    for body in report["bodies"]:
        print(f"body {body['body']}: relax-vs-RELION masked AUC {body['relax_vs_relion_masked_auc']:.5f}; "
              f"RELION repeats {[round(v, 5) for v in body['relion_repeat_vs_relion_masked_auc']]}; "
              f"identical poses {body['identical_final_poses']:.4f}; GT masked AUC "
              f"{[round(v, 4) for v in body['gt_masked_auc'].values()]}")
    for k, v in report["schedule"].items():
        print(f"schedule {Path(k).name}: {v}")
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
