#!/usr/bin/env python3
"""Score a relax auto-refine K=1 run under the map-agreement gate against same-command RELION runs.

Rule (user decision 2026-09-23; band condition dropped): the run passes when it meets the unchanged
scorecard thresholds (merged >= 0.95, each half >= 0.90 cross-engine band FSC-AUC) against at least
one same-command RELION 5.0.1 run. The RELION-vs-RELION band (two-sided [min, max] over all RELION
pairs, per metric) and the per-run band checks are computed and reported; they do not enter the
verdict. Fixed-state replay parity is a separate required check and is not evaluated here. Masked
metrics against the first RELION run, with the dataset's frozen mask (``scripts/masked_fsc.py``),
are reported, not gated.

The thresholds and ``score_curves`` are this checkout's
(``docs/math/em_k1_realdata_science_equivalence_scorecard_v1.json``,
``scripts/summarize_em_k1_realdata_science_equivalence.py``). Every pair, RELION-vs-RELION
included, is recomputed with that code.

The RELION runs come from a named set in ``docs/benchmarks/map_gate_references.json``, which
points at curated fixtures, or from ``--relion NAME=DIR`` with ``--box`` and ``--voxel``.
This replaces the watcher/bench copies (``eval_relion_band_generic.py``, ``score_k1_10097_10k.sh``)
that lived in agent scratch and scored against references that were later deleted.

Usage::

    python scripts/score_k1_map_gate.py --reference-set empiar10097_10k_k1 \\
        --relax-outputs RUN/outputs --label NAME --out SCORE.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fsc_metrics import shell_fsc  # noqa: E402
from scripts.summarize_em_k1_realdata_science_equivalence import score_curves  # noqa: E402

SCORECARD = ROOT / "docs/math/em_k1_realdata_science_equivalence_scorecard_v1.json"
REFERENCE_SETS = ROOT / "docs/benchmarks/map_gate_references.json"
METRICS = ("merged_cross_engine_band_auc", "half1_cross_engine_band_auc", "half2_cross_engine_band_auc")


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 23):
            digest.update(chunk)
    return digest.hexdigest()


def relion_triplet(root) -> dict:
    root = Path(root)
    return {
        "merged": (root / "run_class001.mrc", "relion"),
        "half1": (root / "run_half1_class001_unfil.mrc", "relion"),
        "half2": (root / "run_half2_class001_unfil.mrc", "relion"),
    }


def relax_triplet(root) -> dict:
    root = Path(root)
    return {
        "merged": (root / "final_merged.mrc", "relax"),
        "half1": (root / "final_half1_unfil.mrc", "relax"),
        "half2": (root / "final_half2_unfil.mrc", "relax"),
    }


def map_convention(path, frame: str) -> str:
    """RELION maps and labeled relax maps are in RELION's convention; unlabeled relax maps are legacy RECOVAR."""

    from relax.helpers.map_io import is_relax_map

    return "relion" if frame == "relion" or is_relax_map(path) else "legacy_recovar_sign"


def load_volume(path, frame: str) -> np.ndarray:
    from recovar.utils import helpers

    loader = helpers.load_relion_volume if map_convention(path, frame) == "relion" else helpers.load_mrc
    volume = np.asarray(loader(str(path)), dtype=np.float64)
    if not np.all(np.isfinite(volume)):
        raise ValueError(f"non-finite map {path}")
    return volume


def reference_set(name: str) -> dict:
    sets = json.loads(REFERENCE_SETS.read_text())["sets"]
    if name not in sets:
        raise SystemExit(f"no reference set {name!r} in {REFERENCE_SETS}; known: {', '.join(sorted(sets))}")
    return sets[name]


def masked_scores(label, mask_dataset, relion_maps, relax_maps, out_dir) -> dict:
    """``scripts/masked_fsc.py score`` of this checkout: the relax arm against one RELION run with the frozen mask."""

    command = [
        sys.executable,
        str(ROOT / "scripts" / "masked_fsc.py"),
        "score",
        "--dataset",
        mask_dataset,
        "--label",
        label,
        "--relion-maps",
        *map(str, relion_maps),
        "--relax-maps",
        *map(str, relax_maps),
        "--out-dir",
        str(out_dir),
    ]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV")}
    subprocess.run(command, check=True, env=env)
    result = json.loads((Path(out_dir) / "masked_fsc.json").read_text())
    row = {
        "dataset": result["dataset"],
        "mask_sha256": result["mask"]["sha256"],
        "band": result["band"],
        "evidence": str(Path(out_dir) / "masked_fsc.json"),
        "null_reasons": result["null_reasons"],
        "frame_correlation_in_mask": result.get("frame_correlation_in_mask"),
    }
    for engine in ("relion", "relax"):
        entry = result[engine]
        row[engine] = (
            None
            if entry is None
            else {
                key: entry.get(key)
                for key in (
                    "masked_resolution_A",
                    "masked_sustained_resolution_A",
                    "masked_corrected_band_auc",
                    "unmasked_band_auc",
                )
            }
        )
    cross = result.get("cross_engine")
    row["cross_engine_masked_band_auc"] = None if cross is None else {k: v["band_auc"] for k, v in cross.items()}
    return {"masked": row, "masked_command": command}


def score_map_gate(
    *,
    relion_runs: dict,
    relax_outputs,
    box: int,
    voxel: float,
    label: str,
    mask_dataset: str | None = None,
    skip_masked: str | None = None,
    masked_out_dir=None,
) -> dict:
    """The gate verdict and every pair's metrics; see the module docstring for the rule."""

    if not relion_runs:
        raise ValueError("at least one RELION run is required")
    if not skip_masked and relax_outputs and not mask_dataset:
        raise SystemExit("masked metrics need a frozen-mask dataset key, or skip_masked with a reason")
    thresholds = json.loads(SCORECARD.read_text())["thresholds"]
    runs = {name: relion_triplet(root) for name, root in relion_runs.items()}
    if relax_outputs:
        runs["relax"] = relax_triplet(relax_outputs)
    cache: dict = {}

    def volume(path, frame):
        if str(path) not in cache:
            cache[str(path)] = load_volume(path, frame)
        return cache[str(path)]

    def score(a_name, b_name):
        a, b = runs[a_name], runs[b_name]
        curves = {
            "recovar_final_half_fsc": shell_fsc(volume(*a["half1"]), volume(*a["half2"])),
            "relion_final_half_fsc": shell_fsc(volume(*b["half1"]), volume(*b["half2"])),
            "final_cross_engine_raw": shell_fsc(volume(*a["merged"]), volume(*b["merged"])),
            "final_cross_engine_half1": shell_fsc(volume(*a["half1"]), volume(*b["half1"])),
            "final_cross_engine_half2": shell_fsc(volume(*a["half2"]), volume(*b["half2"])),
        }
        metrics = score_curves(curves, box_size=box, voxel_size_angstrom=voxel, thresholds=thresholds)
        primary = metrics["primary_metrics"]
        row = {m: float(primary[m]) for m in METRICS}
        row["failed_gates"] = list(metrics["failed_gates"])
        row["jointly_resolved_band"] = metrics["jointly_resolved_band"]
        return row

    relion_names = list(relion_runs)
    relion_pairs = {f"{a}__{b}": score(a, b) for a, b in itertools.combinations(relion_names, 2)}
    relax_pairs = {f"relax__{b}": score("relax", b) for b in relion_names} if relax_outputs else {}
    band = (
        {
            m: [min(row[m] for row in relion_pairs.values()), max(row[m] for row in relion_pairs.values())]
            for m in METRICS
        }
        if relion_pairs
        else {}
    )
    merged_thr = float(thresholds["merged_cross_engine_band_auc_min"])
    half_thr = float(thresholds["each_half_cross_engine_band_auc_min"])

    def meets_thresholds(row):
        return (
            row["merged_cross_engine_band_auc"] >= merged_thr
            and row["half1_cross_engine_band_auc"] >= half_thr
            and row["half2_cross_engine_band_auc"] >= half_thr
        )

    threshold_hits = [name for name, row in relax_pairs.items() if meets_thresholds(row)]
    below_band = sorted(f"{n}:{m}" for n, row in relax_pairs.items() for m in METRICS if band and row[m] < band[m][0])
    above_band = sorted(f"{n}:{m}" for n, row in relax_pairs.items() for m in METRICS if band and row[m] > band[m][1])
    result = {
        "label": label,
        "rule": "thresholds met against >= 1 same-command RELION run (user decision 2026-09-23; band condition dropped, reported only)",
        "scorecard": {"path": str(SCORECARD), "sha256": sha256(SCORECARD)},
        "thresholds": {"merged_min": merged_thr, "each_half_min": half_thr},
        "box": int(box),
        "voxel_size_angstrom": float(voxel),
        "maps": {
            run: {
                kind: {"path": str(path), "sha256": sha256(path), "convention": map_convention(path, frame)}
                for kind, (path, frame) in triplet.items()
            }
            for run, triplet in runs.items()
        },
        "relion_pairs": relion_pairs,
        "relax_pairs": relax_pairs,
        "relion_band": band,
        "threshold_met_against": threshold_hits,
        "below_band": below_band,
        "above_band": above_band,
        "band_condition_gating": False,
        "map_gate_pass": bool(threshold_hits) if relax_outputs else None,
        "supporting_check_required": "fixed-state replay parity (separate evidence)",
    }
    if relax_outputs:
        if skip_masked:
            result.update(masked=None, masked_null_reason=f"skipped: {skip_masked}")
        else:
            first = runs[relion_names[0]]
            result.update(
                masked_scores(
                    label,
                    mask_dataset,
                    [first[k][0] for k in ("merged", "half1", "half2")],
                    [runs["relax"][k][0] for k in ("merged", "half1", "half2")],
                    masked_out_dir,
                )
            )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-set", help=f"named RELION runs, box, voxel and mask in {REFERENCE_SETS.name}")
    parser.add_argument("--relion", nargs="+", metavar="NAME=DIR", help="RELION run directories (instead of a set)")
    parser.add_argument("--box", type=int)
    parser.add_argument("--voxel", type=float)
    parser.add_argument("--relax-outputs", help="relax run's outputs; omit to score only the RELION-vs-RELION band")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mask-dataset", help="frozen-mask key (docs/benchmarks/frozen_masks.json); default: the set's"
    )
    parser.add_argument("--skip-masked", metavar="REASON", help="skip masked metrics; the reason is recorded")
    args = parser.parse_args(argv)
    if bool(args.reference_set) == bool(args.relion):
        parser.error("give exactly one of --reference-set and --relion")
    if args.reference_set:
        spec = reference_set(args.reference_set)
        relion_runs, box, voxel = dict(spec["runs"]), int(spec["box"]), float(spec["voxel_size_angstrom"])
        mask_dataset = args.mask_dataset or spec.get("mask_dataset")
    else:
        if args.box is None or args.voxel is None:
            parser.error("--relion needs --box and --voxel")
        relion_runs = dict(item.split("=", 1) for item in args.relion)
        box, voxel, mask_dataset = args.box, args.voxel, args.mask_dataset
    result = score_map_gate(
        relion_runs=relion_runs,
        relax_outputs=args.relax_outputs,
        box=box,
        voxel=voxel,
        label=args.label,
        mask_dataset=mask_dataset,
        skip_masked=args.skip_masked,
        masked_out_dir=Path(args.out).with_suffix("").as_posix() + "_masked",
    )
    if args.reference_set:
        result["reference_set"] = args.reference_set
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for name, row in {**result["relion_pairs"], **result["relax_pairs"]}.items():
        print(name, {m: round(row[m], 6) for m in METRICS}, row["failed_gates"])
    print("threshold met against:", result["threshold_met_against"], "| below band (info only):", result["below_band"])
    if result["map_gate_pass"] is not None:
        print("MAP GATE", "PASS" if result["map_gate_pass"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
