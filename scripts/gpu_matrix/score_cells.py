#!/usr/bin/env python3
"""Score the cross-GPU matrix cells of several GPU models against ground truth and a reference model.

    score_cells.py --arm h100=CELLS_DIR --arm p100=CELLS_DIR ... --reference h100 --out scores.json --work DIR

``CELLS_DIR`` holds one ``<cell>/`` per cell (run_cells.py's ``--out-root``). Every metric is an FSC-AUC
(scripts/fsc_metrics.normalized_fsc_auc of shell_fsc) and is computed for every arm with the same code:

- Refine3D (SPA and tomo): the merged map and the average of the unfiltered half maps against the GT map,
  and the merged map against the reference arm's merged map; the iteration count.
- Class3D SPA: the benchmark scorer (``--class3d-scorer``, Hungarian-matched per-class masked GT FSC-AUC and
  class accuracy).
- VDAM InitialModel (SPA and tomo): scripts/score_initialmodel_maps.py (rigid registration, both hands; masked
  GT FSC-AUC), with every arm paired with the reference arm.
- Tomo Class3D: per-class GT FSC-AUC after a Hungarian match, and the matched per-class FSC-AUC against the
  reference arm.
- PPCA InitialModel: the final state's GEMM precision and log-likelihood, and its mean and loading subspace
  against the reference arm's (same seed and particle order).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "gpu_matrix")]
from cells import FIXTURES  # noqa: E402

from scripts.fsc_metrics import normalized_fsc_auc, shell_fsc  # noqa: E402

CLASS3D_SCORER = "/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_bench_k1plus_20260925/tools/score_class3d.py"
FX = "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures"
MASKS = {
    "k2_5k128": (f"{FX}/data_pdb_k2_5k_128/masks/pdb_k2_5k128_c1/pdb_k2_5k128_c1_mask.mrc", "pdb_k2_5k128_c1"),
    "k4_5k128": (f"{FX}/data_pdb_k4_5k_128/masks/pdb_k4_5k128_c1/pdb_k4_5k128_c1_mask.mrc", "pdb_k4_5k128_c1"),
    "k4_50k256": (f"{FX}/data_pdb_k4_50k_256/masks/pdb_k4_50k256_c1/pdb_k4_50k256_c1_mask.mrc", "pdb_k4_50k256_c1"),
    "k1_5k128": (
        f"{FX}/data_noise1_5k_normalized/masks/noise1_k1_5k128_c1/noise1_k1_5k128_c1_mask.mrc",
        "noise1_k1_5k128_c1",
    ),
    "et09_box64": (
        f"{FX}/cryoet_bench_20260926/cases/et09_box64/masks/etbench_et09_box64_c1/etbench_et09_box64_c1_mask.mrc",
        "etbench_et09_box64_c1",
    ),
    "et15_k2_box64": (
        f"{FX}/cryoet_bench_20260926/cases/et15_k2conf_box64/masks/etbench_et15_k2conf_box64_c1/"
        "etbench_et15_k2conf_box64_c1_mask.mrc",
        "etbench_et15_k2conf_box64_c1",
    ),
}
ET15_GT = f"{FX}/cryoet_bench_20260926/cases/et15_k2conf_box64/project"


def _sha256(path: str) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(a: np.ndarray, b: np.ndarray) -> float:
    return float(normalized_fsc_auc(np.asarray(shell_fsc(a, b), dtype=np.float64)))


def _relax_map(path: Path) -> np.ndarray:
    from relax.helpers.map_io import load_relax_map

    return np.asarray(load_relax_map(path), dtype=np.float64)


def _relion_map(path: str | Path) -> np.ndarray:
    from recovar.utils import helpers

    return np.asarray(helpers.load_relion_volume(str(path)), dtype=np.float64)


def score_refine(arms: dict[str, Path], ref: str, gt: np.ndarray) -> dict:
    out, merged = {}, {}
    for label, d in arms.items():
        if not (d / "final_merged.mrc").exists():
            continue
        merged[label] = _relax_map(d / "final_merged.mrc")
        halves = sum(_relax_map(d / f"final_half{h}_unfil.mrc") for h in (1, 2)) / 2.0
        npz = np.load(d / "refinement_results.npz")
        out[label] = {
            "merged_vs_gt": auc(merged[label], gt),
            "unfil_half_average_vs_gt": auc(halves, gt),
            "iterations": int(np.asarray(npz["ave_Pmax_trajectory"]).size),
            "converged": bool(npz["convergence_has_converged"]),
        }
    for label in out:
        if ref in merged:
            out[label]["merged_vs_reference_arm"] = auc(merged[label], merged[ref])
    return out


def score_class3d(arms: dict[str, Path], fixture: str, k: int, work: Path, n_iter: int = 25) -> dict:
    mask, key = MASKS[fixture]
    runs = [
        {"label": label, "engine": "relax", "seed": 29, "path": str(d)}
        for label, d in arms.items()
        if (d / "refinement_results.npz").exists()
    ]
    config = {
        "src": str(ROOT),
        "data_dir": FIXTURES[fixture],
        "mask": mask,
        "mask_key": key,
        "mask_sha256": _sha256(mask),
        "n_classes": k,
        "n_iter": n_iter,
        "runs": runs,
    }
    cfg, out = work / f"class3d_{fixture}_it{n_iter}_cfg.json", work / f"class3d_{fixture}_it{n_iter}_score.json"
    cfg.write_text(json.dumps(config, indent=1))
    subprocess.run([sys.executable, CLASS3D_SCORER, str(cfg), str(out)], check=True)
    return json.loads(out.read_text())


def _vdam_arm(label: str, d: Path, k: int, it: int) -> dict | None:
    maps = [d / f"run_it{it:03d}_class{c + 1:03d}.mrc" for c in range(k)]
    if not all(p.exists() for p in maps):
        return None
    return {
        "label": label,
        "engine": "relax",
        "maps": [str(p) for p in maps],
        "model_star": str(d / f"run_it{it:03d}_model.star"),
    }


def score_vdam(arms: dict[str, Path], ref: str, cell: str, fixture: str, k: int, it: int, gt, work: Path):
    arm_cfgs = [a for label, d in arms.items() if (a := _vdam_arm(label, d, k, it)) is not None]
    mask, key = MASKS[fixture]
    config = {
        "cells": [
            {
                "id": cell,
                "K": k,
                "reference": {"kind": "gt", "frame": gt[0], "paths": gt[1]},
                "mask": {"key": key, "path": mask, "sha256": _sha256(mask)},
                "arms": arm_cfgs,
                "pairs": [[ref, a["label"]] for a in arm_cfgs if a["label"] != ref],
            }
        ]
    }
    cfg, out = work / f"{cell}_cfg.json", work / f"{cell}_score"
    cfg.write_text(json.dumps(config, indent=1))
    rc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "score_initialmodel_maps.py"),
            str(cfg),
            str(out),
            "--workers",
            os.environ.get("SCORE_WORKERS", "8"),
        ],
    ).returncode
    result = json.loads((out / f"{cell}.json").read_text())
    if "error" in result:
        return {"error": result["error"], "output": str(out)}
    summary = {"config": str(cfg), "output": str(out), "scorer_rc": rc}
    for label, arm in result["arms"].items():
        summary[label] = {k: v for k, v in arm["vs_reference"].items() if k != "per_class"}
    for pair in result.get("pairs", []):
        other = pair["b"] if pair["a"] == ref else pair["a"]
        summary.setdefault(other, {})["vs_reference_arm"] = {
            k: v for k, v in pair.items() if k not in ("a", "b", "pair_fit", "matching", "fsc")
        }
    return summary


def score_tomo_class3d(arms: dict[str, Path], ref: str, gt_paths: list[str]) -> dict:
    from scipy.optimize import linear_sum_assignment

    gts = [_relion_map(p) for p in gt_paths]
    maps = {}
    for label, d in arms.items():
        paths = sorted(d.glob("final_class[0-9][0-9][0-9].mrc"))
        if paths:
            maps[label] = [_relax_map(p) for p in paths]
    out = {}
    for label, vols in maps.items():
        table = np.array([[auc(v, g) for g in gts] for v in vols])
        rows, cols = linear_sum_assignment(-table)
        entry = {"per_class_gt": {f"class{r + 1}->gt{c + 1}": float(table[r, c]) for r, c in zip(rows, cols)}}
        entry["mean_gt"] = float(np.mean(table[rows, cols]))
        if ref in maps:
            cross = np.array([[auc(v, w) for w in maps[ref]] for v in vols])
            r2, c2 = linear_sum_assignment(-cross)
            entry["matched_vs_reference_arm"] = [float(cross[r, c]) for r, c in zip(r2, c2)]
        out[label] = entry
    return out


def score_ppca(arms: dict[str, Path], ref: str) -> dict:
    """Final PPCA state: GEMM precision, log-likelihood, and agreement with the reference arm's state.

    ``theta`` ``[coefficients, 1 + q]`` holds the mean and the q loadings. Agreement: the normalized
    correlation of the means, and the cosines of the principal angles between the loading subspaces.
    """

    states, out = {}, {}
    for label, d in arms.items():
        checkpoints = sorted(d.glob("checkpoint_[0-9][0-9][0-9][0-9].npz"))
        if not checkpoints or not (d / "iterations.jsonl").exists():
            continue
        last = json.loads((d / "iterations.jsonl").read_text().strip().splitlines()[-1])
        states[label] = np.asarray(np.load(checkpoints[-1])["theta"], dtype=np.complex128)
        out[label] = {
            "iteration": int(last["iteration"]),
            "gemm_precision": last.get("gemm_precision"),
            "log_likelihood": float(last["log_likelihood"]),
            "loading_singular_values": last.get("loading_singular_values"),
        }
    for label, theta in states.items():
        if ref not in states:
            break
        base = states[ref]
        mean = np.vdot(theta[:, 0], base[:, 0]).real / (np.linalg.norm(theta[:, 0]) * np.linalg.norm(base[:, 0]))
        qa, _ = np.linalg.qr(theta[:, 1:])
        qb, _ = np.linalg.qr(base[:, 1:])
        cosines = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
        out[label]["mean_corr_vs_reference_arm"] = float(mean)
        out[label]["loading_subspace_cosines_vs_reference_arm"] = [float(c) for c in cosines]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="LABEL=CELLS_DIR")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="cells to score (default: all present)")
    args = parser.parse_args()
    roots = dict(item.split("=", 1) for item in args.arm)
    args.work.mkdir(parents=True, exist_ok=True)

    def arms_of(cell: str) -> dict[str, Path]:
        return {label: Path(root) / cell for label, root in roots.items() if (Path(root) / cell).is_dir()}

    def wanted(cell: str) -> bool:
        return (args.only is None or cell in args.only) and bool(arms_of(cell))

    scores: dict[str, object] = {"arms": roots, "reference": args.reference}
    if wanted("refine_k1_5k128"):
        from recovar.utils import helpers

        gt = np.asarray(helpers.load_mrc(str(Path(FIXTURES["k1_5k128"]) / "reference_gt.mrc")), dtype=np.float64)
        scores["refine_k1_5k128"] = score_refine(arms_of("refine_k1_5k128"), args.reference, gt)
    if wanted("tomo_refine_s1"):
        gt = _relion_map(Path(FIXTURES["et_s1"]) / "reference_gt_relion.mrc")
        scores["tomo_refine_s1"] = score_refine(arms_of("tomo_refine_s1"), args.reference, gt)
    # Robustness v1 cells at the benchmark page's full schedules: unmasked GT FSC-AUC as the other refine cells.
    for cell, fixture, gt_name in (
        ("refine_k1_50k256_full", "k1_50k256", "reference_gt_relion.mrc"),
        ("refine_ms2_448_s1", "ms2_448", "reference_gt_relion.mrc"),
        ("tomo_refine_w2_09_box192", "w2_09_box192", "reference_gt_relion.mrc"),
    ):
        if wanted(cell):
            scores[cell] = score_refine(arms_of(cell), args.reference, _relion_map(Path(FIXTURES[fixture]) / gt_name))
    for cell, fixture, k, n_iter in (
        ("class3d_k2_5k128", "k2_5k128", 2, 25),
        ("class3d_k4_5k128", "k4_5k128", 4, 25),
        ("class3d_k4_50k256_it15", "k4_50k256", 4, 15),
    ):
        if wanted(cell):
            scores[cell] = score_class3d(arms_of(cell), fixture, k, args.work, n_iter=n_iter)
    vdam = (
        # (frame, GT paths) as the benchmark configs (docs/benchmarks/initialmodel_scores) name them.
        ("vdam_k1_5k128", "k1_5k128", 1, 200, ("relion", [f"{FIXTURES['k1_5k128']}/reference_gt_relion.mrc"])),
        (
            "vdam_k4_5k128",
            "k4_5k128",
            4,
            200,
            ("relax", [f"{FIXTURES['k4_5k128']}/reference_gt_class{c:03d}.mrc" for c in range(1, 5)]),
        ),
        (
            "tomo_vdam_k1_et09_it10",
            "et09_box64",
            1,
            10,
            ("relion", [f"{FIXTURES['et09_box64']}/reference_gt_relion.mrc"]),
        ),
        (
            "tomo_vdam_k2_et15_it10",
            "et15_k2_box64",
            2,
            10,
            ("relion", [f"{ET15_GT}/reference_gt_class{c:03d}_relion.mrc" for c in (1, 2)]),
        ),
    )
    for cell, fixture, k, it, gt in vdam:
        if wanted(cell):
            scores[cell] = score_vdam(arms_of(cell), args.reference, cell, fixture, k, it, gt, args.work)
    for cell in ("tomo_class3d_et13_it3", "tomo_class3d_et13_it25"):
        if wanted(cell):
            gt = [f"{FIXTURES['et13_k2']}/reference_gt_class{c:03d}_relion.mrc" for c in (1, 2)]
            scores[cell] = score_tomo_class3d(arms_of(cell), args.reference, gt)
    for cell in ("ppca_tomo_vdam_it24", "ppca_tomo_sgd_it24"):
        if wanted(cell):
            scores[cell] = score_ppca(arms_of(cell), args.reference)
    args.out.write_text(json.dumps(scores, indent=1) + "\n")
    print(json.dumps(scores, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
