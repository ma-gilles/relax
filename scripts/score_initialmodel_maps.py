"""Aligned FSC scoring of VDAM InitialModel final maps (RELION and relax/RECOVAR arms).

InitialModel maps have an arbitrary orientation and handedness, so every map is
rigidly registered (HEALPix-2 seed with x-mirror, then the continuous rigid fit
of relax.diagnostics.gt_registration; contrast sign fixed at +1) before any FSC.

All arrays are RELION file-frame arrays read raw with mrcfile. relax-frame files
(simulator GT class maps) are negated into that frame; the frozen masks share the
voxel layout of both frames (docs/benchmarks/masked_fsc_method.md).

Per arm: one rigid transform is fitted from the arm's population-weighted class
mean to the reference mean (GT mean for synthetic data, RELION's auto-refine map
for real data) and applied to every class (classes of one run share a frame).
Classes are matched to reference classes, and arms to each other, by the
Hungarian assignment on unmasked FSC-AUC.

FSC: shell FSC as scripts/fsc_metrics.shell_fsc (rounded radius, shells up to
n//2 - 2). FSC-AUC: normalized trapezoid over shells 1..end, as
scripts/evaluate_kclass_gt._normalized_fsc_auc. Resolution: first shell below
the threshold, box * pixel / shell.

Usage: python scripts/score_initialmodel_maps.py CONFIG.json OUT_DIR [--cells a,b] [--workers N]

The benchmark rows of docs/benchmarks/relion_vs_relax.md were scored with
docs/benchmarks/initialmodel_scores/config.json (CPU; PYTHONPATH at the checkout).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mrcfile
import numpy as np
from scipy.optimize import linear_sum_assignment

from relax.diagnostics.gt_metrics import relion_alignment_rotations
from relax.diagnostics.gt_registration import (
    RigidFitControls,
    align_volume_rigid_to_reference,
    apply_rigid_volume_transform,
)

APPLY_ORDER = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path, frame="relion"):
    with mrcfile.open(path, permissive=True) as m:
        data = np.asarray(m.data, dtype=np.float64).copy()
        voxel = float(m.voxel_size.x)
    if frame == "relax":
        data = -data
    return data, voxel


def model_populations(star, k):
    """rlnClassDistribution from data_model_classes of a RELION-style model.star."""
    lines = Path(star).read_text().splitlines()
    start = lines.index("data_model_classes")
    cols, rows = [], []
    for line in lines[start + 1 :]:
        s = line.strip()
        if s.startswith("_rln"):
            cols.append(s.split()[0])
        elif cols and s and not s.startswith(("loop_", "#")):
            if s.startswith("data_"):
                break
            rows.append(s.split())
        elif cols and rows and not s:
            break
    idx = cols.index("_rlnClassDistribution")
    pops = np.array([float(r[idx]) for r in rows[:k]])
    return pops


class Shells:
    def __init__(self, n):
        f = np.fft.fftfreq(n) * n
        z, y, x = np.meshgrid(f, f, f, indexing="ij")
        self.idx = np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32).ravel()
        self.n = n

    def ft(self, vol):
        return np.fft.fftn(vol).ravel()

    def fsc(self, fa, fb):
        num = np.bincount(self.idx, weights=np.real(fa * np.conj(fb)))
        pa = np.bincount(self.idx, weights=np.abs(fa) ** 2)
        pb = np.bincount(self.idx, weights=np.abs(fb) ** 2)
        den = np.sqrt(pa * pb)
        out = np.full(num.shape, np.nan)
        np.divide(num, den, out=out, where=den > 0)
        return out[: self.n // 2 - 1]


def auc(fsc):
    v = np.asarray(fsc, dtype=np.float64)
    fin = np.isfinite(v)
    fin[0] = False
    x = np.arange(v.size, dtype=np.float64)[fin]
    y = v[fin]
    x = (x - x[0]) / (x[-1] - x[0])
    return float(np.trapezoid(y, x))


def res(fsc, thr, n, px):
    below = np.flatnonzero(np.asarray(fsc)[1:] < thr)
    if not below.size:
        return None
    return float(n * px / (below[0] + 1))


def fit(moving, reference, rotations):
    t0 = time.time()
    al = align_volume_rigid_to_reference(moving, reference, rotations, controls=RigidFitControls(allow_mirror=True))
    rec = {
        "fit_lowpass_corr": float(al.score),
        "full_res_corr_order1": float(al.corr),
        "mirror_x": bool(al.mirror_x),
        "translation_voxels": [float(t) for t in al.translation_voxels],
        "rotation_matrix": np.asarray(al.rotation_matrix).tolist(),
        "optimizer_success": bool(al.receipt.optimizer_success),
        "translation_outside_quarter_box": bool(al.receipt.translation_outside_quarter_box),
        "seconds": round(time.time() - t0, 1),
    }
    return al, rec


def apply(vol, al):
    return apply_rigid_volume_transform(
        vol, al.rotation_matrix, al.translation_voxels, mirror_x=al.mirror_x, order=APPLY_ORDER
    )


def score_cell(cell):
    t0 = time.time()
    K = int(cell["K"])
    ref_paths = cell["reference"]["paths"]
    refs = [read(p, cell["reference"]["frame"]) for p in ref_paths]
    px = float(cell.get("pixel_size_A") or refs[0][1])
    ref_vols = [r[0] for r in refs]
    n = ref_vols[0].shape[0]
    ref_mean = np.mean(ref_vols, axis=0)
    ref_mean_path = cell["reference"].get("mean_path")
    if ref_mean_path:  # explicit consensus reference (e.g. reference_gt.mrc)
        ref_mean = read(ref_mean_path, cell["reference"]["frame"])[0]
    mask = None
    if cell.get("mask"):
        mask, mvox = read(cell["mask"]["path"])
        if mask.shape != ref_mean.shape:
            raise ValueError(f"{cell['id']}: mask shape {mask.shape} != {ref_mean.shape}")
        if sha256(cell["mask"]["path"]) != cell["mask"]["sha256"]:
            raise ValueError(f"{cell['id']}: mask sha mismatch")
    sh = Shells(n)
    rotations = relion_alignment_rotations(2)
    ref_ft = [sh.ft(v) for v in ref_vols]
    ref_ft_m = [sh.ft(v * mask) for v in ref_vols] if mask is not None else None

    arms = {}
    aligned = {}
    for arm in cell["arms"]:
        maps = [read(p)[0] for p in arm["maps"]]
        if len(maps) != K:
            raise ValueError(f"{cell['id']} {arm['label']}: {len(maps)} maps for K={K}")
        pops = model_populations(arm["model_star"], K) if K > 1 else np.array([1.0])
        w = pops / pops.sum()
        consensus = np.tensordot(w, np.asarray(maps), axes=1)
        al, fit_rec = fit(consensus, ref_mean, rotations)
        vols = [apply(v, al) for v in maps]
        fts = [sh.ft(v) for v in vols]
        fts_m = [sh.ft(v * mask) for v in vols] if mask is not None else None
        aligned[arm["label"]] = {"fts": fts, "fts_m": fts_m, "pops": pops, "vols": vols}
        # vs reference, Hungarian over classes (reference classes = GT classes)
        R = len(ref_vols)
        curves = [[sh.fsc(fts[i], ref_ft[j]) for j in range(R)] for i in range(K)]
        aucs = np.array([[auc(c) for c in row] for row in curves])
        rows, cols = linear_sum_assignment(-aucs)
        per_class = []
        for i, j in zip(rows, cols):
            entry = {
                "class": int(i + 1),
                "reference_class": int(j + 1),
                "population": float(pops[i]),
                "fsc_auc": float(aucs[i, j]),
                "res_05_A": res(curves[i][j], 0.5, n, px),
                "res_0143_A": res(curves[i][j], 0.143, n, px),
                "fsc": [round(float(x), 5) for x in curves[i][j]],
            }
            if mask is not None:
                cm = sh.fsc(fts_m[i], ref_ft_m[j])
                entry.update(
                    {
                        "masked_fsc_auc": auc(cm),
                        "masked_res_05_A": res(cm, 0.5, n, px),
                        "masked_res_0143_A": res(cm, 0.143, n, px),
                        "masked_fsc": [round(float(x), 5) for x in cm],
                    }
                )
            per_class.append(entry)
        matched = [e for e in per_class]
        wsum = sum(e["population"] for e in matched)
        summary = {
            "mean_fsc_auc": float(np.mean([e["fsc_auc"] for e in matched])),
            "weighted_fsc_auc": float(sum(e["population"] * e["fsc_auc"] for e in matched) / wsum),
        }
        if mask is not None:
            summary["mean_masked_fsc_auc"] = float(np.mean([e["masked_fsc_auc"] for e in matched]))
            summary["weighted_masked_fsc_auc"] = float(
                sum(e["population"] * e["masked_fsc_auc"] for e in matched) / wsum
            )
        arms[arm["label"]] = {
            "engine": arm["engine"],
            "maps": arm["maps"],
            "map_sha256": [sha256(p) for p in arm["maps"]],
            "populations": pops.tolist(),
            "fit_to_reference": fit_rec,
            "vs_reference": {"per_class": per_class, **summary},
        }

    pairs = []
    labels = [a["label"] for a in cell["arms"]]
    arm_cfg = {x["label"]: x for x in cell["arms"]}
    for ia in range(len(labels)):
        for ib in range(ia + 1, len(labels)):
            a, b = aligned[labels[ia]], aligned[labels[ib]]
            # Pair frame: b's raw consensus is fitted directly onto a's reference-frame consensus, so the
            # pair's agreement does not depend on how well either map registers to the reference; b then
            # sits in the reference (mask) frame too.
            raw_b = [read(p)[0] for p in arm_cfg[labels[ib]]["maps"]]
            wb = b["pops"] / b["pops"].sum()
            wa = a["pops"] / a["pops"].sum()
            al, fit_rec = fit(
                np.tensordot(wb, np.asarray(raw_b), axes=1), np.tensordot(wa, np.asarray(a["vols"]), axes=1), rotations
            )
            b_vols = [apply(v, al) for v in raw_b]
            b_fts = [sh.ft(v) for v in b_vols]
            curves = [[sh.fsc(a["fts"][i], b_fts[j]) for j in range(K)] for i in range(K)]
            aucs = np.array([[auc(c) for c in row] for row in curves])
            rows, cols = linear_sum_assignment(-aucs)
            w = np.array([(a["pops"][i] + b["pops"][j]) / 2 for i, j in zip(rows, cols)])
            matched = [float(aucs[i, j]) for i, j in zip(rows, cols)]
            via_ref = [auc(sh.fsc(a["fts"][i], b["fts"][j])) for i, j in zip(rows, cols)]
            entry = {
                "a": labels[ia],
                "b": labels[ib],
                "pair_fit": fit_rec,
                "matching": [[int(i + 1), int(j + 1)] for i, j in zip(rows, cols)],
                "per_class_fsc_auc": matched,
                "mean_fsc_auc": float(np.mean(matched)),
                "weighted_fsc_auc": float(np.sum(w * matched) / w.sum()),
                "diagnostic_via_reference_frame_mean_fsc_auc": float(np.mean(via_ref)),
            }
            if K == 1:
                entry["fsc"] = [round(float(x), 5) for x in curves[0][0]]
                entry["res_05_A"] = res(curves[0][0], 0.5, n, px)
                entry["res_0143_A"] = res(curves[0][0], 0.143, n, px)
            if mask is not None:
                m = [auc(sh.fsc(a["fts_m"][i], sh.ft(b_vols[j] * mask))) for i, j in zip(rows, cols)]
                entry["per_class_masked_fsc_auc"] = m
                entry["mean_masked_fsc_auc"] = float(np.mean(m))
                entry["weighted_masked_fsc_auc"] = float(np.sum(w * m) / w.sum())
            pairs.append(entry)

    return {
        "id": cell["id"],
        "K": K,
        "box": n,
        "pixel_size_A": px,
        "reference": {**cell["reference"], "sha256": [sha256(p) for p in ref_paths]},
        "mask": cell.get("mask"),
        "fsc_auc_definition": "normalized trapezoid of the shell FSC over shells 1..n//2-2 (full spectrum)",
        "alignment": "rigid (HEALPix-2 seed + x-mirror, continuous rotation+translation fit, sign +1); cubic apply. Arm vs reference: arm consensus fitted to the reference mean. Pair: b consensus fitted directly onto a's reference-frame consensus.",
        "arms": arms,
        "pairs": pairs,
        "seconds": round(time.time() - t0, 1),
    }


def _run(args):
    cell, out = args
    try:
        result = score_cell(cell)
    except Exception as exc:  # recorded, never hidden
        import traceback

        result = {"id": cell["id"], "error": repr(exc), "traceback": traceback.format_exc()}
    Path(out).write_text(json.dumps(result, indent=1))
    return cell["id"], "error" not in result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("out_dir")
    p.add_argument("--cells", default="")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    cells = json.loads(Path(a.config).read_text())["cells"]
    if a.cells:
        keep = set(a.cells.split(","))
        cells = [c for c in cells if c["id"] in keep]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, str(out / f"{c['id']}.json")) for c in cells]
    ok = True
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for cid, good in ex.map(_run, jobs):
            print(cid, "ok" if good else "ERROR", flush=True)
            ok &= good
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
