"""Aligned FSC scoring of VDAM InitialModel final maps (RELION and relax/RECOVAR arms).

InitialModel maps have an arbitrary orientation and handedness, so every map is rigidly registered before any FSC with
relax.diagnostics.gt_registration.fit_rigid_both_hands: both hands fitted with proper rotations, three coarse starts on the
HEALPix-2 grid, the low-pass fit, then the fine stage (refine_rigid_fit); contrast sign fixed at +1. The hand is chosen on
the full-spectrum FSC-AUC.

All arrays are RELION file-frame arrays read raw with mrcfile. relax-frame files (simulator GT class maps) are negated into
that frame; the frozen masks share the voxel layout of both frames (docs/benchmarks/masked_fsc_method.md).

Per arm: the population-weighted class mean is registered to the reference mean (GT mean for synthetic data, RELION's
auto-refine map for real data), and for K > 1 every class is also registered on its own to the reference mean (a consensus
of different states registers poorly; 2026-10-01). Each (class, reference class) pair is scored with its best candidate
transform (consensus or class fit, either hand) by unmasked FSC-AUC; classes are then matched to reference classes by the
Hungarian assignment. Arm pairs: b's consensus is registered onto a's reference-frame consensus with the same routine.

FSC: shell FSC as scripts/fsc_metrics.shell_fsc (rounded radius, shells up to n//2 - 2). FSC-AUC: normalized trapezoid over
shells 1..end, as scripts/evaluate_kclass_gt._normalized_fsc_auc. Resolution: first shell below the threshold, box * pixel /
shell.

Usage: python scripts/score_initialmodel_maps.py CONFIG.json OUT_DIR [--cells a,b] [--workers N]
(--workers parallelizes the registrations inside a cell; spawned processes, PYTHONPATH at the checkout.)
The benchmark rows of docs/benchmarks/relion_vs_relax.md were scored with configs listed in each row's evidence (CPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mrcfile
import numpy as np
from scipy.optimize import linear_sum_assignment

from relax.diagnostics.gt_metrics import relion_alignment_rotations
from relax.diagnostics.gt_registration import apply_rigid_volume_transform, fit_rigid_both_hands

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


class _Fitted:
    """A rigid transform as a record (from a worker, JSON-able): rotation, translation, mirror along array axis 0."""

    def __init__(self, rec):
        self.rotation_matrix = np.asarray(rec["rotation_matrix"])
        self.translation_voxels = np.asarray(rec["translation_voxels"])
        self.mirror_x = bool(rec["mirror_x"])

    def __call__(self, vol):
        return apply_rigid_volume_transform(
            vol, self.rotation_matrix, self.translation_voxels, mirror_x=self.mirror_x, order=APPLY_ORDER
        )


def _register_both_hands(mv, tg, required=True):
    """{hand: record} from the shared registration (relax.diagnostics.gt_registration.fit_rigid_both_hands)."""
    fits = fit_rigid_both_hands(mv, tg, relion_alignment_rotations(2))
    if required and set(fits) != {"proper", "mirror"}:
        raise ValueError(f"registration failed in hand(s) {sorted({'proper', 'mirror'} - set(fits))}")
    return {
        h: {
            "rotation_matrix": f.rotation_matrix.tolist(),
            "translation_voxels": [float(t) for t in f.translation_voxels],
            "mirror_x": f.mirror_x,
            "fit_lowpass_corr": f.lowpass_score,
            "start_lowpass_corrs": list(f.start_lowpass_scores),
            "failed_starts": list(f.failed_starts),
            "fine_start_corr": f.fine_start_score,
            "fine_corr": f.fine_score,
            "fine_evaluations": f.fine_evaluations,
        }
        for h, f in fits.items()
    }


def _fit_task(task):
    """One registration (both hands) of the moving map (a population-weighted sum of class maps, or one class map) onto the
    target (the mean of target maps in their frame). Runs in a worker process; reads its own maps. Returns [(key + hand, record)]."""
    key, moving, weights, targets, frame, mean_path = task
    mv = np.tensordot(np.asarray(weights, dtype=np.float64), np.asarray([read(p)[0] for p in moving]), axes=1)
    tg = read(mean_path, frame)[0] if mean_path else np.mean([read(p, frame)[0] for p in targets], axis=0)
    try:
        recs = _register_both_hands(mv, tg, required=key[1] == "consensus")
    except ValueError as exc:
        raise ValueError(f"{key}: {exc}") from exc
    return [((*key, hand), rec) for hand, rec in recs.items()]


def score_cell(cell, fit_workers=1):
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
    ref_ft = [sh.ft(v) for v in ref_vols]
    ref_ft_m = [sh.ft(v * mask) for v in ref_vols] if mask is not None else None

    arms = {}
    aligned = {}
    R = len(ref_vols)
    frame = cell["reference"]["frame"]
    # Registration (2026-10-01 fix): a K>1 run's classes need not share the population-weighted consensus's best fit (a
    # consensus of different assembly states registers poorly), so every class is also fitted on its own to the reference
    # mean. Each (class, reference class) pair is scored with its best candidate transform (consensus or class fit, either
    # hand) by unmasked FSC-AUC before the Hungarian match.
    tasks, pops_by = [], {}
    for arm in cell["arms"]:
        if len(arm["maps"]) != K:
            raise ValueError(f"{cell['id']} {arm['label']}: {len(arm['maps'])} maps for K={K}")
        pops = model_populations(arm["model_star"], K) if K > 1 else np.array([1.0])
        pops_by[arm["label"]] = pops
        w = (pops / pops.sum()).tolist()
        tasks.append(((arm["label"], "consensus", None), arm["maps"], w, ref_paths, frame, ref_mean_path))
        if K > 1:
            for i in range(K):
                tasks.append(
                    ((arm["label"], "class_to_mean", i), [arm["maps"][i]], [1.0], ref_paths, frame, ref_mean_path)
                )
    if fit_workers > 1:
        # spawn, not fork: the parent has initialized JAX (multithreaded), which a forked child may deadlock on
        with ProcessPoolExecutor(max_workers=fit_workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            fits = dict(r for rs in ex.map(_fit_task, tasks) for r in rs)
    else:
        fits = dict(r for rs in map(_fit_task, tasks) for r in rs)
    for arm in cell["arms"]:
        lab = arm["label"]
        maps = [read(p)[0] for p in arm["maps"]]
        pops = pops_by[lab]
        cons = {h: _Fitted(fits[(lab, "consensus", None, h)]) for h in ("proper", "mirror")}
        cons_vols = {h: [t(v) for v in maps] for h, t in cons.items()}
        # the consensus hand used for pair frames and unmatched classes: the hand whose consensus matches the reference mean best
        hand = max(
            cons,
            key=lambda h: auc(
                sh.fsc(sh.ft(np.tensordot(pops / pops.sum(), np.asarray(cons_vols[h]), axes=1)), sh.ft(ref_mean))
            ),
        )
        fit_rec = {
            "hand": hand,
            **fits[(lab, "consensus", None, hand)],
            "other_hand": fits[(lab, "consensus", None, "mirror" if hand == "proper" else "proper")],
        }
        vols = cons_vols[hand]
        cands = {i: [(f"consensus_{h}", t, cons_vols[h][i]) for h, t in cons.items()] for i in range(K)}
        for (flab, kind, i, h), rec in fits.items():
            if flab == lab and kind != "consensus":
                a_i = _Fitted(rec)
                cands[i].append((f"{kind}_{h}", a_i, a_i(maps[i])))
        best = {}
        curves = [[None] * R for _ in range(K)]
        aucs = np.zeros((K, R))
        for i in range(K):
            for j in range(R):
                opts = [(src, v) for src, _, v in cands[i]]
                scored = [(auc(sh.fsc(sh.ft(v), ref_ft[j])), src, v) for src, v in opts]
                a_best, src, v = max(scored, key=lambda t: t[0])
                aucs[i, j], best[(i, j)] = a_best, (src, v)
                curves[i][j] = sh.fsc(sh.ft(v), ref_ft[j])
        rows, cols = linear_sum_assignment(-aucs)
        matched_vol = {i: best[(i, j)][1] for i, j in zip(rows, cols)}
        fts = [sh.ft(matched_vol.get(i, vols[i])) for i in range(K)]
        fts_m = [sh.ft(matched_vol.get(i, vols[i]) * mask) for i in range(K)] if mask is not None else None
        aligned[lab] = {
            "fts": fts,
            "fts_m": fts_m,
            "pops": pops,
            "vols": [matched_vol.get(i, vols[i]) for i in range(K)],
        }
        per_class = []
        for i, j in zip(rows, cols):
            entry = {
                "class": int(i + 1),
                "reference_class": int(j + 1),
                "population": float(pops[i]),
                "fsc_auc": float(aucs[i, j]),
                "transform": best[(i, j)][0],
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
            "class_fits": {
                f"{kind}:{i + 1}:{h}": rec
                for (flab, kind, i, h), rec in fits.items()
                if flab == arm["label"] and kind != "consensus"
            },
            "vs_reference": {"per_class": per_class, **summary},
        }

    pairs = []
    labels = [a["label"] for a in cell["arms"]]
    arm_cfg = {x["label"]: x for x in cell["arms"]}
    # Optional cell "pairs": [[a, b], ...] restricts the cross-arm comparisons (all pairs by default).
    wanted = {frozenset(p) for p in cell.get("pairs", [])}
    for ia in range(len(labels)):
        for ib in range(ia + 1, len(labels)):
            if wanted and frozenset((labels[ia], labels[ib])) not in wanted:
                continue
            a, b = aligned[labels[ia]], aligned[labels[ib]]
            # Pair frame: b's raw consensus is fitted directly onto a's reference-frame consensus, so the
            # pair's agreement does not depend on how well either map registers to the reference; b then
            # sits in the reference (mask) frame too.
            raw_b = [read(p)[0] for p in arm_cfg[labels[ib]]["maps"]]
            wb = b["pops"] / b["pops"].sum()
            wa = a["pops"] / a["pops"].sum()
            cons_a = np.tensordot(wa, np.asarray(a["vols"]), axes=1)
            recs = _register_both_hands(np.tensordot(wb, np.asarray(raw_b), axes=1), cons_a)
            cons_a_ft = sh.ft(cons_a)
            hand = max(
                recs,
                key=lambda h: auc(
                    sh.fsc(sh.ft(_Fitted(recs[h])(np.tensordot(wb, np.asarray(raw_b), axes=1))), cons_a_ft)
                ),
            )
            al, fit_rec = _Fitted(recs[hand]), {"hand": hand, **recs[hand]}
            b_vols = [al(v) for v in raw_b]
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
        "alignment": "relax.diagnostics.gt_registration.fit_rigid_both_hands (both hands with proper rotations, 3 coarse HEALPix-2 starts, low-pass fit, fine stage at shell 16 on 48^3 samples; sign +1); cubic apply. Arm vs reference: candidates = arm consensus and (K>1) each class, fitted to the reference mean; each (class, reference class) pair uses its best candidate by unmasked FSC-AUC, then the Hungarian match (2026-10-01). Pair: b consensus registered onto a's reference-frame consensus; diagnostic_via_reference_frame uses the per-class reference-frame maps.",
        "arms": arms,
        "pairs": pairs,
        "seconds": round(time.time() - t0, 1),
    }


def _run(args):
    cell, out, fit_workers = args
    try:
        result = score_cell(cell, fit_workers)
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
    ok = True
    for c in cells:  # cells in sequence; --workers parallelizes the registrations inside a cell
        cid, good = _run((c, str(out / f"{c['id']}.json"), a.workers))
        print(cid, "ok" if good else "ERROR", flush=True)
        ok &= good
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
