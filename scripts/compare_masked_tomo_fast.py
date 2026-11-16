"""Masked arm B fast paths at n = 64, matched field: the Fourier prefilter's shortlist recall and score error, the
per-rotation cost of the exact and prefilter paths (sparse and dense gather), and searches with the prefilter.

usage: compare_masked_tomo_fast.py DATASET OUT_JSON [--n-particles 32]

Recall: for each condition (clean, crowded), start (pilot: rotation ~5 deg per axis, u = 0, PASS1 grid; hard: the A3
start at 8 deg and 1 px, HARD_FIRST[0] grid) and reference (true map; 40 A low-pass), the exact and prefilter scores
of the full first-stage grid of every particle; recall@k = fraction of particles whose exact best rotation is among
the k best prefilter rotations (each by its best shift). The chosen k is the smallest with recall >= 0.995 in every
case; the searches then compare exact and prefiltered schedules (poses and time).
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_masked_tomo_opt as B  # noqa: E402

from relax.masked_tomo import likelihoods as L  # noqa: E402
from relax.masked_tomo import pilot as PL  # noqa: E402

PASS1, PASS2 = (20.0, 3.0, 4.0, 1.0), (3.0, 1.0, 1.0, 0.5)
REFINE_LATER = [(6.0, 1.5, 2.0, 1.0), (1.5, 0.5, 0.5, 0.5)]
HARD_FIRST = [(36.0, 6.0, 4.0, 1.0), (6.0, 1.5, 1.0, 0.5)]
KS = (1, 2, 4, 8, 16, 32, 64, 128)


def build(root, condition, P):
    B.N_WORK, B.S_WORK, B.NOISE_R = 64, 64, 24.0
    d, acq, _ = B.load(root, condition, P, L)
    arm = L.MaskedArm(d, 27.0, 28.0, crop=64, lam_rel=0.1, window=None, n_mc=2, seed=0)
    return arm, d, acq


def grid_scores(arm, vft, i, Q0, u0, stage, approx):
    """Full first-stage grid ``(R, T)`` of one particle, exact or prefilter."""
    radius, step, srad, sstep = stage
    rv = PL.rotvec_lattice(radius, step)
    Qs = PL.candidates(Q0, rv)
    c = np.round(u0 / sstep) * sstep if sstep >= 1 else u0
    us = PL.shift_grid(c, srad, sstep)
    part = {k: v[None] for k, v in arm.particle(i).items() if k in ("V", "c", "filt")}
    out = []
    for r in range(0, Qs.shape[0], 64):
        Q = jnp.asarray(Qs[None, r : r + 64])
        if approx:
            m = int(round(srad / sstep))
            ax = jnp.asarray((c[:, None] + (np.arange(-m, m + 1) * sstep)[None]).astype(np.float32))[None]
            out.append(np.asarray(PL._prefilter_batched(arm)(vft, part, Q, ax))[0])
        else:
            dl, it, gd = PL.split_shifts(us)
            out.append(np.asarray(PL._scores_batched(arm)(vft, part, Q, jnp.asarray(dl)[None], jnp.asarray(it)[None], jnp.asarray(gd)[None]))[0])
    return np.concatenate(out), Qs, us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--n-particles", type=int, default=32)
    ap.add_argument("--parts", default="recall,cost,search")
    a = ap.parse_args()
    parts = a.parts.split(",")
    P = a.n_particles
    rep = {"args": vars(a), "devices": [str(x) for x in jax.devices()]}

    def save():
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    data = {}
    for cond in ("clean", "crowded"):
        arm, d, acq = build(a.dataset, cond, P)
        truth = PL.load_truth(a.dataset, 64, P, d.rt)
        maps = PL.basis_maps(64)["homo"] * float(acq["image_scale"])
        for i in range(P):
            arm.prepare_particle(i)
        data[cond] = (arm, d, truth, maps)
    truth = data["crowded"][2]
    rng = np.random.default_rng(10_000)
    Q0 = np.einsum("pab,pbc->pac", truth["Q"], Rotation.from_rotvec(np.deg2rad(rng.normal(0, 5.0, (P, 3)))).as_matrix()).astype(np.float32)
    rng = np.random.default_rng(20_000)
    Qh = np.einsum("pab,pbc->pac", truth["Q"], Rotation.from_rotvec(np.deg2rad(rng.normal(0, 8.0, (P, 3)))).as_matrix()).astype(np.float32)
    uh = (truth["u"] + rng.normal(0, 1.0, (P, 3))).astype(np.float32)
    starts = {"pilot": (Q0, np.zeros((P, 3), np.float32), PASS1), "hard8": (Qh, uh, HARD_FIRST[0])}

    k_choice = None
    if "recall" in parts:
        rc = {}
        worst = {k: 1.0 for k in KS}
        for cond, (arm, d, tr, maps) in data.items():
            refs = {"true": maps[:1], "lp40": PL.lowpass(maps[0], d.pixel, 40.0)[None]}
            for rname, X in refs.items():
                vft = arm.volumes_ft(jnp.asarray(X))
                for sname, (Qa, ua, stage) in starts.items():
                    ranks, drop, spread, top1_same = [], [], [], []
                    for i in range(P):
                        se, Qs, us = grid_scores(arm, vft, i, Qa[i], ua[i], stage, False)
                        sa, _, _ = grid_scores(arm, vft, i, Qa[i], ua[i], stage, True)
                        best_r = int(np.unravel_index(np.argmax(se), se.shape)[0])
                        order = np.argsort(-sa.max(-1))
                        ranks.append(int(np.nonzero(order == best_r)[0][0]))
                        ra, ta = np.unravel_index(np.argmax(sa), sa.shape)
                        drop.append(float(se.max() - se[ra, ta]))  # exact log-score lost by the prefilter's own argmax
                        dlt = sa - se
                        spread.append(float(np.percentile(np.abs(dlt - np.median(dlt)), 99)))
                        top1_same.append(bool(ra == best_r and ta == int(np.argmax(se[best_r]))))
                    key = f"{cond}/{rname}/{sname}"
                    rc[key] = {"recall": {k: float(np.mean(np.array(ranks) < k)) for k in KS}, "max_rank": int(max(ranks)),
                               "prefilter_argmax_exact_loss_median": float(np.median(drop)), "prefilter_argmax_exact_loss_max": float(max(drop)),
                               "score_error_p99_spread_median": float(np.median(spread)), "prefilter_argmax_equals_exact": float(np.mean(top1_same)),
                               "R": int(se.shape[0]), "T": int(se.shape[1])}
                    for k in KS:
                        worst[k] = min(worst[k], rc[key]["recall"][k])
                    rep["recall"] = rc
                    save()
                    print("recall", key, rc[key], flush=True)
        ok = [k for k in KS if worst[k] >= 0.995]
        k_choice = ok[0] if ok else KS[-1]
        rep["recall_worst"] = worst
        rep["k_choice"] = k_choice
        save()

    arm, d, tr, maps = data["crowded"]
    if "cost" in parts:
        cost = {}
        vft = arm.volumes_ft(jnp.asarray(maps[:1]))
        part = {k: v[None] for k, v in arm.particle(0).items() if k in ("V", "c", "filt")}
        Qs = jnp.asarray(PL.candidates(tr["Q"][0], PL.rotvec_lattice(20.0, 3.0)[:64])[None])
        dense = L.MaskedArm(d, 27.0, 28.0, crop=64, lam_rel=0.1, window=None, n_mc=2, seed=0)
        dense.support_idx = None
        dense._prep = arm._prep
        for name, us in (("int729", PL.shift_grid(np.zeros(3), 4.0, 1.0)), ("half125", PL.shift_grid(np.zeros(3), 1.0, 0.5))):
            dl, it, gd = (jnp.asarray(x)[None] for x in PL.split_shifts(us))
            for lab, am in (("sparse", arm), ("dense", dense)):
                t, out = B.timed(PL._scores_batched(am), vft, part, Qs, dl, it, gd)
                cost[f"exact_{lab}_{name}_ms_per_rot"] = 1e3 * t / 64
                if lab == "sparse":
                    ref_out = np.asarray(out)
                else:
                    cost[f"sparse_vs_dense_{name}_score_maxabs"] = float(np.abs(np.asarray(out) - ref_out).max())
            m = 4 if name == "int729" else 2
            step = 1.0 if name == "int729" else 0.5
            ax = jnp.asarray(np.tile((np.arange(-m, m + 1) * step).astype(np.float32), (3, 1)))[None]
            t, _ = B.timed(PL._prefilter_batched(arm), vft, part, Qs, ax)
            cost[f"prefilter_{name}_ms_per_rot"] = 1e3 * t / 64
        rep["cost"] = cost
        save()
        print("cost", cost, flush=True)

    if "search" in parts:
        k = k_choice or 16
        srch = {"k": k}
        vft_X = maps[:1]
        for sname, (Qa, ua, _) in starts.items():
            sched = [PASS1, PASS2] if sname == "pilot" else HARD_FIRST
            res = {}
            for lab, pk in (("exact", None), (f"prefilter_k{k}", k)):
                arm.prefilter_k = pk
                ids = np.arange(P)
                PL.search(arm, vft_X, ids[:8], Qa[:8], ua[:8], sched, 32)  # compile
                t0 = time.time()
                res[lab] = PL.search(arm, vft_X, ids, Qa, ua, sched, 32)
                srch[f"{sname}_{lab}_ms_per_particle"] = 1e3 * (time.time() - t0) / P
            arm.prefilter_k = None
            e, f = res["exact"], res[f"prefilter_k{k}"]
            ang = PL.angle_deg(e[0], f[0])
            du = np.abs(e[1] - f[1]).max(1)
            srch[f"{sname}_pose_differences"] = int(np.sum((ang > 0.1) | (du > 1e-4)))
            srch[f"{sname}_pmax_maxabs_diff"] = float(np.abs(e[2] - f[2]).max())
            srch[f"{sname}_exact_median_angle_to_truth"] = float(np.median(PL.angle_deg(e[0], tr["Q"])))
            srch[f"{sname}_prefilter_median_angle_to_truth"] = float(np.median(PL.angle_deg(f[0], tr["Q"])))
            rep["search"] = srch
            save()
            print("search", srch, flush=True)
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
