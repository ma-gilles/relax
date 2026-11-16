"""Kernel-box arm B (``KernelBoxMaskedArm``, masked-tomo/notes/TWO_RES.md Option A): reduction, kernel tails and
neighbour leakage against full-field and matched-field ``MaskedArm``.

usage: compare_masked_tomo_kb.py DATASET OUT_JSON [--n-particles 16]

1. reduction: KernelBoxMaskedArm with N = S equals MaskedArm (stats on one chunk, two particles).
2. exactness and wrap: KernelBox N = 128 and N = 96 against full-field MaskedArm at S = 208 (8.5 A): stats and the
   masked template on a chunk; kernel-tail fractions per particle.
3. leakage, split as masked-tomo/scripts/leak_split.py on the noiseless components: measured ||M L y_nbr||^2,
   (a) the true neighbour density inside the mask through B's filter, (b) the rest; all relative to ||M L y_tgt||^2.
   The kernel-box arm's data side is MaskedArm's at the same S, so its leakage is full_n64_S208's.
4. per-rotation cost of stats for every configuration (R = 32 chunk, the n = 64 pilot shift grids).
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_masked_tomo_opt as B  # noqa: E402

from relax.masked_tomo import likelihoods as L  # noqa: E402
from relax.masked_tomo import pilot as PL  # noqa: E402

# The kernel-box and full-field configurations use an 80-px crop: integer shifts stay inside the crop's valid range
# (+-(Cc/2 - mask) = +-12) through the n = 64 schedule's drift; a 64-px crop of a larger box allows only +-4.
CONFIGS = {
    # name: n, S, crop, mask_r, support_r, noise_r, template box (None = MaskedArm)
    "full_n32_S104": (32, 104, 40, 15.0, 13.5, 12.0, None),
    "matched_n64_S64": (64, 64, 64, 28.0, 27.0, 24.0, None),
    "full_n64_S208": (64, 208, 80, 28.0, 27.0, 24.0, None),
    "kb_n64_S208_N128": (64, 208, 80, 28.0, 27.0, 24.0, 128),
    "kb_n64_S208_N96": (64, 208, 80, 28.0, 27.0, 24.0, 96),
    "kb_n64_S128_N128": (64, 128, 80, 28.0, 27.0, 24.0, 128),
    "full_n64_S128": (64, 128, 80, 28.0, 27.0, 24.0, None),
}


def build(cfg, root, condition, P):
    n, S, crop, mr, sr, nr, N = CONFIGS[cfg]
    B.N_WORK, B.S_WORK, B.NOISE_R = n, S, nr
    d, acq, _ = B.load(root, condition, P, L)
    kw = dict(crop=crop, lam_rel=B.LAM_REL, window=None, n_mc=2, seed=0)
    arm = L.MaskedArm(d, sr, mr, **kw) if N is None else L.KernelBoxMaskedArm(d, sr, mr, template_box=N, **kw)
    return arm, d, acq


def placed_density(n, Cc, scene, kind_sel, obj, acq):
    """leak_split.py: the selected objects' working-size maps at their true poses on the centred Cc^3 crop grid."""
    from recovar import utils
    from recovar.simulation import tomo_scene
    from scipy import ndimage

    k0 = int(np.argmin(np.abs(acq["tilt_angles"])))
    m = PL.basis_maps(n)["homo"][0]
    px = float(acq["voxel_size"]) * int(acq["target_box"]) / n
    G = tomo_scene.scene_frame(acq["rots"][scene, k0], acq["aproj"][scene, k0])
    sel = np.nonzero((obj["scene"] == scene) & kind_sel & (np.abs(obj["coefficients"]).sum(1) > 0))[0]
    grid = np.indices((Cc,) * 3).reshape(3, -1).T - Cc // 2
    out = np.zeros(Cc**3)
    for o in sel:
        Q = utils.R_from_relion(obj["eulers"][o][None])[0]
        c = G @ obj["offset"][o] / px
        src = (grid - c) @ Q + n // 2
        out += obj["coefficients"][o][0] * ndimage.map_coordinates(m, src.T, order=3, mode="constant")
    return out.reshape((Cc,) * 3)


def leak_row(arm, d, i, cc, obj, acq, n, S, Cc):
    """One scene of leak_split.py's split for ``arm`` (data side only: the kernel-box arm shares MaskedArm's)."""
    M = np.asarray(arm.mask)
    rt, ctf = jnp.asarray(d.rt[i]), jnp.asarray(d.ctf_patch[i])
    D, lam = arm.weights(ctf, rt)
    filt = np.asarray(arm.particle_from(rt, ctf, D, lam)["filt"]).reshape((S,) * 3)
    t = L.fourier_crop2(cc["target"][i].astype(np.float64), S).reshape(len(rt), -1)
    nb = L.fourier_crop2(cc["neighbor"][i].astype(np.float64), S).reshape(len(rt), -1)
    Vt = np.asarray(arm.measure(jnp.asarray(t), ctf, rt, D, lam)) * M
    Vn = np.asarray(arm.measure(jnp.asarray(nb), ctf, rt, D, lam)) * M
    Nin = placed_density(n, Cc, i, obj["kind"] == 1, obj, acq) * M
    Tin = placed_density(n, Cc, i, obj["kind"] == 0, obj, acq) * M
    lo = (S - Cc) // 2

    def through_filter(x):
        big = np.zeros((S,) * 3)
        big[lo : lo + Cc, lo : lo + Cc, lo : lo + Cc] = x
        return np.real(np.asarray(L.ftu.get_idft3(L.ftu.get_dft3(jnp.asarray(big)) * filt)))[lo : lo + Cc, lo : lo + Cc, lo : lo + Cc]

    fa = through_filter(Nin)
    ft = through_filter(Tin) * M
    scale = float(np.sum(ft * Vt) / np.sum(ft * ft))
    av = fa * M * scale
    pt = np.sum(Vt**2)
    return {"measured": float(np.sum(Vn**2) / pt), "a_inside": float(np.sum(av**2) / pt),
            "b_rest": float(np.sum((Vn - av) ** 2) / pt), "raw_density_inside": float(np.sum(Nin**2) / np.sum(Tin**2)),
            "target_model_fit": float(np.corrcoef(ft.ravel(), Vt.ravel())[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--n-particles", type=int, default=16)
    ap.add_argument("--parts", default="reduction,exact,leakage,cost")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    parts = a.parts.split(",")
    P = a.n_particles
    rep = {"args": vars(a), "devices": [str(x) for x in jax.devices()]}

    def save():
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    def truth_and_maps(d, acq, n):
        truth = PL.load_truth(a.dataset, n, d.rt.shape[0], d.rt)
        maps = PL.basis_maps(n)["homo"] * float(acq["image_scale"])
        return truth, maps

    def chunk_stats(arm, maps, truth, i, us):
        Qs = jnp.asarray(PL.candidates(truth["Q"][i], PL.rotvec_lattice(20.0, 3.0)[:32]))
        dl, it, gd = PL.split_shifts(us)
        vft = arm.volumes_ft(jnp.asarray(maps[:1]))
        fn = jax.jit(arm.stats_multi)
        args = (vft, arm.particle(int(i)), Qs, jnp.asarray(dl), jnp.asarray(it), jnp.asarray(gd))
        out = [np.asarray(x) for x in fn(*args)]
        t, _ = B.timed(fn, *args)
        return out, 1e3 * t / 32

    us_int = PL.shift_grid(np.zeros(3), 4.0, 1.0)  # pilot n = 64 pass 1: 729 integer shifts
    us_half = PL.shift_grid(np.zeros(3), 1.0, 0.5)  # pilot n = 64 pass 2: 125 shifts, 8 sub-pixel parts

    # 1. reduction N = S
    if "reduction" in parts:
        arm_kb, d, acq = build("kb_n64_S128_N128", a.dataset, "crowded", 2)
        arm_ff, _, _ = build("full_n64_S128", a.dataset, "crowded", 2)
        truth, maps = truth_and_maps(d, acq, 64)
        red = []
        for i in range(2):
            for us in (us_int, us_half):
                (ik, gk), _ = chunk_stats(arm_kb, maps, truth, i, us)
                (iff, gff), _ = chunk_stats(arm_ff, maps, truth, i, us)
                red.append({"particle": i, "T": int(us.shape[0]), "inner_rel": B.rel(iff, ik), "gram_rel": B.rel(gff, gk)})
        rep["reduction_N_eq_S"] = red
        save()
        print("reduction", red, flush=True)

    # 2. exactness against full field at S = 208, and the N = 96 wrap
    if "exact" in parts:
        ex = {}
        arms = {c: build(c, a.dataset, "crowded", P)[0] for c in ("full_n64_S208", "kb_n64_S208_N128", "kb_n64_S208_N96")}
        d, acq = arms["full_n64_S208"].d, None
        acq = np.load(os.path.join(a.dataset, "acquisition.npz"))
        truth, maps = truth_and_maps(d, acq, 64)
        for c in ("kb_n64_S208_N128", "kb_n64_S208_N96"):
            tails = [float(arms[c].prepare_particle(i)["kernel_tail"]) for i in range(P)]
            ex[f"{c}_kernel_tail_median"] = float(np.median(tails))
            ex[f"{c}_kernel_tail_max"] = float(np.max(tails))
            ex[f"{c}_exact_shift"] = arms[c].exact_shift
        for i in range(2):
            ref, _ = chunk_stats(arms["full_n64_S208"], maps, truth, i, us_int)
            for c in ("kb_n64_S208_N128", "kb_n64_S208_N96"):
                got, _ = chunk_stats(arms[c], maps, truth, i, us_int)
                ex[f"{c}_p{i}_inner_rel"] = B.rel(ref[0], got[0])
                ex[f"{c}_p{i}_gram_rel"] = B.rel(ref[1], got[1])
                s_ref = np.asarray(L.scores_from_stats(jnp.asarray(ref[0]), jnp.asarray(ref[1]), True)[0])
                s_got = np.asarray(L.scores_from_stats(jnp.asarray(got[0]), jnp.asarray(got[1]), True)[0])
                ex[f"{c}_p{i}_score_maxabs_diff"] = float(np.abs(s_ref - s_got).max())
                ex[f"{c}_p{i}_score_range"] = float(np.ptp(s_ref))
                ex[f"{c}_p{i}_argmax_same"] = bool(np.argmax(s_ref) == np.argmax(s_got))
            # masked template at the true pose
            Qt, ut = jnp.asarray(truth["Q"][i], jnp.float32), jnp.asarray(truth["u"][i], jnp.float32)
            fs = {}
            for c, arm in arms.items():
                v = arm.volumes_ft(jnp.asarray(maps[:1]))
                fs[c] = np.asarray(arm.filtered(v, Qt, ut, arm.particle(i)["filt"])[0] * arm.mask)
            for c in ("kb_n64_S208_N128", "kb_n64_S208_N96"):
                ex[f"{c}_p{i}_template_rel_l2"] = float(np.linalg.norm(fs[c] - fs["full_n64_S208"]) / np.linalg.norm(fs["full_n64_S208"]))
        rep["exact"] = ex
        save()
        print("exact", ex, flush=True)

    # 3. neighbour leakage, split as masked-tomo/scripts/leak_split.py (noiseless components; (a) true neighbour
    #    density inside the mask through B's filter, (b) the rest), relative to ||Vt||^2
    if "leakage" in parts:
        lk = {}
        obj = np.load(os.path.join(a.dataset, "truth", "objects.npz"))
        acq = np.load(os.path.join(a.dataset, "acquisition.npz"))
        comps = {"crop": np.load(os.path.join(a.dataset, "truth", "components_crop.npz")),
                 "patch": np.load(os.path.join(a.dataset, "truth", "components_patch.npz"))}
        for c, src in (("matched_n64_S64", "crop"), ("full_n32_S104", "patch"), ("full_n64_S208", "patch")):
            n, S, Cc = CONFIGS[c][:3]
            cc = comps[src]
            m = min(P, cc["target"].shape[0])
            arm, d, _ = build(c, a.dataset, "crowded", m)
            rows = [leak_row(arm, d, i, cc, obj, acq, n, S, Cc) for i in range(m)]
            lk[c] = {"n_scenes": m, **{f"{k}_median": float(np.median([r[k] for r in rows])) for k in rows[0]},
                     **{f"{k}_p90": float(np.percentile([r[k] for r in rows], 90)) for k in rows[0]}}
            rep["leakage"] = lk
            save()
            print("leakage", c, lk[c], flush=True)
            del arm

    # 4. per-rotation cost of one stats chunk
    if "cost" in parts:
        cost = {}
        for c in a.configs.split(","):
            arm, d, acq = build(c, a.dataset, "crowded", 2)
            truth, maps = truth_and_maps(d, acq, CONFIGS[c][0])
            t0 = time.time()
            arm.prepare_particle(0)
            jax.effects_barrier()
            cost[f"{c}_prepare_s"] = time.time() - t0
            grids = (("int125", PL.shift_grid(np.zeros(3), 2.0, 1.0)), ("half27", PL.shift_grid(np.zeros(3), 0.5, 0.5)))
            if CONFIGS[c][0] == 64:
                grids = (("int729", us_int), ("half125", us_half))
            for name, us in grids:
                try:
                    _, ms = chunk_stats(arm, maps, truth, 0, us)
                    cost[f"{c}_{name}_ms_per_rot"] = ms
                except Exception as e:
                    cost[f"{c}_{name}_ms_per_rot"] = f"failed: {type(e).__name__}: {str(e)[:150]}"
            rep["cost"] = cost
            save()
            del arm
        print("cost", cost, flush=True)
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
