"""Masked-tomo arm B: equivalence and timing of this checkout's ``relax.masked_tomo`` against a reference copy.

The reference is ``likelihoods_ref.py`` / ``pilot_ref.py`` in ``--ref-dir`` (``git show <base>:relax/masked_tomo/...``
with the pilot's import pointed at ``likelihoods_ref``). Both run in one process on the same loaded data.

usage: compare_masked_tomo_opt.py DATASET OUT_JSON --ref-dir DIR [--n-particles 64] [--condition crowded]
       [--stages equiv,profile,A1,A2,B2] [--smoke] [--impls ref,new]
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

N_WORK, SUPPORT_R, MASK_R, CROP, NOISE_R, LAM_REL, LOWPASS_A, W_BAND = 32, 13.5, 15.0, 40, 12.0, 0.1, 40.0, 6
S_WORK = None  # default: the dataset's scene box at the working pixel size
PASS1 = (20.0, 4.0, 2.0, 1.0)
PASS2 = (4.0, 4.0 / 3, 0.5, 0.5)
REFINE_LATER = [(8.0, 2.0, 1.0, 1.0), (2.5, 2.5 / 3, 0.5, 0.5)]
CHUNK = 32


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = np.max(np.abs(a))
    return float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(b)))


def timed(fn, *args, reps=3):
    out = jax.block_until_ready(fn(*args))  # compile + warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def load(root, condition, n_particles, L):
    """``pilot.load_condition`` with the detector patch centre-cropped (real space, bin 1) to ``S`` working px."""
    import mrcfile

    acq = np.load(os.path.join(root, "acquisition.npz"))
    N, M0 = int(acq["target_box"]), int(acq["scene_box"])
    S = S_WORK or M0 * N_WORK // N
    M = S * N // N_WORK
    if M * N_WORK != S * N or M > M0:
        raise ValueError(f"S={S} needs a {S * N / N_WORK}-px bin-1 patch (dataset patch {M0})")
    names = acq["particle_names"][:n_particles]
    crops, patches = [], []
    lo = (M0 - M) // 2
    for nm in names:
        tomo, idx = str(nm).split("/")
        with mrcfile.open(os.path.join(root, condition, "Subtomograms", tomo, f"{idx}_stack2d.mrcs")) as m:
            crops.append(np.asarray(m.data, np.float32))
        with mrcfile.open(os.path.join(root, condition, "ScenePatches", tomo, f"{idx}_patch2d.mrcs")) as m:
            patches.append(np.asarray(m.data, np.float32)[:, lo : lo + M, lo : lo + M])
    P = len(names)
    d = L.prepare(np.stack(crops), np.stack(patches), acq["rots"][:P], acq["ctf_params"][:P], float(acq["voxel_size"]),
                  N_WORK, S, NOISE_R)
    return d, acq, S


def _pruned_crop(arm, qft):
    """Centred inverse DFT + central crop one axis at a time (the crop after each axis): the same map as
    ``arm._to_crop`` with fewer 1D transforms on the later axes."""
    S, Cc = arm.S, arm.Cc
    lo = (S - Cc) // 2
    x = qft.reshape(qft.shape[:-1] + (S, S, S))
    for ax in (-1, -2, -3):
        x = jnp.fft.ifftshift(jnp.fft.ifft(jnp.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
        x = jax.lax.slice_in_dim(x, lo, lo + Cc, axis=x.ndim + ax)
    return jnp.real(x)


def _stats_fftcorr(arm, vft, part, Qs, us):
    """Every integer shift from circular cross-correlations on the crop (FFTs of f, f_a f_b, M V and M)."""
    delta = us[0] - jnp.round(us[0])
    ints = jnp.round(us - delta).astype(jnp.int32) % arm.Cc
    c = part["c"]
    MV = arm.mask * part["V"] / c
    P = vft.shape[0]
    first, second = np.triu_indices(P)
    vol = jnp.moveaxis(vft, 0, -1)
    f = jax.vmap(lambda Q: arm._filtered_channels(vol, Q, delta, part["filt"]))(Qs)  # (R, P, C, C, C)
    ax = (-3, -2, -1)
    corr = lambda A, b: jnp.real(jnp.fft.ifftn(A * jnp.conj(jnp.fft.fftn(b, axes=ax)), axes=ax))
    ci = corr(jnp.fft.fftn(MV), f)  # (R, P, C, C, C): ci[m] = sum_y MV(y + m) f(y)
    cg = corr(jnp.fft.fftn(arm.mask), f[:, first] * f[:, second]) / c
    pick = lambda x: x[..., ints[:, 0], ints[:, 1], ints[:, 2]]
    return jnp.moveaxis(pick(ci), 1, 0), jnp.moveaxis(pick(cg), 1, 0)


def strategies(arm, maps, X3, truth, PL, L, save, report):
    """Timings of batching and precompute alternatives on one particle's pass-1 chunk (new arm)."""
    out = {}
    i = 0
    part = arm.particle(i)
    rv = PL.rotvec_lattice(*PASS1[:2])
    us125 = jnp.asarray(PL.shift_grid(np.zeros(3), 2.0, 1.0))
    for P_, X in (("P1", maps[:1]), ("P3", X3)):
        vft = arm.volumes_ft(jnp.asarray(X))
        st = jax.jit(arm.stats)
        # rotation chunk size (per-rotation cost)
        for R in (8, 16, 32, 64, 128, 256):
            if R > rv.shape[0]:
                break
            Qs = jnp.asarray(PL.candidates(truth["Q"][i], rv[:R]))
            try:
                t, _ = timed(st, vft, part, Qs, us125)
                out[f"{P_}_vmap_rot_R{R}_ms_per_rot"] = 1e3 * t / R
            except Exception as e:  # out of memory at a large chunk is a result, not a failure of the run
                out[f"{P_}_vmap_rot_R{R}_ms_per_rot"] = f"failed: {type(e).__name__}"
        Qs = jnp.asarray(PL.candidates(truth["Q"][i], rv[:32]))
        # rotations sequential (lax.map), shifts as matmuls
        seq = jax.jit(lambda v, p, Q, u: jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1)[:, :, 0],
                                                       jax.lax.map(lambda q: arm.stats(v, p, q[None], u), Q)))
        t, o_seq = timed(seq, vft, part, Qs, us125)
        out[f"{P_}_laxmap_rot_R32_ms_per_rot"] = 1e3 * t / 32
        # shifts by FFT cross-correlation on the crop
        t, o_fft = timed(jax.jit(lambda v, p, Q, u: _stats_fftcorr(arm, v, p, Q, u)), vft, part, Qs, us125)
        out[f"{P_}_fftcorr_R32_ms_per_rot"] = 1e3 * t / 32
        o_ref = st(vft, part, Qs, us125)
        out[f"{P_}_fftcorr_vs_matmul_inner_rel"] = rel(o_ref[0], o_fft[0])
        out[f"{P_}_fftcorr_vs_matmul_gram_rel"] = rel(o_ref[1], o_fft[1])
        out[f"{P_}_laxmap_vs_vmap_inner_rel"] = rel(o_ref[0], o_seq[0])
        # pieces of one R = 32 chunk
        vol = jnp.moveaxis(vft, 0, -1)
        G, S = arm.G, arm.S
        gat = jax.jit(lambda Q: jax.vmap(lambda q: L.trilinear(vol, (arm.k3 @ q) * (G / S) + G // 2))(Q))
        t, g = timed(gat, Qs)
        out[f"{P_}_gather_R32_ms_per_rot"] = 1e3 * t / 32
        g = jnp.moveaxis(g, -1, 1)
        t, a1 = timed(jax.jit(arm._to_crop), g)
        out[f"{P_}_ifft_full_R32_ms_per_rot"] = 1e3 * t / 32
        t, a2 = timed(jax.jit(lambda q: _pruned_crop(arm, q)), g)
        out[f"{P_}_ifft_pruned_R32_ms_per_rot"] = 1e3 * t / 32
        out[f"{P_}_ifft_pruned_vs_full_rel"] = rel(a1, a2)
        f = a1.reshape(32, X.shape[0], -1)
        Vs = jax.vmap(lambda m: jnp.roll(part["V"], shift=(-m[0], -m[1], -m[2]), axis=(0, 1, 2)).reshape(-1))(
            jnp.round(us125).astype(jnp.int32))
        t, _ = timed(jax.jit(lambda f, V: jnp.einsum("rpx,tx->prt", f, V, precision="highest")), f, Vs)
        out[f"{P_}_shift_matmul_R32_T125_ms_per_rot"] = 1e3 * t / 32
        # batching over particles: B particles x 32 rotations in one call
        if X.shape[0] == 3:
            for B in (4, 16):
                ps = [arm.particle(j) for j in range(B)]
                parts = {k: jnp.stack([p[k] for p in ps]) for k in ("V", "c", "filt")}
                QB = jnp.asarray(np.stack([PL.candidates(truth["Q"][j], rv[:32]) for j in range(B)]))
                fb = jax.jit(jax.vmap(lambda p, Q: arm.stats(vft, p, Q, us125), in_axes=(0, 0)))
                try:
                    t, _ = timed(fb, parts, QB)
                    out[f"{P_}_particles_B{B}_R32_ms_per_rot"] = 1e3 * t / (32 * B)
                except Exception as e:
                    out[f"{P_}_particles_B{B}_R32_ms_per_rot"] = f"failed: {type(e).__name__}"
        report["strategies"] = out
        save()
    return out


def main():
    global PASS1, PASS2, REFINE_LATER, N_WORK, S_WORK, CROP, MASK_R, SUPPORT_R, NOISE_R, W_BAND, CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--n-particles", type=int, default=64)
    ap.add_argument("--condition", default="crowded")
    ap.add_argument("--stages", default="equiv,profile,A1,A2,B2")
    ap.add_argument("--impls", default="ref,new")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=32, help="working image size (target box Fourier-cropped to n)")
    ap.add_argument("--S", type=int, default=None, help="scene grid: the detector patch is first cropped in real "
                    "space to the bin-1 size of S working px (a smaller field than the dataset's patch)")
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--mask-r", type=float, default=MASK_R)
    ap.add_argument("--support-r", type=float, default=SUPPORT_R)
    ap.add_argument("--noise-r", type=float, default=NOISE_R)
    ap.add_argument("--w-band", type=int, default=W_BAND)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--search-particles", type=int, default=8)
    ap.add_argument("--template-box", type=int, default=None, help="'new' is KernelBoxMaskedArm with this box")
    ap.add_argument("--schedule", default="n32", choices=("n32", "pilot64"), help="search schedule (run_pilot.py)")
    ap.add_argument("--search-budget-log2", type=int, default=27)
    a = ap.parse_args()
    N_WORK, S_WORK, CROP, MASK_R, SUPPORT_R, NOISE_R, W_BAND, CHUNK = (a.n, a.S, a.crop, a.mask_r, a.support_r,
                                                                       a.noise_r, a.w_band, a.chunk)
    if a.schedule == "pilot64":  # run_pilot.py --n-work 64
        PASS1, PASS2 = (20.0, 3.0, 4.0, 1.0), (3.0, 1.0, 1.0, 0.5)
        REFINE_LATER = [(6.0, 1.5, 2.0, 1.0), (1.5, 0.5, 0.5, 0.5)]
    if a.smoke:
        PASS1, PASS2 = (8.0, 4.0, 1.0, 1.0), (2.0, 2.0, 0.5, 0.5)
        REFINE_LATER = [(4.0, 4.0, 1.0, 1.0)]
    sys.path.insert(0, a.ref_dir)
    import likelihoods_ref as Lref
    import pilot_ref as PLref

    from relax.masked_tomo import likelihoods as Lnew
    from relax.masked_tomo import pilot as PLnew

    mods = {"ref": (Lref, PLref), "new": (Lnew, PLnew)}
    PLnew.SEARCH_PARTICLES, PLnew.SEARCH_BUDGET = a.search_particles, 2**a.search_budget_log2
    impls = a.impls.split(",")
    stages = a.stages.split(",")
    report = {"args": vars(a), "devices": [str(d) for d in jax.devices()], "x64": bool(jax.config.jax_enable_x64)}

    def save():
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    t0 = time.time()
    d, acq, S = load(a.dataset, a.condition, a.n_particles, Lnew)
    report["load_seconds"] = time.time() - t0
    report["S"] = S
    report["geometry"] = {"n": N_WORK, "S": S, "G": 2 * N_WORK, "crop": CROP, "mask_r": MASK_R, "support_r": SUPPORT_R,
                          "pixel_A": float(d.pixel), "field_A": float(S * d.pixel), "chunk": CHUNK}
    P = d.rt.shape[0]
    ids = np.arange(P)
    truth = PLnew.load_truth(a.dataset, N_WORK, P, d.rt)
    maps = PLnew.basis_maps(N_WORK)["homo"] * float(acq["image_scale"])
    mu0 = PLnew.lowpass(maps[0], d.pixel, LOWPASS_A)
    reg = 1.0 / (10.0 * float(np.mean(mu0[Lnew.sphere(N_WORK, SUPPORT_R) > 0] ** 2)))
    rng = np.random.default_rng(10_000 + 0)
    pert = Rotation.from_rotvec(np.deg2rad(rng.normal(0, 5.0, (P, 3)))).as_matrix()
    Q0 = np.einsum("pab,pbc->pac", truth["Q"], pert).astype(np.float32)
    u0 = np.zeros((P, 3), np.float32)
    rng = np.random.default_rng(500)
    W0 = np.stack([PLnew.lowpass(rng.standard_normal((N_WORK,) * 3).astype(np.float32), d.pixel, LOWPASS_A) for _ in range(2)])
    W0 *= 0.1 * np.linalg.norm(mu0) / np.linalg.norm(W0)
    X3 = np.concatenate([mu0[None], W0]).astype(np.float32)

    def make(impl, name="masked"):
        L_, _ = mods[impl]
        if name == "image":
            return L_.ImageArm(d, SUPPORT_R)
        if impl == "new" and a.template_box:
            return L_.KernelBoxMaskedArm(d, SUPPORT_R, MASK_R, template_box=a.template_box, crop=CROP, lam_rel=LAM_REL,
                                         window=None, n_mc=2, seed=0)
        return L_.MaskedArm(d, SUPPORT_R, MASK_R, crop=CROP, lam_rel=LAM_REL, window=None, n_mc=2, seed=0)

    arms = {}
    for impl in impls:
        arm = make(impl)
        t0 = time.time()
        for i in ids:
            arm.prepare_particle(int(i))
        jax.effects_barrier()
        report[f"prepare_seconds_{impl}"] = time.time() - t0
        arms[impl] = arm
    image_arm = make("new", "image")

    # ------------------------------------------------------------------ equivalence of fixed-input operators
    if "equiv" in stages and len(impls) == 2:
        eq = {}
        ra, na = arms["ref"], arms["new"]
        eq["prepare"] = {k: max(rel(ra.prepare_particle(i)[k], na.prepare_particle(i)[k]) for i in ids)
                         for k in ("V", "c", "lam", "D", "filt")}
        rv = PLnew.rotvec_lattice(*PASS1[:2])[:CHUNK]
        cases = []
        for i in ids[:2]:
            for X in (maps[:1], X3):
                for us in (PLnew.shift_grid(np.zeros(3), 2.0, 1.0), PLnew.shift_grid(np.array([0.5, 0.0, -0.5]), 1.0, 1.0)):
                    Qs = jnp.asarray(PLnew.candidates(truth["Q"][i], rv))
                    out = {}
                    for impl, arm in arms.items():
                        vft = arm.volumes_ft(jnp.asarray(X))
                        out[impl] = [np.asarray(x) for x in jax.jit(arm.stats)(vft, arm.particle(int(i)), Qs, jnp.asarray(us))]
                    cases.append({"particle": int(i), "P": int(X.shape[0]), "T": int(us.shape[0]),
                                  "inner": rel(out["ref"][0], out["new"][0]), "gram": rel(out["ref"][1], out["new"][1]),
                                  "inner_scale": float(np.abs(out["ref"][0]).max()), "gram_scale": float(np.abs(out["ref"][1]).max())})
                    # The score that picks poses: compare it directly, with its dynamic range on this grid.
                    s = {impl: np.asarray(Lnew.scores_from_stats(jnp.asarray(o[0]), jnp.asarray(o[1]), True)[0]) for impl, o in out.items()}
                    cases[-1]["score_maxabs_diff"] = float(np.abs(s["ref"] - s["new"]).max())
                    cases[-1]["score_range"] = float(np.ptp(s["ref"]))
                    cases[-1]["argmax_same"] = bool(np.argmax(s["ref"]) == np.argmax(s["new"]))
        eq["stats"] = cases
        half = []
        for i in ids[:2]:
            for X in (maps[:1], X3):
                Qs = PLnew.candidates(truth["Q"][i], PLnew.rotvec_lattice(*PASS2[:2])[:CHUNK])
                us = PLnew.shift_grid(np.array([1.0, 0.0, -1.0]), 0.5, 0.5)
                vr = ra.volumes_ft(jnp.asarray(X))
                s_ref, m_ref, c_ref = PLref.score_grid(ra, vr, ra.particle(int(i)), Qs, us, CHUNK)
                dl, it, gd = PLnew.split_shifts(us)
                vn = na.volumes_ft(jnp.asarray(X))
                inner, gram = jax.jit(na.stats_multi)(vn, na.particle(int(i)), jnp.asarray(Qs), jnp.asarray(dl), jnp.asarray(it), jnp.asarray(gd))
                s_new, m_new, c_new = [np.asarray(x) for x in Lnew.scores_from_stats(inner, gram, True)]
                half.append({"particle": int(i), "P": int(X.shape[0]), "n_subpixel_parts_new": int(dl.shape[0]),
                             "score_maxabs_diff": float(np.abs(s_ref - s_new).max()), "score_range": float(np.ptp(s_ref)),
                             "argmax_same": bool(np.argmax(s_ref) == np.argmax(s_new)),
                             "mean_rel": rel(m_ref, m_new) if X.shape[0] > 1 else 0.0, "cov_rel": rel(c_ref, c_new) if X.shape[0] > 1 else 0.0})
        eq["half_pixel_grid"] = half
        # E-step and M-step operators on 8 particles, 3 components
        sub = ids[:8]
        probs = {impl: mods[impl][1].Problem(arms[impl], sub, truth["Q"][sub].astype(np.float32), truth["u"][sub].astype(np.float32))
                 for impl in impls}
        e = {impl: probs[impl].e_step(jnp.asarray(X3)) for impl in impls}
        eq["e_step"] = {k: rel(e["ref"][j], e["new"][j]) for j, k in enumerate(("means", "covs", "ll"))}
        parts, Qb, ub = probs["ref"].batches[0]
        G = jnp.asarray(np.tile(np.eye(3, dtype=np.float32), (len(sub), 1, 1)) + 0.1)
        nr = {impl: np.asarray(probs[impl]._normal(jnp.asarray(X3), *probs[impl].batches[0], G)) for impl in impls}
        eq["normal_op"] = rel(nr["ref"], nr["new"])
        g = jnp.asarray(np.ones((len(sub), 3), np.float32))
        rh = {impl: np.asarray(probs[impl]._rhs(jnp.asarray(X3), *probs[impl].batches[0], g)) for impl in impls}
        eq["rhs"] = rel(rh["ref"], rh["new"])
        report["equiv"] = eq
        save()
        print("equiv", json.dumps(eq, default=str)[:3000], flush=True)

    # ------------------------------------------------------------------ profile of one chunk
    if "profile" in stages:
        prof = {}
        # Matmul precision on this device: do default-precision float32 dots round operands to TF32?
        k = np.random.default_rng(1).standard_normal((512, 512)).astype(np.float32)
        ex = k.astype(np.float64) @ k.astype(np.float64)
        prof["default_dot_relerr"] = rel(ex, np.asarray(jnp.asarray(k) @ jnp.asarray(k)))
        prof["highest_dot_relerr"] = rel(ex, np.asarray(jnp.matmul(jnp.asarray(k), jnp.asarray(k), precision="highest")))
        Qt = jnp.asarray(Rotation.from_rotvec([0.3, -0.2, 0.7]).as_matrix().astype(np.float32))
        k3 = arms[impls[0]].k3
        c64 = np.asarray(k3, np.float64) @ np.asarray(Qt, np.float64)
        prof["k3_at_Q_eager_maxabs_err"] = float(np.abs(np.asarray(k3 @ Qt) - c64).max())
        prof["k3_at_Q_jit_maxabs_err"] = float(np.abs(np.asarray(jax.jit(lambda a, b: a @ b)(k3, Qt)) - c64).max())
        i = 0
        Qs = jnp.asarray(PLnew.candidates(truth["Q"][i], PLnew.rotvec_lattice(*PASS1[:2])[:CHUNK]))
        us125 = jnp.asarray(PLnew.shift_grid(np.zeros(3), 2.0, 1.0))
        for impl in impls:
            arm = arms[impl]
            part = arm.particle(i)
            for P_, X in (("P1", maps[:1]), ("P3", X3)):
                vft = arm.volumes_ft(jnp.asarray(X))
                f = jax.jit(arm.stats)
                prof[f"{impl}_{P_}_stats_R32_T125"], _ = timed(f, vft, part, Qs, us125)
                prof[f"{impl}_{P_}_stats_R32_T1"], _ = timed(f, vft, part, Qs, us125[:1])
            # pieces (P = 3): trilinear gather and the S^3 inverse DFT + crop for 32 rotations
            vft = arm.volumes_ft(jnp.asarray(X3))
            Lm = mods[impl][0]
            G_, S_ = arm.G, arm.S

            def gather(Qs, vft=vft):
                def one(Q):
                    coords = (arm.k3 @ Q) * (G_ / S_) + G_ // 2
                    return jax.vmap(lambda v: Lm.trilinear(v, coords))(vft)
                return jax.lax.map(one, Qs)

            prof[f"{impl}_gather_R32_P3_lax_map"], g = timed(jax.jit(gather), Qs)
            if impl == "new":
                prof["new_gather_R32_P3_channels_vmap"], _ = timed(jax.jit(lambda Qs: jax.vmap(
                    lambda Q: Lm.trilinear(jnp.moveaxis(vft, 0, -1), (arm.k3 @ Q) * (G_ / S_) + G_ // 2))(Qs)), Qs)
                prof["new_ifft_crop_R32_P3_batched"], _ = timed(jax.jit(arm._to_crop), g)
            else:
                prof["ref_ifft_crop_R32_P3_lax_map"], _ = timed(jax.jit(lambda g: jax.lax.map(lambda x: jax.vmap(arm._to_crop)(x), g)), g)
            # M-step pieces: one normal-operator call (8 particles, P = 3) and its compile
            sub = ids[:8]
            t0 = time.time()
            prob = mods[impl][1].Problem(arm, sub, truth["Q"][sub].astype(np.float32), truth["u"][sub].astype(np.float32))
            G = jnp.asarray(np.tile(np.eye(3, dtype=np.float32), (len(sub), 1, 1)))
            jax.block_until_ready(prob._normal(jnp.asarray(X3), *prob.batches[0], G))
            prof[f"{impl}_normal_first_call_s"] = time.time() - t0
            prof[f"{impl}_normal_B8_P3"], _ = timed(prob._normal, jnp.asarray(X3), *prob.batches[0], G)
            prof[f"{impl}_estats_B8_P3"], _ = timed(prob._estats, jnp.asarray(X3), *prob.batches[0])
            report["profile"] = prof
            save()
        print("profile", json.dumps(prof), flush=True)

    # ------------------------------------------------------------------ batching / precompute strategies (new arm)
    if "strategies" in stages and "new" in arms:
        report["strategies"] = strategies(arms["new"], maps, X3, truth, PLnew, Lnew, save, report)
        print("strategies", json.dumps(report["strategies"]), flush=True)

    if "sweep" in stages and "new" in arms:
        sw = {}
        arm = arms["new"]
        for B, budget in ((1, 2**27), (8, 2**27), (8, 2**28), (16, 2**28), (8, 2**29), (32, 2**29)):
            PLnew.SEARCH_PARTICLES, PLnew.SEARCH_BUDGET = B, budget
            try:
                PLnew.search(arm, maps[:1], ids, Q0[ids], u0[ids], [PASS1, PASS2], CHUNK)  # compile + run
                t0 = time.time()
                PLnew.search(arm, maps[:1], ids, Q0[ids], u0[ids], [PASS1, PASS2], CHUNK)
                sw[f"A1_B{B}_budget2^{int(np.log2(budget))}_s_per_particle"] = (time.time() - t0) / P
            except Exception as e:
                sw[f"A1_B{B}_budget2^{int(np.log2(budget))}_s_per_particle"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
            report["sweep"] = sw
            save()
        PLnew.SEARCH_PARTICLES, PLnew.SEARCH_BUDGET = a.search_particles, 2**a.search_budget_log2
        # M-step particle batch: one normal-operator application over all particles, P = 3
        for ch in (8, 32, 64):
            prob = PLnew.Problem(arm, ids, truth["Q"].astype(np.float32), truth["u"].astype(np.float32), chunk=ch)
            def apply_all(prob=prob):
                Gm = jnp.asarray(np.tile(np.eye(3, dtype=np.float32), (prob.batches[0][1].shape[0], 1, 1)))
                out = 0
                for parts, Qb, ub in prob.batches:
                    out = out + prob._normal(jnp.asarray(X3), parts, Qb, ub, Gm[: Qb.shape[0]])
                return out
            try:
                t, _ = timed(apply_all)
                sw[f"normal_all_chunk{ch}_ms_per_particle"] = 1e3 * t / P
            except Exception as e:
                sw[f"normal_all_chunk{ch}_ms_per_particle"] = f"failed: {type(e).__name__}"
            report["sweep"] = sw
            save()
        print("sweep", json.dumps(sw), flush=True)

    # ------------------------------------------------------------------ stages (timed after a warm-up on 8 particles)
    def run_A1(impl, arm, sel):
        PL_ = plmod[impl]
        return PL_.search(arm, maps[:1], sel, Q0[sel], u0[sel], [PASS1, PASS2], CHUNK)

    def reconstruct(PL_, arm, sel, Q, u, X0):
        prob = PL_.Problem(arm, sel, Q, u)
        return prob.m_step(X0, np.zeros((len(sel), X0.shape[0] - 1)), np.zeros((len(sel), 0)), reg, Lnew.sphere(N_WORK, SUPPORT_R), 40)

    def run_A2(impl, arm, sel):
        PL_ = plmod[impl]
        X = mu0[None].copy()
        Q, u = Q0[sel], u0[sel]
        for r in range(3):
            Q, u, post = PL_.search(arm, X, sel, Q, u, [PASS1, PASS2] if r == 0 else REFINE_LATER, CHUNK)
            X, _ = reconstruct(PL_, arm, sel, Q, u, X)
        return Q, u, X

    def run_B2(impl, arm, sel, n_iter=15):
        PL_ = plmod[impl]
        prob = PL_.Problem(arm, sel, truth["Q"][sel].astype(np.float32), truth["u"][sel].astype(np.float32))
        X = X3.copy()
        hist = []
        for _ in range(n_iter):
            m, c, ll = prob.e_step(jnp.asarray(X))
            hist.append(float(ll.sum()))
            X, _ = prob.m_step(X, m, c, reg, Lnew.sphere(N_WORK, SUPPORT_R), 30, band=[None, W_BAND, W_BAND])
        m, c, ll = prob.e_step(jnp.asarray(X))
        hist.append(float(ll.sum()))
        return X, m, c, hist

    runners = {"A1": run_A1, "A2": run_A2, "B2": run_B2}
    entries = [("image", image_arm)]
    plmod = {"image": PLnew, "new": PLnew, "ref": PLref, "image_ref": PLref}
    lmod = {"image": Lnew, "new": Lnew, "ref": Lref, "image_ref": Lref}
    if "ref" in arms:
        entries.append(("image_ref", Lref.ImageArm(d, SUPPORT_R)))
    entries += list(arms.items())
    allarms = dict(entries)
    timing = report.setdefault("timing", {})
    results = {}
    warm = ids[:8]
    for st in ("A1", "A2", "B2"):
        if st not in stages:
            continue
        for impl, arm in entries:
            t0 = time.time()
            if st == "B2":
                runners[st](impl, arm, warm, n_iter=1)
            else:
                runners[st](impl, arm, warm)
            jax.effects_barrier()
            timing[f"{st}_{impl}_warmup8_s"] = time.time() - t0
            t0 = time.time()
            results[(st, impl)] = runners[st](impl, arm, ids)
            jax.effects_barrier()
            timing[f"{st}_{impl}_s"] = time.time() - t0
            timing[f"{st}_{impl}_s_per_particle"] = timing[f"{st}_{impl}_s"] / P
            print(st, impl, timing[f"{st}_{impl}_s"], flush=True)
            save()
        for ra_, na_ in (("ref", "new"), ("image_ref", "image")):
            if ra_ not in allarms or na_ not in allarms:
                continue
            r, n = results[(st, ra_)], results[(st, na_)]
            cmp = {}
            if st in ("A1", "A2"):
                ang = PLnew.angle_deg(r[0], n[0])
                du = np.abs(r[1] - n[1]).max(1)
                diff = np.nonzero((ang > 0.1) | (du > 1e-4))[0]  # arccos round-off is ~0.03 deg; candidates are >= 0.8 deg apart
                cmp["n_pose_differences"] = int(diff.size)
                cmp["differences"] = []
                X = maps[:1] if st == "A1" else r[2]
                for j in diff:
                    row = {"particle": int(j), "angle_deg": float(ang[j]), "shift_diff": float(du[j])}
                    for impl in (ra_, na_):
                        arm = allarms[impl]
                        vft = arm.volumes_ft(jnp.asarray(X))
                        for who, res in (("ref_choice", r), ("new_choice", n)):
                            s, _, _ = plmod[impl].score_grid(arm, vft, arm.particle(int(j)), res[0][j][None], res[1][j][None], 1)
                            row[f"score_{impl}_at_{who}"] = float(s[0, 0])
                    cmp["differences"].append(row)
                if st == "A2":
                    cmp["X_rel"] = rel(r[2], n[2])
            else:
                cmp = {"X_rel": rel(r[0], n[0]), "m_rel": rel(r[1], n[1]), "c_rel": rel(r[2], n[2]),
                       "ll_hist_rel": rel(r[3], n[3])}
            tag = st if ra_ == "ref" else f"{st}_image"
            report[f"compare_{tag}"] = cmp
            print("compare", tag, json.dumps(cmp, default=str)[:2000], flush=True)
            save()
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
