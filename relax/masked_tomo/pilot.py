"""Crowding pilot driver: the same pose search, M-step, PPCA EM and metrics for both measurement arms.

Arms: ``image`` (A, RELAX's sub-tilt image likelihood), ``masked`` (B, masked whitened scene reconstruction)
and ``masked_matched`` (B on the central target-box detector window only: the matched-field diagnostic).
Truth (poses, states, maps) is loaded only for the labelled oracle diagnostics and for scoring.
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
from scipy.spatial.transform import Rotation

from relax.masked_tomo import likelihoods as L

F32 = jnp.float32


# ----------------------------------------------------------------------------------------------- loading
def load_condition(root, condition, n, S, *, n_particles=None, noise_radius_px, need_patches=True, extra_noise_var=0.0,
                   extra_noise_seed=0, field="patch"):
    """Read a condition's crops (and patches) and Fourier-crop them to the working sizes.

    ``extra_noise_var`` > 0 adds white Gaussian noise of that per-pixel variance (data units) to every bin-1 detector
    patch, one stream per scene from ``extra_noise_seed`` (the same in every condition), and takes the crops from the
    noisy patches: the data of a lower-SNR acquisition of the same scenes (labelled exploratory option)."""
    acq = np.load(os.path.join(root, "acquisition.npz"))
    names = acq["particle_names"]
    P = len(names) if n_particles is None else min(n_particles, len(names))
    crops, patches = [], []
    for p in range(P):
        tomo, idx = str(names[p]).split("/")
        if extra_noise_var > 0:
            with mrcfile.open(os.path.join(root, condition, "ScenePatches", tomo, f"{idx}_patch2d.mrcs")) as m:
                patch = np.asarray(m.data, np.float64)
            rng = np.random.default_rng(np.random.SeedSequence([int(extra_noise_seed), p]))
            patch = (patch + np.sqrt(extra_noise_var) * rng.standard_normal(patch.shape)).astype(np.float32)
            N, M = int(acq["target_box"]), patch.shape[-1]
            lo = (M - N) // 2
            crops.append(patch[:, lo : lo + N, lo : lo + N])
            if need_patches:
                patches.append(patch)
            continue
        with mrcfile.open(os.path.join(root, condition, "Subtomograms", tomo, f"{idx}_stack2d.mrcs")) as m:
            crops.append(np.asarray(m.data, np.float32))
        if need_patches:
            with mrcfile.open(os.path.join(root, condition, "ScenePatches", tomo, f"{idx}_patch2d.mrcs")) as m:
                patches.append(np.asarray(m.data, np.float32))
    crops = np.stack(crops)
    patches = np.stack(patches) if need_patches else None
    if field == "crop":  # arm B's detector field is the target crop itself (matched field, S == n)
        patches = crops
    d = L.prepare(crops, patches, acq["rots"][:P], acq["ctf_params"][:P], float(acq["voxel_size"]), n, S,
                  noise_radius_px)
    return d, acq


def load_truth(root, n, n_particles, rt):
    """Hidden truth for scoring/oracle diagnostics: target ``Q``, ``u`` (working px), coefficients, basis maps."""
    obj = np.load(os.path.join(root, "truth", "objects.npz"))
    tgt = np.nonzero(obj["kind"] == 0)[0][:n_particles]
    assert np.array_equal(obj["scene"][tgt], np.arange(len(tgt)))
    rots = obj["rots"][tgt]  # (P, K, 3, 3)
    Q = np.einsum("pkba,pkbc->pkac", rt, rots)
    q_spread = float(np.abs(Q - Q[:, :1]).max())
    Q = Q[:, 0]
    shifts = obj["shifts_px"][tgt]  # bin-1 px
    acq = np.load(os.path.join(root, "acquisition.npz"))
    bin_factor = int(acq["target_box"]) / n
    u = np.zeros((len(tgt), 3))
    res = 0.0
    for p in range(len(tgt)):
        A = rt[p][:, :2, :].reshape(-1, 3)
        # The simulator applies translate_images(slices, -shift) (simulator.simulate_batch); the model's tilt
        # shift t_k(u) = (rt_k u)[:2] enters as translate_images(slices, +t). So the true u solves rt u = -shift.
        b = -shifts[p].reshape(-1) / bin_factor
        u[p] = np.linalg.lstsq(A, b, rcond=None)[0]
        res = max(res, float(np.abs(A @ u[p] - b).max()))
    coeff = obj["coefficients"][tgt]
    nbr = {k: obj[k] for k in ("scene", "kind", "offset", "coefficients")}
    return {"Q": Q, "u": u, "coeff": coeff, "q_spread": q_spread, "u_residual": res, "objects": nbr}


def basis_maps(n):
    """Working-size basis maps of the families (RECOVAR assets), as in masked-tomo/scripts/make_datasets.py."""
    import recovar.utils as rutils
    from recovar.simulation import tomo_scene

    assets = os.path.join(os.path.dirname(tomo_scene.__file__), "..", "assets")
    vols = np.stack([np.asarray(rutils.load_mrc(os.path.join(assets, f"vol000{i}.mrc")), np.float64) for i in range(3)])
    m = vols.mean(0)
    Xc = (vols - m).reshape(3, -1)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    w = (s[:2, None] / np.sqrt(3)) * Vt[:2]
    het = np.concatenate([m[None], w.reshape(2, *m.shape)])
    return {
        "homo": np.stack([L.fourier_crop3(vols[0], n)]),
        "het2": np.stack([L.fourier_crop3(v, n) for v in het]),
        # het2 with doubled loadings (dense-crowding pilot, for statistical power)
        "het2L": np.stack([L.fourier_crop3(v, n) for v in np.concatenate([het[:1], 2.0 * het[1:]])]),
    }


def lowpass(x, pixel, resolution):
    """Gaussian Fourier low-pass with amplitude 1/e at ``1/resolution``."""
    n = x.shape[-1]
    k = np.sqrt(np.sum(L.grid3(n) ** 2, -1)).reshape((n,) * 3) / (n * pixel)
    ft = np.asarray(L.ftu.get_dft3(jnp.asarray(x, jnp.float64)))
    return np.real(np.asarray(L.ftu.get_idft3(jnp.asarray(ft * np.exp(-((k * resolution) ** 2)))))).astype(np.float32)


# ----------------------------------------------------------------------------------------------- poses
def rotvec_lattice(radius_deg, step_deg):
    m = int(np.floor(radius_deg / step_deg))
    g = np.arange(-m, m + 1) * step_deg
    w = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    w = w[np.linalg.norm(w, axis=1) <= radius_deg + 1e-9]
    return np.deg2rad(w)


def candidates(Q0, rotvecs):
    return np.einsum("ab,rbc->rac", Q0, Rotation.from_rotvec(rotvecs).as_matrix()).astype(np.float32)


def angle_deg(Qa, Qb):
    c = (np.einsum("...ab,...ab->...", Qa, Qb) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def shift_grid(center, radius, step):
    m = int(round(radius / step))
    g = np.arange(-m, m + 1) * step
    return (np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) + center).astype(np.float32)


def _group_by_fraction(us):
    frac = np.round((us - np.round(us)) * 1000) / 1000
    keys = {}
    for i, f in enumerate(map(tuple, frac)):
        keys.setdefault(f, []).append(i)
    return [np.array(v) for v in keys.values()]


def score_grid(arm, vft, part, Qs, us, chunk):
    """Scores ``(R, T)``, latent means ``(q, R, T)`` and covariances ``(R, T, tri)`` of one particle."""
    R, T = Qs.shape[0], us.shape[0]
    P = vft.shape[0]
    pad = (-R) % chunk
    Qp = np.concatenate([Qs, np.repeat(Qs[-1:], pad, 0)]) if pad else Qs
    dep = arm.name != "image"
    s_all = np.zeros((R, T), np.float32)
    m_all = np.zeros((P - 1, R, T), np.float32)
    c_all = np.zeros((R, T, max(1, (P - 1) * P // 2)), np.float32)
    groups = _group_by_fraction(us) if dep else [np.arange(T)]
    for g in groups:
        for a in range(0, Qp.shape[0], chunk):
            inner, gram = _stats_jit(arm, vft, part, jnp.asarray(Qp[a : a + chunk]), jnp.asarray(us[g]))
            s, m, c = L.scores_from_stats(inner, gram, dep)
            b = min(a + chunk, R) - a
            if b <= 0:
                continue
            s_all[a : a + b, g] = np.asarray(s)[:b]
            if P > 1:
                m_all[:, a : a + b][:, :, g] = np.asarray(m)[:, :b]
                c_all[a : a + b][:, g] = np.asarray(c)[:b]
    return s_all, m_all, c_all


_STATS_CACHE = {}


def _stats_jit(arm, vft, part, Qs, us):
    key = (id(arm), Qs.shape, us.shape, vft.shape)
    if key not in _STATS_CACHE:
        _STATS_CACHE[key] = jax.jit(lambda v, p, q, u: arm.stats(v, p, q, u))
    return _STATS_CACHE[key](vft, part, Qs, us)


def split_shifts(us):
    """``us (T, 3) = deltas[gid] + ints``: sub-pixel parts ``deltas (D, 3)`` in ``[0, 1)`` (one per distinct
    fractional part, to 1e-3), integer parts ``ints (T, 3)`` and the part ``gid (T,)`` of every shift."""
    us = np.asarray(us, np.float32)
    frac = us - np.floor(us)
    key = np.round(frac * 1000).astype(np.int64) % 1000
    _, first, gid = np.unique(key, axis=0, return_index=True, return_inverse=True)
    deltas = frac[first].astype(np.float32)
    gid = gid.reshape(-1)
    ints = np.round(us - deltas[gid]).astype(np.int32)
    return deltas, ints, gid.astype(np.int32)


# Complex elements of the largest per-call buffer (particles x rotations x sub-pixel parts x components x S^3) in the
# batched masked search; the call's peak is a small multiple of it (2^27 elements is 1 GiB).
SEARCH_BUDGET = 2**28
SEARCH_PARTICLES = 8
# The image arm is fastest one particle per call with a smaller buffer (GPU job 7483092, n = 64, pilot schedule:
# A1 77 ms per particle against 186 for the per-particle loop of cc40162f and 241 with the masked arms' settings).
SEARCH_BUDGET_IMAGE = 2**26
SEARCH_PARTICLES_IMAGE = 1

_SCORE_CACHE = {}


def _jit_cached(key, arm, build):
    hit = _SCORE_CACHE.get((key, id(arm)))
    if hit is not None and hit[0] is arm:
        return hit[1]
    fn = jax.jit(build())
    _SCORE_CACHE[(key, id(arm))] = (arm, fn)
    return fn


def _scores_batched(arm):
    """Jitted ``(B, R, T)`` exact pose scores of ``B`` particles (each with its own candidates)."""
    if arm.name == "image":

        def build():
            def f(vft, parts, Qs, us):
                def one(part, Q, u):
                    inner, gram = arm.stats(vft, part, Q, u)
                    return L.scores_from_stats(inner, gram, False)[0]

                return jax.vmap(one)(parts, Qs, us)

            return f

        return _jit_cached("image", arm, build)

    def build():
        def f(vft, parts, Qs, deltas, ints, gid):
            def one(part, Q, d, i, g):
                inner, gram = arm.stats_multi(vft, part, Q, d, i, g)
                return L.scores_from_stats(inner, gram, True)[0]

            return jax.vmap(one)(parts, Qs, deltas, ints, gid)

        return f

    return _jit_cached("masked", arm, build)


def _prefilter_batched(arm):
    """Jitted ``(B, R, T)`` prefilter scores of the masked arms: exact inner, unmasked gram (``L._fourier_stats``)."""

    def build():
        def f(vft, parts, Qs, axes):
            def one(part, Q, ax):
                inner, gram = L._fourier_stats(arm, vft, part, Q, ax)
                return L.scores_from_stats(inner, jnp.broadcast_to(gram[:, :, None], gram.shape + (inner.shape[-1],)), True)[0]

            return jax.vmap(one)(parts, Qs, axes)

        return f

    return _jit_cached("prefilter", arm, build)


def _search_batched(arm, X, ids, Q0s, u0s, schedule):
    """:func:`search` with ``SEARCH_PARTICLES`` particles and a memory-sized rotation chunk per call and the latent
    epilogue fused in the same jit. Masked arms: every sub-pixel part of a shift grid in one call; with
    ``arm.prefilter_k`` set, every stage but the last scores the full grid with the Fourier prefilter and rescores
    the ``prefilter_k`` best rotations (by their best shift) exactly, so chosen poses and the reported posterior
    summary always come from exact scores."""
    vft = arm.volumes_ft(jnp.asarray(X))
    image = arm.name == "image"
    P = vft.shape[0]
    if image:
        per_rot = P * arm.d.rt.shape[1] * arm.G**2  # projected padded slices of every tilt
        keys = ("rt", "ctf", "yw", "w")
    else:
        per_rot = P * arm.N**3  # the template box sets the per-rotation buffers
        keys = ("V", "c", "filt")
    Qb = np.array(Q0s, np.float32)
    ub = np.array(u0s, np.float32)
    post = np.zeros(len(ids))
    fn = _scores_batched(arm)
    pk = None if image else getattr(arm, "prefilter_k", None)
    n = len(ids)
    B = min(SEARCH_PARTICLES_IMAGE if image else SEARCH_PARTICLES, n)
    budget = SEARCH_BUDGET_IMAGE if image else SEARCH_BUDGET

    def chunked(call, Qs, D):
        R = Qs.shape[1]
        rc = int(max(1, min(R, budget // (B * D * per_rot))))
        pad = (-R) % rc
        Qp = np.concatenate([Qs, np.repeat(Qs[:, -1:], pad, 1)], 1) if pad else Qs
        return np.concatenate([np.asarray(call(jnp.asarray(Qp[:, r : r + rc]))) for r in range(0, R + pad, rc)], 1)[:, :R]

    for si, (radius, step, srad, sstep) in enumerate(schedule):
        rv = rotvec_lattice(radius, step)
        last = si == len(schedule) - 1
        for a in range(0, n, B):
            js = np.arange(a, a + B).clip(max=n - 1)  # the last batch repeats its last particle
            parts = [arm.particle(int(ids[j])) for j in js]
            parts = {k: jnp.stack([p[k] for p in parts]) for k in keys}
            Qs = np.stack([candidates(Qb[j], rv) for j in js])
            centers = np.stack([np.round(ub[j] / sstep) * sstep if sstep >= 1 else ub[j] for j in js])
            us = np.stack([shift_grid(c, srad, sstep) for c in centers])
            if image:
                uj = jnp.asarray(us)
                exact = lambda Q: fn(vft, parts, Q, uj)  # noqa: E731
                D = 1
            else:
                sp = [split_shifts(u) for u in us]
                D = max(x[0].shape[0] for x in sp)
                deltas = jnp.asarray(np.stack([np.concatenate([x[0], np.repeat(x[0][:1], D - x[0].shape[0], 0)]) for x in sp]))
                ints = jnp.asarray(np.stack([x[1] for x in sp]))
                gid = jnp.asarray(np.stack([x[2] for x in sp]))
                exact = lambda Q: fn(vft, parts, Q, deltas, ints, gid)  # noqa: E731
            if pk and not last and pk < rv.shape[0]:
                m = int(round(srad / sstep))
                g = np.arange(-m, m + 1) * sstep
                axes = jnp.asarray((centers[:, :, None] + g[None, None, :]).astype(np.float32))  # as shift_grid
                approx = chunked(lambda Q: _prefilter_batched(arm)(vft, parts, Q, axes), Qs, 1)
                top = np.argsort(-approx.max(-1), axis=1)[:, :pk]  # (B, k) best rotations by their best shift
                Qs = np.take_along_axis(Qs, top[:, :, None, None], 1)
            s = chunked(exact, Qs, D)
            for b, j in enumerate(js[: min(B, n - a)]):
                r, t = np.unravel_index(np.argmax(s[b]), s[b].shape)
                Qb[j], ub[j] = Qs[b, r], us[b, t]
                p = np.exp(s[b] - s[b].max())
                post[j] = float(p.max() / p.sum())
    return Qb, ub, post


def search(arm, X, ids, Q0s, u0s, schedule, chunk):
    """Local pose search: for every particle and stage ``(radius_deg, step_deg, shift_radius, shift_step)``,
    candidates around the previous best; returns MAP ``Q``, ``u`` and the last stage's posterior summary.
    ``chunk`` is kept for the callers; the rotation chunk is sized by ``SEARCH_BUDGET`` (:func:`_search_batched`)."""
    return _search_batched(arm, X, ids, Q0s, u0s, schedule)


# ----------------------------------------------------------------------------------------------- M-step
def _forward_pair(arm):
    """``lift, proj, data``: the forward map of a particle is ``proj(lift(X), part, Q, u)`` ``(P, m)``, with
    ``lift`` shared by every particle (the masked arms' padded DFTs; the identity for the image arm), so the normal
    operator sums the particles' cotangents before one adjoint of ``lift``."""
    if arm.name == "image":

        def fwd(x, part, Q, u):
            rt = part["rt"]
            rows = jnp.einsum("kab,bc->kac", rt, Q)
            ph = arm.phases(rt, u[None])[0]
            v = arm.volumes_ft(x[None])[0]
            p = arm.slice(v, rows) * part["ctf"] * ph * jnp.sqrt(part["w"])
            return jnp.concatenate([p.real.reshape(-1), p.imag.reshape(-1)])

        def data(part):
            y = part["yw"] / jnp.where(part["w"] > 0, jnp.sqrt(part["w"]), 1.0)
            return jnp.concatenate([y.real.reshape(-1), y.imag.reshape(-1)])

        def proj(X, part, Q, u):  # (P, m)
            return jax.vmap(lambda x: fwd(x, part, Q, u))(X)

        def lift(X):
            return X

    else:

        def proj(vft, part, Q, u):  # (P, m): every component through one gather per trilinear corner
            f = arm.filtered(vft, Q, u, part["filt"]) * arm.mask / jnp.sqrt(part["c"])
            return f.reshape(vft.shape[0], -1)

        def data(part):
            return (part["V"] / jnp.sqrt(part["c"])).reshape(-1)

        lift = arm.volumes_ft

    return lift, proj, data


def _stack_parts(arm, ids):
    parts = [arm.particle(int(i)) for i in ids]
    return {k: jnp.stack([p[k] for p in parts]) for k in parts[0] if k in ("rt", "ctf", "yw", "w", "V", "c", "filt")}


_PROBLEM_CACHE = {}


def _problem_programs(arm):
    """The jitted E-step statistics, normal operator and right-hand side of one arm, compiled once per arm and
    batch shape (not once per :class:`Problem`)."""
    hit = _PROBLEM_CACHE.get(id(arm))
    if hit is not None and hit[0] is arm:
        return hit[1]
    lift, proj, data = _forward_pair(arm)

    def estats(X, parts, Q, u):
        Y = lift(X)

        def one(part, Qi, ui):
            f = proj(Y, part, Qi, ui)
            d = data(part)
            return f @ d, f @ f.T, d @ d

        return jax.vmap(one)(parts, Q, u)

    def normal(X, parts, Q, u, G):
        Y, back = jax.vjp(lift, X)

        def one(part, Qi, ui, Gi):
            f, vjp = jax.vjp(lambda YY: proj(YY, part, Qi, ui), Y)
            return vjp(Gi @ f)[0]

        return back(jnp.sum(jax.vmap(one)(parts, Q, u, G), 0))[0]

    def rhs(X, parts, Q, u, g):
        Y, back = jax.vjp(lift, X)

        def one(part, Qi, ui, gi):
            _, vjp = jax.vjp(lambda YY: proj(YY, part, Qi, ui), Y)
            d = data(part)
            return vjp(gi[:, None] * d[None])[0]

        return back(jnp.sum(jax.vmap(one)(parts, Q, u, g), 0))[0]

    programs = (jax.jit(estats), jax.jit(normal), jax.jit(rhs))
    _PROBLEM_CACHE[id(arm)] = (arm, programs)
    return programs


class Problem:
    """Fixed-pose quadratic pieces of one arm over a set of particles (chunks of ``chunk`` particles)."""

    def __init__(self, arm, ids, Qs, us, chunk=8):
        self.arm, self.chunk = arm, chunk
        self.batches = []
        for a in range(0, len(ids), chunk):
            sl = slice(a, a + chunk)
            self.batches.append((_stack_parts(arm, ids[sl]), jnp.asarray(Qs[sl]), jnp.asarray(us[sl])))
        self._estats, self._normal, self._rhs = _problem_programs(arm)

    def e_step(self, X):
        """Per particle: ``inner (P,)``, ``gram (P,P)``, data norm; then latent mean/cov via RELAX's epilogue."""
        P = X.shape[0]
        out_m, out_c, ll = [], [], []
        first, second = np.triu_indices(P)
        for parts, Q, u in self.batches:
            inner, gram, dd = self._estats(X, parts, Q, u)
            gpack = gram[:, first, second].T[:, :, None]  # (tri, B, 1)
            s, m, c = L.latent_scores(inner.T[:, :, None, None], gpack, jnp.zeros((1, inner.shape[0], 1)))
            out_m.append(np.asarray(m[:, :, 0, 0]).T)
            out_c.append(np.asarray(c[:, 0]) if P > 1 else np.zeros((inner.shape[0], 0)))
            ll.append(np.asarray(s[:, 0, 0]) - 0.5 * np.asarray(dd))
        return np.concatenate(out_m), np.concatenate(out_c), np.concatenate(ll)

    def m_step(self, X0, means, covs, reg, support, iters=40, band=None):
        """Solve ``B (sum_i J_i^T J_i (B X) G_i + reg B X) = B sum_i J_i^T d_i g_i^T`` by CG, with ``B`` the
        projector onto the support (``band`` None) or, per component, onto the Fourier band ``band[p]`` (a radius
        in working Fourier pixels, None for the full band); the forward maps apply the support themselves."""
        P = X0.shape[0]
        q = P - 1
        Gs, gs = [], []
        for i in range(means.shape[0]):
            g = np.concatenate([[1.0], means[i]])
            G = np.outer(g, g)
            if q:
                G[1:, 1:] += L.unpack_tri(covs[i], q)
            Gs.append(G)
            gs.append(g)
        Gs, gs = np.asarray(Gs, np.float32), np.asarray(gs, np.float32)
        sup = jnp.asarray(support)
        if band is None:
            def proj(X):
                return X * sup
        else:
            n = X0.shape[-1]
            r = np.sqrt(np.sum(L.grid3(n) ** 2, -1)).reshape((n,) * 3)
            masks = jnp.asarray(np.stack([np.ones_like(r) if b_ is None else (r <= b_).astype(float) for b_ in band]))

            def proj(X):
                ft = L.ftu.get_dft3(X.astype(jnp.complex64))
                return jnp.real(L.ftu.get_idft3(ft * masks)).astype(X.dtype)
        b = 0
        for k, (parts, Q, u) in enumerate(self.batches):
            sl = slice(k * self.chunk, k * self.chunk + Q.shape[0])
            b = b + self._rhs(jnp.asarray(X0), parts, Q, u, jnp.asarray(gs[sl]))
        b = proj(b)

        def A(X):
            X = proj(X)
            out = reg * X
            for k, (parts, Q, u) in enumerate(self.batches):
                sl = slice(k * self.chunk, k * self.chunk + Q.shape[0])
                out = out + self._normal(X, parts, Q, u, jnp.asarray(Gs[sl]))
            return proj(out)

        x = proj(jnp.asarray(X0))
        r = b - A(x)
        p = r
        rs = jnp.sum(r * r)
        rs0 = rs
        for _ in range(iters):
            Ap = A(p)
            alpha = rs / jnp.sum(p * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = jnp.sum(r * r)
            if float(rs_new) < 1e-10 * float(rs0):
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return np.asarray(proj(x)), float(jnp.sqrt(rs_new / rs0))


# ----------------------------------------------------------------------------------------------- metrics
def fsc(a, b, pixel):
    n = a.shape[-1]
    fa = np.asarray(L.ftu.get_dft3(jnp.asarray(a, jnp.float64))).reshape(-1)
    fb = np.asarray(L.ftu.get_dft3(jnp.asarray(b, jnp.float64))).reshape(-1)
    r = np.round(np.sqrt(np.sum(L.grid3(n) ** 2, -1))).astype(int)
    out = []
    for s in range(n // 2):
        k = r == s
        num = np.real(np.sum(fa[k] * np.conj(fb[k])))
        den = np.sqrt(np.sum(np.abs(fa[k]) ** 2) * np.sum(np.abs(fb[k]) ** 2))
        out.append(num / den if den > 0 else 0.0)
    out = np.array(out)
    freq = np.arange(n // 2) / (n * pixel)
    below = np.nonzero(out < 0.5)[0]
    res05 = 1 / freq[below[0]] if below.size and below[0] > 0 else (np.inf if below.size else 1 / freq[-1])
    return out, float(res05)


def principal_sines(A, B):
    """Sines of the principal angles between the column spans of ``A`` and ``B``."""
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.sqrt(np.clip(1 - s**2, 0, 1))


def latent_r2(z_true, z_est):
    """Gauge-free latent recovery: R^2 of the best affine map from ``z_est`` to each true coordinate."""
    Xd = np.concatenate([z_est, np.ones((len(z_est), 1))], 1)
    out = []
    for j in range(z_true.shape[1]):
        coef, *_ = np.linalg.lstsq(Xd, z_true[:, j], rcond=None)
        resid = z_true[:, j] - Xd @ coef
        out.append(1 - resid.var() / z_true[:, j].var())
    return np.array(out)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _C:
            def __enter__(self):
                self.t0 = time.time()

            def __exit__(self, *a):
                jax.effects_barrier()
                timer.t[name] = timer.t.get(name, 0.0) + time.time() - self.t0

        return _C()
