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
def load_condition(root, condition, n, S, *, n_particles=None, noise_radius_px, need_patches=True):
    acq = np.load(os.path.join(root, "acquisition.npz"))
    names = acq["particle_names"]
    P = len(names) if n_particles is None else min(n_particles, len(names))
    crops, patches = [], []
    for p in range(P):
        tomo, idx = str(names[p]).split("/")
        with mrcfile.open(os.path.join(root, condition, "Subtomograms", tomo, f"{idx}_stack2d.mrcs")) as m:
            crops.append(np.asarray(m.data, np.float32))
        if need_patches:
            with mrcfile.open(os.path.join(root, condition, "ScenePatches", tomo, f"{idx}_patch2d.mrcs")) as m:
                patches.append(np.asarray(m.data, np.float32))
    crops = np.stack(crops)
    patches = np.stack(patches) if need_patches else None
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


def search(arm, X, ids, Q0s, u0s, schedule, chunk):
    """Local pose search: for every particle and stage ``(radius_deg, step_deg, shift_radius, shift_step)``,
    candidates around the previous best; returns MAP ``Q``, ``u`` and the last stage's posterior summary."""
    vft = arm.volumes_ft(jnp.asarray(X))
    Qb = np.array(Q0s, np.float32)
    ub = np.array(u0s, np.float32)
    post = np.zeros(len(ids))
    for radius, step, srad, sstep in schedule:
        rv = rotvec_lattice(radius, step)
        for j, i in enumerate(ids):
            part = arm.particle(int(i))
            Qs = candidates(Qb[j], rv)
            us = shift_grid(np.round(ub[j] / sstep) * sstep if sstep >= 1 else ub[j], srad, sstep)
            s, _, _ = score_grid(arm, vft, part, Qs, us, chunk)
            r, t = np.unravel_index(np.argmax(s), s.shape)
            Qb[j], ub[j] = Qs[r], us[t]
            p = np.exp(s - s.max())
            post[j] = float(p.max() / p.sum())
    return Qb, ub, post


# ----------------------------------------------------------------------------------------------- M-step
def _forward_pair(arm):
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

    else:

        def fwd(x, part, Q, u):
            v = arm.volumes_ft(x[None])
            return (arm.filtered(v, Q, u, part["filt"])[0] * arm.mask / jnp.sqrt(part["c"])).reshape(-1)

        def data(part):
            return (part["V"] / jnp.sqrt(part["c"])).reshape(-1)

    return fwd, data


def _stack_parts(arm, ids):
    parts = [arm.particle(int(i)) for i in ids]
    return {k: jnp.stack([p[k] for p in parts]) for k in parts[0] if k in ("rt", "ctf", "yw", "w", "V", "c", "filt")}


class Problem:
    """Fixed-pose quadratic pieces of one arm over a set of particles (chunks of ``chunk`` particles)."""

    def __init__(self, arm, ids, Qs, us, chunk=8):
        self.arm, self.chunk = arm, chunk
        self.fwd, self.data = _forward_pair(arm)
        self.batches = []
        for a in range(0, len(ids), chunk):
            sl = slice(a, a + chunk)
            self.batches.append((_stack_parts(arm, ids[sl]), jnp.asarray(Qs[sl]), jnp.asarray(us[sl])))
        fwd, data = self.fwd, self.data

        def proj_all(X, part, Q, u):  # (P, m)
            return jax.vmap(lambda x: fwd(x, part, Q, u))(X)

        def estats(X, parts, Q, u):
            def one(part, Qi, ui):
                f = proj_all(X, part, Qi, ui)
                d = data(part)
                return f @ d, f @ f.T, d @ d

            return jax.vmap(one)(parts, Q, u)

        def normal(X, parts, Q, u, G):
            def one(part, Qi, ui, Gi):
                f, vjp = jax.vjp(lambda XX: proj_all(XX, part, Qi, ui), X)
                return vjp(Gi @ f)[0]

            return jnp.sum(jax.vmap(one)(parts, Q, u, G), 0)

        def rhs(X, parts, Q, u, g):
            def one(part, Qi, ui, gi):
                _, vjp = jax.vjp(lambda XX: proj_all(XX, part, Qi, ui), X)
                d = data(part)
                return vjp(gi[:, None] * d[None])[0]

            return jnp.sum(jax.vmap(one)(parts, Q, u, g), 0)

        self._estats = jax.jit(estats)
        self._normal = jax.jit(normal)
        self._rhs = jax.jit(rhs)

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

    def m_step(self, X0, means, covs, reg, support, iters=40):
        """Solve ``sum_i J_i^T J_i X G_i + reg X = sum_i J_i^T d_i g_i^T`` (support-restricted) by CG."""
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
        b = 0
        for k, (parts, Q, u) in enumerate(self.batches):
            sl = slice(k * self.chunk, k * self.chunk + Q.shape[0])
            b = b + self._rhs(jnp.asarray(X0), parts, Q, u, jnp.asarray(gs[sl]))
        b = b * sup

        def A(X):
            out = reg * X
            for k, (parts, Q, u) in enumerate(self.batches):
                sl = slice(k * self.chunk, k * self.chunk + Q.shape[0])
                out = out + self._normal(X, parts, Q, u, jnp.asarray(Gs[sl]))
            return out * sup

        x = jnp.asarray(X0) * sup
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
        return np.asarray(x), float(jnp.sqrt(rs_new / rs0))


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
