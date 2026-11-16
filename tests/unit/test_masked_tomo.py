"""Masked-measurement backend: conventions and dense-Gaussian checks on a tiny scene (float64 diagnostic).

Exact checks assert; approximation checks (Fourier-diagonal mean, ``C ~ c I``) record their size in
``MASKED_TOMO_REPORT`` (a JSON path) when that variable is set, and assert only loose sanity bounds.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
from oracles.masked_tomo_dense import ppca_logpdf, support_pinv
from recovar.core import slicing
from scipy.spatial.transform import Rotation

from relax.masked_tomo import likelihoods as L

jax.config.update("jax_enable_x64", True)
S, n, K, PIX = 16, 8, 5, 17.0
REPORT = {}


def _ctf_params(K):
    p = np.zeros((K, 11))
    p[:, 0] = p[:, 1] = 25000.0
    p[:, 3], p[:, 4], p[:, 5], p[:, 8] = 300.0, 2.7, 0.1, 1.0
    p[:, 9] = np.arange(K) * 3.0
    return p


def _particles(rng, K=K, max_tilt=40.0):
    angles = np.linspace(-max_tilt, max_tilt, K)
    rt = Rotation.from_euler("y", angles, degrees=True).as_matrix() @ Rotation.from_euler("z", 85, degrees=True).as_matrix()
    patches = rng.standard_normal((1, K, S, S))
    d = L.Particles(
        images=None,
        patches=np.asarray(ftu.get_dft2(jnp.asarray(patches[0]))).reshape(1, K, S * S),
        rt=rt[None],
        ctf_img=None,
        ctf_patch=L.ctf_rows(_ctf_params(K), S, PIX)[None],
        noise_img=None,
        noise_patch=np.ones(S * S),
        shell_noise=None,
        n=n,
        S=S,
        pixel=PIX,
    )
    return d, patches


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    d, patches = _particles(rng)
    arm = L.MaskedArm(d, support_radius=3.5, mask_radius=2.5, crop=12, lam_rel=0.1, n_mc=1)
    rt = jnp.asarray(d.rt[0])
    ctf = jnp.asarray(d.ctf_patch[0])
    D, lam = arm.weights(ctf, rt)
    return {"rng": rng, "d": d, "arm": arm, "rt": rt, "ctf": ctf, "D": D, "lam": lam}


def _exact_images(x, Q, u, rt, ctf):
    """``A_g x``: the target placed at pose ``(Q, u)`` and sliced on the scene grid (patch DFTs ``(K, S*S)``)."""
    lo = (S - n) // 2
    big = jnp.zeros((S,) * 3).at[lo : lo + n, lo : lo + n, lo : lo + n].set(x)
    ft = ftu.get_dft3(big).reshape(-1).astype(jnp.complex128)
    rows = jnp.einsum("kab,bc->kac", rt, Q)
    img = slicing.slice_volume(ft, rows, (S, S), (S, S, S), "linear_interp")
    t2 = jnp.einsum("kab,b->ka", rt, u)[:, :2]
    ph = jnp.exp(-2j * jnp.pi * jnp.einsum("pa,ka->kp", jnp.asarray(L.plane_lattice(S)), t2))
    return img * ph * ctf


def _real_to_patch_ft(y, K=K):
    return ftu.get_dft2(y.reshape(K, S, S)).reshape(K, S * S)


def test_adjoint_dot(setup):
    rng = setup["rng"]
    rt = setup["rt"]
    v = rng.standard_normal(S**3) + 1j * rng.standard_normal(S**3)
    y = rng.standard_normal((K, S * S)) + 1j * rng.standard_normal((K, S * S))
    Av = slicing.slice_volume(jnp.asarray(v), rt, (S, S), (S, S, S), "linear_interp")
    Aty = slicing.adjoint_slice_volume(jnp.asarray(y), rt, (S, S), (S, S, S), "linear_interp")
    lhs, rhs = jnp.vdot(Av, y), jnp.vdot(jnp.asarray(v), Aty)
    REPORT["adjoint_rel"] = float(abs(lhs - rhs) / abs(lhs))
    # The projector forms its grid coordinates in float32 (relion_interp's centre), so the pair is adjoint
    # to float32 rounding: measured 1.3e-7 (job 7478820); the float32 default of helpers/float_compare is 1e-6.
    assert abs(lhs - rhs) <= 1e-6 * abs(lhs)


def _dense(setup, mask=None, K=K):
    """Dense ``L`` (masked crop voxels x real patch pixels) and the exact mean map ``x -> L A_g x``."""
    arm, rt, ctf, D, lam = setup["arm"], setup["rt"], setup["ctf"], setup["D"], setup["lam"]
    mask = arm.mask if mask is None else mask
    sel = np.nonzero(np.asarray(mask).reshape(-1) > 0)[0]

    def Lmap(y_real):
        return arm.measure(_real_to_patch_ft(y_real, K), ctf, rt, D, lam).reshape(-1)[sel]

    Lm = np.asarray(jax.jacfwd(Lmap)(jnp.zeros(K * S * S)))
    return Lm, sel, Lmap


def test_mean_and_conventions(setup):
    """B's Fourier-diagonal mean against the exact ``L A_g x``; wrong conventions are much worse."""
    arm, rt, ctf = setup["arm"], setup["rt"], setup["ctf"]
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((n,) * 3) * L.sphere(n, 3.5))
    Q = jnp.asarray(Rotation.random(random_state=3).as_matrix())
    u = jnp.asarray([0.7, -1.2, 0.4])
    _, sel, Lmap = _dense(setup)
    part = arm.particle_from(rt, ctf, setup["D"], setup["lam"])
    exact = Lmap(jnp.real(ftu.get_idft2(_exact_images(x, Q, u, rt, ctf).reshape(K, S, S))).reshape(-1))

    def approx(Qa, ua):
        f = arm.filtered(arm.volumes_ft(x[None]), Qa, ua, part["filt"])[0]
        return f.reshape(-1)[sel]

    errs = {
        "correct": float(jnp.linalg.norm(approx(Q, u) - exact) / jnp.linalg.norm(exact)),
        "Q_transposed": float(jnp.linalg.norm(approx(Q.T, u) - exact) / jnp.linalg.norm(exact)),
        "u_negated": float(jnp.linalg.norm(approx(Q, -u) - exact) / jnp.linalg.norm(exact)),
    }
    REPORT["mean_fourier_diagonal_rel_err"] = errs
    assert errs["correct"] < 0.5 * min(errs["Q_transposed"], errs["u_negated"])


def test_monte_carlo_noise(setup):
    arm = setup["arm"]
    Lm, sel, Lmap = _dense(setup)
    C = Lm @ Lm.T / S**2  # white pixel noise of variance 1/S^2 has unit DFT coefficient variance
    keys = jax.random.split(jax.random.PRNGKey(5), 4000)
    rt, ctf, D, lam = setup["rt"], setup["ctf"], setup["D"], setup["lam"]
    samples = jax.vmap(lambda k: arm.measure(arm.noise_sample(k, K), ctf, rt, D, lam).reshape(-1)[sel])(keys)
    emp = np.cov(np.asarray(samples).T)
    rel = np.linalg.norm(emp - C) / np.linalg.norm(C)
    REPORT["mc_covariance_rel_frobenius_4000"] = float(rel)
    assert rel < 0.1


def test_ppca_score_and_gradient_vs_dense(setup):
    """RELAX's latent epilogue with the exact ``C^+`` equals the dense marginal density up to a data constant."""
    arm, rt, ctf = setup["arm"], setup["rt"], setup["ctf"]
    Lm, sel, Lmap = _dense(setup)
    C = Lm @ Lm.T / S**2
    Cp, _, _ = support_pinv(C)
    rng = np.random.default_rng(2)
    y = rng.standard_normal(K * S * S) * 0.3
    V = np.asarray(Lmap(jnp.asarray(y)))
    supp = L.sphere(n, 3.5)
    mu = jnp.asarray(rng.standard_normal((n,) * 3) * supp)
    W = jnp.asarray(rng.standard_normal((2, n, n, n)) * supp * 0.5)

    def Fmap(x, Q, u):
        return Lmap(jnp.real(ftu.get_idft2(_exact_images(x, Q, u, rt, ctf).reshape(K, S, S))).reshape(-1))

    def ours(mu, W, Q, u):
        f = jnp.stack([Fmap(mu, Q, u)] + [Fmap(W[j], Q, u) for j in range(2)])
        inner = (f @ jnp.asarray(Cp) @ jnp.asarray(V))[:, None, None]
        P = 3
        first, second = np.triu_indices(P)
        gram = jnp.stack([f[a] @ jnp.asarray(Cp) @ f[b] for a, b in zip(first, second)])[:, None, None]
        s, _, _ = L.scores_from_stats(inner, gram, True)
        return s[0, 0]

    def dense(mu, W, Q, u):
        f0 = np.asarray(Fmap(mu, Q, u))
        fW = np.stack([np.asarray(Fmap(W[j], Q, u)) for j in range(2)], -1)
        return ppca_logpdf(V, f0, fW, C)

    poses = [(jnp.asarray(Rotation.random(random_state=s).as_matrix()), jnp.asarray(rng.normal(0, 0.8, 3))) for s in range(4)]
    a = np.array([float(ours(mu, W, Q, u)) for Q, u in poses])
    b = np.array([dense(mu, W, Q, u) for Q, u in poses])
    REPORT["score_minus_dense_spread"] = float(np.ptp(a - b))
    REPORT["score_dynamic_range"] = float(np.ptp(b))
    assert np.ptp(a - b) < 1e-6 * max(1.0, np.ptp(b))
    # gradients in mu and W at one pose against the dense density (finite differences of the dense oracle)
    Q, u = poses[0]
    g_mu, g_W = jax.grad(ours, argnums=(0, 1))(mu, W, Q, u)
    rng2 = np.random.default_rng(9)
    for name, g, base in (("mu", g_mu, mu), ("W", g_W, W)):
        dirn = jnp.asarray(rng2.standard_normal(base.shape)) * (supp if base.ndim == 3 else supp[None])
        eps = 1e-4
        if name == "mu":
            fd = (dense(mu + eps * dirn, W, Q, u) - dense(mu - eps * dirn, W, Q, u)) / (2 * eps)
        else:
            fd = (dense(mu, W + eps * dirn, Q, u) - dense(mu, W - eps * dirn, Q, u)) / (2 * eps)
        an = float(jnp.sum(g * dirn))
        REPORT[f"grad_{name}_rel"] = abs(an - fd) / abs(fd)
        assert abs(an - fd) < 1e-5 * abs(fd)


def test_unmasked_reproduces_image_likelihood(setup):
    """With no crop (Cc = S), mask M = I and the exact ``C^+``, the masked score equals the image likelihood
    (on the projector's disc) up to a data-only constant."""
    d = setup["d"]
    arm = L.MaskedArm(d, support_radius=3.5, mask_radius=2.5, crop=12, lam_rel=0.1, n_mc=1)
    arm.Cc = S
    full = jnp.ones((S,) * 3)
    setup_full = dict(setup, arm=arm)
    Lm, sel, Lmap = _dense(setup_full, mask=full)
    C = Lm @ Lm.T / S**2
    Cp, _, _ = support_pinv(C, rtol=1e-10)
    rt, ctf = setup["rt"], setup["ctf"]
    rng = np.random.default_rng(4)
    y = rng.standard_normal(K * S * S)
    yft = _real_to_patch_ft(jnp.asarray(y))
    V = np.asarray(Lmap(jnp.asarray(y)))
    disc = (np.sum((L.plane_lattice(S) * S) ** 2, -1) <= (S // 2 - 1) ** 2).astype(float)
    supp = L.sphere(n, 3.5)
    out = []
    for s in range(5):
        # The target as a scene volume (embedded, shifted by whole voxels), seen through the scene operator A:
        # the equivalence holds for forward images in the range of A. (A target rotated by interpolating its own
        # DFT is not exactly in that range on this grid: test_mean_and_conventions measures that path.)
        x = rng.standard_normal((n,) * 3) * supp
        lo = (S - n) // 2
        big = np.zeros((S,) * 3)
        big[lo : lo + n, lo : lo + n, lo : lo + n] = x
        big = np.roll(big, tuple(rng.integers(-2, 3, 3)), axis=(0, 1, 2))
        ft = ftu.get_dft3(jnp.asarray(big)).reshape(-1).astype(jnp.complex128)
        Ax = slicing.slice_volume(ft, rt, (S, S), (S, S, S), "linear_interp") * ctf
        image_ll = -0.5 * float(jnp.sum(disc * jnp.abs(yft - Ax) ** 2))
        F = np.asarray(Lmap(jnp.real(ftu.get_idft2(Ax.reshape(K, S, S))).reshape(-1)))
        masked_ll = -0.5 * float((V - F) @ Cp @ (V - F))
        out.append((image_ll, masked_ll))
    out = np.array(out)
    REPORT["unmasked_minus_image_spread"] = float(np.ptp(out[:, 1] - out[:, 0]))
    REPORT["unmasked_image_dynamic_range"] = float(np.ptp(out[:, 0]))
    assert np.ptp(out[:, 1] - out[:, 0]) < 1e-5 * np.ptp(out[:, 0])


@pytest.mark.parametrize("geometry", [(5, 40.0), (41, 60.0)])
def test_white_covariance_approximation(setup, geometry):
    """Size of the ``C ~ c I`` approximation: score differences across poses, exact ``C^+`` against ``1/c``.
    Two tilt geometries: 5 tilts over +-40 deg (pessimistic) and the pilot's 41 tilts over +-60 deg."""
    nk, mt = geometry
    d, _ = _particles(np.random.default_rng(0), K=nk, max_tilt=mt)
    arm = L.MaskedArm(d, support_radius=3.5, mask_radius=2.5, crop=12, lam_rel=0.1, n_mc=1)
    rt, ctf = jnp.asarray(d.rt[0]), jnp.asarray(d.ctf_patch[0])
    D, lam = arm.weights(ctf, rt)
    local = dict(setup, arm=arm, rt=rt, ctf=ctf, D=D, lam=lam)
    Lm, sel, Lmap = _dense(local, K=nk)
    C = Lm @ Lm.T / S**2
    Cp, _, _ = support_pinv(C)
    c = float(np.mean(np.diag(C)))
    rng = np.random.default_rng(6)
    x0 = jnp.asarray(rng.standard_normal((n,) * 3) * L.sphere(n, 3.5))
    Q0 = jnp.asarray(Rotation.random(random_state=1).as_matrix())

    def F(Q, u):
        return np.asarray(Lmap(jnp.real(ftu.get_idft2(_exact_images(x0, Q, u, rt, ctf).reshape(nk, S, S))).reshape(-1)))

    V = F(Q0, jnp.zeros(3)) + np.asarray(Lmap(jnp.asarray(rng.standard_normal(nk * S * S) / S)))
    ex, ap = [], []
    for s in range(24):
        Q = jnp.asarray((Rotation.from_matrix(np.asarray(Q0)) * Rotation.from_rotvec(rng.normal(0, 0.25, 3))).as_matrix())
        u = jnp.asarray(rng.normal(0, 0.7, 3))
        r = V - F(Q, u)
        ex.append(-0.5 * r @ Cp @ r)
        ap.append(-0.5 * r @ r / c)
    ex, ap = np.array(ex), np.array(ap)
    tag = f"K{nk}"
    REPORT[f"white_cov_score_corr_{tag}"] = float(np.corrcoef(ex, ap)[0, 1])
    REPORT[f"white_cov_argmax_same_{tag}"] = bool(np.argmax(ex) == np.argmax(ap))
    REPORT[f"white_cov_slope_{tag}"] = float(np.polyfit(ex, ap, 1)[0])
    REPORT[f"c_mean_diag_{tag}"] = c
    ratio = float(L.dof_ratio(D, lam, arm.k_ok))
    REPORT[f"dof_ratio_{tag}"] = ratio
    REPORT[f"white_cov_slope_dof_corrected_{tag}"] = REPORT[f"white_cov_slope_{tag}"] / ratio
    assert np.isfinite(REPORT[f"white_cov_score_corr_{tag}"])


def teardown_module(module):
    path = os.environ.get("MASKED_TOMO_REPORT")
    if path:
        with open(path, "w") as f:
            json.dump(REPORT, f, indent=1, default=float)
