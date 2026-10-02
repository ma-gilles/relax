"""Fused CUDA stages of the streamed PPCA engine against the XLA formulation (float64 reference).

The window projector, the pass-1 latent epilogue and the pass-2 posterior preparation each replace
an XLA stage of :mod:`relax.ppca_refinement.full_row_stream`; the reduction of the epilogue's
per-row partials replaces the full-score normalization.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from helpers.float_compare import assert_matches

from relax.helpers.fourier_window import make_fourier_window_spec
from relax.ppca_refinement import full_row_stream as frs

IMAGE_SHAPE, VOLUME_SHAPE = (32, 32), (32, 32, 32)
N_HALF, HALF = 32 * 17, 32 * 32 * 17


def _rotations(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    ).astype(np.float32)


def _band(actual, truth):
    actual, truth = np.asarray(actual, np.float64), np.asarray(truth, np.float64)
    finite = np.isfinite(truth)
    assert np.array_equal(finite, np.isfinite(actual)) and np.array_equal(actual[~finite], truth[~finite])
    scale = max(np.max(np.abs(actual[finite])), np.max(np.abs(truth[finite])))
    return float(np.max(np.abs(actual[finite] - truth[finite])) / scale)


def _assert_no_less_accurate(actual, xla32, truth, what):
    """The CUDA stage is at most twice the float32 XLA stage's own distance from float64 (or within 1e-6)."""
    error, reference = _band(actual, truth), _band(xla32, truth)
    assert error <= max(1e-6, 2 * reference), (what, error, reference)


@pytest.mark.gpu
@pytest.mark.parametrize("P", [1, 5, 11])
def test_window_project_matches_recovar_slice(P, custom_cuda_lib, gpu_device, monkeypatch):
    """Planar windowed projections are recovar's ``batch_slice_volume``; products are ``Re(conj(A_i) A_j)``."""
    import recovar.cuda_backproject as cuda_backproject
    from recovar import core

    from relax.cuda.kernels import ppca_window_project_f32

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    spec = make_fourier_window_spec(IMAGE_SHAPE, 28, N_HALF, square=False, include_recon_window=True)
    rng = np.random.default_rng(7)
    rotations = _rotations(rng, 40)
    indices = np.asarray(spec.score_indices, np.int32)
    F = indices.size
    volumes = (rng.standard_normal((P, HALF)) + 1j * rng.standard_normal((P, HALF))).astype(np.complex64)
    with jax.default_device(gpu_device):
        planar, products = ppca_window_project_f32(
            jnp.asarray(volumes.T.copy()),
            jnp.asarray(indices),
            jnp.asarray(rotations),
            image_shape=IMAGE_SHAPE,
            volume_shape=VOLUME_SHAPE,
            max_r=spec.max_r,
            with_products=True,
        )
        reference = core.batch_slice_volume(
            jnp.asarray(volumes),
            jnp.asarray(rotations),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            half_volume=True,
            half_image=True,
            max_r=spec.max_r,
        )
        reference = np.asarray(reference)[:, :, indices]
        empty = ppca_window_project_f32(
            jnp.asarray(volumes.T.copy()),
            jnp.asarray(indices),
            jnp.asarray(rotations),
            image_shape=IMAGE_SHAPE,
            volume_shape=VOLUME_SHAPE,
            max_r=spec.max_r,
            with_products=False,
        )
        # A window padded with -1 (the GPU GEMM alignment) projects zeros there.
        padded = ppca_window_project_f32(
            jnp.asarray(volumes.T.copy()),
            jnp.asarray(np.concatenate([indices, np.full(3, -1, np.int32)])),
            jnp.asarray(rotations),
            image_shape=IMAGE_SHAPE,
            volume_shape=VOLUME_SHAPE,
            max_r=spec.max_r,
            with_products=True,
        )
    planar, products = np.asarray(planar), np.asarray(products)
    padded_planar, padded_products = (np.asarray(x) for x in padded)
    assert np.all(padded_planar[..., F : F + 3] == 0) and np.all(padded_planar[..., 2 * F + 3 :] == 0)
    assert np.all(padded_products[..., F:] == 0)
    assert_matches(padded_planar[..., :F], planar[..., :F])
    assert_matches(padded_planar[..., F + 3 : 2 * F + 3], planar[..., F:])
    assert planar.shape == (P, 40, 2 * F) and products.shape == (P * (P + 1) // 2, 40, F)
    assert np.asarray(empty[1]).shape == (0,)
    assert_matches(np.asarray(empty[0]), planar)
    assert_matches(planar[..., :F], reference.real)
    assert_matches(planar[..., F:], reference.imag)
    assert np.count_nonzero(reference) > 0.5 * reference.size
    first, second = np.triu_indices(P)
    expected = (np.conj(reference[first]) * reference[second]).real
    assert_matches(products, expected.astype(np.float32))


def _latent_problem(rng, q, R, B, T):
    """Inner products and a positive semidefinite Gram as the score GEMMs produce them."""
    P, F = q + 1, 60
    projections = rng.standard_normal((P, R, F)) + 1j * rng.standard_normal((P, R, F))
    ctf2 = rng.uniform(0.0, 0.05, (F, B))
    first, second = np.triu_indices(P)
    gram = np.einsum("krf,fb->krb", (np.conj(projections[first]) * projections[second]).real, ctf2)
    inner = rng.standard_normal((P, R, B, T)) * 3.0
    return inner, gram


def _prior_tables(rng, R, B, T, n_rows):
    rows = np.sort(rng.choice(n_rows, R - 1, replace=False)).astype(np.int32)
    rows = np.append(rows, n_rows).astype(np.int32)  # sentinel row pads the block
    n_coarse_r, n_coarse_t = 3, 2
    rotation_parent = np.append(rng.integers(0, n_coarse_r, n_rows), n_coarse_r).astype(np.int32)
    rotation_log_prior = np.append(rng.normal(size=n_rows) - 5.0, 0.0).astype(np.float32)
    translation_parent = rng.integers(0, n_coarse_t, T).astype(np.int32)
    translation_log_prior = (rng.normal(size=T) - 2.0).astype(np.float32)
    coarse = rng.random((B, n_coarse_r, n_coarse_t)) < 0.8
    coarse[:, 0, :] = True  # every image keeps support
    coarse_mask = np.concatenate([coarse, np.zeros((B, 1, n_coarse_t), bool)], axis=1)
    return rows, (rotation_parent, rotation_log_prior, translation_parent, translation_log_prior, coarse_mask)


@pytest.mark.gpu
@pytest.mark.parametrize("q,T", [(0, 29), (4, 29), (4, 40), (10, 29)])
def test_latent_epilogue_and_posterior_prep_match_xla(q, T, gpu_device):
    """Pass-1 epilogue (kept rows and partials) and pass-2 preparation against the XLA stages in float64."""
    from relax.cuda.kernels import ppca_latent_epilogue_f32, ppca_posterior_prep_f32

    rng = np.random.default_rng(11 + q + T)
    R, B, cap, start = 16, 6, 48, 16
    inner, gram = _latent_problem(rng, q, R, B, T)
    rows, tables = _prior_tables(rng, R, B, T, n_rows=40)
    shift2 = rng.uniform(0.0, 20.0, T).astype(np.float32)
    P = q + 1

    def xla(dtype):
        """The XLA formulation of both stages (``_latent_block``, ``_posterior_block``) in ``dtype``."""
        with jax.default_device(jax.devices("cpu")[0]), jax.default_matmul_precision("highest"):
            rotation_parent, rotation_log_prior, translation_parent, translation_log_prior, coarse_mask = tables
            prior = frs.full_row_pose_log_prior(
                jnp.asarray(coarse_mask),
                jnp.asarray(rotation_parent)[rows],
                jnp.asarray(translation_parent),
                jnp.asarray(rotation_log_prior, dtype)[rows],
                jnp.asarray(translation_log_prior, dtype),
            ).astype(dtype)
            # _latent_block's own inner/Gram products, given directly.
            first, second = np.triu_indices(P)
            index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(first, second))}
            g = jnp.asarray(gram, dtype)
            x = jnp.asarray(inner, dtype)
            rho = g[index[(0, 0)], :, :, None] - 2.0 * x[0]
            prior_rbt = jnp.transpose(prior, (1, 0, 2))
            if q == 0:
                score = -0.5 * rho + prior_rbt
                mean = jnp.zeros((0, R, B, T), dtype)
                covariance = jnp.zeros((R, B, 0), dtype)
            else:
                H = [[g[index[(min(i, j), max(i, j))]] for j in range(1, P)] for i in range(1, P)]
                L, logdet = frs._unit_shift_cholesky(H)
                Lt = [[v if v is None else v[..., None] for v in row] for row in L]
                b = [x[j] - g[index[(0, j)], :, :, None] for j in range(1, P)]
                v = frs._forward_substitute(Lt, b)
                score = -0.5 * (rho - sum(e * e for e in v) + logdet[..., None]) + prior_rbt
                mean = jnp.stack(frs._back_substitute(Lt, v), axis=0)
                inverse = frs._lower_inverse(L)
                covariance = jnp.stack(
                    [sum(inverse[k][i] * inverse[k][j] for k in range(j, q)) for i, j in zip(*np.triu_indices(q))],
                    axis=-1,
                )
            center = jnp.max(score, axis=(0, 2))
            logZ = jnp.log(jnp.sum(jnp.exp(score - center[None, :, None]), axis=(0, 2)))
            kept = frs._empty_kept(cap, B, T, q, dtype)
            kept = kept._replace(
                score=kept.score.at[start : start + R].set(score),
                latent_mean=kept.latent_mean.at[:, start : start + R].set(mean),
                latent_covariance=kept.latent_covariance.at[start : start + R].set(covariance),
            )
            posterior = frs._Posterior(center, logZ, center, None, None)
            carry = frs._MomentCarry(
                None,
                None,
                None,
                jnp.zeros((B, q), dtype),
                jnp.zeros((cap,), dtype),
                dtype(0),
                dtype(0),
                dtype(0),
                jnp.zeros((B,), jnp.int32),
            )
            arrays = frs._StreamArrays(**dict.fromkeys(frs._StreamArrays._fields))._replace(
                shift_squared=jnp.asarray(shift2, dtype)
            )
            weights, sums, carry = frs._posterior_block(carry, arrays, kept, posterior, start, R)
            return {
                "score": score,
                "mean": mean,
                "covariance": covariance,
                "center": center,
                "logZ": logZ,
                "weights": weights,
                "sums": sums,
                "embedding": carry.embedding,
                "mass": carry.rotation_mass,
                "trace": carry.latent_covariance_trace_sum,
                "entropy": carry.pose_entropy_sum,
                "offset": carry.offset_second_sum,
                "significant": carry.n_significant,
            }

    with jax.enable_x64(True):
        truth = {k: np.asarray(v) for k, v in xla(jnp.float64).items()}
    xla32 = {k: np.asarray(v) for k, v in xla(jnp.float32).items()}

    with jax.default_device(gpu_device):
        kept = frs._empty_kept(cap, B, T, q, jnp.float32)
        kept = frs._Kept(
            *ppca_latent_epilogue_f32(
                jnp.asarray(inner, jnp.float32),
                jnp.asarray(gram, jnp.float32),
                jnp.asarray(rows),
                tuple(jnp.asarray(t) for t in tables),
                jnp.int32(start),
                kept,
            )
        )
        epilogue = jax.device_get(kept)
        # Pass 2 from the epilogue's own normalization, as the engine runs it.
        posterior = frs._normalize_partials(kept, jnp.arange(cap, dtype=jnp.int32), n_blocks=2, block_size=R)
        center, logZ = jax.device_get((posterior.center, posterior.centered_logZ))
        weights, sums, partial, count = jax.device_get(
            ppca_posterior_prep_f32(
                kept.score,
                kept.latent_mean,
                kept.latent_covariance,
                posterior.center,
                posterior.centered_logZ,
                jnp.asarray(shift2),
                jnp.int32(start),
                block_size=R,
            )
        )
    rows_of = slice(start, start + R)
    # Rows outside the block keep their initial values.
    assert np.all(np.isneginf(epilogue.score[:start])) and np.all(np.isneginf(epilogue.score[start + R :]))
    assert np.all(epilogue.latent_mean[:, :start] == 0) and np.all(epilogue.part_sum[start + R :] == 0)
    _assert_no_less_accurate(epilogue.score[rows_of], xla32["score"], truth["score"], "score")
    if q:
        _assert_no_less_accurate(epilogue.latent_mean[:, rows_of], xla32["mean"], truth["mean"], "mean")
        _assert_no_less_accurate(
            epilogue.latent_covariance[rows_of], xla32["covariance"], truth["covariance"], "covariance"
        )
    # Partials: per-(row, image) maximum, first maximizing translation, sum of exp(score - maximum).
    score64 = truth["score"]
    assert np.array_equal(np.argmax(epilogue.score[rows_of], axis=-1), epilogue.part_arg[rows_of])
    assert_matches(epilogue.part_max[rows_of], np.max(epilogue.score[rows_of], axis=-1))

    def exp_sums(score):
        top = np.max(score, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore"):
            return np.where(np.isneginf(top[..., 0]), 0.0, np.sum(np.exp(score - top), axis=-1))

    _assert_no_less_accurate(epilogue.part_sum[rows_of], exp_sums(xla32["score"]), exp_sums(score64), "part_sum")
    # Normalization from the partials, then the pass-2 preparation.
    _assert_no_less_accurate(center, xla32["center"], truth["center"], "center")
    _assert_no_less_accurate(logZ, xla32["logZ"], truth["logZ"], "logZ")
    _assert_no_less_accurate(weights, xla32["weights"], truth["weights"], "weights")
    _assert_no_less_accurate(sums, xla32["sums"], truth["sums"], "sums")
    if q:
        _assert_no_less_accurate(np.sum(sums[1:P], axis=1).T, xla32["embedding"], truth["embedding"], "embedding")
    _assert_no_less_accurate(np.sum(partial[0]), xla32["entropy"], truth["entropy"], "entropy")
    _assert_no_less_accurate(np.sum(partial[1]), xla32["offset"], truth["offset"], "offset")
    assert np.array_equal(np.sum(count, axis=0), xla32["significant"])


def test_normalize_partials_matches_full_score_normalization():
    """Center, partition and the first-maximum pose from partials equal the full-score reduction, ties included."""
    rng = np.random.default_rng(3)
    n_blocks, block_size, B, T = 3, 4, 5, 6
    cap = (n_blocks + 1) * block_size
    score = np.round(rng.normal(size=(cap, B, T)), 1).astype(np.float32)  # rounding makes ties
    score[n_blocks * block_size - 1] = -np.inf  # a sentinel row
    score[:, 2, :] = np.float32(1.5)  # image 2: every pose ties
    score[n_blocks * block_size :] = np.float32(9.0)  # rows past the scored blocks are ignored
    rows = np.arange(cap, dtype=np.int32) * 7
    with np.errstate(invalid="ignore"):
        part_max = np.max(score, axis=-1)
        part_arg = np.argmax(score, axis=-1).astype(np.int32)
        part_sum = np.where(np.isneginf(part_max), 0.0, np.sum(np.exp(score - part_max[..., None]), axis=-1))
    kept = frs._empty_kept(cap, B, T, 0, jnp.float32)._replace(
        score=jnp.asarray(score),
        part_max=jnp.asarray(part_max),
        part_arg=jnp.asarray(part_arg),
        part_sum=jnp.asarray(part_sum.astype(np.float32)),
    )
    expected = frs._normalize(jnp.asarray(score), jnp.asarray(rows), n_blocks=n_blocks, block_size=block_size)
    actual = frs._normalize_partials(kept, jnp.asarray(rows), n_blocks=n_blocks, block_size=block_size)
    for name in ("center", "top_score", "top_rotation", "top_translation"):
        assert np.array_equal(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))), name
    assert_matches(np.asarray(actual.centered_logZ), np.asarray(expected.centered_logZ))
