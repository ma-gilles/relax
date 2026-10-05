"""The streamed LHS metric GEMM keeps each voxel's (1+q)x(1+q) metric positive semidefinite under TF32.

TF32 rounds every packed moment entry on its own, so nearly rank-one moment matrices (latent posterior narrow
against a large mean latent) summed into a voxel come out indefinite by about 2^-12 of the largest eigenvalue,
above the coupled-direction check's bound of 32 float32 eps. VDAM stopped there past update 2000 on a realistic
fixture (em_work/ppca_sgd_vs_vdam_20261003/improve_20261004/tables/vdam_crash_dump_eigs.jsonl: smallest/largest
-1.2e-5 to -3.7e-5). ``full_row_stream._metric_dot`` passes the moments through the TF32 GEMM as a TF32 value plus
the exact remainder; rounding the non-negative CTF operand scales a pixel's moment channels together and keeps PSD.
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.ppca_initial_model.update import coupled_direction, metric_floor
from relax.ppca_refinement import full_row_stream
from relax.ppca_refinement.full_row_stream import _metric_dot, _second_moment_sums, tf32_capable

pytestmark = pytest.mark.unit

Q, ROTATIONS, IMAGES, TRANSLATIONS, PIXELS = 2, 4, 64, 3, 40
TF32 = types.SimpleNamespace(gemm_precision="tf32")
_dot = jnp.dot  # the emulation below replaces jnp.dot inside _metric_dot


def _round_tf32(x):
    """float32 to TF32 (10 mantissa bits), round to nearest even, as the tensor cores' operand conversion."""
    bits = np.asarray(x, np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x0FFF) + ((bits >> np.uint32(13)) & np.uint32(1))) & np.uint32(0xFFFFE000)
    return bits.view(np.float32)


def _emulated_tf32_dot(a, b, precision=None):
    """One TF32 pass with float32 accumulation, on any device."""
    return _dot(jnp.asarray(_round_tf32(a)), jnp.asarray(_round_tf32(b)), precision=jax.lax.Precision.HIGHEST)


def _operands():
    """Moment rows (tri(1+q) * R, B) and the CTF operand (B, F), as ``_moment_block`` passes them to the LHS GEMM.

    Latent means near (2.7, -1.9) with a posterior variance of 1e-6: each image's moment matrix is nearly rank one
    with entries spanning 1 to 7 (the large-mean-latent regime of the failing runs).
    """
    rng = np.random.default_rng(20261005)
    gamma = rng.dirichlet(np.ones(TRANSLATIONS), size=(ROTATIONS, IMAGES)).astype(np.float32)
    mean = np.asarray([2.7, -1.9])[:, None, None, None] + 1e-3 * rng.standard_normal(
        (Q, ROTATIONS, IMAGES, TRANSLATIONS)
    )
    covariance = np.zeros((ROTATIONS, IMAGES, Q * (Q + 1) // 2))
    covariance[..., [0, 2]] = 1e-6
    sums = _second_moment_sums(jnp.asarray(gamma), jnp.asarray(mean, jnp.float32), jnp.asarray(covariance, jnp.float32))
    ctf2 = rng.uniform(0.0, 1.0, (IMAGES, PIXELS)).astype(np.float32)
    return np.asarray(sums).reshape(-1, IMAGES), ctf2


def _metric_rows(lhs_images):
    """(tri(P) * R, F) GEMM output -> packed metric rows (R * F, tri(P)), every row on the support."""
    return np.moveaxis(np.asarray(lhs_images).reshape(-1, ROTATIONS * PIXELS), 0, -1)


def _check(rows):
    gradient = np.ones((rows.shape[0], Q + 1), np.complex64)
    return coupled_direction(rows, gradient, np.ones(rows.shape[0], bool), floor=metric_floor(128))


def test_tf32_metric_gemm_keeps_the_metric_semidefinite(monkeypatch):
    sums, ctf2 = _operands()
    reference = _metric_rows(sums.astype(np.float64) @ ctf2.astype(np.float64))
    _check(reference.astype(np.float32))

    plain = _metric_rows(_emulated_tf32_dot(sums, ctf2))
    with pytest.raises(ValueError, match="materially indefinite PPCA metric"):
        _check(plain)

    monkeypatch.setattr(full_row_stream.jnp, "dot", _emulated_tf32_dot)
    split = _metric_rows(_metric_dot(jnp.asarray(sums), jnp.asarray(ctf2), TF32))
    monkeypatch.undo()
    _check(split)
    # Only the non-negative CTF operand is rounded: 2^-11 relative, the TF32 unit (plain TF32 is the same order).
    assert_matches(split, reference, rtol=2**-11)


def test_metric_gemm_on_tf32_tensor_cores():
    device = jax.local_devices()[0]
    sums, ctf2 = _operands()
    if not tf32_capable(device):
        assert_matches(
            np.asarray(_metric_dot(jnp.asarray(sums), jnp.asarray(ctf2), types.SimpleNamespace(gemm_precision="fp32"))),
            sums.astype(np.float64) @ ctf2.astype(np.float64),
        )
        return
    plain = jnp.dot(jnp.asarray(sums), jnp.asarray(ctf2), precision=jax.lax.DotAlgorithmPreset.TF32_TF32_F32)
    with pytest.raises(ValueError, match="materially indefinite PPCA metric"):
        _check(_metric_rows(plain))
    _check(_metric_rows(_metric_dot(jnp.asarray(sums), jnp.asarray(ctf2), TF32)))
