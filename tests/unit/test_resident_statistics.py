"""Tests for the device-resident pass-2 statistics helpers (T9a, CPU only).

Ticket: em_parity_tickets_20260918/T9a_device_statistics_stage.md.
Design: em_device_resident_pass2_design_20260918.md (stage 7).

The row-to-image reductions and the flat-row norm and scale partials the resident
chunk programs use, the image-axis capacity classes, and the guards of
:func:`finalize_statistics`. Run with ``JAX_PLATFORMS=cpu RECOVAR_DISABLE_CUDA=1``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from oracles.projection import compute_norm_residual_per_image

from relax.fine_pass.resident_statistics import (
    finalize_statistics,
    make_resident_statistics,
    resident_image_capacity,
    resolve_statistics_config,
    segment_sum_by_image,
)

pytestmark = pytest.mark.unit


# --- Fixture shapes ---------------------------------------------------------
N_IMAGES = 7
N_SHELLS = 6
N_FINE_TRANS = 4
N_PIX = 11
N_COARSE_ROT = 9
N_GROUPS = 3
CURRENT_SIZE = 8  # -> norm cutoff 4, direct-noise exclusive shell stop 5


def _rng():
    return np.random.default_rng(20260918)


def _config():
    return resolve_statistics_config(
        n_shells=N_SHELLS,
        n_fine_trans=N_FINE_TRANS,
        n_images=N_IMAGES,
        n_coarse_rot=N_COARSE_ROT,
        n_scale_groups=N_GROUPS,
        current_size=CURRENT_SIZE,
        include_unweighted_high_shell=True,
        relion_wavg_atomic_scale_aa=True,
        accumulate_scale=True,
        source_faithful_spectrum_norm=True,
    )


# --- Helper twins -----------------------------------------------------------


def test_segment_sum_by_image_matches_numpy_add_at():
    """The row -> image reduction is ``np.add.at`` with the row axis first."""

    rng = _rng()
    row_image = rng.integers(0, 4, size=17).astype(np.int32)
    values = rng.standard_normal((17, 3)).astype(np.float64)
    expected = np.zeros((4, 3), dtype=np.float64)
    np.add.at(expected, row_image, values)
    got = np.asarray(segment_sum_by_image(jnp.asarray(values), jnp.asarray(row_image), 4))
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_bucketed_float32_posterior_keeps_image_mass_and_index_semantics(monkeypatch):
    """Only the opted-in F32 [rows,T] sum uses bounded row buckets."""

    from relax.fine_pass.resident_statistics import posterior_translation_bucket_scratch_bytes

    capacity, translations, rows = 4, 3, 4097
    rng = np.random.default_rng(20260928)
    ids = rng.choice([0, 1, 2, 3], size=rows, p=[0.02, 0.03, 0.90, 0.05]).astype(np.int32)
    ids[-4:] = [-1, -capacity, -capacity - 1, capacity]
    values = rng.uniform(0, 0.01, size=(rows, translations)).astype(np.float32)
    values[-2:] = 1.0  # Out-of-range rows must be discarded.
    canonical_ids = ids.copy()
    canonical_ids[(canonical_ids < 0) & (canonical_ids >= -capacity)] += capacity
    expected = np.zeros((capacity, translations), np.float64)
    live = (canonical_ids >= 0) & (canonical_ids < capacity)
    np.add.at(expected, canonical_ids[live], values[live].astype(np.float64))
    got = np.asarray(segment_sum_by_image(
        jnp.asarray(values), jnp.asarray(ids), capacity, float32_posterior_bucket_size=128,
    ))
    np.testing.assert_allclose(got, expected, rtol=5e-7, atol=2e-7)
    assert got.dtype == np.float32
    # Norm A2/XA are scalar F32 rows with the same image IDs. Their optional
    # bucket path must retain the translated-posterior padding semantics.
    got_scalar = np.asarray(segment_sum_by_image(
        jnp.asarray(values[:, 0]), jnp.asarray(ids), capacity,
        float32_scalar_bucket_size=128,
    ))
    np.testing.assert_allclose(got_scalar, expected[:, 0], rtol=5e-7, atol=2e-7)
    assert got_scalar.dtype == np.float32
    assert posterior_translation_bucket_scratch_bytes(rows, capacity, translations) == (
        2 * ((rows + 127) // 128) * capacity * translations * 4 + capacity * translations * 4
    )
    with pytest.raises(ValueError, match="float32"):
        segment_sum_by_image(jnp.asarray(values, jnp.float64), jnp.asarray(ids), capacity,
                             float32_posterior_bucket_size=128)


def test_flat_row_norm_residual_matches_host_per_image_helper():
    """The flat-row A2/XA partials reproduce ``compute_norm_residual_per_image``."""

    from relax.fine_pass.resident_block_programs import _flat_row_norm_and_scale_terms

    rng = _rng()
    batch, n_rot = 3, 4
    proj = (rng.standard_normal((batch, n_rot, N_PIX)) + 1j * rng.standard_normal((batch, n_rot, N_PIX))).astype(np.complex64)
    proj_abs2 = (np.abs(proj) ** 2).astype(np.float32)
    summed = (rng.standard_normal((batch, n_rot, N_PIX)) + 1j * rng.standard_normal((batch, n_rot, N_PIX))).astype(np.complex64)
    ctf = rng.random((batch, n_rot, N_PIX)).astype(np.float32)
    nv = rng.random(N_PIX).astype(np.float32) + 0.5

    expected = np.asarray(
        compute_norm_residual_per_image(
            jnp.asarray(proj), jnp.asarray(proj_abs2), jnp.asarray(summed), jnp.asarray(ctf), jnp.asarray(nv)
        ),
        dtype=np.float64,
    )
    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)
    a2_row, xa_row = _flat_row_norm_and_scale_terms(
        jnp.asarray(proj.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(proj_abs2.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(summed.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(ctf.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(nv),
    )
    a2 = np.asarray(segment_sum_by_image(a2_row, jnp.asarray(row_image), batch), dtype=np.float64)
    xa = np.asarray(segment_sum_by_image(xa_row, jnp.asarray(row_image), batch), dtype=np.float64)
    np.testing.assert_allclose(a2 - 2.0 * xa, expected, rtol=1e-6, atol=0.0)


def test_flat_row_scale_terms_match_host_scale_helper():
    """The masked flat-row partials reproduce the algebraic scale statistics."""

    from relax.fine_pass.resident_block_programs import _flat_row_norm_and_scale_terms

    rng = _rng()
    batch, n_rot = 3, 4
    proj = (rng.standard_normal((batch, n_rot, N_PIX)) + 1j * rng.standard_normal((batch, n_rot, N_PIX))).astype(np.complex64)
    proj_abs2 = (np.abs(proj) ** 2).astype(np.float32)
    summed = (rng.standard_normal((batch, n_rot, N_PIX)) + 1j * rng.standard_normal((batch, n_rot, N_PIX))).astype(np.complex64)
    ctf = rng.random((batch, n_rot, N_PIX)).astype(np.float32)
    nv = rng.random(N_PIX).astype(np.float32) + 0.5
    scale = rng.random(batch).astype(np.float32) + 0.5
    mask = rng.random(N_PIX) > 0.3

    # RELION's group-scale XA/AA statistics per image with the old scale divided back out,
    # summed in float64 over the masked pixels that carry mass.
    safe = np.maximum(scale.astype(np.float64), 1e-30)
    nv64 = nv.astype(np.float64)[None, None, :]
    ctf_mass = (ctf != 0.0) & mask[None, None, :]
    cross_mass = (summed != 0.0) & mask[None, None, :]
    expected_aa = np.where(ctf_mass, proj_abs2 * ctf * nv64, 0.0).sum(axis=(1, 2)) / safe**2
    expected_xa = np.where(cross_mass, nv64 * (proj * np.conj(summed)).real, 0.0).sum(axis=(1, 2)) / safe
    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)
    a2_row, xa_row = _flat_row_norm_and_scale_terms(
        jnp.asarray(proj.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(proj_abs2.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(summed.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(ctf.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(nv),
        pixel_mask=jnp.asarray(mask),
    )
    a2 = segment_sum_by_image(a2_row, jnp.asarray(row_image), batch)
    xa = segment_sum_by_image(xa_row, jnp.asarray(row_image), batch)
    np.testing.assert_allclose(
        np.asarray(xa, dtype=np.float64) / safe, np.asarray(expected_xa, dtype=np.float64),
        rtol=1e-6, atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(a2, dtype=np.float64) / safe**2, np.asarray(expected_aa, dtype=np.float64),
        rtol=1e-6, atol=0.0,
    )


@pytest.mark.parametrize(
    "n_images,capacity",
    [
        (1, 256),
        (256, 256),
        (257, 512),
        (650, 1024),
        (1024, 1024),
        (1025, 2048),
        (2049, 4096),
        (4097, 6144),
        (13000, 14336),
        (20000, 20480),
    ],
)
def test_resident_image_capacity_classes(n_images, capacity):
    assert resident_image_capacity(n_images) == capacity
    assert resident_image_capacity(n_images) >= n_images
    assert resident_image_capacity(n_images) - n_images <= max(255, min(n_images - 1, 2047), n_images // 4)


def test_finalize_rejects_a_winner_in_row_padding():
    """The host's ``best rotation points into padding`` guard is preserved.

    A chunk program counts the images whose winner row lies outside their own rows in
    ``invalid_best_rows``; one such image makes finalization refuse.
    """

    config = _config()
    stats = make_resident_statistics(config, max_posterior_dtype=jnp.float32)
    stats = stats._replace(invalid_best_rows=stats.invalid_best_rows + 1)
    with pytest.raises(RuntimeError, match="outside their own row range"):
        finalize_statistics(stats, config=config, n_images=N_IMAGES)


def test_finalize_reports_images_no_chunk_covered():
    """A chunk plan that misses an image must not silently return zeros."""

    config = _config()
    stats = make_resident_statistics(config, max_posterior_dtype=jnp.float32)
    written = jnp.arange(config.image_capacity) < 3  # a chunk wrote images 0-2 only
    stats = stats._replace(
        best_cell=jnp.where(written, 0, stats.best_cell),
        best_local_rot=jnp.where(written, 0, stats.best_local_rot),
    )
    with pytest.raises(RuntimeError, match="never written by a chunk"):
        finalize_statistics(stats, config=config, n_images=N_IMAGES)
