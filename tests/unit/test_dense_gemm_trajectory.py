"""Iteration-boundary checks for the fixed real-fixture trajectory adapter."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.benchmark_dense_gemm_trajectory import (
    _checked_particle_rows,
    _failure_diagnostics,
    _resident_cuda_fixture_program,
)


def _result(*, masses=(0.73, 1.0), numerator=(1 + 2j, 3 - 1j)):
    pair = np.full((5, 2), -np.inf, dtype=np.float32)
    pair[3] = (-1280.0, np.log(4.0))
    pair[1] = (-1260.0, np.log(3.0))
    mass = np.zeros(5, dtype=np.float32)
    mass[[3, 1]] = masses
    return SimpleNamespace(
        next_pair_table=jnp.asarray(pair),
        mass_table=jnp.asarray(mass),
        invalid_normalizer_table=jnp.zeros(5, dtype=jnp.bool_),
        invalid_weight_table=jnp.zeros(5, dtype=jnp.bool_),
        numerator=jnp.asarray(numerator, dtype=jnp.complex64),
        denominator=jnp.asarray((2.0, 1.0), dtype=jnp.float32),
    )


def test_iteration_boundary_preserves_lagged_mass_by_stable_id():
    report = _checked_particle_rows(_result(), (3, 1))
    assert report["particles"] == 2
    np.testing.assert_allclose(report["mass_min_max"], (0.73, 1.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(report["mass_mean"], 0.865, rtol=1e-6, atol=1e-6)
    assert report["logz_min_max"][0] < report["logz_min_max"][1]


@pytest.mark.parametrize("masses", [(0.0, 1.0), (np.nan, 1.0), (np.inf, 1.0)])
def test_iteration_boundary_rejects_invalid_weight_mass(masses):
    with pytest.raises(RuntimeError, match="invalid normalizer or posterior mass"):
        _checked_particle_rows(_result(masses=masses), (3, 1))


def test_iteration_boundary_rejects_nonfinite_volume():
    with pytest.raises(RuntimeError, match="nonfinite BPref accumulators"):
        _checked_particle_rows(_result(numerator=(complex(np.nan, 0), 1 + 0j)), (3, 1))


def test_failure_census_keeps_stable_ids_and_finite_ranges():
    result = _result(masses=(np.inf, 0.0), numerator=(complex(np.nan, 0), 1 + 0j))
    report = _failure_diagnostics(result, (3, 1))
    assert report["nonfinite_mass_ids"] == [3]
    assert report["nonpositive_mass_ids"] == [1]
    assert report["finite_mass_min_max"] == [0.0, 0.0]
    assert report["nonfinite_numerator_voxels"] == 1


def test_resident_cuda_control_streams_all_dense_rows_in_bounded_tiles(monkeypatch):
    """The selected GEMM scorer and resident row owner retain every pose."""
    from relax.cuda import kernels as cuda_kernels
    from relax.dense.gemm_experiment import pad_batch, pad_grid
    from relax.dense.gemm_experiment_kernels import empty_normalizer_table
    from relax.fine_pass import adjoint as sparse_pass2_adjoint
    from relax.fine_pass import resident_pass2
    from relax.scoring import coarse_kernels

    selected = {"score_traces": 0, "row_traces": 0, "adjoint_traces": 0}

    def translate(images, angles, indices, image_shape):
        assert image_shape == (2, 2)
        assert indices.shape == (1,)
        return jnp.broadcast_to(images[:, None, :], (3, angles.shape[0], 1)).reshape(-1, 1)

    def score(projected, _abs2, shifted, weight, initial, active, **kwargs):
        selected["score_traces"] += 1
        assert kwargs["float64"] is False
        assert shifted.shape == (3, 2, 1)
        assert weight.shape == (3, 1) and initial.shape == (3,)
        assert active.dtype == jnp.int32
        return jnp.zeros((3, projected.shape[0], 2), dtype=jnp.float32)

    def weighted_rows(posterior, row_ids, row_local, recon, recon_weight,
                      noise, ctf2, indices, angles, **kwargs):
        selected["row_traces"] += 1
        assert kwargs["cuda_backproject"] is cuda_kernels
        assert hasattr(kwargs["cuda_backproject"], "relion_translate_sum_flat_rows_f32")
        assert kwargs["kernel_ctf_probs"] is False
        assert posterior.shape == (4, 2)
        assert indices.shape == (1,) and angles.shape == (2, 2)
        assert recon_weight.shape == recon.shape and noise.shape == recon.shape
        safe = jnp.maximum(row_local, 0)
        row_mass = jnp.sum(posterior, axis=1)
        valid = row_ids >= 0
        summed = jnp.where(valid[:, None], recon[safe] * row_mass[:, None], 0)
        ctf_probs = jnp.where(valid[:, None], ctf2[safe] * row_mass[:, None], 0)
        return summed, summed, ctf_probs, row_mass

    def adjoint(rows, rotations, volume, **kwargs):
        selected["adjoint_traces"] += 1
        assert rows.shape == (4, 1) and rotations.shape == (4, 3, 3)
        assert kwargs["relion_x_half"] is True and kwargs["use_windowed_adjoint"] is True
        assert kwargs["max_block_bytes"] > 0
        return volume.at[0].add(jnp.sum(rows))

    monkeypatch.setattr(cuda_kernels, "relion_translate_score_f32", translate)
    monkeypatch.setattr(coarse_kernels, "relion_coarse_gaussian_gemm_scores_jit", score)
    monkeypatch.setattr(resident_pass2, "_resident_block_weighted_sums_kernel", weighted_rows)
    monkeypatch.setattr(sparse_pass2_adjoint, "_accumulate_adjoint_block_chunked", adjoint)

    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (3, 3, 3))
    grid = pad_grid(
        rotations, rotations, jnp.ones((2, 1), dtype=jnp.complex64),
        jnp.ones((2, 1), dtype=jnp.complex64),
        rotation_tile=2, translation_tile=1,
    )
    raw = jnp.asarray([[1], [2], [3]], dtype=jnp.complex64)
    batch = pad_batch(
        jnp.zeros((3, 1), dtype=jnp.complex64),
        jnp.ones((3, 1), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        raw, jnp.ones((3, 1), dtype=jnp.float32),
        jnp.zeros((3, 3), dtype=jnp.float32),
        jnp.zeros((3, 2), dtype=jnp.float32),
        np.arange(3, dtype=np.int32),
        image_capacity=3, grid=grid, sentinel_id=3,
        rec_raw_image=raw, rec_weighted_ctf=jnp.ones((3, 1), dtype=jnp.float32),
    )
    geometry = {
        "score_indices": jnp.asarray([0], dtype=jnp.int32),
        "angles": jnp.zeros((2, 2), dtype=jnp.float32),
        "window": SimpleNamespace(
            recon_window_indices=jnp.asarray([0], dtype=jnp.int32),
            relion_x_half_recon_indices=jnp.asarray([0], dtype=jnp.int32),
        ),
    }
    args = SimpleNamespace(images=3, rotation_tile=2, translations=2, row_tile=4)
    program = _resident_cuda_fixture_program(
        args, geometry, (3, 3, 2), 2, (2, 2),
        lambda _reference, rots: jnp.zeros((rots.shape[0], 1), dtype=jnp.complex64),
        mode="exact",
    )
    table = empty_normalizer_table(4)
    result = program(
        jnp.zeros((1, 1), dtype=jnp.complex64),
        jnp.zeros((1,), dtype=jnp.complex64),
        jnp.zeros((1,), dtype=jnp.float32),
        table, table, batch, grid,
    )
    jax.block_until_ready(result.numerator)
    np.testing.assert_allclose(np.asarray(result.numerator), [6.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.denominator), [3.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.batch_mass), np.ones(3), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result.batch_logz), np.full(3, np.log(6)), rtol=1e-6, atol=1e-6,
    )
    assert selected["score_traces"] == 2  # Exact normalizer and reconstruction sweeps.
    assert selected["row_traces"] == 1
    assert selected["adjoint_traces"] == 2

    lagged = _resident_cuda_fixture_program(
        args, geometry, (3, 3, 2), 2, (2, 2),
        lambda _reference, rots: jnp.zeros((rots.shape[0], 1), dtype=jnp.complex64),
        mode="lagged",
    )
    previous = result.next_pair_table.at[:3, 0].add(-jnp.log(jnp.float32(2)))
    lagged_result = lagged(
        jnp.zeros((1, 1), dtype=jnp.complex64),
        jnp.zeros((1,), dtype=jnp.complex64),
        jnp.zeros((1,), dtype=jnp.float32),
        previous, empty_normalizer_table(4), batch, grid,
    )
    jax.block_until_ready(lagged_result.numerator)
    np.testing.assert_allclose(
        np.asarray(lagged_result.batch_mass), np.full(3, 2.0), rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(lagged_result.numerator), [12.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(lagged_result.denominator), [6.0], rtol=1e-6, atol=1e-6)
    assert selected["score_traces"] == 3  # One additional lagged sweep.
