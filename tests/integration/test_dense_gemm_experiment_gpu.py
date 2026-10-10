"""Actual CUDA projector, direct scorer and adjoint checks for dense GEMM."""

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.dense.gemm_experiment import DenseGemmTileConfig, make_batch_program, native_relion_callbacks
from relax.dense.gemm_experiment_kernels import empty_normalizer_table
from relax.projection.projection import relion_projector_half_to_texture_full
from scripts.benchmark_dense_gemm_em import _direct_comparison, _make_case, _production_program, _run_once


@pytest.mark.gpu
@pytest.mark.parametrize("translation_side", ["image", "projection"])
@pytest.mark.parametrize("translation_tile", [1, 3, 5])
def test_native_dense_operator_matches_direct_cuda(translation_side, translation_tile):
    assert jax.default_backend() == "gpu"
    assert os.environ.get("RECOVAR_CUDA_LIB") and os.environ.get("RELAX_CUDA_LIB")
    args = SimpleNamespace(
        seed=20260928,
        images=3,
        rotations=3,
        translations=5,
        rotation_tile=2,
        translation_tile=translation_tile,
        translation_side=translation_side,
        box=8,
        mode="exact",
    )
    case = _make_case(args)
    config = DenseGemmTileConfig(3, 2, translation_tile, translation_side, "exact")
    program = make_batch_program(config, case[3], case[4])
    old = empty_normalizer_table(4)
    result, _ = _run_once(program, case, old)
    comparison = _direct_comparison(args, case, result, old)
    assert comparison["direct_cuda_score_max_abs"] < 2e-4
    assert comparison["direct_cuda_logz_max_abs"] < 2e-4
    assert comparison["direct_cuda_numerator_volume_max_abs"] < 2e-4
    assert comparison["direct_cuda_weight_volume_max_abs"] < 2e-4
    assert not np.asarray(result.invalid_normalizer).any()
    assert not np.asarray(result.invalid_weight).any()
    assert np.allclose(np.asarray(result.batch_mass[:3]), 1, rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
def test_lagged_bootstrap_uses_native_program_and_exposes_mass():
    assert jax.default_backend() == "gpu"
    args = SimpleNamespace(
        seed=20260928,
        images=3,
        rotations=3,
        translations=5,
        rotation_tile=2,
        translation_tile=3,
        translation_side="projection",
        box=8,
        mode="exact",
    )
    case = _make_case(args)
    exact_program = make_batch_program(DenseGemmTileConfig(3, 2, 3, "projection", "exact"), case[3], case[4])
    initial = empty_normalizer_table(4)
    bootstrap, _ = _run_once(exact_program, case, initial)
    lagged_program = make_batch_program(DenseGemmTileConfig(3, 2, 3, "projection", "lagged"), case[3], case[4])
    old = bootstrap.next_pair_table.at[:3, 0].add(jnp.asarray([0.25, -0.1, 0.35], jnp.float32))
    lagged, _ = _run_once(lagged_program, case, old)
    new_pair = lagged.next_pair_table[:3]
    expected_mass = jnp.exp((new_pair[:, 0] - old[:3, 0]) + (new_pair[:, 1] - old[:3, 1]))
    np.testing.assert_allclose(np.asarray(lagged.batch_mass[:3]), np.asarray(expected_mass), rtol=2e-4, atol=2e-4)
    assert not np.asarray(lagged.invalid_normalizer).any()
    assert not np.asarray(lagged.invalid_weight).any()


@pytest.mark.gpu
def test_compiled_batch_reuses_capacity_texture_until_concrete_completion():
    from relax.cuda.kernels import RelionCapacityHalfTextureF32
    from relax.reconstruction.half_volume_mstep import relion_backprojector_volume_shape

    args = SimpleNamespace(
        seed=20260928,
        images=3,
        rotations=3,
        translations=5,
        rotation_tile=2,
        translation_tile=3,
        translation_side="image",
        box=8,
        mode="exact",
    )
    case = _make_case(args)
    reference, batch, grid, _, _, score_indices, rec_indices, _, volume_size = case
    radius = args.box // 4
    texture = RelionCapacityHalfTextureF32(reference, radius, padding_factor=1)
    project, backproject = native_relion_callbacks(
        image_shape=(args.box, args.box),
        score_indices=score_indices,
        rec_indices=rec_indices,
        r_max=radius,
        projector_output_size=args.box,
        volume_shape=relion_backprojector_volume_shape((args.box,) * 3, 1, current_size=2 * radius),
        capacity_texture=texture,
    )
    program = make_batch_program(DenseGemmTileConfig(3, 2, 3, "image", "exact"), project, backproject)
    old = empty_normalizer_table(4)
    first = program(
        reference,
        jnp.zeros((volume_size,), jnp.complex64),
        jnp.zeros((volume_size,), jnp.float32),
        old,
        empty_normalizer_table(4),
        batch,
        grid,
    )
    second = program(
        reference,
        jnp.zeros((volume_size,), jnp.complex64),
        jnp.zeros((volume_size,), jnp.float32),
        old,
        empty_normalizer_table(4),
        batch,
        grid,
    )
    texture.close_after(second.numerator)
    assert texture.closed
    np.testing.assert_allclose(np.asarray(first.numerator), np.asarray(second.numerator), rtol=2e-4, atol=2e-4)
    texture.close()


@pytest.mark.gpu
def test_capacity_texture_refresh_reuses_compiled_projection_and_matches_new_owner():
    from relax.cuda.kernels import RelionCapacityHalfTextureF32

    args = SimpleNamespace(
        seed=20260928, images=3, rotations=3, translations=5,
        rotation_tile=2, translation_tile=3, translation_side="image",
        box=8, padding_factor=2, mode="exact",
    )
    reference, _, grid, _, _, _, _, _, _ = _make_case(args)
    radius = args.box // 4
    owner = RelionCapacityHalfTextureF32(reference, radius, padding_factor=2, reusable_staging=True)
    traces = []

    @jax.jit
    def compiled(rotations):
        traces.append("traced")
        return owner.project(rotations, image_shape=(args.box, args.box))

    rotations = grid.score_rotations[:2]
    first = compiled(rotations)
    changed_reference = reference * jnp.complex64(1.25 + 0.1j)
    elapsed = owner.refresh_after(changed_reference, first)
    second = compiled(rotations)
    fresh = RelionCapacityHalfTextureF32(changed_reference, radius, padding_factor=2)
    expected = fresh.project(rotations, image_shape=(args.box, args.box))
    np.testing.assert_allclose(np.asarray(second), np.asarray(expected), rtol=2e-5, atol=2e-5)
    assert len(traces) == 1
    assert elapsed >= 0
    owner.close_after(second)
    fresh.close_after(expected)


@pytest.mark.gpu
def test_pf2_fused_coarse_control_matches_full_grid_gemm_operator():
    args = SimpleNamespace(
        seed=20260928,
        images=3,
        rotations=3,
        translations=5,
        rotation_tile=2,
        translation_tile=3,
        translation_side="image",
        box=8,
        padding_factor=2,
        mode="exact",
    )
    case = _make_case(args)
    old = empty_normalizer_table(4)
    gemm_program = make_batch_program(DenseGemmTileConfig(3, 2, 3, "image", "exact"), case[3], case[4])
    gemm, _ = _run_once(gemm_program, case, old)
    full_reference = relion_projector_half_to_texture_full(case[0])
    control_case = (full_reference, *case[1:])
    control_program = _production_program(args, case[4], case[7])
    control, _ = _run_once(control_program, control_case, old)
    np.testing.assert_allclose(np.asarray(gemm.batch_logz), np.asarray(control.batch_logz), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(gemm.numerator), np.asarray(control.numerator), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(gemm.denominator), np.asarray(control.denominator), rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
def test_native_bpref_translation_and_particle_grid_adjoint_matches_collapsed_control():
    args = SimpleNamespace(
        seed=20260928, images=3, rotations=3, translations=5, rotation_tile=2,
        translation_tile=3, translation_side="image", box=8, padding_factor=2,
        mode="exact", engine="production",
    )
    case = _make_case(args)
    old = empty_normalizer_table(4)
    full_reference = relion_projector_half_to_texture_full(case[0])
    control_case = (full_reference, *case[1:])
    collapsed, _ = _run_once(_production_program(args, case[4], case[7]), control_case, old)
    args.engine = "native"
    native, _ = _run_once(_production_program(args, case[4], case[7]), control_case, old)
    np.testing.assert_allclose(np.asarray(native.batch_logz), np.asarray(collapsed.batch_logz), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(native.numerator), np.asarray(collapsed.numerator), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(native.denominator), np.asarray(collapsed.denominator), rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
def test_native_bpref_rows_match_gemm_with_heterogeneous_zero_ctf_and_distinct_windows():
    args = SimpleNamespace(
        seed=20260929, images=3, rotations=3, translations=5, rotation_tile=2,
        translation_tile=3, translation_side="projection", box=8, padding_factor=2,
        mode="exact", engine="native",
    )
    case = _make_case(args)
    assert case[1].score_image.shape[1] != case[1].rec_image.shape[1]
    raw = np.asarray(case[1].rec_raw_image)
    ctf = np.linspace(-1.1, 0.9, raw.shape[1], dtype=np.float32)[None, :]
    ctf = np.repeat(ctf, args.images, axis=0) * np.asarray([0.7, 1.0, 1.3], np.float32)[:, None]
    ctf[:, ::4] = 0
    rec_weight = (ctf * ctf * np.float32(0.7)).astype(np.float32)
    batch = case[1]._replace(
        rec_image=jnp.asarray(raw * ctf, jnp.complex64),
        rec_raw_image=jnp.asarray(raw, jnp.complex64),
        rec_weighted_ctf=jnp.asarray(ctf, jnp.float32),
        rec_weight=jnp.asarray(rec_weight, jnp.float32),
    )
    case = (case[0], batch, *case[2:])
    old = empty_normalizer_table(4)
    gemm, _ = _run_once(
        make_batch_program(DenseGemmTileConfig(3, 2, 3, "projection", "exact"), case[3], case[4]),
        case, old,
    )
    native_case = (relion_projector_half_to_texture_full(case[0]), *case[1:])
    native, _ = _run_once(_production_program(args, case[4], case[7]), native_case, old)
    np.testing.assert_allclose(np.asarray(gemm.numerator), np.asarray(native.numerator), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(gemm.denominator), np.asarray(native.denominator), rtol=2e-4, atol=2e-4)
