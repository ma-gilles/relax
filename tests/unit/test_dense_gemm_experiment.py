"""Independent small-grid checks for the opt-in dense GEMM experiment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.dense.gemm_experiment import (
    DenseGemmTileConfig,
    make_batch_program,
    pad_batch,
    pad_grid,
    run_resident_iteration,
)
from relax.dense.gemm_experiment_kernels import empty_normalizer_table, score_tile, weighted_slices


def _case():
    rng = np.random.default_rng(829)
    b, r, t, ps, pr = 3, 5, 5, 7, 9

    def complex_random(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    score_image = complex_random((b, ps))
    score_image[1, 2] = 0
    projection = complex_random((r, ps))
    rec_image = complex_random((b, pr))
    score_weight = rng.uniform(0.1, 1.2, (b, ps)).astype(np.float32)
    score_weight[1, 2] = 0  # zero CTF/noise contribution
    rec_weight = rng.uniform(0.1, 1.2, (b, pr)).astype(np.float32)
    angles = np.array([-0.91, -0.33, 0.0, 0.24, 0.76], np.float32)
    k_score = rng.uniform(-1.5, 1.7, ps).astype(np.float32)
    k_rec = rng.uniform(-1.1, 1.3, pr).astype(np.float32)
    score_phase = np.exp(1j * angles[:, None] * k_score[None, :]).astype(np.complex64)
    rec_phase = np.exp(1j * angles[:, None] * k_rec[None, :]).astype(np.complex64)
    initial = rng.uniform(0.2, 1.0, b).astype(np.float32)
    rot_prior = rng.uniform(-0.3, 0.1, (b, r)).astype(np.float32)
    trans_prior = rng.uniform(-0.2, 0.1, (b, t)).astype(np.float32)
    return (
        score_image,
        projection,
        score_weight,
        initial,
        rec_image,
        rec_weight,
        score_phase,
        rec_phase,
        rot_prior,
        trans_prior,
    )


def test_benchmark_volume_alias_check_uses_live_parameter_indices():
    from types import SimpleNamespace

    from scripts.benchmark_dense_gemm_em import _volume_aliases

    class Compiled:
        def memory_analysis(self):
            return SimpleNamespace(
                temp_size_in_bytes=4,
                argument_size_in_bytes=100,
                output_size_in_bytes=100,
                alias_size_in_bytes=120,
            )

    hlo = (
        "HloModule jit_reset, is_scheduled=true, input_output_alias={ "
        "{0}: (1, {}, may-alias), {1}: (2, {}, may-alias) }, "
        "entry_computation_layout={(c64[10], f32[10])->(c64[10], f32[10])}\n"
    )
    receipt = _volume_aliases(hlo, Compiled(), 10)
    assert receipt["verified"]
    assert receipt["output_to_input"] == {0: 1, 1: 2}
    assert not _volume_aliases(hlo.replace("{1}: (2, {}, may-alias)", "{1}: (1, {}, may-alias)"), Compiled(), 10)[
        "verified"
    ]


def _direct_scores(case):
    image, projection, weight, initial, _, _, phase, _, rp, tp = case
    difference = projection[None, :, None, :] - image[:, None, None, :] * phase[None, None, :, :]
    square = difference.real**2 + difference.imag**2
    return (
        -0.5 * np.sum(square * weight[:, None, None, :], axis=-1)
        - initial[:, None, None]
        + rp[:, :, None]
        + tp[:, None, :]
    )


def _dummy_program(config, projection):
    def project(reference, rotations):
        return reference[rotations[:, 0, 0].astype(jnp.int32)]

    def backproject(y, w, ys, ws, _rotations):
        return y + jnp.sum(ys, axis=0), w + jnp.sum(ws, axis=0)

    # The reference is only a projection table in this independent test.
    return make_batch_program(config, project, backproject), jnp.asarray(projection)


def _run(case, side, tile, *, mode="exact", old=None, particle_ids=(3, 1, 4)):
    image, projection, weight, initial, rec_image, rec_weight, phase, rec_phase, rp, tp = case
    b, r, t = len(image), len(projection), len(phase)
    rotations = np.zeros((r, 3, 3), np.float32)
    rotations[:, 0, 0] = np.arange(r)
    grid = pad_grid(rotations, rotations, phase, rec_phase, rotation_tile=2, translation_tile=tile)
    batch = pad_batch(
        image,
        weight,
        initial,
        rec_image,
        rec_weight,
        rp,
        tp,
        particle_ids,
        image_capacity=4,
        grid=grid,
        sentinel_id=5,
    )
    config = DenseGemmTileConfig(4, 2, tile, side, mode)
    program, reference = _dummy_program(config, projection)
    old_table = empty_normalizer_table(6) if old is None else jnp.asarray(old)
    result = program(
        reference,
        jnp.zeros((rec_image.shape[1],), jnp.complex64),
        jnp.zeros((rec_image.shape[1],), jnp.float32),
        old_table,
        empty_normalizer_table(6),
        batch,
        grid,
    )
    return jax.tree.map(np.asarray, result)


@pytest.mark.parametrize("side", ["image", "projection"])
@pytest.mark.parametrize("tile", [1, 2, 5])
def test_scores_and_weighted_slices_match_direct_reference(side, tile):
    case = _case()
    image, projection, weight, initial, rec_image, rec_weight, phase, rec_phase, rp, tp = case
    scores = np.asarray(
        score_tile(
            jnp.asarray(projection),
            jnp.asarray(image),
            jnp.asarray(weight),
            jnp.asarray(initial),
            jnp.asarray(phase),
            jnp.asarray(rp),
            jnp.asarray(tp),
            jnp.ones((len(image),), bool),
            jnp.ones((len(projection),), bool),
            jnp.ones((len(phase),), bool),
            translation_side=side,
        )
    )
    np.testing.assert_allclose(scores, _direct_scores(case), rtol=2e-5, atol=2e-5)
    logz = np.asarray(jax.nn.logsumexp(jnp.asarray(scores).reshape(len(image), -1), axis=1))
    q = np.exp(scores - logz[:, None, None])
    ys, ws = weighted_slices(
        jnp.asarray(q),
        jnp.asarray(rec_image),
        jnp.asarray(rec_weight),
        jnp.asarray(rec_phase),
        translation_side=side,
    )
    direct_y = np.einsum("brt,btp->rp", q, rec_image[:, None, :] * rec_phase[None, :, :])
    direct_w = np.einsum("brt,bp->rp", q, rec_weight)
    np.testing.assert_allclose(ys, direct_y, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(ws, direct_w, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("side", ["image", "projection"])
@pytest.mark.parametrize("tile", [1, 2, 5])
def test_compiled_two_sweep_exact_tile_and_tail_invariance(side, tile):
    case = _case()
    result = _run(case, side, tile)
    scores = _direct_scores(case)
    logz = np.asarray(jax.nn.logsumexp(jnp.asarray(scores).reshape(3, -1), axis=1))
    q = np.exp(scores - logz[:, None, None])
    direct_y = np.einsum("brt,btp->p", q, case[4][:, None, :] * case[7][None, :, :])
    direct_w = np.einsum("brt,bp->p", q, case[5])
    np.testing.assert_allclose(result.batch_logz[:3], logz, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result.numerator, direct_y, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(result.denominator, direct_w, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(result.batch_mass[:3], 1, rtol=3e-5, atol=3e-5)
    assert result.next_logz_table[3] == result.batch_logz[0]
    assert result.next_logz_table[1] == result.batch_logz[1]
    assert result.next_logz_table[4] == result.batch_logz[2]
    assert not result.invalid_normalizer.any()
    assert not result.invalid_weight.any()


@pytest.mark.parametrize("side", ["image", "projection"])
def test_lagged_stable_ids_and_unrenormalized_mass(side):
    case = _case()
    exact = _run(case, side, 2)
    old = exact.next_pair_table.copy()
    old[[3, 1, 4], 0] += np.array([0.3, -0.2, 0.5], np.float32)
    lagged = _run(case, side, 2, mode="lagged", old=old, particle_ids=(3, 1, 4))
    expected = np.exp(exact.batch_logz[:3] - old[[3, 1, 4]].sum(axis=1))
    np.testing.assert_allclose(lagged.batch_mass[:3], expected, rtol=3e-5, atol=3e-5)
    same = _run(case, side, 2, mode="lagged", old=exact.next_pair_table)
    np.testing.assert_allclose(same.numerator, exact.numerator, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(same.denominator, exact.denominator, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(same.batch_mass[:3], 1, rtol=3e-5, atol=3e-5)
    assert not lagged.invalid_weight.any()


def test_invalid_lagged_normalizer_is_reported():
    case = _case()
    old = np.full((6, 2), -np.inf, np.float32)
    old[[3, 1, 4]] = (1.0, 0.0)
    old[1] = (-np.inf, -np.inf)
    result = _run(case, "image", 2, mode="lagged", old=old)
    assert result.invalid_normalizer[1]
    assert result.invalid_weight[1]


def test_iteration_keeps_normalizers_by_particle_id_across_reordered_batches():
    case = _case()
    image, projection, weight, initial, rec_image, rec_weight, phase, rec_phase, rp, tp = case
    rotations = np.zeros((len(projection), 3, 3), np.float32)
    rotations[:, 0, 0] = np.arange(len(projection))
    grid = pad_grid(rotations, rotations, phase, rec_phase, rotation_tile=2, translation_tile=2)

    def batches(order):
        result = []
        for group in order:
            rows = np.asarray(group, dtype=np.int32)
            result.append(
                pad_batch(
                    image[rows], weight[rows], initial[rows], rec_image[rows], rec_weight[rows],
                    rp[rows], tp[rows], rows, image_capacity=2, grid=grid, sentinel_id=3,
                )
            )
        return result

    exact_program, reference = _dummy_program(DenseGemmTileConfig(2, 2, 2, "projection", "exact"), projection)
    old = empty_normalizer_table(4)
    exact = run_resident_iteration(
        exact_program, reference, batches(((2, 0), (1,))), grid, old, volume_size=rec_image.shape[1]
    )
    lagged_program, _ = _dummy_program(DenseGemmTileConfig(2, 2, 2, "projection", "lagged"), projection)
    lagged = run_resident_iteration(
        lagged_program, reference, batches(((1,), (0, 2))), grid,
        exact.next_pair_table, volume_size=rec_image.shape[1],
    )
    np.testing.assert_allclose(lagged.next_logz_table[:3], exact.next_logz_table[:3], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(lagged.numerator, exact.numerator, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(lagged.denominator, exact.denominator, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(lagged.mass_table[:3], 1, rtol=3e-5, atol=3e-5)
    assert not np.asarray(lagged.invalid_normalizer_table).any()
    assert not np.asarray(lagged.invalid_weight_table).any()
def test_native_reconstruction_square_has_unique_scatter_destinations():
    from scripts.benchmark_dense_gemm_em import _centered_reconstruction_indices, _native_reconstruction_indices

    for size, radius in ((8, 2), (64, 16), (128, 32)):
        indices = np.asarray(_native_reconstruction_indices(size, radius))
        assert len(indices) == 2 * radius * (radius + 1)
        row = indices // (size // 2 + 1)
        col = indices % (size // 2 + 1)
        destinations = (row % (2 * radius)) * (radius + 1) + col
        assert len(np.unique(destinations)) == len(indices)
        assert size - radius not in row
        centered = np.asarray(_centered_reconstruction_indices(indices, size))
        centered_row = centered // (size // 2 + 1)
        assert np.array_equal(centered % (size // 2 + 1), col)
        assert np.array_equal(centered_row, (row + size // 2) % size)


@pytest.mark.parametrize("side", ["image", "projection"])
@pytest.mark.parametrize("tile", [1, 2, 5])
def test_split_normalizer_preserves_large_absolute_offset_and_lagged_mass(side, tile):
    case = list(_case())
    case[3] = np.asarray(case[3], np.float32) + np.float32(20_000)
    case = tuple(case)
    exact = _run(case, side, tile)
    live_pair = exact.next_pair_table[[3, 1, 4]]
    assert np.all(np.isfinite(live_pair))
    assert np.all(live_pair[:, 0] < -19_000)
    np.testing.assert_allclose(exact.batch_mass[:3], 1, rtol=3e-4, atol=3e-4)
    lagged = _run(case, side, tile, mode="lagged", old=exact.next_pair_table)
    np.testing.assert_allclose(lagged.batch_mass[:3], 1, rtol=3e-4, atol=3e-4)
    shifted_old = exact.next_pair_table.copy()
    shifted_old[[3, 1, 4], 0] += np.float32(0.125)
    shifted = _run(case, side, tile, mode="lagged", old=shifted_old)
    np.testing.assert_allclose(shifted.batch_mass[:3], np.exp(-0.125), rtol=4e-4, atol=4e-4)
    assert not shifted.invalid_normalizer[:3].any()
