"""Same-grid CPU checks for the opt-in K-class score GEMM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers.float_compare import assert_matches
from relax.dense.gemm_experiment import (
    DenseGemmTileConfig,
    make_joint_k_batch_program,
    pad_batch,
    pad_grid,
)
from relax.dense.gemm_experiment_kernels import class_score_tile, score_tile


def _case(*, skewed, score_mode):
    rng = np.random.default_rng(20260929)
    k, b, r, t, ps, pr = 4, 2, 3, 3, 5, 6

    def complex_values(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    references = complex_values((k, r, ps))
    images = complex_values((b, ps))
    score_weight = rng.uniform(0.2, 1.0, (b, ps)).astype(np.float32)
    score_phase = np.exp(1j * rng.normal(size=(t, ps))).astype(np.complex64)
    rec_phase = np.exp(1j * rng.normal(size=(t, pr))).astype(np.complex64)
    prior_mass = [0.7, 0.2, 0.09, 0.01] if skewed else [0.25] * k
    priors = (rng.uniform(-0.1, 0.1, (k, b, r))
              + np.log(np.asarray(prior_mass))[:, None, None]).astype(np.float32)
    if skewed:
        priors = priors[:, :1, :]  # exercise the production shared-prior layout
    rotations = np.zeros((r, 3, 3), np.float32)
    rotations[:, 0, 0] = np.arange(r)
    grid = pad_grid(rotations, rotations, score_phase, rec_phase,
                    rotation_tile=2, translation_tile=2)
    batch = pad_batch(
        images, score_weight, np.zeros(b, np.float32),
        complex_values((b, pr)), rng.uniform(0.2, 1.0, (b, pr)).astype(np.float32),
        None, np.zeros((b, t), np.float32), [0, 1],
        image_capacity=3, grid=grid, sentinel_id=2,
    )
    padded_priors = jnp.pad(
        jnp.asarray(priors), ((0, 0), (0, 0 if skewed else 1),
                              (0, grid.score_rotations.shape[0] - r)),
        constant_values=-jnp.inf,
    )

    def project(reference, selected_rotations):
        return reference[selected_rotations[:, 0, 0].astype(jnp.int32)]

    def backproject(y, w, ys, ws, _rotations):
        return y + jnp.sum(ys, axis=0), w + jnp.sum(ws, axis=0)

    return references, batch, grid, padded_priors, project, backproject, score_mode, pr


@pytest.mark.unit
@pytest.mark.parametrize("skewed", [False, True])
@pytest.mark.parametrize("score_mode", ["gaussian", "normalized_cc"])
@pytest.mark.parametrize("translation_side", ["image", "projection"])
def test_class_batched_first_sweep_matches_sequential_exact_grid(skewed, score_mode, translation_side):
    references, batch, grid, priors, project, backproject, mode, pr = _case(
        skewed=skewed, score_mode=score_mode,
    )
    config = DenseGemmTileConfig(3, 2, 2, translation_side, "exact")

    def statistics_step(carry, _ops, _weights, rec_projection, class_id, *_):
        assert rec_projection is not None
        return carry + class_id + 1

    def statistics_finish_class(carry, _ops, _class_id):
        return carry * 10

    def run(batched, cached=False, cache_limit=512 << 20):
        program = make_joint_k_batch_program(
            config, project, backproject, score_mode=mode,
            class_batch_scores=batched, cache_scores=cached,
            score_cache_max_bytes=cache_limit,
            project_reconstruction=project, statistics_step=statistics_step,
            statistics_finish_class=statistics_finish_class,
        )
        return program(
            jnp.asarray(references), jnp.zeros((4, 2, pr), jnp.complex64),
            jnp.zeros((4, 2, pr), jnp.float32), batch, grid, priors,
            jnp.asarray([0, 1, -1], jnp.int32),
            jnp.zeros((), jnp.int32),
        )

    control = run(False)
    assert int(np.asarray(control.statistics_state)) == 49360
    for candidate in (run(True), run(False, cached=True)):
        assert int(np.asarray(candidate.statistics_state)) == 49360
        for name in (
            "joint_normalizer", "class_normalizers", "class_best_scores",
            "numerator", "denominator", "rotation_posterior_sums",
            "class_posterior_sums", "class_translation_marginals",
            "image_posterior_mass",
        ):
            actual = np.asarray(getattr(candidate, name))
            expected = np.asarray(getattr(control, name))
            # These F64 metadata totals only accumulate F32 posterior tiles.
            if name in ("rotation_posterior_sums", "class_posterior_sums"):
                actual, expected = actual.astype(np.float32), expected.astype(np.float32)
            assert_matches(actual, expected, err_msg=name)
        for name in ("class_best_pose_ids", "invalid_normalizer", "invalid_weight"):
            np.testing.assert_array_equal(np.asarray(getattr(candidate, name)),
                                          np.asarray(getattr(control, name)))


@pytest.mark.unit
def test_class_score_cache_rejects_padded_grid_above_cap():
    references, batch, grid, priors, project, backproject, _, pr = _case(
        skewed=False, score_mode="gaussian",
    )
    program = make_joint_k_batch_program(
        DenseGemmTileConfig(3, 2, 2, "image", "exact"), project, backproject,
        cache_scores=True, score_cache_max_bytes=1,
    )
    with pytest.raises(MemoryError, match="score cache needs"):
        program(
            jnp.asarray(references), jnp.zeros((4, 2, pr), jnp.complex64),
            jnp.zeros((4, 2, pr), jnp.float32), batch, grid, priors,
            jnp.asarray([0, 1, -1], jnp.int32),
        )


@pytest.mark.unit
def test_class_score_tile_uses_flattened_projection_axis(monkeypatch):
    from relax.dense import gemm_experiment_kernels as kernels

    k, b, q, u, p = 4, 2, 3, 2, 5
    projections = jnp.ones((k, q, p), jnp.complex64)
    images = jnp.ones((b, p), jnp.complex64)
    weights = jnp.ones((b, p), jnp.float32)
    phases = jnp.ones((u, p), jnp.complex64)
    priors = jnp.zeros((k, b, q), jnp.float32)
    observed = []
    original = kernels.relion_coarse_gemm_terms

    def capture(projection, *args, **kwargs):
        observed.append(projection.shape)
        return original(projection, *args, **kwargs)

    monkeypatch.setattr(kernels, "relion_coarse_gemm_terms", capture)
    actual = class_score_tile(
        projections, images, weights, jnp.zeros(b, jnp.float32), phases,
        priors, jnp.zeros((b, u), jnp.float32), jnp.ones(b, bool),
        jnp.ones(q, bool), jnp.ones(u, bool), translation_side="image",
        score_mode="gaussian", model_power=None,
    )
    assert observed == [(k * q, p)]
    expected = jnp.stack([
        score_tile(
            projections[class_id], images, weights, jnp.zeros(b, jnp.float32),
            phases, priors[class_id], jnp.zeros((b, u), jnp.float32),
            jnp.ones(b, bool), jnp.ones(q, bool), jnp.ones(u, bool),
            translation_side="image",
        )
        for class_id in range(k)
    ])
    assert_matches(np.asarray(actual), np.asarray(expected))


@pytest.mark.unit
def test_cc_class_tie_keeps_smallest_class_and_pose():
    references, batch, grid, priors, project, backproject, _, pr = _case(
        skewed=False, score_mode="normalized_cc",
    )
    references[:] = 1.0 + 0j
    grid = grid._replace(score_phase=jnp.ones_like(grid.score_phase))
    result = make_joint_k_batch_program(
        DenseGemmTileConfig(3, 2, 2, "image", "exact"), project, backproject,
        score_mode="normalized_cc", class_batch_scores=True,
    )(
        jnp.asarray(references), jnp.zeros((4, 2, pr), jnp.complex64),
        jnp.zeros((4, 2, pr), jnp.float32), batch, grid, priors,
        jnp.asarray([0, 1, -1], jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(result.class_best_pose_ids)[:, :2], 0)
    assert_matches(np.asarray(result.class_posterior_sums), np.asarray([2, 0, 0, 0], np.float32))
