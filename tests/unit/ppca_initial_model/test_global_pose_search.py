"""The global pose search of single particles (``Config.pose_search = "global"``).

Pass 1 scores the HEALPix order hp - 1 grid and twice the stage's shift step; pass 2 scores the children of each
image's significant samples on the oversampling engine, each child with its own priors and its own rotation mass.
With fraction 1 and no cap every coarse sample is significant, so pass 2 is the dense child grid: the E-step must
equal the dense stream over that grid with the same priors.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.full_row_tile import accumulate_full_row_tile
from test_full_fine_streaming import IMAGE_SHAPE, N_HALF, _half_volume, _TinyData
from test_oversampled_stream import DIAGNOSTIC_RTOL, STATISTICS, STATISTICS_RTOL, _assert_same_statistics, _spa_stream

from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.state import State
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_stream import full_row_tile_embeddings, prepare_full_row_stream
from relax.ppca_refinement.oversampled_stream import (
    accumulate_oversampled_tile,
    oversampled_tile_embeddings,
    prepare_oversampled_stream,
)
from relax.sampling import grids

pytestmark = pytest.mark.unit


def test_own_child_priors_and_child_mass_equal_the_dense_child_grid():
    """Children with their own priors and per-child rotation mass: with fraction 1 the dense child grid's
    statistics and rotation mass, child for child."""
    coarse, (fine_rotations, fine_translations) = _spa_stream()
    rng = np.random.default_rng(3)
    rotation_prior = np.log(rng.dirichlet(np.ones(len(fine_rotations)))).astype(np.float32)
    translation_prior = np.log(rng.dirichlet(np.ones(len(fine_translations)))).astype(np.float32)
    n_poses = (coarse.arrays.rotations.shape[0] - 1) * coarse.translations.shape[0]
    ostream = prepare_oversampled_stream(
        coarse,
        fine_rotations,
        fine_translations,
        adaptive_fraction=1.0,
        max_significant=n_poses,
        job_chunk=96,
        fine_rotation_log_prior=rotation_prior,
        fine_translation_log_prior=translation_prior,
        child_rotation_mass=True,
    )
    np.testing.assert_array_equal(np.asarray(ostream.fine.arrays.rotation_log_prior[:-1]), rotation_prior)
    np.testing.assert_array_equal(np.asarray(ostream.fine.arrays.translation_log_prior), translation_prior)
    ids = np.arange(4)
    actual = accumulate_oversampled_tile(ostream, ids)
    expected = accumulate_full_row_tile(ostream.fine, ids, [None] * 4)
    _assert_same_statistics(actual, expected)
    assert actual.diagnostics["rotation_mass"].shape == (len(fine_rotations),)
    assert_matches(actual.diagnostics["rotation_mass"], expected.diagnostics["rotation_mass"], rtol=DIAGNOSTIC_RTOL)


def test_embeddings_report_the_dense_child_grids_top_pose():
    """With fraction 1 the embeddings pass gives the dense child grid's latents and top pose (child-grid rows)."""
    coarse, (fine_rotations, fine_translations) = _spa_stream()
    n_poses = (coarse.arrays.rotations.shape[0] - 1) * coarse.translations.shape[0]
    ostream = prepare_oversampled_stream(
        coarse, fine_rotations, fine_translations, adaptive_fraction=1.0, max_significant=n_poses, job_chunk=96
    )
    ids = np.arange(4)
    actual = oversampled_tile_embeddings(ostream, ids)
    expected = full_row_tile_embeddings(ostream.fine, ids, [None] * 4)
    np.testing.assert_array_equal(actual.best_rotation, expected.best_rotation)
    np.testing.assert_array_equal(actual.best_translation, expected.best_translation)
    assert_matches(np.asarray(actual.embeddings), np.asarray(expected.embeddings), rtol=STATISTICS_RTOL)


def test_child_priors_must_have_one_entry_per_child():
    coarse, (fine_rotations, fine_translations) = _spa_stream()
    with pytest.raises(ValueError, match="one entry per child"):
        prepare_oversampled_stream(
            coarse, fine_rotations, fine_translations, fine_rotation_log_prior=np.zeros(3, np.float32)
        )


@pytest.mark.parametrize("hp, shift_range, shift_step", [(1, 4.0, 1.0), (2, 8.0, 2.0)])
def test_global_grids_are_the_stage_order_and_step(hp, shift_range, shift_step):
    """The children cover the order-hp directions as the stage grid does (as many rotations per direction), the
    child shifts sit half a coarse step apart, and the coarse priors are the log-sums of their children's."""
    search = iteration_loop._global_search_grids(hp, shift_range, shift_step)
    stage = grids.get_relion_hidden_rotation_grid(hp, matrices=True)
    assert search.coarse_order == hp - 1
    assert len(search.fine_rotations) == len(stage) == 8 * len(search.coarse_rotations)
    ids = iteration_loop._direction_ids(hp)
    np.testing.assert_array_equal(np.bincount(search.child_direction), np.bincount(ids))
    fine = np.asarray(search.fine_translations).reshape(len(search.coarse_translations), 4, 2)
    np.testing.assert_allclose(fine.mean(axis=1), search.coarse_translations, atol=1e-6)
    np.testing.assert_allclose(np.abs(fine - search.coarse_translations[:, None]), shift_step / 2, atol=1e-6)
    rotation_prior = np.log(np.random.default_rng(1).dirichlet(np.ones(len(ids)))).astype(np.float32)
    # A direction prior is constant over a direction's in-plane angles.
    rotation_prior = np.log(np.bincount(ids, np.exp(rotation_prior))[ids] / np.bincount(ids)[ids]).astype(np.float32)
    priors = iteration_loop._global_search_priors(search, rotation_prior, 3.0, hp)
    first = np.unique(ids, return_index=True)[1]
    np.testing.assert_array_equal(priors["fine_rotation_log_prior"], rotation_prior[first][search.child_direction])
    np.testing.assert_allclose(np.exp(priors["fine_translation_log_prior"]).sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        priors["coarse_rotation_log_prior"],
        np.log(np.exp(priors["fine_rotation_log_prior"].astype(np.float64)).reshape(-1, 8).sum(axis=1)),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        priors["coarse_translation_log_prior"],
        np.log(np.exp(priors["fine_translation_log_prior"].astype(np.float64)).reshape(-1, 4).sum(axis=1)),
        rtol=1e-5,
    )
    mass = np.random.default_rng(2).random(len(search.fine_rotations))
    stage_mass = iteration_loop._stage_rotation_mass(search, mass, hp)
    np.testing.assert_allclose(np.bincount(ids, stage_mass), np.bincount(search.child_direction, mass), rtol=1e-6)


def test_global_search_refuses_what_it_does_not_implement():
    with pytest.raises(ValueError, match="global pose search"):
        Config(pose_search="global", oversampling=1)
    with pytest.raises(ValueError, match="global pose search"):
        Config(pose_search="global", stream_coarse_recompute=False)
    with pytest.raises(ValueError, match="pose_search"):
        Config(pose_search="bnb")


def test_global_expectation_with_every_sample_is_the_dense_child_grid():
    """The controller's global E-step with fraction 1 and no cap equals the dense stream over the child grid with the
    children's own priors; its rotation mass is the child grid's, spread over the stage grid's directions."""
    rng = np.random.default_rng(7)
    images = (rng.standard_normal((6, N_HALF)) + 1j * rng.standard_normal((6, N_HALF))).astype(np.complex64)
    images[:, 0] += 30.0
    theta = np.stack([_half_volume(rng), _half_volume(rng, 0.3), _half_volume(rng, 0.2)], axis=1)
    dataset = _TinyData(images)
    hp, shift_range, shift_step = 1, 2.0, 1.0
    search = iteration_loop._global_search_grids(hp, shift_range, shift_step)
    config = Config(
        q=2,
        stages=((1, 3, hp),),
        pose_search="global",
        target_mass=1.0,
        max_significant=len(search.coarse_rotations) * len(search.coarse_translations),
        shift_range=shift_range,
        shift_step=shift_step,
        image_batch_size=6,
        rotation_block_size=64,
        gemm_precision="fp32",
    )
    noise = np.full(IMAGE_SHAPE[0] // 2 + 1, 40.0, np.float32)
    ids = iteration_loop._direction_ids(hp)
    direction_prior = rng.dirichlet(np.ones(ids.max() + 1))
    rotation_prior = (direction_prior[ids] / np.bincount(ids)[ids]).astype(np.float32)
    state = State(theta, None, noise, 0, np.arange(6), {}, 2.5, 3, {}, rotation_prior, hp)
    ids_all = np.arange(6)
    actual = iteration_loop.expectation(dataset, state, config, ids_all, 1, diameter_ang=6.0)

    priors = iteration_loop._global_search_priors(search, np.log(rotation_prior), 2.5, hp)
    from recovar.reconstruction.noise import make_radial_noise

    dense = prepare_full_row_stream(
        dataset,
        theta[:, 0],
        theta[:, 1:],
        noise_variance=np.asarray(make_radial_noise(noise, IMAGE_SHAPE), np.float32),
        rotations=search.fine_rotations,
        translations=search.fine_translations,
        rotation_log_prior=priors["fine_rotation_log_prior"],
        translation_log_prior=priors["fine_translation_log_prior"],
        rotation_parent=np.zeros(len(search.fine_rotations), np.int32),
        translation_parent=np.zeros(len(search.fine_translations), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=6, q=2, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=6, rotation_block_size=64),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        metric_trace_only=False,
        gemm_precision="fp32",
    )
    expected = accumulate_full_row_tile(dense, ids_all, [None] * 6)
    # The controller merges its tiles' statistics; per-image diagnostics are not kept.
    for name in STATISTICS:
        assert_matches(
            np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)), err_msg=name, rtol=STATISTICS_RTOL
        )
    assert_matches(np.float32(actual.log_likelihood), np.float32(expected.log_likelihood), rtol=DIAGNOSTIC_RTOL)
    for key in ("offset_second_sum_px2", "latent_covariance_trace_mean", "pose_entropy_mean", "pmax_mean"):
        assert_matches(
            np.float32(actual.diagnostics[key]),
            np.float32(expected.diagnostics[key]),
            err_msg=key,
            rtol=DIAGNOSTIC_RTOL,
        )
    assert actual.diagnostics["oversampling"]["pose_search"] == "global"
    assert actual.diagnostics["oversampling"]["pass1_order"] == hp - 1
    stage_mass = iteration_loop._stage_rotation_mass(search, expected.diagnostics["rotation_mass"], hp)
    assert_matches(actual.diagnostics["rotation_mass"], stage_mass, rtol=DIAGNOSTIC_RTOL)
