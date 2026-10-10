"""Adaptive oversampling of the mixture E-step: RELION's two passes per class, one joint normalizer.

References: the single-class oversampled stream (K=1 and identical components), and the mixture full-row
stream on the child grid (every coarse sample significant). CPU, float32; the tolerances are the oversampled
stream's measured reduction-order bands (``test_oversampled_stream``).
"""

import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_mixture_class3d import _stream
from test_mixture_controller import tiny_config, tomo_data
from test_oversampled_stream import STATISTICS_RTOL, _spa_stream
from test_tomo_ppca import make_problem

from relax.ppca_initial_class3d import iteration_loop
from relax.ppca_initial_class3d.config import Config
from relax.ppca_initial_class3d.expectation import oversampled_update, pose_grid
from relax.ppca_initial_class3d.stream import (
    accumulate_mixture_full_row_tile,
    accumulate_mixture_oversampled_tile,
    prepare_mixture_full_row_stream,
    prepare_mixture_oversampled_stream,
)
from relax.ppca_refinement.oversampled_stream import (
    accumulate_oversampled_tile,
    oversampled_tile_embeddings,
    prepare_oversampled_stream,
    relion_child_grids,
)

pytestmark = pytest.mark.unit


def _scaled(stream, factor, seed=11):
    """A distinct class with the same grids and noise: the augmented model scaled a little and another angular
    prior, so both classes keep comparable likelihoods and the responsibilities depend on the particle."""
    n_rotations = int(stream.arrays.rotations.shape[0]) - 1
    prior = np.log(np.random.default_rng(seed).dirichlet(np.ones(n_rotations))).astype(np.float32)
    arrays = stream.arrays._replace(
        augmented=stream.arrays.augmented * np.float32(factor),
        rotation_log_prior=jnp.asarray(np.append(prior, np.float32(0))),
    )
    return stream._replace(arrays=arrays)


def _ostreams(coarse, grids, models, **kwargs):
    fine_rotations, fine_translations = grids
    return [
        prepare_oversampled_stream(model, fine_rotations, fine_translations, job_chunk=16, **kwargs)
        for model in models
    ]


def _same_statistics(actual, expected, rtol=STATISTICS_RTOL):
    for field in ("lhs_tri", "residual_gradient"):
        assert_matches(getattr(actual, field), getattr(expected, field), rtol=rtol)


def test_one_class_equals_the_single_oversampled_stream():
    coarse, grids = _spa_stream()
    ostream, = _ostreams(coarse, grids, [coarse], adaptive_fraction=0.9, max_significant=40)
    ids = np.arange(4)
    single = accumulate_oversampled_tile(ostream, ids)
    mixture = accumulate_mixture_oversampled_tile(prepare_mixture_oversampled_stream([ostream], [1.0]), ids)
    assert_matches(mixture.class_probabilities, np.ones((4, 1), np.float32))
    _same_statistics(mixture.statistics[0], single)
    assert_matches(mixture.residual_num, single.residual_num, rtol=STATISTICS_RTOL)
    assert_matches(mixture.residual_den, single.residual_den)
    assert_matches(mixture.embeddings[:, 0], single.embeddings, rtol=STATISTICS_RTOL)
    assert_matches(mixture.log_likelihood, single.log_likelihood, rtol=1e-6)
    assert_matches(mixture.statistics[0].diagnostics["rotation_mass"], single.diagnostics["rotation_mass"],
                   rtol=STATISTICS_RTOL)
    np.testing.assert_array_equal(mixture.best_rotation_idx[:, 0], single.diagnostics["best_rotation_idx"])
    np.testing.assert_array_equal(mixture.best_translation_idx[:, 0], single.diagnostics["best_translation_idx"])
    np.testing.assert_array_equal(mixture.statistics[0].diagnostics["significant_samples_per_image"],
                                  single.diagnostics["significant_samples_per_image"])


def test_every_sample_significant_equals_the_mixture_full_row_child_grid():
    """With fraction 1, no cap and every child accumulated, the oversampled mixture is the full-row mixture over
    the child grid: the joint class/pose posterior, every class's statistics and the shared noise sums."""
    coarse, grids = _spa_stream()
    models = [coarse, _scaled(coarse, 0.995)]
    n_poses = (coarse.arrays.rotations.shape[0] - 1) * coarse.translations.shape[0]
    ostreams = _ostreams(coarse, grids, models, adaptive_fraction=1.0, max_significant=n_poses, fine_fraction=1.0)
    prior = [0.3, 0.7]
    ids = np.arange(4)
    actual = accumulate_mixture_oversampled_tile(prepare_mixture_oversampled_stream(ostreams, prior), ids)
    dense = accumulate_mixture_full_row_tile(
        prepare_mixture_full_row_stream([o.fine for o in ostreams], prior), ids, [None] * 4
    )
    assert actual.class_probabilities.shape == (4, 2) and np.ptp(actual.class_probabilities[:, 0]) > 1e-3
    assert_matches(actual.class_probabilities, dense.class_probabilities, rtol=STATISTICS_RTOL)
    assert_matches(actual.component_mass, dense.component_mass, rtol=STATISTICS_RTOL)
    for k in range(2):
        _same_statistics(actual.statistics[k], dense.statistics[k])
        assert_matches(actual.statistics[k].embeddings, dense.statistics[k].embeddings, rtol=STATISTICS_RTOL)
        assert_matches(actual.statistics[k].residual_den, np.zeros_like(dense.statistics[k].residual_den))
    assert_matches(actual.residual_num, dense.residual_num, rtol=STATISTICS_RTOL)
    assert_matches(actual.residual_den, dense.residual_den)
    assert_matches(actual.embeddings, dense.embeddings, rtol=STATISTICS_RTOL)
    assert_matches(actual.log_likelihood, dense.log_likelihood, rtol=1e-6)
    assert_matches(actual.max_posterior_per_image, dense.max_posterior_per_image, rtol=STATISTICS_RTOL)
    np.testing.assert_array_equal(actual.best_class, dense.best_class)
    # Pass 2 indexes the child grid, as the full-row mixture over that grid does.
    np.testing.assert_array_equal(actual.best_rotation_idx, dense.best_rotation_idx)
    np.testing.assert_array_equal(actual.best_translation_idx, dense.best_translation_idx)
    np.testing.assert_array_equal(actual.original_image_ids, dense.original_image_ids)
    assert actual.statistics[0].diagnostics["engine"] == "mixture_full_row_adaptive_oversampling"


def test_identical_components_split_priors_and_sum_to_the_single_stream():
    """Tilt particles: two copies of one class give the priors as responsibilities and the single stream's
    statistics as their sum, with the single stream's significant samples and support."""
    translations = np.asarray([[0, 0, 0], [0.7, 0, 0]], np.float32)
    particles, _, problem = make_problem(q=2, image_frames=[[0, 1], [1], [1, 0]], translations=translations)
    problem.update(translations=translations, rotations=problem["rotations"][:3],
                   rotation_log_prior=np.log(np.asarray([0.2, 0.3, 0.5], np.float32)))
    stream = _stream(particles, problem, tomography=True)
    fine_rotations = np.repeat(stream.arrays.rotations[:-1], 8, axis=0)  # any parent-major children
    fine_rotations = np.asarray(fine_rotations, np.float32)
    fine_translations = relion_child_grids(0, translations, 0.7)[1]
    ostream = prepare_oversampled_stream(stream, fine_rotations, fine_translations, adaptive_fraction=0.9,
                                         max_significant=4, job_chunk=8)
    ids = np.arange(particles.n_images)
    single = accumulate_oversampled_tile(ostream, ids)
    two = accumulate_mixture_oversampled_tile(prepare_mixture_oversampled_stream([ostream, ostream], [.2, .8]), ids)
    assert_matches(two.class_probabilities, np.tile(np.asarray([.2, .8], np.float32), (len(ids), 1)))
    for field in ("lhs_tri", "residual_gradient"):
        assert_matches(sum(getattr(s, field) for s in two.statistics), getattr(single, field), rtol=STATISTICS_RTOL)
    assert_matches(two.residual_num, single.residual_num, rtol=STATISTICS_RTOL)
    assert_matches(two.residual_den, single.residual_den)
    for k in range(2):
        assert_matches(two.embeddings[:, k], single.embeddings, rtol=STATISTICS_RTOL)
        np.testing.assert_array_equal(two.statistics[k].diagnostics["significant_samples_per_image"],
                                      single.diagnostics["significant_samples_per_image"])
        assert two.statistics[k].diagnostics["pass2_rows"] == single.diagnostics["pass2_rows"]
    assert_matches(two.log_likelihood, single.log_likelihood, rtol=1e-6)
    np.testing.assert_array_equal(two.best_class, np.ones(len(ids), np.int32))


def test_inference_only_embeddings_are_each_class_conditional_ones():
    coarse, grids = _spa_stream()
    models = [coarse, _scaled(coarse, 0.995)]
    ostreams = _ostreams(coarse, grids, models, adaptive_fraction=0.9, max_significant=40)
    ids = np.asarray([2, 0, 3])
    result = accumulate_mixture_oversampled_tile(
        prepare_mixture_oversampled_stream(ostreams, [.5, .5]), ids, moments=False
    )
    assert result.statistics == () and result.residual_num is None
    for k, ostream in enumerate(ostreams):
        expected = oversampled_tile_embeddings(ostream, ids)
        assert_matches(result.embeddings[:, k], expected.embeddings, rtol=STATISTICS_RTOL)
        np.testing.assert_array_equal(result.original_image_ids, expected.original_image_ids)


def test_padding_particles_have_no_mass():
    coarse, grids = _spa_stream(n_images=3)
    models = [coarse, _scaled(coarse, 0.995)]
    kwargs = dict(adaptive_fraction=0.9, max_significant=40)
    plain = _ostreams(coarse, grids, models, **kwargs)
    padded = _ostreams(coarse, grids, [m._replace(tile_images=4, image_batch_size=4) for m in models], **kwargs)
    ids = np.arange(3)
    expected = accumulate_mixture_oversampled_tile(prepare_mixture_oversampled_stream(plain, [.4, .6]), ids)
    actual = accumulate_mixture_oversampled_tile(prepare_mixture_oversampled_stream(padded, [.4, .6]), ids)
    assert actual.class_probabilities.shape == (3, 2)
    assert_matches(actual.component_mass.sum(), np.float32(3))
    assert_matches(actual.class_probabilities, expected.class_probabilities, rtol=STATISTICS_RTOL)
    for a, b in zip(actual.statistics, expected.statistics, strict=True):
        _same_statistics(a, b)
    assert_matches(actual.residual_num, expected.residual_num, rtol=STATISTICS_RTOL)


def test_mixture_refuses_classes_with_different_oversampling_rules():
    coarse, grids = _spa_stream()
    a, = _ostreams(coarse, grids, [coarse], adaptive_fraction=0.9, max_significant=40)
    b, = _ostreams(coarse, grids, [coarse], adaptive_fraction=0.9, max_significant=20)
    with pytest.raises(ValueError, match="oversampling rule"):
        prepare_mixture_oversampled_stream([a, b], [.5, .5])


@pytest.mark.parametrize("change", [
    {"oversampling": 1, "oversampling_start": 0}, {"oversampling": 0, "oversampling_start": 3},
    {"tomogram_batches": True, "balanced_stochastic_halves": True, "stochastic_batch_size": 4,
     "stochastic_all_iterations": True},
])
def test_configuration_rejects_inconsistent_oversampling_and_batch_options(change):
    with pytest.raises(ValueError):
        Config(**change)


def test_oversampled_updates_start_at_the_second_stage_unless_told():
    config = Config(oversampling=1, stages=((1, 2, 0), (5, 4, 1), (7, 8, 1)), iterations=8)
    assert [oversampled_update(config, i) for i in (1, 4, 5, 8)] == [False, False, True, True]
    early = dataclasses.replace(config, oversampling_start=2)
    assert [oversampled_update(early, i) for i in (1, 2, 5, 8)] == [False, True, True, True]
    assert not any(oversampled_update(Config(oversampling=0), i) for i in (1, 200))
    # The defaults: a schedule scaled to the run, adaptive passes from its second stage, tomogram batches.
    assert Config().stages == ((1, 4, 1), (61, 8, 2), (111, 16, 2), (161, 32, 2))
    short = Config(iterations=50)
    assert short.stages == ((1, 4, 1), (16, 8, 2), (29, 16, 2), (41, 32, 2))
    assert [oversampled_update(short, i) for i in (15, 16, 50)] == [False, True, True]
    assert Config(iterations=3).stages == ((1, 4, 1), (2, 8, 2), (3, 16, 2)) and Config().tomogram_batches


def test_child_pose_grid_carries_the_samplers_euler_rows():
    data = tomo_data()
    config = dataclasses.replace(tiny_config(2), stages=((1, 2, 1),))
    coarse, fine = pose_grid(data, config, 1), pose_grid(data, config, 1, children=True)
    assert fine.rotations.shape[0] == 8 * coarse.rotations.shape[0]
    assert fine.eulers.shape == (fine.rotations.shape[0], 3)
    assert fine.translations.shape[0] == 8 * coarse.translations.shape[0]  # 3D shifts: 8 children each
    # Every child's Euler row reproduces its matrix: the metadata and the grid agree.
    from relax.healpix_sampling import euler_angles_to_matrix

    rebuilt = np.asarray(euler_angles_to_matrix(fine.eulers[:16]), np.float32)
    assert_matches(rebuilt, fine.rotations[:16], rtol=1e-5)


def test_tomogram_batches_draw_whole_groups_and_split_by_parity():
    data = tomo_data()  # four particles in one tilt group
    rng = np.random.default_rng(3)
    selected, halves = iteration_loop.select_tomogram_batch(rng, data, 4)
    np.testing.assert_array_equal(np.sort(selected), np.arange(4))
    assert all(np.all(ids % 2 == half) for half, ids in enumerate(halves))
    selected, _ = iteration_loop.select_tomogram_batch(np.random.default_rng(4), data, 3)
    assert selected.size == 3 and len(set(selected.tolist())) == 3


def test_controller_runs_oversampled_updates_resumes_and_exports_child_grid_poses(tmp_path):
    data = tomo_data()
    config = dataclasses.replace(tiny_config(2), oversampling=1, oversampling_start=1, max_significant=6,
                                 tomogram_batches=True)
    identity = {"case": "oversampled-tomo"}
    direct = iteration_loop.run(data, config, tmp_path / "direct", identity, 6.)
    records = [json.loads(line) for line in (tmp_path / "direct/iterations.jsonl").read_text().splitlines()]
    assert [r["engine"] for r in records] == ["mixture_full_row_adaptive_oversampling"] * 2
    assert [r["batch_draw"] for r in records] == ["tomogram"] * 2
    assert all(0 < r["oversampling"]["class0"]["samples_median"] <= 6 for r in records)
    stopped = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6., stop_after=1)
    assert stopped.iteration == 1
    resumed = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6.,
                                 resume=tmp_path / "resumed/checkpoint_0001.npz")
    assert_matches(resumed.theta, direct.theta)
    assert_matches(resumed.class_prior, direct.class_prior)
    coarse, fine = pose_grid(data, config, 2), pose_grid(data, config, 2, children=True)
    with np.load(tmp_path / "direct/assignments.npz") as a:
        np.testing.assert_array_equal(a["particle_ids"], np.arange(4))
        assert a["euler_grid_deg"].shape[0] == fine.rotations.shape[0] == 8 * coarse.rotations.shape[0]
        assert np.all(a["rotation_index_per_class"] < fine.rotations.shape[0])
        assert_matches(a["rotations"], fine.rotations[a["rotation_index_per_class"][np.arange(4), a["class_labels"]]])
    classes = json.loads((tmp_path / "direct/classes.json").read_text())
    assert classes["pose_grid"].startswith("order-1 children")
    # The approximation is not the model's identity: the full-grid configuration resumes the same checkpoint.
    dense = dataclasses.replace(config, oversampling=0, oversampling_start=None, tomogram_batches=False)
    continued = iteration_loop.run(data, dense, tmp_path / "dense", identity, 6.,
                                   resume=tmp_path / "resumed/checkpoint_0001.npz")
    assert continued.iteration == 2
