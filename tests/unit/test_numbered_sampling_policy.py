"""Expectation sampling decisions use the incoming state and physical iteration."""

import logging

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.run_options import stand_in

from relax import sampling
from relax.helpers.convergence import RefinementState
from relax.helpers.resolution import ImageGeometry
from relax.refinement import image_size_plans, numbered_transitions, trial_grids

pytestmark = pytest.mark.unit
LOG = logging.getLogger(__name__)


def _optics(image_sizes, pixel_sizes, *, multi_shape_halves=False):
    return image_size_plans.RunOptics(
        image_geometry=ImageGeometry(image_shape=(128, 128), pixel_size_angstrom=1.0), model_pixel_size=1.0,
        optics_image_sizes=image_sizes, optics_pixel_sizes=pixel_sizes, multi_shape_halves=multi_shape_halves,
    )


def _options(parity, init_relion_iteration):
    return stand_in.options(
        parity=parity, schedule=stand_in.schedule(init_relion_iteration=init_relion_iteration),
    )


def _stalled_state():
    return RefinementState(
        healpix_order=2, max_healpix_order=5,
        acc_rot=1.0, acc_trans=0.5,
        smallest_changes_optimal_orientations=1.0,
        nr_iter_wo_resol_gain=2, nr_iter_wo_large_hidden_variable_changes=2,
    )


@pytest.mark.parametrize("previous,native,advance", [
    (False, True, False),
    (True, True, True),
    (True, False, False),
])
def test_k1_native_sampling_uses_completed_iteration_counters(previous, native, advance):
    state = _stalled_state()
    result = numbered_transitions.advance_expectation_sampling(
        state, stand_in.adaptive(), iteration=0,
        may_advance_natively=previous and numbered_transitions.uses_native_auto_refine(native_sampling_boundary=native, n_classes=1),
        log=LOG,
    )
    assert result.healpix_order == (3 if advance else 2)
    if not advance:
        assert result is state


def test_class3d_sampling_ignores_completed_iteration_counters():
    state = _stalled_state()
    may_advance = numbered_transitions.uses_native_auto_refine(native_sampling_boundary=True, n_classes=4)
    result = numbered_transitions.advance_expectation_sampling(
        state, stand_in.adaptive(), iteration=0, may_advance_natively=may_advance, log=LOG,
    )
    assert result.healpix_order == 2
    assert result is state


@pytest.mark.parametrize("previous,native", [(False, True), (True, True)])
def test_explicit_schedule_precedes_k1_native_advance(previous, native, monkeypatch):
    state = RefinementState(healpix_order=2, max_healpix_order=5)
    monkeypatch.setattr(numbered_transitions, "update_angular_sampling", lambda *_: pytest.fail("native transition"))
    result = numbered_transitions.advance_expectation_sampling(
        state, stand_in.adaptive(relion_healpix_orders=(2, 3)), iteration=1,
        may_advance_natively=previous and native, log=LOG,
    )
    assert result.healpix_order == 3


def test_explicit_schedule_sets_the_class3d_order(monkeypatch):
    state = RefinementState(healpix_order=2, max_healpix_order=5)
    monkeypatch.setattr(numbered_transitions, "update_angular_sampling", lambda *_: pytest.fail("native transition"))
    result = numbered_transitions.advance_expectation_sampling(
        state, stand_in.adaptive(relion_healpix_orders=(2, 3)), iteration=1, may_advance_natively=False, log=LOG,
    )
    assert result.healpix_order == 3


@pytest.mark.parametrize("seed", [None, 17])
def test_native_perturbation_preserves_physical_iteration_and_rng(seed):
    parity = stand_in.parity(perturb_factor=0.5, perturb_seed=seed)
    rng = np.random.default_rng(23)
    reference_rng = np.random.default_rng(23)
    current = expected = 0.125
    for iteration in range(3):
        expected, _ = sampling.advance_relion_perturbation_for_iteration(
            expected, perturb_factor=0.5, perturb_seed=seed,
            relion_iteration=11 + iteration, rng=reference_rng,
        )
        current = trial_grids.resolve_numbered_perturbation(
            current, _options(parity, 10), iteration=iteration, rng=rng, log=LOG,
        )
        assert_matches(current, expected)
    assert_matches(rng.random(), reference_rng.random())


@pytest.mark.parametrize("sealed", [False, True])
def test_replayed_perturbation_does_not_advance_native_rng(sealed, tmp_path):
    """A sealed sampling state's perturbation and a RELION STAR's, the sampling record the replay source's
    numbered_state installed, are read, not drawn: the native RNG does not move."""
    from relax.parity.relion_replay_source import RelionReplay, RelionReplaySource

    rng = np.random.default_rng(23)
    reference_rng = np.random.default_rng(23)
    options = _options(stand_in.parity(perturb_factor=0.5), 10)
    meta = {"sealed_v3": sealed, "random_perturbation": 0.25, "perturbation_factor": 0.5, "healpix_order": 3}

    source = RelionReplaySource(RelionReplay(perturb_replay_relion_dir=str(tmp_path)), options)
    source._sampling_meta = meta  # as numbered_state installs it
    result = source.random_perturbation(0)
    assert_matches(result, 0.25)
    assert_matches(rng.random(), reference_rng.random())


def test_a_source_without_a_replayed_sampling_leaves_the_perturbation_and_grids_to_the_run(tmp_path):
    """None from the native source and from a replay source with no sampling record or sealed state: the
    controller then computes the perturbation and builds the first coarse grids itself."""
    from relax.parity.relion_replay_source import RelionReplay, RelionReplaySource
    from relax.refinement.ports import InputSource

    options = _options(stand_in.parity(perturb_factor=0.5), 10)
    replay = RelionReplaySource(RelionReplay(perturb_replay_relion_dir=str(tmp_path)), options)
    for source in (InputSource(), replay):
        assert source.random_perturbation(0) is None
        assert source.initial_coarse_grids(initialized_healpix_order=2, voxel_size=1.0, symmetry="C1") is None


def test_replay_restart_uses_physical_iteration(tmp_path):
    from relax.parity.relion_replay_source import RelionReplay, RelionReplaySource

    (tmp_path / "run_it012_optimiser.star").write_text("data_\n\n_rlnRandomSeed 1778628798\n")
    options = _options(stand_in.parity(perturb_factor=0.5), 11)
    replay = RelionReplay(
        perturb_replay_restart_state_iterations=(11,),
        perturb_replay_relion_dir=str(tmp_path),
    )
    source = RelionReplaySource(replay, options)
    source._sampling_meta = {"random_perturbation": -0.06873, "perturbation_factor": 0.5, "healpix_order": 3}
    result = source.random_perturbation(0)
    assert_matches(result, -0.06873074173927307)


def test_disabled_perturbation_preserves_value_without_rng_consumption(monkeypatch):
    monkeypatch.setattr(sampling, "advance_relion_perturbation_for_iteration", lambda *_args, **_kwargs: pytest.fail("RNG advance"))
    result = trial_grids.resolve_numbered_perturbation(
        0.25, _options(stand_in.parity(perturb_factor=0.0), 10), iteration=2, rng=None, log=LOG,
    )
    assert_matches(result, 0.25)


@pytest.mark.parametrize("model_size,optics_boxes,optics_pixels,image_size,model_window,image_window", [
    (128, None, None, 128, None, None),
    (64, None, None, 64, 64, 64),
    (128, [128], [0.5], 64, 128, 64),
    (64, [128], [0.5], 32, 64, 32),
    (64, [128], [2.0], 128, 64, None),
])
def test_particle_remap_retains_independent_model_cutoff(
    model_size, optics_boxes, optics_pixels, image_size, model_window, image_window,
):
    result = image_size_plans.plan_expectation_windows(model_size, _optics(optics_boxes, optics_pixels), log=LOG)
    assert result.image_current_size == image_size
    assert result.model_window_size == model_window
    assert result.image_window_size == image_window


def test_single_shape_cannot_hide_different_optics_window_sizes():
    with pytest.raises(NotImplementedError, match="one remapped image current size"):
        image_size_plans.plan_expectation_windows(64, _optics([128, 128], [1.0, 2.0]), log=LOG)


def test_shape_classes_are_not_remapped_by_the_single_shape_plan():
    """Each shape class remaps its own support: the run-level plan keeps the model size."""
    result = image_size_plans.plan_expectation_windows(
        64, _optics([128], [0.5], multi_shape_halves=True), log=LOG,
    )
    assert result.image_current_size == 64
