"""Numbered expectation owns priors, seeding and interpretation of engine poses."""

import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.dense.score_outputs import HalfScoreResult, PerHalfOutputs
from relax.helpers.orientation_priors import HalfDirectionLogPriors
from relax.helpers.resolution import ImageGeometry
from relax.refinement import expectation
from relax.refinement.expectation_batches import BatchPlanner, HalfBatchPlan
from relax.refinement.half_inputs import HalfSet
from relax.refinement.half_scoring import (
    DenseSamplingSpec,
    DenseVariantPolicy,
    HalfScoringData,
    LocalDiagnosticPolicy,
)
from relax.refinement.local_sampling import LocalSampling, LocalSearchSettings
from relax.refinement.ports import RunObserver
from relax.refinement.refinement_options import RefinementBatching, RefinementOptions, RefinementSchedule
from relax.refinement.tomo_half import TomoSampling
from relax.sampling import TrialGrid

pytestmark = pytest.mark.unit


def numbered_inputs(*, local=False, adaptive=False, n_classes=1, n_units=2):
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
    eulers = np.arange(12, dtype=np.float64).reshape(4, 3)
    translations = np.array([[0, 0], [1, -1]], dtype=np.float32)
    grid = TrialGrid(rotations, eulers, rotations, translations)
    dataset = SimpleNamespace(
        n_units=n_units, n_images=n_units, image_shape=(4, 4), volume_shape=(4, 4, 4),
        _index_layout=SimpleNamespace(original_image_indices_for_local=lambda indices: np.asarray(indices) + 2),
    )
    particles = HalfSet(index=0, dataset=dataset, translations=None,
                        rotation_eulers=np.zeros((n_units, 3), dtype=np.float64))
    half = HalfScoringData(
        particles=particles, reference=np.ones(64, dtype=np.complex64),
        mean_variance=np.ones(64, dtype=np.float32), noise_variance=np.ones(16, dtype=np.float32),
        noise_radial=np.ones(3, dtype=np.float32),
    )
    if local:
        sampling = LocalSampling(
            search=LocalSearchSettings(healpix_order=0, oversampling_order=int(adaptive), sigma_rot=.2, sigma_psi=.3),
            rotations=rotations, translations=translations, base_translations=translations,
            image_window_size=4, coarse_image_window_size=2, perturbation=.125, angular_step_deg=None,
            coarse_angular_step_deg=15.,
        )
    else:
        sampling = DenseSamplingSpec(
            effective_rotations=rotations, current_translations=translations, base_translations=translations,
            current_healpix_order=0, oversampling_order=int(adaptive), translation_step=1.,
            random_perturbation=.125, cs_for_engine=4,
            coarse_angular_step_deg=15.,
        )
    variant = DenseVariantPolicy(
        firstiter_score_mode_this_iter='gaussian', firstiter_winner_take_all_this_iter=False,
        k_class_enabled=n_classes > 1, relion_firstiter_cc_this_iter=False,
        firstiter_coarse_current_size=2 if adaptive else None,
        firstiter_fine_current_size=4 if adaptive else None,
        firstiter_log_label='' if adaptive else '(non-adaptive site) ',
        firstiter_updates_em_kwargs_ibs=adaptive,
    )
    kwargs = dict(
        sampling=sampling, tomo_sampling=None,
        direction_priors=HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None),
        class_log_priors=None, sigma_offset_angstrom=2.,
        batch_planner=BatchPlanner(requested=RefinementBatching(image_batch_size=2, rotation_block_size=2),
                                   image_shape=(4, 4), volume_shape=(4, 4, 4), n_classes=n_classes,
                                   log=logging.getLogger(__name__)),
        image_geometry=ImageGeometry(image_shape=(4, 4), pixel_size_angstrom=1.25),
        padded_volume_shape=(8, 8, 8),
        use_adaptive=adaptive, multi_shape_halves=False, variant=variant,
        options=RefinementOptions(schedule=RefinementSchedule(particle_diameter_ang=3.)),
        local_diagnostics=LocalDiagnosticPolicy(iteration=0, debug_iteration=1,
                                                collect_local_search_profile=False, diagnostic_score_only=False,
                                                local_profile_history=[]) if local else None,
        replay_prior_translations=None, initial_class_assignments=None, single_class_iteration=False,
        scoring_dtype=np.float32, relion_translation_angle_scale=1.,
        iteration=0, numbered_relion_iteration=1, observer=RunObserver(),
    )
    phase = expectation.NumberedExpectation(
        grid=grid, sampling=kwargs.pop('sampling'), variant=kwargs.pop('variant'),
        use_adaptive=kwargs.pop('use_adaptive'), local_diagnostics=kwargs.pop('local_diagnostics'),
    )
    return half, phase, kwargs


def score_flat(half, phase, *, direction_priors, sigma_offset_angstrom, **kwargs):
    """``score_numbered_half`` on this file's flat inputs: the per-half ones go in a ``NumberedHalfInputs``."""
    inputs = expectation.NumberedHalfInputs(half, direction_priors, sigma_offset_angstrom)
    return expectation.score_numbered_half(inputs, phase, **kwargs)


def engine_result(n_units=2):
    return HalfScoreResult(
        ha=np.arange(n_units, dtype=np.int32), Ft_y=np.ones(64, dtype=np.complex64),
        Ft_ctf=np.ones(64, dtype=np.complex64), noise_stats=object(),
        em_stats=SimpleNamespace(max_posterior_per_image=np.ones(n_units, dtype=np.float32),
                                 rotation_posterior_sums=np.ones(4, dtype=np.float32)),
    )


def stub_planning(monkeypatch):
    monkeypatch.setattr(expectation, '_bpref_device_signature_active_for_numbered_half', lambda **kw: False)
    monkeypatch.setattr(expectation, 'prepare_half_batches', lambda *args, **kw: HalfBatchPlan(
        safe_batch_sizes=object(), significance_safe_batch_sizes=object(), fine_image_batch_size=2,
        fine_rotation_block_size=2, coarse_image_batch_size=1, coarse_rotation_block_size=1,
        class_overrides=None,
    ))
    monkeypatch.setattr(expectation.optics_shapes, 'prepare_optics', lambda *args, **kw: object())


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('coarse_step', [None, np.float64(17.25)])
def test_numbered_sizing_and_optics_consume_the_sampling_and_half_operands(
    monkeypatch, local, coarse_step,
):
    stub_planning(monkeypatch)
    half, phase, kwargs = numbered_inputs(local=local, adaptive=True)
    radial = np.array([5., 7., 11.], dtype=np.float64)
    half = replace(half, noise_radial=radial)
    phase = replace(phase, sampling=replace(phase.sampling, coarse_angular_step_deg=coarse_step))
    diameter = np.float64(3.75)
    kwargs['options'] = replace(kwargs['options'], schedule=RefinementSchedule(particle_diameter_ang=diameter))
    kwargs['multi_shape_halves'] = True
    captured = {}
    planning = expectation.prepare_half_batches

    def plan(*args, **inputs):
        captured['sizing'] = inputs['coarse_sizing']
        return planning(*args, **inputs)

    def optics(*args, **inputs):
        captured['optics'] = inputs
        return object()

    monkeypatch.setattr(expectation, 'prepare_half_batches', plan)
    monkeypatch.setattr(expectation.optics_shapes, 'prepare_optics', optics)
    monkeypatch.setattr(expectation, '_score_half_dense_in_bpref_scope', lambda *args: engine_result())
    monkeypatch.setattr(expectation, '_score_half_local_in_bpref_scope', lambda **inputs: engine_result())
    score_flat(half, phase, **kwargs)
    assert captured['sizing'][0] is coarse_step
    assert captured['sizing'][1] is diameter
    assert captured['optics']['coarse_step_deg'] is coarse_step
    assert captured['optics']['particle_diameter_ang'] is diameter
    assert captured['optics']['noise_radial'] is radial
    assert captured['optics']['with_log_prior'] is not local
    assert radial.dtype == np.float64


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('preserve_order', [False, True])
def test_preserved_particle_order_selects_production_arithmetic_in_replays_too(monkeypatch, local, preserve_order):
    """One K=1 arithmetic: a run that preserves RELION's particle order (fresh, or a replay of RELION's state)
    scores with the source-faithful spectrum normalisation; a run that does not, without it."""
    stub_planning(monkeypatch)
    half, phase, kwargs = numbered_inputs(local=local)
    kwargs['options'] = replace(
        kwargs['options'], parity=replace(kwargs['options'].parity, preserve_bpref_particle_order=preserve_order),
    )
    captured = {}
    planning = expectation.prepare_half_batches

    def plan(*args, **inputs):
        captured['batches'] = inputs['source_faithful_spectrum_norm']
        return planning(*args, **inputs)

    def dense(half, sampling, priors, batching, variant, execution, optics):
        captured['execution'] = execution.source_faithful_spectrum_norm
        return engine_result()

    def local_scorer(**inputs):
        captured['execution'] = inputs['execution'].source_faithful_spectrum_norm
        return engine_result()

    monkeypatch.setattr(expectation, 'prepare_half_batches', plan)
    monkeypatch.setattr(expectation, '_score_half_dense_in_bpref_scope', dense)
    monkeypatch.setattr(expectation, '_score_half_local_in_bpref_scope', local_scorer)
    score_flat(half, phase, **kwargs)
    assert captured == {'batches': preserve_order, 'execution': preserve_order}


def test_actual_numbered_half_binding_does_not_read_unused_radial_noise(monkeypatch):
    """A single-particle half scores with its own radial noise curve of the iteration's noise model.

    The subtomogram and empty-half routes (which bind no curve) have no CPU run; ``score_numbered_half``'s
    own tests cover what they do without one.
    """
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement import iteration_loop

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, 'initialize_noise_model', 'noise')
    trace.wrap(iteration_loop, 'HalfScoringData', 'data')
    run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False)
    (noise,) = trace.calls('noise')
    data = trace.calls('data')
    assert len(data) == 2
    for k, call in enumerate(data):
        assert call.kwargs['noise_radial'] is noise.result.radial_per_half[k]


@pytest.mark.parametrize('for_convergence', [False, True], ids=['class3d', 'auto-refine'])
@pytest.mark.parametrize('counts', [None, np.zeros(0, dtype=np.int32), np.array([7, 3], dtype=np.int64)])
def test_significance_statistics_preserve_absent_empty_and_half_arrival_order(counts, for_convergence):
    stats = expectation.SignificanceStatistics()
    first_counts = np.array([2, 5], dtype=np.int32)
    stats.record(1, first_counts, for_convergence=for_convergence)
    stats.record(0, counts, for_convergence=for_convergence)
    assert stats.per_half[1] is first_counts
    assert stats.recorded is None and stats.convergence is None
    stats.combine()
    expected = [2, 5, 7, 3] if counts is not None and counts.size else [2, 5]
    np.testing.assert_array_equal(stats.recorded, expected)
    assert stats.recorded.dtype == np.int32
    if counts is None:
        assert stats.per_half[0] is None
    else:
        assert stats.per_half[0].dtype == np.int32
        assert stats.per_half[0].size == counts.size
    if for_convergence:
        np.testing.assert_array_equal(stats.convergence, expected)
    else:
        assert stats.convergence is None


def test_significance_statistics_distinguish_unrecorded_from_recorded_empty_counts():
    absent = expectation.SignificanceStatistics()
    absent.combine()
    assert absent.recorded is None and absent.convergence is None
    assert absent.per_half == [None, None]
    empty = expectation.SignificanceStatistics()
    empty.record(0, np.zeros(0, dtype=np.int32), for_convergence=True)
    empty.combine()
    assert empty.recorded.shape == empty.convergence.shape == (0,)


@pytest.mark.parametrize('n_units', [0, 2])
@pytest.mark.parametrize('n_classes', [1, 4])
@pytest.mark.parametrize('has_index_layout', [False, True])
def test_recording_uses_published_half_identity_and_preserves_capture_order(
    monkeypatch, n_units, n_classes, has_index_layout,
):
    half, _, _ = numbered_inputs(n_units=n_units, n_classes=n_classes)
    particle_half = replace(half.particles, index=1)
    if not has_index_layout:
        del particle_half.dataset._index_layout
    result = engine_result(n_units)
    result.significant_counts = np.array([3, 6], dtype=np.int64) if n_units else np.zeros(0, dtype=np.int32)
    result.coarse_ha = result.ha
    outputs = PerHalfOutputs()
    outputs.update_from(1, result, dtype=np.float32)
    published_noise = object()
    outputs.noise_stats[1] = published_noise
    significance = expectation.SignificanceStatistics()
    profile_history = []
    events = []
    captures = []

    def profile(history, score, **metadata):
        assert history is profile_history and score is result
        assert significance.per_half == [None, None]
        history.append(metadata)
        events.append('profile')

    def panel(**metadata):
        assert metadata == dict(iteration_index=4, half_index=1)
        assert significance.per_half[1].dtype == np.int32
        np.testing.assert_array_equal(significance.per_half[1], [3, 6])
        events.append('panel')

    def capture(**payload):
        captures.append(payload)
        events.append('capture')

    monkeypatch.setattr(expectation, '_record_score_profile', profile)
    monkeypatch.setattr(expectation.bpref_diagnostics, 'flush_selected_bpref_device_panel', panel)
    monkeypatch.setattr(expectation._parity_dump, 'collect_e_step', capture)
    monkeypatch.setattr(expectation._parity_dump, 'is_active', lambda: True)
    expectation.record_numbered_half(
        result, particle_half, outputs, significance,
        profile_history=profile_history, iteration=4, image_window_size=None,
        healpix_order=2, k_class_enabled=n_classes > 1,
    )
    assert events == (['capture'] if n_units == 0 else ['profile', 'panel', 'capture'])
    assert len(captures) == 1
    payload = captures[0]
    assert payload['half'] == 1
    assert payload['Ft_y'] is result.Ft_y and payload['Ft_ctf'] is result.Ft_ctf
    assert payload['hard_assignment'] is result.ha
    assert payload['noise_stats'] is (result.noise_stats if n_units == 0 else published_noise)
    if n_units == 0:
        assert payload['original_image_indices'].shape == (0,)
        assert significance.per_half == [None, None]
    else:
        if has_index_layout:
            np.testing.assert_array_equal(payload['original_image_indices'], [2, 3])
            assert payload['original_image_indices'].dtype == np.int64
        else:
            assert payload['original_image_indices'] is None
        assert profile_history == [dict(
            phase='iteration', iteration=4, relion_iteration=5, half_index=1,
            current_size=None, healpix_order=2, k_class_enabled=n_classes > 1,
        )]
    significance.combine()
    if n_units:
        np.testing.assert_array_equal(significance.recorded, [3, 6])
    else:
        assert significance.recorded is None
    assert (significance.convergence is None) == (n_units == 0 or n_classes > 1)


@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('explicit_matrices', [False, True])
@pytest.mark.parametrize('explicit_eulers', [False, True])
def test_dense_pose_grid_preserves_canonical_metadata_and_engine_overrides(
    monkeypatch, adaptive, explicit_matrices, explicit_eulers,
):
    stub_planning(monkeypatch)
    half, phase, kwargs = numbered_inputs(adaptive=adaptive)
    result = engine_result()
    matrices = np.repeat((-np.eye(3, dtype=np.float32))[None], 2, axis=0)
    eulers = np.ones((2, 3), dtype=np.float32)
    result.pose_rotations = matrices if explicit_matrices else None
    result.pose_rotation_eulers = eulers if explicit_eulers else None
    monkeypatch.setattr(expectation, '_score_half_dense_in_bpref_scope', lambda *args: result)
    actual = score_flat(half, phase, **kwargs)
    assert actual.pose_rotations is (matrices if explicit_matrices else phase.grid.rotations)
    expected_eulers = eulers if explicit_eulers else None if adaptive and explicit_matrices else phase.grid.rotation_eulers
    assert actual.pose_rotation_eulers is expected_eulers
    assert actual.coarse_ha is actual.ha


@pytest.mark.parametrize('n_classes', [1, 4])
def test_cold_dense_translation_prior_uses_k1_gaussian_and_k4_flat_policy(monkeypatch, n_classes):
    stub_planning(monkeypatch)
    half, phase, kwargs = numbered_inputs(n_classes=n_classes)
    captured = {}

    def score(half, sampling, priors, batching, variant, execution, optics):
        captured['prior'] = priors.translation_log_prior
        return engine_result()

    monkeypatch.setattr(expectation, '_score_half_dense_in_bpref_scope', score)
    actual = score_flat(half, phase, **kwargs)
    if n_classes == 1:
        # RELION ACC multiplies the angstrom-grid squared offset by another pixel_size^2.
        assert_matches(captured['prior'], [0., -(2 * 1.25**4) / (2 * 2.**2)])
    else:
        assert_matches(captured['prior'], np.zeros(2, dtype=np.float32))
    assert actual.translation_search_base is None


def test_local_result_keeps_explicit_coarse_assignments_and_rounded_offset_frame(monkeypatch):
    stub_planning(monkeypatch)
    half, phase, kwargs = numbered_inputs(local=True)
    half.particles.translations = np.array([[.6, -1.6], [1.2, .3]], dtype=np.float64)
    result = engine_result()
    result.coarse_ha = np.array([7, 9], dtype=np.int32)
    captured = {}

    def score(**owners):
        captured.update(owners)
        return result

    monkeypatch.setattr(expectation, '_score_half_local_in_bpref_scope', score)
    actual = score_flat(half, phase, **kwargs)
    assert_matches(actual.translation_search_base, [[1, -2], [1, 0]])
    assert actual.coarse_ha is result.coarse_ha
    assert captured['priors'].translation_search_base is actual.translation_search_base
    assert captured['half'].mean_variance is None
    assert captured['sampling'] is phase.sampling


@pytest.mark.parametrize('local', [False, True])
def test_tomography_seeding_uses_original_unit_rows_without_spa_optics(monkeypatch, local):
    stub_planning(monkeypatch)
    half, phase, kwargs = numbered_inputs(local=local, adaptive=True, n_classes=4)
    kwargs['tomo_sampling'] = TomoSampling(healpix_order=0, oversampling_order=1,
        offset_range_angst=4., offset_step_angst=1., random_perturbation=.125, coarse_size=2, fine_size=4)
    kwargs['initial_class_assignments'] = np.array([3, 1, 0, 2], dtype=np.int32)
    kwargs['class_log_priors'] = np.zeros(4, dtype=np.float32)
    # The direction-prior owner supplies no prior under local search.
    class_direction_prior = None if local else np.zeros((4, 4), dtype=np.float32)
    kwargs['direction_priors'] = HalfDirectionLogPriors(
        rotation_log_prior=None, class_rotation_log_prior=class_direction_prior)
    captured = {}

    def score(dataset, **inputs):
        captured.update(inputs)
        return engine_result()

    def reject(*args, **kw):
        raise AssertionError('tomography must not prepare SPA image priors or optics')

    monkeypatch.setattr(expectation, '_score_tomo_half_in_loop', score)
    monkeypatch.setattr(expectation.optics_shapes, 'prepare_optics', reject)
    monkeypatch.setattr(expectation, 'relion_half_translation_prior_inputs', reject)
    score_flat(half, phase, **kwargs)
    assert_matches(captured['unit_seed_classes'], [0, 2])
    assert captured['sampling'] is kwargs['tomo_sampling']
    assert (captured['local_search'] is not None) == local
    assert captured['class_rotation_log_prior'] is class_direction_prior


@pytest.mark.parametrize('n_classes', [1, 4])
def test_empty_full_layout_does_not_read_unused_volume_geometry(monkeypatch, n_classes):
    stub_planning(monkeypatch)
    half, phase, kwargs = numbered_inputs(n_classes=n_classes, n_units=0)

    class Empty:
        n_units = 0

        @property
        def volume_shape(self):
            raise AssertionError('full/K4 empty layout does not need volume geometry')

    half = replace(half, particles=replace(half.particles, dataset=Empty()))
    actual = score_flat(half, phase, **kwargs)
    assert actual.ha.shape == (0,)
    assert actual.best_pose_translations.shape == (0, 2)
    assert actual.coarse_ha is actual.ha
    if n_classes == 4:
        assert actual.Ft_y is None and actual.Ft_ctf is None
    else:
        assert actual.Ft_y.shape == (8**3,)


@pytest.mark.parametrize('dump_active', [False, True])
@pytest.mark.parametrize('with_layout', [False, True])
def test_recorded_half_hands_the_parity_dump_its_input_rows_only_when_it_dumps(monkeypatch, dump_active, with_layout):
    """The parity dump receives each image's row in the input stack (None for a dataset without an index
    layout, or when no dump is active); a failing layout is an error, not a missing row list (rule 12)."""
    half, _phase, _kwargs = numbered_inputs()
    if not with_layout:
        half = replace(half, particles=replace(half.particles, dataset=SimpleNamespace(
            n_units=2, n_images=2, image_shape=(4, 4), volume_shape=(4, 4, 4))))
    collected = {}
    monkeypatch.setattr(expectation._parity_dump, 'is_active', lambda: dump_active)
    monkeypatch.setattr(expectation._parity_dump, 'collect_e_step', lambda **kw: collected.update(kw))
    monkeypatch.setattr(expectation.bpref_diagnostics, 'flush_selected_bpref_device_panel', lambda **kw: None)
    per_half = PerHalfOutputs()
    expectation.record_numbered_half(
        engine_result(), half.particles, per_half, expectation.SignificanceStatistics(), profile_history=[],
        iteration=0, image_window_size=4, healpix_order=0, k_class_enabled=False,
    )
    if dump_active and with_layout:
        np.testing.assert_array_equal(collected['original_image_indices'], [2, 3])
        assert collected['original_image_indices'].dtype == np.int64
    else:
        assert collected['original_image_indices'] is None

    def broken(indices):
        raise IndexError('layout out of range')

    if dump_active and with_layout:
        half.particles.dataset._index_layout = SimpleNamespace(original_image_indices_for_local=broken)
        with pytest.raises(IndexError, match='layout out of range'):
            expectation.record_numbered_half(
                engine_result(), half.particles, per_half, expectation.SignificanceStatistics(), profile_history=[],
                iteration=0, image_window_size=4, healpix_order=0, k_class_enabled=False,
            )
