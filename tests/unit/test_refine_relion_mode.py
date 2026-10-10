"""Smoke tests for refine_single_volume's RELION-only path.

Verifies:
1. RELION mode runs without error on a tiny dataset (4 images, 8px, 2 iters)
2. Returns the expected dict keys (including RELION-specific ones)
3. Convergence state is a RefinementState instance
4. data_vs_prior_trajectory and ave_Pmax_trajectory are populated
"""

import inspect
import os
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches, matches
from helpers.pass1_programs import clear_pass1_programs
from helpers.reconstruction_settings import reconstruction_settings
from helpers.tiny_refinement import unconverged_accuracy

import relax.refinement.noise_updates as noise_updates
from relax.helpers import oversampling as oversampling_grids
from relax.helpers.orientation_priors import (
    DirectionPrior,
    _combined_class_direction_prior_from_halves,
)
from relax.helpers.resolution import ImageGeometry
from relax.reconstruction import volume_solver
from relax.refinement import map_postprocess
from relax.refinement.half_inputs import HalfPair, initialize_halfsets
from relax.refinement.noise_updates import NoiseModel
from relax.refinement.ports import InputSource, RunObserver
from relax.refinement.refinement_options import OpticsGeometry, ReconstructionPrograms
from relax.refinement.startup_references import StartupHandoff

pytest.importorskip("jax")
import healpy as hp
import jax
import jax.numpy as jnp
import recovar.core.fourier_transform_utils as ftu
from helpers.em_arrays import _hermitian_volume, _make_rotations
from helpers.fake_adaptive_engine import adaptive_result, fake_adaptive_engine, install_fake_adaptive_engine
from helpers.refinement_results import refinement_result
from helpers.refinement_specs import (
    local_half_owners,
    local_iteration_keywords,
    local_iteration_owners,
)
from helpers.run_options import stand_in
from recovar import utils as recovar_utils

import relax.helpers.expected_accuracy as expected_accuracy_module
import relax.helpers.orientation_priors as orientation_priors_module
import relax.local.local_layout as local_layout_module
import relax.parity.relion_replay as relion_replay_module
import relax.refinement.convergence as convergence_policy
import relax.refinement.expectation as expectation_module
import relax.refinement.iteration_loop as iteration_loop_module
import relax.refinement.iteration_planning as iteration_planning_module
import relax.refinement.projector_preparation as projector_preparation
import relax.sampling as sampling_module
from relax.classification.k_class_results import (
    KClassEMResult,
    _resolve_class_mstep_posterior_sums,
    _sum_noise_stats,
)
from relax.dense import score_outputs, scoring_policy
from relax.diagnostics.observers import IntermediatesObserver
from relax.healpix_sampling import euler_angles_to_matrix
from relax.helpers import dtype_policy as dtype_policy_module
from relax.helpers import resolution as resolution_helpers
from relax.helpers.convergence import RefinementState, _relion_optimizer_average_pmax, healpix_angular_step
from relax.helpers.half_volume_mstep import (
    relion_backprojector_volume_shape,
)
from relax.helpers.image_shifts import apply_relion_integer_pre_shifts, integer_pre_shifts_or_none
from relax.helpers.orientation_priors import (
    collapse_rotation_posterior_to_direction_prior,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    normalize_direction_prior_per_half,
    relion_sigma_offset_prior_center,
    relion_translation_prior_center,
    relion_translation_search_base,
)
from relax.helpers.resolution import (
    bootstrap_current_size_from_ini_high_relion,
    bootstrap_current_size_relion,
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    relion_expectation_coarse_size_order,
    relion_local_pass1_current_size,
    relion_optics_image_current_sizes,
    shell_index_to_resolution_angstrom,
)
from relax.helpers.types import LocalEMResult, NoiseStats, RelionStats
from relax.local.local_layout import (
    EXACT_LOCAL_BUCKET_RADIX_ENV,
    LocalHypothesisLayout,
    bucket_local_hypothesis_layout,
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
    build_pass2_hypothesis_layout,
    exact_bucket_rotation_size,
    selected_rotation_matrices,
)
from relax.parity.relion_replay import _replay_control_model_iteration
from relax.parity.relion_replay_source import RelionReplay
from relax.reconstruction import regularization_relion
from relax.refinement import finalization, half_scoring, local_sampling, local_search_iteration
from relax.refinement import maximization as maximization_module
from relax.refinement import numbered_reconstruction as numbered_reconstruction_module
from relax.refinement.iteration_loop import refine_single_volume
from relax.refinement.local_search_iteration import LocalSearchResult
from relax.refinement.map_postprocess import _align_fourier_volume_sign_to_reference
from relax.refinement.noise_updates import (
    _combined_noise_stats,
    _normalize_noise_variance_per_half,
)
from relax.refinement.refinement_options import (
    FinalPassOptions,
    KClassOptions,
    LocalSearchOptions,
    StartState,
    SymmetryOptions,
)
from relax.relion import relion_ctf
from relax.sampling import (
    _get_relion_rotation_grid_eulers_float64,
    apply_relion_rotation_perturbation,
    apply_relion_rotation_perturbation_to_eulers,
    build_local_search_grid_metadata,
    get_local_rotation_grid_fast,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
    get_relion_rotation_grid,
    get_relion_rotation_grid_eulers,
    get_translation_grid,
    relion_angular_sampling_deg,
    rotation_grid_n_in_planes,
    rotation_grid_size,
)
from relax.scoring.pass1_publish import _capture_offset_free_and_absolute_float32_scores
from relax.scoring.pass1_results import Pass1Result
from relax.scoring.significance import _compute_k_class_significance_batched

pytestmark = pytest.mark.unit


def test_relion_optimizer_average_pmax_uses_kclass_mstep_mass():
    pmax = [np.asarray([0.1, 0.2]), np.asarray([0.3])]
    class_mstep_mass = [np.asarray([0.9, 0.08]), np.asarray([1.8, 0.15])]

    combined, average, denominator = _relion_optimizer_average_pmax(
        pmax,
        [np.sum(class_mstep_mass[0]), np.sum(class_mstep_mass[1])],
    )

    np.testing.assert_allclose(combined, [0.1, 0.2, 0.3])
    assert denominator == pytest.approx(0.98)
    assert average == pytest.approx(0.3 / 0.98)
    assert average != pytest.approx(float(np.mean(combined)))


def test_relion_optimizer_average_pmax_uses_k1_mstep_mass():
    _, average, denominator = _relion_optimizer_average_pmax(
        [np.asarray([0.1, 0.2]), np.asarray([0.3])],
        [1.8, 0.9],
    )

    assert denominator == 1.8
    assert average == pytest.approx(0.3 / 1.8)


def test_relion_optimizer_average_pmax_preserves_double_particle_values():
    pmax = [
        np.asarray([0.123456789012345], dtype=np.float64),
        np.asarray([0.987654321098765], dtype=np.float64),
    ]

    combined, average, denominator = _relion_optimizer_average_pmax(pmax)

    assert combined.dtype == np.float64
    assert_matches(combined, np.concatenate(pmax))
    assert average == pmax[0][0]
    assert denominator == 1.0


def test_diagnostic_float64_pass2_iteration_selector(monkeypatch):
    monkeypatch.delenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", raising=False)
    assert dtype_policy_module._diagnostic_float64_pass2_matches(4) is False
    monkeypatch.setenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "4, 7")
    assert dtype_policy_module._diagnostic_float64_pass2_matches(3) is False
    assert dtype_policy_module._diagnostic_float64_pass2_matches(4) is True
    assert dtype_policy_module._diagnostic_float64_pass2_matches(7) is True
    monkeypatch.setenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "4,bad")
    with pytest.raises(ValueError, match="comma-separated integers"):
        dtype_policy_module._diagnostic_float64_pass2_matches(4)


def test_local_search_precision_defaults_to_production_float32(monkeypatch):
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=False),
    )
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_projections=False),
    )
    monkeypatch.delenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", raising=False)

    assert scoring_policy.local_precision(
        scoring_policy.DENSE_PRECISION,
        12,
        pass_index=1,
    ) == dtype_policy_module.DensePrecisionPolicy()
    assert scoring_policy.local_precision(
        scoring_policy.DENSE_PRECISION,
        12,
        pass_index=2,
    ) == dtype_policy_module.DensePrecisionPolicy()


def test_local_search_precision_targeted_diagnostic_upgrades_only_pass2(monkeypatch):
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=False),
    )
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_projections=False),
    )
    monkeypatch.setenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "12")

    assert scoring_policy.local_precision(
        scoring_policy.DENSE_PRECISION,
        12,
        pass_index=1,
    ) == dtype_policy_module.DensePrecisionPolicy()
    assert scoring_policy.local_precision(
        scoring_policy.DENSE_PRECISION,
        12,
        pass_index=2,
    ) == dtype_policy_module.DensePrecisionPolicy(use_float64_scoring=True, use_float64_projections=True)
    assert scoring_policy.local_precision(
        scoring_policy.DENSE_PRECISION,
        11,
        pass_index=2,
    ) == dtype_policy_module.DensePrecisionPolicy()


def test_local_search_precision_global_switches_upgrade_both_passes(monkeypatch):
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=True),
    )
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_projections=True),
    )
    monkeypatch.delenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", raising=False)

    assert scoring_policy.local_precision(
        scoring_policy.DENSE_PRECISION,
        12,
        pass_index=1,
    ) == dtype_policy_module.DensePrecisionPolicy(use_float64_scoring=True, use_float64_projections=True)
    assert scoring_policy.local_precision(
        scoring_policy.DENSE_PRECISION,
        12,
        pass_index=2,
    ) == dtype_policy_module.DensePrecisionPolicy(use_float64_scoring=True, use_float64_projections=True)


def test_dense_global_scoring_dtype_tracks_global_float64_switches(monkeypatch):
    """The coarse pass-1 scorer grid has no per-iteration diagnostic path.

    Unlike ``local_precision``, this dtype selector must react
    to either global switch alone -- there is no pass-2-only diagnostic
    override to reason about at pass 1.
    """

    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=False),
    )
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_projections=False),
    )
    assert scoring_policy._dense_global_scoring_dtype() == np.float32

    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=True),
    )
    assert scoring_policy._dense_global_scoring_dtype() == np.float64

    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=False),
    )
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_projections=True),
    )
    assert scoring_policy._dense_global_scoring_dtype() == np.float64


def test_relion_scoring_rotation_grid_honors_explicit_float64_dtype(monkeypatch):
    """``relion_scoring_rotation_grid``'s working operands follow ``dtype``.

    Regression for the bug the coarse pass-1 grid shared with the already-fixed
    ``relion_adaptive_pass1_rotations``/``_relion_mstep_rotations_from_eulers``:
    an unconditional ``.astype(np.float32)`` on the scorer rotation matrices
    even when the caller runs float64 scoring, though RELION's own
    ``ACC_DOUBLE_PRECISION`` build never narrows this matrix.
    """

    source_eulers = np.asarray(
        [[13.123456789, 47.987654321, 91.234567891]], dtype=np.float64
    )
    monkeypatch.setattr(
        sampling_module,
        "_get_relion_rotation_grid_eulers_float64",
        lambda _order, *, symmetry='C1': source_eulers,
    )

    _rotation_grid_rotations_f32 = sampling_module.relion_scoring_rotation_grid(2)
    rotations_f32 = _rotation_grid_rotations_f32.rotations
    eulers_f32 = _rotation_grid_rotations_f32.rotation_eulers
    _rotation_grid_rotations_f64 = sampling_module.relion_scoring_rotation_grid(2, dtype=np.float64)
    rotations_f64 = _rotation_grid_rotations_f64.rotations
    eulers_f64 = _rotation_grid_rotations_f64.rotation_eulers

    assert rotations_f32.dtype == np.float32
    assert rotations_f64.dtype == np.float64
    np.testing.assert_allclose(rotations_f32, rotations_f64, atol=1e-6, rtol=0.0)
    assert eulers_f32.dtype == np.float32
    assert eulers_f64.dtype == np.float64
    np.testing.assert_allclose(eulers_f32, eulers_f64, atol=5e-6, rtol=0.0)
    assert_matches(eulers_f64, source_eulers)


def test_sealed_sampling_base_grids_honors_explicit_float64_dtype():
    """A restart from a schema-v3 sealed boundary gets the same dtype control."""

    sealed_state = {
        "directions_ipix": np.array([0, 1], dtype=np.int64),
        "rot_angles_deg": np.array([10.0, 190.0], dtype=np.float64),
        "tilt_angles_deg": np.array([37.0, 63.0], dtype=np.float64),
        "psi_angles_deg": np.array([0.0, 120.0, 240.0], dtype=np.float64),
        "translations_x_angstrom": np.array([0.0, 1.5], dtype=np.float64),
        "translations_y_angstrom": np.array([0.0, -1.5], dtype=np.float64),
    }

    rotations_f32, eulers_f32, translations_f32 = relion_replay_module._sealed_sampling_base_grids(
        sealed_state, voxel_size_angstrom=1.0
    )
    rotations_f64, eulers_f64, translations_f64 = relion_replay_module._sealed_sampling_base_grids(
        sealed_state, voxel_size_angstrom=1.0, dtype=np.float64
    )

    assert rotations_f32.dtype == np.float32
    assert rotations_f64.dtype == np.float64
    assert translations_f32.dtype == np.float32
    assert translations_f64.dtype == np.float64
    np.testing.assert_allclose(rotations_f32, rotations_f64, atol=1e-6, rtol=0.0)
    assert eulers_f32.dtype == np.float32
    assert eulers_f64.dtype == np.float64
    np.testing.assert_allclose(eulers_f32, eulers_f64, atol=1e-6, rtol=0.0)


def test_dense_global_prior_helpers_honor_explicit_float64_dtype():
    """The use_local=False offset/orientation log-prior helpers must not force float32.

    Regression for the same bug class already fixed for rotation matrices:
    RELION's own pdf_orientation/pdf_offset computation never narrows below
    RFLOAT/XFLOAT (both double under double-precision builds), so these
    helpers -- used only on the ``if not use_local:`` dense/global path in
    ``refine_single_volume`` -- must accept and honor an explicit
    ``dtype=np.float64`` request instead of silently staying at float32.
    """

    previous_best_translations = np.asarray([[1.6, -2.4], [0.3, 5.1]], dtype=np.float64)
    voxel_size = 3.0

    base_f32 = relion_translation_search_base(previous_best_translations)
    base_f64 = relion_translation_search_base(previous_best_translations, dtype=np.float64)
    assert base_f32.dtype == np.float32
    assert base_f64.dtype == np.float64
    np.testing.assert_allclose(base_f32, base_f64, atol=1e-6, rtol=0.0)

    center_f32 = relion_translation_prior_center(previous_best_translations, voxel_size)
    center_f64 = relion_translation_prior_center(previous_best_translations, voxel_size, dtype=np.float64)
    assert center_f32.dtype == np.float32
    assert center_f64.dtype == np.float64
    np.testing.assert_allclose(center_f32, center_f64, atol=1e-6, rtol=0.0)

    local_center_f32 = relion_translation_prior_center(previous_best_translations, voxel_size)
    local_center_f64 = relion_translation_prior_center(
        previous_best_translations, voxel_size, dtype=np.float64
    )
    assert local_center_f32.dtype == np.float32
    assert local_center_f64.dtype == np.float64
    np.testing.assert_allclose(local_center_f32, local_center_f64, atol=1e-6, rtol=0.0)

    sigma_center_f32 = relion_sigma_offset_prior_center(previous_best_translations)
    sigma_center_f64 = relion_sigma_offset_prior_center(previous_best_translations, dtype=np.float64)
    assert sigma_center_f32.dtype == np.float32
    assert sigma_center_f64.dtype == np.float64
    np.testing.assert_allclose(sigma_center_f32, sigma_center_f64, atol=1e-6, rtol=0.0)

    translations = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    log_prior_f32 = make_relion_translation_log_prior(
        translations, voxel_size, sigma_offset_angstrom=10.0, prior_centers=center_f32
    )
    log_prior_f64 = make_relion_translation_log_prior(
        translations,
        voxel_size,
        sigma_offset_angstrom=10.0,
        prior_centers=center_f64,
        dtype=np.float64,
    )
    assert log_prior_f32.dtype == np.float32
    assert log_prior_f64.dtype == np.float64
    np.testing.assert_allclose(log_prior_f32, log_prior_f64, atol=1e-5, rtol=0.0)

    direction_prior = np.full(12, 1.0 / 12.0, dtype=np.float64)
    direction_log_prior_f32 = make_relion_direction_log_prior(direction_prior, healpix_order=0)
    direction_log_prior_f64 = make_relion_direction_log_prior(direction_prior, healpix_order=0, dtype=np.float64)
    assert direction_log_prior_f32.dtype == np.float32
    assert direction_log_prior_f64.dtype == np.float64
    np.testing.assert_allclose(direction_log_prior_f32, direction_log_prior_f64, atol=1e-5, rtol=0.0)


def test_local_search_precision_rejects_unknown_pass(monkeypatch):
    monkeypatch.delenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", raising=False)
    with pytest.raises(ValueError, match="pass_index"):
        scoring_policy.local_precision(
            scoring_policy.DENSE_PRECISION,
            12,
                pass_index=3,
        )

# ---------------------------------------------------------------------------
# Test constants -- 8x8 images for fast unit tests
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
H, W = IMAGE_SHAPE
N_ROTATIONS = 5
N_TRANSLATIONS = 3
N_IMAGES = 4  # tiny: 2 per half-set
SEED = 42


def test_significance_offset_free_capture_preserves_margin_lost_after_large_common_offset():
    best = np.asarray([1.125, 2.25], dtype=np.float32)
    second = np.asarray([1.0, 2.0], dtype=np.float32)
    common_offset = np.asarray([-100_000_000.0, -100_000_000.0], dtype=np.float64)

    best_offset_free, best_absolute = _capture_offset_free_and_absolute_float32_scores(
        best,
        common_offset,
    )
    second_offset_free, second_absolute = _capture_offset_free_and_absolute_float32_scores(
        second,
        common_offset,
    )

    assert_matches(best_offset_free - second_offset_free, [0.125, 0.25])
    assert_matches(best_absolute, second_absolute)


def test_final_all_data_after_max_iter_env_defaults_to_disabled(monkeypatch, caplog):
    from relax.refinement.refinement_options import FinalPassOptions

    env_name = "RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER"
    monkeypatch.delenv(env_name, raising=False)
    assert FinalPassOptions.from_environ().after_max_iter is False
    monkeypatch.setenv(env_name, "1")
    assert FinalPassOptions.from_environ().after_max_iter is True
    monkeypatch.setenv(env_name, "maybe")
    assert FinalPassOptions.from_environ().after_max_iter is False
    assert "Ignoring invalid RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER" in caplog.text

    def should_run(*, has_converged=False, iteration=5, after_max_iter=False, force=False, **kwargs):
        return finalization._should_run_final_all_data_iteration(
            logger=iteration_loop_module.logger, has_converged=has_converged, iteration=iteration, max_iter=5,
            force_max_iter_after_convergence=force, after_max_iter=after_max_iter, **kwargs,
        )

    assert should_run() is False
    assert should_run(after_max_iter=True) is True
    assert should_run(has_converged=True) is True
    assert should_run(iteration=4, after_max_iter=True) is False
    assert should_run(after_max_iter=True, force=True) is False
    # Class3D never admits the pass after exhaustion; it warns where the option asks for it.
    class_options = stand_in.options(
        schedule=stand_in.schedule(max_iter=5), final_pass=FinalPassOptions(after_max_iter=True),
    )
    caplog.clear()
    assert finalization.class_after_max_iter(SimpleNamespace(has_converged=False), class_options, iteration=5) is False
    assert "Ignoring RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER=1 for K-class" in caplog.text


def test_kclass_final_reconstruction_does_not_predivide_class_accumulators(monkeypatch):
    """The Class3D final pass solves each class from its own raw numerator and denominator, then stacks."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    trace = CallTrace(monkeypatch)
    trace.wrap(finalization.final_reconstruction, "reconstruct_final_class_maps", "final_classes")
    trace.wrap(numbered_reconstruction_module, "_reconstruct_volume_eager", "solve")
    run_tiny_refinement(monkeypatch, n_classes=2, converge_after=2)

    (final,) = trace.calls("final_classes")
    numerator, denominator = final.args[0], final.args[1]
    solves = [call for call in trace.calls("solve") if call.inside == ("final_classes",)]
    assert len(solves) == 2
    for class_idx, solve in enumerate(solves):
        assert_matches(np.asarray(solve.args[0]), np.asarray(denominator[class_idx]))
        assert_matches(np.asarray(solve.args[1]), np.asarray(numerator[class_idx]))
    stacked = np.stack([np.asarray(solve.result).reshape(-1) for solve in solves])
    for half in final.result.halves:
        assert_matches(np.asarray(half), stacked)


@pytest.mark.parametrize(
    ("iteration", "perturb_replay_max_iter", "expected_past_cutoff"),
    [
        (0, None, False),  # no cutoff: every iteration stays in range
        (0, 1, False),  # iteration 0 -> recovar iter 1, at the cutoff
        (1, 1, True),  # iteration 1 -> recovar iter 2, past the cutoff
        (5, 1, True),  # cutoff stays exceeded on every later pass (monotonic)
        (0, 0, True),  # cutoff 0 disables replay from the very first iteration
    ],
)
def test_past_perturb_replay_max_iter_matches_one_indexed_cutoff(
    iteration, perturb_replay_max_iter, expected_past_cutoff
):
    assert (
        relion_replay_module._past_perturb_replay_max_iter(iteration, perturb_replay_max_iter)
        is expected_past_cutoff
    )


def testrefine_single_volume_clears_perturb_replay_dir_past_cutoff_source(monkeypatch, tmp_path):
    """Regression for the bug where --replay-override-max-iter only gated the
    explicit ``replay_iteration_overrides`` dict, leaving
    ``refine_single_volume``'s independent per-iteration
    sampling/model/optimiser STAR reads (including the "Replay override:
    optimiser control <- ..." log line) active for every iteration
    regardless of the cutoff. Once an iteration is past the cutoff the run
    reads nothing more from the replay directory: it is moved away there,
    and every later native-sampling decision is made without it.
    """
    import shutil

    from helpers.tiny_refinement import CallTrace, run_tiny_refinement, write_replay_dir

    replay_dir = write_replay_dir(tmp_path / "relion", max_iter=3)

    def move_replay_dir_away(call):
        if call.result and Path(replay_dir).exists():
            shutil.move(replay_dir, tmp_path / "moved")

    from relax.parity import relion_replay_source
    from relax.parity.relion_replay_source import RelionReplaySource

    trace = CallTrace(monkeypatch)
    trace.wrap(relion_replay_source, "_past_perturb_replay_max_iter", "cutoff", after=move_replay_dir_away)
    trace.wrap(RelionReplaySource, "relion_run_directory", "directory")
    trace.wrap(resolution_helpers, "relion_expectation_coarse_size_order", "coarse_order")
    run_tiny_refinement(
        monkeypatch, max_iter=3, final_after_max_iter=False, converge_after=None,
        parity=dict(perturb_replay_relion_dir=replay_dir, perturb_replay_max_iter=1, low_resol_join_halves_angstrom=0.0),
    )
    # Iteration 1 is in range, iteration 2 is the first past it, and every later question is answered past it.
    cutoffs = [call.result for call in trace.calls("cutoff")]
    assert cutoffs[cutoffs.index(True):] == [True] * (len(cutoffs) - cutoffs.index(True))
    assert not Path(replay_dir).exists()
    # The start-up decisions and the first iteration's get the directory; the two past the cutoff do not.
    directories = {call.args[1]: call.result for call in trace.calls("directory")}
    assert directories == {-1: replay_dir, 0: replay_dir, 1: None, 2: None}
    # Nor does the coarse size keep the replayed sampling order.
    saved_orders = [call.kwargs["replay_saved_healpix_order"] for call in trace.calls("coarse_order")]
    assert saved_orders[0] is not None and saved_orders[1:] == [None, None]


@pytest.mark.parametrize(
    ("iteration", "replay_dir", "cutoff", "expected_directory"),
    [
        (0, "/replay", None, "/replay"),
        (0, "/replay", 0, None),
        (0, "/replay", 1, "/replay"),
        (1, "/replay", 1, None),
        (1, None, 1, None),
    ],
)
def test_replay_source_supplies_the_star_directory_up_to_the_cutoff(iteration, replay_dir, cutoff, expected_directory):
    """The controller samples natively where the source supplies no STAR directory and no sealed sampling state
    set the iteration (the sealed state is the controller's own test)."""
    from relax.parity.relion_replay_source import RelionReplay, RelionReplaySource

    replay = RelionReplay(perturb_replay_max_iter=cutoff, perturb_replay_relion_dir=replay_dir)
    assert RelionReplaySource(replay, stand_in.options()).relion_run_directory(iteration) == expected_directory


def test_replay_translation_grid_preserves_state_grid_for_subtolerance_star_rounding(monkeypatch, tmp_path):
    class State:
        healpix_order = 3
        max_healpix_order = 3
        auto_local_healpix_order = 4
        auto_sampling = True
        do_local_search = False
        sigma_rot = 0.0
        sigma_psi = 0.0
        translation_range = 3.0
        translation_step = 1.0

    class Cryo:
        voxel_size = 1.4166666666666667

    sampling_paths = []

    def fake_sampling_metadata(path):
        sampling_paths.append(path)
        return {
            "random_perturbation": -0.11451,
            "perturbation_factor": 0.5,
            "healpix_order": 3,
            "psi_step": 7.5,
            "offset_range": 4.25,
            "offset_step": 1.416667,
        }

    monkeypatch.setattr(relion_replay_module, "read_relion_sampling_metadata", fake_sampling_metadata)

    state = State()
    state_grid = get_translation_grid(state.translation_range, state.translation_step)
    rounded_star_grid = get_translation_grid(4.25 / Cryo.voxel_size, 1.416667 / Cryo.voxel_size)
    assert state_grid.shape[0] == 29
    assert rounded_star_grid.shape[0] == 25

    direction_priors = [DirectionPrior(None, None), DirectionPrior(None, None)]
    result = relion_replay_module.apply_iter_replay_overrides(
        iter_replay_override=None,
        perturb_replay_relion_dir=str(tmp_path),
        perturb_replay_relion_prefix="custom",
        init_relion_iteration=6,
        iteration=0,
        state=state,
        cs=64,
        image_geometry=ImageGeometry(image_shape=(8, 8), pixel_size_angstrom=Cryo.voxel_size),
        n_classes=1,
        relion_half_inputs=initialize_halfsets(
        (None, None),
            previous_best_translations=None,
            previous_best_rotation_eulers=None,
            image_corrections=None,
            scale_corrections=None,
            optics_group_ids=(None, None), group_ids=None, group_count=None,
        ),
        previous_best_rotations=[None, None],
        noise_model=NoiseModel(
            variance_per_half=[None, None],
            average_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            radial_per_half=[None, None],
            average_radial=None,
        ),
        current_sigma_offset_angstrom=10.0,
        direction_priors=direction_priors,
        dtype=np.float64,
    )

    replay_grid = np.asarray(result.prior_translations)
    assert replay_grid.dtype == np.float64
    assert replay_grid.shape[0] == 29
    assert get_translation_grid(state.translation_range, state.translation_step).shape[0] == 29
    assert state.translation_range == pytest.approx(3.0)
    assert state.translation_step == pytest.approx(1.0)
    assert sampling_paths == [str(tmp_path / "custom_it007_sampling.star")]
    np.testing.assert_allclose(replay_grid, state_grid, rtol=0.0, atol=1e-6)


def test_translation_grid_matches_relion_ceil_boundary_for_every_class_count(monkeypatch):
    # HealpixSampling::setTranslations (healpix_sampling.cpp:344, 413-454) enumerates CEIL(range/step) for
    # every K; floor division dropped the four axial points of 4.25 / 1.416667 A for K>1 too.
    rounded_range = 4.25 / 1.4166666666666667
    rounded_step = 1.416667 / 1.4166666666666667

    k1_grid = sampling_module._translation_grid_for_class_count(
        rounded_range,
        rounded_step,
        n_classes=1,
    )
    k4_grid = sampling_module._translation_grid_for_class_count(
        rounded_range,
        rounded_step,
        n_classes=4,
    )
    monkeypatch.setenv("RELAX_K1_RELION_EXACT_TRANSLATION_GRID", "0")
    diagnostic_control_grid = sampling_module._translation_grid_for_class_count(
        rounded_range,
        rounded_step,
        n_classes=1,
    )

    assert k1_grid.shape == (29, 2)
    assert k4_grid.shape == (29, 2)
    assert diagnostic_control_grid.shape == (25, 2)
    np.testing.assert_allclose(
        k1_grid[[0, 14, -1]],
        np.asarray([[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]]) * rounded_step,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(k4_grid, k1_grid, rtol=0.0, atol=0.0)


def test_k4_translation_grid_matches_relions_recorded_count():
    # RELION's K=4 Class3D reference run (fixture k4_5k128_relion_os0) samples offsets with range 25.5 A and
    # step 8.5 A (run_it001_sampling.star) and logs "TranslationalSampling= 8.5 NrTranslations= 29".
    import re

    from helpers.em_fixtures import fixture_file

    sampling_star = fixture_file("k4_5k128_relion_os0", "run_it001_sampling.star").read_text()
    offset_range = float(re.search(r"_rlnOffsetRange\s+(\S+)", sampling_star).group(1))
    offset_step = float(re.search(r"_rlnOffsetStep\s+(\S+)", sampling_star).group(1))
    log = fixture_file("k4_5k128_relion_os0", "relion_run.log").read_text()
    recorded = {int(n) for step, n in re.findall(r"TranslationalSampling= (\S+) NrTranslations= (\d+)", log)
                if float(step) == offset_step}
    grid = sampling_module._translation_grid_for_class_count(
        offset_range / offset_step, 1.0, n_classes=4, source_units_per_pixel=offset_step
    )
    assert recorded == {grid.shape[0]} == {29}


def test_k1_translation_grid_rejects_invalid_diagnostic_switch(monkeypatch):
    monkeypatch.setenv("RELAX_K1_RELION_EXACT_TRANSLATION_GRID", "sometimes")
    with pytest.raises(ValueError, match="must be a boolean value"):
        sampling_module._translation_grid_for_class_count(3.0, 1.0, n_classes=1)


def test_local_translation_prior_ignores_stale_replay_grid_shape():
    current_grid = get_translation_grid(3.0, 1.0).astype(np.float32)
    stale_replay_grid = get_translation_grid(3.0, 1.0000002352941175).astype(np.float32)
    assert current_grid.shape[0] == 29
    assert stale_replay_grid.shape[0] == 25

    chosen, source, mismatched = half_scoring._local_translation_prior_reference_translations(
        current_translations=current_grid + 0.125,
        base_translations=current_grid,
        replay_prior_translations=stale_replay_grid,
    )

    assert source == "base"
    assert mismatched is True
    np.testing.assert_allclose(chosen, current_grid)


def test_replay_override_preserves_half_specific_sigma_offsets():
    class State:
        healpix_order = 1
        max_healpix_order = 1
        auto_local_healpix_order = 4
        do_local_search = False
        sigma_rot = 0.0
        sigma_psi = 0.0
        translation_range = 3.0
        translation_step = 1.0

    class Cryo:
        voxel_size = 1.0

    direction_priors = [DirectionPrior(None, None), DirectionPrior(None, None)]
    result = relion_replay_module.apply_iter_replay_overrides(
        iter_replay_override={
            "translation_sigma_angstrom": 99.0,
            "translation_sigma_angstrom_per_half": [5.0, 7.0],
        },
        perturb_replay_relion_dir=None,
        init_relion_iteration=0,
        iteration=1,
        state=State(),
        cs=8,
        image_geometry=ImageGeometry(image_shape=(8, 8), pixel_size_angstrom=Cryo.voxel_size),
        n_classes=1,
        relion_half_inputs=initialize_halfsets(
        (None, None),
            previous_best_translations=None,
            previous_best_rotation_eulers=None,
            image_corrections=None,
            scale_corrections=None,
            optics_group_ids=(None, None), group_ids=None, group_count=None,
        ),
        previous_best_rotations=[None, None],
        noise_model=NoiseModel(
            variance_per_half=[None, None],
            average_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            radial_per_half=[None, None],
            average_radial=None,
        ),
        current_sigma_offset_angstrom=10.0,
        direction_priors=direction_priors,
    )

    assert result.current_sigma_offset_angstrom == pytest.approx(6.0)
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([5.0, 7.0])


def test_replay_override_preserves_native_scale_and_rescales_star_image_correction():
    class State:
        healpix_order = 1
        max_healpix_order = 1
        auto_local_healpix_order = 4
        do_local_search = False
        sigma_rot = 0.0
        sigma_psi = 0.0
        translation_range = 3.0
        translation_step = 1.0

    class Cryo:
        voxel_size = 1.0

    group_ids = [np.asarray([0, 1], dtype=np.int64), np.asarray([1], dtype=np.int64)]
    relion_half_inputs = initialize_halfsets(
        (None, None),
        previous_best_translations=None,
        previous_best_rotation_eulers=None,
        image_corrections=[np.asarray([9.0, 9.0], dtype=np.float32), np.asarray([8.0], dtype=np.float32)],
        scale_corrections=[np.asarray([7.0, 7.0], dtype=np.float32), np.asarray([6.0], dtype=np.float32)],
        group_ids=group_ids,
        group_count=3000,
        optics_group_ids=(None, None),
    )

    direction_priors = [DirectionPrior(None, None), DirectionPrior(None, None)]
    relion_replay_module.apply_iter_replay_overrides(
        iter_replay_override={
            "image_corrections": [
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([3.0], dtype=np.float32),
            ],
            "serialized_scale_corrections": [
                np.asarray([4.0, 5.0], dtype=np.float32),
                np.asarray([6.0], dtype=np.float32),
            ],
        },
        perturb_replay_relion_dir=None,
        init_relion_iteration=0,
        iteration=1,
        state=State(),
        cs=8,
        image_geometry=ImageGeometry(image_shape=(8, 8), pixel_size_angstrom=Cryo.voxel_size),
        n_classes=1,
        relion_half_inputs=relion_half_inputs,
        previous_best_rotations=[None, None],
        noise_model=NoiseModel(
            variance_per_half=[None, None],
            average_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            radial_per_half=[None, None],
            average_radial=None,
        ),
        current_sigma_offset_angstrom=10.0,
        direction_priors=direction_priors,
    )

    np.testing.assert_allclose(relion_half_inputs[0].image_corrections, [1.75, 2.8])
    np.testing.assert_allclose(relion_half_inputs[0].scale_corrections, [7.0, 7.0])
    assert_matches(relion_half_inputs[0].group_ids, group_ids[0])
    assert_matches(relion_half_inputs[1].group_ids, group_ids[1])
    assert [particle_half.group_count for particle_half in relion_half_inputs] == [3000, 3000]


def test_replay_explicit_scoring_scale_preserves_image_to_scale_ratio():
    relion_half_inputs = initialize_halfsets(
        (None, None),
        previous_best_translations=None,
        previous_best_rotation_eulers=None,
        image_corrections=None,
        scale_corrections=None,
        optics_group_ids=(None, None), group_ids=None, group_count=None,
    )

    class State:
        healpix_order = max_healpix_order = 1
        auto_local_healpix_order = 4
        do_local_search = False
        sigma_rot = sigma_psi = 0.0
        translation_range = 3.0
        translation_step = 1.0

    class Cryo:
        voxel_size = 1.0

    direction_priors = [DirectionPrior(None, None), DirectionPrior(None, None)]
    relion_replay_module.apply_iter_replay_overrides(
        iter_replay_override={
            "image_corrections": [np.asarray([1.0, 2.0]), np.asarray([], dtype=np.float32)],
            "serialized_scale_corrections": [np.asarray([4.0, 5.0]), np.asarray([], dtype=np.float32)],
            "scoring_scale_corrections": [np.asarray([8.0, 10.0]), np.asarray([], dtype=np.float32)],
        },
        perturb_replay_relion_dir=None,
        init_relion_iteration=0,
        iteration=1,
        state=State(),
        cs=8,
        image_geometry=ImageGeometry(image_shape=(8, 8), pixel_size_angstrom=Cryo.voxel_size),
        n_classes=4,
        relion_half_inputs=relion_half_inputs,
        previous_best_rotations=[None, None],
        noise_model=NoiseModel(
            variance_per_half=[None, None],
            average_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            radial_per_half=[None, None],
            average_radial=None,
        ),
        current_sigma_offset_angstrom=10.0,
        direction_priors=direction_priors,
    )

    np.testing.assert_allclose(relion_half_inputs[0].scale_corrections, [8.0, 10.0])
    np.testing.assert_allclose(relion_half_inputs[0].image_corrections, [2.0, 4.0])


def test_replay_cold_start_falls_back_to_serialized_scale():
    relion_half_inputs = initialize_halfsets(
        (None, None),
        previous_best_translations=None,
        previous_best_rotation_eulers=None,
        image_corrections=None,
        scale_corrections=None,
        optics_group_ids=(None, None), group_ids=None, group_count=None,
    )
    state = SimpleNamespace(
        healpix_order=1,
        max_healpix_order=1,
        auto_local_healpix_order=4,
        auto_sampling=True,
        do_local_search=False,
        sigma_rot=0.0,
        sigma_psi=0.0,
        translation_range=3.0,
        translation_step=1.0,
    )

    direction_priors = [DirectionPrior(None, None), DirectionPrior(None, None)]
    relion_replay_module.apply_iter_replay_overrides(
        iter_replay_override={
            "image_corrections": [np.asarray([1.0, 2.0]), np.asarray([], dtype=np.float32)],
            "serialized_scale_corrections": [np.asarray([4.0, 5.0]), np.asarray([], dtype=np.float32)],
        },
        perturb_replay_relion_dir=None,
        init_relion_iteration=0,
        iteration=0,
        state=state,
        cs=8,
        image_geometry=ImageGeometry(image_shape=(8, 8), pixel_size_angstrom=1.0),
        n_classes=1,
        relion_half_inputs=relion_half_inputs,
        previous_best_rotations=[None, None],
        noise_model=NoiseModel(
            variance_per_half=[None, None],
            average_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            radial_per_half=[None, None],
            average_radial=None,
        ),
        current_sigma_offset_angstrom=10.0,
        direction_priors=direction_priors,
    )

    np.testing.assert_allclose(relion_half_inputs[0].scale_corrections, [4.0, 5.0])
    np.testing.assert_allclose(relion_half_inputs[0].image_corrections, [1.0, 2.0])


@pytest.mark.parametrize("with_resident_state", [False, True])
def test_replay_explicit_paired_image_scale_state_remains_exact(with_resident_state):
    relion_half_inputs = initialize_halfsets(
        (None, None),
        previous_best_translations=None,
        previous_best_rotation_eulers=None,
        image_corrections=(
            [np.asarray([9.0, 9.0]), np.asarray([], dtype=np.float32)]
            if with_resident_state
            else None
        ),
        scale_corrections=(
            [np.asarray([7.0, 7.0]), np.asarray([], dtype=np.float32)]
            if with_resident_state
            else None
        ),
        optics_group_ids=(None, None), group_ids=None, group_count=None,
    )

    with pytest.raises(ValueError, match="Replay scale requires"):
        relion_replay_module._apply_replay_correction_overrides(
            relion_half_inputs=relion_half_inputs,
            replay_override={"scale_corrections": [np.asarray([4.0, 5.0]), None]},
        )

    relion_replay_module._apply_replay_correction_overrides(
        relion_half_inputs=relion_half_inputs,
        replay_override={
            "image_corrections": [np.asarray([1.0, 2.0]), np.asarray([], dtype=np.float32)],
            "serialized_scale_corrections": [np.asarray([4.0, 5.0]), np.asarray([], dtype=np.float32)],
            "scoring_scale_corrections": [np.asarray([4.0, 5.0]), np.asarray([], dtype=np.float32)],
        },
    )

    assert_matches(relion_half_inputs[0].image_corrections, [1.0, 2.0])
    assert_matches(relion_half_inputs[0].scale_corrections, [4.0, 5.0])


def test_final_all_data_replay_uses_shared_live_scale_correction_contract():
    def halves():
        return initialize_halfsets(
            (None, None),
            previous_best_translations=None,
            previous_best_rotation_eulers=None,
            image_corrections=[np.asarray([9.0, 9.0]), np.asarray([], dtype=np.float32)],
            scale_corrections=[np.asarray([8.0, 10.0]), np.asarray([], dtype=np.float32)],
            optics_group_ids=(None, None), group_ids=None, group_count=None,
        )

    override = {
        "image_corrections": [np.asarray([1.0, 2.0]), np.asarray([], dtype=np.float32)],
        "serialized_scale_corrections": [np.asarray([4.0, 5.0]), np.asarray([], dtype=np.float32)],
    }
    shared, final = halves(), halves()
    shared_fields = relion_replay_module._apply_replay_correction_overrides(
        relion_half_inputs=shared, replay_override=override,
    )
    sigma_offset, noise_model = object(), object()
    kept_sigma, kept_noise, final_fields = relion_replay_module._install_final_replay_particle_state(
        override, final, sigma_offset=sigma_offset, noise_model=noise_model, image_shape=IMAGE_SHAPE, dtype=np.float32,
    )
    # The final pass installs the corrections through the numbered iterations' contract, and nothing else.
    assert final_fields == shared_fields
    assert kept_sigma is sigma_offset and kept_noise is noise_model
    for final_half, shared_half in zip(final, shared, strict=True):
        assert_matches(final_half.image_corrections, shared_half.image_corrections)
        assert_matches(final_half.scale_corrections, shared_half.scale_corrections)

    relion_half_inputs = initialize_halfsets(
        (None, None),
        previous_best_translations=None,
        previous_best_rotation_eulers=None,
        image_corrections=[np.asarray([9.0, 9.0]), np.asarray([], dtype=np.float32)],
        scale_corrections=[np.asarray([8.0, 10.0]), np.asarray([], dtype=np.float32)],
        optics_group_ids=(None, None), group_ids=None, group_count=None,
    )
    applied = relion_replay_module._apply_replay_correction_overrides(
        relion_half_inputs=relion_half_inputs,
        replay_override={
            "image_corrections": [np.asarray([1.0, 2.0]), np.asarray([], dtype=np.float32)],
            "serialized_scale_corrections": [np.asarray([4.0, 5.0]), np.asarray([], dtype=np.float32)],
        },
    )

    assert applied == ["image_corrections", "serialized_scale_corrections"]
    np.testing.assert_allclose(relion_half_inputs[0].image_corrections, [2.0, 4.0])
    assert_matches(relion_half_inputs[0].scale_corrections, [8.0, 10.0])


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (None, False),
        ([], False),
        ([{"initial_state": True}], False),
        ([{"initial_state": True}, None, None], False),
        ([None, {"numbered_state": True}], True),
        ([{"initial_state": True}, None, {"numbered_state": True}], True),
    ],
)
def test_final_all_data_replay_ignores_cold_start_only_overrides(overrides, expected):
    assert relion_replay_module._has_numbered_replay_iteration_overrides(overrides) is expected



@pytest.mark.parametrize("replace_noise", [False, True])
def test_final_controller_receives_replayed_state_without_retaining_old_noise(
    half_datasets, init_volume, translations, monkeypatch, replace_noise,
):
    import weakref

    initial_model_refs = []
    initialize = noise_updates.initialize_noise_model

    def record_initial_noise(*args, **kwargs):
        model = initialize(*args, **kwargs)
        initial_model_refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(noise_updates, "initialize_noise_model", record_initial_noise)
    marker = refinement_result()
    replayed_noise = jnp.full(IMAGE_SIZE, 7.0, dtype=jnp.float32)

    def capture(halves, **inputs):
        # Final execution consumes no output-only setup/numbered metadata.
        assert not {
            "hard_assignments",
            "frozen_initial_scoring_state_sha256",
            "expected_accuracy_trial_local_indices",
            "expected_accuracy_trial_particle_ids",
            "setup_phase_seconds",
        }.intersection(inputs)
        assert [half.index for half in halves] == [0, 1]
        assert all(half.dataset is dataset for half, dataset in zip(halves, half_datasets))
        assert inputs["expected_accuracy_inputs"].dataset is half_datasets[0]
        assert inputs["carry"].native_sampling_boundary
        assert not inputs["final_use_local"]
        assert inputs["iteration"] == 0
        assert inputs["carry"].coarse_grids.rotation_grid.healpix_order == 2
        assert inputs["carry"].coarse_grids.rotation_grid.rotation_eulers.shape[0] == inputs["carry"].coarse_grids.rotation_grid.rotations.shape[0]
        assert inputs["ctx"].reconstruction_settings.volume_shape == half_datasets[0].volume_shape
        assert {"scoring_dtype", "logger", "n_classes"}.isdisjoint(inputs)
        if replace_noise:
            assert initial_model_refs[0]() is None
            assert_matches(inputs["carry"].noise_model.variance_per_half[0], replayed_noise)
            assert_matches(inputs["carry"].noise_model.variance_per_half[1], replayed_noise)
            assert_matches(inputs["carry"].sigma_offset.per_half_angstrom, [3.0, 4.0])
        else:
            assert inputs["carry"].noise_model is initial_model_refs[0]()
        return marker

    monkeypatch.setattr(finalization, "run_final_all_data", capture)
    result = _refine_replaying(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32)),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=0, init_current_size=16, init_healpix_order=2),
            final_pass=FinalPassOptions(after_max_iter=True),
        ),
        relion_replay=RelionReplay(final_replay_override={
            "noise_variance": replayed_noise,
            "translation_sigma_angstrom_per_half": [3.0, 4.0],
        }) if replace_noise else None,
    )
    # The final pass's result, with the set-up and numbered metadata added by the controller.
    assert result.maps is marker.maps and result.history is marker.history
    assert marker.numbered is None
    assert result.numbered.hard_assignments == [None, None]
    assert result.numbered.frozen_initial_scoring_state_sha256 is None
    assert result.numbered.expected_accuracy_trial_local_indices is None
    assert result.numbered.expected_accuracy_trial_particle_ids is None
    assert set(result.numbered.setup_phase_seconds) == {
        "mask_and_image_cache", "state_init", "sampling_grid", "initial_arrays",
        "direction_prior", "noise_radial_init", "before_iterations",
    }


def test_final_all_data_runs_with_cold_start_only_override(
    fake_global_estep,
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    original_update = convergence_policy.update_refinement_state

    def force_convergence_after_first_iter(*args, **kwargs):
        updated = original_update(*args, **kwargs)
        updated.has_converged = True
        return updated

    monkeypatch.setattr(
        convergence_policy,
        "update_refinement_state",
        force_convergence_after_first_iter,
    )
    original_shell = resolution_helpers.k1_current_resolution_shell
    forced_final_shell = 3
    resolution_calls = []

    def record_resolution_shell(dvp, **kwargs):
        resolution_calls.append({"dvp": np.asarray(dvp).copy(), **kwargs})
        shell = original_shell(dvp, **kwargs)
        return forced_final_shell if len(resolution_calls) == 1 else shell

    monkeypatch.setattr(resolution_helpers, "k1_current_resolution_shell", record_resolution_shell)
    monkeypatch.setattr(finalization, "k1_current_resolution_shell", record_resolution_shell)

    result = _refine_replaying(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=1, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
            parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
        ),
        relion_replay=RelionReplay(replay_iteration_overrides=[{}, None]),
    )

    assert result.convergence_state.has_converged is True
    assert result.final_all_data_ran is True

    # RELION updates rlnCurrentResolution from the final all-data DVP too.
    # The mock data carry no signal, so the split-half value sits at the
    # --minres_map floor of 5 shells (ml_optimiser.cpp:6819) and the final call
    # is forced to a known shell to check that its result reaches the state.
    grid = int(half_datasets[0].image_shape[0])
    voxel = float(half_datasets[0].voxel_size)
    # Only the final pass goes through k1_current_resolution_shell; the numbered K1
    # iteration estimates its shell in estimate_k1_iteration_resolution.
    final_call = resolution_calls[-1]
    assert len(resolution_calls) == 1
    assert final_call["current_size"] == grid
    assert_matches(final_call["dvp"], result.final_pass.tau2_ssnr.astype(np.float32))
    state = result.convergence_state
    assert state.current_resolution == grid * voxel / forced_final_shell
    assert state.previous_resolution == grid * voxel / 5


def test_last_numbered_state_does_not_trigger_post_cap_final_all_data(
    fake_global_estep,
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    original_update = convergence_policy.update_refinement_state

    def make_post_numbered_state_ready(*args, **kwargs):
        updated = original_update(*args, **kwargs)
        updated.has_converged = False
        updated.has_fine_enough_angular_sampling = True
        updated.nr_iter_wo_resol_gain = 2
        updated.nr_iter_wo_large_hidden_variable_changes = 2
        updated.smallest_changes_optimal_orientations = 0.25
        return updated

    monkeypatch.setattr(
        convergence_policy,
        "update_refinement_state",
        make_post_numbered_state_ready,
    )

    result = refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=1, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
            parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    assert result.convergence_state.has_converged is False
    assert result.final_all_data_ran is False


def _mock_local_search_result(
    base_outputs,
    relion_stats,
    noise_stats,
    kwargs,
    n_units,
    best_pose_details=(),
):
    best_rotations, best_translations = best_pose_details or (None, None)
    return LocalSearchResult(
        Ft_y=base_outputs[0],
        Ft_ctf=base_outputs[1],
        hard_assignment=base_outputs[2],
        best_pose_rotations=best_rotations,
        best_pose_translations=best_translations,
        relion_stats=relion_stats,
        noise_stats=noise_stats if kwargs.get("accumulate_noise", False) else None,
        profile_summary=(
            {"reconstruction_sample_indices_by_image": [None] * int(n_units)}
            if kwargs.get("return_profile", False) else None
        ),
    )


# The global E-step's CPU stand-in (helpers.fake_adaptive_engine) with its default result.
_mock_run_adaptive_em = fake_adaptive_engine()


def _random_half_maps(calls):
    """``Ft_y`` for the fake engine: a new random Hermitian volume per call, so the two half maps differ."""

    def half_map(k, size):
        side = round(size ** (1.0 / 3.0))
        return _hermitian_volume((side, side, side), seed=3000 + 7 * len(calls) + k)

    return half_map


def _mock_reconstruction_accumulator_size(experiment_dataset, kwargs, *, current_size=None):
    """Match the active full-grid or RELION x-half BackProjector layout."""

    padding_factor = int(kwargs.get("reconstruction_padding_factor", 1))
    if kwargs.get("mstep_relion_x_half", False) and "rotation_grid_mstep_rotations" in kwargs:
        if current_size is None and kwargs.get("relion_projector_r_max") is not None:
            current_size = 2 * int(kwargs["relion_projector_r_max"])
        shape = relion_backprojector_volume_shape(
            experiment_dataset.volume_shape,
            padding_factor,
            current_size=current_size,
        )
    else:
        shape = tuple(int(size) * padding_factor for size in experiment_dataset.volume_shape)
    return int(np.prod(shape))


def test_build_local_hypothesis_layout_and_bucketization_preserve_per_image_support(monkeypatch):
    import relax.local.local_layout as local_layout_mod

    call_count = {"value": 0}

    def fake_selector(
        prior_rotation_indices,
        sigma_rot,
        sigma_psi,
        healpix_order,
        sigma_cutoff=3.0,
        *,
        per_image=False,
        grid_metadata=None,
    ):
        assert per_image
        image_idx = call_count["value"]
        call_count["value"] += 1
        if image_idx == 0:
            return np.array([1, 3], dtype=np.int64), np.array([[0.0, -1.0]], dtype=np.float32)
        return np.array([2, 4, 5], dtype=np.int64), np.array([[0.0, -1.0, -2.0]], dtype=np.float32)

    monkeypatch.setattr(local_layout_mod, "get_local_rotation_grid_fast", fake_selector)
    monkeypatch.setattr(
        local_layout_mod,
        "make_relion_translation_log_prior",
        lambda *args, **kwargs: np.zeros((2, 3), dtype=np.float32),
    )

    prior_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
    rotation_grid_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 8, axis=0)
    translations = np.zeros((3, 2), dtype=np.float32)
    prior_translations = np.zeros((2, 2), dtype=np.float32)

    layout = build_local_hypothesis_layout(
        prior_rotations,
        rotation_grid_rotations,
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=4,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata={"mode": "full", "n_pixels": np.int64(192), "n_psi": np.int64(16)},
    )

    assert_matches(layout.rotation_offsets, np.array([0, 2, 5], dtype=np.int64))
    assert_matches(layout.rotation_counts, np.array([2, 3], dtype=np.int32))
    assert_matches(layout.rotation_ids_flat, np.array([1, 3, 2, 4, 5], dtype=np.int32))

    buckets = bucket_local_hypothesis_layout(
        layout, image_batch_size=2, rotation_block_size=16, max_hypotheses_per_microbatch=64
    )
    assert len(buckets) == 1
    assert buckets[0].bucket_image_count == 2
    assert_matches(buckets[0].actual_rotation_counts, np.array([2, 3], dtype=np.int32))
    assert_matches(buckets[0].local_rotation_ids[0, :2], np.array([1, 3], dtype=np.int32))
    assert not np.any(buckets[0].local_rotation_mask[0, 2:])
    assert_matches(buckets[0].local_rotation_ids[1, :3], np.array([2, 4, 5], dtype=np.int32))


def test_build_pass2_hypothesis_layout_preserves_sparse_rotation_translation_mask():
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    significant_samples = [
        np.array([0, 3], dtype=np.int32),  # rot 0/trans 0 and rot 1/trans 1
        np.array([2], dtype=np.int32),  # rot 1/trans 0
    ]

    layout = build_pass2_hypothesis_layout(
        significant_samples,
        n_coarse_rotations=rotation_grid_size(0),
        n_coarse_translations=2,
        nside_level=0,
        translations=translations,
        oversampling_order=0,
        rotation_log_prior=np.arange(rotation_grid_size(0), dtype=np.float32),
        translation_log_prior=np.array([0.0, -2.0], dtype=np.float32),
    )

    assert layout.n_images == 2
    assert_matches(layout.rotation_counts, np.array([2, 1], dtype=np.int32))
    assert_matches(layout.rotation_offsets, np.array([0, 2, 3], dtype=np.int64))
    assert_matches(layout.rotation_posterior_ids_flat, np.array([0, 1, 1], dtype=np.int32))
    assert layout.sample_mask_rows().shape == (3, 2)
    assert_matches(
        layout.sample_mask_rows(),
        np.array(
            [
                [True, False],
                [False, True],
                [True, False],
            ],
            dtype=bool,
        ),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=2,
        rotation_block_size=4,
        max_hypotheses_per_microbatch=64,
    )
    assert len(buckets) == 1
    row_for_image0 = int(np.flatnonzero(buckets[0].image_indices == 0)[0])
    assert_matches(
        buckets[0].local_rotation_posterior_ids[row_for_image0, :2],
        np.array([0, 1], dtype=np.int32),
    )
    assert_matches(buckets[0].local_sample_mask[row_for_image0, :2], layout.sample_mask_rows()[:2])
    assert not np.any(buckets[0].local_sample_mask[row_for_image0, 2:])


@pytest.mark.parametrize("oversampling_order", [0, 1])
def test_build_pass2_hypothesis_layout_accepts_complement_support(oversampling_order):
    from relax.scoring.significant_samples import (
        ComplementSignificantSampleIndices,
        compact_significant_sample_indices_from_mask,
    )

    n_rotations = rotation_grid_size(0)
    mask = np.ones(n_rotations * 2, dtype=bool)
    mask[[0, 1, 3, mask.size - 1]] = False
    compact = compact_significant_sample_indices_from_mask(mask)
    assert isinstance(compact, ComplementSignificantSampleIndices)
    kwargs = dict(
        n_coarse_rotations=n_rotations,
        n_coarse_translations=2,
        nside_level=0,
        translations=np.array([[0., 0.], [1., 0.]], dtype=np.float32),
        oversampling_order=oversampling_order,
        rotation_log_prior=np.arange(n_rotations, dtype=np.float32),
        translation_log_prior=np.array([0., -2.], dtype=np.float32),
    )
    expected = build_pass2_hypothesis_layout([np.flatnonzero(mask)], **kwargs)
    actual = build_pass2_hypothesis_layout([compact], **kwargs)
    for field in fields(expected):
        assert_matches(
            getattr(actual, field.name), getattr(expected, field.name),
            err_msg=field.name,
        )


def test_build_pass2_hypothesis_layout_accepts_fine_translation_log_prior():
    translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    fine_prior = np.arange(8, dtype=np.float32)

    layout = build_pass2_hypothesis_layout(
        [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)],
        n_coarse_rotations=rotation_grid_size(0),
        n_coarse_translations=2,
        nside_level=0,
        translations=translations,
        oversampling_order=1,
        translation_step=2.0,
        fine_translation_log_prior=fine_prior,
    )

    assert layout.translation_grid.shape[0] == fine_prior.shape[0]
    np.testing.assert_allclose(
        layout.translation_log_priors,
        np.broadcast_to(fine_prior[None, :], (2, fine_prior.shape[0])),
    )


def test_build_pass2_hypothesis_layout_preserves_float64_operands():
    delta = 2.0**-40
    layout = build_pass2_hypothesis_layout(
        [np.array([0], dtype=np.int32)],
        n_coarse_rotations=rotation_grid_size(0),
        n_coarse_translations=1,
        nside_level=0,
        translations=np.array([[1.0 + delta, 0.0]], dtype=np.float64),
        oversampling_order=0,
        rotation_log_prior=np.full(rotation_grid_size(0), 1.0 + delta, dtype=np.float64),
        fine_translation_log_prior=np.array([1.0 + delta], dtype=np.float64),
        dtype=np.float64,
    )

    assert layout.rotations_flat.dtype == np.float64
    assert layout.translation_grid.dtype == np.float64
    assert layout.rotation_log_priors_flat.dtype == np.float64
    assert layout.translation_log_priors.dtype == np.float64
    assert layout.translation_grid[0, 0] == 1.0 + delta
    assert layout.rotation_log_priors_flat[0] == 1.0 + delta
    assert layout.translation_log_priors[0, 0] == 1.0 + delta


def test_build_pass2_hypothesis_layout_rejects_rotation_ids_outside_coarse_grid():
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="outside the coarse grid"):
        build_pass2_hypothesis_layout(
            [np.array([4], dtype=np.int32)],  # rot 2/trans 0 for a 2-rotation grid
            n_coarse_rotations=2,
            n_coarse_translations=2,
            nside_level=0,
            translations=translations,
            oversampling_order=0,
        )

    with pytest.raises(ValueError, match="outside the coarse grid"):
        build_pass2_hypothesis_layout(
            [np.array([-1], dtype=np.int32)],
            n_coarse_rotations=2,
            n_coarse_translations=2,
            nside_level=0,
            translations=translations,
            oversampling_order=0,
        )


def test_build_pass2_hypothesis_layout_can_keep_empty_class_support():
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    layout = build_pass2_hypothesis_layout(
        [np.array([], dtype=np.int32)],
        n_coarse_rotations=rotation_grid_size(0),
        n_coarse_translations=2,
        nside_level=0,
        translations=translations,
        oversampling_order=0,
        allow_empty=True,
    )

    assert layout.n_images == 1
    assert_matches(layout.rotation_counts, np.array([1], dtype=np.int32))
    assert layout.sample_mask_rows().shape == (1, 2)
    assert not np.any(layout.sample_mask_rows())


# Moved from relax/local/local_layout.py (PLAN e1): no relax module uses it, only this test.
def _lookup_values_by_id(ids: np.ndarray, values: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``values`` for integer ids without allocating a global id table."""

    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    values_np = np.asarray(values)
    query_np = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if query_np.size == 0:
        return values_np[:0], np.ones(0, dtype=bool)
    if ids_np.size == 0:
        return values_np[:0], np.zeros(query_np.shape, dtype=bool)

    order = np.argsort(ids_np, kind="stable")
    sorted_ids = ids_np[order]
    # Match the previous dense table behavior for duplicate ids: later writes
    # won, so search to the right and take the last matching entry.
    pos = np.searchsorted(sorted_ids, query_np, side="right") - 1
    valid = pos >= 0
    matched = np.zeros(query_np.shape, dtype=bool)
    if np.any(valid):
        matched[valid] = sorted_ids[pos[valid]] == query_np[valid]
    if not np.all(matched):
        return values_np[:0], matched
    return values_np[order[pos]], matched


def test_local_id_lookup_matches_dense_table_duplicate_semantics():
    ids = np.array([4, 2, 4, 9], dtype=np.int32)
    values = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    selected, matched = _lookup_values_by_id(ids, values, np.array([2, 4, 9], dtype=np.int32))

    assert_matches(matched, np.array([True, True, True]))
    np.testing.assert_allclose(selected, np.array([0.5, 0.75, 1.0], dtype=np.float32))
    assert selected.dtype == np.float32


def test_build_local_adaptive_pass2_hypothesis_layout_masks_parent_pairs():
    parent_order = 0
    oversampling_order = 1
    coarse_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_order),
        n_pixels=12,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 3], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
        rotation_log_priors_flat=np.array([0.0, -2.0, -3.0], dtype=np.float32),
        rotation_counts=np.array([2, 1], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.array([[0.0, -1.0], [-4.0, -5.0]], dtype=np.float32),
    )
    significant_samples = [
        np.array([0, 3], dtype=np.int32),  # rot 0/trans 0 and rot 1/trans 1
        np.array([2], dtype=np.int32),  # rot 1/trans 0
    ]

    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        significant_samples,
        parent_order,
        oversampling_order=oversampling_order,
        translation_step=2.0,
    )

    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        2.0,
        oversampling_order=oversampling_order,
    )
    np.testing.assert_allclose(layout.translation_grid, fine_translations, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        layout.translation_log_priors,
        parent_layout.translation_log_priors[:, fine_translation_parent],
    )
    assert layout.n_global_rotations == rotation_grid_size(parent_order)
    assert_matches(layout.rotation_counts, np.array([16, 8], dtype=np.int32))

    child_rots0, parent_map0, child_ids0 = get_oversampled_rotation_grid_from_samples(
        np.array([0, 1], dtype=np.int32),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    assert_matches(layout.rotation_ids_flat[:16], child_ids0.astype(np.int32))
    assert_matches(layout.rotation_posterior_ids_flat[:16], np.array([0, 1], dtype=np.int32)[parent_map0])
    np.testing.assert_allclose(layout.rotations_flat[:16], child_rots0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        layout.rotation_log_priors_flat[:16],
        np.array([0.0, -2.0], dtype=np.float32)[parent_map0],
    )
    expected0 = np.zeros((16, fine_translation_parent.size), dtype=bool)
    expected0[parent_map0 == 0] = fine_translation_parent == 0
    expected0[parent_map0 == 1] = fine_translation_parent == 1
    assert_matches(layout.sample_mask_rows()[:16], expected0)

    _, parent_map1, child_ids1 = get_oversampled_rotation_grid_from_samples(
        np.array([1], dtype=np.int32),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    assert_matches(layout.rotation_ids_flat[16:], child_ids1.astype(np.int32))
    assert_matches(layout.rotation_posterior_ids_flat[16:], np.full(parent_map1.shape, 1, dtype=np.int32))
    np.testing.assert_allclose(layout.rotation_log_priors_flat[16:], np.full(parent_map1.shape, -3.0))
    assert_matches(
        layout.sample_mask_rows()[16:],
        np.broadcast_to(fine_translation_parent[None, :] == 0, layout.sample_mask_rows()[16:].shape),
    )


def test_build_local_adaptive_pass2_hypothesis_layout_empty_significant_samples_fallback_to_local_support():
    parent_order = 0
    oversampling_order = 1
    coarse_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_order),
        n_pixels=12,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        rotation_log_priors_flat=np.array([0.0, -2.0], dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.array([[0.0, -1.0]], dtype=np.float32),
    )

    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [np.array([], dtype=np.int64)],
        parent_order,
        oversampling_order=oversampling_order,
        translation_step=2.0,
    )

    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        2.0,
        oversampling_order=oversampling_order,
    )
    child_rots, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
        np.array([0, 1], dtype=np.int32),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )

    np.testing.assert_allclose(layout.translation_grid, fine_translations, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        layout.translation_log_priors,
        parent_layout.translation_log_priors[:, fine_translation_parent],
    )
    assert_matches(layout.rotation_counts, np.array([16], dtype=np.int32))
    assert_matches(layout.rotation_offsets, np.array([0, 16], dtype=np.int64))
    assert_matches(layout.rotation_ids_flat, child_ids.astype(np.int32))
    assert_matches(layout.rotation_posterior_ids_flat, np.array([0, 1], dtype=np.int32)[parent_map])
    np.testing.assert_allclose(layout.rotations_flat, child_rots, rtol=1e-6, atol=1e-6)
    assert layout.sample_mask_rows() is None
    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=1,
        rotation_block_size=8,
        max_hypotheses_per_microbatch=128,
    )
    assert buckets
    assert all(bucket.local_sample_mask is None for bucket in buckets)


def test_build_local_adaptive_pass2_hypothesis_layout_none_uses_full_parent_support():
    parent_order = 0
    coarse_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_order),
        n_pixels=12,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        rotation_log_priors_flat=np.array([0.0, -2.0], dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.array([[0.0, -1.0]], dtype=np.float32),
    )

    layout_none = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [None],
        parent_order,
        oversampling_order=1,
        translation_step=2.0,
    )
    layout_empty = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [np.zeros(0, dtype=np.int64)],
        parent_order,
        oversampling_order=1,
        translation_step=2.0,
    )

    assert_matches(layout_none.rotation_offsets, layout_empty.rotation_offsets)
    assert_matches(layout_none.rotation_counts, layout_empty.rotation_counts)
    assert_matches(layout_none.rotation_ids_flat, layout_empty.rotation_ids_flat)
    assert_matches(layout_none.rotation_posterior_ids_flat, layout_empty.rotation_posterior_ids_flat)
    assert layout_none.sample_mask_rows() is None
    assert layout_empty.sample_mask_rows() is None


def test_expand_significant_samples_to_full_parent_translations_preserves_rotation_support():
    samples = [
        np.array([0, 3, 4], dtype=np.int64),  # rotations 0 and 1 with n_trans=3
        None,
        np.zeros(0, dtype=np.int64),
    ]

    expanded = half_scoring._expand_significant_samples_to_full_parent_translations(
        samples,
        n_parent_translations=3,
    )

    assert_matches(expanded[0], np.array([0, 1, 2, 3, 4, 5], dtype=np.int64))
    assert expanded[1] is None
    assert_matches(expanded[2], np.zeros(0, dtype=np.int64))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("rotation_only", "rotation_only"),
        ("rotation", "rotation_only"),
        ("full_parent", "full_parent"),
        ("1", None),
        ("default", None),
    ],
)
def test_local_adaptive_pass2_denominator_support_mode(monkeypatch, value, expected):
    monkeypatch.setenv("RELAX_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT", value)

    from relax.refinement.refinement_options import ScoringVariants

    assert ScoringVariants.from_environ().local_adaptive_pass2.denominator_mode == expected


def test_build_local_adaptive_pass2_hypothesis_layout_accepts_int64_packed_samples():
    parent_order = 7
    oversampling_order = 1
    n_coarse_trans = 49
    parent_id = np.int64(50_000_000)
    coarse_trans = np.int64(17)
    significant_sample = parent_id * np.int64(n_coarse_trans) + coarse_trans
    coarse_translations = np.stack(
        [np.arange(n_coarse_trans, dtype=np.float32), np.zeros(n_coarse_trans, dtype=np.float32)],
        axis=1,
    )
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_order),
        n_pixels=12 * (2**parent_order) ** 2,
        n_psi=rotation_grid_n_in_planes(parent_order),
        rotation_offsets=np.array([0, 1], dtype=np.int64),
        rotation_ids_flat=np.array([parent_id], dtype=np.int32),
        rotations_flat=np.eye(3, dtype=np.float32)[None],
        rotation_log_priors_flat=np.array([-2.5], dtype=np.float32),
        rotation_counts=np.array([1], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.zeros((1, n_coarse_trans), dtype=np.float32),
    )

    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [np.array([significant_sample], dtype=np.int64)],
        parent_order,
        oversampling_order=oversampling_order,
        translation_step=1.0,
    )

    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        1.0,
        oversampling_order=oversampling_order,
    )
    _, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
        np.array([parent_id], dtype=np.int64),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    np.testing.assert_allclose(layout.translation_grid, fine_translations, rtol=1e-6, atol=1e-6)
    assert_matches(layout.rotation_ids_flat, child_ids.astype(np.int32))
    np.testing.assert_allclose(layout.rotation_log_priors_flat, np.full(parent_map.shape, -2.5, dtype=np.float32))
    expected_mask = np.broadcast_to(
        fine_translation_parent[None, :] == int(coarse_trans),
        layout.sample_mask_rows().shape,
    )
    assert_matches(layout.sample_mask_rows(), expected_mask)


def test_build_pass2_hypothesis_layout_accepts_int64_packed_samples():
    parent_order = 7
    oversampling_order = 1
    n_coarse_trans = 49
    parent_id = np.int64(50_000_000)
    coarse_trans = np.int64(17)
    significant_sample = parent_id * np.int64(n_coarse_trans) + coarse_trans
    coarse_translations = np.stack(
        [np.arange(n_coarse_trans, dtype=np.float32), np.zeros(n_coarse_trans, dtype=np.float32)],
        axis=1,
    )

    layout = build_pass2_hypothesis_layout(
        [np.array([significant_sample], dtype=np.int64)],
        rotation_grid_size(parent_order),
        n_coarse_trans,
        parent_order,
        coarse_translations,
        oversampling_order=oversampling_order,
        translation_step=1.0,
    )

    _, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
        np.array([parent_id], dtype=np.int64),
        parent_order,
        oversampling_order=oversampling_order,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    _, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        1.0,
        oversampling_order=oversampling_order,
    )
    assert_matches(layout.rotation_ids_flat, child_ids.astype(np.int32))
    assert_matches(layout.rotation_posterior_ids_flat, np.full(parent_map.shape, parent_id, dtype=np.int32))
    expected_mask = np.broadcast_to(
        fine_translation_parent[None, :] == int(coarse_trans),
        layout.sample_mask_rows().shape,
    )
    assert_matches(layout.sample_mask_rows(), expected_mask)


def test_pass2_layout_builders_do_not_allocate_global_rotation_lookup_tables(monkeypatch):
    n_global_rotations = 1_000_000
    original_full = local_layout_module.np.full

    def guarded_full(shape, *args, **kwargs):
        if shape == n_global_rotations or shape == (n_global_rotations,):
            raise AssertionError("pass-2 layout builder allocated a dense global rotation lookup table")
        return original_full(shape, *args, **kwargs)

    monkeypatch.setattr(local_layout_module.np, "full", guarded_full)

    parent_order = 0
    coarse_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parent_layout = LocalHypothesisLayout(
        n_global_rotations=n_global_rotations,
        n_pixels=12,
        n_psi=1,
        rotation_offsets=np.array([0, 2], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1], dtype=np.int32),
        rotations_flat=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        rotation_log_priors_flat=np.array([0.0, -2.0], dtype=np.float32),
        rotation_counts=np.array([2], dtype=np.int32),
        translation_grid=coarse_translations,
        translation_log_priors=np.array([[0.0, -1.0]], dtype=np.float32),
    )
    adaptive_layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        [np.array([0, 3], dtype=np.int32)],
        parent_order,
        oversampling_order=1,
        translation_step=2.0,
    )
    _, parent_map, _ = get_oversampled_rotation_grid_from_samples(
        np.array([0, 1], dtype=np.int32),
        parent_order,
        oversampling_order=1,
        return_rotation_indices=True,
        rotation_index_order="recovar",
    )
    np.testing.assert_allclose(
        adaptive_layout.rotation_log_priors_flat,
        np.array([0.0, -2.0], dtype=np.float32)[parent_map],
    )
    assert adaptive_layout.sample_mask_rows().shape == (16, 8)

    pass2_layout = build_pass2_hypothesis_layout(
        [np.array([0, 3], dtype=np.int32)],
        n_coarse_rotations=n_global_rotations,
        n_coarse_translations=2,
        nside_level=0,
        translations=coarse_translations,
        oversampling_order=1,
        translation_step=2.0,
        rotation_log_prior=np.arange(rotation_grid_size(parent_order), dtype=np.float32),
    )
    assert_matches(pass2_layout.rotation_posterior_ids_flat, np.array([0, 1], dtype=np.int32)[parent_map])
    assert_matches(pass2_layout.sample_mask_rows(), adaptive_layout.sample_mask_rows())


def test_build_local_hypothesis_layout_factorized_matches_per_image_selector():
    healpix_order = 3
    grid_metadata = build_local_search_grid_metadata(healpix_order)
    rotation_grid = get_relion_rotation_grid(healpix_order).astype(np.float32)
    prior_eulers = np.array(
        [
            [12.0, 40.0, 3.0],
            [91.0, 65.0, 29.0],
            [177.0, 23.0, 144.0],
        ],
        dtype=np.float32,
    )

    layout = build_local_hypothesis_layout(
        prior_eulers,
        rotation_grid,
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=healpix_order,
        translations=np.zeros((9, 2), dtype=np.float32),
        prior_translations=np.zeros((3, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata=grid_metadata,
    )

    for image_idx in range(prior_eulers.shape[0]):
        local_ids_ref, local_log_prior_ref = get_local_rotation_grid_fast(
            prior_eulers[image_idx : image_idx + 1],
            np.deg2rad(7.5),
            np.deg2rad(7.5),
            healpix_order,
            sigma_cutoff=3.0,
            per_image=True,
            grid_metadata=grid_metadata,
        )
        start = int(layout.rotation_offsets[image_idx])
        stop = int(layout.rotation_offsets[image_idx + 1])
        assert_matches(layout.rotation_ids_flat[start:stop], np.asarray(local_ids_ref, dtype=np.int32))
        np.testing.assert_allclose(
            layout.rotation_log_priors_flat[start:stop],
            np.asarray(local_log_prior_ref[0], dtype=np.float32),
        )


def test_build_local_hypothesis_layout_parent_expands_relion_coarse_support():
    parent_order = 2
    fine_order = 3
    oversampling_order = fine_order - parent_order
    parent_metadata = build_local_search_grid_metadata(parent_order)
    fine_metadata = build_local_search_grid_metadata(fine_order)
    prior_eulers = np.array(
        [
            [12.0, 40.0, 3.0],
            [91.0, 65.0, 29.0],
        ],
        dtype=np.float32,
    )
    translations = np.zeros((3, 2), dtype=np.float32)
    prior_translations = np.zeros((prior_eulers.shape[0], 2), dtype=np.float32)
    sigma_rot = np.deg2rad(7.5)
    sigma_psi = np.deg2rad(7.5)

    parent_layout = build_local_hypothesis_layout(
        prior_eulers,
        None,
        sigma_rot=sigma_rot,
        sigma_psi=sigma_psi,
        healpix_order=parent_order,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata=parent_metadata,
    )
    expanded_layout = build_local_hypothesis_layout(
        prior_eulers,
        None,
        sigma_rot=sigma_rot,
        sigma_psi=sigma_psi,
        healpix_order=fine_order,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata=fine_metadata,
        local_parent_oversampling_order=oversampling_order,
    )

    assert expanded_layout.n_global_rotations == rotation_grid_size(fine_order)
    assert_matches(expanded_layout.rotation_counts, parent_layout.rotation_counts * 8)

    for image_idx in range(prior_eulers.shape[0]):
        p0, p1 = parent_layout.rotation_offsets[image_idx : image_idx + 2]
        c0, c1 = expanded_layout.rotation_offsets[image_idx : image_idx + 2]
        parent_ids = parent_layout.rotation_ids_flat[p0:p1]
        parent_log_prior = parent_layout.rotation_log_priors_flat[p0:p1]
        child_rots, parent_map, child_ids = get_oversampled_rotation_grid_from_samples(
            parent_ids,
            parent_order,
            oversampling_order=oversampling_order,
            return_rotation_indices=True,
            rotation_index_order="recovar",
        )

        assert_matches(expanded_layout.rotation_ids_flat[c0:c1], child_ids.astype(np.int32))
        np.testing.assert_allclose(expanded_layout.rotation_log_priors_flat[c0:c1], parent_log_prior[parent_map])
        np.testing.assert_allclose(expanded_layout.rotations_flat[c0:c1], child_rots, rtol=1e-6, atol=1e-6)


def test_score_half_local_parent_layout_ignores_global_rotation_prior_for_adaptive_pass2(monkeypatch, rng):
    dataset = MockDataset(2, rng)
    captured = {}

    class StopAfterParentLayout(Exception):
        pass

    def fake_build_local_hypothesis_layout(
        prior_rotations,
        rotation_grid_rotations,
        sigma_rot,
        sigma_psi,
        healpix_order,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        voxel_size,
        *,
        grid_metadata,
        translation_prior_reference_translations=None,
        rotation_log_prior=None,
        rotation_grid_random_perturbation=0.0,
        rotation_grid_angular_sampling_deg=None,
        dtype=np.float32,
    ):
        captured["rotation_log_prior"] = (
            None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=np.float32).copy()
        )
        raise StopAfterParentLayout

    monkeypatch.setattr(half_scoring, "build_local_search_grid_metadata", lambda _order, *, symmetry='C1': {})
    monkeypatch.setattr(half_scoring, "build_local_hypothesis_layout", fake_build_local_hypothesis_layout)

    with pytest.raises(StopAfterParentLayout):
        half_scoring._score_half_local(*local_half_owners(
            k=0,
            experiment_dataset=dataset,
            means_k=jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
            noise_variance_k=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            previous_best_rotation_eulers_k=np.zeros((dataset.n_units, 3), dtype=np.float32),
            local_search_rotations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
            local_search_order=1,
            sigma_rot=np.deg2rad(1.0),
            sigma_psi=np.deg2rad(1.0),
            current_translations=np.zeros((1, 2), dtype=np.float32),
            base_translations=np.zeros((1, 2), dtype=np.float32),
            trans_prior_center=np.zeros((dataset.n_units, 2), dtype=np.float32),
            trans_prior_center_for_engine=np.zeros((dataset.n_units, 2), dtype=np.float32),
            current_sigma_offset_angstrom=1.0,
            disc_type="linear_interp",
            cs_for_engine=None,
            local_pass1_current_size=4,
            image_corrections_k=None,
            scale_corrections_k=None,
            translation_search_base=None,
            max_significants=None,
            iteration=3,
            local_search_random_perturbation=0.0,
            local_search_angular_sampling_deg=relion_angular_sampling_deg(1),
            local_parent_oversampling_order=1,
            diagnostic_score_only=False,
            replay_prior_translations=None,
            collect_local_search_profile=False,
            local_profile_history=[],
        ))

    assert captured["rotation_log_prior"] is None


def test_score_half_local_forwards_mstep_grid(monkeypatch, rng):
    """Forward the mean and M-step grids to local scoring (local searches are K=1 only)."""
    dataset = MockDataset(1, rng)
    score_grid = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    mstep_grid = np.repeat((5.0 * np.eye(3, dtype=np.float32))[None], 2, axis=0)
    captured = {}

    class DispatchCaptured(Exception):
        pass

    def fake_run_local_search_iteration(data, grid, kernel, support):
        captured.update(data=data, grid=grid, kernel=kernel, support=support)
        raise DispatchCaptured

    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=True),
    )
    monkeypatch.setattr(
        scoring_policy,
        "DENSE_PRECISION",
        replace(scoring_policy.DENSE_PRECISION, use_float64_projections=True),
    )
    monkeypatch.delenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", raising=False)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", fake_run_local_search_iteration)
    with pytest.raises(DispatchCaptured):
        half_scoring._score_half_local(*local_half_owners(
            k=0,
            experiment_dataset=dataset,
            means_k=jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
            noise_variance_k=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            previous_best_rotation_eulers_k=np.zeros((dataset.n_units, 3), dtype=np.float32),
            local_search_rotations=score_grid,
            local_search_mstep_rotations=mstep_grid,
            local_search_order=0,
            sigma_rot=np.deg2rad(1.0),
            sigma_psi=np.deg2rad(1.0),
            current_translations=np.zeros((1, 2), dtype=np.float32),
            base_translations=np.zeros((1, 2), dtype=np.float32),
            trans_prior_center=np.zeros((dataset.n_units, 2), dtype=np.float32),
            trans_prior_center_for_engine=np.zeros((dataset.n_units, 2), dtype=np.float32),
            current_sigma_offset_angstrom=1.0,
            disc_type="linear_interp",
            cs_for_engine=None,
            local_pass1_current_size=None,
            image_corrections_k=None,
            scale_corrections_k=None,
            translation_search_base=None,
            max_significants=-1,
            iteration=3,
            local_search_random_perturbation=0.0,
            local_search_angular_sampling_deg=relion_angular_sampling_deg(0),
            local_parent_oversampling_order=0,
            diagnostic_score_only=False,
            replay_prior_translations=None,
            collect_local_search_profile=False,
            local_profile_history=[],
        ))

    assert captured["data"].mean.shape == (VOLUME_SIZE,)
    assert_matches(captured["grid"].rotation_grid_mstep_rotations, mstep_grid)
    assert captured["grid"].generate_relion_mstep_rotations is True
    assert captured["kernel"].use_float64_scoring is True
    assert captured["kernel"].use_float64_projections is True


def test_build_local_hypothesis_layout_parent_expands_translation_grid_and_priors():
    parent_order = 1
    fine_order = 2
    oversampling_order = fine_order - parent_order
    grid_metadata = build_local_search_grid_metadata(fine_order)
    prior_eulers = np.array(
        [
            [12.0, 40.0, 3.0],
            [91.0, 65.0, 29.0],
        ],
        dtype=np.float32,
    )
    translations = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    prior_translations = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    reference_translations = np.array(
        [
            [0.25, 0.25],
            [2.25, 0.25],
            [0.25, 2.25],
        ],
        dtype=np.float32,
    )

    layout = build_local_hypothesis_layout(
        prior_eulers,
        None,
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=fine_order,
        translations=translations,
        prior_translations=prior_translations,
        sigma_offset_angstrom=1.25,
        offset_range_pixels=None,
        voxel_size=1.0,
        grid_metadata=grid_metadata,
        translation_prior_reference_translations=reference_translations,
        local_parent_oversampling_order=oversampling_order,
    )

    fine_translations, translation_parent = get_oversampled_translation_grid(
        translations,
        2.0,
        oversampling_order=oversampling_order,
    )
    coarse_prior = make_relion_translation_log_prior(
        reference_translations,
        1.0,
        1.25,
        prior_translations,
        offset_range_pixels=None,
    )

    np.testing.assert_allclose(layout.translation_grid, fine_translations, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        layout.translation_log_priors,
        coarse_prior[:, translation_parent],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("builder_name", ["parent_expanded", "adaptive_local"])
def test_lazy_pass2_layouts_request_aligned_relion_mstep_rotations(monkeypatch, builder_name):
    calls = []

    def fake_oversampled_rotations(parent_ids, *args, **kwargs):
        _ = args
        parent_ids = np.asarray(parent_ids, dtype=np.int64)
        calls.append(dict(kwargs))
        n_rows = int(parent_ids.size)
        score = np.broadcast_to(3.0 * np.eye(3, dtype=np.float32), (n_rows, 3, 3)).copy()
        mstep = np.broadcast_to(7.0 * np.eye(3, dtype=np.float32), (n_rows, 3, 3)).copy()
        parent_map = np.arange(n_rows, dtype=np.int64)
        child_ids = parent_ids.copy()
        outputs = [score, parent_map]
        if kwargs.get("return_rotation_indices"):
            outputs.append(child_ids)
        if kwargs.get("return_mstep_rotations"):
            outputs.append(mstep)
        if kwargs.get("return_source_eulers"):
            # The adaptive local builder also requests the native RFLOAT source Euler angles.
            outputs.append(np.zeros((n_rows, 3), dtype=np.float64))
        return tuple(outputs)

    monkeypatch.setattr(local_layout_module, "get_oversampled_rotation_grid_from_samples", fake_oversampled_rotations)
    # The adaptive builder expands every image's parents in one uncached call.
    monkeypatch.setattr(local_layout_module, "_compute_oversampled_rotation_grid", fake_oversampled_rotations)
    translations = np.array([[0.0, 0.0]], dtype=np.float32)
    if builder_name == "parent_expanded":
        layout = build_local_hypothesis_layout(
            np.zeros((1, 3), dtype=np.float32),
            None,
            0.0,
            0.0,
            1,
            translations,
            np.zeros((1, 2), dtype=np.float32),
            1.0,
            None,
            1.0,
            grid_metadata=build_local_search_grid_metadata(1),
            local_parent_oversampling_order=1,
            generate_relion_mstep_rotations=True,
        )
    else:
        parent_layout = LocalHypothesisLayout(
            n_global_rotations=rotation_grid_size(0),
            n_pixels=12,
            n_psi=1,
            rotation_offsets=np.array([0, 1], dtype=np.int64),
            rotation_ids_flat=np.array([0], dtype=np.int32),
            rotations_flat=np.eye(3, dtype=np.float32)[None],
            rotation_log_priors_flat=np.zeros(1, dtype=np.float32),
            rotation_counts=np.ones(1, dtype=np.int32),
            translation_grid=translations,
            translation_log_priors=np.zeros((1, 1), dtype=np.float32),
        )
        layout = build_local_adaptive_pass2_hypothesis_layout(
            parent_layout,
            [None],
            0,
            oversampling_order=1,
        )
    assert calls and all(call["return_mstep_rotations"] is True for call in calls)
    assert_matches(
        layout.rotations_flat,
        np.broadcast_to(3.0 * np.eye(3, dtype=np.float32), layout.rotations_flat.shape),
    )
    assert_matches(
        layout.mstep_rotations_flat,
        np.broadcast_to(7.0 * np.eye(3, dtype=np.float32), layout.rotations_flat.shape),
    )


def test_build_local_hypothesis_layout_factorized_chunking_preserves_support(monkeypatch):
    healpix_order = 3
    grid_metadata = build_local_search_grid_metadata(healpix_order)
    rotation_grid = get_relion_rotation_grid(healpix_order).astype(np.float32)
    prior_eulers = np.array(
        [
            [12.0, 40.0, 3.0],
            [91.0, 65.0, 29.0],
            [177.0, 23.0, 144.0],
            [240.0, 81.0, 271.0],
        ],
        dtype=np.float32,
    )
    kwargs = dict(
        sigma_rot=np.deg2rad(7.5),
        sigma_psi=np.deg2rad(7.5),
        healpix_order=healpix_order,
        translations=np.zeros((9, 2), dtype=np.float32),
        prior_translations=np.zeros((prior_eulers.shape[0], 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=1.0,
        voxel_size=1.0,
        grid_metadata=grid_metadata,
    )

    monkeypatch.delenv("RELAX_LOCAL_SELECTOR_CHUNK_SIZE", raising=False)
    full_layout = build_local_hypothesis_layout(prior_eulers, rotation_grid, **kwargs)
    monkeypatch.setenv("RELAX_LOCAL_SELECTOR_CHUNK_SIZE", "1")
    chunked_layout = build_local_hypothesis_layout(prior_eulers, rotation_grid, **kwargs)

    assert_matches(chunked_layout.rotation_offsets, full_layout.rotation_offsets)
    assert_matches(chunked_layout.rotation_counts, full_layout.rotation_counts)
    assert_matches(chunked_layout.rotation_ids_flat, full_layout.rotation_ids_flat)
    np.testing.assert_allclose(chunked_layout.rotation_log_priors_flat, full_layout.rotation_log_priors_flat)


def test_selected_rotation_matrices_match_full_perturbed_grid():
    healpix_order = 2
    random_perturbation = 0.25
    angular_sampling_deg = relion_angular_sampling_deg(healpix_order)
    grid_metadata = build_local_search_grid_metadata(healpix_order)
    full_eulers = get_relion_rotation_grid_eulers(healpix_order).astype(np.float32)
    full_perturbed_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        full_eulers,
        random_perturbation,
        angular_sampling_deg,
    )
    full_mstep_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        _get_relion_rotation_grid_eulers_float64(healpix_order),
        random_perturbation,
        angular_sampling_deg,
    )
    rotation_ids = np.array([0, 3, 17, rotation_grid_size(healpix_order) - 1], dtype=np.int32)

    selected_rotations = selected_rotation_matrices(
        rotation_ids,
        None,
        grid_metadata,
        random_perturbation=random_perturbation,
        angular_sampling_deg=angular_sampling_deg,
    )

    np.testing.assert_allclose(
        selected_rotations,
        np.asarray(full_perturbed_rotations, dtype=np.float32)[rotation_ids],
        atol=1e-6,
        rtol=1e-6,
    )
    selected_mstep_rotations = local_layout_module._selected_mstep_rotation_matrices(
        rotation_ids,
        None,
        grid_metadata,
        random_perturbation=random_perturbation,
        angular_sampling_deg=angular_sampling_deg,
    )
    assert_matches(selected_mstep_rotations, full_mstep_rotations[rotation_ids])


def test_relion_mstep_generation_keeps_source_eulers_float64_until_host_inverse():
    healpix_order = 0
    source_eulers = _get_relion_rotation_grid_eulers_float64(healpix_order)
    public_eulers = get_relion_rotation_grid_eulers(healpix_order)
    angular_sampling_deg = relion_angular_sampling_deg(healpix_order)

    exact_mstep, _ = apply_relion_rotation_perturbation_to_eulers(
        source_eulers,
        0.25,
        angular_sampling_deg,
    )
    late_mstep, _ = apply_relion_rotation_perturbation_to_eulers(
        source_eulers.astype(np.float32),
        0.25,
        angular_sampling_deg,
    )

    assert source_eulers.dtype == np.float64
    assert public_eulers.dtype == np.float32
    assert_matches(public_eulers, source_eulers.astype(np.float32))
    assert np.any(exact_mstep != late_mstep)


def test_exact_local_fine_grid_precompute_auto_policy():
    from relax.refinement.local_sampling import _precompute_exact_local_fine_grid_enabled

    assert _precompute_exact_local_fine_grid_enabled(5, "C1")
    assert not _precompute_exact_local_fine_grid_enabled(6, "C1")


def test_exact_local_bucket_radix_can_collapse_adjacent_power_two_shapes(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_BUCKET_RADIX_ENV, "4")

    assert exact_bucket_rotation_size(32, 5000) == 64
    assert exact_bucket_rotation_size(64, 5000) == 64
    assert exact_bucket_rotation_size(128, 5000) == 256
    assert exact_bucket_rotation_size(256, 5000) == 256
    assert exact_bucket_rotation_size(512, 5000) == 1024
    assert exact_bucket_rotation_size(1024, 5000) == 1024


def test_bucket_local_hypothesis_layout_coarsens_large_exact_neighborhoods_without_4096_floor():
    layout = LocalHypothesisLayout(
        n_global_rotations=2000,
        n_pixels=768,
        n_psi=16,
        rotation_offsets=np.array([0, 1368, 2760, 4176], dtype=np.int64),
        rotation_ids_flat=np.arange(4176, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4176, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4176, dtype=np.float32),
        rotation_counts=np.array([1368, 1392, 1416], dtype=np.int32),
        translation_grid=np.zeros((9, 2), dtype=np.float32),
        translation_log_priors=np.zeros((3, 9), dtype=np.float32),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
    )

    bucket_sizes = sorted(int(bucket.bucket_rotation_count) for bucket in buckets)
    assert bucket_sizes == [2048]
    assert [int(bucket.bucket_image_count) for bucket in buckets] == [10]
    assert_matches(buckets[0].actual_rotation_counts, np.array([1368, 1392, 1416], dtype=np.int32))
    assert buckets[0].local_rotation_mask[0, :1368].all()
    assert not buckets[0].local_rotation_mask[0, 1368:].any()


def test_bucket_local_hypothesis_layout_quantum_env_can_request_finer_tail_shapes(monkeypatch):
    monkeypatch.setenv("RELAX_LOCAL_BUCKET_QUANTUM", "128")
    layout = LocalHypothesisLayout(
        n_global_rotations=2000,
        n_pixels=768,
        n_psi=16,
        rotation_offsets=np.array([0, 1368, 2760, 4176], dtype=np.int64),
        rotation_ids_flat=np.arange(4176, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4176, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4176, dtype=np.float32),
        rotation_counts=np.array([1368, 1392, 1416], dtype=np.int32),
        translation_grid=np.zeros((9, 2), dtype=np.float32),
        translation_log_priors=np.zeros((3, 9), dtype=np.float32),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
    )

    assert sorted(int(bucket.bucket_rotation_count) for bucket in buckets) == [1408, 1536]


def test_bucket_local_hypothesis_layout_batches_moderate_local_search_neighborhoods(monkeypatch):
    monkeypatch.delenv("RELAX_LOCAL_BUCKET_QUANTUM", raising=False)
    monkeypatch.delenv("RELAX_EXACT_LOCAL_BUCKET_QUANTUM", raising=False)
    n_images = 120
    rotation_counts = np.full(n_images, 198, dtype=np.int32)
    rotation_offsets = np.concatenate([[0], np.cumsum(rotation_counts)]).astype(np.int64)
    n_total = int(rotation_counts.sum())
    layout = LocalHypothesisLayout(
        n_global_rotations=2359296,
        n_pixels=12288,
        n_psi=192,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(n_total, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (n_total, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(n_total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((25, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, 25), dtype=np.float32),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=60,
        rotation_block_size=89,
        max_hypotheses_per_microbatch=1211,
    )

    assert {int(bucket.bucket_rotation_count) for bucket in buckets} == {256}
    assert {int(bucket.image_indices.shape[0]) for bucket in buckets} == {4}
    assert len(buckets) == 30
    assert sum(int(bucket.image_indices.shape[0]) * int(bucket.bucket_rotation_count) for bucket in buckets) == 30720


def test_bucket_local_hypothesis_layout_unify_env_collapses_shape_classes(monkeypatch):
    """RELAX_LOCAL_BUCKET_UNIFY=1 collapses ~13 unique bucket shape classes
    into one max-sized class so the JIT only compiles one shape per layout.
    Pins the 7.3× perf win measured on 50k/256 K=1 (commit 8e868d5e)."""

    rotation_counts = np.array([16, 16, 128, 256, 512, 1024, 1280, 1408], dtype=np.int32)
    rotation_ids = np.arange(int(rotation_counts.sum()), dtype=np.int32)
    rotation_offsets = np.concatenate([[0], np.cumsum(rotation_counts)]).astype(np.int64)
    n_total = int(rotation_counts.sum())
    layout = LocalHypothesisLayout(
        n_global_rotations=2000,
        n_pixels=768,
        n_psi=16,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids,
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (n_total, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(n_total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((4, 2), dtype=np.float32),
        translation_log_priors=np.zeros((len(rotation_counts), 4), dtype=np.float32),
    )

    monkeypatch.delenv("RELAX_LOCAL_BUCKET_UNIFY", raising=False)
    default_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
    )
    default_unique_sizes = sorted({int(b.bucket_rotation_count) for b in default_buckets})
    assert len(default_unique_sizes) >= 6  # power-of-2 spread

    monkeypatch.setenv("RELAX_LOCAL_BUCKET_UNIFY", "1")
    unified_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
    )
    unified_sizes = {int(b.bucket_rotation_count) for b in unified_buckets}
    assert unified_sizes == {max(default_unique_sizes)}
    assert len(unified_buckets) == 1
    # All images must remain represented exactly once.
    served_indices = np.sort(np.concatenate([b.image_indices for b in unified_buckets]))
    assert_matches(served_indices, np.arange(len(rotation_counts), dtype=np.int32))


def test_bucket_local_hypothesis_layout_unify_argument_collapses_shape_classes(monkeypatch):
    monkeypatch.delenv("RELAX_LOCAL_BUCKET_UNIFY", raising=False)
    rotation_counts = np.array([16, 128, 512, 1408], dtype=np.int32)
    rotation_ids = np.arange(int(rotation_counts.sum()), dtype=np.int32)
    rotation_offsets = np.concatenate([[0], np.cumsum(rotation_counts)]).astype(np.int64)
    layout = LocalHypothesisLayout(
        n_global_rotations=2000,
        n_pixels=768,
        n_psi=16,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids,
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (int(rotation_counts.sum()), 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(int(rotation_counts.sum()), dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((4, 2), dtype=np.float32),
        translation_log_priors=np.zeros((len(rotation_counts), 4), dtype=np.float32),
    )

    default_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
        unify_bucket_sizes=False,
    )
    explicit_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=10,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=65536,
        unify_bucket_sizes=True,
    )

    default_unique_sizes = sorted({int(b.bucket_rotation_count) for b in default_buckets})
    assert len(default_unique_sizes) > 1
    assert {int(b.bucket_rotation_count) for b in explicit_buckets} == {max(default_unique_sizes)}
    assert_matches(
        np.sort(np.concatenate([b.image_indices for b in explicit_buckets])),
        np.arange(len(rotation_counts), dtype=np.int32),
    )


def test_bucket_local_hypothesis_layout_can_preserve_physical_image_order(monkeypatch):
    monkeypatch.delenv("RELAX_LOCAL_BUCKET_UNIFY", raising=False)
    rotation_counts = np.array([16, 64, 16, 64], dtype=np.int32)
    rotation_offsets = np.concatenate([[0], np.cumsum(rotation_counts)]).astype(np.int64)
    n_total = int(rotation_counts.sum())
    layout = LocalHypothesisLayout(
        n_global_rotations=n_total,
        n_pixels=16,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(n_total, dtype=np.int32),
        rotations_flat=np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (n_total, 3, 3),
        ).copy(),
        rotation_log_priors_flat=np.zeros(n_total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((rotation_counts.size, 1), dtype=np.float32),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=4,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=1024,
        unify_bucket_sizes=False,
        preserve_image_order=True,
    )

    assert_matches(
        np.concatenate([bucket.image_indices for bucket in buckets]),
        np.arange(rotation_counts.size, dtype=np.int32),
    )
    assert [bucket.bucket_rotation_count for bucket in buckets] == [16, 64, 16, 64]


def test_bucket_local_hypothesis_layout_aligns_preserved_chunks_to_pool_three(monkeypatch):
    monkeypatch.delenv("RELAX_LOCAL_BUCKET_UNIFY", raising=False)
    rotation_counts = np.full(7, 16, dtype=np.int32)
    rotation_offsets = np.concatenate([[0], np.cumsum(rotation_counts)]).astype(np.int64)
    n_total = int(rotation_counts.sum())
    layout = LocalHypothesisLayout(
        n_global_rotations=n_total,
        n_pixels=16,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(n_total, dtype=np.int32),
        rotations_flat=np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (n_total, 3, 3),
        ).copy(),
        rotation_log_priors_flat=np.zeros(n_total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((rotation_counts.size, 1), dtype=np.float32),
    )

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=7,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=7 * 16,
        unify_bucket_sizes=False,
        preserve_image_order=True,
    )

    assert [bucket.image_indices.size for bucket in buckets] == [6, 1]
    assert_matches(
        np.concatenate([bucket.image_indices for bucket in buckets]),
        np.arange(7, dtype=np.int32),
    )


def test_relion_projector_indexed_centered_rows_match_full_window(rng):
    from relax.helpers.fourier_window import make_fourier_window_spec
    from relax.helpers.projection import compute_relion_projector_projections_block

    image_shape = (8, 8)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    window_spec = make_fourier_window_spec(image_shape, 6, n_half, include_recon_window=True)
    projector_half = (
        rng.standard_normal((8, 8, 5)) + 1j * rng.standard_normal((8, 8, 5))
    ).astype(np.complex64)
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (3, 3, 3))

    full, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half),
        rotations,
        image_shape,
        r_max=4,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=6,
    )
    indexed, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half),
        rotations,
        image_shape,
        r_max=4,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=6,
        pixel_indices=window_spec.projection_indices,
    )

    np.testing.assert_allclose(
        np.asarray(indexed),
        np.asarray(full)[:, window_spec.projection_indices_np],
        atol=1e-5,
        rtol=1e-5,
    )


def test_relion_projector_texture_full_embeds_positive_x_half():
    from relax.helpers.projection import relion_projector_half_to_texture_full

    projector_half = (
        np.arange(5 * 5 * 3, dtype=np.float32).reshape(5, 5, 3)
        + 1j * np.arange(5 * 5 * 3, dtype=np.float32).reshape(5, 5, 3)[::-1]
    ).astype(np.complex64)
    full = np.asarray(relion_projector_half_to_texture_full(jnp.asarray(projector_half)))

    assert full.shape == (5, 5, 5)
    assert_matches(full[2:], np.transpose(projector_half, (2, 1, 0)))
    assert_matches(full[:2], np.zeros((2, 5, 5), dtype=np.complex64))


def test_relion_projector_texture_route_defaults_on_and_can_be_disabled(monkeypatch):
    from relax.helpers import projection as projection_helpers

    projector_half = jnp.ones((5, 5, 3), dtype=jnp.complex64)
    rotations = jnp.eye(3, dtype=jnp.float32)[None]
    calls = []

    monkeypatch.delenv("RELAX_RELION_PROJECTOR_TEXTURE_INTERP", raising=False)
    monkeypatch.setattr(projection_helpers, "_cuda_projection_available", lambda: True)
    assert projection_helpers._relion_projector_texture_enabled(
        projector_half,
        r_max=1,
        padding_factor=1,
    )
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_TEXTURE_INTERP", "1")

    def fake_texture(projector, rotations_block, image_shape, **kwargs):
        calls.append((tuple(projector.shape), tuple(rotations_block.shape), tuple(image_shape), dict(kwargs)))
        return jnp.full((rotations_block.shape[0], 4 * 3), 2.0 + 1.0j, dtype=jnp.complex64)

    monkeypatch.setattr(projection_helpers, "_project_relion_projector_texture", fake_texture)
    got, _ = projection_helpers.compute_relion_projector_projections_block(
        projector_half,
        rotations,
        (4, 4),
        r_max=1,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=2,
    )
    assert_matches(np.asarray(got), np.full((1, 12), 2.0 + 1.0j, dtype=np.complex64))
    assert calls == [
        (
            (5, 5, 3),
            (1, 3, 3),
            (4, 4),
            {"r_max": 1, "projector_output_size": 2},
        )
    ]

    monkeypatch.setenv("RELAX_RELION_PROJECTOR_TEXTURE_INTERP", "0")
    monkeypatch.setattr(
        projection_helpers,
        "project_relion_projector_half_spectrum_centered_rows",
        lambda *args, **kwargs: jnp.full((1, 12), 7.0 + 0.0j, dtype=jnp.complex64),
    )
    fallback, _ = projection_helpers.compute_relion_projector_projections_block(
        projector_half,
        rotations,
        (4, 4),
        r_max=1,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=2,
    )
    assert_matches(np.asarray(fallback), np.full((1, 12), 7.0 + 0.0j, dtype=np.complex64))
    assert len(calls) == 1

    monkeypatch.setenv("RELAX_RELION_PROJECTOR_TEXTURE_INTERP", "1")
    explicit_fallback, _ = projection_helpers.compute_relion_projector_projections_block(
        projector_half,
        rotations,
        (4, 4),
        r_max=1,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=2,
        relion_texture_interp=False,
    )
    assert_matches(
        np.asarray(explicit_fallback),
        np.full((1, 12), 7.0 + 0.0j, dtype=np.complex64),
    )
    assert len(calls) == 1


def test_global_pass1_relion_projector_texture_defaults_to_texture(monkeypatch):
    from relax.scoring import pass1_plan

    monkeypatch.delenv("RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP", raising=False)
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_TEXTURE_INTERP", "1")
    assert pass1_plan.global_pass1_relion_projector_texture_enabled()

    monkeypatch.setenv("RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP", "1")
    assert pass1_plan.global_pass1_relion_projector_texture_enabled()

    monkeypatch.setenv("RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP", "invalid")
    with pytest.raises(ValueError, match="RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP"):
        pass1_plan.global_pass1_relion_projector_texture_enabled()


def test_texture_centered_crop_masks_current_image_disk():
    from relax.helpers.projection import _texture_centered_crop_to_full

    crop = jnp.ones((1, 4 * 3), dtype=jnp.complex64)
    got = np.asarray(
        _texture_centered_crop_to_full(
            crop,
            image_shape=(4, 4),
            projector_output_size=4,
            mask_current_image_disk=True,
        )
    ).reshape(1, 4, 3)
    expected = np.ones((1, 4, 3), dtype=np.complex64)
    expected[:, 0, 1:] = 0.0
    expected[:, 1, 2] = 0.0
    expected[:, 3, 2] = 0.0
    assert_matches(got, expected)

    unmasked = np.asarray(
        _texture_centered_crop_to_full(
            crop,
            image_shape=(4, 4),
            projector_output_size=4,
            mask_current_image_disk=False,
        )
    ).reshape(1, 4, 3)
    assert_matches(unmasked, np.ones((1, 4, 3), dtype=np.complex64))


@pytest.mark.parametrize(
    ("image_size", "crop_size"),
    [(8, 6), (8, 8)],
)
def test_texture_centered_crop_direct_indices_match_full_scatter(
    rng,
    image_size,
    crop_size,
):
    from relax.helpers.projection import _texture_centered_crop_at_indices, _texture_centered_crop_to_full

    crop_pixels = crop_size * (crop_size // 2 + 1)
    crop = (
        rng.standard_normal((3, crop_pixels))
        + 1j * rng.standard_normal((3, crop_pixels))
    ).astype(np.complex64)
    full = np.asarray(
        _texture_centered_crop_to_full(
            jnp.asarray(crop),
            image_shape=(image_size, image_size),
            projector_output_size=crop_size,
        )
    )

    if crop_size == image_size:
        indices = np.arange(full.shape[1], dtype=np.int32)
    else:
        crop_rows = np.arange(crop_size, dtype=np.int32)
        crop_ky = np.where(crop_rows == 0, crop_size // 2, crop_rows - crop_size // 2)
        crop_cols = np.arange(crop_size // 2 + 1, dtype=np.int32)
        full_rows = crop_ky + image_size // 2
        indices = (
            full_rows[:, None] * (image_size // 2 + 1) + crop_cols[None, :]
        ).reshape(-1)

    direct = np.asarray(
        _texture_centered_crop_at_indices(
            jnp.asarray(crop),
            jnp.asarray(indices),
            image_shape=(image_size, image_size),
            projector_output_size=crop_size,
        )
    )
    assert_matches(direct, full[:, indices])


def test_texture_projector_compact_indices_bypass_full_scatter(monkeypatch):
    from relax.helpers import projection as projection_helpers

    requested = jnp.asarray([6, 7, 9], dtype=jnp.int32)
    monkeypatch.setattr(projection_helpers, "_relion_projector_texture_enabled", lambda *args, **kwargs: True)

    def fake_texture(projector, rotations, image_shape, **kwargs):
        assert_matches(np.asarray(kwargs.pop("pixel_indices")), np.asarray(requested))
        assert kwargs == {"r_max": 1, "projector_output_size": 2}
        return jnp.full((rotations.shape[0], requested.size), 3.0 + 2.0j, dtype=jnp.complex64)

    monkeypatch.setattr(projection_helpers, "_project_relion_projector_texture", fake_texture)
    got, _ = projection_helpers.compute_relion_projector_projections_block(
        jnp.ones((5, 5, 3), dtype=jnp.complex64),
        jnp.eye(3, dtype=jnp.float32)[None],
        (4, 4),
        r_max=1,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=2,
        pixel_indices=requested,
    )
    assert_matches(
        np.asarray(got),
        np.full((1, requested.size), 3.0 + 2.0j, dtype=np.complex64),
    )


def test_texture_projector_compact_implementation_never_builds_full_box(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers import projection as projection_helpers

    crop = jnp.asarray([[0.0 + 1.0j, 1.0 + 2.0j, 2.0 + 3.0j, 3.0 + 4.0j]])
    requested = jnp.asarray([6, 7, 9], dtype=jnp.int32)
    monkeypatch.setattr(
        projection_helpers,
        "relion_projector_half_to_texture_full",
        lambda value: jnp.zeros((5, 5, 5), dtype=jnp.complex64),
    )
    monkeypatch.setattr(
        projection_helpers,
        "project_half_spectrum",
        lambda *args, **kwargs: crop,
    )
    monkeypatch.setattr(
        em_cuda_kernels,
        "project_relion_half_capacity",
        lambda *args, **kwargs: crop,
    )
    monkeypatch.setattr(
        projection_helpers,
        "_texture_centered_crop_to_full",
        lambda *args, **kwargs: pytest.fail("compact projection built the full image box"),
    )

    got = projection_helpers._project_relion_projector_texture(
        jnp.zeros((5, 5, 3), dtype=jnp.complex64),
        jnp.eye(3, dtype=jnp.float32)[None],
        (4, 4),
        r_max=1,
        projector_output_size=2,
        pixel_indices=requested,
    )
    assert_matches(
        np.asarray(got),
        np.asarray(crop)[:, [2, 3, 0]],
    )


def test_relion_projector_cache_reuses_cached_projector_data(monkeypatch, tmp_path):
    import relax.relion.relion_projector_setup as projector_setup

    calls = []

    def fake_projector_builder(
        refs_real, *, current_size, padding_factor, projector_data_dtype, gridding_kernel,
    ):
        assert gridding_kernel == "radial"
        assert projector_data_dtype == "complex128"
        calls.append(np.asarray(refs_real).copy())
        projector_half = np.full((refs_real.shape[0], 3, 3, 2), 7.0 + len(calls), dtype=np.complex64)
        power = np.full((refs_real.shape[0], 3), 0.5 + len(calls), dtype=np.float64)
        return projector_half, power, int(current_size // 2)

    monkeypatch.setattr(projector_setup, "reference_to_relion_projector_half_maps_and_power", fake_projector_builder)
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_CACHE_DIR", str(tmp_path))

    mean_ft = np.zeros((4, 4, 4), dtype=np.complex64)
    mean_ft[0, 0, 0] = 1.0
    first = projector_preparation.prepare_scoring_projector(
        mean_ft.reshape(-1),
        volume_shape=(4, 4, 4),
        current_size=4,
        padding_factor=2,
        n_classes=1,
    )
    second = projector_preparation.prepare_scoring_projector(
        mean_ft.reshape(-1),
        volume_shape=(4, 4, 4),
        current_size=4,
        padding_factor=2,
        n_classes=1,
    )

    assert len(calls) == 1
    assert first.r_max == second.r_max == 2
    assert_matches(first.data, second.data)
    # The class power spectra round-trip through the cache with the slabs.
    assert_matches(first.power_spectrum, second.power_spectrum)
    assert (tmp_path / "SAFE_TO_DELETE").exists()
    assert len(list(tmp_path.glob("projector_*.npz"))) == 1


def test_relion_projector_direct_real_reference_bypasses_fourier_roundtrip(monkeypatch):
    import relax.relion.relion_projector_setup as projector_setup

    captured_real = []

    def fake_projector_builder(
        refs_real, *, current_size, padding_factor, projector_data_dtype, gridding_kernel,
    ):
        assert gridding_kernel == "radial"
        assert projector_data_dtype == "complex128"
        captured_real.append(np.asarray(refs_real).copy())
        projector_half = np.ones((refs_real.shape[0], 3, 3, 2), dtype=np.complex64)
        return projector_half, np.ones((refs_real.shape[0], 3)), int(current_size // 2)

    monkeypatch.setattr(
        projector_setup,
        "reference_to_relion_projector_half_maps_and_power",
        fake_projector_builder,
    )
    mean_ft = np.zeros((4, 4, 4), dtype=np.complex64)
    mean_ft[0, 0, 0] = 7.0 + 3.0j
    exact_real = np.arange(64, dtype=np.float64).reshape(1, 4, 4, 4) / 17.0

    projector_preparation.prepare_scoring_projector(
        mean_ft.reshape(-1),
        volume_shape=(4, 4, 4),
        current_size=4,
        padding_factor=2,
        n_classes=1,
        real_references=exact_real,
    )

    assert len(captured_real) == 1
    assert_matches(captured_real[0], exact_real)


def test_numbered_projector_reuse_preserves_previous_projector_release(
    half_datasets, init_volume, rotations, translations, monkeypatch,
):
    """The projectors a consumer needs stay alive, and the previous iteration's are gone before a new build.

    Half 1's scoring projector is the expected-accuracy estimate's projector of the same references (built
    first and reused, not rebuilt); each half is scored with the projector built for it in this iteration.
    The previous iteration's projectors are released before this iteration's first scoring build (since the
    build moved into ``build_numbered_projectors`` the previous half-2 slab no longer survives into the
    half-1 build: one slab less at that peak).
    """
    import weakref

    old_projectors = []
    built = {}
    events = []
    original_prepare = projector_preparation.prepare_scoring_projector

    class LifetimeChecked(Exception):
        pass

    def transform(real, *, current_size, **kwargs):
        return (
            np.ones((real.shape[0], 3, 3, 2), dtype=np.complex128),
            np.ones((real.shape[0], 3), dtype=np.float64),
            current_size // 2,
        )

    def prepare(*args, **kwargs):
        if kwargs["dump_label"].startswith("iter000"):
            result = original_prepare(*args, **kwargs)
            old_projectors.append(weakref.ref(result))
            built[kwargs["dump_label"]] = result
            return result
        if kwargs.get("reusable") is None and kwargs["dump_label"].endswith("half0"):
            events.append("accuracy_projector_built_first")
            return original_prepare(*args, **kwargs)
        if kwargs.get("reusable") is not None:
            built.clear()
            assert all(reference() is None for reference in old_projectors)
            result = original_prepare(*args, **kwargs)
            assert result is kwargs["reusable"].projector
            events.append("half1_reuses_the_accuracy_projector")
            return result
        assert all(reference() is None for reference in old_projectors)
        events.append("previous_projectors_released_before_half2_build")
        raise LifetimeChecked

    def accuracy(self, **kwargs):
        count = len(kwargs["best_eulers_deg"])
        return SimpleNamespace(
            acc_rot=1.25, acc_trans_angstrom=1.5,
            acc_rot_per_class=np.array([1.25]), acc_trans_per_class_angstrom=np.array([1.5]),
            class_counts=np.array([count]), trial_local_indices=np.arange(count),
            trial_particle_ids=np.arange(count),
        )

    def score(half, phase, **kwargs):
        data = half.data
        if built:
            assert data.projector is built[f"iter000_half{data.particles.index}"]
        grid = phase.grid
        dataset = data.particles.dataset
        shape = kwargs["padded_volume_shape"]
        n_images, size, n_shells = dataset.n_units, int(np.prod(shape)), dataset.image_shape[0] // 2 + 1
        hard_assignments = np.zeros(n_images, dtype=np.int32)
        return score_outputs.HalfScoreResult(
            ha=hard_assignments,
            Ft_y=jnp.zeros(size, dtype=jnp.complex64),
            Ft_ctf=jnp.ones(size, dtype=jnp.complex64),
            em_stats=RelionStats(
                log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(len(grid.rotations), dtype=jnp.float32),
            ),
            noise_stats=NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_images),
            ),
            pose_rotations=grid.rotations, pose_rotation_eulers=grid.rotation_eulers,
            best_pose_rotations=np.tile(np.eye(3, dtype=np.float32), (dataset.n_units, 1, 1)),
            best_pose_rotation_eulers=np.zeros((dataset.n_units, 3), dtype=np.float64),
            best_pose_translations=np.zeros((dataset.n_units, 2), dtype=np.float32),
            coarse_ha=hard_assignments,
            mstep_accumulator_shape=shape,
        )

    import relax.relion.relion_projector_setup as setup

    monkeypatch.delenv("RELAX_RELION_PROJECTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("RELAX_RELION_PROJECTOR_DUMP_DIR", raising=False)
    monkeypatch.setattr(setup, "reference_to_relion_projector_half_maps_and_power", transform)
    monkeypatch.setattr(projector_preparation, "prepare_scoring_projector", prepare)
    monkeypatch.setattr(expected_accuracy_module.Half1AccuracyInputs, "estimate", accuracy)
    monkeypatch.setattr(expectation_module, "score_numbered_half", score)
    monkeypatch.setattr(
        sampling_module, "relion_scoring_rotation_grid",
        lambda order, dtype=None, *, symmetry="C1": sampling_module.RotationGrid(
            rotations=np.asarray(rotations, dtype=dtype),
            rotation_eulers=np.zeros((len(rotations), 3), dtype=dtype),
            healpix_order=order, symmetry=symmetry,
        ),
    )
    with pytest.raises(LifetimeChecked):
        refine_single_volume(
            half_datasets, StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0), HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)), translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=2, init_current_size=4, init_healpix_order=2,
                    max_healpix_order=2, skip_final_iteration=True,
                ),
                adaptive=stand_in.adaptive(adaptive_oversampling=0, coarse_engine="gemm_dense"),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0, optimizer_random_seed=17),
                start=StartState(
                    init_previous_best_rotation_eulers=[np.zeros((dataset.n_units, 3)) for dataset in half_datasets],
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )
    assert len(old_projectors) == 2
    assert events == [
        "accuracy_projector_built_first", "half1_reuses_the_accuracy_projector",
        "previous_projectors_released_before_half2_build",
    ]


def test_numbered_projector_preparation_skips_empty_half(
    half_datasets, init_volume, translations, monkeypatch,
):
    half_datasets[1] = MockDataset(0, np.random.default_rng(9))
    built = []

    class PreparationChecked(Exception):
        pass

    def prepare(*args, **kwargs):
        built.append(kwargs["dump_label"])
        return projector_preparation.PreparedProjector(data=np.ones((1, 3, 3, 2), dtype=np.complex128), r_max=2)

    def score(half, *args, **kwargs):
        assert half.data.particles.index == 0
        assert built == ["iter000_half0"]
        assert half.data.projector.r_max == 2
        raise PreparationChecked

    monkeypatch.setattr(projector_preparation, "prepare_scoring_projector", prepare)
    monkeypatch.setattr(expectation_module, "score_numbered_half", score)
    with pytest.raises(PreparationChecked):
        refine_single_volume(
            half_datasets, StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0), HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)), translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1, init_current_size=4, init_healpix_order=2,
                    max_healpix_order=2, skip_final_iteration=True,
                ),
                adaptive=stand_in.adaptive(adaptive_oversampling=0, coarse_engine="gemm_dense"),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
            ),
            observer=RunObserver(), source=InputSource(),
        )


def test_half0_local_relion_accumulators_offload_to_host():
    result = score_outputs.HalfScoreResult(
        ha=np.zeros(1, dtype=np.int32),
        Ft_y=jnp.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=jnp.complex64),
        Ft_ctf=jnp.asarray([5.0, 6.0], dtype=jnp.float32),
        em_stats=object(),
        noise_stats=object(),
        mstep_full_half_axis=0,
        mstep_accumulator_shape=(3, 3, 3),
    )

    out = score_outputs._maybe_host_offload_half0_local_accumulators(
        half_index=0,
        use_local=True,
        score_result=result,
        log=iteration_loop_module.logger,
    )

    assert out is result
    assert isinstance(out.Ft_y, np.ndarray)
    assert isinstance(out.Ft_ctf, np.ndarray)
    np.testing.assert_allclose(out.Ft_y, np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64))
    np.testing.assert_allclose(out.Ft_ctf, np.array([5.0, 6.0], dtype=np.float32))


def test_half0_local_relion_accumulator_offload_skips_non_x_half():
    result = score_outputs.HalfScoreResult(
        ha=np.zeros(1, dtype=np.int32),
        Ft_y=jnp.asarray([1.0 + 0.0j], dtype=jnp.complex64),
        Ft_ctf=jnp.asarray([1.0], dtype=jnp.float32),
        em_stats=object(),
        noise_stats=object(),
        mstep_full_half_axis=None,
    )

    out = score_outputs._maybe_host_offload_half0_local_accumulators(
        half_index=0,
        use_local=True,
        score_result=result,
        log=iteration_loop_module.logger,
    )

    assert out is result
    assert not isinstance(out.Ft_y, np.ndarray)


def test_run_local_search_iteration_fine_pass_uses_model_sigma_for_translation_prior(monkeypatch, rng):
    from relax.refinement import local_search_iteration as local_iteration_module

    mock_dataset = MockDataset(1, rng)
    captured = {}

    def fake_build_local_hypothesis_layout(
        prior_rotations,
        rotation_grid_rotations,
        sigma_rot,
        sigma_psi,
        healpix_order,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        voxel_size,
        *,
        grid_metadata,
        translation_prior_reference_translations=None,
        rotation_log_prior=None,
        rotation_grid_random_perturbation=0.0,
        rotation_grid_angular_sampling_deg=None,
        local_parent_oversampling_order=0,
        rotation_grid_mstep_rotations=None,
        generate_relion_mstep_rotations=False,
        dtype=np.float32,
    ):
        captured["offset_range_pixels"] = offset_range_pixels
        captured["sigma_offset_angstrom"] = sigma_offset_angstrom
        captured["rotation_grid_random_perturbation"] = rotation_grid_random_perturbation
        captured["rotation_grid_angular_sampling_deg"] = rotation_grid_angular_sampling_deg
        captured["translation_prior_reference_translations"] = (
            None
            if translation_prior_reference_translations is None
            else np.asarray(translation_prior_reference_translations, dtype=np.float32).copy()
        )
        return LocalHypothesisLayout(
            n_global_rotations=1,
            n_pixels=1,
            n_psi=1,
            rotation_offsets=np.array([0, 1], dtype=np.int64),
            rotation_ids_flat=np.array([0], dtype=np.int32),
            rotations_flat=np.repeat(np.eye(3, dtype=dtype)[None, :, :], 1, axis=0),
            rotation_log_priors_flat=np.zeros(1, dtype=dtype),
            rotation_counts=np.array([1], dtype=np.int32),
            translation_grid=np.asarray(translations, dtype=dtype),
            translation_log_priors=np.zeros((1, np.asarray(translations).shape[0]), dtype=dtype),
        )

    def fake_resident_local_search(data, layout, kernel, support, **kwargs):
        captured["reconstruct_significant_only"] = support.reconstruct_significant_only
        captured["adaptive_fraction"] = support.adaptive_fraction
        captured["max_significants"] = support.applied_max_significants
        captured["use_float64_scoring"] = kernel.use_float64_scoring
        captured["relion_exact_score_translation"] = kernel.relion_exact_score_translation
        output = LocalEMResult(
            Ft_y=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            Ft_ctf=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            hard_assignments=np.zeros(mock_dataset.n_units, dtype=np.int32),
            stats=RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(1, dtype=jnp.float32),
            ),
            noise_stats=NoiseStats(
                wsum_sigma2_noise=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_img_power=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=0.0,
            ),
        )
        return output

    monkeypatch.setattr(local_iteration_module, "build_local_hypothesis_layout", fake_build_local_hypothesis_layout)
    monkeypatch.setattr(local_iteration_module, "compute_local_search_resident", fake_resident_local_search)

    prior_rotations = np.zeros((1, 3), dtype=np.float32)
    rotation_grid_rotations = get_relion_rotation_grid(0).astype(np.float32)
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    reference_translations = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    outputs = local_search_iteration._run_local_search_iteration(*local_iteration_owners(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        prior_rotations,
        rotation_grid_rotations,
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=translations,
        prior_translations=np.zeros((1, 2), dtype=np.float32),
        sigma_offset_angstrom=1.25,
        disc_type="linear_interp",
        current_size=4,
        accumulate_noise=True,
        relion_exact_score_translation=True,
        translation_prior_reference_translations=reference_translations,
    ))

    assert captured["offset_range_pixels"] is None
    assert captured["sigma_offset_angstrom"] == 1.25
    assert captured["rotation_grid_random_perturbation"] == 0.0
    assert captured["rotation_grid_angular_sampling_deg"] is None
    assert captured["reconstruct_significant_only"] is True
    assert captured["adaptive_fraction"] == pytest.approx(0.999)
    assert captured["max_significants"] == -1
    assert captured["use_float64_scoring"] is False
    assert captured["relion_exact_score_translation"] is True
    np.testing.assert_allclose(
        captured["translation_prior_reference_translations"],
        reference_translations,
        atol=1e-6,
    )
    assert isinstance(outputs, LocalSearchResult)
    assert outputs.noise_stats is not None
    assert outputs.profile_summary is None
    assert outputs.best_pose_rotations is None
    assert outputs.best_pose_translations is None


def test_run_local_search_iteration_dispatches_aligned_mstep_grid(monkeypatch, rng):
    from relax.refinement import local_search_iteration as local_iteration_module

    dataset = MockDataset(1, rng)
    score_grid = get_relion_rotation_grid(0).astype(np.float32)
    mstep_grid = np.arange(score_grid.size, dtype=np.float32).reshape(score_grid.shape)
    captured = {}

    class DispatchCaptured(Exception):
        pass

    def capture_dispatch(data, layout, kernel, support, **kwargs):
        captured["layout"] = layout
        captured["relion_exact_score_translation"] = kernel.relion_exact_score_translation
        raise DispatchCaptured

    monkeypatch.setattr(local_iteration_module, "compute_local_search_resident", capture_dispatch)

    with pytest.raises(DispatchCaptured):
        local_search_iteration._run_local_search_iteration(*local_iteration_owners(
            dataset,
            jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            np.zeros((1, 3), dtype=np.float32),
            score_grid,
            healpix_order=0,
            sigma_rot=0.0,
            sigma_psi=0.0,
            translations=np.zeros((1, 2), dtype=np.float32),
            prior_translations=np.zeros((1, 2), dtype=np.float32),
            sigma_offset_angstrom=1.0,
            disc_type="linear_interp",
            current_size=4,
            relion_exact_score_translation=True,
            rotation_grid_mstep_rotations=mstep_grid,
            generate_relion_mstep_rotations=True,
        ))

    layout = captured["layout"]
    assert_matches(layout.rotations_flat, score_grid[layout.rotation_ids_flat])
    assert_matches(layout.mstep_rotations_flat, mstep_grid[layout.rotation_ids_flat])
    assert captured["relion_exact_score_translation"] is True

    backward_compatible = build_local_hypothesis_layout(
        np.zeros((1, 3), dtype=np.float32),
        score_grid,
        0.0,
        0.0,
        0,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        1.0,
        None,
        dataset.voxel_size,
        grid_metadata=build_local_search_grid_metadata(0),
    )
    assert backward_compatible.mstep_rotations_flat is None


def test_run_local_search_iteration_plumbs_normalization_log_evidence(monkeypatch, rng):
    from relax.refinement import local_search_iteration as local_iteration_module

    mock_dataset = MockDataset(2, rng)
    layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1, 2], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.zeros((2, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )
    captured = {}
    normalization_log_evidence = np.array([1.25, 2.5], dtype=np.float64)

    def fake_resident_local_search(data, layout, kernel, support, **kwargs):
        captured["normalization_log_evidence"] = support.normalization_log_evidence
        return LocalEMResult(
            Ft_y=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            Ft_ctf=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            hard_assignments=np.zeros(mock_dataset.n_units, dtype=np.int32),
            stats=RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(3, dtype=jnp.float32),
            ),
        )

    monkeypatch.setattr(local_iteration_module, "compute_local_search_resident", fake_resident_local_search)

    outputs = local_search_iteration._run_local_search_iteration(*local_iteration_owners(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy(),
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=layout.translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp",
        current_size=4,
        accumulate_noise=False,
        pass2_layout=layout,
        normalization_log_evidence=normalization_log_evidence,
    ))

    np.testing.assert_allclose(captured["normalization_log_evidence"], normalization_log_evidence)
    assert isinstance(outputs, LocalSearchResult)
    assert outputs.noise_stats is None
    assert outputs.profile_summary is None
    assert outputs.best_pose_rotations is None
    assert outputs.best_pose_translations is None


def test_run_local_search_iteration_plumbs_stats_use_reconstruction_probs(monkeypatch, rng):
    from relax.refinement import local_search_iteration as local_iteration_module

    mock_dataset = MockDataset(2, rng)
    layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1, 2], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.zeros((2, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )
    captured = {}

    def fake_resident_local_search(data, layout, kernel, support, **kwargs):
        captured["stats_use_reconstruction_probs"] = support.stats_use_reconstruction_probs
        return LocalEMResult(
            Ft_y=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            Ft_ctf=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            hard_assignments=np.zeros(mock_dataset.n_units, dtype=np.int32),
            stats=RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(3, dtype=jnp.float32),
            ),
        )

    monkeypatch.setattr(local_iteration_module, "compute_local_search_resident", fake_resident_local_search)

    outputs = local_search_iteration._run_local_search_iteration(*local_iteration_owners(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy(),
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=layout.translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp",
        current_size=4,
        accumulate_noise=False,
        pass2_layout=layout,
        reconstruct_significant_only=True,
        stats_use_reconstruction_probs=True,
    ))

    assert captured["stats_use_reconstruction_probs"] is True
    assert isinstance(outputs, LocalSearchResult)
    assert outputs.noise_stats is None
    assert outputs.profile_summary is None
    assert outputs.best_pose_rotations is None
    assert outputs.best_pose_translations is None


def test_run_local_search_iteration_fine_pass_uses_factorized_prior_metadata_for_perturbed_grid(
    monkeypatch,
    rng,
):
    from recovar import utils

    from relax.refinement import local_search_iteration as local_iteration_module

    mock_dataset = MockDataset(1, rng)
    captured = {}

    def fake_build_local_hypothesis_layout(
        prior_rotations,
        rotation_grid_rotations,
        sigma_rot,
        sigma_psi,
        healpix_order,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        voxel_size,
        *,
        grid_metadata,
        translation_prior_reference_translations=None,
        rotation_log_prior=None,
        rotation_grid_random_perturbation=0.0,
        rotation_grid_angular_sampling_deg=None,
        local_parent_oversampling_order=0,
        rotation_grid_mstep_rotations=None,
        generate_relion_mstep_rotations=False,
        dtype=np.float32,
    ):
        captured["grid_metadata_mode"] = grid_metadata["mode"]
        captured["n_pixels"] = int(grid_metadata["n_pixels"])
        captured["n_psi"] = int(grid_metadata["n_psi"])
        captured["rotation_grid_random_perturbation"] = rotation_grid_random_perturbation
        captured["rotation_grid_angular_sampling_deg"] = rotation_grid_angular_sampling_deg
        captured["scored_rotations"] = np.asarray(rotation_grid_rotations, dtype=dtype).copy()
        return LocalHypothesisLayout(
            n_global_rotations=rotation_grid_rotations.shape[0],
            n_pixels=1,
            n_psi=1,
            rotation_offsets=np.array([0, 1], dtype=np.int64),
            rotation_ids_flat=np.array([0], dtype=np.int32),
            rotations_flat=np.asarray(rotation_grid_rotations[:1], dtype=dtype),
            rotation_log_priors_flat=np.zeros(1, dtype=dtype),
            rotation_counts=np.array([1], dtype=np.int32),
            translation_grid=np.asarray(translations, dtype=dtype),
            translation_log_priors=np.zeros((1, np.asarray(translations).shape[0]), dtype=dtype),
        )

    def fake_resident_local_search(data, layout, kernel, support, **kwargs):
        captured["max_significants"] = support.applied_max_significants
        captured["use_float64_scoring"] = kernel.use_float64_scoring
        return LocalEMResult(
            Ft_y=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            Ft_ctf=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            hard_assignments=np.zeros(mock_dataset.n_units, dtype=np.int32),
            stats=RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(rotation_grid_size(1), dtype=jnp.float32),
            ),
            noise_stats=NoiseStats(
                wsum_sigma2_noise=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_img_power=jnp.zeros(mock_dataset.image_shape[0] // 2 + 1, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=0.0,
            ),
        )

    monkeypatch.setattr(local_iteration_module, "build_local_hypothesis_layout", fake_build_local_hypothesis_layout)
    monkeypatch.setattr(local_iteration_module, "compute_local_search_resident", fake_resident_local_search)

    healpix_order = 1
    canonical_rotations = get_relion_rotation_grid(healpix_order).astype(np.float32)
    perturbed_rotations = apply_relion_rotation_perturbation(
        canonical_rotations,
        random_perturbation=0.3,
        angular_sampling_deg=relion_angular_sampling_deg(healpix_order),
    ).astype(np.float32)
    perturbed_eulers = utils.R_to_relion(perturbed_rotations, degrees=True).astype(np.float32)
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    # A perturbed full Euler table no longer factorizes. RELION still builds
    # local priors from the canonical Healpix direction and psi axes, then
    # applies SamplingPerturbation only when scoring the trial rotations.
    assert (
        build_local_search_grid_metadata(
            healpix_order,
            grid_eulers=perturbed_eulers,
            grid_rotations=perturbed_rotations,
        )["mode"]
        == "full"
    )

    outputs = local_search_iteration._run_local_search_iteration(*local_iteration_owners(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        np.zeros((1, 3), dtype=np.float32),
        perturbed_rotations,
        healpix_order=healpix_order,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=translations,
        prior_translations=np.zeros((1, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp",
        current_size=4,
        accumulate_noise=True,
    ))

    assert captured["grid_metadata_mode"] == "factorized"
    assert captured["n_pixels"] == hp.nside2npix(2**healpix_order)
    assert captured["n_psi"] == rotation_grid_n_in_planes(healpix_order)
    assert captured["max_significants"] == -1
    assert captured["use_float64_scoring"] is False
    assert captured["rotation_grid_random_perturbation"] == 0.0
    assert captured["rotation_grid_angular_sampling_deg"] is None
    np.testing.assert_allclose(captured["scored_rotations"], perturbed_rotations)
    assert isinstance(outputs, LocalSearchResult)
    assert outputs.noise_stats is not None
    assert outputs.profile_summary is None
    assert outputs.best_pose_rotations is None
    assert outputs.best_pose_translations is None


def test_run_local_search_iteration_plumbs_score_only_to_the_resident_probe(monkeypatch, rng):
    from relax.refinement import local_search_iteration as local_iteration_module

    mock_dataset = MockDataset(2, rng)
    layout = LocalHypothesisLayout(
        n_global_rotations=3,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        rotation_ids_flat=np.array([0, 1, 1, 2], dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(4, dtype=np.float32),
        rotation_counts=np.array([2, 2], dtype=np.int32),
        translation_grid=np.zeros((2, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 2), dtype=np.float32),
    )
    captured = {}

    def fake_run_local_em_exact(data, layout, kernel, support, **kwargs):
        captured.update(
            score_only=support.score_only,
            disable_adjoint_y=support.disable_adjoint_y,
            disable_adjoint_ctf=support.disable_adjoint_ctf,
            accumulate_noise=kernel.accumulate_noise,
        )
        return LocalEMResult(
            Ft_y=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            Ft_ctf=jnp.zeros(mock_dataset.volume_size, dtype=mock_dataset.dtype),
            hard_assignments=np.zeros(mock_dataset.n_units, dtype=np.int32),
            stats=RelionStats(
                log_evidence_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(mock_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(mock_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.zeros(3, dtype=jnp.float32),
            ),
            profile={"score_only": support.score_only},
        )

    monkeypatch.setattr(local_iteration_module, "compute_local_search_resident", fake_run_local_em_exact)

    outputs = local_search_iteration._run_local_search_iteration(*local_iteration_owners(
        mock_dataset,
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy(),
        healpix_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        translations=layout.translation_grid,
        prior_translations=np.zeros((2, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp",
        current_size=4,
        accumulate_noise=False,
        pass2_layout=layout,
        return_profile=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        score_only=True,
    ))

    assert captured["score_only"] is True
    assert captured["disable_adjoint_y"] is True
    assert captured["disable_adjoint_ctf"] is True
    assert captured["accumulate_noise"] is False
    assert outputs.profile_summary["score_only"] is True


def test_local_adaptive_parent_support_probe_is_score_only(monkeypatch, rng):
    """The local adaptive pass 1 scores the parents only: no backprojection, sample indices returned."""
    dataset = MockDataset(2, rng)
    captured = {}

    class StopAfterParentProbe(Exception):
        pass

    def parent_probe(data, grid, kernel, support):
        captured.update(support=support)
        raise StopAfterParentProbe

    monkeypatch.setattr(
        half_scoring, "_build_local_adaptive_parent_layout",
        lambda *args: (SimpleNamespace(rotation_counts=np.asarray([2])), 0),
    )
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", parent_probe)
    with pytest.raises(StopAfterParentProbe):
        half_scoring._score_half_local(*local_half_owners(
            k=0,
            experiment_dataset=dataset,
            means_k=jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
            noise_variance_k=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            previous_best_rotation_eulers_k=np.zeros((dataset.n_units, 3), dtype=np.float32),
            local_search_rotations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
            local_search_order=1,
            sigma_rot=np.deg2rad(1.0),
            sigma_psi=np.deg2rad(1.0),
            current_translations=np.zeros((1, 2), dtype=np.float32),
            base_translations=np.zeros((1, 2), dtype=np.float32),
            trans_prior_center=np.zeros((dataset.n_units, 2), dtype=np.float32),
            trans_prior_center_for_engine=np.zeros((dataset.n_units, 2), dtype=np.float32),
            current_sigma_offset_angstrom=1.0,
            disc_type="linear_interp",
            cs_for_engine=None,
            local_pass1_current_size=4,
            image_corrections_k=None,
            scale_corrections_k=None,
            translation_search_base=None,
            max_significants=None,
            iteration=3,
            local_search_random_perturbation=0.0,
            local_search_angular_sampling_deg=relion_angular_sampling_deg(1),
            local_parent_oversampling_order=1,
            diagnostic_score_only=False,
            replay_prior_translations=None,
            collect_local_search_profile=False,
            local_profile_history=[],
        ))

    support = captured["support"]
    assert support.disable_adjoint_y is True and support.disable_adjoint_ctf is True
    assert support.return_reconstruction_sample_indices is True
    assert support.score_only is True
    assert support.return_profile is True


def test_k1_mstep_preserves_retained_pose_support_mass():
    """K=1 must not replace significant-support mass with image count."""
    retained_sumw = 2.9975
    stats = NoiseStats(
        wsum_sigma2_noise=jnp.ones(1, dtype=jnp.float32),
        wsum_img_power=jnp.ones(1, dtype=jnp.float32),
        wsum_sigma2_offset=0.0,
        sumw=retained_sumw,
    )

    got = _resolve_class_mstep_posterior_sums(
        noise_stats=(stats,),
        class_posterior_sums_full=np.asarray([3.0], dtype=np.float64),
        class_posterior_sums_override=None,
    )

    assert_matches(got, np.asarray([retained_sumw], dtype=np.float64))


def _three_image_local_layout(all_rotations):
    """Small ragged pose layout shared by fused/split local-search checks."""
    translations = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    rotation_ids = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int32),
    ]
    rotation_counts = np.asarray([ids.size for ids in rotation_ids], dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    rotation_ids_flat = np.concatenate(rotation_ids).astype(np.int32)
    return LocalHypothesisLayout(
        n_global_rotations=all_rotations.shape[0],
        n_pixels=6,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=np.asarray(all_rotations[rotation_ids_flat], dtype=np.float32),
        rotation_log_priors_flat=np.linspace(0.0, -0.7, rotation_ids_flat.size, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=translations,
        translation_log_priors=np.array(
            [[0.0, -0.5], [-0.2, 0.1], [0.3, -0.4]],
            dtype=np.float32,
        ),
    )


def _sparse_big_jit_local_case(rng):
    dataset = RawRealImageDataset(3, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=571)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    all_rotations = _make_rotations(6, seed=573)
    translations = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    rotation_ids = [
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([0, 2, 4, 5], dtype=np.int32),
    ]
    rotation_counts = np.asarray([ids.size for ids in rotation_ids], dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    rotation_ids_flat = np.concatenate(rotation_ids).astype(np.int32)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=all_rotations.shape[0],
        n_pixels=6,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=np.asarray(all_rotations[rotation_ids_flat], dtype=np.float32),
        rotation_log_priors_flat=np.linspace(0.0, -0.7, rotation_ids_flat.size, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=translations,
        translation_log_priors=np.array(
            [[0.0, -0.5], [-0.2, 0.1], [0.3, -0.4]],
            dtype=np.float32,
        ),
    )
    return dataset, mean, noise_variance, local_layout


def _assert_significance_stats_allclose(actual, expected):
    """Compare every statistic."""
    assert actual.keys() == expected.keys()
    for key in actual.keys():
        actual_value = np.asarray(actual[key])
        expected_value = np.asarray(expected[key])
        if not (np.issubdtype(actual_value.dtype, np.number)
                and np.issubdtype(expected_value.dtype, np.number)):
            np.testing.assert_array_equal(actual_value, expected_value, err_msg=key)
            continue
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            rtol=1e-6,
            atol=1e-6,
            err_msg=key,
        )


def _assert_relion_stats_allclose(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual.log_evidence_per_image),
        np.asarray(expected.log_evidence_per_image),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual.best_log_score_per_image),
        np.asarray(expected.best_log_score_per_image),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual.max_posterior_per_image),
        np.asarray(expected.max_posterior_per_image),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual.rotation_posterior_sums),
        np.asarray(expected.rotation_posterior_sums),
        rtol=1e-5,
        atol=1e-6,
    )


def _assert_noise_stats_allclose(actual, expected):
    assert actual is not None
    assert expected is not None
    for field in actual._fields:
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value is None or expected_value is None:
            assert actual_value is None
            assert expected_value is None
            continue
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=1e-5,
            atol=1e-6,
        )


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _unit_image_mask(dtype=jnp.float32):
    return jnp.linspace(0.2, 1.0, IMAGE_SIZE, dtype=dtype).reshape(IMAGE_SHAPE)


def _raw_real_process(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    if apply_image_mask:
        images = images * _unit_image_mask(images.dtype)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    if apply_image_mask:
        images = images * _unit_image_mask(images.dtype)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


class MockDataset:
    """Minimal mock of CryoEMDataset for unit testing."""

    particles_file = None  # built in memory: no RELION optics table

    def __init__(self, n_images, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.padding = 0
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = n_images
        self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)
        self.image_mask = np.asarray(_unit_image_mask(np.float32), dtype=np.float32)
        self.premultiplied_ctf = False

        self._images = rng.standard_normal((n_images, *IMAGE_SHAPE)).astype(np.float32)

        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)

        class _Backend:
            image_mask = np.asarray(_unit_image_mask(np.float32), dtype=np.float32)
            image_mask_mode = "multiply"

            def set_relion_image_mask(self, pixel_size: float, particle_diameter_ang: float, width_mask_edge_px: float = 5.0):
                self.image_mask_mode = "relion_background_fill"

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)
            backend = _Backend()

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                self.rotation_matrices[idx],
                self.translations[idx],
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def update_poses(self, rots, trans):
        self.rotation_matrices = np.asarray(rots)
        self.translations = np.asarray(trans)

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def original_image_indices_from_local(self, indices=None):
        return np.arange(self.n_images, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)


def test_pass2_operands_route_relion_cuda_norm_and_shift_before_fft(rng):
    """The pass-2 operand preparation (prepare_unshifted_bucket_operands, which the resident
    engine calls once per half) hands RELION's normalization and integer shift to strict CUDA."""
    from recovar.core.configs import ForwardModelConfig
    from recovar.reconstruction import noise as noise_utils

    from relax.sparse_pass2.sparse_pass2_bucket_io import prepare_unshifted_bucket_operands

    dataset = MockDataset(1, rng)
    dataset.image_source.backend.image_mask_mode = "relion_background_fill"
    dataset.image_source.backend.relion_fourier_backend = "relion_cuda"
    captured = {}

    class _CapturedStrictPreprocess(RuntimeError):
        pass

    def capture_process(batch, apply_image_mask=False, **kwargs):
        captured.update(batch=np.asarray(batch), apply_image_mask=apply_image_mask, **kwargs)
        raise _CapturedStrictPreprocess

    config = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp", process_fn=dataset.process_images)
    batch, _, _, ctf_params, _, _, image_indices = next(dataset.iter_batches(1))
    dataset.process_images_half = capture_process
    with pytest.raises(_CapturedStrictPreprocess):
        prepare_unshifted_bucket_operands(
            dataset,
            batch,
            ctf_params,
            image_indices,
            noise_variance_half=noise_utils.to_batched_half_pixel_noise(
                jnp.ones(IMAGE_SIZE, dtype=jnp.float32), IMAGE_SHAPE
            ).squeeze(),
            config=config,
            score_with_masked_images=True,
            image_corrections=np.asarray([0.8], dtype=np.float32),
            scale_corrections=np.asarray([2.0], dtype=np.float32),
            image_pre_shifts=np.asarray([[1.0, -1.0]], dtype=np.float32),
            use_float64_scoring=False,
        )

    # Raw pixels reach strict CUDA unchanged; RELION normalization and the
    # zero-fill shift are explicit operands for the CUDA boundary.
    assert_matches(captured["batch"], dataset._images)
    assert_matches(captured["relion_normalization_factors"], np.asarray([0.4], np.float32))
    assert_matches(captured["relion_integer_shifts"], np.asarray([[1, -1]], np.int32))
    assert captured["apply_image_mask"] is True


def _exact_pass1_on_cpu(monkeypatch):
    """Let pass 1 reach its per-batch preprocessing on CPU: it otherwise needs a CUDA GPU."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: True)
    monkeypatch.setenv("RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE", "0")


def _exact_pass1_inputs(monkeypatch, *, n_images=2, n_classes=1):
    """A pass-1 call on the CPU exact-operand harness (helpers.exact_pass1_harness): the
    dataset, the means, the noise and the projector keywords."""

    from helpers.exact_pass1_harness import ExactPass1Dataset, coded_class_projectors, install_exact_pass1_mocks

    install_exact_pass1_mocks(monkeypatch)
    dataset = ExactPass1Dataset(np.arange(n_images), box=IMAGE_SHAPE[0])
    projector = dict(
        relion_projector_half=coded_class_projectors(n_classes),
        relion_projector_r_max=1,
        relion_projector_texture_interp=True,
        half_spectrum_scoring=True,
    )
    return (
        dataset,
        jnp.zeros((n_classes, dataset.volume_size), dtype=jnp.complex64),
        jnp.ones(dataset.image_size, dtype=jnp.float32),
        projector,
    )


# RELION Projector::data geometry for r_max=1 at padding 1: 2 * (r_max + 1) + 1 = 5.
_EXACT_PASS1_PROJECTOR = dict(
    relion_projector_half=jnp.zeros((1, 5, 5, 3), dtype=jnp.complex64),
    relion_projector_r_max=1,
    relion_projector_texture_interp=True,
    half_spectrum_scoring=True,
)


def test_k_class_firstiter_cc_routes_relion_cuda_norm_and_shift_before_fft(
    rng,
    monkeypatch,
):
    dataset = MockDataset(1, rng)
    dataset.image_source.backend.image_mask_mode = "relion_background_fill"
    dataset.image_source.backend.relion_fourier_backend = "relion_cuda"
    _exact_pass1_on_cpu(monkeypatch)
    captured = {}

    class _CapturedStrictPreprocess(RuntimeError):
        pass

    def capture_process(batch, apply_image_mask=False, **kwargs):
        captured.update(batch=np.asarray(batch), apply_image_mask=apply_image_mask, **kwargs)
        raise _CapturedStrictPreprocess

    dataset.process_images_half = capture_process
    with pytest.raises(_CapturedStrictPreprocess):
        _compute_k_class_significance_batched(
            dataset,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            _make_rotations(1, seed=884),
            np.zeros((1, 2), dtype=np.float32),
            class_log_priors=np.zeros(1, dtype=np.float64),
            adaptive_fraction=1.0,
            max_significants=1,
            image_batch_size=1,
            rotation_block_size=1,
            current_size=None,
            score_with_masked_images=True,
            image_corrections=np.asarray([0.8], dtype=np.float32),
            scale_corrections=np.asarray([2.0], dtype=np.float32),
            image_pre_shifts=np.asarray([[1.0, -1.0]], dtype=np.float32),
            score_mode="normalized_cc",
            collect_significance=False,
            **_EXACT_PASS1_PROJECTOR,
        )

    assert_matches(captured["batch"], dataset._images)
    assert_matches(captured["relion_normalization_factors"], np.asarray([0.4], np.float32))
    assert_matches(captured["relion_integer_shifts"], np.asarray([[1, -1]], np.int32))
    assert captured["apply_image_mask"] is True


def test_coarse_gaussian_routes_relion_cuda_norm_and_shift_before_fft(rng, monkeypatch):
    _exact_pass1_on_cpu(monkeypatch)
    dataset = MockDataset(1, rng)
    dataset.image_source.backend.image_mask_mode = "relion_background_fill"
    dataset.image_source.backend.relion_fourier_backend = "relion_cuda"
    captured = {}

    class _CapturedStrictPreprocess(RuntimeError):
        pass

    def capture_process(batch, apply_image_mask=False, **kwargs):
        captured.update(batch=np.asarray(batch), apply_image_mask=apply_image_mask, **kwargs)
        raise _CapturedStrictPreprocess

    dataset.process_images_half = capture_process
    with pytest.raises(_CapturedStrictPreprocess):
        _compute_k_class_significance_batched(
            dataset,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            _make_rotations(1, seed=886),
            np.zeros((1, 2), dtype=np.float32),
            class_log_priors=np.zeros(1, dtype=np.float64),
            adaptive_fraction=1.0,
            max_significants=1,
            image_batch_size=1,
            rotation_block_size=1,
            current_size=None,
            score_with_masked_images=True,
            image_corrections=np.asarray([0.8], dtype=np.float32),
            scale_corrections=np.asarray([2.0], dtype=np.float32),
            image_pre_shifts=np.asarray([[1.0, -1.0]], dtype=np.float32),
            **_EXACT_PASS1_PROJECTOR,
        )

    assert_matches(captured["batch"], dataset._images)
    assert_matches(captured["relion_normalization_factors"], np.asarray([0.4], np.float32))
    assert_matches(captured["relion_integer_shifts"], np.asarray([[1, -1]], np.int32))
    assert captured["apply_image_mask"] is True


class RawRealImageDataset:
    """Minimal raw real-space dataset for native half-preprocess tests."""

    def __init__(self, n_images, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.padding = 0
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = n_images
        self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = np.float32
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)
        self.premultiplied_ctf = False
        self._images = rng.standard_normal((n_images, *IMAGE_SHAPE)).astype(np.float32)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)

        class _Backend:
            image_mask = None
            image_mask_mode = "multiply"

            def set_relion_image_mask(self, pixel_size: float, particle_diameter_ang: float, width_mask_edge_px: float = 5.0):
                self.image_mask_mode = "relion_background_fill"

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)
            backend = _Backend()

        self.image_source = _ImageSource()

    @property
    def image_mask(self):
        return None

    @property
    def data_multiplier(self):
        return 1.0

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                self.rotation_matrices[idx],
                self.translations[idx],
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def update_poses(self, rots, trans):
        self.rotation_matrices = np.asarray(rots)
        self.translations = np.asarray(trans)

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def original_image_indices_from_local(self, indices=None):
        return np.arange(self.n_images, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def half_datasets(rng):
    ds0 = MockDataset(N_IMAGES // 2, rng)
    ds1 = MockDataset(N_IMAGES // 2, rng)
    return [ds0, ds1]


@pytest.fixture
def init_volume():
    return _hermitian_volume(VOLUME_SHAPE, seed=42)


@pytest.fixture
def rotations():
    return _make_rotations(N_ROTATIONS, seed=12)


@pytest.fixture
def translations():
    return jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)


@pytest.fixture
def fake_global_estep(monkeypatch):
    """The CPU stand-in for the global E-step (``helpers.fake_adaptive_engine``).

    Every global E-step runs ``run_dense_k_class_em_adaptive``, whose pass 2 runs only on
    the device-resident GPU engine. Loop tests replace it; each call backprojects a new
    random Hermitian volume so the two half maps differ. Returns the recorded calls.
    """

    calls = []
    install_fake_adaptive_engine(monkeypatch, calls, Ft_y=_random_half_maps(calls))
    return calls



def _refine_replaying(*args, options, relion_replay=None, **kwargs):
    """``refine_single_volume`` with the input source the command builds for ``relion_replay`` (a
    ``RelionReplay``; None: the native source)."""
    from relax.parity.relion_replay_source import RelionReplaySource

    return refine_single_volume(
        *args, options=options, source=RelionReplaySource.for_run(relion_replay, options),
        **{"observer": RunObserver(), **kwargs},
    )

@pytest.fixture(autouse=True)
def _clear_parity_dump_env(monkeypatch):
    """Isolate these tests from ambient RELION-parity-dump debugging env vars.

    ``_parity_dump.is_active()`` reads ``RELAX_PARITY_DUMP_DIR`` directly, and
    it feeds the ``need_unreg_means`` gate in ``refine_single_volume`` via
    an ``or`` -- so a var left exported in a developer's shell from an earlier
    parity-debugging session silently changes reconstruction call counts and
    intermediate-file output for every test here, regardless of what each
    test's own ``options`` request. ``monkeypatch.delenv`` restores whatever
    value (or absence) existed once the test finishes.
    """
    monkeypatch.delenv("RELAX_PARITY_DUMP_DIR", raising=False)
    monkeypatch.delenv("RELAX_PARITY_TIMING_DIR", raising=False)


# ===========================================================================
# Test 1: RELION-parity smoke test -- runs without error
# ===========================================================================


class TestRelionModeSmokeTest:
    """Call refine_single_volume and verify it runs."""

    @pytest.mark.parametrize("voxel_size", [1.0, 2.0])
    def test_k1_coldstart_supplies_gaussian_translation_prior(
        self, half_datasets, init_volume, rotations, translations, monkeypatch, voxel_size,
    ):
        """Zero initial offsets still have native accelerated pdf_offset, not a flat prior."""
        # RELION-parity refinement on a GPU backend builds its scorer rotations with the
        # strict CUDA primitive and refuses to run with custom CUDA disabled; on CPU the
        # NumPy scorer path runs. Do not inherit a caller's RECOVAR_DISABLE_CUDA.
        monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
        monkeypatch.setattr(
            sampling_module, "relion_scoring_rotation_grid",
            lambda order, dtype, *, symmetry='C1': sampling_module.RotationGrid(rotations=np.asarray(rotations, dtype=dtype), rotation_eulers=np.zeros((N_ROTATIONS, 3), dtype=dtype), healpix_order=order, symmetry=symmetry),
        )
        for dataset in half_datasets:
            dataset.voxel_size = voxel_size

        class PriorChecked(Exception):
            pass

        def check_first_engine_call(dataset, means, *args, **kwargs):
            # Native acc_ml_optimiser_impl.h uses Angstrom sampling translations
            # and multiplies their squared distance by pixel_size**2 again.
            # This fixture uses exactly representable distances and sigma=10 A.
            base = np.asarray(translations, dtype=np.float32)
            expected = -0.5 * np.sum(base**2, axis=-1) * voxel_size**4 / 100.0
            prior = np.asarray(kwargs["translation_log_prior"])
            assert prior.dtype == np.float32
            assert_matches(prior, expected)
            assert np.any(prior < 0.0)
            raise PriorChecked

        monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", check_first_engine_call)
        with pytest.raises(PriorChecked):
            refine_single_volume(
                half_datasets, StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
                HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
                 translations,
                options=stand_in.options(
                    schedule=stand_in.schedule(
                        max_iter=1, init_current_size=4, init_healpix_order=2,
                        max_healpix_order=2,
                    ),
                    execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                    adaptive=stand_in.adaptive(adaptive_oversampling=1),
                    parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
                ),
                observer=RunObserver(), source=InputSource(),
            )


    def test_relion_bootstrap_current_size_matches_benchmark_case(self):
        """128px, 4.25A/px, ini_high=30A should bootstrap from 36 -> 56."""
        assert bootstrap_current_size_relion(36, 128) == 56

    def test_relion_bootstrap_current_size_from_ini_high_matches_benchmark_case(self):
        assert bootstrap_current_size_from_ini_high_relion(128, 4.25, 30.0) == 56

    def test_firstiter_cc_reconstructs_before_tau2_reporting_taper(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """The ini_high tau2 taper changes reported state, not reconstruction."""
        from relax.refinement import numbered_reconstruction as numbered_reconstruction_module
        from relax.refinement import priors

        untapered_tau = [7.0, 11.0]
        taper = np.asarray([1.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
        tau2_call = 0
        reconstruction_tau = []
        reconstruction_tau_is_1d = []

        def fake_tau2_from_weights(*_args, **_kwargs):
            nonlocal tau2_call
            value = untapered_tau[tau2_call]
            tau2_call += 1
            shells = jnp.full(taper.shape, value, dtype=jnp.float32)
            details = {
                "prior_shells": shells,
                "sigma2_shells": jnp.ones_like(shells),
                "avg_weight_shells": jnp.ones_like(shells),
                "shell_sum": jnp.ones_like(shells),
                "shell_count": jnp.ones_like(shells),
                "fsc_shells": jnp.ones_like(shells),
                "ssnr_shells": shells,
            }
            return jnp.full(VOLUME_SIZE, value, dtype=jnp.float32), shells, details

        def fake_reconstruct(*_args, **kwargs):
            reconstruction_tau.append(np.asarray(kwargs["tau"]))
            reconstruction_tau_is_1d.append(kwargs["tau_is_1d"])
            return jnp.ones(VOLUME_SIZE, dtype=jnp.complex64)

        monkeypatch.setattr(
            regularization_relion,
            "compute_relion_tau2_from_weights",
            fake_tau2_from_weights,
        )
        monkeypatch.setattr(
            priors,
            "firstiter_cc_ini_high_tau2_taper",
            lambda *_args, **_kwargs: taper,
        )
        monkeypatch.setattr(numbered_reconstruction_module, "_reconstruct_volume_eager", fake_reconstruct)
        monkeypatch.setattr(
            map_postprocess,
            "apply_relion_initial_lowpass_filter",
            lambda volume, *_args, **_kwargs: volume,
        )

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=4,
                    init_healpix_order=2,
                    max_healpix_order=2,
                    skip_final_iteration=True,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(
                    low_resol_join_halves_angstrom=0.0,
                    emulate_relion_firstiter_cc=True,
                    relion_firstiter_ini_high_angstrom=8.0,
                    use_per_half_mean_variance=False,
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert len(reconstruction_tau) == 2
        assert reconstruction_tau_is_1d == [True, True]
        assert reconstruction_tau[0].shape == taper.shape
        assert reconstruction_tau[1].shape == taper.shape
        assert_matches(reconstruction_tau[0], untapered_tau[0])
        assert_matches(reconstruction_tau[1], untapered_tau[1])
        assert reconstruction_tau[0].dtype == np.float64
        assert reconstruction_tau[1].dtype == np.float64
        assert_matches(result.history.tau2_radial_trajectory[0], untapered_tau[0] * taper)

    def test_align_fourier_volume_sign_to_reference_flips_negative_overlap(self):
        ref = np.array([1.0 + 0.0j, -2.0 + 0.0j], dtype=np.complex64)
        vol = -ref
        aligned, flipped = _align_fourier_volume_sign_to_reference(vol, ref, (2, 1, 1))
        assert flipped is True
        np.testing.assert_allclose(aligned, ref)

    @pytest.mark.parametrize("relative_overlap", [0.5, -0.5, 1e-3, -1e-3])
    def test_align_fourier_volume_sign_matches_the_float64_real_space_overlap(self, relative_overlap):
        """The device overlap takes the same flip decision as the float64 host dot product of the centred real
        volumes, down to a thousandth of the volumes' norms."""

        rng = np.random.default_rng(5)
        shape = (16, 16, 16)
        ref_real = rng.normal(size=shape)
        noise = rng.normal(size=shape)
        ref_c, noise_c = ref_real - ref_real.mean(), noise - noise.mean()
        noise_c -= ref_c * np.vdot(ref_c, noise_c) / np.vdot(ref_c, ref_c)  # orthogonal to the reference
        vol_real = noise_c / np.linalg.norm(noise_c) + relative_overlap * ref_c / np.linalg.norm(ref_c) + 3.0
        to_ft = lambda v: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(v))).astype(np.complex64).reshape(-1)  # noqa: E731
        ref_ft, vol_ft = to_ft(ref_real), to_ft(vol_real)

        def host_overlap(v_ft, r_ft):
            back = lambda f: np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f.reshape(shape)))))  # noqa: E731
            v, r = back(v_ft.astype(np.complex128)), back(r_ft.astype(np.complex128))
            return float(np.dot((r - r.mean()).ravel(), (v - v.mean()).ravel()))

        aligned, flipped = _align_fourier_volume_sign_to_reference(vol_ft, ref_ft, shape)
        assert flipped is (host_overlap(vol_ft, ref_ft) < 0.0) is (relative_overlap < 0)
        assert_matches(np.asarray(aligned), -vol_ft if flipped else vol_ft)

    def test_compute_coarse_image_size_uses_particle_diameter(self):
        """RELION coarse_size should depend on particle diameter, not box size."""
        coarse_from_particle = compute_coarse_image_size(
            14.7,
            4.25,
            128,
            particle_diameter=200.0,
        )
        coarse_from_box = compute_coarse_image_size(
            14.7,
            4.25,
            128,
        )
        assert coarse_from_particle == 52
        assert coarse_from_box == 20
        assert coarse_from_particle > coarse_from_box

    def test_clamp_relion_coarse_image_size_caps_at_current_size(self):
        """RELION clamps coarse_size to current_size, not current_size/2."""
        coarse_size = compute_coarse_image_size(
            7.5,
            4.25,
            128,
            particle_diameter=200.0,
        )
        assert coarse_size == 100
        assert clamp_relion_coarse_image_size(coarse_size, current_size=60, box_size=128) == 60

    def test_clamp_relion_coarse_image_size_allows_small_even_sizes(self):
        """RELION allows adaptive coarse sizes below the generic current-size floor."""
        coarse_size = compute_coarse_image_size(
            30.0,
            4.25,
            128,
            particle_diameter=380.0,
        )
        assert coarse_size == 14
        assert clamp_relion_coarse_image_size(coarse_size, current_size=44, box_size=128) == 14

    def test_relion_optics_image_current_size_preserves_upward_ceil_boundary(self):
        """A rounded STAR angpix can add one even shell without changing model r_max."""
        sizes = relion_optics_image_current_sizes(
            56,
            model_box_size=384,
            model_pixel_size=float(np.float32(544.0 / 384.0)),
            optics_image_sizes=[384],
            optics_pixel_sizes=[1.416667],
        )
        assert_matches(sizes, np.array([58], dtype=np.int64))

    def test_relion_optics_image_current_size_identity_and_full_box_clamp(self):
        pixel_size = float(np.float32(544.0 / 384.0))
        assert_matches(
            relion_optics_image_current_sizes(
                56,
                model_box_size=384,
                model_pixel_size=pixel_size,
                optics_image_sizes=[384],
                optics_pixel_sizes=[pixel_size],
            ),
            np.array([56], dtype=np.int64),
        )
        assert_matches(
            relion_optics_image_current_sizes(
                384,
                model_box_size=384,
                model_pixel_size=pixel_size,
                optics_image_sizes=[384],
                optics_pixel_sizes=[1.416667],
            ),
            np.array([384], dtype=np.int64),
        )

    def test_local_order_transition_keeps_pre_update_coarse_size_and_updated_child_expansion(self):
        """RELION sizes pass 1 before updating order, but expands the new parent grid."""
        incoming_order = 3
        updated_parent_order = 4
        oversampling_order = 1
        fine_order = updated_parent_order + oversampling_order

        pass1_size = relion_local_pass1_current_size(
            pre_update_healpix_order=incoming_order,
            pixel_size=3.28,
            box_size=128,
            particle_diameter=280.0,
            current_size=122,
        )
        updated_order_size = compute_coarse_image_size(
            healpix_angular_step(updated_parent_order),
            3.28,
            128,
            particle_diameter=280.0,
        )
        child_rotations, rotation_parent, _ = get_oversampled_rotation_grid_from_samples(
            np.asarray([0], dtype=np.int32),
            fine_order - oversampling_order,
            oversampling_order=oversampling_order,
            return_rotation_indices=True,
        )
        child_translations, translation_parent = get_oversampled_translation_grid(
            np.zeros((1, 2), dtype=np.float32),
            pixel_offset=1.0,
            oversampling_order=oversampling_order,
        )

        assert pass1_size == 56
        assert updated_order_size == 110
        assert fine_order - oversampling_order == updated_parent_order
        assert child_rotations.shape[0] == rotation_parent.size == 8
        assert child_translations.shape[0] == translation_parent.size == 4
        assert child_rotations.shape[0] * child_translations.shape[0] == 32

    def test_replay_coarse_size_uses_previous_saved_order_not_live_post_mstep_state(self):
        assert (
            relion_expectation_coarse_size_order(
                state_healpix_order=4,
                replay_saved_healpix_order=3,
            )
            == 3
        )
        assert (
            relion_expectation_coarse_size_order(
                state_healpix_order=4,
                replay_saved_healpix_order=None,
            )
            == 4
        )
    def test_make_relion_direction_log_prior_matches_canonical_grid_indices(self):
        order = 2
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()
        rotations = np.asarray(get_relion_rotation_grid(order), dtype=np.float32)
        view_dirs = rotations[:, 2, :].astype(np.float64)
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
        expected_pixels = hp.vec2pix(
            2**order,
            view_dirs[:, 0],
            view_dirs[:, 1],
            view_dirs[:, 2],
            nest=True,
        )

        got = make_relion_direction_log_prior(
            direction_prior,
            order,
            rotations=rotations,
        )
        expected = np.log(direction_prior[expected_pixels]).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
        # RELION numbers C1 directions by NEST pixel (healpix_sampling.cpp:85), so on the unperturbed grid the
        # geometric lookup is the sample-index lookup: rotation r takes direction r % n_pixels.
        np.testing.assert_array_equal(got, make_relion_direction_log_prior(direction_prior, order))

    def test_make_relion_direction_log_prior_tracks_perturbed_view_directions(self):
        order = 3
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()

        perturbed_rotations = apply_relion_rotation_perturbation(
            np.asarray(get_relion_rotation_grid(order), dtype=np.float32),
            random_perturbation=0.3,
            angular_sampling_deg=360.0 / (6 * 2**order),
        ).astype(np.float32)
        view_dirs = perturbed_rotations[:, 2, :].astype(np.float64)
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
        expected_pixels = hp.vec2pix(
            2**order,
            view_dirs[:, 0],
            view_dirs[:, 1],
            view_dirs[:, 2],
            nest=True,
        )

        got = make_relion_direction_log_prior(
            direction_prior,
            order,
            rotations=perturbed_rotations,
        )
        expected = np.log(direction_prior[expected_pixels]).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_make_relion_direction_log_prior_default_keeps_sample_index_prior(self):
        order = 3
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.linspace(1.0, float(n_pixels), n_pixels, dtype=np.float32)
        direction_prior /= direction_prior.sum()

        got = make_relion_direction_log_prior(direction_prior, order)
        expected = np.log(np.repeat(direction_prior[None, :], rotation_grid_n_in_planes(order), axis=0).reshape(-1))
        np.testing.assert_allclose(got, expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_make_relion_direction_log_prior_preserves_zero_prior_as_hard_mask(self):
        order = 2
        n_rot = rotation_grid_size(order)
        n_pixels = n_rot // rotation_grid_n_in_planes(order)
        direction_prior = np.ones(n_pixels, dtype=np.float32)
        direction_prior[3] = 0.0
        direction_prior /= direction_prior.sum()

        got = make_relion_direction_log_prior(direction_prior, order)
        zero_direction_rows = np.arange(n_rot, dtype=np.int64) % n_pixels == 3

        assert np.isneginf(got[zero_direction_rows]).all()
        assert np.isfinite(got[~zero_direction_rows]).all()

    def test_normalize_direction_prior_per_half_preserves_relion_half_models(self):
        half1 = np.array([0.7, 0.3], dtype=np.float32)
        half2 = np.array([0.2, 0.8], dtype=np.float32)

        got = normalize_direction_prior_per_half([half1, half2])

        np.testing.assert_allclose(got[0], half1)
        np.testing.assert_allclose(got[1], half2)

    def test_normalize_direction_prior_per_half_keeps_shared_prior(self):
        shared = np.array([0.7, 0.3], dtype=np.float32)

        got = normalize_direction_prior_per_half(shared)

        np.testing.assert_allclose(got[0], shared)
        np.testing.assert_allclose(got[1], shared)
        assert got[0] is not got[1]

    def test_normalize_noise_variance_per_half_keeps_shared_noise(self):
        shared = jnp.arange(IMAGE_SIZE, dtype=jnp.float32) + 1.0

        got = _normalize_noise_variance_per_half(shared)

        assert len(got) == 2
        np.testing.assert_allclose(np.asarray(got[0]), np.asarray(shared))
        np.testing.assert_allclose(np.asarray(got[1]), np.asarray(shared))

    def test_normalize_noise_variance_per_half_preserves_relion_half_models(self):
        half1 = np.arange(IMAGE_SIZE, dtype=np.float32) + 1.0
        half2 = half1 * 2.0

        got = _normalize_noise_variance_per_half(np.stack([half1, half2]))

        np.testing.assert_allclose(np.asarray(got[0]), half1)
        np.testing.assert_allclose(np.asarray(got[1]), half2)

    def test_combined_noise_stats_sums_half_sufficient_statistics(self):
        """Class3D combines half accumulators before one RELION sigma2 update."""
        stats0 = NoiseStats(
            wsum_sigma2_noise=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
            wsum_sigma2_offset=7.0,
            sumw=11.0,
            wsum_noise_a2=jnp.array([0.5, 1.0, 1.5], dtype=jnp.float32),
        )
        stats1 = NoiseStats(
            wsum_sigma2_noise=jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([40.0, 50.0, 60.0], dtype=jnp.float32),
            wsum_sigma2_offset=70.0,
            sumw=13.0,
            wsum_noise_xa=jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32),
        )

        got = _combined_noise_stats([stats0, stats1])

        np.testing.assert_allclose(np.asarray(got.wsum_sigma2_noise), [11.0, 22.0, 33.0])
        np.testing.assert_allclose(np.asarray(got.wsum_img_power), [44.0, 55.0, 66.0])
        assert got.wsum_sigma2_offset == pytest.approx(77.0)
        assert got.sumw == pytest.approx(24.0)
        np.testing.assert_allclose(np.asarray(got.wsum_noise_a2), [0.5, 1.0, 1.5])
        np.testing.assert_allclose(np.asarray(got.wsum_noise_xa), [2.0, 4.0, 6.0])

    def test_relion_noise_stats_carry_optional_norm_scale_fields(self):
        stats0 = NoiseStats(
            wsum_sigma2_noise=jnp.array([1.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([2.0], dtype=jnp.float32),
            wsum_sigma2_offset=3.0,
            sumw=4.0,
            wsum_norm_correction=jnp.array([5.0, 6.0], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([7.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([8.0], dtype=jnp.float32),
        )
        stats1 = stats0._replace(
            wsum_sigma2_noise=jnp.array([10.0], dtype=jnp.float32),
            wsum_img_power=jnp.array([20.0], dtype=jnp.float32),
            wsum_norm_correction=jnp.array([50.0, 60.0], dtype=jnp.float32),
            wsum_scale_correction_xa=jnp.array([70.0], dtype=jnp.float32),
            wsum_scale_correction_aa=jnp.array([80.0], dtype=jnp.float32),
        )

        summed = _sum_noise_stats((stats0, stats1))
        combined = _combined_noise_stats([stats0, stats1])

        np.testing.assert_allclose(np.asarray(summed.wsum_norm_correction), [55.0, 66.0])
        np.testing.assert_allclose(np.asarray(summed.wsum_scale_correction_xa), [77.0])
        np.testing.assert_allclose(np.asarray(summed.wsum_scale_correction_aa), [88.0])
        assert combined.wsum_norm_correction is None
        assert combined.wsum_scale_correction_xa is None
        assert combined.wsum_scale_correction_aa is None

    def test_relion_refinement_runs_2_iterations(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
    ):
        """RELION-parity refinement completes 2 iterations on a tiny dataset."""
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=2,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        # Basic saved result structure
        fields = result.archive_fields()
        assert "mean" in fields
        assert "means" in fields
        assert "fsc" in fields
        assert "hard_assignments" in fields
        assert "current_sizes" in fields
        assert "fsc_history" in fields
        assert "pixel_resolutions" in fields
        assert "wall_times" in fields

        # RELION-specific keys
        assert "convergence_state" in fields
        assert "data_vs_prior_trajectory" in fields
        assert "healpix_order_trajectory" in fields
        assert "ave_Pmax_trajectory" in fields

    def test_relion_mode_does_not_finalize_after_max_iter_exhaustion(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
    ):
        """RELION does not run final all-data iteration just because max_iter ended."""
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=1, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is False
        assert len(result.history.wall_times) == 1
        assert len(result.history.current_sizes) == 1

    def test_relion_mode_joins_lowres_halves_on_first_iteration(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """RELION joins low-res half accumulators before the first local output iter."""

        join_calls = []
        original_join = regularization_relion.join_halves_at_low_resolution

        def spy_join(*args, **kwargs):
            join_calls.append(
                (
                    kwargs.get("current_resolution_angstrom"),
                    kwargs.get("preserve_inputs"),
                    kwargs.get("return_retained_first_numerator"),
                )
            )
            return original_join(*args, **kwargs)

        monkeypatch.setattr(
            regularization_relion,
            "join_halves_at_low_resolution",
            spy_join,
        )

        refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=1, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(
                    low_resol_join_halves_angstrom=40.0,
                    relion_firstiter_ini_high_angstrom=30.0,
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        expected_resolution = shell_index_to_resolution_angstrom(1, IMAGE_SHAPE[0], half_datasets[0].voxel_size)
        assert len(join_calls) == 1
        assert join_calls[0][0] == pytest.approx(expected_resolution)
        assert join_calls[0][1] is False
        assert join_calls[0][2] is True

    def test_relion_final_iteration_scores_half_maps_after_convergence(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """The final joined reconstruction still scores each half against its own map."""
        original_update = convergence_policy.update_refinement_state
        original_reconstruct = numbered_reconstruction_module._reconstruct_volume_eager
        engine_calls = []
        reconstruction_calls = []
        expected_accuracy_current_sizes = []

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def spy_reconstruct(*args, **kwargs):
            volume = original_reconstruct(*args, **kwargs)
            reconstruction_calls.append((kwargs, volume))
            return volume

        def fake_expected_accuracy(**kwargs):
            expected_accuracy_current_sizes.append(int(kwargs["current_image_size"]))
            n_classes = int(np.asarray(kwargs["class_weights"]).size)
            n_trials = int(np.asarray(kwargs["best_eulers_deg"]).shape[0])
            return SimpleNamespace(
                acc_rot=1.25,
                acc_trans_angstrom=1.5,
                acc_rot_per_class=np.full(n_classes, 1.25, dtype=np.float64),
                acc_trans_per_class_angstrom=np.full(n_classes, 1.5, dtype=np.float64),
                class_counts=np.full(n_classes, n_trials, dtype=np.int64),
                trial_local_indices=np.arange(n_trials, dtype=np.int64),
                trial_particle_ids=np.arange(n_trials, dtype=np.int64),
            )

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        install_fake_adaptive_engine(monkeypatch, engine_calls, Ft_y=_random_half_maps(engine_calls))
        monkeypatch.setattr(numbered_reconstruction_module, "_reconstruct_volume_eager", spy_reconstruct)
        monkeypatch.setattr(
            expected_accuracy_module,
            "relion_half1_trial_order",
            lambda n_particles, *_args, **_kwargs: np.arange(n_particles, dtype=np.int64),
        )
        monkeypatch.setattr(
            expected_accuracy_module,
            "estimate_relion_expected_accuracy",
            fake_expected_accuracy,
        )

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(
                    low_resol_join_halves_angstrom=0.0,
                    perturb_seed=17,
                    optimizer_random_seed=17,
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is True
        assert len(result.history.wall_times) == 2
        assert len(engine_calls) == 4
        assert not np.array_equal(np.asarray(engine_calls[-2]["means"]), np.asarray(engine_calls[-1]["means"]))
        assert expected_accuracy_current_sizes == [IMAGE_SHAPE[0]]
        assert result.final_pass.expected_accuracy_status == "ok"
        assert result.final_pass.acc_rot == pytest.approx(1.25)

        # Final reconstruction produces two unfiltered halves (first, from the
        # pre-join accumulators, so the join can update them in place), then the
        # two regularized halves and the merged map (last: its accumulators are
        # the halves' sum, formed in place once both half maps are solved), all at
        # Nyquist. Check executed calls so grouping half pairs cannot silently
        # omit or reorder a saved product.
        final_calls = reconstruction_calls[-5:]
        assert len(final_calls) == 5
        assert [call[0]["tau"] is None for call in final_calls] == [True, True, False, False, False]
        assert all(call[0]["current_size"] == IMAGE_SHAPE[0] for call in final_calls)
        # The unfiltered halves take the solver's spherical mask and grid correction (its defaults).
        assert all(call[0].get("use_spherical_mask", True) is True for call in final_calls[:2])
        assert all(call[0].get("grid_correct", True) is True for call in final_calls[:2])
        products = [*result.maps.unfiltered_means, *result.maps.means, result.maps.mean]
        for product, (_, reconstructed) in zip(products, final_calls, strict=True):
            assert_matches(np.asarray(product), np.asarray(reconstructed).reshape(-1))

    def test_relion_final_iteration_tau2_uses_half_accumulators(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Final all-data tau2 uses half weights; only final reconstruction sums them."""
        original_update = convergence_policy.update_refinement_state
        original_tau2 = regularization_relion.compute_relion_tau2_from_weights
        ctf_values = [2.0, 4.0, 7.0, 11.0]
        engine_call = {"idx": 0}
        whole_tau2_calls = []

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def fake_adaptive(experiment_dataset, means, mean_variance, noise_variance, coarse_rotations,
                          coarse_translations, fine_rotations, *grids, **kwargs):
            idx = engine_call["idx"]
            engine_call["idx"] += 1
            ctf_value = ctf_values[idx]
            return adaptive_result(
                experiment_dataset,
                means,
                fine_rotations,
                kwargs,
                Ft_y=lambda _k, size: jnp.ones(size, dtype=jnp.complex64),
                Ft_ctf=lambda _k, size: jnp.full(size, ctf_value, dtype=jnp.complex64),
            )

        def spy_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs):
            if kwargs.get("is_whole_instead_of_half", False):
                whole_tau2_calls.append(
                    (
                        np.asarray(Ft_ctf_0).real.copy(),
                        np.asarray(Ft_ctf_1).real.copy(),
                    )
                )
            return original_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs)

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive)
        monkeypatch.setattr(regularization_relion, "compute_relion_tau2_from_weights", spy_tau2)

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is True
        assert engine_call["idx"] == 4
        assert len(whole_tau2_calls) == 1
        final_half0, final_half1 = whole_tau2_calls[0]
        np.testing.assert_allclose(final_half0, ctf_values[2], atol=0.0)
        np.testing.assert_allclose(final_half1, ctf_values[3], atol=0.0)

    @pytest.mark.parametrize(
        ("n_classes", "join_angstrom", "expected"),
        [
            (1, 40.0, ["unfiltered:held", "join:held", "halfmap_prior:released", "halfmap_solve:released"]),
            (1, 0.0, ["unfiltered:held", "halfmap_prior:released", "halfmap_solve:released"]),
            (1, None, ["unfiltered:held", "halfmap_prior:released", "halfmap_solve:released"]),
            (2, 40.0, ["class_priors", "class_solve"]),
        ],
    )
    def test_relion_final_iteration_runs_k1_prejoin_sequence_in_order(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
        n_classes,
        join_angstrom,
        expected,
    ):
        """Final K1 makes unfiltered maps, joins if asked, then releases the pass outputs' accumulators.

        Class3D runs none of the three steps, even with a positive join resolution.
        """
        original_update = convergence_policy.update_refinement_state
        original_outputs = finalization.PerHalfOutputs
        final_reconstruction = finalization.final_reconstruction
        collectors = []
        events = []
        operands = {}

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def record_collector(*args, **kwargs):
            collectors.append(original_outputs(*args, **kwargs))
            return collectors[-1]

        def collector_state():
            held = [value is not None for value in (*collectors[-1].Ft_y, *collectors[-1].Ft_ctf)]
            assert all(held) or not any(held)
            return "held" if all(held) else "released"

        def spy(owner, name, label, *, with_collector_state=True):
            original = getattr(owner, name)

            def wrapped(*args, **kwargs):
                events.append(f"{label}:{collector_state()}" if with_collector_state else label)
                operands[label] = (args, kwargs)
                operands[label + ":result"] = original(*args, **kwargs)
                return operands[label + ":result"]

            monkeypatch.setattr(owner, name, wrapped)

        monkeypatch.setattr(convergence_policy, "update_refinement_state", force_convergence_after_first_iter)
        monkeypatch.setattr(finalization, "PerHalfOutputs", record_collector)
        spy(final_reconstruction, "reconstruct_unfiltered_halfmaps", "unfiltered")
        spy(finalization, "join_half_accumulators_at_low_resolution", "join")
        spy(final_reconstruction, "compute_final_halfmap_prior", "halfmap_prior")
        spy(final_reconstruction, "reconstruct_final_halfmaps", "halfmap_solve")
        spy(final_reconstruction, "compute_final_class_priors", "class_priors", with_collector_state=False)
        spy(final_reconstruction, "reconstruct_final_class_maps", "class_solve", with_collector_state=False)

        k_class = {}
        if n_classes > 1:
            k_class["k_class"] = KClassOptions(
                n_classes=n_classes,
            )
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=join_angstrom),
                **k_class,
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.final_all_data_ran is True
        assert len(collectors) == 1
        assert events == expected
        if n_classes > 1:
            assert result.maps.unfiltered_means is None
            return
        assert result.maps.unfiltered_means is operands["unfiltered:result"]
        # The unfiltered maps read the pass outputs' own arrays, before any join.
        prejoin_numerators, prejoin_denominators = operands["unfiltered"][0]
        prior_numerators, prior_denominators = operands["halfmap_prior"][0]
        if "join" in operands:
            (join_numerators, join_denominators), join_kwargs = operands["join"]
            assert all(a is b for a, b in zip(join_numerators, prejoin_numerators, strict=True))
            assert all(a is b for a, b in zip(join_denominators, prejoin_denominators, strict=True))
            assert join_kwargs["preserve_inputs"] is False
            # The prior and the final solve read what the join returned.
            joined = operands["join:result"]
            assert all(a is b for a, b in zip((*prior_numerators, *prior_denominators), joined, strict=True))
        else:
            assert all(a is b for a, b in zip(prior_numerators, prejoin_numerators, strict=True))
            assert all(a is b for a, b in zip(prior_denominators, prejoin_denominators, strict=True))

    def test_relion_final_iteration_uses_learned_k1_direction_prior(
        self,
        half_datasets,
        init_volume,
        rotations,
        translations,
        monkeypatch,
    ):
        """The final K=1 all-data E-step uses the previous iter's pdf_direction."""
        original_update = convergence_policy.update_refinement_state
        engine_calls = []
        custom_eulers = np.zeros((N_ROTATIONS, 3), dtype=np.float32)
        learned_direction_priors = [
            np.array([0.7, 0.3], dtype=np.float32),
            np.array([0.2, 0.8], dtype=np.float32),
        ]
        expected_rotation_log_priors = [
            np.linspace(0.0, -0.4, N_ROTATIONS, dtype=np.float32),
            np.linspace(-1.0, -1.4, N_ROTATIONS, dtype=np.float32),
        ]
        collapse_calls = []
        make_prior_calls = []

        # RELION-parity refinement on a GPU backend builds its scorer rotations with the
        # strict CUDA primitive and refuses to run with custom CUDA disabled; on CPU the
        # NumPy scorer path runs. Do not inherit a caller's RECOVAR_DISABLE_CUDA.
        monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def fake_rotation_grid_size(_order, *, symmetry='C1'):
            return N_ROTATIONS

        def fake_collapse_rotation_posterior_to_direction_prior(rotation_posterior_sums, healpix_order, dtype=np.float32, *, symmetry='C1'):
            collapse_calls.append((np.asarray(rotation_posterior_sums).shape, int(healpix_order)))
            return np.asarray(learned_direction_priors[len(collapse_calls) - 1], dtype=dtype)

        def fake_make_relion_direction_log_prior(direction_prior, healpix_order, dtype=np.float32, *, symmetry='C1'):
            prior = np.asarray(direction_prior, dtype=dtype)
            make_prior_calls.append((prior.copy(), int(healpix_order)))
            if matches(prior, learned_direction_priors[0]):
                return expected_rotation_log_priors[0]
            if matches(prior, learned_direction_priors[1]):
                return expected_rotation_log_priors[1]
            raise AssertionError(f"unexpected direction prior {prior}")

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        install_fake_adaptive_engine(monkeypatch, engine_calls)
        monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
        monkeypatch.setattr(
            sampling_module,
            "relion_scoring_rotation_grid",
            lambda _order, dtype=None, *, symmetry='C1': sampling_module.RotationGrid(rotations=np.asarray(rotations, dtype=np.float32), rotation_eulers=custom_eulers, healpix_order=_order, symmetry=symmetry),
        )
        # Direction-prior learning and scoring expansion share their owner.
        monkeypatch.setattr(
            orientation_priors_module,
            "collapse_rotation_posterior_to_direction_prior",
            fake_collapse_rotation_posterior_to_direction_prior,
        )
        monkeypatch.setattr(
            orientation_priors_module,
            "make_relion_direction_log_prior",
            fake_make_relion_direction_log_prior,
        )
        monkeypatch.setattr(
            orientation_priors_module,
            "make_relion_direction_log_prior",
            fake_make_relion_direction_log_prior,
        )

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is True
        run_em_rotation_priors = [
            None if call["kwargs"].get("rotation_log_prior") is None else np.asarray(call["kwargs"]["rotation_log_prior"])
            for call in engine_calls
        ]
        assert run_em_rotation_priors[:2] == [None, None]
        np.testing.assert_allclose(run_em_rotation_priors[-2], expected_rotation_log_priors[0])
        np.testing.assert_allclose(run_em_rotation_priors[-1], expected_rotation_log_priors[1])
        assert len(collapse_calls) == 2
        assert [call[0] for call in collapse_calls] == [(N_ROTATIONS,), (N_ROTATIONS,)]
        assert [call[1] for call in collapse_calls] == [2, 2]
        assert len(make_prior_calls) == 4
        assert_matches(make_prior_calls[-2][0], learned_direction_priors[0])
        assert_matches(make_prior_calls[-1][0], learned_direction_priors[1])

    def test_relion_final_iteration_keeps_translation_prior(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """The final all-data E-step still uses RELION's pdf_offset prior."""
        original_update = convergence_policy.update_refinement_state
        engine_calls = []

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        install_fake_adaptive_engine(monkeypatch, engine_calls)

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is True
        run_em_translation_priors = [call["kwargs"].get("translation_log_prior") for call in engine_calls]
        run_em_translation_prior_centers = [call["kwargs"].get("translation_prior_centers") for call in engine_calls]
        assert len(run_em_translation_priors) == 4
        assert len(run_em_translation_prior_centers) == 4
        for final_prior in run_em_translation_priors[-2:]:
            assert final_prior is not None
            final_prior = np.asarray(final_prior)
            assert final_prior.size > 0
            assert np.all(np.isfinite(final_prior))
        for final_centers in run_em_translation_prior_centers[-2:]:
            assert final_centers is not None
            final_centers = np.asarray(final_centers)
            assert final_centers.shape[-1] == 2
            assert np.all(np.isfinite(final_centers))

    def test_relion_final_iteration_uses_each_random_subsets_noise_variance(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """The joined final E-step retains each random subset's noise model."""
        original_update = convergence_policy.update_refinement_state
        engine_calls = []
        replay_noise_h1 = np.linspace(2.0, 3.0, IMAGE_SIZE, dtype=np.float32)
        replay_noise_h2 = np.linspace(5.0, 6.0, IMAGE_SIZE, dtype=np.float32)

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        install_fake_adaptive_engine(monkeypatch, engine_calls)

        result = _refine_replaying(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
            ),
            relion_replay=RelionReplay(
                replay_iteration_overrides=[
                    None,
                    {"noise_variance": [replay_noise_h1, replay_noise_h2]},
                ]
            ),
        )

        assert result.convergence_state.has_converged is True
        run_em_noise = [np.asarray(call["noise_variance"], dtype=np.float32) for call in engine_calls]
        assert len(run_em_noise) == 4
        assert_matches(run_em_noise[-2], replay_noise_h1)
        assert_matches(run_em_noise[-1], replay_noise_h2)

    def test_relion_final_iteration_uses_local_search_when_converged_state_is_local(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """The final all-data E-step follows RELION local-search state."""
        original_update = convergence_policy.update_refinement_state
        engine_calls = []
        local_calls = []

        def fake_rotation_grid_size(_order, symmetry="C1"):
            assert symmetry == "C1"
            return N_ROTATIONS

        def fake_rotation_grid_n_in_planes(_order):
            return 1

        def fake_scoring_rotation_grid(order, dtype=None, *, symmetry="C1"):
            del dtype
            n_rotations = fake_rotation_grid_size(order, symmetry=symmetry)
            return sampling_module.RotationGrid(rotations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], n_rotations, axis=0), rotation_eulers=np.zeros((n_rotations, 3), dtype=np.float32), healpix_order=order, symmetry=symmetry)

        def fake_get_relion_rotation_grid_eulers(order, *args, **kwargs):
            del args, kwargs
            return fake_scoring_rotation_grid(order).rotation_eulers

        def fake_apply_relion_rotation_perturbation_to_eulers(
            eulers,
            random_perturbation,
            angular_sampling_deg,
            *,
            dtype=np.float32,
        ):
            _ = (random_perturbation, angular_sampling_deg)
            n_rows = int(np.asarray(eulers).shape[0])
            score = np.repeat(np.eye(3, dtype=dtype)[None], n_rows, axis=0)
            public_eulers = np.asarray(eulers, dtype=dtype)
            if np.asarray(eulers).dtype == np.float64:
                score = np.repeat((13.0 * np.eye(3, dtype=np.float32))[None], n_rows, axis=0)
            return score, public_eulers

        def force_converged_local_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            updated.do_local_search = True
            updated.healpix_order = max(updated.healpix_order, updated.auto_local_healpix_order)
            return updated

        def fake_local_search(
            experiment_dataset,
            mean,
            noise_variance,
            prior_rotations,
            rotation_grid_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            disc_type,
            current_size,
            **kwargs,
        ):
            _ = (
                noise_variance,
                rotation_grid_rotations,
                sigma_rot,
                sigma_psi,
                translations,
                sigma_offset_angstrom,
                disc_type,
            )
            local_calls.append(
                {
                    "n_units": int(experiment_dataset.n_units),
                    "mean_id": id(mean),
                    "prior_rotations": np.asarray(prior_rotations, dtype=np.float32).copy(),
                    "prior_translations": np.asarray(prior_translations, dtype=np.float32).copy(),
                    "translation_prior_centers": np.asarray(
                        kwargs["translation_prior_centers"],
                        dtype=np.float32,
                    ).copy(),
                    "image_pre_shifts": np.asarray(kwargs["image_pre_shifts"], dtype=np.float32).copy(),
                    "healpix_order": int(healpix_order),
                    "current_size": current_size,
                    "accumulate_noise": kwargs["accumulate_noise"],
                    "return_best_pose_details": kwargs["return_best_pose_details"],
                    "rotation_grid_mstep_rotations": np.asarray(
                        kwargs["rotation_grid_mstep_rotations"],
                        dtype=np.float32,
                    ).copy(),
                    "generate_relion_mstep_rotations": kwargs["generate_relion_mstep_rotations"],
                }
            )
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
            base_outputs = (
                jnp.zeros(recon_vol_size, dtype=jnp.complex64),
                jnp.ones(recon_vol_size, dtype=jnp.complex64),
                np.zeros(experiment_dataset.n_units, dtype=np.int32),
            )
            best_pose_details = (
                np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0),
                np.zeros((experiment_dataset.n_units, 2), dtype=np.float32),
            )
            relion_stats = RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(
                    maximization_module.rotation_grid_size(healpix_order),
                    dtype=jnp.float32,
                ),
            )
            noise_stats = NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=1.0,
                sumw=float(experiment_dataset.n_units),
            )
            return _mock_local_search_result(
                base_outputs,
                relion_stats,
                noise_stats,
                kwargs,
                experiment_dataset.n_units,
                best_pose_details,
            )

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_converged_local_after_first_iter,
        )
        install_fake_adaptive_engine(monkeypatch, engine_calls)
        monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
        monkeypatch.setattr(
            sampling_module,
            "relion_scoring_rotation_grid",
            fake_scoring_rotation_grid,
        )
        monkeypatch.setattr(
            sampling_module,
            "_get_relion_rotation_grid_eulers_float64",
            lambda order, *, symmetry='C1': fake_get_relion_rotation_grid_eulers(order).astype(np.float64),
        )
        monkeypatch.setattr(
            sampling_module,
            "apply_relion_rotation_perturbation_to_eulers",
            fake_apply_relion_rotation_perturbation_to_eulers,
        )
        monkeypatch.setitem(
            make_relion_direction_log_prior.__globals__,
            "rotation_grid_size",
            fake_rotation_grid_size,
        )
        monkeypatch.setitem(
            make_relion_direction_log_prior.__globals__,
            "rotation_grid_n_in_planes",
            fake_rotation_grid_n_in_planes,
        )
        monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_local_search))

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=4),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                local_search=LocalSearchOptions(auto_local_healpix_order=4),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is True
        # The first iteration's global E-step, one per half; the final iteration is local.
        assert len(engine_calls) == 2
        assert len(local_calls) == 2
        assert local_calls[0]["mean_id"] != local_calls[1]["mean_id"]
        assert all(call["current_size"] == IMAGE_SHAPE[0] for call in local_calls)
        assert all(call["accumulate_noise"] for call in local_calls)
        assert all(call["return_best_pose_details"] for call in local_calls)
        assert all(call["healpix_order"] == 4 for call in local_calls)
        assert all(call["generate_relion_mstep_rotations"] is True for call in local_calls)
        for call in local_calls:
            assert_matches(
                call["rotation_grid_mstep_rotations"],
                np.repeat((13.0 * np.eye(3, dtype=np.float32))[None], N_ROTATIONS, axis=0),
            )
        for call, dataset in zip(local_calls, half_datasets):
            assert call["prior_rotations"].shape == (dataset.n_units, 3)
            assert call["prior_translations"].shape == (dataset.n_units, 2)
            assert call["translation_prior_centers"].shape == (dataset.n_units, 2)
            assert call["image_pre_shifts"].shape == (dataset.n_units, 2)
            assert np.all(np.isfinite(call["prior_translations"]))
            assert np.all(np.isfinite(call["translation_prior_centers"]))
            assert np.all(np.isfinite(call["image_pre_shifts"]))

    def test_relion_final_iteration_supports_k_class(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """K-class refinement can run the final all-data iteration."""
        original_update = convergence_policy.update_refinement_state

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
                k_class=KClassOptions(
                    n_classes=2,
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is True
        assert len(result.history.wall_times) == 2
        assert np.asarray(result.maps.class_means).shape == (2, VOLUME_SIZE)
        np.testing.assert_allclose(np.sum(result.maps.class_weights), 1.0, rtol=1e-6, atol=1e-6)
        for half_classes in result.maps.class_assignments:
            assert np.asarray(half_classes).shape == (N_IMAGES // 2,)

    def test_relion_final_iteration_supports_k4_exactly_once(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """K=4 refinement runs exactly one final all-data iteration."""
        original_update = convergence_policy.update_refinement_state
        original_run_final_all_data = iteration_loop_module.finalization.run_final_all_data
        final_all_data_pass_count = 0

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        def record_final_all_data_pass(*args, **kwargs):
            nonlocal final_all_data_pass_count
            final_all_data_pass_count += 1
            return original_run_final_all_data(*args, **kwargs)

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        monkeypatch.setattr(
            iteration_loop_module.finalization,
            "run_final_all_data",
            record_final_all_data_pass,
        )

        n_classes = 4
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
                k_class=KClassOptions(
                    n_classes=n_classes,
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is True
        assert result.final_all_data_ran is True
        assert len(result.history.wall_times) == 2
        assert np.asarray(result.maps.class_means).shape == (n_classes, VOLUME_SIZE)
        np.testing.assert_allclose(np.sum(result.maps.class_weights), 1.0, rtol=1e-6, atol=1e-6)
        for half_classes in result.maps.class_assignments:
            assert np.asarray(half_classes).shape == (N_IMAGES // 2,)
        for half_classes in result.final_pass.class_assignments:
            assert np.asarray(half_classes).shape == (N_IMAGES // 2,)
        assert final_all_data_pass_count == 1

    def test_relion_k4_does_not_finalize_after_max_iter_even_when_diagnostic_force_enabled(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """K-class max-iteration exhaustion never enables final all-data."""
        final_all_data_pass_count = 0
        original_run_final_all_data = iteration_loop_module.finalization.run_final_all_data

        def record_final_all_data_pass(*args, **kwargs):
            nonlocal final_all_data_pass_count
            final_all_data_pass_count += 1
            return original_run_final_all_data(*args, **kwargs)

        monkeypatch.setattr(
            iteration_loop_module.finalization,
            "run_final_all_data",
            record_final_all_data_pass,
        )

        n_classes = 4
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=1, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
                k_class=KClassOptions(
                    n_classes=n_classes,
                ),
                final_pass=FinalPassOptions(after_max_iter=True),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is False
        assert result.final_all_data_ran is False
        assert np.asarray(result.maps.class_means).shape == (n_classes, VOLUME_SIZE)
        assert final_all_data_pass_count == 0

    def test_relion_final_iteration_k_class_adaptive_uses_sparse_pass2_route(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Adaptive K-class final all-data should not fall back to direct dense scoring."""
        original_update = convergence_policy.update_refinement_state

        def force_convergence_after_first_iter(*args, **kwargs):
            updated = original_update(*args, **kwargs)
            updated.has_converged = True
            return updated

        adaptive_calls = []

        def fake_adaptive_k_class(
            experiment_dataset,
            means,
            mean_variance,
            noise_variance,
            coarse_rotations,
            coarse_translations,
            fine_rotations,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
            disc_type,
            **kwargs,
        ):
            _ = (
                mean_variance,
                noise_variance,
                coarse_translations,
                trans_parent_map,
                disc_type,
            )
            adaptive_calls.append(
                {
                    "coarse_current_size": kwargs.get("coarse_current_size"),
                    "fine_current_size": kwargs.get("fine_current_size"),
                    "sparse_pass2": kwargs.get("sparse_pass2"),
                    "relion_fine_mstep_prune": kwargs.get("relion_fine_mstep_prune"),
                    "return_best_pose_details": kwargs.get("return_best_pose_details"),
                    "n_coarse_rot": int(np.asarray(coarse_rotations).shape[0]),
                    "n_fine_rot": int(np.asarray(fine_rotations).shape[0]),
                }
            )
            n_classes = int(np.asarray(means).shape[0])
            n_images = int(experiment_dataset.n_units)
            n_shells = int(experiment_dataset.image_shape[0]) // 2 + 1
            padding_factor = int(kwargs.get("reconstruction_padding_factor", 1))
            recon_vol_size = int(np.prod(experiment_dataset.volume_shape)) * (padding_factor**3)
            n_fine_rot = int(np.asarray(fine_rotations).shape[0])
            per_class_stats = tuple(
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(n_fine_rot, dtype=jnp.float32),
                )
                for _ in range(n_classes)
            )
            per_class_noise = tuple(
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images) / float(n_classes),
                )
                for _ in range(n_classes)
            )
            aggregate_noise = NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_images),
            )
            return KClassEMResult(
                new_means=None,
                Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
                Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
                per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
                class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                class_responsibilities=jnp.full((n_classes, n_images), 1.0 / n_classes, dtype=jnp.float32),
                class_posterior_sums=jnp.full(n_classes, n_images / n_classes, dtype=jnp.float32),
                stats=per_class_stats[0],
                per_class_stats=per_class_stats,
                noise_stats=per_class_noise,
                aggregate_noise_stats=aggregate_noise,
                best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
                best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
                best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
                significant_counts=jnp.ones(n_images, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            convergence_policy,
            "update_refinement_state",
            force_convergence_after_first_iter,
        )
        monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive_k_class)
        monkeypatch.setattr(iteration_planning_module, "relion_coarse_image_size", lambda *_args, **_kwargs: 4)
        monkeypatch.setattr(finalization, "relion_coarse_image_size", lambda *_args, **_kwargs: 4)

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(max_iter=2, init_current_size=4, init_healpix_order=1, max_healpix_order=1),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                adaptive=stand_in.adaptive(adaptive_oversampling=1),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
                k_class=KClassOptions(
                    n_classes=2,
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert result.convergence_state.has_converged is True
        assert len(adaptive_calls) == 4
        final_calls = adaptive_calls[-2:]
        assert all(call["coarse_current_size"] == 4 for call in final_calls)
        assert all(call["fine_current_size"] == IMAGE_SHAPE[0] for call in final_calls)
        assert all(call["sparse_pass2"] is True for call in final_calls)
        assert all(call["relion_fine_mstep_prune"] is True for call in final_calls)
        assert all(call["n_fine_rot"] >= call["n_coarse_rot"] for call in final_calls)

    def test_relion_mode_finite_outputs(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
    ):
        """RELION mode produces finite volumes and valid assignments."""
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=2,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        # Final mean should be finite
        assert np.all(np.isfinite(np.array(result.maps.mean))), "Mean not finite"
        # FSC should be computed
        assert result.history.fsc_history[-1] is not None
        # Hard assignments valid
        for k in range(2):
            ha = result.numbered.hard_assignments[k]
            assert ha is not None
            assert np.all(ha >= 0)

    def test_relion_mode_dense_k_class_finite_outputs(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
    ):
        """Dense non-adaptive RELION loop supports an explicit class axis."""
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                k_class=KClassOptions(
                    n_classes=2,
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert np.all(np.isfinite(np.asarray(result.maps.mean)))
        assert np.asarray(result.maps.class_means).shape == (2, VOLUME_SIZE)
        assert np.asarray(result.maps.means[0]).shape == (2, VOLUME_SIZE)
        assert np.asarray(result.maps.means[1]).shape == (2, VOLUME_SIZE)
        np.testing.assert_allclose(np.asarray(result.maps.means[0]), np.asarray(result.maps.means[1]))
        np.testing.assert_allclose(np.sum(result.maps.class_weights), 1.0, rtol=1e-6, atol=1e-6)
        assert len(result.history.class_mstep_weight_trajectory) == 1
        assert len(result.history.class_assignment_history) == 1
        assert_matches(
            result.history.class_assignment_history[0],
            np.concatenate(
                [
                    np.asarray(result.maps.class_assignments[0], dtype=np.int32),
                    np.asarray(result.maps.class_assignments[1], dtype=np.int32),
                ],
            ),
        )
        for half_idx in range(2):
            assert result.maps.class_assignments[half_idx].shape == (half_datasets[half_idx].n_units,)
            assert np.all(result.maps.class_assignments[half_idx] >= 0)

    def test_relion_mode_uses_engine_pmax(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """ave_Pmax should use half 1's engine posterior maxima, as RELION MPI does."""
        # Each half's engine reports its own posterior maxima and weight total (sumw).
        pmax_per_half = [
            np.linspace(0.2, 0.9, half_datasets[0].n_units, dtype=np.float32),
            np.linspace(0.6, 0.1, half_datasets[1].n_units, dtype=np.float32),
        ]
        engine_calls = []

        def fake_adaptive(experiment_dataset, means, mean_variance, noise_variance, coarse_rotations,
                          coarse_translations, fine_rotations, *grids, **kwargs):
            pmax = pmax_per_half[len(engine_calls)]
            engine_calls.append(kwargs)
            return adaptive_result(experiment_dataset, means, fine_rotations, kwargs, max_posterior=pmax)

        monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive)
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                adaptive=stand_in.adaptive(adaptive_oversampling=1),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert len(engine_calls) == 2
        half1_mass = float(half_datasets[0].n_units)
        expected_ave_pmax = float(np.sum(pmax_per_half[0], dtype=np.float64) / half1_mass)
        assert result.history.ave_Pmax_trajectory == pytest.approx([expected_ave_pmax], abs=1e-6)
        assert result.history.ave_Pmax_denominator_trajectory == pytest.approx([half1_mass], abs=1e-6)
        assert result.convergence_state.ave_Pmax == pytest.approx(expected_ave_pmax, abs=1e-6)

    @pytest.mark.gpu  # pass 2 runs only on the device-resident engine
    def test_relion_mode_forwards_particle_diameter_to_coarse_size(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Adaptive RELION mode should pass the explicit particle diameter through."""
        recorded = {"particle_diameter": None}

        class _Recorded(RuntimeError):
            pass

        def wrap_coarse_image_size(*args, **kwargs):
            # Stop here: the mock half sets have no RELION CUDA preprocessing for pass 1.
            recorded["particle_diameter"] = kwargs.get("particle_diameter")
            raise _Recorded

        monkeypatch.setattr(
            iteration_planning_module,
            "relion_coarse_image_size",
            wrap_coarse_image_size,
        )

        with pytest.raises(_Recorded):
            refine_single_volume(
                half_datasets,
                StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
                HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
                translations,
                options=stand_in.options(
                    schedule=stand_in.schedule(
                        max_iter=1,
                        init_current_size=16,
                        init_healpix_order=1,
                        max_healpix_order=2,
                        particle_diameter_ang=200.0,
                    ),
                    execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=20),
                    adaptive=stand_in.adaptive(adaptive_oversampling=1),
                ),
                observer=RunObserver(), source=InputSource(),
            )

        assert recorded["particle_diameter"] == pytest.approx(200.0)

    def test_relion_translation_log_prior_matches_source_pdf_offset(self, translations):
        log_prior = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            prior_centers=np.zeros(2, dtype=np.float32),
        )
        translations_px = np.asarray(translations)
        expected = -0.5 * np.sum(translations_px**2, axis=1) * (4.25**4) / (10.0**2)
        np.testing.assert_allclose(log_prior, expected, rtol=1e-6, atol=1e-6)
        assert int(np.argmax(log_prior)) == 0

    def test_relion_translation_log_prior_is_flat_without_offset_prior(self, translations):
        log_prior = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            prior_centers=None,
        )
        assert_matches(log_prior, np.zeros(len(translations), dtype=np.float32))

    def test_relion_translation_log_prior_uses_offset_range_when_active(self, translations):
        log_prior_sigma = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            prior_centers=np.zeros(2, dtype=np.float32),
        )
        log_prior_range = make_relion_translation_log_prior(
            np.asarray(translations),
            voxel_size=4.25,
            sigma_offset_angstrom=10.0,
            prior_centers=np.zeros(2, dtype=np.float32),
            offset_range_pixels=3.0,
        )
        # RELION uses sigma = offset_range / 3 while a finite search range is active.
        assert log_prior_range[0] == pytest.approx(0.0)
        assert log_prior_range[1] < log_prior_sigma[1]

    def test_relion_translation_search_base_uses_integer_prescoring_shift(self):
        prev = np.array([[0.5, -0.5], [1.5, -1.5], [-0.49, 0.49]], dtype=np.float32)
        expected = np.array([[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(relion_translation_search_base(prev), expected, rtol=1e-6, atol=1e-6)
        near_half = np.array([[0.49999999, -0.49999999], [0.50000001, -0.50000001]], dtype=np.float64)
        assert_matches(relion_translation_search_base(near_half), np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32))
        assert relion_translation_search_base(np.array([], dtype=np.float32)).shape == (0, 2)

    def test_relion_integer_pre_shift_uses_zero_fill_real_space_convention(self):
        image = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        shifted = apply_relion_integer_pre_shifts(image, np.array([[1, -1]], dtype=np.int32))
        expected = np.array(
            [
                [
                    [0.0, 3.0, 4.0],
                    [0.0, 6.0, 7.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=np.float32,
        )
        assert_matches(shifted, expected)

    def test_integer_pre_shifts_only_selects_integral_offsets(self):
        shifts = np.array([[1.0, -1.0], [0.5, 0.0]], dtype=np.float32)
        assert_matches(
            integer_pre_shifts_or_none(shifts, np.array([0], dtype=np.int32)),
            np.array([[1, -1]], dtype=np.int32),
        )
        assert integer_pre_shifts_or_none(shifts, np.array([1], dtype=np.int32)) is None

    def test_relion_translation_prior_center_matches_accelerated_pdf_offset_units(self):
        prev = np.array([[0.0, -1.0], [1.0, 0.0], [-0.82310355, -0.82310355]], dtype=np.float32)
        expected = np.array([[0.0, 1.0 / 4.25], [-1.0 / 4.25, 0.0], [1.0 / 4.25, 1.0 / 4.25]], dtype=np.float32)
        np.testing.assert_allclose(
            relion_translation_prior_center(prev, voxel_size=4.25),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )
        explicit_prior = np.zeros((1, 2), dtype=np.float32)
        np.testing.assert_allclose(
            relion_translation_prior_center(prev[:1], voxel_size=4.25, prior_offsets=explicit_prior),
            np.array([[0.0, 1.0 / 4.25]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_relion_sigma_offset_prior_center_matches_store_weighted_sums_units(self):
        prev = np.array([[0.0, -1.0], [1.0, 0.0], [-0.82310355, -0.82310355]], dtype=np.float32)
        expected = np.array([[0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(
            relion_sigma_offset_prior_center(prev),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )
        explicit_prior = np.zeros((1, 2), dtype=np.float32)
        np.testing.assert_allclose(
            relion_sigma_offset_prior_center(prev[:1], prior_offsets=explicit_prior),
            np.array([[0.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_direction_prior_round_trip_to_rotation_log_prior(self):
        healpix_order = 1
        n_dirs = 48
        direction_prior = np.zeros(n_dirs, dtype=np.float32)
        direction_prior[:3] = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        direction_prior[3:] = np.finfo(np.float32).tiny
        direction_prior /= direction_prior.sum()

        rotation_log_prior = make_relion_direction_log_prior(direction_prior, healpix_order)
        collapsed = collapse_rotation_posterior_to_direction_prior(
            np.exp(rotation_log_prior),
            healpix_order,
        )

        np.testing.assert_allclose(collapsed, direction_prior, rtol=1e-6, atol=1e-8)

    def test_kclass_direction_prior_combines_half_posteriors_before_collapsing(self):
        """Class3D has one pdf_direction update; RECOVAR halves are parallelism only."""
        healpix_order = 0
        n_rot = rotation_grid_size(healpix_order)
        n_dirs = n_rot // rotation_grid_n_in_planes(healpix_order)
        half0 = np.zeros((2, n_rot), dtype=np.float64)
        half1 = np.zeros((2, n_rot), dtype=np.float64)
        half0[0, 0] = 9.0
        half1[0, 1] = 3.0
        half0[1, 2] = 2.0
        half1[1, 3] = 6.0

        combined = _combined_class_direction_prior_from_halves(
            [half0, half1],
            n_classes=2,
            healpix_order=healpix_order,
        )

        expected = []
        for class_idx in range(2):
            expected.append(
                collapse_rotation_posterior_to_direction_prior(
                    half0[class_idx] + half1[class_idx],
                    healpix_order,
                )
            )
        expected = np.stack(expected, axis=0)
        assert combined.shape == (2, n_dirs)
        np.testing.assert_allclose(combined, expected, rtol=1e-6, atol=1e-8)

    def test_significance_batched_supports_padded_rotation_log_prior(
        self,
        monkeypatch,
        translations,
    ):
        """Rotation priors should work even when the last block is padded."""
        rotations = _make_rotations(5, seed=19)
        dataset, means, noise, projector = _exact_pass1_inputs(monkeypatch)
        sig_rot_any, n_sig, ha, _, _, _ = _compute_k_class_significance_batched(
            dataset,
            noise,
            rotations,
            translations,
            **projector,
            class_log_priors=np.zeros(1, dtype=np.float64),
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=N_IMAGES,
            rotation_block_size=rotations.shape[0] + 1,
            current_size=None,
            rotation_log_prior=np.zeros(rotations.shape[0], dtype=np.float32),
        )

        assert sig_rot_any.shape == (1, rotations.shape[0])
        assert n_sig.shape == (dataset.n_units,)
        assert ha.shape == (dataset.n_units,)

    @pytest.mark.parametrize("current_size", [4, 6])
    @pytest.mark.parametrize("score_mode", ["gaussian", "normalized_cc"])
    @pytest.mark.parametrize("stable_fourier_window_shapes", [False, True])
    def test_k_class_significance_texture_ppref_requests_compact_score_rows(
        self,
        half_datasets,
        monkeypatch,
        current_size,
        score_mode,
        stable_fourier_window_shapes,
    ):
        """Windowed texture scoring must not materialize full projection rows."""
        from relax.helpers import projection as projection_helpers
        from relax.helpers.fourier_window import make_fourier_window_spec
        from relax.scoring import coarse_layout

        dataset, means, noise, projector = _exact_pass1_inputs(monkeypatch)
        window = make_fourier_window_spec(
            IMAGE_SHAPE,
            current_size,
            IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1),
            score_square=score_mode == "normalized_cc",
            score_include_dc=score_mode == "normalized_cc",
        )
        if score_mode == "gaussian":
            # The coarse GEMMs read the square layout planned over the window's rows.
            layout = coarse_layout.plan_coarse_gaussian_square_layout(
                IMAGE_SHAPE,
                current_size,
                np.asarray(window.score_indices_np, dtype=np.int32),
                stable_fourier_window_shapes=stable_fourier_window_shapes,
            )
            expected = np.asarray(layout.score_indices_np, dtype=np.int32)
            expected_output_size = layout.physical_current_size
        else:
            expected = np.asarray(window.score_indices, dtype=np.int32)
            expected_output_size = current_size
        calls = []

        def fake_texture(projector_half, rotations, image_shape, **kwargs):
            pixel_indices = kwargs.get("pixel_indices")
            assert isinstance(pixel_indices, np.ndarray)
            assert kwargs["projector_output_size"] == expected_output_size
            expected_mask_size = current_size if stable_fourier_window_shapes else None
            assert kwargs["current_image_mask_size"] == expected_mask_size
            calls.append(np.asarray(pixel_indices, dtype=np.int32))
            n_pixels = int(pixel_indices.shape[0])
            projection = jnp.ones(
                (rotations.shape[0], n_pixels),
                dtype=jnp.complex64,
            )
            return projection, jnp.ones(projection.shape, dtype=jnp.float32)

        monkeypatch.setattr(
            projection_helpers,
            "compute_relion_projector_projections_block",
            fake_texture,
        )

        rotations = _make_rotations(3, seed=201)
        _compute_k_class_significance_batched(
            dataset,
            noise,
            rotations,
            jnp.zeros((1, 2), dtype=jnp.float32),
            class_log_priors=np.zeros(1, dtype=np.float64),
            adaptive_fraction=1.0,
            max_significants=1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=current_size,
            score_mode=score_mode,
            stable_fourier_window_shapes=stable_fourier_window_shapes,
            collect_significance=False,
            return_class_best=True,
            **projector,
        )

        assert calls
        for requested in calls:
            assert_matches(requested, expected)

    def test_k_class_significance_tail_padding_preserves_outputs(
        self,
        monkeypatch,
    ):
        if jax.default_backend() == "gpu":
            # On a GPU the padded extent changes the float32 reductions; the null band is
            # test_em_stage_glue_programs.test_coarse_pad_env_flag_stays_inside_the_null_band_on_gpu.
            pytest.skip("CPU-only contract")
        dataset, means, noise, projector = _exact_pass1_inputs(monkeypatch, n_classes=2)
        rotations = _make_rotations(3, seed=312)
        translations = jnp.array([[0.0, 0.0], [1.0, -1.0]], dtype=jnp.float32)
        common_kwargs = dict(
            class_log_priors=np.log(np.array([0.55, 0.45], dtype=np.float64)),
            adaptive_fraction=0.999,
            max_significants=-1,
            rotation_block_size=2,
            current_size=6,
            score_with_masked_images=True,
            translation_log_prior=np.asarray([[0.0, -0.1], [-0.2, 0.0]], dtype=np.float32),
            # RELION's CUDA preprocessing shifts by whole pixels.
            image_pre_shifts=np.asarray([[1.0, -1.0], [-2.0, 1.0]], dtype=np.float32),
            return_class_best=True,
            return_class_second=True,
            **projector,
        )

        unpadded = _compute_k_class_significance_batched(
            dataset,
            noise,
            rotations,
            translations,
            image_batch_size=dataset.n_units,
            pad_final_image_batch=False,
            **common_kwargs,
        )
        padded = _compute_k_class_significance_batched(
            dataset,
            noise,
            rotations,
            translations,
            image_batch_size=dataset.n_units + 1,
            pad_final_image_batch=True,
            **common_kwargs,
        )

        from relax.scoring.significant_samples import significant_sample_ids

        for expected, actual in zip(unpadded[:4], padded[:4]):
            assert_matches(np.asarray(actual), np.asarray(expected))
        n_samples = int(rotations.shape[0]) * int(translations.shape[0])
        for expected_by_class, actual_by_class in zip(unpadded[4], padded[4]):
            for expected, actual in zip(expected_by_class, actual_by_class):
                assert_matches(
                    significant_sample_ids(actual, n_samples), significant_sample_ids(expected, n_samples)
                )
        _assert_significance_stats_allclose(padded[5], unpadded[5])

    @pytest.mark.parametrize(
        "original_scores, rescored_scores, expected_pose, expected_ties, expected_changes",
        [
            ((1.0 - 2e-7, 1.0), (0.75, 0.75), 0, "all", "all"),
            ((1.0, 1.0 - 2e-7), (0.5, 0.75), 1, 0, "all"),
        ],
    )
    def test_k1_firstiter_cc_tree_top2_rescore_replaces_bounded_winner(
        self,
        half_datasets,
        init_volume,
        monkeypatch,
        original_scores,
        rescored_scores,
        expected_pose,
        expected_ties,
        expected_changes,
        request,
    ):
        """The direct-texture replay replaces every bounded native winner."""

        import recovar.cuda_backproject as cuda_backproject

        import relax.helpers.projection as projection_module
        import relax.scoring.pass1_program as pass1_program
        import relax.scoring.scoring as scoring_module
        from relax.cuda import kernels as em_cuda_kernels

        dataset = half_datasets[0]
        dataset.image_source.backend.relion_fourier_backend = "relion_cuda"

        def fake_relion_process_half(batch, apply_image_mask=False, **_kwargs):
            return _raw_real_process_half(batch, apply_image_mask=apply_image_mask)

        dataset.process_images_half = fake_relion_process_half
        rotations = _make_rotations(2, seed=6322)
        translations = jnp.zeros((1, 2), dtype=jnp.float32)
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
        monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
        monkeypatch.setattr(em_cuda_kernels, "custom_cuda_requested", lambda: True)
        monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: True)
        monkeypatch.setattr(
            relion_ctf,
            "relion_exact_ctf_half_from_source_star",
            lambda _dataset, indices, image_shape: jnp.ones(
                (
                    len(indices),
                    int(image_shape[0]) * (int(image_shape[1]) // 2 + 1),
                ),
                dtype=jnp.float64,
            ),
        )

        def fake_projector(_ppref, rots, image_shape, **_kwargs):
            n_rot = int(rots.shape[0])
            pixel_indices = _kwargs.get("pixel_indices")
            n_half = (
                int(image_shape[0] * (image_shape[1] // 2 + 1))
                if pixel_indices is None
                else int(np.asarray(pixel_indices).size)
            )
            projection = jnp.ones((n_rot, n_half), dtype=jnp.complex64)
            return projection, jnp.ones((n_rot, n_half), dtype=jnp.float32)

        def fake_tree_rescore(
            shifted_candidates,
            _score_weight_candidates,
            _projection_candidates,
            _half_weights,
            _fftw_order,
            **kwargs,
        ):
            assert kwargs["projector_full"] is not None
            assert kwargs["rotation_matrices"].shape[-2:] == (3, 3)
            scores = jnp.asarray(rescored_scores, dtype=jnp.float32)
            return jnp.broadcast_to(scores, shifted_candidates.shape[:2])

        monkeypatch.setattr(
            projection_module,
            "compute_relion_projector_projections_block",
            fake_projector,
        )
        def fake_cc_gemm_scores(_proj, _shifted, _weight, _count, *, n_images, n_trans):
            return jnp.broadcast_to(
                jnp.asarray(original_scores, dtype=jnp.float32)[None, :, None],
                (n_images, 2, n_trans),
            )

        # The pass-1 program (pass1_program) and the scorer module each bind the scorer.
        clear_pass1_programs(request)
        for module in (scoring_module, pass1_program):
            monkeypatch.setattr(module, "relion_coarse_normalized_cc_gemm_scores_jit", fake_cc_gemm_scores)
        monkeypatch.setattr(
            em_cuda_kernels,
            "relion_translate_score_f32",
            lambda images, _angles, _indices, _shape: images,
        )
        monkeypatch.setattr(
            scoring_module,
            "relion_coarse_normalized_cc_rescore",
            fake_tree_rescore,
        )

        *_, full_stats = _compute_k_class_significance_batched(
            dataset,
            jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
            rotations,
            translations,
            class_log_priors=np.zeros(1, dtype=np.float64),
            adaptive_fraction=1.0,
            max_significants=1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=6,
            half_spectrum_scoring=True,
            relion_projector_half=jnp.zeros((1, 3, 3, 2), dtype=jnp.complex64),
            relion_projector_r_max=1,
            relion_projector_texture_interp=True,
            score_mode="normalized_cc",
            collect_significance=False,
            return_class_best=True,
            tree_rescore_max_margin=4e-6,
        )

        assert full_stats["executed_coarse_backend"] == "exact_cc_gemm"
        assert_matches(
            np.asarray(full_stats["class_hard_assignments"]),
            np.full((1, dataset.n_units), expected_pose, dtype=np.int32),
        )
        assert "class_second_hard_assignments" not in full_stats
        if expected_ties == "all":
            expected_ties = dataset.n_units
        if expected_changes == "all":
            expected_changes = dataset.n_units
        assert full_stats["firstiter_cc_tree_top2_rescore"] == {
            "max_margin": 4e-6,
            "examined_images": dataset.n_units,
            "ambiguous_images": dataset.n_units,
            "exact_score_ties": expected_ties,
            "winner_changes": expected_changes,
        }

    def test_k_class_significance_dump_emits_target_files(
        self,
        monkeypatch,
        tmp_path,
    ):
        """K-class significance pass writes per-image .npz dumps when env vars target an image.

        Regression for codex_k2_dump_20260508_064420_5026: prior to wiring
        ``maybe_dump_k_class_significance_batch`` into the K-class branch, the
        InitialModel K=2 sparse pass-2 emitted no significance debug files even
        with ``RELAX_SIGNIFICANCE_DUMP_DIR`` and the matching original-index
        target set.
        """
        dataset, means, noise, projector = _exact_pass1_inputs(monkeypatch, n_classes=2)
        rotations = _make_rotations(5, seed=31)
        translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
        class_log_priors = np.log(np.array([0.55, 0.45], dtype=np.float64))

        dump_dir = tmp_path / "k_class_sig_dump"
        target_local = 0
        monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
        monkeypatch.setenv(
            "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES",
            str(target_local),
        )

        _compute_k_class_significance_batched(
            dataset,
            noise,
            rotations,
            translations,
            class_log_priors=class_log_priors,
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=6,
            score_with_masked_images=False,
            **projector,
        )

        files = sorted(dump_dir.glob("*.npz"))
        assert files, f"K-class significance dump produced no files in {dump_dir}"
        payload = np.load(files[0])
        assert int(payload["n_classes"]) == 2
        assert int(payload["n_rot"]) == int(rotations.shape[0])
        assert int(payload["n_trans"]) == int(translations.shape[0])
        assert "projected_reference_rotation_ids" not in payload.files
        assert "projected_reference_per_class" not in payload.files
        weights_per_class = payload["weights_per_class"]
        assert weights_per_class.shape == (2, int(rotations.shape[0]) * int(translations.shape[0]))
        assert weights_per_class.sum() == pytest.approx(1.0, abs=1e-6)
        class_log_z = payload["class_log_z"]
        assert class_log_z.shape == (2,)
        assert int(payload["class_assignment"]) in (0, 1)

    def test_k_class_score_probe_ignores_dump_env_when_significance_is_not_collected(
        self,
        monkeypatch,
        tmp_path,
    ):
        """Score-only K-class probes must not abort when a dump env is set."""
        dataset, means, noise, projector = _exact_pass1_inputs(monkeypatch, n_classes=2)
        rotations = _make_rotations(5, seed=37)
        translations = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
        class_log_priors = np.log(np.array([0.55, 0.45], dtype=np.float64))

        dump_dir = tmp_path / "k_class_score_probe_dump"
        monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
        monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "0")

        *_, full_stats = _compute_k_class_significance_batched(
            dataset,
            noise,
            rotations,
            translations,
            class_log_priors=class_log_priors,
            adaptive_fraction=0.999,
            max_significants=-1,
            image_batch_size=dataset.n_units,
            rotation_block_size=2,
            current_size=6,
            score_with_masked_images=False,
            collect_significance=False,
            return_class_best=True,
            **projector,
        )

        assert not list(dump_dir.glob("*.npz"))
        assert full_stats["class_hard_assignments"].shape == (2, dataset.n_units)

    def test_relion_mode_convergence_state(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
    ):
        """Convergence state is a RefinementState with correct fields."""
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=2,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                local_search=LocalSearchOptions(auto_local_healpix_order=3),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        state = result.convergence_state
        assert isinstance(state, RefinementState)
        assert state.auto_local_healpix_order == 3
        # After 2 iterations, iteration counter should be at least 1
        assert state.iteration >= 1
        # ave_Pmax should be in [0, 1]
        assert 0.0 <= state.ave_Pmax <= 1.0

    @pytest.mark.parametrize("double_scoring", [False, True])
    @pytest.mark.parametrize("per_half", [False, True])
    def test_relion_mode_uses_tau2_from_weights_for_prior(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
        tmp_path,
        per_half,
        double_scoring,
    ):
        """RELION mode should compute tau2 from Ft_ctf weights + FSC (RELION order)."""
        from relax.reconstruction import regularization_relion

        monkeypatch.setattr(
            scoring_policy,
            "DENSE_PRECISION",
            replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=double_scoring),
        )
        monkeypatch.setenv("RELAX_USE_FLOAT64_SCORING", "1" if double_scoring else "0")
        called = {"tau2": 0}

        original_tau2 = regularization_relion.compute_relion_tau2_from_weights

        def wrap_tau2(*args, **kwargs):
            called["tau2"] += 1
            return original_tau2(*args, **kwargs)

        monkeypatch.setattr(regularization_relion, "compute_relion_tau2_from_weights", wrap_tau2)

        scoring_priors = []
        scoring_rotations = []
        scoring_grids = []
        monkeypatch.setattr(
            iteration_planning_module, "relion_adaptive_pass1_rotations",
            lambda eulers, *args, **kwargs: euler_angles_to_matrix(eulers).astype(np.float32),
        )
        score_half = finalization._score_half_dense_in_bpref_scope

        def record_scoring_prior(half, sampling, priors, batching, variant, execution, optics):
            scoring_priors.append(np.asarray(half.mean_variance))
            scoring_rotations.append(sampling.coarse_scoring_rotations)
            scoring_grids.append(
                (sampling.effective_rotations, sampling.current_translations)
            )
            return score_half(half, sampling, priors, batching, variant, execution, optics)

        monkeypatch.setattr(finalization, "_score_half_dense_in_bpref_scope", record_scoring_prior)
        monkeypatch.setattr(expectation_module, "_score_half_dense_in_bpref_scope", record_scoring_prior)
        initial_tau2 = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0
        if per_half:
            initial_tau2 = jnp.stack([initial_tau2, 2 * initial_tau2])

        grid_size = int(np.sqrt(IMAGE_SIZE))
        refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), initial_tau2),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                start=StartState(init_fsc=np.ones(grid_size // 2)),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                adaptive=stand_in.adaptive(adaptive_oversampling=0),
                parity=stand_in.parity(use_per_half_mean_variance=per_half),
            ),
            observer=IntermediatesObserver(tmp_path),
            source=InputSource(),
        )

        assert called["tau2"] >= 1
        assert len(scoring_priors) == 2
        for half, prior in enumerate(scoring_priors):
            with np.load(tmp_path / f"manifest_iter0_half{half}.npz") as manifest:
                assert_matches(manifest["mean_variance"], prior)
                assert bool(manifest["use_float64_scoring"]) == double_scoring
                coarse = scoring_rotations[half]
                assert (coarse is None) == double_scoring
                expected = np.array([]) if coarse is None else np.asarray(coarse)
                assert_matches(manifest["coarse_scoring_rotations"], expected)
                assert manifest["coarse_scoring_rotations"].dtype == expected.dtype
                for key, grid in zip(("effective_rotations", "current_translations"), scoring_grids[half]):
                    assert_matches(manifest[key], np.asarray(grid))
                    assert manifest[key].dtype == grid.dtype

    def test_k1_raw_backprojector_fsc_feeds_tau2(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """The raw backprojector FSC drives tau2, as in GUI auto-refine without solvent-corrected FSC."""
        from relax.reconstruction import regularization_relion

        grid_size = int(np.sqrt(IMAGE_SIZE))
        n_shells = grid_size // 2 + 1
        raw_fsc = np.linspace(0.95, 0.55, n_shells, dtype=np.float32)
        raw_fsc[0] = 1.0
        tau2_fsc_inputs = []

        monkeypatch.setattr(
            regularization_relion,
            "compute_relion_fsc_from_backprojector",
            lambda *_args, **_kwargs: jnp.asarray(raw_fsc),
        )

        original_tau2 = regularization_relion.compute_relion_tau2_from_weights

        def wrap_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs):
            tau2_fsc_inputs.append(np.asarray(fsc, dtype=np.float32).copy())
            return original_tau2(Ft_ctf_0, Ft_ctf_1, fsc, *args, **kwargs)

        monkeypatch.setattr(regularization_relion, "compute_relion_tau2_from_weights", wrap_tau2)

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                    particle_diameter_ang=200.0,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                adaptive=stand_in.adaptive(adaptive_oversampling=0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert len(tau2_fsc_inputs) == 2
        for tau2_fsc in tau2_fsc_inputs:
            np.testing.assert_allclose(tau2_fsc, raw_fsc, atol=1e-7)
        np.testing.assert_allclose(np.asarray(result.history.fsc_history[0]), raw_fsc, atol=1e-7)

    def test_firstiter_cc_lowpass_runs_before_solvent_flatten(self, monkeypatch):
        """RELION applies iter-1 ini_high low-pass before solvent flatten."""

        from relax.refinement import numbered_reconstruction as numbered_reconstruction_module

        events = []
        reconstruct_calls = []
        solvent_mask_radii = []

        def fake_reconstruct(*_args, **kwargs):
            reconstruct_calls.append(kwargs)
            return jnp.ones(VOLUME_SIZE, dtype=jnp.complex64)

        def fake_lowpass(volume_ft_flat, *_args, **_kwargs):
            events.append("lowpass")
            return volume_ft_flat

        def fake_idft3(volume_ft):
            events.append("flatten_idft")
            return jnp.ones(VOLUME_SHAPE, dtype=jnp.float32)

        def fake_dft3(volume_real):
            events.append("flatten_dft")
            return jnp.asarray(volume_real, dtype=jnp.complex64)

        def fake_raised_cosine_mask(volume_shape, *, radius, radius_p, offset, dtype=None):
            del offset, dtype
            solvent_mask_radii.append((radius, radius_p))
            return jnp.ones(volume_shape, dtype=jnp.float64)

        monkeypatch.setattr(numbered_reconstruction_module, "_reconstruct_volume_eager", fake_reconstruct)
        monkeypatch.setattr(map_postprocess, "apply_relion_initial_lowpass_filter", fake_lowpass)
        monkeypatch.setattr(map_postprocess.fourier_transform_utils, "get_idft3", fake_idft3)
        monkeypatch.setattr(map_postprocess.fourier_transform_utils, "get_dft3", fake_dft3)

        monkeypatch.setattr(
            map_postprocess.mask, "raised_cosine_mask", fake_raised_cosine_mask,
        )

        settings = reconstruction_settings(
            box_size=8,
            voxel_size=np.float32(2.125),
            volume_shape=VOLUME_SHAPE,
            padding_factor=1,
            projection_padding_factor=1,
            minres_map=1,
            width_mask_edge=5,
            fmask_edge=2,
            tau2_fudge=1.0,
            particle_diameter_angstrom=200.0,
            first_iteration_lowpass_angstrom=30.0, programs=ReconstructionPrograms.from_environ(),
        )
        means = numbered_reconstruction_module.reconstruct_numbered_k1_halfmaps(
            (jnp.ones(VOLUME_SIZE, dtype=jnp.complex64), jnp.ones(VOLUME_SIZE, dtype=jnp.complex64)),
            (jnp.ones(VOLUME_SIZE, dtype=jnp.float32), jnp.ones(VOLUME_SIZE, dtype=jnp.float32)),
            [
                jnp.ones(VOLUME_SHAPE[0] // 2 + 1), jnp.ones(VOLUME_SHAPE[0] // 2 + 1),
            ],
            settings,
            iteration=0,
            current_size=8,
            accumulator_volume_shape=None,
            relion_firstiter_cc_this_iter=True,
        )

        assert events[:3] == ["lowpass", "flatten_idft", "flatten_dft"]
        assert len(reconstruct_calls) == 2
        assert all(call["preserve_output_precision"] is True for call in reconstruct_calls)
        expected_radius = 200.0 / (2.0 * 2.125)
        assert solvent_mask_radii == [(expected_radius, expected_radius + 5.0)] * 2
        assert solvent_mask_radii[0][0] != float(
            np.float32(200.0) / (np.float32(2.0) * np.float32(2.125))
        )


    def test_kclass_reconstruction_uses_1d_tau_shell_prior(self, monkeypatch):
        """K-class M-step reconstruction should index RELION tau2 as shells."""

        from relax.refinement import numbered_reconstruction as numbered_reconstruction_module

        calls = []

        def fake_reconstruct(*_args, **kwargs):
            calls.append(kwargs)
            return jnp.ones(VOLUME_SIZE, dtype=jnp.complex64)

        monkeypatch.setattr(numbered_reconstruction_module, "_reconstruct_volume_eager", fake_reconstruct)

        n_classes = 2
        n_shells = VOLUME_SHAPE[0] // 2 + 1
        tau_shells = jnp.stack(
            [
                jnp.arange(n_shells, dtype=jnp.float32) + 101.0,
                jnp.arange(n_shells, dtype=jnp.float32) + 201.0,
            ],
            axis=0,
        )
        settings = reconstruction_settings(
            box_size=8,
            voxel_size=1.0,
            volume_shape=VOLUME_SHAPE,
            padding_factor=1,
            projection_padding_factor=1,
            minres_map=0,
            width_mask_edge=5,
            fmask_edge=2,
            tau2_fudge=4.0,
            particle_diameter_angstrom=None,
            first_iteration_lowpass_angstrom=None, programs=ReconstructionPrograms.from_environ(),
        )
        means = numbered_reconstruction_module.reconstruct_numbered_class_maps(
            jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.complex64),
            jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32),
            tau_shells,
            settings,
            n_classes=n_classes,
            iteration=0,
            current_size=8,
            accumulator_volume_shape=None,
            relion_firstiter_cc_this_iter=False,
        )

        assert len(calls) == n_classes
        assert all(call["tau_is_1d"] is True for call in calls)
        assert all(call.get("preserve_output_precision", False) is False for call in calls)
        np.testing.assert_allclose(np.asarray(calls[0]["tau"]), np.asarray(tau_shells[0]))
        np.testing.assert_allclose(np.asarray(calls[1]["tau"]), np.asarray(tau_shells[1]))
        assert means[0].shape == (n_classes, VOLUME_SIZE)
        assert_matches(np.asarray(means[0]), np.asarray(means[1]))

    def test_k1_save_intermediates_reconstructs_unregularized_half_maps(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        tmp_path,
    ):
        """K=1 diagnostic unregularized maps use full half accumulators, not class indexing."""

        out_dir = tmp_path / "intermediates"
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                adaptive=stand_in.adaptive(adaptive_oversampling=0),
            ),
            observer=IntermediatesObserver(out_dir),
            source=InputSource(),
        )

        assert len(result.history.current_sizes) == 1
        assert (out_dir / "it000_half1_unreg.mrc").exists()
        assert (out_dir / "it000_half2_unreg.mrc").exists()

    @pytest.mark.parametrize(
        ("optics_pixel", "expected_scale"),
        [(None, 1.0), (1.0, 1.0), (1.0 + 2.0**-20, 1.0 / (1.0 + 2.0**-20))],
    )
    def test_loop_forwards_the_model_over_optics_translation_angle_scale(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
        optics_pixel,
        expected_scale,
    ):
        """Final-Q aa0eccbfd4: the K=1 loop hands the scale to half-set scoring."""

        class _Captured(Exception):
            pass

        captured = {}

        def capture(half, sampling, priors, batching, variant, execution, optics):
            del half, sampling, priors, batching, variant, optics
            captured["execution"] = execution
            raise _Captured

        monkeypatch.setattr(finalization, "_score_half_dense_in_bpref_scope", capture)
        monkeypatch.setattr(expectation_module, "_score_half_dense_in_bpref_scope", capture)
        model_pixel = float(half_datasets[0].voxel_size)
        geometry = (
            OpticsGeometry()
            if optics_pixel is None
            else OpticsGeometry(
                relion_optics_image_sizes=[IMAGE_SHAPE[0]],
                relion_optics_pixel_sizes=[optics_pixel * model_pixel],
                relion_model_pixel_size=model_pixel,
            )
        )
        with pytest.raises(_Captured):
            refine_single_volume(
                half_datasets,
                StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
                HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
                translations,
                options=stand_in.options(
                    schedule=stand_in.schedule(
                        max_iter=1,
                        init_current_size=16,
                        init_healpix_order=2,
                        max_healpix_order=3,
                    ),
                    execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                    adaptive=stand_in.adaptive(adaptive_oversampling=0),
                    optics_geometry=geometry,
                ),
                observer=RunObserver(), source=InputSource(),
            )
        assert captured["execution"].relion_translation_angle_scale == expected_scale


    @pytest.mark.parametrize("n_classes", [1, 2])
    def test_save_intermediates_writes_source_aligned_particle_states(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        tmp_path,
        n_classes,
    ):
        """Each numbered iteration saves resolved per-particle poses beside the maps (final-Q 68ac9d05ab)."""

        out_dir = tmp_path / "intermediates"
        k_class = (
            KClassOptions()
            if n_classes == 1
            else KClassOptions(
                n_classes=n_classes,
            )
        )
        refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                adaptive=stand_in.adaptive(adaptive_oversampling=0),
                k_class=k_class,
            ),
            observer=IntermediatesObserver(out_dir, skip_unregularized=True),
            source=InputSource(),
        )

        for half, dataset in enumerate(half_datasets, start=1):
            with np.load(out_dir / f"it000_particle_state_half{half}.npz") as state:
                n = int(dataset.n_units)
                assert_matches(state["half_local_indices"], np.arange(n))
                assert state["rotation_matrices"].shape == (n, 3, 3)
                assert state["rotation_eulers_deg"].shape == (n, 3)
                assert state["relative_translations_pixels"].shape == (n, 2)
                assert state["absolute_translations_pixels"].shape == (n, 2)
                assert state["max_posterior"].shape == (n,)
                assert state["fine_hard_assignment"].shape == (n,)
                assert state["original_image_indices"].shape == (n,)
                assert_matches(state["one_based_iteration"], [1])
                assert_matches(state["half"], [half])

    def test_k1_save_intermediates_can_skip_unregularized_half_maps(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        tmp_path,
    ):
        """Fast forensic dumps can keep regularized maps without unreg reconstruction."""

        out_dir = tmp_path / "intermediates"
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                adaptive=stand_in.adaptive(adaptive_oversampling=0),
            ),
            observer=IntermediatesObserver(out_dir, skip_unregularized=True),
            source=InputSource(),
        )

        assert len(result.history.current_sizes) == 1
        assert (out_dir / "it000_half1_reg.mrc").exists()
        assert (out_dir / "it000_half2_reg.mrc").exists()
        assert not (out_dir / "it000_half1_unreg.mrc").exists()
        assert not (out_dir / "it000_half2_unreg.mrc").exists()

    def test_relion_mode_current_size_no_longer_uses_weight_based_data_vs_prior(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """RELION mode should derive current_size from FSC-derived SSNR logic."""
        def fail_old_dvp(*args, **kwargs):
            raise AssertionError("RELION mode should not call compute_data_vs_prior")

        monkeypatch.setattr(regularization_relion, "compute_data_vs_prior", fail_old_dvp)

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=2,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
                adaptive=stand_in.adaptive(adaptive_oversampling=0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert len(result.history.current_sizes) == 2

    def test_relion_mode_trajectories_populated(
        self,
        fake_global_estep,
        half_datasets,
        init_volume,
        translations,
    ):
        """RELION-specific trajectories have correct lengths."""
        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=2,
                    init_current_size=16,
                    init_healpix_order=2,
                    max_healpix_order=3,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        n_iters = len(result.history.current_sizes)
        assert n_iters <= 2
        assert len(result.history.healpix_order_trajectory) == n_iters
        assert len(result.history.ave_Pmax_trajectory) == n_iters
        # data_vs_prior is populated starting from iteration 1
        assert len(result.history.data_vs_prior_trajectory) <= n_iters

    def test_relion_mode_updates_sigma_offset_from_posterior_noise_stats(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """Posterior-weighted offset variance should drive sigma_offset in RELION mode."""

        for ds in half_datasets:
            ds.voxel_size = 8.5
        rotations_many = _make_rotations(20, seed=888)
        noise_offset_wsums = [12.0, 20.0]
        call_idx = {"value": 0}

        def fake_adaptive(experiment_dataset, means, mean_variance, noise_variance, coarse_rotations,
                          coarse_translations, fine_rotations, *grids, **kwargs):
            idx = call_idx["value"]
            call_idx["value"] += 1
            offset_wsum = noise_offset_wsums[min(idx, len(noise_offset_wsums) - 1)]
            return adaptive_result(experiment_dataset, means, fine_rotations, kwargs, sigma2_offset=offset_wsum)

        monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive)

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=16,
                    init_healpix_order=1,
                    max_healpix_order=2,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=len(rotations_many)),
                adaptive=stand_in.adaptive(adaptive_oversampling=0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        expected_per_half = [
            np.sqrt(noise_offset_wsums[0] / (2.0 * half_datasets[0].n_units)),
            np.sqrt(noise_offset_wsums[1] / (2.0 * half_datasets[1].n_units)),
        ]
        expected_sigma = float(np.mean(expected_per_half))
        assert result.history.sigma_offset_trajectory[0] == pytest.approx(expected_sigma)
        assert result.history.sigma_offset_per_half_trajectory[0] == pytest.approx(expected_per_half)

    def test_relion_mode_passes_per_half_noise_to_engine(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """RELION mode must score each half-set with its own sigma2_noise."""

        rotations_many = _make_rotations(20, seed=777)
        half1_noise = np.arange(IMAGE_SIZE, dtype=np.float32) + 1.0
        half2_noise = half1_noise * 3.0
        captured_noise = []

        def fake_adaptive(experiment_dataset, means, mean_variance, noise_variance, coarse_rotations,
                          coarse_translations, fine_rotations, *grids, **kwargs):
            captured_noise.append(np.asarray(noise_variance, dtype=np.float32))
            return adaptive_result(experiment_dataset, means, fine_rotations, kwargs)

        monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive)

        refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair(half1_noise, half2_noise),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_current_size=8,
                    init_healpix_order=1,
                    max_healpix_order=2,
                    skip_final_iteration=True,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=len(rotations_many)),
                adaptive=stand_in.adaptive(adaptive_oversampling=0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert len(captured_noise) == 2
        np.testing.assert_allclose(captured_noise[0], half1_noise)
        np.testing.assert_allclose(captured_noise[1], half2_noise)

    @pytest.mark.parametrize("oversampling,preserve_order", [(1, False), (0, True)])
    def test_k1_adaptive_significant_counts_do_not_replace_exact_accuracy(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
        oversampling,
        preserve_order,
    ):
        """K=1 adaptive counts remain diagnostic when exact accuracy is unavailable."""
        monkeypatch.delenv("RELAX_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE", raising=False)
        counts_by_half = [
            np.array([4, 5], dtype=np.int32),
            np.array([6, 7], dtype=np.int32),
        ]
        call_idx = {"value": 0}
        fine_mstep_prune_values = []
        rotations_many = _make_rotations(20, seed=333)

        monkeypatch.setattr(
            sampling_module,
            "relion_scoring_rotation_grid",
            lambda _order, dtype=None, *, symmetry='C1': sampling_module.RotationGrid(rotations=rotations_many, rotation_eulers=np.zeros((len(rotations_many), 3), dtype=np.float32), healpix_order=_order, symmetry=symmetry),
        )
        monkeypatch.setattr(
            projector_preparation,
            "prepare_scoring_projector",
            lambda *_args, **_kwargs: projector_preparation.PreparedProjector(
                data=np.zeros((1, 3, 3, 2), dtype=np.complex64), r_max=1,
            ),
        )

        def fake_build_pass2_grids(effective_rotations, current_translations, base_translations, current_healpix_order, adaptive_oversampling, translation_step, random_perturbation, return_mstep_rotations=False, *, coarse_rotation_ids=None, symmetry='C1'):
            _ = (base_translations, current_healpix_order, adaptive_oversampling, translation_step, random_perturbation)
            coarse_rot = np.asarray(effective_rotations, dtype=np.float32)
            coarse_trans = np.asarray(current_translations, dtype=np.float32)
            rot_parent = np.arange(coarse_rot.shape[0], dtype=np.int32)
            trans_parent = np.arange(coarse_trans.shape[0], dtype=np.int32)
            base = (coarse_rot, coarse_trans, coarse_rot, coarse_trans, rot_parent, trans_parent)
            return (*base, coarse_rot) if return_mstep_rotations else base

        def fake_adaptive_k1(
            experiment_dataset,
            means,
            mean_variance,
            noise_variance,
            coarse_rotations,
            coarse_translations,
            fine_rotations,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
            disc_type,
            **kwargs,
        ):
            _ = (
                means,
                mean_variance,
                noise_variance,
                coarse_translations,
                rot_parent_map,
                trans_parent_map,
                disc_type,
            )
            assert kwargs.get("preserve_bpref_particle_order", False) == preserve_order
            # Pass 1 always scores the exact operands; the removed switch is not passed.
            assert "relion_exact_coarse" not in kwargs
            half_idx = call_idx["value"]
            call_idx["value"] += 1
            fine_mstep_prune_values.append(kwargs.get("relion_fine_mstep_prune"))
            n_images = int(experiment_dataset.n_units)
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
            counts = counts_by_half[half_idx]
            assert counts.shape == (n_images,)
            return KClassEMResult(
                new_means=None,
                Ft_y=jnp.zeros((1, recon_vol_size), dtype=jnp.complex64),
                Ft_ctf=jnp.ones((1, recon_vol_size), dtype=jnp.complex64),
                per_class_hard_assignments=jnp.zeros((1, n_images), dtype=jnp.int32),
                class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                class_responsibilities=jnp.ones((1, n_images), dtype=jnp.float32),
                class_posterior_sums=jnp.array([float(n_images)], dtype=jnp.float32),
                stats=RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(coarse_rotations).shape[0], dtype=jnp.float32),
                ),
                per_class_stats=(
                    RelionStats(
                        log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                        best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                        max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                        rotation_posterior_sums=jnp.ones(np.asarray(coarse_rotations).shape[0], dtype=jnp.float32),
                    ),
                ),
                noise_stats=(
                    NoiseStats(
                        wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                        wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                        wsum_sigma2_offset=0.0,
                        sumw=float(n_images),
                    ),
                ),
                aggregate_noise_stats=NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=float(n_images),
                ),
                best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
                best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
                best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
                significant_counts=jnp.asarray(counts, dtype=jnp.int32),
            )

        monkeypatch.setattr(oversampling_grids, "build_adaptive_pass2_grids", fake_build_pass2_grids)
        monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive_k1)

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_healpix_order=1,
                    max_healpix_order=1,
                    skip_final_iteration=True,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=20),
                adaptive=stand_in.adaptive(relion_current_sizes=[8], adaptive_oversampling=oversampling),
                parity=stand_in.parity(preserve_bpref_particle_order=preserve_order),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert call_idx["value"] == 2
        assert fine_mstep_prune_values == [True, True]
        assert_matches(
            np.asarray(result.history.significant_counts[0], dtype=np.int32),
            np.concatenate(counts_by_half),
        )
        assert np.isnan(result.history.acc_rot_trajectory[0])
        assert result.history.expected_accuracy_status_trajectory == ["unavailable_inputs"]
        assert np.isinf(result.convergence_state.acc_rot)

    def test_k1_zero_oversampling_enters_adaptive_engine(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """K=1 at oversampling 0 without scale groups runs the adaptive engine's single pass.

        The direct dense engine this configuration used was removed on 2026-10-03; the
        adaptive engine's one coarse pass on the current grid is RELION's single pass.
        """

        engine_calls = []
        install_fake_adaptive_engine(monkeypatch, engine_calls)
        rotations_many = _make_rotations(20, seed=334)
        monkeypatch.setattr(
            sampling_module,
            "relion_scoring_rotation_grid",
            lambda _order, dtype=None, *, symmetry='C1': sampling_module.RotationGrid(rotations=rotations_many, rotation_eulers=np.zeros((len(rotations_many), 3), dtype=np.float32), healpix_order=_order, symmetry=symmetry),
        )

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_healpix_order=1,
                    max_healpix_order=1,
                    skip_final_iteration=True,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=20),
                adaptive=stand_in.adaptive(relion_current_sizes=[8], adaptive_oversampling=0),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert np.asarray(result.maps.mean).shape == (VOLUME_SIZE,)
        assert len(engine_calls) == 2
        for call in engine_calls:
            assert call["kwargs"]["oversampling_order"] == 0
            assert call["kwargs"]["coarse_current_size"] == call["kwargs"]["fine_current_size"]
            assert call["kwargs"]["sparse_pass2"] is True
            assert np.asarray(call["fine_rotations"]).shape == np.asarray(call["coarse_rotations"]).shape

    def test_k_class_adaptive_significant_counts_are_recorded_without_convergence_effect(
        self,
        half_datasets,
        init_volume,
        translations,
        monkeypatch,
    ):
        """K-class significant counts are diagnostics, not convergence inputs."""

        monkeypatch.delenv("RELAX_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE", raising=False)
        counts_by_half = [
            np.array([8, 9], dtype=np.int32),
            np.array([10, 11], dtype=np.int32),
        ]
        call_idx = {"value": 0}
        fine_mstep_prune_values = []

        def fake_build_pass2_grids(effective_rotations, current_translations, base_translations, current_healpix_order, adaptive_oversampling, translation_step, random_perturbation, return_mstep_rotations=False, *, coarse_rotation_ids=None, symmetry='C1'):
            _ = (base_translations, current_healpix_order, adaptive_oversampling, translation_step, random_perturbation)
            coarse_rot = np.asarray(effective_rotations, dtype=np.float32)
            coarse_trans = np.asarray(current_translations, dtype=np.float32)
            rot_parent = np.arange(coarse_rot.shape[0], dtype=np.int32)
            trans_parent = np.arange(coarse_trans.shape[0], dtype=np.int32)
            base = (coarse_rot, coarse_trans, coarse_rot, coarse_trans, rot_parent, trans_parent)
            return (*base, coarse_rot) if return_mstep_rotations else base

        def fake_adaptive_k_class(
            experiment_dataset,
            means,
            mean_variance,
            noise_variance,
            coarse_rotations,
            coarse_translations,
            fine_rotations,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
            disc_type,
            **kwargs,
        ):
            _ = (
                means,
                mean_variance,
                noise_variance,
                coarse_translations,
                fine_translations,
                rot_parent_map,
                trans_parent_map,
                disc_type,
            )
            half_idx = call_idx["value"]
            call_idx["value"] += 1
            fine_mstep_prune_values.append(kwargs.get("relion_fine_mstep_prune"))
            n_images = int(experiment_dataset.n_units)
            n_classes = 2
            n_shells = experiment_dataset.image_shape[0] // 2 + 1
            recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
            counts = counts_by_half[half_idx]
            pmax_value = 0.6 if half_idx == 0 else 0.2
            retained_fraction = 0.8 if half_idx == 0 else 0.9
            assert counts.shape == (n_images,)
            per_class_stats = tuple(
                RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(coarse_rotations).shape[0], dtype=jnp.float32),
                )
                for _ in range(n_classes)
            )
            noise_stats = tuple(
                NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=retained_fraction * float(n_images) / float(n_classes),
                )
                for _ in range(n_classes)
            )
            return KClassEMResult(
                new_means=None,
                Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
                Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
                per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
                class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
                class_responsibilities=jnp.full((n_classes, n_images), 0.5, dtype=jnp.float32),
                class_posterior_sums=jnp.array([float(n_images) / 2.0, float(n_images) / 2.0], dtype=jnp.float32),
                stats=RelionStats(
                    log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                    max_posterior_per_image=jnp.full(n_images, pmax_value, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.ones(np.asarray(coarse_rotations).shape[0], dtype=jnp.float32),
                ),
                per_class_stats=per_class_stats,
                noise_stats=noise_stats,
                aggregate_noise_stats=NoiseStats(
                    wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=retained_fraction * float(n_images),
                ),
                best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
                best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
                best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
                significant_counts=jnp.asarray(counts, dtype=jnp.int32),
                class_mstep_posterior_sums=jnp.full(
                    n_classes,
                    retained_fraction * float(n_images) / float(n_classes),
                    dtype=jnp.float32,
                ),
            )

        monkeypatch.setattr(oversampling_grids, "build_adaptive_pass2_grids", fake_build_pass2_grids)
        monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive_k_class)
        monkeypatch.setattr(iteration_planning_module, "relion_coarse_image_size", lambda *_args, **_kwargs: 4)
        monkeypatch.setattr(finalization, "relion_coarse_image_size", lambda *_args, **_kwargs: 4)

        result = refine_single_volume(
            half_datasets,
            StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
            HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
            translations,
            options=stand_in.options(
                schedule=stand_in.schedule(
                    max_iter=1,
                    init_healpix_order=1,
                    max_healpix_order=1,
                    skip_final_iteration=True,
                ),
                execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=20),
                adaptive=stand_in.adaptive(relion_current_sizes=[8], adaptive_oversampling=1),
                parity=stand_in.parity(low_resol_join_halves_angstrom=0.0),
                k_class=KClassOptions(
                    n_classes=2,
                ),
            ),
            observer=RunObserver(), source=InputSource(),
        )

        assert call_idx["value"] == 2
        assert fine_mstep_prune_values == [True, True]
        assert_matches(
            np.asarray(result.history.significant_counts[0], dtype=np.int32),
            np.concatenate(counts_by_half),
        )
        assert np.isnan(result.history.acc_rot_trajectory[0])
        assert np.isinf(result.convergence_state.acc_rot)
        assert result.history.ave_Pmax_trajectory == pytest.approx([0.75], abs=1e-6)
        assert result.history.ave_Pmax_denominator_trajectory == pytest.approx(
            [0.8 * half_datasets[0].n_units],
            abs=1e-6,
        )
        assert result.convergence_state.ave_Pmax == pytest.approx(0.75, abs=1e-6)
        np.testing.assert_allclose(
            result.history.pmax_per_image_history[0],
            np.concatenate(
                [
                    np.full(half_datasets[0].n_units, 0.6),
                    np.full(half_datasets[1].n_units, 0.2),
                ]
            ),
            atol=1e-6,
        )


# ===========================================================================
# Test 3: Local search oversampling regression
# ===========================================================================


def test_local_search_uses_lazy_parent_expanded_fine_rotation_grid_when_oversampling_is_enabled(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Adaptive local search expands RELION coarse parents without materializing the full fine grid."""

    order_sizes = {4: 4, 5: 9}
    grid_calls = []
    local_calls = []

    def fake_rotation_grid_size(order, *, symmetry='C1'):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        grid_calls.append(("rot", order))
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        grid_calls.append(("euler", order))
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        disc_type,
        current_size,
        **kwargs,
    ):
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "rotations_is_none": rotation_grid_rotations is None,
                "rotation_grid_random_perturbation": kwargs.get("rotation_grid_random_perturbation"),
                "rotation_grid_angular_sampling_deg": kwargs.get("rotation_grid_angular_sampling_deg"),
                "local_parent_oversampling_order": kwargs.get("local_parent_oversampling_order"),
                "adaptive_fraction": kwargs.get("adaptive_fraction"),
            }
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_pose_details = (best_rots, best_trans)
        return _mock_local_search_result(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            experiment_dataset.n_units,
            best_pose_details,
        )

    monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(local_sampling, "_precompute_exact_local_fine_grid_enabled", lambda order, **kw: False)
    monkeypatch.setattr(
        sampling_module,
        "relion_scoring_rotation_grid",
        lambda order, *, dtype=np.float32, symmetry="C1": sampling_module.RotationGrid(rotations=fake_get_grid(order).astype(dtype), rotation_eulers=fake_get_grid_eulers(order).astype(dtype), healpix_order=order, symmetry=symmetry),
    )
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_grouped_local_search))
    monkeypatch.setattr(
        orientation_priors_module,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order, *, dtype=np.float32, symmetry='C1': (
            np.ones(
                max(1, fake_rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=1, init_current_size=16, init_healpix_order=4, max_healpix_order=4),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=order_sizes[4]),
            adaptive=stand_in.adaptive(adaptive_oversampling=1),
            start=StartState(init_previous_best_rotation_eulers=[prev_h1, prev_h2]),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    assert not any(kind == "rot" and order == 5 for kind, order in grid_calls)
    assert not any(kind == "euler" and order == 5 for kind, order in grid_calls)
    assert local_calls
    parent_calls = [call for call in local_calls if call["healpix_order"] == 4]
    fine_calls = [call for call in local_calls if call["healpix_order"] == 5]
    assert len(parent_calls) == 2
    assert len(fine_calls) == 2
    for call in fine_calls:
        assert call["healpix_order"] == 5
        assert call["rotations_is_none"]
        assert call["rotation_grid_random_perturbation"] == 0.0
        assert call["rotation_grid_angular_sampling_deg"] == pytest.approx(
            relion_angular_sampling_deg(5, adaptive_oversampling=0),
        )
        assert call["local_parent_oversampling_order"] == 1
        assert call["adaptive_fraction"] == pytest.approx(0.999)


def test_local_search_applies_perturbation_to_generated_fine_rotation_grid(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Selected-only fine local grids must carry the RELION perturbation metadata."""

    order_sizes = {4: 4, 5: 9}
    perturb_calls = []
    local_calls = []

    def fake_rotation_grid_size(order, *, symmetry='C1'):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_advance_relion_perturbation(current, perturb_factor, rng):
        return 0.25

    def fake_apply_relion_rotation_perturbation_to_eulers(
        eulers,
        random_perturbation,
        angular_sampling_deg,
        *,
        dtype=np.float32,
    ):
        perturb_calls.append(
            {
                "n_rot": int(np.asarray(eulers).shape[0]),
                "random_perturbation": float(random_perturbation),
                "angular_sampling_deg": float(angular_sampling_deg),
            }
        )
        sentinel_rotations = np.zeros((np.asarray(eulers).shape[0], 3, 3), dtype=dtype)
        sentinel_rotations[:, 0, 0] = 7.0
        sentinel_eulers = np.full((np.asarray(eulers).shape[0], 3), 5.0, dtype=dtype)
        if np.asarray(eulers).dtype == np.float64:
            sentinel_rotations = np.zeros_like(sentinel_rotations)
            sentinel_rotations[:, 1, 1] = 11.0
        return sentinel_rotations, sentinel_eulers

    def fake_r_to_relion(rotations, degrees=True):
        _ = degrees
        return np.full((np.asarray(rotations).shape[0], 3), 5.0, dtype=np.float32)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        disc_type,
        current_size,
        **kwargs,
    ):
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "rotations_is_none": rotation_grid_rotations is None,
                "rotation_grid_random_perturbation": kwargs.get("rotation_grid_random_perturbation"),
                "rotation_grid_angular_sampling_deg": kwargs.get("rotation_grid_angular_sampling_deg"),
                "generate_relion_mstep_rotations": kwargs.get("generate_relion_mstep_rotations"),
            }
        )
        n_shells = half_datasets[0].image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(half_datasets[0].n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(half_datasets[0].n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(half_datasets[0].n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(half_datasets[0].n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(half_datasets[0].n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], half_datasets[0].n_units, axis=0)
            best_trans = np.zeros((half_datasets[0].n_units, 2), dtype=np.float32)
            best_pose_details = (best_rots, best_trans)
        return _mock_local_search_result(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            half_datasets[0].n_units,
            best_pose_details,
        )

    monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(local_sampling, "_precompute_exact_local_fine_grid_enabled", lambda order, **kw: False)
    monkeypatch.setattr(
        sampling_module,
        "relion_scoring_rotation_grid",
        lambda order, *, dtype=np.float32, symmetry="C1": sampling_module.RotationGrid(rotations=fake_get_grid(order).astype(dtype), rotation_eulers=fake_get_grid_eulers(order).astype(dtype), healpix_order=order, symmetry=symmetry),
    )
    monkeypatch.setattr(
        sampling_module,
        "_get_relion_rotation_grid_eulers_float64",
        lambda order, *, symmetry='C1': fake_get_grid_eulers(order).astype(np.float64),
    )
    monkeypatch.setattr(sampling_module, "advance_relion_perturbation", fake_advance_relion_perturbation)
    monkeypatch.setattr(
        sampling_module,
        "apply_relion_rotation_perturbation_to_eulers",
        fake_apply_relion_rotation_perturbation_to_eulers,
    )
    monkeypatch.setattr(recovar_utils, "R_to_relion", fake_r_to_relion)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_grouped_local_search))
    monkeypatch.setattr(
        orientation_priors_module,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order, *, dtype=np.float32, symmetry='C1': (
            np.ones(12 * (2 ** int(healpix_order)) ** 2, dtype=np.float64)
            / (12 * (2 ** int(healpix_order)) ** 2)
        ),
    )
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=1, init_current_size=16, init_healpix_order=4, max_healpix_order=4),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=order_sizes[4]),
            adaptive=stand_in.adaptive(adaptive_oversampling=1),
            parity=stand_in.parity(perturb_factor=0.5),
            start=StartState(init_previous_best_rotation_eulers=[prev_h1, prev_h2]),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    assert any(
        call["n_rot"] == order_sizes[4]
        and np.isclose(call["angular_sampling_deg"], relion_angular_sampling_deg(4, adaptive_oversampling=0))
        for call in perturb_calls
    )
    assert not any(call["n_rot"] == order_sizes[5] for call in perturb_calls)
    assert local_calls
    fine_calls = [call for call in local_calls if call["healpix_order"] == 5]
    assert fine_calls
    assert fine_calls[0]["rotations_is_none"]
    assert fine_calls[0]["rotation_grid_random_perturbation"] == pytest.approx(0.25)
    assert fine_calls[0]["rotation_grid_angular_sampling_deg"] == pytest.approx(
        relion_angular_sampling_deg(5, adaptive_oversampling=0),
    )
    assert all(call["generate_relion_mstep_rotations"] is True for call in fine_calls)


def test_local_search_uses_negative_previous_offsets_for_translation_prior(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Local-search priors use RELION's pdf_offset units, not pre-shift pixels."""

    order_sizes = {4: 4, 5: 9}
    prev_h1 = np.array([[0.5, -0.25], [1.0, 0.75]], dtype=np.float32)
    prev_h2 = np.array([[-0.75, 0.25], [0.25, -1.25]], dtype=np.float32)
    local_prior_translations = []

    def fake_rotation_grid_size(order, *, symmetry='C1'):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        disc_type,
        current_size,
        **kwargs,
    ):
        local_prior_translations.append(np.asarray(prior_translations, dtype=np.float32).copy())
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_pose_details = (best_rots, best_trans)
        return _mock_local_search_result(
            base_outputs, relion_stats, noise_stats, kwargs,
            experiment_dataset.n_units, best_pose_details,
        )

    monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(local_sampling, "_precompute_exact_local_fine_grid_enabled", lambda order, **kw: False)
    monkeypatch.setattr(
        sampling_module,
        "relion_scoring_rotation_grid",
        lambda order, *, dtype=np.float32, symmetry="C1": sampling_module.RotationGrid(rotations=fake_get_grid(order).astype(dtype), rotation_eulers=fake_get_grid_eulers(order).astype(dtype), healpix_order=order, symmetry=symmetry),
    )
    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", _mock_run_adaptive_em)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_grouped_local_search))
    monkeypatch.setattr(
        orientation_priors_module,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order, *, dtype=np.float32, symmetry='C1': (
            np.ones(
                max(1, fake_rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    prev_eulers_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_eulers_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=1, init_current_size=16, init_healpix_order=4, max_healpix_order=4),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=order_sizes[4]),
            adaptive=stand_in.adaptive(adaptive_oversampling=1),
            start=StartState(
                init_previous_best_rotation_eulers=[prev_eulers_h1, prev_eulers_h2],
                init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
            ),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    assert len(local_prior_translations) == 4
    expected_h1 = -relion_translation_search_base(prev_h1) / 1.0
    expected_h2 = -relion_translation_search_base(prev_h2) / 1.0
    np.testing.assert_allclose(local_prior_translations[0], expected_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(local_prior_translations[1], expected_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(local_prior_translations[2], expected_h2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(local_prior_translations[3], expected_h2, rtol=1e-6, atol=1e-6)


def test_local_search_coarse_translation_prior_mode_uses_unperturbed_base_grid(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):

    order_sizes = {4: 4, 5: 9}
    prev_h1 = np.zeros((half_datasets[0].n_units, 2), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 2), dtype=np.float32)
    recorded_translation_reference_grids = []

    def fake_rotation_grid_size(order, *, symmetry='C1'):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        disc_type,
        current_size,
        **kwargs,
    ):
        recorded_translation_reference_grids.append(
            np.asarray(kwargs["translation_prior_reference_translations"], dtype=np.float32).copy()
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_pose_details = (best_rots, best_trans)
        return _mock_local_search_result(
            base_outputs, relion_stats, noise_stats, kwargs,
            experiment_dataset.n_units, best_pose_details,
        )

    monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(local_sampling, "_precompute_exact_local_fine_grid_enabled", lambda order, **kw: False)
    monkeypatch.setattr(
        sampling_module,
        "relion_scoring_rotation_grid",
        lambda order, *, dtype=np.float32, symmetry="C1": sampling_module.RotationGrid(rotations=fake_get_grid(order).astype(dtype), rotation_eulers=fake_get_grid_eulers(order).astype(dtype), healpix_order=order, symmetry=symmetry),
    )
    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", _mock_run_adaptive_em)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_grouped_local_search))
    monkeypatch.setattr(
        orientation_priors_module,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order, *, dtype=np.float32, symmetry='C1': (
            np.ones(12 * (2 ** int(healpix_order)) ** 2, dtype=np.float64)
            / (12 * (2 ** int(healpix_order)) ** 2)
        ),
    )

    # The mock datasets carry no particle IDs for the accuracy trials.
    monkeypatch.setattr(
        expected_accuracy_module.Half1AccuracyInputs, "estimate", lambda self, **kwargs: unconverged_accuracy()
    )

    prev_eulers_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_eulers_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=1, init_current_size=16, init_healpix_order=4, max_healpix_order=4),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=order_sizes[4]),
            adaptive=stand_in.adaptive(adaptive_oversampling=1),
            start=StartState(
                init_previous_best_rotation_eulers=[prev_eulers_h1, prev_eulers_h2],
                init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
            ),
            parity=stand_in.parity(perturb_factor=0.5, perturb_seed=0),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    assert recorded_translation_reference_grids
    coarse_grid = np.asarray(translations, dtype=np.float32)
    for grid in recorded_translation_reference_grids:
        np.testing.assert_allclose(grid, coarse_grid, rtol=1e-6, atol=1e-6)


def test_local_search_os0_keeps_full_local_support_for_mstep(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """RELION os0 local search keeps all fine candidates in storeWeightedSums."""

    order_sizes = {4: 4}
    reconstruct_flags = []

    def fake_rotation_grid_size(order, *, symmetry='C1'):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        return np.tile(np.eye(3, dtype=np.float32), (fake_rotation_grid_size(order), 1, 1))

    def fake_get_grid_eulers(order):
        return np.zeros((fake_rotation_grid_size(order), 3), dtype=np.float32)

    def fake_local_search(experiment_dataset, *args, **kwargs):
        _ = args
        reconstruct_flags.append(kwargs["reconstruct_significant_only"])
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        return LocalSearchResult(
            Ft_y=jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            Ft_ctf=jnp.ones(recon_vol_size, dtype=jnp.complex64),
            hard_assignment=np.zeros(experiment_dataset.n_units, dtype=np.int32),
            best_pose_rotations=np.tile(np.eye(3, dtype=np.float32)[None, :, :], (experiment_dataset.n_units, 1, 1)),
            best_pose_translations=np.zeros((experiment_dataset.n_units, 2), dtype=np.float32),
            relion_stats=RelionStats(
                log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(order_sizes[4], dtype=jnp.float32),
            ),
            noise_stats=NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(experiment_dataset.n_units),
            ),
        )

    monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(
        sampling_module,
        "relion_scoring_rotation_grid",
        lambda order, *, dtype=np.float32, symmetry="C1": sampling_module.RotationGrid(rotations=fake_get_grid(order).astype(dtype), rotation_eulers=fake_get_grid_eulers(order).astype(dtype), healpix_order=order, symmetry=symmetry),
    )
    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", _mock_run_adaptive_em)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_local_search))
    monkeypatch.setattr(
        orientation_priors_module,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order, *, dtype=np.float32, symmetry='C1': (
            np.ones(max(1, fake_rotation_grid_size(healpix_order)), dtype=np.float64)
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=2, init_current_size=16, init_healpix_order=4, max_healpix_order=4),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=order_sizes[4]),
            adaptive=stand_in.adaptive(adaptive_oversampling=0),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    # Both halves search locally in both iterations: iteration 1's pose-less halves are centred at (0, 0, 0), as
    # RELION reads absent angles (exp_model.cpp:1103-1134), instead of falling back to a global search.
    assert reconstruct_flags == [False] * 4


def test_local_search_coarse_translation_prior_mode_uses_replay_sampling_grid_when_available(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
    tmp_path,
):

    order_sizes = {4: 4, 5: 9}
    prev_h1 = np.zeros((half_datasets[0].n_units, 2), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 2), dtype=np.float32)
    recorded_translation_reference_grids = []

    relion_pixel_size = 4.25
    for ds in half_datasets:
        ds.voxel_size = relion_pixel_size
    replay_offset_range = 2.411663
    replay_offset_step = 1.220812

    def fake_rotation_grid_size(order, *, symmetry='C1'):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        return np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))

    def fake_get_grid_eulers(order):
        order = int(order)
        return np.zeros((order_sizes[order], 3), dtype=np.float32)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        disc_type,
        current_size,
        **kwargs,
    ):
        recorded_translation_reference_grids.append(
            np.asarray(kwargs["translation_prior_reference_translations"], dtype=np.float32).copy()
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_pose_details = (best_rots, best_trans)
        return _mock_local_search_result(
            base_outputs, relion_stats, noise_stats, kwargs,
            experiment_dataset.n_units, best_pose_details,
        )

    monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(
        sampling_module,
        "relion_scoring_rotation_grid",
        lambda order, *, dtype=np.float32, symmetry="C1": sampling_module.RotationGrid(rotations=fake_get_grid(order).astype(dtype), rotation_eulers=fake_get_grid_eulers(order).astype(dtype), healpix_order=order, symmetry=symmetry),
    )
    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", _mock_run_adaptive_em)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_grouped_local_search))
    monkeypatch.setattr(
        orientation_priors_module,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order, *, dtype=np.float32, symmetry='C1': (
            np.ones(max(1, fake_rotation_grid_size(healpix_order)), dtype=np.float64)
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )
    monkeypatch.setattr(
        relion_replay_module,
        "read_relion_sampling_metadata",
        lambda _path: {
            "random_perturbation": sampling_module.relion_sampling_perturbation_for_iteration(
                0.5, 0, 14
            ),
            "perturbation_factor": 0.5,
            "healpix_order": 5,
            "offset_range": replay_offset_range,
            "offset_step": replay_offset_step,
        },
    )
    monkeypatch.setattr(os.path, "exists", lambda _path: False)

    prev_eulers_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_eulers_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    _refine_replaying(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(
                max_iter=1,
                init_current_size=16,
                init_healpix_order=4,
                max_healpix_order=4,
                init_relion_iteration=13,
            ),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=order_sizes[4]),
            adaptive=stand_in.adaptive(adaptive_oversampling=1),
            start=StartState(
                init_previous_best_rotation_eulers=[prev_eulers_h1, prev_eulers_h2],
                init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()],
            ),
            parity=stand_in.parity(perturb_factor=0.5, perturb_seed=0),
        ),
        relion_replay=RelionReplay(perturb_replay_relion_dir=str(tmp_path)),
    )

    assert recorded_translation_reference_grids
    replay_grid = get_translation_grid(
        replay_offset_range / relion_pixel_size,
        replay_offset_step / relion_pixel_size,
    ).astype(np.float32)
    for grid in recorded_translation_reference_grids:
        np.testing.assert_allclose(grid, replay_grid, rtol=1e-6, atol=1e-6)


def test_replay_current_size_uses_control_model_star():
    assert _replay_control_model_iteration(0, 0) == 1
    assert _replay_control_model_iteration(1, 0) == 2
    assert _replay_control_model_iteration(13, 0) == 14


@pytest.mark.parametrize(
    ("replay_source", "rotation_seed"),
    [("iteration_override", 7), ("initial_state", 11)],
)
def test_previous_best_rotations_skip_first_local_dense_bootstrap(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
    replay_source,
    rotation_seed,
):
    """Both replay entry points enter hp4 local search without a dense bootstrap."""

    dense_calls = []
    local_calls = []
    prev_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    def fake_adaptive(experiment_dataset, means, mean_variance, noise_variance, coarse_rotations,
                      coarse_translations, fine_rotations, *grids, **kwargs):
        dense_calls.append(int(np.asarray(coarse_rotations).shape[0]))
        return adaptive_result(experiment_dataset, means, fine_rotations, kwargs)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        disc_type,
        current_size,
        **kwargs,
    ):
        local_calls.append(
            {
                "healpix_order": int(healpix_order),
                "prior_shape": np.asarray(prior_rotations).shape,
                "grid_shape": np.asarray(rotation_grid_rotations).shape,
            }
        )
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            np.zeros(experiment_dataset.n_units, dtype=np.int32),
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(np.asarray(rotation_grid_rotations).shape[0], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            best_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.zeros((experiment_dataset.n_units, 2), dtype=np.float32)
            best_pose_details = (best_rots, best_trans)
        return _mock_local_search_result(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            experiment_dataset.n_units,
            best_pose_details,
        )

    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_grouped_local_search))
    monkeypatch.setattr(
        orientation_priors_module,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order, *, dtype=np.float32, symmetry='C1': (
            np.ones(
                max(1, rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, rotation_grid_size(healpix_order))
        ),
    )

    relion_replay, replay = (
        (RelionReplay(replay_iteration_overrides=[{
            "local_search": True,
            "healpix_order": 4,
            "previous_best_rotation_eulers": [prev_h1, prev_h2],
        }]), StartState())
        if replay_source == "iteration_override"
        else (None, StartState(init_previous_best_rotation_eulers=[prev_h1, prev_h2]))
    )

    _refine_replaying(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(
                max_iter=1,
                init_current_size=16,
                init_healpix_order=4,
                max_healpix_order=4,
                skip_final_iteration=True,
            ),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=512),
            adaptive=stand_in.adaptive(adaptive_oversampling=0),
            start=replay,
        ),
        relion_replay=relion_replay,
    )

    assert local_calls
    assert not dense_calls
    for call in local_calls:
        assert call["healpix_order"] == 4
        assert call["prior_shape"][0] == half_datasets[0].n_units


def test_relion_mode_writes_absolute_translations_from_previous_offset(
    rng,
    init_volume,
    translations,
    monkeypatch,
):
    """RELION-mode writeback should use old_offset + delta."""

    half_datasets = [MockDataset(1, rng), MockDataset(1, rng)]
    for ds in half_datasets:
        ds.voxel_size = 4.25
    prev_h1 = np.array([[1.6, -2.4]], dtype=np.float32)
    prev_h2 = np.array([[-1.6, 2.4]], dtype=np.float32)
    chosen_trans = np.asarray(translations[1], dtype=np.float32)

    def fake_adaptive(experiment_dataset, means, mean_variance, noise_variance, coarse_rotations,
                      coarse_translations, fine_rotations, *grids, **kwargs):
        # The engine reports each image's best translation on the search grid around old_offset.
        n_images = experiment_dataset.n_units
        return adaptive_result(
            experiment_dataset,
            means,
            fine_rotations,
            kwargs,
            pose_assignments=np.full(n_images, 1, dtype=np.int32),
            best_pose_translations=np.repeat(chosen_trans[None, :], n_images, axis=0),
        )

    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive)

    result = refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(
                max_iter=1,
                init_current_size=16,
                init_healpix_order=1,
                max_healpix_order=1,
                skip_final_iteration=True,
            ),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=1),
            adaptive=stand_in.adaptive(adaptive_oversampling=1),
            start=StartState(init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()]),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    expected_h1 = relion_translation_search_base(prev_h1) + chosen_trans[None, :]
    expected_h2 = relion_translation_search_base(prev_h2) + chosen_trans[None, :]

    best_hist = result.history.best_translations_history
    assert len(best_hist) == 1
    np.testing.assert_allclose(
        np.concatenate(best_hist[0], axis=0),
        np.concatenate([expected_h1, expected_h2], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("capture_dump", [False, True])
def test_kclass_recomputes_mstep_tau2_from_iref_power_spectrum(
    rng,
    init_volume,
    monkeypatch,
    tmp_path,
    capture_dump,
):
    """Class3D M-step tau2 comes from current Iref power, not previous model.star."""

    from relax.diagnostics.observers import ClassDumpObserver

    floor_calls = []
    shell_stats = regularization_relion.compute_relion_weight_shell_stats

    def record_shell_stats(*args, **kwargs):
        result = shell_stats(*args, **kwargs)
        if kwargs.get("shell_rounding") == "floor" and inspect.currentframe().f_back.f_code.co_name == "write_class_mstep":
            floor_calls.append(result)
        return result

    monkeypatch.setattr(regularization_relion, "compute_relion_weight_shell_stats", record_shell_stats)

    half_datasets = [MockDataset(1, rng), MockDataset(1, rng)]
    n_classes = 2
    grid_scale = float(VOLUME_SHAPE[0]) ** 4
    class_tau2 = np.asarray(
        [
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ],
        dtype=np.float64,
    )
    iref_tau2 = np.asarray(
        [
            [101.0, 102.0, 103.0, 104.0, 105.0],
            [201.0, 202.0, 203.0, 204.0, 205.0],
        ],
        dtype=np.float64,
    )
    iref_tau2_calls = []

    def fake_iref_tau2(*_args, **kwargs):
        assert kwargs.get("return_details") is True
        class_idx = len(iref_tau2_calls) % n_classes
        iref_tau2_calls.append(class_idx)
        tau2_shells_relion = iref_tau2[class_idx] / grid_scale
        return (
            jnp.full(VOLUME_SIZE, tau2_shells_relion[0], dtype=jnp.float32),
            {"tau2_shells": tau2_shells_relion},
        )

    def fake_adaptive_k_class(
        experiment_dataset,
        means,
        mean_variance,
        noise_variance,
        coarse_rotations,
        coarse_translations,
        rotations,
        *grids_and_disc_type,
        **kwargs,
    ):
        del mean_variance, noise_variance, coarse_rotations, coarse_translations, grids_and_disc_type
        n_images = int(experiment_dataset.n_units)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        per_class_stats = tuple(
            RelionStats(
                log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            )
            for _ in range(n_classes)
        )
        per_class_noise = tuple(
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_images) / float(n_classes),
            )
            for _ in range(n_classes)
        )
        aggregate_noise = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(n_images),
        )
        return KClassEMResult(
            new_means=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
            Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
            Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
            per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
            class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            class_responsibilities=jnp.full((n_classes, n_images), 1.0 / n_classes, dtype=jnp.float32),
            class_posterior_sums=jnp.full(n_classes, n_images / n_classes, dtype=jnp.float32),
            stats=per_class_stats[0],
            per_class_stats=per_class_stats,
            noise_stats=per_class_noise,
            aggregate_noise_stats=aggregate_noise,
            best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
            best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
            best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        )

    monkeypatch.setattr(regularization_relion, "compute_relion_tau2_from_iref_power_spectrum", fake_iref_tau2)
    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive_k_class)

    result = _refine_replaying(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32)),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        options=stand_in.options(
            schedule=stand_in.schedule(
                max_iter=1,
                init_current_size=4,
                init_healpix_order=1,
                max_healpix_order=1,
                skip_final_iteration=True,
            ),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=1),
            adaptive=stand_in.adaptive(adaptive_oversampling=0),
            k_class=KClassOptions(
                n_classes=n_classes,
            ),
        ),
        relion_replay=RelionReplay(replay_iteration_overrides=[{"class_tau2": class_tau2}]),
        observer=ClassDumpObserver(tmp_path) if capture_dump else RunObserver(),
    )

    assert len(result.history.tau2_radial_trajectory) == 1
    np.testing.assert_allclose(result.history.tau2_radial_trajectory[0], iref_tau2, rtol=0.0, atol=1e-5)

    init_tau2_volume = jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32) * 3.0
    init_result = _refine_replaying(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), init_tau2_volume),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        options=stand_in.options(
            schedule=stand_in.schedule(
                max_iter=1,
                init_current_size=4,
                init_healpix_order=1,
                max_healpix_order=1,
                skip_final_iteration=True,
            ),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=1),
            adaptive=stand_in.adaptive(adaptive_oversampling=0),
            k_class=KClassOptions(
                n_classes=n_classes,
            ),
        ),
        observer=ClassDumpObserver(tmp_path) if capture_dump else RunObserver(),
    )
    assert len(init_result.history.tau2_radial_trajectory) == 1
    np.testing.assert_allclose(init_result.history.tau2_radial_trajectory[0], iref_tau2, rtol=0.0, atol=1e-5)
    assert iref_tau2_calls == [0, 1, 0, 1]

    same_iter_tau2 = class_tau2 + 1000.0
    monkeypatch.setenv("RELAX_KCLASS_REPLAY_TAU2", "1")
    replay_result = _refine_replaying(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32)),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        options=stand_in.options(
            schedule=stand_in.schedule(
                max_iter=1,
                init_current_size=4,
                init_healpix_order=1,
                max_healpix_order=1,
                skip_final_iteration=True,
            ),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=1),
            adaptive=stand_in.adaptive(adaptive_oversampling=0),
            k_class=KClassOptions(
                n_classes=n_classes,
            ),
        ),
        relion_replay=RelionReplay(
            replay_iteration_overrides=[{"class_tau2": class_tau2}, {"class_tau2": same_iter_tau2}],
        ),
        observer=ClassDumpObserver(tmp_path) if capture_dump else RunObserver(),
    )

    assert len(replay_result.history.tau2_radial_trajectory) == 1
    np.testing.assert_allclose(replay_result.history.tau2_radial_trajectory[0], same_iter_tau2, rtol=0.0, atol=1e-5)
    assert iref_tau2_calls == [0, 1, 0, 1]

    same_iter_replay_result = _refine_replaying(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones((n_classes, VOLUME_SIZE), dtype=jnp.float32)),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        options=stand_in.options(
            schedule=stand_in.schedule(
                max_iter=1,
                init_current_size=4,
                init_healpix_order=1,
                max_healpix_order=1,
                skip_final_iteration=True,
            ),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=1),
            adaptive=stand_in.adaptive(adaptive_oversampling=0),
            k_class=KClassOptions(
                n_classes=n_classes,
            ),
        ),
        relion_replay=RelionReplay(
            replay_iteration_overrides=[{"class_tau2": class_tau2}, {"class_tau2": same_iter_tau2}],
        ),
        observer=ClassDumpObserver(tmp_path) if capture_dump else RunObserver(),
    )

    assert len(same_iter_replay_result.history.tau2_radial_trajectory) == 1
    np.testing.assert_allclose(
        same_iter_replay_result.history.tau2_radial_trajectory[0],
        same_iter_tau2,
        rtol=0.0,
        atol=1e-5,
    )
    assert iref_tau2_calls == [0, 1, 0, 1]

    assert len(floor_calls) == (8 if capture_dump else 0)
    if capture_dump:
        for class_idx, stats in enumerate(floor_calls[-2:]):
            with np.load(tmp_path / f"recovar_kclass_mstep_it001_c{class_idx + 1:02d}.npz") as saved:
                assert_matches(saved["reconstruct_floor_avg_weight_shells"], stats["avg_weight_shells"])
                assert_matches(saved["reconstruct_floor_shell_count"], stats["shell_count"])


def test_relion_mode_k_class_writes_absolute_translations_from_previous_offset(
    rng,
    init_volume,
    monkeypatch,
):
    """K-class RELION-mode writeback (oversampling 0) should use old_offset + selected delta."""

    half_datasets = [MockDataset(1, rng), MockDataset(1, rng)]
    for ds in half_datasets:
        ds.voxel_size = 4.25
    prev_h1 = np.array([[1.6, -2.4]], dtype=np.float32)
    prev_h2 = np.array([[-1.6, 2.4]], dtype=np.float32)
    selected_by_half = [
        np.array([[0.25, -0.5]], dtype=np.float32),
        np.array([[-0.75, 1.5]], dtype=np.float32),
    ]
    dense_calls = []

    def fake_adaptive_k_class(
        experiment_dataset,
        means,
        mean_variance,
        noise_variance,
        coarse_rotations,
        coarse_translations,
        rotations,
        *grids_and_disc_type,
        **kwargs,
    ):
        half_idx = len(dense_calls)
        dense_calls.append(kwargs)
        n_classes = int(np.asarray(means).shape[0])
        n_images = int(experiment_dataset.n_units)
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        selected = np.broadcast_to(selected_by_half[half_idx], (n_images, 2)).astype(np.float32)
        per_class_stats = tuple(
            RelionStats(
                log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
                max_posterior_per_image=jnp.ones(n_images, dtype=jnp.float32),
                rotation_posterior_sums=jnp.ones(np.asarray(rotations).shape[0], dtype=jnp.float32),
            )
            for _ in range(n_classes)
        )
        per_class_noise = tuple(
            NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_images) / float(n_classes),
            )
            for _ in range(n_classes)
        )
        aggregate_noise = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(n_images),
        )
        return KClassEMResult(
            new_means=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
            Ft_y=jnp.zeros((n_classes, recon_vol_size), dtype=jnp.complex64),
            Ft_ctf=jnp.ones((n_classes, recon_vol_size), dtype=jnp.complex64),
            per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
            class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            class_responsibilities=jnp.full((n_classes, n_images), 1.0 / n_classes, dtype=jnp.float32),
            class_posterior_sums=jnp.full(n_classes, n_images / n_classes, dtype=jnp.float32),
            stats=per_class_stats[0],
            per_class_stats=per_class_stats,
            noise_stats=per_class_noise,
            aggregate_noise_stats=aggregate_noise,
            best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
            best_pose_translations=jnp.asarray(selected, dtype=jnp.float32),
            best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        )

    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake_adaptive_k_class)

    result = refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        options=stand_in.options(
            schedule=stand_in.schedule(
                max_iter=1,
                init_current_size=16,
                init_healpix_order=1,
                max_healpix_order=1,
                skip_final_iteration=True,
            ),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=1),
            adaptive=stand_in.adaptive(adaptive_oversampling=0),
            start=StartState(init_previous_best_translations=[prev_h1.copy(), prev_h2.copy()]),
            k_class=KClassOptions(n_classes=2),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    expected_h1 = relion_translation_search_base(prev_h1) + selected_by_half[0]
    expected_h2 = relion_translation_search_base(prev_h2) + selected_by_half[1]
    np.testing.assert_allclose(
        dense_calls[0]["translation_prior_centers"],
        relion_sigma_offset_prior_center(prev_h1),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        dense_calls[1]["translation_prior_centers"],
        relion_sigma_offset_prior_center(prev_h2),
        rtol=1e-6,
        atol=1e-6,
    )
    expected_translation_log_prior = make_relion_translation_log_prior(
        np.array([[0.0, 0.0]], dtype=np.float32),
        half_datasets[0].voxel_size,
        sigma_offset_angstrom=10.0,
        prior_centers=relion_translation_prior_center(prev_h1, half_datasets[0].voxel_size),
        offset_range_pixels=None,
    )
    np.testing.assert_allclose(
        dense_calls[0]["translation_log_prior"],
        expected_translation_log_prior,
        rtol=1e-6,
        atol=1e-6,
    )
    assert all(call["relion_half_volume_mstep"] is False for call in dense_calls)
    best_hist = result.history.best_translations_history
    assert len(best_hist) == 1
    np.testing.assert_allclose(best_hist[0][0], expected_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(best_hist[0][1], expected_h2, rtol=1e-6, atol=1e-6)
    assert len(dense_calls) == 2


def test_local_search_decodes_hard_assignments_on_fine_grid(
    half_datasets,
    init_volume,
    translations,
    monkeypatch,
):
    """Oversampled local-search assignments must be decoded on the fine grid."""

    order_sizes = {4: 4, 5: 9}
    fine_idx = order_sizes[5] - 1
    trans_idx = 1

    def fake_rotation_grid_size(order, *, symmetry='C1'):
        return order_sizes.get(int(order), order_sizes[4])

    def fake_get_grid(order):
        order = int(order)
        mats = np.tile(np.eye(3, dtype=np.float32), (order_sizes[order], 1, 1))
        for i in range(order_sizes[order]):
            mats[i, 0, 0] = 1.0 + i
        return mats

    def fake_get_grid_eulers(order):
        order = int(order)
        vals = np.arange(order_sizes[order], dtype=np.float32)
        return np.stack([vals, vals + 100.0, vals + 200.0], axis=1)

    def fake_grouped_local_search(
        experiment_dataset,
        mean,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        disc_type,
        current_size,
        **kwargs,
    ):
        n_shells = experiment_dataset.image_shape[0] // 2 + 1
        recon_vol_size = _mock_reconstruction_accumulator_size(experiment_dataset, kwargs)
        assignment = np.full(
            experiment_dataset.n_units,
            fine_idx * np.asarray(translations).shape[0] + trans_idx,
            dtype=np.int32,
        )
        base_outputs = (
            jnp.zeros(recon_vol_size, dtype=jnp.complex64),
            jnp.ones(recon_vol_size, dtype=jnp.complex64),
            assignment,
        )
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(experiment_dataset.n_units, dtype=jnp.float32),
            max_posterior_per_image=jnp.ones(experiment_dataset.n_units, dtype=jnp.float32),
            rotation_posterior_sums=jnp.ones(order_sizes[int(healpix_order)], dtype=jnp.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(experiment_dataset.n_units),
        )
        best_pose_details = ()
        if kwargs.get("return_best_pose_details"):
            fine_rot = selected_rotation_matrices(
                np.array([fine_idx], dtype=np.int32),
                None,
                build_local_search_grid_metadata(int(healpix_order)),
            )[0].astype(np.float32)
            best_rots = np.repeat(fine_rot[None, :, :], experiment_dataset.n_units, axis=0)
            best_trans = np.repeat(
                np.asarray(translations)[trans_idx : trans_idx + 1], experiment_dataset.n_units, axis=0
            )
            best_pose_details = (best_rots, best_trans)
        return _mock_local_search_result(
            base_outputs,
            relion_stats,
            noise_stats,
            kwargs,
            experiment_dataset.n_units,
            best_pose_details,
        )

    monkeypatch.setattr(maximization_module, "rotation_grid_size", fake_rotation_grid_size)
    monkeypatch.setattr(
        sampling_module,
        "relion_scoring_rotation_grid",
        lambda order, *, dtype=np.float32, symmetry="C1": sampling_module.RotationGrid(rotations=fake_get_grid(order).astype(dtype), rotation_eulers=fake_get_grid_eulers(order).astype(dtype), healpix_order=order, symmetry=symmetry),
    )
    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", _mock_run_adaptive_em)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", local_iteration_keywords(fake_grouped_local_search))
    monkeypatch.setattr(
        orientation_priors_module,
        "collapse_rotation_posterior_to_direction_prior",
        lambda rotation_posterior_sums, healpix_order, *, dtype=np.float32, symmetry='C1': (
            np.ones(
                max(1, fake_rotation_grid_size(healpix_order)),
                dtype=np.float64,
            )
            / max(1, fake_rotation_grid_size(healpix_order))
        ),
    )

    prev_eulers_h1 = np.zeros((half_datasets[0].n_units, 3), dtype=np.float32)
    prev_eulers_h2 = np.zeros((half_datasets[1].n_units, 3), dtype=np.float32)

    result = refine_single_volume(
        half_datasets,
        StartupHandoff(HalfPair.shared(init_volume), jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        translations,
        options=stand_in.options(
            schedule=stand_in.schedule(max_iter=1, init_current_size=16, init_healpix_order=4, max_healpix_order=4),
            execution=stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=order_sizes[4]),
            adaptive=stand_in.adaptive(adaptive_oversampling=1),
            start=StartState(init_previous_best_rotation_eulers=[prev_eulers_h1, prev_eulers_h2]),
            parity=stand_in.parity(perturb_factor=0.0),
        ),
        observer=RunObserver(), source=InputSource(),
    )

    expected_rotation = selected_rotation_matrices(
        np.array([fine_idx], dtype=np.int32),
        None,
        build_local_search_grid_metadata(5),
    )
    expected_euler = recovar_utils.R_to_relion(expected_rotation, degrees=True)[0].astype(np.float32)
    observed = np.asarray(result.history.best_rotation_eulers_history[0], dtype=np.float32).reshape(-1, 3)
    assert observed.shape[0] == N_IMAGES
    np.testing.assert_allclose(
        observed,
        np.repeat(expected_euler[None, :], N_IMAGES, axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_canonical_rotation_grid_reuses_relion_euler_table(monkeypatch):
    """The direct grid owner preserves source Eulers without a SciPy round trip."""
    order = 1
    expected_eulers = np.asarray(get_relion_rotation_grid_eulers(order), dtype=np.float32)

    def fail_r_to_relion(*_args, **_kwargs):
        raise AssertionError("generic R_to_relion should not be called for canonical grids")

    monkeypatch.setattr(recovar_utils, "R_to_relion", fail_r_to_relion)
    _rotation_grid_ = sampling_module.relion_scoring_rotation_grid(order)
    _ = _rotation_grid_.rotations
    got = _rotation_grid_.rotation_eulers
    assert_matches(got, expected_eulers)


def test_texture_full_even_nyquist_indices_validate_and_match_full_scatter():
    from relax.helpers.projection import (
        _texture_centered_crop_at_indices,
        _texture_centered_crop_to_full,
        _validate_centered_relion_projector_pixel_indices,
    )

    image_size = 8
    crop_pixels = image_size * (image_size // 2 + 1)
    crop = (
        np.arange(crop_pixels, dtype=np.float32)
        + 1j * np.arange(crop_pixels, dtype=np.float32)[::-1]
    ).astype(np.complex64)[None]
    nyquist_row_indices = np.arange(image_size // 2 + 1, dtype=np.int32)

    _validate_centered_relion_projector_pixel_indices(
        nyquist_row_indices,
        image_shape=(image_size, image_size),
        projector_output_size=image_size,
    )
    full = np.asarray(
        _texture_centered_crop_to_full(
            jnp.asarray(crop),
            image_shape=(image_size, image_size),
            projector_output_size=image_size,
        )
    )
    direct = np.asarray(
        _texture_centered_crop_at_indices(
            jnp.asarray(crop),
            jnp.asarray(nyquist_row_indices),
            image_shape=(image_size, image_size),
            projector_output_size=image_size,
        )
    )

    assert_matches(direct, full[:, nyquist_row_indices])


def test_texture_cropped_projector_rejects_rows_outside_crop():
    from relax.helpers.projection import (
        _validate_centered_relion_projector_pixel_indices,
    )

    with pytest.raises(ValueError, match=r"bad_indices=\[0, 5\]"):
        _validate_centered_relion_projector_pixel_indices(
            np.asarray([0, 5], dtype=np.int32),
            image_shape=(8, 8),
            projector_output_size=6,
        )


def test_texture_full_projector_rejects_out_of_bounds_flat_index():
    from relax.helpers.projection import (
        _validate_centered_relion_projector_pixel_indices,
    )

    with pytest.raises(ValueError, match=r"bad_indices=\[40\]"):
        _validate_centered_relion_projector_pixel_indices(
            np.asarray([40], dtype=np.int32),
            image_shape=(8, 8),
            projector_output_size=8,
        )


def test_texture_centered_crop_preserves_kernel_owned_rounded_outer_shell():
    from relax.helpers.projection import _texture_centered_crop_to_full

    crop = jnp.ones((1, 4 * 3), dtype=jnp.complex64)
    got = np.asarray(
        _texture_centered_crop_to_full(
            crop,
            image_shape=(4, 4),
            projector_output_size=4,
        )
    ).reshape(1, 4, 3)
    expected = np.ones((1, 4, 3), dtype=np.complex64)
    assert_matches(got, expected)


def test_production_k4_firstiter_has_one_joint_winner_and_exact_mstep_mass(rng, monkeypatch):
    """The production firstiter route must reconstruct each image in exactly one class.

    The fine pass is the sparse global-winner subset pass; its device-resident pass 2 is
    replaced by a stand-in that refines each image to its class's coarse winner and
    reports one unit of posterior mass per image.
    """

    import copy

    import relax.classification.k_class as k_class_module
    from relax.helpers.types import SparsePass2Output, make_relion_stats
    from relax.scoring import significance as significance_module
    from relax.sparse_pass2 import dispatch as sparse_dispatch

    class SubsetMockDataset(MockDataset):
        def subset(self, image_indices):
            image_indices = np.asarray(image_indices, dtype=np.int64)
            subset = copy.copy(self)
            subset._images = self._images[image_indices].copy()
            subset.CTF_params = self.CTF_params[image_indices].copy()
            subset.rotation_matrices = self.rotation_matrices[image_indices].copy()
            subset.translations = self.translations[image_indices].copy()
            subset.n_images = int(image_indices.size)
            subset.n_units = int(image_indices.size)
            return subset

    n_classes = 4
    n_images = 7
    expected_classes = np.asarray([0, 1, 1, 2, 3, 3, 3], dtype=np.int32)
    expected_class_mass = np.bincount(expected_classes, minlength=n_classes).astype(np.float32)
    coarse_hard = np.asarray(
        [
            [(class_idx + image_idx) % 4 for image_idx in range(n_images)]
            for class_idx in range(n_classes)
        ],
        dtype=np.int32,
    )
    coarse_log_evidence = np.full((n_classes, n_images), -2.0, dtype=np.float64)
    coarse_best_scores = np.full((n_classes, n_images), -20.0, dtype=np.float32)
    for image_idx, class_idx in enumerate(expected_classes):
        coarse_log_evidence[class_idx, image_idx] = 0.0
        coarse_best_scores[class_idx, image_idx] = 20.0 + image_idx

    score_calls = []

    def fake_joint_coarse_score(*_args, **kwargs):
        score_calls.append(kwargs)
        assert_matches(kwargs["class_log_priors"], np.zeros(n_classes, dtype=np.float64))
        assert kwargs["score_mode"] == "normalized_cc"
        assert kwargs["max_significants"] == 1
        assert kwargs["return_class_best"] is True
        return Pass1Result(
            None,
            None,
            None,
            expected_classes.copy(),
            None,
            {
                "class_log_evidence_per_image": coarse_log_evidence.copy(),
                "class_hard_assignments": coarse_hard.copy(),
                "class_best_log_score_per_image": coarse_best_scores.copy(),
                "class_assignments": expected_classes.copy(),
            },
        )

    monkeypatch.setattr(
        significance_module,
        "_compute_k_class_significance_batched",
        fake_joint_coarse_score,
    )
    sparse_calls = []

    def fake_sparse_pass2(subset, volume, _mean_variance, _noise_variance, translations_arg, samples, **kwargs):
        sparse_calls.append(kwargs)
        n_subset = int(subset.n_units)
        n_trans = int(np.asarray(translations_arg).shape[0])
        # Each image's support is its class's single coarse winner (rotation * n_trans + translation).
        winners = np.asarray([int(np.asarray(sample).reshape(-1)[0]) for sample in samples], dtype=np.int32)
        n_shells = IMAGE_SHAPE[0] // 2 + 1
        return SparsePass2Output(
            jnp.zeros_like(volume),
            jnp.ones_like(jnp.real(volume)),
            winners % n_trans,
            np.repeat(np.eye(3, dtype=np.float32)[None], n_subset, axis=0),
            np.zeros((n_subset, 2), dtype=np.float32),
            winners // n_trans,
            make_relion_stats(
                log_evidence_per_image=np.zeros(n_subset, dtype=np.float32),
                best_log_score_per_image=np.zeros(n_subset, dtype=np.float32),
                max_posterior_per_image=np.ones(n_subset, dtype=np.float32),
                rotation_posterior_sums=np.zeros(2, dtype=np.float32),
            ),
            noise_stats=NoiseStats(
                wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
                wsum_sigma2_offset=0.0,
                sumw=float(n_subset),
            ),
        )

    monkeypatch.setattr(sparse_dispatch, "compute_pass2_stats_sparse", fake_sparse_pass2)

    dataset = SubsetMockDataset(n_images, rng)
    mean = _hermitian_volume(VOLUME_SHAPE, seed=145)
    means = jnp.stack([mean * np.float32(1.0 + 0.05 * class_idx) for class_idx in range(n_classes)])
    mean_variance = jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)
    rotations = _make_rotations(2, seed=146)
    translations = np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)

    result = k_class_module.run_dense_k_class_em_adaptive(
        dataset,
        means,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        rotations,
        translations,
        np.arange(rotations.shape[0], dtype=np.int64),
        np.arange(translations.shape[0], dtype=np.int64),
        "linear_interp",
        firstiter_cc_pass2_only_best_coarse=True,
        coarse_healpix_order=0,
        oversampling_order=0,
        accumulate_noise=True,
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
        sparse_pass2=True,
        image_batch_size=n_images,
        rotation_block_size=rotations.shape[0],
        score_with_masked_images=True,
    )

    assert len(score_calls) == 1
    # One fine pass per class with images, each a winner-take-all normalized-CC pass.
    assert len(sparse_calls) == n_classes
    assert all(call["relion_firstiter_winner_take_all"] is True for call in sparse_calls)
    assert_matches(np.asarray(result.class_assignments), expected_classes)
    expected_poses = coarse_hard[expected_classes, np.arange(n_images)]
    assert_matches(np.asarray(result.pose_assignments), expected_poses)
    assert_matches(np.asarray(result.significant_counts), np.ones(n_images, dtype=np.int32))
    assert_matches(np.asarray(result.stats.max_posterior_per_image), np.ones(n_images))

    finite_class_pose_support = np.stack(
        [np.isfinite(np.asarray(stats.best_log_score_per_image)) for stats in result.per_class_stats],
        axis=0,
    )
    assert_matches(finite_class_pose_support.sum(axis=0), np.ones(n_images, dtype=np.int64))
    assert_matches(np.argmax(finite_class_pose_support, axis=0), expected_classes)
    assert_matches(np.asarray(result.class_mstep_posterior_sums), expected_class_mass)
    assert result.noise_stats is not None
    assert_matches(
        np.asarray([stats.sumw for stats in result.noise_stats], dtype=np.float32),
        expected_class_mass,
    )
    assert result.aggregate_noise_stats is not None
    assert result.aggregate_noise_stats.sumw == pytest.approx(float(n_images))


def test_large_host_reconstruction_padding_retains_device_window(monkeypatch):
    """The donating host gather is crop-only; Fourier padding stays on device."""
    from recovar.reconstruction import relion_functions

    from relax.reconstruction import relion_functions_relion
    from relax.refinement import numbered_reconstruction as numbered_reconstruction_module

    events = []
    host_boundary = np.ones((4, 4, 3), dtype=np.complex64)

    class DeviceBoundary:
        def block_until_ready(self):
            events.append("block")

        def __del__(self):
            events.append("release")

    def fake_stage(*_args, **kwargs):
        events.append("stage")
        assert kwargs["input_half_volume"] is True
        assert kwargs["return_fftw_half_before_ifft"] is True
        assert "return_wiener_half_before_window" not in kwargs
        return DeviceBoundary()

    def fake_device_get(_value):
        events.append("device_get")
        return host_boundary

    def fake_finish(value, *_args, **_kwargs):
        events.append("finish")
        assert events == ["stage", "block", "device_get", "release", "finish"]
        assert value is host_boundary
        return value

    def reject_crop(*_args, **_kwargs):
        raise AssertionError("Fourier padding must not use the host crop helper")

    def reject_donating_stage(*_args, **_kwargs):
        raise AssertionError("Fourier padding must not use the crop-only donating stage")

    monkeypatch.setattr(
        relion_functions,
        "_large_grid_postprocess_single_precision_enabled",
        lambda _voxels: True,
    )
    monkeypatch.setattr(relion_functions, "post_process_from_filter_v2", fake_stage)
    monkeypatch.setattr(
        relion_functions,
        "_post_process_from_filter_v2_donate_numerator",
        reject_donating_stage,
    )
    monkeypatch.setattr(
        relion_functions_relion,
        "divide_large_relion_half_numerator_donate_numerator",
        reject_donating_stage,
    )
    monkeypatch.setattr(
        relion_functions_relion,
        "regularize_large_relion_half_filter_donate_ctf",
        reject_donating_stage,
    )
    monkeypatch.setattr(
        relion_functions_relion,
        "finish_large_relion_postprocess_from_fftw_half",
        fake_finish,
    )
    monkeypatch.setattr(
        volume_solver,
        "_crop_relion_wiener_half_to_fftw_host",
        reject_crop,
    )
    monkeypatch.setattr(volume_solver.jax, "device_get", fake_device_get)

    volume_shape = (2, 2, 2)
    accumulator_shape = (3, 3, 3)
    half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
    returned = numbered_reconstruction_module._reconstruct_volume_eager(
        np.ones(half_shape, dtype=np.float32),
        np.ones(half_shape, dtype=np.complex64),
        volume_shape,
        2,
        tau=np.ones(np.prod(volume_shape), dtype=np.float32),
        tau2_fudge=1.0,
        projection_padding_factor=1,
        accumulator_volume_shape=accumulator_shape, programs=ReconstructionPrograms.from_environ(),
    )

    assert returned is host_boundary
    assert events == ["stage", "block", "device_get", "release", "finish"]


@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("need_unreg", [False, True])
def test_k_class_reconstruction_preserves_data_determined_volume_signs(monkeypatch, n_classes, need_unreg):
    reconstructed = np.asarray([[2 + k + 0j, -1 - k + 0j] for k in range(n_classes)], dtype=np.complex64)
    previous = reconstructed.copy()
    previous[1::2] *= -1
    unregularized = reconstructed * np.complex64(2)
    # A Class3D M-step leaves both halves one shared class stack.
    shared = jnp.asarray(reconstructed)
    means = [shared, shared]
    calls = []

    def reconstruct(_weight, numerator, *_args, **_kwargs):
        calls.append(np.asarray(numerator).copy())
        return numerator

    monkeypatch.setattr(numbered_reconstruction_module, "_reconstruct_volume_eager", reconstruct)
    result = (
        numbered_reconstruction_module.reconstruct_unregularized_class_means(
            jnp.asarray(unregularized),
            jnp.ones_like(jnp.asarray(unregularized).real),
            reconstruction_settings(
                box_size=2, voxel_size=1.0, volume_shape=(2, 1, 1),
                padding_factor=1, projection_padding_factor=1, minres_map=1,
                width_mask_edge=5, fmask_edge=2, tau2_fudge=1.0,
                particle_diameter_angstrom=None, first_iteration_lowpass_angstrom=None, programs=ReconstructionPrograms.from_environ(),
            ),
            n_classes,
        )
        if need_unreg
        else [None, None]
    )
    assert_matches(means[0], reconstructed)
    assert_matches(means[1], reconstructed)
    assert means[0] is means[1]
    if need_unreg:
        assert len(calls) == n_classes
        assert_matches(result[0], unregularized)
        assert_matches(result[1], unregularized)
        assert result[0] is result[1]
    else:
        assert calls == []
        assert result == [None, None]


def test_non_c1_small_rotation_grid_keeps_adaptive_sparse_route():
    """O symmetry has 12 coarse rotations and must not fall through to dense reconstruction.

    Ported from final Q 22efd8065. Without the symmetry clause the D6 O/I1 K=4
    trajectories (5k/128, HEALPix order 1) never built RELION's Projector::data
    and diverged from RELION Class3D in iteration 1.
    """

    def choose(*, adaptive_oversampling, use_local, n_rotations, symmetry):
        return local_sampling._should_use_adaptive_search(
            RefinementState(adaptive_oversampling=adaptive_oversampling),
            stand_in.options(symmetry=SymmetryOptions(point_group=symmetry)),
            use_local=use_local, n_rotations=n_rotations,
        )

    assert choose(adaptive_oversampling=1, use_local=False, n_rotations=12, symmetry="O")
    assert choose(adaptive_oversampling=1, use_local=False, n_rotations=5, symmetry="I1")
    assert choose(adaptive_oversampling=1, use_local=False, n_rotations=144, symmetry="C4")
    assert choose(adaptive_oversampling=1, use_local=False, n_rotations=17, symmetry="C1")
    assert not choose(adaptive_oversampling=1, use_local=False, n_rotations=12, symmetry="C1")
    assert not choose(adaptive_oversampling=0, use_local=False, n_rotations=12, symmetry="O")
    assert not choose(adaptive_oversampling=1, use_local=True, n_rotations=12, symmetry="O")
