"""Behavioral contracts for K-class policy and diagnostic captures."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches, matches

import relax.helpers.oversampling as oversampling_mod
import relax.scoring.significance as sig_mod
from relax.dense import score_outputs, scoring_policy
from relax.diagnostics import coarse_gaussian_diagnostics
from relax.diagnostics import iteration as debug_dumps

pytestmark = pytest.mark.unit

def test_kclass_mstep_defaults_to_relion_x_half_with_full_volume_escape_hatch(monkeypatch):
    """K-class quality parity should use RELION x-half BPref accumulators by default."""

    monkeypatch.delenv("RELAX_K_CLASS_RELION_X_HALF_MSTEP", raising=False)
    assert scoring_policy._k_class_relion_x_half_mstep_enabled() is True

    monkeypatch.setenv("RELAX_K_CLASS_RELION_X_HALF_MSTEP", "0")
    assert scoring_policy._k_class_relion_x_half_mstep_enabled() is False

    monkeypatch.setenv("RELAX_K_CLASS_RELION_X_HALF_MSTEP", "1")
    assert scoring_policy._k_class_relion_x_half_mstep_enabled() is True

def test_k1_relion_x_half_mstep_defaults_on_with_escape_hatch(monkeypatch):
    """K=1 adaptive RELION mode should use x-half BPref layout by default."""

    monkeypatch.delenv(scoring_policy._K1_RELION_X_HALF_MSTEP_ENV, raising=False)
    monkeypatch.setattr(scoring_policy, "_k1_relion_x_half_mstep_default_available", lambda: True)
    assert scoring_policy._k1_relion_x_half_mstep_enabled() is True

    monkeypatch.setenv(scoring_policy._K1_RELION_X_HALF_MSTEP_ENV, "0")
    assert scoring_policy._k1_relion_x_half_mstep_enabled() is False

    monkeypatch.setenv(scoring_policy._K1_RELION_X_HALF_MSTEP_ENV, "1")
    assert scoring_policy._k1_relion_x_half_mstep_enabled() is True

    monkeypatch.setenv(scoring_policy._K1_RELION_X_HALF_MSTEP_ENV, "invalid")
    assert scoring_policy._k1_relion_x_half_mstep_enabled() is True

def test_k1_relion_x_half_mstep_default_disables_when_cuda_unavailable(monkeypatch):
    """The default must not request CUDA-only x-half adjoints on CPU tests."""

    monkeypatch.delenv(scoring_policy._K1_RELION_X_HALF_MSTEP_ENV, raising=False)
    monkeypatch.setattr(scoring_policy, "_k1_relion_x_half_mstep_default_available", lambda: False)
    assert scoring_policy._k1_relion_x_half_mstep_enabled() is False

    monkeypatch.setenv(scoring_policy._K1_RELION_X_HALF_MSTEP_ENV, "1")
    assert scoring_policy._k1_relion_x_half_mstep_enabled() is True


def test_kclass_result_uses_mstep_class_mass_for_relion_priors():
    """RELION Class3D occupancies come from StoreWeightedSums, not full evidence sums."""

    stats = [
        SimpleNamespace(rotation_posterior_sums=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        SimpleNamespace(rotation_posterior_sums=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    result = SimpleNamespace(
        pose_assignments=np.array([0, 1, 2], dtype=np.int32),
        noise_stats=("class0", "class1"),
        class_assignments=np.array([0, 1, 1], dtype=np.int32),
        class_posterior_sums=np.array([1.7, 1.3], dtype=np.float32),
        class_mstep_posterior_sums=np.array([1.2, 1.8], dtype=np.float32),
        per_class_stats=stats,
        Ft_y="ft_y",
        Ft_ctf="ft_ctf",
        stats="stats",
        aggregate_noise_stats="aggregate_noise",
        best_pose_rotations=np.repeat(np.eye(3, dtype=np.float64)[None], 3, axis=0),
        best_pose_translations=np.asarray([[0.1, -0.2], [0.3, -0.4], [0.5, -0.6]], dtype=np.float64),
    )
    prepared = score_outputs.class_em_to_half_result(
        result,
        effective_rotations=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
        rot_pmap_for_collapse=None,
        adaptive_os_local=0,
        require_best_pose_details=True,
        pose_dtype=np.float64,
    )

    np.testing.assert_allclose(prepared.classes.mstep_mass, [1.2, 1.8])
    np.testing.assert_allclose(prepared.classes.evidence_mass, [1.7, 1.3])
    assert prepared.best_pose_rotations.dtype == np.float64
    assert prepared.best_pose_rotation_eulers.dtype == np.float64
    assert prepared.best_pose_translations.dtype == np.float64

def test_class_weight_history_snapshots_mstep_and_full_posterior():
    from relax.helpers import iteration_history

    history = iteration_history.RefinementHistory()
    mstep = np.asarray([0.25, 0.75], dtype=np.float64)
    posterior = np.asarray([0.5, 0.5], dtype=np.float64)
    history.record_class_weights(mstep, posterior)
    mstep[:] = posterior[:] = 0.0
    assert_matches(history.class_mstep_weight_trajectory, [[0.25, 0.75]])
    assert_matches(history.class_full_posterior_weight_trajectory, [[0.5, 0.5]])

def test_kclass_weight_trajectories_record_mstep_and_full_posterior_provenance(monkeypatch, tmp_path):
    """Full-chain NPZ output must expose the class-mass split used in parity debugging."""

    from helpers.tiny_refinement import N_IMAGES, CallTrace, run_tiny_refinement

    from relax.helpers.iteration_history import RefinementHistory
    from relax.refinement import result_files

    trace = CallTrace(monkeypatch).wrap(RefinementHistory, "record_class_weights")
    result = run_tiny_refinement(monkeypatch, n_classes=2, final_after_max_iter=False)
    recorded = trace.calls("record_class_weights")
    assert len(recorded) == 2

    archive = {}
    result_files.write_refinement_archive(
        result, out_path=tmp_path / "refinement.npz", metadata=archive,
        half_indices=(np.arange(N_IMAGES // 2), np.arange(N_IMAGES // 2, N_IMAGES)),
        n_images=N_IMAGES, skip_large_outputs=True,
    )
    for key, argument in (("class_mstep_weight_trajectory", 1), ("class_full_posterior_weight_trajectory", 2)):
        assert archive[key].dtype == np.float64 and archive[key].shape == (2, 2)
        assert_matches(archive[key], np.stack([call.args[argument] for call in recorded]))


def test_production_runner_writes_the_refinement_archive(monkeypatch, tmp_path):
    """relax class3d writes both class-weight trajectories into its archive."""
    from helpers.tiny_main import run_tiny_main

    output = run_tiny_main(monkeypatch, tmp_path, "class3d", "--max_iter", "1", "--n_classes", "2", n_classes=2)
    with np.load(output / "refinement_results.npz", allow_pickle=True) as archive:
        for key in ("class_mstep_weight_trajectory", "class_full_posterior_weight_trajectory"):
            assert archive[key].shape == (1, 2) and archive[key].dtype == np.float64


def test_significance_dump_work_is_gated_before_scoring(monkeypatch, tmp_path):
    """A future-only dump request must not activate diagnostic scoring work."""

    for name in (
        "RELAX_SIGNIFICANCE_DUMP_DIR",
        "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES",
        "RELAX_SIGNIFICANCE_DUMP_CURRENT_SIZE",
        "RELAX_SIGNIFICANCE_DUMP_ITERATION",
    ):
        monkeypatch.delenv(name, raising=False)

    matches = sig_mod._significance_debug_dump_matches
    assert not matches(current_size=32, debug_iteration=1)

    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(tmp_path))
    assert not matches(current_size=32, debug_iteration=1)

    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    assert matches(current_size=32, debug_iteration=1)

    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_CURRENT_SIZE", "64")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ITERATION", "11")
    assert not matches(current_size=32, debug_iteration=1)
    assert not matches(current_size=64, debug_iteration=1)
    assert not matches(current_size=32, debug_iteration=11)
    assert matches(current_size=64, debug_iteration=11)

def test_kclass_dump_writes_operand_arrays_to_npz(monkeypatch, tmp_path):
    """End-to-end behavioral test: a one-particle invocation of
    ``_maybe_dump_k_class_significance_batch`` with RELION's exact coarse
    operands set writes them to the npz with sensible shapes/dtypes.
    """

    n_images = 1
    n_classes = 2
    n_rot = 3
    n_trans = 4
    n_pix = 5

    indices = np.array([0], dtype=np.int64)
    experiment_dataset = SimpleNamespace(
        dataset_indices=np.array([42], dtype=np.int64),
    )

    rotations = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
    translations = np.zeros((n_trans, 2), dtype=np.float32)
    class_weight_mats = [
        np.ones((n_images, n_rot * n_trans), dtype=np.float64) / (n_rot * n_trans)
        for _ in range(n_classes)
    ]
    batch_sig_mask = np.ones(
        (n_images, n_classes * n_rot * n_trans), dtype=bool
    )
    batch_n_sig = np.array([n_classes * n_rot * n_trans], dtype=np.int64)
    hard_assignment_batch = np.array([0], dtype=np.int64)
    class_assignment_batch = np.array([0], dtype=np.int64)
    global_log_z = np.array([0.0], dtype=np.float64)
    class_log_z_values = [np.array([-0.69], dtype=np.float64) for _ in range(n_classes)]
    best_score = np.array([0.0], dtype=np.float64)
    max_posterior = np.array([0.5], dtype=np.float64)
    class_log_priors = np.zeros(n_classes, dtype=np.float64)

    coarse_gaussian_shifted_corrected = np.arange(
        n_images * n_trans * n_pix,
        dtype=np.float32,
    ).reshape(n_images, n_trans, n_pix).astype(np.complex64)
    coarse_gaussian_pixel_weight = np.ones((n_images, n_pix), dtype=np.float32)
    coarse_gaussian_initial_diff2 = np.full(n_images, 3.0, dtype=np.float32)
    coarse_gaussian_score_indices = np.arange(n_pix, dtype=np.int32)
    relion_projector_half = [
        np.full((3, 4, 2), class_index + 1j, dtype=np.complex64)
        for class_index in range(n_classes)
    ]
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ITERATION", "2")
    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=experiment_dataset,
        indices=indices,
        n_classes=n_classes,
        rotations=rotations,
        translations=translations,
        class_weight_mats=class_weight_mats,
        batch_sig_mask=batch_sig_mask,
        batch_n_sig=batch_n_sig,
        hard_assignment_batch=hard_assignment_batch,
        class_assignment_batch=class_assignment_batch,
        global_log_z=global_log_z,
        class_log_z_values=class_log_z_values,
        best_score=best_score,
        max_posterior=max_posterior,
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=class_log_priors,
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=1_000_000,
        coarse_gaussian_shifted_corrected=coarse_gaussian_shifted_corrected,
        coarse_gaussian_pixel_weight=coarse_gaussian_pixel_weight,
        coarse_gaussian_initial_diff2=coarse_gaussian_initial_diff2,
        coarse_gaussian_score_indices=coarse_gaussian_score_indices,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=7,
        projection_padding_factor=1,
        debug_iteration=2,
    )
    files = sorted(os.listdir(dump_dir))
    assert files == ["significance_orig000042_it002_cs014.npz"]
    payload = np.load(dump_dir / files[0])
    for name in (
        "coarse_gaussian_shifted_corrected",
        "coarse_gaussian_pixel_weight",
        "coarse_gaussian_initial_diff2",
        "coarse_gaussian_score_indices",
        "relion_projector_half_per_class",
        "relion_projector_r_max",
        "projection_padding_factor",
    ):
        assert name in payload.files, f"Dump npz is missing schema field {name!r}"
    for retired in ("shifted_data", "ctf2_data", "window_indices", "half_weights", "projected_reference_per_class"):
        assert retired not in payload.files
    assert payload["coarse_gaussian_shifted_corrected"].dtype == np.complex64
    assert payload["relion_projector_half_per_class"].dtype == np.complex64
    assert payload["coarse_gaussian_shifted_corrected"].shape == (n_trans, n_pix)
    assert payload["coarse_gaussian_pixel_weight"].shape == (n_pix,)
    assert float(payload["coarse_gaussian_initial_diff2"]) == 3.0
    assert_matches(payload["coarse_gaussian_score_indices"], coarse_gaussian_score_indices)
    assert payload["relion_projector_half_per_class"].shape == (n_classes, 3, 4, 2)
    assert int(payload["relion_projector_r_max"]) == 7
    assert int(payload["projection_padding_factor"]) == 1
    assert int(payload["n_classes"]) == n_classes
    assert int(payload["n_rot"]) == n_rot
    assert int(payload["n_trans"]) == n_trans
    assert int(payload["debug_iteration"]) == 2
    assert int(payload["one_based_iteration"]) == 2

def test_kclass_significance_dump_iteration_gate_suppresses_other_iterations(monkeypatch, tmp_path):
    dump_dir = tmp_path / "dump"
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ITERATION", "2")

    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=None,
        indices=None,
        n_classes=1,
        rotations=None,
        translations=None,
        class_weight_mats=None,
        batch_sig_mask=None,
        batch_n_sig=None,
        hard_assignment_batch=None,
        class_assignment_batch=None,
        global_log_z=None,
        class_log_z_values=None,
        best_score=None,
        max_posterior=None,
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=None,
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=-1,
        debug_iteration=1,
    )

    assert not dump_dir.exists()

def test_kclass_significance_dump_can_stop_after_durable_target(monkeypatch, tmp_path):
    """The opt-in short-run diagnostic stops only after writing its target."""

    dump_dir = tmp_path / "dump"
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ITERATION", "2")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")

    with pytest.raises(coarse_gaussian_diagnostics.SignificanceDumpComplete) as exc_info:
        sig_mod._maybe_dump_k_class_significance_batch(
            experiment_dataset=SimpleNamespace(
                dataset_indices=np.asarray([42], dtype=np.int64),
            ),
            indices=np.asarray([0], dtype=np.int64),
            n_classes=1,
            rotations=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            translations=np.zeros((3, 2), dtype=np.float32),
            class_weight_mats=[np.full((1, 6), 1.0 / 6.0, dtype=np.float64)],
            batch_sig_mask=np.ones((1, 6), dtype=bool),
            batch_n_sig=np.asarray([6], dtype=np.int64),
            hard_assignment_batch=np.asarray([0], dtype=np.int64),
            class_assignment_batch=np.asarray([0], dtype=np.int64),
            global_log_z=np.asarray([0.0], dtype=np.float64),
            class_log_z_values=[np.asarray([0.0], dtype=np.float64)],
            best_score=np.asarray([0.0], dtype=np.float64),
            max_posterior=np.asarray([1.0], dtype=np.float64),
            rotation_log_prior_padded=None,
            batch_translation_log_prior=None,
            class_log_priors=np.zeros(1, dtype=np.float64),
            current_size=14,
            adaptive_fraction=0.999,
            max_significants=100,
            debug_iteration=2,
        )

    dump_path = dump_dir / "significance_orig000042_it002_cs014.npz"
    assert dump_path.is_file()
    assert exc_info.value.dump_path == str(dump_path)

def test_kclass_significance_stop_without_iteration_uses_unsuffixed_path(monkeypatch, tmp_path):
    """The stop gate must use the same optional suffix as the dump writer."""

    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    dump_path = dump_dir / "significance_orig000042_cs014.npz"
    dump_path.write_bytes(b"durable")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")
    monkeypatch.delenv("RELAX_SIGNIFICANCE_DUMP_ITERATION", raising=False)

    with pytest.raises(coarse_gaussian_diagnostics.SignificanceDumpComplete):
        coarse_gaussian_diagnostics._maybe_stop_after_significance_dump(
            str(dump_path),
            dump_dir=str(dump_dir),
            target_original_indices={42},
            current_size=14,
            debug_iteration=1,
        )

def test_kclass_significance_stop_respects_iteration_gate(monkeypatch, tmp_path):
    """A future target must not stop the current scoring boundary."""

    dump_dir = tmp_path / "dump"
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ITERATION", "3")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")

    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=None,
        indices=None,
        n_classes=1,
        rotations=None,
        translations=None,
        class_weight_mats=None,
        batch_sig_mask=None,
        batch_n_sig=None,
        hard_assignment_batch=None,
        class_assignment_batch=None,
        global_log_z=None,
        class_log_z_values=None,
        best_score=None,
        max_posterior=None,
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=None,
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=-1,
        debug_iteration=2,
    )

    assert not dump_dir.exists()

def test_significance_stop_waits_for_complete_target_set(monkeypatch, tmp_path):
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ITERATION", "2")
    first_path = dump_dir / "significance_orig000042_it002_cs014.npz"
    second_path = dump_dir / "significance_orig000043_it002_cs014.npz"
    first_path.touch()

    coarse_gaussian_diagnostics._maybe_stop_after_significance_dump(
        str(first_path),
        dump_dir=str(dump_dir),
        target_original_indices={42, 43},
        current_size=14,
        debug_iteration=2,
    )

    second_path.touch()
    with pytest.raises(coarse_gaussian_diagnostics.SignificanceDumpComplete):
        coarse_gaussian_diagnostics._maybe_stop_after_significance_dump(
            str(second_path),
            dump_dir=str(dump_dir),
            target_original_indices={42, 43},
            current_size=14,
            debug_iteration=2,
        )

def test_significance_dump_half_selector_is_scoped_to_target_iteration(tmp_path):
    datasets = [
        SimpleNamespace(dataset_indices=np.asarray([2, 4], dtype=np.int64)),
        SimpleNamespace(dataset_indices=np.asarray([1, 3], dtype=np.int64)),
    ]
    environ = {
        "RELAX_SIGNIFICANCE_DUMP_TARGET_HALF": "2",
        "RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET": "1",
        "RELAX_SIGNIFICANCE_DUMP_DIR": str(tmp_path),
        "RELAX_SIGNIFICANCE_DUMP_ITERATION": "2",
        "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1,3",
    }

    assert debug_dumps._significance_dump_half_indices(
        numbered_iteration=1,
        n_classes=1,
        experiment_datasets=datasets,
        environ=environ,
    ) == (0, 1)
    assert debug_dumps._significance_dump_half_indices(
        numbered_iteration=2,
        n_classes=1,
        experiment_datasets=datasets,
        environ=environ,
    ) == (1,)

def test_significance_dump_half_selector_fails_closed(tmp_path):
    datasets = [
        SimpleNamespace(dataset_indices=np.asarray([2, 4], dtype=np.int64)),
        SimpleNamespace(dataset_indices=np.asarray([1, 3], dtype=np.int64)),
    ]
    base_environ = {
        "RELAX_SIGNIFICANCE_DUMP_TARGET_HALF": "2",
        "RELAX_SIGNIFICANCE_DUMP_DIR": str(tmp_path),
        "RELAX_SIGNIFICANCE_DUMP_ITERATION": "2",
        "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1",
    }
    with pytest.raises(RuntimeError, match="STOP_AFTER_TARGET"):
        debug_dumps._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=1,
            experiment_datasets=datasets,
            environ=base_environ,
        )

    target_missing = dict(base_environ)
    target_missing["RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET"] = "1"
    target_missing["RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"] = "2"
    with pytest.raises(RuntimeError, match="not all present"):
        debug_dumps._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=1,
            experiment_datasets=datasets,
            environ=target_missing,
        )
    with pytest.raises(RuntimeError, match="K=1 diagnostic-only"):
        debug_dumps._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=4,
            experiment_datasets=datasets,
            environ={**base_environ, "RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET": "1"},
        )

def test_pass2_norm_dump_half_selector_reaches_only_target_half(tmp_path):
    datasets = [
        SimpleNamespace(dataset_indices=np.asarray([2, 4], dtype=np.int64)),
        SimpleNamespace(dataset_indices=np.asarray([1, 3], dtype=np.int64)),
    ]
    environ = {
        "RELAX_PASS2_DUMP_TARGET_HALF": "2",
        "RELAX_PASS2_DUMP_NORM_RESIDUAL_INPUTS": "1",
        "RELAX_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET": "1",
        "RELAX_PASS2_DUMP_DIR": str(tmp_path),
        "RELAX_PASS2_DUMP_ITERATION": "2",
        "RELAX_PASS2_DUMP_ORIGINAL_INDICES": "1,3",
    }

    assert debug_dumps._significance_dump_half_indices(
        numbered_iteration=1,
        n_classes=1,
        experiment_datasets=datasets,
        environ=environ,
    ) == (0, 1)
    assert debug_dumps._significance_dump_half_indices(
        numbered_iteration=2,
        n_classes=1,
        experiment_datasets=datasets,
        environ=environ,
    ) == (1,)

def test_pass2_norm_dump_half_selector_fails_closed(tmp_path):
    datasets = [
        SimpleNamespace(dataset_indices=np.asarray([2, 4], dtype=np.int64)),
        SimpleNamespace(dataset_indices=np.asarray([1, 3], dtype=np.int64)),
    ]
    base = {
        "RELAX_PASS2_DUMP_TARGET_HALF": "2",
        "RELAX_PASS2_DUMP_DIR": str(tmp_path),
        "RELAX_PASS2_DUMP_ITERATION": "2",
        "RELAX_PASS2_DUMP_ORIGINAL_INDICES": "1",
    }
    with pytest.raises(RuntimeError, match="NORM_RESIDUAL_INPUTS"):
        debug_dumps._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=1,
            experiment_datasets=datasets,
            environ=base,
        )
    with pytest.raises(RuntimeError, match="NORM_RESIDUAL_STOP_AFTER_TARGET"):
        debug_dumps._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=1,
            experiment_datasets=datasets,
            environ={**base, "RELAX_PASS2_DUMP_NORM_RESIDUAL_INPUTS": "1"},
        )
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        debug_dumps._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=1,
            experiment_datasets=datasets,
            environ={
                **base,
                "RELAX_SIGNIFICANCE_DUMP_TARGET_HALF": "2",
            },
        )

def test_relion_adaptive_fraction_preserves_text_to_float_boundary():
    expected = float(np.float32("0.999"))
    assert scoring_policy.RELION_ADAPTIVE_FRACTION == expected
    assert scoring_policy.RELION_ADAPTIVE_FRACTION != 0.999

    # This two-weight boundary is intentionally between Python's binary64
    # literal and RELION's textToFloat value.  It locks down the observed
    # one-sample support effect without depending on a bulky parity fixture.
    weights = np.asarray([[0.999000006, 0.000999994]], dtype=np.float64)
    _, binary64_count = oversampling_mod._find_significant_mask_full_sort(
        weights,
        adaptive_fraction=0.999,
        max_significants=-1,
    )
    _, relion_count = oversampling_mod._find_significant_mask_full_sort(
        weights,
        adaptive_fraction=scoring_policy.RELION_ADAPTIVE_FRACTION,
        max_significants=-1,
    )
    assert int(np.asarray(binary64_count)[0]) == 1
    assert int(np.asarray(relion_count)[0]) == 2

def test_kclass_significance_dump_uses_original_index_mapper(monkeypatch, tmp_path):
    """Subset datasets must target dumps by original RELION image id.

    Directly indexing ``dataset_indices[local_index]`` is wrong for local
    image ids in subset/pass2 debug runs. Prefer the explicit mapper when the
    dataset provides one.
    """

    n_classes = 1
    n_rot = 2
    n_trans = 3
    local_index = 7

    def original_image_indices_from_local(local_indices):
        assert matches(np.asarray(local_indices), np.asarray([local_index]))
        return np.asarray([42], dtype=np.int64)

    experiment_dataset = SimpleNamespace(
        dataset_indices=np.arange(100, 200, dtype=np.int64),
        original_image_indices_from_local=original_image_indices_from_local,
    )
    dump_dir = tmp_path / "dump"
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")

    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=experiment_dataset,
        indices=np.asarray([local_index], dtype=np.int64),
        n_classes=n_classes,
        rotations=np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1)),
        translations=np.zeros((n_trans, 2), dtype=np.float32),
        class_weight_mats=[np.ones((1, n_rot * n_trans), dtype=np.float64) / (n_rot * n_trans)],
        batch_sig_mask=np.ones((1, n_classes * n_rot * n_trans), dtype=bool),
        batch_n_sig=np.asarray([n_classes * n_rot * n_trans], dtype=np.int64),
        hard_assignment_batch=np.asarray([0], dtype=np.int64),
        class_assignment_batch=np.asarray([0], dtype=np.int64),
        global_log_z=np.asarray([0.0], dtype=np.float64),
        class_log_z_values=[np.asarray([0.0], dtype=np.float64)],
        best_score=np.asarray([0.0], dtype=np.float64),
        max_posterior=np.asarray([1.0], dtype=np.float64),
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=np.zeros(n_classes, dtype=np.float64),
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=100,
    )

    payload = np.load(dump_dir / "significance_orig000042_cs014.npz")
    assert int(payload["original_index"]) == 42
    assert int(payload["local_index"]) == local_index


# ----------------------------------------------------------------------
# Pass1 fused gate (env-var contract)
# ----------------------------------------------------------------------

def test_relion_score_window_projection_kwargs_use_image_window_not_model_window():
    """The score-window helper must hand the projector the image window size."""

    from relax.helpers.fourier_window import make_fourier_window_spec
    from relax.sparse_pass2 import sparse_pass2_projection_blocks

    spec = make_fourier_window_spec(
        (64, 64), 34, 64 * 33, reconstruction_current_size=32, square=False, include_recon_window=True
    )
    assert int(2 * spec.max_r) == 32
    kwargs = sparse_pass2_projection_blocks._projection_kwargs_for_relion_score_window(
        spec.projection_kwargs(return_abs2=False),
        use_relion_projector=True,
        current_size=34,
    )
    assert kwargs["projector_output_size"] == 34
    assert kwargs["projector_output_size"] != int(2 * spec.max_r)
