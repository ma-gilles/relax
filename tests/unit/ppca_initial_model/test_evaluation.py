"""Evaluation-only particle identity and class-coordinate checks."""

import json

import numpy as np
import pytest
from recovar.core import fourier_transform_utils as ftu

from scripts.evaluate_vdam_ppca_pilot import (
    _active_band,
    _check_effective_gt_maps,
    _class_mean_coordinates,
    _fsc_report,
    _states_from_model,
)
from scripts.plot_vdam_ppca_comparison import final_class_masses, match_classes
from scripts.prepare_vdam_ppca_fixture import effective_volumes


def test_fixture_correction_is_exact_projector_target_and_legacy_is_unchanged():
    from recovar.simulation import solvent_contrast

    n = 8
    rng = np.random.default_rng(4)
    raw = (rng.normal(size=(3, n**3)) + 1j * rng.normal(size=(3, n**3))).astype(np.complex64)
    unchanged, disabled = effective_volumes(raw, 6.0, n)
    assert unchanged is raw
    assert disabled == {"enabled": False}

    corrected, record = effective_volumes(raw, 6.0, n, atomic_solvent_correction=True)
    assert record["ground_truth_representation"] == solvent_contrast.CORRECTED_EFFECTIVE
    assert corrected.dtype == np.complex64
    np.testing.assert_array_equal(corrected, solvent_contrast.apply_record(raw, record))
    assert not np.array_equal(corrected, raw)
    assert not np.array_equal(solvent_contrast.apply_record(corrected, record), corrected)
    dc = np.ravel_multi_index((n // 2,) * 3, (n,) * 3)
    np.testing.assert_allclose(corrected[:, dc], 0.2 * raw[:, dc], rtol=2e-7)
    with pytest.raises(ValueError, match="atomic_bfactor"):
        effective_volumes(raw, 6.0, n, atomic_solvent_correction=True, atomic_bfactor=-1)


def test_evaluation_rejects_a_map_not_derived_from_projected_truth():
    n = 8
    rng = np.random.default_rng(18)
    maps = rng.normal(size=(3, n, n, n)).astype(np.float32)
    ft = np.asarray(ftu.get_dft3(maps).reshape(3, -1), np.complex64)
    exact = np.asarray(ftu.get_idft3(ft.reshape(3, n, n, n))).real.astype(np.float32)
    _check_effective_gt_maps(ft, exact)
    altered = exact.copy()
    altered[0, 1, 2, 3] += 0.01
    with pytest.raises(ValueError, match="simulated Fourier target"):
        _check_effective_gt_maps(ft, altered)


def test_k3_occupancy_comes_from_matching_final_map_metadata(tmp_path):
    meta = tmp_path / "run_it200_recovar_meta.json"
    maps = [tmp_path / f"run_it200_class{k:03d}.mrc" for k in range(1, 4)]
    meta.write_text(json.dumps({"subset_size": -1, "selected_particle_ids": [2, 0, 1],
                                "class_posterior_sums_full": [0.5, 1.5, 1.0]}))
    np.testing.assert_array_equal(final_class_masses(meta, maps, 3), [0.5, 1.5, 1.0])
    with pytest.raises(ValueError, match="belong"):
        final_class_masses(meta, maps[::-1], 3)
    meta.write_text(json.dumps({"subset_size": 2, "selected_particle_ids": [0, 1],
                                "class_posterior_sums_full": [0.5, 1.5, 0.0]}))
    with pytest.raises(ValueError, match="all-particle"):
        final_class_masses(meta, maps, 3)


def test_class_matching_uses_each_class_once_after_alignment():
    # Both first states prefer class 0 individually; the joint matching must
    # retain the distinct class-1 solution rather than duplicate class 0.
    auc = np.array([[0.99, 0.98, 0.10], [0.97, 0.20, 0.10], [0.10, 0.10, 0.95]])
    assert match_classes(auc) == {0: 1, 1: 0, 2: 2}
    with pytest.raises(ValueError, match="finite square"):
        match_classes(np.array([[0.9, np.nan], [0.2, 0.8]]))


def test_class_means_follow_original_particle_ids():
    labels = np.array([0, 1, 2, 0, 1, 2])
    original_z = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], np.float32)
    ids = np.array([4, 0, 5, 2, 1, 3])
    means = _class_mean_coordinates(ids, original_z[ids], labels)
    np.testing.assert_array_equal(means, [[4, 5], [6, 7], [8, 9]])
    with pytest.raises(ValueError, match="Embedding IDs"):
        _class_mean_coordinates(np.array([0, 1, 2, 3, 4, 4]), original_z, labels)


def test_absent_fsc_measurement_stays_missing():
    zero = np.zeros((8, 8, 8), np.float32)
    result = _fsc_report(zero, zero, 2.0, np.ones_like(zero))
    assert result["raw"]["fsc_auc"] is None
    assert all(value is None for value in result["raw"]["curve"])


def test_active_band_censors_at_declared_cutoff():
    curve = np.array([np.nan, 0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, -0.5])
    report = _active_band(curve, 64, 6.0, 8)
    assert report["shell_max"] == 8
    assert report["nominal_cutoff_ang"] == 48.0
    assert report["resolution"]["0.143"]["censored_at_cutoff"] is True
    assert report["resolution"]["0.143"]["resolution_ang"] == 48.0
    assert report["resolution"]["0.5"]["first_shell_below"] == 6
    assert report["resolution"]["0.5"]["censored_at_cutoff"] is False


def test_state_maps_are_mean_plus_class_mean_loadings():
    n = 8
    maps = np.zeros((3, n, n, n), np.float32)
    maps[0, 2, 3, 4] = 1
    maps[1, 4, 3, 2] = 0.5
    maps[2, 5, 2, 3] = -0.25
    theta = np.asarray(ftu.get_dft3_real(maps).reshape(3, -1).T, np.complex64)
    means = np.array([[1, 0], [0, 1], [-2, 0.5]], np.float32)
    states = _states_from_model(theta, means, n)
    expected = maps[0][None] + means[:, 0, None, None, None] * maps[1] + means[:, 1, None, None, None] * maps[2]
    np.testing.assert_allclose(states, expected, rtol=3e-6, atol=3e-6)
