"""Conical FSC and cFAR (scripts/conical_fsc.py): the cone geometry, the axis convention, wAuC and the missing-cone
signature of a preferred orientation."""

import math

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from scripts.conical_fsc import (
    ConicalFsc,
    axes_rot_tilt_deg,
    cone_axes,
    conical_fsc,
    conical_fsc_summary,
    measured_curves,
    weighted_auc,
)
from scripts.fsc_metrics import shell_fsc


def _correlated_pair(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two maps of one signal with independent noise."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((n, n, n))
    return signal + rng.standard_normal((n, n, n)), signal + rng.standard_normal((n, n, n))


def _pair_missing_a_cone(n: int, missing_deg: float, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two maps sharing a white signal that is absent from the double cone of half-angle ``missing_deg`` about z."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    keep = np.abs(kz) < np.sqrt(kx**2 + ky**2 + kz**2) * math.cos(math.radians(missing_deg))
    signal = np.fft.ifftn(np.fft.fftn(rng.standard_normal((n, n, n))) * keep).real
    return (
        signal + noise * rng.standard_normal((n, n, n)),
        signal + noise * rng.standard_normal((n, n, n)),
    )


def _per_voxel_cfsc(a: np.ndarray, b: np.ndarray, axis: np.ndarray, half_angle_deg: float):
    """The cFSC of one axis by a loop over every Fourier voxel; returns the curve, the counts and the voxels' closest
    approach to the cone edge (in cosine)."""
    n = a.shape[0]
    n_shells = n // 2 - 1
    fa, fb = np.fft.fftn(a), np.fft.fftn(b)
    freqs = np.fft.fftfreq(n) * n
    cos_edge = math.cos(math.radians(half_angle_deg))
    cross, power_a, power_b = np.zeros(n_shells), np.zeros(n_shells), np.zeros(n_shells)
    count = np.zeros(n_shells, dtype=np.int64)
    margin = 1.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                vector = np.array([freqs[i], freqs[j], freqs[k]])
                radius = np.linalg.norm(vector)
                shell = int(np.rint(radius))
                if shell < 1 or shell >= n_shells:
                    continue
                cosine = abs(vector @ axis) / radius
                margin = min(margin, abs(cosine - cos_edge))
                if cosine >= cos_edge:
                    cross[shell] += (fa[i, j, k] * np.conj(fb[i, j, k])).real
                    power_a[shell] += abs(fa[i, j, k]) ** 2
                    power_b[shell] += abs(fb[i, j, k]) ** 2
                    count[shell] += 1
    curve = np.full(n_shells, np.nan)
    np.divide(cross, np.sqrt(power_a * power_b), out=curve, where=power_a * power_b > 0)
    return curve, count, margin


def test_cone_axes_are_unit_vectors_in_the_upper_hemisphere_spread_evenly():
    axes = cone_axes(100)
    assert axes.shape == (100, 3)
    assert_matches(np.linalg.norm(axes, axis=1), np.ones(100))
    assert np.all(axes[:, 2] > 0)
    # Even over the hemisphere: E[z] = 1/2 exactly on this lattice, and E[v v^T] = I/3 up to 0.008 of 1/3 (measured).
    assert_matches(axes[:, 2].mean(), 0.5)
    assert_matches(axes.T @ axes / 100, np.eye(3) / 3, rtol=0.03)


def test_axes_rot_tilt_deg_reads_tilt_from_z_and_rot_from_x():
    rot, tilt = axes_rot_tilt_deg(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert_matches(tilt, np.array([0.0, 90.0, 90.0]))
    assert_matches(rot[1:], np.array([0.0, 90.0]))


@pytest.mark.parametrize("half_angle_deg", [15.0, 25.0])
def test_conical_fsc_matches_a_per_voxel_loop(half_angle_deg):
    a, b = _correlated_pair(12, seed=3)
    axes = cone_axes(5)
    cfsc = conical_fsc(a, b, axes, half_angle_deg)
    for j, axis in enumerate(axes):
        curve, count, margin = _per_voxel_cfsc(a, b, axis, half_angle_deg)
        assert margin > 1e-9, "a voxel sits on the cone edge: the cases below would depend on rounding"
        assert np.array_equal(cfsc.voxels[j], count)
        assert_matches(cfsc.curves[j], curve)


def test_conical_fsc_of_a_map_with_itself_is_one_in_every_shell_the_cone_holds():
    a, _ = _correlated_pair(16, seed=1)
    cfsc = conical_fsc(a, a, cone_axes(6), 25.0)
    held = cfsc.voxels > 0
    assert held.any()
    assert_matches(cfsc.curves[held], np.ones(held.sum()))
    assert np.all(np.isnan(cfsc.curves[~held]))


def test_the_cone_of_the_opposite_axis_is_the_same_cone():
    a, b = _correlated_pair(16, seed=2)
    axis = cone_axes(3)[1]
    up = conical_fsc(a, b, axis[None], 25.0)
    down = conical_fsc(a, b, -axis[None], 25.0)
    assert np.array_equal(up.voxels, down.voxels)
    assert_matches(up.curves, down.curves)


@pytest.mark.parametrize("array_axis", [0, 1, 2])
def test_axes_are_the_x_y_z_of_the_array_axes(array_axis):
    a, b = _pair_missing_a_cone(32, missing_deg=30.0, noise=0.5, seed=7)  # signal absent about array axis 2
    permutation = {0: (2, 1, 0), 1: (0, 2, 1), 2: (0, 1, 2)}[array_axis]
    cfsc = conical_fsc(a.transpose(permutation), b.transpose(permutation), np.eye(3), 20.0)
    mean_along_x_y_z = np.nanmean(measured_curves(cfsc, min_voxels=20), axis=1)
    # Measured: 0.04 along the axis the signal is missing about, 0.76 to 0.79 along the other two.
    assert int(np.argmin(mean_along_x_y_z)) == array_axis
    assert mean_along_x_y_z[array_axis] < 0.2
    assert np.all(np.delete(mean_along_x_y_z, array_axis) > 0.5)


def test_conical_fsc_rejects_what_it_cannot_score():
    a, b = _correlated_pair(8, seed=0)
    with pytest.raises(ValueError, match="cubic maps"):
        conical_fsc(a, b[:4], cone_axes(2), 20.0)
    with pytest.raises(ValueError, match="unit vectors"):
        conical_fsc(a, b, np.array([[0.0, 0.0, 2.0]]), 20.0)
    with pytest.raises(ValueError, match="half_angle_deg"):
        conical_fsc(a, b, cone_axes(2), 90.0)


def test_weighted_auc_sums_the_surface_weighted_curve_before_the_first_crossing():
    curves = np.array(
        [
            [np.nan, 0.9, 0.8, 0.5, 0.1, 0.9],  # crosses at shell 4: shells 1-3 count
            [np.nan, 0.9, 0.8, 0.5, 0.4, 0.3],  # never crosses: every shell counts
            [np.nan, 0.9, np.nan, 0.5, 0.4, 0.3],  # a shell without a value adds nothing and cannot cross
        ]
    )
    wauc, crossing = weighted_auc(curves, 0.143)
    four_pi = 4.0 * math.pi
    assert_matches(
        wauc,
        np.array(
            [
                four_pi * (1 * 0.9 + 4 * 0.8 + 9 * 0.5),
                four_pi * (1 * 0.9 + 4 * 0.8 + 9 * 0.5 + 16 * 0.4 + 25 * 0.3),
                four_pi * (1 * 0.9 + 9 * 0.5 + 16 * 0.4 + 25 * 0.3),
            ]
        ),
    )
    assert crossing.tolist() == [4, 6, 6]


def test_measured_curves_drop_the_shells_a_cone_holds_too_few_voxels_of():
    cfsc = ConicalFsc(
        axes=np.array([[0.0, 0.0, 1.0]]),
        half_angle_deg=20.0,
        curves=np.array([[np.nan, 0.1, 0.9, 0.8]]),
        voxels=np.array([[0, 2, 120, 400]]),
    )
    assert_matches(measured_curves(cfsc, min_voxels=100), np.array([[np.nan, np.nan, 0.9, 0.8]]))


def test_summary_reads_cfar_as_the_smallest_over_the_largest_wauc():
    voxels = np.full((3, 5), 500)
    cfsc = ConicalFsc(
        axes=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        half_angle_deg=20.0,
        curves=np.array(
            [
                [np.nan, 0.9, 0.9, 0.9, 0.9],  # never crosses
                [np.nan, 0.9, 0.9, 0.05, 0.9],  # crosses at shell 3
                [np.nan, 0.9, 0.5, 0.4, 0.05],  # crosses at shell 4
            ]
        ),
        voxels=voxels,
    )
    summary = conical_fsc_summary(cfsc, voxel=2.0, box=10)
    area = {j: float(weighted_auc(cfsc.curves)[0][j]) for j in range(3)}
    assert_matches(summary["cfar"], area[1] / area[0])
    assert summary["min_axis"] == {"rot_deg": 90.0, "tilt_deg": 90.0}
    assert summary["max_axis"]["tilt_deg"] == 90.0 and summary["max_axis"]["rot_deg"] == 0.0
    assert summary["n_axes_without_crossing"] == 1
    assert_matches(summary["worst_resolution_A"], 10 * 2.0 / 3)  # the cone that crosses first, at shell 3
    # one cone never crosses, so it beats the cone that crosses last, at shell 4: no best crossing resolution
    assert summary["best_resolution_A"] is None
    assert_matches(summary["last_shell_resolution_A"], 10 * 2.0 / 4)  # shells 0 to 4: the last is shell 4
    assert summary["first_shell_measured_by_all_cones"] == 1


def test_summary_skips_shells_too_sparse_to_measure_instead_of_crossing_there():
    curves = np.array([[np.nan, 0.0, 0.9, 0.9, 0.9], [np.nan, 0.9, 0.9, 0.9, 0.9]])
    cfsc = ConicalFsc(
        axes=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        half_angle_deg=20.0,
        curves=curves,
        voxels=np.array([[0, 2, 300, 300, 300], [0, 300, 300, 300, 300]]),
    )
    summary = conical_fsc_summary(cfsc, voxel=1.0, box=10, min_voxels=100)
    # The two-voxel shell 1 of the first cone would cross at 0.0; unmeasured, the cones differ by that shell's area.
    weights = 4.0 * math.pi * np.arange(5) ** 2
    assert_matches(summary["cfar"], weights[2:] @ np.full(3, 0.9) / (weights[1:] @ np.full(4, 0.9)))
    assert summary["n_axes_without_crossing"] == 2
    assert summary["worst_resolution_A"] is None and summary["best_resolution_A"] is None
    assert summary["first_shell_measured_by_all_cones"] == 2


def test_summary_gives_the_best_and_worst_crossing_resolution_when_every_cone_crosses():
    cfsc = ConicalFsc(
        axes=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        half_angle_deg=20.0,
        curves=np.array([[np.nan, 0.9, 0.9, 0.05, 0.0], [np.nan, 0.9, 0.05, 0.0, 0.0]]),  # cross at shells 3 and 2
        voxels=np.full((2, 5), 500),
    )
    summary = conical_fsc_summary(cfsc, voxel=1.0, box=12)
    assert_matches(summary["worst_resolution_A"], 12 / 2)
    assert_matches(summary["best_resolution_A"], 12 / 3)
    assert summary["n_axes_without_crossing"] == 0


def test_summary_refuses_a_pair_with_no_signal_in_any_cone():
    cfsc = ConicalFsc(
        axes=np.array([[0.0, 0.0, 1.0]]),
        half_angle_deg=20.0,
        curves=np.array([[np.nan, 0.01, 0.0, 0.0]]),
        voxels=np.full((1, 4), 500),
    )
    with pytest.raises(ValueError, match="no cone has a positive wAuC"):
        conical_fsc_summary(cfsc, voxel=1.0, box=8)


def test_a_missing_cone_about_z_gives_a_low_cfar_with_its_minimum_at_the_pole():
    a, b = _pair_missing_a_cone(48, missing_deg=30.0, noise=0.5, seed=5)
    summary = conical_fsc_summary(conical_fsc(a, b, cone_axes(60), 20.0), voxel=1.0, box=48)
    assert summary["cfar"] < 0.1  # measured: 0, the cones inside the missing cone have no correlated shell
    assert summary["min_axis"]["tilt_deg"] < 30.0
    assert summary["max_axis"]["tilt_deg"] > 50.0


def test_an_isotropic_pair_has_a_cfar_near_one():
    a, b = _correlated_pair(48, seed=6)
    cfsc = conical_fsc(a, b, cone_axes(60), 20.0)
    summary = conical_fsc_summary(cfsc, voxel=1.0, box=48)
    assert summary["cfar"] > 0.8  # measured: 0.87, the spread of 60 cones' areas from their sampling noise
    assert summary["n_axes_without_crossing"] == 60
    # The cones of an isotropic pair average to the global FSC (measured: within 0.008 of it) in the measured shells.
    mean_curve = np.nanmean(measured_curves(cfsc)[:, 12:22], axis=0)
    assert_matches(mean_curve, shell_fsc(a, b)[12:22], rtol=0.03)
