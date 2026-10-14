"""The rigid fit about a symmetry axis used by the tutorial cryo-ET benchmark row (scripts/score_axis_registered_maps.py).

The moved map is generated analytically (Gaussian blobs at rotated and shifted centres), not with the scorer's own
resampling, so the fit and the transform are checked against an independent construction."""

import numpy as np
import pytest

from scripts.fsc_metrics import normalized_fsc_auc, shell_fsc
from scripts.score_axis_registered_maps import (
    fit_about_axis,
    flip_about_x,
    score_pair,
    soft_sphere,
    transform_about_axis,
)

N = 48
ORDER = 6


def _c6_map(angle_deg=0.0, dz=0.0, flip=False):
    """Blobs with C6 symmetry about z through voxel N/2 and no mirror or twofold: three rings at different z, radii, phases."""
    z, y, x = np.indices((N,) * 3, dtype=np.float64) - N // 2
    vol = np.zeros((N,) * 3)
    for ring_z, radius, phase, sigma, weight in (
        (-6.0, 9.0, 0.0, 1.8, 1.0),
        (1.0, 13.0, 17.0, 2.2, 0.7),
        (7.0, 6.0, 41.0, 1.6, 1.3),
    ):
        for k in range(ORDER):
            a = np.deg2rad(phase + angle_deg + 60.0 * k)
            cz, cy, cx = ring_z + dz, radius * np.sin(a), radius * np.cos(a)
            if flip:
                cz, cy = 2 * dz - cz, -cy
            vol += weight * np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    return vol


WINDOW = soft_sphere(N, N / 2 - 3, 3)


@pytest.mark.parametrize("angle,dz", [(23.4, 1.3), (0.0, 0.0), (58.7, -2.6)])
def test_fit_recovers_rotation_and_shift_about_the_axis(angle, dz):
    reference = _c6_map()
    moved = _c6_map(angle_deg=angle, dz=dz)
    fit = fit_about_axis(moved, reference, symmetry_order=ORDER, mask=WINDOW, allow_flip=angle == 23.4)
    period = 360.0 / ORDER
    # the fit undoes the motion: a rotation of -angle (mod 60) in the generator's sense, and -dz
    residual = (fit["angle_deg"] - angle + period / 2) % period - period / 2
    alt = (fit["angle_deg"] + angle + period / 2) % period - period / 2
    assert min(abs(residual), abs(alt)) < 0.1
    assert fit["z_shift_px"] == pytest.approx(-dz, abs=0.05)
    assert not fit["flip"]
    fitted = transform_about_axis(moved, fit["angle_deg"], fit["z_shift_px"], fit["flip"])
    assert normalized_fsc_auc(shell_fsc(fitted * WINDOW, reference * WINDOW)) > 0.999
    assert np.abs(fitted - reference).max() < 2e-2 * reference.max()


def test_score_pair_reports_what_the_fit_gains():
    reference = _c6_map()
    moved = _c6_map(angle_deg=30.0, dz=2.0)
    score = score_pair(moved, reference, symmetry_order=ORDER, window=WINDOW, sphere=None)
    assert score["corr_as_written"] < 0.5
    assert score["fit"]["corr"] > 0.999
    assert score["fsc_auc_window"] > 0.999


def test_flip_is_found_only_when_the_map_is_flipped():
    reference = _c6_map()
    flipped = _c6_map(angle_deg=11.0, dz=0.0, flip=True)
    np.testing.assert_allclose(flip_about_x(flip_about_x(reference)), reference)
    fit = fit_about_axis(flipped, reference, symmetry_order=ORDER, mask=WINDOW)
    assert fit["flip"]
    assert fit["corr"] > 0.999
    assert fit_about_axis(flipped, reference, symmetry_order=ORDER, mask=WINDOW, allow_flip=False)["corr"] < 0.9


def test_identity_transform_changes_nothing_inside_the_window():
    reference = _c6_map()
    same = transform_about_axis(reference, 0.0, 0.0)
    assert np.abs(same - reference).max() < 1e-6 * reference.max()
