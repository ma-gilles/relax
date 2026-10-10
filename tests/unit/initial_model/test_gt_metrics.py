import numpy as np
import pytest

from relax.diagnostics.gt_metrics import (
    DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    DEFAULT_GT_ALIGN_MAX_SHELL,
    align_volume_to_reference,
    alignment_score_box_size,
    centered_correlation,
    first_shell_below_threshold,
    lowpass_volume_by_shell,
    relion_alignment_rotations,
    rotate_volume_about_center,
)


def _asymmetric_volume(n=17):
    z, y, x = np.indices((n, n, n), dtype=np.float64)

    def gaussian(cz, cy, cx, sz, sy, sx, amplitude):
        return amplitude * np.exp(-(((z - cz) / sz) ** 2 + ((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2))

    return (
        gaussian(5, 7, 11, 1.7, 2.5, 1.2, 1.0)
        + gaussian(12, 4, 6, 2.2, 1.1, 2.8, 0.7)
        - gaussian(8, 12, 4, 1.5, 2.0, 1.4, 0.3)
    )


@pytest.mark.unit
def test_align_volume_to_reference_recovers_grid_rotation():
    gt = _asymmetric_volume()
    rot_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated = rotate_volume_about_center(gt, rot_z_90)
    assert centered_correlation(rotated, gt) < 0.2

    rotations = np.stack([np.eye(3), rot_z_90.T, rot_z_90])
    alignment = align_volume_to_reference(rotated, gt, rotations, allow_mirror=False)

    assert alignment.rotation_index == 1
    assert not alignment.mirror_x
    assert alignment.sign == 1
    assert alignment.corr == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_align_volume_to_reference_handles_handedness_when_enabled():
    gt = _asymmetric_volume()
    mirrored = gt[::-1, :, :]

    alignment = align_volume_to_reference(mirrored, gt, np.stack([np.eye(3)]), allow_mirror=True)

    assert alignment.mirror_x
    assert alignment.sign == 1
    assert alignment.corr == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_align_volume_to_reference_does_not_hide_sign_by_default():
    gt = _asymmetric_volume()

    no_sign = align_volume_to_reference(-gt, gt, np.stack([np.eye(3)]), allow_mirror=False)
    with_sign = align_volume_to_reference(
        -gt,
        gt,
        np.stack([np.eye(3)]),
        allow_mirror=False,
        allow_sign=True,
    )

    assert no_sign.sign == 1
    assert no_sign.corr == pytest.approx(-1.0, abs=1e-12)
    assert with_sign.sign == -1
    assert with_sign.corr == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_first_shell_below_threshold_uses_minus_one_sentinel():
    assert first_shell_below_threshold(np.array([1.0, 0.8, 0.2]), 0.5) == 2
    assert first_shell_below_threshold(np.array([1.0, 0.8, 0.7]), 0.5) == -1


@pytest.mark.unit
def test_alignment_score_lowpass_can_crop_to_shell_box():
    gt = _asymmetric_volume(n=33)

    score_box_size = alignment_score_box_size(gt.shape[0], 4)
    low = lowpass_volume_by_shell(gt, 4, output_size=score_box_size)

    assert score_box_size == 9
    assert low.shape == (9, 9, 9)
    assert centered_correlation(low, low) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_default_alignment_grid_is_one_order_finer_than_initialmodel_grid():
    assert DEFAULT_GT_ALIGN_HEALPIX_ORDER == 2
    assert DEFAULT_GT_ALIGN_MAX_SHELL == 8
    assert relion_alignment_rotations(DEFAULT_GT_ALIGN_HEALPIX_ORDER).shape == (4608, 3, 3)


@pytest.mark.unit
def test_order_two_alignment_improves_over_too_coarse_grid():
    gt = _asymmetric_volume(n=17)
    order0_rotations = relion_alignment_rotations(0)
    order2_rotations = relion_alignment_rotations(2)
    true_rotation_index = 1838
    rotated = rotate_volume_about_center(gt, order2_rotations[true_rotation_index].T)

    coarse_alignment = align_volume_to_reference(
        rotated,
        gt,
        order0_rotations,
        score_max_shell=5,
        allow_mirror=False,
    )
    order2_alignment = align_volume_to_reference(
        rotated,
        gt,
        order2_rotations,
        score_max_shell=5,
        allow_mirror=False,
    )

    assert order2_alignment.rotation_index == true_rotation_index
    assert order2_alignment.corr > 0.98
    assert order2_alignment.corr > coarse_alignment.corr + 0.35


def _icosahedral_orbit_volume(n):
    """A smooth map that is exactly RELION-I2 symmetric about the origin voxel n // 2."""
    from relax.sampling.symmetry import rotational_operators

    operators = rotational_operators("I2")
    z, y, x = np.indices((n, n, n), dtype=np.float64) - n // 2
    seeds = (np.array([0.20, 0.08, 0.03]), np.array([-0.05, 0.14, 0.17]))
    volume = np.zeros((n, n, n))
    for seed, amplitude in zip(seeds, (1.0, 0.6)):
        for operator in operators:
            cz, cy, cx = operator @ (seed * n)
            volume += amplitude * np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / (2 * 2.0**2))
    return volume, operators


@pytest.mark.unit
@pytest.mark.parametrize("n", (32, 33))
def test_rotation_is_about_the_origin_voxel_so_symmetric_maps_are_invariant(n):
    """RELION symmetrises about the DFT origin N // 2 (N/2 for even boxes); an exactly I2-symmetric map must be
    invariant under every I2 operator. Rotating an even box about (N-1)/2 shifted it by half a voxel per axis."""
    volume, operators = _icosahedral_orbit_volume(n)
    for operator in operators[1:]:
        rotated = rotate_volume_about_center(volume, operator, order=3)
        assert np.linalg.norm(rotated - volume) / np.linalg.norm(volume) < 1e-3


@pytest.mark.unit
def test_odd_box_rotation_is_unchanged():
    """For odd boxes N // 2 equals the geometric centre (N-1)/2, so the rotation is the same as before."""
    from scipy import ndimage

    volume = _asymmetric_volume(17)
    rotation = relion_alignment_rotations(1)[7]
    center = (np.asarray(volume.shape, dtype=np.float64) - 1.0) * 0.5
    expected = ndimage.affine_transform(
        volume, rotation.T, offset=center - rotation.T @ center, output_shape=volume.shape, order=1, mode="constant"
    )
    np.testing.assert_array_equal(rotate_volume_about_center(volume, rotation, order=1), expected)
