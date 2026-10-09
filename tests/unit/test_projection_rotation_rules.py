"""Projection matrices of images on another grid or magnified follow RELION's per-path rules (relax#24).

RELION builds pass-1 rows on the device: ``make_eulers_3D`` forms ``MBL (A R)`` in float32 with
``MBL = s inv(M3)`` and inverts it with the float32 adjugate (acc_projector_plan_impl.h:151-330, helper.cuh).
The fine pass, the weighted sums and the M-step use host rows: ``generateEulerMatrices`` composes
``L A R`` and its inverse in double and casts to float once (acc_helper_functions_impl.h). Composing the
magnification after the float32 cast moves most magnified matrices by an ulp and flips near-tie poses.
These are cast rules, so the checks are bit identities.
"""

import numpy as np
import pytest

from relax import sampling
from relax.helpers.oversampling import build_adaptive_pass2_grids, project_pass2_rotations
from relax.relion.optics_aberrations import relax_projection_magnification, relion_projection_left_matrix

pytestmark = pytest.mark.unit

MAG = np.array([[1.015, 0.004], [0.004, 0.990]])
ORDER = 2
PERTURBATION = 0.37


def _grids(oversampling):
    rotation_grid = sampling.relion_scoring_rotation_grid(ORDER, dtype=np.float32)
    step = sampling.relion_angular_sampling_deg(ORDER, adaptive_oversampling=0)
    coarse, _eulers = sampling.apply_relion_rotation_perturbation_to_eulers(
        np.asarray(rotation_grid.rotation_eulers, dtype=np.float64), PERTURBATION, step
    )
    translations = np.zeros((1, 2), dtype=np.float32)
    return build_adaptive_pass2_grids(
        coarse, translations, translations, ORDER, oversampling, 1.0, PERTURBATION, return_mstep_rotations=True
    )


def _host64(oversampling, mstep):
    outputs = sampling.get_oversampled_rotation_grid_from_samples(
        np.arange(sampling.rotation_grid_size(ORDER), dtype=np.int64),
        ORDER,
        oversampling_order=oversampling,
        random_perturbation=PERTURBATION,
        return_mstep_rotations=True,
        dtype=np.float64,
    )
    return outputs[2] if mstep else outputs[0]


def _round32(left, rows64, scale):
    return (np.einsum("ij,njk->nik", left, rows64) / scale).astype(np.float32)


@pytest.mark.parametrize("scale", [1.0, 0.971])
def test_host_rows_compose_in_float64_and_cast_once(scale):
    left = relax_projection_magnification(MAG)
    coarse, _ct, fine, _ft, _rp, _tp, mstep = _grids(1)
    projected = project_pass2_rotations(
        coarse,
        fine,
        mstep,
        scale=scale,
        magnification=left,
        coarse_healpix_order=ORDER,
        adaptive_oversampling=1,
        random_perturbation=PERTURBATION,
    )
    expected = (
        _round32(left, _host64(0, False), scale),
        _round32(left, _host64(1, False), scale),
        _round32(left, _host64(1, True), scale),
    )
    for got, want in zip(projected, expected, strict=True):
        assert got.dtype == np.float32
        assert np.array_equal(got, want)
    # The rule matters: composing after the float32 cast changes most rows.
    after_cast = (np.einsum("ij,njk->nik", left, fine) / scale).astype(np.float32)
    assert np.count_nonzero(np.any(after_cast != projected[1], axis=(1, 2))) > fine.shape[0] // 2


def test_unoversampled_fine_and_mstep_rows_are_the_grids_coarse_rows():
    left = relax_projection_magnification(MAG)
    coarse, _ct, fine, _ft, _rp, _tp, mstep = _grids(0)
    projected = project_pass2_rotations(
        coarse,
        fine,
        mstep,
        scale=1.0,
        magnification=left,
        coarse_healpix_order=ORDER,
        adaptive_oversampling=0,
        random_perturbation=PERTURBATION,
    )
    for got in projected:
        assert np.array_equal(got, _round32(left, _host64(0, False), 1.0))


def test_unprojected_images_keep_their_rows():
    coarse, _ct, fine, _ft, _rp, _tp, mstep = _grids(1)
    projected = project_pass2_rotations(
        coarse,
        fine,
        mstep,
        scale=1.0,
        magnification=None,
        coarse_healpix_order=ORDER,
        adaptive_oversampling=1,
        random_perturbation=PERTURBATION,
    )
    assert all(got is given for got, given in zip(projected, (coarse, fine, mstep), strict=True))


def test_rows_from_another_source_are_refused():
    coarse, _ct, fine, _ft, _rp, _tp, mstep = _grids(1)
    with pytest.raises(sampling.RotationProvenanceError, match="fine rows"):
        project_pass2_rotations(
            coarse,
            fine[::-1].copy(),
            mstep,
            scale=1.0,
            magnification=relax_projection_magnification(MAG),
            coarse_healpix_order=ORDER,
            adaptive_oversampling=1,
            random_perturbation=PERTURBATION,
        )


def test_left_matrix_is_relions_mbl():
    m3 = np.eye(3)
    m3[:2, :2] = MAG
    assert np.array_equal(
        relion_projection_left_matrix(0.971, relax_projection_magnification(MAG)), 0.971 * np.linalg.inv(m3)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [1.0, 0.971])
def test_device_rows_are_relions_float32_left_kernel(scale):
    """Pass-1 rows: RELION's float32 ``MBL (A R)`` and adjugate, the kernel tilt images' ``MBL`` use too."""

    rotation_grid = sampling.relion_scoring_rotation_grid(ORDER, dtype=np.float32)
    source = sampling.DevicePass1Source(
        np.asarray(rotation_grid.rotation_eulers, dtype=np.float64),
        PERTURBATION,
        sampling.relion_angular_sampling_deg(ORDER, adaptive_oversampling=0),
        False,
    )
    plain = np.asarray(
        sampling._relion_adaptive_pass1_rotations(
            source.source_eulers_deg, source.random_perturbation, source.angular_sampling_deg
        )
    )
    left = relax_projection_magnification(MAG)
    got = sampling.project_rows(plain, scale, left, device_source=source)
    kernel = np.asarray(
        sampling._relion_device_scoring_rotations_left_f32(
            source.source_eulers_deg,
            sampling.healpix_sampling.euler_angles_to_matrix(
                np.full((1, 3), PERTURBATION * source.angular_sampling_deg)
            )[0],
            relion_projection_left_matrix(scale, left)[None],
        )
    )[0]
    assert got.dtype == np.float32
    assert np.array_equal(got, kernel)
    after_cast = (np.einsum("ij,njk->nik", left, plain.astype(np.float64)) / scale).astype(np.float32)
    assert not np.array_equal(got, after_cast)
