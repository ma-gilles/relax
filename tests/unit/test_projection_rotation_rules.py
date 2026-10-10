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
        sampling.relion_adaptive_pass1_rotations(
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


# relax#57: RELION applies MBL = s inv(M3) only when !mag.isIdentity(), element-wise within
# XMIPP_EQUAL_ACCURACY = 1e-6 (matrix2d.h:1191-1206, macros.h:113-116).
STAR_1P4, HEADER_1P4 = 1.4, float(np.float32(1.4))  # s - 1 = +1.7e-8: float32 header rounding
STAR_CELL9, HEADER_CELL9 = 1.6375, float(np.float32(1.6374975))  # s - 1 = +1.5e-6: EMPIAR-10076 cell 9


@pytest.mark.parametrize(
    ("scale", "magnification", "projects"),
    [
        (1.0, None, False),
        (STAR_1P4 / HEADER_1P4, None, False),
        (1.0 + 0.9e-6, None, False),
        (1.0 - 0.9e-6, None, False),
        (1.0 + 1.1e-6, None, True),
        (STAR_CELL9 / HEADER_CELL9, None, True),
        (1.12, None, True),
        (1.0, relax_projection_magnification(MAG), True),
        (STAR_1P4 / HEADER_1P4, relax_projection_magnification(MAG), True),
    ],
)
def test_projection_follows_relions_identity_test(scale, magnification, projects):
    from relax.relion.optics_aberrations import relion_projection_optics

    applied = relion_projection_optics(scale, magnification)
    assert applied == ((scale, magnification) if projects else (1.0, None))
    rows64 = _host64(0, False)
    rows = rows64.astype(np.float32)
    projected = sampling.project_rows(rows, scale, magnification, host_rows=lambda: rows64, what="test rows")
    assert (projected is rows) != projects


def test_single_shape_runs_project_with_the_model_pixel_scale():
    """A single-shape run's s is its STAR pixel over the reference header's; cell 9's header projects, a float32
    rounded 1.4 A header does not."""
    import logging

    from relax.helpers.resolution import ImageGeometry
    from relax.refinement.image_size_plans import RunOptics
    from relax.refinement.setup_checks import projection_scale_for_run
    from relax.relion.optics_aberrations import relion_projection_optics

    def scale(star, header):
        optics = RunOptics(
            image_geometry=ImageGeometry(image_shape=(256, 256), pixel_size_angstrom=star),
            model_pixel_size=header,
            optics_image_sizes=np.array([256]),
            optics_pixel_sizes=np.array([star]),
            multi_shape_halves=False,
        )
        return projection_scale_for_run(optics, subtomograms=False)

    assert scale(STAR_CELL9, HEADER_CELL9) == pytest.approx(STAR_CELL9 / HEADER_CELL9, rel=1e-15)
    assert relion_projection_optics(scale(STAR_CELL9, HEADER_CELL9))[0] != 1.0
    assert relion_projection_optics(scale(STAR_1P4, HEADER_1P4)) == (1.0, None)
    assert scale(2.125, 2.125) == 1.0


def test_the_mstep_radius_takes_the_scale_relion_projects_with():
    """A class within RELION's identity tolerance backprojects with the plain matrices: its image radius is r_max."""
    from relax.refinement.shape_class_scoring import reconstruction_image_radius

    assert reconstruction_image_radius(28, STAR_1P4 / HEADER_1P4) == 14.0
    assert reconstruction_image_radius(28, 1.36 / HEADER_1P4) == 14.0 * (1.36 / HEADER_1P4)
    assert reconstruction_image_radius(28, STAR_CELL9 / HEADER_CELL9) == 14.0 * (STAR_CELL9 / HEADER_CELL9)
    assert reconstruction_image_radius(None, 1.12) is None


def _local_parent_layout():
    """A local search's parent layout (id rows) on a perturbed order-2 grid, five images, float32."""
    from relax.local.local_layout import build_local_hypothesis_layout

    rng = np.random.default_rng(11)
    eulers = np.column_stack([rng.uniform(0, 360, 5), rng.uniform(20, 160, 5), rng.uniform(0, 360, 5)])
    translations = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    step = sampling.relion_angular_sampling_deg(ORDER, adaptive_oversampling=0)
    return build_local_hypothesis_layout(
        eulers, None, 0.35, 0.35, ORDER, translations, np.zeros((5, 2), dtype=np.float32), 3.0, None, 1.0,
        grid_metadata=sampling.build_local_search_grid_metadata(ORDER),
        translation_prior_reference_translations=translations,
        rotation_grid_random_perturbation=PERTURBATION, rotation_grid_angular_sampling_deg=step, dtype=np.float32,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("scale, magnified", [(1.0, False), (1.0, True), (0.971, True)])
def test_local_parent_rows_are_the_projector_plans_rows(scale, magnified):
    """RELION's coarse pass scores a local search's parents with its plan's device-built matrices
    (AccProjectorPlan::setup, coarse = true), with and without a left matrix: the rows of the global pass 1 for
    the layout's own Euler rows, not the host inverse matrices the layout is built with.

    The external evidence for the rule is a RELION score dump (relax#70: with these rows 0 of 50,220 poses
    differ from RELION's score by more than 100 float32 steps, against 3.4% with the host rows); this test guards
    the rule against regression."""
    from relax.local.local_layout import (
        _rotation_eulers_from_grid_metadata,
        local_layout_device_rotations,
        local_layout_host_rotations,
    )

    layout = _local_parent_layout()
    left = relax_projection_magnification(MAG) if magnified else None
    got = local_layout_device_rotations(layout, scale, left, use_float64=False)
    assert got.dtype == np.float32 and got.shape == np.asarray(layout.rotations_flat).shape

    eulers = _rotation_eulers_from_grid_metadata(
        layout.rotation_ids_flat, sampling.build_local_search_grid_metadata(ORDER), dtype=np.float32
    )
    step = sampling.relion_angular_sampling_deg(ORDER, adaptive_oversampling=0)
    source = sampling.DevicePass1Source(np.asarray(eulers, dtype=np.float64), PERTURBATION, step, False)
    plain = np.asarray(sampling.relion_adaptive_pass1_rotations(source.source_eulers_deg, PERTURBATION, step))
    expected = plain
    if left is not None or scale != 1.0:
        expected = sampling.project_rows(plain, scale, left, device_source=source)
    assert np.array_equal(got, np.asarray(expected))

    # Independently of relax's builders: the plan kernel's statements in NumPy float32 (helpers.relion_plan_rows).
    # Not bit for bit: the device contracts multiply-adds and has its own sincosf. Measured on an H100, in units
    # of the float32 spacing at 1 (1.19e-7), of 17,280 elements: no left matrix 9,198 equal, 67 beyond one unit,
    # largest 2; magnified 5,339 equal, 236 beyond one, largest 3; magnified and scaled 5,342 equal, 318 beyond
    # one, largest 3. The bounds below are those measurements, not a tighter claim.
    from helpers.relion_plan_rows import relion_plan_rows_f32

    right = sampling.healpix_sampling.euler_angles_to_matrix(np.full((1, 3), PERTURBATION * step))[0]
    kernel_left = None if left is None and scale == 1.0 else relion_projection_left_matrix(scale, left)
    mirror = relion_plan_rows_f32(eulers, right, kernel_left)
    unit = float(np.finfo(np.float32).eps)
    plan_steps = np.abs(got.astype(np.float64) - mirror) / unit
    assert plan_steps.max() <= 3.0, plan_steps.max()
    assert (plan_steps > 1.0).mean() < 0.025, (plan_steps > 1.0).sum()

    # They are the same matrices as the host rule's to float32 rounding, and not the same bits.
    host = sampling.project_rows(
        np.asarray(layout.rotations_flat),
        scale,
        left,
        host_rows=lambda: local_layout_host_rotations(layout, mstep=False),
    )
    assert np.abs(got - np.asarray(host)).max() < 5e-6
    assert not np.array_equal(got, np.asarray(host))
    # The host rows are further from the plan kernel: 1,368 / 1,729 / 1,829 elements beyond one unit (five to
    # twenty times the plan rows' count) and a largest difference of 4.75 units in all three cases.
    host_steps = np.abs(np.asarray(host, dtype=np.float64) - mirror) / unit
    assert (host_steps > 1.0).mean() > 0.05, (host_steps > 1.0).sum()
    assert host_steps.max() > plan_steps.max()


def test_only_a_parent_layouts_id_rows_have_a_projector_plan():
    """Oversampled children are fine-pass rows (host rule): asking for their plan rows is an error."""
    import dataclasses

    from relax.local.local_layout import local_layout_device_rotations

    layout = dataclasses.replace(_local_parent_layout(), oversampled_rows=True, id_rows_source=None)
    with pytest.raises(ValueError, match="only a parent layout's id rows"):
        local_layout_device_rotations(layout, 1.0, None, use_float64=False)
