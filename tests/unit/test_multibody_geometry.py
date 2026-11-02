"""Multi-body geometry, body prior and subtraction conventions against RELION (relax.multibody)."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

import relax.multibody.layout as body_layout
from relax.cuda.rotations import relion_make_scoring_rotations_left_f32
from relax.local_search.layout import (
    LocalHypothesisLayout,
    _rotation_eulers_from_grid_metadata,
    relion_mstep_rotations_from_eulers,
)
from relax.multibody import geometry
from relax.multibody.bodies import A_ROT90, RESIDUAL_IDENTITY_EULER, Body, align_with_z, relion_center_of_mass
from relax.multibody.layout import body_prior_log, with_body_rows
from relax.multibody.subtraction import relion_half_to_real, relion_translate_2d, translate_zero_fill
from relax.sampling.device_rotations import pass1_right_matrix
from relax.sampling.grids import build_local_search_grid_metadata
from relax.sampling.healpix import euler_angles_to_matrix
from relax.sampling.perturbation import advance_relion_perturbation_from_seed

pytestmark = pytest.mark.unit


def _body(com=(0.0, 0.0, 0.0), orientation=None, sigma=10.0):
    return Body(
        mask=np.zeros((8, 8, 8)), com=np.asarray(com, dtype=np.float64),
        orientation=np.eye(3) if orientation is None else orientation, max_radius_angstrom=10.0,
        sigma_tilt_deg=sigma, sigma_psi_deg=sigma, sigma_offset_angstrom=2.0, keep_fixed=False,
    )


def test_identity_residual_composes_to_the_consensus_pose():
    """RELION's unmoved body is the residual (0, 90, 0): A O^T A_rot90 R(0,90,0) O = A for any O."""

    assert_matches(A_ROT90 @ euler_angles_to_matrix(np.asarray([RESIDUAL_IDENTITY_EULER]))[0], np.eye(3))
    consensus = euler_angles_to_matrix(np.array([[30.0, 60.0, -20.0], [-150.0, 10.0, 75.0]]))
    orientation = align_with_z(np.array([0.2, -0.7, 0.4]))
    body = _body(orientation=orientation)
    composed = geometry.body_matrices(consensus, body, np.tile(RESIDUAL_IDENTITY_EULER, (2, 1)))
    assert_matches(composed, consensus)


def test_body_rotation_is_not_euler_addition():
    """A residual away from (0, 90, 0) rotates the body about its own frame, not by adding Euler angles."""

    consensus = euler_angles_to_matrix(np.array([[30.0, 60.0, -20.0]]))
    body = _body(orientation=align_with_z(np.array([0.2, -0.7, 0.4])))
    residual = np.array([[5.0, 95.0, 3.0]])
    composed = geometry.body_matrices(consensus, body, residual)[0]
    added = euler_angles_to_matrix(np.array([[35.0, 155.0, -17.0]]))[0]
    assert np.abs(composed - added).max() > 0.1
    assert_matches(composed @ composed.T, np.eye(3))
    relative = consensus[0].T @ composed  # the body's rotation in the particle frame
    angle = np.rad2deg(np.arccos((np.trace(relative) - 1.0) / 2.0))
    assert 1.0 < angle < 10.0


def test_align_with_z_takes_the_axis_to_z():
    for axis in (np.array([0.3, -0.2, 0.9]), np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])):
        matrix = align_with_z(axis)
        assert_matches(matrix @ (axis / np.linalg.norm(axis)), [0.0, 0.0, 1.0])
        assert_matches(matrix @ matrix.T, np.eye(3))


def test_center_of_mass_uses_positive_voxels_and_xmipp_coordinates():
    volume = np.zeros((8, 8, 8))
    volume[5, 2, 6] = 1.0  # (z, y, x) logical (1, -2, 2)
    volume[1, 1, 1] = -3.0  # negative voxels do not count
    assert_matches(relion_center_of_mass(volume), [2.0, -2.0, 1.0])


def test_shifts_follow_relion_formulas():
    consensus = euler_angles_to_matrix(np.array([[0.0, 0.0, 0.0], [90.0, 0.0, 0.0]]))
    body = _body(com=(6.0, -6.0, -1.0))
    projected = geometry.projected_centers(consensus, body)
    assert_matches(projected[0], [6.0, -6.0])
    offsets = np.array([[0.4, 1.5], [-2.5, 0.0]])
    # ROUND halves away from zero (macros.h).
    integer = geometry.integer_image_shifts(offsets, projected)
    # 7.5 and 3.5 round away from zero, as RELION's ROUND (macros.h).
    np.testing.assert_array_equal(integer, [[-6.0, 8.0], [4.0, 6.0]])
    other = geometry.other_body_shifts(np.zeros((2, 2)), offsets, np.array([[0.25, 0.0], [0.0, 0.5]]), integer)
    assert_matches(other, -offsets - np.array([[0.25, 0.0], [0.0, 0.5]]) + integer)


def test_body_prior_cuts_at_three_sigma_from_the_unmoved_pose():
    body = _body(sigma=5.0)
    eulers = np.array([[0.0, 90.0, 0.0], [0.0, 90.0, 14.0], [0.0, 90.0, 16.0], [0.0, 74.0, 0.0], [10.0, 90.0, 0.0]])
    log_prior = body_prior_log(eulers, body)
    assert np.isfinite(log_prior[:2]).all() and np.isneginf(log_prior[2])
    assert np.isneginf(log_prior[3])  # 16 deg from the (0, 90) direction
    assert np.isfinite(log_prior[4]) and log_prior[4] < log_prior[0]
    assert_matches(log_prior[1] - log_prior[0], -0.5 * (14.0 / 5.0) ** 2)


def test_translate_matches_relion_translate2d_zero_fill():
    image = np.arange(16.0).reshape(4, 4)
    moved = relion_translate_2d(image, (1, -1))  # pixel (y, x) -> (y - 1, x + 1)
    assert moved[0, 1] == image[1, 0] and moved[3, 0] == 0.0 and moved[0, 0] == 0.0
    np.testing.assert_array_equal(translate_zero_fill(moved, (1, -1)) [1:, :3], image[1:, :3])


def test_relion_units_real_image_round_trip():
    """The real image of a RELION-normalized centred half spectrum transforms back to it (FFT / N^2)."""

    rng = np.random.default_rng(1)
    image = rng.normal(size=(1, 8, 8))
    half = np.fft.fftshift(np.fft.rfft2(np.fft.fftshift(image, axes=(-2, -1))), axes=(-2,)) / 64.0
    assert_matches(np.asarray(relion_half_to_real(jnp.asarray(half.reshape(1, -1)), (8, 8))), image)


def _relion_star_float(value: float) -> str:
    """How RELION writes a float in a STAR file: %12.6f, or %12.5f when negative (metadata_table.cpp)."""
    return f"{value:.6f}" if value >= 0 else f"{value:.5f}"


def test_multibody_start_perturbation_matches_relion():
    """RELION 5.0.1 run (job 16532387): consensus it010 perturbation 0.220454, multi-body it000..it004 and final."""

    value = advance_relion_perturbation_from_seed(0.220454, 0.5, seed=1)
    assert _relion_star_float(value) == "-0.31950"
    for iteration, expected in enumerate(("0.140548", "-0.43421", "-0.04386", "0.435251", "-0.24606"), start=1):
        value = advance_relion_perturbation_from_seed(value, 0.5, seed=iteration)
        assert _relion_star_float(value) == expected


def test_later_bodies_see_earlier_bodies_new_poses():
    """RELION writes body b's new pose before it prepares body b + 1 of the same particle; a frozen snapshot of the
    previous iteration's poses would be a different iteration."""

    from relax.multibody.estep import BodyScore
    from relax.multibody.refine import score_bodies
    from relax.multibody.subtraction import BodyPoses

    class Half:
        def __init__(self, index, rows):
            self.index, self.rows = index, np.asarray(rows)

    n = 6
    halves = [Half(0, [0, 2, 4]), Half(1, [1, 3, 5])]
    start = [BodyPoses(np.tile([0.0, 90.0, 0.0], (n, 1)), np.zeros((n, 2))) for _ in range(3)]
    seen = []

    def score(b, half, half_poses):
        seen.append((b, half.index, [p.eulers_deg[:, 0].copy() for p in half_poses]))
        new = BodyPoses(half_poses[b].eulers_deg + [10.0 * (b + 1), 0.0, 0.0], half_poses[b].offsets_px + b + 1)
        return BodyScore(result=("result", b, half.index), poses=new)

    results, poses = score_bodies(halves, start, [0, 2], score)
    calls = {(b, h): rot for b, h, rot in seen}
    assert [(b, h) for b, h, _ in seen] == [(0, 0), (2, 0), (0, 1), (2, 1)]
    for h in (0, 1):
        np.testing.assert_array_equal(calls[(0, h)][0], 0.0)  # body 1 sees its previous pose
        np.testing.assert_array_equal(calls[(2, h)][0], 10.0)  # body 3 sees body 1's new pose
        np.testing.assert_array_equal(calls[(2, h)][1], 0.0)  # the fixed body 2 keeps its pose
    np.testing.assert_array_equal(poses[0].eulers_deg[:, 0], 10.0)
    np.testing.assert_array_equal(poses[2].eulers_deg[:, 0], 30.0)
    np.testing.assert_array_equal(poses[2].offsets_px, 3.0)
    assert poses[1] is start[1]
    assert results == {0: [("result", 0, 0), ("result", 0, 1)], 2: [("result", 2, 0), ("result", 2, 1)]}


def test_subtraction_ctf_is_relion_getctf():
    """RELION subtracts CTF-modulated projections with its own CTF, -sin(gamma) = +Q0 at the origin (ctf.h); relax's
    exact rows in the recovar frame carry the opposite sign (relax.relion.ctf._exact_ctf_rows), so
    ``other_body_prediction`` negates them."""

    from relax.relion.ctf import relion_ctf_fftw_half

    row = relion_ctf_fftw_half(np.array([[20000.0, 20000.0, 0.0, 300.0, 2.7, 0.07, 0.0, 1.0, 0.0]]), 16, 4.25)
    assert row[0, 0, 0] == pytest.approx(0.07, abs=1e-9)



def _mask_body(mask):
    return Body(mask=mask, com=np.zeros(3), orientation=np.eye(3), max_radius_angstrom=1.0, sigma_tilt_deg=10.0,
                sigma_psi_deg=10.0, sigma_offset_angstrom=4.0, keep_fixed=False)


def test_subtraction_masks_follow_relion_pointer_body_overlap():
    """RELION's default overlap rule (ml_model.cpp:1400-1456): body o is subtracted from body i's images with
    ``m_o - m_o * (m_i + masks already chosen for the bodies before o)``; a disjoint pair keeps the body's mask."""

    from relax.multibody.bodies import subtraction_masks

    a = np.zeros((4, 4, 4))
    a[:, :, :2] = 1.0
    b = np.zeros((4, 4, 4))
    b[:, :, 1:3] = 0.5
    c = np.zeros((4, 4, 4))
    c[:, :, 3] = 1.0
    bodies = [_mask_body(m) for m in (a, b, c)]
    masks = subtraction_masks(bodies)
    assert masks[0][2] is c and masks[2][0] is a
    assert_matches(masks[0][1], b - b * a)
    assert_matches(masks[2][1], b - b * (c + a))  # body 0's mask was chosen before body 1 for target 2
    assert_matches(masks[1][0], a - a * b)
    normalised = subtraction_masks(bodies, norm_overlap=True)
    assert all(normalised[i][o] is bodies[o].mask for i in range(3) for o in range(3) if i != o)


def _body_layout(**fields):
    return LocalHypothesisLayout(
        n_global_rotations=0, n_pixels=0, n_psi=0, rotation_offsets=np.array([0, 2, 5]),
        rotation_ids_flat=np.array([0, 7, 7, 3, 0]), rotations_flat=np.zeros((5, 3, 3), np.float32),
        rotation_log_priors_flat=np.zeros(5), rotation_counts=np.array([2, 3]),
        translation_grid=np.zeros((1, 2)), translation_log_priors=np.zeros((2, 1)), **fields,
    )


def _body_left_right():
    left = euler_angles_to_matrix(np.array([[11.0, 37.0, -5.0], [130.0, 62.0, 44.0]]))
    return left, euler_angles_to_matrix(np.array([[23.0, 71.0, 9.0]]))[0]


def test_host_body_rows_are_composed_from_float64_and_cast_once():
    """RELION's fine and M-step rows are ``L A R`` of double matrices (generateEulerMatrices), not of cast ones."""

    eulers = np.array([[1.25, 80.5, 3.75], [2.5, 91.0, -1.5], [3.0, 89.5, 0.25], [4.5, 90.5, 2.0], [5.5, 92.0, 1.0]])
    host = relion_mstep_rotations_from_eulers(eulers, dtype=np.float64)
    layout = _body_layout(oversampled_rows=True, source_eulers_flat=eulers, mstep_rotations_flat=host.astype(np.float32))
    layout = replace(layout, rotations_flat=host.astype(np.float32))
    left, right = _body_left_right()
    rows = with_body_rows(layout, left, right)
    expected = (np.repeat(left, [2, 3], axis=0) @ host @ right).astype(np.float32)
    assert rows.rotations_flat.dtype == np.float32
    np.testing.assert_array_equal(rows.rotations_flat, expected)
    np.testing.assert_array_equal(rows.mstep_rotations_flat, expected)


def test_coarse_body_rows_are_the_device_plan_of_each_image(monkeypatch):
    """A body's pass-1 parents take ``L = MBL`` and ``R = perturbation MBR`` to make_eulers_3D, each image its own L."""

    seen = {}

    def fake_kernel(eulers, right, lefts):
        seen["right"] = np.asarray(right)
        eulers, lefts = np.asarray(eulers), np.asarray(lefts)
        return np.einsum("bij,n->bnij", lefts, eulers[:, 0]) + eulers[:, 1][None, :, None, None]

    monkeypatch.setattr(body_layout.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(body_layout, "relion_make_scoring_rotations_left_f32", fake_kernel)
    monkeypatch.setattr(body_layout, "_DEVICE_ROWS_BLOCK_IMAGES", 1)
    layout = _body_layout(id_rows_source=(1, 0.25, 15.0), mstep_rotations_flat=None)
    left, right = _body_left_right()
    rows = with_body_rows(layout, left, right, device_plan=True).rotations_flat
    metadata = build_local_search_grid_metadata(1, symmetry="C1")
    eulers = _rotation_eulers_from_grid_metadata(np.array([0, 7, 7, 3, 0]), metadata, dtype=np.float32)
    image = np.array([0, 0, 1, 1, 1])
    expected = np.einsum("nij,n->nij", left[image].astype(np.float32), eulers[:, 0]) + eulers[:, 1, None, None]
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_array_equal(
        seen["right"],
        (pass1_right_matrix(0.25, 15.0) @ right).astype(np.float32),
    )


@pytest.mark.gpu
def test_coarse_body_rows_match_the_kernel_image_by_image():
    if jax.default_backend() != "gpu":
        pytest.skip("needs the CUDA kernel")
    layout = _body_layout(id_rows_source=(1, 0.25, 15.0), mstep_rotations_flat=None)
    left, right = _body_left_right()
    rows = with_body_rows(layout, left, right, device_plan=True).rotations_flat
    metadata = build_local_search_grid_metadata(1, symmetry="C1")
    eulers = _rotation_eulers_from_grid_metadata(np.array([0, 7, 7, 3, 0]), metadata, dtype=np.float32)
    plan_right = jnp.asarray(pass1_right_matrix(0.25, 15.0) @ right, dtype=jnp.float32)
    for image, (start, stop) in enumerate(((0, 2), (2, 5))):
        built = relion_make_scoring_rotations_left_f32(
            jnp.asarray(eulers[start:stop]), plan_right, jnp.asarray(left[image : image + 1], dtype=jnp.float32)
        )
        np.testing.assert_array_equal(rows[start:stop], np.asarray(built)[0])


def test_accuracy_trials_keep_relions_one_shuffle_and_seed_zero_order(monkeypatch):
    from relax.multibody.refine import _accuracy_trial_order
    from relax.sampling import expected_accuracy

    subsets = np.array([1, 2, 1, 1, 2, 1])
    rows = np.flatnonzero(subsets == 1)
    np.testing.assert_array_equal(_accuracy_trial_order(rows, subsets, 0), np.arange(4))
    calls = []

    def fake_orders(random_subsets, seed, first_iteration):
        calls.append((seed, first_iteration))
        return np.array([5, 0, 3, 2]), np.array([1, 4])

    monkeypatch.setattr(expected_accuracy, "relion_auto_refine_half_orders", fake_orders)
    np.testing.assert_array_equal(_accuracy_trial_order(rows, subsets, 7), [3, 0, 2, 1])
    assert calls == [(7, 1)]
