"""Host contract for low-cardinality exact VDAM Fourier-window shapes."""

from pathlib import Path

import numpy as np
import pytest
from helpers.array_padding import pad_axis
from helpers.cuda_source import read_em_cuda_source
from helpers.float_compare import assert_matches, matches

from relax.fine_pass.scoring import _relion_cuda_fine_full_to_compact_lookup
from relax.fine_pass.wavg import _make_relion_wavg_rectangle, _make_stable_relion_wavg_rectangle
from relax.fourier.fourier_window import (
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
    make_stable_fourier_window_shape_plan,
    stable_fourier_window_current_size,
)
from relax.projection.projection import _texture_centered_crop_at_indices
from relax.reconstruction.half_volume_mstep import crop_relion_x_half_accumulator
from relax.scoring.coarse_layout import coarse_gaussian_fused_logical_lookup, plan_coarse_gaussian_square_layout

pytestmark = pytest.mark.unit

_IMAGE_SHAPE = (128, 128)
_N_HALF = 128 * 65
_GF46_CURRENT_SIZES_THROUGH_80 = (
    30,
    32,
    34,
    38,
    44,
    46,
    48,
    50,
    56,
    60,
    62,
    66,
    68,
    70,
    76,
    78,
    84,
    86,
    88,
    90,
    98,
    104,
    106,
    110,
    114,
    116,
    122,
    126,
    128,
)


def _plan(
    current_size,
    *,
    enabled=True,
    reconstruction_current_size=None,
    quantum=8,
):
    return make_stable_fourier_window_shape_plan(
        _IMAGE_SHAPE,
        current_size,
        _N_HALF,
        enabled=enabled,
        reconstruction_current_size=reconstruction_current_size,
        quantum=quantum,
    )


def _relion_lane_tree_sum(storage, logical_count):
    """Host float32 replay of the fine scorer's 256-lane issue order."""

    block_size = 256
    lanes = np.zeros(block_size, dtype=np.float32)
    for pixel in range(int(logical_count)):
        lane = pixel % block_size
        lanes[lane] = np.float32(lanes[lane] + np.float32(storage[pixel]))
    width = block_size // 2
    while width:
        lanes[:width] = np.float32(lanes[:width] + lanes[width : 2 * width])
        width //= 2
    return lanes[0]


def test_stable_window_size_uses_eight_pixel_classes_and_isolates_full_box():
    assert stable_fourier_window_current_size(30, 128) == 32
    assert stable_fourier_window_current_size(34, 128) == 40
    assert stable_fourier_window_current_size(56, 128) == 56
    assert stable_fourier_window_current_size(70, 128) == 72
    assert stable_fourier_window_current_size(84, 128) == 88
    assert stable_fourier_window_current_size(122, 128) == 126
    assert stable_fourier_window_current_size(126, 128) == 126
    assert stable_fourier_window_current_size(128, 128) == 128


@pytest.mark.parametrize(
    ("current_size", "image_size", "quantum"),
    ((0, 128, 8), (31, 128, 8), (130, 128, 8), (32, 127, 8), (32, 128, 3)),
)
def test_stable_window_size_rejects_invalid_shapes(current_size, image_size, quantum):
    with pytest.raises(ValueError):
        stable_fourier_window_current_size(current_size, image_size, quantum=quantum)


def test_stable_window_runtime_quantum_is_diagnostic_and_fail_closed(monkeypatch):
    from relax.fourier.fourier_window import (
        DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
        STABLE_FOURIER_WINDOW_QUANTUM_ENV,
        stable_fourier_window_quantum,
    )

    monkeypatch.delenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raising=False)
    assert stable_fourier_window_quantum() == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM == 8

    monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, "16")
    assert stable_fourier_window_quantum() == 16

    for invalid in ("0", "3", "nope"):
        monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, invalid)
        with pytest.raises(ValueError, match=STABLE_FOURIER_WINDOW_QUANTUM_ENV):
            stable_fourier_window_quantum()


def _active_coarse_score_indices(current_size: int) -> np.ndarray:
    plan = make_stable_fourier_window_shape_plan(
        _IMAGE_SHAPE,
        current_size,
        _N_HALF,
        enabled=False,
    )
    return np.asarray(plan.logical_spec.score_indices_np, dtype=np.int32)


def test_stable_coarse_square_preserves_logical_issue_prefix(monkeypatch):
    monkeypatch.setenv(
        "RELAX_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM",
        "32",
    )
    logical_size = 70
    layout = plan_coarse_gaussian_square_layout(
        _IMAGE_SHAPE,
        logical_size,
        _active_coarse_score_indices(logical_size),
        stable_fourier_window_shapes=True,
    )
    logical_indices, logical_count = make_fourier_window_indices_np(
        _IMAGE_SHAPE,
        logical_size,
        square=True,
        include_dc=True,
    )
    logical_lookup = _relion_cuda_fine_full_to_compact_lookup(
        _IMAGE_SHAPE,
        logical_size,
        logical_indices,
    )

    assert layout.physical_current_size == 96
    assert layout.logical_square_count == logical_count == 70 * 36
    assert layout.physical_square_count == 96 * 49
    assert_matches(
        layout.score_indices_np[:logical_count],
        logical_indices,
    )
    assert_matches(
        layout.full_to_compact_np[:logical_count],
        logical_lookup,
    )
    assert_matches(
        layout.full_to_compact_np[logical_count:],
        np.arange(logical_count, layout.physical_square_count, dtype=np.int32),
    )
    fused_lookup = coarse_gaussian_fused_logical_lookup(
        layout.full_to_compact_np,
        layout,
        current_size=logical_size,
    )
    assert fused_lookup.shape == (logical_count,)
    assert_matches(np.asarray(fused_lookup), logical_lookup)
    with pytest.raises(ValueError, match="does not match current_size"):
        coarse_gaussian_fused_logical_lookup(
            layout.full_to_compact_np,
            layout,
            current_size=logical_size + 2,
        )
    assert not np.any(layout.score_active_mask_np[logical_count:])
    coords = np.rint(make_frequency_coords_half_np(_IMAGE_SHAPE)).astype(np.int64)
    expected_projector_mask = (
        np.sum(coords[layout.score_indices_np] ** 2, axis=1)
        <= (logical_size // 2) ** 2
    )
    expected_projector_mask[logical_count:] = False
    assert_matches(
        layout.logical_projector_mask_np,
        expected_projector_mask,
    )
    assert not np.any(layout.logical_projector_mask_np[logical_count:])


def test_fused_lookup_strips_q32_physical_tail_for_current_size_26(monkeypatch):
    """Regression for the live current_size=26, physical_size=32 fused failure."""

    monkeypatch.setenv(
        "RELAX_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM",
        "32",
    )
    logical_size = 26
    layout = plan_coarse_gaussian_square_layout(
        _IMAGE_SHAPE,
        logical_size,
        _active_coarse_score_indices(logical_size),
        stable_fourier_window_shapes=True,
    )

    assert layout.logical_square_count == 26 * 14
    assert layout.physical_current_size == 32
    assert layout.physical_square_count == 32 * 17
    assert layout.full_to_compact_np.shape == (32 * 17,)
    fused_lookup = coarse_gaussian_fused_logical_lookup(
        layout.full_to_compact_np,
        layout,
        current_size=logical_size,
    )
    assert fused_lookup.shape == (26 * 14,)
    assert_matches(
        np.asarray(fused_lookup),
        layout.full_to_compact_np[: 26 * 14],
    )


def test_stable_coarse_projector_keeps_logical_disk_boundary(monkeypatch):
    """Physical size 96 must retain the logical size-86 projector disk."""

    monkeypatch.setenv(
        "RELAX_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM",
        "32",
    )
    logical_size = 86
    layout = plan_coarse_gaussian_square_layout(
        _IMAGE_SHAPE,
        logical_size,
        _active_coarse_score_indices(logical_size),
        stable_fourier_window_shapes=True,
    )
    coords = np.rint(make_frequency_coords_half_np(_IMAGE_SHAPE)).astype(np.int64)
    compact_r2 = np.sum(coords[layout.score_indices_np] ** 2, axis=1)
    just_outside = np.flatnonzero(
        (np.arange(layout.physical_square_count) < layout.logical_square_count)
        & (compact_r2 == (logical_size // 2) ** 2 + 1)
    )
    assert just_outside.tolist() == [
        57,
        333,
        783,
        1317,
        1847,
        1935,
        2461,
        2983,
        3413,
        3665,
        3741,
    ]
    assert np.all(layout.score_active_mask_np[just_outside])
    assert not np.any(layout.logical_projector_mask_np[just_outside])

    projection_crop = np.ones(
        (2, layout.physical_square_count),
        dtype=np.complex64,
    )
    # The exact source-pixel disk is the explicit diagnostic arm of the
    # embedding helper; it must use the logical, not the physical, radius.
    masked = _texture_centered_crop_at_indices(
        projection_crop,
        layout.score_indices_np,
        image_shape=_IMAGE_SHAPE,
        projector_output_size=layout.physical_current_size,
        mask_current_image_disk=True,
        current_image_mask_size=np.int32(logical_size),
    )
    masked = np.asarray(masked)
    assert_matches(masked[:, just_outside], np.complex64(0))
    assert_matches(masked[:, layout.logical_projector_mask_np], np.complex64(1))

    # By default the native texture kernel owns image clipping
    # (docs/math/sparse_projection_radius.md): the physical-size-96 projection
    # must hand the kernel the logical size-86 radius, and the embedding helper
    # must then preserve the kernel's values.
    from relax.cuda import kernels as em_cuda_kernels
    from relax.projection import projection

    kernel_calls = []
    physical = int(layout.physical_current_size)
    kernel_crop = np.ones((2, physical * (physical // 2 + 1)), dtype=np.complex64)

    def capture_capacity_kernel(projector_half, rotations, logical_r_max, *, image_shape, padding_factor, image_r_max):
        del projector_half, rotations, logical_r_max, padding_factor
        kernel_calls.append((tuple(image_shape), int(image_r_max)))
        return kernel_crop

    monkeypatch.setattr(em_cuda_kernels, "project_relion_half_capacity", capture_capacity_kernel)
    default_route = np.asarray(
        projection._project_relion_projector_texture(
            np.zeros((5, 5, 3), dtype=np.complex64),
            np.zeros((2, 3, 3), dtype=np.float32),
            _IMAGE_SHAPE,
            r_max=1,
            projector_output_size=physical,
            current_image_mask_size=np.int32(logical_size),
            pixel_indices=layout.score_indices_np,
        )
    )
    assert kernel_calls == [((physical, physical), logical_size // 2)]
    assert_matches(default_route, np.complex64(1))


def test_stable_coarse_square_has_one_shape_across_q32_class(monkeypatch):
    monkeypatch.setenv(
        "RELAX_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM",
        "32",
    )
    layouts = [
        plan_coarse_gaussian_square_layout(
            _IMAGE_SHAPE,
            current_size,
            _active_coarse_score_indices(current_size),
            stable_fourier_window_shapes=True,
        )
        for current_size in (70, 72, 76, 78, 84, 96)
    ]

    assert {layout.physical_current_size for layout in layouts} == {96}
    assert {layout.score_indices_np.shape for layout in layouts} == {(96 * 49,)}
    assert {layout.full_to_compact_np.shape for layout in layouts} == {(96 * 49,)}
    assert len({layout.logical_square_count for layout in layouts}) == 6


def test_disabled_coarse_square_layout_is_legacy_exact(monkeypatch):
    monkeypatch.setenv(
        "RELAX_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM",
        "32",
    )
    current_size = 70
    active = _active_coarse_score_indices(current_size)
    layout = plan_coarse_gaussian_square_layout(
        _IMAGE_SHAPE,
        current_size,
        active,
        stable_fourier_window_shapes=False,
    )
    legacy_indices, legacy_count = make_fourier_window_indices_np(
        _IMAGE_SHAPE,
        current_size,
        square=True,
        include_dc=True,
    )
    legacy_lookup = _relion_cuda_fine_full_to_compact_lookup(
        _IMAGE_SHAPE,
        current_size,
        legacy_indices,
    )

    assert layout.logical_current_size == layout.physical_current_size == current_size
    assert layout.logical_square_count == layout.physical_square_count == legacy_count
    assert_matches(layout.score_indices_np, legacy_indices)
    assert_matches(layout.score_active_mask_np, np.isin(legacy_indices, active))
    coords = np.rint(make_frequency_coords_half_np(_IMAGE_SHAPE)).astype(np.int64)
    assert_matches(
        layout.logical_projector_mask_np,
        np.sum(coords[legacy_indices] ** 2, axis=1) <= (current_size // 2) ** 2,
    )
    assert_matches(layout.full_to_compact_np, legacy_lookup)


def test_shape_policy_is_default_off():
    plan = _plan(70, enabled=False)

    assert plan.logical_current_size == 70
    assert plan.physical_current_size == 70
    assert plan.logical_loop_bounds == (
        plan.physical_score_pixels,
        plan.physical_reconstruction_pixels,
        plan.physical_projection_pixels,
        plan.physical_rectangle_pixels,
    )


def test_adjacent_cutoffs_share_one_physical_signature_but_keep_logical_bounds():
    plans = [_plan(current_size) for current_size in (68, 70, 72)]

    assert {plan.physical_current_size for plan in plans} == {72}
    assert len({plan.physical_signature for plan in plans}) == 1
    assert len({plan.logical_loop_bounds for plan in plans}) == 3


def test_score_and_reconstruction_cutoffs_are_bucketed_independently():
    plan = _plan(70, reconstruction_current_size=84)

    assert plan.logical_current_size == 70
    assert plan.physical_current_size == 72
    assert plan.logical_reconstruction_current_size == 84
    assert plan.physical_reconstruction_current_size == 88
    assert plan.physical_projection_pixels >= plan.logical_projection_pixels


@pytest.mark.parametrize(
    ("current_size", "physical_size", "logical_projection", "physical_projection"),
    (
        (56, 56, 1276, 1276),
        (70, 72, 1980, 2093),
        (84, 88, 2835, 3105),
        (128, 128, 8320, 8320),
    ),
)
def test_gf46_checkpoint_projection_capacity(
    current_size,
    physical_size,
    logical_projection,
    physical_projection,
):
    plan = _plan(current_size)

    assert plan.physical_current_size == physical_size
    assert plan.logical_projection_pixels == logical_projection
    assert plan.physical_projection_pixels == physical_projection


def test_gf46_shape_trajectory_collapses_from_29_signatures_to_14():
    plans = [_plan(current_size) for current_size in _GF46_CURRENT_SIZES_THROUGH_80]

    assert len({plan.logical_current_size for plan in plans}) == 29
    assert len({plan.physical_signature for plan in plans}) == 14
    assert sorted({plan.physical_current_size for plan in plans}) == [
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        126,
        128,
    ]


@pytest.mark.parametrize(
    ("quantum", "expected_physical_sizes"),
    (
        (16, [32, 48, 64, 80, 96, 112, 126, 128]),
        (32, [32, 64, 96, 126, 128]),
    ),
)
def test_larger_diagnostic_quantums_reduce_trajectory_shape_cardinality(
    quantum,
    expected_physical_sizes,
):
    plans = [
        _plan(current_size, quantum=quantum)
        for current_size in _GF46_CURRENT_SIZES_THROUGH_80
    ]

    assert sorted({plan.physical_current_size for plan in plans}) == expected_physical_sizes
    assert all(
        plan.physical_projection_pixels >= plan.logical_projection_pixels
        for plan in plans
    )


def test_padding_appends_storage_without_changing_score_pixel_order():
    plan = _plan(70)
    logical_indices, _ = make_fourier_window_indices_np(_IMAGE_SHAPE, 70)
    padded_indices = pad_axis(
        logical_indices,
        0,
        plan.physical_score_pixels,
        value=0,
    )

    assert_matches(
        padded_indices[: plan.logical_score_pixels],
        logical_indices,
    )
    assert np.all(padded_indices[plan.logical_score_pixels :] == 0)


def test_packed_capacity_reorders_interleaved_physical_support_behind_logical_prefix():
    plan = _plan(70)
    logical = plan.logical_spec.score_indices_np
    physical_sorted = plan.physical_spec.score_indices_np

    # Enlarging a radial window inserts new flat-grid indices throughout the
    # sorted support; slicing the physical spec would therefore change order.
    assert not np.array_equal(physical_sorted[: logical.size], logical)

    packed = plan.packed_indices_np("score")
    assert_matches(packed[: logical.size], logical)
    assert_matches(
        np.sort(packed[logical.size :]),
        np.setdiff1d(physical_sorted, logical, assume_unique=True),
    )


@pytest.mark.parametrize("name", ("score", "recon"))
def test_logical_projection_takes_stay_at_front_of_packed_capacity(name):
    plan = _plan(70, reconstruction_current_size=84)
    packed_projection = plan.packed_indices_np("projection")
    packed_support = plan.packed_indices_np(name)
    packed_take = plan.packed_projection_take_np(name)
    logical_count = getattr(plan, f"logical_{'reconstruction' if name == 'recon' else name}_pixels")

    assert_matches(
        packed_projection[packed_take[:logical_count]],
        packed_support[:logical_count],
    )


def test_runtime_rectangle_stride_is_the_logical_not_physical_width():
    plan = _plan(70)
    first_pixel_on_second_logical_row = plan.logical_current_size // 2 + 1
    logical_half_width = plan.logical_current_size // 2 + 1
    physical_half_width = plan.physical_current_size // 2 + 1

    assert divmod(first_pixel_on_second_logical_row, logical_half_width) == (1, 0)
    assert divmod(first_pixel_on_second_logical_row, physical_half_width) == (0, 36)


def test_runtime_logical_bound_preserves_fine_lane_reduction():
    plan = _plan(70)
    rng = np.random.default_rng(17)
    logical = rng.standard_normal(plan.logical_rectangle_pixels).astype(np.float32)
    padded = pad_axis(
        logical,
        0,
        plan.physical_rectangle_pixels,
        value=np.float32(1.25e5),
    )

    baseline = _relion_lane_tree_sum(logical, plan.logical_rectangle_pixels)
    stable_shape = _relion_lane_tree_sum(padded, plan.logical_rectangle_pixels)
    wrong_physical_bound = _relion_lane_tree_sum(padded, plan.physical_rectangle_pixels)

    assert_matches(stable_shape, baseline)
    assert not matches(wrong_physical_bound, baseline)


def test_runtime_logical_bound_preserves_bpref_issue_sequence():
    plan = _plan(84)
    logical_issues = np.arange(plan.logical_reconstruction_pixels, dtype=np.int32)
    padded_issues = pad_axis(
        logical_issues,
        0,
        plan.physical_reconstruction_pixels,
        value=-1,
    )

    assert_matches(
        padded_issues[: plan.logical_reconstruction_pixels],
        logical_issues,
    )
    assert np.all(padded_issues[plan.logical_reconstruction_pixels :] == -1)


def test_packed_physical_spec_keeps_every_logical_stream_as_its_prefix():
    plan = _plan(70)
    packed = plan.packed_physical_spec()

    assert packed.n_score == plan.physical_score_pixels
    assert packed.n_recon == plan.physical_reconstruction_pixels
    assert packed.n_projection == plan.physical_projection_pixels
    for name, logical_count in (
        ("score", plan.logical_score_pixels),
        ("recon", plan.logical_reconstruction_pixels),
        ("projection", plan.logical_projection_pixels),
    ):
        assert_matches(
            getattr(packed, f"{name}_indices_np")[:logical_count],
            getattr(plan.logical_spec, f"{name}_indices_np"),
        )
    assert_matches(
        np.asarray(packed.score_projection_take[: plan.logical_score_pixels]),
        np.asarray(plan.logical_spec.score_projection_take),
    )
    assert_matches(
        np.asarray(packed.recon_projection_take[: plan.logical_reconstruction_pixels]),
        np.asarray(plan.logical_spec.recon_projection_take),
    )


def test_stable_wavg_rectangle_preserves_logical_fftw_order_and_poison_tail():
    plan = _plan(70)
    logical = _make_relion_wavg_rectangle(
        _IMAGE_SHAPE,
        plan.logical_current_size,
        plan.logical_spec.recon_indices_np,
    )
    stable = _make_stable_relion_wavg_rectangle(_IMAGE_SHAPE, plan)
    logical_rectangle_count = plan.logical_rectangle_pixels
    logical_recon_count = plan.logical_reconstruction_pixels

    assert stable.centered_indices.size == plan.physical_rectangle_pixels
    assert_matches(
        stable.centered_indices[:logical_rectangle_count],
        logical.centered_indices,
    )
    assert_matches(
        stable.exact_positions[:logical_recon_count],
        logical.exact_positions,
    )
    assert_matches(
        stable.shell_indices[:logical_rectangle_count],
        logical.shell_indices,
    )
    assert np.all(stable.shell_indices[logical_rectangle_count:] == -1)
    assert not np.intersect1d(
        stable.centered_indices[:logical_rectangle_count],
        stable.centered_indices[logical_rectangle_count:],
    ).size

    poison = np.full(plan.physical_rectangle_pixels, np.float32(np.nan))
    logical_values = np.arange(logical_rectangle_count, dtype=np.float32)
    poison[:logical_rectangle_count] = logical_values
    assert_matches(poison[:logical_rectangle_count], logical_values)
    assert np.isnan(poison[logical_rectangle_count:]).all()


def test_stable_wavg_reconstruction_tail_cannot_alias_logical_rectangle():
    plan = _plan(70)
    stable = _make_stable_relion_wavg_rectangle(_IMAGE_SHAPE, plan)

    assert np.all(
        stable.exact_positions[plan.logical_reconstruction_pixels :]
        >= plan.logical_rectangle_pixels
    )
    assert np.unique(stable.exact_positions).size == stable.exact_positions.size


def test_every_gf46_stable_window_has_enough_inert_wavg_tail_capacity():
    for current_size in _GF46_CURRENT_SIZES_THROUGH_80:
        plan = _plan(current_size)
        if plan.physical_current_size == plan.logical_current_size:
            continue
        stable = _make_stable_relion_wavg_rectangle(_IMAGE_SHAPE, plan)
        logical_recon_count = plan.logical_reconstruction_pixels
        assert_matches(
            stable.exact_positions[:logical_recon_count],
            _make_relion_wavg_rectangle(
                _IMAGE_SHAPE,
                current_size,
                plan.logical_spec.recon_indices_np,
            ).exact_positions,
        )
        assert np.all(
            stable.exact_positions[logical_recon_count:]
            >= plan.logical_rectangle_pixels
        )
        assert np.all(
            stable.exact_positions[logical_recon_count:]
            < plan.physical_rectangle_pixels
        )


def test_crop_relion_x_half_accumulator_excludes_physical_poison_bitwise():
    physical_shape = (11, 11, 11)
    logical_shape = (7, 7, 7)
    physical_half_width = physical_shape[2] // 2 + 1
    logical_half_width = logical_shape[2] // 2 + 1
    physical = np.full(
        (physical_shape[0], physical_shape[1], physical_half_width),
        np.uint32(0x7FC00001),
        dtype=np.uint32,
    )
    expected = np.arange(np.prod((*logical_shape[:2], logical_half_width)), dtype=np.uint32).reshape(
        (*logical_shape[:2], logical_half_width)
    )
    start = (physical_shape[0] - logical_shape[0]) // 2
    physical[
        start : start + logical_shape[0],
        start : start + logical_shape[1],
        :logical_half_width,
    ] = expected

    cropped = crop_relion_x_half_accumulator(
        physical.reshape(-1),
        physical_shape,
        logical_shape,
    )

    assert_matches(cropped.reshape(expected.shape), expected)
    assert not np.any(cropped == np.uint32(0x7FC00001))


@pytest.mark.parametrize(
    ("physical_shape", "logical_shape"),
    (
        ((10, 10, 10), (7, 7, 7)),
        ((11, 11, 11), (8, 8, 8)),
        ((9, 11, 9), (7, 7, 7)),
        ((7, 7, 7), (9, 9, 9)),
    ),
)
def test_crop_relion_x_half_accumulator_rejects_unsupported_topology(
    physical_shape,
    logical_shape,
):
    with pytest.raises(ValueError, match="nested odd cubic shapes"):
        crop_relion_x_half_accumulator(
            np.zeros(1, dtype=np.float32),
            physical_shape,
            logical_shape,
        )


def test_runtime_cuda_kernels_are_separate_from_default_primitives():
    source = read_em_cuda_source()

    powerclass_start = source.index("void relion_powerclass_spectrum_highres_f32_kernel(")
    powerclass = source[
        powerclass_start : source.index(
            "cudaError_t launch_relion_powerclass_spectrum_highres_f32(",
            powerclass_start,
        )
    ]
    assert "runtime_resolution_limit" not in powerclass
    assert "shell >= resolution_limit" in powerclass
    runtime_powerclass_start = source.index(
        "void relion_powerclass_spectrum_highres_runtime_f32_kernel("
    )
    runtime_powerclass = source[
        runtime_powerclass_start : source.index(
            "cudaError_t launch_relion_powerclass_spectrum_highres_runtime_f32(",
            runtime_powerclass_start,
        )
    ]
    assert "runtime_resolution_limit[0]" in runtime_powerclass

    wavg_start = source.index("relion_wavg_rotation_atomic_triplet_f32_kernel(")
    wavg = source[
        wavg_start : source.index(
            "cudaError_t launch_relion_wavg_rotation_atomic_triplet_add_f32(",
            wavg_start,
        )
    ]
    assert "runtime_logical_pixel_count" not in wavg
    runtime_wavg_start = source.index(
        "relion_wavg_rotation_atomic_runtime_triplet_f32_kernel("
    )
    runtime_wavg = source[
        runtime_wavg_start : source.index(
            "cudaError_t launch_relion_wavg_rotation_atomic_runtime_triplet_add_f32(",
            runtime_wavg_start,
        )
    ]
    assert "pixel < logical_pixel_count" in runtime_wavg
    assert "pixel_capacity + pixel" in runtime_wavg


def test_stable_bpref_uses_capacity_stride_but_logical_native_issue_count():
    source = read_em_cuda_source()
    launcher_start = source.index(
        "cudaError_t launch_relion_vdam_mstep_fused_projector_x_half("
    )
    launcher = source[
        launcher_start : source.index(
            "__device__ __forceinline__ float relion_fine_diff2_update_f32",
            launcher_start,
        )
    ]

    assert "const int64_t image_stride = pixel_capacity;" in launcher
    assert "static_cast<unsigned>(pixel_count)" in launcher
    assert "image_real + particle * image_stride" in launcher
    assert "ctf + particle * image_stride" in launcher
    assert "minvsigma2 + particle * image_stride" in launcher
    assert "launch_relion_vdam_mstep_denominator_f32(" in launcher
    assert "pixel_count," in launcher
    denominator_start = launcher.index("launch_relion_vdam_mstep_denominator_f32(")
    denominator_call = launcher[
        denominator_start + len("launch_relion_vdam_mstep_denominator_f32(") :
        launcher.index(");", denominator_start)
    ]
    assert [argument.strip() for argument in denominator_call.split(",")] == [
        "stream", "ctf", "minvsigma2", "posterior_over_weight_norm",
        "denominator_sum", "n_particles", "rotation_count", "translation_count",
        "pixel_count", "pixel_capacity", "runtime_current_size",
    ]

    handler_start = source.rindex("ffi::Error RelionVdamMstepFusedProjectorXHalfCommon(")
    handler = source[
        handler_start : source.index(
            "XLA_FFI_DEFINE_HANDLER_SYMBOL(",
            handler_start,
        )
    ]
    assert "const int64_t pixel_count = image_h * image_w;" in handler
    assert "image_dims[1] != pixel_capacity" in handler
    assert "denominator_dims[2] != pixel_capacity" in handler


def test_runtime_bpref_ffi_abi_keeps_default_static_target_separate():
    root = Path(__file__).resolve().parents[3]
    python_source = (root / "relax" / "cuda" / "kernels.py").read_text()
    cuda_source = read_em_cuda_source()

    assert "cuda_relion_vdam_mstep_fused_projector_x_half" in python_source
    assert "cuda_relion_vdam_mstep_fused_projector_runtime_x_half" in python_source
    assert "RelionVdamMstepFusedProjectorXHalfCommon(" in cuda_source
    assert "RelionVdamMstepFusedProjectorRuntimeXHalfImpl(" in cuda_source
    static_binding_start = cuda_source.index(
        "XLA_FFI_DEFINE_HANDLER_SYMBOL(\n    RelionVdamMstepFusedProjectorXHalf,"
    )
    runtime_binding_start = cuda_source.index(
        "XLA_FFI_DEFINE_HANDLER_SYMBOL(\n"
        "    RelionVdamMstepFusedProjectorRuntimeXHalf,"
    )
    # Scope each ABI assertion to its own macro; adjacent handlers may add
    # independent operands without changing either of these two contracts.
    static_binding = cuda_source[
        static_binding_start : cuda_source.index(");", static_binding_start) + 2
    ]
    runtime_binding = cuda_source[
        runtime_binding_start : cuda_source.index(");", runtime_binding_start) + 2
    ]
    assert static_binding.count(".Arg<ffi::AnyBuffer>()") == 17
    assert runtime_binding.count(".Arg<ffi::AnyBuffer>()") == 18
    runtime_handler = cuda_source[
        cuda_source.index("ffi::Error RelionVdamMstepFusedProjectorRuntimeXHalfImpl(") :
        cuda_source.index(
            "XLA_FFI_DEFINE_HANDLER_SYMBOL(\n    RelionVdamMstepFusedProjectorXHalf,"
        )
    ]
    assert "ffi::AnyBuffer logical_current_size" in runtime_handler
    assert "&logical_current_size" in runtime_handler
    common_start = cuda_source.rindex(
        "ffi::Error RelionVdamMstepFusedProjectorXHalfCommon("
    )
    common = cuda_source[
        common_start : cuda_source.index(
            "XLA_FFI_DEFINE_HANDLER_SYMBOL(\n    RelionVdamMstepFusedProjectorXHalf,",
            common_start,
        )
    ]
    assert "const ffi::AnyBuffer* runtime_current_size" in common
    assert "logical_current_size must be an S32 scalar" in common
    assert "runtime_current_size->untyped_data()" in common
    assert "runtime_current_size != nullptr &&" in cuda_source
    assert "exact_native_ptx_requested || exact_wavg_predecessor_requested" in cuda_source
