from pathlib import Path

import numpy as np
from helpers.float_compare import assert_matches

from relax.fine_pass.wavg import _make_relion_wavg_rectangle
from relax.fourier.fourier_window import (
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
)
from scripts.analyze_k1_scale_aa_pixels import analyze


def test_relion_wavg_rectangle_matches_native_size60_topology_and_order():
    image_shape = (256, 256)
    current_size = 60
    exact_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        include_dc=True,
        exact_radius=True,
    )

    layout = _make_relion_wavg_rectangle(
        image_shape,
        current_size,
        exact_indices,
    )

    assert layout.centered_indices.size == 60 * 31 == 1860
    assert layout.exact_positions.size == 1411
    assert np.count_nonzero(layout.shell_indices >= 0) == 1462
    assert np.count_nonzero(layout.shell_indices < 0) == 398
    assert np.count_nonzero(
        (layout.shell_indices >= 0)
        & ~np.isin(np.arange(layout.shell_indices.size), layout.exact_positions)
    ) == 51
    # Native FFTW row-major order starts at ky=0 and walks kx=0..N/2.
    half_width = image_shape[1] // 2 + 1
    expected_first_row = image_shape[0] // 2 * half_width + np.arange(31)
    assert_matches(layout.centered_indices[:31], expected_first_row)
    assert_matches(
        layout.centered_indices[layout.exact_positions],
        exact_indices,
    )


def test_relion_wavg_rectangle_maps_complete_square_cc_reconstruction():
    image_shape = (128, 128)
    current_size = 56
    square_exact, _ = make_fourier_window_indices_np(
        image_shape, current_size, square=True,
        include_dc=True, exact_radius=True,
    )
    layout = _make_relion_wavg_rectangle(image_shape, current_size, square_exact)
    assert layout.centered_indices.size == current_size * (current_size // 2 + 1)
    assert layout.exact_positions.size == square_exact.size == 1254
    assert_matches(layout.centered_indices[layout.exact_positions], square_exact)
    with np.testing.assert_raises_regex(ValueError, "complete exact-radius"):
        _make_relion_wavg_rectangle(image_shape, current_size, square_exact[:-1])


def test_relion_wavg_rectangle_separates_optics_image_and_model_sizes():
    image_shape = (384, 384)
    image_current_size = 58
    model_current_size = 56
    model_indices, _ = make_fourier_window_indices_np(
        image_shape,
        model_current_size,
        include_dc=True,
        exact_radius=True,
    )

    layout = _make_relion_wavg_rectangle(
        image_shape,
        image_current_size,
        model_indices,
        reconstruction_current_size=model_current_size,
    )

    assert layout.centered_indices.size == 58 * 30 == 1740
    assert layout.exact_positions.size == 1227
    assert_matches(
        layout.centered_indices[layout.exact_positions],
        model_indices,
    )
    # The particle rectangle retains the image-only rounded cutoff shell for
    # Wavg noise accumulation even though BPref projection terms stop at the
    # smaller model-coordinate radius.
    assert np.max(layout.shell_indices) == image_current_size // 2
    image_only = np.ones(layout.centered_indices.size, dtype=bool)
    image_only[layout.exact_positions] = False
    assert np.count_nonzero((layout.shell_indices >= 0) & image_only) > 0


def test_scale_aa_pixels_joins_fourier_coordinates_and_localizes_operand_delta(tmp_path: Path):
    image_size = 8
    current_size = 4
    divisor = 16.0
    window_indices, _ = make_fourier_window_indices_np(
        (image_size, image_size),
        current_size,
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    coordinates = np.rint(make_frequency_coords_half_np((image_size, image_size))).astype(np.int32)[
        window_indices
    ]
    shells = np.rint(np.linalg.norm(coordinates, axis=1)).astype(np.int32)
    mask = shells <= 1
    aa_native = np.arange(1, window_indices.size + 1, dtype=np.float64) / 100.0
    xa_native = np.arange(2, window_indices.size + 2, dtype=np.float64) / 200.0
    aa_recovar = aa_native * divisor
    aa_recovar[np.flatnonzero(mask)[-1]] *= 1.01
    aa_shell = np.asarray(
        [np.sum(aa_recovar[shells == shell], dtype=np.float64) for shell in range(current_size // 2 + 1)]
    )
    native_direct = np.asarray([0.25, 0.5, 0.75], dtype=np.float64)

    capture = tmp_path / "capture.npz"
    np.savez_compressed(
        capture,
        schema=np.asarray("recovar-k1-scale-xa-aa-chunked-v2"),
        iteration=np.int64(2),
        half=np.int64(1),
        original_index=np.int64(1096),
        group_id=np.int64(109),
        current_size=np.int64(current_size),
        scale_correction_pixel_mask=mask,
        scale_shell_indices=shells,
        scale_aa_per_pixel=aa_recovar.astype(np.float32),
        scale_aa_per_shell=aa_shell,
        scale_aa_atomic_per_pixel=(aa_native * divisor).astype(np.float32),
        scale_xa_per_pixel=(xa_native * divisor * 1.02).astype(np.float32),
        scale_xa_atomic_per_pixel=(xa_native * divisor).astype(np.float32),
        wavg_diff2_atomic_per_shell=(native_direct * divisor).astype(np.float32),
        wavg_diff2_atomic_rectangle_per_shell=(native_direct * divisor).astype(np.float32),
        relion_norm_high_shell=np.float64(0.75 * divisor),
    )

    native = tmp_path / "native.tsv"
    lines = []
    for row in np.flatnonzero(mask):
        x, y = coordinates[row]
        lines.append(
            "acc_scale_pixel\titer=2\tpart_id=109\thalfset=1"
            f"\tj={row}\tx={x}\ty={y}\tshell={shells[row]}"
            f"\taa={aa_native[row]:.17g}\txa={xa_native[row]:.17g}\n"
        )
    native.write_text("".join(reversed(lines)))
    native_components = tmp_path / "native_components.tsv"
    native_components.write_text(
        "".join(
            "acc_components\titer=2\tpart_id=109\thalfset=1"
            f"\tshell={shell}\tdirect_residual={value:.17g}"
            f"\thighres_image_power={0.25 * shell:.17g}\n"
            for shell, value in enumerate(native_direct)
        )
    )

    report = analyze(
        capture,
        native,
        native_noise_components=native_components,
        expected_iteration=2,
        expected_half=1,
        expected_part_id=109,
        expected_original_index=1096,
        image_size=image_size,
        recovar_term_divisor=divisor,
    )

    assert report["coordinate_join"]["shell_labels_exact"]
    assert report["pixel_aa"]["relative_l2"] > 0.0
    assert report["atomic_aa"]["pixel"]["relative_l2"] < 1e-7
    assert report["atomic_aa"]["fixed_order_shell_reduction"]["relative_l2"] < 1e-7
    assert report["xa"]["pixel"]["relative_l2"] > 0.0
    assert report["xa"]["atomic"]["pixel"]["relative_l2"] < 1e-7
    assert report["xa"]["atomic"]["fixed_order_shell_reduction"]["relative_l2"] < 1e-7
    assert report["wavg_direct_residual"]["relative_l2"] < 1e-7
    assert report["wavg_direct_residual"]["recovar_source"] == "full_relion_wavg_rectangle"
    assert report["wavg_direct_residual"]["full_particle_norm"]["relative_abs_error"] < 1e-7
    assert report["classification"] == "atomic Wavg XA/AA treatment captured"
    assert report["pixel_aa"]["largest_abs_residual_pixels"][0]["x"] == int(
        coordinates[np.flatnonzero(mask)[-1], 0]
    )


def test_norm_pixels_sum_chunked_v4_operands_before_native_join(tmp_path: Path):
    image_size = 8
    current_size = 4
    divisor = 16.0
    window_indices, _ = make_fourier_window_indices_np(
        (image_size, image_size),
        current_size,
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    coordinates = np.rint(make_frequency_coords_half_np((image_size, image_size))).astype(np.int32)[
        window_indices
    ]
    shells = np.rint(np.linalg.norm(coordinates, axis=1)).astype(np.int32)
    aa_native = np.arange(1, window_indices.size + 1, dtype=np.float32) / 100.0
    xa_native = np.arange(2, window_indices.size + 2, dtype=np.float32) / 200.0
    aa_chunks = np.stack((aa_native * divisor * 0.25, aa_native * divisor * 0.75))
    xa_chunks = np.stack((xa_native * divisor * 0.5, xa_native * divisor * 0.5))

    capture = tmp_path / "norm_capture.npz"
    np.savez_compressed(
        capture,
        schema=np.asarray("recovar-k1-scale-xa-aa-chunked-v4"),
        iteration=np.int64(2),
        half=np.int64(1),
        original_index=np.int64(78),
        group_id=np.int64(2767),
        current_size=np.int64(current_size),
        scale_shell_indices=shells,
        norm_a2_per_pixel_by_chunk=aa_chunks,
        norm_xa_per_pixel_by_chunk=xa_chunks,
        norm_a2_per_image=np.float64(np.sum(aa_chunks, dtype=np.float64)),
        norm_xa_per_image=np.float64(np.sum(xa_chunks, dtype=np.float64)),
    )
    native = tmp_path / "native.tsv"
    native.write_text(
        "".join(
            "acc_scale_pixel\titer=2\tpart_id=2767\thalfset=1"
            f"\tj={row}\tx={x}\ty={y}\tshell={shells[row]}"
            f"\taa={aa_native[row]:.17g}\txa={xa_native[row]:.17g}\n"
            for row, (x, y) in enumerate(coordinates)
        )
    )

    report = analyze(
        capture,
        native,
        expected_iteration=2,
        expected_half=1,
        expected_part_id=2767,
        expected_original_index=78,
        image_size=image_size,
        recovar_term_divisor=divisor,
        term_source="norm",
    )

    assert report["identity"]["term_source"] == "norm"
    assert report["identity"]["active_pixel_count"] == window_indices.size
    assert report["pixel_aa"]["relative_l2"] < 1e-7
    assert report["xa"]["pixel"]["relative_l2"] < 1e-7
    assert abs(report["term_totals_native_units"]["a2_native_minus_recovar"]) < 1e-7
    assert abs(report["term_totals_native_units"]["xa_native_minus_recovar"]) < 1e-7
    assert abs(report["term_totals_native_units"]["residual_native_minus_recovar"]) < 1e-7
    assert abs(report["recovar_pixel_sum_vs_captured_scalar"]["a2_signed_delta"]) < 1e-5
    assert abs(report["recovar_pixel_sum_vs_captured_scalar"]["xa_signed_delta"]) < 1e-5
