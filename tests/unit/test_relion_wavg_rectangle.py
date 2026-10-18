import numpy as np
from helpers.float_compare import assert_matches

from relax.fine_pass.wavg import make_relion_wavg_rectangle
from relax.fourier.fourier_window import (
    make_fourier_window_indices_np,
)


def test_relion_wavg_rectangle_matches_native_size60_topology_and_order():
    image_shape = (256, 256)
    current_size = 60
    exact_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        include_dc=True,
        exact_radius=True,
    )

    layout = make_relion_wavg_rectangle(
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
    layout = make_relion_wavg_rectangle(image_shape, current_size, square_exact)
    assert layout.centered_indices.size == current_size * (current_size // 2 + 1)
    assert layout.exact_positions.size == square_exact.size == 1254
    assert_matches(layout.centered_indices[layout.exact_positions], square_exact)
    with np.testing.assert_raises_regex(ValueError, "complete exact-radius"):
        make_relion_wavg_rectangle(image_shape, current_size, square_exact[:-1])


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

    layout = make_relion_wavg_rectangle(
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
