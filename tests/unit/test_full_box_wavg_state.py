"""Full-box Wavg sentinel regressions recovered from donor344eca9fb4."""
import numpy as np
import jax.numpy as jnp
from relax.sparse_pass2.sparse_pass2_wavg import _make_relion_wavg_rectangle
from relax.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_powerclass_highres_xi2_half, _relion_powerclass_noise_terms,
)
from relax.helpers.fourier_window import make_fourier_window_indices_np
from helpers.float_compare import assert_matches

def test_full_box_wavg_rectangle_resolves_unwindowed_current_size_sentinel():
    image_shape = (8, 8)
    rectangle = _make_relion_wavg_rectangle(
        image_shape,
        current_size=None,
        recon_window_indices=None,
    )

    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    assert rectangle.centered_indices.shape == (n_half,)
    assert rectangle.exact_positions.shape == (n_half,)
    assert_matches(
        np.sort(rectangle.centered_indices),
        np.arange(n_half, dtype=np.int32),
    )
    assert_matches(
        np.sort(rectangle.exact_positions),
        np.arange(n_half, dtype=np.int32),
    )
    assert_matches(
        rectangle.centered_indices[rectangle.exact_positions],
        np.arange(n_half, dtype=np.int32),
    )
    assert np.any(rectangle.shell_indices == -1)
    rounded_indices, _ = make_fourier_window_indices_np(
        image_shape,
        image_shape[0],
        include_dc=True,
        exact_radius=False,
    )
    assert np.count_nonzero(rectangle.shell_indices >= 0) == rounded_indices.size

    high_shell = _relion_cuda_powerclass_highres_xi2_half(
        jnp.ones((2, n_half), dtype=jnp.complex64),
        image_shape=image_shape,
        current_size=image_shape[0],
    )
    assert_matches(np.asarray(high_shell), np.zeros(2, dtype=np.float32))


def test_full_box_powerclass_noise_terms_has_zero_high_shell():
    highres, norm = _relion_powerclass_noise_terms(
        jnp.ones((2, 40), dtype=jnp.complex64),
        image_shape=(8, 8), current_size=None,
        use_exact_relion_gaussian=True, accumulate_noise=True,
        source_faithful_spectrum_norm=False,
    )
    assert_matches(np.asarray(highres), np.zeros(2, dtype=np.float32))
    # Keep the existing caller-level absence contract; direct Wavg norm fills
    # a zero vector at its own boundary when the full box has no high shell.
    assert norm is None


def test_full_box_resident_statistics_replaces_all_shells_without_norm_cutoff():
    from relax.sparse_pass2.resident_statistics import resolve_statistics_config

    common = dict(n_shells=5, n_fine_trans=1, n_images=2,
                  n_coarse_rot=3, n_scale_groups=1)
    full = resolve_statistics_config(current_size=None, **common)
    cropped = resolve_statistics_config(current_size=4, **common)
    assert full.direct_noise_exclusive_shell_stop == 5
    assert full.norm_unweighted_shell_cutoff is None
    assert cropped.direct_noise_exclusive_shell_stop == 3
    assert cropped.norm_unweighted_shell_cutoff == 2


