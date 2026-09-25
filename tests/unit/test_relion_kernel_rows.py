"""RELION's accelerated-kernel row rule for image windows wider than the model sphere (CPU).

An image on a coarser grid than the reference (scale 1 < s < sqrt(2)) has a Fourier window
wider than 2 r_max. RELION's kernels relabel the rows beyond maxR = min(r_max, imgX - 1)
before projecting them, and the relabelled pixel falls outside the rotated sphere.
``relion_kernel_zero_rows`` must mark exactly those rows. The scalar model below is written
from the RELION source independently of the production helper.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from scipy.spatial.transform import Rotation

from relax.helpers.projection import compute_relion_projector_projections_block, relion_kernel_zero_rows


def _acc_kernel_coordinate(i, x, img_y, max_r, kernel):
    """(x, y) RELION's kernel projects for FFTW row index ``i``, column ``x``.

    Coarse diff2 wraps at maxR (acc/cuda/cuda_kernels/diff2.cuh:86-90); fine diff2 and wavg
    keep the negative rows from imgY - maxR and move the others to x = maxR
    (diff2.cuh:688-694, wavg.cuh:81-86).
    """
    if i <= max_r:
        return x, i
    if kernel == "coarse" or i >= img_y - max_r:
        return x, i - img_y
    return max_r, i


def _acc_kernel_keeps(x, y, matrix, max_r, padding_factor):
    """AccProjectorKernel::project3Dmodel's int-truncated padded radius test (acc_projectorkernel_impl.h:161-179)."""
    xp = (matrix[0, 0] * x + matrix[0, 1] * y) * padding_factor
    yp = (matrix[1, 0] * x + matrix[1, 1] * y) * padding_factor
    zp = (matrix[2, 0] * x + matrix[2, 1] * y) * padding_factor
    return int(xp * xp + yp * yp + zp * zp) <= max_r * max_r * padding_factor * padding_factor


@pytest.mark.unit
@pytest.mark.parametrize("kernel", ["coarse", "fine"])
# Away from the documented edge where imgY/2 equals s * maxR (there the coarse kernel keeps
# the x = 0 pixel of the Nyquist row).
@pytest.mark.parametrize("scale, r_max", [(1.12, 28), (1.3, 19), (1.06, 40)])
def test_zero_rows_are_the_rows_relion_relabels_outside_the_sphere(kernel, scale, r_max):
    img_y = 2 * int(np.ceil(0.5 * scale * 2 * r_max))  # group_current_size of a 2 r_max reference window
    max_r = min(r_max, img_y // 2)
    zero = np.asarray(relion_kernel_zero_rows(img_y, img_y, r_max, kernel)).reshape(img_y, img_y // 2 + 1)
    centred_label = np.arange(img_y) - img_y // 2
    centred_label[0] = img_y // 2  # full window: the positive Nyquist row sits at centred row zero
    for matrix in Rotation.random(4, random_state=3).as_matrix() / scale:
        for row, label in enumerate(centred_label):
            i = label % img_y  # FFTW row index
            for x in range(img_y // 2 + 1):
                xk, yk = _acc_kernel_coordinate(i, x, img_y, max_r, kernel)
                if zero[row, x]:
                    assert not _acc_kernel_keeps(xk, yk, matrix, max_r, 2), (label, x)
                else:
                    assert (xk, yk) == (x, label), (label, x)


@pytest.mark.unit
def test_window_within_the_model_sphere_has_no_rule():
    assert relion_kernel_zero_rows(64, 56, 28, "fine") is None
    assert relion_kernel_zero_rows(64, 64, 32, "coarse") is None
    with pytest.raises(ValueError, match="relion_kernel"):
        relion_kernel_zero_rows(64, 64, 28, "wavg")


@pytest.mark.unit
def test_projection_block_applies_the_named_kernels_rule():
    rng = np.random.default_rng(0)
    projector = jnp.asarray(rng.standard_normal((2 * 8 + 3, 2 * 8 + 3, 10)) + 0j, dtype=jnp.complex128)
    kwargs = dict(r_max=8, padding_factor=1, centered_rows=True, projector_output_size=20, relion_texture_interp=False)
    label = np.arange(20) - 10
    label[0] = 10
    matrices = Rotation.random(2, random_state=5).as_matrix()
    # Unscaled rotations: the rows beyond maxR are outside the sphere anyway, so the rule is inert.
    plain, _ = compute_relion_projector_projections_block(projector, jnp.asarray(matrices), (20, 20), **kwargs)
    assert not np.asarray(plain).reshape(2, 20, 11)[:, np.abs(label) > 8].any()
    scaled = jnp.asarray(matrices / 1.12)
    fine, _ = compute_relion_projector_projections_block(projector, scaled, (20, 20), **kwargs)
    coarse, _ = compute_relion_projector_projections_block(projector, scaled, (20, 20), relion_kernel="coarse", **kwargs)
    fine, coarse = np.asarray(fine).reshape(2, 20, 11), np.asarray(coarse).reshape(2, 20, 11)
    assert not fine[:, np.abs(label) > 8].any() and not coarse[:, label > 8].any()
    # The coarse kernel still projects the negative rows beyond maxR; both keep the rest unchanged.
    assert coarse[:, label < -8].any()
    assert_matches(fine[:, np.abs(label) <= 8], coarse[:, np.abs(label) <= 8])
