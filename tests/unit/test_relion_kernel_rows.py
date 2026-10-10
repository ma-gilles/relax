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

from relax.projection.projection import compute_relion_projector_projections_block, relion_kernel_zero_rows


def _acc_kernel_coordinate(i, x, img_y, max_r, kernel):
    """(x, y) RELION's kernel projects for FFTW row index ``i``, column ``x``.

    Coarse diff2 wraps at maxR (acc/cuda/cuda_kernels/diff2.cuh:86-90); fine diff2 and wavg
    keep the negative rows from imgY - maxR and move the others to x = maxR
    (diff2.cuh:688-694, wavg.cuh:81-86); the SGD backprojection kernel wraps at imgY / 2
    and projects every row (BP.cuh:476-492).
    """
    if kernel == "sgd":
        return x, (i - img_y if i > img_y // 2 else i)
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


@pytest.mark.unit
@pytest.mark.parametrize("scale", [1.12, 1.3])
def test_coarse_band_is_where_relion_projects_wrapped_rows(scale):
    from relax.relion.optics_scale import coarse_rows_wrap_inside

    r_max = 20
    matrix = Rotation.random(1, random_state=2).as_matrix()[0] / scale
    for window in range(2 * r_max - 4, 2 * int(np.ceil(scale * r_max)) + 6, 2):
        max_r = min(r_max, window // 2)
        wrapped_inside = any(
            _acc_kernel_keeps(*_acc_kernel_coordinate(i, x, window, max_r, "coarse"), matrix, max_r, 2)
            for i in range(max_r + 1, window // 2 + 1)
            for x in range(window // 2 + 1)
        )
        # The band is conservative about the rotation: it holds for the x = 0 pixel of any rotation.
        if wrapped_inside:
            assert coarse_rows_wrap_inside(window, r_max, scale), window
        if not coarse_rows_wrap_inside(window, r_max, scale):
            assert not wrapped_inside, window


@pytest.mark.unit
@pytest.mark.parametrize("image_size, window, r_max", [(18, 18, 8), (112, 56, 25), (64, 40, 17), (128, 128, 60)])
def test_coarse_relabel_is_relions_kernel_row_rule(image_size, window, r_max):
    """Every window pixel keeps its label or moves to the row RELION's coarse kernel reads it at."""
    from relax.projection.projection import relion_coarse_relabel

    relabel = relion_coarse_relabel(image_size, window, r_max)
    max_r = min(r_max, window // 2)
    half_width = image_size // 2 + 1
    moved = dict(zip(relabel.positions.tolist(), relabel.grid_indices.tolist()))
    grid = relabel.grid_size
    label = np.arange(image_size) - image_size // 2
    if window == image_size:
        label[0] = image_size // 2
    for row, centred in enumerate(label):
        # Both window Nyquist labels are FFTW row window / 2 (a layout holds one of them).
        if not -(window // 2) <= centred <= window // 2:
            continue
        i = centred % window  # FFTW row of the window
        for x in range(window // 2 + 1):
            xk, yk = _acc_kernel_coordinate(i, x, window, max_r, "coarse")
            position = row * half_width + x
            if position in moved:
                g = moved[position]
                assert (g % (grid // 2 + 1), g // (grid // 2 + 1) - grid // 2) == (xk, yk), (centred, x)
            else:
                assert (xk, yk) == (x, centred), (centred, x)
    assert relabel.row_shift == -window
    assert relion_coarse_relabel(image_size, 2 * r_max, r_max) is None


@pytest.mark.unit
@pytest.mark.parametrize("compact", [False, True])
def test_coarse_projection_in_the_band_projects_relions_relabelled_rows(compact):
    """In the band (window between 2 r_max and 2 s r_max) RELION projects the rows beyond maxR at
    ``i - window`` inside the sphere; the block projector must give those values, not zeros."""
    from relax.projection.projection import project_relion_projector_half_spectrum_centered_rows
    from relax.relion.optics_scale import coarse_rows_wrap_inside

    rng = np.random.default_rng(0)
    r_max, window, scale = 8, 18, 1.3
    assert coarse_rows_wrap_inside(window, r_max, scale)
    projector = jnp.asarray(rng.standard_normal((2 * r_max + 3, 2 * r_max + 3, r_max + 2)) + 0j, dtype=jnp.complex128)
    rotations = jnp.asarray(Rotation.random(3, random_state=5).as_matrix() / scale)
    half_width = window // 2 + 1
    indices = np.arange(window * half_width)
    if compact:
        indices = np.sort(rng.choice(indices, size=indices.size // 2, replace=False))
    coarse, _ = compute_relion_projector_projections_block(
        projector, rotations, (window, window), r_max=r_max, padding_factor=1, centered_rows=True,
        projector_output_size=window, relion_texture_interp=False, relion_kernel="coarse",
        pixel_indices=jnp.asarray(indices, jnp.int32) if compact else None,
    )
    coarse = np.asarray(coarse)
    # Reference: the plain projector on a grid large enough to hold every kernel coordinate.
    grid = 4 * window
    plain = np.asarray(
        project_relion_projector_half_spectrum_centered_rows(projector, rotations, (grid, grid), r_max, 1, grid, False)
    ).reshape(3, grid, grid // 2 + 1)
    label = np.arange(window) - window // 2
    label[0] = window // 2
    for column, index in enumerate(indices):
        row, x = divmod(int(index), half_width)
        xk, yk = _acc_kernel_coordinate(label[row] % window, x, window, r_max, "coarse")
        assert_matches(coarse[:, column], plain[:, yk + grid // 2, xk])
    relabelled = [
        column for column, index in enumerate(indices)
        if label[int(index) // half_width] > r_max
    ]
    assert np.abs(coarse[:, relabelled]).max() > 0  # RELION's wrapped rows are inside the sphere here


@pytest.mark.unit
def test_coarse_projection_band_refuses_routes_that_cannot_relabel():
    rng = np.random.default_rng(0)
    projector = jnp.asarray(rng.standard_normal((2 * 8 + 3, 2 * 8 + 3, 10)) + 0j, dtype=jnp.complex128)
    rotations = jnp.asarray(Rotation.random(2, random_state=5).as_matrix() / 1.3)
    # Uncentred rows have no centred relabel grid.
    with pytest.raises(NotImplementedError, match="does not reproduce"):
        compute_relion_projector_projections_block(
            projector, rotations, (18, 18), r_max=8, padding_factor=1, centered_rows=False,
            projector_output_size=18, relion_texture_interp=False, relion_kernel="coarse",
        )


@pytest.mark.unit
@pytest.mark.parametrize("scale, r_max", [(1.12, 19), (1.3, 8)])
def test_sgd_kernel_projects_every_row_inside_the_sphere(scale, r_max):
    """VDAM's subtracted reference (RELION's SGD backprojection kernel) has no zero rows: the rows the
    fine kernel zeroes hold pixels inside the rotated sphere, which it projects."""
    from relax.projection.projection import project_relion_projector_half_spectrum_centered_rows

    window = 2 * int(np.ceil(0.5 * scale * 2 * r_max))
    assert relion_kernel_zero_rows(window, window, r_max, "sgd") is None
    rng = np.random.default_rng(1)
    projector = jnp.asarray(rng.standard_normal((2 * r_max + 3, 2 * r_max + 3, r_max + 2)) + 0j, dtype=jnp.complex128)
    rotations = jnp.asarray(Rotation.random(3, random_state=7).as_matrix() / scale)
    kwargs = dict(r_max=r_max, padding_factor=1, centered_rows=True, projector_output_size=window, relion_texture_interp=False)
    sgd, _ = compute_relion_projector_projections_block(projector, rotations, (window, window), relion_kernel="sgd", **kwargs)
    fine, _ = compute_relion_projector_projections_block(projector, rotations, (window, window), **kwargs)
    plain = project_relion_projector_half_spectrum_centered_rows(projector, rotations, (window, window), r_max, 1, window, False)
    sgd, fine, plain = (np.asarray(a).reshape(3, window, window // 2 + 1) for a in (sgd, fine, plain))
    label = np.arange(window) - window // 2
    label[0] = window // 2
    beyond = np.abs(label) > r_max
    assert_matches(sgd, plain)
    assert_matches(sgd[:, ~beyond], fine[:, ~beyond])
    assert not fine[:, beyond].any() and np.abs(sgd[:, beyond]).max() > 0
    # Every pixel the SGD kernel projects is one RELION's kernel keeps inside the sphere.
    for matrix, rows in zip(np.asarray(rotations), sgd):
        for row, centred in enumerate(label):
            for x in range(window // 2 + 1):
                xk, yk = _acc_kernel_coordinate(centred % window, x, window, r_max, "sgd")
                if rows[row, x] != 0:
                    assert _acc_kernel_keeps(xk, yk, matrix, r_max, 1), (centred, x)


@pytest.mark.unit
def test_sgd_residual_rows_are_the_reconstruction_pixels_the_fine_rule_zeroes():
    from relax.fine_pass.resident_pass2 import _sgd_residual_rows

    box, window, r_max = 112, 44, 19
    half_width = box // 2 + 1
    label = np.arange(box) - box // 2
    rows, cols = np.divmod(np.arange(box * half_width), half_width)
    radius = np.hypot(label[rows], cols)
    recon = np.flatnonzero(np.rint(radius) < window // 2 + 1)
    rotations = Rotation.random(5, random_state=4).as_matrix() / 1.12
    kwargs = dict(image_shape=(box, box), r_max=r_max, rotations=rotations)
    positions, pixels = _sgd_residual_rows(recon, projector_output_size=window, **kwargs)
    assert np.array_equal(pixels, recon[positions])
    # Rows 20-21 reach the sphere at s = 1.12 (20 / 1.12 = 17.9); row 22 (19.6) does not.
    assert np.array_equal(positions, np.flatnonzero((np.abs(label[rows[recon]]) > r_max) & (np.abs(label[rows[recon]]) <= 21)))
    # A window within the model sphere (every single-grid pass) has none.
    assert _sgd_residual_rows(recon, projector_output_size=2 * r_max, **kwargs) is None
    # A window one row wider from a scale a hair above 1 (a real-data header pixel) has none either.
    near_one = Rotation.random(5, random_state=4).as_matrix() / (1.0 + 1.7e-8)
    assert _sgd_residual_rows(recon, image_shape=(box, box), projector_output_size=2 * r_max + 2, r_max=r_max, rotations=near_one) is None


@pytest.mark.unit
def test_cached_block_projections_give_the_residual_its_sgd_rows():
    """The M-step keeps the cached (fine-rule) rows for the Wavg terms and takes the SGD rows for the residual."""
    from relax.fine_pass.resident_pass2 import _cached_block_projections, _ChunkStageTables

    rng = np.random.default_rng(2)
    union = jnp.asarray(rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5)), dtype=jnp.complex64)
    sgd = jnp.asarray(rng.standard_normal((6, 2)) + 1j * rng.standard_normal((6, 2)), dtype=jnp.complex64)
    fields = {name: None for name in _ChunkStageTables._fields}
    fields.update(
        projection_score_cache=union,
        mstep_grid=jnp.asarray(rng.standard_normal((6, 3, 3)), dtype=jnp.float32),
        union_recon_take=jnp.asarray([4, 1, 3, 0], jnp.int32),
    )
    rows = jnp.asarray([5, 2, 2], jnp.int32)
    plain = _cached_block_projections(_ChunkStageTables(**fields), rows)
    assert len(plain) == 3
    fields.update(residual_sgd_cache=sgd, residual_sgd_take=jnp.asarray([1, 3], jnp.int32))
    recon, recon_abs2, rotations, residual = _cached_block_projections(_ChunkStageTables(**fields), rows)
    assert_matches(recon, plain[0])
    assert_matches(recon_abs2, plain[1])
    assert_matches(rotations, plain[2])
    expected = np.asarray(plain[0]).copy()
    expected[:, [1, 3]] = np.asarray(sgd)[np.asarray(rows)]
    assert_matches(residual, expected)
