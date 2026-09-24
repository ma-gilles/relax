import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_relion_project_half_uses_projector_matrix_directly():
    """RELION accelerator euler matrices are projector matrices, not inverses."""
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half

    n = 8
    volume = np.zeros((n, n, n // 2 + 1), dtype=np.complex128)
    source_value = 7.0 + 3.0j
    # Output pixel (x=1, y=2) under this RELION projector matrix maps to
    # (xp=-2, yp=1, zp=0), then RELION flips negative xp to the Hermitian
    # partner (xp=2, yp=-1, zp=0) and conjugates the interpolated value.
    volume[n // 2, n // 2 - 1, 2] = source_value
    relion_projector_matrix = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    projected = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(relion_projector_matrix),
            n,
            r_max=n // 2,
            padding_factor=1,
        )
    )

    np.testing.assert_allclose(projected[2, 1], np.conj(source_value), rtol=0.0, atol=1e-12)


def test_centered_row_projector_transposes_scorer_rotations():
    """Centered-row RECOVAR scoring uses the transpose at the raw Projector handoff."""
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half
    from relax.helpers.projection import project_relion_projector_half_spectrum_centered_rows

    n = 8
    rng = np.random.default_rng(7)
    volume = (
        rng.normal(size=(n, n, n // 2 + 1)) + 1j * rng.normal(size=(n, n, n // 2 + 1))
    ).astype(np.complex128)
    scorer_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    row_order = np.fft.fftshift(np.arange(n))

    got = np.asarray(
        project_relion_projector_half_spectrum_centered_rows(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation[None]),
            (n, n),
            r_max=n // 2,
            padding_factor=1,
        )
    )
    expected_raw = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation.T),
            n,
            r_max=n // 2,
            padding_factor=1,
        )
    )
    direct_raw = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation),
            n,
            r_max=n // 2,
            padding_factor=1,
        )
    )

    expected = expected_raw[row_order, :].reshape(1, -1)
    direct_centered = direct_raw[row_order, :].reshape(1, -1)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(got - direct_centered)) > 1e-3


def test_centered_row_projector_scatters_cropped_ppref_into_full_box():
    """Cropped RELION ``PPref`` output must land in full-box RECOVAR row order."""
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half
    from relax.helpers.projection import project_relion_projector_half_spectrum_centered_rows

    full_n = 8
    current_size = 4
    rng = np.random.default_rng(11)
    volume = (
        rng.normal(size=(current_size, current_size, current_size // 2 + 1))
        + 1j * rng.normal(size=(current_size, current_size, current_size // 2 + 1))
    ).astype(np.complex128)
    scorer_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    got = np.asarray(
        project_relion_projector_half_spectrum_centered_rows(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation[None]),
            (full_n, full_n),
            r_max=current_size // 2,
            padding_factor=1,
        )
    ).reshape(full_n, full_n // 2 + 1)
    expected_crop = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation.T),
            current_size,
            r_max=current_size // 2,
            padding_factor=1,
        )
    )

    crop_rows = np.arange(current_size, dtype=np.int32)
    crop_ky = np.where(crop_rows <= current_size // 2, crop_rows, crop_rows - current_size)
    full_rows = crop_ky + full_n // 2
    for crop_row, full_row in enumerate(full_rows):
        np.testing.assert_allclose(
            got[full_row, : current_size // 2 + 1],
            expected_crop[crop_row],
            rtol=1e-12,
            atol=1e-12,
        )


def test_centered_row_projector_can_use_explicit_coarse_output_size():
    """RELION pass-1 projects PPref into the current-size Fimg box."""
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half
    from relax.helpers.projection import project_relion_projector_half_spectrum_centered_rows

    full_n = 8
    projector_n = 8
    coarse_n = 4
    rng = np.random.default_rng(13)
    volume = (
        rng.normal(size=(projector_n, projector_n, projector_n // 2 + 1))
        + 1j * rng.normal(size=(projector_n, projector_n, projector_n // 2 + 1))
    ).astype(np.complex128)
    scorer_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    got = np.asarray(
        project_relion_projector_half_spectrum_centered_rows(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation[None]),
            (full_n, full_n),
            r_max=projector_n // 2,
            padding_factor=1,
            projector_output_size=coarse_n,
        )
    ).reshape(full_n, full_n // 2 + 1)
    expected_crop = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(scorer_rotation.T),
            coarse_n,
            r_max=projector_n // 2,
            padding_factor=1,
        )
    )

    crop_rows = np.arange(coarse_n, dtype=np.int32)
    crop_ky = np.where(crop_rows <= coarse_n // 2, crop_rows, crop_rows - coarse_n)
    full_rows = crop_ky + full_n // 2
    for crop_row, full_row in enumerate(full_rows):
        np.testing.assert_allclose(
            got[full_row, : coarse_n // 2 + 1],
            expected_crop[crop_row],
            rtol=1e-12,
            atol=1e-12,
        )


def test_relion_acc_double_floorf_quirk_matches_default_away_from_integer_boundary():
    """Away from an integer coordinate, the GPU floorf-narrowing quirk is a no-op."""
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half

    n = 16
    rng = np.random.default_rng(23)
    volume = (
        rng.normal(size=(n, n, n // 2 + 1)) + 1j * rng.normal(size=(n, n, n // 2 + 1))
    ).astype(np.complex128)
    # A generic (non-axis-aligned) rotation, far from any integer-coordinate edge case.
    rotation = np.asarray(
        [
            [0.36, -0.48, 0.8],
            [0.8, 0.6, 0.0],
            [-0.48, 0.64, 0.6],
        ],
        dtype=np.float64,
    )

    default = np.asarray(
        relion_project_half(jnp.asarray(volume), jnp.asarray(rotation), n, r_max=n // 2, padding_factor=1)
    )
    quirked = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(rotation),
            n,
            r_max=n // 2,
            padding_factor=1,
            relion_acc_double_floorf_quirk=True,
        )
    )
    np.testing.assert_allclose(quirked, default, rtol=1e-6, atol=1e-6)


def test_relion_project_half_truncates_rotated_radius_before_clipping():
    """AccProjectorKernel assigns the positive floating r² sum to ``int``."""
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half

    n = 8
    r_max = 2
    volume = np.ones((n, n, n // 2 + 1), dtype=np.complex128)
    rotation = np.eye(3, dtype=np.float64)
    rotation[0, 0] = 1.0001

    # At output (x=2,y=0), rotated r² is slightly greater than 4. RELION's
    # int conversion yields 4, so the r_max=2 boundary pixel remains valid.
    rotated_r2 = (2.0 * rotation[0, 0]) ** 2
    assert 4.0 < rotated_r2 < 5.0
    projected = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(rotation),
            n,
            r_max=r_max,
            padding_factor=1,
        )
    )

    np.testing.assert_allclose(projected[0, 2], 1.0 + 0.0j, rtol=0.0, atol=1e-12)


def test_relion_acc_double_floorf_quirk_flips_bucket_at_integer_boundary():
    """RELION's GPU no_tex3D/BP.cuh floor with float32 ``floorf`` on the double
    coordinate, unconditionally, even under ``ACC_DOUBLE_PRECISION``
    (relion/src/acc/cuda/cuda_kernels/cuda_device_utils.cuh, BP.cuh). A
    coordinate within a float32 ULP of an integer therefore floors to a
    *different* integer than a native double floor would give, selecting a
    different pair of interpolation neighbors and a fractional weight
    computed against that (now wrong) integer -- not merely a smaller
    rounding error.
    """
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half

    n = 32
    # xp = 10 - 4e-7: float64 floors to 9. Cast to float32, this value rounds
    # up to exactly 10.0 (float32 ULP at this magnitude is ~9.5e-7, so 4e-7 is
    # within half a ULP) and floors to 10 -- a different integer bucket.
    offset = 4e-7
    xp0 = 10.0 - offset
    assert np.floor(xp0) == 9.0
    assert np.floor(np.float64(np.float32(xp0))) == 10.0

    volume = np.zeros((n, n, n // 2 + 1), dtype=np.complex128)
    y0r = n // 2  # yp = 0 -> y0 = 0 -> y0r = 0 - starting_y = n//2
    z0r = n // 2
    # A sharp spike at x-index 10 so the two floor choices (bracket [9,10] vs
    # [10,11]) give visibly different interpolated values.
    volume[z0r, y0r, 9] = 0.0
    volume[z0r, y0r, 10] = 1.0
    volume[z0r, y0r, 11] = 0.0

    # R_relion is a synthetic (non-orthonormal) matrix, chosen only to make
    # xp land exactly on xp0 at pixel (x=1, y=0) with yp = zp = 0.
    rotation = np.asarray(
        [
            [xp0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    default = np.asarray(
        relion_project_half(jnp.asarray(volume), jnp.asarray(rotation), n, r_max=n // 2, padding_factor=1)
    )
    quirked = np.asarray(
        relion_project_half(
            jnp.asarray(volume),
            jnp.asarray(rotation),
            n,
            r_max=n // 2,
            padding_factor=1,
            relion_acc_double_floorf_quirk=True,
        )
    )

    fx_off = xp0 - 9
    fx_on = xp0 - 10
    expected_default = 0.0 + fx_off * (1.0 - 0.0)
    expected_quirked = 1.0 + fx_on * (0.0 - 1.0)

    np.testing.assert_allclose(default[0, 1], expected_default, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(quirked[0, 1], expected_quirked, rtol=0.0, atol=1e-12)
    # The two floor choices select different neighbor pairs; the resulting
    # values differ by ~8e-7 here -- far above float64 rounding noise (~1e-15)
    # and proof the flag actually changes which bucket is used.
    assert abs(quirked[0, 1] - default[0, 1]) > 1e-8


def _acc_kernel_keeps(x, y, matrix, r_max, image_size, padding_factor):
    """RELION's accelerated radius decision for one half-image pixel.

    ``AccProjectorKernel::makeKernel`` clamps the model radius to the image
    radius ``imgMaxR = imgX - 1`` (relion/src/acc/acc_projectorkernel_impl.h:301-310,
    acc/acc_ml_optimiser_impl.h:1902-1907); ``project3Dmodel`` truncates the rotated
    squared radius to ``int`` and compares it with ``maxR**2 * pf**2``
    (acc_projectorkernel_impl.h:161-179). The diff2 kernels wrap rows at ``maxR``
    (acc/cuda/cuda_kernels/diff2.cuh:86-90) and apply no unrotated output disc.
    """
    max_r = min(r_max, image_size // 2)
    if y > max_r:
        y -= image_size
    xp = (matrix[0, 0] * x + matrix[0, 1] * y) * padding_factor
    yp = (matrix[1, 0] * x + matrix[1, 1] * y) * padding_factor
    zp = (matrix[2, 0] * x + matrix[2, 1] * y) * padding_factor
    return int(xp * xp + yp * yp + zp * zp) <= max_r * max_r * padding_factor * padding_factor


def _scaled_rotation(scale):
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = scale
    return matrix


@pytest.mark.parametrize("r_max", [3, 4, 6])
@pytest.mark.parametrize(
    "matrix",
    [
        # Pixel (x=4, y=1): r**2 = 17 * 16.99 / 17 truncates to 16. Kept at
        # r_max 4 by the GPU rule, although 4**2 + 1**2 lies outside the disc.
        _scaled_rotation(np.sqrt(16.99 / 17.0)),
        # Pixel (x=4, y=0): r**2 = 17.6 lies inside r_max 6 but outside the
        # image radius 4, so makeKernel's clamp removes it.
        _scaled_rotation(np.sqrt(1.1)),
        *Rotation.random(4, random_state=31).as_matrix().astype(np.float32).astype(np.float64),
    ],
)
def test_relion_project_half_radius_matches_relion_accelerated_kernel(r_max, matrix):
    """The kept pixels are exactly those RELION's GPU kernel keeps."""
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half

    image_size = 8
    volume_size = 24  # large enough that no kept pixel reads outside the box
    volume = np.ones((volume_size, volume_size, volume_size // 2 + 1), dtype=np.complex128)
    projected = np.asarray(
        relion_project_half(jnp.asarray(volume), jnp.asarray(matrix), image_size, r_max=r_max, padding_factor=1)
    )

    expected = np.asarray(
        [
            [_acc_kernel_keeps(x, row, matrix, r_max, image_size, 1) for x in range(image_size // 2 + 1)]
            for row in range(image_size)
        ]
    )
    # A constant volume interpolates to exactly one wherever a pixel is kept.
    np.testing.assert_array_equal(projected != 0, expected)
    np.testing.assert_allclose(projected[expected], 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("padding_factor", [1, 2])
@pytest.mark.parametrize("logical_size,physical_size", [(30, 32), (60, 64), (46, 56)])
def test_relion_project_half_center_padded_capacity_matches_logical(logical_size, physical_size, padding_factor):
    """Center-padding PPref into a larger capacity and cropping is neutral.

    The projector is complex128 and rotations are float32 matrices widened to
    float64, as in VDAM. Padded and logical projections evaluate the same
    arithmetic on the same texels, so the measured difference is zero; the
    1e-12 relative tolerance only allows for compiler reassociation.
    """
    import jax.numpy as jnp

    from relax.relion.relion_project import relion_project_half

    r_max = logical_size // 2
    pf = padding_factor
    size = 2 * (pf * r_max + 1) + 1
    rng = np.random.default_rng(logical_size * 10 + pf)
    logical = rng.standard_normal((size, size, size // 2 + 1)) + 1j * rng.standard_normal(
        (size, size, size // 2 + 1)
    )
    coord = np.arange(size) - size // 2
    radius2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + np.arange(size // 2 + 1)[None, None, :] ** 2
    logical[radius2 > (pf * r_max) ** 2] = 0.0  # RELION's PPref support, ghost planes included

    capacity = 2 * (pf * (physical_size // 2) + 1) + 1
    offset = (capacity - size) // 2
    padded = np.zeros((capacity, capacity, capacity // 2 + 1), dtype=logical.dtype)
    padded[offset : offset + size, offset : offset + size, : logical.shape[2]] = logical

    rows = np.arange(logical_size)
    signed_y = np.where(rows <= logical_size // 2, rows, rows - logical_size)
    physical_rows = np.where(signed_y >= 0, signed_y, signed_y + physical_size)
    x_half = np.arange(logical_size // 2 + 1)
    shell = (signed_y[:, None] ** 2 + x_half[None, :] ** 2) == r_max**2 + 1

    rotations = Rotation.random(12, random_state=logical_size).as_matrix().astype(np.float32).astype(np.float64)
    shell_pixels_kept = 0
    for matrix in rotations:
        reference = np.asarray(
            relion_project_half(jnp.asarray(logical), jnp.asarray(matrix), logical_size, r_max=r_max, padding_factor=pf)
        )
        candidate = np.asarray(
            relion_project_half(jnp.asarray(padded), jnp.asarray(matrix), physical_size, r_max=r_max, padding_factor=pf)
        )[physical_rows][:, : logical_size // 2 + 1]
        np.testing.assert_allclose(candidate, reference, rtol=0.0, atol=1e-12 * np.max(np.abs(reference)))
        shell_pixels_kept += int(np.count_nonzero(reference[shell]))
    # At pf 1 the comparison must include the r_max**2 + 1 shell the int rule
    # admits. At pf 2 that shell scales to 4 * r_max**2 + 4 and cannot truncate
    # to 4 * r_max**2, so no shell pixel is kept.
    assert (shell_pixels_kept > 0) == (pf == 1)
