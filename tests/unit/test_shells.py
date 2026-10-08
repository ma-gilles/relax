"""``relax.helpers.shells`` against the inline shell rules it replaced, grid kind by grid kind."""

import jax.numpy as jnp
import numpy as np
import pytest

from relax.helpers.shells import fourier_radius_sq, shell_index, shell_of_radius, shell_of_radius_sq
from relax.relion.macros import relion_round_array

SIZES_2D = [(2, 2), (3, 3), (7, 7), (8, 8), (16, 12), (33, 33), (64, 64), (101, 101), (128, 96)]
SIZES_3D = [2, 3, 8, 9, 24, 31]


def _assert_same(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", SIZES_2D)
def test_centred_half_float32_half_even_matches_half_spectrum_plan(shape):
    height, width = shape
    half_width = width // 2 + 1
    vertical_grid = np.arange(-(height // 2), height - height // 2, dtype=np.float32)
    packed_grid = np.arange(0, half_width, dtype=np.float32)
    radial_sq = np.zeros((height, half_width), dtype=np.float32)
    radial_sq = radial_sq + vertical_grid[:, None] ** 2
    radial_sq = radial_sq + packed_grid[None, :] ** 2
    expected = np.rint(np.sqrt(radial_sq)).astype(np.int32)
    actual = shell_index(
        shape, rows="centred", half=True, rule="half_even", real_dtype=np.float32, index_dtype=np.int32
    )
    _assert_same(actual, expected)


@pytest.mark.parametrize("capacity", SIZES_3D)
@pytest.mark.parametrize("padding_factor", [1, 2, 3])
def test_centred_half_3d_padded_half_up_matches_static_shell_voxel_lists(capacity, padding_factor):
    coord = np.arange(capacity, dtype=np.int64) - capacity // 2
    x = np.arange(capacity // 2 + 1, dtype=np.int64)
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + x[None, None, :] ** 2
    _assert_same(fourier_radius_sq((capacity,) * 3, rows="centred", half=True), r2)
    expected = np.floor(np.sqrt(r2.astype(np.float64)) / float(padding_factor) + 0.5).astype(np.int64)
    _assert_same(shell_of_radius_sq(r2, rule="half_up", padding_factor=float(padding_factor)), expected)
    _assert_same(
        shell_index((capacity,) * 3, rows="centred", half=True, rule="half_up", padding_factor=float(padding_factor)),
        expected,
    )


@pytest.mark.parametrize("size", [2, 3, 8, 9, 64, 65])
def test_fftw_half_relion_round_matches_expected_accuracy_and_ctf(size):
    half = size // 2 + 1
    rows = np.arange(size)
    iy = np.where(rows < half, rows, rows - size)[:, None]
    ix = np.arange(half)[None, :]
    r2 = iy * iy + ix * ix
    expected = relion_round_array(np.sqrt(r2.astype(np.float64))).astype(np.int64)
    _assert_same(shell_of_radius_sq(r2, rule="half_up"), expected)
    _assert_same(np.floor(np.sqrt(r2) + 0.5).astype(np.int64), expected)  # relion_ctf's spelling
    _assert_same(shell_index((size, size), rows="fftw", half=True, rule="half_up"), expected)


@pytest.mark.parametrize("shape", SIZES_2D)
def test_fftw_half_half_even_matches_relion_half_plane_loop(shape):
    height, width = shape
    expected = np.zeros((height, width // 2 + 1), dtype=np.int64)
    for iy in range(height):
        ky = iy if iy <= height // 2 else iy - height
        for ix in range(width // 2 + 1):
            expected[iy, ix] = int(np.rint(np.sqrt(float(ky * ky + ix * ix))))
    _assert_same(shell_index(shape, rows="fftw", half=True, rule="half_even"), expected)


@pytest.mark.parametrize("shape", SIZES_2D)
def test_fftfreq_grids_match_integer_grids(shape):
    """The float ``fftfreq(n) * n`` grids of the start-up spectra give the integer grid's shells."""

    height, width = shape
    ky = np.fft.fftfreq(height, d=1.0) * height
    kx_full = np.fft.fftfreq(width, d=1.0) * width
    kx_half = np.arange(width // 2 + 1, dtype=np.float64)
    full = np.round(np.sqrt(ky[:, None] ** 2 + kx_full[None, :] ** 2)).astype(np.int64)
    half = np.round(np.sqrt(ky[:, None] ** 2 + kx_half[None, :] ** 2)).astype(np.int64)
    _assert_same(shell_of_radius_sq(ky[:, None] ** 2 + kx_full[None, :] ** 2, rule="half_even"), full)
    _assert_same(shell_of_radius_sq(ky[:, None] ** 2 + kx_half[None, :] ** 2, rule="half_even"), half)
    _assert_same(shell_index(shape, rows="fftw", half=False, rule="half_even"), full)
    _assert_same(shell_index(shape, rows="fftw", half=True, rule="half_even"), half)


@pytest.mark.parametrize("n", SIZES_3D)
def test_fftw_3d_half_up_matches_reference_initialization_and_solvent_mask(n):
    k = np.fft.fftfreq(n, d=1.0) * n
    kx = np.arange(n // 2 + 1, dtype=np.float64)
    half_r2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + kx[None, None, :] ** 2
    full_r2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    half = np.floor(np.sqrt(half_r2) + 0.5).astype(np.int64)
    full = np.floor(np.sqrt(full_r2) + 0.5).astype(np.int64)
    _assert_same(shell_of_radius_sq(half_r2, rule="half_up"), half)
    _assert_same(shell_of_radius_sq(full_r2, rule="half_up"), full)
    _assert_same(
        shell_of_radius_sq(half_r2, rule="half_up"),
        relion_round_array(np.asarray(np.sqrt(half_r2), dtype=np.float64)).astype(np.int64),
    )
    _assert_same(shell_index((n,) * 3, rows="fftw", half=True, rule="half_up"), half)
    _assert_same(shell_index((n,) * 3, rows="fftw", half=False, rule="half_up"), full)


@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rule", ["half_up", "floor"])
@pytest.mark.parametrize("padding_factor", [1, 2, 3])
def test_radius_rules_match_regularization_host_path(real_dtype, rule, padding_factor):
    radius = np.sqrt(fourier_radius_sq((24, 24, 24), rows="centred", half=True).astype(real_dtype))
    if rule == "half_up":
        expected = np.floor(radius / padding_factor + 0.5).astype(np.int32)
    else:
        expected = np.floor(radius / padding_factor).astype(np.int32)
    _assert_same(shell_of_radius(radius, rule=rule, padding_factor=padding_factor, index_dtype=np.int32), expected)


@pytest.mark.parametrize("real_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("padding_factor", [1, 2])
def test_jax_half_up_matches_projector_setup(real_dtype, padding_factor):
    size = 20
    coord = jnp.arange(size, dtype=jnp.int32) - size // 2
    x = jnp.arange(size // 2 + 1, dtype=jnp.int32)
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + x[None, None, :] ** 2
    expected = jnp.floor(jnp.sqrt(r2.astype(real_dtype)) / padding_factor + 0.5).astype(jnp.int32)
    actual = shell_of_radius_sq(
        r2, rule="half_up", real_dtype=real_dtype, padding_factor=padding_factor, index_dtype=jnp.int32, xp=jnp
    )
    assert isinstance(actual, type(expected))
    _assert_same(actual, expected)


@pytest.mark.parametrize("rule", ["half_up", "floor"])
def test_jax_radius_rules_match_regularization_device_path(rule):
    radius = jnp.sqrt(jnp.asarray(fourier_radius_sq((12, 12, 12), rows="centred", half=False), jnp.float32))
    expected = jnp.floor(radius / 2 + 0.5) if rule == "half_up" else jnp.floor(radius / 2)
    _assert_same(
        shell_of_radius(radius, rule=rule, padding_factor=2, index_dtype=jnp.int32, xp=jnp), expected.astype(jnp.int32)
    )


def test_rules_differ_on_an_exact_half():
    radius = np.array([0.5, 1.5, 2.5])
    _assert_same(shell_of_radius(radius, rule="half_up"), np.array([1, 2, 3]))
    _assert_same(shell_of_radius(radius, rule="half_even"), np.array([0, 2, 2]))
    _assert_same(shell_of_radius(radius, rule="floor"), np.array([0, 1, 2]))


def test_unknown_rule_and_rows_refused():
    with pytest.raises(ValueError, match="rule"):
        shell_of_radius(np.ones(2), rule="round")
    with pytest.raises(ValueError, match="rows"):
        fourier_radius_sq((4, 4), rows="shifted", half=True)
