"""CPU checks of the FFT level 0 of the branch-and-bound pose search (relax.scoring.bnb_fft): its scores equal the
lattice product's on every circular shift, its gather tables, frames, local masses, step and prior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.refinement.refinement_options import BnBSearchOptions
from relax.scoring import bnb_fft, bnb_search


def _half_disc(radius: int) -> np.ndarray:
    """Whole frequencies ``(kx, ky)`` of a disc's half plane ``kx >= 0``, both signs of ``ky`` at ``kx = 0``."""

    k = np.arange(-radius, radius + 1)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    keep = (kx >= 0) & (kx**2 + ky**2 <= radius**2)
    return np.stack([kx[keep], ky[keep]], axis=1)


@pytest.mark.parametrize("swap", [False, True])
def test_fft_block_scores_every_circular_shift(swap):
    """One FFT block gives the lattice product's scores (RELION's phase ``exp(-2 pi i k.t / N)``) on every shift."""

    rng = np.random.default_rng(1)
    n, step, n_img, n_rot = 32, 2.0, 3, 5
    m = int(n / step)
    labels = _half_disc(6)
    if swap:
        labels = labels[:, ::-1].copy()
    n_pix = labels.shape[0]
    y = (rng.normal(size=(n_img, n_pix)) + 1j * rng.normal(size=(n_img, n_pix))).astype(np.complex64)
    weight = rng.uniform(0.1, 2.0, size=(n_img, n_pix)).astype(np.float32)
    projections = (rng.normal(size=(n_rot, n_pix)) + 1j * rng.normal(size=(n_rot, n_pix))).astype(np.complex64)
    initial = rng.uniform(0, 5, size=n_img).astype(np.float32)
    rotation_prior = rng.normal(size=n_rot).astype(np.float32)
    first, second, half = bnb_fft.fft_tables(labels, m)
    assert half == (1 if swap else 0)
    lattice = bnb_fft.fft_lattice(m, step, half)
    lattice_prior = rng.normal(size=m * m).astype(np.float32)
    phase = np.exp(-2j * np.pi * (lattice @ labels.T.astype(np.float64)) / n).astype(np.complex64)  # [T, S]
    shifted = y[:, None, :] * phase[None, :, :]
    weighted, image_term = bnb_search._image_terms(jnp.asarray(shifted), jnp.asarray(weight), jnp.asarray(initial))
    reference = np.asarray(
        bnb_search._dense_block(
            weighted, image_term, jnp.asarray(weight), jnp.asarray(projections), jnp.asarray(rotation_prior),
            jnp.asarray(lattice_prior),
        )
    )  # [B, T, R]
    unshifted, term = bnb_search._image_terms(
        jnp.asarray(y[:, None, :]), jnp.asarray(weight), jnp.asarray(initial)
    )
    wy = unshifted[:, 0, :n_pix] + 1j * unshifted[:, 0, n_pix:]
    mass, tmass, best, arg = bnb_fft._fft_block(
        wy, term[:, 0], jnp.asarray(weight), jnp.asarray(projections), jnp.asarray(rotation_prior),
        jnp.asarray(lattice_prior)[None], jnp.asarray(first, jnp.int32), jnp.asarray(second, jnp.int32), m=m,
    )
    np.testing.assert_allclose(np.asarray(mass), np.asarray(jax.nn.logsumexp(reference, axis=1)), rtol=1e-5, atol=2e-3)
    np.testing.assert_allclose(np.asarray(tmass), np.asarray(jax.nn.logsumexp(reference, axis=2)), rtol=1e-5, atol=2e-3)
    np.testing.assert_allclose(np.asarray(best), reference.max(axis=(1, 2)), rtol=1e-5, atol=2e-3)
    flat = np.swapaxes(reference, 1, 2).reshape(n_img, -1)
    np.testing.assert_array_equal(np.asarray(arg), flat.argmax(axis=1))


def test_fft_tables_place_every_row_once():
    """Every row has one cell, the zero-column rows a mirror cell; frequencies beyond the grid are refused."""

    labels = _half_disc(5)
    first, second, half = bnb_fft.fft_tables(labels, 16)
    assert half == 0
    assert np.array_equal(np.sort(first[first >= 0]), np.arange(labels.shape[0]))
    assert np.array_equal(np.sort(second[second >= 0]), np.nonzero(labels[:, 0] == 0)[0])
    with pytest.raises(ValueError):
        bnb_fft.fft_tables(labels, 8)
    with pytest.raises(ValueError):
        bnb_fft.fft_tables(np.array([[-1, 2], [1, -2]]), 16)


def test_frame_is_the_window_holding_the_most_mass():
    """The frame is the window holding the most posterior mass (a broad lump beats a sharper, lighter peak), at the
    mode among windows that hold the same mass, wherever it lies in the box."""

    m, step, radius = 16, 2.0, 3.0
    lattice = bnb_fft.fft_lattice(m, step, 0)
    kernel = bnb_fft.disc_kernel_hat(m, step, radius)

    def mass_at(points, logs):
        out = np.full(m * m, -60.0, np.float32)
        for p, v in zip(points, logs):
            out[int(np.nonzero(np.all(lattice == np.asarray(p, float), axis=1))[0][0])] = v
        return out

    sharp = mass_at([(10.0, -6.0)], [0.0])
    frame = int(np.asarray(bnb_fft._frames(jnp.asarray(sharp[None]), kernel, m=m))[0])
    np.testing.assert_array_equal(lattice[frame], [10.0, -6.0])
    lump = [(-8.0, 4.0), (-6.0, 4.0), (-8.0, 6.0), (-10.0, 4.0), (-8.0, 2.0)]
    both = mass_at([(10.0, -6.0)] + lump, [0.0] + [np.log(0.4)] * 5)
    frame = int(np.asarray(bnb_fft._frames(jnp.asarray(both[None]), kernel, m=m))[0])
    np.testing.assert_array_equal(lattice[frame], [-8.0, 4.0])


def test_local_masses_wrap_around_the_box():
    """The disc around a frame at the box edge reads the cells across it."""

    m = 8
    mass = np.arange(m * m, dtype=np.float32)[None]
    rows, columns = jnp.asarray([0, 1, -1]), jnp.asarray([0, -1, 1])
    frame = jnp.asarray([7 * m + 0], jnp.int32)  # row 7, column 0
    out = np.asarray(bnb_fft._local_masses(jnp.asarray(mass), frame, rows, columns, m=m))[0]
    np.testing.assert_array_equal(out, [7 * m + 0, 0 * m + 7, 6 * m + 1])


def test_fft_step_is_a_whole_pixel_dividing_the_box():
    """Half RELION's coarse Nyquist spacing rounded down, at most the largest step, at least 1, dividing the box."""

    assert bnb_fft.fft_step(256, 42, 2.0) == 2
    assert bnb_fft.fft_step(128, 42, 2.0) == 1
    assert bnb_fft.fft_step(255, 20, 2.0) == 1
    assert bnb_fft.fft_step(64, 60, 2.0) == 1


def test_lattice_prior_is_flat_or_the_model_at_the_absolute_offset():
    """The lattice prior is zero, or the model's Gaussian at each image's origin plus each shift."""

    lattice = bnb_fft.fft_lattice(8, 2.0, 0)
    origins = np.array([[0.0, 0.0], [5.0, -3.0]])
    flat = bnb_fft.lattice_log_prior(lattice, origins, "flat", pixel_size=2.0, sigma_offset_angstrom=10.0)
    assert flat.shape == (1, 64)
    np.testing.assert_array_equal(np.asarray(flat), 0.0)
    model = np.asarray(bnb_fft.lattice_log_prior(lattice, origins, "model", pixel_size=2.0, sigma_offset_angstrom=10.0))
    expected = -0.5 * np.sum(((origins[:, None, :] + lattice[None]) * 2.0) ** 2, axis=2) / 100.0
    np.testing.assert_allclose(model, expected, rtol=1e-5)


def test_fft_level0_options():
    """The options take lattice or fft and flat or model; an fft level 0 refuses the half band."""

    assert BnBSearchOptions(level0="fft").fft_prior == "model"
    with pytest.raises(ValueError):
        BnBSearchOptions(level0="fft", half_band=True)
    with pytest.raises(ValueError):
        BnBSearchOptions(level0="dense")
