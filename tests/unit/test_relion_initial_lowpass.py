"""Numerical contracts for RELION-compatible initial low-pass filtering."""

import jax.numpy as jnp
import numpy as np
import pytest
from recovar.core import fourier_transform_utils as ftu

from relax.refinement.map_postprocess import _apply_relion_initial_lowpass_filter
from relax.relion.reference_initialization import initial_low_pass_filter_references
from relax.relion.relion_metadata import (
    read_relion_mrc_model_pixel_size,
)

pytestmark = pytest.mark.unit


def test_model_pixel_size_uses_binary64_header_division(tmp_path):
    import mrcfile

    path = tmp_path / "model.mrc"
    with mrcfile.new(path) as handle:
        handle.set_data(np.zeros((6, 6, 6), dtype=np.float32))
        handle.header.cella.x = 17.0
        handle.header.cella.y = 17.0
        handle.header.cella.z = 17.0
        handle.header.mx = 12
        handle.header.my = 12
        handle.header.mz = 12

    expected = 17.0 / 12.0
    assert read_relion_mrc_model_pixel_size(path) == expected
    with mrcfile.open(path, permissive=False) as handle:
        assert float(handle.voxel_size.x) != expected


def test_centered_fft_wrapper_matches_relion_lowpass():
    rng = np.random.default_rng(17)
    volume = rng.normal(size=(1, 8, 8, 8)).astype(np.float32)
    volume_ft = np.asarray(ftu.get_dft3(jnp.asarray(volume[0]))).reshape(-1)

    actual_ft = _apply_relion_initial_lowpass_filter(
        volume_ft,
        (8, 8, 8),
        voxel_size=1.25,
        ini_high_angstrom=4.0,
        filter_edgewidth=2.0,
    )
    actual = np.asarray(ftu.get_idft3(jnp.asarray(actual_ft).reshape(8, 8, 8))).real
    expected = initial_low_pass_filter_references(
        volume.astype(np.float64),
        box_size=8,
        pixel_size=1.25,
        ini_high_ang=4.0,
        filter_edgewidth=2.0,
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n, ini_high", [(64, 9.0), (65, 9.0), (64, 30.0)])
def test_lowpass_in_place_slabs_match_the_whole_mask_products(monkeypatch, n, ini_high):
    """initialLowPassFilterReferences builds its mask a slab of rows at a time and multiplies in place (relax#39);
    the values are those of the whole-mask, out-of-place products ``irfftn(rfftn(v) / size * mask * size)``.
    The slab is forced below the volume so several slabs, and the taper inside them, are exercised."""

    import relax.relion.reference_initialization as ri

    rng = np.random.default_rng(n)
    refs = rng.normal(size=(2, n, n, n))
    edge_width, box, pixel = 2.0, n, 1.5
    radius = box * pixel / ini_high - edge_width / 2.0
    radius_p = radius + edge_width
    kz = np.fft.fftfreq(n, d=1.0) * n
    kx = np.arange(n // 2 + 1, dtype=np.float64)
    r = np.sqrt(kz[:, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2)
    mask = np.where(r < radius, 1.0, 0.0)
    edge = (r >= radius) & (r <= radius_p)
    mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / edge_width)
    expected = np.stack(
        [np.fft.irfftn(np.fft.rfftn(v) / v.size * mask * v.size, s=v.shape) for v in refs]
    )
    assert 0 < np.count_nonzero(edge)

    monkeypatch.setattr(ri, "_MASK_SLAB_ELEMENTS", 5 * n * (n // 2 + 1))  # slabs of 5 rows
    got = ri.initial_low_pass_filter_references(
        refs, box_size=box, pixel_size=pixel, ini_high_ang=ini_high, filter_edgewidth=edge_width
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14 * np.abs(expected).max())
