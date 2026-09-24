"""Loading optics groups on several image shapes (S3b, CPU).

Checks the start-up noise image path RELION takes for a group on another grid
(mask with its own pixel size, ``resizeMap`` to the model pixel size, window to the
model box; ml_optimiser.cpp:2934-2955) against independent references, and that
the multi-shape dataset returns rows by index.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import starfile

from relax.refinement.optics_shapes import MultiShapeDataset, MultiShapeHalf
from relax.relion import initial_noise
from scripts import run_full_refinement as driver

pytestmark = pytest.mark.unit


def _band_limited(size, modes, pixel):
    """A real periodic image of fixed physical frequencies, sampled at ``pixel`` A."""
    x = np.arange(size) * pixel
    image = np.zeros((size, size))
    for (ky, kx), amplitude, phase in modes:
        period = 64.0 * 4.0  # common physical period (A) of every grid below
        image += amplitude * np.cos(2 * np.pi * (ky * x[:, None] + kx * x[None, :]) / period + phase)
    return image


MODES = [((0, 0), 0.3, 0.0), ((1, 2), 1.0, 0.4), ((3, -2), 0.5, -1.1), ((-5, 4), 0.25, 2.0)]


@pytest.mark.parametrize("old_size, new_size", [(64, 80), (64, 48)])
def test_resize_map_resamples_a_band_limited_image(old_size, new_size):
    # Fourier resampling of a band-limited periodic image is exact: the resized image
    # equals the same function sampled on the new grid (origin at array index 0).
    old_pixel = 4.0 * 64 / old_size
    new_pixel = old_pixel * old_size / new_size
    got = initial_noise._relion_resize_map(_band_limited(old_size, MODES, old_pixel), new_size)
    np.testing.assert_allclose(got, _band_limited(new_size, MODES, new_pixel), atol=1e-12)


def test_resize_map_drops_corners_when_enlarging():
    # windowFourierTransform keeps only k**2 <= (N/2)**2 when enlarging (fftw.h:840-849).
    corner = _band_limited(16, [((6, 6), 1.0, 0.0)], 4.0 * 64 / 16)
    assert np.max(np.abs(initial_noise._relion_resize_map(corner, 24))) < 1e-12


@pytest.mark.parametrize("size, ori", [(20, 16), (12, 16)])
def test_model_grid_window_keeps_the_centre(size, ori):
    image = np.zeros((size, size))
    image[size // 2, size // 2] = 1.0
    image[size // 2 - 3, size // 2 + 2] = 2.0
    out = initial_noise._rescale_to_model_grid(image, 2.0, 2.0, ori)
    assert out.shape == (ori, ori)
    assert out[ori // 2, ori // 2] == 1.0 and out[ori // 2 - 3, ori // 2 + 2] == 2.0
    assert out.sum() == 3.0


def test_group_on_another_grid_follows_relion_resize_then_window():
    rng = np.random.default_rng(3)
    ori, model_pixel = 16, 2.0
    same = rng.normal(size=(ori, ori))
    other = rng.normal(size=(14, 14))
    other_pixel = 2.3  # ROUND(14 * 2.3 / 2.0) = 16, already even
    _avg, got = initial_noise.compute_avg_unaligned_and_sigma2(
        iter([(0, same), (1, other)]),
        ori_size=ori,
        pixel_size=model_pixel,
        particle_diameter_ang=20.0,
        width_mask_edge_px=2,
        do_zero_mask=True,
        nr_optics_groups=2,
        minimum_nr_particles=1,
        group_pixel_sizes=[model_pixel, other_pixel],
        model_pixel_size=model_pixel,
    )
    _avg, expected = initial_noise.compute_avg_unaligned_and_sigma2(
        iter(
            [
                (0, initial_noise._softmask_outside_map(same, 20.0 / (2 * model_pixel), 2.0)),
                (
                    1,
                    initial_noise._relion_resize_map(
                        initial_noise._softmask_outside_map(other, 20.0 / (2 * other_pixel), 2.0), 16
                    ),
                ),
            ]
        ),
        ori_size=ori,
        pixel_size=model_pixel,
        particle_diameter_ang=20.0,
        width_mask_edge_px=2,
        do_zero_mask=False,
        nr_optics_groups=2,
        minimum_nr_particles=1,
    )
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12)
    # Every image is masked with its own group's pixel size before it is resized.
    _avg, single = initial_noise.compute_avg_unaligned_and_sigma2(
        iter(
            [
                (0, initial_noise._softmask_outside_map(same, 20.0 / (2 * model_pixel), 2.0)),
                (
                    1,
                    initial_noise._rescale_to_model_grid(
                        initial_noise._softmask_outside_map(other, 20.0 / (2 * other_pixel), 2.0),
                        other_pixel,
                        model_pixel,
                        ori,
                    ),
                ),
            ]
        ),
        ori_size=ori,
        pixel_size=model_pixel,
        particle_diameter_ang=20.0,
        width_mask_edge_px=2,
        do_zero_mask=False,
        nr_optics_groups=2,
        minimum_nr_particles=1,
    )
    np.testing.assert_allclose(got, single, rtol=1e-12)


class _Source:
    def __init__(self, images):
        self.images = images

    def iter_batches(self, *, batch_size, batch_mode, subset_indices):
        assert batch_mode == "images"
        rows = np.asarray(subset_indices)
        for start in range(0, rows.size, batch_size):
            part = rows[start : start + batch_size]
            yield self.images[part], part, part


class _Dataset(SimpleNamespace):
    def subset(self, rows):
        return _Dataset(
            image_shape=self.image_shape,
            volume_shape=self.volume_shape,
            voxel_size=self.voxel_size,
            n_units=len(rows),
            rows=np.asarray(rows),
            image_source=None,
        )


def _dataset(box, pixel, n, offset):
    images = offset + np.arange(n)[:, None, None] + np.zeros((n, box, box))
    return _Dataset(
        image_shape=(box, box), volume_shape=(box,) * 3, voxel_size=pixel, n_units=n, image_source=_Source(images)
    )


def test_multi_shape_dataset_subsets_and_iterates_by_row():
    rows = [np.array([0, 2, 3, 6]), np.array([1, 4, 5])]
    ds = MultiShapeDataset([_dataset(16, 2.0, 4, 100.0), _dataset(12, 2.5, 3, 200.0)], rows)
    assert ds.n_units == 7 and ds.image_shape == (16, 16) and ds.voxel_size == 2.0
    with pytest.raises(AttributeError):
        ds.image_source
    half = ds.subset(np.array([5, 0, 6, 1]))
    assert isinstance(half, MultiShapeHalf) and half.n_units == 4
    # Diagnostics map a half's images back to particle-STAR rows through the index layout.
    np.testing.assert_array_equal(half._index_layout.original_image_indices_for_local(np.arange(4)), [5, 0, 6, 1])
    first, second = half.classes
    np.testing.assert_array_equal(first.image_indices, [1, 2])  # rows 0, 6
    np.testing.assert_array_equal(first.dataset.rows, [0, 3])
    np.testing.assert_array_equal(second.image_indices, [0, 3])  # rows 5, 1
    np.testing.assert_array_equal(second.dataset.rows, [2, 0])
    assert second.scale == pytest.approx(12 * 2.5 / (16 * 2.0))
    got = list(ds.iter_images([5, 0, 6, 1, 3], batch_size=2))
    assert [row for row, _ in got] == [5, 0, 6, 1, 3]
    assert [float(image.flat[0]) for _, image in got] == [202.0, 100.0, 103.0, 200.0, 102.0]
    assert [image.shape for _, image in got] == [(12, 12), (16, 16), (16, 16), (12, 12), (16, 16)]


def test_optics_shape_class_rows(tmp_path):
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1, 2, 3],
            "rlnImagePixelSize": [4.25, 5.44, 4.25],
            "rlnImageSize": [128, 112, 128],
        }
    )
    particles = pd.DataFrame(
        {"rlnImageName": [f"{i}@a.mrcs" for i in range(1, 7)], "rlnOpticsGroup": [2, 1, 3, 2, 1, 1]}
    )
    path = tmp_path / "particles.star"
    starfile.write({"optics": optics, "particles": particles}, path)
    rows = driver._optics_shape_class_rows(path)
    assert [r.tolist() for r in rows] == [[1, 2, 4, 5], [0, 3]]
    starfile.write({"optics": optics.assign(rlnImagePixelSize=4.25, rlnImageSize=128), "particles": particles}, path)
    assert driver._optics_shape_class_rows(path) is None
