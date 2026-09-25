"""RELION's 3D translation sampling for subtomograms, against loop transcriptions of the C++ (CPU)."""

import math

import numpy as np
import pytest

from relax import sampling

pytestmark = pytest.mark.unit


def _relion_set_translations_3d(offset_range, offset_step):
    """HealpixSampling::setTranslations, non-helical is_3d_trans branch (healpix_sampling.cpp:345, 410-433)."""
    maxp = math.ceil(offset_range / offset_step)
    out = []
    for ix in range(-maxp, maxp + 1):
        xoff = ix * offset_step
        for iy in range(-maxp, maxp + 1):
            yoff = iy * offset_step
            max2 = xoff * xoff + yoff * yoff
            for iz in range(-maxp, maxp + 1):
                zoff = iz * offset_step
                if (max2 + zoff * zoff) <= (offset_range * offset_range):
                    out.append((xoff, yoff, zoff))
    return np.array(out)


def _relion_translations_in_pixel_3d(t, offset_step, order, pixel_size, perturbation):
    """HealpixSampling::getTranslationsInPixel, non-helical 3D (healpix_sampling.cpp:1756-1805, 1825-1841)."""
    out = []
    if order == 0:
        out.append([t[0] / pixel_size, t[1] / pixel_size, t[2] / pixel_size])
    else:
        n = round(2.0**order)
        for kx in range(n):
            ox = t[0] - 0.5 * offset_step + (0.5 + kx) * offset_step / n
            for ky in range(n):
                oy = t[1] - 0.5 * offset_step + (0.5 + ky) * offset_step / n
                for kz in range(n):
                    oz = t[2] - 0.5 * offset_step + (0.5 + kz) * offset_step / n
                    out.append([ox / pixel_size, oy / pixel_size, oz / pixel_size])
    if abs(perturbation) > 0.0:
        p = perturbation * offset_step / pixel_size
        out = [[x + p, y + p, z + p] for x, y, z in out]
    return out


@pytest.mark.parametrize("offset_range, offset_step", [(6.0, 3.0), (8.5, 4.25), (5.0, 2.0), (0.0, 2.0)])
def test_3d_grid_is_relions_enumeration_and_keeps_the_boundary(offset_range, offset_step):
    got = sampling.get_relion_translation_grid_3d(offset_range, offset_step)
    np.testing.assert_array_equal(got, _relion_set_translations_3d(offset_range, offset_step))


@pytest.mark.parametrize("order, perturbation", [(0, 0.0), (1, 0.0), (1, 0.37), (2, -0.21)])
def test_3d_oversampling_is_relions_in_value_and_order(order, perturbation):
    step, pixel = 4.25, 4.25 * 1.28
    coarse = sampling.get_relion_translation_grid_3d(8.5, step)
    fine, parent = sampling.relion_translations_in_pixel_3d(
        coarse, step, oversampling_order=order, pixel_size=pixel, random_perturbation=perturbation
    )
    expected = [row for t in coarse for row in _relion_translations_in_pixel_3d(t, step, order, pixel, perturbation)]
    np.testing.assert_array_equal(fine, np.array(expected))
    np.testing.assert_array_equal(parent, np.repeat(np.arange(len(coarse)), 8**order))
