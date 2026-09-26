"""RELION's 3D translation sampling for subtomograms, against loop transcriptions of the C++ (CPU)."""

import math

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax import sampling

pytestmark = pytest.mark.unit


def _relion_set_translations_3d(offset_range, offset_step):
    """HealpixSampling::setTranslations, non-helical is_3d_trans branch (healpix_sampling.cpp:330, 395-418)."""
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
    """HealpixSampling::getTranslationsInPixel, non-helical 3D (healpix_sampling.cpp:1741-1790, 1810-1826)."""
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
    assert_matches(got, _relion_set_translations_3d(offset_range, offset_step), strict=True)


@pytest.mark.parametrize("order, perturbation", [(0, 0.0), (1, 0.0), (1, 0.37), (2, -0.21)])
def test_3d_oversampling_is_relions_in_value_and_order(order, perturbation):
    step, pixel = 4.25, 4.25 * 1.28
    coarse = sampling.get_relion_translation_grid_3d(8.5, step)
    fine, parent = sampling.relion_translations_in_pixel_3d(
        coarse, step, oversampling_order=order, pixel_size=pixel, random_perturbation=perturbation
    )
    expected = [row for t in coarse for row in _relion_translations_in_pixel_3d(t, step, order, pixel, perturbation)]
    assert_matches(fine, np.array(expected), strict=True)
    assert_matches(parent, np.repeat(np.arange(len(coarse)), 8**order))
