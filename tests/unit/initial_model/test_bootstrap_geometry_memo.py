"""The memoized start-up geometry gives the bootstrap's previous per-call arithmetic.

relax.vdam.bootstrap_reconstruction builds the soft-mask radius and weights, the reconstruct
shell indices, the windowing sign and the gridding sinc once per grid. The references below
are the per-call formulas they replaced; outputs must match them, on a cold cache and on a hit.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.relion.macros import PI, relion_round
from relax.vdam import bootstrap_reconstruction as br

pytestmark = pytest.mark.unit


def _reference_soft_mask(volume, radius=-1.0, cosine_width=3.0):
    vol = np.asarray(volume, dtype=np.float64)
    axes = [br._logical_axis(size) for size in vol.shape]
    grids = np.meshgrid(*axes, indexing="ij")
    r = np.sqrt(sum(g.astype(np.int64) * g for g in grids).astype(np.float64))
    if radius < 0:
        radius = float(vol.shape[-1]) / 2.0
    radius_p = radius + cosine_width
    outside = r > radius_p
    edge = ~outside & ~(r < radius)
    raised = np.where(edge, 0.5 + 0.5 * np.cos(PI * (radius_p - r) / cosine_width), 0.0)
    weight = np.where(outside, 1.0, raised).reshape(-1)
    weighted = np.where(outside, vol, raised * vol).reshape(-1)
    counted = outside.reshape(-1) | edge.reshape(-1)
    total = np.cumsum(weight[counted])[-1]
    background = np.cumsum(weighted[counted])[-1] / total
    out = np.where(outside, background, vol)
    return np.where(edge, (1 - raised) * vol + raised * background, out)


def _reference_reconstruct(bp):
    pf, r_max, n = bp.padding_factor, bp.r_max, bp.pad_size
    half = n // 2 + 1
    max_r2 = relion_round(r_max * pf) * relion_round(r_max * pf)
    kp = br._fftw_axis(n, half)[:, None, None]
    ip = br._fftw_axis(n, half)[None, :, None]
    jp = np.arange(half)[None, None, :]
    r2 = kp * kp + ip * ip + jp * jp
    inside = r2 <= max_r2
    centre = n // 2
    source = (kp + centre, ip + centre, np.broadcast_to(jp, r2.shape))
    f_weight = np.where(inside, bp.weight[source], 0.0)
    f_conv = np.where(inside, bp.data[source], 0.0)
    round_max_r2 = relion_round(r_max * pf * r_max * pf)
    shell = np.floor(np.sqrt(r2.astype(np.float64)) / pf).astype(np.int64)
    within = r2 < round_max_r2
    radavg = np.bincount(shell[within], weights=f_weight[within], minlength=r_max)[:r_max]
    counter = np.bincount(shell[within], minlength=r_max)[:r_max].astype(np.float64)
    radavg = radavg / (1000.0 * counter)
    floor = radavg[np.minimum(shell, r_max - 1)]
    weight = np.maximum(f_weight, floor)
    f_conv = np.where(weight == 0.0, f_conv, f_conv / np.where(weight == 0.0, 1.0, weight))
    ori = bp.box_size
    padoridim = relion_round(pf * ori)
    padoridim += padoridim % 2
    new_half = padoridim // 2 + 1
    out = np.zeros((padoridim, padoridim, new_half), dtype=np.complex128)
    kq, iq, jq = br._fftw_axis(n, half), br._fftw_axis(n, half), np.arange(half)
    if new_half > half:
        r2w = kq[:, None, None] ** 2 + iq[None, :, None] ** 2 + jq[None, None, :] ** 2
        k_idx, i_idx, j_idx = np.nonzero(r2w <= (half - 1) * (half - 1))
        out[kq[k_idx] % padoridim, iq[i_idx] % padoridim, jq[j_idx]] = f_conv[k_idx, i_idx, j_idx]
    else:
        ko = br._fftw_axis(padoridim, new_half)
        out = f_conv[(ko % n)[:, None, None], (ko % n)[None, :, None], np.arange(new_half)[None, None, :]]
    out = out * (-1.0) ** (np.add.outer(np.add.outer(np.arange(padoridim), np.arange(padoridim)), np.arange(new_half)))
    real = np.fft.irfftn(out, s=(padoridim,) * 3, axes=(0, 1, 2)) * float(padoridim) ** 3
    start = padoridim // 2 + int(-int(float(ori) / 2.0))
    real = real[start : start + ori, start : start + ori, start : start + ori] / float(pf * pf * pf * ori)
    real = _reference_soft_mask(real)
    k, i, j = np.meshgrid(*[br._logical_axis(ori)] * 3, indexing="ij")
    r = np.sqrt((k * k + i * i + j * j).astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        rval = r / (ori * pf)
        sinc = np.sin(PI * rval) / (PI * rval)
        corrected = real / (sinc * sinc)
    return np.where(r > 0.0, corrected, real)


@pytest.mark.parametrize(
    "shape, radius, width", [((40, 40), 12.5, 5.0), ((24, 24, 24), -1.0, 3.0), ((24, 24, 24), 7.0, 4.0)]
)
def test_soft_mask_matches_the_per_call_formula_cold_and_cached(shape, radius, width):
    br.clear_geometry_caches()
    rng = np.random.default_rng(5)
    for _ in range(2):  # the second call reads the memoized geometry
        volume = rng.standard_normal(shape)
        assert_matches(br.soft_mask_outside_map(volume, radius, width), _reference_soft_mask(volume, radius, width))


@pytest.mark.parametrize("padding_factor", [1, 2])
def test_reconstruct_matches_the_per_call_formula_for_every_class(padding_factor):
    br.clear_geometry_caches()
    rng = np.random.default_rng(9)
    for _ in range(3):  # classes after the first hit the cache
        bp = br.BackProjector3D(16, padding_factor, -1)
        bp.data[...] = rng.standard_normal(bp.data.shape) + 1j * rng.standard_normal(bp.data.shape)
        bp.weight[...] = np.abs(rng.standard_normal(bp.weight.shape))
        assert_matches(bp.reconstruct(), _reference_reconstruct(bp))


def test_window_sign_equals_the_float_power():
    br.clear_geometry_caches()
    _, sign = br._window_geometry(19, 10, 18)
    power = (-1.0) ** np.add.outer(np.add.outer(np.arange(18), np.arange(18)), np.arange(10))
    assert_matches(sign, power)
