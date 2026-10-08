"""The large-grid Wiener solve and FFTW pad run on the CPU backend when the device cannot hold the pad (relax#49)."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.reconstruction import relion_functions as rf

from relax.refinement import mean_helpers

pytestmark = pytest.mark.unit

GIB = 2**30


@pytest.mark.parametrize(
    ("reconstruction_shape", "limit_gib", "on_cpu"),
    [
        ((760,) * 3, 11.24, True),  # box 380 on a 16 GB card: two 1.64 GiB halves against 2.81 GiB
        ((896,) * 3, 10.23, True),  # box 448 on a 16 GB card: two 2.69 GiB halves
        ((760,) * 3, 33.0, False),  # box 380 on a 40 GB card
        ((896,) * 3, 72.0, False),  # box 448 on an 80 GB card
        ((512,) * 3, 11.24, False),  # box 256 on a 16 GB card: two 0.5 GiB halves
    ],
)
def test_pad_moves_to_the_cpu_only_when_two_halves_exceed_the_single_working_set(
    reconstruction_shape, limit_gib, on_cpu
):
    limit = int(limit_gib * GIB)
    assert mean_helpers._relion_pad_exceeds_device_working_set(reconstruction_shape, allocator_limit_bytes=limit) is on_cpu


def test_cpu_route_matches_the_device_route(monkeypatch):
    """The same recovar solve and pad on the CPU backend; on a GPU this compares across backends."""
    import recovar.core.fourier_transform_utils as ftu

    monkeypatch.setenv("RECOVAR_RELION_POSTPROCESS_LARGE_GRID_SINGLE_PRECISION", "auto")
    monkeypatch.setenv("RECOVAR_RELION_POSTPROCESS_SINGLE_PRECISION_MIN_VOXELS", "200")
    volume_shape = (4, 4, 4)
    accumulator_shape = (5, 5, 5)
    half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
    rng = np.random.default_rng(20261008)
    ft_ctf = ftu.half_volume_to_full_volume(
        jnp.asarray(rng.uniform(0.5, 1.5, half_shape).astype(np.float32)), accumulator_shape
    ).reshape(-1)
    ft_y = ftu.half_volume_to_full_volume(
        jnp.asarray((rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)),
        accumulator_shape,
    ).reshape(-1)
    common = dict(
        tau=rng.uniform(0.5, 1.5, np.prod(volume_shape)).astype(np.float64),
        tau2_fudge=1.0,
        minres_map=0,
        current_size=2,
        accumulator_volume_shape=accumulator_shape,
        tau_is_1d=False,
        preserve_output_precision=True,
        relion_filter_scale=float(volume_shape[0] ** 4),
        projection_padding_factor=1,
        use_spherical_mask=True,
        grid_correct=True,
    )
    calls = []
    real = rf.post_process_from_filter_v2
    monkeypatch.setattr(rf, "post_process_from_filter_v2", lambda *a, **k: calls.append(1) or real(*a, **k))

    results = {}
    for route in (False, True):
        monkeypatch.setattr(mean_helpers, "_relion_pad_exceeds_device_working_set", lambda *_a, route=route, **_k: route)
        results[route] = np.asarray(mean_helpers._reconstruct_volume_eager(ft_ctf, ft_y, volume_shape, 2, **common))

    assert len(calls) == 2  # both routes reach the staged pad, not another path
    assert results[True].dtype == results[False].dtype == np.complex64
    assert_matches(results[True], results[False])
