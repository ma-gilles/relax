"""RELION reconstruction in a stable window class equals the logical reconstruction.

With the resident engine's stable windows on, _reconstruct_volume_eager
zero-pads the current-size accumulator to its physical class and passes
RELION's size as recovar's traced logical_current_size, so one program serves
the class. The padded voxels lie outside every logical support.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape
from relax.refinement import mean_helpers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("half", [False, True])
def test_stable_class_reconstruction_matches_the_logical_one(monkeypatch, half):
    vol_shape = (32, 32, 32)
    padding_factor, current_size = 2, 18  # physical class 24 at the quantum-8 ladder
    accumulator_shape = relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=current_size)
    size = accumulator_shape[0]
    shape = (size, size, size // 2 + 1) if half else (size,) * 3
    rng = np.random.default_rng(3)
    weight = jnp.asarray(rng.uniform(0.5, 2.0, size=shape).reshape(-1))
    numerator = jnp.asarray((rng.normal(size=shape) + 1j * rng.normal(size=shape)).reshape(-1))
    tau = jnp.asarray(rng.uniform(0.5, 1.5, size=vol_shape[0] // 2 + 1))
    kwargs = dict(
        tau=tau,
        tau2_fudge=2.0,
        projection_padding_factor=padding_factor,
        current_size=current_size,
        accumulator_volume_shape=accumulator_shape,
        tau_is_1d=True,
    )
    assert mean_helpers._stable_reconstruction_class(current_size, vol_shape, padding_factor, accumulator_shape, True)
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", "0")
    logical = mean_helpers._reconstruct_volume_eager(weight, numerator, vol_shape, padding_factor, **kwargs)
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", "1")
    stable = mean_helpers._reconstruct_volume_eager(weight, numerator, vol_shape, padding_factor, **kwargs)
    assert_matches(np.asarray(stable), np.asarray(logical))


def test_stable_class_needs_a_sub_box_current_size_and_a_1d_prior(monkeypatch):
    monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", raising=False)
    vol_shape = (32, 32, 32)
    shape_18 = relion_backprojector_volume_shape(vol_shape, 2, current_size=18)
    assert mean_helpers._stable_reconstruction_class(18, vol_shape, 2, shape_18, True)[0] == 24
    assert mean_helpers._stable_reconstruction_class(18, vol_shape, 2, shape_18, False) is None
    assert mean_helpers._stable_reconstruction_class(None, vol_shape, 2, shape_18, True) is None
    shape_24 = relion_backprojector_volume_shape(vol_shape, 2, current_size=24)
    assert mean_helpers._stable_reconstruction_class(24, vol_shape, 2, shape_24, True) is None
