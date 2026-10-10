"""RELION reconstruction in the full-box class equals the logical reconstruction.

With the resident engine's stable windows on, _reconstruct_volume_eager
zero-pads the current-size accumulator to the full-box cube and passes
RELION's size as recovar's traced logical_current_size, so one program serves
the run. The padded voxels lie outside every logical support.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape
from relax.reconstruction import volume_solver
from relax.refinement import mean_helpers
from relax.refinement.refinement_options import ReconstructionPrograms

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("current_size", [18, 32])
@pytest.mark.parametrize("half", [False, True])
def test_stable_class_reconstruction_matches_the_logical_one(monkeypatch, half, current_size):
    vol_shape = (32, 32, 32)
    padding_factor = 2
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
    assert volume_solver._stable_reconstruction_class(current_size, vol_shape, padding_factor, accumulator_shape, True)
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", "0")
    logical = mean_helpers._reconstruct_volume_eager(weight, numerator, vol_shape, padding_factor, **kwargs, programs=ReconstructionPrograms.from_environ())
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", "1")
    stable = mean_helpers._reconstruct_volume_eager(weight, numerator, vol_shape, padding_factor, **kwargs, programs=ReconstructionPrograms.from_environ())
    assert_matches(np.asarray(stable), np.asarray(logical))


def test_every_current_size_shares_the_full_box_class_with_a_1d_prior(monkeypatch):
    monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", raising=False)
    vol_shape = (32, 32, 32)
    box_shape = relion_backprojector_volume_shape(vol_shape, 2, current_size=32)
    for current_size in (18, 24, 32):
        shape = relion_backprojector_volume_shape(vol_shape, 2, current_size=current_size)
        assert volume_solver._stable_reconstruction_class(current_size, vol_shape, 2, shape, True) == (32, box_shape)
    shape_18 = relion_backprojector_volume_shape(vol_shape, 2, current_size=18)
    assert volume_solver._stable_reconstruction_class(18, vol_shape, 2, shape_18, False) is None
    assert volume_solver._stable_reconstruction_class(None, vol_shape, 2, shape_18, True) is None
    assert volume_solver._stable_reconstruction_class(18, vol_shape, 2, box_shape, True) is None


@pytest.mark.parametrize("current_size", [18, 24])
@pytest.mark.parametrize("half", [False, True])
def test_unregularized_reconstruction_in_the_box_class_matches_the_logical_one(monkeypatch, half, current_size):
    """No prior, no current_size: padded to the box with recovar's logical_accumulator_size."""

    vol_shape = (32, 32, 32)
    padding_factor = 2
    accumulator_shape = relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=current_size)
    size = accumulator_shape[0]
    shape = (size, size, size // 2 + 1) if half else (size,) * 3
    rng = np.random.default_rng(5)
    weight = jnp.asarray(rng.uniform(0.5, 2.0, size=shape).reshape(-1))
    numerator = jnp.asarray((rng.normal(size=shape) + 1j * rng.normal(size=shape)).reshape(-1))
    kwargs = dict(
        tau=None,
        tau2_fudge=2.0,
        projection_padding_factor=padding_factor,
        accumulator_volume_shape=accumulator_shape,
    )
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", "0")
    assert not ReconstructionPrograms.from_environ().stable_windows
    logical = mean_helpers._reconstruct_volume_eager(weight, numerator, vol_shape, padding_factor, **kwargs, programs=ReconstructionPrograms.from_environ())
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", "1")
    box_shape = relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=32)
    assert volume_solver._stable_unregularized_class(vol_shape, padding_factor, accumulator_shape, None, None) == box_shape
    stable = mean_helpers._reconstruct_volume_eager(weight, numerator, vol_shape, padding_factor, **kwargs, programs=ReconstructionPrograms.from_environ())
    assert_matches(np.asarray(stable), np.asarray(logical))


def test_the_unregularized_box_class_needs_no_prior_and_no_current_size(monkeypatch):
    monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", raising=False)
    vol_shape = (32, 32, 32)
    box_shape = relion_backprojector_volume_shape(vol_shape, 2, current_size=32)
    shape_18 = relion_backprojector_volume_shape(vol_shape, 2, current_size=18)
    assert volume_solver._stable_unregularized_class(vol_shape, 2, shape_18, None, None) == box_shape
    assert volume_solver._stable_unregularized_class(vol_shape, 2, shape_18, jnp.ones(17), None) is None
    assert volume_solver._stable_unregularized_class(vol_shape, 2, shape_18, None, 18) is None
    assert volume_solver._stable_unregularized_class(vol_shape, 2, box_shape, None, None) is None
    assert volume_solver._stable_unregularized_class(vol_shape, 2, None, None, None) is None
