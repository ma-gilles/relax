"""Final reconstruction contracts: buffer lifetime and K=4 output assembly."""

import weakref

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import final_reconstruction
from relax.refinement.mean_helpers import ReconstructionSettings
from relax.relion.geometry import (
    IMAGE_MASK_EDGE_PIXELS,
    PROJECTION_PADDING_FACTOR,
    RECONSTRUCTION_PADDING_FACTOR,
    REFERENCE_FILTER_EDGE_SHELLS,
)

pytestmark = pytest.mark.unit


def reconstruction_settings(*, tau2_fudge=1.0):
    return ReconstructionSettings(
        box_size=4,
        voxel_size=1.5,
        volume_shape=(4, 4, 4),
        padding_factor=RECONSTRUCTION_PADDING_FACTOR,
        projection_padding_factor=PROJECTION_PADDING_FACTOR,
        minres_map=5,
        width_mask_edge=IMAGE_MASK_EDGE_PIXELS,
        fmask_edge=REFERENCE_FILTER_EDGE_SHELLS,
        tau2_fudge=tau2_fudge, particle_diameter_angstrom=None,
        first_iteration_lowpass_angstrom=None,
    )


def test_final_halfmaps_release_each_backprojection_before_the_next(monkeypatch):
    backprojections = [
        (np.ones(64, dtype=np.float32), np.full(64, index, dtype=np.complex64))
        for index in range(3)
    ]
    references = [(weakref.ref(ctf), weakref.ref(y)) for ctf, y in backprojections]
    prior = np.ones(64, dtype=np.float32)
    seen = []

    def reconstruct(ctf, y, volume_shape, padding_factor, *, tau, **kwargs):
        index = len(seen)
        assert backprojections[index] is None
        for previous in range(index):
            assert references[previous][0]() is None
            assert references[previous][1]() is None
        assert tau is prior
        assert_matches(kwargs["tau2_fudge"], 1.0)
        assert ctf.dtype == np.float32
        assert y.dtype == np.complex64
        seen.append(index)
        return y.copy().reshape(volume_shape)

    monkeypatch.setattr(final_reconstruction.mean_helpers, "_reconstruct_volume_eager", reconstruct)
    maps = final_reconstruction.reconstruct_final_halfmaps(
        backprojections,
        prior,
        settings=reconstruction_settings(),
        current_size=4,
        accumulator_shape=(8, 8, 8),
    )
    assert seen == [0, 1, 2]
    assert backprojections == [None, None, None]
    assert all(ctf() is None and y() is None for ctf, y in references)
    assert isinstance(maps.merged, np.ndarray)
    assert len(maps.halves) == 2
    assert all(isinstance(value, np.ndarray) and value.dtype == np.complex64 for value in maps.halves)


def test_final_k4_maps_preserve_classes_and_weighted_mean(monkeypatch):
    numerators = np.stack([np.full(64, value, dtype=np.complex64) for value in (0.1, 0.3, 0.2, 0.4)])
    denominators = np.ones_like(numerators.real)
    shells = np.ones((4, 3), dtype=np.float32)
    weights = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    def reconstruct(ctf, y, volume_shape, padding_factor, *, tau, tau_is_1d, **kwargs):
        assert tau_is_1d
        assert tau.dtype == np.float32
        assert_matches(kwargs["tau2_fudge"], 4.0)
        assert np.shares_memory(ctf, denominators)
        assert np.shares_memory(y, numerators)
        return jnp.asarray(y).reshape(volume_shape)

    monkeypatch.setattr(final_reconstruction.mean_helpers, "_reconstruct_volume_eager", reconstruct)
    maps = final_reconstruction.reconstruct_final_class_maps(
        numerators,
        denominators,
        shells,
        class_weights=weights,
        n_classes=4,
        settings=reconstruction_settings(tau2_fudge=4.0),
        current_size=4,
        accumulator_shape=(8, 8, 8),
    )
    assert maps.halves[0] is maps.halves[1]
    assert maps.halves[0].shape == (4, 64)
    expected = np.sum(weights.astype(np.float32)[:, None] * numerators, axis=0)
    assert_matches(maps.merged, expected, rtol=1e-6)
