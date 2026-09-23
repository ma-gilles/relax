"""Batched symmetry sampling retains scalar RELION Euler rows and row order."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("symmetry", ["C1", "D5", "O", "I1"])
@pytest.mark.parametrize("oversampling", [0, 1])
def test_batched_symmetry_rows_match_scalar_source(symmetry, oversampling):
    from relax.relion_bind._relion_bind_core import (
        get_healpix_sampling_metadata,
        get_oversampled_orientations,
        get_oversampled_orientations_batch,
    )

    metadata = get_healpix_sampling_metadata(3, -1.0, symmetry)
    directions = np.array([len(metadata["rot"]) - 1, 0, 0], dtype=np.int64)
    psi = np.array([1, len(metadata["psi"]) - 1, 1], dtype=np.int64)
    perturbation = 0.173
    expected = np.concatenate(
        [
            get_oversampled_orientations(3, oversampling, int(d), int(p), perturbation, symmetry)
            for d, p in zip(directions, psi, strict=True)
        ]
    )
    actual = get_oversampled_orientations_batch(3, oversampling, directions, psi, perturbation, symmetry)
    assert actual.dtype == np.float64  # Preserve RFLOAT metadata before GPU casts.
    assert actual.shape == (3 * 8**oversampling, 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("symmetry", ["D5", "O", "I1"])
@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("oversampling", [0, 1])
def test_batched_symmetry_rows_match_scalar_source_for_random_rows(symmetry, order, oversampling):
    from relax.relion_bind._relion_bind_core import (
        get_healpix_sampling_metadata,
        get_oversampled_orientations,
        get_oversampled_orientations_batch,
    )

    metadata = get_healpix_sampling_metadata(order, -1.0, symmetry)
    rng = np.random.default_rng(order * 10 + oversampling)
    directions = rng.integers(0, len(metadata["rot"]), size=24, dtype=np.int64)
    psi = rng.integers(0, len(metadata["psi"]), size=24, dtype=np.int64)
    perturbation = -0.152378
    expected = np.concatenate(
        [
            get_oversampled_orientations(order, oversampling, int(d), int(p), perturbation, symmetry)
            for d, p in zip(directions, psi, strict=True)
        ]
    )
    actual = get_oversampled_orientations_batch(order, oversampling, directions, psi, perturbation, symmetry)
    np.testing.assert_array_equal(actual, expected)


def test_symmetric_sampling_is_built_once_per_order_and_symmetry():
    """Per-image local-search calls must not rebuild the point-group grid.

    Rebuilding it per row made 10202's I1 order-4 local-search layout take
    11 h per half-iteration; rebuilding it per call (one call per image) cost
    about 100 ms per image for HCN1's C4 order-4 local search.
    """
    from relax.relion_bind._relion_bind_core import (
        clear_symmetric_sampling_cache,
        get_healpix_sampling_metadata,
        get_oversampled_orientations,
        get_oversampled_orientations_batch,
        symmetric_sampling_cache_info,
    )

    metadata = get_healpix_sampling_metadata(4, -1.0, "C4")
    rng = np.random.default_rng(4)
    directions = rng.integers(0, len(metadata["rot"]), size=64, dtype=np.int64)
    psi = rng.integers(0, len(metadata["psi"]), size=64, dtype=np.int64)

    clear_symmetric_sampling_cache()
    for perturbation in (0.1, -0.3, 0.1):
        for image in range(8):
            rows = slice(8 * image, 8 * image + 8)
            get_oversampled_orientations_batch(4, 1, directions[rows], psi[rows], perturbation, "C4")
    get_oversampled_orientations(4, 0, int(directions[0]), int(psi[0]), 0.2, "C4")
    assert symmetric_sampling_cache_info() == {"builds": 1, "entries": 1}

    get_oversampled_orientations_batch(3, 1, directions[:4] % 4, psi[:4] % 4, 0.1, "C4")
    get_oversampled_orientations_batch(4, 1, directions[:4] % 4, psi[:4], 0.1, "D5")
    get_oversampled_orientations_batch(4, 1, directions[:4], psi[:4], 0.1, "C1")
    assert symmetric_sampling_cache_info() == {"builds": 3, "entries": 3}


@pytest.mark.parametrize("symmetry", ["C4", "D5", "I1"])
@pytest.mark.parametrize("oversampling", [0, 1])
def test_cached_symmetric_sampling_rows_equal_a_fresh_build(symmetry, oversampling):
    """Reuse, including after another perturbation, is bitwise a fresh build."""
    from relax.relion_bind._relion_bind_core import (
        clear_symmetric_sampling_cache,
        get_healpix_sampling_metadata,
        get_oversampled_orientations_batch,
        symmetric_sampling_cache_info,
    )

    order = 4
    metadata = get_healpix_sampling_metadata(order, -1.0, symmetry)
    rng = np.random.default_rng(oversampling)
    directions = rng.integers(0, len(metadata["rot"]), size=32, dtype=np.int64)
    psi = rng.integers(0, len(metadata["psi"]), size=32, dtype=np.int64)
    perturbation = 0.2718

    fresh = {}
    for value in (perturbation, -0.5):
        clear_symmetric_sampling_cache()
        fresh[value] = get_oversampled_orientations_batch(order, oversampling, directions, psi, value, symmetry)
        assert symmetric_sampling_cache_info()["builds"] == 1

    cached_other = get_oversampled_orientations_batch(order, oversampling, directions, psi, perturbation, symmetry)
    cached_again = get_oversampled_orientations_batch(order, oversampling, directions, psi, -0.5, symmetry)
    assert symmetric_sampling_cache_info()["builds"] == 1
    for cached, value in ((cached_other, perturbation), (cached_again, -0.5)):
        assert cached.dtype == fresh[value].dtype == np.float64
        np.testing.assert_array_equal(cached, fresh[value])
