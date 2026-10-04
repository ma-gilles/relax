"""Focused lifetime guards for K=1 references between EM iterations."""

import inspect
import weakref
from pathlib import Path

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import iteration_loop, mean_helpers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128], ids=["production-f32", "diagnostic-f64"])
def test_snapshot_and_release_previous_k1_means_owns_host_copies(complex_dtype):
    first = np.arange(12, dtype=np.float64).astype(complex_dtype).reshape(3, 4)
    second = (first + complex_dtype(2.0 + 3.0j)).copy()
    means = [first, second]
    snapshots = mean_helpers._snapshot_and_release_previous_k1_means(means)

    assert means == [None, None]
    for snapshot, original in zip(snapshots, (first, second), strict=True):
        assert type(snapshot) is np.ndarray
        assert snapshot.flags.owndata
        assert not np.shares_memory(snapshot, original)
        assert_matches(snapshot, original)

    first[...] = complex_dtype(-9.0 + 4.0j)
    assert not np.array_equal(snapshots[0], first)


def test_k1_mean_release_precedes_tau_and_reconstruction():
    source = inspect.getsource(iteration_loop.refine_single_volume)
    initial_alias_release = source.index("del init_volume")
    maximization_call = source.index("k1_maximization(", initial_alias_release)
    maximization = inspect.getsource(iteration_loop.k1_maximization)
    release = maximization.index(
        "previous_means = _snapshot_and_release_previous_k1_means(reference_model.maps)"
    )
    tau_update = maximization.index(
        "estimate_split_half_prior(",
        release,
    )
    reconstruction = maximization.index(
        "reconstruct_numbered_k1_halfmaps(",
        tau_update,
    )

    assert initial_alias_release < maximization_call
    assert release < tau_update < reconstruction
    operation = inspect.getsource(mean_helpers.estimate_split_half_prior)
    assert operation.index("compute_relion_fsc_from_backprojector(") < operation.index("compute_relion_tau2_from_weights(")


def test_production_runner_leaves_cold_start_host_owned_until_normalization():
    repo_root = Path(iteration_loop.__file__).resolve().parents[2]
    runner_source = (repo_root / "relax" / "refinement" / "full_refinement.py").read_text()
    call_start = runner_source.index("result = refine_single_volume(")
    call_stop = runner_source.index("options=RefinementOptions(", call_start)
    production_call = runner_source[call_start:call_stop]

    assert "init_volume=init_vol_ft," in production_call
    assert "init_volume=jnp.asarray(init_vol_ft)" not in production_call


def test_normalize_initial_means_reuses_immutable_shared_reference():
    import jax.numpy as jnp

    shared = jnp.arange(64, dtype=jnp.complex64)
    got = mean_helpers._normalize_initial_means(shared, n_classes=1)

    assert got[0] is got[1]
    assert_matches(np.asarray(got[0]), np.asarray(shared))
    got[0] = got[0].at[0].set(jnp.complex64(-1.0))
    assert got[0] is not got[1]
    assert_matches(np.asarray(got[1]), np.asarray(shared))


@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128], ids=["production-f32", "diagnostic-f64"])
def test_reference_owner_keeps_no_extra_map_alias_after_k1_release(complex_dtype):
    expected = np.arange(12, dtype=np.float64).astype(complex_dtype)
    model = mean_helpers.ReferenceModel(
        maps=[expected.copy(), expected.copy()], tau2=None, tau2_per_half=[None, None],
    )
    references = [weakref.ref(value) for value in model.maps]
    previous = mean_helpers._snapshot_and_release_previous_k1_means(model.maps)
    assert model.maps == [None, None]
    assert all(reference() is None for reference in references)
    for value in previous:
        assert_matches(value, expected)
