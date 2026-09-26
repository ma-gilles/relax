"""K=1 tau2 volumes are parked on the host between M-steps (bigbox census, EMPIAR-10202)."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement.iteration_loop import _host_tau2_volumes

pytestmark = pytest.mark.unit


def test_host_tau2_volumes_moves_every_volume_and_keeps_shared_names_shared():
    shared = jnp.arange(8, dtype=jnp.float32)
    half0 = jnp.arange(8, dtype=jnp.float32) * 2
    half1 = jnp.arange(8, dtype=jnp.float32) * 3
    mean_variance, per_half, signal, signal_per_half = _host_tau2_volumes(
        shared, [shared, shared], shared, [half0, half1]
    )
    assert all(isinstance(v, np.ndarray) for v in (mean_variance, signal, *per_half, *signal_per_half))
    # One host copy per device array: names that shared an array still share it.
    assert mean_variance is signal and per_half[0] is mean_variance and per_half[1] is mean_variance
    assert signal_per_half[0] is not signal_per_half[1]
    assert_matches(signal_per_half[1], np.arange(8, dtype=np.float32) * 3)
    assert mean_variance.dtype == np.float32


def test_host_tau2_volumes_passes_host_arrays_and_none_through():
    host = np.ones(4, dtype=np.float32)
    out = _host_tau2_volumes(host, [host, host], host, None)
    assert out[0] is host and out[1][0] is host and out[2] is host and out[3] is None
