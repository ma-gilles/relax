"""Projection precision is resolved before host slab upload."""
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.relion.relion_projector_setup import cast_relion_projector_for_execution


@pytest.mark.parametrize("classes", [1, 4])
@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("host", [False, True])
def test_local_projector_execution_precision(classes, double, host):
    setup = (np.arange(classes * 7 * 7 * 4).reshape(classes, 7, 7, 4)
             + 1j / 7).astype(np.complex128)
    before = setup.copy()
    source = setup if host else jnp.asarray(setup)
    output = cast_relion_projector_for_execution(source, use_float64_projections=double)
    dtype = np.complex128 if double else np.complex64
    assert output.dtype == dtype
    assert output.shape == setup.shape
    assert isinstance(output, np.ndarray) == host
    assert_matches(np.asarray(output), setup.astype(dtype))
    assert_matches(setup, before)


def test_no_local_relion_projector():
    assert cast_relion_projector_for_execution(None) is None


