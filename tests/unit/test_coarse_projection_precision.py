"""Consumer precision must precede projection, not merely narrow scores."""
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.scoring.pass1_plan import _prepare_coarse_relion_projector

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("classes", [1, 4])
@pytest.mark.parametrize("projection_double", [None, False])
def test_coarse_projector_precision_preserves_source(classes, projection_double):
    source = np.arange(classes * 7 * 7 * 4).reshape(classes, 7, 7, 4).astype(np.complex128)
    source += 2**-27 + 1j * 2**-26
    saved = source.copy()
    result = _prepare_coarse_relion_projector(source, use_float64_projections=projection_double)
    expected_dtype = np.complex128 if projection_double is None else np.complex64
    assert result.dtype == expected_dtype
    assert_matches(np.asarray(result), source.astype(expected_dtype))
    assert_matches(source, saved)


def test_unspecified_projection_precision_preserves_legacy_float32_input():
    source = np.zeros((1, 7, 7, 4), dtype=np.complex64)
    result = _prepare_coarse_relion_projector(source, use_float64_projections=None)
    assert result.dtype == np.complex64
