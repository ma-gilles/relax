"""Class3D never runs RELION's auto-refine sampling or convergence transitions."""

import pytest

from relax.refinement.convergence import uses_native_auto_refine

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "native,k_class,expected",
    [(True, False, True), (True, True, False), (False, False, False), (False, True, False)],
)
def test_only_native_k1_iterations_own_sampling_and_convergence(native, k_class, expected):
    assert uses_native_auto_refine(native_sampling_boundary=native, n_classes=4 if k_class else 1) is expected
