"""Class3D never runs RELION's auto-refine sampling or convergence transitions."""

import pytest

from relax.refinement.iteration_loop import _relion_auto_refine_transitions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "native,k_class,expected",
    [(True, False, True), (True, True, False), (False, False, False), (False, True, False)],
)
def test_only_native_k1_iterations_own_sampling_and_convergence(native, k_class, expected):
    assert _relion_auto_refine_transitions(native_sampling_boundary=native, k_class_enabled=k_class) is expected
