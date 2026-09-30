"""relax's SymList port against RELION's own ``SymList`` (the binding is the oracle)."""

from __future__ import annotations

import numpy as np
import pytest

from relax import symmetry

bind = pytest.importorskip("relax.relion_bind._relion_bind_core")

LABELS = (
    [f"C{n}" for n in (1, 2, 3, 4, 5, 6, 7, 11, 12, 17, 24, 99)]
    + [f"D{n}" for n in (1, 2, 3, 4, 5, 6, 7, 11, 17, 99)]
    + ["T", "O", "I1", "I2", "I3", "I4"]
)


@pytest.mark.parametrize("label", LABELS)
def test_symmetry_operators_match_relion_symlist(label):
    symmetry._operators_float64.cache_clear()
    left, right, point_group, point_group_order = symmetry._operators_float64(label)
    source = bind.get_symmetry_operators(label)
    source_left = np.asarray(source["left"], dtype=np.float64)
    source_right = np.asarray(source["right"], dtype=np.float64)

    assert (point_group, point_group_order) == (int(source["point_group"]), int(source["point_group_order"]))
    assert left.shape == source_left.shape
    assert right.shape == source_right.shape
    # Same operators in the same order; entries agree to libm rounding.
    np.testing.assert_allclose(left, source_left, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(right, source_right, rtol=0.0, atol=1e-14)
    # Exact zeros are where RELION's setSmallValuesToZero put them.
    np.testing.assert_array_equal(right == 0.0, source_right == 0.0)


def test_icosahedral_alias_is_i2():
    assert symmetry.relion_point_group_code("I") == symmetry.relion_point_group_code("I2")
