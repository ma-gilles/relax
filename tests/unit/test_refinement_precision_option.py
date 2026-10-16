"""The refinement's dense precision is an option the command reads once and passes down (code rule 5); the reading
is tested in ``test_refinement_environment.py``."""

import numpy as np
import pytest
from helpers.run_options import stand_in

from relax.numerics.precision import DensePrecisionPolicy

pytestmark = pytest.mark.unit


def test_options_carry_the_given_precision():
    float64 = DensePrecisionPolicy(use_float64_scoring=True)
    options = stand_in.options(precision=float64)
    assert options.precision is float64
    assert options.precision.rotation_real_dtype == np.float64
    assert stand_in.options().precision == DensePrecisionPolicy()
