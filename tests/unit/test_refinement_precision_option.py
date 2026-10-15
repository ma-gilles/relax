"""The refinement's dense precision is an option the command reads once and passes down (code rule 5)."""

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


def test_from_environ_reads_the_float64_switches(monkeypatch):
    monkeypatch.delenv("RELAX_USE_FLOAT64_SCORING", raising=False)
    monkeypatch.delenv("RELAX_USE_FLOAT64_PROJECTIONS", raising=False)
    assert DensePrecisionPolicy.from_environ() == DensePrecisionPolicy()
    monkeypatch.setenv("RELAX_USE_FLOAT64_SCORING", "1")
    assert DensePrecisionPolicy.from_environ() == DensePrecisionPolicy(use_float64_scoring=True)
    monkeypatch.setenv("RELAX_USE_FLOAT64_PROJECTIONS", "1")
    assert DensePrecisionPolicy.from_environ() == DensePrecisionPolicy(use_float64_scoring=True, use_float64_projections=True)
