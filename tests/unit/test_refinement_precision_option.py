"""The refinement's dense precision is an option taken when the options are built (code rule 5)."""

from dataclasses import replace

import numpy as np
import pytest
from helpers.run_options import stand_in

from relax.dense import scoring_policy
from relax.helpers.dtype_policy import DensePrecisionPolicy

pytestmark = pytest.mark.unit


def test_options_take_the_process_precision_when_built(monkeypatch):
    float64 = DensePrecisionPolicy(use_float64_scoring=True)
    monkeypatch.setattr(scoring_policy, "DENSE_PRECISION", float64)
    options = stand_in.options()
    assert options.precision is float64
    assert options.precision.rotation_real_dtype == np.float64


def test_refinement_refuses_a_precision_the_engines_do_not_read():
    from relax.refinement import iteration_loop

    process = scoring_policy.DENSE_PRECISION
    other = replace(process, use_float64_scoring=not process.use_float64_scoring)
    with pytest.raises(ValueError, match="differs from the process's dense precision"):
        iteration_loop.refine_single_volume(*([None] * 5), options=stand_in.options(precision=other))
