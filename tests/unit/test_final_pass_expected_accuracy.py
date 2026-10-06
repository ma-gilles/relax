"""The final all-data pass's expected-accuracy estimate: a failure stops the run (code rule 12)."""

import numpy as np
import pytest
from helpers.tiny_refinement import N_IMAGES, run_tiny_refinement

from relax.helpers import expected_accuracy
from relax.refinement.refinement_options import EngineDebugOptions, ExpectedAccuracyOptions

pytestmark = pytest.mark.unit


def test_a_failing_final_pass_accuracy_estimate_is_an_error_not_a_status(monkeypatch):
    calls = []

    def estimate(self, **operands):
        calls.append(operands)
        raise RuntimeError("estimate failed")

    monkeypatch.setattr(expected_accuracy.Half1AccuracyInputs, "estimate", estimate)
    trial_order = ExpectedAccuracyOptions(half1_trial_order_local=np.arange(N_IMAGES // 2))
    with pytest.raises(RuntimeError, match="estimate failed"):
        run_tiny_refinement(monkeypatch, max_iter=1, debug=EngineDebugOptions(expected_accuracy=trial_order))
    # A first numbered iteration estimates nothing; the one call is the final pass's (it passes no shared projector).
    assert len(calls) == 1 and "projector_data" not in calls[0]
