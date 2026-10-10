"""A failed expected-accuracy estimate raises instead of setting an infinite accuracy.

An infinite accuracy hid a device out-of-memory behind a changed convergence decision
(EMPIAR-10202 final all-data pass, gpuport 15003773).
"""

import logging

import numpy as np
import pytest

from relax.sampling import expected_accuracy


class _FailingInputs:
    trial_order_local = np.arange(4)
    dataset = type("Dataset", (), {"n_units": 4})()

    def estimate(self, **kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 15.35GiB.")


def test_numbered_iteration_estimate_failure_raises():
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        expected_accuracy.estimate_iteration_accuracy(
            _FailingInputs(),
            np.zeros(8, dtype=np.complex64),
            best_eulers_deg=np.zeros((4, 3)),
            class_assignments=None,
            class_weights=np.ones(1),
            sigma2_noise_native=np.ones(3),
            current_size=8,
            image_box_size=8,
            n_classes=1,
            iteration=3,
            native_sampling_boundary=True,
            relion_firstiter_cc_this_iter=False,
            build_shared_projector=False,
            log=logging.getLogger(__name__),
        )
