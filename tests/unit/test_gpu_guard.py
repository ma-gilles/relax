"""CPU checks of the guard that keeps local test runs off GPU 0 of the shared node."""

import pytest
from helpers.gpu_guard import cpu_only_reason

pytestmark = pytest.mark.unit

UUIDS = {
    0: "GPU-2c050a78-6cbe-4581-8204-0bdf25d68848",
    1: "GPU-97adb339-219f-d72d-11c9-74dc92fcff8c",
    2: "GPU-ef985070-011e-0782-6f0a-94b053dcc120",
}


@pytest.mark.parametrize(
    "environ",
    [
        {},  # unset outside Slurm: CUDA would pick GPU 0
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"CUDA_VISIBLE_DEVICES": "1,0"},
        {"CUDA_VISIBLE_DEVICES": UUIDS[0]},
        {"CUDA_VISIBLE_DEVICES": "GPU-2c050a78"},  # a UUID prefix, as CUDA accepts
        {"CUDA_VISIBLE_DEVICES": f"{UUIDS[1]},{UUIDS[0]}"},
    ],
)
def test_gpu0_or_unset_outside_slurm_runs_on_cpu(environ):
    assert cpu_only_reason(environ, UUIDS)


@pytest.mark.parametrize(
    "environ",
    [
        {"CUDA_VISIBLE_DEVICES": "1"},
        {"CUDA_VISIBLE_DEVICES": "2,3"},
        {"CUDA_VISIBLE_DEVICES": UUIDS[1]},
        {"CUDA_VISIBLE_DEVICES": ""},  # no GPU visible: nothing to guard
        {"SLURM_JOB_ID": "123"},  # the scheduler's assignment is respected
        {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"},
        {"RELAX_ALLOW_GPU0": "1"},
    ],
)
def test_other_gpus_slurm_and_the_override_keep_their_selection(environ):
    assert cpu_only_reason(environ, UUIDS) is None


def test_without_nvidia_smi_only_index_zero_is_recognised():
    assert cpu_only_reason({"CUDA_VISIBLE_DEVICES": "0"}, {})
    assert cpu_only_reason({"CUDA_VISIBLE_DEVICES": UUIDS[1]}, {}) is None
