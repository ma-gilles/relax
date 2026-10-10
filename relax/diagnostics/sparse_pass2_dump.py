"""Per-stage timing of the sparse pass-2 image preparation (``sparse_pass2_bucket_io``).

Diagnostic only; none of it changes production arithmetic.
"""

from __future__ import annotations

import time

import jax.numpy as jnp

from relax.runtime.env_flags import parse_env_flag


def _add_sparse_group_timing(group_timing: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if group_timing is None:
        return
    # With the sync knob the stage also absorbs the GPU work it dispatched.
    elapsed_s = float(elapsed_s) + _group_timing_device_barrier_s()
    group_timing[key] = group_timing.get(key, 0.0) + float(elapsed_s)


_SPARSE_KCLASS_GROUP_TIMING_SYNC_ENV = "RELAX_SPARSE_KCLASS_GROUP_TIMING_SYNC"
_GROUP_TIMING_SYNC_STATE: dict[str, object] = {}

def _group_timing_device_barrier_s() -> float:
    """Wait on a default-stream diagnostic token; return the wait in seconds.

    Diagnostic only (``RELAX_SPARSE_KCLASS_GROUP_TIMING_SYNC=1``). JAX
    dispatches asynchronously, so a host-side stage timer otherwise charges
    the GPU work of one stage to whichever later stage first pulls a value.
    The donor uses a tiny computation as a default compute-stream fence.
    This does not establish completion of unrelated custom streams.
    """
    enabled = _GROUP_TIMING_SYNC_STATE.get("enabled")
    if enabled is None:
        enabled = parse_env_flag(_SPARSE_KCLASS_GROUP_TIMING_SYNC_ENV, default=False)
        _GROUP_TIMING_SYNC_STATE["enabled"] = enabled
    if not enabled:
        return 0.0
    token = _GROUP_TIMING_SYNC_STATE.get("token")
    if token is None:
        token = jnp.asarray(0.0, dtype=jnp.float32)
        _GROUP_TIMING_SYNC_STATE["token"] = token
    t0 = time.time()
    (token + jnp.float32(1.0)).block_until_ready()
    return time.time() - t0
