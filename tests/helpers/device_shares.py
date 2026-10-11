"""Read the workers currently declared to share the device (test observer of memory_budget's declaration)."""

from relax.runtime import memory_budget


def concurrent_device_shares() -> int:
    return memory_budget._CONCURRENT_DEVICE_SHARES
