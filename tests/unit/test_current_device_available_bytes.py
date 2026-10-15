"""The one probe of what the device can still hand out (relax.runtime.memory_budget.current_device_available_bytes).

Expectations from the definition: the allocator's headroom, bounded by the physically free memory plus the
pool's unused bytes; the physical reading alone without an allocator reading; ``None`` when nothing is known.
"""

import pytest

from relax.runtime import memory_budget as budget

pytestmark = pytest.mark.unit

GIB = 1024**3


def _readings(monkeypatch, physical, allocator, pool, order=None):
    def reading(name, value):
        def read():
            if order is not None:
                order.append(name)
            return value

        return read

    monkeypatch.setattr(budget, "device_free_memory_bytes", reading("physical", physical))
    monkeypatch.setattr(budget, "jax_allocator_free_memory_bytes", reading("allocator", allocator))
    monkeypatch.setattr(budget, "jax_allocator_pool_free_bytes", reading("pool", pool))


def test_the_allocator_headroom_bounded_by_the_device(monkeypatch):
    order = []
    _readings(monkeypatch, 62 * GIB, 10 * GIB, 1 * GIB, order)
    assert budget.current_device_available_bytes() == 10 * GIB
    assert order == ["physical", "allocator", "pool"]
    _readings(monkeypatch, 2 * GIB, 10 * GIB, 1 * GIB)
    assert budget.current_device_available_bytes() == 3 * GIB


def test_partial_and_absent_readings(monkeypatch):
    _readings(monkeypatch, 5 * GIB, None, None)
    assert budget.current_device_available_bytes() == 5 * GIB
    _readings(monkeypatch, 5 * GIB, 7 * GIB, None)
    assert budget.current_device_available_bytes() == 7 * GIB  # without a pool reading the physical one does not bound
    _readings(monkeypatch, None, None, None)
    assert budget.current_device_available_bytes() is None
