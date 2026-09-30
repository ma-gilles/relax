"""The bounded exact-CTF device row cache returns the host path's binary64 rows."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from relax.relion import relion_ctf

pytestmark = pytest.mark.unit


def _fake_ctf(params, size, pixel_size, *, finish, **_kwargs):
    # Row values depend on the particle's defocus so a wrong slot shows up.
    base = np.arange(size * (size // 2 + 1), dtype=np.float64)
    finish(0, len(params), np.stack([base * 1.0e-3 + row[0] for row in params]))


@pytest.fixture(autouse=True)
def _relax_ctf(monkeypatch):
    monkeypatch.setattr(relion_ctf, "relion_ctf_fftw_half", _fake_ctf)


def _cache(n_particles, image_size):
    particles = pd.DataFrame(
        {
            "rlnDefocusU": np.arange(n_particles, dtype=np.float64) + 1.0e4,
            "rlnDefocusV": np.full(n_particles, 1.1e4),
            "rlnDefocusAngle": np.zeros(n_particles),
            "rlnOpticsGroup": np.ones(n_particles, dtype=np.int64),
        }
    )
    optics = pd.Series(
        {"rlnVoltage": 300.0, "rlnSphericalAberration": 2.7, "rlnAmplitudeContrast": 0.1, "rlnImagePixelSize": 1.5}
    )
    return {
        "particles": particles,
        "optics": {1: optics},
        "slots": np.full(n_particles, -1, dtype=np.int64),
        "rows": None,
        "n_cached": 0,
    }


@pytest.mark.parametrize("budget_gb", ["1", "1.8e-4"])
def test_device_cache_matches_the_host_gather_with_evictions(monkeypatch, tmp_path, budget_gb):
    n_particles, size = 2500, 8
    star = (tmp_path / "particles.star").resolve()
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_source_star", lambda _dataset: star)
    monkeypatch.setenv(relion_ctf._EXACT_CTF_CACHE_GB_ENV, "0")
    # 1.8e-4 GB is about 600 rows of this 8-pixel fixture, so later requests evict.
    monkeypatch.setenv(relion_ctf._EXACT_CTF_DEVICE_GB_ENV, budget_gb)
    dataset = SimpleNamespace(original_image_indices_from_local=lambda indices: np.asarray(indices))
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, size * (size // 2 + 1), 17)
    requests = [
        (np.arange(300), None),
        (np.arange(200, 1100)[::-1], pixels),
        (rng.permutation(2500)[:500], None),
        (np.concatenate([np.arange(40), np.zeros(8, dtype=np.int64)]), pixels),
        (np.arange(2500)[::3], None),
        (rng.permutation(2500)[:550], pixels),
    ]

    monkeypatch.setitem(relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE, (str(star), (size, size)), _cache(n_particles, size))
    host = [
        relion_ctf._relion_exact_ctf_half_from_source_star_host(dataset, r, (size, size), pixel_indices=p)
        for r, p in requests
    ]

    cache = _cache(n_particles, size)
    monkeypatch.setitem(relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE, (str(star), (size, size)), cache)
    for (request, pixel_indices), expected in zip(requests, host):
        actual = relion_ctf._relion_exact_ctf_half_from_source_star(
            dataset, request, (size, size), pixel_indices=pixel_indices
        )
        assert actual.dtype == np.float64
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    state = cache["device_cache"]
    resident = state["row_of_slot"] >= 0
    # The two maps agree, and the cache never held more rows than its capacity.
    assert np.array_equal(state["slot_of_row"][state["row_of_slot"][resident]], np.flatnonzero(resident))
    assert resident.sum() <= state["block"].shape[0]


def test_box800_cache_takes_its_share_of_free_memory_and_yields(monkeypatch):
    """A box-800 row is 2.57 MB: the cache holds a quarter of what the device can still
    hand out (plus what it holds), never less than the request, never a row floor."""

    from relax.sparse_pass2 import sparse_pass2_budget

    row_bytes = 800 * (800 // 2 + 1) * 8
    gib = 1024**3
    monkeypatch.delenv(relion_ctf._EXACT_CTF_DEVICE_GB_ENV, raising=False)
    available = {"bytes": 40 * gib}
    monkeypatch.setattr(sparse_pass2_budget, "_device_free_memory_bytes", lambda: available["bytes"])
    monkeypatch.setattr(sparse_pass2_budget, "_jax_allocator_free_memory_bytes", lambda: available["bytes"])
    monkeypatch.setattr(sparse_pass2_budget, "_jax_allocator_pool_free_bytes", lambda: 0)

    capacity = relion_ctf._exact_ctf_device_capacity(200_000, 128, row_bytes, 0)
    assert capacity == int(0.25 * 40 * gib) // row_bytes  # about 10 GiB, 4180 rows
    # With 8 GiB left for a pass that needs its accumulators, the target drops
    # below three quarters of the current capacity, so the cache shrinks.
    available["bytes"] = 8 * gib
    shrunk = relion_ctf._exact_ctf_device_capacity(200_000, 128, row_bytes, capacity)
    assert shrunk < relion_ctf._EXACT_CTF_DEVICE_SHRINK_BELOW * capacity
    # No free memory: exactly the request, not a fixed row floor.
    available["bytes"] = 0
    assert relion_ctf._exact_ctf_device_capacity(200_000, 128, row_bytes, 0) == 128
    # Never more rows than particles.
    available["bytes"] = 40 * gib
    assert relion_ctf._exact_ctf_device_capacity(1000, 128, row_bytes, 0) == 1000


def test_release_drops_the_device_cache(monkeypatch, tmp_path):
    cache = _cache(10, 8)
    cache["device_cache"] = {"block": None}
    monkeypatch.setitem(relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE, ("x", (8, 8)), cache)
    relion_ctf.release_exact_ctf_device_cache()
    assert "device_cache" not in cache


def test_ensure_device_headroom_releases_only_when_short(monkeypatch):
    from relax.sparse_pass2 import sparse_pass2_budget

    gib = 1024**3
    available = {"bytes": 20 * gib}
    monkeypatch.setattr(sparse_pass2_budget, "_device_free_memory_bytes", lambda: available["bytes"])
    monkeypatch.setattr(sparse_pass2_budget, "_jax_allocator_free_memory_bytes", lambda: available["bytes"])
    monkeypatch.setattr(sparse_pass2_budget, "_jax_allocator_pool_free_bytes", lambda: 0)
    cache = _cache(10, 8)
    cache["device_cache"] = {"block": None}
    monkeypatch.setitem(relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE, ("x", (8, 8)), cache)
    assert relion_ctf.ensure_device_headroom(16 * gib) is False
    assert "device_cache" in cache
    assert relion_ctf.ensure_device_headroom(int(15.35 * gib) + 25 * gib) is True
    assert "device_cache" not in cache
    assert relion_ctf.ensure_device_headroom(40 * gib) is False  # nothing left to free
