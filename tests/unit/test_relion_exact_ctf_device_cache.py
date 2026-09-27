"""The bounded exact-CTF device row cache returns the host path's binary64 rows."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from relax.relion import relion_ctf

pytestmark = pytest.mark.unit


def _cache(n_particles, image_size):
    class Binding:
        @staticmethod
        def get_ctf_images_batch(params, image_w, image_h, *_args):
            # Row values depend on the particle's defocus so a wrong slot shows up.
            base = np.arange(image_h * (image_w // 2 + 1), dtype=np.float64).reshape(image_h, image_w // 2 + 1)
            return np.stack([base * 1.0e-3 + row[0] for row in params])

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
        "relion_bind": Binding(),
        "slots": np.full(n_particles, -1, dtype=np.int64),
        "rows": None,
        "n_cached": 0,
    }


@pytest.mark.parametrize("min_rows", [4096, 600])
def test_device_cache_matches_the_host_gather_with_evictions(monkeypatch, tmp_path, min_rows):
    n_particles, size = 2500, 8
    star = (tmp_path / "particles.star").resolve()
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_source_star", lambda _dataset: star)
    monkeypatch.setattr(relion_ctf, "_EXACT_CTF_DEVICE_MIN_ROWS", min_rows)
    monkeypatch.setenv(relion_ctf._EXACT_CTF_CACHE_GB_ENV, "0")
    monkeypatch.setenv(relion_ctf._EXACT_CTF_DEVICE_GB_ENV, "1e-9")
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
