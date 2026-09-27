"""The exact CTF device mirror returns the host path's binary64 rows."""

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


def test_device_mirror_matches_the_host_gather_across_uploads(monkeypatch, tmp_path):
    n_particles, size = 2500, 8
    star = (tmp_path / "particles.star").resolve()
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_source_star", lambda _dataset: star)
    monkeypatch.setattr(relion_ctf, "_EXACT_CTF_DEVICE_UPLOAD_ROWS", 1000)
    monkeypatch.setenv(relion_ctf._EXACT_CTF_CACHE_GB_ENV, "0")
    dataset = SimpleNamespace(original_image_indices_from_local=lambda indices: np.asarray(indices))
    requests = [np.arange(300), np.arange(200, 1700)[::-1], np.arange(2500)[::7], np.arange(2500)]

    monkeypatch.setitem(relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE, (str(star), (size, size)), _cache(n_particles, size))
    monkeypatch.setenv(relion_ctf._EXACT_CTF_DEVICE_GB_ENV, "0")
    host = [np.asarray(relion_ctf._relion_exact_ctf_half_from_source_star(dataset, r, (size, size))) for r in requests]

    cache = _cache(n_particles, size)
    monkeypatch.setitem(relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE, (str(star), (size, size)), cache)
    monkeypatch.setenv(relion_ctf._EXACT_CTF_DEVICE_GB_ENV, "1")
    for request, expected in zip(requests, host):
        actual = relion_ctf._relion_exact_ctf_half_from_source_star(dataset, request, (size, size))
        assert actual.dtype == np.float64
        assert np.asarray(actual).tobytes() == expected.tobytes()
    assert cache["device_mirrored"] == cache["n_cached"] == n_particles
