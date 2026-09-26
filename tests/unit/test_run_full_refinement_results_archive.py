"""The refinement results archive: a compressed ``.npz`` written at zlib level 1."""

import zipfile

import numpy as np
import pytest

from scripts.run_full_refinement import _savez_deflate_fast

pytestmark = pytest.mark.unit


def test_fast_archive_loads_like_savez_compressed(tmp_path):
    rng = np.random.default_rng(0)
    posterior = np.zeros((2, 4096))
    posterior[:, rng.integers(0, 4096, size=7)] = rng.uniform(size=7)
    arrays = {
        "half0_mean_ft": (rng.normal(size=64) + 1j * rng.normal(size=64)).astype(np.complex128),
        "rotation_posterior_per_half_iter_003": posterior,
        "hard_assignments_half0": rng.integers(0, 9, size=11),
        "setup_phase_names": np.asarray(["read", "setup"]),
        "scalar": np.float64(2.5),
    }
    fast, reference = tmp_path / "fast.npz", tmp_path / "reference.npz"
    _savez_deflate_fast(fast, arrays)
    np.savez_compressed(reference, **arrays)

    with np.load(fast) as got, np.load(reference) as want:
        assert sorted(got.files) == sorted(want.files)
        for name in want.files:
            assert got[name].dtype == want[name].dtype
            np.testing.assert_array_equal(got[name], want[name])
    with zipfile.ZipFile(fast) as archive:
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_DEFLATED}
