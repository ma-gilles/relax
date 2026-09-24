"""The cached RELION CTF batch gather must not depend on how it is cached.

`_relion_exact_ctf_half_from_source_star_host` used to hold one NumPy row per
particle in a dict and rebuild each batch with a Python loop, one small gather
per image. On a K=1 100k/256 run that loop was 38.5 s of self time, 9.4% of the
whole run, so the rows moved into a single block gathered in one indexing
operation. These tests pin the observable behaviour that change has to preserve:
row identity, order, duplicates, and the selected pixel columns.

The cache is pre-populated so the tests never reach the RELION binding, which is
built per run root rather than into the checkout.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.relion import relion_ctf

PIXELS = 12  # a 4x4 image half-spectrum is 4 * (4 // 2 + 1)
PARTICLES = 5


@pytest.fixture
def populated_cache(monkeypatch, tmp_path):
    star = tmp_path / "particles.star"
    star.write_text("")
    monkeypatch.setenv("RELAX_K1_RELION_EXACT_CTF_STAR", str(star))

    rows = np.arange(PARTICLES * PIXELS, dtype=np.float64).reshape(PARTICLES, PIXELS)
    key = (str(star.resolve()), (4, 4))
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE,
        key,
        {
            "particles": None,
            "optics": {},
            "relion_bind": None,
            "slots": np.arange(PARTICLES, dtype=np.int64),
            "rows": rows,
            "n_cached": PARTICLES,
        },
    )
    return SimpleNamespace(particles_file=str(star)), rows


@pytest.mark.unit
def test_cached_ctf_batch_preserves_order_and_duplicates(populated_cache):
    dataset, rows = populated_cache
    indices = np.asarray([3, 0, 3, 1], dtype=np.int64)

    out = relion_ctf._relion_exact_ctf_half_from_source_star_host(
        dataset, indices, (4, 4),
    )

    assert out.dtype == np.float64
    assert out.shape == (indices.size, PIXELS)
    assert_matches(out, rows[indices])


@pytest.mark.unit
def test_cached_ctf_batch_selects_requested_pixels(populated_cache):
    dataset, rows = populated_cache
    indices = np.asarray([2, 2, 4], dtype=np.int64)
    # Unsorted, with a repeat: column order and duplication must be preserved.
    pixel_indices = np.asarray([7, 0, 7, 11], dtype=np.int64)

    out = relion_ctf._relion_exact_ctf_half_from_source_star_host(
        dataset, indices, (4, 4), pixel_indices=pixel_indices,
    )

    assert out.shape == (indices.size, pixel_indices.size)
    assert_matches(out, rows[indices][:, pixel_indices])


@pytest.mark.unit
def test_cached_ctf_batch_fails_closed_on_an_unevaluated_row(populated_cache, monkeypatch):
    """A slot that was never filled must raise, not return another particle's CTF."""

    dataset, _ = populated_cache
    key = next(iter(relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE))
    cache = relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE[key]
    # Leave the slot unset and make evaluating it a no-op, which is what a
    # silently-skipped particle would look like.
    cache["slots"][1] = -1
    monkeypatch.setattr(
        relion_ctf, "original_image_indices", lambda dataset, indices: np.asarray(indices),
    )

    with pytest.raises(Exception):
        relion_ctf._relion_exact_ctf_half_from_source_star_host(
            dataset, np.asarray([1], dtype=np.int64), (4, 4),
        )


@pytest.mark.unit
def test_exact_ctf_takes_relion_defaults_for_absent_ctf_columns(monkeypatch, tmp_path):
    """A STAR without rlnPhaseShift/rlnCtfBfactor/rlnCtfScalefactor gets RELION's 0/0/1.

    CTF::readValue (ctf.cpp:71-74, 81-91) reads the particle row, then its optics
    group, then the default; an optics-level value must win over the default.
    """
    import pandas as pd

    star = tmp_path / "particles.star"
    star.write_text("")
    monkeypatch.setenv("RELAX_K1_RELION_EXACT_CTF_STAR", str(star))
    relion_ctf.clear_exact_ctf_result_cache()
    calls = []

    def get_ctf_image(*args):
        calls.append(args)
        return np.zeros((4, 3), dtype=np.float64)

    particles = pd.DataFrame(
        {"rlnDefocusU": [1.0e4, 2.0e4], "rlnDefocusV": [1.1e4, 2.1e4], "rlnDefocusAngle": [5.0, 6.0],
         "rlnOpticsGroup": [1, 1]}
    )
    optics_row = pd.Series(
        {"rlnVoltage": 300.0, "rlnSphericalAberration": 2.7, "rlnAmplitudeContrast": 0.1,
         "rlnImagePixelSize": 1.5, "rlnCtfScalefactor": 0.75}
    )
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE,
        (str(star.resolve()), (4, 4)),
        {"particles": particles, "optics": {1: optics_row}, "relion_bind": SimpleNamespace(get_ctf_image=get_ctf_image),
         "slots": np.full(2, -1, dtype=np.int64), "rows": None, "n_cached": 0},
    )

    relion_ctf._relion_exact_ctf_half_from_source_star_host(
        SimpleNamespace(particles_file=str(star)), np.asarray([0, 1], dtype=np.int64), (4, 4)
    )

    assert len(calls) == 2
    for args in calls:
        bfactor, phase_shift, scale = args[6], args[13], args[14]
        assert (bfactor, phase_shift, scale) == (0.0, 0.0, 0.75)
