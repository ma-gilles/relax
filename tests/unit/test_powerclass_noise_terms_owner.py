"""The pass-2 RELION powerClass noise terms are selected by one owner (sparse_pass2_scoring)."""

from __future__ import annotations

import pytest

from relax.sparse_pass2 import sparse_pass2_scoring as sp
from relax.sparse_pass2 import sparse_pass2_scoring

pytestmark = pytest.mark.unit


def _with_fakes(monkeypatch):
    calls = []
    monkeypatch.setattr(sparse_pass2_scoring, "_relion_cuda_powerclass_highres_xi2_half", lambda x, **kw: calls.append("xi2") or "xi2")
    monkeypatch.setattr(sparse_pass2_scoring, "_relion_cuda_powerclass_spectrum_highres_norm_units", lambda x, **kw: calls.append("spectrum") or "spectrum")
    monkeypatch.setattr(sparse_pass2_scoring, "_relion_powerclass_highres_xi2_half_to_norm_units", lambda v, shape: calls.append("convert") or ("norm", v))
    return calls


def test_nothing_is_computed_without_exact_scoring_or_noise_accumulation(monkeypatch):
    calls = _with_fakes(monkeypatch)
    out = sp._relion_powerclass_noise_terms("x", image_shape=(8, 8), current_size=8, use_exact_relion_gaussian=False, accumulate_noise=False, source_faithful_spectrum_norm=True)
    assert out == (None, None) and calls == []


def test_exact_scoring_without_current_size_only_needs_highres_xi2(monkeypatch):
    calls = _with_fakes(monkeypatch)
    out = sp._relion_powerclass_noise_terms("x", image_shape=(8, 8), current_size=None, use_exact_relion_gaussian=True, accumulate_noise=True, source_faithful_spectrum_norm=False)
    assert out == ("xi2", None) and calls == ["xi2"]


@pytest.mark.parametrize("source_faithful,expected,calls_expected", [(True, "spectrum", ["xi2", "spectrum"]), (False, ("norm", "xi2"), ["xi2", "convert"])])
def test_noise_accumulation_selects_the_norm_term(monkeypatch, source_faithful, expected, calls_expected):
    calls = _with_fakes(monkeypatch)
    out = sp._relion_powerclass_noise_terms("x", image_shape=(8, 8), current_size=8, use_exact_relion_gaussian=False, accumulate_noise=True, source_faithful_spectrum_norm=source_faithful)
    assert out == ("xi2", expected) and calls == calls_expected


def test_noise_terms_key_on_the_box_and_take_the_current_size_at_runtime():
    """The powerClass terms at a traced current size match the static ones to float32 summation order."""

    import jax.numpy as jnp
    import numpy as np

    from relax.sparse_pass2 import sparse_pass2_scoring as scoring

    rng = np.random.default_rng(4)
    image_shape = (16, 16)
    half = image_shape[0] * (image_shape[1] // 2 + 1)
    images = jnp.asarray(rng.normal(size=(3, half)) + 1j * rng.normal(size=(3, half)), dtype=jnp.complex64)
    for current_size in (8, 10, 12):
        xi2, norm = scoring._relion_powerclass_noise_terms(
            images,
            image_shape=image_shape,
            current_size=current_size,
            use_exact_relion_gaussian=True,
            accumulate_noise=True,
            source_faithful_spectrum_norm=True,
        )
        static_xi2 = scoring._relion_cuda_powerclass_highres_xi2_half(
            images, image_shape=image_shape, current_size=current_size
        )
        static_norm = scoring._relion_cuda_powerclass_spectrum_highres_norm_units(
            images, image_shape=image_shape, current_size=current_size
        )
        # Both terms are float32 sums of at most ``half`` non-negative pixel powers, so the traced and static
        # programs may differ by their summation order: at most half * eps32 times the value (plus the unit scaling).
        rtol = (half + 4) * float(np.finfo(np.float32).eps)
        np.testing.assert_allclose(np.asarray(xi2), np.asarray(static_xi2), rtol=rtol, atol=0)
        np.testing.assert_allclose(np.asarray(norm), np.asarray(static_norm), rtol=rtol, atol=0)
