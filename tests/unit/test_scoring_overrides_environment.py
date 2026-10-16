"""``ScoringOverrides.from_environ``: each scoring variable the command reads once, and how it is parsed.

The command reads the scoring variables once, into ``ScoringOverrides`` (code rule 5); every other test builds
the record with explicit values (rule 22). These are the record's parse tests.
"""

import pytest

from relax.runtime.scoring_overrides import (
    COARSE_GEMM_FLOAT64_ENV,
    EXACT_FLOAT32_ACCUMULATION_ENV,
    F32_COARSE_SUPPORT_ENV,
    F32_FINE_POSTERIOR_ENV,
    PROJECTOR_TEXTURE_INTERP_ENV,
    X_HALF_MSTEP_DOUBLE_ENV,
    ScoringOverrides,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,expected", [(None, None), ("0", False), ("1", True)])
def test_the_coarse_rotated_radius_is_read_once(monkeypatch, token, expected):
    """RELAX_K1_COARSE_ROTATED_RADIUS, read once by the command: unset is None (the pass's default)."""
    monkeypatch.delenv("RELAX_K1_COARSE_ROTATED_RADIUS", raising=False)
    if token is not None:
        monkeypatch.setenv("RELAX_K1_COARSE_ROTATED_RADIUS", token)
    assert ScoringOverrides.from_environ().coarse_rotated_radius is expected


@pytest.mark.parametrize("token", ["typo", "", " 1"])
def test_a_malformed_coarse_rotated_radius_is_refused(monkeypatch, token):
    monkeypatch.setenv("RELAX_K1_COARSE_ROTATED_RADIUS", token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        ScoringOverrides.from_environ()


def test_the_double_switch_is_read_once_by_scoring_overrides(monkeypatch):
    """RELAX_RELION_X_HALF_MSTEP_DOUBLE (``env_flags.parse_env_flag``): unset and 0 off, 1 on."""
    monkeypatch.delenv(X_HALF_MSTEP_DOUBLE_ENV, raising=False)
    assert ScoringOverrides.from_environ().x_half_mstep_double is False is ScoringOverrides().x_half_mstep_double
    monkeypatch.setenv(X_HALF_MSTEP_DOUBLE_ENV, "0")
    assert ScoringOverrides.from_environ().x_half_mstep_double is False
    monkeypatch.setenv(X_HALF_MSTEP_DOUBLE_ENV, "1")
    assert ScoringOverrides.from_environ().x_half_mstep_double is True


def test_scoring_overrides_read_the_f32_fine_posterior_switch(monkeypatch):
    monkeypatch.delenv(F32_FINE_POSTERIOR_ENV, raising=False)
    assert ScoringOverrides.from_environ().f32_fine_posterior is False is ScoringOverrides().f32_fine_posterior
    monkeypatch.setenv(F32_FINE_POSTERIOR_ENV, "1")
    assert ScoringOverrides.from_environ().f32_fine_posterior is True


def test_relion_f32_coarse_support_is_read_once_at_the_boundary(monkeypatch):
    """RECOVAR_K1_RELION_F32_COARSE_SUPPORT (a strict flag): unset is on, the production RELION float32 support;
    0 selects the generic support route; anything else is refused, as before."""
    monkeypatch.delenv(F32_COARSE_SUPPORT_ENV, raising=False)
    assert ScoringOverrides.from_environ().f32_coarse_support is True
    monkeypatch.setenv(F32_COARSE_SUPPORT_ENV, " OFF ")
    assert ScoringOverrides.from_environ().f32_coarse_support is False
    monkeypatch.setenv(F32_COARSE_SUPPORT_ENV, "1")
    assert ScoringOverrides.from_environ().f32_coarse_support is True
    for raw in ("", "maybe"):
        monkeypatch.setenv(F32_COARSE_SUPPORT_ENV, raw)
        with pytest.raises(ValueError, match=f"Unsupported {F32_COARSE_SUPPORT_ENV}"):
            ScoringOverrides.from_environ()


def test_coarse_gemm_float64_is_read_once_at_the_boundary(monkeypatch):
    """RELAX_COARSE_GEMM_FLOAT64: unset is 0 (RELION's float32 GEMMs); exactly 0 or 1 after stripping; anything else
    is refused, as before."""
    monkeypatch.delenv(COARSE_GEMM_FLOAT64_ENV, raising=False)
    assert ScoringOverrides.from_environ().coarse_gemm_float64 is False
    monkeypatch.setenv(COARSE_GEMM_FLOAT64_ENV, " 1 ")
    assert ScoringOverrides.from_environ().coarse_gemm_float64 is True
    for raw in ("", "true", "2"):
        monkeypatch.setenv(COARSE_GEMM_FLOAT64_ENV, raw)
        with pytest.raises(ValueError, match=f"Unsupported {COARSE_GEMM_FLOAT64_ENV}"):
            ScoringOverrides.from_environ()


def test_the_projector_texture_route_is_read_once(monkeypatch):
    """The run's choice (ScoringOverrides.projector_texture_interp): on unless the variable says 0."""
    monkeypatch.delenv(PROJECTOR_TEXTURE_INTERP_ENV, raising=False)
    assert ScoringOverrides.from_environ().projector_texture_interp is True is ScoringOverrides().projector_texture_interp
    monkeypatch.setenv(PROJECTOR_TEXTURE_INTERP_ENV, "0")
    assert ScoringOverrides.from_environ().projector_texture_interp is False
    monkeypatch.setenv(PROJECTOR_TEXTURE_INTERP_ENV, "maybe")
    with pytest.raises(ValueError, match=PROJECTOR_TEXTURE_INTERP_ENV):
        ScoringOverrides.from_environ()


def test_the_exact_float32_accumulation_switch_is_read_once(monkeypatch):
    """RELAX_EM_EXACT_FLOAT32_ACCUMULATION: unset or 0 off, 1 on."""
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv(EXACT_FLOAT32_ACCUMULATION_ENV, raising=False)
        else:
            monkeypatch.setenv(EXACT_FLOAT32_ACCUMULATION_ENV, value)
        assert not ScoringOverrides.from_environ().exact_float32_accumulation
    assert EXACT_FLOAT32_ACCUMULATION_ENV == "RELAX_EM_EXACT_FLOAT32_ACCUMULATION"
    monkeypatch.setenv(EXACT_FLOAT32_ACCUMULATION_ENV, "1")
    assert ScoringOverrides.from_environ().exact_float32_accumulation
