"""The pass-2 RELION fine-scoring flags are resolved by one owner (relax.fine_pass.window)."""

from relax.fine_pass import window as sp


def test_owner_rules():
    exact, f32 = sp.pass2_relion_flags(
        relion_exact_fine_gaussian=True, relion_firstiter_score_mode="normalized_cc", relion_f32_fine_posterior=False,
        f32_fine_posterior_override=True,
    )
    assert (exact, f32) == (False, True)
    exact, f32 = sp.pass2_relion_flags(
        relion_exact_fine_gaussian=True, relion_firstiter_score_mode="gaussian", relion_f32_fine_posterior=False,
        f32_fine_posterior_override=False,
    )
    assert (exact, f32) == (True, False)
    exact, f32 = sp.pass2_relion_flags(
        relion_exact_fine_gaussian=False, relion_firstiter_score_mode="gaussian", relion_f32_fine_posterior=True,
        f32_fine_posterior_override=False,
    )
    assert (exact, f32) == (False, True)


def test_scoring_overrides_read_the_f32_fine_posterior_switch(monkeypatch):
    from relax.runtime.scoring_overrides import F32_FINE_POSTERIOR_ENV, ScoringOverrides

    monkeypatch.delenv(F32_FINE_POSTERIOR_ENV, raising=False)
    assert ScoringOverrides.from_environ().f32_fine_posterior is False is ScoringOverrides().f32_fine_posterior
    monkeypatch.setenv(F32_FINE_POSTERIOR_ENV, "1")
    assert ScoringOverrides.from_environ().f32_fine_posterior is True
