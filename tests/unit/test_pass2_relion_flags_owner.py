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
