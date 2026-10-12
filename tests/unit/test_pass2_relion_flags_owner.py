"""The pass-2 RELION fine-scoring flags are resolved by one owner (sparse_pass2_window)."""

from relax.fine_pass import window as sp


def test_owner_rules(monkeypatch):
    from relax.fine_pass import window

    monkeypatch.setattr(window, "parse_env_flag", lambda name, default=False: False)
    monkeypatch.setattr(window, "relion_x_half_f32_fine_posterior_enabled", lambda: True)
    exact, ffi, f32 = sp._pass2_relion_flags(relion_exact_fine_gaussian=True, relion_firstiter_score_mode="normalized_cc", relion_fine_diff2_fused_ffi=False, relion_f32_fine_posterior=False)
    assert (exact, ffi, f32) == (False, False, True)
    exact, ffi, f32 = sp._pass2_relion_flags(relion_exact_fine_gaussian=True, relion_firstiter_score_mode="gaussian", relion_fine_diff2_fused_ffi=True, relion_f32_fine_posterior=False)
    assert (exact, ffi, f32) == (True, True, True)
