"""The final local pass scores with the numbered passes' source-faithful spectrum norm."""

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("preserve_order", [False, True])
def test_final_local_pass_keeps_the_source_faithful_spectrum_norm(monkeypatch, preserve_order):
    # RELION extends the norm correction and sigma2_noise from exp_power_img beyond current_size/2 the same way
    # in every iteration (ml_optimiser.cpp:10848-10856); the numbered local pass and the final dense pass use
    # the powerClass sums (source_faithful_spectrum_norm) wherever the particle order is preserved, and the
    # final local pass left them out. The tiny run's sampling never reaches the local order (its accuracy
    # estimate is unavailable), so the final pass is handed a local-search state; it stops at the scorer.
    from dataclasses import replace

    import numpy as np
    from helpers.tiny_refinement import run_tiny_refinement

    from relax.refinement import finalization, half_scoring

    class Reached(Exception):
        pass

    seen = []
    final = finalization.run_final_all_data

    def local_final(*args, **kwargs):
        sigma = float(np.deg2rad(2.0))
        carry = kwargs["carry"]
        kwargs["carry"] = replace(carry, state=replace(carry.state, do_local_search=True, sigma_rot=sigma, sigma_psi=sigma))
        return final(*args, **{**kwargs, "final_use_local": True})

    def scorer(half, sampling, priors, batching, execution, *rest):
        seen.append(execution.source_faithful_spectrum_norm)
        raise Reached

    monkeypatch.setattr(finalization, "run_final_all_data", local_final)
    monkeypatch.setattr(half_scoring, "_score_half_local", scorer)
    with pytest.raises(Reached):
        run_tiny_refinement(monkeypatch, max_iter=1, parity=dict(preserve_bpref_particle_order=preserve_order))
    assert seen == [preserve_order]
