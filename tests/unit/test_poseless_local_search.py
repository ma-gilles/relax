"""A local search without input poses is centred at Euler angles (0, 0, 0), as RELION reads absent angles."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_poseless_halves_search_locally_around_zero_in_the_first_iteration(monkeypatch):
    # RELION reads absent rlnAngleRot/Tilt/Psi and origins as 0 (exp_model.cpp:1103-1134) and, with --sigma_ang,
    # searches locally around them from iteration 1 (ml_optimiser.cpp:978-983). relax scored that iteration with a
    # global search instead.
    from helpers.tiny_refinement import run_tiny_refinement

    from relax.local_search import half as local_half
    from relax.refinement.refinement_options import LocalSearchOptions

    class Reached(Exception):
        pass

    centres = []

    def scorer(half, *rest):
        centres.append(np.asarray(half.particles.rotation_eulers))
        raise Reached

    monkeypatch.setattr(local_half, "_score_half_local", scorer)
    with pytest.raises(Reached):
        run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False,
                            local_search=LocalSearchOptions(sigma_ang_deg=5.0))
    assert len(centres) == 1
    np.testing.assert_array_equal(centres[0], np.zeros_like(centres[0]))
