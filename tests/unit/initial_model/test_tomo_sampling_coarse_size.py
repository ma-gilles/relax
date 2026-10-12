"""The subtomogram VDAM coarse size on an iteration that updates the sampling."""

from types import SimpleNamespace

from helpers.vdam import initial_model_state_stub

from relax.vdam.native_options import VdamEnvironment
from relax.vdam.tomo_estep import tomo_initial_model_sampling


def _plan(order):
    return SimpleNamespace(
        healpix_order=order, oversampling=1, offset_range_angstrom=15.566, offset_step_angstrom=4.25,
        random_perturbation=0.242,
    )


def test_coarse_size_is_the_order_before_the_sampling_update():
    # RELION sets image_coarse_size (expectationSetup step A, ml_optimiser.cpp:3914) before it updates the
    # angular sampling (step D, :3945), so the iteration that moves from HEALPix 1 to 2 still scores its coarse
    # pass at the HEALPix-1 size. etob2l_plain (box 64 at 4.25 A, diameter 240): 12 at order 1, 22 at order 2;
    # stock RELION keeps 25 significant samples per particle there, the order-2 size kept 2.6 (job 14971225).
    state = initial_model_state_stub(box_size=64, pixel_size=4.25, current_size=38, Iref=None, Igrad1=None, Igrad2=None)
    assert tomo_initial_model_sampling(state, _plan(1), particle_diameter_ang=240.0).coarse_size == 12
    assert tomo_initial_model_sampling(state, _plan(2), particle_diameter_ang=240.0).coarse_size == 22
    updated = tomo_initial_model_sampling(state, _plan(2), particle_diameter_ang=240.0, pass1_healpix_order=1)
    assert (updated.healpix_order, updated.coarse_size, updated.fine_size) == (2, 12, 38)


def test_coarse_size_equals_the_single_particle_paths_for_the_same_inputs():
    from relax.vdam.adaptive_estep import resolve_sparse_pass1_current_size

    state = initial_model_state_stub(box_size=64, pixel_size=4.25, current_size=38, Iref=None, Igrad1=None, Igrad2=None)
    for before, after in ((1, 2), (2, 3), (3, 3)):
        spa = resolve_sparse_pass1_current_size(state, 38, 240.0, before)
        tomo = tomo_initial_model_sampling(state, _plan(after), particle_diameter_ang=240.0, pass1_healpix_order=before)
        assert tomo.coarse_size == spa


def test_the_subtomogram_estep_takes_the_order_before_the_update(monkeypatch):
    import ast
    import inspect

    import numpy as np
    import pytest

    from relax.vdam import driver, tomo_estep

    seen = {}

    class Reached(Exception):
        pass

    def sampling(state, plan, **kwargs):
        seen.update(kwargs)
        raise Reached

    monkeypatch.setattr(tomo_estep, "tomo_initial_model_sampling", sampling)
    with pytest.raises(Reached):
        tomo_estep.run_tomo_initial_model_estep(
            SimpleNamespace(subset=lambda ids: None), SimpleNamespace(K=1, pseudo_halfsets=True),
            sampling_plan=_plan(2), particle_ids=np.arange(4), halfset_ids=np.array([0, 1, 0, 1]),
            previous_offsets_px=np.zeros((4, 3)), noise_variance=None, relion_projector_half_by_class=None,
            relion_projector_r_max=10, class_rotation_log_prior=None, max_significants=100,
            sigma_offset_angstrom=3.0, particle_diameter_ang=240.0, padding_factor=1, pass1_healpix_order=1,
            environment=VdamEnvironment(),
        )
    assert seen["pass1_healpix_order"] == 1
    # The driver hands the subtomogram E-step the order it read before updating the sampling, as it does for
    # single particles (pass1_healpix_order, set before the expected-accuracy and sampling update).
    calls = [
        node for node in ast.walk(ast.parse(inspect.getsource(driver)))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_tomo_initial_model_estep"
    ]
    assert calls and all(
        any(kw.arg == "pass1_healpix_order" and getattr(kw.value, "id", "") == "pass1_healpix_order" for kw in call.keywords)
        for call in calls
    )


def test_the_subtomogram_estep_honours_the_environments_engine_diagnostics(monkeypatch):
    """deep2 V2-3: RELAX_ADAPTIVE_FRACTION and RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE reach the tilt pass."""
    import numpy as np
    import pytest

    from relax.refinement import tomo_scoring
    from relax.sampling import rotation_grid_size
    from relax.steps.scoring_policy import RELION_ADAPTIVE_FRACTION
    from relax.vdam import tomo_estep

    class Reached(Exception):
        pass

    seen = []

    def score(half, **kwargs):
        seen.append((kwargs["adaptive_fraction"], kwargs["mstep_subtract_ctf_projection"]))
        raise Reached

    monkeypatch.setattr(tomo_scoring, "score_tomo_half", score)
    monkeypatch.setattr(tomo_estep, "tomo_initial_model_sampling", lambda *a, **k: SimpleNamespace(healpix_order=1))
    for adaptive_fraction, subtract in ((None, True), (0.99, False)):
        with pytest.raises(Reached):
            tomo_estep.run_tomo_initial_model_estep(
                SimpleNamespace(subset=lambda ids: None), SimpleNamespace(K=1, pseudo_halfsets=True),
                sampling_plan=_plan(1), particle_ids=np.arange(4), halfset_ids=np.array([0, 1, 0, 1]),
                previous_offsets_px=np.zeros((4, 3)), noise_variance=None,
                relion_projector_half_by_class=np.zeros((1, 2, 2, 2), dtype=np.complex64),
                relion_projector_r_max=10, class_rotation_log_prior=np.zeros((1, rotation_grid_size(1))),
                max_significants=100, sigma_offset_angstrom=3.0, particle_diameter_ang=240.0, padding_factor=1,
                environment=VdamEnvironment(adaptive_fraction=adaptive_fraction, subtract_projected_reference=subtract),
            )
    # The default call is unchanged: RELION's 0.999 and the residual back-projection.
    assert seen == [(RELION_ADAPTIVE_FRACTION, True), (0.99, False)]
