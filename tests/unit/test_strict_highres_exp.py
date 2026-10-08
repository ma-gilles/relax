"""RELION --strict_highres_exp: the E-step scores at the capped size, the M-step keeps the current size."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from helpers.run_options import stand_in

from relax.helpers.resolution import ImageGeometry
from relax.refinement import iteration_planning
from relax.refinement.command_options import validate_strict_highres_exp
from relax.refinement.iteration_planning import ExpectationWindows, relion_strict_highres_image_size

pytestmark = pytest.mark.unit
LOG = logging.getLogger(__name__)


def test_size_is_relion_two_round_box_pixel_over_limit():
    # K2 5k/128 fixture: 128 px at 4.25 A, limit 12 A -> 2 * ROUND(45.33) = 90.
    assert relion_strict_highres_image_size(4.25, 128, 12.0) == 90
    # ROUND is half away from zero: 2 * ROUND(4.5) = 10.
    assert relion_strict_highres_image_size(1.0, 9, 2.0) == 10


def _optics(box=128, pixel=4.25):
    return iteration_planning.RunOptics(
        image_geometry=ImageGeometry(image_shape=(box, box), pixel_size_angstrom=pixel), model_pixel_size=pixel,
        optics_image_sizes=[box], optics_pixel_sizes=[pixel], multi_shape_halves=False,
    )


@pytest.mark.parametrize("current_size, score", [(120, 90), (128, 90), (64, 64)])
def test_windows_cap_only_the_e_step(current_size, score):
    windows = iteration_planning.plan_expectation_windows(
        current_size, _optics(), log=LOG, strict_highres_exp_angstrom=12.0
    )
    # M-step consumers (class_maximization's image_current_size, the summed noise shells) keep the current size.
    assert windows.image_current_size == current_size and windows.model_size == current_size
    assert windows.score_window_size == score
    if score < current_size:
        assert windows.engine_model_window_size == current_size
    off = iteration_planning.plan_expectation_windows(current_size, _optics(), log=LOG)
    assert off.score_window_size == off.image_window_size
    assert off.engine_model_window_size == off.model_window_size


def test_cap_uses_the_image_pixel_size_without_optics_pixel_sizes():
    # RELION's cap is 2 * ROUND(remap_sizes * ori_size * mymodel.pixel_size / limit) (ml_optimiser.cpp:6917,
    # 6932), which is the optics group's box times its image pixel size, whatever the model pixel size. Class3D
    # runs have no per-optics pixel sizes and a model pixel size that may be the MRC header value (544/384 here)
    # instead of the STAR one (1.416667): 384 * 1.416667 / 8.704 = 62.50001 -> 126, 384 * 544/384 / 8.704 =
    # 62.49999 -> 124.
    optics = iteration_planning.RunOptics(
        image_geometry=ImageGeometry(image_shape=(384, 384), pixel_size_angstrom=1.416667),
        model_pixel_size=544 / 384, optics_image_sizes=None, optics_pixel_sizes=None, multi_shape_halves=False,
    )
    windows = iteration_planning.plan_expectation_windows(384, optics, log=LOG, strict_highres_exp_angstrom=8.704)
    assert windows.score_size == relion_strict_highres_image_size(1.416667, 384, 8.704) == 126


def test_off_windows_are_unchanged():
    windows = ExpectationWindows(model_size=64, image_current_size=64, image_box_size=128)
    assert windows.score_window_size == windows.image_window_size == 64
    assert windows.engine_model_window_size == windows.model_window_size


def test_only_class3d_and_no_accuracy_current_size():
    validate_strict_highres_exp(SimpleNamespace(strict_highres_exp=None, n_classes=1))
    validate_strict_highres_exp(SimpleNamespace(strict_highres_exp=12.0, n_classes=2, accuracy_current_size=False))
    with pytest.raises(SystemExit, match="Class3D"):
        validate_strict_highres_exp(SimpleNamespace(strict_highres_exp=12.0, n_classes=1, accuracy_current_size=False))
    with pytest.raises(SystemExit, match="accuracy_current_size"):
        validate_strict_highres_exp(SimpleNamespace(strict_highres_exp=12.0, n_classes=2, accuracy_current_size=True))
    with pytest.raises(SystemExit, match="positive"):
        validate_strict_highres_exp(SimpleNamespace(strict_highres_exp=0.0, n_classes=2, accuracy_current_size=False))


def test_controller_hands_the_cap_to_the_e_step_and_the_current_size_to_the_m_step(monkeypatch):
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement import iteration_loop

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "prepare_numbered_expectation", "phase")
    trace.wrap(iteration_loop, "plan_adaptive_image_size", "coarse")
    trace.wrap(iteration_loop, "class_maximization", "mstep")
    # 8 px at 1 A, limit 4 A: the E-step size is 2 * ROUND(2) = 4 while the current size is the box.
    engine_calls = []
    run_tiny_refinement(
        monkeypatch, n_classes=2, final_after_max_iter=False, schedule={"init_current_size": 8},
        adaptive=stand_in.adaptive(adaptive_oversampling=1, strict_highres_exp_angstrom=4.0), engine_calls=engine_calls,
    )
    phases, coarse, msteps = trace.calls("phase"), trace.calls("coarse"), trace.calls("mstep")
    assert phases and len(phases) == len(coarse) == len(msteps)
    for phase, plan, mstep in zip(phases, coarse, msteps, strict=True):
        windows = phase.args[1]
        assert windows.score_size == 4
        assert phase.result.sampling.cs_for_engine == 4
        assert phase.result.sampling.model_current_size_for_engine == windows.model_size
        assert plan.result.size == 4
        assert mstep.kwargs["image_current_size"] == windows.image_current_size > 4
    # The engine scores at the cap and sums the noise, Wavg and powerClass terms at the full current size.
    assert engine_calls
    for call in engine_calls:
        assert call["kwargs"]["current_size"] == 4
        assert call["kwargs"]["wsum_current_size"] == call["kwargs"]["reconstruction_current_size"] > 4


def test_capped_accuracy_projects_at_the_current_size(monkeypatch):
    # RELION's calculateExpectedAngularErrors projects through PPref at r_max = current_size / 2 and caps only the
    # image; the capped estimate must keep the projector size, the uncapped one leaves it to the image size.
    from relax.helpers import expected_accuracy

    calls = []
    monkeypatch.setattr(expected_accuracy, "estimate_relion_expected_accuracy", lambda **kw: calls.append(kw))
    inputs = expected_accuracy.Half1AccuracyInputs(
        trial_order_local=None, dataset=None, volume_shape=(8, 8, 8), padding_factor=2, sigma2_fudge=1.0,
        optimizer_random_seed=0,
        expected_accuracy=SimpleNamespace(half1_particle_ids=None, half1_ctf_params=None, do_ctf_correction=True),
    )
    common = dict(reference_fourier=None, best_eulers_deg=None, class_ids=None, class_weights=None,
                  sigma2_noise_native=None)
    inputs.estimate(**common, current_image_size=90, projector_current_size=128)
    inputs.estimate(**common, current_image_size=128)
    assert calls[0]["current_image_size"] == 90
    assert calls[0]["group_grid"] == {"projector_current_size": 128}
    assert calls[1]["group_grid"] is None
