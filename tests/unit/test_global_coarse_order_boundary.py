"""The global pass 1 sizes its Fourier window from the incoming sampling order."""

import logging

import pytest
from helpers.run_options import stand_in

import relax.helpers.resolution as resolution_helpers
from relax.helpers.resolution import ImageGeometry
from relax.refinement.iteration_planning import ExpectationWindows, RunOptics, plan_adaptive_image_size

pytestmark = pytest.mark.unit


def _global_window(incoming, *, current=172, sealed=None):
    """The adaptive pass-1 window of a box-380 run at ``current`` for the incoming order ``incoming``; with a
    sealed sampling state (the replay source's), its width."""
    from relax.parity.relion_replay_source import RelionReplay, RelionReplaySource

    options = stand_in.options(schedule=stand_in.schedule(particle_diameter_ang=250.0))
    plan = plan_adaptive_image_size(
        incoming,
        ExpectationWindows(model_size=current, image_current_size=current, image_box_size=380),
        RunOptics(
            image_geometry=ImageGeometry(image_shape=(380, 380), pixel_size_angstrom=1.400011),
            model_pixel_size=1.400011, optics_image_sizes=[380], optics_pixel_sizes=[1.400011],
            multi_shape_halves=False,
        ),
        options,
        log=logging.getLogger(__name__),
    )
    if sealed is not None:
        plan = RelionReplaySource(RelionReplay(sealed_sampling_state=sealed), options).adaptive_coarse_size(
            plan, model_size=current,
        )
    return plan.size


@pytest.mark.parametrize("incoming,expected", [(2, 40), (3, 80), (4, 158)])
def test_global_pass1_window_uses_incoming_order(incoming, expected):
    # Native10073 I7 order2 -> I8 order3: size40, not80. RELION's MPI
    # expectation sizes Fourier windows before calculateExpectedAngularErrors
    # and updateAngularSampling. The updated order still owns candidate grids.
    assert _global_window(incoming) == expected


def test_controller_sizes_pass1_from_the_incoming_order(monkeypatch):
    """The loop hands the pass-1 sizing the order RELION's expectation starts from, not the updated one."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement import iteration_planning

    trace = CallTrace(monkeypatch)
    trace.wrap(resolution_helpers, "relion_expectation_coarse_size_order", "incoming")
    trace.wrap(iteration_planning, "plan_adaptive_image_size", "plan")
    run_tiny_refinement(monkeypatch, final_after_max_iter=False)
    assert trace.labels() == ["incoming", "plan"] * 2
    for incoming, plan in zip(trace.calls("incoming"), trace.calls("plan"), strict=True):
        assert plan.args[0] == incoming.result


def test_global_pass1_window_keeps_current_size_clamp():
    assert _global_window(3, current=60) == 60


def test_global_pass1_window_keeps_explicit_sealed_override():
    assert _global_window(2, sealed={"coarse_size": 36}) == 36
    with pytest.raises(ValueError, match="exceeds active current_size"):
        _global_window(2, sealed={"coarse_size": 174})
