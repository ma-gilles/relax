"""Each controller environment variable is read once, by ``RefinementEnvironment.from_environ`` when the command
builds the run's options (code rule 5), and reaches its consumer as an argument."""

from __future__ import annotations

import pytest
from helpers.tiny_main import controller_inputs
from helpers.tiny_refinement import run_tiny_refinement

from relax.refinement.refinement_options import RefinementEnvironment

pytestmark = pytest.mark.unit


def _from_environ(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    return RefinementEnvironment.from_environ()


def _command_environment(monkeypatch, tmp_path, name, value):
    """The ``options.environment`` ``relax refine`` hands the controller with ``name=value`` set."""
    monkeypatch.setenv(name, value)
    return controller_inputs(monkeypatch, tmp_path, "refine")["options"].environment


def _received(monkeypatch, module, function, keyword, environment, **run):
    """The ``keyword`` arguments ``module.function`` receives in a tiny run with ``environment``."""
    original = getattr(module, function)
    seen = []

    def wrapped(*args, **kwargs):
        seen.append(kwargs[keyword])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, function, wrapped)
    run = {"max_iter": 1, **run}
    run_tiny_refinement(monkeypatch, final_after_max_iter=False, environment=environment, **run)
    return seen


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("  ", None), ("0", False), ("off", False), ("1", True), ("yes", True), ("anything", True)],
)
def test_firstiter_fused_bpref_is_read_once(monkeypatch, value, expected):
    env = _from_environ(monkeypatch, "RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", value)
    assert env.firstiter_fused_bpref is expected
    if value is None:
        assert env == RefinementEnvironment()


def test_firstiter_fused_bpref_reaches_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", "0")
    assert env.firstiter_fused_bpref is False


def test_firstiter_fused_bpref_reaches_the_batch_plan(monkeypatch):
    from relax.refinement import expectation

    monkeypatch.delenv("RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", raising=False)
    seen = _received(monkeypatch, expectation, "prepare_half_batches", "bpref_half",
                     RefinementEnvironment(firstiter_fused_bpref=False))
    assert seen and {half.fused_request for half in seen} == {False}


@pytest.mark.parametrize(("value", "expected"), [(None, frozenset()), (" ", frozenset()), ("4, 7", {4, 7}), ("3,", {3})])
def test_diagnostic_float64_pass2_iterations_are_read_once(monkeypatch, value, expected):
    env = _from_environ(monkeypatch, "RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", value)
    assert env.diagnostic_float64_pass2_iterations == frozenset(expected)


def test_malformed_float64_pass2_iterations_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="comma-separated integers"):
        _from_environ(monkeypatch, "RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "4,bad")


def test_diagnostic_float64_pass2_iterations_reach_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "1")
    assert env.diagnostic_float64_pass2_iterations == frozenset({1})


def test_diagnostic_float64_pass2_iterations_reach_the_expectation(monkeypatch):
    from relax.refinement import expectation

    monkeypatch.delenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", raising=False)
    original = expectation._diagnostic_float64_pass2_matches
    seen = []
    monkeypatch.setattr(
        expectation, "_diagnostic_float64_pass2_matches",
        lambda iteration, iterations: seen.append(iterations) or original(iteration, iterations),
    )
    run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False,
                        environment=RefinementEnvironment(diagnostic_float64_pass2_iterations=frozenset({99})))
    assert seen and set(seen) == {frozenset({99})}


def test_low_pmax_refine_guard_is_read_once(monkeypatch, caplog):
    from relax.refinement.refinement_state import LowPmaxRefineGuard

    for name in ("RELAX_EM_LOW_PMAX_REFINE_MAX_AVE_PMAX", "RELAX_EM_LOW_PMAX_REFINE_MIN_RES_STALL",
                 "RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL"):
        monkeypatch.delenv(name, raising=False)
    assert _from_environ(monkeypatch, "RELAX_EM_LOW_PMAX_REFINE_GUARD", None).low_pmax_refine_guard is None
    assert _from_environ(monkeypatch, "RELAX_EM_LOW_PMAX_REFINE_GUARD", "0").low_pmax_refine_guard is None
    on = _from_environ(monkeypatch, "RELAX_EM_LOW_PMAX_REFINE_GUARD", "1").low_pmax_refine_guard
    assert on == LowPmaxRefineGuard(max_ave_pmax=0.15, min_res_stall=3, require_local_search=True)
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_MAX_AVE_PMAX", "0.2")
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_MIN_RES_STALL", "5")
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL", "no")
    assert RefinementEnvironment.from_environ().low_pmax_refine_guard == LowPmaxRefineGuard(
        max_ave_pmax=0.2, min_res_stall=5, require_local_search=False)
    # An unrecognised value warns and keeps its default, as before.
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL", "maybe")
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_MIN_RES_STALL", "x")
    guard = RefinementEnvironment.from_environ().low_pmax_refine_guard
    assert guard.require_local_search is True and guard.min_res_stall == 3
    assert "RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL" in caplog.text


def test_low_pmax_refine_guard_reaches_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_EM_LOW_PMAX_REFINE_GUARD", "1")
    assert env.low_pmax_refine_guard is not None


def test_low_pmax_refine_guard_reaches_the_sampling_update(monkeypatch):
    from relax.refinement import numbered_transitions
    from relax.refinement.refinement_state import LowPmaxRefineGuard

    monkeypatch.delenv("RELAX_EM_LOW_PMAX_REFINE_GUARD", raising=False)
    guard = LowPmaxRefineGuard(max_ave_pmax=0.5, min_res_stall=2, require_local_search=False)
    seen = _received(monkeypatch, numbered_transitions, "advance_expectation_sampling", "low_pmax_guard",
                     RefinementEnvironment(low_pmax_refine_guard=guard), max_iter=2)
    assert seen and set(seen) == {guard}
    seen = _received(monkeypatch, numbered_transitions, "update_refinement_state", "low_pmax_guard",
                     RefinementEnvironment(low_pmax_refine_guard=guard), max_iter=2)
    assert seen and set(seen) == {guard}
