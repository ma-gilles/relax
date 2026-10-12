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
    run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False, environment=environment, **run)
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
