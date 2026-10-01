"""Label scopes preserve dump names, nesting and environment restoration."""

import os
from contextlib import nullcontext

import pytest

from relax.diagnostics import local_debug

pytestmark = pytest.mark.unit
DENSE = "RELAX_DEBUG_PER_POSE_DUMP_LABEL"
SCORE = "RELAX_LOCAL_SCORE_DUMP_LABEL"
FUSED = "RELAX_LOCAL_FUSED_POSTERIOR_DUMP_LABEL"
NAMES = (DENSE, SCORE, FUSED)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("initial,expected", [(None, "phase"), ("", "phase"), ("arm", "arm_phase")])
@pytest.mark.parametrize("error_type", [None, RuntimeError, KeyboardInterrupt])
def test_scope_restores_absent_empty_and_existing_labels(monkeypatch, local, initial, expected, error_type):
    for name in NAMES:
        if initial is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, initial)
    original = {name: os.environ.get(name) for name in NAMES}
    changed = (SCORE, FUSED) if local else (DENSE,)
    check_error = pytest.raises(error_type, match="expected") if error_type else nullcontext()
    with check_error:
        with local_debug.score_dump_label("phase", local=local) as value:
            assert value is None
            assert {name: os.environ.get(name) for name in changed} == dict.fromkeys(changed, expected)
            for name in set(NAMES) - set(changed):
                assert os.environ.get(name) == original[name]
            if error_type:
                raise error_type("expected")
    assert {name: os.environ.get(name) for name in NAMES} == original


