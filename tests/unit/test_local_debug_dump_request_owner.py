"""Local debug dump environment handling has one diagnostics owner."""

import os
from contextlib import nullcontext

import pytest

from relax.diagnostics import local_debug


def test_parsers_delegate_to_the_owner(monkeypatch, tmp_path):
    for fn, prefix in (
        (local_debug.parse_debug_score_dump_request, "RELAX_LOCAL_SCORE_DUMP"),
        (local_debug.parse_debug_fused_posterior_dump_request, "RELAX_LOCAL_FUSED_POSTERIOR_DUMP"),
        (local_debug.parse_debug_noise_component_dump_request, "RELAX_LOCAL_NOISE_COMPONENT_DUMP"),
    ):
        for key in ("_DIR", "_GLOBAL_INDICES", "_CURRENT_SIZE", "_ITERATION"):
            monkeypatch.delenv(prefix + key, raising=False)
        assert fn() == (None, set(), None, None)
        monkeypatch.setenv(prefix + "_DIR", str(tmp_path / prefix))
        monkeypatch.setenv(prefix + "_GLOBAL_INDICES", "4,9")
        monkeypatch.setenv(prefix + "_CURRENT_SIZE", "64")
        path, targets, sizes, iterations = fn()
        assert path == tmp_path / prefix and path.is_dir() and targets == {4, 9} and sizes == {64} and iterations is None


@pytest.mark.parametrize("error_type", [None, RuntimeError])
def test_suppress_local_debug_dumps_restores_existing_requests(monkeypatch, error_type):
    existing = {
        "RELAX_LOCAL_SCORE_DUMP_DIR": "score",
        "RELAX_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES": "4,9",
        "RELAX_LOCAL_NOISE_COMPONENT_DUMP_CURRENT_SIZE": "64",
    }
    for name, value in existing.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RELAX_UNRELATED", "keep")

    expected_error = pytest.raises(RuntimeError, match="expected") if error_type else nullcontext()
    with expected_error:
        with local_debug.suppress_local_debug_dumps():
            assert all(name not in os.environ for name in existing)
            assert os.environ["RELAX_UNRELATED"] == "keep"
            # Preserve the old scope's exact update-on-exit behavior.
            monkeypatch.setenv("RELAX_LOCAL_SCORE_DUMP_NEW", "created-inside")
            if error_type:
                raise RuntimeError("expected")

    assert {name: os.environ[name] for name in existing} == existing
    assert os.environ["RELAX_LOCAL_SCORE_DUMP_NEW"] == "created-inside"
