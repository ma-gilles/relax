"""Local debug dump environment handling has one diagnostics owner."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
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


def test_record_local_search_profile_owns_history_and_file_capture(tmp_path):
    history = []
    profile = {"elapsed": np.array([1.25], dtype=np.float32)}
    diagnostics = SimpleNamespace(
        collect_local_search_profile=True,
        iteration=3,
        diagnostic_score_only=True,
        local_profile_history=history,
        save_intermediates_dir=tmp_path,
    )

    local_debug.record_local_search_profile(
        SimpleNamespace(profile_summary=profile),
        diagnostics,
        half_index=1,
        parent_mode="full_parent",
    )

    assert len(history) == 1
    assert history[0]["elapsed"] is profile["elapsed"]
    assert history[0]["iteration"] == np.int32(3)
    assert history[0]["half_index"] == np.int32(1)
    assert history[0]["local_adaptive_pass2_parent_mode"] == "full_parent"
    assert history[0]["local_adaptive_pass2_full_parent"] == np.bool_(True)
    assert history[0]["diagnostic_score_only"] == np.bool_(True)
    saved = np.load(tmp_path / "it003_half2_local_profile.npz")
    assert set(saved.files) == {"elapsed"}
    np.testing.assert_array_equal(saved["elapsed"], profile["elapsed"])


def test_record_local_search_profile_is_inert_when_not_requested():
    local_debug.record_local_search_profile(
        SimpleNamespace(),
        SimpleNamespace(collect_local_search_profile=False),
        half_index=0,
        parent_mode="none",
    )
