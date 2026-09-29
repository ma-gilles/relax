"""CPU checks that a tier choice requires a real executed global call."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from helpers import coarse_engine_selection as selection

pytestmark = pytest.mark.unit


def _call(choice: str, *, cc: bool = False) -> dict:
    dense = choice == "gemm_dense"
    images, classes, coarse, fine = 3, 4, 20, 80
    return {
        "requested": choice,
        "resolved": choice,
        "strategy": "dense" if dense else "hybrid",
        "fine_engine": None if dense else "resident",
        "images": images,
        "classes": classes,
        "rotations": 10 if dense else 5,
        "translations": 2 if dense else 1,
        "hypotheses_per_image": fine if dense else coarse,
        "coarse_candidates_per_image": coarse,
        "fine_candidates_per_image": fine,
        "evaluated_fine_candidates_total": images * fine if dense else 30,
        "selected_fine_candidates_total": (images if cc else images * fine) if dense else (images if cc else 15),
        "selected_fine_candidates_known": True,
        "current_size": 94,
        "reconstruction_current_size": 128,
        "score_mode": "normalized_cc" if cc else "gaussian",
        "posterior_policy": "cc_winner" if cc else "gaussian",
        "pruned": False if dense else True,
        "precision": {"score": "float32", "projection": "float32", "mstep": "float32"},
    }


@pytest.mark.parametrize("choice", ["gemm_hybrid", "gemm_dense"])
@pytest.mark.parametrize("cc", [False, True])
def test_selected_standard_record_requires_actual_complete_scoring(tmp_path, monkeypatch, choice, cc):
    monkeypatch.setenv(selection.ENV_NAME, choice)
    record = _call(choice, cc=cc)
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]),
             final_all_data_coarse_engines=json.dumps([]))
    selection.assert_selected_global_execution(tmp_path)
    record["resolved"] = "auto"
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]))
    with pytest.raises(AssertionError, match="fell back"):
        selection.assert_selected_global_execution(tmp_path)


def test_dense_missing_evaluation_or_pruning_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(selection.ENV_NAME, "gemm_dense")
    record = _call("gemm_dense")
    record["evaluated_fine_candidates_total"] -= 1
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]))
    with pytest.raises(AssertionError, match="fine candidate counts|complete expanded grid"):
        selection.assert_selected_global_execution(tmp_path)


@pytest.mark.parametrize("invalid", [-1, float("nan"), None])
def test_hybrid_rejects_invalid_or_missing_candidate_count(tmp_path, monkeypatch, invalid):
    monkeypatch.setenv(selection.ENV_NAME, "gemm_hybrid")
    record = _call("gemm_hybrid")
    record["evaluated_fine_candidates_total"] = invalid
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]))
    with pytest.raises(AssertionError, match="evaluated_fine_candidates_total"):
        selection.assert_selected_global_execution(tmp_path)


@pytest.mark.parametrize("cc", [False, True])
def test_hybrid_accepts_truthful_unknown_selected_count(tmp_path, monkeypatch, cc):
    monkeypatch.setenv(selection.ENV_NAME, "gemm_hybrid")
    record = _call("gemm_hybrid", cc=cc)
    record.update(selected_fine_candidates_total=None, selected_fine_candidates_known=False)
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]))
    selection.assert_selected_global_execution(tmp_path)


@pytest.mark.parametrize("choice", ["gemm_hybrid", "gemm_dense"])
@pytest.mark.parametrize("known,selected", [
    (False, 1), (True, None), (None, None), ("false", None),
])
def test_selected_count_known_flag_and_value_must_agree(tmp_path, monkeypatch, choice, known, selected):
    monkeypatch.setenv(selection.ENV_NAME, choice)
    record = _call(choice)
    record["selected_fine_candidates_known"] = known
    record["selected_fine_candidates_total"] = selected
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]))
    with pytest.raises(AssertionError, match="selected_fine_candidates"):
        selection.assert_selected_global_execution(tmp_path)


@pytest.mark.parametrize("field", ["selected_fine_candidates_total", "selected_fine_candidates_known"])
def test_selected_count_missing_field_fails(tmp_path, monkeypatch, field):
    monkeypatch.setenv(selection.ENV_NAME, "gemm_hybrid")
    record = _call("gemm_hybrid")
    del record[field]
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]))
    with pytest.raises(AssertionError, match="selected_fine_candidates"):
        selection.assert_selected_global_execution(tmp_path)


def test_dense_rejects_unknown_selected_count(tmp_path, monkeypatch):
    monkeypatch.setenv(selection.ENV_NAME, "gemm_dense")
    record = _call("gemm_dense")
    record.update(selected_fine_candidates_total=None, selected_fine_candidates_known=False)
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]))
    with pytest.raises(AssertionError, match="complete expanded grid"):
        selection.assert_selected_global_execution(tmp_path)


@pytest.mark.parametrize("value", [-1, 31])
def test_hybrid_selected_count_must_fit_evaluated_count(tmp_path, monkeypatch, value):
    monkeypatch.setenv(selection.ENV_NAME, "gemm_hybrid")
    record = _call("gemm_hybrid")
    record["selected_fine_candidates_total"] = value
    np.savez(tmp_path / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[record]]))
    with pytest.raises(AssertionError, match="selected_fine_candidates_total|fine candidate counts"):
        selection.assert_selected_global_execution(tmp_path)


def test_vdam_record_and_missing_global_call(tmp_path, monkeypatch):
    monkeypatch.setenv(selection.ENV_NAME, "gemm_hybrid")
    (tmp_path / "run_it001_recovar_meta.json").write_text(json.dumps({"coarse_engine_calls": [_call("gemm_hybrid")]}))
    selection.assert_selected_global_execution(tmp_path, output_kind="vdam")
    (tmp_path / "run_it001_recovar_meta.json").write_text(json.dumps({"coarse_engine_calls": []}))
    with pytest.raises(AssertionError, match="no global coarse calls"):
        selection.assert_selected_global_execution(tmp_path, output_kind="vdam")


def test_auto_has_no_extra_assertion_and_selected_flag_is_public(monkeypatch, tmp_path):
    monkeypatch.setenv(selection.ENV_NAME, "auto")
    assert selection.selected_cli_args() == []
    selection.assert_selected_global_execution(tmp_path)
    monkeypatch.setenv(selection.ENV_NAME, "gemm_dense")
    assert selection.selected_cli_args() == ["--coarse-engine", "gemm_dense"]
    monkeypatch.setenv(selection.ENV_NAME, "invalid")
    with pytest.raises(ValueError, match=selection.ENV_NAME):
        selection.selected_cli_args()


@pytest.mark.parametrize("choice", ["gemm_hybrid", "gemm_dense"])
def test_selected_kclass_replay_uses_cuda_operands(monkeypatch, choice):
    monkeypatch.setenv(selection.ENV_NAME, choice)
    seen = []

    def capture(argv, **_kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(selection.subprocess, "run", capture)
    selection.run_selected_command(["python", "run_k_class_parity.py"], output_kind="kclass")
    assert seen == [["python", "run_k_class_parity.py", "--coarse-engine", choice,
                     "--image-fourier-backend", "relion_cuda", "--accumulate-noise"]]
