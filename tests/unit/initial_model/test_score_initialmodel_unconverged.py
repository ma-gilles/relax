"""The InitialModel/VDAM scorer lists every registration whose optimizer stopped on its budget (2026-10-03)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "score_initialmodel_maps.py"


@pytest.mark.unit
def test_unconverged_fits_are_listed_with_their_path():
    spec = importlib.util.spec_from_file_location("score_initialmodel_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    good = {"lowpass_optimizer_success": True, "fine_optimizer_success": True, "fine_optimizer_message": "ok"}
    bad_fine = {
        **good,
        "fine_optimizer_success": False,
        "fine_optimizer_message": "Maximum number of function evaluations has been exceeded.",
    }
    bad_lowpass = {**good, "lowpass_optimizer_success": False}
    cell = {
        "arms": {"a": {"fit_to_reference": good, "class_fits": {"class_to_mean:2:proper": bad_fine}}},
        "pairs": [{"pair_fit": bad_lowpass}],
    }
    found = module._unconverged_fits(cell)
    assert [f["fit"] for f in found] == ["/arms/a/class_fits/class_to_mean:2:proper", "/pairs[0]/pair_fit"]
    assert found[0]["fine_optimizer_success"] is False and found[0]["lowpass_optimizer_success"] is True
    assert found[1]["lowpass_optimizer_success"] is False
    assert module._unconverged_fits({"arms": {"a": {"fit_to_reference": good}}, "pairs": []}) == []
