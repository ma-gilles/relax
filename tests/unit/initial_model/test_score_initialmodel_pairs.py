"""The InitialModel/VDAM scorer compares only the arm pairs a cell lists; a missing "pairs" key is an error (2026-10-05)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "score_initialmodel_maps.py"


def _module():
    spec = importlib.util.spec_from_file_location("score_initialmodel_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell(**extra):
    return {"id": "c", "arms": [{"label": "a"}, {"label": "b"}, {"label": "c"}], **extra}


@pytest.mark.unit
def test_missing_pairs_is_an_error_naming_the_cell_and_the_all_option():
    with pytest.raises(ValueError, match=r'c: cell has no "pairs" key.*"pairs": "all".*3 serial registrations'):
        _module().wanted_pairs(_cell())


@pytest.mark.unit
def test_pairs_all_empty_and_listed():
    m = _module()
    assert m.wanted_pairs(_cell(pairs="all")) is None
    assert m.wanted_pairs(_cell(pairs=[])) == set()
    assert m.wanted_pairs(_cell(pairs=[["a", "b"], ["c", "a"]])) == {frozenset("ab"), frozenset("ac")}


@pytest.mark.unit
@pytest.mark.parametrize("pair", [["a", "x"], ["a", "a"], ["a", "b", "c"]])
def test_a_pair_must_name_two_different_arms(pair):
    with pytest.raises(ValueError, match="must name two different arms"):
        _module().wanted_pairs(_cell(pairs=[pair]))
