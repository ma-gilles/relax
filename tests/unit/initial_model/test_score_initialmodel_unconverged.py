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


@pytest.mark.unit
def test_reusable_fits_rebuild_every_stored_transform(tmp_path):
    """--reuse-fits (2026-10-10): an arm whose map sha256 values equal a stored arm's gets back both consensus hands and its
    class fits under the scorer's own keys, so a stored value is reproduced without refitting; another map is refitted."""
    spec = importlib.util.spec_from_file_location("score_initialmodel_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    maps = [tmp_path / f"class{i}.mrc" for i in (1, 2)]
    for i, m in enumerate(maps):
        m.write_bytes(bytes([i]) * 16)
    other = tmp_path / "other.mrc"
    other.write_bytes(b"x" * 16)

    def rec(tag):
        return {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation_voxels": [0, 0, 0], "tag": tag}

    stored = {
        "arms": {
            "old_label": {
                "map_sha256": [module.sha256(str(m)) for m in maps],
                "fit_to_reference": {"hand": "mirror", **rec("cons_mirror"), "other_hand": rec("cons_proper")},
                "class_fits": {"class_to_mean:1:proper": rec("c1p"), "class_to_mean:2:mirror": rec("c2m")},
            }
        }
    }
    path = tmp_path / "stored.json"
    path.write_text(__import__("json").dumps(stored))
    reuse = module._load_reuse([str(path)])
    cell = {
        "arms": [
            {"label": "same_maps", "maps": [str(m) for m in maps]},
            {"label": "new_maps", "maps": [str(other), str(maps[1])]},
        ]
    }
    got = module._reusable_fits(cell, reuse, 2)
    assert set(got) == {"same_maps"}
    recs = got["same_maps"]
    assert recs[("same_maps", "consensus", None, "mirror")]["tag"] == "cons_mirror"
    assert "hand" not in recs[("same_maps", "consensus", None, "mirror")]
    assert recs[("same_maps", "consensus", None, "proper")]["tag"] == "cons_proper"
    assert recs[("same_maps", "class_to_mean", 0, "proper")]["tag"] == "c1p"
    assert recs[("same_maps", "class_to_mean", 1, "mirror")]["tag"] == "c2m"
    # a K>1 arm stored without a class fit is not reusable
    del stored["arms"]["old_label"]["class_fits"]["class_to_mean:2:mirror"]
    path.write_text(__import__("json").dumps(stored))
    assert module._reusable_fits(cell, module._load_reuse([str(path)]), 2) == {}
