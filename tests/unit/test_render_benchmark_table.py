"""The RELION-vs-relax benchmark Markdown stays in sync with its JSON baseline."""

import json

import pytest

from scripts.render_benchmark_table import DEFAULT_JSON, DEFAULT_MARKDOWN, load_and_validate, render_markdown


@pytest.mark.unit
def test_markdown_matches_json_baseline():
    rendered = render_markdown(load_and_validate(DEFAULT_JSON))
    assert DEFAULT_MARKDOWN.read_text() == rendered, "run python scripts/render_benchmark_table.py"


@pytest.mark.unit
def test_null_measurement_needs_a_reason(tmp_path):
    table = json.loads(DEFAULT_JSON.read_text())
    row = next(row for row in table["rows"] if row["relax"]["wall_s"] is not None)
    row["relax"]["wall_s"] = None
    row["time_ratio_relax_over_relion"] = None
    row["null_reasons"]["time_ratio_relax_over_relion"] = "test"
    row["null_reasons"].pop("relax.wall_s", None)
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="relax.wall_s"):
        load_and_validate(path)


@pytest.mark.unit
def test_time_ratio_follows_wall_times(tmp_path):
    table = json.loads(DEFAULT_JSON.read_text())
    row = next(row for row in table["rows"] if row["time_ratio_relax_over_relion"] is not None)
    row["time_ratio_relax_over_relion"] += 0.5
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="time ratio"):
        load_and_validate(path)


@pytest.mark.unit
def test_masked_value_needs_the_registered_frozen_mask(tmp_path):
    table = json.loads(DEFAULT_JSON.read_text())
    row = next(row for row in table["rows"] if row["relion"]["masked_resolution_A"] is not None)
    row["mask"]["sha256"] = "0" * 64
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="not the registered frozen mask"):
        load_and_validate(path)
    row["mask"] = None
    row["null_reasons"]["mask"] = "test"
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="without a frozen mask"):
        load_and_validate(path)


@pytest.mark.unit
def test_result_marker_and_per_reference_tables_render(tmp_path):
    table = json.loads(DEFAULT_JSON.read_text())
    row = next(row for row in table["rows"] if row.get("cross_engine_by_relion_run") and row.get("result_marker"))
    rendered = render_markdown(load_and_validate(DEFAULT_JSON))
    symbol = row["result_marker"]["symbol"].replace("*", "\\*")
    assert f"{symbol} {row['dataset']}: {row['result_marker']['note']}" in rendered
    assert "## Comparisons against every RELION run" in rendered
    for entry in row["cross_engine_by_relion_run"]:
        assert f"| {entry['run']} | {', '.join(entry['jobs'])} | {entry['merged']:.4f} |" in rendered
    for entry in row["relion_vs_relion"]:
        assert f"| {entry['pair']} | {entry['merged']:.4f} |" in rendered


@pytest.mark.unit
def test_per_reference_entries_need_every_auc_and_marker_needs_a_note(tmp_path):
    table = json.loads(DEFAULT_JSON.read_text())
    row = next(row for row in table["rows"] if row.get("cross_engine_by_relion_run") and row.get("result_marker"))
    path = tmp_path / "table.json"
    row["cross_engine_by_relion_run"][0]["masked_half1"] = None
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="needs masked_half1 or a null reason"):
        load_and_validate(path)
    row["cross_engine_by_relion_run"][0]["null_reasons"] = {"masked_half1": "test"}
    row["result_marker"]["note"] = ""
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="result_marker needs a symbol and a note"):
        load_and_validate(path)


@pytest.mark.unit
def test_resolution_cells_read_unmasked_then_masked():
    table = load_and_validate(DEFAULT_JSON)
    rendered = render_markdown(table)
    assert "| RELION res (Å) unmasked / masked | relax res (Å) unmasked / masked | Masked X-AUC |" in rendered
    row = next(r for r in table["rows"] if r["relax"]["resolution_A"] is not None and r["relax"]["masked_resolution_A"] is not None)
    letter = "abcdefgh"[list(table["resolution_definitions"]).index(row["relax"]["resolution_definition"])]
    assert f"{row['relax']['resolution_A']:.2f} {letter}" in rendered
    assert f" / {row['relax']['masked_resolution_A']:.2f} |" in rendered


def _initialmodel_row(table):
    return next(row for row in table["rows"] if row.get("table") == "initialmodel" and row["mask"] is not None)


@pytest.mark.unit
def test_initialmodel_rows_render_in_their_own_sections():
    table = load_and_validate(DEFAULT_JSON)
    rendered = render_markdown(table)
    im_rows = [row for row in table["rows"] if row.get("table") == "initialmodel"]
    assert im_rows
    assert "## InitialModel (VDAM): synthetic data" in rendered
    head, _, rest = rendered.partition("## InitialModel (VDAM)")
    for row in im_rows:
        im = row["initial_model"]
        cells = f"{im['relion']['fsc_auc']:.4f} / {im['relax']['fsc_auc']:.4f}"
        assert cells in rest
        assert f"`{row['id']}`" not in head


@pytest.mark.unit
def test_initialmodel_null_needs_a_reason(tmp_path):
    table = json.loads(DEFAULT_JSON.read_text())
    row = _initialmodel_row(table)
    row["initial_model"]["relax"]["masked_fsc_auc"] = None
    row["null_reasons"].pop("initial_model.relax.masked_fsc_auc", None)
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="initial_model.relax.masked_fsc_auc"):
        load_and_validate(path)


@pytest.mark.unit
def test_initialmodel_masked_value_needs_the_registered_frozen_mask(tmp_path):
    table = json.loads(DEFAULT_JSON.read_text())
    row = _initialmodel_row(table)
    row["mask"]["sha256"] = "0" * 64
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="not the registered frozen mask"):
        load_and_validate(path)


@pytest.mark.unit
def test_row_provenance_renders_in_the_note():
    """A row regenerated from transformed relax maps names the note and the corrected map it was scored on."""
    table = load_and_validate(DEFAULT_JSON)
    row = json.loads(json.dumps(table["rows"][0]))
    row["provenance"] = {
        "note": "regenerated post hoc with RELION griddingCorrect on the saved final maps",
        "maps": {"merged": {"corrected": "/c/final_merged.mrc", "corrected_sha256": "a" * 64}},
    }
    table["rows"] = [row]
    rendered = render_markdown(table)
    assert (
        "Provenance: regenerated post hoc with RELION griddingCorrect on the saved final maps; "
        "relax merged `/c/final_merged.mrc` (sha256 `aaaaaaaaaaaaaaaa`)." in rendered
    )
