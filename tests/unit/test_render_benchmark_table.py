"""The RELION-vs-relax benchmark pages (results and provenance) stay in sync with their JSON baseline."""

import json

import pytest

from scripts.render_benchmark_table import (
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    DEFAULT_PROVENANCE,
    TABLE_HEADER,
    load_and_validate,
    status,
    render_markdown,
    render_provenance,
)


@pytest.mark.unit
def test_markdown_matches_json_baseline():
    table = load_and_validate(DEFAULT_JSON)
    assert DEFAULT_MARKDOWN.read_text() == render_markdown(table), "run python scripts/render_benchmark_table.py"
    assert DEFAULT_PROVENANCE.read_text() == render_provenance(table), "run python scripts/render_benchmark_table.py"


@pytest.mark.unit
def test_results_page_holds_tables_and_links_every_row_to_its_provenance():
    """The results page carries no per-row notes; each row links to its anchor on the provenance page."""
    table = load_and_validate(DEFAULT_JSON)
    results = render_markdown(table)
    provenance = render_provenance(table)
    assert "## Notes" not in results and "Jobs:" not in results
    for row in table["rows"]:
        assert f"[notes](relion_vs_relax_provenance.md#{row['id']})" in results
        assert '<a id="' + row["id"] + '"></a>' in provenance
        assert f"`{row['id']}`" in provenance


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
    rendered = render_provenance(load_and_validate(DEFAULT_JSON))
    symbol = row["result_marker"]["symbol"].replace("*", "\\*")
    assert f"{symbol} {row['dataset']}: {row['result_marker']['note']}" in rendered
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
    assert "| RELION res (Å) | relax res (Å) | Quality vs RELION |" in rendered
    assert "reads unmasked / masked" in rendered
    row = next(r for r in table["rows"] if r["relax"]["resolution_A"] is not None and r["relax"]["masked_resolution_A"] is not None)
    letter = "abcdefgh"[list(table["resolution_definitions"]).index(row["relax"]["resolution_definition"])]
    assert f"{row['relax']['resolution_A']:.2f} {letter}" in rendered
    assert f" / {row['relax']['masked_resolution_A']:.2f} |" in rendered


def _initialmodel_row(table):
    return next(row for row in table["rows"] if row.get("table") == "initialmodel" and row["mask"] is not None)


@pytest.mark.unit
def test_one_table_with_workflow_subheaders_in_order():
    """The results page is one table; rows sit under bold workflow subheaders (EM K=1, Class3D, VDAM K=1, VDAM K>1)."""
    table = load_and_validate(DEFAULT_JSON)
    rendered = render_markdown(table)
    assert rendered.count(TABLE_HEADER[0]) == 1
    titles = ["**EM auto-refine (K=1)**", "**Class3D (K>1)**", "**VDAM (K=1)**", "**VDAM (K>1)**"]
    positions = [rendered.index(title) for title in titles]
    assert positions == sorted(positions)
    for row in table["rows"]:
        at = rendered.index(f"#{row['id']})")
        if row.get("table") == "initialmodel":
            title = "**VDAM (K=1)**" if int(row["classes"]) == 1 else "**VDAM (K>1)**"
        else:
            title = "**Class3D (K>1)**" if int(row["classes"]) > 1 else "**EM auto-refine (K=1)**"
        start = rendered.index(title)
        later = [p for p in positions if p > start]
        assert start < at < (min(later) if later else len(rendered)), row["id"]


@pytest.mark.unit
def test_status_mark_follows_quality_and_ratio():
    table = load_and_validate(DEFAULT_JSON)
    rendered = render_markdown(table)
    base = next(r for r in table["rows"] if r["matched"] != "no" and r["time_ratio_relax_over_relion"] is not None)
    cases = [(True, 0.44, "🟢"), (True, 0.6, "🟢"), (True, 0.61, "🟠"), (True, 1.2, "🟠"), (True, 1.21, "🔴"), (False, 0.3, "🔴"), (None, 0.3, "⚪")]
    for passed, ratio, mark in cases:
        row = dict(base, quality_pass=passed, time_ratio_relax_over_relion=ratio)
        assert status(row) == mark, (passed, ratio)
    assert status(dict(base, matched="no")) == "⚪"
    assert status(dict(base, matched="no", quality_pass=False)) == "🔴"
    for row in table["rows"]:
        line = next(x for x in rendered.splitlines() if f"#{row['id']})" in x)
        assert line.startswith(f"| {status(row)} |")
        assert row["quality_reason"] in line


@pytest.mark.unit
def test_every_row_needs_quality_pass_and_a_one_line_reason(tmp_path):
    table = json.loads(DEFAULT_JSON.read_text())
    path = tmp_path / "table.json"
    del table["rows"][0]["quality_pass"]
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="quality_pass"):
        load_and_validate(path)
    table = json.loads(DEFAULT_JSON.read_text())
    table["rows"][0]["quality_reason"] = "a | b"
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="quality_reason"):
        load_and_validate(path)


@pytest.mark.unit
def test_initialmodel_rows_keep_every_auc_on_the_provenance_page():
    """VDAM rows show FSC 0.5 against the reference in the resolution cells; the reference FSC-AUCs, X-AUCs and
    RELION-repeat X-AUC are on the provenance page."""
    table = load_and_validate(DEFAULT_JSON)
    tables = render_markdown(table)
    after = render_provenance(table)
    letters = {name: chr(ord("a") + i) for i, name in enumerate(table["resolution_definitions"])}
    letter = letters["vdam_fsc05_vs_reference"]
    for row in (r for r in table["rows"] if r.get("table") == "initialmodel"):
        im = row["initial_model"]
        line = next(x for x in tables.splitlines() if f"#{row['id']})" in x)
        if im["relion"]["res_05_A"] is not None:
            assert f"| {im['relion']['res_05_A']:.2f} {letter} / " in line
        comparisons = after.partition(f"`{row['id']}`: FSC-AUC of rigidly registered")[2].partition("###")[0]
        for value in (im["relion"]["fsc_auc"], im["relax"]["fsc_auc"], im["cross"]["fsc_auc"], im["cross"]["masked_fsc_auc"]):
            if value is not None:
                assert f"| {value:.4f} |" in comparisons


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
    rendered = render_provenance(table)
    assert (
        "Provenance: regenerated post hoc with RELION griddingCorrect on the saved final maps; "
        "relax merged `/c/final_merged.mrc` (sha256 `aaaaaaaaaaaaaaaa`)." in rendered
    )
