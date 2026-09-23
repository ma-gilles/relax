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
