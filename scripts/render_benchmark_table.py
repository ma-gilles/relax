"""Render the RELION-vs-relax benchmark table from its JSON baseline.

The JSON (tests/baselines/relion_vs_relax_benchmarks.json) is the record; the
Markdown (docs/benchmarks/relion_vs_relax.md) is generated from it. After a
baseline changes, edit the JSON and run

    python scripts/render_benchmark_table.py

``--check`` exits non-zero when the Markdown is stale. Validation rejects a
null measurement without a reason, a time ratio that does not follow from
the row's two wall times, and a time ratio without its like-for-like record
(``time_check`` a-d: GPU and MPI layout, relax diagnostic options, timing
scope, GPU model per job). A masked value must name the dataset's frozen mask
from docs/benchmarks/frozen_masks.json by key and SHA-256.

Optional per-row fields: ``result_marker`` ({symbol, note}) flags the relax
result and adds a footnote; ``cross_engine_by_relion_run`` and
``relion_vs_relion`` list the band FSC-AUCs of relax against every same-command
RELION run and of the RELION runs against each other (``render_per_reference``).
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO / "tests" / "baselines" / "relion_vs_relax_benchmarks.json"
DEFAULT_MARKDOWN = REPO / "docs" / "benchmarks" / "relion_vs_relax.md"
DEFAULT_REGISTRY = REPO / "docs" / "benchmarks" / "frozen_masks.json"

ENGINES = ("relion", "relax")
ENGINE_FIELDS = (
    "resolution_A",
    "masked_resolution_A",
    "masked_band_auc",
    "wall_s",
    "gpu_model",
    "gpu_count",
    "iterations",
)
ROW_FIELDS = ("gt", "cross_engine", "cross_engine_masked_band_auc", "time_ratio_relax_over_relion", "mask")
MASKED_DEFINITION = "relion_postprocess_masked_frozen"
SECTIONS = (("real", "Real data"), ("synthetic", "Synthetic data"))
MATCHED = ("yes", "workload", "no")
RATIO_TOLERANCE = 0.006
PER_REFERENCE_AUCS = ("merged", "half1", "half2")
MASKED_PER_REFERENCE_AUCS = ("masked_merged", "masked_half1", "masked_half2")
# Null reasons repeated in the rendered notes: the fields the table shows.
SHOWN_NULLS = (
    "resolution_A",
    "masked_resolution_A",
    "wall_s",
    "time_ratio_relax_over_relion",
    "gt",
    "relax",
    "relion",
    "mask",
)


def load_and_validate(path, registry=DEFAULT_REGISTRY):
    table = json.loads(Path(path).read_text())
    definitions = table["resolution_definitions"]
    masks = json.loads(Path(registry).read_text())["masks"]
    ids = [row["id"] for row in table["rows"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row id")
    for row in table["rows"]:
        _validate_row(row, definitions)
        _validate_masked(row, masks)
        _validate_per_reference(row)
    return table


def _validate_masked(row, masks):
    """Masked values come only from the dataset's registered frozen mask."""
    rid = row["id"]
    masked = [row[e][f] for e in ENGINES for f in ("masked_resolution_A", "masked_band_auc")]
    masked.append(row["cross_engine_masked_band_auc"])
    if all(value is None for value in masked):
        return
    mask = row["mask"]
    if mask is None:
        raise ValueError(f"{rid}: masked values without a frozen mask")
    if masks.get(mask["dataset"], {}).get("mask_sha256") != mask["sha256"]:
        raise ValueError(f"{rid}: mask {mask['dataset']} is not the registered frozen mask")
    for engine in ENGINES:
        if row[engine]["masked_resolution_A"] is not None:
            if row[engine].get("masked_resolution_definition") != MASKED_DEFINITION:
                raise ValueError(f"{rid}: {engine} masked resolution must use {MASKED_DEFINITION}")


def _validate_per_reference(row):
    """A result marker carries its footnote; per-reference entries carry every band AUC."""
    rid = row["id"]
    marker = row.get("result_marker")
    if marker is not None and not (marker.get("symbol") and marker.get("note")):
        raise ValueError(f"{rid}: result_marker needs a symbol and a note")
    for key, label_key in (("cross_engine_by_relion_run", "run"), ("relion_vs_relion", "pair")):
        for entry in row.get(key, []):
            if not entry.get(label_key):
                raise ValueError(f"{rid}: {key} entry without a {label_key}")
            fields = PER_REFERENCE_AUCS + (MASKED_PER_REFERENCE_AUCS if key == "cross_engine_by_relion_run" else ())
            for field in fields:
                if not isinstance(entry.get(field), (int, float)) and not entry.get("null_reasons", {}).get(field):
                    raise ValueError(f"{rid}: {key} {entry[label_key]} needs {field} or a null reason")


def _validate_row(row, definitions):
    rid = row["id"]
    if row["section"] not in dict(SECTIONS):
        raise ValueError(f"{rid}: unknown section {row['section']!r}")
    if row["matched"] not in MATCHED:
        raise ValueError(f"{rid}: matched must be one of {MATCHED}")
    reasons = row.get("null_reasons", {})
    nulls = [f"{engine}.{field}" for engine in ENGINES for field in ENGINE_FIELDS if row[engine][field] is None]
    nulls += [field for field in ROW_FIELDS if row[field] is None]
    if row["gt"] is not None:
        nulls += [f"gt.{engine}" for engine in ENGINES if row["gt"][engine] is None]
    missing = [field for field in nulls if not reasons.get(field)]
    if missing:
        raise ValueError(f"{rid}: null without a reason: {missing}")
    for engine in ENGINES:
        if row[engine]["resolution_definition"] not in definitions:
            raise ValueError(f"{rid}: unknown resolution definition for {engine}")
    ratio = row["time_ratio_relax_over_relion"]
    walls = (row["relion"]["wall_s"], row["relax"]["wall_s"])
    if ratio is not None:
        if set(row.get("time_check", {})) != set("abcd"):
            raise ValueError(f"{rid}: a time ratio needs time_check entries a-d")
        if None in walls:
            raise ValueError(f"{rid}: time ratio without both wall times")
        if abs(walls[1] / walls[0] - ratio) > RATIO_TOLERANCE:
            raise ValueError(f"{rid}: time ratio {ratio} != {walls[1]}/{walls[0]}")


def render_markdown(table):
    letters = {name: chr(ord("a") + i) for i, name in enumerate(table["resolution_definitions"])}
    lines = [
        "# RELION vs relax benchmarks",
        "",
        "<!-- Generated by scripts/render_benchmark_table.py; do not edit by hand. -->",
        "",
        "Regenerate with `python scripts/render_benchmark_table.py` after updating the JSON when a baseline changes.",
        "",
        "Source of record: [`tests/baselines/relion_vs_relax_benchmarks.json`](../../tests/baselines/relion_vs_relax_benchmarks.json)"
        f" (updated {table['updated']}). relax is the RECOVAR EM code; rows that ran before the relax split"
        " cite the RECOVAR source SHA in the JSON. Ratio is relax wall / RELION wall (lower is faster for relax).",
        "",
        "Resolution definitions (letter after each value):",
        "",
    ]
    lines += [f"- **{letters[name]}**: {text}" for name, text in table["resolution_definitions"].items()]
    lines += [
        "",
        "Masked columns: relion_postprocess corrected masked resolution with the dataset's frozen mask (RELION's"
        " convention, `rlnFinalResolution`); Masked X-AUC is the cross-engine FSC-AUC of the two merged maps, both"
        " multiplied by that mask, over the scorecard band. Reporting only; no gate reads them. Masks, method and"
        " per-run curves: [masked FSC](masked_fsc.md).",
        "",
        "Matched column:",
        "",
    ]
    lines += [f"- **{key}**: {text}" for key, text in table["matched_definition"].items()]
    footnotes = []
    for section, title in SECTIONS:
        rows = [row for row in table["rows"] if row["section"] == section]
        lines += ["", f"## {title}", ""]
        lines += [
            "| Dataset | Workflow | N / box | RELION res (Å) | relax res (Å) | RELION masked (Å) | relax masked (Å)"
            " | Masked X-AUC | RELION time | relax time | Ratio | GPU | Matched? | Date |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for row in rows:
            footnotes.append(row)
            mark = f"[{len(footnotes)}]"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{row['dataset']} {mark}",
                        _workflow(row),
                        f"{row['particles']:,} / {row['box']}",
                        _resolution(row, "relion", letters),
                        _resolution(row, "relax", letters),
                        _masked(row, "relion"),
                        _masked(row, "relax"),
                        _masked_cross(row),
                        _time(row, "relion"),
                        _time(row, "relax"),
                        _ratio(row),
                        _gpu(row),
                        row["matched"],
                        row["date"],
                    ]
                )
                + " |"
            )
    lines += ["", "## Notes", ""]
    for index, row in enumerate(footnotes, start=1):
        lines.append(f"{index}. {_note(row)}")
    for row in footnotes:
        if row.get("result_marker"):
            marker = row["result_marker"]
            lines += ["", f"{_escape(marker['symbol'])} {row['dataset']}: {marker['note']}"]
    per_reference = [row for row in footnotes if row.get("cross_engine_by_relion_run")]
    if per_reference:
        lines += ["", "## Comparisons against every RELION run", ""]
        for row in per_reference:
            lines += [f"### {row['dataset']}", "", *render_per_reference(row), ""]
        lines.pop()
    lines += ["", "## Related scorecards", ""]
    lines += [
        "- [Accepted-best completion ledger](../math/em_parity_best_metrics.md)",
        "- [K=1 real-data science-equivalence scorecard](../math/em_k1_realdata_science_equivalence_scorecard.md)",
        "- [K=1 34-case and K=4 synthetic parity scorecard](../math/em_relion_parity_scorecard.md)",
        "- [K=4 per-class FSC-AUC scorecard](../math/em_k4_class_fsc_auc_scorecard.md)",
        "- [VDAM InitialModel parity scorecard](../math/vdam_relion_parity_scorecard.md)",
    ]
    return "\n".join(lines) + "\n"


def render_per_reference(row):
    """Band FSC-AUC tables: relax against each same-command RELION run, then RELION against RELION."""
    lines = [
        "relax against each same-command RELION run (band FSC-AUC over the scorecard band; masked columns use"
        " the frozen mask; thresholds: merged >= 0.95 and each half >= 0.90):",
        "",
        "| RELION run | Jobs | Merged | Half 1 | Half 2 | Masked merged | Masked half 1 | Masked half 2 | Thresholds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in row["cross_engine_by_relion_run"]:
        cells = [entry["run"], ", ".join(entry.get("jobs", [])) or "—"]
        cells += [_auc(entry.get(field)) for field in PER_REFERENCE_AUCS + MASKED_PER_REFERENCE_AUCS]
        cells.append("met" if entry.get("meets_thresholds") else "not met")
        lines.append("| " + " | ".join(cells) + " |")
    if row.get("relion_vs_relion"):
        lines += [
            "",
            "RELION against RELION (same band FSC-AUCs):",
            "",
            "| Pair | Merged | Half 1 | Half 2 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for entry in row["relion_vs_relion"]:
            lines.append("| " + " | ".join([entry["pair"], *(_auc(entry[f]) for f in PER_REFERENCE_AUCS)]) + " |")
    return lines


def _auc(value):
    return "—" if value is None else f"{value:.4f}"


def _escape(symbol):
    return symbol.replace("*", "\\*")


def _workflow(row):
    return f"{row['workflow']}, {row['symmetry']}, {row['relion_dependence']}"


def _resolution(row, engine, letters):
    value = row[engine]["resolution_A"]
    if value is None:
        return "pending" if row["status"] == "pending" and engine == "relax" else "—"
    marker = _escape(row["result_marker"]["symbol"]) if engine == "relax" and row.get("result_marker") else ""
    return f"{value:.2f} {letters[row[engine]['resolution_definition']]}{marker}"


def _masked(row, engine):
    value = row[engine]["masked_resolution_A"]
    if value is None:
        return "pending" if row["status"] == "pending" and engine == "relax" else "—"
    return f"{value:.2f}"


def _masked_cross(row):
    value = row["cross_engine_masked_band_auc"]
    return "—" if value is None else f"{value:.4f}"


def _time(row, engine):
    wall = row[engine]["wall_s"]
    if wall is None:
        return "pending" if row["status"] == "pending" and engine == "relax" else "—"
    return f"{wall:,.0f} s"


def _ratio(row):
    ratio = row["time_ratio_relax_over_relion"]
    return "—" if ratio is None else f"{ratio:.2f}x"


def _gpu(row):
    parts = [(row[e]["gpu_count"], row[e]["gpu_model"]) for e in ENGINES]
    names = [f"{count}x {model.split()[0]}" if model else "—" for count, model in parts]
    return names[0] if names[0] == names[1] else f"RELION {names[0]}, relax {names[1]}"


def _note(row):
    relax = row["relax"]
    text = [f"`{row['id']}`: relax source `{relax['source_sha'][:9]}`, {relax['source_repo']}."]
    if row["mask"] is not None:
        mask = row["mask"]
        auc = [row[e]["masked_band_auc"] for e in ENGINES]
        text.append(
            f"Frozen mask `{mask['dataset']}` (`{mask['sha256'][:12]}`); masked FSC-AUC over the scorecard band"
            f" RELION {_value(auc[0])}, relax {_value(auc[1])}."
        )
    if row["gt"] is not None:
        gt = row["gt"]
        text.append(f"GT {gt['metric']}: RELION {_value(gt['relion'])}, relax {_value(gt['relax'])}.")
    if row["cross_engine"]:
        text.append(f"Cross-engine: {row['cross_engine'].rstrip('.')}.")
    text += [f"{note}" for note in row.get("notes", [])]
    text += [
        f"`{field}` is null: {reason}."
        for field, reason in row.get("null_reasons", {}).items()
        if field.split(".")[-1] in SHOWN_NULLS
    ]
    if "pending" in row:
        pending = row["pending"]
        text.append(f"Pending: {pending['note']} (job {', '.join(pending['jobs'])}).")
    jobs = relax["jobs"] + row["relion"]["jobs"]
    if jobs:
        text.append(f"Jobs: relax {', '.join(relax['jobs']) or '—'}; RELION {', '.join(row['relion']['jobs']) or '—'}.")
    return " ".join(text)


def _value(value):
    return "—" if value is None else f"{value:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true", help="fail if the Markdown is stale")
    args = parser.parse_args()
    rendered = render_markdown(load_and_validate(args.json))
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit(f"{args.output} is stale; run python scripts/render_benchmark_table.py")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
