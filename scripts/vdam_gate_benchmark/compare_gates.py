"""Tabulate the VDAM gate benchmark: the scores of every run, and the gate difference per seed and dataset.

Reads, per dataset, the scorer output <output_root>/<dataset>/scoring/scores.json and the wall times of the runs
(<output_root>/<dataset>/<gate>/s<seed>/wall.tsv, Snakemake's benchmark files). Writes a CSV of every run (the
dataset's RELION runs included, as context) and a Markdown summary: per dataset the masked and unmasked FSC-AUC of the
reference for each gate at each seed, and the difference of the second gate from the first.

Run:
    python scripts/vdam_gate_benchmark/compare_gates.py --output-root OUT --datasets noise1_50k empiar10097 \
        --gates relion full_data_power --csv OUT/comparison/gate_comparison.csv \
        --markdown OUT/comparison/gate_comparison.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

METRICS = ("masked_fsc_auc", "fsc_auc")


def arm_row(arm: dict) -> dict:
    """The reference scores of one scorer arm: population-weighted FSC-AUCs and the FSC 0.5 resolution (Angstrom)."""
    reference = arm["vs_reference"]
    return {
        "masked_fsc_auc": reference["weighted_masked_fsc_auc"],
        "fsc_auc": reference["weighted_fsc_auc"],
        "res_05_A": reference["per_class"][0].get("res_05_A"),
    }


def wall_seconds(path: Path) -> float | None:
    """The wall time of one run from Snakemake's benchmark file, or None when it was not recorded."""
    if not path.is_file():
        return None
    header, values = path.read_text().splitlines()[:2]
    return float(dict(zip(header.split("\t"), values.split("\t")))["s"])


def collect(output_root: Path, datasets: list[str], gates: list[str]) -> list[dict]:
    """One row per scored arm of every dataset: its gate (``relion_run`` for the RELION runs), seed and scores."""
    rows = []
    for dataset in datasets:
        scores = json.loads((output_root / dataset / "scoring" / "scores.json").read_text())
        for label, arm in scores["arms"].items():
            if arm["engine"] == "relion":
                gate, seed = "relion_run", label
            else:
                gate, _, seed = label.removeprefix("relax_").rpartition("_s")
                if gate not in gates:
                    continue
            wall = wall_seconds(output_root / dataset / gate / f"s{seed}" / "wall.tsv") if arm["engine"] == "relax" else None
            rows.append({"dataset": dataset, "gate": gate, "seed": seed, "arm": label, **arm_row(arm), "wall_s": wall})
    return rows


def markdown(rows: list[dict], datasets: list[str], gates: list[str]) -> str:
    """The summary: per dataset, each gate's scores by seed and the difference of the last gate from the first."""
    reference_gate, other = gates[0], gates[-1]
    lines = [
        "# VDAM gate benchmark",
        "",
        f"FSC-AUC of each run's map to the benchmark row's reference (masked / unmasked), higher is better; "
        f"difference = {other} minus {reference_gate}, same seed. RELION runs are context.",
        "",
    ]
    for dataset in datasets:
        data = [row for row in rows if row["dataset"] == dataset]
        lines += [f"## {dataset}", "", "| arm | gate | masked FSC-AUC | FSC-AUC | FSC 0.5 (A) | wall (min) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for row in data:
            wall = "" if row["wall_s"] is None else f"{row['wall_s'] / 60:.1f}"
            resolution = "" if row["res_05_A"] is None else f"{row['res_05_A']:.2f}"
            lines.append(f"| {row['arm']} | {row['gate']} | {row['masked_fsc_auc']:.4f} | {row['fsc_auc']:.4f} | {resolution} | {wall} |")
        by_gate = {gate: {row["seed"]: row for row in data if row["gate"] == gate} for gate in (reference_gate, other)}
        seeds = sorted(set(by_gate[reference_gate]) & set(by_gate[other]))
        lines += ["", f"Difference, {other} minus {reference_gate}:", "", "| seed | masked FSC-AUC | FSC-AUC |", "| --- | ---: | ---: |"]
        differences = {metric: [by_gate[other][s][metric] - by_gate[reference_gate][s][metric] for s in seeds] for metric in METRICS}
        for index, seed in enumerate(seeds):
            lines.append(f"| {seed} | {differences['masked_fsc_auc'][index]:+.4f} | {differences['fsc_auc'][index]:+.4f} |")
        if seeds:
            lines.append(
                "| mean (sd) | "
                + " | ".join(f"{mean(differences[m]):+.4f} ({pstdev(differences[m]):.4f})" for m in METRICS)
                + " |"
            )
        relion_values = [row["masked_fsc_auc"] for row in data if row["gate"] == "relion_run"]
        if relion_values:
            lines += ["", f"RELION runs, masked FSC-AUC: {min(relion_values):.4f} to {max(relion_values):.4f} ({len(relion_values)} runs)."]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Parse the command line, write the CSV and the Markdown summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--datasets", required=True, nargs="+")
    parser.add_argument("--gates", required=True, nargs="+")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    rows = collect(args.output_root, args.datasets, args.gates)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.markdown.write_text(markdown(rows, args.datasets, args.gates) + "\n")


if __name__ == "__main__":
    main()
