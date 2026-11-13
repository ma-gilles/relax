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
    """The summary: per dataset, each run's scores, and each gate's difference from the first gate at the same seed."""
    reference_gate = gates[0]
    lines = [
        "# VDAM gate benchmark",
        "",
        f"FSC-AUC of each run's map to the benchmark row's reference (masked / unmasked), higher is better; "
        f"differences are each gate minus {reference_gate}, same seed. RELION runs are context.",
        "",
    ]
    for dataset in datasets:
        data = [row for row in rows if row["dataset"] == dataset]
        lines += [f"## {dataset}", "", "| arm | gate | masked FSC-AUC | FSC-AUC | FSC 0.5 (A) | wall (min) |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for row in data:
            wall = "" if row["wall_s"] is None else f"{row['wall_s'] / 60:.1f}"
            resolution = "" if row["res_05_A"] is None else f"{row['res_05_A']:.2f}"
            lines.append(f"| {row['arm']} | {row['gate']} | {row['masked_fsc_auc']:.4f} | {row['fsc_auc']:.4f} | "
                         f"{resolution} | {wall} |")
        by_gate = {gate: {row["seed"]: row for row in data if row["gate"] == gate} for gate in gates}
        lines += ["", f"Difference from {reference_gate} (masked FSC-AUC; mean and sd over seeds):", "",
                  "| gate | " + " | ".join(f"s{seed}" for seed in sorted(by_gate[reference_gate])) + " | mean (sd) |",
                  "| --- |" + " ---: |" * (len(by_gate[reference_gate]) + 1)]
        for gate in gates[1:]:
            seeds = sorted(set(by_gate[reference_gate]) & set(by_gate[gate]))
            differences = [by_gate[gate][s]["masked_fsc_auc"] - by_gate[reference_gate][s]["masked_fsc_auc"] for s in seeds]
            if differences:
                lines.append(f"| {gate} | " + " | ".join(f"{d:+.4f}" for d in differences)
                             + f" | {mean(differences):+.4f} ({pstdev(differences):.4f}) |")
        relion_values = [row["masked_fsc_auc"] for row in data if row["gate"] == "relion_run"]
        if relion_values:
            lines += ["", f"RELION runs, masked FSC-AUC: {min(relion_values):.4f} to {max(relion_values):.4f} "
                          f"({len(relion_values)} runs)."]
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
