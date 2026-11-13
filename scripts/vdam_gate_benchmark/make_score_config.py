"""Write the scorer config of one dataset of the VDAM gate benchmark.

The config is one cell of scripts/score_initialmodel_maps.py: the reference map and frozen mask of the benchmark row's
cell in docs/benchmarks/initialmodel_scores/config.json, the row's RELION runs of the same seeds (the output directories
in the row's evidence in tests/baselines/relion_vs_relax_benchmarks.json; context), and one relax arm per gate and seed of
this workflow. Pairs: the two gates of each seed, and each relax arm with the RELION run of its seed, where there is one.

Run:
    python scripts/vdam_gate_benchmark/make_score_config.py --cells docs/benchmarks/initialmodel_scores/config.json \
        --cell noise1_50k_k1 --rows tests/baselines/relion_vs_relax_benchmarks.json \
        --row noise1_k1_50k256_initialmodel_vdam_200it \
        --dataset-root OUT/noise1_50k --gates relion full_data_power --seeds 29 41 53 \
        --final-map run_it200_class001.mrc --output OUT/noise1_50k/scoring/score_config.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def arm_label(gate: str, seed: str) -> str:
    """The scorer's label of the relax run of ``gate`` at ``seed`` (a RELION run's label has the prefix ``relion_run_``)."""
    return f"relax_{gate}_s{seed}"


def relion_runs_by_seed(row: dict, seeds: list[str], final_map: str) -> dict[str, Path]:
    """The row's RELION runs of ``seeds``: ``{seed: final map}`` from the output directories of its evidence.

    A listed run of a requested seed whose map is missing is an error. A seed with no listed run has no RELION arm.
    """
    runs: dict[str, Path] = {}
    for directory in row["evidence"].get("relion_runs", []):
        match = re.search(r"/s(\d+)/relion/output$", directory)
        if match and match.group(1) in seeds and match.group(1) not in runs:
            map_path = Path(directory) / final_map
            if not map_path.is_file():
                raise FileNotFoundError(f"RELION map of seed {match.group(1)} is missing: {map_path}")
            runs[match.group(1)] = map_path
    return runs


def build_cell(
    cell: dict, relion_runs: dict[str, Path], dataset_root: Path, gates: list[str], seeds: list[str], final_map: str
) -> dict:
    """The scorer cell: ``cell``'s reference and mask, the RELION runs and the workflow's relax arms and pairs."""
    relion_arms = [
        {"label": f"relion_run_s{seed}", "engine": "relion", "maps": [str(path)]} for seed, path in relion_runs.items()
    ]
    relax_arms, labels = [], {}
    for gate in gates:
        for seed in seeds:
            run_dir = dataset_root / gate / f"s{seed}"
            map_path = run_dir / final_map
            if not map_path.is_file():
                raise FileNotFoundError(f"missing run: {map_path}")
            label = arm_label(gate, seed)
            labels[gate, seed] = label
            relax_arms.append(
                {
                    "label": label,
                    "engine": "relax",
                    "maps": [str(map_path)],
                    "model_star": str(map_path).replace("_class001.mrc", "_model.star"),
                }
            )
    pairs = [[labels[gates[0], seed], labels[gate, seed]] for seed in seeds for gate in gates[1:]]
    pairs += [
        [labels[gate, seed], f"relion_run_s{seed}"] for gate in gates for seed in seeds if seed in relion_runs
    ]
    return {**{key: value for key, value in cell.items() if key not in ("arms", "pairs")},
            "arms": relion_arms + relax_arms, "pairs": pairs}


def main() -> None:
    """Parse the command line, write the config."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cells", required=True, help="docs/benchmarks/initialmodel_scores/config.json")
    parser.add_argument("--cell", required=True, help="id of the benchmark row's cell")
    parser.add_argument("--rows", required=True, help="tests/baselines/relion_vs_relax_benchmarks.json")
    parser.add_argument("--row", required=True, help="id of the benchmark row")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gates", required=True, nargs="+")
    parser.add_argument("--seeds", required=True, nargs="+")
    parser.add_argument("--final-map", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cells = {cell["id"]: cell for cell in json.loads(Path(args.cells).read_text())["cells"]}
    rows = {row["id"]: row for row in json.loads(Path(args.rows).read_text())["rows"]}
    relion_runs = relion_runs_by_seed(rows[args.row], args.seeds, args.final_map)
    cell = build_cell(cells[args.cell], relion_runs, args.dataset_root, args.gates, args.seeds, args.final_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cells": [cell]}, indent=1) + "\n")


if __name__ == "__main__":
    main()
