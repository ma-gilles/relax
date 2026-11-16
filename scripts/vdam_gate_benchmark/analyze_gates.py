"""Recompute every VDAM gate on traced runs and summarize the gates each run applied and would have applied.

Reads the traces of traced_initial_model.py (gate_trace.jsonl, batches.npz) of one or more runs. For every M-step it
evaluates, on the run's own statistics, the gates of relax.vdam.gate (RELION's amplitude gate, the power gate, the
full-data power gate with independent batches and the one with per-particle coefficients, the latter with the particles
seen so far, and the power gate with the reference's own noise): rho = 2 fudge signal / disagreement,
gate = min(1, max(rho / (1 + rho), floor)). The window gate needs each M-step's step, which older traces do not hold;
its values are then NaN. The gate of the run's own option must reproduce the
gate the run applied (a check of the trace; the largest difference is reported). The summary is the mean gate over shell
bands of the current radius (inner: shells 1..R/3, middle: R/3..2R/3, outer: 2R/3..R) at a few iterations.

Run:
    python scripts/vdam_gate_benchmark/analyze_gates.py --runs OUT/empiar10081_hcn1/relion/s29 ... \
        --output OUT/comparison/gate_traces.json
Each run directory holds trace/gate_trace.jsonl, trace/batches.npz, run_native_options.json and run_it000_data.star.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relax.vdam.gate import (  # noqa: E402
    advance_noise_coefficients,
    advance_noise_weight,
    advance_window_coefficients,
    full_data_disagreement_scale,
    overlap_disagreement_scale,
    seen_disagreement_scale,
    window_disagreement_scale,
)

GATES = ("relion", "power", "full_data_power", "full_data_power_overlap", "seen_power_overlap", "window_power")
SCALED = GATES[2:]
REPORT_ITERATIONS = (5, 10, 20, 30, 50, 75, 100, 150, 195)


def count_particles(data_star: Path) -> int:
    """Rows of the particles loop of a RELION data STAR file."""
    rows, in_particles, in_loop = 0, False, False
    with data_star.open() as handle:
        for line in handle:
            text = line.strip()
            if text.startswith("data_"):
                in_particles, in_loop = text == "data_particles", False
            elif in_particles and text == "loop_":
                in_loop = True
            elif in_particles and in_loop and text and not text.startswith(("_", "#")):
                rows += 1
    return rows


def scales_by_iteration(
    batches: np.lib.npyio.NpzFile, n_particles: int, steps: dict[int, float | None]
) -> dict[int, dict[str, float]]:
    """The scales of the rescaled gates at each gradient iteration, from the run's batches and steps."""
    offsets = np.concatenate([[0], np.cumsum(batches["sizes"])])
    weight = coefficients = moment = reference = None
    scales = {}
    for k, iteration in enumerate(batches["iterations"]):
        ids = batches["ids"][offsets[k] : offsets[k + 1]]
        halves = batches["halves"][offsets[k] : offsets[k + 1]]
        weight = advance_noise_weight(weight, ids.size)
        coefficients = advance_noise_coefficients(coefficients, ids, halves, n_particles)
        step = steps.get(int(iteration))
        if step is not None and (moment is not None or k == 0):
            moment, reference = advance_window_coefficients(moment, reference, ids, halves, n_particles, step)
        else:
            moment = reference = None
        scales[int(iteration)] = {
            "full_data_power": full_data_disagreement_scale(weight, n_particles),
            "full_data_power_overlap": overlap_disagreement_scale(coefficients, n_particles),
            "seen_power_overlap": seen_disagreement_scale(coefficients),
            "window_power": float("nan") if reference is None else window_disagreement_scale(reference, coefficients),
        }
    return scales


def gate_of(record: dict, gate: str, scales: dict[str, float]) -> np.ndarray:
    """The gate ``gate`` on one M-step's statistics, per shell 0..radius (the M-step's formula and clamps)."""
    if gate == "relion":
        signal, disagreement = np.asarray(record["ref_amp"]), np.asarray(record["dis_amp"])
    else:
        signal, disagreement = np.asarray(record["ref_pow"]), np.asarray(record["dis_pow"])
        disagreement = disagreement * (1.0 if gate == "power" else scales[gate])
    rho = np.where(disagreement > 0, 2 * record["fudge"] * signal / np.where(disagreement > 0, disagreement, 1), 0.0)
    value = rho / (1 + rho)
    floor = np.asarray(record["floor"])
    value = np.minimum(np.where(value >= floor, value, floor), 1.0)
    return np.where(np.asarray(record["counts"]) > 0, value, 0.0)


def band_means(values: np.ndarray, radius: int) -> dict[str, float]:
    """Mean of ``values`` over the inner, middle and outer thirds of shells 1..radius."""
    edges = np.linspace(1, radius + 1, 4).round().astype(int)
    return {
        name: float(np.mean(values[lo:hi])) if hi > lo else float("nan")
        for name, lo, hi in zip(("inner", "middle", "outer"), edges[:-1], edges[1:])
    }


def analyze_run(run_dir: Path) -> dict:
    """The applied gate's reproduction error and the band means of every gate at the report iterations."""
    options = json.loads((run_dir / "run_native_options.json").read_text())
    applied_gate = options["vdam_gate"]
    n_particles = count_particles(run_dir / "run_it000_data.star")
    records = [json.loads(line) for line in (run_dir / "trace" / "gate_trace.jsonl").read_text().splitlines()]
    steps = {record["iteration"]: record.get("stepsize") for record in records}
    scales = scales_by_iteration(np.load(run_dir / "trace" / "batches.npz"), n_particles, steps)
    worst = 0.0
    timeline = []
    for record in records:
        iteration = record["iteration"]
        recomputed = {gate: gate_of(record, gate, scales[iteration]) for gate in GATES}
        worst = max(worst, float(np.max(np.abs(recomputed[applied_gate] - np.asarray(record["gate"])))))
        if iteration in REPORT_ITERATIONS:
            timeline.append(
                {
                    "iteration": iteration,
                    "radius": record["radius"],
                    "fudge": record["fudge"],
                    "scales": scales[iteration],
                    "gates": {gate: band_means(recomputed[gate], record["radius"]) for gate in GATES},
                }
            )
    return {
        "run": str(run_dir),
        "gate": applied_gate,
        "n_particles": n_particles,
        "m_steps": len(records),
        "applied_gate_reproduction_max_abs_error": worst,
        "timeline": timeline,
    }


def main() -> None:
    """Parse the command line, analyze each run, write the JSON and print a compact table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [analyze_run(run) for run in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    for result in results:
        print(f"{result['run']}  gate={result['gate']}  N={result['n_particles']}  steps={result['m_steps']}  "
              f"reproduction error={result['applied_gate_reproduction_max_abs_error']:.2e}")
        print("  it   R  fudge " + " ".join(f"{g[:8]:>8s}" for g in SCALED) + " | "
              + " | ".join(f"{g[:14]:>14s} in/mid/out" for g in GATES))
        for row in result["timeline"]:
            cells = " | ".join(
                "{inner:.2f}/{middle:.2f}/{outer:.2f}".format(**row["gates"][g]).rjust(25) for g in GATES
            )
            print(f"  {row['iteration']:3d} {row['radius']:3d} {row['fudge']:6.2f} "
                  + " ".join(f"{row['scales'][g]:8.3f}" for g in SCALED) + f" | {cells}")


if __name__ == "__main__":
    main()
