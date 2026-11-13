"""Collect a run's per-shell gate trace from iterations.jsonl and compare it with the dataset's oracle tables.

Writes ``<run>/gate_trace.npz`` (or ``-o``): per logged iteration the radius, fudge and step, and [iteration, shell, 2]
arrays of the gate (and, when logged, the estimated gate of an oracle run, the effective gate and sqrt(u)), the signal and
the scaled disagreement. ``summarise`` compares a trace with an oracle npz (scripts/gate_study/oracle_gate.py):

- ``gap``: gate minus the task's oracle W = snr / (1 + snr) (fudge 1), mean over the shells 1..radius;
- ``gap_target``: gate minus the code's own target at the iteration's fudge, the half-map FSC fudge snr/2 / (1 + fudge snr/2);
- ``signal_ratio``: logged signal over the true power in the model's units; ``noise_ratio``: logged scaled disagreement
  over its target 2 V_half = 4 P / snr.

Run: python scripts/gate_study/gate_trace.py RUN_DIR [--oracle ORACLE.npz]
"""

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("gates", "gates_estimated", "signal", "disagreement", "effective_gate_mean", "sqrt_u_mean", "sqrt_u_median")


def read_trace(run: Path) -> dict:
    rows = {key: [] for key in KEYS}
    meta = {"iteration": [], "radius": [], "fudge": [], "step": [], "disagreement_scale": []}
    with open(run / "iterations.jsonl") as stream:
        for line in stream:
            record = json.loads(line)
            if "gates" not in record:
                continue
            for key in meta:
                meta[key].append(record.get(key, np.nan))
            for key in KEYS:
                value = record.get(key)
                rows[key].append(np.asarray(value, np.float64) if value is not None else None)
    out = {key: np.asarray(value) for key, value in meta.items()}
    for key, values in rows.items():
        if values and all(v is not None for v in values):
            out[key] = np.stack(values)
    return out


def summarise(trace: dict, oracle: dict, windows=((1, 100), (101, 500), (501, 1500), (1501, 2500), (2501, 3000))):
    """Per window and channel: mean over iterations and over shells 1..radius of the gaps and ratios."""
    snr = np.stack([oracle["snr_mean"], oracle["snr_loading"]], axis=-1)
    target_noise = 4 * oracle["code_signal"] / np.maximum(snr, 1e-300)
    gates = trace["gates"]
    nshell = gates.shape[1]
    w = oracle["oracle"][:nshell]
    out = []
    for lo, hi in windows:
        pick = np.flatnonzero((trace["iteration"] >= lo) & (trace["iteration"] <= hi))
        if not len(pick):
            continue
        acc = {k: [] for k in ("gap", "gap_target", "signal_ratio", "noise_ratio", "gap_effective", "gap_masked",
                               "sqrt_u_median", "gap_estimated")}
        for i in pick:
            radius = int(trace["radius"][i])
            shells = np.arange(1, min(radius, nshell - 1) + 1)
            fudge = float(trace["fudge"][i])
            target = fudge * snr[shells] / 2 / (1 + fudge * snr[shells] / 2)
            acc["gap"].append((gates[i, shells] - w[shells]).mean(0))
            acc["gap_masked"].append((gates[i, shells] - oracle["oracle_masked"][shells]).mean(0))
            acc["gap_target"].append((gates[i, shells] - target).mean(0))
            acc["signal_ratio"].append(np.exp(np.log(trace["signal"][i, shells] / oracle["code_signal"][shells]).mean(0)))
            acc["noise_ratio"].append(
                np.exp(np.log(trace["disagreement"][i, shells] / target_noise[shells]).mean(0))
            )
            if "effective_gate_mean" in trace:
                acc["gap_effective"].append((trace["effective_gate_mean"][i, shells] - w[shells]).mean(0))
                acc["sqrt_u_median"].append(np.nanmean(trace["sqrt_u_median"][i, shells], axis=0))
            if "gates_estimated" in trace:  # oracle runs: what the estimator would have chosen on this trajectory
                acc["gap_estimated"].append((trace["gates_estimated"][i, shells] - w[shells]).mean(0))
        row = {"window": f"{lo}-{hi}"}
        for key, values in acc.items():
            if values:
                row[key] = np.mean(values, axis=0)
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    trace = read_trace(args.run)
    np.savez(args.output or args.run / "gate_trace.npz", **trace)
    if args.oracle:
        oracle = dict(np.load(args.oracle))
        for row in summarise(trace, oracle):
            print(
                row["window"],
                " ".join(f"{k} {v[0]:+.3f}/{v[1]:+.3f}" for k, v in row.items() if k != "window"),
            )


if __name__ == "__main__":
    main()
