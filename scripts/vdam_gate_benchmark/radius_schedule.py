"""The Fourier radius schedule of a traced run, as relax initial_model's --fourier-radius-schedule argument.

Reads a run's trace/gate_trace.jsonl (traced_initial_model.py: one line per M-step with its iteration and radius) and
writes "R1xC1,R2xC2,..." for iterations 1..NR_ITER, so that a run given that argument works at the traced run's radius in
every iteration (its E-step's current size is twice the radius, its M-step's radius the radius). Fails if an iteration
is missing or repeated.

Run:
    python scripts/vdam_gate_benchmark/radius_schedule.py --trace RUN/trace/gate_trace.jsonl --nr-iter 200 \
        --output OUT/radius_schedule.txt
"""

from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path


def traced_radii(trace: Path, nr_iter: int) -> list[int]:
    """The radius of iterations 1..nr_iter of a traced run."""
    radii: dict[int, int] = {}
    for line in trace.read_text().splitlines():
        record = json.loads(line)
        if record["iteration"] in radii:
            raise ValueError(f"{trace}: iteration {record['iteration']} traced twice")
        radii[record["iteration"]] = int(record["radius"])
    missing = [it for it in range(1, nr_iter + 1) if it not in radii]
    if missing:
        raise ValueError(f"{trace}: no M-step traced at iterations {missing[:10]}")
    return [radii[it] for it in range(1, nr_iter + 1)]


def schedule_argument(radii: list[int]) -> str:
    """Run-length encoding of the radii in the form of --fourier-radius-schedule."""
    return ",".join(f"{radius}x{len(list(run))}" for radius, run in groupby(radii))


def main() -> None:
    """Parse the command line and write the schedule."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--nr-iter", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(schedule_argument(traced_radii(args.trace, args.nr_iter)) + "\n")


if __name__ == "__main__":
    main()
