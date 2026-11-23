"""Run the trials of one target, one setting and one particle count: draw, bin per fit order, fit, score.

A setting is the cap's ``c`` (``kappa_max = 1 / (c delta)^2``) and the lobe count of the fits (``truth``: the
target's own count, required for a known mixture).

Replicates run in parallel processes. Writes one JSON line per (replicate, order, method) and the masses of the
first replicate per order (for the figures).

Run (Snakemake calls it as rule run_trials):
    python scripts/fvml_prior_simulation/run_trials.py --config scripts/fvml_prior_simulation/config.yaml \
        --target targets/run_kent_mild.npz --c 1 --n-lobes 10 --n-particles 10000 --processes 5 \
        --rows results/run_kent_mild/c1_lobes10/n10000.jsonl --masses results/run_kent_mild/c1_lobes10/n10000_masses.npz
``--n-particles inf`` fits the target's own cell probabilities (no sampling noise): the model's floor.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.fvml_prior_simulation.simulation import load_target, run_trial, write_rows  # noqa: E402


def _trial(
    task: tuple[Path, float, str, int | None, int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """One replicate in a worker process."""
    target_path, c, n_lobes, n_particles, replicate, config = task
    fit = config["fit"]
    target = load_target(target_path)
    if n_lobes == "truth":
        n_components = target.n_components
    elif target.mixture is not None:
        raise ValueError(f"{target.name} is a known mixture: its fits use the true lobe count (--n-lobes truth)")
    else:
        n_components = int(n_lobes) + 1
    return run_trial(
        target,
        n_particles,
        replicate,
        [int(o) for o in config["fit_orders"]],
        n_components=n_components,
        n_starts=int(fit["n_starts"]),
        sigma_in_spacings=c,
        cell_levels=int(fit["cell_levels"]),
        max_iter=int(fit["max_iter"]),
        tol=float(fit["tol"]),
        reference_mass=float(fit["reference_mass"]),
    )


def main() -> None:
    """Parse arguments, run the replicates and write rows and masses."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--c", type=float, required=True, help="cap width in grid spacings")
    parser.add_argument("--n-lobes", required=True, help="an integer, or truth")
    parser.add_argument("--n-particles", required=True, help="an integer, or inf")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--masses", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    n_particles = None if args.n_particles == "inf" else int(args.n_particles)
    replicates = 1 if n_particles is None else int(config["replicates"])
    tasks = [(args.target, args.c, args.n_lobes, n_particles, r, config) for r in range(replicates)]
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        results = list(pool.map(_trial, tasks))
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    write_rows([row for rows, _ in results for row in rows], args.rows)
    np.savez(args.masses, **results[0][1])


if __name__ == "__main__":
    main()
