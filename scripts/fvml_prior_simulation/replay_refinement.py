"""Replay the iterations of one half-set of a benchmark refinement with the FvML-mixture prior, warm and cold.

At every iteration of the run, the half-set's estimated viewing directions (``run_itNNN_data.star``, the
max-posterior pose of each particle) are binned on the fit grid: the HEALPix order of that iteration's learned
prior, at most ``--max-fit-order``. A fixed random ``--heldout-fraction`` of the half's particles is held out. Each
candidate lobe count is fitted (cell-mass fit, cap ``kappa_max_for_grid`` with ``--c``) on the rest, and the
candidate with the largest held-out log-likelihood is chosen. The chosen count is then refitted on all the half's
particles, starting from its candidate fit. Two start modes run side by side:

  * ``warm``: each candidate starts from its own fit at the previous iteration (all candidates are kept);
  * ``cold``: each candidate is the best of ``--n-starts`` k-means++ starts (only with ``--cold``).

Rows (JSON lines, ``--rows``): per iteration a ``baseline`` row (RELION's learned prior of the half and the
histogram of its directions); per iteration, mode and lobe count a ``candidate`` row; per iteration and mode a
``final`` row. Each row holds its Jensen-Shannon distance (bits) to the population the dataset was simulated from,
at the iteration's order, and the fits' iterations and seconds.

Run (Snakemake calls it as rule replay_half):
    python scripts/fvml_prior_simulation/replay_refinement.py \
        --refine-dir <benchmark>/sweep_mild_box256/mix_w100_n100k_nz0p5/learned/refine \
        --dataset-dir <benchmark>/sweep_mild_box256/mix_w100_n100k_nz0p5/dataset \
        --half 1 --heldout-fraction 0.2 --lobes 5 10 20 --c 0.25 --cold --rows replay/run_kent_mild/half1_f0.2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relax.sampling.orientation_priors import infer_direction_prior_healpix_order  # noqa: E402
from scripts.fvml_prior_simulation.simulation import (  # noqa: E402
    cell_children,
    heldout_log_likelihood,
    mixture_cell_probabilities,
    mixture_record,
    sphere_grid,
    write_rows,
)
from relax.sampling.fvml_mixture import (  # noqa: E402
    FitResult,
    FvmlMixture,
    fit_fvml_mixture_cells,
    init_fvml_mixture,
    kappa_max_for_grid,
)
from scripts.pose_prior_benchmark.evaluate_run import (  # noqa: E402
    direction_histogram,
    distribution_metrics,
    list_iterations,
    read_direction_prior,
    read_particles,
)
from scripts.pose_prior_benchmark.true_prior import population_prior  # noqa: E402


def population(record: dict[str, Any], order: int) -> np.ndarray:
    """Probability per HEALPix direction of the simulated viewing directions; a mixture record is summed by weight."""
    if record["poses"] == "mixture":
        return sum(c["weight"] * population(c["poses"], order) for c in record["components"])
    if record["poses"] == "kent" and record["dataset_params_option"][0] == 0.0:
        n = hp.nside2npix(1 << order)  # a Kent density with zero concentration is uniform
        return np.full(n, 1.0 / n)
    return population_prior(record, order)


def heldout_keys(keys: list[tuple[int, str]], fraction: float, seed: int) -> set[tuple[int, str]]:
    """A fixed random ``fraction`` of the particle keys (the same set at every iteration)."""
    ordered = sorted(keys)
    rng = np.random.default_rng(seed)
    count = int(round(fraction * len(ordered)))
    return {ordered[i] for i in rng.permutation(len(ordered))[:count]}


def cap_start(mixture: FvmlMixture, kappa_max: float) -> FvmlMixture:
    """The mixture with every concentration clipped to ``kappa_max`` (a warm start under a new cap)."""
    return FvmlMixture(mixture.weights, mixture.means, np.minimum(mixture.kappas, kappa_max), mixture.symmetry)


def timed_fit(cells: np.ndarray, masses: np.ndarray, start: FvmlMixture, **options: Any) -> tuple[FitResult, float]:
    """The cell-mass fit from ``start`` and its wall time in seconds."""
    began = time.perf_counter()
    fit = fit_fvml_mixture_cells(cells, masses, start, **options)
    return fit, time.perf_counter() - began


def replay(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the half's iterations; returns the rows described in the module docstring."""
    record = json.loads((args.dataset_dir / "pose_distribution.json").read_text())
    iterations = [
        i for i in list_iterations(args.refine_dir) if (args.refine_dir / f"run_it{i:03d}_data.star").is_file()
    ]
    seed = zlib.crc32(f"{args.name}/{args.half}/{args.heldout_fraction}".encode())
    start_rng = np.random.default_rng(seed + 1)
    options = {"max_iter": args.max_iter, "tol": args.tol}
    modes = ["warm", "cold"] if args.cold else ["warm"]
    previous: dict[int, FvmlMixture] = {}  # warm state: each lobe count's last candidate fit
    heldout_set: set[tuple[int, str]] | None = None
    rows: list[dict[str, Any]] = []
    for iteration in iterations:
        prior = read_direction_prior(args.refine_dir / f"run_it{iteration:03d}_half{args.half}_model.star")
        if prior is None:
            continue
        order = infer_direction_prior_healpix_order(prior)
        fit_order = min(order, args.max_fit_order)
        particles = read_particles(args.refine_dir / f"run_it{iteration:03d}_data.star")
        mine = particles["half"] == args.half
        keys = [k for k, m in zip(particles["image_key"], mine) if m]
        if heldout_set is None:
            heldout_set = heldout_keys(keys, args.heldout_fraction, seed)
        is_heldout = np.array([k in heldout_set for k in keys])
        rot, tilt = particles["rot"][mine], particles["tilt"][mine]
        train = direction_histogram(rot[~is_heldout], tilt[~is_heldout], fit_order)
        heldout = direction_histogram(rot[is_heldout], tilt[is_heldout], fit_order)
        full = train + heldout
        truth = population(record, order)
        kappa_max = kappa_max_for_grid(full.size, args.c)
        cells = cell_children(fit_order, args.cell_levels)
        directions = sphere_grid(fit_order)
        common = {
            "run": args.name,
            "half": args.half,
            "heldout_fraction": args.heldout_fraction,
            "c": args.c,
            "iteration": iteration,
            "order": order,
            "fit_order": fit_order,
            "kappa_max": kappa_max,
            "n_train": int(train.sum()),
            "n_heldout": int(heldout.sum()),
        }
        histogram = direction_histogram(rot, tilt, order)
        rows.append(
            {
                **common,
                "kind": "baseline",
                "js_bits_learned_prior": distribution_metrics(truth, prior)["js_bits"],
                "js_bits_histogram": distribution_metrics(truth, histogram / histogram.sum())["js_bits"],
                "js_bits_uniform": distribution_metrics(truth, np.full(truth.size, 1.0 / truth.size))["js_bits"],
            }
        )

        def score(fit: FitResult) -> dict[str, Any]:
            """Held-out score, distance to the population and the fit's state, for a row."""
            return {
                "heldout_log_likelihood": heldout_log_likelihood(
                    heldout, mixture_cell_probabilities(fit.mixture, fit_order)
                ),
                "js_bits": distribution_metrics(truth, mixture_cell_probabilities(fit.mixture, order))["js_bits"],
                "objective": float(fit.objective[-1]),
                "converged": bool(fit.converged),
                "lobes_at_cap": int(np.sum(fit.mixture.kappas >= kappa_max * (1.0 - 1e-9))),
                "fit": mixture_record(fit.mixture),
            }

        for mode in modes:
            candidates: dict[int, FitResult] = {}
            heldout_scores: dict[int, float] = {}
            for lobes in args.lobes:
                if mode == "warm" and lobes in previous:
                    fit, seconds = timed_fit(
                        cells, train, cap_start(previous[lobes], kappa_max), kappa_max=kappa_max, **options
                    )
                    n_iter = fit.n_iter
                else:
                    starts = [
                        init_fvml_mixture(directions, train, lobes + 1, kappa_max=kappa_max, rng=start_rng)
                        for _ in range(args.n_starts)
                    ]
                    fits = [timed_fit(cells, train, s, kappa_max=kappa_max, **options) for s in starts]
                    fit = max((f for f, _ in fits), key=lambda f: f.objective[-1])
                    seconds = sum(s for _, s in fits)
                    n_iter = sum(f.n_iter for f, _ in fits)
                if mode == "warm":
                    previous[lobes] = fit.mixture
                candidates[lobes] = fit
                row = {
                    **common,
                    "kind": "candidate",
                    "mode": mode,
                    "lobes": lobes,
                    "n_iter": n_iter,
                    "seconds": seconds,
                }
                rows.append({**row, **score(fit)})
                heldout_scores[lobes] = rows[-1]["heldout_log_likelihood"]
            chosen = max(heldout_scores, key=heldout_scores.get)
            fit, seconds = timed_fit(cells, full, candidates[chosen].mixture, kappa_max=kappa_max, **options)
            final = score(fit)
            final.pop("heldout_log_likelihood")  # the held-out particles are in this fit
            rows.append(
                {
                    **common,
                    "kind": "final",
                    "mode": mode,
                    "lobes": chosen,
                    "n_iter": fit.n_iter,
                    "seconds": seconds,
                    **final,
                }
            )
        print(f"{args.name} half {args.half} f {args.heldout_fraction} iteration {iteration} order {order}", flush=True)
    return rows


def main() -> None:
    """Parse arguments, replay the half-set and write the rows."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refine-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="holds pose_distribution.json")
    parser.add_argument("--name", required=True, help="run name in the rows")
    parser.add_argument("--half", type=int, choices=(1, 2), required=True)
    parser.add_argument("--heldout-fraction", type=float, required=True)
    parser.add_argument("--lobes", type=int, nargs="+", required=True)
    parser.add_argument("--c", type=float, default=0.25, help="cap width in grid spacings")
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--max-fit-order", type=int, default=5)
    parser.add_argument("--cell-levels", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--cold", action="store_true", help="also run the cold-start mode")
    parser.add_argument("--rows", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.heldout_fraction < 1.0:
        raise ValueError("--heldout-fraction must be in (0, 1)")
    rows = replay(args)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    write_rows(rows, args.rows)


if __name__ == "__main__":
    main()
