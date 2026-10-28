#!/usr/bin/env python
"""Write the direction prior that ``relax refine --fixed-direction-prior`` scores a benchmark run with.

One probability per HEALPix direction (NEST order) at ``--order``, positive and summing to one:
  * ``truth``: the dataset's own viewing-direction distribution (``true_prior.population_prior``);
  * ``uniform``: every direction equally likely;
  * ``kent``: an explicit Kent distribution of the viewing directions, in the simulator's parameterization
    (``simulate_dataset.py``), for a prior that disagrees with the data.

Run::

    python scripts/pose_prior_benchmark/write_fixed_prior.py --kind truth --poses-json DATASET/pose_distribution.json \\
        --order 7 --out PRIOR.npy
    python scripts/pose_prior_benchmark/write_fixed_prior.py --kind kent --kappa 8 --beta 2 --mu 1 1 0 \\
        --mu0 1 0 0 --order 7 --out PRIOR.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import healpy as hp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pose_prior_benchmark.true_prior import kent_population_prior, population_prior  # noqa: E402


def fixed_prior(args: argparse.Namespace) -> np.ndarray:
    """The prior ``args`` asks for, validated: finite, positive and summing to one."""
    if args.kind == "uniform":
        n = hp.nside2npix(1 << args.order)
        prior = np.full(n, 1.0 / n)
    elif args.kind == "truth":
        if args.poses_json is None:
            raise SystemExit("--kind truth needs --poses-json")
        prior = population_prior(json.loads(Path(args.poses_json).read_text()), args.order)
    else:
        missing = [name for name in ("kappa", "beta", "mu", "mu0") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--kind kent needs --{', --'.join(missing)}")
        prior = kent_population_prior(args.kappa, args.beta, args.mu, args.mu0, args.order)
    if not (np.all(np.isfinite(prior)) and np.all(prior > 0.0) and np.isclose(prior.sum(), 1.0, rtol=1e-9)):
        raise ValueError(f"the {args.kind} prior at order {args.order} is not a positive probability vector")
    return prior


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("truth", "uniform", "kent"), required=True)
    parser.add_argument(
        "--order", type=int, required=True, help="HEALPix order of the file (relax's --max_healpix_order)"
    )
    parser.add_argument("--out", required=True, help="the .npy to write")
    parser.add_argument("--poses-json", help="the dataset's pose_distribution.json (--kind truth)")
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--mu", type=float, nargs=3)
    parser.add_argument("--mu0", type=float, nargs=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Write the requested prior as a float64 ``.npy``."""
    args = parse_args(argv)
    prior = fixed_prior(args)
    np.save(args.out, prior)
    print(
        f"{args.kind} prior at order {args.order}: {prior.size} directions, min {prior.min():.3g}, max {prior.max():.3g}"
    )


if __name__ == "__main__":
    main()
