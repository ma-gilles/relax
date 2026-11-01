#!/usr/bin/env python
"""Replay one RELION multi-body iteration in relax from RELION's own state (a fixed-state diagnostic).

    python -m scripts.replay_multibody_iteration --consensus <consensus run_itNNN_optimiser.star> \\
        --relion <RELION MultiBody dir> --bodies bodies.star --iteration 2 --out <dir>

relax reads RELION's state after ``iteration - 1`` and RELION's decisions for ``iteration``
(:mod:`relax.multibody.replay`) and writes its iteration in ``--out``, for ``scripts.compare_multibody_relion``.
The other options are RELION's MultiBody command's (``relax multibody``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--relion", required=True, type=Path)
    parser.add_argument("--bodies", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--healpix_order", type=int, default=4)
    parser.add_argument("--auto_local_healpix_order", type=int, default=4)
    parser.add_argument("--offset_range", type=float, default=3.0)
    parser.add_argument("--offset_step", type=float, default=1.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from relax.commands.multibody import run_options
    from relax.multibody.refine import MultibodyCommand, start_from_consensus
    from relax.multibody.replay import replay_iteration

    command = MultibodyCommand(
        body_star=args.bodies, healpix_order=args.healpix_order, auto_local_healpix_order=args.auto_local_healpix_order,
        offset_range_px=args.offset_range, offset_step_px=args.offset_step, oversampling=1,
        reconstruct_subtracted_bodies=True, max_iter=args.iteration,
    )
    run, _ = start_from_consensus(args.consensus, command, run_options(), output=args.out)
    replay_iteration(run, args.relion, args.iteration)


if __name__ == "__main__":
    main()
