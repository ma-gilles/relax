#!/usr/bin/env python
"""Simulate one pose-prior benchmark dataset with RECOVAR's simulator.

A K=1 PDB-derived fixture built by ``scripts/prepare_pdb_k1_relion_sanity_benchmark.py`` whose poses are uniform
or follow RECOVAR's Kent option: the third column of each rotation matrix is Kent-distributed (mean direction
``mu0``, major axis ``mu``) and the in-plane angle is uniform. RELION's viewing direction is the third row, so it
is always uniform in ``rot``: ``mu0 = z`` gives a polar cap of preferred views, ``mu0`` perpendicular to z an
equatorial belt. The Kent sampler is RECOVAR's rejection sampler; when it cannot fill the request it silently
draws the rest from a tangent-normal approximation, so this script stops instead of keeping such a dataset.

Run (GPU node)::

    python scripts/pose_prior_benchmark/simulate_dataset.py --output-dir DIR --n-images 50000 \\
        --grid-size 128 --poses kent --kappa 8 --beta 2 --mu 1 1 0 --mu0 0 0 1

The output directory holds ``particles.star`` (with the true Euler angles), ``particles.<box>.mrcs``,
``reference_gt_relion.mrc``, ``reference_init_relion.mrc`` and ``pose_distribution.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.prepare_pdb_k1_relion_sanity_benchmark import prepare_benchmark  # noqa: E402

# The canonical fixture script's own command-line defaults for everything this benchmark does not vary.
FIXTURE_DEFAULTS = dict(
    pdb_path=None,
    pdb_bfactor=0.0,
    noise_model="white",
    noise_scale_std=0.0,
    contrast_std=0.0,
    volume_radius=0.7,
    percent_outliers=0.0,
    put_extra_particles=False,
    image_offset_n_std=0.0,
    relion_bg_radius_px=None,
    noise_rng_batch_size=None,
    relion_normalize=True,
    streaming_mmap=True,
    streaming_chunk_size=1000,
    disc_type="cubic",
)
KENT_FALLBACK_MESSAGE = "Kent rejection fallback"


class KentFallbackGuard(logging.Handler):
    """Raise when RECOVAR's Kent sampler falls back from rejection sampling to its approximate tail fill."""

    def emit(self, record: logging.LogRecord) -> None:
        """Fail on the sampler's fallback warning."""
        if record.getMessage().startswith(KENT_FALLBACK_MESSAGE):
            raise RuntimeError(
                f"{record.getMessage()}. The Kent parameters accept too few rejection-sampling draws, so the "
                "viewing directions would not follow the requested Kent distribution; lower beta or kappa."
            )


def pose_option(args: argparse.Namespace) -> str | list:
    """Return the simulator's ``dataset_params_option`` for the requested pose distribution."""
    if args.poses == "uniform":
        return "uniform"
    return [float(args.kappa), float(args.beta), [float(v) for v in args.mu], [float(v) for v in args.mu0]]


def simulate(args: argparse.Namespace) -> None:
    """Generate the dataset and write ``pose_distribution.json`` beside it."""
    logging.getLogger("recovar.simulation.simulator").addHandler(KentFallbackGuard())
    option = pose_option(args)
    prepare_benchmark(
        args.output_dir,
        n_images=args.n_images,
        grid_size=args.grid_size,
        voxel_size=None,
        noise_level=args.noise_level,
        dataset_params_option=option,
        init_resolution_ang=args.init_resolution_ang,
        seed=args.seed,
        atomic_volume_kwargs={"atomic_solvent_correction": True, "atomic_bfactor": args.total_bfactor},
        **FIXTURE_DEFAULTS,
    )
    record = {
        "poses": args.poses,
        "dataset_params_option": option,
        "n_images": args.n_images,
        "pose_seed": 0,  # RECOVAR calls its pose generators without a seed, so they use their default 0
        "sphstat_installed": importlib.util.find_spec("sphstat") is not None,
    }
    (args.output_dir / "pose_distribution.json").write_text(json.dumps(record, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--total-bfactor", type=float, default=80.0, help="total B-factor, A^2")
    parser.add_argument("--init-resolution-ang", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=17, help="noise seed")
    parser.add_argument("--poses", choices=("kent", "uniform"), required=True)
    parser.add_argument("--kappa", type=float, help="Kent concentration (--poses kent)")
    parser.add_argument("--beta", type=float, help="Kent ovalness (--poses kent)")
    parser.add_argument("--mu", type=float, nargs=3, help="Kent major-axis vector (--poses kent)")
    parser.add_argument("--mu0", type=float, nargs=3, help="Kent mean direction (--poses kent)")
    args = parser.parse_args(argv)
    kent = {"--kappa": args.kappa, "--beta": args.beta, "--mu": args.mu, "--mu0": args.mu0}
    missing = [flag for flag, value in kent.items() if value is None]
    if args.poses == "kent" and missing:
        parser.error(f"--poses kent needs {', '.join(missing)}")
    if args.poses == "uniform" and len(missing) < len(kent):
        parser.error("--kappa, --beta, --mu and --mu0 apply to --poses kent only")
    return args


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    simulate(parse_args(argv))


if __name__ == "__main__":
    main()
