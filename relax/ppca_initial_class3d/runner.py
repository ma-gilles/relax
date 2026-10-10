"""Multi-class input/output lifecycle, called by ppca_initial_model for K>1."""

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

from relax.ppca_initial_class3d import checkpoint
from relax.ppca_initial_class3d.iteration_loop import run as run_iterations
from relax.ppca_initial_model.checkpoint import file_hash


def tilt_training_identity(ios):
    """Read native input identities before a resume can replace its flat STAR."""
    from relax.commands.ppca_initial_model import tilt_series_hash
    from relax.relion.tomo_input import read_optimisation_set

    particles_star, tomograms_star = read_optimisation_set(ios)
    return {
        "ios_sha256": file_hash(ios),
        "particles_sha256": file_hash(particles_star),
        "tomograms_sha256": file_hash(tomograms_star),
        "tilt_series_sha256": tilt_series_hash(tomograms_star),
    }


def run(args, config):
    # Shared loaders remain owned by the existing command. Import at dispatch
    # time so importing its parser does not import the multi-class controller.
    from relax.commands import ppca_initial_model as command

    if (args.ios is None) == (args.manifest is None):
        raise ValueError("Give either a single-particle manifest or --ios, not both")
    if args.ios and (args.particle_diameter is None or args.particle_diameter <= 0):
        raise ValueError("--ios needs a positive --particle-diameter")
    if not args.ios and args.particle_diameter is not None:
        raise ValueError("Single-particle diameter is specified in its training manifest")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("--stop-after must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Particle refinement runs require Slurm; use unit tests for local numerical checks")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
    from relax.helpers.compilation_cache import activate_recovar_compilation_cache

    activate_recovar_compilation_cache()

    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.ios:
        if args.resume:
            # Reject mismatched native input before replacing particles_2d.star.
            preflight = tilt_training_identity(args.ios)
            preflight.update(particle_diameter_ang=args.particle_diameter, zero_mask=bool(config.zero_mask))
            checkpoint.load(args.resume, config, preflight)
        # Both mask modes use the mixture's merged-stack adapter. K=1 keeps
        # the unchanged command loader and never enters this controller.
        from relax.ppca_initial_class3d.images import load_tilt_training

        data, identity = load_tilt_training(
            args.ios, output,
            zero_mask_diameter_ang=args.particle_diameter if config.zero_mask else None,
        )
        identity["zero_mask"] = bool(config.zero_mask)
        diameter, reading = args.particle_diameter, {"preread": True, "mode": "tomography",
                                                     "zero_mask": bool(config.zero_mask)}
    else:
        if config.zero_mask or config.auto_sampling:
            raise ValueError("--zero-mask and --auto-sampling are implemented for tilt particles (--ios)")
        data, manifest, identity = command.load_training(args.manifest, config.preread_images)
        diameter = manifest["particle_diameter_ang"]
        reading = command.training_image_reading(args.manifest, config.preread_images)[1]
    source = command.source_identity()
    repo = Path(__file__).resolve().parents[2]
    files = sorted(Path(__file__).parent.glob("*.py"))
    source["files"].update({str(path.relative_to(repo)): file_hash(path) for path in files})
    identity.update(source=source, particle_diameter_ang=diameter)
    if args.resume:
        checkpoint.load(args.resume, config, identity)
    (output / "run.json").write_text(json.dumps({
        "command": "ppca_initial_model", "config": dataclasses.asdict(config),
        "identity": identity, "image_reading": reading,
        "model": "class-specific means/loadings; shared class/pose/z across visible tilts",
        "status": "experimental; not scientifically or performance qualified",
    }, indent=2) + "\n")
    run_iterations(data, config, output, identity, diameter, resume=args.resume,
                   stop_after=args.stop_after, stop_file=args.stop_file)
