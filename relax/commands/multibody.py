"""``relax multibody``: RELION's multi-body refinement of a consensus auto-refine.

The command takes the options of the command RELION's GUI writes for a MultiBody job
(``getCommandsMultiBodyJob``, pipeline_jobs.cpp at f2c1a38)::

    relax multibody --continue Refine3D/job010/run_it018_optimiser.star --o MultiBody/job011/run \\
        --solvent_correct_fsc --multibody_masks bodies.star --oversampling 1 --healpix_order 4 \\
        --auto_local_healpix_order 4 --offset_range 3 --offset_step 1.5 --reconstruct_subtracted_bodies

Paths in the RELION files resolve from ``--relion_dir`` (default: the directory above the optimiser STAR's, the
directory relion_refine ran in), and the body STAR's mask paths from the body STAR's directory.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

EXECUTION_FLAGS = ("--dont_combine_weights_via_disc", "--preread_images")
EXECUTION_OPTIONS = ("--pool", "--j", "--gpu", "--scratch_dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relax multibody", description=__doc__.splitlines()[0])
    parser.add_argument("--continue", dest="continue_optimiser_star", required=True,
                        help="the consensus auto-refine's run_itNNN_optimiser.star")
    parser.add_argument("--o", dest="output", required=True, help="output root, e.g. MultiBody/job011/run")
    parser.add_argument("--multibody_masks", required=True, help="the body STAR file")
    parser.add_argument("--solvent_correct_fsc", action="store_true",
                        help="required: a multi-body refinement always corrects the FSC in each body's mask")
    parser.add_argument("--reconstruct_subtracted_bodies", action="store_true",
                        help="reconstruct each body from its subtracted images (the GUI default; required)")
    parser.add_argument("--oversampling", type=int, default=1)
    parser.add_argument("--healpix_order", type=int, default=4)
    parser.add_argument("--auto_local_healpix_order", type=int, default=4)
    parser.add_argument("--offset_range", type=float, default=3.0, help="pixels, as RELION's")
    parser.add_argument("--offset_step", type=float, default=1.5, help="pixels, as RELION's")
    parser.add_argument("--iter", dest="max_iter", type=int, default=999)
    parser.add_argument("--multibody_norm_overlap", action="store_true",
                        help="normalise overlapping body masks instead of RELION's extra overlap projectors")
    parser.add_argument("--sigma_ang", type=float, default=None,
                        help="degrees: the local-search width instead of the consensus run's, as relion_refine's")
    parser.add_argument("--relion_dir", default=None, help="the directory relion_refine ran in")
    parser.add_argument(
        "--continue_iteration", default=None,
        help="a multi-body iteration's run_itNNN_optimiser.star (relax's or RELION's) to continue from, as "
        "relion_refine --continue does; --continue still names the consensus, with the run's other options",
    )
    parser.add_argument(
        "--continue_as_uninterrupted", action="store_true",
        help="with --continue_iteration: resume the run the files came from (its noise, sampling steps and rand() "
        "stream) instead of relion_refine --continue's restart",
    )
    parser.add_argument("--pad", type=float, default=None,
                        help="RELION's padding factor, 1 or 2; the default is the consensus's")
    # RELION's execution options (I/O, threads, devices): they do not change the result, so they are accepted
    # and named in the log as having no effect. Any other flag is an error.
    for flag in EXECUTION_FLAGS:
        parser.add_argument(flag, action="store_true", help="RELION execution option; no effect in relax")
    for option in EXECUTION_OPTIONS:
        parser.add_argument(option, default=None, help="RELION execution option; no effect in relax")
    return parser


def run_options():
    """The refinement options relax's own commands read from the environment, with RELION's defaults."""

    import relax.reconstruction.volume_solver
    import relax.runtime.engine_tuning
    import relax.runtime.scoring_overrides
    from relax.numerics.precision import DensePrecisionPolicy
    from relax.refinement import refinement_options

    return refinement_options.RefinementOptions(
        # The multi-body controller plans its own sizes and sampling; the schedule only carries defaults.
        schedule=refinement_options.RefinementSchedule(init_current_size=0),
        environment=refinement_options.RefinementEnvironment.from_environ(
            engine=relax.runtime.engine_tuning.EngineTuning.from_environ(),
            scoring=relax.runtime.scoring_overrides.ScoringOverrides.from_environ(),
        ),
        variants=refinement_options.ScoringVariants.from_environ(
            reconstruction=relax.reconstruction.volume_solver.ReconstructionPrograms.from_environ(),
        ),
        precision=DensePrecisionPolicy.from_environ(),
        # RELION's particle order with the source-faithful power normalisation, as a fresh K=1 auto-refine.
        parity=replace(refinement_options.RelionParityOptions(), preserve_bpref_particle_order=True),
    )


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not args.solvent_correct_fsc:
        raise SystemExit("relax multibody needs --solvent_correct_fsc, as RELION's MultiBody job always passes")
    if not args.reconstruct_subtracted_bodies:
        raise SystemExit(
            "relax multibody reconstructs from the subtracted images only: pass --reconstruct_subtracted_bodies "
            "(the RELION GUI default; relion_refine's command-line default is off)"
        )
    output = Path(args.output)
    output_dir = output.parent if output.name == "run" else output
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "relax_multibody.log")],
    )
    from relax.multibody.refine import MultibodyCommand, refine, start_from_consensus

    ignored = [f for f in EXECUTION_FLAGS if getattr(args, f[2:])] + [
        f"{o} {getattr(args, o[2:])}" for o in EXECUTION_OPTIONS if getattr(args, o[2:]) is not None
    ]
    if ignored:
        logging.getLogger(__name__).info("RELION execution options with no effect in relax: %s", ", ".join(ignored))

    command = MultibodyCommand(
        body_star=Path(args.multibody_masks),
        healpix_order=args.healpix_order,
        auto_local_healpix_order=args.auto_local_healpix_order,
        offset_range_px=args.offset_range,
        offset_step_px=args.offset_step,
        oversampling=args.oversampling,
        reconstruct_subtracted_bodies=args.reconstruct_subtracted_bodies,
        max_iter=args.max_iter,
        sigma_ang_deg=args.sigma_ang,
        norm_overlap=args.multibody_norm_overlap,
        padding_factor=args.pad,
    )
    run, state = start_from_consensus(
        args.continue_optimiser_star, command, run_options(), output=output_dir, relion_cwd=args.relion_dir
    )
    if args.continue_iteration is not None:
        from relax.multibody.resume import continue_refinement

        continue_refinement(run, args.continue_iteration, as_uninterrupted=args.continue_as_uninterrupted)
    else:
        refine(run, state)


if __name__ == "__main__":
    main()
