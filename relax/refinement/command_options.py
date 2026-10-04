"""Refinement command options, job defaults and supported input-mode admission.

The execution controller owns cache/import setup and when data-dependent checks
run. This module owns CLI configuration; numerical refinement options remain in
``refinement_options.py``.
"""

import argparse
import math
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from relax.diagnostics import frozen_boundary_cli
from relax.diagnostics.state_swap_probe import add_state_swap_probe_arguments
from relax.helpers.particle_io import add_particle_read_arguments
from relax.refinement.refinement_options import RelionConsistencyOptions
from relax.relion import input_poses

if TYPE_CHECKING:
    from relax.relion.relion_worker_scale import RelionDispatchSchedule

## ALL THIS SHOULD OF CONSTANTS SHOULD BE MOVED AND RECORDED SOMEWHERE - NOT PEPPERED THROUGHOUT CODE- OR AT THE TOP?
# pipeline_jobs.cpp:4191 (Refine3D), 3697 (Class3D): "Mask diameter (A)" 200.
RELION_GUI_PARTICLE_DIAMETER_ANG = 200.0


_INITIAL_PROJECTOR_USE_REAL_REFERENCE_ENV = "RELAX_INITIAL_PROJECTOR_USE_REAL_REFERENCE"
_FIRSTITER_CC_TREE_TOP2_RESCORE_DEFAULT_MAX_MARGIN = 4e-6


def resolve_firstiter_controls(
    *,
    firstiter_cc: bool,
    n_classes: int,
    tree_rescore_max_margin: str = "auto",
    environ: MutableMapping[str, str] | None = None,
) -> tuple[bool, float | None]:
    """Resolve the narrow K=1 firstiter-CC parity defaults.

    Returns whether the initial projector takes the real reference and the coarse-tree
    top-2 rescore margin (``RelionParityOptions.firstiter_cc_tree_rescore_max_margin``).
    RELION Class3D also uses first-iteration CC, but the exact coarse-tree
    replay currently supports only K=1.  Keep K>1 and non-firstiter callers
    unchanged. ``tree_rescore_max_margin`` (``--firstiter_cc_tree_rescore_max_margin``)
    is ``auto`` (4e-6 for K=1 with --firstiter_cc, else off), ``off`` or a margin.
    """

    environment = os.environ if environ is None else environ
    use_k1_defaults = bool(firstiter_cc) and int(n_classes) == 1
    initial_projector_default = "1" if use_k1_defaults else "0"
    initial_projector_token = environment.get(
        _INITIAL_PROJECTOR_USE_REAL_REFERENCE_ENV,
        initial_projector_default,
    ).strip().lower()
    if initial_projector_token in {"1", "true", "yes", "on"}:
        use_initial_projector_real = True
    elif initial_projector_token in {"0", "false", "no", "off", ""}:
        use_initial_projector_real = False
    else:
        raise SystemExit(
            f"{_INITIAL_PROJECTOR_USE_REAL_REFERENCE_ENV} must be a boolean token, "
            f"got {initial_projector_token!r}"
        )

    token = str(tree_rescore_max_margin).strip().lower()
    if token == "auto":
        margin = _FIRSTITER_CC_TREE_TOP2_RESCORE_DEFAULT_MAX_MARGIN if use_k1_defaults else None
    elif token == "off":
        margin = None
    else:
        try:
            margin = float(token)
        except ValueError:
            margin = float("nan")
        if not (math.isfinite(margin) and margin >= 0.0):
            raise SystemExit(
                "--firstiter_cc_tree_rescore_max_margin must be auto, off or a finite non-negative float, "
                f"got {tree_rescore_max_margin!r}"
            )
    return use_initial_projector_real, margin



class InitialSampling(NamedTuple):
    """Startup pass orders and the resolved refinement limit with its source."""

    coarse_order: int
    fine_order: int
    max_order: int | None
    max_order_source: str


def resolve_initial_sampling(
    healpix_order: int,
    adaptive_oversampling: int,
    *,
    n_classes: int,
    max_healpix_order: int | None,
) -> InitialSampling:
    """Resolve CLI pass orders and Class3D's fixed or auto-refine's uncapped limit."""
    if healpix_order < 0:
        raise ValueError(f"healpix_order must be non-negative, got {healpix_order}")
    if adaptive_oversampling < 0:
        raise ValueError(f"adaptive_oversampling must be non-negative, got {adaptive_oversampling}")
    if max_healpix_order is None:
        if n_classes > 1:
            max_order = healpix_order
            source = "RELION Class3D fixed --healpix_order"
        else:
            max_order = None
            source = "K=1 auto-refine, uncapped as RELION"
    else:
        if max_healpix_order < healpix_order:
            raise ValueError(
                "max_healpix_order must be >= healpix_order "
                f"({max_healpix_order} < {healpix_order})",
            )
        max_order = max_healpix_order
        source = "explicit CLI"
    return InitialSampling(
        coarse_order=healpix_order,
        fine_order=healpix_order + adaptive_oversampling,
        max_order=max_order,
        max_order_source=source,
    )


def resolve_job_defaults(args) -> None:
    """Resolve the defaults that depend on the job type (Refine3D K=1 or Class3D K>1)."""
    k1 = int(args.n_classes) == 1
    if args.max_iter is None:
        # Auto-refine runs to convergence with nr_iter = --auto_iter_max (999);
        # the Class3D GUI runs 25 iterations.
        args.max_iter = 999 if k1 else 25
    if args.image_fourier_backend == "auto":
        # RELION's CUDA image preprocessing is the one the resident pass 2's exact
        # BPref operands read, for K=1 and Class3D alike.
        args.image_fourier_backend = "relion_cuda"
    if args.apply_initial_lowpass is None:
        args.apply_initial_lowpass = args.frozen_boundary_dir is None


def resolve_standalone_k1_start(args) -> None:
    """Default a fresh K=1 start with no RELION output to RELION's particle table from the input.

    Standalone means relion_refine's own inputs: the particle table rebuilt from
    ``<data_dir>/particles.star`` and the seed (half sets, groups, order). A run
    given ``--relion_half_sets``, ``--relion_init_dir``, a replay directory or a
    frozen boundary is a RELION-seeded debug start and supplies its own half sets.
    """
    fresh_k1 = (
        int(args.n_classes) == 1
        and int(args.init_relion_iteration) == 0
        and args.frozen_boundary_dir is None
        and args.perturb_replay_relion_dir is None
    )
    if args.relion_half_sets_from_input is None:
        args.relion_half_sets_from_input = bool(
            fresh_k1 and args.relion_half_sets is None and args.relion_init_dir is None
        )
    validate_input_half_sets(args)


def validate_input_half_sets(args) -> None:
    """Reject an input-derived RELION particle table where RELION would not build one this way."""
    if not getattr(args, "relion_half_sets_from_input", False):
        return
    problems = []
    if args.relion_half_sets is not None:
        problems.append("it replaces --relion_half_sets")
    if int(args.n_classes) != 1:
        problems.append("it is K=1 auto-refine only")
    if args.frozen_boundary_dir is not None:
        problems.append("a frozen boundary seals its own half sets")
    if problems:
        raise SystemExit("--relion-half-sets-from-input: " + "; ".join(problems))


def validate_multi_shape_args(args, frozen_boundary, double_image_preprocessing):
    """Optics groups on several image shapes run a fresh refinement or classification only.

    Their start-up noise is RELION's estimate from the images on the reference grid;
    replay, frozen boundaries, loaded noise and float64 image preprocessing assume
    one image grid and stay refused.
    """
    reasons = []
    if args.init_noise_from_npz is not None:
        reasons.append("loaded noise is single-shape")
    if frozen_boundary is not None or args.perturb_replay_relion_dir is not None or args.relion_init_dir is not None:
        reasons.append("replayed or frozen RELION state is single-shape")
    if double_image_preprocessing:
        reasons.append("float64 image preprocessing is single-shape")
    if args.relion_softmask_reduction != "control":
        reasons.append("soft-mask reduction probes are single-shape")
    if reasons:
        raise SystemExit("optics groups on several image shapes: " + "; ".join(reasons))


def validate_continue_args(args) -> int:
    """Refuse options a continuation cannot honour; return the run's random seed.

    The run files pin the trajectory, so a RELION-seeded or replayed start, a frozen
    boundary or a sampling oracle would contradict them. The seed must be the original
    run's: it fixes the particle order and the sampling perturbations.
    """

    conflicts = [
        flag
        for flag, value in (
            ("--perturb_replay_relion_dir", args.perturb_replay_relion_dir),
            ("--relion_init_dir", args.relion_init_dir),
            ("--frozen-boundary-dir", args.frozen_boundary_dir),
            ("--init_noise_from_npz", args.init_noise_from_npz),
            ("--init_previous_best_poses_npz", args.init_previous_best_poses_npz),
            ("--relion_current_sizes", args.relion_current_sizes),
            ("--relion_healpix_orders", args.relion_healpix_orders),
            ("--final-replay-relion-dir", args.final_replay_relion_dir),
            ("--relion-projector-capture-dir", args.relion_projector_capture_dir),
            ("--state-swap-variant", args.state_swap_variant),
        )
        if value is not None
    ]
    if int(args.init_relion_iteration) != 0:
        conflicts.append("--init_relion_iteration")
    if args.diagnostic_single_half:
        conflicts.append("--diagnostic_single_half")
    if conflicts:
        raise SystemExit("--continue cannot be combined with " + ", ".join(conflicts))
    from relax.refinement.run_files import read_star_blocks

    general = read_star_blocks(args.continue_optimiser_star).get("optimiser_general", {})
    if "rlnRandomSeed" not in general:
        raise SystemExit(f"{args.continue_optimiser_star} has no rlnRandomSeed")
    seed = int(general["rlnRandomSeed"])
    if args.seed is not None and int(args.seed) != seed:
        raise SystemExit(f"--seed {args.seed} differs from the continued run's seed {seed}")
    return seed


def validate_tomo_args(args, frozen_boundary, double_image_preprocessing):
    """Refuse options the subtomogram (2D-stack) path does not implement yet (S4.2)."""

    if frozen_boundary is not None or args.relion_init_dir is not None or args.init_noise_from_npz is not None:
        raise SystemExit("subtomogram particles start fresh from RELION's inputs (no frozen, replayed or loaded state)")
    if double_image_preprocessing:
        raise SystemExit("subtomogram particles have no float64 scoring diagnostic")
    if args.relion_softmask_reduction != "control":
        raise SystemExit("subtomogram particles have no soft-mask reduction probe")


def parse_refinement_args(argv=None):
    from relax.symmetry import canonicalize_rotational_symmetry

    parser = argparse.ArgumentParser(
        description="RELION-equivalent 3D auto-refine (relax refine, K=1) and 3D classification "
        "(relax class3d, K>1). See docs/user_guide.md."
    )
    input_poses._add_initial_pose_source_argument(parser)
    parser.add_argument(
        "--sym", default="C1", type=canonicalize_rotational_symmetry,
        help="RELION proper rotational point group: Cn, Dn, T, O, or I/I1/I2/I3/I4.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing particles.star, reference_init_relion.mrc, etc.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--skip-large-outputs",
        action="store_true",
        help=(
            "Skip refinement_results.npz and final MRC writes. Intended only for "
            "short timing probes where run logs and benchmark ledgers are sufficient."
        ),
    )
    add_particle_read_arguments(parser)
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Maximum numbered iterations. Default: RELION's --auto_iter_max 999 for K=1 "
        "auto-refine (it stops at convergence; ml_optimiser.cpp:1255, 2543) and the GUI's "
        "25 for Class3D (pipeline_jobs.cpp:3689).",
    )
    parser.add_argument(
        "--healpix_order",
        type=int,
        default=2,
        help="RELION coarse pass-1 HEALPix order. With adaptive oversampling, "
        "pass 2 evaluates healpix_order + adaptive_oversampling. Default: the GUI's "
        "7.5 degree sampling with oversampling 1 (pipeline_jobs.cpp:4205, 4489).",
    )
    parser.add_argument(
        "--max_healpix_order",
        type=int,
        default=None,
        help=(
            "Optional cap on the coarse HEALPix order of RELION-style sampling "
            "updates. If omitted, K>1 Class3D stays fixed at --healpix_order "
            "to match RELION's _rlnDoAutoSampling=0 command path, and K=1 "
            "auto-refine is uncapped as RELION's is. Set explicitly to cap K=1 "
            "or to allow Class3D coarse-grid refinement."
        ),
    )
    parser.add_argument(
        "--auto_local_healpix_order",
        type=int,
        default=4,
        help="RELION --auto_local_healpix_order threshold for switching from "
        "global to local angular searches. RELION's binary default is 4; "
        "set to 3 when comparing against runs launched with "
        "--auto_local_healpix_order 3.",
    )
    parser.add_argument(
        "--offset_range", type=float, default=5.0,
        help="Translation search range (pixels); GUI default 5 (pipeline_jobs.cpp:4209)",
    )
    parser.add_argument(
        "--offset_step", type=float, default=2.0,
        help="Translation step (pixels) before oversampling; the GUI's 1 pixel times 2^oversampling "
        "(pipeline_jobs.cpp:4213, 4504)",
    )
    parser.add_argument(
        "--offset_sigma_angstrom",
        type=float,
        default=10.0,
        help="RELION-style Gaussian translation-prior sigma in Angstrom.",
    )
    parser.add_argument("--adaptive_oversampling", type=int, default=1, help="Oversampling levels (0=off, 1=2x)")
    parser.add_argument(
        "--coarse-engine", "--coarse_engine",
        choices=("auto", "gemm_hybrid", "gemm_dense"), default="auto",
        help=("Global E/M strategy (default: auto). gemm_hybrid retains fine-support pruning; "
              "gemm_dense is an experimental full-grid, no-pruning route with unqualified quality and speed."),
    )
    parser.add_argument(
        "--max_significants",
        type=int,
        default=None,
        help=(
            "Active maximum significant samples per image. Use <=0 for an "
            "uncapped diagnostic. If omitted, resolve RELION's runtime value "
            "from the optimiser STAR; in 3-D gradient mode a saved -1 means "
            "100 times the class count."
        ),
    )
    parser.add_argument(
        "--tau2_fudge",
        type=float,
        default=None,
        help="RELION tau2_fudge regularization strength. If omitted, use "
        "RELION's mode default: 1.0 for K=1 auto-refine and 4.0 for K>1 "
        "Class3D. If --relion_init_dir has run_it000_optimiser.star, its "
        "rlnTau2FudgeFactor/rlnTau2FudgeArg value takes precedence. Higher "
        "values produce smoother volumes (stronger prior).",
    )
    parser.add_argument(
        "--gridding_kernel",
        choices=("radial", "separable"),
        default="radial",
        help="Real-space gridding-correction window of the scoring projector and the "
        "reconstructions. radial (default) is RELION's sinc^2(|x| / (pad * box)); separable is the "
        "per-axis sinc^2 product, the exact transform of the trilinear kernel. separable "
        "is implemented for K=1 single-particle auto-refine only and is refused elsewhere.",
    )
    parser.add_argument(
        "--shell_pair_counting",
        choices=("relion", "once"),
        default="relion",
        help="How the 3-D shell statistics behind tau2, data-vs-prior, the half-map FSC and the "
        "current-size scheduling count Hermitian pairs. relion (default) visits every stored entry of "
        "the x-half volume, which counts the pairs of the kx = 0 plane twice (updateSSNRarrays, "
        "calculateDownSampledFourierShellCorrelation, computeFourierTransformMap, getSpectrum); once "
        "counts every pair once. The 1/1000 weight floor inside the reconstruction keeps RELION's counting.",
    )
    parser.add_argument(
        "--perturb_factor",
        type=float,
        default=0.5,
        help="RELION SamplingPerturbation factor (default 0.5 matching "
        "RELION GUI `--perturb 0.5`). Applies a per-iter random rigid "
        "rotation of the SO(3) trial grid and translation shift, ported "
        "from healpix_sampling.cpp:167-174 / 1909-1934 / 1810-1820. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--perturb_seed",
        type=int,
        default=None,
        help="Optional deterministic seed for the SamplingPerturbation RNG. "
        "If unset, defaults to --seed to match RELION's --random_seed. "
        "Use a negative value for the legacy non-reproducible NumPy path.",
    )
    parser.add_argument(
        "--perturb_replay_relion_dir",
        default=None,
        help="Controlled RELION trajectory replay: read SamplingPerturbInstance "
        "and per-iteration particle/model overrides from run_it{NNN}_* files in "
        "this directory. This is not an autonomous trajectory; omit it and use "
        "--perturb_seed for same-seed autonomous refinement.",
    )
    parser.add_argument(
        "--replay-noise-semantics",
        choices=("continuation", "uninterrupted"),
        default="continuation",
        help=(
            "Diagnostic replay after --init_relion_iteration > 0: 'continuation' "
            "reproduces a RELION --continue restart, which scores both halves with "
            "the half-1 sigma2_noise at its first expectation; 'uninterrupted' keeps "
            "each half's own model-STAR spectrum, as an uninterrupted RELION run does."
        ),
    )
    parser.add_argument(
        "--final-replay-relion-dir",
        default=None,
        help=(
            "Diagnostic-only final-boundary substitution source. Numbered refinement remains "
            "autonomous; selected last-numbered state fields and/or the unnumbered final sampling "
            "state are loaded only after convergence."
        ),
    )
    add_state_swap_probe_arguments(parser)
    parser.add_argument(
        "--final-replay-fields",
        default="all",
        help=(
            "Comma-separated diagnostic final-only groups: poses, sampling, corrections, "
            "noise, direction_prior, norm_factor, scoring_scale, references, or all. "
            "poses replaces previous rotations/translations; sampling replaces translation sigma "
            "and final sampling grid/perturbation; corrections replaces noise, direction prior, "
            "and the norm factor while retaining the resident scoring scale; norm_factor and "
            "scoring_scale form an orthogonal diagnostic pair; references replaces only the two "
            "input half-reference maps used by the final all-data expectation."
        ),
    )
    parser.add_argument(
        "--final-replay-source-iteration",
        type=int,
        default=None,
        help=(
            "Exact last-numbered RELION iteration used by --final-replay-relion-dir. "
            "Defaults to the latest contiguous complete numbered state and fails closed "
            "if it exceeds --max_iter or lacks final convergence provenance."
        ),
    )
    parser.add_argument(
        "--perturb-replay-restart-state-iterations",
        default="",
        help=(
            "Comma-separated numbered RELION sampling-state iterations where a "
            "provenance-qualified continuation restarted its sampling object. "
            "The next expectation reconstructs the unrounded perturbation from "
            "RELION's seed-1 restart state and random_seed+iteration; STAR values "
            "remain strict consistency guards. Example: 11 for a rescue whose "
            "first continued expectation is numbered iteration 12."
        ),
    )
    parser.add_argument(
        "--relion-projector-capture-dir",
        default=None,
        help=(
            "Directory containing a validated live RELION Projector::data capture. "
            "Requires --perturb_replay_relion_dir and "
            "--relion-projector-capture-iteration."
        ),
    )
    parser.add_argument(
        "--relion-projector-capture-manifest",
        default=None,
        help=(
            "Validated SHA-256 manifest for --relion-projector-capture-dir. "
            "Defaults to iterN_VALIDATED_SHA256SUMS inside that directory."
        ),
    )
    parser.add_argument(
        "--relion-projector-capture-iteration",
        type=int,
        default=None,
        help="Numbered RELION expectation iteration represented by the live capture.",
    )
    parser.add_argument(
        "--perturb-replay-restart-provenance",
        default=None,
        help=(
            "Required provenance file for --perturb-replay-restart-state-iterations. "
            "Its resolved path and SHA256 are recorded in refinement_results.npz and "
            "the benchmark ledger."
        ),
    )
    parser.add_argument(
        "--relion-scale-followers",
        type=int,
        default=None,
        help=(
            "Debug-only emulation of relion_refine_mpi's follower-local group-scale state "
            "for a replay against an MPI RELION run. Requires --relion-dispatch-schedule "
            "because RELION assigns --pool chunks dynamically. A standalone run (no "
            "--relion_init_dir/--perturb_replay_relion_dir) is single-process, like a "
            "non-MPI relion_refine, and needs neither; 0 selects that path for a replay."
        ),
    )
    parser.add_argument(
        "--relion-dispatch-schedule",
        default=None,
        help=(
            "NPZ containing per-iteration dynamic MPI follower ownership captured from "
            "the same RELION oracle run. Its state-file and ordered-particle hashes are "
            "verified against the active replay/init directory. Required by default for "
            "strict K>1 replay/init."
        ),
    )
    parser.add_argument(
        "--relion-follower-scale-replay",
        default=None,
        help=(
            "Diagnostic NPZ containing complete follower-scale matrices at selected "
            "numbered RELION iterations. Requires strict K>1 follower topology and "
            "a matching captured dispatch schedule. The first numbered iteration is "
            "rejected because its resident image-normalization state does not yet exist."
        ),
    )
    parser.add_argument(
        "--init_relion_iteration",
        type=int,
        default=0,
        help=(
            "Diagnostic replay offset: treat the first RECOVAR loop iteration "
            "as continuing after this RELION iteration. This is mainly for "
            "profile-only jumps into later local-search iterations."
        ),
    )
    frozen_boundary_cli.add_frozen_boundary_arguments(parser)
    parser.add_argument(
        "--replay_relion_normcorr",
        dest="replay_relion_normcorr",
        action="store_true",
        default=None,
        help="If set together with --perturb_replay_relion_dir, also inject "
        "RELION's per-iter rlnNormCorrection / rlnGroupScaleCorrection into "
        "recovar's E-step at iter 2+. This is the default when "
        "--perturb_replay_relion_dir is set.",
    )
    parser.add_argument(
        "--no-replay_relion_normcorr",
        dest="replay_relion_normcorr",
        action="store_false",
        help="Disable RELION normCorrection / group-scale replay while still "
        "using other per-iteration replay overrides.",
    )
    parser.add_argument(
        "--init_resolution", type=float, default=60.0,
        help="RELION --ini_high in Angstrom: the initial low-pass and the iteration-1 current size; "
        "GUI default 60 (pipeline_jobs.cpp:4172)",
    )
    parser.add_argument(
        "--image-fourier-backend",
        choices=("auto", "host_numpy", "jax_gpu", "relion_cuda"),
        default="auto",
        help=(
            "Fourier preprocessing backend for RELION-masked particle images. auto (default) "
            "is relion_cuda, the source-faithful CUDA normalization, translation and mask path "
            "that the fresh K=1 defaults and the resident pass 2 require, and host_numpy for "
            "Class3D on the compact engine."
        ),
    )
    parser.add_argument(
        "--relion-softmask-reduction",
        choices=("control", "native_lane", "native_atomic"),
        default="control",
        help=(
            "Diagnostic RELION-CUDA soft-mask background reduction. control keeps "
            "the production deterministic reduction; native_lane and native_atomic "
            "probe source-observer addition orders."
        ),
    )
    parser.add_argument("--image_batch_size", type=int, default=500, help="Images per GPU batch")
    parser.add_argument(
        "--rotation_block_size",
        type=int,
        default=40000,
        help="Rotations per block (larger = faster, less Python overhead)",
    )
    parser.add_argument(
        "--overlap_halves",
        action="store_true",
        help=(
            "Run the two half-sets' E-steps in one thread each. The halves are "
            "independent inside the E-step and kernels still serialise on one "
            "stream, so this only stops the host idling. Off by default; it is "
            "a performance experiment, not a scientific setting."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "RELION --random_seed for half-set splitting, particle order, SamplingPerturbation "
            "and optimiser sampling. If omitted with explicit RELION optimiser/init state, "
            "inherit _rlnRandomSeed; otherwise use the time, as relion_refine does for its "
            "default -1 (ml_optimiser.cpp:2827). The seed used is logged and saved."
        ),
    )
    parser.add_argument(
        "--relion_half_sets",
        default=None,
        help="Path to a RELION data STAR file with rlnRandomSubset column. "
        "If given, use RELION's half-set assignments instead of random seed.",
    )
    parser.add_argument(
        "--relion-half-sets-from-input",
        dest="relion_half_sets_from_input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Rebuild RELION's start-up particle table from <data_dir>/particles.star and --seed "
            "(RELION's --random_seed) as relion_refine does: micrograph-name order, random halves "
            "(input rlnRandomSubset, else srand/rand), scale groups. It replaces --relion_half_sets "
            "and is written to <output>/relion_input_state/, and RELION optimiser STARs are then "
            "not discovered under --data_dir (only --relion_optimiser is read). K=1 only. "
            "Default: on for a fresh K=1 start that is given no RELION output (standalone)."
        ),
    )
    parser.add_argument(
        "--relion_optimiser",
        default=None,
        help="Explicit path to a RELION run_optimiser.star (or "
        "run_it{NNN}_optimiser.star). Used to source the particle-diameter "
        "mask + max_significants. If unset, searches data_dir and any "
        "relion_ref*/ subdirectory.",
    )
    parser.add_argument(
        "--particle_diameter_ang",
        type=float,
        default=None,
        help="RELION --particle_diameter in Angstrom for the scoring mask. If omitted, "
        "a supplied RELION optimiser's value is used, else the GUI default 200 "
        "(pipeline_jobs.cpp:4191).",
    )
    parser.add_argument(
        "--width_mask_edge_px",
        type=float,
        default=5.0,
        help="RELION softMaskOutsideMap edge width in pixels when --particle_diameter_ang is provided.",
    )
    parser.add_argument(
        "--relion_current_sizes",
        default=None,
        help="Comma-separated list of per-iteration current_sizes from RELION "
        "(oracle mode). Example: '0,56,30,50,70,98,98,92,88,90'",
    )
    parser.add_argument(
        "--relion_healpix_orders",
        default=None,
        help="Diagnostic oracle mode: comma-separated base HEALPix order for "
        "every numbered iteration. Suppresses autonomous angular-sampling "
        "transitions while leaving maps, posteriors, noise, and poses autonomous.",
    )
    parser.add_argument(
        "--firstiter_cc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="RELION --firstiter_cc: iter-1 uses normalized cross-correlation scoring + "
        "winner-take-all reconstruction + ini_high low-pass on the iter-1 reference. "
        "Default on, as the GUI passes it unless the reference is on the absolute greyscale "
        "(pipeline_jobs.cpp:4161, 4407; Class3D 3647, 3917). Use --no-firstiter_cc to "
        "reproduce a RELION run without it.",
    )
    parser.add_argument(
        "--firstiter_cc_tree_rescore_max_margin",
        default="auto",
        help="--firstiter_cc K=1: rescore an image's top two coarse CC poses on RELION's coarse tree "
        "when their scores differ by at most this margin. auto (default) is 4e-6 for K=1 and off for "
        "K>1; off disables it.",
    )
    parser.add_argument(
        "--apply-initial-lowpass",
        dest="apply_initial_lowpass",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply RELION's ``initialLowPassFilterReferences`` to the init "
        "reference at ``--init_resolution`` before iter-1 expectation, as "
        "relion_refine does whenever ``--ini_high > 0`` (the GUI passes 60). "
        "Default on, except for a frozen boundary, which owns its reference; "
        "--no-apply-initial-lowpass reproduces a RELION run without --ini_high.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=1,
        help="Number of K-class references for Class3D-style refinement. K=1 "
        "is the auto-refine path; K>1 enables joint class×pose EM. With K>1, "
        "either --init_class_volumes must be provided or "
        "<data_dir>/reference_init_class00K_relion.mrc must exist for each K.",
    )
    parser.add_argument(
        "--relion_init_dir",
        default=None,
        help="Initialize from RELION run_it000_model.star (or "
        "run_it000_half{1,2}_model.star for K=1): load noise, per-class tau2, "
        "tau2 fudge factor and translation sigma instead of bootstrapping. "
        "Combine with --perturb_replay_relion_dir to replay sampling jitter. "
        "Matching initialization alone does not establish trajectory parity.",
    )
    parser.add_argument(
        "--init_class_volumes",
        default=None,
        help="Comma-separated paths to K initial reference maps in RELION's map "
        "convention, as relion_refine reads them (K must equal --n_classes). Defaults to "
        "<data_dir>/reference_init_class00{1..K}_relion.mrc when omitted. Each map needs a "
        "RELION or relax header label or '_relion' in its file name.",
    )
    parser.add_argument(
        "--ref_star",
        default=None,
        help="relion_refine's --ref STAR for Class3D: the _rlnReferenceImage maps "
        "(RELION-frame MRCs, relative paths resolved against the STAR's directory) "
        "are the K initial references, K must equal --n_classes, and the class "
        "distribution starts at 1/K as in relion_refine (a _rlnClassDistribution "
        "column is not read). Exclusive with --init_class_volumes. Each map needs a RELION "
        "or relax header label or '_relion' in its file name "
        "(relax.helpers.map_io.require_relion_convention_reference).",
    )
    parser.add_argument(
        "--init_volume",
        default=None,
        help=(
            "Initial reference map for K=1, in RELION's map convention (the file "
            "relion_refine reads with --ref). Defaults to "
            "<data_dir>/reference_init_relion.mrc when omitted. A map without a RELION "
            "or relax header label must have '_relion' in its file name; any other map "
            "is refused (relax.helpers.map_io.require_relion_convention_reference)."
        ),
    )
    parser.add_argument(
        "--init_previous_best_poses_npz",
        default=None,
        help=(
            "Diagnostic only: seed local-search priors from a RECOVAR "
            "refinement_results.npz file containing per-half best Euler and "
            "translation arrays."
        ),
    )
    parser.add_argument(
        "--init_previous_best_poses_iter",
        default="last",
        help=(
            "Iteration selector for --init_previous_best_poses_npz. Use an "
            "integer, 'last' for the latest numbered iteration, or "
            "'final_all_data'."
        ),
    )
    parser.add_argument(
        "--init_noise_from_npz",
        default=None,
        help=(
            "Diagnostic only: initialize sigma2_noise from a RECOVAR "
            "refinement_results.npz noise_radial_iter_### array instead of "
            "estimating it from images."
        ),
    )
    parser.add_argument(
        "--init_noise_iter",
        default="last",
        help="Iteration selector for --init_noise_from_npz. Use an integer or 'last'.",
    )
    parser.add_argument(
        "--skip_final_iteration",
        action="store_true",
        help="Diagnostic only: skip the final all-data Nyquist iteration.",
    )
    parser.add_argument(
        "--timing_dir",
        default=None,
        help=(
            "Optional directory for lightweight per-iteration timing NPZs. "
            "This uses RELAX_PARITY_TIMING_DIR internally and does not "
            "write full parity tensor/volume dumps."
        ),
    )
    parser.add_argument(
        "--save_intermediates_dir",
        default=None,
        help=(
            "Optional debug directory for per-iteration regularized/unregularized "
            "class maps, Fourier accumulators, assignments, and metadata."
        ),
    )
    parser.add_argument(
        "--save_intermediates_skip_unregularized",
        action="store_true",
        help=(
            "When --save_intermediates_dir is set, save regularized maps and "
            "metadata but skip diagnostic unregularized maps. This preserves "
            "regularized-map FSC debugging while avoiding an extra "
            "reconstruction pass per iteration."
        ),
    )
    parser.add_argument(
        "--local_search_profile",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Control exact-local profile collection. The default profiles when "
            "--save_intermediates_dir is provided; use 'on' for timing ledgers "
            "without full intermediate dumps."
        ),
    )
    parser.add_argument(
        "--stop_after_local_search_profile",
        action="store_true",
        help=(
            "Diagnostic mode: stop after the first local-search E-step has "
            "written timing profiles, without running tau/FSC/reconstruction."
        ),
    )
    parser.add_argument(
        "--stop_after_local_search",
        action="store_true",
        help=(
            "Diagnostic mode: stop after the first local-search E-step without "
            "forcing detailed per-bucket profile collection."
        ),
    )
    parser.add_argument(
        "--stop_after_local_search_score_only",
        action="store_true",
        help=(
            "Diagnostic mode: stop after local pass-2 scoring while skipping "
            "the local M-step/noise accumulators. This is for pose/Pmax "
            "debugging only and does not produce maps or FSC-quality outputs."
        ),
    )
    parser.add_argument(
        "--diagnostic_single_half",
        action="store_true",
        help=(
            "Diagnostic local-search speed mode: with a local-search stop flag, "
            "run only half 1 and leave half 2 empty. This is invalid for map/FSC "
            "runs because it bypasses the gold-standard second half."
        ),
    )
    parser.add_argument(
        "--benchmark_ledger_json",
        default=None,
        help="Optional JSON path for an auto-refine quality/performance ledger.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_optimiser_star",
        default=None,
        help=(
            "Continue a relax run after the numbered iteration of this "
            "<output>/run_itNNN_optimiser.star, as relion_refine --continue does. Repeat "
            "the original command (data, seed, sampling and model options) and add this "
            "option; the run files supply the maps, noise, tau2, poses, corrections, "
            "sampling and convergence state. --max_iter stays the last numbered "
            "iteration of the whole run, like RELION's --iter. Two departures from "
            "relion_refine --continue keep a continued run equal to the uninterrupted one: "
            "each half keeps its own noise spectrum (RELION's MPI restart gives both halves "
            "half 1's), and the particle order stays the run's first mt19937 order (RELION "
            "reshuffles with random_seed + iteration). The run files carry full-precision "
            "floats and extra data_relax_* blocks for the same reason."
        ),
    )
    parser.add_argument(
        "--keep-iterations",
        dest="keep_iterations",
        type=int,
        default=0,
        help=(
            "Keep only the run files of the N newest numbered iterations (the final outputs "
            "are always kept); 0 keeps every iteration, as RELION does. At box 800 each "
            "half map is about 2 GB."
        ),
    )
    parser.add_argument(
        "--write-iteration-every",
        dest="write_iteration_every",
        type=int,
        default=1,
        help=(
            "Write RELION's run_itNNN_{optimiser,model,data,sampling}.star and maps after "
            "every Nth numbered iteration (RELION writes every iteration, N=1); 0 turns "
            "them off. Any written iteration can be continued with --continue."
        ),
    )
    parser.add_argument(
        "--write-unfiltered-half-maps",
        dest="write_unfiltered_half_maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With the run files of an auto-refine iteration, also reconstruct and write "
            "run_itNNN_half{1,2}_class001_unfil.mrc, as RELION does "
            "(ml_optimiser_mpi.cpp:3237-3272)."
        ),
    )
    return parser.parse_args(argv)


def require_command_n_classes(command: str, n_classes: int) -> None:
    """``relax refine`` runs K=1 auto-refine and ``relax class3d`` K>1 classification."""

    if command == "refine" and n_classes != 1:
        raise SystemExit(f"relax refine runs 3D auto-refine (K=1); use relax class3d --n_classes {n_classes}")
    if command == "class3d" and n_classes < 2:
        raise SystemExit("relax class3d needs --n_classes K with K >= 2; use relax refine for K=1")
    if command not in {"refine", "class3d"}:
        raise ValueError(f"unknown refinement command {command!r}")


def resolve_consistency_options(args) -> RelionConsistencyOptions:
    """The RELION-consistency options of the command line, refused where the run cannot honour them.

    A run seeded from, replaying or continuing RELION's own state takes statistics, references or
    projectors computed with RELION's rules, so every non-default option is refused there; the
    refinement loop refuses the particle types it cannot honour once the data is loaded.
    """

    options = RelionConsistencyOptions(
        gridding_kernel=args.gridding_kernel, shell_pair_counting=args.shell_pair_counting
    )
    chosen = options.non_default()
    if not chosen:
        return options
    flags = ", ".join(f"--{name} {value}" for name, value in chosen.items())
    relion_state = [
        flag
        for flag, value in (
            ("--perturb_replay_relion_dir", args.perturb_replay_relion_dir),
            ("--relion_init_dir", args.relion_init_dir),
            ("--frozen-boundary-dir", args.frozen_boundary_dir),
            ("--init_noise_from_npz", args.init_noise_from_npz),
            ("--final-replay-relion-dir", args.final_replay_relion_dir),
            ("--relion-projector-capture-dir", args.relion_projector_capture_dir),
            ("--state-swap-variant", args.state_swap_variant),
        )
        if value is not None
    ]
    if int(args.init_relion_iteration) != 0:
        relion_state.append("--init_relion_iteration")
    if relion_state:
        raise SystemExit(f"{flags}: not available with RELION-seeded or replayed state ({', '.join(relion_state)})")
    if options.gridding_kernel != "radial" and int(args.n_classes) != 1:
        raise SystemExit(f"--gridding_kernel {options.gridding_kernel} is implemented for K=1 auto-refine only")
    return options


def find_relion_optimiser_star(args):
    """Locate a RELION run_optimiser.star to source mask + max_significants from.

    Searches an explicit ``--relion_optimiser`` arg first, then sibling
    directories of ``--relion_half_sets``, the ``--data_dir`` itself, and
    finally any ``relion_ref*/`` subdirectory of ``--data_dir`` (matching
    fixtures that name their RELION output ``relion_ref_os0/`` or similar).
    Picks the latest ``run_it{NNN}_optimiser.star`` if no plain
    ``run_optimiser.star`` is present in a candidate directory.

    With ``--relion-half-sets-from-input`` the run starts from relion_refine's
    inputs alone, so only an explicit ``--relion_optimiser`` is used: a RELION
    output found next to the data must not supply the mask, ``ini_high``,
    ``max_significants`` or CTF flag. A Class3D (K>1) run given no RELION
    state (optimiser, init/replay directory or half-set STAR) is such a
    standalone start too.
    """
    explicit = getattr(args, "relion_optimiser", None)
    if explicit:
        p = Path(explicit).resolve()
        if p.exists():
            return p
    if getattr(args, "relion_half_sets_from_input", False):
        return None
    class3d_standalone = int(getattr(args, "n_classes", 1)) > 1 and not any(
        getattr(args, name, None)
        for name in ("relion_init_dir", "perturb_replay_relion_dir", "relion_half_sets")
    )
    if class3d_standalone:
        return None

    search_dirs = []
    # Strict-parity --relion_init_dir / --perturb_replay_relion_dir point
    # directly at the RELION reference run; check those FIRST so the K=4
    # fixture (with `relion_pdb_k4_os0_ref/` subdir name that doesn't match
    # the `relion_ref*` glob) finds its optimiser star and recovar uses
    # RELION's particle-diameter mask instead of the dataset default.
    relion_init_dir = getattr(args, "relion_init_dir", None)
    if relion_init_dir:
        search_dirs.append(Path(relion_init_dir).resolve())
    perturb_replay_dir = getattr(args, "perturb_replay_relion_dir", None)
    if perturb_replay_dir:
        search_dirs.append(Path(perturb_replay_dir).resolve())
    if args.relion_half_sets is not None:
        search_dirs.append(Path(args.relion_half_sets).resolve().parent)
    data_dir = Path(args.data_dir).resolve()
    search_dirs.append(data_dir)
    search_dirs.append(data_dir / "relion_ref")
    # Match `relion_*ref*/` subdirs (covers `relion_ref_os0/`,
    # `relion_pdb_k4_os0_ref/`, `relion_pdb_k2_os0_ref/`, etc.).
    if data_dir.is_dir():
        for sub in sorted(list(data_dir.glob("relion_ref*")) + list(data_dir.glob("relion_*ref*"))):
            if sub.is_dir():
                search_dirs.append(sub)

    seen = set()
    for d in search_dirs:
        d = d.resolve()
        if d in seen:
            continue
        seen.add(d)
        plain = d / "run_optimiser.star"
        if plain.exists():
            return plain
        # Fall back to the last per-iter optimiser STAR in the directory.
        per_iter = sorted(d.glob("run_it*_optimiser.star"))
        if per_iter:
            return per_iter[-1]
    return None


class VerifiedDispatchSchedule(NamedTuple):
    """Captured follower assignments paired with their verified oracle roots."""

    schedule: "RelionDispatchSchedule | None"
    oracle_dirs: list[Path]


def load_verified_dispatch_schedule(args, particles, *, strict_replay: bool) -> VerifiedDispatchSchedule:
    """Admit a CLI dispatch capture only against its oracle and particle order.

    Verify directory manifests before discovering the consumed optimiser and
    sampling files. See ``docs/math/relion_refinement_algorithm.md#command-admission``.
    """
    from relax.relion.relion_worker_scale import (
        load_relion_dispatch_schedule,
        relion_ordered_particle_sha256,
        verify_relion_dispatch_schedule_oracle,
    )

    relion_dispatch_schedule = None
    oracle_dirs = []
    if args.relion_dispatch_schedule is not None:
        if not strict_replay:
            raise SystemExit(
                "--relion-dispatch-schedule is strict K>1 RELION replay/init state only"
            )
        try:
            relion_dispatch_schedule = load_relion_dispatch_schedule(
                args.relion_dispatch_schedule
            )
            for candidate in (args.perturb_replay_relion_dir, args.relion_init_dir):
                if candidate is None:
                    continue
                resolved = Path(candidate).expanduser().resolve()
                if resolved not in oracle_dirs:
                    oracle_dirs.append(resolved)
                    verify_relion_dispatch_schedule_oracle(
                        relion_dispatch_schedule,
                        resolved,
                    )
            def _require_manifested_oracle_file(path, *, label):
                resolved_path = Path(path).expanduser().resolve()
                for oracle_dir in oracle_dirs:
                    try:
                        relative = resolved_path.relative_to(oracle_dir).as_posix()
                    except ValueError:
                        continue
                    if relative not in relion_dispatch_schedule.oracle_artifact_paths:
                        raise ValueError(
                            f"{label} is not included in the verified RELION oracle manifest: "
                            f"{resolved_path}"
                        )
                    return
                raise ValueError(
                    f"{label} must belong to a verified RELION oracle directory: {resolved_path}"
                )

            discovered_optimiser = find_relion_optimiser_star(args)
            if discovered_optimiser is not None:
                _require_manifested_oracle_file(
                    discovered_optimiser,
                    label="consumed RELION optimiser",
                )
            for oracle_dir in oracle_dirs:
                sampling_candidates = list(oracle_dir.glob("run_it*_sampling.star"))
                final_sampling = oracle_dir / "run_sampling.star"
                if final_sampling.exists():
                    sampling_candidates.append(final_sampling)
                for sampling_path in sampling_candidates:
                    _require_manifested_oracle_file(
                        sampling_path,
                        label="consumed RELION sampling state",
                    )
            observed_group_order = relion_ordered_particle_sha256(particles)
            if observed_group_order != relion_dispatch_schedule.particle_order_sha256:
                raise ValueError(
                    "authoritative RELION group/half-set particle order does not match "
                    "the dispatch schedule"
                )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Invalid --relion-dispatch-schedule: {exc}") from exc
    return VerifiedDispatchSchedule(relion_dispatch_schedule, oracle_dirs)


class RelionRuntimeControls(NamedTuple):
    """Resolved CLI/optimiser controls and significant-support provenance."""

    do_ctf_correction: bool | None
    firstiter_ini_high_angstrom: float | None
    max_significants_resolution: dict


def resolve_relion_runtime_controls(
    optimiser_star,
    *,
    max_significants: int | None,
    target_iteration: int,
    firstiter_cc: bool,
    n_classes: int,
    log,
) -> RelionRuntimeControls:
    """Resolve consumed optimiser controls, CLI precedence and RELION defaults.

    ``target_iteration`` is the first numbered iteration of this invocation.
    See ``docs/math/relion_refinement_algorithm.md#command-admission``.
    """
    from relax.relion import relion_metadata

    expected_accuracy_do_ctf_correction = None
    relion_firstiter_ini_high_angstrom = None
    relion_optimiser_metadata = None
    if optimiser_star is not None:
        from relax.relion.relion_metadata import read_relion_optimiser_metadata

        relion_optimiser_metadata = read_relion_optimiser_metadata(optimiser_star)
        expected_accuracy_do_ctf_correction = relion_optimiser_metadata.get("do_correct_ctf")
        if expected_accuracy_do_ctf_correction is not None:
            expected_accuracy_do_ctf_correction = bool(expected_accuracy_do_ctf_correction)
            log.info(
                "RELION expected-accuracy CTF correction: %s (from %s)",
                expected_accuracy_do_ctf_correction,
                optimiser_star,
            )
        optimiser_text = Path(optimiser_star).read_text(errors="ignore")
        relion_firstiter_ini_high_angstrom = relion_metadata._parse_relion_cli_ini_high(optimiser_text)
        if firstiter_cc:
            if relion_firstiter_ini_high_angstrom is None:
                log.info(
                    "RELION firstiter_cc: no positive --ini_high found in %s",
                    optimiser_star,
                )
            else:
                log.info(
                    "RELION firstiter_cc: using --ini_high %.2f A from %s for post-iter1 low-pass",
                    float(relion_firstiter_ini_high_angstrom),
                    optimiser_star,
                )
    max_significants_resolution = None
    if optimiser_star is not None and relion_optimiser_metadata is not None:
        from relax.relion.relion_metadata import resolve_relion_runtime_max_significants

        optimiser_max_significants = relion_optimiser_metadata.get(
            "maximum_significants_arg"
        )
        if optimiser_max_significants is None:
            optimiser_max_significants = relion_metadata._load_relion_max_significants(optimiser_star)
            relion_optimiser_metadata = dict(relion_optimiser_metadata)
            relion_optimiser_metadata["maximum_significants_arg"] = (
                optimiser_max_significants
            )
        max_significants_resolution = resolve_relion_runtime_max_significants(
            override=max_significants,
            optimiser_metadata=relion_optimiser_metadata,
            target_iteration=target_iteration,
            do_firstiter_cc=firstiter_cc,
            n_classes=n_classes,
            reference_dimension=3,
        )
        max_significants = int(
            max_significants_resolution["active_max_significants"]
        )
        log.info(
            "RELION max_significants: saved_arg=%s active=%d source=%s "
            "do_grad=%s (from %s)",
            max_significants_resolution["maximum_significants_argument"],
            max_significants,
            max_significants_resolution["source"],
            max_significants_resolution["do_grad"],
            optimiser_star,
        )
    if max_significants is None:
        from relax.relion.relion_metadata import resolve_relion_runtime_max_significants

        # Without a RELION optimiser STAR, use relion_refine's own --maxsig default of -1
        # (ml_optimiser.cpp:1109) and its runtime resolution (ml_optimiser.cpp:3692-3699).
        max_significants_resolution = resolve_relion_runtime_max_significants(
            override=None,
            optimiser_metadata={"maximum_significants_arg": -1},
            target_iteration=target_iteration,
            do_firstiter_cc=firstiter_cc,
            n_classes=n_classes,
            reference_dimension=3,
        )
        max_significants_resolution["source"] = "relion_cli_default"
        max_significants = int(max_significants_resolution["active_max_significants"])
        log.info("RELION max_significants: --maxsig default -1 -> active %d", max_significants)
    elif max_significants_resolution is None:
        max_significants_resolution = {
            "maximum_significants_argument": None,
            "active_max_significants": int(max_significants),
            "source": "cli_override",
            "gradient_refine": False,
            "do_grad": False,
            "target_iteration": target_iteration,
        }

    return RelionRuntimeControls(
        expected_accuracy_do_ctf_correction,
        relion_firstiter_ini_high_angstrom,
        max_significants_resolution,
    )
