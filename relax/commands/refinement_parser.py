"""The refinement command line: the flags of ``relax refine`` and ``relax class3d`` and their defaults.

``parse_refinement_args`` declares every flag once, with its default (the RELION 5 GUI's job defaults,
``docs/development/relion_defaults.md``), and the retired flags, which refuse with the feature they named
(``RetiredFlag``; ``CAPTURED_PROJECTOR_TAG`` is the git tag that keeps the captured-projector code). The
definitions moved verbatim from ``relax.commands.refinement_arguments``, which validates the parsed arguments
and resolves them into option records.
"""

import argparse

from relax.io.particle_io import add_particle_read_arguments
from relax.parity import frozen_boundary_cli
from relax.parity.state_swap_probe import add_state_swap_probe_arguments
from relax.refinement.refinement_options import (
    FIRSTITER_CC_TREE_RESCORE_DEFAULT_MAX_MARGIN,
    RELAX_MODE_CONSISTENCY,
    RELION_WIDTH_MASK_EDGE_PX,
)
from relax.refinement.refinement_state import LOCAL_SEARCH_HEALPIX_ORDER
from relax.star import input_poses

# The git tag at the last commit of main that has the captured RELION projector (owner, 2026-10-06).
CAPTURED_PROJECTOR_TAG = "retired/captured-projector-20261006"


class RetiredFlag(argparse.Action):
    """A flag whose feature was deleted: giving it is an error that names the feature and, when the code was
    kept, the git tag that keeps it (``tag``; None: nothing to keep)."""

    def __init__(self, option_strings, dest, *, feature, tag=None, **kwargs):
        super().__init__(option_strings, dest, nargs="?", default=argparse.SUPPRESS, help=argparse.SUPPRESS, **kwargs)
        self.feature, self.tag = feature, tag

    def __call__(self, parser, namespace, values, option_string=None):
        where = "" if self.tag is None else f"; its code is at git tag {self.tag}"
        parser.error(f"{option_string} was retired with {self.feature}{where}")


def parse_refinement_args(argv=None):
    from relax.sampling.symmetry import canonicalize_rotational_symmetry

    parser = argparse.ArgumentParser(
        description="RELION-equivalent 3D auto-refine (relax refine, K=1) and 3D classification "
        "(relax class3d, K>1). See docs/user_guide.md."
    )
    input_poses.add_initial_pose_source_argument(parser)
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
        default=LOCAL_SEARCH_HEALPIX_ORDER,
        help="RELION --auto_local_healpix_order threshold for switching from "
        "global to local angular searches. RELION's binary default is 4; "
        "set to 3 when comparing against runs launched with "
        "--auto_local_healpix_order 3.",
    )
    parser.add_argument(
        "--strict_highres_exp",
        type=float,
        default=None,
        help="RELION --strict_highres_exp (GUI 'Limit resolution E-step to'): resolution in Angstrom that limits "
        "both passes of the expectation step to the image size 2*ROUND(box*pixel/limit), capped at the current "
        "size; the maximization keeps the current size. Default: off.",
    )
    parser.add_argument(
        "--sigma_ang",
        type=float,
        default=None,
        help="RELION --sigma_ang: standard deviation in degrees of the Gaussian prior on all three Euler "
        "angles, which makes every iteration a local angular search of +/-3 sigma around each particle's "
        "angles (the input STAR's at iteration 1, 0 where absent). The GUI's 'Local angular search range' R "
        "is --sigma_ang R/3. Auto-refine replaces it by twice the angular step when the sampling is next "
        "refined at or above --auto_local_healpix_order. Default: none (global searches until that order).",
    )
    parser.add_argument(
        "--sigma_tilt",
        type=float,
        default=None,
        help="RELION --sigma_tilt: standard deviation in degrees of a Gaussian prior on the tilt angle alone, "
        "centred on each particle's rlnAngleTiltPrior (its current tilt where absent): directions within "
        "+/-3 sigma of that tilt, every rot and in-plane angle. For particles picked on a membrane or lattice "
        "(RELION's subtomogram tutorial uses 10). Auto-refine replaces it by the local angular search at "
        "--auto_local_healpix_order. Ignored with --sigma_ang, as in relion_refine. Implemented for "
        "subtomogram auto-refine (relax refine). Default: none.",
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
        "--mode",
        choices=("relion", "relax"),
        default="relion",
        help="relion (default) reproduces RELION, including the places where its arithmetic is not "
        "self-consistent. relax sets every RELION-consistency option below to its consistent value: "
        + ", ".join(f"--{name} {value}" for name, value in RELAX_MODE_CONSISTENCY.items())
        + ". An option given explicitly overrides the mode. Where the run does not honour one of them "
        "(Class3D keeps the radial gridding window; no CC iteration; the gemm_dense coarse engine; "
        "CTF-premultiplied images) relax keeps RELION's rule for that option and logs why; it is refused "
        "where no option is available (subtomograms, several image shapes, RELION-seeded or replayed state). "
        "See docs/math/relion_consistency_options.md.",
    )
    parser.add_argument(
        "--gridding_kernel",
        choices=("radial", "separable"),
        default=None,
        help="Real-space gridding-correction window of the scoring projector and the "
        "reconstructions. radial (default) is RELION's sinc^2(|x| / (pad * box)); separable is the "
        "per-axis sinc^2 product, the exact transform of the trilinear kernel. separable "
        "is implemented for K=1 single-particle auto-refine only and is refused elsewhere.",
    )
    parser.add_argument(
        "--shell_pair_counting",
        choices=("relion", "once"),
        default=None,
        help="How the 3-D shell statistics behind tau2, data-vs-prior, the half-map FSC and the "
        "current-size scheduling count Hermitian pairs. relion (default) visits every stored entry of "
        "the x-half volume, which counts the pairs of the kx = 0 plane twice (updateSSNRarrays, "
        "calculateDownSampledFourierShellCorrelation, computeFourierTransformMap, getSpectrum); once "
        "counts every pair once. The 1/1000 weight floor inside the reconstruction keeps RELION's counting.",
    )
    parser.add_argument(
        "--noise_shell_count",
        choices=("relion", "summed"),
        default=None,
        help="Pixel count per shell of the sigma2_noise update. relion (default) counts every shell on "
        "the full image (Npix_per_shell); below the box the sums of shell current_size / 2 run on the "
        "cropped image, which lacks one row, so that shell's sigma2 comes out 3.6-7.1%% low. summed "
        "divides each shell by the pixels the expectation summed.",
    )
    parser.add_argument(
        "--initial_noise_pair_counting",
        choices=("relion", "once"),
        default=None,
        help="How the start-up noise spectrum counts Hermitian pairs. relion (default) averages each "
        "image's power over every stored pixel of its FFTW half, which counts the pairs of the kx = 0 "
        "column twice, unlike every later sigma2 update; once counts every pair once.",
    )
    parser.add_argument(
        "--nyquist_column_counting",
        choices=("relion", "once"),
        default=None,
        help="How the per-image sums count the Hermitian pairs of the Nyquist column of a full-size image. "
        "relion (default) keeps both members of each pair (the support rule drops only kx = 0, ky < 0), so "
        "they count twice in the Gaussian score, the image power, the noise, norm and scale sums and "
        "Npix_per_shell; once counts each pair once.",
    )
    parser.add_argument(
        "--firstiter_cc_support",
        choices=("relion", "gaussian"),
        default=None,
        help="Support of the first-iteration normalized cross-correlation (--firstiter_cc). relion (default) "
        "sums every pixel of the cropped rectangle, as RELION's CC kernels do (both kx = 0 copies, DC, the "
        "corners); gaussian scores it on the support and weights of the Gaussian iterations.",
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
    add_state_swap_probe_arguments(parser)
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
    for flag in ("--relion-projector-capture-dir", "--relion-projector-capture-manifest",
                 "--relion-projector-capture-iteration"):
        parser.add_argument(flag, action=RetiredFlag, feature="the captured RELION projector",
                            tag=CAPTURED_PROJECTOR_TAG)
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
    parser.add_argument("--image_batch_size", type=int, default=500, help="Images per GPU batch")
    parser.add_argument(
        "--rotation_block_size",
        type=int,
        default=40000,
        help="Rotations per block (larger = faster, less Python overhead)",
    )
    # Deleted 2026-10-12: on the GPU it was slower than serial halves and stopped at random with
    # CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (em_work/overlap_fix_20261012).
    parser.add_argument("--overlap_halves", action=RetiredFlag,
                        feature="the overlapped half-set E-steps (the halves run one after another)")
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
        "--solvent_mask",
        default=None,
        help="RELION --solvent_mask: a reference mask (MRC, values 0-1) that replaces the particle-diameter "
        "sphere in the solvent flatten after each M-step. Resampled and windowed onto the model grid as RELION "
        "does; the start-up reference is not masked.",
    )
    parser.add_argument(
        "--solvent_correct_fsc",
        action="store_true",
        help="RELION --solvent_correct_fsc: the half-set FSC behind tau2, the resolution and its report is "
        "the phase-randomisation corrected FSC of the masked unregularised half maps (needs --solvent_mask). "
        "MPI relion_refine applies it; the non-MPI program ignores the flag.",
    )
    parser.add_argument(
        "--width_mask_edge_px",
        type=float,
        default=RELION_WIDTH_MASK_EDGE_PX,
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
        "--skip_align",
        "--skip-align",
        dest="skip_align",
        action="store_true",
        help="RELION --skip_align (Class3D, GUI 'Perform image alignment: No'): classify at the angles and "
        "offsets of the input STAR, with no pose search. Needs --n_classes > 1; --firstiter_cc needs one "
        "--init_volume. The sampling options are not used.",
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
        f"when their scores differ by at most this margin. auto (default) is {FIRSTITER_CC_TREE_RESCORE_DEFAULT_MAX_MARGIN:g} for K=1 and off for "
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
        "(relax.io.map_io.require_relion_convention_reference).",
    )
    parser.add_argument(
        "--init_volume",
        default=None,
        help=(
            "Initial reference map for K=1, in RELION's map convention (the file "
            "relion_refine reads with --ref). Defaults to "
            "<data_dir>/reference_init_relion.mrc when omitted. A map without a RELION "
            "or relax header label must have '_relion' in its file name; any other map "
            "is refused (relax.io.map_io.require_relion_convention_reference)."
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
            "It takes the place of RELAX_PARITY_TIMING_DIR and does not "
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
