"""RELION-equivalent 3D auto-refine (K=1) and 3D classification (K>1).

``relax refine`` and ``relax class3d`` (``relax/commands/``) run
:func:`run_from_command_line`. The run reads RELION inputs from ``--data_dir``
(``particles.star`` and its stacks, and the reference maps), refines them with
:func:`relax.refinement.iteration_loop.refine_single_volume` and saves the
per-iteration arrays and RELION-style run files under ``--output``.
``docs/user_guide.md`` has runnable examples.
"""

# The EM environment markers below are set before the jax, recovar and relax imports,
# so the imports below them are not at the top of the file.
# ruff: noqa: E402
import dataclasses
import importlib
import logging
import os
import sys
import time
from collections.abc import MutableMapping
from pathlib import Path

# This is an EM entry point, so it opts in to recovar's EM-scoped XLA defaults
# (currently --xla_gpu_autotune_level=0, worth 6-11 s of compile per matched
# pair at every measured state with a steady-iteration term of 0 +- 0.5 s; see
# recovar/jax_config.py for the receipts). It must be set before `import jax`,
# which recovar.jax_config performs, and `setdefault` so an explicit
# RECOVAR_EM_XLA_DEFAULTS=0 in the environment still wins.
os.environ.setdefault("RECOVAR_EM_XLA_DEFAULTS", "1")

import jax
import jax.numpy as jnp
import numpy as np

import relax
from relax.dense import scoring_policy
from relax.diagnostics import frozen_boundary_cli, initial_model_replay, observers, replay_inputs
from relax.diagnostics.state_swap_probe import (
    build_state_swap_probe,
    state_swap_probe_loop_index,
    validate_state_swap_probe_application,
)
from relax.helpers import xla_memory_reserve
from relax.helpers.compilation_cache import activate_recovar_compilation_cache
from relax.helpers.dtype_policy import use_float32_matmuls
from relax.parity import oracle_admission, startup_noise_inputs
from relax.parity.relion_replay_source import RelionReplay, RelionReplaySource
from relax.refinement import command_options, particle_loading, startup_noise, startup_references
from relax.refinement.refinement_options import apply_k1_refine3d_env_defaults
from relax.refinement.result_files import (
    RunReport,
    build_archive_metadata,
    print_refinement_summary,
    write_benchmark_ledger,
    write_final_maps,
    write_profile_only_summary,
    write_refinement_archive,
)
from relax.refinement.run_files import RunFileWriter, RunSettings, read_run_files
from relax.relion import input_particle_table, input_poses, relion_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


_CONCRETE_RECOVAR_PROVENANCE_MODULES = (
    "relax",
    "relax.refinement.iteration_loop",
    "relax.refinement.half_scoring",
    "relax.dense.scoring_policy",
    "relax.classification.k_class",
    "relax.scoring.significance",
)
_K1_RELION_LIVE_INITIAL_NOISE_ENV = "RELAX_K1_RELION_LIVE_INITIAL_NOISE"
_STATE_SWAP_FORCE_FRESH_PARTICLE_ORDER_ENV = (
    "RELAX_STATE_SWAP_FORCE_FRESH_PARTICLE_ORDER"
)


def _k1_relion_live_initial_noise_enabled(
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Return whether the source-faithful fresh-K=1 noise diagnostic is active."""

    environment = os.environ if environ is None else environ
    token = environment.get(_K1_RELION_LIVE_INITIAL_NOISE_ENV, "0").strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"Unsupported {_K1_RELION_LIVE_INITIAL_NOISE_ENV}={token!r}",
    )


def _refuse_live_initial_noise(args, *, frozen_boundary, half_sets, mask_params) -> None:
    """Refuse the fresh-K=1 live-noise diagnostic outside a strict fresh K=1 cold start, naming every reason."""
    invalid_reasons = []
    if int(args.n_classes) != 1:
        invalid_reasons.append("n_classes must equal 1")
    if int(args.init_relion_iteration) != 0:
        invalid_reasons.append("init_relion_iteration must equal 0")
    if frozen_boundary is not None:
        invalid_reasons.append("frozen boundary must be absent")
    if args.perturb_replay_relion_dir is not None:
        invalid_reasons.append("perturb replay must be absent")
    if args.relion_init_dir is None:
        invalid_reasons.append("relion_init_dir is required")
    if half_sets.relion_particles is None or args.relion_half_sets is None:
        invalid_reasons.append("RELION half-set data are required")
    if mask_params is None:
        invalid_reasons.append("RELION particle-diameter mask parameters are required")
    if half_sets.noise_source_rows is None:
        invalid_reasons.append("RELION initial-noise source order is required")
    if half_sets.noise_optics_group_ids is None:
        invalid_reasons.append("RELION initial-noise optics groups are required")
    if half_sets.optics_pixel_sizes is None or half_sets.optics_pixel_sizes.size != 1:
        invalid_reasons.append("exactly one RELION optics pixel size is required")
    if invalid_reasons:
        raise ValueError(
            f"{_K1_RELION_LIVE_INITIAL_NOISE_ENV} is restricted to a strict fresh "
            f"K=1 cold start: {'; '.join(invalid_reasons)}",
        )


def _resolve_relion_firstiter_ini_high(
    *,
    optimiser_ini_high: float | None,
    startup_lowpass_ini_high: float | None,
) -> float | None:
    """Return RELION's ``ini_high`` for the post-iteration-1 re-low-pass.

    RELION filters the start-up references (``initialLowPassFilterReferences``,
    ml_optimiser.cpp:3556) and re-filters them after the firstiter_cc
    iteration 1 (ml_optimiser.cpp:6389-6397) with one ``ini_high``. A positive
    ``--ini_high`` on the RELION optimiser command wins. Otherwise the value
    that drove RECOVAR's ``--apply-initial-lowpass`` start-up filter is that
    ``ini_high``. Without either, RELION's ``ini_high`` is unset and no
    re-low-pass runs: ``--init_resolution`` alone is not an ``ini_high``.
    """
    if optimiser_ini_high is not None:
        return float(optimiser_ini_high)
    if startup_lowpass_ini_high is not None:
        return float(startup_lowpass_ini_high)
    return None


def _assert_expected_repo_imports() -> None:
    """Fail fast if EM modules were imported from another editable checkout.

    ``RECOVAR_EXPECTED_REPO_ROOT`` names the relax checkout that must own every concrete EM module;
    ``RELAX_EXPECTED_RECOVAR_ROOT``, when set, names the RECOVAR checkout that must own ``recovar``.
    """
    expected_root_value = os.environ.get("RECOVAR_EXPECTED_REPO_ROOT")
    if not expected_root_value:
        return

    expected_root = Path(expected_root_value).expanduser().resolve()
    checks = [(module_name, expected_root) for module_name in _CONCRETE_RECOVAR_PROVENANCE_MODULES]
    recovar_root_value = os.environ.get("RELAX_EXPECTED_RECOVAR_ROOT")
    if recovar_root_value:
        checks.append(("recovar", Path(recovar_root_value).expanduser().resolve()))
    failures = []
    for module_name, module_root in checks:
        module = importlib.import_module(module_name)
        module_file_value = getattr(module, "__file__", None)
        module_file = Path(module_file_value).resolve() if module_file_value else None
        logger.info("Import provenance: %s=%s", module_name, module_file)
        if module_file is None or not module_file.is_relative_to(module_root):
            failures.append(f"{module_name}={module_file} (expected under {module_root})")

    if failures:
        raise RuntimeError(
            "RECOVAR import provenance failure: expected every concrete EM module under "
            f"{expected_root}, found " + ", ".join(failures)
        )


def _per_image_optics_groups(particles, layout) -> tuple[list, np.ndarray]:
    """Each half's dense optics-group row per image, for noise held as one spectrum per optics group, and the
    number of particles in each group."""
    from relax.helpers.optics_noise import dense_optics_groups

    image_optics_groups, _ = dense_optics_groups(particles["rlnOpticsGroup"])
    per_half = [image_optics_groups[layout.half1_rows], image_optics_groups[layout.half2_rows]]
    return per_half, np.bincount(image_optics_groups)


def _resolve_tau2_fudge(n_classes, cli_tau2_fudge, relion_init_tau2_fudge):
    """Return the effective tau2_fudge and a human-readable source label."""
    if relion_init_tau2_fudge is not None:
        return float(relion_init_tau2_fudge), "RELION it000 optimiser"
    if cli_tau2_fudge is not None:
        return float(cli_tau2_fudge), "explicit CLI"
    if int(n_classes) > 1:
        return 4.0, "RELION Class3D default"
    return 1.0, "RELION auto-refine default"


def _resolve_replay_normcorr(perturb_replay_relion_dir, replay_relion_normcorr):
    """Default normcorr replay on only for explicit RELION replay runs."""
    if replay_relion_normcorr is not None:
        return bool(replay_relion_normcorr)
    return perturb_replay_relion_dir is not None


def _replay_process_start_noise_broadcast(
    replay_noise_semantics,
    init_relion_iteration,
    perturb_replay_relion_dir,
):
    """Whether replay slot 0 scores both halves with the half-1 noise spectrum.

    RELION MPI initialisation broadcasts follower rank 1's sigma2_noise, so a
    fresh start and a ``--continue`` restart both score their first expectation
    with the half-1 spectrum.  An uninterrupted trajectory scores every later
    iteration with each half's own model-STAR spectrum, which is the state a
    fixed-state replay against an uninterrupted RELION run must reproduce.
    """
    if replay_noise_semantics == "continuation":
        return True
    if perturb_replay_relion_dir is None or int(init_relion_iteration) <= 0:
        raise ValueError(
            "--replay-noise-semantics uninterrupted requires --perturb_replay_relion_dir "
            "and --init_relion_iteration > 0"
        )
    return False


def _validate_replay_saved_healpix_order(
    perturb_replay_relion_dir,
    init_relion_iteration,
    healpix_order,
):
    """Require a mid-trajectory replay to enter with RELION's saved sampling order.

    RELION sizes pass-1 images (``image_coarse_size``) at the start of an
    expectation from the sampling state saved by the previous iteration,
    before ``updateAngularSampling`` switches order.  A replay that starts
    after RELION iteration N takes that pre-update order from
    ``--healpix_order``, so it must equal ``run_itNNN_sampling.star``.
    """
    if perturb_replay_relion_dir is None or int(init_relion_iteration) <= 0:
        return
    sampling_star = (
        Path(perturb_replay_relion_dir)
        / f"run_it{int(init_relion_iteration):03d}_sampling.star"
    )
    saved_order = int(relion_metadata.read_relion_sampling_metadata(sampling_star)["healpix_order"])
    if saved_order != int(healpix_order):
        raise ValueError(
            f"--healpix_order {int(healpix_order)} does not match RELION's saved sampling "
            f"order {saved_order} in {sampling_star}; a replay after RELION iteration "
            f"{int(init_relion_iteration)} sizes pass-1 images from that order"
        )


def _replay_complete_initial_particle_state(n_classes, init_relion_iteration):
    """Whether run_it000 poses/corrections seed the first expectation step.

    AutoRefine (K=1) searches around the run_it000 pre-centering offsets, so
    omitting that state changes the first winners.  Class3D (K>1) performs a
    fresh global first-iteration search and does not seed it from the input
    orientations; replaying those poses would incorrectly compose the global
    samples with the supplied particle orientations.
    """

    return int(n_classes) == 1 and int(init_relion_iteration) == 0


def _use_fresh_auto_refine_particle_order(
    args,
    frozen_boundary,
    *,
    environ=None,
) -> bool:
    """Whether this run owns RELION's one-time fresh AutoRefine ordering."""

    env = os.environ if environ is None else environ
    force_state_swap_order = (
        str(env.get(_STATE_SWAP_FORCE_FRESH_PARTICLE_ORDER_ENV, "0"))
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    if force_state_swap_order and (
        args.perturb_replay_relion_dir is None
        or getattr(args, "state_swap_variant", None) is None
        or getattr(args, "state_swap_target_relion_iteration", None) is None
    ):
        raise ValueError(
            f"{_STATE_SWAP_FORCE_FRESH_PARTICLE_ORDER_ENV}=1 requires a complete "
            "state-swap replay diagnostic"
        )
    # A perturbation replay from iteration 0 is still RELION's fresh run, so it
    # takes the fresh order and the production arithmetic that comes with it.
    # A state-swap diagnostic keeps the order it was captured with unless forced.
    return (
        int(args.n_classes) == 1
        and int(args.init_relion_iteration) == 0
        and frozen_boundary is None
        and (
            getattr(args, "state_swap_variant", None) is None
            or force_state_swap_order
        )
    )


def _resolve_optimizer_random_seed(explicit_seed, relion_optimiser_star):
    """Resolve the optimiser seed without silently diverging from RELION.

    An explicit CLI seed always wins.  For strict-parity runs whose RELION
    optimiser was explicitly supplied, inherit ``_rlnRandomSeed`` when the
    CLI seed is omitted.  Otherwise relion_refine's default ``-1`` takes the
    time (``MlOptimiser::initialiseWorkLoad``, ml_optimiser.cpp:2827).
    """
    if explicit_seed is not None:
        return int(explicit_seed), "explicit CLI"

    if relion_optimiser_star is not None:
        from relax.relion.relion_metadata import read_relion_optimiser_metadata

        metadata = read_relion_optimiser_metadata(relion_optimiser_star)
        relion_seed = metadata.get("random_seed")
        if relion_seed is not None:
            return int(relion_seed), f"RELION optimiser {Path(relion_optimiser_star).resolve()}"

    return int(time.time()), "RELION default -1: the time"


def _effective_perturb_seed(args):
    """Resolve the SamplingPerturbation seed used by the refinement loop.

    RELION uses the optimiser ``--random_seed`` for SamplingPerturbation, so
    the CLI-level ``--seed`` must drive the perturbation stream too. An explicit
    ``--perturb_seed`` overrides it; a negative explicit value keeps the legacy
    non-deterministic NumPy perturbation path for diagnostics.
    """
    if args.perturb_seed is None:
        return args.seed
    return None if args.perturb_seed < 0 else args.perturb_seed


def _write_relion_start_particle_table(our_star, input_star, *, seed, output_dir) -> Path:
    """Write RELION's start-up particle table rebuilt from the input STAR and return its path."""
    from relax.relion.input_particle_table import write_relion_start_particle_star

    if not isinstance(our_star, dict) or "optics" not in our_star:
        raise SystemExit("--relion-half-sets-from-input needs an optics table in <data_dir>/particles.star")
    path = Path(output_dir) / "relion_input_state" / "particles_relion_start.star"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_relion_start_particle_star(input_star, path, seed=int(seed))
    return path


def _initial_current_size(voxel_size: float, box_size: int, init_resolution: float) -> int:
    """Twice RELION's --ini_high pixel, ``getPixelFromResolution(1 / ini_high)`` (ml_model.h:441,
    ml_optimiser.cpp:2801): ``2 ROUND(ori_size pixel_size / ini_high)``. The first E-step then adds
    ``incr_size`` shells (``_bootstrap_current_size_relion``). No floor: RELION has none, and the GUI
    default ini_high of 60 A lands below the former 32-pixel floor on small boxes.
    """

    return 2 * int(np.floor(float(box_size) * float(voxel_size) / float(init_resolution) + 0.5))


def _require_relion_convention_reference(path, option: str) -> None:
    """Stop the run when a reference map's RELION convention is not established (relax.helpers.map_io)."""
    from relax.helpers.map_io import require_relion_convention_reference

    try:
        require_relion_convention_reference(path, option=option)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


@xla_memory_reserve.explains_pool_region_failure
def main(command=None):
    """Run a refinement from ``sys.argv``; ``command`` (``refine``/``class3d``) checks ``--n_classes``."""

    # This module imports JAX before recovar, so recovar's cache environment must be applied to the live config.
    cache_directory = activate_recovar_compilation_cache()
    use_float32_matmuls()
    _assert_expected_repo_imports()
    args = command_options.parse_refinement_args()
    if command is not None:
        command_options.require_command_n_classes(command, int(args.n_classes))
    command_options.require_consistency_arguments(args)
    if cache_directory:
        logger.info("Persistent JAX compilation cache: %s", cache_directory)
    if int(args.n_classes) == 1:
        apply_k1_refine3d_env_defaults()
    command_options.resolve_job_defaults(args)
    command_options.resolve_standalone_k1_start(args)
    command_options.validate_sigma_ang(args)
    command_options.validate_strict_highres_exp(args)
    if (
        args.state_swap_target_relion_iteration is not None
        or args.state_swap_variant is not None
        or args.state_swap_replay_relion_references
    ) and int(args.n_classes) != 1:
        raise SystemExit("state-swap diagnostics currently require --n_classes 1")
    state_swap_probe_loop_index(
        target_relion_iteration=args.state_swap_target_relion_iteration,
        variant=args.state_swap_variant,
        replay_relion_references=args.state_swap_replay_relion_references,
        init_relion_iteration=args.init_relion_iteration,
        max_iter=args.max_iter,
    )
    replay_process_start_noise_broadcast = _replay_process_start_noise_broadcast(
        args.replay_noise_semantics,
        args.init_relion_iteration,
        args.perturb_replay_relion_dir,
    )
    if args.frozen_boundary_dir is None:
        # A sealed frozen boundary owns its sampling state, including coarse size.
        _validate_replay_saved_healpix_order(
            args.perturb_replay_relion_dir,
            args.init_relion_iteration,
            args.healpix_order,
        )

    frozen_boundary, fixed_diagnostic_source_paths = frozen_boundary_cli.load_cli_boundary(args)

    command_options.validate_skip_align_args(args)
    if args.continue_optimiser_star is not None:
        args.seed = command_options.validate_continue_args(args)

    # The one RELION optimiser STAR the run reads: a sealed boundary's completed optimiser, or the one the
    # arguments locate (see command_options.relion_optimiser_star).
    fixed_diagnostic_arm = frozen_boundary is not None and frozen_boundary.fixed_diagnostic_arm
    if fixed_diagnostic_arm and fixed_diagnostic_source_paths is None:
        raise ValueError("fixed diagnostic arm lacks sealed source paths")
    optimiser_star = command_options.relion_optimiser_star(
        args,
        sealed_optimiser=fixed_diagnostic_source_paths["completed_optimiser"] if fixed_diagnostic_arm else None,
    )
    args.seed, optimizer_seed_source = _resolve_optimizer_random_seed(
        args.seed, command_options.optimiser_seed_source(args, optimiser_star, sealed=fixed_diagnostic_arm)
    )
    logger.info("Optimiser random seed: %d (%s)", args.seed, optimizer_seed_source)

    if args.timing_dir:
        timing_dir_path = Path(args.timing_dir)
        timing_dir_path.mkdir(parents=True, exist_ok=True)
        os.environ["RELAX_PARITY_TIMING_DIR"] = str(timing_dir_path)
    else:
        timing_dir_path = None

    # relax/__init__.py sized the XLA pool for the projector texture before the backend started.
    if relax._XLA_RESERVE_LOG_LINE is not None:
        logger.info("%s", relax._XLA_RESERVE_LOG_LINE)
    # An entry point that started the backend without that reserve is refused here, not in the final pass.
    from relax.relion.geometry import IMAGE_MASK_EDGE_PIXELS, PROJECTION_PADDING_FACTOR

    xla_memory_reserve.require_projector_texture_reserve(
        xla_memory_reserve.model_box_from_map_headers(xla_memory_reserve.reference_maps_from_argv(sys.argv[1:])),
        PROJECTION_PADDING_FACTOR,
    )

    # Verify GPU
    devices = jax.devices()
    logger.info("JAX devices: %s", devices)
    if not any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in devices):
        logger.error("No GPU available. Aborting.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    particle_inputs = particle_loading.load_particle_inputs(
        args,
        frozen_boundary=frozen_boundary,
        fixed_diagnostic_source_paths=fixed_diagnostic_source_paths,
    )
    ds = particle_inputs.dataset
    # The image box in pixels, read once from the dataset (RECOVAR calls it grid_size).
    box_size = int(ds.grid_size)
    tomo_run = particle_inputs.tomographic
    shape_class_rows = particle_inputs.shape_class_rows
    # The one place --mode is read: after it the run carries the options only.
    consistency_options = command_options.resolve_consistency_options(
        args, subtomograms=tomo_run, several_image_shapes=shape_class_rows is not None, dataset=ds
    )
    relion_mask_params = particle_inputs.mask_parameters
    _double_image_preprocessing = particle_inputs.double_preprocessing
    del particle_inputs
    args._relion_mask_params = relion_mask_params
    particle_diameter_ang = None if relion_mask_params is None else float(relion_mask_params[0])
    # The loader's resolved edge (--width_mask_edge_px, or a found or sealed optimiser's rlnWidthMaskEdge).
    width_mask_edge_px = IMAGE_MASK_EDGE_PIXELS if relion_mask_params is None else float(relion_mask_params[1])
    logger.info("Dataset: %d images, image_shape=%s, voxel_size=%.3f A/px", ds.n_units, ds.image_shape, ds.voxel_size)

    # ---- Create half-sets ----
    n_images = ds.n_units
    if args.n_classes < 1:
        raise SystemExit(f"--n_classes must be >= 1, got {args.n_classes}")
    if args.ref_star is not None and args.n_classes == 1:
        raise SystemExit("--ref_star is the Class3D (K>1) reference list; use --init_volume for K=1")

    import starfile as _starfile

    our_star = _starfile.read(os.path.join(args.data_dir, "particles.star"))
    our_particles = our_star["particles"] if isinstance(our_star, dict) else our_star
    particle_loading.validate_particle_optics(
        our_star.get("optics") if isinstance(our_star, dict) else None,
        tomographic=tomo_run,
    )
    # Keep the input-STAR particle identities available for replay mapping.
    # RELION data STAR rows can be permuted relative to this table, so callers
    # must map by rlnImageName rather than assuming row positions coincide.
    our_names = np.asarray(our_particles["rlnImageName"])
    use_relion_live_initial_noise = _k1_relion_live_initial_noise_enabled()
    # A run that loads no noise state starts from RELION's estimate from the images.
    relion_startup_noise_needed = (
        frozen_boundary is None and args.init_noise_from_npz is None and args.relion_init_dir is None
    )
    relion_model_pixel_size = None

    if args.relion_half_sets_from_input:
        args.relion_half_sets = str(
            _write_relion_start_particle_table(
                our_star,
                os.path.join(args.data_dir, "particles.star"),
                seed=int(args.seed),
                output_dir=args.output,
            )
        )
        logger.info(
            "RELION start-up particle table rebuilt from the input STAR with seed %d: %s",
            int(args.seed),
            args.relion_half_sets,
        )

    use_fresh_auto_refine_order = args.relion_half_sets is not None and _use_fresh_auto_refine_particle_order(
        args, frozen_boundary
    )
    half_sets = particle_loading.split_half_sets(
        our_star,
        ds,
        halfset_path=args.relion_half_sets,
        n_classes=int(args.n_classes),
        seed=int(args.seed),
        init_relion_iteration=args.init_relion_iteration,
        fresh_auto_refine_order=use_fresh_auto_refine_order,
        noise_order_needed=relion_startup_noise_needed or (args.n_classes == 1 and use_relion_live_initial_noise),
        tomographic=tomo_run,
    )
    particle_layout = half_sets.layout
    relion_particles = half_sets.relion_particles

    local_stop_requested = (
        bool(args.stop_after_local_search_profile)
        or bool(args.stop_after_local_search)
        or bool(args.stop_after_local_search_score_only)
    )
    if args.diagnostic_single_half:
        if not local_stop_requested:
            raise SystemExit(
                "--diagnostic_single_half is only valid with --stop_after_local_search, "
                "--stop_after_local_search_profile, or --stop_after_local_search_score_only"
            )
        if args.n_classes != 1:
            raise SystemExit("--diagnostic_single_half is K=1-only")
        logger.warning(
            "Diagnostic single-half local-search probe: running half 1 only (%d images); "
            "half 2 is empty. Do not use this for map/FSC quality.",
            int(particle_layout.half1_rows.size),
        )
        particle_layout = particle_layout._replace(half2_rows=np.empty(0, dtype=np.int64))

    resume_snapshot = None
    if args.continue_optimiser_star is not None:
        if shape_class_rows is not None:
            raise SystemExit(
                "--continue does not support particle STARs with several image shapes yet (relax#38); "
                "restart the run from its input instead"
            )
        resume_snapshot = read_run_files(
            args.continue_optimiser_star,
            image_names=[str(name) for name in our_names],
            half_rows=[particle_layout.half1_rows, particle_layout.half2_rows],
        )
        logger.info(
            "Continuing after numbered iteration %d from %s",
            resume_snapshot.relion_iteration,
            args.continue_optimiser_star,
        )
        if int(args.max_iter) < int(resume_snapshot.relion_iteration):
            raise SystemExit(
                f"--max_iter {args.max_iter} is the last numbered iteration of the whole run "
                f"(RELION's --iter); the run files are already at iteration {resume_snapshot.relion_iteration}"
            )

    ds_half1 = ds.subset(particle_layout.half1_rows)
    ds_half2 = ds.subset(particle_layout.half2_rows)
    logger.info("Half-sets: %d + %d images", ds_half1.n_units, ds_half2.n_units)
    if frozen_boundary is not None:
        frozen_boundary_cli.validate_particle_half_inputs(
            frozen_boundary,
            image_names=our_names,
            half_rows=(particle_layout.half1_rows, particle_layout.half2_rows),
            image_shape=ds.image_shape,
            log=logger,
        )
    prepared_particle_groups = input_particle_table.prepare_particle_group_layout(
        our_particles,
        particle_layout.half1_rows,
        particle_layout.half2_rows,
        halfset_particles=relion_particles,
        halfset_source=args.relion_half_sets,
        replay_dirs=(args.perturb_replay_relion_dir, args.relion_init_dir),
        init_relion_iteration=args.init_relion_iteration,
    )
    group_particle_source = prepared_particle_groups.source
    particle_groups = prepared_particle_groups.layout
    follower_routing = oracle_admission.admit_follower_routing(
        args, group_particle_source, particle_groups, log=logger
    )
    relion_dispatch_schedule = follower_routing.schedule
    follower_topology = follower_routing.topology

    runtime_controls = command_options.resolve_relion_runtime_controls(
        optimiser_star,
        max_significants=args.max_significants,
        target_iteration=int(args.init_relion_iteration) + 1,
        firstiter_cc=bool(args.firstiter_cc),
        n_classes=int(args.n_classes),
        log=logger,
    )
    args.max_significants = runtime_controls.max_significants_resolution["active_max_significants"]
    expected_accuracy_do_ctf_correction = runtime_controls.do_ctf_correction
    relion_firstiter_ini_high_angstrom = runtime_controls.firstiter_ini_high_angstrom

    # ---- Load initial volume ----
    # RELION's Image<RFLOAT>::read() widens a reference MRC (on-disk float32) to RFLOAT (double in the
    # ACC_DOUBLE_PRECISION oracle build); every later step, including the FFT that builds Projector::data,
    # runs at that precision. Narrowing here would defeat float64 projections however carefully later casts
    # widen again (DensePrecisionPolicy.cast_projection_volume cannot recover it). Gated on
    # use_float64_projections alone, as projection_complex_dtype is, so float64 scoring alone costs no
    # projection memory. The process's precision policy owns the flag (RefinementOptions.precision is it).
    _init_volume_use_float64 = scoring_policy.DENSE_PRECISION.use_float64_projections
    _init_volume_dtype = np.float64 if _init_volume_use_float64 else np.float32
    _init_volume_complex_dtype = np.complex128 if _init_volume_use_float64 else np.complex64

    # RELION's ``initialLowPassFilterReferences`` (ml_optimiser.cpp:3556) low-pass filters mymodel.Iref at
    # start-up, gated only on ``ini_high > 0``; ``--apply-initial-lowpass`` mirrors it at --init_resolution.
    _apply_ini_lowpass = bool(getattr(args, "apply_initial_lowpass", False))
    _ini_high_for_lowpass = (
        float(args.init_resolution)
        if _apply_ini_lowpass and float(args.init_resolution) > 0.0
        else None
    )
    if args.firstiter_cc:
        _optimiser_ini_high = relion_firstiter_ini_high_angstrom
        relion_firstiter_ini_high_angstrom = _resolve_relion_firstiter_ini_high(
            optimiser_ini_high=_optimiser_ini_high,
            startup_lowpass_ini_high=_ini_high_for_lowpass,
        )
        if relion_firstiter_ini_high_angstrom is None:
            logger.info(
                "RELION firstiter_cc: no positive --ini_high and no --apply-initial-lowpass; "
                "not applying post-iter1 ini_high low-pass",
            )
        elif _optimiser_ini_high is None:
            logger.info(
                "RELION firstiter_cc: using the --apply-initial-lowpass ini_high %.2f A "
                "for the post-iter1 low-pass",
                relion_firstiter_ini_high_angstrom,
            )
    _use_initial_projector_real, firstiter_cc_tree_rescore_max_margin = command_options.resolve_firstiter_controls(
        firstiter_cc=bool(args.firstiter_cc),
        n_classes=int(args.n_classes),
        tree_rescore_max_margin=args.firstiter_cc_tree_rescore_max_margin,
    )
    if firstiter_cc_tree_rescore_max_margin is not None:
        logger.info(
            "RELION firstiter_cc: coarse-tree top-2 rescore max_margin=%g",
            firstiter_cc_tree_rescore_max_margin,
        )
    if _use_initial_projector_real:
        logger.info(
            "RELION initial projector: direct real-reference handoff enabled "
            "(K=1 firstiter_cc default or explicit environment override)",
        )
    if frozen_boundary is not None:
        if frozen_boundary.volume_shape != tuple(int(value) for value in ds.volume_shape):
            raise SystemExit(
                "Frozen-boundary volume_shape does not match the active dataset: "
                f"boundary={frozen_boundary.volume_shape}, dataset={tuple(ds.volume_shape)}"
            )
        references = startup_references.frozen_boundary_references(
            frozen_boundary, complex_dtype=_init_volume_complex_dtype,
        )
        logger.info(
            "Initial per-half Fourier volumes loaded from frozen boundary %s",
            frozen_boundary.source_dir,
        )
    elif args.n_classes == 1:
        init_mrc_path = args.init_volume or os.path.join(args.data_dir, "reference_init_relion.mrc")
        _require_relion_convention_reference(init_mrc_path, "--init_volume")
        references = startup_references.load_k1_reference(
            init_mrc_path,
            volume_shape=ds.volume_shape,
            ini_high=_ini_high_for_lowpass,
            real_for_projector=_use_initial_projector_real,
            real_dtype=_init_volume_dtype,
            complex_dtype=_init_volume_complex_dtype,
            log=logger,
        )
        relion_model_pixel_size = references.model_pixel_size
    else:
        class_paths, class_option = command_options.resolve_class_reference_paths(args, log=logger)
        for p in class_paths:
            _require_relion_convention_reference(p, class_option)
        references = startup_references.load_class_references(
            class_paths,
            volume_shape=ds.volume_shape,
            ini_high=_ini_high_for_lowpass,
            real_for_projector=_use_initial_projector_real,
            real_dtype=_init_volume_dtype,
            complex_dtype=_init_volume_complex_dtype,
            log=logger,
        )

    # ---- Set up rotation and translation grids ----
    from relax.sampling import _relion_base_translation_grid, rotation_grid_size

    initial_sampling = command_options.resolve_initial_sampling(
        args.healpix_order,
        args.adaptive_oversampling,
        n_classes=args.n_classes,
        max_healpix_order=args.max_healpix_order,
    )
    rotation_grid_order = initial_sampling.coarse_order
    logger.info(
        "RELION grid orders: coarse/pass1=%d, fine/pass2=%d (adaptive_oversampling=%d)",
        initial_sampling.coarse_order,
        initial_sampling.fine_order,
        args.adaptive_oversampling,
    )
    logger.info(
        "Max coarse HEALPix order: %s (%s)",
        "none" if initial_sampling.max_order is None else initial_sampling.max_order,
        initial_sampling.max_order_source,
    )

    from relax.symmetry import (
        canonicalize_rotational_symmetry,
        parse_rotational_symmetry,
        relion_point_group_code,
        symmetry_operator_sha256,
    )

    symmetry = canonicalize_rotational_symmetry(args.sym)
    parsed_symmetry = parse_rotational_symmetry(symmetry)
    symmetry_point_group, symmetry_point_group_order = relion_point_group_code(symmetry)
    symmetry_provenance = {
        "label": symmetry,
        "family": parsed_symmetry.family,
        "operator_count": int(parsed_symmetry.operator_count),
        "operator_sha256": symmetry_operator_sha256(symmetry),
        "relion_point_group": int(symmetry_point_group),
        "relion_point_group_order": int(symmetry_point_group_order),
    }
    n_rotations = rotation_grid_size(rotation_grid_order, symmetry=symmetry)
    logger.info("Symmetry provenance: %s", symmetry_provenance)
    # Iteration 1 starts from the grid every later iteration rebuilds: RELION's (ceil, Angstrom tolerance),
    # in host float64 (HealpixSampling::setTranslations, healpix_sampling.cpp:344, 413-454).
    translations = _relion_base_translation_grid(
        args.offset_range, args.offset_step, n_classes=args.n_classes, voxel_size=ds.voxel_size
    )
    logger.info("Rotation grid: %d rotations (healpix_order=%d)", n_rotations, rotation_grid_order)
    logger.info(
        "Translation grid: %d translations (range=%.1f, step=%.1f)",
        translations.shape[0],
        args.offset_range,
        args.offset_step,
    )

    # ---- Initialize noise and prior ----
    # Use a RELION-style initial sigma2 estimate from particle power spectra
    # instead of a flat unit spectrum, so iteration 1 starts on a comparable
    # likelihood scale.

    optics_group_ids_per_half = None
    if frozen_boundary is not None:
        initial_noise = startup_noise_inputs.frozen_boundary_noise(frozen_boundary, ds.image_shape)
        logger.info(
            "Initial noise/tau2 state is owned by frozen boundary %s",
            frozen_boundary.source_dir,
        )
    elif args.init_noise_from_npz is not None:
        initial_noise = startup_noise_inputs.archived_noise(
            args.init_noise_from_npz, args.init_noise_iter, ds.image_shape, log=logger,
        )
    elif args.relion_init_dir is not None:
        # The RELION-seeded debug start loads RELION's iteration-0 model noise below.
        initial_noise = None
    elif resume_snapshot is not None:
        # The run files own the noise; the loop installs each half's own spectrum.
        initial_noise = startup_noise.continued_noise(resume_snapshot.noise_shells, ds.image_shape)
        if initial_noise.pixel_variance.ndim == 2:
            optics_group_ids_per_half, _ = _per_image_optics_groups(our_particles, particle_layout)
    else:
        if args.n_classes == 1 and args.relion_half_sets is None:
            raise ValueError(
                "RELION start-up noise needs K=1 half sets (Class3D uses the input order), the "
                "particle-diameter mask and the optics pixel size, and no frozen or loaded noise"
            )
        initial_noise = startup_noise.prepare_startup_noise(
            ds,
            source_rows=half_sets.noise_source_rows,
            optics_group_ids=half_sets.noise_optics_group_ids,
            mask_params=relion_mask_params,
            optics_pixel_sizes=half_sets.noise_optics_pixel_sizes,
            output_dtype=np.float64 if _double_image_preprocessing else np.float32,
            pair_counting=consistency_options.initial_noise_pair_counting,
        )
        logger.info(
            "RELION start-up noise from the images: %d shells, scoring dtype=%s",
            initial_noise.radial.size,
            initial_noise.pixel_variance.dtype,
        )
        if initial_noise.pixel_variance.ndim == 2:
            # One spectrum per optics group; every image scores with its own group's.
            optics_group_ids_per_half, group_sizes = _per_image_optics_groups(our_particles, particle_layout)
            logger.info(
                "Per-optics-group noise: %d groups, images per group %s",
                initial_noise.pixel_variance.shape[0],
                group_sizes.tolist(),
            )
    initial_noise_radial = None if initial_noise is None else initial_noise.radial
    noise_variance = None if initial_noise is None else initial_noise.pixel_variance
    del initial_noise

    # Compute initial signal prior from init volume (weak prior). For K>1
    # use class-1 as the representative volume; the engine derives per-class
    # tau2 trajectories from the per-class FSCs once the loop starts.
    mean_variance = startup_references.bootstrap_prior(references.prior_source, ds.volume_shape)

    # ---- STRICT-PARITY: --relion_init_dir override of bootstrapped iter-0 state ----
    # When set, replace the image-bootstrap sigma2_noise + power-spectrum-bootstrap
    # tau2 with RELION's exact iter-0 values from run_it000_model.star. This
    # eliminates the ~1e-3 relative drift between bootstraps that flips ~22%
    # of K=4 iter-1 class assignments and caps cold-start mean_corr at 0.94.
    relion_init_sigma_offset_angstrom = None
    relion_init_tau2_fudge = None
    relion_live_initial_sigma2 = None
    relion_live_initial_noise_variance = None
    if use_relion_live_initial_noise:
        _refuse_live_initial_noise(args, frozen_boundary=frozen_boundary, half_sets=half_sets,
                                   mask_params=relion_mask_params)
        relion_live_initial_sigma2, relion_live_initial_noise_variance = startup_noise_inputs.live_initial_noise(
            ds, half_sets, mask_params=relion_mask_params, log=logger,
        )
    if args.relion_init_dir is not None and frozen_boundary is None:
        initial_model = initial_model_replay.read_initial_model(
            args.relion_init_dir,
            n_classes=args.n_classes,
        )
        replayed_noise = initial_model_replay.prepare_noise(
            initial_model,
            box_size=box_size,
            image_shape=ds.image_shape,
            explicit_noise_radial=initial_noise_radial,
            live_sigma2=relion_live_initial_sigma2,
            log=logger,
        )
        noise_variance = replayed_noise.variance
        initial_model_replay.log_noise_source(replayed_noise, log=logger)
        del replayed_noise
        mean_variance = initial_model_replay.prepare_prior(
            initial_model,
            n_classes=args.n_classes,
            box_size=box_size,
            volume_shape=ds.volume_shape,
        )
        if args.n_classes > 1:
            logger.info(
                "STRICT-PARITY: replaced bootstrapped per-class tau2 with RELION it000 spectra (K=%d)",
                args.n_classes,
            )
        else:
            logger.info("STRICT-PARITY: replaced bootstrapped tau2 with RELION it000 spectrum (K=1)")
        relion_init_tau2_fudge, relion_init_sigma_offset_angstrom = initial_model_replay.read_controls(
            initial_model,
            log=logger,
        )

    relion_start_data_vs_prior = None
    # RELION's start-up tau2 is defined against RELION's start-up noise.
    fresh_relion_start = (
        frozen_boundary is None
        and args.init_noise_from_npz is None
        and args.relion_init_dir is None
        and int(args.init_relion_iteration) == 0
    )
    if references.reference_real is not None and fresh_relion_start:
        mean_variance, relion_start_data_vs_prior = startup_references.relion_start_tau2_and_data_vs_prior(
            references.reference_real,
            initial_noise_radial,
            box_size=box_size,
            volume_shape=ds.volume_shape,
            tau2_fudge=_resolve_tau2_fudge(args.n_classes, args.tau2_fudge, None)[0],
            nr_particles=int(ds_half1.n_units),
            shell_pair_counting=consistency_options.shell_pair_counting,
        )
        logger.info(
            "RELION start-up tau2/data_vs_prior (initialiseDataVersusPrior): %d shells with data_vs_prior > 3",
            int(np.sum(relion_start_data_vs_prior > 3.0)),
        )
    elif references.class_references_real is not None and fresh_relion_start and resume_snapshot is None:
        # Without it every shell entered the first scale-correction sums (subtomogram Class3D it001 group
        # scales 5.6e-4 off RELION).
        relion_start_data_vs_prior = startup_references.class_start_data_vs_prior(
            references.class_references_real,
            initial_noise_radial,
            box_size=box_size,
            volume_shape=ds.volume_shape,
            tau2_fudge=_resolve_tau2_fudge(args.n_classes, args.tau2_fudge, None)[0],
            nr_particles=int(ds.n_units),
            shell_pair_counting=consistency_options.shell_pair_counting,
        )
        logger.info(
            "RELION start-up data_vs_prior per class (initialiseDataVersusPrior): %s shells with data_vs_prior > 3",
            np.sum(relion_start_data_vs_prior > 3.0, axis=1).tolist(),
        )
    # The start-up real maps are read for the last time above; only a fresh run hands the loop
    # its projector reference (through the replay options). Released here, they no longer stay
    # alive through the whole refinement (float64, 4.1 GB at EMPIAR-10202's box 800).
    references = dataclasses.replace(
        references,
        reference_real=None,
        class_references_real=None,
        real_for_projector=None if resume_snapshot is not None else references.real_for_projector,
    )

    if frozen_boundary is not None:
        mean_variance = jnp.asarray(
            np.stack(frozen_boundary.mean_variance_per_half, axis=0)
            if frozen_boundary.fixed_diagnostic_arm
            else frozen_boundary.mean_variance
        )

    # Compute initial current_size from init_resolution, unless an atomic
    # frozen boundary owns the numbered-iteration schedule.
    init_current_size = (
        int(frozen_boundary.current_size)
        if frozen_boundary is not None
        else _initial_current_size(ds.voxel_size, box_size, args.init_resolution)
    )
    logger.info("Initial current_size from resolution %.1f A: %d pixels", args.init_resolution, init_current_size)

    # ---- Run refinement ----
    from relax.refinement.iteration_loop import refine_single_volume
    from relax.refinement.refinement_options import (
        CheckpointOptions,
        ExpectedAccuracyOptions,
        RefinementOptions,
        RelionParityOptions,
        ReplayState,
        SolventOptions,
        SymmetryOptions,
    )

    experiment_datasets = [ds_half1, ds_half2]
    translations_jnp = jnp.asarray(translations)

    logger.info("=" * 70)
    logger.info(
        "Starting RELION-parity refinement: max_iter=%d, adaptive_oversampling=%d",
        args.max_iter,
        args.adaptive_oversampling,
    )
    logger.info("=" * 70)

    adaptive_options = command_options.resolve_adaptive_options(args, log=logger)

    # Build per-iter replay overrides from RELION's per-iter data.star +
    # model.star when --perturb_replay_relion_dir is set. The override always
    # injects RELION's per-iter sigma_offset (parity-critical: recovar's iter-1
    # does not run the C1 sigma_offset update so iter-2 would otherwise use the
    # 10 Å default — 6× too wide vs RELION ~1.6 Å — depressing iter-2 Pmax by
    # ~22%). Per-image normCorrection / group-scale replay is part of strict
    # RELION replay and can be disabled with --no-replay_relion_normcorr for
    # diagnostics.
    replay_target = replay_inputs.ReplayTarget(
        half1_rows=particle_layout.half1_rows,
        half2_rows=particle_layout.half2_rows,
        particle_names=our_names,
        voxel_size=ds.voxel_size,
        box_size=box_size,
        volume_shape=ds.volume_shape,
        noise_dtype=np.float64 if _double_image_preprocessing else np.float32,
    )
    replay_iteration_overrides = None
    if args.perturb_replay_relion_dir is not None:
        if frozen_boundary is not None:
            replay_iteration_overrides = frozen_boundary_cli.empty_replay_slots(args.max_iter)
            logger.info(
                "Diagnostic frozen restart: local replay slot 0 is empty; "
                "sealed per-half scoring state suppresses process-start noise broadcast"
            )
        else:
            replay_iteration_overrides = replay_inputs.numbered_star_replay(
                args.perturb_replay_relion_dir,
                replay_target,
                max_iter=args.max_iter,
                init_relion_iteration=args.init_relion_iteration,
                include_normcorr=_resolve_replay_normcorr(args.perturb_replay_relion_dir, args.replay_relion_normcorr),
                include_k1_state_swap=args.state_swap_target_relion_iteration is not None,
                process_start_noise_broadcast=replay_process_start_noise_broadcast,
            )
            logger.info(
                "Replay noise semantics: %s (slot-0 process-start broadcast=%s)",
                args.replay_noise_semantics,
                replay_process_start_noise_broadcast,
            )

    final_replay = replay_inputs.FinalReplay()
    if args.final_replay_relion_dir is not None:
        final_replay = replay_inputs.final_only_replay(
            args.final_replay_relion_dir,
            replay_target,
            max_iter=args.max_iter,
            explicit_source_iteration=args.final_replay_source_iteration,
            fields=args.final_replay_fields,
            init_relion_iteration=args.init_relion_iteration,
            n_classes=args.n_classes,
            log=logger,
        )

    # ``--relion_init_dir`` is the strict cold-start contract, not merely a
    # noise/tau bootstrap. RELION's run_it000 particle/model state includes
    # large pre-centering offsets on real data; omitting them makes iter-1
    # search around zero and changes the hard firstiter-CC winners even though
    # the starting reference and Pmax values appear to match.
    kclass_firstiter_translations = None
    kclass_firstiter_translation_path = None
    if args.relion_init_dir is not None and _replay_complete_initial_particle_state(
        args.n_classes,
        args.init_relion_iteration,
    ):
        first_state = replay_inputs.k1_initial_state(
            args.relion_init_dir,
            replay_target,
            explicit_noise=noise_variance if args.init_noise_from_npz is not None else None,
            live_noise=relion_live_initial_noise_variance,
            log=logger,
        )
        if first_state is not None:
            if replay_iteration_overrides is None:
                replay_iteration_overrides = [None] * (args.max_iter + 1)
            replay_iteration_overrides[0] = first_state
            logger.info(
                "STRICT-PARITY: loaded complete RELION run_it000 cold-start state "
                "for the first expectation step",
            )
        else:
            logger.warning(
                "STRICT-PARITY: %s did not provide a complete run_it000 data/model "
                "state; first-iteration particle pre-centering remains unset",
                args.relion_init_dir,
            )
    elif args.relion_init_dir is not None and int(args.n_classes) > 1:
        if int(args.init_relion_iteration) == 0:
            kclass_firstiter_translations, kclass_firstiter_translation_path = (
                replay_inputs.class3d_initial_translations(
                    args.relion_init_dir,
                    replay_target,
                    n_classes=args.n_classes,
                    init_relion_iteration=args.init_relion_iteration,
                )
            )
            logger.info(
                "STRICT-PARITY: Class3D first iteration keeps run_it000 input "
                "origins for FFT pre-shifting while orientations, corrections, "
                "priors, and noise remain fresh",
            )
        else:
            logger.info(
                "STRICT-PARITY: Class3D restart does not consume fresh-run "
                "run_it000 particle state",
            )

    if frozen_boundary is not None:
        frozen_boundary_cli.validate_empty_replay_slots(replay_iteration_overrides)

    effective_tau2_fudge, tau2_fudge_source = _resolve_tau2_fudge(
        args.n_classes,
        args.tau2_fudge,
        relion_init_tau2_fudge,
    )
    logger.info("Using tau2_fudge=%.3f (%s)", float(effective_tau2_fudge), tau2_fudge_source)

    t_start = time.time()

    effective_perturb_seed = _effective_perturb_seed(args)
    if frozen_boundary is not None and frozen_boundary.fixed_diagnostic_arm:
        frozen_boundary_cli.validate_fixed_boundary_runtime(
            frozen_boundary,
            args,
            dataset=ds,
            effective_max_healpix_order=initial_sampling.max_order,
            effective_tau2_fudge=effective_tau2_fudge,
            effective_perturb_seed=effective_perturb_seed,
        )
    restart_provenance = oracle_admission.resolve_restart_provenance(args, log=logger)
    logger.info(
        "SamplingPerturbation seed: %s%s",
        "unseeded" if effective_perturb_seed is None else str(effective_perturb_seed),
        " (explicit)" if args.perturb_seed is not None else " (from --seed)",
    )
    initial_poses = input_poses.prepare_initial_poses(
        our_particles,
        particle_layout=particle_layout,
        relion_halfset_particles=relion_particles,
        pixel_size_angstrom=ds.voxel_size,
        data_dir=args.data_dir,
        requested_source=args.initial_pose_source,
        n_classes=args.n_classes,
        init_relion_iteration=args.init_relion_iteration,
        has_relion_half_sets=args.relion_half_sets is not None,
        diagnostic_single_half=args.diagnostic_single_half,
        frozen_boundary=frozen_boundary,
        poses_npz_path=args.init_previous_best_poses_npz,
        pose_iteration=args.init_previous_best_poses_iter,
        has_replay_pose_source=(
            args.relion_init_dir is not None or args.perturb_replay_relion_dir is not None
        ),
        class3d_translations=kclass_firstiter_translations,
        class3d_translation_path=kclass_firstiter_translation_path,
        local_search_at_start=resume_snapshot is None
        and int(args.init_relion_iteration) == 0
        and (args.sigma_ang is not None or (int(args.n_classes) == 1 and args.healpix_order >= args.auto_local_healpix_order)),
        skip_align=bool(args.skip_align),
        log=logger,
    )

    state_swap_probe = build_state_swap_probe(
        target_relion_iteration=args.state_swap_target_relion_iteration,
        variant=args.state_swap_variant,
        replay_relion_references=args.state_swap_replay_relion_references,
        init_relion_iteration=args.init_relion_iteration,
        max_iter=args.max_iter,
        replay_iteration_overrides=replay_iteration_overrides,
    )
    if state_swap_probe is not None:
        logger.warning(
            "State-swap diagnostic: physical RELION target=%d, "
            "zero-based RECOVAR loop index=%d, variant=%s, RELION references=%s",
            int(state_swap_probe["target_relion_iteration"]),
            int(state_swap_probe["iteration"]),
            state_swap_probe["variant"],
            bool(state_swap_probe["replay_relion_references"]),
        )

    run_file_writer = None
    if int(args.write_iteration_every) > 0:
        run_file_writer = RunFileWriter(
            args.output,
            settings=RunSettings(
                output_root=os.path.join(args.output, "run"),
                random_seed=int(args.seed),
                nr_iter=int(args.max_iter),
                particle_diameter=float(particle_diameter_ang or 0.0),
                # RELION writes its live values (ml_optimiser.cpp:1660-1665): the run's mask edge, and the
                # Refine3D job's --low_resol_join_halves 40 (pipeline_jobs.cpp:4509) or, for Class3D, the
                # binary's default -1 (ml_optimiser.cpp:895).
                width_mask_edge=int(width_mask_edge_px),
                low_resol_join_halves=RunSettings.low_resol_join_halves if int(args.n_classes) == 1 else -1.0,
                adaptive_oversampling=int(args.adaptive_oversampling),
                auto_local_healpix_order=int(args.auto_local_healpix_order),
                strict_highres_exp=-1.0 if args.strict_highres_exp is None else float(args.strict_highres_exp),
                max_significants=int(args.max_significants),
                symmetry=symmetry,
                healpix_order_original=int(initial_sampling.coarse_order),
                offset_range_original_angstrom=float(args.offset_range) * float(ds.voxel_size),
                offset_step_original_angstrom=float(args.offset_step) * float(ds.voxel_size),
                perturbation_factor=float(args.perturb_factor),
                do_solvent_fsc=bool(args.solvent_correct_fsc),
                solvent_mask_name="None" if args.solvent_mask is None else str(args.solvent_mask),
                command_line=" ".join(sys.orig_argv),
                mode=args.mode,
            ),
            input_star=os.path.join(args.data_dir, "particles.star"),
            group_names=prepared_particle_groups.group_names,
            half_rows=[particle_layout.half1_rows, particle_layout.half2_rows],
            write_every=int(args.write_iteration_every),
            # RELION writes run_itNNN_half{1,2}_class001_unfil.mrc when it corrects the FSC.
            write_unfiltered_maps=bool(args.write_unfiltered_half_maps or args.solvent_correct_fsc),
            keep_iterations=int(args.keep_iterations),
        )
    continued_iterations = 0 if resume_snapshot is None else int(resume_snapshot.relion_iteration)

    # The start-up tau2 is a box-scale volume (float64 at start-up: 4.1 GB at box 800).
    # Hand it over as a host array and drop the device copy, so this frame does not
    # hold it on the device for the whole refinement (census, bigbox 14468686). The
    # host array and init_vol_ft stay alive until the refinement returns: this frame
    # names them and CPython keeps a call's arguments referenced for its duration.
    initial_mean_variance_host = np.asarray(jax.device_get(mean_variance))
    del mean_variance
    run_options = RefinementOptions(
        symmetry=SymmetryOptions(point_group=symmetry),
        schedule=command_options.resolve_schedule(
            args,
            initial_sampling=initial_sampling,
            init_current_size=init_current_size,
            ini_high_angstrom=_ini_high_for_lowpass,
            init_data_vs_prior=relion_start_data_vs_prior,
            image_mask=(particle_diameter_ang, width_mask_edge_px),
            relion_init_sigma_offset_angstrom=relion_init_sigma_offset_angstrom,
            frozen_boundary=frozen_boundary,
            continued_iterations=None if resume_snapshot is None else continued_iterations,
        ),
        batching=command_options.resolve_batching(args),
        overlap=command_options.resolve_overlap(args),
        adaptive=adaptive_options,
        parity=RelionParityOptions(
            tau2_fudge=effective_tau2_fudge,
            perturb_factor=args.perturb_factor,
            perturb_seed=effective_perturb_seed,
            optimizer_random_seed=args.seed,
            relion_optics_image_sizes=half_sets.optics_image_sizes,
            relion_optics_pixel_sizes=half_sets.optics_pixel_sizes,
            optics_group_ids_per_half=optics_group_ids_per_half,
            relion_model_pixel_size=relion_model_pixel_size,
            image_fourier_backend=args.image_fourier_backend,
            emulate_relion_firstiter_cc=bool(args.firstiter_cc),
            relion_firstiter_ini_high_angstrom=(
                relion_firstiter_ini_high_angstrom if args.firstiter_cc else None
            ),
            use_per_half_mean_variance=(
                frozen_boundary is not None and frozen_boundary.fixed_diagnostic_arm
            ),
            preserve_bpref_particle_order=use_fresh_auto_refine_order,
            firstiter_cc_tree_rescore_max_margin=firstiter_cc_tree_rescore_max_margin,
        ),
        consistency=consistency_options,
        solvent=SolventOptions(mask_path=args.solvent_mask, correct_fsc=bool(args.solvent_correct_fsc)),
        local_search=command_options.resolve_local_search(args),
        k_class=command_options.resolve_k_class(
            args, trial_order=particle_layout.accuracy_trial_order_local, resumed=resume_snapshot is not None
        ),
        checkpoint=CheckpointOptions(writer=run_file_writer, resume=resume_snapshot),
        replay=ReplayState(
            init_reference_real=None if resume_snapshot is not None else references.real_for_projector,
            init_group_ids=list(particle_groups.group_ids_per_half),
            init_group_count=particle_groups.n_groups,
            init_relion_optics_group_count=particle_groups.n_optics_groups,
            init_previous_best_translations=(
                None
                if initial_poses.poses is None
                else initial_poses.poses["previous_best_translations"]
            ),
            init_previous_best_rotation_eulers=(
                None
                if initial_poses.poses is None
                else initial_poses.poses["previous_best_rotation_eulers"]
            ),
            init_angle_priors=None if initial_poses.poses is None else initial_poses.poses.get("angle_priors"),
            init_image_corrections=(
                initial_poses.image_corrections if frozen_boundary is None else frozen_boundary.image_corrections
            ),
            init_scale_corrections=(
                initial_poses.scale_corrections if frozen_boundary is None else frozen_boundary.scale_corrections
            ),
            init_direction_prior=(
                None if frozen_boundary is None else frozen_boundary.direction_prior_per_half
            ),
            preserve_initial_direction_prior=frozen_boundary is not None,
        ),
        expected_accuracy=ExpectedAccuracyOptions(
            half1_base_order_local=particle_layout.accuracy_base_order_local,
            half1_trial_order_local=particle_layout.accuracy_trial_order_local,
            half1_optics_group_ids=particle_layout.accuracy_optics_group_ids,
            half1_particle_ids=particle_layout.accuracy_particle_ids,
            half1_ctf_params=half_sets.accuracy_ctf_params,
            do_ctf_correction=expected_accuracy_do_ctf_correction,
        ),
    )
    # One stable-window class history for the refinement (see refine_single_volume).
    from relax.sparse_pass2.resident_pass2 import stable_window_class_history

    with stable_window_class_history():
        result = refine_single_volume(
            experiment_datasets=experiment_datasets,
            init_volume=references.fourier,
            init_noise_variance=(
                noise_variance if optics_group_ids_per_half is None else [noise_variance, noise_variance]
            ),
            init_mean_variance=initial_mean_variance_host,
            translations=translations_jnp,
            options=run_options,
            observer=observers.command_observer(args),
            source=RelionReplaySource.for_run(
                RelionReplay(
                    perturb_replay_relion_dir=args.perturb_replay_relion_dir,
                    perturb_replay_restart_state_iterations=restart_provenance.iterations,
                    replay_iteration_overrides=replay_iteration_overrides,
                    final_replay_override=final_replay.override,
                    final_replay_reference_maps=final_replay.reference_maps,
                    final_replay_source_iteration=final_replay.source_iteration,
                    final_sampling_replay_relion_dir=final_replay.sampling_dir,
                    state_swap_probe=state_swap_probe,
                    follower_topology=follower_topology,
                    **oracle_admission.frozen_boundary_replay(frozen_boundary),
                ),
                run_options,
            ),
        )
    # The options (with their replay slots and start-up arrays) live no longer than the refinement, as before.
    del run_options

    if run_file_writer is not None:
        run_file_writer.wait()  # the last iteration's files, written in the background
    validate_state_swap_probe_application(
        state_swap_probe,
        result.history.state_swap_probe_applied_relion_iterations,
    )

    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Refinement complete in %.1fs (%d iterations)", total_time, args.max_iter)
    logger.info("=" * 70)

    report = RunReport(
        data_dir=str(Path(args.data_dir).resolve()),
        output_dir=str(Path(args.output).resolve()),
        timing_dir=timing_dir_path,
        total_time_s=total_time,
        n_images=n_images,
        image_shape=tuple(ds.image_shape),
        volume_shape=tuple(ds.volume_shape),
        voxel_size=ds.voxel_size,
        healpix_order=args.healpix_order,
        auto_local_healpix_order=args.auto_local_healpix_order,
        sigma_ang=args.sigma_ang,
        adaptive_oversampling=args.adaptive_oversampling,
        max_significants=args.max_significants,
        max_significants_resolution=runtime_controls.max_significants_resolution,
        restart=restart_provenance,
        max_iter=args.max_iter,
        random_seed=args.seed,
        random_seed_source=optimizer_seed_source,
        n_rotations=n_rotations,
        n_translations=translations.shape[0],
        initial_sampling=initial_sampling,
        frozen_boundary=frozen_boundary,
        symmetry_provenance=symmetry_provenance,
        initial_pose_source=initial_poses.provenance,
        diagnostic_single_half=args.diagnostic_single_half,
        state_swap_probe=state_swap_probe,
    )
    if result.profile_only:
        write_profile_only_summary(result, report, benchmark_ledger_json=args.benchmark_ledger_json)
        return

    # ---- Save results ----
    save_dict = build_archive_metadata(
        result,
        args=args,
        dataset=ds,
        effective_tau2_fudge=effective_tau2_fudge,
        follower_replay=follower_topology.replay,
        frozen_boundary=frozen_boundary,
        initial_sampling=initial_sampling,
        initial_pose_source=initial_poses.provenance,
        max_significants_resolution=runtime_controls.max_significants_resolution,
        n_images=n_images,
        n_rotations=n_rotations,
        n_translations=translations.shape[0],
        optimizer_seed_source=optimizer_seed_source,
        particle_diameter_ang=particle_diameter_ang,
        particle_layout=particle_layout,
        restart=restart_provenance,
        relion_dispatch_schedule=relion_dispatch_schedule,
        state_swap_probe=state_swap_probe,
        symmetry_provenance=symmetry_provenance,
        tau2_fudge_source=tau2_fudge_source,
        total_time=total_time,
        use_fresh_auto_refine_order=use_fresh_auto_refine_order,
    )

    git_provenance = write_refinement_archive(
        result,
        out_path=os.path.join(args.output, "refinement_results.npz"),
        metadata=save_dict,
        half_indices=(particle_layout.half1_rows, particle_layout.half2_rows),
        n_images=n_images,
        skip_large_outputs=args.skip_large_outputs,
    )

    if args.benchmark_ledger_json:
        write_benchmark_ledger(args.benchmark_ledger_json, result, report, git_provenance)

    write_final_maps(
        result.maps,
        output_dir=args.output,
        volume_shape=ds.volume_shape,
        pixel_size_angstrom=ds.voxel_size,
        n_classes=args.n_classes,
        skip_large_outputs=args.skip_large_outputs,
    )

    print_refinement_summary(result, total_time_s=total_time, box_size=ds.image_shape[0], pixel_size=ds.voxel_size)


def run_from_command_line(command):
    """Run ``relax <command>`` (``refine`` or ``class3d``) on ``sys.argv``.

    A requested diagnostic dump that stops the run early exits with status 0.
    """

    try:
        main(command=command)
    except RuntimeError as exc:
        if (
            exc.__class__.__name__ == "SignificanceDumpComplete"
            and os.environ.get("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET") == "1"
        ):
            logger.info(
                "RECOVAR coarse-significance dump completed; stopping before "
                "pass-2/M-step work: %s",
                exc,
            )
            sys.exit(0)
        raise
