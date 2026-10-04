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
import importlib
import json
import logging
import os
import platform
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
import jaxlib
import numpy as np
from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.utils.file_hash import sha256_file as _sha256_file

import relax
from relax.diagnostics import frozen_boundary_cli, initial_model_replay, parity_dump, relion_replay
from relax.diagnostics.parity_provenance import git_head_or_none
from relax.diagnostics.state_swap_probe import (
    build_state_swap_probe,
    state_swap_probe_loop_index,
    validate_state_swap_probe_application,
)
from relax.helpers import iteration_history
from relax.helpers.compilation_cache import activate_recovar_compilation_cache
from relax.helpers.dtype_policy import use_float32_matmuls
from relax.refinement import command_options, particle_loading, startup_noise
from relax.refinement.noise_updates import noise_pixel_rows
from relax.refinement.refinement_options import apply_k1_refine3d_env_defaults
from relax.refinement.result_files import (
    build_archive_metadata,
    profile_rows_for_json,
    write_final_maps,
    write_refinement_archive,
)
from relax.refinement.run_files import RunFileWriter, RunSettings, read_run_files
from relax.relion import input_particle_table, input_poses, relion_metadata
from relax.relion.geometry import (
    REFERENCE_FILTER_EDGE_SHELLS,
)
from relax.relion.input_particle_table import relion_class3d_seed_classes
from relax.relion.relion_worker_scale import prepare_follower_topology

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


def _shell_index_to_resolution_angstrom(shell_index, grid_size, voxel_size):
    if voxel_size <= 0:
        return float(shell_index)
    shell_index = float(shell_index)
    if shell_index <= 0:
        return float("inf")
    return float(grid_size) * float(voxel_size) / shell_index


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
    if replay_noise_semantics != "uninterrupted":
        raise ValueError(f"unknown replay noise semantics: {replay_noise_semantics!r}")
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
    *,
    replay_prefix="run",
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
        / f"{replay_prefix}_it{int(init_relion_iteration):03d}_sampling.star"
    )
    saved_order = int(relion_metadata.read_relion_sampling_metadata(sampling_star)["healpix_order"])
    if saved_order != int(healpix_order):
        raise ValueError(
            f"--healpix_order {int(healpix_order)} does not match RELION's saved sampling "
            f"order {saved_order} in {sampling_star}; a replay after RELION iteration "
            f"{int(init_relion_iteration)} sizes pass-1 images from that order"
        )


def _relion_start_tau2_and_data_vs_prior(
    reference_real,
    initial_noise_radial,
    *,
    grid_size: int,
    volume_shape,
    tau2_fudge: float,
    nr_particles: int,
    pdf_class: float = 1.0,
    shell_pair_counting: str = "relion",
):
    """RELION's start-up tau2 (RECOVAR units) and data_vs_prior (RELION units) of one class.

    ``MlModel::initialiseDataVersusPrior`` (ml_model.cpp:1557) on the start-up
    reference after ``initialLowPassFilterReferences``, with the initial noise
    (averaged over optics groups when it has one row per group) and the particle count: each
    auto-refine half model counts its own particles (K=1); a Class3D model counts all of them,
    with the class's start-up ``pdf_class`` (1/K). ``reference_real`` is in RECOVAR's frame and
    ``initial_noise_radial`` is RELION sigma2 times ``grid_size**4``.
    """

    from recovar.utils.helpers import recovar_volume_to_relion

    from relax.relion.reference_initialization import relion_initial_tau2_and_data_vs_prior

    n4 = float(grid_size) ** 4
    sigma2 = np.asarray(initial_noise_radial, dtype=np.float64)
    if sigma2.ndim == 2:
        # The unweighted mean over optics groups that have noise (ml_model.cpp:1560-1573).
        sigma2 = np.mean(sigma2[np.sum(sigma2, axis=1) > 0.0], axis=0)
    sigma2 = sigma2.reshape(-1) / n4
    n_shells = int(grid_size) // 2 + 1
    if sigma2.size < n_shells:
        raise ValueError(f"initial noise spectrum has {sigma2.size} shells, need {n_shells}")
    tau2, data_vs_prior = relion_initial_tau2_and_data_vs_prior(
        recovar_volume_to_relion(np.asarray(reference_real, dtype=np.float64)),
        tau2_fudge=float(tau2_fudge),
        avg_sigma2_noise=sigma2[:n_shells],
        nr_particles=int(nr_particles),
        pdf_class=float(pdf_class),
        shell_pair_counting=shell_pair_counting,
    )
    mean_variance = jnp.asarray(
        utils.make_radial_image(tau2 * n4, volume_shape, extend_last_frequency=True)
    ).reshape(-1)
    return mean_variance, data_vs_prior


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


def _refine_sampling_kwargs(args, init_healpix_order):
    """Return sampling kwargs forwarded from the CLI into ``refine_single_volume``."""
    return {
        "init_healpix_order": init_healpix_order,
        "auto_local_healpix_order": args.auto_local_healpix_order,
        "init_translation_range": args.offset_range,
        "init_translation_step": args.offset_step,
    }


_FINAL_REPLAY_GROUP_KEYS = {
    "poses": {
        "previous_best_translations",
        "previous_best_rotations",
        "previous_best_rotation_eulers",
    },
    "sampling": {
        "translation_sigma_angstrom",
        "translation_sigma_angstrom_per_half",
    },
    "corrections": {
        "noise_variance",
        "direction_prior",
        "image_corrections",
        "serialized_scale_corrections",
    },
    "noise": {"noise_variance"},
    "direction_prior": {"direction_prior"},
    # The serialized image correction is norm-factor * source-scale.  Pair it
    # with that source scale so the replay layer preserves only the RELION
    # norm factor on the resident scoring scale.
    "norm_factor": {"image_corrections", "serialized_scale_corrections"},
    # An explicit live scoring-scale oracle.  This is intentionally excluded
    # from ``all`` because a generic leader model STAR need not represent the
    # scale resident on every follower rank.
    "scoring_scale": {"scoring_scale_corrections"},
    "references": set(),
}
_FINAL_REPLAY_ALL_GROUPS = {"poses", "sampling", "corrections", "references"}


def _select_final_replay_override(source_override, requested_fields):
    """Select diagnostic final-boundary groups without touching numbered state."""
    requested_groups = {
        token.strip().lower()
        for token in str(requested_fields).split(",")
        if token.strip()
    }
    valid_groups = set(_FINAL_REPLAY_GROUP_KEYS) | {"all"}
    unknown_groups = sorted(requested_groups - valid_groups)
    if not requested_groups or unknown_groups:
        raise ValueError(
            "--final-replay-fields requires one or more of "
            "poses,sampling,corrections,noise,direction_prior,norm_factor,"
            "scoring_scale,references,all; "
            f"unknown={unknown_groups}"
        )
    if "all" in requested_groups:
        requested_groups = (requested_groups - {"all"}) | set(_FINAL_REPLAY_ALL_GROUPS)
    selected_keys = set().union(*(_FINAL_REPLAY_GROUP_KEYS[group] for group in requested_groups))
    selected_override = {
        key: value for key, value in source_override.items() if key in selected_keys
    }
    return requested_groups, selected_override


def _relion_optimiser_star_for_runtime(
    args,
    *,
    frozen_boundary=None,
    fixed_diagnostic_source_paths=None,
):
    """Use only the sealed completed optimiser for the fixed schema-v3 arm."""

    if frozen_boundary is not None and frozen_boundary.fixed_diagnostic_arm:
        if fixed_diagnostic_source_paths is None:
            raise ValueError("fixed diagnostic arm lacks sealed source paths")
        return Path(fixed_diagnostic_source_paths["completed_optimiser"]).resolve()
    return command_options.find_relion_optimiser_star(args)


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


def _explicit_relion_optimiser_for_seed(args):
    """Return a seed source only when the user explicitly selected RELION state.

    ``command_options.find_relion_optimiser_star`` also performs convenient incidental
    discovery under ``data_dir``.  That discovery is useful for masks and
    support caps, but must not unexpectedly change a standalone run's RNG.
    """
    explicit_relion_state = any(
        getattr(args, name, None)
        for name in ("relion_optimiser", "relion_init_dir", "perturb_replay_relion_dir")
    )
    return command_options.find_relion_optimiser_star(args) if explicit_relion_state else None


def _effective_perturb_seed(args):
    """Resolve the SamplingPerturbation seed used by the refinement loop.

    RELION uses the optimiser ``--random_seed`` for SamplingPerturbation, so
    the CLI-level ``--seed`` must drive the perturbation stream too. An explicit
    ``--perturb_seed`` overrides it; a negative explicit value keeps the legacy
    non-deterministic NumPy perturbation path for diagnostics.
    """
    explicit = getattr(args, "perturb_seed", None)
    if explicit is not None:
        return None if int(explicit) < 0 else int(explicit)
    seed = getattr(args, "seed", None)
    return None if seed is None else int(seed)


def _write_relion_start_particle_table(our_star, input_star, *, seed, output_dir) -> Path:
    """Write RELION's start-up particle table rebuilt from the input STAR and return its path."""
    from relax.relion.input_particle_table import write_relion_start_particle_star

    if not isinstance(our_star, dict) or "optics" not in our_star:
        raise SystemExit("--relion-half-sets-from-input needs an optics table in <data_dir>/particles.star")
    path = Path(output_dir) / "relion_input_state" / "particles_relion_start.star"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_relion_start_particle_star(input_star, path, seed=int(seed))
    return path


def _initial_current_size(voxel_size: float, grid_size: int, init_resolution: float) -> int:
    """Twice RELION's --ini_high pixel, ``getPixelFromResolution(1 / ini_high)`` (ml_model.h:441,
    ml_optimiser.cpp:2801): ``2 ROUND(ori_size pixel_size / ini_high)``. The first E-step then adds
    ``incr_size`` shells (``_bootstrap_current_size_relion``). No floor: RELION has none, and the GUI
    default ini_high of 60 A lands below the former 32-pixel floor on small boxes.
    """

    return 2 * int(np.floor(float(grid_size) * float(voxel_size) / float(init_resolution) + 0.5))


def _require_relion_convention_reference(path, option: str) -> None:
    """Stop the run when a reference map's RELION convention is not established (relax.helpers.map_io)."""
    from relax.helpers.map_io import require_relion_convention_reference

    try:
        require_relion_convention_reference(path, option=option)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def main(command=None):
    """Run a refinement from ``sys.argv``; ``command`` (``refine``/``class3d``) checks ``--n_classes``."""

    # This module imports JAX before recovar, so recovar's cache environment must be applied to the live config.
    cache_directory = activate_recovar_compilation_cache()
    use_float32_matmuls()
    _assert_expected_repo_imports()
    args = command_options.parse_refinement_args()
    if command is not None:
        command_options.require_command_n_classes(command, int(args.n_classes))
    consistency_options = command_options.resolve_consistency_options(args)
    if cache_directory:
        logger.info("Persistent JAX compilation cache: %s", cache_directory)
    if int(args.n_classes) == 1:
        apply_k1_refine3d_env_defaults()
    command_options.resolve_job_defaults(args)
    command_options.resolve_standalone_k1_start(args)
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

    if args.continue_optimiser_star is not None:
        args.seed = command_options.validate_continue_args(args)

    seed_optimiser_star = (
        _relion_optimiser_star_for_runtime(
            args,
            frozen_boundary=frozen_boundary,
            fixed_diagnostic_source_paths=fixed_diagnostic_source_paths,
        )
        if frozen_boundary is not None and frozen_boundary.fixed_diagnostic_arm
        else _explicit_relion_optimiser_for_seed(args)
    )
    args.seed, optimizer_seed_source = _resolve_optimizer_random_seed(args.seed, seed_optimiser_star)
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
    from relax.helpers import xla_memory_reserve
    from relax.relion.geometry import PROJECTION_PADDING_FACTOR

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
    tomo_run = particle_inputs.tomographic
    shape_class_rows = particle_inputs.shape_class_rows
    relion_mask_params = particle_inputs.mask_parameters
    _double_image_preprocessing = particle_inputs.double_preprocessing
    del particle_inputs
    args._relion_mask_params = relion_mask_params
    particle_diameter_ang = None if relion_mask_params is None else float(relion_mask_params[0])
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
    relion_optics_image_sizes = None
    relion_optics_pixel_sizes = None
    relion_model_pixel_size = None
    expected_accuracy_half1_ctf_params = None
    use_relion_live_initial_noise = _k1_relion_live_initial_noise_enabled()
    # A run that loads no noise state starts from RELION's estimate from the images.
    relion_startup_noise_needed = (
        frozen_boundary is None and args.init_noise_from_npz is None and args.relion_init_dir is None
    )
    relion_fresh_initial_noise_source_rows = None
    relion_fresh_initial_noise_optics_group_ids = None
    class3d_noise_optics_pixel_sizes = None
    relion_particles = None
    use_fresh_auto_refine_order = False

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

    if args.relion_half_sets is not None:
        # Use RELION's half-set split from rlnRandomSubset
        logger.info("Loading RELION half-set assignments from %s", args.relion_half_sets)
        halfset_source = input_particle_table.read_relion_halfset_source(
            args.relion_half_sets,
            n_classes=args.n_classes,
            log=logger,
        )
        relion_particles = halfset_source.tables["particles"]
        relion_optics_image_sizes = halfset_source.optics_image_sizes
        relion_optics_pixel_sizes = halfset_source.optics_pixel_sizes
        use_fresh_auto_refine_order = _use_fresh_auto_refine_particle_order(
            args,
            frozen_boundary,
        )
        halfset_inputs = particle_loading.prepare_relion_halfset_inputs(
            our_particles,
            relion_particles,
            source_path=args.relion_half_sets,
            random_seed=args.seed if use_fresh_auto_refine_order else None,
            prepare_noise_order=(
                args.n_classes == 1
                and (relion_startup_noise_needed or use_relion_live_initial_noise)
            ),
            tomographic=tomo_run,
            image_grid_size=None if tomo_run else ds.grid_size,
            log=logger,
        )
        particle_layout = halfset_inputs.layout
        relion_fresh_initial_noise_source_rows = halfset_inputs.noise_source_rows
        relion_fresh_initial_noise_optics_group_ids = halfset_inputs.noise_optics_group_ids
        expected_accuracy_half1_ctf_params = halfset_inputs.accuracy_ctf_params
        del halfset_inputs
        if use_fresh_auto_refine_order:
            logger.info(
                "Applied RELION fresh paired AutoRefine particle order (mt19937) with effective seed %d; "
                "BPref will preserve this physical order",
                int(args.seed) + 1,
            )
    elif args.n_classes == 1:
        raise SystemExit(
            "K=1 auto-refine uses RELION's half sets: a fresh start rebuilds them from the "
            "input STAR (--relion-half-sets-from-input, the default); a RELION-seeded, "
            "replayed or frozen start needs --relion_half_sets"
        )
    else:
        particle_layout = input_particle_table.prepare_class3d_particle_layout(
            our_particles,
            n_particles=n_images,
            random_seed=args.seed,
            init_relion_iteration=args.init_relion_iteration,
        )
    if args.n_classes > 1 and relion_startup_noise_needed:
        (
            relion_fresh_initial_noise_source_rows,
            relion_fresh_initial_noise_optics_group_ids,
        ) = startup_noise.class3d_noise_order(our_particles)
        if not isinstance(our_star, dict) or "optics" not in our_star:
            raise SystemExit("Class3D RELION start-up noise needs an optics table in the particle STAR")
        class3d_noise_optics_pixel_sizes = np.asarray(
            our_star["optics"]["rlnImagePixelSize"],
            dtype=np.float64,
        )

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
            raise SystemExit("--continue does not support particle STARs with several image shapes yet")
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
    strict_relion_scale_context = bool(
        args.n_classes > 1
        and (args.perturb_replay_relion_dir is not None or args.relion_init_dir is not None)
    )
    dispatch = command_options.load_verified_dispatch_schedule(
        args,
        group_particle_source.particles,
        strict_replay=strict_relion_scale_context,
    )
    relion_dispatch_schedule = dispatch.schedule
    follower_topology = prepare_follower_topology(
        args.relion_scale_followers,
        relion_dispatch_schedule,
        particle_groups,
        strict_replay=strict_relion_scale_context,
        replay_path=args.relion_follower_scale_replay,
        oracle_dir=dispatch.oracle_dirs[0] if relion_dispatch_schedule is not None else None,
        random_seed=args.seed,
        init_relion_iteration=args.init_relion_iteration,
        max_iter=args.max_iter,
        group_source=group_particle_source.path,
        logger=logger,
    )

    optimiser_star = _relion_optimiser_star_for_runtime(
        args,
        frozen_boundary=frozen_boundary,
        fixed_diagnostic_source_paths=fixed_diagnostic_source_paths,
    )
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
    # References are RELION-convention maps, the same files relion_refine reads with --ref;
    # load_relion_volume puts them in the internal frame (relax.helpers.map_io), and get_dft3
    # gives the centered Fourier volume.
    # NEVER use raw `mrcfile.open` + `np.fft.fftn(np.fft.ifftshift(...))` here:
    # that produces a Fourier volume with the right values but at WRONG array
    # indices (DC at corner instead of center), so `slice_volume` reads
    # Nyquist as if it were DC and projections are off by ~2400x in amplitude
    # at low frequencies.
    from recovar.utils.helpers import load_relion_volume

    # RELION's Image<RFLOAT>::read() widens a reference MRC (on-disk float32)
    # to RFLOAT (double, in our ACC_DOUBLE_PRECISION oracle build) as part of
    # the read itself (src/ml_model.cpp:MlModel::readImages -> Iref.push_back
    # (img()), where Iref is std::vector<MultidimArray<RFLOAT>>). Every
    # downstream step -- including the FFT that builds Projector::data
    # (MultidimArray<Complex>, Complex = tComplex<RFLOAT>) -- then runs at
    # that same double precision. Match that here instead of narrowing to
    # float32/complex64 immediately after the (inherently float32-on-disk)
    # MRC read, which would otherwise defeat RELAX_USE_FLOAT64_PROJECTIONS
    # no matter how carefully every later cast is fixed (a "narrow-then-
    # widen" bug: DensePrecisionPolicy.cast_projection_volume, gated on this
    # same flag, cannot recover precision already lost here). Gated on
    # RELAX_USE_FLOAT64_PROJECTIONS specifically (not also
    # _SCORING/_dense_global_scoring_dtype's OR) to match
    # projection_complex_dtype's own condition exactly and avoid forcing an
    # unrequested precision/memory cost on the projection path when a
    # caller wants float64 scoring without float64 projections.
    _init_volume_use_float64 = bool(
        os.environ.get("RELAX_USE_FLOAT64_PROJECTIONS", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    _init_volume_dtype = np.float64 if _init_volume_use_float64 else np.float32
    _init_volume_complex_dtype = np.complex128 if _init_volume_use_float64 else np.complex64

    # RELION's ``initialLowPassFilterReferences`` (ml_optimiser.cpp:3556) low-
    # pass-filters mymodel.Iref in place at startup, gated only on
    # ``ini_high > 0`` (not on ``--firstiter_cc``). With ``--apply-initial-
    # lowpass``, mirror that behavior: apply LP at ``--init_resolution`` to
    # the reference before iter-1 expectation. The Fourier mask edge is

    def _apply_ini_high_lowpass_real(volume_real, volume_shape, voxel_size, ini_high):
        from relax.relion.reference_initialization import initial_low_pass_filter_references

        filtered = initial_low_pass_filter_references(
            np.asarray(volume_real, dtype=np.float64)[None, ...],
            ori_size=int(volume_shape[0]),
            pixel_size=float(voxel_size),
            ini_high_ang=float(ini_high),
            filter_edgewidth=float(REFERENCE_FILTER_EDGE_SHELLS),
        )[0]
        return np.asarray(filtered, dtype=np.float64)

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
    init_reference_real_for_projector = None
    relion_start_reference_real = None
    relion_start_class_references_real = None

    if frozen_boundary is not None:
        if frozen_boundary.volume_shape != tuple(int(value) for value in ds.volume_shape):
            raise SystemExit(
                "Frozen-boundary volume_shape does not match the active dataset: "
                f"boundary={frozen_boundary.volume_shape}, dataset={tuple(ds.volume_shape)}"
            )
        init_vol_ft = np.stack(frozen_boundary.means, axis=0)
        merged_init_ft = np.mean(init_vol_ft.astype(np.complex128), axis=0).astype(_init_volume_complex_dtype)
        init_vol_real = np.asarray(
            ftu.get_idft3(jnp.asarray(merged_init_ft).reshape(ds.volume_shape)).real,
            dtype=_init_volume_dtype,
        )
        logger.info(
            "Initial per-half Fourier volumes loaded from frozen boundary %s",
            frozen_boundary.source_dir,
        )
    elif args.n_classes == 1:
        init_mrc_path = args.init_volume or os.path.join(args.data_dir, "reference_init_relion.mrc")
        _require_relion_convention_reference(init_mrc_path, "--init_volume")
        init_vol_real = load_relion_volume(init_mrc_path).astype(_init_volume_dtype)
        relion_model_pixel_size = relion_metadata._read_relion_mrc_model_pixel_size(init_mrc_path)
        if not np.isfinite(relion_model_pixel_size) or relion_model_pixel_size <= 0.0:
            raise SystemExit(
                f"Initial RELION reference has invalid voxel size {relion_model_pixel_size}: "
                f"{init_mrc_path}"
            )
        assert init_vol_real.shape == ds.volume_shape, (
            f"Volume shape mismatch: {init_vol_real.shape} vs {ds.volume_shape}"
        )
        if _ini_high_for_lowpass is not None:
            # RELION filters ``mymodel.Iref`` in model coordinates.  The
            # particle STAR optics pixel size can be a rounded serialization
            # (for example 1.416667 versus the MRC header ratio
            # 544.0 / 384 = 1.4166666666666667 A/px),
            # which is enough to flip marginal firstiter-CC winners.
            filtered_real = _apply_ini_high_lowpass_real(
                init_vol_real,
                ds.volume_shape,
                relion_model_pixel_size,
                _ini_high_for_lowpass,
            )
            if _use_initial_projector_real:
                init_reference_real_for_projector = filtered_real
            relion_start_reference_real = filtered_real
            init_vol_real = filtered_real.astype(_init_volume_dtype, copy=False)
            logger.info(
                "Applied RELION initialLowPassFilterReferences to init reference: ini_high=%.2f A, fmask_edge=%d shells",
                _ini_high_for_lowpass, REFERENCE_FILTER_EDGE_SHELLS,
            )
        else:
            relion_start_reference_real = np.asarray(init_vol_real, dtype=np.float64)
            if _use_initial_projector_real:
                init_reference_real_for_projector = relion_start_reference_real
        init_vol_ft = (
            np.array(ftu.get_dft3(jnp.asarray(init_vol_real)))
            .astype(_init_volume_complex_dtype)
            .reshape(-1)
        )
        logger.info(
            "Initial volume loaded from %s: shape=%s model_pixel_size=%.9g A/px",
            init_mrc_path,
            init_vol_real.shape,
            relion_model_pixel_size,
        )
    else:
        if args.init_volume is not None:
            # relion_refine --K K with one --ref map: every class starts from it and the first iteration scores
            # each particle against one random class (do_generate_seeds, ml_model.cpp:1007-1010); with
            # --firstiter_cc that is the second iteration, after a CC iteration against class 1 alone.
            if args.ref_star is not None or args.init_class_volumes:
                raise SystemExit("--init_volume is Class3D's one reference; --ref_star and --init_class_volumes list K")
            class_paths = [args.init_volume] * int(args.n_classes)
        elif args.ref_star is not None:
            if args.init_class_volumes:
                raise SystemExit("--ref_star and --init_class_volumes are exclusive")
            class_paths, star_distribution = relion_metadata.read_relion_reference_star(args.ref_star)
            class_paths = [str(p) for p in class_paths]
            if star_distribution is not None and not np.allclose(
                star_distribution, 1.0 / len(class_paths), rtol=0.0, atol=1e-6
            ):
                logger.warning(
                    "--ref_star rlnClassDistribution %s is not read; relion_refine starts "
                    "a fresh Class3D run from 1/K",
                    star_distribution.tolist(),
                )
        elif args.init_class_volumes:
            class_paths = [p.strip() for p in args.init_class_volumes.split(",")]
        else:
            class_paths = [
                os.path.join(args.data_dir, f"reference_init_class{k + 1:03d}_relion.mrc")
                for k in range(args.n_classes)
            ]
        if len(class_paths) != args.n_classes:
            raise SystemExit(f"--init_class_volumes count {len(class_paths)} != --n_classes {args.n_classes}")
        class_option = (
            "--ref_star" if args.ref_star is not None else "--init_class_volumes" if args.init_class_volumes else "data_dir"
        )
        for p in class_paths:
            _require_relion_convention_reference(p, class_option)
        per_class_ft = []
        per_class_real_for_projector = []
        relion_start_class_references_real = []
        for k, p in enumerate(class_paths):
            vol_real = np.asarray(load_relion_volume(p)).astype(_init_volume_dtype)
            assert vol_real.shape == ds.volume_shape, (
                f"Class {k + 1} volume shape mismatch at {p}: {vol_real.shape} vs {ds.volume_shape}"
            )
            if _ini_high_for_lowpass is not None:
                filtered_real = _apply_ini_high_lowpass_real(
                    vol_real, ds.volume_shape, ds.voxel_size, _ini_high_for_lowpass,
                )
                if _use_initial_projector_real:
                    per_class_real_for_projector.append(filtered_real)
                relion_start_class_references_real.append(np.asarray(filtered_real, dtype=np.float64))
                vol_real = filtered_real.astype(_init_volume_dtype, copy=False)
            else:
                relion_start_class_references_real.append(np.asarray(vol_real, dtype=np.float64))
                if _use_initial_projector_real:
                    per_class_real_for_projector.append(np.asarray(vol_real, dtype=np.float64))
            vol_ft = np.array(ftu.get_dft3(jnp.asarray(vol_real))).astype(_init_volume_complex_dtype).reshape(-1)
            per_class_ft.append(vol_ft)
            logger.info("Class %d initial volume loaded from %s", k + 1, p)
        if _ini_high_for_lowpass is not None:
            logger.info(
                "Applied RELION initialLowPassFilterReferences to %d init references: ini_high=%.2f A, fmask_edge=%d shells",
                args.n_classes, _ini_high_for_lowpass, REFERENCE_FILTER_EDGE_SHELLS,
            )
        # Stack to (K, V); refine_single_volume._normalize_initial_means handles the
        # per-half broadcast.
        init_vol_ft = np.stack(per_class_ft, axis=0)
        if _use_initial_projector_real:
            init_reference_real_for_projector = np.stack(
                per_class_real_for_projector,
                axis=0,
            )
        # For downstream init_PS estimation, use class-1 as the representative
        # (K-class noise/prior bootstrap currently uses a single spectrum).
        init_vol_real = np.asarray(load_relion_volume(class_paths[0])).astype(_init_volume_dtype)

    # ---- Set up rotation and translation grids ----
    from relax.sampling import get_translation_grid, rotation_grid_size

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
    translations = get_translation_grid(args.offset_range, args.offset_step).astype(np.float32)
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

    from recovar.reconstruction import noise as recon_noise

    optics_group_ids_per_half = None
    if frozen_boundary is not None:
        noise_variance = frozen_boundary_cli.expand_boundary_noise(
            frozen_boundary.noise_radial_per_half,
            ds.image_shape,
        )
        initial_noise_radial = np.mean(
            np.stack(frozen_boundary.noise_radial_per_half, axis=0),
            axis=0,
        )
        logger.info(
            "Initial noise/tau2 state is owned by frozen boundary %s",
            frozen_boundary.source_dir,
        )
    elif args.init_noise_from_npz is not None:
        init_noise = iteration_history._load_init_noise_radial_npz(args.init_noise_from_npz, args.init_noise_iter)
        initial_noise_radial = init_noise["noise_radial"]
        noise_variance = recon_noise.make_radial_noise(initial_noise_radial, ds.image_shape)
        logger.info(
            "Diagnostic init: loaded sigma2_noise from %s iter=%s: min=%.3e median=%.3e max=%.3e",
            args.init_noise_from_npz,
            init_noise["iteration"],
            float(np.min(np.asarray(initial_noise_radial))),
            float(np.median(np.asarray(initial_noise_radial))),
            float(np.max(np.asarray(initial_noise_radial))),
        )
    elif args.relion_init_dir is not None:
        # The RELION-seeded debug start loads RELION's iteration-0 model noise below.
        initial_noise_radial = None
        noise_variance = None
    elif resume_snapshot is not None:
        # The run files own the noise; the loop installs each half's own spectrum.
        initial_noise_radial = np.mean(np.stack(resume_snapshot.noise_shells, axis=0), axis=0)
        noise_variance = np.asarray(noise_pixel_rows(resume_snapshot.noise_shells[0], ds.image_shape))
        if noise_variance.ndim == 2:
            from relax.helpers.optics_noise import dense_optics_groups

            image_optics_groups, _ = dense_optics_groups(our_particles["rlnOpticsGroup"])
            optics_group_ids_per_half = [image_optics_groups[particle_layout.half1_rows], image_optics_groups[particle_layout.half2_rows]]
    else:
        if args.n_classes == 1 and args.relion_half_sets is None:
            raise ValueError(
                "RELION start-up noise needs K=1 half sets (Class3D uses the input order), the "
                "particle-diameter mask and the optics pixel size, and no frozen or loaded noise"
            )
        initial_noise = startup_noise.prepare_startup_noise(
            ds,
            source_rows=relion_fresh_initial_noise_source_rows,
            optics_group_ids=relion_fresh_initial_noise_optics_group_ids,
            mask_params=relion_mask_params,
            optics_pixel_sizes=(
                relion_optics_pixel_sizes if args.n_classes == 1 else class3d_noise_optics_pixel_sizes
            ),
            output_dtype=np.float64 if _double_image_preprocessing else np.float32,
            pair_counting=consistency_options.initial_noise_pair_counting,
        )
        initial_noise_radial = initial_noise.radial
        noise_variance = initial_noise.pixel_variance
        del initial_noise
        logger.info(
            "RELION start-up noise from the images: %d shells, scoring dtype=%s",
            initial_noise_radial.size,
            noise_variance.dtype,
        )
        if noise_variance.ndim == 2:
            # One spectrum per optics group; every image scores with its own group's.
            from relax.helpers.optics_noise import dense_optics_groups

            image_optics_groups, _ = dense_optics_groups(our_particles["rlnOpticsGroup"])
            optics_group_ids_per_half = [image_optics_groups[particle_layout.half1_rows], image_optics_groups[particle_layout.half2_rows]]
            logger.info(
                "Per-optics-group noise: %d groups, images per group %s",
                noise_variance.shape[0],
                np.bincount(image_optics_groups).tolist(),
            )

    # Compute initial signal prior from init volume (weak prior). For K>1
    # use class-1 as the representative volume; the engine derives per-class
    # tau2 trajectories from the per-class FSCs once the loop starts.
    from recovar.reconstruction.regularization import average_over_shells

    if frozen_boundary is not None:
        init_PS_source = jnp.asarray(merged_init_ft)
    elif args.n_classes > 1:
        init_PS_source = jnp.asarray(per_class_ft[0])
    else:
        init_PS_source = jnp.asarray(init_vol_ft)
    init_PS = average_over_shells(jnp.abs(init_PS_source) ** 2, ds.volume_shape)
    del init_PS_source
    from recovar import utils

    init_prior = utils.make_radial_image(init_PS, ds.volume_shape, extend_last_frequency=True)
    # Scale by a factor to provide regularization without being too strong
    mean_variance = jnp.asarray(init_prior * 0.5 + jnp.max(init_prior) * 1e-4)
    del init_prior

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
        if relion_particles is None or args.relion_half_sets is None:
            invalid_reasons.append("RELION half-set data are required")
        if relion_mask_params is None:
            invalid_reasons.append("RELION particle-diameter mask parameters are required")
        if relion_fresh_initial_noise_source_rows is None:
            invalid_reasons.append("RELION initial-noise source order is required")
        if relion_fresh_initial_noise_optics_group_ids is None:
            invalid_reasons.append("RELION initial-noise optics groups are required")
        if relion_optics_pixel_sizes is None or relion_optics_pixel_sizes.size != 1:
            invalid_reasons.append("exactly one RELION optics pixel size is required")
        if invalid_reasons:
            raise ValueError(
                f"{_K1_RELION_LIVE_INITIAL_NOISE_ENV} is restricted to a strict fresh "
                f"K=1 cold start: {'; '.join(invalid_reasons)}",
            )
        relion_live_initial_sigma2_per_group = startup_noise.estimate_startup_sigma2(
            ds,
            source_rows=relion_fresh_initial_noise_source_rows,
            optics_group_ids=relion_fresh_initial_noise_optics_group_ids,
            image_pixel_size=float(relion_optics_pixel_sizes[0]),
            particle_diameter_ang=float(relion_mask_params[0]),
            width_mask_edge_px=int(relion_mask_params[1]),
        )
        if relion_live_initial_sigma2_per_group.shape[0] != 1:
            raise NotImplementedError(
                "fresh K=1 live-noise scoring currently requires one optics group",
            )
        relion_live_initial_sigma2 = relion_live_initial_sigma2_per_group[0]
        relion_live_initial_noise_variance = startup_noise.scoring_noise_from_sigma2(
            relion_live_initial_sigma2,
            grid_size=int(ds.grid_size),
            # The fresh K=1 pass scores with RELION's exact (double) BPref operands.
            output_dtype=np.float64,
        )
        logger.warning(
            "STRICT-PARITY: fresh K=1 RELION live initial noise enabled: particles=%d "
            "source_rows_head=%s sigma2_head=%s",
            min(1000, int(np.asarray(relion_fresh_initial_noise_source_rows).size)),
            np.asarray(relion_fresh_initial_noise_source_rows, dtype=np.int64)[:5].tolist(),
            np.asarray(relion_live_initial_sigma2[:5]),
        )
    if args.relion_init_dir is not None and frozen_boundary is None:
        initial_model = initial_model_replay.read_initial_model(
            args.relion_init_dir,
            n_classes=args.n_classes,
        )
        replayed_noise = initial_model_replay.prepare_noise(
            initial_model,
            grid_size=ds.grid_size,
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
            grid_size=ds.grid_size,
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
    if relion_start_reference_real is not None and fresh_relion_start:
        mean_variance, relion_start_data_vs_prior = _relion_start_tau2_and_data_vs_prior(
            relion_start_reference_real,
            initial_noise_radial,
            grid_size=int(ds.grid_size),
            volume_shape=ds.volume_shape,
            tau2_fudge=_resolve_tau2_fudge(args.n_classes, args.tau2_fudge, None)[0],
            nr_particles=int(ds_half1.n_units),
            shell_pair_counting=consistency_options.shell_pair_counting,
        )
        logger.info(
            "RELION start-up tau2/data_vs_prior (initialiseDataVersusPrior): %d shells with data_vs_prior > 3",
            int(np.sum(relion_start_data_vs_prior > 3.0)),
        )
    elif relion_start_class_references_real is not None and fresh_relion_start and resume_snapshot is None:
        # Class3D: each class's start-up data_vs_prior over all particles at pdf_class 1/K. The first
        # iteration's scale-correction sums take only the shells where it exceeds 3 (ml_optimiser.cpp:10473);
        # without it every shell entered them (subtomogram Class3D it001 group scales 5.6e-4 off RELION).
        # The class tau2 volumes are still the loop's own (only the scale gate reads this curve).
        n_classes = len(relion_start_class_references_real)
        relion_start_data_vs_prior = np.stack(
            [
                _relion_start_tau2_and_data_vs_prior(
                    reference,
                    initial_noise_radial,
                    grid_size=int(ds.grid_size),
                    volume_shape=ds.volume_shape,
                    tau2_fudge=_resolve_tau2_fudge(args.n_classes, args.tau2_fudge, None)[0],
                    nr_particles=int(ds.n_units),
                    pdf_class=1.0 / n_classes,
                    shell_pair_counting=consistency_options.shell_pair_counting,
                )[1]
                for reference in relion_start_class_references_real
            ],
            axis=0,
        )
        logger.info(
            "RELION start-up data_vs_prior per class (initialiseDataVersusPrior): %s shells with data_vs_prior > 3",
            np.sum(relion_start_data_vs_prior > 3.0, axis=1).tolist(),
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
        else _initial_current_size(ds.voxel_size, ds.grid_size, args.init_resolution)
    )
    logger.info("Initial current_size from resolution %.1f A: %d pixels", args.init_resolution, init_current_size)

    # ---- Run refinement ----
    from relax.refinement.iteration_loop import refine_single_volume
    from relax.refinement.refinement_options import (
        AdaptiveOptions,
        CheckpointOptions,
        EngineDebugOptions,
        ExpectedAccuracyOptions,
        HalfOverlapOptions,
        KClassOptions,
        LocalSearchOptions,
        RefinementBatching,
        RefinementOptions,
        RefinementSchedule,
        RelionParityOptions,
        ReplayState,
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

    # Parse oracle current_sizes if provided
    oracle_current_sizes = None
    if args.relion_current_sizes is not None:
        oracle_current_sizes = [int(x) for x in args.relion_current_sizes.split(",")]
        logger.info("Oracle mode: using RELION current_sizes=%s", oracle_current_sizes)
    oracle_healpix_orders = None
    if args.relion_healpix_orders is not None:
        oracle_healpix_orders = [int(x) for x in args.relion_healpix_orders.split(",")]
        logger.info("Oracle mode: using RELION healpix_orders=%s", oracle_healpix_orders)

    # Build per-iter replay overrides from RELION's per-iter data.star +
    # model.star when --perturb_replay_relion_dir is set. The override always
    # injects RELION's per-iter sigma_offset (parity-critical: recovar's iter-1
    # does not run the C1 sigma_offset update so iter-2 would otherwise use the
    # 10 Å default — 6× too wide vs RELION ~1.6 Å — depressing iter-2 Pmax by
    # ~22%). Per-image normCorrection / group-scale replay is part of strict
    # RELION replay and can be disabled with --no-replay_relion_normcorr for
    # diagnostics.
    replay_iteration_overrides = None
    if args.perturb_replay_relion_dir is not None:
        if frozen_boundary is not None:
            replay_iteration_overrides = frozen_boundary_cli.projector_only_replay_slots(args.max_iter)
            logger.info(
                "Diagnostic frozen restart: local replay slot 0 is projector-only; "
                "sealed per-half scoring state suppresses process-start noise broadcast"
            )
        else:
            replay_normcorr = _resolve_replay_normcorr(
                args.perturb_replay_relion_dir,
                args.replay_relion_normcorr,
            )
            replay_iteration_overrides = relion_replay._build_replay_iteration_overrides(
                args.perturb_replay_relion_dir,
                particle_layout.half1_rows,
                particle_layout.half2_rows,
                # Numbered expectation k consumes the state written before it, so
                # iterations 1..N use run_it000..run_it{N-1}.  After convergence,
                # RELION's unnumbered all-data expectation consumes the state just
                # written by iteration N and therefore needs run_it{N} as the
                # extra final-only override.
                int(args.max_iter),
                ds_voxel=ds.voxel_size,
                ds_grid=ds.grid_size,
                include_normcorr=replay_normcorr,
                init_relion_iteration=args.init_relion_iteration,
                particle_names=our_names,
                include_k1_mean_variance=(args.state_swap_target_relion_iteration is not None),
                include_k1_scoring_scale=(args.state_swap_target_relion_iteration is not None),
                strict=True,
                process_start_noise_broadcast=replay_process_start_noise_broadcast,
                noise_dtype=np.float64 if _double_image_preprocessing else np.float32,
            )
            logger.info(
                "Replay noise semantics: %s (slot-0 process-start broadcast=%s)",
                args.replay_noise_semantics,
                replay_process_start_noise_broadcast,
            )

    final_replay_override = None
    final_replay_reference_maps = None
    final_replay_source_iteration = None
    final_sampling_replay_relion_dir = None
    if args.final_replay_relion_dir is not None:
        final_replay_dir = Path(args.final_replay_relion_dir).resolve()
        complete_iterations = relion_replay._complete_relion_numbered_state_iterations(final_replay_dir)
        source_iteration = relion_replay._resolve_final_replay_source_iteration(
            configured_max_iter=args.max_iter,
            explicit_source_iteration=args.final_replay_source_iteration,
            complete_iterations=complete_iterations,
        )
        final_replay_source_iteration = source_iteration
        final_optimiser_path = final_replay_dir / "run_optimiser.star"
        final_sampling_path = final_replay_dir / "run_sampling.star"
        if not final_optimiser_path.is_file() or not final_sampling_path.is_file():
            raise ValueError(
                "diagnostic final-only substitution requires unnumbered run_optimiser.star "
                f"and run_sampling.star in {final_replay_dir}"
            )
        from relax.relion.relion_metadata import read_relion_optimiser_metadata

        final_optimiser_metadata = read_relion_optimiser_metadata(final_optimiser_path)
        if not bool(final_optimiser_metadata.get("has_converged", False)):
            raise ValueError(
                f"diagnostic final-only oracle does not report convergence: {final_optimiser_path}"
            )
        final_overrides = relion_replay._build_replay_iteration_overrides(
            final_replay_dir,
            particle_layout.half1_rows,
            particle_layout.half2_rows,
            source_iteration,
            ds_voxel=ds.voxel_size,
            ds_grid=ds.grid_size,
            include_normcorr=True,
            init_relion_iteration=args.init_relion_iteration,
            particle_names=our_names,
            strict=True,
            noise_dtype=np.float64 if _double_image_preprocessing else np.float32,
        )
        source_override = final_overrides[-1]
        if source_override is None:
            raise ValueError("diagnostic final-only substitution did not load a last-numbered override")
        # Expose the model-STAR scale as a scorer oracle only at this explicit
        # final diagnostic boundary.  Numbered replay continues to treat it as
        # serialization provenance because general MPI leader/follower layouts
        # do not guarantee that it is every scorer's resident scale.
        source_override = dict(source_override)
        serialized_scale = source_override.get("serialized_scale_corrections")
        if serialized_scale is not None:
            source_override["scoring_scale_corrections"] = serialized_scale
        requested_groups, final_replay_override = _select_final_replay_override(
            source_override,
            args.final_replay_fields,
        )
        if "references" in requested_groups:
            if args.n_classes != 1:
                raise ValueError(
                    "diagnostic final-only reference substitution currently requires --n-classes=1"
                )
            final_replay_reference_maps = relion_replay._load_final_replay_reference_maps(
                final_replay_dir,
                source_iteration,
                ds.volume_shape,
            )
        if "sampling" in requested_groups:
            final_sampling_replay_relion_dir = str(final_replay_dir)
        logger.info(
            "Diagnostic final-only substitution: source_iteration=%d groups=%s fields=%s source=%s",
            source_iteration,
            sorted(requested_groups),
            sorted(final_replay_override),
            final_replay_dir,
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
        initial_overrides = relion_replay._build_replay_iteration_overrides(
            args.relion_init_dir,
            particle_layout.half1_rows,
            particle_layout.half2_rows,
            0,
            ds_voxel=ds.voxel_size,
            ds_grid=ds.grid_size,
            include_normcorr=True,
            init_relion_iteration=0,
            particle_names=our_names,
            include_initial_state=True,
            strict=True,
            noise_dtype=np.float64 if _double_image_preprocessing else np.float32,
        )
        if initial_overrides[0] is not None:
            if args.init_noise_from_npz is not None:
                initial_overrides[0] = dict(initial_overrides[0])
                explicit_noise = (
                    list(noise_variance)
                    if isinstance(noise_variance, (list, tuple))
                    else [noise_variance, noise_variance]
                )
                initial_overrides[0]["noise_variance"] = [
                    np.asarray(value, dtype=np.float64).copy()
                    for value in explicit_noise
                ]
                logger.info(
                    "STRICT-PARITY: first expectation preserves explicit "
                    "--init-noise-from-npz instead of rounded model-STAR noise",
                )
            elif relion_live_initial_noise_variance is not None:
                initial_overrides[0] = dict(initial_overrides[0])
                initial_overrides[0]["noise_variance"] = [
                    np.asarray(relion_live_initial_noise_variance, dtype=np.float64).copy(),
                    np.asarray(relion_live_initial_noise_variance, dtype=np.float64).copy(),
                ]
                logger.info(
                    "STRICT-PARITY: first expectation consumes computed live "
                    "binary64 K=1 startup noise instead of rounded model-STAR noise",
                )
            if replay_iteration_overrides is None:
                replay_iteration_overrides = [None] * (args.max_iter + 1)
            replay_iteration_overrides[0] = initial_overrides[0]
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
            initial_overrides = relion_replay._build_replay_iteration_overrides(
                args.relion_init_dir,
                particle_layout.half1_rows,
                particle_layout.half2_rows,
                0,
                ds_voxel=ds.voxel_size,
                ds_grid=ds.grid_size,
                include_normcorr=False,
                init_relion_iteration=0,
                particle_names=our_names,
                include_initial_state=True,
                strict=True,
            )
            kclass_firstiter_translations = input_poses._kclass_firstiter_translation_seed(
                initial_overrides[0],
                n_classes=args.n_classes,
                init_relion_iteration=args.init_relion_iteration,
            )
            kclass_firstiter_translation_path = (
                Path(args.relion_init_dir).expanduser().resolve() / "run_it000_data.star"
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

    captured_projector = None
    if args.relion_projector_capture_dir is not None:
        if args.perturb_replay_relion_dir is None:
            raise SystemExit(
                "--relion-projector-capture-dir requires --perturb_replay_relion_dir"
            )
        if args.relion_projector_capture_iteration is None:
            raise SystemExit(
                "--relion-projector-capture-dir requires "
                "--relion-projector-capture-iteration"
            )
        capture_dir = Path(args.relion_projector_capture_dir).expanduser().resolve()
        capture_manifest = (
            Path(args.relion_projector_capture_manifest).expanduser().resolve()
            if args.relion_projector_capture_manifest is not None
            else capture_dir
            / f"iter{int(args.relion_projector_capture_iteration)}_VALIDATED_SHA256SUMS"
        )
        try:
            captured_projector = frozen_boundary_cli.attach_projector_capture(
                replay_iteration_overrides,
                capture_dir=capture_dir,
                manifest_path=capture_manifest,
                capture_iteration=args.relion_projector_capture_iteration,
                init_relion_iteration=args.init_relion_iteration,
                relion_replay_dir=args.perturb_replay_relion_dir,
                volume_shape=ds.volume_shape,
                n_classes=args.n_classes,
                validated_frozen_boundary_iteration=(
                    None
                    if frozen_boundary is None
                    else frozen_boundary.completed_relion_iteration
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid captured RELION projector replay: {exc}") from exc
    elif (
        args.relion_projector_capture_manifest is not None
        or args.relion_projector_capture_iteration is not None
    ):
        raise SystemExit(
            "--relion-projector-capture-manifest/iteration require "
            "--relion-projector-capture-dir"
        )
    if frozen_boundary is not None:
        frozen_boundary_cli.validate_projector_only_replay_slots(
            replay_iteration_overrides,
            projector_slot=None if captured_projector is None else captured_projector.replay_slot,
        )

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
    perturb_replay_restart_state_iterations = tuple(
        sorted(
            {
                int(token.strip())
                for token in args.perturb_replay_restart_state_iterations.split(",")
                if token.strip()
            }
        )
    )
    if any(value < 0 for value in perturb_replay_restart_state_iterations):
        raise SystemExit("--perturb-replay-restart-state-iterations values must be non-negative")
    if perturb_replay_restart_state_iterations and args.perturb_replay_relion_dir is None:
        raise SystemExit(
            "--perturb-replay-restart-state-iterations requires --perturb_replay_relion_dir"
        )
    perturb_replay_restart_provenance_path = None
    perturb_replay_restart_provenance_sha256 = None
    if perturb_replay_restart_state_iterations:
        if args.perturb_replay_restart_provenance is None:
            raise SystemExit(
                "--perturb-replay-restart-state-iterations requires "
                "--perturb-replay-restart-provenance"
            )
        perturb_replay_restart_provenance_path = Path(
            args.perturb_replay_restart_provenance
        ).expanduser().resolve()
        if not perturb_replay_restart_provenance_path.is_file():
            raise SystemExit(
                "--perturb-replay-restart-provenance is not a file: "
                f"{perturb_replay_restart_provenance_path}"
            )
        perturb_replay_restart_provenance_sha256 = _sha256_file(
            perturb_replay_restart_provenance_path
        )
        logger.info(
            "SamplingPerturbation restart provenance: iterations=%s path=%s sha256=%s",
            list(perturb_replay_restart_state_iterations),
            perturb_replay_restart_provenance_path,
            perturb_replay_restart_provenance_sha256,
        )
    elif args.perturb_replay_restart_provenance is not None:
        raise SystemExit(
            "--perturb-replay-restart-provenance requires "
            "--perturb-replay-restart-state-iterations"
        )
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

    sampling_kwargs = _refine_sampling_kwargs(args, initial_sampling.coarse_order)

    run_file_writer = None
    if int(args.write_iteration_every) > 0:
        if shape_class_rows is not None:
            logger.warning("RELION run files are not written for particle STARs with several image shapes")
        else:
            run_file_writer = RunFileWriter(
                args.output,
                settings=RunSettings(
                    output_root=os.path.join(args.output, "run"),
                    random_seed=int(args.seed),
                    nr_iter=int(args.max_iter),
                    particle_diameter=float(particle_diameter_ang or 0.0),
                    adaptive_oversampling=int(args.adaptive_oversampling),
                    auto_local_healpix_order=int(sampling_kwargs["auto_local_healpix_order"]),
                    max_significants=int(args.max_significants),
                    symmetry=symmetry,
                    healpix_order_original=int(initial_sampling.coarse_order),
                    offset_range_original_angstrom=float(args.offset_range) * float(ds.voxel_size),
                    offset_step_original_angstrom=float(args.offset_step) * float(ds.voxel_size),
                    perturbation_factor=float(args.perturb_factor),
                    command_line=" ".join(sys.orig_argv),
                ),
                input_star=os.path.join(args.data_dir, "particles.star"),
                half_rows=[particle_layout.half1_rows, particle_layout.half2_rows],
                write_every=int(args.write_iteration_every),
                write_unfiltered_maps=bool(args.write_unfiltered_half_maps),
                keep_iterations=int(args.keep_iterations),
            )
    continued_iterations = 0 if resume_snapshot is None else int(resume_snapshot.relion_iteration)

    # The start-up tau2 is a box-scale volume (float64 at start-up: 4.1 GB at box 800).
    # Hand it over as a host array and drop the device copy, so this frame does not
    # hold it on the device for the whole refinement; the loop stages its own copy
    # and releases it after the first tau2 update (census, bigbox 14468686).
    initial_mean_variance_host = np.asarray(jax.device_get(mean_variance))
    del mean_variance
    result = refine_single_volume(
        experiment_datasets=experiment_datasets,
        init_volume=init_vol_ft,
        init_noise_variance=(
            noise_variance if optics_group_ids_per_half is None else [noise_variance, noise_variance]
        ),
        init_mean_variance=initial_mean_variance_host,
        translations=translations_jnp,
        options=RefinementOptions(
            symmetry=SymmetryOptions(point_group=symmetry),
            disc_type=os.environ.get("RELAX_DISC_TYPE_OVERRIDE", "linear_interp"),
            schedule=RefinementSchedule(
                # --max_iter counts from iteration 1 of the whole run, as RELION's --iter.
                max_iter=int(args.max_iter) - continued_iterations,
                init_current_size=init_current_size,
                init_fsc=None if frozen_boundary is None else frozen_boundary.fsc,
                ini_high_angstrom=_ini_high_for_lowpass,
                init_data_vs_prior=relion_start_data_vs_prior,
                init_ave_Pmax=None if frozen_boundary is None else frozen_boundary.ave_pmax,
                init_has_high_fsc_at_limit=(
                    None if frozen_boundary is None else frozen_boundary.has_high_fsc_at_limit
                ),
                init_relion_incr_size=(
                    10 if frozen_boundary is None else frozen_boundary.relion_incr_size
                ),
                init_healpix_order=sampling_kwargs["init_healpix_order"],
                max_healpix_order=initial_sampling.max_order,
                init_translation_range=sampling_kwargs["init_translation_range"],
                init_translation_step=sampling_kwargs["init_translation_step"],
                init_translation_sigma_angstrom=(
                    frozen_boundary.translation_sigma_angstrom_per_half
                    if frozen_boundary is not None
                    else (
                        relion_init_sigma_offset_angstrom
                        if relion_init_sigma_offset_angstrom is not None
                        else args.offset_sigma_angstrom
                    )
                ),
                particle_diameter_ang=particle_diameter_ang,
                init_relion_iteration=(
                    args.init_relion_iteration if resume_snapshot is None else continued_iterations
                ),
                skip_final_iteration=bool(args.skip_final_iteration),
            ),
            batching=RefinementBatching(
                image_batch_size=args.image_batch_size,
                rotation_block_size=args.rotation_block_size,
            ),
            overlap=HalfOverlapOptions(
                overlap_halves=bool(args.overlap_halves),
            ),
            adaptive=AdaptiveOptions(
                relion_current_sizes=oracle_current_sizes,
                relion_healpix_orders=oracle_healpix_orders,
                adaptive_oversampling=args.adaptive_oversampling,
                coarse_engine=args.coarse_engine,
                max_significants=args.max_significants,
            ),
            parity=RelionParityOptions(
                tau2_fudge=effective_tau2_fudge,
                perturb_factor=args.perturb_factor,
                perturb_seed=effective_perturb_seed,
                optimizer_random_seed=args.seed,
                relion_optics_image_sizes=relion_optics_image_sizes,
                relion_optics_pixel_sizes=relion_optics_pixel_sizes,
                optics_group_ids_per_half=optics_group_ids_per_half,
                relion_model_pixel_size=relion_model_pixel_size,
                perturb_replay_relion_dir=args.perturb_replay_relion_dir,
                perturb_replay_restart_state_iterations=perturb_replay_restart_state_iterations,
                final_sampling_replay_relion_dir=final_sampling_replay_relion_dir,
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
            local_search=LocalSearchOptions(
                auto_local_healpix_order=sampling_kwargs["auto_local_healpix_order"],
                local_search_profile_mode=args.local_search_profile,
            ),
            k_class=KClassOptions(
                n_classes=args.n_classes,
                first_iteration_seed_classes=(
                    relion_class3d_seed_classes(
                        particle_layout.accuracy_trial_order_local, int(args.seed), int(args.n_classes)
                    )
                    if args.n_classes > 1 and args.init_volume is not None and resume_snapshot is None
                    else None
                ),
            ),
            checkpoint=CheckpointOptions(writer=run_file_writer, resume=resume_snapshot),
            replay=ReplayState(
                init_reference_real=None if resume_snapshot is not None else init_reference_real_for_projector,
                init_refinement_state_fields=(
                    None if frozen_boundary is None else frozen_boundary.refinement_state_fields
                ),
                replay_iteration_overrides=replay_iteration_overrides,
                final_replay_override=final_replay_override,
                final_replay_reference_maps=final_replay_reference_maps,
                final_replay_source_iteration=final_replay_source_iteration,
                init_group_ids=list(particle_groups.group_ids_per_half),
                init_group_count=particle_groups.n_groups,
                relion_scale_follower_count=follower_topology.n_followers,
                relion_scale_follower_owners_by_iteration=follower_topology.owners_by_iteration,
                relion_scale_reduction_mode=follower_topology.reduction_mode,
                relion_follower_scale_replay=follower_topology.replay,
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
            debug=EngineDebugOptions(
                save_intermediates_dir=args.save_intermediates_dir,
                save_intermediates_skip_unregularized=bool(args.save_intermediates_skip_unregularized),
                state_swap_probe=state_swap_probe,
                assert_initial_scoring_state_immutable=frozen_boundary is not None,
                stop_after_local_search_profile=bool(args.stop_after_local_search_profile),
                stop_after_local_search=bool(args.stop_after_local_search),
                stop_after_local_search_score_only=bool(args.stop_after_local_search_score_only),
                sealed_sampling_state=(
                    frozen_boundary.sampling_state
                    if frozen_boundary is not None and frozen_boundary.fixed_diagnostic_arm
                    else None
                ),
                sealed_scoring_context=(
                    {
                        "schema": frozen_boundary.schema,
                        "completed_relion_iteration": frozen_boundary.completed_relion_iteration,
                        "consumer_relion_iteration": frozen_boundary.consumer_relion_iteration,
                        "source_sha256": frozen_boundary.source_sha256,
                        "source_roles": frozen_boundary.source_roles,
                        "runtime_config": frozen_boundary.runtime_config,
                        "map_lineage": frozen_boundary.map_lineage,
                    }
                    if frozen_boundary is not None and frozen_boundary.fixed_diagnostic_arm
                    else None
                ),
                expected_accuracy=ExpectedAccuracyOptions(
                    half1_base_order_local=particle_layout.accuracy_base_order_local,
                    half1_trial_order_local=particle_layout.accuracy_trial_order_local,
                    half1_optics_group_ids=particle_layout.accuracy_optics_group_ids,
                    half1_particle_ids=particle_layout.accuracy_particle_ids,
                    half1_ctf_params=expected_accuracy_half1_ctf_params,
                    do_ctf_correction=expected_accuracy_do_ctf_correction,
                ),
            ),
        ),
    )

    if run_file_writer is not None:
        run_file_writer.wait()  # the last iteration's files, written in the background
    validate_state_swap_probe_application(
        state_swap_probe,
        result.get("state_swap_probe_applied_relion_iterations"),
    )

    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Refinement complete in %.1fs (%d iterations)", total_time, args.max_iter)
    logger.info("=" * 70)

    if result.get("profile_only"):
        local_profile_rows = profile_rows_for_json(result.get("local_profile_history", []))
        global_profile_rows = profile_rows_for_json(result.get("global_profile_history", []))
        setup_phase_seconds = {
            str(key): float(value) for key, value in result.get("setup_phase_seconds", {}).items()
        }
        timing_rows = parity_dump._collect_timing_rows(timing_dir_path)
        timing_summary = parity_dump._summarize_timing_rows(timing_rows)
        profile_summary = {
            "symmetry": symmetry_provenance,
            "initial_pose_source_requested": initial_poses.provenance.requested_source,
            "initial_pose_source_resolved": initial_poses.provenance.resolved_source,
            "initial_pose_source_sha256": initial_poses.provenance.sha256,
            "profile_only": True,
            "stop_after_local_search_score_only": bool(result.get("stop_after_local_search_score_only", False)),
            "git_commit": git_head_or_none(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "jax_version": getattr(jax, "__version__", None),
            "jaxlib_version": getattr(jaxlib, "__version__", None),
            "jax_devices": [str(device) for device in jax.devices()],
            "data_dir": str(Path(args.data_dir).resolve()),
            "output_dir": str(Path(args.output).resolve()),
            "timing_dir": str(timing_dir_path.resolve()) if timing_dir_path is not None else None,
            "total_time_s": float(total_time),
            "current_sizes": [int(x) for x in result.get("current_sizes", [])],
            "wall_times_trajectory": [float(x) for x in result.get("wall_times", [])],
            "n_images": int(n_images),
            "image_shape": [int(x) for x in ds.image_shape],
            "volume_shape": [int(x) for x in ds.volume_shape],
            "voxel_size": float(ds.voxel_size),
            "healpix_order": int(args.healpix_order),
            "auto_local_healpix_order": int(args.auto_local_healpix_order),
            "adaptive_oversampling": int(args.adaptive_oversampling),
            "max_significants": int(args.max_significants),
            "max_significants_resolution": runtime_controls.max_significants_resolution,
            "diagnostic_single_half": bool(args.diagnostic_single_half),
            "setup_phase_seconds": setup_phase_seconds,
            "local_profile_rows": local_profile_rows,
            "global_profile_rows": global_profile_rows,
            "timing_rows": timing_rows,
            "timing_summary": timing_summary,
            "perturb_replay_restart_state_iterations": list(
                perturb_replay_restart_state_iterations
            ),
            "perturb_replay_restart_provenance_path": (
                str(perturb_replay_restart_provenance_path)
                if perturb_replay_restart_provenance_path is not None
                else None
            ),
            "perturb_replay_restart_provenance_sha256": (
                perturb_replay_restart_provenance_sha256
            ),
            "relion_projector_replay_slot": (
                None if captured_projector is None else captured_projector.replay_slot
            ),
            "relion_projector_source_manifest_sha256": (
                None if captured_projector is None else captured_projector.source_manifest_sha256
            ),
            "relion_projector_capture_dir": (
                None
                if captured_projector is None
                else str(captured_projector.source_dir)
            ),
            "relion_projector_capture_manifest": (
                None
                if captured_projector is None
                else str(captured_projector.source_manifest)
            ),
            "state_swap_probe": state_swap_probe,
            "state_swap_probe_applied_relion_iterations": [
                int(iteration)
                for iteration in result.get(
                    "state_swap_probe_applied_relion_iterations",
                    [],
                )
            ],
        }
        profile_path = Path(args.output) / "local_search_profile_only.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w", encoding="utf-8") as f:
            json.dump(profile_summary, f, indent=2, sort_keys=True)
        logger.info("Profile-only summary saved to %s", profile_path)
        if args.benchmark_ledger_json:
            ledger_path = Path(args.benchmark_ledger_json)
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("w", encoding="utf-8") as f:
                json.dump(profile_summary, f, indent=2, sort_keys=True)
            logger.info("Benchmark ledger saved to %s", ledger_path)
        print("\n" + "=" * 70)
        print("LOCAL SEARCH PROFILE ONLY")
        print("=" * 70)
        print(f"Profiles: {len(local_profile_rows)}")
        print(f"Total wall time: {total_time:.1f}s")
        if result.get("current_sizes"):
            print(f"Current size: {result['current_sizes'][-1]}")
        print(f"Summary JSON: {profile_path}")
        print("=" * 70)
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
        perturb_replay_restart_provenance_path=perturb_replay_restart_provenance_path,
        perturb_replay_restart_provenance_sha256=perturb_replay_restart_provenance_sha256,
        perturb_replay_restart_state_iterations=perturb_replay_restart_state_iterations,
        relion_dispatch_schedule=relion_dispatch_schedule,
        captured_projector=captured_projector,
        state_swap_probe=state_swap_probe,
        symmetry_provenance=symmetry_provenance,
        tau2_fudge_source=tau2_fudge_source,
        total_time=total_time,
        use_fresh_auto_refine_order=use_fresh_auto_refine_order,
    )

    archive_report = write_refinement_archive(
        result,
        out_path=os.path.join(args.output, "refinement_results.npz"),
        metadata=save_dict,
        half_indices=(particle_layout.half1_rows, particle_layout.half2_rows),
        n_images=n_images,
        skip_large_outputs=args.skip_large_outputs,
    )

    timing_rows = parity_dump._collect_timing_rows(timing_dir_path)
    timing_summary = parity_dump._summarize_timing_rows(timing_rows)
    if args.benchmark_ledger_json:
        ledger_path = Path(args.benchmark_ledger_json)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = {
            "git_commit": git_head_or_none(),
            "git_provenance": archive_report.git_provenance,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "jax_version": getattr(jax, "__version__", None),
            "jaxlib_version": getattr(jaxlib, "__version__", None),
            "jax_devices": [str(device) for device in jax.devices()],
            "data_dir": str(Path(args.data_dir).resolve()),
            "output_dir": str(Path(args.output).resolve()),
            "timing_dir": str(timing_dir_path.resolve()) if timing_dir_path is not None else None,
            "max_iter": int(args.max_iter),
            "random_seed": int(args.seed),
            "random_seed_source": str(optimizer_seed_source),
            "n_iterations_emitted": int(len(result.get("current_sizes", []))),
            "n_wall_times": int(len(result.get("wall_times", []))),
            "total_time_s": float(total_time),
            "wall_times_trajectory": [float(x) for x in result.get("wall_times", [])],
            "current_sizes": [int(x) for x in result.get("current_sizes", [])],
            "pixel_resolutions": [float(x) for x in result.get("pixel_resolutions", [])],
            "ave_Pmax_trajectory": [float(x) for x in result.get("ave_Pmax_trajectory", [])],
            "n_images": int(n_images),
            "image_shape": [int(x) for x in ds.image_shape],
            "volume_shape": [int(x) for x in ds.volume_shape],
            "voxel_size": float(ds.voxel_size),
            "n_rotations": int(n_rotations),
            "n_translations": int(translations.shape[0]),
            "healpix_order": int(args.healpix_order),
            "coarse_healpix_order": int(initial_sampling.coarse_order),
            "finest_healpix_order": int(initial_sampling.fine_order),
            "max_healpix_order": None if initial_sampling.max_order is None else int(initial_sampling.max_order),
            "max_healpix_order_source": str(initial_sampling.max_order_source),
            "auto_local_healpix_order": int(args.auto_local_healpix_order),
            "adaptive_oversampling": int(args.adaptive_oversampling),
            "max_significants": int(args.max_significants),
            "max_significants_resolution": runtime_controls.max_significants_resolution,
            "setup_phase_seconds": archive_report.setup_phase_seconds,
            "local_profile_rows": archive_report.local_profile_rows,
            "global_profile_rows": archive_report.global_profile_rows,
            "timing_rows": timing_rows,
            "timing_summary": timing_summary,
            "perturb_replay_restart_state_iterations": list(
                perturb_replay_restart_state_iterations
            ),
            "perturb_replay_restart_provenance_path": (
                str(perturb_replay_restart_provenance_path)
                if perturb_replay_restart_provenance_path is not None
                else None
            ),
            "perturb_replay_restart_provenance_sha256": (
                perturb_replay_restart_provenance_sha256
            ),
            "relion_projector_replay_slot": (
                None if captured_projector is None else captured_projector.replay_slot
            ),
            "relion_projector_source_manifest_sha256": (
                None if captured_projector is None else captured_projector.source_manifest_sha256
            ),
            "relion_projector_capture_dir": (
                None
                if captured_projector is None
                else str(captured_projector.source_dir)
            ),
            "relion_projector_capture_manifest": (
                None
                if captured_projector is None
                else str(captured_projector.source_manifest)
            ),
            "frozen_boundary_dir": (
                None if frozen_boundary is None else str(frozen_boundary.source_dir)
            ),
            "frozen_boundary_manifest_sha256": (
                None if frozen_boundary is None else frozen_boundary.source_manifest_sha256
            ),
            "frozen_boundary_sha256": (
                None if frozen_boundary is None else frozen_boundary.boundary_sha256
            ),
            "frozen_boundary_completed_relion_iteration": (
                None if frozen_boundary is None else frozen_boundary.completed_relion_iteration
            ),
        }
        with ledger_path.open("w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
        logger.info("Benchmark ledger saved to %s", ledger_path)

    write_final_maps(
        result,
        output_dir=args.output,
        volume_shape=ds.volume_shape,
        pixel_size_angstrom=ds.voxel_size,
        n_classes=args.n_classes,
        skip_large_outputs=args.skip_large_outputs,
    )

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("REFINEMENT SUMMARY")
    print("=" * 70)
    print(f"{'Iter':>4s}  {'CurSize':>8s}  {'PixRes':>8s}  {'ResA':>8s}  {'Time(s)':>8s}", end="")
    if any(c is not None for c in result["significant_counts"]):
        print(f"  {'MedSig':>8s}", end="")
    print()
    print("-" * 70)

    for i in range(len(result["current_sizes"])):
        cs = result["current_sizes"][i]
        pr = result["pixel_resolutions"][i]
        res_a = _shell_index_to_resolution_angstrom(pr, ds.image_shape[0], ds.voxel_size)
        wt = result["wall_times"][i]
        line = f"{i + 1:4d}  {cs:8d}  {pr:8.1f}  {res_a:8.2f}  {wt:8.1f}"
        if result["significant_counts"][i] is not None:
            med_sig = int(np.median(np.asarray(result["significant_counts"][i])))
            line += f"  {med_sig:8d}"
        print(line)

    print("-" * 70)
    print(f"Total wall time: {total_time:.1f}s")
    # A continuation from a converged state runs only the final all-data pass,
    # which records no per-iteration row.
    if result["current_sizes"]:
        print(f"Final current_size: {result['current_sizes'][-1]}")
        print(f"Final pixel resolution: {result['pixel_resolutions'][-1]:.1f}")
    # RELION reports the final all-data iteration's current resolution (updateCurrentResolution after it,
    # ml_optimiser_mpi.cpp:4329); without that pass, the last numbered iteration's.
    final_state = result.get("convergence_state")
    if result.get("final_all_data_ran") and final_state is not None:
        print(f"Final resolution: {float(final_state.current_resolution):.2f} A (final all-data iteration)")
    elif result["current_sizes"]:
        print(
            "Final resolution: "
            f"{_shell_index_to_resolution_angstrom(result['pixel_resolutions'][-1], ds.image_shape[0], ds.voxel_size):.2f} A"
        )
    print("=" * 70)


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
        if (
            exc.__class__.__name__ == "BPrefContributionDumpComplete"
            and os.environ.get("RELAX_BPREF_CONTRIBUTION_STOP_AFTER_TARGET") == "1"
        ):
            logger.info(
                "RECOVAR BPref contribution dump completed; stopping at the "
                "requested pass-2 boundary: %s",
                exc,
            )
            sys.exit(0)
        if (
            exc.__class__.__name__ == "Pass2DumpComplete"
            and (
                os.environ.get("RELAX_PASS2_DUMP_STOP_AFTER_TARGET") == "1"
                or os.environ.get("RELAX_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET") == "1"
            )
        ):
            logger.info(
                "RECOVAR pass-2 operand dump completed; stopping at the "
                "requested fine-score boundary or norm/scale boundary: %s",
                exc,
            )
            sys.exit(0)
        raise
