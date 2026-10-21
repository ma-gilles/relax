"""Refinement command options, job defaults and supported input-mode admission.

The execution controller owns cache/import setup and when data-dependent checks
run. This module validates the parsed command line and resolves it into option
records; the flags themselves are declared in ``refinement_parser.py``, and
numerical refinement options remain in ``refinement_options.py``.
"""

import dataclasses
import logging
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from relax.refinement.refinement_options import (
    FIRSTITER_CC_TREE_RESCORE_DEFAULT_MAX_MARGIN,
    AdaptiveOptions,
    ExecutionOptions,
    InitialSampling,
    JobDefaults,
    KClassOptions,
    LocalSearchOptions,
    RandomSeed,
    RefinementSchedule,
    RelionConsistencyOptions,
    StartState,
    relax_mode_consistency,
)

if TYPE_CHECKING:
    from relax.star.metadata import MaxSignificantsResolution

logger = logging.getLogger(__name__)

def resolve_firstiter_cc_tree_rescore_margin(
    *,
    firstiter_cc: bool,
    n_classes: int,
    tree_rescore_max_margin: str,
) -> float | None:
    """The coarse-tree top-2 rescore margin (``RelionParityOptions.firstiter_cc_tree_rescore_max_margin``).

    RELION Class3D also uses first-iteration CC, but the exact coarse-tree replay currently supports only
    K=1. ``tree_rescore_max_margin`` (``--firstiter_cc_tree_rescore_max_margin``) is ``auto``
    (``FIRSTITER_CC_TREE_RESCORE_DEFAULT_MAX_MARGIN`` for K=1 with --firstiter_cc, else off), ``off`` or a
    margin.
    """

    use_k1_defaults = bool(firstiter_cc) and int(n_classes) == 1
    token = str(tree_rescore_max_margin).strip().lower()
    if token == "auto":
        margin = FIRSTITER_CC_TREE_RESCORE_DEFAULT_MAX_MARGIN if use_k1_defaults else None
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
    return margin


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


def resolve_job_defaults(args) -> JobDefaults:
    """Resolve the defaults that depend on the job type (Refine3D K=1 or Class3D K>1)."""
    k1 = int(args.n_classes) == 1
    max_iter = args.max_iter
    if max_iter is None:
        # Auto-refine runs to convergence with nr_iter = --auto_iter_max (999);
        # the Class3D GUI runs 25 iterations.
        max_iter = 999 if k1 else 25
    image_fourier_backend = args.image_fourier_backend
    if image_fourier_backend == "auto":
        # RELION's CUDA image preprocessing is the one the resident pass 2's exact
        # BPref operands read, for K=1 and Class3D alike.
        image_fourier_backend = "relion_cuda"
    apply_initial_lowpass = args.apply_initial_lowpass
    if apply_initial_lowpass is None:
        apply_initial_lowpass = args.frozen_boundary_dir is None
    return JobDefaults(max_iter, image_fourier_backend, apply_initial_lowpass)


def validate_strict_highres_exp(args) -> None:
    """Admit RELION's --strict_highres_exp: a positive limit, Class3D (K>1).

    RELION's GUI offers it for Class2D and Class3D only (pipeline_jobs.cpp:3706); auto-refine's final all-data
    pass would also need the cap, which relax does not implement, so K=1 refuses it.
    """
    if args.strict_highres_exp is None:
        return
    if not args.strict_highres_exp > 0:
        raise SystemExit(f"--strict_highres_exp must be positive, got {args.strict_highres_exp}")
    if int(args.n_classes) == 1:
        raise SystemExit("--strict_highres_exp is supported for Class3D (K>1) only")


def validate_sigma_ang(args) -> None:
    """Admit RELION's --sigma_ang: a positive width (Refine3D and Class3D), not with --skip_align."""
    if args.sigma_ang is not None and not args.sigma_ang > 0:
        raise SystemExit(f"--sigma_ang must be positive, got {args.sigma_ang}")
    if args.sigma_ang is not None and getattr(args, "skip_align", False):
        # relion_refine drops the orientational prior when it skips the alignment (ml_optimiser.cpp:2381-2390).
        raise SystemExit("--sigma_ang has no effect with --skip_align, which searches no orientations")


def resolve_standalone_k1_start(args) -> bool:
    """Whether the run builds RELION's particle table from the input (``--relion-half-sets-from-input``).

    Unset, a fresh K=1 start with no RELION output defaults to it.

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
    from_input = args.relion_half_sets_from_input
    if from_input is None:
        from_input = bool(fresh_k1 and args.relion_half_sets is None and args.relion_init_dir is None)
    validate_input_half_sets(args, from_input=from_input)
    return from_input


def validate_input_half_sets(args, *, from_input: bool) -> None:
    """Reject an input-derived RELION particle table where RELION would not build one this way."""
    if not from_input:
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
    if getattr(args, "skip_align", False):
        reasons.append("--skip_align is single-shape")
    if reasons:
        raise SystemExit("optics groups on several image shapes: " + "; ".join(reasons))


def validate_skip_align_args(args) -> None:
    """Refuse what RELION's --skip_align route does not cover in relax yet, before any data is read."""

    if not getattr(args, "skip_align", False):
        return
    reasons = []
    if int(args.n_classes) <= 1:
        reasons.append("it classifies, so it needs --n_classes > 1")
    if args.continue_optimiser_star is not None:
        reasons.append("--continue is not implemented for it")
    if args.initial_pose_source == "none" or args.relion_init_dir is not None:
        reasons.append("its poses are the input STAR's (no other pose source)")
    if reasons:
        raise SystemExit("--skip_align: " + "; ".join(reasons))


def validate_continue_args(args) -> None:
    """Refuse options a continuation cannot honour.

    The run files pin the trajectory, so a RELION-seeded or replayed start, a frozen
    boundary or a sampling oracle would contradict them. The seed must be the original
    run's (:func:`resolve_seed`).
    """

    conflicts = [
        flag
        for flag, value in (
            ("--perturb_replay_relion_dir", args.perturb_replay_relion_dir),
            ("--relion_init_dir", args.relion_init_dir),
            ("--frozen-boundary-dir", args.frozen_boundary_dir),
            ("--init_noise_from_npz", args.init_noise_from_npz),
            ("--relion_current_sizes", args.relion_current_sizes),
            ("--relion_healpix_orders", args.relion_healpix_orders),
        )
        if value is not None
    ]
    if int(args.init_relion_iteration) != 0:
        conflicts.append("--init_relion_iteration")
    if args.diagnostic_single_half:
        conflicts.append("--diagnostic_single_half")
    if conflicts:
        raise SystemExit("--continue cannot be combined with " + ", ".join(conflicts))


def resolve_seed(args, optimiser_star, *, sealed: bool) -> RandomSeed:
    """The run's random seed, decided once.

    A continued run keeps its run files' seed (it fixes the particle order and the sampling perturbations;
    a different ``--seed`` is refused) and records it as an explicit seed, as before. Otherwise an explicit
    ``--seed`` wins; a run without one inherits ``_rlnRandomSeed`` from the optimiser
    :func:`optimiser_seed_source` names; else relion_refine's default ``-1`` takes the time
    (``MlOptimiser::initialiseWorkLoad``, ml_optimiser.cpp:2827).
    """
    explicit_seed = args.seed
    if args.continue_optimiser_star is not None:
        explicit_seed = _continued_run_seed(args)
    if explicit_seed is not None:
        return RandomSeed(int(explicit_seed), "explicit CLI")
    seed_star = optimiser_seed_source(args, optimiser_star, sealed=sealed)
    if seed_star is not None:
        from relax.star.metadata import read_relion_optimiser_metadata

        relion_seed = read_relion_optimiser_metadata(seed_star).get("random_seed")
        if relion_seed is not None:
            return RandomSeed(int(relion_seed), f"RELION optimiser {Path(seed_star).resolve()}")
    return RandomSeed(int(time.time()), "RELION default -1: the time")


def _continued_run_seed(args) -> int:
    """The continued run's ``rlnRandomSeed``; a different explicit ``--seed`` is refused."""
    from relax.star.blocks import read_star_blocks

    general = read_star_blocks(args.continue_optimiser_star).get("optimiser_general", {})
    if "rlnRandomSeed" not in general:
        raise SystemExit(f"{args.continue_optimiser_star} has no rlnRandomSeed")
    seed = int(general["rlnRandomSeed"])
    if args.seed is not None and int(args.seed) != seed:
        raise SystemExit(f"--seed {args.seed} differs from the continued run's seed {seed}")
    return seed


def validate_sigma_tilt(args) -> None:
    """Admit RELION's --sigma_tilt on the parsed command line: a positive width, not with --skip_align, and
    dropped when --sigma_ang is given, as relion_refine does (call after ``validate_sigma_ang``)."""
    if args.sigma_tilt is None:
        return
    if not args.sigma_tilt > 0:
        raise SystemExit(f"--sigma_tilt must be positive, got {args.sigma_tilt}")
    if getattr(args, "skip_align", False):
        raise SystemExit("--sigma_tilt has no effect with --skip_align, which searches no orientations")
    if args.sigma_ang is not None:
        # parseInitial: `if (_sigma_ang > 0.) ... else if (_sigma_rot > 0. || _sigma_tilt > 0. || ...)`.
        logger.warning(
            "--sigma_tilt %g is ignored: --sigma_ang %g sets the width of all three angles, as in relion_refine",
            args.sigma_tilt, args.sigma_ang,
        )
        args.sigma_tilt = None


def validate_sigma_tilt_route(args, *, subtomograms: bool) -> None:
    """Refuse --sigma_tilt on the routes that do not implement it (after ``validate_sigma_tilt``).

    Called where the particle loader learns the route. The loader is also driven with argument records that
    hold only the fields it reads (as its ``skip_align`` reads allow), so an absent ``sigma_tilt`` is no option.

    The prior needs each particle's own orientation list in a search that is otherwise global. The subtomogram
    passes have it, for one class and for K (the local-search coarse pass, relax.scoring.tomo_coarse). The
    single-particle global pass scores every particle on one shared grid.
    """

    if getattr(args, "sigma_tilt", None) is None:
        return
    if not subtomograms:
        raise SystemExit(
            "--sigma_tilt is implemented for subtomogram particles (RELION 5 2D stacks) only: the "
            "single-particle global search has no per-particle orientation prior"
        )


def validate_tomo_args(args, frozen_boundary, double_image_preprocessing):
    """Refuse options the subtomogram (2D-stack) path does not implement."""

    if getattr(args, "skip_align", False):
        raise SystemExit("--skip_align is not implemented for subtomogram particles yet")
    if frozen_boundary is not None or args.relion_init_dir is not None or args.init_noise_from_npz is not None:
        raise SystemExit("subtomogram particles start fresh from RELION's inputs (no frozen, replayed or loaded state)")
    if double_image_preprocessing:
        raise SystemExit("subtomogram particles have no float64 scoring diagnostic")


def require_command_n_classes(command: str, n_classes: int) -> None:
    """``relax refine`` runs K=1 auto-refine and ``relax class3d`` K>1 classification."""

    if command == "refine" and n_classes != 1:
        raise SystemExit(f"relax refine runs 3D auto-refine (K=1); use relax class3d --n_classes {n_classes}")
    if command == "class3d" and n_classes < 2:
        raise SystemExit("relax class3d needs --n_classes K with K >= 2; use relax refine for K=1")
    if command not in {"refine", "class3d"}:
        raise ValueError(f"unknown refinement command {command!r}")


def _explicit_consistency_options(args) -> dict[str, str]:
    """The RELION-consistency options given on the command line, by name."""

    names = RelionConsistencyOptions.__dataclass_fields__
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def require_consistency_arguments(args) -> None:
    """Refuse, from the command line alone, the consistency options and ``--mode relax`` the run cannot honour.

    A run seeded from, replaying or continuing RELION's own state takes statistics, references or
    projectors computed with RELION's rules, so every non-default option and ``--mode relax`` are
    refused there. An explicit option is refused where its route does not honour it; ``--mode relax``
    skips such an option instead (``resolve_consistency_options``).
    """

    explicit = RelionConsistencyOptions(**_explicit_consistency_options(args))
    chosen = explicit.non_default()
    requested = ["--mode relax"] if args.mode == "relax" else []
    requested += [f"--{name} {value}" for name, value in chosen.items()]
    if not requested:
        return
    relion_state = [
        flag
        for flag, value in (
            ("--perturb_replay_relion_dir", args.perturb_replay_relion_dir),
            ("--relion_init_dir", args.relion_init_dir),
            ("--frozen-boundary-dir", args.frozen_boundary_dir),
            ("--init_noise_from_npz", args.init_noise_from_npz),
        )
        if value is not None
    ]
    if int(args.init_relion_iteration) != 0:
        relion_state.append("--init_relion_iteration")
    if relion_state:
        raise SystemExit(
            f"{', '.join(requested)}: not available with RELION-seeded or replayed state ({', '.join(relion_state)})"
        )
    if explicit.firstiter_cc_support != "relion" and not args.firstiter_cc:
        raise SystemExit(f"--firstiter_cc_support {explicit.firstiter_cc_support} needs --firstiter_cc (the CC iteration)")
    if explicit.gridding_kernel != "radial" and int(args.n_classes) != 1:
        raise SystemExit(f"--gridding_kernel {explicit.gridding_kernel} is implemented for K=1 auto-refine only")


def resolve_consistency_options(
    args, *, subtomograms: bool = False, several_image_shapes: bool = False, dataset=None
) -> RelionConsistencyOptions:
    """The run's RELION-consistency options: the command line's, under ``--mode relax`` the mode's.

    This is the one place the mode is read; everything after it sees the options only. With
    ``--mode relion`` the options are those given explicitly (RELION's rule otherwise). With
    ``--mode relax`` they are ``RELAX_MODE_CONSISTENCY`` with the explicit options on top, less the
    ones this route does not honour, each logged with its reason (``relax_mode_consistency``);
    subtomogram particles and optics groups on several image shapes honour none and refuse the mode.
    ``dataset`` is read under ``--mode relax`` only, for whether its STAR stores CTF-premultiplied images.
    """

    require_consistency_arguments(args)
    explicit = _explicit_consistency_options(args)
    if args.mode != "relax":
        return RelionConsistencyOptions(**explicit)
    no_option_route = [
        reason
        for reason, present in (
            ("subtomogram particles", subtomograms),
            ("optics groups on several image shapes", several_image_shapes),
        )
        if present
    ]
    if no_option_route:
        raise SystemExit(
            "--mode relax: the RELION-consistency options are implemented for single-particle refinement of one "
            f"image shape from relax's own state; not with {', '.join(no_option_route)}"
        )
    from relax.relion import ctf

    options, skipped = relax_mode_consistency(
        explicit,
        n_classes=int(args.n_classes),
        has_cc_iteration=bool(args.firstiter_cc),
        coarse_engine=args.coarse_engine,
        ctf_premultiplied=(
            dataset is not None
            and ctf.dataset_has_premultiplied_ctf(dataset, tuple(int(v) for v in dataset.image_shape))
        ),
    )
    logger.info("--mode relax: RELION-consistency options %s", options.non_default())
    for name, reason in skipped.items():
        logger.info("--mode relax: %s keeps RELION's rule (%s): %s", name, getattr(options, name), reason)
    return options


def relion_optimiser_star(args, *, relion_half_sets_from_input: bool, sealed_optimiser=None):
    """The one RELION optimiser STAR a run reads, or None.

    Its mask, ``ini_high``, ``max_significants`` and CTF flag come from it, and (``optimiser_seed_source``)
    possibly the random seed. A sealed boundary's fixed arm reads only its completed optimiser
    (``sealed_optimiser``); any other run reads the one ``find_relion_optimiser_star`` locates.
    """
    if sealed_optimiser is not None:
        return Path(sealed_optimiser).resolve()
    return find_relion_optimiser_star(args, relion_half_sets_from_input=relion_half_sets_from_input)


def optimiser_seed_source(args, optimiser_star, *, sealed: bool):
    """The optimiser STAR whose ``_rlnRandomSeed`` a run without ``--seed`` inherits, or None.

    The sealed optimiser, or the run's optimiser when the user named RELION state (``--relion_optimiser``,
    ``--relion_init_dir``, ``--perturb_replay_relion_dir``). An optimiser found only by searching under
    ``--data_dir`` supplies the mask and the support caps but must not change a standalone run's random
    numbers.
    """
    named = any(getattr(args, name, None) for name in ("relion_optimiser", "relion_init_dir", "perturb_replay_relion_dir"))
    return optimiser_star if sealed or named else None


def find_relion_optimiser_star(args, *, relion_half_sets_from_input: bool):
    """Locate a RELION run_optimiser.star to source mask + max_significants from.

    Searches an explicit ``--relion_optimiser`` arg first, then sibling
    directories of ``--relion_half_sets``, the ``--data_dir`` itself, and
    finally any ``relion_ref*/`` subdirectory of ``--data_dir`` (matching
    fixtures that name their RELION output ``relion_ref_os0/`` or similar).
    Picks the latest ``run_it{NNN}_optimiser.star`` if no plain
    ``run_optimiser.star`` is present in a candidate directory.

    With ``--relion-half-sets-from-input`` (``relion_half_sets_from_input``, as
    :func:`resolve_standalone_k1_start` resolved it) the run starts from relion_refine's
    inputs alone, so only an explicit ``--relion_optimiser`` is used: a RELION
    output found next to the data must not supply the mask, ``ini_high``,
    ``max_significants`` or CTF flag. A Class3D (K>1) run given no RELION
    state (optimiser, init/replay directory or half-set STAR) is such a
    standalone start too.
    """
    explicit = getattr(args, "relion_optimiser", None)
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--relion_optimiser {explicit}: no such file")
        return p
    if relion_half_sets_from_input:
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


class RelionRuntimeControls(NamedTuple):
    """Resolved CLI/optimiser controls and significant-support provenance."""

    do_ctf_correction: bool | None
    firstiter_ini_high_angstrom: float | None
    max_significants_resolution: "MaxSignificantsResolution"


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
    from relax.star import metadata as relion_metadata

    expected_accuracy_do_ctf_correction = None
    relion_firstiter_ini_high_angstrom = None
    relion_optimiser_metadata = None
    if optimiser_star is not None:
        from relax.star.metadata import read_relion_optimiser_metadata

        relion_optimiser_metadata = read_relion_optimiser_metadata(optimiser_star)
        stored_do_correct_ctf = relion_optimiser_metadata.get("do_correct_ctf")
        if stored_do_correct_ctf is not None:
            expected_accuracy_do_ctf_correction = bool(stored_do_correct_ctf)
            log.info(
                "RELION expected-accuracy CTF correction: %s (from %s)",
                expected_accuracy_do_ctf_correction,
                optimiser_star,
            )
        optimiser_text = Path(optimiser_star).read_text(errors="ignore")
        relion_firstiter_ini_high_angstrom = relion_metadata.parse_relion_cli_ini_high(optimiser_text)
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
        from relax.star.metadata import resolve_relion_runtime_max_significants

        optimiser_max_significants = relion_optimiser_metadata.get(
            "maximum_significants_arg"
        )
        if optimiser_max_significants is None:
            optimiser_max_significants = relion_metadata.load_relion_max_significants(optimiser_star)
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
        max_significants = int(max_significants_resolution.active_max_significants)
        log.info(
            "RELION max_significants: saved_arg=%s active=%d source=%s "
            "do_grad=%s (from %s)",
            max_significants_resolution.maximum_significants_argument,
            max_significants,
            max_significants_resolution.source,
            max_significants_resolution.do_grad,
            optimiser_star,
        )
    if max_significants is None:
        from relax.star.metadata import resolve_relion_runtime_max_significants

        # Without a RELION optimiser STAR, use relion_refine's own --maxsig default of -1
        # (ml_optimiser.cpp:1109) and its runtime resolution (ml_optimiser.cpp:3692-3699).
        max_significants_resolution = dataclasses.replace(
            resolve_relion_runtime_max_significants(
                override=None,
                optimiser_metadata={"maximum_significants_arg": -1},
                target_iteration=target_iteration,
                do_firstiter_cc=firstiter_cc,
                n_classes=n_classes,
                reference_dimension=3,
            ),
            source="relion_cli_default",
        )
        max_significants = int(max_significants_resolution.active_max_significants)
        log.info("RELION max_significants: --maxsig default -1 -> active %d", max_significants)
    elif max_significants_resolution is None:
        max_significants_resolution = relion_metadata.MaxSignificantsResolution(
            maximum_significants_argument=None,
            active_max_significants=int(max_significants),
            source="cli_override",
            gradient_refine=False,
            do_grad=False,
            target_iteration=target_iteration,
        )

    return RelionRuntimeControls(
        expected_accuracy_do_ctf_correction,
        relion_firstiter_ini_high_angstrom,
        max_significants_resolution,
    )


def resolve_class_reference_paths(args, *, log: logging.Logger) -> tuple[list[str], str]:
    """Class3D's start-up maps and the option that named them: one --init_volume for every class, the
    --ref_star list, --init_class_volumes, or the data directory's reference_init_class00K_relion.mrc."""
    import numpy as np

    from relax.star import metadata as relion_metadata

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
            log.warning(
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
    return class_paths, class_option


def resolve_adaptive_options(args, *, max_significants: int, log) -> AdaptiveOptions:
    """The adaptive sampling options, with RELION's per-iteration current sizes and HEALPix orders when the
    run follows a RELION oracle (``--relion_current_sizes``, ``--relion_healpix_orders``, comma lists).

    ``max_significants`` is the active cap (``resolve_relion_runtime_controls``), not ``--max_significants``.
    """
    oracle_current_sizes = None
    if args.relion_current_sizes is not None:
        oracle_current_sizes = [int(x) for x in args.relion_current_sizes.split(",")]
        log.info("Oracle mode: using RELION current_sizes=%s", oracle_current_sizes)
    oracle_healpix_orders = None
    if args.relion_healpix_orders is not None:
        oracle_healpix_orders = [int(x) for x in args.relion_healpix_orders.split(",")]
        log.info("Oracle mode: using RELION healpix_orders=%s", oracle_healpix_orders)
    return AdaptiveOptions(
        relion_current_sizes=oracle_current_sizes,
        relion_healpix_orders=oracle_healpix_orders,
        adaptive_oversampling=args.adaptive_oversampling,
        coarse_engine=args.coarse_engine,
        max_significants=max_significants,
        strict_highres_exp_angstrom=args.strict_highres_exp,
    )


def resolve_execution(args) -> ExecutionOptions:
    """The E-step's image and rotation block sizes and the environment's JAX cache switch."""
    return ExecutionOptions.from_environ(
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
    )


def resolve_local_search(args) -> LocalSearchOptions:
    """When local angular searches start (``--auto_local_healpix_order``, ``--sigma_ang``), how they are
    profiled (``--local_search_profile``) and whether the run is a probe of the first one
    (``--stop_after_local_search*``)."""
    return LocalSearchOptions(
        auto_local_healpix_order=args.auto_local_healpix_order,
        sigma_ang_deg=args.sigma_ang,
        sigma_tilt_deg=args.sigma_tilt,
        local_search_profile_mode=args.local_search_profile,
        stop_after_local_search_profile=bool(args.stop_after_local_search_profile),
        stop_after_local_search=bool(args.stop_after_local_search),
        stop_after_local_search_score_only=bool(args.stop_after_local_search_score_only),
    )


def resolve_start_state(
    args,
    *,
    particle_groups,
    initial_poses,
    ini_high_angstrom,
    init_data_vs_prior,
    relion_init_sigma_offset_angstrom,
    frozen_boundary,
    init_data_vs_prior_half2=None,
) -> StartState:
    """The state the run starts from: the particles' groups, poses and corrections and the first iteration's
    resolution and translation-prior seeds.

    A frozen boundary owns the translation sigma, the corrections and the direction prior (its first FSC, Pmax,
    high-FSC flag and RELION increment reach the run through the input source,
    ``relax.parity.relion_replay_source.RelionReplay.from_frozen_boundary``); a RELION-seeded start
    (``--relion_init_dir``) the sigma offset; otherwise the flags and the input particles do.
    """
    poses = initial_poses.poses
    return StartState(
        init_group_ids=list(particle_groups.group_ids_per_half),
        init_group_count=particle_groups.n_groups,
        init_relion_optics_group_count=particle_groups.n_optics_groups,
        init_previous_best_translations=None if poses is None else poses["previous_best_translations"],
        init_previous_best_rotation_eulers=None if poses is None else poses["previous_best_rotation_eulers"],
        init_angle_priors=None if poses is None else poses.get("angle_priors"),
        init_image_corrections=(
            initial_poses.image_corrections if frozen_boundary is None else frozen_boundary.image_corrections
        ),
        init_scale_corrections=(
            initial_poses.scale_corrections if frozen_boundary is None else frozen_boundary.scale_corrections
        ),
        init_direction_prior=None if frozen_boundary is None else frozen_boundary.direction_prior_per_half,
        init_translation_sigma_angstrom=(
            frozen_boundary.translation_sigma_angstrom_per_half
            if frozen_boundary is not None
            else (
                relion_init_sigma_offset_angstrom
                if relion_init_sigma_offset_angstrom is not None
                else args.offset_sigma_angstrom
            )
        ),
        ini_high_angstrom=ini_high_angstrom,
        init_data_vs_prior=init_data_vs_prior,
        init_data_vs_prior_half2=init_data_vs_prior_half2,
    )


def resolve_schedule(
    args,
    *,
    max_iter: int,
    initial_sampling,
    init_current_size,
    image_mask,
    continued_iterations,
) -> RefinementSchedule:
    """The numbered schedule: how many iterations remain, where the first one starts and its sampling.

    ``continued_iterations``: the run files' last numbered iteration with ``--continue`` (None for a fresh
    run).
    """
    return RefinementSchedule(
        # --max_iter counts from iteration 1 of the whole run, as RELION's --iter.
        max_iter=int(max_iter) - (continued_iterations or 0),
        init_current_size=init_current_size,
        init_healpix_order=initial_sampling.coarse_order,
        max_healpix_order=initial_sampling.max_order,
        init_translation_range=args.offset_range,
        init_translation_step=args.offset_step,
        # The loader's (particle diameter or None, edge width): the RELION image mask the run resolved.
        particle_diameter_ang=image_mask[0],
        width_mask_edge_px=image_mask[1],
        init_relion_iteration=args.init_relion_iteration if continued_iterations is None else continued_iterations,
        skip_final_iteration=bool(args.skip_final_iteration),
    )


def resolve_k_class(args, *, random_seed: int, trial_order, resumed: bool) -> KClassOptions:
    """The class count, ``--skip_align`` and, for a fresh Class3D run from one reference (``--init_volume``),
    each particle's class in RELION's first iteration, drawn in the expected-accuracy trial order ``trial_order``."""
    from relax.star.input_particle_table import relion_class3d_seed_classes

    return KClassOptions(
        n_classes=args.n_classes,
        skip_align=bool(args.skip_align),
        first_iteration_seed_classes=(
            relion_class3d_seed_classes(trial_order, random_seed, int(args.n_classes))
            if args.n_classes > 1 and args.init_volume is not None and not resumed
            else None
        ),
    )
