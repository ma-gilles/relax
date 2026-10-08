"""Startup pose sources, matched input-STAR poses and normalization corrections."""

import argparse
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from recovar.utils.file_hash import sha256_file as _sha256_file

from relax.helpers import iteration_history
from relax.relion import relion_metadata

if TYPE_CHECKING:
    from relax.relion.input_particle_table import ParticleLayout


class PoseProvenance(NamedTuple):
    """Requested and selected startup pose source, reused by run reports."""

    requested_source: str
    resolved_source: str
    path: Path | None
    sha256: str | None


class InitialPoses(NamedTuple):
    """Selected pose payload and corrections in the established half-local order."""

    poses: dict | None
    image_corrections: list[np.ndarray] | None
    scale_corrections: list[np.ndarray] | None
    provenance: PoseProvenance


def _input_numeric_columns(input_particles, columns, *, field: str) -> np.ndarray:
    """Read finite float64 input-STAR columns as an ``(N, len(columns))`` array."""
    missing = [column for column in columns if column not in input_particles.columns]
    if missing:
        raise ValueError(
            f"RECOVAR input STAR is missing {field} columns: {', '.join(missing)}",
        )
    try:
        values = np.stack(
            [np.asarray(input_particles[column], dtype=np.float64) for column in columns],
            axis=1,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RECOVAR input STAR {field} columns must be numeric") from exc
    expected_shape = (len(input_particles), len(columns))
    if values.shape != expected_shape:
        raise ValueError(
            f"RECOVAR input STAR {field} array has shape {values.shape}, "
            f"expected {expected_shape}",
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"RECOVAR input STAR {field} values must be finite")
    return values


def _input_star_origins_pixels(input_particles, *, voxel_size: float):
    """Return input origins in pixels and their source units.

    rlnOriginX/YAngst are divided by ``voxel_size``; pixel origins are used as
    read; absent origins are zero, as relion_refine reads them
    (Experiment::read, exp_model.cpp:1104-1144). Subtomogram particles (a
    ``rlnOriginZAngst`` column) have 3D origins (exp_model.cpp:1007-1020).
    """
    if "rlnOriginZAngst" in input_particles.columns:
        angstrom_columns = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")
        pixel_columns = ("rlnOriginX", "rlnOriginY", "rlnOriginZ")
    else:
        angstrom_columns = ("rlnOriginXAngst", "rlnOriginYAngst")
        pixel_columns = ("rlnOriginX", "rlnOriginY")
    has_angstrom = [column in input_particles.columns for column in angstrom_columns]
    has_pixels = [column in input_particles.columns for column in pixel_columns]
    if any(has_angstrom) and not all(has_angstrom):
        raise ValueError(f"RECOVAR input STAR must provide all of {angstrom_columns}")
    if any(has_pixels) and not all(has_pixels):
        raise ValueError(f"RECOVAR input STAR must provide all of {pixel_columns}")
    if all(has_angstrom):
        if not np.isfinite(voxel_size) or float(voxel_size) <= 0.0:
            raise ValueError("voxel_size must be positive and finite for Angstrom origins")
        translations = _input_numeric_columns(
            input_particles,
            angstrom_columns,
            field="Angstrom-origin",
        ) / float(voxel_size)
        return translations, "angstrom"
    if all(has_pixels):
        return _input_numeric_columns(input_particles, pixel_columns, field="pixel-origin"), "pixel"
    return np.zeros((len(input_particles), len(angstrom_columns)), dtype=np.float64), "implicit_zero"


def _load_input_star_previous_best_poses(
    input_particles,
    relion_halfset_particles,
    half1_idx,
    half2_idx,
    *,
    voxel_size: float,
):
    """Load deposited poses and norm corrections in the half-local refinement order.

    ``half1_idx`` and ``half2_idx`` index the RECOVAR input particle table,
    whereas ``--relion_half_sets`` may be in a different row order.  Bind the
    two tables by full RELION image identity and fail closed if the supplied
    split is stale, incomplete, or inconsistent with the half-local layout.
    Translations are returned in pixels, matching ``ReplayState`` and RELION's
    previous-best-pose convention. Norm corrections are float64 and 1 when
    the input STAR has no ``rlnNormCorrection``, as in relion_refine.
    """

    input_rows = relion_metadata._particle_identity_rows(
        input_particles,
        label="RECOVAR input STAR",
    )
    halfset_rows = relion_metadata._particle_identity_rows(
        relion_halfset_particles,
        label="RELION half-set STAR",
    )
    if set(input_rows) != set(halfset_rows):
        missing = len(set(input_rows) - set(halfset_rows))
        extra = len(set(halfset_rows) - set(input_rows))
        raise ValueError(
            "RELION half-set STAR and RECOVAR input STAR do not contain the "
            "same rlnImageName/stack identities "
            f"(missing={missing}, extra={extra})",
        )

    n_particles = len(input_particles)
    if len(input_rows) != n_particles:
        raise ValueError("RECOVAR input STAR identity count does not match its particle rows")

    half_indices = []
    for label, values in (("half1_idx", half1_idx), ("half2_idx", half2_idx)):
        indices = np.asarray(values, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional, got {indices.shape}")
        if indices.size and (
            int(np.min(indices)) < 0 or int(np.max(indices)) >= n_particles
        ):
            raise ValueError(f"{label} contains an out-of-bounds RECOVAR particle row")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{label} contains duplicate RECOVAR particle rows")
        half_indices.append(indices)
    if np.intersect1d(half_indices[0], half_indices[1]).size:
        raise ValueError("half1_idx and half2_idx overlap")
    combined_indices = np.concatenate(half_indices)
    if combined_indices.size != n_particles or not np.array_equal(
        np.sort(combined_indices),
        np.arange(n_particles, dtype=np.int64),
    ):
        raise ValueError(
            "half1_idx and half2_idx must form an exact partition of the input STAR rows",
        )

    if "rlnRandomSubset" not in relion_halfset_particles.columns:
        raise ValueError("RELION half-set STAR is missing rlnRandomSubset")
    random_subsets = np.asarray(
        relion_halfset_particles["rlnRandomSubset"],
        dtype=np.int64,
    ).reshape(-1)
    if random_subsets.shape != (len(relion_halfset_particles),):
        raise ValueError("RELION half-set rlnRandomSubset has an invalid shape")
    if not np.all(np.isin(random_subsets, (1, 2))):
        raise ValueError("RELION half-set rlnRandomSubset values must be 1 or 2")

    identities_by_input_row = [None] * n_particles
    for identity, row in input_rows.items():
        identities_by_input_row[row] = identity
    for half, indices in enumerate(half_indices, start=1):
        supplied_subsets = np.asarray(
            [random_subsets[halfset_rows[identities_by_input_row[int(row)]]] for row in indices],
            dtype=np.int64,
        )
        if not np.all(supplied_subsets == half):
            bad_rows = indices[supplied_subsets != half]
            raise ValueError(
                f"half{half}_idx disagrees with RELION rlnRandomSubset for "
                f"{bad_rows.size} input rows",
            )

    # relion_refine sets each absent angle label to 0 when it reads the input
    # (Experiment::read, exp_model.cpp:1104-1136), so a STAR without angles is
    # seeded with zeros rather than rejected.
    angle_columns = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    eulers = np.zeros((n_particles, len(angle_columns)), dtype=np.float64)
    for axis, column in enumerate(angle_columns):
        if column in input_particles.columns:
            eulers[:, axis] = _input_numeric_columns(
                input_particles, (column,), field="Euler-angle"
            )[:, 0]

    translations, translation_units = _input_star_origins_pixels(
        input_particles,
        voxel_size=voxel_size,
    )

    # relion_refine reads rlnNormCorrection as each particle's starting norm
    # correction and uses 1 when the label is absent (exp_model.cpp:1112,1145;
    # ml_optimiser.cpp:12376-12377).
    if "rlnNormCorrection" in input_particles.columns:
        norm_corrections = _input_numeric_columns(
            input_particles, ("rlnNormCorrection",), field="norm-correction"
        )[:, 0]
        if np.any(norm_corrections <= 0.0):
            raise ValueError("RECOVAR input STAR rlnNormCorrection values must be positive")
    else:
        norm_corrections = np.ones(n_particles, dtype=np.float64)

    # rlnAngle*Prior centre --sigma_ang searches below the local-search order (local_search_centre_half);
    # relion_refine marks an absent prior 999 (ml_optimiser.cpp, getFourierTransformsAndCtfs).
    prior_columns = ("rlnAngleRotPrior", "rlnAngleTiltPrior", "rlnAnglePsiPrior")
    angle_priors = None
    if any(column in input_particles.columns for column in prior_columns):
        angle_priors = np.full((n_particles, 3), np.nan, dtype=np.float64)
        for axis, column in enumerate(prior_columns):
            if column in input_particles.columns:
                values = _input_numeric_columns(input_particles, (column,), field="angle-prior")[:, 0]
                angle_priors[:, axis] = np.where(np.abs(values - 999.0) < 0.01, np.nan, values)

    eulers_per_half = [
        np.ascontiguousarray(eulers[indices], dtype=np.float32)
        for indices in half_indices
    ]
    translations_per_half = [
        np.ascontiguousarray(translations[indices], dtype=np.float32)
        for indices in half_indices
    ]
    for half, (half_eulers, half_translations, indices) in enumerate(
        zip(eulers_per_half, translations_per_half, half_indices, strict=True),
        start=1,
    ):
        if half_eulers.shape != (indices.size, 3):
            raise ValueError(f"half-{half} input Euler array has an invalid shape")
        if half_translations.shape != (indices.size, translations.shape[1]):
            raise ValueError(f"half-{half} input translation array has an invalid shape")
        if not np.all(np.isfinite(half_eulers)) or not np.all(np.isfinite(half_translations)):
            raise ValueError(f"half-{half} input poses are not finite after float32 conversion")

    return {
        "iteration": "input_star",
        "previous_best_rotation_eulers": eulers_per_half,
        "previous_best_translations": translations_per_half,
        "translation_units": translation_units,
        "norm_corrections": [
            np.ascontiguousarray(norm_corrections[indices], dtype=np.float64)
            for indices in half_indices
        ],
        "angle_priors": None if angle_priors is None else [angle_priors[indices] for indices in half_indices],
    }


def _load_input_star_class3d_translations(input_particles, rows, *, voxel_size: float, with_orientations: bool = False):
    """Return a fresh Class3D run's input origins in its all-data particle order.

    ``with_orientations`` (RELION ``--skip_align``) also returns the input Euler angles: the run
    classifies at the input poses instead of searching (absent angle labels read as zero, as
    relion_refine reads them).

    Class3D rounds and applies the input origins before the image FFT but does
    not centre its first global search on the input orientations (see
    ``_kclass_firstiter_translation_seed``), so only translations are
    returned: ``rows`` indexes the input STAR in the order of the single
    all-data accumulator, and the second accumulator is empty. With local
    searches from the start (``--sigma_ang``, ``with_angles``) the input angles
    are the searches' centres, 0 where the STAR has none, as relion_refine reads them.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n_particles = len(input_particles)
    if rows.ndim != 1:
        raise ValueError(f"Class3D particle rows must be one-dimensional, got {rows.shape}")
    if not np.array_equal(np.sort(rows), np.arange(n_particles, dtype=np.int64)):
        raise ValueError("Class3D particle rows must be a permutation of the input STAR rows")
    translations, translation_units = _input_star_origins_pixels(
        input_particles,
        voxel_size=voxel_size,
    )
    selected = np.ascontiguousarray(translations[rows], dtype=np.float32)
    if not np.all(np.isfinite(selected)):
        raise ValueError("Class3D input origins are not finite after float32 conversion")
    eulers = [None, None]
    if with_orientations:
        angles = np.zeros((n_particles, 3), dtype=np.float64)
        for axis, column in enumerate(("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")):
            if column in input_particles.columns:
                angles[:, axis] = _input_numeric_columns(input_particles, (column,), field="Euler-angle")[:, 0]
        eulers = [np.ascontiguousarray(angles[rows], dtype=np.float32), np.empty((0, 3), dtype=np.float32)]
        if not np.all(np.isfinite(eulers[0])):
            raise ValueError("Class3D input Euler angles are not finite")
    # relion_refine reads rlnNormCorrection for Class3D as for auto-refine (Experiment::read) and
    # scales each image by avg_norm / normcorr from iteration 1; 1 where the label is absent.
    norm_corrections = np.ones(n_particles, dtype=np.float64)
    if "rlnNormCorrection" in input_particles.columns:
        norm_corrections = _input_numeric_columns(input_particles, ("rlnNormCorrection",), field="norm-correction")[:, 0]
        if np.any(norm_corrections <= 0.0):
            raise ValueError("RECOVAR input STAR rlnNormCorrection values must be positive")
    return {
        "iteration": "input_star" if with_orientations else "input_star_translation_only",
        "previous_best_rotation_eulers": eulers,
        "previous_best_translations": [selected, np.empty((0, 2), dtype=np.float32)],
        "translation_units": translation_units,
        "norm_corrections": [np.ascontiguousarray(norm_corrections[rows]), np.empty(0, dtype=np.float64)],
    }


def _initial_corrections_from_norm(norm_corrections_per_half):
    """Return the start-up ``(image_corrections, scale_corrections)`` half pairs.

    RELION starts ``avg_norm_correction`` at 1 (ml_optimiser.cpp:1268) and every
    group scale at 1, so the scoring operand ``(avg_norm / normcorr) * scale``
    is ``1 / normcorr`` and the scale operand is 1. Unit norms return
    ``(None, None)``, the refinement's representation of unit corrections.
    """

    halves = [np.asarray(values, dtype=np.float64).reshape(-1) for values in norm_corrections_per_half]
    if len(halves) != 2:
        raise ValueError("norm corrections must be a two-half pair")
    if all(np.all(half == 1.0) for half in halves):
        return None, None
    image_corrections = [np.asarray(1.0 / half, dtype=np.float32) for half in halves]
    scale_corrections = [np.ones(half.shape, dtype=np.float32) for half in halves]
    return image_corrections, scale_corrections


def _add_initial_pose_source_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--initial-pose-source",
        choices=("auto", "input-star", "none"),
        default="auto",
        help=(
            "Initial previous-best poses for a fresh refinement. 'auto' "
            "loads K=1 Euler angles and origins from <data_dir>/particles.star when "
            "--relion_half_sets or --relion-half-sets-from-input supplies the matched "
            "random halves (absent angles are 0, as in relion_refine), and for a fresh "
            "Class3D (K>1) run loads only the input origins, which RELION applies "
            "before the first global search (absent origins are 0); 'input-star' "
            "requires that production path explicitly; 'none' preserves an unseeded "
            "search. Diagnostic/replay pose sources retain ownership."
        ),
    )


def _resolve_input_star_pose_seed(
    requested_source: str,
    *,
    n_classes: int,
    init_relion_iteration: int,
    has_relion_half_sets: bool,
    has_competing_pose_source: bool,
    diagnostic_single_half: bool,
) -> bool:
    """Resolve the production input-STAR pose seed or reject ambiguous use."""

    source = str(requested_source).strip().lower()
    if source not in {"auto", "input-star", "none"}:
        raise ValueError(f"unsupported initial pose source {requested_source!r}")
    if source == "none":
        return False

    class3d = int(n_classes) > 1
    incompatibilities = []
    if int(init_relion_iteration) != 0:
        incompatibilities.append("it requires a fresh --init_relion_iteration 0 run")
    if class3d and bool(has_relion_half_sets):
        incompatibilities.append("Class3D splits no random halves; drop --relion_half_sets")
    if not class3d and not bool(has_relion_half_sets):
        incompatibilities.append("it requires --relion_half_sets")
    if bool(has_competing_pose_source):
        incompatibilities.append("a diagnostic/replay pose source already owns initialization")
    if bool(diagnostic_single_half):
        incompatibilities.append("it requires both gold-standard halves")

    if source == "auto":
        return not incompatibilities
    if incompatibilities:
        raise ValueError("--initial-pose-source input-star " + "; ".join(incompatibilities))
    return True


def _kclass_firstiter_translation_seed(
    initial_override,
    *,
    n_classes,
    init_relion_iteration,
):
    """Select RELION's run_it000 origins for a fresh Class3D search (debug path).

    This serves ``--relion_init_dir`` comparisons only; a standalone run reads
    the same origins from the input STAR
    (``_load_input_star_class3d_translations``). RELION Class3D does not center its first global angular search on the
    input orientations, but it does round and apply the input origins before
    taking the image FFT.  Keep those two pieces of state independent: this
    helper deliberately returns translations only and cannot expose the
    orientations, normalization corrections, priors, or noise carried by the
    broader replay override.
    """

    if int(n_classes) <= 1 or int(init_relion_iteration) != 0:
        return None
    if initial_override is None:
        raise ValueError("fresh Class3D translation initialization is missing run_it000 state")
    translations = initial_override.get("previous_best_translations")
    if not isinstance(translations, (list, tuple)) or len(translations) != 2:
        raise ValueError("fresh Class3D translation initialization requires two half arrays")

    selected = []
    for half_index, values in enumerate(translations, start=1):
        if values is None:
            raise ValueError(
                "fresh Class3D translation initialization is missing "
                f"half-{half_index} input origins"
            )
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            # A single all-data Class3D process owns an intentionally empty
            # second accumulator. Generic replay extraction loses the
            # trailing coordinate dimension when indexing that empty half.
            array = np.empty((0, 2), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                "fresh Class3D half-"
                f"{half_index} input origins have shape {array.shape}; expected (N, 2)"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"fresh Class3D half-{half_index} input origins contain non-finite values"
            )
        selected.append(np.ascontiguousarray(array).copy())
    return selected


def _zero_local_search_centres(input_particles, particle_layout: "ParticleLayout", pixel_size_angstrom: float) -> dict:
    """Previous-best poses of (0, 0, 0) and zero origins in each half's order, as many origin axes as the input."""
    origins, _ = _input_star_origins_pixels(input_particles, voxel_size=pixel_size_angstrom)
    halves = (particle_layout.half1_rows, particle_layout.half2_rows)
    return {
        "iteration": "000_zero_angles",
        "previous_best_rotation_eulers": [np.zeros((len(rows), 3), dtype=np.float64) for rows in halves],
        "previous_best_translations": [np.zeros((len(rows), origins.shape[1]), dtype=np.float64) for rows in halves],
        "translation_units": "pixel",
    }


def prepare_initial_poses(
    input_particles,
    *,
    particle_layout: "ParticleLayout",
    relion_halfset_particles,
    pixel_size_angstrom: float,
    data_dir,
    requested_source: str,
    n_classes: int,
    init_relion_iteration: int,
    has_relion_half_sets: bool,
    diagnostic_single_half: bool,
    frozen_boundary,
    poses_npz_path,
    pose_iteration,
    has_replay_pose_source: bool,
    class3d_translations,
    class3d_translation_path,
    local_search_at_start: bool = False,
    log: Logger,
    skip_align: bool = False,
) -> InitialPoses:
    """Select and load startup poses, their corrections and source provenance.

    ``local_search_at_start``: iteration 1 is a local angular search (``--sigma_ang``, or a K=1 start at
    or above ``--auto_local_healpix_order``). relion_refine then centres each search on the input angles,
    which are 0 where the STAR has none; a start without any pose source is centred at (0, 0, 0) and zero
    origins the same way, with a warning.

    See ``docs/math/relion_refinement_algorithm.md#startup-particle-state-and-norm-corrections``.
    """
    has_competing_initial_pose_source = (
        frozen_boundary is not None
        or poses_npz_path is not None
        or has_replay_pose_source
    )
    try:
        use_input_star_pose_seed = _resolve_input_star_pose_seed(
            requested_source,
            n_classes=n_classes,
            init_relion_iteration=init_relion_iteration,
            has_relion_half_sets=has_relion_half_sets,
            has_competing_pose_source=has_competing_initial_pose_source,
            diagnostic_single_half=diagnostic_single_half,
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid initial pose source: {exc}") from exc

    resolved_initial_pose_source = "diagnostic_replay" if has_competing_initial_pose_source else "none"
    initial_pose_source_path = None
    initial_pose_source_sha256 = None
    init_previous_best_poses = None
    initial_image_corrections = None
    initial_scale_corrections = None
    if frozen_boundary is not None:
        init_previous_best_poses = {
            "iteration": f"{frozen_boundary.completed_relion_iteration - 1:03d}",
            "previous_best_rotation_eulers": list(
                frozen_boundary.previous_best_rotation_eulers
            ),
            "previous_best_translations": list(
                frozen_boundary.previous_best_translations
            ),
        }
    elif poses_npz_path is not None:
        init_previous_best_poses = iteration_history._load_init_previous_best_poses_npz(
            poses_npz_path,
            pose_iteration,
        )
        log.info(
            "Diagnostic local-search seed: loaded previous best poses from %s (iter=%s; half sizes=%s)",
            poses_npz_path,
            init_previous_best_poses["iteration"],
            [
                arr.shape[0]
                for arr in init_previous_best_poses["previous_best_rotation_eulers"]
            ],
        )

    elif class3d_translations is not None:
        init_previous_best_poses = {
            "iteration": "000_translation_only",
            # Local searches from iteration 1 (--sigma_ang) are centred on the input angles, 0 where the STAR
            # has none (exp_model.cpp:1104-1134, ml_optimiser.cpp:978-983), in the all-data accumulator's order.
            "previous_best_rotation_eulers": (
                _load_input_star_class3d_translations(
                    input_particles, particle_layout.half1_rows, voxel_size=pixel_size_angstrom,
                    with_orientations=True,
                )["previous_best_rotation_eulers"]
                if local_search_at_start
                else [None, None]
            ),
            "previous_best_translations": class3d_translations,
        }
        resolved_initial_pose_source = "relion_run_it000_translations"
        initial_pose_source_path = class3d_translation_path
        initial_pose_source_sha256 = _sha256_file(initial_pose_source_path)
        log.info(
            "Production fresh Class3D translation initialization: source=%s "
            "sha256=%s half_sizes=%s (orientations intentionally unset)",
            initial_pose_source_path,
            initial_pose_source_sha256,
            [arr.shape[0] for arr in class3d_translations],
        )
    elif use_input_star_pose_seed and n_classes > 1:
        input_pose_path = (Path(data_dir) / "particles.star").resolve()
        try:
            init_previous_best_poses = _load_input_star_class3d_translations(
                input_particles,
                particle_layout.half1_rows,
                voxel_size=pixel_size_angstrom,
                # --skip_align classifies at the input poses; --sigma_ang centres its local searches on them.
                with_orientations=skip_align or local_search_at_start,
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid input-STAR Class3D origin initialization: {exc}") from exc
        initial_image_corrections, initial_scale_corrections = _initial_corrections_from_norm(
            init_previous_best_poses["norm_corrections"],
        )
        resolved_initial_pose_source = "input_star" if local_search_at_start else "input_star_translations"
        initial_pose_source_path = input_pose_path
        initial_pose_source_sha256 = _sha256_file(input_pose_path)
        log.info(
            "Fresh Class3D translation initialization: source=%s sha256=%s "
            "translation_units=%s particles=%d norm_corrections=%s (%s)",
            input_pose_path,
            initial_pose_source_sha256,
            init_previous_best_poses["translation_units"],
            init_previous_best_poses["previous_best_translations"][0].shape[0],
            "unit" if initial_image_corrections is None else "from input rlnNormCorrection",
            "orientations kept: --skip_align"
            if skip_align
            else "orientations: the local searches' centres"
            if local_search_at_start
            else "orientations intentionally unset",
        )
    elif use_input_star_pose_seed:
        input_pose_path = (Path(data_dir) / "particles.star").resolve()
        try:
            init_previous_best_poses = _load_input_star_previous_best_poses(
                input_particles,
                relion_halfset_particles,
                particle_layout.half1_rows,
                particle_layout.half2_rows,
                voxel_size=pixel_size_angstrom,
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid input-STAR pose initialization: {exc}") from exc
        initial_image_corrections, initial_scale_corrections = _initial_corrections_from_norm(
            init_previous_best_poses["norm_corrections"],
        )
        resolved_initial_pose_source = "input_star"
        initial_pose_source_path = input_pose_path
        initial_pose_source_sha256 = _sha256_file(input_pose_path)
        log.info(
            "Production fresh-run pose initialization: source=%s sha256=%s "
            "translation_units=%s half_sizes=%s norm_corrections=%s",
            input_pose_path,
            initial_pose_source_sha256,
            init_previous_best_poses["translation_units"],
            [
                arr.shape[0]
                for arr in init_previous_best_poses["previous_best_rotation_eulers"]
            ],
            "unit" if initial_image_corrections is None else "from input rlnNormCorrection",
        )

    if local_search_at_start:
        absent = [c for c in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi") if c not in input_particles.columns]
        if init_previous_best_poses is None:
            init_previous_best_poses = _zero_local_search_centres(input_particles, particle_layout, pixel_size_angstrom)
            resolved_initial_pose_source = "zero_angles"
            log.warning(
                "Local angular searches from iteration 1 without a pose source: every search is centred at "
                "Euler angles (0, 0, 0) with zero origins, as relion_refine centres particles without input angles"
            )
        elif resolved_initial_pose_source == "input_star" and absent:
            log.warning(
                "Local angular searches from iteration 1: the input STAR lacks %s, so those angles are 0 "
                "for every particle and the searches are centred there, as in relion_refine",
                ", ".join(absent),
            )

    return InitialPoses(
        poses=init_previous_best_poses,
        image_corrections=initial_image_corrections,
        scale_corrections=initial_scale_corrections,
        provenance=PoseProvenance(
            requested_source=requested_source,
            resolved_source=resolved_initial_pose_source,
            path=initial_pose_source_path,
            sha256=initial_pose_source_sha256,
        ),
    )
