"""Scoring payloads and per-half adapters for dense and local refinement.

The controller owns the lifetime of these containers and their array buffers.
Adapters preserve engine layouts and normalize only the fields documented by
their existing casts and coarse-grid reductions.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.helpers.half_volume_mstep import (
    half_volume_accumulator_shape,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
)
from relax.helpers.types import make_noise_stats, make_relion_stats
from relax.relion.geometry import RECONSTRUCTION_PADDING_FACTOR


@dataclass(frozen=True, kw_only=True)
class ClassScoreSummary:
    """Assignments and prior/noise statistics for one half's class model."""

    assignments: np.ndarray
    mstep_mass: np.ndarray
    evidence_mass: np.ndarray
    rotation_mass: np.ndarray
    noise_stats: object | None


@dataclass
class HalfScoreResult:
    """Scoring result for one halfset, from a dense or local engine.

    Per-image fields follow the halfset dataset's local image order. ``ha``
    encodes pose assignments in the scoring grid; adaptive paths may also
    provide ``coarse_ha`` and explicit best poses for downstream pose export.
    Class assignments and posterior summaries accompany K-class results.

    Accumulators retain the engine's layout and device until reconstruction
    or explicit host offloading. Interpret them with ``mstep_full_half_axis``
    and ``mstep_accumulator_shape``; K-class results retain their class axis.
    This container does not copy arrays or standardize their precision.
    """

    # Always populated by every scoring branch.
    ha: np.ndarray
    Ft_y: object
    Ft_ctf: object
    em_stats: object
    noise_stats: object

    # Local-search and explicit best-pose paths.
    best_pose_rotations: np.ndarray | None = None
    best_pose_rotation_eulers: np.ndarray | None = None
    best_pose_translations: np.ndarray | None = None

    # Diagnostics emitted by some branches.
    coarse_ha: np.ndarray | None = None
    pose_rotations: object | None = None
    pose_rotation_eulers: object | None = None
    significant_counts: np.ndarray | None = None
    profile_summary: dict | None = None
    mstep_full_half_axis: int | None = None
    mstep_accumulator_shape: tuple[int, int, int] | None = None

    classes: ClassScoreSummary | None = None
    translation_search_base: object | None = None


def empty_half_result(
    *,
    volume_shape,
    padded_volume_shape,
    n_classes: int,
    n_shells: int,
    n_rotations: int,
    translation_dimension: int,
    image_window_size: int | None,
    model_support_size: int | None,
    use_x_half_mstep: bool,
) -> HalfScoreResult:
    """Zero expectation payload with the selected reconstruction layout."""
    accumulator_shape = (
        relion_backprojector_volume_shape(
            volume_shape,
            RECONSTRUCTION_PADDING_FACTOR,
            current_size=image_window_size if model_support_size is None else model_support_size,
        )
        if use_x_half_mstep else None
    )
    if n_classes > 1:
        Ft_y = Ft_ctf = None
    elif use_x_half_mstep:
        x_half_shape = half_volume_accumulator_shape(accumulator_shape)
        flat_x_half_shape = (int(np.prod(x_half_shape)),)
        Ft_y_x_half = jnp.zeros(flat_x_half_shape, dtype=jnp.complex128)
        Ft_ctf_x_half = jnp.zeros(flat_x_half_shape, dtype=jnp.complex128)
        Ft_y, Ft_ctf = relion_x_half_accumulators_to_public_layout(
            Ft_y_x_half, Ft_ctf_x_half, accumulator_shape,
        )
    else:
        flat_padded_shape = (int(np.prod(padded_volume_shape)),)
        Ft_y = jnp.zeros(flat_padded_shape, dtype=jnp.complex128)
        Ft_ctf = jnp.zeros(flat_padded_shape, dtype=jnp.complex128)
    assignments = np.zeros(0, dtype=np.int32)
    classes = ClassScoreSummary(
        assignments=np.zeros(0, dtype=np.int32),
        mstep_mass=np.zeros(n_classes, dtype=np.float32),
        evidence_mass=np.zeros(n_classes, dtype=np.float32),
        rotation_mass=np.zeros((n_classes, n_rotations), dtype=np.float32),
        noise_stats=None,
    )
    em_stats = make_relion_stats(
        log_evidence_per_image=jnp.zeros(0, dtype=jnp.float32),
        best_log_score_per_image=jnp.zeros(0, dtype=jnp.float32),
        max_posterior_per_image=jnp.zeros(0, dtype=jnp.float32),
        rotation_posterior_sums=jnp.zeros(n_rotations, dtype=jnp.float32),
    )
    noise_stats = make_noise_stats(
        wsum_sigma2_noise=jnp.zeros(n_shells, dtype=jnp.float32),
        wsum_img_power=jnp.zeros(n_shells, dtype=jnp.float32),
        wsum_sigma2_offset=0.0,
        sumw=0.0,
    )
    return HalfScoreResult(
        ha=assignments,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        em_stats=em_stats,
        noise_stats=noise_stats,
        best_pose_rotations=np.zeros((0, 3, 3), dtype=np.float32),
        best_pose_rotation_eulers=np.zeros((0, 3), dtype=np.float32),
        best_pose_translations=np.zeros((0, translation_dimension), dtype=np.float32),
        mstep_full_half_axis=0 if use_x_half_mstep else None,
        mstep_accumulator_shape=accumulator_shape,
        classes=classes,
    )


def _host_offload_array(value):
    """Copy a retained accumulator to host memory and release its device buffer."""

    if isinstance(value, np.ndarray):
        return value
    host_value = np.asarray(jax.device_get(value))
    delete = getattr(value, "delete", None)
    if callable(delete):
        try:
            delete()
        except RuntimeError:
            pass
    return host_value


def _maybe_host_offload_half0_local_accumulators(
    *,
    half_index: int,
    use_local: bool,
    score_result: HalfScoreResult,
    log,
) -> HalfScoreResult:
    """Keep finished half-0 local accumulators off GPU while half 1 scores.

    Local search is K=1: the numbered controller rejects it for Class3D.
    """

    if int(half_index) != 0 or not use_local:
        return score_result
    if score_result.mstep_full_half_axis is None:
        return score_result

    ft_y_nbytes = int(np.size(score_result.Ft_y)) * int(np.dtype(getattr(score_result.Ft_y, "dtype")).itemsize)
    ft_ctf_nbytes = int(np.size(score_result.Ft_ctf)) * int(np.dtype(getattr(score_result.Ft_ctf, "dtype")).itemsize)
    score_result.Ft_y = _host_offload_array(score_result.Ft_y)
    score_result.Ft_ctf = _host_offload_array(score_result.Ft_ctf)
    log.info(
        "Offloaded half-1 local RELION M-step accumulators to host before scoring half-2 "
        "(Ft_y=%.2f GB, Ft_ctf=%.2f GB)",
        ft_y_nbytes / 1e9,
        ft_ctf_nbytes / 1e9,
    )
    return score_result


@dataclass(frozen=True)
class PerHalfOutputs:
    """Own the two halfset slots used during one scoring phase.

    Every field is a separate two-element list indexed by halfset (0 or 1),
    including fields whose names start with ``class_``. Per-image arrays stay
    in each halfset's local order; the two halfsets may have different sizes.
    Class counts and class/rotation summaries live inside their halfset slot.
    Unpopulated slots are ``None``; an empty scored half can hold empty arrays.

    ``frozen=True`` prevents rebinding fields. The controller deliberately aliases
    these lists and records complete scoring results with ``update_from``.
    """

    hard_assignments: list = field(default_factory=lambda: [None, None])
    Ft_y: list = field(default_factory=lambda: [None, None])
    Ft_ctf: list = field(default_factory=lambda: [None, None])
    coarse_ha: list = field(default_factory=lambda: [None, None])
    max_posterior: list = field(default_factory=lambda: [None, None])
    rotation_posterior: list = field(default_factory=lambda: [None, None])
    class_assignments: list = field(default_factory=lambda: [None, None])
    class_posterior: list = field(default_factory=lambda: [None, None])
    class_full_posterior: list = field(default_factory=lambda: [None, None])
    class_rotation_posterior: list = field(default_factory=lambda: [None, None])
    noise_stats: list = field(default_factory=lambda: [None, None])
    noise_stats_per_class: list = field(default_factory=lambda: [None, None])
    best_pose_rotations: list = field(default_factory=lambda: [None, None])
    best_pose_rotation_eulers: list = field(default_factory=lambda: [None, None])
    best_pose_translations: list = field(default_factory=lambda: [None, None])
    translation_search_bases: list = field(default_factory=lambda: [None, None])
    pose_rotations: list = field(default_factory=lambda: [None, None])
    pose_rotation_eulers: list = field(default_factory=lambda: [None, None])
    mstep_full_half_axis: list = field(default_factory=lambda: [None, None])
    mstep_accumulator_shape: list = field(default_factory=lambda: [None, None])

    def update_from(self, half_index: int, score_result: HalfScoreResult, *, dtype=np.float32) -> None:
        """Store one half's payload, retaining arrays except for posterior casts.

        Missing optional pose fields leave existing slot values intact. Layout
        metadata is always replaced, including ``None``. A K-class payload
        replaces its class summaries together, including absent noise statistics.
        """
        self.hard_assignments[half_index] = score_result.ha
        self.Ft_y[half_index] = score_result.Ft_y
        self.Ft_ctf[half_index] = score_result.Ft_ctf
        self.noise_stats[half_index] = score_result.noise_stats
        self.max_posterior[half_index] = np.asarray(
            score_result.em_stats.max_posterior_per_image,
            dtype=dtype,
        )
        self.rotation_posterior[half_index] = np.asarray(
            score_result.em_stats.rotation_posterior_sums,
            dtype=dtype,
        )
        if score_result.best_pose_rotations is not None:
            self.best_pose_rotations[half_index] = score_result.best_pose_rotations
        if score_result.best_pose_rotation_eulers is not None:
            self.best_pose_rotation_eulers[half_index] = score_result.best_pose_rotation_eulers
        if score_result.best_pose_translations is not None:
            self.best_pose_translations[half_index] = score_result.best_pose_translations
        if score_result.coarse_ha is not None:
            self.coarse_ha[half_index] = score_result.coarse_ha
        if score_result.pose_rotations is not None:
            self.pose_rotations[half_index] = score_result.pose_rotations
        if score_result.pose_rotation_eulers is not None:
            self.pose_rotation_eulers[half_index] = score_result.pose_rotation_eulers
        self.mstep_full_half_axis[half_index] = score_result.mstep_full_half_axis
        self.mstep_accumulator_shape[half_index] = score_result.mstep_accumulator_shape
        if score_result.classes is not None:
            self.class_assignments[half_index] = score_result.classes.assignments
            self.class_posterior[half_index] = score_result.classes.mstep_mass
            self.class_full_posterior[half_index] = score_result.classes.evidence_mass
            self.class_rotation_posterior[half_index] = score_result.classes.rotation_mass
            self.noise_stats_per_class[half_index] = score_result.classes.noise_stats


def class_em_to_half_result(
    k_class_result,
    *,
    effective_rotations,
    rot_pmap_for_collapse,
    adaptive_os_local: int,
    require_best_pose_details: bool = True,
    pose_dtype: np.dtype = np.float32,
) -> HalfScoreResult:
    """Adapt class EM statistics and canonical poses to one half's result.

    Class occupancy uses M-step mass; rotation sums use the coarse prior grid.
    The controller records the returned arrays in its half slots.
    """
    result = HalfScoreResult(
        ha=np.asarray(k_class_result.pose_assignments, dtype=np.int32),
        Ft_y=k_class_result.Ft_y,
        Ft_ctf=k_class_result.Ft_ctf,
        em_stats=k_class_result.stats,
        noise_stats=k_class_result.aggregate_noise_stats,
    )
    class_noise_stats = k_class_result.noise_stats
    class_assignments = np.asarray(k_class_result.class_assignments, dtype=np.int32)
    class_mass_for_priors = getattr(k_class_result, "class_mstep_posterior_sums", None)
    if class_mass_for_priors is None:
        class_mass_for_priors = k_class_result.class_posterior_sums
    class_posterior = np.asarray(class_mass_for_priors, dtype=np.float64)
    class_full_posterior = np.asarray(k_class_result.class_posterior_sums, dtype=np.float64)
    # Collapse fine-grid rotation posteriors to coarse via the parent map
    # when iter-1 firstiter_cc routes through the adaptive 2-pass engine
    # with adaptive_oversampling > 0; downstream
    # _combined_class_direction_prior_from_halves expects the coarse-grid
    # shape (n_rot_coarse,).
    n_rot_coarse = int(effective_rotations.shape[0])
    per_class_rot_post_coarse = []
    for stats in k_class_result.per_class_stats:
        rot_post = np.asarray(stats.rotation_posterior_sums, dtype=np.float64)
        if rot_post.shape[0] == n_rot_coarse:
            per_class_rot_post_coarse.append(rot_post)
        elif rot_pmap_for_collapse is not None and adaptive_os_local > 0:
            coarse_post = np.zeros(n_rot_coarse, dtype=np.float64)
            np.add.at(
                coarse_post,
                np.asarray(rot_pmap_for_collapse, dtype=np.int64),
                rot_post,
            )
            per_class_rot_post_coarse.append(coarse_post)
        else:
            raise RuntimeError(
                f"Unexpected K-class rotation_posterior_sums shape {rot_post.shape}; expected ({n_rot_coarse},)"
            )
    class_rotation_posterior = np.stack(per_class_rot_post_coarse, axis=0)
    result.classes = ClassScoreSummary(
        assignments=class_assignments,
        mstep_mass=class_posterior,
        evidence_mass=class_full_posterior,
        rotation_mass=class_rotation_posterior,
        noise_stats=class_noise_stats,
    )
    if require_best_pose_details:
        if k_class_result.best_pose_rotations is None or k_class_result.best_pose_translations is None:
            raise RuntimeError("Dense K-class path did not return best pose details")
        best_rots = np.asarray(k_class_result.best_pose_rotations, dtype=pose_dtype)
        result.best_pose_rotations = best_rots
        source_eulers = getattr(k_class_result, "best_pose_eulers_deg", None)
        result.best_pose_rotation_eulers = (
            np.asarray(source_eulers, dtype=np.float64)
            if source_eulers is not None
            else utils.R_to_relion(best_rots, degrees=True).astype(pose_dtype)
        )
        result.best_pose_translations = np.asarray(k_class_result.best_pose_translations, dtype=pose_dtype)
    return result


def _collapse_fine_pose_assignments_to_coarse(
    pose_assignments,
    *,
    rot_parent_map,
    trans_parent_map,
    n_trans_coarse: int,
    n_trans_fine: int,
):
    pose = np.asarray(pose_assignments, dtype=np.int64)
    rot_idx = pose // int(n_trans_fine)
    trans_idx = pose % int(n_trans_fine)
    coarse_rot = np.asarray(rot_parent_map, dtype=np.int64)[rot_idx]
    coarse_trans = np.asarray(trans_parent_map, dtype=np.int64)[trans_idx]
    return (coarse_rot * int(n_trans_coarse) + coarse_trans).astype(np.int32, copy=False)


def _select_single_class_accumulator(value, *, label: str):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2 or int(shape[0]) != 1:
        raise RuntimeError(f"K=1 adaptive {label} accumulator must have leading class axis 1; got {shape}")
    return value[0]


def _collapse_single_class_stats_to_coarse(stats, *, rot_parent_map, n_rot_coarse: int, dtype=np.float32):
    rot_post = np.asarray(stats.rotation_posterior_sums, dtype=np.float64)
    n_rot_coarse = int(n_rot_coarse)
    if rot_post.shape == (n_rot_coarse,):
        return stats
    if rot_parent_map is None:
        raise RuntimeError(
            f"K=1 adaptive rotation_posterior_sums has shape {rot_post.shape}; expected ({n_rot_coarse},)"
        )
    rot_parent = np.asarray(rot_parent_map, dtype=np.int64)
    if rot_post.shape != rot_parent.shape:
        raise RuntimeError(
            "K=1 adaptive rotation posterior and parent map disagree: "
            f"{rot_post.shape} vs {rot_parent.shape}"
        )
    coarse_post = np.zeros(n_rot_coarse, dtype=np.float64)
    np.add.at(coarse_post, rot_parent, rot_post)
    runtime_dtype = dtype
    return make_relion_stats(
        log_evidence_per_image=np.asarray(stats.log_evidence_per_image, dtype=runtime_dtype),
        best_log_score_per_image=np.asarray(stats.best_log_score_per_image, dtype=runtime_dtype),
        max_posterior_per_image=np.asarray(stats.max_posterior_per_image, dtype=runtime_dtype),
        rotation_posterior_sums=coarse_post.astype(runtime_dtype),
    )


def _record_score_profile(
    profile_history: list,
    score_result,
    *,
    phase: str,
    iteration: int,
    relion_iteration: int,
    half_index: int,
    current_size: int | None,
    healpix_order: int | None,
    k_class_enabled: bool,
) -> None:
    profile = getattr(score_result, "profile_summary", None)
    if not profile:
        return
    row = dict(profile)
    row.update(
        {
            "phase": str(phase),
            "iteration": np.int32(iteration),
            "relion_iteration": np.int32(relion_iteration),
            "half_index": np.int32(half_index),
            "current_size": np.int32(-1 if current_size is None else int(current_size)),
            "healpix_order": np.int32(-1 if healpix_order is None else int(healpix_order)),
            "k_class_enabled": bool(k_class_enabled),
        }
    )
    profile_history.append(row)


def _combine_optional_half_accumulators(left, right, *, label: str):
    """Combine half accumulators, treating an empty Class3D half as absent."""

    if left is None:
        if right is None:
            raise RuntimeError(f"{label} accumulators are missing for both halves")
        return right
    if right is None:
        return left
    return left + right


def _resolve_mstep_accumulator_shape(per_half_shapes, default_shape):
    """Return the common M-step accumulator shape for this iteration."""

    present = [tuple(int(v) for v in shape) for shape in per_half_shapes if shape is not None]
    if not present:
        return tuple(int(v) for v in default_shape)
    first = present[0]
    if any(shape != first for shape in present[1:]):
        raise RuntimeError(f"Per-half M-step accumulator shapes disagree: {present}")
    return first


def _resolve_mstep_full_half_axis(per_half_axes, default_axis=-1):
    """Return the common RELION half-complex axis for shell statistics."""

    present = [int(axis) for axis in per_half_axes if axis is not None]
    if not present:
        return int(default_axis)
    first = present[0]
    if any(axis != first for axis in present[1:]):
        raise RuntimeError(f"Per-half M-step full-half axes disagree: {present}")
    return first
