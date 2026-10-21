"""The particle table's update from one E-step's report.

What RELION writes per particle after an E-step (``_data.star``): the winning pose and offsets, class,
Pmax, significant-sample count and log-likelihood contribution. The driver calls these once per iteration.
"""

from __future__ import annotations

import numpy as np

from relax.sampling.orientation_priors import relion_round_away_from_zero
from relax.vdam.state import NativeParticleState


def _ensure_field(arr: np.ndarray | None, shape: tuple, dtype, fill=0) -> np.ndarray:
    if arr is None or arr.shape != shape:
        return np.full(shape, fill, dtype=dtype) if fill != 0 else np.zeros(shape, dtype=dtype)
    return arr


def update_particle_state_from_estep_meta(
    particle_state: NativeParticleState,
    meta: dict,
    translations: np.ndarray,
) -> None:
    selected = meta.get("selected_particle_ids")
    if selected is None:
        return
    ids = np.asarray(selected, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return
    N = particle_state.translation_offsets.shape[0]
    if np.any(ids < 0) or np.any(ids >= N):
        raise ValueError("selected_particle_ids contains entries outside the particle state table")
    particle_state.visited = _ensure_field(particle_state.visited, (N,), bool)
    particle_state.visited[ids] = True

    if (pose := meta.get("pose_assignments")) is not None:
        assignments = np.asarray(pose, dtype=np.int64).reshape(-1)
        trans = np.asarray(translations, dtype=np.float64)
        translation_ids = np.mod(assignments, int(trans.shape[0]))
        base = relion_round_away_from_zero(particle_state.translation_offsets[ids])
        particle_state.translation_offsets[ids] = base + trans[translation_ids, :2]
        particle_state.pose_assignments = _ensure_field(particle_state.pose_assignments, (N,), np.int32, -1)
        particle_state.pose_assignments[ids] = assignments.astype(np.int32, copy=False)

    if (offsets := meta.get("tomo_offsets_px", meta.get("image_offsets_px"))) is not None:
        # Subtomogram particles (3D) and images on several shapes (rounded in each image's own
        # pixels): RELION's new offset, the rounded old one plus the winning shift.
        particle_state.translation_offsets[ids] = np.asarray(offsets, dtype=np.float64)

    if (rot := meta.get("best_pose_rotations")) is not None:
        particle_state.best_pose_rotations = _ensure_field(particle_state.best_pose_rotations, (N, 3, 3), np.float32)
        particle_state.best_pose_rotations[ids] = np.asarray(rot, dtype=np.float32)

    source_eulers = meta.get("best_pose_eulers_deg")
    if rot is not None or source_eulers is not None:
        particle_state.best_pose_eulers_valid = _ensure_field(particle_state.best_pose_eulers_valid, (N,), bool, False)
        particle_state.best_pose_eulers_valid[ids] = False
    if source_eulers is not None:
        eulers = np.asarray(source_eulers)
        if eulers.dtype != np.float64 or eulers.shape != (ids.size, 3) or not np.all(np.isfinite(eulers)):
            raise ValueError("source Euler metadata must be finite float64 [selected_particles, 3]")
        particle_state.best_pose_eulers_deg = _ensure_field(particle_state.best_pose_eulers_deg, (N, 3), np.float64)
        particle_state.best_pose_eulers_deg[ids] = eulers
        valid = np.asarray(meta.get("best_pose_eulers_valid", np.ones(ids.size, dtype=bool)))
        if valid.dtype != bool or valid.shape != (ids.size,):
            raise ValueError("source Euler validity must be boolean [selected_particles]")
        particle_state.best_pose_eulers_valid[ids] = valid

    if (cls := meta.get("class_assignments")) is not None:
        particle_state.class_assignments[ids] = np.asarray(cls, dtype=np.int32).reshape(-1)

    if (pmax := meta.get("max_posterior_per_image")) is not None:
        particle_state.max_posterior[ids] = np.asarray(pmax, dtype=np.float32).reshape(-1)

    if (dll := meta.get("log_likelihood_contribution")) is not None:
        particle_state.log_likelihood_contribution = _ensure_field(particle_state.log_likelihood_contribution, (N,), np.float64, 0.0)
        particle_state.log_likelihood_contribution[ids] = np.asarray(dll, dtype=np.float64).reshape(-1)
    if (nsig := meta.get("significant_counts")) is not None:
        particle_state.significant_counts = _ensure_field(particle_state.significant_counts, (N,), np.int32, 0)
        particle_state.significant_counts[ids] = np.asarray(nsig, dtype=np.int32).reshape(-1)


def add_log_likelihood_contributions(meta: dict, state, optics_state, optics_group_ids, tilt_images) -> None:
    """Add RELION's per-particle dLL to the E-step's ``meta`` when it carries the log evidence."""

    if (evidence := meta.get("log_evidence_per_image")) is None:
        return
    ids = np.asarray(meta["selected_particle_ids"], dtype=np.int64)
    from relax.relion.log_likelihood import relion_log_likelihood_contributions

    meta["log_likelihood_contribution"] = relion_log_likelihood_contributions(
        evidence,
        sigma2_noise=state.sigma2_noise,
        groups=np.zeros(ids.size, np.int64) if optics_group_ids is None else np.asarray(optics_group_ids)[ids],
        n_images=np.ones(ids.size) if tilt_images is None else np.diff(tilt_images.image_offsets)[ids],
        box_size=int(state.box_size),
        current_size=int(state.effective_current_size),
        group_grids=optics_group_grids(optics_state, optics_group_ids, int(state.box_size)),
    )


def optics_group_grids(optics_state, optics_group_ids, model_box):
    """Each optics group's ``(box, scale, remap)`` on several image shapes, else None (one grid), by
    :func:`relax.relion.optics_groups.optics_group_grid`, from the images' own boxes and pixel sizes."""

    if optics_state is None or optics_state.image_box is None or optics_group_ids is None:
        return None
    from relax.relion.optics_groups import optics_group_grid

    groups = np.asarray(optics_group_ids)
    grids = {}
    for group in np.unique(groups):
        rows = groups == group
        box = int(np.unique(np.asarray(optics_state.image_box)[rows]).item())
        pixel = float(np.unique(np.asarray(optics_state.image_pixel_size)[rows]).item())
        grids[int(group)] = optics_group_grid(box, pixel, model_box, float(optics_state.pixel_size))
    return grids
