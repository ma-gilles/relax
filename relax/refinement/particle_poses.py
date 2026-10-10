"""The particles' poses across an iteration: the poses a pass resolved (``ParticlePoses``), the update the
halves take from them (``ParticlePoseUpdate``) and their comparison with the previous iteration's
(``PoseComparison``)."""

from dataclasses import dataclass

import numpy as np
from recovar import utils

from relax.refinement.refinement_state import concatenate_pose_stacks_or_none


def best_rotation_matrices(halves, *, dtype) -> list:
    """Each half's rotation matrices of its current best angles (None without angles; an empty half an empty
    ``(0, 3, 3)`` stack), in ``dtype``."""

    return [
        None
        if half.rotation_eulers is None
        else np.zeros((0, 3, 3), dtype=dtype)
        if len(half.rotation_eulers) == 0
        else np.asarray(utils.R_from_relion(np.asarray(half.rotation_eulers), degrees=True), dtype=dtype)
        for half in halves
    ]


@dataclass(frozen=True, eq=False)
class ParticlePoses:
    """One half's selected orientations and relative/absolute shifts in pixels."""

    rotations: np.ndarray
    eulers_deg: np.ndarray
    relative_translations_pixels: np.ndarray
    translations_pixels: np.ndarray


def resolve_particle_poses(
    assignments,
    translation_grid,
    *,
    previous_translations,
    own_pixel_factors,
    best_pose_rotations,
    best_pose_rotation_eulers,
    best_pose_translations,
    pose_rotations,
    local_sampling,
    dtype,
) -> ParticlePoses:
    """Resolve explicit engine poses or grid IDs into persistent particle poses.

    Supplied Euler metadata stays float64; deferred local matrices stay float32.
    ``own_pixel_factors`` are the images' reference -> own pixel factors
    (:func:`relax.refinement.optics_shapes.image_translation_factors`; None on one grid): the
    previous offset is rounded in each image's own pixels, as RELION does.
    See ``docs/math/relion_refinement_algorithm.md``, section 3.
    """
    from recovar import utils

    from relax.local_search.layout import selected_rotation_matrices
    from relax.relion.metadata import relion_metadata_translations
    from relax.sampling import build_local_search_grid_metadata

    if best_pose_rotations is not None:
        rotations = np.asarray(best_pose_rotations, dtype=dtype)
        eulers = (
            np.asarray(best_pose_rotation_eulers, dtype=np.float64)
            if best_pose_rotation_eulers is not None
            else utils.R_to_relion(rotations, degrees=True).astype(dtype)
        )
        translations = np.asarray(best_pose_translations, dtype=dtype)
    else:
        rotation_ids = assignments // translation_grid.shape[0]
        translation_ids = assignments % translation_grid.shape[0]
        if local_sampling is not None:
            if local_sampling.rotations is None:
                grid_metadata = build_local_search_grid_metadata(
                    local_sampling.search.healpix_order, symmetry=local_sampling.search.symmetry,
                )
                rotations = selected_rotation_matrices(
                    rotation_ids,
                    None,
                    grid_metadata,
                    random_perturbation=local_sampling.perturbation,
                    angular_sampling_deg=local_sampling.angular_step_deg,
                )
                eulers = utils.R_to_relion(np.asarray(rotations), degrees=True).astype(dtype)
            else:
                rotations = np.asarray(local_sampling.rotations, dtype=dtype)[rotation_ids]
                if local_sampling.rotation_eulers is not None:
                    eulers = np.asarray(local_sampling.rotation_eulers, dtype=dtype)[rotation_ids]
                else:
                    eulers = utils.R_to_relion(np.asarray(rotations), degrees=True).astype(dtype)
        else:
            rotations = np.asarray(pose_rotations, dtype=dtype)[rotation_ids]
            eulers = utils.R_to_relion(np.asarray(rotations), degrees=True).astype(dtype)
        translations = np.asarray(translation_grid)[translation_ids]
    return ParticlePoses(
        rotations=rotations,
        eulers_deg=eulers,
        relative_translations_pixels=translations,
        translations_pixels=relion_metadata_translations(
            previous_translations, translations, own_pixel_factors=own_pixel_factors, dtype=dtype,
        ),
    )


@dataclass(frozen=True, eq=False)
class ParticlePoseUpdate:
    """Previous pose snapshots and resolved poses for one numbered iteration."""

    previous_rotations: list[np.ndarray | None]
    previous_translations_pixels: list[np.ndarray | None]
    current: tuple[ParticlePoses, ParticlePoses]


@dataclass(frozen=True, eq=False)
class PoseComparison:
    """Half-ordered pose stacks consumed by convergence-change measurements."""

    current_rotations: np.ndarray | None
    previous_rotations: np.ndarray | None
    current_translations_pixels: np.ndarray | None
    previous_translations_pixels: np.ndarray | None


def prepare_particle_pose_update(
    scores,
    halves,
    translation_grid,
    *,
    previous_rotations,
    local_sampling,
    dtype,
) -> ParticlePoseUpdate:
    """Snapshot prior poses and resolve engine outputs without installing state.

    Snapshot both halves before resolving either. Canonical Euler metadata,
    relative search shifts and absolute particle offsets retain their existing
    precision and frames. See ``docs/math/relion_refinement_algorithm.md``,
    section 3. The controller owns publication and diagnostic ordering.
    """
    from relax.refinement.optics_shapes import image_translation_factors

    previous_rotations_snapshot = [
        np.asarray(rot).copy() if rot is not None else None for rot in previous_rotations
    ]
    previous_translations_snapshot = [
        np.asarray(half.translations).copy() if half.translations is not None else None
        for half in halves
    ]
    current = tuple(
        resolve_particle_poses(
            scores.hard_assignments[k],
            translation_grid,
            previous_translations=previous_translations_snapshot[k],
            own_pixel_factors=image_translation_factors(halves[k].dataset),
            best_pose_rotations=scores.best_pose_rotations[k],
            best_pose_rotation_eulers=scores.best_pose_rotation_eulers[k],
            best_pose_translations=scores.best_pose_translations[k],
            pose_rotations=scores.pose_rotations[k],
            local_sampling=local_sampling,
            dtype=dtype,
        )
        for k in range(2)
    )
    return ParticlePoseUpdate(
        previous_rotations=previous_rotations_snapshot,
        previous_translations_pixels=previous_translations_snapshot,
        current=current,
    )


def prepare_pose_comparison(
    pose_update: ParticlePoseUpdate,
    *,
    translation_dimension: int,
    dtype,
    log,
) -> PoseComparison:
    """Prepare aligned current/previous pose stacks in physical half order.

    Missing or malformed stacks retain the existing convergence primitive's
    refusal and logging behavior. Publication has already happened in the
    controller; this operation neither updates poses nor decides convergence.
    """
    current_rotations = concatenate_pose_stacks_or_none(
        [poses.rotations for poses in pose_update.current],
        trailing_shape=(3, 3),
        label="current rotation",
        dtype=dtype,
        logger=log,
    )
    previous_rotations = concatenate_pose_stacks_or_none(
        pose_update.previous_rotations,
        trailing_shape=(3, 3),
        label="previous rotation",
        dtype=dtype,
        logger=log,
    )
    current_translations_pixels = concatenate_pose_stacks_or_none(
        [poses.translations_pixels for poses in pose_update.current],
        trailing_shape=(translation_dimension,),
        label="current translation",
        dtype=dtype,
        logger=log,
    )
    previous_translations_pixels = concatenate_pose_stacks_or_none(
        pose_update.previous_translations_pixels,
        trailing_shape=(translation_dimension,),
        label="previous translation",
        dtype=dtype,
        logger=log,
    )

    return PoseComparison(
        current_rotations=current_rotations,
        previous_rotations=previous_rotations,
        current_translations_pixels=current_translations_pixels,
        previous_translations_pixels=previous_translations_pixels,
    )
