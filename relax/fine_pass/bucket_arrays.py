"""Per-image pass-2 hypotheses: each image's fine rotations, log priors and compact candidate mask.

Expands an image's significant coarse support to its oversampled children and gathers the rows. Preserves
particle order, precision and scoring/M-step aliases; it neither executes scoring nor chooses scientific or
device policies.
"""
from __future__ import annotations

import logging

import numpy as np

from relax.fine_pass.compact_candidates import SparseCandidateMask
from relax.fine_pass.significant_samples import ComplementSignificantSampleIndices
from relax.runtime.shape_buckets import pow2_ceil

logger = logging.getLogger(__name__)


def _rotation_table_key(table) -> tuple:
    """Content key for the device table cache: shape, dtype and a SHA-1 of the bytes."""

    import hashlib

    table = np.ascontiguousarray(table)
    return (table.shape, str(table.dtype), hashlib.sha1(table.view(np.uint8)).hexdigest())


def _children_of(parent, by_parent, offsets, parents) -> np.ndarray:
    """The fine rotations whose parent is in ``parents`` (unique ids), in ascending order. ``by_parent`` lists
    the fine rotations sorted by parent and ``offsets`` (``[n_coarse_rot + 1]``) where each parent's run
    starts; ``parent`` is each fine rotation's parent, for a selection that holds a large part of them."""

    starts = offsets[parents]
    counts = offsets[parents + 1] - starts
    n_children = int(counts.sum())
    if 16 * n_children > parent.size:  # most of the grid: one pass over it beats sorting the runs
        selected = np.zeros(offsets.size - 1, dtype=bool)
        selected[parents] = True
        return np.flatnonzero(selected[parent]).astype(np.int64)
    # Position j of parent p's run is starts[p] + j: each run's start, repeated, plus the position in the run.
    run_first = np.cumsum(counts) - counts
    positions = np.repeat(starts - run_first, counts) + np.arange(n_children, dtype=np.int64)
    return np.sort(by_parent[positions])


def relion_parent_execution_key(parent_ids, *, n_coarse_rot: int, nside_level: int) -> np.ndarray:
    """RELION's execution key for coarse rotation ids.

    RECOVAR's coarse grid is psi-slow and direction-fast
    (``id = psi * n_directions + direction``); RELION executes parents
    direction-major (``direction * n_psi + psi``). The direction count comes
    from the grid itself, ``n_coarse_rot / n_psi``, so a symmetry-reduced grid
    keeps its own direction count. The host pass-2 preparation and the resident
    CSR tables both order rows with this key.
    """

    from relax.sampling.grids import rotation_grid_n_in_planes

    n_psi = int(rotation_grid_n_in_planes(nside_level))
    if n_coarse_rot <= 0 or n_coarse_rot % n_psi:
        raise ValueError(
            f"RELION parent execution order needs whole psi rows: {n_coarse_rot} coarse "
            f"rotations with {n_psi} psi angles at healpix level {int(nside_level)}"
        )
    n_directions = n_coarse_rot // n_psi
    parent_ids = np.asarray(parent_ids, dtype=np.int64).reshape(-1)
    if parent_ids.size and (int(parent_ids.min()) < 0 or int(parent_ids.max()) >= n_coarse_rot):
        raise ValueError("RELION parent execution key is outside the coarse grid")
    return (parent_ids % n_directions) * n_psi + parent_ids // n_directions


def prepare_per_image_pass2_inputs(
    significant_sample_indices,
    n_coarse_rot,
    n_coarse_trans,
    nside_level,
    oversampling_order,
    n_fine_trans,
    fine_translation_parent,
    rotation_log_prior,
    random_perturbation,
    fine_source_eulers_override=None,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    relion_parent_execution_order=False,
    dtype: np.dtype = np.float32,
    symmetry_label: str = "C1",
    coarse_rotation_ids=None,
    per_image_rotation_log_prior=None,
):
    """Compute per-image oversampled rotations / parent maps / candidate masks.

    Builds each image's oversampled children exactly as the per-image reference
    pass 2 did (removed on 2026-10-03), so the batched path is a strict per-image
    equivalent.

    ``dtype`` controls the precision of every fine/oversampled rotation
    matrix this function builds or accepts: the RELION-supplied fine
    rotation override (``fine_rotations_override`` /
    ``fine_mstep_rotations_override``) *and* the standard
    ``get_oversampled_rotation_grid_from_samples`` grid built when no
    override is supplied. RELION's own fine-search rotation matrices stay
    ``RFLOAT`` (double) end to end in a double-precision build; pass
    ``precision_policy.score_real_dtype`` from the caller so this matches
    ``use_float64_scoring`` instead of always narrowing to float32.

    ``coarse_rotation_ids`` makes the coarse grid a compact subset of the HEALPix grid at
    ``nside_level`` (coarse id ``c`` is grid rotation ``coarse_rotation_ids[c]``; subtomogram
    local searches, relax.steps.tomo_half); RELION's parent execution order is then the
    grid's. ``per_image_rotation_log_prior`` replaces ``rotation_log_prior`` by one prior per
    image over that image's significant coarse rotations in ascending order (RELION's local
    orientation prior is the particle's own).
    """
    from relax.sampling.grids import rotation_grid_size
    from relax.sampling.oversampled_grids import get_oversampled_rotation_grid_from_samples

    if per_image_rotation_log_prior is not None and rotation_log_prior is not None:
        raise ValueError("a pass takes one shared rotation prior or one per image, not both")
    if coarse_rotation_ids is not None:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if coarse_rotation_ids.shape != (int(n_coarse_rot),):
            raise ValueError("coarse_rotation_ids must give one grid rotation per coarse rotation")
        grid_rotation_count = int(rotation_grid_size(nside_level, symmetry_label))
        if coarse_rotation_ids.size and (
            int(coarse_rotation_ids.min()) < 0 or int(coarse_rotation_ids.max()) >= grid_rotation_count
        ):
            raise ValueError("coarse_rotation_ids must index the HEALPix grid")
    from relax.sampling.symmetry import canonicalize_rotational_symmetry

    symmetry_label = canonicalize_rotational_symmetry(symmetry_label)
    # A compact coarse grid (coarse_rotation_ids) is checked against the symmetric grid above.
    if symmetry_label != "C1" and coarse_rotation_ids is None:
        expected_coarse_rot = rotation_grid_size(nside_level, symmetry_label)
        if int(n_coarse_rot) != int(expected_coarse_rot):
            raise ValueError(
                f"{symmetry_label} sparse pass-2 coarse rotation count mismatch: "
                f"{n_coarse_rot} != {expected_coarse_rot}"
            )

    n_images = len(significant_sample_indices)
    per_image_source_eulers = []
    per_image_oversampled_rots = []
    per_image_oversampled_mstep_rots = []
    per_image_parent_map = []
    per_image_oversampled_rot_indices = []
    per_image_unique_rot = []
    per_image_log_prior = []
    per_image_candidate_mask = []
    per_image_lists = {
        "source_eulers": per_image_source_eulers,
        "oversampled_rots": per_image_oversampled_rots,
        "oversampled_mstep_rots": per_image_oversampled_mstep_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
    }
    full_unique_rot = np.arange(n_coarse_rot, dtype=np.int32)
    full_support_rotation_cache = None
    full_support_log_prior_cache = None
    full_support_candidate_mask_cache = None

    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=dtype)
    else:
        rotation_log_prior_np = None

    fine_rotations_np = None
    fine_mstep_rotations_np = None
    fine_parent_np = None
    if fine_rotations_override is None and fine_rotation_parent_override is None:
        pass
    elif fine_rotations_override is not None and fine_rotation_parent_override is not None:
        fine_rotations_np = np.asarray(fine_rotations_override, dtype=dtype)
        fine_parent_np = np.asarray(fine_rotation_parent_override, dtype=np.int64)
        if fine_parent_np.ndim != 1:
            raise ValueError("fine_rotation_parent_override must be a 1D array")
        if fine_rotations_np.shape[0] != fine_parent_np.shape[0]:
            raise ValueError(
                "fine_rotations_override and fine_rotation_parent_override disagree on rotation count: "
                f"{fine_rotations_np.shape[0]} vs {fine_parent_np.shape[0]}",
            )
        if int(fine_parent_np.min(initial=0)) < 0 or int(fine_parent_np.max(initial=-1)) >= int(n_coarse_rot):
            raise ValueError("fine_rotation_parent_override values must be in [0, n_coarse_rot)")
        # The fine rotations by parent, once: an image takes its parents' children from here instead of testing
        # every fine rotation (74 s of a local pass 2 of the real tilt-series case went to that mask, py-spy).
        fine_by_parent = np.argsort(fine_parent_np, kind="stable")
        fine_parent_offsets = np.concatenate(([0], np.cumsum(np.bincount(fine_parent_np, minlength=n_coarse_rot))))
    else:
        raise ValueError("fine_rotations_override and fine_rotation_parent_override must be provided together")

    if fine_mstep_rotations_override is not None:
        if fine_rotations_np is None:
            raise ValueError("fine_mstep_rotations_override requires fine_rotations_override")
        fine_mstep_rotations_np = np.asarray(fine_mstep_rotations_override, dtype=dtype)
        if fine_mstep_rotations_np.shape != fine_rotations_np.shape:
            raise ValueError(
                "fine_mstep_rotations_override must match fine_rotations_override shape: "
                f"{fine_mstep_rotations_np.shape} vs {fine_rotations_np.shape}",
            )

    fine_source_eulers = None
    if fine_source_eulers_override is not None:
        fine_source_eulers = np.asarray(fine_source_eulers_override)
        if (
            fine_rotations_np is None
            or fine_source_eulers.dtype != np.float64
            or fine_source_eulers.shape != (len(fine_rotations_np), 3)
            or not np.all(np.isfinite(fine_source_eulers))
        ):
            raise ValueError(
                "fine source Euler metadata must match the supplied rotation rows as finite float64 triples"
            )

    def _reorder_children(rotations, parent_map, rotation_indices, source_eulers, parent_ids):
        if not relion_parent_execution_order:
            return rotations, parent_map, rotation_indices, source_eulers
        parent_ids = np.asarray(parent_ids, dtype=np.int64).reshape(-1)
        if parent_ids.shape != np.asarray(parent_map).shape:
            raise ValueError("RELION parent execution keys must match fine rotations")
        if coarse_rotation_ids is None:
            relion_parent_key = relion_parent_execution_key(
                parent_ids, n_coarse_rot=n_coarse_rot, nside_level=nside_level
            )
        else:
            relion_parent_key = relion_parent_execution_key(
                coarse_rotation_ids[parent_ids], n_coarse_rot=grid_rotation_count, nside_level=nside_level
            )
        order = np.argsort(relion_parent_key, kind="stable")
        return (
            np.asarray(rotations)[order],
            np.asarray(parent_map)[order],
            np.asarray(rotation_indices)[order],
            None if source_eulers is None else source_eulers[order],
        )

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        coarse_excluded = None
        if sig_samples is None:
            unique_rot = full_unique_rot
            use_full_candidate_mask = True
            use_full_rotation_support = True
            coarse_rot = unique_rot
            coarse_trans = None
        elif isinstance(sig_samples, ComplementSignificantSampleIndices):
            if int(sig_samples.total_size) != int(n_coarse_rot * n_coarse_trans):
                raise ValueError(
                    "Complement significant sample mask total_size does not match coarse pose grid: "
                    f"{int(sig_samples.total_size)} vs {int(n_coarse_rot * n_coarse_trans)}",
                )
            unique_rot = full_unique_rot
            use_full_candidate_mask = False
            use_full_rotation_support = True
            coarse_rot = unique_rot
            coarse_trans = None
            coarse_excluded = np.asarray(sig_samples.excluded_indices, dtype=np.int32).reshape(-1)
        else:
            use_full_candidate_mask = False
            use_full_rotation_support = False
            sig_samples = np.asarray(sig_samples, dtype=np.int32).reshape(-1)
            if sig_samples.size == 0:
                coarse_rot = np.empty(0, dtype=np.int32)
                coarse_trans = np.empty(0, dtype=np.int32)
                unique_rot = np.array([0], dtype=np.int32)
            else:
                coarse_rot = sig_samples // n_coarse_trans
                coarse_trans = sig_samples % n_coarse_trans
                unique_rot = np.unique(coarse_rot)

        if unique_rot.size == 0:
            raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")

        if use_full_rotation_support:
            if full_support_rotation_cache is None:
                if fine_rotations_override is None and fine_rotation_parent_override is None:
                    full_rots, full_parent_map, full_rot_indices, full_eulers = (
                        get_oversampled_rotation_grid_from_samples(
                            full_unique_rot,
                            nside_level,
                            oversampling_order=oversampling_order,
                            random_perturbation=random_perturbation,
                            return_rotation_indices=True,
                            return_source_eulers=True,
                            dtype=dtype,
                            **({} if symmetry_label == "C1" else {"symmetry": symmetry_label}),
                        )
                    )
                    full_support_rotation_cache = (
                        np.asarray(full_rots, dtype=dtype),
                        np.asarray(full_parent_map, dtype=np.int32),
                        np.asarray(full_rot_indices, dtype=np.int64),
                        full_eulers,
                    )
                else:  # Paired overrides were validated before entering the image loop.
                    full_support_rotation_cache = (
                        fine_rotations_np,
                        fine_parent_np.astype(np.int32, copy=False),
                        np.arange(fine_rotations_np.shape[0], dtype=np.int64),
                        fine_source_eulers,
                    )
                full_support_rotation_cache = _reorder_children(
                    *full_support_rotation_cache,
                    parent_ids=full_unique_rot[full_support_rotation_cache[1]],
                )
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = full_support_rotation_cache
        elif fine_rotations_override is None and fine_rotation_parent_override is None:
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = (
                get_oversampled_rotation_grid_from_samples(
                    unique_rot,
                    nside_level,
                    oversampling_order=oversampling_order,
                    random_perturbation=random_perturbation,
                    return_rotation_indices=True,
                    return_source_eulers=True,
                    dtype=dtype,
                    **({} if symmetry_label == "C1" else {"symmetry": symmetry_label}),
                )
            )
            oversampled_rots = np.asarray(oversampled_rots, dtype=dtype)
            parent_map = np.asarray(parent_map, dtype=np.int32)
            oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int64)
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = _reorder_children(
                oversampled_rots,
                parent_map,
                oversampled_rot_indices,
                source_eulers,
                unique_rot[parent_map],
            )
        else:  # Paired overrides were validated before entering the image loop.
            oversampled_rot_indices = _children_of(fine_parent_np, fine_by_parent, fine_parent_offsets, unique_rot)
            oversampled_rots = fine_rotations_np[oversampled_rot_indices]
            parent_map = np.searchsorted(unique_rot, fine_parent_np[oversampled_rot_indices]).astype(np.int32)
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = _reorder_children(
                oversampled_rots,
                parent_map,
                oversampled_rot_indices,
                None if fine_source_eulers is None else fine_source_eulers[oversampled_rot_indices],
                fine_parent_np[oversampled_rot_indices],
            )

        oversampled_mstep_rots = (
            oversampled_rots if fine_mstep_rotations_np is None else fine_mstep_rotations_np[oversampled_rot_indices]
        )

        if per_image_rotation_log_prior is not None:
            if use_full_rotation_support:
                raise ValueError("a per-image rotation prior needs each image's explicit coarse support")
            image_prior = np.asarray(per_image_rotation_log_prior[image_idx], dtype=dtype).reshape(-1)
            if image_prior.size == 0 and coarse_rot.size == 0:
                # No candidate (a class the image is not scored against): the placeholder rotation, masked out.
                image_prior = np.zeros(unique_rot.size, dtype=dtype)
            if image_prior.shape != (unique_rot.size,):
                raise ValueError(
                    f"image {image_idx}'s rotation prior covers {image_prior.size} rotations, "
                    f"its support {unique_rot.size}"
                )
            local_rotation_log_prior = image_prior[parent_map]
        elif use_full_rotation_support and full_support_log_prior_cache is not None:
            local_rotation_log_prior = full_support_log_prior_cache
        elif rotation_log_prior_np is not None:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map]
            if use_full_rotation_support:
                local_rotation_log_prior = local_rotation_log_prior.astype(dtype, copy=False)
        else:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=dtype)
        if use_full_rotation_support:
            full_support_log_prior_cache = local_rotation_log_prior

        if use_full_candidate_mask:
            if full_support_candidate_mask_cache is None:
                full_support_candidate_mask_cache = SparseCandidateMask(
                    mode="full",
                    n_rows=oversampled_rots.shape[0],
                    n_fine_trans=n_fine_trans,
                    count=int(oversampled_rots.shape[0]) * int(n_fine_trans),
                )
            candidate_mask = full_support_candidate_mask_cache
        elif coarse_excluded is not None:
            excluded = np.unique(coarse_excluded.astype(np.int64, copy=False))
            if excluded.size and (
                int(excluded.min(initial=0)) < 0 or int(excluded.max(initial=-1)) >= int(n_coarse_rot * n_coarse_trans)
            ):
                raise ValueError("Complement significant sample exclusions must index the coarse pose grid")
            excluded_rot = excluded // int(n_coarse_trans)
            excluded_trans = excluded % int(n_coarse_trans)
            fine_rot_children = np.bincount(np.asarray(parent_map, dtype=np.int64), minlength=int(n_coarse_rot))
            fine_trans_children = np.bincount(
                np.asarray(fine_translation_parent, dtype=np.int64),
                minlength=int(n_coarse_trans),
            )
            excluded_fine_count = int(
                np.sum(fine_rot_children[excluded_rot] * fine_trans_children[excluded_trans], dtype=np.int64),
            )
            candidate_mask = SparseCandidateMask(
                mode="coarse_exclude",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_excluded=excluded.astype(np.int32, copy=False),
                fine_translation_parent=fine_translation_parent,
                count=int(oversampled_rots.shape[0]) * int(n_fine_trans) - excluded_fine_count,
            )
        elif coarse_trans.size == 0:
            candidate_mask = SparseCandidateMask(
                mode="empty",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                count=0,
            )
        else:
            coarse_valid = np.zeros((unique_rot.size, n_coarse_trans), dtype=bool)
            coarse_valid[np.searchsorted(unique_rot, coarse_rot), coarse_trans] = True
            # count = sum over fine rows of the parent coarse row's fine-translation
            # count; O(cR*T + R) instead of expanding the (R, T) table per image.
            fine_per_coarse_row = coarse_valid[:, fine_translation_parent].sum(axis=1, dtype=np.int64)
            candidate_mask = SparseCandidateMask(
                mode="coarse",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_valid=coarse_valid,
                fine_translation_parent=fine_translation_parent,
                count=int(fine_per_coarse_row[parent_map].sum()),
            )

        per_image_source_eulers.append(source_eulers)
        per_image_oversampled_rots.append(oversampled_rots)
        per_image_oversampled_mstep_rots.append(oversampled_mstep_rots)
        per_image_parent_map.append(parent_map)
        per_image_oversampled_rot_indices.append(oversampled_rot_indices)
        per_image_unique_rot.append(unique_rot)
        per_image_log_prior.append(local_rotation_log_prior.astype(dtype, copy=False))
        per_image_candidate_mask.append(candidate_mask)

    assert len(per_image_oversampled_rots) == n_images
    # Global fine-grid tables (paired override only): every image's rows are
    # gathers of these by ``oversampled_rot_indices``, so the device can gather
    # them from one resident copy instead of receiving the rows per chunk.
    rotation_table = None if fine_rotations_np is None else np.asarray(fine_rotations_np, dtype=dtype)
    mstep_rotation_table = (
        None if fine_mstep_rotations_np is None else np.asarray(fine_mstep_rotations_np, dtype=dtype)
    )
    return {
        "source_eulers": per_image_source_eulers,
        "oversampled_rots": per_image_oversampled_rots,
        "oversampled_mstep_rots": per_image_oversampled_mstep_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
        "rotation_table": rotation_table,
        "rotation_table_key": None if rotation_table is None else _rotation_table_key(rotation_table),
        "mstep_rotation_table": mstep_rotation_table,
        "mstep_rotation_table_key": None if mstep_rotation_table is None else _rotation_table_key(mstep_rotation_table),
        # One coarse-row capacity for the whole pass: the device index builder
        # otherwise compiled a family per chunk-wise coarse-row count (census job
        # 13837258: 28 compiles, 18.7 s). Padded coarse rows are inert.
        "coarse_rows_capacity": pow2_ceil(
            max((int(np.asarray(u).shape[0]) for u in per_image_unique_rot), default=1), minimum=64
        ),
    }
