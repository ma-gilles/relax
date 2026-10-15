"""The resident pass's candidate tables over every class, built in bounded image blocks.

Each class's significant samples become its fine candidate rows (the device-compacted CSR when the coarse
pass left one, the host build otherwise); the classes are merged and each image block of at most
``BLOCK_UNITS`` rows plus significant ids is built when the chunk loop first reaches it. The reconstruction
groups (VDAM's pseudo-halfsets) give each unit its M-step accumulator slot group. Read by the resident
driver (``resident_pass2``); imports nothing from it.
"""

from __future__ import annotations

import time
from dataclasses import replace as dataclass_replace

import numpy as np

from relax.fine_pass.bucket_arrays import prepare_per_image_pass2_inputs
from relax.fine_pass.resident_candidates import (
    BLOCK_UNITS,
    CandidateTableBlocks,
    map_over_classes,
    merge_class_tables,
    table_block_starts,
)
from relax.fine_pass.resident_significance import (
    build_resident_candidate_tables_from_csr,
    csr_candidate_rows_per_image,
    fine_rotation_children,
    resident_candidate_tables,
    resident_significance_csr,
)


def _class_candidate_tables(
    significant_sample_indices,
    rotation_log_prior,
    *,
    n_images,
    n_coarse_rot,
    n_coarse_trans,
    nside_level,
    oversampling_order,
    n_fine_trans,
    fine_translation_parent,
    random_perturbation,
    fine_source_eulers_override,
    fine_rotations_override,
    fine_mstep_rotations_override,
    fine_rotation_parent_override,
    use_relion_f32_fine_posterior,
    dtype,
    symmetry_label,
    coarse_rotation_ids=None,
    per_image_rotation_log_prior=None,
):
    """One class's per-image hypotheses and candidate table (T5), exactly the K=1 build; the fine candidates
    take RELION's parent execution order with the float32 fine posterior.

    ``coarse_rotation_ids`` and ``per_image_rotation_log_prior`` are the compact coarse grid and
    the per-unit priors of a subtomogram local search (``prepare_per_image_pass2_inputs``); the priors
    are over this class's support.
    Returns ``(tables, hypothesis_prep_seconds, table_seconds)``.
    """

    prep_t0 = time.time()
    relion_parent_execution_order = use_relion_f32_fine_posterior
    significance_csr = resident_significance_csr(
        significant_sample_indices,
        n_images=n_images,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
    )
    per_image_inputs = None if significance_csr is not None else prepare_per_image_pass2_inputs(
        significant_sample_indices,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=random_perturbation,
        fine_source_eulers_override=fine_source_eulers_override,
        fine_rotations_override=fine_rotations_override,
        fine_mstep_rotations_override=fine_mstep_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        relion_parent_execution_order=relion_parent_execution_order,
        dtype=dtype,
        symmetry_label=symmetry_label,
        coarse_rotation_ids=coarse_rotation_ids,
        per_image_rotation_log_prior=per_image_rotation_log_prior,
    )
    if significance_csr is not None and (coarse_rotation_ids is not None or per_image_rotation_log_prior is not None):
        raise ValueError("a compact coarse grid or per-image priors take the host candidate-table path")
    prep_s = time.time() - prep_t0

    table_t0 = time.time()
    tables = resident_candidate_tables(
        significance_csr,
        per_image_inputs,
        n_coarse_trans=n_coarse_trans,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=random_perturbation,
        fine_rotation_parent_override=fine_rotation_parent_override,
        relion_parent_execution_order=relion_parent_execution_order,
        dtype=dtype,
        symmetry_label=symmetry_label,
    )
    if int(tables.n_images) != int(n_images):
        raise ValueError(
            f"candidate table covers {tables.n_images} images but the dataset has {n_images}"
        )
    return tables, prep_s, time.time() - table_t0


def candidate_table_blocks(
    class_supports,
    class_rotation_priors,
    *,
    whole: bool,
    reconstruction_group_ids,
    reconstruction_group_count,
    **table_kwargs,
):
    """The pass's candidate tables over every class, as :class:`CandidateTableBlocks`.

    With every class's device-compacted CSR the tables are built one image
    block of at most ``BLOCK_UNITS`` rows plus significant ids at a time, when
    the chunk loop first reaches the block; only the per-image row counts are
    computed here. The
    rows, their order and their fields are the whole-pass build's, since each
    image's rows depend on that image alone. ``whole`` (the zero-oversampling
    coarse reuse reads every image's table) and a host-path support build the
    one-block tables as before. Returns ``(blocks, hypothesis_prep_s, table_s)``.
    """

    n_images = int(table_kwargs["n_images"])
    csrs = [
        resident_significance_csr(
            support,
            n_images=n_images,
            n_coarse_rot=table_kwargs["n_coarse_rot"],
            n_coarse_trans=table_kwargs["n_coarse_trans"],
        )
        for support in class_supports
    ]

    def with_groups(tables, start, stop):
        group_ids = None if reconstruction_group_ids is None else np.asarray(reconstruction_group_ids)[start:stop]
        return with_reconstruction_groups(tables, group_ids, reconstruction_group_count, n_images=stop - start)

    def merged(tables_by_class):
        return tables_by_class[0] if len(tables_by_class) == 1 else merge_class_tables(tables_by_class)

    # A subtomogram local search's compact coarse grid and per-unit priors take the host build. The priors are
    # one list per unit over its support's coarse rotations; K classes give one such list per class.
    per_image_priors = table_kwargs.pop("per_image_rotation_log_prior", None)
    compact_local = table_kwargs.get("coarse_rotation_ids") is not None or per_image_priors is not None
    if per_image_priors is None:
        class_image_priors = (None,) * len(class_supports)
    elif len(class_supports) == 1:
        class_image_priors = (per_image_priors,)
    else:
        class_image_priors = tuple(per_image_priors)
        if len(class_image_priors) != len(class_supports):
            raise ValueError("a K-class local search gives each class its particles' priors")
    if whole or compact_local or any(csr is None for csr in csrs):
        class_builds = map_over_classes(
            lambda item: _class_candidate_tables(
                item[0], item[1], per_image_rotation_log_prior=item[2], **table_kwargs
            ),
            zip(class_supports, class_rotation_priors, class_image_priors),
        )
        table_t0 = time.time()
        tables = with_groups(merged([built for built, _, _ in class_builds]), 0, n_images)
        prep_s = sum(class_prep_s for _, class_prep_s, _ in class_builds)
        table_s = sum(class_table_s for _, _, class_table_s in class_builds) + time.time() - table_t0
        return CandidateTableBlocks.whole(tables), prep_s, table_s

    table_t0 = time.time()
    children = fine_rotation_children(
        n_coarse_rot=table_kwargs["n_coarse_rot"],
        nside_level=table_kwargs["nside_level"],
        oversampling_order=table_kwargs["oversampling_order"],
        random_perturbation=table_kwargs["random_perturbation"],
        fine_rotation_parent_override=table_kwargs["fine_rotation_parent_override"],
        symmetry_label=table_kwargs["symmetry_label"],
    )
    rows_per_image = np.sum(
        map_over_classes(lambda csr: csr_candidate_rows_per_image(csr, children[0]), csrs), axis=0
    )
    row_offsets = np.zeros(n_images + 1, dtype=np.int64)
    row_offsets[1:] = np.cumsum(rows_per_image)
    # A block is counted in rows plus the classes' significant ids its builders read (BLOCK_UNITS).
    unit_offsets = row_offsets.copy()
    unit_offsets[1:] += np.cumsum(np.sum([csr.counts() for csr in csrs], axis=0, dtype=np.int64))
    csr_kwargs = dict(
        nside_level=table_kwargs["nside_level"],
        oversampling_order=table_kwargs["oversampling_order"],
        n_fine_trans=table_kwargs["n_fine_trans"],
        fine_translation_parent=table_kwargs["fine_translation_parent"],
        random_perturbation=table_kwargs["random_perturbation"],
        fine_rotation_parent_override=table_kwargs["fine_rotation_parent_override"],
        relion_parent_execution_order=table_kwargs["use_relion_f32_fine_posterior"],
        dtype=table_kwargs["dtype"],
        symmetry_label=table_kwargs["symmetry_label"],
        children=children,
    )

    def build_block(start, stop):
        return with_groups(
            merged(
                map_over_classes(
                    lambda item: build_resident_candidate_tables_from_csr(
                        item[0].image_block(start, stop), rotation_log_prior=item[1], **csr_kwargs
                    ),
                    zip(csrs, class_rotation_priors),
                )
            ),
            start,
            stop,
        )

    blocks = CandidateTableBlocks(
        n_images=n_images,
        n_fine_trans=int(table_kwargs["n_fine_trans"]),
        n_coarse_trans=int(table_kwargs["n_coarse_trans"]),
        n_classes=len(csrs),
        n_slot_groups=1 if reconstruction_group_count is None else int(reconstruction_group_count),
        row_offsets=row_offsets,
        block_starts=table_block_starts(unit_offsets, BLOCK_UNITS),
        build_block=build_block,
    )
    return blocks, 0.0, time.time() - table_t0


def with_reconstruction_groups(tables, group_ids, group_count, *, n_images: int):
    """Give each unit its M-step accumulator slot group (VDAM's pseudo-halfsets).

    ``group_ids`` is the caller's per-image group (VDAM: the subset schedule's
    pseudo-halfset, RELION's ``part_id % 2``, acc_ml_optimiser_impl.h:4800-4804);
    nothing is derived from dataset positions. ``group_count`` fixes the number of
    groups, so a group without images still has its (zero) accumulators.
    """

    if group_ids is None and group_count is None:
        return tables
    if group_ids is None or group_count is None:
        raise ValueError("reconstruction_group_ids and reconstruction_group_count go together")
    group_ids = np.asarray(group_ids, dtype=np.int32).reshape(-1)
    if group_ids.shape != (int(n_images),):
        raise ValueError(f"reconstruction_group_ids must have one entry per image ({n_images}), got {group_ids.shape}")
    return dataclass_replace(tables, unit_slot_offset=group_ids, n_slot_groups=int(group_count))
