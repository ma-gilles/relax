"""A resident chunk's program spec and operands, and their compile-ahead warm-up.

The same constructors build a chunk's row arrays, stage operands and tables for the chunk loop (placed on
the device) and for the warm-up (placed as avals); :func:`chunk_program_path`, :func:`chunk_program_keys`
and :func:`chunk_programs_for_path` name the programs a chunk runs, and
:func:`submit_resident_chunk_warmup` compiles them ahead on the compile-ahead pool.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass import (
    policy,
    resident_chunk_inputs,
    resident_chunk_posterior,
    resident_chunk_records,
    resident_chunk_stages,
    resident_mstep_blocks,
)
from relax.fine_pass.resident_candidates import CandidateTableBlocks, chunk_segment_offsets, materialize_chunk
from relax.fine_pass.resident_operands import gather_resident_chunk_operands
from relax.sampling.translation_prior import translation_prior_centers_for_images, translation_sqdist_angstrom

_PLACE_AS_AVAL = resident_chunk_records.Placement(
    array=lambda value, dtype: jax.ShapeDtypeStruct(np.shape(value), jnp.dtype(dtype)),
    scalar=lambda value, dtype: jax.ShapeDtypeStruct((), jnp.dtype(dtype)),
    many=lambda items: {
        name: jax.ShapeDtypeStruct(np.shape(value), jnp.dtype(dtype)) for name, (value, dtype) in items.items()
    },
)


def row_projection_ids(host_chunk, n_fine_rot: int | None, slot_of_projection=None) -> np.ndarray:
    """Each row's projection: its fine rotation, offset by ``class * n_fine_rot`` for K>1.

    The class-stacked projection caches and grids hold class ``k``'s fine grid at
    ``k * n_fine_rot``, so every consumer that gathers by a row's rotation id
    gathers its class's projection with the same statement. With a
    significant-parent cache (``slot_of_projection``, :class:`_ProjectionSlots`)
    the id is the projection's cache row instead; padded rows take row 0.
    """

    row_fine_rot = np.asarray(host_chunk["row_fine_rot"], dtype=np.int64)
    if n_fine_rot is not None:
        row_fine_rot = row_fine_rot + np.asarray(host_chunk["row_class"], dtype=np.int64) * int(n_fine_rot)
    if slot_of_projection is None:
        return row_fine_rot.astype(np.int32)
    n_valid_rows = int(host_chunk["n_valid_rows"])
    slots = np.zeros(row_fine_rot.shape, dtype=np.int32)
    slots[:n_valid_rows] = slot_of_projection[row_fine_rot[:n_valid_rows]]
    if bool(np.any(slots[:n_valid_rows] < 0)):
        raise RuntimeError("a candidate row's fine rotation is not in the significant-parent projection cache")
    return slots


def chunk_class_layout(host_chunk, chunk, *, n_classes: int, n_fine_trans: int, place) -> resident_chunk_records.ChunkClassLayout:
    """The (slot, class) posterior sub-segments of one chunk."""

    n_segments = chunk.image_capacity * int(n_classes)
    row_class = np.asarray(host_chunk["row_class"], dtype=np.int64)
    row_segment = np.full(chunk.row_capacity, n_segments, dtype=np.int64)
    row_segment[:chunk.n_valid_rows] = (
        np.asarray(host_chunk["row_image_local"][:chunk.n_valid_rows], dtype=np.int64) * int(n_classes)
        + row_class[:chunk.n_valid_rows]
    )
    if np.any(np.diff(row_segment[:chunk.n_valid_rows]) < 0):
        raise ValueError("chunk rows must be image-major, then class-major")
    segment_rows = np.bincount(row_segment[:chunk.n_valid_rows], minlength=n_segments)
    segment_row_start = np.concatenate([[0], np.cumsum(segment_rows)])
    return resident_chunk_records.ChunkClassLayout(
        row_class=place.array(row_class, jnp.int32),
        row_segment=place.array(row_segment, jnp.int32),
        segment_offsets=place.array(segment_row_start * int(n_fine_trans), jnp.int32),
        segment_row_start=place.array(segment_row_start[:-1], jnp.int64),
    )


def _chunk_mstep_layout(host_chunk, *, place) -> resident_chunk_records.ChunkMstepLayout:
    """The chunk's accumulator slot per row (docs/development/resident_segments.md).

    Slot ``a = class + K * slot_offset[unit]``: a class's rows (Class3D, RELION's
    ``BPref[iclass]``, ml_optimiser.cpp:10826) and, for VDAM, a pseudo-halfset's
    (``iclass + (part_id % 2) * nr_classes``, acc_ml_optimiser_impl.h:4800-4804).
    The slot-major order itself is taken on the device, over the live rows only
    (:func:`_make_mstep_block_inputs`).
    """

    return resident_chunk_records.ChunkMstepLayout(row_slot=place.array(host_chunk["row_slot"], jnp.int32))


def chunk_host_rows(tables: CandidateTableBlocks, chunk):
    """``(block tables, materialized host rows, chunk in block numbering)`` of one chunk.

    The rows are :func:`materialize_chunk`'s over the chunk's table block, with
    ``image_ids`` back in the pass's image numbering.
    """

    block_tables, local_chunk, image_base = tables.chunk_tables(chunk)
    host_chunk = materialize_chunk(block_tables, local_chunk)
    host_chunk["image_ids"][:chunk.n_valid_images] += np.int32(image_base)
    return block_tables, host_chunk, local_chunk


def make_chunk_row_arrays(
    tables, chunk, n_fine_trans, *, place, n_fine_rot=None, slot_of_projection=None
) -> resident_chunk_records.ChunkRowArrays:
    """One chunk's row-aligned inputs, on the device or as avals.

    Everything here is host NumPy over the plan, so the aval placement costs
    only the materialize and does no device work at all. The shapes are the
    chunk's capacity class and nothing else, which
    ``tests/unit/test_chunk_row_avals.py`` pins: that is why warming one chunk
    per class covers every chunk of that class.

    A K-class table (``tables.n_classes > 1``) needs ``n_fine_rot``: its rows
    carry projection ids (:func:`row_projection_ids`) and the class layout.
    """

    tables, host_chunk, local_chunk = chunk_host_rows(tables, chunk)
    segment_offsets_np = chunk_segment_offsets(tables, local_chunk, n_fine_trans=n_fine_trans)
    image_row_start_np = segment_offsets_np.astype(np.int64)[:chunk.image_capacity] // int(n_fine_trans)
    image_row_count_np = (
        segment_offsets_np.astype(np.int64)[1:] - segment_offsets_np.astype(np.int64)[:-1]
    ) // int(n_fine_trans)
    if tables.n_classes > 1 and n_fine_rot is None:
        raise ValueError("a K-class chunk needs n_fine_rot for its projection ids")
    classes = None
    if tables.n_classes > 1:
        classes = chunk_class_layout(
            host_chunk, chunk, n_classes=tables.n_classes, n_fine_trans=n_fine_trans, place=place
        )
    mstep = None
    if int(tables.n_slots) > 1:
        mstep = _chunk_mstep_layout(host_chunk, place=place)
    placed = place.many(
        {
            "row_image_local": (host_chunk["row_image_local"], jnp.int32),
            "row_fine_rot": (
                row_projection_ids(host_chunk, n_fine_rot if tables.n_classes > 1 else None, slot_of_projection),
                jnp.int32,
            ),
            "row_log_prior": (host_chunk["row_log_prior"], jnp.float32),
            "row_mask_bits": (host_chunk["row_mask_bits"], jnp.uint32),
            "row_mask_mode": (host_chunk["row_mask_mode"], jnp.int8),
            "image_ids": (host_chunk["image_ids"], jnp.int32),
            "n_valid_rows": (host_chunk["n_valid_rows"], jnp.int32),
            "n_valid_images": (host_chunk["n_valid_images"], jnp.int32),
            "segment_offsets": (segment_offsets_np, jnp.int32),
            "image_row_start": (image_row_start_np, jnp.int64),
            "image_row_count": (image_row_count_np, jnp.int64),
        }
    )
    return resident_chunk_records.ChunkRowArrays(**placed, classes=classes, mstep=mstep)


def make_chunk_translation_sqdist(
    default,
    *,
    translation_prior_centers_np,
    image_indices,
    image_capacity: int,
    n_valid_images,
    fine_translations,
    voxel_size,
):
    """The chunk's per-image prior squared distances, at image capacity.

    A per-image host table built at capacity keeps the program keyed on the
    capacity class and takes the eager device ops out of the chunk loop. Padded
    slots multiply a zero posterior, so their value is never observable; they
    are zeroed anyway.
    """

    if translation_prior_centers_np is None:
        return default
    padded_image_indices = resident_chunk_inputs.pad_batch_to_capacity(
        np.asarray(image_indices).reshape(-1, 1), image_capacity
    ).reshape(-1)
    centers = translation_prior_centers_for_images(
        translation_prior_centers_np,
        padded_image_indices,
        batch_size=image_capacity,
    )
    sqdist_np = np.asarray(translation_sqdist_angstrom(fine_translations, centers, voxel_size))
    sqdist_np = np.where(
        (np.arange(image_capacity) < int(n_valid_images))[:, None], sqdist_np, 0.0
    )
    return jnp.asarray(sqdist_np)


def make_chunk_stage_operands(recon, translation_sqdist_ang) -> resident_chunk_records.ChunkStageOperands:
    """Name one chunk's operands out of whatever produced them.

    ``recon`` is the per-chunk preparation's dict, the once-per-half gather's
    dict, or -- for the compile-ahead warm-up -- the same gather's output under
    ``jax.eval_shape``, which is a dict of the same keys holding avals.
    """

    return resident_chunk_records.ChunkStageOperands(
        score_input=recon["score_input"],
        corr_img_score=recon["corr_img_score"],
        highres_xi2_half=recon["highres_xi2_half"],
        translation_prior=recon["translation_prior"],
        shifted_recon=recon.get("shifted_recon"),
        shifted_noise=recon.get("shifted_noise"),
        recon_image=recon.get("recon_image"),
        recon_weight=recon.get("recon_weight"),
        noise_image=recon.get("noise_image"),
        ctf2_over_nv_recon=recon["ctf2_over_nv_recon"],
        direct_ctf_rfloat_recon=recon["direct_ctf_rfloat_recon"],
        image_power_shells=recon["image_power_shells"],
        relion_norm_high_shell=recon["relion_norm_high_shell"],
        raw_translated_wavg_rectangle=recon["raw_translated_wavg_rectangle"],
        raw_translated_wavg_for_atomic=recon["raw_translated_wavg_for_atomic"],
        scale=recon["scale"],
        group_ids=recon["group_ids"],
        translation_sqdist_ang=translation_sqdist_ang,
        optics_groups=recon.get("optics_groups"),
        score_shifted_cc=recon.get("score_shifted_cc"),
        cc_half_batch_norm=recon.get("cc_half_batch_norm"),
        bpref_ctf2_over_nv_recon=recon.get("bpref_ctf2_over_nv_recon"),
    )


def chunk_program_path() -> str:
    """Which programs a chunk of this half will actually submit.

    ``"per-stage"`` is the default path's three glue programs, ``"eager"`` the
    loose dispatch with no program to warm. The compile-ahead warm-up reads this
    so it cannot warm a path the loop does not take. (A third path, one fused
    program per chunk, was removed in relax#76: nothing selected it.)
    """

    if policy.resident_glue_jit_enabled():
        return "per-stage"
    return "eager"


class _ChunkWarmup(NamedTuple):
    """What the warm-up queued, in the terms the chunk loop can be compared in."""

    classes: tuple
    keys: frozenset
    path: str


def describe_chunk_spec_prediction(
    *,
    predicted_rfloat_ctf_wavg,
    predicted_bpref_recon_operand,
    predicted_translate_sum_kernel,
    operands,
) -> str:
    """Name every spec boolean the warm-up predicted differently from the loop.

    Returns an empty string when they agree. The loop reads these three from the
    prepared operands; the warm-up has to predict them from configuration before
    the preparation runs, so this is the same predict-early verify-late check the
    operand tree gets, applied to the part of the program key that is not an aval.
    """

    if operands is None:
        return "the half prepared no resident operands, so no spec was used"
    problems = []
    for name, predicted, actual in (
        ("use_rfloat_ctf_wavg", bool(predicted_rfloat_ctf_wavg),
         operands.direct_ctf_rfloat_recon is not None),
        ("bpref_recon_operand", bool(predicted_bpref_recon_operand),
         operands.recon_weight is not None),
        ("use_translate_sum_kernel", bool(predicted_translate_sum_kernel), True),
    ):
        if predicted != actual:
            problems.append(f"{name}: predicted {predicted}, really {actual}")
    return "; ".join(problems)


def chunk_program_keys(path: str, spec) -> frozenset:
    """The ``(program name, spec)`` keys a chunk of ``spec`` will submit.

    A compiled program is identified by its function and its static argument,
    so this is the unit in which "what the warm-up compiled" and "what the loop
    ran" are the same kind of thing. Comparing capacity classes alone would
    miss a spec that differs in one of the booleans the warm-up has to predict
    from configuration, which is a real way for a warmed program to go unused.
    """

    return frozenset(
        (program.__name__, spec) for program in chunk_programs_for_path(path)
    )


def chunk_programs_for_path(path: str) -> tuple:
    """The jitted programs a chunk will submit on ``path``.

    One list, read by the compile-ahead warm-up and asserted against the
    runners by `tests/unit/test_em_compile_ahead_consumer.py`. Warming a
    program the runner does not submit, or missing one it does, costs compile
    time and buys nothing; that happened once, when a since-removed fused
    chunk program was warmed while the loop ran the per-stage one, and was found
    by reading a census rather than by a test.
    """

    if path == "per-stage":
        return (
            resident_chunk_posterior.resident_chunk_posterior_program,
            resident_mstep_blocks.resident_mstep_block_program,
            resident_chunk_stages.resident_chunk_statistics_program,
        )
    if path == "eager":
        return ()
    raise ValueError(f"unknown chunk program path {path!r}")


def make_chunk_program_spec(
    *,
    row_capacity,
    image_capacity,
    n_fine_trans,
    n_score_pixels,
    n_recon_pixels,
    n_rect,
    mstep_block_rows,
    adaptive_fraction,
    current_size,
    mstep_current_size,
    image_shape,
    recon_volume_shape,
    max_adjoint_block_bytes,
    stats_config,
    use_rfloat_ctf_wavg,
    use_translate_sum_kernel,
    bpref_recon_operand,
    mstep_max_r=None,
    reuse_coarse_normalization=False,
    firstiter_cc=False,
    n_slots=1,
    mstep_subtract_ctf_projection=False,
    n_classes=1,
    stable_window=False,
    union_native_fft_size=0,
    presum_adjoint=False,
) -> resident_chunk_records.ChunkProgramSpec:
    """The static key of one chunk program.

    The environment-read fields are the reason this is a function
    and not a literal at each call site: a warm-up that read them at a
    different moment, or not at all, would key its program differently from the
    loop's and warm nothing.
    """

    return resident_chunk_records.ChunkProgramSpec(
        row_capacity=int(row_capacity),
        image_capacity=int(image_capacity),
        n_fine_trans=int(n_fine_trans),
        n_score_pixels=int(n_score_pixels),
        n_recon_pixels=int(n_recon_pixels),
        n_rect=int(n_rect),
        mstep_block_rows=int(mstep_block_rows),
        adaptive_fraction=float(adaptive_fraction),
        current_size=int(current_size),
        mstep_current_size=int(mstep_current_size),
        image_shape=tuple(int(v) for v in image_shape),
        recon_volume_shape=tuple(int(v) for v in recon_volume_shape),
        max_adjoint_block_bytes=int(max_adjoint_block_bytes),
        stats_config=stats_config,
        use_rfloat_ctf_wavg=bool(use_rfloat_ctf_wavg),
        use_translate_sum_kernel=bool(use_translate_sum_kernel),
        bpref_recon_operand=bool(bpref_recon_operand),
        mstep_max_r=float(int(mstep_current_size) // 2) if mstep_max_r is None else mstep_max_r,
        kernel_ctf_probs=policy.kernel_ctf_probs_enabled(),
        reuse_coarse_normalization=bool(reuse_coarse_normalization),
        firstiter_cc=bool(firstiter_cc),
        n_slots=int(n_slots),
        mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
        n_classes=int(n_classes),
        stable_window=bool(stable_window),
        union_native_fft_size=int(union_native_fft_size),
        presum_adjoint=bool(presum_adjoint),
    )


def submit_resident_chunk_warmup(
    pool,
    *,
    chunks,
    tables,
    n_fine_trans,
    half_operand_avals,
    stage_tables,
    carry,
    translation_angles,
    rect_indices,
    exact_positions,
    image_shape,
    spec_kwargs,
    translation_prior_centers_np,
    fine_translations,
    voxel_size,
    default_translation_sqdist,
):
    """Queue the chunk programs of every capacity class the plan will run.

    Called in the window between the admission check and the per-half operand
    preparation. The preparation is 2.4-5.8 s of host-bound device dispatch on
    the main thread, and the chunk programs the loop will need after it are
    fully determined by then: the capacity classes are in ``chunks``, the
    iteration-global tables exist, and the operands the programs consume are
    described by ``half_operand_avals`` without being prepared.

    Nothing here can change a result. The warm-up hands the helper thread
    shape/dtype stand-ins only, and if it describes a program the loop does not
    run, the loop compiles its own as before; the cost is the wasted warm-up and
    the log line below is how that is noticed.

    Returns the capacity classes submitted, for the hit-rate line.
    """

    import dataclasses

    from relax.fine_pass.resident_operands import ResidentHalfOperands

    # Warm the programs the configured path will actually submit: the per-stage
    # path with the glue JIT on is what production runs, and it is three
    # programs. The path is decided here, before any work, and named in the log line.
    path = chunk_program_path()
    if path == "eager":
        return ()

    def as_aval(value):
        if value is None:
            return None
        return jax.ShapeDtypeStruct(
            tuple(int(d) for d in np.shape(value)), jnp.dtype(value.dtype)
        )

    expected_programs = chunk_programs_for_path(path)

    def _checked(work):
        # The warm-up must submit exactly the programs the runner for this path
        # submits. Raising here lands in the pool's own error record, so a
        # mismatch is reported and the run is untouched.
        got = tuple(program for program, _, _ in work)
        if set(got) != set(expected_programs):
            raise ValueError(
                "compile-ahead would warm "
                f"{sorted(p.__name__ for p in got)} on the {path} path, but that path "
                f"runs {sorted(p.__name__ for p in expected_programs)}"
            )
        return work

    table_avals = jax.tree_util.tree_map(as_aval, stage_tables)
    carry_avals = jax.tree_util.tree_map(as_aval, carry)
    angle_aval = as_aval(translation_angles)
    rect_aval = as_aval(rect_indices)
    exact_aval = as_aval(exact_positions)

    names = [
        f.name for f in dataclasses.fields(ResidentHalfOperands) if not f.name.startswith("n_")
    ]
    present = [n for n in names if getattr(half_operand_avals, n) is not None]
    scalars = {
        f.name: getattr(half_operand_avals, f.name)
        for f in dataclasses.fields(ResidentHalfOperands)
        if f.name.startswith("n_")
    }
    present_avals = [getattr(half_operand_avals, n) for n in present]

    submitted = []
    submitted_keys = set()
    for chunk in chunks:
        capacity_class = (int(chunk.row_capacity), int(chunk.image_capacity))
        if capacity_class in submitted:
            continue
        row_capacity, image_capacity = capacity_class
        row_avals = make_chunk_row_arrays(tables, chunk, n_fine_trans, place=_PLACE_AS_AVAL)
        sqdist = make_chunk_translation_sqdist(
            default_translation_sqdist,
            translation_prior_centers_np=translation_prior_centers_np,
            image_indices=np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64),
            image_capacity=image_capacity,
            n_valid_images=chunk.n_valid_images,
            fine_translations=fine_translations,
            voxel_size=voxel_size,
        )
        spec = make_chunk_program_spec(
            row_capacity=row_capacity, image_capacity=image_capacity, **spec_kwargs
        )

        def thunk(_row=row_avals, _spec=spec, _images=image_capacity, _sq=as_aval(sqdist)):
            # The chunk operands are predicted by tracing the real gather, not
            # by a second constructor: the gather's own shape validation runs on
            # the way through, and there is no place for the two to disagree.
            def gather(slots, angles, rect, exact, *arrays):
                fields = dict(scalars)
                fields.update({name: None for name in names})
                fields.update(dict(zip(present, arrays)))
                return gather_resident_chunk_operands(
                    ResidentHalfOperands(**fields),
                    slots,
                    translation_angles=angles,
                    rect_indices=rect,
                    exact_positions=exact,
                    image_shape=image_shape,
                )

            recon = jax.eval_shape(
                gather,
                jax.ShapeDtypeStruct((_images,), jnp.int32),
                angle_aval,
                rect_aval,
                exact_aval,
                *present_avals,
            )
            operand_avals = make_chunk_stage_operands(recon, _sq)
            # The per-stage path runs three programs.
            # Their later inputs are earlier stages' outputs, so they are taken
            # by tracing those stages rather than described a second time.
            posterior_avals = jax.eval_shape(
                partial(resident_chunk_posterior.resident_chunk_posterior_program, spec=_spec),
                _row,
                operand_avals,
                table_avals,
            )
            mstep_avals = jax.eval_shape(
                partial(resident_mstep_blocks.initial_mstep_carry, spec=_spec),
                carry_avals[0][0],
                carry_avals[1][0],
                operand_avals,
                table_avals,
            )
            block_avals = jax.eval_shape(
                partial(resident_mstep_blocks.make_mstep_block_inputs, n_slots=int(_spec.n_slots)), _row, posterior_avals
            )
            return _checked([
                (
                    resident_chunk_posterior.resident_chunk_posterior_program,
                    (_row, operand_avals, table_avals),
                    {"spec": _spec},
                ),
                (
                    resident_mstep_blocks.resident_mstep_block_program,
                    (
                        jax.ShapeDtypeStruct((), jnp.int32),
                        block_avals,
                        operand_avals,
                        table_avals,
                        mstep_avals,
                    ),
                    {"spec": _spec},
                ),
                (
                    resident_chunk_stages.resident_chunk_statistics_program,
                    (
                        carry_avals[2],
                        _row,
                        operand_avals,
                        table_avals,
                        posterior_avals,
                        mstep_avals,
                    ),
                    {"spec": _spec},
                ),
            ])

        if pool.submit_thunk(f"resident chunk rows={row_capacity} images={image_capacity}", thunk):
            submitted.append(capacity_class)
            submitted_keys.update(chunk_program_keys(path, spec))
    return _ChunkWarmup(classes=tuple(submitted), keys=frozenset(submitted_keys), path=path)


def make_chunk_stage_tables(
    *,
    projection_score_cache,
    projection_recon_cache,
    projection_recon_abs2_cache,
    mstep_grid,
    coarse_parent_grid,
    fine_translation_parent_device,
    half_weights,
    translation_angles,
    full_to_compact,
    noise_variance_for_noise,
    shell_indices_noise,
    exact_positions_device,
    recon_pixel_indices,
    relion_x_half_recon_indices,
    image_tables,
    cache_slot_fine_rot=None,
    coarse_reuse=None,
    window_logical=None,
    union_score_take=None,
    union_recon_take=None,
    residual_sgd_cache=None,
    residual_sgd_take=None,
) -> resident_chunk_records.ChunkStageTables:
    """Assemble the iteration-global tables every chunk of a half reads.

    One builder, called by the chunk loop with the real arrays and by the
    compile-ahead warm-up with their shape/dtype stand-ins. Assembling the
    tuple twice would be a place for the warm-up to drift from the loop: a
    warmed program with one field's dtype wrong is never used, which costs
    compile time and is invisible unless someone reads the hit rate.
    """

    return resident_chunk_records.ChunkStageTables(
        projection_score_cache=projection_score_cache,
        projection_recon_cache=projection_recon_cache,
        projection_recon_abs2_cache=projection_recon_abs2_cache,
        mstep_grid=mstep_grid,
        coarse_parent_grid=coarse_parent_grid,
        fine_translation_parent=fine_translation_parent_device,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions=exact_positions_device,
        recon_pixel_indices=recon_pixel_indices,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        shell_indices_half=image_tables.shell_indices_half,
        wavg_shell_indices=image_tables.wavg_shell_indices,
        wavg_scale_pixel_mask=image_tables.wavg_scale_pixel_mask,
        cache_slot_fine_rot=cache_slot_fine_rot,
        coarse_reuse=coarse_reuse,
        window_logical=window_logical,
        union_score_take=union_score_take,
        union_recon_take=union_recon_take,
        residual_sgd_cache=residual_sgd_cache,
        residual_sgd_take=residual_sgd_take,
    )
