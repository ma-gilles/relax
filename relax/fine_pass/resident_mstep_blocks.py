"""The M-step of a resident chunk, walked in fixed-size row blocks.

Each block gathers its projections, takes the pixel-axis programs and backprojects its rows into the slot's
accumulators; the carry (:class:`~relax.fine_pass.resident_chunk_records.ChunkMstepCarry`) has static avals
per capacity class. :func:`run_resident_mstep_blocks` runs the blocks alone for a caller that owns its
posterior (the local pass).
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass import resident_block_programs, resident_chunk_records
from relax.fine_pass.adjoint import accumulate_adjoint_block_chunked
from relax.fine_pass.resident_scoring import cache_dtype, cache_rows
from relax.reconstruction.half_volume_mstep import RowPartial


@partial(jax.jit, static_argnames=("n_slots",))
def make_mstep_block_inputs(rows, posterior, *, n_slots: int) -> "resident_chunk_records.MstepBlockInputs":
    """The M-step block program's row inputs, from the chunk and its posterior.

    Only rows the pruned posterior keeps enter the M-step. RELION backprojects
    and sums only weights at or above the significant weight
    (``significant_weight`` in collect2jobs and the backprojection kernel,
    acc_ml_optimiser_impl.h:4819); the fine posterior here already zeroes the
    rest, so a row with no positive cell adds exact zeros to every M-step
    accumulator. At the K4 100k/256 early state 5-10% of the scored rows are
    live, and the M-step was about 70% of the chunk time.

    The rows are taken slot-major (a single slot for K=1 refinement; a class
    for Class3D; a class and pseudo-halfset for VDAM, :class:`ChunkMstepLayout`),
    live rows of a slot first in their chunk order, then every row that is not
    live; ``slot_offsets`` holds each slot's live run, so each slot is one run
    of blocks and the rows of a boundary block outside it get no weight. Grouping the live rows into fewer blocks changes
    which rows share a block, so the per-block partial sums are added in a
    different grouping; the contributions themselves are unchanged.

    Shared with the compile-ahead warm-up so the warmed signature is the one
    the per-stage loop submits. ``projections`` is None here: the block program
    gathers them from the tables itself.
    """

    row_is_live = posterior.row_is_valid & jnp.any(posterior.row_posterior > 0, axis=1)
    row_slot = jnp.zeros_like(rows.row_image_local) if rows.mstep is None else rows.mstep.row_slot
    key = jnp.where(row_is_live, row_slot, jnp.int32(n_slots)).astype(jnp.int32)
    order = jnp.argsort(key, stable=True)
    live_per_slot = jnp.bincount(key, length=int(n_slots) + 1)[: int(n_slots)]
    slot_offsets = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(live_per_slot).astype(jnp.int32)]
    )
    return resident_chunk_records.MstepBlockInputs(
        row_image_local=rows.row_image_local[order],
        kernel_row_image_ids=posterior.kernel_row_image_ids[order],
        row_posterior=posterior.row_posterior[order],
        row_fine_rot=rows.row_fine_rot[order],
        projections=None,
        slot_offsets=slot_offsets,
    )


class _MstepOnlyStatsConfig(NamedTuple):
    """The statistics fields the M-step block body reads.

    ``run_resident_mstep_blocks`` runs the M-step alone for a caller that owns
    its own scoring, posterior and statistics (local search, T12). Handing it
    these fields in the shape ``ChunkProgramSpec`` expects keeps one M-step
    body without making that caller build a full statistics configuration.
    The local search sums its norm rows into their images with the scatter-add.
    """

    n_shells: int
    n_optics_groups: int = 1
    float32_bucketed_image_sums: bool = False


def cached_block_projections(tables: resident_chunk_records.ChunkStageTables, block_fine_rot):
    """The global pass's block projections: three gathers by fine-rotation id.

    Split out so the M-step block body takes its projections as operands. The
    global pass keeps gathering them out of the per-iteration caches exactly
    where it did before; local search (T12) has no cacheable fine grid and
    slices the projections it computed for the chunk's own rows instead.
    """

    if tables.union_recon_take is not None:
        # The union cache: the reconstruction window is taken out of the
        # gathered rows and |recon|^2 is formed as the three-cache build formed
        # it (place_windowed_projection_block).
        recon = cache_rows(tables.projection_score_cache, block_fine_rot)[:, tables.union_recon_take]
        recon_abs2 = (jnp.abs(recon) ** 2).astype(jnp.real(recon).dtype)
        if tables.residual_sgd_take is None:
            return recon, recon_abs2, tables.mstep_grid[block_fine_rot]
        # VDAM's subtracted reference: the SGD kernel's projection, which keeps
        # the rows the fine and wavg kernels zero (_sgd_residual_rows).
        residual = recon.at[:, tables.residual_sgd_take].set(
            cache_rows(tables.residual_sgd_cache, block_fine_rot).astype(recon.dtype)
        )
        return recon, recon_abs2, tables.mstep_grid[block_fine_rot], residual
    return (
        tables.projection_recon_cache[block_fine_rot],
        tables.projection_recon_abs2_cache[block_fine_rot],
        tables.mstep_grid[block_fine_rot],
    )


def _resident_mstep_block(
    *,
    block_row_image,
    block_kernel_ids,
    block_posterior,
    block_projections,
    block_sum_ids=None,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    carry: resident_chunk_records.ChunkMstepCarry,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkMstepCarry:
    """One pixel-axis row block: weighted sums, Wavg, noise, both adjoints.

    Statement for statement the body the per-stage loop ran inline; every path
    calls this one copy, so the only differences between them are how the
    block's rows are sliced (static Python slice versus ``dynamic_slice``) and
    where its projections come from. ``block_projections`` is the block's
    ``(projection, |projection|^2, M-step rotations)``, with VDAM's residual
    projection as a fourth entry when it differs (``_sgd_residual_rows``): the
    global pass passes ``cached_block_projections(tables, block_fine_rot)``,
    local search passes a slice of the projections it computed for this chunk.
    """

    from relax.cuda import fine as cuda_fine

    proj, proj_abs2, block_mstep_rotations = block_projections[:3]
    # The SGD residual's projection when it differs from the Wavg one (cached_block_projections).
    residual_proj = block_projections[3] if len(block_projections) > 3 else proj
    if tables.window_logical is None:
        logical_recon_pixels = jnp.asarray(spec.n_recon_pixels, dtype=jnp.int32)
        logical_rect_pixels = jnp.asarray(spec.n_rect, dtype=jnp.int32)
    else:
        logical_recon_pixels = tables.window_logical.recon_pixels
        logical_rect_pixels = tables.window_logical.rect_pixels
    block_optics_groups = (
        None if operands.optics_groups is None else operands.optics_groups[block_row_image]
    )

    if spec.use_translate_sum_kernel:
        summed, summed_masked, ctf_probs, _probs_sum_t = resident_block_programs.resident_block_weighted_sums_kernel(
            block_posterior,
            block_kernel_ids,
            block_row_image,
            operands.recon_image,
            operands.recon_weight,
            operands.noise_image,
            operands.ctf2_over_nv_recon,
            tables.recon_pixel_indices,
            tables.translation_angles,
            image_shape=spec.image_shape,
            n_recon_pixels=int(spec.n_recon_pixels),
        )
    else:
        summed, summed_masked, ctf_probs, _probs_sum_t = resident_block_programs.resident_block_weighted_sums(
            block_posterior,
            block_row_image,
            operands.shifted_recon,
            operands.shifted_noise,
            operands.ctf2_over_nv_recon,
        )

    # The backprojected CTF sum. For CTF-premultiplied images its weight is not
    # the noise statistics' CTF^2 / sigma2 (sparse_pass2_bucket_io.premultiplied_bpref_weights).
    bpref_ctf2_over_nv = operands.ctf2_over_nv_recon
    bpref_ctf_probs = ctf_probs
    if operands.bpref_ctf2_over_nv_recon is not None:
        bpref_ctf2_over_nv = operands.bpref_ctf2_over_nv_recon
        bpref_ctf_probs, _ = resident_block_programs.resident_block_ctf_probs(block_posterior, block_row_image, bpref_ctf2_over_nv)

    if spec.stable_window:
        # The physical tail holds real pixels above RELION's cutoff: they add
        # nothing to the weighted sums, the Wavg terms, the noise or either
        # adjoint (the translate-sum kernel's own bound is the physical count).
        recon_live = jnp.arange(int(spec.n_recon_pixels), dtype=jnp.int32) < logical_recon_pixels
        summed = jnp.where(recon_live, summed, jnp.zeros((), summed.dtype))
        summed_masked = jnp.where(recon_live, summed_masked, jnp.zeros((), summed_masked.dtype))
        ctf_probs = jnp.where(recon_live, ctf_probs, jnp.zeros((), ctf_probs.dtype))
        bpref_ctf_probs = jnp.where(recon_live, bpref_ctf_probs, jnp.zeros((), bpref_ctf_probs.dtype))

    if spec.mstep_subtract_ctf_projection:
        summed = resident_block_programs.resident_block_residual(summed, _probs_sum_t, residual_proj, bpref_ctf2_over_nv, block_row_image)

    # RELION Wavg triplet in the flat-row layout, then its rotation atomics.
    # The host tail picks the sequential RELION reducer when the pass carries
    # the RFLOAT CTF operand and the algebraic form otherwise; follow the same
    # branch, resolved on the host into the program's static configuration.
    if not spec.use_rfloat_ctf_wavg:
        exact_terms = resident_block_programs.resident_block_wavg_algebraic_terms(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            tables.noise_variance_for_noise,
            operands.scale,
            block_row_image,
            block_optics_groups,
        )
    else:
        exact_terms = cuda_fine.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
            jnp.asarray(proj, dtype=jnp.complex64),
            block_kernel_ids,
            jnp.asarray(operands.direct_ctf_rfloat_recon, dtype=jnp.float32),
            jnp.asarray(operands.scale, dtype=jnp.float32),
            jnp.asarray(operands.raw_translated_wavg_for_atomic, dtype=jnp.complex64),
            block_posterior,
            logical_recon_pixels,
        )
    # The image power is added once per chunk after the block walk
    # (add_chunk_wavg_image_power); the blocks add their rows' exact terms at
    # their rectangle positions, the rest of the rectangle receiving nothing.
    wavg_triplet_pixels = cuda_fine.relion_wavg_exact_atomic_flat_rows_triplet_add_f32(
        jnp.asarray(exact_terms, dtype=jnp.float32),
        block_kernel_ids,
        jnp.asarray(tables.exact_positions, dtype=jnp.int32),
        carry.wavg_triplet_pixels,
        logical_rect_pixels,
    )

    noise_summed_masked, noise_ctf_probs = summed_masked, ctf_probs
    if operands.image_noise_scale is not None:
        # The noise and norm terms are linear in these two sums; the backprojection keeps them whole.
        row_scale = jnp.asarray(operands.image_noise_scale, dtype=jnp.float32)[block_row_image][:, None]
        noise_summed_masked = summed_masked * row_scale.astype(summed_masked.real.dtype)
        noise_ctf_probs = ctf_probs * row_scale.astype(ctf_probs.dtype)
    block_shells, block_a2, block_xa = resident_block_programs.resident_block_noise_and_norm(
        proj,
        proj_abs2,
        noise_summed_masked,
        noise_ctf_probs,
        tables.noise_variance_for_noise,
        tables.shell_indices_noise,
        block_row_image,
        block_optics_groups,
        n_shells=int(spec.stats_config.n_shells),
        image_capacity=int(spec.image_capacity),
        float32_bucketed_image_sums=spec.stats_config.float32_bucketed_image_sums,
    )

    if spec.presum_adjoint:
        # The adjoint is linear in the rows, so a projection's rows are summed
        # here and backprojected once per pass (backproject_projection_sums).
        # Padded and other-slot rows carry zero sums.
        Ft_y = carry.Ft_y.at[block_sum_ids].add(summed.astype(carry.Ft_y.dtype))
        Ft_ctf = carry.Ft_ctf.at[block_sum_ids].add(bpref_ctf_probs.astype(carry.Ft_ctf.dtype))
        return carry._replace(
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            wavg_triplet_pixels=wavg_triplet_pixels,
            noise_shells=carry.noise_shells + block_shells,
            a2_per_image=carry.a2_per_image + block_a2,
            xa_per_image=carry.xa_per_image + block_xa,
        )
    Ft_y, Ft_ctf = backproject_block_rows(
        summed, bpref_ctf_probs, block_mstep_rotations, carry.Ft_y, carry.Ft_ctf, tables=tables, spec=spec
    )
    return carry._replace(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=wavg_triplet_pixels,
        noise_shells=carry.noise_shells + block_shells,
        a2_per_image=carry.a2_per_image + block_a2,
        xa_per_image=carry.xa_per_image + block_xa,
    )


def backproject_block_rows(summed, ctf_probs, block_mstep_rotations, Ft_y, Ft_ctf, *, tables, spec):
    """Both windowed adjoints of one row block: ``summed`` into ``Ft_y``, ``ctf_probs`` into ``Ft_ctf``.

    The block body's adjoint (:func:`_resident_mstep_block`), and the tilt M-step's once it has merged
    each row's translation-block partials (``resident_tilt_mstep.tilt_mstep_program``, relax#27).
    """

    runtime_mstep_max_r = None
    if spec.stable_window and tables.window_logical is not None:
        runtime_mstep_max_r = tables.window_logical.mstep_max_r
    adjoint_kwargs = dict(
        window_indices=tables.relion_x_half_recon_indices,
        use_windowed_adjoint=True,
        image_shape=spec.image_shape,
        volume_shape=spec.recon_volume_shape,
        disc_type="linear_interp",
        half_image=True,
        half_volume=True,
        max_r=spec.mstep_max_r,
        relion_x_half=True,
        max_block_bytes=int(spec.max_adjoint_block_bytes),
        runtime_max_r=runtime_mstep_max_r,
    )
    Ft_y = accumulate_adjoint_block_chunked(
        summed, block_mstep_rotations, Ft_y, log_label="resident-y-window", **adjoint_kwargs
    )
    Ft_ctf = accumulate_adjoint_block_chunked(
        ctf_probs, block_mstep_rotations, Ft_ctf, log_label="resident-ctf-window", **adjoint_kwargs
    )
    return Ft_y, Ft_ctf


def _mstep_block_operand_dtypes(
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    projection_dtypes=None,
) -> dict:
    """Dtypes of the M-step block's intermediates, by promotion arithmetic.

    Every statement between the block's operands and its three accumulating
    outputs promotes; none of them casts, except the one explicit
    ``astype(float64)`` on the noise shells. So the output dtypes follow from
    the operand dtypes alone, on the host, without tracing anything:

    * ``summed_masked`` is the translate-and-sum kernel's declared complex64
      output, or ``compute_local_weighted_sums`` of the float32 posterior
      against the noise tile;
    * ``ctf_probs`` is the kernel's declared float32 fourth output when it is
      selected, and otherwise ``compute_local_ctf_sums_from_probs_sum_t`` of
      the float32 translation mass against the gathered CTF row;
    * ``A2`` promotes ``|proj|^2``, ``ctf_probs`` and the noise variance;
    * ``XA`` promotes the noise variance against the real part of
      ``proj * conj(summed_masked)``.

    tests/unit/test_resident_driver_glue.py checks this against ``jax.eval_shape`` of
    the real block stages.
    """

    if projection_dtypes is None:
        # The global pass reads the per-iteration caches; local search has no
        # cache and hands the dtypes of the projections it just computed.
        projection_dtypes = (
            (
                cache_dtype(tables.projection_score_cache),
                jnp.real(jnp.zeros((), dtype=cache_dtype(tables.projection_score_cache))).dtype,
            )
            if tables.union_recon_take is not None
            else (
                tables.projection_recon_cache.dtype,
                tables.projection_recon_abs2_cache.dtype,
            )
        )
    proj_dtype, proj_abs2_dtype = (jnp.dtype(value) for value in projection_dtypes)
    noise_dtype = jnp.dtype(tables.noise_variance_for_noise.dtype)

    if spec.use_translate_sum_kernel:
        summed_masked_dtype = jnp.dtype(jnp.complex64)
    else:
        summed_masked_dtype = jnp.dtype(
            jnp.result_type(jnp.float32, operands.shifted_noise.dtype)
        )
    ctf_probs_dtype = jnp.dtype(jnp.result_type(jnp.float32, operands.ctf2_over_nv_recon.dtype))

    a2_dtype = jnp.dtype(jnp.result_type(proj_abs2_dtype, ctf_probs_dtype, noise_dtype))
    cross_dtype = jnp.dtype(jnp.result_type(proj_dtype, summed_masked_dtype))
    # ``np.zeros`` rather than ``jnp.zeros``: this asks for the real part's
    # dtype, not for a value, and the device version dispatched one
    # ``convert_element_type`` per chunk to allocate a 0-d array that is read
    # for its dtype and thrown away. NumPy's promotion of a real part is the
    # same table JAX consults.
    xa_dtype = jnp.dtype(
        jnp.result_type(noise_dtype, np.zeros((), dtype=cross_dtype).real.dtype)
    )
    return {
        "proj": proj_dtype,
        "proj_abs2": proj_abs2_dtype,
        "summed_masked": summed_masked_dtype,
        "ctf_probs": ctf_probs_dtype,
        "noise": noise_dtype,
        "a2": a2_dtype,
        "xa": xa_dtype,
        # ``resident_block_noise_and_norm`` casts the binned shells to float64
        # before they leave the block, so the carry is float64 whatever the
        # operands promote to.
        "noise_shells": jnp.dtype(jnp.float64),
    }


@lru_cache(maxsize=None)
def zero_block_partials(shapes_and_dtypes: tuple) -> Callable[[], tuple]:
    """One program per capacity class that allocates the zero accumulators.

    ``jnp.zeros`` outside a jit is two eager dispatches, a
    ``convert_element_type`` of the scalar zero and a ``broadcast_in_dim`` to
    the shape; the M-step carry has four of them and the driver builds one
    carry per chunk, which on the early state was 3336 eager dispatches over
    two iterations. Inside a program with static shapes and dtypes the same
    four buffers cost none, and each call still returns fresh buffers, which
    the donated M-step block program requires.

    Keyed on the shapes and dtypes, so a class compiles once and every chunk
    of that class reuses it.
    """

    @jax.jit
    def build():
        return tuple(jnp.zeros(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes)

    return build


def _noise_shell_shape(stats_config) -> tuple:
    """``(n_shells,)``, or ``(G, n_shells)`` with G optics groups."""

    groups = int(stats_config.n_optics_groups)
    return ((groups,) if groups > 1 else ()) + (int(stats_config.n_shells),)


def initial_mstep_carry(
    Ft_y,
    Ft_ctf,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    projection_dtypes=None,
) -> resident_chunk_records.ChunkMstepCarry:
    """Zero-initialized block accumulators with the block stages' own dtypes.

    The per-image ``A2``/``XA`` partials take whatever dtype the noise and
    projection operands promote to. Those dtypes are computed from the operand
    dtypes and the capacity class by :func:`_mstep_block_operand_dtypes`, which
    is host arithmetic; the chunk loop used to learn them by tracing two
    ``jax.eval_shape`` probes per chunk instead, three traces of Python work
    for an answer that is the same for every chunk of a class.
    Zero-initializing (rather than seeding with the first block, as an earlier
    revision did) makes the loop a ``lax.fori_loop`` carry; adding a leading
    zero changes no float value except the unobservable ``-0.0`` case.
    """

    dtypes = _mstep_block_operand_dtypes(
        operands, tables, spec=spec, projection_dtypes=projection_dtypes
    )
    n_class_scale = 2 if int(getattr(spec, "n_classes", 1)) > 1 else 0
    # K>1: the per-image class scale sums are zeroed in the same program.
    wavg_triplet_pixels, noise_shells, a2_per_image, xa_per_image, *scale_sums = zero_block_partials(
        (
            ((spec.image_capacity, int(spec.n_rect), 3), jnp.dtype(jnp.float32)),
            (_noise_shell_shape(spec.stats_config), jnp.dtype(dtypes["noise_shells"])),
            ((spec.image_capacity,), jnp.dtype(dtypes["a2"])),
            ((spec.image_capacity,), jnp.dtype(dtypes["xa"])),
        )
        + (((spec.image_capacity,), jnp.dtype(jnp.float64)),) * n_class_scale
    )()
    class_scale = {}
    if n_class_scale:
        class_scale = dict(scale_xa_per_image=scale_sums[0], scale_aa_per_image=scale_sums[1])
    return resident_chunk_records.ChunkMstepCarry(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=wavg_triplet_pixels,
        noise_shells=noise_shells,
        a2_per_image=a2_per_image,
        xa_per_image=xa_per_image,
        **class_scale,
    )


def resident_mstep_block_at(
    block_start,
    blocks: resident_chunk_records.MstepBlockInputs,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    carry: resident_chunk_records.ChunkMstepCarry,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkMstepCarry:
    """Slice one block out of the chunk's row arrays and run the block body.

    The per-stage loop did these four slices and three gathers as loose eager
    operations, one dispatch each per block. They are the same slices: every
    row capacity is a whole number of blocks, so ``block_start + block_rows``
    never exceeds the capacity and ``dynamic_slice_in_dim`` never clamps, which
    makes the rows it returns the rows the Python slice returned.
    """

    block_rows = int(spec.mstep_block_rows)

    def take(values):
        return jax.lax.dynamic_slice_in_dim(values, block_start, block_rows, axis=0)

    block_sum_ids = None
    if blocks.projections is None:
        block_row_ids = take(blocks.row_fine_rot)
        block_projections = cached_block_projections(tables, block_row_ids)
        if spec.presum_adjoint:
            # The row's fine rotation within its class: its projection id, or
            # a class's (or a streamed chunk's) slot mapped back through the table.
            block_sum_ids = (
                block_row_ids
                if tables.cache_slot_fine_rot is None
                else jnp.asarray(tables.cache_slot_fine_rot, dtype=jnp.int32)[block_row_ids]
            )
    else:
        block_projections = blocks.projections
    if spec.presum_adjoint and blocks.row_sum_ids is not None:
        block_sum_ids = take(blocks.row_sum_ids)
    block_kernel_ids = take(blocks.kernel_row_image_ids)
    block_posterior = take(blocks.row_posterior)
    if blocks.class_row_range is not None:
        # A boundary block's rows of the neighbouring class are treated as
        # padding (no weight, kernel id -1). A block past the capacity is
        # clamped by the slice, but its unclamped positions are all at or past
        # ``hi``, so every row of it is excluded.
        row = block_start + jnp.arange(block_rows, dtype=jnp.int32)
        in_class = (row >= blocks.class_row_range[0]) & (row < blocks.class_row_range[1])
        block_kernel_ids = jnp.where(in_class, block_kernel_ids, jnp.int32(-1))
        block_posterior = jnp.where(in_class[:, None], block_posterior, jnp.zeros((), block_posterior.dtype))
    return _resident_mstep_block(
        block_row_image=take(blocks.row_image_local),
        block_kernel_ids=block_kernel_ids,
        block_posterior=block_posterior,
        block_projections=block_projections,
        block_sum_ids=block_sum_ids,
        operands=operands,
        tables=tables,
        carry=carry,
        spec=spec,
    )


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(4,))
def resident_mstep_block_program(
    block_start,
    blocks: resident_chunk_records.MstepBlockInputs,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    carry: resident_chunk_records.ChunkMstepCarry,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkMstepCarry:
    """One M-step block as one program, keyed on the capacity and pixel class.

    Same statements, same order, same dtypes as the loose dispatch; the only
    change is where the JIT boundary sits. The carry is donated so the two
    half-volumes the windowed adjoint accumulates into keep being updated in
    place, as they are when the adjoint FFI is dispatched on its own.
    """

    return resident_mstep_block_at(
        block_start,
        blocks,
        operands,
        tables,
        carry,
        spec=spec,
    )


def run_resident_mstep_blocks(
    block_projections=None,
    *,
    chunk_projections=None,
    row_capacity: int,
    n_valid_rows: int,
    mstep_block_rows: int,
    image_capacity: int,
    row_image_local,
    kernel_row_image_ids,
    row_posterior,
    recon,
    n_rect: int,
    n_shells: int,
    n_recon_windowed: int,
    noise_variance_for_noise,
    shell_indices_noise,
    exact_positions_device,
    Ft_y_total,
    Ft_ctf_total,
    image_shape,
    recon_volume_shape,
    mstep_current_size,
    mstep_max_r=None,
    relion_x_half_recon_indices,
    max_adjoint_block_bytes,
    glue_jit: bool,
    n_optics_groups: int = 1,
    row_ids=None,
    recon_pixel_indices=None,
    translation_angles=None,
    row_partial: RowPartial | None = None,
    window_logical: resident_chunk_records.WindowLogicalSizes | None = None,
):
    """Walk one chunk's pixel axis in row blocks: Wavg, noise and both adjoints (``glue_jit``:
    ``EngineTuning.resident_glue_jit``, the block program jitted or run eagerly).

    With ``row_partial`` (the pass's partial accumulator pair) the blocks scatter into it, and it is added
    into ``Ft_y_total``/``Ft_ctf_total`` every ``blocks_per_flush`` blocks and after the last (relax#65).

    Exactly one of ``block_projections`` and ``chunk_projections`` is given.

    ``window_logical`` (:func:`~relax.fine_pass.resident_stable_windows.window_logical_sizes`) is set for a
    stable-window pass: the pixel axes are the physical class's, RELION's logical recon and Wavg pixel
    counts and adjoint radius are runtime values, and the physical tail adds nothing.

    ``row_ids`` (int32 ``[row_capacity]``) is the chunk row each M-step row
    reads its ``chunk_projections`` at; ``None`` is the identity. The row
    arrays (``row_image_local``, ``kernel_row_image_ids``, ``row_posterior``)
    are already in that order, and ``n_valid_rows`` counts the rows that carry
    weight, so a caller that puts its live rows first launches only their
    blocks.

    ``block_projections(start, stop)`` returns this block's reconstruction-window
    projection, its ``|proj|^2`` and its M-step rotations. The global pass 2
    gathers all three out of the per-iteration fine-rotation caches; local
    search (T12) slices them out of the projections it computed for the chunk's
    own rows, because its fine grid is not cacheable.

    ``chunk_projections`` (P3-G) is the same three arrays for the **whole**
    chunk, in the layout's flat row order. The caller then does no slicing at
    all: the arrays ride in the stage tables and the block program takes a
    ``dynamic_slice`` of the chunk's row ids and reads its rows inside the jit,
    exactly as the global pass reads its per-iteration caches at the block's
    fine-rotation ids. The row ids are ``0 .. row_capacity-1``, so the read is
    the identity gather of the rows the callback sliced, value for value.

    The body is ``_resident_mstep_block``, the same copy the global pass runs in
    both its per-stage and its jitted form, so no accumulator has a second
    implementation. Only the M-step fields of the stage containers are filled
    here: this entry point runs the M-step alone, and the scoring, posterior and
    statistics fields are unused by that body.
    """

    use_translate_sum_kernel = recon.get("recon_image") is not None
    if use_translate_sum_kernel:
        if recon_pixel_indices is None or translation_angles is None:
            raise ValueError(
                "run_resident_mstep_blocks needs recon_pixel_indices and translation_angles "
                "to translate unshifted per-image operands inside the translate-sum kernel"
            )
    elif recon.get("shifted_recon") is None or recon.get("shifted_noise") is None:
        # Fail closed rather than hand ``None`` to the XLA weighted sums.
        raise ValueError(
            "run_resident_mstep_blocks takes either the unshifted per-image operands "
            "('recon_image'/'noise_image', T16) or the per-chunk pre-shifted tiles "
            "('shifted_recon'/'shifted_noise'); this chunk carries neither "
            f"(keys present: {sorted(k for k, v in recon.items() if v is not None)})."
        )

    if (block_projections is None) == (chunk_projections is None):
        raise ValueError(
            "run_resident_mstep_blocks takes exactly one of block_projections "
            "(a callback returning one block's arrays) and chunk_projections "
            "(the chunk's whole row arrays); got "
            f"block_projections={'set' if block_projections is not None else 'None'} "
            f"and chunk_projections={'set' if chunk_projections is not None else 'None'}."
        )
    if chunk_projections is not None:
        chunk_proj, chunk_proj_abs2, chunk_mstep_rotations = chunk_projections
        for name, value in (
            ("projection", chunk_proj),
            ("|projection|^2", chunk_proj_abs2),
            ("M-step rotations", chunk_mstep_rotations),
        ):
            if int(value.shape[0]) != int(row_capacity):
                raise ValueError(
                    "chunk_projections must carry the chunk's whole row axis: "
                    f"the {name} array has {int(value.shape[0])} rows, the chunk "
                    f"capacity is {int(row_capacity)}."
                )

    spec = resident_chunk_records.ChunkProgramSpec(
        row_capacity=int(row_capacity),
        image_capacity=int(image_capacity),
        n_fine_trans=int(row_posterior.shape[1]),
        n_score_pixels=0,
        n_recon_pixels=int(n_recon_windowed),
        n_rect=int(n_rect),
        mstep_block_rows=int(mstep_block_rows),
        adaptive_fraction=0.0,
        current_size=0,
        mstep_current_size=int(mstep_current_size),
        mstep_max_r=float(int(mstep_current_size) // 2) if mstep_max_r is None else mstep_max_r,
        image_shape=tuple(int(v) for v in image_shape),
        recon_volume_shape=tuple(int(v) for v in recon_volume_shape),
        max_adjoint_block_bytes=int(max_adjoint_block_bytes),
        stats_config=_MstepOnlyStatsConfig(
            n_shells=int(n_shells), n_optics_groups=int(n_optics_groups)
        ),
        use_rfloat_ctf_wavg=recon["direct_ctf_rfloat_recon"] is not None,
        # Unshifted per-image operands take T15's kernel, which translates them
        # inside the reduction; pre-shifted tiles take the XLA statement.
        use_translate_sum_kernel=use_translate_sum_kernel,
        bpref_recon_operand=use_translate_sum_kernel and recon.get("recon_weight") is not None,
        glue_jit=glue_jit,
        stable_window=window_logical is not None,
    )
    operands = resident_chunk_records.ChunkStageOperands(
        score_input=None,
        corr_img_score=None,
        highres_xi2_half=None,
        translation_prior=None,
        shifted_recon=recon.get("shifted_recon"),
        shifted_noise=recon.get("shifted_noise"),
        recon_image=recon.get("recon_image"),
        recon_weight=recon.get("recon_weight"),
        noise_image=recon.get("noise_image"),
        ctf2_over_nv_recon=recon["ctf2_over_nv_recon"],
        direct_ctf_rfloat_recon=recon["direct_ctf_rfloat_recon"],
        image_power_shells=None,
        relion_norm_high_shell=None,
        raw_translated_wavg_rectangle=recon["raw_translated_wavg_rectangle"],
        raw_translated_wavg_for_atomic=recon["raw_translated_wavg_for_atomic"],
        scale=recon["scale"],
        group_ids=None,
        translation_sqdist_ang=None,
        optics_groups=recon.get("optics_groups"),
        bpref_ctf2_over_nv_recon=recon.get("bpref_ctf2_over_nv_recon"),
    )
    tables = resident_chunk_records.ChunkStageTables(
        projection_score_cache=None,
        # P3-G: the chunk's own row arrays stand where the global pass keeps
        # its per-iteration fine-rotation caches, so the block program reads
        # its rows inside the jit instead of taking them from a host callback.
        projection_recon_cache=None if chunk_projections is None else chunk_proj,
        projection_recon_abs2_cache=(
            None if chunk_projections is None else chunk_proj_abs2
        ),
        mstep_grid=None if chunk_projections is None else chunk_mstep_rotations,
        coarse_parent_grid=None,
        fine_translation_parent=None,
        half_weights=None,
        translation_angles=(
            jnp.asarray(translation_angles, dtype=jnp.float32) if use_translate_sum_kernel else None
        ),
        full_to_compact=None,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions=exact_positions_device,
        # The kernel addresses the reconstruction window by pixel index.
        recon_pixel_indices=(
            jnp.asarray(recon_pixel_indices, dtype=jnp.int32) if use_translate_sum_kernel else None
        ),
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        shell_indices_half=None,
        wavg_shell_indices=None,
        wavg_scale_pixel_mask=None,
        window_logical=window_logical,
    )

    block_rows = int(mstep_block_rows)
    carry = None
    if chunk_projections is None:
        chunk_row_ids = None
        projection_dtypes = None
    else:
        # A host-side index vector, so it costs one transfer for the chunk and
        # no eager primitive: the block program slices it and gathers the rows.
        chunk_row_ids = (
            jnp.asarray(np.arange(int(row_capacity), dtype=np.int32))
            if row_ids is None
            else jnp.asarray(row_ids, dtype=jnp.int32)
        )
        projection_dtypes = (chunk_proj.dtype, chunk_proj_abs2.dtype)

    def start_carry(projections):
        return initial_mstep_carry(
            Ft_y_total if row_partial is None else row_partial.y,
            Ft_ctf_total if row_partial is None else row_partial.ctf,
            operands,
            tables,
            spec=spec,
            projection_dtypes=(
                projection_dtypes
                if projections is None
                else (projections[0].dtype, projections[1].dtype)
            ),
        )

    for start in range(0, int(row_capacity), block_rows):
        if start >= int(n_valid_rows):
            # Every row of this block is chunk padding: its posterior is zero,
            # so the weighted sums, the Wavg terms, the noise partials and both
            # adjoint scatters are all exactly zero and adding them changes no
            # accumulator bit.
            break
        stop = start + block_rows
        block = slice(start, stop)
        if chunk_projections is None:
            projections = block_projections(start, stop)
        else:
            projections = None
        if carry is None:
            carry = start_carry(projections)
        if glue_jit:
            carry = resident_mstep_block_program(
                resident_chunk_records.device_int32(start),
                resident_chunk_records.MstepBlockInputs(
                    row_image_local=row_image_local,
                    kernel_row_image_ids=kernel_row_image_ids,
                    row_posterior=row_posterior,
                    row_fine_rot=chunk_row_ids,
                    projections=projections,
                ),
                operands,
                tables,
                carry,
                spec=spec,
            )
            if row_partial is not None:
                carry, Ft_y_total, Ft_ctf_total = row_partial.block_done(
                    carry, Ft_y_total, Ft_ctf_total, block_rows=block_rows
                )
            continue
        carry = _resident_mstep_block(
            block_row_image=row_image_local[block],
            block_kernel_ids=kernel_row_image_ids[block],
            block_posterior=row_posterior[block],
            block_projections=(
                cached_block_projections(tables, chunk_row_ids[block])
                if projections is None
                else projections
            ),
            operands=operands,
            tables=tables,
            carry=carry,
            spec=spec,
        )
        if row_partial is not None:
            carry, Ft_y_total, Ft_ctf_total = row_partial.block_done(
                carry, Ft_y_total, Ft_ctf_total, block_rows=block_rows
            )
    if carry is None:
        # A chunk with no live rows: the accumulators are the incoming ones and
        # the partials are the zeros the statistics program expects.
        carry = start_carry(
            block_projections(0, block_rows) if chunk_projections is None else None
        )
    if row_partial is not None:
        carry, Ft_y_total, Ft_ctf_total = row_partial.flush(carry, Ft_y_total, Ft_ctf_total)
    else:
        Ft_y_total, Ft_ctf_total = carry.Ft_y, carry.Ft_ctf
    carry = resident_block_programs.add_chunk_wavg_image_power(
        carry, operands, tables, row_posterior, kernel_row_image_ids, spec=spec
    )

    return (
        Ft_y_total,
        Ft_ctf_total,
        carry.wavg_triplet_pixels,
        carry.noise_shells,
        carry.a2_per_image,
        carry.xa_per_image,
    )
