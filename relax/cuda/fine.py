"""RELION fine-pass kernels on the GPU: the fine diff2 scores (rectangular, masked, pairs, f32 and f64, and the fused
translate-and-score over flat candidate rows) and the Wavg reductions of the fine M-step (rotation-atomic,
sequential and exact-position flat-row triplets).

Each wrapper registers the library targets through ``library._ensure_ffi()`` or
``library._ensure_optional_ffi(target)`` before its first FFI call; the masked fine diff2 and the flat-row Wavg
targets are optional ABIs with support probes here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from recovar.cuda_backproject import (
    custom_cuda_requested,
)

from relax.cuda import library


def _validate_relion_fine_diff2_inputs(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    *,
    real_dtype=jnp.float32,
) -> None:
    real_dtype = jnp.dtype(real_dtype)
    complex_dtype = jnp.dtype(jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64)
    if reference.dtype != complex_dtype:
        raise TypeError(f"reference must be {complex_dtype.name}, got {reference.dtype}")
    if shifted_image.dtype != complex_dtype:
        raise TypeError(f"shifted_image must be {complex_dtype.name}, got {shifted_image.dtype}")
    if weight.dtype != real_dtype:
        raise TypeError(f"weight must be {real_dtype.name}, got {weight.dtype}")
    if full_to_compact.dtype != jnp.int32:
        raise TypeError(
            f"full_to_compact must be int32, got {full_to_compact.dtype}"
        )
    if full_to_compact.ndim != 1 or full_to_compact.shape[0] <= 0:
        raise ValueError(
            "full_to_compact must be a nonempty rank-1 array, got "
            f"{full_to_compact.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION fine diff2 requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION fine diff2 was explicitly requested but custom CUDA is disabled"
        )
    library._ensure_ffi()


def _validate_relion_fine_rectangular_shapes(reference, shifted_image, weight):
    """Validate the rectangular candidate layout shared by both precisions."""
    if reference.ndim != 3 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "rectangular fine diff2 expects reference/shifted rank 3 and "
            f"weight rank 2, got {reference.shape}, {shifted_image.shape}, "
            f"{weight.shape}"
        )
    if (
        reference.shape[0] != shifted_image.shape[0]
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != shifted_image.shape[2]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[1] <= 0
        or reference.shape[2] <= 0
    ):
        raise ValueError(
            "rectangular fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )


def _validate_relion_fine_pairs_shapes(reference, shifted_image, weight):
    """Validate the pairs candidate layout shared by both precisions."""
    if reference.ndim != 3 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "pair fine diff2 expects reference/shifted rank 3 and weight rank "
            f"2, got {reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    if (
        reference.shape != shifted_image.shape
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or reference.shape[2] <= 0
    ):
        raise ValueError(
            "pair fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )


@jax.jit
def relion_fine_diff2_rectangular_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Evaluate RELION's SM80 fine Gaussian tree on a rectangular grid.

    Shapes are ``reference=(B,R,N)``, ``shifted_image=(B,T,N)``,
    ``weight=(B,N)``, ``full_to_compact=(F,)``, and ``initial_diff2=(B,)``.
    One CUDA block evaluates each output hypothesis ``(B,R,T)`` with the
    production 256-lane topology, explicit binary32 FMA rounding boundaries,
    and RELION's final in-kernel ``sum + sum_init`` operation.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    _validate_relion_fine_rectangular_shapes(reference, shifted_image, weight)
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((reference.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        reference.shape[0],
    ):
        raise ValueError(
            "rectangular fine diff2 initial_diff2 must be float32 with shape "
            f"({reference.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact)


def relion_fine_diff2_rectangular_masked_supported() -> bool:
    """Return whether the loaded library exports the masked fine diff2 target.

    The masked kernel is an optional ABI, so a library built before it exists
    stays loadable; callers fall back to the unmasked rectangular kernel.
    """

    try:
        library._ensure_ffi()
        symbol_name = library._OPTIONAL_FFI_REGISTRATIONS[
            library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32
        ][0]
        return getattr(library._get_lib(), symbol_name, None) is not None
    except Exception:
        return False


@jax.jit
def relion_fine_diff2_rectangular_masked_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    candidate_mask: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Rectangular fine diff2 that skips cells whose candidate mask is False.

    Same operands and output layout as :func:`relion_fine_diff2_rectangular_f32`
    plus ``candidate_mask=(B,R,T)`` bool.  Masked cells return 0 without any
    pixel work, mirroring RELION's fine pass, which only evaluates significant
    (orientation, translation) pairs (``makeJobsForDiff2Fine``).  Valid cells
    are bitwise identical to the unmasked kernel.  Optional FFI target: it
    requires a library built with ``RelionFineDiff2RectangularMaskedF32``.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    _validate_relion_fine_rectangular_shapes(reference, shifted_image, weight)
    candidate_mask = jnp.asarray(candidate_mask)
    expected = (reference.shape[0], reference.shape[1], shifted_image.shape[1])
    if candidate_mask.shape != expected or candidate_mask.dtype != jnp.bool_:
        raise ValueError(
            "masked rectangular fine diff2 candidate_mask must be bool with shape "
            f"{expected}, got {candidate_mask.shape} {candidate_mask.dtype}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((reference.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        reference.shape[0],
    ):
        raise ValueError(
            "masked rectangular fine diff2 initial_diff2 must be float32 with shape "
            f"({reference.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    library._ensure_optional_ffi(library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32)
    out_type = jax.ShapeDtypeStruct(expected, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact, candidate_mask)


def _prepare_relion_fine_diff2_fused_translate_flat_rows_operands(
    reference: jax.Array,
    row_image_ids: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    reference = jnp.asarray(reference)
    row_image_ids = jnp.asarray(row_image_ids)
    image = jnp.asarray(image)
    translation_angles = jnp.asarray(translation_angles)
    weight = jnp.asarray(weight)
    full_to_compact = jnp.asarray(full_to_compact)
    if reference.dtype != jnp.complex64 or image.dtype != jnp.complex64:
        raise TypeError(
            "flat-row RELION fine diff2 reference/image must be complex64, got "
            f"{reference.dtype} and {image.dtype}"
        )
    if row_image_ids.dtype != jnp.int32:
        raise TypeError(
            "flat-row RELION fine diff2 row_image_ids must be int32, got "
            f"{row_image_ids.dtype}"
        )
    if translation_angles.dtype != jnp.float32 or weight.dtype != jnp.float32:
        raise TypeError(
            "flat-row RELION fine diff2 angles/weight must be float32, got "
            f"{translation_angles.dtype} and {weight.dtype}"
        )
    if full_to_compact.dtype != jnp.int32:
        raise TypeError(
            "flat-row RELION fine diff2 lookup must be int32, got "
            f"{full_to_compact.dtype}"
        )
    if (
        reference.ndim != 2
        or row_image_ids.shape != (reference.shape[0],)
        or image.ndim != 2
        or translation_angles.ndim not in (2, 3)
        or translation_angles.shape[-1] != 2
        # [T, 2] shared, or [B, T, 2] one table per image (tilt images).
        or (translation_angles.ndim == 3 and translation_angles.shape[0] != image.shape[0])
        or weight.shape != image.shape
        or reference.shape[1] != image.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or image.shape[0] <= 0
        or translation_angles.shape[-2] <= 0
    ):
        raise ValueError(
            "flat-row RELION fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {row_image_ids.shape}, {image.shape}, "
            f"{translation_angles.shape}, {weight.shape}, {full_to_compact.shape}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((image.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        image.shape[0],
    ):
        raise ValueError(
            "flat-row RELION fine diff2 initial_diff2 must be float32 with shape "
            f"({image.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    return (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )


@jax.jit
def relion_fine_diff2_fused_translate_runtime_flat_rows_f32(
    reference: jax.Array,
    row_image_ids: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    logical_current_size: jax.Array,
    initial_diff2: jax.Array | None = None,
    translation_chunk_live: jax.Array | None = None,
) -> jax.Array:
    """Evaluate packed rotation rows inside a fixed physical pixel capacity.

    The scalar logical size controls the exact RELION rectangle traversal;
    physical-only score rows remain inert.

    ``translation_chunk_live`` (bool ``[rows, ceil(T / 4)]``, see
    :func:`relion_fine_diff2_translation_chunk_live`) names the chunks of four
    translations that hold a candidate cell; the kernel skips the others and
    writes their cells as ``+inf``. A traversed chunk computes each of its
    translations as without it, so the cells of live chunks are unchanged.
    """

    (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    ) = _prepare_relion_fine_diff2_fused_translate_flat_rows_operands(
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )
    logical_current_size = jnp.asarray(logical_current_size, dtype=jnp.int32)
    if logical_current_size.shape != ():
        raise ValueError(
            "runtime flat-row RELION fine diff2 logical_current_size must be a scalar, "
            f"got {logical_current_size.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "runtime flat-row fused RELION fine diff2 requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "runtime flat-row fused RELION fine diff2 was explicitly requested "
            "but custom CUDA is disabled"
        )
    library._ensure_ffi()

    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], translation_angles.shape[-2]),
        jnp.float32,
    )
    operands = (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        logical_current_size,
    )
    if translation_chunk_live is None:
        return jax.ffi.ffi_call(
            library._TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_FLAT_ROWS_F32,
            out_type,
            vmap_method="sequential",
        )(*operands)
    expected_live_shape = (
        reference.shape[0],
        # [T, 2] shared or [B, T, 2] per image (tilt images): T is the second-to-last axis.
        -(-int(translation_angles.shape[-2]) // RELION_FINE_DIFF2_TRANSLATION_CHUNK),
    )
    if tuple(translation_chunk_live.shape) != expected_live_shape:
        raise ValueError(
            f"translation_chunk_live must have shape {expected_live_shape}, "
            f"got {tuple(translation_chunk_live.shape)}"
        )
    library._ensure_optional_ffi(library._TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_MASKED_FLAT_ROWS_F32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_MASKED_FLAT_ROWS_F32,
        out_type,
        vmap_method="sequential",
    )(*operands, jnp.asarray(translation_chunk_live, dtype=jnp.uint8))


# Translations per block of the fused fine-diff2 kernels (kRelionFineDiff2Ref3dJobChunk).
RELION_FINE_DIFF2_TRANSLATION_CHUNK = 4


def relion_fine_diff2_translation_chunk_live(candidate_mask: jax.Array) -> jax.Array:
    """Which chunks of four translations of each row hold a candidate cell.

    ``candidate_mask`` is bool ``[rows, T]``; the result is bool
    ``[rows, ceil(T / 4)]``, the kernels' translation blocking.
    """

    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    n_rows, n_translations = candidate_mask.shape
    chunk = RELION_FINE_DIFF2_TRANSLATION_CHUNK
    n_chunks = -(-int(n_translations) // chunk)
    padded = jnp.pad(candidate_mask, ((0, 0), (0, n_chunks * chunk - int(n_translations))))
    return jnp.any(padded.reshape(n_rows, n_chunks, chunk), axis=-1)


@jax.jit
def relion_fine_diff2_pairs_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate RELION's SM80 fine Gaussian tree for compact candidate pairs.

    Shapes are ``reference=(B,P,N)``, ``shifted_image=(B,P,N)``,
    ``weight=(B,N)``, and ``full_to_compact=(F,)``; output is ``(B,P)``.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    _validate_relion_fine_pairs_shapes(reference, shifted_image, weight)
    out_type = jax.ShapeDtypeStruct(reference.shape[:2], jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@jax.jit
def relion_fine_diff2_rectangular_f64(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """ACC_DOUBLE_PRECISION specialization of the rectangular fine tree."""

    _validate_relion_fine_diff2_inputs(
        reference, shifted_image, weight, full_to_compact,
        real_dtype=jnp.float64,
    )
    _validate_relion_fine_rectangular_shapes(reference, shifted_image, weight)
    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
        jnp.float64,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_F64,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@jax.jit
def relion_fine_diff2_pairs_f64(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """ACC_DOUBLE_PRECISION specialization of the compact-pair fine tree."""

    _validate_relion_fine_diff2_inputs(
        reference, shifted_image, weight, full_to_compact,
        real_dtype=jnp.float64,
    )
    _validate_relion_fine_pairs_shapes(reference, shifted_image, weight)
    out_type = jax.ShapeDtypeStruct(reference.shape[:2], jnp.float64)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_PAIRS_F64,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@jax.jit
def relion_wavg_rotation_atomic_runtime_triplet_add_f32(
    terms: jax.Array,
    accumulator: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Run Wavg atomics over a logical prefix of fixed-capacity storage."""

    library._ensure_ffi()
    terms = jnp.asarray(terms)
    accumulator = jnp.asarray(accumulator)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if terms.dtype != jnp.float32 or terms.ndim != 4 or terms.shape[-1] != 3:
        raise ValueError(
            "runtime RELION Wavg triplets require float32 [batch,rotation,pixel,3] terms"
        )
    if accumulator.dtype != jnp.float32 or accumulator.shape != (
        terms.shape[0],
        terms.shape[2],
        3,
    ):
        raise ValueError(
            "runtime RELION Wavg triplets require a matching [batch,pixel,3] accumulator"
        )
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={1: 0},
        vmap_method="sequential",
    )(terms, accumulator, logical_pixel_count)


@jax.jit
def relion_wavg_sequential_runtime_flat_rows_triplet_f32(
    projections: jax.Array,
    row_image_ids: jax.Array,
    raw_ctf: jax.Array,
    scale: jax.Array,
    shifted_images: jax.Array,
    posterior: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Accumulate Wavg triplets over packed candidate rows.

    ``projections`` is ``(Q,P)`` and ``posterior`` is ``(Q,T)``; every row's
    image/CTF/scale/shifted-image address comes from ``row_image_ids`` ``(Q,)``
    instead of the rectangular ``[B,R]`` grid, the same substitution the
    flat-row fine scorer makes.  The CUDA implementation shares the rectangular
    kernel's per-pixel binary32 arithmetic and its translation-storage order, so
    a flattened rectangular problem returns bitwise identical triplets.  Rows
    whose id is negative are padding: they read nothing and stay zero.  The
    output is ``(Q,P,3)`` holding ``[XA, AA, diff2]``.
    """

    projections = jnp.asarray(projections)
    row_image_ids = jnp.asarray(row_image_ids)
    raw_ctf = jnp.asarray(raw_ctf)
    scale = jnp.asarray(scale)
    shifted_images = jnp.asarray(shifted_images)
    posterior = jnp.asarray(posterior)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if projections.dtype != jnp.complex64 or projections.ndim != 2:
        raise ValueError(
            "flat-row RELION Wavg expects complex64 projections[Q,P]"
        )
    row_count, pixel_capacity = projections.shape
    if row_image_ids.dtype != jnp.int32 or row_image_ids.shape != (row_count,):
        raise ValueError("flat-row RELION Wavg expects int32 row_image_ids[Q]")
    if raw_ctf.dtype != jnp.float32 or raw_ctf.ndim != 2 or (
        raw_ctf.shape[1] != pixel_capacity
    ):
        raise ValueError("flat-row RELION Wavg expects float32 raw_ctf[B,P]")
    batch_size = raw_ctf.shape[0]
    if scale.dtype != jnp.float32 or scale.shape != (batch_size,):
        raise ValueError("flat-row RELION Wavg expects float32 scale[B]")
    if (
        shifted_images.dtype != jnp.complex64
        or shifted_images.ndim != 3
        or shifted_images.shape[0] != batch_size
        or shifted_images.shape[2] != pixel_capacity
    ):
        raise ValueError(
            "flat-row RELION Wavg expects complex64 shifted_images[B,T,P]"
        )
    if posterior.dtype != jnp.float32 or posterior.shape != (
        row_count,
        shifted_images.shape[1],
    ):
        raise ValueError("flat-row RELION Wavg expects float32 posterior[Q,T]")
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    if jax.default_backend() != "gpu":
        raise RuntimeError("flat-row RELION Wavg requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("flat-row RELION Wavg requires custom CUDA")
    library._ensure_optional_ffi(
        library._TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32
    )
    output_type = jax.ShapeDtypeStruct((row_count, pixel_capacity, 3), jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32,
        output_type,
        vmap_method="sequential",
    )(
        projections,
        row_image_ids,
        raw_ctf,
        scale,
        shifted_images,
        posterior,
        logical_pixel_count,
    )


@jax.jit
def relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
    terms: jax.Array,
    row_image_ids: jax.Array,
    accumulator: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Atomically add packed-row Wavg triplets into per-image accumulators.

    ``terms`` is ``(Q,P,3)``, ``row_image_ids`` ``(Q,)`` and ``accumulator``
    ``(B,P,3)``.  The launch keeps one block per row, so a flattened
    rectangular problem issues exactly the same multiset of per-cell atomic
    adds as :func:`relion_wavg_rotation_atomic_runtime_triplet_add_f32`, and
    the linear block index matches that kernel's ``rotation + image *
    rotation_count``.  The order in which those adds land is hardware
    scheduled, so the accumulated float32 sums agree bitwise only when the
    summands are exactly representable; the rectangular kernel does not
    reproduce itself bitwise either at realistic rotation counts.  Rows whose
    id is negative contribute nothing.
    """

    terms = jnp.asarray(terms)
    row_image_ids = jnp.asarray(row_image_ids)
    accumulator = jnp.asarray(accumulator)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if terms.dtype != jnp.float32 or terms.ndim != 3 or terms.shape[-1] != 3:
        raise ValueError(
            "flat-row RELION Wavg atomics require float32 [row, pixel, 3] terms"
        )
    row_count, pixel_capacity = terms.shape[0], terms.shape[1]
    if row_image_ids.dtype != jnp.int32 or row_image_ids.shape != (row_count,):
        raise ValueError(
            "flat-row RELION Wavg atomics require int32 row_image_ids[Q]"
        )
    if (
        accumulator.dtype != jnp.float32
        or accumulator.ndim != 3
        or accumulator.shape[1] != pixel_capacity
        or accumulator.shape[2] != 3
    ):
        raise ValueError(
            "flat-row RELION Wavg atomics require a float32 [batch,pixel,3] accumulator"
        )
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    library._ensure_optional_ffi(
        library._TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32
    )
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(terms, row_image_ids, accumulator, logical_pixel_count)


def relion_wavg_exact_atomic_flat_rows_triplet_add_f32(
    exact_terms: jax.Array,
    row_image_ids: jax.Array,
    exact_positions: jax.Array,
    accumulator: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """:func:`relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32` on the exact pixels only.

    ``exact_terms`` is ``(Q, P_exact, 3)`` and ``exact_positions`` ``(P_exact,)``
    their positions in the ``(B, P_rect, 3)`` accumulator's rectangle; the
    rectangle's other positions would have received zeros, which change no
    accumulator value. Each image's run of rows is summed in row order and
    added with one atomic per cell, as in the rectangle form. Positions at or
    past ``logical_pixel_count`` are skipped.
    """

    exact_terms = jnp.asarray(exact_terms)
    row_image_ids = jnp.asarray(row_image_ids)
    exact_positions = jnp.asarray(exact_positions)
    accumulator = jnp.asarray(accumulator)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if exact_terms.dtype != jnp.float32 or exact_terms.ndim != 3 or exact_terms.shape[-1] != 3:
        raise ValueError("exact-position RELION Wavg atomics require float32 [row, exact pixel, 3] terms")
    row_count, n_exact = exact_terms.shape[0], exact_terms.shape[1]
    if row_image_ids.dtype != jnp.int32 or row_image_ids.shape != (row_count,):
        raise ValueError("exact-position RELION Wavg atomics require int32 row_image_ids[Q]")
    if exact_positions.dtype != jnp.int32 or exact_positions.shape != (n_exact,):
        raise ValueError("exact-position RELION Wavg atomics require int32 exact_positions[P_exact]")
    if (
        accumulator.dtype != jnp.float32
        or accumulator.ndim != 3
        or accumulator.shape[2] != 3
        or accumulator.shape[1] < n_exact
    ):
        raise ValueError("exact-position RELION Wavg atomics require a float32 [batch,rect pixel,3] accumulator")
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    library._ensure_optional_ffi(library._TARGET_RELION_WAVG_EXACT_ATOMIC_FLAT_ROWS_TRIPLET_ADD_F32)
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_WAVG_EXACT_ATOMIC_FLAT_ROWS_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={3: 0},
        vmap_method="sequential",
    )(exact_terms, row_image_ids, exact_positions, accumulator, logical_pixel_count)


def relion_wavg_sequential_runtime_flat_rows_triplet_f32_supported() -> bool:
    """Return whether the loaded library exports the flat-row Wavg target."""

    return library._optional_target_supported(
        library._TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32
    )


def relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported() -> bool:
    """Return whether the loaded library exports the flat-row Wavg atomics."""

    return library._optional_target_supported(
        library._TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32
    )
