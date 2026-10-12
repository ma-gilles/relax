"""RELION's Wavg atomics over a rectangular ``[image, rotation]`` block and over packed flat rows, on the GPU.

Test oracle: moved from relax/cuda/fine.py, where only tests called them. The flat-row atomics are the reference
the live exact-position kernel (``relion_wavg_exact_atomic_flat_rows_triplet_add_f32``) is compared with, and the
rectangular atomics the reference for the flat-row form. Their native handlers stay in the relax library.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from relax.cuda import library


@jax.jit
def relion_wavg_rotation_atomic_runtime_triplet_add_f32(
    terms: jax.Array,
    accumulator: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Run Wavg atomics over a logical prefix of fixed-capacity storage."""

    library.ensure_ffi()
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
        library.TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={1: 0},
        vmap_method="sequential",
    )(terms, accumulator, logical_pixel_count)


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
    library.ensure_optional_ffi(
        library.TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32
    )
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library.TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(terms, row_image_ids, accumulator, logical_pixel_count)
