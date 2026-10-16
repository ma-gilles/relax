"""RELION image translation kernels on the GPU: the phase-shift translations of scoring and BPref operands (f32 and
f64) and the flat-row translate-and-sum reduction of the resident M-step.

Each wrapper registers the library targets through ``library.ensure_ffi()`` or
``library.ensure_optional_ffi(target)`` before its first FFI call.
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.cuda_backproject import (
    custom_cuda_requested,
)

from relax.cuda import library


@functools.partial(jax.jit, static_argnums=(3,))
def relion_translate_score_f32(
    images: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate score images with RELION's accelerated float32 arithmetic.

    ``translation_angles`` contains RELION's per-translation ``(tx, ty)``
    radians, ``[T, 2]`` shared by the batch or ``[B, T, 2]`` per image (one
    launch for a subtomogram's tilt images, each with its own phases).
    ``pixel_indices`` use RECOVAR's centered half-spectrum layout.
    The CUDA primitive evaluates ``sincosf(x*tx + y*ty)`` and the explicit
    real/imaginary products used by RELION's fine Gaussian scorer. The output
    is flattened in image-major, translation-major order to match
    :func:`apply_half_translation_phases`.

    This is a strict CUDA primitive. Callers that support a non-CUDA fallback
    must select it before invoking this function.
    """

    if images.dtype != jnp.complex64:
        raise TypeError(f"images must be complex64, got {images.dtype}")
    if translation_angles.dtype != jnp.float32:
        raise TypeError(
            f"translation_angles must be float32, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if not (
        (translation_angles.ndim == 2 and translation_angles.shape[1:] == (2,))
        or (translation_angles.ndim == 3 and translation_angles.shape[0] == images.shape[0]
            and translation_angles.shape[2:] == (2,))
    ):
        raise ValueError(
            "translation_angles must have shape (translations, 2) or (batch, translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION score translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION score translation was explicitly requested but custom CUDA is disabled"
        )
    library.ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[-2],
            images.shape[1],
        ),
        jnp.complex64,
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_TRANSLATE_SCORE_F32,
        out_type,
        vmap_method="sequential",
    )(
        images,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(jax.jit, static_argnums=(3,))
def relion_translate_score_f64(
    images: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate score/M-step images with RELION's CUDA double arithmetic."""

    if images.dtype != jnp.complex128:
        raise TypeError(f"images must be complex128, got {images.dtype}")
    if translation_angles.dtype != jnp.float64:
        raise TypeError(
            f"translation_angles must be float64, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION score translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION score translation was explicitly requested but custom CUDA is disabled"
        )
    library.ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex128,
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_TRANSLATE_SCORE_F64,
        out_type,
        vmap_method="sequential",
    )(
        images,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(jax.jit, static_argnums=(4,))
def relion_translate_bpref_f32(
    images: jax.Array,
    weighted_ctf: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate raw images, then apply BPref weights in RELION float32 order.

    RELION evaluates ``sincosf`` and the explicit complex translation before
    multiplying each translated real/imaginary component by ``weighted_ctf``.
    Keeping this as one CUDA primitive avoids the extra rounding boundary in
    RECOVAR's algebraically equivalent weight-then-translate path.
    """

    if images.dtype != jnp.complex64:
        raise TypeError(f"images must be complex64, got {images.dtype}")
    if weighted_ctf.dtype != jnp.float32:
        raise TypeError(f"weighted_ctf must be float32, got {weighted_ctf.dtype}")
    if translation_angles.dtype != jnp.float32:
        raise TypeError(
            f"translation_angles must be float32, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if weighted_ctf.shape != images.shape:
        raise ValueError(
            f"weighted_ctf must have shape {images.shape}, got {weighted_ctf.shape}"
        )
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION BPref translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION BPref translation was explicitly requested but custom CUDA is disabled"
        )
    library.ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex64,
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_TRANSLATE_BPREF_F32,
        out_type,
        vmap_method="sequential",
    )(
        images,
        weighted_ctf,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(jax.jit, static_argnums=(4,))
def relion_translate_bpref_f64(
    images: jax.Array,
    weighted_ctf: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate raw images, then apply BPref weights in RELION double order."""

    if images.dtype != jnp.complex128:
        raise TypeError(f"images must be complex128, got {images.dtype}")
    if weighted_ctf.dtype != jnp.float64:
        raise TypeError(f"weighted_ctf must be float64, got {weighted_ctf.dtype}")
    if translation_angles.dtype != jnp.float64:
        raise TypeError(
            f"translation_angles must be float64, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if weighted_ctf.shape != images.shape:
        raise ValueError(
            f"weighted_ctf must have shape {images.shape}, got {weighted_ctf.shape}"
        )
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION BPref translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION BPref translation was explicitly requested but custom CUDA is disabled"
        )
    library.ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex128,
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_TRANSLATE_BPREF_F64,
        out_type,
        vmap_method="sequential",
    )(
        images,
        weighted_ctf,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(
    jax.jit, static_argnames=("image_shape", "rows_per_block")
)
def relion_translate_sum_flat_rows_f32(
    recon_image: jax.Array,
    noise_image: jax.Array,
    row_image_ids: jax.Array,
    posterior: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    n_valid_rows: jax.Array,
    logical_pixel_count: jax.Array,
    recon_weight: jax.Array | None = None,
    ctf2_over_nv: jax.Array | None = None,
    *,
    image_shape: Tuple[int, int],
    rows_per_block: int = 0,
) -> tuple[jax.Array, ...]:
    """Translate and posterior-weight two per-image operands over packed rows.

    This is the flat-row form of the resident M-step's weighted sums. For every
    row ``r`` with image ``row_image_ids[r]`` and every reconstruction pixel
    ``p`` it returns

    ``summed[r, p]        = sum_t posterior[r, t] * shift_t(recon_image[id, p])``
    ``summed_masked[r, p] = sum_t posterior[r, t] * shift_t(noise_image[id, p])``
    ``probs_sum_t[r]      = sum_t posterior[r, t]``

    The translation is applied inside the reduction with the phase and complex
    rotation of :func:`relion_translate_score_f32`, so the kernel replaces a
    pre-shifted ``[images, translations, pixels]`` tile and its contraction
    without ever materialising the tile. ``pixel_indices`` are RECOVAR's
    centered packed-half indices, the same operand
    :func:`relion_translate_score_f32` takes.

    ``recon_weight`` switches the first operand to the exact RELION BPref
    convention, the one ``prepare_bucket_io`` uses when
    ``relion_exact_bpref_operands`` is selected: pass the raw BPref image as
    ``recon_image`` and the weighted CTF as ``recon_weight``, and the kernel
    reproduces :func:`relion_translate_bpref_f32`, whose imaginary component
    and post-rotation weighting round differently from the score primitive.
    The noise operand always uses the score convention, which is how
    ``shifted_score_half_with_dc`` is built in both modes.

    ``n_valid_rows`` and ``logical_pixel_count`` are device int32 scalars, so a
    capacity-shaped chunk keeps one traced program: rows at or past
    ``n_valid_rows``, rows whose image id is negative, and pixels at or past
    ``logical_pixel_count`` are written as zeros and read nothing. The sum over
    translations is sequential in increasing ``t`` with a rounded multiply and a
    rounded add; the XLA path it replaces contracts the same products at
    ``Precision.HIGHEST``, so the two agree to a few float32 ulp and exactly at
    ``T == 1``.

    ``ctf2_over_nv`` adds the M-step block's fourth output. Supplied, the call
    returns ``(summed, summed_masked, probs_sum_t, ctf_probs)`` with

    ``ctf_probs[r, p] = probs_sum_t[r] != 0 ? probs_sum_t[r] * ctf2[id, p] : 0``

    which is :func:`relax.fine_pass.local_backprojection.compute_local_ctf_sums_from_probs_sum_t`
    term for term, including its predicate on the mass rather than on the
    product, and bitwise against it on the same ``probs_sum_t``. Omitted, the
    call returns the three-output form and reads no CTF operand.

    ``rows_per_block`` selects how many packed rows share one ``sincosf``
    evaluation. Zero picks the kernel's default; it changes performance only,
    never the arithmetic. A launch with at least 64 rows per image (a lone tilt
    particle's M-step block) first writes each image's shifted operands into a
    stream-ordered table of at most 1 GiB and sums from it (relax#98); the
    values, and so the outputs, are the inline path's.
    """

    recon_image = jnp.asarray(recon_image)
    noise_image = jnp.asarray(noise_image)
    row_image_ids = jnp.asarray(row_image_ids)
    posterior = jnp.asarray(posterior)
    translation_angles = jnp.asarray(translation_angles)
    pixel_indices = jnp.asarray(pixel_indices)
    n_valid_rows = jnp.asarray(n_valid_rows, dtype=jnp.int32)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if recon_image.dtype != jnp.complex64 or noise_image.dtype != jnp.complex64:
        raise TypeError(
            "flat-row translate-and-sum expects complex64 image operands, got "
            f"{recon_image.dtype} and {noise_image.dtype}"
        )
    if recon_image.ndim != 2 or noise_image.shape != recon_image.shape:
        raise ValueError(
            "flat-row translate-and-sum expects matching [batch, pixels] image "
            f"operands, got {recon_image.shape} and {noise_image.shape}"
        )
    batch_size, pixel_capacity = (int(size) for size in recon_image.shape)
    if row_image_ids.dtype != jnp.int32 or row_image_ids.ndim != 1:
        raise ValueError(
            "flat-row translate-and-sum expects int32 row_image_ids[Q]"
        )
    row_count = int(row_image_ids.shape[0])
    # [T, 2] shared by every image, or [B, T, 2], one table per image (tilt images; one row per block).
    per_image_angles = translation_angles.ndim == 3
    if translation_angles.dtype != jnp.float32 or translation_angles.ndim not in (2, 3) or (
        translation_angles.shape[-1] != 2
        or (per_image_angles and int(translation_angles.shape[0]) != batch_size)
    ):
        raise ValueError(
            "flat-row translate-and-sum expects float32 translation_angles[T,2] or [B,T,2]"
        )
    n_trans = int(translation_angles.shape[-2])
    if posterior.dtype != jnp.float32 or posterior.shape != (row_count, n_trans):
        raise ValueError(
            "flat-row translate-and-sum expects float32 posterior[Q,T], got "
            f"{posterior.shape} {posterior.dtype}"
        )
    if pixel_indices.dtype != jnp.int32 or pixel_indices.shape != (
        pixel_capacity,
    ):
        raise ValueError(
            "flat-row translate-and-sum expects int32 pixel_indices[P], got "
            f"{pixel_indices.shape} {pixel_indices.dtype}"
        )
    if n_valid_rows.shape != () or logical_pixel_count.shape != ():
        raise ValueError(
            "n_valid_rows and logical_pixel_count must be int32 scalars"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(
            f"image_shape must contain two positive sizes, got {image_shape}"
        )
    if int(rows_per_block) not in (0, 1, 2, 4, 8):
        raise ValueError(
            f"rows_per_block must be 0, 1, 2, 4 or 8, got {rows_per_block}"
        )
    bpref_recon = recon_weight is not None
    if bpref_recon:
        recon_weight = jnp.asarray(recon_weight)
        if recon_weight.dtype != jnp.float32 or recon_weight.shape != (
            batch_size,
            pixel_capacity,
        ):
            raise ValueError(
                "flat-row translate-and-sum expects float32 recon_weight[B,P], "
                f"got {recon_weight.shape} {recon_weight.dtype}"
            )
    else:
        recon_weight = jnp.zeros((1, 1), dtype=jnp.float32)
    write_ctf_probs = ctf2_over_nv is not None
    if write_ctf_probs:
        ctf2_over_nv = jnp.asarray(ctf2_over_nv)
        if ctf2_over_nv.dtype != jnp.float32 or ctf2_over_nv.shape != (
            batch_size,
            pixel_capacity,
        ):
            raise ValueError(
                "flat-row translate-and-sum expects float32 ctf2_over_nv[B,P], "
                f"got {ctf2_over_nv.shape} {ctf2_over_nv.dtype}"
            )
    else:
        ctf2_over_nv = jnp.zeros((1, 1), dtype=jnp.float32)
    if batch_size <= 0 or pixel_capacity <= 0 or row_count <= 0 or n_trans <= 0:
        raise ValueError(
            "flat-row translate-and-sum operands must be non-empty, got "
            f"batch={batch_size} pixels={pixel_capacity} rows={row_count} "
            f"translations={n_trans}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "flat-row translate-and-sum requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "flat-row translate-and-sum was explicitly requested but custom "
            "CUDA is disabled"
        )
    library.ensure_optional_ffi(library.TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32)

    image_h, image_w = (int(size) for size in image_shape)
    outputs = (
        jax.ShapeDtypeStruct((row_count, pixel_capacity), jnp.complex64),
        jax.ShapeDtypeStruct((row_count, pixel_capacity), jnp.complex64),
        jax.ShapeDtypeStruct((row_count,), jnp.float32),
        jax.ShapeDtypeStruct(
            (row_count, pixel_capacity) if write_ctf_probs else (1, 1),
            jnp.float32,
        ),
    )
    results = jax.ffi.ffi_call(
        library.TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32,
        outputs,
        vmap_method="sequential",
    )(
        recon_image,
        noise_image,
        recon_weight,
        ctf2_over_nv,
        row_image_ids,
        posterior,
        translation_angles,
        pixel_indices,
        n_valid_rows,
        logical_pixel_count,
        image_h=np.int64(image_h),
        image_half_width=np.int64(image_w // 2 + 1),
        rows_per_block=np.int64(int(rows_per_block)),
        bpref_recon=np.int64(int(bpref_recon)),
        write_ctf_probs=np.int64(int(write_ctf_probs)),
    )
    # The unused fourth buffer is a 1x1 placeholder; the three-output form
    # never exposes it.
    return tuple(results) if write_ctf_probs else tuple(results[:3])
