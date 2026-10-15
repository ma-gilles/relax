"""RELION's direct fine-pass ``diff2`` scoring of a padded pass-2 bucket (``scripts/lib/bucketed_pass2.py``).

The CUDA REF3D fine kernel's lane-tree sum, its common-minimum diff2-to-log-weight conversion and the optional
candidate-masked fused kernel, over ``(images, rotations, translations)`` bucket tensors.

Script library: moved from relax/fine_pass/scoring.py, where no relax command reached it; relax's pass 2 scores
with relax/fine_pass/resident_scoring.py, which follows the same conversion order.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from relax.fine_pass.scoring import (
    _RELION_CUDA_FINE_REF3D_BLOCK_SIZE,
    _RELION_FINE_DIFF2_FUSED_FFI_ENV,
    _relion_cuda_fine_reduce_lanes,
    relion_cuda_fine_pixel_weights,
)
from relax.runtime.env_flags import parse_env_flag

_RELION_FINE_DIFF2_MASKED_ENV = "RELAX_RELION_FINE_DIFF2_MASKED"
_DEFAULT_RELION_FINE_DIFF2_MASKED = True


def _fine_diff2_masked_enabled() -> bool:
    """Skip candidate-mask-excluded cells inside the fused rectangular fine kernel.

    RELION's fine pass evaluates only significant (orientation, translation)
    pairs; the rectangular K=1 scorer evaluates every cell of the padded bucket
    (all translations for every row), most of which the candidate mask then
    discards.  At the matched hp3 regime only 5.7-7.6% of the cells are
    admitted, and the matched pair (job 14082785) measured warm iteration 2 at
    67.4/67.4 s masked off versus 65.2/64.6 s masked on, with every numerical
    delta inside the same-source band, so this is on by default.  A library
    without the optional target falls back to the unmasked kernel.
    """

    return parse_env_flag(
        _RELION_FINE_DIFF2_MASKED_ENV, default=_DEFAULT_RELION_FINE_DIFF2_MASKED
    )


def _relion_cuda_fine_diff2_sum(
    reference,
    shifted_image,
    pixel_weight,
    relion_full_to_compact=None,
    *,
    use_fused_ffi=False,
    candidate_mask=None,
):
    """Accumulate direct Gaussian diff2 without materializing ``(..., N)``.

    Pinned RELION CUDA's ``REF3D=true, DATA3D=false`` path uses
    ``D2F_BLOCK_SIZE_REF3D=256`` and ``D2F_CHUNK_REF3D=7``. The chunk controls
    translation batching; thread ``tid`` still accumulates pixels
    ``tid + pass * 256`` sequentially in XFLOAT (float32), followed by the
    shared-memory 128,64,...,1 reduction tree. Keeping only the 256 lanes per
    hypothesis avoids the much larger hypothesis-by-pixel temporary.
    """

    complex_dtype = jnp.result_type(reference, shifted_image, jnp.complex64)
    real_dtype = jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32
    reference = jnp.asarray(reference, dtype=complex_dtype)
    shifted_image = jnp.asarray(shifted_image, dtype=complex_dtype)
    pixel_weight = jnp.asarray(pixel_weight, dtype=real_dtype)
    n_values = int(reference.shape[-1])
    if shifted_image.shape[-1] != n_values or pixel_weight.shape[-1] != n_values:
        raise ValueError(
            "RELION CUDA fine diff2 operands must have the same pixel count: "
            f"reference={reference.shape[-1]}, shifted={shifted_image.shape[-1]}, "
            f"weight={pixel_weight.shape[-1]}"
        )
    output_shape = jnp.broadcast_shapes(
        reference.shape[:-1], shifted_image.shape[:-1], pixel_weight.shape[:-1]
    )
    if n_values == 0:
        return jnp.zeros(output_shape, dtype=real_dtype)

    if relion_full_to_compact is None:
        relion_full_to_compact = jnp.arange(n_values, dtype=jnp.int32)
    else:
        relion_full_to_compact = jnp.asarray(relion_full_to_compact, dtype=jnp.int32)
        if relion_full_to_compact.ndim != 1:
            raise ValueError(
                "RELION full-to-compact lookup must be one-dimensional, got "
                f"{relion_full_to_compact.shape}"
            )
    if bool(use_fused_ffi) or parse_env_flag(
        _RELION_FINE_DIFF2_FUSED_FFI_ENV,
        default=False,
    ):
        from relax.cuda import fine as cuda_fine

        if reference.ndim == 4:
            if (
                shifted_image.ndim != 4
                or pixel_weight.ndim != 4
                or reference.shape[2] != 1
                or shifted_image.shape[1] != 1
                or pixel_weight.shape[1:3] != (1, 1)
                or reference.shape[0] != shifted_image.shape[0]
                or reference.shape[0] != pixel_weight.shape[0]
            ):
                raise ValueError(
                    "fused rectangular fine diff2 received unsupported broadcast shapes: "
                    f"{reference.shape}, {shifted_image.shape}, {pixel_weight.shape}"
                )
            if (
                candidate_mask is not None
                and real_dtype == jnp.float32
                and cuda_fine.relion_fine_diff2_rectangular_masked_supported()
            ):
                # RELION-style pair pruning: skip (row, translation) cells the
                # candidate mask excludes.  Valid cells stay bitwise identical.
                return cuda_fine.relion_fine_diff2_rectangular_masked_f32(
                    reference[:, :, 0, :],
                    shifted_image[:, 0, :, :],
                    pixel_weight[:, 0, 0, :],
                    relion_full_to_compact,
                    jnp.asarray(candidate_mask, dtype=bool),
                )
            fine_diff2_rectangular = (
                cuda_fine.relion_fine_diff2_rectangular_f64
                if real_dtype == jnp.float64
                else cuda_fine.relion_fine_diff2_rectangular_f32
            )
            return fine_diff2_rectangular(
                reference[:, :, 0, :],
                shifted_image[:, 0, :, :],
                pixel_weight[:, 0, 0, :],
                relion_full_to_compact,
            )
        if reference.ndim == 3 and shifted_image.ndim == 3 and pixel_weight.ndim == 3:
            if (
                reference.shape == shifted_image.shape
                and pixel_weight.shape[1] == 1
                and reference.shape[0] == pixel_weight.shape[0]
            ):
                fine_diff2_pairs = (
                    cuda_fine.relion_fine_diff2_pairs_f64
                    if real_dtype == jnp.float64
                    else cuda_fine.relion_fine_diff2_pairs_f32
                )
                return fine_diff2_pairs(
                    reference,
                    shifted_image,
                    pixel_weight[:, 0, :],
                    relion_full_to_compact,
                )
            if (
                reference.shape[1] == 1
                and shifted_image.shape[0] == 1
                and pixel_weight.shape[:2] == (1, 1)
            ):
                fine_diff2_rectangular = (
                    cuda_fine.relion_fine_diff2_rectangular_f64
                    if real_dtype == jnp.float64
                    else cuda_fine.relion_fine_diff2_rectangular_f32
                )
                return fine_diff2_rectangular(
                    reference[:, 0, :][None, :, :],
                    shifted_image[0, :, :][None, :, :],
                    pixel_weight[0, 0, :][None, :],
                    relion_full_to_compact,
                )[0]
        raise ValueError(
            "fused fine diff2 received unsupported operand ranks/shapes: "
            f"{reference.shape}, {shifted_image.shape}, {pixel_weight.shape}"
        )
    full_image_size = int(relion_full_to_compact.shape[0])
    block_size = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
    n_passes = (full_image_size + block_size - 1) // block_size
    padded_size = n_passes * block_size
    relion_full_to_compact = jnp.pad(
        relion_full_to_compact,
        [(0, padded_size - full_image_size)],
        constant_values=-1,
    )
    lanes = jnp.zeros(output_shape + (block_size,), dtype=real_dtype)

    def accumulate_pass(pass_index, lane_values):
        start = pass_index * block_size
        compact_rows = jax.lax.dynamic_slice_in_dim(
            relion_full_to_compact, start, block_size, axis=-1
        )
        valid_pixel = compact_rows >= 0
        safe_rows = jnp.where(valid_pixel, compact_rows, 0)
        ref_pass = jnp.take(reference, safe_rows, axis=-1)
        img_pass = jnp.take(shifted_image, safe_rows, axis=-1)
        weight_pass = jnp.take(pixel_weight, safe_rows, axis=-1)
        diff_real = ref_pass.real - img_pass.real
        diff_imag = ref_pass.imag - img_pass.imag
        terms = (
            (diff_real * diff_real + diff_imag * diff_imag)
            * jnp.asarray(0.5, dtype=real_dtype)
            * weight_pass
        )
        terms = jnp.where(valid_pixel, terms, jnp.asarray(0.0, dtype=real_dtype))
        return lane_values + terms

    lanes = jax.lax.fori_loop(0, n_passes, accumulate_pass, lanes)
    return _relion_cuda_fine_reduce_lanes(lanes)


@jax.jit
def _relion_cuda_fine_diff2_min(diff2, candidate_mask):
    """Return one finite XFLOAT minimum per image over a raw diff2 tensor."""

    minimum = _relion_cuda_fine_partition_diff2_min_or_inf(diff2, candidate_mask)
    return jnp.where(
        jnp.isfinite(minimum),
        minimum,
        jnp.asarray(0.0, dtype=minimum.dtype),
    )


@jax.jit
def _relion_cuda_fine_partition_diff2_min_or_inf(diff2, candidate_mask):
    """Reduce one partition, retaining ``+inf`` for all-invalid images."""

    diff2 = jnp.asarray(diff2)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    if diff2.shape != candidate_mask.shape:
        raise ValueError(
            "RELION raw diff2 and candidate mask shapes must match: "
            f"diff2={diff2.shape}, mask={candidate_mask.shape}"
        )
    if diff2.ndim < 2:
        raise ValueError(f"RELION raw diff2 needs a leading image axis, got {diff2.shape}")
    valid = candidate_mask & jnp.isfinite(diff2)
    reduction_axes = tuple(range(1, diff2.ndim))
    return jnp.min(jnp.where(valid, diff2, jnp.inf), axis=reduction_axes)


@jax.jit
def _relion_cuda_fine_diff2_to_scores(
    diff2,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    *,
    min_diff2=None,
):
    """Apply RELION's XFLOAT fine diff2-to-log-weight conversion order.

    RELION first finds one common minimum over the full valid fine candidate
    set for each image. Its CUDA conversion kernel then evaluates, in XFLOAT,
    ``((orientation_log_prior + translation_log_prior) + min_diff2) - diff2``.
    The common-min term cancels algebraically in normalized probabilities but
    its placement changes float32 tie-breaking at diff2 magnitudes around 1e3.

    ``min_diff2`` may be supplied by a caller that splits one image's full
    candidate set across score chunks or classes. Otherwise it is computed
    over the candidate set represented by ``diff2``. K-class callers must
    therefore supply an external minimum spanning every class; a per-class
    call is not a claim of K-class bit parity.
    """

    diff2 = jnp.asarray(diff2)
    real_dtype = diff2.dtype
    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=real_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=real_dtype)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    valid = candidate_mask & jnp.isfinite(diff2)
    if min_diff2 is None:
        local_min = _relion_cuda_fine_diff2_min(diff2, candidate_mask)
    else:
        local_min = jnp.asarray(min_diff2, dtype=real_dtype)
    has_valid = jnp.any(valid, axis=tuple(range(1, diff2.ndim)))
    local_min = jnp.where(has_valid, local_min, jnp.asarray(0.0, dtype=real_dtype))
    min_shape = (diff2.shape[0],) + (1,) * (diff2.ndim - 1)
    # RELION's exponentiation kernel rejects candidates below the supplied
    # global minimum. This is normally impossible for a self-consistent
    # partition, but is observable at cross-partition float32 boundaries.
    valid = valid & (diff2 >= local_min.reshape(min_shape))

    scores = rotation_log_prior + translation_log_prior
    scores = jax.lax.optimization_barrier(scores)
    scores = scores + local_min.reshape(min_shape)
    scores = jax.lax.optimization_barrier(scores)
    scores = scores - diff2
    scores = jnp.where(valid & jnp.isfinite(scores), scores, -jnp.inf)
    return scores


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2_raw(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
    candidate_mask=None,  # (B, R, T) bool: skip excluded cells in the CUDA kernel
):
    """Return positive float32 RELION fine-pass costs without priors or centering.

    ``candidate_mask`` (opt-in via :func:`_fine_diff2_masked_enabled`) makes the
    fused CUDA path skip excluded (row, translation) cells like RELION's
    ``makeJobsForDiff2Fine``; those cells return 0, which every consumer
    already masks (``candidate_mask & isfinite``).
    """

    weights = relion_cuda_fine_pixel_weights(
        corr_img_score, jnp.asarray(half_weights)[None, :]
    )
    diff2 = _relion_cuda_fine_diff2_sum(
        proj_half[:, :, None, :],
        shifted_corrected[:, None, :, :],
        weights[:, None, None, :],
        relion_full_to_compact,
        use_fused_ffi=use_fused_ffi,
        candidate_mask=candidate_mask,
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=diff2.dtype)[:, None, None]
    return diff2


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (B, R) real
    translation_log_prior,  # (B, T) real
    candidate_mask,  # (B, R, T) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    min_diff2=None,  # (B,) optional external common minimum across chunks/classes
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """RELION GPU-style direct ``diff2`` scoring for pass-2 diagnostics.

    RELION's CUDA fine-search kernel first corrects the image by the same
    scalar factors carried by the projection-norm weight, then accumulates a
    direct ``|Fref - Fimg_corrected_shift|^2 * corr_img`` form.  This is
    algebraically equivalent to the dense cross-minus-norm expression but has
    different float32 rounding. The positive diff2 values are converted with
    RELION's common-min and prior-addition order below.

    Extremely small CTF/noise combinations can still overflow the direct form
    on long 256px runs.  Treat non-finite candidates as impossible hypotheses
    rather than letting NaNs enter posterior and noise accumulators.
    """

    score_dtype = (
        jnp.float64
        if jnp.result_type(shifted_corrected, corr_img_score) == jnp.complex128
        or jnp.asarray(corr_img_score).dtype == jnp.float64
        else jnp.float32
    )
    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=score_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=score_dtype)
    diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        relion_full_to_compact,
        highres_xi2_half,
        use_fused_ffi=use_fused_ffi,
        candidate_mask=candidate_mask if _fine_diff2_masked_enabled() else None,
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        rotation_log_prior[:, :, None],
        translation_log_prior[:, None, :],
        candidate_mask,
        min_diff2=min_diff2,
    )
