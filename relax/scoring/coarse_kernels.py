"""JAX scoring kernels for the coarse pass (RELION coarse GEMMs, normalized-CC rescoring, certificates)."""

import os
from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def _relion_coarse_128lane_float32_reduce(values):
    """Reduce packed pixel contributions like ``diff2_CC_coarse``.

    RELION assigns pixels ``lane + 128 * pass`` to each CUDA lane, accumulates
    the passes sequentially in float32, then applies a 64, 32, ..., 1 shared
    memory tree.  ``values`` may have arbitrary leading dimensions; its last
    axis is the packed FFTW pixel identity.
    """

    values = jnp.asarray(values, dtype=jnp.float32)
    n_pixels = int(values.shape[-1])
    n_passes = (n_pixels + 127) // 128
    padded = jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, n_passes * 128 - n_pixels)])
    passes = padded.reshape(values.shape[:-1] + (n_passes, 128))
    lanes = jnp.zeros(values.shape[:-1] + (128,), dtype=jnp.float32)
    for pass_index in range(n_passes):
        lanes = lanes + passes[..., pass_index, :]
    for stride in (64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[..., :stride] + lanes[..., stride : 2 * stride]
    return lanes[..., 0]


@jax.jit
def _relion_coarse_cc_atomic_score_from_components(numerator, norm):
    """Apply RELION's 128 identical atomic additions to coarse CC scores.

    ``cuda_kernel_diff2_CC_coarse`` deliberately has every one of its 128
    threads atomically add the same reduced score divided by 128.  The
    repeated float32 rounding is observable at near ties and is not equivalent
    to multiplying one rounded contribution by 128. The loop is unrolled so the
    additions fuse into one elementwise pass over a full coarse score cube.
    """

    numerator = jnp.asarray(numerator, dtype=jnp.float32)
    norm = jnp.asarray(norm, dtype=jnp.float32)
    contribution = numerator / (
        jnp.asarray(128.0, dtype=jnp.float32)
        * jnp.sqrt(jnp.maximum(norm, jnp.asarray(1e-30, dtype=jnp.float32)))
    )

    def add_once(_, accumulated):
        return accumulated + contribution

    return jax.lax.fori_loop(
        0,
        128,
        add_once,
        jnp.zeros_like(contribution),
        unroll=True,
    )


@jax.jit
def _relion_coarse_normalized_cc_rescore_jax(
    shifted_candidates,
    score_weight_candidates,
    projection_candidates,
    half_weights,
    fftw_order,
):
    """Portable JAX replay of RELION's float32 coarse lane tree."""

    shifted = jnp.asarray(shifted_candidates, dtype=jnp.complex64)[..., fftw_order]
    score_weight = jnp.asarray(score_weight_candidates, dtype=jnp.float32)[..., fftw_order]
    projection = jnp.asarray(projection_candidates, dtype=jnp.complex64)[..., fftw_order]
    weights = jnp.asarray(half_weights, dtype=jnp.float32)[fftw_order]
    numerator_pixels = jnp.real(jnp.conj(shifted) * projection) * weights
    norm_pixels = score_weight * (jnp.abs(projection) ** 2) * weights
    numerator = _relion_coarse_128lane_float32_reduce(numerator_pixels)
    norm = _relion_coarse_128lane_float32_reduce(norm_pixels)
    return _relion_coarse_cc_atomic_score_from_components(numerator, norm)


def relion_coarse_normalized_cc_rescore(
    shifted_candidates,
    score_weight_candidates,
    projection_candidates,
    half_weights,
    fftw_order,
    *,
    projector_full=None,
    rotation_matrices=None,
    current_size=None,
    padding_factor=None,
    projector_max_r=None,
    translation_angles=None,
    numerator_weight_candidates=None,
):
    """Rescore bounded candidates with RELION's native coarse CUDA tree.

    Inputs have shape ``(..., n_pixels)`` in RECOVAR compact centered-row
    order.  ``fftw_order`` maps that last axis to RELION's packed current-size
    FFTW order. On GPU the custom kernel preserves CUDA operand contraction,
    the 128-lane tree, ``sqrtf``, and RELION's repeated atomic additions. The
    JAX implementation remains a portable CPU fallback for structural tests.
    """

    shifted = jnp.asarray(shifted_candidates, dtype=jnp.complex64)
    score_weight = jnp.asarray(score_weight_candidates, dtype=jnp.float32)
    if shifted.shape != score_weight.shape or shifted.ndim < 2:
        raise ValueError(
            "shifted and score-weight candidates must have one common "
            f"rank-2-or-higher shape, got {shifted.shape} and "
            f"{score_weight.shape}",
        )
    native_texture_requested = projector_full is not None
    native_texture_args = (
        rotation_matrices,
        current_size,
        padding_factor,
        projector_max_r,
    )
    if native_texture_requested and any(value is None for value in native_texture_args):
        raise ValueError(
            "native texture normalized-CC replay requires rotations, current "
            "size, padding factor, and projector maximum radius",
        )
    projection = None
    if projection_candidates is not None:
        projection = jnp.asarray(projection_candidates, dtype=jnp.complex64)
        if projection.shape != shifted.shape:
            raise ValueError(
                "projection candidates must match shifted candidates, got "
                f"{projection.shape} and {shifted.shape}",
            )
    if jax.default_backend() == "gpu":
        from recovar import cuda_backproject

        from relax.cuda import kernels as em_cuda_kernels

        if cuda_backproject.custom_cuda_requested():
            n_pixels = int(shifted.shape[-1])
            leading_shape = shifted.shape[:-1]
            if native_texture_requested:
                rotations = jnp.asarray(rotation_matrices, dtype=jnp.float32)
                if rotations.shape != leading_shape + (3, 3):
                    raise ValueError(
                        "candidate rotations must match candidate leading shape, "
                        f"got {rotations.shape} for {leading_shape}",
                    )
                candidate_translation_angles = None
                if translation_angles is not None:
                    candidate_translation_angles = jnp.asarray(
                        translation_angles,
                        dtype=jnp.float32,
                    )
                    if candidate_translation_angles.shape != leading_shape + (2,):
                        raise ValueError(
                            "translation angles must match candidate leading shape, "
                            f"got {candidate_translation_angles.shape} for {leading_shape}",
                        )
                    candidate_translation_angles = candidate_translation_angles.reshape(
                        -1, 2
                    )
                native_numerator_weight = None
                if numerator_weight_candidates is not None:
                    native_numerator_weight = jnp.asarray(
                        numerator_weight_candidates,
                        dtype=jnp.float32,
                    )
                    if native_numerator_weight.shape != shifted.shape:
                        raise ValueError(
                            "numerator weights must match image candidates, got "
                            f"{native_numerator_weight.shape} and {shifted.shape}",
                        )
                    native_numerator_weight = native_numerator_weight.reshape(
                        -1, n_pixels
                    )
                native_scores = (
                    em_cuda_kernels.relion_coarse_normalized_cc_native_texture_pairs_f32(
                        jnp.asarray(projector_full, dtype=jnp.complex64),
                        rotations.reshape(-1, 3, 3),
                        shifted.reshape(-1, n_pixels),
                        score_weight.reshape(-1, n_pixels),
                        jnp.asarray(half_weights, dtype=jnp.float32),
                        jnp.asarray(fftw_order, dtype=jnp.int32),
                        int(current_size),
                        int(padding_factor),
                        int(projector_max_r),
                        translation_angles=candidate_translation_angles,
                        numerator_weight=native_numerator_weight,
                    )
                )
                return native_scores.reshape(leading_shape)
            if projection is None:
                raise ValueError(
                    "preprojected normalized-CC replay requires projection candidates"
                )
            native_shape = (1, int(shifted.size // n_pixels), n_pixels)
            native_scores = em_cuda_kernels.relion_coarse_normalized_cc_pairs_f32(
                shifted.reshape(native_shape),
                score_weight.reshape(native_shape),
                projection.reshape(native_shape),
                jnp.asarray(half_weights, dtype=jnp.float32),
                jnp.asarray(fftw_order, dtype=jnp.int32),
            )
            return native_scores.reshape(leading_shape)
    if native_texture_requested:
        raise RuntimeError(
            "native texture normalized-CC replay requires custom CUDA on a JAX GPU"
        )
    if projection is None:
        raise ValueError("portable normalized-CC replay requires projection candidates")
    return _relion_coarse_normalized_cc_rescore_jax(
        shifted,
        score_weight,
        projection,
        jnp.asarray(half_weights, dtype=jnp.float32),
        jnp.asarray(fftw_order, dtype=jnp.int32),
    )


def _e_step_block_score_components(
    shifted,
    ctf2_over_nv,
    proj_weighted,
    proj_abs2,
    n_images,
    n_trans,
):
    """Return the cross and model-energy GEMM components of one rotation block.

    ``cross[i, r, t] = -2 Re(conj(shifted[i, t]) . proj_weighted[r])`` recovers
    the full inner product from half-spectrum pixels (the half weights are
    absorbed into the projections once per block), and
    ``norms[i, r] = ctf2_over_nv[i] . proj_abs2[r]`` is the model energy. Every
    dense scorer (residual, windowed and normalized-CC) builds its score from
    these two HIGHEST-precision GEMMs. The cross term is one real GEMM over
    ``[Re, Im]``-packed operands, ``Re(conj(s) . p) = s_re . p_re + s_im . p_im``,
    so the unused imaginary half of a complex product is never formed.
    """

    rot_block_size = proj_weighted.shape[0]
    shifted_packed = jnp.concatenate([shifted.real, shifted.imag], axis=-1)
    proj_packed = jnp.concatenate([proj_weighted.real, proj_weighted.imag], axis=-1)
    cross = -2.0 * jnp.matmul(
        shifted_packed,
        proj_packed.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv,
        proj_abs2.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    return cross, norms


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10))
def _e_step_block_scores_windowed(
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_windowed_weighted,
    proj_abs2_windowed,
    half_weights_windowed,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """E-step for one rotation block using windowed half-spectrum GEMMs."""

    del batch_norm, half_weights_windowed, n_windowed, image_shape, volume_shape
    cross, norms = _e_step_block_score_components(
        shifted_windowed,
        ctf2_over_nv_windowed,
        proj_windowed_weighted,
        proj_abs2_windowed,
        n_images,
        n_trans,
    )
    residuals = cross + norms[..., None]
    return -0.5 * residuals


def coarse_gemm_float64_requested() -> bool:
    """Whether ``RELAX_COARSE_GEMM_FLOAT64=1`` asks for binary64 coarse GEMMs."""

    token = os.environ.get("RELAX_COARSE_GEMM_FLOAT64", "0").strip()
    if token not in {"0", "1"}:
        raise ValueError(f"Unsupported RELAX_COARSE_GEMM_FLOAT64={token!r}")
    return token == "1"


def relion_coarse_gemm_terms(
    projected_reference,
    shifted_corrected,
    pixel_weight,
    actual_image_count,
    *,
    n_images: int,
    n_trans: int,
    wide,
):
    """RELION's coarse cross and model-energy terms as two real-packed GEMMs.

    Returns ``X = Re(conj(w y) . p)`` as ``[B, T, R]``, the model energy
    ``A = w . |p|^2`` as ``[B, R]``, the image energy ``C = w . |y|^2`` as
    ``[B, T]`` and the active-image mask; images past ``actual_image_count``
    carry zero weight. ``projected_reference`` is complex ``[R, P]`` or already
    packed as float ``[R, 2 P]`` (``[Re | Im]``,
    :func:`relax.projection.projection.project_relion_coarse_packed_rows`). ``X`` is one real GEMM over ``[Re, Im]``-packed operands
    (the complex product's imaginary half is never formed) and ``A`` is a second
    GEMM. Both run at ``wide`` precision with an explicit full-precision dot
    algorithm (float32 never falls to TF32).
    """

    active = jnp.arange(n_images, dtype=jnp.int32) < jnp.asarray(
        actual_image_count,
        dtype=jnp.int32,
    )
    weight = jnp.where(active[:, None], pixel_weight, 0).astype(wide)
    shifted_re = jnp.where(active[:, None, None], shifted_corrected.real, 0).astype(wide)
    shifted_im = jnp.where(active[:, None, None], shifted_corrected.imag, 0).astype(wide)
    if jnp.iscomplexobj(projected_reference):
        reference_re = projected_reference.real.astype(wide)
        reference_im = projected_reference.imag.astype(wide)
        reference_packed = jnp.concatenate([reference_re, reference_im], axis=-1)
    else:
        reference_packed = projected_reference.astype(wide)
        n_pixels = reference_packed.shape[-1] // 2
        reference_re, reference_im = reference_packed[..., :n_pixels], reference_packed[..., n_pixels:]

    # float32 names its full-precision algorithm so it never falls to TF32;
    # binary64 dots are always binary64, and XLA's small-dot emitters reject
    # the explicit F64 algorithm.
    algorithm = (
        jax.lax.Precision.HIGHEST
        if wide == jnp.float64
        else jax.lax.DotAlgorithmPreset.F32_F32_F32
    )
    weighted_packed = jnp.concatenate(
        [shifted_re * weight[:, None, :], shifted_im * weight[:, None, :]],
        axis=-1,
    ).reshape(n_images * n_trans, -1)
    cross = jax.lax.dot(
        weighted_packed,
        reference_packed.T,
        precision=algorithm,
        preferred_element_type=wide,
    ).reshape(n_images, n_trans, -1)
    model_energy = jax.lax.dot(
        weight,
        (reference_re * reference_re + reference_im * reference_im).T,
        precision=algorithm,
        preferred_element_type=wide,
    )
    image_energy = jnp.sum(
        (shifted_re * shifted_re + shifted_im * shifted_im) * weight[:, None, :],
        axis=-1,
    )
    return cross, model_energy, image_energy, active


@partial(
    jax.jit,
    static_argnames=("n_images", "n_trans", "image_shape", "volume_shape", "float64", "translation_major"),
)
def relion_coarse_gaussian_gemm_scores_jit(
    projected_reference,
    projected_reference_abs2,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    n_images: int,
    n_trans: int,
    image_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
    float64: bool = False,
    translation_major: bool = False,
):
    """Score exact RELION coarse operands with two real-packed GEMMs.

    RELION's direct square ``d0 + 0.5 sum_k w_k |p_k - y_k|^2`` expands into
    ``d0 + 0.5 A + 0.5 C - X`` over the terms of
    :func:`relion_coarse_gemm_terms`. ``RELAX_COARSE_GEMM_FLOAT64=1``
    promotes the stored operands to binary64 instead, for measuring the
    expansion's cancellation; float32 is the production arithmetic.
    Scores are ``[B, R, T]``, or ``[B, T, R]`` (the GEMM's own layout, the same
    values) with ``translation_major``.
    """

    del projected_reference_abs2, image_shape, volume_shape
    out_dtype = pixel_weight.dtype
    wide = jnp.float64 if float64 else out_dtype
    cross, model_energy, image_energy, active = relion_coarse_gemm_terms(
        projected_reference,
        shifted_corrected,
        pixel_weight,
        actual_image_count,
        n_images=n_images,
        n_trans=n_trans,
        wide=wide,
    )
    initial = jnp.where(active, initial_diff2, 0).astype(wide)
    half = jnp.asarray(0.5, dtype=wide)
    if translation_major:
        scores = (
            cross
            - half * model_energy[:, None, :]
            - half * image_energy[:, :, None]
            - initial[:, None, None]
        )
    else:
        scores = (
            cross.swapaxes(1, 2)
            - half * model_energy[:, :, None]
            - half * image_energy[:, None, :]
            - initial[:, None, None]
        )
    return jnp.where(active[:, None, None], scores, 0).astype(out_dtype)


@partial(jax.jit, static_argnames=("n_images", "n_trans"))
def relion_coarse_normalized_cc_gemm_scores_jit(
    projected_reference,
    shifted_corrected,
    pixel_weight,
    actual_image_count,
    *,
    n_images: int,
    n_trans: int,
):
    """Score exact RELION ``--firstiter_cc`` coarse operands with the coarse GEMMs.

    ``cuda_kernel_diff2_CC_coarse`` weights both its numerator and its norm by
    ``corr_img``; with ``pixel_weight = corr_img * half_weights`` they are the
    cross term ``X`` and model energy ``A`` of :func:`relion_coarse_gemm_terms`,
    and the score is ``X / sqrt(A)`` through RELION's 128 atomic additions
    (:func:`_relion_coarse_cc_atomic_score_from_components`). Returns
    ``[B, R, T]`` float32 scores; padded images score zero.
    """

    cross, model_energy, _image_energy, active = relion_coarse_gemm_terms(
        projected_reference,
        shifted_corrected,
        pixel_weight,
        actual_image_count,
        n_images=n_images,
        n_trans=n_trans,
        wide=jnp.float32,
    )
    scores = _relion_coarse_cc_atomic_score_from_components(
        cross.swapaxes(1, 2),
        model_energy[:, :, None],
    )
    return jnp.where(active[:, None, None], scores, 0)


@partial(jax.jit, static_argnums=())
def update_logsumexp(max_s, sum_exp, scores_block):
    """Streaming logsumexp update from one score block.

    Robust to all-(-inf) score blocks (K-class adaptive 2-pass with an
    empty significance mask): a finite ``safe_new_max`` is used inside the
    exp so we never form -inf - (-inf) = NaN; ``new_max`` is still returned
    as -inf so the streaming logsumexp is exactly -inf for empty inputs.
    """

    accumulator_dtype = sum_exp.dtype
    scores_flat = scores_block.reshape(scores_block.shape[0], -1)
    block_max = jnp.max(scores_flat, axis=1)
    new_max = jnp.maximum(max_s, block_max)
    safe_new_max = jnp.where(jnp.isfinite(new_max), new_max, jnp.zeros_like(new_max))
    exp_terms = jnp.sum(
        jnp.exp((scores_flat - safe_new_max[:, None]).astype(accumulator_dtype)),
        axis=1,
    )
    safe_max_s = jnp.where(jnp.isfinite(max_s), max_s, jnp.zeros_like(max_s))
    old_term = jnp.where(
        jnp.isfinite(max_s),
        sum_exp * jnp.exp((safe_max_s - safe_new_max).astype(accumulator_dtype)),
        jnp.zeros_like(sum_exp),
    )
    sum_exp = old_term + exp_terms
    return new_max, sum_exp
