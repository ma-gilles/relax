"""RELION's image preprocessing of one dense batch: the dataset's images, CTF and noise on the half spectrum.

These read the dataset's optics (odd-aberration demodulation, the exact RELION CTF rows) and its image
backend's per-image operands; the translation phase tables they apply are :mod:`relax.fourier.preprocessing`'s.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from relax.cuda import (
    preprocess as _relion_cuda_preprocessor_registration,  # noqa: F401  (registers the relion_cuda preprocessor, relax split seam S2)
)
from relax.fourier.half_spectrum import make_half_image_weights
from relax.fourier.preprocessing import (
    apply_half_translation_phases,
    half_translation_phase_table,
)


def image_preprocess_backend(experiment_dataset):
    image_source = getattr(experiment_dataset, "image_source", None)
    return getattr(image_source, "backend", image_source)


def relion_preprocess_backend(experiment_dataset):
    """Root image backend of a dataset, following subset ``parent`` links."""

    image_source = getattr(experiment_dataset, "image_source", None)
    while hasattr(image_source, "parent"):
        image_source = image_source.parent
    return getattr(image_source, "backend", image_source)


def uses_relion_cuda_image_preprocessing(experiment_dataset) -> bool:
    """Whether the dataset's root backend runs RELION's CUDA image preprocessing.

    The fresh K=1 refinement defaults (source-faithful powerClass
    normalization, exact BPref operands, exact coarse operands) score from
    that preprocessing and fail closed without it.
    """

    return getattr(relion_preprocess_backend(experiment_dataset), "relion_fourier_backend", None) == "relion_cuda"


def _cast_shift_inputs(
    processed_half,
    ctf_half,
    noise_variance_half=None,
    translation_phases_half=None,
    *,
    score_complex_dtype=None,
    score_real_dtype=None,
):
    if score_complex_dtype is not None:
        processed_half = processed_half.astype(score_complex_dtype)
        if translation_phases_half is not None:
            translation_phases_half = translation_phases_half.astype(score_complex_dtype)
    if score_real_dtype is not None:
        ctf_half = ctf_half.astype(score_real_dtype)
        if noise_variance_half is not None:
            noise_variance_half = noise_variance_half.astype(score_real_dtype)
    return processed_half, ctf_half, noise_variance_half, translation_phases_half


def _norm_inputs(processed_half, noise_variance_half=None, half_weights=None, *, norm_real_dtype=None):
    if norm_real_dtype is None:
        return processed_half, noise_variance_half, half_weights
    norm_complex_dtype = jnp.complex128 if norm_real_dtype == jnp.float64 else jnp.complex64
    processed_half = processed_half.astype(norm_complex_dtype)
    if noise_variance_half is not None:
        noise_variance_half = noise_variance_half.astype(norm_real_dtype)
    if half_weights is not None:
        half_weights = half_weights.astype(norm_real_dtype)
    return processed_half, noise_variance_half, half_weights


# (batch, height, width, device kind) -> whether the batched centered rFFT is bitwise the per-image one.
_BATCHED_FFT_IS_PER_IMAGE: dict[tuple, bool] = {}


def batched_fft_is_per_image(shape) -> bool:
    """Whether one batched cuFFT of ``shape`` ``[n, H, W]`` gives every image bitwise its per-image transform.

    RELION transforms each image with its own plan, so the exact paths ask the image backend for one plan per
    image (``relion_fft_per_image``): a ``lax.map`` of one transform per image, millions of small launches per
    pass (a tutorial bin-6 pass-2 half, ~2.2M of its 3.9M). cuFFT's batched plan is bitwise the per-image one
    for some sizes and not for others (on H100 and A100: boxes 64, 96 and 192 match at batches 1 to 1024, boxes
    128 and 256 differ at batches of 32 and more, ~2.6e-7 relative). The answer is probed once per shape and
    device, on a random float32 batch through the backend's two transforms, and the batched plan is used only
    on a bitwise match. Without a GPU the answer is no.
    """

    import jax

    if jax.default_backend() != "gpu":
        return False
    n, height, width = (int(v) for v in shape)
    key = (n, height, width, jax.devices()[0].device_kind)
    if key not in _BATCHED_FFT_IS_PER_IMAGE:
        from recovar.data_io.image_backends import _centered_rfft2_jax_batched, _centered_rfft2_jax_per_image

        probe = jnp.asarray(np.random.default_rng(0).standard_normal((n, height, width)), dtype=jnp.float32)
        batched = np.asarray(_centered_rfft2_jax_batched(probe))
        per_image = np.asarray(_centered_rfft2_jax_per_image(probe))
        _BATCHED_FFT_IS_PER_IMAGE[key] = batched.dtype == per_image.dtype and np.array_equal(
            batched.view(np.uint32), per_image.view(np.uint32)
        )
    return _BATCHED_FFT_IS_PER_IMAGE[key]


# (height, width, device kind) -> how many images one cuFFT plan may transform at the per-image arithmetic.
_PER_IMAGE_FFT_SUB_BATCH: dict[tuple, int] = {}
# The largest sub-batch probed. On H100 and A100 the batched plan is bitwise the per-image one up to 15 images at
# 256x256 and 31 at 128x128 (relax tomoglobal2 probes); 16 per plan is about 13x faster than one.
_MAX_PER_IMAGE_FFT_SUB_BATCH = 16


def per_image_fft_sub_batch(height: int, width: int) -> int:
    """How many images one cuFFT plan may transform with every image bitwise its per-image transform.

    Where one batched plan of the whole batch is not bitwise (:func:`batched_fft_is_per_image`), the backend still
    need not run one plan per image: cuFFT's batched plan is bitwise the per-image one for small batches, and the
    backend can map over sub-batches of ``s`` images and transform the ``n % s`` rest as one batch
    (``relion_fft_sub_batch``). The probe runs once per image shape and device, on a random float32 batch: every
    batch size 1 to :data:`_MAX_PER_IMAGE_FFT_SUB_BATCH` through the batched plan against the per-image one, then
    the sub-batch path itself for the largest power of two ``s`` whose sizes all matched, on ``3s - 1`` images
    (full sub-batches and the largest rest). The answer is that ``s``, or 1 (one plan per image) without a GPU or
    a match.
    """

    import jax

    if jax.default_backend() != "gpu":
        return 1
    key = (int(height), int(width), jax.devices()[0].device_kind)
    if key not in _PER_IMAGE_FFT_SUB_BATCH:
        from recovar.data_io.image_backends import (
            _centered_rfft2_jax_batched,
            _centered_rfft2_jax_per_image,
            _centered_rfft2_jax_sub_batches,
        )

        def bits(values):
            return np.asarray(values).view(np.uint32)

        rng = np.random.default_rng(0)
        probe = jnp.asarray(rng.standard_normal((_MAX_PER_IMAGE_FFT_SUB_BATCH, *key[:2])), dtype=jnp.float32)
        per_image = bits(_centered_rfft2_jax_per_image(probe))
        matched = 0
        for size in range(1, _MAX_PER_IMAGE_FFT_SUB_BATCH + 1):
            if not np.array_equal(bits(_centered_rfft2_jax_batched(probe[:size])), per_image[:size]):
                break
            matched = size
        sub_batch = 1 << (matched.bit_length() - 1) if matched >= 2 else 1
        if sub_batch > 1:
            images = jnp.asarray(rng.standard_normal((3 * sub_batch - 1, *key[:2])), dtype=jnp.float32)
            if not np.array_equal(
                bits(_centered_rfft2_jax_sub_batches(images, sub_batch)), bits(_centered_rfft2_jax_per_image(images))
            ):
                sub_batch = 1
        _PER_IMAGE_FFT_SUB_BATCH[key] = sub_batch
    return _PER_IMAGE_FFT_SUB_BATCH[key]


def process_half_image(
    experiment_dataset,
    batch,
    apply_image_mask: bool,
    *,
    relion_preprocess_kwargs=None,
    image_indices=None,
):
    """The batch's images on RECOVAR's centered half spectrum, as every EM path scores them.

    Images of optics groups with beam tilt or odd Zernike terms are demodulated
    here, as RELION demodulates each particle image before scoring and
    backprojection (:func:`relax.relion.optics_aberrations.demodulate_odd_aberrations`);
    that needs the batch's ``image_indices``.
    """

    from relax.relion.optics_aberrations import demodulate_odd_aberrations

    process_half_fn = getattr(experiment_dataset, "process_images_half", None)
    if process_half_fn is None:
        raise ValueError("Dense EM requires experiment_dataset.process_images_half")
    kwargs = {} if relion_preprocess_kwargs is None else dict(relion_preprocess_kwargs)
    if kwargs.get("relion_fft_per_image") and np.ndim(batch) == 3:
        if batched_fft_is_per_image(np.shape(batch)):
            kwargs["relion_fft_per_image"] = False  # the batched plan is bitwise the per-image one at this shape
        else:
            sub_batch = per_image_fft_sub_batch(*np.shape(batch)[1:])
            if sub_batch > 1:
                kwargs["relion_fft_sub_batch"] = sub_batch  # plans of up to this many images are bitwise per image
    processed = process_half_fn(batch, apply_image_mask=apply_image_mask, **kwargs)
    return demodulate_odd_aberrations(experiment_dataset, processed, image_indices)


def _dense_batch_ctf_half(experiment_dataset, ctf_params, config, ctf_real_dtype, image_indices):
    """The batch's CTF on RECOVAR's centered half spectrum.

    Datasets whose CTF needs the optics table (CTF-premultiplied images, even Zernike
    terms, magnification; :func:`relax.relion.optics_aberrations.dataset_needs_exact_ctf`)
    take relax's exact RELION CTF rows of the batch's ``image_indices``
    (:mod:`relax.relion.ctf`), as RELION scores them; the generic evaluator
    refuses those datasets. Other datasets keep the generic CTF of ``ctf_params``.
    """

    from relax.relion.optics_aberrations import dataset_needs_exact_ctf

    if image_indices is not None and dataset_needs_exact_ctf(experiment_dataset):
        from relax.relion.ctf import relion_exact_ctf_half_from_source_star

        rows = relion_exact_ctf_half_from_source_star(experiment_dataset, image_indices, config.image_shape)
        return rows.astype(jnp.float32 if ctf_real_dtype is None else ctf_real_dtype)
    if ctf_real_dtype is not None:
        ctf_params = jnp.asarray(ctf_params, dtype=ctf_real_dtype)
    return config.compute_ctf_half(ctf_params)


def _dense_batch_half_inputs(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    apply_image_mask: bool,
    *,
    ctf_real_dtype=None,
    relion_preprocess_kwargs=None,
    image_indices=None,
):
    processed_half = process_half_image(
        experiment_dataset,
        batch,
        apply_image_mask,
        relion_preprocess_kwargs=relion_preprocess_kwargs,
        image_indices=image_indices,
    )
    ctf_half = _dense_batch_ctf_half(experiment_dataset, ctf_params, config, ctf_real_dtype, image_indices)
    noise_variance_half = jnp.asarray(noise_variance)
    translation_phases_half = half_translation_phase_table(translations, config.image_shape)
    return processed_half, ctf_half, noise_variance_half, translation_phases_half


def preprocess_batch(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    score_with_masked_images=False,
    *,
    score_complex_dtype=None,
    score_real_dtype=None,
    norm_real_dtype=None,
    relion_preprocess_kwargs=None,
    return_unshifted_score_weighted=False,
    image_indices=None,
):
    """Preprocess one dense image batch for E-step scoring."""

    processed_half, ctf_half, noise_variance_half, translation_phases_half = _dense_batch_half_inputs(
        experiment_dataset,
        batch,
        ctf_params,
        noise_variance,
        translations,
        config,
        score_with_masked_images,
        ctf_real_dtype=score_real_dtype,
        relion_preprocess_kwargs=relion_preprocess_kwargs,
        image_indices=image_indices,
    )
    half_weights = make_half_image_weights(config.image_shape)
    shifted_half, norm_integrand, ctf2_over_nv_half, score_weighted_half = _preprocess_batch_elementwise(
        processed_half,
        ctf_half,
        noise_variance_half,
        translation_phases_half,
        half_weights,
        score_complex_dtype=score_complex_dtype,
        score_real_dtype=score_real_dtype,
        norm_real_dtype=norm_real_dtype,
    )
    # The shell reduction follows the elementwise chain (fusing the two would change the accumulation XLA emits
    # and move the last bit of ``batch_norm``).
    batch_norm = jnp.sum(norm_integrand, axis=-1, keepdims=True).real
    if return_unshifted_score_weighted:
        return shifted_half, batch_norm, ctf2_over_nv_half, score_weighted_half
    return shifted_half, batch_norm, ctf2_over_nv_half


def _preprocess_batch_elementwise(
    processed_half,
    ctf_half,
    noise_variance_half,
    translation_phases_half,
    half_weights,
    *,
    score_complex_dtype,
    score_real_dtype,
    norm_real_dtype,
):
    """:func:`preprocess_batch`'s elementwise arithmetic, without the reduction.

    ``norm_integrand``
    is the summand of ``batch_norm``, returned unreduced so the caller keeps
    the reduction in its own program.
    """

    shift_processed_half, shift_ctf_half, shift_noise_half, shift_phases_half = _cast_shift_inputs(
        processed_half,
        ctf_half,
        noise_variance_half,
        translation_phases_half,
        score_complex_dtype=score_complex_dtype,
        score_real_dtype=score_real_dtype,
    )
    score_weighted_half = shift_processed_half * shift_ctf_half / shift_noise_half
    shifted_half = apply_half_translation_phases(score_weighted_half, shift_phases_half)
    norm_processed_half, norm_noise_half, norm_half_weights = _norm_inputs(
        processed_half,
        noise_variance_half,
        half_weights,
        norm_real_dtype=norm_real_dtype,
    )
    norm_integrand = (
        jnp.abs(norm_processed_half) ** 2 / norm_noise_half
    ) * norm_half_weights[None, :]
    weight_ctf_half = shift_ctf_half if score_real_dtype is not None else ctf_half
    weight_noise_half = shift_noise_half if score_real_dtype is not None else noise_variance_half
    ctf2_over_nv_half = weight_ctf_half**2 / weight_noise_half
    return shifted_half, norm_integrand, ctf2_over_nv_half, score_weighted_half


def prepare_reconstruction_batch(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    *,
    score_complex_dtype=None,
    score_real_dtype=None,
    relion_preprocess_kwargs=None,
    image_indices=None,
):
    """Preprocess one dense image batch for the unmasked M-step path."""

    processed_half, ctf_half, noise_variance_half, translation_phases_half = _dense_batch_half_inputs(
        experiment_dataset,
        batch,
        ctf_params,
        noise_variance,
        translations,
        config,
        False,
        ctf_real_dtype=score_real_dtype,
        relion_preprocess_kwargs=relion_preprocess_kwargs,
        image_indices=image_indices,
    )
    shift_processed_half, shift_ctf_half, shift_noise_half, shift_phases_half = _cast_shift_inputs(
        processed_half,
        ctf_half,
        noise_variance_half,
        translation_phases_half,
        score_complex_dtype=score_complex_dtype,
        score_real_dtype=score_real_dtype,
    )
    return apply_half_translation_phases(
        shift_processed_half * shift_ctf_half / shift_noise_half,
        shift_phases_half,
    )


def prepare_batch_preprocess_operands(
    experiment_dataset,
    batch,
    image_indices,
    *,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    dtype: np.dtype = np.float32,
):
    """Select typed per-image operands for host or strict CUDA preprocessing.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision);
    callers running double-precision scoring should pass ``np.float64`` --
    these are RELION's per-image group-scale/normalization correction
    factors (RFLOAT, never narrowed), multiplied directly into the
    CTF^2/noise weighting used by both pass-1 and pass-2 scoring. The
    ``relion_cuda``-backend branch below is unaffected: it's a real CUDA
    FFI kernel input, hardware-locked to float32 independent of this
    ``dtype``.
    """

    from relax.fourier.image_shifts import integer_pre_shifts_or_none

    image_indices_np = np.asarray(image_indices)
    batch_size = int(batch.shape[0])
    integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, image_indices_np, batch=batch)
    backend = image_preprocess_backend(experiment_dataset)
    relion_cuda_preprocess = getattr(backend, "relion_fourier_backend", None) == "relion_cuda"
    if relion_cuda_preprocess and image_pre_shifts is not None and integer_pre_shifts is None:
        raise RuntimeError("relion_cuda preprocessing requires integral RELION image pre-shifts")
    if relion_cuda_preprocess and integer_pre_shifts is None:
        integer_pre_shifts = np.zeros((batch_size, 2), dtype=np.int32)

    batch_scale_np = (
        np.asarray(scale_corrections)[image_indices_np].astype(dtype, copy=False)
        if scale_corrections is not None
        else np.ones(batch_size, dtype=dtype)
    )
    batch_corr_np = (
        np.asarray(image_corrections)[image_indices_np].astype(dtype, copy=False)
        if image_corrections is not None
        else None
    )
    relion_preprocess_kwargs = None
    if relion_cuda_preprocess:
        from relax.cuda.preprocess import note_relion_preprocess_batch

        # An invalid image of this batch is reported by its dataset index (relax#16).
        note_relion_preprocess_batch(image_indices_np)
        normalization_factors = (
            batch_corr_np / batch_scale_np
            if batch_corr_np is not None
            else np.ones(batch_size, dtype=np.float32)
        )
        relion_preprocess_kwargs = {
            "relion_normalization_factors": jnp.asarray(normalization_factors, dtype=jnp.float32),
            "relion_integer_shifts": jnp.asarray(integer_pre_shifts, dtype=jnp.int32),
        }
    return (
        relion_cuda_preprocess,
        integer_pre_shifts,
        batch_corr_np,
        batch_scale_np,
        relion_preprocess_kwargs,
    )
