"""Float32 dense GEMM tiles for the opt-in resident K1 experiment.

The score convention is the absolute RELION coarse Gaussian convention in
``relax.scoring.scoring._relion_coarse_gaussian_gemm_scores_jit``.  Inputs are
already assembled by the canonical RELION operand path.  This module only
changes the placement and tiling of the translation phase.
See ``docs/math/dense_gemm_experiment.md`` for the score and memory contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from relax.scoring.scoring import (
    _relion_coarse_cc_atomic_score_from_components,
    _relion_coarse_gemm_terms,
)


def score_model_power(projection, pixel_weight):
    """Translation-invariant RELION model-square term for one rotation tile."""
    projection_power = projection.real * projection.real + projection.imag * projection.imag
    return jax.lax.dot(
        pixel_weight,
        projection_power.T,
        precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
        preferred_element_type=jnp.float32,
    )


def score_tile(
    projection,
    score_image,
    pixel_weight,
    initial_diff2,
    phase,
    rotation_prior,
    translation_prior,
    valid_images,
    valid_rotations,
    valid_translations,
    *,
    translation_side: str,
    model_power=None,
):
    """Return absolute scores ``[B,Q,U]`` using float32 GEMMs.

    ``score_image`` is the unshifted, CTF-corrected complex image.  Its
    ``pixel_weight`` includes the established noise and Fourier metric.  The
    phase is the RELION image-translation phase at the score pixel indices.
    """
    b, p = score_image.shape
    q = projection.shape[0]
    u = phase.shape[0]
    if translation_side == "image":
        shifted = score_image[:, None, :] * phase[None, :, :]
        cross_buq, calculated_power, image_energy, _ = _relion_coarse_gemm_terms(
            projection,
            shifted,
            pixel_weight,
            jnp.sum(valid_images),
            n_images=b,
            n_trans=u,
            wide=jnp.float32,
        )
        cross = cross_buq.swapaxes(1, 2)
        image_power = jnp.asarray(0.5, jnp.float32) * image_energy
        power = calculated_power if model_power is None else model_power
    elif translation_side == "projection":
        weighted = score_image * pixel_weight
        shifted_projection = projection[:, None, :] * jnp.conj(phase)[None, :, :]
        weighted_packed = jnp.concatenate([weighted.real, weighted.imag], axis=1)
        shifted_packed = jnp.concatenate([shifted_projection.real, shifted_projection.imag], axis=2).reshape(
            q * u, 2 * p
        )
        cross = jax.lax.dot(
            weighted_packed,
            shifted_packed.T,
            precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
            preferred_element_type=jnp.float32,
        ).reshape(b, q, u)
        power = score_model_power(projection, pixel_weight) if model_power is None else model_power
        # Keep translation-dependent phase-norm roundoff in the absolute
        # image term without allocating a B*U*P translated image.  This
        # reduction order differs from the native direct-square kernel and
        # must be measured on matched operands.
        image_abs2 = score_image.real * score_image.real + score_image.imag * score_image.imag
        phase_abs2 = phase.real * phase.real + phase.imag * phase.imag
        image_power = jnp.asarray(0.5, jnp.float32) * jax.lax.dot(
            image_abs2 * pixel_weight,
            phase_abs2.T,
            precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
            preferred_element_type=jnp.float32,
        )
    else:
        raise ValueError(f"unknown translation_side {translation_side!r}")

    scores = (
        cross
        - jnp.asarray(0.5, jnp.float32) * power[:, :, None]
        - image_power[:, None, :]
        - initial_diff2[:, None, None]
        + rotation_prior[:, :, None]
        + translation_prior[:, None, :]
    )
    valid = valid_images[:, None, None] & valid_rotations[None, :, None] & valid_translations[None, None, :]
    return jnp.where(valid, scores, -jnp.inf)


def cc_score_tile(
    projection,
    score_image,
    pixel_weight,
    phase,
    valid_images,
    valid_rotations,
    valid_translations,
    *,
    translation_side: str,
    model_power=None,
):
    """RELION first-iteration normalized CC on one complete-grid tile.

    The shared coarse CC reduction owns the score algebra.  Priors are
    deliberately absent: RELION's first-iteration CC winner ignores them.
    """
    b, p = score_image.shape
    q, u = projection.shape[0], phase.shape[0]
    if translation_side == "image":
        shifted = score_image[:, None, :] * phase[None, :, :]
        cross_buq, calculated_power, _, _ = _relion_coarse_gemm_terms(
            projection,
            shifted,
            pixel_weight,
            jnp.sum(valid_images),
            n_images=b,
            n_trans=u,
            wide=jnp.float32,
        )
        cross = cross_buq.swapaxes(1, 2)
        power = calculated_power if model_power is None else model_power
    elif translation_side == "projection":
        weighted = score_image * pixel_weight
        shifted_projection = projection[:, None, :] * jnp.conj(phase)[None, :, :]
        weighted_packed = jnp.concatenate([weighted.real, weighted.imag], axis=1)
        shifted_packed = jnp.concatenate([shifted_projection.real, shifted_projection.imag], axis=2).reshape(
            q * u, 2 * p
        )
        cross = jax.lax.dot(
            weighted_packed,
            shifted_packed.T,
            precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
            preferred_element_type=jnp.float32,
        ).reshape(b, q, u)
        power = score_model_power(projection, pixel_weight) if model_power is None else model_power
    else:
        raise ValueError(f"unknown translation_side {translation_side!r}")
    scores = _relion_coarse_cc_atomic_score_from_components(
        cross, power[:, :, None]
    )
    valid = valid_images[:, None, None] & valid_rotations[None, :, None] & valid_translations[None, None, :]
    return jnp.where(valid, scores, -jnp.inf)


def class_score_tile(
    projections,
    score_image,
    pixel_weight,
    initial_diff2,
    phase,
    class_rotation_prior,
    translation_prior,
    valid_images,
    valid_rotations,
    valid_translations,
    *,
    translation_side: str,
    score_mode: str,
    model_power,
):
    """Score K classes in one GEMM, retaining separate class/pose scores.

    The native projector still supplies one [Q, P] block per class. Flattening
    those blocks to [K*Q, P] shares the translated image operand across K.
    Priors are added only after the GEMM; normalized CC ignores them.
    """
    k, q, p = projections.shape
    flat_projections = projections.reshape(k * q, p)
    flat_valid_rotations = jnp.tile(valid_rotations, k)
    if score_mode == "normalized_cc":
        flat_scores = cc_score_tile(
            flat_projections, score_image, pixel_weight, phase,
            valid_images, flat_valid_rotations, valid_translations,
            translation_side=translation_side, model_power=model_power,
        )
    elif score_mode == "gaussian":
        b = score_image.shape[0]
        flat_prior = jnp.broadcast_to(class_rotation_prior, (k, b, q))
        flat_prior = flat_prior.transpose(1, 0, 2).reshape(b, k * q)
        flat_scores = score_tile(
            flat_projections, score_image, pixel_weight, initial_diff2,
            phase, flat_prior, translation_prior, valid_images,
            flat_valid_rotations, valid_translations,
            translation_side=translation_side, model_power=model_power,
        )
    else:
        raise ValueError(f"unsupported dense GEMM score mode {score_mode!r}")
    b, _, u = flat_scores.shape
    return flat_scores.reshape(b, k, q, u).transpose(1, 0, 2, 3)


def _real_weighted_complex_dot(weights, values):
    """Contract real weights with complex pixels in one float32 GEMM.

    See ``docs/math/dense_gemm_experiment.md#translation-layouts-and-accumulation``.
    """
    p = values.shape[1]
    packed = jnp.concatenate((values.real, values.imag), axis=1)
    result = jax.lax.dot(
        weights, packed,
        precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
        preferred_element_type=jnp.float32,
    )
    return jax.lax.complex(result[:, :p], result[:, p:])


def weighted_numerator_slices(q, rec_image, rec_phase, *, translation_side: str):
    """Reduce images/translations to complex slices; denominator is tile-invariant."""
    b, r, u = q.shape
    p = rec_image.shape[1]
    if translation_side == "image":
        shifted = rec_image[:, None, :] * rec_phase[None, :, :]
        numerator = _real_weighted_complex_dot(
            q.swapaxes(0, 1).reshape(r, b * u), shifted.reshape(b * u, p)
        )
    elif translation_side == "projection":
        by_translation = _real_weighted_complex_dot(
            q.reshape(b, r * u).T, rec_image
        ).reshape(r, u, p)
        numerator = jnp.sum(by_translation * rec_phase[None, :, :], axis=1)
    else:
        raise ValueError(f"unknown translation_side {translation_side!r}")
    return numerator.astype(jnp.complex64)


def weighted_denominator_slices(rotation_mass, rec_weight):
    """One denominator GEMM after all translation tiles of a rotation tile."""
    return jnp.matmul(
        rotation_mass.T,
        rec_weight,
        precision=jax.lax.Precision.HIGHEST,
    ).astype(jnp.float32)


def weighted_slices(q, rec_image, rec_weight, rec_phase, *, translation_side: str):
    """Reduce all images/translations to one native-compatible slice per rotation."""
    numerator = weighted_numerator_slices(q, rec_image, rec_phase, translation_side=translation_side)
    denominator = weighted_denominator_slices(jnp.sum(q, axis=2), rec_weight)
    return numerator, denominator


def tile_logsumexp(scores):
    """Stable per-image log-sum-exp over one fixed candidate tile."""
    return jax.nn.logsumexp(scores.reshape(scores.shape[0], -1), axis=1)


def empty_normalizer_table(count):
    """Float32 [particle, (maximum score, log shifted sum)] with empty rows."""
    return jnp.full((count, 2), -jnp.inf, dtype=jnp.float32)


def tile_normalizer(scores):
    """Keep the absolute score gauge split from its small log-sum correction."""
    flat = scores.reshape(scores.shape[0], -1)
    maximum = jnp.max(flat, axis=1)
    shifted = flat - jnp.where(jnp.isfinite(maximum), maximum, 0)[:, None]
    log_sum = jax.nn.logsumexp(shifted, axis=1)
    return jnp.stack((maximum, log_sum), axis=1)


def merge_normalizers(left, right):
    """Online max/log-sum-exp merge; empty padded rows stay (-inf,-inf)."""
    maximum = jnp.maximum(left[:, 0], right[:, 0])
    safe_maximum = jnp.where(jnp.isfinite(maximum), maximum, 0)
    left_maximum = jnp.where(jnp.isfinite(left[:, 0]), left[:, 0], safe_maximum)
    right_maximum = jnp.where(jnp.isfinite(right[:, 0]), right[:, 0], safe_maximum)
    log_sum = jnp.logaddexp(
        left[:, 1] + (left_maximum - safe_maximum),
        right[:, 1] + (right_maximum - safe_maximum),
    )
    return jnp.stack((maximum, log_sum), axis=1)


def normalizer_logz(pair):
    """Collapsed absolute logZ is for reporting, never posterior weights."""
    return pair[..., 0] + pair[..., 1]
