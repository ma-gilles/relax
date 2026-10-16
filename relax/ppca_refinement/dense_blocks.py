"""The dataset blocks of dense PPCA refinement: one input contract for every dense consumer.

``prepare_dense_ppca_dataset_inputs`` coerces the augmented half volumes and the pose masks once;
``iter_dense_ppca_dataset_blocks`` (and ``iter_dense_ppca_dataset_block_groups``) yields each image block's
preprocessed images (``prepare_dense_ppca_image_batch``), augmented projections and per-image pose log prior.
The statistics, the fused EM iteration and the embeddings (``dense_dataset``), the streamed engine and the
scripts read them.
"""

from __future__ import annotations

from functools import partial
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise as noise_utils

from relax.fourier.fourier_window import make_fourier_window_spec
from relax.fourier.half_spectrum import make_scoring_half_image_weights
from relax.ppca_refinement.engine import DensePPCAFusedBlock
from relax.ppca_refinement.initialization import real_volume_to_centered_fourier_half
from relax.relion.image_preprocessing import prepare_reconstruction_batch, preprocess_batch


class DensePPCADatasetBlockInputs(NamedTuple):
    """Resolved static inputs for a dense PPCA dataset iteration."""

    augmented_half_volumes: jax.Array
    q: int
    image_shape: tuple[int, int]
    volume_shape: tuple[int, int, int]
    score_mask: jax.Array
    recon_mask: jax.Array
    score_indices: jax.Array | None
    recon_indices: jax.Array | None
    use_window: bool
    backprojection_max_r: float | None
    projection_max_r: float | None


def _coerce_one_volume_to_half(volume, volume_shape, *, volume_domain: str) -> jax.Array:
    """Flatten one volume to centered half-Fourier (recovar canonical shape).

    ``volume_domain`` must be ``"real"`` (full real-space grid), ``"fourier_half"``
    (already centered + Hermitian-packed flat), or ``"auto"`` (pick based on size).
    """

    volume_shape = tuple(int(x) for x in volume_shape)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    arr = np.asarray(volume)
    domain = str(volume_domain)
    if domain == "auto":
        domain = "fourier_half" if arr.size == half_size else "real"

    if domain == "fourier_half":
        if arr.size != half_size:
            raise ValueError(f"fourier_half volume has {arr.size} elements, expected {half_size}")
        return jnp.asarray(arr.reshape(-1), dtype=jnp.complex64)
    if domain == "real":
        if tuple(arr.shape) != volume_shape:
            raise ValueError(f"real volume shape {arr.shape} != {volume_shape}")
        return jnp.asarray(real_volume_to_centered_fourier_half(arr), dtype=jnp.complex64)
    raise ValueError(f"volume_domain must be 'auto', 'fourier_half', or 'real', got {domain!r}")


def coerce_augmented_half_volumes(
    mu,
    W=None,
    *,
    volume_shape,
    q: int | None = None,
    volume_domain: str = "auto",
) -> tuple[jax.Array, int]:
    """Stack ``[mu, W_1, ..., W_q]`` as flattened half-Fourier volumes.

    ``W`` is either ``(q, *volume_shape)`` real-space or ``(half_size, q)``
    half-Fourier flat; pass ``None`` when ``q == 0``.
    """

    volume_shape = tuple(int(x) for x in volume_shape)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    mu_half = _coerce_one_volume_to_half(mu, volume_shape, volume_domain=volume_domain)
    if W is None:
        q_resolved = 0 if q is None else int(q)
        return mu_half[None, :], q_resolved

    W_arr = np.asarray(W)
    if W_arr.ndim >= 2 and tuple(W_arr.shape[1:]) == volume_shape:
        # (q, *volume_shape) real-space
        q_resolved = int(W_arr.shape[0]) if q is None else int(q)
        loadings = [W_arr[k] for k in range(q_resolved)]
    elif W_arr.ndim == 2 and W_arr.shape[0] == half_size:
        # (half_size, q) half-Fourier
        q_resolved = int(W_arr.shape[1]) if q is None else int(q)
        loadings = [W_arr[:, k] for k in range(q_resolved)]
    else:
        raise ValueError(f"W must be shaped (q, *volume_shape) or (half_size, q); got {W_arr.shape}")

    if q_resolved == 0:
        return mu_half[None, :], 0
    loading_halves = [
        _coerce_one_volume_to_half(loading, volume_shape, volume_domain=volume_domain) for loading in loadings
    ]
    return jnp.concatenate([mu_half[None, :], jnp.stack(loading_halves, axis=0)], axis=0), q_resolved


def _mask_from_indices(n_half: int, indices, *, dtype=jnp.float32) -> jax.Array:
    if indices is None:
        return jnp.ones((int(n_half),), dtype=dtype)
    mask = jnp.zeros((int(n_half),), dtype=dtype)
    return mask.at[jnp.asarray(indices, dtype=jnp.int32)].set(jnp.asarray(1.0, dtype=dtype))


def prepare_dense_ppca_dataset_inputs(
    experiment_dataset,
    mu,
    W=None,
    *,
    volume_shape=None,
    q: int | None = None,
    volume_domain: str = "auto",
    current_size: int | None = None,
    relion_unit_half_weights: bool = False,
    square_window: bool = False,
    full_real_observation: bool = False,
) -> DensePPCADatasetBlockInputs:
    """Resolve augmented volumes and Fourier-window masks for dense PPCA."""

    image_shape = tuple(int(x) for x in experiment_dataset.image_shape)
    volume_shape = tuple(int(x) for x in (experiment_dataset.volume_shape if volume_shape is None else volume_shape))
    H, W_img = image_shape
    n_half = int(H * (W_img // 2 + 1))
    augmented_half, q = coerce_augmented_half_volumes(
        mu,
        W,
        volume_shape=volume_shape,
        q=q,
        volume_domain=volume_domain,
    )
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
    )
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=relion_unit_half_weights,
    )
    score_indices = window_spec.recon_indices if full_real_observation else window_spec.score_indices
    score_mask = _mask_from_indices(n_half, score_indices) * half_weights
    recon_mask = _mask_from_indices(n_half, window_spec.recon_indices)
    return DensePPCADatasetBlockInputs(
        augmented_half_volumes=augmented_half,
        q=q,
        image_shape=image_shape,
        volume_shape=volume_shape,
        score_mask=score_mask.astype(jnp.float32),
        recon_mask=recon_mask.astype(jnp.float32),
        score_indices=score_indices,
        recon_indices=window_spec.recon_indices,
        use_window=bool(window_spec.use_window),
        backprojection_max_r=window_spec.max_r,
        projection_max_r=window_spec.max_r,
    )


@partial(
    jax.jit,
    static_argnames=(
        "image_shape",
        "volume_shape",
        "disc_type",
        "max_r",
        "relion_texture_interp",
    ),
)
def _project_augmented_half_volumes(
    augmented_half_volumes,
    rotations_block,
    image_shape,
    volume_shape,
    disc_type,
    *,
    max_r,
    relion_texture_interp: bool = False,
) -> jax.Array:
    """Slice the augmented [μ, W] half-volumes at every rotation in the block.

    Returns ``proj`` shaped ``(R, P, n_freq)`` where ``R = len(rotations_block)``
    and ``P = q + 1``. Used by both the dense iterator and the local iterator.
    """
    kwargs = {}
    if max_r is not None:
        kwargs["max_r"] = max_r
    projections = core.batch_slice_volume(
        jnp.asarray(augmented_half_volumes),
        jnp.asarray(rotations_block),
        image_shape,
        volume_shape,
        disc_type,
        half_volume=True,
        half_image=True,
        relion_texture_interp=relion_texture_interp,
        **kwargs,
    )
    return jnp.swapaxes(projections, 0, 1)


def _per_image_pose_prior_block(
    *,
    batch_start: int,
    batch_count: int,
    r0: int,
    r1: int,
    n_trans: int,
    rotation_log_prior,
    translation_log_prior,
    rotation_translation_mask,
    class_log_prior: float,
) -> jax.Array | None:
    R = int(r1 - r0)
    prior = None
    if rotation_log_prior is not None:
        rot = np.asarray(rotation_log_prior)
        if rot.ndim == 1:
            rot_block = np.broadcast_to(rot[r0:r1][None, :, None], (batch_count, R, n_trans))
        elif rot.ndim == 2:
            rot_block = np.broadcast_to(
                rot[batch_start : batch_start + batch_count, r0:r1, None],
                (batch_count, R, n_trans),
            )
        else:
            raise ValueError("rotation_log_prior must be 1D or 2D")
        prior = rot_block if prior is None else prior + rot_block
    if translation_log_prior is not None:
        trans = np.asarray(translation_log_prior)
        if trans.ndim == 1:
            trans_block = np.broadcast_to(trans[None, None, :], (batch_count, R, n_trans))
        elif trans.ndim == 2:
            trans_block = np.broadcast_to(
                trans[batch_start : batch_start + batch_count, None, :],
                (batch_count, R, n_trans),
            )
        else:
            raise ValueError("translation_log_prior must be 1D or 2D")
        prior = trans_block if prior is None else prior + trans_block
    if rotation_translation_mask is not None:
        mask = np.asarray(rotation_translation_mask, dtype=bool)
        if mask.ndim == 2:
            mask_block = np.broadcast_to(mask[r0:r1][None, :, :], (batch_count, R, n_trans))
        elif mask.ndim == 3:
            mask_block = mask[batch_start : batch_start + batch_count, r0:r1, :]
        else:
            raise ValueError("rotation_translation_mask must be 2D or 3D")
        mask_prior = np.where(mask_block, 0.0, -np.inf)
        prior = mask_prior if prior is None else prior + mask_prior
    if class_log_prior != 0.0:
        class_prior = np.full((batch_count, R, n_trans), float(class_log_prior), dtype=np.float32)
        prior = class_prior if prior is None else prior + class_prior
    return None if prior is None else jnp.asarray(prior, dtype=jnp.float32)


def _pose_mask_block_has_support(
    rotation_translation_mask, *, batch_start: int, batch_count: int, r0: int, r1: int
) -> bool:
    if rotation_translation_mask is None:
        return True
    mask = np.asarray(rotation_translation_mask, dtype=bool)
    if mask.ndim == 2:
        return bool(np.any(mask[r0:r1, :]))
    if mask.ndim == 3:
        return bool(np.any(mask[batch_start : batch_start + batch_count, r0:r1, :]))
    raise ValueError("rotation_translation_mask must be 2D or 3D")


class DensePPCAImageBatch(NamedTuple):
    """Weighted, shifted image-batch operands shared by every rotation block."""

    Y1_score: jax.Array
    ctf2_score: jax.Array
    Y1_recon: jax.Array
    ctf2_recon: jax.Array
    y_norm: jax.Array
    observation_power: jax.Array | None


def prepare_dense_ppca_image_batch(
    experiment_dataset,
    resolved: DensePPCADatasetBlockInputs,
    config: ForwardModelConfig,
    noise_variance_half,
    translations: np.ndarray,
    batch_data,
    ctf_params,
    indices,
    *,
    image_scale_corrections: np.ndarray | None = None,
    score_with_masked_images: bool = False,
    relion_unit_half_weights: bool = False,
    collect_observation: bool = False,
) -> DensePPCAImageBatch:
    """Preprocess one image batch into windowed PPCA score and M-step operands."""

    n_trans = int(translations.shape[0])
    batch_count = int(len(indices))
    shifted_score_half, batch_norm, ctf2_over_nv_half = preprocess_batch(
        experiment_dataset,
        batch_data,
        ctf_params,
        noise_variance_half,
        translations,
        config,
        score_with_masked_images=score_with_masked_images,
    )
    if score_with_masked_images:
        shifted_recon_half = prepare_reconstruction_batch(
            experiment_dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            translations,
            config,
        )
    else:
        shifted_recon_half = shifted_score_half

    observation_power = None
    if collect_observation:
        if score_with_masked_images or relion_unit_half_weights or image_scale_corrections is not None:
            raise ValueError("Residual statistics require unmasked, unit-contrast full-Hermitian observations")
        processed = experiment_dataset.process_images_half(batch_data, apply_image_mask=False)
        observation_power = jnp.sum(jnp.abs(processed.reshape(batch_count, -1)) ** 2, axis=0)
    F = int(shifted_score_half.shape[-1])
    if image_scale_corrections is None:
        batch_scale = jnp.ones((batch_count,), dtype=shifted_score_half.real.dtype)
    else:
        scale_arr = np.asarray(image_scale_corrections, dtype=np.float32)
        batch_scale = jnp.asarray(
            scale_arr[np.asarray(indices, dtype=np.int64)], dtype=shifted_score_half.real.dtype
        )
    batch_scale_sq = batch_scale**2
    Y1_score_full = (
        shifted_score_half.reshape(batch_count, n_trans, F)
        * batch_scale[:, None, None]
        * resolved.score_mask[None, None, :]
    )
    ctf2_score_full = ctf2_over_nv_half * batch_scale_sq[:, None] * resolved.score_mask[None, :]
    Y1_recon_full = (
        shifted_recon_half.reshape(batch_count, n_trans, F)
        * batch_scale[:, None, None]
        * resolved.recon_mask[None, None, :]
    )
    ctf2_recon_full = ctf2_over_nv_half * batch_scale_sq[:, None] * resolved.recon_mask[None, :]
    if resolved.score_indices is None:
        Y1_score = Y1_score_full
        ctf2_score = ctf2_score_full
    else:
        Y1_score = Y1_score_full[:, :, resolved.score_indices]
        ctf2_score = ctf2_score_full[:, resolved.score_indices]
    if resolved.recon_indices is None:
        Y1_recon = Y1_recon_full
        ctf2_recon = ctf2_recon_full
    else:
        Y1_recon = Y1_recon_full[:, :, resolved.recon_indices]
        ctf2_recon = ctf2_recon_full[:, resolved.recon_indices]
    y_norm = jnp.asarray(batch_norm).reshape(batch_count)
    return DensePPCAImageBatch(Y1_score, ctf2_score, Y1_recon, ctf2_recon, y_norm, observation_power)


def iter_dense_ppca_dataset_blocks(
    experiment_dataset,
    mu,
    W=None,
    noise_variance=None,
    rotations=None,
    translations=None,
    *,
    disc_type: str = "linear_interp",
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int | None = None,
    q: int | None = None,
    volume_domain: str = "auto",
    image_indices: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    rotation_translation_mask: np.ndarray | None = None,
    image_scale_corrections: np.ndarray | None = None,
    class_log_prior: float = 0.0,
    score_with_masked_images: bool = False,
    relion_unit_half_weights: bool = False,
    square_window: bool = False,
    relion_texture_interp: bool = True,
    skip_empty_pose_blocks: bool = False,
    collect_observation: bool = False,
    full_real_observation: bool = False,
) -> Iterable[DensePPCAFusedBlock]:
    """Yield prepared dense PPCA fused blocks directly from a dataset."""

    if rotations is None or translations is None:
        raise ValueError("rotations and translations are required")
    if noise_variance is None:
        raise ValueError("noise_variance is required")
    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    if translations.ndim != 2 or translations.shape[1] != 2:
        raise ValueError(f"translations must have shape [T, 2], got {translations.shape}")

    resolved = prepare_dense_ppca_dataset_inputs(
        experiment_dataset,
        mu,
        W,
        q=q,
        volume_domain=volume_domain,
        current_size=current_size,
        relion_unit_half_weights=relion_unit_half_weights,
        square_window=square_window,
        full_real_observation=full_real_observation,
    )
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, resolved.image_shape).squeeze()
    image_indices = (
        np.arange(getattr(experiment_dataset, "n_units", experiment_dataset.n_images))
        if image_indices is None
        else np.asarray(image_indices)
    )
    batch_iter = experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    )

    batch_start = 0
    for batch_data, _rots, _trans, ctf_params, _noise, _particle_indices, indices in batch_iter:
        batch_count = int(len(indices))
        batch = prepare_dense_ppca_image_batch(
            experiment_dataset,
            resolved,
            config,
            noise_variance_half,
            translations,
            batch_data,
            ctf_params,
            indices,
            image_scale_corrections=image_scale_corrections,
            score_with_masked_images=score_with_masked_images,
            relion_unit_half_weights=relion_unit_half_weights,
            collect_observation=collect_observation,
        )
        Y1_score, ctf2_score, Y1_recon, ctf2_recon, y_norm, observation_power = batch

        for r0 in range(0, n_rot, int(rotation_block_size)):
            r1 = min(r0 + int(rotation_block_size), n_rot)
            if skip_empty_pose_blocks and not _pose_mask_block_has_support(
                rotation_translation_mask,
                batch_start=batch_start,
                batch_count=batch_count,
                r0=r0,
                r1=r1,
            ):
                continue
            rotations_block = rotations[r0:r1]
            proj_aug = _project_augmented_half_volumes(
                resolved.augmented_half_volumes,
                rotations_block,
                resolved.image_shape,
                resolved.volume_shape,
                disc_type,
                max_r=resolved.projection_max_r,
                relion_texture_interp=relion_texture_interp,
            )
            if resolved.score_indices is not None:
                proj_aug = proj_aug[:, :, resolved.score_indices]
            pose_log_prior = _per_image_pose_prior_block(
                batch_start=batch_start,
                batch_count=batch_count,
                r0=r0,
                r1=r1,
                n_trans=n_trans,
                rotation_log_prior=rotation_log_prior,
                translation_log_prior=translation_log_prior,
                rotation_translation_mask=rotation_translation_mask,
                class_log_prior=class_log_prior,
            )
            yield DensePPCAFusedBlock(
                Y1=Y1_score,
                proj_aug=proj_aug,
                ctf2_over_noise=ctf2_score,
                y_norm=y_norm,
                rotations=jnp.asarray(rotations_block),
                pose_log_prior=pose_log_prior,
                Y1_recon=Y1_recon,
                ctf2_over_noise_recon=ctf2_recon,
                recon_window_indices=resolved.recon_indices,
                use_recon_window=bool(resolved.use_window),
                backprojection_max_r=resolved.backprojection_max_r,
                batch_start=batch_start,
                rotation_start=r0,
                score_window_indices=resolved.score_indices,
                observation_power=observation_power,
                coefficient_noise=noise_variance_half if collect_observation else None,
            )
        batch_start += batch_count


def iter_dense_ppca_dataset_block_groups(
    experiment_dataset,
    mu,
    W=None,
    noise_variance=None,
    rotations=None,
    translations=None,
    image_scale_corrections=None,
    **kwargs,
) -> Iterable[tuple[DensePPCAFusedBlock, ...]]:
    """Yield all retained rotation blocks for one image batch as a normalization group."""

    group = []
    current_batch_start = None
    for block in iter_dense_ppca_dataset_blocks(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        rotations,
        translations,
        image_scale_corrections=image_scale_corrections,
        **kwargs,
    ):
        block_batch_start = int(block.batch_start)
        if current_batch_start is None:
            current_batch_start = block_batch_start
        elif block_batch_start != current_batch_start:
            if group:
                yield tuple(group)
            group = []
            current_batch_start = block_batch_start
        group.append(block)
    if group:
        yield tuple(group)
