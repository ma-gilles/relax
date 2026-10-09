"""The rotation blocks and the log priors of one pass 1, validated and padded once before its first batch."""

from typing import Any, NamedTuple

import numpy as np


class RotationBlocks(NamedTuple):
    """The pass's rotations and rotation prior padded to a whole number of blocks.

    ``rotations_padded`` is ``[R_padded, 3, 3]``: the rotations followed by identities up to a whole number of
    blocks (the padded tail rows are scored and discarded). ``rotation_log_prior_padded`` is ``[K, R_padded]`` of the
    score dtype (zero in the padding), or ``None`` without a rotation prior.
    """

    rotations_padded: Any
    rotation_log_prior_padded: Any


def plan_rotation_blocks(
    rotations, rotation_log_prior, *, n_classes: int, rotation_block_size: int, score_real_dtype
) -> RotationBlocks:
    """Pad ``rotations`` and ``rotation_log_prior`` (``[R]`` shared by the classes, ``[K, R]``, or ``None``).

    Refuses a rotation prior of another shape.
    """

    n_rot = int(rotations.shape[0])
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate(
            [
                rotations,
                np.tile(np.eye(3, dtype=np.asarray(rotations).dtype), (pad_size, 1, 1)),
            ],
            axis=0,
        )
    else:
        rotations_padded = rotations

    rotation_log_prior_padded = None
    if rotation_log_prior is not None:
        prior = np.asarray(rotation_log_prior, dtype=score_real_dtype)
        if prior.ndim == 1:
            if prior.shape != (n_rot,):
                raise ValueError(f"rotation_log_prior must have shape ({n_rot},), got {prior.shape}")
            prior = np.broadcast_to(prior[None, :], (n_classes, n_rot)).copy()
        elif prior.shape != (n_classes, n_rot):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_rot},) or ({n_classes}, {n_rot}), got {prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.pad(
                prior,
                ((0, 0), (0, n_rot_padded - n_rot)),
                mode="constant",
            )
        else:
            rotation_log_prior_padded = prior
    return RotationBlocks(rotations_padded, rotation_log_prior_padded)


def validated_translation_log_prior(translation_log_prior, *, n_images: int, n_trans: int, score_real_dtype):
    """``translation_log_prior`` as an array of the score dtype: ``[T]`` (shared by every image) or ``[N, T]``.

    ``None`` stays ``None``. Refuses another shape.
    """

    if translation_log_prior is None:
        return None
    translation_log_prior = np.asarray(translation_log_prior, dtype=score_real_dtype)
    if translation_log_prior.ndim == 1:
        if translation_log_prior.shape != (n_trans,):
            raise ValueError(
                f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}"
            )
    elif translation_log_prior.ndim == 2:
        if translation_log_prior.shape != (n_images, n_trans):
            raise ValueError(
                "translation_log_prior must have shape "
                f"({n_images}, {n_trans}) when image-specific, got {translation_log_prior.shape}",
            )
    else:
        raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions")
    return translation_log_prior
