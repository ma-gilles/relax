"""One image batch's inputs to pass 1: the preprocessed images and the per-image rows, before any score operand."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from relax.helpers.image_shifts import apply_relion_integer_pre_shifts
from relax.helpers.preprocessing import prepare_batch_preprocess_operands
from relax.relion.relion_coarse_operands import _repeat_pad_batch_axis


def batch_image_count(batch_data) -> int:
    """The number of image rows of a batch, from its static shape.

    Counting rows must not read the images back from the device: an array that cannot be converted to NumPy (or an
    abstract one, under ``jax.eval_shape``) still has a row count.
    """

    return batch_data.shape[0]


def _pad_significance_preprocess_inputs(
    batch_data,
    integer_pre_shifts,
    batch_scale,
    relion_preprocess_kwargs,
    *,
    target_size: int,
):
    """Give a tail significance batch the same compiled image shape as full batches."""

    actual_size = int(batch_data.shape[0])
    target_size = max(actual_size, int(target_size))
    if actual_size == target_size:
        return (
            batch_data,
            integer_pre_shifts,
            batch_scale,
            relion_preprocess_kwargs,
        )
    padded_kwargs = None
    if relion_preprocess_kwargs is not None:
        padded_kwargs = {
            key: jnp.asarray(_repeat_pad_batch_axis(value, target_size))
            for key, value in relion_preprocess_kwargs.items()
        }
    return (
        _repeat_pad_batch_axis(batch_data, target_size),
        (None if integer_pre_shifts is None else _repeat_pad_batch_axis(integer_pre_shifts, target_size)),
        _repeat_pad_batch_axis(batch_scale, target_size),
        padded_kwargs,
    )


@dataclass(frozen=True)
class BatchInputPlan:
    """What pass 1 prepares every image batch from, fixed for the pass.

    ``image_corrections``, ``scale_corrections`` and ``image_pre_shifts`` are per-image RELION factors indexed by
    dataset image (``None``: absent). ``pad_final_image_batch`` repeat-pads a short last batch to ``image_batch_size``
    rows so that every program sees one image extent. ``noise_variance_half`` is the one shared spectrum; with a
    ``noise_table_host`` ``[G, P]`` and ``image_groups_host`` (each image's optics group) every image takes its own
    group's row. ``translation_log_prior`` is ``None``, ``[T]`` (shared by every image) or ``[N, T]`` (one row per
    image of the pass).
    """

    experiment_dataset: Any
    image_corrections: Any
    scale_corrections: Any
    image_pre_shifts: Any
    score_real_dtype: Any
    pad_final_image_batch: bool
    image_batch_size: int
    noise_variance_half: Any
    noise_table_host: Any
    image_groups_host: Any
    translation_log_prior: Any


class BatchInputs(NamedTuple):
    """One image batch after preprocessing; ``batch_size`` rows, of which the first ``len(indices)`` are images.

    ``batch_data`` is the device batch, ``batch_scale_np`` the per-row scale, ``relion_preprocess_kwargs`` the operands
    of RELION's CUDA preprocessing, ``batch_image_indices`` the dataset index of every row (repeat-padded as the images
    are), ``batch_noise_half`` the noise spectrum of every row (or the one shared spectrum), and
    ``translation_log_prior`` ``None`` or the batch's ``[T]`` / ``[batch_size, T]`` prior on the device.
    ``real_space_pre_shift_applied`` says that the integral pre-shifts were applied to the images (the others are
    applied to the Fourier rows by the coarse operands).
    """

    batch_data: Any
    batch_size: int
    batch_scale_np: Any
    relion_preprocess_kwargs: Any
    batch_image_indices: Any
    batch_noise_half: Any
    real_space_pre_shift_applied: bool
    translation_log_prior: Any


def prepare_batch_inputs(plan: BatchInputPlan, batch_data, indices, *, start_idx: int, end_idx: int) -> BatchInputs:
    """Preprocess one batch of the pass: images ``start_idx:end_idx`` of the pass, dataset images ``indices``.

    Refuses a dataset whose images RELION's CUDA preprocessing cannot take: pass 1 scores RELION's exact coarse operands.
    """

    actual_batch_size = len(indices)
    (
        relion_cuda_preprocess,
        integer_pre_shifts,
        _,
        batch_scale_np,
        relion_preprocess_kwargs,
    ) = prepare_batch_preprocess_operands(
        plan.experiment_dataset,
        batch_data,
        indices,
        image_corrections=plan.image_corrections,
        scale_corrections=plan.scale_corrections,
        image_pre_shifts=plan.image_pre_shifts,
        dtype=plan.score_real_dtype,
    )
    if plan.pad_final_image_batch and actual_batch_size < int(plan.image_batch_size):
        (
            batch_data,
            integer_pre_shifts,
            batch_scale_np,
            relion_preprocess_kwargs,
        ) = _pad_significance_preprocess_inputs(
            batch_data,
            integer_pre_shifts,
            batch_scale_np,
            relion_preprocess_kwargs,
            target_size=int(plan.image_batch_size),
        )
    batch_size = batch_image_count(batch_data)
    # The batch's dataset images, repeat-padded as batch_data is.
    batch_image_indices = _repeat_pad_batch_axis(np.asarray(indices), batch_size)
    # Each image's own optics-group spectrum; the one shared spectrum otherwise.
    batch_noise_half = plan.noise_variance_half
    if plan.noise_table_host is not None:
        batch_groups = plan.image_groups_host[np.asarray(indices, dtype=np.int64)]
        batch_noise_half = jnp.asarray(_repeat_pad_batch_axis(plan.noise_table_host[batch_groups], batch_size))
    real_space_pre_shift_applied = integer_pre_shifts is not None
    if real_space_pre_shift_applied and not relion_cuda_preprocess:
        batch_data = apply_relion_integer_pre_shifts(batch_data, integer_pre_shifts)
    batch_data = jnp.asarray(batch_data)
    if plan.translation_log_prior is None:
        batch_translation_log_prior = None
    elif plan.translation_log_prior.ndim == 1:
        batch_translation_log_prior = jnp.asarray(plan.translation_log_prior)
    else:
        batch_translation_log_prior_np = np.asarray(plan.translation_log_prior[start_idx:end_idx])
        if batch_size > actual_batch_size:
            batch_translation_log_prior_np = _repeat_pad_batch_axis(
                batch_translation_log_prior_np,
                batch_size,
            )
        batch_translation_log_prior = jnp.asarray(batch_translation_log_prior_np)

    if not relion_cuda_preprocess or relion_preprocess_kwargs is None:
        raise ValueError("pass 1 scores RELION's exact coarse operands and needs RELION's CUDA image preprocessing")
    return BatchInputs(
        batch_data=batch_data,
        batch_size=batch_size,
        batch_scale_np=batch_scale_np,
        relion_preprocess_kwargs=relion_preprocess_kwargs,
        batch_image_indices=batch_image_indices,
        batch_noise_half=batch_noise_half,
        real_space_pre_shift_applied=real_space_pre_shift_applied,
        translation_log_prior=batch_translation_log_prior,
    )
