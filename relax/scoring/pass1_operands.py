"""One image batch's operands of the pass-1 score program, by contract: the Gaussian GEMM's and the normalized CC's.

A pass scores with one of the two, decided once from ``score_mode``; the controller builds that route's operand plan
and calls ``plan.prepare(batch, indices)`` for every batch. Both return the same :class:`ScoreOperands`.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from relax.fourier.image_shifts import tiled_half_image_phase_factors
from relax.scoring.coarse_operands import (
    assemble_relion_cc_coarse_operands,
    assemble_relion_exact_coarse_gaussian_operands,
    process_relion_exact_coarse_half_image,
    relion_cc_inverse_power_from_processed,
    repeat_pad_batch_axis,
)
from relax.scoring.gaussian_plan import CoarseGaussianPlan
from relax.scoring.pass1_batch import BatchInputs


class ScoreOperands(NamedTuple):
    """One batch's operands of the score program, which also feed the tree rescore and the score dump.

    ``shifted`` is ``[B, T, S]`` complex64: the images' scored rows translated by every translation (``B`` rows of the
    padded batch, ``S`` scored rows). ``unshifted`` is the same rows before translation, ``[B, S]``. ``pixel_weight`` is
    the per-pixel weight of the score (the Gaussian's noise and CTF weight, or the normalized CC's ``corr_img`` times
    the half-spectrum weights). ``initial_diff2`` is the Gaussian's high-resolution image power (``None`` for the
    normalized CC), ``corr_img`` the normalized CC's correlation image (``None`` for the Gaussian).
    """

    shifted: Any
    unshifted: Any
    pixel_weight: Any
    initial_diff2: Any
    corr_img: Any

    def program_inputs(self) -> tuple:
        """The three arrays the score program takes, at its dtypes: complex64, float32 and float32 (or ``None``)."""

        return (
            jnp.asarray(self.shifted, dtype=jnp.complex64),
            jnp.asarray(self.pixel_weight, dtype=jnp.float32),
            None if self.initial_diff2 is None else jnp.asarray(self.initial_diff2, dtype=jnp.float32),
        )


@dataclass(frozen=True)
class GaussianOperandPlan:
    """What the Gaussian route prepares every batch's operands from, fixed for the pass.

    ``gaussian_plan`` gives the scored rows and the power kernel. ``current_size`` is the call's current size; the
    GEMM runs at the plan's physical size when ``stable_fourier_window_shapes`` (then ``score_size`` is the runtime
    size). ``coarse_kernel_window`` and ``coarse_kernel_r_max`` name the rows RELION's coarse kernel shifts at a
    relabelled position (``None`` when no row is).
    """

    experiment_dataset: Any
    gaussian_plan: CoarseGaussianPlan
    image_shape: Any
    half_weights: Any
    translations_source: Any
    relion_translation_angle_scale: float
    score_with_masked_images: bool
    nyquist_column_counting: str
    scale_corrections_enabled: bool
    use_float64_scoring: bool
    stable_fourier_window_shapes: bool
    score_size: int
    current_size: int | None
    coarse_kernel_window: int | None
    coarse_kernel_r_max: int | None

    @property
    def score_indices(self):
        """The scored rows of the half spectrum, on the device."""

        return self.gaussian_plan.score_indices

    def prepare(self, batch: BatchInputs, indices) -> ScoreOperands:
        """The Gaussian operands of ``batch``, whose images are the dataset images ``indices``."""

        from relax.fourier.half_spectrum import redundant_nyquist_column_pixels

        processed_direct = process_relion_exact_coarse_half_image(
            self.experiment_dataset,
            batch.batch_data,
            self.score_with_masked_images,
            relion_preprocess_kwargs=batch.relion_preprocess_kwargs,
            image_indices=batch.batch_image_indices,
        )
        if self.nyquist_column_counting != "relion":
            # The score weights are zero on these pixels; zeroing them here also takes them
            # out of powerClass's high-shell image power (the diff2 constant).
            processed_direct = jnp.where(
                jnp.asarray(redundant_nyquist_column_pixels(self.image_shape))[None, :],
                jnp.zeros((), dtype=processed_direct.dtype),
                processed_direct,
            )
        operands = assemble_relion_exact_coarse_gaussian_operands(
            self.experiment_dataset,
            processed_direct,
            indices,
            use_float64_scoring=self.use_float64_scoring,
            batch_scale_np=batch.batch_scale_np,
            batch_size=batch.batch_size,
            score_indices=self.gaussian_plan.score_indices,
            score_indices_np=self.gaussian_plan.score_indices_np,
            score_active_mask=self.gaussian_plan.score_active_mask,
            translations_source=self.translations_source,
            relion_translation_angle_scale=self.relion_translation_angle_scale,
            image_shape=self.image_shape,
            noise_variance_half=batch.batch_noise_half,
            scale_corrections_enabled=self.scale_corrections_enabled,
            half_weights=self.half_weights,
            powerclass=self.gaussian_plan.powerclass,
            current_size=(
                self.gaussian_plan.square_layout.physical_current_size
                if self.stable_fourier_window_shapes
                else self.current_size
            ),
            runtime_current_size=(
                jnp.asarray(self.score_size, dtype=jnp.int32) if self.stable_fourier_window_shapes else None
            ),
            coarse_kernel_window=self.coarse_kernel_window,
            coarse_kernel_r_max=self.coarse_kernel_r_max,
        )
        return ScoreOperands(
            shifted=operands.shifted_corrected,
            unshifted=operands.unshifted_corrected,
            pixel_weight=operands.pixel_weight,
            initial_diff2=operands.initial_diff2,
            corr_img=None,
        )


@dataclass(frozen=True)
class CcOperandPlan:
    """What the normalized-CC route prepares every batch's operands from, fixed for the pass.

    ``window_indices`` are the scored rows of the half spectrum when the pass uses a window (``None``: all rows) and
    ``score_indices`` the same rows on the device; ``score_half_weights`` weights them. ``support_power_weights`` is
    ``score_half_weights`` when the image power is taken on the Gaussian support (``firstiter_cc_support="gaussian"``),
    else ``None``. ``translation_angles`` is RELION's float32 ``[T, 2]`` table of the translations.
    """

    experiment_dataset: Any
    image_shape: Any
    image_pre_shifts: Any
    window_indices: Any
    score_indices: Any
    score_half_weights: Any
    support_power_weights: Any
    translation_angles: Any
    n_trans: int
    score_with_masked_images: bool
    scale_corrections_enabled: bool

    def prepare(self, batch: BatchInputs, indices) -> ScoreOperands:
        """The normalized-CC operands of ``batch``, whose images are the dataset images ``indices``."""

        from relax.cuda import kernels as em_cuda_kernels
        from relax.relion.ctf import relion_exact_ctf_half_from_source_star

        processed = process_relion_exact_coarse_half_image(
            self.experiment_dataset,
            batch.batch_data,
            self.score_with_masked_images,
            relion_preprocess_kwargs=batch.relion_preprocess_kwargs,
            image_indices=batch.batch_image_indices,
        )
        phase_factors = None
        if self.image_pre_shifts is not None and not batch.real_space_pre_shift_applied:
            shifts = np.asarray(self.image_pre_shifts)[np.asarray(indices)]
            phase_factors = tiled_half_image_phase_factors(
                self.image_shape,
                jnp.asarray(repeat_pad_batch_axis(shifts, batch.batch_size)),
                1,
            )
        # The padded rows of a short last batch repeat its first image's CTF row.
        operands = assemble_relion_cc_coarse_operands(
            processed,
            relion_exact_ctf_half_from_source_star(
                self.experiment_dataset,
                repeat_pad_batch_axis(np.asarray(indices), batch.batch_size),
                self.image_shape,
            ),
            relion_cc_inverse_power_from_processed(
                processed,
                self.window_indices,
                self.support_power_weights,
            ),
            jnp.asarray(batch.batch_scale_np, dtype=jnp.float32),
            phase_factors=phase_factors,
            window_indices=self.window_indices,
            scale_corrections_enabled=self.scale_corrections_enabled,
        )
        shifted = em_cuda_kernels.relion_translate_score_f32(
            operands.windowed_unshifted,
            self.translation_angles,
            self.score_indices,
            self.image_shape,
        ).reshape(batch.batch_size, self.n_trans, -1)
        return ScoreOperands(
            shifted=shifted,
            unshifted=operands.windowed_unshifted,
            pixel_weight=operands.windowed_corr_img * self.score_half_weights,
            initial_diff2=None,
            corr_img=operands.windowed_corr_img,
        )
