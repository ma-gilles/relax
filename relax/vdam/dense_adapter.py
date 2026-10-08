"""InitialModel E-step configuration and class inputs for the adaptive K-class route.

The hidden variable axis is ``class x pose``; pseudo-halfsets share one E-step
(reconstruction accumulators are split per halfset) so projection/scoring isn't
duplicated while the VDAM M-step still gets independent halfset BackProjectors.
The E-step itself runs in :mod:`relax.vdam.adaptive_estep`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from recovar.reconstruction.noise import make_radial_noise
from recovar.utils.helpers import get_gpu_memory_total

from relax.helpers.orientation_priors import (
    relion_round_away_from_zero,
    relion_sigma_offset_prior_center,
)
from relax.refinement.optics_shapes import MultiShapeDataset
from relax.relion import relion_projector_setup
from relax.sparse_pass2.engine_record import take_coarse_engine_calls, take_pass_engines
from relax.vdam import native_sampling
from relax.vdam.adaptive_estep import run_adaptive_initial_model_estep
from relax.vdam.estep_common import (
    DenseInitialModelEstepConfig,
    DenseInitialModelEstepResult,
)
from relax.vdam.native_options import NativeInitialModelOptions
from relax.vdam.native_sampling import NativeSamplingPlan
from relax.vdam.state import InitialModelState

INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE = 256
INITIAL_MODEL_LOCAL_BATCH_REFERENCE_COUNT_40GB = 32

_INACTIVE_CLASS_LOG_PRIOR = -1.0e30
_RELION_PROJECTOR_DUMP_DIR_ENV = "RELAX_INITIAL_MODEL_PROJECTOR_DUMP_DIR"
# VDAM prepares its projector one way: the device FFT in double, narrowed to
# the complex64 slab that RELION's GPU projector holds as a float texture
# (the single float32 texture path of a7977c8). The corrected power spectrum
# that seeds tau2 stays double.


logger = logging.getLogger(__name__)


@dataclass
class _IterationProjectorContext:
    """One refresh-to-E-step handoff; never a cache across iterations."""

    prepared: tuple | None = None
    reference: np.ndarray | None = None
    geometry: tuple | None = None

    def refresh(self, state, *, padding_factor):
        # Clear even if construction fails, so stale data cannot survive a retry.
        self.prepared = self.reference = self.geometry = None
        inputs, power = prepare_relion_projector_class_inputs_and_power(state, padding_factor=padding_factor)
        self.prepared = inputs
        self.reference = state.Iref
        self.geometry = (
            int(state.iter), int(state.ori_size), int(state.current_size), int(state.K), int(padding_factor),
        )
        return replace(state, tau2_class=power)

    def take(self, state, *, padding_factor):
        if self.prepared is None:
            raise ValueError("no projector refresh ran before this E-step")
        inputs, reference, geometry = self.prepared, self.reference, self.geometry
        self.prepared = self.reference = self.geometry = None
        expected = (
            int(state.iter), int(state.ori_size), int(state.current_size), int(state.K), int(padding_factor),
        )
        if reference is not state.Iref or geometry != expected:
            raise ValueError("projector refresh/E-step reference or geometry changed")
        return inputs


def _configure_relion_image_mask(dataset, opts: NativeInitialModelOptions) -> None:
    """Configure dataset preprocessing to match InitialModel scoring masks."""

    backend = dataset.image_source.backend
    backend.set_relion_image_mask(
        pixel_size=float(dataset.voxel_size),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
    )
    from relax.cuda import (
        kernels as _em_cuda_kernels,  # noqa: F401  (registers the relion_cuda preprocessor, relax split seam S2)
    )

    backend.set_relion_fourier_backend(opts.image_fourier_backend)


def _noise_variance_from_sigma2(sigma2_noise: np.ndarray, ori_size: int) -> np.ndarray:
    """Convert RELION normalized shell power to engine-frame radial noise (unnormalised FFT).

    One optics group gives the ``[P]`` pixel row; several give one row per group, ``[G, P]``
    (each image is then scored with its group's row through ``optics_group_ids``).
    """
    n4 = int(ori_size) ** 4
    # Keep RELION's RFLOAT shell spectrum through the reciprocal used by the
    # guarded exact coarse path.  The downstream float32 kernels already cast
    # their ordinary operands explicitly; narrowing here first loses up to a
    # few ULP in Minvsigma2 and changes near-threshold candidate weights.
    rows = np.stack(
        [
            np.asarray(make_radial_noise(group * n4, (ori_size, ori_size)), dtype=np.float64).reshape(-1)
            for group in np.asarray(sigma2_noise, dtype=np.float64)
        ]
    )
    return rows[0] if rows.shape[0] == 1 else rows


def _effective_initial_model_image_batch_size(
    requested: int,
    *,
    grid_size: int,
    gpu_memory_gb: float,
) -> int:
    """Conservatively cap exact-local batches for large InitialModel grids.

    Exact fine search has a transient that scales approximately with
    ``batch * grid_size**2`` in addition to its resident projector/cache
    state.  The user-facing batch remains an upper bound; 128-pixel jobs keep
    their established behavior, while 256+ grids scale from 32 images on a
    40 GB accelerator.
    """

    if requested < 1:
        raise ValueError(f"image_batch_size must be positive, got {requested}")
    if grid_size < 1:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if gpu_memory_gb <= 0:
        raise ValueError(f"gpu_memory_gb must be positive, got {gpu_memory_gb}")
    if grid_size < INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE:
        return int(requested)
    scaled_cap = int(
        INITIAL_MODEL_LOCAL_BATCH_REFERENCE_COUNT_40GB
        * (INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE / float(grid_size)) ** 2
        * (float(gpu_memory_gb) / 40.0)
    )
    return min(int(requested), max(1, scaled_cap))


def _dense_estep_config(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    sampling_plan: NativeSamplingPlan,
    translation_offsets: np.ndarray,
    sigma_offset_angstrom: float,
    class_log_priors: np.ndarray,
    pass1_healpix_order: int,
) -> DenseInitialModelEstepConfig:
    image_pre_shifts = relion_round_away_from_zero(translation_offsets)
    coarse_translations = np.asarray(
        sampling_plan.coarse_translations
        if sampling_plan.coarse_translations is not None
        else sampling_plan.translations,
        dtype=np.float32,
    )
    coarse_prior_translations = np.asarray(
        sampling_plan.coarse_prior_translations
        if sampling_plan.coarse_prior_translations is not None
        else coarse_translations,
        dtype=np.float32,
    )
    # InitialModel's accelerated pdf_offset uses the rounded absolute old
    # offset, independently of the same integer shift being pre-applied to the
    # image. RELION computes this prior on the coarse translation grid and
    # reuses each parent value for all oversampled children.
    coarse_translation_log_prior = native_sampling._translation_log_prior(
        coarse_prior_translations,
        voxel_size=float(dataset.voxel_size),
        sigma_angstrom=float(sigma_offset_angstrom),
        old_offsets=image_pre_shifts,
    )
    if sampling_plan.translation_parent is None:
        if int(np.asarray(sampling_plan.translations).shape[0]) != int(coarse_translation_log_prior.shape[1]):
            raise ValueError(
                "translation grid and coarse prior must have the same length without an oversampling parent map"
            )
        translation_log_prior = coarse_translation_log_prior
    else:
        translation_parent = np.asarray(sampling_plan.translation_parent, dtype=np.int64)
        if translation_parent.shape != (int(np.asarray(sampling_plan.translations).shape[0]),):
            raise ValueError("translation_parent must contain one coarse parent per fine translation")
        if np.any(translation_parent < 0) or int(translation_parent.max(initial=-1)) >= int(
            coarse_translation_log_prior.shape[1]
        ):
            raise ValueError("translation_parent contains indices outside the coarse translation prior")
        translation_log_prior = coarse_translation_log_prior[:, translation_parent]

    engine_kwargs: dict = {
        "score_with_masked_images": True,
        "reconstruct_with_masked_images": False,
        # VDAM --grad subtracts Frefctf (ml_optimiser.cpp:10092-10105); lifts BPref CC +0.91→+0.996.
        "reconstruction_subtract_projected_reference": True,
        "relion_firstiter_score_mode": "gaussian",
        "image_pre_shifts": image_pre_shifts,
        "translation_prior_centers": relion_sigma_offset_prior_center(translation_offsets),
    }
    engine_kwargs.update(
        healpix_order=int(sampling_plan.healpix_order),
        oversampling_order=int(sampling_plan.oversampling),
        translation_step=float(sampling_plan.offset_step_px),
        random_perturbation=float(sampling_plan.random_perturbation),
        coarse_translations=coarse_translations,
        particle_diameter_ang=float(opts.particle_diameter),
        pass1_healpix_order=int(pass1_healpix_order),
    )
    # The adaptive route rebuilds RELION's fine translations from the
    # unperturbed host grid (``prepare_adaptive_pass2_grids``).
    if sampling_plan.coarse_base_translations is None:
        raise ValueError("the adaptive route needs the sampling plan's host-double coarse grid")
    engine_kwargs["coarse_base_translations"] = np.asarray(
        sampling_plan.coarse_base_translations, dtype=np.float64
    )
    if (adaptive_fraction := opts.environment.adaptive_fraction) is not None:
        engine_kwargs["adaptive_fraction"] = adaptive_fraction
    if not opts.environment.subtract_projected_reference:
        engine_kwargs["reconstruction_subtract_projected_reference"] = False
    if isinstance(dataset, MultiShapeDataset):
        # Each image shape rebuilds its pre-shifts and coarse pdf_offset in its own pixels.
        engine_kwargs["multi_shape_translations"] = dict(
            offsets_px=np.asarray(translation_offsets, dtype=np.float64),
            coarse_prior_translations=coarse_prior_translations,
            sigma_angstrom=float(sigma_offset_angstrom),
        )
    engine_kwargs["translation_log_prior"] = translation_log_prior
    engine_kwargs["coarse_translation_log_prior"] = coarse_translation_log_prior

    grid_size = int(dataset.image_shape[0])
    gpu_memory_gb = (
        float(get_gpu_memory_total())
        if grid_size >= INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE
        else 40.0
    )
    effective_image_batch_size = _effective_initial_model_image_batch_size(
        int(opts.image_batch_size),
        grid_size=grid_size,
        gpu_memory_gb=gpu_memory_gb,
    )
    return DenseInitialModelEstepConfig(
        noise_variance=noise_variance,
        translations=sampling_plan.translations,
        image_batch_size=effective_image_batch_size,
        rotation_block_size=int(opts.rotation_block_size),
        coarse_engine=str(opts.coarse_engine),
        padding_factor=int(opts.padding_factor),
        class_log_priors=class_log_priors,
        engine_kwargs=engine_kwargs,
    )


def class_log_priors_from_state(state: InitialModelState) -> np.ndarray:
    """Log class priors from ``state.pdf_class`` (collapsed classes get a finite sentinel)."""
    weights = np.asarray(state.pdf_class, dtype=np.float64)
    if weights.shape != (state.K,):
        raise ValueError(f"state.pdf_class must have shape ({state.K},), got {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("state.pdf_class must contain non-negative finite class probabilities")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("state.pdf_class must contain at least one positive class probability")
    out = np.full(state.K, _INACTIVE_CLASS_LOG_PRIOR, dtype=np.float64)
    positive = weights > 0.0
    out[positive] = np.log(weights[positive] / total)
    return out


def _dense_engine_kwargs(state: InitialModelState, config: DenseInitialModelEstepConfig) -> dict[str, Any]:
    engine_kwargs = {
        "current_size": None if state.current_size <= 0 else state.current_size,
        # RELION's radial window at the full box too (ml_optimiser.cpp:5784-5793, :6841-6880).
        "window_at_box": True,
        "projection_padding_factor": config.padding_factor,
        "reconstruction_padding_factor": config.padding_factor,
        "half_spectrum_scoring": True,
        # RELION InitialModel BPref uses the rounded radial reconstruction support
        # encoded by Minvsigma2, not the full square Fourier crop.
        "recon_square_window": False,
        "recon_exact_radius": False,
        # RELION InitialModel scores the full rounded Fourier crop emitted by its
        # CUDA projector, including the few crop-corner pixels outside r_max.
        "projection_mask_current_image_disk": False,
    }
    engine_kwargs.update(config.engine_kwargs)
    return engine_kwargs


def prepare_relion_projector_class_inputs_and_power(
    state: InitialModelState,
    *,
    padding_factor: int,
) -> tuple[tuple[np.ndarray, int], np.ndarray]:
    """Produce scoring operands and tau2 from the identical corrected FFT (RELION's linear interpolator)."""
    half_maps, power, r_max = relion_projector_setup.reference_to_relion_projector_half_maps_and_power(
        state.Iref,
        current_size=state.current_size if state.current_size > 0 else state.ori_size,
        padding_factor=padding_factor,
    )
    return _finish_relion_projector_class_inputs(state, padding_factor, half_maps, r_max), power


def _finish_relion_projector_class_inputs(
    state: InitialModelState,
    padding_factor: int,
    projector_half_by_class: np.ndarray,
    projector_r_max: int,
) -> tuple[np.ndarray, int]:
    projector_dump_dir = os.environ.get(_RELION_PROJECTOR_DUMP_DIR_ENV, "").strip()
    if projector_dump_dir:
        os.makedirs(projector_dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(projector_dump_dir, f"iter{int(state.iter):03d}_relion_projector_half.npz"),
            projector_half=np.asarray(projector_half_by_class),
            projector_r_max=np.int64(projector_r_max),
            current_size=np.int64(
                state.current_size if state.current_size > 0 else state.ori_size
            ),
            padding_factor=np.int64(padding_factor),
            iteration=np.int64(state.iter),
        )
    return projector_half_by_class, int(projector_r_max)


def _resolve_class_inputs(
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
) -> tuple[Any, Any, np.ndarray, int]:
    """The iteration's RELION projector (prepared by the projector refresh), with NaN stand-ins for the
    dense class means.

    The resident adaptive route scores with the projector and reads only K and the dtype of the dense
    N^3 means and their power (a NaN stand-in left 12-iteration K=1 and K=2 maps unchanged, job
    14512033), so ``(K, 1)`` NaN arrays replace them and any read shows up as NaN.
    """
    if config.relion_projector_half_by_class is None or config.relion_projector_r_max is None:
        raise ValueError("the E-step needs the iteration's RELION projector: relion_projector_half_by_class and _r_max")
    relion_projector_half_by_class = np.asarray(config.relion_projector_half_by_class)
    relion_projector_r_max = int(config.relion_projector_r_max)
    means = np.full((int(state.K), 1), np.nan, dtype=np.complex64)
    mean_variance = np.full((int(state.K), 1), np.nan, dtype=np.float32)
    return means, mean_variance, relion_projector_half_by_class, relion_projector_r_max


def run_dense_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    particle_ids: np.ndarray | None = None,
    halfset_ids: np.ndarray | None = None,
) -> DenseInitialModelEstepResult:
    """Run the InitialModel E-step with RELION-compatible pseudo-halfset routing.

    VDAM's one E-step route is the adaptive pass-1/pass-2 route on the
    device-resident pass 2; the dense E-step was removed on 2026-10-03.
    """
    class_log_priors = (
        class_log_priors_from_state(state) if config.class_log_priors is None else np.asarray(config.class_log_priors)
    )
    engine_kwargs = _dense_engine_kwargs(state, config)
    selected_particle_ids = (
        np.arange(int(experiment_dataset.n_images), dtype=np.int64)
        if particle_ids is None
        else np.asarray(particle_ids, dtype=np.int64)
    )
    if state.pseudo_halfsets:
        selected_halfset_ids = (
            np.arange(selected_particle_ids.size, dtype=np.int32) % 2
            if halfset_ids is None
            else np.asarray(halfset_ids, dtype=np.int32)
        )
    else:
        selected_halfset_ids = None
    means, mean_variance, relion_projector_half_by_class, relion_projector_r_max = _resolve_class_inputs(state, config)
    result = run_adaptive_initial_model_estep(
        experiment_dataset,
        state,
        config,
        class_log_priors=class_log_priors,
        joint_particle_ids=selected_particle_ids,
        joint_halfset_ids=selected_halfset_ids,
        means=means,
        mean_variance=mean_variance,
        relion_projector_half_by_class=relion_projector_half_by_class,
        relion_projector_r_max=relion_projector_r_max,
        engine_kwargs=engine_kwargs,
    )
    result.meta["pass2_engines"] = take_pass_engines()
    result.meta["coarse_engine_calls"] = take_coarse_engine_calls()
    return result
