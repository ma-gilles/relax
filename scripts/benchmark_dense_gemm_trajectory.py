"""Controlled fixed-grid, Gaussian K1 dense-GEMM trajectory on RELION data.

This opt-in experiment starts from a half-specific RELION model checkpoint.
It keeps that checkpoint's noise and tau spectra fixed so exact and lagged
arms see the same score/reconstruction state. It is not AutoRefine replay:
there is no significant-support pruning, adaptive grid, or noise/tau update.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import starfile
from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.reconstruction import noise as recon_noise
from recovar.utils.helpers import load_relion_volume

from relax.refinement.run_files import read_star_blocks

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixedGaussianCheckpoint:
    """Half-specific iteration-1 RELION model state in RECOVAR units."""

    model_paths: tuple[Path, Path]
    map_paths: tuple[Path, Path]
    real_maps: tuple[np.ndarray, np.ndarray]
    fourier_maps: tuple[jax.Array, jax.Array]
    noise_radial: tuple[np.ndarray, np.ndarray]
    noise_half: tuple[jax.Array, jax.Array]
    tau_radial: tuple[np.ndarray, np.ndarray]
    current_size: int
    tau2_fudge: float
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class FixedHalfMetadata:
    """Particle-order checkpoint corrections and prior arrays for one half."""

    source_ids: np.ndarray
    image_corrections: np.ndarray
    scale_corrections: np.ndarray
    image_pre_shifts: np.ndarray
    previous_offsets: np.ndarray
    rotation_log_prior: np.ndarray
    translation_log_prior: np.ndarray


def load_fixed_gaussian_checkpoint(data_dir: Path, model_dir: Path, image_shape) -> FixedGaussianCheckpoint:
    """Read two real RELION model STARs/maps and retain their own spectra.

    RELION stores sigma2_noise and reference tau2 in units smaller by N**4
    than the RECOVAR reconstruction path. The same conversion is used by
    ``relax.refinement.run_files.read_run_files`` and the strict replay loader.
    """
    n = int(image_shape[0])
    if tuple(image_shape) != (n, n) or n <= 0 or n % 2:
        raise ValueError("checkpoint expects one even square image shape")
    model_paths = tuple(model_dir / f"run_it001_half{h}_model.star" for h in (1, 2))
    models = tuple(read_star_blocks(path) for path in model_paths)
    current_sizes = [int(model["model_general"]["rlnCurrentImageSize"]) for model in models]
    fudges = [float(model["model_general"]["rlnTau2FudgeFactor"]) for model in models]
    if current_sizes[0] != current_sizes[1] or fudges[0] != fudges[1]:
        raise ValueError("half-specific RELION model geometry or tau2 fudge differs")
    if any(len([key for key in model if key.startswith("model_optics_group_")]) != 1 for model in models):
        raise NotImplementedError("controlled checkpoint requires one optics group")
    frame = float(n) ** 4
    noise_radial = tuple(
        np.asarray(model["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64) * frame
        for model in models
    )

    tau_radial = tuple(
        np.asarray(model["model_class_1"]["rlnReferenceTau2"], dtype=np.float64) * frame
        for model in models
    )
    if any(values.size != n // 2 + 1 or not np.isfinite(values).all() or np.any(values < 0)
           for values in (*noise_radial, *tau_radial)):
        raise ValueError("checkpoint has invalid or incomplete noise/tau spectra")
    map_paths = tuple(
        (model_dir / model["model_classes"]["rlnReferenceImage"][0]).resolve()
        for model in models
    )
    real_maps = tuple(np.asarray(load_relion_volume(path), dtype=np.float32) for path in map_paths)
    if any(values.shape != (n, n, n) or not np.isfinite(values).all() for values in real_maps):
        raise ValueError("checkpoint reference maps have invalid shape or values")
    fourier_maps = tuple(jnp.asarray(ftu.get_dft3(jnp.asarray(values)), dtype=jnp.complex64).reshape(-1)
                         for values in real_maps)
    noise_half = tuple(
        jnp.asarray(
            recon_noise.to_batched_half_pixel_noise(
                recon_noise.make_radial_noise(values, image_shape), image_shape,
            ).squeeze(),
            dtype=jnp.float32,
        )
        for values in noise_radial
    )
    paths = (*model_paths, *map_paths, model_dir / "run_it001_data.star")
    return FixedGaussianCheckpoint(
        model_paths, map_paths, real_maps, fourier_maps,
        noise_radial, noise_half, tau_radial, current_sizes[0], fudges[0],
        {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    )


def load_half_metadata(models, data_table, half_rows, *, voxel_size, base_translations, score_rotations):
    """Bind RELION checkpoint normalization, scale and pose priors by STAR row.

    The serialized model group scale is explicitly treated as the fixed live
    scale for this controlled standalone continuation. RELION's in-memory
    scorer rank scale can differ from the leader's serialized value.
    """
    from relax.helpers.orientation_priors import (
        initial_direction_priors_from_snapshot,
        make_relion_translation_log_prior,
        relion_direction_log_priors_for_half,
        relion_half_translation_prior_inputs,
        relion_translation_search_base,
    )

    direction_pdfs = [np.asarray(model["model_pdf_orient_class_1"]["rlnOrientationDistribution"], np.float32)
                      for model in models]
    global_prior, global_order, _, _ = initial_direction_priors_from_snapshot(
        direction_pdfs, n_classes=1, dtype=np.float32, log=log, symmetry="C1", expected_order=3,
    )
    result = []
    for half, rows in enumerate(half_rows):
        model = models[half]
        table = data_table.iloc[rows]
        groups = np.asarray(table["rlnGroupNumber"], dtype=np.int64) - 1
        group_scales = np.asarray(model["model_groups"]["rlnGroupScaleCorrection"], np.float64)
        normcorr = np.asarray(table["rlnNormCorrection"], np.float64)
        if np.any(groups < 0) or np.any(groups >= len(group_scales)) or np.any(normcorr <= 0):
            raise ValueError("checkpoint group or normalization rows are invalid")
        scale = group_scales[groups].astype(np.float32)
        avg_norm = float(model["model_general"]["rlnNormCorrectionAverage"])
        correction = ((avg_norm / normcorr) * scale).astype(np.float32)
        previous = np.column_stack((
            np.asarray(table["rlnOriginXAngst"], np.float64),
            np.asarray(table["rlnOriginYAngst"], np.float64),
        )) / float(voxel_size)
        previous = previous.astype(np.float32)
        pre_shift = relion_translation_search_base(previous, dtype=np.float32)
        direction = relion_direction_log_priors_for_half(
            use_local=False, scoring_healpix_order=3, n_classes=1,
            class_direction_prior=None, class_direction_prior_order=None,
            global_direction_prior=global_prior[half], global_direction_prior_order=global_order[half],
            sealed_sampling_state=None, dtype=np.float32, log=log, half_index=half, symmetry="C1",
        ).rotation_log_prior
        if direction is None or len(direction) != score_rotations:
            raise ValueError("checkpoint direction PDF does not align with the fixed SO(3) grid")
        prior_inputs = relion_half_translation_prior_inputs(
            previous, voxel_size=voxel_size, base_translations=base_translations,
            current_translations=base_translations, dtype=np.float32,
        )
        translation = make_relion_translation_log_prior(
            prior_inputs.prior_translations, voxel_size,
            float(model["model_general"]["rlnSigmaOffsetsAngst"]),
            prior_centers=prior_inputs.prior_center,
            offset_range_pixels=None, dtype=np.float32,
        )
        result.append(FixedHalfMetadata(
            np.asarray(rows, dtype=np.int32), correction, scale,
            np.asarray(pre_shift, dtype=np.int32), previous,
            np.asarray(direction, dtype=np.float32), np.asarray(translation, dtype=np.float32),
        ))
    return tuple(result)


def build_fixed_fixture_grid(dataset, checkpoint, model_dir, *, rotations, translations, rotation_tile, translation_tile):
    """Freeze a deterministic subset of the canonical checkpoint's full SO(3)/shift grid."""
    from relax.dense import scoring_policy
    from relax.dense.gemm_experiment import native_phase_table, pad_grid
    from relax.helpers.oversampling import prepare_adaptive_pass2_grids
    from relax.refinement.iteration_planning import build_initial_coarse_grids
    from relax.relion.optics_aberrations import projection_rotations
    from relax.relion.relion_metadata import read_relion_sampling_metadata, read_relion_sampling_symmetry
    from relax.scoring.coarse_layout import plan_coarse_gaussian_square_layout
    from relax.sparse_pass2.sparse_pass2_bucket_io import _relion_translation_angles_f32
    from relax.sparse_pass2.sparse_pass2_window import _pass2_window_setup, _sparse_pass2_window_setup

    sampling_path = model_dir / "run_it001_sampling.star"
    sampling_state = read_relion_sampling_metadata(sampling_path)
    symmetry = read_relion_sampling_symmetry(sampling_path)
    if symmetry != "C1" or sampling_state["healpix_order"] != 3:
        raise NotImplementedError("controlled fixture expects C1 HEALPix order 3")
    pixel_size = float(dataset.voxel_size)
    base = build_initial_coarse_grids(
        3, None,
        translation_range=float(sampling_state["offset_range"]) / pixel_size,
        translation_step=float(sampling_state["offset_step"]) / pixel_size,
        n_classes=1, voxel_size=pixel_size, symmetry="C1",
        dtype=scoring_policy.DENSE_PRECISION.rotation_real_dtype,
    )
    pose_grid = prepare_adaptive_pass2_grids(
        base.rotation_grid.rotations, base.translations, base.base_translations,
        healpix_order=base.rotation_grid.healpix_order, adaptive_oversampling=0,
        translation_step=float(sampling_state["offset_step"]) / pixel_size,
        random_perturbation=0.0, coarse_rotation_ids=None, symmetry=base.rotation_grid.symmetry,
    )
    score_all = np.asarray(projection_rotations(pose_grid.coarse_rotations, 1.0), dtype=np.float32)
    bp_all = np.asarray(projection_rotations(pose_grid.fine_mstep_rotations, 1.0), dtype=np.float32)
    trans_all = np.asarray(pose_grid.coarse_translation_phase_source, dtype=np.float32)
    if rotations <= 0 or rotations > len(score_all) or translations <= 0 or translations > len(trans_all):
        raise ValueError("requested fixed-grid subset exceeds checkpoint sampling grid")
    rotation_ids = np.linspace(0, len(score_all) - 1, rotations, dtype=np.int32)
    translation_ids = np.linspace(0, len(trans_all) - 1, translations, dtype=np.int32)
    if len(np.unique(rotation_ids)) != rotations or len(np.unique(translation_ids)) != translations:
        raise ValueError("fixed-grid selection repeated a pose")
    n = int(dataset.image_shape[0])
    window_plan = _pass2_window_setup(
        dataset.image_shape, current_size=checkpoint.current_size,
        reconstruction_current_size=checkpoint.current_size,
        half_spectrum_scoring=True, square_window=False,
        relion_firstiter_score_mode="gaussian", use_exact_relion_gaussian=True,
        use_float64_scoring=False, window_at_box=True,
    )
    window = _sparse_pass2_window_setup(
        dataset, disc_type="linear_interp", image_shape=dataset.image_shape,
        current_size=checkpoint.current_size, n_half=window_plan.n_half,
        mstep_current_size=window_plan.mstep_current_size, square_window=False,
        window_spec_kwargs=window_plan.window_spec_kwargs,
        use_relion_x_half_mstep=True, log_label="Dense GEMM fixture",
    )
    active_score = (
        np.arange(n * (n // 2 + 1), dtype=np.int32)
        if window.window_spec.score_indices_np is None
        else np.asarray(window.window_spec.score_indices_np, dtype=np.int32)
    )
    score_layout = plan_coarse_gaussian_square_layout(
        dataset.image_shape, checkpoint.current_size, active_score,
        stable_fourier_window_shapes=False,
    )
    score_indices = jnp.asarray(score_layout.score_indices_np, dtype=jnp.int32)
    score_mask = jnp.asarray(score_layout.score_active_mask_np)
    angles = _relion_translation_angles_f32(
        trans_all[translation_ids], dataset.image_shape, angle_scale=1.0,
    )
    score_phase = native_phase_table(angles, score_indices, dataset.image_shape)
    rec_phase = native_phase_table(angles, window.recon_window_indices, dataset.image_shape)
    grid = pad_grid(
        score_all[rotation_ids], bp_all[rotation_ids], score_phase, rec_phase,
        rotation_tile=rotation_tile, translation_tile=translation_tile,
    )
    return {
        "grid": grid, "window": window, "score_indices": score_indices,
        "score_mask": score_mask, "score_layout": score_layout,
        "angles": angles, "rotation_ids": rotation_ids,
        "translation_ids": translation_ids, "base_translations": trans_all,
        "translation_values": trans_all[translation_ids],
        "full_rotation_count": len(score_all), "full_translation_count": len(trans_all),
    }


def prepare_real_batch(dataset, checkpoint, half_index, metadata, geometry, local_indices, *, batch_capacity, sentinel_id):
    """Assemble one batch through the existing coarse and BPref operand owners."""
    from relax.dense.gemm_experiment import pad_batch
    from relax.helpers.batch_fetch import iter_indexed_batches
    from relax.helpers.half_spectrum import (
        make_relion_noise_shell_indices_half,
        make_scoring_half_image_weights,
        mask_relion_noise_shell_indices_to_current_window,
    )
    from relax.helpers.preprocessing import prepare_batch_preprocess_operands
    from relax.relion.relion_coarse_operands import (
        _assemble_relion_exact_coarse_gaussian_operands,
        _process_relion_exact_coarse_half_image,
    )
    from relax.sparse_pass2.resident_operands import prepare_resident_half_operands
    from relax.sparse_pass2.sparse_pass2_scoring import _relion_cuda_powerclass_highres_xi2_half
    from relax.sparse_pass2.sparse_pass2_wavg import _make_relion_wavg_rectangle

    local_indices = np.asarray(local_indices, dtype=np.int32)
    if local_indices.ndim != 1 or len(local_indices) == 0 or len(local_indices) > batch_capacity:
        raise ValueError("local batch rows must fit one fixed image capacity")
    fetched = list(iter_indexed_batches(dataset, local_indices, len(local_indices)))
    if len(fetched) != 1:
        raise RuntimeError("canonical batch fetch yielded an unexpected number of chunks")
    batch_data, _, _, ctf_params, _, _, fetched_indices = fetched[0]
    if not np.array_equal(np.asarray(fetched_indices), local_indices):
        raise ValueError("canonical batch fetch changed particle order")
    _, _, _, batch_scale, preprocess_kwargs = prepare_batch_preprocess_operands(
        dataset, batch_data, local_indices,
        image_corrections=metadata.image_corrections,
        scale_corrections=metadata.scale_corrections,
        image_pre_shifts=metadata.image_pre_shifts,
        dtype=np.float32,
    )
    if preprocess_kwargs is None:
        raise RuntimeError("real fixture requires canonical RELION CUDA image preprocessing")
    raw_device = jnp.asarray(batch_data)
    processed = _process_relion_exact_coarse_half_image(
        dataset, raw_device, True, relion_preprocess_kwargs=preprocess_kwargs,
    )
    score_indices = geometry["score_indices"]
    score_half_weights = make_scoring_half_image_weights(
        dataset.image_shape, relion_half_sum=True, exclude_relion_redundant_x0=True,
    )
    exact = _assemble_relion_exact_coarse_gaussian_operands(
        dataset, processed, local_indices,
        batch_scale_np=batch_scale, actual_batch_size=len(local_indices),
        batch_size=len(local_indices), score_indices=score_indices,
        score_indices_np=geometry["score_layout"].score_indices_np,
        score_active_mask=geometry["score_mask"],
        translations_source=geometry["translation_values"],
        image_shape=dataset.image_shape, noise_variance_half=checkpoint.noise_half[half_index],
        scale_corrections_enabled=True, half_weights=score_half_weights,
        powerclass=_relion_cuda_powerclass_highres_xi2_half,
        current_size=checkpoint.current_size, use_float64_scoring=False,
    )

    window = geometry["window"]
    rect = _make_relion_wavg_rectangle(
        dataset.image_shape, checkpoint.current_size, window.recon_window_indices,
        reconstruction_current_size=checkpoint.current_size,
    )
    noise_shells = mask_relion_noise_shell_indices_to_current_window(
        make_relion_noise_shell_indices_half(dataset.image_shape),
        dataset.image_shape, checkpoint.current_size, window.window_indices,
    )
    selected_prior = metadata.translation_log_prior[np.ix_(local_indices, geometry["translation_ids"])]
    bucket_kwargs = dict(
        noise_variance_half=checkpoint.noise_half[half_index],
        fine_translations=geometry["translation_values"],
        config=window.config, n_trans=len(geometry["translation_ids"]),
        score_with_masked_images=True, half_spectrum_scoring=True,
        image_corrections=metadata.image_corrections,
        scale_corrections=metadata.scale_corrections,
        image_pre_shifts=metadata.image_pre_shifts,
        use_float64_scoring=False, score_only=False, score_mode="gaussian",
        window_indices=window.window_indices,
        recon_window_indices=window.recon_window_indices,
        translation_phases_half=None,
        relion_score_translation_angles=exact.translation_angles,
        return_windowed_shifted=window.windowed_prepare,
        relion_exact_normalized_cc_operands=False,
        relion_exact_bpref_operands=True, noise_optics_groups=None,
    )
    resident = prepare_resident_half_operands(
        dataset, local_indices, bucket_io_kwargs=bucket_kwargs,
        window_indices=window.window_indices,
        recon_window_indices=window.recon_window_indices,
        wavg_rect_indices=rect.centered_indices,
        noise_shell_indices_half=noise_shells,
        n_noise_shells=dataset.image_shape[0] // 2 + 1,
        image_shape=dataset.image_shape, current_size=checkpoint.current_size,
        n_fine_trans=len(geometry["translation_ids"]),
        use_exact_relion_gaussian=True, accumulate_noise=False,
        source_faithful_spectrum_norm=True,
        fine_translation_prior_2d=metadata.translation_log_prior[:, geometry["translation_ids"]],
        scale_corrections_np=metadata.scale_corrections,
        group_ids_np=None, precision_policy=None,
        image_batch_size=batch_capacity, optics_groups_np=None,
        relion_native_fine_units=False, log_summary=False,
        staged_batch=(raw_device, ctf_params, np.asarray(fetched_indices)),
    )
    # The resident owner rounds its image axis to a 256-row capacity class.
    # Only requested rows belong to this trajectory batch.
    raw = jnp.asarray(resident.recon_image[: len(local_indices)], dtype=jnp.complex64)
    weighted_ctf = jnp.asarray(resident.recon_weight[: len(local_indices)], dtype=jnp.float32)
    return pad_batch(
        exact.unshifted_corrected, exact.pixel_weight, exact.initial_diff2,
        raw * weighted_ctf, resident.ctf2_over_nv_recon[: len(local_indices)],
        np.broadcast_to(metadata.rotation_log_prior[geometry["rotation_ids"]],
                        (len(local_indices), len(geometry["rotation_ids"]))),
        selected_prior,
        metadata.source_ids[local_indices], image_capacity=batch_capacity,
        grid=geometry["grid"], sentinel_id=sentinel_id,
        rec_raw_image=raw, rec_weighted_ctf=weighted_ctf,
    )


def load_fixture_execution_state(data_dir, model_dir, *, rotations, translations, rotation_tile, translation_tile):
    """Resolve frozen STAR identities, native image backend and fixed grid."""
    from relax.relion.relion_metadata import _load_relion_mask_params

    data_star = model_dir / "run_it001_data.star"
    dataset = load_dataset(
        str(data_star), datadir=str(data_dir), lazy=True,
        dtype=np.complex64, absent_angles_zero=True,
    )
    checkpoint = load_fixed_gaussian_checkpoint(data_dir, model_dir, dataset.image_shape)
    data_table = starfile.read(data_star)["particles"]
    if len(data_table) != dataset.n_units or not data_table["rlnImageName"].is_unique:
        raise ValueError("checkpoint STAR identities do not match dataset")
    if not np.array_equal(dataset.dataset_indices, np.arange(dataset.n_units)):
        raise ValueError("fixture loader changed STAR row order; remap particles by rlnImageName")
    half_rows = tuple(
        np.flatnonzero(np.asarray(data_table["rlnRandomSubset"], np.int32) == h).astype(np.int32)
        for h in (1, 2)
    )
    if sum(map(len, half_rows)) != dataset.n_units:
        raise ValueError("checkpoint halfsets do not cover every particle")
    dataset.halfset_indices = half_rows
    mask = _load_relion_mask_params(model_dir / "run_it001_optimiser.star")
    if mask is None or float(mask[0]) <= 0:
        raise ValueError("numbered RELION checkpoint has no scoring mask")
    backend = dataset.image_source.backend
    backend.set_relion_image_mask(
        pixel_size=dataset.voxel_size,
        particle_diameter_ang=float(mask[0]),
        width_mask_edge_px=int(mask[1]),
    )
    dataset.image_source.image_mask = backend.image_mask
    backend.set_relion_fourier_backend("relion_cuda")
    geometry = build_fixed_fixture_grid(
        dataset, checkpoint, model_dir, rotations=rotations, translations=translations,
        rotation_tile=rotation_tile, translation_tile=translation_tile,
    )
    models = tuple(read_star_blocks(path) for path in checkpoint.model_paths)
    half_metadata = load_half_metadata(
        models, data_table, half_rows, voxel_size=dataset.voxel_size,
        base_translations=geometry["base_translations"],
        score_rotations=geometry["full_rotation_count"],
    )
    halves = tuple(dataset.get_halfset_dataset(h) for h in (0, 1))
    return dataset, halves, checkpoint, half_metadata, geometry


def probe_real_operands(args, model_dir):
    """Run one real two-half operand assembly without an EM iteration."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("real operand probe requires a selected CUDA GPU")
    dataset, halves, checkpoint, metadata, geometry = load_fixture_execution_state(
        args.data_dir, model_dir, rotations=args.rotations, translations=args.translations,
        rotation_tile=args.rotation_tile, translation_tile=args.translation_tile,
    )
    rows = []
    for half in (0, 1):
        begin = time.perf_counter()
        batch = prepare_real_batch(
            halves[half], checkpoint, half, metadata[half], geometry,
            np.arange(min(args.images, halves[half].n_units), dtype=np.int32),
            batch_capacity=args.images, sentinel_id=dataset.n_units,
        )
        jax.block_until_ready(batch.score_image)
        jax.block_until_ready(batch.rec_image)
        rows.append({
            "half": half + 1,
            "seconds": time.perf_counter() - begin,
            "score_shape": list(batch.score_image.shape),
            "recon_shape": list(batch.rec_image.shape),
            "score_dtype": str(batch.score_image.dtype),
            "recon_dtype": str(batch.rec_image.dtype),
            "source_ids": np.asarray(batch.particle_ids[: args.images]).tolist(),
            "score_weight_nonzero": int(np.count_nonzero(np.asarray(batch.score_weight))),
            "recon_weight_nonzero": int(np.count_nonzero(np.asarray(batch.rec_weight))),
            "initial_diff2_range": [float(np.min(np.asarray(batch.initial_diff2))),
                                    float(np.max(np.asarray(batch.initial_diff2)))],
        })
    return {
        "schema": "relax.dense_gemm_em.fixture_operand_probe.v1",
        "checkpoint": inspect_checkpoint(args.data_dir, model_dir),
        "policy": "fixed iteration-1 maps/noise/tau, serialized group scale as live scale",
        "grid": {
            "full_rotations": geometry["full_rotation_count"],
            "full_translations": geometry["full_translation_count"],
            "selected_rotation_ids": geometry["rotation_ids"].tolist(),
            "selected_translation_ids": geometry["translation_ids"].tolist(),
        },
        "batches": rows,
    }


def _projector_slab(fourier_map, volume_shape, current_size):
    """Build the canonical RELION Projector data at the map-update boundary."""
    from relax.relion.relion_projector_setup import reference_to_relion_projector_half_maps_and_power

    real_map = jnp.real(ftu.get_idft3(jnp.asarray(fourier_map).reshape(volume_shape)))
    slabs, _power, radius = reference_to_relion_projector_half_maps_and_power(
        real_map[None, ...], current_size=current_size, padding_factor=2,
        projector_data_dtype=np.complex64,
    )
    if radius != current_size // 2 or slabs.shape[0] != 1:
        raise ValueError("checkpoint projector has the wrong radius or class count")
    return jnp.asarray(slabs[0], dtype=jnp.complex64), radius


def _finalize_and_reconstruct(result, checkpoint, half, volume_shape, bp_shape):
    """Use the existing BPref finalizer and RELION-style map solver."""
    from relax.helpers.half_volume_mstep import (
        finalize_half_volume_bpref,
        relion_x_half_accumulators_to_public_layout,
    )
    from relax.refinement.mean_helpers import _reconstruct_volume_eager
    from relax.refinement.refinement_options import ReconstructionPrograms

    numerator, denominator = finalize_half_volume_bpref(
        result.numerator, result.denominator, bp_shape, logger=log,
        label="Dense GEMM fixed-grid fixture", symmetry_label="C1", relion_x_half=True,
    )
    numerator, denominator = relion_x_half_accumulators_to_public_layout(
        numerator, denominator, bp_shape,
    )
    new_map = _reconstruct_volume_eager(
        denominator, numerator, volume_shape, 2,
        tau=checkpoint.tau_radial[half], tau2_fudge=checkpoint.tau2_fudge,
        projection_padding_factor=2, current_size=checkpoint.current_size,
        accumulator_volume_shape=bp_shape, tau_is_1d=True,
        preserve_output_precision=True, relion_filter_scale=float(volume_shape[0] ** 4), programs=ReconstructionPrograms.from_environ(),
    ).reshape(-1)
    if new_map.dtype != jnp.complex64 or new_map.size != int(np.prod(volume_shape)):
        raise ValueError("map reconstruction changed Fourier dtype or volume geometry")
    return new_map, numerator, denominator


def _checked_particle_rows(result, source_ids):
    """Reject invalid full-grid normalizers, weights or volume values."""
    ids = np.asarray(source_ids, dtype=np.int32)
    pair = np.asarray(result.next_pair_table[ids])
    mass = np.asarray(result.mass_table[ids])
    bad_normalizer = np.asarray(result.invalid_normalizer_table[ids])
    bad_weight = np.asarray(result.invalid_weight_table[ids])
    if (bad_normalizer.any() or bad_weight.any() or not np.isfinite(pair).all()
            or not np.isfinite(mass).all() or np.any(mass <= 0)):
        raise RuntimeError("dense fixture iteration has an invalid normalizer or posterior mass")
    numerator = np.asarray(result.numerator)
    denominator = np.asarray(result.denominator)
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise RuntimeError("dense fixture iteration has nonfinite BPref accumulators")
    return {
        "particles": int(ids.size),
        "mass_min_max": [float(mass.min()), float(mass.max())],
        "mass_mean": float(mass.mean()),
        "logz_min_max": [float(np.min(np.sum(pair, axis=1))), float(np.max(np.sum(pair, axis=1)))],
        "numerator_norm": float(np.linalg.norm(numerator)),
        "denominator_norm": float(np.linalg.norm(denominator)),
    }


def _failure_diagnostics(result, source_ids):
    """Small JSON-safe numerical census, used only after an iteration fails."""
    if result is None:
        return None
    ids = np.asarray(source_ids, dtype=np.int32)
    pair = np.asarray(result.next_pair_table[ids])
    mass = np.asarray(result.mass_table[ids])
    finite_mass = mass[np.isfinite(mass)]
    finite_logz = np.sum(pair, axis=1)
    finite_logz = finite_logz[np.isfinite(finite_logz)]
    return {
        "particles": int(ids.size),
        "invalid_normalizer_ids": ids[np.asarray(result.invalid_normalizer_table[ids])].tolist(),
        "invalid_weight_ids": ids[np.asarray(result.invalid_weight_table[ids])].tolist(),
        "nonfinite_pair_ids": ids[~np.isfinite(pair).all(axis=1)].tolist(),
        "nonfinite_mass_ids": ids[~np.isfinite(mass)].tolist(),
        "nonpositive_mass_ids": ids[np.isfinite(mass) & (mass <= 0)].tolist(),
        "finite_mass_min_max": None if not finite_mass.size else [float(finite_mass.min()), float(finite_mass.max())],
        "finite_logz_min_max": None if not finite_logz.size else [float(finite_logz.min()), float(finite_logz.max())],
        "nonfinite_numerator_voxels": int(np.count_nonzero(~np.isfinite(np.asarray(result.numerator)))),
        "nonfinite_denominator_voxels": int(np.count_nonzero(~np.isfinite(np.asarray(result.denominator)))),
    }


def _native_fixture_program(args, geometry, bp_shape, current_size, image_shape, *, mode):
    """Canonical fused score plus native per-image-row translation and adjoint.

    This keeps the same full dense candidate grid and posterior policy as the
    GEMM arm, while using existing CUDA owners for both scorer and M-step.
    """
    from relax.cuda.kernels import (
        relion_coarse_diff2_projector_f32,
        relion_fused_x_half_backproject_particle_grid_indexed,
        relion_translate_sum_flat_rows_f32,
    )
    from relax.dense.gemm_experiment import DenseGemmBatchResult
    from relax.dense.gemm_experiment_kernels import (
        empty_normalizer_table,
        merge_normalizers,
        normalizer_logz,
        tile_normalizer,
    )

    qsize = args.rotation_tile
    count = args.translations
    exact = mode == "exact"
    lookup = jnp.asarray(
        geometry["score_layout"].full_to_compact_np[: current_size * (current_size // 2 + 1)],
        dtype=jnp.int32,
    )
    angles = jnp.asarray(geometry["angles"], dtype=jnp.float32)
    centered_indices = jnp.asarray(geometry["window"].recon_window_indices, dtype=jnp.int32)
    native_indices = jnp.asarray(geometry["window"].relion_x_half_recon_indices, dtype=jnp.int32)
    row_ids = jnp.repeat(jnp.arange(args.images, dtype=jnp.int32), qsize)
    row_count = jnp.asarray(args.images * qsize, dtype=jnp.int32)
    pixel_count = jnp.asarray(centered_indices.size, dtype=jnp.int32)

    @partial(jax.jit, donate_argnames=("numerator", "denominator", "next_pair_table"))
    def program(reference, numerator, denominator, old_pair_table, next_pair_table, batch, grid):
        nrot = grid.score_rotations.shape[0] // qsize
        old_pair = old_pair_table[batch.particle_ids]

        def scores_for(index):
            start = index * qsize
            rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, start, qsize, axis=0)
            diff2 = relion_coarse_diff2_projector_f32(
                reference, rotations, batch.score_image, angles,
                batch.score_weight, batch.initial_diff2, lookup,
                current_size=current_size, physical_image_size=image_shape[0],
                model_max_r=current_size // 2, padding_factor=2,
                canonical_reduction=True,
            )
            rprior = jax.lax.dynamic_slice_in_dim(batch.rotation_prior, start, qsize, axis=1)
            valid_rot = jax.lax.dynamic_slice_in_dim(grid.valid_rotations, start, qsize, axis=0)
            scores = -diff2 + rprior[:, :, None] + batch.translation_prior[:, None, :count]
            return jnp.where(batch.valid_images[:, None, None] & valid_rot[None, :, None], scores, -jnp.inf)

        if exact:
            normalizer = jax.lax.fori_loop(
                0, nrot,
                lambda index, pair: merge_normalizers(pair, tile_normalizer(scores_for(index))),
                empty_normalizer_table(args.images),
            )
        else:
            normalizer = old_pair

        def reconstruct(index, state):
            y_volume, w_volume, calculated_pair, mass = state
            scores = scores_for(index)
            if not exact:
                calculated_pair = merge_normalizers(calculated_pair, tile_normalizer(scores))
            weights = jnp.where(
                batch.valid_images[:, None, None],
                jnp.exp((scores - normalizer[:, 0, None, None]) - normalizer[:, 1, None, None]),
                jnp.float32(0),
            )
            data_flat, _, _, weight_flat = relion_translate_sum_flat_rows_f32(
                batch.rec_raw_image, batch.rec_raw_image, row_ids,
                weights.reshape(args.images * qsize, count), angles, centered_indices,
                row_count, pixel_count, recon_weight=batch.rec_weighted_ctf,
                ctf2_over_nv=batch.rec_weight, image_shape=image_shape,
            )
            data_rows = data_flat.reshape(args.images, qsize, centered_indices.size)
            weight_rows = weight_flat.reshape(args.images, qsize, centered_indices.size)
            bp_rotations = jax.lax.dynamic_slice_in_dim(grid.backprojection_rotations, index * qsize, qsize, axis=0)
            particle_rotations = jnp.broadcast_to(bp_rotations[None], (args.images, qsize, 3, 3))
            y_volume, w_volume = relion_fused_x_half_backproject_particle_grid_indexed(
                y_volume, w_volume, data_rows, weight_rows, native_indices,
                particle_rotations, image_shape, bp_shape, float(current_size // 2),
            )
            return y_volume, w_volume, calculated_pair, mass + jnp.sum(weights, axis=(1, 2))

        numerator, denominator, calculated_pair, mass = jax.lax.fori_loop(
            0, nrot, reconstruct,
            (numerator, denominator, empty_normalizer_table(args.images),
             jnp.zeros((args.images,), dtype=jnp.float32)),
        )
        pair = normalizer if exact else calculated_pair
        next_pair_table = next_pair_table.at[batch.particle_ids].set(pair)
        invalid_normalizer = batch.valid_images & (
            ~jnp.isfinite(pair).all(axis=1) | (~jnp.isfinite(old_pair).all(axis=1) if not exact else False)
        )
        invalid_weight = batch.valid_images & (~jnp.isfinite(mass) | (mass <= 0))
        return DenseGemmBatchResult(
            numerator, denominator, normalizer_logz(next_pair_table), next_pair_table,
            normalizer_logz(pair), mass, invalid_normalizer, invalid_weight,
        )

    return program


def _resident_cuda_fixture_program(
    args, geometry, bp_shape, current_size, image_shape, project, *, mode,
):
    """Production-selected GEMM score and SPA resident row M-step on a dense grid.

    Coarse Gaussian scoring currently selects shared GEMMs by default. The
    resident M-step keeps its CUDA translate-sum, XLA CTF-mass convention and
    chunked indexed adjoint. Only the support is changed to every R x T pose.
    """
    from relax.cuda import kernels as cuda_backproject
    from relax.cuda.kernels import relion_translate_score_f32
    from relax.dense.gemm_experiment import DenseGemmBatchResult
    from relax.dense.gemm_experiment_kernels import (
        empty_normalizer_table,
        merge_normalizers,
        normalizer_logz,
        tile_normalizer,
    )
    from relax.scoring.scoring import _relion_coarse_gaussian_gemm_scores_jit
    from relax.sparse_pass2.resident_pass2 import _resident_block_weighted_sums_kernel
    from relax.sparse_pass2.sparse_pass2_adjoint import _accumulate_adjoint_block_chunked
    from relax.sparse_pass2.sparse_pass2_budget import _max_adjoint_block_bytes_for_pass

    qsize = args.rotation_tile
    image_capacity = args.images
    n_translations = args.translations
    row_tile = int(getattr(args, "row_tile", 2048))
    if row_tile <= 0:
        raise ValueError("resident CUDA row_tile must be positive")
    row_count = image_capacity * qsize
    padded_rows = -(-row_count // row_tile) * row_tile
    n_row_tiles = padded_rows // row_tile
    max_adjoint_block_bytes = _max_adjoint_block_bytes_for_pass()
    exact = mode == "exact"
    score_indices = jnp.asarray(geometry["score_indices"], dtype=jnp.int32)
    recon_indices = jnp.asarray(geometry["window"].recon_window_indices, dtype=jnp.int32)
    native_indices = jnp.asarray(geometry["window"].relion_x_half_recon_indices, dtype=jnp.int32)
    angles = jnp.asarray(geometry["angles"], dtype=jnp.float32)
    n_recon_pixels = int(recon_indices.size)

    @partial(jax.jit, donate_argnames=("numerator", "denominator", "next_pair_table"))
    def program(reference, numerator, denominator, old_pair_table, next_pair_table, batch, grid):
        nrot = grid.score_rotations.shape[0] // qsize
        old_pair = old_pair_table[batch.particle_ids]
        actual_images = jnp.sum(batch.valid_images, dtype=jnp.int32)
        shifted = relion_translate_score_f32(
            batch.score_image, angles, score_indices, image_shape,
        ).reshape(image_capacity, n_translations, score_indices.size)

        def scores_for(index):
            start = index * qsize
            rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, start, qsize, axis=0)
            projections = project(reference, rotations)
            raw_scores = _relion_coarse_gaussian_gemm_scores_jit(
                projections, None, shifted, batch.score_weight, batch.initial_diff2,
                actual_images, n_images=image_capacity, n_trans=n_translations,
                image_shape=image_shape, volume_shape=bp_shape, float64=False,
            )
            rotation_prior = jax.lax.dynamic_slice_in_dim(batch.rotation_prior, start, qsize, axis=1)
            valid_rotations = jax.lax.dynamic_slice_in_dim(grid.valid_rotations, start, qsize, axis=0)
            scores = raw_scores + rotation_prior[:, :, None] + batch.translation_prior[:, None, :n_translations]
            return jnp.where(
                batch.valid_images[:, None, None] & valid_rotations[None, :, None],
                scores, -jnp.inf,
            )

        if exact:
            normalizer = jax.lax.fori_loop(
                0, nrot,
                lambda index, pair: merge_normalizers(pair, tile_normalizer(scores_for(index))),
                empty_normalizer_table(image_capacity),
            )
        else:
            normalizer = old_pair

        def reconstruct(index, state):
            y_volume, w_volume, calculated_pair, mass = state
            scores = scores_for(index)
            if not exact:
                calculated_pair = merge_normalizers(calculated_pair, tile_normalizer(scores))
            weights = jnp.where(
                batch.valid_images[:, None, None],
                jnp.exp((scores - normalizer[:, 0, None, None]) - normalizer[:, 1, None, None]),
                jnp.float32(0),
            )
            flat_weights = jnp.pad(
                weights.reshape(row_count, n_translations),
                ((0, padded_rows - row_count), (0, 0)),
            )
            rotations = jax.lax.dynamic_slice_in_dim(
                grid.backprojection_rotations, index * qsize, qsize, axis=0,
            )

            def accumulate_rows(row_index, volumes):
                y_acc, w_acc = volumes
                row_start = row_index * row_tile
                positions = row_start + jnp.arange(row_tile, dtype=jnp.int32)
                valid_rows = positions < row_count
                row_local = jnp.where(valid_rows, positions // qsize, 0)
                row_ids = jnp.where(valid_rows, row_local, -1)
                row_rotation = jnp.where(valid_rows, positions % qsize, 0)
                posterior_rows = jax.lax.dynamic_slice_in_dim(
                    flat_weights, row_start, row_tile, axis=0,
                )
                summed, _unused_masked, ctf_probs, _row_mass = _resident_block_weighted_sums_kernel(
                    posterior_rows, row_ids, row_local,
                    batch.rec_raw_image, batch.rec_weighted_ctf, batch.rec_raw_image,
                    batch.rec_weight, recon_indices, angles,
                    image_shape=image_shape, n_recon_pixels=n_recon_pixels,
                    kernel_ctf_probs=False, cuda_backproject=cuda_backproject,
                )
                row_rotations = rotations[row_rotation]
                adjoint_kwargs = dict(
                    window_indices=native_indices, use_windowed_adjoint=True,
                    image_shape=image_shape, volume_shape=bp_shape,
                    disc_type="linear_interp", half_image=True, half_volume=True,
                    max_r=float(current_size // 2), relion_x_half=True,
                    max_block_bytes=max_adjoint_block_bytes, log_label="dense-resident-cuda",
                )
                y_acc = _accumulate_adjoint_block_chunked(
                    summed, row_rotations, y_acc, **adjoint_kwargs,
                )
                w_acc = _accumulate_adjoint_block_chunked(
                    ctf_probs, row_rotations, w_acc, **adjoint_kwargs,
                )
                return y_acc, w_acc

            y_volume, w_volume = jax.lax.fori_loop(
                0, n_row_tiles, accumulate_rows, (y_volume, w_volume),
            )
            return (
                y_volume, w_volume, calculated_pair,
                mass + jnp.sum(weights, axis=(1, 2)),
            )

        numerator, denominator, calculated_pair, mass = jax.lax.fori_loop(
            0, nrot, reconstruct,
            (numerator, denominator, empty_normalizer_table(image_capacity),
             jnp.zeros((image_capacity,), dtype=jnp.float32)),
        )
        pair = normalizer if exact else calculated_pair
        next_pair_table = next_pair_table.at[batch.particle_ids].set(pair)
        invalid_normalizer = batch.valid_images & (
            ~jnp.isfinite(pair).all(axis=1) | (~jnp.isfinite(old_pair).all(axis=1) if not exact else False)
        )
        invalid_weight = batch.valid_images & (~jnp.isfinite(mass) | (mass <= 0))
        return DenseGemmBatchResult(
            numerator, denominator, normalizer_logz(next_pair_table), next_pair_table,
            normalizer_logz(pair), mass, invalid_normalizer, invalid_weight,
        )

    return program


def run_fixed_fixture_trajectory(args, model_dir):
    """Execute two half maps on one unchanged dense pose grid for each iteration."""
    from relax.cuda.kernels import RelionCapacityHalfTextureF32
    from relax.dense.gemm_experiment import (
        DenseGemmTileConfig,
        make_batch_program,
        native_relion_callbacks,
        run_resident_iteration,
    )
    from relax.dense.gemm_experiment_kernels import empty_normalizer_table
    from relax.helpers.half_volume_mstep import half_volume_accumulator_shape, relion_backprojector_volume_shape
    from relax.helpers.projection import relion_projector_half_to_texture_full
    from relax.reconstruction.regularization_relion import compute_relion_fsc_from_backprojector

    if jax.default_backend() != "gpu":
        raise RuntimeError("a fixed fixture trajectory requires a selected CUDA GPU")
    if args.iterations < 1 or args.half_particles <= 0 or args.images <= 0:
        raise ValueError("trajectory needs at least one iteration and positive batch/half sizes")
    dataset, halves, checkpoint, metadata, geometry = load_fixture_execution_state(
        args.data_dir, model_dir, rotations=args.rotations, translations=args.translations,
        rotation_tile=args.rotation_tile, translation_tile=args.translation_tile,
    )
    volume_shape = (dataset.image_shape[0],) * 3
    bp_shape = relion_backprojector_volume_shape(volume_shape, 2, current_size=checkpoint.current_size)
    volume_size = int(np.prod(half_volume_accumulator_shape(bp_shape)))
    source_ids = tuple(m.source_ids[: min(args.half_particles, len(m.source_ids))] for m in metadata)
    fourier_maps = list(checkpoint.fourier_maps)
    references = []
    old_tables = [empty_normalizer_table(dataset.n_units + 1) for _ in range(2)]
    owners = []
    programs = []
    refresh_seconds = [0.0, 0.0]
    projector_setup_seconds = [0.0, 0.0]
    last_completions = [None, None]
    receipts = []
    array_payload = {}
    iteration = -1
    half = -1
    stage = "setup"
    result = None
    try:
        for half in (0, 1):
            setup_start = time.perf_counter()
            slab, radius = _projector_slab(fourier_maps[half], volume_shape, checkpoint.current_size)
            if args.engine in {"gemm", "resident_cuda"}:
                references.append(slab)
                owner = RelionCapacityHalfTextureF32(slab, radius, padding_factor=2, reusable_staging=True)
                owners.append(owner)
                project, backproject = native_relion_callbacks(
                    image_shape=dataset.image_shape,
                    score_indices=geometry["score_indices"],
                    rec_indices=geometry["window"].relion_x_half_recon_indices,
                    r_max=radius, projector_output_size=dataset.image_shape[0],
                    volume_shape=bp_shape, padding_factor=2, capacity_texture=owner,
                )
                if args.engine == "gemm":
                    programs.append({
                        mode: make_batch_program(
                            DenseGemmTileConfig(args.images, args.rotation_tile, args.translation_tile,
                                                args.translation_side, mode), project, backproject,
                        ) for mode in ("exact", "lagged")
                    })
                else:
                    programs.append({
                        mode: _resident_cuda_fixture_program(
                            args, geometry, bp_shape, checkpoint.current_size,
                            dataset.image_shape, project, mode=mode,
                        ) for mode in ("exact", "lagged")
                    })
            else:
                references.append(relion_projector_half_to_texture_full(slab))
                owners.append(None)
                programs.append({
                    mode: _native_fixture_program(
                        args, geometry, bp_shape, checkpoint.current_size, dataset.image_shape, mode=mode,
                    ) for mode in ("exact", "lagged")
                })
            jax.block_until_ready(references[half])
            projector_setup_seconds[half] += time.perf_counter() - setup_start

        for iteration in range(args.iterations):
            mode = "exact" if iteration == 0 else args.mode
            half_accumulators = []
            half_maps = []
            iteration_rows = []
            for half in (0, 1):
                start = time.perf_counter()
                stage = "batch"
                result = None
                prefix = f"iteration{iteration + 1}_half{half + 1}"
                if args.save_arrays:
                    array_payload[f"{prefix}_particle_ids"] = np.asarray(source_ids[half])
                    array_payload[f"{prefix}_old_pair"] = np.asarray(old_tables[half][source_ids[half]])
                local_count = len(source_ids[half])
                batches = (
                    prepare_real_batch(
                        halves[half], checkpoint, half, metadata[half], geometry,
                        np.arange(begin, min(begin + args.images, local_count), dtype=np.int32),
                        batch_capacity=args.images, sentinel_id=dataset.n_units,
                    ) for begin in range(0, local_count, args.images)
                )
                result = run_resident_iteration(
                    programs[half][mode], references[half], batches, geometry["grid"],
                    old_tables[half], volume_size=volume_size,
                )
                last_completions[half] = result.numerator
                if args.save_arrays:
                    array_payload[f"{prefix}_pair"] = np.asarray(result.next_pair_table[source_ids[half]])
                    array_payload[f"{prefix}_mass"] = np.asarray(result.mass_table[source_ids[half]])
                    array_payload[f"{prefix}_xhalf_numerator"] = np.asarray(result.numerator)
                    array_payload[f"{prefix}_xhalf_denominator"] = np.asarray(result.denominator)
                stage = "normalizer_and_mass"
                diagnostics = _checked_particle_rows(result, source_ids[half])
                if mode == "lagged":
                    ids = source_ids[half]
                    previous_pair = np.asarray(old_tables[half][ids], dtype=np.float64)
                    next_pair = np.asarray(result.next_pair_table[ids], dtype=np.float64)
                    expected_mass = np.exp(np.sum(next_pair - previous_pair, axis=1))
                    actual_mass = np.asarray(result.mass_table[ids], dtype=np.float64)
                    diagnostics["lagged_mass_identity_max_relative_error"] = float(
                        np.max(np.abs(actual_mass - expected_mass) / np.maximum(expected_mass, np.finfo(float).tiny))
                    )
                stage = "map_update"
                next_map, numerator, denominator = _finalize_and_reconstruct(
                    result, checkpoint, half, volume_shape, bp_shape,
                )
                # This concrete map depends on the final native projector use
                # even if BPref finalization consumed its input accumulator.
                last_completions[half] = next_map
                if not np.isfinite(np.asarray(next_map)).all():
                    raise RuntimeError("dense fixture map update produced nonfinite Fourier voxels")
                half_accumulators.append((numerator, denominator))
                half_maps.append(next_map)
                if args.save_arrays:
                    array_payload[f"{prefix}_fourier_map"] = np.asarray(next_map)
                old_tables[half] = result.next_pair_table
                iteration_rows.append({
                    "half": half + 1, "mode": mode, "seconds_including_stage_and_map": time.perf_counter() - start,
                    **diagnostics,
                })
            stage = "fsc"
            fsc = compute_relion_fsc_from_backprojector(
                half_accumulators[0][0], half_accumulators[1][0],
                half_accumulators[0][1], half_accumulators[1][1], volume_shape,
                padding_factor=2, r_max=checkpoint.current_size // 2,
                accumulator_volume_shape=bp_shape, output_dtype=jnp.float32,
                full_is_hermitian=True,
            )
            fsc_host = np.asarray(fsc)
            if not np.isfinite(fsc_host).all():
                raise RuntimeError("dense fixture FSC has nonfinite shells")
            if args.save_arrays:
                array_payload[f"iteration{iteration + 1}_fsc"] = fsc_host
            receipts.append({
                "iteration": iteration + 1, "mode": mode, "halves": iteration_rows,
                "fsc": fsc_host.astype(float).tolist(),
            })
            fourier_maps = half_maps
            if iteration + 1 < args.iterations:
                stage = "projector_refresh" if args.engine != "native" else "projector_setup"
                for half in (0, 1):
                    setup_start = time.perf_counter()
                    slab, radius = _projector_slab(fourier_maps[half], volume_shape, checkpoint.current_size)
                    projector_setup_seconds[half] += time.perf_counter() - setup_start
                    if args.engine != "native":
                        refresh_seconds[half] += owners[half].refresh_after(slab, last_completions[half], logical_r_max=radius)
                        references[half] = slab
                    else:
                        setup_start = time.perf_counter()
                        references[half] = relion_projector_half_to_texture_full(slab)
                        jax.block_until_ready(references[half])
                        projector_setup_seconds[half] += time.perf_counter() - setup_start
        stage = "save_arrays"
        if args.save_arrays:
            array_path = args.output.with_suffix(".arrays.npz")
            np.savez_compressed(array_path, **array_payload)
            arrays = {"path": str(array_path), "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest()}
        else:
            arrays = None
        return {
            "schema": "relax.dense_gemm_em.fixed_fixture_trajectory.v1",
            "checkpoint": inspect_checkpoint(args.data_dir, model_dir),
            "policy": "fixed grid/noise/tau/scales/priors; exact bootstrap; no support pruning",
            "configuration": {
                "engine": args.engine, "mode": args.mode, "translation_side": args.translation_side,
                "iterations": args.iterations,
                "half_particles": [len(ids) for ids in source_ids], "batch_capacity": args.images,
                "rotations": args.rotations, "translations": args.translations,
                "rotation_tile": args.rotation_tile, "translation_tile": args.translation_tile,
                "row_tile": args.row_tile if args.engine == "resident_cuda" else None,
                "score_owner": (
                    "production_selected_coarse_gemm"
                    if args.engine == "resident_cuda" else
                    "dense_experiment_gemm" if args.engine == "gemm" else
                    "historical_fused_projector"
                ),
                "mstep_owner": (
                    "spa_resident_cuda_chunked_rows"
                    if args.engine == "resident_cuda" else
                    "collapsed_rotation_slices" if args.engine == "gemm" else
                    "historical_particle_grid"
                ),
                "score_pixels": int(geometry["score_indices"].size),
                "recon_pixels": int(geometry["window"].relion_x_half_recon_indices.size),
                "bp_shape": list(bp_shape), "volume_size": volume_size,
            },
            "refresh_seconds": refresh_seconds,
            "projector_setup_seconds": projector_setup_seconds,
            "iterations": receipts, "arrays": arrays,
        }
    except Exception as exc:
        failure = {
            "schema": "relax.dense_gemm_em.fixed_fixture_failure.v1",
            "iteration": None if iteration < 0 else iteration + 1,
            "half": None if half < 0 else half + 1,
            "stage": stage,
            "mode": None if iteration < 0 else ("exact" if iteration == 0 else args.mode),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_iterations": receipts,
        }
        try:
            failure["numerical_diagnostics"] = _failure_diagnostics(result, source_ids[half]) if half >= 0 else None
        except Exception as diagnostic_exc:
            failure["diagnostic_error"] = str(diagnostic_exc)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.save_arrays and array_payload:
            try:
                array_path = args.output.with_suffix(".arrays.npz")
                np.savez_compressed(array_path, **array_payload)
                failure["arrays"] = {
                    "path": str(array_path), "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                }
            except Exception as save_exc:
                failure["arrays_error"] = str(save_exc)
        args.output.write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        for half, owner in enumerate(owners):
            if owner is not None and not owner.closed:
                if last_completions[half] is not None:
                    owner.close_after(last_completions[half])
                else:
                    owner.close()


def benchmark_fixed_fixture_batch(args, model_dir):
    """Time one warmed, preassembled real batch with reusable device volumes.

    Preprocessing, projector staging, exact bootstrap, compilation and map
    reconstruction are recorded separately from repeated batch operator time.
    The frozen map/operands isolate the native and GEMM operator comparison;
    trajectory reports remain responsible for changing maps across iterations.
    """
    from relax.cuda.kernels import RelionCapacityHalfTextureF32
    from relax.dense.gemm_experiment import DenseGemmTileConfig, make_batch_program, native_relion_callbacks
    from relax.dense.gemm_experiment_kernels import empty_normalizer_table
    from relax.helpers.half_volume_mstep import half_volume_accumulator_shape, relion_backprojector_volume_shape
    from relax.helpers.projection import relion_projector_half_to_texture_full
    benchmark_module = importlib.import_module(
        "scripts.benchmark_dense_gemm_em" if __package__ else "benchmark_dense_gemm_em"
    )
    _compiled_memory = benchmark_module._compiled_memory
    _memory_stats = benchmark_module._memory_stats
    _native_identity = benchmark_module._native_identity
    _profiler_boundary = benchmark_module._profiler_boundary
    _reusable_program = benchmark_module._reusable_program
    _volume_aliases = benchmark_module._volume_aliases

    if jax.default_backend() != "gpu" or args.images <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("batch benchmark needs CUDA, positive B/repeats and nonnegative warmup")
    started = time.perf_counter()
    dataset, halves, checkpoint, metadata, geometry = load_fixture_execution_state(
        args.data_dir, model_dir, rotations=args.rotations, translations=args.translations,
        rotation_tile=args.rotation_tile, translation_tile=args.translation_tile,
    )
    setup_seconds = time.perf_counter() - started
    half = args.half - 1
    if args.images > halves[half].n_units:
        raise ValueError("batch capacity exceeds fixture half particle count")
    volume_shape = (dataset.image_shape[0],) * 3
    bp_shape = relion_backprojector_volume_shape(volume_shape, 2, current_size=checkpoint.current_size)
    volume_size = int(np.prod(half_volume_accumulator_shape(bp_shape)))
    batch_start = time.perf_counter()
    batch = prepare_real_batch(
        halves[half], checkpoint, half, metadata[half], geometry,
        np.arange(args.images, dtype=np.int32), batch_capacity=args.images,
        sentinel_id=dataset.n_units,
    )
    jax.block_until_ready(batch.score_image)
    jax.block_until_ready(batch.rec_image)
    batch_preprocess_seconds = time.perf_counter() - batch_start
    projector_start = time.perf_counter()
    slab, radius = _projector_slab(checkpoint.fourier_maps[half], volume_shape, checkpoint.current_size)
    texture = None
    if args.engine in {"gemm", "resident_cuda"}:
        texture = RelionCapacityHalfTextureF32(slab, radius, padding_factor=2, reusable_staging=True)
        reference = slab
        project, backproject = native_relion_callbacks(
            image_shape=dataset.image_shape, score_indices=geometry["score_indices"],
            rec_indices=geometry["window"].relion_x_half_recon_indices,
            r_max=radius, projector_output_size=dataset.image_shape[0],
            volume_shape=bp_shape, padding_factor=2, capacity_texture=texture,
        )

        if args.engine == "gemm":
            def make_program(mode):
                return make_batch_program(
                    DenseGemmTileConfig(args.images, args.rotation_tile, args.translation_tile,
                                        args.translation_side, mode), project, backproject,
                )
        else:
            def make_program(mode):
                return _resident_cuda_fixture_program(
                    args, geometry, bp_shape, checkpoint.current_size,
                    dataset.image_shape, project, mode=mode,
                )

    else:
        reference = relion_projector_half_to_texture_full(slab)

        def make_program(mode):
            return _native_fixture_program(
                args, geometry, bp_shape, checkpoint.current_size, dataset.image_shape, mode=mode,
            )

    jax.block_until_ready(reference)
    projector_setup_seconds = time.perf_counter() - projector_start

    last_completion = None
    try:
        table_size = dataset.n_units + 1
        empty = empty_normalizer_table(table_size)
        zeros_y = jnp.zeros((volume_size,), dtype=jnp.complex64)
        zeros_w = jnp.zeros((volume_size,), dtype=jnp.float32)
        bootstrap_start = time.perf_counter()
        bootstrap = make_program("exact")(
            reference, zeros_y, zeros_w, empty, empty_normalizer_table(table_size),
            batch, geometry["grid"],
        )
        jax.block_until_ready(bootstrap.numerator)
        bootstrap_seconds = time.perf_counter() - bootstrap_start
        last_completion = bootstrap.numerator
        old_table = bootstrap.next_pair_table if args.mode == "lagged" else empty
        reset = _reusable_program(make_program(args.mode))
        scratch_y = jnp.zeros((volume_size,), dtype=jnp.complex64)
        scratch_w = jnp.zeros((volume_size,), dtype=jnp.float32)
        scratch_next = empty_normalizer_table(table_size)
        compile_start = time.perf_counter()
        compiled = reset.lower(
            reference, scratch_y, scratch_w, old_table, scratch_next, batch, geometry["grid"],
        ).compile()
        compile_seconds = time.perf_counter() - compile_start
        hlo = compiled.as_text()
        memory = _compiled_memory(compiled)
        aliases = _volume_aliases(hlo, compiled, volume_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".hlo.txt").write_text(hlo)
        if not aliases["verified"]:
            raise RuntimeError(f"warmed batch executable lacks both volume aliases: {aliases}")

        def invoke():
            nonlocal scratch_y, scratch_w, scratch_next, last_completion
            output = compiled(
                reference, scratch_y, scratch_w, old_table, scratch_next, batch, geometry["grid"],
            )
            jax.block_until_ready(output.numerator)
            jax.block_until_ready(output.denominator)
            scratch_y, scratch_w, scratch_next = output.numerator, output.denominator, output.next_pair_table
            last_completion = output.numerator
            return output

        for _ in range(args.warmup):
            invoke()
        device = jax.devices("gpu")[0]
        memory_before = _memory_stats(device)
        durations = []
        if args.profile_warm_repeats:
            _profiler_boundary(True)
        try:
            for _ in range(args.repeats):
                begin = time.perf_counter()
                result = invoke()
                durations.append(time.perf_counter() - begin)
        finally:
            if args.profile_warm_repeats:
                _profiler_boundary(False)
        memory_after = _memory_stats(device)
        ids = metadata[half].source_ids[:args.images]
        diagnostics = _checked_particle_rows(
            type("IterationRows", (), {
                "next_pair_table": result.next_pair_table,
                "mass_table": jnp.zeros((table_size,), jnp.float32).at[ids].set(result.batch_mass),
                "invalid_normalizer_table": jnp.zeros((table_size,), jnp.bool_).at[ids].set(result.invalid_normalizer),
                "invalid_weight_table": jnp.zeros((table_size,), jnp.bool_).at[ids].set(result.invalid_weight),
                "numerator": result.numerator, "denominator": result.denominator,
            })(), ids,
        )
        arrays = None
        if args.save_arrays:
            array_path = args.output.with_suffix(".arrays.npz")
            np.savez_compressed(
                array_path, particle_ids=ids,
                pair=np.asarray(result.next_pair_table[ids]), mass=np.asarray(result.batch_mass),
                numerator=np.asarray(result.numerator), denominator=np.asarray(result.denominator),
            )
            arrays = {"path": str(array_path), "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest()}
        return {
            "schema": "relax.dense_gemm_em.fixed_fixture_batch_benchmark.v1",
            "configuration": {
                "engine": args.engine, "mode": args.mode, "half": args.half,
                "translation_side": args.translation_side, "batch_capacity": args.images,
                "rotations": args.rotations, "translations": args.translations,
                "rotation_tile": args.rotation_tile, "translation_tile": args.translation_tile,
                "score_pixels": int(batch.score_image.shape[1]),
                "recon_pixels": int(batch.rec_image.shape[1]),
                "candidate_count": args.images * args.rotations * args.translations,
                "exact_score_sweeps": 2 if args.mode == "exact" else 1,
                "native_per_image_row_adjoint": args.engine == "native",
                "score_owner": (
                    "production_selected_coarse_gemm"
                    if args.engine == "resident_cuda" else
                    "dense_experiment_gemm" if args.engine == "gemm" else
                    "historical_fused_projector"
                ),
                "mstep_owner": (
                    "spa_resident_cuda_chunked_rows"
                    if args.engine == "resident_cuda" else
                    "collapsed_rotation_slices" if args.engine == "gemm" else
                    "historical_particle_grid"
                ),
                "row_tile": args.row_tile if args.engine == "resident_cuda" else None,
            },
            "source_sha256": checkpoint.source_sha256, "native": _native_identity(),
            "dataset_and_grid_setup_seconds": setup_seconds,
            "batch_preprocess_seconds": batch_preprocess_seconds,
            "projector_setup_seconds": projector_setup_seconds,
            "bootstrap_seconds": bootstrap_seconds,
            "compile_seconds": compile_seconds,
            "warm_repeats_seconds": durations,
            "warm_median_seconds": float(np.median(durations)),
            "compiled_memory": memory, "volume_aliases": aliases,
            "device_memory_before": memory_before, "device_memory_after": memory_after,
            "diagnostics": diagnostics, "arrays": arrays,
        }
    finally:
        if texture is not None and not texture.closed:
            if last_completion is not None:
                texture.close_after(last_completion)
            else:
                texture.close()

def inspect_checkpoint(data_dir: Path, model_dir: Path) -> dict:
    """CPU-only fixture/state receipt before scheduling a GPU trajectory."""
    data_star = model_dir / "run_it001_data.star"
    ds = load_dataset(str(data_star), datadir=str(data_dir), lazy=True,
                      dtype=np.complex64, absent_angles_zero=True)
    checkpoint = load_fixed_gaussian_checkpoint(data_dir, model_dir, ds.image_shape)
    table = starfile.read(data_star)["particles"]
    if len(table) != ds.n_units or not table["rlnImageName"].is_unique:
        raise ValueError("checkpoint particle identities are not unique or complete")
    if not np.array_equal(ds.dataset_indices, np.arange(ds.n_units)):
        raise ValueError("fixture loader changed STAR row order; remap particles by rlnImageName")
    ds.halfset_indices = tuple(
        np.flatnonzero(np.asarray(table["rlnRandomSubset"], dtype=np.int32) == h).astype(np.int32)
        for h in (1, 2)
    )
    half_ids = tuple(np.asarray(ds.halfset_original_image_indices(h), dtype=np.int32) for h in (0, 1))
    if sum(len(ids) for ids in half_ids) != ds.n_units or np.unique(np.concatenate(half_ids)).size != ds.n_units:
        raise ValueError("fixture halfsets do not partition stable particle IDs")
    return {
        "schema": "relax.dense_gemm_em.fixture_checkpoint.v1",
        "kind": "RELION iteration-1 half-specific Gaussian checkpoint, fixed noise and tau",
        "data_dir": str(data_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "particle_count": int(ds.n_units),
        "half_counts": [int(len(ids)) for ids in half_ids],
        "image_shape": list(ds.image_shape),
        "voxel_size": float(ds.voxel_size),
        "current_size": checkpoint.current_size,
        "tau2_fudge": checkpoint.tau2_fudge,
        "noise_min_max": [[float(np.min(v)), float(np.max(v))] for v in checkpoint.noise_radial],
        "tau_min_max": [[float(np.min(v)), float(np.max(v))] for v in checkpoint.tau_radial],
        "source_sha256": checkpoint.source_sha256,
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-checkpoint", action="store_true")
    parser.add_argument("--probe-operands", action="store_true")
    parser.add_argument("--benchmark-batch", action="store_true")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--engine", choices=("gemm", "native", "resident_cuda"), default="gemm")
    parser.add_argument("--mode", choices=("exact", "lagged"), default="exact")
    parser.add_argument("--translation-side", choices=("image", "projection"), default="image")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--half-particles", type=int, default=16)
    parser.add_argument("--half", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-warm-repeats", action="store_true")
    parser.add_argument("--images", type=int, default=2)
    parser.add_argument("--rotations", type=int, default=8)
    parser.add_argument("--translations", type=int, default=3)
    parser.add_argument("--rotation-tile", type=int, default=4)
    parser.add_argument("--translation-tile", type=int, default=1)
    parser.add_argument("--row-tile", type=int, default=2048)
    return parser.parse_args()


def main():
    args = arguments()
    model_dir = args.model_dir or args.data_dir / "relion_ref_os0"
    start = time.perf_counter()
    if args.inspect_checkpoint:
        report = inspect_checkpoint(args.data_dir, model_dir)
    elif args.probe_operands:
        report = probe_real_operands(args, model_dir)
    elif args.benchmark_batch:
        report = benchmark_fixed_fixture_batch(args, model_dir)
    else:
        report = run_fixed_fixture_trajectory(args, model_dir)
    report["inspect_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "schema": report["schema"]}))


if __name__ == "__main__":
    main()
