"""Diagnostic capture policy and writers for dense single-volume refinement.

Extracted from ``iteration_loop.py``:

- ``_maybe_dump_noise_update_debug`` writes per-iteration RELION-parity
  noise-update sufficient statistics when ``RELAX_NOISE_DEBUG_DUMP_DIR``
  is set.
- ``_save_iteration_intermediates`` writes per-iteration regularized and
  unregularized volumes, FSC, noise, tau2, hard assignments, and metadata
  when ``--save_intermediates_dir`` is provided.
- Half selectors validate terminating significance/noise captures and explicit
  numbered-half device captures before the controller dispatches scoring.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from relax.diagnostics import parity_dump as _parity_dump
from relax.helpers.env_flags import parse_int_set
from relax.helpers.half_spectrum import make_half_image_weights, make_shell_indices_half
from relax.relion.geometry import PROJECTION_PADDING_FACTOR, RECONSTRUCTION_PADDING_FACTOR
from relax.relion.relion_metadata import _relion_half_plane_shell_counts
from relax.sampling import rotation_grid_size
from relax.symmetry import canonicalize_rotational_symmetry, symmetry_operator_sha256

if TYPE_CHECKING:
    from relax.helpers.convergence import RefinementState
    from relax.refinement.half_inputs import HalfSet
    from relax.refinement.numbered_reconstruction import ReconstructionSettings
    from relax.refinement.particle_poses import ParticlePoses
    from relax.refinement.ports import DenseHalfScored
    from relax.relion.relion_normalization import NormScaleCorrectionReport

logger = logging.getLogger(__name__)


_SIGNIFICANCE_DUMP_TARGET_HALF_ENV = "RELAX_SIGNIFICANCE_DUMP_TARGET_HALF"
_PASS2_NORM_DUMP_TARGET_HALF_ENV = "RELAX_PASS2_DUMP_TARGET_HALF"


def dump_numbered_iteration(
    iteration: int,
    *,
    init_relion_iteration: int,
    state: RefinementState,
    current_size: int,
    sigma_offset_angstrom,
    random_perturbation,
    settings: ReconstructionSettings,
    pixel_size_angstrom,
    ave_pmax,
    fsc,
    noise_variance,
    means,
    unfiltered_means,
    poses: tuple[ParticlePoses, ParticlePoses],
    half_inputs: tuple[HalfSet, HalfSet],
    corrections: NormScaleCorrectionReport,
    scale_correction_data_vs_prior,
    log,
) -> None:
    """Adapt completed-iteration operands to the parity capture schema.

    Capture admission and timing-only selection remain with the controller.
    See ``docs/math/relion_refinement_algorithm.md#diagnostic-capture``.
    """
    try:
        _parity_dump.dump_iteration(
            iteration=iteration,
            init_relion_iteration=int(init_relion_iteration),
            current_size=int(current_size),
            sigma_offset=float(sigma_offset_angstrom),
            translation_step=float(state.translation_step),
            translation_range=float(state.translation_range),
            random_perturbation=float(random_perturbation) if random_perturbation is not None else 0.0,
            random_perturbation_instance=int(state.perturbation_instance)
            if hasattr(state, "perturbation_instance")
            else 0,
            tau2_fudge=float(settings.tau2_fudge),
            voxel_size=pixel_size_angstrom,
            box_size=int(settings.box_size),
            volume_shape=tuple(settings.volume_shape),
            ave_pmax=float(ave_pmax),
            fsc=np.asarray(fsc, dtype=np.float64),
            sigma2_noise=np.asarray(noise_variance, dtype=np.float64),
            means=means,
            unreg_means=unfiltered_means,
            new_iter_best_rotation_eulers=[pose.eulers_deg for pose in poses],
            new_iter_best_translations=[pose.translations_pixels for pose in poses],
            image_corrections=[half.image_corrections for half in half_inputs],
            scale_corrections=[half.scale_corrections for half in half_inputs],
            group_ids=[half.group_ids for half in half_inputs],
            group_counts=[half.group_count for half in half_inputs],
            group_scale_corrections=corrections.group_scale_corrections_per_half,
            norm_corrections=corrections.norm_corrections_per_half,
            avg_norm_corrections=corrections.avg_norm_correction_per_half,
            zero_norm_residual_counts=corrections.zero_norm_residual_counts,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
        )
    except Exception as exc:
        log.warning("parity_dump.dump_iteration failed at iter %d: %s", iteration, exc)


def _significance_dump_half_indices(
    *,
    numbered_iteration: int,
    n_classes: int,
    experiment_datasets,
    environ=None,
) -> tuple[int, ...]:
    """Select one half only at an explicitly terminating diagnostic boundary."""

    env = os.environ if environ is None else environ
    raw_significance_half = str(env.get(_SIGNIFICANCE_DUMP_TARGET_HALF_ENV, "")).strip()
    raw_pass2_half = str(env.get(_PASS2_NORM_DUMP_TARGET_HALF_ENV, "")).strip()
    if raw_significance_half and raw_pass2_half:
        raise RuntimeError(
            f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} and "
            f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} are mutually exclusive"
        )
    if not raw_significance_half and not raw_pass2_half:
        return (0, 1)
    pass2_norm_mode = bool(raw_pass2_half)
    target_half_env = (
        _PASS2_NORM_DUMP_TARGET_HALF_ENV
        if pass2_norm_mode
        else _SIGNIFICANCE_DUMP_TARGET_HALF_ENV
    )
    raw_half = raw_pass2_half if pass2_norm_mode else raw_significance_half
    if pass2_norm_mode:
        if str(env.get("RELAX_PASS2_DUMP_NORM_RESIDUAL_INPUTS", "")).strip() != "1":
            raise RuntimeError(
                f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} requires "
                "RELAX_PASS2_DUMP_NORM_RESIDUAL_INPUTS=1"
            )
        if str(env.get("RELAX_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET", "")).strip() != "1":
            raise RuntimeError(
                f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} requires "
                "RELAX_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET=1"
            )
    elif str(env.get("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "")).strip() != "1":
        raise RuntimeError(
            f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} requires "
            "RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1"
        )
    if int(n_classes) != 1:
        raise RuntimeError(f"{target_half_env} is K=1 diagnostic-only")
    try:
        target_half = int(raw_half)
    except ValueError as exc:
        raise ValueError(f"{target_half_env} must be 1 or 2") from exc
    if target_half not in {1, 2}:
        raise ValueError(f"{target_half_env} must be 1 or 2")

    prefix = "RELAX_PASS2_DUMP" if pass2_norm_mode else "RELAX_SIGNIFICANCE_DUMP"
    raw_iteration = str(env.get(f"{prefix}_ITERATION", "")).strip()
    raw_targets = str(env.get(f"{prefix}_ORIGINAL_INDICES", "")).strip()
    dump_dir = str(env.get(f"{prefix}_DIR", "")).strip()
    if not raw_iteration or not raw_targets or not dump_dir:
        raise RuntimeError(
            f"{target_half_env} requires an explicit dump directory, "
            "iteration, and original-index target set"
        )
    try:
        target_iteration = int(raw_iteration)
        target_indices = {
            int(token) for token in raw_targets.replace(",", " ").split()
        }
    except ValueError as exc:
        raise ValueError("significance dump iteration and original indices must be integers") from exc
    if target_iteration <= 0 or not target_indices:
        raise ValueError("significance dump iteration and original-index target set must be nonempty")
    if int(numbered_iteration) != target_iteration:
        return (0, 1)

    selected_half_indices = set(
        np.asarray(experiment_datasets[target_half - 1].dataset_indices, dtype=np.int64).tolist()
    )
    missing = sorted(target_indices - selected_half_indices)
    if missing:
        raise RuntimeError(
            "significance dump targets are not all present in the selected half: "
            f"half={target_half} missing={missing}"
        )
    logger.info(
        "RECOVAR %s diagnostic: iteration=%d entering only half=%d "
        "for %d target particles; completion is mandatory before returning from the half",
        "pass-2 norm operand" if pass2_norm_mode else "coarse-significance",
        int(numbered_iteration),
        target_half,
        len(target_indices),
    )
    return (target_half - 1,)


def _replay_manifest_array(value, dtype=None):
    """Replay manifests use a float64 empty sentinel regardless of field dtype."""
    return np.array([]) if value is None else np.asarray(value, dtype=dtype)


def write_numbered_half_manifest(save_dir, scored: DenseHalfScored) -> None:
    """Write ``manifest_iter<i>_half<k>.npz``: the operands one half of a numbered single-pass dense iteration
    was scored with, for a deterministic replay.

    Reads ``scored.grid`` (the iteration's trial grid), ``sampling.coarse_scoring_rotations`` and
    ``random_perturbation``, the half's direction and translation log priors, ``particles`` (corrections),
    ``half`` (reference, prior and noise) and the engine's per-image maximum posterior.
    """
    from relax.dense import scoring_policy

    (iteration, half_index, grid, sampling, direction_priors, translation_log_prior, particles,
     translation_search_base, previous_translations, half, image_window_size, perturb_factor, result) = scored
    _manifest_path = os.path.join(save_dir, f"manifest_iter{iteration}_half{half_index}.npz")
    _manifest = {
        "effective_rotations": np.asarray(grid.rotations),
        "coarse_scoring_rotations": _replay_manifest_array(
            sampling.coarse_scoring_rotations,
        ),
        "current_translations": np.asarray(grid.translations),
        "rotation_log_prior": _replay_manifest_array(direction_priors.rotation_log_prior, dtype=np.float64),
        "translation_log_prior": _replay_manifest_array(translation_log_prior, dtype=np.float64),
        "image_corrections": _replay_manifest_array(
            particles.image_corrections, dtype=np.float64,
        ),
        "scale_corrections": _replay_manifest_array(
            particles.scale_corrections, dtype=np.float64,
        ),
        "image_pre_shifts": _replay_manifest_array(translation_search_base, dtype=np.float32),
        "absolute_previous_translations": _replay_manifest_array(
            previous_translations, dtype=np.float32,
        ),
        "mean_vol_ft": np.asarray(half.reference),
        "mean_variance": np.asarray(half.mean_variance),
        "noise_variance": np.asarray(half.noise_variance),
        "current_size": np.int32(image_window_size) if image_window_size is not None else np.int32(-1),
        "half_spectrum_scoring": np.bool_(True),
        "use_float64_scoring": np.bool_(scoring_policy.DENSE_PRECISION.use_float64_scoring),
        "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
        "reconstruction_padding_factor": np.int32(RECONSTRUCTION_PADDING_FACTOR),
        "score_with_masked_images": np.bool_(True),
        "perturbation_instance": np.float64(sampling.random_perturbation),
        "perturbation_factor": np.float64(perturb_factor),
        "iteration": np.int32(iteration),
        "half_index": np.int32(half_index),
        "ave_Pmax": np.float64(float(np.mean(result.em_stats.max_posterior_per_image))),
    }
    np.savez(_manifest_path, **_manifest)
    logger.info("Manifest dumped: %s", _manifest_path)


def write_final_half_manifest(
    save_dir,
    half,
    final_sampling,
    final_inputs,
    *,
    translation_search_base,
    reference,
    reference_model,
    noise_variance,
    current_size,
    precision,
    use_local: bool,
    log,
) -> None:
    """Write ``manifest_final_half<k>.npz``: the operands one half was scored with in the final all-data pass.

    Reads from ``half``: its index, corrections and previous translations; from ``final_sampling``: the
    trial grid and the perturbation settings; from ``final_inputs`` (the half's ``PreparedFinalHalf``): the
    direction and translation log priors and the engine prior centre; ``reference_model.tau2``;
    ``precision.use_float64_scoring`` and ``use_float64_projections``.
    """
    _manifest_path = os.path.join(save_dir, f"manifest_final_half{half.index}.npz")
    _manifest = {
        "effective_rotations": np.asarray(final_sampling.grid.rotations, dtype=np.float32),
        "current_translations": np.asarray(final_sampling.grid.translations, dtype=np.float32),
        "rotation_log_prior": _replay_manifest_array(final_inputs.directions.rotation_log_prior, dtype=np.float64),
        "translation_log_prior": np.asarray(final_inputs.translation_log_prior, dtype=np.float64),
        "translation_prior_centers": np.asarray(final_inputs.translations.engine_prior_center, dtype=np.float64),
        "image_corrections": _replay_manifest_array(half.image_corrections, dtype=np.float64),
        "scale_corrections": _replay_manifest_array(half.scale_corrections, dtype=np.float64),
        "image_pre_shifts": _replay_manifest_array(translation_search_base, dtype=np.float32),
        "absolute_previous_translations": _replay_manifest_array(
            half.translations, dtype=np.float32,
        ),
        "mean_vol_ft": np.asarray(reference),
        "mean_variance": np.asarray(reference_model.tau2),
        "noise_variance": np.asarray(noise_variance),
        "current_size": np.int32(current_size),
        "half_spectrum_scoring": np.bool_(True),
        "use_float64_scoring": np.bool_(precision.use_float64_scoring),
        "use_float64_projections": np.bool_(precision.use_float64_projections),
        "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
        "reconstruction_padding_factor": np.int32(RECONSTRUCTION_PADDING_FACTOR),
        "score_with_masked_images": np.bool_(True),
        "perturbation_instance": np.float64(final_sampling.settings.random_perturbation),
        "perturbation_factor": np.float64(final_sampling.settings.perturbation_factor),
        "perturbation_applied": np.bool_(final_sampling.settings.perturbation is not None),
        "perturbation_relion_iteration": np.int32(final_sampling.settings.relion_iteration),
        "local_search": np.bool_(use_local),
        "iteration": np.int32(-1),
        "half_index": np.int32(half.index),
    }
    np.savez(_manifest_path, **_manifest)
    log.info("Final manifest dumped: %s", _manifest_path)


def _dump_array_or_empty(arr):
    if arr is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(arr)


def _source_image_indices(dataset) -> np.ndarray:
    """Original-stack image indices of a half dataset, in its local order."""

    index_layout = getattr(dataset, "_index_layout", None)
    if index_layout is None:
        return np.arange(dataset.n_images, dtype=np.int64)
    return np.asarray(
        index_layout.original_image_indices_for_local(np.arange(dataset.n_images, dtype=np.int32)),
        dtype=np.int64,
    )


def _save_iteration_particle_states(
    save_dir: str,
    poses,
    per_half,
    significance,
    *,
    iteration: int,
    original_image_indices_per_half,
) -> None:
    """Write source-aligned resolved particle state for one sealed iteration.

    Reads from ``poses`` (each half's ``ParticlePoses``): the rotations, Euler angles and relative and
    absolute translations; from ``per_half`` (the iteration's ``PerHalfOutputs``): ``max_posterior``,
    ``hard_assignments`` and ``coarse_ha``; from ``significance``: the per-half significant-sample counts.
    """

    os.makedirs(save_dir, exist_ok=True)
    per_half_fields = {
        "rotation_matrices": [half_poses.rotations for half_poses in poses],
        "rotation_eulers_deg": [half_poses.eulers_deg for half_poses in poses],
        "relative_translations_pixels": [half_poses.relative_translations_pixels for half_poses in poses],
        "absolute_translations_pixels": [half_poses.translations_pixels for half_poses in poses],
        "max_posterior": per_half.max_posterior,
        "significant_counts": significance.per_half,
        "fine_hard_assignment": per_half.hard_assignments,
        "coarse_hard_assignment": per_half.coarse_ha,
        "original_image_indices": original_image_indices_per_half,
    }
    for half_index in range(2):
        payload = {
            name: _dump_array_or_empty(values[half_index])
            for name, values in per_half_fields.items()
        }
        populated_lengths = {
            name: int(value.shape[0])
            for name, value in payload.items()
            if value.ndim > 0 and value.size > 0
        }
        if populated_lengths and len(set(populated_lengths.values())) != 1:
            raise ValueError(
                "particle-state fields are not source-aligned for "
                f"half {half_index + 1}: {populated_lengths}"
            )
        particle_count = next(iter(populated_lengths.values()), 0)
        payload["half_local_indices"] = np.arange(particle_count, dtype=np.int64)
        payload["zero_based_iteration"] = np.asarray([iteration], dtype=np.int32)
        payload["one_based_iteration"] = np.asarray([iteration + 1], dtype=np.int32)
        payload["half"] = np.asarray([half_index + 1], dtype=np.int32)
        output_path = os.path.join(
            save_dir,
            f"it{iteration:03d}_particle_state_half{half_index + 1}.npz",
        )
        np.savez(output_path, **payload)
        logger.info("Saved particle-state diagnostics to %s", output_path)


def _save_iteration_intermediates(
    save_dir: str,
    numerators,
    denominators,
    reference_model,
    noise_model,
    per_half,
    trial_grid,
    sampling_plan,
    options,
    *,
    iteration: int,
    unreg_means,
    fsc,
    cs: int,
    state,
    volume_shape,
    voxel_size: float,
) -> None:
    """Write per-iteration intermediate volumes + diagnostics to ``save_dir``.

    ``numerators`` and ``denominators`` are the two halves' M-step accumulators as the reconstruction used
    them. Reads ``reference_model.maps`` and ``tau2``; ``noise_model.average_variance`` and
    ``variance_per_half``; ``per_half.hard_assignments`` and ``coarse_ha``; ``trial_grid.rotations`` and
    ``translations``; ``sampling_plan.local`` (its search order, when the search is local);
    ``state.healpix_order`` and ``sigma_rot``; ``options.k_class.n_classes`` and ``symmetry.point_group``.
    """
    from relax.helpers.map_io import write_map_from_ft

    Ft_y_0, Ft_y_1 = numerators
    Ft_ctf_0, Ft_ctf_1 = denominators
    means = reference_model.maps
    mean_variance = reference_model.tau2
    noise_variance = noise_model.average_variance
    noise_variance_per_half = noise_model.variance_per_half
    hard_assignments = per_half.hard_assignments
    coarse_ha = per_half.coarse_ha
    effective_rotations = trial_grid.rotations
    current_translations = trial_grid.translations
    use_local = sampling_plan.local is not None
    local_search_order = sampling_plan.local.search.healpix_order if use_local else None
    symmetry = options.symmetry.point_group
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"it{iteration:03d}_Ft_y_0.npy"), _dump_array_or_empty(Ft_y_0))
    np.save(os.path.join(save_dir, f"it{iteration:03d}_Ft_y_1.npy"), _dump_array_or_empty(Ft_y_1))
    np.save(os.path.join(save_dir, f"it{iteration:03d}_Ft_ctf_0.npy"), _dump_array_or_empty(Ft_ctf_0))
    np.save(os.path.join(save_dir, f"it{iteration:03d}_Ft_ctf_1.npy"), _dump_array_or_empty(Ft_ctf_1))
    for k_half in range(2):
        class_indices_to_save = range(options.k_class.n_classes) if options.k_class.n_classes > 1 else (None,)
        for class_idx in class_indices_to_save:
            suffix = f"_class{class_idx + 1}" if class_idx is not None else ""
            mean_to_save = means[k_half][class_idx] if class_idx is not None else means[k_half]
            write_map_from_ft(
                os.path.join(save_dir, f"it{iteration:03d}_half{k_half + 1}{suffix}_reg.mrc"),
                np.asarray(mean_to_save).reshape(-1),
                volume_shape,
                voxel_size=voxel_size,
            )
            if unreg_means[k_half] is not None:
                unreg_to_save = unreg_means[k_half][class_idx] if class_idx is not None else unreg_means[k_half]
                write_map_from_ft(
                    os.path.join(save_dir, f"it{iteration:03d}_half{k_half + 1}{suffix}_unreg.mrc"),
                    np.asarray(unreg_to_save).reshape(-1),
                    volume_shape,
                    voxel_size=voxel_size,
                )
    np.save(
        os.path.join(save_dir, f"it{iteration:03d}_fsc.npy"),
        np.asarray(fsc) if fsc is not None else np.array([], dtype=np.float32),
    )
    np.save(os.path.join(save_dir, f"it{iteration:03d}_noise.npy"), np.asarray(noise_variance))
    for k_half, noise_k in enumerate(noise_variance_per_half):
        np.save(
            os.path.join(save_dir, f"it{iteration:03d}_noise_half{k_half + 1}.npy"),
            np.asarray(noise_k),
        )
    np.save(os.path.join(save_dir, f"it{iteration:03d}_tau2.npy"), np.asarray(mean_variance))
    for k_half in range(2):
        if hard_assignments[k_half] is not None:
            np.save(
                os.path.join(save_dir, f"it{iteration:03d}_ha_half{k_half + 1}.npy"),
                hard_assignments[k_half],
            )
    canonical_symmetry = canonicalize_rotational_symmetry(symmetry)
    if use_local and canonical_symmetry == "C1":
        n_rotations = rotation_grid_size(local_search_order)
    elif use_local:
        n_rotations = rotation_grid_size(local_search_order, canonical_symmetry)
    else:
        n_rotations = effective_rotations.shape[0]
    iter_meta = {
        "iteration": iteration,
        "current_size": int(cs),
        "n_rotations": int(n_rotations),
        "n_translations": int(current_translations.shape[0]),
        "healpix_order": int(state.healpix_order),
        "local_search": bool(use_local),
        "sigma_rot": float(state.sigma_rot),
        "symmetry_label": canonical_symmetry,
        "symmetry_operator_sha256": symmetry_operator_sha256(canonical_symmetry),
    }
    np.save(os.path.join(save_dir, f"it{iteration:03d}_meta.npy"), iter_meta)
    np.save(
        os.path.join(save_dir, f"it{iteration:03d}_rotations.npy"),
        (np.asarray(effective_rotations) if not use_local else np.empty((0, 3, 3), dtype=np.float32)),
    )
    np.save(
        os.path.join(save_dir, f"it{iteration:03d}_translations.npy"),
        np.asarray(current_translations),
    )
    for k_half in range(2):
        if coarse_ha[k_half] is not None:
            np.save(
                os.path.join(save_dir, f"it{iteration:03d}_coarse_ha_half{k_half + 1}.npy"),
                np.asarray(coarse_ha[k_half], dtype=np.int32),
            )
    logger.info("Saved intermediate volumes to %s (iteration %d)", save_dir, iteration)


def _maybe_dump_noise_update_debug(
    *,
    iteration: int,
    current_size: int | None,
    image_shape,
    noise_stats_per_half,
    previous_noise_radial_per_half,
    noise_from_res_per_half,
    noise_from_res,
):
    """Write raw noise M-step terms for RELION parity debugging when requested."""

    dump_dir = os.environ.get("RELAX_NOISE_DEBUG_DUMP_DIR")
    if not dump_dir:
        return
    requested_iterations = parse_int_set(os.environ.get("RELAX_NOISE_DEBUG_DUMP_ITERATION"))
    if requested_iterations is not None and int(iteration) not in requested_iterations:
        return

    os.makedirs(dump_dir, exist_ok=True)
    n_shells = int(image_shape[0]) // 2 + 1
    shell_indices_half = np.asarray(make_shell_indices_half(image_shape), dtype=np.int64)
    half_counts = np.bincount(shell_indices_half, minlength=n_shells).astype(np.float64)[:n_shells]
    half_weights = np.asarray(make_half_image_weights(image_shape), dtype=np.float64)
    half_weighted_counts = np.bincount(shell_indices_half, weights=half_weights, minlength=n_shells).astype(
        np.float64,
    )[:n_shells]

    payload = {
        "zero_based_iteration": np.array([int(iteration)], dtype=np.int32),
        "one_based_iteration": np.array([int(iteration) + 1], dtype=np.int32),
        "current_size": np.array([-1 if current_size is None else int(current_size)], dtype=np.int32),
        "image_shape": np.asarray(image_shape, dtype=np.int32),
        "shell_index_half": shell_indices_half.astype(np.int32),
        "half_shell_counts": half_counts,
        "half_weighted_shell_counts": half_weighted_counts,
        "relion_half_plane_shell_counts": _relion_half_plane_shell_counts(image_shape),
        "mean_sigma2_noise": np.asarray(noise_from_res, dtype=np.float64),
    }
    for half_id, stats_k in enumerate(noise_stats_per_half, start=1):
        prefix = f"half{half_id}"
        wsum_sigma2 = np.asarray(stats_k.wsum_sigma2_noise, dtype=np.float64)
        img_power = np.asarray(stats_k.wsum_img_power, dtype=np.float64)
        payload[f"{prefix}_wsum_sigma2_noise"] = wsum_sigma2
        payload[f"{prefix}_wsum_img_power"] = img_power
        payload[f"{prefix}_wsum_total"] = wsum_sigma2 + img_power
        # One weight sum, or one per optics group.
        payload[f"{prefix}_sumw"] = np.atleast_1d(np.asarray(stats_k.sumw, dtype=np.float64))
        if getattr(stats_k, "wsum_norm_correction", None) is not None:
            payload[f"{prefix}_wsum_norm_correction"] = np.asarray(
                stats_k.wsum_norm_correction,
                dtype=np.float64,
            )
        payload[f"{prefix}_sigma2_noise"] = np.asarray(noise_from_res_per_half[half_id - 1], dtype=np.float64)
        payload[f"{prefix}_previous_sigma2_noise"] = np.asarray(
            previous_noise_radial_per_half[half_id - 1],
            dtype=np.float64,
        )
        if getattr(stats_k, "wsum_noise_a2", None) is not None:
            payload[f"{prefix}_wsum_noise_a2"] = np.asarray(stats_k.wsum_noise_a2, dtype=np.float64)
        if getattr(stats_k, "wsum_noise_xa", None) is not None:
            payload[f"{prefix}_wsum_noise_xa"] = np.asarray(stats_k.wsum_noise_xa, dtype=np.float64)

    path = os.path.join(dump_dir, f"recovar_noise_update_it{int(iteration) + 1:03d}.npz")
    np.savez_compressed(path, **payload)
    logger.info("Wrote RECOVAR noise update debug dump: %s", path)
