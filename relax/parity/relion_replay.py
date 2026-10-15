"""RELION replay state, captured sampling grids and iteration overrides.

Translate recorded sampling/model metadata into the grids, priors and per-half
corrections consumed by refinement. The controller owns when replay overrides
are applied; these helpers preserve the captured ordering, units and dtypes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from relax.refinement.noise_updates import (
    NoiseModel,
    noise_model_from_pixels,
)
from relax.refinement.ports import OptimiserAccuracyReplay
from relax.refinement.sigma_offset import SigmaOffset
from relax.runtime.env_flags import parse_env_flag_or_false
from relax.sampling.grids import translation_grid_for_class_count
from relax.sampling.healpix import angular_sampling_deg
from relax.sampling.orientation_priors import (
    DirectionPrior,
    class_weights_from_direction_prior,
    infer_direction_prior_healpix_order,
    normalize_class_direction_prior,
    normalize_class_direction_prior_per_half,
    normalize_direction_prior_per_half,
)
from relax.sampling.perturbation import relion_sampling_perturbation_for_iteration
from relax.star.metadata import (
    read_relion_direction_prior,
    read_relion_direction_priors,
    read_relion_model_metadata,
    read_relion_optimiser_metadata,
    read_relion_sampling_metadata,
)
from relax.steps.half_inputs import HalfSet, optional_half_arrays

if TYPE_CHECKING:
    from relax.reconstruction.resolution import ImageGeometry


logger = logging.getLogger("relax.diagnostics.relion_replay")  # the name before the move to relax.parity: log rows keep it


_KCLASS_REPLAY_TAU2_ENV = "RELAX_KCLASS_REPLAY_TAU2"


def sealed_sampling_base_grids(sealed_sampling_state, *, voxel_size_angstrom, dtype: np.dtype = np.float32):
    """Construct scorer grids directly from a schema-v3 captured sampling state.

    ``dtype`` controls the returned rotation matrices, working Euler grid, and
    translations, matching ``relion_scoring_rotation_grid``'s policy: pass ``np.float64`` under
    float64 scoring/projections so a restart from a sealed boundary keeps the
    same coarse-grid precision as a fresh (non-restarted) run.
    """

    state = sealed_sampling_state
    directions = np.asarray(state["directions_ipix"], dtype=np.int64)
    rot = np.asarray(state["rot_angles_deg"], dtype=np.float64)
    tilt = np.asarray(state["tilt_angles_deg"], dtype=np.float64)
    psi = np.asarray(state["psi_angles_deg"], dtype=np.float64)
    if directions.ndim != 1 or directions.size < 1:
        raise ValueError("sealed sampling directions must be a nonempty vector")
    if rot.shape != directions.shape or tilt.shape != directions.shape or psi.ndim != 1 or psi.size < 1:
        raise ValueError("sealed sampling Euler component shapes are inconsistent")
    source_eulers = np.stack(
        [
            np.tile(rot, psi.size),
            np.tile(tilt, psi.size),
            np.repeat(psi, directions.size),
        ],
        axis=1,
    )
    from relax.sampling.grids import relion_mstep_rotations_from_eulers

    rotations = relion_mstep_rotations_from_eulers(source_eulers, dtype=dtype)
    eulers = source_eulers.astype(dtype)
    voxel_size = float(voxel_size_angstrom)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("sealed sampling requires a finite positive voxel size")
    tx = np.asarray(state["translations_x_angstrom"], dtype=np.float64)
    ty = np.asarray(state["translations_y_angstrom"], dtype=np.float64)
    if tx.shape != ty.shape or tx.ndim != 1 or tx.size < 1:
        raise ValueError("sealed sampling translation component shapes are inconsistent")
    translations = np.stack([tx / voxel_size, ty / voxel_size], axis=1).astype(dtype)
    return rotations, eulers, jnp.asarray(translations, dtype=dtype)


def restore_convergence_state_from_replay_restart(state, replay, init_relion_iteration) -> None:
    """Restore convergence counters from a RELION optimiser/model STAR at a
    perturbation-replay restart iteration.

    RELION's convergence counters are not initialized against an infinite
    previous resolution -- a replay restart resumes them from the previous
    optimiser/model STAR instead of the fresh-run FSC/ini_high state. This is
    the loop-boundary counterpart to ``apply_iter_replay_overrides`` below,
    which handles the same ``perturb_replay_relion_dir`` source per mid-loop
    iteration.
    """
    init_opt_star = os.path.join(
        replay.perturb_replay_relion_dir,
        f"{replay.perturb_replay_relion_prefix}_it{init_relion_iteration:03d}_optimiser.star",
    )
    init_model_star = os.path.join(
        replay.perturb_replay_relion_dir,
        f"{replay.perturb_replay_relion_prefix}_it{init_relion_iteration:03d}_half1_model.star",
    )
    if os.path.exists(init_model_star):
        model_meta = read_relion_model_metadata(init_model_star)
        resolution_angstrom = float(model_meta["current_resolution"])
        if np.isfinite(resolution_angstrom) and resolution_angstrom > 0.0:
            state.current_resolution = resolution_angstrom
            state.previous_resolution = resolution_angstrom
    if os.path.exists(init_opt_star):
        opt_meta = read_relion_optimiser_metadata(init_opt_star)
        state.nr_iter_wo_resol_gain = int(opt_meta.get("number_iter_without_resolution_gain") or 0)
        hidden_variable_changes = int(opt_meta.get("number_iter_without_changing_assignments") or 0)
        state.nr_iter_wo_large_hidden_variable_changes = hidden_variable_changes
        state.nr_iter_wo_assignment_changes = hidden_variable_changes
        if opt_meta.get("overall_accuracy_rotations") is not None:
            state.acc_rot = float(opt_meta["overall_accuracy_rotations"])
        if opt_meta.get("overall_accuracy_translations_angst") is not None:
            state.acc_trans = float(opt_meta["overall_accuracy_translations_angst"])
        if opt_meta.get("smallest_changes_orientations") is not None:
            state.smallest_changes_optimal_orientations = float(opt_meta["smallest_changes_orientations"])
        if opt_meta.get("smallest_changes_offsets") is not None:
            state.smallest_changes_optimal_offsets_angstrom = float(opt_meta["smallest_changes_offsets"])
        if opt_meta.get("smallest_changes_classes") is not None:
            state.smallest_changes_optimal_classes = float(opt_meta["smallest_changes_classes"])
        if opt_meta.get("has_converged") is not None:
            state.has_converged = bool(int(opt_meta["has_converged"]))
    logger.info(
        "Replay convergence init from RELION iter %03d: res=%.2f A, "
        "stalls=(res=%d,hvc=%d), smallest=(rot=%.3f deg, trans=%.3f A, class=%.3f)",
        init_relion_iteration,
        state.current_resolution,
        state.nr_iter_wo_resol_gain,
        state.nr_iter_wo_large_hidden_variable_changes,
        state.smallest_changes_optimal_orientations,
        state.smallest_changes_optimal_offsets_angstrom,
        state.smallest_changes_optimal_classes,
    )


def _replay_perturbation_seed(
    replay_dir: str,
    relion_iteration: int,
    explicit_seed: int | None,
    replay_prefix: str = "run",
) -> int | None:
    """Return the RELION optimiser seed that generated a sampling state."""
    if explicit_seed is not None:
        return int(explicit_seed)
    candidates = [
        os.path.join(replay_dir, f"{replay_prefix}_it{int(relion_iteration):03d}_optimiser.star"),
        os.path.join(replay_dir, f"{replay_prefix}_optimiser.star"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        seed = read_relion_optimiser_metadata(path).get("random_seed")
        if seed is not None:
            return int(seed)
    return None


def resolve_replay_random_perturbation(
    *,
    star_value: float,
    perturbation_factor: float,
    relion_iteration: int,
    replay_dir: str,
    replay_prefix: str = "run",
    explicit_seed: int | None,
    restart_state_iteration: int | None = None,
) -> tuple[float, str]:
    """Recover RELION's live perturbation without STAR decimal truncation (the STAR value without a seed)."""
    seed = _replay_perturbation_seed(
        replay_dir,
        relion_iteration,
        explicit_seed,
        replay_prefix=replay_prefix,
    )
    if seed is None:
        return float(star_value), "star-fallback"

    exact = relion_sampling_perturbation_for_iteration(
        float(perturbation_factor),
        int(seed),
        int(relion_iteration),
        restart_state_iteration=restart_state_iteration,
    )
    # RELION writes this field with as few as five digits after the decimal.
    # Treat the STAR value as a provenance guard, not as the arithmetic input.
    if not np.isclose(exact, float(star_value), rtol=0.0, atol=5.1e-6):
        raise ValueError(
            "Seed-reconstructed SamplingPerturbation disagrees with replay STAR: "
            f"iteration={relion_iteration} seed={seed} exact={exact:+.12g} "
            f"star={float(star_value):+.12g}"
        )
    source = "seed-exact"
    if restart_state_iteration is not None:
        source = f"seed-exact-restart@{int(restart_state_iteration)}"
    return float(exact), source


def class_tau2_replay(*, iteration, n_classes, iter_replay_override, replay, logger):
    """Select captured Class3D priors without changing the normal Iref-derived policy.

    Return spectra, the diagnostic enable flag and its source label. A captured
    array is validated even when diagnostic use is disabled.
    """
    kclass_tau2_source = "previous Iref power spectra"
    replay_class_tau2 = None
    replay_tau2_enabled = parse_env_flag_or_false(_KCLASS_REPLAY_TAU2_ENV, logger=logger)
    tau2_replay_override = iter_replay_override
    tau2_replay_label = "current replay override"
    if replay_tau2_enabled:
        # RELION updates mymodel.tau2_class during expectation setup
        # from the current Iref, then uses that same model state for
        # maximization. Therefore run_itNNN_model.star contains the
        # tau2 prior used by iteration NNN, not the prior for NNN+1.
        same_iter_index = iteration + 1
        tau2_replay_override = None
        if replay.replay_iteration_overrides is not None and same_iter_index < len(replay.replay_iteration_overrides):
            tau2_replay_override = replay.replay_iteration_overrides[same_iter_index]
            tau2_replay_label = f"same-iteration replay override index={same_iter_index}"
        if tau2_replay_override is None or tau2_replay_override.get("class_tau2") is None:
            logger.warning(
                "Diagnostic %s=1 requested same-numbered Class3D tau2 at iter=%d, "
                "but replay override index %d is unavailable; falling back to current override",
                _KCLASS_REPLAY_TAU2_ENV,
                iteration + 1,
                same_iter_index,
            )
            tau2_replay_override = iter_replay_override
            tau2_replay_label = "current replay override fallback"
    if tau2_replay_override is not None and tau2_replay_override.get("class_tau2") is not None:
        replay_class_tau2 = np.asarray(tau2_replay_override["class_tau2"], dtype=np.float64)
        replay_class_tau2_shape = replay_class_tau2.shape
        if len(replay_class_tau2_shape) != 2 or replay_class_tau2_shape[0] != n_classes:
            raise ValueError(
                "class_tau2 replay override must have shape "
                f"({n_classes}, n_shells), got {replay_class_tau2_shape}",
            )
        if replay_tau2_enabled:
            kclass_tau2_source = f"RELION replay class_tau2 ({tau2_replay_label})"
            logger.info(
                "Diagnostic %s=1: Class3D tau2 replay override used at iter=%d from %s with shape=%s",
                _KCLASS_REPLAY_TAU2_ENV,
                iteration + 1,
                tau2_replay_label,
                replay_class_tau2_shape,
            )
        else:
            logger.info(
                "Class3D tau2 replay override available at iter=%d with shape=%s; "
                "M-step tau2 is recomputed from previous Iref power spectra",
                iteration + 1,
                replay_class_tau2_shape,
            )
    return replay_class_tau2, replay_tau2_enabled, kclass_tau2_source


def _state_swap_reference_dir(probe, perturb_replay_relion_dir, *, iteration: int):
    """The RELION directory a state-swap probe reads its references from, or ``None`` to keep the run's own.

    Only the probe's target iteration replaces references, and only when the probe asks for it. Reads the
    ``probe``'s ``iteration`` and ``replay_relion_references`` entries (None: no probe).
    """

    if (
        probe is None
        or int(probe.get("iteration", -1)) != int(iteration)
        or not bool(probe.get("replay_relion_references", False))
    ):
        return None
    if perturb_replay_relion_dir is None:
        logger.warning(
            "RELION reference replay requested at iteration %d but perturb_replay_relion_dir is unset; "
            "keeping RECOVAR references",
            int(iteration) + 1,
        )
        return None
    return Path(perturb_replay_relion_dir)


def _load_relion_replay_reference(
    relion_dir, *, prefix: str, relion_iter: int, iteration_number: int, half_idx: int, class_number: int, volume_shape
):
    """One RELION map as a flat centered Fourier volume: the half's own file, else the one both halves share."""

    from recovar.core import fourier_transform_utils
    from recovar.utils.helpers import load_relion_volume as _load_relion_volume

    map_path = relion_dir / (
        f"{prefix}_it{relion_iter:03d}_half{half_idx + 1}_"
        f"class{class_number:03d}.mrc"
    )
    if not map_path.exists():
        shared_path = relion_dir / (
            f"{prefix}_it{relion_iter:03d}_class{class_number:03d}.mrc"
        )
        if shared_path.exists():
            map_path = shared_path
    if not map_path.exists():
        raise FileNotFoundError(
            "State-swap probe requested RELION reference "
            f"for scoring iteration {iteration_number}, half {half_idx + 1}, "
            f"class {class_number}, but {map_path} is missing"
        )
    real_volume = np.asarray(_load_relion_volume(str(map_path)), dtype=np.float32)
    if tuple(real_volume.shape) != tuple(volume_shape):
        raise ValueError(
            f"RELION replay reference {map_path} has shape {real_volume.shape}, "
            f"expected {tuple(volume_shape)}"
        )
    reference = jnp.asarray(fourier_transform_utils.get_dft3(real_volume).reshape(-1))
    logger.info(
        "Debug RELION reference replay: scoring iter %d half %d class %d <- %s",
        iteration_number,
        half_idx + 1,
        class_number,
        map_path,
    )
    return reference


def replay_k1_relion_references(
    reference_model, options, *, probe, iteration: int, replay_dir, replay_prefix, volume_shape,
):
    """The two K=1 scoring references: RELION's half maps at a state-swap ``probe``'s target, else the model's own.

    ``replay_dir`` is the replay directory still live at this iteration and ``replay_prefix`` its file prefix.
    Reads ``reference_model.maps``; from ``options``: ``schedule.init_relion_iteration``.
    """

    relion_dir = _state_swap_reference_dir(probe, replay_dir, iteration=iteration)
    if relion_dir is None:
        return reference_model.maps
    return [
        _load_relion_replay_reference(
            relion_dir,
            prefix=replay_prefix,
            relion_iter=int(options.schedule.init_relion_iteration) + int(iteration),
            iteration_number=int(iteration) + 1,
            half_idx=half_idx,
            class_number=1,
            volume_shape=volume_shape,
        )
        for half_idx in range(2)
    ]


def replay_class_relion_references(
    reference_model, options, *, probe, iteration: int, replay_dir, replay_prefix, volume_shape,
):
    """Each half's class stack of scoring references: RELION's class maps at a state-swap ``probe``'s target.

    Elsewhere the model's own maps. ``replay_dir`` is the replay directory still live at this iteration.
    Reads ``reference_model.maps``; from ``options``: ``schedule.init_relion_iteration`` and
    ``k_class.n_classes`` (``replay_prefix``: the replay's file prefix).
    """

    relion_dir = _state_swap_reference_dir(probe, replay_dir, iteration=iteration)
    if relion_dir is None:
        return reference_model.maps
    return [
        jnp.stack(
            [
                _load_relion_replay_reference(
                    relion_dir,
                    prefix=replay_prefix,
                    relion_iter=int(options.schedule.init_relion_iteration) + int(iteration),
                    iteration_number=int(iteration) + 1,
                    half_idx=half_idx,
                    class_number=class_idx + 1,
                    volume_shape=volume_shape,
                )
                for class_idx in range(int(options.k_class.n_classes))
            ],
            axis=0,
        )
        for half_idx in range(2)
    ]


def _replay_control_model_iteration(init_relion_iteration: int, loop_iteration: int) -> int:
    """Return the RELION model.star index whose control state governs this replay step."""
    return int(init_relion_iteration) + int(loop_iteration) + 1


def _install_final_replay_particle_state(
    final_replay_override,
    halves,
    *,
    sigma_offset: SigmaOffset,
    noise_model: NoiseModel,
    image_shape,
    dtype,
):
    """Install the final-pass replay fields no mode reads; ``(sigma_offset, noise_model, installed field names)``.

    The translation prior width, the previous poses and corrections of ``halves`` (written in place) and the
    noise model. An absent field keeps the argument's value.
    """
    from relax.parity.state_swap_runtime import copy_half_pair

    _final_replay_fields = []
    _final_replay_sigma_per_half = final_replay_override.get("translation_sigma_angstrom_per_half")
    if _final_replay_sigma_per_half is not None:
        sigma_offset = SigmaOffset.from_halves(_final_replay_sigma_per_half)
        _final_replay_fields.append("translation_sigma_angstrom_per_half")
    _final_replay_sigma = final_replay_override.get("translation_sigma_angstrom")
    if _final_replay_sigma is not None and _final_replay_sigma_per_half is None:
        sigma_offset = SigmaOffset(float(_final_replay_sigma))
        _final_replay_fields.append("translation_sigma_angstrom")
    _final_replay_prev_trans = final_replay_override.get("previous_best_translations")
    if _final_replay_prev_trans is not None:
        for particle_half, value_for_half in zip(halves, copy_half_pair(_final_replay_prev_trans), strict=True):
            particle_half.translations = value_for_half
        _final_replay_fields.append("previous_best_translations")
    _final_replay_prev_eulers = final_replay_override.get("previous_best_rotation_eulers")
    if _final_replay_prev_eulers is not None:
        for particle_half, value_for_half in zip(halves, copy_half_pair(_final_replay_prev_eulers), strict=True):
            particle_half.rotation_eulers = value_for_half
        _final_replay_fields.append("previous_best_rotation_eulers")
    _final_replay_fields.extend(
        _apply_replay_correction_overrides(
            relion_half_inputs=halves,
            replay_override=final_replay_override,
        )
    )
    _final_replay_noise = final_replay_override.get("noise_variance")
    if _final_replay_noise is not None:
        noise_model = noise_model_from_pixels(
            _final_replay_noise, image_shape, dtype=dtype,
        )
        _final_replay_fields.append("noise_variance")
    return sigma_offset, noise_model, _final_replay_fields


def _log_final_replay_fields(log: logging.Logger, override_index: int, fields: list[str]) -> None:
    log.info(
        "RELION replay: final all-data replays last numbered RELION state "
        "(previous_state_index=%d, fields=%s)",
        override_index,
        ",".join(fields) if fields else "<none>",
    )


def apply_k1_final_replay_state(
    final_replay_override,
    halves,
    direction_priors,
    *,
    sigma_offset: SigmaOffset,
    noise_model: NoiseModel,
    healpix_order: int,
    image_shape,
    symmetry: str,
    dtype,
    override_index: int,
    log: logging.Logger,
):
    """Install the last numbered RELION state before the K=1 final all-data pass; ``(sigma_offset, noise_model)``.

    Each field ``final_replay_override`` supplies replaces the run's own: the translation prior width, the
    previous poses and corrections of ``halves`` and the entries of ``direction_priors`` (both written in
    place), and the noise model. An absent field keeps the argument's value. A prior at another HEALPix
    order than ``healpix_order`` is installed at its own order, which scoring ignores: RELION resets
    ``pdf_direction`` to an even distribution whenever the number of directions changes
    (``MlModel::initialisePdfDirection``, ``ml_model.cpp:1150-1165``); it never remaps it.
    """
    sigma_offset, noise_model, _final_replay_fields = _install_final_replay_particle_state(
        final_replay_override,
        halves,
        sigma_offset=sigma_offset,
        noise_model=noise_model,
        image_shape=image_shape,
        dtype=dtype,
    )
    _final_replay_dir_prior = final_replay_override.get("direction_prior")
    if _final_replay_dir_prior is not None:
        _final_replay_priors = normalize_direction_prior_per_half(
            _final_replay_dir_prior, dtype=dtype
        )
        for _half_idx in range(2):
            if _final_replay_priors[_half_idx] is None:
                continue
            _prior = np.asarray(_final_replay_priors[_half_idx], dtype=dtype)
            _prior_order = infer_direction_prior_healpix_order(
                _prior,
                symmetry=symmetry, expected_order=healpix_order,
            )
            direction_priors[_half_idx] = DirectionPrior(_prior, _prior_order)
        _final_replay_fields.append("direction_prior")
    _log_final_replay_fields(log, override_index, _final_replay_fields)
    return sigma_offset, noise_model


def apply_class_final_replay_state(
    final_replay_override,
    halves,
    direction_priors,
    *,
    sigma_offset: SigmaOffset,
    noise_model: NoiseModel,
    n_classes: int,
    healpix_order: int,
    image_shape,
    symmetry: str,
    dtype,
    override_index: int,
    log: logging.Logger,
):
    """Install the last numbered RELION state before the Class3D final all-data pass; ``(sigma_offset, noise_model)``.

    Each field ``final_replay_override`` supplies replaces the run's own: the translation prior width, the
    previous poses and corrections of ``halves`` and the entries of ``direction_priors`` (both written in
    place, a half's class rows normalised per class at ``dtype``), and the noise model. An absent field keeps
    the argument's value. A prior at another HEALPix order than ``healpix_order`` keeps its own order, which
    scoring ignores, as RELION resets it (``apply_k1_final_replay_state``).
    """
    sigma_offset, noise_model, _final_replay_fields = _install_final_replay_particle_state(
        final_replay_override,
        halves,
        sigma_offset=sigma_offset,
        noise_model=noise_model,
        image_shape=image_shape,
        dtype=dtype,
    )
    _final_replay_dir_prior = final_replay_override.get("direction_prior")
    if _final_replay_dir_prior is not None:
        _final_replay_priors = normalize_class_direction_prior_per_half(
            _final_replay_dir_prior,
            n_classes,
            dtype=dtype,
        )
        for _half_idx in range(2):
            if _final_replay_priors[_half_idx] is None:
                continue
            _prior_k = np.asarray(_final_replay_priors[_half_idx], dtype=dtype)
            _prior_order_k = infer_direction_prior_healpix_order(
                _prior_k[0],
                symmetry=symmetry, expected_order=healpix_order,
            )
            _prior_k = normalize_class_direction_prior(_prior_k, n_classes, dtype=dtype)
            direction_priors[_half_idx] = DirectionPrior(_prior_k, _prior_order_k)
        _final_replay_fields.append("direction_prior")
    _log_final_replay_fields(log, override_index, _final_replay_fields)
    return sigma_offset, noise_model


def apply_final_replay_state(
    final_replay_override,
    halves,
    direction_priors,
    *,
    sigma_offset: SigmaOffset,
    noise_model: NoiseModel,
    n_classes: int,
    healpix_order: int,
    image_shape,
    symmetry: str,
    dtype,
    override_index: int,
    log: logging.Logger,
):
    """Install the last numbered RELION state before the final all-data pass; ``(sigma_offset, noise_model)``.

    This is the one remaining mode decision of the final replay state, to be removed when the K=1 and
    Class3D trajectories call ``apply_k1_final_replay_state`` and ``apply_class_final_replay_state``
    directly.
    """
    if n_classes > 1:
        return apply_class_final_replay_state(
            final_replay_override,
            halves,
            direction_priors,
            sigma_offset=sigma_offset,
            noise_model=noise_model,
            n_classes=n_classes,
            healpix_order=healpix_order,
            image_shape=image_shape,
            symmetry=symmetry,
            dtype=dtype,
            override_index=override_index,
            log=log,
        )
    return apply_k1_final_replay_state(
        final_replay_override,
        halves,
        direction_priors,
        sigma_offset=sigma_offset,
        noise_model=noise_model,
        healpix_order=healpix_order,
        image_shape=image_shape,
        symmetry=symmetry,
        dtype=dtype,
        override_index=override_index,
        log=log,
    )


def _apply_replay_correction_overrides(*, relion_half_inputs, replay_override) -> list[str]:
    """Apply replay norm/scale state while distinguishing serialized and live scale."""

    replay_image_value = replay_override.get("image_corrections")
    serialized_scale_value = replay_override.get("serialized_scale_corrections")
    scoring_scale_value = replay_override.get("scoring_scale_corrections")
    if "scale_corrections" in replay_override:
        raise ValueError("Replay scale requires serialized_scale_corrections or scoring_scale_corrections")

    resident_dtypes = [
        np.asarray(value).dtype
        for values in ([particle_half.image_corrections for particle_half in relion_half_inputs], [particle_half.scale_corrections for particle_half in relion_half_inputs])
        for value in values
        if value is not None
    ]
    correction_dtype = np.result_type(*resident_dtypes) if resident_dtypes else None
    replay_images = optional_half_arrays(replay_image_value, dtype=correction_dtype)
    serialized_scales = optional_half_arrays(serialized_scale_value, dtype=correction_dtype)
    scoring_scales = optional_half_arrays(scoring_scale_value, dtype=correction_dtype)

    for half_idx in range(2):
        resident_image = relion_half_inputs[half_idx].image_corrections
        resident_scale = relion_half_inputs[half_idx].scale_corrections
        override_image = replay_images[half_idx]
        serialized_scale = serialized_scales[half_idx]
        explicit_scoring_scale = scoring_scales[half_idx]

        # A model STAR is leader-serialized provenance, not necessarily the
        # scale resident on the scorer rank. Preserve a live native scale
        # unless an explicit scoring oracle is supplied. A cold start has no
        # resident scale and therefore falls back to the serialized one.
        target_scale = (
            explicit_scoring_scale
            if explicit_scoring_scale is not None
            else resident_scale
            if resident_scale is not None
            else serialized_scale
        )
        base_image = override_image if override_image is not None else resident_image
        base_scale = (
            serialized_scale
            if override_image is not None and serialized_scale is not None
            else resident_scale
        )

        if target_scale is not None:
            target_scale = np.asarray(target_scale, dtype=correction_dtype)
            if not np.all(np.isfinite(target_scale)) or np.any(target_scale <= 0.0):
                raise ValueError("scoring scale corrections must be finite and positive")
        if base_scale is not None:
            base_scale = np.asarray(base_scale, dtype=correction_dtype)
            if not np.all(np.isfinite(base_scale)) or np.any(base_scale <= 0.0):
                raise ValueError("source scale corrections must be finite and positive")
        if base_image is not None:
            base_image = np.asarray(base_image, dtype=correction_dtype)
            if target_scale is not None and base_scale is None:
                raise ValueError("Cannot preserve image_corrections/scale without a source scale")
            if target_scale is not None and base_scale is not None:
                if base_image.shape != base_scale.shape or base_image.shape != target_scale.shape:
                    raise ValueError("image, source-scale, and scoring-scale corrections must have matching shapes")
                base_image = base_image * (target_scale / base_scale)
            relion_half_inputs[half_idx].image_corrections = base_image
        if explicit_scoring_scale is not None or resident_scale is None:
            relion_half_inputs[half_idx].scale_corrections = target_scale

    applied_fields = []
    if replay_image_value is not None:
        applied_fields.append("image_corrections")
        logger.info("Replay override: image_corrections <- norm state rescaled to live scoring scale")
    if serialized_scale_value is not None:
        applied_fields.append("serialized_scale_corrections")
        logger.info("Replay provenance: serialized_scale_corrections recorded; resident scoring scale preserved")
    if scoring_scale_value is not None:
        applied_fields.append("scoring_scale_corrections")
        logger.info("Replay override: scoring scale corrections <- explicit scorer oracle")
    return applied_fields


@dataclass
class ReplayOverrideResult:
    """Iteration-state values touched by replay overrides.

    State, particle halves and direction-prior owners are updated in place.
    Replacements, including the noise model, are assigned by the controller.
    """

    cs: int
    prior_translations: Any  # jnp.ndarray or None — used downstream by local-search prior
    previous_best_rotations: list
    noise_model: NoiseModel
    current_sigma_offset_angstrom: float
    replay_meta: dict | None  # parsed sampling.star (or None); used downstream by perturbation apply
    current_sigma_offset_angstrom_per_half: list[float] | None = None
    class_weights: np.ndarray | None = None


def sealed_rotation_ids_for_scoring(sealed_sampling_state, trial_grid, *, use_local: bool):
    """A sealed capture's rotation ids for a global scorer grid, checked against the trial grid; else None."""
    if sealed_sampling_state is None or use_local:
        return None
    rotation_ids = _sealed_sampling_rotation_ids(sealed_sampling_state)
    if rotation_ids.shape != (int(trial_grid.rotations.shape[0]),):
        raise RuntimeError(
            "sealed captured rotation IDs do not match the directly materialized scorer grid"
        )
    return rotation_ids


def _sealed_sampling_rotation_ids(sealed_sampling_state):
    """Map captured direction/psi rows to canonical coarse rotation IDs."""

    direction_ids = np.asarray(sealed_sampling_state["directions_ipix"], dtype=np.int64)
    n_psi = int(np.asarray(sealed_sampling_state["psi_angles_deg"]).size)
    order = int(sealed_sampling_state["healpix_order_original"])
    n_pixels = 12 * (4**order)
    return np.concatenate(
        [direction_ids + psi_index * n_pixels for psi_index in range(n_psi)]
    ).astype(np.int64, copy=False)


def read_optimiser_accuracy_replay(
    *,
    replay_dir,
    replay_prefix,
    init_relion_iteration,
    iteration: int,
    sealed_sampling_state,
    acc_rot,
    acc_trans,
    convergence_acc_rot,
    convergence_acc_trans,
    logger,
) -> OptimiserAccuracyReplay:
    """Read RELION's numbered optimiser accuracies for the convergence update.

    With a replay directory and no sealed sampling state, the numbered
    optimiser STAR for this iteration is selected. When it exists, its finite
    ``overall_accuracy_rotations`` / ``overall_accuracy_translations_angst``
    replace both the reported and the convergence accuracies. Read or parse
    failures are logged as warnings and keep whatever was assigned before the
    failure, as the controller did inline. Sealed replay leaves every input
    unchanged.
    """

    metadata = None
    optimiser_star = None
    optimiser_iteration = None
    if replay_dir is not None and sealed_sampling_state is None:
        optimiser_iteration = int(init_relion_iteration) + iteration + 1
        optimiser_star = os.path.join(
            replay_dir,
            f"{replay_prefix}_it{optimiser_iteration:03d}_optimiser.star",
        )
        if os.path.exists(optimiser_star):
            try:
                metadata = read_relion_optimiser_metadata(optimiser_star)
                relion_acc_rot = metadata.get("overall_accuracy_rotations")
                relion_acc_trans_angst = metadata.get("overall_accuracy_translations_angst")
                if relion_acc_rot is not None and np.isfinite(float(relion_acc_rot)):
                    acc_rot = float(relion_acc_rot)
                    convergence_acc_rot = acc_rot
                if relion_acc_trans_angst is not None and np.isfinite(float(relion_acc_trans_angst)):
                    acc_trans = float(relion_acc_trans_angst)
                    convergence_acc_trans = acc_trans
                logger.info(
                    "Replay override: optimiser accuracy <- %s (acc_rot=%.3f deg, acc_trans=%s Å)",
                    optimiser_star,
                    float(acc_rot) if acc_rot is not None else float("nan"),
                    f"{acc_trans:.3f}" if acc_trans is not None else "unset",
                )
            except Exception as exc:
                logger.warning(
                    "Replay override: failed to read optimiser metadata from %s: %s", optimiser_star, exc
                )
    return OptimiserAccuracyReplay(
        metadata=metadata,
        optimiser_star=optimiser_star,
        optimiser_iteration=optimiser_iteration,
        acc_rot=acc_rot,
        acc_trans=acc_trans,
        convergence_acc_rot=convergence_acc_rot,
        convergence_acc_trans=convergence_acc_trans,
    )


def apply_optimiser_convergence_replay(
    state,
    *,
    metadata,
    optimiser_star,
    optimiser_iteration,
    replay_dir,
    replay_prefix,
    logger,
):
    """Apply numbered optimiser controls after the native state update.

    ``metadata`` exists only for a replay directory without sealed sampling
    (``read_optimiser_accuracy_replay``), so ``replay_dir`` is a path here.
    Mutate the existing state in metadata order. Missing fields retain their
    computed values. An unnumbered final optimiser may close the numbered
    replay only when the next sampling STAR is absent.
    ``read_optimiser_accuracy_replay`` supplies the accuracy overrides before
    the controller's state update; the controller calls this afterward.

    Unlike restart restoration, missing numbered counters retain computed
    values rather than defaulting to zero. Keep those contracts separate.
    """
    _relion_res_stalls = metadata.get("number_iter_without_resolution_gain")
    _relion_hvc_stalls = metadata.get("number_iter_without_changing_assignments")
    if _relion_res_stalls is not None:
        state.nr_iter_wo_resol_gain = int(_relion_res_stalls)
    if _relion_hvc_stalls is not None:
        _hvc = int(_relion_hvc_stalls)
        state.nr_iter_wo_large_hidden_variable_changes = _hvc
        state.nr_iter_wo_assignment_changes = _hvc
    _relion_changes = (
        ("changes_optimal_orientations", "current_changes_optimal_orientations", float),
        ("changes_optimal_offsets", "current_changes_optimal_offsets_angstrom", float),
        ("changes_optimal_classes", "current_changes_optimal_classes", float),
        ("smallest_changes_orientations", "smallest_changes_optimal_orientations", float),
        ("smallest_changes_offsets", "smallest_changes_optimal_offsets_angstrom", float),
        ("smallest_changes_classes", "smallest_changes_optimal_classes", float),
    )
    for _meta_key, _state_attr, _cast in _relion_changes:
        _value = metadata.get(_meta_key)
        if _value is not None:
            setattr(state, _state_attr, _cast(_value))
    _relion_has_converged = metadata.get("has_converged")
    if _relion_has_converged is not None:
        state.has_converged = bool(int(_relion_has_converged))

    # RELION's final all-data pass is stored as unnumbered
    # run_sampling.star/run_optimiser.star.  Numbered strict-replay
    # streams therefore end one iteration before the final pass; do not
    # request run_it{N+1}_sampling.star when RELION already recorded the
    # final convergence state in run_optimiser.star.
    if not state.has_converged:
        _next_sampling_star = os.path.join(
            replay_dir,
            f"{replay_prefix}_it{optimiser_iteration + 1:03d}_sampling.star",
        )
        _final_sampling_star = os.path.join(
            replay_dir,
            f"{replay_prefix}_sampling.star",
        )
        _final_optimiser_star = os.path.join(
            replay_dir,
            f"{replay_prefix}_optimiser.star",
        )
        if (
            not os.path.exists(_next_sampling_star)
            and os.path.exists(_final_sampling_star)
            and os.path.exists(_final_optimiser_star)
        ):
            try:
                _final_optimiser_meta = read_relion_optimiser_metadata(_final_optimiser_star)
                _final_has_converged = _final_optimiser_meta.get("has_converged")
                if _final_has_converged is not None and bool(int(_final_has_converged)):
                    state.has_converged = True
                    logger.info(
                        "Replay override: RELION final optimiser convergence <- %s "
                        "(numbered replay ended after %s)",
                        _final_optimiser_star,
                        optimiser_star,
                    )
            except Exception as exc:
                logger.warning(
                    "Replay override: failed to read final optimiser metadata from %s: %s",
                    _final_optimiser_star,
                    exc,
                )
    logger.info(
        "Replay override: optimiser control <- %s "
        "(res_stalls=%d, hvc_stalls=%d, changes=(rot=%.3f deg, trans=%.3f A, class=%.0f), "
        "smallest=(rot=%.3f deg, trans=%.3f A, class=%.0f), converged=%s)",
        optimiser_star,
        state.nr_iter_wo_resol_gain,
        state.nr_iter_wo_large_hidden_variable_changes,
        state.current_changes_optimal_orientations,
        state.current_changes_optimal_offsets_angstrom,
        state.current_changes_optimal_classes,
        state.smallest_changes_optimal_orientations,
        state.smallest_changes_optimal_offsets_angstrom,
        state.smallest_changes_optimal_classes,
        state.has_converged,
    )


@dataclass(frozen=True)
class _ReplaySamplingStar:
    """One numbered RELION sampling STAR with the controls read from it; offsets in pixels."""

    path: str
    meta: dict
    healpix_order: int
    psi_step_deg: float
    offset_range: float
    offset_step: float


def _read_replay_sampling_star(
    perturb_replay_relion_dir, perturb_replay_relion_prefix, *, relion_iteration: int, pixel_size_angstrom
) -> _ReplaySamplingStar:
    """Read the sampling STAR RELION wrote for numbered iteration ``relion_iteration``."""

    _star = os.path.join(
        perturb_replay_relion_dir,
        f"{perturb_replay_relion_prefix}_it{relion_iteration:03d}_sampling.star",
    )
    _replay_meta = read_relion_sampling_metadata(_star)
    _relion_hp = int(_replay_meta["healpix_order"])
    _relion_psi_step_deg = float(_replay_meta.get("psi_step", angular_sampling_deg(_relion_hp)))
    # RELION stores offset_{range,step} in Angstroms; convert to px.
    _relion_offset_range = float(_replay_meta["offset_range"]) / pixel_size_angstrom
    _relion_offset_step = float(_replay_meta["offset_step"]) / pixel_size_angstrom
    return _ReplaySamplingStar(
        path=_star,
        meta=_replay_meta,
        healpix_order=_relion_hp,
        psi_step_deg=_relion_psi_step_deg,
        offset_range=_relion_offset_range,
        offset_step=_relion_offset_step,
    )


def install_sealed_sampling(state, sealed_sampling_state, *, iteration: int, image_geometry: ImageGeometry, dtype):
    """Install a sealed frozen-boundary sampling state on ``state``.

    Returns ``(current size, prior translations, replay meta)``.
    """

    if int(iteration) != 0:
        raise ValueError("sealed frozen-boundary sampling currently owns exactly one iteration")
    _px = image_geometry.pixel_size_angstrom
    _relion_hp = int(sealed_sampling_state["healpix_order_original"])
    if state.max_healpix_order is not None and _relion_hp > int(state.max_healpix_order):
        raise ValueError(
            "sealed sampling HEALPix order exceeds runtime maximum: "
            f"sealed={_relion_hp} max={state.max_healpix_order}"
        )
    state.healpix_order = _relion_hp
    state.do_local_search = bool(state.auto_sampling and state.healpix_order >= state.auto_local_healpix_order)
    state.sigma_rot = np.deg2rad(float(sealed_sampling_state["sigma_rot_deg"]))
    state.sigma_psi = np.deg2rad(float(sealed_sampling_state["sigma_psi_deg"]))
    state.translation_range = float(sealed_sampling_state["offset_range_angstrom"]) / _px
    state.translation_step = float(sealed_sampling_state["offset_step_angstrom"]) / _px
    sealed_x = np.asarray(sealed_sampling_state["translations_x_angstrom"], dtype=np.float64)
    sealed_y = np.asarray(sealed_sampling_state["translations_y_angstrom"], dtype=np.float64)
    prior_translations = jnp.asarray(
        np.stack([sealed_x / _px, sealed_y / _px], axis=1),
        dtype=dtype,
    )
    cs = int(sealed_sampling_state["current_size"])
    replay_meta = {
        "healpix_order": _relion_hp,
        "psi_step": float(sealed_sampling_state["psi_step_deg"]),
        "offset_range": float(sealed_sampling_state["offset_range_angstrom"]),
        "offset_step": float(sealed_sampling_state["offset_step_angstrom"]),
        "perturbation_factor": float(sealed_sampling_state["perturbation_factor"]),
        "random_perturbation": float(sealed_sampling_state["random_perturbation"]),
        "sealed_v3": True,
    }
    logger.info(
        "Frozen-boundary v3 owns sampling: consumer_iter=%d hp=%d current/coarse=%d/%d "
        "translations=%d rp=%+.12g",
        int(sealed_sampling_state["consumer_relion_iteration"]),
        _relion_hp,
        cs,
        int(sealed_sampling_state["coarse_size"]),
        int(prior_translations.shape[0]),
        float(sealed_sampling_state["random_perturbation"]),
    )
    return cs, prior_translations, replay_meta


def _install_replay_sampling_controls(
    state,
    cs: int,
    sampling: _ReplaySamplingStar,
    *,
    replay_translations,
    state_translations,
    perturb_replay_relion_dir: str,
    perturb_replay_relion_prefix: str,
    init_relion_iteration: int,
    iteration: int,
):
    """Install RELION's numbered sampling controls on ``state``; ``(current size, prior translations)``.

    ``replay_translations`` is the translation grid of the STAR's range and step and
    ``state_translations`` the grid of the state's own, both built by the caller's grid rule. The HEALPix
    order, local-search activation and prior sigmas, translation range and step are written to ``state``;
    the current size comes from the control model STAR of the same iteration.
    """

    _model_star = None
    _model_meta = None
    _translation_grid_differs = state_translations.shape != replay_translations.shape
    if not _translation_grid_differs:
        _translation_grid_differs = not np.allclose(
            state_translations,
            replay_translations,
            rtol=0.0,
            atol=1e-6,
        )
    _translation_params_differ = (
        abs(float(state.translation_range) - sampling.offset_range) > 1e-6
        or abs(float(state.translation_step) - sampling.offset_step) > 1e-6
    )
    if _translation_grid_differs and not _translation_params_differ:
        logger.info(
            "Replay override: preserving current translation grid for sub-tolerance "
            "RELION replay rounding: range %.9g -> %.9g px, step %.9g -> %.9g px "
            "(translation grid n=%d vs rounded n=%d)",
            float(state.translation_range),
            sampling.offset_range,
            float(state.translation_step),
            sampling.offset_step,
            int(state_translations.shape[0]),
            int(replay_translations.shape[0]),
        )
        replay_translations = state_translations
    prior_translations = jnp.array(replay_translations)
    _capped_hp = (
        sampling.healpix_order
        if state.max_healpix_order is None
        else min(sampling.healpix_order, state.max_healpix_order)
    )
    if state.healpix_order != _capped_hp:
        if _capped_hp < sampling.healpix_order:
            logger.info(
                "Replay override: healpix_order %d -> %d (RELION %d capped by max_healpix_order=%d, from %s)",
                state.healpix_order,
                _capped_hp,
                sampling.healpix_order,
                state.max_healpix_order,
                sampling.path,
            )
        else:
            logger.info(
                "Replay override: healpix_order %d -> %d (from %s)",
                state.healpix_order,
                _capped_hp,
                sampling.path,
            )
        state.healpix_order = _capped_hp
    _replay_do_local = bool(state.auto_sampling and state.healpix_order >= state.auto_local_healpix_order)
    if state.do_local_search != _replay_do_local:
        logger.info(
            "Replay override: local_search %s -> %s (healpix_order=%d, auto_local_healpix_order=%d)",
            state.do_local_search,
            _replay_do_local,
            state.healpix_order,
            state.auto_local_healpix_order,
        )
        state.do_local_search = _replay_do_local
        if _replay_do_local:
            state.sigma_rot = 0.0
            state.sigma_psi = 0.0
    # RELION's run_itNNN_model.star is written after iteration N, but its
    # current_size and local-prior sigma fields are the controls used by
    # that same iteration's E-step.  Other fields in the same file, such
    # as current resolution and average Pmax, are post-iteration state.
    # Sampling perturbation uses the same N suffix.
    # Reuse it for both current_size and local-prior sigmas.
    _cs_iter = _replay_control_model_iteration(init_relion_iteration, iteration)
    _model_star_candidates = [
        os.path.join(
            perturb_replay_relion_dir,
            f"{perturb_replay_relion_prefix}_it{_cs_iter:03d}_half1_model.star",
        ),
        os.path.join(
            perturb_replay_relion_dir,
            f"{perturb_replay_relion_prefix}_it{_cs_iter:03d}_model.star",
        ),
    ]
    _model_star = next((path for path in _model_star_candidates if os.path.exists(path)), None)
    if _model_star is not None:
        _model_meta = read_relion_model_metadata(_model_star)
    if _replay_do_local:
        _relion_sigma_rot_deg = None
        _relion_sigma_psi_deg = None
        if _model_meta is not None:
            _sigma_rot_deg = _model_meta.get("sigma_prior_rot_angle")
            _sigma_tilt_deg = _model_meta.get("sigma_prior_tilt_angle")
            _sigma_psi_deg = _model_meta.get("sigma_prior_psi_angle")
            _dir_candidates = [
                float(value)
                for value in (_sigma_rot_deg, _sigma_tilt_deg)
                if value is not None and float(value) > 0.0
            ]
            if _dir_candidates:
                _relion_sigma_rot_deg = max(_dir_candidates)
            if _sigma_psi_deg is not None and float(_sigma_psi_deg) > 0.0:
                _relion_sigma_psi_deg = float(_sigma_psi_deg)
        if _relion_sigma_rot_deg is None:
            _relion_sigma_rot_deg = sampling.psi_step_deg
            logger.info(
                "Replay override: model local prior sigma missing; falling back to RELION psi_step %.3f deg",
                sampling.psi_step_deg,
            )
        if _relion_sigma_psi_deg is None:
            _relion_sigma_psi_deg = _relion_sigma_rot_deg
        _relion_sigma_rot_rad = np.deg2rad(_relion_sigma_rot_deg)
        _relion_sigma_psi_rad = np.deg2rad(_relion_sigma_psi_deg)
        if (
            abs(float(state.sigma_rot) - _relion_sigma_rot_rad) > 1e-8
            or abs(float(state.sigma_psi) - _relion_sigma_psi_rad) > 1e-8
        ):
            logger.info(
                "Replay override: local prior sigma %.3f/%.3f deg -> %.3f/%.3f deg (from %s)",
                float(np.rad2deg(state.sigma_rot)),
                float(np.rad2deg(state.sigma_psi)),
                _relion_sigma_rot_deg,
                _relion_sigma_psi_deg,
                _model_star if _model_star is not None else sampling.path,
            )
        state.sigma_rot = _relion_sigma_rot_rad
        state.sigma_psi = _relion_sigma_psi_rad
    if _translation_params_differ:
        logger.info(
            "Replay override: translation_range %.9g -> %.9g px, step %.9g -> %.9g px "
            "(translation grid n=%d -> %d)",
            float(state.translation_range),
            sampling.offset_range,
            float(state.translation_step),
            sampling.offset_step,
            int(state_translations.shape[0]),
            int(replay_translations.shape[0]),
        )
        state.translation_range = sampling.offset_range
        state.translation_step = sampling.offset_step

    # Override current_size from the RELION model star for the replayed
    # iteration's E-step controls.
    if _model_meta is not None:
        _relion_cs = int(_model_meta["current_image_size"])
        if _relion_cs <= 0:
            logger.info(
                "Replay override: ignoring non-positive current_size=%d from %s",
                _relion_cs,
                _model_star,
            )
        elif cs != _relion_cs:
            logger.info(
                "Replay override: current_size %d -> %d (from %s)",
                cs,
                _relion_cs,
                _model_star,
            )
            cs = _relion_cs
    return cs, prior_translations


def _replay_star_priors_due(*, iteration: int, preserve_existing_direction_prior: bool, iter_replay_override) -> bool:
    """Whether the previous iteration's model STAR supplies this iteration's direction priors."""

    if iteration > 0 and not preserve_existing_direction_prior:
        return iter_replay_override is None or iter_replay_override.get("direction_prior") is None
    if preserve_existing_direction_prior:
        logger.info(
            "Replay control: preserving sealed existing direction priors; "
            "external model prior reload suppressed"
        )
    return False


def _replayed_prior_source_order(direction_row, *, symmetry: str, expected_order):
    """The HEALPix order of one replayed direction row, which the numbered replay installs unchanged."""

    # Preserve the source grid identity. The scorer treats a prior
    # from another order as uniform, matching RELION
    # updateAngularSampling/initialisePdfDirection. Remapping here
    # would incorrectly retain learned anisotropy after refinement.
    return infer_direction_prior_healpix_order(
        direction_row,
        **{'symmetry': symmetry, 'expected_order': expected_order} if symmetry != 'C1' else {},
    )


def _install_replay_override_particle_state(
    iter_replay_override,
    relion_half_inputs,
    *,
    previous_best_rotations: list,
    noise_model: NoiseModel,
    current_sigma_offset_angstrom: float,
    current_sigma_offset_angstrom_per_half,
    image_shape,
    iteration: int,
    dtype,
):
    """Install the override fields no mode reads.

    The translation prior width, the previous poses and corrections of ``relion_half_inputs`` (written in
    place), the previous rotations and the noise model. An absent field keeps the argument's value.
    Returns ``(sigma offset, its per-half pair, previous rotations, noise model)``.
    """

    _replay_sigma_per_half = iter_replay_override.get("translation_sigma_angstrom_per_half")
    if _replay_sigma_per_half is not None:
        replayed = SigmaOffset.from_halves(_replay_sigma_per_half)
        current_sigma_offset_angstrom_per_half = list(replayed.per_half_angstrom)
        current_sigma_offset_angstrom = replayed.shared_angstrom
        logger.info(
            "Replay override: sigma_offset <- half1 %.4f A, half2 %.4f A, mean %.4f A (iter=%d)",
            current_sigma_offset_angstrom_per_half[0],
            current_sigma_offset_angstrom_per_half[1],
            current_sigma_offset_angstrom,
            iteration + 1,
        )
    _replay_sigma = iter_replay_override.get("translation_sigma_angstrom")
    if _replay_sigma is not None and _replay_sigma_per_half is None:
        current_sigma_offset_angstrom = float(_replay_sigma)
        current_sigma_offset_angstrom_per_half = list(SigmaOffset(_replay_sigma).per_half_angstrom)
        logger.info(
            "Replay override: sigma_offset <- %.4f A (iter=%d)",
            current_sigma_offset_angstrom,
            iteration + 1,
        )
    _replay_prev_trans = iter_replay_override.get("previous_best_translations")
    if _replay_prev_trans is not None:
        for particle_half, value_for_half in zip(relion_half_inputs, optional_half_arrays(
            _replay_prev_trans, dtype=dtype
        ), strict=True):
            particle_half.translations = value_for_half
        logger.info(
            "Replay override: previous_best_translations <- half1=%s half2=%s",
            "set" if relion_half_inputs[0].translations is not None else "none",
            "set" if relion_half_inputs[1].translations is not None else "none",
        )
    _replay_prev_rots = iter_replay_override.get("previous_best_rotations")
    if _replay_prev_rots is not None:
        previous_best_rotations = optional_half_arrays(_replay_prev_rots, dtype=dtype)
        logger.info(
            "Replay override: previous_best_rotations <- half1=%s half2=%s",
            "set" if previous_best_rotations[0] is not None else "none",
            "set" if previous_best_rotations[1] is not None else "none",
        )
    _replay_prev_eulers = iter_replay_override.get("previous_best_rotation_eulers")
    if _replay_prev_eulers is not None:
        for particle_half, value_for_half in zip(relion_half_inputs, optional_half_arrays(_replay_prev_eulers), strict=True):
            particle_half.rotation_eulers = value_for_half
        logger.info(
            "Replay override: previous_best_rotation_eulers <- half1=%s half2=%s",
            "set" if relion_half_inputs[0].rotation_eulers is not None else "none",
            "set" if relion_half_inputs[1].rotation_eulers is not None else "none",
        )
    _apply_replay_correction_overrides(
        relion_half_inputs=relion_half_inputs,
        replay_override=iter_replay_override,
    )
    _replay_noise = iter_replay_override.get("noise_variance")
    if _replay_noise is not None:
        noise_model = noise_model_from_pixels(
            _replay_noise, image_shape, dtype=dtype,
        )
        logger.info("Replay override: sigma2_noise <- per-half model.star arrays")
    return (
        current_sigma_offset_angstrom,
        current_sigma_offset_angstrom_per_half,
        previous_best_rotations,
        noise_model,
    )


def apply_k1_iter_replay_overrides(
    *,
    iter_replay_override: dict | None,
    perturb_replay_relion_dir: str | None,
    perturb_replay_relion_prefix: str = "run",
    init_relion_iteration: int,
    iteration: int,
    state,
    cs: int,
    image_geometry: ImageGeometry,
    relion_half_inputs: tuple[HalfSet, HalfSet],
    previous_best_rotations: list,
    noise_model: NoiseModel,
    current_sigma_offset_angstrom: float,
    current_sigma_offset_angstrom_per_half: list[float] | None = None,
    direction_priors,
    preserve_existing_direction_prior: bool = False,
    sealed_sampling_state: dict | None = None,
    dtype: np.dtype = np.float32,
    symmetry: str = "C1",
) -> ReplayOverrideResult:
    """Apply per-iteration replay overrides to the in-flight state of a K=1 iteration.

    Each half's direction prior is one vector, installed at its source HEALPix order; the model STAR of
    a half that RELION did not write leaves that half's prior as it is.

    See ``docs/math/em_parity_program.md`` under the 2026-07-15 targeted
    posterior discriminators for the serialized-versus-runtime scale contract.

    Mutates ``state`` and ``relion_half_inputs``, and replaces entries of the ``direction_priors`` list in place. Returns explicit new values for everything else.

    Two override sources, applied in order:

    1. ``perturb_replay_relion_dir``: read RELION's per-iter sampling.star +
       (control) model.star + (previous-iter) half-model.star, override
       healpix order, local-search activation, sigma priors, translation
       range/step, current_size, and direction priors.
    2. ``iter_replay_override`` dict: explicit overrides for sigma_offset,
       previous-best poses, image corrections, serialized/scoring scale
       corrections, noise variance and direction priors.
    """

    _replay_prior_translations = None
    _replay_meta = None
    _current_sigma_offset_angstrom_per_half = list(
        SigmaOffset(current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half).per_half_angstrom
    )

    if sealed_sampling_state is not None:
        cs, _replay_prior_translations, _replay_meta = install_sealed_sampling(
            state, sealed_sampling_state, iteration=iteration, image_geometry=image_geometry, dtype=dtype
        )
    elif perturb_replay_relion_dir is not None:
        _sampling = _read_replay_sampling_star(
            perturb_replay_relion_dir,
            perturb_replay_relion_prefix,
            relion_iteration=init_relion_iteration + iteration + 1,
            pixel_size_angstrom=image_geometry.pixel_size_angstrom,
        )
        _replay_meta = _sampling.meta
        _px = image_geometry.pixel_size_angstrom
        cs, _replay_prior_translations = _install_replay_sampling_controls(
            state,
            cs,
            _sampling,
            replay_translations=translation_grid_for_class_count(
                _sampling.offset_range, _sampling.offset_step, n_classes=1, source_units_per_pixel=_px
            ).astype(dtype),
            state_translations=translation_grid_for_class_count(
                float(state.translation_range), float(state.translation_step), n_classes=1, source_units_per_pixel=_px
            ).astype(dtype),
            perturb_replay_relion_dir=perturb_replay_relion_dir,
            perturb_replay_relion_prefix=perturb_replay_relion_prefix,
            init_relion_iteration=init_relion_iteration,
            iteration=iteration,
        )
        if _replay_star_priors_due(
            iteration=iteration,
            preserve_existing_direction_prior=preserve_existing_direction_prior,
            iter_replay_override=iter_replay_override,
        ):
            _prior_iter = init_relion_iteration + iteration
            for _half_idx in range(2):
                _prior_star = os.path.join(
                    perturb_replay_relion_dir,
                    f"{perturb_replay_relion_prefix}_it{_prior_iter:03d}_half{_half_idx + 1}_model.star",
                )
                if not os.path.exists(_prior_star):
                    continue
                _relion_direction_prior = read_relion_direction_prior(_prior_star, dtype=dtype)
                _relion_direction_prior_order = _replayed_prior_source_order(
                    _relion_direction_prior, symmetry=symmetry, expected_order=state.healpix_order
                )
                direction_priors[_half_idx] = DirectionPrior(_relion_direction_prior, _relion_direction_prior_order)
                logger.info(
                    "Replay override: direction prior half-%d <- %s (%d directions, range=[%.6f, %.6f], zeros=%d)",
                    _half_idx + 1,
                    _prior_star,
                    len(_relion_direction_prior),
                    float(_relion_direction_prior.min()),
                    float(_relion_direction_prior.max()),
                    int(np.sum(_relion_direction_prior == 0)),
                )

    if iter_replay_override is not None:
        (
            current_sigma_offset_angstrom,
            _current_sigma_offset_angstrom_per_half,
            previous_best_rotations,
            noise_model,
        ) = _install_replay_override_particle_state(
            iter_replay_override,
            relion_half_inputs,
            previous_best_rotations=previous_best_rotations,
            noise_model=noise_model,
            current_sigma_offset_angstrom=current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half=_current_sigma_offset_angstrom_per_half,
            image_shape=image_geometry.image_shape,
            iteration=iteration,
            dtype=dtype,
        )
        _replay_dir_prior = iter_replay_override.get("direction_prior")
        if _replay_dir_prior is not None:
            replay_priors = normalize_direction_prior_per_half(_replay_dir_prior, dtype=dtype)
            for _half_idx in range(2):
                if replay_priors[_half_idx] is None:
                    continue
                prior_k = np.asarray(replay_priors[_half_idx], dtype=dtype)
                prior_order_k = _replayed_prior_source_order(
                    prior_k, symmetry=symmetry, expected_order=state.healpix_order
                )
                direction_priors[_half_idx] = DirectionPrior(prior_k, prior_order_k)
                logger.info(
                    "Replay override: direction prior half-%d <- provided override (%d directions, range=[%.6f, %.6f], zeros=%d)",
                    _half_idx + 1,
                    len(prior_k),
                    float(prior_k.min()),
                    float(prior_k.max()),
                    int(np.sum(prior_k == 0)),
                )

    return ReplayOverrideResult(
        cs=cs,
        prior_translations=_replay_prior_translations,
        previous_best_rotations=previous_best_rotations,
        noise_model=noise_model,
        current_sigma_offset_angstrom=current_sigma_offset_angstrom,
        replay_meta=_replay_meta,
        current_sigma_offset_angstrom_per_half=_current_sigma_offset_angstrom_per_half,
    )


def apply_class_iter_replay_overrides(
    *,
    iter_replay_override: dict | None,
    perturb_replay_relion_dir: str | None,
    perturb_replay_relion_prefix: str = "run",
    init_relion_iteration: int,
    iteration: int,
    state,
    cs: int,
    image_geometry: ImageGeometry,
    n_classes: int,
    relion_half_inputs: tuple[HalfSet, HalfSet],
    previous_best_rotations: list,
    noise_model: NoiseModel,
    current_sigma_offset_angstrom: float,
    current_sigma_offset_angstrom_per_half: list[float] | None = None,
    direction_priors,
    preserve_existing_direction_prior: bool = False,
    sealed_sampling_state: dict | None = None,
    dtype: np.dtype = np.float32,
    symmetry: str = "C1",
) -> ReplayOverrideResult:
    """Apply per-iteration replay overrides to the in-flight state of a Class3D iteration.

    Each half's direction prior is one row per class, normalised per class and installed at its source
    HEALPix order; a half without its own model STAR reads the model STAR both halves share. The class
    weights are the row sums of the replayed prior when it has them.

    The two override sources, their order and what is written in place are those of
    ``apply_k1_iter_replay_overrides``.
    """

    _replay_prior_translations = None
    _replay_meta = None
    _replay_class_weights = None
    _current_sigma_offset_angstrom_per_half = list(
        SigmaOffset(current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half).per_half_angstrom
    )

    if sealed_sampling_state is not None:
        cs, _replay_prior_translations, _replay_meta = install_sealed_sampling(
            state, sealed_sampling_state, iteration=iteration, image_geometry=image_geometry, dtype=dtype
        )
    elif perturb_replay_relion_dir is not None:
        _sampling = _read_replay_sampling_star(
            perturb_replay_relion_dir,
            perturb_replay_relion_prefix,
            relion_iteration=init_relion_iteration + iteration + 1,
            pixel_size_angstrom=image_geometry.pixel_size_angstrom,
        )
        _replay_meta = _sampling.meta
        _px = image_geometry.pixel_size_angstrom
        cs, _replay_prior_translations = _install_replay_sampling_controls(
            state,
            cs,
            _sampling,
            replay_translations=translation_grid_for_class_count(
                _sampling.offset_range, _sampling.offset_step, n_classes=n_classes, source_units_per_pixel=_px
            ).astype(dtype),
            state_translations=translation_grid_for_class_count(
                float(state.translation_range),
                float(state.translation_step),
                n_classes=n_classes,
                source_units_per_pixel=_px,
            ).astype(dtype),
            perturb_replay_relion_dir=perturb_replay_relion_dir,
            perturb_replay_relion_prefix=perturb_replay_relion_prefix,
            init_relion_iteration=init_relion_iteration,
            iteration=iteration,
        )
        if _replay_star_priors_due(
            iteration=iteration,
            preserve_existing_direction_prior=preserve_existing_direction_prior,
            iter_replay_override=iter_replay_override,
        ):
            _prior_iter = init_relion_iteration + iteration
            for _half_idx in range(2):
                _prior_star = os.path.join(
                    perturb_replay_relion_dir,
                    f"{perturb_replay_relion_prefix}_it{_prior_iter:03d}_half{_half_idx + 1}_model.star",
                )
                if not os.path.exists(_prior_star):
                    # Class3D writes one shared model.star rather than
                    # auto-refine-style half-model STAR files.  During
                    # strict replay, use that shared direction prior for
                    # both RECOVAR halfsets.
                    _prior_star = os.path.join(
                        perturb_replay_relion_dir,
                        f"{perturb_replay_relion_prefix}_it{_prior_iter:03d}_model.star",
                    )
                    if not os.path.exists(_prior_star):
                        continue
                _relion_direction_prior = read_relion_direction_priors(_prior_star, n_classes, dtype=dtype)
                inferred_weights = class_weights_from_direction_prior(_relion_direction_prior, n_classes)
                if inferred_weights is not None:
                    _replay_class_weights = inferred_weights
                _relion_direction_prior_order = _replayed_prior_source_order(
                    _relion_direction_prior[0], symmetry=symmetry, expected_order=state.healpix_order
                )
                # The class rows read from a model STAR are normalised at float32 whatever the runtime dtype.
                direction_priors[_half_idx] = DirectionPrior(
                    normalize_class_direction_prior(_relion_direction_prior, n_classes, dtype=np.float32),
                    _relion_direction_prior_order,
                )
                logger.info(
                    "Replay override: class direction prior half-%d <- %s (%d classes, %d directions)",
                    _half_idx + 1,
                    _prior_star,
                    direction_priors[_half_idx].values.shape[0],
                    direction_priors[_half_idx].values.shape[1],
                )

    if iter_replay_override is not None:
        (
            current_sigma_offset_angstrom,
            _current_sigma_offset_angstrom_per_half,
            previous_best_rotations,
            noise_model,
        ) = _install_replay_override_particle_state(
            iter_replay_override,
            relion_half_inputs,
            previous_best_rotations=previous_best_rotations,
            noise_model=noise_model,
            current_sigma_offset_angstrom=current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half=_current_sigma_offset_angstrom_per_half,
            image_shape=image_geometry.image_shape,
            iteration=iteration,
            dtype=dtype,
        )
        _replay_dir_prior = iter_replay_override.get("direction_prior")
        if _replay_dir_prior is not None:
            inferred_weights = class_weights_from_direction_prior(_replay_dir_prior, n_classes)
            if inferred_weights is not None:
                _replay_class_weights = inferred_weights
            replay_priors = normalize_class_direction_prior_per_half(_replay_dir_prior, n_classes, dtype=dtype)
            for _half_idx in range(2):
                if replay_priors[_half_idx] is None:
                    continue
                prior_k = np.asarray(replay_priors[_half_idx], dtype=dtype)
                prior_order_k = _replayed_prior_source_order(
                    prior_k[0], symmetry=symmetry, expected_order=state.healpix_order
                )
                direction_priors[_half_idx] = DirectionPrior(
                    normalize_class_direction_prior(prior_k, n_classes, dtype=dtype), prior_order_k
                )
                logger.info(
                    "Replay override: class direction prior half-%d <- provided override (%d classes, %d directions)",
                    _half_idx + 1,
                    direction_priors[_half_idx].values.shape[0],
                    direction_priors[_half_idx].values.shape[1],
                )

    return ReplayOverrideResult(
        cs=cs,
        prior_translations=_replay_prior_translations,
        previous_best_rotations=previous_best_rotations,
        noise_model=noise_model,
        current_sigma_offset_angstrom=current_sigma_offset_angstrom,
        replay_meta=_replay_meta,
        current_sigma_offset_angstrom_per_half=_current_sigma_offset_angstrom_per_half,
        class_weights=_replay_class_weights,
    )


def apply_iter_replay_overrides(
    *,
    iter_replay_override: dict | None,
    perturb_replay_relion_dir: str | None,
    perturb_replay_relion_prefix: str = "run",
    init_relion_iteration: int,
    iteration: int,
    state,
    cs: int,
    image_geometry: ImageGeometry,
    n_classes: int,
    relion_half_inputs: tuple[HalfSet, HalfSet],
    previous_best_rotations: list,
    noise_model: NoiseModel,
    current_sigma_offset_angstrom: float,
    current_sigma_offset_angstrom_per_half: list[float] | None = None,
    direction_priors,
    preserve_existing_direction_prior: bool = False,
    sealed_sampling_state: dict | None = None,
    dtype: np.dtype = np.float32,
    symmetry: str = "C1",
) -> ReplayOverrideResult:
    """Apply per-iteration replay overrides to the in-flight iteration state.

    This is the one remaining mode decision of the per-iteration replay, to be removed when the K=1 and
    Class3D trajectories call ``apply_k1_iter_replay_overrides`` and ``apply_class_iter_replay_overrides``
    directly.
    """

    if n_classes > 1:
        return apply_class_iter_replay_overrides(
            iter_replay_override=iter_replay_override,
            perturb_replay_relion_dir=perturb_replay_relion_dir,
            perturb_replay_relion_prefix=perturb_replay_relion_prefix,
            init_relion_iteration=init_relion_iteration,
            iteration=iteration,
            state=state,
            cs=cs,
            image_geometry=image_geometry,
            n_classes=n_classes,
            relion_half_inputs=relion_half_inputs,
            previous_best_rotations=previous_best_rotations,
            noise_model=noise_model,
            current_sigma_offset_angstrom=current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            direction_priors=direction_priors,
            preserve_existing_direction_prior=preserve_existing_direction_prior,
            sealed_sampling_state=sealed_sampling_state,
            dtype=dtype,
            symmetry=symmetry,
        )
    return apply_k1_iter_replay_overrides(
        iter_replay_override=iter_replay_override,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        perturb_replay_relion_prefix=perturb_replay_relion_prefix,
        init_relion_iteration=init_relion_iteration,
        iteration=iteration,
        state=state,
        cs=cs,
        image_geometry=image_geometry,
        relion_half_inputs=relion_half_inputs,
        previous_best_rotations=previous_best_rotations,
        noise_model=noise_model,
        current_sigma_offset_angstrom=current_sigma_offset_angstrom,
        current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
        direction_priors=direction_priors,
        preserve_existing_direction_prior=preserve_existing_direction_prior,
        sealed_sampling_state=sealed_sampling_state,
        dtype=dtype,
        symmetry=symmetry,
    )


def select_final_sampling_star(
    replay_dir, replay_prefix, *, final_iteration, previous_iteration, require_final_state,
):
    """Select final-pass sampling metadata, preserving strict replay admission.

    Strict replay requires both unnumbered final state files, even when an
    explicit final-numbered sampling file exists. Selection then prefers that
    numbered file, the unnumbered final file, and finally the last numbered file.
    Return the selected path, provenance label and ordered candidate list for
    diagnostics. Path and label are None when no candidate exists.
    """
    if require_final_state:
        required = [
            os.path.join(replay_dir, f"{replay_prefix}_sampling.star"),
            os.path.join(replay_dir, f"{replay_prefix}_optimiser.star"),
        ]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(
                "Strict RELION final all-data replay requires the unnumbered final "
                "sampling and optimiser state; missing " + ", ".join(missing)
            )
    candidates = [
        (os.path.join(replay_dir, f"{replay_prefix}_it{final_iteration:03d}_sampling.star"), "final-numbered"),
        (os.path.join(replay_dir, f"{replay_prefix}_sampling.star"), "final"),
        (os.path.join(replay_dir, f"{replay_prefix}_it{previous_iteration:03d}_sampling.star"), "last-numbered"),
    ]
    for path, source in candidates:
        if os.path.exists(path):
            return path, source, candidates
    return None, None, candidates
