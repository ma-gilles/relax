"""RELION replay at the final all-data pass: the last numbered state, the final sampling star and the shared
correction overrides.

``apply_final_replay_state`` (K=1 or Class3D) installs RELION's last numbered particle state, priors and
noise model before the final pass and logs the fields it replayed; ``select_final_sampling_star`` picks the
sampling star the final pass replays. ``apply_replay_correction_overrides`` applies RELION's norm and scale
corrections; the per-iteration replay (``relion_replay``) uses it too.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from relax.refinement.noise_updates import NoiseModel, noise_model_from_pixels
from relax.refinement.sigma_offset import SigmaOffset
from relax.sampling.orientation_priors import (
    DirectionPrior,
    infer_direction_prior_healpix_order,
    normalize_class_direction_prior,
    normalize_class_direction_prior_per_half,
    normalize_direction_prior_per_half,
)
from relax.steps.half_inputs import optional_half_arrays

logger = logging.getLogger("relax.diagnostics.relion_replay")  # the name before the move to relax.parity: log rows keep it



def _copy_half_pair(values):
    """Each half's array copied (None stays None)."""
    return [None if value is None else np.asarray(value).copy() for value in values]

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
        for particle_half, value_for_half in zip(halves, _copy_half_pair(_final_replay_prev_trans), strict=True):
            particle_half.translations = value_for_half
        _final_replay_fields.append("previous_best_translations")
    _final_replay_prev_eulers = final_replay_override.get("previous_best_rotation_eulers")
    if _final_replay_prev_eulers is not None:
        for particle_half, value_for_half in zip(halves, _copy_half_pair(_final_replay_prev_eulers), strict=True):
            particle_half.rotation_eulers = value_for_half
        _final_replay_fields.append("previous_best_rotation_eulers")
    _final_replay_fields.extend(
        apply_replay_correction_overrides(
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


def apply_replay_correction_overrides(*, relion_half_inputs, replay_override) -> list[str]:
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
