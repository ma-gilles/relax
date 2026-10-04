"""RELION replay state, captured sampling grids and iteration overrides.

Translate recorded sampling/model metadata into the grids, priors and per-half
corrections consumed by refinement. The controller owns when replay overrides
are applied; these helpers preserve the captured ordering, units and dtypes.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.helpers.convergence import healpix_angular_step
from relax.helpers.env_flags import parse_env_flag_or_false
from relax.helpers.orientation_priors import (
    DirectionPrior,
    class_weights_from_direction_prior,
    infer_direction_prior_healpix_order,
    normalize_class_direction_prior,
    normalize_class_direction_prior_per_half,
    normalize_direction_prior_per_half,
    remap_direction_prior_to_healpix_order,
    remap_half_direction_prior_to_healpix_order,
)
from relax.refinement.half_inputs import (
    HalfSet,
    SigmaOffset,
    _as_sigma_offset_half_pair,
    _normalize_sigma_offset_per_half,
    optional_half_arrays,
    sigma_offset_from_halves,
)
from relax.refinement.noise_updates import (
    NoiseModel,
    noise_model_from_pixels,
)
from relax.refinement.refinement_options import RefinementOptions
from relax.relion import relion_metadata
from relax.relion.initial_noise import (
    read_relion_single_optics_sigma2_noise,
    relion_mpi_process_start_scoring_noise_pair,
)
from relax.relion.relion_metadata import (
    read_relion_direction_prior,
    read_relion_direction_priors,
    read_relion_model_metadata,
    read_relion_optimiser_metadata,
    read_relion_sampling_metadata,
)
from relax.sampling import (
    _translation_grid_for_class_count,
    relion_sampling_perturbation_for_iteration,
)

if TYPE_CHECKING:
    from relax.helpers.resolution import ImageGeometry


logger = logging.getLogger(__name__)

_KCLASS_REPLAY_TAU2_ENV = "RELAX_KCLASS_REPLAY_TAU2"


def _class_tau2_replay(*, iteration, n_classes, iter_replay_override, replay, logger):
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


def _numbered_relion_iteration(init_relion_iteration: int, local_iteration: int) -> int:
    """Map a restart-local zero-based loop index to RELION's numbered iteration."""

    return int(init_relion_iteration) + int(local_iteration) + 1


def _past_perturb_replay_max_iter(iteration: int, perturb_replay_max_iter: int | None) -> bool:
    """Return whether ``iteration`` (0-indexed) is past the diagnostic replay cutoff.

    ``perturb_replay_max_iter`` is 1-indexed to match
    ``scripts/run_multi_iter_parity.py``'s ``--replay-override-max-iter``
    (and ``replay_iteration_overrides``, which the same flag also gates).
    ``None`` means "no cutoff": every iteration stays in range.
    """

    if perturb_replay_max_iter is None:
        return False
    return (int(iteration) + 1) > int(perturb_replay_max_iter)


def _native_sampling_boundary_for_iteration(options, *, iteration: int, replay_dir: str | None) -> bool:
    """Return whether this physical iteration owns sampling and convergence.

    A diagnostic replay cutoff is a real ownership boundary, not merely a
    guard around STAR reads. Once crossed, RECOVAR must resume its native
    expected-accuracy, angular-sampling, and convergence transitions.
    ``replay_dir`` is the replay directory still live at this iteration. Reads from ``options``:
    ``parity.perturb_replay_max_iter`` and ``debug.sealed_sampling_state``.
    """

    replay_active = replay_dir is not None and not _past_perturb_replay_max_iter(
        iteration,
        options.parity.perturb_replay_max_iter,
    )
    return not replay_active and options.debug.sealed_sampling_state is None


def _check_final_replay_source_iteration(*, replay, diagnostic_override, numbered_iteration_count) -> None:
    """Refuse a final-only substitution whose source is not the boundary the numbered iterations reached."""
    if (
        (diagnostic_override is not None or replay.final_replay_reference_maps is not None)
        and replay.final_replay_source_iteration is not None
        and numbered_iteration_count != int(replay.final_replay_source_iteration)
    ):
        raise RuntimeError(
            "Diagnostic final-only substitution source does not match autonomous convergence boundary: "
            f"source_iteration={int(replay.final_replay_source_iteration)} "
            f"numbered_iteration_count={numbered_iteration_count}"
        )


def _prepare_k1_final_replay_references(
    *,
    replay,
    diagnostic_override,
    numbered_iteration_count,
    means,
    final_join_means,
    logger,
):
    """Validate the K=1 final-only substitution boundary and prepare the two half references.

    Without supplied reference maps, return the existing reference list itself.
    Supplied maps retain each half's current dtype. Override-state application
    remains in the controller after this selection and validation boundary.
    """
    _check_final_replay_source_iteration(
        replay=replay, diagnostic_override=diagnostic_override, numbered_iteration_count=numbered_iteration_count
    )
    if replay.final_replay_reference_maps is not None:
        if len(replay.final_replay_reference_maps) != 2:
            raise ValueError("Diagnostic final-only RELION reference substitution requires exactly two half maps")
        expected_reference_shape = tuple(np.asarray(means[0]).shape)
        candidate_reference_shapes = [
            tuple(np.asarray(reference).shape) for reference in replay.final_replay_reference_maps
        ]
        if any(shape != expected_reference_shape for shape in candidate_reference_shapes):
            raise ValueError(
                "Diagnostic final-only RELION reference shape mismatch: "
                f"expected={expected_reference_shape} got={candidate_reference_shapes}"
            )
        final_join_means = [
            jnp.asarray(reference, dtype=means[half_idx].dtype)
            for half_idx, reference in enumerate(replay.final_replay_reference_maps)
        ]
        logger.info(
            "Diagnostic final-only RELION reference substitution at numbered boundary %d",
            numbered_iteration_count,
        )
    return final_join_means


def _validate_class_final_replay_boundary(*, replay, diagnostic_override, numbered_iteration_count) -> None:
    """Validate the Class3D final-only substitution boundary; Class3D has no reference substitution."""
    _check_final_replay_source_iteration(
        replay=replay, diagnostic_override=diagnostic_override, numbered_iteration_count=numbered_iteration_count
    )
    if replay.final_replay_reference_maps is not None:
        raise RuntimeError("Diagnostic final-only RELION reference substitution is currently K=1 only")


def _prepare_final_replay_references(
    *,
    replay,
    diagnostic_override,
    numbered_iteration_count,
    means,
    final_join_means,
    k_class_enabled,
    logger,
):
    """Validate the final-only substitution boundary and prepare half references.

    This is the one remaining mode decision of the final reference
    preparation, to be removed when the K=1 trajectory calls
    ``_prepare_k1_final_replay_references`` and the Class3D trajectory calls
    ``_validate_class_final_replay_boundary`` directly. Class3D keeps the
    reference list it was given.
    """
    if k_class_enabled:
        _validate_class_final_replay_boundary(
            replay=replay, diagnostic_override=diagnostic_override, numbered_iteration_count=numbered_iteration_count
        )
        return final_join_means
    return _prepare_k1_final_replay_references(
        replay=replay,
        diagnostic_override=diagnostic_override,
        numbered_iteration_count=numbered_iteration_count,
        means=means,
        final_join_means=final_join_means,
        logger=logger,
    )


def _select_final_replay_override(*, requested_index, diagnostic_override, replay_overrides, has_overrides, logger):
    """Select a final-pass state without copying it or applying its fields.

    Explicit diagnostic state wins, including an empty dictionary. Otherwise
    clamp to the last recorded slot; a missing slot in a nonempty history is
    an error. The caller retains replay admission and mutation ordering.
    """
    override_index = requested_index
    if diagnostic_override is not None:
        override = diagnostic_override
        logger.info(
            "Diagnostic final-only RELION state substitution at previous-state index %d (fields=%s)",
            requested_index,
            ",".join(sorted(override)) or "<none>",
        )
    else:
        override = None
    if diagnostic_override is None and has_overrides:
        override_index = min(requested_index, int(len(replay_overrides)) - 1)
        override = replay_overrides[override_index]
        if override_index != requested_index:
            logger.info(
                "RELION replay: final all-data requested previous-state index %d, using last available numbered replay override index %d",
                requested_index,
                override_index,
            )
    if override is None:
        if has_overrides:
            raise RuntimeError(
                f"Strict RELION final all-data replay is missing the requested previous-state override at index {requested_index}"
            )
        logger.info(
            "RELION replay: final all-data requested last numbered state replay, but no replay override exists for previous-state index %d",
            requested_index,
        )
    return override_index, override


def _has_numbered_replay_iteration_overrides(replay_iteration_overrides) -> bool:
    """Return whether replay contains state beyond the cold-start boundary.

    Override slot zero is also used by ``--relion_init_dir`` to seed RELION's
    iter-0 particle/model state for an otherwise autonomous refinement.  That
    cold-start state must not implicitly turn the final all-data iteration
    into a numbered replay.  Genuine trajectory replay populates at least one
    later slot; an explicit final-replay environment override remains handled
    separately by the caller.
    """
    if replay_iteration_overrides is None or len(replay_iteration_overrides) <= 1:
        return False
    return any(override is not None for override in replay_iteration_overrides[1:])


def _validate_k1_bpref_particle_order_scope(
    *,
    preserve_bpref_particle_order: bool,
    init_relion_iteration: int,
    perturb_replay_relion_dir,
    replay_iteration_overrides,
    sealed_sampling_state,
    sealed_scoring_context,
    allow_replayed_bpref_particle_order: bool = False,
    allow_state_swap_fresh_bpref_particle_order: bool = False,
    continues_own_run: bool = False,
) -> None:
    """Fail closed unless RELION's physical order can be preserved for this K=1 run.

    Preserving RELION's order selects the production K=1 arithmetic
    (``_fresh_k1_spectrum_norm_default``), so every K=1 run that knows the
    order uses it: a fresh run, a fresh run that replays RELION's
    perturbations and per-iteration state (``--perturb_replay_relion_dir``),
    and an imported-boundary replay that reconstructs the native order
    (``allow_replayed_bpref_particle_order``). A sealed boundary fixes its own
    arithmetic, so it is refused. A ``--continue`` of relax's own run
    (``continues_own_run``) keeps that run's fresh order, and with it the
    same arithmetic.
    """

    if not preserve_bpref_particle_order:
        return
    if sealed_sampling_state is not None or sealed_scoring_context is not None:
        raise ValueError(
            "RELION BPref particle-order preservation cannot alter a sealed boundary"
        )
    if allow_state_swap_fresh_bpref_particle_order:
        if int(init_relion_iteration) != 0:
            raise ValueError(
                "state-swap RELION BPref particle-order preservation requires a fresh iteration-0 run"
            )
        if perturb_replay_relion_dir is None:
            raise ValueError(
                "state-swap RELION BPref particle-order preservation requires perturbation replay"
            )
        if not _has_numbered_replay_iteration_overrides(replay_iteration_overrides):
            raise ValueError(
                "state-swap RELION BPref particle-order preservation requires numbered replay state"
            )
        return
    if int(init_relion_iteration) > 0 and continues_own_run:
        if perturb_replay_relion_dir is not None or _has_numbered_replay_iteration_overrides(
            replay_iteration_overrides
        ):
            raise ValueError("a continued run cannot also replay a RELION trajectory")
        return
    if int(init_relion_iteration) > 0:
        if not allow_replayed_bpref_particle_order:
            raise ValueError(
                "RELION BPref particle-order preservation at an imported iteration requires "
                "the native RELION order (allow_replayed_bpref_particle_order)"
            )
        if perturb_replay_relion_dir is None:
            raise ValueError(
                "replayed RELION BPref particle-order preservation requires perturbation replay"
            )
        return
    if (
        _has_numbered_replay_iteration_overrides(replay_iteration_overrides)
        and perturb_replay_relion_dir is None
    ):
        raise ValueError(
            "numbered replay state without perturbation replay has no RELION trajectory"
        )


def _validate_bpref_particle_order_scope(
    *,
    preserve_bpref_particle_order: bool,
    n_classes: int,
    init_relion_iteration: int,
    perturb_replay_relion_dir,
    replay_iteration_overrides,
    sealed_sampling_state,
    sealed_scoring_context,
    allow_replayed_bpref_particle_order: bool = False,
    allow_state_swap_fresh_bpref_particle_order: bool = False,
    continues_own_run: bool = False,
) -> None:
    """Fail closed unless RELION's physical order can be preserved for this run.

    This is the one remaining mode decision of the order-scope check, to be
    removed when the K=1 trajectory calls
    ``_validate_k1_bpref_particle_order_scope`` directly: Class3D refuses the
    preservation and has no scope to check.
    """

    if int(n_classes) != 1:
        if preserve_bpref_particle_order:
            raise ValueError("RELION BPref particle-order preservation is K=1-only")
        return
    _validate_k1_bpref_particle_order_scope(
        preserve_bpref_particle_order=preserve_bpref_particle_order,
        init_relion_iteration=init_relion_iteration,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        replay_iteration_overrides=replay_iteration_overrides,
        sealed_sampling_state=sealed_sampling_state,
        sealed_scoring_context=sealed_scoring_context,
        allow_replayed_bpref_particle_order=allow_replayed_bpref_particle_order,
        allow_state_swap_fresh_bpref_particle_order=allow_state_swap_fresh_bpref_particle_order,
        continues_own_run=continues_own_run,
    )


def _state_swap_reference_dir(options, perturb_replay_relion_dir, *, iteration: int):
    """The RELION directory a state-swap probe reads its references from, or ``None`` to keep the run's own.

    Only the probe's target iteration replaces references, and only when the probe asks for it. Reads
    ``options.debug.state_swap_probe`` (its ``iteration`` and ``replay_relion_references`` entries).
    """

    probe = options.debug.state_swap_probe
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


def replay_k1_relion_references(reference_model, options, *, iteration: int, replay_dir, volume_shape):
    """The two K=1 scoring references: RELION's half maps at a state-swap probe's target, else the model's own.

    ``replay_dir`` is the replay directory still live at this iteration. Reads ``reference_model.maps``;
    from ``options``: ``debug.state_swap_probe``, ``parity.perturb_replay_relion_prefix`` and
    ``schedule.init_relion_iteration``.
    """

    relion_dir = _state_swap_reference_dir(options, replay_dir, iteration=iteration)
    if relion_dir is None:
        return reference_model.maps
    return [
        _load_relion_replay_reference(
            relion_dir,
            prefix=options.parity.perturb_replay_relion_prefix,
            relion_iter=int(options.schedule.init_relion_iteration) + int(iteration),
            iteration_number=int(iteration) + 1,
            half_idx=half_idx,
            class_number=1,
            volume_shape=volume_shape,
        )
        for half_idx in range(2)
    ]


def replay_class_relion_references(reference_model, options, *, iteration: int, replay_dir, volume_shape):
    """Each half's class stack of scoring references: RELION's class maps at a state-swap probe's target.

    Elsewhere the model's own maps. ``replay_dir`` is the replay directory still live at this iteration.
    Reads ``reference_model.maps``; from ``options``: ``debug.state_swap_probe``,
    ``parity.perturb_replay_relion_prefix``, ``schedule.init_relion_iteration`` and ``k_class.n_classes``.
    """

    relion_dir = _state_swap_reference_dir(options, replay_dir, iteration=iteration)
    if relion_dir is None:
        return reference_model.maps
    return [
        jnp.stack(
            [
                _load_relion_replay_reference(
                    relion_dir,
                    prefix=options.parity.perturb_replay_relion_prefix,
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


def _maybe_debug_replay_relion_references(reference_model, options, *, iteration: int, replay_dir, volume_shape):
    """Replace scoring references with RELION maps for a state-swap probe.

    This is the one remaining mode decision of the reference replay (read from
    ``options.k_class.n_classes``), to be removed when the K=1 and Class3D
    trajectories call ``replay_k1_relion_references`` and
    ``replay_class_relion_references`` directly.
    """

    if int(options.k_class.n_classes) == 1:
        return replay_k1_relion_references(
            reference_model, options, iteration=iteration, replay_dir=replay_dir, volume_shape=volume_shape,
        )
    return replay_class_relion_references(
        reference_model, options, iteration=iteration, replay_dir=replay_dir, volume_shape=volume_shape,
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


def _resolve_replay_random_perturbation(
    *,
    star_value: float,
    perturbation_factor: float,
    relion_iteration: int,
    replay_dir: str,
    replay_prefix: str = "run",
    explicit_seed: int | None,
    precision_mode: str,
    restart_state_iteration: int | None = None,
) -> tuple[float, str]:
    """Recover RELION's live perturbation without STAR decimal truncation."""
    if precision_mode not in {"auto", "seed_exact", "star"}:
        raise ValueError(f"Unsupported perturb_replay_precision={precision_mode!r}")
    if precision_mode == "star":
        return float(star_value), "star"

    seed = _replay_perturbation_seed(
        replay_dir,
        relion_iteration,
        explicit_seed,
        replay_prefix=replay_prefix,
    )
    if seed is None:
        if precision_mode == "seed_exact":
            raise ValueError(
                "perturb_replay_precision='seed_exact' requires perturb_seed or "
                "_rlnRandomSeed in a replay optimiser STAR"
            )
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


def _perturbation_restart_state_iteration(
    restart_state_iterations,
    relion_iteration: int,
) -> int | None:
    """Return the latest explicit restart boundary preceding an iteration."""
    if restart_state_iterations is None:
        return None
    candidates = [
        int(value)
        for value in restart_state_iterations
        if int(value) < int(relion_iteration)
    ]
    return max(candidates) if candidates else None





@dataclass(frozen=True)
class RelionProjectorReplayState:
    """Exact per-half RELION ``Projector::data`` captured at one boundary.

    The source manifest binds the in-memory arrays to a sealed capture.  The
    iteration loop validates the remaining geometry metadata against the live
    replay before handing these slabs to the production scorer.
    """

    projector_half_by_half: tuple[np.ndarray, np.ndarray]
    projector_r_max_by_half: tuple[int, int]
    current_size: int
    padding_factor: int
    volume_shape: tuple[int, int, int]
    n_classes: int
    source_manifest_sha256: str


def _parse_relion_projector_replay_state(value, *, n_classes: int) -> RelionProjectorReplayState | None:
    """Validate an explicit captured-projector replay override.

    This intentionally accepts one atomic mapping instead of independent
    arrays and scalars.  A partial projector override would otherwise fall
    back to rebuilding some state from the resident half-map and no longer be
    an exact frozen-boundary replay.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("relion_projector_state must be a mapping")
    required = {
        "projector_half_by_half",
        "projector_r_max_by_half",
        "current_size",
        "padding_factor",
        "volume_shape",
        "n_classes",
        "source_manifest_sha256",
    }
    missing = sorted(required.difference(value))
    extra = sorted(set(value).difference(required))
    if missing or extra:
        raise ValueError(
            "relion_projector_state keys must match the version-1 contract exactly; "
            f"missing={missing}, extra={extra}"
        )

    captured_n_classes = int(value["n_classes"])
    if captured_n_classes != int(n_classes) or captured_n_classes <= 0:
        raise ValueError(
            "captured RELION projector class count does not match replay: "
            f"captured={captured_n_classes}, replay={int(n_classes)}"
        )
    current_size = int(value["current_size"])
    padding_factor = int(value["padding_factor"])
    if current_size <= 0 or padding_factor <= 0:
        raise ValueError("captured RELION projector current_size and padding_factor must be positive")

    volume_shape_values = tuple(int(item) for item in value["volume_shape"])
    if len(volume_shape_values) != 3 or any(item <= 0 for item in volume_shape_values):
        raise ValueError("captured RELION projector volume_shape must contain three positive dimensions")

    manifest_sha256 = str(value["source_manifest_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise ValueError("captured RELION projector source_manifest_sha256 must be 64 lowercase hex digits")

    projectors = value["projector_half_by_half"]
    r_max_values = value["projector_r_max_by_half"]
    if not isinstance(projectors, (list, tuple)) or len(projectors) != 2:
        raise ValueError("captured RELION projector state must contain exactly two half-set slabs")
    if not isinstance(r_max_values, (list, tuple)) or len(r_max_values) != 2:
        raise ValueError("captured RELION projector state must contain exactly two half-set r_max values")

    normalized_projectors = []
    normalized_r_max = []
    for half_idx, (projector, r_max) in enumerate(zip(projectors, r_max_values, strict=True), start=1):
        array = np.asarray(projector)
        if array.dtype != np.dtype(np.complex64):
            raise TypeError(
                f"captured RELION half-{half_idx} projector must be complex64, got {array.dtype}"
            )
        if array.ndim != 4 or int(array.shape[0]) != captured_n_classes:
            raise ValueError(
                "captured RELION projector slabs must have shape "
                f"(n_classes, z, y, x_half); half-{half_idx} has {array.shape}"
            )
        if any(int(size) <= 0 for size in array.shape[1:]):
            raise ValueError(f"captured RELION half-{half_idx} projector has an empty spatial dimension")
        if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
            raise ValueError(f"captured RELION half-{half_idx} projector contains non-finite values")
        r_max_int = int(r_max)
        if r_max_int < 0 or r_max_int >= min(int(size) for size in array.shape[1:]):
            raise ValueError(
                f"captured RELION half-{half_idx} projector r_max={r_max_int} is outside its slab"
            )
        normalized = np.ascontiguousarray(array).copy()
        normalized.setflags(write=False)
        normalized_projectors.append(normalized)
        normalized_r_max.append(r_max_int)

    if normalized_projectors[0].shape != normalized_projectors[1].shape:
        raise ValueError("captured RELION half-set projector slab shapes must match")

    return RelionProjectorReplayState(
        projector_half_by_half=(normalized_projectors[0], normalized_projectors[1]),
        projector_r_max_by_half=(normalized_r_max[0], normalized_r_max[1]),
        current_size=current_size,
        padding_factor=padding_factor,
        volume_shape=volume_shape_values,
        n_classes=captured_n_classes,
        source_manifest_sha256=manifest_sha256,
    )


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
    from relax.diagnostics.state_swap_runtime import _copy_half_pair

    _final_replay_fields = []
    _final_replay_sigma_per_half = final_replay_override.get("translation_sigma_angstrom_per_half")
    if _final_replay_sigma_per_half is not None:
        sigma_offset = sigma_offset_from_halves(
            _normalize_sigma_offset_per_half(_final_replay_sigma_per_half)
        )
        _final_replay_fields.append("translation_sigma_angstrom_per_half")
    _final_replay_sigma = final_replay_override.get("translation_sigma_angstrom")
    if _final_replay_sigma is not None and _final_replay_sigma_per_half is None:
        sigma_offset = SigmaOffset(
            float(_final_replay_sigma),
            _as_sigma_offset_half_pair(float(_final_replay_sigma)),
        )
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
    place, a half's prior vector remapped to ``healpix_order``), and the noise model. An absent field keeps
    the argument's value.
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
            if _prior_order != healpix_order:
                _prior = remap_direction_prior_to_healpix_order(
                    _prior,
                    _prior_order,
                    healpix_order,
                    symmetry=symmetry,
                    dtype=dtype,
                )
                _prior_order = healpix_order
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
    place, a half's class rows remapped to ``healpix_order`` and then normalised per class at ``dtype``), and
    the noise model. An absent field keeps the argument's value.
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
            if _prior_order_k != healpix_order:
                _prior_k = remap_half_direction_prior_to_healpix_order(
                    _prior_k,
                    _prior_order_k,
                    healpix_order,
                    n_classes=n_classes,
                    dtype=dtype,
                    symmetry=symmetry,
                )
                _prior_order_k = healpix_order
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
    relion_projector_state: RelionProjectorReplayState | None = None


def _sealed_sampling_base_grids(sealed_sampling_state, *, voxel_size_angstrom, dtype: np.dtype = np.float32):
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
    from relax.sampling import _relion_mstep_rotations_from_eulers

    rotations = _relion_mstep_rotations_from_eulers(source_eulers, dtype=dtype)
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


def _sealed_sampling_rotation_ids(sealed_sampling_state):
    """Map captured direction/psi rows to canonical coarse rotation IDs."""

    direction_ids = np.asarray(sealed_sampling_state["directions_ipix"], dtype=np.int64)
    n_psi = int(np.asarray(sealed_sampling_state["psi_angles_deg"]).size)
    order = int(sealed_sampling_state["healpix_order_original"])
    n_pixels = 12 * (4**order)
    return np.concatenate(
        [direction_ids + psi_index * n_pixels for psi_index in range(n_psi)]
    ).astype(np.int64, copy=False)


def _restore_convergence_state_from_replay_restart(state, options: RefinementOptions) -> None:
    """Restore convergence counters from a RELION optimiser/model STAR at a
    perturbation-replay restart iteration.

    RELION's convergence counters are not initialized against an infinite
    previous resolution -- a replay restart resumes them from the previous
    optimiser/model STAR instead of the fresh-run FSC/ini_high state. This is
    the loop-boundary counterpart to ``apply_iter_replay_overrides`` below,
    which handles the same ``perturb_replay_relion_dir`` source per mid-loop
    iteration.
    """
    parity, schedule = options.parity, options.schedule
    init_relion_iteration = int(schedule.init_relion_iteration)
    init_opt_star = os.path.join(
        parity.perturb_replay_relion_dir,
        f"{parity.perturb_replay_relion_prefix}_it{init_relion_iteration:03d}_optimiser.star",
    )
    init_model_star = os.path.join(
        parity.perturb_replay_relion_dir,
        f"{parity.perturb_replay_relion_prefix}_it{init_relion_iteration:03d}_half1_model.star",
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


@dataclass
class OptimiserAccuracyReplay:
    """Numbered optimiser accuracy override read before the state update.

    ``metadata`` and ``optimiser_iteration`` are ``None`` unless the numbered
    optimiser STAR was read; ``optimiser_star`` is the selected path whenever
    replay is active. The accuracy fields carry the caller's values with finite
    RELION accuracies substituted.
    """

    metadata: dict | None
    optimiser_star: str | None
    optimiser_iteration: int | None
    acc_rot: float | None
    acc_trans: float | None
    convergence_acc_rot: float | None
    convergence_acc_trans: float | None


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
    _relion_psi_step_deg = float(_replay_meta.get("psi_step", healpix_angular_step(_relion_hp)))
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


def _install_sealed_sampling(state, sealed_sampling_state, *, iteration: int, image_geometry: ImageGeometry, dtype):
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


def _replayed_projector_state(iter_replay_override, *, n_classes: int) -> RelionProjectorReplayState | None:
    """The captured RELION projector of an override, validated against the run's class count."""

    _replay_projector_state = _parse_relion_projector_replay_state(
        iter_replay_override.get("relion_projector_state"),
        n_classes=n_classes,
    )
    if _replay_projector_state is not None:
        logger.info(
            "Replay override: exact RELION Projector::data <- manifest %s",
            _replay_projector_state.source_manifest_sha256,
        )
    return _replay_projector_state


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
        current_sigma_offset_angstrom_per_half = _normalize_sigma_offset_per_half(_replay_sigma_per_half)
        current_sigma_offset_angstrom = float(
            0.5
            * (
                current_sigma_offset_angstrom_per_half[0]
                + current_sigma_offset_angstrom_per_half[1]
            )
        )
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
        current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(_replay_sigma)
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
       corrections, noise variance, direction priors, and a sealed exact
       per-half RELION projector state.
    """

    _replay_prior_translations = None
    _replay_meta = None
    _replay_projector_state = None
    _current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(
        current_sigma_offset_angstrom
        if current_sigma_offset_angstrom_per_half is None
        else current_sigma_offset_angstrom_per_half
    )

    if sealed_sampling_state is not None:
        cs, _replay_prior_translations, _replay_meta = _install_sealed_sampling(
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
            replay_translations=_translation_grid_for_class_count(
                _sampling.offset_range, _sampling.offset_step, n_classes=1, source_units_per_pixel=_px
            ).astype(dtype),
            state_translations=_translation_grid_for_class_count(
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
        _replay_projector_state = _replayed_projector_state(iter_replay_override, n_classes=1)
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
        relion_projector_state=_replay_projector_state,
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
    _replay_projector_state = None
    _current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(
        current_sigma_offset_angstrom
        if current_sigma_offset_angstrom_per_half is None
        else current_sigma_offset_angstrom_per_half
    )

    if sealed_sampling_state is not None:
        cs, _replay_prior_translations, _replay_meta = _install_sealed_sampling(
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
            replay_translations=_translation_grid_for_class_count(
                _sampling.offset_range, _sampling.offset_step, n_classes=n_classes, source_units_per_pixel=_px
            ).astype(dtype),
            state_translations=_translation_grid_for_class_count(
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
        _replay_projector_state = _replayed_projector_state(iter_replay_override, n_classes=n_classes)
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
        relion_projector_state=_replay_projector_state,
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


def _build_replay_iteration_overrides(
    relion_dir,
    half1_idx,
    half2_idx,
    max_iter,
    ds_voxel,
    ds_grid,
    *,
    include_normcorr,
    init_relion_iteration=0,
    particle_names=None,
    include_initial_state=False,
    include_k1_mean_variance=False,
    include_k1_scoring_scale=False,
    strict=False,
    process_start_noise_broadcast=True,
    noise_dtype=np.float32,
):
    """Build per-iter replay overrides keyed on recovar iteration index.

    For each recovar iteration k >= 1 (i.e. iter 2 onwards in RELION terms),
    reads RELION's run_it{k:03d}_data.star + half1/half2 model.star
    (or the shared Class3D run_it{k:03d}_model.star) and builds an
    override dict containing:
      * image_corrections: per-image (avg_norm/normcorr) * group_scale
      * serialized_scale_corrections: per-image model-STAR group scale,
        retained as provenance rather than forced onto the live scorer
      * scoring_scale_corrections: optional exact K=1 split-half scorer scale
        for state-swap diagnostics whose half-specific model STARs are owned
        by the two scoring ranks
      * previous_best_translations / previous_best_rotation_eulers: RELION's
        previous hard assignments for local-search centering

    This matches scripts/run_multi_iter_parity.py::_load_relion_iteration_override
    (the proven replay logic). The recovar iter-k override is read from
    RELION iter-k's model+data (since recovar iter-k corresponds to RELION
    iter-(k+1), and the per-image scalings used at the start of RELION
    iter-(k+1) are the ones written by RELION iter-k's M-step).

    ``init_relion_iteration`` is normally zero. Diagnostic profile runs can
    set it to a later RELION iteration to jump directly into local search; in
    that case override slot 0 is sourced from the upstream RELION iteration
    instead of being left empty.

    When ``include_initial_state`` is true, slot 0 is loaded from RELION
    iteration 0 as well. This is required for a strict cold-start replay:
    run_it000 carries the particle pre-centering offsets, initial orientations,
    image/scale corrections, and direction prior that RELION uses in its first
    expectation step.

    ``noise_dtype`` controls only the expanded pixel representation of the
    RFLOAT model-STAR spectrum. Double-scoring replays must retain float64 so
    the reciprocal is not narrowed before construction of ``Minvsigma2``.
    """
    import re as _re
    from pathlib import Path as _Path

    import starfile as _sf

    relion_dir = _Path(relion_dir).resolve()

    def _model_has_class_direction_priors(model):
        return any(str(key).startswith("model_pdf_orient_class_") for key in model)

    def _read_model_direction_prior(model_path, model):
        if not _model_has_class_direction_priors(model):
            return None
        from relax.relion.relion_metadata import (
            read_relion_direction_prior,
            read_relion_direction_priors,
        )

        has_multiple_classes = any(
            str(key).startswith("model_pdf_orient_class_") and not str(key).endswith("_1")
            for key in model
        )
        if has_multiple_classes:
            return read_relion_direction_priors(model_path)
        return read_relion_direction_prior(model_path)

    noise_dtype = np.dtype(noise_dtype)
    if noise_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"noise_dtype must be float32 or float64, got {noise_dtype}")

    def _read_model_noise_variance(model, *, image_shape):
        radial = read_relion_single_optics_sigma2_noise(
            model,
            context="replay model",
        )
        if radial is None:
            return None
        radial = radial * float(ds_grid) ** 4
        return np.asarray(
            utils.make_radial_image(jnp.asarray(radial), image_shape, extend_last_frequency=True),
            dtype=noise_dtype,
        ).reshape(-1)

    def _read_model_class_tau2(model):
        if not isinstance(model, dict):
            return None
        class_tau2 = []
        for key, table in model.items():
            match = _re.fullmatch(r"model_class_(\d+)", str(key))
            if match is None:
                continue
            col = "rlnReferenceTau2" if "rlnReferenceTau2" in table.columns else None
            if col is None and "rlnReferenceSigma2" in table.columns:
                col = "rlnReferenceSigma2"
            if col is None:
                continue
            class_tau2.append(
                (
                    int(match.group(1)),
                    np.asarray(table[col], dtype=np.float64) * float(ds_grid) ** 4,
                )
            )
        if not class_tau2:
            return None
        class_tau2.sort(key=lambda item: item[0])
        return np.stack([tau2 for _, tau2 in class_tau2], axis=0)

    # Index i is consumed by iteration_loop for recovar iter i+1 during the
    # numbered refinement, and by the final all-data pass as len(current_sizes).
    # Allocate one extra slot so convergence on the last configured numbered
    # iteration can replay RELION run_it{max_iter:03d}_data.star.
    overrides = [None] * (max_iter + 1)
    init_relion_iteration = int(init_relion_iteration)
    for recovar_iter in range(0, max_iter + 1):
        # recovar iter k uses corrections computed by RELION iter k (which were
        # written into run_it{k}_data.star). Fresh non-replay runs retain the
        # historical empty slot 0; strict cold-start replay explicitly loads
        # run_it000 because it contains nonzero particle pre-centering offsets
        # and the other state consumed by RELION's first expectation step.
        relion_iter = init_relion_iteration + recovar_iter
        if relion_iter < 0 or (relion_iter == 0 and not include_initial_state):
            continue
        data_star = relion_dir / f"run_it{relion_iter:03d}_data.star"
        model_h1 = relion_dir / f"run_it{relion_iter:03d}_half1_model.star"
        model_h2 = relion_dir / f"run_it{relion_iter:03d}_half2_model.star"
        model_shared = relion_dir / f"run_it{relion_iter:03d}_model.star"
        if model_h1.exists() and model_h2.exists():
            model_paths = (model_h1, model_h2)
        elif model_shared.exists():
            model_paths = (model_shared, model_shared)
        else:
            model_paths = None
        if not data_star.exists() or model_paths is None:
            missing = []
            if not data_star.exists():
                missing.append(str(data_star))
            if model_paths is None:
                missing.append(f"{model_h1} + {model_h2} or {model_shared}")
            message = (
                f"Replay override for recovar iter {recovar_iter + 1} "
                f"(RELION iter {relion_iter:03d}) is missing {'; '.join(missing)}"
            )
            if strict:
                raise ValueError(message)
            logger.warning("%s — leaving unset", message)
            continue

        data = _sf.read(str(data_star))
        parts = data["particles"] if isinstance(data, dict) else data
        m1 = _sf.read(str(model_paths[0]))
        m2 = _sf.read(str(model_paths[1]))

        replay_identity_rows = relion_metadata._particle_identity_rows(
            parts,
            label=f"RELION replay STAR {data_star}",
        )

        nc = np.asarray(parts["rlnNormCorrection"], dtype=np.float64)

        def _scalar(table, key):
            v = table[key]
            return float(v if isinstance(v, (int, float)) else v.iloc[0] if hasattr(v, "iloc") else v[0])

        avg_norm_h1 = _scalar(m1["model_general"], "rlnNormCorrectionAverage")
        avg_norm_h2 = _scalar(m2["model_general"], "rlnNormCorrectionAverage")

        # rlnSigmaOffsetsAngst is RELION's per-iter translation sigma. RELION
        # iter (k+1) loads it from the iter-k model.star and uses it to build
        # pdf_offset (acc_ml_optimiser_impl.h::pdf_offset). recovar's iter-1
        # does not accumulate sigma2_offset moments (no per-image prior centers
        # exist yet), so without an explicit override the iter-2 E-step uses
        # the default init sigma (10 Å) instead of the data-driven RELION
        # value, which is ~6× too wide and depresses iter-2 Pmax by ~22%.
        sigma_offset_h1 = _scalar(m1["model_general"], "rlnSigmaOffsetsAngst")
        sigma_offset_h2 = _scalar(m2["model_general"], "rlnSigmaOffsetsAngst")
        sigma_offset_per_half = [float(sigma_offset_h1), float(sigma_offset_h2)]
        sigma_offset_avg = 0.5 * (sigma_offset_per_half[0] + sigma_offset_per_half[1])
        noise_h1 = _read_model_noise_variance(m1, image_shape=(int(ds_grid), int(ds_grid)))
        noise_h2 = _read_model_noise_variance(m2, image_shape=(int(ds_grid), int(ds_grid)))
        direction_prior_h1 = _read_model_direction_prior(model_paths[0], m1)
        direction_prior_h2 = _read_model_direction_prior(model_paths[1], m2)
        class_tau2 = _read_model_class_tau2(m1)

        groups_h1 = m1.get("model_groups")
        groups_h2 = m2.get("model_groups")
        scale_h1 = (
            np.asarray(groups_h1["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h1 is not None and "rlnGroupScaleCorrection" in groups_h1.columns
            else np.array([1.0])
        )
        scale_h2 = (
            np.asarray(groups_h2["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h2 is not None and "rlnGroupScaleCorrection" in groups_h2.columns
            else np.array([1.0])
        )
        group_no = (
            np.asarray(parts["rlnGroupNumber"], dtype=int)
            if "rlnGroupNumber" in parts.columns
            else np.ones(len(parts), dtype=int)
        )
        pp_scale_h1 = scale_h1[np.clip(group_no - 1, 0, len(scale_h1) - 1)]
        pp_scale_h2 = scale_h2[np.clip(group_no - 1, 0, len(scale_h2) - 1)]
        combined_h1 = (avg_norm_h1 / nc) * pp_scale_h1
        combined_h2 = (avg_norm_h2 / nc) * pp_scale_h2

        # Map RELION particle order to recovar's half1/half2 ordering.
        # half1_idx / half2_idx are row positions in RECOVAR's input STAR,
        # Match the complete ``(index, stack path)`` identity. Numeric stack
        # indices can repeat across multi-stack real-data STAR files.
        if particle_names is None:
            particle_identities = None
        else:
            particle_identities = [
                relion_metadata._relion_image_identity(name, label="RECOVAR input STAR")
                for name in particle_names
            ]
            if len(set(particle_identities)) != len(particle_identities):
                raise ValueError("RECOVAR input contains duplicate rlnImageName/stack identities")

        def _to_half(values, half_idx):
            rows = np.asarray(half_idx, dtype=np.int64)
            if particle_identities is None:
                return np.asarray(values, dtype=np.float32)[rows]
            identities = [particle_identities[int(row)] for row in rows]
            missing = sorted({identity for identity in identities if identity not in replay_identity_rows})
            if missing:
                preview = ", ".join(f"{index}@{stack}" for index, stack in missing[:8])
                raise ValueError(
                    f"RELION replay STAR is missing {len(missing)} RECOVAR particle identities "
                    f"(preview: {preview})"
                )
            return np.asarray(
                [values[replay_identity_rows[identity]] for identity in identities],
                dtype=np.float32,
            )

        corr_h1 = _to_half(combined_h1, half1_idx)
        corr_h2 = _to_half(combined_h2, half2_idx)
        scale_corr_h1 = _to_half(pp_scale_h1, half1_idx)
        scale_corr_h2 = _to_half(pp_scale_h2, half2_idx)

        trans_h1 = None
        trans_h2 = None
        if "rlnOriginXAngst" in parts.columns and "rlnOriginYAngst" in parts.columns:
            offsets = np.stack(
                [
                    np.asarray(parts["rlnOriginXAngst"], dtype=np.float64) / float(ds_voxel),
                    np.asarray(parts["rlnOriginYAngst"], dtype=np.float64) / float(ds_voxel),
                ],
                axis=1,
            )
            trans_h1 = _to_half(offsets, half1_idx)
            trans_h2 = _to_half(offsets, half2_idx)
        elif "rlnOriginX" in parts.columns and "rlnOriginY" in parts.columns:
            offsets = np.stack(
                [
                    np.asarray(parts["rlnOriginX"], dtype=np.float64),
                    np.asarray(parts["rlnOriginY"], dtype=np.float64),
                ],
                axis=1,
            )
            trans_h1 = _to_half(offsets, half1_idx)
            trans_h2 = _to_half(offsets, half2_idx)

        angle_cols = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
        rot_h1 = None
        rot_h2 = None
        euler_h1 = None
        euler_h2 = None
        if all(col in parts.columns for col in angle_cols):
            eulers = np.stack([np.asarray(parts[col], dtype=np.float64) for col in angle_cols], axis=1)
            rotations = utils.R_from_relion(eulers, degrees=True).astype(np.float32)
            rot_h1 = _to_half(rotations, half1_idx)
            rot_h2 = _to_half(rotations, half2_idx)
            euler_h1 = _to_half(eulers, half1_idx)
            euler_h2 = _to_half(eulers, half2_idx)

        override_k = {
            "translation_sigma_angstrom": sigma_offset_avg,
            "translation_sigma_angstrom_per_half": sigma_offset_per_half,
            "previous_best_translations": [trans_h1, trans_h2],
            "previous_best_rotations": [rot_h1, rot_h2],
            "previous_best_rotation_eulers": [euler_h1, euler_h2],
        }
        if noise_h1 is not None and noise_h2 is not None:
            override_k["noise_variance"] = relion_mpi_process_start_scoring_noise_pair(
                noise_h1,
                noise_h2,
                # RELION performs this broadcast once in MPI initialise().
                # Later uninterrupted iterations update each follower's noise
                # independently, so only replay slot 0 is process-start state.
                split_random_halves=(
                    bool(process_start_noise_broadcast)
                    and recovar_iter == 0
                    and model_paths[0] != model_paths[1]
                ),
            )
        if direction_prior_h1 is not None and direction_prior_h2 is not None:
            override_k["direction_prior"] = [direction_prior_h1, direction_prior_h2]
        if class_tau2 is not None and class_tau2.shape[0] > 1:
            override_k["class_tau2"] = class_tau2
        elif class_tau2 is not None and include_k1_mean_variance:
            override_k["mean_variance"] = np.asarray(
                utils.make_radial_image(
                    class_tau2[0],
                    (int(ds_grid), int(ds_grid), int(ds_grid)),
                    extend_last_frequency=True,
                ),
                dtype=np.float64,
            ).reshape(-1)
        if include_normcorr:
            override_k["image_corrections"] = [corr_h1, corr_h2]
            override_k["serialized_scale_corrections"] = [scale_corr_h1, scale_corr_h2]
            if include_k1_scoring_scale:
                if model_paths[0] == model_paths[1]:
                    raise ValueError(
                        "exact K=1 scoring-scale replay requires distinct half-specific "
                        f"model STARs; got shared source {model_paths[0]}"
                    )
                override_k["scoring_scale_corrections"] = [scale_corr_h1, scale_corr_h2]
        overrides[recovar_iter] = override_k
        if include_normcorr:
            logger.info(
                "Replay override recovar iter %d: image_corr means=(%s, %s), serialized_scale_corr means=(%s, %s), "
                "sigma_offset=(half1 %.4f Å, half2 %.4f Å, mean %.4f Å)",
                recovar_iter + 1,
                _format_replay_mean_for_log(corr_h1),
                _format_replay_mean_for_log(corr_h2),
                _format_replay_mean_for_log(scale_corr_h1),
                _format_replay_mean_for_log(scale_corr_h2),
                sigma_offset_per_half[0],
                sigma_offset_per_half[1],
                sigma_offset_avg,
            )
        else:
            logger.info(
                "Replay override recovar iter %d: sigma_offset=(half1 %.4f Å, half2 %.4f Å, mean %.4f Å) "
                "(normcorr replay disabled)",
                recovar_iter + 1,
                sigma_offset_per_half[0],
                sigma_offset_per_half[1],
                sigma_offset_avg,
            )

    return overrides



def _format_replay_mean_for_log(values) -> str:
    arr = np.asarray(values)
    if arr.size == 0:
        return "empty"
    return f"{float(arr.mean()):.4f}"


def _load_final_replay_reference_maps(relion_dir, source_iteration, volume_shape):
    """Load the exact K=1 half references consumed by RELION finalization."""
    from recovar.core import fourier_transform_utils
    from recovar.utils.helpers import load_relion_volume

    relion_dir = Path(relion_dir)
    references = []
    for half_number in (1, 2):
        map_path = relion_dir / (
            f"run_it{int(source_iteration):03d}_half{half_number}_class001.mrc"
        )
        if not map_path.is_file():
            raise ValueError(
                "diagnostic final-only reference substitution is missing "
                f"{map_path}"
            )
        reference_real = np.asarray(load_relion_volume(str(map_path)), dtype=np.float32)
        if tuple(reference_real.shape) != tuple(volume_shape):
            raise ValueError(
                f"diagnostic final-only reference {map_path} has shape "
                f"{reference_real.shape}, expected {tuple(volume_shape)}"
            )
        references.append(
            jnp.asarray(fourier_transform_utils.get_dft3(reference_real).reshape(-1))
        )
        logger.info(
            "Diagnostic final-only reference half %d <- %s",
            half_number,
            map_path,
        )
    return references


def _resolve_final_replay_source_iteration(
    *, configured_max_iter, explicit_source_iteration, complete_iterations
):
    """Bind a finite RELION oracle to one exact last-numbered boundary."""
    complete = sorted({int(value) for value in complete_iterations})
    if not complete:
        raise ValueError("final replay oracle has no complete numbered RELION states")
    source_iteration = (
        max(complete)
        if explicit_source_iteration is None
        else int(explicit_source_iteration)
    )
    if source_iteration not in complete:
        raise ValueError(
            f"requested final replay source iteration {source_iteration} is not a complete oracle state; "
            f"available={complete}"
        )
    if source_iteration > int(configured_max_iter):
        raise ValueError(
            f"final replay source iteration {source_iteration} exceeds configured max_iter={configured_max_iter}"
        )
    missing_prefix = sorted(set(range(0, source_iteration + 1)) - set(complete))
    if missing_prefix:
        raise ValueError(
            f"final replay oracle is not contiguous through iteration {source_iteration}; missing={missing_prefix}"
        )
    return source_iteration


def _complete_relion_numbered_state_iterations(relion_dir):
    """Return iterations with data, sampling, and half/shared model state."""
    import re

    relion_dir = Path(relion_dir).resolve()
    complete = []
    for data_path in relion_dir.glob("run_it[0-9][0-9][0-9]_data.star"):
        match = re.fullmatch(r"run_it([0-9]{3})_data\.star", data_path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        sampling = relion_dir / f"run_it{iteration:03d}_sampling.star"
        half_models = (
            relion_dir / f"run_it{iteration:03d}_half1_model.star",
            relion_dir / f"run_it{iteration:03d}_half2_model.star",
        )
        shared_model = relion_dir / f"run_it{iteration:03d}_model.star"
        if sampling.is_file() and (all(path.is_file() for path in half_models) or shared_model.is_file()):
            complete.append(iteration)
    return sorted(complete)
