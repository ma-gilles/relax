"""A RELION run as the input source of the numbered iterations (code rule 15).

Each numbered iteration takes from RELION what the run supplies: its override slot (the numbered STAR
replay's per-iteration state, ``--relion_init_dir``'s run_it000 state, captured projectors) and, while the
STAR replay is live (``--perturb_replay_relion_dir`` up to ``perturb_replay_max_iter``), the sampling
controls and direction priors of its numbered STAR files. ``relax.parity.relion_replay`` reads and
installs them. A frozen boundary (``--frozen-boundary-dir``) supplies a sealed sampling state, its
``RefinementState`` fields and the scoring state it checks before the first iteration
(``relax.parity.frozen_boundary``); a state-swap probe (``--state-swap-*``) swaps components of the
replayed state back to the run's own (``relax.parity.state_swap_runtime``).

The command resolves what to replay into a ``RelionReplay`` and builds the source
(``RelionReplaySource.for_run``); the run's options hold none of it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from relax.parity.frozen_boundary import restore_diagnostic_frozen_boundary_state
from relax.parity.relion_replay import (
    apply_final_replay_state,
    apply_iter_replay_overrides,
    apply_optimiser_convergence_replay,
    class_tau2_replay,
    install_sealed_sampling,
    read_optimiser_accuracy_replay,
    replay_class_relion_references,
    replay_k1_relion_references,
    resolve_replay_random_perturbation,
    restore_convergence_state_from_replay_restart,
    sealed_rotation_ids_for_scoring,
    sealed_sampling_base_grids,
    select_final_sampling_star,
)
from relax.parity.state_swap_runtime import apply_state_swap_probe, snapshot_state_swap_inputs
from relax.refinement.image_size_plans import CoarseImageSize
from relax.refinement.ports import ClassTau2, FinalState, InputSource, NumberedState, ScoringState, StartSeeds
from relax.refinement.reference_state import class_mixture_from_weights
from relax.refinement.setup_checks import has_numbered_replay_iteration_overrides
from relax.refinement.sigma_offset import SigmaOffset
from relax.refinement.trial_grids import CoarseGrids, FinalSamplingSettings, native_final_sampling_settings
from relax.runtime.env_flags import parse_env_true_flag
from relax.sampling import grids as sampling_grids
from relax.star.metadata import (
    read_relion_sampling_metadata,
)

# The controller's log: what the replay installs is logged under its name, as before.
logger = logging.getLogger("relax.refinement.iteration_loop")
# The replay readers moved here from relion_replay log under its name, so their log rows keep it.
_replay_logger = logging.getLogger("relax.diagnostics.relion_replay")

FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE_ENV = "RELAX_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE"
FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV = "RELAX_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE"


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


def _frozen_scoring_state_arrays(
    *,
    means,
    mean_variance,
    mean_variance_per_half=None,
    relion_half_inputs,
    noise_variance_per_half,
    current_sigma_offset_angstrom_per_half,
    direction_priors,
    experiment_datasets=None,
    sealed_sampling_state=None,
    sealed_scoring_context=None,
):
    """Materialize every K=1 scoring primitive owned by a frozen restart."""

    pair_fields = {
        "mean": means,
        "previous_best_rotation_eulers": [particle_half.rotation_eulers for particle_half in relion_half_inputs],
        "previous_best_translations": [particle_half.translations for particle_half in relion_half_inputs],
        "image_corrections": [particle_half.image_corrections for particle_half in relion_half_inputs],
        "scale_corrections": [particle_half.scale_corrections for particle_half in relion_half_inputs],
        "noise_variance": noise_variance_per_half,
        "direction_prior": [p.values for p in direction_priors],
    }
    arrays = {}
    for field_name, values in pair_fields.items():
        if values is None or len(values) != 2:
            raise RuntimeError(
                f"Frozen scoring-state ownership requires two {field_name} arrays"
            )
        for half_index, value in enumerate(values):
            if value is None:
                raise RuntimeError(
                    "Frozen scoring-state ownership requires "
                    f"{field_name} for half {half_index + 1}"
                )
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.number):
                raise RuntimeError(
                    f"Frozen scoring-state field {field_name} half {half_index + 1} "
                    f"must be numeric, got {array.dtype}"
                )
            if not np.all(np.isfinite(array)):
                raise RuntimeError(
                    f"Frozen scoring-state field {field_name} half {half_index + 1} "
                    "contains non-finite values"
                )
            arrays[f"{field_name}.half{half_index + 1}"] = np.ascontiguousarray(array).copy()

    if mean_variance_per_half is None:
        tau2 = np.asarray(mean_variance)
        if not np.issubdtype(tau2.dtype, np.number) or not np.all(np.isfinite(tau2)):
            raise RuntimeError("Frozen scoring-state mean_variance must be a finite numeric array")
        arrays["mean_variance"] = np.ascontiguousarray(tau2).copy()
    else:
        if len(mean_variance_per_half) != 2:
            raise RuntimeError("Frozen scoring-state ownership requires two mean_variance arrays")
        for half_index, value in enumerate(mean_variance_per_half):
            tau2 = np.asarray(value)
            if not np.issubdtype(tau2.dtype, np.number) or not np.all(np.isfinite(tau2)):
                raise RuntimeError(
                    f"Frozen scoring-state mean_variance half {half_index + 1} must be finite numeric"
                )
            arrays[f"mean_variance.half{half_index + 1}"] = np.ascontiguousarray(tau2).copy()

    sigma_pair = np.asarray(current_sigma_offset_angstrom_per_half, dtype=np.float64)
    if sigma_pair.shape != (2,) or not np.all(np.isfinite(sigma_pair)):
        raise RuntimeError(
            "Frozen scoring-state ownership requires two finite translation sigmas"
        )
    arrays["translation_sigma_angstrom_per_half"] = sigma_pair
    if experiment_datasets is not None:
        if len(experiment_datasets) != 2:
            raise RuntimeError("Frozen scoring-state ownership requires two datasets")
        for half_index, dataset in enumerate(experiment_datasets):
            ctf_params = np.asarray(dataset.CTF_params)
            if not np.issubdtype(ctf_params.dtype, np.number) or not np.all(np.isfinite(ctf_params)):
                raise RuntimeError(
                    f"Frozen scoring-state CTF parameters half {half_index + 1} must be finite numeric"
                )
            arrays[f"ctf_params.half{half_index + 1}"] = np.ascontiguousarray(ctf_params).copy()
    if sealed_sampling_state is not None:
        for key, value in sorted(sealed_sampling_state.items()):
            if isinstance(value, np.ndarray):
                array = np.asarray(value)
            elif isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                array = np.asarray(value)
            else:
                array = np.frombuffer(str(value).encode("utf-8"), dtype=np.uint8)
            arrays[f"sampling.{key}"] = np.ascontiguousarray(array).copy()
    if sealed_scoring_context is not None:
        context_bytes = json.dumps(
            sealed_scoring_context,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        arrays["sealed_scoring_context_json"] = np.frombuffer(context_bytes, dtype=np.uint8).copy()
    return arrays


def _assert_frozen_scoring_state_unchanged(expected, actual):
    """Fail closed if replay or setup mutated a sealed scoring primitive."""

    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "Frozen scoring-state fields changed: "
            f"missing={missing}, extra={extra}"
        )
    hashes = {}
    for field_name in sorted(expected):
        expected_array = np.asarray(expected[field_name])
        actual_array = np.asarray(actual[field_name])
        if expected_array.shape != actual_array.shape:
            raise RuntimeError(
                f"Frozen scoring-state field {field_name} shape changed: "
                f"{expected_array.shape} != {actual_array.shape}"
            )
        if expected_array.dtype != actual_array.dtype:
            raise RuntimeError(
                f"Frozen scoring-state field {field_name} dtype changed: "
                f"{expected_array.dtype} != {actual_array.dtype}"
            )
        if not np.all(np.isfinite(actual_array)):
            raise RuntimeError(
                f"Frozen scoring-state field {field_name} contains non-finite values"
            )
        if not np.array_equal(expected_array, actual_array):
            raise RuntimeError(
                f"Frozen scoring-state field {field_name} was overwritten before scoring"
            )
        hashes[field_name] = {
            "dtype": str(actual_array.dtype),
            "shape": list(actual_array.shape),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(actual_array).view(np.uint8)
            ).hexdigest(),
        }
    return hashes


@dataclass(frozen=True)
class RelionReplay:
    """What a run replays from a RELION run, as the command resolved it.

    ``perturb_replay_relion_dir`` (``--perturb_replay_relion_dir``): the run whose numbered sampling, model and
    optimiser STAR files set each iteration's sampling, priors and convergence controls, with their file
    ``perturb_replay_relion_prefix``, up to ``perturb_replay_max_iter`` iterations (None: all; 0: none after the
    start-up state; ``--replay-override-max-iter`` of scripts/run_multi_iter_parity.py), its perturbation
    precision and the iterations whose saved RNG state a restart resumes from.
    ``replay_iteration_overrides``: one slot (a dict, or None) per numbered iteration, the state installed in
    it (the numbered STAR replay's, ``--relion_init_dir``'s run_it000 state, captured projectors).
    ``final_replay_*`` and ``final_sampling_replay_relion_dir``: a final-only replay
    (``--final-replay-relion-dir``). The last two switches force (``RELAX_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE``)
    or forbid (``RELAX_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE``) replaying the last numbered state in
    the final pass; ``from_environ``, which the command calls, reads them.
    A frozen boundary (``--frozen-boundary-dir``): its ``sealed_sampling_state`` and ``sealed_scoring_context``,
    and its ``frozen_refinement_state_fields`` (with them, the source checks, right before the first iteration
    scores, that the scoring state is the one bound before it). ``start_seeds``: the first FSC, Pmax, high-FSC
    flag and RELION increment the first iteration starts from (a frozen boundary's, or the completed RELION
    iteration's of scripts/run_multi_iter_parity.py; ``relax.refinement.ports.StartSeeds``). ``state_swap_probe``: the
    state-swap probe's settings (``relax.parity.state_swap_probe.build_state_swap_probe``).
    ``preserve_initial_direction_prior``: the run keeps its start-up direction priors (a frozen boundary's) instead
    of reloading them from the STAR replay's model files.
    ``follower_topology``: an MPI RELION run's followers, its captured dispatch schedule (which follower scored
    which particle in each iteration, ``--relion-dispatch-schedule``) and follower-scale replay
    (``relax.refinement.follower_scale.PreparedFollowerTopology``; None or no followers: none).
    """

    perturb_replay_relion_dir: str | None = None
    perturb_replay_relion_prefix: str = "run"
    perturb_replay_max_iter: int | None = None
    perturb_replay_restart_state_iterations: tuple[int, ...] = ()
    replay_iteration_overrides: Any | None = None
    final_replay_override: Any | None = None
    final_replay_reference_maps: Any | None = None
    final_replay_source_iteration: int | None = None
    final_sampling_replay_relion_dir: str | None = None
    # From the environment, by from_environ (the command calls it).
    final_all_data_replay_last_numbered_state: bool = False
    final_all_data_disable_replay_last_numbered_state: bool = False
    sealed_sampling_state: Any | None = None
    sealed_scoring_context: Any | None = None
    frozen_refinement_state_fields: Any | None = None
    start_seeds: StartSeeds | None = None
    state_swap_probe: dict | None = None
    preserve_initial_direction_prior: bool = False
    follower_topology: Any | None = None

    def __post_init__(self):
        if self.perturb_replay_max_iter is not None and self.perturb_replay_max_iter < 0:
            raise ValueError(f"perturb_replay_max_iter must be non-negative, got {self.perturb_replay_max_iter!r}")
        iterations = tuple(sorted({int(value) for value in self.perturb_replay_restart_state_iterations}))
        if any(value < 0 for value in iterations):
            raise ValueError("perturbation replay restart-state iterations must be non-negative")
        if iterations and self.perturb_replay_relion_dir is None:
            raise ValueError("perturbation replay restart-state iterations require perturb_replay_relion_dir")
        object.__setattr__(self, "perturb_replay_restart_state_iterations", iterations)

    @classmethod
    def from_frozen_boundary(cls, frozen_boundary, **fields) -> RelionReplay:
        """``fields`` and what a frozen boundary (``--frozen-boundary-dir``; None: none) replays: its
        ``RefinementState`` fields (whose presence also makes the source check that the scoring state is
        unchanged before the first iteration), its start seeds (first FSC, Pmax, high-FSC flag and RELION
        increment), its start-up direction priors, kept, and (its fixed diagnostic arm only) its sealed sampling
        state and scoring context."""
        if frozen_boundary is None:
            return cls(**fields)
        return cls(
            **fields,
            preserve_initial_direction_prior=True,
            frozen_refinement_state_fields=frozen_boundary.refinement_state_fields,
            start_seeds=StartSeeds(
                fsc=frozen_boundary.fsc, ave_pmax=frozen_boundary.ave_pmax,
                has_high_fsc_at_limit=frozen_boundary.has_high_fsc_at_limit,
                relion_incr_size=frozen_boundary.relion_incr_size,
            ),
            sealed_sampling_state=frozen_boundary.sampling_state if frozen_boundary.fixed_diagnostic_arm else None,
            sealed_scoring_context=(
                {
                    "schema": frozen_boundary.schema,
                    "completed_relion_iteration": frozen_boundary.completed_relion_iteration,
                    "consumer_relion_iteration": frozen_boundary.consumer_relion_iteration,
                    "source_sha256": frozen_boundary.source_sha256,
                    "source_roles": frozen_boundary.source_roles,
                    "runtime_config": frozen_boundary.runtime_config,
                    "map_lineage": frozen_boundary.map_lineage,
                }
                if frozen_boundary.fixed_diagnostic_arm
                else None
            ),
        )

    @classmethod
    def from_environ(cls, frozen_boundary=None, **fields) -> RelionReplay:
        """``from_frozen_boundary(frozen_boundary, **fields)`` with the final pass's last-numbered-state replay
        switches read from the environment (``RELAX_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE``,
        ``..._DISABLE_REPLAY_LAST_NUMBERED_STATE``)."""
        return cls.from_frozen_boundary(
            frozen_boundary,
            **fields,
            final_all_data_replay_last_numbered_state=parse_env_true_flag(FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE_ENV),
            final_all_data_disable_replay_last_numbered_state=parse_env_true_flag(
                FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV
            ),
        )

    @property
    def replays_relion_state(self) -> bool:
        """Whether the run's state comes from a RELION run: override slots, a STAR replay directory, a final-pass
        replay (its override, reference maps or sampling directory), the forced last-numbered-state replay, a
        frozen boundary (sealed sampling state, sealed scoring context, state fields), start seeds or a state-swap
        probe.
        The one field list: ``replays`` and the consistency refusal (``InputSource.replays_relion_state``) read it."""
        return any(
            value is not None
            for value in (
                self.replay_iteration_overrides, self.perturb_replay_relion_dir, self.final_replay_override,
                self.final_replay_reference_maps, self.final_sampling_replay_relion_dir, self.sealed_sampling_state,
                self.sealed_scoring_context, self.frozen_refinement_state_fields, self.start_seeds,
                self.state_swap_probe,
            )
        ) or self.final_all_data_replay_last_numbered_state

    @property
    def replays(self) -> bool:
        """Whether there is anything to replay: RELION state (``replays_relion_state``) or a follower topology
        (followers or a captured dispatch schedule), which emulates RELION's MPI layout on relax's own state."""
        return self.replays_relion_state or (
            self.follower_topology is not None
            and (int(self.follower_topology.n_followers or 0) > 0 or self.follower_topology.replay is not None)
        )


class RelionReplaySource(InputSource):
    """Replays ``replay`` (a RELION run's state) into the numbered iterations and the final pass of a run with
    ``options``."""

    def __init__(self, replay: RelionReplay, options):
        self.replay = replay
        self.options = options
        self._cutoff_announced = False
        self._state_swap_snapshot = None
        self._bound_scoring_state = None
        # The sampling record (a RELION STAR's or a sealed state's) the last numbered_state installed, or None.
        self._sampling_meta = None
        # The STAR directory of the last iteration asked about: the one the final pass replays from.
        self._live_directory = replay.perturb_replay_relion_dir
        if replay.perturb_replay_restart_state_iterations:
            logger.info(
                "Perturbation replay restart provenance: saved-state iterations=%s",
                list(replay.perturb_replay_restart_state_iterations),
            )

    @classmethod
    def for_run(cls, replay: RelionReplay | None, options) -> InputSource:
        """The source of a run with ``options`` that replays ``replay``: the native source when there is
        nothing to replay."""
        return cls(replay, options) if replay is not None and replay.replays else InputSource()

    @property
    def relion_replay(self) -> RelionReplay:
        return self.replay

    @property
    def sealed_sampling_state(self):
        return self.replay.sealed_sampling_state

    @property
    def sealed_scoring_context(self):
        return self.replay.sealed_scoring_context

    @property
    def start_seeds(self):
        return StartSeeds() if self.replay.start_seeds is None else self.replay.start_seeds

    @property
    def swaps_state(self):
        return self.replay.state_swap_probe is not None

    @property
    def replays_relion_trajectory(self):
        return self.replay.perturb_replay_relion_dir is not None or self.replay.replay_iteration_overrides is not None

    @property
    def starts_from_frozen_boundary(self):
        return (
            self.replay.sealed_sampling_state is not None or self.replay.frozen_refinement_state_fields is not None
        )

    @property
    def follower_topology(self):
        return self.replay.follower_topology

    def replays_relion_state(self):
        return self.replay.replays_relion_state

    def restore_boundary_state(self, state):
        """A frozen boundary's ``RefinementState`` fields, installed over the initial ones (the start FSC's
        resolution among them)."""
        if self.replay.frozen_refinement_state_fields is not None:
            restore_diagnostic_frozen_boundary_state(state, self.replay.frozen_refinement_state_fields)

    def initial_coarse_grids(self, *, initialized_healpix_order, voxel_size, symmetry):
        """A sealed sampling state's captured grids, checked against the initialized HEALPix order (None without
        one: the run's own)."""
        if self.replay.sealed_sampling_state is None:
            return None
        return build_sealed_initial_coarse_grids(
            self.replay.sealed_sampling_state,
            initialized_healpix_order=initialized_healpix_order,
            voxel_size=voxel_size,
            symmetry=symmetry,
            dtype=self.options.precision.rotation_real_dtype,
            log=logger,
        )

    def coarse_grids(self, iteration, grids, state, *, voxel_size, dtype):
        """When RELION's numbered sampling STAR set this iteration's sampling (not a sealed state's) at the grid's
        HEALPix order, the translation grid of its replayed range and step, rebuilt where it changed."""
        if (
            self._star_directory(iteration) is None
            or self.replay.sealed_sampling_state is not None
            or state.healpix_order != grids.rotation_grid.healpix_order
        ):
            return grids
        base_translations = grids.base_translations
        current_translations = grids.translations
        # Translation params may have changed under replay without an
        # hp_order bump. Regenerate the translation grid to match RELION.
        _new_t_source = sampling_grids.relion_base_translation_grid(
            state.translation_range,
            state.translation_step,
            n_classes=self.options.k_class.n_classes,
            voxel_size=voxel_size,
        )
        _new_t = jnp.asarray(_new_t_source, dtype=dtype)
        if _new_t.shape != base_translations.shape or not jnp.allclose(
            _new_t,
            np.asarray(base_translations, dtype=dtype),
        ):
            current_translations = _new_t
            base_translations = _new_t_source
            logger.info(
                "Replay: regenerated translation grid: %d translations (range=%.2f px, step=%.2f px)",
                current_translations.shape[0],
                state.translation_range,
                state.translation_step,
            )
        return CoarseGrids(grids.rotation_grid, base_translations, current_translations)

    def adaptive_coarse_size(self, plan, *, model_size):
        """A sealed sampling state's exact pass-1 width, at most the model's current size."""
        if self.replay.sealed_sampling_state is None:
            return plan
        coarse_size = int(self.replay.sealed_sampling_state["coarse_size"])
        if coarse_size > model_size:
            raise ValueError(
                "sealed sampling coarse_size exceeds active current_size: "
                f"coarse={coarse_size} current={model_size}"
            )
        logger.info("Frozen-boundary v3 directly owns adaptive pass-1 coarse_size=%d", coarse_size)
        return CoarseImageSize(size=coarse_size, angular_step_deg=plan.angular_step_deg)

    def scoring_rotation_ids(self, trial_grid, *, use_local):
        """A sealed global grid's captured rotation ids."""
        return sealed_rotation_ids_for_scoring(self.replay.sealed_sampling_state, trial_grid, use_local=use_local)

    def _swaps_at(self, iteration):
        probe = self.replay.state_swap_probe
        return probe is not None and int(probe.get("iteration", -1)) == int(iteration)

    def state_swap_snapshot(self, iteration, scoring):
        """The run's own state at the probe's target iteration, before RELION's is installed (a copy)."""
        if self._swaps_at(iteration):
            self._state_swap_snapshot = snapshot_state_swap_inputs(**_state_swap_keywords(scoring))

    def scoring_references(self, iteration, reference_model, *, volume_shape):
        """RELION's maps at the probe's target iteration when the probe replays references, else the model's."""
        replay = replay_k1_relion_references if int(self.options.k_class.n_classes) == 1 else replay_class_relion_references
        return replay(
            reference_model, self.options, probe=self.replay.state_swap_probe, iteration=iteration,
            replay_dir=self._star_directory(iteration), replay_prefix=self.replay.perturb_replay_relion_prefix,
            volume_shape=volume_shape,
        )

    def swapped_state(self, iteration, scoring, *, volume_shape):
        """The probe's variant of the replayed state at its target iteration: its components restored from the
        run's own snapshot. The ``state`` and ``halves`` it restores are updated in place."""
        if not self._swaps_at(iteration):
            return None
        swapped = apply_state_swap_probe(
            probe=self.replay.state_swap_probe,
            iteration=iteration,
            recovar_snapshot=self._state_swap_snapshot,
            volume_shape=volume_shape,
            **_state_swap_keywords(scoring),
        )
        return scoring._replace(
            current_size=swapped.cs,
            reference_model=swapped.reference_model,
            noise_model=swapped.noise_model,
            previous_best_rotations=swapped.previous_best_rotations,
            sigma_offset=SigmaOffset(
                swapped.current_sigma_offset_angstrom, swapped.current_sigma_offset_angstrom_per_half,
            ),
            direction_priors=swapped.direction_priors,
        )

    def _frozen_arrays(self, arrays):
        return _frozen_scoring_state_arrays(
            means=arrays.reference_model.maps,
            mean_variance=arrays.reference_model.tau2,
            mean_variance_per_half=arrays.reference_model.tau2_per_half,
            relion_half_inputs=arrays.halves,
            noise_variance_per_half=arrays.noise_model.variance_per_half,
            current_sigma_offset_angstrom_per_half=arrays.sigma_offset.per_half_angstrom,
            direction_priors=arrays.direction_priors,
            experiment_datasets=arrays.experiment_datasets,
            sealed_sampling_state=self.replay.sealed_sampling_state,
            sealed_scoring_context=self.replay.sealed_scoring_context,
        )

    def scoring_state_bound(self, arrays):
        """A frozen boundary binds the scoring state it checks before the first iteration (K=1 only)."""
        if self.replay.frozen_refinement_state_fields is None:
            return
        if int(self.options.k_class.n_classes) > 1:
            raise RuntimeError("Frozen scoring-state immutability assertion currently supports K=1 only")
        self._bound_scoring_state = self._frozen_arrays(arrays)

    def scoring_state_checked(self, iteration, arrays):
        """Right before the first iteration scores: the bound scoring state is unchanged (its digest)."""
        if self._bound_scoring_state is None or iteration != 0:
            return None
        digest = _assert_frozen_scoring_state_unchanged(self._bound_scoring_state, self._frozen_arrays(arrays))
        logger.info(
            "Frozen scoring-state ownership verified immediately before physical iteration %d scoring",
            int(self.options.schedule.init_relion_iteration) + iteration + 1,
        )
        return digest

    def _slot(self, iteration: int):
        slots = self.replay.replay_iteration_overrides
        return slots[iteration] if slots is not None and iteration < len(slots) else None

    def _star_directory(self, iteration: int):
        if self.replay.perturb_replay_relion_dir is None or _past_perturb_replay_max_iter(iteration, self.replay.perturb_replay_max_iter):
            return None
        return self.replay.perturb_replay_relion_dir

    def relion_run_directory(self, iteration):
        """The STAR replay's directory up to ``perturb_replay_max_iter`` (announced once when it ends), else None."""
        directory = self._live_directory = self._star_directory(iteration)
        if directory is None and self.replay.perturb_replay_relion_dir is not None and not self._cutoff_announced:
            self._cutoff_announced = True
            logger.info(
                "Replay override: disabling RELION per-iteration STAR replay from "
                "iteration %d onward (--replay-override-max-iter %d)",
                iteration + 1,
                self.replay.perturb_replay_max_iter,
            )
        return directory

    def final_state(self, inputs, *, means, numbered_iteration_count, halves, direction_priors, healpix_order,
                    image_geometry):
        """The final pass's RELION state: a final-only replay's references and state, else the last numbered
        slot's (when the replay has numbered slots, or RELAX_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE asks and
        RELAX_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE does not forbid it). Installs the particle state
        in the ``halves`` and the priors in ``direction_priors`` in place, as apply_final_replay_state does."""
        options, replay = self.options, self.replay
        k_class_enabled = options.k_class.n_classes > 1
        final_replay_override = self.replay.final_replay_override
        sigma_offset, noise_model = inputs.sigma_offset, inputs.noise_model
        disabled = self.replay.final_all_data_disable_replay_last_numbered_state
        has_overrides = self.replay.replay_iteration_overrides is not None and len(self.replay.replay_iteration_overrides) > 0
        join_means = _prepare_final_replay_references(
            replay=replay,
            diagnostic_override=final_replay_override,
            numbered_iteration_count=numbered_iteration_count,
            means=means,
            final_join_means=inputs.join_means,
            k_class_enabled=k_class_enabled,
            logger=logger,
        )
        replay_last_numbered_state = final_replay_override is not None or (
            not disabled
            and (
                self.replay.final_all_data_replay_last_numbered_state
                or has_numbered_replay_iteration_overrides(self.replay.replay_iteration_overrides)
            )
        )
        if replay_last_numbered_state:
            override_index, override = _select_final_replay_override(
                requested_index=numbered_iteration_count,
                diagnostic_override=final_replay_override,
                replay_overrides=self.replay.replay_iteration_overrides,
                has_overrides=has_overrides,
                logger=logger,
            )
            if override is not None:
                sigma_offset, noise_model = apply_final_replay_state(
                    override,
                    halves,
                    direction_priors,
                    sigma_offset=sigma_offset,
                    noise_model=noise_model,
                    n_classes=options.k_class.n_classes,
                    healpix_order=healpix_order,
                    image_shape=image_geometry.image_shape,
                    symmetry=options.symmetry.point_group,
                    dtype=options.precision.rotation_real_dtype,
                    override_index=override_index,
                    log=logger,
                )
        elif not k_class_enabled and disabled and has_overrides:
            logger.info(
                "Diagnostic %s=1: final all-data skips automatic last-numbered RELION state replay",
                FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV,
            )
        return FinalState(join_means, sigma_offset, noise_model)

    def final_sampling_settings(self, state, image_geometry, *, grid_order, relion_iteration, native):
        """RELION's final (or last numbered) sampling STAR of the final-pass replay directory, else of the live
        STAR replay; with neither, the run's own. A replay without the STAR leaves the final grid unperturbed."""
        options, parity = self.options, self.options.parity
        last_numbered_iteration = relion_iteration - 1
        active_replay_dir = self._live_directory
        replay_dir = (
            self.replay.final_sampling_replay_relion_dir
            if self.replay.final_sampling_replay_relion_dir is not None
            else active_replay_dir
        )
        if replay_dir is None:
            return native()
        star, source, candidates = select_final_sampling_star(
            replay_dir, self.replay.perturb_replay_relion_prefix,
            final_iteration=relion_iteration,
            previous_iteration=last_numbered_iteration,
            require_final_state=self.replay.replay_iteration_overrides is not None,
        )
        if star is not None:
            metadata = read_relion_sampling_metadata(star)
            replay_iteration = last_numbered_iteration if source == "last-numbered" else relion_iteration
            perturbation, perturbation_source = resolve_replay_random_perturbation(
                star_value=metadata["random_perturbation"],
                perturbation_factor=metadata["perturbation_factor"],
                relion_iteration=replay_iteration,
                replay_dir=str(replay_dir),
                replay_prefix=self.replay.perturb_replay_relion_prefix,
                explicit_seed=parity.perturb_seed,
                restart_state_iteration=_perturbation_restart_state_iteration(
                    self.replay.perturb_replay_restart_state_iterations, replay_iteration,
                ),
            )
            pixel_size = image_geometry.pixel_size_angstrom
            settings = FinalSamplingSettings(
                relion_iteration=relion_iteration,
                grid_order=grid_order,
                perturbation_order=metadata["healpix_order"],
                translation_range=metadata["offset_range"] / pixel_size,
                translation_step=metadata["offset_step"] / pixel_size,
                pixel_size_angstrom=pixel_size,
                perturbation_factor=metadata["perturbation_factor"],
                perturbation=perturbation,
                sampling_star=star,
                sampling_star_source=source,
            )
            _log_replayed_translation_grid_change(
                settings, replay_dir=replay_dir,
                replay_prefix=self.replay.perturb_replay_relion_prefix, n_classes=options.k_class.n_classes,
            )
            logger.info(
                "Perturbation replay: final all-data relion_iter=%d rp=%+.12g pf=%.3f "
                "relion_hp_order=%d offset_range=%.3f px offset_step=%.3f px source=%s/%s",
                replay_iteration, settings.random_perturbation, settings.perturbation_factor,
                settings.perturbation_order, settings.translation_range, settings.translation_step,
                source, perturbation_source,
            )
            return settings
        logger.info(
            "Perturbation replay: final all-data sampling STAR missing for relion_iter=%d (%s); "
            "leaving final trial grid unperturbed",
            relion_iteration, ", ".join(path for path, _source in candidates),
        )
        # A final-only replay directory historically applies a zero perturbation
        # when native perturbation is enabled, without advancing its RNG.
        return native_final_sampling_settings(
            state, image_geometry, options, grid_order=grid_order, relion_iteration=relion_iteration,
            perturbation=0.0 if active_replay_dir is None and parity.perturb_factor > 0 else None,
        )

    def class_tau2(self, iteration, n_classes):
        """The slot's captured Class3D prior, used when RELAX_KCLASS_REPLAY_TAU2 asks for it (validated always)."""
        shells, enabled, label = class_tau2_replay(
            iteration=iteration, n_classes=n_classes, iter_replay_override=self._slot(iteration),
            replay=self.replay, logger=logger,
        )
        return ClassTau2(shells if enabled else None, label)

    def restore_convergence_state(self, state):
        """A replay restart (init_relion_iteration > 0, no sealed sampling state) resumes RELION's convergence
        counters from its optimiser and model STAR files."""
        options = self.options
        if (
            self.replay.sealed_sampling_state is not None
            or self.replay.perturb_replay_relion_dir is None
            or int(options.schedule.init_relion_iteration) <= 0
        ):
            return False
        restore_convergence_state_from_replay_restart(state, self.replay, options.schedule.init_relion_iteration)
        return True

    def convergence_accuracy(self, iteration, accuracy):
        """RELION's numbered optimiser accuracies, while the STAR replay is live and no sealed state is."""
        return read_optimiser_accuracy_replay(
            replay_dir=self._star_directory(iteration),
            replay_prefix=self.replay.perturb_replay_relion_prefix,
            init_relion_iteration=self.options.schedule.init_relion_iteration,
            iteration=iteration,
            sealed_sampling_state=self.replay.sealed_sampling_state,
            acc_rot=accuracy.acc_rot,
            acc_trans=accuracy.acc_trans,
            convergence_acc_rot=accuracy.convergence_acc_rot,
            convergence_acc_trans=accuracy.convergence_acc_trans,
            logger=logger,
        )

    def apply_optimiser_controls(self, iteration, state, accuracy):
        """The numbered optimiser STAR's counters, changes and convergence flag, installed in ``state``."""
        if accuracy.metadata is not None:
            apply_optimiser_convergence_replay(
                state,
                metadata=accuracy.metadata,
                optimiser_star=accuracy.optimiser_star,
                optimiser_iteration=accuracy.optimiser_iteration,
                replay_dir=self._star_directory(iteration),
                replay_prefix=self.replay.perturb_replay_relion_prefix,
                logger=logger,
            )

    def random_perturbation(self, iteration):
        """The perturbation of the sampling this iteration's ``numbered_state`` installed: a sealed sampling
        state's as captured, RELION's sampling STAR's (exact from its seed where the precision allows); otherwise
        None, and the run computes its own. Only the run's own advances the run's RNG."""
        sampling_meta = self._sampling_meta
        if sampling_meta is None:
            return None
        if sampling_meta.get("sealed_v3", False):
            perturbation = float(sampling_meta["random_perturbation"])
            logger.info(
                "Perturbation replay: iter=%d rp=%+.12g pf=%.3f relion_hp_order=%d source=%s",
                iteration + 1, perturbation,
                float(sampling_meta["perturbation_factor"]),
                int(sampling_meta["healpix_order"]), "sealed_frozen_boundary_v3",
            )
            return perturbation
        parity = self.options.parity
        relion_iteration = self.options.schedule.init_relion_iteration + iteration + 1
        restart_iteration = _perturbation_restart_state_iteration(
            self.replay.perturb_replay_restart_state_iterations, relion_iteration,
        )
        perturbation, source = resolve_replay_random_perturbation(
            star_value=float(sampling_meta["random_perturbation"]),
            perturbation_factor=float(sampling_meta["perturbation_factor"]),
            relion_iteration=relion_iteration,
            replay_dir=str(self._star_directory(iteration)),
            replay_prefix=self.replay.perturb_replay_relion_prefix,
            explicit_seed=parity.perturb_seed,
            restart_state_iteration=restart_iteration,
        )
        logger.info(
            "Perturbation replay: iter=%d rp=%+.12g pf=%.3f relion_hp_order=%d source=%s",
            iteration + 1, perturbation,
            float(sampling_meta["perturbation_factor"]),
            int(sampling_meta["healpix_order"]), source,
        )
        return perturbation

    def numbered_state(self, iteration, inputs, *, state, halves, direction_priors, image_geometry):
        """RELION's sampling controls, priors, particle state, noise, tau2 and class weights for this iteration.

        A sealed sampling state installs the sampling controls first (and the STAR replay's are not read); the
        sampling record installed is kept for this iteration's ``random_perturbation``.
        Updates ``state``'s sampling controls, the ``halves``' poses and corrections and the
        ``direction_priors`` list in place, as ``apply_iter_replay_overrides`` does.
        """
        options = self.options
        sealed = self.replay.sealed_sampling_state
        sealed_meta = None
        if sealed is not None:
            current_size, prior_translations, sealed_meta = install_sealed_sampling(
                state, sealed, iteration=iteration, image_geometry=image_geometry, dtype=options.precision.rotation_real_dtype,
            )
            inputs = inputs._replace(current_size=current_size, prior_translations=prior_translations)
        slot = self._slot(iteration)
        result = apply_iter_replay_overrides(
            iter_replay_override=slot,
            perturb_replay_relion_dir=None if sealed is not None else self._star_directory(iteration),
            perturb_replay_relion_prefix=self.replay.perturb_replay_relion_prefix,
            init_relion_iteration=options.schedule.init_relion_iteration,
            iteration=iteration,
            state=state,
            cs=inputs.current_size,
            image_geometry=image_geometry,
            n_classes=int(options.k_class.n_classes),
            relion_half_inputs=halves,
            previous_best_rotations=inputs.previous_best_rotations,
            noise_model=inputs.noise_model,
            current_sigma_offset_angstrom=inputs.sigma_offset.shared_angstrom,
            current_sigma_offset_angstrom_per_half=inputs.sigma_offset.per_half_angstrom,
            direction_priors=direction_priors,
            preserve_existing_direction_prior=self.replay.preserve_initial_direction_prior,
            dtype=options.precision.rotation_real_dtype,
            symmetry=options.symmetry.point_group,
        )
        mean_variance = inputs.mean_variance
        replay_mean_variance = None if slot is None else slot.get("mean_variance")
        if replay_mean_variance is not None:
            replay_mean_variance = np.asarray(replay_mean_variance, dtype=np.float64).reshape(-1)
            expected_shape = tuple(mean_variance.shape)
            if replay_mean_variance.shape != expected_shape:
                raise ValueError(
                    f"K=1 replay mean_variance shape mismatch: expected {expected_shape}, got {replay_mean_variance.shape}"
                )
            mean_variance = jnp.asarray(replay_mean_variance)
            logger.info("Replay override: K=1 tau2/mean_variance <- model.star")
        class_mixture = inputs.class_mixture
        if int(options.k_class.n_classes) > 1 and result.class_weights is not None:
            class_mixture = class_mixture_from_weights(np.asarray(result.class_weights, dtype=np.float64))
            logger.info(
                "Replay override: class priors <- direction-prior row sums (%s)",
                ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_mixture.weights)),
            )
        self._sampling_meta = sealed_meta if sealed is not None else result.replay_meta
        return NumberedState(
            current_size=result.cs,
            noise_model=result.noise_model,
            sigma_offset=SigmaOffset(result.current_sigma_offset_angstrom, result.current_sigma_offset_angstrom_per_half),
            previous_best_rotations=result.previous_best_rotations,
            mean_variance=mean_variance,
            class_mixture=class_mixture,
            prior_translations=inputs.prior_translations if sealed is not None else result.prior_translations,
            sampling_healpix_order=(
                None if self._sampling_meta is None else int(self._sampling_meta["healpix_order"])
            ),
        )


def build_sealed_initial_coarse_grids(
    sealed_sampling_state,
    *,
    initialized_healpix_order,
    voxel_size,
    symmetry: str,
    dtype,
    log: logging.Logger,
) -> CoarseGrids:
    """Materialize and validate a schema-v3 sealed initial sampling grid, its operands in ``dtype`` (the run's
    ``precision.rotation_real_dtype``)."""

    rotations, rotation_eulers, current_translations = sealed_sampling_base_grids(
        sealed_sampling_state,
        voxel_size_angstrom=voxel_size,
        dtype=dtype,
    )
    healpix_order = int(sealed_sampling_state["healpix_order_original"])
    if healpix_order != int(initialized_healpix_order):
        raise ValueError(
            "sealed sampling HEALPix order does not match initialized boundary: "
            f"sealed={healpix_order} init={initialized_healpix_order}"
        )
    log.info(
        "Frozen-boundary v3 directly materialized %d Euler rows and %d translations",
        int(rotation_eulers.shape[0]),
        int(current_translations.shape[0]),
    )
    return CoarseGrids(
        rotation_grid=sampling_grids.RotationGrid(
            rotations=rotations, rotation_eulers=rotation_eulers,
            healpix_order=healpix_order, symmetry=symmetry,
        ),
        base_translations=np.asarray(current_translations, dtype=np.float64),
        translations=current_translations,
    )


def _state_swap_keywords(scoring: ScoringState) -> dict:
    """The state-swap probe's keywords for the run's ``scoring`` state."""
    return dict(
        state=scoring.state,
        cs=scoring.current_size,
        reference_model=scoring.reference_model,
        noise_model=scoring.noise_model,
        relion_half_inputs=scoring.halves,
        previous_best_rotations=scoring.previous_best_rotations,
        current_sigma_offset_angstrom=scoring.sigma_offset.shared_angstrom,
        current_sigma_offset_angstrom_per_half=scoring.sigma_offset.per_half_angstrom,
        direction_priors=scoring.direction_priors,
    )


def _log_replayed_translation_grid_change(settings, *, replay_dir, replay_prefix, n_classes):
    if settings.sampling_star_source != "final":
        return
    numbered_path = os.path.join(
        replay_dir, f"{replay_prefix}_it{settings.relion_iteration - 1:03d}_sampling.star",
    )
    if not os.path.exists(numbered_path):
        return
    numbered = read_relion_sampling_metadata(numbered_path)
    numbered_range = numbered["offset_range"] / settings.pixel_size_angstrom
    numbered_step = numbered["offset_step"] / settings.pixel_size_angstrom
    numbered_grid = sampling_grids.relion_base_translation_grid(
        numbered_range, numbered_step,
        n_classes=n_classes, voxel_size=settings.pixel_size_angstrom,
    ).astype(np.float32)
    final_grid = sampling_grids.relion_base_translation_grid(
        settings.translation_range, settings.translation_step,
        n_classes=n_classes, voxel_size=settings.pixel_size_angstrom,
    ).astype(np.float32)
    if numbered_grid.shape != final_grid.shape or not np.allclose(
        numbered_grid, final_grid, rtol=0.0, atol=1e-6,
    ):
        logger.info(
            "RELION final all-data sampling grid differs from last numbered sampling: "
            "numbered n=%d range=%.9g step=%.9g hp=%d; final n=%d range=%.9g step=%.9g hp=%d",
            numbered_grid.shape[0], numbered_range, numbered_step, numbered["healpix_order"],
            final_grid.shape[0], settings.translation_range, settings.translation_step,
            settings.perturbation_order,
        )
