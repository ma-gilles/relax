"""Strict RELION MPI-follower topology and the controller's follower-scale dispatch.

``prepare_follower_topology`` admits the follower topology at the command edge (MPI admission, the
captured dispatch schedule, numbered particle ownership);
``setup_relion_follower_scale_state`` and the ``dispatch_*`` functions give the iteration loop its setup,
per-iteration and final dispatch and correction update. The scale state and its RELION reduction are
``relax.refinement.follower_scale_state``; the dispatch-schedule artifacts are
``relax.refinement.follower_oracles``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from relax.refinement.follower_oracles import RelionDispatchSchedule
from relax.refinement.follower_scale_state import (
    RELION_SCALE_REDUCTION_MODES,
    RelionFollowerScaleState,
    make_relion_follower_scale_state,
    relion_rank1_serialized_scales,
    relion_scale_reduction_mode_from_optimiser_star,
    relion_worker_group_ids,
    select_relion_follower_scales,
    update_relion_follower_scales,
    validate_relion_follower_scale_start,
)
from relax.refinement.iteration_history import RefinementHistory
from relax.relion.normalization import format_relion_correction_range

if TYPE_CHECKING:
    from relax.refinement.refinement_options import RefinementOptions

logger = logging.getLogger(__name__)


def relion_class3d_follower_owners_from_schedule(
    schedule: RelionDispatchSchedule,
    *,
    particle_ids_by_image,
    random_seed: int,
    relion_iteration: int,
) -> np.ndarray:
    """Map one captured dynamic dispatch row into RECOVAR image order."""

    if int(random_seed) != int(schedule.random_seed):
        raise ValueError(
            "RELION dispatch schedule random_seed does not match the refinement "
            f"({schedule.random_seed} != {int(random_seed)})"
        )
    matches = np.flatnonzero(schedule.relion_iterations == int(relion_iteration))
    if matches.size != 1:
        raise ValueError(
            f"RELION dispatch schedule does not contain iteration {int(relion_iteration)}"
        )
    particle_ids = np.asarray(particle_ids_by_image, dtype=np.int64).reshape(-1)
    row = np.asarray(schedule.owner_by_sorted_position[int(matches[0])], dtype=np.int64)
    sorted_particle_ids = np.asarray(
        schedule.original_particle_id_by_sorted_position[int(matches[0])], dtype=np.int64
    )
    if row.shape != sorted_particle_ids.shape or row.shape != particle_ids.shape:
        raise ValueError(
            "RELION dispatch schedule particle count does not match the authoritative data STAR "
            f"({row.size} != {sorted_particle_ids.size})"
        )
    owner_by_particle = np.empty(row.size, dtype=np.int64)
    owner_by_particle[sorted_particle_ids] = row
    return owner_by_particle[particle_ids]


@dataclass(frozen=True, eq=False)
class PreparedFollowerTopology:
    """Resolved MPI topology and per-iteration particle ownership."""

    n_followers: int
    reduction_mode: str | None
    owners_by_iteration: dict[int, list[np.ndarray]] | None


def prepare_follower_topology(
    requested_followers,
    schedule,
    group_layout,
    *,
    strict_replay,
    oracle_dir,
    random_seed,
    init_relion_iteration,
    max_iter,
    group_source=None,
    logger=logger,
) -> PreparedFollowerTopology:
    """Validate follower admission and prepare ownership from a verified schedule.

    Oracle verification precedes this operation. Owners remain in each half's
    image order; the reduction mode comes from the oracle's recorded command.
    See docs/math/relion_refinement_algorithm.md#follower-topology-admission-before-refinement.
    """
    if requested_followers is None:
        if strict_replay and schedule is None:
            raise SystemExit(
                "Strict K>1 RELION replay requires --relion-dispatch-schedule captured "
                "from the same oracle run: expectation follower ownership is a dynamic "
                "MPI work queue and cannot be reconstructed from --seed. Pass "
                "--relion-scale-followers 0 for a non-MPI RELION oracle (single-process "
                "group scales)."
            )
        relion_scale_followers = (
            0 if schedule is None else schedule.n_followers
        )
    else:
        relion_scale_followers = int(requested_followers)
        if relion_scale_followers < 0:
            raise SystemExit("--relion-scale-followers must be non-negative")
        if relion_scale_followers > 0 and not strict_replay:
            raise SystemExit(
                "--relion-scale-followers is strict K>1 RELION replay/init state only; "
                "provide --relion_init_dir or --perturb_replay_relion_dir"
            )
    if relion_scale_followers > 0 and schedule is None:
        raise SystemExit(
            "--relion-scale-followers > 0 requires --relion-dispatch-schedule; "
            "seed-only/static ownership is not RELION-exact"
        )
    if (
        schedule is not None
        and relion_scale_followers != schedule.n_followers
    ):
        raise SystemExit(
            "--relion-scale-followers disagrees with --relion-dispatch-schedule "
            f"({relion_scale_followers} != {schedule.n_followers})"
        )
    if relion_scale_followers > 0 and group_layout is None:
        raise SystemExit("Strict RELION follower-scale emulation requires an authoritative group layout")
    if relion_scale_followers > 0 and int(init_relion_iteration) > 0:
        raise SystemExit(
            "Strict RELION follower-scale emulation cannot cold-start after iteration 0 from "
            "a leader-serialized STAR; rerun from iteration 0 (full follower-scale checkpoint "
            "input is not implemented)"
        )
    if group_layout is not None:
        logger.info(
            "Native RELION group layout: source=%s full_groups=%d half1_present=%d half2_present=%d",
            group_layout.source,
            group_layout.n_groups,
            int(np.unique(group_layout.group_ids_per_half[0]).size),
            int(np.unique(group_layout.group_ids_per_half[1]).size),
        )
        if group_source is not None:
            logger.info("Native RELION group layout provenance: %s", group_source)
    if relion_scale_followers > 0:
        logger.info(
            "Strict RELION follower-scale topology: followers=%d physical_groups=%d "
            "optics_groups=%d dynamic_schedule=%s",
            relion_scale_followers,
            group_layout.n_groups,
            group_layout.n_optics_groups,
            schedule.source,
        )
        logger.info(
            "Verified RELION dispatch oracle: oracle_id=%s artifacts=%d particle_star=%s",
            schedule.oracle_id,
            len(schedule.oracle_artifact_paths),
            schedule.particle_star_relative_path,
        )
    relion_scale_reduction_mode = None
    if relion_scale_followers > 0:
        # Which physical groups RELION reduced across followers is a property of
        # the oracle run's weight-combination path (relion_worker_scale docstring).
        reduction_source = oracle_dir / "run_it000_optimiser.star"
        try:
            relion_scale_reduction_mode, relion_command = (
                relion_scale_reduction_mode_from_optimiser_star(reduction_source)
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Cannot determine the RELION group-scale reduction: {exc}") from exc
        logger.info(
            "Strict RELION follower-scale reduction: mode=%s dont_combine_weights_via_disc=%s source=%s",
            relion_scale_reduction_mode,
            "--dont_combine_weights_via_disc" in relion_command.split(),
            reduction_source,
        )
    relion_scale_follower_owners_by_iteration = None
    if relion_scale_followers > 0:
        relion_scale_follower_owners_by_iteration = {}
        for relion_iteration_value in schedule.relion_iterations:
            relion_iteration = int(relion_iteration_value)
            try:
                owners_half1 = relion_class3d_follower_owners_from_schedule(
                    schedule,
                    particle_ids_by_image=group_layout.particle_ids_per_half[0],
                    random_seed=int(random_seed),
                    relion_iteration=relion_iteration,
                )
            except (RuntimeError, ValueError) as exc:
                raise SystemExit(
                    f"RELION dispatch schedule cannot supply iteration {relion_iteration}: {exc}"
                ) from exc
            relion_scale_follower_owners_by_iteration[relion_iteration] = [
                owners_half1,
                np.zeros(group_layout.particle_ids_per_half[1].size, dtype=np.int64),
            ]
        required_numbered_iterations = set(
            range(
                int(init_relion_iteration) + 1,
                int(init_relion_iteration) + int(max_iter) + 1,
            )
        )
        missing_numbered_iterations = sorted(
            required_numbered_iterations - set(relion_scale_follower_owners_by_iteration)
        )
        if missing_numbered_iterations:
            raise SystemExit(
                "RELION dispatch schedule cannot supply requested numbered iterations: "
                f"{missing_numbered_iterations}"
            )
    return PreparedFollowerTopology(
        n_followers=relion_scale_followers,
        reduction_mode=relion_scale_reduction_mode,
        owners_by_iteration=relion_scale_follower_owners_by_iteration,
    )


@dataclass
class RelionFollowerScaleSetup:
    """RELION per-follower group-scale emulation state for one refinement run.

    Mutable: ``follower_scale_state``, ``follower_owners_per_half``, and
    ``scale_stats_group_ids_per_half`` are reassigned once per numbered
    iteration by the iteration loop's dispatch helpers. ``follower_count == 0``
    (equivalently ``follower_scale_state is None``) means the strict
    follower-topology emulation is off and every downstream RELION-parity
    branch that depends on it should fall through to ordinary behavior.
    """

    follower_count: int
    follower_scale_state: RelionFollowerScaleState | None
    follower_owners_per_half: list
    follower_owners_by_iteration: dict | None
    scale_stats_group_ids_per_half: list = field(default_factory=list)
    scale_stats_group_count_per_half: list = field(default_factory=list)
    physical_group_count: int = 0
    scale_reduction_mode: str | None = None

    def result_outputs(self, history: RefinementHistory) -> FollowerScaleOutputs:
        """This run's follower-scale outputs, as host copies taken now; every field None when emulation is off."""
        if self.follower_scale_state is None:
            return FollowerScaleOutputs()
        return FollowerScaleOutputs(
            follower_scales=np.asarray(self.follower_scale_state.scales, dtype=np.float64),
            rank1_serialized=relion_rank1_serialized_scales(self.follower_scale_state),
            follower_owners_half1=np.asarray(self.follower_owners_per_half[0], dtype=np.int64),
            follower_owners_half1_trajectory=np.asarray(
                history.relion_follower_owners_half1_trajectory, dtype=np.int64
            ),
            follower_scales_numbered_pre_score_trajectory=np.asarray(
                history.relion_scale_follower_scales_numbered_pre_score_trajectory, dtype=np.float64
            ),
            follower_scales_numbered_post_mstep_trajectory=np.asarray(
                history.relion_scale_follower_scales_numbered_post_mstep_trajectory, dtype=np.float64
            ),
        )


@dataclass(frozen=True, kw_only=True)
class FollowerScaleOutputs:
    """A run's RELION follower-scale outputs (``relion_scale_*`` in the archive); all None when off.

    Host copies taken when the controller returns, so the result does not keep the mutable
    ``RelionFollowerScaleSetup`` alive.
    """

    follower_scales: np.ndarray | None = None
    rank1_serialized: np.ndarray | None = None
    follower_owners_half1: np.ndarray | None = None
    follower_owners_half1_trajectory: np.ndarray | None = None
    follower_scales_numbered_pre_score_trajectory: np.ndarray | None = None
    follower_scales_numbered_post_mstep_trajectory: np.ndarray | None = None

    def archive_fields(self) -> dict:
        return {
            "relion_scale_follower_scales": self.follower_scales,
            "relion_scale_rank1_serialized": self.rank1_serialized,
            "relion_scale_follower_owners_half1": self.follower_owners_half1,
            "relion_scale_follower_owners_half1_trajectory": self.follower_owners_half1_trajectory,
            "relion_scale_follower_scales_numbered_pre_score_trajectory": (
                self.follower_scales_numbered_pre_score_trajectory
            ),
            "relion_scale_follower_scales_numbered_post_mstep_trajectory": (
                self.follower_scales_numbered_post_mstep_trajectory
            ),
        }


def k1_follower_scale_state(
    options: RefinementOptions,
    *,
    topology: PreparedFollowerTopology | None,
    relion_half_inputs,
    experiment_datasets,
) -> RelionFollowerScaleSetup:
    """K=1's follower-scale state: the inert default. A follower topology is refused (follower-local scale
    emulation is strict K-class state) after the start check every topology passes."""
    follower_count = 0 if topology is None else int(topology.n_followers or 0)
    if follower_count > 0:
        validate_relion_follower_scale_start(
            n_followers=follower_count,
            init_relion_iteration=options.schedule.init_relion_iteration,
        )
        raise ValueError("RELION follower-local scale emulation is strict K-class state only")
    return setup_relion_follower_scale_state(
        options, topology=topology, relion_half_inputs=relion_half_inputs, experiment_datasets=experiment_datasets,
    )


def setup_relion_follower_scale_state(
    options: RefinementOptions,
    *,
    topology: PreparedFollowerTopology | None,
    relion_half_inputs,
    experiment_datasets,
) -> RelionFollowerScaleSetup:
    """Build (or return the inert default for) RELION's per-follower
    group-scale emulation state.

    ``topology`` is the replayed RELION run's follower topology and captured
    dispatch schedule (the input source's; None: no followers).
    See this module's docstring for the MPI-follower parity rationale.
    Updates each particle half's ``scale_corrections`` when the
    strict follower topology is active, seeding each half's per-particle
    scale from the newly constructed follower state.
    """
    schedule = options.schedule

    follower_count = 0 if topology is None else int(topology.n_followers or 0)
    owners_by_iteration = None if topology is None else topology.owners_by_iteration
    follower_scale_state = None
    follower_owners_per_half = [None, None]
    follower_owners_by_iteration = None
    scale_stats_group_ids_per_half = [particle_half.group_ids for particle_half in relion_half_inputs]
    scale_stats_group_count_per_half = [particle_half.group_count for particle_half in relion_half_inputs]
    physical_group_count = 0
    scale_reduction_mode = None

    validate_relion_follower_scale_start(
        n_followers=follower_count,
        init_relion_iteration=schedule.init_relion_iteration,
    )
    if follower_count > 0:
        if relion_half_inputs[0].group_ids is None:
            raise ValueError("RELION follower-local scale emulation requires physical group IDs")
        if owners_by_iteration is None:
            raise ValueError(
                "RELION follower-local scale emulation requires a captured per-iteration "
                "dynamic dispatch schedule; seed-only ownership is not exact"
            )
        physical_group_count = int(relion_half_inputs[0].group_count or 0)
        if physical_group_count < 1:
            raise ValueError("RELION follower-local scale emulation requires a positive group count")
        optics_group_count = int(options.start.init_relion_optics_group_count or 0)
        if optics_group_count < 1:
            raise ValueError("RELION follower-local scale emulation requires a positive optics-group count")
        scale_reduction_mode = topology.reduction_mode
        if scale_reduction_mode not in RELION_SCALE_REDUCTION_MODES:
            raise ValueError(
                "RELION follower-local scale emulation requires the oracle's group-scale "
                f"reduction mode {RELION_SCALE_REDUCTION_MODES}, got {scale_reduction_mode!r}"
            )

        if isinstance(owners_by_iteration, Mapping):
            raw_owner_items = owners_by_iteration.items()
        else:
            raw_owner_items = (
                (schedule.init_relion_iteration + schedule_idx + 1, owner_pair)
                for schedule_idx, owner_pair in enumerate(owners_by_iteration)
            )
        follower_owners_by_iteration = {}
        for relion_iteration, owner_pair in raw_owner_items:
            if relion_iteration in follower_owners_by_iteration:
                raise ValueError(
                    f"RELION dispatch schedule contains duplicate iteration {relion_iteration}"
                )
            if len(owner_pair) != 2:
                raise ValueError("each RELION dispatch schedule row must contain two half arrays")
            normalized_pair = []
            for half_idx in range(2):
                owners = np.asarray(owner_pair[half_idx], dtype=np.int64).reshape(-1)
                expected_size = int(experiment_datasets[half_idx].n_units)
                if owners.shape != (expected_size,):
                    raise ValueError(
                        f"RELION dispatch owners iteration {relion_iteration} half {half_idx + 1} "
                        f"have shape {owners.shape}, expected {(expected_size,)}"
                    )
                if owners.size and (
                    int(np.min(owners)) < 0
                    or int(np.max(owners)) >= follower_count
                ):
                    raise ValueError("RELION dispatch schedule contains out-of-bounds followers")
                normalized_pair.append(owners.copy())
            follower_owners_by_iteration[relion_iteration] = normalized_pair

        required_numbered_iterations = range(
            schedule.init_relion_iteration + 1,
            schedule.init_relion_iteration + int(schedule.max_iter) + 1,
        )
        missing_numbered_iterations = [
            relion_iteration
            for relion_iteration in required_numbered_iterations
            if relion_iteration not in follower_owners_by_iteration
        ]
        if missing_numbered_iterations:
            raise ValueError(
                "RELION dispatch schedule does not cover every requested numbered iteration; "
                f"missing {missing_numbered_iterations}"
            )

        group_counts = np.zeros(physical_group_count, dtype=np.float64)
        for half in relion_half_inputs:
            group_ids_k = half.group_ids
            if group_ids_k is not None and np.asarray(group_ids_k).size:
                group_counts += np.bincount(
                    np.asarray(group_ids_k, dtype=np.int64),
                    minlength=physical_group_count,
                )[:physical_group_count]
        initial_group_scales = np.ones(physical_group_count, dtype=np.float64)
        initial_scale_sums = np.zeros(physical_group_count, dtype=np.float64)
        initial_scale_counts = np.zeros(physical_group_count, dtype=np.float64)
        for half in relion_half_inputs:
            group_ids_k = half.group_ids
            scales_k = half.scale_corrections
            if group_ids_k is None or scales_k is None or np.asarray(group_ids_k).size == 0:
                continue
            groups_k = np.asarray(group_ids_k, dtype=np.int64)
            scales_np = np.asarray(scales_k, dtype=np.float64)
            initial_scale_sums += np.bincount(
                groups_k,
                weights=scales_np,
                minlength=physical_group_count,
            )[:physical_group_count]
            initial_scale_counts += np.bincount(
                groups_k,
                minlength=physical_group_count,
            )[:physical_group_count]
        present_initial = initial_scale_counts > 0.0
        initial_group_scales[present_initial] = (
            initial_scale_sums[present_initial] / initial_scale_counts[present_initial]
        )
        follower_scale_state = make_relion_follower_scale_state(
            n_followers=follower_count,
            group_counts=group_counts,
            n_optics_groups=optics_group_count,
            initial_group_scales=initial_group_scales,
        )
        first_relion_iteration = schedule.init_relion_iteration + 1
        follower_owners_per_half = [
            owners.copy()
            for owners in follower_owners_by_iteration[first_relion_iteration]
        ]
        scale_stats_group_ids_per_half = []
        for half_idx in range(2):
            physical_groups = np.asarray(relion_half_inputs[half_idx].group_ids, dtype=np.int64)
            owners = follower_owners_per_half[half_idx]
            scale_stats_group_ids_per_half.append(
                relion_worker_group_ids(
                    physical_groups,
                    owners,
                    n_groups=physical_group_count,
                )
            )
            selected_scales = select_relion_follower_scales(
                follower_scale_state,
                group_ids=physical_groups,
                follower_owners=owners,
            )
            relion_half_inputs[half_idx].scale_corrections = selected_scales
        scale_stats_group_count_per_half = [
            follower_count * physical_group_count,
            follower_count * physical_group_count,
        ]
        logger.info(
            "Strict RELION follower-scale state initialized: followers=%d groups=%d optics_groups=%d "
            "scale_reduction=%s rank1_particles=%d rank2_particles=%d",
            follower_count,
            physical_group_count,
            optics_group_count,
            scale_reduction_mode,
            int(np.count_nonzero(follower_owners_per_half[0] == 0)),
            int(np.count_nonzero(follower_owners_per_half[0] == 1)) if follower_count > 1 else 0,
        )

    return RelionFollowerScaleSetup(
        follower_count=follower_count,
        follower_scale_state=follower_scale_state,
        follower_owners_per_half=follower_owners_per_half,
        follower_owners_by_iteration=follower_owners_by_iteration,
        scale_stats_group_ids_per_half=scale_stats_group_ids_per_half,
        scale_stats_group_count_per_half=scale_stats_group_count_per_half,
        physical_group_count=physical_group_count,
        scale_reduction_mode=scale_reduction_mode,
    )


def _require_relion_follower_owners(
    owners_by_relion_iteration,
    *,
    relion_iteration: int,
    stage: str,
):
    """Return an exact captured owner row, failing closed when it is absent."""

    try:
        return owners_by_relion_iteration[relion_iteration]
    except KeyError as exc:
        raise RuntimeError(
            "RELION dispatch schedule does not contain "
            f"{stage} iteration {relion_iteration}; refusing to reuse stale follower owners"
        ) from exc


def _remap_relion_follower_runtime_inputs(
    *,
    state,
    relion_half_inputs,
    follower_owners_per_half,
    physical_group_count: int,
    dtype,
):
    """Remap live scoring scales and XA/AA group IDs to a new dispatch row."""

    stats_group_ids_per_half = []
    for half_idx in range(2):
        physical_groups = np.asarray(relion_half_inputs[half_idx].group_ids, dtype=np.int64)
        owners = np.asarray(follower_owners_per_half[half_idx], dtype=np.int64)
        old_scales = relion_half_inputs[half_idx].scale_corrections
        old_image_corrections = relion_half_inputs[half_idx].image_corrections
        if old_scales is None or old_image_corrections is None:
            raise RuntimeError(
                "RELION follower dispatch remap requires resident image and scale corrections"
            )
        old_scales = np.asarray(old_scales, dtype=np.float64)
        old_image_corrections = np.asarray(old_image_corrections, dtype=np.float64)
        norm_factor = np.ones_like(old_scales, dtype=np.float64)
        np.divide(
            old_image_corrections,
            old_scales,
            out=norm_factor,
            where=old_scales != 0.0,
        )
        selected_scales = select_relion_follower_scales(
            state,
            group_ids=physical_groups,
            follower_owners=owners,
        ).astype(dtype)
        relion_half_inputs[half_idx].scale_corrections = selected_scales
        relion_half_inputs[half_idx].image_corrections = np.asarray(
            norm_factor * selected_scales,
            dtype=dtype,
        )
        stats_group_ids_per_half.append(
            relion_worker_group_ids(
                physical_groups,
                owners,
                n_groups=physical_group_count,
            )
        )
    return stats_group_ids_per_half


def dispatch_relion_follower_scale_for_numbered_iteration(
    setup: RelionFollowerScaleSetup,
    history: RefinementHistory,
    *,
    iteration: int,
    numbered_relion_iteration: int,
    relion_half_inputs,
    dtype,
    logger,
) -> None:
    """Recompute this numbered iteration's follower dispatch, remap group ids, and record pre-score
    telemetry. Mutates ``setup.follower_owners_per_half``,
    ``setup.follower_scale_state``, and ``setup.scale_stats_group_ids_per_half``
    in place. Only called when ``setup.follower_scale_state is not None``.
    """
    setup.follower_owners_per_half = [
        owners.copy()
        for owners in _require_relion_follower_owners(
            setup.follower_owners_by_iteration,
            relion_iteration=numbered_relion_iteration,
            stage="numbered",
        )
    ]
    if iteration > 0:
        setup.scale_stats_group_ids_per_half = _remap_relion_follower_runtime_inputs(
            state=setup.follower_scale_state,
            relion_half_inputs=relion_half_inputs,
            follower_owners_per_half=setup.follower_owners_per_half,
            physical_group_count=setup.physical_group_count,
            dtype=dtype,
        )
    history.record_follower_scale_pre_score(
        np.asarray(setup.follower_scale_state.scales, dtype=np.float64).copy(),
        setup.follower_owners_per_half[0].copy(),
    )
    logger.info(
        "Strict RELION dynamic dispatch: numbered_iter=%d rank_particle_counts=%s",
        numbered_relion_iteration,
        [
            int(np.count_nonzero(setup.follower_owners_per_half[0] == follower))
            for follower in range(setup.follower_count)
        ],
    )


def dispatch_relion_follower_scale_for_final_all_data(
    setup: RelionFollowerScaleSetup,
    *,
    init_relion_iteration: int,
    numbered_iteration_count: int,
    relion_half_inputs,
    dtype,
    logger,
) -> None:
    """Recompute the final all-data dispatch row and remap group ids -- no
    replay override, no telemetry (the final all-data pass is never replayed
    and is not part of the numbered trajectory). Mutates
    ``setup.follower_owners_per_half`` and ``setup.scale_stats_group_ids_per_half``
    in place. Only called when ``setup.follower_scale_state is not None``.
    """
    final_dispatch_relion_iteration = init_relion_iteration + numbered_iteration_count + 1
    setup.follower_owners_per_half = [
        owners.copy()
        for owners in _require_relion_follower_owners(
            setup.follower_owners_by_iteration,
            relion_iteration=final_dispatch_relion_iteration,
            stage="final all-data",
        )
    ]
    setup.scale_stats_group_ids_per_half = _remap_relion_follower_runtime_inputs(
        state=setup.follower_scale_state,
        relion_half_inputs=relion_half_inputs,
        follower_owners_per_half=setup.follower_owners_per_half,
        physical_group_count=setup.physical_group_count,
        dtype=dtype,
    )
    logger.info(
        "Strict RELION dynamic dispatch: final_all_data_iter=%d rank_particle_counts=%s",
        final_dispatch_relion_iteration,
        [
            int(np.count_nonzero(setup.follower_owners_per_half[0] == follower))
            for follower in range(setup.follower_count)
        ],
    )


def update_relion_follower_corrections(
    follower_setup: RelionFollowerScaleSetup,
    *,
    noise_stats_per_half,
    norm_scale_update,
    relion_half_inputs,
    relion_firstiter_cc_this_iter: bool,
    dtype,
    logger,
):
    """Install follower scales and image corrections after a numbered M-step.

    Mutates the setup's scale state and both halves' runtime correction arrays.
    Returns only the rank-1 model-STAR diagnostic, whose second half
    remains absent; the updated state stays in its owner. The controller owns scheduling and history recording and
    supplies its selected precision and logger.
    """
    relion_follower_scale_state = follower_setup.follower_scale_state
    relion_follower_owners_per_half = follower_setup.follower_owners_per_half
    scale_xa = noise_stats_per_half[0].wsum_scale_correction_xa
    scale_aa = noise_stats_per_half[0].wsum_scale_correction_aa
    if not relion_firstiter_cc_this_iter and (scale_xa is None or scale_aa is None):
        raise RuntimeError("Follower-local scale update requires expanded XA/AA statistics")
    relion_follower_scale_state = update_relion_follower_scales(
        relion_follower_scale_state,
        wsum_signal_product=scale_xa,
        wsum_reference_power=scale_aa,
        reduction_mode=follower_setup.scale_reduction_mode,
        relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
    )
    follower_setup.follower_scale_state = relion_follower_scale_state
    for half_idx in range(2):
        physical_groups = np.asarray(relion_half_inputs[half_idx].group_ids, dtype=np.int64)
        selected_scales = select_relion_follower_scales(
            relion_follower_scale_state,
            group_ids=physical_groups,
            follower_owners=relion_follower_owners_per_half[half_idx],
        ).astype(dtype, copy=False)
        normcorr = np.asarray(
            norm_scale_update.norm_corrections_per_half[half_idx],
            dtype=np.float64,
        )
        # None: the norm was not estimated, and normcorr is the input norm around RELION's average 1.
        avg_norm = norm_scale_update.avg_norm_correction_per_half[half_idx]
        avg_norm = 1.0 if avg_norm is None else float(avg_norm)
        norm_factor = np.ones_like(normcorr, dtype=np.float64)
        np.divide(avg_norm, normcorr, out=norm_factor, where=normcorr > 0.0)
        relion_half_inputs[half_idx].scale_corrections = selected_scales
        relion_half_inputs[half_idx].image_corrections = np.asarray(
            norm_factor * selected_scales,
            dtype=dtype,
        )
    group_scale_corrections_for_dump = [
        relion_rank1_serialized_scales(relion_follower_scale_state),
        None,
    ]
    logger.info(
        "Strict RELION follower-scale update: rank1=%s rank2=%s; "
        "rank1 model-STAR diagnostic retained separately",
        format_relion_correction_range(relion_follower_scale_state.scales[0]),
        format_relion_correction_range(relion_follower_scale_state.scales[1])
        if relion_follower_scale_state.n_followers > 1
        else "none",
    )
    return group_scale_corrections_for_dump
