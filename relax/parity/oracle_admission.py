"""Admission of a run's RELION oracle inputs at the command (code rule 15).

A captured follower dispatch schedule is admitted against its verified oracle directories and the particle
order, the followers' topology is built from it, and a perturbation replay's restart provenance is checked. The
command (``relax.refinement.full_refinement``) calls these; the algorithm's modules never import them. What a
frozen boundary replays is ``RelionReplay.from_frozen_boundary``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from relax.refinement.command_options import find_relion_optimiser_star
from relax.refinement.refinement_options import RestartProvenance

if TYPE_CHECKING:
    from relax.relion.relion_worker_scale import RelionDispatchSchedule


class VerifiedDispatchSchedule(NamedTuple):
    """Captured follower assignments paired with their verified oracle roots."""

    schedule: "RelionDispatchSchedule | None"
    oracle_dirs: list[Path]


def load_verified_dispatch_schedule(
    args, particles, *, strict_replay: bool, relion_half_sets_from_input: bool
) -> VerifiedDispatchSchedule:
    """Admit a CLI dispatch capture only against its oracle and particle order.

    Verify directory manifests before discovering the consumed optimiser and
    sampling files. See ``docs/math/relion_refinement_algorithm.md#command-admission``.
    """
    from relax.relion.relion_worker_scale import (
        load_relion_dispatch_schedule,
        relion_ordered_particle_sha256,
        verify_relion_dispatch_schedule_oracle,
    )

    relion_dispatch_schedule = None
    oracle_dirs = []
    if args.relion_dispatch_schedule is not None:
        if not strict_replay:
            raise SystemExit(
                "--relion-dispatch-schedule is strict K>1 RELION replay/init state only"
            )
        try:
            relion_dispatch_schedule = load_relion_dispatch_schedule(
                args.relion_dispatch_schedule
            )
            for candidate in (args.perturb_replay_relion_dir, args.relion_init_dir):
                if candidate is None:
                    continue
                resolved = Path(candidate).expanduser().resolve()
                if resolved not in oracle_dirs:
                    oracle_dirs.append(resolved)
                    verify_relion_dispatch_schedule_oracle(
                        relion_dispatch_schedule,
                        resolved,
                    )
            def _require_manifested_oracle_file(path, *, label):
                resolved_path = Path(path).expanduser().resolve()
                for oracle_dir in oracle_dirs:
                    try:
                        relative = resolved_path.relative_to(oracle_dir).as_posix()
                    except ValueError:
                        continue
                    if relative not in relion_dispatch_schedule.oracle_artifact_paths:
                        raise ValueError(
                            f"{label} is not included in the verified RELION oracle manifest: "
                            f"{resolved_path}"
                        )
                    return
                raise ValueError(
                    f"{label} must belong to a verified RELION oracle directory: {resolved_path}"
                )

            discovered_optimiser = find_relion_optimiser_star(
                args, relion_half_sets_from_input=relion_half_sets_from_input
            )
            if discovered_optimiser is not None:
                _require_manifested_oracle_file(
                    discovered_optimiser,
                    label="consumed RELION optimiser",
                )
            for oracle_dir in oracle_dirs:
                sampling_candidates = list(oracle_dir.glob("run_it*_sampling.star"))
                final_sampling = oracle_dir / "run_sampling.star"
                if final_sampling.exists():
                    sampling_candidates.append(final_sampling)
                for sampling_path in sampling_candidates:
                    _require_manifested_oracle_file(
                        sampling_path,
                        label="consumed RELION sampling state",
                    )
            observed_group_order = relion_ordered_particle_sha256(particles)
            if observed_group_order != relion_dispatch_schedule.particle_order_sha256:
                raise ValueError(
                    "authoritative RELION group/half-set particle order does not match "
                    "the dispatch schedule"
                )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Invalid --relion-dispatch-schedule: {exc}") from exc
    return VerifiedDispatchSchedule(relion_dispatch_schedule, oracle_dirs)


class FollowerRouting(NamedTuple):
    """Which follower scores which particles: the admitted dispatch capture (or None) and the topology."""

    schedule: "RelionDispatchSchedule | None"
    topology: object


def admit_follower_routing(
    args, group_source, particle_groups, *, random_seed: int, relion_half_sets_from_input: bool, log
) -> FollowerRouting:
    """Admit the followers' dispatch capture against its oracle and build their topology.

    A capture is strict K>1 replay state only: it is admitted when the run replays or starts from a RELION
    directory, and the topology then follows it. ``group_source`` is the particle table the groups came from.
    """
    from relax.relion.relion_worker_scale import prepare_follower_topology

    strict_replay = bool(
        args.n_classes > 1
        and (args.perturb_replay_relion_dir is not None or args.relion_init_dir is not None)
    )
    dispatch = load_verified_dispatch_schedule(
        args,
        group_source.particles,
        strict_replay=strict_replay,
        relion_half_sets_from_input=relion_half_sets_from_input,
    )
    topology = prepare_follower_topology(
        args.relion_scale_followers,
        dispatch.schedule,
        particle_groups,
        strict_replay=strict_replay,
        replay_path=args.relion_follower_scale_replay,
        oracle_dir=dispatch.oracle_dirs[0] if dispatch.schedule is not None else None,
        random_seed=random_seed,
        init_relion_iteration=args.init_relion_iteration,
        max_iter=args.max_iter,
        group_source=group_source.path,
        logger=log,
    )
    return FollowerRouting(schedule=dispatch.schedule, topology=topology)


def resolve_restart_provenance(args, *, log) -> RestartProvenance:
    """Admit --perturb-replay-restart-state-iterations with its replay directory and provenance file."""
    from recovar.utils.file_hash import sha256_file

    iterations = tuple(
        sorted(
            {
                int(token.strip())
                for token in args.perturb_replay_restart_state_iterations.split(",")
                if token.strip()
            }
        )
    )
    if any(value < 0 for value in iterations):
        raise SystemExit("--perturb-replay-restart-state-iterations values must be non-negative")
    if iterations and args.perturb_replay_relion_dir is None:
        raise SystemExit(
            "--perturb-replay-restart-state-iterations requires --perturb_replay_relion_dir"
        )
    path = None
    sha256 = None
    if iterations:
        if args.perturb_replay_restart_provenance is None:
            raise SystemExit(
                "--perturb-replay-restart-state-iterations requires "
                "--perturb-replay-restart-provenance"
            )
        path = Path(
            args.perturb_replay_restart_provenance
        ).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(
                "--perturb-replay-restart-provenance is not a file: "
                f"{path}"
            )
        sha256 = sha256_file(
            path
        )
        log.info(
            "SamplingPerturbation restart provenance: iterations=%s path=%s sha256=%s",
            list(iterations),
            path,
            sha256,
        )
    elif args.perturb_replay_restart_provenance is not None:
        raise SystemExit(
            "--perturb-replay-restart-provenance requires "
            "--perturb-replay-restart-state-iterations"
        )
    return RestartProvenance(iterations, path, sha256)
