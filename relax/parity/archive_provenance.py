"""What a run replayed from RELION, as keys of its refinement archive (code rule 15).

The command adds them to the archive metadata (``relax.refinement.result_files.build_archive_metadata``,
``replay_provenance``): a perturbation replay's restart provenance, the retired captured projector's keys, a
frozen boundary, a state-swap probe and the follower-scale replay and dispatch capture of an MPI RELION run. A run
that replays nothing writes the same keys with their empty values (the last two groups only when present).
"""

import numpy as np


def replay_archive_metadata(
    *, restart, frozen_boundary, state_swap_probe, state_swap_probe_applied_relion_iterations, follower_replay,
    relion_dispatch_schedule,
) -> dict:
    """The archive keys of a run's replay provenance, in their stored order."""
    save_dict = {
        "perturb_replay_restart_state_iterations": np.asarray(
            restart.iterations,
            dtype=np.int64,
        ),
        "perturb_replay_restart_provenance_path": np.asarray(
            ""
            if restart.path is None
            else str(restart.path)
        ),
        "perturb_replay_restart_provenance_sha256": np.asarray(
            restart.sha256 or ""
        ),
        # The captured RELION projector is retired (git tag retired/captured-projector-20261006); its keys
        # keep the values a run without one wrote.
        "relion_projector_replay_slot": np.int64(-1),
        "relion_projector_source_manifest_sha256": np.asarray(""),
        "relion_projector_capture_dir": np.asarray(""),
        "relion_projector_capture_manifest": np.asarray(""),
        "frozen_boundary_dir": np.asarray(
            "" if frozen_boundary is None else str(frozen_boundary.source_dir)
        ),
        "frozen_boundary_manifest_sha256": np.asarray(
            "" if frozen_boundary is None else frozen_boundary.source_manifest_sha256
        ),
        "frozen_boundary_sha256": np.asarray(
            "" if frozen_boundary is None else frozen_boundary.boundary_sha256
        ),
        "frozen_boundary_completed_relion_iteration": np.int64(
            -1 if frozen_boundary is None else frozen_boundary.completed_relion_iteration
        ),
        "state_swap_probe_target_relion_iteration": np.int64(
            -1
            if state_swap_probe is None
            else int(state_swap_probe["target_relion_iteration"])
        ),
        "state_swap_probe_loop_index": np.int64(
            -1 if state_swap_probe is None else int(state_swap_probe["iteration"])
        ),
        "state_swap_probe_variant": np.asarray(
            "" if state_swap_probe is None else str(state_swap_probe["variant"])
        ),
        "state_swap_probe_replay_relion_references": np.bool_(
            False
            if state_swap_probe is None
            else bool(state_swap_probe["replay_relion_references"])
        ),
        "state_swap_probe_applied_relion_iterations": np.asarray(
            state_swap_probe_applied_relion_iterations,
            dtype=np.int64,
        ),
        "state_swap_probe_replay_override_keys": np.asarray(
            [] if state_swap_probe is None else state_swap_probe["replay_override_keys"],
            dtype=np.str_,
        ),
        "state_swap_probe_required_replay_override_keys": np.asarray(
            []
            if state_swap_probe is None
            else state_swap_probe["required_replay_override_keys"],
            dtype=np.str_,
        ),
    }
    if follower_replay is not None:
        save_dict["relion_follower_scale_replay_iterations"] = np.asarray(
            follower_replay.relion_iterations,
            dtype=np.int64,
        )
        save_dict["relion_follower_scale_replay_source"] = np.asarray(
            follower_replay.source
        )
        save_dict["relion_follower_scale_replay_oracle_id"] = np.asarray(
            follower_replay.oracle_id
        )
        save_dict["relion_follower_scale_replay_boundary"] = np.asarray(
            follower_replay.boundary
        )
        save_dict["relion_follower_scale_replay_source_artifacts"] = np.asarray(
            follower_replay.source_artifact_relative_paths
        )
    if relion_dispatch_schedule is not None:
        save_dict["relion_dispatch_oracle_id"] = np.asarray(
            relion_dispatch_schedule.oracle_id
        )
        save_dict["relion_dispatch_oracle_manifest_sha256"] = np.asarray(
            relion_dispatch_schedule.oracle_manifest_sha256
        )
        save_dict["relion_dispatch_particle_order_sha256"] = np.asarray(
            relion_dispatch_schedule.particle_order_sha256
        )
    return save_dict
