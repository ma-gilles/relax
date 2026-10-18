"""What a run replayed from RELION, as keys of its refinement archive (code rule 15).

The command adds them to the archive metadata (``relax.refinement.result_files.build_archive_metadata``,
``replay_provenance``): a frozen boundary and the dispatch capture of an MPI RELION run.
A run that replays nothing writes the same keys with their empty values (the dispatch capture only when
present).
"""

import numpy as np


def replay_archive_metadata(
    *, frozen_boundary, relion_dispatch_schedule,
) -> dict:
    """The archive keys of a run's replay provenance, in their stored order."""
    save_dict = {
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
    }
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
