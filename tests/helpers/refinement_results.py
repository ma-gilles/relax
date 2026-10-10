"""A ``RefinementResult`` built by hand, for tests of its consumers (writers, the command, stand-in passes).

``refinement_result(...)`` fills every record a test does not name with an empty value: no maps, an empty
``RefinementHistory``, the replay off, the follower-scale emulation off, no numbered metadata, no final pass.
Pass ``history=RefinementHistory(...)``, ``maps=ModelMaps(...)`` and so on to set what the test reads.
"""

from __future__ import annotations

from relax.refinement.iteration_history import RefinementHistory
from relax.refinement.refinement_result import ModelMaps, NumberedMetadata, RefinementResult, ReplayTelemetry
from relax.relion.worker_scale import FollowerScaleOutputs


def numbered_metadata(**fields) -> NumberedMetadata:
    """Set-up and numbered metadata, None where not given."""
    names = (
        "hard_assignments", "frozen_initial_scoring_state_sha256", "expected_accuracy_trial_local_indices",
        "expected_accuracy_trial_particle_ids", "setup_phase_seconds",
    )
    return NumberedMetadata(**{name: fields.pop(name, None) for name in names}, **fields)


def refinement_result(**records) -> RefinementResult:
    records.setdefault("maps", ModelMaps(mean=None, means=None))
    records.setdefault("convergence_state", None)
    records.setdefault("history", RefinementHistory())
    records.setdefault("replay", ReplayTelemetry(requested_iterations=None, applied_iterations=None))
    records.setdefault("follower_scale", FollowerScaleOutputs())
    records.setdefault("numbered", None)
    return RefinementResult(**records)
