"""Comparison machinery that enters the refinement through its input-source port (code rule 15).

``relion_replay_source.RelionReplaySource`` replays a RELION run's per-iteration state. The command builds
the source and calls the rest: ``oracle_admission`` (a dispatch capture, follower routing, a replay's restart
provenance), ``startup_noise_inputs`` (start-up noise taken from elsewhere) and ``archive_provenance`` (the
archive's replay keys). The algorithm (``relax.refinement``) imports only ``relax.refinement.ports``.

The sources the replay reads live here too: ``relion_replay`` (per-iteration RELION overrides and readers),
``replay_inputs`` and ``initial_model_replay`` (a replay's start-up inputs), ``frozen_boundary`` with its
flags in ``frozen_boundary_cli`` (a sealed restart boundary), ``state_swap_probe`` (flags) with
``state_swap_runtime`` (snapshot and restore at a replay boundary), and ``parity_provenance`` (the worktree
gate of parity runs). Moved modules keep their old logger names, so log rows are unchanged.
"""
