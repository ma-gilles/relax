"""Comparison machinery that enters the refinement through its input-source port (code rule 15).

``relion_replay_source.RelionReplaySource`` replays a RELION run's per-iteration state. The command builds
the source and calls the rest: ``oracle_admission`` (a dispatch capture, follower routing, a replay's restart
provenance), ``startup_noise_inputs`` (start-up noise taken from elsewhere) and ``archive_provenance`` (the
archive's replay keys). The algorithm (``relax.refinement``) imports only ``relax.refinement.ports``.
"""
