"""Comparison machinery that enters the refinement through its input-source port (code rule 15).

``relion_replay_source.RelionReplaySource`` replays a RELION run's per-iteration state. The command builds
the source; the algorithm (``relax.refinement``) imports only ``relax.refinement.ports``.
"""
