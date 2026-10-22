"""Halves scored on another GPU: rank 0's side and the other ranks' task loop (``relax refine --gpu 0:1``).

A multi-GPU run is one process per GPU (:mod:`relax_cli.launch`). Rank 0 runs the controller as a
one-GPU run does; with two ranks, rank 1 scores half 2. Rank 0 sends the half's scoring call (the function
and its arguments, as the one-GPU run would call it) before it scores half 1 itself, and publishes half 2's
returned result after half 1's, in the one-GPU order. Each half's arithmetic is the one-GPU run's: its sums
are made on one GPU, so nothing is reduced across GPUs.

The other ranks run the command's start-up (their own datasets, options and run context, from the same
command line) and then :func:`serve_remote_halves`: each task is a call, run between the same deferred
preprocess checks as a one-GPU expectation, whose result returns with the engine records it made. The run
objects (:func:`run_objects`) are each rank's own and travel by name (:mod:`relax.parallel.channel`).

Only the native computation is admitted (:func:`require_rank_route`): replayed RELION state and diagnostic
observers are refused, since their captures would be split across processes.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection, Listener

import numpy as np

from relax.parallel import channel
from relax.refinement import refinement_options, setup_checks
from relax.refinement.ports import RunObserver
from relax.runtime import engine_record
from relax.runtime.host_memory import return_freed_heap
from relax.runtime.timing import Stopwatch

logger = logging.getLogger("relax.refinement.iteration_loop")


@dataclass(frozen=True)
class RankLink:
    """Rank 0's view of a multi-GPU run before its controller starts: the listener the other ranks connect
    to and the number of ranks."""

    listener: Listener
    world: int


@dataclass(frozen=True)
class HalfTask:
    """One half's scoring call: ``call(*args)``, the call a one-GPU run makes for half ``half_index``."""

    half_index: int
    call: object
    args: tuple


@dataclass(frozen=True)
class HalfReply:
    """A scored half and the engine records its scoring made (``engine_record``), with its local-search
    profile rows (``LocalDiagnosticPolicy.local_profile_history``) and the rank's seconds."""

    result: object
    pass_engines: list
    coarse_engine_calls: list
    local_profile_rows: list
    seconds: float


@dataclass(frozen=True)
class HalfFailure:
    """A rank's scoring call raised: the traceback, as text."""

    half_index: int
    text: str


def require_rank_route(observer: RunObserver, source) -> None:
    """Refuse a multi-GPU run of what it does not support: a RELION replay or sealed state (``source``) and
    any observer other than the do-nothing ``RunObserver()`` (its captures and dumps would be split between
    the ranks' processes)."""

    if type(observer) is not RunObserver:
        raise NotImplementedError(
            f"--gpu with more than one device runs no observers; this run has {type(observer).__name__} "
            "(a timing directory, saved intermediates or a capture/dump environment variable). Run on one GPU."
        )
    if source.relion_replay is not None or source.sealed_sampling_state is not None or source.replays_relion_state():
        raise NotImplementedError(
            "--gpu with more than one device runs the native computation only, not a RELION replay"
        )


def run_objects(ctx, options, experiment_datasets, local_profile_history: list) -> dict[str, object]:
    """What every rank holds for the whole run, by name; a reference to one travels as its name.

    ``ctx`` is the run context (``setup_checks.build_run_context``), ``options`` the validated options,
    ``local_profile_history`` the list local searches append their profiles to (rank 0's history list; a
    rank's own list, emptied for each task, elsewhere).
    """

    return {
        "ctx": ctx,
        "options": options,
        "dataset_half1": experiment_datasets[0],
        "dataset_half2": experiment_datasets[1],
        "batch_planner": ctx.batch_planner,
        "expectation_probe": ctx.expectation_probe,
        "maximization_probe": ctx.maximization_probe,
        "k_class_probe": ctx.k_class_probe,
        "local_profile_history": local_profile_history,
    }


def run_fingerprint(ctx, experiment_datasets) -> dict:
    """What the ranks must agree on before rank 0 sends them work: the particles of each half, the image and
    volume grids, and the device model."""

    import hashlib

    import jax

    def digest(dataset):
        sha = hashlib.sha256()
        for name in ("CTF_params", "unit_image_offsets", "image_projections"):
            value = getattr(dataset, name, None)
            if value is not None:
                sha.update(np.ascontiguousarray(np.asarray(value)).tobytes())
        return sha.hexdigest()

    return {
        "units": [int(dataset.n_units) for dataset in experiment_datasets],
        "datasets": [digest(dataset) for dataset in experiment_datasets],
        "image_shape": tuple(ctx.image_geometry.image_shape),
        "volume_shape": tuple(ctx.volume_shape),
        "device_kind": jax.devices()[0].device_kind,
    }


class RemoteHalves:
    """Rank 0's halves scored on other ranks: ``owners[k]`` is the rank that scores half ``k`` (0: here)."""

    def __init__(self, connections: dict[int, Connection], owners: tuple[int, int], objects: dict[str, object]):
        self._connections = connections
        self._owners = owners
        self._objects = objects

    def remote(self, half_index: int) -> bool:
        return self._owners[half_index] != 0

    def submit(self, half_index: int, call, *args) -> None:
        """Send half ``half_index``'s scoring call to its rank."""

        channel.send(self._connections[self._owners[half_index]], HalfTask(half_index, call, args), self._objects)

    def result(self, half_index: int):
        """Half ``half_index``'s scored result, once its rank returns it; its engine records and profile rows
        join this process's, after those of the halves published before it."""

        rank = self._owners[half_index]
        clock = Stopwatch()
        reply = channel.receive(self._connections[rank], self._objects, peer=f"rank {rank}")
        if isinstance(reply, HalfFailure):
            raise channel.RankFailed(f"rank {rank} failed scoring half {reply.half_index + 1}:\n{reply.text}")
        engine_record.append_taken(reply.pass_engines, reply.coarse_engine_calls)
        self._objects["local_profile_history"].extend(reply.local_profile_rows)
        logger.info(
            "Half %d scored on rank %d in %.1fs (rank 0 waited %.1fs for it)",
            half_index + 1,
            rank,
            reply.seconds,
            clock.seconds,
        )
        return reply.result


def accept_remote_halves(link: RankLink, objects: dict[str, object], fingerprint: dict) -> RemoteHalves:
    """Accept the other ranks and check that each loaded the same run (``fingerprint``). With two ranks,
    rank 1 scores half 2."""

    if link.world != 2:
        raise NotImplementedError(f"a multi-GPU run takes two GPUs (one per half); --gpu lists {link.world}")
    clock = Stopwatch()
    connections = channel.accept_ranks(link.listener, link.world)
    for rank, connection in connections.items():
        theirs = channel.receive(connection, objects, peer=f"rank {rank}")
        if theirs != fingerprint:
            raise channel.RankFailed(f"rank {rank} loaded another run: {theirs} against rank 0's {fingerprint}")
    logger.info("Multi-GPU run: %d ranks connected in %.1fs; rank 1 scores half 2", link.world, clock.seconds)
    return RemoteHalves(connections, (0, 1), objects)


def serve_remote_halves(
    connection: Connection, *, experiment_datasets, options, observer: RunObserver, source, k_class_probe
) -> None:
    """A rank's task loop: build the run context as rank 0 does, report the run's fingerprint, then run each
    half's scoring call and return its reply, until rank 0 closes the connection (the run's end).

    A failed call is reported to rank 0 and raised here, so this rank exits non-zero.
    """

    from relax.classification.probes import NoProbe
    from relax.cuda.preprocess import expectation_relion_preprocess_checks
    from relax.fine_pass.resident_stable_windows import stable_window_class_history

    options = refinement_options.with_validated_sampling_schedule(options)
    ctx = setup_checks.build_run_context(
        experiment_datasets,
        options,
        replays_relion_state=source.replays_relion_state(),
        observer=observer,
        k_class_probe=NoProbe() if k_class_probe is None else k_class_probe,
    )
    local_profile_history = []
    objects = run_objects(ctx, options, experiment_datasets, local_profile_history)
    channel.send(connection, run_fingerprint(ctx, experiment_datasets), objects)
    engine_record.take_pass_engines()
    engine_record.take_coarse_engine_calls()
    with stable_window_class_history():
        while True:
            try:
                task = channel.receive(connection, objects, peer="rank 0")
            except channel.RankFailed:
                logger.info("Rank 0 closed the connection: the run is over")
                return
            local_profile_history.clear()
            clock = Stopwatch()
            try:
                with expectation_relion_preprocess_checks():
                    result = task.call(*task.args)
            except BaseException:
                channel.send(connection, HalfFailure(task.half_index, traceback.format_exc()), objects)
                raise
            reply = HalfReply(
                result=result,
                pass_engines=engine_record.take_pass_engines(),
                coarse_engine_calls=engine_record.take_coarse_engine_calls(),
                local_profile_rows=list(local_profile_history),
                seconds=clock.seconds,
            )
            channel.send(connection, reply, objects)
            del result, reply
            return_freed_heap(f"half {task.half_index + 1}'s E-step on this rank")
