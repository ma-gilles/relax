"""Halves scored on other GPUs: rank 0's side and the other ranks' task loop (``relax refine --gpu 0:1``).

A multi-GPU run is one process per GPU (:mod:`relax_cli.launch`). Rank 0 runs the controller as a
one-GPU run does; with two ranks, rank 1 scores half 2 (:class:`RemoteHalves`). With three or four ranks a
subtomogram run scores both halves in shards on every rank, rank 0 included (:class:`ShardPool`). Rank 0 sends the half's scoring call (the function
and its arguments, as the one-GPU run would call it) before it scores half 1 itself, and publishes half 2's
returned result after half 1's, in the one-GPU order. Each half's arithmetic is the one-GPU run's: its sums
are made on one GPU, so nothing is reduced across GPUs.

The other ranks run the command's start-up (their own datasets, options and run context, from the same
command line) and then :func:`serve_remote_halves`: each task is a call, run between the same deferred
preprocess checks as a one-GPU expectation, whose result returns with the engine records it made. The run
objects (:func:`run_objects`) are each rank's own and travel by name (:mod:`relax.parallel.channel`).

Shards (:mod:`relax.steps.tomo_shards`, :mod:`relax.parallel.shards`): rank 0 stops each half's scorer at its
subtomogram call, sizes the half's coarse pass plan once (``TomoHalfJob.coarse_plan``: every shard scores with
it), cuts the half's particles into shards by predicted cost (whole particle groups of a frame-shared global
coarse pass, which decide its near-cut rescoring) and hands the shards of both halves
out heaviest first, each to the next free rank. A rank receives a half's call once (:class:`ShardJob`) and then
the particles of each shard it scores (:class:`ShardTask`). Rank 0 joins each half's shards in shard order
(:func:`relax.steps.score_outputs.merge_half_shards`), so neither which rank ran a shard nor when changes the
result. Every particle is scored as in the one-GPU call; only the order of the half's additive sums changes.

Only the native computation is admitted (:func:`require_rank_route`): replayed RELION state and diagnostic
observers are refused, since their captures would be split across processes.
"""

from __future__ import annotations

import logging
import threading
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


@dataclass(frozen=True)
class ShardJob:
    """A half's subtomogram call (``TomoHalfJob``), which a rank keeps under ``key`` until
    :class:`ReleaseShardJobs` and scores shards of. It has no reply."""

    key: tuple
    job: object


@dataclass(frozen=True)
class ShardTask:
    """Score the particles ``units`` (half-local, in this order) of the call kept under ``key``; the reply is a
    :class:`HalfReply`."""

    key: tuple
    half_index: int
    units: np.ndarray


@dataclass(frozen=True)
class ReleaseShardJobs:
    """Drop the calls kept under ``keys``. It has no reply."""

    keys: tuple


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

    shards = False

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


class ShardPool:
    """Rank 0's subtomogram halves scored in shards on every rank, rank 0 included, and joined per half.

    ``connections`` are the other ranks', ``objects`` the run objects. A particle's predicted cost is its
    significant samples when its half was last scored here (1 before that) times its tilt images
    (:func:`relax.parallel.shards.particle_costs`). Each other rank is served by a thread of rank 0 that only
    moves encoded frames and reads replies to host arrays, so rank 0's own shards run on the main thread
    meanwhile; the replies are joined on the main thread.
    """

    shards = True

    def __init__(self, connections: dict[int, Connection], objects: dict[str, object]):
        self._connections = connections
        self._objects = objects
        self._significant_counts: dict[int, np.ndarray] = {}
        self._calls = 0

    def score_halves(self, scored: dict) -> dict:
        """The halves' scorer results from ``scored`` (half index -> the scorer's return value): a
        ``DeferredTomoHalf`` is scored in shards and completed; any other value is already the half's result."""

        from relax.parallel.shards import heaviest_first, particle_costs
        from relax.steps.score_outputs import merge_half_shards
        from relax.steps.tomo_shards import DeferredTomoHalf

        deferred = {k: item for k, item in scored.items() if isinstance(item, DeferredTomoHalf)}
        if not deferred:
            return dict(scored)
        self._calls += 1
        clock = Stopwatch()
        ranks = len(self._connections) + 1
        shards_by_half, costs_by_task, jobs = {}, [], {}
        for k, item in deferred.items():
            dataset = item.job.data.particles.dataset
            if dataset is not self._objects[f"dataset_half{k + 1}"]:
                raise RuntimeError(f"half {k + 1}'s call does not score the run's half-{k + 1} dataset, so it cannot be sent")
            previous = self._significant_counts.get(k)
            costs = particle_costs(
                previous if previous is not None and previous.size == item.job.n_units else None,
                np.diff(dataset.unit_image_offsets),
            )
            jobs[k], shards_by_half[k] = item.job.cut(costs, ranks)
            costs_by_task.extend(((k, s), float(costs[units].sum())) for s, units in enumerate(shards_by_half[k]))
        queue = list(reversed(heaviest_first(costs_by_task)))
        lock, stop = threading.Lock(), threading.Event()
        job_frames = {k: channel.encode(ShardJob((self._calls, k), job), self._objects) for k, job in jobs.items()}
        task_frames = {
            (k, s): channel.encode(ShardTask((self._calls, k), k, units), self._objects)
            for k, shards in shards_by_half.items()
            for s, units in enumerate(shards)
        }
        replies, ran, failures = {}, {rank: [] for rank in range(ranks)}, []

        def next_task():
            with lock:
                return None if stop.is_set() or not queue else queue.pop()

        def serve(rank: int) -> None:
            connection, sent = self._connections[rank], []
            try:
                while (key := next_task()) is not None:
                    if key[0] not in sent:
                        channel.send_frames(connection, job_frames[key[0]])
                        sent.append(key[0])
                    channel.send_frames(connection, task_frames[key])
                    reply = channel.decode(channel.receive_frames(connection, peer=f"rank {rank}"), {}, host_arrays=True)
                    if isinstance(reply, HalfFailure):
                        raise channel.RankFailed(f"rank {rank} failed scoring a shard of half {reply.half_index + 1}:\n{reply.text}")
                    replies[key] = reply
                    ran[rank].append(reply.seconds)
                if sent:
                    channel.send(connection, ReleaseShardJobs(tuple((self._calls, k) for k in sent)), {})
            except BaseException as error:
                failures.append(error)
                stop.set()

        threads = [threading.Thread(target=serve, args=(rank,), daemon=True) for rank in self._connections]
        for thread in threads:
            thread.start()
        try:
            while (key := next_task()) is not None:
                task_clock = Stopwatch()
                result = jobs[key[0]].shard(shards_by_half[key[0]][key[1]]).score()
                replies[key] = HalfReply(result, None, None, None, task_clock.seconds)
                ran[0].append(task_clock.seconds)
        except BaseException:
            stop.set()
            raise
        for thread in threads:
            thread.join()
        if failures:
            raise failures[0]
        # The encoded calls (each half's projector among them) are not needed for the merge.
        job_frames.clear()
        task_frames.clear()
        done = {}
        for k, item in deferred.items():
            parts = []
            for s in range(len(shards_by_half[k])):
                reply = replies.pop((k, s))
                if reply.pass_engines is not None:
                    engine_record.append_taken(reply.pass_engines, reply.coarse_engine_calls)
                    self._objects["local_profile_history"].extend(reply.local_profile_rows)
                parts.append(reply.result)
            result = merge_half_shards(parts, shards_by_half[k], item.job.n_units)
            del parts
            if result.significant_counts is not None:
                self._significant_counts[k] = np.asarray(result.significant_counts)
            done[k] = item.complete(result)
        logger.info(
            "Shards of halves %s: %s shards (%s alone) on %d ranks in %.1fs; per rank shards/busy s: %s",
            ",".join(str(k + 1) for k in deferred),
            "+".join(str(len(shards_by_half[k])) for k in deferred),
            "+".join(str(sum(units.size == 1 for units in shards_by_half[k])) for k in deferred),
            ranks,
            clock.seconds,
            " ".join(f"{rank}:{len(seconds)}/{sum(seconds):.1f}" for rank, seconds in ran.items()),
        )
        return {k: done.get(k, item) for k, item in scored.items()}


def half_results(score, calls: dict, remote_halves):
    """Each half's scored result, ``(half index, result)`` in the order of ``calls`` (half index -> the arguments
    of ``score``), as the one-GPU loop takes them: a half is scored when its turn comes, after the previous half
    was published.

    ``remote_halves`` is None on one GPU. A :class:`RemoteHalves` gets the calls of the halves its ranks own sent
    first; a :class:`ShardPool` scores every half's subtomogram call in shards before the first is returned
    (``score(..., defer_tomo=True)``).
    """

    if remote_halves is not None and remote_halves.shards:
        scored = remote_halves.score_halves({k: score(*args, defer_tomo=True) for k, args in calls.items()})
        for k in calls:
            yield k, scored.pop(k)
        return
    remote = () if remote_halves is None else tuple(k for k in calls if remote_halves.remote(k))
    for k in remote:
        remote_halves.submit(k, score, *calls[k])
    for k, args in calls.items():
        yield k, remote_halves.result(k) if k in remote else score(*args)


def accept_remote_halves(
    link: RankLink, objects: dict[str, object], fingerprint: dict, *, tomographic: bool, n_classes: int
) -> RemoteHalves | ShardPool:
    """Accept the other ranks and check that each loaded the same run (``fingerprint``). With two ranks,
    rank 1 scores half 2; with three or four, a K=1 subtomogram run scores both halves' shards on every rank."""

    if link.world > 2 and not (tomographic and n_classes == 1):
        raise NotImplementedError(
            f"--gpu lists {link.world} devices: more than two GPUs score K=1 subtomogram halves in shards only; "
            "run single particles or Class3D on two GPUs (one per half)"
        )
    if not 2 <= link.world <= 4:
        raise NotImplementedError(f"a multi-GPU run takes two to four GPUs on one node; --gpu lists {link.world}")
    clock = Stopwatch()
    connections = channel.accept_ranks(link.listener, link.world)
    for rank, connection in connections.items():
        theirs = channel.receive(connection, objects, peer=f"rank {rank}")
        if theirs != fingerprint:
            raise channel.RankFailed(f"rank {rank} loaded another run: {theirs} against rank 0's {fingerprint}")
    if link.world > 2:
        logger.info(
            "Multi-GPU run: %d ranks connected in %.1fs; every rank scores shards of both halves",
            link.world,
            clock.seconds,
        )
        return ShardPool(connections, objects)
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
    shard_jobs = {}
    with stable_window_class_history():
        while True:
            try:
                task = channel.receive(connection, objects, peer="rank 0")
            except channel.RankClosed:
                logger.info("Rank 0 closed the connection: the run is over")
                return
            if isinstance(task, ShardJob):
                shard_jobs[task.key] = task.job
                continue
            if isinstance(task, ReleaseShardJobs):
                for key in task.keys:
                    shard_jobs.pop(key, None)
                return_freed_heap("the released shard calls on this rank")
                continue
            local_profile_history.clear()
            clock = Stopwatch()
            try:
                with expectation_relion_preprocess_checks():
                    if isinstance(task, ShardTask):
                        result = shard_jobs[task.key].shard(task.units).score()
                    else:
                        result = task.call(*task.args)
            except BaseException:
                text = traceback.format_exc()
                logger.error("Scoring half %d failed on this rank:\n%s", task.half_index + 1, text)
                try:
                    channel.send(connection, HalfFailure(task.half_index, text), objects)
                except Exception as send_error:  # rank 0 is gone: the scoring error above is the one raised
                    logger.error("Could not report the failure to rank 0: %s", send_error)
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
            return_freed_heap(f"half {task.half_index + 1}'s scoring on this rank")
