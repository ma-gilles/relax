"""The image tiles of a PPCA update scored on several GPUs (``relax ppca_initial_model --gpu 0:1``).

A multi-GPU run is one process per GPU (:mod:`relax_cli.launch`), as for ``relax refine``. Rank 0 runs the
controller (:func:`relax.ppca_initial_model.iteration_loop.run`): initialization, batch selection, the update,
checkpoints and every output. The other ranks load the same particles and then :func:`serve_tile_ranks`: for
each E-step rank 0 sends the model and the id groups (:class:`ExpectationTask`), every rank prepares the same
streams, rank 0 sends the tile size it planned (:class:`TilePlan`), and every rank cuts the same tiles. The tiles
form fixed blocks (:func:`tile_blocks`: runs of consecutive tiles of about equal predicted cost), which the ranks
take from rank 0's queue as they finish the previous one, heaviest first (:meth:`LeaderTiles.run_blocks`, one block
sent ahead while the queue is long), so a rank whose tiles run slower than predicted takes fewer. A block returns its tiles' statistics
without their volumes and its volume sums per id group in tile order (:class:`BlockShare`); rank 0 adds the blocks'
volume sums in block order and puts every other statistic back in tile order (``iteration_loop._join_tile_shares``).
Section 18 of ``docs/math/vdam_ppca_algorithm.md`` states what is summed where and how the result compares with one
GPU.

The blocks are a function of the tile costs and the number of ranks, and every sum has a fixed order, so a run is
reproducible for a given number of GPUs: neither which rank scored a block nor when changes it.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection, Listener

import numpy as np

from relax.parallel import channel
from relax.runtime.timing import Stopwatch

logger = logging.getLogger("relax.ppca_initial_model.iteration_loop")


@dataclass(frozen=True)
class ExpectationTask:
    """One E-step: ``_expectation(dataset, state, config, groups, iteration, ...)`` as rank 0 runs it. ``state``
    holds the model fields the E-step reads (:func:`expectation_state`)."""

    key: int
    state: object
    groups: list
    iteration: int
    embeddings_only: bool
    diameter_ang: float | None
    noise_inflation: float


@dataclass(frozen=True)
class TilePlan:
    """The tile size rank 0 planned for E-step ``key``; every rank cuts its tiles with it."""

    key: int
    tile_size: int


@dataclass(frozen=True)
class BlockTask:
    """Score block ``block`` of E-step ``key``; the reply is a :class:`BlockShare`."""

    key: int
    block: int


@dataclass(frozen=True)
class BlockShare:
    """Block ``block`` of E-step ``key``: each of its tiles' statistics by tile index (without the volumes, which
    ``totals`` holds: per id group, the volume sums over the block's tiles in tile order), and its seconds."""

    key: int
    block: int
    parts: dict
    totals: list
    seconds: float


@dataclass(frozen=True)
class EndBlocks:
    """No more blocks of E-step ``key``; the reply is a :class:`RankDone`."""

    key: int


@dataclass(frozen=True)
class RankDone:
    """A rank's E-step ``key`` is over: its phases (``iteration_loop._Laps``)."""

    key: int
    timing: dict


@dataclass(frozen=True)
class TileFailure:
    """A rank's E-step raised: the traceback, as text."""

    text: str


# The most ranks of a PPCA run: the GPUs of one node (relax_cli.launch.MAX_RANKS).
MAX_WORLD = 8

# Blocks per rank of an E-step: enough that the last blocks even out the ranks' measured speeds, few enough that a
# block (one accumulate call, one reply) holds several tiles.
BLOCKS_PER_RANK = 8

# A rank is sent its next block ahead (while it scores one, hiding the reply's round trip) only while the queue holds
# at least this many blocks per rank. A block sent ahead at the end is one a rank that finishes first cannot take:
# with one ahead to the end, rank 0 (which takes blocks as it finishes them) waited a block's time at the end of
# every ET update on four H100 (job 7475535: about 1 s of 7 to 8 s, 0.8 s of each update's loss to the busiest rank).
AHEAD_QUEUE_PER_RANK = 2


def tile_blocks(costs, world: int) -> list[list[int]]:
    """The E-step's blocks: ``min(tiles, BLOCKS_PER_RANK * world)`` runs of consecutive tiles, cut where the
    cumulative predicted cost crosses each equal share. A function of ``costs`` and ``world`` alone."""

    costs = np.asarray(costs, np.float64)
    n = costs.size
    count = min(n, BLOCKS_PER_RANK * world)
    cumulative = np.cumsum(costs)
    blocks, start = [], 0
    for j in range(1, count + 1):
        end = n if j == count else int(np.searchsorted(cumulative, cumulative[-1] * j / count, side="left")) + 1
        end = min(max(end, start + 1), n - (count - j))  # every block keeps a tile, and so do the ones after it
        blocks.append(list(range(start, end)))
        start = end
    return blocks


def expectation_state(state):
    """The model fields an E-step reads (theta, noise, offset variance, direction prior), without the optimizer's
    moments, momentum and batch order, which only rank 0 keeps."""

    return dataclasses.replace(state, moments=None, sgd_momentum=None, order=None, rng_state=None, initialization=None)


def run_fingerprint(dataset, config, identity, input_poses=None) -> dict:
    """What the ranks must agree on before rank 0 sends them work: the inputs and code (``identity``), the
    configuration, the particle count and grids, the pose prior's input rotations and the device model."""

    import jax

    from relax.ppca_refinement.pose_prior import centres_sha256

    return {
        "identity": identity,
        "config": dataclasses.asdict(config),
        "n_images": int(dataset.n_images),
        "image_shape": tuple(int(s) for s in dataset.image_shape),
        "input_poses_sha256": None if input_poses is None else centres_sha256(input_poses),
        "device_kind": jax.devices()[0].device_kind,
    }


def run_objects(dataset, config) -> dict[str, object]:
    """What every rank holds for the whole run, by name; a reference to one travels as its name."""

    return {"dataset": dataset, "config": config}


def require_tile_route(config, world: int) -> None:
    """Refuse a multi-GPU run of what it does not support."""

    if not 2 <= world <= MAX_WORLD:
        raise NotImplementedError(
            f"a multi-GPU PPCA run takes two to {MAX_WORLD} GPUs of one node; --gpu lists {world}"
        )
    if not config.stream_coarse_recompute:
        raise NotImplementedError(
            "--gpu with more than one device splits the streamed engine's tiles; the host-mask reference engine "
            "(--no-stream-coarse-recompute) runs on one GPU"
        )
    if config.fixed_core is not None:
        raise NotImplementedError(
            "--fixed-core fits per-particle core scales on all particles before training; run it on one GPU"
        )


def require_gpu_ranks(gpu, ranks) -> None:
    """Refuse a process whose rank count is not ``--gpu``'s device count (``ranks``: this process's
    ``relax_cli.launch.RankEnvironment``, None outside a multi-GPU run): an entry point that skipped the ``relax``
    console script would run the whole update on each GPU."""

    from relax_cli import launch

    devices = 0 if gpu is None else len(launch.gpu_devices(gpu))
    world = 1 if ranks is None else ranks.world
    if max(devices, 1) != world:
        raise RuntimeError(
            f"--gpu {gpu!r} lists {devices} devices but this process is one of {world} ranks: start multi-GPU "
            "PPCA runs through `relax ppca_initial_model` or `python -m relax_cli ppca_initial_model`"
        )


class LeaderTiles:
    """Rank 0's side of one E-step: started by :meth:`TileRanks.begin`, which sent the task."""

    rank = 0

    def __init__(self, ranks: TileRanks, key: int):
        self._ranks = ranks
        self.key = key
        self.world = ranks.world
        self.record = None

    def plan(self, tile_size: int) -> int:
        """Send the tile size planned here to every rank."""

        channel.broadcast(self._ranks.connections, TilePlan(self.key, int(tile_size)), self._ranks.objects)
        return int(tile_size)

    def run_blocks(self, blocks, costs, run_block, laps) -> list[BlockShare]:
        """Every block's share, in block order: rank 0 scores blocks on this thread (``run_block(b)`` returns the
        block's parts and totals) while one thread per other rank hands that rank the next block, heaviest first,
        one ahead of the block it is scoring while the queue holds :data:`AHEAD_QUEUE_PER_RANK` blocks per rank.
        The threads only move frames; the replies are read here, those that arrived after each of rank 0's own
        blocks (their volume sums go to the device while rank 0 scores its next block) and the rest at the end."""

        costs = np.asarray(costs, np.float64)
        block_cost = [float(costs[block].sum()) for block in blocks]
        queue = deque(sorted(range(len(blocks)), key=lambda b: (-block_cost[b], b)))
        lock, stop = threading.Lock(), threading.Event()
        task_frames = [channel.encode(BlockTask(self.key, b), {}) for b in range(len(blocks))]
        end_frames = channel.encode(EndBlocks(self.key), {})
        received, done, failures = {}, {}, []
        ran = {rank: [] for rank in range(self.world)}

        def next_block(ahead=False):
            with lock:
                if stop.is_set() or not queue or (ahead and len(queue) < AHEAD_QUEUE_PER_RANK * self.world):
                    return None
                return queue.popleft()

        def serve(rank, connection):
            sent = deque()
            try:
                while True:
                    while len(sent) < 2 and (b := next_block(ahead=bool(sent))) is not None:
                        channel.send_frames(connection, task_frames[b])
                        sent.append(b)
                    if not sent:
                        break
                    frames = channel.receive_frames(connection, peer=f"rank {rank}")
                    with lock:
                        received[sent[0]] = (rank, frames)
                    ran[rank].append(sent.popleft())
                channel.send_frames(connection, end_frames)
                done[rank] = channel.receive_frames(connection, peer=f"rank {rank}")
            except BaseException as error:
                failures.append(error)
                stop.set()

        threads = [
            threading.Thread(target=serve, args=(rank, connection), daemon=True)
            for rank, connection in sorted(self._ranks.connections.items())
        ]
        for thread in threads:
            thread.start()
        own, read = {}, {}

        def read_replies():
            with lock:
                arrived = [(b, received.pop(b)) for b in list(received)]
            for b, (rank, frames) in arrived:
                reply = channel.decode(frames, self._ranks.objects)
                if isinstance(reply, TileFailure):
                    raise channel.RankFailed(f"rank {rank} failed a block of an E-step:\n{reply.text}")
                if not isinstance(reply, BlockShare) or (reply.key, reply.block) != (self.key, b):
                    raise channel.RankFailed(f"rank {rank} returned {reply!r} for block {b} of E-step {self.key}")
                read[b] = reply

        try:
            while (b := next_block()) is not None:
                clock = Stopwatch()
                parts, totals = run_block(b)
                own[b] = BlockShare(self.key, b, parts, totals, clock.seconds)
                ran[0].append(b)
                read_replies()
        except BaseException:
            stop.set()
            raise
        finally:
            for thread in threads:
                thread.join()
        laps.mark("own_blocks_then_wait")
        read_replies()
        shares, timing = [], {0: laps.record()}
        for b in range(len(blocks)):
            if b in own:
                shares.append(own.pop(b))
            elif b in read:
                shares.append(read.pop(b))
            else:
                break
        if failures:
            raise failures[0]
        if len(shares) != len(blocks):
            raise channel.RankFailed(f"E-step {self.key}: {len(blocks) - len(shares)} blocks never returned")
        for rank, frames in sorted(done.items()):
            reply = channel.decode(frames, {})
            if not isinstance(reply, RankDone) or reply.key != self.key:
                raise channel.RankFailed(f"rank {rank} ended E-step {self.key} with {reply!r}")
            timing[rank] = reply.timing
        seconds = {share.block: share.seconds for share in shares}
        self.record = {
            "world": self.world,
            "blocks": [len(ran[rank]) for rank in range(self.world)],
            "tiles": [sum(len(blocks[b]) for b in ran[rank]) for rank in range(self.world)],
            "cost": [round(sum(block_cost[b] for b in ran[rank])) for rank in range(self.world)],
            "seconds": [round(sum(seconds[b] for b in ran[rank]), 3) for rank in range(self.world)],
            "timing": [timing[rank] for rank in range(self.world)],
        }
        logger.info(
            "E-step %d on %d ranks: blocks %s, tiles %s, cost %s, busy seconds %s",
            self.key,
            self.world,
            self.record["blocks"],
            self.record["tiles"],
            self.record["cost"],
            self.record["seconds"],
        )
        return shares


class TileRanks:
    """Rank 0's link to the other ranks for the whole run: ``connections`` by rank, the run objects."""

    def __init__(self, connections: dict[int, Connection], objects: dict[str, object]):
        self.connections = connections
        self.objects = objects
        self.world = len(connections) + 1
        self._calls = 0

    def begin(self, state, groups, iteration, *, embeddings_only, diameter_ang, noise_inflation) -> LeaderTiles:
        """Send one E-step to every rank (they prepare their streams while rank 0 prepares its own)."""

        self._calls += 1
        task = ExpectationTask(
            self._calls,
            expectation_state(state),
            [np.asarray(ids) for ids in groups],
            int(iteration),
            bool(embeddings_only),
            diameter_ang,
            float(noise_inflation),
        )
        channel.broadcast(self.connections, task, self.objects)
        return LeaderTiles(self, self._calls)

    def close(self) -> None:
        """End the run: the other ranks' task loops return when their connection closes."""

        for connection in self.connections.values():
            connection.close()


class FollowerTiles:
    """A rank's side of one E-step that rank 0 sent it."""

    def __init__(
        self, connection: Connection, objects: dict[str, object], task: ExpectationTask, rank: int, world: int
    ):
        self._connection = connection
        self._objects = objects
        self.key = task.key
        self.rank = rank
        self.world = world

    def begin(self, state, groups, iteration, **_) -> FollowerTiles:
        return self

    def plan(self, tile_size: int) -> int:
        """Rank 0's tile size; this rank's own plan (``tile_size``) must hold it."""

        plan = channel.receive(self._connection, self._objects, peer="rank 0")
        if not isinstance(plan, TilePlan) or plan.key != self.key:
            raise channel.RankFailed(f"rank {self.rank} expected the tile plan of E-step {self.key}, got {plan!r}")
        if plan.tile_size > tile_size:
            raise RuntimeError(
                f"rank 0 planned {plan.tile_size} images per tile; rank {self.rank}'s device fits {tile_size}"
            )
        return plan.tile_size

    def run_blocks(self, blocks, costs, run_block, laps) -> None:
        """Score each block rank 0 sends and return its share, until rank 0 ends the E-step."""

        while True:
            task = channel.receive(self._connection, self._objects, peer="rank 0")
            if isinstance(task, EndBlocks) and task.key == self.key:
                laps.mark("blocks_wait")
                channel.send(self._connection, RankDone(self.key, laps.record()), self._objects)
                return None
            if not isinstance(task, BlockTask) or task.key != self.key:
                raise channel.RankFailed(f"rank {self.rank} expected a block of E-step {self.key}, got {task!r}")
            laps.mark("blocks_wait")
            clock = Stopwatch()
            parts, totals = run_block(task.block)
            channel.send(
                self._connection, BlockShare(self.key, task.block, parts, totals, clock.seconds), self._objects
            )
            laps.mark("blocks_send")


def accept_tile_ranks(listener: Listener, world: int, objects: dict[str, object], fingerprint: dict) -> TileRanks:
    """Accept the other ranks and check that each loaded the same run (``fingerprint``)."""

    clock = Stopwatch()
    connections = channel.accept_ranks(listener, world)
    for rank, connection in connections.items():
        theirs = channel.receive(connection, objects, peer=f"rank {rank}")
        if theirs != fingerprint:
            raise channel.RankFailed(f"rank {rank} loaded another run: {theirs} against rank 0's {fingerprint}")
    logger.info(
        "Multi-GPU PPCA: %d ranks connected in %.1fs; every rank scores tiles of each E-step", world, clock.seconds
    )
    return TileRanks(connections, objects)


def serve_tile_ranks(
    connection: Connection, rank: int, world: int, *, dataset, config, fingerprint: dict, input_poses=None
) -> None:
    """A rank's task loop: report the run's fingerprint, then score this rank's tiles of each E-step rank 0 sends,
    until rank 0 closes the connection (the run's end). A failed E-step is reported to rank 0 and raised here, so
    this rank exits non-zero. ``input_poses``: this rank's own read of the pose prior's centres (the fingerprint
    holds their hash)."""

    from relax.ppca_initial_model import iteration_loop
    from relax.runtime.host_memory import return_freed_heap

    objects = run_objects(dataset, config)
    channel.send(connection, fingerprint, objects)
    while True:
        try:
            task = channel.receive(connection, objects, peer="rank 0")
        except channel.RankClosed:
            logger.info("Rank 0 closed the connection: the run is over")
            return
        try:
            iteration_loop._expectation(
                dataset,
                task.state,
                config,
                task.groups,
                task.iteration,
                embeddings_only=task.embeddings_only,
                diameter_ang=task.diameter_ang,
                noise_inflation=task.noise_inflation,
                ranks=FollowerTiles(connection, objects, task, rank, world),
                input_poses=input_poses,
            )
        except BaseException:
            text = traceback.format_exc()
            logger.error("E-step %d failed on rank %d:\n%s", task.key, rank, text)
            try:
                channel.send(connection, TileFailure(text), objects)
            except Exception as send_error:  # rank 0 is gone: the error above is the one raised
                logger.error("Could not report the failure to rank 0: %s", send_error)
            raise
        del task
        return_freed_heap(f"E-step tiles on rank {rank}")
