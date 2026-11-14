"""The image tiles of a PPCA update scored on several GPUs (``relax ppca_initial_model --gpu 0:1``).

A multi-GPU run is one process per GPU (:mod:`relax_cli.launch`), as for ``relax refine``. Rank 0 runs the
controller (:func:`relax.ppca_initial_model.iteration_loop.run`): initialization, batch selection, the update,
checkpoints and every output. The other ranks load the same particles and then :func:`serve_tile_ranks`: for
each E-step rank 0 sends the model and the id groups (:class:`ExpectationTask`), every rank prepares the same
streams, rank 0 sends the tile size it planned (:class:`TilePlan`), and every rank cuts the same tiles and scores
the tiles :func:`assign_tiles` gives it. A rank returns its tiles' statistics without their volumes and its
running volume sums per id group (:class:`TileShare`); rank 0 joins them in tile order and adds the volume sums
in rank order (``iteration_loop._join_tile_shares``). Section 18 of ``docs/math/vdam_ppca_algorithm.md`` states
what is summed where and how the result compares with one GPU.

The assignment is static (a function of the tile costs and the number of ranks), and each sum has a fixed order,
so a run is reproducible for a given number of GPUs: neither which rank finished first nor when changes it.
"""

from __future__ import annotations

import dataclasses
import logging
import traceback
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
class TileShare:
    """A rank's part of E-step ``key``: each of its tiles' statistics by tile index (without the volumes, which
    ``totals`` holds: per id group, the volume sums over its tiles in tile order), and its seconds."""

    key: int
    parts: dict
    totals: list
    seconds: float


@dataclass(frozen=True)
class TileFailure:
    """A rank's E-step raised: the traceback, as text."""

    text: str


def assign_tiles(costs, world: int) -> np.ndarray:
    """The rank of each tile: heaviest first, each to the rank with the least cost so far (the lowest rank on a
    tie). A function of ``costs`` and ``world`` alone, so every rank computes the same assignment."""

    costs = np.asarray(costs, np.float64)
    owners = np.empty(costs.size, np.int64)
    load = np.zeros(world)
    for k in sorted(range(costs.size), key=lambda k: (-costs[k], k)):
        rank = int(np.argmin(load))  # the first minimum: the lowest rank on a tie
        owners[k] = rank
        load[rank] += costs[k]
    return owners


def expectation_state(state):
    """The model fields an E-step reads (theta, noise, offset variance, direction prior), without the optimizer's
    moments, momentum and batch order, which only rank 0 keeps."""

    return dataclasses.replace(state, moments=None, sgd_momentum=None, order=None, rng_state=None, initialization=None)


def run_fingerprint(dataset, config, identity) -> dict:
    """What the ranks must agree on before rank 0 sends them work: the inputs and code (``identity``), the
    configuration, the particle count and grids, and the device model."""

    import jax

    return {
        "identity": identity,
        "config": dataclasses.asdict(config),
        "n_images": int(dataset.n_images),
        "image_shape": tuple(int(s) for s in dataset.image_shape),
        "device_kind": jax.devices()[0].device_kind,
    }


def run_objects(dataset, config) -> dict[str, object]:
    """What every rank holds for the whole run, by name; a reference to one travels as its name."""

    return {"dataset": dataset, "config": config}


def require_tile_route(config, world: int) -> None:
    """Refuse a multi-GPU run of what it does not support."""

    if not 2 <= world <= 4:
        raise NotImplementedError(f"a multi-GPU PPCA run takes two to four GPUs of one node; --gpu lists {world}")
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
        self._gather = None
        self._clock = Stopwatch()
        self.record = None

    def plan(self, tile_size: int) -> int:
        """Send the tile size planned here to every rank and start receiving their shares."""

        channel.broadcast(self._ranks.connections, TilePlan(self.key, int(tile_size)), self._ranks.objects)
        self._gather = channel.Gather(self._ranks.connections)
        return int(tile_size)

    def exchange(self, share: TileShare, costs, owners) -> list[TileShare]:
        """Every rank's share, rank 0's first, in rank order, once all arrived."""

        own_seconds = self._clock.seconds
        replies = self._gather.records(self._ranks.objects)
        for rank, reply in replies.items():
            if isinstance(reply, TileFailure):
                raise channel.RankFailed(f"rank {rank} failed its tiles of an E-step:\n{reply.text}")
            if reply.key != self.key:
                raise channel.RankFailed(f"rank {rank} returned E-step {reply.key}, not {self.key}")
        shares = [share, *replies.values()]
        costs = np.asarray(costs, np.float64)
        self.record = {
            "world": self.world,
            "tiles": [int(np.sum(owners == rank)) for rank in range(self.world)],
            "cost": [float(costs[owners == rank].sum()) for rank in range(self.world)],
            "seconds": [round(own_seconds, 3), *(round(reply.seconds, 3) for reply in replies.values())],
            "joined_seconds": round(self._clock.seconds, 3),
        }
        logger.info(
            "E-step %d on %d ranks: tiles %s, cost %s, seconds %s; joined at %.1fs",
            self.key,
            self.world,
            self.record["tiles"],
            [round(c) for c in self.record["cost"]],
            self.record["seconds"],
            self.record["joined_seconds"],
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
        self._clock = Stopwatch()

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

    def exchange(self, share: TileShare, costs, owners) -> None:
        """Send this rank's share to rank 0."""

        channel.send(self._connection, dataclasses.replace(share, seconds=self._clock.seconds), self._objects)
        return None


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


def serve_tile_ranks(connection: Connection, rank: int, world: int, *, dataset, config, fingerprint: dict) -> None:
    """A rank's task loop: report the run's fingerprint, then score this rank's tiles of each E-step rank 0 sends,
    until rank 0 closes the connection (the run's end). A failed E-step is reported to rank 0 and raised here, so
    this rank exits non-zero."""

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
