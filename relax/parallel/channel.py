"""The channel between the ranks of a multi-GPU refinement: pickled records over an abstract Unix socket.

Rank 0 listens at the launcher's address (:mod:`relax_cli.launch`) and every other rank connects to it
once, when its start-up is done. A message is a pickle (protocol 5) of one record. Its arrays travel as
separate frames, without a copy into the pickle: a NumPy array as its buffer, a JAX array as its host copy,
placed on the receiving rank's device when it is read (bitwise the same values, dtype and weak type).

Objects that each rank holds for the whole run (its datasets, its run context, its options) are never sent:
both ends name them in a table of run objects, and a reference to one travels as its name and is read back as
the receiving rank's own object (``persistent_id``). The table must name the same objects on every rank.

A closed connection is an error: a rank that waits on a rank that died raises :class:`RankFailed`.
"""

from __future__ import annotations

import io
import pickle
import time
from multiprocessing.connection import Client, Connection, Listener

import numpy as np

# How long a follower retries its connection while rank 0 has not started listening yet.
CONNECT_TIMEOUT_SECONDS = 3600.0
_CONNECT_RETRY_SECONDS = 1.0


class RankFailed(RuntimeError):
    """The other end of a rank connection closed or reported a failure."""


def _restore_device_array(host: np.ndarray, weak_type: bool):
    import jax
    from jax._src.lax import lax as lax_internal

    value = jax.device_put(host)
    return lax_internal._convert_element_type(value, value.dtype, weak_type=True) if weak_type else value


class _Pickler(pickle.Pickler):
    def __init__(self, file, *, names_by_id: dict[int, str], buffer_callback):
        super().__init__(file, protocol=5, buffer_callback=buffer_callback)
        self._names_by_id = names_by_id

    def persistent_id(self, obj):
        return self._names_by_id.get(id(obj))

    def reducer_override(self, obj):
        import jax

        if isinstance(obj, jax.Array):
            return _restore_device_array, (np.asarray(obj), bool(getattr(obj, "weak_type", False)))
        return NotImplemented


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, *, objects: dict[str, object], buffers):
        super().__init__(file, buffers=buffers)
        self._objects = objects

    def persistent_load(self, name):
        if name not in self._objects:
            raise pickle.UnpicklingError(f"run object {name!r} is not in this rank's table")
        return self._objects[name]


def encode(record, objects: dict[str, object]) -> list:
    """``record`` as frames: the pickle, then each out-of-band buffer. ``objects`` is the run-object table."""

    names_by_id = {id(obj): name for name, obj in objects.items()}
    buffers = []
    stream = io.BytesIO()
    _Pickler(stream, names_by_id=names_by_id, buffer_callback=buffers.append).dump(record)
    return [stream.getvalue(), *(buffer.raw() for buffer in buffers)]


def decode(frames: list, objects: dict[str, object]):
    """The record of :func:`encode`'s frames, with run objects read from this rank's ``objects``."""

    return _Unpickler(io.BytesIO(frames[0]), objects=objects, buffers=frames[1:]).load()


def send(connection: Connection, record, objects: dict[str, object]) -> None:
    """Send one record: the frame count, then the frames."""

    frames = encode(record, objects)
    connection.send(len(frames))
    for frame in frames:
        connection.send_bytes(frame)


def receive(connection: Connection, objects: dict[str, object], *, peer: str):
    """Receive one record from ``peer`` (named in errors). The buffers are writable copies."""

    try:
        count = connection.recv()
        frames = [
            connection.recv_bytes() if index == 0 else bytearray(connection.recv_bytes()) for index in range(count)
        ]
    except (EOFError, ConnectionError, OSError) as error:
        raise RankFailed(f"the connection to {peer} closed ({type(error).__name__}): that rank stopped") from error
    return decode(frames, objects)


def listen(address: str, authkey: bytes) -> Listener:
    """Rank 0's listener at the launcher's address."""

    return Listener(address, family="AF_UNIX", authkey=authkey)


def accept_ranks(listener: Listener, world: int) -> dict[int, Connection]:
    """Accept every other rank once; each first sends its rank number. Returns the connections by rank."""

    connections = {}
    while len(connections) < world - 1:
        connection = listener.accept()
        rank = connection.recv()
        if rank in connections or not 0 < rank < world:
            raise RankFailed(f"rank {rank} connected twice or is not a rank of a {world}-rank run")
        connections[rank] = connection
    return connections


def connect(address: str, authkey: bytes, rank: int, *, timeout_seconds: float = CONNECT_TIMEOUT_SECONDS) -> Connection:
    """A follower's connection to rank 0, retried until rank 0 listens (at most ``timeout_seconds``)."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            connection = Client(address, family="AF_UNIX", authkey=authkey)
            break
        except (ConnectionRefusedError, FileNotFoundError):
            if time.monotonic() > deadline:
                raise RankFailed(f"rank {rank} could not reach rank 0 within {timeout_seconds:.0f} s") from None
            time.sleep(_CONNECT_RETRY_SECONDS)
    connection.send(rank)
    return connection
