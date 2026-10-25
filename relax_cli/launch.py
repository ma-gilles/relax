"""Start one refinement process per GPU of RELION's ``--gpu`` list, before JAX starts.

``relax refine --gpu 0:1`` (RELION's syntax: device ids separated by ``:`` or ``,``) runs one process per
listed device. The process the user started becomes the launcher: it starts the ranks with the same command
line, each with ``CUDA_VISIBLE_DEVICES`` naming its one device, waits for them and exits with their status.
Rank 0 runs the refinement and the other ranks score the halves it sends them
(``relax.refinement.remote_halves``). A rank that fails ends the run: the launcher stops the other ranks and
exits non-zero. Without ``--gpu``, or with one device, nothing is launched and the process runs as before
(one listed device other than 0 restricts ``CUDA_VISIBLE_DEVICES`` to it).

The ``relax`` console script (:func:`relax_cli.main`) calls :func:`launch_ranks` before anything imports ``relax``,
whose import starts the JAX backend on every visible GPU; this module imports only the standard library.
"""

from __future__ import annotations

import os
import re
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

RANK_ENV = "RELAX_RANK"
WORLD_ENV = "RELAX_WORLD"
ADDRESS_ENV = "RELAX_RANK_ADDRESS"
AUTHKEY_ENV = "RELAX_RANK_AUTHKEY"
# How long a rank has to stop after SIGTERM before it is killed, and after rank 0 finished.
TERMINATE_GRACE_SECONDS = 30.0
FINISH_GRACE_SECONDS = 300.0
_POLL_SECONDS = 0.5


@dataclass(frozen=True)
class RankEnvironment:
    """This process's rank in a multi-GPU run, as its launcher set it: the rank, the number of ranks, the
    channel address (an abstract Unix socket) and its authentication key."""

    rank: int
    world: int
    address: str  # with its leading NUL
    authkey: bytes

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


def rank_environment(environ=None) -> RankEnvironment | None:
    """This process's :class:`RankEnvironment`, or None outside a multi-GPU run."""

    environ = os.environ if environ is None else environ
    if RANK_ENV not in environ:
        return None
    return RankEnvironment(
        rank=int(environ[RANK_ENV]),
        world=int(environ[WORLD_ENV]),
        address="\0" + environ[ADDRESS_ENV],  # an abstract socket: the leading NUL cannot travel in the environment
        authkey=bytes.fromhex(environ[AUTHKEY_ENV]),
    )


# The most ranks of one run: the GPUs of one node (relax.refinement.remote_halves).
MAX_RANKS = 4


def gpu_devices(value: str) -> tuple[str, ...]:
    """The device ids of RELION's ``--gpu`` value: ``"0:1"``, ``"0,1"`` or ``""`` (the default device).

    RELION separates the ranks' devices with ``:`` and one rank's devices with ``,``; relax runs one rank per
    listed device, so both separate ranks here. A device listed twice is refused: two ranks never share a GPU.
    """

    devices = tuple(token.strip() for token in re.split(r"[:,]", value) if token.strip())
    for device in devices:
        if not device.isdigit():
            raise ValueError(f"--gpu {value!r}: device ids are non-negative integers")
    if len(set(int(device) for device in devices)) != len(devices):
        raise ValueError(f"--gpu {value!r} lists a device twice; each rank needs its own GPU")
    return tuple(str(int(device)) for device in devices)


def gpu_argument(argv) -> str | None:
    """The value of ``--gpu`` in a command line (``--gpu X`` or ``--gpu=X``), or None when absent."""

    argv = list(argv)
    for index, token in enumerate(argv):
        if token == "--gpu":
            return argv[index + 1] if index + 1 < len(argv) else ""
        if token.startswith("--gpu="):
            return token[len("--gpu=") :]
    return None


def output_argument(argv) -> str | None:
    """The value of ``--output`` in a command line (``--output X`` or ``--output=X``), or None when absent."""

    argv = list(argv)
    for index, token in enumerate(argv):
        if token == "--output" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--output="):
            return token[len("--output=") :]
    return None


def physical_device(device: str, environ) -> str:
    """The ``CUDA_VISIBLE_DEVICES`` entry of device id ``device``: its entry in the visible list, or the id."""

    visible = [entry.strip() for entry in environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if entry.strip()]
    if not visible:
        return device
    if int(device) >= len(visible):
        raise ValueError(
            f"--gpu device {device} is not visible: CUDA_VISIBLE_DEVICES={environ['CUDA_VISIBLE_DEVICES']}"
        )
    return visible[int(device)]


def rank_directory(output: str, rank: int) -> str:
    """Where rank ``rank`` (> 0) writes: a hidden directory of the run's output, never the run files."""

    return os.path.join(output, f".rank{rank}")


def launch_ranks(argv, orig_argv, environ) -> int | None:
    """Start the ranks of a multi-GPU run and return the run's exit status, or None to run in this process.

    None when this process is a rank already, or when ``--gpu`` lists at most one device. Otherwise this
    process starts one rank per device with ``orig_argv`` (``sys.orig_argv``), waits for them, and returns 0
    when every rank exited 0, else the first failure's status (128 + signal for a signal), after stopping the
    other ranks. Rank ``r > 0`` writes its output to ``rank_directory(--output, r)``.
    """

    if RANK_ENV in environ:
        return None
    value = gpu_argument(argv)
    if value is None:
        return None
    devices = gpu_devices(value)
    if len(devices) > MAX_RANKS:
        raise ValueError(
            f"--gpu {value!r} lists {len(devices)} devices: a multi-GPU run takes at most {MAX_RANKS} GPUs of one node"
        )
    if len(devices) <= 1:
        if devices and devices[0] != "0":
            environ["CUDA_VISIBLE_DEVICES"] = physical_device(devices[0], environ)
        return None
    physical = [physical_device(device, environ) for device in devices]
    command = [sys.executable, *list(orig_argv)[1:]]
    address = f"relax-ranks-{os.getpid()}-{secrets.token_hex(8)}"
    authkey = secrets.token_hex(32)
    output = output_argument(argv)
    ranks = []
    for rank, device in enumerate(physical):
        env = dict(environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": device,
                RANK_ENV: str(rank),
                WORLD_ENV: str(len(physical)),
                ADDRESS_ENV: address,
                AUTHKEY_ENV: authkey,
            }
        )
        log = None
        if rank > 0 and output is not None:
            os.makedirs(rank_directory(output, rank), exist_ok=True)
            log = open(os.path.join(rank_directory(output, rank), f"rank{rank}.log"), "w")
        ranks.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT if log else None))
        print(f"relax: rank {rank} on CUDA device {device} (pid {ranks[-1].pid})", file=sys.stderr, flush=True)
    return wait_for_ranks(ranks)


def _status(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def _stop(ranks, grace_seconds: float) -> None:
    """SIGTERM every live rank, then SIGKILL those still alive after ``grace_seconds``."""

    for process in ranks:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    for process in ranks:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _reap(rank: int, process) -> int | None:
    """The rank's exit code once it exited (None while it runs), reaped here to report its peak host memory."""

    if process.returncode is not None:
        return process.returncode
    pid, status, usage = os.wait4(process.pid, os.WNOHANG)
    if pid == 0:
        return None
    process.returncode = os.waitstatus_to_exitcode(status)
    print(
        f"relax: rank {rank} exited with status {_status(process.returncode)}; "
        f"peak resident memory {usage.ru_maxrss / 1024**2:.2f} GiB",
        file=sys.stderr,
        flush=True,
    )
    return process.returncode


def wait_for_ranks(
    ranks, *, grace_seconds: float = TERMINATE_GRACE_SECONDS, finish_grace_seconds: float = FINISH_GRACE_SECONDS
) -> int:
    """Wait for the ranks; on the first non-zero exit, or a signal to this process, stop the others.

    Returns 0 when every rank exited 0. A rank still running ``finish_grace_seconds`` after rank 0 exited 0 is
    a failure (it did not stop when the run ended): it is stopped and the status is 1.
    """

    received = []

    def forward(signum, frame):
        received.append(signum)

    previous = {signum: signal.signal(signum, forward) for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        leader_done_at = None
        while True:
            if received:
                print(f"relax: launcher received signal {received[0]}; stopping the ranks", file=sys.stderr, flush=True)
                _stop(ranks, grace_seconds)
                return 128 + received[0]
            codes = [_reap(rank, process) for rank, process in enumerate(ranks)]
            failed = [(rank, code) for rank, code in enumerate(codes) if code not in (None, 0)]
            if failed:
                rank, code = failed[0]
                print(
                    f"relax: rank {rank} exited with status {_status(code)}; stopping the other ranks",
                    file=sys.stderr,
                    flush=True,
                )
                _stop(ranks, grace_seconds)
                return _status(code)
            if all(code == 0 for code in codes):
                return 0
            if codes[0] == 0:
                leader_done_at = time.monotonic() if leader_done_at is None else leader_done_at
                if time.monotonic() - leader_done_at > finish_grace_seconds:
                    print("relax: a rank did not stop after rank 0 finished; stopping it", file=sys.stderr, flush=True)
                    _stop(ranks, grace_seconds)
                    return 1
            time.sleep(_POLL_SECONDS)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
