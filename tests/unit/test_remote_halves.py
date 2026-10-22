"""A multi-GPU run (``relax refine --gpu 0:1``): the launcher, the rank channel and a half scored elsewhere.

The CPU tests run a tiny refinement whose half 2 is sent through the channel's encoding and back, as a rank
would receive and return it (``LoopbackHalves``): every record the controller sends must travel, and the run's
result must equal the one-GPU run's bitwise. The launcher tests start real processes.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.tiny_refinement import run_tiny_refinement

from relax.parallel import channel
from relax.refinement import iteration_loop, remote_halves
from relax.runtime import engine_record
from relax_cli import launch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "devices"),
    [("", ()), ("0", ("0",)), ("0:1", ("0", "1")), ("2,3", ("2", "3")), (" 1 : 0 ", ("1", "0"))],
)
def test_gpu_devices_read_relion_lists(value, devices):
    assert launch.gpu_devices(value) == devices


@pytest.mark.parametrize("value", ["0:0", "a", "0:-1"])
def test_gpu_devices_refuse_shared_or_invalid_ids(value):
    with pytest.raises(ValueError):
        launch.gpu_devices(value)


def test_gpu_and_output_arguments_are_read_before_parsing():
    argv = ["relax", "refine", "--data_dir", "d", "--output", "out", "--gpu", "0:1"]
    assert launch.gpu_argument(argv) == "0:1"
    assert launch.gpu_argument(["--gpu=1"]) == "1"
    assert launch.gpu_argument(["relax", "refine"]) is None
    assert launch.output_argument(argv) == "out"
    assert launch.output_argument(["--output=x"]) == "x"


def test_one_listed_device_runs_in_this_process():
    environ = {"CUDA_VISIBLE_DEVICES": "3,5"}
    assert launch.launch_ranks(["refine", "--gpu", "0"], [], environ) is None
    assert environ["CUDA_VISIBLE_DEVICES"] == "3,5"  # device 0 is the default device: nothing changes
    assert launch.launch_ranks(["refine", "--gpu", "1"], [], environ) is None
    assert environ["CUDA_VISIBLE_DEVICES"] == "5"
    assert launch.launch_ranks(["refine"], [], {"RELAX_RANK": "1"}) is None
    with pytest.raises(ValueError):
        launch.physical_device("2", {"CUDA_VISIBLE_DEVICES": "3,5"})


def _sleeper(seconds):
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def test_a_failed_rank_stops_the_others_and_the_run_fails():
    ranks = [_sleeper(120), subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"]), _sleeper(120)]
    start = time.monotonic()
    assert launch.wait_for_ranks(ranks, grace_seconds=5) == 3
    assert time.monotonic() - start < 30
    assert all(process.poll() is not None for process in ranks)


def test_a_killed_rank_fails_the_run_with_its_signal():
    ranks = [_sleeper(120), _sleeper(120)]
    ranks[1].kill()
    assert launch.wait_for_ranks(ranks, grace_seconds=5) == 128 + 9
    assert ranks[0].poll() is not None


def test_a_rank_that_does_not_stop_after_rank_0_fails_the_run():
    ranks = [subprocess.Popen([sys.executable, "-c", "pass"]), _sleeper(120)]
    assert launch.wait_for_ranks(ranks, grace_seconds=5, finish_grace_seconds=1) == 1
    assert ranks[1].poll() is not None


def test_all_ranks_exiting_zero_is_success():
    ranks = [subprocess.Popen([sys.executable, "-c", "pass"]) for _ in range(3)]
    assert launch.wait_for_ranks(ranks) == 0


def test_channel_records_keep_values_dtypes_weak_types_and_run_objects():
    run_object = object()
    objects = {"thing": run_object}
    weak = jnp.asarray(2.5)
    record = {
        "complex": jnp.asarray(np.arange(6, dtype=np.complex64).reshape(2, 3) * (1 + 2j)),
        "weak": weak,
        "host": np.arange(10, dtype=np.float64)[::2],
        "object_ref": run_object,
        "partial": partial(np.add, 1),
    }
    decoded = channel.decode(channel.encode(record, objects), {"thing": "theirs"})
    assert isinstance(decoded["complex"], jax.Array)
    assert decoded["complex"].dtype == jnp.complex64
    assert np.asarray(decoded["complex"]).tobytes() == np.asarray(record["complex"]).tobytes()
    assert decoded["weak"].weak_type and weak.weak_type
    assert np.array_equal(decoded["host"], record["host"])
    assert decoded["object_ref"] == "theirs"
    assert decoded["partial"](1) == 2
    with pytest.raises(pickle.UnpicklingError):
        channel.decode(channel.encode(record, objects), {})


def test_channel_over_a_connection_reports_a_closed_peer():
    from multiprocessing import Pipe

    ours, theirs = Pipe()
    channel.send(ours, {"x": np.ones(3)}, {})
    assert np.array_equal(channel.receive(theirs, {}, peer="rank 1")["x"], np.ones(3))
    ours.close()
    with pytest.raises(channel.RankFailed, match="rank 0"):
        channel.receive(theirs, {}, peer="rank 0")


class LoopbackHalves(remote_halves.RemoteHalves):
    """Half 2 scored in this process, through the channel encoding both ways, with its own engine records."""

    def __init__(self, objects):
        super().__init__({}, (0, 1), objects)
        self.tasks = {}
        self.sent = []

    def submit(self, half_index, call, *args):
        frames = channel.encode(remote_halves.HalfTask(half_index, call, args), self._objects)
        self.sent.append(sum(len(frame) for frame in frames))
        self.tasks[half_index] = channel.decode(frames, self._objects)

    def result(self, half_index):
        task = self.tasks.pop(half_index)
        published = (engine_record.take_pass_engines(), engine_record.take_coarse_engine_calls())
        result = task.call(*task.args)
        reply = remote_halves.HalfReply(
            result, engine_record.take_pass_engines(), engine_record.take_coarse_engine_calls(), [], 0.0
        )
        engine_record.append_taken(*published)
        reply = channel.decode(channel.encode(reply, self._objects), self._objects)
        engine_record.append_taken(reply.pass_engines, reply.coarse_engine_calls)
        return reply.result


def _flat_leaves(value, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(value, dict):
        for key in sorted(value):
            _flat_leaves(value[key], f"{prefix}.{key}", out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flat_leaves(item, f"{prefix}[{index}]", out)
    elif isinstance(value, (np.ndarray, jax.Array)) or np.isscalar(value):
        out[prefix] = np.asarray(value)
    return out


# K=1 runs its final pass after the iteration cap; Class3D its numbered iterations (the stand-in never converges).
@pytest.mark.parametrize("n_classes", [1, 2], ids=["k1-with-final-pass", "class3d"])
def test_half_2_scored_through_the_channel_matches_the_one_gpu_run(monkeypatch, n_classes):
    one_gpu = run_tiny_refinement(monkeypatch, n_classes=n_classes)
    loops = []

    def accept(link, objects, fingerprint):
        assert link.world == 2 and fingerprint["units"] == [5, 5]
        loops.append(LoopbackHalves(objects))
        return loops[-1]

    monkeypatch.setattr(remote_halves, "accept_remote_halves", accept)
    monkeypatch.setattr(
        iteration_loop,
        "refine_single_volume",
        partial(iteration_loop.refine_single_volume, ranks=remote_halves.RankLink(listener=None, world=2)),
    )
    two_gpu = run_tiny_refinement(monkeypatch, n_classes=n_classes)

    # Half 2 of each numbered iteration (two) and of the K=1 final pass went through the channel.
    assert len(loops) == 1 and len(loops[0].sent) == (3 if n_classes == 1 else 2) and not loops[0].tasks
    ours, theirs = _flat_leaves(one_gpu.archive_fields()), _flat_leaves(two_gpu.archive_fields())
    assert sorted(ours) == sorted(theirs)
    timings = ("time", "seconds", "wall")
    differing = [
        key for key in ours if ours[key].tobytes() != theirs[key].tobytes() and not any(t in key for t in timings)
    ]
    assert not differing, differing
    for ours_map, theirs_map in zip(
        _flat_leaves(one_gpu.maps.means).values(), _flat_leaves(two_gpu.maps.means).values()
    ):
        assert ours_map.tobytes() == theirs_map.tobytes()


def test_require_rank_route_refuses_observers_and_replays():
    from relax.parity.relion_replay_source import RelionReplay
    from relax.refinement.ports import RunObserver

    class Capture(RunObserver):
        pass

    class Source:
        relion_replay = None
        sealed_sampling_state = None

        def replays_relion_state(self):
            return False

    remote_halves.require_rank_route(RunObserver(), Source())
    with pytest.raises(NotImplementedError, match="observers"):
        remote_halves.require_rank_route(Capture(), Source())
    replay = Source()
    replay.relion_replay = RelionReplay()
    with pytest.raises(NotImplementedError, match="native"):
        remote_halves.require_rank_route(RunObserver(), replay)


_TWO_RANKS = r"""
import importlib.util, os, sys
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

launch = load("launch", sys.argv[1])
channel = load("channel", sys.argv[2])
ranks = launch.rank_environment()
if ranks is None:
    sys.exit(launch.launch_ranks(["x", "--gpu", "0:1", "--output", sys.argv[3]], sys.orig_argv, dict(os.environ)))
if ranks.is_leader:
    listener = channel.listen(ranks.address, ranks.authkey)
    (connection,) = channel.accept_ranks(listener, ranks.world).values()
    channel.send(connection, {"task": np.arange(4.0)}, {})
    reply = channel.receive(connection, {}, peer="rank 1")
    assert np.array_equal(reply["result"], np.arange(4.0) * 2), reply
    print("rank 0 got", reply["result"].tolist(), "from rank 1 on device", reply["device"])
else:
    connection = channel.connect(ranks.address, ranks.authkey, ranks.rank, timeout_seconds=60)
    task = channel.receive(connection, {}, peer="rank 0")
    channel.send(connection, {"result": task["task"] * 2, "device": os.environ["CUDA_VISIBLE_DEVICES"]}, {})
"""


def test_two_processes_launched_and_connected(tmp_path):
    """The launcher starts two ranks of a command; rank 1 reaches rank 0 through the environment it was given."""

    from pathlib import Path

    import relax.parallel.channel

    script = tmp_path / "two_ranks.py"
    script.write_text(_TWO_RANKS)
    env = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "7,9"}
    completed = subprocess.run(
        [sys.executable, str(script), launch.__file__, relax.parallel.channel.__file__, str(tmp_path / "out")],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "rank 0 got [0.0, 2.0, 4.0, 6.0] from rank 1 on device 9" in completed.stdout
    assert "peak resident memory" in completed.stderr
    assert Path(tmp_path / "out" / ".rank1" / "rank1.log").is_file()


def _python(code: str, *argv: str, tmp_path) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": str(Path(launch.__file__).resolve().parents[1])}  # fmt: skip
    return subprocess.run([sys.executable, "-c", code, *argv], env=env, capture_output=True, text=True, timeout=300,
                          cwd=tmp_path)  # fmt: skip


def test_importing_relax_with_gpu_in_argv_starts_no_process(tmp_path):
    """Only the console script launches ranks: ``import relax`` under a command line naming two GPUs does not."""

    out = tmp_path / "out"
    code = (
        "import sys, multiprocessing; sys.argv = sys.argv[:1] + sys.argv[1:]; import relax, relax.commands.refine; "
        "print('children', len(multiprocessing.active_children()))"
    )
    completed = _python(code, "refine", "--gpu", "0:1", "--output", str(out), tmp_path=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "relax: rank" not in completed.stderr
    assert not out.exists()


def test_the_console_script_package_imports_neither_relax_nor_jax(tmp_path):
    completed = _python("import sys, relax_cli; print(sorted(m for m in ('relax', 'jax') if m in sys.modules))", tmp_path=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip() == "[]"
