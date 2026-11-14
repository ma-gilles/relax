"""A multi-GPU PPCA run (``relax ppca_initial_model --gpu 0:1``): tiles scored on several ranks, joined on rank 0.

The channel helpers and the tile assignment are tested in this process. The ranks are tested as they run: the
launcher starts real processes on the CPU, and their results are compared with one process's. Only the order of
each update's volume sums changes with the number of ranks (section 18 of docs/math/vdam_ppca_algorithm.md): the
selected particles, the counts and the tile order are exact, the floats agree to float32 reduction noise.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from multiprocessing import Pipe
from pathlib import Path

import numpy as np
import pytest
from conftest import repo_python_command, repo_subprocess_env
from helpers.float_compare import assert_matches

from relax.parallel import channel
from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model.ranks import assign_tiles
from relax_cli import launch

pytestmark = pytest.mark.unit

TESTS = Path(__file__).resolve().parents[2]


def test_broadcast_sends_one_record_to_every_rank_and_gather_returns_them_in_rank_order():
    """Rank 3 answers first and rank 1 last: the gathered records are still in rank order."""
    pairs = {rank: Pipe() for rank in (1, 2, 3)}
    ours = {rank: pair[0] for rank, pair in pairs.items()}
    channel.broadcast(ours, {"task": np.arange(3.0)}, {})
    gather = channel.Gather(ours)

    def answer(rank, delay):
        theirs = pairs[rank][1]
        task = channel.receive(theirs, {}, peer="rank 0")
        time.sleep(delay)
        channel.send(theirs, {"rank": rank, "value": task["task"] * rank}, {})

    threads = [threading.Thread(target=answer, args=(rank, 0.3 / rank)) for rank in (1, 2, 3)]
    for thread in threads:
        thread.start()
    records = gather.records({})
    for thread in threads:
        thread.join()
    assert list(records) == [1, 2, 3]
    assert [record["rank"] for record in records.values()] == [1, 2, 3]
    for rank, record in records.items():
        np.testing.assert_array_equal(record["value"], np.arange(3.0) * rank)


def test_gather_raises_when_a_rank_closed():
    ours, theirs = Pipe()
    gather = channel.Gather({1: ours})
    theirs.close()
    with pytest.raises(channel.RankFailed, match="rank 1"):
        gather.records({})


def test_ordered_sum_adds_left_to_right_and_skips_ranks_without_a_value():
    values = [np.float32(1e8), None, np.float32(1.0), np.float32(-1e8)]
    # (1e8 + 1) rounds to 1e8 in float32, so the left-to-right sum is 0, not the 1 another order gives.
    assert channel.ordered_sum(values) == np.float32(0.0)
    assert channel.ordered_sum([None, None]) is None


def test_tile_assignment_is_a_balanced_function_of_the_costs():
    costs = [5, 9, 1, 9, 3, 3, 7, 2]
    owners = assign_tiles(costs, 3)
    # Heaviest first to the least loaded rank, the lower tile and the lower rank first on a tie.
    np.testing.assert_array_equal(owners, [2, 0, 1, 1, 0, 1, 2, 0])
    np.testing.assert_array_equal(assign_tiles(costs, 3), owners)
    loads = np.bincount(owners, weights=costs, minlength=3)
    assert loads.max() - loads.min() <= max(costs)
    np.testing.assert_array_equal(assign_tiles(costs, 1), np.zeros(len(costs)))
    # Fewer tiles than ranks: each tile has its own rank, the last ranks have none.
    np.testing.assert_array_equal(assign_tiles([1, 4], 4), [1, 0])


@dataclass(frozen=True)
class _Part:
    lhs_tri: object
    residual_gradient: object
    n_images: int


def test_joined_tile_sums_add_each_ranks_tile_order_sum_in_rank_order():
    """A group's joined sums are each rank's running sum over its tiles (in tile order), added in rank order;
    the stripped tiles come back in the group's tile order whichever rank scored them."""
    rng = np.random.default_rng(0)
    n_tiles = 9
    parts = [
        _Part(rng.standard_normal((5, 6)).astype(np.float32), rng.standard_normal((5, 3)).astype(np.float32), k)
        for k in range(n_tiles)
    ]
    tiles = [(k % 2, 0, np.arange(k + 1)) for k in range(n_tiles)]  # (group, noise group, ids)
    owners = assign_tiles([len(t[2]) for t in tiles], 3)
    shares = []
    for rank in range(3):
        mine = {k for k in range(n_tiles) if owners[k] == rank}
        sums = [iteration_loop.TileSums(k for k in range(n_tiles) if tiles[k][0] == g and k in mine) for g in range(2)]
        for k in sorted(mine, key=lambda k: -k):  # completion order: largest first
            sums[tiles[k][0]].add(k, parts[k])
        stripped = {k: part for s in sums for k, part in zip(s.order, s.parts())}
        shares.append(SimpleShare(stripped, [s.totals for s in sums]))
    _, joined = iteration_loop._join_tile_shares(shares, tiles, 2, embeddings_only=False)
    for group in range(2):
        order = [k for k in range(n_tiles) if tiles[k][0] == group]
        assert [part.n_images for part in joined[group].parts()] == order
        for name in ("lhs_tri", "residual_gradient"):
            expected = None
            for rank in range(3):
                rank_sum = None
                for k in order:
                    if owners[k] == rank:
                        rank_sum = getattr(parts[k], name) if rank_sum is None else rank_sum + getattr(parts[k], name)
                if rank_sum is not None:
                    expected = rank_sum if expected is None else expected + rank_sum
            np.testing.assert_array_equal(joined[group].totals[name], expected)  # the same additions in order
            assert_matches(joined[group].totals[name], sum(getattr(parts[k], name) for k in order))
    with pytest.raises(ValueError, match="exactly once"):
        iteration_loop._join_tile_shares(shares[:2], tiles, 2, embeddings_only=False)


@dataclass(frozen=True)
class SimpleShare:
    parts: dict
    totals: list


# The ranks' processes: a subtomogram run of the command (``controller``) or single-particle E-steps (``estep``).
_RANKS = r"""
import os, sys
mode, tests, out, world = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
extra = sys.argv[5:]
sys.path[:0] = [tests, os.path.join(tests, "unit", "ppca_initial_model")]
from relax_cli import launch

gpu = ["--gpu", ":".join(str(rank) for rank in range(world))] if world > 1 else []
if world > 1 and launch.rank_environment() is None:
    import relax  # the launcher of this CPU test imports this checkout too (repo_python_command checks it)

    sys.exit(launch.launch_ranks(["ppca_initial_model", *gpu, "-o", out], sys.orig_argv, dict(os.environ)))
os.environ["SLURM_JOB_ID"] = "1"
import numpy as np

if mode == "controller":
    import test_tomo_ppca
    from relax.commands import ppca_initial_model as command

    command.load_tilt_training = lambda ios, output, *args, **options: (
        test_tomo_ppca._controller_particles(), {"test": True}, None
    )
    command.source_identity = lambda: {"test": True}
    command.main(
        ["--ios", "optimisation_set.star", "--particle-diameter", "120", "-o", out, "--q", "2", "--seed", "12",
         "--iterations", "3", "--stages", "[[1, 3, 0], [3, 4, 1]]", "--shift-range", "1", "--shift-step", "1",
         "--image-batch-size", "1", "--rotation-block-size", "32", "--ppca-gemm-precision", "fp32", *gpu, *extra]
    )
    sys.exit(0)

import jax.numpy as jnp
from test_full_fine_streaming import N_HALF, _half_volume, _TinyData
from relax.parallel import channel
from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model import ranks as tile_ranks
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.state import State

rng = np.random.default_rng(7)
images = (rng.standard_normal((11, N_HALF)) + 1j * rng.standard_normal((11, N_HALF))).astype(np.complex64)
images[:, 0] += 30.0
data = _TinyData(images)
theta = np.stack([_half_volume(rng), _half_volume(rng, 0.3), _half_volume(rng, 0.2)], axis=1)
state = State(jnp.asarray(theta), None, jnp.full(5, 40.0, jnp.float32), 1, np.arange(11), {}, 2.0, 0, {})
config = {
    "dense": Config(q=2, stages=((1, 3, 0),), shift_range=1, shift_step=1, image_batch_size=2,
                    rotation_block_size=32, gemm_precision="fp32"),
    "oversampled": Config(q=2, stages=((1, 3, 0),), oversampling=1, target_mass=0.9, shift_range=1, shift_step=1,
                          image_batch_size=2, rotation_block_size=32, gemm_precision="fp32"),
}[extra[0]]
groups = [np.asarray([0, 2, 4, 6, 8, 10]), np.asarray([1, 3, 5, 7, 9])]
ranks_env = launch.rank_environment()
if not ranks_env.is_leader:
    tile_ranks.serve_tile_ranks(
        channel.connect(ranks_env.address, ranks_env.authkey, ranks_env.rank), ranks_env.rank, world,
        dataset=data, config=config, fingerprint=tile_ranks.run_fingerprint(data, config, {}),
    )
    sys.exit(0)
listener = channel.listen(ranks_env.address, ranks_env.authkey)
ranks = tile_ranks.accept_tile_ranks(
    listener, world, tile_ranks.run_objects(data, config), tile_ranks.run_fingerprint(data, config, {})
)
saved = {}
name = extra[0]
if True:
    for label, many in (("one", None), ("many", ranks)):
        stats = iteration_loop.expectation_groups(data, state, config, groups, 1, diameter_ang=6.0, ranks=many)
        for g, s in enumerate(stats):
            for field in ("lhs_tri", "residual_gradient", "residual_num", "residual_den", "embeddings",
                          "original_image_ids", "n_images", "log_likelihood"):
                saved[f"{name}/{label}/{g}/{field}"] = np.asarray(getattr(s, field))
            saved[f"{name}/{label}/{g}/rotation_mass"] = np.asarray(s.diagnostics["rotation_mass"])
        final = iteration_loop.expectation(
            data, state, config, np.arange(11), 1, embeddings_only=True, diameter_ang=6.0, ranks=many
        )
        for field in ("embeddings", "original_image_ids", "log_evidence", "max_posterior"):
            saved[f"{name}/{label}/final/{field}"] = np.asarray(getattr(final, field))
ranks.close()
np.savez(os.path.join(out, f"estep_{name}.npz"), **saved)
"""


def _run_ranks(tmp_path, mode, out, world, *extra):
    script = tmp_path / "ranks.py"
    script.write_text(_RANKS)
    env = repo_subprocess_env()
    env.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"})
    env.pop("RELAX_RANK", None)
    Path(out).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        repo_python_command(str(script), mode, str(TESTS), str(out), str(world), *extra),
        env=env, capture_output=True, text=True, timeout=900, cwd=tmp_path,
    )  # fmt: skip
    logs = "".join(path.read_text() for path in Path(out).glob(".rank*/rank*.log"))
    assert completed.returncode == 0, completed.stderr[-6000:] + logs[-6000:]
    return completed


@pytest.mark.parametrize("name", ["dense", "oversampled"])
def test_single_particle_e_steps_on_two_ranks_match_one_rank(tmp_path, name):
    """A dense or oversampled E-step and the final embedding pass of single particles, on rank 0 alone and split
    between two ranks: the particle order and counts are exact, every statistic agrees to float32 noise."""
    out = tmp_path / "estep"
    _run_ranks(tmp_path, "estep", out, 2, name)
    saved = np.load(out / f"estep_{name}.npz")
    keys = sorted({key.split("/", 2)[2] for key in saved.files if key.split("/")[1] == "one"})
    assert len(keys) == 2 * 9 + 4 and len(saved.files) == 2 * len(keys)
    for key in keys:
        ours, theirs = saved[f"{name}/one/{key}"], saved[f"{name}/many/{key}"]
        assert np.all(np.isfinite(ours))
        if key.endswith(("original_image_ids", "n_images")):
            np.testing.assert_array_equal(theirs, ours)
        else:
            assert_matches(theirs, ours)
    # Both halves and the final pass covered every particle once.
    np.testing.assert_array_equal(np.sort(saved[f"{name}/many/final/original_image_ids"]), np.arange(11) + 100)


def _checkpoints(output):
    out = {}
    for path in sorted(Path(output).glob("checkpoint_*.npz")):
        with np.load(path) as arrays:
            out[path.name] = {name: arrays[name] for name in arrays.files if name != "metadata"}
            out[path.name]["metadata"] = json.loads(str(arrays["metadata"]))
    return out


@pytest.mark.parametrize(
    "options, worlds",
    [([], (2, 3)), (["--optimizer", "momentum_sgd"], (2,))],
    ids=["vdam", "momentum-sgd"],
)
def test_subtomogram_command_on_several_ranks_matches_one_rank(tmp_path, options, worlds):
    """``relax ppca_initial_model --ios`` on one, two and three ranks: three updates (two stages) and the final
    embedding pass. One particle per tile, so each rank holds several tiles of a half, interleaved with the other
    ranks': the volume sums are associated differently. Each update selects the same particles; theta, the
    moments or velocity, the noise and the priors after every update, and the final embeddings, agree to float32
    noise; rank 0 alone writes the outputs."""
    reference = tmp_path / "one"
    _run_ranks(tmp_path, "controller", reference, 1, *options)
    ours = _checkpoints(reference)
    ours_rows = [json.loads(line) for line in (reference / "iterations.jsonl").read_text().splitlines()]
    assert sorted(ours) == [f"checkpoint_000{k}.npz" for k in range(4)]
    for world in worlds:
        output = tmp_path / f"world{world}"
        _run_ranks(tmp_path, "controller", output, world, *options)
        theirs = _checkpoints(output)
        assert sorted(theirs) == sorted(ours)
        for name in ours:
            assert sorted(theirs[name]) == sorted(ours[name])
            assert theirs[name]["metadata"]["rng_state"] == ours[name]["metadata"]["rng_state"]
            np.testing.assert_array_equal(theirs[name]["order"], ours[name]["order"])
            for field, value in ours[name].items():
                if field in ("metadata", "order"):
                    continue
                assert_matches(theirs[name][field], value, rtol=SECOND_MOMENT_RTOL if field == "second" else None)
            # A blend of the float32 posterior offset sums, kept as a Python float: compared at float32.
            assert_matches(
                np.float32(theirs[name]["metadata"]["offset_variance"]),
                np.float32(ours[name]["metadata"]["offset_variance"]),
            )
        rows = [json.loads(line) for line in (output / "iterations.jsonl").read_text().splitlines()]
        assert [row["iteration"] for row in rows] == [1, 2, 3]
        for row, our_row in zip(rows, ours_rows):
            assert row["particle_ids"] == our_row["particle_ids"] and row["half_counts"] == our_row["half_counts"]
            assert "ranks" not in our_row
            assert row["ranks"]["world"] == world and sum(row["ranks"]["tiles"]) >= world
            assert all(tiles > 0 for tiles in row["ranks"]["tiles"])
        embeddings, our_embeddings = np.load(output / "embeddings.npz"), np.load(reference / "embeddings.npz")
        np.testing.assert_array_equal(embeddings["particle_ids"], our_embeddings["particle_ids"])
        assert_matches(embeddings["z"], our_embeddings["z"])
        # Rank 0 writes the run's files; another rank only its log (and the particle STAR its loader writes).
        for rank in range(1, world):
            hidden = output / f".rank{rank}"
            assert (hidden / f"rank{rank}.log").is_file()
            assert not list(hidden.glob("checkpoint_*")) and not (hidden / "run.json").exists()


# VDAM's second moment averages the squared per-frequency direction, a metric solve that amplifies the volume sums'
# association (one rank's tile-order sum against two ranks' sums added in rank order) at its ill-conditioned
# frequencies: 9.7e-6 relative in 2 to 6 of its 4608 entries over the three updates on two ranks (CPU, job 7473156),
# where theta moved by 8.8e-8, the first moment by 2.6e-7 and the embeddings by 5.1e-7 (default rtol 1e-6);
# test_tomo_ppca's tile-size invariance takes 1e-5 for a comparable reordering of the statistics.
SECOND_MOMENT_RTOL = 2e-5


def test_one_device_and_a_skipped_launcher_refuse_or_run_alone():
    """``--gpu`` with one device runs this process; a process that lists two devices without being a rank (the
    launcher was skipped) is refused, and so is an unsupported route."""
    from relax.ppca_initial_model import ranks
    from relax.ppca_initial_model.config import Config

    ranks.require_gpu_ranks(None, None)
    ranks.require_gpu_ranks("0", None)
    with pytest.raises(RuntimeError, match="relax ppca_initial_model"):
        ranks.require_gpu_ranks("0:1", None)
    two = launch.RankEnvironment(rank=0, world=2, address="\0x", authkey=b"")
    ranks.require_gpu_ranks("0:1", two)
    ranks.require_tile_route(Config(), 2)
    with pytest.raises(NotImplementedError, match="one GPU"):
        ranks.require_tile_route(Config(stream_coarse_recompute=False), 2)
    with pytest.raises(NotImplementedError, match="two to four"):
        ranks.require_tile_route(Config(), 5)
    assert launch.output_argument(["ppca_initial_model", "-o", "run"]) == "run"
