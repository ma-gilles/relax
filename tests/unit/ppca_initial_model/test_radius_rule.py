"""The data_vs_prior Fourier radius (Config.radius_rule): RELION's current-resolution rule for the PPCA mean."""

import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.triangular import unpack_tri_to_full

from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.radius_rule import mean_data_vs_prior, next_radius, radius_for
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import empty_moments, metric_floor
from relax.reconstruction.resolution_criteria import RELION_MINRES_MAP

pytestmark = pytest.mark.unit

BOX = 8
N_FREQ = BOX * BOX * (BOX // 2 + 1)


def _write_star(root, count, box=BOX):
    """A RELION particle STAR of ``count`` noise images of ``box`` pixels with one optics group."""
    import mrcfile

    from scripts.lib.ppca_training_star import write_training_star

    root.mkdir(parents=True, exist_ok=True)
    images = np.random.default_rng(19).normal(size=(count, box, box)).astype(np.float32)
    with mrcfile.new(root / "particles.mrcs", overwrite=True) as stack:
        stack.set_data(images)
        stack.voxel_size = 2
    ctf = np.tile([15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1))
    ctf[:, 0] += 100 * np.arange(count)
    return write_training_star(root / "particles.star", "particles.mrcs", ctf, voxel_size=2, box=box)


def _state(radius=2, next_radius=None, iteration=0):
    theta = jnp.zeros((N_FREQ, 2), jnp.complex64)
    return State(
        theta,
        empty_moments(theta),
        jnp.ones(BOX // 2 + 1, jnp.float32),
        iteration,
        np.arange(4),
        {},
        2.0,
        radius,
        {},
        next_radius=next_radius,
    )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"radius_rule": "fixed"}, "stages or data_vs_prior"),
        ({"radius_rule": "data_vs_prior", "incr_size": 0}, "positive integer"),
        ({"radius_rule": "data_vs_prior", "incr_size": True}, "positive integer"),
    ],
)
def test_the_radius_rule_options_are_validated(options, message):
    with pytest.raises(ValueError, match=message):
        Config(**options)


def test_the_default_radius_comes_from_the_stages_table():
    config = Config(stages=((1, 4, 3), (5, 8, 3), (9, 40, 3)))
    state = _state(radius=99, next_radius=99)
    assert [radius_for(config, state, i, 64) for i in (1, 4, 5, 9)] == [4, 4, 8, 31]


def test_the_data_vs_prior_radius_starts_at_the_first_stage_and_follows_the_planned_radius():
    config = Config(stages=((1, 4, 3), (5, 8, 3)), radius_rule="data_vs_prior")
    assert radius_for(config, _state(), 1, 64) == 4
    assert radius_for(config, _state(radius=4, next_radius=15, iteration=1), 2, 64) == 15
    # The final embeddings are inferred at the iteration of the state, with the radius that iteration used.
    assert radius_for(config, _state(radius=15, next_radius=25, iteration=2), 2, 64) == 15


@pytest.mark.parametrize(
    "shell_before_drop, radius, incr_size, expected",
    [
        (12, 10, 10, 22),  # the resolution shell plus RELION's 10 shells
        (2, 4, 10, RELION_MINRES_MAP + 10),  # RELION never reports a resolution below --minres_map
        (12, 30, 10, 30),  # the radius never decreases
        (28, 20, 10, 31),  # and stays one shell below Nyquist
        (12, 10, 3, 15),
    ],
)
def test_the_next_radius_is_the_resolution_shell_plus_the_increment(shell_before_drop, radius, incr_size, expected):
    dvp = np.where(np.arange(33) <= shell_before_drop, 5.0, 0.5)
    assert next_radius(dvp, radius=radius, grid_size=64, incr_size=incr_size) == expected


def test_the_weight_conversion_is_the_metric_floors():
    """RELION's BPref weight is the metric over its floor: a floor of 1 in RELION units is 0.5 / N^4 here."""
    for n in (8, 64, 128):
        assert_matches(2 * n**4 * metric_floor(n), np.float32(1.0))


def test_the_packed_metric_holds_the_mean_weight_first():
    metric = np.random.default_rng(0).normal(size=(5, 6)).astype(np.float32)
    assert_matches(unpack_tri_to_full(jnp.asarray(metric), 3)[:, 0, 0], metric[:, 0])


def test_data_vs_prior_is_the_shell_average_weight_times_the_shell_average_power():
    """Shell s below the radius: fudge * mean(weight / floor) * mean(|mean|^2 / N^4) over its rows; zero beyond."""
    shape = (BOX,) * 3
    rng = np.random.default_rng(1)
    weight = rng.uniform(0.5, 2.0, N_FREQ).astype(np.float32) * metric_floor(BOX)
    mean = (rng.normal(size=N_FREQ) + 1j * rng.normal(size=N_FREQ)).astype(np.complex64)
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(shape), np.int32).reshape(-1)
    radii = np.asarray(ftu.get_grid_of_radial_distances_real(shape, rounded=False)).reshape(-1)
    radius, fudge = 3, 2.0

    result = mean_data_vs_prior(weight, mean, shape, fudge=fudge, radius=radius)

    expected = np.zeros(BOX // 2 + 1)
    for shell in range(radius + 1):
        rows = (shells == shell) & (radii < radius)
        expected[shell] = fudge * np.mean(weight[rows] / metric_floor(BOX)) * np.mean(np.abs(mean[rows]) ** 2 / BOX**4)
    assert result.shape == expected.shape
    assert_matches(result, expected, rtol=1e-5)  # float32 weight and mean
    assert np.all(result[: radius + 1] > 0) and np.all(result[radius + 1 :] == 0)
    # More data or a stronger prior only raises it: doubling the weight doubles every shell.
    assert_matches(mean_data_vs_prior(2 * weight, mean, shape, fudge=fudge, radius=radius), 2 * result, rtol=1e-5)


def test_the_planned_radius_is_saved_and_an_older_checkpoint_resumes_without_one(tmp_path):
    config = Config(q=1, radius_rule="data_vs_prior")
    identity = {"input": "x"}
    checkpoint.save(tmp_path / "a.npz", _state(radius=4, next_radius=7), config, identity)
    assert checkpoint.load(tmp_path / "a.npz", config, identity).next_radius == 7

    with np.load(tmp_path / "a.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    metadata = json.loads(str(arrays["metadata"]))
    del metadata["next_radius"]
    np.savez(tmp_path / "old.npz", **{**arrays, "metadata": json.dumps(metadata)})
    assert checkpoint.load(tmp_path / "old.npz", config, identity).next_radius is None


def test_an_older_checkpoint_of_the_stages_rule_keeps_its_identity():
    """The new settings at their defaults are no part of the identity of a checkpoint written without them."""
    identity = checkpoint._identity_config(dataclasses.asdict(Config(q=1)))
    assert "radius_rule" not in identity and "incr_size" not in identity
    changed = checkpoint._identity_config(dataclasses.asdict(Config(q=1, radius_rule="data_vs_prior")))
    assert changed["radius_rule"] == "data_vs_prior"


def _run_command(star, output, *extra):
    """``relax ppca_initial_model`` on ``star`` for 8 iterations of a fixed batch of 8 particles."""
    from relax.commands import ppca_initial_model

    argv = [
        "--i", str(star), "--particle-diameter", "12", "-o", str(output), "--q", "1", "--seed", "11",
        "--iterations", "8", "--stages", "[[1,2,1]]", "--stochastic-batch-size", "8", "--stochastic-all-iterations",
        "--image-batch-size", "4", "--rotation-block-size", "64", "--skip-final-embeddings", *extra,
    ]  # fmt: skip
    ppca_initial_model.main(argv)


def test_a_data_vs_prior_run_logs_its_radii_and_resumes_to_the_same_radii_and_model(monkeypatch, tmp_path):
    """The first iteration runs at the table's radius, later ones at the planned radius (the box's limit of 3 here,
    since RELION's minimum resolution plus the increment exceeds it); a run stopped at 4 and resumed ends at the
    radii and the model of the uninterrupted run."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    options = ("--radius-rule", "data_vs_prior", "--incr-size", "1")
    full = tmp_path / "full"
    _run_command(star, full, *options)
    log = [json.loads(line) for line in open(full / "iterations.jsonl")]
    assert [row["radius"] for row in log] == [2, 3, 3, 3, 3, 3, 3, 3]
    assert all(len(row["data_vs_prior"]) == BOX // 2 + 1 and row["next_radius"] == 3 for row in log)

    split = tmp_path / "split"
    _run_command(star, split, *options, "--stop-after", "4")
    _run_command(star, split, *options, "--resume", str(split / "checkpoint_0004.npz"))
    resumed = [json.loads(line) for line in open(split / "iterations.jsonl")]
    assert [row["radius"] for row in resumed] == [row["radius"] for row in log]
    with np.load(full / "checkpoint_0008.npz") as a, np.load(split / "checkpoint_0008.npz") as b:
        assert_matches(b["theta"], a["theta"])


def test_momentum_sgd_follows_the_data_vs_prior_radius_too(monkeypatch, tmp_path):
    """The momentum step keeps the packed metric under the rule (its mean entry is the rule's data) and the radii
    are the VDAM run's: the first at the table's radius, the rest at the planned one."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    options = ("--optimizer", "momentum_sgd", "--radius-rule", "data_vs_prior", "--incr-size", "1")
    _run_command(star, tmp_path / "run", *options)
    log = [json.loads(line) for line in open(tmp_path / "run" / "iterations.jsonl")]
    assert [row["radius"] for row in log] == [2, 3, 3, 3, 3, 3, 3, 3]
    assert all(len(row["data_vs_prior"]) == BOX // 2 + 1 and row["next_radius"] == 3 for row in log)


def test_the_stages_rule_logs_no_data_vs_prior(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    _run_command(star, tmp_path / "run")
    log = [json.loads(line) for line in open(tmp_path / "run" / "iterations.jsonl")]
    assert all("data_vs_prior" not in row and "next_radius" not in row for row in log)
    assert [row["radius"] for row in log] == [2] * 8
