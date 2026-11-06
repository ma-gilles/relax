"""The direction prior of the global search (Config.direction_prior): uniform, or RELION's learned pdf_direction."""

import argparse
import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.commands import ppca_initial_model
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import empty_moments

pytestmark = pytest.mark.unit

BOX = 8


def _write_star(root, count):
    """A RELION particle STAR of ``count`` noise images of ``BOX`` pixels with one optics group."""
    import mrcfile

    from scripts.lib.ppca_training_star import write_training_star

    root.mkdir(parents=True, exist_ok=True)
    images = np.random.default_rng(19).normal(size=(count, BOX, BOX)).astype(np.float32)
    with mrcfile.new(root / "particles.mrcs", overwrite=True) as stack:
        stack.set_data(images)
        stack.voxel_size = 2
    ctf = np.tile([15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1))
    ctf[:, 0] += 100 * np.arange(count)
    return write_training_star(root / "particles.star", "particles.mrcs", ctf, voxel_size=2, box=BOX)


def _run_command(star, output, *extra):
    """``relax ppca_initial_model`` on ``star`` for 8 iterations of a fixed batch of 8 particles."""
    argv = [
        "--i", str(star), "--particle-diameter", "12", "-o", str(output), "--q", "1", "--seed", "11",
        "--iterations", "8", "--stages", "[[1,2,1]]", "--stochastic-batch-size", "8", "--stochastic-all-iterations",
        "--image-batch-size", "4", "--rotation-block-size", "64", "--skip-final-embeddings", *extra,
    ]  # fmt: skip
    ppca_initial_model.main(argv)


def _state():
    n_freq = BOX * BOX * (BOX // 2 + 1)
    theta = jnp.zeros((n_freq, 2), jnp.complex64)
    return State(theta, empty_moments(theta), jnp.ones(BOX // 2 + 1, jnp.float32), 0, np.arange(4), {}, 2.0, 2, {})


def test_the_prior_is_uniform_by_default_and_the_option_takes_only_learned_or_uniform():
    parser = argparse.ArgumentParser()
    ppca_initial_model.add_args(parser)
    required = ["--i", "particles.star", "--particle-diameter", "100", "-o", "out"]
    assert Config().direction_prior == "uniform"
    assert parser.parse_args(required).direction_prior == "uniform"
    assert parser.parse_args([*required, "--direction-prior", "learned"]).direction_prior == "learned"
    with pytest.raises(ValueError, match="direction prior must be learned or uniform"):
        Config(direction_prior="flat")


def test_the_learned_prior_is_no_part_of_a_checkpoints_identity_and_a_resume_cannot_change_the_prior(tmp_path):
    learned = Config(direction_prior="learned")
    assert "direction_prior" not in checkpoint._identity_config(dataclasses.asdict(learned))
    assert checkpoint._identity_config(dataclasses.asdict(Config()))["direction_prior"] == "uniform"
    checkpoint.save(tmp_path / "learned.npz", _state(), learned, {"input": "x"})
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load(tmp_path / "learned.npz", Config(), {"input": "x"})


def test_a_default_run_keeps_no_prior_and_starts_as_the_learned_run_does(monkeypatch, tmp_path):
    """Both runs score iteration 1 with the uniform prior; after it only the learned run has one, with its mass off uniform."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    _run_command(star, tmp_path / "learned", "--direction-prior", "learned")
    _run_command(star, tmp_path / "uniform")
    logs = {name: [json.loads(line) for line in open(tmp_path / name / "iterations.jsonl")] for name in ("learned", "uniform")}
    assert_matches(logs["uniform"][0]["log_likelihood"], logs["learned"][0]["log_likelihood"])
    assert all(row["direction_prior"] is None for row in logs["uniform"])
    with np.load(tmp_path / "learned" / "checkpoint_0008.npz") as learned, np.load(tmp_path / "uniform" / "checkpoint_0008.npz") as uniform:
        assert uniform["direction_prior"].size == 0
        prior = learned["direction_prior"]
        assert prior.size > 0 and prior.max() > prior.min()
        assert np.isfinite(uniform["theta"]).all()
