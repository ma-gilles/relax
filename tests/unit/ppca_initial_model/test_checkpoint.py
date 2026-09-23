"""Checkpoint contracts reject altered inputs, policy and source."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from relax.ppca_initial_model.checkpoint import load, save
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import Moments

pytestmark = pytest.mark.unit


def test_checkpoint_preserves_state_and_rejects_mismatch(tmp_path):
    theta = jnp.ones((5, 3), jnp.complex64)
    rng = np.random.default_rng(11)
    moments = Moments(
        jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3).astype(jnp.complex64) * (1 + 1j),
        jnp.arange(10, dtype=jnp.float32).reshape(5, 2) + 0.5,
        jnp.asarray([[True, False, True, False, True], [False, True, False, True, False]]),
    )
    state = State(
        theta,
        moments,
        jnp.ones(3, jnp.float32),
        7,
        np.arange(8),
        rng.bit_generator.state,
        2.0,
        4,
        {"seed": 11},
        np.ones(12, np.float32) / 12,
        0,
    )
    config = Config()
    path = tmp_path / "checkpoint.npz"
    identity = {"input": "abc", "source": "def"}
    save(path, state, config, identity)
    restored = load(path, config, identity)
    np.testing.assert_array_equal(restored.theta, theta)
    np.testing.assert_array_equal(restored.moments.first, state.moments.first)
    np.testing.assert_array_equal(restored.moments.second, state.moments.second)
    np.testing.assert_array_equal(restored.moments.initialized, state.moments.initialized)
    np.testing.assert_array_equal(restored.noise, state.noise)
    np.testing.assert_array_equal(restored.order, state.order)
    np.testing.assert_array_equal(restored.direction_prior, state.direction_prior)
    assert restored.rng_state == state.rng_state
    assert restored.iteration == 7
    assert restored.offset_variance == state.offset_variance
    assert restored.radius == state.radius
    assert restored.initialization == state.initialization
    assert restored.direction_order == state.direction_order
    for bad in [{"input": "changed", "source": "def"}, {"input": "abc", "source": "changed"}]:
        with pytest.raises(ValueError, match="identity mismatch"):
            load(path, config, bad)
    with pytest.raises(ValueError, match="identity mismatch"):
        load(path, dataclasses.replace(config, seed=12), identity)
    assert not path.with_suffix(".tmp").exists()


def test_pilot_schedule_and_final_all_data():
    config = Config()
    assert config.schedule(1, 20000)[0] == 200
    assert config.schedule(61, 20000)[0] == 218
    assert config.schedule(160, 20000)[0] == 2000
    assert config.schedule(200, 20000)[0] == 20000
    assert config.stage(60) == (4, 1)
    assert config.stage(61) == (8, 2)
    assert config.stage(161) == (32, 3)


def test_small_pilot_fixed_stochastic_batch_and_resume_identity(tmp_path):
    config = Config(iterations=60, stages=((1, 4, 1), (21, 8, 2)), stochastic_batch_size=200)
    assert config.schedule(1, 2000)[0] == 200
    assert config.schedule(59, 2000)[0] == 200
    assert config.schedule(60, 2000)[0] == 2000
    assert config.stage(20) == (4, 1)
    assert config.stage(21) == (8, 2)
    with pytest.raises(ValueError, match="Stochastic batch size"):
        Config(stochastic_batch_size=0)
    theta = jnp.zeros((5, 3), jnp.complex64)
    moments = Moments(jnp.zeros((2, 5, 3), jnp.complex64), jnp.zeros((5, 2), jnp.float32), jnp.zeros((2, 5), bool))
    state = State(theta, moments, jnp.ones(3, jnp.float32), 20, np.arange(8), np.random.default_rng(11).bit_generator.state, 1.0, 4, {"seed": 11})
    path = tmp_path / "small.npz"
    save(path, state, config, {"fixture": "fixed"})
    assert load(path, config, {"fixture": "fixed"}).iteration == 20
    with pytest.raises(ValueError, match="identity mismatch"):
        load(path, dataclasses.replace(config, stochastic_batch_size=None), {"fixture": "fixed"})
def test_rejected_resume_preserves_existing_run_metadata(tmp_path, monkeypatch):
    """A source/config mismatch must not damage the prior run receipt."""
    from relax.commands import ppca_initial_model as command
    from relax.ppca_initial_model import checkpoint, iteration_loop

    output = tmp_path / "run"
    output.mkdir()
    receipt = output / "run.json"
    receipt.write_bytes(b'{"original":true}\n')
    resume = output / "checkpoint_0110.npz"
    resume.write_bytes(b"mismatched checkpoint")
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setattr(command, "load_training", lambda _path: (object(), {"particle_diameter_ang": 1}, {}))
    monkeypatch.setattr(command, "source_identity", lambda: {"test": True})
    monkeypatch.setattr(checkpoint, "load", lambda *_args: (_ for _ in ()).throw(ValueError("identity mismatch")))
    monkeypatch.setattr(iteration_loop, "run", lambda *_args, **_kwargs: pytest.fail("run called"))
    with pytest.raises(ValueError, match="identity mismatch"):
        command.main(["manifest.json", "-o", str(output), "--resume", str(resume)])
    assert receipt.read_bytes() == b'{"original":true}\n'
