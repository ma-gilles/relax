"""A run extended past its VDAM schedule: Config.schedule_iterations holds the last phase for the iterations beyond it."""

import dataclasses

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.ppca_command_run import rows, run_command, write_star

from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config

N_IMAGES = 100_000
LENGTH = 300


def _config(iterations: int, **extra) -> Config:
    """Batches of 300 on every iteration, the way the cahra_01 runs go."""
    return Config(iterations=iterations, stochastic_batch_size=300, stochastic_all_iterations=True, **extra)


def test_the_iterations_of_the_schedule_are_those_of_a_run_that_long_and_the_ones_past_it_hold_its_last_phase():
    plain, extended, doubled = _config(LENGTH), _config(2 * LENGTH, schedule_iterations=LENGTH), _config(2 * LENGTH)
    for t in range(1, LENGTH + 1):
        assert extended.schedule(t, N_IMAGES) == plain.schedule(t, N_IMAGES)
        assert extended.step_factor(t, N_IMAGES) == plain.step_factor(t, N_IMAGES)
    for t in range(LENGTH + 1, 2 * LENGTH + 1):
        assert extended.schedule(t, N_IMAGES) == plain.schedule(LENGTH - 1, N_IMAGES)
        assert extended.step_factor(t, N_IMAGES) == plain.step_factor(LENGTH - 1, N_IMAGES)
    # The run's own last iteration is the schedule's full-data one; the held phase is not.
    assert extended.step_factor(LENGTH, N_IMAGES) < extended.step_factor(LENGTH + 1, N_IMAGES)
    # A run twice as long is at an earlier place of its schedule there, with a larger step factor.
    assert doubled.step_factor(LENGTH + 1, N_IMAGES) > extended.step_factor(LENGTH + 1, N_IMAGES)


def test_the_schedule_is_as_long_as_the_run_unless_a_length_is_given_and_it_fits_the_run():
    assert Config(iterations=20).schedule_length == 20
    assert Config(iterations=40, schedule_iterations=20).schedule_length == 20
    for length in (0, 1, 41):
        with pytest.raises(ValueError, match="schedule's iterations"):
            Config(iterations=40, schedule_iterations=length)


def test_a_checkpoint_of_the_shorter_run_is_the_extended_runs_model_and_a_longer_schedule_is_not():
    short = dataclasses.asdict(Config(iterations=20))
    extended = dataclasses.asdict(Config(iterations=40, schedule_iterations=20))
    longer = dataclasses.asdict(Config(iterations=40))
    assert checkpoint._identity_config(extended) == checkpoint._identity_config(short)
    assert checkpoint._identity_config(longer) != checkpoint._identity_config(short)
    assert checkpoint._identity_config({**short, "schedule_iterations": 20}) == checkpoint._identity_config(short)
    # A checkpoint written before the field existed has none.
    assert checkpoint._identity_config({k: v for k, v in short.items() if k != "schedule_iterations"}) == (
        checkpoint._identity_config(short)
    )


def test_a_finished_run_extended_with_resume_follows_the_run_that_was_extended_from_the_start(monkeypatch, tmp_path):
    """8 iterations, then 8 more from its last checkpoint, against 16 iterations on the 8-iteration schedule at once."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = write_star(tmp_path / "data", 16)
    longer = ("--iterations", "16", "--schedule-iterations", "8")
    run_command(star, tmp_path / "straight", *longer)
    run_command(star, tmp_path / "extended")
    run_command(star, tmp_path / "extended", *longer, "--resume", str(tmp_path / "extended" / "checkpoint_0008.npz"))
    straight, extended = (rows(tmp_path / name) for name in ("straight", "extended"))

    assert sorted(extended) == list(range(1, 17))
    for t in range(1, 17):
        assert_matches(extended[t]["gates"], straight[t]["gates"])
        assert_matches(extended[t]["disagreement_scale"], straight[t]["disagreement_scale"], rtol=1e-12)
    assert np.isfinite(np.asarray(straight[16]["gates"])).all()
