"""The opt-in translation search: a per-iteration shift grid (shift_stages) and a held offset prior sigma."""

import argparse
import dataclasses
from types import SimpleNamespace

import pytest
from helpers.ppca_command_run import rows, run_command, write_star

from relax.commands import ppca_initial_model
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.iteration_loop import _prior_offset_variance
from relax.sampling import grids

pytestmark = pytest.mark.unit

STAGES = ((1, 4, 2), (5, 6, 1))


def test_the_shift_grid_is_the_last_stage_started_and_the_fixed_range_and_step_without_stages():
    config = Config(shift_stages=STAGES)
    assert [config.shift_grid(iteration) for iteration in (1, 4, 5, 3000)] == [(4, 2), (4, 2), (6, 1), (6, 1)]
    assert Config(shift_range=3, shift_step=1).shift_grid(7) == (3, 1)


@pytest.mark.parametrize(
    "stages, message",
    [
        (((2, 4, 2),), "the first at iteration 1"),
        (((1, 4),), "the first at iteration 1"),
        (((1, 4, 2), (1, 6, 1)), "increasing first iterations"),
        (((1, 4, 2), (3, 6, 0)), "positive step"),
        (((1, -1, 2),), "nonnegative range"),
    ],
)
def test_invalid_shift_stages_are_rejected(stages, message):
    with pytest.raises(ValueError, match=message):
        Config(shift_stages=stages)


def test_the_held_sigma_is_in_pixels_per_axis_and_replaces_the_learned_variance_only_when_set():
    state, dataset = SimpleNamespace(offset_variance=7.0), SimpleNamespace(voxel_size=2.0)
    assert _prior_offset_variance(dataset, state, Config()) == 7.0
    assert _prior_offset_variance(dataset, state, Config(offset_sigma_angstrom=20.0)) == 100.0
    for sigma in (0.0, -3.0):
        with pytest.raises(ValueError, match="offset sigma must be positive"):
            Config(offset_sigma_angstrom=sigma)


def test_the_options_are_parsed_and_stages_exclude_an_explicit_range_or_step(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    ppca_initial_model.add_args(parser)
    required = ["--i", "particles.star", "--particle-diameter", "100", "-o", str(tmp_path)]
    args = parser.parse_args([*required, "--shift-stages", "[[1,20,5],[1626,21,3]]", "--offset-sigma-angstrom", "20"])
    assert args.shift_stages == "[[1,20,5],[1626,21,3]]" and args.offset_sigma_angstrom == 20.0
    assert parser.parse_args(required).shift_stages is None
    assert parser.parse_args(required).offset_sigma_angstrom is None
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    with pytest.raises(ValueError, match="replaces --shift-range and --shift-step"):
        ppca_initial_model.main([*required, "--shift-stages", "[[1,4,2]]", "--shift-step", "1"])


def test_the_defaults_are_no_part_of_a_checkpoints_identity_and_the_options_are():
    identity = checkpoint._identity_config
    default = identity(dataclasses.asdict(Config()))
    assert "shift_stages" not in default and "offset_sigma_angstrom" not in default
    staged = identity(dataclasses.asdict(Config(shift_stages=STAGES, offset_sigma_angstrom=20.0)))
    assert staged["shift_stages"] == [list(stage) for stage in STAGES] and staged["offset_sigma_angstrom"] == 20.0


def test_a_run_searches_each_iterations_grid_and_logs_it(monkeypatch, tmp_path):
    """Iterations 1 to 4 search the 5-point grid (range 2, step 2), 5 to 8 the 13-point one (range 4, step 2)."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = write_star(tmp_path / "data", 16)
    run_command(star, tmp_path / "run", "--shift-stages", "[[1,2,2],[5,4,2]]", "--offset-sigma-angstrom", "6")
    logged = rows(tmp_path / "run")
    searched = [(logged[t]["shift_range_px"], logged[t]["shift_step_px"]) for t in range(1, 9)]
    assert searched == [(2, 2)] * 4 + [(4, 2)] * 4
    assert [len(grids.get_relion_translation_grid(max_pixel=r, pixel_offset=s)) for r, s in ((2, 2), (4, 2))] == [5, 13]
    # The learned variance is still updated and logged while the held one is used.
    assert all(logged[t]["offset_variance_px2"] > 0 for t in range(1, 9))


def test_the_held_sigma_reaches_the_expectation(monkeypatch, tmp_path):
    """Iteration 1 sees no update, so two runs that differ only in the held sigma differ only through the prior."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = write_star(tmp_path / "data", 16)
    for name, sigma in (("narrow", "2"), ("wide", "200")):
        run_command(star, tmp_path / name, "--shift-stages", "[[1,4,2]]", "--offset-sigma-angstrom", sigma)
    narrow, wide = rows(tmp_path / "narrow")[1], rows(tmp_path / "wide")[1]
    assert narrow["log_likelihood"] != pytest.approx(wide["log_likelihood"], rel=1e-4)
