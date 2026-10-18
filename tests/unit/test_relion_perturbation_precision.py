from pathlib import Path

import pytest

from relax.parity.relion_replay import resolve_replay_random_perturbation
from relax.star.metadata import read_relion_optimiser_metadata


def _write_optimizer(path: Path, seed: int) -> None:
    path.write_text(f"data_\n\n_rlnRandomSeed {seed}\n")


def test_optimizer_metadata_reads_random_seed(tmp_path):
    path = tmp_path / "run_it002_optimiser.star"
    _write_optimizer(path, 20260712)
    assert read_relion_optimiser_metadata(path)["random_seed"] == 20260712


def test_seed_exact_replay_recovers_unrounded_relion_value(tmp_path):
    _write_optimizer(tmp_path / "run_it002_optimiser.star", 20260712)
    value, source = resolve_replay_random_perturbation(
        star_value=0.405200,
        perturbation_factor=0.5,
        relion_iteration=2,
        replay_dir=str(tmp_path),
        explicit_seed=None,
    )
    assert source == "seed-exact"
    assert value == 0.4052000939846039
    assert value != 0.405200


def test_auto_replay_falls_back_to_star_without_seed(tmp_path):
    value, source = resolve_replay_random_perturbation(
        star_value=-0.04961,
        perturbation_factor=0.5,
        relion_iteration=1,
        replay_dir=str(tmp_path),
        explicit_seed=None,
    )
    assert source == "star-fallback"
    assert value == -0.04961


def test_seed_reconstruction_checks_star_consistency(tmp_path):
    _write_optimizer(tmp_path / "run_it002_optimiser.star", 20260712)
    with pytest.raises(ValueError, match="disagrees with replay STAR"):
        resolve_replay_random_perturbation(
            star_value=0.4,
            perturbation_factor=0.5,
            relion_iteration=2,
            replay_dir=str(tmp_path),
            explicit_seed=None,
            )
