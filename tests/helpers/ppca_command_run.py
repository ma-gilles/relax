"""A tiny ``relax ppca_initial_model`` run on noise images, for tests of options that reach the iteration loop."""

import json
from pathlib import Path

import numpy as np

from relax.commands import ppca_initial_model

BOX = 8


def write_star(root: Path, count: int) -> Path:
    """A RELION particle STAR of ``count`` noise images of ``BOX`` pixels with one optics group; returns its path."""
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


def run_command(star: Path, output: Path, *extra: str) -> None:
    """``relax ppca_initial_model`` on ``star`` for 8 iterations of a fixed batch of 8 particles, plus ``extra`` flags."""
    argv = [
        "--i", str(star), "--particle-diameter", "12", "-o", str(output), "--q", "1", "--seed", "11",
        "--iterations", "8", "--stages", "[[1,2,1]]", "--stochastic-batch-size", "8", "--stochastic-all-iterations",
        "--image-batch-size", "4", "--rotation-block-size", "64", "--skip-final-embeddings", *extra,
    ]  # fmt: skip
    ppca_initial_model.main(argv)


def rows(output: Path) -> dict[int, dict]:
    """The rows of a run's iterations.jsonl by iteration number."""
    return {row["iteration"]: row for row in map(json.loads, open(output / "iterations.jsonl"))}
