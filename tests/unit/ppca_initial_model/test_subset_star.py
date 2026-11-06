"""scripts/ppca_schedule/subset_star.py: the training STAR restricted to the images of one true state."""

import numpy as np
import pytest

from scripts.lib.ppca_training_star import write_training_star
from scripts.ppca_schedule.subset_star import subset_star

pytestmark = pytest.mark.unit


def _star(root, count):
    root.mkdir(parents=True, exist_ok=True)
    (root / "particles.mrcs").write_bytes(b"stack")
    ctf = np.tile([15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1))
    ctf[:, 0] += 100 * np.arange(count)
    return write_training_star(root / "particles.star", "particles.mrcs", ctf, voxel_size=2, box=8)


def test_the_subset_keeps_the_optics_and_exactly_the_rows_of_the_state_under_their_own_names(tmp_path):
    star = _star(tmp_path / "all", 6)
    labels = np.array([0, 1, 1, 0, 2, 1])
    assert subset_star(star, labels, 1, tmp_path / "state1") == 3
    text = (tmp_path / "state1" / "particles.star").read_text().splitlines()
    assert "data_optics" in text and "data_particles" in text
    names = [line.split()[0] for line in text if "@particles.mrcs" in line]
    assert names == ["2@particles.mrcs", "3@particles.mrcs", "6@particles.mrcs"]
    assert (tmp_path / "state1" / "particles.mrcs").is_symlink()
    assert (tmp_path / "state1" / "particles.mrcs").read_bytes() == b"stack"
    defocus = [float(line.split()[2]) for line in text if "@particles.mrcs" in line]
    assert defocus == [15100.0, 15200.0, 15500.0]


def test_a_state_without_images_is_an_error(tmp_path):
    star = _star(tmp_path / "all", 3)
    with pytest.raises(ValueError, match="label 7"):
        subset_star(star, np.array([0, 1, 2]), 7, tmp_path / "none")
