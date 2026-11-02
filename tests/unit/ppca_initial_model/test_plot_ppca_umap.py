"""The latents and true states that scripts/plot_ppca_umap.py pairs up, and its state colours."""

import numpy as np
import pytest

from scripts import plot_ppca_umap as umap_plot

pytestmark = pytest.mark.unit


def write_latents(plots, ids: np.ndarray, z: np.ndarray) -> None:
    """A latents_pc_ordered.npz holding ``z`` rows for the particle ``ids``."""
    np.savez(plots / "latents_pc_ordered.npz", particle_ids=ids, z=z)


def test_labels_follow_particle_ids_when_the_latents_are_shuffled(tmp_path):
    """Row i of the latents is particle ids[i]; its label is labels[ids[i]], returned in particle-ID order."""
    ids = np.array([3, 0, 4, 1, 2], np.int32)
    z = np.arange(10, dtype=np.float32).reshape(5, 2)
    labels = np.array([10, 11, 12, 13, 14])
    write_latents(tmp_path, ids, z)
    np.savez(tmp_path / "truth.npz", labels=labels)

    z_sorted, labels_sorted = umap_plot.load_latents(tmp_path, tmp_path / "truth.npz")

    np.testing.assert_array_equal(labels_sorted, labels)  # particle 0, 1, ... 4
    np.testing.assert_array_equal(z_sorted, z[np.argsort(ids)])


def test_no_labels_file_gives_no_labels(tmp_path):
    write_latents(tmp_path, np.arange(4, dtype=np.int32), np.zeros((4, 3), np.float32))

    z, labels = umap_plot.load_latents(tmp_path, None)

    assert z.shape == (4, 3)
    assert labels is None


def test_a_particle_listed_twice_is_rejected(tmp_path):
    write_latents(tmp_path, np.array([0, 1, 1, 3], np.int32), np.zeros((4, 2), np.float32))

    with pytest.raises(ValueError, match="exactly once"):
        umap_plot.load_latents(tmp_path, None)


def test_a_label_count_that_differs_from_the_particle_count_is_rejected(tmp_path):
    write_latents(tmp_path, np.arange(4, dtype=np.int32), np.zeros((4, 2), np.float32))
    np.savez(tmp_path / "truth.npz", labels=np.zeros(7, np.int64))

    with pytest.raises(ValueError, match="7 labels for 4 particles"):
        umap_plot.load_latents(tmp_path, tmp_path / "truth.npz")


@pytest.mark.parametrize("n_states", [3, 10, 11])
def test_every_state_gets_its_own_colour(n_states):
    colors = umap_plot.state_colors(np.repeat(np.arange(n_states), 2))

    assert sorted(colors) == list(range(n_states))
    assert len({tuple(color) for color in colors.values()}) == n_states
