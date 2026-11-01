"""The adaptive-subset inputs: the ID join, the distances and the nested thresholds."""

import numpy as np
import pytest

from relax.multibody.subsets import _join_order, distances_to, nested_thresholds, read_flex_projections


def test_flex_projections_join_by_image_name_not_row(tmp_path):
    path = tmp_path / "scores.txt"
    path.write_text("b@s.mrcs 2.0 20.0\na@s.mrcs 1.0 10.0\n")
    scores = read_flex_projections(path, np.array(["a@s.mrcs", "b@s.mrcs"]))
    np.testing.assert_array_equal(scores, [[1.0, 10.0], [2.0, 20.0]])


def test_join_refuses_missing_or_repeated_names():
    with pytest.raises(ValueError):
        _join_order(np.array(["a"]), np.array(["a", "b"]), what="scores")
    with pytest.raises(ValueError):
        _join_order(np.array(["a", "a"]), np.array(["a", "b"]), what="scores")


def test_distances_are_euclidean_in_the_chosen_components():
    scores = np.array([[3.0, 100.0, 4.0], [0.0, 7.0, 0.0]])
    np.testing.assert_allclose(distances_to(scores, [0, 2], [0.0, 0.0]), [5.0, 0.0])


def test_thresholds_are_nested_and_end_at_all_particles():
    distances = np.arange(100.0)
    halves = [np.arange(0, 100, 2), np.arange(1, 100, 2)]
    thresholds = nested_thresholds(distances, n_candidates=10, n_min_per_half=5, halves=halves)
    assert np.all(np.diff(thresholds) > 0)
    assert thresholds[-1] >= distances.max()
    assert min(np.sum(distances[rows] <= thresholds[0]) for rows in halves) >= 5
