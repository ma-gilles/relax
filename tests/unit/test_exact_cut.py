"""The exact cut: rotations the coarse GEMM scores leave undecided at the significance cut are scored by the direct square."""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax.numpy as jnp

from relax.scoring import exact_cut

pytestmark = pytest.mark.unit


def test_first_rows_are_each_images_true_columns_in_order():
    rows = np.zeros((4, 9), bool)
    rows[0, [1, 4, 8]] = True
    rows[2] = True
    rows[3, 7] = True
    ids, use = exact_cut.first_rows(jnp.asarray(rows), capacity=4)
    ids, use = np.asarray(ids), np.asarray(use)
    np.testing.assert_array_equal(use, [[True, True, True, False], [False] * 4, [True] * 4, [True, False, False, False]])
    np.testing.assert_array_equal(ids[0, :3], [1, 4, 8])
    np.testing.assert_array_equal(ids[2], [0, 1, 2, 3])  # an image with more than the capacity keeps its first
    assert ids[3, 0] == 7
    # Row zero is written once although the empty slots carry id zero.
    table = np.arange(4 * 9 * 2, dtype=np.float32).reshape(4, 9, 2)
    new = -np.ones((4, 4, 2), np.float32)
    put = np.asarray(exact_cut.put_rows(jnp.asarray(table), jnp.asarray(ids), jnp.asarray(use), jnp.asarray(new)))
    expected = table.copy()
    expected[0, [1, 4, 8]] = -1.0
    expected[2, :4] = -1.0
    expected[3, 7] = -1.0
    assert_matches(put, expected)
    assert exact_cut.row_capacity(1, 576) == 64 and exact_cut.row_capacity(65, 576) == 128
    assert exact_cut.row_capacity(100, 80) == 80


@pytest.mark.parametrize("limit", [5, 16, 64, 300])
@pytest.mark.parametrize("width", [1, 4, 16])
def test_window_rows_are_slices_of_each_images_first_true_columns(limit, width):
    """Each window of :func:`exact_cut.window_rows` is that slice of each image's first ``limit`` true columns: the
    same ids and slots in use, zero ids past an image's count, nothing past ``limit``; ``first_rows`` is the
    window from rank zero."""

    rng = np.random.default_rng(width * 1000 + limit)
    rows = rng.random((7, 300)) < rng.uniform(0.0, 0.6, size=(7, 1))
    rows[0] = False
    capacity = -(-limit // width) * width
    ids = np.zeros((rows.shape[0], capacity), np.int32)
    use = np.zeros((rows.shape[0], capacity), bool)
    for image, row in enumerate(rows):
        columns = np.flatnonzero(row)[:limit]
        ids[image, : columns.size] = columns
        use[image, : columns.size] = True
    for first in range(0, capacity, width):
        got_ids, got_use = (
            np.asarray(v)
            for v in exact_cut.window_rows(jnp.asarray(rows), jnp.int32(first), jnp.int32(limit), width=width)
        )
        np.testing.assert_array_equal(got_use, use[:, first : first + width])
        np.testing.assert_array_equal(got_ids, ids[:, first : first + width])
    got_ids, got_use = (np.asarray(v) for v in exact_cut.first_rows(jnp.asarray(rows), capacity=limit))
    np.testing.assert_array_equal(got_use, use[:, :limit])
    np.testing.assert_array_equal(got_ids, ids[:, :limit])


def test_undecided_rotations_need_the_cut_set_by_max_significants():
    """Samples within the margin of the cut are undecided; without a diff2 (scores that carry their priors) the
    minimum's neighbourhood is not; an image cut by the adaptive fraction, or with a negative margin, has none."""

    n_rot, n_trans, cap = 4, 3, 2
    values = np.full((4, n_rot * n_trans), -50.0, np.float32)
    values[:, 0] = 0.0
    values[:, 4] = -3.0  # rank two, rotation 1
    values[:, 9] = np.float32(-3.0005)  # a hair below the cut, rotation 3
    mask = np.zeros(values.shape, bool)
    mask[:, [0, 4]] = True
    cutoff_count = np.array([cap, cap, cap - 1, cap], np.int32)
    margin = np.array([1e-3, 1e-5, 1e-3, -1.0], np.float32)
    rows, most, _ = exact_cut.undecided_rotations(
        jnp.asarray(values), jnp.asarray(mask), jnp.asarray(cutoff_count), jnp.asarray(margin), None, None,
        n_trans=n_trans, max_significants=cap,
    )
    np.testing.assert_array_equal(
        np.asarray(rows),
        [[False, True, False, True], [False, True, False, False], [False] * 4, [False] * 4],
    )
    assert int(most) == 2


def _mass_cut(values, fraction):
    """RELION's adaptive-fraction cut of one image's log weights: the samples kept."""
    weights = np.exp(values.astype(np.float64) - values.max())
    order = np.argsort(weights, kind="stable")
    cumulative = np.cumsum(weights[order])
    first_kept = int(np.searchsorted(cumulative, (1.0 - fraction) * cumulative[-1], side="right"))
    kept = np.zeros(values.size, bool)
    kept[order[first_kept:]] = True
    return kept


def test_undecided_rotations_at_the_adaptive_fraction_cut_cover_every_cut_within_the_bound():
    """For images the adaptive fraction cuts: perturb every log weight by up to half the margin, many times;
    a sample whose membership ever changes lies in an undecided rotation."""

    rng = np.random.default_rng(3)
    n_images, n_rot, n_trans, fraction = 6, 40, 5, 0.999
    values = (-rng.exponential(4.0, size=(n_images, n_rot * n_trans))).astype(np.float32)
    values[:, 0] = 0.0
    margin = np.array([1e-4, 1e-3, 1e-2, 5e-2, 0.2, -1.0], np.float32)
    mask = np.stack([_mass_cut(row, fraction) for row in values])
    cutoff_count = mask.sum(axis=1).astype(np.int32)
    rows, most, _ = exact_cut.undecided_rotations(
        jnp.asarray(values), jnp.asarray(mask), jnp.asarray(cutoff_count), jnp.asarray(margin), None, None,
        n_trans=n_trans, max_significants=0, adaptive_fraction=fraction,
    )
    rows = np.asarray(rows)
    assert not rows[5].any()  # a negative margin: not scored again
    counts = rows.sum(axis=1)
    assert np.all(np.diff(counts[:5]) >= 0) and counts[0] < n_rot // 4 and counts[4] >= 1
    assert int(most) == counts.max()
    n_flipped = 0
    for image in range(5):
        flipped = np.zeros(n_rot * n_trans, bool)
        for _ in range(300):
            moved = values[image] + rng.uniform(-0.5, 0.5, size=values.shape[1]).astype(np.float32) * margin[image]
            flipped |= _mass_cut(moved, fraction) != mask[image]
        n_flipped += int(flipped.sum())
        assert not flipped.reshape(n_rot, n_trans)[~rows[image]].any(), f"image {image}"
    assert n_flipped >= 1  # the perturbations do move the cut somewhere

    # Sorting the samples of the leading rotations only gives the same rows for every image it resolves, and
    # says which images it does not (each image's rows are its own, whatever the others in the call).
    args = (jnp.asarray(values), jnp.asarray(mask), jnp.asarray(cutoff_count), jnp.asarray(margin), None, None)
    static = dict(n_trans=n_trans, max_significants=0, adaptive_fraction=fraction)
    for sorted_rotations in (n_rot - 5, 2):
        partial_rows, _, unresolved = exact_cut.undecided_rotations(*args, **static, sorted_rotations=sorted_rotations)
        unresolved = np.asarray(unresolved)
        assert unresolved.shape == (values.shape[0],)
        np.testing.assert_array_equal(np.asarray(partial_rows)[~unresolved], rows[~unresolved])
    assert unresolved.any()  # two rotations do not hold the 0.1% tail of these weights


def test_the_error_bound_scales_with_the_pixels_and_the_largest_term():
    bound = exact_cut.gemm_error_bound(jnp.float32(100.0), jnp.float32(400.0), jnp.float32(50.0), 60)
    largest = 50.0 + 50.0 + 200.0 + 200.0
    assert_matches(float(bound), (4 * 60 + 11) * 2.0**-24 * largest, rtol=1e-6)
    assert_matches(float(exact_cut.largest_term(jnp.float32(100.0), jnp.float32(400.0), jnp.float32(50.0))), largest)
