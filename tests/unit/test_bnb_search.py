"""CPU checks of the branch-and-bound pose search's host pieces (relax.scoring.bnb_search): its translation grid and
prior, the rotation hierarchy, the keep and significance rules and the window centre."""

import jax.numpy as jnp
import numpy as np

from relax.sampling.grids import relion_scoring_rotation_grid
from relax.scoring import bnb_search


def test_translation_grid_is_the_lattice_disk():
    """The grid holds every lattice point within the radius, the origin among them."""
    grid = bnb_search.translation_grid(3.0, 1.0)
    assert grid.shape == (29, 2)
    assert np.all(np.sum(grid**2, axis=1) <= 9.0 + 1e-9)
    assert np.any(np.all(grid == 0.0, axis=1))


def test_offset_prior_is_consistent_units():
    """The prior of an offset t pixels at pixel size a and width sigma is -|t a|^2 / (2 sigma^2)."""
    prior = bnb_search.offset_log_prior(np.array([[3.0, 4.0]]), pixel_size=2.0, sigma_offset_angstrom=5.0)
    np.testing.assert_allclose(prior, [-0.5 * 100.0 / 25.0], rtol=1e-6)


def test_children_cover_the_finer_grid_with_overlap():
    """Every order-2 rotation sits under its nearest order-1 parent and under every parent within the radius; with no
    radius the cells partition the finer grid."""
    coarse = relion_scoring_rotation_grid(1).rotations
    fine = relion_scoring_rotation_grid(2).rotations
    angles = bnb_search.rotation_angles_deg(coarse, fine)
    table = bnb_search.children_table(coarse, fine, radius_deg=0.0)
    np.testing.assert_array_equal(np.sort(table[table >= 0]), np.arange(fine.shape[0]))
    assert angles.min(axis=1).max() < 30.0
    wide = bnb_search.children_table(coarse, fine, radius_deg=22.5)
    for parent in (0, 100, 575):
        expected = np.flatnonzero((angles[:, parent] <= 22.5) | (np.argmin(angles, axis=1) == parent))
        np.testing.assert_array_equal(np.sort(wide[parent][wide[parent] >= 0]), expected)
    same = bnb_search.children_table(fine, fine, radius_deg=0.0)
    np.testing.assert_array_equal(same[:, 0], np.arange(fine.shape[0]))


def test_children_expand_once_per_image():
    """A child under two kept parents is scored once."""
    children = np.array([[0, 1, 2], [2, 3, -1], [4, -1, -1]], dtype=np.int32)
    kept = np.array([[0, 1, -1], [2, -1, -1]], dtype=np.int32)
    rows = bnb_search._expand_children(kept, children)
    assert sorted(rows[0][rows[0] >= 0].tolist()) == [0, 1, 2, 3]
    assert sorted(rows[1][rows[1] >= 0].tolist()) == [4]


def test_coarse_prior_bounds_its_children():
    """A coarse rotation's prior is the largest prior of its children."""
    table = np.array([[0, 1, -1], [2, 3, 4]], dtype=np.int32)
    prior = np.array([-1.0, -3.0, -0.5, -2.0, -4.0], dtype=np.float32)
    np.testing.assert_allclose(bnb_search.coarse_level_prior(prior, table), [-1.0, -0.5])


def test_keep_takes_the_mass_and_the_margin():
    """A level keeps the smallest set carrying the mass fraction, and every rotation within the margin."""
    masses = np.array([0.5, 0.3, 0.1999, 1e-4], dtype=np.float64)
    log_mass = jnp.asarray(np.concatenate([np.log(masses), [-np.inf]])[None].astype(np.float32))
    labels = jnp.arange(5, dtype=jnp.int32)[None]
    kept, count = bnb_search._keep(log_mass, labels, 0.9, 0.0, max_kept=8)
    assert int(count[0]) == 3 and set(np.asarray(kept[0][:3])) == {0, 1, 2}
    kept, count = bnb_search._keep(log_mass, labels, 0.5, 9.5, max_kept=8)
    assert int(count[0]) == 4
    kept, count = bnb_search._keep(log_mass, labels, 0.9999, 20.0, max_kept=2)
    assert kept.shape == (1, 2) and int(count[0]) == 2


def test_significance_is_the_smallest_set_of_the_fraction():
    """Significant samples are the highest-mass samples until their mass reaches the fraction, at most the cap."""
    scores = jnp.asarray(np.log(np.array([[0.1, 0.6, 0.25, 0.05], [0.97, 0.01, 0.01, 0.01]], dtype=np.float32)))
    mask = np.asarray(bnb_search._significant_mask(scores, 0.9, max_significants=-1))
    np.testing.assert_array_equal(mask, [[True, True, True, False], [True, False, False, False]])
    mask = np.asarray(bnb_search._significant_mask(scores, 0.9, max_significants=2))
    np.testing.assert_array_equal(mask[0], [False, True, True, False])


def test_window_centre_holds_the_most_mass():
    """A sharp posterior centres the window on its mode's pixel; a broad one near its mean; far mass does not pull it."""
    grid = bnb_search.translation_grid(8.0, 0.5)
    sharp = -0.5 * np.sum((grid - np.array([2.2, -1.4])) ** 2, axis=1) / 0.25
    far = np.sum((grid - np.array([-6.0, 6.0])) ** 2, axis=1) < 0.3
    sharp = np.where(far, sharp.max() - 1.0, sharp)
    broad = -0.5 * np.sum((grid - np.array([0.6, 0.4])) ** 2, axis=1) / 9.0
    centres, spread, hold = bnb_search.window_centres(np.stack([sharp, broad]), grid, 2.0)
    np.testing.assert_array_equal(centres, [[2.0, -1.0], [1.0, 0.0]])
    assert spread[1] > 2.0 and hold[1] > 6.0
    clean = -0.5 * np.sum((grid - np.array([2.0, -1.0])) ** 2, axis=1) / 0.25
    _, _, hold = bnb_search.window_centres(clean[None], grid, 2.0)
    assert 1.0 < hold[0] < 2.0


def test_block_scores_are_pass1_gaussian_scores():
    """The search's dense and gathered blocks give pass 1's coarse Gaussian scores on the same operands."""
    from relax.scoring.coarse_kernels import relion_coarse_gaussian_gemm_scores_jit

    rng = np.random.default_rng(0)
    n_img, n_trans, n_rot, n_pix = 3, 5, 7, 11
    shifted = (rng.normal(size=(n_img, n_trans, n_pix)) + 1j * rng.normal(size=(n_img, n_trans, n_pix))).astype(
        np.complex64
    )
    weight = rng.uniform(0.1, 2.0, size=(n_img, n_pix)).astype(np.float32)
    projections = (rng.normal(size=(n_rot, n_pix)) + 1j * rng.normal(size=(n_rot, n_pix))).astype(np.complex64)
    initial = rng.uniform(0, 5, size=n_img).astype(np.float32)
    reference = relion_coarse_gaussian_gemm_scores_jit(
        jnp.asarray(projections), None, jnp.asarray(shifted), jnp.asarray(weight), jnp.asarray(initial), n_img,
        n_images=n_img, n_trans=n_trans, image_shape=(8, 8), volume_shape=(8, 8, 8),
    )
    weighted, image_term = bnb_search._image_terms(jnp.asarray(shifted), jnp.asarray(weight), jnp.asarray(initial))
    zeros_r, zeros_t = jnp.zeros(n_rot, jnp.float32), jnp.zeros(n_trans, jnp.float32)
    dense = bnb_search._dense_block(weighted, image_term, jnp.asarray(weight), jnp.asarray(projections), zeros_r, zeros_t)
    np.testing.assert_allclose(np.swapaxes(np.asarray(dense), 1, 2), np.asarray(reference), rtol=1e-5, atol=1e-3)
    rows = jnp.asarray(np.array([[0, 3, -1], [6, 2, 1], [5, -1, -1]], dtype=np.int32))
    gathered = np.asarray(
        bnb_search._gathered_block(weighted, image_term, jnp.asarray(weight), jnp.asarray(projections), rows, zeros_r,
                                   zeros_t)
    )
    for b in range(n_img):
        for c, r in enumerate(np.asarray(rows[b])):
            if r < 0:
                assert np.all(np.isneginf(gathered[b, :, c]))
            else:
                np.testing.assert_allclose(gathered[b, :, c], np.asarray(reference)[b, r], rtol=1e-5, atol=1e-3)


def test_significant_ids_follow_the_columns():
    """Each significant column ``(candidate, w)`` becomes ``rotation * W + w``, ascending, padded with -1."""
    mask = jnp.asarray(np.array([[True, False, False, True, True, False], [False, False, False, True, False, False]]))
    rotations = jnp.asarray(np.array([[3, 7, 9], [2, 5, -1]], dtype=np.int32))
    ids = np.asarray(bnb_search._significant_ids(mask, rotations, size=4, n_window=2))
    np.testing.assert_array_equal(ids, [[6, 15, 18, -1], [11, -1, -1, -1]])


def test_window_moves_only_on_evidence():
    """A sharp posterior far from the previous offset moves the window; a broad one keeps it at the rounded previous
    offset when the best window would hold little more of its mass."""
    grid = bnb_search.translation_grid(10.0, 0.5)
    sharp = -0.5 * np.sum((grid - np.array([6.0, -5.0])) ** 2, axis=1) / 0.25
    broad = -0.5 * np.sum((grid - np.array([1.0, 1.0])) ** 2, axis=1) / 25.0
    previous = np.array([[0.2, 0.3], [-0.4, 1.6]])
    centres, _, _ = bnb_search.window_centres(np.stack([sharp, broad]), grid, 4.0, previous=previous, recentre_gain=0.1)
    np.testing.assert_array_equal(centres, [[6.0, -5.0], [0.0, 2.0]])
    centres, _, _ = bnb_search.window_centres(np.stack([sharp, broad]), grid, 4.0, previous=previous, recentre_gain=0.0)
    np.testing.assert_array_equal(centres[0], [6.0, -5.0])
