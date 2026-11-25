"""The FvML-mixture direction prior of a refinement (relax.sampling.fvml_direction_prior) and its input source
(relax.parity.fvml_prior_source): the fit, point groups, the prior on a rotation grid, the model STAR blocks, the
K=1 and Class3D sources and the runs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.tiny_main import controller_inputs
from helpers.tiny_refinement import run_tiny_refinement

from relax.parity.fvml_prior_source import PROVENANCE_FILE, FvmlDirectionPriorSource
from relax.refinement.ports import InputSource
from relax.refinement.run_files import _list_block, _loop_block
from relax.sampling import fvml_direction_prior as fdp
from relax.sampling.fvml_mixture import (
    FvmlMixture,
    cell_log_probabilities,
    healpix_cell_points,
    kappa_max_for_grid,
    weighted_cell_log_probabilities,
)
from relax.sampling.grids import rotation_grid_size
from relax.sampling.orientation_priors import DirectionPrior, HalfDirectionLogPriors
from relax.star.blocks import read_star_blocks

pytestmark = pytest.mark.unit


def _unit(rot_deg: float, tilt_deg: float) -> np.ndarray:
    rot, tilt = np.radians(rot_deg), np.radians(tilt_deg)
    return np.array([np.cos(rot) * np.sin(tilt), np.sin(rot) * np.sin(tilt), np.cos(tilt)])


# Two separated lobes and a uniform share.
TRUTH = FvmlMixture([0.2, 0.5, 0.3], np.stack([_unit(30.0, 40.0), _unit(200.0, 120.0)]), [30.0, 80.0])


def _masses(order: int) -> np.ndarray:
    """The truth's cell probabilities at ``order``: the posterior masses of infinitely many particles."""
    return np.exp(cell_log_probabilities(TRUTH, order, fdp.CELL_LEVELS))


def _options(*, n_classes=1, point_group="C1", sigma_ang_deg=None, init_healpix_order=2):
    """The fields ``FvmlDirectionPriorSource.for_run`` reads."""
    return SimpleNamespace(
        k_class=SimpleNamespace(n_classes=n_classes),
        symmetry=SimpleNamespace(point_group=point_group),
        local_search=SimpleNamespace(sigma_ang_deg=sigma_ang_deg),
        schedule=SimpleNamespace(init_healpix_order=init_healpix_order),
        precision=SimpleNamespace(rotation_real_dtype=np.float32),
    )


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


# ---- the grid ----


def test_cell_points_are_the_nest_children_of_each_pixel():
    points = healpix_cell_points(1, 2)
    assert points.shape == (48, 16, 3)
    parents = hp.vec2pix(2, points[..., 0], points[..., 1], points[..., 2], nest=True)
    assert np.array_equal(parents, np.repeat(np.arange(48)[:, None], 16, axis=1))


def test_cell_probabilities_are_normalized_and_uniform_for_the_uniform_component():
    assert_matches(np.exp(cell_log_probabilities(TRUTH, 3, 2)).sum(), 1.0)
    flat = FvmlMixture([1.0, 0.0], [[0.0, 0.0, 1.0]], [5.0])
    assert_matches(np.exp(cell_log_probabilities(flat, 2, 2)), np.full(192, 1.0 / 192))


def test_cell_log_probabilities_stay_finite_where_a_probability_underflows():
    sharp = FvmlMixture([0.0, 1.0], [[0.0, 0.0, 1.0]], [5000.0])
    log_p = cell_log_probabilities(sharp, 3, 2)
    assert np.all(np.isfinite(log_p)) and np.exp(log_p).min() == 0.0


# ---- the fit ----


def test_cold_fit_recovers_the_mixture_from_its_cell_masses():
    fit = fdp.fit_direction_prior(_masses(3), 3, symmetry="C1", n_lobes=2, previous=None, seed=0)
    assert fit.start == "cold" and fit.fit_order == 3 and fit.converged
    assert fit.kappa_max == kappa_max_for_grid(768, fdp.CAP_SPACINGS)
    order = np.argsort(-fit.mixture.weights[1:])
    for fitted, true in zip(order, range(2)):
        assert _angle_deg(fit.mixture.means[fitted], TRUTH.means[true]) < 1.0
        assert abs(fit.mixture.kappas[fitted] / TRUTH.kappas[true] - 1.0) < 0.05
    assert abs(fit.mixture.weights[0] - 0.2) < 0.01


def test_warm_fit_starts_from_the_previous_one():
    cold = fdp.fit_direction_prior(_masses(3), 3, symmetry="C1", n_lobes=2, previous=None, seed=0)
    warm = fdp.fit_direction_prior(_masses(3), 3, symmetry="C1", n_lobes=2, previous=cold, seed=0)
    assert warm.start == "warm"
    assert warm.em_iterations < cold.em_iterations
    assert warm.objective >= cold.objective - 1e-9


def test_warm_start_with_another_lobe_count_is_refused():
    cold = fdp.fit_direction_prior(_masses(2), 2, symmetry="C1", n_lobes=2, previous=None, seed=0)
    with pytest.raises(ValueError, match="2 lobes, the prior 3"):
        fdp.fit_direction_prior(_masses(2), 2, symmetry="C1", n_lobes=3, previous=cold, seed=0)


def test_c1_masses_fold_onto_their_nest_parents():
    assert_matches(fdp.fold_masses(_masses(4), 4, 2, "C1"), _masses(4).reshape(-1, 16).sum(axis=1))


def test_masses_finer_than_the_fit_order_are_fitted_as_their_fold(monkeypatch):
    monkeypatch.setattr(fdp, "FIT_MAX_ORDER", 2)
    fine = fdp.fit_direction_prior(_masses(4), 4, symmetry="C1", n_lobes=2, previous=None, seed=0)
    folded = fdp.fold_masses(_masses(4), 4, 2, "C1")
    coarse = fdp.fit_direction_prior(folded, 2, symmetry="C1", n_lobes=2, previous=None, seed=0)
    assert fine.fit_order == 2
    for name in ("weights", "means", "kappas"):
        assert_matches(getattr(fine.mixture, name), getattr(coarse.mixture, name))


@pytest.mark.parametrize(
    "masses, message",
    [(np.ones(47), "C1 directions of order 1"), (np.zeros(48), "not all zero"), (-np.ones(48), "non-negative")],
)
def test_masses_that_are_not_a_direction_histogram_are_refused(masses, message):
    with pytest.raises(ValueError, match=message):
        fdp.fit_direction_prior(masses, 1, symmetry="C1", n_lobes=1, previous=None, seed=0)


def test_rotation_log_prior_repeats_the_direction_log_prior_over_the_in_plane_angles():
    fit = fdp.fit_direction_prior(_masses(2), 2, symmetry="C1", n_lobes=2, previous=None, seed=0)
    for order in (1, 2, 3):
        table = fdp.rotation_log_prior(fit, order, dtype=np.float32)
        n_rot, n_directions = rotation_grid_size(order, "C1"), 12 * 4**order
        assert table.shape == (n_rot,) and table.dtype == np.float32
        expected = fdp.direction_log_probabilities(fit, order)[np.arange(n_rot) % n_directions]
        assert_matches(table, expected.astype(np.float32))


# ---- point groups ----

GROUP_ORDERS = {"C2": 2, "C3": 3, "C7": 7, "D2": 4, "D3": 6, "D7": 14, "T": 12, "O": 24, "I": 60}


def _symmetric_truth(symmetry: str, kappas=(30.0, 80.0)) -> FvmlMixture:
    """``TRUTH``'s two lobes (rot, tilt) (30, 40) and (200, 110), copied by the group's rotations."""
    means = np.stack([_unit(30.0, 40.0), _unit(200.0, 110.0)])
    return FvmlMixture([0.2, 0.5, 0.3], means, list(kappas), fdp.symmetry_maps(symmetry))


def _cell_masses(mixture: FvmlMixture, order: int, symmetry: str) -> np.ndarray:
    """The mixture's probability of each of RELION's directions' cells at ``order``."""
    cells = fdp.orbit_cells(order, symmetry)
    return np.exp(weighted_cell_log_probabilities(mixture, cells.children, cells.weights, cells.starts))


def _sphere_children(order: int) -> np.ndarray:
    """Centres of every HEALPix pixel ``CELL_LEVELS`` orders finer than ``order``."""
    nside = 1 << (order + fdp.CELL_LEVELS)
    return np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True), axis=1)


@pytest.mark.parametrize("symmetry", sorted(GROUP_ORDERS))
def test_symmetry_maps_are_the_groups_rotations_identity_first(symmetry):
    maps = fdp.symmetry_maps(symmetry)
    assert maps.shape == (GROUP_ORDERS[symmetry], 3, 3)
    assert_matches(maps[0], np.eye(3))
    assert_matches(maps @ maps.transpose(0, 2, 1), np.broadcast_to(np.eye(3), maps.shape))
    assert_matches(np.linalg.det(maps), np.ones(len(maps)))
    products = np.einsum("iab,jbc->ijac", maps, maps).reshape(-1, 1, 9)
    # Closed up to RELION's rounded constants (measured: T 5.9e-7, I 5.1e-8, the others 2e-15).
    assert np.abs(products - maps.reshape(1, -1, 9)).max(axis=2).min(axis=1).max() < 1e-6


PIXEL_RADIUS_ORDER_3 = np.radians(7.47)  # relax.sampling.healpix.max_pixrad(3)


def _widest_hole(maps: np.ndarray, symmetry: str) -> float:
    """The largest angle from a pixel centre at order 3 to the nearest image under ``maps`` of RELION's directions."""
    kept = np.stack(hp.pix2vec(8, fdp.direction_pixels(3, symmetry), nest=True), axis=1)
    centres = np.stack(hp.pix2vec(8, np.arange(768), nest=True), axis=1)
    images = np.einsum("gab,mb->gma", maps, kept).reshape(-1, 3)
    return float(np.arccos(np.clip((centres @ images.T).max(axis=1).min(), -1.0, 1.0)))


@pytest.mark.parametrize("symmetry", sorted(GROUP_ORDERS))
def test_relions_directions_and_their_images_cover_the_sphere(symmetry):
    """The convention check: the maps move RELION's asymmetric unit onto the rest of the sphere, so every pixel
    centre lies within a pixel radius of an image of a kept direction (measured: at most 6.8 degrees)."""
    assert _widest_hole(fdp.symmetry_maps(symmetry), symmetry) < PIXEL_RADIUS_ORDER_3


@pytest.mark.parametrize("symmetry", sorted(set(GROUP_ORDERS) - {"C2"}))
def test_the_cover_fails_in_another_frame(symmetry):
    """The check above tells frames apart: the maps conjugated by a 37 degree rotation about x leave holes of 11 to
    44 degrees (measured; C2's tilted two-fold still covers the sphere to 5.4 degrees, so it is left out)."""
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(0.65), -np.sin(0.65)], [0.0, np.sin(0.65), np.cos(0.65)]])
    assert _widest_hole(tilt @ fdp.symmetry_maps(symmetry) @ tilt.T, symmetry) > PIXEL_RADIUS_ORDER_3


@pytest.mark.parametrize("symmetry", sorted(GROUP_ORDERS))
@pytest.mark.parametrize("order", [2, 3, 4])
def test_every_orbit_has_points_in_the_cell_region(symmetry, order):
    region = np.zeros(12 * 4**order, dtype=bool)
    region[fdp.cell_region(order, symmetry)] = True
    images = np.einsum("gab,nb->gna", fdp.symmetry_maps(symmetry), _sphere_children(order))
    pixels = hp.vec2pix(1 << order, images[..., 0], images[..., 1], images[..., 2], nest=True)
    assert region[pixels].any(axis=0).all()


def test_c1_cells_are_the_pixels_nest_children():
    cells = fdp.orbit_cells(3, "C1")
    assert_matches(cells.children, healpix_cell_points(3, fdp.CELL_LEVELS).reshape(-1, 3))
    assert np.all(cells.weights == 1.0)
    assert np.array_equal(cells.starts, np.arange(768) * 16)


@pytest.mark.parametrize("symmetry", sorted(GROUP_ORDERS))
@pytest.mark.parametrize("order", [3, 4])
def test_cells_tile_the_sphere(symmetry, order):
    """Every direction has a cell, and the cells' weights add up to the sphere (measured: within 1.0%, C7 and D7 at
    order 3; exactly in C2 and D2, whose images of pixel centres are pixel centres)."""
    cells = fdp.orbit_cells(order, symmetry)
    n_directions = fdp.direction_pixels(order, symmetry).size
    assert cells.starts.size == n_directions and np.all(np.diff(np.append(cells.starts, cells.weights.size)) > 0)
    assert abs(cells.weights.sum() / (12 * 4 ** (order + fdp.CELL_LEVELS)) - 1.0) < 0.015


def test_a_uniform_mixture_gives_each_direction_its_cells_area():
    flat = FvmlMixture([1.0, 0.0], [[0.0, 0.0, 1.0]], [5.0], fdp.symmetry_maps("T"))
    cells = fdp.orbit_cells(3, "T")
    area = np.add.reduceat(cells.weights, cells.starts)
    assert_matches(_cell_masses(flat, 3, "T"), area / area.sum())


def test_directions_relion_keeps_twice_share_their_orbits():
    """Above 4000 directions RELION skips its seam pass, so C2 at order 5 keeps 64 directions together with their
    image; each pair shares its orbits and so its prior."""
    order, symmetry = 5, "C2"
    pixels = fdp.direction_pixels(order, symmetry)
    row_of_pixel = {int(p): r for r, p in enumerate(pixels)}
    centres = np.stack(hp.pix2vec(32, pixels, nest=True), axis=1)
    images = centres @ fdp.symmetry_maps(symmetry)[1].T
    image_pixels = hp.vec2pix(32, images[:, 0], images[:, 1], images[:, 2], nest=True)
    pairs = np.array([(r, row_of_pixel[int(p)]) for r, p in enumerate(image_pixels) if int(p) in row_of_pixel])
    assert len(pairs) == 64
    masses = _cell_masses(_symmetric_truth(symmetry), order, symmetry)
    assert np.all(masses > 0.0)
    assert_matches(masses[pairs[:, 0]], masses[pairs[:, 1]], rtol=1e-9)  # same children, summed in another order


@pytest.mark.parametrize(
    "symmetry, kappas",
    [("C3", (30.0, 80.0)), ("D2", (30.0, 80.0)), ("D7", (30.0, 80.0)), ("T", (30.0, 80.0)), ("O", (60.0, 150.0))],
)
def test_symmetric_fit_recovers_the_mixture_from_its_cell_masses(symmetry, kappas):
    """At order 3 (measured: lobes within 0.11 degrees and 3% in kappa, the prior within 3.3% of the truth's). I is
    left out: 60 overlapping copies of these lobes leave its parameters unidentifiable, and a fit takes 100 s."""
    truth = _symmetric_truth(symmetry, kappas)
    masses = _cell_masses(truth, 3, symmetry)
    fit = fdp.fit_direction_prior(masses, 3, symmetry=symmetry, n_lobes=2, previous=None, seed=0)
    assert fit.symmetry == symmetry and fit.converged
    assert fit.kappa_max == kappa_max_for_grid(768, fdp.CAP_SPACINGS)
    assert abs(fit.mixture.weights[0] - 0.2) < 0.01
    maps = fdp.symmetry_maps(symmetry)
    for true_mean, true_kappa, true_weight in zip(truth.means, truth.kappas, truth.weights[1:]):
        nearest = [float((maps @ mean @ true_mean).max()) for mean in fit.mixture.means]
        lobe = int(np.argmax(nearest))
        assert np.degrees(np.arccos(min(nearest[lobe], 1.0))) < 0.5
        assert abs(fit.mixture.kappas[lobe] / true_kappa - 1.0) < 0.05
        assert abs(fit.mixture.weights[lobe + 1] - true_weight) < 0.01
    prior = np.exp(fdp.direction_log_probabilities(fit, 3))
    assert np.abs(prior / masses - 1.0).max() < 0.05


def test_symmetric_masses_finer_than_the_fit_order_are_folded_onto_it(monkeypatch):
    """C3 masses at order 5 fitted at order 4: the prior at order 5 within 3% of the truth's, by a median of
    0.035% (measured); folding keeps the total."""
    monkeypatch.setattr(fdp, "FIT_MAX_ORDER", 4)
    truth = _symmetric_truth("C3")
    masses = _cell_masses(truth, 5, "C3")
    assert_matches(fdp.fold_masses(masses, 5, 4, "C3").sum(), 1.0)
    fit = fdp.fit_direction_prior(masses, 5, symmetry="C3", n_lobes=2, previous=None, seed=0)
    assert fit.fit_order == 4
    error = np.abs(np.exp(fdp.direction_log_probabilities(fit, 5)) / masses - 1.0)
    assert error.max() < 0.05 and np.median(error) < 1e-3


def test_symmetric_rotation_log_prior_repeats_the_direction_log_prior_over_the_in_plane_angles():
    fit = fdp.fit_direction_prior(_cell_masses(_symmetric_truth("C3"), 2, "C3"), 2, symmetry="C3", n_lobes=2,
                                  previous=None, seed=0)
    for order in (2, 3):
        table = fdp.rotation_log_prior(fit, order, dtype=np.float32)
        n_rot, n_directions = rotation_grid_size(order, "C3"), fdp.direction_pixels(order, "C3").size
        assert table.shape == (n_rot,)
        expected = fdp.direction_log_probabilities(fit, order)[np.arange(n_rot) % n_directions]
        assert_matches(table, expected.astype(np.float32))


def test_warm_start_with_another_symmetry_is_refused():
    cold = fdp.fit_direction_prior(_masses(2), 2, symmetry="C1", n_lobes=2, previous=None, seed=0)
    masses = _cell_masses(_symmetric_truth("C2"), 2, "C2")
    with pytest.raises(ValueError, match="symmetry C1, the prior C2"):
        fdp.fit_direction_prior(masses, 2, symmetry="C2", n_lobes=2, previous=cold, seed=0)


# ---- the model STAR blocks ----


def test_fit_round_trips_through_its_star_blocks(tmp_path):
    fit = fdp.fit_direction_prior(_masses(2), 2, symmetry="C1", n_lobes=2, previous=None, seed=0)
    path = tmp_path / "model.star"
    path.write_text(
        _list_block(fdp.STAR_FIT_BLOCK, fdp.star_fit_items(fit))
        + _loop_block(fdp.STAR_COMPONENT_BLOCK, fdp.star_component_columns(fit))
    )
    blocks = read_star_blocks(path)
    columns = blocks[fdp.STAR_COMPONENT_BLOCK]
    assert columns["relax_fvml_component"] == ["0", "1", "2"]
    for lobe in range(2):
        rot, tilt = (float(columns[f"relax_fvml_{a}_deg"][lobe + 1]) for a in ("rot", "tilt"))
        assert _angle_deg(_unit(rot, tilt), fit.mixture.means[lobe]) < 1e-6
    read = fdp.fit_from_star(blocks[fdp.STAR_FIT_BLOCK], columns, symmetry="C1")
    for name in ("weights", "means", "kappas"):
        assert_matches(getattr(read.mixture, name), getattr(fit.mixture, name))
    for name in ("symmetry", "fit_order", "kappa_max", "start", "em_iterations", "converged", "objective", "seconds"):
        assert getattr(read, name) == getattr(fit, name), name


def test_symmetric_fit_reads_back_with_the_runs_point_group(tmp_path):
    fit = fdp.fit_direction_prior(_cell_masses(_symmetric_truth("D2"), 2, "D2"), 2, symmetry="D2", n_lobes=2,
                                  previous=None, seed=0)
    fit_block, component_block = fdp.star_block_names(None)
    path = tmp_path / "model.star"
    path.write_text(
        _list_block(fit_block, fdp.star_fit_items(fit)) + _loop_block(component_block, fdp.star_component_columns(fit))
    )
    blocks = read_star_blocks(path)
    read = fdp.fit_from_star(blocks[fit_block], blocks[component_block], symmetry="D2")
    assert read.symmetry == "D2"
    assert_matches(read.mixture.symmetry, fdp.symmetry_maps("D2"))
    assert_matches(fdp.direction_log_probabilities(read, 3), fdp.direction_log_probabilities(fit, 3), rtol=1e-12)


def test_class_fits_have_class_numbered_blocks():
    assert fdp.star_block_names(None) == (fdp.STAR_FIT_BLOCK, fdp.STAR_COMPONENT_BLOCK)
    assert fdp.star_block_names(1) == (f"{fdp.STAR_FIT_BLOCK}_class_2", f"{fdp.STAR_COMPONENT_BLOCK}_class_2")


# ---- the source ----


@pytest.mark.parametrize(
    "replacing, options, flags, message",
    [
        (FvmlDirectionPriorSource(5, n_classes=1, symmetry="C1", dtype=np.float32), _options(), {}, "cannot be combined"),
        (InputSource(), _options(n_classes=2, sigma_ang_deg=3.0), {}, "global searches only"),
        (InputSource(), _options(point_group="I", init_healpix_order=2), {}, "only 2 directions at HEALPix order 2"),
        (InputSource(), _options(), {"tomographic": True}, "single-particle"),
        (InputSource(), _options(), {"multi_gpu": True}, "one GPU"),
    ],
)
def test_runs_the_prior_cannot_serve_are_refused(replacing, options, flags, message):
    with pytest.raises(ValueError, match=message):
        FvmlDirectionPriorSource.for_run(
            5, replacing, options, **{"tomographic": False, "multi_gpu": False, **flags}
        )


def test_source_fits_each_half_and_warm_starts_it():
    source = FvmlDirectionPriorSource.for_run(2, InputSource(), _options(), tomographic=False, multi_gpu=False)
    learned = tuple(DirectionPrior(_masses(2).astype(np.float32), 2) for _ in range(2))
    first = source.learned_direction_priors(learned, previous=[DirectionPrior(None, None)] * 2)
    assert [p.fits[0].start for p in first] == ["cold", "cold"]
    for prior in first:
        assert prior.healpix_order == 2 and prior.values.dtype == np.float32 and len(prior.fits) == 1
        assert_matches(prior.values, np.exp(fdp.direction_log_probabilities(prior.fits[0], 2)).astype(np.float32))
    second = source.learned_direction_priors((None, learned[1]), previous=list(first))
    assert second[0] is None
    assert second[1].fits[0].start == "warm"


def test_source_scores_with_the_halfs_fit_and_natively_before_it():
    source = FvmlDirectionPriorSource(2, n_classes=1, symmetry="C1", dtype=np.float32)
    own = HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None)
    assert source.direction_log_prior(0, own, prior=DirectionPrior(None, None), healpix_order=2, use_local=False) is own
    fit = fdp.fit_direction_prior(_masses(2), 2, symmetry="C1", n_lobes=2, previous=None, seed=0)
    for use_local in (False, True):
        log_prior = source.direction_log_prior(
            1, own, prior=DirectionPrior(None, 2, (fit,)), healpix_order=3, use_local=use_local
        )
        assert log_prior.class_rotation_log_prior is None
        assert_matches(log_prior.rotation_log_prior, fdp.rotation_log_prior(fit, 3, dtype=np.float32))
        assert not log_prior.rotation_log_prior.flags.writeable


# A second class's distribution, unlike TRUTH's.
OTHER = FvmlMixture([0.3, 0.4, 0.3], np.stack([_unit(120.0, 60.0), _unit(300.0, 150.0)]), [50.0, 20.0])


def _class_learned(order: int = 2) -> tuple:
    """Class3D's learned prior at ``order``: TRUTH's masses for class 1, OTHER's for class 2, the same in both
    halves (as ``learn_class_direction_priors`` copies it)."""
    values = np.stack([_masses(order), np.exp(cell_log_probabilities(OTHER, order, fdp.CELL_LEVELS))])
    return DirectionPrior(values.astype(np.float32), order), DirectionPrior(values.astype(np.float32), order)


def _class_source(n_lobes: int = 2) -> FvmlDirectionPriorSource:
    return FvmlDirectionPriorSource.for_run(
        n_lobes, InputSource(), _options(n_classes=2), tomographic=False, multi_gpu=False
    )


def test_class_source_fits_each_class_once_and_both_halves_share_it():
    source = _class_source()
    learned = _class_learned()
    first = source.learned_direction_priors(learned, previous=[DirectionPrior(None, None)] * 2)
    assert first[0] is first[1]
    prior = first[0]
    assert prior.healpix_order == 2 and prior.values.shape == (2, 192) and prior.values.dtype == np.float32
    assert [fit.start for fit in prior.fits] == ["cold", "cold"]
    for class_index, (fit, truth) in enumerate(zip(prior.fits, (TRUTH, OTHER), strict=True)):
        # Fitted to the class's own masses, cold starts seeded by the class.
        direct = fdp.fit_direction_prior(
            learned[0].values[class_index], 2, symmetry="C1", n_lobes=2, previous=None, seed=class_index
        )
        assert_matches(fit.mixture.means, direct.mixture.means)
        for true_mean in truth.means:
            assert min(_angle_deg(mean, true_mean) for mean in fit.mixture.means) < 1.0
        assert_matches(prior.values[class_index], np.exp(fdp.direction_log_probabilities(fit, 2)).astype(np.float32))
    second = source.learned_direction_priors(learned, previous=list(first))
    assert [fit.start for fit in second[0].fits] == ["warm", "warm"]


@pytest.mark.parametrize(
    "learned",
    [
        (_class_learned()[0], DirectionPrior(_class_learned()[0].values[::-1].copy(), 2)),
        (_class_learned()[0], None),
    ],
)
def test_class_source_refuses_halves_with_different_learned_priors(learned):
    with pytest.raises(ValueError, match="same learned direction prior"):
        _class_source().learned_direction_priors(learned, previous=[DirectionPrior(None, None)] * 2)


def test_class_source_keeps_the_priors_when_nothing_was_learned():
    assert _class_source().learned_direction_priors((None, None), previous=[DirectionPrior(None, None)] * 2) == (
        None, None,
    )


def test_class_source_scores_each_class_with_its_mixture():
    source = _class_source()
    prior = source.learned_direction_priors(_class_learned(), previous=[DirectionPrior(None, None)] * 2)[0]
    own = HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None)
    log_prior = source.direction_log_prior(1, own, prior=prior, healpix_order=3, use_local=False)
    assert log_prior.rotation_log_prior is None
    assert log_prior.class_rotation_log_prior.shape == (2, rotation_grid_size(3, "C1"))
    assert not log_prior.class_rotation_log_prior.flags.writeable
    for row, fit in zip(log_prior.class_rotation_log_prior, prior.fits, strict=True):
        assert_matches(row, fdp.rotation_log_prior(fit, 3, dtype=np.float32))
    with pytest.raises(ValueError, match="global searches only"):
        source.direction_log_prior(1, own, prior=prior, healpix_order=3, use_local=True)


def test_symmetric_source_fits_on_the_asymmetric_unit():
    source = FvmlDirectionPriorSource.for_run(
        2, InputSource(), _options(point_group="D2"), tomographic=False, multi_gpu=False
    )
    masses = _cell_masses(_symmetric_truth("D2"), 2, "D2").astype(np.float32)
    priors = source.learned_direction_priors(
        (DirectionPrior(masses, 2), None), previous=[DirectionPrior(None, None)] * 2
    )
    assert priors[1] is None
    (fit,) = priors[0].fits
    assert fit.symmetry == "D2" and priors[0].values.shape == (48,)
    own = HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None)
    log_prior = source.direction_log_prior(0, own, prior=priors[0], healpix_order=3, use_local=True)
    assert log_prior.rotation_log_prior.shape == (rotation_grid_size(3, "D2"),)
    assert_matches(log_prior.rotation_log_prior, fdp.rotation_log_prior(fit, 3, dtype=np.float32))


class _Recording(FvmlDirectionPriorSource):
    fits: list = []

    def learned_direction_priors(self, learned, *, previous):
        priors = super().learned_direction_priors(learned, previous=previous)
        self.fits.append([p.fits[0] for p in priors])
        return priors


def test_every_expectation_after_the_first_scores_with_the_halfs_latest_fit(monkeypatch):
    """Two numbered iterations and the final pass: the first scores natively, each later one with its half's fit."""
    calls = []
    _Recording.fits = []
    run_tiny_refinement(
        monkeypatch, max_iter=2, engine_calls=calls,
        source_for=lambda options: _Recording.for_run(1, InputSource(), options, tomographic=False, multi_gpu=False),
    )
    assert len(calls) == 6 and len(_Recording.fits) == 2
    assert [fit.start for fit in _Recording.fits[0]] == ["cold", "cold"]
    assert [fit.start for fit in _Recording.fits[1]] == ["warm", "warm"]
    for call in calls[:2]:
        assert call["kwargs"]["rotation_log_prior"] is None
    # Iteration 2 scores with iteration 1's fits; the final pass with iteration 2's (halves in order).
    for call, fit in zip(calls[2:], _Recording.fits[0] + _Recording.fits[1], strict=True):
        assert_matches(np.asarray(call["kwargs"]["rotation_log_prior"]), fdp.rotation_log_prior(fit, 2, dtype=np.float32))


class _ClassRecording(FvmlDirectionPriorSource):
    fits: list = []

    def learned_direction_priors(self, learned, *, previous):
        priors = super().learned_direction_priors(learned, previous=previous)
        self.fits.append(priors[0].fits)
        return priors


def test_every_class3d_expectation_after_the_first_scores_each_class_with_its_latest_fit(monkeypatch):
    """Three Class3D iterations: the first scores natively, each later one (both halves) with the previous fits."""
    calls = []
    _ClassRecording.fits = []
    run_tiny_refinement(
        monkeypatch, n_classes=2, max_iter=3, engine_calls=calls,
        source_for=lambda options: _ClassRecording.for_run(1, InputSource(), options, tomographic=False, multi_gpu=False),
    )
    assert len(calls) == 6 and len(_ClassRecording.fits) == 3
    assert [fit.start for fit in _ClassRecording.fits[0]] == ["cold", "cold"]
    assert [fit.start for fit in _ClassRecording.fits[1]] == ["warm", "warm"]
    for call in calls[:2]:
        assert call["kwargs"]["class_rotation_log_prior"] is None
    for iteration, fits in ((1, _ClassRecording.fits[0]), (2, _ClassRecording.fits[1])):
        for call in calls[2 * iteration : 2 * iteration + 2]:
            table = np.asarray(call["kwargs"]["class_rotation_log_prior"])
            assert table.shape[0] == 2 and call["kwargs"]["rotation_log_prior"] is None
            (order,) = [o for o in range(5) if rotation_grid_size(o, "C1") == table.shape[1]]
            for row, fit in zip(table, fits, strict=True):
                assert_matches(row, fdp.rotation_log_prior(fit, order, dtype=np.float32))


# ---- the command ----


def test_flag_selects_the_fvml_source_and_records_its_settings(monkeypatch, tmp_path):
    inputs = controller_inputs(monkeypatch, tmp_path, "refine", "--fvml-direction-prior", "5")
    assert type(inputs["source"]) is FvmlDirectionPriorSource and inputs["source"].n_lobes == 5
    record = json.loads((tmp_path / "out" / PROVENANCE_FILE).read_text())
    assert record["lobes"] == 5 and record["cap_spacings"] == fdp.CAP_SPACINGS


def test_flag_with_a_fixed_prior_is_refused(monkeypatch, tmp_path):
    path = tmp_path / "prior.npy"
    np.save(path, np.full(192, 1.0 / 192))
    with pytest.raises(ValueError, match="cannot be combined"):
        controller_inputs(
            monkeypatch, tmp_path, "refine", "--max_healpix_order", "2", "--fixed-direction-prior", path,
            "--fvml-direction-prior", "5",
        )


def test_flag_in_a_symmetric_refinement_fits_on_the_asymmetric_unit(monkeypatch, tmp_path):
    inputs = controller_inputs(monkeypatch, tmp_path, "refine", "--sym", "C3", "--fvml-direction-prior", "5")
    source = inputs["source"]
    assert (source.n_classes, source.symmetry) == (1, "C3")
    record = json.loads((tmp_path / "out" / PROVENANCE_FILE).read_text())
    assert (record["classes"], record["symmetry"]) == (1, "C3")


def test_flag_in_class3d_fits_each_class(monkeypatch, tmp_path):
    inputs = controller_inputs(
        monkeypatch, tmp_path, "class3d", "--n_classes", "2", "--fvml-direction-prior", "3", n_classes=2
    )
    source = inputs["source"]
    assert type(source) is FvmlDirectionPriorSource and (source.n_lobes, source.n_classes) == (3, 2)
    assert json.loads((tmp_path / "out" / PROVENANCE_FILE).read_text())["classes"] == 2


def test_flag_in_class3d_with_local_searches_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="global searches only"):
        controller_inputs(
            monkeypatch, tmp_path, "class3d", "--n_classes", "2", "--sigma_ang", "5", "--fvml-direction-prior", "3",
            n_classes=2,
        )
