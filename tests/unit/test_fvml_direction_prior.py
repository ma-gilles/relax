"""The FvML-mixture direction prior of a refinement (relax.sampling.fvml_direction_prior) and its input source
(relax.parity.fvml_prior_source): the fit, the prior on a rotation grid, the model STAR blocks and the run."""

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
from relax.sampling.fvml_mixture import FvmlMixture, cell_log_probabilities, healpix_cell_points, kappa_max_for_grid
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


def _options(*, n_classes=1, point_group="C1"):
    """The fields ``FvmlDirectionPriorSource.for_run`` reads."""
    return SimpleNamespace(
        k_class=SimpleNamespace(n_classes=n_classes),
        symmetry=SimpleNamespace(point_group=point_group),
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
    fit = fdp.fit_direction_prior(_masses(3), 3, n_lobes=2, previous=None, seed=0)
    assert fit.start == "cold" and fit.fit_order == 3 and fit.converged
    assert fit.kappa_max == kappa_max_for_grid(768, fdp.CAP_SPACINGS)
    order = np.argsort(-fit.mixture.weights[1:])
    for fitted, true in zip(order, range(2)):
        assert _angle_deg(fit.mixture.means[fitted], TRUTH.means[true]) < 1.0
        assert abs(fit.mixture.kappas[fitted] / TRUTH.kappas[true] - 1.0) < 0.05
    assert abs(fit.mixture.weights[0] - 0.2) < 0.01


def test_warm_fit_starts_from_the_previous_one():
    cold = fdp.fit_direction_prior(_masses(3), 3, n_lobes=2, previous=None, seed=0)
    warm = fdp.fit_direction_prior(_masses(3), 3, n_lobes=2, previous=cold, seed=0)
    assert warm.start == "warm"
    assert warm.em_iterations < cold.em_iterations
    assert warm.objective >= cold.objective - 1e-9


def test_warm_start_with_another_lobe_count_is_refused():
    cold = fdp.fit_direction_prior(_masses(2), 2, n_lobes=2, previous=None, seed=0)
    with pytest.raises(ValueError, match="2 lobes, the prior 3"):
        fdp.fit_direction_prior(_masses(2), 2, n_lobes=3, previous=cold, seed=0)


def test_masses_finer_than_the_fit_order_are_summed_onto_it(monkeypatch):
    monkeypatch.setattr(fdp, "FIT_MAX_ORDER", 2)
    fine = fdp.fit_direction_prior(_masses(4), 4, n_lobes=2, previous=None, seed=0)
    coarse = fdp.fit_direction_prior(_masses(4).reshape(-1, 16).sum(axis=1), 2, n_lobes=2, previous=None, seed=0)
    assert fine.fit_order == 2
    assert_matches(fine.mixture.weights, coarse.mixture.weights)
    assert_matches(fine.mixture.kappas, coarse.mixture.kappas)


@pytest.mark.parametrize(
    "masses, message",
    [(np.ones(47), "C1 directions of order 1"), (np.zeros(48), "not all zero"), (-np.ones(48), "non-negative")],
)
def test_masses_that_are_not_a_direction_histogram_are_refused(masses, message):
    with pytest.raises(ValueError, match=message):
        fdp.fit_direction_prior(masses, 1, n_lobes=1, previous=None, seed=0)


def test_rotation_log_prior_repeats_the_direction_log_prior_over_the_in_plane_angles():
    fit = fdp.fit_direction_prior(_masses(2), 2, n_lobes=2, previous=None, seed=0)
    for order in (1, 2, 3):
        table = fdp.rotation_log_prior(fit, order, dtype=np.float32)
        n_rot, n_directions = rotation_grid_size(order, "C1"), 12 * 4**order
        assert table.shape == (n_rot,) and table.dtype == np.float32
        expected = fdp.direction_log_probabilities(fit, order)[np.arange(n_rot) % n_directions]
        assert_matches(table, expected.astype(np.float32))


# ---- the model STAR blocks ----


def test_fit_round_trips_through_its_star_blocks(tmp_path):
    fit = fdp.fit_direction_prior(_masses(2), 2, n_lobes=2, previous=None, seed=0)
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
    read = fdp.fit_from_star(blocks[fdp.STAR_FIT_BLOCK], columns)
    for name in ("weights", "means", "kappas"):
        assert_matches(getattr(read.mixture, name), getattr(fit.mixture, name))
    for name in ("fit_order", "kappa_max", "start", "em_iterations", "converged", "objective", "seconds"):
        assert getattr(read, name) == getattr(fit, name), name


# ---- the source ----


@pytest.mark.parametrize(
    "replacing, options, flags, message",
    [
        (FvmlDirectionPriorSource(5, dtype=np.float32), _options(), {}, "cannot be combined"),
        (InputSource(), _options(n_classes=2), {}, "K=1"),
        (InputSource(), _options(point_group="C2"), {}, "C1"),
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
    assert [p.fit.start for p in first] == ["cold", "cold"]
    for prior in first:
        assert prior.healpix_order == 2 and prior.values.dtype == np.float32
        assert_matches(prior.values, np.exp(fdp.direction_log_probabilities(prior.fit, 2)).astype(np.float32))
    second = source.learned_direction_priors((None, learned[1]), previous=list(first))
    assert second[0] is None
    assert second[1].fit.start == "warm"


def test_source_scores_with_the_halfs_fit_and_natively_before_it():
    source = FvmlDirectionPriorSource(2, dtype=np.float32)
    own = HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None)
    assert source.direction_log_prior(0, own, prior=DirectionPrior(None, None), healpix_order=2, use_local=False) is own
    fit = fdp.fit_direction_prior(_masses(2), 2, n_lobes=2, previous=None, seed=0)
    for use_local in (False, True):
        log_prior = source.direction_log_prior(
            1, own, prior=DirectionPrior(None, 2, fit), healpix_order=3, use_local=use_local
        )
        assert log_prior.class_rotation_log_prior is None
        assert_matches(log_prior.rotation_log_prior, fdp.rotation_log_prior(fit, 3, dtype=np.float32))
        assert not log_prior.rotation_log_prior.flags.writeable


class _Recording(FvmlDirectionPriorSource):
    fits: list = []

    def learned_direction_priors(self, learned, *, previous):
        priors = super().learned_direction_priors(learned, previous=previous)
        self.fits.append([p.fit for p in priors])
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
