"""A fixed direction prior as the input source of a K=1 refinement (relax.parity.fixed_direction_prior)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.tiny_main import controller_inputs
from helpers.tiny_refinement import run_tiny_refinement
from recovar.utils.file_hash import sha256_file

from relax.local_search.layout import build_local_adaptive_pass2_hypothesis_layout, build_local_hypothesis_layout
from relax.parity.fixed_direction_prior import PROVENANCE_FILE, FixedDirectionPrior, FixedDirectionPriorSource
from relax.refinement import local_half
from relax.refinement.ports import InputSource
from relax.runtime.engine_tuning import EngineTuning
from relax.sampling.grids import build_local_search_grid_metadata, rotation_grid_n_in_planes, rotation_grid_size
from relax.sampling.healpix import euler_angles_to_matrix
from relax.sampling.orientation_priors import HalfDirectionLogPriors, make_relion_direction_log_prior

pytestmark = pytest.mark.unit


def _prior(order: int, seed: int = 0) -> np.ndarray:
    """A positive, non-uniform probability per direction of HEALPix ``order``."""
    values = np.random.default_rng(seed).uniform(0.5, 2.0, 12 * 4**order)
    return values / values.sum()


def _prior_file(tmp_path, order: int, name: str = "prior.npy", seed: int = 0):
    path = tmp_path / name
    np.save(path, _prior(order, seed))
    return path


def _options(*, n_classes=1, point_group="C1", max_healpix_order=2):
    """The fields ``FixedDirectionPriorSource.for_run`` reads."""
    return SimpleNamespace(
        k_class=SimpleNamespace(n_classes=n_classes),
        symmetry=SimpleNamespace(point_group=point_group),
        schedule=SimpleNamespace(max_healpix_order=max_healpix_order),
        precision=SimpleNamespace(rotation_real_dtype=np.float32),
    )


def _source(tmp_path, order=2) -> FixedDirectionPriorSource:
    return FixedDirectionPriorSource.for_run(
        _prior_file(tmp_path, order), InputSource(), _options(max_healpix_order=order), tomographic=False, multi_gpu=False
    )


# ---- the file ----


def test_prior_file_is_read_with_its_order_and_digest(tmp_path):
    path = _prior_file(tmp_path, 2)
    prior = FixedDirectionPrior.from_file(path)
    assert prior.order == 2
    assert prior.sha256 == sha256_file(path)
    assert prior.path == str(path.resolve())
    assert_matches(prior.values, _prior(2))


def _save(tmp_path, values, name="bad.npy"):
    path = tmp_path / name
    np.save(path, values)
    return path


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: _prior(2).reshape(12, -1), "1-D float"),
        (lambda: np.arange(192), "1-D float"),
        (lambda: np.full(100, 0.01), "Cannot infer healpix order"),
        (lambda: np.r_[0.0, np.full(191, 1.0 / 191)], "finite positive"),
        (lambda: np.r_[np.nan, np.full(191, 1.0 / 191)], "finite positive"),
        (lambda: np.full(192, 1.0 / 190), "not 1"),
    ],
    ids=["two-dimensional", "integers", "not-a-healpix-count", "zero", "nan", "sum"],
)
def test_prior_file_that_is_not_a_prior_is_refused(tmp_path, make, message):
    with pytest.raises(ValueError, match=message):
        FixedDirectionPrior.from_file(_save(tmp_path, make()))


def test_npz_archive_is_refused(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, prior=_prior(2))
    with pytest.raises(ValueError, match="1-D float"):
        FixedDirectionPrior.from_file(path)


def test_nest_children_are_the_directions_of_the_next_coarser_order():
    # The premise of at_order: HEALPix NEST pixel c of order k lies in pixel c // 4 of order k - 1.
    nside = 4
    pixels = np.arange(hp.nside2npix(nside))
    coarse = hp.vec2pix(nside // 2, *hp.pix2vec(nside, pixels, nest=True), nest=True)
    np.testing.assert_array_equal(coarse, pixels // 4)


def test_coarser_order_is_the_exact_sum_of_the_children(tmp_path):
    prior = FixedDirectionPrior.from_file(_prior_file(tmp_path, 3))
    for order in (0, 1, 2):
        factor = 4 ** (3 - order)
        by_hand = np.array([prior.values[p * factor : (p + 1) * factor].sum() for p in range(12 * 4**order)])
        assert_matches(prior.at_order(order), by_hand)
        assert_matches(prior.at_order(order).sum(), 1.0)
    assert_matches(prior.at_order(3), prior.values)


def test_finer_order_than_the_file_is_refused(tmp_path):
    prior = FixedDirectionPrior.from_file(_prior_file(tmp_path, 2))
    with pytest.raises(ValueError, match="finer than"):
        prior.at_order(3)


# ---- admission ----


class _OtherSource(InputSource):
    pass


@pytest.mark.parametrize(
    ("replacing", "options", "flags", "message"),
    [
        (_OtherSource(), _options(), {}, "replay or a frozen boundary"),
        (InputSource(), _options(n_classes=2), {}, "K=1"),
        (InputSource(), _options(point_group="C2"), {}, "C1 symmetry"),
        (InputSource(), _options(), {"tomographic": True}, "single-particle"),
        (InputSource(), _options(), {"multi_gpu": True}, "one GPU"),
        (InputSource(), _options(max_healpix_order=None), {}, "needs --max_healpix_order"),
        (InputSource(), _options(max_healpix_order=3), {}, "needs --max_healpix_order"),
    ],
    ids=["replay", "classes", "symmetry", "tomography", "multi-gpu", "uncapped", "cap-above-the-file"],
)
def test_runs_the_prior_cannot_serve_are_refused(tmp_path, replacing, options, flags, message):
    kwargs = {"tomographic": False, "multi_gpu": False, **flags}
    with pytest.raises(ValueError, match=message):
        FixedDirectionPriorSource.for_run(_prior_file(tmp_path, 2), replacing, options, **kwargs)


def test_admitted_run_scores_in_the_scoring_dtype(tmp_path):
    source = _source(tmp_path)
    assert source.dtype == np.float32
    assert source.prior.order == 2


def test_provenance_names_the_file(tmp_path):
    source = _source(tmp_path)
    record = json.loads(source.write_provenance(tmp_path).read_text())
    assert record == {"path": source.prior.path, "sha256": source.prior.sha256, "healpix_order": 2}
    assert (tmp_path / PROVENANCE_FILE).is_file()


# ---- the log prior the scorers receive ----


@pytest.mark.parametrize("use_local", [False, True])
@pytest.mark.parametrize("half_index", [0, 1])
def test_log_prior_is_the_fixed_prior_on_every_rotation_of_the_grid(tmp_path, half_index, use_local):
    source = _source(tmp_path, order=2)
    own = HalfDirectionLogPriors(rotation_log_prior=np.zeros(3, dtype=np.float32), class_rotation_log_prior=None)
    order = 1
    log_prior = source.direction_log_prior(half_index, own, healpix_order=order, use_local=use_local)
    n_rot, n_directions = rotation_grid_size(order, "C1"), 12 * 4**order
    # Rotation r of the grid lies on direction r % n_directions.
    by_hand = np.log(source.prior.at_order(order))[np.arange(n_rot) % n_directions]
    assert log_prior.class_rotation_log_prior is None
    assert_matches(log_prior.rotation_log_prior, by_hand.astype(np.float32))
    assert_matches(log_prior.rotation_log_prior, make_relion_direction_log_prior(source.prior.at_order(order), order))
    assert not log_prior.rotation_log_prior.flags.writeable


def test_native_source_leaves_the_log_prior_alone():
    own = HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=None)
    assert InputSource().direction_log_prior(0, own, healpix_order=3, use_local=True) is own


# ---- the local search selects with its Gaussian and weights every hypothesis of a rotation by the prior alone ----

ORDER = 1
N_IMAGES = 5
TRANSLATIONS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)


def _eulers():
    rng = np.random.default_rng(7)
    return np.column_stack(
        [rng.uniform(0, 360, N_IMAGES), rng.uniform(20, 160, N_IMAGES), rng.uniform(0, 360, N_IMAGES)]
    )


def _layout(rotation_log_prior):
    return build_local_hypothesis_layout(
        _eulers(), None, 0.35, 0.35, ORDER, TRANSLATIONS, np.zeros((N_IMAGES, 2), dtype=np.float32), 3.0, None, 1.0,
        grid_metadata=build_local_search_grid_metadata(ORDER), translation_prior_reference_translations=TRANSLATIONS,
        rotation_log_prior=rotation_log_prior, dtype=np.float32, sigma_tilt_deg=0.0, engine_tuning=EngineTuning(),
    )


def _table(source, order=ORDER):
    return source.direction_log_prior(
        0, HalfDirectionLogPriors(None, None), healpix_order=order, use_local=True
    ).rotation_log_prior


def _expanded_layout(rotation_log_prior):
    """The local layout at HEALPix order ORDER + 1 whose rows are the children of order-ORDER parents."""
    return build_local_hypothesis_layout(
        _eulers(), None, 0.35, 0.35, ORDER + 1, TRANSLATIONS, np.zeros((N_IMAGES, 2), dtype=np.float32), 3.0, None, 1.0,
        grid_metadata=build_local_search_grid_metadata(ORDER + 1),
        translation_prior_reference_translations=TRANSLATIONS, rotation_log_prior=rotation_log_prior,
        local_parent_oversampling_order=1, dtype=np.float32, sigma_tilt_deg=0.0, engine_tuning=EngineTuning(),
    )


def _within_three_widths(eulers, sigma_rad, order):
    """Per image, the (direction, psi index) pairs of the order-``order`` grid within three widths of its pose.

    Directions are healpy's NEST pixel centres; the width is on the angle between a direction and the pose's, and on
    the wrapped in-plane difference.
    """
    n_psi = rotation_grid_n_in_planes(order)
    pixels = np.arange(hp.nside2npix(2**order))
    directions = np.array(hp.pix2vec(2**order, pixels, nest=True)).T
    psi_grid = np.arange(n_psi) * 360.0 / n_psi
    cutoff = 3.0 * np.rad2deg(sigma_rad)
    pairs = []
    for rot, tilt, psi in eulers:
        centre = euler_angles_to_matrix(np.array([[rot, tilt, psi]]))[0, 2]
        angle = np.rad2deg(np.arccos(np.clip(directions @ centre, -1.0, 1.0)))
        gap = np.abs(psi_grid - psi % 360.0)
        gap = np.minimum(gap, 360.0 - gap)
        pairs.append({(int(d), int(p)) for d in np.flatnonzero(angle < cutoff) for p in np.flatnonzero(gap < cutoff)})
    return pairs


def test_local_layout_weights_each_row_by_the_fixed_prior_alone(tmp_path):
    source = _source(tmp_path, order=ORDER)
    without, with_prior = _layout(None), _layout(_table(source))
    ids = np.asarray(with_prior.rotation_ids_flat)
    by_hand = np.log(source.prior.at_order(ORDER))[ids % (12 * 4**ORDER)].astype(np.float32)
    assert_matches(np.asarray(with_prior.rotation_log_priors_flat), by_hand)
    # The Gaussian log priors of the layout without it are not this table: they are what the prior leaves out.
    assert np.max(np.abs(np.asarray(without.rotation_log_priors_flat) - by_hand)) > 0.1


def test_local_layout_selects_the_orientations_within_three_widths_with_or_without_the_prior(tmp_path):
    with_prior, without = _layout(_table(_source(tmp_path, order=ORDER))), _layout(None)
    np.testing.assert_array_equal(with_prior.rotation_ids_flat, without.rotation_ids_flat)
    np.testing.assert_array_equal(with_prior.rotation_offsets, without.rotation_offsets)
    n_directions = 12 * 4**ORDER
    expected = _within_three_widths(_eulers(), 0.35, ORDER)
    for image, pairs in enumerate(expected):
        ids = np.asarray(with_prior.rotation_ids_flat)[
            with_prior.rotation_offsets[image] : with_prior.rotation_offsets[image + 1]
        ]
        assert {(int(i % n_directions), int(i // n_directions)) for i in ids} == pairs
    # The window trims the grid; a selection of every rotation would make the comparison vacuous.
    assert any(len(pairs) < n_directions * rotation_grid_n_in_planes(ORDER) for pairs in expected)


def test_oversampled_children_carry_their_parents_fixed_prior_alone(tmp_path):
    source = _source(tmp_path, order=ORDER)
    without, with_prior = _expanded_layout(None), _expanded_layout(_table(source))
    ids = np.asarray(with_prior.rotation_ids_flat)
    np.testing.assert_array_equal(ids, without.rotation_ids_flat)
    # A child's direction is NEST pixel p of order ORDER + 1; its parent's is pixel p // 4 of order ORDER.
    parent_direction = (ids % (12 * 4 ** (ORDER + 1))) // 4
    by_hand = np.log(source.prior.at_order(ORDER))[parent_direction].astype(np.float32)
    assert_matches(np.asarray(with_prior.rotation_log_priors_flat), by_hand)


def test_adaptive_pass2_children_inherit_their_parent_prior(tmp_path):
    parents = _layout(_table(_source(tmp_path, order=ORDER)))
    children = build_local_adaptive_pass2_hypothesis_layout(
        parents, [None] * N_IMAGES, ORDER, oversampling_order=1, random_perturbation=0.0, dtype=np.float32
    )
    for image in range(N_IMAGES):
        parent = np.asarray(parents.rotation_log_priors_flat)[
            parents.rotation_offsets[image] : parents.rotation_offsets[image + 1]
        ]
        child = np.asarray(children.rotation_log_priors_flat)[
            children.rotation_offsets[image] : children.rotation_offsets[image + 1]
        ]
        assert_matches(np.unique(child), np.unique(parent))


def test_local_half_builds_its_parent_layout_with_the_prior(tmp_path):
    source = _source(tmp_path, order=ORDER)

    def parent_layout(table):
        half = SimpleNamespace(
            particles=SimpleNamespace(rotation_eulers=_eulers(), dataset=SimpleNamespace(voxel_size=1.0))
        )
        sampling = SimpleNamespace(
            search=SimpleNamespace(parent_order=ORDER, sigma_rot=0.35, sigma_psi=0.35, symmetry="C1"),
            translations=TRANSLATIONS,
            perturbation=0.0,
        )
        priors = local_half.LocalPriorSpec(
            trans_prior_center=np.zeros((N_IMAGES, 2), dtype=np.float32),
            trans_prior_center_for_engine=None,
            current_sigma_offset_angstrom=3.0,
            translation_search_base=None,
            direction_log_prior=table,
        )
        layout, order = local_half._build_local_adaptive_parent_layout(
            half, sampling, priors, TRANSLATIONS, np.float32, EngineTuning()
        )
        assert order == ORDER
        return layout

    without, with_prior = parent_layout(None), parent_layout(_table(source))
    ids = np.asarray(without.rotation_ids_flat)
    np.testing.assert_array_equal(with_prior.rotation_ids_flat, ids)
    by_hand = np.log(source.prior.at_order(ORDER))[ids % (12 * 4**ORDER)].astype(np.float32)
    assert_matches(np.asarray(with_prior.rotation_log_priors_flat), by_hand)
    assert np.ptp(by_hand) > 0.0


# ---- the controller ----


class _Recording(FixedDirectionPriorSource):
    seen: list = []

    def direction_log_prior(self, half_index, log_prior, *, healpix_order, use_local):
        self.seen.append((half_index, healpix_order, use_local))
        return super().direction_log_prior(half_index, log_prior, healpix_order=healpix_order, use_local=use_local)


def test_every_expectation_of_a_run_scores_with_the_fixed_prior(monkeypatch, tmp_path):
    """Two numbered iterations and the final pass, both halves: each asks the source and scores with its prior."""
    path = _prior_file(tmp_path, 2)
    native_calls, fixed_calls = [], []
    run_tiny_refinement(monkeypatch, max_iter=2, engine_calls=native_calls)
    _Recording.seen = []
    run_tiny_refinement(
        monkeypatch, max_iter=2, engine_calls=fixed_calls,
        source_for=lambda options: _Recording.for_run(path, InputSource(), options, tomographic=False, multi_gpu=False),
    )
    assert len(fixed_calls) == len(native_calls) == len(_Recording.seen) == 6
    assert _Recording.seen == [(half, 2, False) for _ in range(3) for half in (0, 1)]
    expected = make_relion_direction_log_prior(FixedDirectionPrior.from_file(path).at_order(2), 2)
    for call in fixed_calls:
        assert_matches(np.asarray(call["kwargs"]["rotation_log_prior"]), expected)
    # The first expectation of a native run has no learned prior to score with.
    assert native_calls[0]["kwargs"]["rotation_log_prior"] is None


# ---- the command ----


def test_flag_selects_the_fixed_prior_source_and_records_the_file(monkeypatch, tmp_path):
    path = _prior_file(tmp_path, 2)
    inputs = controller_inputs(
        monkeypatch, tmp_path, "refine", "--max_healpix_order", "2", "--fixed-direction-prior", path
    )
    assert type(inputs["source"]) is FixedDirectionPriorSource
    record = json.loads((tmp_path / "out" / PROVENANCE_FILE).read_text())
    assert record["sha256"] == sha256_file(path)


def test_flag_without_a_capped_order_is_refused(monkeypatch, tmp_path):
    path = _prior_file(tmp_path, 2)
    with pytest.raises(ValueError, match="needs --max_healpix_order"):
        controller_inputs(monkeypatch, tmp_path, "refine", "--fixed-direction-prior", path)


def test_native_command_has_the_native_source(monkeypatch, tmp_path):
    assert type(controller_inputs(monkeypatch, tmp_path, "refine")["source"]) is InputSource
