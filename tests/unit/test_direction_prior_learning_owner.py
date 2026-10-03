"""Prior learning preserves numerical values, half identity and history copies."""

from __future__ import annotations

import logging
from functools import partial

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers import orientation_priors as op
from relax.helpers.iteration_history import RefinementHistory
from relax.sampling import rotation_grid_size

pytestmark = pytest.mark.unit
ORDER = 1
N_ROT = rotation_grid_size(ORDER)
learn_k1 = partial(
    op.learn_k1_direction_priors,
    direction_prior_order=ORDER,
    expected_rotation_count=N_ROT,
    dtype=np.float32,
    log=logging.getLogger(__name__),
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_k1_learns_one_prior_per_half_at_the_scoring_order(dtype):
    rng = np.random.default_rng(0)
    posteriors = [rng.uniform(0.0, 1.0, N_ROT).astype(np.float32) for _ in range(2)]
    learned = learn_k1(posteriors, dtype=dtype)
    for posterior, prior in zip(posteriors, learned, strict=True):
        expected = op.collapse_rotation_posterior_to_direction_prior(
            np.asarray(posterior, dtype=np.float64), ORDER, dtype=dtype,
        )
        assert prior.values.dtype == dtype
        assert_matches(prior.values, expected, strict=True)
        assert prior.healpix_order == ORDER


def test_k1_size_mismatch_rejects_the_update():
    posteriors = [np.ones(N_ROT, dtype=np.float32), np.ones(N_ROT + 1, dtype=np.float32)]
    assert learn_k1(posteriors) == (None, None)


def test_k1_skips_a_half_whose_prior_cannot_form_a_log_prior(monkeypatch, caplog):
    calls = []

    def failing_log_prior(direction_prior, healpix_order, *, symmetry="C1"):
        calls.append(int(healpix_order))
        if len(calls) == 1:
            raise ValueError("bad support")

    monkeypatch.setattr(op, "make_relion_direction_log_prior", failing_log_prior)
    posteriors = [np.ones(N_ROT, dtype=np.float32), np.ones(N_ROT, dtype=np.float32)]
    with caplog.at_level(logging.WARNING, logger=__name__):
        learned = learn_k1(posteriors)
    assert calls == [ORDER, ORDER]
    assert learned[0] is None
    assert learned[1].healpix_order == ORDER
    assert "Skipping K=1 direction prior update for half-1 at healpix_order=1: bad support" in caplog.text


def test_kclass_combines_halves_with_independent_copies():
    rng = np.random.default_rng(1)
    n_classes = 2
    posteriors = [rng.uniform(0.0, 1.0, (n_classes, N_ROT)) for _ in range(2)]
    learned = op.learn_class_direction_priors(
        posteriors, n_classes=n_classes, healpix_order=ORDER, dtype=np.float32,
    )
    expected = op._combined_class_direction_prior_from_halves(
        posteriors, n_classes, ORDER, dtype=np.float32,
    )
    for prior in learned:
        assert prior.values.shape == expected.shape and prior.values.dtype == np.float32
        assert_matches(prior.values, expected, strict=True)
        assert prior.healpix_order == ORDER
    assert learned[0].values is not learned[1].values
    assert not np.shares_memory(learned[0].values, learned[1].values)


def test_history_records_float64_copies_of_k1_priors_and_none_for_missing():
    values = np.asarray([0.25, 0.75], dtype=np.float32)
    priors = [op.HalfDirectionPriors(shared=op.DirectionPrior(values, 0)), op.HalfDirectionPriors()]
    history = RefinementHistory()
    history.record_direction_prior(priors, k_class_enabled=False)
    stored = history.direction_prior_trajectory_per_half[0]
    assert stored[1] is None
    assert stored[0].dtype == np.float64 and stored[0].tolist() == [0.25, 0.75]
    assert not np.shares_memory(stored[0], values)


def test_history_records_class_zero_of_each_half_for_kclass():
    values = np.asarray([[0.1, 0.9], [0.6, 0.4]], dtype=np.float64)
    priors = [op.HalfDirectionPriors(classes=op.DirectionPrior(values, 0)),
              op.HalfDirectionPriors(classes=op.DirectionPrior(values.copy(), 0))]
    history = RefinementHistory()
    history.record_direction_prior(priors, k_class_enabled=True)
    stored = history.direction_prior_trajectory_per_half[0]
    assert [s.tolist() for s in stored] == [[0.1, 0.9], [0.1, 0.9]]
    assert all(s.dtype == np.float64 and not np.shares_memory(s, values) for s in stored)


def test_history_records_float64_rotation_posterior_copies():
    posterior = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    history = RefinementHistory()
    history.record_rotation_posterior([posterior, None])
    stored = history.rotation_posterior_trajectory_per_half[0]
    assert stored[1] is None
    assert stored[0].dtype == np.float64
    assert_matches(stored[0], posterior)
    assert not np.shares_memory(stored[0], posterior)
