"""Zero oversampling (``--adaptive_oversampling 0``) with more than one class: RELION's arithmetic (relax#73).

RELION's accelerated expectation runs two passes at every oversampling (``nr_sampling_passes = 2``,
acc_ml_optimiser_impl.h:3787 of the page build). At oversampling 0 the second pass

* does not reset the first pass's sum and maximum (``if (exp_ipass==0 || adaptive_oversampling!=0)``, :1984-1988);
  the first pass set them over every sample of every class with ``pdf_class > 0`` (``op.sum_weight``, :2274;
  ``op.max_weight``, :2314),
* keeps every sample it is given (``significant_weight = sorted[0]``, :2532),
* divides the fine weights, shifted by their own maximum, by that kept sum in the weighted sums (:2784-2785),
* adds those weights to each class's ``wsum pdf_class`` (:2843), so a particle's class shares sum to the
  retained share of the first pass's sum, not to 1, and
* reports ``Pmax = op.max_weight / op.sum_weight`` (:2924), the first pass's.

relax does this for one class and, on the caller's request, for several
(``reuse_zero_oversampling_coarse_state``, docs/math/zero_oversampling.md). The reference below is written from
those lines, not from relax's one-class code.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp
from helpers.engine_settings import RESIDENT_PRODUCTION_ENGINE
from helpers.float_compare import assert_matches

from relax.runtime.engine_tuning import EngineTuning
from relax.runtime.scoring_overrides import ScoringOverrides

pytestmark = pytest.mark.unit

# The keyword the Class3D and InitialModel callers pass to take RELION's zero-oversampling route with K>1 (relax#73).
CLASS_REUSE_KEYWORD = "zero_oversampling_coarse_state_for_classes"


def _relion_second_pass_at_oversampling_zero(fine_scores, class_ids, coarse_sum_weight, n_classes):
    """RELION's second pass at oversampling 0 for one particle, in float32 as its GPU code.

    ``fine_scores`` are the log weights (with priors) of the samples the first pass kept, over every class;
    ``coarse_sum_weight`` is the first pass's ``op.sum_weight`` in its own maximum-shifted convention. Returns
    each kept sample's weight in the weighted sums and each class's share of the particle.
    """

    scores = np.asarray(fine_scores, dtype=np.float32)
    # expf(score + (50 - max)): the shift is the second pass's own (convertAllSquaredDifferencesToWeights).
    raw = np.exp((scores + (np.float32(50.0) - scores.max())).astype(np.float32)).astype(np.float32)
    weights = (raw / np.float32(coarse_sum_weight)).astype(np.float32)  # :2784-2785, the kept sum; no cut (:2532)
    shares = np.asarray([weights[np.asarray(class_ids) == k].sum(dtype=np.float64) for k in range(n_classes)])
    return weights, shares


def _particle(seed, n_classes, n_kept):
    """Kept samples of one particle: scores spread over a few units, classes mixed."""

    rng = np.random.default_rng(seed)
    scores = (-1000.0 - 6.0 * rng.random(n_kept)).astype(np.float32)
    class_ids = rng.integers(0, n_classes, size=n_kept)
    # The first pass's sum in its own convention: above the kept raw weights' sum, since it also holds the
    # samples the first pass cut (the last 0.1% of the mass) and its maximum-shift differs.
    raw = np.exp((scores + (np.float32(50.0) - scores.max())).astype(np.float32)).astype(np.float64)
    coarse_sum = np.float32(raw.sum() * (1.0 + 1.0e-3 * rng.random()))
    return scores, class_ids, coarse_sum


@pytest.mark.parametrize("n_classes", [1, 2, 4])
def test_fine_weights_divide_by_the_kept_coarse_sum_for_any_number_of_classes(n_classes):
    """The float32 primitive, given a particle's kept samples over all classes and the first pass's sum, gives
    RELION's second-pass weights: no cut, no renormalisation, class shares summing to the retained share."""

    from relax.fine_pass.posterior import relion_f32_fine_probabilities

    for seed in range(4):
        scores, class_ids, coarse_sum = _particle(seed, n_classes, n_kept=37)
        expected, shares = _relion_second_pass_at_oversampling_zero(scores, class_ids, coarse_sum, n_classes)
        _normalized, kept_weights, mask, n_kept, sum_weight, _threshold = relion_f32_fine_probabilities(
            scores[None],
            adaptive_fraction=0.999,
            normalization_sum_weight=np.asarray([coarse_sum], dtype=np.float32),
            keep_all=True,
        )
        assert int(n_kept[0]) == scores.size and bool(np.all(np.asarray(mask)))
        assert_matches(np.asarray(sum_weight), np.asarray([coarse_sum], dtype=np.float32))
        assert_matches(np.asarray(kept_weights)[0], expected)
        retained = float(np.asarray(kept_weights, dtype=np.float64).sum())
        assert_matches(np.float32(shares.sum()), np.float32(retained))
        # The retained share is what the first pass's cut left: below 1, and not renormalised to it.
        assert 0.998 < retained < 1.0


@pytest.mark.parametrize("n_classes,passes_keyword,enabled", [
    (1, False, True),  # Refine3D: the one-class route, as today
    (4, False, False),  # a caller that does not ask: pass 2 normalizes on the retained samples
    (4, True, True),  # Class3D and InitialModel ask for RELION's route (relax#73)
])
def test_adaptive_engine_hands_the_coarse_state_to_pass_2_for_classes_on_request(
    monkeypatch, n_classes, passes_keyword, enabled,
):
    """At oversampling 0 with the soft Gaussian score, pass 2 receives pass 1's sum, Pmax and winner."""

    import relax.classification.k_class as k_class_module
    from relax.classification.k_class import run_dense_k_class_em_adaptive
    from relax.classification.k_class_results import assemble_result
    from relax.scoring import significance
    from relax.scoring.pass1_results import Pass1Result, Pass1Stats
    from relax.types import make_relion_stats

    coarse_calls, fine_calls = [], []
    coarse_sum = np.asarray([123.0], dtype=np.float32)
    coarse_pmax = np.asarray([0.123], dtype=np.float32)
    coarse_pose = np.asarray([0], dtype=np.int32)

    def coarse(*args, **kwargs):
        coarse_calls.append(kwargs)
        return Pass1Result(
            None, np.ones(1, dtype=np.int32), coarse_pose, np.zeros(1, dtype=np.int32),
            [[np.asarray([0], dtype=np.int32)] for _ in range(n_classes)],
            Pass1Stats(**{name: None for name in Pass1Stats.__dataclass_fields__}
                       | {"significant_cutoff_counts": np.ones(1, dtype=np.int32),
                          "relion_f32_sum_weight": coarse_sum, "relion_f32_max_posterior": coarse_pmax}),
        )

    stats = make_relion_stats(
        log_evidence_per_image=np.zeros(1, dtype=np.float32), best_log_score_per_image=np.zeros(1, dtype=np.float32),
        max_posterior_per_image=np.ones(1, dtype=np.float32), rotation_posterior_sums=np.zeros(1, dtype=np.float32),
    )
    result = assemble_result(
        class_log_evidence=np.zeros((n_classes, 1)),
        Ft_y=[jnp.ones(4, dtype=jnp.complex64)] * n_classes, Ft_ctf=[jnp.ones(4, dtype=jnp.float32)] * n_classes,
        per_class_hard_assignments=np.zeros((n_classes, 1), dtype=np.int32),
        per_class_stats=tuple(stats for _ in range(n_classes)), noise_stats=None,
    )

    def fine(*args, **kwargs):
        fine_calls.append(kwargs)
        return result

    monkeypatch.setattr(significance, "compute_k_class_significance_batched", coarse)
    monkeypatch.setattr(k_class_module, "run_sparse_k_class_adaptive_pass2", fine)
    run_dense_k_class_em_adaptive(
        SimpleNamespace(n_images=1), jnp.zeros((n_classes, 4), dtype=jnp.complex64),
        jnp.ones(4), jnp.ones(1), np.eye(3)[None], np.zeros((1, 2)),
        np.eye(3)[None], np.zeros((1, 2)), np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64),
        "linear_interp", coarse_healpix_order=0, oversampling_order=0, sparse_pass2=True,
        mstep_relion_x_half=True, relion_firstiter_score_mode="gaussian",
        relion_firstiter_winner_take_all=False, use_float64_scoring=False,
        **({CLASS_REUSE_KEYWORD: True} if passes_keyword else {}),
        engine_tuning=EngineTuning(),
        scoring_overrides=ScoringOverrides(),
    )
    assert len(coarse_calls) == len(fine_calls) == 1
    assert coarse_calls[0].get("return_relion_f32_normalization", False) is enabled
    fine_kwargs = fine_calls[0]["engine_kwargs"]
    for key, value in (("relion_f32_normalization_sum_weight", coarse_sum),
                       ("relion_coarse_hard_assignment", coarse_pose),
                       ("relion_coarse_max_posterior", coarse_pmax)):
        if enabled:
            assert_matches(fine_kwargs[key], value)
        else:
            assert key not in fine_kwargs
    # The winner's class goes with it when there are classes.
    if enabled and n_classes > 1:
        assert_matches(fine_kwargs["relion_coarse_class_assignment"], np.zeros(1, dtype=np.int32))
    else:
        assert "relion_coarse_class_assignment" not in fine_kwargs


# ---------------------------------------------------------------------------
# GPU: the resident K-class pass with the kept coarse state
# ---------------------------------------------------------------------------


def _resident_gpu_available() -> bool:
    from test_resident_pass2_driver import _gpu_available

    return _gpu_available()


requires_resident_gpu = pytest.mark.skipif(not _resident_gpu_available(), reason="the resident stages are CUDA FFI targets")


def _rel_l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(b))


@requires_resident_gpu
def test_duplicated_class_at_zero_oversampling_is_the_one_class_pass_split_in_half():
    """Two copies of one class at prior 1/2 each, with RELION's kept coarse state.

    By the lines in the module docstring: the class prior shifts every score and the second pass's own maximum
    alike, so each duplicated sample's raw weight equals the one-class raw weight; the first pass's sum runs
    over twice the samples and is twice the one-class sum, and its maximum is the same, so Pmax is half. Each
    class's weighted sums are then half the one-class sums, each class's share half the one-class retained
    share (the two sum to the retained share, not to 1), the winner is the first pass's, and the noise sums
    are the one-class ones. The one-class numbers come from the resident K=1 pass.
    """

    from test_resident_k_class_pass2 import _resident
    from test_resident_zero_oversampling import _os0_driver_args

    from relax.fine_pass import resident_pass2 as rp

    args = _os0_driver_args()
    single = rp.compute_pass2_stats_resident(**args, engine_tuning=RESIDENT_PRODUCTION_ENGINE, scoring_overrides=ScoringOverrides())
    k_args = dict(args)
    support = k_args.pop("significant_sample_indices")
    prior = k_args.pop("rotation_log_prior")
    volume = k_args.pop("volume")
    k_args.pop("normalization_other_score_log_z", None)
    k_args["relion_f32_normalization_sum_weight"] = 2.0 * np.asarray(args["relion_f32_normalization_sum_weight"])
    k_args["relion_coarse_max_posterior"] = 0.5 * np.asarray(args["relion_coarse_max_posterior"])
    n_images = int(args["experiment_dataset"].n_units)
    # Equal copies tie; RELION's first maximum is in the first class of its class-major weights.
    coarse_classes = np.zeros(n_images, dtype=np.int32)
    coarse_classes[1::2] = 1  # either copy is a valid winner for the pass: the class it is handed is kept
    k_args["relion_coarse_class_assignment"] = coarse_classes
    half = (prior + np.float32(np.log(0.5))).astype(np.float32)
    doubled = _resident(k_args, jnp.stack([volume, volume]), [support, support], [half, half])

    retained = float(single.noise_stats.sumw)
    # The winner is the first pass's: its class, and in that class the one-class pass's (coarse) pose.
    assert np.array_equal(np.asarray(doubled.coarse_class_assignments), coarse_classes)
    images = np.arange(n_images)
    n_fine_trans = int(np.asarray(args["fine_translations_override"]).shape[0])
    assert np.array_equal(
        np.asarray(doubled.per_class_hard_assignments)[coarse_classes, images],
        np.asarray(single.best_rotation_indices, dtype=np.int64) * n_fine_trans
        + np.asarray(single.hard_assignment, dtype=np.int64) % n_fine_trans,
    )
    # The coarse sums of the fixture are arbitrary, so the retained share is not near the image count.
    assert abs(retained - n_images) > 1e-3 * n_images
    for k in range(2):
        assert _rel_l2(0.5 * np.asarray(single.Ft_y), doubled.Ft_y[k]) < 1e-6, f"Ft_y class {k}"
        assert _rel_l2(0.5 * np.asarray(single.Ft_ctf), doubled.Ft_ctf[k]) < 1e-6, f"Ft_ctf class {k}"
    assert_matches(np.float32(doubled.class_reconstruction_posterior_sums), np.float32(np.full(2, 0.5 * retained)))
    assert_matches(
        np.asarray(doubled.stats.max_posterior_per_image), 0.5 * np.asarray(args["relion_coarse_max_posterior"]),
    )
    assert abs(retained - float(doubled.noise_stats.sumw)) <= 1e-6 * retained
