"""A Class3D class that receives no particle weight (relax#68), against RELION's rules.

RELION's ``pdf_class`` of such a class is exactly zero (``maximizationOtherParameters``,
ml_optimiser.cpp:5172-5180); a class at zero is left out of every later expectation
(acc_ml_optimiser_impl.h:1069, :1617, :2378) and its maximization is not run: its
``data_vs_prior`` stays, and its reference is kept in the iteration that emptied it and zeroed
afterwards (ml_optimiser.cpp:4958-5026).
"""

import logging
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import iteration_loop, maximization, numbered_reconstruction, priors, reference_state

pytestmark = pytest.mark.unit


def test_a_class_without_weight_gets_exactly_zero_and_a_minus_infinite_log_prior():
    weights = reference_state._class_weights_from_posterior(
        [np.array([3.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])], 3, np.full(3, 1.0 / 3.0)
    )
    assert_matches(weights, [0.8, 0.0, 0.2], rtol=1e-15)
    assert weights[1] == 0.0  # no floor: the class is empty, not rare
    mixture = reference_state.class_mixture_from_weights(weights)
    assert mixture.log_priors[1] == -np.inf
    assert_matches(mixture.log_priors[[0, 2]], np.log([0.8, 0.2]), rtol=1e-15)
    # An expectation that produced nothing at all keeps the previous weights.
    kept = reference_state._class_weights_from_posterior([np.zeros(3), None], 3, weights)
    assert_matches(kept, weights, rtol=0)


def test_an_empty_class_never_returns():
    """Scored at zero prior a class receives no weight, so its next weight is zero again."""
    scored = np.array([1.0, 0.0])
    assert reference_state.emptied_classes(np.array([0.9, 0.1]), scored) == [1]
    assert reference_state.emptied_classes(scored, scored) == []  # already empty: reported once
    assert reference_state.emptied_classes(np.array([0.5, 0.5]), np.array([0.4, 0.6])) == []


def _settings(calls):
    def reconstruct(denominator, numerator, **kwargs):
        calls.append(int(np.asarray(numerator)[0]))
        return jnp.asarray(numerator, dtype=jnp.float32) * 2.0

    return SimpleNamespace(reconstruct=reconstruct)


def test_a_class_without_weight_is_not_solved_and_keeps_or_zeroes_its_reference():
    calls = []
    numerators = [jnp.full(4, k, dtype=jnp.float32) for k in (1, 2, 3)]
    denominators = [jnp.ones(4, dtype=jnp.float32)] * 3
    kept = jnp.arange(4, dtype=jnp.float32) + 10.0
    stack = numbered_reconstruction._reconstruct_class_maps(
        numerators, denominators, [None] * 3, _settings(calls), n_classes=3, iteration=4, current_size=8,
        accumulator_volume_shape=None, unsolved={0: kept, 2: None},
    )
    assert calls == [2]  # only the class that received weight is solved
    assert_matches(np.asarray(stack[0]), np.asarray(kept), rtol=0)  # emptied this iteration: previous reference
    assert_matches(np.asarray(stack[1]), np.full(4, 4.0), rtol=0)
    assert_matches(np.asarray(stack[2]), np.zeros(4), rtol=0)  # empty before: zero reference
    solved_all = numbered_reconstruction._reconstruct_class_maps(
        numerators, denominators, [None] * 3, _settings([]), n_classes=3, iteration=4, current_size=8,
        accumulator_volume_shape=None,
    )
    assert_matches(np.asarray(solved_all[1]), np.asarray(stack[1]), rtol=0)  # the others are unchanged
    with pytest.raises(RuntimeError, match="every class is empty"):
        numbered_reconstruction._reconstruct_class_maps(
            numerators[:1], denominators[:1], [None], _settings([]), n_classes=1, iteration=4, current_size=8,
            accumulator_volume_shape=None, unsolved={0: None},
        )


@pytest.mark.parametrize("scored_weight, expect_kept", [(0.25, True), (0.0, False)])
def test_class_maximization_freezes_the_curve_of_a_class_without_weight(monkeypatch, scored_weight, expect_kept):
    """The M-step is gated on the weight the expectation scored with: a class that was scored and got nothing
    keeps its reference this once; a class already at zero gets a zero reference. Either way its
    data_vs_prior row is the previous one, which still takes part in the current size (ml_optimiser.cpp:5602-5619)."""
    n_shells = 5
    Ft_y = jnp.asarray(np.arange(8, dtype=np.float32).reshape(2, 4) + 1.0)
    Ft_ctf = jnp.asarray(np.stack([np.ones(4, np.float32), np.zeros(4, np.float32)]))  # class 2 received nothing
    previous_maps = jnp.asarray(np.stack([np.full(4, 5.0, np.float32), np.full(4, 7.0, np.float32)]))
    previous_curve = np.stack([np.full(n_shells, 30.0), np.full(n_shells, 20.0)])
    new_curve = np.stack([np.full(n_shells, 40.0), np.zeros(n_shells)])
    seen = {}

    def estimate_class_priors(*args, **kwargs):
        return priors.ClassPriorAggregation(
            variance="variance", shells=["shells1", "shells2"], data_vs_prior=new_curve.copy(), details_per_class=[]
        )

    def reconstruct(numerators, denominators, shells, settings, **kwargs):
        seen["unsolved"] = kwargs["unsolved"]
        return ["maps", "maps"]

    monkeypatch.setattr(maximization, "estimate_class_priors", estimate_class_priors)
    monkeypatch.setattr(maximization, "reconstruct_numbered_class_maps", reconstruct)
    monkeypatch.setattr(maximization, "_stack_class_tau2_update_details", lambda details: {})
    reference_model = SimpleNamespace(maps=[previous_maps, previous_maps], tau2=None)
    operands = SimpleNamespace(
        numerators=(Ft_y, None), denominators=(Ft_ctf, None), halves=(), noise_stats=(), image_current_size=8,
        accumulator_shape=None, full_half_axis=-1, projector_power_spectrum=None,
    )
    this_iteration = SimpleNamespace(
        iteration=11, current_size=8, first_iteration=SimpleNamespace(relion_firstiter_cc=False)
    )
    options = SimpleNamespace(k_class=SimpleNamespace(n_classes=2), parity=SimpleNamespace())
    ctx = SimpleNamespace(reconstruction_settings=None, scoring_dtype=np.float32, maximization_probe=None)

    mstep = maximization.class_maximization(
        reference_model, operands, ctx, options, this_iteration, class_tau2=SimpleNamespace(source=None),
        scored_class_weights=np.array([1.0 - scored_weight, scored_weight]), previous_data_vs_prior=previous_curve,
    )

    assert list(seen["unsolved"]) == [1]
    if expect_kept:
        assert_matches(np.asarray(seen["unsolved"][1]), np.full(4, 7.0), rtol=0)
    else:
        assert seen["unsolved"][1] is None
    assert_matches(np.asarray(mstep.data_vs_prior)[0], new_curve[0], rtol=0)
    assert_matches(np.asarray(mstep.data_vs_prior)[1], previous_curve[1], rtol=0)


def test_a_jump_in_significant_samples_is_reported_once_per_iteration(caplog):
    history = [np.array([1, 2, 1]), np.array([2, 1, 1])]
    with caplog.at_level(logging.WARNING, logger=iteration_loop.logger.name):
        iteration_loop._warn_if_significant_samples_jumped(history, 1)
        assert caplog.records == []
        history.append(np.array([4000, 1, 1]))
        iteration_loop._warn_if_significant_samples_jumped(history, 2)
    assert len(caplog.records) == 1
    assert "Iteration 3 kept 4002 significant samples, 1000 times" in caplog.records[0].getMessage()
    iteration_loop._warn_if_significant_samples_jumped([None, np.array([5])], 1)  # nothing to compare with


@pytest.mark.parametrize("dead_advantage", [300.0, 1.0e6])
def test_a_dead_class_scored_in_the_coarse_pass_does_not_shift_the_live_classes(monkeypatch, dead_advantage):
    """While an empty class is still scored in pass 1, nothing taken across classes may see it. RELION never
    evaluates it (acc_ml_optimiser_impl.h:1069), so its min_diff2 comes from the live classes. A particle
    that fits the live class poorly has a smaller squared difference against the dead class's zero reference
    (here by hundreds of units and by 1e6): the pre-prior maximum, and with it the float32 frame of the
    weights, the significance cut and the best pose, must equal those of the run without the dead class."""
    from relax.scoring import coarse_publication, pass1_program

    batch, rows, n_trans = 3, 4, 2
    rng = np.random.default_rng(68)
    # Live scores with gaps at the float32 scale of a score near 1e3 (as real -diff2/2 values have).
    live = (-1000.0 - rng.uniform(0.0, 3.0, (batch, rows, n_trans))).astype(np.float32)
    dead = (live + np.float32(dead_advantage)).astype(np.float32)  # every pose of the dead class "fits" better
    tables = {0.0: jnp.asarray(live), 1.0: jnp.asarray(dead)}
    monkeypatch.setattr(
        pass1_program, "relion_coarse_gaussian_gemm_scores_jit",
        lambda reference, *args, **kwargs: tables[float(np.asarray(reference).ravel()[0].real)],
    )

    def block(state, class_index, class_log_prior):
        return pass1_program._pass1_block_update(
            state, jnp.full((rows, 1), class_index, jnp.complex64), jnp.zeros((batch, n_trans, 1), jnp.complex64),
            None, None, batch, (jnp.asarray(class_log_prior, jnp.float32), None), None, class_index, 0,
            rows=rows, block_rows=rows, n_trans=n_trans, image_shape=(4, 4), volume_shape=(4, 4, 4), float64=False,
            score_kind="gaussian", exact_weight_order=False, return_class_best=True, track_class_second=False,
        )

    def initial(n_classes):
        constants = (jnp.full(batch, -jnp.inf), jnp.zeros(batch, jnp.float64), jnp.zeros(batch, jnp.int32))
        return pass1_program.pass1_initial_state(constants, n_classes)

    def class_state(state, k):
        (global_terms, (class_max, class_sum), best, class_poses, raw_max) = state
        return (global_terms, (class_max[k], class_sum[k]), best, tuple(entry[k] for entry in class_poses), raw_max)

    live_state, live_values, _ = block(class_state(initial(2), 0), 0, 0.0)
    dead_in = (live_state[0], (jnp.full(batch, -jnp.inf), jnp.zeros(batch, jnp.float64)), live_state[2],
               tuple(entry for entry in class_state(initial(2), 1)[3]), live_state[4])
    with_dead_state, dead_values, _ = block(dead_in, 1, -np.inf)

    # The pre-prior maximum, the running best pose and the global sums are those of the live class alone.
    assert_matches(np.asarray(with_dead_state[4]), np.asarray(live_state[4]), rtol=0)
    assert_matches(np.asarray(with_dead_state[4]), live.reshape(batch, -1).max(axis=1), rtol=0)
    for a, b in zip(with_dead_state[2], live_state[2]):
        assert_matches(np.asarray(a), np.asarray(b), rtol=0)
    for a, b in zip(with_dead_state[0], live_state[0]):
        assert_matches(np.asarray(a), np.asarray(b), rtol=0)
    assert np.all(np.isneginf(np.asarray(dead_values)))

    def posterior(values, raw_max):
        return coarse_publication.coarse_support_posterior(
            values, raw_max, None, None, exact_weight_order=False, n_trans=n_trans, adaptive_fraction=0.999,
            max_significants=0, tie_score_ulps=0,
        )

    absent = jnp.full_like(live_values, -jnp.inf)
    with_dead = posterior(jnp.concatenate([live_values, dead_values], axis=1), with_dead_state[4])
    without = posterior(jnp.concatenate([live_values, absent], axis=1), live_state[4])
    for name in ("weights", "mask", "n_significant", "cutoff_count", "sum_weight", "winner", "pmax"):
        got, expected = np.asarray(with_dead[name]), np.asarray(without[name])
        assert np.all(np.isfinite(got.astype(np.float64))), name
        assert np.array_equal(got, expected), name  # the same arithmetic on the same operands
    assert np.all(np.asarray(with_dead["sum_weight"]) > 0.0)
    assert not np.any(np.asarray(with_dead["mask"])[:, rows * n_trans:])  # no dead-class sample is significant


def _local_layout(n_translations=5):
    from relax.local_search import layout as local_layout

    rng = np.random.default_rng(680)
    n_rows = 10
    mask = rng.integers(0, 2, (n_rows, n_translations), dtype=np.uint8).view(bool)
    rotations = rng.standard_normal((n_rows, 3, 3)).astype(np.float32)
    return local_layout.LocalHypothesisLayout(
        n_global_rotations=1, n_pixels=1, n_psi=1,
        rotation_offsets=np.array([0, 3, 10]), rotation_counts=np.array([3, 7]),
        rotation_ids_flat=np.arange(n_rows, dtype=np.int64),
        rotations_flat=rotations,
        rotation_log_priors_flat=rng.standard_normal(n_rows).astype(np.float32),
        translation_grid=np.zeros((n_translations, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, n_translations), dtype=np.float32),
        sample_mask_bits=np.packbits(mask, axis=1, bitorder="little"),
    )


_ROW_FIELDS = (
    "rotation_ids_flat", "rotations_flat", "rotation_log_priors_flat", "sample_mask_bits", "row_class_flat",
)


@pytest.mark.parametrize("empty", [(1,), (0, 2)])
def test_a_local_search_builds_no_rows_for_an_empty_class(empty):
    """Class3D local searches (--sigma_ang) score every class at each particle's local orientations, with no
    class prior in the weights (as RELION's). An empty class is therefore removed from the search itself:
    no row is built for it, and the rows of the other classes are those of the full expansion, bit for
    bit, in the same order, so the engine receives for them the operands it would receive without it."""
    from relax.local_search import layout as local_layout

    full = local_layout.expand_local_layout_classes(_local_layout(), 3)
    kept = local_layout.drop_local_layout_classes(full, empty)
    keep = ~np.isin(full.row_class_flat, empty)
    live = 3 - len(empty)
    assert kept.n_classes == 3  # the class axis of the results is unchanged; the class just has no rows
    assert not np.isin(kept.row_class_flat, empty).any()
    assert_matches(kept.rotation_counts, np.array([3, 7]) * live)
    assert_matches(kept.rotation_offsets, np.array([0, 3 * live, 10 * live]))
    for name in _ROW_FIELDS:
        got, expected = np.asarray(getattr(kept, name)), np.asarray(getattr(full, name))[keep]
        assert got.dtype == expected.dtype and np.array_equal(got, expected), name
    assert_matches(kept.sample_mask_rows(), full.sample_mask_rows()[keep])
    # Per image the surviving rows are whole classes, in class order.
    for image in range(2):
        start, stop = kept.rotation_offsets[image:image + 2]
        classes = kept.row_class_flat[start:stop]
        assert sorted(set(classes.tolist())) == [k for k in range(3) if k not in empty]
        assert np.all(np.diff(classes) >= 0)


def test_dropping_no_class_or_every_class_from_a_local_search():
    from relax.local_search import layout as local_layout

    full = local_layout.expand_local_layout_classes(_local_layout(), 2)
    assert local_layout.drop_local_layout_classes(full, ()) is full
    with pytest.raises(ValueError, match="cannot drop classes"):
        local_layout.drop_local_layout_classes(full, (0, 1))
    with pytest.raises(ValueError, match="only a class-expanded layout"):
        local_layout.drop_local_layout_classes(_local_layout(), (0,))


def test_run_files_of_an_emptied_class_read_back_for_a_continued_run():
    """The model STAR holds RELION's joint rows, each class's direction prior times its weight: an empty class
    (weight 0) has a zero row. Reading the files back (--continue) recovers the live classes' conditionals
    and gives the empty class the uniform row, which it never uses; dividing by its zero weight made the
    continued run stop on a non-finite prior."""
    from relax.refinement import run_files
    from relax.sampling import orientation_priors

    rng = np.random.default_rng(681)
    weights = np.array([0.7, 0.0, 0.3])
    conditional = rng.random((3, 48))
    conditional /= conditional.sum(axis=1, keepdims=True)
    joint = conditional * weights[:, None]  # what the writer stores (ml_optimiser.cpp:5325)
    assert np.all(joint[1] == 0.0)

    read = run_files._class_conditionals(joint, weights)

    assert np.all(np.isfinite(read))
    assert_matches(read[[0, 2]], conditional[[0, 2]], rtol=1e-12)
    assert_matches(read[1], np.full(48, 1.0 / 48), rtol=0)
    normalized = orientation_priors.normalize_class_direction_prior(read, 3, dtype=np.float64)
    assert_matches(normalized.sum(axis=1), np.ones(3), rtol=1e-12)
    # No class empty: the plain inverse, unchanged.
    full = np.array([0.5, 0.2, 0.3])
    assert np.array_equal(run_files._class_conditionals(conditional * full[:, None], full), conditional * full[:, None] / full[:, None])
