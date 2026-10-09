"""Pass 1's run stages, one at a time: the batch steps, the publish, the order of the loop and the typed result.

Runs on the CPU exact-operand harness (``helpers.exact_pass1_harness``): seven images in batches of three, so the last
batch holds one image and is repeat-padded to three rows.
"""

import dataclasses
import logging

import jax
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_em_stage_glue_programs import _assert_significance_results_match, _significance_call

from relax.scoring import pass1_publish, significance
from relax.scoring.pass1_assembly import log_batch_timing
from relax.scoring.pass1_plan import plan_pass1
from relax.scoring.pass1_publish import publish_batch
from relax.scoring.pass1_request import Pass1Request
from relax.scoring.pass1_results import Pass1Outputs, Pass1Result, Pass1Stats
from relax.scoring.pass1_step import PreparedBatch, ScoredBatch, prepare_batch, score_batch
from relax.scoring.tree_rescore import TreeRescoreBatch, TreeRescoreTotals, log_tree_rescore_totals

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _cpu_stand_ins_only():
    """The stages run on the harness's CPU stand-ins; on a GPU backend they are not what is being qualified."""

    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only contract: the stage tests run on the exact-operand harness's CPU stand-ins")

TAIL_START = 6  # images 0-2 and 3-5 fill two batches; image 6 is the tail


@pytest.fixture
def make(monkeypatch):
    """``make(n_classes, **request_overrides)`` -> ``(plan, args, kwargs)`` of a planned pass on the harness."""

    monkeypatch.setenv("RELAX_EM_PREFETCH_BATCHES", "0")

    def build(n_classes=2, **overrides):
        args, kwargs = _significance_call(monkeypatch, n_classes=n_classes)
        kwargs.update(overrides)
        return plan_pass1(Pass1Request(*args, **kwargs)), args, kwargs

    return build


def _batch(plan, args, which):
    """``(batch_data, indices)`` of batch number ``which`` of the pass."""

    batch_data, *_, indices = list(args[0].iter_batches(plan.image_batch_size))[which]
    return batch_data, indices


def _prepared(plan, args, which=2):
    batch_data, indices = _batch(plan, args, which)
    start = which * plan.image_batch_size
    return prepare_batch(plan, batch_data, indices, start, start + len(indices))


# ------------------------------------------------------------------------------------------- prepare_batch


def test_a_prepared_batch_has_its_operands_on_the_device_at_the_program_dtypes(make):
    plan, args, _ = make()
    batch = _prepared(plan, args, which=0)
    assert isinstance(batch, PreparedBatch)
    assert (batch.start_idx, batch.end_idx) == (0, 3) and len(batch.indices) == 3
    shifted, pixel_weight, initial_diff2 = batch.program_inputs
    # [B rows, T translations, S scored rows] complex64, then float32 weights and the float32 high-resolution power.
    assert shifted.shape == (3, 3, 12) and shifted.dtype == np.complex64
    assert pixel_weight.shape == (3, 12) and pixel_weight.dtype == np.float32
    assert initial_diff2.shape == (3,) and initial_diff2.dtype == np.float32
    assert batch.operands.corr_img is None
    assert batch.dump_targets.enabled is False and batch.dump_rows is None


def test_the_tail_batch_is_repeat_padded_to_the_full_extent_but_keeps_its_one_image(make):
    plan, args, _ = make()
    batch = _prepared(plan, args)
    assert (batch.start_idx, batch.end_idx) == (TAIL_START, TAIL_START + 1)
    assert len(batch.indices) == 1
    assert batch.inputs.batch_size == 3
    assert batch.operands.shifted.shape[0] == 3


def test_without_padding_the_tail_batch_has_its_own_extent(make, monkeypatch):
    monkeypatch.setenv("RELAX_COARSE_PAD_FINAL_IMAGE_BATCH", "0")
    plan, args, _ = make()
    batch = _prepared(plan, args)
    assert batch.inputs.batch_size == 1 and batch.operands.shifted.shape[0] == 1


# --------------------------------------------------------------------------------------------- score_batch


def test_a_scored_batch_counts_its_real_rows_and_the_padded_extent(make):
    plan, args, _ = make()
    scored = score_batch(plan, _prepared(plan, args), defer_publish=False)
    assert isinstance(scored, ScoredBatch) and scored.rescored is None
    outputs = scored.outputs
    assert (outputs.actual_batch_size, outputs.batch_size) == (1, 3)
    assert (outputs.start_idx, outputs.end_idx) == (TAIL_START, TAIL_START + 1)
    assert outputs.sig_mask.shape == (3, 2 * 5 * 3)  # [B, K * R * T], class-major
    assert len(outputs.class_log_z_values) == 2
    assert outputs.class_best_scores is not None and outputs.class_second_best_scores is None


def test_a_batch_that_waits_for_its_readback_holds_none_of_its_large_arrays(make):
    plan, args, _ = make()
    prepared = _prepared(plan, args)
    now = score_batch(plan, prepared, defer_publish=False).outputs
    waiting = score_batch(plan, prepared, defer_publish=True).outputs
    for name in (
        "weights",
        "significant_weight",
        "translation_log_prior",
        "operands",
        "dump_target_pre_prior_blocks_per_class",
        "dump_target_with_prior_blocks_per_class",
    ):
        assert getattr(waiting, name) is None, name
    for name in ("weights", "significant_weight", "operands"):
        assert getattr(now, name) is not None, name
    # What publishing needs from a waiting batch is kept.
    for name in ("sig_mask", "sig_rot_mask", "n_sig", "cutoff_count", "sum_weight", "pmax", "best_argmax", "best_class",
                 "best_score", "global_log_z", "class_log_z_values", "class_best_scores", "class_best_argmaxes"):
        assert getattr(waiting, name) is not None, name


def test_the_float32_route_publishes_the_winner_of_its_weights_for_one_class(make):
    plan, args, _ = make(n_classes=1)
    outputs = score_batch(plan, _prepared(plan, args), defer_publish=False).outputs
    assert plan.support_plan.exact_weight_order is True
    assert_matches(np.asarray(outputs.best_argmax), np.argmax(np.asarray(outputs.weights), axis=1))


def test_the_generic_route_has_the_posterior_weights_but_none_of_the_float32_sums(make, monkeypatch):
    monkeypatch.setenv("RECOVAR_K1_RELION_F32_COARSE_SUPPORT", "0")
    plan, args, _ = make()
    outputs = score_batch(plan, _prepared(plan, args), defer_publish=False).outputs
    assert outputs.weights is not None and outputs.sig_mask is not None
    assert outputs.pmax is None and outputs.sum_weight is None and outputs.significant_weight is None
    # The generic posterior is exp(score - log Z) over every pose of the row.
    assert_matches(np.asarray(outputs.weights).sum(axis=1), np.ones(3), rtol=1e-5)


def test_a_pass_that_collects_no_support_scores_without_a_support(make):
    plan, args, _ = make(collect_significance=False)
    outputs = score_batch(plan, _prepared(plan, args), defer_publish=False).outputs
    assert outputs.sig_mask is None and outputs.weights is None and outputs.n_sig is None
    assert outputs.global_log_z is not None and outputs.best_score is not None


# -------------------------------------------------------------------------------------------- publish_batch


def _published(plan, args, *, defer=False, **replaced):
    """The tail batch scored and published into fresh sentinel-filled results; ``replaced`` overrides its output fields."""

    prepared = _prepared(plan, args)
    outputs = dataclasses.replace(score_batch(plan, prepared, defer_publish=defer).outputs, **replaced)
    results = Pass1Outputs.allocate(plan.output_plan)
    results.hard_assignment[:] = -999
    results.class_assignment[:] = -999
    results.n_sig_all[:] = -999
    results.cutoff_count_all[:] = -999
    results.normalization_log_z[:] = np.nan
    publish_batch(outputs, results, plan.output_plan, plan.dump_context)
    return outputs, results


def test_publishing_a_batch_writes_only_its_own_rows_of_the_per_image_results(make):
    plan, args, _ = make()
    outputs, results = _published(plan, args)
    for name in ("hard_assignment", "class_assignment", "n_sig_all", "cutoff_count_all"):
        assert (getattr(results, name)[:TAIL_START] == -999).all(), name
    assert np.isnan(results.normalization_log_z[:TAIL_START]).all()
    row = TAIL_START
    assert results.hard_assignment[row] == int(np.asarray(outputs.best_argmax)[0])
    assert results.class_assignment[row] == int(np.asarray(outputs.best_class)[0])
    assert results.n_sig_all[row] == int(np.asarray(outputs.n_sig)[0]) > 0
    assert results.cutoff_count_all[row] == int(np.asarray(outputs.cutoff_count)[0])


def test_publishing_stores_the_log_evidence_and_the_class_evidence_of_the_real_row(make):
    plan, args, _ = make()
    outputs, results = _published(plan, args)
    log_z = float(np.asarray(outputs.global_log_z)[0])
    assert results.normalization_log_z[TAIL_START] == pytest.approx(log_z)
    # No image-energy offset on the exact scorers: the evidence is the normalizer.
    assert results.normalization_log_evidence[TAIL_START] == pytest.approx(log_z)
    assert results.log_evidence[TAIL_START] == pytest.approx(log_z, rel=1e-6)
    for class_index, class_log_z in enumerate(outputs.class_log_z_values):
        assert results.class_log_evidence[class_index, TAIL_START] == pytest.approx(float(np.asarray(class_log_z)[0]))


def test_the_float32_route_publishes_its_row_maximum_as_the_maximum_posterior(make):
    """Distinctive support values make the route visible: the maximum posterior is the support's, not a recomputation."""

    plan, args, _ = make()
    _, results = _published(plan, args, pmax=np.full(3, 0.123, dtype=np.float32), sum_weight=np.full(3, 4.5, dtype=np.float32))
    assert results.max_posterior[TAIL_START] == np.float32(0.123)
    assert results.relion_f32_sum_weight[TAIL_START] == np.float32(4.5)
    assert results.relion_f32_max_posterior is None


def test_asking_for_the_normalization_also_stores_the_row_maximum_as_float32(make):
    plan, args, _ = make(return_relion_f32_normalization=True)
    _, results = _published(plan, args, pmax=np.full(3, 0.25, dtype=np.float32))
    assert results.relion_f32_max_posterior.dtype == np.float32
    assert results.relion_f32_max_posterior[TAIL_START] == np.float32(0.25)


def test_the_generic_route_derives_the_maximum_posterior_from_the_best_score(make, monkeypatch):
    monkeypatch.setenv("RECOVAR_K1_RELION_F32_COARSE_SUPPORT", "0")
    plan, args, _ = make()
    outputs, results = _published(plan, args)
    expected = np.exp(float(np.asarray(outputs.best_score)[0]) - float(np.asarray(outputs.global_log_z)[0]))
    assert results.max_posterior[TAIL_START] == pytest.approx(expected, rel=1e-6)
    assert results.relion_f32_sum_weight is None


def test_the_class_winners_are_published_for_each_class_and_the_runner_up_when_asked(make):
    plan, args, _ = make(return_class_second=True)
    outputs, results = _published(plan, args)
    for class_index in range(2):
        assert results.class_best_log_score[class_index, TAIL_START] == np.asarray(
            outputs.class_best_scores[class_index], dtype=np.float32
        )[0]
        assert results.class_hard_assignment[class_index, TAIL_START] == int(
            np.asarray(outputs.class_best_argmaxes[class_index])[0]
        )
        assert results.class_second_best_log_score[class_index, TAIL_START] == np.asarray(
            outputs.class_second_best_scores[class_index], dtype=np.float32
        )[0]
        assert results.class_second_hard_assignment[class_index, TAIL_START] == int(
            np.asarray(outputs.class_second_best_argmaxes[class_index])[0]
        )
    # Zero image offset: the native and the absolute scores are the same numbers.
    assert_matches(results.class_best_offset_free_log_score[:, TAIL_START], results.class_best_log_score[:, TAIL_START])


def test_a_batch_compacts_its_support_on_the_device_and_leaves_the_host_rows_unset(make):
    plan, args, _ = make()
    _, results = _published(plan, args)
    assert [len(counts) for counts in results.device_significance_counts] == [1, 1]
    assert results.device_significance_starts == [[TAIL_START], [TAIL_START]]
    assert all(rows[TAIL_START] is None for rows in results.significant_sample_indices)
    assert results.sig_rot_any.shape == (2, 5) and results.sig_rot_any.any()


def test_a_deferred_batch_publishes_the_same_rows_as_one_published_at_once(make):
    plan, args, _ = make()
    _, at_once = _published(plan, args, defer=False)
    _, deferred = _published(plan, args, defer=True)
    for name in ("hard_assignment", "class_assignment", "n_sig_all", "cutoff_count_all", "normalization_log_z",
                 "max_posterior", "class_log_evidence", "class_best_log_score", "sig_rot_any"):
        assert_matches(getattr(deferred, name)[..., TAIL_START:] if name != "sig_rot_any" else deferred.sig_rot_any,
                       getattr(at_once, name)[..., TAIL_START:] if name != "sig_rot_any" else at_once.sig_rot_any)


def test_a_dump_batch_keeps_a_host_mask_and_its_support_rows_and_hands_the_dump_what_it_needs(make, monkeypatch):
    plan, args, _ = make()
    prepared = _prepared(plan, args)
    outputs = score_batch(plan, prepared, defer_publish=False).outputs
    dump_batch = dataclasses.replace(
        outputs,
        debug_dump_enabled=True,
        dump_target_local_positions=None,
        dump_target_pre_prior_blocks_per_class=None,
        dump_target_with_prior_blocks_per_class=None,
    )
    calls = []
    monkeypatch.setattr(pass1_publish, "maybe_dump_k_class_significance_batch", lambda **kwargs: calls.append(kwargs))
    results = Pass1Outputs.allocate(plan.output_plan)
    publish_batch(dump_batch, results, plan.output_plan, plan.dump_context)
    (dump,) = calls
    assert dump["batch_sig_mask"].dtype == bool and dump["batch_sig_mask"].shape == (3, 30)
    assert len(dump["class_weight_mats"]) == 2 and dump["class_weight_mats"][0].shape == (3, 15)
    assert dump["n_classes"] == 2 and dump["relion_f32_sum_weight"] is not None
    # No device compaction: the support rows come from the host mask.
    assert [len(counts) for counts in results.device_significance_counts] == [0, 0]
    assert all(rows[TAIL_START] is not None for rows in results.significant_sample_indices)


# -------------------------------------------------------------------------------------- the order of the loop


def _loop_events(monkeypatch, plan):
    events = []
    for name, record in (
        ("prepare_batch", lambda *a, **k: ("prepare", a[3])),
        ("score_batch", lambda plan_, batch, **k: ("score", batch.start_idx, k["defer_publish"])),
        ("publish_batch", lambda batch, *a, **k: ("publish", batch.start_idx)),
    ):
        real = getattr(significance, name)

        def spy(*args, _real=real, _record=record, **kwargs):
            events.append(_record(*args, **kwargs))
            return _real(*args, **kwargs)

        monkeypatch.setattr(significance, name, spy)
    significance.run_pass1(plan)
    return events


def test_the_float32_route_publishes_each_batch_after_the_next_batchs_operands_and_before_its_program(make, monkeypatch):
    plan, _, _ = make()
    assert _loop_events(monkeypatch, plan) == [
        ("prepare", 0), ("score", 0, True),
        ("prepare", 3), ("publish", 0), ("score", 3, True),
        ("prepare", 6), ("publish", 3), ("score", 6, True),
        ("publish", 6),
    ]


@pytest.mark.parametrize("how", ["generic_route", "no_support"])
def test_a_batch_that_does_not_defer_is_published_before_the_next_batch_is_prepared(make, monkeypatch, how):
    if how == "generic_route":
        monkeypatch.setenv("RECOVAR_K1_RELION_F32_COARSE_SUPPORT", "0")
        plan, _, _ = make()
    else:
        plan, _, _ = make(collect_significance=False)
    assert _loop_events(monkeypatch, plan) == [
        ("prepare", 0), ("score", 0, False), ("publish", 0),
        ("prepare", 3), ("score", 3, False), ("publish", 3),
        ("prepare", 6), ("score", 6, False), ("publish", 6),
    ]


# ----------------------------------------------------------------------------------------- the typed result


def test_running_the_plan_is_what_the_keyword_entry_does(make):
    plan, args, kwargs = make()
    result = significance.run_pass1(plan)
    assert isinstance(result, Pass1Result)
    _assert_significance_results_match(result, significance._compute_k_class_significance_batched(*args, **kwargs))


def test_the_stats_are_typed_fields_with_the_shapes_and_dtypes_of_the_pass(make):
    plan, args, _ = make()
    stats = significance.run_pass1(plan).stats
    assert isinstance(stats, Pass1Stats)
    assert stats.executed_coarse_backend == "gemm_macro"
    assert stats.normalization_log_z.shape == (7,) and stats.normalization_log_z.dtype == np.float64
    assert stats.class_log_evidence_per_image.shape == (2, 7)
    assert stats.significant_cutoff_counts.shape == (7,) and stats.significant_cutoff_counts.dtype == np.int32
    assert stats.class_best_log_score_per_image.shape == (2, 7) and stats.class_hard_assignments.shape == (2, 7)
    assert stats.class_second_best_log_score_per_image is None and stats.class_second_hard_assignments is None
    assert stats.relion_f32_sum_weight.shape == (7,) and stats.relion_f32_sum_weight.dtype == np.float32
    assert stats.relion_f32_max_posterior is None and stats.coarse_significance_support_audit is None
    assert dataclasses.is_dataclass(stats) and stats.__dataclass_params__.frozen


def test_asking_for_the_float32_normalization_returns_both_arrays(make):
    plan, _, _ = make(return_relion_f32_normalization=True)
    stats = significance.run_pass1(plan).stats
    assert stats.relion_f32_sum_weight.shape == stats.relion_f32_max_posterior.shape == (7,)
    assert stats.relion_f32_max_posterior.dtype == np.float32


def test_the_supports_of_a_collecting_pass_are_one_device_compacted_list_per_class(make):
    plan, _, _ = make()
    result = significance.run_pass1(plan)
    assert len(result.significant_sample_indices) == 2
    for rows in result.significant_sample_indices:
        assert len(rows) == 7 and hasattr(rows, "csr")
    assert (result.n_sig_all > 0).all() and result.sig_rot_any.shape == (2, 5)


def test_a_pass_that_collects_no_support_returns_none_for_it_and_zero_counts(make):
    plan, _, _ = make(collect_significance=False)
    result = significance.run_pass1(plan)
    assert result.significant_sample_indices is None
    assert (result.n_sig_all == 0).all() and (result.stats.significant_cutoff_counts == 0).all()


def test_the_support_audit_hashes_the_supports_and_refuses_a_pass_that_collected_none(make, monkeypatch):
    monkeypatch.setenv("RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT", "1")
    audit = significance.run_pass1(make()[0]).stats.coarse_significance_support_audit
    assert audit["n_classes"] == 2 and audit["n_images"] == 7 and audit["support_ids_included"] is False
    assert len(audit["aggregate_support_sha256"]) == 64 and "per_class_image_support_ids" not in audit
    monkeypatch.setenv("RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS", "1")
    assert "per_class_image_support_ids" in significance.run_pass1(make()[0]).stats.coarse_significance_support_audit
    with pytest.raises(RuntimeError, match="requires collect_significance=True"):
        significance.run_pass1(make(collect_significance=False)[0])


def test_the_audit_of_the_same_pass_is_the_same_hash(make, monkeypatch):
    monkeypatch.setenv("RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT", "1")
    first = significance.run_pass1(make()[0]).stats.coarse_significance_support_audit
    second = significance.run_pass1(make()[0]).stats.coarse_significance_support_audit
    assert first["aggregate_support_sha256"] == second["aggregate_support_sha256"]


# ----------------------------------------------------------------------------------------- timing and counts


def test_the_batch_timing_line_reports_the_batches_and_images_and_a_pass_without_batches_logs_nothing(caplog):
    with caplog.at_level(logging.INFO, logger="relax.scoring.pass1_assembly"):
        log_batch_timing([], 0.0, 0)
        assert not caplog.records
        log_batch_timing([0.0, 1.0, 3.0], 0.0, 7)
    (record,) = caplog.records
    assert record.getMessage().startswith("K-class coarse pass-1 batch timing: batches=3 images=7 ")


def test_the_tree_rescore_totals_add_up_over_the_batches_and_are_logged_once(caplog):
    batch = TreeRescoreBatch(state=None, ambiguous_images=2, exact_ties=1, winner_changes=1)
    totals = TreeRescoreTotals().after_batch(3, batch).after_batch(3, batch)
    assert (totals.examined, totals.ambiguous, totals.exact_ties, totals.winner_changes) == (6, 4, 2, 2)
    assert TreeRescoreTotals() == TreeRescoreTotals(0, 0, 0, 0)
    with caplog.at_level(logging.WARNING, logger="relax.scoring.tree_rescore"):
        log_tree_rescore_totals(totals)
    (record,) = caplog.records
    assert record.args == (6, 4, 2, 2)


# ----------------------------------------------------------------------------- the normalized-CC route


@pytest.fixture
def make_cc(make):
    """``make_cc(**request_overrides)`` -> ``(plan, args, kwargs)`` of a one-class normalized-CC pass (``--firstiter_cc``)."""

    def build(**overrides):
        plan, args, kwargs = make(
            n_classes=1,
            score_mode="normalized_cc",
            return_class_best=True,
            adaptive_fraction=1.0,
            max_significants=1,
            rotation_log_prior=None,
            translation_log_prior=None,
            **overrides,
        )
        return plan, args, kwargs

    return build


def test_a_cc_batch_is_prepared_with_the_correlation_image_and_no_high_resolution_power(make_cc):
    plan, args, _ = make_cc()
    batch = _prepared(plan, args)
    operands = batch.operands
    assert operands.corr_img is not None and operands.initial_diff2 is None
    # The score's pixel weight is the correlation image times the half-spectrum weights.
    assert_matches(
        np.asarray(operands.pixel_weight),
        np.asarray(operands.corr_img) * np.asarray(plan.route.operand_plan.score_half_weights),
        rtol=1e-6,
    )
    shifted, pixel_weight, initial_diff2 = batch.program_inputs
    assert shifted.dtype == np.complex64 and pixel_weight.dtype == np.float32 and initial_diff2 is None


def test_a_cc_batch_has_the_posterior_weights_and_none_of_the_float32_route_sums(make_cc):
    plan, args, _ = make_cc()
    outputs = score_batch(plan, _prepared(plan, args), defer_publish=False).outputs
    assert outputs.weights is not None and outputs.sig_mask is not None
    assert outputs.pmax is None and outputs.sum_weight is None and outputs.significant_weight is None


def test_a_cc_pass_never_defers_its_publish(make_cc, monkeypatch):
    plan, _, _ = make_cc()
    assert _loop_events(monkeypatch, plan) == [
        ("prepare", 0), ("score", 0, False), ("publish", 0),
        ("prepare", 3), ("score", 3, False), ("publish", 3),
        ("prepare", 6), ("score", 6, False), ("publish", 6),
    ]


def test_a_cc_pass_reports_its_backend_and_derives_the_maximum_posterior_from_the_best_score(make_cc):
    plan, _, _ = make_cc()
    result = significance.run_pass1(plan)
    stats = result.stats
    assert stats.executed_coarse_backend == "exact_cc_gemm"
    assert stats.relion_f32_sum_weight is None and stats.relion_f32_max_posterior is None
    assert stats.class_best_log_score_per_image.shape == (1, 7) and stats.class_hard_assignments.shape == (1, 7)
    assert (result.class_assignment == 0).all()
    assert_matches(
        stats.max_posterior_per_image,
        np.exp(stats.best_log_score_per_image - stats.normalization_log_z).astype(np.float32),
        rtol=1e-5,
    )
