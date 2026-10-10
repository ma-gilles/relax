import ast
import dataclasses
import inspect
import logging
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import refinement_specs
from helpers.float_compare import assert_matches

from relax.classification import k_class_results
from relax.classification.k_class_results import KClassEMResult
from relax.dense import scoring_policy
from relax.helpers import batch_planning, oversampling
from relax.helpers.batch_planning import (
    estimate_relion_em_batch_sizes,
    safe_dense_k_class_rotation_block_size,
    safe_firstiter_cc_image_batch_size,
)
from relax.helpers.types import NoiseStats, make_relion_stats
from relax.refinement import (
    dense_half,
    half_inputs,
    local_half,
    local_sampling,
    shape_class_scoring,
)
from relax.refinement.half_inputs import HalfSet
from relax.refinement.refinement_options import ScoringVariants
from relax.sparse_pass2 import local_search_records


def _dense_owners(**values):
    """Build the seven explicit dense-scoring owners from concise test values."""

    sampling_state = values.pop("state")
    owners = (
        half_inputs.HalfScoringData(
            particles=HalfSet(
                index=values.pop("k"),
                dataset=values.pop("experiment_dataset"),
                image_corrections=values.pop("image_corrections_k"),
                scale_corrections=values.pop("scale_corrections_k"),
            ),
            reference=values.pop("means_k"),
            mean_variance=values.pop("mean_variance"),
            noise_variance=values.pop("noise_variance_k"),
        ),
        refinement_specs.dense_sampling_spec(
            effective_rotations=values.pop("effective_rotations"),
            current_translations=values.pop("current_translations"),
            base_translations=values.pop("base_translations"),
            current_healpix_order=values.pop("current_healpix_order"),
            oversampling_order=sampling_state.adaptive_oversampling,
            translation_step=sampling_state.translation_step,
            random_perturbation=values.pop("random_perturbation"),
            image_window_size=values.pop("cs_for_engine"),
            coarse_rotation_ids=values.pop("coarse_rotation_ids", None),
        ),
        dense_half.DensePriorSpec(
            rotation_log_prior_k=values.pop("rotation_log_prior_k"),
            class_rotation_log_prior_k=values.pop("class_rotation_log_prior_k"),
            translation_log_prior=values.pop("translation_log_prior"),
            translation_search_base=values.pop("translation_search_base"),
            trans_prior_center_for_engine=values.pop("trans_prior_center_for_engine"),
            class_log_priors=values.pop("class_log_priors"),
        ),
        refinement_specs.dense_batch_policy(
            image_batch_size=values.pop("image_batch_size"),
            safe_batch_sizes=values.pop("safe_batch_sizes"),
            max_significants=values.pop("max_significants"),
            significance_safe_batch_sizes=values.pop("significance_safe_batch_sizes", None),
            k_class_image_batch_size_override=values.pop("k_class_image_batch_size_override", None),
            k_class_rotation_block_size_override=values.pop("k_class_rotation_block_size_override", None),
        ),
        refinement_specs.dense_variant_policy(
            score_mode=values.pop("firstiter_score_mode_this_iter"),
            winner_take_all=values.pop("firstiter_winner_take_all_this_iter"),
            k_class_enabled=(k_class_enabled := values.pop("k_class_enabled")),
            firstiter_cc=values.pop("relion_firstiter_cc_this_iter"),
            coarse_window_size=values.pop("firstiter_coarse_current_size", None),
            fine_window_size=values.pop("firstiter_fine_current_size", None),
        ),
        refinement_specs.dense_execution_policy(
            disc_type=values.pop("disc_type"),
            bpref_device_signature_active=values.pop("bpref_device_signature_active", False),
            debug_iteration=values.pop("debug_iteration", None),
            relion_x_half_mstep=ScoringVariants.from_environ().relion_x_half_mstep(k_class=k_class_enabled),
            precision=scoring_policy.DENSE_PRECISION,
        ),
        shape_class_scoring.OpticsSpec.single_shape(),
    )
    assert not values, f"unmapped dense owner values: {sorted(values)}"
    return owners


def _firstiter_cc_dispatch(
    *,
    mean,
    image_shape,
    n_rotations,
    n_translations,
    image_batch_size,
    em_kwargs,
    safe_batch_sizes,
    coarse_current_size,
    fine_current_size,
    projection_scale=1.0,
    effective_device_source=None,
):
    """Run the first-iteration CC dispatch on the dense owners of a small half."""

    half, sampling, priors, batching, variant, execution, _optics = _dense_owners(
        k=0,
        experiment_dataset=SimpleNamespace(image_shape=image_shape),
        means_k=mean,
        mean_variance=None,
        noise_variance_k=None,
        effective_rotations=np.zeros((n_rotations, 3, 3), dtype=np.float32),
        current_translations=np.zeros((n_translations, 2), dtype=np.float32),
        base_translations=np.zeros((n_translations, 2), dtype=np.float32),
        current_healpix_order=1,
        state=SimpleNamespace(adaptive_oversampling=1, translation_step=2.0),
        random_perturbation=0.0,
        disc_type="linear_interp",
        image_batch_size=image_batch_size,
        rotation_log_prior_k=None,
        class_rotation_log_prior_k=None,
        translation_log_prior=None,
        translation_search_base=None,
        trans_prior_center_for_engine=None,
        image_corrections_k=None,
        scale_corrections_k=None,
        firstiter_score_mode_this_iter="normalized_cc",
        firstiter_winner_take_all_this_iter=True,
        cs_for_engine=None,
        class_log_priors=None,
        k_class_enabled=True,
        relion_firstiter_cc_this_iter=True,
        safe_batch_sizes=safe_batch_sizes,
        max_significants=None,
        firstiter_coarse_current_size=coarse_current_size,
        firstiter_fine_current_size=fine_current_size,
    )
    return dense_half._score_kclass_firstiter_cc_pass2(
        half,
        dataclasses.replace(sampling, effective_device_source=effective_device_source),
        priors,
        batching,
        variant,
        execution,
        projection_scale=projection_scale,
        magnification=None,
        em_kwargs=em_kwargs,
    )


def test_firstiter_cc_core_keeps_owner_dependencies_visible():
    function = dense_half._score_kclass_firstiter_cc_pass2
    assert tuple(inspect.signature(function).parameters) == (
        "half", "sampling", "priors", "batching", "variant", "execution",
        "projection_scale", "magnification", "em_kwargs",
    )

    tree = ast.parse(inspect.getsource(function))
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            [*node.targets] if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    stable_field_names = {
        field.name
        for owner in (
            half_inputs.HalfScoringData,
            dense_half.DenseSamplingSpec,
            dense_half.DensePriorSpec,
            dense_half.DenseBatchPolicy,
            dense_half.DenseVariantPolicy,
            dense_half.DenseExecutionPolicy,
        )
        for field in dataclasses.fields(owner)
    }
    assert assigned_names.isdisjoint(stable_field_names)


def test_dense_half_core_keeps_owner_dependencies_visible():
    function = dense_half._score_half_dense_one_shape
    assert tuple(inspect.signature(function).parameters) == (
        "half", "sampling", "priors", "batching", "variant", "execution", "optics",
    )

    tree = ast.parse(inspect.getsource(function))
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ([*node.targets] if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    stable_field_names = {
        field.name
        for owner in (
            half_inputs.HalfScoringData,
            dense_half.DenseSamplingSpec,
            dense_half.DensePriorSpec,
            dense_half.DenseBatchPolicy,
            dense_half.DenseVariantPolicy,
            dense_half.DenseExecutionPolicy,
            shape_class_scoring.OpticsSpec,
        )
        for field in dataclasses.fields(owner)
    }
    # The K=1 oversampling-0 copies of firstiter_*_current_size were dead stores (deep2 H-B6).
    route_planning_locals = {"symmetry"}
    assert assigned_names & stable_field_names == route_planning_locals


def test_local_half_core_keeps_owner_dependencies_visible():
    function = local_half._score_half_local_one_shape
    assert tuple(inspect.signature(function).parameters) == (
        "half", "sampling", "priors", "batching", "execution", "diagnostics", "optics",
    )

    tree = ast.parse(inspect.getsource(function))
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ([*node.targets] if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    stable_field_names = {
        field.name
        for owner in (
            half_inputs.HalfScoringData,
            local_sampling.LocalSampling,
            local_half.LocalPriorSpec,
            local_half.LocalBatchPolicy,
            local_half.LocalExecutionPolicy,
            local_half.LocalDiagnosticPolicy,
            shape_class_scoring.OpticsSpec,
        )
        for field in dataclasses.fields(owner)
    }
    assert not assigned_names & stable_field_names


def test_local_iteration_core_keeps_owner_dependencies_visible():
    function = local_half._run_local_search_iteration
    assert tuple(inspect.signature(function).parameters) == (
        "data", "grid", "kernel", "support",
    )

    tree = ast.parse(inspect.getsource(function))
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ([*node.targets] if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    stable_field_names = {
        field.name
        for owner in (
            local_search_records.LocalSearchData,
            local_half.LocalSearchGridSpec,
            local_search_records.LocalSearchKernelPolicy,
            local_search_records.LocalSearchSupportPolicy,
        )
        for field in dataclasses.fields(owner)
    }
    normalized_or_planned_locals = {
        "prior_rotations",
        "prior_translations",
    }
    assert assigned_names & stable_field_names == normalized_or_planned_locals


def test_firstiter_winner_take_all_assembly_reports_unit_pmax_across_score_normalizations():
    """RELION reports Pmax=1 after firstiter-CC binarizes the winning weight."""
    per_class_stats = (
        make_relion_stats(
            log_evidence_per_image=np.array([1_000.0], dtype=np.float32),
            best_log_score_per_image=np.array([-1_000.0], dtype=np.float32),
            max_posterior_per_image=np.ones(1, dtype=np.float32),
            rotation_posterior_sums=np.ones(1, dtype=np.float32),
        ),
    )

    result = k_class_results._assemble_result(
        class_log_evidence=np.array([[1_000.0]], dtype=np.float64),
        new_means=None,
        Ft_y=[jnp.zeros(1, dtype=jnp.complex64)],
        Ft_ctf=[jnp.zeros(1, dtype=jnp.float32)],
        per_class_hard_assignments=np.zeros((1, 1), dtype=np.int32),
        per_class_stats=per_class_stats,
        noise_stats=None,
        firstiter_winner_take_all=True,
    )

    assert_matches(np.asarray(result.stats.max_posterior_per_image), np.ones(1))


def test_firstiter_cc_budget_preserves_256_k4_completion_batch_size():
    # K=4 completion benchmarks use 256^2 images and 116 fine translations
    # at adaptive_oversampling=1. The cap must not collapse the requested
    # batch size 50 back to single digits on A100/H100 runs.
    assert safe_firstiter_cc_image_batch_size(116, (256, 256)) >= 50


def test_firstiter_cc_budget_still_caps_larger_tiles():
    assert 1 <= safe_firstiter_cc_image_batch_size(137, (384, 384)) < 250


def test_firstiter_cc_budget_env_override_lifts_debug_cap(monkeypatch):
    default_batch = safe_firstiter_cc_image_batch_size(116, (256, 256))
    assert default_batch == 70

    monkeypatch.setenv("RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET", str(3 * 268_435_456))

    assert safe_firstiter_cc_image_batch_size(116, (256, 256)) >= 187


def test_firstiter_cc_budget_env_override_rejects_invalid(monkeypatch):
    monkeypatch.setenv("RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET", "0")

    try:
        safe_firstiter_cc_image_batch_size(116, (256, 256))
    except ValueError as exc:
        assert "RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET" in str(exc)
    else:
        raise AssertionError("invalid firstiter budget override did not raise")


def test_kclass_adaptive_grid_batch_plan_uses_fine_grid_for_pass2():
    calls = []

    def fake_safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        calls.append((int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch))
        if int(n_rot) == 4608:
            return 44, 275
        if int(n_rot) == 576:
            return 50, 576
        raise AssertionError((n_rot, n_trans))

    plan = batch_planning._plan_kclass_adaptive_grid_batch_sizes(
        coarse_rotations=np.zeros((576, 3, 3), dtype=np.float32),
        coarse_translations=np.zeros((29, 2), dtype=np.float32),
        fine_rotations=np.zeros((4608, 3, 3), dtype=np.float32),
        fine_translations=np.zeros((116, 2), dtype=np.float32),
        n_classes=4,
        image_shape=(256, 256),
        coarse_current_size=40,
        fine_current_size=90,
        safe_batch_sizes=fake_safe_batch_sizes,
    )

    assert calls == [
        (4608, 116, 4, (256, 256), 90),
        (576, 29, 4, (256, 256), 40),
    ]
    assert plan.pass2_image_batch_size == 44
    assert plan.pass2_rotation_block_size == 275
    assert plan.significance_image_batch_size == 50
    assert plan.significance_rotation_block_size == 576


def test_firstiter_cc_adaptive_dispatch_clamps_against_fine_translation_grid(monkeypatch):
    captured = {}
    fine_trans = np.zeros((116, 2), dtype=np.float32)

    def fake_grids(*args, **kwargs):
        coarse_rot = np.zeros((576, 3, 3), dtype=np.float32)
        coarse_trans = np.zeros((29, 2), dtype=np.float32)
        fine_rot = np.zeros((4608, 3, 3), dtype=np.float32)
        rot_parent = np.zeros(fine_rot.shape[0], dtype=np.int64)
        trans_parent = np.zeros(fine_trans.shape[0], dtype=np.int64)
        outputs = (coarse_rot, coarse_trans, fine_rot, fine_trans, rot_parent, trans_parent)
        if kwargs.get("return_mstep_rotations", False):
            return (*outputs, np.full_like(fine_rot, 0.25))
        return outputs

    def fake_adaptive(*args, **kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(dense_half, "build_adaptive_pass2_grids", fake_grids)
    monkeypatch.setattr(dense_half, "run_dense_k_class_em_adaptive", fake_adaptive)

    def fake_safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        assert classes == 2
        assert image_shape_for_batch == (256, 256)
        if (int(n_rot), int(n_trans), current_size_for_batch) == (4608, 116, 90):
            return 5, 999
        if (int(n_rot), int(n_trans), current_size_for_batch) == (576, 29, 40):
            return 120, 700
        raise AssertionError((n_rot, n_trans, current_size_for_batch))

    dispatch = _firstiter_cc_dispatch(
        mean=np.zeros((2, 4), dtype=np.complex64),
        image_shape=(256, 256),
        n_rotations=576,
        n_translations=29,
        image_batch_size=200,
        em_kwargs={"image_batch_size": 88, "rotation_block_size": 576},
        safe_batch_sizes=fake_safe_batch_sizes,
        coarse_current_size=40,
        fine_current_size=90,
    )

    assert dispatch.result == "result"
    assert dispatch.n_fine_translations == 116
    expected_fine_ibs = min(88, safe_firstiter_cc_image_batch_size(116, (256, 256)))
    expected_coarse_ibs = min(120, safe_firstiter_cc_image_batch_size(29, (256, 256)))
    assert captured["image_batch_size"] == expected_fine_ibs
    assert captured["rotation_block_size"] == min(576, safe_dense_k_class_rotation_block_size(116, expected_fine_ibs))
    assert captured["significance_image_batch_size"] == expected_coarse_ibs
    assert captured["significance_rotation_block_size"] == min(
        700,
        safe_dense_k_class_rotation_block_size(29, expected_coarse_ibs),
    )
    assert captured["firstiter_cc_pass2_only_best_coarse"] is True
    assert captured["relion_fine_mstep_prune"] is True
    assert np.all(captured["fine_mstep_rotations_override"] == 0.25)


@pytest.mark.parametrize("n_classes", [1, 4], ids=["k1", "k4"])
@pytest.mark.parametrize("separate_coarse", [False, True])
def test_firstiter_cc_dispatch_uses_coarse_batch_for_significance(monkeypatch, caplog, n_classes, separate_coarse):
    captured = {}
    dispatch = {}
    original_dispatch = dense_half._score_kclass_firstiter_cc_pass2

    def capture_dispatch(*owners, **values):
        dispatch.update(values)
        return original_dispatch(*owners, **values)

    caplog.set_level(logging.INFO, logger="relax.dense.half_scoring")
    calls = []

    class TinyDataset:
        particles_file = None  # built in memory: no RELION optics table
        image_shape = (256, 256)

    def fake_grids(*args, **kwargs):
        coarse_rot = np.zeros((576, 3, 3), dtype=np.float32)
        coarse_trans = np.zeros((29, 2), dtype=np.float32)
        fine_rot = np.zeros((4608, 3, 3), dtype=np.float32)
        fine_trans = np.zeros((116, 2), dtype=np.float32)
        rot_parent = np.arange(fine_rot.shape[0], dtype=np.int64) % coarse_rot.shape[0]
        trans_parent = np.arange(fine_trans.shape[0], dtype=np.int64) % coarse_trans.shape[0]
        outputs = (coarse_rot, coarse_trans, fine_rot, fine_trans, rot_parent, trans_parent)
        if kwargs.get("return_mstep_rotations", False):
            return (*outputs, np.full_like(fine_rot, 0.25))
        return outputs

    def fake_safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        calls.append((int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch))
        if (int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch) == (
            576,
            29,
            None,
            None,
            90,
        ):
            return 187, 700
        if (int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch) == (
            4608,
            116,
            n_classes,
            (256, 256),
            90,
        ):
            return 5, 999
        if (int(n_rot), int(n_trans), classes, image_shape_for_batch, current_size_for_batch) == (
            576,
            29,
            n_classes,
            (256, 256),
            40,
        ):
            return 187, 700
        raise AssertionError((n_rot, n_trans, classes, image_shape_for_batch, current_size_for_batch))

    def coarse_planner(*args, **kwargs):
        fake_safe_batch_sizes(*args, **kwargs)
        return 133, 333

    def fake_adaptive(*args, **kwargs):
        captured.update(kwargs)
        captured["mean"] = args[1]
        n_images = 3
        n_fine_rot = 4608
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=np.zeros(n_images, dtype=np.float32),
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_fine_rot, dtype=np.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(2, dtype=jnp.float32),
            wsum_img_power=jnp.ones(2, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(n_images),
        )
        return KClassEMResult(
            new_means=jnp.zeros((n_classes, 4), dtype=jnp.complex64),
            Ft_y=jnp.zeros((n_classes, 4), dtype=jnp.complex64),
            Ft_ctf=jnp.ones((n_classes, 4), dtype=jnp.float32),
            per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
            class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            class_responsibilities=jnp.ones((n_classes, n_images), dtype=jnp.float32),
            class_posterior_sums=jnp.ones(n_classes, dtype=jnp.float32),
            stats=stats,
            per_class_stats=(stats,) * n_classes,
            noise_stats=(noise_stats,) * n_classes,
            aggregate_noise_stats=noise_stats,
            best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
            best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
            best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        )

    monkeypatch.setattr(dense_half, "build_adaptive_pass2_grids", fake_grids)
    monkeypatch.setattr(dense_half, "run_dense_k_class_em_adaptive", fake_adaptive)

    monkeypatch.setattr(dense_half, "_score_kclass_firstiter_cc_pass2", capture_dispatch)
    means = jnp.zeros(4 if n_classes == 1 else (n_classes, 4), dtype=jnp.complex64)
    coarse_ids = np.arange(576, dtype=np.int32)

    result = dense_half._score_half_dense(*_dense_owners(
        k=0,
        experiment_dataset=TinyDataset(),
        means_k=means,
        mean_variance=jnp.ones(4, dtype=jnp.float32),
        noise_variance_k=jnp.ones(4, dtype=jnp.float32),
        effective_rotations=np.zeros((576, 3, 3), dtype=np.float32),
        current_translations=np.zeros((29, 2), dtype=np.float32),
        base_translations=np.zeros((29, 2), dtype=np.float32),
        current_healpix_order=1,
        state=SimpleNamespace(adaptive_oversampling=1, translation_step=2.0),
        random_perturbation=0.0,
        disc_type="linear_interp",
        image_batch_size=187,
        rotation_log_prior_k=None,
        class_rotation_log_prior_k=None,
        translation_log_prior=None,
        translation_search_base=None,
        trans_prior_center_for_engine=None,
        image_corrections_k=None,
        scale_corrections_k=None,
        firstiter_score_mode_this_iter="normalized_cc",
        firstiter_winner_take_all_this_iter=True,
        cs_for_engine=90,
        class_log_priors=None,
        k_class_enabled=n_classes > 1,
        relion_firstiter_cc_this_iter=True,
        safe_batch_sizes=fake_safe_batch_sizes,
        significance_safe_batch_sizes=coarse_planner if separate_coarse else None,
        max_significants=None,
        firstiter_coarse_current_size=40,
        firstiter_fine_current_size=90,
        bpref_device_signature_active=True,
        debug_iteration=7,
        coarse_rotation_ids=coarse_ids,
    ))

    assert calls == [
        (576, 29, None, None, 90),
        (4608, 116, n_classes, (256, 256), 90),
        (576, 29, n_classes, (256, 256), 40),
    ]
    assert captured["image_batch_size"] == safe_firstiter_cc_image_batch_size(116, (256, 256))
    assert captured["significance_image_batch_size"] == (133 if separate_coarse else 187)
    assert captured["rotation_block_size"] == min(700, safe_dense_k_class_rotation_block_size(116, captured["image_batch_size"]))
    # K-class applies the existing coarse score-tile cap; K=1 retains its batch.
    assert captured["significance_rotation_block_size"] == (333 if separate_coarse else (700 if n_classes == 1 else 368))
    assert captured["bpref_device_signature_active"] is True
    assert captured["debug_iteration"] == 7
    assert np.all(captured["fine_mstep_rotations_override"] == 0.25)
    assert result.ha.shape == (3,)
    assert result.coarse_ha.shape == (3,)

    assert captured["mean"].shape == (n_classes, 4)
    label = "K=1 " if n_classes == 1 else ""
    assert any(f"STRICT-PARITY {label}routing iter-1" in record.getMessage() for record in caplog.records)
    # The engine receives the clamped copy; the dispatch leaves the caller's dictionary alone.
    assert dispatch["em_kwargs"]["image_batch_size"] == 187
    if n_classes == 1:
        assert captured["coarse_rotation_ids"] is None
    else:
        assert captured["mean"] is means
        assert captured["coarse_rotation_ids"] is coarse_ids
        assert captured["coarse_rotation_ids"] is coarse_ids


@pytest.mark.parametrize("max_significants", [None, 5])
def test_kclass_nonfirstiter_adaptive_dispatch_sizes_actual_fine_grid(monkeypatch, max_significants):
    captured = {}

    class TinyDataset:
        particles_file = None  # built in memory: no RELION optics table
        image_shape = (256, 256)

    def fake_grids(*args, **kwargs):
        coarse_rot = np.zeros((576, 3, 3), dtype=np.float32)
        coarse_trans = np.zeros((29, 2), dtype=np.float32)
        fine_rot = np.zeros((4608, 3, 3), dtype=np.float32)
        fine_trans = np.zeros((116, 2), dtype=np.float32)
        rot_parent = np.arange(fine_rot.shape[0], dtype=np.int64) % coarse_rot.shape[0]
        trans_parent = np.arange(fine_trans.shape[0], dtype=np.int64) % coarse_trans.shape[0]
        outputs = (coarse_rot, coarse_trans, fine_rot, fine_trans, rot_parent, trans_parent)
        if kwargs.get("return_mstep_rotations", False):
            return (*outputs, np.full_like(fine_rot, 0.25))
        return outputs

    def fake_safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        assert classes in {None, 4}
        assert image_shape_for_batch in {None, (256, 256)}
        if (int(n_rot), int(n_trans), current_size_for_batch) == (4608, 116, 90):
            return 44, 275
        if (int(n_rot), int(n_trans), current_size_for_batch) == (576, 29, 40):
            return 50, 576
        if (int(n_rot), int(n_trans), current_size_for_batch) == (576, 29, 90):
            return 50, 2000
        raise AssertionError((n_rot, n_trans, current_size_for_batch))

    def fake_adaptive(*args, **kwargs):
        captured.update(kwargs)
        n_images = 3
        n_classes = 4
        n_fine_rot = 4608
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float32),
            best_log_score_per_image=np.zeros(n_images, dtype=np.float32),
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_fine_rot, dtype=np.float32),
        )
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.ones(2, dtype=jnp.float32),
            wsum_img_power=jnp.ones(2, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(n_images),
        )
        return KClassEMResult(
            new_means=jnp.zeros((n_classes, 4), dtype=jnp.complex64),
            Ft_y=jnp.zeros((n_classes, 4), dtype=jnp.complex64),
            Ft_ctf=jnp.ones((n_classes, 4), dtype=jnp.float32),
            per_class_hard_assignments=jnp.zeros((n_classes, n_images), dtype=jnp.int32),
            class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            pose_assignments=jnp.zeros(n_images, dtype=jnp.int32),
            class_responsibilities=jnp.ones((n_classes, n_images), dtype=jnp.float32) / n_classes,
            class_posterior_sums=jnp.ones(n_classes, dtype=jnp.float32),
            stats=stats,
            per_class_stats=tuple(stats for _ in range(n_classes)),
            noise_stats=tuple(noise_stats for _ in range(n_classes)),
            aggregate_noise_stats=noise_stats,
            best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
            best_pose_translations=jnp.zeros((n_images, 2), dtype=jnp.float32),
            best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        )

    monkeypatch.setattr(oversampling, "build_adaptive_pass2_grids", fake_grids)
    monkeypatch.setattr(dense_half, "run_dense_k_class_em_adaptive", fake_adaptive)

    result = dense_half._score_half_dense(*_dense_owners(
        k=0,
        experiment_dataset=TinyDataset(),
        means_k=jnp.zeros((4, 4), dtype=jnp.complex64),
        mean_variance=jnp.ones(4, dtype=jnp.float32),
        noise_variance_k=jnp.ones(4, dtype=jnp.float32),
        effective_rotations=np.zeros((576, 3, 3), dtype=np.float32),
        current_translations=np.zeros((29, 2), dtype=np.float32),
        base_translations=np.zeros((29, 2), dtype=np.float32),
        current_healpix_order=1,
        state=SimpleNamespace(adaptive_oversampling=1, translation_step=2.0),
        random_perturbation=0.0,
        disc_type="linear_interp",
        image_batch_size=50,
        rotation_log_prior_k=None,
        class_rotation_log_prior_k=None,
        translation_log_prior=None,
        translation_search_base=None,
        trans_prior_center_for_engine=None,
        image_corrections_k=None,
        scale_corrections_k=None,
        firstiter_score_mode_this_iter="gaussian",
        firstiter_winner_take_all_this_iter=False,
        cs_for_engine=90,
        class_log_priors=np.zeros(4, dtype=np.float32),
        k_class_enabled=True,
        relion_firstiter_cc_this_iter=False,
        safe_batch_sizes=fake_safe_batch_sizes,
        max_significants=max_significants,
        k_class_image_batch_size_override=50,
        k_class_rotation_block_size_override=2000,
        firstiter_coarse_current_size=40,
        firstiter_fine_current_size=90,
    ))

    assert captured["image_batch_size"] == 44
    assert captured["rotation_block_size"] == 275
    assert captured["significance_image_batch_size"] == 50
    assert captured["significance_rotation_block_size"] == 576
    assert captured["sparse_pass2"] is True
    assert np.all(captured["fine_mstep_rotations_override"] == 0.25)
    assert result.ha.shape == (3,)
    # The keywords both adaptive dense routes pass the engine identically.
    assert captured["accumulate_noise"] is True
    assert captured["adaptive_fraction"] == dense_half.RELION_ADAPTIVE_FRACTION
    assert captured["max_significants"] == (-1 if max_significants is None else 5)
    assert captured["relion_fine_mstep_prune"] is True
    assert type(captured["coarse_healpix_order"]) is int and captured["coarse_healpix_order"] == 1
    assert np.all(captured["class_log_priors"] == 0.0)


def test_dense_global_k1_batch_plan_accounts_for_pose_pixel_tile():
    plan = estimate_relion_em_batch_sizes(
        requested_image_batch_size=500,
        requested_rotation_block_size=40000,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=42,
        current_size=56,
    )

    assert plan.image_batch_size == 187
    assert plan.rotation_block_size < 9000
    assert plan.pose_pixel_tile_gb <= plan.projection_budget_gb * 1.01


def test_dense_global_k1_high_current_size_keeps_pose_pixel_tile_below_large_allocations():
    plan = estimate_relion_em_batch_sizes(
        requested_image_batch_size=500,
        requested_rotation_block_size=40000,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=42,
        current_size=184,
    )

    assert 100 <= plan.image_batch_size < 150
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb * 1.01
    assert plan.rotation_block_size < 250
    assert plan.pose_pixel_tile_gb < 1.7


def test_firstiter_cc_dispatch_projects_every_grid_through_the_shape_class_matrices(monkeypatch):
    """Images on another grid score, and back-project, with the class's projection matrices."""
    captured = {}
    fine_trans = np.zeros((9, 2), dtype=np.float32)

    def fake_grids(*args, **kwargs):
        coarse_rot = np.full((12, 3, 3), 2.0, dtype=np.float32)
        fine_rot = np.full((96, 3, 3), 4.0, dtype=np.float32)
        outputs = (
            coarse_rot, np.zeros((5, 2), dtype=np.float32), fine_rot, fine_trans,
            np.zeros(96, dtype=np.int64), np.zeros(9, dtype=np.int64),
        )
        return (*outputs, np.full_like(fine_rot, 8.0))

    def fake_adaptive(*args, **kwargs):
        captured["coarse_rot"], captured["fine_rot"] = args[4], args[6]
        captured.update(kwargs)
        return "result"

    def fake_projection(coarse, fine, mstep, **kwargs):
        captured["projection"] = kwargs
        return coarse / 2.0, fine / 2.0, mstep / 2.0

    monkeypatch.setattr(dense_half, "build_adaptive_pass2_grids", fake_grids)
    monkeypatch.setattr(dense_half, "project_pass2_rotations", fake_projection)
    monkeypatch.setattr(dense_half, "run_dense_k_class_em_adaptive", fake_adaptive)
    source = object()

    _firstiter_cc_dispatch(
        mean=np.zeros((1, 4), dtype=np.complex64),
        image_shape=(64, 64),
        n_rotations=12,
        n_translations=5,
        image_batch_size=20,
        em_kwargs={"image_batch_size": 20, "rotation_block_size": 96},
        safe_batch_sizes=None,
        coarse_current_size=None,
        fine_current_size=None,
        projection_scale=2.0,
        effective_device_source=source,
    )

    assert np.all(captured["coarse_rot"] == 1.0)
    assert np.all(captured["fine_rot"] == 2.0)
    assert np.all(captured["fine_mstep_rotations_override"] == 4.0)
    # The grid's own provenance picks RELION's per-path rule for each set of rows.
    projection = captured["projection"]
    assert projection["scale"] == 2.0 and projection["magnification"] is None
    assert projection["coarse_device_source"] is source and projection["grid_device_source"] is source
    assert (projection["coarse_healpix_order"], projection["adaptive_oversampling"]) == (1, 1)


def test_firstiter_cc_global_winner_pass2_carries_each_images_optics_group(monkeypatch):
    """The --firstiter_cc fine pass gives the resident engine each subset image's optics group."""
    from relax.classification import k_class
    from relax.sparse_pass2 import dispatch

    captured = {}

    class _Stop(Exception):
        pass

    def fake_pass2(dataset, *args, **kwargs):
        captured.update(kwargs, dataset=dataset)
        raise _Stop

    class _Dataset:
        volume_shape = (16, 16, 16)

        def subset(self, indices):
            return ("subset", tuple(int(i) for i in indices))

    monkeypatch.setattr(dispatch, "compute_pass2_stats_sparse", fake_pass2)
    optics = np.array([0, 1, 1, 0, 1], dtype=np.int32)
    with pytest.raises(_Stop):
        k_class._run_sparse_firstiter_global_winner_subset_pass2(
            _Dataset(),
            np.zeros((1, 4), dtype=np.complex64),
            np.ones((1, 4), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 3, 3), dtype=np.float32),
            None,
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            [[np.zeros(1, dtype=np.int64)] * 5],
            "linear_interp",
            coarse_result=SimpleNamespace(class_log_evidence=np.zeros((1, 5))),
            coarse_class_assignments=np.zeros(5, dtype=np.int32),
            n_rot_coarse=1,
            n_fine_trans=1,
            healpix_order=1,
            oversampling_order=0,
            accumulate_noise=True,
            return_best_pose_details=False,
            pass2_kwargs={
                "optics_group_ids": optics,
                "reconstruction_volume_current_size": 12,
                "reconstruction_image_radius": 5.5,
            },
        )

    assert captured["dataset"] == ("subset", (0, 1, 2, 3, 4))
    np.testing.assert_array_equal(captured["optics_group_ids"], optics)
    assert captured["reconstruction_volume_current_size"] == 12
    assert captured["reconstruction_image_radius"] == 5.5


def test_firstiter_cc_global_winner_pass2_keeps_the_optics_group_noise_table_with_as_many_groups_as_classes(
    monkeypatch,
):
    """A ``[G, P]`` noise table is per optics group, never per class, even when G equals K.

    With two classes and two optics groups the class-0 subset pass took row 0 as "class 0's
    noise", so every image backprojected with group 1's spectrum (SPA Class3D, 2026-09-30).
    """
    from relax.classification import k_class
    from relax.sparse_pass2 import dispatch

    captured = {}

    class _Stop(Exception):
        pass

    def fake_pass2(dataset, mean, mean_variance, noise_variance, *args, **kwargs):
        captured.update(kwargs, noise_variance=np.asarray(noise_variance), mean_variance=np.asarray(mean_variance))
        raise _Stop

    class _Dataset:
        volume_shape = (16, 16, 16)

        def subset(self, indices):
            return ("subset", tuple(int(i) for i in indices))

    monkeypatch.setattr(dispatch, "compute_pass2_stats_sparse", fake_pass2)
    noise = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    mean_variance = np.array([[1.0] * 4, [2.0] * 4], dtype=np.float32)
    optics = np.array([0, 1, 1, 0, 1], dtype=np.int32)
    with pytest.raises(_Stop):
        k_class._run_sparse_firstiter_global_winner_subset_pass2(
            _Dataset(),
            np.zeros((2, 4), dtype=np.complex64),
            mean_variance,
            noise,
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 3, 3), dtype=np.float32),
            None,
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            [[np.zeros(1, dtype=np.int64)] * 5] * 2,
            "linear_interp",
            coarse_result=SimpleNamespace(class_log_evidence=np.zeros((2, 5))),
            coarse_class_assignments=np.zeros(5, dtype=np.int32),
            n_rot_coarse=1,
            n_fine_trans=1,
            healpix_order=1,
            oversampling_order=0,
            accumulate_noise=True,
            return_best_pose_details=False,
            pass2_kwargs={"optics_group_ids": optics},
        )

    np.testing.assert_array_equal(captured["noise_variance"], noise)
    np.testing.assert_array_equal(captured["optics_group_ids"], optics)
    # The per-class mean variance is still the class's own.
    np.testing.assert_array_equal(captured["mean_variance"], mean_variance[0])
