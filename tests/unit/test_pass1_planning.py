"""Pass 1's planning stages, one at a time: the window, the priors, the blocks, the support plan and the route.

The pure stages are called on small arrays. The route and the planner's refusals run ``plan_pass1`` on the CPU exact-operand
harness (``helpers.exact_pass1_harness``), whose CUDA stand-ins let the planner build its records without a GPU.
"""

import jax
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_em_stage_glue_programs import _significance_call

from relax.scoring.pass1_batch import resolve_noise_tables
from relax.scoring.pass1_operands import CcOperandPlan, GaussianOperandPlan
from relax.scoring.pass1_plan import plan_pass1
from relax.scoring.pass1_priors import plan_rotation_blocks, validated_translation_log_prior
from relax.scoring.pass1_request import Pass1Request
from relax.scoring.pass1_results import OutputPlan, Pass1Outputs
from relax.scoring.pass1_scores import ProgramStatics, block_prior_terms, score_blocks
from relax.scoring.pass1_support import exact_order_rotation_prior
from relax.scoring.pass1_window import plan_scoring_window

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _cpu_stand_ins_only():
    """The stages run on the harness's CPU stand-ins; on a GPU backend they are not what is being qualified."""

    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only contract: the stage tests run on the exact-operand harness's CPU stand-ins")

SHAPE = (8, 8)
N_HALF = SHAPE[0] * (SHAPE[1] // 2 + 1)
DC_ROW = (SHAPE[0] // 2) * (SHAPE[1] // 2 + 1)  # the half-spectrum index of frequency (0, 0)


# --------------------------------------------------------------------------------------------------- the window


def _window(score_mode="gaussian", current_size=4, **overrides):
    options = dict(
        score_mode=score_mode,
        half_spectrum_scoring=True,
        square_window=False,
        window_at_box=False,
        nyquist_column_counting="relion",
        firstiter_cc_support="relion",
    )
    options.update(overrides)
    return plan_scoring_window(SHAPE, N_HALF, current_size, **options)


def test_a_pass_without_a_current_size_scores_the_whole_half_spectrum():
    window = _window(current_size=None)
    assert window.use_window is False
    assert window.window_indices is None
    assert window.score_size == SHAPE[0]
    assert window.score_half_weights is window.half_weights


def test_a_gaussian_window_scores_the_rows_of_its_current_size_and_weights_exactly_those():
    window = _window("gaussian", current_size=4)
    indices = np.asarray(window.window_indices)
    assert window.use_window is True and window.score_size == 4
    assert indices.size == np.unique(indices).size and indices.min() >= 0 and indices.max() < N_HALF
    assert np.asarray(window.score_half_weights).shape == indices.shape
    assert np.asarray(window.half_weights).shape == (N_HALF,)
    assert DC_ROW not in indices.tolist()
    assert window.cc_gaussian_support is False


def test_the_gaussian_window_at_the_box_keeps_RELIONs_radial_window_which_is_smaller_than_the_half_spectrum():
    at_box = _window("gaussian", current_size=SHAPE[0], window_at_box=True)
    assert at_box.use_window is True and at_box.score_size == SHAPE[0]
    assert 0 < np.asarray(at_box.window_indices).size < N_HALF


def test_the_normalized_cc_scores_a_square_window_with_its_dc_row():
    window = _window("normalized_cc", current_size=4)
    indices = np.asarray(window.window_indices)
    assert window.use_window is True and window.score_size == 4
    assert DC_ROW in indices.tolist()
    assert indices.size > np.asarray(_window("gaussian", current_size=4).window_indices).size


@pytest.mark.parametrize("support, expected", [("relion", False), ("gaussian", True)])
def test_the_cc_takes_its_image_power_on_the_gaussian_support_only_when_asked(support, expected):
    assert _window("normalized_cc", firstiter_cc_support=support).cc_gaussian_support is expected
    assert _window("gaussian", firstiter_cc_support=support).cc_gaussian_support is False


# ------------------------------------------------------------------------------------- rotations and the priors

ROTATIONS = np.tile(np.eye(3, dtype=np.float32), (5, 1, 1))


def test_rotations_are_padded_with_identities_to_whole_blocks_and_the_prior_with_zeros():
    rotations = ROTATIONS.copy()
    rotations[:, 0, 1] = np.arange(5)
    blocks = plan_rotation_blocks(
        rotations, np.arange(5, dtype=np.float32), n_classes=2, rotation_block_size=3, score_real_dtype=np.float32
    )
    assert blocks.rotations_padded.shape == (6, 3, 3)
    assert_matches(blocks.rotations_padded[:5], rotations)
    assert_matches(blocks.rotations_padded[5], np.eye(3))
    # [R] is shared by the classes: [K, R_padded], zero in the padding, of the score dtype.
    assert blocks.rotation_log_prior_padded.shape == (2, 6)
    assert blocks.rotation_log_prior_padded.dtype == np.float32
    assert_matches(blocks.rotation_log_prior_padded[1], [0, 1, 2, 3, 4, 0])


def test_whole_blocks_are_not_copied_and_a_missing_prior_stays_missing():
    blocks = plan_rotation_blocks(ROTATIONS, None, n_classes=2, rotation_block_size=5, score_real_dtype=np.float64)
    assert blocks.rotations_padded is ROTATIONS
    assert blocks.rotation_log_prior_padded is None


def test_a_per_class_rotation_prior_is_kept_as_it_is_and_padded():
    prior = np.arange(10, dtype=np.float64).reshape(2, 5)
    blocks = plan_rotation_blocks(ROTATIONS, prior, n_classes=2, rotation_block_size=3, score_real_dtype=np.float64)
    assert blocks.rotation_log_prior_padded.dtype == np.float64
    assert_matches(blocks.rotation_log_prior_padded[:, :5], prior)
    assert_matches(blocks.rotation_log_prior_padded[:, 5], [0, 0])


@pytest.mark.parametrize(
    "prior, message",
    [
        (np.zeros(4), r"rotation_log_prior must have shape \(5,\), got \(4,\)"),
        (np.zeros((3, 5)), r"rotation_log_prior must have shape \(5,\) or \(2, 5\), got \(3, 5\)"),
    ],
)
def test_a_rotation_prior_of_the_wrong_shape_is_refused(prior, message):
    with pytest.raises(ValueError, match=message):
        plan_rotation_blocks(ROTATIONS, prior, n_classes=2, rotation_block_size=3, score_real_dtype=np.float32)


def test_the_translation_prior_is_shared_per_image_or_absent_and_of_the_score_dtype():
    assert validated_translation_log_prior(None, n_images=4, n_trans=3, score_real_dtype=np.float32) is None
    shared = validated_translation_log_prior([0.0, -1.0, -2.0], n_images=4, n_trans=3, score_real_dtype=np.float32)
    assert shared.shape == (3,) and shared.dtype == np.float32
    per_image = validated_translation_log_prior(np.zeros((4, 3)), n_images=4, n_trans=3, score_real_dtype=np.float64)
    assert per_image.shape == (4, 3) and per_image.dtype == np.float64


@pytest.mark.parametrize(
    "prior, message",
    [
        (np.zeros(2), r"translation_log_prior must have shape \(3,\), got \(2,\)"),
        (np.zeros((3, 3)), r"translation_log_prior must have shape \(4, 3\) when image-specific, got \(3, 3\)"),
        (np.zeros((4, 3, 1)), "translation_log_prior must be 1D or 2D, got 3 dimensions"),
    ],
)
def test_a_translation_prior_of_the_wrong_shape_is_refused(prior, message):
    with pytest.raises(ValueError, match=message):
        validated_translation_log_prior(prior, n_images=4, n_trans=3, score_real_dtype=np.float32)


# ------------------------------------------------------------------------------------------ the noise spectrum


def test_one_shared_spectrum_has_no_group_table():
    tables = resolve_noise_tables(np.ones(SHAPE[0] * SHAPE[1]), SHAPE, None)
    assert tables.noise_variance_half.shape == (N_HALF,)
    assert tables.noise_table_host is None and tables.image_groups_host is None


def test_a_row_per_optics_group_keeps_the_table_on_the_host_and_each_image_group_as_int32():
    tables = resolve_noise_tables(np.ones((2, SHAPE[0] * SHAPE[1])), SHAPE, np.array([0, 1, 1]))
    assert tables.noise_variance_half.shape == (2, N_HALF)
    assert tables.noise_table_host.shape == (2, N_HALF)
    assert tables.image_groups_host.dtype == np.int32
    assert_matches(tables.image_groups_host, [0, 1, 1])


def test_a_per_group_table_without_the_group_of_each_image_is_refused():
    with pytest.raises(ValueError, match="a per-optics-group noise table needs optics_group_ids"):
        resolve_noise_tables(np.ones((2, SHAPE[0] * SHAPE[1])), SHAPE, None)


# ---------------------------------------------------------------------------------------- blocks and the support


def test_the_blocks_run_class_by_class_with_a_short_tail_block_traced_at_the_block_size():
    # (class, first rotation, real rows, padded rows)
    assert score_blocks(2, 5, 3) == ((0, 0, 3, 3), (0, 3, 2, 3), (1, 0, 3, 3), (1, 3, 2, 3))
    assert score_blocks(1, 4, 4) == ((0, 0, 4, 4),)


def test_each_block_carries_its_class_prior_and_its_slice_of_the_rotation_prior():
    padded = plan_rotation_blocks(
        ROTATIONS, np.arange(5, dtype=np.float32), n_classes=2, rotation_block_size=3, score_real_dtype=np.float32
    ).rotation_log_prior_padded
    blocks = score_blocks(2, 5, 3)
    terms = block_prior_terms(blocks, np.log([0.25, 0.75]), padded, 3)
    assert len(terms) == len(blocks)
    class_priors = [float(prior) for prior, _ in terms]
    assert_matches(class_priors, np.log([0.25, 0.25, 0.75, 0.75]).astype(np.float32))
    assert terms[0][0].dtype == np.float32
    assert_matches(terms[0][1], [0, 1, 2])
    assert_matches(terms[1][1], [3, 4, 0])
    assert all(rotation_prior is None for _, rotation_prior in block_prior_terms(blocks, np.log([0.25, 0.75]), None, 3))


def test_the_exact_weight_order_adds_the_first_class_prior_to_the_rotation_prior():
    padded = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0]])
    prior = exact_order_rotation_prior(np.log([0.5]), padded, 5)
    assert prior.dtype == np.float32
    assert_matches(prior, np.arange(5) + np.log(0.5), rtol=1e-6)
    assert_matches(exact_order_rotation_prior(np.log([0.5]), None, 5), np.full(5, np.log(0.5)), rtol=1e-6)


def test_the_program_statics_are_one_hashable_value_that_equal_settings_share():
    settings = dict(
        n_trans=3, image_shape=(4, 4), volume_shape=(4, 4, 4), float64=False, score_kind="gaussian",
        exact_weight_order=False, return_class_best=True,
    )
    statics = ProgramStatics(**settings)
    assert statics == ProgramStatics(**settings) and hash(statics) == hash(ProgramStatics(**settings))
    assert statics != ProgramStatics(**{**settings, "score_kind": "normalized_cc"})
    assert statics.track_class_second is False and statics.return_values is True
    assert len({statics, ProgramStatics(**settings)}) == 1


@pytest.mark.parametrize(
    "flags, absent",
    [
        (dict(return_class_best=False, return_class_second=False), ("class_best_log_score", "class_hard_assignment")),
        (dict(return_class_best=True, return_class_second=False), ("class_second_best_log_score",)),
        (dict(return_class_best=True, return_class_second=True), ()),
    ],
)
def test_the_arrays_of_a_pass_exist_only_for_the_results_it_returns(flags, absent):
    plan = OutputPlan(
        n_classes=2, n_rot=3, n_trans=2, n_images=4, score_real_dtype=np.float64, collect_significance=True,
        relion_f32_coarse_support_enabled=True, return_relion_f32_normalization=False, **flags,
    )
    outputs = Pass1Outputs.allocate(plan)
    assert outputs.hard_assignment.shape == (4,) and outputs.hard_assignment.dtype == np.int32
    assert outputs.log_evidence.dtype == np.float64 and outputs.normalization_log_z.dtype == np.float64
    assert outputs.class_log_evidence.shape == (2, 4)
    assert outputs.sig_rot_any.shape == (2, 3) and not outputs.sig_rot_any.any()
    assert outputs.relion_f32_sum_weight.dtype == np.float32 and outputs.relion_f32_max_posterior is None
    assert all(getattr(outputs, name) is None for name in absent)
    if flags["return_class_second"]:
        assert outputs.class_second_hard_assignment.shape == (2, 4)
    assert len(outputs.significant_sample_indices) == 2 and len(outputs.significant_sample_indices[0]) == 4


def test_a_pass_that_collects_no_support_has_no_support_rows():
    plan = OutputPlan(
        n_classes=1, n_rot=3, n_trans=2, n_images=4, score_real_dtype=np.float32, collect_significance=False,
        relion_f32_coarse_support_enabled=True, return_relion_f32_normalization=False, return_class_best=False,
        return_class_second=False,
    )
    outputs = Pass1Outputs.allocate(plan)
    assert outputs.significant_sample_indices is None and outputs.relion_f32_sum_weight is None


# ------------------------------------------------------------------------------------------- the route, planned


def _plan(monkeypatch, n_classes=2, **overrides):
    args, kwargs = _significance_call(monkeypatch, n_classes=n_classes)
    kwargs.update(overrides)
    return plan_pass1(Pass1Request(*args, **kwargs)), args, kwargs


def test_a_gaussian_pass_plans_the_gemm_route_on_the_float32_support(monkeypatch):
    plan, _, _ = _plan(monkeypatch)
    route = plan.route
    assert (route.score_kind, route.executed_backend) == ("gaussian", "gemm_macro")
    assert route.float32_support is True
    assert isinstance(route.operand_plan, GaussianOperandPlan)
    assert route.compact_rows is not None and route.tree_rescore_plan is None
    assert route.rotation_block_size == 2 and route.projection_cache_plan is None
    assert plan.output_plan.relion_f32_coarse_support_enabled is True
    assert plan.n_images == 7 and plan.image_batch_size == 3 and plan.current_size == 4


def test_the_environment_switch_to_the_generic_support_route_is_read_once_at_the_route(monkeypatch):
    monkeypatch.setenv("RECOVAR_K1_RELION_F32_COARSE_SUPPORT", "0")
    plan, _, _ = _plan(monkeypatch, n_classes=1)
    assert plan.route.float32_support is False
    assert plan.output_plan.relion_f32_coarse_support_enabled is False
    assert plan.score_program_plan.statics.exact_weight_order is False
    assert plan.support_plan.exact_weight_order is False


@pytest.mark.parametrize("n_classes, exact", [(1, True), (2, False)])
def test_RELIONs_weight_order_is_planned_for_one_class_on_the_float32_route_only(monkeypatch, n_classes, exact):
    plan, _, _ = _plan(monkeypatch, n_classes=n_classes)
    assert plan.score_program_plan.statics.exact_weight_order is exact
    assert plan.support_plan.exact_weight_order is exact
    if exact:
        padded = plan.score_program_plan.rotations_padded
        assert padded.shape[0] == 6
        assert plan.support_plan.exact_rotation_prior.shape == (5,)
        assert_matches(
            plan.support_plan.exact_rotation_prior, np.linspace(0.0, -0.4, 5, dtype=np.float32) + np.log(1.0), rtol=1e-6
        )
    else:
        assert plan.support_plan.exact_rotation_prior is None


def test_the_program_plan_carries_its_blocks_and_the_settings_it_is_compiled_for(monkeypatch):
    plan, _, _ = _plan(monkeypatch, n_classes=2)
    program = plan.score_program_plan
    assert program.blocks == score_blocks(2, 5, 2)
    assert len(program.prior_terms) == len(program.blocks) and program.n_classes == 2
    assert program.statics == ProgramStatics(
        n_trans=3, image_shape=(4, 4), volume_shape=(4, 4, 4), float64=False, score_kind="gaussian",
        exact_weight_order=False, return_class_best=True, track_class_second=False, return_values=True,
    )


def test_a_pass_that_asks_for_the_runner_up_tracks_it_in_the_program(monkeypatch):
    plan, _, _ = _plan(monkeypatch, return_class_second=True)
    assert plan.score_program_plan.statics.track_class_second is True
    assert plan.output_plan.return_class_second is True


def test_a_pass_that_collects_no_support_scores_without_values(monkeypatch):
    plan, _, _ = _plan(monkeypatch, collect_significance=False)
    assert plan.score_program_plan.statics.return_values is False
    assert plan.collect_significance is False


def _cc_kwargs(kwargs):
    kwargs.update(score_mode="normalized_cc", return_class_best=True, adaptive_fraction=1.0, max_significants=1)
    kwargs.pop("rotation_log_prior"), kwargs.pop("translation_log_prior")
    return kwargs


def test_a_normalized_cc_pass_plans_the_exact_cc_route_without_the_float32_support(monkeypatch):
    args, kwargs = _significance_call(monkeypatch, n_classes=1)
    plan = plan_pass1(Pass1Request(*args, **_cc_kwargs(kwargs)))
    route = plan.route
    assert (route.score_kind, route.executed_backend) == ("normalized_cc", "exact_cc_gemm")
    assert route.float32_support is False and route.projection_cache_plan is None
    assert isinstance(route.operand_plan, CcOperandPlan)
    assert route.tree_rescore_plan is None and plan.tree_rescore_enabled is False
    assert plan.score_program_plan.statics.score_kind == "normalized_cc"
    assert plan.score_program_plan.statics.exact_weight_order is False


@pytest.mark.parametrize(
    "overrides, error, message",
    [
        ({"n_classes": 2}, ValueError, "supports K=1 only"),
        ({"return_class_best": False}, ValueError, "requires return_class_best=True"),
        ({}, RuntimeError, "requires the custom CUDA backend"),
    ],
    ids=["two_classes", "without_class_best", "on_a_cpu_backend"],
)
def test_the_tree_rescore_is_refused_where_it_cannot_run(monkeypatch, overrides, error, message):
    args, kwargs = _significance_call(monkeypatch, n_classes=overrides.pop("n_classes", 1))
    kwargs = _cc_kwargs(kwargs)
    kwargs.update(overrides, tree_rescore_max_margin=0.5)
    with pytest.raises(error, match=message):
        plan_pass1(Pass1Request(*args, **kwargs))


def test_the_tail_batch_is_padded_by_default_and_the_environment_can_turn_it_off(monkeypatch):
    plan, _, _ = _plan(monkeypatch)
    assert plan.batch_input_plan.pad_final_image_batch is True
    monkeypatch.setenv("RELAX_COARSE_PAD_FINAL_IMAGE_BATCH", "0")
    unpadded, _, _ = _plan(monkeypatch)
    assert unpadded.batch_input_plan.pad_final_image_batch is False
    explicit, _, _ = _plan(monkeypatch, pad_final_image_batch=True)
    assert explicit.batch_input_plan.pad_final_image_batch is True


# ----------------------------------------------------------------------- what the planner refuses with its resources


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"use_float64_scoring": True}, "float32 scoring and projections"),
        ({"relion_projector_half": None}, "the supplied RELION projector"),
        ({"relion_projector_texture_interp": False}, "texture interpolation of that projector"),
        ({"half_spectrum_scoring": False}, "half-spectrum scoring"),
    ],
    ids=["float64", "no_projector", "no_texture", "full_spectrum"],
)
def test_pass_1_refuses_what_RELIONs_exact_coarse_operands_cannot_score(monkeypatch, overrides, message):
    with pytest.raises(ValueError, match="pass 1 scores RELION's exact coarse operands and needs .*" + message):
        _plan(monkeypatch, **overrides)


def test_several_missing_requirements_are_reported_together(monkeypatch):
    with pytest.raises(ValueError) as refused:
        _plan(monkeypatch, relion_projector_half=None, half_spectrum_scoring=False, use_float64_scoring=True)
    assert str(refused.value).count(",") == 2


def test_a_projector_of_another_class_count_than_the_class_priors_is_refused(monkeypatch):
    from helpers.exact_pass1_harness import coded_class_projectors

    with pytest.raises(ValueError, match="relion_projector_half must have shape"):
        _plan(monkeypatch, n_classes=2, relion_projector_half=coded_class_projectors(3))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rotation_log_prior": np.zeros(4, dtype=np.float32)}, "rotation_log_prior must have shape"),
        ({"translation_log_prior": np.zeros(2, dtype=np.float32)}, "translation_log_prior must have shape"),
        ({"noise_variance": None}, "a per-optics-group noise table needs optics_group_ids"),
    ],
    ids=["rotation_prior", "translation_prior", "noise_rows_without_groups"],
)
def test_the_planner_refuses_priors_and_noise_tables_that_do_not_fit_the_pass(monkeypatch, overrides, message):
    args, kwargs = _significance_call(monkeypatch)
    noise_rows = overrides.pop("noise_variance", False)
    if noise_rows is None:
        args = (args[0], np.ones((2, args[0].image_size), dtype=np.float32), *args[2:])
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        plan_pass1(Pass1Request(*args, **kwargs))
