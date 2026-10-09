"""A Pass1Request refuses what its own fields contradict, before any planning."""

import dataclasses

import numpy as np
import pytest

from relax.scoring.pass1_request import Pass1Request

pytestmark = pytest.mark.unit

N_ROT, N_TRANS = 5, 3


def _request(**overrides):
    """A consistent request over 5 rotations and 3 translations; the dataset is never read by the request."""

    fields = dict(
        class_log_priors=np.zeros(1),
        adaptive_fraction=0.9,
        max_significants=4,
        image_batch_size=2,
        rotation_block_size=3,
        current_size=4,
    )
    fields.update(overrides)
    return Pass1Request(
        object(),
        np.ones(16),
        np.tile(np.eye(3, dtype=np.float32), (N_ROT, 1, 1)),
        np.zeros((N_TRANS, 2), dtype=np.float32),
        **fields,
    )


def test_a_consistent_request_builds_with_the_documented_defaults():
    request = _request()
    assert request.score_mode == "gaussian"
    assert request.collect_significance is True
    assert request.return_class_best is False and request.return_class_second is False
    assert request.tree_rescore_max_margin is None
    assert request.symmetry_label == "C1"


def test_the_request_is_frozen():
    request = _request()
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.image_batch_size = 7


def test_only_the_four_inputs_are_positional():
    with pytest.raises(TypeError):
        Pass1Request(object(), np.ones(16), np.zeros((1, 3, 3)), np.zeros((1, 2)), np.zeros(1))


REFUSALS = [
    pytest.param(
        {"return_class_second": True},
        "return_class_second requires return_class_best",
        id="runner_up_without_best",
    ),
    pytest.param(
        {"return_relion_f32_normalization": True, "collect_significance": False},
        "RELION float32 normalization requires Gaussian float32 significance",
        id="normalization_without_support",
    ),
    pytest.param(
        {"return_relion_f32_normalization": True, "score_mode": "normalized_cc"},
        "RELION float32 normalization requires Gaussian float32 significance",
        id="normalization_of_a_cc_pass",
    ),
    pytest.param(
        {"return_relion_f32_normalization": True, "use_float64_scoring": True},
        "RELION float32 normalization requires Gaussian float32 significance",
        id="normalization_in_float64",
    ),
    pytest.param({"score_mode": "euclidean"}, "score_mode must be 'gaussian' or 'normalized_cc', got 'euclidean'", id="score_mode"),
    pytest.param(
        {"translation_phase_source": np.zeros((2, 2))},
        r"translation_phase_source must match translations: \(2, 2\) != \(3, 2\)",
        id="phase_source_shape",
    ),
    pytest.param({"image_batch_size": 0}, "image_batch_size must be positive", id="batch_size_zero"),
    pytest.param({"image_batch_size": -3}, "image_batch_size must be positive", id="batch_size_negative"),
    pytest.param(
        {"coarse_rotation_ids": np.arange(4)},
        r"coarse_rotation_ids must have shape \(5,\), got \(4,\)",
        id="rotation_ids_shape",
    ),
    pytest.param({"coarse_healpix_order": -1}, "coarse_healpix_order must be non-negative, got -1", id="healpix_order"),
    pytest.param(
        {"relion_projector_half": np.zeros((1, 3, 3, 2))},
        "relion_projector_r_max is required when relion_projector_half is provided",
        id="projector_without_r_max",
    ),
    pytest.param({"tree_rescore_max_margin": -1.0}, "tree_rescore_max_margin must be a finite non-negative float", id="margin_negative"),
    pytest.param({"tree_rescore_max_margin": float("nan")}, "tree_rescore_max_margin must be a finite", id="margin_nan"),
    pytest.param({"tree_rescore_max_margin": float("inf")}, "tree_rescore_max_margin must be a finite", id="margin_inf"),
]


@pytest.mark.parametrize("overrides, message", REFUSALS)
def test_a_request_refuses_what_its_own_fields_contradict(overrides, message):
    with pytest.raises(ValueError, match=message):
        _request(**overrides)


def test_a_non_integer_batch_size_raises_the_type_error_of_operator_index():
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        _request(image_batch_size=2.5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"image_batch_size": np.int64(2)},
        {"tree_rescore_max_margin": 0.0},
        {"coarse_rotation_ids": np.arange(N_ROT)},
        {"coarse_rotation_ids": np.arange(N_ROT).reshape(1, -1)},
        {"coarse_healpix_order": 0},
        {"translation_phase_source": np.ones((N_TRANS, 2))},
        {"relion_projector_half": np.zeros((1, 3, 3, 2)), "relion_projector_r_max": 1},
        {"return_relion_f32_normalization": True},
        {"return_class_best": True, "return_class_second": True},
    ],
    ids=lambda overrides: ",".join(sorted(overrides)),
)
def test_the_boundary_values_the_refusals_leave_open_are_accepted(overrides):
    _request(**overrides)


def test_the_first_contradiction_in_the_fixed_order_is_the_one_reported():
    """A request with several defects reports the first of: runner-up, normalization, score mode, phase source, batch size."""

    with pytest.raises(ValueError, match="return_class_second requires"):
        _request(return_class_second=True, score_mode="euclidean", image_batch_size=0)
    with pytest.raises(ValueError, match="score_mode must be"):
        _request(score_mode="euclidean", image_batch_size=0)
    with pytest.raises(ValueError, match="image_batch_size must be positive"):
        _request(image_batch_size=0, coarse_healpix_order=-1)
