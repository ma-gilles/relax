"""The settings the contrast-aware inference takes from a checkpoint."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "ppca_contrast"))
from infer_for_suite import config_from_metadata, scorer_order  # noqa: E402

from relax.ppca_initial_model.config import Config  # noqa: E402

pytestmark = pytest.mark.unit


def test_scorer_order_is_at_least_three_and_follows_the_stage_of_the_checkpoint():
    table = [[1, 11, 1], [126, 12, 2], [1626, 24, 3], [2500, 31, 3]]
    assert scorer_order({"stages": table}, 3000) == 3
    assert scorer_order({"stages": table}, 100) == 3  # the scorer never infers below order 3
    assert scorer_order({"stages": [[1, 4, 1], [10, 8, 4]]}, 50) == 4
    assert scorer_order({"stages": [[1, 4, 1]]}, 0) == 3


def test_inference_config_keeps_the_model_settings_and_sets_the_search():
    saved = json.loads(
        json.dumps(
            dataclasses.asdict(
                Config(q=10, optimizer="momentum_sgd", contrast="latent", contrast_sd=0.25, pc_stages=((1, 0), (50, 10)),
                       stochastic_batch_size=300, stochastic_all_iterations=True, balanced_stochastic_halves=True)
            )
        )
    )
    saved["retired_setting"] = 1  # keys the Config no longer has are ignored
    config = config_from_metadata(saved, order=3, radius=31, shift_range=6, shift_step=2)
    assert config.contrast == "latent" and config.contrast_sd == 0.25 and config.q == 10
    assert config.stages == ((1, 31, 3),) and config.pc_stages is None and config.active_pcs(1) == 10
    assert (config.shift_range, config.shift_step, config.oversampling, config.gemm_precision) == (6, 2, 0, "fp32")
    assert config.radius_rule == config.order_rule == "stages" and config.contrast_start == 1
