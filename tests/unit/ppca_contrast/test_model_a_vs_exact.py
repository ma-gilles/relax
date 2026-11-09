"""Model A against the exact physical model B at the per-image level (scripts/ppca_contrast/model_a_vs_exact.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "ppca_contrast"))
import model_a_vs_exact as check  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def spread():
    """Contrast sd 0.3 at a loading-to-mean energy ratio of 1, on 400 images and a 201-node grid."""
    return check.compare(1.0, 0.3, n_images=400, n_grid=201)


def test_model_a_contrast_posterior_matches_the_exact_one(spread):
    assert spread["corr_c_A_vs_B"] > 0.995
    assert spread["mean_sd_c_A"] == pytest.approx(spread["mean_sd_c_B"], rel=0.05)
    assert spread["corr_c_A"] == pytest.approx(spread["corr_c_B"], abs=0.005)


def test_unit_contrast_latents_lose_accuracy_that_model_a_recovers_by_dividing_by_the_contrast(spread):
    assert spread["r2_z_A_over_c"] == pytest.approx(spread["r2_z_B"], abs=0.01)
    assert spread["r2_z_A_over_c"] > spread["r2_z_unit"] + 0.03
    assert spread["r2_z_A_scaled"] == pytest.approx(spread["r2_z_unit"], abs=0.01)  # the scale is what the latent lacks


def test_pose_evidence_margins_of_model_a_and_b_agree_and_beat_unit_contrast(spread):
    assert spread["A_margin"] == pytest.approx(spread["B_margin"], rel=0.02)
    assert spread["A_margin"] > spread["unit_margin"]


def test_modelling_an_absent_contrast_spread_changes_nothing():
    row = check.compare(0.3, 0.0, model_sd=0.15, n_images=300, n_grid=121)
    assert row["r2_z_A_over_c"] == pytest.approx(row["r2_z_unit"], abs=0.01)
    assert row["A_margin"] == pytest.approx(row["unit_margin"], rel=0.02)
    assert row["A_pose_correct"] == pytest.approx(row["unit_pose_correct"], abs=0.02)
