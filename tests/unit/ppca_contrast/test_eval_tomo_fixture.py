"""The latent scores of the cryo-ET fixture evaluation (scripts/ppca_contrast/eval_tomo_fixture.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "ppca_contrast"))
import eval_tomo_fixture as evaluation  # noqa: E402

pytestmark = pytest.mark.unit


def _clusters(separation: float):
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 60)
    centres = separation * np.eye(3)[labels]
    return centres + rng.normal(size=(180, 3)), labels


def test_separated_clusters_score_high_and_overlapping_ones_low():
    far, labels = _clusters(12.0)
    near, _ = _clusters(0.0)
    assert evaluation.nearest_centroid_accuracy(far, labels) > 0.99
    assert evaluation.kmeans_ari(far, labels) > 0.99
    assert evaluation.between_state_r2(far, labels) > 0.9
    assert evaluation.nearest_centroid_accuracy(near, labels) < 0.5
    assert evaluation.kmeans_ari(near, labels) < 0.1
    assert evaluation.between_state_r2(near, labels) < 0.05


def test_score_run_reads_the_contrast_estimate(tmp_path):
    z, labels = _clusters(8.0)
    contrast = np.linspace(0.5, 1.5, 180)
    order = np.random.default_rng(1).permutation(180)  # the run lists particles in another order
    np.savez(tmp_path / "embeddings.npz", particle_ids=order, z=z[order], contrast=contrast[order], contrast_sd=0.25)
    row = evaluation.score_run(tmp_path, labels, contrast)
    assert row["corr(c)"] == pytest.approx(1.0) and row["sd prior"] == 0.25 and row["nn accuracy"] > 0.99
    np.savez(tmp_path / "embeddings.npz", particle_ids=order, z=z[order])
    assert "corr(c)" not in evaluation.score_run(tmp_path, labels, contrast)
