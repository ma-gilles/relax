"""Mid-trajectory replay noise: slot 0 reproduces a RELION ``--continue`` restart, whose MPI initialisation
scores both halves with the half-1 sigma2_noise, with one noise row per optics group."""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.parity import replay_inputs

pytestmark = pytest.mark.unit


def _write_iteration(tmp_path, iteration, half1_noise, half2_noise):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnNormCorrection": [1.0, 1.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    starfile.write({"particles": particles}, tmp_path / f"run_it{iteration:03d}_data.star")
    for half, noise in ((1, half1_noise), (2, half2_noise)):
        # One spectrum, or one row per optics group (subtomograms: one group per tilt).
        groups = np.atleast_2d(np.asarray(noise, dtype=np.float64))
        starfile.write(
            {
                "model_general": pd.DataFrame({"rlnNormCorrectionAverage": [1.0], "rlnSigmaOffsetsAngst": [2.0]}),
                **{
                    f"model_optics_group_{g + 1}": pd.DataFrame({"rlnSigma2Noise": row})
                    for g, row in enumerate(groups)
                },
                "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
            },
            tmp_path / f"run_it{iteration:03d}_half{half}_model.star",
        )


def _slot0_noise(tmp_path, *, init_relion_iteration):
    overrides = replay_inputs._build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=0,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=False,
        init_relion_iteration=init_relion_iteration,
    )
    return overrides[0]["noise_variance"]


def test_replay_keeps_one_noise_row_per_optics_group(tmp_path):
    n_groups, n_shells = 8, 5
    half1 = np.arange(1, n_groups + 1, dtype=np.float64)[:, None] * np.ones(n_shells)
    _write_iteration(tmp_path, 10, half1, 100.0 + half1)

    noise_h1, noise_h2 = _slot0_noise(tmp_path, init_relion_iteration=10)
    assert noise_h1.shape == (n_groups, 8 * 8)
    # A constant spectrum per group expands to a constant pixel row, scaled by ds_grid**4 as for one group;
    # slot 0 is RELION's process start, which scores both halves with the half-1 spectrum.
    assert_matches(noise_h1, np.repeat(half1[:, :1] * 8**4, 8 * 8, axis=1).astype(noise_h1.dtype))
    assert_matches(noise_h2, noise_h1)
