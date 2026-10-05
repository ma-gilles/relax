"""Mid-trajectory replay noise semantics: RELION continuation versus uninterrupted.

A replay that starts after RELION iteration N > 0 either reproduces a RELION
``--continue`` restart (MPI initialisation scores both halves with the half-1
sigma2_noise) or an uninterrupted trajectory (each half keeps its own spectrum).
Fixed-state comparisons against an uninterrupted RELION reference need the
latter; before this option the CLI could only request the former.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import command_options
from relax.refinement import full_refinement as run_full_refinement

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


def _slot0_noise(tmp_path, *, semantics, init_relion_iteration):
    broadcast = run_full_refinement._replay_process_start_noise_broadcast(
        semantics,
        init_relion_iteration,
        tmp_path,
    )
    overrides = run_full_refinement.relion_replay._build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=0,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=False,
        init_relion_iteration=init_relion_iteration,
        process_start_noise_broadcast=broadcast,
    )
    return overrides[0]["noise_variance"]


def test_replay_noise_semantics_default_is_continuation():
    args = command_options.parse_refinement_args(["--data_dir", "data", "--output", "out"])
    assert args.replay_noise_semantics == "continuation"


def test_uninterrupted_replay_keeps_half_specific_slot0_noise(tmp_path):
    _write_iteration(tmp_path, 10, [1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0])

    continuation_h1, continuation_h2 = _slot0_noise(tmp_path, semantics="continuation", init_relion_iteration=10)
    assert_matches(continuation_h2, continuation_h1)

    uninterrupted_h1, uninterrupted_h2 = _slot0_noise(tmp_path, semantics="uninterrupted", init_relion_iteration=10)
    assert float(np.min(uninterrupted_h1)) == pytest.approx(1.0 * 8**4)
    assert float(np.min(uninterrupted_h2)) == pytest.approx(6.0 * 8**4)
    assert_matches(uninterrupted_h1, continuation_h1)


def test_replay_keeps_one_noise_row_per_optics_group(tmp_path):
    n_groups, n_shells = 8, 5
    half1 = np.arange(1, n_groups + 1, dtype=np.float64)[:, None] * np.ones(n_shells)
    _write_iteration(tmp_path, 10, half1, 100.0 + half1)

    uninterrupted_h1, uninterrupted_h2 = _slot0_noise(tmp_path, semantics="uninterrupted", init_relion_iteration=10)
    assert uninterrupted_h1.shape == (n_groups, 8 * 8)
    assert uninterrupted_h2.shape == (n_groups, 8 * 8)
    # A constant spectrum per group expands to a constant pixel row, scaled by ds_grid**4 as for one group.
    assert_matches(uninterrupted_h1, np.repeat(half1[:, :1] * 8**4, 8 * 8, axis=1).astype(uninterrupted_h1.dtype))
    assert_matches(uninterrupted_h2, np.repeat((100.0 + half1[:, :1]) * 8**4, 8 * 8, axis=1).astype(uninterrupted_h2.dtype))

    continuation_h1, continuation_h2 = _slot0_noise(tmp_path, semantics="continuation", init_relion_iteration=10)
    assert_matches(continuation_h2, continuation_h1)
    assert_matches(continuation_h1, uninterrupted_h1)


@pytest.mark.parametrize(
    ("init_relion_iteration", "replay_dir"),
    [(0, "relion"), (10, None)],
)
def test_uninterrupted_replay_requires_a_mid_trajectory_replay(init_relion_iteration, replay_dir):
    with pytest.raises(ValueError, match="requires --perturb_replay_relion_dir"):
        run_full_refinement._replay_process_start_noise_broadcast(
            "uninterrupted",
            init_relion_iteration,
            replay_dir,
        )


@pytest.mark.parametrize("semantics", ["continuation", "uninterrupted"])
def test_cli_passes_selected_noise_semantics_to_replay_overrides(monkeypatch, tmp_path, semantics):
    """The STAR replay builds its overrides with the process-start noise broadcast the semantics select."""
    from helpers.tiny_main import controller_inputs, write_tiny_data_dir
    from helpers.tiny_refinement import write_replay_dir

    from relax.diagnostics import relion_replay

    broadcasts = []

    def build(relion_dir, half1_rows, half2_rows, max_iter, **kwargs):
        broadcasts.append(kwargs["process_start_noise_broadcast"])
        return [None] * (int(max_iter) + 1)

    monkeypatch.setattr(relion_replay, "_build_replay_iteration_overrides", build)
    data = write_tiny_data_dir(tmp_path / "data", extra_columns={"rlnRandomSubset": np.arange(12) % 2 + 1})
    init_iteration = 0 if semantics == "continuation" else 1
    replay_dir = write_replay_dir(tmp_path / "relion", max_iter=3)
    expected = run_full_refinement._replay_process_start_noise_broadcast(semantics, init_iteration, replay_dir)
    controller_inputs(
        monkeypatch, tmp_path, "refine", "--max_iter", "2", "--replay-noise-semantics", semantics,
        "--init_relion_iteration", str(init_iteration), "--perturb_replay_relion_dir", replay_dir,
        "--relion_half_sets", "<DATA>/particles.star", data=data,
    )
    assert broadcasts == [expected]
