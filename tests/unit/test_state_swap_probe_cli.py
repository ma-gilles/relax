"""CLI and ordering contract for the resident-state swap diagnostic."""

from types import SimpleNamespace

import numpy as np
import pytest

from relax.helpers.orientation_priors import DirectionPrior
from relax.parity.state_swap_probe import (
    _STATE_SWAP_VARIANT_COMPONENTS,
    REQUIRED_STATE_SWAP_REPLAY_KEYS,
    build_state_swap_probe,
    state_swap_probe_loop_index,
    state_swap_variant_choices,
    validate_state_swap_probe_application,
)
from relax.parity.state_swap_runtime import (
    _apply_state_swap_probe,
    _scale_state_swap_reference_maps,
    _snapshot_state_swap_inputs,
)
from relax.refinement.half_inputs import initialize_halfsets
from relax.refinement.mean_helpers import ReferenceModel
from relax.refinement.noise_updates import NoiseModel

pytestmark = pytest.mark.unit


def _parse_state_swap_args(*tokens):
    from relax.refinement.command_options import parse_refinement_args

    return parse_refinement_args(["--data_dir", "unused", "--output", "unused", *tokens])


def _loop_index(args, *, init_relion_iteration=0, max_iter=5):
    return state_swap_probe_loop_index(
        target_relion_iteration=args.state_swap_target_relion_iteration,
        variant=args.state_swap_variant,
        replay_relion_references=args.state_swap_replay_relion_references,
        init_relion_iteration=init_relion_iteration,
        max_iter=max_iter,
    )


def _complete_replay_override():
    return {key: object() for key in REQUIRED_STATE_SWAP_REPLAY_KEYS}


def test_state_swap_cli_default_is_inert():
    args = _parse_state_swap_args()

    assert _loop_index(args) is None
    assert build_state_swap_probe(
        target_relion_iteration=args.state_swap_target_relion_iteration,
        variant=args.state_swap_variant,
        replay_relion_references=args.state_swap_replay_relion_references,
        init_relion_iteration=0,
        max_iter=5,
        replay_iteration_overrides=None,
    ) is None
    validate_state_swap_probe_application(None, [])


def test_state_swap_cli_builds_case20_iteration5_maps_noise_probe():
    args = _parse_state_swap_args(
        "--state-swap-target-relion-iteration",
        "5",
        "--state-swap-variant",
        "recovar_maps_tau2_noise",
        "--state-swap-replay-relion-references",
    )
    replay_overrides = [None, None, None, None, _complete_replay_override()]

    probe = build_state_swap_probe(
        target_relion_iteration=args.state_swap_target_relion_iteration,
        variant=args.state_swap_variant,
        replay_relion_references=args.state_swap_replay_relion_references,
        init_relion_iteration=0,
        max_iter=5,
        replay_iteration_overrides=replay_overrides,
    )

    assert probe == {
        "iteration": 4,
        "target_relion_iteration": 5,
        "variant": "recovar_maps_tau2_noise",
        "replay_relion_references": True,
        "replay_override_keys": tuple(sorted(REQUIRED_STATE_SWAP_REPLAY_KEYS)),
        "required_replay_override_keys": tuple(sorted(REQUIRED_STATE_SWAP_REPLAY_KEYS)),
    }
    validate_state_swap_probe_application(probe, [5])


@pytest.mark.parametrize(
    ("physical_target", "expected_loop_index"),
    [(4, 0), (5, 1), (6, 2)],
)
def test_state_swap_physical_iteration_mapping_handles_nonzero_start(
    physical_target,
    expected_loop_index,
):
    args = _parse_state_swap_args(
        "--state-swap-target-relion-iteration",
        str(physical_target),
        "--state-swap-variant",
        "all_relion",
        "--state-swap-replay-relion-references",
    )

    assert _loop_index(args, init_relion_iteration=3, max_iter=3) == expected_loop_index


@pytest.mark.parametrize(
    ("tokens", "match"),
    [
        (("--state-swap-target-relion-iteration", "5"), "must be provided together"),
        (("--state-swap-variant", "all_relion"), "must be provided together"),
        (
            (
                "--state-swap-target-relion-iteration",
                "5",
                "--state-swap-variant",
                "all_relion",
            ),
            "require --state-swap-replay-relion-references",
        ),
    ],
)
def test_state_swap_cli_rejects_incomplete_boundary_contract(tokens, match):
    args = _parse_state_swap_args(*tokens)

    with pytest.raises(ValueError, match=match):
        _loop_index(args)


@pytest.mark.parametrize("target", [3, 6])
def test_state_swap_cli_rejects_target_outside_emitted_iterations(target):
    args = _parse_state_swap_args(
        "--state-swap-target-relion-iteration",
        str(target),
        "--state-swap-variant",
        "recovar_tau2_only",
        "--state-swap-replay-relion-references",
    )

    with pytest.raises(ValueError, match="expected 4..5"):
        _loop_index(args, init_relion_iteration=3, max_iter=2)


def test_state_swap_cli_choices_match_iteration_loop_variants():
    assert state_swap_variant_choices() == tuple(sorted(_STATE_SWAP_VARIANT_COMPONENTS))
    args = _parse_state_swap_args(
        "--state-swap-target-relion-iteration",
        "2",
        "--state-swap-variant",
        "not_a_variant",
        "--state-swap-replay-relion-references",
    )

    with pytest.raises(ValueError, match="unknown state-swap variant"):
        _loop_index(args, max_iter=2)


@pytest.mark.parametrize(
    "replay_overrides",
    [None, [None, None], [None, {}]],
)
def test_state_swap_cli_requires_nonempty_replay_context_at_target(replay_overrides):
    with pytest.raises(ValueError, match="replay"):
        build_state_swap_probe(
            target_relion_iteration=2,
            variant="recovar_tau2_only",
            replay_relion_references=True,
            init_relion_iteration=0,
            max_iter=2,
            replay_iteration_overrides=replay_overrides,
        )


def test_state_swap_cli_requires_complete_target_replay_context():
    incomplete = _complete_replay_override()
    del incomplete["mean_variance"]

    with pytest.raises(ValueError, match="missing.*mean_variance"):
        build_state_swap_probe(
            target_relion_iteration=2,
            variant="recovar_tau2_only",
            replay_relion_references=True,
            init_relion_iteration=0,
            max_iter=2,
            replay_iteration_overrides=[None, incomplete],
        )


def test_state_swap_cli_requires_exact_scoring_scale_oracle():
    incomplete = _complete_replay_override()
    del incomplete["scoring_scale_corrections"]

    with pytest.raises(ValueError, match="missing.*scoring_scale_corrections"):
        build_state_swap_probe(
            target_relion_iteration=2,
            variant="all_relion",
            replay_relion_references=True,
            init_relion_iteration=0,
            max_iter=2,
            replay_iteration_overrides=[None, incomplete],
        )


@pytest.mark.parametrize("applied", [None, [], [4], [5, 5]])
def test_state_swap_application_telemetry_fails_closed(applied):
    probe = {
        "iteration": 4,
        "target_relion_iteration": 5,
        "variant": "all_relion",
        "replay_relion_references": True,
    }

    with pytest.raises(ValueError, match="application mismatch"):
        validate_state_swap_probe_application(probe, applied)


def test_full_runner_propagates_and_serializes_state_swap_probe(monkeypatch, tmp_path):
    """The command builds the probe from the replayed state and hands it to the controller."""
    from helpers.tiny_main import controller_inputs, write_tiny_data_dir
    from helpers.tiny_refinement import CallTrace, write_replay_dir

    from relax.parity import relion_replay
    from relax.parity.state_swap_probe import REQUIRED_STATE_SWAP_REPLAY_KEYS
    from relax.refinement import full_refinement

    def overrides(relion_dir, half1_rows, half2_rows, max_iter, **kwargs):
        complete = dict.fromkeys(REQUIRED_STATE_SWAP_REPLAY_KEYS, object())
        return [complete for _ in range(int(max_iter) + 1)]

    monkeypatch.setattr(relion_replay, "_build_replay_iteration_overrides", overrides)
    trace = CallTrace(monkeypatch).wrap(full_refinement, "build_state_swap_probe", "probe")
    data = write_tiny_data_dir(tmp_path / "data", extra_columns={"rlnRandomSubset": np.arange(12) % 2 + 1})
    inputs = controller_inputs(
        monkeypatch, tmp_path / "swap", "refine", "--max_iter", "3",
        "--perturb_replay_relion_dir", write_replay_dir(tmp_path / "relion", max_iter=3),
        "--relion_half_sets", "<DATA>/particles.star", "--state-swap-target-relion-iteration", "2",
        "--state-swap-variant", "recovar_direction_prior", "--state-swap-replay-relion-references", data=data,
    )
    (probe,) = trace.calls("probe")
    assert inputs["source"].relion_replay.state_swap_probe is probe.result
    assert (probe.result["target_relion_iteration"], probe.result["iteration"]) == (2, 1)


def test_archive_carries_empty_state_swap_probe_fields(monkeypatch, tmp_path):
    from helpers.tiny_main import run_tiny_main

    output = run_tiny_main(monkeypatch, tmp_path, "refine", "--max_iter", "1")
    with np.load(output / "refinement_results.npz", allow_pickle=True) as archive:
        assert int(archive["state_swap_probe_target_relion_iteration"]) == -1
        assert int(archive["state_swap_probe_loop_index"]) == -1
        assert str(archive["state_swap_probe_variant"]) == ""
        for field in ("state_swap_probe_replay_relion_references", "state_swap_probe_applied_relion_iterations",
                      "state_swap_probe_replay_override_keys", "state_swap_probe_required_replay_override_keys"):
            assert field in archive.files


def _direction_prior_pair(seed, order=2):
    rng = np.random.default_rng(seed)
    pair = []
    for _ in range(2):
        values = rng.random(12 * 4**order) + 0.05
        pair.append(values / values.sum())
    return pair


def _state_swap_run(monkeypatch, trace):
    """A K=1 run whose second iteration is the state-swap target, restoring the run's own direction prior."""
    from helpers.tiny_refinement import run_tiny_refinement

    from relax.parity.relion_replay_source import RelionReplay

    run_tiny_refinement(
        monkeypatch, max_iter=3, final_after_max_iter=False, parity=dict(low_resol_join_halves_angstrom=0.0),
        relion_replay=RelionReplay(
            replay_iteration_overrides=[
                None if index == 0 else {"direction_prior": _direction_prior_pair(200 + index)} for index in range(3)
            ],
            state_swap_probe={"iteration": 1, "variant": "recovar_direction_prior"},
        ),
    )
    return trace


def test_relion_references_are_applied_before_state_restoration(monkeypatch):
    from helpers.tiny_refinement import CallTrace

    from relax.parity import relion_replay_source

    trace = CallTrace(monkeypatch)
    trace.wrap(relion_replay_source, "replay_k1_relion_references", "references")
    trace.wrap(relion_replay_source, "_apply_state_swap_probe", "swap")
    _state_swap_run(monkeypatch, trace)
    assert trace.labels() == ["references", "references", "swap", "references"]


def test_state_swap_snapshot_is_bounded_to_target_iteration(monkeypatch):
    """Only the target iteration snapshots the run's own state, before its replayed state."""
    from helpers.tiny_refinement import CallTrace

    from relax.parity import relion_replay_source
    from relax.parity.relion_replay_source import RelionReplaySource

    trace = CallTrace(monkeypatch)
    trace.wrap(RelionReplaySource, "numbered_state", "replay")
    trace.wrap(relion_replay_source, "_snapshot_state_swap_inputs", "snapshot")
    trace.wrap(relion_replay_source, "_apply_state_swap_probe", "swap")
    _state_swap_run(monkeypatch, trace)
    assert trace.labels() == ["replay", "snapshot", "replay", "swap", "replay"]
    snapshot = trace.calls("snapshot")[0].result
    assert trace.calls("swap")[0].kwargs["recovar_snapshot"] is snapshot


def test_sigma_offset_state_swap_preserves_asymmetric_half_values():
    state = SimpleNamespace(marker="resident")
    half_inputs = initialize_halfsets(
        (None, None),
        image_corrections=[np.array([1.0]), np.array([2.0])],
        scale_corrections=[np.array([3.0]), np.array([4.0])],
        previous_best_translations=[np.zeros((1, 2)), np.ones((1, 2))],
        previous_best_rotation_eulers=[np.zeros((1, 3)), np.ones((1, 3))],
    )
    direction_priors = [DirectionPrior(np.array([0.4]), 3), DirectionPrior(np.array([0.6]), 3)]
    snapshot_tau2 = np.array([5.0])
    snapshot = _snapshot_state_swap_inputs(
        state=state,
        cs=52,
        reference_model=ReferenceModel(
            maps=[np.array([1.0]), np.array([2.0])], tau2=snapshot_tau2,
            tau2_per_half=[snapshot_tau2, snapshot_tau2],
        ),
        noise_model=NoiseModel(
            variance_per_half=[np.array([6.0]), np.array([7.0])],
            average_variance=np.array([6.5]),
            radial_per_half=[np.array([8.0]), np.array([9.0])],
            average_radial=np.array([8.5]),
        ),
        relion_half_inputs=half_inputs,
        previous_best_rotations=[np.eye(3)[None], np.eye(3)[None]],
        current_sigma_offset_angstrom=3.0,
        current_sigma_offset_angstrom_per_half=[2.0, 4.0],
        direction_priors=direction_priors,
    )

    direction_priors = [DirectionPrior(np.array([0.5]), 4), DirectionPrior(np.array([0.5]), 4)]
    current_tau2 = np.array([50.0])
    restored = _apply_state_swap_probe(
        probe={"iteration": 6, "variant": "recovar_sigma_offset"},
        iteration=6,
        recovar_snapshot=snapshot,
        state=state,
        cs=52,
        volume_shape=(1, 1, 1),
        reference_model=ReferenceModel(
            maps=[np.array([10.0]), np.array([20.0])], tau2=current_tau2,
            tau2_per_half=[current_tau2, current_tau2],
        ),
        noise_model=NoiseModel(
            variance_per_half=[np.array([60.0]), np.array([70.0])],
            average_variance=np.array([65.0]),
            radial_per_half=[np.array([80.0]), np.array([90.0])],
            average_radial=np.array([85.0]),
        ),
        relion_half_inputs=half_inputs,
        previous_best_rotations=[np.eye(3)[None], np.eye(3)[None]],
        current_sigma_offset_angstrom=3.1,
        current_sigma_offset_angstrom_per_half=[2.9, 3.3],
        direction_priors=direction_priors,
    )

    restored_sigma = restored.current_sigma_offset_angstrom
    restored_sigma_per_half = restored.current_sigma_offset_angstrom_per_half
    assert restored_sigma == pytest.approx(3.0)
    assert restored_sigma_per_half == pytest.approx([2.0, 4.0])
    restored_sigma_per_half[0] = 99.0
    assert snapshot["current_sigma_offset_angstrom_per_half"] == pytest.approx([2.0, 4.0])


def test_state_swap_global_map_scaling_recovers_scalar_target():
    source = [
        np.arange(1, 9, dtype=np.float64).astype(np.complex128),
        (np.arange(1, 9, dtype=np.float64) * (1.0 + 2.0j)).astype(np.complex128),
    ]
    target = [1.25 * source[0], 0.75 * source[1]]

    scaled, summaries = _scale_state_swap_reference_maps(
        source,
        target,
        mode="global",
        volume_shape=(2, 2, 2),
    )

    np.testing.assert_allclose(scaled[0], target[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaled[1], target[1], rtol=1e-12, atol=1e-12)
    assert [summary["scale_min"] for summary in summaries] == pytest.approx(
        [1.25, 0.75]
    )
    assert [summary["relative_l2_after"] for summary in summaries] == pytest.approx(
        [0.0, 0.0],
        abs=1e-15,
    )


def test_state_swap_shell_map_scaling_recovers_shell_amplitudes_and_preserves_phase():
    volume_shape = (4, 4, 4)
    frequencies = np.fft.fftfreq(4) * 4
    grids = np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    shell_labels = np.rint(np.sqrt(sum(grid * grid for grid in grids))).astype(
        np.int32
    ).reshape(-1)
    source = np.exp(1j * np.linspace(0.1, 2.5, shell_labels.size))
    shell_scales = 0.8 + 0.15 * shell_labels
    target = source * shell_scales

    scaled, summaries = _scale_state_swap_reference_maps(
        [source],
        [target],
        mode="shell",
        volume_shape=volume_shape,
    )

    np.testing.assert_allclose(scaled[0], target, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.angle(scaled[0]),
        np.angle(source),
        rtol=0.0,
        atol=1e-12,
    )
    assert summaries[0]["scale_min"] == pytest.approx(float(np.min(shell_scales)))
    assert summaries[0]["scale_max"] == pytest.approx(float(np.max(shell_scales)))
