"""The VDAM gate's statistic (Config.gate_statistic): RELION's amplitude shell means, or the power shell means."""

import argparse
import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.commands import ppca_initial_model
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import (
    FIRST_MOMENT_DECAY,
    FIRST_MOMENT_WEIGHT,
    NoiseTrack,
    advance_noise_track,
    empty_moments,
    full_data_disagreement_scale,
    stochastic_update,
    theta_disagreement_scale,
)

pytestmark = pytest.mark.unit

BOX = 8


def _complex_gaussian(rng, power, size):
    """Circular complex Gaussian samples with E|x|^2 = power."""
    return np.sqrt(power / 2) * (rng.standard_normal(size) + 1j * rng.standard_normal(size))


def _gate_of_one_shell(theta, directions, *, gate_statistic, fudge=1.0):
    """The mean channel's gate of ``stochastic_update`` on first-call moments, all voxels in shell 0."""
    n = theta.shape[0]
    _, _, info = stochastic_update(
        jnp.asarray(theta),
        empty_moments(jnp.asarray(theta)),
        jnp.asarray(directions),
        np.ones((2, n), bool),
        np.zeros(n, np.int32),
        step=0.5,
        fudge=fudge,
        image_size=BOX,
        gate_statistic=gate_statistic,
        disagreement_scale=1.0,
    )
    return float(info["gates"][0, 0])


def test_the_power_gate_follows_the_scalar_recurrence_with_squared_magnitudes():
    n = BOX
    scale = n**2
    value = 0.7 + 0.3j
    first = [0j, 0j]
    second = 1.0
    theta = jnp.asarray([[scale * value]], jnp.complex64)
    moments = empty_moments(theta)
    for iteration, directions in enumerate([(0.3 + 0.2j, -0.1 + 0.4j), (0.1 - 0.2j, 0.5 + 0.1j)]):
        for half in range(2):
            first[half] = directions[half] if iteration == 0 else 0.9 * first[half] + 0.1 * directions[half]
        ratio = abs(directions[1] - directions[0]) ** 2 / (abs(sum(directions) / 2) ** 2 + 1e-12)
        second = 0.999 * second + 0.001 * ratio
        gradient = sum(first) / 2 / (np.sqrt(second) + 1e-12)
        rho = 2 * 1.7 * abs(value) ** 2 / abs(first[0] - first[1]) ** 2
        gate = rho / (1 + rho)
        value += 0.8 * (gate * gradient - (1 - gate) * value)
        theta, moments, info = stochastic_update(
            theta,
            moments,
            jnp.asarray(directions, jnp.complex64).reshape(2, 1, 1) * scale,
            np.ones((2, 1), bool),
            np.zeros(1, np.int32),
            step=0.8,
            fudge=1.7,
            image_size=n,
            gate_statistic="power",
            disagreement_scale=1.0,
        )
        assert_matches(theta[0, 0] / scale, value, rtol=2e-6)
        assert_matches(info["gates"][0, 0], gate, rtol=2e-6)


def test_the_loading_channel_takes_the_mean_square_over_the_pcs():
    """One voxel, q = 2: signal ||W||^2 / q = 25 / 2 and disagreement ||dW||^2 / q = 2 / 2 on the loading channel."""
    theta = jnp.asarray([[0.5, 3.0, 4.0]], jnp.complex64)
    directions = jnp.asarray([[[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]], jnp.complex64)
    fudge = 1.5
    for statistic, signal, disagreement in (("power", 12.5, 1.0), ("amplitude", np.sqrt(12.5), 1.0)):
        _, _, info = stochastic_update(
            theta,
            empty_moments(theta),
            directions,
            np.ones((2, 1), bool),
            np.zeros(1, np.int32),
            step=0.5,
            fudge=fudge,
            image_size=BOX,
            gate_statistic=statistic,
            disagreement_scale=1.0,
        )
        rho = 2 * fudge * signal / disagreement
        assert_matches(info["signal"][0, 1], signal, rtol=1e-6)
        assert_matches(info["disagreement"][0, 1], disagreement, rtol=1e-6)
        assert_matches(info["gates"][0, 1], rho / (1 + rho), rtol=1e-6)


@pytest.mark.parametrize("snr", [0.05, 0.5, 4.0])
def test_the_power_gate_estimates_the_half_map_fsc_and_the_amplitude_gate_does_not(snr):
    """Signal power ``snr`` and unit noise power in each half, so the expected FSC is snr / (1 + snr).

    The two halves' directions are S + N_h with the model theta = S; the FSC is computed from the halves directly. Over
    20 seeds of this construction the largest |power gate - FSC| of the three cases is 4.0e-3 (sampling error of a
    1e5-voxel shell), so the tolerance is 0.01.
    """
    rng = np.random.default_rng(23)
    n = 100_000
    signal = _complex_gaussian(rng, snr, n)
    halves = np.stack([signal + _complex_gaussian(rng, 1.0, n) for _ in range(2)])
    fsc = np.real(np.sum(halves[0] * halves[1].conj())) / np.sqrt(
        np.sum(abs(halves[0]) ** 2) * np.sum(abs(halves[1]) ** 2)
    )
    theta = np.stack([signal, np.zeros(n)], axis=-1).astype(np.complex64)
    directions = np.stack([np.stack([h, np.zeros(n)], axis=-1) for h in halves]).astype(np.complex64)

    power_gate = _gate_of_one_shell(theta, directions, gate_statistic="power")
    amplitude_gate = _gate_of_one_shell(theta, directions, gate_statistic="amplitude")

    assert power_gate == pytest.approx(fsc, abs=0.01)
    assert power_gate == pytest.approx(snr / (1 + snr), abs=0.01)
    # Amplitude shell means make the gate follow about the square root of the power SNR: open below an SNR of 2, tight above.
    assert (1 if snr < 2 else -1) * (amplitude_gate - power_gate) > 0.02


def test_the_gate_noise_reference_is_batch_by_default_and_the_option_takes_only_batch_full_data_or_theta():
    parser = argparse.ArgumentParser()
    ppca_initial_model.add_args(parser)
    required = ["--i", "particles.star", "--particle-diameter", "100", "-o", "out"]
    assert Config().gate_noise_reference == "batch"
    assert parser.parse_args(required).gate_noise_reference == "batch"
    for reference in ("full_data", "theta"):
        assert parser.parse_args([*required, "--gate-noise-reference", reference]).gate_noise_reference == reference
    with pytest.raises(ValueError, match="gate noise reference must be batch, full_data or theta"):
        Config(gate_noise_reference="data")
    with pytest.raises(ValueError, match="momentum SGD has no gate"):
        Config(optimizer="momentum_sgd", gate_noise_reference="full_data")
    assert "gate_noise_reference" not in checkpoint._identity_config(dataclasses.asdict(Config()))
    assert (
        checkpoint._identity_config(dataclasses.asdict(Config(gate_noise_reference="full_data")))[
            "gate_noise_reference"
        ]
        == "full_data"
    )


def test_the_gate_statistic_is_amplitude_by_default_and_the_option_takes_only_amplitude_or_power():
    parser = argparse.ArgumentParser()
    ppca_initial_model.add_args(parser)
    required = ["--i", "particles.star", "--particle-diameter", "100", "-o", "out"]
    assert Config().gate_statistic == "amplitude"
    assert parser.parse_args(required).gate_statistic == "amplitude"
    assert parser.parse_args([*required, "--gate-statistic", "power"]).gate_statistic == "power"
    with pytest.raises(ValueError, match="gate statistic must be amplitude or power"):
        Config(gate_statistic="fsc")
    with pytest.raises(ValueError, match="momentum SGD has no gate"):
        Config(optimizer="momentum_sgd", gate_statistic="power")
    Config(optimizer="momentum_sgd")


def _state():
    n_freq = BOX * BOX * (BOX // 2 + 1)
    theta = jnp.zeros((n_freq, 2), jnp.complex64)
    return State(theta, empty_moments(theta), jnp.ones(BOX // 2 + 1, jnp.float32), 0, np.arange(4), {}, 2.0, 2, {})


def test_the_amplitude_gate_is_no_part_of_a_checkpoints_identity_and_a_resume_cannot_change_the_statistic(tmp_path):
    power = Config(gate_statistic="power")
    assert "gate_statistic" not in checkpoint._identity_config(dataclasses.asdict(Config()))
    assert checkpoint._identity_config(dataclasses.asdict(power))["gate_statistic"] == "power"
    checkpoint.save(tmp_path / "power.npz", _state(), power, {"input": "x"})
    checkpoint.load(tmp_path / "power.npz", power, {"input": "x"})
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load(tmp_path / "power.npz", Config(), {"input": "x"})
    checkpoint.save(tmp_path / "amplitude.npz", _state(), Config(), {"input": "x"})
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load(tmp_path / "amplitude.npz", power, {"input": "x"})


def _write_star(root, count):
    """A RELION particle STAR of ``count`` noise images of ``BOX`` pixels with one optics group."""
    import mrcfile

    from scripts.lib.ppca_training_star import write_training_star

    root.mkdir(parents=True, exist_ok=True)
    images = np.random.default_rng(19).normal(size=(count, BOX, BOX)).astype(np.float32)
    with mrcfile.new(root / "particles.mrcs", overwrite=True) as stack:
        stack.set_data(images)
        stack.voxel_size = 2
    ctf = np.tile([15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1))
    ctf[:, 0] += 100 * np.arange(count)
    return write_training_star(root / "particles.star", "particles.mrcs", ctf, voxel_size=2, box=BOX)


def _run_command(star, output, *extra):
    """``relax ppca_initial_model`` on ``star`` for 8 iterations of a fixed batch of 8 particles."""
    argv = [
        "--i", str(star), "--particle-diameter", "12", "-o", str(output), "--q", "1", "--seed", "11",
        "--iterations", "8", "--stages", "[[1,2,1]]", "--stochastic-batch-size", "8", "--stochastic-all-iterations",
        "--image-batch-size", "4", "--rotation-block-size", "64", "--skip-final-embeddings", *extra,
    ]  # fmt: skip
    ppca_initial_model.main(argv)


def test_the_option_reaches_the_update_a_power_run_scores_iteration_one_as_the_amplitude_run_does(
    monkeypatch, tmp_path
):
    """No gate acts before the first update, so iteration 1 agrees; the gates and the model differ afterwards."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    _run_command(star, tmp_path / "amplitude")
    _run_command(star, tmp_path / "power", "--gate-statistic", "power")
    assert json.loads((tmp_path / "power" / "run.json").read_text())["config"]["gate_statistic"] == "power"
    logs = {
        name: [json.loads(line) for line in open(tmp_path / name / "iterations.jsonl")]
        for name in ("amplitude", "power")
    }
    assert_matches(logs["power"][0]["log_likelihood"], logs["amplitude"][0]["log_likelihood"])
    gates = {name: np.asarray([row["gates"] for row in rows]) for name, rows in logs.items()}
    assert np.isfinite(gates["power"]).all() and np.isfinite(gates["amplitude"]).all()
    assert not np.allclose(gates["power"][-1], gates["amplitude"][-1])
    with (
        np.load(tmp_path / "amplitude" / "checkpoint_0008.npz") as amplitude,
        np.load(tmp_path / "power" / "checkpoint_0008.npz") as power,
    ):
        assert np.isfinite(power["theta"]).all()
        assert not np.allclose(power["theta"], amplitude["theta"])


def test_the_noise_weight_is_the_sum_of_squared_moment_coefficients_over_the_batches():
    """first_t = decay^(t-1) d_1 + sum_i weight * decay^(t-i) d_i, each d_i with noise power 4 / count_i (two halves)."""
    counts = [300] * 7 + [1200, 1200, 50_000, 300]
    track = None
    for count in counts:
        track = advance_noise_track(track, count, 0.5)
    t = len(counts)
    explicit = FIRST_MOMENT_DECAY ** (2 * (t - 1)) * 4 / counts[0] + sum(
        FIRST_MOMENT_WEIGHT**2 * FIRST_MOMENT_DECAY ** (2 * (t - i)) * 4 / counts[i - 1] for i in range(2, t + 1)
    )
    assert_matches(track.weight, explicit, rtol=1e-12)
    assert advance_noise_track(None, 300, 0.5).weight == 4 / 300


def test_for_a_constant_batch_the_scale_is_the_batch_share_over_the_running_average_noise_reduction():
    """(b / N) / e with e = (1 - decay) / (1 + decay) = (1 - decay)^2 / (1 - decay^2), from the first iteration on."""
    track = None
    for _ in range(400):
        track = advance_noise_track(track, 300, 0.5)
    reduction = (1 - FIRST_MOMENT_DECAY) / (1 + FIRST_MOMENT_DECAY)
    assert reduction == pytest.approx(FIRST_MOMENT_WEIGHT**2 / (1 - FIRST_MOMENT_DECAY**2), rel=1e-12)
    assert_matches(full_data_disagreement_scale(track.weight, 100_000), 300 / 100_000 / reduction, rtol=1e-9)
    assert full_data_disagreement_scale(advance_noise_track(None, 300, 0.5).weight, 100_000) == pytest.approx(
        300 / 100_000
    )


@pytest.mark.parametrize(
    "counts",
    [[300] * 150 + [1200] * 20 + [300] * 150, [300] * 150 + [50_000]],
    ids=["batches 300, 1200, 300", "batches of 300 then one all-data iteration"],
)
def test_the_scaled_disagreement_recovers_the_noise_of_two_full_data_half_maps_through_the_update(counts):
    """Halves of noise power sigma^2 * N / b_t in iteration t (sigma^2 the full-data half's), through stochastic_update.

    The disagreement of two full-data half maps is 2 sigma^2. After the schedule the scaled disagreement of the power
    statistic is 2 sigma^2 and the unscaled one 7 to 9 times larger. The mean of 2e4 voxels has a sampling error of 0.7%; over 6
    seeds of this construction the scaled value's largest error is 2.2% (first schedule) and 0.8% (second), so rtol is 0.05.
    """
    n, n_images, sigma2 = 20_000, 50_000, 1.0
    rng = np.random.default_rng(31)
    theta = jnp.zeros((n, 2), jnp.complex64)
    moments, track = empty_moments(theta), None
    for count in counts:
        track = advance_noise_track(track, count, 0.5)
        scale = full_data_disagreement_scale(track.weight, n_images)
        noise = np.zeros((2, n, 2), np.complex64)
        for half in range(2):
            noise[half, :, 0] = _complex_gaussian(rng, sigma2 * n_images / count, n)
        theta, moments, info = stochastic_update(
            theta, moments, jnp.asarray(noise), np.ones((2, n), bool), np.zeros(n, np.int32),
            step=0.5, fudge=1.0, image_size=BOX, gate_statistic="power", disagreement_scale=scale,
        )  # fmt: skip
    assert float(info["disagreement"][0, 0]) == pytest.approx(2 * sigma2, rel=0.05)
    assert float(info["disagreement"][0, 0]) / scale > 3 * 2 * sigma2


def _theta_noise_by_coefficients(counts, steps):
    """Noise power of theta (one particle's units) after the iterations, from each iteration's coefficient in z and theta.

    z = (m_0 + m_1) / 2 follows z = decay z + weight n_t (the first iteration z = n_1) and theta' = (1 - step) theta + step z,
    each iteration's batch noise n_t having power 1 / count_t.
    """
    n = len(counts)
    z, theta = np.zeros(n), np.zeros(n)
    for t, step in enumerate(steps):
        z = FIRST_MOMENT_DECAY * z
        z[t] = 1.0 if t == 0 else FIRST_MOMENT_WEIGHT
        theta = (1 - step) * theta + step * z
    return float(np.sum(theta**2 / np.asarray(counts, float))), float(np.sum(z**2 / np.asarray(counts, float)))


def test_the_theta_noise_recursion_is_the_coefficient_sum_over_the_batches_and_steps():
    rng = np.random.default_rng(41)
    counts = [int(c) for c in rng.choice([300, 300, 1200, 50_000], size=60)]
    steps = rng.uniform(0.01, 0.9, size=60)
    track = None
    for count, step in zip(counts, steps):
        track = advance_noise_track(track, count, float(step))
    theta, moment = _theta_noise_by_coefficients(counts, steps)
    assert_matches(track.theta, theta, rtol=1e-12)
    assert_matches(track.moment, moment, rtol=1e-12)


def test_the_theta_scale_is_the_full_data_scale_when_theta_averages_all_particles():
    track = NoiseTrack(weight=3.7e-3, moment=1e-3, theta=1 / 50_000, cross=2e-4)
    assert_matches(theta_disagreement_scale(track), full_data_disagreement_scale(track.weight, 50_000), rtol=1e-12)


def test_the_theta_scale_through_the_update_recovers_the_noise_theta_carries():
    """One particle's noise power 1, so each half's direction has 2 / count_t; the step falls from 0.5 to 0.02.

    The scaled disagreement of the power statistic is 4 * (theta's noise power) and the simulated theta noise
    (theta' = (1 - step) theta + step z with z the averaged first moment the update returns) has the tracked power. The
    mean of 2e4 voxels has a sampling error of 0.7%; over 6 seeds of this construction the largest error is 0.5% for the
    simulated theta noise and 2.2% for the scaled disagreement, so rtol is 0.05.
    """
    n = 20_000
    counts = [300] * 150 + [1200] * 20 + [300] * 150
    steps = np.linspace(0.5, 0.02, len(counts))
    rng = np.random.default_rng(43)
    theta = jnp.zeros((n, 2), jnp.complex64)
    moments, track, simulated = empty_moments(theta), None, np.zeros(n, np.complex128)
    for count, step in zip(counts, steps):
        track = advance_noise_track(track, count, float(step))
        noise = np.zeros((2, n, 2), np.complex64)
        for half in range(2):
            noise[half, :, 0] = _complex_gaussian(rng, 2.0 / count, n)
        theta, moments, info = stochastic_update(
            theta, moments, jnp.asarray(noise), np.ones((2, n), bool), np.zeros(n, np.int32),
            step=float(step), fudge=1.0, image_size=BOX, gate_statistic="power",
            disagreement_scale=theta_disagreement_scale(track),
        )  # fmt: skip
        averaged = (np.asarray(moments.first[0, :, 0]) + np.asarray(moments.first[1, :, 0])) / 2
        simulated = (1 - step) * simulated + step * averaged
    assert float(np.mean(abs(simulated) ** 2)) == pytest.approx(track.theta, rel=0.05)
    assert float(info["disagreement"][0, 0]) == pytest.approx(4 * track.theta, rel=0.05)


def test_the_amplitude_statistic_takes_the_square_root_of_the_variance_scale():
    rng = np.random.default_rng(37)
    theta = jnp.asarray(_complex_gaussian(rng, 1.0, (50, 3)), jnp.complex64)
    directions = jnp.asarray(_complex_gaussian(rng, 1.0, (2, 50, 3)), jnp.complex64)
    common = dict(step=0.5, fudge=1.0, image_size=BOX, gate_statistic="amplitude")
    runs = [
        stochastic_update(theta, empty_moments(theta), directions, np.ones((2, 50), bool), np.zeros(50, np.int32), disagreement_scale=s, **common)[2]
        for s in (1.0, 0.04)
    ]  # fmt: skip
    assert_matches(runs[1]["disagreement"], runs[0]["disagreement"] * 0.2, rtol=1e-6)


def _rows(output):
    """The rows of a run's iterations.jsonl by iteration number."""
    return {row["iteration"]: row for row in map(json.loads, open(output / "iterations.jsonl"))}


@pytest.mark.parametrize("reference", ["full_data", "theta"])
def test_a_noise_reference_reaches_the_update_and_a_resumed_run_keeps_it(reference, monkeypatch, tmp_path):
    """16 images in batches of 8: every iteration's logged scale is the reference's formula on the explicit sums."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    option = ("--gate-statistic", "power", "--gate-noise-reference", reference)
    _run_command(star, tmp_path / "batch", "--gate-statistic", "power")
    _run_command(star, tmp_path / "straight", *option)
    _run_command(star, tmp_path / "resumed", *option, "--stop-after", "4")
    _run_command(star, tmp_path / "resumed", *option, "--resume", str(tmp_path / "resumed" / "checkpoint_0004.npz"))
    batch, straight, resumed = (_rows(tmp_path / name) for name in ("batch", "straight", "resumed"))

    steps = [straight[t]["step"] for t in range(1, 9)]
    for t in range(1, 9):
        weight = FIRST_MOMENT_DECAY ** (2 * (t - 1)) * 4 / 8 + sum(
            FIRST_MOMENT_WEIGHT**2 * FIRST_MOMENT_DECAY ** (2 * (t - i)) * 4 / 8 for i in range(2, t + 1)
        )
        if reference == "full_data":
            expected = 4 / (16 * weight)
        else:
            expected = 4 * _theta_noise_by_coefficients([8] * t, steps[:t])[0] / weight
        assert_matches(straight[t]["disagreement_scale"], expected, rtol=1e-9)
        assert batch[t]["disagreement_scale"] == 1.0
    assert_matches(straight[1]["log_likelihood"], batch[1]["log_likelihood"])
    assert not np.allclose(straight[8]["gates"], batch[8]["gates"])
    for t in range(5, 9):
        assert_matches(resumed[t]["disagreement_scale"], straight[t]["disagreement_scale"], rtol=1e-12)
        assert_matches(resumed[t]["gates"], straight[t]["gates"])
