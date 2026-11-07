"""The VDAM gate: shell means of the power (the half-map FSC under independent noise) and its noise scale."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.ppca_command_run import rows, run_command, write_star

from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.update import (
    FIRST_MOMENT_DECAY,
    FIRST_MOMENT_WEIGHT,
    advance_noise_weight,
    empty_moments,
    full_data_disagreement_scale,
    noise_reference_images,
    stochastic_update,
)

pytestmark = pytest.mark.unit

BOX = 8


def _complex_gaussian(rng, power, size):
    """Circular complex Gaussian samples with E|x|^2 = power."""
    return np.sqrt(power / 2) * (rng.standard_normal(size) + 1j * rng.standard_normal(size))


def _gate_of_one_shell(theta, directions, *, fudge=1.0):
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
        disagreement_scale=1.0,
    )
    return float(info["gates"][0, 0])


def test_the_loading_channel_takes_the_mean_square_over_the_pcs():
    """One voxel, q = 2: signal ||W||^2 / q = 25 / 2 and disagreement ||dW||^2 / q = 2 / 2 on the loading channel."""
    theta = jnp.asarray([[0.5, 3.0, 4.0]], jnp.complex64)
    directions = jnp.asarray([[[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]], jnp.complex64)
    fudge = 1.5
    _, _, info = stochastic_update(
        theta,
        empty_moments(theta),
        directions,
        np.ones((2, 1), bool),
        np.zeros(1, np.int32),
        step=0.5,
        fudge=fudge,
        image_size=BOX,
        disagreement_scale=1.0,
    )
    rho = 2 * fudge * 12.5 / 1.0
    assert_matches(info["signal"][0, 1], 12.5, rtol=1e-6)
    assert_matches(info["disagreement"][0, 1], 1.0, rtol=1e-6)
    assert_matches(info["gates"][0, 1], rho / (1 + rho), rtol=1e-6)


@pytest.mark.parametrize("snr", [0.05, 0.5, 4.0])
def test_the_gate_estimates_the_half_map_fsc(snr):
    """Signal power ``snr`` and unit noise power in each half, so the expected FSC is snr / (1 + snr).

    The two halves' directions are S + N_h with the model theta = S; the FSC is computed from the halves directly. Over
    20 seeds of this construction the largest |gate - FSC| of the three cases is 4.0e-3 (sampling error of a
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

    gate = _gate_of_one_shell(theta, directions)

    assert gate == pytest.approx(fsc, abs=0.01)
    assert gate == pytest.approx(snr / (1 + snr), abs=0.01)


def test_a_checkpoint_of_the_retired_gate_options_resumes_only_at_the_values_that_remain():
    """The study arms recorded gate_statistic and gate_noise_reference; only power and full_data are the model now."""
    config = dataclasses.asdict(Config())
    recorded = {**config, "gate_statistic": "power", "gate_noise_reference": "full_data"}
    assert checkpoint._identity_config(recorded) == checkpoint._identity_config(config)
    for retired in ({"gate_statistic": "amplitude"}, {"gate_noise_reference": "batch"}):
        assert checkpoint._identity_config({**config, **retired}) != checkpoint._identity_config(config)


def test_the_noise_weight_is_the_sum_of_squared_moment_coefficients_over_the_batches():
    """first_t = decay^(t-1) d_1 + sum_i weight * decay^(t-i) d_i, each d_i with noise power 4 / count_i (two halves)."""
    counts = [300] * 7 + [1200, 1200, 50_000, 300]
    weight = None
    for count in counts:
        weight = advance_noise_weight(weight, count)
    t = len(counts)
    explicit = FIRST_MOMENT_DECAY ** (2 * (t - 1)) * 4 / counts[0] + sum(
        FIRST_MOMENT_WEIGHT**2 * FIRST_MOMENT_DECAY ** (2 * (t - i)) * 4 / counts[i - 1] for i in range(2, t + 1)
    )
    assert_matches(weight, explicit, rtol=1e-12)
    assert advance_noise_weight(None, 300) == 4 / 300


def test_for_a_constant_batch_the_scale_is_the_batch_share_over_the_running_average_noise_reduction():
    """(b / N) / e with e = (1 - decay) / (1 + decay) = (1 - decay)^2 / (1 - decay^2), from the first iteration on."""
    weight = None
    for _ in range(400):
        weight = advance_noise_weight(weight, 300)
    reduction = (1 - FIRST_MOMENT_DECAY) / (1 + FIRST_MOMENT_DECAY)
    assert reduction == pytest.approx(FIRST_MOMENT_WEIGHT**2 / (1 - FIRST_MOMENT_DECAY**2), rel=1e-12)
    assert_matches(full_data_disagreement_scale(weight, 100_000), 300 / 100_000 / reduction, rtol=1e-9)
    assert full_data_disagreement_scale(advance_noise_weight(None, 300), 100_000) == pytest.approx(300 / 100_000)


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
    moments, weight = empty_moments(theta), None
    for count in counts:
        weight = advance_noise_weight(weight, count)
        scale = full_data_disagreement_scale(weight, n_images)
        noise = np.zeros((2, n, 2), np.complex64)
        for half in range(2):
            noise[half, :, 0] = _complex_gaussian(rng, sigma2 * n_images / count, n)
        theta, moments, info = stochastic_update(
            theta, moments, jnp.asarray(noise), np.ones((2, n), bool), np.zeros(n, np.int32),
            step=0.5, fudge=1.0, image_size=BOX, disagreement_scale=scale,
        )  # fmt: skip
    assert float(info["disagreement"][0, 0]) == pytest.approx(2 * sigma2, rel=0.05)
    assert float(info["disagreement"][0, 0]) / scale > 3 * 2 * sigma2


def test_the_scale_reaches_the_update_and_a_resumed_run_keeps_it(monkeypatch, tmp_path):
    """16 images in batches of 8: every iteration's logged scale is (1 / (N * weight)) with the explicit weight sum."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = write_star(tmp_path / "data", 16)
    run_command(star, tmp_path / "straight")
    run_command(star, tmp_path / "resumed", "--stop-after", "4")
    run_command(star, tmp_path / "resumed", "--resume", str(tmp_path / "resumed" / "checkpoint_0004.npz"))
    straight, resumed = (rows(tmp_path / name) for name in ("straight", "resumed"))

    for t in range(1, 9):
        weight = FIRST_MOMENT_DECAY ** (2 * (t - 1)) * 4 / 8 + sum(
            FIRST_MOMENT_WEIGHT**2 * FIRST_MOMENT_DECAY ** (2 * (t - i)) * 4 / 8 for i in range(2, t + 1)
        )
        assert_matches(straight[t]["disagreement_scale"], 4 / (16 * weight), rtol=1e-9)
    assert np.isfinite(np.asarray(straight[8]["gates"])).all()
    for t in range(5, 9):
        assert_matches(resumed[t]["disagreement_scale"], straight[t]["disagreement_scale"], rtol=1e-12)
        assert_matches(resumed[t]["gates"], straight[t]["gates"])


def test_the_seen_reference_counts_the_particles_of_the_iterations_so_far_up_to_the_dataset():
    """``seen`` is min(seen, n_images); ``dataset`` ignores how many were seen."""
    assert noise_reference_images(300, 100_000, "seen") == 300
    assert noise_reference_images(99_999, 100_000, "seen") == 99_999
    assert noise_reference_images(250_000, 100_000, "seen") == 100_000
    assert noise_reference_images(300, 100_000, "dataset") == 100_000
    assert noise_reference_images(250_000, 100_000, "dataset") == 100_000


@pytest.mark.parametrize(
    ("counts", "reference_images"),
    [([300] * 10, 3_000), ([300] * 100, 30_000), ([300] * 150 + [1200] * 20, 50_000)],
    ids=["10 batches of 300", "100 batches of 300", "past the dataset"],
)
def test_the_scaled_disagreement_recovers_the_noise_of_half_maps_of_the_images_seen(counts, reference_images):
    """Halves of noise power v / count per iteration, v = sigma^2 * n_images: the full-data half has sigma^2.

    With the ``seen`` reference the half maps whose noise the scaled disagreement recovers have the particles seen
    so far, at most n_images = 50,000, so the disagreement is 2 sigma^2 * n_images / reference_images. The mean of 2e4
    voxels has a sampling error of 0.7%; over 6 seeds of this construction the largest error is 2.4%, so rtol is 0.05.
    """
    n, n_images, sigma2 = 20_000, 50_000, 1.0
    rng = np.random.default_rng(37)
    theta = jnp.zeros((n, 2), jnp.complex64)
    moments, weight, seen = empty_moments(theta), None, 0
    for count in counts:
        weight = advance_noise_weight(weight, count)
        seen += count
        scale = full_data_disagreement_scale(weight, noise_reference_images(seen, n_images, "seen"))
        noise = np.zeros((2, n, 2), np.complex64)
        for half in range(2):
            noise[half, :, 0] = _complex_gaussian(rng, sigma2 * n_images / count, n)
        theta, moments, info = stochastic_update(
            theta, moments, jnp.asarray(noise), np.ones((2, n), bool), np.zeros(n, np.int32),
            step=0.5, fudge=1.0, image_size=BOX, disagreement_scale=scale,
        )  # fmt: skip
    assert noise_reference_images(seen, n_images, "seen") == reference_images
    assert float(info["disagreement"][0, 0]) == pytest.approx(2 * sigma2 * n_images / reference_images, rel=0.05)


def test_the_seen_reference_reaches_the_update_and_a_resumed_run_keeps_it(monkeypatch, tmp_path):
    """16 images in batches of 8: iteration t's reference is min(8 t, 16) and its scale 4 / (reference * weight)."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = write_star(tmp_path / "data", 16)
    run_command(star, tmp_path / "straight", "--gate-noise-images", "seen")
    run_command(star, tmp_path / "resumed", "--gate-noise-images", "seen", "--stop-after", "4")
    run_command(
        star,
        tmp_path / "resumed",
        "--gate-noise-images",
        "seen",
        "--resume",
        str(tmp_path / "resumed" / "checkpoint_0004.npz"),
    )
    straight, resumed = (rows(tmp_path / name) for name in ("straight", "resumed"))

    for t in range(1, 9):
        weight = FIRST_MOMENT_DECAY ** (2 * (t - 1)) * 4 / 8 + sum(
            FIRST_MOMENT_WEIGHT**2 * FIRST_MOMENT_DECAY ** (2 * (t - i)) * 4 / 8 for i in range(2, t + 1)
        )
        assert straight[t]["gate_noise_images"] == min(8 * t, 16)
        assert_matches(straight[t]["disagreement_scale"], 4 / (min(8 * t, 16) * weight), rtol=1e-9)
    for t in range(5, 9):
        assert resumed[t]["gate_noise_images"] == straight[t]["gate_noise_images"]
        assert_matches(resumed[t]["disagreement_scale"], straight[t]["disagreement_scale"], rtol=1e-12)
        assert_matches(resumed[t]["gates"], straight[t]["gates"])


def test_the_gate_reference_is_part_of_a_checkpoints_identity_only_when_it_is_not_the_default():
    """A checkpoint written before the option resumes at ``dataset``; ``seen`` is another model."""
    config = dataclasses.asdict(Config())
    assert "gate_noise_images" not in checkpoint._identity_config(config)
    assert checkpoint._identity_config({**config, "gate_noise_images": "seen"}) != checkpoint._identity_config(config)
    with pytest.raises(ValueError, match="noise images"):
        Config(gate_noise_images="batch")
