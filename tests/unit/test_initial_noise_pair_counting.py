"""The ``initial_noise_pair_counting`` option: the start-up noise spectrum's Hermitian-pair counting.

RELION's start-up noise (``calculateSumOfPowerSpectraAndAverageImage``, ml_optimiser.cpp:2968-2977,
and ``getSpectrum`` of the average image, fftw.cpp:1010-1039) averages the power of every stored
pixel of the FFTW half image, so the pairs of the ``kx = 0`` column (and an even image's Nyquist
column), whose two members are both stored, count twice. Every later sigma2 update counts each
pair once. ``"once"`` makes the start-up spectrum the shell mean over the whole transform. The
oracle below weights the stored half in plain numpy.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.tiny_refinement import record_calls

from relax.refinement import startup_noise
from relax.relion import initial_noise

pytestmark = pytest.mark.unit


def _spectrum_oracle(image, n_shells, counting):
    """Shell mean of |F|^2 over the stored FFTW half (RELION's 1 / (H W) scale).

    ``relion`` counts every stored pixel once; ``once`` gives the columns whose Hermitian mates are
    stored too (kx = 0, and an even width's Nyquist column) the weight 1/2.
    """
    height, width = image.shape
    power = np.abs(np.fft.rfft2(image) / (height * width)) ** 2
    ky = np.fft.fftfreq(height, d=1.0 / height)
    kx = np.arange(width // 2 + 1)
    shell = np.floor(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2) + 0.5).astype(np.int64)
    pair_weight = np.ones(power.shape)
    if counting == "once":
        pair_weight[:, 0] = 0.5
        if width % 2 == 0:
            pair_weight[:, width // 2] = 0.5
    keep = shell < n_shells
    total = np.bincount(shell[keep], weights=(power * pair_weight)[keep], minlength=n_shells)
    return total / np.maximum(np.bincount(shell[keep], weights=pair_weight[keep], minlength=n_shells), 1e-300)


def _image(size, seed):
    rng = np.random.default_rng(seed)
    y = np.arange(size)[:, None]
    # Horizontal stripes put extra power on the kx = 0 column, where the two countings differ.
    return rng.normal(size=(size, size)) + 2.0 * np.cos(2 * np.pi * 2 * y / size)


@pytest.mark.parametrize("counting", ["relion", "once"])
@pytest.mark.parametrize("size", [8, 16, 7, 9])
def test_image_spectrum_follows_the_counting_rule(counting, size):
    image = _image(size, seed=1)
    n_shells = size // 2 + 1
    spectrum = startup_noise._POWER_SPECTRUM[counting](image, n_shells)
    assert_matches(spectrum, _spectrum_oracle(image, n_shells, counting), rtol=1e-12)


def test_default_spectrum_is_relions():
    assert startup_noise._POWER_SPECTRUM["relion"] is initial_noise._radial_power_spectrum
    images = [_image(8, seed) for seed in range(3)]
    kwargs = dict(
        box_size=8, pixel_size=1.0, particle_diameter_ang=8.0, width_mask_edge_px=2, do_zero_mask=False,
        nr_optics_groups=1,
    )
    _, default = initial_noise.compute_avg_unaligned_and_sigma2(iter((0, image) for image in images), **kwargs)
    assert_matches(default[0], _sigma2_oracle(images, 5, "relion"), rtol=1e-12)


def test_relion_overweights_the_zero_column():
    """Power on kx = 0 counts twice in RELION's mean, so the striped shell is higher than the whole-grid mean."""
    image = _image(16, seed=1)
    relion = startup_noise._POWER_SPECTRUM["relion"](image, 9)
    once = startup_noise._POWER_SPECTRUM["once"](image, 9)
    assert relion[2] > 1.3 * once[2]
    # A shell holding only self-paired or doubly stored pixels is unchanged: the DC shell.
    assert_matches(relion[0], once[0], rtol=1e-12)


def _sigma2_oracle(images, n_shells, counting):
    """setSigmaNoiseEstimates: mean image spectrum / 2 minus the average image's spectrum / 2."""
    mean_spectrum = np.mean([_spectrum_oracle(image, n_shells, counting) for image in images], axis=0)
    average = np.mean(images, axis=0)
    return mean_spectrum / 2.0 - _spectrum_oracle(average, n_shells, counting) / 2.0


@pytest.mark.parametrize("counting", ["relion", "once"])
@pytest.mark.parametrize("size", [8, 16])
def test_start_up_sigma2_follows_the_counting_rule(counting, size):
    images = [_image(size, seed) for seed in range(6)]
    n_shells = size // 2 + 1
    average, sigma2 = initial_noise.compute_avg_unaligned_and_sigma2(
        iter((0, image) for image in images),
        box_size=size,
        pixel_size=1.0,
        particle_diameter_ang=float(size),
        width_mask_edge_px=2,
        do_zero_mask=False,
        nr_optics_groups=1,
        power_spectrum=startup_noise._POWER_SPECTRUM[counting],
    )
    assert_matches(average, np.mean(images, axis=0), rtol=1e-12)
    expected = _sigma2_oracle(images, n_shells, counting)
    assert np.all(expected > 0)
    assert_matches(sigma2[0], expected, rtol=1e-12)


def test_start_up_noise_hands_the_estimate_the_counting(monkeypatch):
    """prepare_startup_noise -> estimate_startup_sigma2 -> compute_avg_unaligned_and_sigma2."""
    from types import SimpleNamespace

    received = []

    def compute(image_iter, **kwargs):
        received.append({function: name for name, function in startup_noise._POWER_SPECTRUM.items()}[kwargs["power_spectrum"]])
        return np.zeros((8, 8)), np.full((1, 5), 0.25)

    monkeypatch.setattr(startup_noise, "compute_avg_unaligned_and_sigma2", compute)
    estimates = record_calls(monkeypatch, startup_noise, "estimate_startup_sigma2")
    dataset = SimpleNamespace(grid_size=8, n_units=3, image_source=SimpleNamespace(iter_batches=lambda **_: iter(())))
    for counting in ("relion", "once"):
        startup_noise.prepare_startup_noise(
            dataset,
            source_rows=np.arange(3),
            optics_group_ids=np.ones(3, dtype=np.int64),
            mask_params=(12.0, 3),
            optics_pixel_sizes=np.array([1.25]),
            output_dtype=np.float32,
            pair_counting=counting,
        )
    assert received == ["relion", "once"]
    assert [kwargs["pair_counting"] for _, kwargs in estimates] == ["relion", "once"]
    # The caller names the rule: no default below the command.
    with pytest.raises(TypeError, match="pair_counting"):
        startup_noise.prepare_startup_noise(
            dataset, source_rows=np.arange(3), optics_group_ids=np.ones(3, dtype=np.int64), mask_params=(12.0, 3),
            optics_pixel_sizes=np.array([1.25]), output_dtype=np.float32,
        )
