"""RELION's reference mask reading and solvent-corrected FSC (relax/reconstruction/solvent_mask.py).

The references here are written from RELION's documented rule with explicit loops in float64:
getFSC labels the FFTW half voxel (kp, ip, jp) with ROUND(sqrt(kp^2 + ip^2 + jp^2)) and keeps shells
below N/2 + 1 (fftw.cpp:481-512); randomize_at is the first shell i > 0 with unmasked FSC < 0.8; the
true FSC is the masked FSC below randomize_at + 2 and (masked - random) / (1 - random), or 0 when
random > masked, above it, then 0 beyond current_size / 2 (ml_optimiser_mpi.cpp:3340-3396). The random
phases are rnd_unif(0, 2 PI) draws of one glibc stream, one per FFTW half voxel at |k| >= randomize_at in
storage order, half 1 then half 2 (fftw.cpp:436-458). The oracle test replays RELION's own runs.
"""

import math

import mrcfile
import numpy as np
import pytest
from helpers.em_fixtures import fixture_dir
from helpers.float_compare import assert_matches
from helpers.reconstruction_settings import reconstruction_settings

from relax.numerics import relion_random
from relax.reconstruction import solvent_mask
from relax.refinement import map_postprocess
from relax.refinement.refinement_options import ReconstructionPrograms

pytestmark = pytest.mark.unit


def _loop_fsc(map1, map2):
    n = map1.shape[0]
    ft1, ft2 = np.fft.rfftn(map1), np.fft.rfftn(map2)
    num, d1, d2 = np.zeros(n // 2 + 1), np.zeros(n // 2 + 1), np.zeros(n // 2 + 1)
    for k in range(n):
        kp = k if k < n // 2 + 1 else k - n
        for i in range(n):
            ip = i if i < n // 2 + 1 else i - n
            for j in range(n // 2 + 1):
                r = math.sqrt(kp * kp + ip * ip + j * j)
                idx = int(r + 0.5)
                if idx >= n // 2 + 1:
                    continue
                z1, z2 = ft1[k, i, j], ft2[k, i, j]
                num[idx] += (z1.conjugate() * z2).real
                d1[idx] += abs(z1) ** 2
                d2[idx] += abs(z2) ** 2
    return num / np.sqrt(d1 * d2)


def _loop_randomize(volume, shell, generator):
    n = volume.shape[0]
    ft = np.fft.rfftn(volume)
    for k in range(n):
        kp = k if k < n // 2 + 1 else k - n
        for i in range(n):
            ip = i if i < n // 2 + 1 else i - n
            for j in range(n // 2 + 1):
                if kp * kp + ip * ip + j * j >= shell * shell:
                    mag = abs(ft[k, i, j])
                    phase = float(relion_random.rnd_unif(generator, 0.0, 2.0 * math.pi))
                    ft[k, i, j] = complex(mag * math.cos(phase), mag * math.sin(phase))
    return np.fft.irfftn(ft, s=volume.shape, axes=(0, 1, 2))


def _maps(n=12, seed=0, noise=0.6):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, n, n))
    # A smooth common signal and independent noise: high FSC at low shells, low at high ones.
    signal = np.fft.irfftn(
        np.fft.rfftn(signal) * np.exp(-0.15 * solvent_mask._fftw_shells(n)[1]), s=(n,) * 3, axes=(0, 1, 2)
    )
    signal /= signal.std()
    half1 = signal + noise * rng.normal(size=signal.shape)
    half2 = signal + noise * rng.normal(size=signal.shape)
    grid = np.indices((n, n, n)) - n // 2
    mask = np.clip(1.5 - np.sqrt((grid**2).sum(0)) / 3.0, 0.0, 1.0)
    return half1, half2, mask


def test_fsc_matches_the_loop_reference():
    half1, half2, _ = _maps()
    np.testing.assert_allclose(
        solvent_mask.real_space_fsc(half1, half2), _loop_fsc(half1, half2), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("current_size", [12, 8])
def test_corrected_fsc_matches_the_rule(current_size):
    half1, half2, mask = _maps()
    stream = relion_random.GlibcRand(7)
    stream.rand_array(5)  # a stream already advanced, as after earlier iterations
    got, details = solvent_mask.solvent_corrected_fsc(half1, half2, mask, current_size=current_size, stream=stream)

    unmasked = _loop_fsc(half1, half2)
    randomize_at = next(i for i in range(1, unmasked.size) if unmasked[i] < 0.8)
    assert details["randomize_at"] == randomize_at
    masked = _loop_fsc(half1 * mask, half2 * mask)
    generator = relion_random.GlibcRand(7)
    for _ in range(5):
        generator.rand()
    random1 = _loop_randomize(half1, randomize_at, generator)
    random2 = _loop_randomize(half2, randomize_at, generator)
    assert details["draws"] == generator.draws - 5 == stream.draws - 5
    assert stream.rand() == generator.rand()  # the stream continues where RELION's would
    random_masked = _loop_fsc(random1 * mask, random2 * mask)
    expected = np.zeros_like(masked)
    for i in range(masked.size):
        if i < randomize_at + 2:
            expected[i] = masked[i]
        elif random_masked[i] > masked[i]:
            expected[i] = 0.0
        else:
            expected[i] = (masked[i] - random_masked[i]) / (1.0 - random_masked[i])
    expected[current_size // 2 + 1 :] = 0.0
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
    # The correction acts: some corrected shell inside current_size differs from the masked FSC.
    if randomize_at + 2 <= current_size // 2:
        assert np.any(
            np.abs(got[randomize_at + 2 : current_size // 2 + 1] - masked[randomize_at + 2 : current_size // 2 + 1])
            > 1e-6
        )


def test_corrected_fsc_without_a_low_shell_is_the_unmasked_fsc():
    half1, _, mask = _maps()
    stream = relion_random.GlibcRand(0)
    got, details = solvent_mask.solvent_corrected_fsc(half1, half1, mask, current_size=12, stream=stream)
    assert details["randomize_at"] == -1
    assert details["draws"] == stream.draws == 0  # an iteration without randomize_at draws nothing
    np.testing.assert_allclose(got, _loop_fsc(half1, half1), rtol=1e-12)


def _write(path, data, pixel):
    with mrcfile.new(path) as handle:
        handle.set_data(np.asarray(data, dtype=np.float32))
        handle.voxel_size = pixel


def test_mask_is_clipped_and_windowed_about_the_centre(tmp_path):
    data = np.zeros((8, 8, 8))
    data[4, 4, 4] = 1.7
    data[0, 0, 0] = -0.3
    _write(tmp_path / "m.mrc", data, 2.0)
    same = solvent_mask.read_solvent_mask(tmp_path / "m.mrc", box_size=8, pixel_size=2.0)
    assert_matches([same.min(), same.max(), same[4, 4, 4]], [0.0, 1.0, 1.0])
    padded = solvent_mask.read_solvent_mask(tmp_path / "m.mrc", box_size=12, pixel_size=2.0)
    assert padded.shape == (12, 12, 12)
    assert_matches([padded[6, 6, 6], padded.sum()], [1.0, 1.0])
    cropped = solvent_mask.read_solvent_mask(tmp_path / "m.mrc", box_size=6, pixel_size=2.0)
    assert cropped.shape == (6, 6, 6)
    assert_matches(cropped[3, 3, 3], 1.0)


def test_mask_on_another_pixel_size_is_resampled(tmp_path):
    grid = np.indices((16, 16, 16)) - 8
    blob = np.exp(-0.5 * (grid**2).sum(0) / 3.0**2)  # sigma 6 A
    _write(tmp_path / "m.mrc", blob, 2.0)
    # RELION: rescale to ROUND(16 * 2.0 / 4.0) = 8 voxels, then window to the 16 box.
    resampled = solvent_mask.read_solvent_mask(tmp_path / "m.mrc", box_size=16, pixel_size=4.0)
    assert resampled.shape == (16, 16, 16)
    assert 0.0 <= resampled.min() and resampled.max() <= 1.0
    # The blob keeps its size in Angstrom: sigma 1.5 voxels at 4 A.
    assert resampled[8, 8, 8] == pytest.approx(1.0, abs=0.02)
    assert resampled[8, 8, 10] == pytest.approx(math.exp(-0.5 * (2 / 1.5) ** 2), abs=0.02)
    # Within 0.001 A no resampling happens.
    same = solvent_mask.read_solvent_mask(tmp_path / "m.mrc", box_size=16, pixel_size=2.0005)
    assert_matches(same, blob.astype(np.float32))


def _settings(**kwargs):

    base = dict(
        box_size=8,
        voxel_size=2.0,
        volume_shape=(8, 8, 8),
        padding_factor=2,
        projection_padding_factor=2,
        minres_map=5,
        width_mask_edge=5,
        fmask_edge=2,
        tau2_fudge=1.0,
        particle_diameter_angstrom=12.0,
        first_iteration_lowpass_angstrom=None,
    )
    return reconstruction_settings(**{**base, **kwargs}, programs=ReconstructionPrograms.from_environ())


def test_user_mask_replaces_the_diameter_sphere():

    user = np.zeros((8, 8, 8))
    user[2:5, 3:6, 1:7] = 1.0
    settings = _settings(solvent_mask=user)
    assert map_postprocess.solvent_flatten_requested(settings)
    assert_matches(np.asarray(map_postprocess.numbered_solvent_mask(settings, dtype=np.float64)), user)
    # Without a diameter a user mask still flattens.
    assert map_postprocess.solvent_flatten_requested(_settings(solvent_mask=user, particle_diameter_angstrom=None))


def test_corrected_fsc_needs_a_mask_of_the_model_shape():
    with pytest.raises(ValueError, match="needs --solvent_mask"):
        _settings(solvent_correct_fsc=True)
    with pytest.raises(ValueError, match="not the model"):
        _settings(solvent_mask=np.ones((6, 6, 6)))


def test_mask_file_reaches_the_internal_frame(tmp_path):
    from relax.refinement.setup_checks import _internal_solvent_mask

    data = np.zeros((8, 8, 8))
    data[1, 2, 3] = 1.0  # file (z, y, x)
    _write(tmp_path / "m.mrc", data, 2.0)
    internal = _internal_solvent_mask(str(tmp_path / "m.mrc"), 8, 2.0)
    assert_matches([internal[3, 2, 1], internal.sum()], [1.0, 1.0])
    assert _internal_solvent_mask(None, 8, 2.0) is None


def test_half_spectrum_fsc_depends_on_the_halved_axis():
    # RELION halves the FFTW spectrum along the map file's x axis; relax's internal frame is the file's
    # transpose, so the corrected FSC is computed after transposing back (priors.solvent_corrected_fsc).
    half1, half2, mask = _maps()
    mask = mask * (1.0 + 0.3 * np.indices(mask.shape)[0] / mask.shape[0])  # break the mask's symmetry
    file_order = solvent_mask.real_space_fsc(half1 * mask, half2 * mask)
    transposed = solvent_mask.real_space_fsc(
        np.transpose(half1 * mask, (2, 1, 0)), np.transpose(half2 * mask, (2, 1, 0))
    )
    assert np.abs(file_order - transposed).max() > 1e-6


def _read_map(path):
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float64)


@pytest.mark.parametrize(
    ("case", "iterations", "split_draws"),
    # The ET input carries rlnRandomSubset; the SPA input does not, so RELION's split drew one rand() per
    # particle (5000) from the same leader stream first (PROVENANCE.json of the fixture).
    [("tomo_seed2", (1, 2, 3), 0), ("spa_seed2", (1, 2, 3, 4), 5000)],
)
def test_corrected_fsc_replays_relion_leader_stream(case, iterations, split_draws):
    """RELION's own --solvent_correct_fsc curves, iteration after iteration, from one stream (relax#35).

    The stream is seeded once with random_seed 2, gives the split its draws, and then runs on through the
    iterations; each iteration's corrected FSC on RELION's unfil halves, in the file's axis order, is that
    iteration's rlnGoldStandardFsc to RELION's print precision (6 decimals).
    """
    import starfile

    root = fixture_dir("solvent_fsc_stream_relion_seed2") / case
    mask = _read_map(root / "solvent_ref_mask.mrc")
    stream = relion_random.init_random_generator(2)
    stream.rand_array(split_draws)
    randomised = 0
    for iteration in iterations:
        stem = root / f"run_it{iteration:03d}"
        model = starfile.read(f"{stem}_half1_model.star", always_dict=True)
        current_size = int(model["model_general"]["rlnCurrentImageSize"])
        expected = model["model_class_1"]["rlnGoldStandardFsc"].to_numpy(float)
        got, details = solvent_mask.solvent_corrected_fsc(
            _read_map(f"{stem}_half1_class001_unfil.mrc"),
            _read_map(f"{stem}_half2_class001_unfil.mrc"),
            mask,
            current_size=current_size,
            stream=stream,
        )
        randomised += details["randomize_at"] > 0
        np.testing.assert_allclose(got[: expected.size], expected, rtol=0.0, atol=1e-6)
    assert randomised >= 2  # the stream carried across randomising iterations
