"""RELION start-up tau2 and data_vs_prior (MlModel::initialiseDataVersusPrior) for the K=1 start."""

import numpy as np
import pytest
import starfile
from helpers.em_fixtures import fixture_file

from relax.vdam.init import relion_initial_tau2_and_data_vs_prior
from scripts import run_full_refinement

pytestmark = pytest.mark.unit


def _reference_spectrum(volume):
    """Independent getSpectrum(POWER_SPECTRUM) on the full complex FFT, rounded shell radius."""
    n = volume.shape[0]
    fourier = np.fft.fftn(volume) / volume.size
    k = np.fft.fftfreq(n) * n
    # FFTW's half transform keeps x in [0, N/2]; the full FFT stores x = N/2 as -N/2.
    kx = np.abs(k)
    radius = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kx[None, None, :] ** 2)
    keep = np.broadcast_to(((k >= 0) | (k == -(n // 2)))[None, None, :], radius.shape)
    shell = np.floor(radius + 0.5).astype(int)
    n_shells = n // 2 + 1
    out = np.zeros(n_shells)
    for s in range(n_shells):
        sel = keep & (shell == s)
        out[s] = np.mean(np.abs(fourier[sel]) ** 2)
    return out


def test_initial_tau2_and_data_vs_prior_follow_relion_formula():
    rng = np.random.default_rng(3)
    n = 8
    volume = rng.normal(size=(n, n, n))
    sigma2 = rng.uniform(0.5, 2.0, size=n // 2 + 1)

    tau2, dvp = relion_initial_tau2_and_data_vs_prior(volume, tau2_fudge=2.0, avg_sigma2_noise=sigma2, nr_particles=40)

    # ml_model.cpp:1591-1621: tau2 = fudge * spectrum * N^2 / 2; evidence = N_part / sigma2 / (2 i).
    expected_tau2 = 2.0 * _reference_spectrum(volume) * n * n / 2.0
    shell_factor = np.ones(n // 2 + 1)
    shell_factor[1:] = 2.0 * np.arange(1, n // 2 + 1)
    np.testing.assert_allclose(tau2, expected_tau2, rtol=1e-12)
    np.testing.assert_allclose(dvp, 40.0 / sigma2 / shell_factor * expected_tau2, rtol=1e-12)


def test_initial_tau2_rejects_nonpositive_noise():
    with pytest.raises(ValueError, match="positive"):
        relion_initial_tau2_and_data_vs_prior(
            np.ones((4, 4, 4)), tau2_fudge=1.0, avg_sigma2_noise=np.zeros(3), nr_particles=1
        )


@pytest.mark.parametrize("fixture", ["k1_5k128", "k1_50k256"])
def test_k1_start_matches_relion_run_it000_model(fixture):
    """Dry check against RELION's own start-up model on the K=1 fixtures (half 1)."""
    model_path = fixture_file(f"{fixture}_relion_os0", "run_it000_half1_model.star")
    reference_path = fixture_file(f"{fixture}_data", "reference_init_relion.mrc")
    import mrcfile
    from recovar.utils import helpers

    from relax.refinement.mean_helpers import initial_low_pass_filter_references

    with mrcfile.open(reference_path, permissive=True) as mrc:
        pixel_size = float(mrc.voxel_size.x)
    reference = np.asarray(helpers.load_mrc(str(reference_path)), dtype=np.float64)
    n = reference.shape[0]
    reference = initial_low_pass_filter_references(
        reference[None], ori_size=n, pixel_size=pixel_size, ini_high_ang=30.0, filter_edgewidth=2.0
    )[0]
    model = starfile.read(model_path)
    sigma2 = np.asarray(model["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64)
    n_half_particles = int(np.sum(model["model_groups"]["rlnGroupNrParticles"]))

    mean_variance, dvp = run_full_refinement._relion_k1_start_tau2_and_data_vs_prior(
        reference,
        sigma2 * float(n) ** 4,
        grid_size=n,
        volume_shape=(n, n, n),
        tau2_fudge=1.0,
        nr_particles=n_half_particles,
    )

    relion = model["model_class_1"]
    relion_tau2 = np.asarray(relion["rlnReferenceTau2"], dtype=np.float64)
    relion_dvp = np.asarray(relion["rlnSsnrMap"], dtype=np.float64)
    # The STAR prints six significant digits; shells past the ini_high filter hold ~1e-37 round-off.
    signal = relion_tau2 > 1e-12 * relion_tau2.max()
    np.testing.assert_allclose(dvp[signal], relion_dvp[signal], rtol=5e-6)
    np.testing.assert_array_equal(dvp > 3.0, relion_dvp > 3.0)
    assert mean_variance.shape == (n**3,)
