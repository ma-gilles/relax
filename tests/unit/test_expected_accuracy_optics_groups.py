"""Expected accuracy with one noise spectrum per optics group (RELION binding, CPU).

RELION scores every trial particle with its own optics group's CTF constants and
noise (``calculateExpectedAngularErrors``). Two groups with identical constants and
noise must reproduce the one-group estimate, and different constants must run.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from relax.helpers.expected_accuracy import estimate_relion_expected_accuracy

N = 16
SHAPE = (N, N, N)


def _inputs(rng, voltages):
    x = np.arange(N) - N / 2
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    vol = np.exp(-((xx - 2) ** 2 + yy**2 + zz**2) / 6.0) + 0.5 * np.exp(-(xx**2 + (yy + 3) ** 2 + zz**2) / 3.0)
    reference = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol))).reshape(-1)
    n_particles = len(voltages)
    ctf = np.zeros((n_particles, 11))
    ctf[:, 0] = rng.uniform(10000, 20000, n_particles)
    ctf[:, 1] = ctf[:, 0] - 100
    ctf[:, 3] = voltages
    ctf[:, 4] = 2.7
    ctf[:, 5] = 0.1
    return dict(
        reference_fourier=reference,
        volume_shape=SHAPE,
        best_eulers_deg=rng.uniform(-180, 180, (n_particles, 3)),
        class_ids=np.zeros(n_particles, dtype=np.int32),
        class_weights=np.ones(1),
        dataset=SimpleNamespace(voxel_size=4.0, CTF_params=ctf),
        trial_order_local=rng.permutation(n_particles),
        current_image_size=N,
        padding_factor=2,
        sigma2_fudge=1.0,
        random_seed=7,
        random_seed_particle_ids=np.arange(n_particles) + 100,
        do_ctf_correction=True,
    )


@pytest.mark.unit
def test_two_identical_groups_reproduce_one_group():
    pytest.importorskip("relax.relion_bind._relion_bind_core")
    rng = np.random.default_rng(0)
    kwargs = _inputs(rng, [300.0] * 8)
    sigma2 = np.linspace(2.0, 1.0, N // 2 + 1) * float(N) ** 4
    one = estimate_relion_expected_accuracy(sigma2_noise_native=sigma2, **kwargs)
    two = estimate_relion_expected_accuracy(
        sigma2_noise_native=np.stack([sigma2, sigma2]),
        optics_group_ids=np.array([0, 1, 1, 0, 1, 0, 0, 1]),
        **kwargs,
    )
    assert one.acc_rot < 999.0
    np.testing.assert_allclose(two.acc_rot, one.acc_rot, rtol=1e-12)
    np.testing.assert_allclose(two.acc_trans_angstrom, one.acc_trans_angstrom, rtol=1e-12)
    assert int(two.class_counts[0]) == int(one.class_counts[0]) == 8


@pytest.mark.unit
def test_groups_with_different_constants_and_noise():
    pytest.importorskip("relax.relion_bind._relion_bind_core")
    rng = np.random.default_rng(1)
    groups = np.array([0, 1, 1, 0, 1, 0])
    kwargs = _inputs(rng, np.where(groups == 0, 300.0, 200.0))
    sigma2 = np.stack([np.full(N // 2 + 1, 1.0), np.full(N // 2 + 1, 4.0)]) * float(N) ** 4
    result = estimate_relion_expected_accuracy(sigma2_noise_native=sigma2, optics_group_ids=groups, **kwargs)
    assert result.acc_rot < 999.0 and int(result.class_counts[0]) == 6
    with pytest.raises(NotImplementedError):
        estimate_relion_expected_accuracy(sigma2_noise_native=sigma2[0], **kwargs)
