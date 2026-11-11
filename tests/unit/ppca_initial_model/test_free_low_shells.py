"""--likelihood-min-shell: no likelihood weight on image shells below K (tomo.free_low_shells) and a model held at
zero on the 3D shells below K (initialization.bandlimit_and_mask min_shell)."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu

from relax.fourier.half_spectrum import make_shell_indices_half
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.initialization import bandlimit_and_mask, zero_low_shells
from relax.ppca_initial_model.iteration_loop import _frozen_noise_shells
from relax.ppca_initial_model.tomo import TiltParticles, free_low_shells

N = 16
IMAGE_SHAPE = (N, N)
VOLUME_SHAPE = (N, N, N)


def _particles(n_images=6, seed=0):
    rng = np.random.default_rng(seed)
    F = N * (N // 2 + 1)
    half = (rng.standard_normal((n_images, F)) + 1j * rng.standard_normal((n_images, F))).astype(np.complex64)
    ctf = rng.uniform(0.1, 1.0, (n_images, F)).astype(np.float32)

    def read(ids):
        ids = np.asarray(ids)
        return None, jnp.asarray(half[ids]), jnp.asarray(ctf[ids])

    particles = TiltParticles(
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        voxel_size=2.0,
        image_offsets=np.array([0, 3, n_images]),
        image_frame=np.array([0, 1, 2, 0, 1, 2]),
        particle_group=np.array([0, 0]),
        group_frames=(np.tile(np.eye(3), (3, 1, 1)),),
        read=read,
    )
    return particles, half, ctf


@pytest.mark.parametrize("K", [1, 2, 3])
def test_free_low_shells_zeroes_the_images_and_ctf_below_k_only(K):
    particles, half, ctf = _particles()
    freed = free_low_shells(particles, K)
    _, got_half, got_ctf = freed.read(np.arange(6))
    low = np.asarray(make_shell_indices_half(IMAGE_SHAPE)).reshape(-1) < K
    assert low.sum() > 0 and (~low).sum() > 0
    assert np.all(np.asarray(got_half)[:, low] == 0) and np.all(np.asarray(got_ctf)[:, low] == 0)
    np.testing.assert_array_equal(np.asarray(got_half)[:, ~low], half[:, ~low])
    np.testing.assert_array_equal(np.asarray(got_ctf)[:, ~low], ctf[:, ~low])
    assert freed.free_shells == K and _frozen_noise_shells(freed) == tuple(range(K))


def test_zero_shells_leave_the_particles_and_frozen_shells_as_they_were():
    particles, _, _ = _particles()
    assert free_low_shells(particles, 0) is particles
    assert _frozen_noise_shells(particles) == ()
    assert _frozen_noise_shells(dataclasses.replace(particles, dc_free=True)) == (0,)
    assert _frozen_noise_shells(dataclasses.replace(particles, dc_free=True, free_shells=3)) == (0, 1, 2)


def test_bandlimit_min_shell_zeroes_only_the_low_3d_shells():
    rng = np.random.default_rng(1)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(VOLUME_SHAPE)))
    theta = jnp.asarray((rng.standard_normal((half_size, 3)) + 1j * rng.standard_normal((half_size, 3))), jnp.complex64)
    mask = jnp.ones(VOLUME_SHAPE, jnp.float32)
    plain = np.asarray(bandlimit_and_mask(theta, VOLUME_SHAPE, 6, mask))
    cut = np.asarray(bandlimit_and_mask(theta, VOLUME_SHAPE, 6, mask, 2))
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(VOLUME_SHAPE), np.int64).reshape(-1)
    assert np.all(cut[shells < 2] == 0)
    np.testing.assert_array_equal(cut[shells >= 2], plain[shells >= 2])
    np.testing.assert_array_equal(np.asarray(zero_low_shells(jnp.asarray(plain), VOLUME_SHAPE, 2)), cut)


def test_negative_min_shell_is_refused():
    with pytest.raises(ValueError, match="minimum shell"):
        Config(likelihood_min_shell=-1)
