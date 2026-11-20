"""The masked-volume likelihood of subtomogram particles (``relax/ppca_initial_model/masked_volume.py``).

The stream's volume projection is RECOVAR's half-volume trilinear gather at ``k @ R`` and its backprojection that
gather's adjoint; tile scores are checked against a float64 full-spectrum evaluation (Hermitian partners expanded,
gather on the full volume written here), the M-step statistics against the closed-form single-pose optimum, and the
whole preprocessing against a noiseless particle made by the image route's own forward model.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
from helpers.float_compare import assert_matches
from helpers.full_row_tile import accumulate_full_row_tile
from recovar.core import relion_interp
from scipy.spatial.transform import Rotation

from relax.fourier.preprocessing import relion_half_translation_lattice
from relax.ppca_initial_model.masked_volume import (
    half_ball,
    load_volume_tile,
    prepare_volume_observations,
    prepare_volume_stream,
)
from relax.ppca_initial_model.tomo import TiltParticles, load_tilt_tile, tilt_shifts
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_kernels import _backproject_volume, _project_volume, score_tile
from relax.ppca_refinement.full_row_stream import prepare_full_row_stream, read_tile

pytestmark = pytest.mark.unit


def _half_volume(rng, n, scale=1.0, radius=None):
    real = rng.standard_normal((n, n, n))
    if radius is not None:
        g = np.indices((n, n, n)) - n // 2
        real = real * (np.sum(g**2, 0) <= radius**2)
    full = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(real)))
    return (scale * np.asarray(ftu.full_volume_to_half_volume(jnp.asarray(full), (n, n, n))).reshape(-1)).astype(
        np.complex64
    )


def test_volume_gather_on_a_central_plane_is_the_slice():
    """At ``k = (k0, k1, 0)`` the volume projection reads what RECOVAR's slicer reads for pixel ``(k0, k1)``."""
    n = 12
    rng = np.random.default_rng(0)
    volume = _half_volume(rng, n)
    R = Rotation.random(random_state=1).as_matrix().astype(np.float32)
    indices, r2 = half_ball((n, n, n), n // 2 - 1)
    k = np.stack(np.unravel_index(indices, (n, n, n // 2 + 1)), -1) - np.asarray([n // 2, n // 2, 0])
    plane = indices[k[:, 2] == 0]
    gathered = np.asarray(_project_volume(jnp.asarray(volume)[None], jnp.asarray(plane), jnp.asarray(R)[None], (n, n, n)))
    image = np.asarray(
        relion_interp.project(jnp.asarray(volume), jnp.asarray(R)[None], (n, n), (n, n, n), order=1, half_volume=True)
    ).reshape(-1)
    pixels = np.asarray(ftu.get_k_coordinate_of_each_pixel((n, n), voxel_size=1, scaled=False))
    lookup = {(int(round(a)), int(round(b))): i for i, (a, b) in enumerate(pixels)}
    kp = k[k[:, 2] == 0]
    expected = image[[lookup[(int(a), int(b))] for a, b, _ in kp]]
    assert_matches(gathered[0, 0], expected, rtol=1e-5)


def test_volume_backprojection_is_the_projection_adjoint():
    """``Re<P x, y> = Re<x, P* y>`` over the half volume, for complex ``x`` and ``y``."""
    n = 10
    rng = np.random.default_rng(3)
    indices, _ = half_ball((n, n, n), n // 2 - 1)
    rotations = Rotation.random(3, random_state=4).as_matrix().astype(np.float32)
    x = _half_volume(rng, n)
    y = (rng.standard_normal((1, 3, indices.size)) + 1j * rng.standard_normal((1, 3, indices.size))).astype(np.complex64)
    Px = np.asarray(_project_volume(jnp.asarray(x)[None], jnp.asarray(indices), jnp.asarray(rotations), (n, n, n)))
    Py = np.asarray(_backproject_volume(jnp.asarray(y), jnp.asarray(indices), jnp.asarray(rotations), (n, n, n)))
    left = np.sum(np.real(np.conj(Px.astype(np.complex128)) * y))
    right = np.sum(np.real(np.conj(x.astype(np.complex128)) * Py[0]))
    assert_matches(left, right, rtol=1e-5)


def test_volume_projection_and_backprojection_are_recovar_half_volume_interpolation():
    """The channels-last gather of the expanded volume equals RECOVAR's half-volume trilinear gather on the window, and
    its transposed gather RECOVAR's half-volume scatter (Hermitian partners conjugated), for every channel."""
    from recovar.core.relion_interp import _scatter_trilinear_half, _trilinear_gather_half

    from relax.ppca_refinement.full_row_kernels import _volume_grid_coordinates

    n = 12
    shape = (n, n, n)
    rng = np.random.default_rng(7)
    indices, _ = half_ball(shape, n // 2 - 1)
    rotations = Rotation.random(4, random_state=8).as_matrix().astype(np.float32)
    volumes = np.stack([_half_volume(rng, n) for _ in range(3)])
    g = np.asarray(_volume_grid_coordinates(jnp.asarray(indices), jnp.asarray(rotations), shape)).reshape(-1, 3)
    half_shape = (n, n, n // 2 + 1)
    reference = np.stack([np.asarray(_trilinear_gather_half(jnp.asarray(v).reshape(half_shape), *jnp.asarray(g).T, n, n, n))
                          for v in volumes]).reshape(3, len(rotations), -1)  # fmt: skip
    actual = np.asarray(_project_volume(jnp.asarray(volumes), jnp.asarray(indices), jnp.asarray(rotations), shape))
    assert_matches(actual, reference, rtol=1e-5)
    images = (rng.standard_normal((2,) + actual.shape[1:]) + 1j * rng.standard_normal((2,) + actual.shape[1:])).astype(np.complex64)
    reference = np.stack([np.asarray(_scatter_trilinear_half(int(np.prod(half_shape)), *jnp.asarray(g).T, jnp.asarray(im).reshape(-1), n, n, n))
                          for im in images])  # fmt: skip
    actual = np.asarray(_backproject_volume(jnp.asarray(images), jnp.asarray(indices), jnp.asarray(rotations), shape))
    assert_matches(actual, reference, rtol=1e-5)
    real = images.real.astype(np.float32)
    actual_real = np.asarray(_backproject_volume(jnp.asarray(real), jnp.asarray(indices), jnp.asarray(rotations), shape))
    reference_real = np.stack([np.asarray(_scatter_trilinear_half(int(np.prod(half_shape)), *jnp.asarray(g).T, jnp.asarray(im).reshape(-1), n, n, n))
                               for im in real])  # fmt: skip
    assert actual_real.dtype == np.float32
    assert_matches(actual_real, reference_real, rtol=1e-5)


FRAMES = Rotation.from_euler("zyx", [[85, a, 0] for a in range(-60, 61, 6)], degrees=True).as_matrix()


def _particles(n, half, ctf, n_particles):
    K = len(FRAMES)
    offsets = (np.arange(n_particles + 1) * K).astype(np.int64)

    def read(ids):
        ids = np.asarray(ids)
        return None, jnp.asarray(half[ids]), jnp.asarray(ctf[ids])

    return TiltParticles(
        image_shape=(n, n),
        volume_shape=(n, n, n),
        voxel_size=10.0,
        image_offsets=offsets,
        image_frame=np.tile(np.arange(K), n_particles).astype(np.int64),
        particle_group=np.zeros(n_particles, np.int64),
        group_frames=(np.asarray(FRAMES, np.float64),),
        read=read,
    )


def _forward_images(n, volume, R, t, ctf):
    """The image route's model of one particle at pose ``(R, t)``: ``CTF P(Aproj_f R) x`` shifted by
    ``[Aproj_f t]_{1,2}`` (the conjugate of the phase the tilt tile puts on the data)."""
    rows = np.einsum("fab,bc->fac", FRAMES, R).astype(np.float32)
    proj = np.asarray(
        relion_interp.project(
            jnp.asarray(volume), jnp.asarray(rows), (n, n), (n, n, n), order=1, half_volume=True, half_image=True,
            max_r=n // 2 - 1,
        )
    )  # fmt: skip
    lattice = np.asarray(relion_half_translation_lattice((n, n)), np.float64)
    shifts = tilt_shifts(FRAMES, np.asarray(t, np.float32)[None])[:, 0]  # (K, 2)
    phase = np.exp(2j * np.pi * np.einsum("kd,pd->kp", shifts, lattice))
    return (ctf * proj * phase).astype(np.complex64)


def _streams(particles, mu, W, rotations, translations, radius, q, *, image=False):
    grid = dict(
        rotations=rotations,
        translations=translations,
        rotation_log_prior=np.zeros(len(rotations), np.float32),
        translation_log_prior=np.zeros(len(translations), np.float32),
        rotation_parent=np.zeros(len(rotations), np.int32),
        translation_parent=np.zeros(len(translations), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=2 * radius, q=q, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=particles.n_images, rotation_block_size=4),
        gemm_precision="fp32",
    )
    if image:
        n = particles.grid_size
        return prepare_full_row_stream(
            particles, mu, W, noise_variance=[np.ones((n, n), np.float32)], tile_loader=load_tilt_tile,
            scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True), **grid,
        )  # fmt: skip
    return prepare_volume_stream(particles, mu, W, **grid)


def _noiseless_problem(n=16, n_particles=3, seed=0):
    rng = np.random.default_rng(seed)
    volume = _half_volume(rng, n, radius=n // 3)
    ctf = np.ones((n_particles * len(FRAMES), n * (n // 2 + 1)), np.float32)
    rotations = Rotation.random(6, random_state=seed + 1).as_matrix().astype(np.float32)
    translations = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1], [-1, 1, 0]], np.float32)
    truth = [(p % 6, (p + 1) % 5) for p in range(n_particles)]
    half = np.concatenate(
        [_forward_images(n, volume, rotations[r], translations[t], ctf[: len(FRAMES)]) for r, t in truth]
    )
    particles = _particles(n, half, ctf, n_particles)
    return particles, volume, rotations, translations, truth


def test_noiseless_particles_score_best_at_their_true_pose():
    """Particles made by the image route's forward model: the masked-volume stream's top pose is the true one (and
    the image route's, as a check of the construction)."""
    n = 16
    particles, volume, rotations, translations, truth = _noiseless_problem(n)
    noise = np.ones((1, n // 2 + 1), np.float32)
    observations = prepare_volume_observations(particles, noise, mask_radius_px=n // 2 - 1, lam_rel=0.01, n_mc=1)
    masked = TiltParticles(**{**particles.__dict__, "volume_observations": observations})
    mu = jnp.asarray(volume)
    W = jnp.zeros((volume.size, 1), jnp.complex64)
    ids = np.arange(particles.n_images)
    for stream in (
        _streams(masked, mu, W, rotations, translations, n // 2 - 1, 1),
        _streams(particles, mu, W, rotations, translations, n // 2 - 1, 1, image=True),
    ):
        tile, _, layout = read_tile(stream, ids, [None] * ids.size, collect_observation=False)
        _, posterior = score_tile(stream, tile, layout["n_blocks"])
        rows = np.asarray(tile.rows)[np.asarray(posterior.top_rotation)]
        assert [(int(r), int(t)) for r, t in zip(rows, np.asarray(posterior.top_translation))] == truth


def _full_from_half_ball(values, indices, n):
    """Full-ball dictionary ``k -> value`` from half-ball values, with ``value(-k) = conj(value(k))``."""
    k = np.stack(np.unravel_index(indices, (n, n, n // 2 + 1)), -1) - np.asarray([n // 2, n // 2, 0])
    out = {}
    for kk, v in zip(map(tuple, k), values):
        out[kk] = v
        out.setdefault(tuple(-np.asarray(kk)), np.conj(v))
    return out


def _gather_full(volume_full, coords, n):
    """Float64 trilinear read of a full centred volume ``(n, n, n)`` at grid coordinates (zero outside)."""
    b = np.floor(coords).astype(int)
    f = coords - b
    out = np.zeros(len(coords), np.complex128)
    for d in np.ndindex(2, 2, 2):
        i = b + np.asarray(d)
        w = np.prod(np.where(np.asarray(d)[None], f, 1 - f), axis=1)
        ok = np.all((i >= 0) & (i < n), axis=1)
        ic = np.clip(i, 0, n - 1)
        out += np.where(ok, w * volume_full[ic[:, 0], ic[:, 1], ic[:, 2]], 0)
    return out


def test_volume_tile_scores_match_the_full_spectrum_sums():
    """Pass-1 scores against float64 sums over the full ball: ``inner = Re sum conj(x_R) data e^{-2 pi i k.t/n}``,
    ``gram = sum filt2 Re(conj x_R,p x_R,q)``, and the PPCA score of ``latent_scores``' formula."""
    n, q = 10, 2
    particles, _, rotations, translations, _ = _noiseless_problem(n, n_particles=2, seed=4)
    rng = np.random.default_rng(9)
    half = (rng.standard_normal((particles.image_offsets[-1], n * (n // 2 + 1))) * 3).astype(np.complex64)
    particles = _particles(n, half, np.full(half.shape, 0.7, np.float32), 2)
    noise = np.full((1, n // 2 + 1), 2.0, np.float32)
    observations = prepare_volume_observations(particles, noise, mask_radius_px=3.5, n_mc=1)
    masked = TiltParticles(**{**particles.__dict__, "volume_observations": observations})
    theta = np.stack([_half_volume(rng, n, s) for s in (1.0, 0.3, 0.2)], 1)
    radius = 3
    stream = _streams(masked, jnp.asarray(theta[:, 0]), jnp.asarray(theta[:, 1:]), rotations, translations, radius, q)
    ids = np.arange(2)
    tile, _, layout = read_tile(stream, ids, [None, None], collect_observation=False)
    kept, _ = score_tile(stream, tile, layout["n_blocks"])
    score = np.asarray(kept.score)[: len(rotations)]
    with jax.enable_x64(True):
        fulls = [np.asarray(ftu.half_volume_to_full_volume(jnp.asarray(theta[:, p]), (n, n, n))).reshape(n, n, n)
                 for p in range(q + 1)]  # fmt: skip
    window = observations.radius2 <= radius**2
    expected = np.zeros_like(score, dtype=np.float64)
    for b in ids:
        data = _full_from_half_ball(observations.data[b][window], observations.indices[window], n)
        filt2 = _full_from_half_ball(observations.filt2[b][window], observations.indices[window], n)
        ks = np.asarray(list(data), np.float64)
        d = np.asarray([data[tuple(k)] for k in map(tuple, ks.astype(int))])
        f2 = np.real([filt2[tuple(k)] for k in map(tuple, ks.astype(int))])
        for r, R in enumerate(rotations.astype(np.float64)):
            xr = np.stack([_gather_full(v, ks @ R + n // 2, n) for v in fulls])  # (P, K)
            gram = np.real(np.einsum("k,pk,qk->pq", f2, np.conj(xr), xr))
            for t, u in enumerate(translations.astype(np.float64)):
                inner = np.real(np.conj(xr) @ (d * np.exp(-2j * np.pi * ks @ u / n)))
                H = gram[1:, 1:]
                bvec = inner[1:] - gram[0, 1:]
                M = np.eye(q) + H
                expected[r, b, t] = -0.5 * (
                    gram[0, 0] - 2 * inner[0] - bvec @ np.linalg.solve(M, bvec) + np.linalg.slogdet(M)[1]
                )
    # float32 gathers and GEMMs against float64 sums, relative to the largest |score|.
    assert_matches(score, expected, rtol=1e-4)


def test_single_pose_newton_step_is_the_closed_form_optimum():
    """One particle, the identity rotation and one shift (posterior 1): ``x + residual / lhs`` is the B-diag
    optimum ``data e^{-2 pi i k.t/n} / filt2`` on the window's ``kz > 0`` voxels."""
    n = 10
    particles, _, _, _, _ = _noiseless_problem(n, n_particles=1, seed=6)
    rng = np.random.default_rng(1)
    half = (rng.standard_normal((particles.image_offsets[-1], n * (n // 2 + 1))) * 3).astype(np.complex64)
    particles = _particles(n, half, np.full(half.shape, 0.9, np.float32), 1)
    observations = prepare_volume_observations(particles, np.ones((1, n // 2 + 1), np.float32), mask_radius_px=3.0,
                                               n_mc=1)  # fmt: skip
    masked = TiltParticles(**{**particles.__dict__, "volume_observations": observations})
    x = _half_volume(rng, n)
    radius = 4
    t = np.asarray([[1.0, -2.0, 0.5]], np.float32)
    stream = _streams(masked, jnp.asarray(x), None, np.eye(3, dtype=np.float32)[None], t, radius, 0)
    stats = accumulate_full_row_tile(stream, np.arange(1), [None])
    lhs = np.asarray(stats.lhs_tri)[:, 0]
    residual = np.asarray(stats.residual_gradient)[:, 0]
    window = observations.radius2 <= radius**2
    indices = observations.indices[window]
    k = np.stack(np.unravel_index(indices, (n, n, n // 2 + 1)), -1) - np.asarray([n // 2, n // 2, 0])
    inner = k[:, 2] > 0
    expected = observations.data[0][window] * np.exp(-2j * np.pi * k @ t[0] / n) / observations.filt2[0][window]
    step = x[indices] + residual[indices] / lhs[indices]
    assert_matches(step[inner], expected[inner].astype(np.complex64), rtol=1e-4)


def _controller_particles():
    n = 16
    rng = np.random.default_rng(2)
    n_particles = 8
    raw = rng.standard_normal((n_particles * len(FRAMES), n, n)).astype(np.float32)
    half = np.asarray(ftu.get_dft2_real(raw)).reshape(len(raw), -1).astype(np.complex64)
    particles = _particles(n, half, np.full(half.shape, 0.8, np.float32), n_particles)
    return TiltParticles(**{**particles.__dict__, "read": lambda ids: (raw[ids], *particles.read(ids)[1:])})


@pytest.mark.parametrize(
    "options",
    [
        {"q": 2, "tomo_likelihood": "masked", "masked_mask_radius_px": 6.0},
        {"q": 0, "single_volume": True, "tomo_likelihood": "masked", "masked_mask_radius_px": 6.0},
        {"q": 0, "single_volume": True, "hold_noise": True},
        {"q": 2, "tomo_likelihood": "masked", "masked_mask_radius_px": 6.0, "masked_exact_mstep": True},
    ],
)
def test_controller_trains_with_the_masked_likelihood_and_a_single_volume(tmp_path, options):
    """Two all-particle updates and the final pass through the controller: the noise is held, the outputs finite,
    the final poses written."""
    import json

    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import run

    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    particles = _controller_particles()
    config = Config(
        iterations=2, stages=((1, 3, 0),), shift_range=1, shift_step=1, image_batch_size=3, rotation_block_size=32,
        final_poses=True, **options,
    )  # fmt: skip
    try:
        state = run(particles, config, tmp_path, {"test": True}, diameter_ang=100.0)
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    rows = [json.loads(line) for line in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1, 2]
    assert all(row["noise_held"] for row in rows)
    from relax.ppca_initial_model import checkpoint

    start = checkpoint.load(tmp_path / "checkpoint_0000.npz", config, {"test": True})
    np.testing.assert_array_equal(np.asarray(state.noise), np.asarray(start.noise))
    assert state.theta.shape[1] == config.q + 1 and np.all(np.isfinite(np.asarray(state.theta)))
    embeddings = np.load(tmp_path / "embeddings.npz")
    assert embeddings["z"].shape == (particles.n_images, config.q)
    assert embeddings["best_rotation_matrix"].shape == (particles.n_images, 3, 3)
    assert embeddings["best_translation_px"].shape == (particles.n_images, 3)
    assert (tmp_path / "masked_observations.json").exists() == (config.tomo_likelihood == "masked")


def test_masked_and_single_volume_options_are_validated():
    from relax.ppca_initial_model.config import Config

    with pytest.raises(ValueError, match="positive integer q"):
        Config(q=0)
    with pytest.raises(ValueError, match="single-volume"):
        Config(q=2, single_volume=True)
    with pytest.raises(ValueError, match="positive masked_mask_radius_px"):
        Config(tomo_likelihood="masked")
    with pytest.raises(ValueError, match="need tomo_likelihood masked"):
        Config(masked_mask_radius_px=5.0)
    with pytest.raises(ValueError, match="streamed engine without oversampling"):
        Config(tomo_likelihood="masked", masked_mask_radius_px=5.0, oversampling=1)
    assert Config(tomo_likelihood="masked", masked_mask_radius_px=5.0).holds_noise()
    assert not Config().holds_noise()


# ----------------------------------------------------------------------------------------------- given poses
def _pose_streams(particles, poses, volume, rotations, translations, *, masked):
    """Streams of the particles composed with ``poses`` and of the plain particles, both likelihoods' readers."""
    from relax.ppca_initial_model.pose_input import particles_at_poses

    n = particles.grid_size
    composed = particles_at_poses(particles, poses)
    if masked:
        noise = np.ones((1, n // 2 + 1), np.float32)
        kw = dict(mask_radius_px=n // 2 - 1, lam_rel=0.01, n_mc=1)
        composed = TiltParticles(**{**composed.__dict__,
                                    "volume_observations": prepare_volume_observations(composed, noise, **kw)})  # fmt: skip
    mu, W = jnp.asarray(volume), jnp.zeros((volume.size, 1), jnp.complex64)
    return composed, _streams(composed, mu, W, rotations, translations, n // 2 - 1, 1, image=not masked)


def _top_poses(stream, ids):
    tile, _, layout = read_tile(stream, ids, [None] * ids.size, collect_observation=False)
    kept, posterior = score_tile(stream, tile, layout["n_blocks"])
    return kept, posterior, np.asarray(tile.rows)


def test_image_route_at_a_given_pose_scores_as_the_full_grid_at_that_pose():
    """Composing each particle's frames with its pose and shifting its images by the pose's shift is the same image
    likelihood: the given pose's score (identity, zero shift) equals the plain stream's score at that grid pose."""
    from relax.ppca_initial_model.pose_input import InputPoses

    n = 16
    particles, volume, rotations, translations, truth = _noiseless_problem(n)
    rng = np.random.default_rng(3)
    _, half, ctf = particles.read(np.arange(particles.image_offsets[-1]))
    noisy = (np.asarray(half) + 0.3 * rng.standard_normal(half.shape)).astype(np.complex64)
    particles = _particles(n, noisy, np.asarray(ctf), particles.n_images)
    ids = np.arange(particles.n_images)
    # Each particle's given pose is another particle's grid pose, so scores are not all at the maximum.
    given = [((p + 2) % 6, (p + 3) % 5) for p in ids]
    poses = InputPoses(rotations[[r for r, _ in given]].astype(np.float64), translations[[t for _, t in given]].astype(np.float64))
    mu, W = jnp.asarray(volume), jnp.zeros((volume.size, 1), jnp.complex64)
    plain = _streams(particles, mu, W, rotations, translations, n // 2 - 1, 1, image=True)
    kept, _, rows = _top_poses(plain, ids)
    full = np.asarray(kept.score)
    _, fixed = _pose_streams(particles, poses, volume, np.eye(3, dtype=np.float32)[None], np.zeros((1, 3), np.float32),
                             masked=False)  # fmt: skip
    kept_fixed, _, _ = _top_poses(fixed, ids)
    position = {int(r): k for k, r in enumerate(rows[: len(rotations)])}
    expected = np.asarray([full[position[r], p, t] for p, (r, t) in enumerate(given)])
    assert_matches(np.asarray(kept_fixed.score)[0, :, 0], expected, rtol=1e-5)


@pytest.mark.parametrize("masked", [False, True])
def test_local_search_about_the_true_pose_peaks_at_the_identity(masked):
    """Noiseless particles given their true poses: over the local search's rotations (HEALPix order 2 centred on the
    identity, within 3 x 6 deg) and shifts within 1 px, every particle's top pose is the identity and zero shift."""
    from relax.ppca_initial_model.pose_input import InputPoses, local_rotations
    from relax.sampling import grids

    n = 16
    particles, volume, rotations, translations, truth = _noiseless_problem(n)
    poses = InputPoses(rotations[[r for r, _ in truth]].astype(np.float64), translations[[t for _, t in truth]].astype(np.float64))
    local, _, _ = local_rotations(grids.get_relion_hidden_rotation_grid(2, matrices=True), 6.0)
    identity = np.flatnonzero(np.abs(local - np.eye(3)).max(axis=(1, 2)) < 1e-6)
    assert identity.size == 1  # the centred grid holds the given pose
    shifts = grids.get_relion_translation_grid_3d(1.0, 1.0).astype(np.float32)
    _, stream = _pose_streams(particles, poses, volume, local, shifts, masked=masked)
    _, posterior, rows = _top_poses(stream, np.arange(particles.n_images))
    top = rows[np.asarray(posterior.top_rotation)]
    np.testing.assert_array_equal(top, np.full(particles.n_images, identity[0]))
    np.testing.assert_array_equal(shifts[np.asarray(posterior.top_translation)], np.zeros((particles.n_images, 3)))


@pytest.mark.parametrize(
    "options",
    [
        {"pose_mode": "fixed"},
        {"pose_mode": "local", "local_sigma_deg": 10.0, "tomo_likelihood": "masked", "masked_mask_radius_px": 6.0},
        {"pose_mode": "fixed", "tomo_likelihood": "masked", "masked_mask_radius_px": 6.0},
    ],
)
def test_controller_trains_at_given_poses(tmp_path, options):
    """Fixed and local given poses through the controller, both likelihoods; fixed mode's final poses are the given
    ones (particle_ids reorder the file's rows)."""
    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import run

    particles = _controller_particles()
    n = particles.n_images
    rng = np.random.default_rng(5)
    rotations = Rotation.random(n, random_state=6).as_matrix()
    translations = rng.normal(0, 1.0, (n, 3))
    ids = rng.permutation(n)
    np.savez(tmp_path / "poses.npz", rotations=rotations[ids], translations_px=translations[ids], particle_ids=ids)
    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    config = Config(
        q=2, iterations=2, stages=((1, 3, 1),), shift_range=1, shift_step=1, image_batch_size=3, rotation_block_size=32,
        final_poses=True, init_poses=str(tmp_path / "poses.npz"), **options,
    )  # fmt: skip
    try:
        state = run(particles, config, tmp_path / "run", {"test": True}, diameter_ang=100.0)
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    assert np.all(np.isfinite(np.asarray(state.theta)))
    embeddings = np.load(tmp_path / "run" / "embeddings.npz")
    order = np.asarray(embeddings["particle_ids"])
    if options["pose_mode"] == "fixed":
        assert_matches(embeddings["best_rotation_matrix"], rotations[order].astype(np.float32), rtol=1e-6)
        assert_matches(embeddings["best_translation_px"], translations[order].astype(np.float32), rtol=1e-6)
    else:
        angle = np.degrees(np.arccos(np.clip(
            (np.einsum("pab,pab->p", embeddings["best_rotation_matrix"], rotations[order]) - 1) / 2, -1, 1)))  # fmt: skip
        assert np.all(angle < 30.0 + 1e-3)  # inside the 3 sigma cone


@pytest.mark.parametrize("order", [2, 4])
def test_local_grid_is_centred_on_the_given_pose(order):
    """The local grid holds the identity, keeps the HEALPix grid's spacing (centring is an isometry) and only
    rotations within 3 sigma."""
    from relax.ppca_initial_model.pose_input import local_rotations
    from relax.sampling import grids

    grid = grids.get_relion_hidden_rotation_grid(order, matrices=True)
    local, log_prior, _ = local_rotations(grid, 3.0 * 2 ** (2 - order))
    angle = np.degrees(np.arccos(np.clip((np.trace(local, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    assert angle.min() < 1e-3 and angle.max() < 9.0 * 2 ** (2 - order)
    assert int(np.argmax(log_prior)) == int(np.argmin(angle))
    assert np.isclose(np.exp(log_prior.astype(np.float64)).sum(), 1.0, rtol=1e-5)


def test_pose_files_are_checked(tmp_path):
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.pose_input import read_pose_file

    np.savez(tmp_path / "bad.npz", rotations=np.tile(np.eye(3), (3, 1, 1)), translations_px=np.zeros((3, 3)),
             particle_ids=np.array([0, 0, 2]))  # fmt: skip
    with pytest.raises(ValueError, match="exactly once"):
        read_pose_file(tmp_path / "bad.npz", 3)
    np.savez(tmp_path / "mirror.npz", rotations=np.tile(np.diag([1.0, 1.0, -1.0]), (3, 1, 1)), translations_px=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="proper rotation"):
        read_pose_file(tmp_path / "mirror.npz", 3)
    with pytest.raises(ValueError, match="both init_poses and pose_mode"):
        Config(init_poses="x.npz")
    with pytest.raises(ValueError, match="tilt coarse pass"):
        Config(init_poses="x.npz", pose_mode="fixed", tilt_coarse_pass=True)


def test_exact_masked_mstep_residual_is_the_masked_gradient():
    """One particle, the identity rotation and one whole-pixel shift (posterior 1): with --masked-exact-mstep the
    residual is ``data phase_t - filt phase_t DFT(M F^-1[filt x^ conj(phase_t)]) / (c s)`` (the masked likelihood's
    gradient), evaluated here in float64 on the full box, on the window's kz > 0 voxels."""
    n = 10
    particles, _, _, _, _ = _noiseless_problem(n, n_particles=1, seed=6)
    rng = np.random.default_rng(2)
    half = (rng.standard_normal((particles.image_offsets[-1], n * (n // 2 + 1))) * 3).astype(np.complex64)
    particles = _particles(n, half, np.full(half.shape, 0.9, np.float32), 1)
    observations = prepare_volume_observations(particles, np.ones((1, n // 2 + 1), np.float32), mask_radius_px=3.0,
                                               n_mc=1)  # fmt: skip
    masked = TiltParticles(**{**particles.__dict__, "volume_observations": observations})
    x = _half_volume(rng, n)
    radius = 4
    # An integer shift: the shift is moved onto the mask (a band-limited roll), exact for whole pixels only.
    t = np.asarray([[1.0, -2.0, 1.0]], np.float32)
    grid = dict(
        rotations=np.eye(3, dtype=np.float32)[None], translations=t, rotation_log_prior=np.zeros(1, np.float32),
        translation_log_prior=np.zeros(1, np.float32), rotation_parent=np.zeros(1, np.int32),
        translation_parent=np.zeros(1, np.int32), n_coarse_rotations=1, n_coarse_translations=1,
        geometry=GeometryConfig(current_size=2 * radius, q=0, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=1, rotation_block_size=1), gemm_precision="fp32",
    )  # fmt: skip
    stream = prepare_volume_stream(masked, jnp.asarray(x), None, exact_masked_mstep=True, **grid)
    residual = np.asarray(accumulate_full_row_tile(stream, np.arange(1), [None]).residual_gradient)[:, 0]
    window = observations.radius2 <= radius**2
    indices = observations.indices[window]
    k = np.stack(np.unravel_index(indices, (n, n, n // 2 + 1)), -1) - np.asarray([n // 2, n // 2, 0])
    phase = np.exp(-2j * np.pi * k @ t[0].astype(np.float64) / n)
    scale = float(observations.scale[0])
    filt = np.sqrt(observations.filt2[0][window].astype(np.float64) * scale)
    with jax.enable_x64(True):
        G = np.zeros(n * n * (n // 2 + 1), np.complex128)
        G[indices] = filt * x[indices].astype(np.complex128) * np.conj(phase)
        full = np.asarray(ftu.half_volume_to_full_volume(jnp.asarray(G), (n, n, n))).reshape(n, n, n)
        u = np.real(np.asarray(ftu.get_idft3(jnp.asarray(full))))
        from relax.ppca_initial_model.masked_volume import sphere

        masked_u = sphere(n, 3.0).astype(np.float64) * u
        D = np.asarray(ftu.full_volume_to_half_volume(ftu.get_dft3(jnp.asarray(masked_u.astype(np.complex128))), (n, n, n))).reshape(-1)
    expected = observations.data[0][window] * phase - filt * phase * D[indices] / scale
    inner = k[:, 2] > 0
    assert_matches(residual[indices][inner], expected[inner].astype(np.complex64), rtol=1e-4)
    # The B-diag residual differs (the mask matters here).
    bdiag = observations.data[0][window] * phase - observations.filt2[0][window] * x[indices]
    assert np.abs(residual[indices][inner] - bdiag[inner]).max() > 1e-2 * np.abs(expected[inner]).max()


@pytest.mark.parametrize("likelihood", [{}, {"tomo_likelihood": "masked", "masked_mask_radius_px": 6.0}])
def test_centred_output_mean_is_the_population_average(tmp_path, likelihood):
    """--center-latents-output: model_centred.npz's mean is mu + W zbar of the run's final model and latents, the
    exported latents are centred, zbar is recorded, and the checkpoint (the EM state) keeps mu."""
    from relax.ppca_initial_model import checkpoint, iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import run

    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    particles = _controller_particles()
    config = Config(q=2, iterations=2, stages=((1, 3, 0),), shift_range=1, shift_step=1, image_batch_size=3,
                    rotation_block_size=32, center_latents_at_output=True, **likelihood)  # fmt: skip
    try:
        state = run(particles, config, tmp_path, {"test": True}, diameter_ang=100.0)
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    embeddings = np.load(tmp_path / "embeddings.npz")
    centred = np.load(tmp_path / "model_centred.npz")
    zbar = embeddings["zbar"]
    np.testing.assert_allclose(embeddings["z"].mean(axis=0), 0.0, atol=1e-5 * max(1.0, np.abs(zbar).max()))
    theta = np.asarray(state.theta)
    raw_z = embeddings["z"] + zbar
    population = theta[:, 0] + theta[:, 1:] @ raw_z.mean(axis=0).astype(theta.dtype)
    assert_matches(centred["theta"][:, 0], population, rtol=1e-5)
    assert_matches(centred["theta"][:, 1:], theta[:, 1:], rtol=1e-6)
    saved = checkpoint.load(tmp_path / "checkpoint_0002.npz", config, {"test": True})
    assert_matches(np.asarray(saved.theta), theta, rtol=1e-6)


def test_masked_noise_scale_scales_c_only():
    """--masked-noise-scale s: the per-particle operands scale as data / s and filt2 / s (c times s), nothing else."""
    n = 10
    particles, _, _, _, _ = _noiseless_problem(n, n_particles=2, seed=4)
    noise = np.ones((1, n // 2 + 1), np.float32)
    base = prepare_volume_observations(particles, noise, mask_radius_px=3.0, n_mc=1)
    hot = prepare_volume_observations(particles, noise, mask_radius_px=3.0, n_mc=1, noise_scale=1.5)
    assert_matches(hot.filt2 * 1.5, base.filt2, rtol=1e-5)
    assert_matches(hot.data * 1.5, base.data, rtol=1e-5)
    assert_matches(hot.y_norm * 1.5, base.y_norm, rtol=1e-5)


def test_hybrid_schedule_switches_likelihood_at_the_given_update(tmp_path):
    """--masked-from-update N: the image likelihood scores the updates before N and the masked one from N on (recorded
    per update), with the noise held throughout."""
    import json

    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import run

    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    particles = _controller_particles()
    config = Config(q=2, iterations=3, stages=((1, 3, 0),), shift_range=1, shift_step=1, image_batch_size=3,
                    rotation_block_size=32, tomo_likelihood="masked", masked_mask_radius_px=6.0, masked_from_update=3)  # fmt: skip
    try:
        run(particles, config, tmp_path, {"test": True}, diameter_ang=100.0)
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    rows = [json.loads(line) for line in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    assert [row["likelihood"] for row in rows] == ["image", "image", "masked"]
    assert all(row["noise_held"] for row in rows)
    assert len({json.dumps(row["noise"]) for row in rows}) == 1
    with pytest.raises(ValueError, match="masked_from_update"):
        Config(iterations=3, tomo_likelihood="masked", masked_mask_radius_px=6.0, masked_from_update=4)


def test_derived_calibration_scales_c_by_the_temperature():
    """--masked-calibrate: c times the recorded temperature tau (data, filt2, ||MV||^2/c divided by it); tau and its
    noise and signal ratios are finite and positive, and the signal ratio (2 E[Delta] over the c I model's value,
    data simulated through the forward operator) is near 1: the masked mean matches the template."""
    n = 10
    particles, _, _, _, _ = _noiseless_problem(n, n_particles=3, seed=4)
    noise = np.ones((1, n // 2 + 1), np.float32)
    base = prepare_volume_observations(particles, noise, mask_radius_px=3.5, n_mc=2)
    cal = prepare_volume_observations(particles, noise, mask_radius_px=3.5, n_mc=2, calibrate=True)
    k = cal.record["calibration_factor"]
    assert np.isfinite(k) and k > 0
    assert k == pytest.approx(cal.record["tau"]) and base.record["calibration_factor"] == 1.0
    assert base.record["tau"] == pytest.approx(k, rel=1e-5)
    assert all(np.isfinite(cal.record["noise_ratio_quantiles"])) and min(cal.record["noise_ratio_quantiles"]) > 0
    assert 0.5 < cal.record["signal_ratio_quantiles"][1] < 1.5
    assert_matches(cal.filt2 * k, base.filt2, rtol=1e-5)
    assert_matches(cal.data * k, base.data, rtol=1e-5)
    assert_matches(cal.y_norm * k, base.y_norm, rtol=1e-5)


def test_temperature_estimate_and_rescale_match_the_calibrated_observations():
    """estimate_temperature on every particle with the preprocessing's map (the particles' own) and seed is the
    recorded tau, and rescale_observations by tau turns the uncalibrated operands into the calibrated ones."""
    from relax.ppca_initial_model.masked_volume import estimate_temperature, rescale_observations

    n = 10
    from relax.ppca_initial_model.iteration_loop import _mean_map

    particles, volume, _, _, _ = _noiseless_problem(n, n_particles=3, seed=4)
    noise = np.ones((1, n // 2 + 1), np.float32)
    field = _mean_map(np.asarray(volume).reshape(-1, 1), (n,) * 3).astype(np.float32)  # the particles' own map
    base = prepare_volume_observations(particles, noise, mask_radius_px=3.5, n_mc=2, calibration_map=field)
    cal = prepare_volume_observations(particles, noise, mask_radius_px=3.5, n_mc=2, calibrate=True, calibration_map=field)
    estimate = estimate_temperature(particles, noise, field, mask_radius_px=3.5, n_mc=2, n_particles=10)
    assert cal.record["tau_usable"] and estimate["n_particles"] == 3
    assert estimate["tau"] == pytest.approx(cal.record["tau"], rel=1e-5)
    assert estimate["tau"] == pytest.approx(estimate["noise_ratio"] / estimate["signal_ratio"])
    scaled = rescale_observations(base, estimate["tau"])
    assert_matches(scaled.data, cal.data, rtol=1e-5)
    assert_matches(scaled.filt2, cal.filt2, rtol=1e-5)
    assert_matches(scaled.y_norm, cal.y_norm, rtol=1e-5)
    assert_matches(scaled.scale, cal.scale, rtol=1e-5)


def test_per_stage_temperature_is_logged_at_every_stage_start(tmp_path):
    """--masked-calibrate-per-stage: the temperature log holds the start-up tau and one re-estimate per stage start;
    the factor in effect stays finite and positive (a non-positive estimate keeps the previous one); the option needs
    --masked-calibrate."""
    import json

    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import run

    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    particles = _controller_particles()
    config = Config(q=2, iterations=3, stages=((1, 3, 0), (3, 4, 0)), shift_range=1, shift_step=1, image_batch_size=3,
                    rotation_block_size=32, tomo_likelihood="masked", masked_mask_radius_px=6.0, masked_calibrate=True,
                    masked_calibrate_per_stage=True)  # fmt: skip
    try:
        run(particles, config, tmp_path, {"test": True}, diameter_ang=100.0)
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    rows = [json.loads(line) for line in (tmp_path / "masked_temperature.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1, 3]
    assert rows[0]["start_up"] and rows[1]["stage"] == [4, 0]
    assert all(np.isfinite(row["calibration_factor"]) and row["calibration_factor"] > 0 for row in rows)
    assert rows[1]["calibration_factor"] == (rows[1]["tau"] if rows[1]["applied"] else rows[0]["tau"])
    with pytest.raises(ValueError, match="masked_calibrate_per_stage needs masked_calibrate"):
        Config(iterations=3, tomo_likelihood="masked", masked_mask_radius_px=6.0, masked_calibrate_per_stage=True)
    with pytest.raises(ValueError, match="need tomo_likelihood masked"):
        Config(iterations=3, masked_calibrate_per_stage=True)


def test_derived_calibration_without_a_usable_temperature_is_none():
    """--masked-calibrate with a map that carries no pose signal (zero: tau is not finite and positive) leaves c
    unscaled (factor 1) and records the estimate as unusable."""
    n = 10
    particles, _, _, _, _ = _noiseless_problem(n, n_particles=3, seed=4)
    noise = np.ones((1, n // 2 + 1), np.float32)
    zero = np.zeros((n,) * 3, np.float32)
    base = prepare_volume_observations(particles, noise, mask_radius_px=3.5, n_mc=2, calibration_map=zero)
    cal = prepare_volume_observations(particles, noise, mask_radius_px=3.5, n_mc=2, calibrate=True, calibration_map=zero)
    assert not cal.record["tau_usable"] and cal.record["calibration_factor"] == 1.0
    assert_matches(cal.filt2, base.filt2, rtol=1e-6)
    assert_matches(cal.data, base.data, rtol=1e-6)


def test_likelihood_temperature_is_the_noise_divided_by_it(tmp_path):
    """--likelihood-temperature beta: an E-step (scores, posteriors and the M-step's statistics) equals the E-step at
    temperature 1 with the noise times 1 / beta, and differs from the untempered one; the option is checked."""
    from relax.ppca_initial_model import checkpoint, iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import expectation_groups, run

    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    particles = _controller_particles()
    config = Config(q=2, iterations=1, stages=((1, 3, 0),), shift_range=1, shift_step=1, image_batch_size=3,
                    rotation_block_size=32)  # fmt: skip
    try:
        run(particles, config, tmp_path, {"test": True}, diameter_ang=100.0)
        state = checkpoint.load(tmp_path / "checkpoint_0000.npz", config, {"test": True})
        halves = [np.arange(0, 4), np.arange(4, 8)]
        hot = expectation_groups(particles, state, dataclasses.replace(config, likelihood_temperature=2.0), halves, 1,
                                 diameter_ang=100.0)  # fmt: skip
        ref = expectation_groups(particles, state, config, halves, 1, diameter_ang=100.0, noise_inflation=0.5)
        cold = expectation_groups(particles, state, config, halves, 1, diameter_ang=100.0)
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    for a, b, c in zip(hot, ref, cold):
        for field in ("lhs_tri", "residual_gradient"):
            assert_matches(np.asarray(getattr(a, field)), np.asarray(getattr(b, field)), rtol=1e-6)
        assert not np.allclose(np.asarray(a.residual_gradient), np.asarray(c.residual_gradient), rtol=1e-3)
    with pytest.raises(ValueError, match="likelihood_temperature must be finite and positive"):
        Config(likelihood_temperature=0.0)
    with pytest.raises(ValueError, match="likelihood_temperature applies to the image likelihood"):
        Config(iterations=3, tomo_likelihood="masked", masked_mask_radius_px=6.0, likelihood_temperature=2.0)
