"""Subtomogram PPCA tiles against a brute-force float64 joint-Gaussian model (algorithm section 16).

The reference never forms the engine's inner-product statistics: for each particle and pose it
stacks the particle's visible tilt images into one real vector and evaluates
``log N(y; A mu, A W W^T A^T + D^-1)`` with the dense joint covariance, so the latent coordinate
is shared by the tilts by construction. Projections and adjoints use RECOVAR's slicing primitives
in float64; everything specific to tilt series (frames, shared latent, per-tilt CTF and 3D shifts,
invisible tilts) is evaluated independently here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
from helpers.float_compare import assert_matches
from recovar import core
from scipy.spatial.transform import Rotation

from relax.helpers.half_spectrum import make_half_image_weights, make_shell_indices_half
from relax.helpers.preprocessing import relion_half_translation_lattice
from relax.ppca_initial_model.tomo import TiltParticles, load_tilt_tile, tilt_shifts, tilt_tiles
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.engine import _enforce_augmented_x0
from relax.ppca_refinement.full_row_stream import accumulate_full_row_tile, prepare_full_row_stream

pytestmark = pytest.mark.unit

IMAGE_SHAPE = (8, 8)
VOLUME_SHAPE = (8, 8, 8)
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
NOISE = 40.0
FRAMES = Rotation.from_euler("zyx", [[85, -40, 0], [85, -15, 3], [85, 10, -2], [85, 35, 1]], degrees=True).as_matrix()
# Particle 1 does not see frame 2; particle 2 lists its images in another frame order.
IMAGE_FRAMES = [[0, 1, 2, 3], [0, 1, 3], [3, 0, 2, 1]]
TRANSLATIONS = np.asarray(
    [[0, 0, 0], [1.5, 0, 0], [-1.5, 0, 0], [0, 1.5, 0], [0, -1.5, 0], [0, 0, 1.5], [0, 0, -1.5]], np.float32
)


def _half_volume(rng, scale=1.0):
    real = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    full = np.fft.fftshift(np.fft.fftn(real)).astype(np.complex64)
    return (scale * np.asarray(ftu.full_volume_to_half_volume(full, VOLUME_SHAPE)).reshape(-1)).astype(np.complex64)


def _particles(images, ctf, image_frames, frames):
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in image_frames])]).astype(np.int64)

    def read(ids):
        ids = np.asarray(ids)
        return None, jnp.asarray(images[ids]), jnp.asarray(ctf[ids])

    return TiltParticles(
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        voxel_size=1.0,
        image_offsets=offsets,
        image_frame=np.concatenate(image_frames).astype(np.int64),
        particle_group=np.zeros(len(image_frames), np.int64),
        group_frames=(np.asarray(frames, np.float64),),
        read=read,
    )


def make_problem(q=2, image_frames=IMAGE_FRAMES, frames=FRAMES, translations=TRANSLATIONS, seed=5):
    rng = np.random.default_rng(seed)
    n_images = sum(len(f) for f in image_frames)
    images = (rng.standard_normal((n_images, N_HALF)) + 1j * rng.standard_normal((n_images, N_HALF))) * 6
    ctf = rng.uniform(0.4, 1.3, (n_images, N_HALF)).astype(np.float32)
    mu = _half_volume(rng)
    W = np.stack([_half_volume(rng, 0.4 / (j + 1)) for j in range(q)], axis=1)
    rotations = Rotation.random(5, random_state=11).as_matrix().astype(np.float32)
    rotation_log_prior = np.log(rng.dirichlet(np.ones(5))).astype(np.float32)
    translation_log_prior = (-np.sum(translations**2, axis=-1) / 6.0).astype(np.float32)
    particles = _particles(images.astype(np.complex64), ctf, image_frames, frames)
    stream = prepare_full_row_stream(
        particles,
        mu,
        W,
        noise_variance=np.full(IMAGE_SHAPE, NOISE, np.float32),
        rotations=rotations,
        translations=translations,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        rotation_parent=np.zeros(5, np.int32),
        translation_parent=np.zeros(len(translations), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=6, q=q, volume_domain="fourier_half"),
        # Two-row blocks leave a one-row remainder block.
        schedule=ScheduleConfig(image_batch_size=len(image_frames), rotation_block_size=2),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        tile_loader=load_tilt_tile,
        # Exact float32 GEMMs on every platform: the bands below are float32 against float64.
        gemm_precision="fp32",
    )
    return (
        particles,
        stream,
        dict(
            images=images,
            ctf=ctf,
            mu=mu,
            W=W,
            rotations=rotations,
            rotation_log_prior=rotation_log_prior,
            translation_log_prior=translation_log_prior,
        ),
    )


def brute_force(particles, stream, problem):
    """Float64 log-likelihood, posterior, latent moments and M-step statistics of the dense joint model."""
    with jax.enable_x64(True):
        return _brute_force(particles, stream, problem)


def _brute_force(particles, stream, p):
    static = stream.static
    window = np.asarray(stream.resolved.score_indices)
    weights = np.asarray(make_half_image_weights(IMAGE_SHAPE), np.float64)
    lattice = np.asarray(relion_half_translation_lattice(IMAGE_SHAPE), np.float64)
    theta = np.concatenate([p["mu"][:, None], p["W"]], axis=1).T.astype(np.complex128)  # (P, half volume)
    P = theta.shape[0]
    q = P - 1
    frames = particles.group_frames[0]
    R, T = len(p["rotations"]), len(TRANSLATIONS)

    def project(rotation):
        out = core.batch_slice_volume(
            jnp.asarray(theta),
            jnp.asarray(rotation[None], jnp.float64),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            half_volume=True,
            half_image=True,
            relion_texture_interp=False,
            max_r=static.projection_max_r,
        )
        return np.asarray(out)[:, 0, :]  # (P, n_half)

    tilt_rotation = {
        (r, k): frames[k] @ p["rotations"][r].astype(np.float64) for r in range(R) for k in range(len(frames))
    }
    projections = {key: project(rot) for key, rot in tilt_rotation.items()}
    D = weights[window] / NOISE  # pixel precision of the half-spectrum metric
    real = lambda z: np.concatenate([z.real, z.imag], axis=-1)  # noqa: E731
    n_particles = particles.n_images
    log_likelihood = 0.0
    embeddings = np.zeros((n_particles, q))
    rotation_mass = np.zeros(R)
    offset_second = 0.0
    rhs_images = {key: np.zeros((P, N_HALF), np.complex128) for key in projections}
    lhs_images = {key: np.zeros((P * (P + 1) // 2, N_HALF)) for key in projections}
    power = np.sum(np.abs(p["images"]) ** 2, axis=0)
    tri = list(zip(*np.triu_indices(P)))
    for i in range(n_particles):
        image_ids, _ = particles.particle_images([i])
        slots = particles.image_frame[image_ids]
        y = p["images"][image_ids].astype(np.complex128)
        C = p["ctf"][image_ids].astype(np.float64)
        outside = np.ones(N_HALF, bool)
        outside[window] = False
        offset = -0.5 * np.sum(np.abs(y[:, outside]) ** 2 * weights[outside] / NOISE)
        scores, moments = np.zeros((R, T)), {}
        for r in range(R):
            for t in range(T):
                shifts = tilt_shifts(frames, TRANSLATIONS[t : t + 1])[slots, 0].astype(np.float64)  # (n, 2)
                phase = np.exp(-2j * np.pi * shifts @ lattice[window].T)  # (n, F)
                shifted = y[:, window] * phase
                A = np.stack([C[n, window] * projections[(r, k)][:, window] for n, k in enumerate(slots)], axis=1)
                residual = real(shifted - A[0]).reshape(-1)
                B = np.stack([real(A[j]).reshape(-1) for j in range(1, P)], axis=1)
                d = np.tile(np.concatenate([D, D]), len(slots))
                cov = np.diag(1 / d) + B @ B.T
                solve_r, solve_B = np.linalg.solve(cov, residual), np.linalg.solve(cov, B)
                _, logdet_cov = np.linalg.slogdet(cov)
                # The engine's absolute score drops log det D^-1 and keeps the image energy outside the window.
                scores[r, t] = -0.5 * residual @ solve_r - 0.5 * (logdet_cov + np.sum(np.log(d))) + offset
                mean = B.T @ solve_r
                covariance = np.eye(q) - B.T @ solve_B
                moments[r, t] = (mean, covariance, shifted, A)
        log_prior = p["rotation_log_prior"][:, None].astype(np.float64) + p["translation_log_prior"][None, :]
        total = scores + log_prior
        log_z = np.max(total) + np.log(np.sum(np.exp(total - np.max(total))))
        gamma = np.exp(total - log_z)
        log_likelihood += log_z
        rotation_mass += gamma.sum(axis=1)
        offset_second += np.sum(gamma * np.sum(TRANSLATIONS.astype(np.float64) ** 2, axis=-1)[None])
        for (r, t), (mean, covariance, shifted, A) in moments.items():
            g = gamma[r, t]
            embeddings[i] += g * mean
            alpha = np.concatenate([[1.0], mean])
            G = np.outer(alpha, alpha)
            G[1:, 1:] += covariance
            for n, k in enumerate(slots):
                y_recon = shifted[n] * C[n, window] / NOISE
                rhs_images[(r, k)][:, window] += g * alpha[:, None] * y_recon[None]
                for c, (a, b) in enumerate(tri):
                    lhs_images[(r, k)][c, window] += g * G[a, b] * C[n, window] ** 2 / NOISE
                predicted = alpha @ A[:, n]  # CTF-weighted projection of mu + W m
                loading = A[1:, n]  # (q, F)
                expected = (
                    np.abs(shifted[n] - predicted) ** 2
                    + np.einsum("jf,jl,lf->f", loading.conj(), covariance, loading).real
                )
                power[window] += g * (expected - np.abs(shifted[n]) ** 2)

    def adjoint(images, rotations):
        return np.asarray(
            core.adjoint_slice_volume(
                jnp.asarray(images),
                jnp.asarray(rotations),
                IMAGE_SHAPE,
                VOLUME_SHAPE,
                "linear_interp",
                half_image=True,
                half_volume=True,
                max_r=static.backprojection_max_r,
            )
        )

    keys = list(projections)
    rotations = np.stack([tilt_rotation[key] for key in keys])
    residual = np.zeros((P, theta.shape[1]), np.complex128)
    lhs = np.zeros((len(tri), theta.shape[1]))
    for c in range(P):
        images = np.stack(
            [
                rhs_images[key][c]
                - sum(lhs_images[key][tri.index((min(c, b), max(c, b)))] * projections[key][b] for b in range(P))
                for key in keys
            ]
        )
        images[:, np.setdiff1d(np.arange(N_HALF), window)] = 0
        residual[c] = adjoint(images, rotations)
    for c in range(len(tri)):
        lhs[c] = adjoint(np.stack([lhs_images[key][c] for key in keys]).astype(np.complex128), rotations).real
    lhs = np.asarray(_enforce_augmented_x0(jnp.asarray(lhs, jnp.complex128), VOLUME_SHAPE).real)
    shells = np.asarray(make_shell_indices_half(IMAGE_SHAPE))
    residual_num = np.zeros(shells.max() + 1)
    np.add.at(residual_num, shells, weights * power)
    residual_den = np.zeros(shells.max() + 1)
    np.add.at(residual_den, shells, weights * len(particles.image_frame))
    return dict(
        log_likelihood=log_likelihood,
        embeddings=embeddings,
        rotation_mass=rotation_mass,
        offset_second_sum_px2=offset_second,
        lhs_tri=lhs.T,
        residual_gradient=residual.T,
        residual_num=residual_num,
        residual_den=residual_den,
    )


@pytest.fixture(scope="module")
def problem():
    particles, stream, arrays = make_problem()
    return particles, stream, arrays, brute_force(particles, stream, arrays)


def test_tilt_tile_matches_brute_force_joint_gaussian(problem):
    particles, stream, arrays, truth = problem
    actual = accumulate_full_row_tile(stream, np.arange(3), [None] * 3)
    # float32 engine against float64: the largest band measured over seeds 5-7 is 4.0e-6 (lhs_tri,
    # seed 6); 2e-5 leaves a 5x margin.
    rtol = 2e-5
    assert_matches(actual.log_likelihood, truth["log_likelihood"], rtol=rtol)
    assert_matches(np.asarray(actual.embeddings), truth["embeddings"], rtol=rtol)
    assert_matches(actual.diagnostics["rotation_mass"], truth["rotation_mass"], rtol=rtol)
    assert_matches(actual.diagnostics["offset_second_sum_px2"], truth["offset_second_sum_px2"], rtol=rtol)
    assert_matches(np.asarray(actual.lhs_tri), truth["lhs_tri"], rtol=rtol)
    assert_matches(np.asarray(actual.residual_gradient), truth["residual_gradient"], rtol=rtol)
    assert_matches(np.asarray(actual.residual_num), truth["residual_num"], rtol=rtol)
    assert_matches(np.asarray(actual.residual_den), truth["residual_den"], rtol=1e-6)
    np.testing.assert_array_equal(actual.original_image_ids, np.arange(3))


def test_summed_per_tilt_scores_are_a_different_model(problem):
    """The shared latent matters: per-tilt marginals (one z per tilt) give another likelihood."""
    particles, stream, arrays, truth = problem
    separate = 0.0
    for image in range(len(particles.image_frame)):
        one = _particles(
            arrays["images"][image : image + 1].astype(np.complex64),
            arrays["ctf"][image : image + 1],
            [[particles.image_frame[image]]],
            FRAMES,
        )
        one_stream = stream._replace(dataset=one)
        separate += accumulate_full_row_tile(one_stream, [0], [None]).log_likelihood
    assert abs(separate - truth["log_likelihood"]) > 1e-3 * abs(truth["log_likelihood"])


def test_hidden_tilt_equals_a_zero_tilt():
    """A frame the particle does not see contributes what a zero image with a zero CTF does."""
    hidden, stream, arrays = make_problem(image_frames=[[0, 1, 3]], seed=8)
    images = np.insert(arrays["images"], 2, 0, axis=0).astype(np.complex64)
    ctf = np.insert(arrays["ctf"], 2, 0, axis=0)
    zero = _particles(images, ctf, [[0, 1, 2, 3]], FRAMES)
    a = accumulate_full_row_tile(stream, [0], [None])
    b = accumulate_full_row_tile(stream._replace(dataset=zero), [0], [None])
    for name in ("embeddings", "lhs_tri", "residual_gradient", "residual_num"):
        assert_matches(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)))
    assert_matches(a.log_likelihood, b.log_likelihood)
    assert np.all(np.asarray(b.residual_den) > np.asarray(a.residual_den))  # the zero image is one more observation


def test_identity_frame_tile_is_the_single_particle_tile():
    """One identity frame and in-plane shifts give the single-particle stream's statistics."""
    from test_full_fine_streaming import _TinyData

    planar = TRANSLATIONS[np.abs(TRANSLATIONS[:, 2]) == 0]
    particles, stream, arrays = make_problem(image_frames=[[0], [0], [0]], frames=np.eye(3)[None], translations=planar)
    spa = prepare_full_row_stream(
        _TinyData(arrays["images"].astype(np.complex64)),
        arrays["mu"],
        arrays["W"],
        noise_variance=np.full(IMAGE_SHAPE, NOISE, np.float32),
        rotations=arrays["rotations"],
        translations=planar[:, :2],
        rotation_log_prior=arrays["rotation_log_prior"],
        translation_log_prior=arrays["translation_log_prior"],
        rotation_parent=np.zeros(5, np.int32),
        translation_parent=np.zeros(len(planar), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=6, q=2, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=3, rotation_block_size=2),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
    )
    # _TinyData's identity CTF: give the tilt particles the same unit CTF.
    unit = _particles(
        arrays["images"].astype(np.complex64), np.ones_like(arrays["ctf"]), [[0], [0], [0]], np.eye(3)[None]
    )
    a = accumulate_full_row_tile(stream._replace(dataset=unit), np.arange(3), [None] * 3)
    b = accumulate_full_row_tile(spa, np.arange(3), [None] * 3)
    for name in ("embeddings", "lhs_tri", "residual_gradient", "residual_num", "residual_den"):
        assert_matches(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)), rtol=2e-6)
    assert_matches(a.log_likelihood, b.log_likelihood)


def test_tiles_keep_one_tilt_group():
    particles = _particles(np.zeros((4, N_HALF), np.complex64), np.ones((4, N_HALF), np.float32), [[0]] * 4, FRAMES)
    particles = TiltParticles(**{**particles.__dict__, "particle_group": np.asarray([1, 0, 1, 1])})
    tiles = tilt_tiles(particles, np.asarray([3, 1, 0, 2]), 2)
    assert [t.tolist() for t in tiles] == [[3, 0], [2], [1]]


@pytest.mark.parametrize("optimizer", ["vdam", "momentum_sgd"])
def test_controller_runs_subtomogram_particles(tmp_path, optimizer):
    """Two all-particle updates and the final embedding through the InitialModel controller."""
    import json

    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import run

    # Other tests monkeypatch the rotation grid behind these per-order caches.
    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()

    n, n_particles = 16, 8
    rng = np.random.default_rng(2)
    image_frames = [[0, 1, 2], [0, 2], [1, 2, 0], [0, 1, 2]] * 2
    n_images = sum(len(f) for f in image_frames)
    raw = rng.standard_normal((n_images, n, n)).astype(np.float32)
    half = np.asarray(ftu.get_dft2_real(raw)).reshape(n_images, -1).astype(np.complex64)
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in image_frames])]).astype(np.int64)
    particles = TiltParticles(
        image_shape=(n, n),
        volume_shape=(n, n, n),
        voxel_size=10.0,
        image_offsets=offsets,
        image_frame=np.concatenate(image_frames).astype(np.int64),
        particle_group=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        group_frames=(FRAMES[:3], FRAMES[1:]),
        read=lambda ids: (raw[ids], jnp.asarray(half[ids]), jnp.full((len(ids), half.shape[1]), 0.8, jnp.float32)),
    )
    config = Config(
        q=2,
        iterations=2,
        stages=((1, 3, 0),),
        oversampling=0,
        stream_coarse_recompute=True,
        shift_range=1,
        shift_step=1,
        image_batch_size=3,
        rotation_block_size=32,
        optimizer=optimizer,
    )
    state = run(particles, config, tmp_path, {"test": True}, diameter_ang=120.0)
    rows = [json.loads(line) for line in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1, 2]
    assert all(sum(row["half_counts"]) == n_particles for row in rows)
    assert np.all(np.isfinite(np.asarray(state.theta))) and np.all(np.asarray(state.noise) > 0)
    embeddings = np.load(tmp_path / "embeddings.npz")
    np.testing.assert_array_equal(np.sort(embeddings["particle_ids"]), np.arange(n_particles))
    assert embeddings["z"].shape == (n_particles, 2)
