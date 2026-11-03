"""Local pose likelihood derivatives and the pose feature map, on a small analytic volume (CPU, float64 diagnostic)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from helpers.float_compare import assert_matches

from relax.fourier.preprocessing import relion_half_translation_lattice, translation_phases
from relax.multibody import pose_neighborhoods as pn
from relax.multibody import pose_uq as uq
from relax.multibody.subtraction import relion_translate_2d
from relax.projection.projection import project_relion_projector_half_spectrum_centered_rows_at_indices as project

BOX = 32
PAD = 2
PIXEL = 1.7


def random_rotations(rng, n):
    q = np.linalg.qr(rng.standard_normal((n, 3, 3)))[0]
    q[:, :, 0] *= np.sign(np.linalg.det(q))[:, None]
    return q


def analytic_volume() -> jnp.ndarray:
    """RELION projector data of an asymmetric cluster of Gaussians, ``[z, y, x half]``: smooth at every cell."""
    n = BOX * PAD + 3
    half = n // 2 + 1
    centre = n // 2
    z, y, x = np.meshgrid(np.arange(n) - centre, np.arange(n) - centre, np.arange(half), indexing="ij")
    k = np.stack([x, y, z], axis=-1) / float(BOX * PAD)
    blobs = np.array([[3.0, 1.0, -2.0, 1.0], [-4.0, 2.0, 1.0, 0.7], [1.0, -5.0, 3.0, 1.2], [0.0, 3.5, -4.0, 0.5], [5.0, -2.0, 4.5, 0.9]])
    out = np.zeros(x.shape, dtype=np.complex128)
    for bx, by, bz, a in blobs:
        out += a * np.exp(-2.0 * np.pi**2 * 2.2**2 * np.sum(k**2, axis=-1)) * np.exp(-2j * np.pi * (k @ np.array([bx, by, bz])))
    return jnp.asarray(out)


def case(seed=2, n=6):
    rng = np.random.default_rng(seed)
    shape = (BOX, BOX)
    half = BOX * (BOX // 2 + 1)
    rows, cols = np.unravel_index(np.arange(half), (BOX, BOX // 2 + 1))
    ky, kx = rows - BOX // 2, cols
    window = np.flatnonzero((kx**2 + ky**2 <= (BOX // 2 - 1) ** 2) & ~((kx == 0) & (ky <= 0)))
    lattice = jnp.asarray(relion_half_translation_lattice(shape), dtype=jnp.float64)[window]
    consensus = random_rotations(rng, n)
    local = uq.BodyOperands(
        volume=analytic_volume(), r_max=BOX // 2, padding_factor=PAD, image_shape=shape, window=window, lattice=lattice, pixel_size=PIXEL,
        image=jnp.zeros((n, window.size), dtype=jnp.complex128), ctf=jnp.asarray(rng.uniform(-1.2, 1.2, (n, window.size))),
        weight=jnp.asarray(np.tile(rng.uniform(2e4, 4e4, (1, window.size)), (n, 1))), cameras=consensus @ random_rotations(rng, n), consensus=consensus,
    )
    return local, rng


def with_truth(ops, theta):
    """An operands record whose image is the noiseless prediction at ``theta`` about its stored pose."""
    from dataclasses import replace

    truth = uq.recentred(ops, theta)
    clean = uq.predictions(truth)
    return replace(ops, image=clean * translation_phases(jnp.asarray(theta[:, 3:]) / ops.pixel_size, ops.lattice))


def test_noiseless_polish_recovers_a_known_displacement():
    ops, rng = case()
    n = len(ops.cameras)
    theta = np.concatenate([rng.normal(0, 0.006, (n, 3)), rng.normal(0, 0.05, (n, 2))], axis=1)
    target = with_truth(ops, theta)
    local = uq.polish(target, iterations=6, step=0.3)
    assert_matches(local.correction, theta, rtol=5e-3)
    assert np.all(local.score_analysis < 1e-3 * np.maximum(local.score_stored, 1e-3))


def test_moving_model_and_moving_image_score_the_same():
    ops, rng = case()
    from dataclasses import replace

    n = len(ops.cameras)
    ops = replace(ops, image=uq.predictions(ops) * 1.1 + 0.002)
    theta = np.zeros((n, 5))
    theta[:, 3:] = rng.normal(0, 0.8, (n, 2))
    moved_model = np.asarray(uq.scores(ops, theta[:, None]))
    moved_image = np.asarray(uq.scores(uq.recentred(ops, theta), np.zeros((n, 1, 5))))
    assert_matches(moved_model, moved_image, rtol=1e-11)


def test_fisher_matches_differences_of_the_prediction_in_a_fixed_frame():
    ops, _ = case()
    n = len(ops.cameras)
    fisher = uq.expected_fisher(ops)
    h = 1e-4
    weight = np.asarray(ops.weight)
    J = []
    for k in range(5):
        step = np.zeros(5)
        step[k] = h * (1.0 if k < 3 else 30.0)

        def predict(sign):
            moved = uq.recentred(ops, np.tile(sign * step, (n, 1)))
            phase = translation_phases(jnp.tile(jnp.asarray(sign * step[3:] / ops.pixel_size), (n, 1)), ops.lattice)
            return np.asarray(uq.predictions(moved) * phase)

        J.append((predict(1.0) - predict(-1.0)) / (2.0 * step[k]))
    J = np.stack(J, axis=-1)
    direct = np.einsum("npa,np,npb->nab", J.conj(), weight, J).real
    assert_matches(fisher, direct, rtol=1e-2)  # measured 4.4e-3: trilinear cells, a few pixels straddle a boundary


def test_observed_hessian_equals_fisher_at_a_noiseless_optimum():
    ops, _ = case()
    target = with_truth(ops, np.zeros((len(ops.cameras), 5)))
    fisher = uq.expected_fisher(target)
    whiten = uq.whitening(fisher)
    _, gradient, hessian = uq.local_derivatives(target, whiten, 0.5)
    # Whitened units, so one means one standard deviation. Measured 0.0014 and 0.05: interpolation-cell asymmetry.
    assert np.max(np.abs(gradient)) < 0.01
    assert np.max(np.abs(hessian - np.eye(5))) < 0.1


def test_central_derivatives_are_exact_on_a_quadratic():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((2, 5, 5))
    H = A @ np.swapaxes(A, 1, 2)
    g = rng.standard_normal((2, 5))
    design = 0.7 * uq.central_design()
    values = 3.0 + design @ g.T + 0.5 * np.einsum("ma,nab,mb->nm", design, H, design).T
    values = values.T if values.shape[0] != 2 else values
    got_g, got_h = uq.central_derivatives(values, 0.7)
    assert_matches(got_g, g, rtol=1e-12)
    assert_matches(got_h, H, rtol=1e-12)


def moments(rng, count=800):
    points = rng.standard_normal((count, 3)) @ np.diag([30.0, 12.0, 21.0]) + np.array([14.0, -22.0, 9.0])
    weights = rng.uniform(0.1, 1.0, count)
    weights /= weights.sum()
    centroid = weights @ points
    centred = points - centroid
    return points, weights, centroid, (centred * weights[:, None]).T @ centred


def test_feature_distance_is_the_corresponding_point_rms():
    rng = np.random.default_rng(3)
    points, weights, centroid, covariance = moments(rng)
    Q = random_rotations(rng, 2)
    t = rng.normal(0, 10, (2, 3))
    L = uq.covariance_root(covariance)
    x = [np.concatenate([(Q[i] @ L).reshape(9), t[i] + Q[i] @ centroid]) for i in range(2)]
    moved = (t[0] + points @ Q[0].T) - (t[1] + points @ Q[1].T)
    assert_matches(np.sum((x[0] - x[1]) ** 2), np.sum(weights * np.sum(moved**2, axis=1)), rtol=1e-11)


def test_feature_jacobian_matches_finite_differences_with_its_coupling():
    rng = np.random.default_rng(4)
    _, _, centroid, covariance = moments(rng)
    L = uq.covariance_root(covariance)
    n = 8
    Q, A = random_rotations(rng, n), random_rotations(rng, n)
    P = uq.camera_projection(A)
    s = rng.normal(0, 3, (n, 2))
    J = uq.feature_jacobian(Q, P, L, centroid)

    def feature(theta):
        rotated = np.asarray(uq.exp_so3(jnp.asarray(theta[:, :3]))) @ Q
        return uq.observed_features(rotated, s + theta[:, 3:], P, L, centroid)

    h = 1e-6
    direct = np.zeros_like(J)
    for k in range(5):
        step = np.zeros((n, 5))
        step[:, k] = h
        direct[:, :, k] = (feature(step) - feature(-step)) / (2.0 * h)
    assert_matches(J, direct, rtol=1e-8)
    S = np.tile(np.diag([1e-4, 2e-4, 1e-4, 0.1, 0.2]) + 1e-5, (n, 1, 1))
    assert np.linalg.matrix_rank(uq.propagate(J, S)[0], tol=1e-8) == 5


def test_observed_shift_is_the_camera_projection_of_the_translation():
    rng = np.random.default_rng(5)
    n = 16
    consensus = random_rotations(rng, n)
    composed = consensus @ random_rotations(rng, n)
    pivot = np.array([7.3, -11.2, 4.1])
    origins = rng.uniform(-6, 6, (n, 2))
    body = rng.uniform(-0.9, 0.9, (n, 2))
    Q, lift, normal = pn.body_pose_lines(consensus, composed, pivot, origins, body, PIXEL)
    assert_matches(Q, np.swapaxes(consensus, 1, 2) @ composed, rtol=1e-12)
    P = uq.camera_projection(consensus)
    projected = (consensus @ pivot)[:, :2]
    integer = np.round(origins - projected)  # the fractions here are not half-integers
    assert_matches(np.einsum("nij,nj->ni", P, lift), PIXEL * (origins - projected - integer - body), rtol=1e-10)
    assert np.max(np.abs(np.einsum("nij,nj->ni", P, normal))) < 1e-12


def test_native_translations_center_an_off_center_body_at_the_physical_shift():
    """A noiseless blob placed where consensus plus ``s`` puts the pivot: the native ``r`` then ``f`` bring it to the
    origin, the sign the local parameter ``ds = -a df`` relies on."""
    size = 96
    rng = np.random.default_rng(6)
    grid = np.arange(size) - size // 2
    lattice = relion_half_translation_lattice((size, size))
    for _ in range(4):
        consensus = random_rotations(rng, 1)
        pivot = np.array([9.3, -7.1, 12.4])
        origin = rng.uniform(-4, 4, 2)
        body = rng.uniform(-0.8, 0.8, 2)
        projected = (consensus @ pivot)[0, :2]
        integer = np.round(origin - projected)
        physical = PIXEL * (origin - projected - integer - body)
        position = projected - origin + physical / PIXEL
        blob = np.exp(-((grid[None] - position[0]) ** 2 + (grid[:, None] - position[1]) ** 2) / (2.0 * 3.0**2))
        moved = relion_translate_2d(blob, integer.astype(int))
        spectrum = np.fft.fftshift(np.fft.rfft2(np.fft.ifftshift(moved)), axes=0).reshape(-1)
        shifted = spectrum * np.asarray(translation_phases(jnp.asarray(body), jnp.asarray(lattice, dtype=jnp.float64)))
        back = np.fft.fftshift(np.fft.irfft2(np.fft.ifftshift(shifted.reshape(size, size // 2 + 1), axes=0), s=(size, size)))
        x = np.sum(back.sum(axis=0) * grid) / back.sum()
        y = np.sum(back.sum(axis=1) * grid) / back.sum()
        assert np.hypot(x, y) < 1e-3


def test_finite_angle_moments_match_first_order_at_small_angles():
    rng = np.random.default_rng(7)
    _, _, centroid, covariance = moments(rng)
    L = uq.covariance_root(covariance)
    Q, A = random_rotations(rng, 3), random_rotations(rng, 3)
    P = uq.camera_projection(A)
    s = rng.normal(0, 3, (3, 2))
    S = np.tile(np.diag([3e-6, 3e-6, 3e-6, 0.04, 0.04]), (3, 1, 1))
    bias, spread = uq.finite_angle_moments(Q, s, P, L, centroid, S, samples=20000, seed=1)
    R = uq.propagate(uq.feature_jacobian(Q, P, L, centroid), S)
    assert np.max(np.abs(spread - R)) < 0.05 * np.max(np.abs(R))
    assert np.max(np.abs(bias)) < 0.05


def test_project_helper_keeps_the_prediction_axes():
    ops, _ = case(n=2)
    direct = project(ops.volume, jnp.asarray(ops.cameras), ops.image_shape, ops.r_max, ops.padding_factor, None, jnp.asarray(ops.window))
    assert_matches(uq.predictions(ops), ops.ctf * direct, rtol=1e-13)
