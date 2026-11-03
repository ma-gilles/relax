"""Local uncertainty of a body's fitted pose from its fixed-pose image likelihood, and its map into pose features.

An opt-in analysis of a finished multi-body run. Original poses stay the reconstruction inputs; nothing here writes
them. Everything is conditional on the consensus pose, the half maps, the other body's pose, the CTF, the scale, the
norm, the noise spectrum and the image mask: conditional means it cannot contain the error of any of those.

Local parameters ``theta = (omega_x, omega_y, omega_z, s_x, s_y)``: radians and physical image-plane angstroms. In the
common molecular frame the rotation is the LEFT increment ``Q(omega) = exp([omega]x) Q0``, so with consensus camera
``A`` the composed camera is ``B(omega) = A Q(omega) = exp([A omega]x) B0``. A physical displacement ``ds`` moves the
body in the image by ``+ds``, which enters the native correction as ``df = -ds / a`` with pixel size ``a``, the integer
shift and the fractional centring frozen (``relax.multibody.pose_neighborhoods`` holds ``s = a (e - f)``).

The objective is RELION's raw fine score, before priors and posterior normalisation, on the same prepared operands::

    ell(theta) = 1/2 sum_k w_k |Y_k - c_k mu_k(theta)|^2       w_k = 1 / sigma2_k, zero at DC

with ``Y`` the norm-corrected masked image translated by the stored ``r + f`` (fixed observation frame), ``c`` the CTF
times the group scale, ``mu`` the projection times the phase of the local shift, all in RELION's native FFT units.
``|phase| = 1``, so moving the model and moving the image give the same score and the same observed Hessian. The
expected Fisher matrix differentiates the prediction alone, ``F = Re(J^H W J)``, and needs no residual.

Trilinear interpolation makes ``mu`` only piecewise smooth, so its second derivative is not a usable point quantity.
The observed gradient and Hessian are therefore central differences of the score itself, in coordinates that whiten
``F``, with a step of ``RELATIVE_STEP`` expected standard deviations: the curvature at the scale of the uncertainty.
All of this is a float64 diagnostic: second differences of a float32 score are mostly rounding. The native float32
kernel is the numerical reference for the score.

Rotation-feature map (section "features"): with ``L L^T = C`` the mask's centred covariance and ``m`` its centroid about
the native pivot, ``y = [vec(Q L), s + P Q m]``, ``vec`` in C order (rows of ``Q L`` in turn).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from relax.fourier.preprocessing import relion_half_translation_lattice, translation_phases
from relax.projection.projection import project_relion_projector_half_spectrum_centered_rows_at_indices

PARAMETERS = 5
RELATIVE_STEP = 1.0  # central-difference step, in standard deviations of the expected Fisher matrix
FIT_CHECK = 2.0  # the local quadratic is checked at this many standard deviations of H along its eigenvectors
TRUST_RADIUS = 3.0  # the largest Newton step of a polish, in standard deviations
POLISH_ITERATIONS = 3

# Status bits of a particle, kept with every record; zero is qualified.
STATUS_NOT_POSITIVE = 1  # H has a nonpositive eigenvalue
STATUS_FLAT = 2  # H has a direction much flatter than F: its uncertainty is not local
STATUS_NOT_STATIONARY = 4  # the polished point is still not stationary
STATUS_POOR_QUADRATIC = 8  # the score leaves the quadratic within FIT_CHECK standard deviations
STATUS_TRUST_LIMITED = 16  # the polish hit its trust radius: a competing mode is possible
FLAT_EIGENVALUE = 0.1  # relative to F, the smallest eigenvalue of H in the whitened frame
STATIONARY_STEP = 0.25  # the last Newton step, standard deviations
QUADRATIC_RATIO = (0.5, 1.5)  # actual over predicted score rise, at FIT_CHECK


@dataclass(frozen=True)
class BodyOperands:
    """One body's prepared score of ``n`` particles in native units, over the score window's ``P`` pixels."""

    volume: jax.Array  # RELION projector data of the body's map, complex128
    r_max: int
    padding_factor: int
    image_shape: tuple
    window: np.ndarray  # [P] indices of the centred half spectrum
    lattice: jax.Array  # [P, 2] cycles per pixel
    pixel_size: float
    image: jax.Array  # [n, P] Y at the stored translation
    ctf: jax.Array  # [n, P] c
    weight: jax.Array  # [n, P] w
    cameras: np.ndarray  # [n, 3, 3] composed cameras B at the stored pose
    consensus: np.ndarray  # [n, 3, 3] A


def skew(vectors: jax.Array) -> jax.Array:
    """``[omega]x`` of ``[..., 3]``."""
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(*vectors.shape[:-1], 3, 3)


def exp_so3(vectors: jax.Array) -> jax.Array:
    """Rodrigues ``exp([omega]x)``; the small-angle series keeps it accurate at the tiny angles a Hessian uses."""
    angle2 = jnp.sum(vectors * vectors, axis=-1)[..., None, None]
    safe = jnp.where(angle2 > 1e-12, angle2, 1.0)
    angle = jnp.sqrt(safe)
    a = jnp.where(angle2 > 1e-12, jnp.sin(angle) / angle, 1.0 - angle2 / 6.0)
    b = jnp.where(angle2 > 1e-12, (1.0 - jnp.cos(angle)) / safe, 0.5 - angle2 / 24.0)
    K = skew(vectors)
    return jnp.eye(3, dtype=vectors.dtype) + a * K + b * (K @ K)


def log_so3(rotations: np.ndarray) -> np.ndarray:
    """The rotation vector of ``[..., 3, 3]``, radians."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)).as_rotvec().reshape(
        *np.shape(rotations)[:-2], 3
    )


# --- the score --------------------------------------------------------------------------------------------------


def _project(volume, cameras, image_shape, r_max, padding_factor, window):
    return project_relion_projector_half_spectrum_centered_rows_at_indices(
        volume, cameras, image_shape, r_max, padding_factor, None, window
    )


@partial(jax.jit, static_argnames=("image_shape", "r_max", "padding_factor"))
def _scores(volume, cameras, consensus, theta, shifts_px, image, ctf, weight, lattice, window, *, image_shape, r_max, padding_factor):
    """``ell`` at ``[n, M]`` local parameters ``theta`` from cameras ``[n, 3, 3]``, with the angstrom shifts already in
    pixels in ``shifts_px [n, M, 2]``."""
    n, m = theta.shape[:2]
    camera_vectors = jnp.einsum("nij,nmj->nmi", consensus, theta[..., :3])
    rows = exp_so3(camera_vectors) @ cameras[:, None]
    projected = _project(volume, rows.reshape(n * m, 3, 3), image_shape, r_max, padding_factor, window).reshape(n, m, -1)
    predicted = ctf[:, None] * projected * translation_phases(shifts_px, lattice)
    residual = image[:, None] - predicted
    return 0.5 * jnp.sum(weight[:, None] * (residual.real**2 + residual.imag**2), axis=-1)


def predictions(ops: BodyOperands) -> jax.Array:
    """The prediction ``c mu`` ``[n, P]`` at the stored pose."""
    return ops.ctf * _project(ops.volume, jnp.asarray(ops.cameras), ops.image_shape, ops.r_max, ops.padding_factor, jnp.asarray(ops.window))


def scores(ops: BodyOperands, theta) -> jax.Array:
    """``ell`` ``[n, M]`` at local parameters ``theta [n, M, 5]`` about the stored poses."""
    theta = jnp.asarray(theta, dtype=jnp.float64)
    return _scores(
        ops.volume, jnp.asarray(ops.cameras), jnp.asarray(ops.consensus), theta, theta[..., 3:] / ops.pixel_size, ops.image,
        ops.ctf, ops.weight, ops.lattice, jnp.asarray(ops.window), image_shape=ops.image_shape, r_max=ops.r_max,
        padding_factor=ops.padding_factor,
    )


@partial(jax.jit, static_argnames=("image_shape", "r_max", "padding_factor"))
def _prediction_jacobian(volume, cameras, consensus, ctf, lattice, window, pixel_size, *, image_shape, r_max, padding_factor):
    """``J = d(c mu) / d theta`` at ``theta = 0`` ``[n, P, 5]``, fixed observation frame: exact derivative of the
    trilinear interpolant for the rotations, analytic phase for the shifts."""

    def predict(omega):
        rows = exp_so3(jnp.einsum("nij,nj->ni", consensus, omega)) @ cameras
        return ctf * _project(volume, rows, image_shape, r_max, padding_factor, window)

    zero = jnp.zeros((cameras.shape[0], 3), dtype=jnp.float64)
    mu, rotation = jax.vmap(lambda k: jax.jvp(predict, (zero,), (zero.at[:, k].set(1.0),)), out_axes=(None, 0))(jnp.arange(3))
    shifts = [(-2j * jnp.pi * lattice[:, k] / pixel_size) * mu for k in range(2)]
    return jnp.moveaxis(jnp.concatenate([rotation, jnp.stack(shifts)]), 0, -1)


@jax.jit
def _weighted_gram(J, weight):
    weighted = J * weight[..., None]
    return jnp.einsum("npa,npb->nab", J.conj(), weighted).real


def expected_fisher(ops: BodyOperands) -> np.ndarray:
    """``F = Re(J^H W J)`` ``[n, 5, 5]`` at the stored pose."""
    J = _prediction_jacobian(
        ops.volume, jnp.asarray(ops.cameras), jnp.asarray(ops.consensus), ops.ctf, ops.lattice, jnp.asarray(ops.window),
        jnp.asarray(ops.pixel_size), image_shape=ops.image_shape, r_max=ops.r_max, padding_factor=ops.padding_factor,
    )
    return np.asarray(_weighted_gram(J, ops.weight))


def central_design() -> np.ndarray:
    """The 51 stencil points in units of the step: origin, ``+-e_a``, ``(+-e_a, +-e_b)``."""
    points = [np.zeros(PARAMETERS)]
    identity = np.eye(PARAMETERS)
    for a in range(PARAMETERS):
        points += [identity[a], -identity[a]]
    for a in range(PARAMETERS):
        for b in range(a + 1, PARAMETERS):
            points += [p * identity[a] + q * identity[b] for p in (1.0, -1.0) for q in (1.0, -1.0)]
    return np.asarray(points)


def central_derivatives(values: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian ``[n, 5], [n, 5, 5]`` from :func:`central_design` values ``[n, 51]``."""
    n = len(values)
    gradient, hessian = np.zeros((n, PARAMETERS)), np.zeros((n, PARAMETERS, PARAMETERS))
    centre = values[:, 0]
    index = 1 + 2 * PARAMETERS
    for a in range(PARAMETERS):
        plus, minus = values[:, 1 + 2 * a], values[:, 2 + 2 * a]
        gradient[:, a] = (plus - minus) / (2.0 * step)
        hessian[:, a, a] = (plus + minus - 2.0 * centre) / step**2
    for a in range(PARAMETERS):
        for b in range(a + 1, PARAMETERS):
            pp, pm, mp, mm = (values[:, index + k] for k in range(4))
            hessian[:, a, b] = hessian[:, b, a] = (pp + mm - pm - mp) / (4.0 * step**2)
            index += 4
    return gradient, hessian


def whitening(fisher: np.ndarray) -> np.ndarray:
    """``T [n, 5, 5]`` with ``T^T F T = I``, so ``theta = T u``."""
    return np.swapaxes(np.linalg.inv(np.linalg.cholesky(fisher)), 1, 2)


def local_derivatives(ops: BodyOperands, whiten: np.ndarray, step: float = RELATIVE_STEP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ell(0), g, H)`` in the whitened frame, by central differences of the score."""
    design = step * central_design()
    theta = np.einsum("nij,mj->nmi", whiten, design)
    values = np.asarray(scores(ops, theta))
    gradient, hessian = central_derivatives(values, step)
    return values[:, 0], gradient, hessian


def recentred(ops: BodyOperands, theta: np.ndarray) -> BodyOperands:
    """``ops`` with its reference pose moved by ``theta [n, 5]``: the camera composed and the image re-translated."""
    from dataclasses import replace

    theta = jnp.asarray(theta, dtype=jnp.float64)
    vectors = jnp.einsum("nij,nj->ni", jnp.asarray(ops.consensus), theta[:, :3])
    cameras = np.asarray(exp_so3(vectors) @ jnp.asarray(ops.cameras))
    phases = translation_phases(-theta[:, 3:] / ops.pixel_size, ops.lattice)
    return replace(ops, cameras=cameras, image=ops.image * phases)


@dataclass(frozen=True)
class LocalPose:
    """One batch's local analysis about the analysis pose. Gradient and Hessian are per local parameter; the
    uncertainty is ``covariance = H^-1`` only where ``status == 0``."""

    correction: np.ndarray  # [n, 5] analysis pose minus stored pose
    score_stored: np.ndarray  # [n]
    score_analysis: np.ndarray  # [n]
    gradient: np.ndarray  # [n, 5]
    hessian: np.ndarray  # [n, 5, 5]
    fisher: np.ndarray  # [n, 5, 5]
    whitened_eigenvalues: np.ndarray  # [n, 5] of H in F's frame, ascending
    last_step: np.ndarray  # [n] the last proposed Newton step, standard deviations
    fit_ratio: np.ndarray  # [n, 10] actual over predicted score rise at FIT_CHECK along H's eigenvectors
    status: np.ndarray  # [n]
    cameras: np.ndarray  # [n, 3, 3] analysis cameras


def _newton(gradient: np.ndarray, hessian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The whitened Newton step, positive-definite part only (a damped device, not information), capped at the trust
    radius, and whether the cap was reached."""
    values, vectors = np.linalg.eigh(hessian)
    safe = np.maximum(values, FLAT_EIGENVALUE)
    step = -np.einsum("nij,nj->ni", vectors, np.einsum("nji,nj->ni", vectors, gradient) / safe)
    length = np.linalg.norm(step, axis=1)
    limited = length > TRUST_RADIUS
    return step * np.minimum(1.0, TRUST_RADIUS / np.maximum(length, 1e-30))[:, None], limited


def polish(ops: BodyOperands, *, iterations: int = POLISH_ITERATIONS, step: float = RELATIVE_STEP) -> LocalPose:
    """A bounded Newton polish of the likelihood from the stored pose, each move kept only if the score falls, then the
    derivatives at the resulting point and the local checks. Analysis only."""
    n = len(ops.cameras)
    correction = np.zeros((n, PARAMETERS))
    limited = np.zeros(n, dtype=bool)
    current = ops
    start = None
    for _ in range(iterations):
        fisher = expected_fisher(current)
        whiten = whitening(fisher)
        value, gradient, hessian = local_derivatives(current, whiten, step)
        start = value if start is None else start
        move, capped = _newton(gradient, hessian)
        theta = np.einsum("nij,nj->ni", whiten, move)
        trial = np.asarray(scores(current, theta[:, None]))[:, 0]
        accept = trial < value
        limited |= capped & accept
        theta = theta * accept[:, None]
        current = recentred(current, theta)
        correction = _compose(ops, correction, theta)
    fisher = expected_fisher(current)
    whiten = whitening(fisher)
    value, gradient, hessian = local_derivatives(current, whiten, step)
    move, _ = _newton(gradient, hessian)
    last = np.linalg.norm(move, axis=1)
    values, vectors = np.linalg.eigh(hessian)
    directions = FIT_CHECK * vectors / np.sqrt(np.maximum(values, FLAT_EIGENVALUE))[:, None]
    probe = np.concatenate([directions, -directions], axis=2)  # [n, 5, 10]
    theta = np.einsum("nij,njm->nmi", whiten, probe)
    actual = np.asarray(scores(current, theta)) - value[:, None]
    predicted = np.sum(np.swapaxes(probe, 1, 2) * gradient[:, None], axis=2) + 0.5 * np.einsum(
        "nam,nab,nbm->nm", probe, hessian, probe
    )
    ratio = actual / predicted
    status = np.zeros(n, dtype=np.int32)
    status |= STATUS_NOT_POSITIVE * (values[:, 0] <= 0.0)
    status |= STATUS_FLAT * ((values[:, 0] > 0.0) & (values[:, 0] < FLAT_EIGENVALUE))
    status |= STATUS_NOT_STATIONARY * (last > STATIONARY_STEP)
    status |= STATUS_POOR_QUADRATIC * ((ratio.min(axis=1) < QUADRATIC_RATIO[0]) | (ratio.max(axis=1) > QUADRATIC_RATIO[1]))
    status |= STATUS_TRUST_LIMITED * limited
    inverse = np.linalg.inv(whiten)
    transposed = np.swapaxes(inverse, 1, 2)
    return LocalPose(
        correction=correction,
        score_stored=start,
        score_analysis=value,
        gradient=np.einsum("nji,nj->ni", inverse, gradient),
        hessian=transposed @ hessian @ inverse,
        fisher=fisher,
        whitened_eigenvalues=values,
        last_step=last,
        fit_ratio=ratio,
        status=status,
        cameras=current.cameras,
    )


def _compose(ops: BodyOperands, correction: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """``correction`` then ``theta``: left rotations compose as matrices, shifts add."""
    rotation = np.asarray(exp_so3(jnp.asarray(theta[:, :3])) @ exp_so3(jnp.asarray(correction[:, :3])))
    return np.concatenate([log_so3(rotation), correction[:, 3:] + theta[:, 3:]], axis=1)


def covariance_from_hessian(hessian: np.ndarray) -> np.ndarray:
    """``H^-1`` through its Cholesky factor; raises on a nonpositive ``H``, which a qualified particle cannot have."""
    root = np.linalg.cholesky(hessian)
    inverse = np.linalg.inv(root)
    return np.swapaxes(inverse, 1, 2) @ inverse


# --- features -----------------------------------------------------------------------------------------------------


def camera_projection(consensus: np.ndarray) -> np.ndarray:
    """``P_i = A_i[0:2, :]``, ``[n, 2, 3]``."""
    return np.asarray(consensus, dtype=np.float64)[:, :2, :]


def covariance_root(covariance: np.ndarray) -> np.ndarray:
    """``L`` with ``L L^T = C``, lower Cholesky; a degenerate mask raises here rather than inventing rotational error."""
    return np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))


def observed_features(rotations, shifts, P, root, centroid) -> np.ndarray:
    """``y = [vec(Q L), s + P Q m]`` ``[..., 11]``."""
    rotations = np.asarray(rotations, dtype=np.float64)
    head = (rotations @ root).reshape(*rotations.shape[:-2], 9)
    moved = np.einsum("...ij,j->...i", rotations, centroid)
    return np.concatenate([head, np.asarray(shifts) + np.einsum("...ij,...j->...i", P, moved)], axis=-1)


def feature_jacobian(rotations, P, root, centroid) -> np.ndarray:
    """``d y / d theta`` ``[n, 11, 5]`` at the analysis pose, with the rotation-translation coupling of ``P Q m``."""
    rotations = np.asarray(rotations, dtype=np.float64)
    n = len(rotations)
    generators = np.asarray(skew(jnp.eye(3, dtype=jnp.float64)))
    J = np.zeros((n, 11, PARAMETERS))
    for k in range(3):
        moved = generators[k] @ rotations
        J[:, :9, k] = (moved @ root).reshape(n, 9)
        J[:, 9:, k] = np.einsum("nij,njk,k->ni", P, moved, centroid)
    J[:, 9, 3] = J[:, 10, 4] = 1.0
    return J


def propagate(jacobian: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """``R_i = J S J^T``, rank at most five."""
    return jacobian @ covariance @ np.swapaxes(jacobian, 1, 2)


def finite_angle_moments(rotations, shifts, P, root, centroid, covariance, *, samples: int = 2000, seed: int = 0):
    """Exact-transform mean bias and covariance ``([n, 11], [n, 11, 11])`` of ``y`` under the local Gaussian."""
    n = len(rotations)
    rng = np.random.default_rng(seed)
    bias, spread = np.zeros((n, 11)), np.zeros((n, 11, 11))
    roots = np.linalg.cholesky(covariance)
    for i in range(n):
        draw = rng.standard_normal((samples, PARAMETERS)) @ roots[i].T
        moved = np.asarray(exp_so3(jnp.asarray(draw[:, :3]))) @ rotations[i]
        y = observed_features(moved, shifts[i] + draw[:, 3:], np.broadcast_to(P[i], (samples, 2, 3)), root, centroid)
        centre = observed_features(rotations[i], shifts[i], P[i], root, centroid)
        bias[i] = y.mean(axis=0) - centre
        spread[i] = np.cov(y.T)
    return bias, spread
