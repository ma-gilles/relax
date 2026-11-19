"""The masked-volume likelihood of subtomogram particles (opt-in ``Config.tomo_likelihood = "masked"``).

Each particle is observed through a fixed linear map of its tilt images (the masked-tomo pilot's arm B,
``relax/masked_tomo/likelihoods.py``; design note ``masked-tomo/notes/ABINITIO.md`` of the project):

``V = M F^-1 [(D + lam)^-1/2 b]``, ``b = sum_f P_f^* (CTF_f y_f / sigma^2)``, ``D = sum_f P_f^* (CTF_f^2 / sigma^2)``

with ``P_f`` the slice at the tilt's ``Aproj_f`` (RECOVAR's projector and its adjoint, theta's own half-volume
convention), ``M`` a sphere about the box centre and ``lam`` a fixed fraction of the mean observed ``D``. Its mean
at pose ``(R, t)`` is ``M F^-1 [filt x^(k @ R) phase_t(k)]`` with ``filt = D / sqrt(D + lam)`` (labelled
approximation ``A^* Sigma^-1 A ~ D``) and its covariance ``c I`` on the mask (labelled; ``c`` the Monte Carlo voxel
variance of ``V`` under the noise model times the pilot's dof factor).

The inner product with the masked observation is exact in Fourier space; the Gram drops the mask (``||F x||^2``
instead of ``||M F x||^2``: "B-diag", the pilot's Fourier prefilter approximation). The likelihood is then the
image route's with one 3D frame per particle: tile operand ``filt A^ phase_t / s`` for the shifted images and
``filt^2 / (c s)`` for ``CTF^2 / sigma^2`` (``A^ = DFT(M V / c)``, ``s`` the DFT's Parseval factor), the 3D gather
``x^(k @ R)`` for the slices and its adjoint for the backprojection (``full_row_kernels`` with
``StreamStatic.observation = "volume"``). Everything else in the streamed engine (pose grids, priors, latent
epilogue, posterior, pass 2, the update) is shared.

The noise spectrum defines the fixed map, so a masked run keeps the start-up noise estimate.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import recovar.core.fourier_transform_utils as ftu
from recovar.core import relion_interp
from recovar.reconstruction import noise as noise_utils
from recovar.reconstruction.noise import make_radial_noise

from relax.fourier.preprocessing import translation_phases
from relax.ppca_refinement.dense_blocks import coerce_augmented_half_volumes
from relax.ppca_refinement.full_row_kernels import (
    FullRowStream,
    StreamArrays,
    StreamStatic,
    TileArrays,
    real_imag,
    resolve_gemm_precision,
)
from relax.ppca_refinement.full_row_stream import device_block_starts, tile_size_bucket, tile_support
from relax.ppca_refinement.occupancy import check_occupancy


@dataclasses.dataclass(frozen=True)
class VolumeObservations:
    """Every particle's masked-volume operands on the half ball ``|k| <= radius`` of the half volume.

    ``indices`` ``(F,)`` are the ball's half-volume voxels (ascending), ``radius2`` their squared radii;
    ``data`` ``(n, F)`` complex64 is ``filt A^ / s`` and ``filt2`` ``(n, F)`` float32 ``filt^2 / (c s)`` per particle
    (unweighted: the Hermitian weight is applied per stage window); ``y_norm`` ``(n,)`` is ``||M V||^2 / c``.
    ``record`` holds the preprocessing's parameters and per-particle summaries.
    """

    indices: np.ndarray
    radius2: np.ndarray
    data: np.ndarray
    filt2: np.ndarray
    y_norm: np.ndarray
    record: dict
    # Each particle's c s (noise variance times the DFT's Parseval factor), for the exact masked M-step.
    scale: np.ndarray | None = None


def sphere(box: int, radius: float) -> np.ndarray:
    g = np.indices((box,) * 3) - box // 2
    return (np.sum(g**2, 0) <= radius**2).astype(np.float32)


def half_ball(volume_shape, radius: int):
    """Half-volume voxels with ``|k| <= radius`` (ascending flat indices) and their squared radii."""
    n0, n1, n2 = (int(n) for n in volume_shape)
    i0, i1, i2 = np.meshgrid(np.arange(n0) - n0 // 2, np.arange(n1) - n1 // 2, np.arange(n2 // 2 + 1), indexing="ij")
    r2 = (i0**2 + i1**2 + i2**2).reshape(-1)
    indices = np.flatnonzero(r2 <= radius**2).astype(np.int32)
    return indices, r2[indices].astype(np.int32)


def half_ball_weights(indices, volume_shape) -> np.ndarray:
    """Hermitian weights of half-volume voxels for full-spectrum sums: 1 on the ``kz = 0`` plane (both partners are
    stored), 2 elsewhere (the ball excludes the Nyquist plane)."""
    n2 = int(volume_shape[2])
    return np.where(np.asarray(indices) % (n2 // 2 + 1) == 0, 1.0, 2.0).astype(np.float32)


def parseval_factor(box: int) -> float:
    """``s`` with ``sum_x u v = sum_k conj(U) V / s`` for RECOVAR's centred DFT (``norm="backward"``: ``box^3``)."""
    e = np.random.default_rng(0).standard_normal((box,) * 3)
    return float(np.sum(np.abs(np.asarray(ftu.get_dft3(jnp.asarray(e, jnp.float64)))) ** 2) / np.sum(e**2))


def dof_ratio(s, weights) -> jax.Array:
    """In-band coefficients per effective degree of freedom of the whitened noise, ``N sum s^2 / (sum s)^2``,
    ``s = D / (D + lam)`` over the half ball with Hermitian weights (``relax.masked_tomo.likelihoods.dof_ratio``)."""
    return jnp.sum(weights) * jnp.sum(weights * s**2) / jnp.sum(weights * s) ** 2


# The derived temperature's wrong poses (angle in degrees, axis): a small, a medium and a large rotation.
CALIBRATION_ROTATIONS = ((10.0, (1.0, 0.0, 0.0)), (30.0, (0.0, 1.0, 0.0)), (180.0, (1.0, 1.0, 1.0)))


def _gridding_correction(box: int) -> np.ndarray:
    """Real-space ``1 / sinc^2`` per axis for a trilinear Fourier interpolation on a ``box`` grid."""
    x = (np.arange(box) - box // 2) / box
    w = np.sinc(x) ** 2
    return (1.0 / (w[:, None, None] * w[None, :, None] * w[None, None, :])).astype(np.float32)


def _rotated(volume: np.ndarray, angle_deg: float, axis) -> np.ndarray:
    """``volume`` rotated by ``angle_deg`` about ``axis`` through the box centre ``n // 2`` (trilinear)."""
    from scipy.ndimage import affine_transform
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(np.deg2rad(angle_deg) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()
    centre = np.full(3, volume.shape[0] // 2, float)
    return affine_transform(volume, R, offset=centre - R @ centre, order=1, mode="constant").astype(np.float32)


@functools.partial(jax.jit, static_argnames=("image_shape", "volume_shape", "n_mc"))
def _particle_operands(half, ctf, frames, nv, mask, ball, weights, key, lam_rel, parseval, noise_scale, fields, source, *,
                       image_shape, volume_shape, n_mc):
    """One particle's ``(data, filt2, y_norm, c, lam, noise_ratio, signal_ratio)`` on the half ball ``ball`` from its
    tilt half spectra ``half`` ``(K, n_half)``, CTF rows ``ctf`` (zero rows for padding tilts), frames ``(K, 3, 3)``
    and noise ``nv`` ``(n_half,)``; ``fields`` ``(1 + J, n, n, n)`` a map and its ``J`` rotations (the derived
    temperature's poses, see below) and ``source`` the half DFT of ``fields[0]`` padded 2x and gridding-corrected (the
    projector's input for the simulated signal)."""
    n = int(volume_shape[0])
    max_r = n // 2 - 1

    def backproject(images):
        return relion_interp.backproject(
            images, frames, image_shape, volume_shape, order=1, half_volume=True, half_image=True, max_r=max_r
        )

    D = jnp.real(backproject((ctf * ctf / nv).astype(jnp.complex64)))
    inside = jnp.zeros(D.shape, bool).at[ball].set(True)
    D = jnp.where(inside, D, 0.0)
    observed = inside & (D > 0)
    lam = lam_rel * jnp.sum(jnp.where(observed, D, 0.0)) / jnp.sum(observed)
    whiten = jnp.where(inside, 1.0 / jnp.sqrt(D + lam), 0.0)

    def measure(images):
        b = jnp.where(inside, backproject((ctf * images / nv).astype(jnp.complex64)), 0)
        full = ftu.half_volume_to_full_volume(b * whiten, volume_shape)
        return jnp.real(ftu.get_idft3(full.reshape(volume_shape)))

    V = measure(half) * mask
    # Derived temperature (calibration): ``fields[0]`` is a map in the particle's frame (the start-up mean) and
    # ``fields[1:]`` its rotations. Pose score differences Delta = s(fields[0]) - s(fields[j]) for data simulated from
    # fields[0] through the forward operator (2x padded, gridding-corrected trilinear projection) (CTF, tilt projections, backprojection, whitening, mask), with the noise
    # by Monte Carlo: tau = Var(Delta) / (2 E[Delta]), 1 for a calibrated likelihood. Reported per particle as
    # noise ratio Var(Delta) / nominal and signal ratio 2 E[Delta] / nominal, nominal = ||M g||^2 / c0 the c I
    # model's value for the template difference g (tau = noise ratio / signal ratio).
    filt_full = ftu.half_volume_to_full_volume(jnp.where(inside, D * whiten, 0.0).astype(jnp.complex64), volume_shape)
    templates = jnp.real(ftu.get_idft3(filt_full.reshape(volume_shape)[None] * ftu.get_dft3(fields.astype(jnp.complex64))))
    diffs = mask[None] * (templates[:1] - templates[1:])
    padded_shape = tuple(2 * v for v in volume_shape)
    clean = ctf * relion_interp.project(
        source, frames, image_shape, padded_shape, order=1, half_volume=True, half_image=True, max_r=max_r
    )
    signal = measure(clean) * mask
    variance = 0.0
    projections = []
    for _ in range(n_mc):
        key, sub = jax.random.split(key)
        white = jax.random.normal(sub, (half.shape[0],) + tuple(image_shape), jnp.float32)
        noise = ftu.get_dft2_real(white).reshape(half.shape[0], -1) * jnp.sqrt(nv / (image_shape[0] * image_shape[1]))
        Ve = measure(noise)
        variance = variance + jnp.sum(mask * Ve**2) / jnp.sum(mask)
        projections.append(jnp.sum(diffs * Ve[None], axis=(1, 2, 3)))
    s_ball = D[ball] / (D[ball] + lam)
    c0 = variance / n_mc * dof_ratio(s_ball, weights)
    nominal = jnp.sum(diffs**2, axis=(1, 2, 3)) / c0
    gram = jnp.sum(templates**2, axis=(1, 2, 3))  # B-diag: the unmasked Gram
    mean_delta = (jnp.sum(signal[None] * diffs, axis=(1, 2, 3)) - 0.5 * (gram[0] - gram[1:])) / c0
    var_delta = jnp.mean(jnp.stack(projections) ** 2, axis=0) / c0**2  # the noise term has mean zero
    noise_ratio = jnp.mean(var_delta / nominal)
    signal_ratio = jnp.mean(2 * mean_delta / nominal)
    c = c0 * noise_scale
    A = ftu.full_volume_to_half_volume(ftu.get_dft3((V / c).astype(jnp.complex64)), volume_shape).reshape(-1)
    filt = D[ball] * whiten[ball]
    data = (filt * A[ball] / parseval).astype(jnp.complex64)
    filt2 = (filt * filt / (c * parseval)).astype(jnp.float32)
    return data, filt2, jnp.sum(V * V) / c, c, lam, noise_ratio, signal_ratio


def prepare_volume_observations(particles, noise, *, mask_radius_px: float, lam_rel: float = 0.1, n_mc: int = 2,
                                seed: int = 0, noise_scale: float = 1.0, calibrate: bool = False,
                                calibration_map: np.ndarray | None = None) -> VolumeObservations:
    """:class:`VolumeObservations` of every particle of ``particles`` (a ``TiltParticles``) under the noise rows
    ``noise`` ``(G, S)`` (one per noise group; held for the whole run). The Monte Carlo draws of particle ``p`` use
    ``PRNGKey(seed * 100003 + p)``, so the operands are a function of the particles, the noise and these options.
    ``noise_scale`` multiplies ``c`` (a temperature for the c I approximation's measured 1.3-1.5x overconfidence;
    1.0: none). ``calibrate`` (opt-in) derives the scale instead: the quasi-likelihood temperature
    tau = Var(Delta) / (2 E[Delta]) of the pose score difference Delta between a map at the particle's frame and the
    map rotated (10 deg about x, 30 deg about y, 180 deg about (1,1,1)), for data simulated from the map through the
    forward operator (CTF, tilt projections, backprojection, whitening, mask) and the noise model by Monte Carlo;
    tau is 1 for a calibrated likelihood and T when the scores are T times sharper. The map is ``calibration_map``
    (a real map of the run's model, e.g. the start-up mean). The run's c is multiplied by mean noise ratio / mean
    signal ratio over particles (Var(Delta) and 2 E[Delta] over the c I model's value; times ``noise_scale``).
    Without ``calibration_map`` the templates are white fields on a sphere of the mask radius minus 4 px."""
    image_shape = tuple(int(n) for n in particles.image_shape)
    volume_shape = tuple(int(n) for n in particles.volume_shape)
    n = volume_shape[0]
    if mask_radius_px <= 0 or mask_radius_px > n // 2 - 1:
        raise ValueError(f"The masked likelihood's mask radius must be in (0, {n // 2 - 1}] px, got {mask_radius_px}")
    ball, r2 = half_ball(volume_shape, n // 2 - 1)
    weights = jnp.asarray(half_ball_weights(ball, volume_shape))
    mask = jnp.asarray(sphere(n, mask_radius_px))
    if calibration_map is None:
        base = np.random.default_rng(seed).standard_normal((n,) * 3).astype(np.float32) * sphere(n, max(mask_radius_px - 4, 1))
    else:
        base = np.asarray(calibration_map, np.float32)
    fields = jnp.asarray(np.stack([base] + [_rotated(base, angle, axis) for angle, axis in CALIBRATION_ROTATIONS]))
    padded = np.zeros((2 * n,) * 3, np.float32)
    padded[n // 2 : n // 2 + n, n // 2 : n // 2 + n, n // 2 : n // 2 + n] = base
    padded = padded * _gridding_correction(2 * n)
    source = ftu.full_volume_to_half_volume(ftu.get_dft3(jnp.asarray(padded, jnp.complex64)), (2 * n,) * 3).reshape(-1)
    parseval = parseval_factor(n)
    noise = np.asarray(noise, np.float32).reshape(-1, np.asarray(noise).shape[-1])
    nv_rows = [
        jnp.asarray(noise_utils.to_batched_half_pixel_noise(make_radial_noise(row, image_shape), image_shape))
        .reshape(-1)
        .astype(jnp.float32)
        for row in noise
    ]
    groups = particles.particle_noise_group
    K = max(len(frames) for frames in particles.group_frames)
    count = particles.n_images
    data = np.zeros((count, ball.size), np.complex64)
    filt2 = np.zeros((count, ball.size), np.float32)
    y_norm = np.zeros(count, np.float32)
    cs, lams, visible = np.zeros(count), np.zeros(count), np.zeros(count, np.int64)
    noise_ratios, signal_ratios = np.zeros(count), np.zeros(count)
    for p in range(count):
        images, _ = particles.particle_images([p])
        _, half, ctf = particles.read(images)
        frames = particles.image_projections(images).astype(np.float32)
        k = images.size
        pad = K - k
        half = jnp.pad(jnp.asarray(half), ((0, pad), (0, 0)))
        ctf = jnp.pad(jnp.asarray(ctf, jnp.float32), ((0, pad), (0, 0)))
        frames = np.concatenate([frames, np.tile(np.eye(3, dtype=np.float32), (pad, 1, 1))])
        out = _particle_operands(
            half, ctf, jnp.asarray(frames), nv_rows[int(groups[p]) if len(nv_rows) > 1 else 0], mask,
            jnp.asarray(ball), weights, jax.random.PRNGKey(seed * 100003 + p), jnp.float32(lam_rel),
            jnp.float32(parseval), jnp.float32(noise_scale), fields, source, image_shape=image_shape, volume_shape=volume_shape, n_mc=int(n_mc),
        )  # fmt: skip
        data[p], filt2[p], y_norm[p], cs[p], lams[p], noise_ratios[p], signal_ratios[p] = (np.asarray(x) for x in out)
        visible[p] = k
    bad = np.flatnonzero(
        ~(np.isfinite(cs) & (cs > 0) & np.isfinite(y_norm) & np.isfinite(lams) & np.all(np.isfinite(data), axis=1))
    )
    if bad.size:
        raise ValueError(
            f"The masked likelihood's preprocessing is nonfinite for {bad.size} particle(s) (training indices "
            f"{bad[:20].tolist()}): noise variance c {cs[bad[:5]].tolist()}, ||MV||^2/c {y_norm[bad[:5]].tolist()}, "
            f"lam {lams[bad[:5]].tolist()}. Check those particles' tilt images and CTFs."
        )
    tau = float(np.mean(noise_ratios) / np.mean(signal_ratios))
    calibration = tau if calibrate else 1.0
    if calibrate:  # c times tau: the operands carrying 1 / c scale down by it
        data, filt2, y_norm, cs = data / calibration, filt2 / calibration, y_norm / calibration, cs * calibration
    record = {
        "likelihood": "masked volume, B-diag (exact masked inner, unmasked Fourier Gram)",
        "mask_radius_px": float(mask_radius_px),
        "lam_rel": float(lam_rel),
        "noise_scale": float(noise_scale),
        "calibrate": bool(calibrate),
        "calibration_factor": calibration,
        "tau": tau,
        "noise_ratio_quantiles": np.quantile(noise_ratios, [0.05, 0.5, 0.95]).tolist(),
        "signal_ratio_quantiles": np.quantile(signal_ratios, [0.05, 0.5, 0.95]).tolist(),
        "n_mc": int(n_mc),
        "seed": int(seed),
        "ball_radius": n // 2 - 1,
        "c_quantiles": np.quantile(cs, [0, 0.5, 1]).tolist(),
        "lam_quantiles": np.quantile(lams, [0, 0.5, 1]).tolist(),
        "y_norm_quantiles": np.quantile(y_norm, [0, 0.5, 1]).tolist(),
        "visible_tilts_quantiles": np.quantile(visible, [0, 0.5, 1]).tolist(),
    }
    return VolumeObservations(ball, r2, data, filt2, y_norm, record, (cs * parseval).astype(np.float32))


@dataclasses.dataclass(frozen=True)
class VolumeWindow:
    """A stage's window: the ball positions ``positions`` inside the stage radius, their half-volume voxels,
    Hermitian weights and translation lattice (cycles per pixel), and the observations they index."""

    positions: np.ndarray
    weights: jax.Array
    lattice: jax.Array
    observations: VolumeObservations


def prepare_volume_stream(
    particles,
    mu,
    W=None,
    *,
    rotations,
    translations,
    rotation_log_prior,
    translation_log_prior,
    rotation_parent,
    translation_parent,
    n_coarse_rotations: int,
    n_coarse_translations: int,
    geometry,
    schedule,
    metric_trace_only: bool = False,
    gemm_precision: str = "auto",
    device=None,
    pass2_mass_floor: float = 0.0,
    exact_masked_mstep: bool = False,
    **_image_only,
) -> FullRowStream:
    """The :class:`FullRowStream` of the masked-volume likelihood: :func:`relax.ppca_refinement.full_row_stream.
    prepare_full_row_stream`'s grids and priors, with the stage window the half ball of radius
    ``geometry.current_size // 2`` and tiles read by :func:`load_volume_tile` from ``particles.volume_observations``.
    The image-only options of the image route's stream (noise, scoring, exact metric trace, occupancy) are not
    used: the caller passes them for both routes and :func:`refuse_image_only_options` checks them."""
    refuse_image_only_options(_image_only)
    observations = particles.volume_observations
    if observations is None:
        raise ValueError("The masked-volume stream needs the particles' volume observations")
    device = jax.local_devices()[0] if device is None else device
    volume_shape = tuple(int(n) for n in particles.volume_shape)
    radius = int(geometry.current_size) // 2
    positions = np.flatnonzero(observations.radius2 <= radius**2).astype(np.int32)
    indices = observations.indices[positions]
    k = np.stack(np.unravel_index(indices, (volume_shape[0], volume_shape[1], volume_shape[2] // 2 + 1)), -1)
    k = (k - np.asarray([volume_shape[0] // 2, volume_shape[1] // 2, 0])).astype(np.float32)
    rotations = np.asarray(rotations, np.float32)
    translations = np.asarray(translations, np.float32)
    n_rot = int(rotations.shape[0])
    with jax.default_device(device):
        augmented, q = coerce_augmented_half_volumes(
            mu, W, volume_shape=volume_shape, q=geometry.q, volume_domain=geometry.volume_domain
        )
        window = VolumeWindow(
            positions=positions,
            weights=jnp.asarray(half_ball_weights(indices, volume_shape)),
            lattice=jnp.asarray(k / np.float32(volume_shape[0])),
            observations=observations,
        )
        n_half = int(particles.image_shape[0]) * (int(particles.image_shape[1]) // 2 + 1)
        block_size = int(schedule.rotation_block_size)
        arrays = StreamArrays(
            augmented=augmented,
            augmented_voxel_major=None,
            rotations=jnp.asarray(np.concatenate([rotations, np.eye(3, dtype=np.float32)[None]])),
            rotation_log_prior=jnp.asarray(np.append(np.asarray(rotation_log_prior, np.float32), np.float32(0))),
            translation_log_prior=jnp.asarray(np.asarray(translation_log_prior, np.float32)),
            rotation_parent=jnp.asarray(np.append(np.asarray(rotation_parent, np.int32), np.int32(n_coarse_rotations))),
            translation_parent=jnp.asarray(np.asarray(translation_parent, np.int32)),
            score_indices=jnp.asarray(indices),
            gemm_window=None,
            # A volume stream keeps no noise sums (its noise is fixed by the preprocessing): a zero half image.
            coefficient_noise=jnp.zeros((n_half,), jnp.float32),
            shift_squared=jnp.sum(jnp.asarray(translations) ** 2, axis=-1),
            volume_mask_ft=(
                ftu.get_dft3(jnp.asarray(sphere(volume_shape[0], observations.record["mask_radius_px"]), jnp.complex64))
                .reshape(-1)
                if exact_masked_mstep
                else None
            ),
            volume_translations=jnp.asarray(translations) if exact_masked_mstep else None,
        )
    static = StreamStatic(
        image_shape=tuple(int(n) for n in particles.image_shape),
        volume_shape=volume_shape,
        disc_type="linear_interp",
        projection_max_r=None,
        backprojection_max_r=None,
        relion_texture_interp=False,
        basis_size=int(q) + 1,
        metric_trace_only=bool(metric_trace_only),
        exact_metric_trace=False,
        cuda_kernels=False,
        gemm_precision=resolve_gemm_precision(gemm_precision, device),
        observation="volume",
        exact_masked_mstep=bool(exact_masked_mstep),
    )
    return FullRowStream(
        dataset=particles,
        arrays=arrays,
        static=static,
        forward_config=None,
        resolved=window,
        noise_variance_half=jnp.zeros((n_half,), jnp.float32),
        translations=translations,
        n_coarse_rotations=int(n_coarse_rotations),
        n_coarse_translations=int(n_coarse_translations),
        image_batch_size=int(schedule.image_batch_size),
        block_starts=device_block_starts(n_rot, block_size, device),
        rotation_block_size=block_size,
        rotation_parent=np.asarray(rotation_parent, np.int32),
        device=device,
        tile_loader=load_volume_tile,
        pass2_mass_floor=float(pass2_mass_floor),
        occupancy=check_occupancy(None),
    )


def refuse_image_only_options(options: dict) -> None:
    """The image stream's options a volume stream accepts only at their defaults (the controller passes them)."""
    if options.get("exact_metric_trace"):
        raise ValueError("The exact metric-diagonal trace needs the image route's CUDA moment scatter")
    if options.get("occupancy") is not None:
        raise ValueError("The masked-volume likelihood has no empty component")
    loader = options.get("tile_loader")
    if loader is not None and loader is not load_volume_tile:
        from relax.ppca_initial_model.tomo import load_tilt_tile

        if loader is not load_tilt_tile:
            raise ValueError("The masked-volume stream reads its own tiles")


@functools.partial(jax.jit, static_argnames=("pad",))
def _volume_operands(data, filt2, y_norm, weights, lattice, translations, *, pad):
    """``Y1 (2F, B T)``, ``ctf2 (F, B)``, ``Y1_recon (B T, 2F)``, ``ctf2_recon (B, F)`` and ``y_norm (B,)`` of
    particles ``data``/``filt2`` ``(b, F)``, padded with ``pad`` zero particles (``tomo._tilt_operands_body`` with
    one frame and the 3D phases ``exp(-2 pi i k . t)``)."""
    data = jnp.pad(data, ((0, pad), (0, 0)))
    filt2 = jnp.pad(filt2, ((0, pad), (0, 0)))
    y_norm = jnp.pad(y_norm, (0, pad))
    B, F = data.shape
    T = translations.shape[0]
    phases = translation_phases(translations, lattice)  # (T, F)
    shifted = data[:, None, :] * phases[None]  # (B, T, F)
    Y1 = jnp.transpose(real_imag(shifted * weights), (2, 0, 1)).reshape(2 * F, B * T)
    Y1_recon = real_imag(shifted).reshape(B * T, 2 * F)
    return Y1, (filt2 * weights).T, Y1_recon, filt2, y_norm


def load_volume_tile(stream, image_indices, significant_rows, *, collect_observation: bool):
    """:class:`FullRowStream` tile reader of the masked-volume likelihood (one 3D frame per particle); padded to
    the stream's planned bucket with zero particles after the real ones, as the subtomogram reader pads."""
    window: VolumeWindow = stream.resolved
    observations = window.observations
    ids = np.asarray(image_indices, np.int64)
    B = ids.size
    padded = B if stream.tile_images is None else tile_size_bucket(B, stream.tile_images)
    positions = window.positions
    data = observations.data[ids][:, positions]
    filt2 = observations.filt2[ids][:, positions]
    y_norm = observations.y_norm[ids]
    with jax.default_device(stream.device):
        data, filt2, y_norm = jax.device_put((data, filt2, y_norm), stream.device)
        Y1, ctf2, Y1_recon, ctf2_recon, y_norm = _volume_operands(
            data, filt2, y_norm, window.weights, window.lattice, jnp.asarray(stream.translations), pad=padded - B
        )
        coarse_mask, rows, layout = tile_support(stream, significant_rows)
        tile = TileArrays(
            coarse_mask=jnp.concatenate([coarse_mask, jnp.repeat(coarse_mask[:1], padded - B, axis=0)]),
            rows=rows,
            Y1=Y1,
            ctf2=ctf2,
            Y1_recon=Y1_recon,
            ctf2_recon=ctf2_recon,
            y_norm=y_norm,
            frames=None,
            noise_groups=jnp.zeros((1,), jnp.int32),
            volume_scale=(
                jnp.pad(jnp.asarray(observations.scale[ids]), (0, padded - B), constant_values=1.0)
                if stream.static.exact_masked_mstep
                else None
            ),
        )
    layout.update(
        n_real=B,
        n_observations=B,
        original_ids=stream.dataset.original_image_indices_from_local(ids),
        frames=None,
    )
    observation = jnp.zeros(stream.arrays.coefficient_noise.shape, jnp.float32) if collect_observation else None
    return tile, observation, layout


def volume_operand_bytes(stream, n_particles: int) -> tuple[int, int]:
    """``(peak, resident)`` device bytes of :func:`load_volume_tile` for ``n_particles``: the shifted ``(B, T, F)``
    complex stack and its weighted copy, the two planar operands and the metric rows."""
    T = int(stream.translations.shape[0])
    F = int(stream.resolved.positions.size)
    planar = 2 * 2 * T * F  # Y1 and Y1_recon
    resident = planar + 2 * F
    peak = resident + 2 * 2 * T * F + 4 * F
    return 4 * int(n_particles) * peak, 4 * int(n_particles) * resident


load_volume_tile.operand_bytes = volume_operand_bytes
load_volume_tile.max_frames = lambda stream: 1
