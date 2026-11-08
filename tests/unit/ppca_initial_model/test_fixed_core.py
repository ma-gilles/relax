"""The fixed-core residual of subtomogram PPCA (relax/ppca_initial_model/fixed_core.py, tomo.subtract_fixed_core).

Tilt images are made in float64 by an independent forward model, ``CTF e^{+2 pi i s.k} P(Aproj) V`` with
``s = [Aproj t]_{1,2}`` written out here, from RECOVAR's slicing primitive. The checks: the core prediction is
the stream's own projection (its projector, frame matrices and shift phases); images of ``core + X`` plus noise
leave ``X``'s images plus noise, through the tile reader with a hidden tilt and tile padding; on the residual the
empty hypothesis scores as the core-only model does on the observed images; the per-particle core scale.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
from helpers.float_compare import assert_matches
from helpers.full_row_tile import accumulate_full_row_tile
from recovar import core
from scipy.spatial.transform import Rotation

from relax.fourier.half_spectrum import make_half_image_weights
from relax.fourier.preprocessing import relion_half_translation_lattice
from relax.ppca_initial_model.fixed_core import (
    FixedCore,
    core_basis_volumes,
    core_prediction,
    extend_axial_core,
    image_core_shifts,
    symmetrize_core,
)
from relax.ppca_initial_model.tomo import (
    TiltParticles,
    estimate_core_coefficients,
    estimate_core_scales,
    estimate_residual_background,
    load_tilt_tile,
    subtract_fixed_core,
    tilt_shift_phases,
)
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_kernels import frame_rotations, project
from relax.ppca_refinement.full_row_stream import prepare_full_row_stream

pytestmark = pytest.mark.unit

N = 16
IMAGE_SHAPE = (N, N)
VOLUME_SHAPE = (N, N, N)
N_HALF = N * (N // 2 + 1)
NOISE = 4.0
TILTS = Rotation.from_euler("y", [-45, -20, 0, 25, 50], degrees=True).as_matrix()
# Two subtomogram matrices: group 0 (particles 0, 2 and 3) and group 1 (particle 1, four frames).
SUBTOMOGRAMS = Rotation.from_euler("zyz", [[30, 70, -10], [-100, 40, 25]], degrees=True).as_matrix()
GROUP_FRAMES = (TILTS @ SUBTOMOGRAMS[0], TILTS[:4] @ SUBTOMOGRAMS[1])
PARTICLE_GROUP = np.array([0, 1, 0, 0])
# Particle 2 does not see frame 2 and lists its images in another frame order.
IMAGE_FRAMES = [[0, 1, 2, 3, 4], [0, 1, 2, 3], [4, 0, 3, 1], [0, 1, 2, 3, 4]]
OFFSETS = np.array([[0.7, -1.3, 0.4], [-0.5, 0.9, 1.6], [1.2, 0.3, -0.8], [-1.1, -0.6, 0.2]])
# The float32 core prediction against the float64 forward model, relative to the core images' scale: measured
# 6.4e-7 to 1.39e-6 over problem seeds 3-7 (the removed core 6.5e-7 to 1.38e-6, the tiles' Y1 differences 7.5e-7
# to 1.46e-6); 3e-6 leaves a 2x margin. A residual inherits this error in the core's units, not its own.
PROJECTOR_RTOL = 3e-6


def _half_volume(rng, scale):
    real = rng.standard_normal(VOLUME_SHAPE)
    full = np.fft.fftshift(np.fft.fftn(real))
    return (scale * np.asarray(ftu.full_volume_to_half_volume(full, VOLUME_SHAPE)).reshape(-1)).astype(np.complex64)


def _layout():
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in IMAGE_FRAMES])]).astype(np.int64)
    image_frame = np.concatenate(IMAGE_FRAMES).astype(np.int64)
    owner = np.repeat(np.arange(len(IMAGE_FRAMES)), np.diff(offsets))
    projections = np.stack([GROUP_FRAMES[PARTICLE_GROUP[p]][f] for p, f in zip(owner, image_frame)])
    return offsets, image_frame, owner, projections


def reference_images(volume_half, projections, ctf, offsets):
    """Float64 ``CTF e^{+2 pi i s.k} P(Aproj) V`` with ``s = (Aproj t)[:2]``: the image the stream predicts
    for the model ``V`` at the 3D translation ``t`` (it compares ``y e^{-2 pi i s.k}`` with ``CTF P(Aproj) V``)."""
    with jax.enable_x64(True):
        sliced = core.batch_slice_volume(
            jnp.asarray(volume_half, jnp.complex128)[None],
            jnp.asarray(projections, jnp.float64),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            half_volume=True,
            half_image=True,
            relion_texture_interp=False,
            max_r=N // 2 - 1,
        )
        sliced = np.asarray(sliced)[0]
    shifts = np.einsum("nab,nb->na", projections, offsets)[:, :2]
    lattice = np.asarray(relion_half_translation_lattice(IMAGE_SHAPE), np.float64)
    return ctf * sliced * np.exp(2j * np.pi * shifts @ lattice.T)


def _particles(half, ctf):
    offsets, image_frame, _, _ = _layout()

    def read(ids):
        ids = np.asarray(ids)
        return None, jnp.asarray(half[ids], jnp.complex64), jnp.asarray(ctf[ids], jnp.float32)

    return TiltParticles(
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        voxel_size=2.0,
        image_offsets=offsets,
        image_frame=image_frame,
        particle_group=PARTICLE_GROUP,
        group_frames=tuple(np.asarray(f, np.float64) for f in GROUP_FRAMES),
        read=read,
    )


def make_problem(seed=3, core_scale=3.0, scales=None, noise=True):
    """Observed tilt images ``c_i P(core) + P(X) + noise`` at each particle's offset, and their parts (float64)."""
    rng = np.random.default_rng(seed)
    _, _, owner, projections = _layout()
    n_images = owner.size
    ctf = rng.uniform(0.3, 1.2, (n_images, N_HALF))
    volumes = {"core": _half_volume(rng, core_scale), "X": _half_volume(rng, 1.0)}
    parts = {k: reference_images(v, projections, ctf, OFFSETS[owner]) for k, v in volumes.items()}
    epsilon = np.sqrt(NOISE / 2) * (
        rng.standard_normal((n_images, N_HALF)) + 1j * rng.standard_normal((n_images, N_HALF))
    )
    epsilon = epsilon if noise else np.zeros_like(epsilon)
    scales = np.ones(len(IMAGE_FRAMES)) if scales is None else np.asarray(scales)
    observed = scales[owner, None] * parts["core"] + parts["X"] + epsilon
    fixed = FixedCore.from_volume(
        volumes["core"], volume_shape=VOLUME_SHAPE, voxel_size=2.0, volume_domain="fourier_half"
    )
    return dict(ctf=ctf, volumes=volumes, parts=parts, noise=epsilon, observed=observed, fixed=fixed)


def _stream(particles, mu, *, rotations, translations, current_size=None, occupancy=None, tile_images=None):
    W = np.zeros((mu.size, 1), np.complex64)
    R, T = len(rotations), len(translations)
    stream = prepare_full_row_stream(
        particles,
        np.asarray(mu, np.complex64),
        W,
        noise_variance=np.full(IMAGE_SHAPE, NOISE, np.float32),
        rotations=np.asarray(rotations, np.float32),
        translations=np.asarray(translations, np.float32),
        rotation_log_prior=np.full(R, -np.log(R), np.float32),
        translation_log_prior=np.full(T, -np.log(T), np.float32),
        rotation_parent=np.zeros(R, np.int32),
        translation_parent=np.zeros(T, np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=current_size, q=1, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=particles.n_images, rotation_block_size=2),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        tile_loader=load_tilt_tile,
        gemm_precision="fp32",
        occupancy=occupancy,
    )
    return stream._replace(tile_images=tile_images)


@pytest.mark.parametrize("current_size", [None, 10])
def test_core_prediction_is_the_streams_projection(current_size):
    """The core prediction of every visible tilt equals the stream's own ``CTF S P(Aproj R) mu`` with ``mu`` the
    core, at ``R = I`` and the particle's offset: its projector (``project`` of ``frame_rotations``), its float32
    frames, and the conjugate of its shift phases (``tilt_shift_phases``); on the score window when there is one."""
    p = make_problem()
    particles = _particles(p["observed"], p["ctf"])
    for particle in range(particles.n_images):
        stream = _stream(
            particles,
            p["volumes"]["core"],
            rotations=np.eye(3)[None],
            translations=OFFSETS[particle : particle + 1],
            current_size=current_size,
        )
        assert (stream.resolved.score_indices is None) == (current_size is None)
        tile, _, layout = load_tilt_tile(stream, [particle], [None], collect_observation=False)
        projected = project(stream.arrays, frame_rotations(tile, stream.arrays.rotations[:1]), stream.static)[0]
        phases = tilt_shift_phases(stream, layout, stream.translations)[0]  # (K, F)
        images, _ = particles.particle_images([particle])
        frames = particles.image_frame[images]
        window = np.arange(N_HALF) if current_size is None else np.asarray(stream.resolved.score_indices)
        ctf = jnp.asarray(p["ctf"][images], jnp.float32)
        stream_prediction = ctf[:, window] * projected[frames] * jnp.conj(phases[frames])
        projections = particles.image_projections(images)
        ours = core_prediction(
            p["fixed"], IMAGE_SHAPE, projections, ctf, image_core_shifts(projections, OFFSETS[[particle] * images.size])
        )
        assert ours.dtype == jnp.complex64 and ours.shape == (images.size, N_HALF)
        assert_matches(np.asarray(ours)[:, window], np.asarray(stream_prediction))
        # And both are the float64 forward model of the core (box N - 1 clip, + phase for + shifts).
        assert_matches(np.asarray(ours), p["parts"]["core"][images], rtol=PROJECTOR_RTOL)


def test_residual_is_the_attachment_images_plus_noise():
    """``read`` of the subtracted particles returns ``P(X) + noise`` image by image, in any image order: what it
    removed from the observed ``P(core) + P(X) + noise`` is the core's images, to the projector's float32 band."""
    p = make_problem()
    subtracted = subtract_fixed_core(_particles(p["observed"], p["ctf"]), p["fixed"], offsets=OFFSETS)
    order = np.random.default_rng(0).permutation(p["observed"].shape[0])
    raw, half, ctf = subtracted.read(order)
    assert raw is None and half.dtype == jnp.complex64 and half.shape == (order.size, N_HALF)
    half = np.asarray(half)
    assert_matches(p["observed"][order] - half, p["parts"]["core"][order], rtol=PROJECTOR_RTOL)
    assert_matches(np.asarray(ctf), p["ctf"][order])
    # The same error in the core's units: |residual - (P(X) + noise)| against max |P(core)|.
    error = np.max(np.abs(half - (p["parts"]["X"] + p["noise"])[order]))
    assert error <= PROJECTOR_RTOL * np.max(np.abs(p["parts"]["core"]))
    # The core is about three times X: the residual is far from the observed images.
    assert np.max(np.abs(p["parts"]["core"])) > 1e4 * error


@pytest.mark.parametrize("tile_images", [None, 16])
def test_tilt_tile_of_the_residual(tile_images):
    """The tile reader's operands, ``y_norm`` and observation power on the subtracted particles equal those of
    particles whose images are ``P(X) + noise``: with a hidden tilt (particle 2) and, with a planned tile size,
    a padded tile."""
    p = make_problem()
    observed = _particles(p["observed"], p["ctf"])
    arms = {
        "subtracted": subtract_fixed_core(observed, p["fixed"], offsets=OFFSETS),
        "observed": observed,
        "direct": _particles(p["parts"]["X"] + p["noise"], p["ctf"]),
        "core": _particles(p["parts"]["core"], p["ctf"]),
    }
    grid = dict(rotations=Rotation.random(3, random_state=4).as_matrix(), translations=OFFSETS, current_size=12)
    ids = np.array([0, 2, 3])  # group 0: five frames, particle 2 hides frame 2
    tiles = {}
    for name, particles in arms.items():
        stream = _stream(particles, p["volumes"]["X"], tile_images=tile_images, **grid)
        tiles[name] = load_tilt_tile(stream, ids, [None] * 3, collect_observation=True)
    tile, observation, layout = tiles["subtracted"]
    if tile_images is not None:
        assert tile.y_norm.shape[0] > layout["n_real"] == 3
    assert layout["n_observations"] == 14
    hidden = np.asarray(tile.ctf2).reshape(5, -1, tile.y_norm.shape[0])[2, :, 1]  # particle 2, frame 2
    assert np.all(hidden == 0)  # masked, not computed
    # The image operands are linear in the images: observed minus residual is the core's, in its units.
    for name in ("Y1", "Y1_recon"):
        removed = np.asarray(getattr(tiles["observed"][0], name)) - np.asarray(getattr(tile, name))
        assert_matches(removed, np.asarray(getattr(tiles["core"][0], name)), rtol=PROJECTOR_RTOL)
    for name in ("ctf2", "ctf2_recon"):
        assert_matches(np.asarray(getattr(tile, name)), np.asarray(getattr(tiles["direct"][0], name)))
    # Image energies of the residual (measured 2.0e-7 to 3.9e-7 over seeds 3-7: the default band).
    assert_matches(np.asarray(tile.y_norm), np.asarray(tiles["direct"][0].y_norm))
    # The observed power per pixel squares the residual, whose error is in the core's units: measured 4.1e-7 to
    # 3.2e-6 of its largest pixel over seeds 3-7; 1e-5 leaves a 3x margin.
    assert_matches(np.asarray(observation), np.asarray(tiles["direct"][1]), rtol=1e-5)


def test_empty_hypothesis_on_the_residual_is_the_core_only_model():
    """On the residual with ``mu = 0, W = 0`` every pose scores 0, as the empty component does: P(occupied) is the
    prior, and the log-likelihood of each particle is that of the core-only model (``mu`` the core at the
    particle's pose) on the observed images."""
    p = make_problem()
    observed = _particles(p["observed"], p["ctf"])
    subtracted = subtract_fixed_core(observed, p["fixed"], offsets=OFFSETS)
    zero = np.zeros_like(p["volumes"]["core"])
    for particle in range(observed.n_images):
        ids = [particle]
        core_only = _stream(observed, p["volumes"]["core"], rotations=np.eye(3)[None], translations=OFFSETS[ids])
        core_only = accumulate_full_row_tile(core_only, ids, [None])
        empty = _stream(
            subtracted,
            zero,
            rotations=Rotation.random(3, random_state=4).as_matrix(),
            translations=OFFSETS,
            occupancy=0.3,
        )
        empty = accumulate_full_row_tile(empty, ids, [None])
        assert_matches(empty.diagnostics["p_occupied_per_image"], np.array([0.3], np.float32))
        # The core-only arm cancels the observed images' energy against its score: measured 1.2e-6 to 3.8e-6 of
        # the log-likelihood over seeds 3-7; 1e-5 leaves a 2.6x margin.
        assert_matches(np.array(empty.log_likelihood), np.array(core_only.log_likelihood), rtol=1e-5)
        # The float64 reference: -1/2 sum_tilts sum_pixels w |y - P(core)|^2 / sigma^2.
        images, _ = observed.particle_images(ids)
        weights = np.asarray(make_half_image_weights(IMAGE_SHAPE), np.float64)
        residual = p["observed"][images] - p["parts"]["core"][images]
        reference = -0.5 * np.sum(np.abs(residual) ** 2 * weights / NOISE)
        # The float32 default band (measured 2.2e-7 to 4.3e-7 over seeds 3-7).
        assert_matches(np.array(empty.log_likelihood), np.array(reference), rtol=1e-6)


SCALES = (0.6, 1.0, 1.45, 0.8)


def _shell_noise():
    n_shells = N // 2 + 1  # make_radial_noise's radial profile: one value per integer radius up to N / 2
    return np.full((1, n_shells), NOISE, np.float32)


def test_core_scale_is_off_by_default_and_recovered_on_noiseless_data():
    p = make_problem(scales=SCALES, noise=False)
    observed = _particles(p["observed"], p["ctf"])
    no_x = _particles(p["observed"] - p["parts"]["X"], p["ctf"])
    scales = estimate_core_scales(no_x, p["fixed"], _shell_noise(), offsets=OFFSETS)
    assert_matches(scales, np.asarray(SCALES), rtol=1e-6)  # float32 default band; measured 4.2e-8 to 8.9e-8
    # Unscaled (the default) the residual keeps (c_i - 1) P(core): only P(core) is removed; with the fitted
    # scales c_i P(core) is.
    _, _, owner, _ = _layout()
    every = np.arange(owner.size)
    default = subtract_fixed_core(observed, p["fixed"], offsets=OFFSETS)
    removed = p["observed"] - np.asarray(default.read(every)[1])
    assert_matches(removed, p["parts"]["core"], rtol=PROJECTOR_RTOL)
    scaled = subtract_fixed_core(observed, p["fixed"], offsets=OFFSETS, scales=scales)
    removed = p["observed"] - np.asarray(scaled.read(every)[1])
    assert_matches(removed, np.asarray(SCALES)[owner, None] * p["parts"]["core"], rtol=PROJECTOR_RTOL)
    with pytest.raises(ValueError, match="already"):
        subtract_fixed_core(scaled, p["fixed"])
    with pytest.raises(ValueError, match="observed images"):
        estimate_core_scales(scaled, p["fixed"], _shell_noise())


def test_core_scale_prior_shrinks_toward_one():
    """On noisy data the scale is the float64 weighted least-squares fit of each particle's stacked tilts, and
    under a Gaussian prior about 1 the augmented least-squares (prior) fit, which lies between 1 and it."""
    p = make_problem(core_scale=0.3, scales=SCALES)
    observed = _particles(p["observed"], p["ctf"])
    noise = _shell_noise()
    fitted = estimate_core_scales(observed, p["fixed"], noise, offsets=OFFSETS)
    prior_sd = 0.1
    shrunk = estimate_core_scales(observed, p["fixed"], noise, offsets=OFFSETS, prior_sd=prior_sd)
    weights = np.sqrt(np.asarray(make_half_image_weights(IMAGE_SHAPE), np.float64) / NOISE)
    for i in range(observed.n_images):
        images, _ = observed.particle_images([i])
        design = (p["parts"]["core"][images] * weights).reshape(-1)
        target = (p["observed"][images] * weights).reshape(-1)
        A = np.concatenate([design.real, design.imag])[:, None]
        y = np.concatenate([target.real, target.imag])
        plain = np.linalg.lstsq(A, y, rcond=None)[0][0]
        prior = np.linalg.lstsq(np.vstack([A, [[1 / prior_sd]]]), np.append(y, 1 / prior_sd), rcond=None)[0][0]
        # The float32 default band (measured 5.7e-8 to 3.8e-7 over seeds 3-7).
        assert_matches(np.array(fitted[i]), np.array(plain), rtol=1e-6)
        assert_matches(np.array(shrunk[i]), np.array(prior), rtol=1e-6)
        assert abs(shrunk[i] - 1) < abs(fitted[i] - 1) and (shrunk[i] - 1) * (fitted[i] - 1) > 0


def test_core_map_reads_back_as_the_volume_it_holds(tmp_path):
    """A RELION-convention core map (one voxel size per header axis) is the core its array holds."""
    import mrcfile
    from recovar.utils.helpers import load_relion_volume

    n, voxel = 16, 8.0
    relion = np.random.default_rng(4).normal(size=(n, n, n)).astype(np.float32)
    path = tmp_path / "core.mrc"
    with mrcfile.new(path) as m:
        m.set_data(relion)
        m.voxel_size = voxel
    core = FixedCore.from_map(path)
    reference = FixedCore.from_volume(
        np.asarray(load_relion_volume(str(path)), np.float32),
        volume_shape=(n, n, n),
        voxel_size=voxel,
        volume_domain="real",
    )
    assert core.volume_shape == (n, n, n) and core.voxel_size == voxel
    assert_matches(np.asarray(core.volume_half), np.asarray(reference.volume_half))
    with mrcfile.new(tmp_path / "aniso.mrc") as m:
        m.set_data(relion)
        m.voxel_size = (voxel, voxel, 2 * voxel)
    with pytest.raises(ValueError, match="anisotropic"):
        FixedCore.from_map(tmp_path / "aniso.mrc")


def _tube(radius, shift_x=0.0, n=N, width=1.2):
    """A smooth hollow tube along axis 2 (the internal frame's z) about (shift_x, 0), real space, float64."""
    x = (np.arange(n) - n // 2)[:, None, None] - shift_x
    y = (np.arange(n) - n // 2)[None, :, None]
    rho = np.sqrt(x**2 + y**2) + 0.0 * np.arange(n)[None, None, :]
    return np.exp(-0.5 * ((rho - radius) / width) ** 2)


def _real_to_half(volume):
    full = np.fft.fftshift(np.fft.fftn(volume)).astype(np.complex64)
    return np.asarray(ftu.full_volume_to_half_volume(full, VOLUME_SHAPE)).reshape(-1).astype(np.complex64)


def test_core_basis_coefficients_are_recovered_and_absorb_shape_error():
    _, _, owner, projections = _layout()
    rng = np.random.default_rng(6)
    ctf = rng.uniform(0.3, 1.2, (owner.size, N_HALF))
    basis_real = core_basis_volumes(_tube(4.0))
    halves = [_real_to_half(v) for v in basis_real]  # one Fourier layout for the cores and the reference images
    basis = [
        FixedCore.from_volume(h, volume_shape=VOLUME_SHAPE, voxel_size=2.0, volume_domain="fourier_half")
        for h in halves
    ]
    parts = [reference_images(h, projections, ctf, OFFSETS[owner]) for h in halves]
    # Noiseless images that are exactly the basis at known per-particle coefficients give those coefficients back.
    truth = np.array([[0.7, 0.05, 0.1, -0.2], [1.1, -0.03, 0.0, 0.15], [0.9, 0.02, -0.12, 0.0], [1.3, 0.0, 0.05, 0.05]])
    exact = sum(truth[owner, k, None] * parts[k] for k in range(4))
    fitted = estimate_core_coefficients(_particles(exact, ctf), basis, _shell_noise(), offsets=OFFSETS)
    # The ridge (1e-6 of each diagonal entry) biases the nearly collinear derivatives by about that much.
    assert_matches(fitted, truth, rtol=1e-4)
    # A tube 3% wider and 0.3 px off the axis: the scale alone leaves the wall's edges in the residual, the basis
    # (dilation and shifts) removes most of it.
    observed = reference_images(_real_to_half(_tube(4.12, shift_x=0.3)), projections, ctf, OFFSETS[owner])
    particles = _particles(observed, ctf)
    every = np.arange(owner.size)
    scales = estimate_core_scales(particles, basis[0], _shell_noise(), offsets=OFFSETS)
    scale_only = np.asarray(subtract_fixed_core(particles, basis[0], offsets=OFFSETS, scales=scales).read(every)[1])
    coefficients = estimate_core_coefficients(particles, basis, _shell_noise(), offsets=OFFSETS)
    with_basis = np.asarray(
        subtract_fixed_core(particles, tuple(basis), offsets=OFFSETS, scales=coefficients).read(every)[1]
    )
    basis_power, scale_power = (float(np.sum(np.abs(r) ** 2)) for r in (with_basis, scale_only))
    assert basis_power < 0.2 * scale_power, (basis_power, scale_power)  # measured ratio 0.023
    # C + c_1 D = C((1 + c_1) r): the 3% wider tube reads as c_1 / c_0 of about -0.03 (measured -0.023 to -0.027).
    assert np.all((coefficients[:, 1] / coefficients[:, 0] < -0.01) & (coefficients[:, 1] / coefficients[:, 0] > -0.05))


def test_extend_axial_core_tiles_a_uniform_core_and_refuses_others():
    tube = _tube(4.0).astype(np.float32)
    long = extend_axial_core(tube, 3 * N)
    m = long.shape[0]
    pad = (m - N) // 2
    assert long.shape == (m, m, m)
    assert_matches(long[pad : pad + N, pad : pad + N, pad : pad + N], tube)  # the original slab, centred
    assert_matches(long[pad : pad + N, pad : pad + N, :], np.tile(tube[:, :, :1], (1, 1, m)))  # uniform along z
    assert np.all(long[:pad] == 0) and np.all(long[:, :pad] == 0)  # zero padding across the axis
    bead = tube * np.exp(-(((np.arange(N) - N // 2) / 3.0) ** 2))[None, None, :]
    with pytest.raises(ValueError, match="not uniform"):
        extend_axial_core(bead, 2 * N)


def test_a_larger_core_box_captures_the_filament_beyond_the_image_box():
    """A tube tilted out of the image plane: the image box's own tube stops at the box faces, the image does not.
    The prediction from an M box converges to the 2M box's in the cropped image; the N-box prediction does not."""
    rng = np.random.default_rng(8)
    # The tube axis (particle z) at 0, 45 and 60 degrees out of the image plane (a rotation about y by 90 - angle).
    tilts = Rotation.from_euler("zy", [[0, 90], [20, 45], [-35, 30]], degrees=True).as_matrix()
    tube = _tube(3.0).astype(np.float32)

    def predict(box):
        volume = extend_axial_core(tube, box) if box > N else tube
        fixed = FixedCore.from_volume(volume, volume_shape=(box,) * 3, voxel_size=2.0, volume_domain="real")
        ctf = np.ones((len(tilts), box * (box // 2 + 1)), np.float32)  # no CTF: the crop alone is tested
        return np.asarray(core_prediction(fixed, IMAGE_SHAPE, tilts, ctf, np.zeros((len(tilts), 2), np.float32)))

    small, large, larger = predict(N), predict(2 * N), predict(4 * N)
    near, far = (np.linalg.norm(a - larger, axis=1) / np.linalg.norm(larger, axis=1) for a in (large, small))
    # Measured: in plane 0.0007 (the image box holds the whole projected tube); at 45 and 60 degrees out of the
    # plane the N box misses 0.39 and 0.52 of the 4N prediction, the 2N box 0.14 and 0.20.
    assert far[0] < 0.05
    assert np.all(far[1:] > 0.2) and np.all(near[1:] < 0.75 * far[1:]), (near, far)


def test_free_offset_zeroes_the_residual_mean_and_marks_the_particles():
    from relax.fourier.half_spectrum import half_spectrum_dc_index

    p = make_problem(seed=5)
    _, _, owner, _ = _layout()
    every = np.arange(owner.size)
    dc = half_spectrum_dc_index(IMAGE_SHAPE)
    shifted = p["observed"].copy()
    shifted[:, dc] += np.random.default_rng(1).normal(0, 50, owner.size)  # a per-image offset: the DC pixel only
    observed = _particles(shifted, p["ctf"])
    plain = subtract_fixed_core(observed, p["fixed"], offsets=OFFSETS)
    free = subtract_fixed_core(observed, p["fixed"], offsets=OFFSETS, free_offset=True)
    assert not plain.dc_free and free.dc_free
    residual, reference = np.asarray(free.read(every)[1]), np.asarray(plain.read(every)[1])
    assert np.all(residual[:, dc] == 0)
    keep = np.arange(N_HALF) != dc
    assert_matches(residual[:, keep], reference[:, keep])


def test_frozen_noise_shells_keep_their_value_and_the_rest_update():
    from relax.ppca_initial_model.noise import update_noise

    previous = np.array([[5.0, 2.0, 3.0], [4.0, 1.0, 6.0]], np.float32)
    numerator = np.array([[0.0, 8.0, 9.0], [0.0, 2.0, 3.0]])  # shell 0 holds no power (DC-free particles)
    denominator = np.array([[4.0, 4.0, 3.0], [2.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="Nonpositive"):
        update_noise(previous, numerator, denominator, full_data=True)
    frozen = np.asarray(update_noise(previous, numerator, denominator, full_data=True, frozen_shells=(0,)))
    unfrozen = np.asarray(update_noise(previous[:, 1:], numerator[:, 1:], denominator[:, 1:], full_data=True))
    assert_matches(frozen[:, 0], previous[:, 0])
    assert_matches(frozen[:, 1:], unfrozen)


def test_symmetrize_core_keeps_a_tube_and_averages_an_offset_blob_down():
    tube = _tube(4.0).astype(np.float32)
    assert_matches(symmetrize_core(tube, "z"), tube)
    # A ring wide against the 1-voxel radial bins survives the azimuthal average up to the linear interpolation
    # over radius (measured 3.7e-2 of its peak for this width; a 1.2-voxel-wide ring loses ~0.1).
    wide = _tube(4.0, width=2.5).astype(np.float32)
    assert_matches(symmetrize_core(wide, "cylinder"), wide, rtol=5e-2)
    blob = np.zeros((N, N, N), np.float32)
    blob[N // 2 + 5, N // 2, N // 2] = 1.0
    assert symmetrize_core(blob, "z").max() == pytest.approx(1.0 / N)
    assert symmetrize_core(blob, "cylinder").max() < 0.2 / N


def test_residual_background_absorbs_a_shared_leftover_the_core_misses():
    """Images = c_i P(core) + P(ring) + noise with a cylindrical ring the core map lacks: after the core, the
    symmetrised residual mean recovers the ring, and fitting [core, background] removes it."""
    _, _, owner, projections = _layout()
    rng = np.random.default_rng(9)
    n_images = owner.size
    ctf = rng.uniform(0.3, 1.2, (n_images, N_HALF)).astype(np.float32)
    core_real, ring_real = _tube(4.0).astype(np.float32), 0.4 * _tube(6.5, width=0.8).astype(np.float32)

    def fixed(v):
        return FixedCore.from_volume(v, volume_shape=VOLUME_SHAPE, voxel_size=2.0, volume_domain="real")

    shifts = np.zeros((n_images, 2), np.float32)
    parts = [
        np.asarray(core_prediction(fixed(v), IMAGE_SHAPE, projections, ctf, shifts)) for v in (core_real, ring_real)
    ]
    scales = np.array([0.8, 1.2, 1.0, 0.9])
    noise = 0.05 * np.std(parts[1]) * (rng.standard_normal(parts[0].shape) + 1j * rng.standard_normal(parts[0].shape))
    observed = _particles(scales[owner, None] * parts[0] + parts[1] + noise, ctf)
    every = np.arange(n_images)
    flat = _shell_noise()
    core_only = subtract_fixed_core(
        observed, fixed(core_real), scales=estimate_core_scales(observed, fixed(core_real), flat)
    )
    background = estimate_residual_background(core_only, "cylinder", ridge=1e-2)
    # The core-only fit's scale takes up the ring's component along the core, so the background holds the rest of
    # the ring (measured correlation with the whole ring 0.68).
    assert np.corrcoef(background.ravel(), ring_real.ravel())[0, 1] > 0.5
    basis = (fixed(core_real), fixed(background))
    coefficients = estimate_core_coefficients(observed, basis, flat)
    both = subtract_fixed_core(observed, basis, scales=coefficients)
    left = [float(np.sum(np.abs(np.asarray(x.read(every)[1])) ** 2)) for x in (core_only, both)]
    assert left[1] < 0.2 * left[0], left
