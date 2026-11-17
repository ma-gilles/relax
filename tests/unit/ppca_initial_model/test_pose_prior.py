"""The per-particle pose prior (``--pose-prior-sigma-deg``; relax/ppca_refinement/pose_prior.py, section 19).

The prior's values are checked against an independent SciPy reference, its convention against the rotation grid
(an input pose on the grid peaks at that grid row) and the projector (``R`` and ``R G`` project a ``G``-symmetric
map alike), and the streams that score with it against the float64 brute-force joint Gaussian of
``test_tomo_ppca`` with the same per-particle prior. Off (the default) the streams carry no prior at all.
"""

import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
import test_tomo_ppca as tomo
from helpers.float_compare import assert_matches
from helpers.full_row_tile import accumulate_full_row_tile
from recovar import core
from scipy.spatial.transform import Rotation

from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.pose_prior import read_input_rotations
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.tomo import load_tilt_tile
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_stream import full_row_tile_embeddings, prepare_full_row_stream, read_tile
from relax.ppca_refinement.pose_prior import (
    PosePrior,
    device_prior,
    log_normalizer,
    outer_rotation_log_prior,
    paired_rotation_log_prior,
)
from relax.sampling import grids
from relax.sampling.symmetry import rotational_operators

pytestmark = pytest.mark.unit


def reference_log_prior(rotations, centres, sigma_deg, symmetry="C1"):
    """Float64 ``-theta^2 / (2 sigma^2)`` ``(R, B)`` with SciPy's rotation magnitude, unnormalized."""
    operators = rotational_operators(symmetry, dtype=np.float64)
    theta = np.full((len(rotations), len(centres)), np.inf)
    for G in operators:
        for b, centre in enumerate(np.asarray(centres, np.float64)):
            relative = np.einsum("ji,rjk->rik", centre, np.asarray(rotations, np.float64) @ G)
            theta[:, b] = np.minimum(theta[:, b], Rotation.from_matrix(relative).magnitude())
    return -(theta**2) / (2 * np.deg2rad(sigma_deg) ** 2)


def reference_log_norm(rotations, sigma_deg, symmetry="C1"):
    values = reference_log_prior(rotations, np.eye(3)[None], sigma_deg, symmetry)[:, 0]
    return np.log(np.mean(np.exp(values - values.max()))) + values.max()


def _prior(centres, sigma_deg, symmetry="C1"):
    return PosePrior(np.asarray(centres, np.float32), sigma_deg, rotational_operators(symmetry, dtype=np.float64))


@pytest.mark.parametrize("symmetry, sigma", [("C1", 12.0), ("D7", 7.0), ("C4", 40.0)])
def test_prior_matches_an_independent_reference(symmetry, sigma):
    rotations = Rotation.random(40, random_state=3).as_matrix().astype(np.float32)
    centres = Rotation.random(6, random_state=4).as_matrix().astype(np.float32)
    grid = grids.get_relion_hidden_rotation_grid(1, matrices=True).astype(np.float32)
    prior = _prior(centres, sigma, symmetry)
    log_norm = reference_log_norm(grid, sigma, symmetry)
    # The float32 grid's matrices are orthonormal to about 1e-7, which SciPy's magnitude projects away.
    assert log_normalizer(prior, grid) == pytest.approx(log_norm, rel=1e-6)
    expected = reference_log_prior(rotations, centres, sigma, symmetry) - log_norm
    actual = np.asarray(
        outer_rotation_log_prior(device_prior(prior, grid), jnp.asarray(rotations), jnp.asarray(centres))
    )
    # A float32 trace (error ~1e-7) gives theta to ~1e-3 rad near pi, where d theta / d tr = -1 / (2 sin theta):
    # at most 2e-3 / pi of the largest |log-prior| pi^2 / (2 sigma^2). Below 120 degrees it is ~1e-6 of it.
    assert_matches(actual, expected, rtol=1e-3)
    near = reference_log_prior(rotations, centres, 1.0, symmetry) > -((2 * np.pi / 3) ** 2) / (2 * np.deg2rad(1.0) ** 2)
    assert near.sum() > 10
    assert_matches(actual[near], expected[near], rtol=2e-5)
    paired = paired_rotation_log_prior(
        device_prior(prior, grid), jnp.asarray(rotations[:30].reshape(6, 5, 3, 3)), jnp.asarray(centres)
    )
    assert_matches(np.asarray(paired), np.stack([actual[5 * j : 5 * j + 5, j] for j in range(6)]), rtol=1e-5)


def test_symmetry_copies_of_the_input_pose_get_the_top_prior():
    """With D7 every copy ``R0 G`` is the input pose (theta 0); without symmetry it is ``G``'s angle away."""
    centre = Rotation.random(1, random_state=8).as_matrix().astype(np.float32)
    operators = rotational_operators("D7", dtype=np.float64)
    copies = (centre[0].astype(np.float64) @ operators).astype(np.float32)
    grid = grids.get_relion_hidden_rotation_grid(1, matrices=True).astype(np.float32)
    symmetric = device_prior(_prior(centre, 5.0, "D7"), grid)
    values = np.asarray(outer_rotation_log_prior(symmetric, jnp.asarray(copies), jnp.asarray(centre)))[:, 0]
    assert_matches(values, np.full(14, -float(symmetric.log_norm)), rtol=1e-5)
    plain = device_prior(_prior(centre, 5.0), grid)
    values = np.asarray(outer_rotation_log_prior(plain, jnp.asarray(copies), jnp.asarray(centre)))[:, 0]
    angles = Rotation.from_matrix(operators).magnitude()
    assert_matches(values, -(angles**2) / (2 * np.deg2rad(5.0) ** 2) - float(plain.log_norm), rtol=1e-4)


def test_symmetric_map_projects_alike_at_r_and_r_times_g():
    """The convention the prior's ``R G`` copies assume: a map invariant under D2 (flips about the box centre)
    projects alike at ``R`` and ``R G``, and not at ``G R``."""
    n = 16
    rng = np.random.default_rng(0)
    real = rng.standard_normal((n, n, n))
    flip = np.r_[0, n - 1 : 0 : -1]  # x -> -x about the centre n / 2
    symmetric = real + real[:, flip][:, :, flip] + real[flip][:, :, flip] + real[flip][:, flip]
    full = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(symmetric)))
    half = jnp.asarray(ftu.full_volume_to_half_volume(full.astype(np.complex64), (n, n, n)).reshape(1, -1))
    operators = rotational_operators("D2", dtype=np.float64)
    assert np.allclose(np.abs(operators), np.abs(np.round(operators))) and len(operators) == 4
    R = Rotation.random(3, random_state=1).as_matrix()

    def project(rotations):
        images = core.batch_slice_volume(
            half, jnp.asarray(rotations, jnp.float32), (n, n), (n, n, n), "linear_interp", half_volume=True
        )
        return np.asarray(images)[0]

    # Away from the box edge, where a flipped sample would leave the grid.
    inside = np.asarray(ftu.get_grid_of_radial_distances((n, n), rounded=False)).reshape(-1) < n // 2 - 1
    reference = project(R)[:, inside]
    for G in operators[1:]:
        assert_matches(project(R @ G)[:, inside], reference, rtol=1e-5)
        assert np.max(np.abs(project(G @ R)[:, inside] - reference)) > 0.1 * np.max(np.abs(reference))


def _write_star(path, eulers):
    lines = ["data_optics", "", "loop_", "_rlnOpticsGroup #1", "1", "", "data_particles", "", "loop_"]
    lines += ["_rlnAngleRot #1", "_rlnAngleTilt #2", "_rlnAnglePsi #3", "_rlnOpticsGroup #4"]
    lines += [f"{rot:.9f} {tilt:.9f} {psi:.9f} 1" for rot, tilt, psi in np.asarray(eulers, np.float64)]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("order", [1, 2])
def test_input_pose_on_the_grid_peaks_at_that_grid_row(tmp_path, order):
    """A particle STAR whose angles are grid rows' canonical Euler angles: read back, each particle's prior is
    largest at its own row (the grid's matrices are RELION's Euler_angles2matrix of those angles)."""
    eulers = grids.get_relion_hidden_rotation_grid(order, matrices=False)
    rows = np.asarray([0, 7, len(eulers) // 3, len(eulers) - 1])
    _write_star(tmp_path / "particles.star", eulers[rows])
    centres = read_input_rotations(tmp_path / "particles.star")
    grid = iteration_loop._rotation_grid(order)
    values = outer_rotation_log_prior(device_prior(_prior(centres, 3.0), grid), jnp.asarray(grid), jnp.asarray(centres))
    np.testing.assert_array_equal(np.argmax(np.asarray(values), axis=0), rows)


def test_reader_refuses_a_star_without_angles(tmp_path):
    (tmp_path / "particles.star").write_text("data_particles\n\nloop_\n_rlnOpticsGroup #1\n1\n")
    with pytest.raises(ValueError, match="rlnAngleRot"):
        read_input_rotations(tmp_path / "particles.star")


def _tilt_stream(particles, arrays, *, pose_prior, image_frames, noise=None, batch=3):
    noise = [np.full(tomo.IMAGE_SHAPE, n, np.float32) for n in noise] if noise is not None else None
    return prepare_full_row_stream(
        particles,
        arrays["mu"],
        arrays["W"],
        noise_variance=noise if noise is not None else np.full(tomo.IMAGE_SHAPE, tomo.NOISE, np.float32),
        rotations=arrays["rotations"],
        translations=tomo.TRANSLATIONS,
        rotation_log_prior=arrays["rotation_log_prior"],
        translation_log_prior=arrays["translation_log_prior"],
        rotation_parent=np.zeros(5, np.int32),
        translation_parent=np.zeros(len(tomo.TRANSLATIONS), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=6, q=2, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=batch, rotation_block_size=2),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        tile_loader=load_tilt_tile,
        gemm_precision="fp32",
        pose_prior=pose_prior,
    )


def _centres(arrays, n, seed):
    """Input poses near the problem's candidate rotations, so the prior reorders them."""
    rng = np.random.default_rng(seed)
    near = arrays["rotations"][rng.integers(0, 5, n)].astype(np.float64)
    return (near @ Rotation.from_rotvec(rng.normal(scale=0.3, size=(n, 3))).as_matrix()).astype(np.float32)


def _assert_tile_matches(actual, truth, rtol=2e-5):
    """The one-group tile's band (test_tomo_ppca.test_tilt_tile_matches_brute_force_joint_gaussian)."""
    assert_matches(actual.log_likelihood, truth["log_likelihood"], rtol=rtol)
    assert_matches(np.asarray(actual.embeddings), truth["embeddings"], rtol=rtol)
    assert_matches(actual.diagnostics["rotation_mass"], truth["rotation_mass"], rtol=rtol)
    assert_matches(np.asarray(actual.lhs_tri), truth["lhs_tri"], rtol=rtol)
    assert_matches(np.asarray(actual.residual_gradient), truth["residual_gradient"], rtol=rtol)
    assert_matches(np.asarray(actual.residual_num), truth["residual_num"], rtol=rtol)


@pytest.mark.parametrize("planned", [None, 16])
def test_tilt_tile_with_pose_prior_matches_brute_force(planned):
    """One tilt group, unpadded and padded: the stream's statistics are the joint Gaussian's with each particle's
    prior added to the shared rotation prior, and the prior moves the rotation mass."""
    particles, _, arrays = tomo.make_problem(seed=6)
    centres = _centres(arrays, 3, seed=1)
    prior = _prior(centres, 25.0, "C2")
    stream = _tilt_stream(particles, arrays, pose_prior=prior, image_frames=tomo.IMAGE_FRAMES)._replace(
        tile_images=planned
    )
    image_prior = (
        reference_log_prior(arrays["rotations"], centres, 25.0, "C2") - log_normalizer(prior, arrays["rotations"])
    ).T
    truth = tomo.brute_force(particles, stream, {**arrays, "image_rotation_log_prior": image_prior})
    actual = accumulate_full_row_tile(stream, np.arange(3), [None] * 3)
    _assert_tile_matches(actual, truth)
    plain = tomo.brute_force(particles, stream, arrays)
    assert np.max(np.abs(truth["rotation_mass"] - plain["rotation_mass"])) > 0.05


def _mixed(seed=12):
    """test_tomo_ppca.make_mixed_problem's particles (three tilt groups, two noise groups) and model."""
    _, _, arrays = tomo.make_problem(image_frames=tomo.MIXED_IMAGE_FRAMES, seed=seed)
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in tomo.MIXED_IMAGE_FRAMES])]).astype(np.int64)
    images, ctf = arrays["images"].astype(np.complex64), arrays["ctf"]
    particles = tomo.TiltParticles(
        image_shape=tomo.IMAGE_SHAPE,
        volume_shape=tomo.VOLUME_SHAPE,
        voxel_size=1.0,
        image_offsets=offsets,
        image_frame=np.concatenate(tomo.MIXED_IMAGE_FRAMES).astype(np.int64),
        particle_group=tomo.MIXED_PARTICLE_GROUP,
        group_frames=tuple(np.asarray(f, np.float64) for f in tomo.MIXED_GROUP_FRAMES),
        read=lambda ids: (None, jnp.asarray(images[np.asarray(ids)]), jnp.asarray(ctf[np.asarray(ids)])),
        noise_group=tomo.MIXED_NOISE_GROUP,
    )
    return particles, arrays


@pytest.mark.parametrize("planned", [None, 10])
def test_mixed_tile_with_pose_prior_matches_brute_force(planned):
    """Five particles of three tilt groups and two noise groups in one mixed tile (particle order differs from the
    keys' slot order): each particle takes its own input pose."""
    particles, arrays = _mixed()
    centres = _centres(arrays, 5, seed=2)
    prior = _prior(centres, 25.0)
    stream = _tilt_stream(
        particles, arrays, pose_prior=prior, image_frames=tomo.MIXED_IMAGE_FRAMES, noise=tomo.MIXED_NOISE, batch=5
    )._replace(tile_images=planned)
    image_prior = (
        reference_log_prior(arrays["rotations"], centres, 25.0) - log_normalizer(prior, arrays["rotations"])
    ).T
    truth = tomo.brute_force(
        particles,
        stream,
        {**arrays, "noise": tomo.MIXED_NOISE[tomo.MIXED_NOISE_GROUP], "image_rotation_log_prior": image_prior},
    )
    tile, _, _ = read_tile(stream, np.arange(5), [None] * 5, collect_observation=False)
    assert tile.frames.ndim == 4 and tile.pose_centres.shape == (tile.coarse_mask.shape[0], 3, 3)
    _assert_tile_matches(accumulate_full_row_tile(stream, np.arange(5), [None] * 5), truth)
    embedded = full_row_tile_embeddings(stream, np.arange(5), [None] * 5)
    assert_matches(np.asarray(embedded.embeddings), truth["embeddings"], rtol=2e-5)
    # The sparse mixed pass 1 (coarse parents, every coarse pose significant) takes the same prior.
    sparse = accumulate_full_row_tile(tomo._coarse_stream(stream), np.arange(5), [None] * 5)
    _assert_tile_matches(sparse, truth)


def test_oversampled_children_take_the_prior_of_the_dense_child_grid():
    """Adaptive oversampling with fraction 1: pass 2's children score as the dense child grid with the prior."""
    from test_full_fine_streaming import IMAGE_SHAPE, _TinyData
    from test_oversampled_stream import _assert_same_statistics, _parent_mass, _spa_stream

    from relax.ppca_refinement.oversampled_stream import accumulate_oversampled_tile, prepare_oversampled_stream

    coarse, (fine_rotations, fine_translations) = _spa_stream()
    # The centres are indexed by original image id: _TinyData's are 100 + the local ids.
    centres = Rotation.random(104, random_state=5).as_matrix().astype(np.float32)
    grid = np.asarray(coarse.arrays.rotations[:-1])
    prior = _prior(centres, 20.0)
    assert isinstance(coarse.dataset, _TinyData) and coarse.static.image_shape == IMAGE_SHAPE
    coarse = coarse._replace(
        arrays=coarse.arrays._replace(pose_prior=device_prior(prior, grid)), pose_centres=prior.centres
    )
    n_poses = len(grid) * coarse.translations.shape[0]
    ostream = prepare_oversampled_stream(
        coarse, fine_rotations, fine_translations, adaptive_fraction=1.0, max_significant=n_poses, job_chunk=96
    )
    ids = np.arange(4)
    actual = accumulate_oversampled_tile(ostream, ids)
    expected = accumulate_full_row_tile(ostream.fine, ids, [None] * 4)
    _assert_same_statistics(actual, expected)
    # Against the same run without the prior, the children's mass moves.
    plain = accumulate_full_row_tile(
        ostream.fine._replace(arrays=ostream.fine.arrays._replace(pose_prior=None), pose_centres=None),
        ids,
        [None] * 4,
    )
    assert np.max(np.abs(_parent_mass(expected, 8) - _parent_mass(plain, 8))) > 0.05


def test_streams_without_a_pose_prior_carry_none():
    """Off: neither the stream nor its tiles hold a prior, so every program is the one without it."""
    particles, stream, _ = tomo.make_problem(seed=6)
    assert stream.arrays.pose_prior is None and stream.pose_centres is None
    tile, _, _ = read_tile(stream, np.arange(3), [None] * 3, collect_observation=False)
    assert tile.pose_centres is None
    config = Config()
    assert config.pose_prior_sigma_deg is None and not config.pose_prior_active(1)
    assert iteration_loop._pose_prior(config, 1, None, 3) is None


def _zero_model_state(particles, q=2):
    """A zero model: every pose scores alike, so each particle's posterior is its prior."""
    n_half = particles.volume_shape[0] ** 2 * (particles.volume_shape[0] // 2 + 1)
    theta = jnp.zeros((n_half, q + 1), jnp.complex64)
    noise = jnp.ones((1, particles.grid_size // 2 + 1), jnp.float32)
    return State(theta, None, noise, 0, np.arange(particles.n_images), {}, 1.0, 0, {})


@pytest.mark.parametrize("options", [{}, {"tilt_coarse_pass": True, "max_significant": 2}])
def test_controller_top_pose_is_the_input_pose(options):
    """Through the controller's E-step (tiles of two tilt groups, padding, shuffled ids; with the coarse pass at
    two significant coarse poses per particle): with a zero model each particle's top pose is its input grid row,
    the posterior being its prior. Without the prior in the coarse pass its pruning would keep two arbitrary cells
    (every coarse pose scores alike) and lose that row."""
    particles = tomo._controller_particles()
    hp = 2
    eulers = grids.get_relion_hidden_rotation_grid(hp, matrices=False)
    rows = np.random.default_rng(4).choice(len(eulers), particles.n_images, replace=False)
    from relax.sampling.healpix import euler_angles_to_matrix

    input_poses = euler_angles_to_matrix(eulers[rows]).astype(np.float32)
    config = Config(
        q=2, stages=((1, 3, hp),), stream_coarse_recompute=True, shift_range=1, shift_step=1, image_batch_size=3,
        rotation_block_size=64, pose_prior_sigma_deg=4.0, uniform_direction_prior=True, **options,
    )  # fmt: skip
    iteration_loop._rotation_grid.cache_clear()
    try:
        final = iteration_loop.expectation(
            particles, _zero_model_state(particles), config, np.asarray([5, 2, 7, 0, 3, 6, 1, 4]), 1,
            embeddings_only=True, input_poses=input_poses,
        )  # fmt: skip
    finally:
        iteration_loop._rotation_grid.cache_clear()
    np.testing.assert_array_equal(final.best_rotation, rows[final.original_image_ids])
    zero = np.flatnonzero(~np.any(grids.get_relion_translation_grid_3d(1, 1), axis=1))
    np.testing.assert_array_equal(final.best_translation, np.repeat(zero, 8))  # the prior's zero-centred shift


def test_controller_refuses_a_missing_or_misshaped_input_pose_set():
    config = Config(pose_prior_sigma_deg=5.0)
    with pytest.raises(ValueError, match="input pose"):
        iteration_loop._pose_prior(config, 1, None, 3)
    with pytest.raises(ValueError, match="rotations for 3 particles"):
        iteration_loop._pose_prior(config, 1, np.zeros((2, 3, 3)), 3)


def test_config_validates_and_schedules_the_prior():
    assert Config(pose_prior_sigma_deg=5.0, pose_prior_until=3).pose_prior_active(3)
    assert not Config(pose_prior_sigma_deg=5.0, pose_prior_until=3).pose_prior_active(4)
    assert Config(pose_prior_sigma_deg=5.0, pose_prior_symmetry="D7").pose_prior_active(10**6)
    for options, message in [
        ({"pose_prior_symmetry": "D7"}, "need --pose-prior-sigma-deg"),
        ({"pose_prior_until": 4}, "need --pose-prior-sigma-deg"),
        ({"pose_prior_sigma_deg": 0.0}, "finite and positive"),
        ({"pose_prior_sigma_deg": 5.0, "pose_prior_symmetry": "d7"}, "canonical"),
        ({"pose_prior_sigma_deg": 5.0, "pose_prior_until": 0}, "positive update"),
        ({"pose_prior_sigma_deg": 5.0, "band_length_ang": 100.0}, "filament band"),
        ({"pose_prior_sigma_deg": 5.0, "stream_coarse_recompute": False}, "streamed engine"),
    ]:
        with pytest.raises(ValueError, match=message):
            Config(**options)


def test_checkpoint_round_trips_the_pose_prior(tmp_path):
    from relax.ppca_initial_model.checkpoint import load, save

    state = State(jnp.ones((5, 3), jnp.complex64), None, jnp.ones(3, jnp.float32), 4, np.arange(8), {}, 2.0, 4, {})
    config = Config(optimizer="momentum_sgd", pose_prior_sigma_deg=6.0, pose_prior_symmetry="D7", pose_prior_until=10)
    state = dataclasses.replace(state, sgd_momentum=jnp.zeros((5, 3), jnp.complex64))
    identity = {"input": "abc"}
    save(tmp_path / "checkpoint.npz", state, config, identity)
    assert load(tmp_path / "checkpoint.npz", config, identity).iteration == 4
    meta = json.loads(str(np.load(tmp_path / "checkpoint.npz")["metadata"]))
    stored = {key: value for key, value in meta["config"].items() if key.startswith("pose_prior")}
    assert stored == {"pose_prior_sigma_deg": 6.0, "pose_prior_symmetry": "D7", "pose_prior_until": 10}
    for changed in ({"pose_prior_sigma_deg": 7.0}, {"pose_prior_symmetry": "C1"}, {"pose_prior_until": None}):
        with pytest.raises(ValueError, match="identity mismatch"):
            load(tmp_path / "checkpoint.npz", dataclasses.replace(config, **changed), identity)
    # A checkpoint from before the option existed resumes without it.
    plain = dataclasses.replace(config, pose_prior_sigma_deg=None, pose_prior_symmetry="C1", pose_prior_until=None)
    save(tmp_path / "old.npz", state, plain, identity)
    with np.load(tmp_path / "old.npz") as arrays:
        stored = dict(arrays)
    meta = json.loads(str(stored["metadata"]))
    for key in ("pose_prior_sigma_deg", "pose_prior_symmetry", "pose_prior_until"):
        del meta["config"][key]
    stored["metadata"] = json.dumps(meta)
    np.savez(tmp_path / "old.npz", **stored)
    assert load(tmp_path / "old.npz", plain, identity).iteration == 4
    with pytest.raises(ValueError, match="identity mismatch"):
        load(tmp_path / "old.npz", config, identity)


def test_run_records_and_resumes_the_released_prior(tmp_path):
    """Two updates with the prior until update 1: the log says which updates had it, and the run resumes from its
    first checkpoint with the same option."""
    particles = tomo._controller_particles()
    input_poses = Rotation.random(particles.n_images, random_state=2).as_matrix().astype(np.float32)
    config = Config(
        q=2, iterations=2, stages=((1, 3, 0),), stream_coarse_recompute=True, shift_range=1, shift_step=1,
        image_batch_size=3, rotation_block_size=32, pose_prior_sigma_deg=10.0, pose_prior_until=1,
    )  # fmt: skip
    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    try:
        iteration_loop.run(
            particles, config, tmp_path, {"test": True}, diameter_ang=120.0, stop_after=1, input_poses=input_poses
        )
        state = iteration_loop.run(
            particles, config, tmp_path, {"test": True}, diameter_ang=120.0, input_poses=input_poses,
            resume=tmp_path / "checkpoint_0001.npz",
        )  # fmt: skip
        with pytest.raises(ValueError, match="identity mismatch"):
            iteration_loop.run(
                particles, dataclasses.replace(config, pose_prior_sigma_deg=11.0), tmp_path, {"test": True},
                diameter_ang=120.0, input_poses=input_poses, resume=tmp_path / "checkpoint_0001.npz",
            )  # fmt: skip
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    rows = [json.loads(line) for line in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    assert [(row["iteration"], row["pose_prior"]) for row in rows] == [(1, True), (2, False)]
    assert state.iteration == 2


def test_ios_command_reads_the_input_poses(tmp_path, monkeypatch):
    """``relax ppca_initial_model --ios ... --pose-prior-sigma-deg``: the optimisation set's particle STAR angles
    centre the prior, and a pose count that is not the particle count is refused."""
    from relax.commands import ppca_initial_model as command
    from relax.ppca_initial_model import pose_prior as input_module
    from relax.star import tomo_input

    particles = tomo._controller_particles()
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setattr(command, "load_tilt_training", lambda *args, **options: (particles, {"test": True}, None))
    monkeypatch.setattr(command, "source_identity", lambda: {"test": True})
    monkeypatch.setattr(tomo_input, "read_optimisation_set", lambda ios: ("particles.star", "tomograms.star"))
    read = []

    def poses(star, n=particles.n_images):
        read.append(star)
        return Rotation.random(n, random_state=1).as_matrix().astype(np.float32)

    monkeypatch.setattr(input_module, "read_input_rotations", poses)
    arguments = [
        "--ios", "optimisation_set.star", "--particle-diameter", "120", "--q", "2", "--iterations", "20",
        "--stages", "[[1, 3, 0]]", "--shift-range", "1", "--shift-step", "1", "--image-batch-size", "3",
        "--rotation-block-size", "32", "--skip-final-embeddings", "--stop-after", "1",
        "--pose-prior-sigma-deg", "8", "--pose-prior-symmetry", "d2", "--pose-prior-until", "5",
    ]  # fmt: skip
    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    try:
        command.main(arguments + ["-o", str(tmp_path / "run")])
        monkeypatch.setattr(input_module, "read_input_rotations", lambda star: poses(star, particles.n_images - 1))
        with pytest.raises(ValueError, match="poses for 8 particles"):
            command.main(arguments + ["-o", str(tmp_path / "short")])
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    assert read == ["particles.star", "particles.star"]
    config = json.loads((tmp_path / "run" / "run.json").read_text())["config"]
    assert (config["pose_prior_sigma_deg"], config["pose_prior_symmetry"], config["pose_prior_until"]) == (8.0, "D2", 5)
    rows = [json.loads(line) for line in (tmp_path / "run" / "iterations.jsonl").read_text().splitlines()]
    assert [row["pose_prior"] for row in rows] == [True]


def test_ranks_agree_on_the_input_poses():
    from relax.ppca_initial_model.ranks import run_fingerprint

    particles = tomo._controller_particles()
    poses = Rotation.random(8, random_state=1).as_matrix().astype(np.float32)
    config = Config(pose_prior_sigma_deg=5.0)
    same = run_fingerprint(particles, config, {"x": 1}, poses)
    assert same == run_fingerprint(particles, config, {"x": 1}, poses.copy())
    assert same != run_fingerprint(particles, config, {"x": 1}, poses[::-1].copy())
    assert run_fingerprint(particles, Config(), {"x": 1})["input_poses_sha256"] is None
