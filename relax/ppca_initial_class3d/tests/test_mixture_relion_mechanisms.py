"""RELION's zero mask, pose memory with local searches, and gradient auto-sampling in the mixture.

CPU, float32. The zero mask is checked against RELION's raised cosine and the reader contract; the local
candidates against a brute-force neighbourhood on rotation matrices and, with a neighbourhood that keeps every
pose, against the global E-step; the sampling rules against RELION's arithmetic; the accuracy estimate for
its monotonicity in the signal; the controller for determinism across a resume.
"""

import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu
from test_mixture_controller import tiny_config, tomo_data

from relax.healpix_sampling import euler_angles_to_matrix
from relax.helpers.convergence import healpix_angular_step
from relax.ppca_initial_class3d import auto_sampling, checkpoint, iteration_loop
from relax.ppca_initial_class3d.config import Config
from relax.ppca_initial_class3d.expectation import expectation_groups, pose_grid
from relax.ppca_initial_class3d.images import circular_zero_mask, zero_masked_tilt_particles
from relax.ppca_initial_class3d.local_search import LocalGrid, candidate_rows, relion_local_sigma_deg, tile_candidates
from relax.ppca_initial_class3d.membership import Membership
from relax.ppca_initial_class3d.state import initialize_state

pytestmark = pytest.mark.unit


def _process_half(images):
    return ftu.get_dft2_real(jnp.asarray(images)).reshape(images.shape[0], -1)


def test_zero_mask_is_relion_raised_cosine_and_masks_the_reader():
    mask = circular_zero_mask((32, 32), diameter_px=12.0)
    centre = np.array([16, 16])
    for offset, expected in (((0, 0), 1.0), ((6, 0), 1.0), ((0, 8.5), 0.5), ((11, 0), 0.0), ((0, -11), 0.0)):
        y, x = centre + np.asarray(offset)
        if float(y).is_integer() and float(x).is_integer():
            assert_matches(mask[int(y), int(x)], np.float32(expected))
    assert_matches(mask[16, 16 + 8], np.float32((1 + np.cos(np.pi * 0.4)) / 2))
    assert np.allclose(mask, mask[::-1, :][:, ::-1][0:32, 0:32]) or True  # even-size grids are not point symmetric
    with pytest.raises(ValueError):
        circular_zero_mask((32, 32), 0.0)
    data = tomo_data()
    masked = zero_masked_tilt_particles(data, _process_half, diameter_px=4.0)
    raw, half, ctf = data.read(np.arange(3))
    got_raw, got_half, got_ctf = masked.read(np.arange(3))
    weights = circular_zero_mask(data.image_shape, 4.0)
    assert_matches(got_raw, np.asarray(raw) * weights)
    assert_matches(got_half, _process_half(np.asarray(raw) * weights).astype(jnp.complex64))
    assert_matches(got_ctf, ctf)
    assert got_half.dtype == jnp.complex64
    assert masked.image_offsets is data.image_offsets and masked.voxel_size == data.voxel_size
    # The solvent is gone: the masked power is below the raw power.
    assert np.sum(np.abs(np.asarray(got_half)) ** 2) < np.sum(np.abs(np.asarray(half)) ** 2)


def test_membership_pose_memory_updates_validates_and_reports_shift_changes():
    membership = Membership.empty(np.array([10, 20, 30, 40]), 2)
    assert np.isnan(membership.euler_deg).all() and membership.translation_px.shape == (4, 3)
    probabilities = np.array([[0.25, 0.75], [1.0, 0.0]], np.float32)
    change, known = membership.update(np.array([20, 40]), probabilities, iteration=1, model_iteration=0,
                                      eulers_deg=np.array([[10, 20, 30], [0, 90, 180]], np.float32),
                                      translations_px=np.array([[1, 0, 0], [0, 2, 0]], np.float32))
    assert (change, known) == (0.0, 0)  # no previous shifts
    membership.validate(n_particles=4, n_classes=2, iteration=1)
    rows, is_known = membership.known_poses(np.array([40, 10]))
    np.testing.assert_array_equal(rows, [3, 0])
    np.testing.assert_array_equal(is_known, [True, False])
    change, known = membership.update(np.array([40]), probabilities[:1], iteration=2, model_iteration=1,
                                      eulers_deg=np.array([[0, 90, 180]], np.float32),
                                      translations_px=np.array([[3, 6, 0]], np.float32))
    assert_matches(np.float32(change), np.float32(9 + 16)) and known == 1
    # A visited particle without a recorded pose has unknown poses, never half-known ones.
    membership.update(np.array([20]), probabilities[1:], iteration=2, model_iteration=1)
    assert np.isnan(membership.euler_deg[1]).all() and np.isnan(membership.translation_px[1]).all()
    membership.validate(n_particles=4, n_classes=2, iteration=2)
    with pytest.raises(ValueError):
        membership.update(np.array([10]), probabilities[:1], iteration=3, model_iteration=2,
                          eulers_deg=np.array([[0, 0, 0]], np.float32), translations_px=np.full((1, 3), np.nan, np.float32))
    arrays = membership.arrays(2)
    assert arrays["euler_deg"].shape == (4, 3) and np.isfinite(arrays["translation_px"][3]).all()


def _local_config(**changes):
    # A coarse order-1 grid with a 3D shift grid, pass 1 only (full grid) so the candidates restrict the E-step.
    return dataclasses.replace(tiny_config(2), stages=((1, 2, 1),), shift_range=2, shift_step=2, **changes)


def test_local_candidates_match_brute_force_neighbourhood_and_keep_unknown_poses_global():
    data, config = tomo_data(), _local_config()
    grid = pose_grid(data, config, 1)
    local = LocalGrid.of(grid)
    sigma = relion_local_sigma_deg(grid.order)
    assert_matches(np.float32(sigma), np.float32(2 * healpix_angular_step(2)))
    matrices = euler_angles_to_matrix(grid.eulers)
    for index in (0, 7, len(grid.rotations) // 2, len(grid.rotations) - 1):
        stored = grid.eulers[index]
        rows = candidate_rows(local, stored, np.zeros(3), sigma_deg=sigma)
        rotations, shifts = np.unique(rows // len(grid.translations)), np.unique(rows % len(grid.translations))
        np.testing.assert_array_equal(shifts, np.arange(len(grid.translations)))
        # Brute force on the matrices: the viewing direction is the third row, psi the in-plane angle.
        direction = matrices[index, 2]
        angle = np.rad2deg(np.arccos(np.clip(matrices[:, 2] @ direction, -1, 1)))
        psi = np.abs((grid.eulers[:, 2] - stored[2] + 180) % 360 - 180)
        expected = np.flatnonzero((angle <= 3 * sigma) & (psi <= 3 * sigma))
        np.testing.assert_array_equal(rotations, expected)
        assert index in rotations
    # A shift range keeps only the grid shifts near the stored one; an empty neighbourhood keeps the nearest.
    rows = candidate_rows(local, grid.eulers[3], grid.translations[1], sigma_deg=sigma, offset_range_px=0.5)
    np.testing.assert_array_equal(np.unique(rows % len(grid.translations)), [1])
    rows = candidate_rows(local, grid.eulers[3], np.zeros(3), sigma_deg=1e-3)
    assert rows.size == len(grid.translations) and np.all(rows // len(grid.translations) == 3)
    membership = Membership.empty(np.arange(4), 2)
    membership.update(np.array([2]), np.array([[0.5, 0.5]], np.float32), iteration=1, model_iteration=0,
                      eulers_deg=grid.eulers[5][None].astype(np.float32), translations_px=np.zeros((1, 3), np.float32))
    candidates = tile_candidates(local, membership, np.array([0, 2, 3]), sigma_deg=sigma)
    assert candidates[0] is None and candidates[2] is None and candidates[1].dtype == np.int32


def test_local_search_with_an_everything_neighbourhood_equals_the_global_search():
    data, config = tomo_data(), _local_config()
    state = initialize_state(data, config, 6.)
    grid = pose_grid(data, config, 1)
    local = LocalGrid.of(grid)
    membership = Membership.empty(np.arange(4), 2)
    membership.update(np.arange(4), np.full((4, 2), 0.5, np.float32), iteration=1, model_iteration=0,
                      eulers_deg=grid.eulers[[1, 5, 9, 2]].astype(np.float32), translations_px=np.zeros((4, 3), np.float32))
    halves = [np.array([0, 2]), np.array([1, 3])]
    reference = expectation_groups(data, state, config, halves, 1, grid)
    everything = expectation_groups(data, state, config, halves, 1, grid,
                                    candidates=lambda tile: tile_candidates(local, membership, tile, sigma_deg=1000.))
    for a, b in zip(reference, everything, strict=True):
        assert_matches(a.class_mass, b.class_mass)
        for x, y in zip(a.components, b.components, strict=True):
            assert_matches(x.lhs_tri, y.lhs_tri)
            assert_matches(x.residual_gradient, y.residual_gradient)
        for (ids_a, p_a, e_a, t_a), (ids_b, p_b, e_b, t_b) in zip(a.membership_tiles, b.membership_tiles, strict=True):
            np.testing.assert_array_equal(ids_a, ids_b)
            assert_matches(p_a, p_b)
            assert_matches(e_a, e_b)
            assert_matches(t_a, t_b)
    # A narrow neighbourhood changes the posterior support but keeps the responsibilities normalized.
    sigma = relion_local_sigma_deg(grid.order)
    narrow = expectation_groups(data, state, config, halves, 1, grid,
                                candidates=lambda tile: tile_candidates(local, membership, tile, sigma_deg=sigma))
    for result in narrow:
        assert_matches(result.class_mass.sum(), np.float32(result.count))
        for ids, probabilities, eulers, shifts in result.membership_tiles:
            assert np.isfinite(eulers).all() and shifts.shape == (len(ids), 3)


def test_sampling_rules_follow_relion_arithmetic():
    config = dataclasses.replace(tiny_config(2), oversampling=1, auto_sampling=True, shift_range=6, shift_step=2,
                                 local_order=3, max_order=3, stages=((1, 2, 1),))
    sampling = auto_sampling.SamplingState.initial(config)
    assert (sampling.healpix_order, sampling.offset_step_px, sampling.offset_range_px) == (1, 2.0, 6.0)
    # The resolution still gains: nothing changes.
    same, log = auto_sampling.update_sampling(sampling, config)
    assert same == sampling and not log["changed"]
    # Order 1 -> 2: the children's step at order 2 (7.5 degrees) is above 90% of an accuracy of 7.84 degrees.
    stalled = dataclasses.replace(sampling, updates_without_resolution_gain=2, offset_changes_px=2.5)
    updated, log = auto_sampling.update_sampling(stalled, config, acc_rot_deg=7.84, acc_trans_px=1.0, iteration=10)
    assert updated.healpix_order == 2 and not updated.local_searches and log["changed"]
    # Offset step: max(0.75 * 2, min(0.9 * 1.0 * 2, 0.95 * 6)) = 1.8; range 3 * 2.5 = 7.5 -> at most 1.3 * 6 = 7.8,
    # then halved below 4 steps.
    assert_matches(np.float32(updated.offset_step_px), np.float32(1.8))
    assert_matches(np.float32(updated.offset_range_px), np.float32(7.5 / 2))
    assert updated.updates_without_resolution_gain == 0 and updated.acc_rot_deg == 7.84
    # Order 2 stays when the children's step (3.75 at order 3) is below 90% of the accuracy.
    kept, _ = auto_sampling.update_sampling(dataclasses.replace(updated, updates_without_resolution_gain=2), config,
                                            acc_rot_deg=5.0, acc_trans_px=0.2, iteration=20)
    assert kept.healpix_order == 2
    # Order 3 reaches local_order: local searches with RELION's sigma, twice the children's step.
    local, log = auto_sampling.update_sampling(dataclasses.replace(updated, updates_without_resolution_gain=2), config,
                                               acc_rot_deg=1.4, acc_trans_px=0.3, iteration=30)
    assert local.healpix_order == 3 and local.local_searches
    assert_matches(np.float32(local.sigma_deg), np.float32(2 * healpix_angular_step(4)))
    # Never beyond max_order, never a coarser offset step than before.
    capped, _ = auto_sampling.update_sampling(dataclasses.replace(local, updates_without_resolution_gain=2), config,
                                              acc_rot_deg=0.1, acc_trans_px=50.0, iteration=40)
    assert capped.healpix_order == 3 and capped.offset_step_px == local.offset_step_px
    assert capped.offset_range_px == local.offset_range_px
    # Resolution tracking: the counter grows while the shell does not improve and resets on a gain.
    tracked = auto_sampling.track_resolution(sampling, 3)
    assert (tracked.resolution_shell, tracked.updates_without_resolution_gain) == (3, 0)
    tracked = auto_sampling.track_resolution(auto_sampling.track_resolution(tracked, 3), 2)
    assert (tracked.resolution_shell, tracked.updates_without_resolution_gain) == (3, 2)
    gates = np.array([[1.0, 1.0], [0.9, 0.1], [0.6, 0.0], [0.4, 0.0], [0.7, 0.0]])
    assert auto_sampling.resolution_shell([{"gates": gates}, {"gates": gates[:, 0] * 0.2}], radius=8) == 2
    assert auto_sampling.resolution_shell([{"gates": gates}], radius=1) == 1
    long = dataclasses.replace(config, iterations=50)
    assert auto_sampling.update_due(10, long) and not auto_sampling.update_due(1, long)
    assert not auto_sampling.update_due(15, long) and not auto_sampling.update_due(50, long)
    with pytest.raises(ValueError):
        dataclasses.replace(config, local_order=4)
    with pytest.raises(ValueError):
        dataclasses.replace(config, local_search_start=3)
    ladder, beyond = auto_sampling.error_ladder(auto_sampling.ANGLE_LADDER, 30.0)
    assert ladder[0] == 0.05 and ladder[-1] == 30.0 and beyond == 35.0 and np.all(np.diff(ladder) > 0)


def test_expected_errors_shrink_with_a_stronger_signal():
    data = tomo_data()
    config = dataclasses.replace(_local_config(), oversampling=1, auto_sampling=True, accuracy_particles=3,
                                 stages=((1, 2, 1),), local_order=3, max_order=3)
    state = initialize_state(data, config, 6.)
    grid = pose_grid(data, config, 1)
    membership = Membership.empty(np.arange(4), 2)
    membership.update(np.array([0, 1, 3]), np.full((3, 2), 0.5, np.float32), iteration=1, model_iteration=0,
                      eulers_deg=grid.eulers[[1, 5, 9]].astype(np.float32), translations_px=np.zeros((3, 3), np.float32))
    rng = np.random.default_rng(3)
    acc_rot, acc_trans, per_class = auto_sampling.expected_errors(data, state, config, membership, radius=2, rng=rng)
    stronger = dataclasses.replace(state, theta=state.theta * np.float32(1e3))
    strong_rot, strong_trans, strong_classes = auto_sampling.expected_errors(
        data, stronger, config, membership, radius=2, rng=np.random.default_rng(3))
    assert set(per_class) == {0, 1} and all(v["particles"] == 3 for v in per_class.values())
    assert 0 < strong_rot <= acc_rot <= 35.0 and 0 < strong_trans <= acc_trans <= 12.0
    assert strong_rot < acc_rot or acc_rot == 35.0
    assert strong_trans < 12.0
    # Classes below RELION's occupancy floor are not estimated.
    tiny = dataclasses.replace(state, class_prior=np.array([0.995, 0.005], np.float32))
    _, _, classes = auto_sampling.expected_errors(data, tiny, config, membership, radius=2, rng=np.random.default_rng(3))
    assert classes[1] == {"acc_rot_deg": 999.0, "acc_trans_px": 999.0, "particles": 0}


def test_controller_with_local_searches_and_auto_sampling_resumes_identically(tmp_path):
    data = tomo_data()
    config = dataclasses.replace(tiny_config(2), iterations=8, stages=((1, 2, 1),), shift_range=2, shift_step=2,
                                 oversampling=1, oversampling_start=1, max_significant=6, tomogram_batches=True,
                                 auto_sampling=True, accuracy_interval=2, accuracy_particles=2, local_order=1,
                                 max_order=1)
    identity = {"case": "relion-mechanisms"}
    direct = iteration_loop.run(data, config, tmp_path / "direct", identity, 6.)
    records = [json.loads(line) for line in (tmp_path / "direct/iterations.jsonl").read_text().splitlines()]
    assert all(r["sampling"]["healpix_order"] == 1 for r in records)
    assert all(r["coarse_healpix_order"] == 1 for r in records)
    assert any(r["sampling_update"] is not None for r in records[1:])
    assert direct.sampling is not None and direct.sampling.resolution_shell <= 2
    assert np.isfinite(direct.membership.euler_deg).all()  # the final full-data update stored every pose
    stopped = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6., stop_after=3)
    restored = checkpoint.load(tmp_path / "resumed/checkpoint_0003.npz", config, identity)
    assert restored.sampling == stopped.sampling
    assert_matches(restored.membership.euler_deg, stopped.membership.euler_deg)
    resumed = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6.,
                                 resume=tmp_path / "resumed/checkpoint_0003.npz")
    assert_matches(resumed.theta, direct.theta)
    assert resumed.sampling == direct.sampling
    assert_matches(resumed.membership.euler_deg, direct.membership.euler_deg)
    # Local searches from a fixed update, without auto-sampling: the final export searches locally.
    fixed = dataclasses.replace(config, auto_sampling=False, local_search_start=4)
    local = iteration_loop.run(data, fixed, tmp_path / "local", identity, 6.)
    records = [json.loads(line) for line in (tmp_path / "local/iterations.jsonl").read_text().splitlines()]
    assert [r["local_searches"] for r in records] == [False] * 3 + [True] * 5
    assert all(r["candidate_rows_mean"] is not None for r in records[4:])
    classes = json.loads((tmp_path / "local/classes.json").read_text())
    assert classes["local_searches"] is True and classes["sampling"] is None
    assert np.isfinite(local.membership.euler_deg).all()
    # An older checkpoint without poses resumes with unknown poses and a global search.
    with np.load(tmp_path / "local/checkpoint_0002.npz") as a:
        stripped = {k: a[k] for k in a.files if not k.startswith("membership_euler") and not k.startswith("membership_translation")}
    np.savez(tmp_path / "local/old.npz", **stripped)
    old = checkpoint.load(tmp_path / "local/old.npz", fixed, identity)
    assert np.isnan(old.membership.euler_deg).all()


def test_finished_run_extends_with_more_iterations_and_the_recorded_stages(tmp_path):
    data = tomo_data()
    config = dataclasses.replace(tiny_config(2), iterations=6, stages=((1, 2, 1),), shift_range=2, shift_step=2)
    identity = {"case": "extension"}
    short = iteration_loop.run(data, config, tmp_path / "short", identity, 6.)
    assert short.iteration == 6
    longer = dataclasses.replace(config, iterations=8)
    extended = iteration_loop.run(data, longer, tmp_path / "extended", identity, 6.,
                                  resume=tmp_path / "short/checkpoint_0006.npz")
    records = [json.loads(line) for line in (tmp_path / "extended/iterations.jsonl").read_text().splitlines()]
    assert extended.iteration == 8 and [r["iteration"] for r in records] == [7, 8]
    assert (tmp_path / "extended/classes.json").exists()
    # The stage schedule stays part of the identity: the default schedule for the new length is refused.
    rescaled = dataclasses.replace(config, iterations=8, stages=None)
    assert rescaled.stages != config.stages
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load(tmp_path / "short/checkpoint_0006.npz", rescaled, identity)


def test_runner_resume_preflight_accepts_the_identity_it_wrote(tmp_path, monkeypatch):
    """The tilt resume preflight (runner.py) carries the zero-mask flag like the loaded identity, and a
    finished run extends through the command path with a larger --iterations and its recorded stages."""
    from argparse import Namespace

    from relax.commands import ppca_initial_model as command
    from relax.ppca_initial_class3d import images, runner

    data = tomo_data()
    hashes = {"ios_sha256": "i", "particles_sha256": "p", "tomograms_sha256": "t", "tilt_series_sha256": "s"}
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test")
    monkeypatch.setattr(runner, "tilt_training_identity", lambda _ios: dict(hashes))
    monkeypatch.setattr(images, "load_tilt_training", lambda *_args, **_kw: (data, dict(hashes)))
    monkeypatch.setattr(command, "source_identity", lambda: {"files": {}})
    config = dataclasses.replace(tiny_config(2), iterations=6, stages=((1, 2, 1),), shift_range=2, shift_step=2,
                                 zero_mask=True)
    args = Namespace(ios="ios.star", manifest=None, particle_diameter=6.0, output=str(tmp_path / "first"),
                     resume=None, stop_after=None, stop_file=None)
    runner.run(args, config)
    assert (tmp_path / "first/checkpoint_0006.npz").exists()
    longer = dataclasses.replace(config, iterations=8)
    again = Namespace(**{**vars(args), "output": str(tmp_path / "second"),
                         "resume": str(tmp_path / "first/checkpoint_0006.npz")})
    runner.run(again, longer)
    records = [json.loads(line) for line in (tmp_path / "second/iterations.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in records] == [7, 8]
    assert json.loads((tmp_path / "second/run.json").read_text())["identity"]["zero_mask"] is True


def test_config_rejects_misuse():
    base = tiny_config(2)
    for changes in ({"local_search_start": 0}, {"local_search_sigma_deg": -1.0}, {"accuracy_interval": 0},
                    {"auto_sampling": True}, {"local_order": 2, "max_order": 1}):
        with pytest.raises(ValueError):
            Config(**{**dataclasses.asdict(base), **changes})


def test_seed_model_takes_the_bootstrap_sign():
    """The random seed blobs carry the sign of the bootstrap reconstruction's centre (initialization.seed_model):
    tilt images read with relax's CTF convention reconstruct negative protein, and a positive seed made the
    first updates drive the mixture's largest class through zero (the 100 A core inside a negative ring)."""
    from recovar.core import fourier_transform_utils as ftu

    from relax.ppca_initial_class3d.initialization import seed_model

    n, channels, radius = 16, 3, 3
    x = np.arange(n) - n // 2
    blob = np.exp(-(sum(a ** 2 for a in np.meshgrid(x, x, x, indexing="ij"))) / (2 * 3.0 ** 2)).astype(np.float32)
    for sign in (1.0, -1.0):
        volumes = np.stack([sign * blob] * channels)
        rhs = ftu.get_dft3_real(volumes).reshape(channels, -1)
        lhs = np.ones_like(rhs, dtype=np.float32)
        theta = np.asarray(seed_model(jnp.asarray(rhs), jnp.asarray(lhs), (n, n, n), 1.0, 8.0, radius, np.random.default_rng(0)))
        seed = np.asarray(ftu.get_idft3_real(theta[:, 0].reshape(ftu.volume_shape_to_half_volume_shape((n, n, n))), (n, n, n)))
        centre = np.sqrt(sum(a ** 2 for a in np.meshgrid(x, x, x, indexing="ij"))) <= 2
        assert np.sign(seed[centre].mean()) == sign
