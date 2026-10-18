"""Gradual PCs, latent centering and rigid-motion projection of the mixture (pc_policy.py).

CPU, float32. The opening, PC-test and band-test schedules against their definitions; an opening PC's scale
and orthogonality; the PC test's gains, rule and closed model on synthetic values; its second scoring pass
against the update's own E-step evidence (full grid and two passes); the band test's cut model, the mean solved
alone where the loadings are held at zero, and the held loadings and first moments; the controller for closed
loadings held at exactly zero, openings at their updates, the tests' gains, the per-class caps, band radii and
determinism across a resume; centering for leaving every particle's map unchanged; the latent-moment estimator
against packed normal equations; the projection for removing a loading that is a rigid motion of the mean and
keeping one that is not.
"""

import argparse
import dataclasses
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.triangular import pack_upper_tri
from test_mixture_controller import tiny_config, tomo_data

from relax.ppca_initial_class3d import checkpoint, iteration_loop, pc_policy
from relax.ppca_initial_class3d.config import Config
from relax.ppca_initial_class3d.expectation import expectation_groups, pose_grid
from relax.ppca_initial_model.update import Moments, coupled_direction

pytestmark = pytest.mark.unit


def test_open_schedule_spreads_the_pcs_between_the_two_fractions():
    every = Config(n_classes=2, q=5, iterations=50)
    assert pc_policy.pc_open_updates(every) == (1,) * 5
    assert pc_policy.class_active_pcs(every, 1, [5, 5]) == [5, 5]
    assert pc_policy.pc_test_updates(every) == {}
    early = dataclasses.replace(every, pc_start_fraction=0.3)
    openings = pc_policy.pc_open_updates(early)
    assert openings == (16, 22, 28, 35, 41)
    assert [pc_policy.scheduled_pcs(early, u) for u in (15, 16, 21, 22, 41, 50)] == [0, 1, 1, 2, 5, 5]
    assert pc_policy.pc_test_updates(early) == {21: 1, 27: 2, 34: 3, 40: 4, 46: 5}
    assert pc_policy.class_active_pcs(early, 30, [5, 1]) == [3, 1]
    late = dataclasses.replace(every, pc_start_fraction=0.55)
    assert pc_policy.pc_open_updates(late) == (29, 32, 35, 38, 41)
    with pytest.raises(ValueError):
        dataclasses.replace(every, pc_start_fraction=0.9)


def test_band_tests_close_every_stage_after_the_first():
    config = Config(n_classes=2, q=5, iterations=50, pc_band_test=True)
    assert config.stages == ((1, 4, 1), (16, 8, 2), (29, 16, 2), (41, 32, 2))
    assert pc_policy.pc_band_test_updates(config, 64) == {28: (4, 8), 40: (8, 16), 49: (16, 31)}
    assert pc_policy.pc_band_test_updates(dataclasses.replace(config, pc_band_test=False), 64) == {}
    assert pc_policy.class_pc_radii(None, 2, 16) == [None, None]
    assert pc_policy.class_pc_radii([None, 8], 2, 16) == [None, 8]
    assert pc_policy.class_pc_radii([16, 8], 2, 16) == [None, 8]


def test_an_opening_pc_is_small_and_orthogonal_to_the_class():
    n = 16
    shape = (n, n, n)
    x = np.arange(n) - n // 2
    grid = np.meshgrid(x, x, x, indexing="ij")
    mean = np.exp(-(sum(g ** 2 for g in grid)) / 8).astype(np.float32)
    theta = np.zeros((1, n * n * (n // 2 + 1), 3), np.complex64)
    theta[0, :, 0] = np.asarray(pc_policy._to_spectra(mean[None]))[:, 0]
    config = Config(n_classes=1, q=2, iterations=10, pc_start_fraction=0.2)
    opened = np.asarray(pc_policy.open_pcs(theta, config, previously_active=[0], active=[1], volume_shape=shape,
                                           radius=7, mask=np.ones(shape, np.float32)))
    real = pc_policy._to_real(opened[0], shape)
    assert_matches(np.linalg.norm(real[1]), pc_policy.PC_OPEN_NORM_SHARE * np.linalg.norm(real[0]), rtol=1e-5)
    assert abs(np.sum(real[0] * real[1])) < 1e-4 * np.linalg.norm(real[0]) * np.linalg.norm(real[1])
    assert np.all(opened[0, :, 2] == 0)


def test_controller_holds_closed_pcs_at_zero_and_opens_them_on_schedule(tmp_path):
    data = tomo_data()
    config = dataclasses.replace(tiny_config(2), q=2, iterations=6, stages=((1, 2, 0),), shift_range=2, shift_step=2,
                                 pc_start_fraction=0.34)
    assert pc_policy.pc_open_updates(config) == (3, 6)
    assert pc_policy.pc_test_updates(config) == {5: 1}
    identity = {"case": "gradual-pcs"}
    direct = iteration_loop.run(data, config, tmp_path / "direct", identity, 6.)
    records = [json.loads(line) for line in (tmp_path / "direct/iterations.jsonl").read_text().splitlines()]
    assert [r["active_pcs"] for r in records[:4]] == [[0, 0], [0, 0], [1, 1], [1, 1]]
    tests = records[4]["pc_tests"]
    assert [t["class"] for t in tests] == [0, 1] and all(t["pc"] == 1 for t in tests)
    assert all(np.isfinite(t["gain"]) and t["pc1_gain"] == t["gain"] for t in tests)
    kept = [t["kept"] for t in tests]
    assert kept == [pc_policy.keep_tested_pc(1, t["gain"], t["gain"], t["class_mass"]) for t in tests]
    assert records[4]["active_pcs_after_tests"] == [1 if keep else 0 for keep in kept]
    assert records[5]["active_pcs"] == [2 if keep else 0 for keep in kept]  # a closed class opens nothing more
    for update, record in enumerate(records, start=1):
        with np.load(tmp_path / f"direct/checkpoint_{update:04d}.npz", allow_pickle=False) as saved:
            theta, meta = saved["theta"], json.loads(str(saved["metadata"]))
        norms = np.linalg.norm(theta[..., 1:], axis=1)  # (K, q)
        counts = record.get("active_pcs_after_tests", record["active_pcs"])
        for k, count in enumerate(counts):
            assert np.all(norms[k, count:] == 0) and np.all(norms[k, :count] > 0), (update, k)
        assert meta["pc_caps"] == [2 if keep else 0 for keep in kept] or update < 5
        assert meta["pc_first_gain"] == ([t["gain"] for t in tests] if update >= 5 else None)
    stopped = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6., stop_after=4)
    assert stopped.iteration == 4
    resumed = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6.,
                                 resume=tmp_path / "resumed/checkpoint_0004.npz")
    assert_matches(resumed.theta, direct.theta)


def test_controller_runs_with_centering_and_rigid_projection(tmp_path):
    data = tomo_data()
    config = dataclasses.replace(tiny_config(2), q=2, iterations=6, stages=((1, 2, 0),), shift_range=2, shift_step=2,
                                 center_latents=True, project_rigid_motions=True)
    state = iteration_loop.run(data, config, tmp_path / "run", {"case": "centering"}, 6.)
    assert np.all(np.isfinite(np.asarray(state.theta)))
    records = [json.loads(line) for line in (tmp_path / "run/iterations.jsonl").read_text().splitlines()]
    assert all(r["active_pcs"] == [2, 2] and len(r["latent_centers"]) == 2 for r in records)
    assert all(0 <= f <= 1 for r in records for f in r["rigid_motion_removed_fraction"])


def test_centering_keeps_every_particle_map():
    rng = np.random.default_rng(3)
    theta = (rng.normal(size=(2, 7, 4)) + 1j * rng.normal(size=(2, 7, 4))).astype(np.complex64)
    centers = rng.normal(size=(2, 3)).astype(np.float32)
    centred = np.asarray(pc_policy.center_latents(theta, centers))
    for k in range(2):
        z = rng.normal(size=3).astype(np.float32)
        before = theta[k, :, 0] + theta[k, :, 1:] @ z
        after = centred[k, :, 0] + centred[k, :, 1:] @ (z - centers[k])
        assert_matches(after, before)


def test_evidence_gain_weights_each_particle_by_its_class_probability():
    def half(ids, probabilities, log_likelihood):
        ids = np.asarray(ids)
        return SimpleNamespace(membership_tiles=[(ids, np.asarray(probabilities, np.float32), None, None)],
                               image_log_likelihood=[(ids, np.asarray(log_likelihood))])

    stats = [half([3, 8], [[1.0, 0.0], [0.25, 0.75]], [-10.0, -20.0]), half([5], [[0.0, 1.0]], [-7.0])]
    gains, mass = pc_policy.evidence_gains(stats, {3: -12.0, 8: -19.0, 5: -9.0}, 2)
    # Gains per particle 2, -1, 2: class 0 holds 3 (weight 1) and 8 (0.25); class 1 holds 8 (0.75) and 5 (1).
    assert_matches(gains, [(2 - 0.25) / 1.25, (-0.75 + 2) / 1.75])
    assert_matches(mass, [1.25, 1.75])
    with pytest.raises(KeyError):
        pc_policy.evidence_gains(stats, {3: -12.0, 8: -19.0}, 2)


def test_a_pc_stays_for_a_positive_gain_large_enough_against_pc1():
    keep = pc_policy.keep_tested_pc
    assert keep(1, 0.5, 0.5, 100.0) and not keep(1, -0.1, -0.1, 100.0)
    share = pc_policy.PC_GAIN_SHARE
    assert keep(2, 1.5 * share, 1.0, 100.0) and not keep(2, 0.5 * share, 1.0, 100.0)
    assert not keep(3, -1.0, 1.0, 100.0)
    assert not keep(1, 3.0, 3.0, pc_policy.MIN_CLASS_PARTICLES / 2)


def test_closing_a_tested_pc_keeps_the_map_at_the_class_mean_latent():
    rng = np.random.default_rng(4)
    theta = (rng.normal(size=(2, 7, 3)) + 1j * rng.normal(size=(2, 7, 3))).astype(np.complex64)
    centers = rng.normal(size=(2, 2)).astype(np.float32)
    closed = np.asarray(pc_policy.close_tested_pc(theta, [1], 2, centers))
    assert_matches(closed[0], theta[0])
    assert np.all(closed[1, :, 2] == 0)
    z = centers[1]
    assert_matches(closed[1, :, 0] + closed[1, :, 1] * z[0], theta[1, :, 0] + theta[1, :, 1:] @ z)


@pytest.mark.parametrize("engine", ["full_grid", "two_passes"])
def test_the_closed_pc_pass_scores_the_batch_as_the_update_does(tmp_path, engine):
    data = tomo_data()
    config = dataclasses.replace(tiny_config(2), q=2, shift_range=2, shift_step=2)
    if engine == "two_passes":
        config = dataclasses.replace(config, oversampling=1, oversampling_start=1, max_significant=6)
    state = iteration_loop.run(data, config, tmp_path / "run", {"case": engine}, 6., stop_after=1)
    grid, halves = pose_grid(data, config, 2), [np.array([0, 2]), np.array([1, 3])]
    children = pose_grid(data, config, 2, children=True) if engine == "two_passes" else None
    radius = min(config.stage(2)[0], data.grid_size // 2 - 1)
    stats = expectation_groups(data, state, config, halves, 2, grid, radius=radius, diameter_ang=6., child_grid=children)
    same = iteration_loop.closed_pc_evidence(data, state, config, halves, 2, grid, state.theta, radius=radius,
                                             diameter_ang=6.,
                                             candidates=None)
    assert sorted(same) == [0, 1, 2, 3]
    update = {i: v for half in stats for ids, values in half.image_log_likelihood
              for i, v in zip(ids.tolist(), values.tolist())}
    # float32 evidence, held in float64: the float32 band
    assert_matches([same[i] for i in range(4)], [update[i] for i in range(4)], rtol=1e-6)
    _, mass = pc_policy.evidence_gains(stats, same, 2)
    assert_matches(mass.sum(), 4.0, rtol=1e-6)
    closed = pc_policy.close_tested_pc(state.theta, [0, 1], 2, np.zeros((2, 2), np.float32))
    without = iteration_loop.closed_pc_evidence(data, state, config, halves, 2, grid, closed, radius=radius,
                                                diameter_ang=6.,
                                                candidates=None)
    assert any(abs(without[i] - same[i]) > 1e-3 for i in range(4))


def test_the_band_cut_keeps_the_map_at_the_class_mean_latent():
    rng = np.random.default_rng(6)
    theta = (rng.normal(size=(2, 9, 3)) + 1j * rng.normal(size=(2, 9, 3))).astype(np.complex64)
    radii = np.arange(9, dtype=np.float32) / 2
    centers = rng.normal(size=(2, 2)).astype(np.float32)
    cut = np.asarray(pc_policy.truncate_pc_band(theta, [0], 2.0, radii, centers))
    assert_matches(cut[1], theta[1])
    assert np.all(cut[0, radii > 2, 1:] == 0)
    assert_matches(cut[0, radii <= 2], theta[0, radii <= 2])
    z = centers[0]
    assert_matches(cut[0, :, 0] + cut[0, :, 1:] @ z, theta[0, :, 0] + theta[0, :, 1:] @ z)


def test_the_mean_is_solved_alone_where_the_loadings_are_held():
    rng = np.random.default_rng(7)
    q, frequencies = 2, 6
    factors = rng.normal(size=(frequencies, q + 1, q + 1)).astype(np.float32)
    full = np.einsum("fij,fkj->fik", factors, factors) + np.eye(q + 1, dtype=np.float32)
    gradient = rng.normal(size=(frequencies, q + 1)).astype(np.float32)
    high = np.arange(frequencies) >= 3
    support = np.ones(frequencies, bool)
    coupled, _ = coupled_direction(pack_upper_tri(jnp.asarray(full)), jnp.asarray(gradient), support, floor=1e-6)
    alone, _ = coupled_direction(pc_policy.uncouple_mean(pack_upper_tri(jnp.asarray(full)), q, high),
                                 jnp.asarray(gradient), support, floor=1e-6)
    coupled, alone = np.asarray(coupled), np.asarray(alone)
    assert_matches(alone[~high], coupled[~high], rtol=1e-5)
    assert_matches(alone[high, 0], gradient[high, 0] / full[high, 0, 0], rtol=1e-5)
    assert_matches(alone[high, 1:], np.linalg.solve(full[high, 1:, 1:], gradient[high, 1:][..., None])[..., 0],
                   rtol=1e-5)


def test_held_loadings_and_their_first_moments_stay_zero_beyond_the_radius():
    rng = np.random.default_rng(8)
    theta = (rng.normal(size=(2, 9, 3)) + 1j * rng.normal(size=(2, 9, 3))).astype(np.complex64)
    first = (rng.normal(size=(2, 9, 3)) + 1j * rng.normal(size=(2, 9, 3))).astype(np.complex64)
    history = Moments(jnp.asarray(first), jnp.ones((9, 2), jnp.float32), jnp.ones((2, 9), bool))
    radii = np.arange(9, dtype=np.float32) / 2
    held, moments = pc_policy.limit_pc_band(theta, (history, history), [None, 2], radii)
    held, first_held = np.asarray(held), np.asarray(moments[1].first)
    assert_matches(held[0], theta[0])
    assert_matches(held[1, :, 0], theta[1, :, 0])
    assert np.all(held[1, radii > 2, 1:] == 0) and np.all(first_held[:, radii > 2, 1:] == 0)
    assert_matches(held[1, radii <= 2], theta[1, radii <= 2])
    assert_matches(first_held[:, :, 0], first[:, :, 0])
    assert moments[0] is history


@pytest.mark.parametrize("rule", ["measured", "always_keep"])
def test_controller_tests_the_band_and_holds_cut_loadings(tmp_path, monkeypatch, rule):
    # The tiny classes hold less than MIN_CLASS_PARTICLES, so the measured rule cuts both bands at the first test;
    # a rule that keeps every band reaches the second test.
    if rule == "always_keep":
        monkeypatch.setattr(iteration_loop, "keep_band", lambda gain, mass: True)
    data = tomo_data()
    config = dataclasses.replace(tiny_config(2), q=2, iterations=7, stages=((1, 1, 0), (3, 2, 0), (5, 3, 0)),
                                 shift_range=2, shift_step=2, pc_band_test=True)
    assert pc_policy.pc_band_test_updates(config, data.grid_size) == {4: (1, 2), 6: (2, 3)}
    identity = {"case": "pc-band"}
    direct = iteration_loop.run(data, config, tmp_path / "direct", identity, 6.)
    records = [json.loads(line) for line in (tmp_path / "direct/iterations.jsonl").read_text().splitlines()]
    radii = np.asarray(ftu.get_grid_of_radial_distances_real(data.volume_shape, rounded=False)).reshape(-1)
    caps = [None, None]
    for update, record in enumerate(records, start=1):
        tests = record.get("pc_band_tests", [])
        assert [t["class"] for t in tests] == ([k for k in range(2) if caps[k] is None] if update in (4, 6) else [])
        for t in tests:
            assert t["kept"] == iteration_loop.keep_band(t["gain"], t["class_mass"]) and np.isfinite(t["gain"])
            if not t["kept"]:
                caps[t["class"]] = t["from_radius"]
        assert record["pc_radius"] == [record["radius"] if c is None else c for c in caps]
        with np.load(tmp_path / f"direct/checkpoint_{update:04d}.npz", allow_pickle=False) as saved:
            theta, meta = saved["theta"], json.loads(str(saved["metadata"]))
        assert meta["pc_band_caps"] == caps
        for k, radius in enumerate(record["pc_radius"]):
            assert np.all(theta[k, radii > radius, 1:] == 0)
    assert caps == ([1, 1] if rule == "measured" else [None, None])
    assert [len(r.get("pc_band_tests", [])) for r in records] == [0, 0, 0, 2, 0, 2 if rule == "always_keep" else 0, 0]
    stopped = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6., stop_after=4)
    assert stopped.iteration == 4
    resumed = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6.,
                                 resume=tmp_path / "resumed/checkpoint_0004.npz")
    assert_matches(resumed.theta, direct.theta)


def test_latent_centre_is_the_weighted_mean_latent_of_the_normal_equations():
    rng = np.random.default_rng(5)
    zbar = np.asarray([0.7, -1.2], np.float32)
    weights = rng.uniform(0.5, 2.0, size=6).astype(np.float32)
    full = np.zeros((6, 3, 3), np.float32)
    full[:, 0, 0] = weights
    full[:, 0, 1:] = full[:, 1:, 0] = weights[:, None] * zbar
    full[:, 1:, 1:] = weights[:, None, None] * (np.eye(2) + np.outer(zbar, zbar))
    component = SimpleNamespace(lhs_tri=pack_upper_tri(jnp.asarray(full)))
    stats = [SimpleNamespace(components=[component, component]) for _ in range(2)]
    centers, second = pc_policy.latent_moments(stats, 2, np.ones(6, bool))
    assert_matches(centers, np.stack([zbar, zbar]))
    assert_matches(second[0], np.eye(2) + np.outer(zbar, zbar))


def test_rigid_projection_removes_a_shift_of_the_mean_and_keeps_other_variation():
    n = 16
    shape = (n, n, n)
    x = np.arange(n) - n // 2
    grid = np.meshgrid(x, x, x, indexing="ij")
    def blob(dx=0.0):
        return np.exp(-((grid[0] - 1.5 - dx) ** 2 + (grid[1] + 1) ** 2 / 2 + grid[2] ** 2 / 3) / 4).astype(np.float32)

    mean = blob()
    shifted = blob(0.2) - mean                            # a small translation of the mean, to first order
    other = np.exp(-(sum(g ** 2 for g in grid)) / 6).astype(np.float32) * np.cos(grid[2])  # not a rigid motion
    spectra = np.asarray(pc_policy._to_spectra(np.stack([mean, shifted, other])))
    theta = spectra[None]
    mask = np.ones(shape, np.float32)
    projected, removed = pc_policy.project_rigid_motions(theta, [2], volume_shape=shape, radius=n, mask=mask)
    real = pc_policy._to_real(np.asarray(projected)[0], shape)
    assert np.linalg.norm(real[1]) < 0.1 * np.linalg.norm(shifted)
    basis = pc_policy.rigid_motion_basis(mean, mask)
    other_rigid = np.linalg.norm(basis @ other.reshape(-1)) / np.linalg.norm(other)
    assert_matches(np.linalg.norm(real[2]) / np.linalg.norm(other), np.sqrt(1 - other_rigid ** 2), rtol=1e-3)
    assert 0 < removed[0] < 1


def test_old_checkpoints_compare_equal_with_the_new_settings_at_their_defaults():
    config = dataclasses.asdict(tiny_config(2))
    old = {k: v for k, v in config.items() if k not in checkpoint.ADDED_CONFIG_DEFAULTS}
    assert checkpoint._configuration(old) == checkpoint._configuration(config)
    changed = dict(config, center_latents=True)
    assert checkpoint._configuration(old) != checkpoint._configuration(changed)


def test_command_maps_the_switches_to_the_config(monkeypatch):
    from relax.commands import ppca_initial_model as command
    from relax.ppca_initial_class3d import runner

    captured = []
    monkeypatch.setattr(runner, "run", lambda args, config: captured.append(config))
    parser = argparse.ArgumentParser()
    command.add_args(parser)
    base = ["--ios", "ios.star", "--particle-diameter", "6", "-o", "out", "--K", "2", "--q", "3",
            "--iterations", "50"]
    from relax.ppca_initial_class3d.command import dispatch
    assert dispatch(parser.parse_args(base))
    assert dispatch(parser.parse_args(base + ["--gradual-pcs", "--center-latents", "--project-rigid-motions"]))
    assert dispatch(parser.parse_args(base + ["--gradual-pcs", "--pc-start-fraction", "0.55"]))
    assert dispatch(parser.parse_args(base + ["--gradual-pc-band", "--no-adaptive-radius"]))
    with pytest.raises(ValueError):  # the band test follows the stages' radii
        dispatch(parser.parse_args(base + ["--gradual-pc-band"]))
    plain, gradual, late, band = captured
    assert plain.pc_start_fraction is None and not plain.center_latents and not plain.project_rigid_motions
    assert not plain.pc_band_test and band.pc_band_test and not gradual.pc_band_test
    assert gradual.pc_start_fraction == 0.3 and gradual.center_latents and gradual.project_rigid_motions
    assert late.pc_start_fraction == 0.55
