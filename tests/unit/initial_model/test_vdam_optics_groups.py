"""InitialModel (VDAM) with several optics groups on one image grid: start-up, noise, optics, accuracy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from relax.helpers.expected_accuracy import ExpectedAccuracy
from relax.relion import initial_model_io
from relax.relion.initial_noise import relion_startup_positions
from relax.vdam import bootstrap_reconstruction as br
from relax.vdam import estep_setup, native_sampling
from relax.vdam.bootstrap_iref import initialise_denovo_state
from relax.vdam.estep_common import estep_sums
from relax.vdam.model_update import update_noise_from_estep
from relax.vdam.ports import NoProbe


def test_startup_positions_fill_each_group_then_stop():
    # ml_optimiser.cpp:3072-3366: a full group's particles are skipped, the loop stops once all are full.
    groups = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1])
    assert relion_startup_positions(groups, np.ones(9), 2).tolist() == [0, 1, 2, 5]
    # Images count against the quota: one 3-image particle fills a group of quota 3.
    assert relion_startup_positions(groups, np.full(9, 3), 3).tolist() == [0, 2]
    # A group that never fills keeps the loop to the end, still skipping the full group.
    assert relion_startup_positions(np.array([0, 0, 0, 1]), np.ones(4), 2).tolist() == [0, 1, 3]


def test_bootstrap_class_is_the_position_modulo_k():
    rng = np.random.default_rng(5)
    size, n = 16, 12
    images = rng.standard_normal((n, size, size))
    positions = np.array([0, 3, 4, 7, 8, 11, 13, 14, 17, 20, 21, 22])
    common = dict(
        ctf_images=None,
        box_size=size,
        pixel_size=2.0,
        particle_diameter_ang=0.7 * size * 2.0,
        width_mask_edge_px=3.0,
        do_zero_mask=True,
        random_seed=9,
        padding_factor=1,
    )
    refs, _ = br.bootstrap_references(
        images=images, nr_classes=2, minimum_nr_particles=n, particle_positions=positions, **common
    )
    for k in range(2):
        mine = positions % 2 == k
        alone, _ = br.bootstrap_references(
            images=images[mine],
            nr_classes=1,
            minimum_nr_particles=int(mine.sum()),
            particle_seed_ids=positions[mine],
            **common,
        )
        np.testing.assert_allclose(refs[k], alone[0], rtol=0.0, atol=1e-12 * np.max(np.abs(alone[0])))


def test_particle_optics_follow_each_particles_group():
    main = pd.DataFrame({"_rlnOpticsGroup": [2, 1, 2]})
    optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": [1, 2],
            "_rlnVoltage": [300.0, 200.0],
            "_rlnSphericalAberration": [2.7, 1.4],
            "_rlnAmplitudeContrast": [0.1, 0.07],
            "_rlnImagePixelSize": [4.25, 4.25],
            "_rlnImageSize": [128, 128],
        }
    )

    class _Dataset:
        voxel_size = 4.25

    particle_optics = initial_model_io._particle_optics(main, optics, _Dataset())
    np.testing.assert_allclose(particle_optics.voltage, [200.0, 300.0, 200.0], rtol=1e-12)
    np.testing.assert_allclose(particle_optics.Cs, [1.4, 2.7, 1.4], rtol=1e-12)
    np.testing.assert_allclose(particle_optics.Q0, [0.07, 0.1, 0.07], rtol=1e-12)
    other_grid = optics.assign(_rlnImagePixelSize=[4.25, 5.44])
    with pytest.raises(ValueError, match="one dataset per image shape"):
        initial_model_io._particle_optics(main, other_grid, _Dataset())


def test_noise_rows_per_group():
    spectra = np.abs(np.random.default_rng(1).standard_normal((2, 9))) + 0.1
    rows = estep_setup.noise_variance_from_sigma2(spectra, 16)
    assert rows.shape == (2, 256)
    np.testing.assert_allclose(rows[1], estep_setup.noise_variance_from_sigma2(spectra[1:], 16), rtol=1e-12)
    assert estep_setup.noise_variance_from_sigma2(spectra[:1], 16).shape == (256,)


def _state(n_groups, size=16):
    state = initialise_denovo_state(
        box_size=size, pixel_size=2.0, K=1, nr_iter=10, n_directions=48, nr_optics_groups=n_groups, pseudo_halfsets=True
    )
    state.sigma2_noise = np.full((n_groups, size // 2 + 1), 2.0)
    state.subset_size = 100
    return state


def test_noise_update_per_group_and_empty_group_keeps_its_spectrum():
    # maximizationOtherParameters (ml_optimiser.cpp:6316-6372): each group normalised on its own sums;
    # a group without noise sums is not touched.
    rng = np.random.default_rng(2)
    shells = 9
    wsum = np.abs(rng.standard_normal((3, shells))) * 1e6
    power = np.abs(rng.standard_normal((3, shells))) * 1e6
    wsum[2] = 0.0
    power[2] = 0.0
    sumw = np.array([40.0, 25.0, 0.0])
    meta = dict(wsum_sigma2_noise=wsum, wsum_img_power=power, noise_sumw=sumw)
    updated = update_noise_from_estep(_state(3), estep_sums(meta), do_grad=True, mu=0.9)
    for g in range(2):
        alone = update_noise_from_estep(
            _state(1),
            estep_sums(dict(wsum_sigma2_noise=wsum[g], wsum_img_power=power[g], noise_sumw=float(sumw[g]))),
            do_grad=True,
            mu=0.9,
        )
        np.testing.assert_allclose(updated.sigma2_noise[g], alone.sigma2_noise[0], rtol=1e-15)
    np.testing.assert_allclose(updated.sigma2_noise[2], 2.0, rtol=1e-15)


def test_noise_floors_follow_the_running_average():
    # RELION blends first, then copies the previous shell into values below 1e-14 (ml_optimiser.cpp:5255-5282):
    # a shell without data (beyond the Nyquist of a group on another grid) decays by mu.
    shells = 9
    rng = np.random.default_rng(3)
    wsum = np.abs(rng.standard_normal(shells)) * 1e6
    power = np.abs(rng.standard_normal(shells)) * 1e6
    wsum[6:] = 0.0
    power[6:] = 0.0
    state = _state(1)
    state.sigma2_noise = np.full((1, shells), 1e-10)
    updated = update_noise_from_estep(
        state, estep_sums(dict(wsum_sigma2_noise=wsum, wsum_img_power=power, noise_sumw=10.0)), do_grad=True, mu=0.9
    )
    np.testing.assert_allclose(updated.sigma2_noise[0, 6:], 0.9e-10, rtol=1e-15)
    state.sigma2_noise = np.full((1, shells), 1e-14)
    updated = update_noise_from_estep(
        state, estep_sums(dict(wsum_sigma2_noise=wsum, wsum_img_power=power, noise_sumw=10.0)), do_grad=True, mu=0.9
    )
    # 0.9e-14 is below 1e-14: those shells take shell 5's updated value.
    np.testing.assert_allclose(updated.sigma2_noise[0, 6:], updated.sigma2_noise[0, 5], rtol=1e-15)


class _Particles:
    def __init__(self, n):
        self.translation_offsets = np.zeros((n, 2))
        self.class_assignments = np.zeros(n, dtype=np.int32)
        self.best_pose_eulers_deg = np.zeros((n, 3))
        self.best_pose_eulers_valid = np.ones(n, dtype=bool)
        self.best_pose_rotations = None
        self.best_pose_rotation_ids = None


def test_expected_accuracy_runs_once_per_group_with_its_noise_and_optics(monkeypatch):
    n = 8
    groups = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    state = _state(2)
    state.sigma2_noise = np.stack([np.full(9, 1.0), np.full(9, 3.0)])
    state.Iref = np.zeros((1, 16, 16, 16))
    state.pdf_class = np.ones(1)
    optics = initial_model_io.NativeOpticsState(
        voltage=np.where(groups == 0, 300.0, 200.0),
        Cs=np.where(groups == 0, 2.7, 1.4),
        Q0=np.where(groups == 0, 0.1, 0.07),
        pixel_size=2.0,
        defU=np.zeros(n),
        defV=np.zeros(n),
        defAngle=np.zeros(n),
        phase_shift=np.zeros(n),
    )
    calls = []

    def fake_estimator(**kwargs):
        trials = np.asarray(kwargs["trial_local_indices"])
        calls.append((trials.tolist(), float(kwargs["sigma2_noise_relion"][0]), kwargs["voltage"]))
        rot = 1.0 if kwargs["voltage"] == 300.0 else 3.0
        return ExpectedAccuracy(
            acc_rot=rot,
            acc_trans_angstrom=rot,
            acc_rot_per_class=np.array([rot]),
            acc_trans_per_class_angstrom=np.array([rot]),
            class_counts=np.array([trials.size]),
            trial_local_indices=trials,
            trial_particle_ids=trials,
            trial_rot_per_class=np.full((1, trials.size), rot),
            trial_trans_per_class_angstrom=np.full((1, trials.size), rot),
        )

    monkeypatch.setattr(native_sampling, "estimate_relion_expected_accuracy_from_prepared_inputs", fake_estimator)
    sampling_state = replace(
        native_sampling.initial_sampling_state(
            native_sampling.NativeInitialModelOptions(fn_img="x", outputname="y"), pixel_size=2.0
        )
    )
    estimate = native_sampling.estimate_native_sampling_accuracy(
        sampling_state,
        state,
        _Particles(n),
        optics,
        particle_order=np.arange(n),
        random_seed=1,
        padding_factor=1,
        sigma2_fudge=1.0,
        optics_group_ids=groups,
        probe=NoProbe(),
    )
    meta = estimate.meta()
    assert calls == [
        ([0, 3, 5, 6], pytest.approx(1.0, rel=1e-12), pytest.approx(300.0, rel=1e-12)),
        ([1, 2, 4, 7], pytest.approx(3.0, rel=1e-12), pytest.approx(200.0, rel=1e-12)),
    ]
    # Trial-count-weighted class means: (4 * 1 + 4 * 3) / 8.
    assert meta["estimated_acc_rot"] == pytest.approx(2.0)
    assert meta["estimated_acc_class_counts"].tolist() == [8]
