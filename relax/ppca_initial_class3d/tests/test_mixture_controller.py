"""Real tiny SPA/TOMO controller runs, restart state, and truthful map outputs."""

import dataclasses
import json
import pickle

import jax.numpy as jnp
import mrcfile
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu
from scipy.spatial.transform import Rotation

from relax.commands.ppca_initial_model import load_training
from relax.ppca_initial_class3d import checkpoint, iteration_loop, runner
from relax.ppca_initial_class3d.config import Config
from relax.ppca_initial_model.checkpoint import file_hash
from relax.ppca_initial_model.tomo import TiltParticles

pytestmark = pytest.mark.unit


def tiny_config(k):
    # The full grid: the single-model parity reference scores it, and these tiny grids are below RELION's cap.
    return Config(n_classes=k, q=1, iterations=2, stages=((1, 2, 0),), shift_range=0, oversampling=0,
                  image_batch_size=2, rotation_block_size=16, gemm_precision="fp32", pass2_mass_floor=0)


def spa_data(root):
    root.mkdir()
    images = np.random.default_rng(41).normal(size=(4, 8, 8)).astype(np.float32)
    with mrcfile.new(root / "particles.mrcs") as mrc:
        mrc.set_data(images)
        mrc.voxel_size = 1
    # Marc's optics contract now requires an explicit STAR source for disk
    # particles. This fixture declares ordinary, non-premultiplied optics.
    (root / "particles.star").write_text(
        "data_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnOpticsGroupName #2\n"
        "_rlnImageSize #3\n_rlnImagePixelSize #4\n_rlnVoltage #5\n"
        "_rlnSphericalAberration #6\n_rlnAmplitudeContrast #7\n_rlnCtfDataAreCtfPremultiplied #8\n"
        "1 opticsGroup1 8 1 300 2.7 0.07 0\n\ndata_particles\n\nloop_\n"
        "_rlnImageName #1\n_rlnOpticsGroup #2\n_rlnDefocusU #3\n_rlnDefocusV #4\n"
        "_rlnDefocusAngle #5\n_rlnPhaseShift #6\n"
        + "".join(f"{i + 1:06d}@{root / 'particles.mrcs'} 1 15000 15000 0 0\n" for i in range(4))
    )
    values = {
        "neutral_poses.pkl": (np.broadcast_to(np.eye(3), (4, 3, 3)).copy(), np.zeros((4, 2))),
        "ctf.pkl": np.tile([8, 1, 15000, 15000, 0, 300, 2.7, 0.07, 0], (4, 1)).astype(np.float32),
    }
    for name, value in values.items():
        with open(root / name, "wb") as stream:
            pickle.dump(value, stream)
    np.save(root / "particle_ids.npy", np.arange(4))
    names = ["particles.star", *values, "particle_ids.npy"]
    manifest = dict(schema="recovar-ppca-training-v1", n_images=4, box=8, voxel_size=1,
                    particle_diameter_ang=6, shift_range_px=0, particles=names[0], neutral_poses=names[1],
                    ctf=names[2], particle_ids=names[3], contrast=1, image_multiplier=1,
                    files={name: file_hash(root / name) for name in names})
    path = root / "training.json"
    path.write_text(json.dumps(manifest))
    return load_training(path, "on")[0]


def tomo_data():
    frames = Rotation.from_euler("y", [-30, 0, 30], degrees=True).as_matrix()
    slots = [[0, 1, 2], [0, 2], [0, 1], [0, 1, 2]]
    offsets = np.r_[0, np.cumsum([len(x) for x in slots])]
    real = np.random.default_rng(41).normal(size=(offsets[-1], 8, 8)).astype(np.float32)
    half = ftu.get_dft2_real(jnp.asarray(real)).reshape(len(real), -1)

    def read(ids):
        return real[ids], half[ids], jnp.ones((len(ids), 40), jnp.float32)

    # Interleaved noise owners reorder physical particles during the final E-step.
    return TiltParticles((8, 8), (8, 8, 8), 1., offsets, np.concatenate(slots),
                         np.zeros(4, dtype=int), (frames,), read, noise_group=np.asarray([1, 0, 1, 0]))


def test_spa_k1_complete_updates_match_single_ppca(tmp_path, monkeypatch):
    from relax.ppca_initial_class3d import state as mixture_state
    from relax.ppca_initial_model import iteration_loop as single
    from relax.ppca_initial_model.config import Config as SingleConfig
    from relax.ppca_initial_model.initialization import initialize

    data, config = spa_data(tmp_path / "inputs"), tiny_config(1)
    legacy = SingleConfig(**{f.name: getattr(config, f.name) for f in dataclasses.fields(SingleConfig)})
    expected = single.run(data, legacy, tmp_path / "single", {"case": "spa"}, 6.)
    # This direct-engine K=1 check isolates update algebra, not mixture-only
    # signed bootstrap. The production K=1 command dispatches upstream outright.
    monkeypatch.setattr(mixture_state, "initialize", initialize)
    actual = iteration_loop.run(data, config, tmp_path / "mixture", {"case": "spa"}, 6.)
    # Same algorithm and arithmetic except mixture reductions; the standard float32 band is used.
    assert_matches(actual.theta[0], expected.theta)
    assert_matches(actual.noise, expected.noise)
    assert_matches(actual.moments[0].first, expected.moments.first)
    assert_matches(actual.moments[0].second, expected.moments.second)
    assert_matches(actual.direction_prior, expected.direction_prior)
    assert_matches(np.float32(actual.offset_variance), np.float32(expected.offset_variance))
    with np.load(tmp_path / "mixture/assignments.npz") as a:
        assert_matches(a["class_probabilities"], np.ones((4, 1), np.float32))
        np.testing.assert_array_equal(a["particle_ids"], np.arange(4))
        assert a["translations_px"].shape == (4, 2)


def test_tomo_k2_restart_and_all_particle_outputs(tmp_path):
    data, config = tomo_data(), tiny_config(2)
    identity = {"case": "two-noise-group-tomo"}
    direct = iteration_loop.run(data, config, tmp_path / "direct", identity, 6.)
    stopped = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6., stop_after=1)
    assert stopped.iteration == 1
    resumed = iteration_loop.run(data, config, tmp_path / "resumed", identity, 6.,
                                 resume=tmp_path / "resumed/checkpoint_0001.npz")
    assert resumed.iteration == direct.iteration == 2
    assert_matches(resumed.theta, direct.theta)
    assert_matches(resumed.noise, direct.noise)
    assert_matches(resumed.class_prior, direct.class_prior)
    assert resumed.rng_state == direct.rng_state
    with np.load(tmp_path / "resumed/assignments.npz") as a:
        np.testing.assert_array_equal(a["particle_ids"], np.arange(4))
        assert a["translations_px"].shape == (4, 3)
        assert_matches(a["class_probabilities"].sum(1), np.ones(4, np.float32))
        assert a["class_conditional_z"].shape == (4, 2, 1)
        assert np.all(np.isfinite(a["class_conditional_z"]))
    half_shape = ftu.volume_shape_to_half_volume_shape(data.volume_shape)
    expected = ftu.get_idft3_real(resumed.theta[0, :, 0].reshape(half_shape), data.volume_shape)
    with mrcfile.open(tmp_path / "resumed/class000_model_mean.mrc") as mrc:
        assert_matches(mrc.data, expected)
        assert_matches(np.float32(mrc.voxel_size.x), np.float32(1))
    assert (tmp_path / "resumed/class000_mean_gridding_corrected.mrc").is_file()
    assert not (tmp_path / "resumed/class000_mean.mrc").exists()


def test_bad_tomography_resume_cannot_overwrite_derived_star(tmp_path, monkeypatch):
    from relax.commands import ppca_initial_model as command

    output = tmp_path / "output"
    output.mkdir()
    derived = output / "particles_2d.star"
    derived.write_text("original derived star\n")
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test")
    monkeypatch.setattr(runner, "tilt_training_identity", lambda _ios: {"particles_sha256": "wrong"})
    monkeypatch.setattr(command, "load_tilt_training", lambda *_args: pytest.fail("conversion before validation"))

    def reject(*_args):
        raise ValueError("identity mismatch")

    monkeypatch.setattr(checkpoint, "load", reject)
    with pytest.raises(ValueError, match="identity mismatch"):
        command.main(["--ios", "wrong.star", "--K", "2", "--q", "1", "--particle-diameter", "6",
                      "-o", str(output), "--resume", "checkpoint.npz"])
    assert derived.read_text() == "original derived star\n"


def test_class_prior_counts_particles_and_smooths_empty_component():
    prior = iteration_loop.update_class_prior(np.array([.5, .5], np.float32), np.array([3., 1.], np.float32),
                                              pseudocount=1, full_data=True)
    assert_matches(prior, np.asarray([4/6, 2/6], np.float32))
    empty = iteration_loop.update_class_prior(prior, np.array([4., 0.], np.float32), pseudocount=1, full_data=True)
    assert_matches(empty, np.asarray([5/6, 1/6], np.float32))
