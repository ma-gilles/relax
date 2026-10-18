"""Training-only fixture loading and actual initial-noise normalization."""

import json
import pickle

import numpy as np
import pytest

from relax.commands.ppca_initial_model import PREREAD_MEMORY_FRACTION, load_training, resolve_preread_images
from relax.ppca_initial_model.checkpoint import file_hash
from relax.ppca_initial_model.noise import relion_to_coefficient_variance
from relax.relion.initial_noise import compute_avg_unaligned_and_sigma2

pytestmark = pytest.mark.unit


def test_noise_only_initial_estimator():
    # 4096 independent images: DC has the largest relative sampling uncertainty,
    # sqrt(2/4096). A fixed six-standard-error bound covers every shell without
    # tuning against the realization. The sample-mean subtraction costs 1/M.
    n, count, variance = 8, 4096, 0.125
    images = np.random.default_rng(93).normal(0, np.sqrt(variance), (count, n, n)).astype(np.float32)
    _, sigma = compute_avg_unaligned_and_sigma2(
        ((0, image) for image in images),
        box_size=n,
        pixel_size=1,
        particle_diameter_ang=n,
        width_mask_edge_px=0,
        do_zero_mask=False,
        nr_optics_groups=1,
        minimum_nr_particles=count,
    )
    expected = n * n * variance * (1 - 1 / count)
    np.testing.assert_allclose(
        relion_to_coefficient_variance(sigma[0], (n, n)), expected, rtol=6 * np.sqrt(2 / count), atol=0
    )


def _write_bare_mrcs_manifest(tmp_path, count, n):
    """A training manifest of the documented bare-.mrcs form: an image stack, a cryoDRGN CTF pickle, neutral poses."""
    import mrcfile

    images = np.random.default_rng(19).normal(size=(count, n, n)).astype(np.float32)
    with mrcfile.new(tmp_path / "particles.mrcs") as stack:
        stack.set_data(images)
        stack.voxel_size = 2
    poses = (np.broadcast_to(np.eye(3), (count, 3, 3)).copy(), np.zeros((count, 2)))
    ctf = np.tile([n, 2, 15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1)).astype(np.float32)
    for name, value in [("neutral_poses.pkl", poses), ("ctf.pkl", ctf)]:
        with open(tmp_path / name, "wb") as stream:
            pickle.dump(value, stream)
    np.save(tmp_path / "particle_ids.npy", np.arange(count))
    names = ["particles.mrcs", "ctf.pkl", "neutral_poses.pkl", "particle_ids.npy"]
    manifest = dict(
        schema="recovar-ppca-training-v1",
        n_images=count,
        box=n,
        voxel_size=2,
        particle_diameter_ang=14,
        shift_range_px=6,
        particles=names[0],
        ctf=names[1],
        neutral_poses=names[2],
        particle_ids=names[3],
        contrast=1,
        image_multiplier=1,
        files={name: file_hash(tmp_path / name) for name in names},
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, images, names


def test_training_loader_needs_no_truth_and_checks_inputs(tmp_path):
    count = 4
    path, manifest, images, names = _write_bare_mrcs_manifest(tmp_path, count, 8)
    data, _, identity = load_training(path)
    assert data.n_images == count
    assert identity == {"manifest_sha256": file_hash(path)}
    assert data.data_multiplier == manifest["image_multiplier"]
    np.testing.assert_allclose(np.asarray(data.get_image_real(0)), images[0], rtol=1e-6, atol=1e-6)
    # Image reading is not identity: a preread and a per-tile stack give the same images.
    for setting in ("on", "off"):
        data, _, read_identity = load_training(path, setting)
        assert read_identity == identity
        np.testing.assert_allclose(np.asarray(data.get_image_real(3)), images[3], rtol=1e-6, atol=1e-6)
    for change, message in [
        ({"truth": "absent.npz"}, "Unexpected"),
        ({"files": {}}, "recorded identity"),
        ({"voxel_size": 3}, "pixel size"),
    ]:
        path.write_text(json.dumps({**manifest, **change}))
        with pytest.raises(ValueError, match=message):
            load_training(path)
    np.save(tmp_path / names[3], np.arange(count)[::-1])
    manifest["files"][names[3]] = file_hash(tmp_path / names[3])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stable original"):
        load_training(path)


def test_preread_decision():
    """auto prereads a stack of at most a quarter of the memory; on/off override; the record says why."""
    gb = 10**9
    assert resolve_preread_images("auto", 2 * gb, 64 * gb)[0]
    assert resolve_preread_images("auto", int(PREREAD_MEMORY_FRACTION * 64 * gb), 64 * gb)[0]
    assert not resolve_preread_images("auto", 17 * gb, 64 * gb)[0]
    assert resolve_preread_images("on", 40 * gb, 64 * gb)[0]
    assert not resolve_preread_images("off", 1, 64 * gb)[0]
    _, record = resolve_preread_images("auto", 17 * gb, 64 * gb)
    assert record == {"setting": "auto", "preread": False, "stack_gb": 17.0, "memory_gb": 64.0, "memory_fraction": 0.25}
    with pytest.raises(ValueError, match="auto, on or off"):
        resolve_preread_images("yes", 1, 1)


def test_source_identity_without_git_checkout(monkeypatch):
    """A plain file snapshot (no git) still has an identity: its file hashes, with no head."""
    import subprocess

    from relax.commands import ppca_initial_model as command

    def no_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(command.subprocess, "check_output", no_git)
    identity = command.source_identity()
    assert identity["head"] is None
    assert "relax/ppca_initial_model/iteration_loop.py" in identity["files"]


def test_source_identity_covers_the_shared_modules_ppca_imports():
    """The code hash covers the relax modules PPCA imports, also lazily, not only its own packages, and lists
    them in one order in every call."""
    from relax.commands import ppca_initial_model as command

    first = list(command.source_identity()["files"])
    assert first == list(command.source_identity()["files"])
    # initialization.py and tomo.py import the ini-high shell; engine.py imports the adjoint inside a function.
    assert "relax/reconstruction/resolution.py" in first
    assert "relax/projection/adjoint.py" in first
    assert "relax/relion/project.py" in first
    assert "relax/refinement/iteration_loop.py" not in first


def test_one_expectation_step_runs_from_a_bare_mrcs_manifest(tmp_path):
    """A bare .mrcs with a pickled CTF declares no RELION optics table, so the RELION image preprocessing (odd
    aberration demodulation) leaves its images alone instead of refusing them (regression of 364dba79)."""
    import jax.numpy as jnp

    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.state import State
    from relax.relion.ctf import dataset_optics_source_star

    count, n = 6, 8
    path, _, _, _ = _write_bare_mrcs_manifest(tmp_path, count, n)
    data, _, _ = load_training(path)
    assert dataset_optics_source_star(data) is None
    n_freq = n * n * (n // 2 + 1)
    theta = jnp.asarray(np.random.default_rng(3).normal(size=(n_freq, 3)) * 1e-2, jnp.complex64)
    state = State(
        theta, iteration_loop.empty_moments(theta), jnp.ones(n // 2 + 1, jnp.float32), 0, np.arange(count),
        np.random.default_rng(1).bit_generator.state, 2.0, 2, {"seed": 1},
    )
    config = Config(stages=((1, 2, 0),), oversampling=0, image_batch_size=3, rotation_block_size=64)
    stats = iteration_loop.expectation(data, state, config, np.arange(count), 1)
    assert stats.n_images == count
    assert np.isfinite(float(stats.log_likelihood))


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "host_mask"])
def test_final_embedding_pass_reports_each_images_scores(tmp_path, stream):
    """The embedding pass returns, per image in particle order, its log-evidence (log-sum-exp of the pose
    log-scores), its best pose log-score and that pose's posterior probability."""
    import jax.numpy as jnp

    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.state import State

    count, n = 6, 8
    path, _, _, _ = _write_bare_mrcs_manifest(tmp_path, count, n)
    data, _, _ = load_training(path)
    n_freq = n * n * (n // 2 + 1)
    theta = jnp.asarray(np.random.default_rng(3).normal(size=(n_freq, 3)) * 1e-2, jnp.complex64)
    state = State(
        theta, iteration_loop.empty_moments(theta), jnp.ones(n // 2 + 1, jnp.float32), 0, np.arange(count),
        np.random.default_rng(1).bit_generator.state, 2.0, 2, {"seed": 1},
    )
    config = Config(
        stages=((1, 2, 0),), oversampling=0, image_batch_size=3, rotation_block_size=64, stream_coarse_recompute=stream
    )
    final = iteration_loop.expectation(data, state, config, np.arange(count), 1, embeddings_only=True)
    assert final.embeddings.shape == (count, 2)
    for name in ("log_evidence", "best_pose_log_score", "max_posterior", "null_log_likelihood"):
        values = getattr(final, name)
        assert values.shape == (count,) and values.dtype == np.float32 and np.all(np.isfinite(values)), name
    # log-sum-exp over the poses is at least the largest pose score, and the best pose's posterior is in (0, 1].
    assert np.all(final.log_evidence >= final.best_pose_log_score - 1e-3 * np.abs(final.best_pose_log_score))
    assert np.all((final.max_posterior > 0) & (final.max_posterior <= 1))
    # "No particle" scores the whitened image alone, -|y|^2 / 2: never positive, and in both absolute scores.
    assert np.all(final.null_log_likelihood <= 0)
    np.testing.assert_allclose(
        final.max_posterior, np.exp(final.best_pose_log_score.astype(np.float64) - final.log_evidence), rtol=1e-3
    )
