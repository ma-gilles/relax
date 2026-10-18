"""Particle STAR loading for single-particle PPCA and actual initial-noise normalization."""

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


def _write_training_star(root, count, n, images=None):
    """A RELION particle STAR of the kind ``relax ppca_initial_model --i`` reads: a stack and one optics group."""
    import mrcfile

    from scripts.lib.ppca_training_star import write_training_star

    root.mkdir(parents=True, exist_ok=True)
    if images is None:
        images = np.random.default_rng(19).normal(size=(count, n, n)).astype(np.float32)
    with mrcfile.new(root / "particles.mrcs", overwrite=True) as stack:
        stack.set_data(images)
        stack.voxel_size = 2
    ctf = np.tile([15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1))
    ctf[:, 0] += 100 * np.arange(count)
    return write_training_star(root / "particles.star", "particles.mrcs", ctf, voxel_size=2, box=n), images


def test_training_loader_reads_the_star_as_stored(tmp_path):
    """Images as stored (no inversion) in STAR row order, CTF from the STAR; the identity is the STAR and its stack,
    not the way the images are read."""
    from relax.commands.ppca_initial_model import training_identity

    count = 4
    path, images = _write_training_star(tmp_path, count, 8)
    data, identity, reading = load_training(path)
    assert data.n_images == count and data.grid_size == 8 and data.voxel_size == 2
    assert identity == {
        "particles_star_sha256": file_hash(path),
        "particle_stacks_sha256": training_identity(path, [tmp_path / "particles.mrcs"])["particle_stacks_sha256"],
    }
    assert reading["setting"] == "auto"
    np.testing.assert_allclose(np.asarray(data.get_image_real(0)), images[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(data.CTF_params)[:, 0], 15000 + 100 * np.arange(count))
    for setting in ("on", "off"):
        data, read_identity, reading = load_training(path, setting)
        assert read_identity == identity and reading["preread"] == (setting == "on")
        np.testing.assert_allclose(np.asarray(data.get_image_real(3)), images[3], rtol=1e-6, atol=1e-6)
    # An edited stack, under the same STAR, is another input.
    _write_training_star(tmp_path, count, 8, images[::-1].copy())
    _, edited, _ = load_training(path)
    assert edited["particles_star_sha256"] == identity["particles_star_sha256"]
    assert edited["particle_stacks_sha256"] != identity["particle_stacks_sha256"]


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


def test_one_expectation_step_runs_from_a_particle_star(tmp_path):
    """The RELION image preprocessing (odd aberration demodulation) reads the STAR's optics table; one expectation
    step runs on the loaded particles."""
    import jax.numpy as jnp

    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.state import State
    from relax.relion.ctf import dataset_optics_source_star

    count, n = 6, 8
    path, _ = _write_training_star(tmp_path, count, n)
    data, _, _ = load_training(path)
    assert dataset_optics_source_star(data) == path
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
    path, _ = _write_training_star(tmp_path, count, n)
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


def test_negated_images_seed_the_negated_model(tmp_path):
    """The seed's blob fields take the bootstrap's magnitude and its sign: a stack and its negation (a simulator
    stack and its RELION-sign STAR copy) start from exactly negated models, with the same noise."""
    from relax.ppca_initial_model.initialization import initialize

    count, n = 64, 16
    z, y = np.indices((n, n)) - n // 2
    projection = (3.0 * np.exp(-(z**2 + y**2) / 6.0)).astype(np.float32)
    noise = np.random.default_rng(4).normal(size=(count, n, n)).astype(np.float32)
    starts = []
    for sign, folder in ((1, "white"), (-1, "dark")):
        star, _ = _write_training_star(tmp_path / folder, count, n, images=sign * (projection[None] + 0.1 * noise))
        data, _, _ = load_training(star)
        theta, noise_spectrum, _ = initialize(data, seed=3, diameter_ang=24.0, q=2)
        starts.append((np.asarray(theta), np.asarray(noise_spectrum)))
    (theta_white, noise_white), (theta_dark, noise_dark) = starts
    np.testing.assert_array_equal(theta_dark, -theta_white)
    np.testing.assert_array_equal(noise_dark, noise_white)
