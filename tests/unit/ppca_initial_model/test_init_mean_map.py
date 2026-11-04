"""The mean map a run starts from (--init-mean-map): the mean column it sets, the run's use of it, its refusals."""

import json

import jax.numpy as jnp
import mrcfile
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu

from relax.io.map_io import write_map
from relax.ppca_initial_model.initialization import support_mask, with_mean_volume

pytestmark = pytest.mark.unit

BOX = 16
SHAPE = (BOX,) * 3


def _blob(center: tuple[float, float, float]) -> np.ndarray:
    """A float32 Gaussian blob of ``BOX`` pixels per side, off the box center so an axis swap or a flip shows."""
    z, y, x = np.indices(SHAPE).astype(np.float32)
    return np.exp(-((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2) / 4.0).astype(np.float32)


def _theta(q: int) -> jnp.ndarray:
    """Random half-Fourier ``(freq, 1 + q)`` columns standing in for the seed model."""
    freq = BOX * BOX * (BOX // 2 + 1)
    rng = np.random.default_rng(5)
    return jnp.asarray(rng.normal(size=(freq, 1 + q)) + 1j * rng.normal(size=(freq, 1 + q)), jnp.complex64)


def _write_star(root, count):
    """A RELION particle STAR of ``count`` noise images of 8 pixels with one optics group."""
    from scripts.lib.ppca_training_star import write_training_star

    root.mkdir(parents=True, exist_ok=True)
    images = np.random.default_rng(19).normal(size=(count, 8, 8)).astype(np.float32)
    with mrcfile.new(root / "particles.mrcs", overwrite=True) as stack:
        stack.set_data(images)
        stack.voxel_size = 2
    ctf = np.tile([15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1))
    ctf[:, 0] += 100 * np.arange(count)
    return write_training_star(root / "particles.star", "particles.mrcs", ctf, voxel_size=2, box=8)


def _run_command(star, output, *extra):
    """``relax ppca_initial_model`` on ``star`` for 2 iterations of a fixed batch of 8 particles."""
    from relax.commands import ppca_initial_model

    argv = [
        "--i", str(star), "--particle-diameter", "12", "-o", str(output), "--q", "1", "--seed", "11",
        "--iterations", "2", "--stages", "[[1,2,1]]", "--stochastic-batch-size", "8",
        "--stochastic-all-iterations", "--image-batch-size", "4", "--rotation-block-size", "64",
        "--skip-final-embeddings", *extra,
    ]  # fmt: skip
    ppca_initial_model.main(argv)


def test_the_mean_column_is_the_masked_map_and_the_loadings_are_untouched():
    """Inside the kept radius column 0 is the spectrum of the map times the support, outside it is zero; the other
    columns stay."""
    volume = _blob((9.0, 7.0, 8.0))
    mask = support_mask(BOX, 14.0)
    theta = _theta(2)
    out = with_mean_volume(theta, volume, volume_shape=SHAPE, mask=mask)
    spectrum = np.asarray(ftu.get_dft3_real((volume * np.asarray(mask))[None])).reshape(-1)
    kept = np.asarray(ftu.get_grid_of_radial_distances_real(SHAPE, rounded=False)).reshape(-1) <= BOX // 2
    assert 0 < kept.sum() < kept.size
    assert_matches(np.asarray(out[:, 0])[kept], spectrum[kept])
    assert not np.any(np.asarray(out[:, 0])[~kept])
    assert_matches(np.asarray(out[:, 1:]), np.asarray(theta[:, 1:]))
    assert out.dtype == theta.dtype


def test_a_map_of_another_box_is_refused():
    """The map must have the particles' box: a mismatch is an error, not a resampling."""
    with pytest.raises(ValueError, match="box"):
        with_mean_volume(_theta(1), np.zeros((8, 8, 8), np.float32), volume_shape=SHAPE, mask=support_mask(BOX, 14.0))


def test_a_run_starts_from_the_map_and_records_it(monkeypatch, tmp_path):
    """checkpoint_0000 holds the map as its mean and the seed's own loading; run.json and the checkpoint identity
    name the map; the same run without the option starts from the seed's mean."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    volume = _blob((5.0, 3.0, 4.0))[:8, :8, :8].copy()
    path = tmp_path / "mean.mrc"
    write_map(path, volume, voxel_size=2)
    seeded, started = tmp_path / "seeded", tmp_path / "started"
    _run_command(star, seeded)
    _run_command(star, started, "--init-mean-map", str(path))

    with (
        np.load(seeded / "checkpoint_0000.npz", allow_pickle=False) as seed,
        np.load(started / "checkpoint_0000.npz", allow_pickle=False) as init,
    ):
        expected = with_mean_volume(
            jnp.asarray(seed["theta"]), volume, volume_shape=(8, 8, 8), mask=support_mask(8, 12 / 2)
        )
        assert_matches(np.asarray(init["theta"][:, 0]), np.asarray(expected[:, 0]))
        assert_matches(np.asarray(init["theta"][:, 1:]), np.asarray(seed["theta"][:, 1:]))
        assert np.max(np.abs(np.asarray(init["theta"][:, 0]) - np.asarray(seed["theta"][:, 0]))) > 0
        identity = json.loads(str(init["metadata"]))["identity"]
    run = json.loads((started / "run.json").read_text())
    assert run["config"]["init_mean_map"] == str(path)
    assert len(identity["init_mean_map_sha256"]) == 64
    assert "init_mean_map_sha256" not in json.loads((seeded / "run.json").read_text())["identity"]


def test_a_map_without_a_convention_label_is_refused(monkeypatch, tmp_path):
    """A map that is not known to be in RELION's convention would start the run from the negated map."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    path = tmp_path / "plain.mrc"
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.zeros((8, 8, 8), np.float32))
    with pytest.raises(ValueError, match="sign convention is unknown"):
        _run_command(star, tmp_path / "run", "--init-mean-map", str(path))
