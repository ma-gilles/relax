"""RELION particle reading modes: default streaming, --scratch_dir and --preread_images.

The three modes must serve the same images and CTF rows bit for bit; only where
the bytes come from differs.
"""

import argparse
import os
import signal

import mrcfile
import numpy as np
import pandas as pd
import pytest
from recovar.data_io import staging
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.data_io.starfile import write_star

from relax.helpers import particle_io
from relax.helpers.particle_io import (
    ParticleReadPolicy,
    add_particle_read_arguments,
    assert_reads_from_scratch,
    prepare_particle_reads,
)

pytestmark = pytest.mark.unit

N_PER_STACK = 12
D = 16


@pytest.fixture(autouse=True)
def _isolated_process_state(monkeypatch):
    """prepare_particle_reads sets RECOVAR_CACHE_DIR and may install a SIGTERM handler."""

    monkeypatch.delenv("RECOVAR_CACHE_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_PREREAD_IMAGES", raising=False)
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


def _fixture(tmp_path):
    """Two stacks referenced by relative paths, rows in a non-physical order."""

    rng = np.random.default_rng(7)
    (tmp_path / "Extract").mkdir()
    stacks = {}
    for name in ("mic1.mrcs", "mic2.mrcs"):
        data = rng.standard_normal((N_PER_STACK, D, D)).astype(np.float32)
        with mrcfile.new(tmp_path / "Extract" / name, overwrite=True) as mrc:
            mrc.set_data(data)
        stacks[f"Extract/{name}"] = data
    rows = [(stack, i) for stack in stacks for i in range(N_PER_STACK)]
    order = rng.permutation(len(rows))
    rows = [rows[k] for k in order]
    n = len(rows)
    data = pd.DataFrame(
        {
            "_rlnImageName": [f"{i + 1:06d}@{stack}" for stack, i in rows],
            "_rlnOpticsGroup": ["1"] * n,
            "_rlnMicrographName": [stack for stack, _ in rows],
            "_rlnAngleRot": rng.uniform(-180, 180, n),
            "_rlnAngleTilt": rng.uniform(0, 180, n),
            "_rlnAnglePsi": rng.uniform(-180, 180, n),
            "_rlnOriginXAngst": rng.uniform(-3, 3, n),
            "_rlnOriginYAngst": rng.uniform(-3, 3, n),
            "_rlnDefocusU": rng.uniform(9000, 20000, n),
            "_rlnDefocusV": rng.uniform(9000, 20000, n),
            "_rlnDefocusAngle": rng.uniform(0, 180, n),
        }
    )
    optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": ["1"],
            "_rlnImagePixelSize": [1.5],
            "_rlnImageSize": [D],
            "_rlnVoltage": [300.0],
            "_rlnSphericalAberration": [2.7],
            "_rlnAmplitudeContrast": [0.1],
        }
    )
    star = tmp_path / "particles.star"
    write_star(str(star), data, optics)
    expected = np.stack([stacks[stack][i] for stack, i in rows])
    return str(star), expected


def _load(star, policy):
    scratch = prepare_particle_reads(star, policy)
    ds = load_dataset(star, lazy=not policy.preread_images, absent_angles_zero=True)
    assert_reads_from_scratch(ds, scratch)
    return ds, scratch


def _reads(ds):
    """Everything refinement reads: whole-set, scattered and half-set rows, and CTF rows."""

    n = ds.n_units
    scattered = np.array([n - 1, 3, 0, 17, 5, 5, 11], dtype=np.int32)
    half = ds.subset(np.arange(1, n, 2))
    return {
        "all": ds.image_source.host_images(np.arange(n)),
        "scattered": ds.image_source.host_images(scattered),
        "half": half.image_source.host_images(np.arange(half.n_units)),
        "ctf": np.asarray(ds.CTF_params),
        "half_ctf": np.asarray(half.CTF_params),
        "rotations": np.asarray(ds.rotation_matrices),
        "translations": np.asarray(ds.translations),
    }


def test_default_scratch_and_preread_read_identical_bytes(tmp_path):
    star, expected = _fixture(tmp_path)
    scratch_root = tmp_path / "local"
    scratch_root.mkdir()

    results = {}
    for mode, policy in {
        "default": ParticleReadPolicy(),
        "scratch": ParticleReadPolicy(scratch_dir=str(scratch_root)),
        "preread": ParticleReadPolicy(preread_images=True),
    }.items():
        ds, scratch = _load(star, policy)
        loader = ds.image_source.backend.source
        assert (loader._cached is not None) == (mode == "preread")
        if mode == "scratch":
            assert scratch is not None and len(scratch.staged) == 2
            assert all(path.startswith(scratch.directory) for path in loader.stack_files())
        else:
            assert scratch is None
            assert all(path.startswith(str(tmp_path / "Extract")) for path in loader.stack_files())
        results[mode] = _reads(ds)
        if scratch is not None:
            scratch.cleanup()
            assert not os.path.exists(scratch.directory)

    np.testing.assert_array_equal(results["default"]["all"], expected)
    for mode in ("scratch", "preread"):
        for key, value in results["default"].items():
            other = results[mode][key]
            assert other.dtype == value.dtype, (mode, key)
            assert np.array_equal(other, value), (mode, key)
    assert os.listdir(scratch_root) == []


def test_scratch_copy_serves_every_read_after_the_sources_move(tmp_path):
    star, expected = _fixture(tmp_path)
    ds, scratch = _load(star, ParticleReadPolicy(scratch_dir=str(tmp_path)))
    os.rename(tmp_path / "Extract", tmp_path / "Extract_moved")
    try:
        np.testing.assert_array_equal(ds.image_source.host_images(np.arange(ds.n_units)), expected)
    finally:
        scratch.cleanup()


def test_scratch_too_small_fails_before_copying(tmp_path, monkeypatch):
    star, _ = _fixture(tmp_path)
    scratch_root = tmp_path / "local"
    scratch_root.mkdir()

    class _Usage:
        free = 5 * 1024**3

    monkeypatch.setattr(staging.shutil, "disk_usage", lambda path: _Usage)
    with pytest.raises(staging.StagingSpaceError, match="kept free"):
        prepare_particle_reads(star, ParticleReadPolicy(scratch_dir=str(scratch_root), keep_free_scratch_gb=10.0))
    assert os.listdir(scratch_root) == []
    assert os.environ.get("RECOVAR_CACHE_DIR") is None


def test_missing_scratch_dir_is_an_error(tmp_path):
    star, _ = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="--scratch_dir"):
        prepare_particle_reads(star, ParticleReadPolicy(scratch_dir=str(tmp_path / "absent")))


def test_preread_ignores_scratch_dir_as_relion_does(tmp_path):
    star, _ = _fixture(tmp_path)
    scratch_root = tmp_path / "local"
    scratch_root.mkdir()
    assert prepare_particle_reads(star, ParticleReadPolicy(preread_images=True, scratch_dir=str(scratch_root))) is None
    assert os.listdir(scratch_root) == []


def test_default_turns_off_recovar_implicit_tmpdir_staging(tmp_path, monkeypatch):
    star, _ = _fixture(tmp_path)
    local_tmp = tmp_path / "node_tmp"
    local_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(local_tmp))
    monkeypatch.setattr(staging, "filesystem_type", lambda path: "xfs")
    assert staging.get_cache_dir() == str(local_tmp)

    ds, scratch = _load(star, ParticleReadPolicy())
    assert scratch is None
    assert os.environ["RECOVAR_CACHE_DIR"] == ""
    assert not (local_tmp / "recovar_cache").exists()
    assert all(path.startswith(str(tmp_path / "Extract")) for path in ds.image_source.backend.source.stack_files())


def test_scratch_installs_sigterm_exit_so_cleanup_runs(tmp_path):
    star, _ = _fixture(tmp_path)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    scratch = prepare_particle_reads(star, ParticleReadPolicy(scratch_dir=str(tmp_path)))
    try:
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as raised:
            handler(signal.SIGTERM, None)
        assert raised.value.code == 128 + signal.SIGTERM
    finally:
        scratch.cleanup()


def test_arguments_default_to_the_relion_gui():
    parser = argparse.ArgumentParser()
    add_particle_read_arguments(parser)
    assert ParticleReadPolicy.from_args(parser.parse_args([])) == ParticleReadPolicy()
    assert ParticleReadPolicy() == ParticleReadPolicy(
        preread_images=False, scratch_dir="", keep_free_scratch_gb=particle_io.DEFAULT_KEEP_FREE_SCRATCH_GB
    )
    parsed = parser.parse_args(["--scratch_dir", "/tmp", "--keep_free_scratch", "2", "--preread_images"])
    assert ParticleReadPolicy.from_args(parsed) == ParticleReadPolicy(
        preread_images=True, scratch_dir="/tmp", keep_free_scratch_gb=2.0
    )
