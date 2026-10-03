"""RELION particle reading modes: default streaming, --scratch_dir and --preread_images.

The three modes must serve the same images and CTF rows bit for bit; only where
the bytes come from differs.
"""

import argparse
import os
import signal
from types import SimpleNamespace

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
from helpers.float_compare import assert_matches, matches
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
from relax.refinement import particle_loading

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


def _loading_args(root, **overrides):
    values = dict(
        data_dir=str(root), output=str(root), preread_images=False,
        scratch_dir="", keep_free_scratch_gb=0.0, particle_diameter_ang=24.0,
        width_mask_edge_px=5.0, relion_softmask_reduction="control",
        image_fourier_backend="host_numpy", n_classes=1,
        relion_init_dir=None, init_noise_from_npz=None, perturb_replay_relion_dir=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("mode", ["streaming", "scratch", "preread"])
@pytest.mark.parametrize("double_preprocessing", [False, True])
def test_command_particle_loading_preserves_images_identity_and_precision(
    tmp_path, monkeypatch, mode, double_preprocessing,
):
    from recovar.data_io import cryoem_dataset

    _, expected = _fixture(tmp_path)
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    monkeypatch.setenv("RELAX_USE_FLOAT64_SCORING", "1" if double_preprocessing else "0")
    calls = []

    def recorded_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return load_dataset(*args, **kwargs)

    monkeypatch.setattr(cryoem_dataset, "load_dataset", recorded_load)
    loaded = particle_loading.load_particle_inputs(_loading_args(
        tmp_path, preread_images=mode == "preread",
        scratch_dir=str(scratch_dir) if mode == "scratch" else "",
    ))
    dataset = loaded.dataset
    assert not loaded.tomographic and loaded.shape_class_rows is None
    assert loaded.double_preprocessing is double_preprocessing
    assert_matches(loaded.mask_parameters, (24.0, 5.0))
    assert len(calls) == 1
    assert calls[0]["dtype"] is (np.complex128 if double_preprocessing else np.complex64)
    assert calls[0]["lazy"] is (mode != "preread")
    assert calls[0]["absent_angles_zero"] is True
    assert_matches(dataset.image_source.host_images(np.arange(dataset.n_units)), expected)
    rows = np.array([17, 2, 5, 0], dtype=np.int64)
    assert_matches(dataset.subset(rows).image_source.host_images(np.arange(rows.size)), expected[rows])


def test_command_particle_loading_keeps_shape_rows_and_group_mask_geometry(tmp_path, monkeypatch):
    _, expected = _fixture(tmp_path)
    monkeypatch.setenv("RELAX_USE_FLOAT64_SCORING", "0")
    star_path = tmp_path / "particles.star"
    tables = starfile.read(star_path)
    second_group = tables["particles"]["rlnImageName"].str.contains("mic2.mrcs")
    tables["particles"].loc[second_group, "rlnOpticsGroup"] = 2
    second_optics = tables["optics"].copy()
    second_optics["rlnOpticsGroup"] = 2
    second_optics["rlnImageSize"] = 12
    second_optics["rlnImagePixelSize"] = 2.0
    tables["optics"] = pd.concat([tables["optics"], second_optics], ignore_index=True)
    second_stack = tmp_path / "Extract" / "mic2.mrcs"
    with mrcfile.open(second_stack) as mrc:
        cropped = np.array(mrc.data[:, 2:-2, 2:-2], copy=True)
    with mrcfile.new(second_stack, overwrite=True) as mrc:
        mrc.set_data(cropped)
    starfile.write(tables, star_path, overwrite=True)

    loaded = particle_loading.load_particle_inputs(_loading_args(tmp_path))
    dataset = loaded.dataset
    assert not loaded.tomographic and len(loaded.shape_class_rows) == 2
    assert dataset.n_units == expected.shape[0]
    for index, (box, pixel_size) in enumerate([(16, 1.5), (12, 2.0)]):
        rows = np.flatnonzero(np.asarray(second_group) == bool(index))
        assert np.array_equal(loaded.shape_class_rows[index], rows)
        images = expected[rows] if index == 0 else expected[rows, 2:-2, 2:-2]
        shape_dataset = dataset.datasets[index]
        assert shape_dataset.image_shape == (box, box)
        assert_matches(shape_dataset.voxel_size, pixel_size)
        assert_matches(shape_dataset.image_source.host_images(np.arange(rows.size)), images)
        assert shape_dataset.image_source.backend.image_mask.shape == (box, box)
        # Both grids span 24 A; their masks must use their own physical pixel size.
        backend = shape_dataset.image_source.backend
        corner_radius = np.sqrt(2) * (box // 2)
        mask_radius = 24.0 / (2 * pixel_size)
        expected_corner = 0.5 + 0.5 * np.cos(np.pi * (corner_radius - mask_radius) / 5.0)
        assert_matches(backend.image_mask[0, 0], expected_corner, rtol=1e-6)
        assert backend.image_mask_mode == "relion_background_fill"
    subset_rows = np.array([17, 2, 5, 0], dtype=np.int64)
    half = dataset.subset(subset_rows)
    assert half.n_units == subset_rows.size
    for shape_class in half.classes:
        source_rows = subset_rows[shape_class.image_indices]
        images = expected[source_rows]
        if shape_class.dataset.image_shape == (12, 12):
            images = images[:, 2:-2, 2:-2]
        assert_matches(shape_class.dataset.image_source.host_images(np.arange(source_rows.size)), images)


@pytest.mark.parametrize("format_name", ["subtomogram", "multiple_shapes"])
@pytest.mark.parametrize("unsupported", ["float64", "frozen", "loaded_noise", "initial_state"])
def test_command_refuses_unsupported_input_state_before_loading(
    tmp_path, monkeypatch, format_name, unsupported,
):
    from recovar.data_io import cryoem_dataset

    monkeypatch.setenv("RELAX_USE_FLOAT64_SCORING", "1" if unsupported == "float64" else "0")
    monkeypatch.setattr(particle_loading, "is_relion5_2d_stack_star", lambda _: format_name == "subtomogram")
    monkeypatch.setattr(particle_loading, "optics_shape_class_rows", lambda _: [np.array([0]), np.array([1])])

    def unexpected_load(*args, **kwargs):
        raise AssertionError("unsupported input reached image loading")

    monkeypatch.setattr(cryoem_dataset, "load_dataset", unexpected_load)
    monkeypatch.setattr(particle_loading, "load_tomo_dataset", unexpected_load)
    args = _loading_args(
        tmp_path, init_noise_from_npz="noise.npz" if unsupported == "loaded_noise" else None,
        relion_init_dir="seeded" if unsupported == "initial_state" else None,
    )
    with pytest.raises(SystemExit, match="subtomogram particles|optics groups on several image shapes"):
        particle_loading.load_particle_inputs(args, frozen_boundary=object() if unsupported == "frozen" else None)


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

    assert_matches(results["default"]["all"], expected)
    for mode in ("scratch", "preread"):
        for key, value in results["default"].items():
            other = results[mode][key]
            assert other.dtype == value.dtype, (mode, key)
            assert matches(other, value), (mode, key)
    assert os.listdir(scratch_root) == []


def test_scratch_copy_serves_every_read_after_the_sources_move(tmp_path):
    star, expected = _fixture(tmp_path)
    ds, scratch = _load(star, ParticleReadPolicy(scratch_dir=str(tmp_path)))
    os.rename(tmp_path / "Extract", tmp_path / "Extract_moved")
    try:
        assert_matches(ds.image_source.host_images(np.arange(ds.n_units)), expected)
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
