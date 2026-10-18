"""The manifest-to-STAR converter: the retired recovar-ppca-training-v1 data, in RELION's sign, read as relax reads it."""

import json
import pickle

import numpy as np
import pytest

from relax.commands.ppca_initial_model import load_training
from relax.ppca_initial_model.checkpoint import file_hash
from scripts.prepare_vdam_ppca_star import convert

pytestmark = pytest.mark.unit


def _write_manifest(root, count, n, ids=None):
    """A recovar-ppca-training-v1 manifest: bare .mrcs, cryoDRGN CTF pickle, neutral poses, particle ids."""
    import mrcfile

    root.mkdir()
    rng = np.random.default_rng(19)
    images = rng.normal(size=(count, n, n)).astype(np.float32)
    with mrcfile.new(root / "particles.mrcs") as stack:
        stack.set_data(images)
        stack.voxel_size = 2
    poses = (np.broadcast_to(np.eye(3), (count, 3, 3)).copy(), np.zeros((count, 2)))
    ctf = np.tile([n, 2, 15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1)).astype(np.float32)
    ctf[:, 2] += rng.uniform(-3000, 3000, count).astype(np.float32)
    ctf[:, 3] += rng.uniform(-3000, 3000, count).astype(np.float32)
    ctf[:, 4] = rng.uniform(0, 180, count).astype(np.float32)
    ctf[:, 8] = rng.uniform(0, 30, count).astype(np.float32)
    for name, value in [("neutral_poses.pkl", poses), ("ctf.pkl", ctf)]:
        with open(root / name, "wb") as stream:
            pickle.dump(value, stream)
    np.save(root / "particle_ids.npy", np.arange(count) if ids is None else ids)
    names = ["particles.mrcs", "ctf.pkl", "neutral_poses.pkl", "particle_ids.npy"]
    manifest = dict(
        schema="recovar-ppca-training-v1", n_images=count, box=n, voxel_size=2, particle_diameter_ang=14,
        shift_range_px=6, particles=names[0], ctf=names[1], neutral_poses=names[2], particle_ids=names[3],
        contrast=1, image_multiplier=1, files={name: file_hash(root / name) for name in names},
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root / "manifest.json", images


def _manifest_dataset(manifest_path):
    """The retired load_training's read of a manifest (as stored, RECOVAR sign, CTF from the pickle): the reference."""
    from recovar.data_io.cryoem_dataset import load_dataset

    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    return load_dataset(
        str(root / manifest["particles"]),
        poses_file=str(root / manifest["neutral_poses"]),
        ctf_file=str(root / manifest["ctf"]),
        uninvert_data=False,
        dtype=np.complex64,
        lazy=False,
    )


def test_the_converted_star_loads_the_negated_images_and_the_same_ctf(tmp_path):
    count, n = 6, 8
    manifest_path, _ = _write_manifest(tmp_path / "manifest", count, n)
    record = convert(manifest_path, tmp_path / "star")
    assert record["particle_diameter_ang"] == 14 and record["source_manifest_sha256"] == file_hash(manifest_path)
    reference = _manifest_dataset(manifest_path)
    converted, identity, _ = load_training(tmp_path / "star/particles.star", "on")
    assert identity["particles_star_sha256"] == record["files"]["particles.star"]
    assert converted.n_images == count and converted.grid_size == n and converted.voxel_size == reference.voxel_size
    np.testing.assert_array_equal(np.asarray(converted.CTF_params), np.asarray(reference.CTF_params))
    np.testing.assert_array_equal(np.asarray(converted.get_CTF(np.arange(count))), np.asarray(reference.get_CTF(np.arange(count))))
    for i in range(count):
        np.testing.assert_array_equal(np.asarray(converted.get_image(i)), -np.asarray(reference.get_image(i)))


def test_star_rows_follow_the_particle_ids(tmp_path):
    """Row r of the STAR is the particle whose id is r (its image negated, its CTF)."""
    import mrcfile

    count, n = 5, 8
    ids = np.array([3, 0, 4, 1, 2])
    manifest_path, images = _write_manifest(tmp_path / "manifest", count, n, ids)
    convert(manifest_path, tmp_path / "star")
    with mrcfile.open(tmp_path / "star/particles.mrcs") as stack:
        np.testing.assert_array_equal(stack.data, -images[np.argsort(ids)])
    with open(manifest_path.parent / "ctf.pkl", "rb") as stream:
        defocus_u = pickle.load(stream)[:, 2]
    data, _, _ = load_training(tmp_path / "star/particles.star")
    np.testing.assert_array_equal(np.asarray(data.CTF_params)[:, 0], defocus_u[np.argsort(ids)])


def test_the_converter_refuses_to_write_beside_its_input(tmp_path):
    manifest_path, _ = _write_manifest(tmp_path / "manifest", 2, 8)
    with pytest.raises(ValueError, match="outside"):
        convert(manifest_path, manifest_path.parent / "star")
