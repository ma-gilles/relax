"""Training-only fixture loading and actual initial-noise normalization."""

import json
import pickle

import numpy as np
import pytest

from relax.commands.ppca_initial_model import load_training
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
        ori_size=n,
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


def test_training_loader_needs_no_truth_and_checks_inputs(tmp_path):
    import mrcfile

    count, n = 4, 8
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
    data, _, identity = load_training(path)
    assert data.n_images == count
    assert identity == {"manifest_sha256": file_hash(path)}
    assert data.data_multiplier == manifest["image_multiplier"]
    np.testing.assert_allclose(np.asarray(data.get_image_real(0)), images[0], rtol=1e-6, atol=1e-6)
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
