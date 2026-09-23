"""Frozen-mask bookkeeping and masked-FSC parsing (scripts/masked_fsc.py)."""

import argparse
import json

import numpy as np
import pytest

from scripts import masked_fsc


def _mrc(path, data, voxel=1.5):
    masked_fsc.write_mrc_raw(path, data, voxel)
    return path


def _mask_record(tmp_path, dataset="toy_c1", reproduced=True, passed=True):
    mask = np.zeros((16, 16, 16), dtype=np.float32)
    mask[4:12, 4:12, 4:12] = 1.0
    mask_path = _mrc(tmp_path / "mask.mrc", mask)
    record = {
        "dataset": dataset,
        "symmetry": "C1",
        "box": 16,
        "pixel_size_A": 1.5,
        "source_role": "ground truth",
        "generation": {"kind": "recipe"},
        "mask": {"file": "mask.mrc", "sha256": masked_fsc.sha256_file(mask_path)},
        "sanity": {"passed": passed, "summary": "PASS"},
        "reproducibility": {"reproduced": reproduced},
    }
    path = tmp_path / "MASK.json"
    path.write_text(json.dumps(record))
    return path, record


@pytest.mark.unit
def test_recipe_threshold_follows_written_rule():
    lowpass = np.zeros((10, 10, 10), dtype=np.float32)
    lowpass[5, 5, 5] = 1.0
    lowpass.flat[:2] = -0.5
    record = masked_fsc.recipe_threshold(lowpass)
    median = float(np.median(lowpass))
    level = float(np.percentile(lowpass, 99.99))
    assert record["initial_threshold"] == pytest.approx(median + 0.05 * (level - median))


@pytest.mark.unit
def test_recipe_threshold_rejects_map_without_positive_density():
    with pytest.raises(ValueError, match="no positive density"):
        masked_fsc.recipe_threshold(np.zeros((4, 4, 4), dtype=np.float32))


@pytest.mark.unit
def test_recipe_pixels_reproduce_realdata_widths_and_scale():
    for pixel in (1.31, 1.345, 1.4000112):
        assert masked_fsc.recipe_pixels(pixel) == (5, 8)
    assert masked_fsc.recipe_pixels(2.125) == (3, 5)
    assert masked_fsc.recipe_pixels(4.25) == (2, 3)


@pytest.mark.unit
def test_canonical_label_makes_file_hash_independent_of_timestamp(tmp_path):
    data = np.ones((8, 8, 8), dtype=np.float32)
    paths = [_mrc(tmp_path / f"m{i}.mrc", data) for i in range(2)]
    for i, path in enumerate(paths):
        raw = bytearray(path.read_bytes())
        raw[masked_fsc.LABEL_OFFSET : masked_fsc.LABEL_OFFSET + 30] = f"Relion 5.0.1   0{i}-Sep-26 12:00".encode()
        path.write_bytes(bytes(raw))
    assert masked_fsc.sha256_file(paths[0]) != masked_fsc.sha256_file(paths[1])
    assert masked_fsc.header_without_label_sha256(paths[0]) == masked_fsc.header_without_label_sha256(paths[1])
    assert masked_fsc.payload_sha256(paths[0]) == masked_fsc.payload_sha256(paths[1])
    old = masked_fsc.canonicalize_label(paths[0])
    masked_fsc.canonicalize_label(paths[1])
    assert old.startswith("Relion 5.0.1")
    assert masked_fsc.sha256_file(paths[0]) == masked_fsc.sha256_file(paths[1])


@pytest.mark.unit
def test_scoring_refuses_dataset_without_frozen_mask(tmp_path):
    mask_json, record = _mask_record(tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"masks": {"toy_c1": {"mask_json": str(mask_json), "mask_sha256": record["mask"]["sha256"]}}})
    )
    path, _ = masked_fsc.load_frozen_mask("toy_c1", registry)
    assert path == tmp_path / "mask.mrc"
    with pytest.raises(KeyError, match="no frozen mask"):
        masked_fsc.load_frozen_mask("other_c1", registry)


@pytest.mark.unit
def test_scoring_refuses_a_changed_mask_file(tmp_path):
    mask_json, record = _mask_record(tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"masks": {"toy_c1": {"mask_json": str(mask_json), "mask_sha256": record["mask"]["sha256"]}}})
    )
    _mrc(tmp_path / "mask.mrc", np.zeros((16, 16, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="does not match"):
        masked_fsc.load_frozen_mask("toy_c1", registry)


@pytest.mark.unit
def test_register_requires_verification_and_never_replaces_a_mask(tmp_path):
    registry = tmp_path / "registry.json"
    unverified, _ = _mask_record(tmp_path, reproduced=False)
    with pytest.raises(SystemExit, match="verify-mask"):
        masked_fsc.cmd_register(argparse.Namespace(registry=registry, mask_json=[str(unverified)]))
    mask_json, _ = _mask_record(tmp_path)
    masked_fsc.cmd_register(argparse.Namespace(registry=registry, mask_json=[str(mask_json)]))
    assert "toy_c1" in json.loads(registry.read_text())["masks"]
    other = tmp_path / "other"
    other.mkdir()
    changed, _ = _mask_record(other)
    _mrc(other / "mask.mrc", np.full((16, 16, 16), 0.5, dtype=np.float32))
    record = json.loads(changed.read_text())
    record["mask"]["sha256"] = masked_fsc.sha256_file(other / "mask.mrc")
    changed.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="different frozen mask"):
        masked_fsc.cmd_register(argparse.Namespace(registry=registry, mask_json=[str(changed)]))


@pytest.mark.unit
def test_read_postprocess_star_blocks(tmp_path):
    star = tmp_path / "postprocess.star"
    star.write_text(
        "\n# version 50001\n\ndata_general\n\n_rlnFinalResolution    5.309211\n_rlnBfactorUsedForSharpening 0.0\n\n"
        "# version 50001\n\ndata_fsc\n\nloop_ \n_rlnSpectralIndex #1 \n_rlnFourierShellCorrelationCorrected #2 \n"
        "0 1.0 \n1 0.9 \n2 0.1 \n\n"
    )
    general = masked_fsc.read_star_table(star, "general")
    fsc = masked_fsc.read_star_table(star, "fsc")
    assert float(general["rlnFinalResolution"]) == pytest.approx(5.309211)
    np.testing.assert_allclose(fsc["rlnFourierShellCorrelationCorrected"], [1.0, 0.9, 0.1])


@pytest.mark.unit
def test_sustained_crossing_and_band_auc():
    curve = np.array([1.0, 0.9, 0.1, 0.5, 0.1, 0.1, 0.1, 0.0])
    assert masked_fsc.first_sustained_below(curve, 1 / 7, 3) == 4
    assert masked_fsc.first_sustained_below(curve, 1 / 7, 1) == 2
    assert masked_fsc.first_sustained_below(np.ones(6), 1 / 7, 3) is None
    assert masked_fsc.band_auc(curve, 1, 3) == pytest.approx((0.9 / 2 + 0.1 + 0.5 / 2) / 2)
    with pytest.raises(ValueError):
        masked_fsc.band_auc(curve, 2, 2)


@pytest.mark.unit
def test_frame_correlation_detects_sign_and_mismatch():
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(12, 12, 12))
    mask = np.ones_like(volume)
    assert masked_fsc.frame_correlation(volume, volume, mask) == pytest.approx(1.0)
    assert masked_fsc.frame_correlation(volume, -volume, mask) == pytest.approx(-1.0)
    assert masked_fsc.frame_correlation(volume, rng.normal(size=volume.shape), mask) < masked_fsc.FRAME_CORRELATION_MIN


@pytest.mark.unit
def test_registry_entries_are_verified_and_hashed():
    registry = json.loads(masked_fsc.DEFAULT_REGISTRY.read_text())
    assert registry["schema"] == "relax-frozen-mask-registry-v1"
    assert registry["masks"], "registry is empty"
    for dataset, entry in registry["masks"].items():
        assert len(entry["mask_sha256"]) == 64, dataset
        assert entry["mask_json"].endswith("/MASK.json"), dataset
        assert entry["sanity"].startswith("PASS"), dataset
