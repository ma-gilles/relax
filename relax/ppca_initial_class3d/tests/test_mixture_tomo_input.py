"""Merged-stack addresses belong to the multi-class loader, not the shared reader."""

from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
from helpers.float_compare import assert_matches
from recovar.data_io import starfile
from recovar.data_io.starfile import read_star, star_column
from unit.test_tomo_input import GRID, VOXEL
from unit.test_tomo_input import project as project

from relax.ppca_initial_class3d.images import load_tilt_training
from relax.ppca_initial_class3d.tomo_input import load_tomo_dataset, merged_stack_image_names
from relax.ppca_initial_model.checkpoint import file_hash

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def merged_project(project, tmp_path_factory):
    out, _ = project
    root = tmp_path_factory.mktemp("mixture_merged_tomo")
    particles, optics = read_star(str(out / "particles.star"))
    stacks = list(map(str, star_column(particles, "rlnImageName", required=True)))
    frames, starts = [], []
    for name in stacks:
        with mrcfile.open(out / name, permissive=True) as mrc:
            data = np.asarray(mrc.data, np.float32).reshape(-1, GRID, GRID).copy()
        starts.append(sum(len(frame) for frame in frames) + 1)
        frames.append(data)
    merged = root / "merged.mrcs"
    with mrcfile.new(merged) as mrc:
        mrc.set_data(np.concatenate(frames))
        mrc.voxel_size = VOXEL
    particles["_rlnImageName"] = [f"{start:06d}@{merged}" for start in starts]
    particles_star = root / "particles.star"
    starfile.write_star(str(particles_star), particles, data_optics=optics)
    ios = root / "optimisation_set.star"
    starfile.write_star(str(ios), pd.DataFrame({
        "_rlnTomoParticlesFile": [str(particles_star)],
        "_rlnTomoTomogramsFile": [str(out / "tomograms.star")],
    }))
    return ios, dict(zip(stacks, starts)), merged


def test_merged_stack_names_resolve_one_based_offsets():
    assert merged_stack_image_names(["000001@000020@one.mrcs", "000003@000020@one.mrcs"]) == [
        "000020@one.mrcs", "000022@one.mrcs",
    ]
    assert merged_stack_image_names(["000001@particle1.mrcs", "000003@particle2.mrcs"]) is None


@pytest.mark.parametrize("names, message", [
    (["1@20@all.mrcs", "1@particle.mrcs"], "mixes merged-stack slices"),
    (["0@20@all.mrcs"], "1-based"),
    (["1@0@all.mrcs"], "1-based"),
])
def test_merged_stack_names_refuse_invalid_layouts(names, message):
    with pytest.raises(ValueError, match=message):
        merged_stack_image_names(names)


def test_merged_flat_star_preserves_all_non_image_metadata(project, merged_project, tmp_path):
    out, flat = project
    ios, first_of, merged = merged_project
    from relax.relion.tomo_input import read_optimisation_set

    particles, tomograms = read_optimisation_set(ios)
    output = tmp_path / "flat_merged.star"
    tomo = load_tomo_dataset(particles, tomograms, output, datadir=str(out), lazy=False)
    original, original_optics = read_star(str(flat))
    actual, actual_optics = read_star(str(output))
    expected = []
    for name in map(str, star_column(original, "rlnImageName", required=True)):
        tilt, stack = name.split("@", 1)
        expected.append(f"{first_of[stack] + int(tilt) - 1:06d}@{merged}")
    assert list(map(str, star_column(actual, "rlnImageName", required=True))) == expected
    assert list(original.columns) == list(actual.columns)
    for column in original.columns:
        if column != "_rlnImageName":
            assert_matches(actual[column].to_numpy(), original[column].to_numpy(), err_msg=column)
    for column in original_optics.columns:
        assert_matches(actual_optics[column].to_numpy(), original_optics[column].to_numpy(), err_msg=column)
    assert tomo.n_units == 6
    assert_matches(tomo.voxel_size, VOXEL)


@pytest.mark.parametrize("mask_diameter", [None, 64.0], ids=["unmasked", "zero_masked"])
def test_merged_training_matches_particle_stacks_with_and_without_mask(
    project, merged_project, tmp_path, mask_diameter,
):
    out, _ = project
    merged_ios, _, _ = merged_project
    ios = out / "optimisation_set.star"
    originals = [ios, out / "particles.star", out / "tomograms.star", merged_ios, merged_ios.parent / "particles.star"]
    hashes = {path: file_hash(path) for path in originals}
    native, native_identity = load_tilt_training(ios, tmp_path / "native", zero_mask_diameter_ang=mask_diameter)
    merged, identity = load_tilt_training(merged_ios, tmp_path / "merged", zero_mask_diameter_ang=mask_diameter)
    assert identity["ios_sha256"] == hashes[merged_ios]
    assert identity["particles_sha256"] == hashes[merged_ios.parent / "particles.star"]
    assert identity["tomograms_sha256"] == native_identity["tomograms_sha256"]
    assert identity["tilt_series_sha256"] == native_identity["tilt_series_sha256"]
    assert native.n_images == merged.n_images == 6
    for field in ("image_offsets", "image_frame", "particle_group", "noise_group"):
        assert_matches(getattr(merged, field), getattr(native, field), err_msg=field)
    for actual, expected in zip(merged.group_frames, native.group_frames, strict=True):
        assert_matches(actual, expected)
    # Non-monotonic requests also preserve image identity and exact per-tilt CTFs.
    image_ids = np.arange(native.image_offsets[-1])[::-1].copy()
    for actual, expected in zip(merged.read(image_ids), native.read(image_ids), strict=True):
        assert_matches(actual, expected)
    assert {path: file_hash(path) for path in originals} == hashes


def test_adapter_does_not_replace_shared_loader():
    from relax.ppca_initial_class3d import images
    from relax.refinement import tomo_half
    from relax.relion import tomo_input

    assert tomo_half.load_tomo_dataset.__module__ == "relax.refinement.tomo_half"
    assert tomo_input.flatten_relion5_tomo.__module__ == "relax.relion.tomo_input"
    assert not hasattr(tomo_input, "merged_stack_image_names")
    assert images.load_tilt_training.__module__ == "relax.ppca_initial_class3d.images"
    assert Path(load_tomo_dataset.__code__.co_filename).parent.name == "ppca_initial_class3d"
