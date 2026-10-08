"""Class3D start-up references take RELION's model pixel size from their MRC headers."""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import startup_references

pytestmark = pytest.mark.unit


def _write_map(path, volume, pixel_size):
    import mrcfile

    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(volume, dtype=np.float32))
        handle.header.cella = (np.float32(pixel_size * volume.shape[0]),) * 3
    return str(path)


def test_class_references_are_low_passed_in_the_header_pixel_size(tmp_path):
    # RELION filters Iref with radius = ori_size * mymodel.pixel_size / ini_high (ml_optimiser.cpp:3563), and
    # mymodel.pixel_size is the reference header's value for every K (ml_model.cpp:899-918), not the particle
    # STAR's (often a rounded serialisation, 1.416667 for 544/384).
    import logging

    from relax.relion import relion_metadata

    rng = np.random.default_rng(5)
    shape = (16, 16, 16)
    paths = [_write_map(tmp_path / f"ref{k}.mrc", rng.standard_normal(shape), 544 / 384) for k in range(2)]
    header = relion_metadata._read_relion_mrc_model_pixel_size(paths[0])
    refs = startup_references.load_class_references(
        paths, volume_shape=shape, ini_high=10.0, real_for_projector=False, real_dtype=np.float32,
        complex_dtype=np.complex64, log=logging.getLogger(__name__),
    )
    from recovar.utils.helpers import load_relion_volume

    for path, got in zip(paths, refs.class_references_real):
        vol = np.asarray(load_relion_volume(path)).astype(np.float32)
        assert_matches(got, startup_references._initial_lowpass_real(vol, shape, header, 10.0))


def test_class_references_with_different_header_pixel_sizes_are_refused(tmp_path):
    # MlModel::initialiseFromImages: "different models have different pixel sizes in their headers!"
    import logging

    shape = (16, 16, 16)
    paths = [_write_map(tmp_path / "a.mrc", np.zeros(shape), 1.0), _write_map(tmp_path / "b.mrc", np.zeros(shape), 1.01)]
    with pytest.raises(SystemExit, match="different pixel sizes"):
        startup_references.load_class_references(
            paths, volume_shape=shape, ini_high=None, real_for_projector=False, real_dtype=np.float32,
            complex_dtype=np.complex64, log=logging.getLogger(__name__),
        )
