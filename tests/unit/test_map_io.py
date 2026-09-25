"""relax map files carry RELION's convention (relax.helpers.map_io).

The fixture cases use the fast tier's K1 5k/128 data and its RELION oversampling-1 oracle: a
volume held in relax's internal frame is written back as the RELION map it came from, and the
reference relax now reads (``reference_init_relion.mrc``, RELION's ``--ref``) gives the same
internal volume as the RECOVAR-frame reference relax read before.
"""

from __future__ import annotations

import mrcfile
import numpy as np
import pytest
from helpers.em_fixtures import fixture_file
from helpers.float_compare import assert_matches
from helpers.map_sign import SIGN_CORRELATION_MIN, file_correlation
from recovar.utils import helpers

from relax.helpers import map_io

pytestmark = pytest.mark.unit


def _raw(path) -> np.ndarray:
    with mrcfile.open(str(path), permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float64).copy()


def _volume(n: int = 8) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n, n, n)).astype(np.float32)


def test_write_map_writes_relion_convention_with_relion_header(tmp_path):
    volume = _volume()
    path = tmp_path / "map.mrc"
    map_io.write_map(path, volume, voxel_size=1.5)
    reference = tmp_path / "relion.mrc"
    helpers.write_relion_mrc(str(reference), volume, voxel_size=1.5)

    assert_matches(_raw(path), _raw(reference))
    assert_matches(_raw(path), -np.transpose(volume, (2, 1, 0)).astype(np.float64))
    with mrcfile.open(str(path)) as handle:
        header = handle.header
        assert int(header.mode) == 2
        assert (int(header.mapc), int(header.mapr), int(header.maps)) == (1, 2, 3)
        assert int(header.ispg) == 0
        assert (int(header.nxstart), int(header.nystart), int(header.nzstart)) == (0, 0, 0)
        assert (float(header.origin.x), float(header.origin.y), float(header.origin.z)) == (0.0, 0.0, 0.0)
        assert float(handle.voxel_size.x) == pytest.approx(1.5)
    assert map_io.map_labels(path) == [map_io.RELAX_MAP_LABEL]
    assert map_io.is_relax_map(path)


def test_load_relax_map_round_trips_the_internal_frame(tmp_path):
    volume = _volume()
    path = tmp_path / "map.mrc"
    map_io.write_map(path, volume)
    assert_matches(map_io.load_relax_map(path), volume)
    assert_matches(helpers.load_relion_volume(str(path)), volume)


def test_write_map_from_ft_matches_real_space_write(tmp_path):
    import jax.numpy as jnp
    from recovar.core import fourier_transform_utils as ftu

    volume = _volume()
    volume_ft = np.asarray(ftu.get_dft3(jnp.asarray(volume))).reshape(-1)
    map_io.write_map_from_ft(tmp_path / "ft.mrc", volume_ft, volume.shape)
    assert_matches(map_io.load_relax_map(tmp_path / "ft.mrc"), volume, rtol=1e-5)  # float32 FFT round trip


def test_unlabeled_maps_need_the_legacy_option(tmp_path):
    volume = _volume()
    legacy = tmp_path / "legacy.mrc"
    helpers.write_mrc(str(legacy), volume)
    with pytest.raises(ValueError, match="legacy_recovar_sign"):
        map_io.load_relax_map(legacy)
    assert_matches(map_io.load_relax_map(legacy, legacy_recovar_sign=True), volume)

    relion_file = tmp_path / "relion.mrc"
    with mrcfile.new(str(relion_file)) as handle:
        handle.set_data(volume)
        handle.header.label[0] = b"Relion 5.0.1   25-Sep-26  03:00:00"
        handle.header.nlabl = 1
    with pytest.raises(ValueError, match="written by RELION"):
        map_io.load_relax_map(relion_file, legacy_recovar_sign=True)


def test_relion_reference_gives_the_internal_reference_relax_used_before():
    relion_frame = fixture_file("k1_5k128_data", "reference_init_relion.mrc")
    recovar_frame = fixture_file("k1_5k128_data", "reference_init.mrc")
    assert_matches(helpers.load_relion_volume(str(relion_frame)), helpers.load_mrc(str(recovar_frame)))


@pytest.mark.parametrize("name", ["run_class001.mrc", "run_half1_class001_unfil.mrc"])
def test_written_map_correlates_positively_with_relion(tmp_path, name):
    relion_map = fixture_file("k1_5k128_relion_os1", name)
    internal = helpers.load_relion_volume(str(relion_map))
    written = tmp_path / "final.mrc"
    map_io.write_map(written, internal, voxel_size=4.25)
    legacy = tmp_path / "legacy.mrc"
    helpers.write_mrc(str(legacy), internal, voxel_size=4.25)

    assert file_correlation(written, relion_map) >= SIGN_CORRELATION_MIN
    assert file_correlation(legacy, relion_map) <= -SIGN_CORRELATION_MIN  # the pre-change writer
    assert_matches(_raw(written), _raw(relion_map))
