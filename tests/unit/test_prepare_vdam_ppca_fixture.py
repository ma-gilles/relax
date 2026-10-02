"""State selection and early validation for reproducible simulator fixtures."""

import numpy as np
import pytest

from scripts.prepare_vdam_ppca_fixture import DEFAULT_MAPS, prepare, validate_state_inputs

pytestmark = pytest.mark.unit


def test_original_maps_remain_default(tmp_path):
    for name in DEFAULT_MAPS:
        (tmp_path / name).touch()
    maps, counts = validate_state_inputs(tmp_path, [6667, 6667, 6666])
    assert [path.name for path in maps] == list(DEFAULT_MAPS)
    assert counts == (6667, 6667, 6666)


def test_six_maps_preserve_order_and_counts(tmp_path):
    names = [f"state{i}.mrc" for i in range(6)]
    for name in names:
        (tmp_path / name).touch()
    expected = [16667, 16667, 16667, 16667, 16666, 16666]
    maps, counts = validate_state_inputs(tmp_path, expected, names)
    assert [path.name for path in maps] == names
    assert counts == tuple(expected)
    assert sum(counts) == 100000


@pytest.mark.parametrize(
    "counts,names",
    [
        ([1], ["a.mrc", "b.mrc"]),
        ([], []),
        ([0], ["a.mrc"]),
        ([-1], ["a.mrc"]),
        ([1.5], ["a.mrc"]),
        ([True], ["a.mrc"]),
        ([np.bool_(True)], ["a.mrc"]),
        ([1, 1], ["a.mrc", "a.mrc"]),
    ],
)
def test_invalid_states_rejected_without_creating_output(tmp_path, counts, names):
    (tmp_path / "a.mrc").touch()
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        prepare(output, tmp_path, counts, maps=names)
    assert not output.exists()


def test_missing_map_rejected_before_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        prepare(output, tmp_path, [1], maps=["absent.mrc"])
    assert not output.exists()
