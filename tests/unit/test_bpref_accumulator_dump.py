"""Accumulator captures keep their schema, stage and per-half arrays."""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.diagnostics.reconstruction import write_bpref_accumulators

_SCHEMA_BY_STAGE = {"pre_lowres_join": "recovar-bpref-prejoin-v2", "post_lowres_join": "recovar-bpref-accum-v2"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stage", "loaded_stage"),
    [("prejoin", "pre_lowres_join"), ("accum", "post_lowres_join")],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("run_id", [None, "repeat-2"])
def test_dump_round_trip(tmp_path, monkeypatch, stage, loaded_stage, dtype, run_id):
    if run_id is None:
        monkeypatch.delenv("RELAX_BPREF_BOUNDARY_DUMP_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("RELAX_BPREF_BOUNDARY_DUMP_RUN_ID", run_id)
    numerators = [np.array([1 + 2j, 3 - 4j], dtype=dtype), np.array([5j, 6], dtype=dtype)]
    weights = [np.array([7 + 8j, 9], dtype=dtype), np.array([10, 11j], dtype=dtype)]
    original = [value.copy() for value in numerators + weights]
    dump_dir = tmp_path / "new" / "capture"

    write_bpref_accumulators(
        str(dump_dir),
        stage=stage,
        iteration=2,
        current_size=56,
        padding_factor=2,
        box_size=128,
        voxel_size=4.25,
        volume_shape=(128, 128, 128),
        accumulator_shape=(123, 123, 123),
        Ft_y_0=numerators[0],
        Ft_y_1=numerators[1],
        Ft_ctf_0=weights[0],
        Ft_ctf_1=weights[1],
    )

    path = dump_dir / f"recovar_bpref_{stage}_it003.npz"
    assert list(dump_dir.iterdir()) == [path]
    with np.load(path, allow_pickle=False) as saved:
        assert saved["run_id"].item() == ("unset" if run_id is None else run_id)
        assert saved["iteration"].item() == 3
        for key in (
            "iteration",
            "current_size",
            "padding_factor",
            "grid_size",
            "volume_shape",
            "mstep_accumulator_shape",
        ):
            assert saved[key].dtype == np.int32
        assert saved["voxel_size"].dtype == np.float32
    with np.load(path, allow_pickle=False) as saved:
        assert saved["schema"].item() == _SCHEMA_BY_STAGE[loaded_stage]
        for half_index in (0, 1):
            numerator = np.asarray(saved[f"Ft_y_{half_index}"])
            weight = np.asarray(saved[f"Ft_ctf_{half_index}"])
            assert numerator.dtype == numerators[half_index].dtype
            assert weight.dtype == weights[half_index].real.dtype
            assert_matches(numerator, numerators[half_index])
            assert_matches(weight, weights[half_index].real)
    for value, before in zip(numerators + weights, original, strict=True):
        assert_matches(value, before)
