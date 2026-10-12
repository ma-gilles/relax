"""The K=1 map-agreement gate scorer on tiny synthetic RELION/relax runs (CPU only)."""

from __future__ import annotations

import json

import mrcfile
import numpy as np
import pytest

from scripts import score_k1_map_gate as gate

BOX = 24


def _signal(seed):
    """A smooth random map whose power falls with frequency, so half-map FSCs cross 1/7 inside the box."""

    rng = np.random.default_rng(seed)
    freq = np.fft.fftfreq(BOX) * BOX
    z, y, x = np.meshgrid(freq, freq, freq, indexing="ij")
    radius = np.sqrt(x * x + y * y + z * z)
    spectrum = np.fft.fftn(rng.standard_normal((BOX, BOX, BOX))) * np.exp(-((radius / 4.0) ** 2))
    return np.real(np.fft.ifftn(spectrum)).astype(np.float32)


def _write_relion_run(root, signal, seed, noise):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    halves = [signal + noise * rng.standard_normal(signal.shape).astype(np.float32) for _ in range(2)]
    for name, data in (
        ("run_half1_class001_unfil.mrc", halves[0]),
        ("run_half2_class001_unfil.mrc", halves[1]),
        ("run_class001.mrc", 0.5 * (halves[0] + halves[1])),
    ):
        with mrcfile.new(str(root / name)) as handle:
            handle.set_data(np.ascontiguousarray(data, dtype=np.float32))
            handle.voxel_size = 2.0
    return root


def _write_relax_run(root, relion_like_root, perturbation=0.0, seed=0):
    """relax outputs as the driver writes them (RELION-convention arrays labeled by map_io.write_map):
    the maps of ``relion_like_root`` plus a small independent perturbation, as two engines that agree."""

    from recovar.utils.helpers import load_relion_volume

    from relax.io.map_io import write_map

    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    for src, dst in (
        ("run_half1_class001_unfil.mrc", "final_half1_unfil.mrc"),
        ("run_half2_class001_unfil.mrc", "final_half2_unfil.mrc"),
        ("run_class001.mrc", "final_merged.mrc"),
    ):
        data = np.asarray(load_relion_volume(str(relion_like_root / src)), dtype=np.float32)
        data = data + perturbation * float(np.std(data)) * rng.standard_normal(data.shape).astype(np.float32)
        write_map(root / dst, data, voxel_size=2.0)
    return root


@pytest.fixture
def runs(tmp_path):
    signal = _signal(0)
    relion = {f"relion_{k}": _write_relion_run(tmp_path / f"relion_{k}", signal, seed=10 + k, noise=0.02) for k in range(2)}
    agreeing = _write_relax_run(tmp_path / "relax_ok", relion["relion_0"], perturbation=0.01)
    other = _write_relax_run(tmp_path / "relax_bad", _write_relion_run(tmp_path / "src_bad", _signal(1), 21, 0.02))
    return relion, agreeing, other


def test_matching_run_passes_and_band_is_reported(runs, tmp_path):
    relion, agreeing, _ = runs
    result = gate.score_map_gate(
        relion_runs=relion, relax_outputs=agreeing, box=BOX, voxel=2.0, label="t", skip_masked="unit test"
    )
    assert result["map_gate_pass"] is True
    assert "relax__relion_0" in result["threshold_met_against"]
    assert set(result["relion_band"]) == set(gate.METRICS)
    assert result["maps"]["relax"]["merged"]["convention"] == "relion"
    assert result["maps"]["relion_0"]["merged"]["convention"] == "relion"
    assert result["masked"] is None and "unit test" in result["masked_null_reason"]


def test_unrelated_run_fails(runs):
    relion, _, other = runs
    result = gate.score_map_gate(
        relion_runs=relion, relax_outputs=other, box=BOX, voxel=2.0, label="t", skip_masked="unit test"
    )
    assert result["map_gate_pass"] is False
    assert result["threshold_met_against"] == []


def test_cli_with_explicit_runs_and_masked_metrics_required(runs, tmp_path):
    relion, agreeing, _ = runs
    out = tmp_path / "score.json"
    relion_args = [f"{name}={path}" for name, path in relion.items()]
    with pytest.raises(SystemExit, match="masked metrics"):
        gate.main(
            [
                "--relion",
                *relion_args,
                "--box",
                str(BOX),
                "--voxel",
                "2.0",
                "--relax-outputs",
                str(agreeing),
                "--label",
                "t",
                "--out",
                str(out),
            ]
        )
    assert (
        gate.main(
            [
                "--relion",
                *relion_args,
                "--box",
                str(BOX),
                "--voxel",
                "2.0",
                "--relax-outputs",
                str(agreeing),
                "--label",
                "t",
                "--out",
                str(out),
                "--skip-masked",
                "unit test",
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["map_gate_pass"] is True


def test_reference_sets_point_at_curated_fixtures():
    sets = json.loads(gate.REFERENCE_SETS.read_text())["sets"]
    assert "empiar10097_10k_k1" in sets
    for spec in sets.values():
        assert spec["mask_dataset"]
        for path in spec["runs"].values():
            assert "/em_fixtures/" in path and "/em_work/" not in path
