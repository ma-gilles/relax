"""relax's fixture generators apply one total B-factor (scripts/fixture_bfactor.py).

The source maps the generators build carry B = 0 (recovar's generate_trajectory_volumes default of 80
never applies) and the EM-development preset adds the requested total. The measured total is the slope
of log(|F_effective| / |F_source(B=0)|) against -|q|^2 / 4 over the resolved shells.
"""

import numpy as np
import pytest
from recovar.core import fourier_transform_utils as ftu
from recovar.simulation import solvent_contrast

pytestmark = pytest.mark.unit

GRID, VOXEL = 32, 2.0


def _measured_total_bfactor(effective_ft, reference_ft):
    freqs = np.fft.fftshift(np.fft.fftfreq(GRID, d=VOXEL))
    qz, qy, qx = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    q2 = (qx**2 + qy**2 + qz**2).reshape(-1)
    ref = np.abs(np.asarray(reference_ft).reshape(-1))
    eff = np.abs(np.asarray(effective_ft).reshape(-1))
    keep = (q2 > 0) & (q2 < (0.4 / VOXEL) ** 2) & (ref > 1e-3 * ref.max())
    ratio = np.log(eff[keep] / ref[keep])
    slope = np.sum(ratio * (-q2[keep] / 4)) / np.sum((q2[keep] / 4) ** 2)
    return slope


def _effective(source_real, kwargs):
    """The volume the projector sees: the source map times the preset's B term (solvent term off, a = 0)."""
    record = solvent_contrast.make_record(
        True, voxel_size=VOXEL, grid_size=GRID, a=0.0, B=solvent_contrast.DEFAULT_B, atomic_bfactor=kwargs["atomic_bfactor"]
    )
    return solvent_contrast.apply_record(ftu.get_dft3(source_real).reshape(1, -1), record)[0]


def _write_pdb(path):
    rng = np.random.default_rng(3)
    lines = [
        f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C"
        for i, (x, y, z) in enumerate(rng.uniform(-12, 12, size=(40, 3)))
    ]
    path.write_text("\n".join(lines) + "\nEND\n")


@pytest.mark.parametrize("total", (15.0, 40.0))
def test_pdb_source_is_unblurred_and_the_preset_sets_the_total(tmp_path, total):
    from recovar import utils
    from recovar.simulation import simulate_scattering_potential as ssp

    from scripts import prepare_cryobench_pdb_multiclass_relion_parity_benchmark as prep
    from scripts.fixture_bfactor import resolve_total_bfactor

    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _write_pdb(pdb_dir / "000_test.pdb")
    prefix, _ = prep._generate_volumes_from_pdbs(
        pdb_dir, tmp_path / "out", grid_size=GRID, voxel_size=VOXEL, pdb_bfactor=0.0, symmetry="C1", force=True
    )
    source = utils.load_mrc(f"{prefix}0000.mrc")
    raw = ssp.generate_molecule_spectrum_from_pdb_id(str(pdb_dir / "000_test.pdb"), voxel_size=VOXEL, grid_size=GRID)
    assert _measured_total_bfactor(ftu.get_dft3(source), raw) == pytest.approx(0.0, abs=0.5)  # no PDB B column read
    kwargs, record = resolve_total_bfactor({"atomic_solvent_correction": True, "atomic_bfactor": total}, baked_in=0.0, source="pdb")
    assert _measured_total_bfactor(_effective(source, kwargs), raw) == pytest.approx(total, abs=0.5)
    assert record["total_bfactor_A2"] == total and record["baked_in_bfactor_A2"] == 0.0


def test_trajectory_source_is_built_at_b0_explicitly_and_the_preset_sets_the_total(monkeypatch, tmp_path):
    from recovar import utils
    from recovar.simulation.trajectory_generation import generate_trajectory_volumes

    from scripts import prepare_pdb_k1_relion_sanity_benchmark as prep

    calls = []

    def recording_generate(**kwargs):
        calls.append(kwargs)
        return generate_trajectory_volumes(**kwargs)

    captured = {}

    class _Stop(Exception):
        pass

    def fake_generate_synthetic_dataset(*_args, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(prep, "generate_trajectory_volumes", recording_generate)
    monkeypatch.setattr(prep.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)
    with pytest.raises(_Stop):
        prep.prepare_benchmark(
            tmp_path / "k1", n_images=1, grid_size=GRID, voxel_size=VOXEL, noise_level=1.0, noise_model="white",
            dataset_params_option="uniform", init_resolution_ang=30.0, pdb_path=None, pdb_bfactor=0.0,
            noise_scale_std=0.0, contrast_std=0.0, volume_radius=0.7, percent_outliers=0.0, put_extra_particles=False,
            image_offset_n_std=0.0, relion_bg_radius_px=None, noise_rng_batch_size=None, relion_normalize=True,
            streaming_mmap=False, streaming_chunk_size=500, disc_type="cubic", seed=17,
            atomic_volume_kwargs={"atomic_solvent_correction": True, "atomic_bfactor": 40.0},
        )
    assert calls and all(call["Bfactor"] == 0.0 for call in calls)  # never recovar's default of 80
    source = utils.load_mrc(str(tmp_path / "k1/pdb_state/vol0000.mrc"))
    generate_trajectory_volumes(
        output_dir=str(tmp_path / "ref"), grid_size=GRID, n_volumes=1, voxel_size=VOXEL, Bfactor=0.0,
        max_rotation_degrees=0.0, pdb_path=None, output_prefix=str(tmp_path / "ref/vol"),
    )
    reference = ftu.get_dft3(utils.load_mrc(str(tmp_path / "ref/vol0000.mrc")))
    assert captured["atomic_bfactor"] == 40.0
    assert _measured_total_bfactor(_effective(source, captured), reference) == pytest.approx(40.0, abs=0.5)


def test_asset_maps_refuse_a_total_below_their_baked_in_b():
    from scripts.fixture_bfactor import ASSET_BAKED_BFACTOR, resolve_total_bfactor

    kwargs, record = resolve_total_bfactor(None, baked_in=ASSET_BAKED_BFACTOR, source="assets")
    assert kwargs["atomic_bfactor"] == 0.0 and record["total_bfactor_A2"] == 100.0
    kwargs, record = resolve_total_bfactor({"atomic_solvent_correction": True, "atomic_bfactor": 130.0}, baked_in=100.0, source="assets")
    assert kwargs["atomic_bfactor"] == 30.0 and record["preset_atomic_bfactor_A2"] == 30.0
    with pytest.raises(ValueError, match="below the 100.0"):
        resolve_total_bfactor({"atomic_solvent_correction": True, "atomic_bfactor": 40.0}, baked_in=100.0, source="assets")
