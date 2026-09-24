"""EM/VDAM fixture scripts enable recovar's atomic-volume preset and load its truth."""

import json
import pickle
import sys

import numpy as np
import pytest
from recovar.simulation import solvent_contrast

pytestmark = pytest.mark.unit


def _k1_prepare_kwargs(tmp_path, pdb_bfactor, **extra):
    return dict(
        n_images=1,
        grid_size=16,
        voxel_size=34.0,
        noise_level=1.0,
        noise_model="white",
        dataset_params_option="uniform",
        init_resolution_ang=30.0,
        pdb_path=None,
        pdb_bfactor=pdb_bfactor,
        noise_scale_std=0.0,
        contrast_std=0.0,
        volume_radius=0.7,
        percent_outliers=0.0,
        put_extra_particles=False,
        image_offset_n_std=0.0,
        relion_bg_radius_px=None,
        noise_rng_batch_size=None,
        relion_normalize=True,
        streaming_mmap=False,
        streaming_chunk_size=500,
        disc_type="cubic",
        seed=17,
        **extra,
    )


def _run_k1_prepare(monkeypatch, tmp_path, pdb_bfactor, **extra):
    from scripts import prepare_pdb_k1_relion_sanity_benchmark as prep

    captured = {}

    def fake_generate_trajectory_volumes(**kwargs):
        prefix = kwargs["output_prefix"]
        for idx in range(kwargs["n_volumes"]):
            path = prep.Path(f"{prefix}{idx:04d}.mrc")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake mrc")

    def fake_generate_synthetic_dataset(output_folder, *_args, **kwargs):
        captured.update(kwargs)
        output_dir = prep.Path(output_folder)
        (output_dir / "particles.star").write_text("data_particles\n")
        (output_dir / "particles.16.mrcs").write_bytes(b"fake stack")
        prep.utils.pickle_dump((np.eye(3)[None], np.zeros((1, 2))), output_dir / "poses.pkl")
        prep.utils.pickle_dump(np.zeros((1, 10)), output_dir / "ctf.pkl")
        prep.utils.pickle_dump({"image_assignment": np.array([0])}, output_dir / "simulation_info.pkl")

    monkeypatch.setattr(prep, "generate_trajectory_volumes", fake_generate_trajectory_volumes)
    monkeypatch.setattr(prep.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)
    monkeypatch.setattr(prep, "_write_references", lambda *_args, **_kwargs: None)
    prep.prepare_benchmark(tmp_path, **_k1_prepare_kwargs(tmp_path, pdb_bfactor, **extra))
    return captured


def test_k1_prepare_enables_preset_with_bfactor_100_for_unblurred_pdb_volumes(monkeypatch, tmp_path):
    captured = _run_k1_prepare(monkeypatch, tmp_path, pdb_bfactor=0.0)
    assert captured["atomic_solvent_correction"] is True
    assert "atomic_bfactor" not in captured  # recovar's default B_atomic = 100 A^2


def test_k1_prepare_adds_only_solvent_term_to_already_bfactored_volumes(monkeypatch, tmp_path):
    captured = _run_k1_prepare(monkeypatch, tmp_path, pdb_bfactor=80.0)
    assert captured["atomic_solvent_correction"] is True
    assert captured["atomic_bfactor"] == 0.0


def test_k1_prepare_opt_out_and_explicit_bfactor_are_forwarded(monkeypatch, tmp_path):
    off = _run_k1_prepare(
        monkeypatch, tmp_path / "off", pdb_bfactor=80.0, atomic_volume_kwargs={"atomic_solvent_correction": False}
    )
    assert off["atomic_solvent_correction"] is False and "atomic_bfactor" not in off
    explicit = _run_k1_prepare(
        monkeypatch,
        tmp_path / "explicit",
        pdb_bfactor=80.0,
        atomic_volume_kwargs={"atomic_solvent_correction": True, "atomic_bfactor": 50.0},
    )
    assert explicit["atomic_bfactor"] == 50.0


@pytest.mark.parametrize(
    "module_name, required",
    [
        ("prepare_pdb_k1_relion_sanity_benchmark", []),
        ("prepare_cryobench_pdb_multiclass_relion_parity_benchmark", ["--pdb-dir", "pdbs"]),
        ("prepare_relion_parity_benchmark", []),
        ("prepare_relion_multiclass_parity_benchmark", []),
    ],
)
def test_prepare_clis_default_on_with_opt_out(monkeypatch, tmp_path, module_name, required):
    import importlib

    prep = importlib.import_module(f"scripts.{module_name}")
    captured = {}
    monkeypatch.setattr(prep, "prepare_benchmark", lambda *args, **kwargs: captured.update(kwargs))
    base = [module_name, "--output-dir", str(tmp_path), *required]

    monkeypatch.setattr(sys, "argv", base)
    prep.main()
    assert captured["atomic_volume_kwargs"] == {
        "atomic_solvent_correction": True,
        "solvent_contrast_a": None,
        "solvent_contrast_B": None,
        "atomic_bfactor": None,
    }
    monkeypatch.setattr(sys, "argv", base + ["--no-atomic-solvent-correction"])
    prep.main()
    assert captured["atomic_volume_kwargs"]["atomic_solvent_correction"] is False


def test_cryobench_manifest_points_fixture_validation_at_simulated_truth(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from relax.ppca_refinement import fixture_validation
    from scripts import prepare_cryobench_pdb_multiclass_relion_parity_benchmark as prep

    shape = (4, 4, 4)
    fourier = np.stack([np.asarray(prep.ftu.get_dft3(np.full(shape, k + 1.0))).reshape(-1) for k in range(2)])
    dataset = SimpleNamespace(
        volume_shape=shape, voxel_size=2.0, get_valid_frequency_indices=lambda *, rad: np.ones(64, np.float32)
    )
    heterogeneous = SimpleNamespace(volumes=fourier, get_mean=lambda: fourier.mean(axis=0))
    monkeypatch.setattr(prep, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(prep.utils, "pickle_load", lambda path: {})
    monkeypatch.setattr(prep.synthetic_dataset, "load_heterogeneous_reconstruction", lambda info: heterogeneous)
    raw = [tmp_path / f"vol{k:04d}.mrc" for k in range(2)]
    for path in raw:
        path.write_bytes(b"raw input")
    manifest = [{"class_index": k, "class_number": k + 1, "volume_path": str(raw[k])} for k in range(2)]
    (tmp_path / "class_manifest.json").write_text(json.dumps(manifest))

    prep._write_class_references(tmp_path, 4, 2, 2)

    rows = json.loads((tmp_path / "class_manifest.json").read_text())
    assert [row["volume_path"] for row in rows] == [str(p) for p in raw]
    expected = [tmp_path / f"reference_gt_class{k:03d}.mrc" for k in (1, 2)]
    assert [row["gt_volume_path"] for row in rows] == [str(p) for p in expected]
    paths = fixture_validation._manifest_volume_paths_in_recovar_order(
        tmp_path / "class_manifest.json", recovar_to_relion=None
    )
    assert paths == expected
    # Manifests written before the gt_volume_path field keep their input paths.
    (tmp_path / "old_manifest.json").write_text(json.dumps(manifest))
    old = fixture_validation._manifest_volume_paths_in_recovar_order(
        tmp_path / "old_manifest.json", recovar_to_relion=None
    )
    assert old == raw


def _corrected_simulation_info(path):
    info = {
        "image_assignment": np.array([0, 1], dtype=np.int64),
        "grid_size": 4,
        solvent_contrast.METADATA_KEY: solvent_contrast.make_record(True, voxel_size=2.0, grid_size=4),
    }
    with path.open("wb") as f:
        pickle.dump(info, f)
    return path


def test_gt_weighted_init_refuses_raw_maps_for_corrected_dataset(tmp_path):
    from scripts.prepare_gt_weighted_ppca_init import prepare_gt_weighted_ppca_init

    with pytest.raises(ValueError, match="--apply-simulation-scale"):
        prepare_gt_weighted_ppca_init(
            volume_paths=[tmp_path / "vol0000.mrc", tmp_path / "vol0001.mrc"],
            simulation_info_path=_corrected_simulation_info(tmp_path / "simulation_info.pkl"),
            output_dir=tmp_path / "out",
            q=1,
            frame="recovar",
            write_maps=False,
        )


def test_synthetic_recovery_check_refuses_raw_maps_for_corrected_dataset(tmp_path):
    from scripts.check_ppca_synthetic_recovery import run_checks

    run_dir = tmp_path / "run"
    (run_dir / "embedding_best_pose").mkdir(parents=True)
    np.savez(run_dir / "final_ppca_dense.npz", placeholder=np.zeros(1))
    np.save(run_dir / "embedding_best_pose" / "embedding_z.npy", np.zeros((2, 1)))
    with pytest.raises(ValueError, match="simulation-info"):
        run_checks(
            run_dir=run_dir,
            simulation_info_path=_corrected_simulation_info(tmp_path / "simulation_info.pkl"),
            volume_glob=str(tmp_path / "vol*.mrc"),
            healpix_order=1,
            rotation_source="healpix",
            offset_range_px=0.0,
            offset_step_px=1.0,
            q=1,
            gt_volume_source="mrc-glob",
            translation_source="grid",
        )


@pytest.mark.parametrize(
    "module_name, writer, extra",
    [
        ("prepare_relion_parity_benchmark", "_write_reference_volumes", {}),
        ("prepare_relion_multiclass_parity_benchmark", "_write_class_references", {"n_classes": 2, "init_radius": 3}),
    ],
)
def test_asset_volume_scripts_add_only_the_solvent_term(monkeypatch, tmp_path, module_name, writer, extra):
    """recovar's bundled asset maps already carry B = 100 A^2, so B_atomic defaults to 0 here."""
    import importlib

    prep = importlib.import_module(f"scripts.{module_name}")
    captured = []
    monkeypatch.setattr(prep.simulator, "generate_synthetic_dataset", lambda *args, **kwargs: captured.append(kwargs))
    monkeypatch.setattr(prep, writer, lambda *args, **kwargs: None)
    common = dict(n_images=4, grid_size=16, noise_level=1.0, relion_normalize=False, disc_type="cubic", **extra)

    prep.prepare_benchmark(str(tmp_path / "default"), **common)
    prep.prepare_benchmark(
        str(tmp_path / "explicit"),
        atomic_volume_kwargs={"atomic_solvent_correction": True, "atomic_bfactor": 100.0},
        **common,
    )
    prep.prepare_benchmark(str(tmp_path / "off"), atomic_volume_kwargs={"atomic_solvent_correction": False}, **common)

    assert captured[0]["atomic_solvent_correction"] is True and captured[0]["atomic_bfactor"] == 0.0
    assert captured[1]["atomic_bfactor"] == 100.0
    assert captured[2]["atomic_solvent_correction"] is False and "atomic_bfactor" not in captured[2]
