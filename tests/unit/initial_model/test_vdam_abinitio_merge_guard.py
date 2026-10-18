from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from relax.diagnostics.gt_metrics import (
    DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    DEFAULT_GT_ALIGN_MAX_SHELL,
    relion_alignment_rotations,
)
from scripts.run_vdam_abinitio_merge_guard import _default_output_root, build_guard_commands, run_guard

REPO_ROOT = Path(__file__).resolve().parents[3]


def _initial_model_package_source() -> str:
    """Every module of the native InitialModel package; the guarded wiring may live in any owner."""
    return "\n".join(path.read_text() for path in sorted((REPO_ROOT / "relax/vdam").glob("*.py")))


def _load_long_guard_module():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    module_path = REPO_ROOT / "tests/long_test/test_em_parity_long.py"
    spec = importlib.util.spec_from_file_location("em_parity_long_guard_for_unit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vdam_alignment_defaults_are_merge_guarded():
    assert DEFAULT_GT_ALIGN_HEALPIX_ORDER == 2
    assert DEFAULT_GT_ALIGN_MAX_SHELL == 8
    assert relion_alignment_rotations(DEFAULT_GT_ALIGN_HEALPIX_ORDER).shape == (4608, 3, 3)


def test_gt_alignment_cli_wiring_survives_merges():
    expected_tokens = [
        "--gt_align",
        "--gt_align_healpix_order",
        "DEFAULT_GT_ALIGN_HEALPIX_ORDER",
        "--gt_align_max_shell",
        "DEFAULT_GT_ALIGN_MAX_SHELL",
        "aligned_fsc_vs_gt",
        "gt_align_rotation_index",
    ]
    for relative_path in [
        "scripts/run_multi_iter_parity.py",
        "scripts/postprocess_multi_iter_gt.py",
    ]:
        text = (REPO_ROOT / relative_path).read_text()
        missing = [token for token in expected_tokens if token not in text]
        assert not missing, f"{relative_path} lost GT-alignment wiring: {missing}"


def test_standalone_abinitio_evaluator_cli_wiring_survives_merges():
    text = (REPO_ROOT / "scripts/evaluate_ab_initio_gt.py").read_text()
    expected_tokens = [
        "--volume_frame",
        "--gt_frame",
        "choices=(\"relion\", \"recovar\")",
        "--gt_align",
        "aligned_prefix = f\"{label}_aligned\"",
        "_fsc_vs_gt",
        "Native InitialModel writes RELION-frame",
        "load_relion_volume",
        "output_json",
        "output_npz",
    ]
    missing = [token for token in expected_tokens if token not in text]
    assert not missing, f"standalone evaluator lost merge-guarded contract: {missing}"


def test_merge_guard_plan_contains_cpu_and_gpu_gates():
    cpu_names = [command.name for command in build_guard_commands("cpu")]
    assert cpu_names == [
        "py_compile",
        "vdam_abinitio_contracts",
        "vdam_frozen_scorecard",
        "initial_model_vdam_unit_slice",
        "initial_model_unit_suite",
        "em_fast_guard",
    ]

    quick_names = [command.name for command in build_guard_commands("cpu", quick=True)]
    assert "initial_model_unit_suite" not in quick_names
    assert "initial_model_vdam_unit_slice" in quick_names
    assert "em_fast_guard" in quick_names

    all_names = [command.name for command in build_guard_commands("all", quick=True)]
    assert "native_initialmodel_k2_smoke_gpu" in all_names
    assert "em_parity_fast_gpu" in all_names
    assert "extract_em_parity_fast_tables" in all_names

    gpu_names = [command.name for command in build_guard_commands("gpu")]
    assert gpu_names[0] == "native_initialmodel_k2_smoke_gpu"


def test_merge_guard_output_root_can_use_overflow_scratch(monkeypatch, tmp_path):
    overflow = tmp_path / "overflow"
    monkeypatch.setenv("VDAM_ABINITIO_GUARD_OUTPUT_ROOT", str(overflow))
    monkeypatch.delenv("RELAX_AGENT_SCRATCH_ROOT", raising=False)
    assert _default_output_root() == overflow

    agent_root = tmp_path / "agent-root"
    monkeypatch.delenv("VDAM_ABINITIO_GUARD_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("RELAX_AGENT_SCRATCH_ROOT", str(agent_root))
    assert _default_output_root() == agent_root


def test_merge_guard_redirects_runtime_caches_to_output_dir():
    guard = (REPO_ROOT / "scripts/run_vdam_abinitio_merge_guard.py").read_text()
    expected_tokens = [
        'pytest_cache_dir = output_dir / "pytest_cache"',
        '"-o cache_dir={pytest_cache_dir}',
        '"PYTHONPYCACHEPREFIX"',
        'pycache_dir = output_dir / "pycache"',
    ]
    missing = [token for token in expected_tokens if token not in guard]
    assert not missing, f"VDAM merge guard lost output-dir cache redirection: {missing}"


def test_long_native_quality_guard_uses_relion_initialmodel_reference():
    long_test = (REPO_ROOT / "tests/long_test/test_em_parity_long.py").read_text()
    slurm_launcher = (REPO_ROOT / "scripts/run_em_parity_long_slurm.sh").read_text()

    expected_long_tokens = [
        "K1_NATIVE_RELION_DIR",
        'fixture_root("k1_50k256_relion_initialmodel_it008")',
        "_assert_relion_initialmodel_reference",
        "--grad",
        "--denovo_3dref",
        "_rlnDoGradientRefine",
        "_rlnDoAutoRefine",
        "run_it008_class001.mrc",
        "relion_initialmodel_it008",
        "_relion_frame_map_similarity",
        "k1_native_initialmodel_vdam_vs_relion_it008_corr",
        ">= 0.999",
        "direct VDAM vs RELION it008",
    ]
    missing = [token for token in expected_long_tokens if token not in long_test]
    assert not missing, f"native InitialModel long guard lost RELION --grad reference checks: {missing}"

    # The RELION --grad InitialModel reference is a curated manifest fixture; the launcher
    # verifies it before submitting and the test checks its command header.
    expected_launcher_tokens = ["tests/helpers/em_fixtures.py", "k1_50k256_relion_initialmodel_it008"]
    missing = [token for token in expected_launcher_tokens if token not in slurm_launcher]
    assert not missing, f"EM-long launcher no longer verifies the RELION InitialModel reference: {missing}"
    manifest = json.loads((REPO_ROOT / "tests/fixtures/em_fixture_manifest.json").read_text())["sets"]
    headers = manifest["k1_50k256_relion_initialmodel_it008"]["provenance"]["relion_optimiser_headers"]
    command = next(iter(headers.values()))["command"]
    assert "--grad" in command and "--denovo_3dref" in command, command


def test_native_vdam_subset_order_uses_relion_sorted_idx_base_order():
    driver = (REPO_ROOT / "relax/vdam/driver.py").read_text()
    iteration_loop = _initial_model_package_source()  # the subset draw lives in its owner module

    expected_driver_tokens = [
        "particle_order = experiment_read_order(main_star)",
        "experiment_read_order(main_star)",
        "particle_order=particle_order",
    ]
    missing = [token for token in expected_driver_tokens if token not in driver]
    assert not missing, f"native InitialModel driver lost RELION sorted_idx base-order wiring: {missing}"

    expected_loop_tokens = [
        "particle_order: Sequence[int] | None = None",
        "base_order = np.asarray(particle_order",
        "shuffled = base_order.copy()",
        "shuffled = base_order[permutation]",
    ]
    missing = [token for token in expected_loop_tokens if token not in iteration_loop]
    assert not missing, f"VDAM iteration loop lost RELION sorted_idx base-order handling: {missing}"


def test_native_vdam_solvent_flattening_is_separate_from_zero_mask():
    driver = (REPO_ROOT / "relax/vdam/driver.py").read_text()
    iteration_loop = (REPO_ROOT / "relax/vdam/iteration_loop.py").read_text()
    command = (REPO_ROOT / "relax/commands/initial_model.py").read_text()
    options = (REPO_ROOT / "relax/vdam/native_options.py").read_text()

    expected_tokens = [
        "do_solvent: bool = True",
        "if opts.do_solvent",
        "relion_solvent_mask",
        "relion_solvent_flatten_state",
        '"do_solvent": args.solvent',
    ]
    haystack = "\n".join([driver, iteration_loop, command, options])
    missing = [token for token in expected_tokens if token not in haystack]
    assert not missing, f"native InitialModel lost RELION --flatten_solvent post-M-step wiring: {missing}"


def test_vdam_frozen_trajectory_runner_and_fsc_auditor_are_merge_guarded():
    guard = (REPO_ROOT / "scripts/run_vdam_abinitio_merge_guard.py").read_text()
    runner = (REPO_ROOT / "scripts/run_vdam_relion_parity_case.py").read_text()
    auditor = (REPO_ROOT / "scripts/audit_vdam_fsc_trajectory.py").read_text()
    sbatch = (REPO_ROOT / "scripts/run_vdam_relion_parity_case.sbatch").read_text()
    expected_tokens = [
        "test_audit_vdam_fsc_trajectory.py",
        "test_run_vdam_relion_parity_case.py",
        "build_relion_command",
        "build_recovar_command",
        "--require-custom-cuda",
        'env["JAX_PLATFORMS"] = "cuda,cpu"',
        "runtime_environment.json",
        "materialize(",
        "cwd=fixture_dir",
        "paired_gpu_uuid.json",
        "signed shellwise FSC and normalized non-DC FSC-AUC only",
        "CHECKPOINTS = (0, 1, 2, 4, 8)",
        "artifact_topology_exact",
        "correlation_used",
        "--gres=gpu:1",
        "RELAX_CUDA_LIB",
        "cuda_backproject.cuda_available()",
        "VDAM parity provenance/GPU/CUDA-FFI gate passed",
        "binding_path.is_file()",
        '--threads "${RELION_THREADS:-8}"',
        '--relion-refine "${RELION_REFINE}"',
        'mkdir -p "${RELION_ACC_DUMP_DIR}"',
    ]
    haystack = "\n".join([guard, runner, auditor, sbatch])
    assert 'pathlib.Path(_relion_bind_core.__file__).resolve()).startswith' not in sbatch
    missing = [token for token in expected_tokens if token not in haystack]
    assert not missing, f"VDAM trajectory runner/auditor lost required wiring: {missing}"


def test_vdam_full_trajectory_expansion_covers_seed_and_parameter_robustness():
    suite = json.loads(
        (REPO_ROOT / "docs/math/vdam_k1_full_trajectory_expansion_v2.json").read_text()
    )

    assert suite["schema"] == "recovar.vdam_relion_parity_suite.v1"
    assert suite["suite_id"] == "vdam-k1-full-trajectory-expansion-v2"
    assert suite["acceptance_contract"]["required_checkpoints"] == list(range(201))
    cases = suite["cases"]
    assert [case["id"] for case in cases] == [f"vdam-gf{index:02d}" for index in range(23, 43)]
    assert all(case["definition"]["nr_classes"] == 1 for case in cases)
    assert all(case["definition"]["nr_iter"] == 200 for case in cases)
    assert all(case["definition"]["random_seed"] == 17 for case in cases[:12])
    assert all("seed_replica" in case["coverage"] for case in cases[:12])
    assert all("parameter_override" in case["coverage"] for case in cases[12:])
    coverage = {axis for case in cases for axis in case["coverage"]}
    assert {
        "uniform_poses",
        "anisotropic_poses",
        "kent_poses",
        "white_noise",
        "radial_noise",
        "low_noise",
        "high_noise",
        "very_high_noise",
        "outliers",
        "severe_outliers",
        "extreme_outliers",
        "junk_particles",
        "no_ctf",
        "translations",
        "high_resolution",
        "midscale",
        "tau2_fudge",
        "healpix_order",
        "oversampling",
        "offset_search",
        "particle_diameter",
        "padding_factor",
    } <= coverage
    by_id = {case["id"]: case["definition"] for case in cases}
    assert by_id["vdam-gf35"]["tau2_fudge"] == 2
    assert by_id["vdam-gf36"]["tau2_fudge"] == 8
    assert by_id["vdam-gf37"]["healpix_order"] == 2
    assert by_id["vdam-gf38"]["oversampling"] == 0
    assert (by_id["vdam-gf39"]["offset_range_px"], by_id["vdam-gf39"]["offset_step_px"]) == (4, 1)
    assert (by_id["vdam-gf40"]["offset_range_px"], by_id["vdam-gf40"]["offset_step_px"]) == (8, 2)
    assert by_id["vdam-gf41"]["particle_diameter_angstrom"] == 160
    assert by_id["vdam-gf42"]["padding_factor"] == 2


def test_vdam_case_runner_uses_job_scoped_runtime_roots():
    runner = (REPO_ROOT / "scripts/run_vdam_relion_parity_case.sbatch").read_text()
    expected_tokens = [
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/",
        'export TMPDIR="${RUNTIME_ROOT}/tmp"',
        'export PIXI_HOME="${RUNTIME_ROOT}/pixi_home"',
        'export RATTLER_CACHE_DIR="${RUNTIME_ROOT}/rattler_cache"',
    ]
    missing = [token for token in expected_tokens if token not in runner]
    assert not missing, f"VDAM case runner lost job-scoped runtime roots: {missing}"


def test_vdam_paired_runner_imports_from_clean_environment():
    runner = (REPO_ROOT / "scripts/run_vdam_relion_parity_case.sbatch").read_text()

    assert '"${PIXI_PY}" -m scripts.run_vdam_relion_parity_case "${RUN_ARGS[@]}"' in runner
    assert '"${PIXI_PY}" scripts/run_vdam_relion_parity_case.py' not in runner


def test_native_vdam_tau2_refresh_and_ssnr_diagnostics_are_merge_guarded():
    """Protect the K=1 current-size parity fix.

    RELION refreshes tau2_class from Projector::computeFourierTransformMap
    before each E-step. Native InitialModel must keep that behavior, and must
    write sigma2/coverage diagnostics so DVP/current-size regressions are
    visible in model.star.
    """
    package = _initial_model_package_source()
    # 7811f82d5 moved the InitialModel model.star writer (write_model_star),
    # which serializes the sigma2/coverage diagnostics, to the RELION STAR owner.
    star_io = (REPO_ROOT / "relax/vdam/initial_model_io.py").read_text()
    bind = (REPO_ROOT / "relax/relion_bind/initialmodel_bind.cpp").read_text()
    tests = (REPO_ROOT / "tests/unit/initial_model/test_iteration_loop.py").read_text()
    # RELION's step-by-step M-step (the SSNR diagnostics) is the oracle path in diagnostics.
    native_mstep = (REPO_ROOT / "scripts/vdam_native_mstep.py").read_text()

    haystack = "\n".join([package, star_io, bind, tests, native_mstep])
    expected_tokens = [
        "def prepare_relion_projector_class_inputs_and_power",
        "vdam_projector_power_spectrum",
        "projector_refresh_fn: Callable[..., InitialModelState],",
        "projector_padding_factor=int(opts.padding_factor)",
        "projector_refresh_fn=projector_context.refresh",
        "current = projector_refresh_fn(current, padding_factor=projector_padding_factor)",
        "sigma2_class:  (K, S)",
        "fourier_coverage_class: (K, S)",
        "new_sigma2_class[k] = np.asarray(sigma2",
        "new_fourier_coverage_class[k] = np.asarray(fourier_coverage",
        "state.sigma2_class[k]",
        "state.fourier_coverage_class[k]",
        "test_iteration_loop_refreshes_tau2_before_estep",
    ]
    missing = [token for token in expected_tokens if token not in haystack]
    assert not missing, f"native InitialModel lost tau2 refresh/SSNR diagnostic wiring: {missing}"


def test_relion_initialmodel_reference_checker_rejects_autorefine(tmp_path):
    mod = _load_long_guard_module()

    initialmodel_dir = tmp_path / "initialmodel"
    initialmodel_dir.mkdir()
    (initialmodel_dir / "run_it008_optimiser.star").write_text(
        "\n".join(
            [
                "# RELION optimiser; version test",
                (
                    "# --o out/run --iter 8 --grad --denovo_3dref --i particles.star "
                    "--ctf --K 1 --sym C1 --flatten_solvent --zero_mask "
                    "--dont_combine_weights_via_disc --pool 3 --pad 1 "
                    "--particle_diameter 200 --oversampling 1 --healpix_order 1 "
                    "--offset_range 6 --offset_step 2 --auto_sampling --tau2_fudge 4 "
                    "--j 4 --random_seed 0"
                ),
                "",
                "data_optimiser_general",
                "_rlnCurrentIteration 8",
                "_rlnNumberOfIterations 8",
                "_rlnDoSplitRandomHalves 0",
                "_rlnDoGradientRefine 1",
                "_rlnDoAutoRefine 0",
            ]
        )
    )
    mod._assert_relion_initialmodel_reference(initialmodel_dir, expected_iter=8)

    autorefine_dir = tmp_path / "autorefine"
    autorefine_dir.mkdir()
    (autorefine_dir / "run_it008_optimiser.star").write_text(
        "\n".join(
            [
                "# RELION optimiser; version test",
                "# --i particles.star --ref reference_init_relion.mrc --o out/run --auto_refine --split_random_halves --pad 2 --iter 8",
                "",
                "data_optimiser_general",
                "_rlnCurrentIteration 8",
                "_rlnNumberOfIterations 999",
                "_rlnDoSplitRandomHalves 1",
                "_rlnDoGradientRefine 0",
                "_rlnDoAutoRefine 1",
            ]
        )
    )
    try:
        mod._assert_relion_initialmodel_reference(autorefine_dir, expected_iter=8)
    except AssertionError as exc:
        assert "--grad" in str(exc)
        assert "--denovo_3dref" in str(exc)
    else:
        raise AssertionError("auto-refine fixture was accepted as native InitialModel reference")


def test_merge_guard_dry_run_writes_reproducibility_ledger(tmp_path):
    ledger = run_guard(tier="cpu", quick=True, output_dir=tmp_path, dry_run=True)

    assert ledger["schema"] == "vdam_abinitio_merge_guard.v1"
    assert ledger["ok"] is True
    assert ledger["dry_run"] is True
    assert ledger["provenance"]["ok"] is True
    assert [command["name"] for command in ledger["commands"]] == [
        "py_compile",
        "vdam_abinitio_contracts",
        "vdam_frozen_scorecard",
        "initial_model_vdam_unit_slice",
        "em_fast_guard",
    ]

    summary_path = Path(ledger["summary_path"])
    assert summary_path.exists()
    stored = json.loads(summary_path.read_text())
    assert stored["git"]["commit"]
    assert stored["commands"][0]["skipped"] is True


def test_merge_guard_dry_run_records_a_node_without_nvidia_smi(tmp_path, monkeypatch):
    """CPU-only Slurm nodes have no nvidia-smi; the ledger records that instead of crashing."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").symlink_to(shutil.which("git"))
    monkeypatch.setenv("PATH", str(bin_dir))
    assert shutil.which("nvidia-smi") is None

    ledger = run_guard(tier="cpu", quick=True, output_dir=tmp_path / "guard", dry_run=True)

    assert ledger["ok"] is True
    assert ledger["provenance"]["ok"] is True
    assert not ledger["git"]["commit"].startswith("<failed")
    assert ledger["gpu_snapshot_start"].startswith("<nvidia-smi unavailable>")
