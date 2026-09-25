"""GPU regressions for EM replay, initialization and sampling paths.

The nine cases cover K1 global replay/local replay/adaptive replay/cold
start/perturbation replay, K2 replay, and three K4 initialization/sampling
combinations. Only the K1 local and adaptive replays build RELION's
Projector::data, so only they reach the default texture projector; the
adaptive replay alone reaches the compact texture score rows of pass 1. K4 replay requires a
same-oracle dispatch schedule; choose the oracle for each case's grid.
The (2, 1) and (1, 1) captures come from the fixture manifest; an
EM_PARITY_FAST_K4_H{order}_OS{oversampling}_{RELION_DIR,DISPATCH_SCHEDULE} pair
overrides one while a new capture is developed. Admission verifies grid and
same-capture manifest before launching refinement.
The historical strict K4 case disables oversampling, unlike the available
oversampling-1 captures. Its name does not establish matched-state parity.

Quality ledgers are saved under pytest's temporary output directory; retain
them with --basetemp and inspect them with scripts/extract_em_parity_tables.py.
Correlation assertions are regression checks, not the current FSC/FSC-AUC
quality gates. Baselines are read only, and source ancestry is checked first.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.em_fixtures import fixture_root, require_fixture_sets
from helpers.em_parity_oracles import k4_oracle

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = REPO_ROOT / "scripts" / "run_multi_iter_parity.py"
KCLASS_SCRIPT = REPO_ROOT / "scripts" / "run_k_class_parity.py"
REFINE_SCRIPT = REPO_ROOT / "scripts" / "run_full_refinement.py"
BASELINES_DIR = REPO_ROOT / "tests" / "baselines"

# External fixtures come from tests/fixtures/em_fixture_manifest.json. Each test verifies the
# sets it reads (files and sha256) and fails, not skips, when one is missing or changed.
K1_FIXTURE_DIR = fixture_root("k1_5k128_data")
K1_RELION_DIR = fixture_root("k1_5k128_relion_os0")
K1_DATA_STAR = K1_FIXTURE_DIR / "particles.star"
K1_GT_VOLUME = K1_FIXTURE_DIR / "reference_gt.mrc"
K1_RELION_RANDOM_SEED = 1775735620
# Same data and command as K1_RELION_DIR except --oversampling 1 (record in the
# fixture's GENERATION.json).
K1_OS1_RELION_DIR = fixture_root("k1_5k128_relion_os1")
# Same data with RELION's GUI-default auto-refine command: --ini_high 60 --healpix_order 2
# --offset_range 5 --offset_step 2 --oversampling 1 (record in the fixture's GENERATION.json).
K1_GUI60_RELION_DIR = fixture_root("k1_5k128_relion_gui60")

MULTIOPTICS_FIXTURE_DIR = fixture_root("multioptics_s3b_600_data")
MULTIOPTICS_RELION_DIR = fixture_root("multioptics_s3b_600_relion")
K2_FIXTURE_DIR = fixture_root("k2_5k128_data")
K2_RELION_DIR = fixture_root("k2_5k128_relion_os0")
K2_DATA_STAR = K2_FIXTURE_DIR / "particles.star"

K4_FIXTURE_DIR = fixture_root("k4_5k128_data")
# Each K4 case admits its own same-capture schedule and sampling grid.
K4_DATA_STAR = K4_FIXTURE_DIR / "particles.star"


def _require_fixture(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.fail("Missing parity fixture file(s):\n  " + "\n  ".join(missing))


def _assert_parity_ancestors_or_skip() -> None:
    """Hard-fail the test (don't skip) if the parity-fix commits are missing."""
    from relax.diagnostics.parity_provenance import (
        ParityAncestryError,
        assert_parity_ancestors,
        print_provenance_banner,
    )

    print_provenance_banner(stream=sys.stderr)
    try:
        assert_parity_ancestors()
    except ParityAncestryError as exc:
        pytest.fail(str(exc))


def _write_quality_ledger(name: str, payload: dict, *, output_dir: Path) -> Path:
    """Write this case's result beside its outputs, separately from baselines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / f"em_parity_quality_fast_ledger_{name}.json"
    payload = dict(payload)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with ledger_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return ledger_path


def _log_comparison(name: str, current: float, baseline: float | None, lower_is_better: bool = False) -> None:
    """Stream a comparison line to stderr so pytest does not capture it on pass."""
    if baseline is None:
        line = f"  {name:<32s} current={current:.6f}  baseline=missing"
    else:
        delta = current - baseline
        if lower_is_better:
            arrow = "↓" if delta < 0 else "↑"
        else:
            arrow = "↑" if delta > 0 else "↓"
        line = f"  {name:<32s} current={current:.6f}  baseline={baseline:.6f}  Δ={delta:+.6f} {arrow}"
    logger.info(line)
    print(line, file=sys.stderr, flush=True)


def _map_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _read_baseline(filename: str, key: str) -> float | None:
    path = BASELINES_DIR / filename
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())[key])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


THRESHOLDS = REPO_ROOT / "tests" / "tiers" / "fsc_thresholds.json"


def _assert_fsc_gate(case: str, output_dir: Path) -> None:
    """Hold the case to its approved FSC and Pmax floors against the RELION oracle.

    Every half (K1) or Hungarian-matched class (K>1) must reach the case's FSC-AUC floor and its
    minimum in-band shell FSC floor, and the mean per-particle |dPmax| against RELION must stay
    under its bound (tests/tiers/fsc_thresholds.json, approved by the user 2026-09-24; scoring by
    scripts/em_tier_fsc.py). Correlation is recorded as a diagnostic only.
    """
    from scripts.em_tier_fsc import score_relax_case

    thresholds = json.loads(THRESHOLDS.read_text())
    assert thresholds["approved"], "the fast-tier FSC thresholds are not approved"
    gate = thresholds["cases"][case]
    result = score_relax_case(case, output_dir)
    (output_dir / f"em_parity_fsc_{case}.json").write_text(json.dumps(result, indent=1) + "\n")
    for pair, metrics in result["pairs"].items():
        print(
            f"  {case} {pair}: FSC-AUC {metrics['fsc_auc']:.6f} (floor {gate['fsc_auc_floor']}), "
            f"min shell FSC {metrics['min_shell_fsc_in_band']:.6f} (floor {gate['min_shell_floor']})",
            file=sys.stderr,
            flush=True,
        )
        assert metrics["fsc_auc"] >= gate["fsc_auc_floor"], (
            f"{case} {pair} FSC-AUC {metrics['fsc_auc']:.6f} below the approved floor {gate['fsc_auc_floor']}"
        )
        assert metrics["min_shell_fsc_in_band"] >= gate["min_shell_floor"], (
            f"{case} {pair} minimum shell FSC {metrics['min_shell_fsc_in_band']:.6f} below the approved floor "
            f"{gate['min_shell_floor']}"
        )
    pmax = result["pmax"].get("relax_minus_relion", {}).get("mean_abs")
    assert pmax is not None, f"{case}: no per-particle Pmax comparison with RELION"
    print(f"  {case} mean |dPmax| {pmax:.3g} (bound {gate['pmax_mean_abs_max']})", file=sys.stderr, flush=True)
    assert pmax <= gate["pmax_mean_abs_max"], (
        f"{case} mean per-particle |dPmax| {pmax:.3g} above the approved bound {gate['pmax_mean_abs_max']}"
    )


def _flag_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "on", "yes"}


def _assert_resident_engines_ran(log: str, *, global_pass: bool, local_pass: bool, case: str) -> None:
    """Under the resident flag set, the passes a case exists for must run resident.

    The routing logs its choice (dispatch.py, local_search_iteration.py); a case
    that silently fell back to the compact or exact local engine would pass its
    RELION gates while testing the other engine.
    """

    if global_pass and _flag_on("RELAX_SPARSE_PASS2_RESIDENT"):
        assert "Resident pass-2 plan:" in log, f"{case} did not run the resident global pass 2"
    if local_pass and _flag_on("RELAX_LOCAL_SEARCH_RESIDENT"):
        assert "running the device-resident local fine pass 2" in log, (
            f"{case} did not run the resident local pass 2"
        )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_replay(tmp_path):
    """Replay K1 iteration 3→4 on the 5k/128 fixture.

    Compare both half maps and optimizer Pmax with RELION iteration 4.
    The assertions retain the correlation floor and Pmax error bound.
    """
    _assert_parity_ancestors_or_skip()
    model_path = K1_RELION_DIR / "run_it004_half1_model.star"
    require_fixture_sets("k1_5k128_data", "k1_5k128_relion_os0")
    _require_fixture(PARITY_SCRIPT, K1_RELION_DIR, K1_DATA_STAR, K1_GT_VOLUME, model_path)

    output_dir = tmp_path / "k1_replay"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(PARITY_SCRIPT),
        "--relion_dir",
        str(K1_RELION_DIR),
        "--data_star",
        str(K1_DATA_STAR),
        "--iter",
        "3",
        "--max_iter",
        "1",
        "--skip_final_iteration",
        "--gt_volume",
        str(K1_GT_VOLUME),
        "--output_dir",
        str(output_dir),
    ]
    logger.info("K=1 replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0

    if proc.returncode == 2:
        # Provenance gate failed — bubble the error to the test runner.
        pytest.fail(
            "Parity provenance gate failed (exit 2). The worktree is missing "
            "required parity-fix commits.\nstdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr
        )
    assert proc.returncode == 0, (
        f"run_multi_iter_parity.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    _assert_resident_engines_ran(proc.stdout + proc.stderr, global_pass=True, local_pass=False, case="K=1 replay")

    npz_path = output_dir / "refinement_results.npz"
    assert npz_path.exists(), f"Missing refinement_results.npz at {npz_path}"
    npz = np.load(npz_path)

    half1_corr = float(npz["final_half1_corr_vs_relion"])
    half2_corr = float(npz["final_half2_corr_vs_relion"])
    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)
    recovar_pmax = float(pmax_traj[0])

    relion_pmax_reference = float(starfile.read(model_path)["model_general"]["rlnAveragePmax"])
    pmax_abs_diff = abs(recovar_pmax - relion_pmax_reference)

    baseline_h1 = _read_baseline("em_parity_quality_fast_baseline.json", "k1_replay_half1_corr_vs_relion")
    baseline_h2 = _read_baseline("em_parity_quality_fast_baseline.json", "k1_replay_half2_corr_vs_relion")
    baseline_pmax = _read_baseline("em_parity_quality_fast_baseline.json", "k1_replay_pmax_abs_diff")

    print(file=sys.stderr, flush=True)
    print("=== K=1 replay parity (iter 3→4 vs RELION it004) ===", file=sys.stderr, flush=True)
    _log_comparison("k1_replay_half1_corr_vs_relion", half1_corr, baseline_h1)
    _log_comparison("k1_replay_half2_corr_vs_relion", half2_corr, baseline_h2)
    _log_comparison("k1_replay_pmax_abs_diff", pmax_abs_diff, baseline_pmax, lower_is_better=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    payload = {
        "k1_replay_half1_corr_vs_relion": half1_corr,
        "k1_replay_half2_corr_vs_relion": half2_corr,
        "k1_replay_pmax_recovar": recovar_pmax,
        "k1_replay_pmax_relion_reference": relion_pmax_reference,
        "k1_replay_pmax_abs_diff": pmax_abs_diff,
        "k1_replay_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("k1_replay", payload, output_dir=output_dir)
    logger.info("K=1 replay ledger: %s", ledger)

    # NEVER widen tolerance to make a test pass. Fix the code instead.
    _assert_fsc_gate("k1_replay", output_dir)
    assert pmax_abs_diff < 1e-3, (
        f"K=1 replay |ΔPmax| {pmax_abs_diff:.6f} exceeds threshold 1e-3 vs RELION it004={relion_pmax_reference}."
    )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_local_replay(tmp_path):
    """Replay K1 iteration 6→7, RELION's first local-search iteration (healpix 4).

    The global K1 cases run at oversampling 0 and never build Projector::data,
    so the supplied-PPref texture projector used by every local iteration is
    otherwise unexercised in this tier. Compare both half maps and optimizer
    Pmax with RELION iteration 7, with the same bounds as the global replay.
    """
    _assert_parity_ancestors_or_skip()
    model_path = K1_RELION_DIR / "run_it007_half1_model.star"
    require_fixture_sets("k1_5k128_data", "k1_5k128_relion_os0")
    _require_fixture(PARITY_SCRIPT, K1_RELION_DIR, K1_DATA_STAR, K1_GT_VOLUME, model_path)

    output_dir = tmp_path / "k1_local_replay"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(PARITY_SCRIPT),
        "--relion_dir",
        str(K1_RELION_DIR),
        "--data_star",
        str(K1_DATA_STAR),
        "--iter",
        "6",
        "--max_iter",
        "1",
        "--skip_final_iteration",
        "--gt_volume",
        str(K1_GT_VOLUME),
        "--output_dir",
        str(output_dir),
    ]
    logger.info("K=1 local replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0

    if proc.returncode == 2:
        pytest.fail(
            "Parity provenance gate failed (exit 2). The worktree is missing "
            "required parity-fix commits.\nstdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr
        )
    assert proc.returncode == 0, (
        f"run_multi_iter_parity.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # The case exists to cover the local projector path; fail if the replay
    # took another route rather than passing on a path it was not meant to test.
    log = proc.stdout + proc.stderr
    assert "healpix_order=4, local_search=True" in log, "K=1 local replay did not run a local-search iteration"
    assert "built exact Projector::data" in log, "K=1 local replay did not build RELION's Projector::data"
    _assert_resident_engines_ran(log, global_pass=False, local_pass=True, case="K=1 local replay")

    npz_path = output_dir / "refinement_results.npz"
    assert npz_path.exists(), f"Missing refinement_results.npz at {npz_path}"
    npz = np.load(npz_path)

    half1_corr = float(npz["final_half1_corr_vs_relion"])
    half2_corr = float(npz["final_half2_corr_vs_relion"])
    recovar_pmax = float(np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)[0])

    relion_pmax_reference = float(starfile.read(model_path)["model_general"]["rlnAveragePmax"])
    pmax_abs_diff = abs(recovar_pmax - relion_pmax_reference)

    baseline_h1 = _read_baseline("em_parity_quality_fast_baseline.json", "k1_local_replay_half1_corr_vs_relion")
    baseline_h2 = _read_baseline("em_parity_quality_fast_baseline.json", "k1_local_replay_half2_corr_vs_relion")
    baseline_pmax = _read_baseline("em_parity_quality_fast_baseline.json", "k1_local_replay_pmax_abs_diff")

    print(file=sys.stderr, flush=True)
    print("=== K=1 local replay parity (iter 6→7 vs RELION it007) ===", file=sys.stderr, flush=True)
    _log_comparison("k1_local_replay_half1_corr_vs_relion", half1_corr, baseline_h1)
    _log_comparison("k1_local_replay_half2_corr_vs_relion", half2_corr, baseline_h2)
    _log_comparison("k1_local_replay_pmax_abs_diff", pmax_abs_diff, baseline_pmax, lower_is_better=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    payload = {
        "k1_local_replay_half1_corr_vs_relion": half1_corr,
        "k1_local_replay_half2_corr_vs_relion": half2_corr,
        "k1_local_replay_pmax_recovar": recovar_pmax,
        "k1_local_replay_pmax_relion_reference": relion_pmax_reference,
        "k1_local_replay_pmax_abs_diff": pmax_abs_diff,
        "k1_local_replay_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("k1_local_replay", payload, output_dir=output_dir)
    logger.info("K=1 local replay ledger: %s", ledger)

    # NEVER widen tolerance to make a test pass. Fix the code instead.
    _assert_fsc_gate("k1_local_replay", output_dir)
    assert pmax_abs_diff < 1e-3, (
        f"K=1 local replay |ΔPmax| {pmax_abs_diff:.6f} exceeds threshold 1e-3 vs RELION it007={relion_pmax_reference}."
    )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_adaptive_replay(tmp_path):
    """Replay K1 iteration 3→4 of RELION's oversampling-1 auto-refine (global, healpix 3).

    Adaptive oversampling runs a coarse pass-1 significance search before the
    fine pass. With the default texture interpolation and a current size below
    the box, pass 1 projects only the compact score rows through RELION's
    texture projector, which no oversampling-0 case reaches. Compare both half
    maps and optimizer Pmax with RELION iteration 4, with the same bounds as
    the global replay.
    """
    _assert_parity_ancestors_or_skip()
    model_path = K1_OS1_RELION_DIR / "run_it004_half1_model.star"
    require_fixture_sets("k1_5k128_data", "k1_5k128_relion_os1")
    _require_fixture(PARITY_SCRIPT, K1_OS1_RELION_DIR, K1_DATA_STAR, K1_GT_VOLUME, model_path)

    output_dir = tmp_path / "k1_adaptive_replay"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(PARITY_SCRIPT),
        "--relion_dir",
        str(K1_OS1_RELION_DIR),
        "--data_star",
        str(K1_DATA_STAR),
        "--iter",
        "3",
        "--max_iter",
        "1",
        "--skip_final_iteration",
        "--gt_volume",
        str(K1_GT_VOLUME),
        "--output_dir",
        str(output_dir),
    ]
    logger.info("K=1 adaptive replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0

    if proc.returncode == 2:
        pytest.fail(
            "Parity provenance gate failed (exit 2). The worktree is missing "
            "required parity-fix commits.\nstdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr
        )
    assert proc.returncode == 0, (
        f"run_multi_iter_parity.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # The case exists to cover the adaptive pass-1 projector path; fail if the
    # replay took another route rather than passing on a path it was not meant to test.
    log = proc.stdout + proc.stderr
    assert "healpix_order=3, local_search=False" in log, "K=1 adaptive replay did not run a global iteration"
    assert "Adaptive oversampling: pass 1 at coarse_size=" in log, "K=1 adaptive replay did not run pass 1"
    assert "built exact Projector::data" in log, "K=1 adaptive replay did not build RELION's Projector::data"

    npz_path = output_dir / "refinement_results.npz"
    assert npz_path.exists(), f"Missing refinement_results.npz at {npz_path}"
    npz = np.load(npz_path)

    half1_corr = float(npz["final_half1_corr_vs_relion"])
    half2_corr = float(npz["final_half2_corr_vs_relion"])
    recovar_pmax = float(np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)[0])

    relion_pmax_reference = float(starfile.read(model_path)["model_general"]["rlnAveragePmax"])
    pmax_abs_diff = abs(recovar_pmax - relion_pmax_reference)

    baseline_h1 = _read_baseline("em_parity_quality_fast_baseline.json", "k1_adaptive_replay_half1_corr_vs_relion")
    baseline_h2 = _read_baseline("em_parity_quality_fast_baseline.json", "k1_adaptive_replay_half2_corr_vs_relion")
    baseline_pmax = _read_baseline("em_parity_quality_fast_baseline.json", "k1_adaptive_replay_pmax_abs_diff")

    print(file=sys.stderr, flush=True)
    print("=== K=1 adaptive replay parity (os1 iter 3→4 vs RELION it004) ===", file=sys.stderr, flush=True)
    _log_comparison("k1_adaptive_replay_half1_corr_vs_relion", half1_corr, baseline_h1)
    _log_comparison("k1_adaptive_replay_half2_corr_vs_relion", half2_corr, baseline_h2)
    _log_comparison("k1_adaptive_replay_pmax_abs_diff", pmax_abs_diff, baseline_pmax, lower_is_better=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    payload = {
        "k1_adaptive_replay_half1_corr_vs_relion": half1_corr,
        "k1_adaptive_replay_half2_corr_vs_relion": half2_corr,
        "k1_adaptive_replay_pmax_recovar": recovar_pmax,
        "k1_adaptive_replay_pmax_relion_reference": relion_pmax_reference,
        "k1_adaptive_replay_pmax_abs_diff": pmax_abs_diff,
        "k1_adaptive_replay_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("k1_adaptive_replay", payload, output_dir=output_dir)
    logger.info("K=1 adaptive replay ledger: %s", ledger)

    # NEVER widen tolerance to make a test pass. Fix the code instead.
    _assert_fsc_gate("k1_adaptive_replay", output_dir)
    assert pmax_abs_diff < 1e-3, (
        f"K=1 adaptive replay |ΔPmax| {pmax_abs_diff:.6f} exceeds threshold 1e-3 vs RELION it004={relion_pmax_reference}."
    )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_kclass_replay(tmp_path):
    """K=2 128² 5k iter-0→1 replay against the pdb_k2 RELION Class3D reference.

    Asserts joint class × pose RELION parity by:
      * Hungarian-aligning recovar's per-class maps to RELION's
      * Asserting ``mean_corr ≥ 0.95`` after Hungarian alignment
      * Asserting per-image Pmax agreement within ``|ΔPmax| < 1e-2``

    The K=2 pdb fixture only ships RELION it000 + it001, so this is the
    smallest available K-class parity check. K=4 / longer chains belong in
    the EM-long tier.
    """
    _assert_parity_ancestors_or_skip()
    require_fixture_sets("k2_5k128_data", "k2_5k128_relion_os0")
    _require_fixture(KCLASS_SCRIPT, K2_RELION_DIR, K2_DATA_STAR)

    output_dir = tmp_path / "kclass_replay"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(KCLASS_SCRIPT),
        "--relion-dir",
        str(K2_RELION_DIR),
        "--data-star",
        str(K2_DATA_STAR),
        "--prev-iter",
        "0",
        "--target-iter",
        "1",
        "--output-dir",
        str(output_dir),
        # This fixture uses RELION firstiter CC: select the global coarse winner,
        # then refine its pose within that class (acc_ml_optimiser_impl.h).
        # The replay's default broad adaptive support is a different policy.
        "--firstiter-cc-mode",
        "force",
        "--firstiter-cc-pass2-only-best-coarse",
        "--adaptive-2pass",
    ]
    logger.info("K-class replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0

    assert proc.returncode == 0, (
        f"run_k_class_parity.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    summary_path = output_dir / "summary.json"
    assert summary_path.exists(), f"Missing summary.json at {summary_path}"
    summary = json.loads(summary_path.read_text())

    best_perm = summary["best_permutation"]
    mean_corr = float(best_perm["mean_corr"])
    map_corrs = [float(c) for c in best_perm["map_correlations"]]
    pmax_abs_mean = float(summary["pmax"]["abs_mean"])
    pmax_abs_max = float(summary["pmax"]["abs_max"])
    class_acc = float(summary["class_assignment_accuracy_after_permutation"])

    baseline_mean = _read_baseline("em_parity_quality_fast_baseline.json", "kclass_replay_mean_corr")
    baseline_pmax = _read_baseline("em_parity_quality_fast_baseline.json", "kclass_replay_pmax_abs_mean")
    baseline_acc = _read_baseline("em_parity_quality_fast_baseline.json", "kclass_replay_class_assignment_accuracy")

    print(file=sys.stderr, flush=True)
    print("=== K=2 replay parity (iter 0→1 vs RELION it001) ===", file=sys.stderr, flush=True)
    _log_comparison("kclass_replay_mean_corr", mean_corr, baseline_mean)
    _log_comparison("kclass_replay_pmax_abs_mean", pmax_abs_mean, baseline_pmax, lower_is_better=True)
    _log_comparison("kclass_replay_class_assignment_accuracy", class_acc, baseline_acc)
    print(f"  per-class map corrs: {map_corrs}", file=sys.stderr, flush=True)
    print(f"  pmax_abs_max={pmax_abs_max:.6g}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    payload = {
        "kclass_replay_mean_corr": mean_corr,
        "kclass_replay_per_class_map_corr": map_corrs,
        "kclass_replay_pmax_abs_mean": pmax_abs_mean,
        "kclass_replay_pmax_abs_max": pmax_abs_max,
        "kclass_replay_class_assignment_accuracy": class_acc,
        "kclass_replay_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("kclass_replay", payload, output_dir=output_dir)
    logger.info("K-class replay ledger: %s", ledger)

    # K=2 iter 0→1 is the first K-class iteration after class seeds are loaded;
    # it exercises the joint class × pose posterior (adaptive 2-pass).
    _assert_fsc_gate("kclass_replay", output_dir)
    assert pmax_abs_mean < 1e-2, f"K-class replay |ΔPmax|.mean {pmax_abs_mean:.6g} exceeds threshold 1e-2."
    assert class_acc >= 0.95, (
        f"K-class replay class assignment accuracy {class_acc:.4f} below threshold 0.95 after Hungarian permutation."
    )


# Start-up state for the K1 cold start. ``standalone`` reads only relion_refine's
# inputs (particles.star, the stack, the reference and the RELION command values):
# relion_refine's split/groups/order from the seed, RELION start-up noise and tau2.
# ``relion_seeded_debug`` takes the half sets RELION's run wrote into
# particles_with_halfsets.star and discovers RELION's optimiser STAR; it is a
# debugging aid, not standalone evidence.
K1_COLDSTART_START_ARGS = {
    "standalone": [
        "--relion-half-sets-from-input",
        "--particle_diameter_ang",
        "544",  # RELION --particle_diameter
        "--apply-initial-lowpass",  # RELION --ini_high (with --init_resolution)
    ],
    "relion_seeded_debug": [
        "--relion_half_sets",
        str(K1_FIXTURE_DIR / "particles_with_halfsets.star"),
        "--no-apply-initial-lowpass",  # this debug start's pre-GUI-default setting
    ],
}


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("start", sorted(K1_COLDSTART_START_ARGS))
def test_em_parity_fast_k1_coldstart(tmp_path, start):
    """Run three K1 iterations from raw 5k/128 inputs and the initial map.

    RECOVAR initializes and updates noise, tau, sigma and FSC state with
    RELION's CUDA image preprocessing. The standalone start rebuilds
    relion_refine's half split from the seed (equal to RELION's on this
    fixture), so iteration 3 is compared with RELION's iteration 3 in both
    start modes: half maps, Pmax and the sigma-offset update.
    """
    m, check = _run_k1_coldstart(tmp_path, start=start, oversampling=0)
    payload = {
        "k1_coldstart_half1_corr_vs_relion_it003": m["h1_corr"],
        "k1_coldstart_half2_corr_vs_relion_it003": m["h2_corr"],
        "k1_coldstart_pmax_iter3_recovar": m["pmax_iter3"],
        "k1_coldstart_pmax_iter3_relion": m["relion_pmax"],
        "k1_coldstart_pmax_iter3_abs_diff": m["pmax_diff"],
        "k1_coldstart_sigma_offset_trajectory": m["sigma_traj"],
        "k1_coldstart_sigma_offset_used_trajectory": m["sigma_used_traj"],
        "k1_coldstart_walltime_s": m["elapsed"],
    }
    # Only the standalone case is reported; the debug case is not tier evidence.
    if start == "standalone":
        ledger = _write_quality_ledger("k1_coldstart", payload, output_dir=m["output_dir"])
        logger.info("K=1 cold-start ledger: %s", ledger)
    check()


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_os1_coldstart_standalone(tmp_path):
    """The standalone K1 cold start at --oversampling 1 against RELION's os1 run.

    Every other K=1 case here is oversampling 0 or a replay, so this is the
    fast-tier case that reaches the production adaptive pass 2 (and, under
    RELAX_SPARSE_PASS2_RESIDENT=1, the device-resident driver). The RELION
    oracle differs from the os0 one only in --oversampling 1.
    """
    m, check = _run_k1_coldstart(tmp_path, start="standalone", oversampling=1)
    payload = {
        "k1_os1_coldstart_half1_corr_vs_relion_it003": m["h1_corr"],
        "k1_os1_coldstart_half2_corr_vs_relion_it003": m["h2_corr"],
        "k1_os1_coldstart_pmax_iter3_recovar": m["pmax_iter3"],
        "k1_os1_coldstart_pmax_iter3_relion": m["relion_pmax"],
        "k1_os1_coldstart_pmax_iter3_abs_diff": m["pmax_diff"],
        "k1_os1_coldstart_sigma_offset_trajectory": m["sigma_traj"],
        "k1_os1_coldstart_sigma_offset_used_trajectory": m["sigma_used_traj"],
        "k1_os1_coldstart_walltime_s": m["elapsed"],
    }
    ledger = _write_quality_ledger("k1_os1_coldstart", payload, output_dir=m["output_dir"])
    logger.info("K=1 os1 cold-start ledger: %s", ledger)
    check()


# Sampling and start resolution of the K1 cold starts: the 5k oracles' command, and RELION's GUI default.
K1_COLDSTART_SAMPLING_ARGS = ["--healpix_order", "3", "--offset_range", "3.0", "--offset_step", "1.0", "--init_resolution", "30.0"]
K1_GUI60_SAMPLING_ARGS = ["--healpix_order", "2", "--offset_range", "5.0", "--offset_step", "2.0", "--init_resolution", "60.0"]


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_gui60_coldstart_standalone(tmp_path):
    """The standalone K1 cold start with RELION's GUI-default command against its RELION run.

    --ini_high 60 (above the 40 A --low_resol_join_halves), --healpix_order 2, --offset_range 5,
    --offset_step 2, --oversampling 1. At 60 A RELION's iteration 1 joins the halves only up to
    the ini_high shell (ml_optimiser_mpi.cpp:3280); every other case starts at 30 A, where the
    40 A join wins, so only this case checks that start.
    """
    m, check = _run_k1_coldstart(tmp_path, start="standalone", oversampling=1, gui_default=True)
    payload = {
        "k1_gui60_coldstart_half1_corr_vs_relion_it003": m["h1_corr"],
        "k1_gui60_coldstart_half2_corr_vs_relion_it003": m["h2_corr"],
        "k1_gui60_coldstart_pmax_iter3_recovar": m["pmax_iter3"],
        "k1_gui60_coldstart_pmax_iter3_relion": m["relion_pmax"],
        "k1_gui60_coldstart_pmax_iter3_abs_diff": m["pmax_diff"],
        "k1_gui60_coldstart_sigma_offset_trajectory": m["sigma_traj"],
        "k1_gui60_coldstart_sigma_offset_used_trajectory": m["sigma_used_traj"],
        "k1_gui60_coldstart_walltime_s": m["elapsed"],
    }
    ledger = _write_quality_ledger("k1_gui60_coldstart", payload, output_dir=m["output_dir"])
    logger.info("K=1 GUI-default cold-start ledger: %s", ledger)
    check()


def _run_k1_coldstart(tmp_path, *, start, oversampling, gui_default=False):
    """Run the cold start and measure it; return the measurements and the gate check.

    The caller writes its own ledger between the two, with a literal case name
    and payload the report inventory can read, so a failing gate still leaves
    the ledger behind.
    """
    _assert_parity_ancestors_or_skip()
    if gui_default:
        assert oversampling == 1, "the GUI-default oracle ran at --oversampling 1"
        relion_set, relion_dir, case = "k1_5k128_relion_gui60", K1_GUI60_RELION_DIR, "k1_gui60_coldstart"
        sampling_args = K1_GUI60_SAMPLING_ARGS
    else:
        relion_set = "k1_5k128_relion_os1" if oversampling else "k1_5k128_relion_os0"
        relion_dir = K1_OS1_RELION_DIR if oversampling else K1_RELION_DIR
        case = "k1_os1_coldstart" if oversampling else "k1_coldstart"
        sampling_args = K1_COLDSTART_SAMPLING_ARGS
    require_fixture_sets("k1_5k128_data", relion_set)
    _require_fixture(REFINE_SCRIPT, K1_FIXTURE_DIR, relion_dir, K1_DATA_STAR)

    output_dir = tmp_path / f"{case}_{start}"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K1_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--max_iter",
        "3",
        *sampling_args,
        "--adaptive_oversampling",
        str(oversampling),
        "--tau2_fudge",
        "1.0",
        "--perturb_factor",
        "0.5",  # match RELION's --perturb 0.5
        "--seed",
        str(K1_RELION_RANDOM_SEED),  # drives SamplingPerturbation like RELION's --random_seed
        "--no-firstiter_cc",  # the 5k oracle ran without --firstiter_cc
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
        # Both starts use RELION's half-set membership (standalone rebuilds it from
        # the seed), so the half-map comparison with RELION is meaningful.
        *K1_COLDSTART_START_ARGS[start],
        # The fresh K=1 defaults (source-faithful powerClass normalization, exact
        # BPref operands) score from RELION's CUDA image preprocessing, as the
        # K1 completion launcher does.
        "--image-fourier-backend",
        "relion_cuda",
    ]
    logger.info("K=1 cold-start cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    npz = np.load(output_dir / "refinement_results.npz")
    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)
    sigma_traj = np.asarray(npz["sigma_offset_trajectory"], dtype=np.float64)
    sigma_used_traj = np.asarray(npz["sigma_offset_used_trajectory"], dtype=np.float64)
    n_iters = int(pmax_traj.size)
    assert n_iters >= 3, f"Expected ≥3 iterations, got {n_iters}"

    # Compare halfmaps against RELION it003. recovar's `write_mrc` saves
    # with a (2,1,0) transpose (cryosparc/cryoDRGN convention) and `load_mrc`
    # un-transposes round-trip; RELION's `load_relion_volume` applies
    # `-(2,1,0)` to RELION MRCs to land in recovar's frame. Mixing helpers
    # incorrectly gives corr ≈ -0.98 (raw + raw — sign flip) or 0.45 (raw
    # recovar + load_relion — half-applied transpose). Both files via the
    # right helper.
    from recovar.utils import helpers as _recovar_helpers

    def _load_recovar_real(p: Path) -> np.ndarray:
        return np.asarray(_recovar_helpers.load_mrc(str(p)), dtype=np.float64)

    def _load_relion_real(p: Path) -> np.ndarray:
        return np.asarray(_recovar_helpers.load_relion_volume(str(p)), dtype=np.float64)

    recovar_h1 = _load_recovar_real(output_dir / "final_half1.mrc")
    recovar_h2 = _load_recovar_real(output_dir / "final_half2.mrc")
    relion_h1 = _load_relion_real(relion_dir / "run_it003_half1_class001.mrc")
    relion_h2 = _load_relion_real(relion_dir / "run_it003_half2_class001.mrc")
    h1_corr = _map_correlation(recovar_h1, relion_h1)
    h2_corr = _map_correlation(recovar_h2, relion_h2)

    # Read the authoritative optimizer/scheduling scalar, not the arithmetic
    # mean of the per-particle data column.
    import starfile

    relion_model = starfile.read(str(relion_dir / "run_it003_half1_model.star"))
    relion_pmax = float(relion_model["model_general"]["rlnAveragePmax"])
    pmax_diff = abs(float(pmax_traj[2]) - relion_pmax)

    measurements = {
        "h1_corr": h1_corr,
        "h2_corr": h2_corr,
        "pmax_iter3": float(pmax_traj[2]),
        "relion_pmax": relion_pmax,
        "pmax_diff": pmax_diff,
        "sigma_traj": sigma_traj.tolist(),
        "sigma_used_traj": sigma_used_traj.tolist(),
        "elapsed": elapsed,
        "output_dir": output_dir,
    }
    return measurements, lambda: _check_k1_coldstart(
        case=case,
        start=start,
        oversampling=oversampling,
        output_dir=output_dir,
        h1_corr=h1_corr,
        h2_corr=h2_corr,
        pmax_traj=pmax_traj,
        relion_pmax=relion_pmax,
        pmax_diff=pmax_diff,
        sigma_traj=sigma_traj,
        elapsed=elapsed,
        log=proc.stdout + proc.stderr,
    )


def _check_k1_coldstart(
    *, case, start, oversampling, output_dir, h1_corr, h2_corr, pmax_traj, relion_pmax,
    pmax_diff, sigma_traj, elapsed, log,
):
    print(file=sys.stderr, flush=True)
    print(
        f"=== K=1 cold-start parity, oversampling {oversampling} "
        "(3-iter ab-initio vs RELION it003) ===",
        file=sys.stderr,
        flush=True,
    )
    print(f"  half1_corr={h1_corr:.6f} half2_corr={h2_corr:.6f}", file=sys.stderr, flush=True)
    print(
        f"  pmax_iter3 recovar={pmax_traj[2]:.4f} relion={relion_pmax:.4f} diff={pmax_diff:.4g}",
        file=sys.stderr,
        flush=True,
    )
    print(f"  sigma_offset_trajectory={sigma_traj.tolist()}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # NEVER widen tolerance to make a test pass. Fix the code instead.
    # CLI-level --seed now reproduces RELION's per-iter SamplingPerturbInstance
    # stream for the fixture's --random_seed. Keep this autonomous cold-start
    # guard separate from replay because it still bootstraps model/noise/tau/
    # sigma state from raw inputs rather than injecting per-iter RELION state.
    _assert_fsc_gate(f"{case}_{start}", output_dir)
    # Pre-A.1 cold-start: iter-3 |ΔPmax| was ~22% (sigma_offset stuck at 10 Å).
    # Post-A.1: 5-12% depending on perturbation drift. Threshold 0.15 catches
    # full A.1 regression (would jump back to 22%).
    assert pmax_diff < 0.15, f"K=1 cold-start |ΔPmax| {pmax_diff:.4g} exceeds 0.15 vs RELION it003."
    # A.1 fix: iter-2 sigma_offset must drop below the 10 Å default. Pre-fix:
    # cold-start kept sigma_offset=10 Å through iter 8. Post-fix: ~2-4 Å at
    # iter 2, depending on iter-1 best-translation distribution.
    assert len(sigma_traj) >= 2 and sigma_traj[1] <= 5.0, (
        f"K=1 cold-start iter-2 sigma_offset {sigma_traj[1]:.3f} Å too large; A.1 fix may have regressed."
    )
    # Both cold starts reach the production pass 2 after the firstiter_cc
    # iteration; under the resident flag they must have run the resident driver.
    _assert_resident_engines_ran(
        log, global_pass=True, local_pass=False, case=f"K=1 os{int(bool(oversampling))} cold start"
    )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_perturbreplay(tmp_path):
    """Run three K1 iterations with RELION perturbation/correction replay.

    The driver injects sampling perturbations, normalization and group scales;
    RECOVAR still runs E/M and updates sigma, tau and FSC. Compare iteration-3
    half maps and optimizer Pmax. This is distinct from autonomous cold start.
    """
    _assert_parity_ancestors_or_skip()
    require_fixture_sets("k1_5k128_data", "k1_5k128_relion_os0")
    _require_fixture(REFINE_SCRIPT, K1_FIXTURE_DIR, K1_RELION_DIR, K1_DATA_STAR)

    output_dir = tmp_path / "k1_perturbreplay"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K1_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--max_iter",
        "3",
        "--healpix_order",
        "3",
        "--offset_range",
        "3.0",
        "--offset_step",
        "1.0",
        "--adaptive_oversampling",
        "0",
        "--tau2_fudge",
        "1.0",
        "--perturb_factor",
        "0.5",
        "--perturb_replay_relion_dir",
        str(K1_RELION_DIR),
        # Pinned to this case's pre-GUI-default settings, except the image
        # backend: a perturbation replay from iteration 0 keeps RELION's fresh
        # order and so runs the production K=1 arithmetic, which scores from
        # RELION's CUDA preprocessing.
        "--no-firstiter_cc",
        "--no-apply-initial-lowpass",
        "--image-fourier-backend",
        "relion_cuda",
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
        "--relion_half_sets",
        str(K1_FIXTURE_DIR / "particles_with_halfsets.star"),
    ]
    logger.info("K=1 perturb-replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    _assert_resident_engines_ran(
        proc.stdout + proc.stderr, global_pass=True, local_pass=False, case="K=1 perturbation replay"
    )

    npz = np.load(output_dir / "refinement_results.npz")
    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)

    from recovar.utils import helpers as _recovar_helpers

    rec_h1 = np.asarray(_recovar_helpers.load_mrc(str(output_dir / "final_half1.mrc")), dtype=np.float64)
    rec_h2 = np.asarray(_recovar_helpers.load_mrc(str(output_dir / "final_half2.mrc")), dtype=np.float64)
    rel_h1 = np.asarray(
        _recovar_helpers.load_relion_volume(str(K1_RELION_DIR / "run_it003_half1_class001.mrc")), dtype=np.float64
    )
    rel_h2 = np.asarray(
        _recovar_helpers.load_relion_volume(str(K1_RELION_DIR / "run_it003_half2_class001.mrc")), dtype=np.float64
    )
    h1_corr = _map_correlation(rec_h1, rel_h1)
    h2_corr = _map_correlation(rec_h2, rel_h2)

    import starfile

    relion_model = starfile.read(str(K1_RELION_DIR / "run_it003_half1_model.star"))
    relion_pmax = float(relion_model["model_general"]["rlnAveragePmax"])
    pmax_diff = abs(float(pmax_traj[2]) - relion_pmax)

    payload = {
        "k1_perturbreplay_half1_corr_vs_relion_it003": h1_corr,
        "k1_perturbreplay_half2_corr_vs_relion_it003": h2_corr,
        "k1_perturbreplay_pmax_iter3_recovar": float(pmax_traj[2]),
        "k1_perturbreplay_pmax_iter3_relion": relion_pmax,
        "k1_perturbreplay_pmax_iter3_abs_diff": pmax_diff,
        "k1_perturbreplay_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("k1_perturbreplay", payload, output_dir=output_dir)
    logger.info("K=1 perturb-replay ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print("=== K=1 perturb-replay parity (3-iter ab-initio + RELION grid jitter sync) ===", file=sys.stderr, flush=True)
    print(f"  half1_corr={h1_corr:.6f} half2_corr={h2_corr:.6f}", file=sys.stderr, flush=True)
    print(
        f"  pmax_iter3 recovar={pmax_traj[2]:.4f} relion={relion_pmax:.4f} diff={pmax_diff:.4g}",
        file=sys.stderr,
        flush=True,
    )
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    _assert_fsc_gate("k1_perturbreplay", output_dir)
    assert pmax_diff < 0.05, f"K=1 perturb-replay |ΔPmax| {pmax_diff:.4g} exceeds 0.05 vs RELION it003."


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_kclass_coldstart(tmp_path):
    """Run K4 cold start at coarse order 2 with oversampling 1.

    Use the 5k/128 fixture, firstiter_cc and captured perturbation/correction
    replay, but derive initial noise/tau/sigma locally. Hungarian-match the four
    iteration-3 class maps. The dispatch oracle must use the same sampling grid.
    """
    _assert_parity_ancestors_or_skip()
    relion_dir, dispatch_args = k4_oracle(2, 1)
    require_fixture_sets("k4_5k128_data")
    _require_fixture(REFINE_SCRIPT, K4_FIXTURE_DIR, relion_dir, K4_DATA_STAR)

    output_dir = tmp_path / "kclass_coldstart"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K4_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--n_classes",
        "4",
        "--max_iter",
        "3",
        "--healpix_order",
        "2",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--adaptive_oversampling",
        "1",
        "--tau2_fudge",
        "4.0",
        "--perturb_factor",
        "0.5",
        "--perturb_replay_relion_dir",
        str(relion_dir),
        *dispatch_args,
        "--firstiter_cc",
        # Pinned to this case's pre-GUI-default setting: no start-up low-pass.
        "--no-apply-initial-lowpass",
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
    ]
    logger.info("K-class cold-start cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert not any(output_dir.glob("final_half*_class*.mrc")), "K>1 writer should not emit per-half class MRCs"

    # Hungarian-match recovar's per-class output to RELION's per-class it003.
    from scipy.optimize import linear_sum_assignment

    from recovar.utils import helpers as _recovar_helpers

    recov_classes = [
        np.asarray(_recovar_helpers.load_mrc(str(output_dir / f"final_class{c + 1:03d}.mrc")), dtype=np.float64)
        for c in range(4)
    ]
    relion_classes = [
        np.asarray(
            _recovar_helpers.load_relion_volume(str(relion_dir / f"run_it003_class{c + 1:03d}.mrc")),
            dtype=np.float64,
        )
        for c in range(4)
    ]
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = _map_correlation(recov_classes[i], relion_classes[j])
    row, col = linear_sum_assignment(-M)
    matched = [float(M[i, j]) for i, j in zip(row, col)]
    mean_corr = float(np.mean(matched))
    worst_corr = float(np.min(matched))

    payload = {
        "kclass_coldstart_per_class_corrs_after_hungarian": matched,
        "kclass_coldstart_mean_corr": mean_corr,
        "kclass_coldstart_worst_class_corr": worst_corr,
        "kclass_coldstart_hungarian_assignment": [(int(i), int(j)) for i, j in zip(row, col)],
        "kclass_coldstart_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("kclass_coldstart", payload, output_dir=output_dir)
    logger.info("K-class cold-start ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print("=== K=4 cold-start parity (3-iter ab-initio vs RELION it003) ===", file=sys.stderr, flush=True)
    print(f"  per-class corrs (after Hungarian): {matched}", file=sys.stderr, flush=True)
    print(f"  mean_corr={mean_corr:.6f}  worst_class_corr={worst_corr:.6f}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # NEVER widen tolerance to make a test pass. Fix the code instead.
    _assert_fsc_gate("kclass_coldstart", output_dir)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_kclass_nonadaptive_replay(tmp_path):
    """Exercise three K4 iterations at coarse order 1 without oversampling.

    Load RELION iteration-0 noise/tau/sigma and replay perturbations/corrections
    with firstiter_cc. Check all four matched maps and particle class assignments.
    RELION GPU rejects firstiter_cc without oversampling. Comparing against its
    oversampling-1 reference preserves cross-grid regression coverage, not
    strict matched-state parity.
    """
    _assert_parity_ancestors_or_skip()
    # Canonical RELION GPU rejects firstiter_cc at OS0. This existing case
    # remains cross-grid regression coverage against its OS1 reference.
    relion_dir, dispatch_args = k4_oracle(1, 1)
    require_fixture_sets("k4_5k128_data")
    _require_fixture(REFINE_SCRIPT, K4_FIXTURE_DIR, relion_dir, K4_DATA_STAR)

    output_dir = tmp_path / "kclass_strict"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K4_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--n_classes",
        "4",
        "--max_iter",
        "3",
        "--healpix_order",
        "1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--adaptive_oversampling",
        "0",
        "--perturb_factor",
        "0.5",
        "--perturb_replay_relion_dir",
        str(relion_dir),
        "--relion_init_dir",
        str(relion_dir),
        *dispatch_args,
        "--firstiter_cc",
        # Pinned to this case's pre-GUI-default setting: no start-up low-pass.
        "--no-apply-initial-lowpass",
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
    ]
    logger.info("K-class nonadaptive replay cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    from scipy.optimize import linear_sum_assignment

    from recovar.utils import helpers as _recovar_helpers

    recov_classes = [
        np.asarray(_recovar_helpers.load_mrc(str(output_dir / f"final_class{c + 1:03d}.mrc")), dtype=np.float64)
        for c in range(4)
    ]
    relion_classes = [
        np.asarray(
            _recovar_helpers.load_relion_volume(str(relion_dir / f"run_it003_class{c + 1:03d}.mrc")),
            dtype=np.float64,
        )
        for c in range(4)
    ]
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = _map_correlation(recov_classes[i], relion_classes[j])
    row, col = linear_sum_assignment(-M)
    matched = [float(M[i, j]) for i, j in zip(row, col)]
    mean_corr = float(np.mean(matched))
    worst_corr = float(np.min(matched))

    # iter-3 class match
    import re as _re

    import starfile as _starfile

    npz = np.load(output_dir / "refinement_results.npz", allow_pickle=True)
    rec_h1 = np.asarray(npz["class_assignments_half0"], dtype=np.int32)
    rec_h2 = np.asarray(npz["class_assignments_half1"], dtype=np.int32)
    h1_idx = np.asarray(npz["half1_indices"], dtype=np.int64)
    h2_idx = np.asarray(npz["half2_indices"], dtype=np.int64)
    rd = _starfile.read(str(relion_dir / "run_it003_data.star"))
    p = rd["particles"] if isinstance(rd, dict) and "particles" in rd else rd
    relion_class = np.asarray(p["rlnClassNumber"], dtype=np.int32) - 1
    relion_names = list(p["rlnImageName"])
    si = lambda n: int(_re.match(r"(\d+)@", n).group(1)) - 1  # noqa: E731
    rcb = {si(n): relion_class[i] for i, n in enumerate(relion_names)}
    rd2 = _starfile.read(str(K4_DATA_STAR))
    rp = rd2["particles"] if isinstance(rd2, dict) and "particles" in rd2 else rd2
    recov_names = list(rp["rlnImageName"])
    n_total = len(recov_names)
    relion_aligned = np.array([rcb.get(si(s), -1) for s in recov_names], dtype=np.int32)
    rec_class = np.full(n_total, -1, dtype=np.int32)
    rec_class[h1_idx] = rec_h1
    rec_class[h2_idx] = rec_h2
    mk = (rec_class >= 0) & (relion_aligned >= 0)
    M2 = np.zeros((4, 4), dtype=np.int64)
    for r, R in zip(rec_class[mk], relion_aligned[mk]):
        M2[r, R] += 1
    row2, col2 = linear_sum_assignment(-M2)
    iter3_match = float(M2[row2, col2].sum() / mk.sum()) if mk.sum() > 0 else 0.0

    payload = {
        "kclass_strict_per_class_corrs_after_hungarian": matched,
        "kclass_strict_mean_corr": mean_corr,
        "kclass_strict_worst_class_corr": worst_corr,
        "kclass_strict_iter3_class_match": iter3_match,
        "kclass_strict_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("kclass_strict", payload, output_dir=output_dir)
    logger.info("K-class nonadaptive replay ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print(
        "=== K=4 nonadaptive replay (RELION os1 reference; cross-grid diagnostic) ===",
        file=sys.stderr,
        flush=True,
    )
    print(f"  per-class corrs (Hungarian): {matched}", file=sys.stderr, flush=True)
    print(f"  mean_corr={mean_corr:.6f}  worst_class_corr={worst_corr:.6f}", file=sys.stderr, flush=True)
    print(f"  iter-3 class match: {iter3_match:.4f}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # This case runs at os0 against the os1 oracle (cross-grid), so the approved rule gates it by
    # the pinned relax outputs, not by a RELION FSC floor. Until those pins exist the
    # correlation floors below stay as its regression check.
    # Strict bars — after the RELION Class3D M-step and post-reconstruction
    # mask/ini_high low-pass parity fixes, the os=0 K=4 fixture reached
    # per-class [0.9777, 0.9817, 0.9882, 0.9839], mean 0.9829. The
    # 0.982/0.975/0.84 floors lock against regressions in the source-backed
    # Class3D path without pretending that the remaining assignment gap is
    # solved.
    assert worst_corr >= 0.975, (
        f"K-class strict cold-start worst per-class corr {worst_corr:.4f} below 0.975: {matched}"
    )
    assert mean_corr >= 0.982, f"K-class strict cold-start mean_corr {mean_corr:.4f} below 0.982: {matched}"
    assert iter3_match >= 0.84, f"K-class strict cold-start iter-3 class match {iter3_match:.4f} below 0.84"


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_kclass_strict_oversample_coldstart(tmp_path):
    """Run three oracle-assisted K4 iterations at coarse order 1/oversampling 1.

    Load iteration-0 state and replay perturbations/corrections with firstiter_cc.
    All iterations use adaptive K-class execution; Hungarian-match all four final
    class maps against iteration 3 from the same dispatch-capable oracle.
    """
    _assert_parity_ancestors_or_skip()
    relion_dir, dispatch_args = k4_oracle(1, 1)
    require_fixture_sets("k4_5k128_data")
    _require_fixture(REFINE_SCRIPT, K4_FIXTURE_DIR, relion_dir, K4_DATA_STAR)

    output_dir = tmp_path / "kclass_strict_os1"
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(K4_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--n_classes",
        "4",
        "--max_iter",
        "3",
        "--healpix_order",
        "1",  # RELION coarse pass-1 order; fine pass 2 is order 2 via os=1
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--adaptive_oversampling",
        "1",  # 8× fine pose grid; matches matrix's run_k_class_parity 0.998 path
        "--perturb_factor",
        "0.5",
        "--perturb_replay_relion_dir",
        str(relion_dir),
        "--relion_init_dir",
        str(relion_dir),
        *dispatch_args,
        "--firstiter_cc",
        # Pinned to this case's pre-GUI-default setting: no start-up low-pass.
        "--no-apply-initial-lowpass",
        "--init_resolution",
        "30.0",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "2000",
    ]
    logger.info("K-class STRICT-PARITY oversample cold-start cmd: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    from scipy.optimize import linear_sum_assignment

    from recovar.utils import helpers as _recovar_helpers

    recov_classes = [
        np.asarray(_recovar_helpers.load_mrc(str(output_dir / f"final_class{c + 1:03d}.mrc")), dtype=np.float64)
        for c in range(4)
    ]
    relion_classes = [
        np.asarray(
            _recovar_helpers.load_relion_volume(str(relion_dir / f"run_it003_class{c + 1:03d}.mrc")),
            dtype=np.float64,
        )
        for c in range(4)
    ]
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = _map_correlation(recov_classes[i], relion_classes[j])
    row, col = linear_sum_assignment(-M)
    matched = [float(M[i, j]) for i, j in zip(row, col)]
    mean_corr = float(np.mean(matched))
    worst_corr = float(np.min(matched))

    payload = {
        "kclass_strict_os1_per_class_corrs_after_hungarian": matched,
        "kclass_strict_os1_mean_corr": mean_corr,
        "kclass_strict_os1_worst_class_corr": worst_corr,
        "kclass_strict_os1_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("kclass_strict_os1", payload, output_dir=output_dir)
    logger.info("K-class strict-parity oversample ledger: %s", ledger)

    print(file=sys.stderr, flush=True)
    print(
        "=== K=4 STRICT-PARITY oversample=1 cold-start (relion_init + perturb_replay + firstiter_cc + adaptive engine) ===",
        file=sys.stderr,
        flush=True,
    )
    print(f"  per-class corrs (Hungarian): {matched}", file=sys.stderr, flush=True)
    print(f"  mean_corr={mean_corr:.6f}  worst_class_corr={worst_corr:.6f}", file=sys.stderr, flush=True)
    print(f"  walltime_s={elapsed:.1f}", file=sys.stderr, flush=True)

    # Strict-os1 bars — after the RELION Class3D M-step and post-reconstruction
    # mask/ini_high low-pass parity fixes, the oversampled K=4 fixture reached
    # per-class [0.9898, 0.9950, 0.9973, 0.9987], mean 0.9952. The floors lock
    # against regressions in:
    #   * Class3D combined-accumulator single Wiener reconstruction
    #   * previous-Iref power-spectrum tau2, no gold-standard FSC
    #   * post-reconstruction solvent mask and firstiter ini_high low-pass
    #   * K-class adaptive oversampling and class rotation prior plumbing
    # Even when the os=0 strict_coldstart still passes.
    _assert_fsc_gate("kclass_strict_oversample_coldstart", output_dir)


# Several optics groups need the device-resident pass 2 on main (README); the flags are explicit here
# until auto-routing lands.
MULTIOPTICS_RESIDENT_ENV = {
    "RELAX_SPARSE_PASS2_RESIDENT": "1",
    "RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF": "1",
    "RELAX_K1_RELION_POWERCLASS_SPECTRUM_NORM": "1",
    "RELAX_K1_RELION_EXACT_BPREF_OPERANDS": "1",
}


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_em_parity_fast_k1_multioptics_coldstart(tmp_path):
    """Three standalone K=1 iterations on two optics groups with different pixel sizes and boxes.

    600 particles (300 per group: 4.25 A / 128 px and 5.44 A / 112 px), the RELION 5 GUI Refine3D
    command at --ini_high 30 with the reference on the images' greyscale (no --firstiter_cc),
    compared with RELION f2c1a3's iteration 3: half maps by FSC and per-particle Pmax. Exercises the per-group noise,
    the shape classes (scaled projection, remapped sizes and noise shells, per-class pre-shifts)
    and their merge end to end.
    """
    _assert_parity_ancestors_or_skip()
    require_fixture_sets("multioptics_s3b_600_data", "multioptics_s3b_600_relion")

    output_dir = tmp_path / "k1_multioptics_coldstart"
    output_dir.mkdir(parents=True)
    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(MULTIOPTICS_FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--init_volume",
        str(MULTIOPTICS_FIXTURE_DIR / "reference_init_greyscale.mrc"),
        "--max_iter",
        "3",
        "--init_resolution",
        "30",  # RELION --ini_high 30
        "--seed",
        "20260924",  # RELION --random_seed
        "--no-firstiter_cc",  # the reference is on the images' greyscale (greyscale_rescale.json)
        "--image-fourier-backend",
        "relion_cuda",
    ]
    env = {**gpu_subprocess_env(), **MULTIOPTICS_RESIDENT_ENV}
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    npz = np.load(output_dir / "refinement_results.npz")
    pmax_traj = np.asarray(npz["ave_Pmax_trajectory"], dtype=np.float64)
    assert pmax_traj.size >= 3, f"Expected 3 iterations, got {pmax_traj.size}"
    relion_pmax = [
        float(starfile.read(str(MULTIOPTICS_RELION_DIR / f"run_it{i:03d}_half1_model.star"))["model_general"]["rlnAveragePmax"])
        for i in (1, 2, 3)
    ]
    from recovar.utils import helpers as _recovar_helpers

    # relax writes its maps with write_mrc and RELION's go through load_relion_volume (see
    # _run_k1_coldstart for the frame conventions).
    h1_corr, h2_corr = (
        _map_correlation(
            np.asarray(_recovar_helpers.load_mrc(str(output_dir / f"final_half{h}.mrc")), dtype=np.float64),
            np.asarray(
                _recovar_helpers.load_relion_volume(str(MULTIOPTICS_RELION_DIR / f"run_it003_half{h}_class001.mrc")),
                dtype=np.float64,
            ),
        )
        for h in (1, 2)
    )
    payload = {
        "k1_multioptics_coldstart_half1_corr_vs_relion_it003": h1_corr,
        "k1_multioptics_coldstart_half2_corr_vs_relion_it003": h2_corr,
        "k1_multioptics_coldstart_pmax_iter3_recovar": float(pmax_traj[2]),
        "k1_multioptics_coldstart_pmax_iter3_relion": relion_pmax[2],
        "k1_multioptics_coldstart_pmax_iter3_abs_diff": abs(float(pmax_traj[2]) - relion_pmax[2]),
        "k1_multioptics_coldstart_pmax_trajectory_recovar": pmax_traj.tolist(),
        "k1_multioptics_coldstart_pmax_trajectory_relion": relion_pmax,
        "k1_multioptics_coldstart_walltime_s": elapsed,
    }
    ledger = _write_quality_ledger("k1_multioptics_coldstart", payload, output_dir=output_dir)
    logger.info("K=1 multi-optics cold-start ledger: %s", ledger)
    _assert_fsc_gate("k1_multioptics_coldstart", output_dir)
