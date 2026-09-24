"""Medium-tier end-to-end K1 auto-refine on the 5k/128 fixture, scored with FSC against a RELION band.

relax runs standalone (only relion_refine's inputs: particles.star, the stack, the reference
and the command values of the RELION oversampling-1 reference) to convergence and through the
final all-data iteration. The RELION band is that reference plus its same-command repeats
(fixture sets ``k1_5k128_relion_os1`` and ``k1_5k128_relion_os1_repeats``).

Measured and written to the case ledger: GT FSC-AUC of the final merged map for relax and every
RELION run, cross-engine FSC-AUC of relax against every RELION run and of the RELION runs
against each other, the minimum in-band shell FSC, and the iteration counts. The FSC band gate
applies once ``tests/tiers/fsc_thresholds.json`` records the user's approval; until then the
test asserts only that the run converged and finished, and reports the FSC values.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import gpu_subprocess_env
from helpers.em_fixtures import fixture_root, require_fixture_sets

REPO_ROOT = Path(__file__).resolve().parents[2]
REFINE_SCRIPT = REPO_ROOT / "scripts" / "run_full_refinement.py"
THRESHOLDS = REPO_ROOT / "tests" / "tiers" / "fsc_thresholds.json"
sys.path.insert(0, str(REPO_ROOT))
from scripts.fsc_metrics import normalized_fsc_auc, shell_fsc  # noqa: E402

RELION_COMMAND_TOKENS = (
    "--auto_refine",
    "--split_random_halves",
    "--particle_diameter 544",
    "--ini_high 30",
    "--ctf",
    "--flatten_solvent",
    "--zero_mask",
    "--low_resol_join_halves 40",
    "--norm",
    "--scale",
    "--healpix_order 3",
    "--offset_range 3.0",
    "--offset_step 1.0",
    "--oversampling 1",
    "--pad 2",
    "--random_seed 1775735620",
)


def _relion_runs() -> dict[str, Path]:
    """The RELION reference and its same-command repeats (each a directory with run_class001.mrc)."""
    runs = {"reference": fixture_root("k1_5k128_relion_os1")}
    repeats = fixture_root("k1_5k128_relion_os1_repeats")
    for d in sorted(p for p in repeats.iterdir() if (p / "run_class001.mrc").exists()):
        runs[d.name] = d
    return runs


def _fsc(a: np.ndarray, b: np.ndarray, band: int) -> dict:
    curve = np.asarray(shell_fsc(a, b), dtype=np.float64)
    return {"fsc_auc": float(normalized_fsc_auc(curve)), "min_shell_fsc_in_band": float(np.nanmin(curve[1:band]))}


def _approved_gate(case: str) -> dict | None:
    thresholds = json.loads(THRESHOLDS.read_text())
    return thresholds["cases"].get(case) if thresholds.get("approved") else None


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_k1_5k128_standalone_autorefine(tmp_path):
    from recovar.utils import helpers

    require_fixture_sets("k1_5k128_data", "k1_5k128_relion_os1", "k1_5k128_relion_os1_repeats")
    data, relion_ref = fixture_root("k1_5k128_data"), fixture_root("k1_5k128_relion_os1")
    header = (relion_ref / "run_it000_optimiser.star").read_text().splitlines()[1]
    missing = [t for t in RELION_COMMAND_TOKENS if f" {t} " not in f" {header} "]
    assert not missing, f"RELION reference command lacks {missing}: {header}"

    output_dir = tmp_path / "k1_e2e_5k_standalone"
    cmd = [
        sys.executable,
        str(REFINE_SCRIPT),
        "--data_dir",
        str(data),
        "--output",
        str(output_dir),
        "--max_iter",
        "25",
        "--healpix_order",
        "3",
        "--offset_range",
        "3.0",
        "--offset_step",
        "1.0",
        "--adaptive_oversampling",
        "1",
        "--tau2_fudge",
        "1.0",
        "--perturb_factor",
        "0.5",
        "--seed",
        "1775735620",
        "--perturb_seed",
        "1775735620",
        "--relion-half-sets-from-input",
        "--no-firstiter_cc",  # the os1 oracle ran without --firstiter_cc
        "--particle_diameter_ang",
        "544",
        "--apply-initial-lowpass",
        "--init_resolution",
        "30.0",
        "--image-fourier-backend",
        "relion_cuda",
        "--image_batch_size",
        "200",
        "--rotation_block_size",
        "8192",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    assert proc.returncode == 0, (
        f"run_full_refinement.py exited {proc.returncode}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    )

    npz = np.load(output_dir / "refinement_results.npz")
    converged = bool(npz["convergence_has_converged"])
    final_all_data = bool(npz["final_all_data_ran"])
    n_iter = int(np.asarray(npz["ave_Pmax_trajectory"]).size)
    relion_iters = len(list(relion_ref.glob("run_it0[0-9][0-9]_optimiser.star"))) - 1

    gt = np.asarray(helpers.load_mrc(str(data / "reference_gt.mrc")), dtype=np.float64)
    band = gt.shape[0] // 2
    relax_map = np.asarray(helpers.load_mrc(str(output_dir / "final_merged.mrc")), dtype=np.float64)
    relion = {
        k: np.asarray(helpers.load_relion_volume(str(p / "run_class001.mrc")), dtype=np.float64)
        for k, p in _relion_runs().items()
    }
    payload = {
        "walltime_s": elapsed,
        "converged": converged,
        "final_all_data_ran": final_all_data,
        "relax_iterations": n_iter,
        "relion_reference_iterations": relion_iters,
        "relax_vs_gt": _fsc(relax_map, gt, band),
        "relion_vs_gt": {k: _fsc(v, gt, band) for k, v in relion.items()},
        "relax_vs_relion": {k: _fsc(relax_map, v, band) for k, v in relion.items()},
        "relion_vs_relion": {
            f"{a}|{b}": _fsc(relion[a], relion[b], band) for a, b in itertools.combinations(relion, 2)
        },
    }
    (output_dir / "em_tier_e2e_ledger_k1_5k128.json").write_text(json.dumps(payload, indent=1) + "\n")
    band_gt = [v["fsc_auc"] for v in payload["relion_vs_gt"].values()]
    print(
        f"\nK1 5k e2e: relax GT FSC-AUC {payload['relax_vs_gt']['fsc_auc']:.6f}; RELION band "
        f"[{min(band_gt):.6f}, {max(band_gt):.6f}]; relax vs RELION reference FSC-AUC "
        f"{payload['relax_vs_relion']['reference']['fsc_auc']:.6f}; iterations relax {n_iter} / RELION {relion_iters}",
        file=sys.stderr,
        flush=True,
    )

    assert converged and final_all_data, (
        f"relax did not converge and finish (converged={converged}, final={final_all_data})"
    )
    gate = _approved_gate("e2e_k1_5k128_standalone")
    if gate is not None:
        assert payload["relax_vs_gt"]["fsc_auc"] >= min(band_gt) - gate["gt_fsc_auc_below_band"], payload
        worst = min(v["fsc_auc"] for v in payload["relax_vs_relion"].values())
        assert worst >= gate["min_cross_fsc_auc"], payload
