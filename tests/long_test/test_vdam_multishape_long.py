"""Several-shape VDAM InitialModel for the full 200 iterations (long tier, robustness cell 4).

The multioptics K2 fixture (10k particles; 128 px at 4.25 A and 112 px at 4.86 A) runs the
production ``relax initial_model`` command of the benchmark row
multioptics_k2_10k128_initialmodel_vdam_k1 for 200 iterations, seed 1, on a cold cache.
The several-shape seed runs of af9e5870 crashed after iteration 110 (job 15071919), and that
commit had passed smoke and medium; this test runs the whole schedule.

Pass criteria:
  * The command exits 0 and writes iteration 200 for every particle of both optics groups.
  * The log has no memory-pool growth or out-of-memory notes.
  * The iteration-200 map's masked FSC-AUC against the class-1 ground truth (the row's metric:
    scripts/score_initialmodel_maps.py, registered with fit_rigid_both_hands, frozen mask
    multioptics_k2_c1) is at or above the floor of case ``vdam_multishape_200it`` in
    tests/tiers/fsc_thresholds.json, once that case is approved. The floor is a regression and
    crash guard derived from relax and GPU RELION (the row's reference) seed runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.em_fixtures import fixture_root, require_fixture_sets

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = fixture_root("multioptics_k2_10k128_data")
GT_DIR = fixture_root("multioptics_k2_10k128_gt")
MASK_KEY = "multioptics_k2_c1"
CASE = "vdam_multishape_200it"
N_ITER = 200
MEMORY_NOTES = ("allocator pool grew", "RESOURCE_EXHAUSTED", "out of memory", "Out of memory")


@pytest.mark.em_parity_long
@pytest.mark.gpu
@pytest.mark.integration
def test_vdam_multishape_200_iterations(tmp_path):
    require_fixture_sets("multioptics_k2_10k128_data", "multioptics_k2_10k128_gt")
    out = tmp_path / "vdam_multishape"
    out.mkdir()
    cmd = [
        sys.executable,
        "-c",
        "import logging, sys; logging.basicConfig(level=logging.INFO); "
        "from relax.commands.initial_model import main; sys.exit(main(sys.argv[1:]))",
        "--i",
        str(DATA_DIR / "particles.star"),
        "--datadir",
        str(DATA_DIR),
        "--o",
        str(out / "run"),
        "--nr_iter",
        str(N_ITER),
        "--grad_write_iter",
        "10",
        "--K",
        "1",
        "--sym",
        "C1",
        "--particle_diameter",
        "200",
        "--oversampling",
        "1",
        "--healpix_order",
        "1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--tau2_fudge",
        "4",
        "--random_seed",
        "1",
        "--gpu",
        "0",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    elapsed = time.time() - t0
    (out / "run.log").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, f"relax initial_model exited {proc.returncode}\nstderr:\n{proc.stderr[-6000:]}"
    log = proc.stdout + proc.stderr
    notes = [note for note in MEMORY_NOTES if note in log]
    assert not notes, f"memory notes in the log: {notes}"

    particles = starfile.read(DATA_DIR / "particles.star", always_dict=True)
    data = starfile.read(out / f"run_it{N_ITER:03d}_data.star", always_dict=True)["particles"]
    assert len(data) == len(particles["particles"])
    groups = set(particles["optics"]["rlnOpticsGroup"].astype(int))
    assert set(data["rlnOpticsGroup"].astype(int)) == groups and len(groups) == 2

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.masked_fsc import load_frozen_mask
    from scripts.score_initialmodel_maps import _unconverged_fits, score_cell

    mask_path, mask_record = load_frozen_mask(MASK_KEY)
    cell = {
        "id": CASE,
        "K": 1,
        "reference": {"kind": "gt", "frame": "relion", "paths": [str(GT_DIR / "reference_gt_class001_relion.mrc")]},
        "mask": {"key": MASK_KEY, "path": str(mask_path), "sha256": mask_record["mask"]["sha256"]},
        "arms": [
            {
                "label": "relax",
                "engine": "relax",
                "maps": [str(out / f"run_it{N_ITER:03d}_class001.mrc")],
                "model_star": str(out / f"run_it{N_ITER:03d}_model.star"),
            }
        ],
        "pairs": [],
    }
    scored = score_cell(cell)
    unconverged = _unconverged_fits({"arms": scored["arms"], "pairs": scored["pairs"]})
    assert not unconverged, f"GT registration stopped on its budget: {unconverged}"
    auc = float(scored["arms"]["relax"]["vs_reference"]["weighted_masked_fsc_auc"])
    payload = {"case": CASE, "walltime_s": elapsed, "weighted_masked_fsc_auc": auc}
    (out / f"{CASE}_ledger.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n{CASE}: masked FSC-AUC {auc:.5f}; wall {elapsed:.0f} s", file=sys.stderr, flush=True)

    thresholds = json.loads((REPO_ROOT / "tests" / "tiers" / "fsc_thresholds.json").read_text())
    gate = thresholds["cases"].get(CASE)
    if gate is not None and gate.get("approved"):
        assert auc >= gate["fsc_auc_floor"], payload
