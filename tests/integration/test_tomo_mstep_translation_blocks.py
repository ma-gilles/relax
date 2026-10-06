"""A subtomogram M-step split into translation blocks keeps its BPref weight (relax#27, GPU).

The tilt M-step backprojects each row once. When its particles' translations do not fit one tile block, each
row's partials are merged across the blocks before that one adjoint (``resident_tilts._tilt_mstep_program``).
Without the merge, every block's partial reached a voxel separately, and a partial below half an ulp of the
voxel's float32 total was lost. That loss only shows at a realistic size, where the DC and low-shell voxels add
up tens of thousands of rows: on this fixture, blocks of 2 lost Ft_ctf relL2 1.0-1.2e-5 against one block
(DC 1.2-1.4e-4, summed weight -7e-6), where same-code repeats differ by 6e-8.

One VDAM iteration of the et09_box64 benchmark (2000 particles x 41 tilts, box 64) is continued from RELION's
iteration-190 state twice: at the default tile budget (one block) and at a budget of 5e7 bytes
(``RELAX_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES``: blocks of a few translations). Each pass logs its blocks
(``resident_tilts.TiltMstepCensus``), so the test checks the forced run reached several blocks and the default
one did not. The two runs' BPref pairs (dumped through ``RELAX_INITIAL_MODEL_ACCUM_DUMP_DIR``) and maps must
then agree inside bands set from measured same-code noise.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from conftest import gpu_subprocess_env
from helpers.em_fixtures import fixture_root, require_fixture_sets

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu]

DATA, STATE = "cryoet_et09_box64_data", "cryoet_et09_box64_relion_vdam_it190"
FORCED_TILE_BYTES = "50000000"  # blocks of a few translations at every current size of the fixture
# Bands from four local A100 continuations of 6c091764 (two at the default budget, two forced, 2026-10-06;
# em_work/relax_bindw_20260930/tomo_et09/guard_6c09176). Same-code repeats: Ft_y relL2 4.6e-7, Ft_ctf relL2
# 1.2e-7, summed weight 1.3e-9 and 2.7e-10, map 1 - FSC 1.6e-13. Forced blocks of 1 against one block (four
# pairs): Ft_y 5.0-5.2e-7, Ft_ctf 1.2-1.3e-7, summed weight 1.8-2.3e-9, 1 - FSC 1.5e-13, i.e. inside the repeat
# spread. Without the merge, blocks of 2 lost Ft_ctf 1.0-1.2e-5, summed weight 7-8e-6 and Ft_y 2.7e-6 (relax#27),
# more with more blocks. Each bound is 3-10 times the largest measured value and below the loss: with the merge
# disabled (probe, same fixture, forced budget) this test measured Ft_y 2.8e-6, Ft_ctf 1.2e-5 and summed weight
# 7.9e-6, failing all three. The map check is a sanity bound only: one iteration moves the map too little for
# the loss to show there (1 - FSC 1.9e-13 in that probe). The test takes about 250 s on an A100.
BANDS = {"Ft_y relL2": 1.5e-6, "Ft_ctf relL2": 5e-7, "Ft_ctf summed weight": 2e-8}
MAX_MAP_ONE_MINUS_FSC = 2e-12


def _continue(out: Path, extra_env: dict[str, str]) -> str:
    project, state = fixture_root(DATA), fixture_root(STATE)
    optimiser = state / "run_it190_optimiser.star"
    (out / "out").mkdir(parents=True)
    (out / "optimisation_set.star").write_text(
        "\n# version 50001\n\ndata_\n\nloop_\n_rlnTomoParticlesFile #1\n_rlnTomoTomogramsFile #2\n"
        f"{state / 'run_it190_data.star'} {project / 'tomograms.star'}\n"
    )
    command = [
        sys.executable, "-c",
        "import logging, sys; logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s'); "
        "[logging.getLogger(n).setLevel(logging.WARNING) for n in ('jax', 'absl')]; "
        "from relax.commands.initial_model import main; sys.exit(main(sys.argv[1:]))",
        "--ios", str(out / "optimisation_set.star"), "--datadir", str(project), "--o", str(out / "out" / "run"),
        "--nr_iter", "200", "--grad_write_iter", "1", "--K", "1", "--sym", "C1", "--particle_diameter", "240",
        "--oversampling", "1", "--healpix_order", "1", "--offset_range", "6", "--offset_step", "2",
        "--tau2_fudge", "4", "--random_seed", "2", "--gpu", "0", "--j", "8",
        "--diagnostic-continue-optimiser", str(optimiser), "--diagnostic-stop-after-iteration", "191",
        "--diagnostic-continue-input-order",
    ]  # fmt: skip
    env = dict(gpu_subprocess_env(), RELAX_INITIAL_MODEL_ACCUM_DUMP_DIR=str(out / "accum"), **extra_env)
    log = out / "run.log"
    with open(log, "w") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    text = log.read_text()
    assert result.returncode == 0, text[-4000:]
    return text


def _multi_block_chunks(log: str) -> list[int]:
    """Each pass's multi-block chunk count from its census line."""

    counts = [int(m) for m in re.findall(r"Tilt M-step translation blocks: .* multi-block chunks (\d+)/\d+", log)]
    assert counts, "no tilt M-step census line"
    return counts


def _rel_l2(a, b) -> float:
    a, b = np.asarray(a, dtype=np.complex128), np.asarray(b, dtype=np.complex128)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b)))


def _map_fsc(a: Path, b: Path) -> np.ndarray:
    fa = np.fft.fftn(np.asarray(mrcfile.open(a).data, dtype=np.float64))
    fb = np.fft.fftn(np.asarray(mrcfile.open(b).data, dtype=np.float64))
    n = fa.shape[0]
    k = np.fft.fftfreq(n) * n
    shells = np.rint(np.sqrt(sum(g**2 for g in np.meshgrid(k, k, k, indexing="ij")))).astype(int)
    return np.array(
        [
            np.real(np.vdot(fa[shells == s], fb[shells == s]))
            / np.sqrt(np.vdot(fa[shells == s], fa[shells == s]).real * np.vdot(fb[shells == s], fb[shells == s]).real)
            for s in range(1, n // 2)
        ]
    )


def test_translation_blocked_tilt_mstep_keeps_the_one_block_weight(tmp_path):
    require_fixture_sets(DATA, STATE)
    one = _continue(tmp_path / "one_block", {})
    forced = _continue(tmp_path / "blocks", {"RELAX_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES": FORCED_TILE_BYTES})
    assert max(_multi_block_chunks(one)) == 0
    assert min(_multi_block_chunks(forced)) > 0

    measured = {}
    for half in (0, 1):
        a = np.load(tmp_path / "one_block" / "accum" / f"accum_h{half}_k0.npz")
        b = np.load(tmp_path / "blocks" / "accum" / f"accum_h{half}_k0.npz")
        weight_a, weight_b = np.asarray(a["bp_weight_unscaled"]), np.asarray(b["bp_weight_unscaled"])
        values = {
            "Ft_y relL2": _rel_l2(a["bp_data_unscaled"], b["bp_data_unscaled"]),
            "Ft_ctf relL2": _rel_l2(weight_a, weight_b),
            "Ft_ctf summed weight": abs(
                float(np.sum(weight_b, dtype=np.float64)) / float(np.sum(weight_a, dtype=np.float64)) - 1.0
            ),
        }
        print(f"half {half + 1}: " + ", ".join(f"{name} {value:.2e}" for name, value in values.items()))
        for name, value in values.items():
            measured[name] = max(measured.get(name, 0.0), value)
    fsc = _map_fsc(
        tmp_path / "one_block" / "out" / "run_it191_class001.mrc",
        tmp_path / "blocks" / "out" / "run_it191_class001.mrc",
    )
    print(f"map 1 - FSC one block vs blocks: largest {1.0 - fsc.min():.2e} (shell {int(np.argmin(fsc)) + 1})")
    for name, value in measured.items():
        assert value < BANDS[name], f"{name} {value:.2e} >= {BANDS[name]:.1e}"
    assert 1.0 - fsc.min() < MAX_MAP_ONE_MINUS_FSC, f"map 1 - FSC {1.0 - fsc.min():.2e}"
