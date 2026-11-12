# PPCA recipe v2 (code folder for BEST_PRACTICE v1.68), assembled 2026-11-11

> On the public branch `ppca/best-v1`: this checkout's own relax tree (ac8e62db) is the pinned relax, so the
> `relax_pin/` patches are not needed; point `--source` at the repository root. Della paths in this file are provenance.

Every file here is a byte copy of the file that produced the evidence rows; the sha256 is the check (the driver records its own
sha256 as `script_sha256` in each run's `arm.json`). Scores are the metrics suite v3.2/v3.3 (decoded-class FSC-AUC; D sets: the
distinctive protein) with the mean-only and true-model (rank-q noiseless) references beside them; real data: mean-map FSC to the
consensus map. Seeds as stated; s11 unless noted.

## Files

| file | sha256 | what |
|---|---|---|
| drivers/coupled_arm_v2.py | 8294284e537027675b14b0c95681f7c8bd8f79766930bf80618ab91eea877a43 | recipe v2 SGD driver: (a) centred, (b) real-data wide window |
| drivers/coupled_arm_shift.py | a3cc9708b6703db70183c8aab3ee70f1e5f9d2a489e9928ccb05a87c1b72d5cb | v2 + opt-in shift flags (--recentre, --offset-gate-pmax, --init-theta/--init-raw/--init-carry): (c) |
| drivers/centre_util.py | 86032e1493fcfda3bb9d3716ab3f042d7038ea0a0b1e302f266fd022260ddb7b | module imported by coupled_arm_shift.py (keep it in the same folder) |
| drivers/coupled_arm_fft2_ablate.py | 0037f6270e9789135e97b8e188b8ec33fedfa7c60feec1b39c6a1ae68088981d | merged driver (v2 --fixed-step + FFT translation search); shifted-set VDAM / FFT routes |
| drivers/coupled_arm_vdam_bestv1.py | 67233a3c10215d16850af409401d9bd484eb59768d3067221a810fa80b1a36af | best-v1 driver (= best_v1/coupled_arm.py); the VDAM-simple reference rows (d) |
| stage2/infer_all.py | 43fbf31584bf96ea48ea08f5bc499e808b94caf36a7474b6d3f0863eb630de63 | (c) stage 2: latents for all particles from a checkpoint (uses the metrics suite's inference; not self-contained) |
| stage2/zselect.py | 039befdc937003c5cc49334b1d73b839590c3da99931018966040ad5bb4d2aba | (c) stage 2: top fraction by robust-standardised abs(z) |
| stage2/make_subset.py | e7cea61135b86e4b506d8e4eab1ed46538de8a6e7d1d021e1f098d05e77a7ccf | (c) stage 2: subset training folder + manifest |
| vdam_step_3000.npy | 7a5261e2232b97ad1088e839f673ca3bb3c897556c9c595bfb670266b8816752 | lr file of every SGD arm (identical bytes to the runs' C__realistic__noise1/noise3 *_vdam_step.npy and best_v1/vdam_step_3000.npy) |
| example_arm_v2_*.json | see SHA256SUMS | arms for (a), (b), (c1-c3), paths parameterised |

relax: the public relax at `relax_pin/RELAX_COMMIT` (bc8fd58d) plus `relax_pin/patches/` = branch ws2-gated-sgd at ac8e62db,
the tree every row below ran on. Build with `relax_pin/setup.sh` and `relax_pin/build_natives.sh`; run with `relax_pin/env.sh`
(unchanged from best v1).

## How to run one arm

```
export WORK=<writable dir>; source relax_pin/env.sh          # sets PY, RELAX_SRC, natives, caches, TMPDIR
$PY best_v2/drivers/<driver> --source $RELAX_SRC --manifest <manifest.json> --eval-every 500 <arm "data"> \
    --output <out>/<arm name> --seed <seed> --stop-after <stop_after> <arm "driver" list> -- <arm "relax" list>
```
(the order used by the study's queue runner). Arm fields map one to one onto that line; `_driver`/`_note` are comments.
Memory: one wide-window or recentred q-10 arm per A100 80 GB (relax sizes HEALPix-3 tiles to free GPU memory; 68.7 GB peak seen).
`--iterations` (default 6000) caps runs silently; pass `--iterations <stop>` for longer runs.

## (a) STANDARD: v2os1, centred data (driver coupled_arm_v2.py)

```
--q 10 --milestones 25,...,500,600,700,800,900,1000,1250,...,3000 --lr-schedule file:best_v2/vdam_step_3000.npy
--uniform-direction-prior --sgd-curv split --sgd-dir coupled --adaptive-eta-pc-only --adaptive-eta-pc-floor 0.25
--fixed-step 0.10 --os-from-update 1626 --os-rotations-only
-- --optimizer momentum_sgd --sgd-learning-rate 0.9550 --stages <stock 21-stage table, shell 11->31, HEALPix 1->2->3 at 1626> --oversampling 1
data: --shift-step 1 --shift-range 2   (zero-centred +-2 px)
```
Full JSON: example_arm_v2_a_centred.json. Evidence (best_v1_20261105/os1_v2, s11; mean-only / true-model):
C n1 .937 (.837 / 1.000), C n3 .916 (.837 / 1.000), Ribosembly n3 .860 (.780 / .980); os1 adopted over os0 (C n1 v2 .915).
T10 n1 .594 (pose 61 deg): unsolved by any recipe.

## (b) CANDIDATE: real data as picked, wide held-prior window "W" (driver coupled_arm_v2.py)

(a) plus, in the driver list: `--offset-sigma-hold 20 --shift-sched [[1, 8, 2], [1626, 8, 1]]` (window 48 A, 12 A steps to 1625,
then 6 A; written for 6 A/px, convert in A for other pixel sizes: 10073 at 8.313 A/px ran [[1, 6, 1], [1626, 6, 1]]).
Full JSON: example_arm_v2_b_real_wide.json (= realshift arms/e10345_WR.json W).
Evidence (realshift_20261109, mean-map FSC to consensus; A = (a) on the as-picked stack, B = (a) on the pre-centred twin):
EMPIAR-10073 W .963 (A .913, B .973); EMPIAR-10345 W .916 (A .700, B .930).

## (c) CANDIDATE: shifted minority, three stages (D10_shift / D20_shift; driver coupled_arm_shift.py)

1. Stage 1, all particles (example_arm_v2_c1_stage1_gate5.json): v2 os0 with `--recentre --shift-sched [[1, 4, 1], [1626, 4, 1]]
   --fixed-step 0.10 --offset-gate-pmax 0.5`, q 2, lr .7292. Alone it does not pose the minority.
2. Stage 2: `infer_all.py <stage-1>/checkpoint_3000.npz <manifest> z.npz`; `zselect.py z.npz 0.10 idx.npy`;
   `make_subset.py <training dir> idx.npy <subset dir>`; then v2 os0 on the subset (example_arm_v2_c2_stage2_subset.json):
   zero-centred +-2 px, `--fixed-step 0.15`, lr .7.
3. Stage 3, all particles from the stage-2 model (example_arm_v2_c3_stage3_late.json): stage-1 flags without the gate
   (`--recentre --shift-sched [[1, 4, 1], [1626, 4, 1]] --fixed-step 0.10`) plus `--init-theta <stage-2>/checkpoint_3000.npz
   --init-raw --init-carry noise`, relax `--stages [[1, 31, 3]]` (late-stage start: final band, HEALPix 3 from update 1).
Evidence (shiftprog_20261109 + closer suite rows; distinctive protein; mean-only .669 / true-model .998):
D10_shift stage 3 s11 / s12 / s13 = .907 / .910 / .910, pose 4.1-4.2 deg, 97% < 10, minority pose kept at every checkpoint
(stage 2 alone .68-.74; (a) on D10_shift .226). D20_shift s11 stage 3: direct fit 3.8 deg / 100% (suite row pending).
D5_shift: fails at every stage and with VDAM (signal-limited at 5%). Cost ~1.2 GPU-h per seed.
Provenance note: the s11 stage-1 run recorded an earlier driver build (script_sha256 5b64ebe6..., not kept); s12, s13 stage 1,
all stage-2 and stage-3 runs and D20/D5 ran a3cc9708.

## (d) OPTION: VDAM-simple for C_shift / Ribosembly_shift n3 (driver coupled_arm_vdam_bestv1.py, or coupled_arm_fft2_ablate.py)

```
--q 10 --milestones 25,...,500,750,...,3000 --vdam-ko noadapt --offset-sigma-hold 20 --shift-sched [[1, 8, 2], [1626, 8, 1]]
-- --optimizer vdam --stages <stock 21-stage table> --oversampling 0
data: --shift-step 1 --shift-range 2; stop 3000 of the driver's default 6000-update VDAM schedule; learned direction prior
```
Evidence (Roey Yadgar's benchmark, driver 67233a3c, H100, s11; mean-only / true-model): C_shift n1 .873 (SGD v2 .840),
C_shift n3 .834 (SGD v2 .818); Ribosembly_shift n3 .755, ~59% < 10 deg, below mean-only .780 (stress set; every SGD arm .16).
The same arm on coupled_arm_fft2_ablate.py (A100) gives Ribosembly_shift n3 .755 / 59% (identical config dict).
Fails D10_shift and D5_shift from scratch (direct fit 106 / 100 deg). For 9 A/px sets convert the window in A ([[1, 5, 1], [1626, 5, 1]]).

Real data (2026-11-12, BEST_PRACTICE v1.74): VDAM-simple with the wide window (coupled_arm_fft2_ablate.py 0037f627) is the best
as-picked row on EMPIAR-10097 (mean-map FSC .829 vs W .709, pre-centred .853); on 10073 / 10345 / 10076 it is .942 / .894 / .914
(W .963 / .916 / .947). Run it next to W where W leaves a large gap to pre-centred.

## (e) NOT ADOPTED

- Roey's VDAM shell gate (powers + full-data scale): neutral on shifted sets against his own plain arm; diverged in our VDAM-simple port.
- Fixed mean (selection map, 30 A partial hold): fails on 11846 and synthetic Ribosembly_shift; on 10345 adds FSC but not pose; needs a reference.
- VDAM as stage 3: destroys the D10_shift subset model (116 deg by update 500); on Ribosembly_shift n3 worse than SGD stage 3 (55% vs 62% < 10).
- SGD stage 3 on real data (seeded from the VDAM-simple run with drivers/seed_offsets.py f0d810ae, --init-offsets; coupled_arm_shift.py a3cc9708): ties W on 10073 / 10345 / 10076 (-.006 to -.008) and lowers VDAM on 10097 (.829 -> .792); fails the transfer rule (1 PASS / 3 TIE).
