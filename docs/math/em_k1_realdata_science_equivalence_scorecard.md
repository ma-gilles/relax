# K=1 real-data science-equivalence scorecard

This fixed scorecard is separate from strict full-spectrum RELION numerical
parity. Comparable unmasked, unaligned within-engine half-map FSC is always
mandatory. Cross-engine equivalence can pass directly in the canonical frame
or through one pinned continuous proper-SO(3) rotation and translation fitted
from low-frequency merged maps and applied unchanged to both split halves.
Reflection, density-sign, and scale fitting are forbidden. A common-mask FSC
is reported as supporting evidence only and can never rescue a failure.

## Frozen primary gates

| Gate | Threshold |
| --- | ---: |
| Half-map resolution ratio | <= 1.05 |
| Half-FSC curve RMSE in the jointly resolved band | <= 0.02 |
| Absolute half-FSC band-AUC difference | <= 0.02 |
| Merged cross-engine band FSC-AUC, raw canonical **or** proper-rigid route | >= 0.95 |
| Each cross-engine half-map band FSC-AUC on the same route | >= 0.90 |

The joint band is shells 1 through one shell before the earlier first
three-shell-sustained crossing below `1/7`. Resolution uses
`box_size * voxel_size_angstrom / crossing_shell`.

## Frozen calibration replay

The completed 10073, 10345, and 10097 runs calibrate the metric only. They
ran descendant commits and do not count in the PR #158 scoring denominator.
A calibration status is determined only by the mandatory unmasked, unaligned
within-engine half-map gates; cross-engine qualification is reported separately.

| Case | Half-map calibration | Half RMSE | Half AUC delta | Raw merged/min-half AUC | Proper merged/min-half AUC | Qualified route |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `empiar-10073-native-c1` | pass | 0.002646 | 0.001260 | 0.967771/0.940802 | -- | `raw_canonical` |
| `empiar-10345-native-c1` | pass | 0.003242 | 0.000165 | 0.975045/0.965089 | -- | `raw_canonical` |
| `empiar-10097-native-c1` | pass | 0.004926 | 0.001501 | 0.932992/0.885803 | 0.932290/0.884820 | `none_unqualified` |

The RECOVAR merged maps of `empiar-10073-native-c1`, `empiar-10345-native-c1`, `empiar-10097-native-c1`
came from a final all-data pass that skipped RELION's gridding
correction (the unfiltered half maps were corrected). Their merged cross-engine
values were regenerated post hoc with RELION griddingCorrect on the saved final
maps (division by the radial sinc^2); each case's `post_hoc_regeneration` record
pins the corrected map, the tool and the superseded artifacts and values.

These are deliberately parity-calibration refinements, not reproductions of
the deposited publication workflows. Both engines crossed the
three-consecutive-shell unmasked half-map FSC 1/7 threshold at shell 81 on
10073 and shell 49 on 10345. Under the frozen resolution formula, these are
6.568 A and 8.235 A, respectively. On 10097, RECOVAR and RELION cross at
shells 44 and 45 (7.622 A and 7.452 A). These values are worse than the
3.7 A deposited
resolution for [EMD-8012](https://www.ebi.ac.uk/emdb/EMD-8012) and the 3.51 A
focused resolution for [EMD-20795](https://www.ebi.ac.uk/emdb/EMD-20795),
which also deposits a 3.8 A sharpened full-complex map.

The absolute gap is expected from the frozen calibration protocol. For 10073,
normalization intentionally drops the supplied refined Euler angles and
origins. The 10345 source STAR has no Euler columns and only zero or invalid
in-plane origins. Both engines therefore start from the same newly generated
de-novo K=1 model, and the reported maps and FSCs are unmasked, unsharpened,
and unpostprocessed. The deposited 10073 workflow instead used EMD-2966
low-pass filtered to 60 A. The exact published 10345 complex did not use 3D
classification, but its final maps used non-uniform and local-resolution
refinement, local-resolution estimation, sharpening, and local filtering;
the deposited primary map is a focused refinement. C1 is the appropriate
symmetry for 10073 and 10345 and is not the cause of that gap.

The 10073 and 10345 cases establish that RECOVAR and RELION reach essentially
the same reconstruction under the matched protocol. The 10097 within-engine
half-map comparison also passes strongly, but its raw cross-engine AUCs
(0.932992 merged; 0.885803/0.888938 halves) miss the frozen cross-engine
gates. Corrected job 13276576 tested the allowed proper-SO(3)+translation
route after explicitly adding canonical identity to the HEALPix seed set.
It fitted only a 0.315-degree rotation and 0.044-voxel translation, but its
0.932290/0.884820/0.887813 aligned AUCs still miss the same gates. The route
is therefore recorded as unqualified and does not rescue 10097; a small
global rigid drift does not explain the residual cross-engine difference.
Job 13275901 is retained only as a superseded audit artifact because its
seed search omitted identity and selected a false distant orientation.

These calibration results do not establish
that this intentionally stripped-down protocol reproduces the published
reconstruction. Absolute high-resolution achievement is tested separately by
the frozen 10202 case below.

## Supporting RELION corrected-masked FSC

These measurements are supporting-only: they do not enter any acceptance
gate, cannot rescue an unmasked failure, and do not change the scoring
denominator. Each mask was generated only from the RELION merged map and
then passed byte-for-byte to both engines' independent half-map postprocess.

| Dataset | RECOVAR corrected masked (A; shell) | RELION corrected masked (A; shell) | Curve RMSE | AUC delta | Mask SHA-256 prefix |
| --- | ---: | ---: | ---: | ---: | --- |
| 10073 | 4.156; 128 | 4.092; 130 | 0.004180 | 0.001600 | `49ba576a4c44` |
| 10345 | 5.240; 77 | 5.240; 77 | 0.004790 | 0.000313 | `14f572e49552` |
| 10097 | 5.782; 58 | 5.684; 59 | 0.008370 | 0.002113 | `b47439f30167` |

RELION first low-pass filtered each merged map to 15 A, then used
`relion_mask_create --extend_inimask 5 --width_soft_edge 8`. Both
postprocess calls used `--force_mask --skip_fsc_weighting --low_pass 0
--randomize_at_fsc 0.8 --random_seed 42`. Exact per-dataset argv, input
hashes, mask thresholds, and output hashes are sealed in the aggregate
artifact with SHA-256 `0c6930e801b2d4364fdde8d69d3982250b397f20c6bf17caecc9b6862f0f28ad`.

Producer jobs were `13273806` for 10073/10345 and `13274377` for
10097; requested and allocated resources matched. The first job's nonzero
state occurred only after its two retained datasets, at the later 10097
component audit. The isolated second job reused the literal audited 10097
mask. Regenerating that mask changed only MRC header-statistic bytes 249,
250, and 253; the voxel payload was identical.

## Available EMPIAR-10202 per-engine evidence

This is a deliberately partial report. RELION is sealed and complete;
RECOVAR and every cross-engine acceptance metric remain pending.
The RELION-only result cannot pass the fixed scoring case.

| Engine | Status | Unmasked FSC=0.143 (A) | Corrected masked FSC=0.143 (A) | <= 3.0 A arm | Jobs |
| --- | --- | ---: | ---: | --- | --- |
| RELION | complete | 2.511554 | 2.122559 | pass | `13217551` / `13254149` |
| RECOVAR | pending | -- | -- | pending | -- |

The 2.511554-A value is RELION's final unmasked FSC estimate sealed by
the postprocess result manifest. The 2.122559-A corrected masked value is
supporting-only and cannot rescue an unmasked or cross-engine failure.
The fixed scorecard's three-consecutive-shell joint-band metric is still
unavailable until RECOVAR supplies its independent half maps.

RELION refinement stdout SHA-256: `582e90f851b4f389e953113b0cf2b4cb3f46f833567015c34a35e5551c475fea`.
Matched harness manifest SHA-256: `5764c09af9db73ac7523db7cf34e1422449e158fb40e84b0e3aef2ee49a4e279`.
Postprocess result-manifest SHA-256: `de865eba64a3d8a9c7693af5e356cbece72572630bbb184462841f8517f2899d`.
Common mask SHA-256: `dcc3fd17e7f728b3164416f695b257fef5fceeca7c2a4ba04ea6d82c9a933b17`.

### Non-scoring matched iteration-11 checkpoint

The interrupted RECOVAR trajectory completed iteration 11 before a CUDA
out-of-memory failure in iteration 12. This checkpoint cannot score the
case, but RELION iteration 11 was postprocessed with the identical mask,
executable, and FSC convention. The same-iteration resolution crossings
are identical:

| FSC | RECOVAR (A; shell) | RELION (A; shell) | Resolved-band RMSE | Resolved-band AUC delta |
| --- | ---: | ---: | ---: | ---: |
| Raw unmasked | 2.521600; 250 | 2.521600; 250 | 0.004645 | 0.000260 |
| Corrected masked (supporting only) | 2.541935; 248 | 2.541935; 248 | 0.011651 | 0.001028 |

The raw comparison uses shells 1--249, ending immediately before the
shared three-shell-sustained crossing. The corrected-masked comparison
uses its own shells 1--247 band. This is strong evidence that RECOVAR had
already reached RELION's same-iteration half-map quality, but terminal
equivalence and the <=3.0-A gate remain pending until an uninterrupted
trajectory produces sealed final half maps.

RECOVAR checkpoint/postprocess jobs: `13339556` / `13355910`. Matched RELION iteration-11
postprocess job: `13356820`. Replacement
full-run job `13356985` at subject commit `6e414838463e` failed in numbered iteration 12 at current size 564 with a CUDA OOM after completing numbered iteration 11.
The failure is recorded as a compact-planner routing/headroom defect,
not as a scientific-resolution failure. The fail-closed compact-planner
successor was captured running in job `13363818`
at commit `8069ac01508d`, after completing numbered iteration 8.

Matched iteration-11 summary SHA-256: `cb60a01c3e293817c39b745d3e42d8e2c47f440d231001983ccc2e7577250704`.
Resolved curve comparison SHA-256: `1bb1bf853d9e7403a49fa3771017773ff26ef178b22258f4d26ffb420f6ccf2e`.

## Fixed scoring case

Equivalence and absolute high-resolution achievement are reported
separately. The target passes overall only when both pass.

| Case | Overall | Equivalence | Route | High resolution | RECOVAR half FSC (A) | RELION half FSC (A) | Raw merged AUC | Proper merged AUC | Half RMSE |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `empiar-10202-set06-k1-I1` | pending | -- | `--` | pending | -- | -- | -- | -- | -- |

The independent high-resolution gate requires **each** engine's unmasked
half-map FSC resolution to be <= 3.0 A. For context,
EMD-9012 records 1.86 A deposited validation and
1.94 A EMDB-calculated unmasked half-map resolution.
These reference values provide context; they are not substituted for either
engine's measured result.

The deposited 30,515-particle half split is already frozen: 15,258 in
half 1 and 15,257 in half 2, hashed as one `uint8` `rlnRandomSubset`
value per source-STAR row. STAR normalization must preserve particle
order, deposited halves, Euler angles, and origins.
The complete 78,118,401,024-byte stack is pinned by SHA-256 prefix
`8eecf0fb` (not merely by its MRC header).

The older `particles.native.star` artifact with SHA-256 prefix
`c13cb927` is explicitly rejected because it rerandomized halves and
dropped deposited pose/shift metadata.

This deposited/FREALIGN frame requires explicit `I1`; bare RELION `I`
canonicalizes to `I2` and is forbidden here. The frozen operator sequence
is identity followed by RELION `SymList` order; its rounded little-endian
`(left, right)` float64 digest begins `093a0876`.

The preparation contract is fully frozen: normalized STAR SHA-256 prefix
`d66afb30`; RELION/RECOVAR initial-map file prefixes `4f83710c` and
`d77516a0`; exact shared canonical-array prefix `b617f90d`. There are no
pending preparation hashes. The subject commit must match exactly;
ancestry is insufficient. The RELION arm is complete; the case remains
pending until the RECOVAR full refinement and sealed two-engine FSC
analysis complete.

## Reproduction and artifact replay

From the repository root, this command re-hashes and replays every completed
10073/10345/10097 unmasked and masked artifact, verifies the partial and
matched-iteration 10202 records, and checks that this generated Markdown is fresh:

```bash
pixi run python scripts/summarize_em_k1_realdata_science_equivalence.py --verify-calibrations --verify-masked-support --verify-target-partial --check-markdown
```

The original unmasked producer submissions are recorded verbatim in their
sealed `SUBMITTED_JOBS.md` files:

```bash
sbatch --export=ALL,DATASET_ID=10073 /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/scripts/run_dataset_native.sbatch
sbatch --export=ALL,DATASET_ID=10345 /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/scripts/run_dataset_native.sbatch
sbatch --parsable --export=NONE /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution_replacement_10097_20260828T211155EDT/scripts/run_dataset_native_10097.sbatch
```

| Evidence | Frozen producer/collector reference | SHA-256 prefix |
| --- | --- | --- |
| 10073/10345 unmasked launcher | `/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/scripts/run_dataset_native.sbatch` | `f9544ae3eb4e` |
| 10073/10345 submission record | `/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/SUBMITTED_JOBS.md` | `cf984cb270ca` |
| 10097 unmasked launcher | `/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution_replacement_10097_20260828T211155EDT/scripts/run_dataset_native_10097.sbatch` | `b732039b378e` |
| 10097 submission record | `/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution_replacement_10097_20260828T211155EDT/SUBMITTED_JOBS.md` | `205e3f8ef27e` |
| Signed-FSC collector | `/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/scripts/collect_metrics.py` | `63d1a8f9f0a7` |
| 10073/10345 masked launcher | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/realdata_relion_masked_fsc_10073_10345_10097_r2_20260831/scripts/run_masked_fsc.sbatch` | `3ed871acbd6f` |
| 10073/10345 masked driver | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/realdata_relion_masked_fsc_10073_10345_10097_r2_20260831/scripts/run_masked_fsc.py` | `b8406ed10880` |
| 10097 masked launcher | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/realdata_relion_masked_fsc_10097_exactmask_20260831/scripts/run_exact_mask_postprocess.sbatch` | `71fbc466963e` |
| 10097 masked driver | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/realdata_relion_masked_fsc_10097_exactmask_20260831/scripts/run_exact_mask_postprocess.py` | `991a5214de7a` |
| Masked aggregate builder | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/realdata_relion_masked_fsc_final_10073_10345_10097_20260831/scripts/build_aggregate_report.py` | `36ce5201fe80` |

No original `sbatch` argv was separately sealed for the two masked-FSC jobs,
so none is reconstructed here. Their exact launchers and Python drivers are
pinned above, while the repository replay command verifies their retained
outputs without launching new science jobs.

## Diagnostics

The pinned producer searches a HEALPix order-1 proper-rotation grid, refines
at order 2, and then continuously refines a rotation vector and subpixel
translation on a 65-cubed compact fit using full-box shells through 32.
Its single transform is reported and
applied unchanged to merged, half-1, and half-2 maps. Aligned unmasked FSC
may replace only failed raw cross-engine gates; it cannot change the three
mandatory half-map-quality gates.

The producer also constructs one engine-symmetric soft mask from the two
aligned merged maps, hashes it, and applies it identically to both engines.
Masked within-engine and cross-engine FSC values are included in the report
but are never read by any acceptance gate, so masking cannot conceal poor
independent half-map quality.
All six diagnostic input-map hashes must exactly match the six corresponding
hashes in the external collector. Production evidence also binds and hashes
the launch manifest, finalizer command and canonical argv, and a separate
immutable execution envelope; any supplied binding is checked fail-closed.

## Frozen-mask masked FSC (reporting only)

Each dataset has one frozen mask used for every arm (registry
`docs/benchmarks/frozen_masks.json`; method in `docs/benchmarks/masked_fsc_method.md`).
Masked resolution is relion_postprocess `rlnFinalResolution` (RELION's convention), with the
sustained three-shell crossing in parentheses; AUCs use this scorecard's jointly resolved band.
No gate reads these values. All rows, including synthetic data: `docs/benchmarks/masked_fsc.md`.

| Run | Mask | Band | RELION masked (Å) | relax masked (Å) | RELION masked AUC | relax masked AUC | Unmasked AUC RELION / relax | Cross-engine masked AUC merged / h1 / h2 | GT masked AUC RELION / relax |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10073_q427a08bd8_mt19937` | `empiar10073_c1` `030ad3a85199` | 1-80 | 4.12 (4.09) | 4.12 (4.09) | 0.9230 | 0.9233 | 0.7193 / 0.7194 | 0.9987 / 0.9962 / 0.9964 | — |
| `empiar10073_cand2_087287024` | `empiar10073_c1` `030ad3a85199` | 1-80 | 4.12 (4.09) | 4.12 (4.09) | 0.9230 | 0.9230 | 0.7193 / 0.7193 | 0.9989 / 0.9968 / 0.9966 | — |
| `speedbench_10073_r1` | `empiar10073_c1` `030ad3a85199` | 1-80 | 4.12 (4.09) | 4.12 (4.09) | 0.9230 | 0.9225 | 0.7192 / 0.7187 | 0.9973 / 0.9926 / 0.9926 | — |
| `empiar10081_hcn1_relion_reference_14313014` | `empiar10081_hcn1_c4` `4003c7dea2f3` | — | 3.70 (3.66) | — | — | — | — / — | — | — |
| `hcn1_relax_a32798c_14332336` | `empiar10081_hcn1_c4` `4003c7dea2f3` | 1-79 | 3.70 (3.66) | 3.58 (3.54) | 0.9159 | 0.9156 | 0.7024 / 0.7021 | 0.9657 / 0.9303 / 0.9226 | — |
| `hcn1_relax_main_460a763_14364095` | `empiar10081_hcn1_c4` `4003c7dea2f3` | 1-80 | 3.70 (3.66) | 3.70 (3.66) | 0.9118 | 0.9117 | 0.6954 / 0.6954 | 0.9999 / 0.9998 / 0.9998 | — |
| `10097_q427a08bd8_mt19937` | `empiar10097_c1` `3c184a29e87c` | 1-44 | 5.78 (5.68) | 5.99 (5.88) | 0.8725 | 0.8724 | 0.6826 / 0.6824 | 0.9821 / 0.9603 / 0.9623 | — |
| `empiar10097_10k_B1_j1_abba_r2` | `empiar10097_c1` `3c184a29e87c` | 1-20 | 22.36 (15.97) | 22.36 (15.24) | 0.6804 | 0.6902 | 0.6610 / 0.6615 | 0.9947 / 0.9871 / 0.9855 | — |
| `empiar10097_10k_B2_j1_abba_r2` | `empiar10097_c1` `3c184a29e87c` | 1-20 | 22.36 (15.97) | 22.36 (15.24) | 0.6804 | 0.6930 | 0.6610 / 0.6625 | 0.9945 / 0.9869 / 0.9857 | — |
| `empiar10097_10k_B3_j2_baab_r2` | `empiar10097_c1` `3c184a29e87c` | 1-20 | 22.36 (15.97) | 22.36 (15.24) | 0.6804 | 0.6929 | 0.6610 / 0.6614 | 0.9947 / 0.9875 / 0.9860 | — |
| `empiar10097_10k_B4_j2_baab_r2` | `empiar10097_c1` `3c184a29e87c` | 1-20 | 22.36 (15.97) | 22.36 (15.97) | 0.6804 | 0.6893 | 0.6610 / 0.6600 | 0.9946 / 0.9868 / 0.9852 | — |
| `empiar10097_cand2_087287024` | `empiar10097_c1` `3c184a29e87c` | 1-44 | 5.78 (5.68) | 5.99 (5.50) | 0.8725 | 0.8740 | 0.6826 / 0.6827 | 0.9822 / 0.9610 / 0.9619 | — |
| `empiar10097_relion_repeat_r1` | `empiar10097_c1` `3c184a29e87c` | — | 6.10 (5.99) | — | — | — | — / — | — | — |
| `empiar10097_relion_repeat_r2` | `empiar10097_c1` `3c184a29e87c` | — | 5.99 (5.88) | — | — | — | — / — | — | — |
| `empiar10202_relion_reference_13217551` | `empiar10202_set6_i1` `dcc3fd17e7f7` | — | 2.12 (2.12) | — | — | — | — / — | — | — |
| `10345_q427a08bd8_mt19937` | `empiar10345_c1` `f8c6764657f8` | 1-48 | 5.31 (5.24) | 5.31 (5.24) | 0.9274 | 0.9274 | 0.6726 / 0.6725 | 0.9981 / 0.9964 / 0.9967 | — |
| `empiar10345_cand2_087287024` | `empiar10345_c1` `f8c6764657f8` | 1-48 | 5.31 (5.24) | 5.24 (5.17) | 0.9274 | 0.9273 | 0.6726 / 0.6722 | 0.9980 / 0.9963 / 0.9965 | — |
| `flipqual_64499e8_10345` | `empiar10345_c1` `f8c6764657f8` | 1-48 | 5.31 (5.24) | 5.31 (5.17) | 0.9273 | 0.9278 | 0.6728 / 0.6728 | 0.9981 / 0.9964 / 0.9965 | — |

## Current relax runs against every RELION run (reporting only)

Rows from `tests/baselines/relion_vs_relax_benchmarks.json` whose RELION reference has same-command
repeats. These are the current relax runs, not the frozen calibration runs above.

### EMPIAR-10073 (flip pair) (relax `64499e87c`)

relax against each same-command RELION run (band FSC-AUC over the scorecard band; masked columns use the frozen mask; thresholds: merged >= 0.95 and each half >= 0.90):

| RELION run | Jobs | Merged | Half 1 | Half 2 | Masked merged | Masked half 1 | Masked half 2 | Thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flip-job RELION arm | 14397072 | 0.9918 | 0.9840 | 0.9843 | 0.9989 | 0.9968 | 0.9970 | met |
| repeat 14397251 | 14397251 | 0.9904 | 0.9812 | 0.9820 | 0.9987 | 0.9962 | 0.9965 | met |
| speedbench r1 (--preread_images) | 14331579 | 0.9913 | 0.9826 | 0.9838 | 0.9989 | 0.9965 | 0.9968 | met |

RELION against RELION (same band FSC-AUCs):

| Pair | Merged | Half 1 | Half 2 |
| --- | ---: | ---: | ---: |
| flip-job RELION arm vs repeat 14397251 | 0.9907 | 0.9826 | 0.9818 |
| flip-job RELION arm vs speedbench r1 (--preread_images) | 0.9919 | 0.9847 | 0.9838 |
| repeat 14397251 vs speedbench r1 (--preread_images) | 0.9904 | 0.9815 | 0.9815 |

### EMPIAR-10097 (flip pair, cd26a5e)\* (relax `cd26a5e41`)

\* RELION's four runs split into its two known 10097 pose basins: this job's RELION arm with speedbench r1 (0.9780/0.9623/0.9622) and the d3d62ca arm with its repeat (0.9770/0.9596/0.9609); across basins 0.9287-0.9299. relax lands in the d3d62ca basin (0.9758/0.9605/0.9564 and 0.9761/0.9586/0.9586) and matches its own paired RELION arm only at the cross-basin level (0.9296/0.8820/0.8858). It passes under the user's 2026-09-23 rule (thresholds met against at least one same-command RELION run; band condition dropped). All comparisons: see the per-reference table.

relax against each same-command RELION run (band FSC-AUC over the scorecard band; masked columns use the frozen mask; thresholds: merged >= 0.95 and each half >= 0.90):

| RELION run | Jobs | Merged | Half 1 | Half 2 | Masked merged | Masked half 1 | Masked half 2 | Thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| this job's RELION arm | 14407806 | 0.9296 | 0.8820 | 0.8858 | 0.9824 | 0.9615 | 0.9625 | not met |
| d3d62ca job's RELION arm | 14400302 | 0.9758 | 0.9605 | 0.9564 | 0.9941 | 0.9875 | 0.9856 | met |
| d3d62ca job's RELION repeat | 14400302 | 0.9761 | 0.9586 | 0.9586 | 0.9941 | 0.9868 | 0.9861 | met |
| speedbench r1 (--preread_images) | 14331581 | 0.9297 | 0.8816 | 0.8857 | 0.9825 | 0.9615 | 0.9622 | not met |

RELION against RELION (same band FSC-AUCs):

| Pair | Merged | Half 1 | Half 2 |
| --- | ---: | ---: | ---: |
| this job's RELION arm vs d3d62ca job's RELION arm | 0.9299 | 0.8828 | 0.8861 |
| this job's RELION arm vs d3d62ca job's RELION repeat | 0.9287 | 0.8807 | 0.8856 |
| this job's RELION arm vs speedbench r1 (--preread_images) | 0.9780 | 0.9623 | 0.9622 |
| d3d62ca job's RELION arm vs d3d62ca job's RELION repeat | 0.9770 | 0.9596 | 0.9609 |
| d3d62ca job's RELION arm vs speedbench r1 (--preread_images) | 0.9298 | 0.8823 | 0.8862 |
| d3d62ca job's RELION repeat vs speedbench r1 (--preread_images) | 0.9288 | 0.8804 | 0.8858 |

### EMPIAR-10345 (flip pair) (relax `64499e87c`)

relax against each same-command RELION run (band FSC-AUC over the scorecard band; masked columns use the frozen mask; thresholds: merged >= 0.95 and each half >= 0.90):

| RELION run | Jobs | Merged | Half 1 | Half 2 | Masked merged | Masked half 1 | Masked half 2 | Thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flip-job RELION arm | 14397074 | 0.9822 | 0.9747 | 0.9761 | 0.9981 | 0.9964 | 0.9965 | met |
| repeat 14397251 | 14397251 | 0.9822 | 0.9751 | 0.9757 | 0.9981 | 0.9964 | 0.9965 | met |
| speedbench r1 (--preread_images) | 14331583 | 0.9818 | 0.9743 | 0.9754 | 0.9980 | 0.9962 | 0.9964 | met |

RELION against RELION (same band FSC-AUCs):

| Pair | Merged | Half 1 | Half 2 |
| --- | ---: | ---: | ---: |
| flip-job RELION arm vs repeat 14397251 | 0.9879 | 0.9829 | 0.9836 |
| flip-job RELION arm vs speedbench r1 (--preread_images) | 0.9864 | 0.9811 | 0.9812 |
| repeat 14397251 vs speedbench r1 (--preread_images) | 0.9866 | 0.9810 | 0.9820 |

### EMPIAR-10345 (flip pair, cd26a5e) (relax `cd26a5e41`)

relax against each same-command RELION run (band FSC-AUC over the scorecard band; masked columns use the frozen mask; thresholds: merged >= 0.95 and each half >= 0.90):

| RELION run | Jobs | Merged | Half 1 | Half 2 | Masked merged | Masked half 1 | Masked half 2 | Thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| this job's RELION arm | 14407833 | 0.9825 | 0.9750 | 0.9764 | 0.9980 | 0.9964 | 0.9965 | met |
| 64499e8 flip job's RELION arm | 14397074 | 0.9824 | 0.9747 | 0.9765 | 0.9981 | 0.9964 | 0.9966 | met |
| repeat 14397251 | 14397251 | 0.9823 | 0.9748 | 0.9761 | 0.9981 | 0.9964 | 0.9966 | met |
| speedbench r1 (--preread_images) | 14331583 | 0.9819 | 0.9738 | 0.9759 | 0.9980 | 0.9962 | 0.9965 | met |

RELION against RELION (same band FSC-AUCs):

| Pair | Merged | Half 1 | Half 2 |
| --- | ---: | ---: | ---: |
| this job's RELION arm vs 64499e8 flip job's RELION arm | 0.9871 | 0.9824 | 0.9820 |
| this job's RELION arm vs repeat 14397251 | 0.9870 | 0.9823 | 0.9817 |
| this job's RELION arm vs speedbench r1 (--preread_images) | 0.9882 | 0.9832 | 0.9840 |
| 64499e8 flip job's RELION arm vs repeat 14397251 | 0.9879 | 0.9829 | 0.9836 |
| 64499e8 flip job's RELION arm vs speedbench r1 (--preread_images) | 0.9864 | 0.9811 | 0.9812 |
| repeat 14397251 vs speedbench r1 (--preread_images) | 0.9866 | 0.9810 | 0.9820 |

### EMPIAR-10081 (HCN1, resident engine, timed pair) (relax `3c7e1c571`)

relax against each same-command RELION run (band FSC-AUC over the scorecard band; masked columns use the frozen mask; thresholds: merged >= 0.95 and each half >= 0.90):

| RELION run | Jobs | Merged | Half 1 | Half 2 | Masked merged | Masked half 1 | Masked half 2 | Thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| this job's RELION arm | 14410266 | 0.9991 | 0.9988 | 0.9988 | 0.9999 | 0.9998 | 0.9998 | met |
| reference 14313014 (seed 42) | 14313014 | 0.9709 | 0.9611 | 0.9621 | 0.9984 | 0.9972 | 0.9974 | met |
| seed repeat 14363852 (seed 20260924) | 14363852 | 0.9317 | 0.8002 | 0.7980 | 0.9918 | 0.9506 | 0.9489 | not met |

RELION against RELION (same band FSC-AUCs):

| Pair | Merged | Half 1 | Half 2 |
| --- | ---: | ---: | ---: |
| this job's RELION arm vs reference 14313014 (seed 42) | 0.9709 | 0.9612 | 0.9620 |
| this job's RELION arm vs seed repeat 14363852 (seed 20260924) | 0.9318 | 0.8003 | 0.7979 |
| reference 14313014 (seed 42) vs seed repeat 14363852 (seed 20260924) | 0.9296 | 0.7986 | 0.7965 |

### EMPIAR-10097 10k subset (resident engine) (relax `589ce095c`)

relax against each same-command RELION run (band FSC-AUC over the scorecard band; masked columns use the frozen mask; thresholds: merged >= 0.95 and each half >= 0.90):

| RELION run | Jobs | Merged | Half 1 | Half 2 | Masked merged | Masked half 1 | Masked half 2 | Thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| this job's RELION arm | 14445853 | 0.9953 | 0.9927 | 0.9929 | 0.9993 | 0.9979 | 0.9980 | met |
| 2026-09-16 sampler run | — | 0.9950 | 0.9927 | 0.9918 | 0.9992 | 0.9979 | 0.9977 | met |
| 2026-09-18 repeat c | — | 0.9948 | 0.9926 | 0.9913 | 0.9991 | 0.9979 | 0.9974 | met |
| 2026-09-18 repeat d | — | 0.9952 | 0.9913 | 0.9936 | 0.9993 | 0.9974 | 0.9983 | met |
| 2026-09-18 repeat e | — | 0.9787 | 0.9663 | 0.9662 | 0.9963 | 0.9903 | 0.9901 | met |
| bench attempt 1 RELION arm | 14434594 | 0.9786 | 0.9669 | 0.9657 | 0.9962 | 0.9900 | 0.9899 | met |
| bench attempt 2 RELION arm | 14445196 | 0.9950 | 0.9914 | 0.9926 | 0.9992 | 0.9974 | 0.9979 | met |

RELION against RELION (same band FSC-AUCs):

| Pair | Merged | Half 1 | Half 2 |
| --- | ---: | ---: | ---: |
| this job's RELION arm vs 2026-09-16 sampler run | 0.9990 | 0.9986 | 0.9983 |
| this job's RELION arm vs 2026-09-18 repeat c | 0.9983 | 0.9980 | 0.9966 |
| this job's RELION arm vs 2026-09-18 repeat d | 0.9961 | 0.9934 | 0.9941 |
| this job's RELION arm vs 2026-09-18 repeat e | 0.9792 | 0.9658 | 0.9682 |
| this job's RELION arm vs bench attempt 1 RELION arm | 0.9790 | 0.9661 | 0.9677 |
| this job's RELION arm vs bench attempt 2 RELION arm | 0.9957 | 0.9920 | 0.9943 |
| 2026-09-16 sampler run vs 2026-09-18 repeat c | 0.9987 | 0.9986 | 0.9974 |
| 2026-09-16 sampler run vs 2026-09-18 repeat d | 0.9957 | 0.9930 | 0.9935 |
| 2026-09-16 sampler run vs 2026-09-18 repeat e | 0.9789 | 0.9655 | 0.9677 |
| 2026-09-16 sampler run vs bench attempt 1 RELION arm | 0.9787 | 0.9659 | 0.9670 |
| 2026-09-16 sampler run vs bench attempt 2 RELION arm | 0.9957 | 0.9919 | 0.9944 |
| 2026-09-18 repeat c vs 2026-09-18 repeat d | 0.9956 | 0.9931 | 0.9930 |
| 2026-09-18 repeat c vs 2026-09-18 repeat e | 0.9790 | 0.9660 | 0.9677 |
| 2026-09-18 repeat c vs bench attempt 1 RELION arm | 0.9787 | 0.9663 | 0.9668 |
| 2026-09-18 repeat c vs bench attempt 2 RELION arm | 0.9955 | 0.9915 | 0.9942 |
| 2026-09-18 repeat d vs 2026-09-18 repeat e | 0.9786 | 0.9658 | 0.9666 |
| 2026-09-18 repeat d vs bench attempt 1 RELION arm | 0.9789 | 0.9660 | 0.9674 |
| 2026-09-18 repeat d vs bench attempt 2 RELION arm | 0.9975 | 0.9951 | 0.9971 |
| 2026-09-18 repeat e vs bench attempt 1 RELION arm | 0.9958 | 0.9946 | 0.9920 |
| 2026-09-18 repeat e vs bench attempt 2 RELION arm | 0.9786 | 0.9657 | 0.9669 |
| bench attempt 1 RELION arm vs bench attempt 2 RELION arm | 0.9790 | 0.9670 | 0.9674 |

## Code references

- `scripts/summarize_em_k1_realdata_science_equivalence.py`: scorecard validation, FSC band metrics, provenance gates, and rendering.
- `scripts/collect_em_k1_science_diagnostics.py`: continuous proper-SO(3)+translation fitting and common-mask FSC artifacts.
- `scripts/masked_fsc.py`: frozen masks, relion_postprocess masked FSC and the frozen-mask table.
- `tests/unit/test_summarize_em_k1_realdata_science_equivalence.py`: deterministic metric, provenance, calibration, and non-rescue tests.
- `/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/scripts/collect_metrics.py`: external signed-FSC artifact collector pinned by SHA-256 in the manifest.
