# RECOVAR / RELION EM Parity Scorecard

**K=1 fixed-suite score: 31 / 34 passing (34 / 34 evaluated; 34 / 34 intermediate-topology passes).**

**K=4 fixed-trajectory score: 41 / 60 direct class checks passing (9 / 15 iterations pass all classes).**

Suite: `k1-gui-grid0-local-highshell-full34` (version 2; denominator frozen at 34).
Frozen case-definition SHA-256: `9e3f2cb7192eb2cbf8a50181cf47de8562adfb98734bab05a736fb7d4d404fc1`.

A checked box means the complete autonomous FSC/FSC-AUC trajectory contract passed. Unchecked cases remain in the denominator. New diagnostics do not enter this suite; changing the case set or scientific definitions requires a new suite version.

The artifact-pinned fixture manifest is checked into the repository and binds all 34 cases (470,170,958,467 bytes) to exact file sizes and SHA-256 digests. Manifest SHA-256: `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`. Regenerated inputs are non-scoring replicates.

Acceptance uses shellwise FSC and normalized FSC-AUC, exact schedule/topology, convergence/finalization semantics, same-physical-GPU RELION/RECOVAR pairs, final all-data gridding correction always on (RELION `griddingCorrect`), and no forced K-class-like finalization. Correlation is not computed or gated.

Evidence snapshot: `em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v13`, generated `2026-09-24T13:19:31+00:00`, JSON SHA-256 `b76e50acf6a9e62bc76571a540c70edb7d6fb558ab0024a00a0fc4d84456435c` (`docs/math/em_k1_full34_superseding_ledger_v13.json`).
K=4 evidence snapshot: `k4-relion-cuda-4181d340-20260725`, JSON SHA-256 `bc10d0555488b22f0bc8d54afe5afc5288064ddb4708bd1c75f3b55dd4c0060a`.
Progress: +11 passing cases since the first frozen snapshot; +0 since the previous snapshot.

## K=1 fixed cases

| Done | Case | Fixture | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |
|---|---|---|---|---|---:|---:|---|
| [x] | `k1-01` | `baseline_100k_g256_white_noise1_bf80` | pass | pass | 0.999640760 | +0.000048888 | science 11384176; trajectory 11384362; intermediate 11384363 |
| [x] | `k1-02` | `more_images_200k_g256_white_noise1_bf80` | pass | pass | 0.999719401 | +0.000120823 | science 11501888; trajectory 11501907; intermediate 11501907 |
| [x] | `k1-03` | `more_images_300k_g256_white_noise1_bf80` | pass | pass | 0.999838550 | +0.000003100 | science 11587631; trajectory 11632847; intermediate 11632847 |
| [ ] | `k1-04` | `high_noise_100k_g256_white_noise3_bf80` | fail | pass | 0.992868806 | -0.000002752 | science 11384179; trajectory 11384368; intermediate 11384369 |
| [ ] | `k1-05` | `very_high_noise_100k_g256_white_noise10_bf80` | fail | pass | 0.987359540 | +0.000042922 | science 11384180; trajectory 11384370; intermediate 11384371 |
| [x] | `k1-06` | `noctf_control_100k_g256_white_noise3_bf80` | pass | pass | 0.998781294 | -0.000030798 | science 11384181; trajectory 11384372; intermediate 11384373 |
| [x] | `k1-07` | `anisotropic_100k_g256_white_noise1_bf80` | pass | pass | 0.999991524 | +0.000003243 | science 12694866; trajectory 12695233; intermediate 12695233 |
| [x] | `k1-08` | `anisotropic_high_noise_100k_g256_white_noise3_bf80` | pass | pass | 0.997847790 | +0.000003192 | science 11384183; trajectory 11384376; intermediate 11384377 |
| [x] | `k1-09` | `high_res_near_nyquist_100k_g384_white_noise1_bf0` | pass | pass | 0.996940930 | -0.000051335 | science 11432807; trajectory 11454201; intermediate 11432810 |
| [ ] | `k1-10` | `high_res_anisotropic_100k_g384_radial_noise3_bf0` | fail | pass | 0.984714379 | -0.000036073 | science 11421265; trajectory 11454202; intermediate 11421267 |
| [x] | `k1-11` | `small_baseline_3k_g128_white_noise1_bf80` | pass | pass | 0.999992948 | -0.000008921 | science 11384186; trajectory 11384382; intermediate 11384383 |
| [x] | `k1-12` | `small_very_high_noise_3k_g128_white_noise10_bf80` | pass | pass | 0.999987930 | -0.000005832 | science 11384187; trajectory 11384384; intermediate 11384385 |
| [x] | `k1-13` | `small_anisotropic_3k_g128_white_noise3_bf80` | pass | pass | 0.999785096 | +0.000009255 | science 11385531; trajectory 11385557; intermediate 11385558 |
| [x] | `k1-14` | `small_noctf_3k_g128_white_noise3_bf80` | pass | pass | 0.999275189 | -0.000069213 | science 11385532; trajectory 11385559; intermediate 11385560 |
| [x] | `k1-15` | `small_outliers_3k_g128_pct20_noise1_bf80` | pass | pass | 0.999909025 | +0.000029590 | science 11385533; trajectory 11385561; intermediate 11385562 |
| [x] | `k1-16` | `small_anisotropic_outliers_3k_g128_pct25_noise3_bf80` | pass | pass | 0.998491656 | -0.000144344 | science 11385534; trajectory 11385563; intermediate 11385564 |
| [x] | `k1-17` | `small_extra_particles_3k_g128_noise1_bf80` | pass | pass | 0.999999128 | +0.000002410 | science 11385535; trajectory 11385565; intermediate 11385566 |
| [x] | `k1-18` | `small_contrast_noise_scale_3k_g128_noise1_bf80` | pass | pass | 0.999955886 | +0.000109745 | science 11385536; trajectory 11385567; intermediate 11385568 |
| [x] | `k1-19` | `small_image_offset_3k_g128_noise1_bf80` | pass | pass | 0.999724848 | +0.000010612 | science 11385537; trajectory 11385569; intermediate 11385570 |
| [x] | `k1-20` | `small_high_res_radial_3k_g256_noise3_bf0` | pass | pass | 0.999989207 | -0.000003305 | science 11498687; trajectory 11498738; intermediate 11498738 |
| [x] | `k1-21` | `small_kent_angles_3k_g128_white_noise3_bf80` | pass | pass | 0.999990548 | -0.000001875 | science 11385539; trajectory 11385573; intermediate 11385574 |
| [x] | `k1-22` | `small_severe_outliers_3k_g128_radial_noise5_bf80` | pass | pass | 0.999533673 | +0.000042260 | science 12377247; trajectory 12377829; intermediate 12377829 |
| [x] | `k1-23` | `small_noctf_radial_3k_g128_noise3_bf80` | pass | pass | 0.999968104 | -0.000009885 | science 11501524; trajectory 11501622; intermediate 11501622 |
| [x] | `k1-24` | `small_kent_outliers_3k_g128_pct20_noise3_bf80` | pass | pass | 0.999741866 | +0.000000693 | science 11655858; trajectory 11655936; intermediate 11655936 |
| [x] | `k1-25` | `tiny_baseline_1k_g128_white_noise3_bf80` | pass | pass | 0.999995905 | -0.000000489 | science 11385543; trajectory 11385581; intermediate 11385582 |
| [x] | `k1-26` | `tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80` | pass | pass | 0.999992192 | +0.000000970 | science 12371765; trajectory 12371765; intermediate 12371765 |
| [x] | `k1-27` | `small_extreme_outliers_3k_g128_pct70_noise1_bf80` | pass | pass | 0.999818380 | +0.000029889 | science 11385545; trajectory 11385587; intermediate 11385588 |
| [x] | `k1-28` | `small_kent_extra_offset_3k_g128_noise3_bf80` | pass | pass | 0.999999281 | -0.000003339 | science 11384203; trajectory 11384427; intermediate 11384428 |
| [x] | `k1-29` | `small_low_noise_3k_g128_white_noise0p2_bf80` | pass | pass | 0.999997367 | -0.000006132 | science 11384204; trajectory 11384429; intermediate 11384430 |
| [x] | `k1-30` | `small_low_noise_kent_3k_g128_white_noise0p2_bf80` | pass | pass | 0.999999781 | +0.000000894 | science 11384205; trajectory 11384433; intermediate 11384434 |
| [x] | `k1-31` | `mid_10k_g128_white_noise1_bf80` | pass | pass | 0.999981664 | -0.000019323 | science 11384206; trajectory 11384436; intermediate 11384437 |
| [x] | `k1-32` | `mid_10k_kent_g128_radial_noise3_bf80` | pass | pass | 0.999963868 | -0.000011179 | science 11635967; trajectory 11638090; intermediate 11638090 |
| [x] | `k1-33` | `max_images_400k_g128_white_noise1_bf80` | pass | pass | 0.999997977 | -0.000008541 | science 11508260; trajectory 11508286; intermediate 11508286 |
| [x] | `k1-34` | `max_images_400k_g128_radial_noise3_nonuniform_bf80` | pass | pass | 0.996376446 | -0.000022918 | science 11384210; trajectory 11384443; intermediate 11384444 |

## K=4 fixed trajectory

Each row contains four class-level FSC-AUC checks at the frozen `0.995` gate. A checked row passes all four classes; unchecked rows and failed class checks remain in their frozen denominators.

| Done | Iteration | Class checks passed |
|---|---:|---:|
| [x] | 1 | 4 / 4 |
| [x] | 2 | 4 / 4 |
| [x] | 3 | 4 / 4 |
| [x] | 4 | 4 / 4 |
| [x] | 5 | 4 / 4 |
| [x] | 6 | 4 / 4 |
| [x] | 7 | 4 / 4 |
| [x] | 8 | 4 / 4 |
| [x] | 9 | 4 / 4 |
| [ ] | 10 | 3 / 4 |
| [ ] | 11 | 0 / 4 |
| [ ] | 12 | 2 / 4 |
| [ ] | 13 | 0 / 4 |
| [ ] | 14 | 0 / 4 |
| [ ] | 15 | 0 / 4 |

## Progress history

| Snapshot | Suite | Date (UTC) | Commit boundary | Passed | Δ passed | Failed | Not evaluated/error |
|---|---:|---|---|---:|---:|---:|---:|
| `strict-k1-v1-old-head-20260721` | 1 | 2026-07-21T04:33:00.281935+00:00 | `ac5177d2b0cd` | 20 | — | 12 | 2 |
| `strict-k1-v3-20260721` | 1 | 2026-07-21T10:35:40.626248+00:00 | `ac5177d2b0cd`, `9d1722781e1d` | 21 | +1 | 13 | 0 |
| `strict-k1-v4-20260722` | 1 | 2026-07-22T15:57:09.593124+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db` | 22 | +1 | 12 | 0 |
| `strict-k1-v5-20260722` | 1 | 2026-07-22T19:00:51.329249+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038` | 23 | +1 | 11 | 0 |
| `strict-k1-v6-20260724` | 1 | 2026-07-24T01:04:11.826284+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038`, `84143872a517`, `a2be302cdc08` | 25 | +2 | 9 | 0 |
| `strict-k1-v7-20260726` | 1 | 2026-07-26T15:48:00+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038`, `84143872a517`, `a2be302cdc08`, `4c8b043a9b80` | 26 | +1 | 8 | 0 |
| `strict-k1-v8-20260726` | 1 | 2026-07-26T20:50:08+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038`, `84143872a517`, `a2be302cdc08`, `4c8b043a9b80`, `916ab17a4c80` | 27 | +1 | 7 | 0 |
| `strict-k1-v9-20260727` | 1 | 2026-07-27T07:29:46+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038`, `84143872a517`, `a2be302cdc08`, `4c8b043a9b80`, `916ab17a4c80`, `31c4a0ca203b` | 28 | +1 | 6 | 0 |
| `strict-k1-v10-20260814` | 1 | 2026-08-14T10:59:00+00:00 | `36dac0171859`, `7f0e2348dbee` | 29 | +1 | 5 | 0 |
| `strict-k1-v11-20260814` | 1 | 2026-08-14T14:46:03+00:00 | `e791e87502b5` | 30 | +1 | 4 | 0 |
| `strict-k1-v12-20260821` | 1 | 2026-08-21T05:30:00+00:00 | `fdec6f931d22` | 31 | +1 | 3 | 0 |
| `strict-k1-suite2-gridding-20260924` | 2 | 2026-09-24T13:19:31+00:00 | `df88eab29940`, `d4da7fdf99f1` | 31 | +0 | 3 | 0 |

## Suite version 2 regeneration

Version 1 scored final merged maps written without RELION's final gridding correction (`backprojector.cpp` `griddingCorrect`); relax now always applies it. The correction is the last real-space step of the reconstruction, so each case's final merged-map metrics were regenerated post hoc with RELION griddingCorrect on the saved final maps (`scripts/regenerate_em_k1_scorecard_final_gridding.py`): the saved uncorrected `final_merged.mrc` is divided by the radial sinc^2 (padding factor 2) and re-scored with the audit's FSC shells and sign policy. Each case first reproduced its recorded version 1 values from the same files. Numbered-iteration metrics, split halves and RELION maps are unchanged. Version 1 remains as history in `docs/math/em_relion_parity_scorecard_v1.json` (SHA-256 `da7b8b502abd0f549d8ac2b691dd415d372819dcaf48926550e98b7b546afcef`).

No case changed pass/fail under the unchanged thresholds.

| Case | v1 result | v1 cross-engine FSC-AUC | v1 GT delta | v2 result | v2 cross-engine FSC-AUC | v2 GT delta |
|---|---|---:|---:|---|---:|---:|
| `k1-01` | pass | 0.998379294 | +0.008163340 | pass | 0.999640760 | +0.000048888 |
| `k1-02` | pass | 0.998574606 | +0.005625197 | pass | 0.999719401 | +0.000120823 |
| `k1-03` | pass | 0.998782733 | +0.005426332 | pass | 0.999838550 | +0.000003100 |
| `k1-04` | fail | 0.991556309 | +0.003869282 | fail | 0.992868806 | -0.000002752 |
| `k1-05` | fail | 0.985743479 | +0.000544950 | fail | 0.987359540 | +0.000042922 |
| `k1-06` | pass | 0.997522945 | +0.005563842 | pass | 0.998781294 | -0.000030798 |
| `k1-07` | pass | 0.998345357 | -0.000382787 | pass | 0.999991524 | +0.000003243 |
| `k1-08` | pass | 0.996260789 | +0.001007928 | pass | 0.997847790 | +0.000003192 |
| `k1-09` | pass | 0.995510893 | +0.003664545 | pass | 0.996940930 | -0.000051335 |
| `k1-10` | fail | 0.983006504 | +0.000128347 | fail | 0.984714379 | -0.000036073 |
| `k1-11` | pass | 0.998515876 | +0.019981607 | pass | 0.999992948 | -0.000008921 |
| `k1-12` | pass | 0.998135578 | +0.002422120 | pass | 0.999987930 | -0.000005832 |
| `k1-13` | pass | 0.997569995 | +0.011223852 | pass | 0.999785096 | +0.000009255 |
| `k1-14` | pass | 0.997775943 | +0.017579713 | pass | 0.999275189 | -0.000069213 |
| `k1-15` | pass | 0.998431014 | +0.019058454 | pass | 0.999909025 | +0.000029590 |
| `k1-16` | pass | 0.996556471 | +0.008243245 | pass | 0.998491656 | -0.000144344 |
| `k1-17` | pass | 0.998794039 | +0.016078006 | pass | 0.999999128 | +0.000002410 |
| `k1-18` | pass | 0.998712222 | +0.014739591 | pass | 0.999955886 | +0.000109745 |
| `k1-19` | pass | 0.998259358 | +0.020106905 | pass | 0.999724848 | +0.000010612 |
| `k1-20` | pass | 0.998129368 | +0.001149427 | pass | 0.999989207 | -0.000003305 |
| `k1-21` | pass | 0.998345537 | +0.010110173 | pass | 0.999990548 | -0.000001875 |
| `k1-22` | pass | 0.997767346 | +0.009642596 | pass | 0.999533673 | +0.000042260 |
| `k1-23` | pass | 0.998342408 | +0.012298496 | pass | 0.999968104 | -0.000009885 |
| `k1-24` | pass | 0.998090087 | +0.008280115 | pass | 0.999741866 | +0.000000693 |
| `k1-25` | pass | 0.998192576 | +0.009181804 | pass | 0.999995905 | -0.000000489 |
| `k1-26` | pass | 0.997747377 | +0.008697039 | pass | 0.999992192 | +0.000000970 |
| `k1-27` | pass | 0.998332271 | +0.010086417 | pass | 0.999818380 | +0.000029889 |
| `k1-28` | pass | 0.998534963 | +0.016603039 | pass | 0.999999281 | -0.000003339 |
| `k1-29` | pass | 0.998867525 | +0.014987020 | pass | 0.999997367 | -0.000006132 |
| `k1-30` | pass | 0.998823366 | +0.013967656 | pass | 0.999999781 | +0.000000894 |
| `k1-31` | pass | 0.998725941 | +0.016924536 | pass | 0.999981664 | -0.000019323 |
| `k1-32` | pass | 0.998274347 | +0.003849433 | pass | 0.999963868 | -0.000011179 |
| `k1-33` | pass | 0.999734254 | +0.000244294 | pass | 0.999997977 | -0.000008541 |
| `k1-34` | pass | 0.995757412 | +0.002869240 | pass | 0.996376446 | -0.000022918 |

## Frozen-mask masked FSC (reporting only)

Each fixture has one frozen mask made from its ground-truth map (registry `docs/benchmarks/frozen_masks.json`,
method `docs/benchmarks/masked_fsc_method.md`). Maps are those of the autonomous run
`em_k1_guigrid_localhighshell_full34_autonomous_ac5177d2_20260719T174000Z`; a case whose scorecard science job
differs was superseded by a later run and its masked values describe the ac5177d2 maps. No gate reads these values.
These values predate suite version 2 and were not regenerated with the final gridding correction.

| Case | Maps job | Scorecard job? | RELION masked (Å) | relax masked (Å) | Masked AUC RELION / relax | Cross-engine masked AUC | GT masked AUC RELION / relax |
|---|---|---|---:|---:|---:|---:|---:|
| `k1-01` | 11384176 | yes | 4.35 | 4.35 | 0.9220 / 0.9220 | 1.0000 | 0.7794 / 0.7795 |
| `k1-02` | 11385530 | no | 4.32 | 4.32 | 0.9124 / 0.9124 | 1.0000 | 0.6666 / 0.6666 |
| `k1-03` | 11384178 | no | 4.28 | 4.28 | 0.9226 / 0.9226 | 1.0000 | 0.6726 / 0.6726 |
| `k1-04` | 11384179 | yes | 6.80 | 6.80 | 0.9133 / 0.9133 | 0.9998 | 0.7502 / 0.7502 |
| `k1-05` | 11384180 | yes | 20.15 | 20.15 | 0.9421 / 0.9421 | 0.9999 | 0.6269 / 0.6271 |
| `k1-06` | 11384181 | yes | 4.65 | 4.65 | 0.9295 / 0.9295 | 1.0000 | 0.7626 / 0.7626 |
| `k1-07` | 11384182 | no | 8.92 | 8.92 | 0.8647 / 0.8638 | 0.9987 | 0.4810 / 0.4988 |
| `k1-08` | 11384183 | yes | 8.92 | 8.92 | 0.8537 / 0.8531 | 1.0000 | 0.4892 / 0.4893 |
| `k1-09` | 11384224 | no | 3.70 | — | — / — | — | — / — |
| `k1-10` | 11384225 | no | 19.43 | — | — / — | — | — / — |
| `k1-11` | 11384186 | yes | 8.77 | 8.77 | 0.8826 / 0.8826 | 1.0000 | 0.9466 / 0.9466 |
| `k1-12` | 11384187 | yes | 25.90 | 25.90 | 0.9573 / 0.9573 | 1.0000 | 0.7551 / 0.7551 |
| `k1-13` | 11385531 | yes | 21.76 | 21.76 | 0.8964 / 0.8964 | 1.0000 | 0.7954 / 0.7954 |
| `k1-14` | 11385532 | yes | 9.38 | 9.38 | 0.9075 / 0.9075 | 1.0000 | 0.9224 / 0.9223 |
| `k1-15` | 11385533 | yes | 9.07 | 9.07 | 0.9058 / 0.9059 | 1.0000 | 0.9501 / 0.9502 |
| `k1-16` | 11385534 | yes | 20.92 | 20.92 | 0.9271 / 0.9270 | 1.0000 | 0.7612 / 0.7612 |
| `k1-17` | 11385535 | yes | 8.63 | 8.63 | 0.9272 / 0.9272 | 1.0000 | 0.9477 / 0.9477 |
| `k1-18` | 11385536 | yes | 8.63 | 8.63 | 0.9164 / 0.9164 | 1.0000 | 0.9085 / 0.9086 |
| `k1-19` | 11385537 | yes | 8.77 | 8.77 | 0.8788 / 0.8788 | 1.0000 | 0.9462 / 0.9462 |
| `k1-20` | 11385538 | no | 24.73 | 24.73 | 0.9576 / 0.9577 | 0.9999 | — / — |
| `k1-21` | 11385539 | yes | 14.32 | 14.32 | 0.9334 / 0.9334 | 1.0000 | 0.8809 / 0.8809 |
| `k1-22` | 11385540 | no | 14.70 | 15.11 | 0.9084 / 0.9084 | 0.9984 | 0.6435 / 0.6418 |
| `k1-23` | 11385541 | no | 10.88 | 10.88 | 0.9030 / 0.9013 | 0.9976 | 0.8647 / 0.8716 |
| `k1-24` | 11385542 | no | 16.00 | 16.00 | 0.9327 / 0.9322 | 0.9999 | 0.8606 / 0.8607 |
| `k1-25` | 11385543 | yes | 21.76 | 21.76 | 0.9259 / 0.9259 | 1.0000 | 0.9271 / 0.9271 |
| `k1-26` | 11385544 | no | 20.92 | 20.92 | 0.9118 / 0.9114 | 0.9993 | 0.6653 / 0.6661 |
| `k1-27` | 11385545 | yes | 14.32 | 14.32 | 0.9598 / 0.9598 | 1.0000 | 0.7767 / 0.7767 |
| `k1-28` | 11384203 | yes | 8.77 | 8.77 | 0.8994 / 0.8994 | 1.0000 | 0.8680 / 0.8680 |
| `k1-29` | 11384204 | yes | 8.50 | 8.50 | 0.9286 / 0.9286 | 1.0000 | 0.9530 / 0.9530 |
| `k1-30` | 11384205 | yes | 8.63 | 8.63 | 0.9121 / 0.9121 | 1.0000 | 0.8678 / 0.8678 |
| `k1-31` | 11384206 | yes | 8.63 | 8.63 | 0.9237 / 0.9237 | 1.0000 | 0.9573 / 0.9572 |
| `k1-32` | 11384207 | no | 27.20 | 27.20 | 0.9506 / 0.9509 | 0.9998 | 0.7912 / 0.7917 |
| `k1-33` | 11384208 | no | 8.50 | 8.50 | 0.9834 / 0.9969 | 0.9885 | 0.9629 / 0.9835 |
| `k1-34` | 11384210 | yes | 8.77 | 8.77 | 0.8448 / 0.8446 | 0.9995 | 0.5142 / 0.5143 |

## Archived experiment history

The [original post-snapshot diagnostics](https://github.com/ma-gilles/recovar-experiments/blob/6ced5f78857ad7cc75d5eee257ccbaa8dc17ffe7/docs/math/em_relion_parity_scorecard.md#current-k1-case-7-bounded-metric)
retain the historical interventions, failed runs and evidence hashes. Their source-specific
results do not qualify the current implementation or change the frozen suite above.

Generate this PR-ready table with:

```bash
pixi run python scripts/summarize_em_relion_parity_scorecard.py
```

Verify that the checked scorecard and frozen snapshots are current with:

```bash
pixi run python scripts/summarize_em_relion_parity_scorecard.py \
  --check docs/math/em_relion_parity_scorecard.md
```

After a terminal strict auditor passes, build a fail-closed candidate
superseding ledger with `--proposal-output`. The command validates the
pinned fixture-manifest bytes and re-hashes every materialized byte, clean source and
submitted job/case-table identity, same physical GPU, autonomous
FSC/topology audits, convergence/finalization contract, and evidence
hashes. It never mutates the checked scorecard. For example:

```bash
pixi run python scripts/summarize_em_relion_parity_scorecard.py \
  --proposal-previous-ledger /absolute/path/to/current-ledger.json \
  --proposal-ledger-schema em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v14 \
  --proposal-generated-utc 2026-07-26T21:00:00+00:00 \
  --proposal-status-note "Case k1-NN passed immutable strict evidence." \
  --proposal-evidence 'k1-NN|/absolute/path/to/case-root|SCIENCE_JOB|AUDIT_JOB' \
  --proposal-output /absolute/path/to/proposed-ledger.json
```

Launch a scoring rerun with `--scorecard`. This fail-closed mode requires the
checked-in fixture manifest/root pair and forces autonomous RELION pairing,
per-iteration RECOVAR maps and valid convergence-only finalization. Final
all-data maps are always gridding-corrected; the launcher refuses the retired
`RELAX_FINAL_ALL_DATA_GRID_CORRECT` selector. For example:

```bash
EM_K1_MATRIX_FIXTURE_MANIFEST="$PWD/docs/math/em_relion_parity_fixture_manifest_v2.json" \
EM_K1_MATRIX_FIXTURE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex \
EM_K1_MATRIX_CASES=2,3 \
./scripts/run_em_k1_robustness_matrix_slurm.sh --scorecard
```
