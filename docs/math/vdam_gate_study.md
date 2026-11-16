# VDAM gate study: is the shrinkage gate the bottleneck? (final report, 2026-11-16)

Branch `gate-study` (from `origin/ppca_schedule` 65417d0e). Output root `/scratch/gpfs/CRYOEM/gilleslab/em_work/gate_study_20261113/`.
Predeclarations, adjustments, decisions and the RESULT: `PREDECLARED.md` (append-only). GPU tally and run log: `NOTES.md`.

## Conclusion

On the eight synthetic suite datasets the VDAM gate is not the bottleneck of the end metrics. Phase 0 compared the
current gate (A) with two oracle gates computed from the ground truth: the task's oracle O (W = F/(F + 1/P)) and a
code-matched oracle Ocm, whose noise model passed a known-truth check. The predeclared rule was met only formally:
O beats A on decoded FSC-AUC on Cshift n1 and D20 n1 by +.0013, beyond A's seed range (.0005 and .0012). The same O is
worse beyond spread on C n1 (-.0030) and C n3 (-.0153). Ocm, the validated oracle, is neither better nor worse beyond spread
anywhere on the primary metric. The mechanism tables explain why: late in a run the current gate is within .03 (shell
average) of Ocm on every dataset, so a better gate can gain at most about .002 FSC-AUC here. Lead and owner stopped the
study after Phase 0 (2026-11-16). No Phase 1 arm was run.

Two departures of the gate from its oracle are real:
- **H1 (signal feedback) holds on a zero-signal channel.** On H1 the loading channel has no true signal (oracle 0), and
  the gate stays at .95-.99 for the whole run. On the primary metric this costs nothing (decoded .976 against .978,
  within spread). Closing it (O or Ocm) does improve two secondary metrics beyond spread: per-image FSC-AUC .958 to .978,
  and suite pose 4.25 to 4.17 deg.
- **The newest shells are under-opened.** Shells 30-31 (opened at iterations 2376 and 2500) end at gate .16-.44 against
  Ocm .50-.93.

## 1. What was run

| item | value |
|---|---|
| code | gate-study fae5525e (training, frozen snapshot `snapshots/fae5525e`); default path = ppca_schedule 801bd46b/65417d0e (docs-only difference) plus two off-by-default options: `--gate-oracle FILE --gate-oracle-key KEY` (replace the estimated gate by a [shells, 2] table) and `--gate-log` (per-shell sqrt(u) and effective gate) |
| settings | s3_vdam_uniform of 18.16/18.17: VDAM, uniform direction prior, 3000 updates of 300 images (balanced halves), box-64 recipe stages (shell 11 to 31, HEALPix 1/2/3), +-2 px in 1 px, oversampling 0; q 10 (C, Cshift), 9 (T10), 2 (D, H1); Cshift only: best-v1 shift stages `[[1,8,2],[1626,8,1]]`, offset sigma held at 20 A |
| datasets | C n1, C n3, Cshift n1, T10 n1, D10 n1, D20 n1, D40 n1, H1 n3 (box-64 crops, ppca_suite_b150_20261102); training STARs converted by `scripts/prepare_vdam_ppca_star.py` (data identical to Roey's) |
| arms | A (current gate, logged); O (task's oracle, fudge 1); Ocm (code-matched oracle, section 2) |
| seeds | 11, 12, 13 |
| hardware | training on H100 (A and O side by side in one 2-GPU job; Ocm in 1-GPU jobs); suite GPU stage on A100 |
| scoring | metric suite v3.2 at checkpoint 3000, on mean-negated checkpoint copies (section 5); CPU stage on Polar (polarscore) |
| jobs | P0 training 17476513-17476520 (A, O), P0b 17480342-17480349 (Ocm); scoring of record: GPU 17553142, 17642810/12/13, 17642827/28/29; Polar 425999, 426000, 426605-... (polarscore submitted.tsv, tags gsnV, gsnP, gsnS) |

Commands: `tools/train_arm.sh` / `tools/train_arm_v2.sh` write each run's full command, commit, host, GPU and job id to
`runs/<TAG>/<ARM>/seed<S>/cmd.txt`.

## 2. The oracle, and two corrections to the prompt's derivation

The oracle tables (`scripts/gate_study/oracle_gate.py`, `oracle/<TAG>.npz`) use only `evaluation/`. Per shell,
snr = N kappa(r) snr_img(r) P_channel/P_probe. Here snr_img comes from the dataset's noise probe (clean signal against
its noise), kappa = n2D(r)/n3D(r) observations per voxel per image, and the channel powers come from the
population-weighted mean and the top-q loadings sqrt(lambda_j) u_j. W = snr/(1+snr) is the task's oracle O. The shared
loading channel uses the mean of the per-PC snr, which is the MSE-optimal common scalar.

Correction 1, interpolation and mask. The task takes the voxel noise of the full-data reconstruction as 1/F. A
known-truth check measured otherwise (`scripts/gate_study/oracle_check.py`, job 17479461). It reconstructs one state per
dataset at its true rotations with the code's linear interpolation and mask, and takes the half-map FSC. The measured
SNR is 3.0x the prediction unmasked: linear interpolation averages neighbouring voxels' noise ((2/3)^3 = .30). It is
5.6x masked, and the factor is flat over shells 13-30 and the same on H1, C n3, T10 and D10.

The code-matched oracle Ocm = W(c_m snr), with c_m = 5.645 (shells 0-30) and 2.746 (edge shell 31), was fitted on H1
alone (`scripts/gate_study/oracle_code_matched.py`). Held-out error of the masked full-data weight, max over shells
13-30:

| oracle | max error on held-out sets |
|---|---|
| Ocm | .008 (C n1), .023 (C n3), .022 (D10), .007 (T10) |
| task's O | .15-.34 |
| analytic mask-aware snr/f | up to .08 (dropped) |

The loading channel was checked separately (`scripts/gate_study/oracle_check_loading.py`, job 17480409). Two states form
an exact rank-1 PPCA with known z = -1/+1. The measured factor is median 5.57 (IQR 5.04-5.95) against 5.645, so the
mean-channel factor carries over.

Correction 2, the target. The code's gate (derivation doc 7.1) targets the half-map FSC, P/(P + 2/F), not the
full-data Wiener weight P/(P + 1/F). With the default fudge schedule (1 to 4) the late target is 2x the Wiener SNR.
This difference is visible in the gate's "gap to target" readouts but is small next to correction 1.

Other prompt points that did not hold or were not applicable:
- `relax ppca_initial_model` refuses q = 0. The mean channel served as the single-volume case, and H1 n3 as the
  homogeneous set.
- The task's metric list (ARI, subspace, R2) conflicts with SHARED_RULES (v3.2 headline only). Marc chose v3.2.

## 3. Results (suite v3.2, checkpoint 3000, 3 seeds; full tables `/scratch/gpfs/CRYOEM/gilleslab/em_work/gate_study_20261113/analysis/tables_by_metric.md`)

Decoded-class FSC-AUC (primary; D sets: distinctive protein). Verdicts against A: "better/worse beyond spread" = same
direction in all three seed pairs and a mean difference larger than A's seed range.

| dataset | mean-only | true model | A | O | Ocm | O vs A | Ocm vs A |
|---|---|---|---|---|---|---|---|
| C n1 | .837 | .992 | .909 [.908, .909] | .906 [.905, .906] | .909 [.908, .909] | worse | within |
| C n3 | .837 | .979 | .881 [.880, .881] | .865 [.865, .866] | .879 [.878, .880] | worse | within |
| Cshift n1 | .837 | n/a | .879 [.879, .879] | .880 [.879, .881] | .880 [.879, .881] | better (+.0013) | within |
| T10 n1 | .382 | .996 | .617 [.563, .646] | .626 [.584, .658] | .594 [.556, .651] | within | within |
| D10 n1 | .669 | .997 | .883 [.881, .886] | .886 [.884, .887] | .871 [.834, .891] | within | within |
| D20 n1 | .849 | .996 | .933 [.932, .933] | .934 [.934, .935] | .934 [.933, .936] | better (+.0013) | within |
| D40 n1 | .944 | .995 | .955 [.953, .956] | .953 [.952, .955] | .955 [.954, .957] | within | within |
| H1 n3 | 1.000 | n/a | .976 [.975, .977] | .978 [.977, .978] | .978 [.977, .978] | within | within |

Suite pose median (deg; decoded frame; D sets: distinctive protein):

| dataset | A | O | Ocm | O vs A | Ocm vs A |
|---|---|---|---|---|---|
| C n1 | 3.97 [3.96, 3.98] | 3.92 [3.92, 3.93] | 3.96 [3.96, 3.97] | better | within |
| C n3 | 4.48 [4.47, 4.48] | 4.48 [4.48, 4.48] | 4.46 [4.46, 4.47] | within | within |
| Cshift n1 | 4.65 [4.62, 4.66] | 4.70 [4.68, 4.73] | 4.62 [4.62, 4.63] | worse | within |
| T10 n1 | 27.0 [9.5, 45.2] | 49.0 [9.1, 92.0] | 47.2 [9.7, 81.1] | within | within |
| D10 n1 | 3.81 [3.77, 3.85] | 3.79 [3.76, 3.82] | 6.34 [3.79, 11.42] | within | worse (Ocm s12) |
| D20 n1 | 3.76 [3.72, 3.79] | 3.77 [3.75, 3.78] | 3.80 [3.78, 3.82] | within | within |
| D40 n1 | 3.83 [3.80, 3.87] | 3.83 [3.73, 3.96] | 3.76 [3.72, 3.79] | within | within |
| H1 n3 | 4.25 [4.24, 4.26] | 4.17 [4.17, 4.18] | 4.17 [4.16, 4.17] | better | better |

T10 states solved (per-state suite pose median < 10 deg): A 4.3 [4, 5], O 4.0 [3, 5], Ocm 3.3 [2, 5]. T10 pose is bimodal:
each arm has seeds near 9 deg and seeds at 45-92 deg.

Secondary metrics move in mixed directions. They are reported, not used for the verdict.
- O's more closed loading gate raises II z->E beyond spread on C n1 (.672 to .766), C n3 (.932 to .981) and Cshift
  (.821 to .843). It also raises BP-class FSC-AUC on C n1 and Cshift, while lowering decoded FSC-AUC on C n1 and C n3.
  This is the same trade as 18.16, where the more closed power gate gave a better loading subspace and a worse ARI on C n3.
- Ocm is within spread of A on almost every secondary metric. The exceptions:
  - better: per-image (Cshift, H1), BP-class (Cshift), BP gain (C n3, Cshift);
  - worse: N_struct (D40), H1 BP gain, which is about 0 in every arm;
  - worse on D10 seed 12, which posed badly at 11.4 deg (decoded .834).

## 4. Mechanism: the gate against the oracles

Gate minus oracle, averaged over shells 1..radius and seeds, for iteration windows 101-500 / 501-1500 / 1501-2500 /
2501-3000. Each cell is mean channel / loading channel, for the baseline A (`/scratch/gpfs/CRYOEM/gilleslab/em_work/gate_study_20261113/analysis/mechanism_P0.md`).

| dataset | vs Ocm, 101-500 | 501-1500 | 1501-2500 | 2501-3000 | vs task O, 2501-3000 |
|---|---|---|---|---|---|
| C n1 | -.002/-.004 | -.006/-.008 | -.015/-.033 | -.010/-.024 | -.001/+.105 |
| C n3 | -.002/+.002 | -.006/+.011 | -.015/+.008 | -.009/+.023 | +.017/+.235 |
| Cshift n1 | -.002/-.004 | -.008/-.006 | -.018/-.024 | -.013/-.015 | -.004/+.114 |
| T10 n1 | -.000/-.001 | -.002/-.004 | -.023/-.031 | -.023/-.036 | +.022/+.013 |
| D10 n1 | +.008/+.000 | +.011/-.001 | +.000/-.027 | +.014/-.030 | +.152/+.042 |
| D20 n1 | +.002/-.001 | +.001/-.003 | -.017/-.027 | -.015/-.030 | +.060/+.017 |
| D40 n1 | -.001/-.002 | -.004/-.004 | -.022/-.024 | -.024/-.027 | +.004/+.008 |
| H1 n3 | -.002/+.990 | -.007/+.979 | -.014/+.955 | -.010/+.947 | +.011/+.947 |

Per-shell picture at iteration 3000 (loading channel; the mean channel is close to 1 everywhere):
`/scratch/gpfs/CRYOEM/gilleslab/em_work/gate_study_20261113/analysis/gate_vs_oracle_loading.png` and `/scratch/gpfs/CRYOEM/gilleslab/em_work/gate_study_20261113/analysis/gate_vs_oracle_mean.png`. A is the seed mean, with the seed range as
a band; dashed is Ocm, dotted is O. On H1 the O and Ocm lines lie on top of each other at 0.

Estimates against the truth, late window (2501-3000), baseline, geometric mean over shells:
- The logged signal (shell mean |theta|^2) is .72-.87 of the true power for the C/H1 mean and 1.1-4.5 for the D/T10
  mean. For the loadings it is only .11-.65 of the true power: the learned loadings are under-powered.
- The scaled disagreement is .14-.33 of the task's noise target (mean) and .04-.15 (loadings). For the mean this is the
  interpolation/mask factor of section 2 (1/5.6 = .18 masked, 1/3 unmasked). Both estimates are biased low, and their
  ratio, the gate, lands near Ocm.

Second moment (H4): in the baseline the median sqrt(u) over shells late in the run is .80-1.04 (mean) and .94-1.22
(loadings). The
effective gate phi/(phi + (1-phi) sqrt(u)) is within .04 of phi everywhere. The 1/sqrt(u) factor barely moves the
fixed point here. My pre-run prediction of sqrt(u) near 2 was wrong.

## 5. Hypotheses

| hypothesis | status |
|---|---|
| H1 signal feedback | Held on the zero-signal channel: H1 loadings gate .95-.99 throughout, oracle 0. The "stuck at 0" branch was not seen on a signal channel; the newest shells (30-31) are under-opened at the end, which may be the same fixed point not yet left. |
| H2 per-batch division | Not tested (Phase 1 arm B not run). The estimators of arms C/D (regularised half solves, no single-batch inversion) pass unit tests on Gaussian data: unbiased P and kappa, batch-size invariant, model cancels (`tests/unit/ppca_initial_model/test_gate_estimators.py`). |
| H3 shell-level noise | Not tested directly. With uniform views, F_all per voxel is close to a shell constant, so a per-shell oracle was used. |
| H4 1/sqrt(u) | Weak here: sqrt(u) about 1, effective gate within .04 of the gate. |
| H5 PPCA coupling | Not tested (arm G not run). |
| "The oracle beats the baseline" | Only formally (O, +.0013 on two sets, and worse on two others); not for the validated oracle Ocm. |

## 6. Not run, not verified

- **Phase 1** (arms B-G) was not run. Lead and owner stopped the study after Phase 0. A minimal mechanism Phase 1
  (~14 GPU-h) remains available if the owner asks:
  - arms E (B+C+D, u kept) and C against A, on H1 n3, C n3 and D20 n1, seeds 11-13;
  - a log-only batch-size check, b = 1000 and 3000, on C n3 and H1.
  It would show whether E shuts H1's loading gate and tracks Ocm. Its end-metric gain is capped near Ocm's,
  about .001-.002.
- Not done: batch-size invariance, the fudge sweep, the low-pass start test, and real data.
- Not verified:
  - the per-voxel (H3) oracle;
  - the per-component loading oracle in the run's latent gauge (the shared-channel scalar was used);
  - whether Ocm's assumption that the loading channel's noise equals the mean channel's holds for q > 1 with
    correlated latents (the check used q = 1).

## 7. Process record

- **Missing splits.** The first GPU scoring stage (17479728-30) failed at RECON/TRUTH because the new work dir had no
  `splits/`. The inference was kept; the splits were then made, byte-identical to the lead's.
- **Sign bug.** The suite's first pass scored raw checkpoints of STAR-trained runs. Their mean has the opposite sign to
  the suite's simulator-sign data: pose 130-137 deg, decoded below mean-only. All 72 first-pass rows are invalid, kept in
  `msuite/invalid_signbug_20261114/` (38 GPU-h). The re-score used mean-negated copies (`ckpt_suite/`, provenance file).
  A guard now refuses any entry list that is not made of recorded copies (`scripts/gate_study/check_entries.py`, run by
  `msuite/submit_gpu.sh`). Two check rows passed before the re-score.
- **Registration failure.** gs_D40_n1_O_s11 failed suite registration on the sign-inverted model. Suite v3.2 defines no
  handling for this; one was predeclared (not scored, the dataset becomes undecidable) but was not needed: no corrected
  row failed.
- **Staging delay.** The re-score's protocol rows reached Polar 7 h after their GPU jobs ended, because I missed the
  watcher notice.

GPU-hours: 95.8 of 150.
- training 20.9 (P0 14.1, Ocm 6.8);
- suite GPU stage, first pass 37.7, invalid (8.2 of it inference that was reused before the sign bug was found);
- re-score 37.2 (12.5 protocol, 24.2 Cshift wide grid, 0.5 check rows).

Polar CPU scoring is not counted.
