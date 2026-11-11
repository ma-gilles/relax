# PPCA per-particle contrast: progress log

Branch `ppca_contrast` (worktree `ppca-contrast` under the scratch worktree root), branched from
`ppca_schedule` at 2fd7b0e8. The derivation is in [ppca_contrast.md](../math/ppca_contrast.md). This file is the
running report: the plan and its status first, then dated entries, newest last.

## Goal

Let the pose-marginal PPCA (`relax ppca_initial_model`, and the dense refinement that shares its engine) model a
per-particle contrast `c_i` that multiplies the signal, so that images whose signal is scaled by ice
thickness or exposure no longer distort the loadings and the poses. Done when contrast is implemented,
tested, and performs to a satisfactory level on simulated data with and without contrast.

## Plan and status

| Step | What | Status |
| --- | --- | --- |
| 1 | Math: sufficient statistics, exact prior, quantization | done, [ppca_contrast.md](../math/ppca_contrast.md); model A matches the exact model B (section 8) |
| 2 | Implementation shared by initial model and refinement | model A in the initial model (`--contrast latent`; the loadings are kept orthogonal to the mean by default) and, as helpers (tied column, fold, centring, projection), in `contrast.py` for any engine; the dense refinement loop has no production controller for it yet; CPU tests pass |
| 3a | Contrast-spread simulated datasets with a paired control | done: C n1 box 64 at sd 0, 0.15, 0.30 |
| 3b | Cost of ignoring contrast (baseline runs) | done for seed 11: at sd 0.30 kmeans ARI falls from 0.99 to 0.57-0.63, poses and FSC hold |
| 3c | Contrast model vs baseline; harm check at unit contrast | done (rounds 3 to 7): at sd 0.30 and noise 1 kmeans ARI 0.57-0.66 to 0.93-0.98 (3 seeds for VDAM, 14 for constrained SGD), noise 3 and log-normal one to two seeds; no change at sd 0; cryo-ET fixture: SGD calibrated, VDAM not settled |
| 5 | Model B (contrast on the whole signal, integrated per pose) in the stream | implemented 2026-11-11 (`--contrast quadrature`, math note section 11): unit tests (CPU and GPU) and the node study pass; not yet run on the suites (arms `qo_*`) |
| 6 | VDAM with the latent on the real data set cahra_01 at three tau2 fudges, with paired runs without it | submitted 2026-11-11, queued (per-user job limit); report script ready |
| 4 | Heuristics (when to switch contrast on, prior width) and revisit | not needed so far: the two failures first found were fixed structurally (math note sections 9 and 10); a late start and a fixed sd did not help the first one (noise level 1, plain M-step); the fixed sd on the noise 3 case that still fails is round 9 |

## Decisions so far

- Two models, one derivation. **Model A**: contrast scales the mean only (`y = A[(1 + delta) mu + W z]`,
  `delta ~ N(0, tau^2)`): exact Gaussian posterior, an extra latent tied to the mean, no new kernel. **Model B**: the
  physical model (contrast scales the whole signal): no conjugate prior for `q >= 1`; `c` is integrated by
  quadrature. Model A goes in first and is the candidate for production; model B is the reference that says how
  accurate A is and what A gives up.
- The M-step statistics are `E[c alpha]` and `E[c^2 alpha alpha^T]` per pose: the engine's pass 2 already takes
  weights and moment sums of exactly these shapes, so only the per-pose latent stage is model-specific.
- Work on the unit-contrast default is untouched: contrast is an opt-in option, and with it off the code path is
  the existing one.

## Log

### 2026-11-09 start

- Worktree and branch created; `doctor` passes (natives, RELION binding and idle local GPUs warn, as on the
  parent branch).
- Read the engine: pass 1 (`score_block`, `_latent_block`), pass 2 (`moment_block`, `_posterior_block`),
  the controller (`iteration_loop._expectation`, `run`) and RECOVAR's `contrast_posterior` and its note.
  Findings that shaped the plan are in the math document, section 4 and 6.
- Earlier work on this question in this repository: section 16.10 of `vdam_ppca_algorithm.md` (cryo-ET, a grid
  and a point estimate, removed 2026-10-03). The point estimate tied from the ground-truth start and lost from a
  random start (pose median 24 deg against 5.7 deg) because misaligned particles received a low contrast. That
  is the failure to watch.
- Simulator: `simulate_data(..., per_image_contrast, ...)` scales the signal and adds the noise after, as the
  model assumes. The suite builder passes ones; the contrast-spread datasets will be built from a suite
  dataset's own truth (volumes, rotations, CTF, noise spectrum) with the contrast drawn from a truncated
  normal.

### 2026-11-09 model, code and first data

- **Model A against the exact model B** (math document, section 8, `scripts/ppca_contrast/model_a_vs_exact.py`, four
  unit tests). At a known pose, model A's contrast posterior equals the exact one (correlation 0.998-0.999, same
  width) at every loading ratio tried, its latent divided by the contrast estimate recovers B's accuracy to 0.01, and
  the pose evidence agrees to 0.5 percent. Unit contrast loses 0.05-0.07 of latent R^2 at contrast sd 0.3. Modelling
  a contrast that is absent costs nothing. Decision: model A is the model; B is not built into the stream.
- **Implementation.** `relax/ppca_refinement/contrast.py` (tied column, gradient and metric fold, contrast split,
  EM update of the sd; usable by any engine whose statistics have the augmented layout). In the stream, pass 2 now
  sums `gamma E[alpha_p^2]` per latent. The initial model has `--contrast latent`, `--contrast-sd`,
  `--no-contrast-estimate-sd`, `--contrast-start` (State and checkpoint carry the sd; old checkpoints resume).
  Tests: the engine on the tied column against an explicit Gaussian, the folded gradient against finite differences
  of the model A likelihood, `T^T L T`, EM recovering the true sd, the single-pose second moments through the real
  stream, the controller's sd update and checkpoint round trip (21 new tests, plus the four of the model comparison). The existing stream, checkpoint,
  tile-plan and oversampling tests pass (97 passed, 16 GPU-only skipped).
- **Datasets.** `simulate_contrast_dataset.py` re-simulates a suite dataset from its truth (volumes, rotations, CTF,
  noise spectrum, simulator seed) with the signal of particle `i` scaled by `c_i ~ N(1, sd^2)` (redrawn below
  0.2). C n1, box 64 (the `C__realistic__noise1__b128crop64` truth), 100,000 particles, sd 0, 0.15 and 0.30. Paired
  by construction and checked: `img(sd 0.30) - img(sd 0) = 2 (img(sd 0.15) - img(sd 0))` to 7.5e-7 relative on
  the particles that were not redrawn, so only the signal scale differs.
- **Evaluation.** The suite scorer infers with its own unit-contrast code, so a contrast-trained model is scored
  through `scripts/ppca_contrast/infer_for_suite.py`: the run's own E-step on the scorer's images and protocol grid
  writes the scorer's `inference.npz` (z of the scorer's sign, divided by the contrast estimate by default, best
  poses) and the scorer reuses it. The baseline arms use the scorer's own inference (this is the "ignoring contrast"
  case). The Snakefile passes the options for any arm whose `--contrast` is not `none`.
- Baseline runs launched (round 1): `s3_vdam_uniform` and `s3_scale1_4_uniform`, seed 11, on the three datasets.

### 2026-11-09 (evening) baselines, round 1 of the contrast latent, and a gauge failure

**Baselines** (seed 11, C n1 box 64, 100,000 particles, current defaults: uniform direction prior; the scorer's own
inference, which ignores contrast). Suite scorer at the last checkpoint; in-run latent scores from the final embedding
pass.

| contrast sd | arm | kmeans ARI | state FSC | mean FSC | pose median (deg) | pose < 10 deg | gmm ARI | nn acc | R^2 states |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | VDAM | 0.992 | 0.949 | 0.999 | 3.94 | 0.999 | 0.994 | 0.994 | 0.761 |
| 0 | SGD 1/4 | 1.000 | 0.941 | 0.730 | 5.01 | 0.990 | 0.993 | 0.995 | 0.761 |
| 0.15 | VDAM | 0.976 | 0.953 | 0.999 | 3.97 | 0.998 | 0.989 | 0.991 | 0.741 |
| 0.15 | SGD 1/4 | 0.964 | 0.921 | 0.984 | 5.46 | 0.965 | 0.982 | 0.986 | 0.727 |
| 0.30 | VDAM | 0.573 | 0.955 | 0.998 | 4.04 | 0.981 | 0.863 | 0.951 | 0.591 |
| 0.30 | SGD 1/4 | 0.631 | 0.920 | 0.986 | 5.39 | 0.933 | 0.908 | 0.944 | 0.614 |

Ignoring a contrast spread of 0.30 leaves the poses (4.0-5.4 deg), the state FSC and the mean FSC where they were
and takes the clustering of the latents down: kmeans ARI 0.99 to 0.57-0.63, GMM ARI 0.99 to 0.86-0.91, between-state
R^2 0.76 to 0.59-0.61. At 0.15 the loss is a few percent of ARI. The contrast acts on the latents (and the loadings),
not on the alignment, in this data.

**Round 1 of the contrast latent** (uncentred sd update, started from iteration 1, initial sd 0.2): both optimizers
drifted along the scale gauge of the math note (section 9). VDAM: the sd hit its cap (0.5) on all three datasets
including sd 0, the mean map shrank to 0.28 of the baseline's power (c00, iteration 1000), the contrast estimate's mean went
to 1.8. Even so, on sd 0.30 the latents beat the baseline's on between-state R^2 (0.710 against 0.591) and nearest-centroid
accuracy (0.972 against 0.951), the contrast estimate correlated 0.959 with the truth (0.868 at sd 0.15), and at sd 0 nothing
was lost (GMM ARI 0.993 against 0.994). Momentum SGD: the sd estimate was right for 500 iterations (0.29-0.30 at true 0.30,
0.07 at true 0), then the mean-map power ran away and the curvature check stopped all three runs (iterations 2050-2250).
The suite scorer was not run on the VDAM arms (not worth the GPU time on a gauge-drifted model).

Decision: the sd update uses the centred second moment (math note, section 9), committed with a test. Round 2 runs it
against a fixed sd (0.3) and a late start (iteration 800), for both optimizers, on the sd 0 and sd 0.30 datasets.

### 2026-11-09 (night) rounds 2 and 3: the scale gauge and its fix

Same data and settings as above (C n1 box 64, 100,000 particles, seed 11).

**Round 2** (sd from the centred second moment; a fixed sd of 0.3; a start at iteration 800; all with the plain M-step):
- centred sd: at true sd 0 the sd falls to 0.03 and the gain stays at 1.0 (VDAM and SGD); at true sd 0.30 VDAM still ended
  with sd 0.5 and gain 1.79, SGD with sd 0.14, gain 0.50 and a mean-volume power of 1.27 against 0.30 for the baseline.
- late start (iteration 800): at the first contrast iteration the mean contrast estimate was already 1.27 (the
  regularised mean volume is below the images' amplitude), and the run ended exactly as the early start did (VDAM gain 1.79).
- fixed sd 0.3: VDAM gain 1.38 at true sd 0.30 and 1.39 at true sd 0 (the gain drifts without any spread); SGD stopped on the
  negative-curvature check at iterations 2100-2200 (both datasets).
Conclusion (math note, section 9): with the population mean of the contrast pinned to 1, the latent takes the difference
between the regularised mean volume and the images, and the M-step lowers the mean volume to compensate. No sd rule or
start time removes that.

Why the fixed sd did not pin it (re-read 2026-11-10 from the round 2 traces; arguments, not new experiments). The ambiguity is between the
mean volume's amplitude and the population mean `g` of the latent, and a fixed sd pins the spread of the contrast, not its mean: the
`N(0, 1)` prior pulls on `g` with the share of the precision it holds, which is the posterior variance of `e` (0.07 to 0.09 at noise
level 1, 0.19 to 0.29 at level 3, from the last 200 iterations of the round 3 to 5 runs), against a mean volume that is regularised or
still growing. In `cf_vdam` the latent's mean went from +0.07 to +1.1 within 200 iterations on both datasets, and stayed at +0.9 to +1.4
(mean contrast 1.26 to 1.42); at true sd 0 a fixed sd of 0.3 gives the latent the room (`c = 1 + 0.3 g`), whereas the estimated sd
falls to 0.03 there and removes it. A fixed sd was run only at noise level 1 (`cf_*` with the plain M-step, `cgf_*` with the centred
one, where it gave the maps of the estimated sd), never on the noise 3 case that fails; round 9 does (`cgf_*` and `cof_*`, seeds 11 and 12).

**Round 3** (the fix: the M-step takes the moments of `e - m`, `m` the population mean of the latent; the E-step is
unchanged; `--contrast-center`, on by default; implemented on the pass-2 moments of the stream and on each pose's moments of
the dense engine; tests: the shift algebra, the engine's gradient on shifted moments against finite differences of the
gain-free objective, the stream's metric against `S L S^T`, dense and streamed engines agreeing). All ten runs completed,
mean-volume power 0.30-0.32 (the baseline's), mean contrast estimate 0.90-1.04, and the estimated sd converged on the truth
(VDAM: 0.306 at true 0.30, 0.158 at 0.15, 0.028 at 0; SGD: 0.280, 0.144, 0.038). Fixing the sd at 0.3 instead of estimating it gave
the same maps at true sd 0.30 and a small loss on SGD at true sd 0 (pose 5.7 deg against 5.0, state FSC 0.925 against 0.941).

Results (suite scorer and in-run latent scores; contrast arms scored through their own E-step, baselines through the
scorer's; seed 11):

| true sd | arm | kmeans ARI | state FSC | pose med (deg) | pose < 10 | subspace (deg) | energy | gmm ARI | nn acc | R^2 states | corr(c) | sd prior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | VDAM, no contrast model | 0.992 | 0.949 | 3.94 | 0.999 | 35.3 | 0.934 | 0.994 | 0.994 | 0.761 | | |
| 0 | VDAM, contrast latent | 0.996 | 0.948 | 3.92 | 1.000 | 35.8 | 0.934 | 0.994 | 0.994 | 0.761 | | 0.028 |
| 0 | SGD 1/4, no contrast model | 1.000 | 0.941 | 5.01 | 0.990 | 88.6 | 0.296 | 0.993 | 0.995 | 0.761 | | |
| 0 | SGD 1/4, contrast latent | 0.988 | 0.937 | 5.09 | 0.986 | 89.1 | 0.303 | 0.993 | 0.995 | 0.764 | | 0.038 |
| 0.15 | VDAM, no contrast model | 0.976 | 0.953 | 3.97 | 0.998 | 38.8 | 0.930 | 0.989 | 0.991 | 0.741 | | |
| 0.15 | VDAM, contrast latent | 0.980 | 0.950 | 3.94 | 0.996 | 38.5 | 0.932 | 0.990 | 0.992 | 0.750 | 0.870 | 0.158 |
| 0.15 | SGD 1/4, no contrast model | 0.964 | 0.921 | 5.46 | 0.965 | 70.3 | 0.803 | 0.982 | 0.986 | 0.727 | | |
| 0.15 | SGD 1/4, contrast latent | 0.976 | 0.927 | 5.68 | 0.960 | 89.2 | 0.290 | 0.991 | 0.994 | 0.758 | 0.820 | 0.144 |
| 0.30 | VDAM, no contrast model | 0.573 | 0.955 | 4.04 | 0.981 | 90.0 | 0.871 | 0.863 | 0.951 | 0.591 | | |
| 0.30 | VDAM, contrast latent | 0.937 | 0.959 | 3.97 | 0.978 | 34.8 | 0.932 | 0.957 | 0.973 | 0.715 | 0.960 | 0.306 |
| 0.30 | SGD 1/4, no contrast model | 0.631 | 0.920 | 5.39 | 0.933 | 89.3 | 0.769 | 0.908 | 0.944 | 0.614 | | |
| 0.30 | SGD 1/4, contrast latent | 0.932 | 0.921 | 5.64 | 0.935 | 89.1 | 0.281 | 0.824 | 0.971 | 0.707 | 0.925 | 0.279 |

Reading it. The map-frame columns (subspace angle, energy, and the scorer's mean FSC) depend on the global rotation a run ends
in; the SGD runs end rotated away from the truth frame in most cells (direct fit: no mirror; map-frame state FSC lower than the direct
one), so only the VDAM rows compare on them. The frame-free columns are kmeans ARI, state FSC, pose, the latent scores and the contrast
columns.
- At true sd 0.30 the contrast latent returns the clustering of the latents to near the unit-contrast level: kmeans ARI 0.573 to 0.937
  (VDAM) and 0.631 to 0.932 (SGD), unit-contrast reference 0.992 and 1.000; VDAM's loading subspace angle 90 deg to 34.8 deg
  (unit-contrast reference 35.3), energy 0.871 to 0.932 (0.934), between-state R^2 0.591 to 0.715 (0.761). State FSC and pose are
  unchanged within the seed noise (state FSC 0.955 to 0.959; pose 4.04 to 3.97 deg): as the baselines showed, contrast leaves the
  alignment alone. GMM ARI for SGD is lower with the latent (0.824 against 0.908) while kmeans ARI, nearest-centroid accuracy and R^2 are
  higher: the GMM fit of one run, a single seed.
- At true sd 0.15: small gains (SGD gmm ARI 0.982 to 0.991, R^2 0.727 to 0.758; VDAM within 0.01).
- At true sd 0 (the harm check): nothing moves beyond the run-to-run level (VDAM pose 3.94 to 3.92 deg, state FSC 0.949 to 0.948,
  kmeans ARI 0.992 to 0.996; SGD pose 5.01 to 5.09 deg, state FSC 0.941 to 0.937, kmeans ARI 1.000 to 0.988).
- The contrast estimate correlates 0.96 (VDAM) and 0.93 (SGD) with the true contrast at sd 0.30, 0.87 and 0.82 at sd 0.15 (a per-image
  posterior sd of about 0.06 against a spread of 0.15 caps it near 0.91).

**Cost.** Controlled run, one A100, radius 31, HEALPix order 3, q = 10, 300 images per update, momentum SGD, the median of updates 11 to 40:
1.069 s without the latent, 1.315 s with it (+23 percent): one more inner-product channel and tri(P+1) against tri(P) packed moment
channels (78 against 66).

**Default path.** `scripts/dev/ppca_fingerprint.py` against the parent branch (CPU, four cases: VDAM, momentum SGD, host-mask dense,
oversampling): every checkpoint array, result and file is identical; the only differences are the new config keys and their dictionary
sizes (176 leaf differences, all in `metadata/config`, `run.json` config or the key counts), checked before the centring commit; to be
re-run at the end.

Next: round 4 (seeds 12 and 13 at true sd 0.30 and 0; noise level 3; log-normal contrast), below.

### 2026-11-10 round 4 (seeds, noise level 3, log-normal), cryo-ET, and a loading parallel to the mean

Round 4 ran snapshot v6 (the round-3 code) on the datasets C n1 box 64 with contrast sd 0.30 and 0 (seeds 12 and 13), the
same two at noise level 3 (three times the injected noise variance; seed 11) and log-normal contrast of sd 0.30 (seed 11).
Baselines are `s3_vdam_uniform` and `s3_scale1_4_uniform` (SGD) with the contrast ignored; `cg_*` are the contrast-latent arms
of round 3. Per cell: kmeans ARI / state FSC / pose median (deg) are in `$R/table_r4r5.md`; the lines below are the reading.

| dataset | seed | arm | kmeans ARI, baseline | kmeans ARI, contrast latent | corr(c) | estimated sd (true) |
| --- | --- | --- | --- | --- | --- | --- |
| sd 0.30, noise 1 | 12 | VDAM | 0.568 | 0.934 | 0.959 | 0.302 (0.30) |
| sd 0.30, noise 1 | 13 | VDAM | 0.580 | 0.952 | 0.959 | 0.307 |
| sd 0.30, noise 1 | 12 | SGD | 0.656 | 0.941 | 0.930 | 0.276 |
| sd 0.30, noise 1 | 13 | SGD | 0.645 | 0.952 | 0.919 | 0.284 |
| sd 0, noise 1 | 12, 13 | VDAM | 0.992, 0.980 | 0.980, 0.984 | | 0.029 |
| sd 0, noise 1 | 12, 13 | SGD | 0.984, 0.996 | 0.996, 1.000 | | 0.037 |
| log-normal 0.30, noise 1 | 11 | VDAM | 0.619 | 0.968 | 0.961 | 0.310 |
| log-normal 0.30, noise 1 | 11 | SGD | 0.948 | 0.948 | 0.928 | 0.285 |
| sd 0.30, noise 3 | 11 | VDAM | 0.385 | 0.647 | 0.901 | 0.309 |
| sd 0.30, noise 3 | 11 | SGD | 0.346 | **0.297** | **-0.103** | **0.089** |
| sd 0, noise 3 | 11 | VDAM | 0.882 | 0.878 | | 0.034 |
| sd 0, noise 3 | 11 | SGD | 0.671 | 0.738 | | 0.030 |

- The round-3 result replicates on two more seeds (kmeans ARI 0.57-0.66 to 0.93-0.95 at sd 0.30, no change at sd 0, within
  the 0.02 run-to-run level); state FSC and pose are unchanged; the sd estimates are 0.28-0.31 against the true 0.30.
- Log-normal contrast (a skewed distribution; the model is Gaussian): VDAM 0.619 to 0.968; the SGD baseline was already at 0.948
  on this dataset (single seed), so SGD gains nothing there.
- Noise level 3 (SNR probe 0.041 against 0.124) with sd 0.30: VDAM 0.385 to 0.647 and a contrast correlation of 0.90, but the
  unit-contrast reference at this noise level is 0.878 and 0.882 (VDAM) so about half of the loss remains; sd 0 at noise 3 shows
  no harm.
- **SGD at noise 3 and sd 0.30 fails**: the sd estimate decays from 0.28 (iteration 1600, order 2 to 3 stage change at 1700) to
  0.09 (the order 2 to 3 stage change is at iteration 1626), the gain (mean of the latent) drifts to -0.74, the estimate has a correlation of -0.10 with the true contrast and
  the kmeans ARI is below the baseline (0.297 against 0.346). The centred second moment of the latent stays at 0.97 to 1.02 during the
  decay, a fixed point of the sd update, which loses 0.05 percent per step at 0.99 and compounds.

**Diagnosis** (math note section 10). A loading parallel to the mean volume is a second contrast latent, and the likelihood is flat
along the split of variance between it and the contrast latent. The cosine squared between each loading and the mean in the final
checkpoints: contrast latent runs that work have 0.000-0.003 (all VDAM runs, SGD at noise 1); the baselines at sd 0.30 have
0.2-0.5 on several loadings (the contrast variation is that direction); the failing SGD run keeps 0.18, 0.08 and 0.16 on three
loadings and 0.0-0.003 on the others. This is the strongest evidence I have and not a proof: I have not shown that removing the parallel
component prevents the decay (the round 5 below does).

**Timeline check** (2026-11-10, per-checkpoint, `scripts/ppca_contrast/contrast_variance_split.py` on the failing run, seed 12 and the
constrained arm). The diagnosis above put the loadings first; the checkpoints do not. (1) Until iteration 1600 the loadings left the mean
direction (summed squared coefficient 0.126 to 0.029) while `sd` rose to 0.28, as the latent should make them. (2) The sd fall starts at the
stage change to order 3 (iteration 1626) in all four noise-3 SGD runs, including both constrained ones (0.294 to 0.268 by iteration 1800), with a
fall of the gain (-0.2 to -0.3). It is not the loadings. It is not a motion of the scale gauge either (I first wrote that, and the shells
refute it): at that stage the mean's total power rises 6 to 9 percent in SGD, in the baseline without a latent too, and by 0.5 percent
in VDAM, but the rise is in shells 11 to 24 (x1.1 to 2.3) while shells 1 to 10 do not change (projection 1.00 baseline, 0.99 with the
latent). At the other 16 stage starts, where only the radius opens, the power moves 0 to 3 percent. (3) Afterwards the unconstrained
run moves variance from the latent to the parallel loadings (`sd^2` 0.055 to 0.008, parallel 0.029 to 0.091 with the flat inner product, sum
0.084 to 0.099 in seed 11; with the noise-weighted product 0.022 to 0.050 and the sum falls 0.077 to 0.058; in seed 12 the sum falls 0.075 to
0.058), and the constrained runs, which cannot, stop at 0.23. So the constraint is supported for the continued decay by an intervention.
(4) The contrast estimates are biased in both optimizers' final models: against the true contrast (mean 1.004, sd 0.296) the orthogonal SGD
run has mean 0.874 and slope 0.56, VDAM 1.055 and 0.85, so the mean contrast of 0.87 is not a free gauge: it is an E-step amplitude bias that
the centred M-step (which fits the mean to the images' own amplitude whatever the population gain) does not correct.

**Radius probe** (2026-11-10, job 17255539, `$R/radius_probe`, `scripts/ppca_contrast/radius_probe_table.py`): the run's own E-step
(`infer_for_suite.py`, order 3, the scorer's protocol grid, 2538 evaluation images) re-run on a checkpoint at Fourier radii 8 to 31,
estimate against the true contrast (mean 1.004, sd 0.296), seed 11, noise level 3, sd 0.30:

| checkpoint | radius | mean | sd | estimate = a + b truth | corr |
| --- | --- | --- | --- | --- | --- |
| co_sgd 3000 | 8 | 0.998 | 0.142 | 0.669 + 0.330 | 0.691 |
| | 12 | 1.001 | 0.192 | 0.471 + 0.530 | 0.820 |
| | 16 | 0.989 | 0.216 | 0.364 + 0.626 | 0.860 |
| | 20 | 0.971 | 0.216 | 0.346 + 0.626 | 0.861 |
| | 24 | 0.940 | 0.209 | 0.337 + 0.603 | 0.858 |
| | 31 | 0.871 | 0.198 | 0.307 + 0.565 | 0.849 |
| co_sgd 1600 | 8 / 16 / 24 | 1.009 / 1.011 / 0.986 | 0.192 / 0.263 / 0.260 | slope 0.449 / 0.762 / 0.757 | 0.693 / 0.860 / 0.866 |
| co_vdam 3000 | 8 | 1.032 | 0.216 | 0.526 + 0.506 | 0.697 |
| | 16 | 1.034 | 0.270 | 0.234 + 0.801 | 0.880 |
| | 31 | 1.052 | 0.280 | 0.197 + 0.856 | 0.907 |

The probe reproduces the training E-step at radius 31 (0.871 and slope 0.565 against 0.874 and 0.557 in the run's own embeddings). The
momentum SGD model's mean contrast estimate is unbiased at radius 8 to 12 (1.00) and falls to 0.87 as the shells 20 to 31 enter, at the end
of training but hardly at iteration 1600 (1.01 to 0.986); its slope and correlation stop improving past radius 16 to 20. VDAM's estimate
does not fall with the radius (1.03 to 1.05) and its correlation keeps rising to 0.907. So the E-step's amplitude fit is attenuated by the
high shells of the SGD mean. I read that as template power that does not match the images (the shells 11 to 24 that grow at the order 3
stage; not shown directly, a misaligned but real signal would act the same way), which VDAM's regularised mean does not have. Whether the
sd decay follows from it is what round 8 (order change moved, arms `t_*` of `arms.yaml`) tests.

**Cryo-ET fixture** (`cryoet_ppca_k3conf_contrast30_box64`: 399 subtomograms, 3 states, contrast sd 0.30 shared by the tilts of
a particle; q = 2, 200 iterations, seeds 11 and 12, scored with `scripts/ppca_contrast/eval_tomo_fixture.py`):

| run | nn accuracy | kmeans ARI | R^2 states | corr(c) | estimated mean c | estimated sd (cap 0.5) |
| --- | --- | --- | --- | --- | --- | --- |
| VDAM, no contrast model, s11 / s12 | 0.855 / 0.907 | 0.275 / 0.344 | 0.482 / 0.556 | | | |
| VDAM, contrast latent, s11 / s12 | 0.880 / 0.865 | 0.354 / 0.333 | 0.360 / 0.326 | 0.72 / 0.82 | 1.50 / 1.46 | 0.500 / 0.442 |
| SGD, no contrast model, s11 / s12 | 0.822 / 0.872 | 0.229 / 0.317 | 0.426 / 0.434 | | | |
| SGD, contrast latent, s11 / s12 | 0.920 / 0.905 | 0.470 / **0.014** | 0.364 / 0.162 | 0.82 / 0.75 | 1.08 / 1.06 | 0.360 / 0.419 |
| VDAM, unit-contrast fixture, no model, s11 / s12 | 0.915 / 0.905 | 0.431 / 0.390 | 0.529 / 0.543 | | | |
| SGD, unit-contrast fixture, no model, s11 / s12 | 0.875 / 0.782 | 0.587 / 0.243 | 0.555 / 0.330 | | | |
| VDAM, contrast latent, orthogonal loadings, s11 / s12 | 0.825 / 0.867 | 0.462 / 0.390 | 0.237 / 0.402 | 0.80 / 0.82 | 1.62 / 1.60 | 0.500 / 0.500 |
| SGD, contrast latent, orthogonal loadings, s11 / s12 | 0.915 / 0.872 | 0.388 / 0.594 | 0.372 / 0.488 | 0.89 / 0.93 | 1.02 / 1.03 | 0.308 / 0.309 |

- The fixture is hard even without contrast: the unit-contrast fixture's kmeans ARI is 0.24-0.59 and varies with the seed by more
  than the effect size, so latent scores on 399 particles cannot separate the arms except by large differences.
- The plain contrast latent does not clearly help: the contrast estimate carries information (correlation 0.72-0.82), but the sd runs
  to its cap with VDAM and swings with SGD, and one SGD run collapsed (ARI 0.014). These 200-iteration runs are not stationary: the
  mean volume's power changes by a factor of 4 between iterations 110 and 200 (also without the latent), the contrast gain follows
  it with swings of about 1 at each radius opening, and the sd rises with them.
- With orthogonal loadings SGD is stable and calibrated: the estimated sd is 0.308 and 0.309 (true 0.30), the mean estimate 1.02,
  the correlation with the true contrast 0.89 and 0.93 (0.82 and 0.75 without the constraint), and kmeans ARI 0.39 and 0.59, at the
  unit-contrast fixture's level (0.59, 0.24; the no-model runs on the contrast fixture reach 0.23, 0.32). VDAM with orthogonal loadings
  still has the sd at its cap and a mean estimate of 1.6 (the gain drift), with ARI at the unit-contrast level (0.46, 0.39) and a lower R^2.
  Not resolved for VDAM on this short schedule.

### 2026-11-10 round 5: orthogonal loadings (snapshot v7, `--contrast-orthogonal`)

Arms `co_*` are `cg_*` plus the constraint. kmeans ARI (baseline = contrast ignored; `cg` = round 3 or 4, no constraint; `co` = constrained):

| dataset | seed | VDAM baseline / cg / co | SGD baseline / cg / co |
| --- | --- | --- | --- |
| sd 0.30, noise 1 | 11 | 0.573 / 0.937 / 0.949 | 0.631 / 0.932 / 0.936 |
| sd 0.30, noise 1 | 12 | 0.568 / 0.934 / 0.937 | 0.656 / 0.941 / **0.391** |
| sd 0.30, noise 3 | 11 | 0.385 / 0.647 / 0.567 | 0.346 / 0.297 / 0.494 |
| sd 0.30, noise 3 | 12 | 0.351 / 0.616 / 0.614 | 0.322 / 0.347 / 0.564 |
| sd 0, noise 3 | 11 | 0.882 / 0.878 / 0.886 | 0.671 / 0.738 / 0.695 |
| log-normal 0.30, noise 1 | 11 | 0.619 / 0.968 / 0.980 | 0.948 / 0.948 / 0.960 |

- SGD at noise 3 and sd 0.30: the sd no longer collapses (estimated sd of the contrast 0.198 and 0.195, prior 0.23, against 0.031 and 0.056 with prior
  0.09 and 0.12 without the constraint) and the kmeans ARI rises over the baseline by 0.15 and 0.24 (0.346 to 0.494, 0.322 to 0.564). The sd
  still ends below the true 0.30. Two seeds.
- VDAM: no change beyond the run-to-run level (noise 1: +0.012 and +0.003; noise 3: -0.080 and -0.002, with GMM ARI, nearest-centroid accuracy
  and R^2 within 0.03 of the unconstrained run; kmeans ARI at noise 3 moves by 0.05 to 0.08 between runs of one arm). Unit contrast
  (sd 0 at noise 3) and log-normal contrast: no harm.
- **One SGD run is a failure that is not contrast-related in kind**: `co_sgd` at noise 1, seed 12 ended with state FSC 0.715, pose median 9.7 deg
  (51 percent within 10 deg), mean FSC 0.82 and kmeans ARI 0.391. The contrast statistics of the run are normal (sd 0.26, gain -0.37, mean
  power 0.29, the log-likelihood of the others) and the damage is already there at iteration 1000 (pose median 17.6 deg against 7.3 for
  `cg_sgd` of that seed): the run settled in a worse ab-initio map. Whether the constraint made that more likely is open: one failure in three
  constrained SGD runs at noise 1 (state FSC 0.935, 0.715, 0.938) against none in the 18 SGD runs at noise 1 without it (state FSC 0.90 to 0.95, pose 4.95 to 5.96 deg). Round 6 runs seeds 13 to 16 of `co_sgd` and `cg_sgd` at this dataset.

**Round 6** (seeds 13 to 16 of `cg_sgd` and `co_sgd` at sd 0.30, noise 1): all eight runs are healthy. kmeans ARI `co_sgd` 0.941, 0.956, 0.948, 0.949
and `cg_sgd` 0.952, 0.848, 0.948, 0.941; state FSC 0.90 to 0.93 and pose median 5.6 to 5.96 deg in every run; estimated contrast sd
0.264 to 0.272. `cg_sgd` seed 13 reproduces its round-4 run (kmeans ARI 0.952, state FSC 0.904, pose 5.96 deg both times). So the failure of
seed 12 is one in seven constrained SGD runs at noise 1 (against none in 22 unconstrained): too few to say whether the constraint makes
a bad ab-initio basin more likely. Round 7 runs `co_sgd` on seeds 17 to 24.

**Default-path fingerprint** (`scripts/dev/ppca_fingerprint.py`, one host for both halves, four CPU cases: VDAM, momentum SGD, host-mask dense,
oversampling; parent `ppca_schedule` 2fd7b0e8 against this branch at fe7ee1f2). Every checkpoint array, result and file is identical.
The only differences: the new config keys and the key counts of `config` and `metadata` (39 to 48, 11 to 13: the contrast settings, the
contrast gain in the metadata), the source identity, and three VDAM log lines of the tile plan whose counted stream memory moves by 0.01 GiB
(0.64 to 0.63 GiB; the tile plan itself, 20 of 20 images, is the same). Not covered, as the tool says: the GPU kernels and subtomogram input.

**Round 7** (`co_sgd`, seeds 17 to 24, sd 0.30, noise 1): all eight runs are healthy: kmeans ARI 0.925 to 0.952, state FSC 0.89 to 0.95,
pose median 5.5 to 6.0 deg, estimated contrast sd 0.261 to 0.266. Over rounds 5 to 7 the constrained SGD arm at noise 1 has one failure in 15
runs (seed 12) against none in 22 for the unconstrained SGD arm and the baseline: Fisher's exact test gives p about 0.4, no evidence of a
cost; the failure is an ab-initio basin of that seed (bad at iteration 1000, normal contrast statistics).

**Decision**: under `--contrast latent` the loadings are kept orthogonal to the mean by default (`--no-contrast-orthogonal` turns it
off; decided once at the command boundary, `contrast_orthogonal_setting`; the `Config` default stays off, so the identity of runs without the
latent and of old checkpoints does not change). Basis: SGD's sd collapse at noise 3 is gone (kmeans ARI +0.15 and +0.24 over the baseline on two
seeds), SGD on the cryo-ET fixture is calibrated, VDAM is unchanged within run-to-run noise on the six simulated comparisons (kmeans ARI differences -0.08 to +0.01; on the cryo-ET fixture the VDAM sd stays at its cap with and without it), and the constraint
is the identifiability condition of the model (math note section 10). The `cg_*` arms pass `--no-contrast-orthogonal` and keep their meaning.

**Unit tests** (CPU, `tests/unit/{ppca_contrast,ppca_initial_model,ppca_refinement}`, `test_ppca_initial_model_momentum_sgd.py`,
`test_tomo_input.py`; run before the default flip): all pass except the GPU tests and one test needing the RELION binding, which skip.

### 2026-11-10 (evening) rounds 8 and 9 (snapshots v9 and v10, the gate of `ppca_schedule` 2fd7b0e8)

Both rounds ran before the rebase onto the new VDAM gate; C n3 box 64, true sd 0.30, momentum SGD unless stated.

**Round 8: the HEALPix order change moved** (arms `t_*`: the stages table with order 3 never opened, opened at iteration 1001 or at 2376; the default
opens it at 1626; the radius ramp is the same; seed 11; kmeans ARI, state FSC, the sd the run ended at, the correlation of its contrast estimate):

| arm | order 3 at | kmeans ARI | state FSC | sd prior at the end | corr(c) |
| --- | --- | --- | --- | --- | --- |
| baseline | never / 1001 / 2376 / 1626 (default) | 0.265 / 0.345 / 0.306 / 0.346 | 0.897 / 0.904 / 0.900 / 0.902 | | |
| `cg_sgd` (unconstrained) | never / 1001 / 2376 / 1626 | 0.317 / 0.333 / 0.299 / 0.297 | 0.865 / 0.868 / 0.863 / 0.865 | 0.135 / 0.080 / 0.088 / 0.089 | 0.33 / -0.15 / -0.02 / -0.10 |
| `co_sgd` (orthogonal) | never / 1001 / 2376 / 1626 | 0.431 / 0.473 / 0.461 / 0.494 | 0.875 / 0.879 / 0.876 / 0.879 | 0.262 / 0.227 / 0.232 / 0.232 | 0.83 / 0.83 / 0.83 / 0.83 |

- The mean power jump of the baseline follows the order change wherever it is: x1.060 at 1001, x1.072 at 2376, x1.060 at 1626, and x1.01 in the
  run where it never opens. So the order change causes the jump (shells 11 to 24, see above), for SGD.
- The sd decline is not only the order change. With the order never opened the orthogonal run's sd still falls from 0.294 (iteration 1600) to
  0.262, the order change takes it to 0.23, and the unconstrained run's sd collapses in all three tables (0.135 with no order change, 0.08 to 0.09
  with it); in the unconstrained runs the steepest fall starts at the order change (iteration 1200 in the 1001 table, 2400 in the 2376 one). So the
  radius ramp (more high shells entering the E-step, the radius probe above) drives a part of the drift, and the order change accelerates it.
- The orthogonal arm is better than the unconstrained one in every table (kmeans ARI +0.11 to +0.20), as in rounds 5 to 7.

**Round 9: the sd fixed at 0.3** (`cgf_*` unconstrained loadings, `cof_*` orthogonal; centred M-step; seeds 11 and 12; the old-gate
references are rounds 4, 5 and 5b):

| arm | seed | kmeans ARI | state FSC | corr(c) | est. sd of c | reference: estimated-sd arm ARI |
| --- | --- | --- | --- | --- | --- | --- |
| SGD `cgf_sgd` | 11 / 12 | 0.203 / 0.393 | 0.882 / 0.910 | 0.07 / 0.51 | 0.171 / 0.190 | `cg_sgd` 0.297 / 0.347 |
| SGD `cof_sgd` | 11 / 12 | 0.481 / 0.568 | 0.877 / 0.915 | 0.84 / 0.83 | 0.230 / 0.229 | `co_sgd` 0.494 / 0.564 |
| VDAM `cgf_vdam` | 11 / 12 | 0.656 / 0.635 | 0.920 / 0.906 | 0.90 / 0.90 | 0.276 / 0.276 | `cg_vdam` 0.647 / 0.616 |
| VDAM `cof_vdam` | 11 / 12 | 0.626 / 0.621 | 0.891 / 0.905 | 0.90 / 0.90 | 0.276 / 0.276 | `co_vdam` 0.567 / 0.614 |

(baselines: SGD 0.346 / 0.322, VDAM 0.385 / 0.351.)
- A fixed sd does not substitute for the constraint. Unconstrained, the loadings still grow a component along the mean on top of the fixed
  latent: seed 11 has `sd^2` held at 0.09 and the loadings' parallel variance rising from 0.022 (iteration 1800) to 0.072 (3000), a total 1.8 times the true
  0.09; the gain drifts to -0.89 (mean contrast 0.73; orthogonal: -0.49 and 0.85) and the contrast correlation is 0.07. Seed 12 is milder
  (parallel 0.006 to 0.021, correlation 0.51). The kmeans ARI is worse on seed 11 and better on seed 12 than the estimated-sd arm: no consistent
  gain from fixing the sd, and a contrast estimate that does not follow the truth on one seed.
- With the constraint a fixed sd changes nothing (SGD 0.481 / 0.568 against 0.494 / 0.564), and VDAM moves within seed noise (the largest step,
  `co_vdam` seed 11 0.567 to 0.626, is one seed). The gain drift of SGD is not removed by fixing the sd, as the E-step attenuation explains
  (radius probe above).

### 2026-11-10 (night) rebase onto the new VDAM gate of `ppca_schedule` and the regression check (snapshot v11, round 10)

**The rebase.** `ppca_schedule` moved from 2fd7b0e8 to 801bd46b (17 commits: the power gate statistic with the full-data noise scale
(`advance_noise_weight`, `full_data_disagreement_scale`, `stochastic_update(..., disagreement_scale)`), `--shift-stages`,
`--offset-sigma-angstrom`, retired config keys, the math note's sections 18.16 and 18.17, fingerprint re-anchors). The 33 commits of this branch were
replayed with `git rebase ppca_schedule` (backup branch `backup/ppca_contrast_pre_rebase_20261110`, old tip c3db2b08). The only conflicts were in
`checkpoint.py` `_ADDED_CONFIG_DEFAULTS` (both sets of keys kept); `commands/ppca_initial_model.py`, `config.py` and `iteration_loop.py`
merged without conflicts and were read: the contrast fold, the orthogonal projection and the new `disagreement_scale` argument sit side by side.

Checks on the rebased tip (CPU): the unit suites of `ppca_contrast`, `ppca_initial_model`, `ppca_refinement`, the momentum SGD test and the
subtomogram input test: 537 passed, 29 skipped (GPU-only tests and one RELION-binding test), no failures. Default-path fingerprint (job 17262291,
base `ppca_schedule` against the rebased tip, same host): 248 differences, all metadata (the contrast config keys, key counts, `contrast_gain` and
`contrast_sd` in checkpoint metadata), none in an array, result or status leaf, six log rows (tile-plan lines). The fingerprint does not cover GPU
kernels or subtomogram input (the cryo-ET result below is why that matters).

**Round 10** re-ran the recent matrix on the rebased tip (snapshot v11, 23 runs, then seeds 13 and 14 in round 10b, 10 runs; same datasets,
arms and flags as rounds 3 to 7; `scripts/ppca_contrast/compare_rounds.py` pairs each run with the latest old run of the same dataset, arm and seed).
The runs differ from the old ones from the first iteration (mean power at iteration 1: 0.1023 against 0.1009), because the gate differs; the radius,
order, step and fudge schedules are identical. The momentum SGD arms do not use the gate: `co_sgd` seed 11 and the SGD baselines are identical to
three decimals, so the rebase itself changed nothing for SGD.

kmeans ARI, VDAM, new gate (old gate in brackets; baseline is `s3_vdam_uniform`):

| dataset (true sd, noise) | baseline | `co_vdam` (default under `--contrast latent`) | `cg_vdam` (unconstrained) |
| --- | --- | --- | --- |
| C n1, sd 0.30, seeds 11 to 14 | 0.585 0.589 0.607 0.574 (old 0.568 0.580 on seeds 12, 13) | 0.941 0.956 0.949 0.960 (old 0.949 0.937 on seeds 11, 12) | 0.937 0.913 0.917 0.930 (old 0.937 0.934 0.952 on seeds 11 to 13) |
| C n3, sd 0.30, seeds 11 to 14 | 0.392 0.370 0.402 0.387 (old 0.385 0.351) | 0.670 0.634 0.668 0.673 (old 0.567 0.614) | seeds 11, 12: 0.690 0.639 (old 0.647 0.616) |
| C n1, sd 0 (unit contrast), seed 11 | 0.984 | 0.976 (SGD 0.992 against baseline 1.000) | |
| C n1, log-normal sd 0.30, seed 11 | 0.940 (old 0.619) | 0.984 (old 0.980) | 0.956 (old 0.968) |

- **Default arm (`co_vdam`)**: the gain over the baseline holds: +0.34 to +0.39 at noise 1 (old +0.37), +0.26 to +0.29 at noise 3 (old +0.18 to +0.26),
  -0.008 at unit contrast (no harm). State FSC of `co_vdam` at C n1 is 0.926 0.953 0.957 0.948 (old 0.961 0.953), the baselines' state FSC also moves by 0.03
  to 0.04 between the gates in both directions (log-normal 0.955 to 0.916, C n3 seed 11 0.925 to 0.897, seed 12 0.894 to 0.926), so this size of change is
  not attributable to the contrast model. The log-normal case is the one place where the gain shrinks (+0.04): the baseline seed 11 went from 0.619 to 0.940
  under the new gate; with one seed on that dataset it says nothing yet about the gain.
- **Loading subspace** (owner's second-priority metric): under VDAM the angle of every contrast run is 10 to 25 degrees worse (C n1 `co_vdam`: 36.6 before, 47.6 to 55.2 now),
  the true energy captured 0.93 to 0.90, and the loading power relative to the mean (`PC/mean`) 0.09 to 0.12 (C n3: 0.12 to 0.19; its baseline 0.18 to 0.25).
  The same shift appears without contrast: on the unit-contrast dataset the VDAM baseline's angle went 35.3 (round 1) to 50.0 and `PC/mean` 0.089 to 0.122, so
  it follows the new gate (in the C n1 contrast runs the loading power at iteration 500 is 0.11 against 0.017, six times larger). SGD unchanged.
- **Low mean FSC under VDAM** (raw mean FSC over shells 1 to 15 below 0.97 at iteration 3000): none in 30 older VDAM runs (11 baseline, 19 contrast, rounds 1 to 7), now 4
  in 27, and they are two different things (all at iteration 3000, siblings are the other seeds of the same arm):

  | run | kmeans ARI | state FSC | pose <10 deg | pose median | mean FSC (shell 1 / 8 / 15) | energy captured |
  | --- | --- | --- | --- | --- | --- | --- |
  | C n3 `co_vdam` seed 14 | 0.673 (sibling 0.668) | 0.904 (0.919) | 0.885 (0.881) | 4.75 (4.72) | 1.00 / 0.79 / 0.16 (1.00 / 1.00 / 0.99) | 0.36 (0.91) |
  | C n3 baseline seed 13 | 0.402 (0.387) | 0.902 (0.909) | 0.876 (0.877) | 4.82 (4.78) | 1.00 / 0.79 / 0.16 | 0.33 (0.83) |
  | C n3 `cof_vdam` seed 12, round 9, old gate | 0.621 | 0.905 | 0.881 | 4.83 | 1.00 / 0.79 / 0.16 | 0.37 |
  | C n1 `cg_vdam` seed 12 | 0.913 (0.930 to 0.937) | 0.806 (0.926 to 0.938) | 0.650 (0.98) | 5.09 (4.05 to 4.14) | 1.00 / 0.94 / 0.91 | 0.61 (0.90) |
  | C n1 `cg_vdam` seed 13 | 0.917 | 0.770 | 0.551 | 6.05 | 1.00 / 0.89 / 0.78 | 0.51 |

  The three noise 3 runs (baseline, default contrast arm, and one old-gate run) have the same raw mean FSC curve to two decimals, although they differ in seed, arm and gate;
  the scores that are measured in the frame of the direct pose fit (ARI, state FSC, pose) are unchanged, and only the raw mean FSC and the energy captured against the true span move.
  That points to the frame in which the model is reported rather than to a worse map, but it is an inference: the mean map and the frame were not compared directly, and the
  direct fit's `mirror` flag does not line up with the failures (it is true in good runs too). The two unconstrained `cg_vdam` runs at noise 1 are a real but moderate degradation:
  state FSC 0.77 to 0.81 against 0.93, fewer poses within 10 degrees (0.55 to 0.65 against 0.98), the mean off by 0.05 to 0.10 in FSC, ARI 0.02 lower and still far above the
  baseline's 0.59. For `cg_vdam` seed 12 the contrast side is healthy (sd 0.306, loadings' parallel variance 0.0004, gain +0.08), so it is not the loading gauge of rounds 4 and 5.
  The default arm `co_vdam` has 0 of 4 such runs at noise 1 and 1 of 4 at noise 3 (the frame kind); the counts are too small to give rates.
- **Overall effect of the gate on these suites** (paired runs, same dataset, arm and seed; 7 baseline and 11 contrast VDAM runs): baseline kmeans ARI +0.057 on average (6 of 7
  higher; +0.014 without the log-normal seed whose baseline went 0.619 to 0.940), contrast arms +0.012 (7 of 11; `co_vdam` at C n3 seed 11 +0.103); state FSC -0.007 (baseline) and -0.036
  (contrast, driven by the two `cg_vdam` runs); pose median +0.04 and +0.32 degrees; the loading subspace angle +2 and +22 degrees and the energy captured -0.03 and -0.09. ARI is
  slightly better, the rest is within the gate-to-gate spread except the loading metrics and the two `cg_vdam` runs. Only simulated suites were run here, not the real-data or schedule
  cases the gate was written for.
- Momentum SGD: `co_sgd` C n1 seeds 11 and 12 equal the old values (0.936 and 0.404 against 0.936 and 0.391); the seed 12 failure of the earlier rounds is unchanged.

**Cryo-ET fixtures, VDAM, 200 iterations, q 2** (nearest-centroid accuracy of the latents / kmeans ARI; chance is 0.33 / 0). Old: snapshot v6, old gate.

| run | old gate (v6) | rebased tip (v11) | `ppca_schedule` tip, no contrast code (v12) | v12 with the disagreement scale set to 1 (v13, diagnostic) |
| --- | --- | --- | --- | --- |
| contrast-30 fixture, no contrast, seeds 11 / 12 | 0.855 / 0.907, 0.275 / 0.344 | 0.331 / 0.338, 0.001 / -0.002 | 0.331 / 0.338, 0.001 / -0.002 | 0.404 / 0.416, 0.005 / 0.002 |
| contrast-30 fixture, `--contrast latent` | 0.880 / 0.865, 0.354 / 0.333 | 0.373 / 0.371, 0.003 / 0.004 | | |
| unit-contrast fixture, no contrast | 0.915 / 0.905, 0.431 / 0.390 | 0.398 / 0.388, 0.004 / 0.006 | 0.398 / 0.388, 0.004 / 0.006 | 0.456 / 0.368, 0.035 / 0.001 |

- The collapse is the new gate's, not the rebase's: the plain `ppca_schedule` tip (v12, no contrast code) reproduces the rebased runs without contrast to three decimals.
- In the logs the loadings die: at iteration 200 the singular values are [80.9, 0.0] (old gate [813, 657]). The disagreement scale reaches 19 from iteration 100 (1/e with e = 0.0526, the formula's
  limit for a batch equal to the data), because the batch is then the whole 399-particle dataset (half counts 200 and 199); the gate of the loading channel in shells 3 and 4 (counted from 0) falls
  to 0.05 to 0.16 by iteration 35 to 50 while the mean's stays at 0.92 and above. The scale is derived for independent batches (`full_data_disagreement_scale`: 4 / (n_images weight)); with a
  batch equal to the data, consecutive iterations carry the same noise, so the moving average does not reduce it.
- Diagnostic (v13 only: the scale returns 1; not committed anywhere): the gates stay near 1 and the loadings survive (singular values [159, 103] and [149, 101] at iteration 200),
  but the state recovery stays near chance (accuracy 0.37 to 0.46, ARI 0 to 0.04). So the scale is what shuts the loading channel off, and removing it alone does not restore the old
  result; what else differs from the old gate (the power statistic instead of the amplitude one) is not isolated.
- The contrast latent on this fixture now ends at sd 0.300 and mean contrast 1.11 with correlation 0.82 (old VDAM runs ended at the cap, mean 1.5 to 1.6), but with the loadings dead the
  model is a mean-only one and the states are not found, so this is not an improvement of the contrast model.

**Status.** Contrast on simulated suites: the default arm holds its ARI and FSC gains under the new gate and does no harm at unit contrast; SGD is unchanged. Open: the
VDAM loading metrics moved with the gate (also in the baseline), the mean-map failure mode under VDAM at noise 3 and in the unconstrained arm, and the cryo-ET collapse under VDAM,
which belongs to the gate (to be decided by its owner). No smoke or medium tier was run on the rebased branch; nothing is pushed.

Run directories: `$R/round10`, `$R/round10b`, `$R/tomo_r10`, `$R/tomo_r10_base`, `$R/tomo_r10_noscale` with `$R` = `/scratch/gpfs/CRYOEM/gilleslab/ry2228/data/ppca_contrast`; the old-against-new
tables are `$R/round10_compare.md`.

### 2026-11-11 audit of the earlier contrast failures, model B, and the cahra_01 runs

The owner asked to move away from the cryo-ET fixtures (the gate there is theirs), to run VDAM with the latent on cahra_01 at three tau2 fudges, and to
build model B with a discretization that is accurate without a large node count.

**Audit** (the notes in this repository and RECOVAR that say contrast fails, against what happened here). Per-run numbers come from
`scripts/ppca_contrast/contrast_sd_trajectory.py` on rounds 10 and 10b; the cryo-ET numbers are the table of round 4/5 (old gate).

- Section 16.10 of `vdam_ppca_algorithm.md` (October 2026: a point estimate of the contrast from a random start gives low contrasts to misaligned particles, the
  estimated spread inflates (0.415 against 0.288) and the poses do not converge, pose median 24 deg against 5.7 deg). *SPA suites, VDAM*: the final state is
  fine (sd 0.29 to 0.31 against 0.30, pose median 3.97 against 4.04 deg), but the mechanism shows as a transient: in 9 of the 10 noise-1 VDAM runs of rounds 10 and 10b
  (and the log-normal one) the sd rises to 0.40 to 0.50 between about iteration 450 and 1020 (the stage changes of the radius ramp) and comes back (peak 0.50, at the
  cap for two iterations, in `co_vdam` seed 12); at noise 3 the peak is 0.34 in all six runs, SGD 0.32. *Cryo-ET fixture, VDAM, old gate*: the full symptom: the sd sits at
  the cap, the mean contrast estimate is 1.5 to 1.6 and the state R^2 is lower than without the latent (0.36 and 0.33, or 0.24 and 0.40 with the orthogonal loadings, against
  0.48 and 0.56); poses were not measured there (the outputs hold no estimated poses). On the rebased tip that fixture collapses with and without the latent (the base gate), and in
  the collapsed contrast runs the contrast estimate with the true contrast regressed out correlates +0.38 and +0.39 with the pose posterior's maximum (poorly determined poses
  get lower contrasts), the same signature, but there the poses are collapsed anyway.
- RECOVAR's warm-up (random loadings make `H >> I` and the contrast collapses to the grid boundary; the contrast is skipped for 3 iterations): there is no grid here; the
  analogue is the sd cap, reached on the cryo-ET VDAM runs and for two iterations on one SPA run. `--contrast-start` defaults to 1.
- RECOVAR's profile-MAP over-estimate (no log-determinant term): not present, model A integrates the contrast and its second moments enter the M-step; the bias here is the opposite
  one under SGD (the estimate shrinks toward 1: slope 0.33 at radius 8, 0.63 at 16).
- RECOVAR's sign mismatch (contrasts pegged at the minimum): not seen (mean estimates 1.015 to 1.024 under VDAM); no flipped-image test was run.
- RECOVAR's inflated variance estimate under unknown contrast, and the identifiability remark of section 6: seen and fixed (the baseline's loadings carry the contrast: subspace angle
  90 deg and energy 0.871 against 34.8 deg and 0.932 with the latent; the loading parallel to the mean is a second contrast latent, math note section 10). The orthogonality
  constraint has not been tested where it costs (oracle shares along the mean: Ribosembly 25 percent, K4 17, D40 7).
- Section 16.10's other finding, no gain in the maps, still holds: the gain is in the latents' clustering (ARI), not in state FSC, mean FSC or poses.

**Model B** (`--contrast quadrature`; math note section 11; `relax/ppca_refinement/contrast_b.py`). The contrast scales the whole signal, `c ~ N(kappa, (kappa tau)^2)`, and is
integrated per pose: the stream's columns are `[mu, W]` (no tied column, no extra GEMM channel), the latent `z` is integrated in closed form per node in the eigenbasis of `H`, and the
nodes are Gauss-Hermite nodes on model A's closed-form posterior of the contrast. Pass 1 keeps the inner products and the Gram, pass 2 recomputes the nodes and forms
`E[c alpha]`, `E[c^2 alpha alpha^T]` in the layout of the unit-contrast M-step. XLA elementwise stage on CPU and GPU (the CUDA projector and GEMMs of the stream are used on GPU).
Same two gauge fixes as model A, in their B form (M-step moments of `c / kappa`, prior sd `kappa tau` with the relative spread of the batch; orthogonal loadings by default).
Embeddings are `[E z, E c]`; `infer_for_suite.py` and the final embeddings use `iteration_loop.contrast_latents` for both models.

- Node study (`scripts/ppca_contrast/quadrature_accuracy.py`, twelve settings, results in the math note): 3 adaptive nodes keep the pose posterior within 3e-4 total variation of a
  dense reference, `E[c]` within 1.2 percent of its posterior sd and the M-step moments within 0.3 percent; equally spaced nodes need 65 for the same, nodes on the prior more than 33
  and still miss (`E[c]` off by 1.7 sd). Default 3 nodes; 5 nodes are within 3e-3 nats of the reference.
- Tests: `test_contrast_b.py` (dense reference, no spread equals the unit-contrast score and posterior moments, no loadings equals the conjugate Gaussian, more nodes converge),
  `test_contrast_b_stream.py` (the stream at no spread equals the plain stream on one pose and over many rotation blocks and translations; pass 2 against the dense reference; config,
  prior, controller update) and the subtomogram controller test with model B (VDAM and SGD). All pass on CPU and on one A100 (the CUDA projector path; the padded `weights` layout
  of the GPU stream was a bug found there).
- Cost, indicative only (microbenchmark of the elementwise stage on random operands, one A100, one rotation block of 64 rotations, 150 images, 25 translations, `q = 5`): model B's XLA stage
  0.39 ms in pass 1 and 0.75 ms for passes 1 and 2 together, model A's XLA stage 0.10 and 0.12 ms, the pass-1 inner-product GEMM 0.12 ms. At 99 translations B is 2.0 ms for both passes.
  The engine's model A uses relax's CUDA epilogue, so the real ratio is not this one; the per-iteration cost on the suites (`qo_*` against `co_*`) is not measured yet.
- Not done: the suites with `qo_vdam`, `qo1_vdam`, `qo5_vdam`, `qo_sgd` (arms in `arms.yaml`), the cost on the engine, whether B needs the gauge fixes, refinement (no controller),
  smoke and medium tiers.

**cahra_01** (real data, 1,613,664 particles, box 128; no ground truth; the dataset's poses and consensus are not used). Six VDAM runs on the recipe schedule of `cahra_01_fixed/gate_only` with
the flags of that run, seed 11, the same code (snapshot v14 = branch tip `1ca90083`): `none_default`, `latent_default`, `none_0.1`, `latent_0.1`, `none_0.01`, `latent_0.01`
(`--tau2-fudge F --tau2-fudge-scheme plain` for 0.1 and 0.01, VDAM's 1 rising to 4 for default; `--contrast latent` with its defaults: centred M-step, orthogonal loadings). The runs without
the latent are on the same snapshot so that a difference is not the 17 gate commits. Scripts and outputs under
`/scratch/gpfs/CRYOEM/gilleslab/ry2228/data/ppca_real_data/cahra_01_contrast/` (`run_vdam.sbatch ARM`, `smoke.sbatch`); report with `scripts/ppca_contrast/cahra_contrast_report.py`
(final state, sd, gain and estimate trajectories, the intraclass correlation of the contrast estimate over micrographs, whether the latents of a run without the latent carry the contrast).
Jobs 17337515 (smoke), 17337517 to 17337522; all still pending at the time of writing: the per-user running-job limit (16) is held by the owner's other project.

**Local GPU test run (one A100, outside Slurm, unit tests only), tip `ab6532a6`:** `tests/unit/ppca_contrast`, `tests/unit/ppca_initial_model` and the momentum SGD file: 468 passed, 3 skipped, 2 failed;
`tests/unit/ppca_refinement`: all pass once `test_ppca_stream_kernels.py` builds the moment carry with its `second_moment_diagonal` field (four failures before, from the earlier
commit that added the field). The two failures are resume tests that compare a resumed and a full run at rtol 1e-6 on float32: `test_radius_rule.py::..._resumes_to_the_same_radii_and_model` and
`test_pc_schedule.py::..._resumes_to_the_same_model`. They fail the same way on the base `ppca_schedule` tip `801bd46b` (four runs each, same GPU model: radius rule 4 of 4 on the base and 4 of 4
on this branch, PC schedule 1 of 4 and 2 of 4), so they are not from the contrast changes; the radius-rule test is the known GPU flake of the gate merge. No tolerance was changed (the owner's call).

