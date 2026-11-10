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
| 3c | Contrast model vs baseline; harm check at unit contrast | done (rounds 3 to 7): at sd 0.30 kmeans ARI 0.57-0.66 to 0.93-0.98 over 3 to 20 seeds (noise 1), noise 3 and log-normal; no change at sd 0; cryo-ET fixture: SGD calibrated, VDAM not settled |
| 4 | Heuristics (when to switch contrast on, prior width) and revisit | not needed: both failures found were gauges, fixed structurally (math note sections 9 and 10); a late start and a fixed sd did not help |

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
  0.09, the gain (mean of the latent) drifts to -0.74, the estimate has a correlation of -0.10 with the true contrast and
  the kmeans ARI is below the baseline (0.297 against 0.346). The centred second moment of the latent stays at 0.97 to 1.02 during the
  decay, a fixed point of the sd update, which loses 0.05 percent per step at 0.99 and compounds.

**Diagnosis** (math note section 10). A loading parallel to the mean volume is a second contrast latent, and the likelihood is flat
along the split of variance between it and the contrast latent. The cosine squared between each loading and the mean in the final
checkpoints: contrast latent runs that work have 0.000-0.003 (all VDAM runs, SGD at noise 1); the baselines at sd 0.30 have
0.2-0.5 on several loadings (the contrast variation is that direction); the failing SGD run keeps 0.18, 0.08 and 0.16 on three
loadings and 0.0-0.003 on the others. This is the strongest evidence I have and not a proof: I have not shown that removing the parallel
component prevents the decay (the round 5 below does).

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
seeds), SGD on the cryo-ET fixture is calibrated, VDAM is unchanged within run-to-run noise in every comparison (13 pairs), and the constraint
is the identifiability condition of the model (math note section 10). The `cg_*` arms pass `--no-contrast-orthogonal` and keep their meaning.

**Unit tests** (CPU, `tests/unit/{ppca_contrast,ppca_initial_model,ppca_refinement}`, `test_ppca_initial_model_momentum_sgd.py`,
`test_tomo_input.py`; run before the default flip): all pass except the GPU tests and one test needing the RELION binding, which skip.
