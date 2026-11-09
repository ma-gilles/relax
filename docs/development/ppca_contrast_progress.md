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
| 2 | Implementation shared by initial model and refinement | model A in the initial model (`--contrast latent`) and in `contrast.py` for any engine; CPU tests pass; GPU smoke run in progress |
| 3a | Contrast-spread simulated datasets with a paired control | done: C n1 box 64 at sd 0, 0.15, 0.30 |
| 3b | Cost of ignoring contrast (baseline runs) | running (VDAM and momentum SGD at the default settings, seed 11) |
| 3c | Contrast model vs baseline; harm check at unit contrast | not started |
| 4 | Heuristics (when to switch contrast on, prior width) and revisit | not started |

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
