# PPCA ab initio: four development axes

Written 2026-10-04 from the owner's direction, for whoever works next on `relax ppca_initial_model`
(`--optimizer vdam|momentum_sgd`). The aim is to improve PPCA under both optimisers, partly by comparing
them. These are the axes to think along; they are not to be attacked all at once. Each entry separates what
is measured from what is a hypothesis. The measurements are from one synthetic three-state single-particle
set (20,000 particles, box 64, 6 Å per voxel, noise 1, three seeds or more per arm) unless stated; that set
has one defocus, no shifts and a weak state, so treat the numbers as evidence about mechanism, not as
benchmarks. The algorithm is in `docs/math/vdam_ppca_algorithm.md`.

## 1. Frequency marching: by annealing or by the VDAM gate

Some control of which frequencies move, and when, appears necessary. We want to understand how each
optimiser provides it.

- Measured. Plain SGD with every frequency open from the first update converges to an overfit state with
  wrong poses (5 of 5 runs); learning rate, batch size, step decay and preconditioning do not change that.
  Inflating the E-step noise 8 times and decaying it to 1 over 750 updates makes the same step align 6 of 6
  with every frequency open. A hard radius ramp alone aligned 3 of 3 from start radius 7, 1 of 3 from 5 and
  0 of 3 from 10. VDAM aligns in every run; its half-set gate limits each shell's step by the agreement of
  the two half-sets, which acts as marching chosen by the data.
- Measured. With the true model, a particle's pose is not identifiable below Fourier radius 16 on this set
  (0.7% of best poses within 10 degrees at radius 10, 74% at 16). Marching has to start where pose
  information exists.
- Open. What each scheme does shell by shell over time (which shells move, when poses commit), compared on
  the same data.

## 2. Regularisation: loadings and latents trade scale

VDAM-style regularisation shrinks the loadings (the PCs). The model can answer by enlarging the latent
coordinates, and the two are hard to tell apart, so PPCA is hard to regularise.

- Measured. The weak state is posed only while the gate is tight (gate about 0.78–0.90 on the mean and
  0.61–0.77 on the loadings); when the gate schedule loosens it is lost again while alignment stays. The
  latent posterior covariance trace is 0.25–0.57 under VDAM against 0.03–0.09 under plain SGD (prior 2).
  "Posing the weak state" is learning the second, weaker component: where it fails, two states share one
  latent point.
- Measured. VDAM's mean map is worse than SGD's (FSC against the true mean about 0.76 against 0.95), and
  under VDAM the weak state sits at latent zero.
- Hypothesis (owner). This scale trade is why, under VDAM, the mean appears to be at much lower resolution
  than the PCs, and in the end the mean is absorbed into the PCs.
- Measured: the hypothesis holds as a gauge effect. VDAM's latents are not centred (`|E[z]|` 1.6–1.7, rms
  `|z|` 2.8 against a unit prior, loadings 3–6 times smaller than SGD's), so part of the average structure
  sits in `W E[z]`. Per band, the FSC of VDAM's mean against the true mean is .88/.71/.75/.65/.28/.37, and of
  `mu + W E[z]` .97/.91/.92/.87/.54/.67 (SGD's mean: .97/.94/.92/.88/.73/.70). Score and report the mean in
  the centred gauge; an earlier "loss of the common frame" under VDAM was this artefact.
- Measured (continuation runs, three seeds). The gate makes the three-state solution the attractor: VDAM
  recovers the weak state from an SGD model that had lost it (3 of 3, within 200 updates). The gate on the
  mean is what poses the weak state; the gate on the loadings alone gives the clusters without the pose.
  Plain SGD keeps whichever solution it is given at learning rate 1.2 and destroys the three-state one at 4.8
  within 100–300 updates (loadings grow sevenfold, latents are pinned, then the pose is lost, the clusters
  merge last); a smaller step (0.3) recovers nothing.
- Open. A regulariser that fixes the scale split between loadings and latents; whether shrinkage with no
  half-sets can do what the gate does. Under test: re-whitening the latents after each update.

## 3. PCs can stand in for pose sampling that is too coarse

A PC can represent the error of a pose grid that is too coarse instead of real heterogeneity. That defeats
frequency marching and its efficiency, because the model appears to gain resolution that the sampling
cannot support.

- Proposal (owner). Estimate the resolution attainable under a sampling of X degrees and Y translation,
  with X and Y chosen to be about equal in physical units at the particle radius, and forbid the PCs any
  frequency content beyond a fraction of it (for example three quarters). A starting estimate is
  `d = c * max(R * dtheta, dt)` for particle radius `R`, angular step `dtheta` in radians and translation
  step `dt`; the constant `c` is to be calibrated by reconstructing with true poses snapped to the grid.
- Metric needed: did the PCs absorb pose error? Two candidates. (a) Align the PCs to the mean: a small pose
  error of the mean is, to first order, a combination of the mean's six rigid-motion derivatives (three
  rotations, three translations), so measure the share of each PC's power, per shell, that lies in that
  span and compare it with the share for the true components. (b) Look at the latent space: absorbed pose
  error shows as streaks, one true state spread along a line or split into several clusters; on synthetic
  data, regress the latent coordinates on each particle's residual pose error.
- Measured on saved models. Both optimisers absorb pose error into the latents, SGD much more. Within the
  strong state, the regression of the latent on the residual rotation error has R2 .70–.86 under plain SGD
  and .35–.55 under VDAM (null .02); the PCs' rigid-motion share in shells 11–15 is .17–.21 under SGD and
  .03–.05 under VDAM (truth .02). Both rise when the gate loosens.
- Measured with poses given and held fixed (no pose search; exact EM; 20,000 particles). Absorption rises
  with the injected error: rotation error of 0 / 2 / 4 / 8 degrees gives clustering ARI 1.00 / 1.00 / .95 /
  .34. Shift error dominates (measured on a variant of the set with a defocus spread, shifts and non-negative
  truth): a shift error of 1.68 pixels rms alone gives ARI .18 where 4 degrees of rotation alone leaves 1.00,
  and 2 degrees with 0.84 pixel already merges the clusters (.41–.45) at two components. Rotation and
  translation errors of equal size at the particle radius do not
  behave alike; they do roughly if the radius is replaced by the radius of gyration (about a third of it).
  Extra components take pose error and leave the state separation intact for rotation error (ARI .95 / .98 /
  .98 at 2 / 3 / 8 components), and help but do not suffice for shift error.
- Constant. For rotation, the PCs' rigid share leaves the truth's at the shell where `d = c R dtheta` with
  `c` about 3.5 (`R` the particle radius). For shifts the departure is at or below the lowest shell scanned
  (`c` at least 15).
- Metric. Use the clustering score (adjusted Rand index of a mixture fitted to the latent means, against the
  true states) as the headline, not the between-state R2: R2 falls with the number of components and mixes
  pose error with heterogeneity.

## 4. Refining the pose grid

How and when the rotation and translation grids are refined as resolution grows, and how that interacts
with axes 1 and 3 (a grid refined too late invites absorption; refined too early it costs time with no
gain).

- Measured. With true poses snapped to the training grid (7.5 degrees, 2 pixels) on a set with real shifts,
  exact EM cannot separate three states (ARI .43): the translation step, not the rotation step, limits
  heterogeneity there. Refine or interpolate the shift grid before band-limiting the PCs.
- Open. The refinement schedule, and its cost.
