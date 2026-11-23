# Conical-FSC regularization: RELION's tau2 rule per cone of Fourier space

**Status (2026-11-23, branch `dev/cfsc_regularization`).** Derived (sections 2-4), implemented as the opt-in K=1 option
`relax refine --cfsc-regularization` (section 5) and checked on synthetic accumulators and on the tiny CPU refinement
(section 6). The benchmark test on the pose-prior benchmark's `kent_cap_offsets` scenario is in section 7. The
implementation is [`relax/reconstruction/conical_regularization.py`](../../relax/reconstruction/conical_regularization.py);
its tests are [`tests/unit/test_conical_regularization.py`](../../tests/unit/test_conical_regularization.py).

## 1. Summary

RELION regularizes every reconstruction of an auto-refine with a Wiener-like prior whose signal power `tau2` is
estimated, shell by shell, from the FSC of the two half maps. The rule assumes that one number describes a shell: the
signal-to-noise ratio, the Fourier weight and the signal power are taken to be the same in every direction of the shell.
Under a strongly preferred orientation none of the three holds. In the pose-prior benchmark the half-map conical FSCs
(cFSCs) differ strongly between cones of one shell: in the `kent_cap_offsets` scenario the half-map cone area ratio
(cFAR, the smallest cone's area under the cFSC over the largest one's, 1 for an isotropic map) is 0.020 to 0.057
across the direction-prior variants.

We therefore apply RELION's rule inside double cones of Fourier space instead of whole shells, and give every voxel the
estimate of the cones that contain it. The construction keeps everything else of RELION: the same downsampled half
averages, the same FSC-to-SSNR conversion and clamps, the same `sigma2` (inverse mean weight), the same support of the
MAP term, and RELION's shell value as the anchor, so that equal statistics in every cone give exactly RELION's prior.

- **What RELION's rule estimates.** Under the usual Gaussian model, the shell rule returns the signal power divided by
  the Jensen factor `J_s = <W>_s <1/W>_s >= 1` of the shell's weights (section 2.2). In an isotropic data set `J_s` is
  close to 1; under a preferred orientation the weight spans orders of magnitude within a shell, `J_s` is large, and
  RELION over-regularizes the whole shell. A cone is more homogeneous in weight, so we expect its Jensen factor to
  be much smaller.
- **Pose errors are directional too.** The FSC measures the signal that survives the pose errors. When the rarely seen
  views are also the badly aligned ones, the surviving signal depends on direction, and only a directional estimate
  follows it (section 2.3).
- **The rule (section 3).** For each cone `j` and shell `s`, RELION's estimate `SSNR_j(s) sigma2_j(s)` over the shell's
  `SSNR(s) sigma2(s)` is a ratio `rho_j(s)`; a voxel's `tau2` is RELION's shell `tau2` times the mean ratio of the cones
  that contain it.
- **Noise in a cone's FSC (section 3.3).** A cone holds far fewer voxels than a shell, and RELION's clamp turns the
  positive noise of a near-zero FSC into SSNR while it clips the negative noise. Each shell's cone FSCs are therefore
  shrunk toward the shell FSC by the share of their spread that is not noise; cones that differ by noise alone keep
  RELION's value.
- **What does not change.** The shell FSC still sets the resolution, the image size (`current_size`), the reported
  `tau2` and `data_vs_prior`. Only the prior term inside the reconstructions changes.

## 2. RELION's rule and what it estimates

### 2.1 The rule

We will denote by `p` the padding factor, by `T` the `--tau2_fudge` factor and by `W_h(k)`, `D_h(k)` the padded Fourier
weight and data accumulators of half `h` (`BackProjector::weight` and `data`), `k` a padded integer frequency. RELION
first averages each half onto the native grid (`BackProjector::getDownsampledAverage`): with
`B(nu) = {k : ROUND(k / p) = nu}`,

```
W^_h(nu) = sum_{k in B(nu)} W_h(k),        F^_h(nu) = sum_{k in B(nu)} D_h(k) / W^_h(nu),
```

where `F^_h` is the unregularized half reconstruction on the native grid and `W^_h` its weight. A shell is
`S_s = {nu : ROUND(|nu|) = s, |nu| <= r_max}` with `r_max = current_size / 2`. The half-set FSC is

```
FSC(s) = sum_{S_s} Re(F^_1 conj F^_2) / sqrt( sum_{S_s} |F^_1|^2  sum_{S_s} |F^_2|^2 ).
```

`BackProjector::updateSSNRarrays` clamps it, `c = min(max(FSC, 0.001), 0.999)` (for the final joined map first
`c = sqrt(2c / (c + 1))`), and sets

```
SSNR(s) = T c / (1 - c),     sigma2_h(s) = 1 / (p^3 <W_h>_s),     tau2_h(s) = SSNR(s) sigma2_h(s),        (1)
```

where `<W_h>_s` is the mean padded weight of the shell (`ires = ROUND(|k| / p)`, `|k| < p r_max`). Finally
`BackProjector::reconstruct` adds the prior to every padded weight of the MAP support,

```
F_h(k) = D_h(k) / ( W_h(k) + 1 / (p^3 T tau2_h(ires(k))) ),    |k| < p r_max,  ires(k) >= minres_map,     (2)
```

and floors the regularized weight at 1/1000 of its shell average. The code is
`regularization.compute_relion_fsc_from_backprojector` and `regularization.compute_relion_tau2_from_weights`, and the
solve is RECOVAR's `adjust_regularization_relion_style`.

### 2.2 What the shell rule estimates

We recall the model behind (1). An unregularized half voxel is the signal plus noise of variance `1 / W^_h`,
`F^_h(nu) = S(nu) + N_h(nu)` with `E|N_h(nu)|^2 = 1 / W^_h(nu)`, independent between the halves, and the signal power
`E|S|^2 = tau2` is constant on a region `R`. Then, for halves of similar weight `W^`,

```
FSC_R ~ tau2 / (tau2 + <1/W^>_R),    so    SSNR_R = FSC_R / (1 - FSC_R) ~ tau2 / <1/W^>_R,
```

and the estimate (1), whose `sigma2` is the inverse of the *mean* weight (`p^3 <W>` is the mean of `W^`), is

```
tau2^_R = SSNR_R / <W^>_R = tau2 / J_R,        J_R = <W^>_R <1/W^>_R >= 1,                               (3)
```

with equality only when the weight is constant on `R` (Cauchy-Schwarz). The optimal Wiener factor of a voxel is
`W^ tau2 / (W^ tau2 + 1)`; RELION's uses `tau2 / J_s` in place of `tau2` and therefore shrinks every voxel of the shell
more than the model asks for. The shrinkage matters where `W^ tau2` is of order 1 or below: in poorly sampled directions
at every resolution, and in well-sampled ones near the resolution limit.

In an isotropic data set the weight varies mildly within a shell and `J_s` is close to 1. Under a preferred
orientation it does not. For example, if a fraction `f = 0.1` of a shell has weight `0.01 W0` and the rest `W0`, then
`J = (0.9 + 0.001)(0.9 + 10) = 9.8`: the shell's `tau2` is underestimated tenfold. A cone of 20 degrees contains
directions of more similar weight, so we expect its Jensen factor to be much closer to 1 where the shell's is large.

### 2.3 Directional signal loss

The FSC measures the signal that survives the pose errors, not the structure's power. If the poses of the particles
that fill a region of Fourier space are off by small rotations and shifts, the half map's expected value is the
attenuated signal `A(nu) S(nu)`, with `A` the characteristic function of the effective displacement, and the model above
holds with `tau2` replaced by `tau2 A(nu)^2`. Under a preferred orientation the rare views fill the poorly sampled cones
and are also the hardest to align (fewer particles of each view, and a learned prior that pulls them toward the common
views), so `A` depends on direction. A shell estimate averages `tau2 A^2` over directions that carry very different
amounts of surviving signal; a cone estimate follows it. Note that the Wiener filter with the *surviving* signal power is
the right one for the current poses: it is the minimum-mean-square-error estimate of the expected half map.

## 3. The conical rule

### 3.1 Cones

We take `J` axes `a_1, ..., a_J` spread uniformly over the hemisphere (a Fibonacci lattice,
`conical_regularization.cone_axes`, as the benchmark's cFSC uses) and a half-angle `alpha`. The double cone of `a_j` is

```
C_j = {nu != 0 : |nu . a_j| >= |nu| cos(alpha)},
```

a cone and its opposite, since the Fourier transform of a real map is Hermitian. We require the cones to cover the
sphere (every direction within `alpha` of some `+-a_j`; `ConicalPriorSettings` refuses settings that do not). The
defaults are `alpha = 20` degrees and `J = 100` (neighbouring axes about 14 degrees apart), the benchmark's cFSC
settings; every direction then lies in about six cones.

### 3.2 Per-cone statistics

The cone-shell bins are `S_{j,s} = S_s ∩ C_j`, on the same native averages `F^_h`, `W^_h` as RELION's FSC. In each bin
we apply (1):

```
cFSC_j(s) = sum_{S_{j,s}} Re(F^_1 conj F^_2) / sqrt( sum_{S_{j,s}} |F^_1|^2  sum_{S_{j,s}} |F^_2|^2 ),
SSNR_j(s) = T c_j / (1 - c_j),      c_j = clamp(cFSC_j(s))  (RELION's clamp and whole-map transform),
sigma2_{h,j}(s) = 1 / <W^_h>_{S_{j,s}}.
```

The same sums over the whole shell give the native-grid counterparts `SSNR~(s)` and `sigma2~_h(s)` (this shell FSC is
RELION's backprojector FSC with every Hermitian pair counted once; the test
`test_shell_fsc_matches_relion_backprojector_fsc_with_pairs_counted_once` checks it).

### 3.3 Noise in the cone FSC

A cone's FSC over `m` native voxels (`m / 2` independent complex values, since `nu` and `-nu` are one) scatters around
its true value `F` with variance about `(1 - F^2)^2 / m`. The shell FSC has the same variance with the shell's
`n ~ 16.6 m` voxels (a double cone of 20 degrees holds `1 - cos 20 = 6.0%` of a shell), so a cone's FSC is about four
times noisier. Where the true FSC is near zero, RELION's
clamp `max(FSC, 0.001)` clips the negative fluctuations but turns the positive ones into SSNR, so the plain per-cone rule
gives random cones of a signal-free shell a `tau2` up to a hundred times the shell's. We measured this on a
2000-particle, box-64 smoke refinement: at the edge shell of its window (shell FSC -0.005), cone FSCs of 0.04 to 0.14
gave ratios of 90 to 190. In a refinement it matters most in the final iteration, which reconstructs the whole box,
far beyond the resolution.

We therefore shrink each shell's cone FSCs toward its shell FSC by the share of their spread that is not noise (an
empirical-Bayes estimate):

```
lambda_s = max(0, 1 - mean_j[(1 - FSC~(s)^2)^2 / m_{j,s}] / var_j[cFSC_j(s)]),
cFSC'_j(s) = FSC~(s) + lambda_s (cFSC_j(s) - FSC~(s)),                                                  (4)
```

where the mean and the variance run over the measured cones of the shell (`lambda_s = 0` with fewer than two), and
`SSNR_j(s)` in (5) is RELION's SSNR of `cFSC'_j(s)`. A shell whose cones differ by noise alone gets `lambda_s ~ 0`,
that is, RELION's value; a shell with a real anisotropy keeps most of it. In the smoke refinement, `lambda` is 0.8 to
0.9 in the shells with signal (cone FSCs spread 2.5 to 4 times the noise) and 0 at the signal-free edge shell, where
the cone FSCs spread less than the predicted noise (sd 0.042 against 0.053).

### 3.4 The ratio and the per-voxel prior

The cone-to-shell ratio of half `h` is

```
rho_{h,j}(s) = [ SSNR_j(s) / SSNR~(s) ] * [ sigma2_{h,j}(s) / sigma2~_h(s) ]
             = [ SSNR_j(s) / SSNR~(s) ] * [ <W^_h>_{S_s} / <W^_h>_{S_{j,s}} ],                            (5)
```

the fudge factor cancelling. A bin with fewer than `m = 100` native voxels (`MIN_CONE_VOXELS`, the benchmark's
measurement threshold: an FSC of `m` voxels has a standard error of about `1 / sqrt(m / 2)`), without power or without
weight keeps the shell value, `rho = 1`. At 20 degrees a cone holds about `0.76 s^2` voxels of shell `s`, so the cones
are used from shell 12 on; below it the prior is RELION's.

A native voxel takes the mean ratio of the cones that contain it, at its shell `s(nu) = ROUND(|nu|)` (clipped to
`1 .. r_max`):

```
rho_h(nu) = (1 / |J(nu)|) sum_{j in J(nu)} rho_{h,j}(s(nu)),       J(nu) = {j : nu in C_j},                (6)
```

a box kernel of radius `alpha` over the axes. The prior of a padded voxel is RELION's shell value times the ratio of its
native voxel, and it enters (2) unchanged otherwise:

```
tau2_h(k) = tau2_h(ires(k)) * rho_h(ROUND(k / p)),
F_h(k) = D_h(k) / ( W_h(k) + 1 / (p^3 T tau2_h(k)) ),      on the support of (2).                        (7)
```

Expanding (7) with (1) and (5) shows what it is: `tau2_h(k)` is the mean over the voxel's cones of RELION's own estimate
`SSNR_j sigma2_{h,j}`, times `sigma2_h / sigma2~_h`, the ratio of RELION's padded-grid shell `sigma2` to the native-grid
one (about 1, it absorbs the difference between the two binnings). By (3), each cone's estimate is its surviving signal
power over its own Jensen factor.

### 3.5 Where it applies

- **Numbered split-half M-steps.** Each half has its own `tau2_h(s)` and its own ratio, from its own weight `W^_h`,
  as RELION computes `sigma2` per half; the FSC terms are shared.
- **The final all-data M-step.** RELION converts the joined FSC with the whole-map transform and computes `sigma2` from
  the summed weights; the conical rule does the same, with one ratio from `W^_1 + W^_2`, used for both final half maps
  and the merged map.

### 3.6 Properties

1. **It reduces to RELION.** When every cone's FSC and mean weight equal the shell's, or the cones' FSCs differ by
   noise alone and their weights agree, `rho = 1` and (7) is (2). With
   `rho = 1` the implementation reproduces RELION's regularized map to float precision, in both accumulator layouts
   (`test_unit_ratios_reproduce_relion_shell_reconstruction`) and in a whole refinement
   (`test_refinement_with_unmeasured_cones_gives_relions_maps`).
2. **It stays gold standard.** The cFSCs correlate the two independent halves; each half is regularized with its own
   weight, as in RELION.
3. **It needs no new parameter beyond the cones.** The half-angle and the number of axes set the angular resolution of
   the prior; the shrinkage is estimated from the data; `m` is the benchmark's existing measurement threshold.
4. **Cost.** The statistics take `O(N J)` operations for the `N` native voxels inside `r_max` (each tested against the
   `J` axes once per M-step) and the prior term `O(N_p)` for the `N_p` padded voxels.

## 4. Choices we made, and their alternatives

1. **Anchoring on RELION's value.** We could set `tau2(k)` directly to the mean of `SSNR_j sigma2_j`. The ratio form
   (5)-(7) gives the same up to the binning of `sigma2` and makes the reduction to RELION exact, which keeps every other
   part of RELION's arithmetic (the clamps, the double fudge factor of (1)-(2), the padded-grid `sigma2`) as it is.
2. **The mean of the ratios.** (6) averages `tau2` linearly. A geometric mean or a mean of `1 / tau2` would weigh low
   cones more; we chose the linear kernel average of the signal power, the simplest smooth interpolation, and note that
   it regularizes the least of the three at a boundary between cones of very different SSNR.
3. **Shrinkage instead of a threshold.** A significance threshold on each cone FSC (zero below a few standard errors)
   would also remove the noise floor, but it biases weak real signal to zero and adds a parameter. The shrinkage (4)
   needs neither, follows the per-cone noise model, and is exact in the two limits (no anisotropy: RELION; anisotropy
   far above the noise: the plain per-cone rule). The noise model ignores the overlap of neighbouring cones and the
   correlation of neighbouring native voxels; the smoke data's signal-free shell suggests it overestimates the noise
   slightly, which shrinks a little more than needed.
4. **The resolution and the image size stay RELION's.** RELION's current resolution is the last shell with
   `SSNR > 1` (shell FSC above about 0.5), and it sets the image size of the next expectation. Under a preferred
   orientation this, too, is a shell average: in the benchmark's `kent_cap_offsets` baseline RELION's resolution stays at
   21.8 A (shell 25, `current_size` 114) for ten iterations, although the final map reaches 8.2 A against the ground
   truth. A cone-aware window (for example the resolution of the best cones) is a separate change; keeping it fixed here
   isolates the effect of the prior.
5. **`data_vs_prior` and the reported `tau2` stay RELION's.** The model STARs keep RELION's shell curves; the cone
   statistics are written next to them (section 5).

## 5. Implementation

`relax refine --cfsc-regularization [--cfsc-half-angle 20] [--cfsc-axes 100]` builds `ConicalPriorSettings`
(`refinement_arguments.resolve_conical_prior`), which `ReconstructionSettings.conical_prior` carries to every M-step.

- **Numbered M-steps.** `maximization.k1_maximization` calls `priors.estimate_split_half_cone_prior` after RELION's
  split-half prior; it calls `conical_regularization.estimate_cone_prior` with each half's shell curve. The solves
  (`numbered_reconstruction._reconstruct_k1_maps`) take `ConePrior.regularized_weight(h, W_h)`, the weight plus the
  term `1 / (p^3 T tau2_h(k))` of (7), and run without a shell prior. RELION's `reconstruct` adds the term to `Fweight`
  and then floors the regularized weight at 1/1000 of its shell average; adding it before the solve gives the same
  regularized weight and the same floor, so the solver's support, floor and windowing stay RELION's.
- **The final M-step.** `final_reconstruction.compute_final_halfmap_prior` builds the whole-data prior (summed weights,
  RELION's whole-map SSNR) and `reconstruct_final_halfmaps` solves the three maps with it.
- **Statistics.** `native_half_averages` bins every padded voxel of both halves onto the native grid in the
  accumulators' own axis frame (the full grid, so a packed half and its full expansion give the same statistics); the
  cones and the voxels the term is added to share that frame. `cone_shell_sums` gives the sums of section 3.2,
  `cone_fsc_shrinkage` gives `lambda_s` of (4), `cone_to_shell_ratios` gives (5), `voxel_ratio_cubes` gives (6) on the
  native cube of half-width `r_max + 1` (which holds `ROUND(k / p)` for every `k` of the support), and
  `ConePrior.weight_term` gives the term of (7) on the accumulator, with RECOVAR's `ires` and support.
- **Records.** Each half model STAR of a numbered iteration carries the blocks `data_relax_cfsc_regularization` (half
  angle, axes, minimum voxels) and `data_relax_cfsc_regularization_bins` (one row per cone-shell bin: axis, voxels,
  cFSC, the shell's `lambda`, ratio, whether measured); the log has one line per M-step with the spread of the
  ratios.
- **Admission.** K=1 single-particle runs only; not with `--solvent_correct_fsc` (which replaces the FSC the ratios
  are anchored to), a RELION replay or a frozen boundary, nor on a reconstruction grid large enough for RECOVAR's
  single-precision path (200 M voxels; box 256 at padding 2 is 134 M).

## 6. Checks

`tests/unit/test_conical_regularization.py` (CPU): unit ratios reproduce RELION's regularized map (packed and full
layouts) and RELION's maps in a tiny refinement; the native shell FSC equals RELION's backprojector FSC with pairs
counted once; a packed half and its full expansion give the same prior; a better-sampled cone gets a ratio below 1 and
the others above when the FSC saturates; a cone without shared signal gets a ratio far below 1; cones that differ by
noise alone keep ratios near 1 (the shrinkage cuts the median |log ratio| at least fivefold, measured 15 to 25 fold);
the whole-data prior has one weight set; RELION's SSNR transform; the cone coverage check; the STAR blocks parse back.

## 7. Benchmark test

The test case is the pose-prior benchmark's `kent_cap_offsets` scenario (100k particles, box 256, noise level 1, a
Kent cap of preferred views with `kappa = 8`, offsets of 7 A per axis), where the learned direction prior falls well
short of the true one: in the Della runs of `offsets_100k_box256_noise1` (relax 29fde1de), the aligned ground-truth
FSC-AUC is 0.326 with RELION's learned prior and 0.366 with the true prior, and the median pose error is 37 and 10
degrees.

*Results pending: the runs `learned` (control, RELION's regularization rerun on Grace), `learned_cfsc` and
`fixed_true_cfsc` are queued.*
