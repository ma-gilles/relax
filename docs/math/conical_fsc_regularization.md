# Conical-FSC regularization: RELION's tau2 rule per cone of Fourier space

**Status (2026-11-23, branch `dev/cfsc_regularization`).** Derived (sections 2-4), implemented as the opt-in K=1 option
`relax refine --cfsc-regularization` (section 5) and checked on synthetic accumulators and on the tiny CPU refinement
(section 6), and tested on one cell of the pose-prior benchmark (section 7), where it improves the learned-prior
refinement's ground-truth FSC area from 0.319 to 0.347 (the true prior gives 0.371) and its poses with it. The
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
  `tau2` and `data_vs_prior`. Only the prior term inside the reconstructions changes. Since the half maps change, the
  shell FSC computed from them changes too (section 7.3).

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
   truth. A cone-aware window (for example the resolution of the best cones) is a separate change. We keep RELION's
   rule; we note, however, that its input, the FSC of the regularized half maps, does change with the cone prior, so
   the resolution and the image size do not stay those of a RELION run (section 7.3).
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
  accumulators' own axis frame, one plane of the full grid at a time (a packed half is expanded plane by plane,
  `regularization.packed_half_plane_to_full_numpy`, so a packed half and its full expansion give the same statistics);
  the cones and the voxels the term is added to share that frame. `cone_shell_sums` gives the sums of section 3.2,
  `cone_fsc_shrinkage` gives `lambda_s` of (4), `cone_to_shell_ratios` gives (5), `voxel_ratio_cubes` gives (6) on the
  native cube of half-width `r_max + 1` (which holds `ROUND(k / p)` for every `k` of the support), and
  `ConePrior.regularized_weight` adds the term of (7) to the weight a block of planes at a time, with RECOVAR's `ires`
  and support.
- **Records.** Each half model STAR of a numbered iteration carries the blocks `data_relax_cfsc_regularization` (half
  angle, axes, minimum voxels) and `data_relax_cfsc_regularization_bins` (one row per cone-shell bin: axis, voxels,
  cFSC, the shell's `lambda`, ratio, whether measured); the log has one line per M-step with the spread of the
  ratios.
- **Precision.** The regularized weight has the dtype the shell prior would have in the same solve. The numbered solves
  promote RELION's shell prior to float64, as RELION's `BackProjector::reconstruct` works in double, so the weight is
  float64 there. From 200 M voxels in the accumulator or in the padded reconstruction grid (box 293 and up at padding
  2), RECOVAR solves in single precision (`relion_functions.post_process_from_filter_v2`): it casts the weight and the
  shell prior to float32, and its host-staged solve (`volume_solver.reconstruct_volume_eager`) takes only a float32
  weight, so the cone prior's weight is float32 there (`conical_regularization.large_grid_single_precision`, decided in
  `estimate_cone_prior`). A host weight (large grids are solved from host-staged accumulators) gets the term a block of
  planes at a time in one new float32 array, and a device weight gets it in a donated buffer, so no accumulator-sized
  float64 array is made.
- **Admission.** K=1 single-particle runs only; not with `--solvent_correct_fsc` (which replaces the FSC the ratios
  are anchored to), a RELION replay or a frozen boundary.

## 6. Checks

`tests/unit/test_conical_regularization.py` (CPU): unit ratios reproduce RELION's regularized map (packed and full
layouts) and RELION's maps in a tiny refinement; the native shell FSC equals RELION's backprojector FSC with pairs
counted once; a packed half and its full expansion give the same prior; a better-sampled cone gets a ratio below 1 and
the others above when the FSC saturates; a cone without shared signal gets a ratio far below 1; cones that differ by
noise alone keep ratios near 1 (the shrinkage cuts the median |log ratio| at least fivefold, measured 15 to 25 fold);
the whole-data prior has one weight set; RELION's SSNR transform; the cone coverage check; the STAR blocks parse back.
On RECOVAR's large-grid path, forced on the test grid by lowering its voxel threshold, the weight is float32 and unit
ratios reproduce RELAX's shell-prior map on that path, in the full, packed host (host-staged, including the Stage-A
branch that takes only float32) and packed device layouts; below the threshold the weight keeps the caller's dtype.
`tests/unit/test_regularization_relion.py` checks the plane-by-plane expansion of a packed half against the full one.

End to end, we reran `learned_cfsc` of Section 7 (same command, seed and card model, RTX 8000) with RECOVAR's voxel
threshold lowered to 1.5e7, so that every reconstruction, including the 512^3 inverse-FFT grid of box 256, took the
large-grid path (job 62763381, the large-grid change on 2852a3db). Its per-iteration ground-truth FSC area stays within
6.3e-5 of the double-precision run through iteration 12, and the median cone ratio within 0.002. At iteration 13 the
auto-sampling stayed at HEALPix order 3 for two more iterations, where the double-precision run moved to order 4, and
the trajectories part from there: the run stopped after 18 iterations instead of 17, with a ground-truth FSC area of
0.358 against 0.347 and a median direction error of 37.5 against 40.6 degrees. We read this as one draw of the spread
that a flipped discrete decision causes, not as a gain from single precision. The difference also includes RECOVAR's
own float32 arrays on that path, which this run cannot separate from the cone prior's float32 weight.

## 7. Benchmark test

The test case is the cell `off_w100_n100k_nz1` of the pose-prior benchmark's mild-cap offsets sweep
(`sweep_mild_offsets_box256`): 100k particles of a Kent cap of preferred views (`kappa = 4`, `beta = 1`), box 256, noise
level 1, offsets of 7 A per axis. There the learned direction prior falls well short of the true one. In the Della runs
(relax c32edba9 for `learned`, `fixed_true` and `fixed_uniform`, d2dff905 for `fvml`), the aligned ground-truth FSC-AUC
is 0.319 with RELION's learned prior, 0.326 with the FvML-mixture prior, 0.303 with a uniform prior and 0.371 with the
true prior, and the median pose error is 48.5, 46.6, 63.0 and 22.3 degrees.

We ran three refinements on Grace from the snapshot of commit 2faa4a37, with the benchmark's refinement settings
(optimiser seed 1): `learned` (RELION's regularization, the control) and
`learned_cfsc` (`--cfsc-regularization`, 20 degree cones on 100 axes) on two RTX 8000 cards of one node, and
`fixed_true_cfsc` (the true direction prior and the cone rule) on a V100. Each was scored by `evaluate_run.py`. In
addition, every run's final poses were reconstructed by `relion_reconstruct` without any regularization
(`reconstruct_with_poses.py`), which scores the poses alone. The control matches the Della run (commit c32edba9, same
seed) to within 0.0004 in ground-truth FSC area; its per-iteration resolution and `ave_Pmax` agree, and so do
its median pose errors. We therefore compare `fixed_true_cfsc` with the Della `fixed_true` run. Same-seed
replicates on other cards give 0.3190 for the control and 0.3490 for `learned_cfsc` (A100s; the latter stopped one
iteration earlier), and 0.3823 again for `fixed_true_cfsc` (a second V100); the hardware spread is thus at most
0.0023, against a gain of 0.028 to 0.030. Both rows of a pair use the same seed, so the run-to-run spread over seeds
is not measured here.

### 7.1 Results

| Variant | Machine | Iter. | GT FSC area | GT FSC area, poses only | GT res. 0.143 (A) | Direction error, median (deg) | In-plane error, median (deg) | Origin error, median (A) | Particles within 10 deg | GT cFAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_uniform` | Della | 16 | 0.304 | | 9.22 | 63.1 | 38.3 | 7.15 | 28.9% | 0.053 |
| `learned` | Della | 15 | 0.319 | 0.286 | 8.92 | 48.5 | 27.5 | 6.59 | 33.4% | 0.051 |
| `learned` (control) | Grace | 15 | 0.319 | 0.286 | 8.77 | 48.5 | 27.5 | 6.58 | 33.4% | 0.050 |
| `fvml` | Della | 16 | 0.326 | 0.292 | 8.77 | 46.6 | 25.7 | 6.51 | 34.2% | 0.048 |
| **`learned_cfsc`** | Grace | 17 | **0.347** | **0.310** | 8.24 | 40.6 | 19.2 | 6.08 | 37.7% | 0.036 |
| `fixed_true` | Della | 17 | 0.371 | 0.329 | 8.00 | 22.3 | 6.4 | 4.91 | 43.2% | 0.046 |
| **`fixed_true_cfsc`** | Grace | 14 | **0.382** | **0.339** | 7.66 | 20.9 | 5.8 | 4.73 | 43.9% | 0.033 |

The GT columns are those of the final map aligned to the ground truth; "poses only" is the unregularized
reconstruction of the final poses (0.671 with the true poses); "particles within 10 deg" counts the particles whose
direction and in-plane errors are both below 10 degrees. Figures and the per-cone numbers are in the run root
`/vast/palmer/scratch/lederman/ry295/pose_prior_benchmark/cfsc_regularization` (`mild_w100_n100k_nz1/comparison`,
`analysis`, made by `scripts/analyze_cfsc.py` and `scripts/summary_figure.py` there).

### 7.2 What we observe

- **The learned prior gains about half of its gap to the true prior.** The ground-truth FSC area rises from 0.319 to
  0.347, which is 53% of the gap to the true prior's 0.371, and more than the FvML-mixture prior gains (0.326). The
  poses improve with it: the median direction error falls from 48.5 to 40.6 degrees, the in-plane error from 27.5 to
  19.2 degrees, and the share of particles within 10 degrees rises from 33.4% to 37.7% (the true prior: 43.2%). The
  learned direction prior is closer to the truth as well (Jensen-Shannon divergence 0.161 to 0.143 bits).
- **The gain is in the poses, not in the final filter.** The unregularized maps of the final poses gain 0.024 (0.286
  to 0.310), 85% of the refined maps' 0.028; so most of the improvement comes from better alignment during the
  refinement, and little from the cone prior acting on the final map.
- **The true prior gains too, less.** With the true prior the area rises from 0.371 to 0.382 (poses only: 0.329 to
  0.339) and the median direction error from 22.3 to 20.9 degrees. This pair crosses machines; the learned pair shows
  that the cross-machine difference is about 0.0004.
- **The gain lies in the cones away from the rarely seen direction.** Per cone, the final map's ground-truth area
  rises by 23% on average (geometric mean) in the 91 cones farther than 25 degrees from z with the learned prior, and
  by 7% with the true prior; the same runs on two machines differ there by 3% (standard deviation of the log ratio).
  The nine cones within 25 degrees of z, where the half maps share almost no signal beyond shell 22, do not change on
  average (0.99 and 1.05), with a spread of the size of their machine-to-machine noise. The cone along z loses 4% and
  18% of its area.
- **The cFAR falls** (0.050 to 0.036, and 0.046 to 0.033) because the best cones improve and the worst do not; it
  does not indicate a worse map.
- **The renders agree.** In ChimeraX renders of the final maps (`render_volume.py`, every map at the isosurface that
  encloses the ground truth's volume), the cone rule's map resolves features that the control's map merges and that
  both true-prior maps show: a hole in the central band and the helix above it (a bump in the control, a stub with the
  cone rule, a rod with the true prior). The same holds for the poses-only reconstructions low-passed alike at 8 A, so
  it is the poses. The true poses' reconstruction is not smeared along the preferred axis, while every estimated map
  is; in this cell the smearing comes from misaligned particles, not from missing views. Same-seed runs on other
  cards look the same. The images are in the run root's `analysis/renders` (`scripts/render_montage.py`).

### 7.3 Mechanism: the shell FSC and the image size

The cone rule leaves RELION's resolution rule as it is, but that rule reads the FSC of the regularized half maps,
which the cone prior changes. At iteration 3 the half-map FSC of shell 22 (24.7 A) is 0.503 in the control and 0.557
with the cone rule. Split by cone, the equatorial cones (axis farther than 65 degrees from z) go from 0.631 to 0.665,
the cones between 25 and 65 degrees stay at 0.315 and 0.317, and the cones near z go from 0.099 to 0.090. The shell FSC
thus rises more than any group of cones: the harder regularization of the signal-free cones near z lowers their noise
power, which weighs the shell FSC toward the cones that carry signal. The reported resolution follows (25.9 to 23.65 A
at iteration 3), and so does the image size: `current_size` is 114 at iteration 8 with the cone rule and 82 in the
control. The gain above therefore mixes two effects, the cone prior inside each reconstruction and the larger image
size it leads to; a run with the cone prior and the control's `current_size` schedule would separate them.

### 7.4 Cost

Estimating the cone prior takes 0.2 to 0.5 s per M-step while the image size is 56 to 72, 2.0 to 3.4 s at 114 to 132,
and 25 to 27 s for the final pass at the full size 256 (both cFSC runs). It runs on the host and grows as `r_max^3`
times the number of cones: at box 256 the membership tests and sums of `cone_shell_sums` and `voxel_ratio_cubes`, which
compute the membership of every voxel once each, take three quarters of it, and the native binning the rest. The larger
image size makes the iterations slower (770 to 1140 s against 450 to 580 s at HEALPix order 3 on the RTX 8000), and the
cFSC run took 4.4 h against 3.6 h for the control on the same node.

At large boxes we measured the host cost on synthetic packed accumulators (complex64 data, float32 weights) at the full
image size (`day` node, job 62763345; the code before the large-grid change, which refused these grids, for
comparison):

| Box (accumulator) | Accumulators | Statistics: time, memory | Weight: time, memory | Before: statistics, weight memory |
| --- | --- | --- | --- | --- |
| 512 (1027^3) | 12.1 GiB | 238 s, 11.6 GiB | 21 s, 2.4 GiB | 25.7 GiB, 12.1 GiB |
| 800 (1603^3) | 46.1 GiB | 1147 s, 42.3 GiB | 74 s, 8.6 GiB | 97.3 GiB, 46.0 GiB |

Memory is the resident peak above the accumulators; the statistics keep the per-voxel ratio cubes (2.0 and 7.7 GiB)
for the solves, and the weight is one float32 copy of a half's weight. A full-size M-step at box 800 thus needs about as
much host memory again as its accumulators and 19 min of host time; numbered M-steps at smaller image sizes need the
fraction `(current_size / box)^3` of both.

### 7.5 Limitations

One benchmark cell and one seed. The cell has a strong preferred orientation with noise level 1 and offsets; the cone
rule should be tested on other cells of the sweeps (other cap widths, noise levels and particle counts, and an
isotropic cell, where it should change little) and with a second seed before it is relied upon.
