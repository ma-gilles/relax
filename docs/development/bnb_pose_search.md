# Branch-and-bound global pose search (`--pose_search bnb`)

Record of the branch-and-bound pose search on branch `dev/global_search` (from `main` at `0d9565a6`, with the
offset study's `--offset_prior_units consistent` commit). It replaces pass 1 of the global K=1 searches with a
search over every translation the offset prior allows, coarse to fine over HEALPix orders, after the
branch-and-bound of cryoSPARC (Punjani et al. 2017, Nature Methods 14:290, Supplementary Note 2). The default
reproduces RELION 5.0.1; the option is opt-in. `--bnb_level0 fft` replaces level 0 with a search over every shift
of the box by FFT (last section).

## Why

The offset study (`docs/development/offset_search_benchmark.md` on branch `dev/offset_quadrature`) found two
failures of RELION's offset marginalization on uncentred particles:

1. RELION scores each particle's translations in a window around its rounded previous offset, and narrows the
   window from the last offset changes. A particle whose true offset lies outside the window cannot reach it, and
   its pose goes wrong with it: on the 8c9c set (15 A offsets per axis), RELION loses 22-26% of the particles.
2. A window wide enough to cover the prior (`--offset_range_sigmas 3`) recovers them but coarsens the translation
   step to a quarter of its radius, and costs a stale first order-3 iteration: RELION's pass 1 of the first
   iteration of an order scores at the previous order's coarse size (24 instead of 46 on the box-128 set), and its
   99.9% significance then keeps 146k parents per image, two thirds of the run's time.

## The search

For each image of a half, at the iteration's HEALPix order `n` (`relax/scoring/bnb_search.py`, per-half adapter
`relax/scoring/bnb_half.py`):

1. Level 0 scores every rotation of order `max(2, n - 1)` against every translation on a lattice within
   `--bnb_translation_sigmas` (4) widths of the offset prior around the origin, at RELION's coarse size of that
   order. The scores are pass 1's Gaussian scores on RELION's exact coarse operands and projector (a pass-1 plan of
   the level's window with one zero translation; the search translates the operands itself), with the direction
   prior and the offset prior in consistent units.
2. Each image keeps the rotations carrying `--bnb_mass_fraction` (0.9999) of its translation-marginal posterior,
   those within `--bnb_margin` (10) nats of its best, and its `--bnb_min_kept` (16) best.
3. The next level scores the children of the kept rotations at the next order (the finer rotations within 0.75 of
   the parent's angular step: the cells overlap) at that order's coarse size, on a disk of translations around
   each image's level-0 centre. The last level is the iteration's own coarse grid.
4. The RELION window widens, when needed, to the radius that holds 99% of the offset posterior of half of the
   particles (`--bnb_window_quantile` 0.5), at most the search radius and six steps per radius.
5. Each image's window centre is its previous offset (the origin before any) unless the whole pixel whose window holds
   the most posterior mass of the offset holds `--bnb_recentre_gain` (0.1) more of it; then the window moves there.
   The centre becomes the RELION pre-shift: the offset prior centres, the reported offsets
   (`relax/refinement/particle_poses.py`) and the change metrics follow it.
6. The window significance (`window_significance`) gives the resident pass 2 each image's 99.9% supports over its
   kept rotations and the window around its centre; pass 2 is unchanged.

## Design measurements

All on the 8c9c offset set (box 256, 2.125 A per pixel, 50k particles, 15 A offsets per axis), iteration 1 from the
ground truth low-passed at 30 A, against the true poses.

| Variant | Within 5 degrees | Offset error > 5 px | Search, s per half | GPU |
| --- | ---: | ---: | --- | --- |
| RELION's search (`priorfix`, offset study) | 19.9% | 51.8% | (pass 1) | A100 |
| Covering window (`--offset_range_sigmas 3`, offset study) | 42.0% | 11.6% | (pass 1) | A100 |
| Level 0 at order 1 (coarse size 22), then order 2 | 20.4% | 35.0% | 97 / 46 | V100 |
| Dense order 2 (no coarser level) | 48.8% | 5.7% | 31 / 30 | V100 |
| Order 2 at half its size (20), then at 42 | 26.9% | 28.7% | 48 / 38 | RTX 3090 |

Iteration 1 at HEALPix order 3 (`--healpix_order 3`), against a dense order-3 search over the global translations:

| Variant | Within 5 degrees | Offset error > 5 px | Directions differing from the dense search by > 5 degrees | Search, s per half | GPU |
| --- | ---: | ---: | ---: | --- | --- |
| Dense order 3 | 67.4% | 4.3% | | 171 / 165 | V100 |
| Order 2 then 3, nearest-parent children, no minimum kept | 65.3% | 4.8% | 7.2% | 124 / 71 | RTX 3090 |
| Order 2 then 3, overlapping children, 16 kept at least | 67.3% | 4.4% | 0.7% | 81 / 76 | V100 |
| The same, order-3 level on a disk around each centre | 67.0% | 5.1% | 4.5% | 35 / 32 | RTX 3090 |

Below RELION's order-2 coarse size the posterior is confidently wrong at high SNR, so the search never prunes
there (`--bnb_min_order 2`, no half-band level). At noise 1 (box-128 set) recentring alone gives the windowed
search's accuracy and the window width gives the covering window's; recentring every window on a broad posterior
moved the particles by noise at order 3 (mean direction change 38-43 degrees per iteration against 19-37 for the
covering window, final GT FSC AUC 0.607 against 0.680), hence the recentring gain of step 5, decided at the
widened window's radius (at RELION's radius 78% of the windows still moved).

The window quantile, whole runs on the box-128 noise-1 set (below):

| `--bnb_window_quantile` | Map FSC AUC | Order-3 iterations | Offset change at order 3, A per iteration | Wall (s) | GPU |
| --- | ---: | ---: | --- | ---: | --- |
| 0 (RELION's radius, recentred) | 0.672 | 4 | 5.5, 4.9, 4.8, 4.8 | 2221 | RTX 8000 |
| 0.25 | 0.681 | 4 | 5.2, 4.7, 4.8, 4.8 | 3231 | V100-SXM2 |
| 0.5 | 0.740 | 8 | 12.1, 10.4, 9.9, 9.0, 8.7, 8.2, 7.8, 8.0 | 4995 | RTX 8000 |
| 0.5, seed 2 | 0.739 | 9 | 12.3, 10.4, 9.9, 9.2, 8.8, 8.2, 8.3, 7.9, 7.9 | 5287 | RTX 8000 |
| 0.5, after the batching | 0.732 | 7 | 12.1, 10.4, 9.9, 9.0, 8.7, 8.1, 8.0 | 4887 | RTX 8000 |
| 0.75 | 0.686 | 3 | 14.2, 11.9, 12.3 | 5182 | RTX 8000 |

Widening at order 2 recovers particles (directions within 5 degrees after the order-2 iterations: 20% at 0, 26% at
0.5). At order 3 the default's windows (6-9 px radius, 1.1-1.5 px step) let the offsets keep settling: their changes
and the direction changes shrink by more than 3% every iteration, so RELION's convergence rule
(`nr_iter_wo_large_hidden_variable_changes`) holds the global order-3 search for 7-9 iterations, and every one of
them moves more particles to the right pose (direction error > 5 degrees with offset error > 3 px: 42% -> 27%). The
other quantiles leave order 3 after 3-4 iterations while it still improves; the local searches that follow do not
recover a misplaced particle. The run after the batching left order 3 one iteration earlier (its changes shrank by
2.3%) and ends at 0.732; the spread of the three runs at 0.5 follows their order-3 iteration counts. At high SNR the
quantile changes nothing (box-128 noise 0.25: AUC 0.952 at 0.5, 0.953 at 0.75).

The search's own cost, one order-3 iteration of EMPIAR-10076 (box 128, 131,899 particles) continued from
iteration 5, V100-PCIE-32GB, per half:

| Batching | Search, s | Level 0 / level 1 scores, s | Iteration, s |
| --- | ---: | --- | ---: |
| 32 images in dataset order | 210 / 173 | 26 / 114, 26 / 103 | 468 |
| 32 images in order of their parent counts | 78 / 77 | 24 / 18, 24 / 16 | 236 |
| The same, 128 / 64 images (sized by grid and window) | 65 / 56 | 19 / 17, 19 / 14 | 199 |

A batch scores as many candidates as its widest image, and a few images with broad posteriors keep the 4096-rotation
cap; in dataset order they set the width of most batches. The resolution and `ave_Pmax` of the iteration are the
same (8.22 A, 0.7740 / 0.7739).

## Results

`evaluate.py` of the offset study: the map FSC AUC against the true map after alignment (up to Nyquist), the offset
and direction errors of the final poses against the true poses, the number of iterations and the wall time of the
whole `relax refine` run (global search from zero offsets and the low-passed initial model). Times compare only
within a GPU model. RELION's search is the offset study's `priorfix` (`--offset_prior_units consistent`, otherwise
the defaults); BnB adds `--pose_search bnb`.

5nrl box 128, noise 1 (`off15_b128_n20k`: 20k particles, 4.25 A per pixel, 15 A offsets per axis):

| Search | Map FSC AUC | Offset error median (px) | < 0.5 px | > 3 px | Direction < 5 deg | Iterations | Wall (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RELION | 0.617 | 2.20 | 30% | 47% | 41% | 15 | 3418 | A100-PCIE-40GB |
| BnB, window quantile 0.75 | 0.686 | 0.66 | 42% | 37% | 53% | 13 | 3964 | A100-PCIE-40GB |
| RELION | 0.618 | 2.18 | 29% | 47% | 41% | 15 | 3963 | RTX 8000 |
| BnB | 0.732 | 0.47 | 53% | 28% | 63% | 16 | 4887 | RTX 8000 |
| BnB before the batching (seed 1 / 2) | 0.740 / 0.739 | 0.45 / 0.46 | 54% | 27% | 65% | 17 / 18 | 4995 / 5287 | RTX 8000 |
| Covering window (offset study `win3_priorfix`) | 0.680 | 0.71 | 40% | 37% | 51% | 13 | 3212 | A100-SXM4-80GB |
| RELION from the true poses (offset study `oracle_priorfix`) | 0.702 | 0.59 | 44% | 34% | 56% | 17 | 2513 | A100-SXM4-80GB |

The same at noise 0.25 (`off15_b128_n20k_snr4`, four times the SNR):

| Search | Map FSC AUC | Offset error median (px) | < 0.5 px | > 3 px | Direction < 5 deg | Iterations | Wall (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RELION | 0.953 | 0.10 | 95% | 5% | 95% | 21 | 1420 | A100-PCIE-40GB |
| BnB, window quantile 0.75 | 0.953 | 0.11 | 98% | 2% | 98% | 16 | 1300 | A100-PCIE-40GB |
| BnB before the batching | 0.952 | 0.10 | 98% | 2% | 98% | 17 | 1522 | V100-PCIE-32GB |

8c9c, box 256 (`off15_8c9c_b256_n50k`: 50k particles, 2.125 A per pixel, 15 A offsets per axis):

| Search | Map FSC AUC | Offset error median (px) | < 0.5 px | > 3 px | Direction < 5 deg | Iterations | Wall (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RELION | 0.849 | 0.14 | 77% | 23% | 77% | 28 | 4260 | V100-SXM2-32GB |
| BnB before the batching | 0.861 | 0.18 | 97% | 2% | 98% | 18 | 3180 | V100-SXM2-32GB |
| RELION | 0.852 | 0.17 | 77% | 22% | 78% | 34 | 3234 | A100-SXM4-80GB |
| BnB | 0.861 | 0.18 | 97% | 2% | 98% | 18 | 2065 | A100-SXM4-80GB |
| BnB, `--bnb_search_precision tf32` | 0.861 | 0.18 | 97% | 2% | 98% | 18 | 1971 | A100-SXM4-80GB |

8c9c with 30 A offsets per axis (`off30_8c9c_b256_n50k`, 14.1 px):

| Search | Map FSC AUC | Offset error median (px) | < 0.5 px | > 3 px | Direction < 5 deg | Iterations | Wall (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RELION's window at 15 px, step 3 (`--offset_range 15 --offset_step 3`) | 0.862 | 0.18 | 98% | 2% | 98% | 24 | 3447 | V100-SXM2-32GB |
| BnB before the batching | 0.865 | 0.16 | 98% | 2% | 98% | 17 | 4237 | V100-SXM2-32GB |
| BnB | 0.864 | 0.18 | 97% | 2% | 98% | 18 | 4195 | V100-SXM2-32GB |
| RELION (offset study) | 0.752 | 11.91 | 2% | 65% | 36% | 23 | 1992 | A100-SXM4-80GB |
| Covering window (offset study `win3_priorfix`) | 0.454 | 8.86 | 0% | 94% | 10% | 23 | 1485 | A100-SXM4-80GB |
| BnB | 0.865 | 0.16 | 98% | 2% | 98% | 17 | 3097 | A100-SXM4-80GB |
| RELION's window at 15 px, step 3 (offset study) | 0.862 | 0.18 | 98% | 2% | 98% | 24 | 2804 | A100-PCIE-40GB |

EMPIAR-10076, box 128 (`empiar10076_b128`: 131,899 particles, 3.275 A per pixel; the reference is the consensus, the
published poses and their map):

| Search | Map FSC AUC | Offset error median (px) | < 0.5 px | > 3 px | Direction < 5 deg | Iterations | Wall (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RELION | 0.960 | 0.22 | 72% | 16% | 77% | 21 | 2108 | A100-SXM4-80GB |
| BnB | 0.962 | 0.16 | 72% | 17% | 76% | 16 | 2126 | A100-SXM4-80GB |
| BnB before the batching | 0.962 | 0.16 | 72% | 17% | 76% | 16 | 4065 | V100-SXM2-32GB |

Runtime against RELION's search on the same GPU:

- 8c9c: 25% faster on a V100 (3180 against 4260 s) and 36% faster on an A100 (2065 against 3234 s). An order-2
  iteration costs more (64-96 against 21-28 s on the A100), but RELION runs 13 of them before its order-3 search and
  BnB 4.
- EMPIAR-10076: equal on an A100 (2126 against 2108 s, 16 against 21 iterations).
- Box 128, noise 1: 23% slower on an RTX 8000 (4887 against 3963 s). The posteriors are broad, so the wider windows
  send more samples to pass 2 (an order-2 iteration takes 476-536 s against 141-304 s), and BnB's order-3
  iterations take 306-649 s against RELION's 72-90 s after its first one (2437 s).
- 30 A offsets: 22% slower than RELION with the window set by hand on a V100 (4195 against 3447 s). BnB's order-2
  iterations score the global grid (about 2500 translations, 94 s of search per half on an A100) where the hand-set
  window scores 81. On an A100 BnB takes 3097 s against 1992 s of RELION's default window, which ends at 0.752.
- `--bnb_search_precision tf32` (A100) leaves every 8c9c number unchanged and shortens the run by 5% (level-0
  scores at order 2: 16.7 -> 4.5 s per half); the search is a small part of an 8c9c iteration. It stays opt-in.

In short, on uncentred data BnB reaches the accuracy of a window that covers the offsets without that window being
set (30 A offsets: 0.864 against 0.752, and 0.862 with the window set by hand), leaves 2% of the 8c9c particles
more than 3 px off where RELION leaves 22-23% (65% at 30 A), and at noise 1 beats RELION started from the true poses
(0.732-0.740 against 0.702) and leaves 28% of the particles more than 3 px off against RELION's 47%. On EMPIAR-10076
and at high SNR it equals RELION.

## FFT level 0 (`--bnb_level0 fft`)

Level 0 above searches the lattice within 4 widths of the offset prior (`sigma` 10 A at the start). On particles
picked much farther off centre than that, does the prior keep the search from opening?

### The prior opens; the mask does not move

8c9c with 60 A offsets per axis (`off60_8c9c_b256_n50k`: true radius median 71 A, 90% 129 A, 25% beyond 100 A), at
noise 0.2 (as the 15 A set) and 0.5, with the lattice level 0 from `sigma` 10 A (the default) and 60 A. A particle is
found when its offset error is below 3 px (after the frame shift) and its direction error below 10 degrees.

| Start `sigma` | Noise | Found after iteration 1 | Found at the end | Map FSC AUC | > 3 px | Wall (s) | GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 10 A | 0.2 | 20% | 82% | 0.853 | 18% | 4398 | A100-SXM4-80GB |
| 60 A | 0.2 | 64% | 82% | 0.854 | 18% | 5126 | A100-SXM4-80GB |
| 10 A | 0.5 | 11% | 73% | 0.750 | 27% | 4155 | A100-SXM4-80GB |
| 60 A | 0.5 | 30% | 73% | 0.751 | 27% | 5120 | A100-SXM4-80GB |

The prior does cut the far particles in iterations 1-2, but RELION's `sigma` update opens it (10 -> 22 -> 44 -> 49 A)
by iteration 3, and starting at 60 A ends at the same place. What stops the rest is the mask: the lattice level 0
scores the images unshifted, so the 240 A mask stays at the box centre and cuts a particle more than about 100 A out,
while RELION (and BnB's finer levels and window) shift each image by its rounded previous offset before masking. At
the end (noise 0.2) the share found is 99% within 80 A, 95% at 80-100 A, 64% at 100-120 A, 8% at 120-160 A and none
beyond, whatever the start.

### The search

`relax/scoring/bnb_fft.py`, used for level 0 by `--bnb_level0 fft`; the finer levels, the window and pass 2 are
unchanged.

1. Of the score `X - A/2 - C/2`, only `X(t) = Re sum_k G(k) exp(2 pi i k.t / N)`, `G = w conj(y) p_r`, depends on the
   shift. One inverse real FFT of `G` on an `M x M` grid (`M = N / step`) gives it at every circular shift of the box
   on a lattice of `step` pixels: half RELION's coarse Nyquist spacing, whole pixels dividing the box, at most
   `--bnb_max_step_px` (2 px on both boxes here). It equals the lattice product on every shift (unit test; on 8c9c
   iteration 1 the rotation masses agree to 2e-3 and the best poses all agree).
2. Each image is first shifted to its origin, its rounded previous offset (zero fill), and masked there, as RELION
   does; the lattice is the circular shifts around the origin.
3. The prior over the box is the model's Gaussian, centred on the box centre as RELION's (it does not follow the
   origin), at the full offsets (origin plus shift), untruncated (`--bnb_fft_prior model`, the default), or flat.
4. Rotations are kept by their whole-box translation-marginal posterior as before. Each image's frame is the lattice
   point whose window (the search radius) holds the most posterior mass; the finer levels and the window search
   around it.

The block is memory-bound: 0.585 us per (image, rotation) pair at box 256 (`M` = 128, A100-SXM4), whatever `sigma`;
the gather and inverse FFT alone take 0.417 us, and neither the inverse DFT as two matmuls (0.637 us) nor leaner
reductions (0.570 us) help. The lattice GEMM costs about 0.54 us per pair at 2501 translations and grows with them: on
the 60 A set its level-0 scores take 8 s per half at `sigma` 10 A, 37 s at 22 A and 115 s at 44-50 A, the FFT's 70 s
at every `sigma`.

### Ablations on the 60 A set

| Level 0 | Noise | Found after iteration 1 | Found at the end | Map FSC AUC | > 3 px | Iterations | Wall (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lattice | 0.2 | 20% | 82% | 0.853 | 18% | 17 | 4398 |
| FFT, flat prior, mask at the box centre | 0.2 | 64% | 82% | 0.857 | 18% | 18 | 3907 |
| FFT, flat prior | 0.2 | 64% | 95% | 0.867 | 5% | 20 | 4311 |
| FFT (default) | 0.2 | 58% | 95% | 0.867 | 5% | 20 | 4363 |
| Lattice | 0.5 | 11% | 73% | 0.750 | 27% | 16 | 4155 |
| FFT, flat prior, mask at the box centre | 0.5 | 30% | 73% | 0.753 | 26% | 15 | 3290 |
| FFT, flat prior | 0.5 | 30% | 83% | 0.770 | 17% | 15 | 3148 |
| FFT (default) | 0.5 | 25% | 83% | 0.769 | 17% | 18 | 4237 |

All on A100-SXM4-80GB. Found at the end by true radius:

| True radius (A) | Particles | Lattice, noise 0.2 | FFT, noise 0.2 | Lattice, noise 0.5 | FFT, noise 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-40 | 9,980 | 99% | 98% | 94% | 91% |
| 40-60 | 9,702 | 99% | 98% | 94% | 90% |
| 60-80 | 9,728 | 99% | 98% | 92% | 89% |
| 80-100 | 8,067 | 95% | 98% | 79% | 86% |
| 100-120 | 5,764 | 64% | 96% | 41% | 79% |
| 120-160 | 5,370 | 8% | 81% | 4% | 59% |
| 160-200 | 1,193 | 0% | 45% | 0% | 30% |
| 200-400 | 196 | 0% | 23% | 0% | 15% |

Without the shift to the origin the FFT finds the far particles no better than the lattice; with it they are found
out to 160 A and beyond. Near the centre it finds a few fewer (1 point at noise 0.2, 3-4 at noise 0.5), with either
prior. With a flat prior, noise peaks anywhere in the box win at low SNR: on the box-128 noise-1 set the first
iteration put the offsets at a median 51 A (true 18 A), `sigma` jumped to 49 A and kept growing to 59 A, and the run
ended at AUC 0.402 (the lattice: 0.740); at noise 0.25, 16% of the particles ended more than 3 px off (the lattice:
2%). The model's prior gives the flat prior's recovery on the 60 A set and the lattice's results at low SNR, hence the
default.

### Results

The FFT level 0 (`--bnb_level0 fft`) against the lattice (the default above), same code, data and GPU model per row;
map FSC AUC, share of particles more than 3 px off, wall time.

| Set | Lattice | FFT | FFT, flat prior | GPU |
| --- | --- | --- | --- | --- |
| 60 A offsets, noise 0.2 | 0.853, 18%, 4398 s | 0.867, 5%, 4363 s | 0.867, 5%, 4311 s | A100-SXM4-80GB |
| 60 A offsets, noise 0.5 | 0.750, 27%, 4155 s | 0.769, 17%, 4237 s | 0.770, 17%, 3148 s | A100-SXM4-80GB |
| 30 A offsets | 0.865, 2%, 3097 s | 0.860, 2%, 2741 s | 0.860, 2%, 2726 s | A100-SXM4-80GB |
| 8c9c, 15 A offsets | 0.861, 2%, 2065 s | 0.860, 2%, 2553 s | | A100-SXM4-80GB |
| EMPIAR-10076 | 0.962, 17%, 2126 s | 0.962, 17%, 2575 s | | A100-SXM4-80GB |
| Box 128, noise 0.25 | 0.952, 2%, 1075 s | 0.952, 2%, 1156 s | 0.945, 16%, 1894 s | A100-PCIE-40GB |
| Box 128, noise 1 | 0.740, 27%, 4120 s | 0.731, 28%, 3852 s | 0.402, 83%, 5069 s | A100-PCIE-40GB |

RELION on the same sets (above): 0.752 at 30 A offsets, 0.852 at 15 A, 0.960 on EMPIAR-10076, 0.953 and 0.617 on the
box-128 sets. The box-128 noise-1 runs of the lattice spread over 0.732-0.740 with their order-3 iteration counts
(above); the FFT run left order 3 after 7 iterations against the lattice's 8 here, as the lattice run at 0.732 did.

Run time follows the offset prior. The FFT level costs the same at every `sigma`; the lattice's grows with it:

- 60 A offsets: an order-2 iteration takes 330-355 s against the lattice's 405-465 s once `sigma` has opened, and the
  first iteration 187 s against 81 s. The FFT runs still take about the lattice's time (-1% and +2%): with the far
  particles moving, they spend 7 order-2 iterations against 5 at noise 0.2 and 5 order-3 iterations against 3 at
  noise 0.5.
- 30 A offsets: 11% faster (order-2 iterations from the third 253-254 s against 269-280 s, order 3 261-284 s against
  300-319 s).
- Small offsets: 24% longer on 8c9c at 15 A (order-2 iterations 186-199 s against 64-96 s, order 3 239-263 s against
  146-164 s), 21% on EMPIAR-10076 (order 2: 171-181 s against 107-118 s), 8% on box 128 at noise 0.25; 7% shorter on
  box 128 at noise 1, one order-3 iteration fewer.

In short, the FFT level 0 with the shift to the origin finds the particles picked far off centre that the lattice
level 0 loses to its centred mask (60 A offsets: 82% -> 95% found, AUC 0.853 -> 0.867; at noise 0.5 73% -> 83%,
0.750 -> 0.769), and matches it elsewhere (AUC within 0.005, and 0.731 against 0.740 at noise 1). It costs 8-24% more
time where the offsets are small and saves 11% at 30 A.

A lattice level 0 that shifts and masks each image at its origin (a per-image offset prior) would likely find the
far particles too, since the prior opens on its own; the FFT would then pay only where the lattice's disc is larger
than about 2500 translations at box 256. Not tried.

## Not covered

- Class3D, subtomograms, symmetry other than C1, several image shapes and runs on several GPUs: the option refuses
  them.
- The local searches (HEALPix order 4 and up) keep RELION's windows around the offsets the global searches left.
- The test tiers: GPU runs are Slurm jobs on Grace, not `pixi run test-smoke` (fixtures are on Della).
