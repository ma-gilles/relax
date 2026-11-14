# Branch-and-bound global pose search (`--pose_search bnb`)

Record of the branch-and-bound pose search on branch `dev/bnb_pose_search` (from `main` at `0d9565a6`, with the
offset study's `--offset_prior_units consistent` commit). It replaces pass 1 of the global K=1 searches with a
search over every translation the offset prior allows, coarse to fine over HEALPix orders, after the
branch-and-bound of cryoSPARC (Punjani et al. 2017, Nature Methods 14:290, Supplementary Note 2). The default
reproduces RELION 5.0.1; the option is opt-in.

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
4. Each image's window centre is the whole pixel whose RELION window holds the most posterior mass of the offset
   (the mode of a sharp posterior, near the mean of a broad one). It becomes the RELION pre-shift: the offset prior
   centres, the reported offsets (`relax/refinement/particle_poses.py`) and the change metrics follow it.
5. The RELION window widens, when needed, to the radius that holds 99% of the offset posterior of 75% of the
   particles (`--bnb_window_quantile`), at most four steps per radius.
6. The window significance (`window_significance`) gives the resident pass 2 each image's 99.9% supports over its
   kept rotations and the window around its centre; pass 2 is unchanged.

## Design measurements

All on the 8c9c offset set (box 256, 2.125 A per pixel, 50k particles, 15 A offsets per axis), iteration 1 from the
ground truth low-passed at 30 A, against the true poses.

| Variant | Within 5 degrees | Offset error > 5 px | Search, s per half (GPU) |
| --- | ---: | ---: | --- |
| Level 0 at order 1 (coarse size 22), order 2 | 21% | 35% | 97 / 46 (V100) |
| Dense order 2 (no coarser level) | 49% | 5.7% | 31 / 30 (V100) |
| Order 2 at half its size (20), then 42 | 27% | 29% | 48 / 38 (V100) |
| Covering window (`--offset_range_sigmas 3`, offset study) | 42% | 12% | (pass 1) |
| Dense order 3, global translations | 67.4% | 4.3% | 171 / 165 |
| Order 2 then 3, nearest-parent children, no minimum | 65.3% | 4.8% | 124 / 71 |
| Order 2 then 3, overlapping children, 16 kept at least | 67.3% | 4.4% | 81 / 76 |
| The same, order-3 level on a disk around each centre | 67.0% | 5.1% | 35 / 32 (RTX 3090) |

Below RELION's order-2 coarse size the posterior is confidently wrong at high SNR, so the search never prunes
there (`--bnb_min_order 2`, no half-band level). At noise 1 (box-128 set) recentring alone gives the windowed
search's accuracy and the window width gives the covering window's.

## Results

To be filled: the four benchmark sets of the offset study, each with RELION's search (`priorfix`) on the same
branch and GPU model, and the offset study's runs for reference.

## Not covered

- Class3D, subtomograms, symmetry other than C1, several image shapes and runs on several GPUs: the option refuses
  them.
- The local searches (HEALPix order 4 and up) keep RELION's windows around the offsets the global searches left.
- The test tiers: GPU runs are Slurm jobs on Grace, not `pixi run test-smoke` (fixtures are on Della).
