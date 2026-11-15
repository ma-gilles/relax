# PPCA initial model: global pose search (`--pose-search global`)

A port of `relax refine`'s global pose search (`docs/development/bnb_pose_search.md`, branch `dev/global_search`)
to `relax ppca_initial_model`. The PPCA E-step (`relax/ppca_refinement`) shares no code with refine's search, so the
port reuses PPCA's own adaptive-oversampling engine instead.

## What it does

The default PPCA search scores every pose of the stage's grid on every update: all rotations of the stage's HEALPix
order times a zero-centred shift window (the study's default "VW" window is +-48 A, 2 px steps to update 1625, then 1 px
at box 64). Its cost grows with the window's area, which is why a wide window costs about three times the narrow one,
and why particles beyond it (EMPIAR-11846, offsets to 80 A) are out of reach.

With `--pose-search global` every update runs two passes on the oversampling engine
(`relax/ppca_refinement/oversampled_stream.py`):

1. Pass 1 scores the HEALPix order hp - 1 grid and the stage's window at twice its shift step, at RELION's coarse image
   size for order hp - 1 (`iteration_loop.pass1_image_size`). Each coarse sample's prior is the log-sum of its
   children's.
2. Each image keeps its significant samples (the adaptive fraction `target_mass`, 0.999, at most `--maxsig`).
3. Pass 2 scores only their children: RELION's order-1 children (the order-hp directions with half psi steps, and four
   shifts a quarter coarse step from their parent), i.e. the stage's order and shift step, at the stage's image size.
   Each child carries its own direction prior and offset prior. The model update reads pass 2's posterior.

The direction prior's mass is kept per child rotation and mapped onto the stage grid's directions, so the learned
direction prior updates as in the dense search. `iteration_loop._global_search_grids` builds the grids (cached per
stage), `_global_search_priors` the priors and `_stage_rotation_mass` the mapping.

Compared with refine's search: one coarse level (refine's default `levels 1` also scores one coarser order), children
of a coarse sample are fixed shifts and rotations (no overlap radius: the engine's jobs need parent-major children), no
recentring (PPCA has no stored offsets: the window is absolute every update), and the cap is RELION's `--maxsig`.

## Checks

`tests/unit/ppca_initial_model/test_global_pose_search.py`: with fraction 1 and no cap, pass 2 is the dense child
grid, and the global E-step's statistics, latents, log-likelihood, offsets and rotation mass equal the dense stream's
over that grid with the same priors (stream level and controller level); the grids cover the stage's directions; the
priors and the mass mapping. The PPCA unit tests pass (`tests/unit/ppca_initial_model`,
`test_ppca_initial_model_momentum_sgd.py`).

## Results

### Setup

The study's datasets and scorer live on Della, which was out of reach from Grace, so the benchmark ran on Grace
look-alikes (scripts and outputs: `/vast/palmer/scratch/lederman/ry295/ppca_global_search`). EMPIAR-10097 is not on
Grace.

- Ribosembly: 100,000 images of the CryoBench Ribosembly stack (SNR 0.01, 16 assembly states, uniform views),
  Fourier-cropped to box 64 (6 A/px), in three shift variants that share images, CTFs and views: `centred` (no shifts),
  `cryobench` (the stack's own shifts: radius median 24 A, max 42 A) and `wide` (plus a 20 A/axis Gaussian: median
  32 A, 19% beyond 48 A, 0.2% beyond 84 A). The truth is the simulation's poses, states and 16 state maps.
- EMPIAR-10076 (real): 131,899 particles at box 64 (6.55 A/px); the reference is a consensus refinement's poses (offsets
  to 48 A) and its reconstruction (one state).

Each pair trains the study's VW recipe (3000 updates, orders 1 to 3, a +-48 A window at 2 px then 1 px steps from
update 1626) with `--pose-search grid` (dense) or `global` (`--maxsig 100` unless noted). Nothing else differs. Every
trained model is scored by the same dense inference (`evaluate_ppca.py`: order 3, 1 px shifts over a window of 12 px,
20 px for `wide` and 10 px for 10076, flat priors, 10,000 particles): the share of particles within 10 deg of the truth
after a global frame fit, the median pose error, the mean map's FSC AUC against the population mean, and the decoded
state maps' macro FSC AUC (each state's mean latent).

### Speed

Training time, same GPU per pair (update seconds summed over 3000 updates; seed 11):

| Set (GPU) | Dense | Global | Ratio | Order 3 s/update, dense / global | Orders 1-2 s, dense / global |
|---|---|---|---|---|---|
| centred (L40S) | 4144 | 948 | 4.4x | 2.75 / 0.29 | 390 / 521 |
| cryobench (L40S) | 4237 | 1184 | 3.6x | 2.73 / 0.27 | 445 / 683 |
| wide (L40S) | 4671 | 1201 | 3.9x | 3.06 / 0.30 | 442 / 684 |
| wide, 84 A window (L40S) | 11506 | 1284 | 9.0x | 8.06 / 0.34 | 619 / 707 |
| cryobench, maxsig 500 (L40S) | | 1652 | | 0.31 | 1110 |
| 10076 (RTX 3090) | see below | 2303 | | 7.86 / 0.65 | 684 / 1214 |
| 10076, maxsig 500 (RTX 3090) | | 4478 | | 1.33 | 2503 |

The dense 10076 run ran out of memory at update 1946 on the RTX 3090 (see Notes) and finished on an L40S, so its total
is not comparable; its order-3 rate on the 3090 (radius 24 to 26 only, where the global median covers 24 to 31) is.
The global search pays at orders 1 and 2, where pass 1 plus the children of up to 100 coarse samples per image cost
more than the small dense grid, and wins 9 to 12x per update at order 3 (24x at an 84 A window). Its cost barely grows
with the window: 84 A instead of 48 A costs it 7% in total, and the dense search 2.5x. Replicates on other GPUs (global
seeds 12 and 13 on RTX 3090s: 1265 to 1283 s; dense cryobench seed 12 on an A100: 4391 s) are not in the table.

One E-step on the same checkpoint and 600 images (`compare_estep.py`, V100, dense-trained checkpoints):

| Checkpoint | Dense s | maxsig 100 | maxsig 500 | Capped (100) | Top pose within 10 deg of dense (100) |
|---|---|---|---|---|---|
| cryobench, update 1000 (order 2) | 0.77 | 1.1x | 0.4x | 65% | 0.50 |
| cryobench, 1700 / 2500 / 3000 (order 3) | 18 / 29 / 29 | 36 / 56 / 56x | 28 / 46 / 45x | 3-5% | 0.93-0.95 |
| wide, 1000 (order 2) | 0.94 | 1.3x | 0.5x | 69% | 0.43 |
| wide, 1700 / 2500 / 3000 | 22 / 31 / 30 | 36 / 51 / 51x | 24 / 38 / 38x | 5-10% | 0.84-0.86 |
| 10076, 1700 / 2500 / 3000 | 28 / 44 / 44 | 22 / 31 / 31x | 7 / 10 / 10x | 45-54% | 0.81-0.85 |

At order 2 the posterior is broad: two thirds of the images hit the cap and the top poses agree with dense for only half
of them. At order 3 on Ribosembly a median image keeps 2 coarse samples (significant mass 0.9999). On 10076 half the
images hit the cap (mass p5 0.57 to 0.65 at maxsig 100, 0.76 to 0.82 at 500).

### Quality

Common evaluation of the trained models (within 10 deg / pose median deg / mean-map AUC / decoded-state AUC):

| Set | Dense | Global |
|---|---|---|
| centred, seed 11 | .997 / 3.90 / .863 / .829 | .626 / 5.04 / .608 / .555 (split frame) |
| centred, seed 12 | .998 / 3.89 / .865 / .828 | .997 / 3.98 / .843 / .811 |
| centred, seed 13 | .628 / 4.77 / .577 / .389 (split frame) | .996 / 3.98 / .846 / .812 |
| cryobench, seed 11 | .997 / 3.93 / .860 / .824 | .997 / 4.00 / .842 / .807 |
| cryobench, seed 12 | .997 / 3.97 / .861 / .825 | .998 / 3.97 / .838 / .805 |
| cryobench, maxsig 500 | | .997 / 3.95 / .843 / .808 |
| wide | .998 / 3.92 / .824 / .813 | .996 / 4.02 / .798 / .788 |
| wide, 84 A window | .998 / 3.95 / .863 / .826 | .997 / 3.99 / .843 / .807 |
| 10076 | .782 / 4.48 / .922 / .935 | .779 / 4.51 / .908 / .925 |
| 10076, maxsig 500 | | .780 / 4.55 / .912 / .926 |

- Poses: the global-trained models place particles as well as the dense ones on every set (within 10 deg differs by at
  most 0.003; on 10076 both fall from 0.84 below 12 A of offset to 0.55 at 36 to 48 A).
- Maps: at matched settings the global-trained maps are lower by 0.014 to 0.026 in mean-map AUC and 0.010 to 0.025 in
  decoded AUC. The good seeds of each arm sit within 0.004 of each other (`centred`: dense .863/.865, global
  .843/.846; `cryobench`: dense .860/.861, global .842/.838), so the gap is systematic, not seed noise. A cap of 500
  does not close it (cryobench .843, 10076 .912).
- The failed seeds (global seed 11 and dense seed 13 on `centred`) are the same optimum: states 0 to 4 and most of
  state 7 (37% of particles) sit in a second rigid frame (98% within 10 deg once that frame is fitted separately; 72 deg
  from the main frame in the global run, the other hand in the dense run). Each arm fails one seed in three, which
  matches the study's warning that one seed is not enough.
- The wide window: training with an 84 A window instead of 48 A raises both arms' maps on `wide` by about 0.04 (mean
  map: dense .824 to .863, global .798 to .843) with the same poses and offsets in the common evaluation, presumably
  because 19% of the particles lie beyond 48 A and train off-centre in the narrow window (not tested further). The
  gap between the arms stays (0.020 mean map, 0.019 decoded).

### Where the global E-step differs from the dense one

`compare_estep.py --child-dense` also runs the dense scorer over the global search's child grid (its rotations, shifts
and priors), which separates the child grid's half-step offset from the stage grid from the significance cut (final
checkpoints, order 3, maxsig 100):

| Checkpoint | Child-dense vs dense: log-evidence median (p5, p95) / latent rel. diff | Global vs child-dense: log-evidence median (p5) / latent rel. diff / top pose within 10 deg |
|---|---|---|
| cryobench, dense-trained | -1.1 (-30, +26) / 0.99 | 0.0 (-42) / 0.0007 / 0.94 |
| wide, dense-trained | -0.9 (-29, +22) / 0.88 | 0.0 (-41) / 0.0067 / 0.89 |
| cryobench, global-trained | +1.7 (-25, +25) / 0.96 | 0.0 (-36) / 0.0006 / 0.94 |
| wide, global-trained | +0.6 (-27, +24) / 0.86 | 0.0 (-35) / 0.0033 / 0.88 |

- The grid offset explains the latent difference between the searches (about 1 in relative norm: at SNR 0.01 a
  single image's latent mean follows its top pose), and the log-evidence sign follows the grid the model was trained on.
- Against its own grid's dense scorer, the global search agrees for most images (median 0, latents within 0.7%), but
  the top pose of 6 to 12% of images moves by more than 10 deg (log-evidence p5 -35 to -42) although each image keeps
  0.999 of pass 1's mass. Pass 1 ranks coarse cells by their centre pose, at a coarser image size, so a cell whose
  centre is poor but whose child is best can be dropped. Pass 1 at the stage's full image size does not help (within
  10 deg 0.91 and 0.86 against 0.94 and 0.89), so the image size is not the cause. The cell-centre ranking is the
  likely one (not tested directly).

### Notes

- The dense path's memory planner has too little margin: the dense 10076 run ran out of memory on a 24 GB RTX 3090 at
  update 1946 (radius 26; planned 18.64 of 18.87 GiB, then a 503 MiB allocation failed), and the dense evaluation at a
  20 px window ran out of memory with the default tiles (all `wide` evaluations use `--image-batch-size 8`). Both are
  pre-existing; the global training runs did not hit them.
- GPU types differ between runs (the Grace queue placed them): timings compare only within a pair on the same GPU.
- recovar's native preflight needs an exact SASS match, so the native pack has to include sm_89 for L40S nodes.

### Next

- Neighbour expansion: also score the children of each significant coarse cell's neighbours (as PPCA's tilt coarse pass
  does with `_coarse_neighbors`, or refine's overlap radius). This targets the 6 to 12% misses, which are the leading
  (untested) explanation of the map gap, since the cap is not.
- Orders 1 and 2: use the dense search there (while the posterior is broad the global search is slower and its top
  pose differs from the dense one for half the images) and switch at order 3, where all the gain is.
- The study's own datasets and scorer on Della, and seeds 12 and 13 on the D sets, before any claim beyond these
  look-alikes.
