# PPCA ab initio: current best recipe, v1 (2026-11-05)

Status (2026-11-06): **the exhaustive translation grid is rejected for SGD; the translation choice is open (v1b, relrfine, being scored). Oversampling from HEALPix 3 passes and is a candidate addition.**

Suite v3.1, seed 11, decoded score (distinctive protein for D), next to the same SGD with the old ±2 px grid. References: mean only / true model.

| set | v1, exhaustive grid | old ±2 px grid | mean only / true |
|---|---|---|---|
| C n1 | .918 | .914 | .837 / .992 |
| C n3 | .832 | .897 | .837 / .979 |
| H1 n3 | .979 | .981 | 1.000 / – |
| Ribosembly n3 | .164 (no poses) | .783 | .780 / .932 |
| T10 n1 | .407 | | .382 / .996 |
| D10 / D20 / D40 | .251 / .618 / .625 | seed 12: .409 / .952 / .961 | .669 / .849 / .944 |
| C_shift n1 | .814 | VDAM relrfine .874 | .837 / – |

Oversampling (rotations only, 3.75° from update 1626), SGD with the old grid, suite v3.1 at seed 11:
- C n1: .932 against .914 (BP .939 against .928).
- C n3: .912 against .897 (BP .912 against .906).
- It passes the predeclared rule (decoded +.01 on both sets, nothing down by more than .01). It is a candidate for v2, from one seed.

## 1. Code

- relax: public relax at `relax_pin/RELAX_COMMIT` plus `relax_pin/patches/`, the same tree as branch
  ws2-gated-sgd at ac8e62db. Build with `relax_pin/setup.sh` and `build_natives.sh`, and run with `relax_pin/env.sh`.
- Driver (outside relax; `--source` points at the relax tree): `coupled_arm.py`, sha256 67233a3c10215d16850af409401d9bd484eb59768d3067221a810fa80b1a36af.
  It merges the round-17 SGD driver (`coupled_arm_sgd.py`, 944c3de4) with the shift-study additions (`coupled_arm_shifts.py`, 9fa5682e).
  - Check 1: with no shift flags, after 25 updates of C n1 it matches the SGD driver to theta rel L2 1.2e-5. Two runs of the SGD driver differ by 1.3e-5, so this is float noise.
  - Check 2: with the exhaustive-grid flags it logs the same 49-offset ±48 Å window and σ = 20 Å as the shift driver, and matches it to theta rel L2 6.9e-6.
  - The two source drivers are kept for provenance.
- Memory: one exhaustive-grid arm per A100. relax sizes its HEALPix-3 tiles to free GPU memory, and three arms on one 80 GB GPU ran out of memory.
- `vdam_step_3000.npy`: the learning-rate schedule, the same for every data set at 3000 updates.
- `example_arm_bestv1.json`: one full arm (C n1), in the queue format coupled_queue.py reads.

## 2. The recipe

Units: Å are the definition. Pixels are given for box 64 at 6 Å/px. Convert with your pixel size and round to whole
pixels.

**Optimizer: plain SGD with an H⁻¹ direction and a PC gate.**
- `relax: --optimizer momentum_sgd --sgd-learning-rate <lr> --oversampling 0` (momentum 0).
- `driver: --uniform-direction-prior --sgd-curv split --sgd-dir coupled --adaptive-eta-pc-only --adaptive-eta-pc-floor 0.25 --lr-schedule file:vdam_step_3000.npy`
- **Direction:** per Fourier voxel, H_f⁻¹ G_f. H_f is the (1+q)×(1+q) Gauss-Newton curvature of the batch
  log-likelihood in that voxel's mean and PC values. So this is a damped per-voxel Newton (EM) step, full within a
  voxel and with no coupling between voxels.
- **Step size:** lr × the VDAM schedule. The schedule is .54 flat to about update 1800, then decays: .23 at 2000,
  .08 at 2500 and .04 at 3000. A factor k, fixed at update 1, matches the first step's size to the split-curvature
  step.
- **lr:** matched per data set at update 1 by `match_lr.py` (MATCH:1.95), giving .66-.73 on our sets.
- **PC gate:** the mean is ungated. The PC step is multiplied, per shell, by ρ = clip(EMA over 20 updates of the
  correlation between the two half-batch PC gradients, .25, 1). There is no shrink term.
- **Prior:** the orientation prior is kept uniform. There is no momentum, moving average or fudge.

**Sampling (box 64):**
- `--stages` with HEALPix order 1 until update 125, order 2 until 1625, then order 3 (7.5°).
- Band radius +1 shell every 125 updates, from shell 11 to shell 31 at update 2500.
- 3000 updates of 300 images, q = 2 to 10.

**Translations: the exhaustive absolute grid.**
- `driver: --offset-sigma-hold 20 --shift-sched '[[1, 8, 2], [1626, 8, 1]]'`, with no `--recentre`.
- A zero-centred window of ±48 Å (±8 px), searched in full at every update. Steps are 12 Å (2 px), then 6 Å (1 px)
  from HEALPix order 3.
- The Gaussian prior is centred at 0 with σ held at 20 Å. No per-particle offsets are stored.
- Cost: about 1.8× the training time of relrfine (4.95 s against 2.56 s per update at order 3 on an A100).

## 3. Why each choice (evidence in BEST_PRACTICE.md)

| choice | evidence |
|---|---|
| H⁻¹ direction | diagonal SGD fails C at 2 seeds (decoded .882-.889 against .914 / .897); scalar fails outright |
| PC gate, floor .25 | no PC control → PC runaway (C, T10); gate off after 500 → crash on T10; floor .05 → PCs too small |
| uniform orientation prior | learned prior collapses on the b150 sets (no solvent correction) |
| exhaustive translation grid | only rule that poses shifted Ribosembly (6.3° against relrfine's 108°); equals relrfine on shifted C n3 (suite .834 / .833) |
| no noise anneal | rescues some D seeds but costs Ribosembly 14 points and T10 one protein |

## 4. Known limits

- D10 and D20 are seed-sensitive: the distinctive protein poses at 1 of 3 and 2 of 3 seeds. A fresh-seed test of
  floor .5 against .25 is running.
- T10 from scratch poses 3-5 of 10 proteins. The pseudo-symmetric proteins fail.
- Ribosembly: the decoded maps are at the mean-only level (.783 against .780), so no real heterogeneity is recovered.
- The SGD latents carry more pose-error information than VDAM's (II z→E .54 against .82 on C n1).
- At box 64, oversampling should gain at most about .01-.02 on back-projected maps; a test is running. At box 128,
  order 3 undersamples shells 32-63.
- Never run on shifted D, H1 or T10, or on real data (validation running).

## Driver caveats (2026-11-06)

- **Gate state on resume:** the PC gate's moving averages live only in the driver process and are not saved in checkpoints. A resumed run restarts them. In one test this moved theta by 3% (rel L2 .031) over 500 updates; two fresh reruns agree to 1.4e-5. Run uninterrupted, or label resumed runs "ρ_pc EMA reset at N". Fix to do: save the gate state with each checkpoint.
- **Update cap:** the driver's `--iterations` (default 6000) caps every run silently. For runs longer than 6000 updates, pass `--iterations <stop>`.

## 5. Validation (fills in as results land)

Synthetic, seed 11, suite v3.1, each row next to the mean-only and true-model references and the old ±2 px row: pending.
Real data (one seed, SGD part of the recipe, exhaustive translation grid; descriptive, no ground truth; pose is a diagnostic against the published refinement):
- **EMPIAR-10076** (131,899 particles, recipe as written): mean-map FSC .80 to the RELION map; 49% of poses within 10°. 10 structured PCs; PC1 is an occupancy-like lobe whose tail is the authors' unassigned particles. In the all-particle UMAP, junk and assembly families separate into lobes, and the pose disagreements concentrate in the junk lobe and the class-1 lobe.
- **EMPIAR-10180** (327,490 particles, consensus-centred, i.e. uses consensus translations not rotations, 6600 updates): mean-map FSC .96 to the deposited map, above 0.143 to Nyquist (17 Å); 86% of poses within 10°. 10 structured PCs, including a mobile peripheral lobe.
- **EMPIAR-11846** (286,801 dirty picks): **correction 2026-11-06.** Our first reference, a back-projection at cryoDRGN-AI's published poses, is ~1.5× too large and does not match the deposited model PDB 8EKI; likely a pose-convention mismatch in our back-projection. Against a map made from 8EKI (FSC, shells 1-31): relax homogeneous ab initio on the pre-centred selection .82, which recovers the complex; our SGD PPCA means .26-.29, partial; relax ab initio uncentred .21-.25. The earlier "collapse" readings and the box-edge explanation were based on the faulty reference.

