# Prompt: run the current best PPCA ab-initio recipe on a real cryo-ET dataset

> **Update 2026-11-12 (owner decision): the default recipe for real data is now VDAM-simple + the wide held-prior
> translation window ("VW"), not the SGD recipe v1 below.** See `best_v2/RECIPE_v2.md`, section DEFAULT, and
> `best_v2/example_arm_default_vw.json` (driver `best_v2/drivers/coupled_arm_fft2_ablate.py`). Settings: VDAM optimizer with
> `--vdam-ko noadapt`, q 10, 3000 updates, a fixed window centred on the box centre of about ±45-50 Å (coarse 2-px steps, then
> 1 px from HEALPix order 3 at update 1626), shift prior held at 20 Å. For subtomograms, widen the window if your measured
> per-particle shift spread exceeds it, as below. Keep the SGD recipe v1 only as a comparison arm.


You are working for Marc A. Gilles on PPCA ab-initio reconstruction with relax (github.com/ma-gilles/relax, public)
and RECOVAR (public). The single-particle team has settled a "current best" recipe (v1). Your job is to run it on
real cryo-ET (tilt-series) data on your cluster, implementing what's missing for tilt data. Report what works and
what doesn't.

## Get the material

- **Recipe and code:** branch `best-practice` of github.com/ma-gilles/ppca_study (private; ask the owner for
  access if you can't clone).
  - Read `best_v1/RECIPE.md` first, then `BEST_PRACTICE.md` sections 0 and 3-5.
  - `best_v1/` holds the verified driver `coupled_arm.py` (the SGD and shift drivers merged; sources kept), the learning-rate schedule `vdam_step_3000.npy` and an example arm.
  - `relax_pin/` rebuilds the exact relax tree the recipe ran on: public relax at `RELAX_COMMIT` plus 3 patches,
    equal to branch ws2-gated-sgd at ac8e62db. It also builds the native CUDA libraries.
- **If you can't reach GitHub:** the same material is on Della at
  `mg6942@della.princeton.edu:/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_study_practice/` (best_v1/, relax_pin/).
- **Earlier tilt-series handoff:** `/scratch/gpfs/CRYOEM/gilleslab/em_work/tilt_ppca_handoff_20261101/` on Della.
  - It has the relax bundle (`relax_ws2-gated-sgd_ac8e62db.bundle`).
  - `tilt_drivers/` has a working synthetic tilt driver (`science_run_skip.py`), a real-data one
    (`science_run_real.py`) and scorers (`evaluate_arm.py`, `pose_metrics.py`).
  - Its `PROMPT.md` describes how tilt PPCA already works in relax.
- **Tilt PPCA in relax:** `relax ppca_initial_model --ios <RELION 5 optimisation set>`. Read
  `docs/math/vdam_ppca_algorithm.md` section 16 first.

## The recipe, v1 (implement exactly; don't tune before the first full run)

- **Optimizer:** `--optimizer momentum_sgd`, momentum 0, oversampling 0. In the driver:
  `--uniform-direction-prior --sgd-curv split --sgd-dir coupled --adaptive-eta-pc-only --adaptive-eta-pc-floor 0.25`
  and `--lr-schedule file:vdam_step_3000.npy`.
  - **Direction:** per Fourier voxel, H_f⁻¹ G_f, where H_f is the (1+q)×(1+q) Gauss-Newton curvature (a damped
    per-voxel Newton/EM step).
  - **Step size:** lr × the schedule. The schedule is .54 to about update 1800, then decays to .04 at 3000. The first
    step's size is matched to the split-curvature step by a factor k fixed at update 1.
  - **lr:** matched per dataset at update 1 (round 17 `match_lr.py`, MATCH:1.95; values .66-.73 on our data).
  - **PC gate:** the mean is ungated. The PC step is multiplied, per shell, by clip(EMA over 20 updates of the
    correlation between the two half-batch PC gradients, .25, 1).
  - **Orientation prior:** uniform throughout.
- **Translations:** an exhaustive absolute zero-centred window, searched every update, with no stored per-particle
  offsets.
  - Window ±48 Å. Steps 12 Å, then 6 Å from HEALPix order 3.
  - Gaussian prior centred at 0, σ held at 20 Å.
  - These were set for single particles. For subtomograms, take the per-particle 3D shift spread from your picking
    or alignment, and widen the window if it exceeds ±48 Å. Keep steps tied to the angular step:
    step ≈ θ × D/2, with D the particle diameter. Use whole pixels only.
- **Sampling:** grow the band in physical frequency, not pixels.
  - With L = box × voxel size in Å, start at 35 Å and add 1/(384 Å) every 125 updates, up to your target or
    Nyquist − 1.
  - HEALPix order 1 above 32 Å, order 2 down to 16 Å, order 3 below that.
  - 3000 updates of 300 particles (scale the updates to keep at least about 6 passes over the data). q = 4-10.
  - Seeds 11 and 12, random start.
- **What to implement:** the drivers only read single-particle stacks, and the recipe's options are driver patches.
  Add them to relax as normal `ppca_initial_model` options that also work with `--ios`, using the drivers as the
  reference: H⁻¹ direction, PC gate with floor, uniform orientation prior, lr schedule, absolute translation window
  with held prior. Check in the logs that each is really on: the gate values, k, the shift grid, and σ fixed at 20 Å.
  - Work on your own branch, rebase on current relax main, and open a pull request.
  - Don't push to relax or recovar main. Don't change recovar without the owner's OK.

## Test order

1. **Synthetic first.** Simulate one or two small heterogeneous tilt datasets with
   `recovar.simulation.relion_tomo.generate_relion5_tomo_dataset`, with known states and several thousand particles.
   The recipe should recover poses and states there before you trust real data.
2. **Comparison on synthetic data.** Run VDAM-simple (`--optimizer vdam`, driver `--vdam-ko noadapt`) on the same
   synthetic set.
3. **Real data.** Your real ET dataset. Pick one with a published subtomogram average and, ideally, known
   heterogeneity.

## Reporting (the owner's fixed rules; don't invent metrics)

- **Synthetic:** per-state FSC (common frame and independently aligned, kept apart), latent R², and within-state pose
  error (`pose_metrics.py`). Give each map score next to its references: the true mean map alone, and the true model.
- **Real:** map FSC to the published average after fitting rotation, shift and hand; pose agreement with the
  published alignment (one global rotation + hand; median, p90, % under 10°).
- **Always include:**
  - raw mean and PC volumes with their scale (|PC|/|mean|); look at them before reporting;
  - latent scatters in PC coordinates (PC1-2, PC3-4), not only UMAP;
  - wall time and GPU model.
- "Nonfinite or materially indefinite PPCA metric" is a known failure: report it.
- Check real exit codes; an out-of-memory run can look finished.

## Known limits of v1 on single-particle data (so you know what to watch)

- Datasets where the interesting state is 10-20% of particles are seed-sensitive: it poses at some seeds only.
- Pseudo-symmetric proteins can settle in a wrong frame.
- The SGD latents carry some pose-error information (worse than VDAM on this).
- Some sets recover poses but little heterogeneity (map scores at the mean-only level).
