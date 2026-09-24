# relax defaults against the RELION 5 GUI

relax's refinement defaults are the job defaults of the RELION 5.0.1 GUI
(`relion`), which is how RELION is normally run. They are not
`relion_refine`'s command-line defaults. The source is
`/scratch/gpfs/GILLES/mg6942/relion/src` at f2c1a38. `pipeline_jobs.cpp` holds the GUI's
job options (`initialise*Job`) and the command each job writes
(`getCommands*Job`). `ml_optimiser.cpp` holds `relion_refine`'s own defaults
for every option the GUI does not pass. A run that reproduces a particular RELION
command passes that command's values explicitly. The fast, long and completion
tiers do this, and the RELION-seeded and replay debug starts supply the
remaining state from the RELION run.

Audited on 2026-09-24. "Changed" means relax's default moved to the GUI
value in that audit.

## Deterministic start-up methods (all job types)

| Method | relax before | relax now (RELION) | RELION source | Changed |
| --- | --- | --- | --- | --- |
| K=1 half sets | seeded NumPy split unless `--relion_half_sets` or `--relion-half-sets-from-input` | input `rlnRandomSubset` when every row has one, else RELION's seeded assignment; default for a fresh K=1 start (`--relion-half-sets-from-input`), debug starts pass `--relion_half_sets` | `exp_model.cpp:261-403` | yes |
| K=1 particle order | libc `random_shuffle` (`legacy`) | `std::shuffle` with `mt19937(seed + iter)`, then stable sort by optics group; the only order (the libc order and `--relion-particle-shuffle` are removed) | `exp_model.cpp:406-456` | yes |
| Start-up noise | recovar pipeline estimator, RELION's with `--initial-noise-bootstrap relion` | RELION's estimate from up to 1000 masked images per optics group; the only estimator | `ml_optimiser.cpp:3068-3072` and [start-up noise](../math/relion_refinement_algorithm.md#start-up-noise) | yes |
| K=1 start tau2 / data_vs_prior | heuristic unless RELION noise | `MlModel::initialiseDataVersusPrior` on the low-passed reference | `ml_model.cpp:1557` | yes |
| Random seed | 42 | the time, as for `--random_seed -1`; logged and saved | `ml_optimiser.cpp:1232, 2827` | yes |

## 3D auto-refine (Refine3D) and 3D classification (Class3D)

`scripts/run_full_refinement.py` runs both (K=1 is auto-refine, K>1 is Class3D).

| Option (relax) | relax before | RELION GUI default | GUI source (`pipeline_jobs.cpp`) | `relion_refine` CLI default | Changed |
| --- | --- | --- | --- | --- | --- |
| `--data_dir`, `--output` (`--i`, `--o`) | a deleted path | required (empty field is an error) | 4382; 3892 | required | yes (required) |
| `--max_iter` (auto-refine `--auto_iter_max`, Class3D `--iter`) | 10 | auto-refine: none, runs to convergence (limit 999); Class3D: 25 | 3689, 3956 | `--auto_iter_max 999` replaces `--iter` in auto-refine (`ml_optimiser.cpp:1255, 2543`); `--iter` 50 | yes |
| `--healpix_order` | 3 | 2 (7.5 deg with oversampling 1) | 4205, 4489; 3714, 4000 | 2 | yes |
| `--adaptive_oversampling` | 1 | 1 | 4480; 3992 | 1 | no |
| `--auto_local_healpix_order` | 4 | 4 (1.8 deg with oversampling 1) | 4217, 4499 | 4 | no |
| `--offset_range` (px) | 3 | 5 | 4209; 3718 | 6 | yes |
| `--offset_step` (px, before oversampling) | 1 | 2 (1 px times 2^oversampling) | 4213, 4504; 3722, 4015 | 2 | yes |
| `--init_resolution` (`--ini_high`) | 30 A | 60 A | 4172; 3658 | -1 (off) | yes |
| `--apply-initial-lowpass` | off | on (applied when `ini_high > 0`) | 4172; 3658 | off | yes (off for a frozen boundary) |
| `--firstiter_cc` | off | on ("Ref. map is on absolute greyscale?" No) | 4161, 4407; 3647, 3917 | off | yes |
| `--particle_diameter_ang` | a RELION optimiser found next to the data, else the dataset mask | 200 A | 4191; 3697 | -1 (box size) | yes (a supplied RELION optimiser still wins) |
| `--tau2_fudge` | 1 auto-refine, 4 Class3D | not passed (1) auto-refine; 4 Class3D | 3683-3684, 3957 | -1 (1 auto-refine, 4 otherwise, `ml_optimiser.cpp:12015-12026`) | no |
| `--sym` | C1 | C1 | 4175; 3661 | c1 | no |
| `--n_classes` (`--K`) | 1 | 1 | 3681 | 1 | no |
| `--perturb_factor` | 0.5 | not passed | - | 0.5 (`ml_optimiser.cpp:968`) | no |
| `--offset_sigma_angstrom` (`--offset`) | 10 A | not passed | - | 10 A (`ml_optimiser.cpp:902`) | no |
| `--width_mask_edge_px` (`--maskedge`) | 5 | not passed | - | 5 | no |
| `--max_significants` (`--maxsig`) | resolved as RELION | not passed | - | -1 | no |
| `--image-fourier-backend` | host_numpy | (implementation) | - | - | yes: `auto` is relion_cuda for K=1, which the fresh K=1 start requires, and host_numpy for Class3D |
| padding (`--pad`) | 2, fixed | 2 ("Skip padding?" No) | 4297, 4439; 3809, 3942 | 2 | no |
| `--low_resol_join_halves` | 40 A, fixed | 40 A | 4509 | -1 | no |
| `--ctf`, `--flatten_solvent`, `--zero_mask`, `--norm --scale` | always on | on | 4450-4463, 4510; 3949-3968, 4028 | off | no |
| `--trust_ref_size` | reference must match the box | on (resize) | 4171; 3657 | off | not implemented |
| `--preread_images` ("Pre-read all particles into RAM?") | no option (implicit pre-read of stacks up to 16 GiB) | No: images stream from the stacks | 4298; 3810 | off | yes (1e21746, `relax/helpers/particle_io.py`; also InitialModel, replacing `--lazy`) |
| `--scratch_dir` ("Copy particles to scratch directory"), `--keep_free_scratch` | no option (recovar's implicit `TMPDIR` staging) | empty (no copy); keep 10 GB free | 4299-4305 | empty; 10 | yes (1e21746) |
| `--pool`, `--dont_combine_weights_via_disc` | (no option) | 3, on | 4294 | 1, off | no numerical effect |
| final all-data gridding correction (`griddingCorrect`) | off unless `RELAX_FINAL_ALL_DATA_GRID_CORRECT=1` | always on (not an option) | `backprojector.cpp:2021`, `projector.cpp:595-627` | always on | yes (556d342): always on; the selector and its env var are retired. Qualified end to end by K1 100k/256 job 14365794 |
| `--solvent_mask`, `--solvent_correct_fsc`, `--blush`, `--auto_ignore_angles`, `--helix`, `--relax_sym`, `--sigma_ang`, `--fast_subsets`, `--strict_highres_exp`, `--skip_align` | (no option) | off / not passed | 4200-4221; 3695-3734 | off | not implemented |

## 3D initial model (InitialModel / VDAM)

`relax initial_model` (`relax/commands/initial_model.py`) mirrors the GUI
command, which fixes the sampling line (`pipeline_jobs.cpp:3544-3549`).

| Option (relax) | relax before | RELION GUI default | GUI source (`pipeline_jobs.cpp`) | `relion_refine` CLI default | Changed |
| --- | --- | --- | --- | --- | --- |
| `--nr-iter` | 200 | 200 | 3376 | 200 with `--grad` (`ml_optimiser.cpp:2220-2224`) | no |
| `--tau2-fudge` | 4 | 4 | 3377, 3549 | 4 | no |
| `--K` | 1 | 1 | 3382, 3519 | 1 | no |
| `--sym` / `--run-in-c1` | C1 / on | C1 / on | 3384-3385, 3520-3527 | c1 | no |
| `--particle-diameter` | 200 | 200 | 3386 | -1 | no |
| `--solvent` / `--zero-mask` / `--ctf` | on / on / on | on / on / on | 3390, 3530-3531, 3395 | off | no |
| `--healpix-order` / `--oversampling` | 1 / 1 | 1 / 1 | 3548 | 2 / 1 | no |
| `--offset-range` / `--offset-step` | 6 / 2 | 6 / 2 | 3548 | 6 / 2 | no |
| `--padding-factor` | 1 | 1 | 3544 | 2 | no |
| `--grad-write-iter` | 10 | not passed | - | 10 | no |
| `--random-seed` | 0 (RELION's "skip randomisation") | not passed: -1, the time | - | -1 (`ml_optimiser.cpp:1232, 2827, 3726`) | yes |

## Oracle builds and particle order

relax implements only RELION 5.0.1 f2c1a3's particle order (`std::shuffle` with
one `mt19937`, `exp_model.cpp:406-456`). The MOLBIO module build
(`relion/5.0.1/gcc-11.5.0-gpu`) and the d476e6f dispatch-capture build still
use libc `srand`/`random_shuffle`; their STAR headers say `version 5.0.1`
without a commit. The fixture manifest carries an `oracle_build_note` on every
oracle they wrote:

- `em_fixtures/em_symmetry_matrix/k1_c4/relion_ref` was captured with the
  legacy-order build, so its seeded K1 comparison is invalid. It is superseded
  by `relion_ref_f2c1a3/` (Slurm 14367701, same command and inputs, f2c1a3
  build; manifest set `symmetry_k1_c4_relion_f2c1a3`); the old copy stays,
  marked by `relion_ref.SUPERSEDED.md`. The two captures have identical
  `rlnCurrentResolution` trajectories (inf, 60.44, 30.22, 30.22, 30.22, 27.2 A
  for iterations 0-5) and `rlnAveragePmax` within 1e-6.
- The K4 oracles (`k4_fast_oracles/*`, the K4 100k dispatch oracle, the twelve
  K4 symmetry cases) draw Class3D's expected-accuracy trial particles from the
  libc shuffle; relax draws them from mt19937 (`relion_class3d_trial_layout`).
  This is a known oracle-build difference, not a relax bug. The dispatch
  schedule fixes the processing order of those replays.

K1 oracles for the fast, long and completion tiers and the real-data rows are
f2c1a3. The K1 robustness matrix runs the f2c1a3 `relion_refine_mpi` by path
and sha256, and the completion bench refuses a standalone K1 oracle from any
other build.
