# Current EM/VDAM development scope

Updated September 21, 2026. Cleanup and scientific qualification remain in
progress. The [task queue](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/CURRENT_TASK.md)
contains the active work list; detailed experiment histories live in the private
`ma-gilles/recovar-experiments` repository.

## Scope and invariants

Refactor EM/VDAM for readable, succinct code and direct APIs; integrate qualified
peer work; remove demonstrated dead or duplicate code and completed experiments.
CUDA, FFI and their unique numerical coverage are in scope. Avoid generic
executors or mode switches that obscure distinct algorithms.

Validation is EM/VDAM-only. Do not run RECOVAR-wide SPA/ET, downstream, outlier,
indices, stress or heterogeneity suites, and do not regenerate their baselines.
Structural cleanup preserves scientific defaults, casts, reduction order, JIT
boundaries, layouts and buffer lifetime. EM/VDAM APIs and CLIs may change when
maintained callers migrate together; main heterogeneity APIs and serialized
formats remain compatible. Production EM is float32; double precision is a
labeled diagnostic only.

## Architecture and ownership

`relax/` is the implementation root. Standard refinement and VDAM retain
separate controllers and schedules while sharing sampling, scoring, candidate
layouts and accumulation where their semantics match. RELION runtime adapters,
replay tools, diagnostics and independent references have explicit owners. See
the [codebase map](codebase.md) and [implementation guide](em_implementation.md).

The lead integration branch is `codex/integrate-pr180` in
`recovar_structural_cleanup_20260907`. It contains the selected EM peer head
`1923240c0` and VDAM peer head `5a7fc9037`, plus integration fixes and cleanup.
The two peer branches remain independent sources of future candidates; integrate
at deliberate checkpoints without interrupting their work. Continue selective
review of Roey's `dense_em_refactor`, keeping the clearer implementation where
approaches overlap.

Completed experiment material belongs in the private
[recovar-experiments](https://github.com/ma-gilles/recovar-experiments) archive
with source identity and reproduction records. Main retains production code,
maintained workflows, current scorecards and unique numerical tests. The exact
pre-cleanup status and real-data evidence inventories are preserved in
[archive commit 8b62d4e](https://github.com/ma-gilles/recovar-experiments/tree/8b62d4e1389cb7c106de2436ac38df6e0b7ca172/snapshots/em_development_records_20260921).

## Current evidence and open gates

The integrated CPU EM/InitialModel scope and focused GPU kernel suites pass. The
VDAM merge guard passes its maintained 12-case panel. The current consolidated
scorecards report K1 31/34, K4 direct 41/60, K4 all-class 9/15 and VDAM 12/12;
these are progress measures, not completion. The three recorded real-data K1
calibration cases remain in the maintained science-equivalence scorecard, while
the 10202 target is still pending.

The supported K4 comparison still has a class below its FSC-AUC gate. Saved
launch-ladder contributors agree in a bounded sample, while GPU accumulators are
not bitwise repeatable even on the control. Neither finding justifies a tolerance
change or a rounding-noise dismissal. A BPref accumulator guard now stops a
known non-finite compact-engine failure at its source; the mechanism remains an
open defect.

Class3D (K>1) starts standalone by default: it reads only relion_refine's inputs
([launch recipe](em_parity_runbook.md#standalone-class3d-launch)). On the K4
50k/256 fixture two standalone runs match the non-MPI RELION reference's final
resolution, ground-truth FSC-AUC and class agreement. OPEN (small): their
per-class FSC-AUC against the RELION reference is 0.0002-0.0007 below the band
of three same-seed RELION repeats, comparable to the 8e-4 difference between the
two relax runs (A100 and H100); the cause is unexplained. Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_kclassstandalone_20260923/band50k/GATE.json`.

Multi-optics on another grid (2026-09-25): an optics group whose box x pixel size
exceeds the reference's (scale s > 1) gets a Fourier window wider than the model
sphere. RELION's accelerated kernels relabel the image rows beyond
maxR = min(PPref.r_max, imgX - 1): the coarse diff2 kernel wraps them negative
(acc/cuda/cuda_kernels/diff2.cuh:86-90, 163-164), and the fine diff2 and wavg kernels
move them to the pixel (maxR, i) for both the projection and the image shift
(diff2.cuh:494-530, wavg.cuh:81-86). relax reproduces this
([sparse_projection_radius.md](../math/sparse_projection_radius.md)). With it, the
S3b fast case matches RELION to 1e-8. OPEN, a RELION defect not yet reproduced:
for s >= sqrt(2) the moved fine pixel falls inside the sphere and RELION scores a
wrong pixel at a wrong phase. relax refuses such groups (list the widest optics
group first); the reproduction is planned inside the S4.2 scorer changes
(MANAGER_DECISIONS items 11 and 13). Resolved: the M-step of a class at its full box (S3b optics
group 2 from iteration 3) dropped RELION's N/2..N/2+1/2 ring (shells 50-51). RELION backprojects the
rounded support and bounds only the rotated reference radius (BP.cuh:322), so the class window now
backprojects that support and its reference-sphere clip is the exact cut; the multi-optics replay of
iteration 3 then matches RELION's BPref, and the S3b end-to-end run follows RELION's trajectory
through iteration 6. Also OPEN: a coarse window strictly between
2 maxR and about 2 s maxR, where RELION's wrapped coarse rows land inside the
sphere. Only the fused coarse scorer, the global default, reproduces it. The
non-fused coarse projection and the local parent pass refuse it.

RELION float32 BPref accumulation band (2026-09-25, not reproduced; lead decision). RELION's GPU backprojector adds every
particle of a half into one float32 volume with `atomicAdd` (acc/acc_backprojector.h:41; acc/cuda/cuda_kernels/BP.cuh:157-169)
and reads it back once per iteration (ml_optimiser_mpi.cpp:1857). At iteration 1, when posteriors are broad, the rounding drops
tiny terms. On S3b the DC BPref weight is 0.32 % below `0.999 sum Q^2/sigma2[0]` for RELION and 0.083 % below it for relax,
decaying with radius. RELION's retained mass is 0.999 (14428592), and a float32 sequential-sum simulation reproduces RELION's DC
to 1e-4. By it3 the formula holds to 1e-5. relax loses less because each optics shape class starts from its own zero
accumulator. Single-optics runs are unaffected. This is not an algorithmic choice and depends on accumulation order, so relax
does not reproduce it. It also does not explain the S3b final gap: a free run from RELION's it001 (14431963) ends at GT FSC-AUC
0.8929, where RELION reaches 0.9055 and relax standalone 0.8923. OPEN: that run follows RELION to 1e-3 through it012, and relax's
ave_Pmax sits 0.003-0.009 below RELION's in the HEALPix 5-7 local-search iterations (one-step replays 14434388). Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/multioptics_spa_s3_20260924/s3b_relion_bpref_it1_20260925`,
`s3b_relion_it1_mass_20260925`, `s3b_freerun_from_it001_20260925`.

VDAM K>1 (2026-09-25): relax main 237e76b normalizes sigma2_noise, pdf_class,
sigma2_offset and ave_Pmax by RELION's retained (significant-pruned) class mass;
it used the full mass before, which moved every K>1 trajectory from iteration 1.
On pdb K=2 5k/128 the class splits now match RELION at seeds 29/41/53. The seed-29
spread is closed as basin sampling (2026-09-25): 8 relax vs 7 RELION runs have equal
means (0.3275 vs 0.3312, Welch p = 0.29) and relax's sd is 2.1x RELION's (0.0081 vs
0.0039, F p = 0.049, Levene p = 0.023). There is no per-step mechanism: repeats of both
engines first differ at iterations 48-61 and diverge at the same rate through iteration
70, and one-step relax replays from RELION's checkpoints at iterations 60-199 match
RELION's next iteration at its 6-decimal STAR rounding (class-2 weight within 4.5e-7,
alternating sign; class assignment 100%). Evidence and tools:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamk2_20260924/HANDOFF.json`.

VDAM on the resident engine (2026-09-25, in progress): `--pass2_engine adaptive` runs the
InitialModel E-step on auto-refine's adaptive route (`relax/vdam/adaptive_estep.py`), so its
pass 2 is the compact or, with `RELAX_SPARSE_PASS2_RESIDENT=1`, the device-resident engine.
RELION's three `--grad` E-step differences map onto it: the residual backprojection
(`mstep_subtract_ctf_projection`, now on resident), two pseudo-halfset BPref slots (one call per
pseudo-halfset for now) and the coarse-only `maximum_significants = 100 K`. K=1 only. The switch
is transitional: remove with the old path (the exact-local VDAM route in
`relax/vdam/sparse_pass2_estep.py`) once the resident route is qualified and made the default.
Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamres_20260925/HANDOFF.json`.
Quality on noise1 50k/256 seed 29 (200 iterations, job 14422222, scored 14425217): relax resident
GT FSC-AUC 0.34131 unmasked / 0.75598 masked, inside four RELION runs 0.34125-0.34141 / 0.75515-0.75603;
relax-RELION pair FSC-AUC 0.971-0.984 against RELION-RELION 0.977-0.983. Speed gate (manager item 11,
not met): wall 4549 s against RELION 1451 s on the same node, slower than the exact-local default; the
E-step is 90% of it and grows from 9 s to 31 s per iteration. The resident route stays opt-in and the
exact-local route stays until resident is no slower at equal quality on 10097 and noise1 50k.

OPEN (compact, not fixed: the compact engine is to be deleted): without scale-correction groups the
compact sparse pass 2 takes its non-atomic noise arithmetic, 19% apart in `wsum_sigma2_noise` from
RELION's scale-1 Wavg triplet on the algebraic-Wavg test fixture (repro: `_vdam_args(residual=True,
groups=False)` in `tests/unit/test_resident_vdam_estep.py`, compact against resident, job 14421464).
The resident engine runs RELION's triplet at scale 1 (relax 40b14ac). The default VDAM route (exact-local,
`run_local_k_class_em`) never calls the compact engine and is unaffected; with RELION's exact BPref
operands the compact VDAM arm matched it (accumulators 1.7e-6, maps 4e-7, jobs 14421292/14421464).

Class3D local searches, K>1 (2026-09-25): the per-class local route now keeps
RELION's joint per-particle pass-2 support (3601775); it matches the
class-segmented pass to 1e-7. No benchmark reaches this route (Class3D keeps
HEALPix 1-2, below auto_local_healpix_order 4). Later engine task: give the
class-segmented pass padding factor 2 and a per-class scale-correction mask
(RELION's `data_vs_prior_class[iclass] > 3`), route Class3D local K>1 through
it, then delete the per-class K>1 loop (one implementation; K=1 keeps its path).
Update 2026-09-25: Class3D no longer reaches local searches at all. RELION switches
to local searches from the HEALPix order only under auto-refine: the iteration-0
switch is inside `if (do_auto_refine)` (ml_optimiser.cpp:2541-2565) and later
switches come from `updateAngularSampling`, which Class3D never calls
(ml_optimiser.cpp:3936-3938); relax has no `--sigma_ang`. relax switched Class3D
to local searches at HEALPix >= 4 (fixed 7381f84, replays 016e261), and the
Class3D local K>1 route (the K>1 branch of `_run_local_search_iteration` and the
class arms of the local half scorer) is deleted as unreachable. The per-class
passes in `run_local_k_class_em` stay: they serve K=1 with an external
normalizer (VDAM zero oversampling) and are the reference for the segmented
pass's tests. The segmented-routing task above is therefore moot unless a
`--sigma_ang` Class3D workflow is added.

Class3D noise statistics, compact K>1 (2026-09-25): OPEN against the compact
K-class engine (`compute_k_class_pass2_stats_sparse_fused`, today's default Class3D
pass 2). It has no RELION direct low-shell Wavg residual, so its
`wsum_sigma2_noise` is not RELION's, and on a small fixture one shell goes
negative. Repro (on the branch below): `tests/unit/test_resident_k_class_pass2.py::_k_class_args(3)`
(the 8x8 resident driver fixture with 3 classes). The class sum of the compact
noise tuples is -21.4 at shell 0, and 3835.6 against the resident 4604.7 at K=2.
Job 14421852 (candidate 91596ff) measured rel L2 0.24 at K=2 and 0.71 at K=3
against the resident K-class pass. The two agree on everything else to the
default band: evidence, per-class winners, joint Pmax, class mass and both BPrefs.
The resident K-class pass computes the noise with RELION's arithmetic
(`resident_pass2.compute_k_class_pass2_stats_resident`, branch
claude/ressym-kclass-20260925, landing after its kclass and K4 qualification). Its
noise is pinned by the duplicated-class test against the K=1 resident pass. The
compact K>1 engine is not fixed; it is deleted with compact. The K4 long and 100k
runs against RELION on the resident route will show whether this moved the
existing K4 rows.

Final all-data maps are now always gridding-corrected, as in RELION; the former
default-off selector made the K1 100k/256 masked GT FSC 0.0008 lower than both
same-command RELION repeats over shells 1-60. Pinned merged-map records were
regenerated post hoc from the saved maps (scorecard v2). OPEN (small): with the
correction, relax's GT FSC-AUC sits just below both same-command RELION repeats
on two synthetic fixtures, by about as much as the repeats differ from each
other. K1 100k/256: masked 3.3e-5 / 5.2e-5 below rep2 / rep1 in the end-to-end
qualification run 14365794 (1.6-3.5e-5 post hoc on job 14320204, merged map and
each unfiltered half; repeats differ by 1.9e-5). K1 50k/256 noise 1 (aligned GT): relax
0.35122 vs RELION 0.35129 and its repeat 0.35127. More RELION repeats are needed
to tell a defect from run-to-run variation.

On the 10k EMPIAR-10097 fixture at 256 px on one H100, the resident K1 path
reduced the cold auto-refine gap from 3.26x to about 1.7-1.8x RELION, with a
further small gain from building the projector on device. This qualifies only
that fixture. Remaining cost is concentrated in new-shape transitions and the
final all-data iteration; the latter still uses the exact local path. The
100k/256 K1 and exactly-K4 quality and performance gates remain open.

Explained (2026-09-24, test tiers): the medium K1 5k/128 standalone end-to-end run (medium
14375481, relax 319cd10) scored GT FSC-AUC 0.6084 on relax's `final_merged.mrc` against
0.5959-0.5960 for RELION's `run_class001.mrc`. The cause is the map kind. RELION's map is
gridding-corrected, and relax at 319cd10 did not apply a final gridding correction. Applying
RELION's pad-2 sinc^2 correction to relax's map gives 0.5959, inside the band. The unfiltered half
maps agree: average GT FSC-AUC relax 0.59519, RELION 0.59516-0.59519. This is resolved by the
always-on final gridding correction (user decision; relax df88eab retires the option). The
tier gates both the unfiltered half-map average and the merged map, and requires
`final_all_data_grid_correct` to be recorded True in `refinement_results.npz`.

K=1 engine (2026-09-25): the device-resident pass 2, local search and device significance are the
Refine3D default, with the qualified flag set, including the first-iteration CC pass and
`--adaptive_oversampling 0`. Unset, a pass the resident checks refuse runs on the earlier engine
with a logged reason, recorded per iteration in `pass2_engine_trajectory`; an explicit `=1` makes
that refusal an error. The jitted stage glue and the local image-capacity
ladder are set at the K=1 entry points (`apply_k1_refine3d_env_defaults`), since VDAM shares that
code and qualifies its own defaults. Transitional A/B off switches, to be removed with the compact
engine (not permanent variants): `RELAX_SPARSE_PASS2_RESIDENT=0`, `RELAX_LOCAL_SEARCH_RESIDENT=0`,
`RELAX_COARSE_SIGNIFICANCE_DEVICE=0`, `RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF=0`,
`RELAX_K1_RELION_WAVG_SEQUENTIAL_CUDA=0`, `RELAX_COARSE_PAD_FINAL_IMAGE_BATCH=0`,
`RELAX_EM_JIT_STAGE_GLUE=0`, `RELAX_LOCAL_IMAGE_CAPACITY_LADDER=0`. Flip pairs (relax vs
RELION wall, same node): EMPIAR-10073 1.05x and 10345 1.01x pass the scorecard thresholds; K1
50k/256 runs at 0.65x, with the resident masked GT FSC 9e-5 below the three-run RELION band
(compact inside it). OPEN (small, both engines): on 10345 and K1 50k/256 relax's map agreement with RELION
sits just below RELION's run-to-run band with equal own quality (10345 merged band FSC-AUC
0.982 vs 0.986-0.988, masked 0.9980 vs 0.9985-0.9987; 50k 0.9994 vs 0.9997). Compact shows the
same gap, and RELION native FFT units did not change it. Per-step attribution (measured, K1 50k/256): one-step replays from
the uninterrupted RELION run's states, scoring each half with its own sigma2_noise, deviate from
RELION's next iteration by 4.0e-5 / 9.5e-6 (half 1 / half 2) at it2, 3.2e-5 / 4.7e-6 at it5,
1.5e-4 / 1.8e-4 at it9 and 5.9e-5 / 1.1e-4 at it13 (half-map rel L2 inside the frozen mask; resident
and compact agree to the digits shown except it13, where resident sits closer, mean abs dPmax 3.5e-4 vs 6.6e-4).
RELION's own CPU path, stepped from the same stored states, differs from its GPU path by 3.6e-4 /
3.1e-4 at it2 and 1.4e-4 / 1.1e-4 at it13, while two GPU repeats agree to 3e-7-5e-6. relax's per-step
deviation is therefore at or inside RELION's CPU/GPU arithmetic band (1.0-33x smaller; equal at
it13 half 2). That the full-run
agreement gap (0.9994 vs 0.9997) is the compounding of such arithmetic-path differences is inferred,
not yet measured; an end-to-end RELION CPU-vs-GPU run on the K1 5k/128 fixture is running to test it.
The earlier attribution (1.3-4.2e-4 per step, "about 2.5x RELION's drift") is superseded: the replay
harness then scored half 2 with half 1's sigma2_noise (the RELION MPI restart broadcast); those
replays (relax a7977c8) read 4.3e-4-7.6e-4 for half 2 at it2/it5, against the per-half-noise values
above (relax 64b08ef, whose half 1 also moved from 6.6e-5 to 4.0e-5 at it2). Replays with per-half noise: jobs 14425333 (relax), 14411352 (RELION CPU),
under `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_speed_20260923/replay_50k_iters_cns_20260925`
and `replay_50k_iters_20260925/relion_cont`.

Tested, no effect (2026-09-25, not landed): RELION's CUDA path normalizes each real image by
`(XFLOAT)(avg_norm_correction / normcorr)` (`acc/acc_ml_optimiser_impl.h:875`, f2c1a38). relax
recovers that factor as `float32(combined) / float32(group_scale)`. On the K1 50k/256 fixture this
differs by 1-2 ULP (1.2e-7 relative) for 34% of images at it2 and it13. A port of a parallel
session's fix carried the once-rounded host ratio separately. It moved relax itself by a mean
|dPmax| of 2e-5, but not toward RELION. One-step replays from the uninterrupted RELION run's state
(job 14421433, main a1786bf, H100) were compared with RELION's next iteration of that run:

| Replay | Map rel L2, control -> fix | Mean abs dPmax, control -> fix |
| --- | --- | --- |
| it2, resident | 2.49566e-4 -> 2.49566e-4 | 5.975e-4 -> 5.975e-4 |
| it2, compact | 2.49567e-4 -> 2.49566e-4 | 5.975e-4 -> 5.975e-4 |
| it13, resident | 1.25412e-4 -> 1.25457e-4 | 9.834e-4 -> 9.830e-4 |
| it13, compact | 1.34754e-4 -> 1.34758e-4 | 1.1648e-3 -> 1.1648e-3 |

The table's metrics are the half-map rel L2 inside the frozen mask and per-particle Pmax, both
against RELION's next iteration. The no-effect conclusion holds (both arms share the setup), but the
absolute gaps are superseded: these replays scored half 2 with half 1's sigma2_noise (the harness's
former restart-broadcast default), which inflates half 2 against the uninterrupted run; see the K=1
engine paragraph above for the per-half numbers. Observation: the it13 compact
arm's per-particle Pmax was identical between control and fix, so that route may not consume the
carried factor. Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_normfix_20260925/ab50k_it2_it13/`
(`PERSTEP_DEVIATION.json`, `PMAX_AB.json`).

Map sign convention: every map relax writes is in RELION's map convention, so a relax map and
the RELION map of the same run agree voxel for voxel and in sign (`relax/helpers/map_io.py`).
relax holds volumes internally in RECOVAR's frame, the negated transpose of RELION's file array
(`relion_volume_to_recovar`); the sign entered at the map-file boundary, where the refinement,
the parity harness and the per-iteration dumps wrote with RECOVAR's `write_mrc` and the K1
refinement read its reference with `load_mrc`. Maps are now written with `write_map` (labelled
`relax map, RELION sign and axis convention`) and references are read with `load_relion_volume`:
`--init_volume` and `--init_class_volumes` take the same file as relion_refine's `--ref`, and the
data-directory defaults are `reference_init_relion.mrc` and `reference_init_class00K_relion.mrc`.
Read a relax map with `load_relax_map`; a map written before this convention has no label and
holds the negated array, and the scorers read it as such (`legacy_recovar_sign=True`).
Ground-truth `reference_gt*.mrc` files stay in RECOVAR's frame and are read with `load_mrc`.

Follow the unchanged [quantitative gates](../math/em_parity_program.md) and
[validation ladder](em_parity_runbook.md#validation-ladder): matched-state
scores, support, posteriors, poses and accumulators; synthetic then real K1;
then exactly K4 with Hungarian matching and per-class FSC/FSC-AUC. Completion
requires production float32, matched inputs/seeds/maps/masks, matching
convergence/finalization and at least 100,000 particles at 256x256 or larger on
matched GPU classes. Correlation, diagnostic double, partial iterations and
missing measurements do not satisfy those gates. Scientific acceptance precedes
speed qualification.

Use frozen pixi environments, Slurm for integration and long GPU work, and
sealed identified native libraries. Follow the local GPU0 reservation, the
[EM contract](../../relax/AGENTS.md), [benchmark contract](benchmarks.md)
and [agent workflow](agent_workflow.md). Repeat checks only for changed behavior,
failures, unresolved concerns or required qualification.
