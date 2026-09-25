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
(MANAGER_DECISIONS items 11 and 13). OPEN: the M-step of a class at its full box (S3b optics group 2 from iteration 3) differs from
RELION at the reference-sphere edge (shells 50-51 at iteration 3); under investigation, localised
with the multi-optics replay to the class's reconstruction window. Also OPEN: a coarse window strictly between
2 maxR and about 2 s maxR, where RELION's wrapped coarse rows land inside the
sphere. Only the fused coarse scorer, the global default, reproduces it. The
non-fused coarse projection and the local parent pass refuse it.

VDAM K>1 (2026-09-25): relax main 237e76b normalizes sigma2_noise, pdf_class,
sigma2_offset and ave_Pmax by RELION's retained (significant-pruned) class mass;
it used the full mass before, which moved every K>1 trajectory from iteration 1.
On pdb K=2 5k/128 the class splits now match RELION at seeds 29/41/53. OPEN
(small): seed 29 relax is 0.001-0.008 below three same-seed RELION runs (more
repeats running); seed 53 is inside the band. Evidence and tools:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamk2_20260924/HANDOFF.json`.

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
against RELION's next iteration. RELION's one-step repeat noise (continue A vs B) is 6e-7 (it2)
and 4e-6 (it13) in map rel L2, so the per-step gap has another cause. Observation: the it13 compact
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
