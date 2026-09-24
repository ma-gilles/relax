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
