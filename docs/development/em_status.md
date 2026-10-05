# Current EM/VDAM development scope

<a id="current-readability-work-half-image-preprocessing-ownership"></a>

## Current readability refactor

The completed operation ownership is integrated with GitHub main
`80f2b2b36cee01d110276d189a3b061c78746c86`. The thirteen code/guide commits through
`69abd77` retain their changes; the subsequent commit adds PPCA measurement
documentation only. The incoming work includes
pass-1 runner-up, exact-operand, normalization and dump-schema changes,
PPCA work and tomography safeguards. Incoming command/controller changes are
adapted to the existing refactor owners. Scientific sequencing and visible
updates remain unchanged by these interface migrations.

Both refactor packages are since merged with GitHub main
`d1ba3e83406c81ab77891e25d1a775d4a818e674`: the first package at `1b73943`, the
second at `e026a64`; the branch then takes main through
`4efac62a95353714bd6d120e38b595e99015ca52` (merge `b06644c`), whose only
refinement change is the float32 matmul setting in `full_refinement.py`. Main's changes after `80f2b2b` are kept as main wrote
them: the numbered iteration builds the scoring projector on every route (the
"no projector scoring path" error is gone), the adaptive pass-2 grids come from
`oversampling.prepare_adaptive_pass2_grids` with deferred fine rotations, and
pass 1 scores RELION's exact coarse operands only.

The first-CC margin now resolves in `command_options.resolve_firstiter_controls`,
then enters the existing parity settings at the original command boundary.
Numbered expectation and final dense execution consume those settings; the
batch planner reads the same value. `None` disables rescoring; `0.0` is enabled.
K1 CLI `auto` still means 4e-6 with firstiter_cc, while K4 `auto` stays off.
Particle-format optics admission stays immediately after the input STAR is read;
tomography uses main's qualified-feature set and preserves tilt versus particle
metadata indexing.

Focused CPU verification executed 538 cases: 536 passed, and two newly extended
staging expectations assumed the compact coarse planner also applied to enabled
rescoring. Main uses the generic coarse planner in that case. The corrected
nine-case staging suite passed, covering off, zero and the default margin.
The failed full-run receipt is preserved; its repair changes no numerical
production AST or tolerance. Five GPU cases were deselected for fresh GPU tiers.
Whole-source import lint and native source/import checks passed. The six
principal prior, reconstruction, noise, planning and half-input numerical owners
are byte-identical to the preceding implementation.

Earlier frozen `0b46b2` smoke **14909958** passed: 466 GPU cases, 251 controller
cases and all three required parity replays. One optional operand test lacked
its exact-CTF STAR fixture. Its medium **14909959** remains on its original
immutable source. The queued K1 pair **14910610** is preserved under a root-owned
synchronization hold to avoid redundant production work. These jobs do not
qualify the incoming engine changes. The unchanged native-source digest permits
reuse of the already built immutable libraries; runtime library/GPU checks still
run before and after new qualification.

Current integration, receipts, preserved jobs and required gates (was `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/refactor_finish_sync_69abd77_20261003/HANDOFF.json`; no longer available)
track the new candidate. Fresh same-source smoke/medium, float32 K1 and exactly
K4 quality, characterized real-data confirmation, ordinary paired memory/speed,
the milestone long tier and qualified main delivery remain required. No
production quality or performance acceptance is claimed.



The remaining numbered Class3D prior aggregation, regularized reconstruction,
postprocessing, first-CC reporting taper and paired row/noise/accuracy preparation
are integrated in their existing responsibility owners. The controller retains
mode/source admission, explicit scientific state installation, scheduling/history
publication and the transition to final all-data. Three obsolete reconstruction
entry points and their maintained direct/indirect callers are migrated.

Unused temporary expiry after last use is explicitly authorized by the user on
October 2. Selected scientific products retain their required storage; unused
class originals, full CTF tables and taper/admission scratch may expire. Existing
casts, ordered numerical calls, JIT/donation boundaries and final consuming solves
are preserved. Final prior/reconstruction orchestration stays separate because
its replay, CTF, precision and release policies differ; shared mathematical
primitives retain one implementation.

The [complete calling flow and implementations](final_local_sampling_patch_review.md#integrated-controller-ownership-changes)
show the scientific order, producer/consumer ownership and actual caller together.
Source spans are 2731/1767 for numerical/command controllers;
these counts are review signals, not design acceptance. Current CPU checks are recorded above; earlier passing receipts describe their
own source only. The milestone is incomplete until the frozen float32 K1/exactly-K4
scientific, real-data, memory and matched-GPU speed gates pass and delivery to main
is verified. Existing comments, frozen candidates, controls and jobs are preserved.

<a id="current-readability-work-input-geometry-and-replay-boundary"></a>

## Earlier readability work: input geometry and replay boundary

Replay now consumes validated `ImageGeometry`; the generic controller dataset
alias and two silent invalid-pixel substitutions are removed. Original scalar
types remain explicit for existing host arithmetic. Scientific state writes,
sampling order and array lifetimes are unchanged. See the
[actual producer and calling flow](final_local_sampling_patch_review.md#input-geometry-and-replay-boundary).

All 185 selected CPU cases pass with zero skips; source/test lint, structural
comparison and diff checks pass. The initial structural proof required a script
normalization repair, with no production/test change. Exact source and commands
are in the handoff (`/scratch/gpfs/GILLES/mg6942/tmp/relax_input_geometry_20261002T210920Z/HANDOFF.json`; mg6942 scratch outside the repository).

Controller spans 2,917 lines and command 1,860. Representative phase ownership,
whole dead/shared-code audit and the prior Class3D scratch-lifetime choice remain
open. Production float32 K1/exactly-K4, real-data quality, memory and matched-GPU
speed await complete-flow review and a new freeze. Original work, frozen sources,
controls and jobs are preserved. No new GPU job, merge or publication.

<a id="current-readability-work-sealed-restart-runtime-ownership"></a>

## Earlier readability work: sealed restart runtime ownership

The existing diagnostic CLI owner now holds projector-only slots, captured-state
attachment and float32 scoring-noise expansion with its admission/source helpers.
Controller mode/source selection, noise installation, capture publication and
validation timing remain visible. No new function, type, module or wrapper.
See the [actual callers and implementations](final_local_sampling_patch_review.md#sealed-restart-runtime-adaptation).

The affected inventory passed 223 cases and found one stale source guard from
the preceding expectation move; after test-only migration all three affected
guards pass, zero skips. Numerical assertions/tolerances are unchanged. Structural
comparison, isolated diagnostic import, source/test lint and diff checks pass.
Exact source, commands and failure repair are in the package handoff (`/scratch/gpfs/GILLES/mg6942/tmp/relax_sealed_restart_owner_20261002T204921Z/HANDOFF.json`; mg6942 scratch outside the repository).

Controller remains 2,913 lines and command 1,860. Representative design and the
prior Class3D scratch-lifetime choice remain open. Final production float32
K1/exactly-K4, real-data quality, memory and matched-GPU speed await complete-flow
review and a new freeze. No new GPU job, merge or publication; original work,
frozen sources, controls and jobs remain preserved.

<a id="current-readability-work-numbered-expectation-preparation"></a>

## Earlier readability work: numbered expectation preparation

The existing expectation owner now binds the canonical trial grid, dense/local
sampling support and local diagnostic policy once for both halves. The
controller retains mode decisions, explicit tomography adaptation, half-specific
operands, score publication, serial/overlap dispatch and the memory boundary.
See the [complete construction and implementation](final_local_sampling_patch_review.md#numbered-expectation-preparation).

Initial checks passed 132 cases and found eight incorrect new window expectations
plus one source guard needing its new owner. After test-only repairs, 43 repair
cases, 27 affected controller cases and all 108 CPU EM guards pass with zero
skips. Structural, import and lint checks pass. Exact source, commands, repaired
failures and limitations are in the package handoff (`/scratch/gpfs/GILLES/mg6942/tmp/relax_numbered_expectation_preparation_20261002T202923Z/HANDOFF.json`; mg6942 scratch outside the repository).

The controller remains 2,913 lines, command 1,860; scorer has 19 inputs and
producer 18, with 34 half-worker captures. The representative design remains
unfinished. Full production float32 K1/exactly-K4, real-data quality, memory and
matched-GPU speed await complete-flow review and freeze. The prior Class3D
lifetime choice remains pending. No new GPU job, merge or publication;
original workspace, frozen sources, controls and jobs remain preserved.

<a id="current-readability-work-command-oracle-admission-and-runtime-controls"></a>

## Earlier readability work: command oracle admission and runtime controls

Command preparation now owns verified dispatch admission and CLI/optimiser
runtime-control resolution in the existing command-options module. The controller
keeps authoritative group identity, strict replay admission, follower preparation,
frozen-aware source choice and visible option installation before reference
loading. See the [complete current caller and owners](final_local_sampling_patch_review.md#command-oracle-admission-and-runtime-controls)
and [checks/source receipt](final_search_patch_status.md#current-command-oracle-admission-and-runtime-controls).

The initial run passed 208 cases and exposed 12 new fixture setup errors from
unsorted manifest names; all 26 new cases pass after fixing that fixture alone.
Existing 194 cases passed initially, with zero skips. Structural/import checks
pass. Command remains 1,860 lines and numerical controller 2,949; the representative
flow and phase ownership remain unfinished. Production K1/exactly-K4, real-data
quality, memory and matched-GPU performance await design review and freeze. No
new job, merge or publication. Earlier sources/jobs and pending Class3D lifetime
choice remain preserved.

## Earlier readability work: correction reporting and capture

Correction measurements now have one small reporting result in the normalization
owner, reused by checkpoint averages and parity captures. The existing diagnostic
owner performs capture encoding/conversion and preserves warning handling. The
controller retains normalization admission, native/follower correction writes,
convergence, checkpoint order and full/timing-only selection. See the
[complete current operation](final_local_sampling_patch_review.md#correction-reporting-and-completed-iteration-capture)
and [checks and source receipt](final_search_patch_status.md#current-correction-reporting-and-capture).

The initial focused run passed 88 cases and found five failures from two new
fixture mistakes. After repair without assertion/tolerance changes, all 13 new
reporting/capture and 25 affected controller cases pass with zero skips. Existing
normalization, follower, snapshot-lifetime and timing checks passed in the first
run. Structural and import checks pass. The controller remains 2,949 lines,
command 1,992; the half worker still captures 38 names. The representative design
and phase interface remain unfinished. Class3D scratch lifetime awaits the prior
user choice. Full production K1/exactly-K4, real-data quality, memory and matched GPU
speed remain open. No new GPU submission, merge or publication; previous
sources and jobs are preserved.

## Earlier readability work: half input ownership

The existing half model operands carry borrowed radial noise; local/dense
sampling carries coarse angular metadata; particle diameter comes from the
already-required run options. No new function, type, module or flag was added.
Scoring, frame writes, offloading, publication and recording remain visible. See
the [complete current operation](final_local_sampling_patch_review.md#numbered-half-input-ownership)
and [checks and source receipt](final_search_patch_status.md#earlier-half-input-ownership).

The initial focused command passed 122 cases and found three stale indirect
fixture cases. After migration without assertion changes, all three final
preparation and 45 affected controller cases pass with zero skips. Structural
comparison, import lint and pinned imports pass. At that source the controller spanned 2,968
lines, command 1,992; the half worker captures 38 names and the score interface
has 23 inputs. Phase ownership and the representative design remain unfinished.
Class3D scratch lifetime awaits the existing user choice. Full production
K1/exactly-K4, real-data quality, memory and matched-GPU performance remain open;
existing sources/jobs are preserved. No new GPU submission or publication.

## Earlier readability work: half-expectation recording

Half profiles, support counts and E-step captures now have their existing
expectation owner. The controller retains scoring, metadata updates, offloading,
payload publication and the recording call. Count collection preserves worker
recording order, separate Class3D/native policies, absent/empty arrays and
combination after preprocessing drains. See the
[complete current flow](final_local_sampling_patch_review.md#half-expectation-recording-and-support-counts)
and [checks and source receipt](final_search_patch_status.md#earlier-half-expectation-recording).

The initial focused command passed 127 cases and failed one stale checkpoint
fixture. After migration, all 19 checkpoint and 45 affected controller cases pass
with zero skips. Structural comparison and import lint pass. The numerical
controller at that source spanned 2,969 lines, command 1,992; the half worker captured 40
names. Phase construction and the representative refactor remain unfinished.
Class3D scratch-lifetime approval remains pending. Full production K1/exactly-K4,
real-data quality, memory and matched-GPU speed gates remain open. No new GPU
submission or publication; existing sources/jobs are preserved.

## Earlier readability work: reconstruction settings

The representative refactor is still being completed before full production
qualification. The existing reconstruction settings owner now binds invariant
regularization, particle-mask and initial-filter inputs once for numbered/final
priors, reconstruction, postprocessing and Class3D captures. Windows, accumulator
layout, first-CC phase decisions and model updates remain explicit. See the
[complete current calling flow](final_local_sampling_patch_review.md#reconstruction-settings-and-visible-model-updates)
and [checks and source receipt](final_search_patch_status.md#earlier-reconstruction-settings-cleanup).

58 focused and 19 affected controller cases pass. The final capture-interface
migration passes all 11 capture cases; import and structural checks pass. The
unused command noise validator is removed after caller/dynamic-name screening;
33 affected startup, command and import cases pass. The
numerical controller at that source spanned 3,037 lines and command 1,992 lines. Class3D prior
aggregation is waiting for the concrete buffer-lifetime choice sent to the user.
Other controller responsibilities remain unfinished. Full production K1,
exactly-K4, real-data quality, peak-memory and matched-GPU performance gates remain
open. Existing sources/jobs are preserved; no new GPU submission or publication.

## Earlier readability work: numbered sampling

The refactor is actively completing code before further production qualification,
following the user's October 2 clarification. Angular-transition policy now
belongs to `refinement/convergence.py`; numbered perturbation and scoring-window
planning belong to `refinement/iteration_planning.py`. The controller preserves
pre-update coarse order, replay/accuracy timing, grid execution and explicit
state writes. See the [complete calling flow](final_local_sampling_patch_review.md#numbered-sampling-policy-and-fourier-windows)
and [current package receipt](final_search_patch_status.md#earlier-numbered-sampling-cleanup).

Focused CPU checks pass. At that source the numerical controller spanned 3,052 lines and command
1,992 lines; both are unfinished. Continue phase construction/publication and
reconstruction ownership work with buffer-lifetime review. Full production K1,
exactly K4, real-data quality and repeated matched-GPU performance remain open.
Preserve existing frozen sources/jobs; no new submission or publication occurred.

## Current main integration and Class3D shape results

Projector preparation now owns reuse of the accuracy projector, reference
conversion, the established disk cache and optional dumps in the existing
`projector_preparation.py` module. `ProjectorReuse` is produced with the accuracy
projector; it pairs the same borrowed reference identity with its active window
and fixed image support. Accuracy and scoring consume it directly. No new module
or forwarding wrapper was added.

The controller retains captured-projector admission, empty-half handling, ordered
half iteration and explicit projector installation. Its old-projector list reset
and last-projector assignment remain at their original release boundaries. The
complete producer, operation and caller are in the [review](final_local_sampling_patch_review.md#projector-preparation-and-accuracy-reuse).

This package passed **79 affected CPU tests, 108 guards and 96 actual old/new
preparation-stage comparisons**, with no skipped cases in those test receipts.
Controller lifetime tests exercise old-list release, the last-projector alias,
accuracy/scoring reuse and an empty half. Cold transform/cache arithmetic,
captured validation, finalization order and the remaining controller statements
match the previous source after the reviewed adapters are expanded. Lint,
guide mirrors, metrics, pinned imports and native-source hashes pass. Existing
deliberate complex128 slabs and float64 power spectra remain unchanged; this is
not a switch to double-precision EM.

The candidate incorporates cached main `7412fd8`; remote refresh is unverified.
Original dirty work/comments and all earlier frozen sources/jobs are preserved.
The current handoff (`/scratch/gpfs/GILLES/mg6942/tmp/relax_projector_reuse_owner_20261002T134633Z/HANDOFF.json`; mg6942 scratch outside the repository) records exact source identities,
commands and the two failed new-fixture receipts followed by their passing repairs.
The previous main integration (`/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T131551Z/HANDOFF.json`; mg6942 scratch outside the repository)
retains its 257 distinct affected CPU cases, 108 guards and 37 unexecuted GPU cases
for that source. Its explicitly built and frozen CUDA 12.8 libraries are reused
after current source/hash checks.

The command `main()` remains **1,992 lines**, and the numerical controller **3,184**.
Both are unfinished. Class3D aggregation and broader sampling extraction still
require explicit lifecycle/RNG decisions. The next step is review of this complete
projector flow and the remaining controller responsibilities, before broad automation.

GPU smoke, production float32 K1/exactly K4/real-data quality, repeated matched-GPU
timing and peak RSS remain unqualified. The older K1 convergence discrepancy is open.
This session cannot reach the GPU driver or Slurm sockets. No new jobs or publication
occurred; inspect the earlier uncertain submission's remote state before retrying.


## Current startup pose source ownership

Startup pose selection, loading, corrections and provenance now have one
operation in `relion/input_poses.py`. The command retains source setup order and
visible replay/frozen-correction construction. Reports reuse provenance produced
with the selected data. See the [complete caller, result types and implementation](final_local_sampling_patch_review.md#startup-pose-source-ownership).

414 affected CPU tests pass, including 34 new source cases. All 504 actual
previous/current cases match (344 successful selections, 160 matching refusals);
logging and 80 borrowed-array checks match. Eight archive comparisons/roundtrips,
full-command AST expansion, import/lint and native-source hashes pass. The 108
CPU guards pass on the unchanged engine path; exact version/reuse scope is in the
handoff (`/scratch/gpfs/GILLES/mg6942/tmp/relax_initial_pose_owner_20261002T112730Z/HANDOFF.json`; mg6942 scratch outside the repository).

The command remains **1,992 lines** in a 2,762-line file; the numerical controller
remains **3,194**. Both are unfinished. Pose preparation has 18 explicit inputs;
archive adaptation has 28. Next, complete projector/restart provenance at its
producers and larger controller boundaries after buffer-lifetime review.

This tested package is based on main `ab6595d`. Cached main has advanced to
`c70f1a1`; integration follows this immutable snapshot. Remote refresh is unverified. Original dirty work/comments and frozen candidates/jobs are preserved.
No publication or new GPU submission occurred. GPU smoke, float32 K1/exactly K4
and real-data quality plus repeated matched-GPU timing/peak RSS remain unqualified.
GPU driver and scheduler access still fail; inspect the prior uncertain submission
before retry. Human review of the completed example precedes broad automation.

## Current command configuration and archive metadata

Earlier configuration/archive owner and main integration (`/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T111111Z/HANDOFF.json`; mg6942 scratch outside the repository)
remain preserved.

## Refactor history and scientific gates

Read the concise [current qualification status](final_search_patch_status.md)
for source-specific numerical findings, preserved smoke/job evidence and the
remaining float32 K1/K4/real-data/performance gates. Earlier readability source
reports are retained in the immutable [previous status][refactor-history].
The following preserved headings keep existing links; their old next actions
are historical evidence rather than current instructions.

## Current iteration-zero model replay owner

[Recorded source and evidence][previous-status].

## Earlier main integration

[Recorded source and evidence][previous-status].

## Current checkpoint capture ownership

[Recorded source and evidence][previous-status].

## Earlier iteration convergence policy

[Recorded source and evidence][previous-status].

## Earlier startup sampling state

[Recorded source and evidence][previous-status].

## Earlier particle row layout

[Recorded source and evidence][previous-status].

## Earlier particle input preparation

[Recorded source and evidence][previous-status].

## Earlier particle-pose transition

[Recorded source and evidence][previous-status].

## Earlier iteration-resolution ownership

[Recorded source and evidence][previous-status].

## Earlier command-option ownership

[Recorded source and evidence][previous-status].

## Earlier particle-group ownership

[Recorded source and evidence][previous-status].

## Earlier follower-topology preparation

[Recorded source and evidence][previous-status].

## Readability refactor: particle pose interpretation

[Recorded source and evidence][previous-status].

## Readability refactor: normalization update

[Recorded source and evidence][previous-status].

## Readability refactor: rotation-grid ownership

[Recorded source and evidence][previous-status].

## Readability refactor: final all-data controller

[Recorded source and evidence][previous-status].

[refactor-history]: /scratch/gpfs/GILLES/mg6942/tmp/relax_initial_model_replay_20261002T091611Z/source/candidate/docs/development/em_status.md
[previous-status]: /scratch/gpfs/GILLES/mg6942/tmp/relax_initial_model_replay_20261002T091611Z/source/candidate/docs/development/em_status.md

## Subtomogram InitialModel and first-iteration CC (October 1, 2026)

Subtomogram particles (RELION 5 2D stacks) run RELION's VDAM InitialModel
(`relax initial_model --ios optimisation_set.star`, RELION's `--grad --denovo_3dref`) and
`--firstiter_cc` in Refine3D and Class3D. One implementation: the VDAM driver takes a
`TomoDataset` and scores through the subtomogram Refine3D/Class3D pass
(`relax/vdam/tomo_estep.py` over `tomo_half.score_tomo_half`).

Maps lost in the 2026-10-03 cleanup incident (benchw job 14936017): the relax arms of the rows below from
`em_work/cryoet_vdam_20261001` (et09/et15 optics-group, optics K1, spa_k2same, multishape, subtomogram Refine3D
`--firstiter_cc`); their scores are retained in `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_evidence/etvdam_scores_20261003/`.

- Start-up: RELION counts tilt images against `minimum_nr_particles_sigma2_noise` (10,
  ml_optimiser.cpp:2574, :3058), so the first particle of each optics group gives the noise
  spectrum and the bootstrap; each of its tilt images backprojects at `Aproj R` with its
  dose-damped CTF, and the reference gets no blobs or soft mask (:2707). relax matches RELION's
  `run_it000` (sigma2 noise to 4e-7, class map to 3e-8 relative L2; et09_box64, et15_k2conf_box64 seed 1;
  `em_work/cryoet_vdam_20261001/checks/bootstrap_vs_relion.py`).
- E-step: every tilt image backprojects VDAM's residual into its particle's pseudo-halfset slot
  `class + K * (part_id % 2)`; sigma2_offset divides by 3; auto-sampling keeps at least half the
  offset step (:9832). A tilt chunk whose (slot, row) projections do not fit is projected and scored
  a block of image slots at a time (`resident_tilts.tilt_projection_slot_block`).
- RELION behaviour, reproduced: with one optics group the bootstrap fills class 0 only, and an empty
  class stays an exact zero map (its tau2 is zero), so K>1 VDAM on one optics group is K=1 plus empty
  classes in RELION and relax alike (et15 seed 1: class 2 std 0.0 at iterations 0-100 in both,
  pdf_class at iteration 10 0.655/0.345 relax, 0.643/0.357 RELION). Its tied rotations give single
  particles tens of thousands of significant samples.
- `--firstiter_cc`: the coarse pass adds every tilt image's normalized CC on RELION's square
  current-size crop (DC and x=0 column included), with RELION's 128 atomic additions per image, and
  keeps the first maximum in RELION's orientation order (`tomo_coarse.particle_coarse_cc_winners`);
  pass 2 scores its children (`resident_tilts.tilt_cc_scores`: each image's CC operands translated per
  slot). Class3D's CC iteration is K=1 against class 0 (ml_optimiser.cpp:4389). Iteration 1 of
  et01_base seed 1 against RELION 5.0.1 (mpiscale build, MPI 3x4, H100), `run_it001_data.star`:
  orientations identical for 2000/2000 particles, offsets for 1998/2000
  (`em_work/cryoet_vdam_20261001/cc_it1`). Iteration 1 took 332 s in relax, about 60 min in RELION.
- Speed (open): at et09 iteration 100 (590 particles, HEALPix 3, 180 3D coarse translations) the
  relax iteration is 2.1x RELION's (165 s vs about 78 s), 120 s of it the coarse pass: the per-image
  coarse kernel takes 128 translations per launch, so 180 need two launches that each project all
  36864 orientations. The coarse pass also pulled every particle's tilt-image matrices to the host
  (a sync per particle); they now stay on the device, built once per particle batch (identical
  coarse cut): one late et09 E-step (1000 particles) 137 s -> 107 s, coarse 120 s -> 89 s, et13
  Class3D 25 iterations -4.7%. The coarse kernel `relion_coarse_diff2_projector_f32_kernel` is now
  93% of the E-step's GPU time (85 ms per particle; nsys,
  `em_work/cryoet_vdam_20261001/profile`).
- The subtomogram coarse pass now scores each tilt image with the SPA coarse GEMM scorer
  (`tomo_coarse._images_coarse_gemm_diff2`, `scoring._relion_coarse_gaussian_gemm_scores_jit`): the persistent
  projector texture writes the GEMM's packed `[Re | Im]` reference rows in one launch per slot block
  (`ProjectRelionHalfCapacityTextureCompactPacked`), diff2 is summed translation-major in `img_id` order and each
  significance flush is read back after the next is dispatched. Late E-step on one H100, same node, against the
  direct-square kernel (job 14863136): et09 K=1 coarse 78.3 -> 37.9 s (E-step 95.7 -> 55.7 s), et15 K=2 coarse
  433.5 -> 133.6 s (E-step 448.7 -> 148.5 s); every particle's coarse support is identical, Pmax within 1.1e-2
  relative (p99 3.5e-3, the expansion's float32 rounding). The scorer's GEMM runs near the float32 peak; the
  projection is texture-bound. Translating every tilt image of a block in one launch with its own phases
  (`relion_translate_score_f32` with `[B, T, 2]` angles; coarse scorer and the tilt pass-2 Wavg window) brings the
  same A/B (job 14864249) to et09 E-step 95.7 -> 45.9 s (pass 2 16.2 -> 7.6 s) and et15 448.4 -> 143.2 s, supports
  again identical. et09_box64 K=1 seed 2 end to end on one H100 (job 14864247): 6391 s, against 18434 s for relax
  before this work and 9955 / 8858 s for stock RELION; per-iteration average Pmax tracks the earlier relax run
  (it200 0.659 vs 0.658; RELION 0.669-0.670).
- Exact cut of the subtomogram coarse pass (2026-10-05, speedw; `relax/scoring/exact_cut.py`,
  `tomo_coarse.particle_coarse_supports`). The GEMM scorer's expansion `d0 + 0.5 A + 0.5 C - X` rounds differently
  from RELION's direct square; a particle's diff2 is the sum of about 40 tilt images' (thousands, float32 unit 2e-4)
  while its samples' log weights at the significance cut are closer than that, so the rounding moved samples across
  the cut: on etob2l_plain, iteration 2 from an identical iteration 1, the kept sets matched RELION's for 194 of 200
  particles, and on w2_09 (box 192 Refine3D) particle TS_06/154 kept one coarse sample where RELION keeps two and
  lost the one that wins the fine pass. Each flush now cuts twice. The first cut is on the GEMM scores; the rotations
  holding a sample the scores do not decide are scored again by RELION's fused direct-square kernel
  (`relion_coarse_diff2_projector_per_image_f32`, every translation of the rotation, the images added in slot
  order) and the flush is cut again on those values. Undecided means within twice the error bound of the cut:
  of the `max_significants` rank where that sets the cut, and of the adaptive-fraction threshold otherwise (on the
  sorted weights, between the first sample whose running sum exceeds the tail shrunk by `exp(-margin)` and the
  first whose sum exceeds the tail grown by `exp(margin)`); for a particle with such a sample, the rotations within
  the bound of its smallest diff2 too, because RELION's `prior + min_diff2 - diff2` rounds `prior + min_diff2` per
  sample at the diff2's float32 unit. The bound is the worst case of the two float32 forms,
  `(4 P + 11) 2^-24 (d0 + 0.5 max_r A + 0.5 C + sqrt(max_r A * C))` per image for `P` score pixels
  (`exact_cut.gemm_error_bound`), plus one unit of the largest diff2 per image for the slot-order sums; no fitted
  constant. Observed GEMM minus direct square: at most 8, 22 and 30 float32 units of the diff2 at 84, 544 and 2664
  pixels over 1e6 samples each (GPU test), 6 units over 14M samples of etob2l. Results: etob2l it1 -> it2 kept sets
  199, 199, 198, 199, 199 of 200 in five local runs on earlier heads and 198, 197, 198 on the landed code
  (RELION's exact ties are particles 683, 412 and 622; 895 and 980 sit within one float32 unit of the cut), BPref
  relative L2 8.4e-5 / 2.5e-4 per half (GEMM 1.3e-3 / 6.5e-4;
  RELION against RELION 1.4e-4 / 1.6e-4); w2_09 iteration 1 equal to RELION for all 2000 particles and the full
  run's map agreement back to 0.999999 (etw, job 14993708, on the earlier head b2d719dd). Samples scored again on
  et09_box64 K=1: 0.5-1.0% in iterations 1-3, 0.005% by iteration 10-20, 0.0035% from iteration 30 on (one or two
  rotations per particle of 36864). Gate (i) (etob2l it1 -> it2): passes the kept-set reading, does not pass
  etvdam's it2 map bound, and is about 4x closer to RELION than main. Kept sets 198, 197, 198 of 200 in three local
  runs (speedw), BPref relative L2 8.2e-5 / 2.5e-4 per half. etvdam's it2 map relative L2 against 8 RELION runs:
  kept sets 196, 195, 200, median 7.0e-4 to 1.09e-3 and max 9.9e-4 to 1.38e-3 against the 8.9e-4 bound (RELION's
  8-run pairs: median 4.4e-4, max 1.10e-3); main's GEMM scorer median 3.85e-3, max 4.14e-3; the cap-only re-score
  ca31baaf (one run) median 2.48e-4, max 8.23e-4. The mass-cut part of the re-score moves it2 away from RELION
  relative to cap-only; why is open (follow-up, speedw). Cost, full et09 K=1 runs: +8.3% on H100 (3 concurrent arms
  on one node, noisy; job 15009587, main 5926 s against 6420 s), +5.3% on A100 (local, concurrent; 10838 against
  11413 s). Per span of iterations on the H100: +19% in 50-100, +7% in 100-150, +4.5% in 150-199. Follow-up (open):
  a clean same-node H100 pair (main against this commit only), and a flush profile of iterations 50-100 (samples
  re-scored per flush, time per flush, whether a tighter bound or skipping groups with no sample near the cut gets
  the cost under 3% with gate (i) kept). Earlier placements of the second cut cost more and were dropped: per batch
  behind the next batch's scores +13.1% (job 14978218), per batch before them +9.1% (job 14997577, where the model
  bound `2 sqrt(2 P)` units read +7.5% and was not adopted).
- Subtomogram Refine3D `--firstiter_cc` (RELION's default command; RELION 5.0.1 mpiscale MPI 3x4, H100;
  relax accuracy-only arms on A100). Masked GT FSC-AUC on main 22ea0b2 against RELION's same-seed range
  (stock r1/r2, eto_plain also r3/r4, plus the double-BP build where run):
  | case | RELION range | relax 22ea0b2 | pre-d28c258 relax |
  | --- | --- | --- | --- |
  | et01_base s1 | 0.991616-0.991619 | 0.991596 (-2.0e-5) | 0.991610 |
  | et01_base s2 | 0.991803 | 0.991802 (-6e-7) | 0.991800 |
  | et01_base s3 | 0.992231-0.992234 | 0.992243 (+9e-6) | 0.992118 |
  | eto_plain s1 | 0.85743-0.85762 | 0.85836 (+7.4e-4) | 0.85626 |
  | eto_plain s2 | 0.86155-0.86171 | 0.86163 (inside) | 0.86130 |
  | eto_plain s3 | 0.86394-0.86398 | 0.86387 (-6.9e-5) | 0.86388 |
  The pre-d28c258 arms (coarse CC pass projected in double, not RELION's float texture) moved 2-8 particles at
  the deterministic CC iteration 1 and sat below RELION in 4 of 6 rows; on main iteration 1 moves 0, or 1 at
  RELION's own near-tie rate, and the residuals have both signs. A one-step replay from RELION's eto_plain s1
  iteration-1 state (state-swap all_relion, uninterrupted noise) leaves iteration 2's half-1 map 8.7e-3 from
  RELION's: one particle (TS_02/68) keeps one more coarse sample at the 0.999 significance cut and moves 6
  degrees, two others change their sample count; the next step (2 -> 3) agrees to 5.5e-5. RELION's own
  same-seed runs flip 1-2 particles at that cut in iteration 2 (up to 49 by iteration 6, s2 r1 vs r2), so
  relax's residual source is RELION's run-to-run cut flips, amplified by the trajectory. PASSING (2026-10-03):
  with RELION's r1-r4 and double-BP runs relax is not below at most seeds (eto_plain s2 inside, s1 above, s3
  6.9e-5 below; et01 one seed above, two within 2e-5 below a range of width <= 3e-6) and its residuals have both
  signs at RELION's cut-flip rate. Main's later 3e28f96 (group scale clamp) changes behaviour only when a group's
  scale median or mean is <= 0, which these plain fixtures never reach, and 30860a4 is premultiplied-only, so the
  22ea0b2 arms stand. Wall 0.43-0.46x (et01, H100, earlier arms). The relax maps of these arms were lost in the
  2026-10-03 cleanup incident; the scores are retained in
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_evidence/etvdam_scores_20261003/` (et01_cc_main_22ea0b2.json,
  eto_plain_relax_22ea0b2.json, eto_plain_relion_r3r4.json, eto_plain_relion_bpd.json, SHA256SUMS).
- Subtomogram VDAM K=1 (et09_box64, one optics group, stock seeding, s2): masked GT FSC-AUC
  0.98808 inside RELION's same-seed range [0.98795, 0.98816]; wall about 2x RELION before the
  coarse matrices fix.
- Subtomogram VDAM expected accuracy (2026-10-02): relax dropped the particle STAR's input angles, so it
  skipped RELION's expected accuracy until every trial particle had been visited (et09: iteration 60)
  and searched translations coarser from iteration 30 (offset step 3.0 A vs 2.125 / 1.06 / 0.66 A at
  iterations 30 / 40 / 50), which raised the mid-run Pmax (0.94-0.96 vs RELION 0.76-0.85 at it30-60).
  Unvisited particles now keep their input angles, as in RELION (ml_optimiser.cpp:9505). Re-run at
  1a107f5 (main 7412fd8 + this fix + VDAM optics; one H100 per arm, cold; stock RELION non-MPI; masked GT
  FSC-AUC, 181e99d scorer), relax / RELION same-seed range:

  | Case | s1 | s2 | s3 | Wall relax / RELION |
  | --- | --- | --- | --- | --- |
  | et09 K=1, one group | 0.98877 / [0.98875, 0.98879] | 0.98823, 0.98821 / [0.98824, 0.98839] | 0.98805 / [0.98733, 0.98739] | 0.53-0.71x |
  | et09 K=1, 8 groups | 0.98843 / [0.98841, 0.98848] | 0.98806 / [0.98803, 0.98809] | 0.98811 / [0.98816, 0.98817] | 0.66-0.73x |
  | et15 K=2, 4 groups | 0.98394 / [0.98378, 0.98387] | 0.98203 / [0.98211, 0.98212] | 0.98244 / [0.98234, 0.98242] | 0.42-0.45x |

  Translation sampling now follows RELION step for step (et09 s2). The mid-run Pmax still differs (relax
  0.67-0.77 vs RELION 0.76-0.79 at it40-70): one-iteration continuations from RELION's own checkpoints with
  RELION's sampling perturbation (diagnostic `--diagnostic-continue-input-order`, subtomograms included) match
  RELION at it10, 20, 30, 40, 60, 70 and 80 (average Pmax within 2e-3, best poses equal for 98-100% of the
  subset, map one-step difference 0.5-4e-3 of a 4-20e-2 step), so it is trajectory divergence, not a per-step
  difference (RELION's own `--continue` at it050 departs from its run: tau2_fudge 4.0, step size 0.3).
  Inside or above at two of three seeds in each case. Seed 2 with one more stock RELION run each (job
  14914672; all maps rescored together, `scores/s2_band`): et09 one group RELION 0.988369 / 0.988392 /
  0.988241 / 0.988241, relax 0.988232 / 0.988206 (0.9e-5 and 3.5e-5 below the range, which is 1.5e-4 wide):
  OPEN, a relax rerun on current main is scored next. et15 RELION 0.982124 / 0.982101 / 0.982157, relax
  0.982034 / 0.982044 / 0.982124: the ranges overlap, relax's mean is 6.0e-5 lower. Both seed-2 rows sit
  on the low side. A replay audit of both seeds found no per-step defect: one
  iteration from RELION's checkpoints (it010, 100, 190; RELION's sampling perturbation) has RELION's
  update scale (per-shell projection 0.993-1.001), tau2 equal at RELION's print precision, populations
  to 1e-6 and best poses for 99.6-100% of the subset; a float64 M-step changes the step by 3e-7. The
  remaining one-step difference comes from the E-step (per-particle Pmax 1e-6 to 1e-3, noise sums 1e-5
  to 5e-4, no sign), against final-map gaps of 2-8e-5 and seed-to-seed spreads of about 2e-3. Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/cryoet_vdam_ogtomo_20261002`,
  `em_work/cryoet_vdam_20261001/{tomo_cont,audit}`.
- Subtomogram VDAM at low SNR (etoptics bin-2 fixtures `etob2l_plain` / `etob2l_evenz`, SNR 0.005, 1000
  particles, 8 optics groups, seed 1; 2026-10-04): inside RELION's same-seed range, no code change. relax's
  average Pmax at iteration 200 is 0.450 (0.448 on main b44f33e) against 0.607 for the first stock RELION
  run, because relax refines the offset step at iteration 80 (to 0.9945 A) and that run at iteration 90 (to
  1.275 A). Two more stock RELION runs at the same seed (job 14952405) refine at iteration 80 to 0.975 A and
  end at Pmax 0.449 and 0.449, with relax's resolution-shell sequence; the first run is the outlier. The
  split is RELION's: `updateAngularSampling` runs every 10 iterations and proceeds only after an iteration
  without resolution gain, and the resolution (last shell before the first SSNR below 1) hinges on shell
  11, whose SSNR after iteration 64 is 1.0096 (RELION) and 1.0036 (relax) while shells 13-15 are already
  above 1; same-seed RELION runs differ by 0.3-1.0 % in SSNR by iteration 60. One iteration from RELION's
  iteration 60, 70 and 80 states (job 14952421) reproduces RELION's next resolution and its SSNR to 1e-4 at
  shells 8-25 (per-particle Pmax difference unsigned, median below 4e-4); the map after that step differs
  by 1.9e-3 to 4.6e-3, against 4e-6 to 7e-6 between two RELION builds. Same job, same node, one H100
  each: relax 7004 s, the two RELION runs 10205 and 9900 s (the first RELION run, with the coarser step,
  took 5010 s on another node). Final maps: masked GT FSC-AUC 0.97645 (relax) and 0.97646 (first RELION
  run). The beam-tilt and magnification cases refine at iteration 90 in both programs. OPEN under the
  range rule: relax is outside RELION's same-seed range early in the run. Map relative L2 to RELION,
  every iteration written in both: 0 at iteration 1, 4.1e-3 at iteration 2, 2.1e-2 at iteration 10, where
  the three RELION runs differ by 0.4e-3 to 4.6e-3 and two relax revisions by 0.2e-3 (by iteration 60 the
  ranges touch: RELION 3.5-4.8e-2, relax 5.3-5.9e-2). The first departing step is iteration 2, from an
  identical state: best poses and offsets equal for all 200 particles, Pmax within 2e-4, and the update
  differs by 1 % of its norm, evenly over shells and without a scale. Every particle is at the coarse cap
  of 100 significant samples there. Localized (2026-10-04, jobs 14967796 and 14970694; RELION's coarse kept
  sets and BPref accumulators dumped for that step): the coarse scorer's rounding at the significance cap.
  The subtomogram coarse pass uses the SPA coarse GEMM scorer, whose expansion differs from RELION's
  direct square by about one float32 unit of a diff2 near 2640 (2.4e-4); at the cap 7 % of particles have
  a relative gap under 1e-3 at the cut and 1.5 % an exact tie, so the kept set differs for 6 of 200
  particles (both programs use the same inclusive cut and the same normalisation). BPref data then
  differs by 1.3e-3 and 6.5e-4 in the two pseudo-halves and the map by 3.3e-3. With the coarse pass on the
  direct-square kernel (diagnostic) the kept sets agree for 197 of 200, BPref data to 1.3e-4 and 2.8e-4 and
  the map to 2.6e-4 to 4.1e-4, against 198 of 200, 1.4e-4 and 1.6e-4, and 1.7e-4 between two RELION runs.
  A second defect, fixed (2026-10-04): on an iteration that updates the sampling (10, 20, ...) RELION sets
  the coarse image size before it updates the angular sampling (expectationSetup, ml_optimiser.cpp:3914
  before :3945), so that iteration still scores its coarse pass at the previous order's size. The
  single-particle E-step already took that order; the subtomogram E-step used the updated one (coarse size
  22 instead of 12 at iteration 10 of this case): 2.5 significant samples per particle against RELION's
  24.2, a different best pose for 28.5 % of the particles and a different offset for 22.5 %, where three
  RELION runs agree on all 200. With the fix: 24.5 significant samples, 8.5 % and 6.5 % (iteration 20:
  20.0 % -> 10.5 % of poses). Every subtomogram VDAM row measured before this fix crossed those iterations
  with the wrong coarse size and is rerun on it. The trajectory is still outside RELION's same-seed range
  after both changes: per iteration, best poses differ from RELION for 1.7 % (iterations 2-9), 3.6 %
  (10-19) and 6.9 % (20-30) of the particles with a direct-square coarse pass (4.1 %, 6.1 %, 10.0 % with
  the GEMM scorer), where RELION runs differ from each other by 0.5 %, 1.0 % and 2.5 %; OPEN.
  What the remaining difference is (2026-10-04, RELION's and relax's coarse diff2 rows and prior terms dumped
  per particle from fresh runs, jobs 14973183 and 14973184): from an identical state (iteration 2) the old
  offsets are equal for 200 of 200 particles, sigma2_offset is equal, the orientation and offset priors
  agree to 1e-5, and the coarse diff2 rows agree to a residual standard deviation of 3.8e-4 (1.6 float32
  units of a diff2 near 2640; largest cell 1.7e-3 to 2.4e-3) with the direct-square pass. There is no
  prior, unit or pre-shift difference. That rounding moves the membership of the two significance cuts
  (the cap for 1.5 % of particles at iteration 2, the 0.999 mass cut for 6-15 % later) and now and then the
  winner; a different winner changes the particle's carried offset and prior, which then differ for good
  (two such particles of 200 at iteration 8), so the difference compounds. The rows are judged on the
  outcome once the exact re-score at both cuts is in.
  Restart-based replays are superseded: relax's diagnostic continuation (`--diagnostic-continue-optimiser`;
  there is no user-facing resume for VDAM, Refine3D or Class3D, so this is audit code only) does not rebuild
  the E-step input that RELION's `--continue` has. From RELION's iteration-8 checkpoint its coarse diff2
  rows differ from RELION's by a residual standard deviation of 1.15, against 0.045 between fresh runs at
  that iteration. Every pose, Pmax and significant-sample figure quoted on this page from a one-iteration
  continuation (the et09/et15 audits above, the iteration-60/70/80 audit of this case) is superseded by
  the fresh-run measurements; their resolution, SSNR and tau2 comparisons are M-step outputs and are not
  the quantities at issue.
  Single-particle VDAM is not affected (2026-10-04, jobs 14974373 and 14974374, plain K=1 10k/256, seed 1,
  iteration 2, where all 200 images sit at the cap of 100 and 7 % have a relative gap under 1e-3 at the
  cut): relax on main against RELION, and RELION against RELION:

  | | map it1 | map it2 | BPref data per half | poses differing |
  | --- | --- | --- | --- | --- |
  | RELION vs RELION | 6.3e-7 | 7.4e-7 | 3.9e-6, 3.7e-6 | - |
  | relax main 10001a8 | 8.8e-7 | 1.1e-6 | 5.4e-6, 5.0e-6 | 0 of 200 |
  | relax with the exact re-score | 7.9e-7 | 1.0e-6 | 5.2e-6, 4.9e-6 | 0 of 200 |

  With one image per particle the GEMM scorer reproduces RELION's cut; the subtomogram difference comes
  from adding 41 tilt images' diff2. The exact re-score is therefore a subtomogram change only. K=4
  (pdb_k4_5k_128, job 14971087): maps within 1.3e-5 over iterations 1-3, the cap of 400 does not bind.
- VDAM on optics groups of several image shapes (single particles; multioptics_k2_10k128, K=1; on main
  from 2026-10-05, the scores below are from the branch head 0d73d5b and are rerun on main): OPEN. CPU RELION is the reference on these rows only, because GPU RELION drops
  the coarse scale difference of a group on another box (ma-gilles/relax#12); CPU RELION is
  bit-reproducible, so a same-seed range has no width, and the rule the user set is the seed distribution:
  relax passes if its seed mean is not below the reference's by more than the standard error of the
  difference on the same seeds. Masked GT FSC-AUC: relax 0.64479 and GPU RELION 0.64484 over 12 seeds
  (paired difference -0.00005 +/- 0.00057); CPU RELION 0.65387, 0.64695, 0.64833, 0.64955, 0.64244,
  0.65417 on seeds 1-6 (seed 2 run twice, identical), relax 0.0044 below (standard error 0.0015) and GPU
  RELION 0.0052 below (0.0011), so the row does not pass under that rule. The reference choice (CPU or GPU
  RELION) is open until the user decides. The CPU advantage comes from the trajectory, not from the
  update: one RELION iteration continued from the same iteration-150 state (seed 1, from the CPU and from
  the GPU checkpoint) takes the same step on the CPU and on the GPU (difference 5-6 % of the step, step
  correlation 0.998, amplitude ratio 1.00 in every shell, poses 0.1 %, offsets 1 % in the second group
  from #12/#13), while the CPU and GPU states at iteration 150 differ. On seed 1 the masked score of CPU
  RELION and relax is equal to iteration 140 (0.6506 and 0.6505); the gap opens on the plateau after it.
  The lower rlnAveragePmax of CPU RELION in model.star is bookkeeping (0.816 against 0.837 from the same
  state while the particles' Pmax differ by 0.002). Such continuations must pass the step size: RELION
  --continue leaves is_3d_model uninitialised and may take 0.3 instead of 0.5 (ma-gilles/relax#23).
  At iteration 1, from the same start, relax with CPU RELION's translation
  defect emulated (ma-gilles/relax#13, diagnostic) reproduces CPU RELION (map 4.6e-4, 1 % of the other
  group's poses) and relax as written differs by that defect (4.1e-3; GPU RELION 7.8e-3); end to end the
  emulation scores 0.0008 +/- 0.0003 below relax over 10 seeds, so the defect is not what CPU RELION
  gains from. K=2 on this fixture collapses to one class in both programs and is a plumbing check only.
  Scores: `em_evidence/etvdam_scores_20261003/cryoet_vdam_20261001/{scores/mshape_0d73d5b,scores/mshape_seeds,mshape_cpu_extra,mshape_late,mshape_cont150_step}`.
  SPA VDAM with K>1 uses the same scorer. The exact re-score at both cuts landed as 880483a5. Seed 2 with
  its patch on base d4825f3 (ee2f296; jobs 15009958, 15009959; two runs each): et09 one-group 0.988372 and
  0.988242 inside RELION's 0.988241-0.988392 (main b44f33e had 0.988222 and 0.988187); et15 optics-group
  K=2 0.982055 and 0.982094 just below RELION's 0.982101-0.982157 (ranges overlap through an earlier relax
  run at 0.982124). Evidence: `em_evidence/.../cryoet_vdam_20261001/{et09,et15}_s2_ee2f296`. The
  diagnostic continuation is not valid from a checkpoint whose next iteration updates the sampling (9,
  19, ...): it does not carry the previous offset step into RELION's subtomogram rule (3.0 A against 4.25 A
  at iteration 10); the forward run is right. Evidence: `em_work/cryoet_vdam_20261001/{relion/etob2l_plain,plain_probe,audit/etob2l_plain_s1}`,
  summary in `em_evidence/etvdam_scores_20261003/cryoet_vdam_20261001/plain_probe_summary.json`.
- With one optics group RELION's subtomogram start-up seeds only class 1 (each group's first
  particle fills the 10-image quota, into class `position % K`), so K>1 VDAM on the etbench
  fixtures keeps classes 2..K empty in both programs (ma-gilles/relax#11). With one optics group
  per tomogram, as RELION 5 imports, every class is seeded; K>1 tomo VDAM is qualified on
  fixtures re-labelled that way (in progress, together with InitialModel on several optics groups).
  Evidence and job IDs: `/scratch/gpfs/CRYOEM/gilleslab/em_work/cryoet_vdam_20261001/HANDOFF.json`.

## PPCA dense speed and TF32 default (October 2, 2026)

The streamed PPCA engine's fused GPU stages are on relax main (`c6a83a8f`,
`751ae47a`): a 10076 q4/HP4 VDAM update takes 11.0 s on A100 and 4.9 s on H100
in fp32, against 41.5 s and 18.5 s for the September control (3.8-4.5x A100,
3.3-4.0x H100, every scoped PPCA gate passing). The four fp32 GEMMs then take
84% of the busy device time. The stream GEMMs now default to TF32 on sm_80+
(`--ppca-gemm-precision auto`; fp32 on V100 and as a switch), adopted because
eleven-state runs from GT and from CP4000 and 10076 runs stay inside the fp32
seed-to-seed spread on GT state FSC, pose accuracy, latent R^2 and
log-likelihood (table in [section 14](../math/vdam_ppca_algorithm.md)). TF32
makes a 10076 VDAM update 1.9x faster on H100 (2.5 s) and 2.4x on A100 (4.8 s),
about 8.6x the September control on A100. Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_ppca_dense_speed_20261001/HANDOFF.json`.

SPA PPCA speed work stopped here. The aligned TF32 GEMMs, the per-warp latent epilogue
and the register moment scatter followed. The particle stack is now read into
host memory by default when it fits (`--ppca-preread-images auto`, at most 25%
of the job's memory, as recorded in `run.json`). On a cold page cache,
reading each tile from disk cost up to 8 s per update. The tables give the
median warm VDAM update at the default precision (tf32):

| Fixture | September control | Now, H100 (job 14904148) | Factor |
| --- | --- | --- | --- |
| 10076 q4/HP4 | 18.5 s | 1.68 s (1.81-1.85 s without preread) | 11x |
| eleven-state q10/HP3 | 6.2 s | 0.75 s (0.76-0.81 s without preread) | 8.3x |

| Fixture | September control | Now, A100 | Factor |
| --- | --- | --- | --- |
| 10076 q4/HP4 | 41.5 s | 4.6-5.0 s | 8.3-9.0x |
| eleven-state q10/HP3 | 15.5 s | 2.1-2.6 s | 6.0-7.4x |

The A100 runs read from disk and shared the host.
At 10076 the device is busy 1.48 s per update on H100: GEMMs .58 s, moment
scatter .41 s, epilogue and posterior preparation .21 s, window projection .18 s
(nsys job 14900161). No remaining kernel is more than about 10% of an update.
A per-shell separable VDAM metric that would shrink the scatter was rejected on
science ([section 14](../math/vdam_ppca_algorithm.md)).

## Refinement readability example (October 1, 2026)

Recorded readability packages and their source-specific evidence (`/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T111111Z/final_source/docs/development/em_status.md#refinement-readability-example-october-1-2026`; mg6942 scratch outside the repository).

## PPCA coarse optimizer comparison (September 28, 2026)

The opt-in [PPCA momentum SGD and two-pass coarse route](../math/ppca_momentum_sgd.md)
are on relax main at `2abfe2b2`; VDAM and the existing engine remain defaults.
At noise level 1, two paired 600-update runs used one mean plus two loadings,
20,000 particles, radius 31, HP3, 29 shifts, batch 300 and positivity off.
The metric is per-state rigid+hand-fitted mean FSC over shells 1–15 on the
same fixed 600-particle evaluation subset. GT labels were used only to score
state maps, not during refinement.

| Seed | PPCA VDAM state FSC | PPCA momentum SGD state FSC | Median pose error, VDAM / SGD |
| --- | --- | --- | --- |
| 11 | .181 / .230 / .200 | .282 / .323 / .235 | 130.2° / 125.1° |
| 12 | .202 / .238 / .239 | .300 / .264 / .269 | 134.6° / 128.7° |

SGD has higher final fitted FSC in all six comparisons and avoids VDAM's late
high-band map-power growth. Neither optimizer recovers reliable poses or three
states; latent between-state R² remains at most .106. The archived native K3
VDAM run recovers state 2 in both seeds (FSC .884/.892) but is a different model
and protocol, so it is context rather than a matched optimizer arm. Whole-update
medians were 3.017/3.026 seconds for VDAM and 2.980/2.988 seconds for SGD
(seeds 11/12). Training used 2.074 allocated A100 GPU-hours. Full science
scorecard, registered plots, signed MRCs, exact jobs and reproduction commands
are in `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_momentum_gemm_20260928/science_compare/REVIEW.md`.

The integrated source passed local A100 smoke, all 28 medium-tier items and a
source-exact full-grid PPCA gate (24/24 float32 comparison fields, zero pose
flips). Medium job 14643773 passed its tier but its appended gate launcher
failed before execution; gate retry 14654414 passed. The two-pass route saves
residency but was slower than retained scoring at matched B150 (2.89 vs 2.17 s
warm); no dense CUDA/GEMM PPCA adapter was integrated. The next bounded science
check is the curvature-maximum frequency and per-band gradient distribution at
a saved checkpoint before changing the optimizer.

## Coarse K3 SGD comparison (September28,2026)

The opt-in momentum SGD/noise implementation has passed scoped numerical checks,
the medium tier and fresh integration smoke. Matched fixed-HP3, 600-update
noise-1 runs with positivity off recover state 2 in both seeds; states 0/1
remain poorly separated. The completed fixed-settings robustness matrix finds
condition-dependent recovery: preferred orientations improve the candidate's
three-state result, while high noise and nonzero shifts leave both arms with
poor poses. General robustness is not established. See the
[method](../math/momentum_sgd.md), [noise-1 scorecard](../math/momentum_sgd_science_scorecard.md),
[robustness scorecard](../math/momentum_sgd_robustness_scorecard.md),
[source and validation report](../benchmarks/momentum_sgd_coarse_20260928.md),
and [robustness report](../benchmarks/momentum_sgd_robustness_20260928.md).
Native VDAM defaults are unchanged.

## Dense GEMM coarse engine options (September 30)

`coarse_engine=auto|gemm_hybrid|gemm_dense` keeps `auto` as the production default
in standard EM and VDAM, including K>1. `gemm_hybrid` selects GEMM coarse
scoring followed by the existing pruned fine pass. `gemm_dense` is an explicitly
experimental no-pruning route over the full configured pose grid, with exact
per-image normalization; it is not a RELION-parity or performance-qualified
replacement. On the matched 5k-image, box-128, K4 os1 A100 update, warmed
whole-process times were 37.64 s for `auto` and 37.55 s for cached full dense
(job 14694770). Their minimum per-class one-update RELION FSC-AUC was
0.999999993 versus 0.963218, respectively. Neither the measured process tie
nor the divergent result supports using full dense by default. The isolated
kernel speedups are not end-to-end production speedups. Detailed source and
evidence are in the [integration record](dense_gemm_coarse_engine_plan_20260928.md).

## Opt-in dense GEMM K1 experiment (September 28)

The requested full dense grid experiment has exact two-sweep and lagged one-sweep
normalization, image/projection translation expansion, and serial/mixed/full
translation tiles. Exact three-iteration reconstruction closely agrees with the
native dense control; lagged absolute normalization overflows at iteration two.
Production defaults are unchanged. [Math and measured scope](../math/dense_gemm_experiment.md)
and the [detailed plan](dense_gemm_experiment_plan_20260928.md) link the evidence.
The final single raw-batch staging seam passes real GPU operand/upload checks
14632476. Medium14632505 passed all 27 items and 13 FSC comparisons; integrated
smoke14633325 passed all four items and three FSC comparisons. Both enforced
pinned comparisons passed; detailed receipts record the optional fixture skips.
The historical timing control used tomography scoring and a deprecated adjoint
schedule, so its speed ratio does not measure current SPA performance. A corrected
control now uses production-selected GEMM scoring and SPA CUDA row accumulation,
retaining every dense hypothesis. Its matched H100 benchmark 14634396 measured 20.3×/27.9× warmed dense-operator gains
at B100/B512. Full5000-particle, three-iteration run 14634727 measured 271.34 s versus
41.43 s process time, with worst map relative L2 difference 2.75e-5 and half-FSC
difference 1.55e-6. Both warmed profiles show no host transfers or CUDA allocations.
Final post-rebase smoke 14634842 and focused CPU checks passed on main d02b2c46.
These are controlled dense comparisons; historical production-runtime extrapolations
remain withdrawn. The optional experiment does not change production defaults.

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

## Feature gaps (moved from the README, 2026-09-29)

The RELION 5 options and behaviours relax does not support yet, row by row with what each would take, are
in [relion_feature_gaps.md](relion_feature_gaps.md) (the roadmap); this section records what is supported and
how it was qualified.

- Several optics groups: the default Refine3D (K=1) command runs them, including groups on other
  pixel sizes and boxes, with no settings (2026-09-30). The fresh K=1 pass always uses RELION's
  powerClass spectrum and exact BPref operands; the `RELAX_K1_RELION_POWERCLASS_SPECTRUM_NORM` and
  `RELAX_K1_RELION_EXACT_BPREF_OPERANDS` switches are retired. The `--firstiter_cc` iteration takes
  them too (each shape class's projection matrices, each image's noise row). Qualification on S3b
  (10k particles, 4.25 A/128 px + 5.44 A/112 px), `--no-firstiter_cc` (the reference is on the
  images' greyscale), same H100 node as MPI-scale-patched RELION 3x4 (job 14748349): map gate PASS
  (merged/half1/half2 0.9962/0.9940/0.9945 against two same-command RELION runs), masked GT band
  FSC-AUC 0.9950 vs RELION 0.9946, masked resolution 8.5 A both, wall 400 s vs 784 s (0.51x).
  Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_multioptics_default_20260929/score_7c3ddbe`.
  With `--firstiter_cc` (2026-09-30): group 2's (s = 1.12) Gaussian coarse window (56 px, r_max 25)
  lies between 2 r_max and 2 s r_max from iteration 2, where RELION's coarse kernel projects and
  shifts the rows beyond maxR at `i - window` (see "Multi-optics on another grid" below); the coarse
  GEMM operands now do the same (`helpers.projection.relion_coarse_relabel`, the window's Nyquist
  row included). S3b against two same-command MPI-scale-patched RELION 3x4 runs (they agree to 1.0):
  map gate PASS (merged/half1/half2 0.9989/0.9983/0.9982, job 14796180 at 30cfae8), masked GT band
  FSC-AUC 0.9954 vs RELION 0.9954, masked resolution 8.5 A both; wall 459 s vs 783 s (0.59x) on one
  H100 node (pair 14762768). GEMM scores match the fused coarse kernel to 2e-7 in the band
  (`tests/unit/test_relion_coarse_relabel_gpu.py`). Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_multioptics_default_20260929/score_30cfae8_fcc`.
  Class3D (K>1) runs several optics groups on one image shape, single-particle (2026-09-30) and
  subtomogram (et16_k2conf_optics2, `/scratch/gpfs/CRYOEM/gilleslab/em_work/cryoet_class3d_20260929`).
  The per-group noise table is never read as per-class (a [G, P] table with G == K was taken as one
  row per class, so every image backprojected with group 1's spectrum). Qualified relative to one
  optics group on the multioptics_k2same_10k128_20260930 fixture (K=2, one reference, --firstiter_cc,
  against MPI-scale-patched RELION 3x4): the CC iteration matches (maps 2.9e-5, one CC pose tie). The
  seed iteration's gap to stock RELION (maps 2.9e-3; 5.6e-3 with one optics group) is stock RELION's
  float32 backprojection: the double-accumulation RELION build differs from stock by the same 5.6e-3,
  and relax matches it at 6.6e-5/1.0e-4 (one optics group, iteration 2). InitialModel (VDAM, single
  particles and subtomograms) takes several optics groups on one image grid (2026-10-02): RELION's
  per-group start-up particles for the noise and the bootstrap (`relion_startup_positions`), each
  image scored with its group's noise row, the noise updated per group, the expected accuracy per
  group. Start-up against stock RELION non-MPI: every group's sigma2_noise to 4e-7, class maps to
  1.5e-8 (et15 with one group per tomogram) and exactly (SPA two groups). End to end, masked GT
  FSC-AUC (181e99d scorer) inside or above stock RELION's same-seed range at all three seeds on
  et09 with one group per tomogram (8 groups, K=1); the SPA two-group K=2 fixture collapses to one
  class in both programs (a plumbing check only). Subtomogram K=2 on et15 with one group per
  tomogram (4 groups; relax 0de4063 job 14835375, stock RELION r1 14817602 and r2 14835377, one H100
  each), population-weighted masked GT FSC-AUC relax / RELION same-seed range: s1 0.98391 /
  [0.98378, 0.98387] (above), s2 0.98202 / [0.98211, 0.98212] (9e-5 below; these relax runs predate the subtomogram expected-accuracy
  fix, re-run above), s3 0.98240 /
  [0.98234, 0.98242] (inside); class populations equal RELION's to 1e-3 at every seed; wall
  52542 / 48365 / 54054 s against RELION 35984-39046 s (1.33-1.41x). Groups on other grids stay
  refused (etw's multishape K>1 route). Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/cryoet_vdam_20261001`
  (`scores/et15_og_k2`).
  Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_k2optics_20260930`.
- Class3D (K>1) on optics groups of several image shapes (2026-10-01): each half is scored per shape
  class and merged (`optics_shapes.merge_k_class_outputs`), as K=1 is. Qualified on
  multioptics_k2_10k128_20260930 (K=2, 128 px at 4.25 A and 112 px at 5.44 A, --firstiter_cc), one
  seed (1), against stock RELION 5.0.1 on the CPU as the reference, whose classification pass applies
  the scale difference (ma-gilles/relax#12): masked GT FSC-AUC 0.4659 vs 0.4612, class accuracy 0.7134
  vs 0.7105 (two identical CPU runs); relax-vs-RELION per-class cross FSC-AUC 0.970/0.976. RELION's GPU run, which skips the
  scale difference in pass 1, reaches 0.3880/0.6102. VDAM on several shapes is not wired yet; its
  engine-level loop will use `optics_shapes.shape_class_engine_inputs` and `merge_k_class_engine_results`.
  Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_k2optics_20260930/score_twoshape_68d7ea1_out.json`.
  Same-seed reproducibility (2026-10-04, etw): same-code relax runs of this fixture end on different
  trajectories (it25 class maps 2.6e-2 apart, class assignments equal for 99.5%; bindw jobs 14972806,
  14973509), and so do same-seed RELION runs, so a pair on this fixture is compared against a same-seed range
  on both sides, not against one run. relax (main b4845c2, jobs 14978430, 14986271): iteration 1's E-step
  outputs are identical; its M-step shell sums differ in the last bits (tau2 shell sum 1e-7, class maps
  1.8e-7 relative), and at iteration 2 Pmax differs for 95% of the images (max 4e-6), the significant count for
  10-16 images and one best rotation. `RELAX_EM_DETERMINISTIC_REDUCTIONS=1` does not change this. Two RELION
  5.0.1 GPU runs (non-MPI, seed 1, the oracle command, 5 iterations, job 14978654), run a against run b:

  | Iteration | Significant count differs | Pmax differs (max) | Poses differ | Class maps, relative L2 |
  | --- | --- | --- | --- | --- |
  | 1 | 0 | 0 | 0 | 1.7e-8 |
  | 2 | 92 | 1414 (4e-6) | 1 | 7e-7 |
  | 3 | 19 | 6803 (9e-5) | 1 | 3e-6 to 5e-6 |
  | 4 | 3 | 9385 (0.027) | 0 | 2e-5 |
  | 5 | 3 | 7347 (0.32) | 2 | 0.9e-3 to 1.3e-3 |

  Class assignments are equal in all five iterations. The split is GPU last-bit differences in the
  iteration-1 reconstruction amplified by this fixture's near-ties, in both programs. Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_onengine_20260926/mshape_repro_20261004`.
  Resident stable windows on another grid (2026-10-03): a pass whose images are on another grid than the
  reference takes the stable-window plan on its own image grid, and its reference-model cube takes its own
  physical class (`resident_pass2._stable_reference_volume_class`); before, every such pass kept the logical
  window and recompiled per current size. Same fixture and seed, 25 iterations, cold: 754 -> 555 s, XLA
  compile 485 -> 275 s; iteration-2 maps differ from the control by 2.6e-7; masked GT FSC-AUC 0.4662 vs
  0.4664. Same-node pair on H100 (job 14945409): relax 485 s, stock GPU RELION 715 s. The remaining
  compile is the chunk posterior, M-step block and statistics programs (64-79 signatures) and eager
  primitives on new shapes.
  Distance to stock CPU RELION on this fixture (2.0e-3 at iteration 1, 5e-3 at iteration 2) is RELION's
  CPU translation defect (ma-gilles/relax#13): its CPU path phase-shifts trial translations with the model
  box, so a group on another box is shifted by box_g / ori_size times the nominal offset. With relax's
  group translations scaled the same way (diagnostic, not landed) iteration 1 agrees to 2.6e-4 with 14
  group-2 rotations differing instead of 476; the same-shape two-optics control agrees to 3.3e-4. relax
  keeps the physical shift, as RELION's GPU path does.
- Optics features in Refine3D (K=1): CTF-premultiplied particles
  (`rlnCtfDataAreCtfPremultiplied`), beam tilt and odd Zernike aberrations (image
  demodulation), even Zernike aberrations and anisotropic magnification (`rlnMagMat*`).
  Class3D (K>1) takes all four, including RELION's average-CTF² correction of
  data_vs_prior for premultiplied data (`setAverageCTF2`); magnification is qualified against
  stock RELION on the CPU (below).
  InitialModel (single particles, 2026-10-02) takes premultiplied particles, odd and even
  aberrations: the bootstrap CTF with each group's even Zernike gamma, the expected accuracy with the
  trials' optics-table CTFs, the M-step's average-CTF² SSNR correction (RELION's updateSSNRarrays
  oracle takes avgctf2); magnification and subtomograms stay refused. Iteration 1 against stock RELION
  non-MPI: map relative L2 7.1e-5 (premult), 1.4e-5 (btilt), 5.5e-7 (evenz). End to end (relax c26e7b7,
  jobs 14859714/16/18; stock RELION r1 14859713/15/17, r2 14862958-60; H100; 181e99d scorer, same-seed
  pairs, `em_work/cryoet_vdam_20261001/scores/optics_k1_ss`), masked GT FSC-AUC relax / RELION range:
  premult 0.4404 / [0.4395, 0.4406], 0.0566 / [0.0570, 0.0636], 0.4456 / [0.4455, 0.4463]; btilt
  0.0714 / [0.0598, 0.0895], 0.4442 / [0.4443, 0.4444], 0.4403 / [0.4402, 0.4402]; evenz 0.3787 /
  [0.4390, 0.4468], 0.4436 / [0.4434, 0.4437], 0.4398 / [0.4396, 0.4396] (seeds 1-3): inside or above
  at two of three seeds per feature. premult s2 and btilt s1 fail to converge in BOTH programs (about
  0.06 against 0.44), so they are weak evidence and the verdict rests on the converged seeds: premult
  s1, s3 inside; btilt s2 1.1e-4 below, s3 above; evenz s2 inside, s3 above. evenz s1 (first run 6e-2
  below, RELION's own r1-r2 agreement only 0.60) is a chaotic seed: two relax same-seed repeats (job
  14869744) score 0.44279 and 0.44277, inside RELION's range. Walls equal RELION's (464-669 s vs 436-570 s).
  Qualification, 2026-09-30: 10k/256 fixtures under
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/optics_*_k1_10k256_20260929`, one relax run
  (cand_ccb229f, H100) against same-command, same-seed RELION 5.0.1 runs (patched MPI build,
  3 ranks x 4 threads, H100); map gate from `scripts/score_k1_map_gate.py`, masked GT FSC-AUC
  from `scripts/masked_fsc.py` (mask `synth_k1_50k256_snr05_c1`).

  | Feature | Status | relax GT FSC-AUC | RELION runs | Map gate |
  | --- | --- | --- | --- | --- |
  | Premultiplied (job 14744731) | qualified | 0.94516 | 0.94464, 0.94460 | pass |
  | Even Zernike (14744732) | qualified | 0.94908 | 0.94891, 0.94829 | pass |
  | Beam tilt + odd Zernike (14720565, 14735008) | provisional | 0.94858, 0.94891 | 0.95462, 0.94915, 0.94901 | pass |
  | Magnification (14744733) | provisional | 0.94954 | 0.95099, 0.94966, 0.94985 | pass |

  Beam tilt: relax is 1.0e-4 and 4.3e-4 below the three-run RELION band; relax's own spread is
  3.3e-4, and its map agrees with the third RELION run at 0.998 (masked merged FSC band AUC).
  Its phase matches RELION's code in unit tests. Magnification: relax is 1.2e-4 below the
  three-run band, with 0.998 map agreement against the third RELION run. Whether sub-5e-4 gaps
  where relax's map reproduces a RELION run count as ties is with the user.

  Class3D K=2 qualification, 2026-09-30: 10k/256 fixtures
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/optics_*_k2_10k256_20260930` (first 10k of
  synth_pdb_k2_50k256_snr05), the synth_pdb_k2 Class3D command at seed 42; one relax run
  (cand_76ae34a, job 14795328) against four same-command RELION 5.0.1 non-MPI runs, two float
  builds (14774524-7) and two double-precision back-projection builds (14795376-7). Masked mean
  GT FSC-AUC (mask `synth_pdb_k2_50k256_snr05_c1`, classes matched to GT by Hungarian
  assignment; scorer em_work/relax_bench_k1plus_20260925/tools/score_class3d.py); scores in
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_bindw_20260930/classes/<feature>/score_76ae34a.json`.

  | Feature | Status | relax | RELION float | RELION double | class accuracy relax / RELION |
  | --- | --- | --- | --- | --- | --- |
  | Premultiplied | qualified | 0.288584 | 0.284788, 0.287513 | 0.287528, 0.288820 | 0.9531 / 0.9526-0.9567 |
  | Beam tilt + odd Zernike | provisional | 0.208885 | 0.208436, 0.208290 | 0.209495, 0.209034 | 0.8439 / 0.8433-0.8448 |
  | Even Zernike | provisional | 0.213436 | 0.213352, 0.213999 | 0.213522, 0.214689 | 0.8494 / 0.8493-0.8536 |
  | Magnification | qualified against RELION CPU (below) | 0.208381 | 0.213430, 0.215135 | 0.214712, 0.214480 | 0.8373 / 0.8471-0.8543 |

  Beam tilt and even Zernike are inside the four-run band and 1.5e-4 and 8.6e-5 below the
  double-accumulation pair.

  Magnification (K>1), qualified against stock RELION CPU (2026-10-02). RELION's GPU
  classification scores pass 1 without the optics group's magnification and scale difference
  (predefined coarse projector plans, relax#12), so the GPU runs above are not the reference;
  relax applies them in every pass, as RELION's CPU path and its GPU auto-refine do. Reference:
  stock RELION 5.0.1 f2c1a38 on the CPU (no `--gpu`, `--j 32`; deterministic, `--j 24` gives the
  same run bit for bit); relax cand_460ff32 (main fd64ead with magnification accepted); jobs
  14835479, 14835480, 14852027 (relax), 14824531, 14835481, 14835482, 14852025, 14852026 (RELION
  CPU); scores in `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_bindw_20260930/classes/mag/`
  (`score_seeds.json`, `score_seeds2_relax.json`, `score_seeds3.json`). Benchmark multi-seed rule:
  every relax run is inside or above the across-seed RELION band [0.205033, 0.211092], and relax
  is not below its same-seed RELION run at three of five seeds. It is below at s42 and s29; the
  worst seed is s29 at -1.7e-3. The reference is RELION CPU, not RELION GPU, because of relax#12.

  | Seed | relax (masked mean GT FSC-AUC) | RELION CPU | relax vs same-seed RELION |
  | --- | --- | --- | --- |
  | 42 | 0.206533, 0.207609, 0.209553 | 0.211092 (repeated) | below (best run -1.5e-3) |
  | 29 | 0.209362, 0.209036 | 0.211077 | below (-1.7e-3) |
  | 53 | 0.211691, 0.211337 | 0.210008 | above |
  | 61 | 0.210088 | 0.205033 | above |
  | 67 | 0.208088 | 0.205250 | above |

  Iteration-1 poses: relax and RELION CPU agree for 98.6-98.9% of particles at each seed. RELION
  disagrees with itself at the same level on this data: its GPU auto-refine K=1 (magnification
  applied, equal to relax K=1 in 100% of poses) and its CPU K=1 agree for 98.45% (jobs 14812274,
  14851745), against 99.24% on the even Zernike fixture, where RELION CPU is 2.4-3.7e-3 below
  RELION GPU and relax (0.210942 vs 0.213352-0.214689).
- CTF: relax evaluates RELION's CTF rows (`CTF::getFftwImage`, including the per-particle
  `rlnCtfBfactor` and `rlnCtfScalefactor`) in its own host float64 code
  (`relax/relion/relion_ctf.py`), cast to float32 before GPU scoring. Production CTF no longer
  calls RELION code; the RELION binding is only the unit-test oracle
  (`tests/unit/test_relion_ctf_formula.py`).
- Follow-up: the exact-CTF device row cache (`relion_ctf._exact_ctf_device_rows`) still keeps
  float64 rows on the GPU, and the elementwise ops on them run in float64 before the cast to
  float32. They are cheap. Moving them to float32 is a parity-qualified change that has not
  been made yet.
- Refused permanently, with a clear message: `rlnCtfDataAreCtfCorrected` and several different
  `rlnMtfFileName` values.
- Subtomogram particles (RELION 5 2D stacks) run the first-iteration cross-correlation
  (`--firstiter_cc`) in Refine3D and Class3D since 2026-10-01 (see "Subtomogram InitialModel
  and first-iteration CC" below); its end-to-end qualification against RELION's default command
  is running.
- Subtomogram InitialModel (`relax initial_model --ios`) takes several optics groups on one image
  grid (above). RELION's subtomogram VDAM with one optics group leaves classes 2..K empty (below), so
  K>1 tests use one optics group per tomogram, as RELION 5 imports (et15_ogtomo; the recovar tomo
  simulator writes that by default since recovar 24e3fa27b).

## RELION binding removal (started 2026-09-29)

User rule (2026-09-29): production code must not rely on RELION code. The compiled binding
(`relax/relion_bind`, `_relion_bind_core`) stays only as the oracle of unit tests, diagnostics
and parity tools. Each port keeps RELION's exact semantics and gets a CPU test against the
binding (integers exact, floats at a measured relative tolerance). A lint test will fail on any
import of `relion_bind` outside `relax/relion_bind`, `relax/diagnostics`, `relax/reference` and
`tests`; until the last port lands it carries an allowlist that shrinks with each port.

Production call sites at relax main `93bf20e`; the status column records each port. Every port has a CPU test against
the binding in `tests/unit/test_*_vs_relion_bind.py` (and `tests/unit/initial_model/`).

| Module | Function | Binding call | Runs that use it | What it computes | Status |
| --- | --- | --- | --- | --- | --- |
| `symmetry.py` | `_operators_float64` | `get_symmetry_operators` | every run (C1 hash/point-group code; operators for non-C1) | RELION `SymList` ordered L/R operators and point-group code | ported (`symmetry._operators_float64`) |
| `sampling.py` | `_get_relion_grid_metadata` | `get_healpix_directions` (C1), `get_healpix_sampling_metadata` (non-C1) | every run | HEALPix NEST directions (rot, tilt), retained pixel ids, psi grid; asymmetric-unit pruning for non-C1 | ported (`healpix_sampling.py`) |
| `sampling.py` | `get_relion_rotation_grid`, `_get_relion_rotation_grid_eulers_float64` | `get_coarse_orientations` | every run | the coarse (direction, psi) Euler grid in RELION order | ported (`healpix_sampling.py`) |
| `sampling.py` | oversampled-orientation builder | `get_oversampled_orientations_batch` (NumPy fallback for C1; required for non-C1) | every run (fine pass) | `HealpixSampling::getOrientations` children with random perturbation | ported (`healpix_sampling.py`) |
| `sampling.py` | inverse scoring matrices | `euler_angles_to_inverse_matrices` (NumPy fallback exists) | every run | `Euler_angles2matrix` then `Matrix2D::inv` | ported (`healpix_sampling.py`) |
| `sampling.py` | device scoring perturbation | `euler_angles_to_matrix` (NumPy fallback exists) | every run with perturbation | perturbation Euler matrix | ported (`healpix_sampling.py`) |
| `sampling.py` | `_relion_rnd_unif_scaled_first_draw` | `vdam_rnd_unif_range_sequence` (glibc `ctypes` fallback) | every run (perturbation draw) | `init_random_generator(seed)` then `rnd_unif(low, high)` | ported (`helpers/relion_random.py`) |
| `relion/relion_projector_setup.py` | `reference_to_relion_projector_half_maps_and_power` | `compute_fourier_transform_map` | `--projector_setup_backend native` and geometry the JAX path does not take (odd box, padding > 2, interpolator 0) | `Projector::computeFourierTransformMap` | removed: one device setup, other geometry refused; RELION's transform is `diagnostics/native_projector_setup.py` |
| `reconstruction/regularization_relion.py` | tau2 from reference | `compute_fourier_transform_map` | callers that pass no `projector_power_spectrum` | projector power spectrum | ported (device projector power) |
| `relion/relion_ctf.py` | exact CTF | `get_ctf_images_batch` | CTF images | `CTF::getFftwImage` | ported by optics (0cbfb8b) |
| `helpers/expected_accuracy.py` | `estimate_relion_expected_accuracy_from_prepared_inputs` | `vdam_expected_angular_errors` | auto-refine, Class3D, InitialModel, tomo | `MlOptimiser::calculateExpectedAngularErrors` | ported (`helpers/relion_expected_accuracy.py`, CTF from `relion_ctf.relion_ctf_fftw_half`) |
| `helpers/expected_accuracy.py` | `relion_auto_refine_half_orders`, `relion_half1_trial_order`, `relion_class3d_trial_layout` | `auto_refine_randomise_half_orders_mt19937` | auto-refine, Class3D | `std::shuffle` with `mt19937(seed + iter)` | ported (`relion_random.shuffled_orders`) |
| `vdam/subset_schedule.py` | subset shuffle | `vdam_randomise_particles_order` | InitialModel | `std::shuffle` with `mt19937(seed + iter)` | ported (`relion_random.shuffled_orders`) |
| `vdam/iteration_loop.py` | `refresh_tau2_from_projector_power` | `vdam_projector_power_spectrum` | InitialModel | `MlModel::setFourierTransformMaps` tau2 | ported (device projector power) |
| `vdam/bootstrap_iref.py` | `compute_bootstrap_iref_via_cpp` | `vdam_bootstrap_iref` | InitialModel | random-angle bootstrap reconstruction (ml_optimiser.cpp:3127-3205) | ported (`vdam/bootstrap_reconstruction.py`, now `compute_bootstrap_iref` / `postprocess_bootstrap_iref`) |
| `vdam/bootstrap_iref.py` | `postprocess_bootstrap_iref_via_cpp` | `vdam_postprocess_initial_iref` | InitialModel | blobs, low-pass and soft mask with the C `rand()` stream | ported (`vdam/bootstrap_reconstruction.py`, now `compute_bootstrap_iref` / `postprocess_bootstrap_iref`) |
| `vdam/mstep_single_class.py` | `_get_bindings` and the step-by-step path | `vdam_reweight_grad`, `vdam_first_moment`, `vdam_second_moment`, `vdam_apply_momenta`, `vdam_update_ssnr_arrays_from_bpref`, `vdam_reconstruct_grad` | InitialModel (module required by the default JAX transaction; step path only with dumps/replay or `use_native_transaction=False`) | VDAM M-step | done: production runs the transaction; RELION's steps are `diagnostics/vdam_native_mstep.py` |
| `relion/relion_vdam_mstep.py` | `_first_moment_initializes`, `relion_vdam_m_step_host` | `vdam_first_moment_initializes`, `vdam_m_step_transaction` (FFT grid < 16) | InitialModel | serial first-moment sum test; small-grid M-step | ported (serial host sum; grids < 16 refused) |
| `commands/ppca_initial_model.py` | `source_identity` | module file hash | PPCA initial model | provenance hash of the binding | removed (no binding in provenance) |

`commands/build_relion_bind.py` builds the oracle and is a tool, not a production path.

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

## One engine: removal TODO

User decision (2026-09-26): relax keeps one pass-2 engine, the device-resident one
(`relax/sparse_pass2/resident_*.py`), as RELION keeps one algorithm. Every other pass-2
engine or route is deprecated: its docstring says so, and a run that routes a pass to it
logs one `DEPRECATED engine` warning per engine, pass kind and reason
(`relax.sparse_pass2.engine_record.warn_deprecated_engine`), next to the per-iteration
`pass2_engine_trajectory` entry. They are removed in the order below once resident covers
what still routes to them. Inventory and line estimates (relax bc6d3e1):
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_onengine_20260926/PLAN.md`.

| Deprecated engine or route | What still routes to it on main | Resident work needed |
|---|---|---|
| (none) | Dense `run_em`, `dense_big_jit.py`, `k_class.run_dense_k_class_em` and the per-image reference route were the last; removed 2026-10-03 (item 6) | - |

Removal order:

1. Tests first (done 2026-09-26): `tests/helpers/relion_estep_reference.py` restates RELION's
   GPU fine pass in NumPy float64, and `tests/unit/test_resident_relion_reference.py` pins the
   resident drivers to it on the 8x8 fixtures: K=1 at current size 6 and the full box, K=2/3
   Class3D, wide translation grids, VDAM's residual backprojection, zero oversampling, C4 and
   the local fine pass. Sibling-engine agreement cannot validate a convention, so the tests
   that pin resident against compact or exact local are deleted with the engine they compare
   against.
2. Compact, with the K=1 exact-local adaptive route and the dispatch fallback (done
   2026-09-26, about 22k lines of `relax/`): a configuration the resident checks refuse is now
   an error, subset and focused replays take RELION's atomic Wavg arithmetic (user decision),
   the compact-only diagnostic dumps are retired, and pass 2 needs a CUDA GPU (user decision:
   no CPU back end; CPU correctness runs on the NumPy reference). The one memory refusal left is
   a smallest chunk that does not fit the device (the joint chunk planner, 4c3b2af, shrinks
   every larger plan to fit).
3. The full-box final pass on resident local (done, 1178448), then the parent probe.
4. The VDAM exact-local route (done 2026-09-27: `vdam/sparse_pass2_estep.py`, `k_class.run_local_k_class_em`,
   `--pass2_engine local/local_segmented` and the four options only it read are removed; the
   adaptive route is VDAM's only E-step route, and a configuration it refuses is an error).
5. The exact local engine (done 2026-09-30, about 24k lines of `relax/`): the parent probe runs on the
   resident local pass, and `local/local_em_engine.run_local_em_exact`, the `local/` modules only it reached
   and the helpers, diagnostics and tests that served it are removed; `local_layout` and the M-step row
   helpers in `local_backprojection` stay.
6. Dense `run_em` and the per-image reference (done 2026-10-03, about 10k lines with their tests): `dense/em_engine.py`,
   `dense/dense_big_jit.py`, `reference/sparse_pass2.py`, `scoring/score_constraints.py`, `k_class.run_dense_k_class_em`, the dense
   adaptive pass 2 and its fallbacks, the `_m_step_block_*` kernels and the direct routes in `half_scoring` are removed. Every
   global pass runs `run_dense_k_class_em_adaptive` (pass 1 in `significance.py`, pass 2 on the resident engine); `sparse_pass2=False`
   is refused, `scripts/run_k_class_parity.py` refuses a replay without `--adaptive-2pass` and always runs the production pass 2 (x-half M-step, noise sums), and `RELAX_K1_DENSE_PASS2`,
   `RELAX_K_CLASS_DENSE_PASS2*`, `RELAX_K1_SKIP_SIGNIFICANCE_PRUNING` and `RELAX_DISABLE_SPARSE_PASS2` are retired. The dense block
   scorers stay as test references in `tests/helpers/dense_block_scores.py` and `tests/helpers/dense_posterior_reference.py`.

Tomography (S4) runs only on the resident engine (`compute_tilt_pass2_stats_resident`)
and pins none of these.

One coarse path for every K (done 2026-10-02, speedw; team-lead decision: pass 1 is CUDA-only, like
pass 2): every pass 1 scores RELION's exact coarse operands in the pass-1 program, the Gaussian passes
with the coarse GEMMs and the `--firstiter_cc` passes with RELION's coarse CC.
`relax.scoring.significance._require_exact_pass1_operands` refuses a pass without the supplied RELION
texture projector, half-spectrum float32 scoring or the custom CUDA backend, and the batch loop refuses
data without RELION's CUDA image preprocessing. The generic dense scorer (`_score_block`, `_add_priors`,
the score cache, the generic preprocessing and operands, the manual and dense coarse projectors), the
`relion_exact_coarse` switch and the switches that selected those paths are removed (retired in
`relax/renamed_environment.json`). The generic scorer's arithmetic stays as a float64 test oracle
(`tests/helpers/generic_coarse_reference.py`, `tests/unit/test_pass1_program_generic_reference.py`), and
`tests/helpers/exact_pass1_harness.py` runs pass 1 on CPU for the significance unit tests. The generic
kernels of `relax/scoring/scoring.py` that only dense `run_em` used left with it (item 6).

Pass 1 on stable Fourier-window shapes: tried 2026-10-03, no gain, not landed (speedw; team-lead decision).

Stable Fourier-window quantum 16 for Class3D and auto-refine (default 8): tried 2026-10-04, not adopted (etw;
team-lead decision). Same-node cold H100 pairs through `RELAX_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM=16`:
multishape Class3D K=2 10k/128 475 -> 429 s (one pair, job 14954459, real compiles of 0.2 s or more 184 -> 147); pdb
K4 50k/256 Class3D 421/418 -> 407/403 s (two rounds with the GPUs swapped, job 14971138, real compiles 63-64 -> 56-57,
maps inside the same-quantum repeat spread of 2e-3 to 6e-3); K=1 auto-refine on 10k/256 351/347 -> 351/348 s (job
14971165, real compiles unchanged), with one quantum-16 run 7.3e-3 from the other three in the merged map against
4.8e-4 between the two quantum-8 runs, which two runs per quantum cannot attribute. Peak device memory (nvidia-smi)
was equal at 80 GB; 16 GB was not measured. A 3.5% gain on K4 and none on K=1 does not justify a second quantum scoped
to one workflow.

Eager single-primitive call sites as jitted programs, the four not already on main (data_vs_prior tail mask,
scale-correction pixel mask, per-image noise rows, `_pad_accumulator_to_class`): tried 2026-10-03/04, not landed (etw;
team-lead decision). They remove about 220 sub-0.2 s compile events, about 5 s of a 480 s cold multishape Class3D K=2
run (call-site census, job 14955029; it3 maps equal to 7e-7, medium 14955020 pass before the rebase): 1% is below what
a pair resolves and not worth a medium run. The window gathers and the exact CTF gather of the same series are on main
as 65f16e5 and 54cbb65.

Exact cut in the single-particle pass 1: tried 2026-10-04, not landed (speedw; team-lead decision). Pass 1 scores
with the same GEMM scorer and binds `max_significants` for long (plain 10k K=1: every image at the cap of 100 until
about iteration 30; ribosome K4: 30-84% of images in iterations 1-20), but with one image per particle the GEMM cut
already reproduces RELION's: plain 10k K=1, iteration 2 from scratch, relax main against RELION it2 map 1.1e-6 and
BPref 5.4e-6 / 5.0e-6 with RELION against RELION at 7.4e-7 and 3.9e-6 / 3.7e-6, 0 of 200 poses differing; the
re-score changed nothing measurable (etvdam, jobs 14974373 and 14974374). It re-scored 0.02-0.6% of the samples and
cost +7% on plain 10k K=1 (208 -> 223 s) and +26% on ribosome K4 100k (1552 -> 1954 s), same node, one H100 each
(job 14974269). The subtomogram pass needs it because it sums about 40 tilt images' diff2.

Pass-1 batch loop (2026-10-04, speedw; relax 0ae37fd, 54cbb65, 4efac62, 3636596): after the score program, one
program per batch (`coarse_publication.coarse_support_posterior`) forms the K=1 RELION-order log weights, the float32
posterior, the winner, Pmax and the rotation mask; the device CTF row gather is one program
(`relion_ctf._gather_ctf_rows`); a batch's host read-backs (`_publish_batch` in
`significance._compute_k_class_significance_batched`) run once the next batch's operands are on the device and
before that batch's score program, and the image preprocess kernel's finite check is read at the end of the loop
(`kernels.deferred_relion_preprocess_checks`), so the device scores a batch while the host prepares the next. That
check's queue belongs to the loop: a failure names pass 1 and each batch with an invalid image (batch number, positions
in the pass, dataset image range; the kernel reports a count per call), it is raised before pass 1 returns, and a loop
that fails for another reason drops its unread checks. The check reads the soft-mask background, so it sees a
non-finite pixel there, as the per-call check did. A dump
batch publishes at once. Outputs are unchanged: fixed-state replays (plain 10k K=1 it130 -> 131; ribosome it150 ->
151 at K15, K4, K1) give identical per-particle `rlnNrOfSignificantSamples`, Pmax and class. Measured: median
60-image coarse batch 73-98 ms -> 42-51 ms (one A100, it130 -> 131); plain 10k/256 K=1 full VDAM run on one H100
node, both arms concurrent, 257 s -> 221 s with the first three commits (job 14957274; no RELION arm in that job).
K15 is GPU-bound in pass 1 and moves by about 2%.

Non-finite pixels inside the particle mask (2026-10-04, etw; relax#16): the preprocess kernel counted an image as
invalid only from its soft-mask background sums, which read the pixels outside the mask, so a NaN inside it entered
the E-step and each workflow stopped later from a different sum, naming no image (refine after a whole iteration).
The kernel now also counts every image with a non-finite pixel anywhere (`relion_count_nonfinite_images_kernel`,
with or without a mask) and returns the smallest batch position of an invalid image beside the count. An eager call
reads that count itself and names the image: at once outside a deferred block ("dataset image N (position p of the
batch)", from the batch `prepare_batch_preprocess_operands` noted, which covers pass 2 and the subtomogram coarse
pass), at the end of the loop inside one (pass 1, with its batch label). A call inside a compiled program keeps the
kernel's own fail-closed check. The start-up noise estimate names a non-finite image too (single particles and
several shapes by dataset image, VDAM by read order, subtomograms by particle and tilt image). GPU tests: the kernel
cases in `tests/unit/test_cuda_relion_preprocess.py`, the pass-1 loop with the pixel inside the mask, and one case
per workflow (`tests/integration/test_non_finite_image_stops_before_output_gpu.py`: refine, Class3D, VDAM and
subtomogram refine stop on a NaN at an image centre, name the image and write no map). The native sources changed:
natives built before this commit do not match.

Late plain 10k K=1 iterations after these changes (py-spy, 60 s, relax 3636596): pass 2 41% of the main thread, pass 1
30%, the VDAM M-step 12%, the expected-accuracy estimate 7%. Open: pass 2 waits 11 s of the 60 s for the
iteration's images to be read a second time (pass 1 already read them); the expected-accuracy estimate rebuilds a
host float64 projector from the references at every iteration.
`significance._compute_k_class_significance_batched` can score on a quantized physical window (runtime current size,
zero-weight capacity rows; `stable_fourier_window_shapes`, default off). Turned on as the one path, a K1 noise1 5k
standalone refine (oversampling 1, healpix 3, 12 iterations, cold cache, A100, JAX_LOG_COMPILES) compiled 2350
programs in both arms, wall 895 vs 911 s: the coarse current size stays 38 (physical 40) through the run, so pass 1
compiles once per signature (_coarse_pass1_blocks x4, exact operands x4, powerclass x10, window operands x10: about
10 s of 330 s compile). etw's multishape census agrees (VDAM K1 pass 1 x16 / 6.8 s of about 170 s; Class3D not in the
top 25). The compile cost sits in eager JAX primitives on new shapes (1362 compiles, 121 s: broadcast_in_dim x378,
add, convert_element_type, gather, ...) and the pass-2/M-step programs.

Pass 1 as one program per image batch (speedw, 2026-10-02): with the cached coarse GEMM scorer, every
class and rotation block of an image batch is scored, given its priors and reduced (class and global
logsumexps, best pose and class) in one jitted program
(`relax.scoring.significance._coarse_pass1_blocks`) instead of the per-class, per-block eager loop.
H100 replay of ribosembly K15 50k it150 -> 151 (job 14877086): warm E-step 33.6 -> 30.4 s (RELION it151
40.1 s), K4 25.7 -> 25.1 s, K1 10.0 -> 9.9 s; every image's significant-sample count equal to the loop's.
The same program folds one class and rotation block per call on that block's projection when the
projection cache does not fit the budget, and scores the exact-operand normalized-CC (`--firstiter_cc`)
passes with RELION's coarse CC. Since 2026-10-02 it also tracks each class's runner-up
(`return_class_second`), serves score-only passes, and the K=1 `--firstiter_cc` coarse-tree top-2 rescore
(margin: the run option `--firstiter_cc_tree_rescore_max_margin`, default 4e-6 for K=1) scores the
exact operands with the CC GEMMs. RELION-band check (control cf583b3 vs a2cc776, K=1 auto-refine,
`--firstiter_cc`, seed 1775735620; c1band_a2cc776, jobs 14904272/14904273): the coarse-probe winners
near the 4e-6 ties differ (noise1_50k 63 vs 14 winner changes, pdb_k1_100k 18 vs 2), but the fine pass
lands identically (run_it001_data.star poses equal between the arms; versus RELION's it001, 5 of 50000
and 0 of 100000 particles differ in both arms, RELION-vs-RELION 3 of 50000); both arms meet the map
gate against every RELION run, masked GT FSC-AUC noise1 0.679361 vs 0.679342 (RELION 0.679354),
pdb 0.692246 vs 0.692217 (RELION 0.692196). The per-class loop went with the generic scorer
(One coarse path for every K, above).
Removed (2026-10-02, team-lead's deletion list D1-D5, D7), each switch refused when set
(`relax/renamed_environment.json` "retired", checked when relax is imported): the fused per-block pass-1
program (`RELAX_PASS1_FUSED`); the fused-projector coarse scorer family (`RECOVAR_K1_COARSE_FUSED_PROJECTOR`,
`_CANONICAL_REDUCTION`, `_SINGLE_LANE_CANONICAL`, `_NATIVE_ATOMIC_REDUCTION`, `_PREHALF_WEIGHT`,
`_MULTISTREAM_WORKERS`), the native-texture and the rectangular coarse scorers and the backend
selector (`RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=0` is refused); the certified K=1 coarse GEMM hybrid with its
compact posterior, device transaction, real-cross certificate, row partition, CUDA posterior transaction and
runtime-prefix dump (modules `coarse_gemm_hybrid`, `coarse_partition`, `coarse_device_*`, the publication
path, the certificate scoring, their tests and the combined-true200 gate tooling); the coarse selector
audit. The fused projector kernel (`relion_coarse_diff2_projector_f32`, the subtomogram test reference),
the rectangular kernels (f32, f64) and `coarse_publication._posterior_statistics` stay.
Removed (2026-10-02, D6): the paired direct-vs-GEMM coarse score capture
(`RELAX_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR`, `_DIAGNOSTIC_ORIGINAL_INDICES`, its call scopes and manifests)
and the all-particle streaming rescore (`RELAX_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR`, `_STREAM_TOPK`,
module `coarse_gemm_streaming`), with the `coarse_gaussian_gemm_qualification` stats block; both forms of
each switch are retired. The GEMM-vs-direct comparisons stay as unit tests of the scorer and the pass-1
program (`tests/unit/test_coarse_gaussian_gemm_macro.py`, with the summary helper in
`tests/helpers/score_diagnostics.py`). `RELAX_SIGNIFICANCE_DUMP_*` (the RELION first-divergence dumps)
reads the targets' pre-prior and with-prior scores from the pass-1 program (`dump_rows`; the dump records
`score_capture_mode="pass1_program_target_rows"`), so a dump no longer leaves the program.

Projection kernel (2026-09-27, kspeed, from team-lead's TODO): `project_relion_half_capacity`
and the half-storage branch of `relax.helpers.projection._project_relion_projector_texture` take
every slab whose texels fit the staging kernel's int32 indexing (box 800 at padding 2 is 1603 x
1603 x 802), in launches of at most 65535 rotations, so one kernel serves every size. A resident
local half stages one `RelionCapacityHalfTextureF32` for every slab, a plane group at a time
(1 GiB of staging beside the texture); `resident_local_pass2._open_resident_local_projector_texture`
and its persistent texture are gone. The per-call `relion_projector_half_texture_f32` branch now
serves only the geometry the half-storage kernel does not take (odd output sizes, padding other
than 1 or 2). The XLA pool reserve (`relax/helpers/xla_memory_reserve.py`) sizes the texture.

## Current evidence and open gates

The integrated CPU EM/InitialModel scope and focused GPU kernel suites pass. The
VDAM merge guard passes its maintained 12-case panel. The current consolidated
scorecards report K1 31/34, K4 direct 41/60, K4 all-class 9/15 and VDAM 12/12;
these are progress measures, not completion. The three recorded real-data K1
calibration cases remain in the maintained science-equivalence scorecard, while
the 10202 target is still pending.

Class3D seed-iteration gap (resolved 2026-10-01): on a seed iteration with flat posteriors (Pmax 0.004,
about 4500 significant samples per particle) relax's class maps were 0.27% below stock RELION's,
uniformly. The double-accumulation RELION build (double device backprojection) moves RELION by the same
5.6e-3 and relax matches it to 1e-4 at iteration 2: stock RELION's float32 accumulation of many small
weights, not a relax difference. Per-particle significant-sample counts differ for 25% of particles by
float32-level score differences under flat posteriors (the seed iteration scores one class per
particle, through the K=1 RELION coarse weight order). Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_k2optics_20260930`
(diag_oneopt_relion_bpd). Subtomogram Class3D (et16) hard single-particle offset switches remain OPEN.

EMPIAR-10202 (box 800, I1) progress on the resident K=1 engine: iteration 3
(current size 304) spent 4,530 s per half in the pass-2 M-step on 589ce09, which
visited all 55.4M candidate rows (bigbox 14450997, cancelled). With the live-row
M-step of 68a3cf0 the pass-2 chunk loop takes 52.9 s and 49.4 s per half and pass 1
about 128 s per half (bigbox 14456981), against RELION's 485 s for the whole
iteration on 2 H100s. The same runs exposed two box-800 memory faults, both fixed:
the cached-path chunk gather (131072 rows x 36514 pixels = 35.7 GiB, now bounded by
measured free memory) and the low-resolution half join, which moved a physically
large grid's host accumulators back to the device before the 1600^3 inverse FFT.

RELION's projector (`Projector::computeFourierTransformMap`) has one device build
(`relax/relion/relion_projector_setup.py::_build_projector_window`, bigbox 2026-09-27): the padded
transform one axis at a time inside a window, mask and shell power on the device. It replaced the
whole-volume rfftn build, which needed about 100 GB at box 800 and left 10202 on the native host
binding (two single-threaded builds, 117 s per iteration in py-spy 14561585). At 10202 it22 the
logged projector step is 14.1 s, against 61.8 s native (bigbox 14564062, 14561585); padded
128-512 costs 0.85-1.02x the old device build at full radius and
0.40-0.79x at half radius (bench 14564054). The device build is the default; native is the test
reference.

K=1 auto-refine HEALPix cap removed (2026-09-27, team-lead decision): relax capped K=1 sampling
at HEALPix order 7 by default, and at the cap `convergence.update_angular_sampling` never latches
`has_fine_enough_angular_sampling`, so a dataset that needs a finer grid never converges. RELION's
auto-refine has no cap (ml_optimiser.cpp `updateAngularSampling`, 11731-11753: the order goes up
while the old step is at least 75% of `acc_rot`, and the flag latches below it). On EMPIAR-10202
RELION went to order 8 at iteration 21 and 9 at 24 and converged at 27 (14475516), while relax
continuations sat at order 7 and 2.96 A for 23 and 44 iterations (bigbox 14522322, 14514496). The
default is now uncapped; `--max_healpix_order` remains an explicit opt-in cap.

Knife-edge note, pdb K=1 100k/256 (bench 14445854): relax converged at HEALPix 6, RELION refined to 7 at
its iteration 16 and converged at 18. Every latch input matched through iteration 15 (resolution, current
size, stall counters, the 0.46875 deg step) except `acc_rot` at the start of iteration 16: relax
0.6250000000000003, RELION 0.624, around the 0.625 threshold of `old_rottilt_step < 0.75 * acc_rot`
(ml_optimiser.cpp 11746-11748). relax's expected-accuracy binding accumulates the trial errors in RELION's
order and type (double, trial order, one division) and reproduces RELION exactly on RELION's own inputs:
0.6270000000000002 from `run_it014`, 0.6240000000000003 from `run_it015` (bigbox 2026-09-27). The flip is
therefore trajectory, one 0.1 deg step in one of the 100 trial particles from relax's own iteration-15 maps
and poses, not a parity defect.

Run files and `--continue` (2026-09-26): auto-refine and Class3D write RELION's
`run_itNNN_{optimiser,model,data,sampling}.star` and maps every iteration and continue from
them ([algorithm map section 8](../math/relion_refinement_algorithm.md)). Two departures from
relion_refine `--continue` keep a continued run equal to the uninterrupted one: each half keeps
its own noise (RELION's MPI restart broadcasts half 1's, ml_optimiser_mpi.cpp:750-758) and the
particle order stays the run's first mt19937 order (RELION reshuffles with random_seed + iter,
exp_model.cpp:406-446); STAR floats are written at full precision. On K1 5k/128 to convergence
(17 iterations, local search from 8) and K4 5k/128 (12 iterations), runs continued from several
iterations stay inside the same-code repeat band at every later iteration, with identical
schedules and convergence iteration (Slurm 14470000/1, 14470259, 14470264). On K1 C4 (14470328,
14470680, 14475183) continuations from iterations 4, 6, 8, 12, 17 and 20 match to 1e-6; from 5, the
iteration after a HEALPix refinement, every restored input equals the uninterrupted run's except the
references, which differ by 2.6e-7 after the float32 MRC round trip that RELION's `--continue` also
makes, and that flips one half-2 particle's significance support (44071 vs 44072 samples): half maps
differ by 1.9e-4 at iteration 6, the kind of single-particle flip a same-code repeat shows there. This
C4 fixture is chaotic from iteration 7 (one of two repeats converges at 19 instead of 21). VDAM has
no continuation yet.

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
(diff2.cuh:268-304, wavg.cuh:81-86). relax reproduces this
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
through iteration 6. A coarse window strictly between
2 maxR and about 2 s maxR, where RELION's wrapped coarse rows land inside the
sphere: the fused coarse scorer and the coarse GEMM operands reproduce it (the
projection and the image shift at the relabelled row, `relion_coarse_relabel`).
Still OPEN: the local parent pass refuses such a window (the S3b `--firstiter_cc`
run never reaches one); its image-shift relabel belongs in the resident parent probe.

Resolved (2026-09-29, cryo-ET): subtomogram runs write RELION's run files and continue from them (e435bfa4). The half
rows already map the particles onto the tomo particles.star; data.star now carries rlnOriginZAngst for the 3D offsets.
Check 14643890 (S1 seed 20260925): a run continued from iteration 6 ends on the uninterrupted run's map (FSC-AUC 1.0,
same iterations 7-9).

Resolved (2026-09-28, cryo-ET S4.2): relax's ground-truth FSC-AUC on the S1 depth-fix fixture was 0.003-0.004 below RELION's
(0.9346-0.9367 against 0.9381-0.9409). Two bugs, both found by comparing each iteration with RELION's `_model.star`/`_data.star`
and by one-iteration replays from RELION's state: the tilt fold added tilt-image power above the norm cutoff with mass 1 instead
of 1 / n_images (noise beyond the current size 39x RELION's; 3409d7bc), and pass 2 backprojected the tilt images without their
group scale (RELION's ctfs = Fctf * scale, acc_ml_optimiser_impl.h:4370-4404; the it002 replay from RELION's it001 state had
BPref data 4.9% off, 1.5% after the fix, the rest being a few particles' pose flips; 9f8efc89). Gate 14624974: relax GT
0.9354-0.9409 against RELION 0.9381-0.9409; seeds 20260926 and 20260927 reproduce the same-seed RELION maps (relax-vs-RELION
FSC-AUC 1.0000), seed 20260925 leaves RELION's trajectory at iteration 5 (0.9354). Also RELION-matched on the way: no norm
correction for subtomograms (608a997b).

RELION float32 BPref accumulation band (2026-09-25, not reproduced; lead decision). RELION's GPU backprojector adds every
particle of a half into one float32 volume with `atomicAdd` (acc/acc_backprojector.h:41; acc/cuda/cuda_kernels/BP.cuh:157-169)
and reads it back once per iteration (ml_optimiser_mpi.cpp:1719). At iteration 1, when posteriors are broad, the rounding drops
tiny terms. On S3b the DC BPref weight is 0.32 % below `0.999 sum Q^2/sigma2[0]` for RELION and 0.083 % below it for relax,
decaying with radius. RELION's retained mass is 0.999 (14428592), and a float32 sequential-sum simulation reproduces RELION's DC
to 1e-4. By it3 the formula holds to 1e-5. relax loses less because each optics shape class starts from its own zero
accumulator. Single-optics runs are unaffected. This is not an algorithmic choice and depends on accumulation order, so relax
does not reproduce it. It also does not explain the S3b final gap: a free run from RELION's it001 (14431963) ends at GT FSC-AUC
0.8929, where RELION reaches 0.9055 and relax standalone 0.8923. Resolved (2026-09-25): that gap was the exact local
engine at the full box, which scored the whole half grid. RELION's resolution pointers still cut the corners there
(ires < image_current_size / 2 + 1), and for optics group 2 (s = 1.12) the corners project inside the model sphere, so
relax's local scores carried a rotation- and shift-dependent term (RELION ACC dumps at it013: coarse diff2 residual std
10-11, with the same argmin). The local search now scores RELION's window at the box (window_at_box, as the resident
drivers). One-step replays from it012: group-2 Pmax correlation 0.59 -> 0.998 (group 1 unchanged, 0.9999); standalone
S3b GT FSC-AUC 0.9057 (RELION 0.9055, main before the fix 0.8923), relax vs RELION 0.9957 (was 0.9755), 844 s wall
(job 14451343). Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/multioptics_spa_s3_20260924/s3b_it013_dumps_20260925`,
`/scratch/gpfs/CRYOEM/gilleslab/em_work/exactlocal_box_20260925`, `s3b_winbox_{ctrl_01fddc2,cand_9c41737}_20260925`.

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

VDAM K>1 benchmark misses (2026-10-02): synth_pdb_k2 50k/256 (s41/s61/s67) and ribosembly K15 s61
were closed by same-command repeats (both engines are bimodal there). One-iteration replays at pdb K2
s67 from double-accumulation RELION checkpoints (it10/20/50/100/150, the same particle subset in every
arm) put relax-vs-RELION inside the spread between two RELION builds (float and double accumulation)
from the same input: class assignment 100% in every pair, populations within 3e-6, mean Pmax and
significant-count differences of either sign. Best pose and Pmax per particle are near-ties under flat
posteriors (the two RELION builds disagree on 12-100% of best poses after one step). Stock RELION
`--continue` restarts the subset shuffle from the input order (`sorted_idx` is not checkpointed), so a
matched replay needs the same order on both sides. The per-class tau2 "difference" of 2e-5 to
1.6e-4 seen there is RELION's model.star rounding (values at or above 1e-4 get 6 fixed decimals, e.g.
0.002124 for 0.0021235641): all 1290 relax tau2 values fall within half a unit of RELION's last printed
digit. Table:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/etw_vdam_k2_onestep_20261002/compare2_s67_it{10,20,50,100,150}.json`.

VDAM defaults (2026-09-25, landed from the vdamspeed stack): RELION's ternary max/min in the JAX
M-step FSC estimate; the exact K=1 pass 2 always packs flat local rows; the float32 JAX M-step
transaction is the only default route (float64 stays a diagnostic); one projector setup, the device
FFT in double narrowed to the complex64 slab RELION's GPU projector holds as a float texture
(`--projector-setup-backend` is gone from InitialModel); the noise carry starts at its block dtype.
PPCA: the VDAM comparison arm of the PPCA pilots now builds its projector in double for float32 runs
too (before, the JAX setup followed the M-step dtype through `_projector_setup_dtype`, removed with
its test); PPCA's own code does not call the VDAM projector. Quality, scored on cand_04e38a9 (this
stack plus bc30d20, the per-class device coarse significance that follows it on main; the same code
before the rebase): pdb K=2 5k/128
population-weighted GT FSC-AUC 0.3204 / 0.3883 / 0.4485 at seeds 29 / 41 / 53, against RELION
0.3229-0.3354 (7 runs) / 0.3797 / 0.4528-0.4575 and the same-job pre-stack control 0.3110 / 0.1360 /
0.4410; that is 0.0025 / 0.0043 below RELION's seed-29 / seed-53 range, inside relax's seed-29 basin
spread (0.3161-0.3381). EMPIAR-10097 0.2248 / 0.3565 (unmasked / masked) inside RELION seed-41 runs
0.2225-0.2254 / 0.3550-0.3588. Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamspeed_20260924/scoring` (jobs 14427596, 14427597).

VDAM on the resident engine (2026-09-25; K>1 default 2026-09-26, K=1 default 2026-09-26): the adaptive
route runs the InitialModel E-step on auto-refine's adaptive route (`relax/vdam/adaptive_estep.py`) with
the device-resident pass 2. `--pass2_engine auto` (the default) selects it for every K
(`relax.vdam.dense_adapter.vdam_pass2_route`); a configuration the adaptive route
refuses before device work runs exact-local with a logged reason, `adaptive` makes the refusal an
error, and each iteration's `run_itNNN_recovar_meta.json` records `pass2_engine` and `pass2_engines`.
Gate (200 iterations, relax 47103e5, one H100 + 8 CPUs per arm, uncapped; GT FSC-AUC against the
same-seed RELION runs, `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_kclass_20260925/vdam_gate/GATE_SUMMARY.json`):

| case | RELION band | exact-local | resident | walls RELION / local / resident (s) |
|---|---|---|---|---|
| pdb K2 s29 | 0.20307-0.20755 | 0.20480 | 0.20706 | 883 / 2665 / 1682 |
| pdb K2 s41 | 0.22357-0.22536 | 0.22500 | 0.22650 | 962 / 2537 / 1518 |
| pdb K2 s53 | 0.24476-0.24661 | 0.24450 | 0.24677 | 966 / 2548 / 1519 |
| pdb K4 s29 | 0.07542-0.07554 | 0.07439 | 0.07577 | 773* / 1932 / 1750 |
| pdb K4 s41 | 0.12389-0.12641 | 0.12733 | 0.12141 | 1087* / 2502 / 1933 |
| pdb K4 s53 | 0.07777-0.07800 | 0.07860 | 0.07757 | 794* / 2087 / 1995 |
| noise1 50k K1 s29 | 0.34125-0.34141 | 0.34134 | 0.34134 | 1565 / 2232 / 2382 |

(* bench's RELION jobs 14445855-7 on other nodes.) Resident is 0.60-0.96x the exact-local wall at K>1 and
1.07x at K=1 at 47103e5; K=1 moved to resident on 2026-09-26 (below). K4 needs more RELION seeds before any
quality claim: single seeds scatter about ±2.5e-3 around RELION's two same-seed runs, for both routes.
RELION's three `--grad` E-step differences map onto it: the residual backprojection
(`mstep_subtract_ctf_projection`, now on resident), the pseudo-halfset BPref slots (one resident pass
whose accumulator slot is `class + K * pseudo-halfset`, `docs/development/resident_segments.md`) and the
coarse-only `maximum_significants = 100 K`. K>1 runs through the same pass (2026-09-25; pdb K2 seed 29
iterations 1-12 match the exact-local route to 7.9e-6 in the maps with identical classes and angles,
job 14444404). The exact-local VDAM route was removed on 2026-09-27; resident is VDAM's only route.
Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamres_20260925/HANDOFF.json`.
Quality on noise1 50k/256 seed 29 (200 iterations, job 14422222, scored 14425217): relax resident
GT FSC-AUC 0.34131 unmasked / 0.75598 masked, inside four RELION runs 0.34125-0.34141 / 0.75515-0.75603;
relax-RELION pair FSC-AUC 0.971-0.984 against RELION-RELION 0.977-0.983. Speed gate (manager item 11,
not met): wall 4549 s against RELION 1451 s on the same node, slower than the exact-local default; the
E-step is 90% of it and grows from 9 s to 31 s per iteration. The resident route stays opt-in and the
exact-local route stays until resident is no slower at equal quality on 10097 and noise1 50k.
EMPIAR-10097 seed 41 (200 iterations, job 14425100 at the cache fix cbd7b0b, scored 14434198 against the
RELION auto-refine map in the frozen mask): relax resident FSC-AUC 0.2248 unmasked / 0.3578 masked, at the top
of four seed-41 RELION runs 0.2225-0.2243 / 0.3550-0.3573; relax-RELION pair FSC-AUC 0.701-0.732 against
RELION-RELION 0.688-0.773. Wall 6954 s against RELION 3025 s on the same node (2.3x); the exact-local default
measured 6835 s in a separate job (vdamspeed j14, not a matched pair), so the 10097 speed gate is not shown
met. The run predates the image-capacity change (834b3b3), which removes the per-subset re-trace.
K=1 default flip (2026-09-26): with stable Fourier windows and S2 (relax 5802a5e) resident is faster
than exact-local at K=1: noise1 50k s29 resident 1303 s, 0.92x its same-node RELION (job 14502014), where
exact-local took 1437 s against RELION 1421 s (job 14497728); EMPIAR-10097 s41 on one node 2469 s against
exact-local 4274 s and RELION 2957 s (job 14504146).
10097 quality (scored 14508334, `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamfast_20260926/e10097_route_1dafaae/scores`):
resident GT FSC-AUC 0.2269 unmasked / 0.3611 masked, exact-local 0.2236 / 0.3552, five seed-41 RELION
runs 0.2225-0.2259 / 0.3550-0.3605. `--pass2_engine auto` now selects resident for every K.

VDAM coarse scorer, native vs GEMM (2026-09-26, relax main bc6d3e1; same node, one H100 + 8 CPUs per
arm, uncapped). The default exact-operand coarse pass is a real-packed float32 GEMM pair
(`relax/scoring/scoring.py::_relion_coarse_gaussian_gemm_scores_jit`); the native arm is RELION's fused
per-pair projector/diff2 kernel (`RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=0`, and for the K>1 resident route
also `RECOVAR_K1_COARSE_GAUSSIAN_FFI=1 RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1 RECOVAR_K1_COARSE_FUSED_PROJECTOR=1`, because K>1 resident
otherwise scores pass 1 with the generic dense GEMM; that native arm was removed on 2026-10-02, see "Pass 1
as one program per image batch"):

| case, route | RELION | native coarse | GEMM coarse (default) |
|---|---|---|---|
| noise1 50k K1 s29, exact-local | 1422 s | 2238 s (1.57x) | 1592 s (1.12x) |
| pdb K2 s29, resident | 993 s | 2181 s (2.20x) | 1249 s (1.26x) |

GEMM quality (relax 12fe042 gate, `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamfast_20260926/gate_s1_12fe042/scores`):
mean GT FSC-AUC inside or above RELION's same-seed runs for pdb K2 s29/s41/s53 and K4 s29, noise1 masked
0.7554 (exact-local) and 0.7547/0.7555 (resident repeat) against RELION 0.7552-0.7560. Float32 against
binary64 GEMM scores differ by at most 0.02 (about 20 ulp) near the best pose. The late iterations are
already faster than RELION on the GEMM path (noise1 its 151-200: 544 s against about 766 s); the remaining
gap is early and middle iterations, where XLA compilation is 46-70% of main-thread time
(whole-run py-spy, `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamfast_20260926/e11_latepy`,
`e15_localpy`). Evidence: gates 14474812 and 14475131 in `.../relax_vdamfast_20260926/gate_main_bc6d3e1`.

OPEN (compact, not fixed: the compact engine is to be deleted): without scale-correction groups the
compact sparse pass 2 takes its non-atomic noise arithmetic, 19% apart in `wsum_sigma2_noise` from
RELION's scale-1 Wavg triplet on the algebraic-Wavg test fixture (repro: `_vdam_args(residual=True,
groups=False)` in `tests/unit/test_resident_vdam_estep.py`, compact against resident, job 14421464).
The resident engine runs RELION's triplet at scale 1 (relax 40b14ac).

Class3D local searches, K>1 (2026-09-25): the per-class local route now keeps
RELION's joint per-particle pass-2 support (3601775); it matches the
class-segmented pass to 1e-7. No benchmark reaches this route (Class3D keeps
HEALPix 1-2, below auto_local_healpix_order 4). Later engine task: give the
class-segmented pass padding factor 2 and a per-class scale-correction mask
(RELION's `data_vs_prior_class[iclass] > 3`), route Class3D local K>1 through
it, then delete the per-class K>1 loop (one implementation; K=1 keeps its path).
Update 2026-09-25: Class3D no longer reaches local searches at all. RELION switches
to local searches from the HEALPix order only under auto-refine: the iteration-0
switch is inside `if (do_auto_refine)` (ml_optimiser.cpp:2302-2326) and later
switches come from `updateAngularSampling`, which Class3D never calls
(ml_optimiser.cpp:3550-3552); relax has no `--sigma_ang`. relax switched Class3D
to local searches at HEALPix >= 4 (fixed 7381f84, replays 016e261), and the
Class3D local K>1 route (the K>1 branch of `_run_local_search_iteration` and the
class arms of the local half scorer) is deleted as unreachable. The per-class
and segmented K-class local passes (`run_local_k_class_em`) were removed with
the VDAM exact-local route on 2026-09-27.

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
(`resident_pass2.compute_k_class_pass2_stats_resident`). Its noise is pinned by the
duplicated-class test against the K=1 resident pass. The compact K>1 engine is not
fixed; it is deleted with compact.

Class3D on the resident engine (2026-09-25): the K-class pass 2 runs on the resident
engine (the only pass-2 engine since the compact removal of 2026-09-26) and records its engine. Rows carry the class axis
(`docs/development/resident_segments.md`). K4 50k/256 long (15 iterations, candidate e2f5197,
job 14428127 against the RELION repeat of 14427006): GT min FSC-AUC 0.216797 (masked 0.216068),
inside RELION's band [0.216623, 0.216957] (masked [0.215882, 0.216170]); the compact long run
14410744 scores 0.216536, 8.7e-5 below it. K4 100k/256 completion (job 14426696, CPU-matched
pair): GT min FSC-AUC 0.26407 (masked 0.26795), above RELION's [0.26276, 0.26322] (masked
[0.26642, 0.26679]); wall 14620 s against RELION 4666 s.

RELION MPI scale-group pack/unpack defect (RELION 5.0.1 f2c1a38; relax issue #1): the pieced
`MlWsumModel::pack` / `unpack` that `relion_refine_mpi` uses to combine weighted sums in CUDA builds
(`src/ml_model.cpp:2056`, `:2241`) size the scale-group loop with `sigma2_noise.size()`, the number of
optics groups, so only scale group 1's `wsum_signal_product` / `wsum_reference_power` cross ranks. With
`--scale` and several followers per data set, every other group's scale correction comes from one
follower's particles; with one particle per group (the simulated fixtures) half the groups collapse to a
common value. relax implements the non-MPI behaviour. Affected: Class3D (no split halves) run with more
than one follower. Auto-refine with one follower per half-set (the 3-rank arms of every K=1 benchmark row)
skips the within-half combine, and its final join's scale update is not used for any reconstruction, so the
K=1 rows are unaffected (checked on the 10097 row: per-group scales vary normally in both halves). Evidence:
K4 5k/128 GUI Class3D, three seeds: relax equals non-MPI RELION (class agreement 1.0000 at all 25
iterations; GT class accuracy and FSC-AUC identical on seed 29), while MPI RELION splits from it at
iteration 3 (0.979-0.981) and ends in another solution (GT class accuracy 0.992 vs 0.976); two MPI repeats
also split at iteration 3 (0.981). A two-line patch (use `wsum_signal_product.size()`) makes
`relion_refine_mpi` equal to non-MPI RELION on that case (agreement 1.0000 at every iteration, scale
corrections to 4e-5; jobs 14456979, 14457083). Patched build, patch, verification and report draft:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relion_patched_mpi_scale_20260926/` (run with
`SLURM_MPI_TYPE=pmix_v3`). Class3D benchmark references should be non-MPI RELION or this patched build.

K4 5k/128 seed 29 score shift (2026-10-03, closed as a tie): main scored masked GT FSC-AUC 0.475817 against
0.476104 for the 09-27 run and both RELION builds. Bisect over the 271 relax commits since ddf88c8: the first
change is a9c89cb (K-class M-step sums each projection's rows, then backprojects once), which moves relax
toward double-accumulation RELION (iteration-2 class maps 7.5e-7 from it, against 1.15e-5 for per-row
backprojection and 1.55e-5 for stock RELION) and still scores 0.476104. The drop is one particle
(2254@particles.128.mrcs) taking a different pose at iteration 4, and later unrelated commits toggle it on
and off. Its coarse posterior puts 0.9990002 or 0.9989997 of the mass on the top coarse sample against
RELION's adaptive fraction 0.999 (margin about 2e-7, a few float32 ulps): one or two coarse samples are
significant, and the fine winner follows. RELION keeps one (rlnNrOfSignificantSamples 1, pose with Pmax
0.8263). Enabling the capture alone swaps the side. A float tie, not a relax difference. Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/etw_k4bisect_20261003` (score_out.json, capture/*/sigdump).

K4 100k/256 class agreement (formerly OPEN; explained by the defect above): relax's class agreement with the
two MPI RELION runs was 0.878-0.894 at iteration 15 while they agree with each other at 0.9335. Non-MPI RELION
with the same command (job 14455196, 3643 s) agrees with those MPI runs at 0.8785-0.8954, the same level, and
with relax at 0.9780; map FSC-AUC relax vs non-MPI RELION 0.9696 against 0.842-0.857 for non-MPI vs MPI RELION.
GT min FSC-AUC: relax 0.26407 (masked 0.26795), non-MPI RELION 0.26182 (0.26522), MPI RELION 0.26276-0.26322
(0.26642-0.26679). Compact scores like resident (long run 14410744). The one-step replays are no longer needed.
Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_kclass_20260925/band_100k/BAND_nompi.json`,
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_kclass_20260925/k4_100k_relion_nompi/TRAJ_relax_vs_nompi.json`.

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
final all-data iteration. The 100k/256 K1 and exactly-K4 quality and
performance gates remain open.

Final all-data pass on resident local (2026-09-27, 1178448..73c2a22). The full-box
pass routes to the resident local driver, whose M-step now walks only rows with
weight and translates unshifted per-image operands inside the translate-sum kernel.
Same-node 10097 10k gate (job 14518565, scored 14518568): final iteration 123.3 s
on the exact local engine, 92.6 s on resident; run walls main c423fb7 603 s,
candidate 595 s, RELION MPI 3x4 592 s; map gate passes against all five
same-command RELION runs (the candidate landed in the RELION basin of run a1,
cross-engine merged 0.9938). The fast parity tier (13/13) and the resident GPU unit
files (78 passed, job 14518566) pass. The NumPy reference test covers the box case
at noise 200; at noise 2 the resident Ft_y relative L2 is 1.18e-5 against the exact
engine's 2.3e-5 (job 14508921), above the test's 1e-5 bound, pending a decision on a
derived float32 bound.

Wall references (2026-09-26, user decision). The speed target is at most 0.5x RELION's cold
wall for one full-size run, as a same-node pair: K=1 on EMPIAR-10097 and the 50k/100k synthetic
fixtures, Class3D on K4 100k. Compile amortized over a run's iterations counts; warm-cache walls
are not quoted. 5k fixtures serve iteration and profiling only, since compile dominates them.
Quote each wall against the fastest RELION configuration that runs, and show both when both run.
Auto-refine (`--split_random_halves`) runs only under MPI (non-MPI exits at start), so K=1
references are `relion_refine_mpi` 3 ranks x 4 threads (K=1 5k/128: 196 and 200 s, jobs 14497840
and 14499533). Class3D runs both; non-MPI `--j 12` is faster (K4 5k/128: 268-271 s against MPI
382-390 s) and is also the quality reference (MPI scale-group defect above). The 268-271 s walls in
`tests/baselines/relion_vs_relax_benchmarks.json` rows `pdb_k4_5k_class3d_25it_*` are that Class3D
non-MPI reference and are labelled correctly; no non-MPI K=1 wall exists. Since ed7d5d7 the
refinement entry point activates the persistent compilation cache, so a cold arm needs its own
empty `JAX_COMPILATION_CACHE_DIR` (or `RECOVAR_DISABLE_JAX_CACHE=1`).

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
`--adaptive_oversampling 0`. A global pass the resident checks refuse is an error (the compact
engine is deleted), and so is a local fine pass they refuse (2026-09-27); only the pass-1 parent
probe still runs on the deprecated exact-local engine, with a logged reason recorded per iteration
in `pass2_engine_trajectory`. The jitted stage glue and the local image-capacity
ladder are set at the K=1 entry points (`apply_k1_refine3d_env_defaults`), since VDAM shares that
code and qualifies its own defaults. Transitional A/B off switches (not permanent variants;
`RELAX_SPARSE_PASS2_RESIDENT` and `RELAX_LOCAL_SEARCH_RESIDENT` are retired and setting either is an
error): `RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF=0`,
`RELAX_K1_RELION_WAVG_SEQUENTIAL_CUDA=0`, `RELAX_COARSE_PAD_FINAL_IMAGE_BATCH=0`,
`RELAX_EM_JIT_STAGE_GLUE=0`, `RELAX_LOCAL_IMAGE_CAPACITY_LADDER=0`. Device coarse significance has no
switch: every class's coarse support is compacted on the device whenever the ids are collected
(K>1 and VDAM too); score dumps keep the host mask
(`relax/sparse_pass2/resident_significance.py`). Flip pairs (relax vs
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
it13 half 2). End to end, such
arithmetic-path differences compound into a larger map disagreement than relax's (measured on the K1
5k/128 os0 fixture, auto-refine to convergence plus the final all-data iteration; cross FSC-AUC of the
unfiltered half-map average / merged map against the RELION GPU oracle): RELION's CPU path 0.99772 /
0.99719 (job 14426055, converged at iteration 14 like the GPU runs); relax main b0276ba 0.99990 /
0.99986 (two identical arms 14446428, 1.000000 / 0.999999 to each other); RELION GPU same-command
repeats 0.99999 / 0.99998 and 0.99900 / 0.99814. relax therefore sits inside RELION's GPU repeat band
and about 20x closer to the GPU oracle than RELION's own CPU path. The 50k/256 full-run gap (0.9994 vs
0.9997) has no CPU-path end-to-end run; it is attributed by the same mechanism, not measured at 50k.
Scores: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_landing_20260925/score_5k_all.json`.
The earlier attribution (1.3-4.2e-4 per step, "about 2.5x RELION's drift") is superseded: the replay
harness then scored half 2 with half 1's sigma2_noise (the RELION MPI restart broadcast); those
replays (relax a7977c8) read 4.3e-4-7.6e-4 for half 2 at it2/it5, against the per-half-noise values
above (relax 64b08ef, whose half 1 also moved from 6.6e-5 to 4.0e-5 at it2). Replays with per-half noise: jobs 14425333 (relax), 14411352 (RELION CPU),
under `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_speed_20260923/replay_50k_iters_cns_20260925`
and `replay_50k_iters_20260925/relion_cont`.

Tested, no effect (2026-09-25, not landed): RELION's CUDA path normalizes each real image by
`(XFLOAT)(avg_norm_correction / normcorr)` (`acc/acc_ml_optimiser_impl.h:432`, f2c1a38). relax
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
