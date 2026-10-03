# Refinement readability refactor: current status

<a id="current-half-image-preprocessing-ownership"></a>

## Review checkpoint — October 3

This branch is published for design review at the user’s explicit request.
The readability design is unfinished; no merge or performance acceptance is
claimed. Start with the [active deeper plan](refactor_extension_plan.md#active-deeper-design-pass--october-3)
and [complete calling flow](final_local_sampling_patch_review.md).

Current checkpoint: local smoke passed (466 GPU cases, 251 controller cases,
108 guards and three required replays; one optional fixture skip). Medium
14928923 is running on H100; K1 14912912 depends on it. Old held job 14910610
was cancelled before execution at the user’s request. Full K1/exactly K4 quality,
real-data confirmation, repeated matched-GPU speed/memory, milestone long and
main delivery remain open. Ordinary performance repeats await design review.

The numerical files match the immutable checkpoint based on 80f2b2; subsequent
changes describe the deeper design plan. Scratch evidence links require cluster
access. The following integration history describes that checkpoint, not approval
of its architecture.

### Draft publication checks

Guide mirror/link validation passed. Full changed-Python Ruff check reports 33
findings (30 automatically fixable); formatter check reports 131 files needing
formatting and 18 already formatted. These are open cleanup items. Diff whitespace
checks also flag preserved user comments in `comments.md` and `finalization.py`,
and a trailing blank line in `startup_noise.py`. No broad formatting was applied
to this numerical checkpoint. Focused CPU and smoke results above do not certify
these style checks or the unfinished production gates.

User authorization: “push to a branch or something so i can ask other agents to
think about it and take a look” (October 3). This is a draft review publication
under CONTRIBUTING.md; it is not authorization to waive main's merge gates.

### Questions for independent reviewers

- Where does an operation force the reader to understand unrelated state?
- Which repeated branches encode the same decision, and which are distinct science?
- Which optional fields permit impossible states? Trace their producers.
- Which new abstractions merely forward or collect arguments?
- Propose one complete caller-and-owner replacement, including state writes and
  array lifetimes. Avoid file-count or line-count targets.
- Identify concrete removal candidates with caller evidence; preserve independent
  numerical references and all scientific/precision/performance gates.

Review scope starts with numbered reconstruction in `iteration_loop.py` and
`mean_helpers.py`, then finalization. Existing principles and the extension plan
remain the shared guidance; do not start a competing architecture document.

## Checkpoint integration history

The completed operation ownership is integrated with GitHub main
`80f2b2b36cee01d110276d189a3b061c78746c86`. The thirteen code/guide commits through
`69abd77` retain their changes; the subsequent commit adds PPCA measurement
documentation only. The incoming work includes
pass-1 runner-up, exact-operand, normalization and dump-schema changes,
PPCA work and tomography safeguards. Incoming command/controller changes are
adapted to the existing refactor owners. Scientific sequencing and visible
updates remain unchanged by these interface migrations.

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

[Current integration, receipts, preserved jobs and required gates](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/refactor_finish_sync_69abd77_20261003/HANDOFF.json)
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
Source spans are 2760/1766 for numerical/command controllers;
these counts are review signals, not design acceptance. Current CPU checks are recorded above; earlier passing receipts describe their
own source only. The milestone is incomplete until the frozen float32 K1/exactly-K4
scientific, real-data, memory and matched-GPU speed gates pass and delivery to main
is verified. Existing comments, frozen candidates, controls and jobs are preserved.

<a id="current-input-geometry-and-replay-boundary"></a>

## Earlier input geometry and replay boundary

Replay now consumes validated `ImageGeometry`; the generic controller dataset
alias and two silent invalid-pixel substitutions are removed. Original scalar
types remain explicit for existing host arithmetic. Scientific state writes,
sampling order and array lifetimes are unchanged. See the
[actual producer and calling flow](final_local_sampling_patch_review.md#input-geometry-and-replay-boundary).

All 185 selected CPU cases pass with zero skips; source/test lint, structural
comparison and diff checks pass. The initial structural proof required a script
normalization repair, with no production/test change. Exact source and commands
are in the [handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_input_geometry_20261002T210920Z/HANDOFF.json).

Controller spans 2,917 lines and command 1,860. Representative phase ownership,
whole dead/shared-code audit and the prior Class3D scratch-lifetime choice remain
open. Production float32 K1/exactly-K4, real-data quality, memory and matched-GPU
speed await complete-flow review and a new freeze. Original work, frozen sources,
controls and jobs are preserved. No new GPU job, merge or publication.

<a id="current-sealed-restart-runtime-ownership"></a>

## Earlier sealed restart runtime ownership

The existing diagnostic CLI owner now holds projector-only slots, captured-state
attachment and float32 scoring-noise expansion with its admission/source helpers.
Controller mode/source selection, noise installation, capture publication and
validation timing remain visible. No new function, type, module or wrapper.
See the [actual callers and implementations](final_local_sampling_patch_review.md#sealed-restart-runtime-adaptation).

The affected inventory passed 223 cases and found one stale source guard from
the preceding expectation move; after test-only migration all three affected
guards pass, zero skips. Numerical assertions/tolerances are unchanged. Structural
comparison, isolated diagnostic import, source/test lint and diff checks pass.
Exact source, commands and failure repair are in the [package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_sealed_restart_owner_20261002T204921Z/HANDOFF.json).

Controller remains 2,913 lines and command 1,860. Representative design and the
prior Class3D scratch-lifetime choice remain open. Final production float32
K1/exactly-K4, real-data quality, memory and matched-GPU speed await complete-flow
review and a new freeze. No new GPU job, merge or publication; original work,
frozen sources, controls and jobs remain preserved.

<a id="current-numbered-expectation-preparation"></a>

## Earlier numbered expectation preparation

The existing expectation owner now binds the canonical trial grid, dense/local
sampling support and local diagnostic policy once for both halves. The
controller retains mode decisions, explicit tomography adaptation, half-specific
operands, score publication, serial/overlap dispatch and the memory boundary.
See the [complete construction and implementation](final_local_sampling_patch_review.md#numbered-expectation-preparation).

Initial checks passed 132 cases and found eight incorrect new window expectations
plus one source guard needing its new owner. After test-only repairs, 43 repair
cases, 27 affected controller cases and all 108 CPU EM guards pass with zero
skips. Structural, import and lint checks pass. Exact source, commands, repaired
failures and limitations are in the [package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_numbered_expectation_preparation_20261002T202923Z/HANDOFF.json).

The controller remains 2,913 lines, command 1,860; scorer has 19 inputs and
producer 18, with 34 half-worker captures. The representative design remains
unfinished. Full production float32 K1/exactly-K4, real-data quality, memory and
matched-GPU speed await complete-flow review and freeze. The prior Class3D
lifetime choice remains pending. No new GPU job, merge or publication;
original workspace, frozen sources, controls and jobs remain preserved.

<a id="current-command-oracle-admission-and-runtime-controls"></a>

## Earlier command oracle admission and runtime controls

The existing command-options owner now performs verified dispatch admission and
CLI/optimiser control resolution. The controller retains group-source identity,
the strict Class3D replay decision, follower preparation, frozen-aware optimiser
selection and visible control installation before reference loading. Real
manifest checks and saved significant-support argument handling can now be
changed without reading numerical refinement or reference loading. See the
[complete caller and implementations](final_local_sampling_patch_review.md#command-oracle-admission-and-runtime-controls).

The first focused run passed 208 cases and had 12 new fixture setup errors from
unsorted manifest names. After sorting that fixture, all 26 new admission cases
pass. The production source is unchanged between runs; the 194 existing follower,
command/replay, default, import and archive cases passed initially. Zero skips;
no tolerance/baseline changes. Real NPZ/manifest/STAR admission, caller operand
identity, refusal/error scope and CLI/saved/default precedence are covered.
Structural comparison confirms the moved computations and unchanged controller
outside admission; it records successful-resolution-before-installation of CLI
scalars. Imports, lint and diff checks pass. Exact receipts and saved source are
in the [package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_command_admission_20261002T201344Z/HANDOFF.json).

The numerical controller remains 2,949 lines and the command spans 1,860. The
23-input scorer and 38-capture half worker remain unfinished. Full production
K1/exactly-K4 and real-data quality, memory and matched-GPU speed await finished
code, complete-flow review and a new frozen candidate. Class3D scratch-lifetime
choice is still pending. No new GPU job, merge, publication or qualification;
original workspace, frozen sources, controls and jobs remain preserved.

<a id="current-correction-reporting-and-capture"></a>

## Earlier correction reporting and capture

The normalization owner now provides `NormScaleCorrectionReport`, a small
borrowed reporting result reused by checkpoint averages and parity captures.
It replaces four parallel reporting locals and distinguishes selected rank-1
follower scales from installed particle scales. The existing diagnostic owner
adapts completed-iteration operands to the capture schema and preserves its
encoding casts and warning policy. The controller retains correction admission
and installation, convergence, checkpoint-before-capture order and full versus
timing-only selection. See the [complete current calling flow and owners](final_local_sampling_patch_review.md#correction-reporting-and-completed-iteration-capture).

The initial focused run passed **88 of 93 cases**. Five failures arose from two
new fixture mistakes: assuming a for-loop where the controller uses a while-loop,
and omitting E-step snapshots required for per-half serialization. After those
fixture repairs, all **13 new reporting/capture cases** and **25 affected
controller cases** pass (237 deselected), with zero skips. Existing normalization,
follower, checkpoint-lifetime and timing cases passed in the first run. Structural
comparison inlines the actual helper/report and confirms the original controller
operations, without removing numerical operations, casts, conditions, state
writes, handlers or timing calls. All pre-existing normalization and diagnostic
functions/types are unchanged. Pinned runtime imports, import lint and diff checks
pass. Exact commands, source manifests, XML, repaired fixture failures and proof
are in the [package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_correction_reporting_20261002T194117Z/HANDOFF.json).

This adds one type and one adaptation function in existing owners, with no new
module, forwarding method, flag or kernel. The controller still spans **2,949
lines** and command `main()` **1,992**. Its half worker still captures 38 names;
phase construction and the representative refactor remain unfinished. Class3D
scratch-lifetime choice remains pending. Full production K1/exactly-K4, real-data
quality, peak memory and matched-GPU performance remain unqualified until the
complete flow is finished, reviewed and frozen. No new GPU job, merge or
publication; original work, frozen sources, controls and jobs are preserved.

## Earlier half input ownership

The existing half operands now borrow reference-grid radial noise; local/dense
sampling carries its coarse angular step; particle diameter comes from the
already-required immutable schedule. No new type, function, module, flag or
forwarding layer was introduced. The same tomography/empty-half noise guard
remains in the controller, alongside explicit score/frame/offload/publication and
recording order. See the [complete construction, caller and operation](final_local_sampling_patch_review.md#numbered-half-input-ownership).

The first focused command passed **122 cases** and failed three final-preparation
fixture cases that imported the numbered fixture's retired argument dictionary.
That indirect fixture now reads the half owner without changing its independent
numerical assertions. All **three final-preparation cases** and **45 affected
controller cases** then passed (217 deselected), with zero skips. Eight added
cases check unchanged scalar/table identity at batch/optics consumers and execute
the actual controller noise binding with unused metadata unavailable. Structural
comparison resolves only the three proven-equal bindings and confirms the
original controller/scoring operations. All lower scoring adapters and other
expectation functions/types remain unchanged. Import lint, pinned imports and
diff checks pass. Exact commands, source manifests, XML, producer/lifetime traces
and the repaired failure are in the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_expectation_input_ownership_20261002T191937Z/HANDOFF.json).

At that source the controller spanned **2,968 lines**, and command `main()` **1,992**. Its
63-line half worker captures 38 names; the 23-input score operation and phase
construction remain unfinished. The Class3D scratch-lifetime choice remains
pending. Full production K1/exactly-K4, real-data quality, peak memory and
matched-GPU speed remain unqualified. No new GPU job, merge, publication or
scientific/performance acceptance; existing candidates, controls and jobs remain
preserved. Finish and review the representative flow before freezing it for
those final gates.

## Earlier half-expectation recording

The editable cached-main `a9e0668` checkout now puts half profiles, support-count
collection and E-step captures with the existing expectation owner.
`SignificanceStatistics` replaces five parallel count locals with per-half counts
and their distinct recorded/convergence policies. Combination retains recording
order, absent/empty semantics and its position after preprocessing drains.
Scoring, metadata writes, offloading and payload publication remain visible in
the controller. See the [complete caller, implementations and consumers](final_local_sampling_patch_review.md#half-expectation-recording-and-support-counts).

The first focused command ran 128 cases: **127 passed and one stale checkpoint
fixture failed** after its old local was retired. The fixture was migrated without
weakening its map-lifetime check; all **19 checkpoint cases** then passed. All
**45 affected controller cases** passed (217 deselected), with zero skips in both
commands. Fifteen new cases cover ordering, count policy, empty halves, metadata
fallbacks and borrowed accumulator identity. Structural comparison inlines the
actual recorder/count methods and confirms the original controller operations;
all four existing expectation functions are unchanged. Import lint, pinned
runtime imports and diff checks pass. Exact commands, source manifests, XML,
logs and the repaired failure are in the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_expectation_publication_20261002T184926Z/HANDOFF.json).
The manifests include untracked owners/tests; the tracked diff hash alone is
not their source identity.

At that source the numerical controller spanned **2,969 lines**, and command `main()`
**1,992**. Its half worker now spans 65 lines and captures 40 names; the phase
interface and representative refactor remain unfinished. The earlier Class3D
scratch-lifetime choice is still pending. No new module, GPU job, numerical or
performance acceptance, merge or publication occurred. Existing frozen sources,
controls and running/uncertain jobs remain intact. Full production K1/exactly-K4,
real-data quality, peak memory and matched-GPU speed remain unqualified until the
representative flow is finished, reviewed and frozen.

## Earlier reconstruction settings cleanup

The editable cached-main `a9e0668` checkout now binds regularization strength,
particle diameter and the initial filter in its existing run-level
`ReconstructionSettings`. Numbered/final priors, regularized/unregularized maps,
postprocessing and Class3D captures consume that owner. Changing windows,
accumulator metadata, first-CC decisions and model updates remain explicit.
No new module, function, forwarding layer or kernel was introduced. See the
[complete construction, callers and implementations](final_local_sampling_patch_review.md#reconstruction-settings-and-visible-model-updates).

58 focused cases and 19 affected controller cases pass (243 deselected, zero
skips). The final diagnostic-interface migration passes all 11 capture cases.
Pinned imports, import lint and diff checks pass. AST comparison confirms 64
existing operation bodies after enumerated interface substitutions, including
unchanged numerical expressions, casts, array operations and release ordering.
Exact commands, source identities, XML results and structural proof are in the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_reconstruction_settings_20261002T180508Z/HANDOFF.json).
The two command receipts capture unchanged source throughout each check;
the final capture receipt covers the capture-interface edit.

A subsequent dead-code audit removed the command's unused
`_validate_initial_noise_radial`. It was already uncalled on cached main, and
repository searches found no import, dynamic registration, CLI, notebook or
serialized-name consumer. The command module's AST is identical apart from that
definition. All 33 affected startup-noise, command and import checks pass with
zero skips; the final dead-code receipt covers this deletion. The scoped owner
inventory found production/script references for all 83 inspected functions;
that inventory is a screening tool, not a whole-repository dead-code proof.

At that source the numerical controller spanned **3,037 lines**, and command `main()`
**1,992**. Class3D aggregation remains inline pending the user choice about
releasing unused scratch after its last use versus preserving every current
lifetime. The extension plan records that choice; no unused-buffer collector was
added. The full representative refactor and its design review are unfinished.
Production K1/exactly-K4 quality, peak memory and matched-GPU performance remain
unqualified. No new GPU job, publication or merge occurred. Earlier frozen
candidates, controls, packets and running/uncertain jobs are preserved.

## Earlier numbered sampling cleanup

Code cleanup has resumed in the editable cached-main `a9e0668` checkout.
`convergence.py` owns native versus explicit angular transitions;
`iteration_planning.py` owns numbered perturbation resolution and independent
model/particle Fourier windows. The controller retains pre-update sizing order,
replay/accuracy timing, grid allocations and explicit state/history writes.
An unreachable matrix-only perturbation branch and its unused imports are gone.
See the [complete actual caller and implementations](final_local_sampling_patch_review.md#numbered-sampling-policy-and-fourier-windows).

71 focused cases, 87 affected caller cases and 108 CPU guards pass. Eleven
post-cleanup trial-grid/caller/import checks pass; structural comparison confirms
unchanged numerical grid blocks and scientific phase order. Documentation,
guide mirrors and engineering metrics pass their checks. The
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_numbered_sampling_policy_20261002T173238Z/HANDOFF.json)
records exact commands and source identities, including repaired fixture failures.

At that source the numbered controller spanned **3,052 lines**, and command
`main()` **1,992**.
The refactor is unfinished. Complete the representative calling flow and review
it before freezing another candidate for full production quality/performance
qualification. No new GPU job, numerical/performance acceptance or publication
occurred. The previous frozen candidate, controls, qualification packets and
uncertain/running jobs remain preserved; their evidence does not qualify these edits.

## Current cached-main synchronization

The editable checkout is now on cached main `a9e0668`. Four incoming commits
change resident physical-capacity projection/finalization and their tests, plus
benchmark reporting. None overlaps the refactor edits. The fast-forward preserves
those edits, the original dirty workspace/comments and every frozen candidate.
Remote refresh remains unverified.

Six focused crop/window CPU cases and 108 EM guards pass, with no skips. Pinned
imports, native-source hashes and guide mirrors pass. See the
[current handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T154930Z/HANDOFF.json).
The projector implementation and controller dimensions below are unchanged.
Incoming engine code requires fresh GPU qualification; the older checks do not
certify that path.

The immutable `7412fd8` package now also has a CPU-admitted real-data comparison:
84,266 particles in unchanged 42,133/42,133 halfsets, 1,642 stacks and 1,667 total
scientific files, including 21 reference maps and the frozen mask. Full file
hashes, row/half identity, both command parses and ten reporting/scratch-reader
cases pass. Its [preparation receipt](/scratch/gpfs/GILLES/mg6942/tmp/relax_projector_reuse_owner_20261002T134633Z/qualification/production/real10345/CPU_PREPARATION.json)
records CPU preparation only. No GPU job was submitted and no quality or speed
qualification is claimed. Preserve that packet and its prior failed-launcher
evidence while preparing the newer source.

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
The [current handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_projector_reuse_owner_20261002T134633Z/HANDOFF.json) records exact source identities,
commands and the two failed new-fixture receipts followed by their passing repairs.
The [previous main integration](/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T131551Z/HANDOFF.json)
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
[handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_initial_pose_owner_20261002T112730Z/HANDOFF.json).

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

Startup sampling and archive formatting retain their established owners.
[Previous package and main integration](/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T111111Z/HANDOFF.json)
remain source-specific evidence; the current package above identifies latest code.

## Current source and qualification

The editable checkout is
`/scratch/gpfs/GILLES/mg6942/tmp/relax_main_integration_20261002T030428Z/checkout`.
See the [principles](refactor_principles.md) and [extension plan](refactor_extension_plan.md).
Expectation has 26 parameters and final all-data has 38; preserve visible state
updates and preceding-iteration convergence while completing those boundaries.

## Preserved findings that still matter

| Source-specific evidence | Current interpretation |
| --- | --- |
| Older K1 pair: control 18 iterations/HP7, candidate 16/HP6 | Unresolved convergence finding. Trace the first divergence with matched float32 inputs/state/candidates; it has not been dismissed as roundoff. |
| Older K4 pair: 15 iterations and strong class FSC | Preserved evidence for that source; one cold pair does not qualify latest quality or speed. |
| Optics K1 pair 14828256: both arms 18 iterations; cross FSC-AUC about 0.999713; GT AUC delta about -0.000035 | Preserved terminal CLI results for that source; one paired timing is insufficient for speed qualification. |
| H100 smoke 14840134 and 14841713 | Passed on their older frozen sources, not the latest combined candidate. |
| EMPIAR-10345 launcher | Earlier failure corrected at CPU admission; latest real-data GPU run remains unexecuted. |
| Pose-package sbatch observation timeout, no job ID | Remote state remains unknown. Inspect scheduler/accounting before a retry; missing local output does not establish that no job started. |

The complete source identities, measured values, commands, job IDs and negative
findings remain in the immutable [previous status][previous-status] and its
linked receipts. Original outputs and sources have not been changed.

## Historical source receipts

The headings below retain existing links. Their evidence and old next actions
describe the recorded source; the current section above is the active handoff.

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

## Earlier particle-pose operation

[Recorded source and evidence][previous-status].

## Cached-main integration and controller audit

[Recorded source and evidence][previous-status].

## Earlier source-specific receipts

[Recorded source and evidence][previous-status].

## Review and remaining controller work

[Recorded source and evidence][previous-status].

## Follow-up: dense execution defaults

[Recorded source and evidence][previous-status].

## Checks by frozen source

[Recorded source and evidence][previous-status].

## Direction-prior ownership follow-up

[Recorded source and evidence][previous-status].

## Noise-model lifecycle follow-up

[Recorded source and evidence][previous-status].

## Reference-model and class-prior estimation

[Recorded source and evidence][previous-status].

## Expectation batch preparation

[Recorded source and evidence][previous-status].

## Complete numerical scoring results

[Recorded source and evidence][previous-status].

## Numbered sampling and empty expectation preparation

[Recorded source and evidence][previous-status].

## Numbered half expectation and explicit publication

[Recorded source and evidence][previous-status].

## Final SPA half priors and optics

[Recorded source and evidence][previous-status].

## Command result files and publication

[Recorded source and evidence][previous-status].

## Numerical finding: older K1 convergence differs

[Recorded source and evidence][previous-status].

## Real-data launcher repair

[Recorded source and evidence][previous-status].

## Earlier evidence, preserved

[Recorded source and evidence][previous-status].

## Source, reproduction and next actions

[Recorded source and evidence][previous-status].

[previous-status]: /scratch/gpfs/GILLES/mg6942/tmp/relax_initial_model_replay_20261002T091611Z/source/candidate/docs/development/final_search_patch_status.md
