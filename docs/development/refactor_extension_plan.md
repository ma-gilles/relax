# Extending the refinement refactor

This is a proposed sequence following provisional acceptance of the optics
boundary. The [complete example](final_local_sampling_patch_review.md) now shows
more surrounding context; its [qualification receipt](final_search_patch_status.md)
tracks the open numerical gates. Apply the
[agreed principles](refactor_principles.md) to every package below. Production
quality and performance remain unqualified until the recorded gates pass.

The review criterion is how much surrounding code a developer or agent must
understand to change one responsibility safely. Keep each coherent operation
with its private helpers and small result types. Every extracted layer must own
a decision, computation or necessary adaptation. File length and function count
do not determine success.

## Active deeper design pass — October 3

The frozen 80f2b2 candidate is a numerical checkpoint. The user has not accepted
its conditional complexity or the complete readability design. Existing medium
14928923 and K1 14912912 continue against that immutable source. Do not launch
ordinary performance repeats for a design still being revised. All final quality,
performance and delivery requirements below remain in force for the final source.

### Execution plan and model policy — October 3

This section supersedes the planning-only status below. The user authorized
bounded agent implementation, then requested a complete plan and explicitly
prohibited Astra-high implementation. The whole architecture remains the scope;
separating two reconstruction functions is only the first prerequisite.

Use Sol at medium reasoning for routine implementation, migrations and focused
checks, honoring this session's earlier Sol preference. The primary owns the
architecture, integration, scientific acceptance and job registry. Reserve
expensive architectural reasoning for unresolved design decisions and milestone
review; do not assign it routine coding or test waiting. Scripts observe jobs.
Only one agent writes any given source file. Maximum active team is primary plus
three workers. Use short task packets and isolated worktrees; no full-history
forks for new implementation workers.

#### Packages, dependencies and acceptance

| Package | Work and concrete deliverable | Dependency / acceptance criterion |
| --- | --- | --- |
| P0 — capability and temporal contracts | Matrix of production, diagnostic, rejected and unresolved routes; side-by-side K1/Class3D transition ledger; direct/programmatic/qualification caller inventory | Before controller split. Every proposed deletion names its callers and reachability evidence; replay-installed and computed state are accounted for. |
| P1 — coherent scientific operations | Distinct numbered half/class reconstruction; review prior estimation, sampling, expectation and noise contracts; remove repeated adaptation and historical call-site flags | First reconstruction patch can proceed now. Review caller and complete implementation; no ambiguous half/class axis and no duplicated numerical formula. Preserve final versus numbered policy differences. |
| P2 — persistent and consumable ownership | Separate scientific reference models from particle execution partitions; explicit pose/noise/sampling ownership; transfer/release accumulator owners and aliases at actual last consumers | Uses P0/P1. Producer-consumer/lifetime table, tests of state publication and buffer ownership. Preserve zero-copy class aliasing until separately qualified; no generic context object. |
| P3 — explicit scientific trajectories | Separate K1 auto-refinement and Class3D controllers from coherent operations; keep VDAM separate and modality-specific expectation within each trajectory | Uses P0–P2. Complete flow review proves original convergence, replay, RNG, sizing, first-CC and checkpoint timing. No copied numerical engine; shared orchestration retained only when its meaning is stable. |
| P4 — configuration, modality and interventions | Ordinary startup state outside ReplayState; expected accuracy outside debug; meaningful execution/window settings replacing historical DenseVariantPolicy fields; distinct SPA/tomo preparation and capture contracts | Ownership decided with P0/P2; integrate serially with P3 where files overlap. Input admission rejects invalid stable combinations, while restored/computed-result checks remain. Observation cannot silently alter computation. |
| P5 — simplify and retire | Remove redundant branches, obsolete APIs, duplicate sources of truth, dead families and forwarding-only layers; migrate callers/tests/docs; targeted style cleanup | After relevant replacement contracts exist. Each retained branch has an owner and reason. Diagnostic Class3D final paths and live comparison engines stay until explicitly replaced or retired with usage evidence. |
| P6 — review, qualify and deliver | Complete calling-flow artifact, reusable principles and extension guidance; final frozen source/control; scoped scientific ladder, K1/exactly K4/real quality, memory and ordinary matched-GPU speed; main integration | No completion while required gates or design defects remain. Current checkpoint runs qualify only their source. Review the full architecture before expensive final repeats. |

P1 is not limited to moving code. For each operation, record what information its
caller no longer needs to know, what branching/argument adaptation disappeared,
and what numerical or execution choice remains intentionally visible. Avoid new
objects whose only purpose is to gather locals at one call site.

P2 must trace all accumulator aliases, including `PerHalfOutputs` and the last
`HalfScoreResult` in finalization. Clearing local names alone does not establish
release. Measure lifetime/performance changes separately from mathematically
identical code extraction. Do not call retained memory a leak without evidence.

P4 includes a separate bug investigation for final tomography intermediate
capture referencing SPA-only `final_inputs`. Reproduce before fixing; do not
fabricate SPA operands. Do not bundle a confirmed behavior fix invisibly into
structural equivalence claims.

#### How a package executes

1. Primary assigns one falsifiable objective, source fingerprint, exact writable
   files, CPU/GPU budget, expected checks and stop condition. Audits are read-only.
2. Sol implements in an isolated worktree, migrates maintained consumers and runs
   focused checks. It submits no Slurm jobs and changes no baselines or tolerances.
3. Primary reviews full caller/owner/state/lifetime changes and verifies evidence.
   Resolve material design choices with concrete examples; no per-edit approvals.
4. Publish cohesive review-branch updates with executed/pending checks explicit.
   Preserve current freezes/jobs. Qualified pieces land on main under repository
   gates; user-authorized draft review publication does not imply acceptance.
5. Advance the dependent package. Do not parallelize changes to the same controller
   or have multiple agents independently redesign shared types.

Initial team: one Sol reconstruction writer (P1), read-only workflow/capability
auditor (P0), read-only ownership auditor (P2). When audits finish, use Sol workers
for independent migrations/tests or the next disjoint package. Primary handles
shared-controller integration and architecture review. Do not keep an expensive
model active solely to poll jobs or narrate progress.

#### Goal and completion

Retain the existing full completion goal. This plan adds architecture milestones
without weakening final-source scientific/performance or main-delivery requirements.
No new goal is needed to schedule subagents. The user need not repeatedly return
a replacement goal to keep implementation moving. Report package status, evidence,
remaining design decisions and exact live jobs in one concise handoff.

### Architecture review basis — preceding planning phase

The user requested big-picture planning before implementation and supplied an
independent static review of commit `cad2012`. At that planning phase no implementation was authorized. The execution section
above records the subsequent bounded implementation authorization. Existing frozen checkpoint jobs remain unchanged.

Working recommendation: distinct K1 auto-refinement and Class3D scientific
controllers; shared scoring, joint posterior, backprojection and reconstruction
machinery. Tomography gets meaningful expectation/preparation contracts within
the appropriate scientific trajectory, not initially duplicate controllers.
VDAM remains a separate optimizer/workflow using shared numerical operations.
No base-refiner hierarchy, generic pipeline framework or event bus is proposed.

The reason to split is different scientific state and progression, rather than
branch count. K1 owns independent half maps; Class3D owns one reference stack
scored through two particle partitions. The existing two-slot storage may remain
at an adapter boundary during migration, without array copies. Such an adapter
must have explicit consumers and a retirement condition, not become a permanent
second source of model state.

#### Evidence checked locally against the review branch

| Finding | Current evidence | Remaining uncertainty |
| --- | --- | --- |
| Model slots have different scientific meanings | `ReferenceModel`, `initialize_reference_model`, `reference_model_from_snapshot`: class restoration aliases map slot 1 to slot 0 | Trace every restart/publication consumer before changing representation |
| Generic reconstruction reselects the mode | At `932c72d`, `reconstruct_regularized_means`: solve, filter, scalar mask arithmetic and flatten branches. Since replaced by `reconstruct_numbered_k1_halfmaps` and `reconstruct_numbered_class_maps`, which hold no mode test | Preserve class postprocessing through both slots; deduplication changes execution |
| Accumulator collectors retain arrays | `PerHalfOutputs.update_from` stores `Ft_y` and `Ft_ctf` references | Full reference graph and measured GPU/host memory impact remain to establish |
| Final tomography capture depends on SPA-only preparation | `run_final_all_data`: `final_inputs` assigned in SPA arm and used by common intermediate manifest | Static defect candidate; establish reachable admission and reproduce before a separate fix |
| Scientific configuration has misleading ownership | `EngineDebugOptions.expected_accuracy`, historical fields in `DenseVariantPolicy` | Trace producers, defaults, precedence and direct programmatic callers |

The external review did not execute numerical checks. Its architectural findings
are not qualification evidence. One gate description needs correction: the
repository requires K1 and exactly K4 production workloads of at least 100k/256,
with the synthetic K1 trajectory followed by characterized real-data confirmation.
It does not require both completion fixtures to be real data. Preserve the actual
approved fixtures and gates; do not silently substitute a new validation objective.

#### Required planning deliverables before source changes

1. Capability matrix covering CLI, programmatic, replay and qualification callers:
   ordinary supported, supported diagnostic, rejected, and unresolved combinations.
   Do not equate untested with unsupported or obsolete.
2. Side-by-side K1/Class3D transition ledger: consumed completed state, decisions,
   writes, next consumers, captures, RNG advancement and release/completion points.
   Include preceding-iteration convergence, incoming local sizing order, numbered
   versus final perturbation precision and current versus next scale-gating spectra.
3. Ownership map separating scientific references from execution partitions,
   persistent particle state from iteration results, and borrowing from consumption.
   Identify actual owners/aliases before designing a `take` or `consume` interface.
4. Target calling flow and operation contracts for the hybrid, compared with a
   shared controller. Demonstrate representative convergence, class-prior and
   tomography changes. Account for duplicated orchestration and shared invariants.
5. Configuration/intervention map: startup source state, scientific controls,
   execution controls, passive observations and intrusive interventions. Replay may
   return to native progression; do not create a separate replay engine by default.
6. Removal ledger: obsolete candidates with caller evidence; live comparison paths
   to retain; missing-shell fallbacks to investigate; reachable Class3D finalization
   paths. A recommendation is not permission to drop a supported scientific route.
7. Cohesive migration and validation plan, reviewed with the user before execution.
   Separate confirmed bug fixes and execution optimizations from structural work.

#### Proposed migration sequence after plan acceptance

- Establish capability and transition contracts.
- Complete explicit numbered half-map and class-map reconstruction operations.
- Separate persistent reference ownership and define accumulator consumption.
- Separate K1/Class3D controllers using those operations; provide modality-specific
  expectation preparation without multiplying global/local/backend controllers.
- Rationalize configuration and interventions; retire proven obsolete interfaces.
- Review the complete example, freeze, qualify final float32 quality and matched
  GPU memory/speed, then deliver under the existing integration rules.

Each stage must preserve scientifically timed state publication and numerical
execution. First-class continuation remains supported. Comparison backends and
intrusive research modes should stay available until their users and contracts
are established; isolation is not deletion. The status of Class3D diagnostic final
passes is an open capability question, not settled by native fixed-iteration use.

### First package: reconstruction mode ownership

Evidence when this package was planned (`932c72d`): `iteration_loop.py` selected
K1 versus Class3D operands before calling
`mean_helpers.reconstruct_regularized_means`. That operation dispatched again for
the solve, first-CC filtering, solvent-mask scalar arithmetic, dtype selection
and flattening. Its `numerators`, `denominators` and `tau` arguments had
different axis meanings by mode. A maintainer had to inspect both layers to know
the contract.

Current code: steps 1 to 5 are implemented. `reconstruct_numbered_k1_halfmaps`
and `reconstruct_numbered_class_maps` replace that function, each with
accumulators of one layout and shell-curve priors only, and the controller
dispatches once at this boundary. The
[choice inventory](final_local_sampling_patch_review.md#reconstruction-choice-inventory)
traces every producer and consumer; it showed the full-volume prior route of the
numbered operations to be unreachable in production, so `tau_is_1d` is removed
from them and kept only on the eager solve, which the final K1 path calls with a
full volume. Each operation writes out its own capture, first-CC low-pass, mask
and flatten sequence over its two slots, with no mode flag below the dispatch;
premask capture, the diameter test, the mask radius and construction and the
closing log are shared helpers without a mode argument, and the class operation
has its own per-class low-pass and flatten helpers beside the K1 ones. The
`run_mean_reconstruction` test builder is removed and each test calls the operation it exercises. The class
operation still postprocesses both execution slots. For step 4, the
[finalization mode-guard map](final_local_sampling_patch_review.md#finalization-mode-guard-map)
lists every mode test by entry path; `run_final_all_data` now enters the K1
unfiltered reconstruction, optional low-resolution join and collector release
through one mode decision, in their previous order, with the join resolution
as a nested option. The final prior and final solve remain separate decisions
because shared sums and the shared resolution update lie between them. The
step 6 review with the user has not taken place.

Package 2, lane A (the user chose on October 3 that a one-line shared statement
may be written in both arms of a controller decision, with nothing reordered):
the numbered M-step, from the accumulator audit to the `"recon"` stage mark, is
one K1/Class3D decision instead of seven. Each arm lists its own ordered steps:
accumulator combine or optional low-resolution join, previous references, prior
estimate, tau2 install, reconstruction, optional first-CC reporting taper and,
for K1, the tau2 volumes' move to the host. The
[M-step mode map](final_local_sampling_patch_review.md#m-step-mode-map) gives
the before and after. `estimate_iteration_resolution` is replaced by
`estimate_k1_iteration_resolution` and `estimate_class_iteration_resolution`,
called from the existing class-assignment decision; the K1 FSC fallback, which
no prior estimate could reach, is removed. The Class3D rule (each class's shell
without the split-half recheck) has one home, `class_resolution_shells`, which
the class operation, the final pass's `relion_current_resolution_shell` and
`plan_class_image_size` call. The profile-only return passes no
class products, the direction-prior order is computed once per iteration, and
`_maybe_host_offload_half0_local_accumulators` and `record_noise_and_tau2` no
longer take the mode. `refine_single_volume` tests `k_class_enabled` 29 times
instead of 39.

The user then ruled that unreachable or production-unset code is deleted under
the cleanup rules. Removed on that ruling, each after tracing its producers:
the `do_solvent_fsc_correction` option with the block it gated in
`estimate_split_half_prior` and the solvent-corrected FSC kernel in
`regularization_relion.py` (no producer on any entry path);
`_k1_data_vs_prior_for_scheduling` and its raw-FSC arm (the K1 curve is always
set before `plan_halfmap_image_size` runs), with `plan_class_image_size` now
calling the one truncation helper; the Class3D shared-prior fallback in
`relion_direction_log_priors_for_half` (no Class3D path fills `shared`); and
the always-`None` `state_current_size` operand of `audit_prejoin_accumulators`.
Two changes alter execution without altering values: the K1 host move no
longer copies the two per-half reconstruction tau2 volumes that nothing reads
again, and a continued run no longer builds the start-up reference model,
noise model, previous rotations, direction priors and sampling perturbation
that its snapshot replaces (its "Perturbation init" log line no longer
appears). Still present: `SplitHalfPrior.fsc_for_update`, now always the raw
FSC, because removing it reaches the growth history and the snapshot format;
and the class log priors, still computed before a continued Class3D run
replaces them.

1. Inventory each reconstruction choice with its producer and consumers:
   split-half versus combined classes; prior volume versus shell curves;
   first-CC filter/taper; optional solvent mask; host staging/large-box behavior;
   diagnostic capture and replay; final versus numbered reconstruction.
   Label each as scientific, execution/storage, input validation or diagnostic.
   Mark a case obsolete only after tracing supported entry paths and consumers.
2. Replace the ambiguous numbered reconstruction interface with explicit K1
   and class operations in the existing reconstruction owner. Each owns its
   solve and ordered postprocessing. Reuse existing lower-level mathematics;
   do not build a strategy hierarchy or add forwarding-only methods. The
   controller selects the scientific operation once at that boundary and visibly
   installs the resulting maps. Mode-specific operands remain mode-specific.
3. Keep prior estimation and important state publication visible: Class3D's
   previous-reference power prior differs from K1's current split-half FSC.
   Keep reference-map release before the solve and first-CC reporting taper
   after it. Do not merge branches across diagnostics or writes without tracing
   their timing. Review shell/full-volume fallback producers before retirement.
4. Consolidate finalization's repeated K1 guards for unfiltered reconstruction,
   optional low-resolution join and old accumulator reference release as one
   ordered sequence. Final reconstruction retains its separate scientific policy.
5. Migrate the actual controller, test builder and source-inspection tests in
   one package. Tests must check ordering, operand layouts/precision, shared
   class-map identity, host staging/donation and postprocessing behavior; no
   test should require retaining an obsolete function name.
6. Present caller, implementations and state/lifetime map together in the existing
   review document. Explain which mode cases and dependencies disappeared.
   Review this complete example with the user before extending the pattern.

### Decisions that must survive the cleanup

| Decision | Producer/lifetime | Required preservation |
| --- | --- | --- |
| K1 versus Class3D | Validated class count, run invariant | Distinct prior formulas and map axes; no generic optional-field payload |
| First-CC phase | Numbered iteration state | Filter before solvent masking; reporting taper after solve |
| Prior shell representation | Prior estimator/replay, per iteration | Trace every producer before removing full-volume routes |
| Retained first numerator | Low-resolution joining, one reconstruction | Release after half-0 reconstruction at the established completion boundary |
| Previous maps | Reference model, crossing iterations | K1 snapshot/release and Class3D borrowed device references differ |
| Unfiltered final maps | Pre-join final accumulators | Reconstruct before mutation; preserve host/device memory limits |
| Diagnostic captures | Optional run settings, phase-specific | Preserve pre-mask/pre-join operands and capture ordering |
| Convergence/sampling | Evolving iteration state | Keep preceding-iteration decisions at their current time |

### Next packages after example review

Apply the same producer-to-consumer audit to numbered sampling, replay/startup
precedence and expectation input admission. Separate unsupported combinations
from checks on newly computed engine results. Remove redundant invariant checks;
keep validations whose truth can change after replay or iteration updates.
Do not replace visible branches with dispatch tables merely to reduce counts.

### Verification and completion

Use focused existing reconstruction/lifecycle/first-CC tests while implementing.
Record changed paths and unresolved GPU/memory concerns. Current checkpoint jobs
only qualify their frozen source. After complete-flow review, freeze the next
candidate, run its required scientific ladder and production K1/exactly K4 gates,
then perform the approved repeated matched-GPU memory/speed qualification.
Update standing evidence and integrate with main under the existing contracts.

## Completion milestone approved October 2

| Requirement | Evidence required for completion |
| --- | --- |
| Prior estimation, reconstruction, postprocessing, taper and input preparation have substantive owners | Actual complete controller flow, implementations, producer/consumer/lifetime review; no locals context or forwarding chains |
| Numerical/runtime contracts preserved; unused temporaries expire after last use | Focused regressions and scientific checks on the final float32 source; recorded array/layout/dtype/reduction/RNG/JIT boundaries |
| Obsolete EM interfaces retired | Direct and indirect source/test/script/notebook/serialized-name inventory, migrated callers and collected affected tests |
| Reusable model for further refactoring | Current complete-flow review, accepted principles and responsibility-based extension plan in existing documents |
| Production quality, peak memory and speed accepted | Frozen control/candidate provenance; scoped scientific ladder and K1/exactly K4 production evidence; matched GPU models and approved gates without tolerance/baseline widening |
| Qualified delivery synchronized with main | Verified integration state and checks at delivered source, exact receipts and no required validation pending |

The user explicitly authorizes unused temporary expiry after last use. Complete
all code and documentation before freezing; preserve original comments, existing
candidates, controls and jobs. CPU checks alone do not establish completion.

## Finish and qualify the example

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
immutable source. The never-started K1 pair **14910610** was cancelled at the user’s explicit
request on October 3. These jobs do not
qualify the incoming engine changes. The unchanged native-source digest permits
reuse of the already built immutable libraries; runtime library/GPU checks still
run before and after new qualification.

Current integration, receipts, preserved jobs and required gates (was `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/refactor_finish_sync_69abd77_20261003/HANDOFF.json`; no longer available)
track the new candidate. Fresh same-source smoke/medium, float32 K1 and exactly
K4 quality, characterized real-data confirmation, ordinary paired memory/speed,
the milestone long tier and qualified main delivery remain required. No
production quality or performance acceptance is claimed.



Current order, clarified by the user on October 2: finish the representative
code with focused checks, review the complete calling flow, freeze the accepted
candidate, then run production numerical and performance qualification. Keep
existing frozen candidates, prepared packets and running/uncertain jobs intact.
Do not substitute benchmark preparation or repeated access polling for the
remaining code cleanup.

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
Source spans are 2735/1766 for numerical/command controllers;
these counts are review signals, not design acceptance. Current CPU checks are recorded above; earlier passing receipts describe their
own source only. The milestone is incomplete until the frozen float32 K1/exactly-K4
scientific, real-data, memory and matched-GPU speed gates pass and delivery to main
is verified. Existing comments, frozen candidates, controls and jobs are preserved.

### Next extension after representative review and qualification

1. Review this complete producer/controller/owner example in context. Obtain user
   design feedback before extending the pattern across unrelated responsibilities.
2. Apply the same ownership method to a different operation, such as an existing
   VDAM state update: trace its representations and actual lifetimes first, then
   decide whether a shared primitive or a distinct scientific policy is justified.
3. Automate inventories, caller migrations and source/receipt checks only after
   both examples are accepted. One writer owns each source/build; immutable jobs
   are never repointed at moving worktrees.
4. Qualify each affected scope using unchanged numerical and performance gates;
   publish cohesive accepted work to main rather than accumulate completed branches.

### Earlier extension reasoning and source-specific receipts

The sections below record earlier proposals and receipts. Their old controller
counts and pending lifetime decisions are historical. The October 2 expiry approval
and completion criteria above govern the current milestone.

Global and local scoring now share host cell-offset layout in the existing
candidate-table owner. The [complete operation and actual callers](final_local_sampling_patch_review.md#shared-candidate-chunk-layout)
retain representation, placement, kernel order and lifetimes; 198 selected CPU
cases pass. The whole-private-callgraph and substantial exact-body duplicate
screens are recorded with their limits in the package handoff. Further ownership
work and full-flow review remain required; these screens do not certify all
dynamic consumers or duplicated mathematics.

The dead-code audit retired six disconnected private declarations for a future
capacity planner and three unused import bindings. The [review and evidence](final_local_sampling_patch_review.md#retired-private-capacity-planning-family)
show the closed-family reachability check and unchanged live planning operations;
43 affected CPU cases pass. This is scoped evidence, not whole-codebase closure.
Current controller/command spans remain 2,883/1,860. Finish reconstruction ownership
after resolving the pending lifetime decision, then review the full flow before
freezing for expensive qualification.

Half image preprocessing now shares the existing particle-input owner. The
[actual call and complete implementation](final_local_sampling_patch_review.md#half-image-preprocessing-ownership)
retain setup timing, backend/refusal/mask order and shape-class/reference pixel
semantics. Only borrowed aliases leave controller scope; existing half/dataset
owners retain numerical storage. Current controller/command spans are
2,883/1,860, with 80 affected CPU cases passing. Continue phase ownership
and complete-flow review; the Class3D scratch-lifetime choice remains pending.

Replay now consumes the existing validated input geometry; the controller's
generic dataset alias and two invalid-pixel substitutions are retired. The
[actual producer, call, publication and consumers](final_local_sampling_patch_review.md#input-geometry-and-replay-boundary)
show fixed geometry separately from the original pixel scalar required by host
promotion. Current controller/command spans are 2,917/1,860. The 185 focused CPU
cases pass; remaining phase ownership and the Class3D lifetime choice still
precede representative-flow acceptance and expensive qualification.

Sealed restart runtime adaptation now shares its existing diagnostic CLI owner
with invocation/source admission. The [actual frozen branches, capture caller and
four implementations](final_local_sampling_patch_review.md#sealed-restart-runtime-adaptation)
retain noise installation, ordinary versus frozen replay selection, capture
publication and validation timing. No new operation or container is added; every
moved numerical body is unchanged. Diagnostic slot/dtype/capture changes can
leave native refinement and normal command admission unopened. Main controller
spans remain unchanged; continue the substantive phase-ownership work rather
than treating relocated functions as controller completion.

Numbered expectation now prepares one shared trial-grid/sampling/policy result
in its existing owner. The [complete current construction, implementation and
half dispatch](final_local_sampling_patch_review.md#numbered-expectation-preparation)
keep mode selection, tomography adaptation, half-specific computation,
publication and the memory boundary visible. The scorer has 19 inputs and the
producer 18; the half worker still captures 34 names. Trace remaining operands
to their real owners rather than adding them to this phase result. The current
checks establish source transport and focused CPU behavior, not production
quality, memory or performance acceptance.

The command-options owner now owns verified dispatch admission and
CLI/optimiser runtime-control resolution. The [complete current caller and both
operations](final_local_sampling_patch_review.md#command-oracle-admission-and-runtime-controls)
retain group-source identity, strict replay admission, frozen-aware optimiser
selection, follower preparation and visible installation of scientific controls.
Runtime results feed both execution and saved provenance. No new module or
numerical operation is added. This isolates command admission details while
reference loading and numerical phase construction remain unfinished.

The current correction-reporting package gives checkpoint averages and parity
captures one small reporting result with the existing normalization owner. The
existing diagnostic module owns capture encoding/conversion and warning handling.
The [complete current caller and implementations](final_local_sampling_patch_review.md#correction-reporting-and-completed-iteration-capture)
retain normalization admission, native/follower correction writes, convergence,
checkpoint-before-capture order and full/timing-only selection. This adds one
result type and one substantive adaptation function, with no module or forwarding
method. It preserves native-result/group-ID roots and keeps rank-1 reporting
separate from each particle's installed follower scales.

The current half-input pass reuses `HalfScoringData` for its borrowed radial
noise and local/dense sampling results for coarse angular metadata. Particle
diameter comes from the existing required run schedule. No new layer or type
was introduced. The [complete current caller and operation](final_local_sampling_patch_review.md#numbered-half-input-ownership)
show all three producer/consumer boundaries and the preserved empty/tomography
noise guard. Its earlier 23-input interface and 38 captures prompted the current phase
preparation package; do not replace them with an object assembled from all controller locals.

The existing expectation owner now records half profiles, support counts and
E-step captures through `record_numbered_half`. `SignificanceStatistics` owns
per-half counts and the two distinct aggregate policies; it preserves recording
order under overlap and combines after the existing preprocessing drain. The
controller retains offloading, payload/metadata installation and the recording
invocation. See the [complete actual caller, owners and downstream consumers](final_local_sampling_patch_review.md#half-expectation-recording-and-support-counts).
This adds no module or numerical kernel. Further phase construction still
requires producer/lifetime review; do not group all
remaining locals merely to reduce captures.

The existing `ReconstructionSettings` now owns regularization strength, particle
diameter and the initial reference filter alongside reconstruction geometry.
Numbered/final priors, MAP reconstruction, unregularized maps, postprocessing and
Class3D captures consume it without repeated overrides. The
[complete construction, calling flow and implementations](final_local_sampling_patch_review.md#reconstruction-settings-and-visible-model-updates)
show which run invariants belong together and which changing phase inputs remain
explicit. This adds no module, function, forwarding layer or numerical kernel.

Class3D prior aggregation still retains its per-class full-grid arrays after
stacking. A stage returning only the aggregates releases those arrays earlier;
retaining unused arrays in a result merely to emulate the controller is not an
acceptable design. The user has a pending concrete choice about allowing unused
scratch to expire after its last use versus preserving every existing lifetime.
The unresolved constraint comes from the [root contract](../../AGENTS.md), which
requires preserving memory lifetime during structural cleanup.
Keep this aggregation in place until that choice is resolved. Do not move model
writes or retain all class weight summaries to work around it.

Numbered sampling now separates angular-transition policy in `convergence.py`
from perturbation/window planning in `iteration_planning.py`. The controller
retains the pre-update order, replay and accuracy timing, grid allocations,
local/adaptive selection and explicit state/history writes. The complete
producer/caller and policy implementations are in the
[existing review](final_local_sampling_patch_review.md#numbered-sampling-policy-and-fourier-windows).
This package introduces no new module or numerical kernel. It does not complete
sampling phase construction or the broader controller.

The refactor is staged over cached main `a9e0668` in an isolated checkout;
the shared dirty workspace and all earlier frozen sources remain preserved.
See the [current integration status](final_search_patch_status.md#current-cached-main-synchronization).
Incoming physical-capacity projection/finalization code did not overlap the
refactor edits. Its six focused CPU cases and the 108 EM guards pass; the combined
source needs fresh GPU validation. Preserve the already-prepared `7412fd8` jobs
and admitted real-data packet as evidence for that source, rather than relabeling
them as qualification of the newer engine.
The subsequent pose and follower-preparation packages extend the controller
sources; incoming coarse-posterior/native code and the combined candidate require
fresh qualification. The command and numbered controllers still need substantial
responsibility boundaries.

The example covers particle-half ownership, prepared projectors, final sampling
resolution, dense/local scoring inputs and final reconstruction stages. Shared
startup formulas and adaptive grids have neutral owners used by refinement and
VDAM. Reference maps/tau2, noise and direction priors now have persistent owners in
`mean_helpers.py`, `noise_updates.py` and `orientation_priors.py`. Their producers,
replay/restart paths, updates and consumers are shown in the existing review.
Substantial controller operations remain the next design work.

The reviewed optics package gives optics preparation and its result types the
existing `optics_shapes.py` owner. Numbered and final callers receive named
class-pixel operands directly; the obsolete forwarding route is retired.
The user provisionally accepted this ownership boundary, with a request for
more surrounding context. The review now includes native convergence and the
numbered local branch alongside the producer and dense/local consumers. Model-half
migration has established reference/tau2, direction-prior and noise owners while numerical qualification is unresolved. After this example is accepted, test the principles on a different
responsibility before dispatching broad automation.

The numerical controller still spans 2,913 lines; the command entry point spans
1,860 lines. The 59-line half worker captures 34 surrounding names.
These are concrete coupling signals, and this is not the target calling flow.
Its responsibilities should become initialization, iteration planning,
expectation, reconstruction/prior estimation, parameter and convergence updates,
checkpointing, and finalization. Extract those stages after their state has real
owners; an object assembled from every local variable would conceal the coupling.
The number of imports should fall as responsibilities move, without hiding
unrelated operations behind a namespace to reduce an import count.

Preserve the controller's existing execution order and convergence decision
timing, including decisions based on the preceding iteration. The controller
must show important mode decisions, state updates and the transition to
finalization. Extraction must not move a decision earlier or later merely to
make a proposed loop look regular.

Image-size planning now isolates initial, split-half and Class3D policy in the
existing iteration planning owner. Review its complete producer/caller and the
separate raw/growth signals before automating this pattern. Convergence, explicit
scheduling updates, current-size oracles and replay/angular order remain visible.
The 14-input half-map planner still warrants producer/lifetime review; shortening
that interface alone is not the next objective. Its CPU checks do not qualify
production quality or GPU performance.

Class3D capture now owns floor-shell diagnostics and file serialization in the
existing reconstruction diagnostic module. Its caller passes the established
prior result and reconstruction settings at the original per-class gate. The
larger prior aggregation still has several live per-class arrays; trace their
release sites before extracting the loop. Do not replace it with a callback or
collector that retains every weight statistic or hides model updates merely to
reduce the controller's size.

Numbered and final Class3D prior estimation now share `estimate_class_prior`
with the existing mean/reconstruction owner. Replay admission, first-iteration
projector selection, CTF correction and stacked output precision remain explicit
in their callers. Its implementation preserves the timing of reference/projector
and denominator views; replay does not materialize unused reference slices. Similar VDAM size-growth and mask
code has different rounding, precision or buffer lifetime; these are separate
numerical investigations, not mechanical substitutions in this package.

Expectation batch preparation now has an operation owner in
`expectation_batches.py`. It resolves compact allocation admission, separate
coarse/fine callbacks and per-shape batch plans. Numbered and final scoring share
the same run planner; available memory is read at invocation time. The existing
memory estimator remains in `helpers/batch_planning.py`. The controller still
selects the engine and constructs its policy explicitly. This is a bounded
part of expectation, not the completed expectation stage. Its actual caller,
implementation and consumers are in the existing review; GPU qualification of
this newer source remains open.

Scoring inputs now contain numerical operands, and adapters return complete
`HalfScoreResult` values, including poses and a coherent class summary. The
numbered/final controller records those values explicitly. The shape adapter
preserves the existing release of unused class summaries between shape calls.
This resolves a prerequisite for extracting the remaining expectation operation;
diagnostic policy and array/reduction ownership still need to retain their timing.

Numbered local sampling now has a substantive producer in the existing
`local_sampling.py` owner. Its `LocalSampling` result is reused by half scoring,
tomography, empty-half statistics and pose export instead of rebuilding it from
parallel locals at each call. Source Euler rows remain paired with generated
matrices. Numbered and final preparation preserve their different window-order
and supplied-grid policies. Complete zero payloads belong to the existing
scoring-result owner; their selection, recording and diagnostics remain visible
in the controller. These are prerequisites for the remaining expectation
operation, whose broader extraction and GPU qualification remain open.

Numbered half expectation now has a substantive operation in `expectation.py`:
it owns half-specific priors, batching, class seeding, optics and dense/local/tomo
accumulation. The controller retains phase/mode construction, serial/overlap
dispatch, explicit publication, offloading, diagnostics and the transition to
reconstruction. Results carry the pose frame and translation base. Phase
containers are released at the existing iteration memory boundary. Review the
complete construction and consumer in the existing artifact; the operation's
23-parameter interface and broader phase configuration still need review, and
latest-source GPU qualification remains open.

Final SPA half preparation now belongs to the same expectation owner. Its
small `PreparedFinalHalf` result keeps computed translation/direction/optics
operands together for local/dense scoring and manifest export. The controller
retains mode/policy selection, tomography, native pose export, publication and
diagnostic order. Final all-data execution now has a separate controller, described below; its interface remains unfinished. The canonical refinement entry point was inspected: it already takes five
numerical inputs and grouped options. No legacy flat settings interface remains.
An earlier proposed next action was stale; preserve the existing canonical API.

Command result publication now has a substantive `result_files.py` owner for
archive layouts/compression, profile conversion and final maps. The command
retains source metadata and the archive/ledger/map/summary sequence. Its small
`ArchiveReport` contains computed values reused by the archive and ledger;
caller-owned serialized arrays retain their lifetime. Diagnostic admission now
has a coherent CLI owner in `diagnostics/frozen_boundary_cli.py`: flag registration,
source/arm validation, resolved-config binding and sealed runtime adaptation stay
with their private helpers.
The command retains seed and experiment setup order and both validation timings.
Split-half prior estimation is now a substantive reconstruction operation.
Trace Class3D prior aggregation or the remaining map phase next; the command and
numerical controllers are still unfinished.

Review construction, consumers and lifecycle updates together. A short call
which reconstructs an object from twenty parallel lists is not an exemplar.
Require one owner per concept, resolved settings at the right lifetime, and a
documented reason for every remaining argument and conversion. Preserve
canonical Euler metadata, precision, operation order and buffer lifetime.

Acceptance evidence for this package:

| Requirement | Evidence required |
| --- | --- |
| Readable calling flow | User review of the complete extract and the actual implementation |
| Migrated callers | Focused unit tests, replay/checkpoint/worker-scale coverage and EM fast guard |
| Production numerical behavior | Approved smoke/medium/long gates for the affected paths, including autonomous refinement and final all-data scoring |
| K1 and exactly K4 quality | Matched 100k/256 float32 control/candidate runs, existing FSC gates, per-class K4 Hungarian matching and all required ledgers |
| Real-data confirmation | A characterized real-particle refinement with matched inputs, metadata, initialization and masks |
| Refinement speed | Paired control/candidate repeats on identical GPU models, with cold compilation, process wall time, stage times and memory reported separately |
| Durable instructions | Principles, source extract, exact source fingerprints, commands, outcomes and remaining limitations |

Freeze the control at the pre-refactor integration commit and identify the
candidate by HEAD, binary diff hash and untracked source hashes. Keep both
snapshots immutable. Verify pinned imports and native source/binary hashes.
Compare quality by the approved FSC gates and measured GPU noise bands. Never
use bitwise floating-point equality or regenerate a baseline to accept the
candidate.

For timing, run both arms in one cryoem job with two identical GPUs, one arm per
GPU, symmetric resources and matched data/cache treatment. Retain repeats and
swap devices between rounds where practical. Use the established 10% regression
warning policy, and investigate a persistent slowdown even when quality passes.
Small CPU tests and stored historical timings cannot establish production speed.

## Final all-data handoff and remaining design work

`finalization.py` now owns admission policy and the post-replay accuracy, sampling,
expectation, joining, prior, final-resolution and reconstruction sequence. The
numbered controller keeps convergence/admission and replay writes at their existing
boundary, so replay releases the old noise owner before entering the new frame.
Complete caller, producer/lifetime inventory and final implementation are in the
[existing review](final_local_sampling_patch_review.md#5-final-all-data-orchestration-boundary).

Rotation matrices, working Euler rows, order and point group now have one producer
and lifecycle in `sampling.RotationGrid`. Startup, numbered rebuild and final
sampling consume that owner directly. The complete producer/callers are in the
[existing review](final_local_sampling_patch_review.md#rotation-grid-ownership-and-remaining-controller-work).
This removes one source of interface coupling without completing the controllers.

The 38 explicit inputs and 762-line final controller still expose unresolved
sampling/class model and output-format coupling. Review their actual producers,
updates and consumers before shortening the interface; do not create a locals bag.
The operation isolates a scientifically distinct phase, but it is not a finished
automation template. The command and numbered controllers are still unfinished.
Latest CPU source/caller checks pass; GPU/production/performance gates remain open.

## Class3D prior aggregation: lifetime audit before extraction

The numbered class loop's numerical inputs and four aggregate outputs have real
owners already, but its current scratch lifetime needs explicit treatment:

| Live local | Producer and last numerical use | Current retention and extraction risk |
| --- | --- | --- |
| `mean_signal_variance_per_class` | Each class prior's full-grid variance; read by `jnp.stack` | List and stacked class volume coexist through the rest of the iteration, until the next class-loop assignment. Returning only the stack releases large buffers earlier. |
| `mean_signal_variance_shells_per_class` | Each class prior's shells; read by `jnp.stack` | Radial arrays remain alongside the stacked shell array. |
| `data_vs_prior_per_class` | Each class prior's curve; host-converted and stacked | Device/host curve references remain alongside the scheduling array. |
| `tau2_update_details_per_class` | Each class's detail mapping; read by the detail stacker | Small radial arrays remain after stacking. |
| Last `class_prior` | Last class calculation, capture and list appends | Keeps that class's variance and weight summary alive after the loop. |

Replay admission precedes CTF2 preparation and the class loop. First-iteration
references versus reusable projector power remain a caller decision. The capture
and per-class logs precede stacking; history/scheduling writes follow the stacked
curve before detail aggregation, then tau2/model writes and reconstruction. CC
tapering happens after reconstruction and updates scheduling again. Preserve this
sequence rather than moving all writes to the end of an idealized stage.

Do not collect every `ClassPriorEstimate` just to shorten the loop: that retains
unconsumed weight statistics for all classes. Do not return unused full-grid parts
in a result merely to imitate accidental retention. Establish the required large
buffer lifetime deliberately, with matched-GPU memory/timing evidence, as a
separate change from moving the arithmetic. Then extract the substantive operation
with its existing settings and small computed outputs, leaving model/history
writes visible. The current access restriction prevents that GPU lifetime check;
this audit records the concrete issue rather than concealing it in a context.

Startup preparation meanwhile has one cohesive owner for ordering, image adaptation,
noise estimation and expansion, with explicit source selection in the command.
Its actual caller/implementation is in the existing review. Broader initialization
and finalization remain separate required work; these CPU checks do not establish
production speed or quality.

Normalization preparation and reporting now share their existing numerical owner.
The caller retains availability/mode decisions, physical group IDs, visible array
installation, follower dispatch and history/convergence ordering. No temporary
context or new output type was needed. Review the [complete operation](final_local_sampling_patch_review.md#normalization-and-scale-update-ownership)
before extending it. The broader parameter stage and its diagnostic publication
still need ownership work; this bounded update does not finish the controller.

Particle dataset preparation now has a complete operation boundary in
`particle_loading.py`. Reading policy, format admission, dataset construction and
mask configuration stay together, with the existing RECOVAR loaders underneath.
The five-field result is computed dataset/format/preprocessing metadata, not an
input bag assembled to shorten a signature. Shared optimiser discovery stays
with CLI policy, and seed/runtime lookups retain their timing. Review construction,
actual controller consumption and array/scratch retention together in the current
example. This boundary leaves half-set assignment, replay/reference setup,
startup-noise source selection and model/state installation as the next command
responsibilities to establish. No broad extraction is authorized by line counts;
the remaining scientific and performance gates still require executed GPU work.

Particle row preparation now belongs to the existing particle-table owner.
The computed result pairs input rows with the accuracy interpretation required by
downstream consumers, rather than hiding inputs behind a temporary context.
Fresh/replay selection, all-data mode, source priorities, diagnostic admission and
state installation stay visible. Review construction, index frames, private
scratch release and all consumers in the current example. This finishes one
identity responsibility; reference/replay/initial state preparation and the
large scientific controllers still need real operation boundaries. Source
organization alone does not certify GPU memory, timing or scientific quality.

Startup state now has one complete operation with the existing planning owner.
Its source precedence, schema admission and scalar initialization return the
existing state, while the controller retains grid construction and later model
and particle restoration. Review that caller and operation together. Next trace
iteration planning or parameter/convergence transitions as coherent responsibilities;
reference/prior aggregation requires an explicit large-buffer lifetime decision.
Do not mistake the remaining 3,286-line controller for the completed template.
Numerical and runtime qualification of latest source remains outstanding.

## Proposed packages after acceptance

| Package | Owners and boundaries to change | Reviewable outcome | Required checks |
| --- | --- | --- | --- |
| 1. Model-half lifecycle | `iteration_loop.py`, `noise_updates.py`, `mean_helpers.py`, replay/state-swap and `iteration_snapshot.py` | Reference maps/tau2 now have persistent ownership, completing the bounded direction-prior and noise lifecycle migration; array/order pairs remain together through restart, replay, reconstruction and updates. Final scoring reads those owners directly. | Noise/reconstruction/prior contracts, checkpoint/replay tests, fast guard, smoke; medium when numerical semantics change |
| 2. Optics preparation | `optics_shapes.py`, scoring input construction and shape adapters | Preparation returns typed optics operands directly; no intermediate dictionary followed by `.get` unpacking into another container. Shape transformations retain numerical meaning and subset identity. | Single/multiple shapes, pixel-unit conversion, noise remapping, scale routing, first-iteration and local/dense tests; smoke |
| 3. Finalization orchestration | Controller final expectation, accumulation, reconstruction and reporting | A finalization entry point with explicit meaningful inputs and results; shared setup precedes per-half work, diagnostics remain visible and writes are applied explicitly. | Final-pass admission, merged weighted sums, local/global, K1/K4 and tomography paths, saved formats, fast guard and smoke |
| 4. Precision and execution configuration | `scoring_policy.py`, dense/local engine adapters and diagnostic selectors | Effective settings resolved once at their proper scope; retire arbitrary forwarding dictionaries and duplicate defaults while retaining existing diagnostic capabilities and JIT specialization. | Production float32 and diagnostic precision contracts, engine routing, JIT/static inputs and native ABI checks; smoke and affected scientific tier |
| 5. Numbered iteration stages | Controller sampling, expectation, reconstruction and parameter updates | Named functions own substantial scientific steps; the main algorithm and state updates are visible without forwarding layers or objects holding every local variable. | Sampling/RNG progression, complete trajectory state, K1/K4 convergence and finalization, applicable medium/long gates |

Package 5 now includes batch preparation, complete results, numbered half
expectation and split-half FSC/tau2 estimation. The latter owns current-FSC and
independent half-weight priors; the controller retains joining, K1/Class3D policy,
state writes, reconstruction and post-reconstruction taper/host parking. Raw FSC
reporting remains distinct from corrected-FSC scheduling. Review the remaining phase configuration/publication boundary,
then extract the next substantive reconstruction/parameter operation. Finalization
and command initialization remain separate required packages. Keep the loop's
preceding-iteration convergence boundary and finalization admission visible.

Package 1 must migrate producers and consumers, rather than introducing a model
snapshot only at final scoring. Keep particle state distinct from changing model
buffers and pass-specific prepared projectors. Numerical routines may consume
temporary pairs of references where their mathematical API requires them;
the controller must have an explicit persistent owner and visible updates.

The bounded optics package has replaced fixed class translation override
keys with named results and moved preparation to its owner. Package 2's broader
follow-up should assess the older `class_kwargs` remapping interface and existing
batch override dictionaries against their schemas; preserve genuinely mode-dependent data and
reject unknown arguments instead of silently filtering them.

Keep structural cleanup separate from an algorithm, precision or performance
change. Record an intentional behavior decision explicitly and qualify it with
its own control. Revisit NumPy versus JAX only as a separate measured experiment.

## Instructions for each agent work package

1. Read the current handoff, scoped contracts and principles; inspect current
   source and evidence instead of assuming an earlier snippet remains current.
2. Identify one coherent lifetime and its producers, updates, consumers and
   serialized boundaries. Record which files the agent exclusively owns.
3. Trace each argument to its producer. State its meaning, valid absence,
   units/precision and the consumer which needs it. Remove copied or derivable
   fields only when their numerical contracts really agree.
   Check branch availability and argument evaluation timing: a new call can read
   a local or launch an array view that the original branch never needed.
4. Make the construction and consumption readable together. Remove superseded
   private interfaces and migrate maintained tests, scripts and imports in the
   same package. Keep independent numerical references independent.
   Account for new wrappers and dependencies as well as removed controller
   code. State the typical safe change and the surrounding details its maintainer
   can leave unopened. Preserve explicit state updates and existing decision timing.
5. Collect affected tests through indirect imports as well as direct callers.
   Run focused checks, then freeze a candidate for its required scientific tier.
   Preserve the control and every queued/running candidate.
6. Report executed, quality-accepted and performance-qualified outcomes
   separately. Include failures, skips, job IDs, source/native fingerprints and
   reproduction commands. Missing measurements remain open requirements.
7. Land each qualified package on main, then refresh the handoff and delete its
   temporary branch under the repository delivery contract. A blocked publication
   or validation gate remains explicit; it is not a reason to widen tolerances.

Delegation requires authorization and the EM subagent contract. A package handoff
contains one current conclusion, accepted design decisions, file ownership,
immutable evidence links, unresolved failures and the next executable action.
Do not dispatch a repository-wide mechanical migration before this example has
passed review and production qualification.

Particle pose interpretation now has a substantive operation alongside its
persistent half-set owner. The [current producer and calling flow](final_local_sampling_patch_review.md#particle-pose-interpretation-and-visible-state-update)
keep particle identity, supplied Euler precision and relative/absolute shifts
together. State installation, history, diagnostics and convergence timing remain
visible. This does not complete initialization, reconstruction or the parameter
stage; large Class3D buffer lifetime remains a separate qualification issue.

Follower-topology preparation now belongs to the existing worker-scale owner.
The [actual command flow and operation](final_local_sampling_patch_review.md#command-follower-topology-preparation)
retain strict replay admission and oracle verification before preparing runtime
ownership. The command remains 2,430 lines; CLI resolution, particle initialization
and source/restart adaptation still require coherent boundaries. This bounded
preparation does not complete the command or qualify production performance.

Particle-table source and group-layout ownership now reside with the ordering and
numbering operations they use. The [actual producer and command consumers](final_local_sampling_patch_review.md#particle-table-source-and-group-layout-ownership)
show source provenance, required layouts, preserved cardinalities and runtime
adaptation together. This removes a command dependency for metadata changes;
the 2,422-line command and 3,389-line numerical controller remain unfinished.
Continue substantial particle initialization and numbered-stage boundaries before
treating the complete calling flow as an automation template.

## Command configuration ownership

Flag registration, job defaults and supported-mode validation now share a command
owner. Import/cache setup and validation timing stay visible in execution.
The existing namespace remains the input contract; no large context or wrapper
was introduced. Review its caller and the complete owner in the existing review.
This isolates configuration changes but leaves the 2,423-line command controller
and 3,389-line numerical controller unfinished. Continue substantial scientific
operations after tracing their state and array lifetimes. The cached-main update
to `bc31839` changes native projector code and requires a fresh source-matched
native build and GPU qualification; earlier GPU evidence does not qualify it.

## Post-reconstruction resolution observations

The existing resolution owner now selects the authoritative prior curve and
computes the observed/first-iteration scheduling shells as one operation. Its
result is reused by diagnostics, history, future scheduling and convergence.
State updates and convergence timing remain in the controller. Review the full
post-reconstruction flow in the existing review before extending the pattern.
The numerical controller still spans 3,330 lines; the command spans 2,423.
Next trace a substantial initialization or parameter-update responsibility and
its actual buffer/model ownership. The new native set is source-matched and
built for A100/H100; GPU smoke and all production quality/performance gates
remain open. Do not treat CPU coverage or compilation as GPU qualification.

The pose-transition package gives snapshots, engine interpretation and aligned
convergence inputs one owner, with publication/export still between the two
computations. Review the [actual combined flow](final_local_sampling_patch_review.md#particle-pose-transition)
before extending it. This is a prerequisite for the remaining parameter-update
operation, not a completed numerical controller. Trace noise/normalization
admission and follower installation next, then finish the larger expectation and
initialization interfaces. Broad extraction and production/performance acceptance
remain open. Historical review excerpts now link to their immutable predecessor.

The pose-transition receipt passes 586 CPU cases, 108 fast-guard cases, four
import/structure contracts and 384 direct comparisons; two GPU-only cases remain
unexecuted. The prepared smoke script uses frozen source and source-matched
A100/H100 natives, with one H100, eight CPUs, 128 GB and a 25-minute limit based
on the prior 773-second H100 smoke plus compilation/CPU margin. It has no job ID.
Do not rerun native builds or submit duplicates. Inspect existing jobs when
scheduler access returns, then submit the prepared script and record the actual
GPU model and tier receipt. Smoke is followed by the existing production K1,
exactly K4, real-data and repeated timing/memory requirements; these gates remain
unfulfilled and prior-source receipts do not qualify the current candidate.

One submission of the prepared smoke script timed out after repeated Slurm
stream-socket denials. No job ID was returned. Preserve the script and its
`SUBMISSION_ATTEMPT.json`; inspect the scheduler for this unique source/job when
access returns before any retry. An observation timeout is not a terminal job
receipt. Existing candidates and queued/running jobs remain untouched.

### Numbered convergence policy boundary

`refinement/convergence.py` now owns accuracy admission and replay/state-update
precedence, returning a named state/accuracy result to the controller. Its pose,
geometry and options operands already have producers; no temporary input context
was introduced. Review the [actual top boundary, update and consumers](final_local_sampling_patch_review.md#iteration-convergence-policy)
together. The controller still selects native mode, performs preceding-iteration
convergence, installs the new state and records offset/history/checkpoint updates.
GPU/production/performance qualification remains open. Continue toward substantial
reconstruction and checkpoint ownership, accounting for retained class volumes
and snapshot arrays before moving those blocks; the current controllers are not
the template's completed form.

### Checkpoint owner and remaining controller context

Array/schema capture now lives with `IterationSnapshot` and its run-level
capture settings. Scheduling and growth policy remain in orchestration. The
original header assignment is a required lifetime boundary: do not collapse it
into a complete capture call that retains two sets of host maps during copying.
Review the [complete caller and owner](final_local_sampling_patch_review.md#checkpoint-array-capture)
and the [actual responsibility map](final_local_sampling_patch_review.md#controller-scope-and-remaining-coupling).
Next address substantial reference/replay or Class3D reconstruction aggregation
after resolving their retained buffers. The command and numbered controllers
remain unfinished, and widespread extraction awaits human flow review and
production scientific/performance qualification.

### Iteration-zero source replay

The model reader now returns ordered file/table pairs with explicit source
identity, reused by startup noise, priors and optimiser controls. This replaces
controller-local parsed dictionaries and redundant reference fields, while
state installation/reporting stays visible at its existing boundary. Review the
[complete calling flow and source owner](final_local_sampling_patch_review.md#iteration-zero-model-replay).
Source and computed-result types have concrete consumers; they do not collect
all startup locals. RECOVAR formulas and MPI broadcast keep their shared owners.

Next inspect remaining command reference setup and configuration construction,
or numbered sampling preparation, as complete responsibilities. Large class
reference/variance lists still need a separately measured lifetime decision
before extraction. Preserve preceding-iteration convergence and final admission.
Human review and production scientific/performance qualification remain open;
do not automate widespread extraction from these CPU results alone.


## Next command ownership boundaries

Startup sampling configuration is now produced once and reused through options,
run files and reporting. Archive field adaptation is with the file owner. Review
their complete construction and consumers in the existing artifact; the formatter
still has a wide explicit interface.

Next, trace pose-source, projector-capture and perturbation-restart provenance at
their actual validation/loading boundaries. Natural source records must keep
identity/path/hash relationships and be reused by scoring configuration and the
profile/archive/ledger formats. Do not assemble a record from publication locals.
Then complete substantial controller operations after the remaining reference and
Class3D buffer-lifetime audits. Preserve preceding-iteration convergence timing,
explicit model writes and finalization. Broader automation follows human review
and production float32 quality/performance qualification.

Startup pose selection now has a complete operation in `relion/input_poses.py`,
with producer-owned provenance reused by profile and archive output. The full
seed/correction payload keeps existing aliases and source metadata. Next trace
projector capture and perturbation-restart provenance at their validated producers
and existing replay/report consumers. Then complete the larger controller stages;
the pose operation's 18 inputs and archive formatter's 28 remain review points.

## Current numbered-controller boundary review

Paired-half execution remains with expectation; projector transformation, disk
cache/dumps and accuracy-projector reuse now have the existing projector owner.
The accuracy producer supplies a named identity/window binding consumed by both
accuracy and scoring. The controller keeps replay/mode admission, empty halves,
ordered persistent-half iteration and explicit projector installation. The old
list reset and last-projector assignment retain their release timing. See the
[complete producer, caller and implementation](final_local_sampling_patch_review.md#projector-preparation-and-accuracy-reuse).

This package passed 79 affected CPU tests, 108 guards and 96 actual old/new stage
comparisons. It does not complete the 3,184-line numerical controller. Review this
whole calling flow with the user before automating broader extraction. Then test
the accepted principles on another responsibility, resolving the Class3D retained
buffers or sampling RNG/lifecycle questions rather than collecting all locals in
a new context. Convergence timing and explicit model/history writes remain visible.

The [current receipt](final_search_patch_status.md) identifies source at cached
main `7412fd8` and the matching frozen native libraries. GPU smoke, production
float32 scientific gates and repeated matched-GPU timing/RSS remain unexecuted
because driver/scheduler access is unavailable. Preserve the uncertain prior
submission until its remote state can be checked.
