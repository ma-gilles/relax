# Refactor principles agreed with the user

Recorded September 30, 2026; clarified October 1, 2026. These are the user's accepted working preferences
for the readability refactor. Refine them through review of a representative
implementation before applying them broadly. Scientific gates remain mandatory.

The optics ownership boundary received provisional user acceptance on October 1.
The user found it hard to judge out of context. Include the surrounding controller,
actual mode decisions and visible updates in subsequent reviews. Large existing
controllers remain unfinished; acceptance of one owner does not certify their flow.

## Objective and review process

Optimize for how much code a developer or agent must understand to make a safe
change. File length, function count and import count are observations, not the
design objective. Numerical behavior and GPU performance remain essential.

Make the algorithm and its data flow understandable while preserving numerical
behavior. First improve one coherent section, review and iterate with the user,
then test the principles on a different section before broader automation.
Review the main calling code first, then the implementations underneath it.
Ask concrete questions with contrasting code examples, one at a time when a
material preference remains unresolved. Do not repeat already accepted choices.
Delegation still requires the authorization described in the root contract.

User clarification on October 2: finish the representative code and review its
complete calling flow before freezing the candidate for expensive production
quality and performance runs. Use focused checks during implementation. Preserve
existing candidates and jobs; their results remain evidence for their own source.
The final scientific gates are deferred until that review, not waived.

User clarification on October 2: release unused temporary arrays after their
last use and finish the complete refactor and qualification. Exact accidental
Python-frame retention is not a design constraint. Preserve scientific input/output
ownership, asynchronous execution requirements, numerical operations and deliberate
metadata precision. Record and qualify memory/performance effects on the final
frozen float32 candidate; do not introduce unused retention fields.

Keep work synced with integration main. Preserve the user's uncommitted comments
and other work when updating; inspect incoming overlap and follow the existing
validation rules. Never update a source snapshot being used by a running job.

## Interfaces and ownership

- Introduce classes only for natural abstractions with demonstrated reuse.
  Grouping arguments at one call site does not justify a class. Review an
  actual cohesive diff, including construction and consumers, rather than
  asking the user to approve isolated idealized snippets.
- Question parameters and conversions against actual callers. For coarse
  sizing the inspected production paths derive angles from HEALPix orders;
  migrate toward accepting the explicit sizing order and converting inside
  the owner, preserving the pre-update versus updated-order distinction.
- Establish integer and other configuration types at parsing/construction,
  then trust those contracts inside computation. Avoid repeated coercions,
  type checks, and tests of already-established invariants. Preserve necessary
  numerical dtype conversions and validate external input boundaries.
- Group related inputs into meaningful concepts when many values travel
  together. Pass those groups explicitly to computational functions rather
  than hiding all dependencies in a refinement object with methods accessing
  everything through `self`.
- Trace each input's producer, lifetime, consumers and update sites before
  grouping it. A domain group needs shared invariants and lifecycle; assembling
  twenty controller locals into a temporary context does not establish ownership.
- Put a half's borrowed model operands on its existing half input and grid
  metadata on its resolved sampling result. Trace reference-grid versus
  image-shape units before adapting either. If a required run-options owner
  already supplies an invariant, remove duplicate field overrides. Preserve
  guarded metadata reads for routes that do not consume those operands.
- A reused phase result can bind canonical grids, resolved sampling and policy
  when both halves consume them at the same boundary. Its producer should own
  their actual adaptation, not collect unrelated locals. Keep half-specific
  operands, output publication and scientifically distinct tomography sampling
  explicit. Release borrowed phase roots at the original iteration boundary.
  Preserve the distinction between canonical pose rows and device coarse
  geometry, and between model support and particle-window sentinels.
- Give metadata consumers the existing geometry owner when they need shape
  and physical pixel size, rather than a generic dataset. Preserve the original
  scalar separately where its type controls host arithmetic promotion; replacing
  it everywhere with an equal normalized float is a numerical change. Keep fixed
  input geometry independent of evolving Fourier windows and model support.
- A scientific phase should not consume values it only copies into an output
  mapping. Publish setup/numbered metadata at its existing controller owner after
  the phase returns, preserving borrowed arrays and the saved schema. Pair a
  captured state with source identity at its actual producer; let archive/profile/
  ledger consumers read that record instead of maintaining parallel source facts.
  Preserve CLI admission, optional metadata reads and exceptional error routing.
- Source-format admission belongs with the source identity/ordering owner. Return
  the original parsed table when particle consumers need it, plus admitted optics
  geometry; keep source selection and particle-frame policy in orchestration.
- Select the authoritative group source and compute its half-axis layout in one
  operation. Dispatch verification and follower provenance must consume that same
  admitted source. Validate required columns during selection, then trust that
  internal contract while mapping IDs. Keep source choice, physical group axes and
  image row order distinct; bind source and layout without copying their storage.
- Reused external output field names justify small schema formatters in the result
  owner. Keep model selection, scientific computation, merge precedence and mode
  overrides visible at callers. Final diagnostic formatting owns its sentinels,
  serialization casts and recording drains at the original return boundary.
- Pass an individual field when that is all the function needs. Grouping is
  not a reason to give every function the entire model or sampling state.
- Return updates and apply state assignments visibly at the call site, even
  when several fields change. Avoid `result.apply(model, state)` hiding writes.
- Preserve half, class, optics and particle identity through selection and
  updates. Review array metadata, aliases, retained references, donation and
  buffer lifetime alongside the interface.
- Carry known metadata, including sampling order and symmetry, with its data;
  require it at interfaces instead of reconstructing it from array sizes.
- Image backend admission and mask installation belong with particle input
  adaptation. Keep the controller's setup order/timing visible. Preserve model
  versus shape-class pixel units and verify persistent dataset/backend ownership
  before removing borrowed local aliases; scope changes alone need not release
  numerical storage. CLI mask-source precedence remains its own admission rule.
- Give the algorithm a consistent particle interface. Adapt external inputs at
  the boundary without unnecessary array copies. This is not a decision to
  replace RECOVAR's numerical core or duplicate its implementation.
- Use named result types for fixed outputs; dictionaries suit genuinely variable
  keys. Prefer small data classes for coherent data, not an object holding every
  local variable solely to shorten a signature.
- Use distinct configuration types when modes require different fields. Avoid
  flags plus optional fields that permit invalid combinations.
- Use explicit parameters and reject unexpected arguments. Avoid silently
  filtering or forwarding a shared arbitrary `**kwargs` dictionary.
- Keep an obvious primary input positional; name settings, booleans, sizes,
  and otherwise ambiguous arguments.
- Migrate maintained internal callers and remove obsolete interfaces rather
  than retaining forwarding wrappers. Preserve required public APIs and saved
  formats under the repository contract.

For example, if reconstruction needs only rotations, pass `sampling.rotations`.
If expectation needs the grid, its metadata, and priors, pass a coherent sampling
object. Return a noise estimate and assign `model.noise = estimate.variance`
where the iteration is orchestrated.

## Algorithm structure

- Keep a coherent operation, its private helpers and its small result types
  together. Extract a module when it establishes meaningful ownership or
  isolates implementation details. Avoid one-function-per-file fragmentation,
  generic helper collections and arbitrary splits of a long function.
- Extract coherent algorithm steps into named functions even with one caller.
  Sampling preparation, expectation/accumulation, reconstruction, noise
  estimation and convergence policy are useful boundaries. Each layer should
  own a decision, computation or necessary adaptation; avoid forwarding chains.
- Keep the main sequence visible: sampling, expectation/accumulation,
  reconstruction, important mode decisions, state updates and the transition
  to finalization, in the existing execution order. Preserve the timing of
  convergence decisions, including decisions based on the preceding iteration.
  Do not rearrange operations to fit an idealized loop.
- Give finalization its own function with explicit inputs and outputs when its
  boundary is established. Do not mix the final pass into numbered iterations.
- Reconstruction diagnostics own target selection, capture serialization and
  finite auditing before cross-half mixing. Keep the audit call and numerical
  combine/join/snapshot decisions visible in orchestration. Return only the
  scalar target match needed by later capture; do not retain numerical scratch
  or collect model/history state merely to shorten this diagnostic interface.
- Keep diagnostic calls visible in orchestration; put recording, conversion,
  and enable/disable details in the diagnostic implementation. Sealed restart
  slot admission, projector attachment and scoring-noise adaptation belong with
  the existing diagnostic CLI owner and its private helpers. Keep mode/source
  selection, returned-array installation and provenance publication visible.
  Move substantial existing operations together rather than introducing another
  wrapper or leaving obsolete command re-exports.
- Organize modules by responsibility, keeping related functions and types
  together. Avoid general `helpers.py`, `types.py`, or `constants.py` dumping
  grounds. Inspect existing owners before adding modules.
- Put reusable layout arithmetic beside its existing table/chunk definitions.
  Global and local engines should consume that owner directly. Preserve host
  representation, capacity checks and device placement when sharing identical
  production bodies; keep independent numerical references separate. Compare
  all caller operations, including indirect consumers and source guards.
- Branch around operations that differ and share the common sequence when
  semantics, ordering, and resource lifetime are actually the same.
- Encode mandatory behavior in the function's contract. Do not expose toggles
  for unsupported alternatives; investigate existing diagnostic uses before
  removing an option.

## Configuration and explanation

- User clarification: remove silent substitutions for invalid physical metadata,
  such as `voxel_size if voxel_size > 0 else 1.0`. Validate finite, positive
  physical pixel sizes at the input boundary and pass validated values directly.
  If a workflow intentionally uses pixel units, represent that explicitly.
  This authorizes rejecting invalid metadata instead of preserving its fallback;
  preserve numerical behavior for valid inputs and test affected entry paths.
- Resolve environment and other configuration at startup, preserving precedence
  and defaults, then pass effective settings explicitly.
- Trace settings to their actual consumer when reviewing behavior or measurement.
  A recorded environment variable alone does not establish effective configuration.
  The production comparison now follows the maintained particle-reading CLI;
  ignored historical preread selectors cannot certify its reading mode.
- Validate configuration at construction. Keep checks on computed results
  where those results become available. Preserve failures and supported inputs
  when changing validation boundaries.
- Define each default once with the configuration that owns it.
- Bind invariant reconstruction settings once from the resolved run options.
  Prior estimation, regularized and unregularized reconstruction, postprocessing
  and captures consume that owner. Do not also pass a field separately when
  its settings object is already required. Keep changing windows, accumulator
  layouts, prior operands and first-CC phase decisions explicit in their callers.
  Trace scalar types before replacing an old input with normalized geometry;
  equal values alone do not establish identical arithmetic.
- Keep numerical constants with their domain and name their units. Image-mask
  pixels and reference-filter Fourier shells are different quantities. Refinement
  padding defaults do not replace VDAM's deliberate padding choices.
- Shared numerical operations belong below their workflow callers. Refinement
  and VDAM use the same reference-initialization and adaptive-grid owners;
  neither workflow imports the other's controller to obtain those operations.
- Name operations by what they compute. Reserve `relion` for interoperability
  or a meaningful algorithm distinction, rather than ordinary step names.
- Use descriptive intermediate names; keep familiar, unambiguous abbreviations
  such as CTF and FFT.
- Document shapes, units, layouts, and relevant coordinate conventions at
  numerical function boundaries.
- Comments explain non-obvious reasons and constraints, not a paraphrase of the
  next statement. Keep mathematical and source references where useful.
- User review clarification: keep docstrings concise. Document shapes and units
  where they resolve real ambiguity, rather than repeating established types
  and conventions on every helper. Refactoring instructions such as "preserve
  existing dtypes" belong in development guidance, not function documentation.
  State subtle runtime semantics briefly, close to their owner.

## Deletion policy

The user prefers aggressive cleanup and accepts repair/rework. Actively remove
obsolete branches, wrappers, and settings when evidence supports deletion;
do not require exhaustive proof or approval for every removal. Check maintained
callers, entry points, registration, tests, notebooks, and imported/serialized
names as applicable. Record what was removed and why in the change description.

Trace private families as a group: mutually referring descriptors, validators
and fingerprints can be disconnected even when every name has a reference.
Check outside constructors/consumers, registration and serialized names before
retirement. Keep the surviving live operations unchanged and preserve independent
numerical references; test-only numerical references are not dead production code.

Passing tests alone does not establish that a mode is unused. Investigate
uncertain numerical significance before deletion. Retiring reachable supported
scientific functionality is a behavior decision to discuss with the user.
Keep deletions cohesive and recoverable and compare against a frozen control.
Tolerance for interface repair does not authorize silent scientific changes.

Collect tests through indirect consumers, diagnostic owners and source guards,
not just names of edited functions. Migrate their constructors and assertions
to the real interfaces while preserving numerical checks. Run the repository's
import-lint check before freezing a candidate. Refresh current engineering
structure metrics after extraction; preserve their historical baseline and
every scientific baseline.

## Numerical and performance boundaries

The numbered sampling package separates host policy from grid execution without
moving either boundary. Capture the coarse-sizing order before replay/accuracy;
advance sampling after accuracy with preceding-iteration counters; then resolve
perturbation at its original position. Numbered and physical RELION iteration
indices differ after continuation. Replay and sealed perturbations must not
consume the native RNG.

Model and particle Fourier cutoffs are one small computed concept, rather than
unrelated controller locals. Keep their full-window interpretation explicit:
after optics remapping, a full model width can no longer use the engine's None
sentinel, which would inherit the particle cutoff. Carry computed sizes and
derive their engine representation together. Do not include reference maps or
other large operands in a host size result. Adaptive coarse sizing still uses
the incoming order and remains conditional on the selected search mode.

An unreachable branch can be removed from its producer's contract without a
numerical experiment: the deleted matrix-only perturbation fallback tested
whether the result of `np.asarray` was None. Keep real coordinate-conversion
fallbacks elsewhere until their producer contracts and consumers are traced.

Preserve current NumPy/JAX choices in the first structural refactor. Also
preserve casts, precision, operation/reduction order, random-state progression,
layouts, JIT boundaries, transfers, buffer lifetime, state updates, scientific
defaults, and serialized formats. Preserve deliberate higher-precision metadata;
production EM remains float32. Use existing gates, never bitwise float equality
or loosened tolerances, and qualify K1 and exactly K4 as the affected scope requires.

The noise-lifecycle example demonstrates that a cached mean can encode a
phase-specific precision contract. Trace its actual producer before replacing it
with a getter or recomputing it. Preserve partial diagnostic replacements and
existing array aliases alongside the production update.

The reference/prior example preserves K1 map release and tau2 host parking.
Trace view creation as well as transfers: passing pre-sliced arrays can launch
GPU work earlier or materialize unused views on diagnostic replay. A shared
operation may accept a stack and its class index to own that timing. Its callers
still own scientifically different replay, first-iteration and CTF policy.

The expectation batch planner groups immutable run requests and fixed grid
geometry. Available device memory is still queried when each pass is planned;
it is not a run setting that can be cached during construction. Its named result
contains sizes and callbacks, while the controller constructs local/dense engine
policies and keeps mode selection visible. The shared allocation formulas retain
their existing owner below refinement.

Extraction also changes when arguments are evaluated. Trace whether each value
exists in every supported branch before moving a guarded read into a call.
The batch-planning caller passes no coarse size outside adaptive scoring and no
shape-sizing operands outside adaptive multi-shape scoring. Preserve that
absence explicitly; do not initialize placeholder locals merely to make the
extraction callable. Review and exercise the actual caller, not just its new
function with a fully populated fixture.

Return a complete numerical result before extracting a controller operation.
The scoring example removes the split between a returned common payload and
hidden writes to pose/class collectors. Input operands no longer include the
controller's output collector. Class assignments, M-step/full-evidence mass,
rotation sums and noise form a reused result payload; the controller records it
at the existing update boundary. Shape adapters still release summaries their
existing merge does not consume before scoring the next shape. Grouping results
must not extend an unused buffer's lifetime across later GPU work.

Reuse an established domain type at its producer before extracting its
consumers. Numbered local sampling now produces `LocalSampling` directly, with
source Euler rows attached, and the half scorer and pose export read that same
result. Keep scientifically different producers together without forcing them
through flags: numbered windows can depend on the preceding sizing order;
final windows use the resolved parent order. Read optional optics rows only in
the branch that needs their sizing geometry. Empty-result construction belongs
with its layout-aware result owner, while orchestration still selects the
operation and visibly records its output.

Trace attribute reads as well as numerical calls at an extracted call site.
Empty full/K-class results do not read the dataset's volume geometry; pass no
geometry operand for those layouts. Exercise the actual call with unused
metadata unavailable, rather than supplying every field to the new function.

An expectation operation should return the pose frame needed to interpret its
assignments, including canonical Euler rows and the applied translation base.
Keep result publication, accumulator offloading and invocation of post-score
recording explicit in the controller. Retain the prior producer's named result instead
of splitting it into parallel rotation/class lists and rebuilding it later.
Release new phase containers at the existing memory boundary: borrowed references
can otherwise retain previous grid buffers during the next projector allocation.
Review lifetime as well as numerical call order when moving phase construction.

Half-expectation recording keeps profile conversion, support-count collection
and E-step captures with their existing expectation owner. The controller shows
scoring, metadata installation, offloading, payload publication and the recording
call. Support counts have per-half slots and separate recorded/convergence
aggregates because those consumers use different policies. Preserve arrival
order under overlap, absent versus supplied-empty arrays, and the point of
combination after preprocessing drains. Do not sort halves or combine counts
inside a worker merely to simplify the interface. Removing redundant aliases is
safe only when the same arrays remain retained by established payload owners.
Trace diagnostic metadata fallbacks too: identity rows and missing source rows
are distinct capture semantics even when two helpers look similar.

Final half preparation illustrates a small result type with a real purpose:
translation, direction and optics operands are conditioned on the same half and
scoring frame, and are consumed by both scoring routes and manifest export.
Keep those computed operands together without moving model updates or diagnostic
publication into the producer. Distinguish score-prior centers from sigma-offset
centers; their numerical frames differ. Preserve the final mode decision's
position before expected-accuracy updates, and admit captured prior rows only
for the grid whose identity those rows describe.

Result publication illustrates another ownership boundary: keep archive
schemas, compression, profile conversion and map export together, while the
command shows publication order and run-specific metadata. A small returned
report can hold computed values shared by multiple output formats; it must not
capture arbitrary controller locals. Retain serialized array references through
their existing reporting boundary. Trace the current API before implementing a
plan: the canonical refinement entry point already uses grouped options.

Diagnostic admission illustrates boundary ownership. Keep CLI flag registration,
source binding, failure adaptation and their private helpers together; preserve
validation timing in orchestration. A CLI adapter can consume its existing parsed
arguments without fabricating a controller-local context. Return loaded state and
its source bindings directly, preserving array references. Separate input/schema
validation from resolved-runtime validation when their inputs become available at
different points in startup.

Split-half prior estimation illustrates a scientific operation boundary: own
current-FSC and half-specific weight calculations together, while the controller
shows joining, the different Class3D prior policy, state replacement and later
taper/host parking. Keep raw reported FSC distinct from the corrected FSC used
for priors and growth. A returned result must not retain device arrays beyond
their existing replacement/parking boundary; release the temporary container.
Trace scalar types as well as values before substituting normalized geometry:
NumPy scalar promotion can change an existing radius calculation. Record when
private host intermediates now expire after their last use, and qualify memory
and timing alongside numerical outputs.

The Class3D capture example uses the existing prior result and reconstruction
settings at its actual per-class boundary. The diagnostic owner computes its
floor-shell statistics and preserves its file schema; orchestration retains the
gate and scientific class loop. Distinguish diagnostic floor statistics from the
round-shell statistics used to estimate priors. Report private radial scratch
expiry after serialization and qualify enabled-capture timing/memory. A format
change should not require reading the scientific controller.

A new Python function does not require another JIT region or GPU pass. Preserve
fusion and specialization; do not materialize large intermediates simply to make
source stages look independent. Shared RECOVAR mathematics keeps one
implementation in RECOVAR.

A separate future experiment may compare the existing NumPy/JAX mixture with
a substantial JAX path, then selected larger JIT regions. The user has not
approved a blanket NumPy-to-JAX conversion or a permanent dual-backend API.
Record suspicious conversions; do not endorse them merely because they are
preserved for this phase. Measure completed execution and pipeline throughput,
including synchronization and transfers, at matched precision.

## First design review

Judge the complete change: caller, implementation, types, construction and
indirect consumers, including new wrappers and dependencies. Counts remain
review signals subject to repository budgets. State which files and assumptions
a maintainer can leave unopened for a typical change; source length alone cannot
justify the boundary.

The current optics preparation example puts its result types and computation
with the existing optics owner. Named class-pixel operands replace temporary
dictionaries; dense/local adapters consume them after image remapping. Review
the cold-start prior policy, class identity and numerical operation order
together. Test callers of an obsolete forwarding route were moved to production
dispatch while keeping their independent expected-value assertions.

The [final-search proposal](final_search_refactor_proposal.md) is a concrete
application awaiting user review, not an accepted implementation or scientific
qualification. Maintain that distinction as the design evolves.

The pose-transition example shows why one scientific responsibility can need two
operations in one owner: snapshot/resolve occurs before explicit publication,
while convergence-stack preparation follows publication and particle export.
A small named result can carry computed data between these points without hiding
state writes. Review snapshot precision, canonical Euler identity, relative versus
absolute offsets and both half slots together. Grouping calculated results is
different from collecting all controller inputs. Inspect temporary host retention
as well as GPU buffers when several locals become one result.

Keep the current review readable. Link earlier immutable source excerpts with
their provenance instead of accumulating competing historical implementations in
the current example. Preserve established links and acceptance distinctions.

## Current review implementation

The [complete local-sampling review](final_local_sampling_patch_review.md)
supersedes earlier idealized snippets for this review. The implementation moves
precision selection into the existing policy and passes symmetry explicitly.
Numerically meaningful dtype choices remain explicit even when hidden policy
lookups are removed. Final local sampling reconstructs grids from explicit settings; the user requested
removing the array-count-based reuse shortcut. Qualify resulting numerical changes,
including replay behavior, before merging.

Diagnostic precision selection belongs centrally at scoring setup. Do not pass
individual debug precision booleans or per-pass precision bundles through the
refinement controller. Production scoring/projection remain float32; preserve
intentional higher-precision metadata and existing diagnostic capabilities.

Avoid initializing placeholders only to overwrite them in the selected branch.
Assign results where the deciding operation occurs. Avoid aliases that merely
unpack a policy into booleans, repeated casts of established configuration
values, and local copies of fields without a distinct meaning.

Trace a conversion to its producer before deleting it. Parsed STAR fields,
typed estimator results and normalized scalar helpers should be used directly.
Conversions that determine array precision, host arithmetic or serialization
still belong at the relevant boundary. Final-pass geometry consumers use the
explicit image geometry; adapting the external dataset happens at entry.
Inspect array shape without converting or downloading its contents. When selecting
one grid from several candidates, select first and convert only the chosen grid.
Pass symmetry explicitly, including C1, when the consumer accepts that label;
do not conditionally construct a keyword dictionary to omit a supported default.

The current implementation separates local-search settings (orders, angular
widths and symmetry) from prepared arrays. Keep those settings attached to the
sampling result, so consumers share their owner instead of unpacking copies.
Fixed input-image geometry is separate from changing Fourier windows and model
support. A geometry object describes one physical grid, not all optics groups
or an output volume. Translation arrays can remain explicit when another class
would only shorten an argument list. These are reviewable implementation choices,
not approval to migrate the whole codebase before reviewing the example.

Iterate over coherent halfset inputs rather than repeatedly indexing parallel
lists in scoring code. During migration, gather references at one explicit
boundary; preserve half order, lazy reads, array identity and reference lifetime.
Keep shared configuration outside the halfset. Do not claim end-to-end state
ownership while earlier iteration/replay paths still maintain parallel lists.

The particle-half example now uses persistent ownership from initialization
through restart/replay, updates and final scoring. A late snapshot assembled
from parallel lists is not the target design. Bound future migrations by a
coherent state lifetime: finish its producers and consumers, remove superseded
representations, and leave other scientific concepts with their current explicit
owners. The present slice covers particle state; it does not claim model/noise
ownership. Keep pass-specific references and projector buffers out of permanent
particle state. Shared local/dense input types should reference that state instead
of copying its individual fields into separate wrappers.

## Review each argument and owner before extending the template

For every argument in a proposed exemplar, identify its producer, meaning,
lifetime, consumer, units/precision where relevant, and valid absent states.
An argument justified only by the old signature needs investigation. Review the
construction and update sites together with the final call.

- A prepared array and the metadata required to interpret it travel together.
  `PreparedProjector` owns its projection slabs, Fourier cutoff and power spectrum;
  its producer returns that result directly. Cache/replay paths return the same
  type. Do not split it into parallel lists and reconstruct the pairing later.
- A half index identifies output slots and serialized half order. Persistent
  input relationships belong to the half's state, and prepared operands belong
  to the expectation that produced them. A `zip` at an existing array boundary
  does not establish persistent ownership of those arrays.
- Store optics membership on the particle half. A scoring input references that
  owner; a shape-specific view supplies its subset membership. It must not carry
  a second independently replaceable copy of the same membership.
- Shared local/dense optics have one `OpticsSpec`. Dense shape-specific batch
  overrides belong to batching. Keep distinct mode types where their required
  fields genuinely differ; do not merge them merely to reduce class count.
- Interpolation method is an execution setting. Sampling preparation needs grid
  geometry and numerical precision; it does not need an interpolation method
  merely because later scoring uses one.
- Resolve run-wide scoring dtype once at run setup. Retain explicit dtype on
  allocations, host/device transfers and numerical conversions. An unchanged
  mathematical value can still have a different dtype. Local diagnostic pass
  precision has its own scope; do not propagate it to global priors or outputs.
- Name numerical producers for their operation. Include a dtype in the name
  only when that dtype is a fixed contract. `relion_scoring_rotation_grid`
  supports diagnostic float64 and therefore must not be named `...float32`.
- Preparation functions return operands ready for their consumers. Pass the
  selected dtype once at that boundary and perform necessary array conversions
  inside it. Keep low-level host geometry distinct when other consumers need
  its unrounded coordinates; moving a cast across perturbation changes numbers.
- Resolve native/replayed sampling in its preparation step. Return the resolved
  settings and grids as a coherent result consumed by scoring and reporting.
  Keep source-selection paths and candidate lists local to resolution. Do not
  initialize controller placeholders and repeatedly overwrite them across
  branches. Store optional perturbation once; derive legacy zero values and
  applied flags at consumer/serialization boundaries.
- Construct shared final-pass sampling and policies once before processing the
  two halves. Keep per-half references, noise, prepared projectors and priors at
  their own lifetime. Do not give a shared policy fields that change per half.
- Resolve settings before constructing scoring operands. Dense sampling carries
  oversampling order and translation step, rather than retaining a mutable
  refinement state that consumers repeatedly inspect. Optics views transform
  the step directly in image-pixel units without cloning the schedule.
- Omit arguments equal to an established constructor default when they convey
  no additional scientific choice. Do not change a scientific default to make
  a call shorter.

The whole iteration controller is not yet the template. Particle halves,
reference maps/tau2, noise and direction priors now have persistent owners.
The current exemplar also covers prepared projectors, sampling and scoring
configuration. Initialization, phase orchestration, model updates and finalization
still need substantial operation boundaries.

The [extension plan](refactor_extension_plan.md) proposes the next bounded
packages and their required checks. Broad migration awaits review and production
qualification of this example.
Extend those concepts only after reviewing their producers and consumers and
passing the applicable numerical gates. The review extract and status state the
current implementation and evidence; an attractive call site is not acceptance.

## Execution-default cleanup lesson

Keep scientific constants with their domain owner and reusable precision policy
with its existing type. A shared dictionary mixing fixed physical arguments and
diagnostic switches obscures which values can change. State fixed engine
arguments at the adapter and resolve switches once at their established lifetime.
Use the same resolved policy for grids, priors, batching and engine inputs.
Preserve import-time versus call-time environment semantics during cleanup;
compare effective arguments at the consumer as well as the configuration producer.

The direction-prior example tests ownership at another responsibility. Keep a
prior's probability array and its sampling order in one payload throughout
initialization, replay, learning and scoring. Preserve missing-payload metadata
at saved-state boundaries. Return learned payloads and apply model updates
visibly; put prior computations with their existing orientation-prior owner,
rather than a reconstruction module. A consistent payload type reduces optional
object checks while preserving the actual absence of a learned distribution.
This implementation remains under review and scientific qualification.

### Image-size planning lesson

Resolution crossing and window-growth latches can consume different spectra.
Keep each signal's producer and history/restart precedence explicit; do not
substitute an absent-value fallback for a recorded invalid history row. A result
record may group computed outputs while its controller keeps important state
writes and ordering visible. Preserve different startup/Class3D/K1 policies rather
than adding flags to a universal planner. Trace scalar types at the original
producer before replacing them with a normalized geometry field; a redundant
looking conversion may change NumPy promotion. Document shortened scratch
lifetimes explicitly and qualify runtime effects separately from CPU agreement.

### Startup preparation lesson

Separate command source selection from a coherent data-preparation operation.
The estimator needs ordered image rows, optics metadata and mask settings, not
an argparse namespace or unrelated frozen/loaded-source flags. Keep native host
sigma2 separate from its N^4 frame conversion and scoring cast. A two-field named
result can make the actual caller's assignments explicit without retaining new
buffers or adding a forwarding layer. Move all related private helpers together
and migrate indirect consumers before retiring command re-exports.

For aggregate GPU results, distinguish small radial scratch from full-grid
per-class arrays. Stacking can leave both originals and the stack live. A function
boundary that drops unused originals changes memory lifetime even if arithmetic
is identical. Do not invent result fields or a collector for unconsumed buffers;
record and qualify the deliberate lifetime decision separately.

The final all-data example shows why replaceable state must be resolved before a
phase handoff: moving noise replay into a child frame would keep the caller's old
noise owner alive. Keep that replacement at its existing boundary, then pass the
resolved owner. Trace parent-frame retention as well as the child: angular state
copies share pose arrays, numbered class labels remain in their existing collector,
and class weights are small host vectors. A coherent phase can establish ownership
while still exposing an oversized interface. Record that limitation and finish its
producer ownership before treating it as an automation template.

## Rotation-grid lifecycle lesson

Construct related array/metadata ownership at its producer and retain it through
updates. `RotationGrid` keeps scoring matrices, working Euler rows, order and point
group together from startup through rebuild/final preparation. Consumers needing
only one field receive it directly. Do not widen working-row precision merely
because the same container carries metadata. A temporary container also retains
its arrays: release it before replacement where the former operation released
base buffers. Keep only required fields in later results. Review the actual
producer, adaptation, replacement and final consumer together.

The current command and numerical controllers remain too coupled to be the final
example. Responsibility boundaries must let a maintainer change one operation
without understanding thousands of lines of unrelated setup, replay or updates.
Keep the scientific sequence, preceding-iteration convergence and state writes
visible while establishing those boundaries.


## Normalization update lesson

Keep adaptation, numerical estimation and domain-specific reporting with their
existing operation owner. `relion_normalization.py` now adapts persistent particle
halves, audits sufficient statistics, computes the existing named result and
formats its report. The caller retains admission, first-CC/tomography/follower
policy and visible correction assignments. Logging stays after follower updates.
Reuse the existing numerical result rather than inventing a context type.
Preserve caller-owned group-ID lifetimes and physical versus statistic routing.
Correction measurements used by checkpoints and captures have a small reporting
result with the normalization owner. Distinguish selected rank-1 follower reports
from installed per-particle scales; those values and axes are not interchangeable.
Keep the native numerical result alive at its existing boundary. A reporting
container may replace parallel aliases when its arrays keep the same roots
across numerical work; it must not retain otherwise-unused scientific scratch.

Completed-iteration capture adaptation belongs with its diagnostic owner:
encoding casts, pose/particle lists and existing warning handling are one coherent
operation. The controller retains capture admission, timing-only selection and
checkpoint order. Pass model fields individually when only one field is needed.
Prove run-settings substitutions against their producers and retain raw scalar
types where normalization would change the existing contract.
A preparation function is useful when it owns this adaptation/audit; forwarding
the same arguments alone would not establish a boundary.


## Particle-pose interpretation lesson

A stage result should contain computed fields with real consumers. Selected
matrices, source Euler rows and relative/absolute pixel shifts form one particle
pose result used by state installation, diagnostics and convergence adaptation.
Keep installation and history copies explicit in the controller. Grouping these
arrays does not authorize forcing all fields to the scoring dtype, inventing a
new GPU pass or dropping supported matrix/grid paths. Review small host scratch
lifetime alongside the unchanged large-buffer owners when qualifying production.

## Command preparation lesson

Keep source discovery and oracle verification at the command preparation
boundary, before scientific execution. The existing command-options owner can
own source-dependent admission and CLI/optimiser precedence; the controller
retains the strict replay decision, frozen-aware source selection and visible
installation of resolved controls. Preserve verification order: directory
manifests before consumed-file discovery, then particle-order admission. Use
related result fields directly for execution/output; a schedule and its verified
roots form one concept, as do resolved controls and their provenance. Keep
format/formula implementations with their existing RELION owners.

Pass individual scalar fields when resolution needs only those fields. Use the
CLI namespace only where CLI source discovery/admission actually requires it;
do not make it a numerical context. Validate real portable manifests and saved
metadata alongside the actual caller. Schema-correct fixtures must use sorted,
unique manifest paths. Preserve refusal of unmanifested numbered/final sampling
and distinguish CLI active-cap overrides from saved argument sentinels. Do not
construct parallel result variables for every returned field.

## Particle-group identity lesson

Keep row ordering, source lookup and group layouts with the particle-table owner.
Pair selected data with the provenance used by downstream verification; retain
that pairing from the producer. A required layout result can remove impossible
absence branches without dropping real external-input validation or full model
cardinalities. Adapt legacy container formats at the runtime boundary. Test all
producer branches through their actual field consumers: successful tuple-unpacking
tests alone did not cover the new named fresh-start result.

### Command configuration and execution

Keep flag registration, defaults and supported-mode admission with the command
configuration owner. Preserve cache/import setup before parsing and retain checks
that depend on loaded metadata where that metadata becomes available. Moving a
parser can isolate a responsibility without shortening its controller; report
that distinction explicitly. Migrate tests to inspect option definitions and
runtime routing at their actual owners, preserving the full assertions.

### Resolution observations and later decisions

Keep related computed scientific signals together when several real consumers
use them. The resolution result separates the observed shell from the
first-iteration scheduling override, and carries the selected curve used by
history and future scheduling. Preserve the time of unit conversion, state
writes and preceding-iteration convergence checks. Account for even small host
array retention changes when moving a calculation into a function. Update
inherited source guards to cover every current controller owner without dropping
scientific or diagnostic assertions.

### Particle-loading example

One substantial command operation can own format admission, I/O policy and
backend setup without becoming a numerical controller. Keep its format metadata
with the dataset and inspect construction and consumption together. A named
computed result is justified by the relationship between particle identity,
image-grid rows, mask geometry and preprocessing precision; it is not a substitute
for establishing ownership of a large collection of inputs. Use the already
parsed command at the command-input boundary rather than inventing another
temporary options class. Numerical operations still take only the operands and
settings they require. Share source-discovery policy with the existing CLI owner
and preserve when consumers read it. Trace adapter and scratch cleanup ownership
before moving loader locals out of the controller. The full example and its
source-specific limitations are in the [review](final_local_sampling_patch_review.md#particle-input-preparation).

### Particle row-layout example

Keep source rows with the identity frame needed to interpret their consumers.
Build that result at its producer, carry it through replay/resume/output and
retire the old independently named representation. Do not conflate input rows,
RELION table IDs and half-local accuracy permutations merely because they share
an integer dtype. Small named results can express those relationships without
owning mutable model state. Keep mode/source selection and diagnostic admission
visible. A private helper is justified when it releases scratch before a derived
allocation; deleting that boundary to reduce function count would change lifetime.
The [complete example](final_local_sampling_patch_review.md#particle-row-layout)
shows construction, actual consumers and the remaining large-controller limits.

### Startup state ownership

Initialization can own complete source precedence and schema admission while
returning an existing domain state directly. Use established configuration and
physical geometry rather than inventing a new context or unpacking every setting
at the caller. Keep the phase position, timers, later array restoration and
scientific state updates explicit. Review scalar normalization at its original
consumer: here the resolution helpers already convert a positive pixel scalar
before arithmetic. Compare original statements and all resulting state fields,
including refusal paths. A successful host extraction does not qualify GPU
quality, timing or retained large-buffer lifetimes. See the
[complete example](final_local_sampling_patch_review.md#startup-sampling-state).

Convergence policy can own accuracy admission and replay precedence without
moving the decision to enter finalization. Return the new state explicitly and
leave the next permitted loop-boundary decision visible. Reuse existing pose,
geometry and configuration owners; an explicit interface is preferable to a
temporary collection of unrelated controller locals. Carry the computed accuracy
result to reporting without rebuilding it. Preserve fresh-run one-shot counter
resets separately from continuation/replay controls, including their update order.

Trace every assignment to an apparent configuration alias before replacing it
with a read from startup options. The controller disables the active replay
directory after its limit; convergence must consume that live value explicitly.
Frozen configuration and runtime state have different lifetimes even when they
start with the same value. Exercise the transition, not only replay enabled at
startup.

### Checkpoint capture and retained host arrays

An operation can retain an existing header/capture boundary when assignment
timing controls large-array lifetime. Returning a complete result in one call
would keep the preceding snapshot alive during new map copying. Preserve the
header replacement before array capture, then return the established complete
snapshot. Do not invent an unused retention payload to reproduce this effect.
Keep copy order, aliases, precision tags and background-writer retention with
the schema owner, and exercise the actual caller's release point. The controller
still owns checkpoint admission and growth-state policy.

Show unfinished controllers in context: identify their actual responsibilities,
remaining embedded implementation and state/lifetime constraints. Smaller
counts or isolated clean snippets do not establish an acceptable template.

### Replay source and staged result ownership

Keep parsed tables paired with their source paths at the reader, then reuse that
source across noise, prior and optimiser operations. Derived reference access
can eliminate duplicate fields without reconstructing source identity from array
sizes. A computed result used for both installation and diagnostics is a useful
small type; release its borrowed full-grid arrays before a later replacement.
Retain explicit controller assignments when they establish release/reporting
boundaries. Each stage must own actual source policy, computation or adaptation.

Trace the preceding producer before replacing a flag with optional data. In the
admitted iteration-zero branch, the prior noise selector produces an explicit
NPZ array or `None`; this makes the direct operand equivalent to the old NPZ
gate. Prove that producer relationship and exercise the actual caller.

Verify independent expected-value conventions against the pinned formulation
before interpreting a failed test. Startup radial expansion rounds shells;
other reconstruction/prior estimators can use different shell rules. Correct
an erroneous new reference with evidence, preserving production and tolerances.
Preserve lazy imports alongside numerical call order when moving code.

## Command configuration and stored metadata example

- Resolve dependent settings at their producer. Startup coarse/fine sampling
  orders and the maximum/source have shared policy and are reused by numerical
  configuration, run files and reporting. Their record is separate from mutable
  iteration state and from generated sampling arrays.
- Reuse an existing complete source record before passing redundant fields.
  Archive symmetry fields come from the same provenance used by JSON reporting;
  pass only the follower replay field when that is what formatting consumes.
- Keep stored-format conversion with its file owner. Explicit NPZ casts,
  optional-field presence, sentinels and original half-row order are schema
  contracts. Test roundtrips and actual previous/current payloads when moving
  them; do not mistake serialization precision for EM precision.
- A wide formatting interface exposes unfinished provenance ownership. Trace
  pose/projector/restart sources to their validated producers and reused profile,
  archive and ledger consumers before further grouping. Do not hide the inputs
  in a temporary context object.
- Migrate indirect source-location checks with an extraction. Preserve their
  assertions and add producer-to-consumer binding coverage. Use immutable source
  equality to reuse unaffected checks after test-only repairs; failed receipts
  remain recorded rather than being overwritten.

The startup-pose example pairs a complete selected seed with its corrections and
provenance at the actual source decision. Report consumers receive provenance
only; scoring receives the arrays/corrections it needs. Preserve source-specific
metadata and borrowed arrays when they establish identity or existing lifetime.
Keep source precedence, failure timing and normalization formulas explicit in
the operation. An 18-input boundary still needs review; fixed named results and
a shorter controller do not by themselves finish that interface.

The Class3D integration supplies another concrete example: a shape merge owns
its class-summary reductions and returns them with its score result. Per-half
installation remains in the controller. Adapt new callers to that established
result boundary; keep K1 scratch release at its original boundary and preserve
the incoming dtype, shell-remapping and reduction contracts.

The paired-half execution example keeps thread launch/join, ordered exception
propagation and device-budget sharing together in the expectation owner. The
controller keeps its mode decision, callback and explicit state publication.
Preserve callback scope and join/release boundaries when moving execution work.
A result type should not carry otherwise unused scratch to imitate an old scope;
settle a changed large-buffer lifecycle as a separate measured decision.

The projector example keeps reuse, transformation, cache and dumps with their
existing owner. Bind reference identity and window metadata at the accuracy
producer, then reuse that binding directly; retain image-support and output-volume
conventions when they differ. Keep controller assignments when they control large
buffer release. A complete replacement-list result can retain the old list longer
and release its last-result alias earlier. Exercise the actual caller's lifetime,
empty-half and reuse paths; review producer and consumer together before extension.

### Numbered reconstruction and reporting-prior ownership

- A complete numbered map operation owns the substantive solve loops, premask
  capture, initial reference filtering and solvent flattening. Keep private solve
  frames when they release promoted priors or class scratch before postprocessing.
  Install ready maps explicitly; do not return raw maps solely to retain an alias.
- First-CC reconstruction uses untapered priors. Reporting taper follows it. If a
  class scheduling/history write occurs between the curve and shell/detail taper,
  leave that publication visible and extract only the remaining coherent work.
- Numbered and final phases can share formulas while keeping different policy
  orchestration. Replay, premultiplied-CTF adjustment, diagnostic ordering, final
  DVP precision and consuming accumulator release are concrete reasons to keep
  those owners distinct; a flag-heavy common wrapper would obscure the contract.
- Pair half rows with startup-noise and accuracy frames at the existing particle
  preparation boundary. Return only consumed selections; the unused full CTF
  source table can expire after advanced indexing creates an independent copy.
  Frozen identity/shell validation owns its scratch and refusal details, while
  dataset subsetting and the diagnostic admission decision remain visible.


## Conditional complexity

User clarification on October 3: the current frozen candidate is a numerical
checkpoint, not acceptance of the readability design. Audit the supported cases
and repeated mode decisions as well as operation ownership.

- Remove obsolete cases after tracing maintained callers and supported input
  contracts. Lack of coverage alone does not make a case obsolete.
- Combine adjacent checks of the same invariant when operation order, state
  updates and array release points remain unchanged.
- Resolve run-invariant source/configuration choices at their owning boundary.
  Keep iteration-dependent convergence and sampling decisions at their original
  time; a changing state is not a startup option.
- Validate required input relationships once where they become known. Keep checks
  on newly computed engine outputs and restored state where failures can arise.
- Keep genuine scientific alternatives visible. Distinct K1 and Class3D prior
  formulas do not become one formula merely to eliminate a conditional.
- Judge the full implementation: moving every branch into tiny helpers or a
  dispatch table does not reduce the number of cases a maintainer must understand.

Initial review targets: finalization repeats the K1 guard around unfiltered
reconstruction, optional low-resolution joining and accumulator release; these
are an ordered K1 sequence suitable for consolidation. Numbered reconstruction
repeatedly selects K1/Class3D around accumulator preparation, prior estimation
and model publication; inspect their whole data flow before consolidating across
intervening shared work. Replay/startup precedence and local-search state checks
need producer tracing before deletion. None of these observations establishes
that a scientific mode is dead.
