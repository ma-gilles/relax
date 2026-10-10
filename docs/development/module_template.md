# The shape of a well-formed module

What a module of this repository should look like when it is done, with the refinement controller
(`relax/refinement/`) as the worked example. Each element has: why; where the code shows it today; and one
counter-example from the same code before it was changed. The rules behind it are
[refactor_rules.md](refactor_rules.md) (the code rules); the procedure for getting there is
[refactor_procedure.md](refactor_procedure.md).

| Element | Why | Shown today by | Counter-example, as the code was |
| --- | --- | --- | --- |
| **Options are resolved once, at the command boundary.** Flags, modes and defaults become frozen option records there; nothing below reads the command line or the mode again. | A reader of the algorithm sees what the run does, not how it was asked for; a new mode is one new resolver. | `command_options.resolve_consistency_options`: the one place `--mode` is read. It returns a frozen `RelionConsistencyOptions`; every later step reads `options.consistency.<field>`. | `full_refinement.main` keeps most run facts three times (as `args.x`, as a local, as an option field) and has about 60 conditionals on `args.`. It is still in that state. |
| **The controller shows the sequence.** One function per phase reads top to bottom as the algorithm: the calls in order, and between them every install (assignment to run state), history write and release point. | Order, state and memory lifetime are the behaviour that a move can break silently; they must be readable in one place. | `iteration_loop.refine_single_volume` (the numbered iterations) and `finalization.run_final_all_data` (the final pass): `state.current_resolution = ...`, `history.record_class_weights(...)`, `final_outs.Ft_y[0] = final_outs.Ft_y[1] = None` are statements of the controller. | `class_maximization` appended to a history list that was passed in; the write was invisible at the call. It now returns the curve and the loop appends it. |
| **Operations take operands and return results.** No writes to arguments, no reads of globals for run facts; the caller assigns. | An operation can then be tested alone, reordered knowingly, and shared between modes. | `final_reconstruction.compute_final_halfmap_prior` returns a `HalfmapPrior`; `priors.estimate_split_half_prior` returns a `SplitHalfPrior`; `class_maximization` returns a `ClassMaximization`. | The final pass took `logger` and `scoring_dtype` as parameters and returned a dict the controller then extended; one 40-line diagnostic dump was written inline between scoring steps. |
| **A result is typed; the report is built from it, never the other way round.** An operation (or a callback) returns a named result; a log or report dictionary is derived from that result, and no caller reads an operand of the next step out of a report. | A dictionary that is both transport and report hides which keys the algorithm needs, and a log edit can change the numbers (rule 9). | `refinement.maximization.class_maximization` returns a `ClassMaximization`; the loop takes its fields and writes the history itself. | VDAM's E-step callback returns `(accumulators, meta)` and the loop read the M-step operands out of `meta`. `estep_common.estep_sums` now builds a frozen `EstepSums` from it once, at the call; the callback's own result type is still open (`vdam_rules_status.md`, item 3). |
| **A record is one concept, built once, typed, read-only.** Fields belong together by meaning, not because they are computed in the same stretch. | A record that is one concept shortens every signature it enters; a bundle related by timing hides what a step needs. | `iteration_planning.RunOptics` (the run's image and optics geometry), `local_sampling.ExpectationSampling` (where and how finely one iteration scores), `mean_helpers.ReconstructionSettings`; all frozen dataclasses. | The first `ExpectationSampling` also held the direction-prior order and a sealed capture's rotation ids, because they were computed nearby. They went back to the loop. The loop also bound 16 local names to the fields of `PerHalfOutputs`; it now reads `per_half.<field>`. |
| **Split by contract, not by flag.** The controller decides K=1 or Class3D; where the modes need different operands, invariants or state transitions, each branch calls its mode's operation, and the operations share the steps whose contract is shared. A one-line dispatcher at the decision is allowed. A parameter that selects a legitimate variant of one coherent operation (a layout, a precision, a JAX static specialisation) may stay (rule 6). Do not write a whole sequence twice to remove one flag: measure first. | A mode flag whose arguments are half unused in each mode is two functions in one body, with dead arguments for each. | `mean_helpers.reconstruct_numbered_k1_halfmaps` and `reconstruct_numbered_class_maps`; `resolution.k1_current_resolution_shell` and `class_current_resolution_shell` (shared zeroing, then one line each); the single `if k_class_enabled:` of the final reconstruction, each branch its mode's whole sequence. In `relax/vdam`, the optimizer is two records with the same methods called at the same points of the loop (`VdamUpdate`, `MomentumSgdUpdate`), built once by the driver. | One resolution function took `k_class_enabled` and both its callers passed a literal. The final reconstruction tested the mode eight times in 230 lines. Splitting the whole numbered loop into two trajectories was built and measured (about 1,500 lines written twice) and not kept. |
| **Engines are underneath and know nothing of the run.** The scoring and reconstruction engines take arrays and policies and return arrays; they do not see state, history or options. | The engine can be replaced, profiled or stood in for (the CPU fingerprint does this) without touching the controller. | `classification/k_class.run_dense_k_class_em_adaptive`, called through `half_scoring` with `DenseSamplingSpec`, `DensePriorSpec`, `DenseBatchPolicy`, `DenseExecutionPolicy`. | `tomo_half.score_tomo_half` still takes 26 separate parameters, and `DenseVariantPolicy` still carries `k_class_enabled` to the engine. |
| **Logs belong to the controller; observation and comparison enter through ports.** Logs may be reordered by a refactor. A dump, capture, profile or timing is a `RunObserver` hook (`relax/refinement/ports.py`, the writing in `relax/diagnostics/observers.py`); an input taken from RELION is an `InputSource` method whose default is the native value (implemented in `relax/parity`). The command chooses both once (code rule 15). | Logs are how a run is compared with RELION and with itself; dumps and replays must not lengthen, reorder or steer the algorithm, and the algorithm must read the same with or without them. | `observer.maps_reconstructed(ReconstructedIteration(...))` and `source.numbered_state(...)` in `iteration_loop.refine_single_volume`; `IntermediatesObserver`, `RelionReplaySource`. | The controller tested `debug.save_intermediates_dir` five times and `_parity_dump.is_active()` at each stage; a 60-line block applied RELION's per-iteration replay inline and the replay settings were fields of `RelionParityOptions` and `StartState`. |
| **A signature tells the reader what the function needs.** The smallest coherent input: a record when it is the concept the function works on, fields when it reads a few of a large object (rule 10); one owner per fact (rule 7); types fixed where the value is built. Ten or more parameters is a signal to look for a missing record or a split by contract (rules 6, 8 and 10); it is not by itself a reason not to build. | The signature is the only part most readers see. If it says `options`, the reader must assume everything. | `iteration_loop._should_use_adaptive_search`: its docstring names the one field of `state` and the one of `options` it reads. | The final pass took `options` and also `n_classes`, the sigma offset as two arguments although `SigmaOffset` exists, and class weights, log priors and assignments separately although `ClassMixture` exists: 33 parameters, now 28. |

## What a new module should be able to answer

1. Where are its options resolved, and does anything below that point read a flag, a mode or the environment?
2. Which function is the controller, and can its installs, history writes and releases be listed by reading it?
3. Does every other function return its result and leave its arguments unchanged?
4. Is each record one concept with a stated lifetime (run, iteration, batch), rather than values computed
   together?
5. Where is the mode decided, and does any callee take a mode whose values need different operands,
   invariants or state transitions?
6. Which checks protect it, chosen by its kind (the fingerprint for a controller, independent references and
   device tests for a kernel, precedence, rejection and round-trip tests for a parser), and which routes do
   only the GPU tiers reach?

## What the worked example taught (apply to the next module)

- Harness first, command entry included; then convert every test that reads source.
- One verify per commit, but search and run every test file that names what moved.
- Run new test helpers once on a GPU node.
- Environment variables enter through an options record with a default factory, read once.
- Group values into records only after tracing each array's release point.
- A ceiling warning needs a stated reason; lower the ceilings after each slice.

## Where the worked example still falls short

The current list of open items, coverage limits and decisions is
[refinement_rules_status.md](refinement_rules_status.md); keep it current as items close.

- **Shared glue.** The body of the numbered loop is 1,247 lines; 847 of them are shared by both modes, and
  256 of those are calls with wide argument lists, 52 installs, 29 history writes, 64 logs. The loop reads as
  a sequence, but a long one. 29 functions of the package still take ten or more parameters, the widest 28.
- **The start-up decides the mode once, except for the grid.** A fresh and a continued run are one
  `if resume is None` with a block each, and each block releases the start-up arrays it was given. The
  initial coarse grid still chooses its inputs inline (schedule or restored state), because the fresh
  direction prior needs the built grid's order.
- **The local-search route has no CPU check.** The fingerprint's stand-in replaces the dense engine only;
  local search, tomography, symmetry other than C1 and multi-shape optics are verified by the GPU tiers
  alone. Every change to them rests on tests that take half an hour and do not compare an ordered trace.
- **The mode in the scoring engine is a variant (decided).** `half_scoring.DenseVariantPolicy` carries
  `k_class_enabled`, and the dense half scoring branches on it twice, below the controller's decision. It
  selects a variant of one coherent operation; revisit only if K=1 and Class3D need different operands there.
- Also unfinished at the boundary: `full_refinement.main` is 1,020 lines (was 1,791). The fingerprint runs it
  (19 `main_*` cases: K=1 and Class3D runs, `--continue`, schedules, start-up noise, the ledger, refused
  commands; not the frozen boundary or RELION replay, which only the GPU tiers run) and its start-up maps,
  start-up noise and prior, half sets, replay inputs, follower routing, restart
  provenance and reports are now functions with their own tests. What remains inline is the order of those
  calls and the translation of about 200 `args.` reads into the controller's option records (the call alone is
  170 lines); `command_options` resolves only part of them. About 20 lines of `relax/refinement/` outside the
  command modules read environment variables, mostly diagnostic dump directories.
