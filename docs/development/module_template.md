# The shape of a well-formed module

What a module of this repository should look like when it is done, with the refinement controller
(`relax/refinement/`) as the worked example. Each element has: why; where the code shows it today; and one
counter-example from the same code before it was changed. The rules behind it are
[refactor_rules.md](refactor_rules.md); the procedure for getting there is
[refactor_procedure.md](refactor_procedure.md).

| Element | Why | Shown today by | Counter-example, as the code was |
| --- | --- | --- | --- |
| **Options are resolved once, at the command boundary.** Flags, modes and defaults become frozen option records there; nothing below reads the command line or the mode again. | A reader of the algorithm sees what the run does, not how it was asked for; a new mode is one new resolver. | `command_options.resolve_consistency_options`: the one place `--mode` is read. It returns a frozen `RelionConsistencyOptions`; every later step reads `options.consistency.<field>`. | `full_refinement.main` keeps most run facts three times (as `args.x`, as a local, as an option field) and has about 60 conditionals on `args.`. It is still in that state. |
| **The controller shows the sequence.** One function per phase reads top to bottom as the algorithm: the calls in order, and between them every install (assignment to run state), history write and release point. | Order, state and memory lifetime are the behaviour that a move can break silently; they must be readable in one place. | `iteration_loop.refine_single_volume` (the numbered iterations) and `finalization.run_final_all_data` (the final pass): `state.current_resolution = ...`, `history.record_class_weights(...)`, `final_outs.Ft_y[0] = final_outs.Ft_y[1] = None` are statements of the controller. | `class_maximization` appended to a history list that was passed in; the write was invisible at the call. It now returns the curve and the loop appends it. |
| **Operations take operands and return results.** No writes to arguments, no reads of globals for run facts; the caller assigns. | An operation can then be tested alone, reordered knowingly, and shared between modes. | `final_reconstruction.compute_final_halfmap_prior` returns a `HalfmapPrior`; `mean_helpers.estimate_split_half_prior` returns a `SplitHalfPrior`; `class_maximization` returns a `ClassMaximization`. | The final pass took `logger` and `scoring_dtype` as parameters and returned a dict the controller then extended; one 40-line diagnostic dump was written inline between scoring steps. |
| **A record is one concept, built once, typed, read-only.** Fields belong together by meaning, not because they are computed in the same stretch. | A record that is one concept shortens every signature it enters; a bundle related by timing hides what a step needs. | `iteration_planning.RunOptics` (the run's image and optics geometry), `local_sampling.ExpectationSampling` (where and how finely one iteration scores), `mean_helpers.ReconstructionSettings`; all frozen dataclasses. | The first `ExpectationSampling` also held the direction-prior order and a sealed capture's rotation ids, because they were computed nearby. They went back to the loop. The loop also bound 16 local names to the fields of `PerHalfOutputs`; it now reads `per_half.<field>`. |
| **One mode decision; no mode flag below it.** The controller decides K=1 or Class3D; each branch calls operations of its mode, which share steps that take no mode. A one-line dispatcher at the decision is allowed; a flag in a callee is not. Do not write a whole sequence twice to remove one flag: measure first. | A flag in a callee means two functions in one body, usually with dead arguments for each mode. | `mean_helpers.reconstruct_numbered_k1_halfmaps` and `reconstruct_numbered_class_maps`; `resolution.k1_current_resolution_shell` and `class_current_resolution_shell` (shared zeroing, then one line each); the single `if k_class_enabled:` of the final reconstruction, each branch its mode's whole sequence. | One resolution function took `k_class_enabled` and both its callers passed a literal. The final reconstruction tested the mode eight times in 230 lines. Splitting the whole numbered loop into two trajectories was built and measured (about 1,500 lines written twice) and not kept. |
| **Engines are underneath and know nothing of the run.** The scoring and reconstruction engines take arrays and policies and return arrays; they do not see state, history or options. | The engine can be replaced, profiled or stood in for (the CPU fingerprint does this) without touching the controller. | `classification/k_class.run_dense_k_class_em_adaptive`, called through `half_scoring` with `DenseSamplingSpec`, `DensePriorSpec`, `DenseBatchPolicy`, `DenseExecutionPolicy`. | `tomo_half.score_tomo_half` still takes 26 separate parameters, and `DenseVariantPolicy` still carries `k_class_enabled` to the engine. |
| **Logs belong to the controller, diagnostics to `relax/diagnostics/`.** Logs may be reordered by a refactor; a diagnostic dump is one call that reads records. | Logs are how a run is compared with RELION and with itself; dumps must not lengthen or reorder the algorithm. | The final pass and its helpers log under the controller's logger name; `diagnostics.iteration.write_final_half_manifest` and `dump_numbered_iteration`. | The manifest of a final half was 40 inline lines in the scoring loop. Conversely, `dump_numbered_iteration` keeps 18 arguments: each maps to one field of one record, so taking records would only rename them. |
| **A signature tells the reader what the function needs.** Fewer than ten parameters; one representation per operand; an owner object passed whole only when the docstring names the fields read; types fixed where the value is built. | The signature is the only part most readers see. If it says `options`, the reader must assume everything. | `iteration_loop._should_use_adaptive_search`: its docstring names the one field of `state` and the one of `options` it reads. | The final pass took `options` and also `n_classes`, the sigma offset as two arguments although `SigmaOffset` exists, and class weights, log priors and assignments separately although `ClassMixture` exists: 33 parameters, now 28. |

## What a new module should be able to answer

1. Where are its options resolved, and does anything below that point read a flag, a mode or the environment?
2. Which function is the controller, and can its installs, history writes and releases be listed by reading it?
3. Does every other function return its result and leave its arguments unchanged?
4. Is each record one concept? Could its docstring be one sentence without "and"?
5. Where is the mode decided, and does `grep` find the mode's name in any callee?
6. Which checks execute it on a CPU, and which routes do only the GPU tiers reach?

## Where the worked example still falls short

- **Shared glue.** The body of the numbered loop is 1,272 lines; 872 of them are shared by both modes, and
  260 of those are calls with wide argument lists, 54 installs, 29 history writes, 76 logs. The loop reads as
  a sequence, but a long one. 29 functions of the package still take ten or more parameters, the widest 28.
- **Two blocks that have not moved.** The projector preparation (57 lines) is pinned by a test of
  release order: as a function it would release the previous half-2 projector one build earlier. The
  start-up before the loop (564 lines) interleaves a fresh and a continued start with timer marks, logs and
  three releases; log order no longer blocks it (rule 12), but the three releases must keep their place.
  The projector preparation needs the owner to accept a changed release order.
- **The local-search route has no CPU check.** The fingerprint's stand-in replaces the dense engine only;
  local search, tomography, symmetry other than C1 and multi-shape optics are verified by the GPU tiers
  alone. Every change to them rests on tests that take half an hour and do not compare an ordered trace.
- **The mode still reaches the scoring engine.** `half_scoring.DenseVariantPolicy` carries `k_class_enabled`,
  and the dense half scoring in `half_scoring.py` branches on it twice, below the controller's decision.
  The reconstruction side has no such flag left; the scoring side has.
- Also unfinished at the boundary: `full_refinement.main` (1,791 lines) is not covered by the fingerprint,
  and about 20 lines of `relax/refinement/` outside the command modules read environment variables.
