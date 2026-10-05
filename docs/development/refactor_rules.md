# Code rules

For agents changing any module of this repository. One principle governs the rest: **preserve scientific
contracts and required execution behaviour; do not preserve incidental source structure because a test
happens to observe it.** Order of priority: correctness, then GPU performance, then clarity. The owner's
rulings and their history are in `refactor_principles.md`, which decides where the two differ; how to run a
change is `refactor_procedure.md`; the target shape of a module is `module_template.md`.

## Behaviour

1. **Structure changes do not change numbers.** A refactor keeps defaults, scheduling, dtypes and scalar
   promotion, casts, operation and reduction order, random-state progression, coordinate conventions,
   saved formats and restart semantics; the default path stays bit-identical and reproduces RELION.
   Production EM is float32. Numerical, execution and structural changes go in separate commits, each with
   its own evidence. Do not fix an apparent mathematical inconsistency inside a refactor.
2. **Preserve supported interfaces, not internals.** Public functions, CLI behaviour, output schemas, saved
   formats, restart contracts and required diagnostics are contracts. Private functions, local names and the
   order of ordinary log messages are not: move them and migrate their callers and tests. Keep a
   compatibility wrapper only for a real compatibility obligation.
3. **Required lifetimes, not old scopes.** Trace each large host or device buffer through records, views,
   closures, callbacks and asynchronous consumers. Do not extend a lifetime by accident, and do not shorten
   one a consumer needs; releasing an unused buffer earlier is allowed. Preserve donation, aliasing, stream
   and FFI requirements. State any intentional lifetime change and its peak-memory effect.
4. **JAX compilation is an execution contract.** A Python function boundary is not a jit boundary: do not
   add jit regions, transfers, synchronisation or large intermediates to match the source decomposition.
   Keep Python side effects out of traced code, keep compiled-function identity stable, and choose static
   versus traced fields of any record that crosses a transformation on purpose (static fields are compile
   keys). Host-device syncs inside a loop are deliberate.

## Structure

5. **Decide once, at the owning boundary.** Arguments, modes, environment variables and defaults become
   option records where they enter, with their precedence preserved and each default defined once. Below
   that point nothing reads `args`, `os.environ` or a mode string.
6. **Split by contract, not by flag.** If two variants need different operands, invariants or state
   transitions, they are two functions, chosen by one decision; a function whose arguments are half unused
   in each mode is two contracts. A parameter that selects a legitimate variant of one coherent operation
   (a layout, a precision, a JAX static specialisation) may stay. Share an implementation where the contract
   is shared; measure before duplicating a sequence to remove a flag.
7. **One authoritative owner per fact.** No two independently maintained copies of the same value. Aliases
   that mark a different lifetime or moment, and derived representations (host/device, coordinate frames),
   are fine. Keep arrays with the metadata needed to interpret them.
8. **Configuration, state and results are different things.** Configuration is immutable once resolved
   (a frozen dataclass is not deep immutability: say what may not be mutated). State has one explicit owner.
   Results are returned as named types, not dicts. A record holds values that belong to one concept with a
   stated lifetime (run, iteration, batch), not values that happen to be computed together.
9. **Visible scientific transitions.** In a controller, persistent updates of run state, history and output
   files are statements of the controller; operations return results and do not modify their arguments
   unless they declare it (in-place buffers are allowed when stated). Diagnostics observe and never steer: a
   value the algorithm uses never travels in a log or diagnostics structure.
10. **Signatures show the dependency.** Pass the smallest coherent input: a record when it is the concept the
    function works on, fields when it reads a few of a large object. Document what is not obvious: shapes,
    dtypes, units, frames, absent states, mutation.
11. **Layers point down:** command, controller, operations, engines (and RECOVAR). Engines take arrays and
    policies and know nothing of options, state or history. Workflows do not import each other's internals;
    an authoritative formula has one home.

## Checks

12. **Check at the edge, then trust; never fall back silently.** Validate external inputs, restored state and
    newly computed invariants where they become available. An unsupported combination refuses with a
    message naming it. A supported fallback is explicit and tested; a silent substitution, skip or
    success-shaped result is a defect.
13. **Test behaviour with independent evidence.** Focused numerical tests with independently justified
    expectations, and integration through real callers, covering affected modes, non-default settings,
    boundaries and failure paths. No new test reads or executes source text; convert an existing one when the
    code it pins changes, keeping its assertions. Never weaken a tolerance, assertion or reference to make a
    change pass.
14. **Evidence matches risk, and says what it does not cover.** The fingerprint harness covers controller
    moves on a stand-in engine, not engine numbers, GPU order, memory or lifetimes. Use real-engine checks
    for numerical paths and device measurements for execution changes. Every report separates what was read,
    executed, numerically compared and performance-measured, and lists what was not.

## Changes from v1 (refactor_rules.md, main 2297cf4)

- v1 1–3 (share steps; no mode flag below the decision; do not split to satisfy it) → rule 6, now a contract
  rule: an absolute ban on mode parameters was wrong (JAX static specialisations, layout selectors, and the
  owner's retained `tau_is_1d` are legitimate).
- v1 4 → rule 7 (one owner, not one name: meaningful aliases stay). v1 5 → rule 1. v1 6–7 → rule 8 with
  lifetimes. v1 8 → rule 10. v1 9 → procedure. v1 10 → rule 9.
- v1 11 (release points never move) → rule 3. v1 contradicted the owner's ruling of 2 October.
- v1 12 (log order) → rule 2, per the owner's ruling of 5 October.
- v1 13 → rule 13. v1 16 (source-reading tests make locals an interface) dropped: those tests are converted.
- v1 14, 15, 18 → procedure. v1 17 (ceilings only go down) → procedure, as review signals with slack
  (owner, 2026-10-05). v1 19 → rule 14.
- New: rules 1 (as a rule), 2, 4, 5, 8, 11, 12, 14.
