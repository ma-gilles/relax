# Refactor rules: the short list

Nineteen rules from the work of 2026-10-04 on `relax/refinement/iteration_loop.py`, for agents who refactor
other modules. Each rule names the case that taught it. "The loop" is `refine_single_volume`; numbers are
measured unless marked as estimates. The longer record of principles and lessons, with its history, is
`docs/development/refactor_principles.md`; where the two differ, the owner's rulings recorded there decide.

## What to build

1. **Share steps, not sequences.** When two modes run the same sequence, give them shared operations and
   let each mode write its own order. Case: `mean_helpers.py` has `reconstruct_numbered_k1_halfmaps` and
   `reconstruct_numbered_class_maps`; each spells out its sequence over helpers that take no mode.
2. **No mode flag below the point where the mode is decided.** A callee that needs the mode is two callees.
   Case: `update_k1_iteration_convergence` and `update_class_iteration_convergence` take different operands
   (the native boundary; the class assignments), so a flag would also have carried dead arguments.
3. **Do not split two modes into two functions to satisfy rule 2.** Measure first. Case: splitting the loop
   into a K=1 and a Class3D trajectory was built and measured: about 1,500 lines written twice, 1,317 added.
   It was not landed. The 11 one-line dispatchers that remain are the cheaper cost.
4. **One representation per operand.** If the same value travels under several names, keep one. Case: the
   loop bound 16 names to the lists of `PerHalfOutputs` (six never read). It now reads `per_half.<field>`.
5. **Types are fixed at construction.** Do not convert a scalar's type on the way into a record and then
   use the record where the original was used. Case: `ImageGeometry.pixel_size_angstrom` is `float(...)` of
   the dataset's pixel size; the loop keeps the dataset's own scalar for host arithmetic, and the two were
   not interchanged.
6. **A record is one concept: built once, typed, fixed fields, read-only to the steps.** Good cases:
   `RunOptics` (the run's image and optics geometry), `ExpectationSampling` (the windows and the local
   sampling: where and how finely one iteration scores), `ExpectationStatistics`.
7. **Refuse a bundle related only by timing.** Case: the first `ExpectationSampling` also held the
   direction-prior order and a sealed capture's rotation ids because they were computed in the same stretch.
   They went back to the loop. Also refused: a per-iteration "context" (iteration, current size, replay
   directory), which is the loop's locals under one name, two of them rebound mid-iteration.
8. **An owner object may be taken whole only when the docstring names the fields read.** Counter-examples,
   where the signature said more before: `_should_use_adaptive_search(state, options, ...)` reads one field
   of each; `plan_adaptive_image_size(..., options, ...)` reads two unrelated option groups;
   `refresh_coarse_grids` now derives a rule from the replay directory and an option, out of the loop's
   sight. Declined for this reason: passing `state` to three planners for `ave_Pmax` alone.
9. **Do not build a step that only renames arguments.** Case: `dump_numbered_iteration` has 18 arguments;
   each maps to one field of one record, so taking records would remove none.

## What must stay visible

10. **Installs, history writes and release points stay in the controller.** A step returns; the controller
    assigns. Case: `class_maximization` appended to a history list passed in; it now returns the curve and
    the loop writes `history.data_vs_prior_trajectory`.
11. **Release points do not move, even to release earlier.** Case: extracting the projector preparation
    would release the previous half-2 projector one build earlier, because a loop variable kept it alive.
    A weak-reference test pins it; the extraction was not done.
12. **Log order is behaviour.** The fingerprint compares the ordered trace. Case: moving the direction
    priors or the pass-1 size plan into the sampling step would have reordered their log records.
13. **A rule moved into a callee needs a test there.** Case: `plan_expectation_windows` now decides that
    shape classes are not remapped at run level; it got a unit test in the same commit.

## How to verify and land

14. **Every landing needs fingerprint cases that leave the defaults.** Case: no case set a consistency
    option, so removing a name read only under `noise_shell_count == "summed"` showed 0 differences; lint
    caught it. Two cases with non-default options now catch it (both fail with `NameError`).
15. **Search all of `tests/` and `scripts/` for every moved or removed name before a GPU tier.** The CPU
    unit list does not run GPU unit files. Case: a GPU-only test monkeypatched
    `iteration_loop.plan_expectation_windows` after the step had moved it; the medium tier failed on it.
16. **A function-local name is also an interface when tests execute slices of the function.** 22 test files
    read or execute text of the loop; renaming a local broke their fabricated namespaces. `grep` the name.
17. **Ceilings only go down.** Lower them as the last commit of a slice. The one exception is a requested
    feature, with the new value and the reason in its commit. Case: the consistency options raised the span
    by ten lines on purpose; the refactor slices after it lowered it again (2,193 to 2,036).
18. **One idea per commit, each fingerprint-clean against its parent,** then the stack against
    `origin/main`. After a rebase re-run lint: a clean rebase can still leave a dangling name (rule 14).
19. **Say what the checks do not cover.** Case: local search has no fingerprint case (the local route does
    not reach the stand-in engine); every report on a step that touched it said so and relied on the tiers.
