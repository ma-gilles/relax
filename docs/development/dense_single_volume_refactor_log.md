# Dense single-volume refinement refactor log

This is the compact execution record for
[`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md).
Detailed test output and run artifacts belong in their recorded run roots.

## 2026-09-24 — work package 0 started

### Objective

Establish a reproducible structural baseline and characterization boundary
before moving production code.

### Source and workspace

- Branch: `relax_refactor`
- HEAD: `1f530682c9087c304012cd75376ca71fdd604e43`
- Merge base with `main`: `1f530682c9087c304012cd75376ca71fdd604e43`
- Initial tracked diff: empty
- Initial diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Pre-existing untracked paths, not owned by this work: `.vscode/`,
  `relax/cuda/librelax_cuda.so.sources.sha256`

### Initial measurements

| Metric | Value |
| --- | ---: |
| Production files | 10 |
| Physical production lines | 10,397 |
| Nonblank, non-comment production lines | 9,232 |
| Production functions | 100 |
| Functions with at least 10 parameters | 13 |
| Functions with at least 20 parameters | 5 |
| Largest parameter count | 63 |
| Largest function span | 4,753 lines |

### Work in progress

- Added the durable refactor plan and this execution log.
- Added a standard-library structural reporter and checked metrics snapshot.
- No production algorithm code has changed in this package.

### Acceptance for package 0

- The structural reporter reproduces the recorded baseline and detects scope or
  metric drift.
- Focused reporter tests pass.
- Development-guide links and mirrors pass.
- The CPU EM fast guard passes from the frozen checkout environment.

### Result

- Structural snapshot check: passed.
- Focused reporter tests: 2 passed.
- Ruff check and format check: passed.
- CPU EM fast guard: 99 passed in 60.67 seconds after loading
  `FFTW/3.3.10-GCC-12.2.0` for the existing RELION binding. The first attempt
  failed four C4 cases at binding load because `libfftw3.so.3` was absent from
  the runtime path; no numerical test failed.
- `scripts/check_agent_guides.py`: existing external scratch links in the
  development guides are unavailable in this checkout. The new plan/log link
  is valid; no agent guide was edited.

## 2026-09-24 — initial coarse-grid specification boundary

### Objective

Start work package 1 with one isolated host-side planning boundary, replacing
the first 10-parameter controller helper without changing grid construction.

### Change

- Added `refinement.iteration_planning.InitialCoarseGridSpec` and
  `InitialCoarseGrids`, both frozen shallow containers.
- Moved initial sealed/canonical coarse-grid construction from the controller
  to `build_initial_coarse_grids`.
- The controller constructs one specification and consumes the returned grid
  fields in the same order. No casts, grid calls or array conversions changed.
- Migrated the focused ownership tests to the new owner.

### Structural delta

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Physical production lines | 10,397 | 10,434 | +37 |
| Nonblank, non-comment lines | 9,232 | 9,261 | +29 |
| Functions with at least 10 parameters | 13 | 12 | -1 |
| Largest function span | 4,753 | 4,755 | +2 |

The line increase is specification/result documentation, explicit field names
and module ownership; the moved calculation itself is unchanged.

### Validation

- Structural snapshot check: passed.
- Initial-grid ownership plus metric tests: 10 passed.
- Ruff check and format check for affected Python files: passed.
- CPU EM fast guard with the FFTW runtime loaded: 99 passed in 59.96 seconds
  on the exact final source available for GPU qualification.
- GPU smoke tier: blocked before submission. The repository runner could not
  create its configured `/scratch/gpfs/...` run root on this host. Retrying
  with the explicit writable run root
  `/home/ry295/palmer_scratch/tmp/relax_smoke_1f53068_20260924_1801`
  reached the mandatory fixture check, which reported all five pinned fixture
  sets unavailable under `/scratch/gpfs/...`. No smoke tests ran and no test
  receipt or Slurm job ID exists; the fixture check was not bypassed.

## 2026-09-28 — rebase onto current `origin/main`

### Rebase result

- Rebased from `1f530682c9087c304012cd75376ca71fdd604e43` onto
  `363ed495c1692e521f243bf6c0b6b2a35841f5a2`.
- Preserved current `origin/main` continuation, multi-shape optics, projection
  scaling, exact-coarse scoring, batching and regression behavior.
- Replayed the initial coarse-grid ownership boundary with the continuation's
  restored sampling values.
- Dropped seven later historical implementation/receipt commits whose
  pre-multi-shape scoring bodies conflicted with current production behavior.
  Their ownership changes will be reimplemented against the current code rather
  than overriding upstream behavior.
- Renamed the surviving owner from `InitialCoarseGridRequest` to
  `InitialCoarseGridSpec`; production specification bodies retain field
  ownership instead of flattening their contents into parallel locals.

### Active structural baseline

| Metric | Current `origin/main` | Coarse-grid spec | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,184 | 14,221 | +37 |
| Nonblank, non-comment lines | 12,410 | 12,439 | +29 |
| Production functions | 223 | 223 | 0 |
| Functions with at least 10 parameters | 17 | 16 | -1 |
| Functions with at least 20 parameters | 8 | 8 | 0 |
| Largest parameter count | 67 | 67 | 0 |
| Largest function span | 5,233 | 5,235 | +2 |

The checked snapshot now uses the rebased `origin/main` tree as its comparison
baseline. Historical measurements above remain the record of the original
work package rather than a comparison against today's production tree.

## 2026-09-28 — first-iteration CC specification ownership

### Change

- Replaced the 26-parameter first-iteration CC dispatch with
  `FirstIterCCSpec` and five cohesive owners for data, grid, scoring policy,
  batching and execution controls.
- The scoring body keeps those five owners visible; it does not unpack their
  fields into parallel local variables. An AST regression enforces that rule.
- K=1 and K-class differences are derived with `dataclasses.replace`, while
  preserving the current exact-coarse preprocessing decision, batch mutation,
  grid construction and engine-call order.

### Structural delta from the rebased baseline

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Physical production lines | 14,185 | 14,309 | +124 |
| Nonblank, non-comment lines | 12,410 | 12,507 | +97 |
| Functions with at least 10 parameters | 17 | 15 | -2 |
| Functions with at least 20 parameters | 8 | 7 | -1 |
| Largest parameter count | 67 | 67 | 0 |
| Largest function span | 5,233 | 5,235 | +2 |

The line increase is explicit ownership documentation and construction. The
dispatch has one parameter and no forwarding/compatibility layer.

### Validation

- First-iteration CC focused suite: 19 passed.
- First-iteration, projector, BPref-scope and K-class regression set: 77 passed.
- Ruff lint, Python compilation, structural snapshot and diff checks: passed.
- The attempted historical regression command named a test file removed on
  current `origin/main`; the current test inventory was used instead.

## 2026-09-28 — periodic rebase onto `e2401c4c`

- Rebased all four refactor commits onto
  `e2401c4c53182f6aed4e12b060b28257ef86894f`.
- Conflicts: none.
- The three upstream commits changed tomography execution and status
  documentation; they replayed unchanged and did not require an EM resolution
  choice.
- The active upstream structural baseline is now 14,185 physical production
  lines, 12,410 nonblank/non-comment lines, 17 functions with at least 10
  parameters and 8 with at least 20. The first-iteration package remains a
  reduction to 15 and 7 respectively.

## 2026-09-28 — dense half-scoring specification ownership

### Change

- Replaced the 58-parameter dense half scorer and dictionary-backed half plan
  with `DenseHalfScoringSpec` and cohesive owners for half data, sampling,
  priors, batching, route policy, execution policy and optics adaptations.
- The scoring body reads stable values through those seven owners. It creates
  local variables only for five values changed by route planning plus the
  canonicalized symmetry; an AST regression prevents wholesale field
  unpacking.
- Normal numbered iterations and the final all-data pass now construct the
  same typed boundary. Multi-shape execution derives each shape's shallow spec
  with `dataclasses.replace`, preserving the existing authoritative optics
  transformations and fixed-order result merge.
- The shared adaptive-engine keyword builder now receives the specification
  instead of forwarding its own 13-argument subset. No compatibility wrapper
  or parallel kwargs API remains.

### Structural delta from the active upstream baseline

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,185 | 14,525 | +340 |
| Nonblank, non-comment lines | 12,410 | 12,695 | +285 |
| Production functions | 223 | 224 | +1 |
| Functions with at least 10 parameters | 17 | 13 | -4 |
| Functions with at least 20 parameters | 8 | 6 | -2 |
| Largest parameter count | 67 | 67 | 0 |
| Largest function span | 5,234 | 5,273 | +39 |

The cumulative line increase covers the initial-grid, firstiter-CC and dense
half-scoring ownership contracts. In this package, explicit constructors and
the typed multi-shape derivation add more lines than the removed argument and
kwargs plumbing; the numerical bodies and their statement order remain in
place. The next controller-splitting packages must recover that orchestration
growth rather than moving code outside the counted package.

### Validation

- Implementation commit:
  `f05a396949946e2963d6f89f431f075acbba448e`.
- Dense routing, multi-shape, firstiter-CC, BPref-scope, zero-coarse and
  K-class focused set: 149 passed.
- Numbered/final controller handoff and tau2/scoring-prior probes: 18 passed,
  365 deselected.
- Complete symmetry unit suite: 360 passed.
- Production Ruff lint, Python compilation and diff checks: passed. The edited
  `test_refine_relion_mode.py` retains unrelated pre-existing import-order
  findings; it was not broadly reformatted.
- CPU EM fast guard: 102 passed in 57.80 seconds on committed source
  `25ac2dd356202b591767b766427c889086c1b934` after loading
  `FFTW/3.3.10-GCC-12.2.0`; exit 0. The initial sandboxed invocation could not
  acquire Pixi's shared-prefix lock, so the identical command was rerun with
  access to that lock.
- GPU smoke tier: infrastructure-blocked before execution. The default
  `/scratch/gpfs/...` run root is unavailable on this host. With the explicit
  writable run root
  `/home/ry295/palmer_scratch/tmp/relax_smoke_a2a1b6e_20260928_1039`, the
  runner froze the committed source and then failed mandatory verification for
  all five pinned K1/K2 fixture sets because their `/scratch/gpfs/...` inputs
  are absent. No GPU test ran; no Slurm job ID or test receipt exists.

## 2026-09-28 — exact-local half-scoring specification ownership

### Change

- Replaced the 49-parameter exact-local scorer and its `**kwargs` wrappers with
  `LocalHalfScoringSpec` and cohesive owners for half data, sampling, priors,
  batching, execution, diagnostics and optics adaptations.
- The scorer keeps those seven owners visible and does not unpack their stable
  fields. An AST ownership regression enforces the boundary.
- Numbered and final all-data local-search calls construct the same typed spec.
  Multi-shape execution continues to use `optics_shapes.class_kwargs` for the
  authoritative per-image slicing, translation scaling, Fourier sizes and
  reference-grid projection controls, then derives shallow per-shape specs with
  `dataclasses.replace`.
- Device-signature scope now reads its diagnostic policy directly; no local
  scorer compatibility/forwarding API remains. Test-only concise fixtures use a
  rejecting builder that fails on any unmapped former argument.

### Structural delta from the active upstream baseline

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,185 | 14,815 | +630 |
| Nonblank, non-comment lines | 12,410 | 12,956 | +546 |
| Production functions | 223 | 225 | +2 |
| Functions with at least 10 parameters | 17 | 12 | -5 |
| Functions with at least 20 parameters | 8 | 5 | -3 |
| Largest parameter count | 67 | 67 | 0 |
| Largest function span | 5,234 | 5,369 | +135 |

This package adds 290 physical production lines while removing the 49-argument
boundary. The increase is the explicit local ownership contracts, two typed
controller constructions and the per-shape derivation. It is accepted
provisionally for ownership clarity; later controller extraction must
consolidate identical construction without recreating a forwarding layer.

### Validation

- Implementation commit:
  `58b371d59b88bbca9ff6ef39f576e3ed9cd5c594`.
- AST-normalized comparison against the parent commit: passed. After removing
  only the seven owner aliases and mapping `owner.field` to its former parameter
  name, the entire exact-local numerical/control body is identical.
- Broad selected local-search/controller run: 194 behavioral cases passed; its
  sole failure was a source-text assertion expecting the former one-line
  diagnostic argument. The assertion was migrated to the typed policy.
- Corrected dense/local, BPref, multi-shape, ownership, complete symmetry and
  two direct local-controller regressions: 439 passed.
- Production and affected-helper Ruff lint, Python compilation and diff checks:
  passed. Unrelated pre-existing import-order findings in the large legacy test
  modules were left untouched.
- CPU EM fast guard: 102 passed in 59.92 seconds on implementation commit
  `58b371d59b88bbca9ff6ef39f576e3ed9cd5c594`; exit 0. The host has no CUDA
  device, so the expected JAX CUDA-plugin discovery warning preceded the
  forced-CPU run without affecting the checks.
- GPU smoke remains infrastructure-blocked because the five mandatory
  `/scratch/gpfs` fixture sets are absent on this host, as recorded in the
  preceding package. No new submission was attempted against the same missing
  inventory.

## 2026-09-28 — local-search iteration specification ownership

### Change

- Replaced the remaining 67-parameter production boundary,
  `_run_local_search_iteration`, with `LocalSearchIterationSpec` and cohesive
  owners for data, grids, batching, numerical-kernel policy, posterior/support
  policy and diagnostics.
- The implementation keeps those six owners visible. It introduces locals only
  for image and rotation batch sizes and the two prior arrays because those
  values are deliberately normalized or replanned inside the function. An AST
  ownership regression rejects flattening the stable fields back into locals.
- Exact-local parent, broad-denominator diagnostic and final fine passes now
  derive named variants from one shallow base specification with
  `dataclasses.replace`. Their distinct precision, projection, support,
  accumulation, returned-detail and debug policies remain explicit.
- Direct tests use a rejecting test-only builder for concise fixtures. Production
  has no compatibility wrapper, legacy kwargs adapter or field-unpacking layer.

### Structural delta from the active upstream baseline

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,185 | 14,878 | +693 |
| Nonblank, non-comment lines | 12,410 | 12,998 | +588 |
| Production functions | 223 | 225 | +2 |
| Functions with at least 10 parameters | 17 | 11 | -6 |
| Functions with at least 20 parameters | 8 | 4 | -4 |
| Largest parameter count | 67 | 28 | -39 |
| Largest function span | 5,234 | 5,369 | +135 |

This package adds 63 physical production lines while removing the largest
argument boundary in the counted package. The remaining maximum is the
28-parameter reconstruction/postprocessing boundary in `mean_helpers.py`.

### Validation

- Implementation commit:
  `0d053e4365a78924127f57979ec109cadcf78e32`.
- AST-normalized comparison against its parent: passed. After removing the six
  owner aliases and the four mutable planning initializers, then translating
  `owner.field` back to its former parameter name, the complete local-search
  control/numerical body is identical.
- Focused owner, call-contract and exact-local variant tests: 23 passed across
  the direct controller and half-scoring selections.
- Resident local dispatch: 23 passed and 12 CUDA-only cases skipped. The
  independent RELION reference file passed 4 CPU cases and skipped 16
  CUDA-only cases.
- Broadened dense/local, BPref, symmetry, optics, firstiter and controller set:
  594 passed and 28 CUDA-only cases skipped. Five replay cases could not set up
  because their pinned `/scratch/gpfs/...` RELION fixture is absent; this is the
  same external fixture block recorded for smoke.
- Production and affected-helper Ruff lint, Python compilation, snapshot check
  and diff checks: passed.
- CPU EM fast guard: 102 passed in 57.50 seconds on the implementation commit
  after loading `FFTW/3.3.10-GCC-12.2.0`. The first invocation intentionally
  documented the repository setup trap: 98 passed and four C4 cases failed to
  load the existing binding without `libfftw3.so.3`; the corrected invocation
  passed all cases.
- GPU smoke remains infrastructure-blocked because the five mandatory
  `/scratch/gpfs` fixture sets are still absent. No Slurm job or GPU receipt was
  produced.

## 2026-09-28 — mean-reconstruction specification ownership

### Change

- Replaced the 28-parameter regularized reconstruction/post-processing
  boundary with `MeanReconstructionSpec` and explicit owners for caller-owned
  means, M-step accumulators, tau2/regularization inputs, Fourier geometry and
  post-processing policy.
- `_reconstruct_and_postprocess_means` keeps all five owners visible. The only
  former field copied to a local is the retained half-0 device numerator,
  because its deliberate release between half reconstructions is part of the
  memory-lifetime contract. An AST ownership regression enforces this rule.
- The numbered-iteration controller constructs the specification at the
  reconstruction boundary. Direct unit tests use a rejecting test-only builder;
  production has no legacy adapter or field-unpacking layer.
- K=1 versus K-class reconstruction, per-half versus class tau2, host staging,
  initial low-pass filtering, solvent flattening, debug capture and retained
  buffer release remain in their original order.

### Structural delta from the active upstream baseline

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,185 | 14,959 | +774 |
| Nonblank, non-comment lines | 12,410 | 13,060 | +650 |
| Production functions | 223 | 225 | +2 |
| Functions with at least 10 parameters | 17 | 10 | -7 |
| Functions with at least 20 parameters | 8 | 3 | -5 |
| Largest parameter count | 67 | 26 | -41 |
| Largest function span | 5,234 | 5,395 | +161 |

This package adds 81 physical production lines while eliminating the previous
28-parameter maximum. The largest remaining boundary is the 26-parameter
iteration snapshot capture.

### Validation

- Implementation commit:
  `0d1247a8305978dcf86a9d952922babc8083752e`.
- AST-normalized comparison against its parent: passed. After removing the five
  owner aliases and the retained-buffer initializer, then translating
  `owner.field` back to its former parameter name, the complete reconstruction
  and post-processing body is identical.
- Reconstruction ownership, retained-buffer, solvent-mask and K1/K-class
  policy tests: 25 passed; 381 unrelated cases deselected.
- K1 mean lifetime and controller ordering guards: 5 passed.
- Production and affected-helper Ruff lint, Python compilation, snapshot check
  and diff checks: passed.
- CPU EM fast guard: 102 passed in 57.59 seconds on the implementation commit
  with `FFTW/3.3.10-GCC-12.2.0` loaded.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-28 — unregularized-mean specification ownership

### Change

- Replaced the 14-parameter unregularized-map/sign-publication boundary with
  `UnregularizedMeanSpec` and owners for current/previous means, M-step
  accumulators and reconstruction policy.
- The function keeps all three owners visible and does not unpack stable fields;
  an AST ownership regression enforces the boundary.
- K1 versus K-class accumulator selection, optional diagnostic reconstruction,
  shared K-class map identity, sign alignment and log order remain unchanged.

### Structural delta from the active upstream baseline

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,187 | 15,150 | +963 |
| Nonblank, non-comment lines | 12,410 | 13,207 | +797 |
| Production functions | 223 | 225 | +2 |
| Functions with at least 10 parameters | 17 | 7 | -10 |
| Functions with at least 20 parameters | 8 | 2 | -6 |
| Largest parameter count | 67 | 21 | -46 |
| Largest function span | 5,234 | 5,464 | +230 |

This package adds 46 physical production lines. Excluding the two tomography
boundaries, five single-volume functions remain at 10+ parameters.

### Validation

- Implementation commit:
  `37ac32c76815a68d65b22c1c6c78e69458091ac8`.
- AST-normalized comparison against its parent: passed after removing the three
  owner aliases and translating `owner.field` back to its former parameter
  name; the complete reconstruction/sign body is identical.
- Reconstruction ownership and K-class sign-publication tests: 16 passed; 379
  unrelated cases deselected.
- Production and affected-test Ruff lint, Python compilation, snapshot check
  and diff checks: passed.
- CPU EM fast guard: 102 passed in 57.40 seconds on the implementation commit
  with `FFTW/3.3.10-GCC-12.2.0` loaded.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-28 — iteration-snapshot specification ownership

### Change

- Replaced the 26-parameter snapshot capture boundary with
  `IterationSnapshotSpec` and cohesive owners for run identity, reference/model
  state, sampling/convergence state and per-particle state.
- `capture_iteration_snapshot` keeps all four owners visible and does not unpack
  any stable field. An AST ownership regression enforces the one-spec boundary
  and rejects field flattening.
- The numbered-iteration checkpoint branch constructs the snapshot spec at the
  existing `writer.due` boundary. Host copies, dtype recording, K1/K-class
  aliasing, scalar conversion and direction-prior metadata remain in their
  original order.

### Structural delta from the active upstream baseline

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,187 | 15,027 | +840 |
| Nonblank, non-comment lines | 12,410 | 13,111 | +701 |
| Production functions | 223 | 225 | +2 |
| Functions with at least 10 parameters | 17 | 9 | -8 |
| Functions with at least 20 parameters | 8 | 2 | -6 |
| Largest parameter count | 67 | 21 | -46 |
| Largest function span | 5,234 | 5,418 | +184 |

This package adds 66 physical production lines. The only remaining 20+
argument functions are the 21- and 20-parameter tomography scoring boundaries,
which are outside the current dense single-volume scope. The largest remaining
single-volume boundary has 17 parameters.

### Validation

- Implementation commit:
  `b7afa62dd8e859e36fc4838def343a42d0440951`.
- AST-normalized comparison against its parent: passed. After removing only the
  four owner aliases and translating `owner.field` back to its former parameter
  name, the complete snapshot capture body is identical.
- Snapshot/run-file ownership and round-trip suite: 13 passed.
- Production and affected-test Ruff lint, Python compilation, snapshot check
  and diff checks: passed.
- CPU EM fast guard: 102 passed in 57.92 seconds on the implementation commit
  with `FFTW/3.3.10-GCC-12.2.0` loaded.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-28 — periodic rebase onto `e0f50017`

- `origin/main` advanced from `e2401c4c53182f6aed4e12b060b28257ef86894f`
  to `e0f5001739058ff894895857dddb938196031632`.
- Conflicts: none. All 16 refactor commits replayed unchanged, so no manual
  resolution was required and upstream was retained wholesale.
- The three upstream commits fix tomography group scaling and update its
  benchmark/status evidence. They do not overlap the single-volume refactor.
- Upstream adds two physical lines to `tomo_half.py` without changing the
  nonblank count or any tracked argument metric. The active baseline is now
  14,187 physical lines; the rebased candidate is 15,027.
- Rebased implementation receipts and the checked snapshot now reference the
  rewritten commit IDs. The frozen smoke worktree path retains its historical
  pre-rebase hash because that path names the run that actually occurred.
- Fresh post-rebase ownership, reconstruction, snapshot, local-controller and
  resident-reference selection: 124 passed and 28 CUDA-only cases skipped.
- Fresh post-rebase CPU EM fast guard: 102 passed in 56.88 seconds with
  `FFTW/3.3.10-GCC-12.2.0` loaded.

## 2026-09-28 — direction-prior update specification ownership

### Change

- Replaced the 16-parameter direction-prior publication boundary with
  `DirectionPriorUpdateSpec` and owners for posterior inputs, caller-owned
  mutable priors, scoring-grid identity and execution precision/logging.
- `update_learned_direction_priors` keeps those four owners visible and does
  not unpack stable fields. An AST ownership regression enforces the boundary.
- The numbered-iteration controller constructs the specification where it
  already resolves the K1 scoring order and exhaustive K-class grid size.
  K1/K-class route selection, symmetry handling, validation/warning order and
  independent per-half copies remain unchanged.

### Structural delta from the active upstream baseline

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,187 | 15,104 | +917 |
| Nonblank, non-comment lines | 12,410 | 13,173 | +763 |
| Production functions | 223 | 225 | +2 |
| Functions with at least 10 parameters | 17 | 8 | -9 |
| Functions with at least 20 parameters | 8 | 2 | -6 |
| Largest parameter count | 67 | 21 | -46 |
| Largest function span | 5,234 | 5,452 | +218 |

This package adds 77 physical production lines. Excluding the two out-of-scope
tomography scoring boundaries, six single-volume functions remain at 10+
parameters and the largest has 17.

### Validation

- Implementation commit:
  `5e21a5020fe01ce060bd32376aaba2880f55fca9`.
- AST-normalized comparison against its parent: passed after removing the four
  owner aliases and translating `owner.field` back to its former parameter
  name; the complete update body is identical.
- Direction-prior update, ownership and snapshot-consumption tests: 18 passed.
- Production and affected-test Ruff lint, Python compilation, snapshot check
  and diff checks: passed.
- CPU EM fast guard: 102 passed in 56.83 seconds on the implementation commit
  with `FFTW/3.3.10-GCC-12.2.0` loaded.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-28 — owner-visible state boundaries

- Commit `1811ebe8378802dd3d829c52cc9dfba15744b331` removed the call-only
  reconstruction, direction-prior, unregularized-mean and snapshot wrappers.
  Their functions now expose three to five cohesive owners directly.
- Focused ownership and behavior tests: 26 passed, 405 deselected. CPU EM fast
  guard: 102 passed in 58.74 seconds.
- Production totals fell to 15,094 physical and 13,163 nonblank lines.

## 2026-09-28 — owner-visible pass boundaries

- Commit `b1966cea4758c12449ac1e822c6ff4942a3f1692` removed the call-only
  first-iteration CC and local-search pass wrappers. Their functions now expose
  five and six cohesive owners directly.
- Focused pass, resident-reference and local-controller tests: 70 passed, with
  28 CUDA-only skips. CPU EM fast guard: 102 passed in 57.04 seconds.
- Production totals are 15,043 physical and 13,119 nonblank lines. Dense/local
  half scoring and initial-grid planning remain in the corrective audit.

## 2026-09-28 — owner-visible signature correction started

The earlier packages over-corrected long parameter lists by introducing
single aggregate `*Spec` call bundles. Although their numerical bodies retained
cohesive subowners, the public helper signatures hid those dependencies. The
plan now requires a small set of owner arguments to be visible at each
boundary, and permits a one-object signature only when that object has a real
lifecycle beyond forwarding one call.

The corrective audit covers initial-grid planning, first-iteration CC, dense
and exact-local half scoring, one local-search pass, mean reconstruction,
direction-prior updates, unregularized means and iteration snapshots. The
aggregate wrappers will be removed; their existing data/state/policy owner
types remain the unit of ownership. Numerical statements, JAX boundaries and
call order are not changed by this correction.

## 2026-09-28 — periodic rebase onto `5c965cf2`

- `origin/main` advanced from `e0f50017` to `5c965cf2`; all 26 feature commits
  were replayed.
- Conflicts occurred in `relax/refinement/local_search_iteration.py`,
  `tests/unit/test_refine_relion_mode.py` and
  `tests/unit/test_resident_local_pass2.py`.
- Resolution prioritized upstream behavior: the fine-pass fallback to the
  exact-local implementation remains deleted, score-only parent evaluation
  remains exact, fine scoring remains resident, and unsupported configurations
  continue to fail. Tests removed upstream were not restored. Surviving callers
  and tests were migrated to the explicit owner APIs.
- Post-rebase focused checks passed after migrating three surviving local-pass
  tests. The fresh CPU EM fast guard passed all 102 cases.

## 2026-09-28 — owner-visible boundary audit completed

### Change

- Removed the remaining call-only `InitialCoarseGridSpec`,
  `DenseHalfScoringSpec` and `LocalHalfScoringSpec` aggregates.
- Initial-grid construction now exposes sampling, sealed replay state and the
  logger. Dense scoring exposes data, sampling, priors, batching, variant,
  execution and optics owners; exact-local scoring exposes the analogous seven
  owners with diagnostics in place of the dense variant policy.
- Shape-class derivation returns those same owner groups instead of rebuilding
  an all-input specification. `DenseHalfScoringPlan` remains because it has a
  real lifecycle: both half plans are materialized before execution to preserve
  the overlap seam.
- The complete corrective audit now covers all nine boundaries listed in the
  plan. No production function introduced by the refactor accepts a single
  call-only aggregate, and none immediately unpacks owner fields into aliases.

### Structural delta from `origin/main` at `5c965cf2`

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Production files | 15 | 16 | +1 |
| Physical production lines | 14,222 | 15,066 | +844 |
| Nonblank, non-comment lines | 12,424 | 13,126 | +702 |
| Production functions | 224 | 226 | +2 |
| Functions with at least 10 parameters | 17 | 7 | -10 |
| Functions with at least 20 parameters | 8 | 2 | -6 |
| Largest parameter count | 67 | 22 | -45 |
| Largest function span | 5,271 | 5,419 | +148 |

The only 20+ argument functions are the two tomography scoring boundaries,
which are outside the current single-volume scope. The largest remaining
single-volume boundary has 17 parameters.

### Validation

- Initial-grid implementation commit: `8d1ff202153759e5a6ffb896004ec2925d14d85b`;
  eight focused tests passed.
- Half-scoring implementation commit: `5781475`; owner-construction readability
  commit: `79ef49a`.
- Scoring, adaptive ownership, diagnostic scope, optics-shape, symmetry and
  loop behavior suite: 555 passed. Five additional cases could not start
  because the pinned RELION fixture
  `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0/run_class001.mrc`
  is absent. The nine directly affected loop/source cases passed separately.
- Production and affected-test Ruff lint, Python compilation, diff checks and
  the refreshed metrics snapshot check passed.
- CPU EM fast guard: 102 passed in 57.02 seconds with
  `FFTW/3.3.10-GCC-12.2.0` loaded.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-28 — owner-visible low-resolution half join

- Commit `c913f37` replaced the 13-argument pre-Wiener half-join boundary with
  three explicit owners: `HalfAccumulatorPair`, `HalfJoinGeometry` and
  `HalfJoinPolicy`.
- The implementation reads the owners at their points of use. It does not
  flatten their fields into local aliases, and the two numbered/final callers
  show the accumulator, geometry and policy construction directly.
- Numerical argument order, previous-resolution selection, input preservation
  and retained-device-numerator behavior are unchanged.
- Focused ownership, strict numerical comparison and resolution-scheduling
  tests: 37 passed.
- CPU EM fast guard: 102 passed in 57.22 seconds with
  `FFTW/3.3.10-GCC-12.2.0` loaded.
- Production totals are 15,105 physical and 13,156 nonblank lines. Functions
  with at least 10 parameters fell from 17 on `origin/main` to 6; excluding the
  two out-of-scope tomography boundaries, four remain.

## 2026-09-28 — owner-visible posterior-noise update

- Commit `f4c3c0b` replaced the 10-argument posterior-noise boundary with
  `NoisePosteriorState`, `NoiseUpdateContext` and the explicit optional debug
  hook.
- The state owner retains the existing shallow ownership of the mutable
  per-half variance list. K1 updates still replace its entries, K-class still
  returns a new shared pair, and first-iteration CC still returns the original
  state objects.
- Focused K1, per-optics-group, K-class and controller tests: 15 passed, with
  368 unrelated cases deselected. CPU EM fast guard: 102 passed in 57.11
  seconds with `FFTW/3.3.10-GCC-12.2.0` loaded.
- Production totals are 15,136 physical and 13,181 nonblank lines. Functions
  with at least 10 parameters fell to 5; excluding the two tomography scoring
  boundaries, `_reconstruct_volume_eager`, nested `_safe_batch_sizes` and the
  run-file reader constructor remain.

## 2026-09-29 — posterior-noise readability correction

- Commit `036942f` supersedes the call-only `NoisePosteriorState` and
  `NoiseUpdateContext` design from `f4c3c0b`. The helper now exposes the four
  noise-state values, image shape, two algorithm choices and diagnostic callback
  directly.
- Iteration number and current size are diagnostic metadata, not numerical
  dependencies. The controller now binds them to the dump callback before the
  call; the numerical helper neither receives nor hides them in a context.
- This correction leaves the tracked large-argument count unchanged at 5. It
  was accepted because the signature and call site are easier to read, not
  because it improves a metric. Production size fell to 15,110 physical and
  13,162 nonblank lines as a secondary effect.
- Focused K1, optics-group, K-class and controller tests: 15 passed. CPU EM fast
  guard: 102 passed in 57.60 seconds with `FFTW/3.3.10-GCC-12.2.0` loaded.

## 2026-09-29 — container-lifecycle rule and half-join correction

- Plan commit `a8daf3e` makes container lifecycle an acceptance rule. A
  constructor nested in a call and a container assigned to a throwaway local
  before one call are the same design problem. Containers now require real
  storage, reuse, variant derivation, serialization or invariant ownership.
- Implementation commit `e854f8d` supersedes the call-only
  `HalfAccumulatorPair`, `HalfJoinGeometry` and `HalfJoinPolicy` design from
  `c913f37`. The join receives the two natural half sequences and names its
  geometry, resolution and buffer-lifetime dependencies explicitly at both
  call sites.
- The tracked count of functions with at least 10 parameters rises from 5 to 6
  because the truthful 11-parameter join boundary is easier to understand than
  three objects created solely for that call. This is an intentional example of
  readability taking priority over the structural statistics.
- Production totals are 15,069 physical and 13,130 nonblank lines. There are
  226 functions; 6 have at least 10 parameters and the 2 functions with at
  least 20 parameters remain the out-of-scope tomography boundaries.
- Focused argument-forwarding, strict numerical comparison and
  resolution-scheduling tests: 37 passed. Ruff and Python compilation passed.
  CPU EM fast guard: 102 passed in 57.92 seconds with
  `FFTW/3.3.10-GCC-12.2.0` loaded.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-29 — periodic rebase onto `c4f7a4b`

- `origin/main` advanced from `5c965cf2` to `c4f7a4b`; all 40 feature commits
  were replayed.
- One conflict occurred in `relax/refinement/local_search_iteration.py`. Main
  now permits `current_size=None` for a regular full-box resident local pass;
  an older refactor commit tried to restore the superseded `ValueError`. The
  resolution kept main's full-box behavior and comment while retaining the
  refactored owner-based call structure.
- No other conflicts occurred. On the rebased candidate, the 37 focused
  half-join tests passed and the CPU EM fast guard passed all 102 cases in
  59.52 seconds.

## 2026-09-29 — explicit initial-grid and adaptive-pass variants

- Commit `041e325` removed the call-only `InitialGridSampling` object. Startup
  now chooses explicitly between ordinary RELION coarse-grid construction and
  sealed replay-grid materialization. The returned `InitialCoarseGrids` remains
  because it is a real multi-value result consumed after either branch.
- Commit `246f9d5` removed the call-only `AdaptivePassPlan`. The common helper
  now derives only genuinely shared keywords from the existing dense owners;
  route-local batch sizes, Fourier sizes, oversampling order and sparse M-step
  choice are visible beside each adaptive engine call.
- Initial-grid ownership and behavior tests: 8 passed. Adaptive K1/K-class
  routing, symmetry, BPref signature and zero-coarse-geometry tests: 470 passed
  with 9 expected gimbal-lock warnings. Ruff and Python compilation passed.
  The combined CPU EM fast guard passed all 102 cases in 57.97 seconds.
- Production totals are 15,059 physical and 13,120 nonblank lines across 227
  functions. Six functions have at least 10 parameters and the two 20+
  functions remain the out-of-scope tomography boundaries. These statistics
  did not drive either design.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-29 — explicit direction-prior and unregularized-map variants

- Commit `c8c222f` removed four call-only direction-prior input classes and
  split learning into the intentionally distinct K=1 per-half collapse and
  K-class combined-prior algorithms. Grid eligibility remains explicit in the
  controller, and each helper names only the values its algorithm uses.
- Commit `0447c4b` removed three call-only unregularized-reconstruction input
  classes. K=1 half reconstruction, K-class shared reconstruction, K=1 sign
  alignment and K-class sign sharing are now separate operations; the
  controller shows which sequence runs and when diagnostic reconstruction is
  skipped.
- Direction-prior unit tests: 11 passed. The learned-prior controller handoff:
  1 passed with 2 existing complex-cast warnings. Focused reconstruction,
  diagnostic-map and sign tests: 4 passed, 381 deselected, with 6 existing
  complex-cast warnings. Production and focused ownership-test Ruff checks and
  Python compilation passed. The combined CPU EM fast guard passed all 102
  cases in 58.97 seconds.
- Production totals are 14,991 physical and 13,068 nonblank lines across 231
  functions. Six functions have at least 10 parameters; the two 20+ functions
  remain the out-of-scope tomography boundaries. The higher function count is
  the deliberate result of separating distinct algorithms, not a target.
- Snapshot capture still has four call-only shadow inputs. It is deferred to
  the controller-state split because flattening them would create a
  25-argument serializer; the correction needs a real longer-lived checkpoint
  owner rather than another cosmetic signature change.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-29 — explicit regularized-reconstruction phases

- Commit `9856080` removed the five call-only mean-reconstruction input
  classes. The controller now selects the K=1 or K-class reconstruction
  algorithm explicitly, publishes the resulting maps, and invokes a separate
  post-processing phase.
- `ReconstructionSettings` is retained because it has a real lifecycle: it is
  created once before the iteration loop and reused by reconstruction and
  post-processing on every iteration. Inputs are read where they are used
  rather than flattened into aliases at helper entry.
- Accumulator release, retained-device-numerator release, post-processing
  order and K=1 versus K-class reconstruction calls are unchanged.
- Focused reconstruction-ownership, solvent-mask-memory and K=1 lifecycle
  tests: 29 passed. The initial targeted controller subset passed 7 cases with
  395 deselected and 4 existing complex-cast warnings. Production and focused
  test Ruff checks, Python compilation and diff checks passed. CPU EM fast
  guard: 102 passed in 58.61 seconds with `FFTW/3.3.10-GCC-12.2.0` loaded.

## 2026-09-29 — staged iteration-snapshot lifecycle

- Commit `ed48596` removed the four input classes created solely for
  `capture_iteration_snapshot`. `SnapshotCapture` now owns only run-level
  settings, is created once before the loop and is reused for every numbered
  checkpoint.
- One snapshot assembly persists across explicit scalar, map/spectrum, prior
  and particle capture phases before `finish` publishes the serialized
  `IterationSnapshot`. Each phase exposes its actual inputs and retains the
  prior host-copy, dtype and K=1/K-class serialization behavior.
- Run-file and lifecycle tests: 16 passed. Ruff, Python compilation and diff
  checks passed. A first guard invocation without the FFTW module failed four
  C4 loader cases with missing `libfftw3.so.3`; the correctly configured CPU EM
  fast guard passed all 102 cases in 58.63 seconds.
- The completed lifecycle audit leaves 15,009 physical and 13,096 nonblank
  production lines across 237 functions. Seven functions have at least 10
  parameters; the two 20+ functions remain the out-of-scope tomography
  boundaries. The increase reflects explicit phases and the truthful
  10-parameter K-class reconstruction boundary, not a metric target.
- GPU smoke remains infrastructure-blocked by the unchanged absence of the five
  mandatory `/scratch/gpfs` fixture sets. No Slurm job or GPU receipt was
  produced.

## 2026-09-29 — truthful dense-scoring lifecycle

- Commit `9190234` removed `DenseHalfScoringPlan` and its forwarding executor.
  Contrary to its old documentation, both half plans were never materialized
  before execution: each half constructed and consumed its plan immediately
  inside its worker. The controller now calls dense scoring with the seven
  actual owners visible, and retains the sampling owner for the parity manifest.
- Commit `df15c92` removed `DenseHalfScoringOutputs`, which copied the real
  score result into a second object only to be immediately unpacked. The
  controller now publishes score fields directly and shows the adaptive versus
  single-pass pose and coarse-assignment choice in place.
- The 17-parameter eager reconstruction boundary was audited and deliberately
  retained. Its explicit options select independently tested monolithic,
  stable-window and host-staged numerical paths; grouping them would improve a
  statistic without establishing truthful ownership.
- Dense ownership, adaptive-engine, overlap and BPref-scope tests passed 62
  cases after each implementation commit. Seven controller tests that
  intercept the dense boundary also passed, with 8 existing complex-cast
  warnings. Ruff, Python compilation and diff checks passed. CPU EM fast guards
  passed all 102 cases in 58.83 and 58.28 seconds.
- Production totals are 14,935 physical and 13,037 nonblank lines across 235
  functions. Seven functions have at least 10 parameters and the two 20+
  functions remain the out-of-scope tomography boundaries. The unchanged large
  argument count is expected: this package removed false lifecycle objects.
- The smoke-tier invocation at candidate `60089fb` stopped before fixture
  verification or Slurm submission because this host cannot create the required
  `/scratch/gpfs/CRYOEM/.../relax_test_tiers` run root (`PermissionError` at
  `/scratch`). No GPU tests ran and no receipt or job ID was produced.

## 2026-09-29 — explicit direct dense variants

- Commit `e7a29b1` extracted the legacy direct K=1 single-pass route as
  `_score_direct_k1_dense`. Its four inputs name the half data, sampling,
  execution controls and already-resolved engine keywords; unsupported optics,
  keyword removal, engine call and result construction retain their original
  order.
- Commit `cbfec1e` extracted the distinct direct K-class fallback as
  `_score_direct_kclass_dense`. It keeps K-class priors and best-pose policy
  explicit while the dispatcher continues to own common K-class result
  collapse and publication.
- Direct K=1 structural and controller tests passed 2 cases with 390 unrelated
  cases deselected and 2 existing complex-cast warnings. Direct K-class
  structural, prior, best-pose and single-class tests passed 4 cases with 67
  deselected. Ruff, Python compilation and diff checks passed. CPU EM fast
  guards passed all 102 cases in 59.00 and 58.78 seconds.
- Production totals are 14,968 physical and 13,064 nonblank lines across 237
  functions. Seven functions have at least 10 parameters; the two 20+
  functions remain the out-of-scope tomography boundaries. The extra lines and
  functions make intentionally different algorithms visible; they are not a
  metric improvement.

## 2026-09-29 — periodic rebase onto `a021081`

- `origin/main` advanced from `c4f7a4b` to `a021081`; all 57 feature commits
  were replayed.
- Conflicts occurred only in `relax/refinement/iteration_loop.py` and
  `relax/refinement/iteration_snapshot.py`, during three historical snapshot
  commits. Main added per-class `MlModel::acc_rot` and `acc_trans` checkpoint
  state while the branch changed snapshot ownership.
- The resolution retained main's initialization, expected-accuracy updates,
  resume behavior, float64 host copies and serialized
  `acc_rot_per_class`/`acc_trans_per_class_angstrom` fields. The final staged
  design captures those values in `SnapshotCapture.begin` before its existing
  map, prior and particle phases. Obsolete intermediate snapshot specifications
  were not restored.
- No scoring conflicts occurred. The rebased direct K=1 and K-class route
  extractions are commits `b750eab` and `8c0eb03`.
- Ruff and Python compilation passed. The complete run-file suite passed 16
  cases; 8 focused snapshot and direct-route cases passed with 452 deselected
  and 2 existing complex-cast warnings. CPU EM fast guard: 102 passed in 58.96
  seconds with `FFTW/3.3.10-GCC-12.2.0` loaded.
- Production totals after incorporating main are 15,008 physical and 13,099
  nonblank lines across 237 functions. Seven functions have at least 10
  parameters and the two 20+ functions remain out-of-scope tomography
  boundaries.

## 2026-09-29 — explicit adaptive K-class route

- Commit `55ee5c9` extracted ordinary adaptive K-class grid planning, batch
  sizing, sparse/dense backend selection and engine invocation into
  `_score_adaptive_kclass_dense`.
- The helper receives the existing half, sampling, prior, batching, variant and
  execution owners plus the already-resolved engine keywords and symmetry. It
  returns the unchanged engine result and the exact pass-2 grid object; the
  dispatcher still performs parent-map collapse and publishes the common
  K-class result. First-iteration CC remains a visibly separate route.
- Focused structural, final all-data, finite-output and significant-count tests
  passed 4 cases with 390 deselected and 2 expected gimbal-lock warnings.
  Broader K-class semantics, adaptive-owner and first-iteration batching suites
  passed 76 cases with 9 expected gimbal-lock warnings. Ruff, Python
  compilation and diff checks passed. CPU EM fast guard: 102 passed in 59.34
  seconds with `FFTW/3.3.10-GCC-12.2.0` loaded.
- Production totals are 15,032 physical and 13,128 nonblank lines across 238
  functions. Seven functions have at least 10 parameters and the two 20+
  functions remain out-of-scope tomography boundaries. The additional helper
  makes an intentional algorithm variant visible; argument statistics are
  unchanged.

## 2026-09-29 — explicit adaptive K=1 route and diagnostic ownership

- Commit `6caf34d` moved the dense pass-2 float64 environment lookup from the
  scorer to the two iteration-controller construction sites. The resolved
  boolean is carried by `DenseExecutionPolicy`; a structural test prevents the
  environment selector from returning to dense scoring.
- Commit `4495c6f` extracted ordinary adaptive K=1 grid construction, backend
  selection, optics scaling and engine invocation into
  `_score_adaptive_k1_dense`. Its explicit inputs reuse the seven persistent
  scoring owners plus the base engine keywords and normalized symmetry. It
  returns the existing engine result and trial grid, both of which remain live
  in the dispatcher; no call-only specification or result wrapper was added.
  First-iteration CC and common K=1 result publication remain visibly separate.
- Ruff, Python compilation and 13 focused route cases passed. The broader
  adaptive-route, optics and symmetry suites passed 402 cases. CPU EM fast
  guard passed 102 cases in 59.00 seconds. Binding-dependent commands initially
  failed to load `libfftw3.so.3`; their unchanged reruns passed with
  `FFTW/3.3.10-GCC-12.2.0` loaded.
- Production totals are 15,087 physical and 13,178 nonblank lines across 239
  functions. The dense one-shape dispatcher fell from 477 to 394 lines in this
  package. Seven functions still have at least 10 parameters and the two 20+
  functions remain out-of-scope tomography boundaries; these counts are
  descriptive evidence, not the design target.

## 2026-09-29 — rebase onto current main after adaptive K=1

- Fetched and rebased all 63 branch commits from `a021081` onto
  `origin/main` `836a37f`. There were no conflicts and therefore no manual
  resolutions. Main's intervening commit refreshed the frozen real-data
  scorecard Markdown only; refinement production metrics were unchanged.
- The rebase rewrote the diagnostic-ownership and adaptive K=1 commits to
  `6caf34d` and `4495c6f`. Post-rebase Ruff and Python compilation passed, as
  did 13 focused adaptive K=1 cases and the 102-case CPU EM fast guard in
  59.27 seconds with `FFTW/3.3.10-GCC-12.2.0` loaded.

## 2026-09-29 — exact-local override ownership

- Commit `423e33b` moved local pass-1/pass-2 precision and the full-parent,
  rotation-only and denominator-support experimental choices out of
  `_score_half_local_one_shape`. They are explicit fields of the existing
  `LocalDiagnosticPolicy`; no additional request or call-only bundle was
  introduced.
- Commit `8929943` tightened the placement after review: each choice is now
  resolved once per numbered iteration and once per final all-data iteration,
  then shared by both halves and all shape classes. Support overrides retain
  their original condition and are not parsed when parent oversampling is off.
- Ruff and Python compilation passed. Sixteen focused local-policy cases and
  41 broader controller/result-contract cases passed; the placement follow-up
  passed 5 selected cases. CPU EM fast guard passed 102 cases in 58.70 seconds
  with `FFTW/3.3.10-GCC-12.2.0` loaded.
- Production totals are 15,146 physical and 13,237 nonblank lines across 239
  functions. The exact-local one-shape scorer fell from 549 to 543 lines while
  the top-level controller grew to make ownership explicit. Seven functions
  still have at least 10 parameters. These statistics record the structural
  movement; they did not drive the design.
