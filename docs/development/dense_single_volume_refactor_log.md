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
  `f628d5aa2b2352f32f058a5517d95f54f6af1548`.
- Dense routing, multi-shape, firstiter-CC, BPref-scope, zero-coarse and
  K-class focused set: 149 passed.
- Numbered/final controller handoff and tau2/scoring-prior probes: 18 passed,
  365 deselected.
- Complete symmetry unit suite: 360 passed.
- Production Ruff lint, Python compilation and diff checks: passed. The edited
  `test_refine_relion_mode.py` retains unrelated pre-existing import-order
  findings; it was not broadly reformatted.
- CPU EM fast guard: 102 passed in 57.80 seconds on committed source
  `a2a1b6eaa0e2d21b48c3c228296d27e6a663aa48` after loading
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
  `8be72f334e4a7c6dcd6c0715a22da0f83e86c1f6`.
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
  `8be72f334e4a7c6dcd6c0715a22da0f83e86c1f6`; exit 0. The host has no CUDA
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
  `f4237f0f530e1f30d60f12e1354f5dd6095f2104`.
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
  `e75a07fd356876a209011204d802758ab4c1a635`.
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
| Physical production lines | 14,185 | 15,025 | +840 |
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
  `7f91db404d02100bba4701f6d32e252088908ec7`.
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
