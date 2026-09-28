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
