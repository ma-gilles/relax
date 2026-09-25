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

## 2026-09-24 — initial coarse-grid request boundary

### Objective

Start work package 1 with one isolated host-side planning boundary, replacing
the first 10-parameter controller helper without changing grid construction.

### Change

- Added `refinement.iteration_planning.InitialCoarseGridRequest` and
  `InitialCoarseGrids`, both frozen shallow containers.
- Moved initial sealed/canonical coarse-grid construction from the controller
  to `build_initial_coarse_grids`.
- The controller constructs one request and consumes the returned grid fields
  in the same order. No casts, grid calls or array conversions changed.
- Migrated the focused ownership tests to the new owner.

### Structural delta

| Metric | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| Physical production lines | 10,397 | 10,434 | +37 |
| Nonblank, non-comment lines | 9,232 | 9,261 | +29 |
| Functions with at least 10 parameters | 13 | 12 | -1 |
| Largest function span | 4,753 | 4,755 | +2 |

The line increase is request/result documentation, explicit field names and
module ownership; the moved calculation itself is unchanged.

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
