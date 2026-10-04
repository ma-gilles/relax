# RELAX agent guide

RELAX reimplements RELION's refinement in JAX with CUDA kernels: `relax refine` (Refine3D, and subtomogram
averaging on RELION 5 particles), `relax class3d`, `relax initial_model` (VDAM), `relax ppca_initial_model`.
It imports RECOVAR at one commit pinned in `pixi.toml`; RECOVAR never imports RELAX.
Priorities, in order: correctness, GPU performance, clarity.

## Start

- Run `pixi run doctor` (under any Python: `python scripts/dev/doctor.py`). Fix each `FAIL` with the command
  printed under it; a `WARN` names something a task may need.
- Work in your own worktree and branch. Never modify the source snapshot of a queued or running job; the tier
  commands freeze their own copy.
- Before editing a file, read the `AGENTS.md` in its directory and in each directory above it. They exist in
  `relax/`, `relax/refinement/`, `relax/sparse_pass2/`, `relax/ppca_refinement/`, `tests/`, `scripts/`, `docs/`.
- Accounts, writable roots, local GPUs, Slurm requests, data layout and cleanup are governed by the owner's
  global instruction file, which the tool loads at session start. This repository's guides do not repeat it.
- The owner's explicit instruction outranks every guide and skill. If a rule here blocks the task, quote the
  rule and its file, and ask.

## Rules

1. Production EM is float32. Double precision is a labelled diagnostic, never a default and never the fix.
2. The default path reproduces RELION 5.0.1, including its inconsistencies. An intentional difference is a
   named opt-in option, tested and qualified.
3. Do not change a tolerance, gate, threshold, baseline or pinned output without the owner's instruction. This
   covers `tests/tiers/fsc_thresholds.json`, `tests/tiers/pinned_fast_cases.json`, the pinned references in
   `tests/baselines/`, every tolerance in a test, and the `pixi run regen-*` commands.
4. No test or merge check requires bitwise or ULP-exact float equality; compare with
   `tests/helpers/float_compare.py`. Integers and discrete logic stay exact.
5. No fallback, skip or default that hides a failure. A missing measurement is not a pass.
6. Change RECOVAR in RECOVAR (`dev2`), then repin here. Never copy, wrap or keep a variant of its code
   (CONTRIBUTING.md, "Changing RECOVAR").
7. Delete, without asking, the code that `docs/development/refactor_principles.md` covers: unreachable arms,
   options nothing in production sets, test-only cases, redundant checks, unused temporaries. One decision per
   commit, with its trace (the entry paths checked, what was removed) in the message.
8. Ask the owner about a choice of taste or of capability, one question at a time. Continue the work that does
   not depend on the answer.
9. When two modes share a sequence, share the steps and write the sequence in each mode. Pass no mode flag
   below the point where the mode is decided.
10. A structural change preserves casts, reduction order, JIT boundaries, required buffer lifetimes, serialized
    formats and scientific defaults. Commit correctness, performance and structure separately.
11. Keep diffs focused: do not reformat untouched code; commit no data, binaries, run outputs or credentials.
12. Merge bug fixes and qualified work to `main` yourself, often: rebase onto `origin/main`, pass the tier for
    the change on the rebased commit, push as a fast-forward. Never force-push (CONTRIBUTING.md, "Landing").
13. Delegate to sub-agents only when the owner has allowed it; one writer per file (`relax/SUBAGENTS.md`).

## Commands

| Need | Command |
| --- | --- |
| Environment, natives, fixtures, write access, idle GPUs | `pixi run doctor` |
| CPU guard: undefined names, import boundaries, four fast test files | `pixi run test-em-fast-guard` |
| One test file (CPU unless `CUDA_VISIBLE_DEVICES` names an idle GPU 1-3) | `pixi run python -m pytest -v tests/unit/<file>.py` |
| Guides and the file links of every tracked Markdown file | `python scripts/check_agent_guides.py` |
| What a tier would run for this checkout | `python scripts/run_test_tier.py plan smoke` |
| Qualify | `pixi run test-smoke`, `pixi run test-medium`, `pixi run test-long` |
| Is a control-candidate difference inside GPU noise | `python scripts/em_tier_noise_envelope.py check --control <basetemp> --candidate <basetemp>` |

## Which tier

| Change | Tier |
| --- | --- |
| docs, tests, scripts | CPU checks: `pixi run test-em-fast-guard`, the affected unit tests, `python scripts/check_agent_guides.py` |
| engine or numerical code | `pixi run test-smoke` |
| a numerical change | `pixi run test-medium` |
| a default flip, an engine replacement, a milestone | `pixi run test-long` |

Each tier command freezes the checkout, verifies fixtures, builds the natives, runs as one Slurm job and writes
`RECEIPT.json` in its run root. Only smoke may run on a local GPU. When the default run root is not writable
for your account, pass `--run-root <directory>`. Budgets, contents and pass criteria: CONTRIBUTING.md.

## Map

| Path | Owns |
| --- | --- |
| `relax/commands/`, `relax/command_line.py` | the `relax <command>` entry points |
| `relax/refinement/` | Refine3D and Class3D: options, controller, expectation, reconstruction, finalization, run files |
| `relax/scoring/` | coarse pass: scores, significance, candidate layouts |
| `relax/sparse_pass2/` | fine pass: the device-resident pass-2 engine (CUDA only) |
| `relax/local/`, `relax/classification/` | exact local-search layout and backprojection; K-class execution and results |
| `relax/dense/` | score payloads and scoring policy; the dense GEMM coarse pass and experiment |
| `relax/reconstruction/`, `relax/relion/` | RELION M-step, regularization and noise; RELION metadata, CTF, projector, normalization |
| `relax/vdam/`, `relax/ppca_initial_model/`, `relax/sgd_initial_model/` | InitialModel variants |
| `relax/ppca_refinement/` | pose-marginal PPCA refinement |
| `relax/helpers/`, `relax/sampling.py`, `relax/healpix_sampling.py`, `relax/symmetry.py` | shared layouts, planning, grids |
| `relax/cuda/` | CUDA kernels and their FFI (`librelax_cuda.so`) |
| `relax/diagnostics/`, `relax/reference/`, `relax/relion_bind/` | capture and replay; independent references; the RELION binding (never imported by production code) |
| `tests/tiers/`, `tests/baselines/`, `tests/fixtures/` | gates and pinned outputs; pinned references and the benchmark ledger; the fixture manifest |
| `scripts/` | the tier runner, `dev/doctor.py`, launchers, scorecard renderers, analysis |

## Where to look

- Algorithm to code, step by step: `docs/math/relion_refinement_algorithm.md`.
- Workflow entry points: `docs/development/codebase.md`. RELION defaults: `docs/development/relion_defaults.md`.
- Refactor rules with their examples: `docs/development/refactor_principles.md`.
- Conventions shared with RECOVAR (numerical source, CUDA and FFI, documentation): RECOVAR's own guides,
  `recovar/CLAUDE.md`, `recovar/cuda/CLAUDE.md` and `docs/CLAUDE.md` in its repository.
- State of the work: `docs/development/em_status.md`. It is a ledger of more than 100 KB: search it for your topic.
- Polar as a Slurm target: `docs/development/polar_agents.md`.
- Planned, not present yet: a per-change check command, a behaviour-fingerprint harness, an evidence store, a
  short status page, project skills.

## Reporting

State the change and its reason, the checks with their job IDs, receipts and artifact paths, the outcomes,
what is not yet qualified, and `git status --short --branch`. Separate executed, quality-accepted and
performance-qualified results. For a run outside the tier commands, record `git rev-parse HEAD` and `git diff HEAD | sha256sum`.
Do not claim completion while a required job is pending or a required gate is open.
