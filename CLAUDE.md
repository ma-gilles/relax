# relax development contract

relax is RELION in JAX. It imports RECOVAR (pinned in `pyproject.toml` and `pixi.toml`)
for the shared numerical core; RECOVAR never imports relax.
Engineering priorities are correctness, GPU performance, then clarity.

## Agent collaboration

- Treat requests for implementation as authorization to complete the work.
  Resolve routine choices from context and continue independent work while
  material questions remain open. Prepare a reviewable result before requesting
  any still-required approval for publication or an external action.
- Explicit user instructions take precedence over skill guidance. Apply scoped
  repository requirements to the affected workflow. If an instruction blocks
  progress, cite its file and exact rule, explaining the unresolved decision.
- Incorporate corrections and answer side questions while retaining the active
  objective, completed work and running jobs. Use a concise handoff when moving
  to a fresh thread; link evidence instead of repeating experiment histories.
- Delegate only when authorized; for EM, follow `relax/SUBAGENTS.md`.
  Preserve the user's Terra/Astra workload choices and exclusive source ownership.
- Report outcomes and limitations in plain, concise prose. Use tables for
  comparisons and keep required evidence in linked artifacts.
- Complete the checks required for the affected scope. Repeat or broaden them
  only for changed behavior, failures, unresolved concerns or required qualification.
  Documentation/instruction edits use mirror and link checks; numerical fixes
  still require focused regressions and the applicable scientific ladder.

See [the agent workflow](docs/development/agent_workflow.md) for session-specific
model/delegation evidence and compact task handoffs.

## Changing recovar from relax work

relax imports recovar and pins one recovar commit (`pyproject.toml`, `pixi.toml`).
When relax work needs recovar to change (a new optional argument, a fix, a hook or a
shared helper), make that change in recovar itself. This is permitted and encouraged.
Do not copy recovar code into relax with a twist, wrap it to patch its behaviour, or
keep a second variant of a recovar formula: one implementation, in recovar.

1. Branch from recovar `origin/dev2`, commit the change with its focused tests, and
   push it to recovar `dev2` as a fast-forward (never force; never `dev` or `main`).
   Keep recovar's own defaults and public API unchanged unless the user decides
   otherwise; recovar must keep working without relax, and must never import relax.
2. Repin relax to the new `dev2` commit in `pyproject.toml` and `pixi.toml`,
   regenerate `pixi.lock`, and land the relax side on relax `main` in the same batch.

## Start and resume

1. Establish the task, checkout, branch and current evidence before editing.
   User instructions and authorization persist across turns. Do not ask again
   for work already covered by the task.
2. Read the applicable scoped guides below. Historical experiment notes are
   evidence for their recorded source; their old next actions are not current
   instructions. Use the [codebase map](docs/development/codebase.md) to locate
   the workflow entry point and the modules that own its state and kernels.
3. Print `git rev-parse HEAD`, `git status --short --branch`,
   `git diff HEAD --stat`, and `git diff HEAD | sha256sum` before validation.
   Record untracked files used by a run. A worktree name is not provenance.
4. Choose one concrete change and its smallest useful check. Preserve unrelated
   work. Keep control checkouts and queued/running candidates immutable.

Metrics are tracked in standing artifacts, not only in run reports: the EMPIAR
real-data scorecard, the synthetic fixed-suite scorecards, the pinned completion
baselines and the per-run parity ledgers. Read the value your change touches
before running, and write the new value back after. `relax/CLAUDE.md`
lists each artifact with the command that regenerates or checks it, and records
two setup traps that make the parity tier look red for reasons that are not
numerical. End-to-end runs are what fill the real-data rows; short tests do not
substitute for them.

| Affected area | Read before working |
| --- | --- |
| Python source and numerical conventions | [recovar/CLAUDE.md](https://github.com/ma-gilles/recovar/blob/dev2/recovar/CLAUDE.md) (applies to relax source too) |
| Tests, tolerances and baselines | [tests/CLAUDE.md](tests/CLAUDE.md) |
| EM and RELION parity | [relax/AGENTS.md](relax/AGENTS.md) |
| PPCA refinement | [relax/ppca_refinement/AGENTS.md](relax/ppca_refinement/AGENTS.md) |
| CUDA and FFI | [recovar/cuda/CLAUDE.md](https://github.com/ma-gilles/recovar/blob/dev2/recovar/cuda/CLAUDE.md) (applies to `relax/cuda/` too) |
| Documentation | [recovar docs/CLAUDE.md](https://github.com/ma-gilles/recovar/blob/dev2/docs/CLAUDE.md) (same conventions for relax `docs/`) |

## Context and work packages

Use [the agent workflow](docs/development/agent_workflow.md): compact file handoffs,
on-demand history, scripted evidence and one publication per cohesive batch.
Model selection and delegation are session choices, not scientific requirements.
Cost control must not reduce the scientific goal or gates.

## Implement and review

- Prefer small functions with explicit inputs, units, layouts and ownership.
  Use simple containers when they clarify state; avoid forwarding layers.
- Separate correctness repairs, performance changes and structural cleanup.
  Preserve numerical casts, reduction order, JIT boundaries, memory lifetime,
  serialized formats and scientific defaults during cleanup. Preserve non-EM
  public APIs. EM APIs may change when this simplifies the implementation;
  migrate affected callers, tests and documentation in the same change.
- Remove private dead code only after checking callers, dynamic registration,
  CLI entry points, tests, notebooks and serialized/imported names. Keep
  independent numerical references independent of production code.
- For shared-helper extraction, trace imports through dependent modules as
  well as direct call sites. Collect the applicable test inventory before a
  broad run; testing only the edited modules can miss a retired re-export.
- Validate assumptions early. Do not hide errors with fallback results, skipped
  checks or fabricated success. Resolve TODOs with evidence before removing them.
- Keep math documentation linked to implementing functions, and docstrings
  linked back to the documented formulation. Update both when behavior changes.
- Never widen a scientific tolerance or change `tests/baselines/` without an
  explicit user instruction. Missing measurements are not passing comparisons.
- Keep diffs focused. Do not reformat unrelated code or commit large datasets,
  checkpoints, binaries, generated run outputs or credentials.

## Environment and validation

Use the checkout's frozen pixi environment. Before Python imports, select CPU
or assigned GPU visibility and remove Python/conda contamination. Verify
relax imports from this checkout, and RECOVAR (at the pinned commit) and JAX from its
`.pixi/envs/default`.
Explicitly build and identify custom CUDA libraries before GPU qualification;
the current runtime loader can build missing libraries automatically.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for exact setup, validation and PR
requirements, [Della development](docs/development/della.md) for cluster resources
and paper-data paths, and [benchmark contracts](docs/development/benchmarks.md)
for reusable accuracy and performance evidence. Use Slurm for integration,
multi-iteration, long or contention-sensitive GPU work. Reserve local GPUs for
short checks following the user's device policy.

For EM-only work, use the scoped EM validation ladder. Shared pipeline or
repository-wide cleanup requires the applicable SPA/ET and downstream checks
as well. Existing authorization for that scope covers its necessary validation;
a genuinely new scientific objective requires a separate decision.

## Test tiers

Pick the tier from the change, run it with one command, and record its receipt
(SHA, tier, pass/fail, job id, GPU model; the tier commands write it, and append it to
`$RELAX_TEST_RECEIPTS` when set) in your handoff. Budgets, pass criteria and
fixtures are in [CONTRIBUTING.md](CONTRIBUTING.md#test-tiers). This is a written
rule, not a hook.

| Change | Tier |
| --- | --- |
| docs, tests, scripts | CPU checks (`pixi run test-em-fast-guard`, the affected unit tests, `python scripts/check_agent_guides.py`) |
| engine or numerical code | `pixi run test-smoke` |
| a numerical change | `pixi run test-medium` |
| a default flip, an engine replacement, a milestone | `pixi run test-long` |

Every tier runs on Slurm in the cryoem partition, each as one job (smoke 1 GPU,
medium and long one multi-GPU job). Only smoke may run locally: when cryoem cannot
start it promptly, it runs on one idle local GPU 1-3 (never GPU 0). The medium tier also runs
periodically on `main`. Baseline regeneration (`pixi run regen-*`) runs only on
the user's explicit request.

## Branches and delivery

relax `main` is the integration branch. Feature branches are fine for isolation, but
merge each into `main` as a fast-forward as soon as its checks pass; do not park
finished branches. recovar changes follow "Changing recovar from relax work" above.
Branches are fine for isolation, but they are temporary: once a branch is merged, delete it
(local and remote); if it is abandoned, record why in the task handoff and delete it. Do not
leave finished or dead branches behind. Keep only `main`/`dev`/`dev2`, active work and
branches the user explicitly asked to keep.
Preserve an explicitly pinned control.

Merge criterion: judge a candidate against its control by the approved gates in
`tests/tiers/fsc_thresholds.json`, never by bitwise equality. GPU runs of the same code are
not bit-reproducible; racing reductions move printed metrics by about 1e-13 to 1e-6.
`tests/tiers/gpu_noise_envelope.json` records that same-code spread per fast-tier case and
metric: check a control-candidate difference with `python scripts/em_tier_noise_envelope.py
check --control <basetemp> --candidate <basetemp>` instead of rerunning. A difference outside
the envelope is a real change of the numbers, which the gates then judge.

No bitwise floats (user rule, 2026-09-24): no test or merge check requires bitwise or
ULP-exact equality of floating-point values, not even under
`RELAX_EM_DETERMINISTIC_REDUCTIONS=1`. Compare floats with relative bands sized to the
measured noise plus a margin (`tests/helpers/float_compare.py`); integers and discrete logic
(shapes, counts, indices, capacities) stay exact, and float-tie-dependent discrete outputs
(hard assignments, significance counts) allow a small measured flip fraction. This approves
turning bitwise float asserts into measured bands, not widening an existing tolerance beyond
noise.

Merge as you go: land each qualified piece on `main` as soon as its checks pass, not at the
end of the task. Keep a list of your unmerged commits (SHA, subject, what blocks each) in
your handoff. Rebasing an implementation creates a new candidate that needs fresh
validation, except when every commit it moves over changes only docs, tests or scripts (no
code under `relax/`, no native sources, no pixi manifest or lock): then the CPU checks of the
rebased head suffice.

Validate forward: when a candidate passed its GPU validation at the previous head, the rebase
is conflict-free, and main's new code commits touch neither the candidate's files nor its
engine path, push after the CPU checks at the rebased head and run the GPU validation
afterwards, fixing forward if it fails. A fresh GPU run before pushing is required only when
main's new code overlaps the candidate's files or engine path. This keeps a candidate from
chasing a main that moves every hour.
Never force-push unless explicitly asked. Before pushing or opening a PR, follow
all applicable checks and table requirements in CONTRIBUTING.md and scoped guides.

Report the change, its reason, exact checks and job IDs, outcomes and unresolved
limitations, reproduction commands, artifact paths, `git status --short --branch`
and `git diff HEAD --stat`. Distinguish executed, quality-accepted and
performance-qualified results. Do not claim completion with required jobs pending.

Every `AGENTS.md` and `CLAUDE.md` in the same directory must remain byte-for-byte
identical (root, `relax/`, `relax/ppca_refinement/`, `tests/`). Run
`python scripts/check_agent_guides.py` after editing any of them.
