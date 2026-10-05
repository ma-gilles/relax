# Contributing to RELAX

The rules, the package map and the tier table are in the [agent guide](AGENTS.md). This file is the reference
behind them: environment, native libraries, the test tiers, landing, commits, and changing RECOVAR.

## Environment

```bash
pixi install --frozen        # the default environment: RECOVAR at the commit pinned in pixi.toml
pixi run doctor              # checks the environment, natives, fixtures, write access and idle GPUs
```

`pixi run doctor` changes nothing and prints the fix under each `WARN` or `FAIL`. A worktree without `.pixi`
can link an environment built from the same `pixi.lock`; doctor prints the command.

Start every Python process from a clean environment, and choose the backend before the first import:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu      # CPU; for a GPU: the assigned device and JAX_PLATFORMS=cuda,cpu
```

`XLA_PYTHON_CLIENT_PREALLOCATE` is left unset for anything that is measured or compared with a user's run
(benchmark arms, the GPU matrix, end-to-end refinements): `relax` does not set it, so users run JAX's
default, one preallocated memory pool. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` only where several
processes must share one GPU: the test tiers and unit tests do (`scripts/run_test_tier.py`,
`scripts/run_em_fast_guard.sh`), because a pytest process that holds the GPU backend also starts `relax`
commands as subprocesses on the same card. Without preallocation the pool grows in separate regions and a
long large-box refinement can fail on one contiguous allocation (relax#20), so it is not a setting for
benchmarks.

Never reuse a process that already initialized the wrong backend. Give each run its own
`JAX_COMPILATION_CACHE_DIR` and `RECOVAR_JAX_CACHE_DIR`, and record whether they began empty.

## Native libraries

```bash
pixi run build-cuda                                         # librelax_cuda.so
RELION_SRC_DIR=/path/to/relion/src pixi run build-relion-bind   # the RELION 5.0.1 binding (tests and diagnostics)
```

- Load a CUDA toolkit module (`module load cudatoolkit/12.8`) only for a `make` step, in its own shell. With
  the module loaded, the pixi environment's JAX finds no CUDA-enabled jaxlib and falls back to the CPU.
- `RELAX_CUDA_LIB` and `RECOVAR_CUDA_LIB` name an explicit library. The loaders never build or rewrite an
  explicit path: they load it when it exists and exports the required symbols, and stop otherwise.
- An unpinned cache build is rebuilt only when the sha256 of its sources differs from
  `<library>.sources.sha256`. Never rebuild a library another process is using.
- For qualification, record the loaded path and the library hash. A successful import does not prove that the
  intended library ran.
- The tiers build their own natives (`scripts/build_test_natives.sh`) and record a digest of the native
  sources in `NATIVE.json`. A tool that reuses prebuilt natives runs
  `python scripts/native_sources.py check <natives_dir> --root <checkout>` first, and
  `python scripts/native_sources.py imports <snapshot>` to confirm that `relax` imports from the frozen source.

## Validation during development

Run the smallest meaningful check first. Use focused tests between edits, then the tier for the change.

| Scope | First check | Then |
| --- | --- | --- |
| helpers, reporting, scripts, documents | the affected unit tests on CPU; `python scripts/check_agent_guides.py` | affected callers |
| controller, scoring, engine | `pixi run test-em-fast-guard`, the unit tests naming the changed module | the [tier](#test-tiers) for the change |

- Some unit tests need the RELION binding on CPU. Without it they skip with a reason; under
  `--require-natives` the session stops instead. A skip is not a pass for qualification.
- Record selected and executed counts, skips and the exit status. Keep full logs.
- Report CPU-only, native-oracle, GPU and scientific-parity results separately.

## Test tiers

| Tier | Command | Budget | Runs as | Contents |
| --- | --- | --- | --- | --- |
| smoke | `pixi run test-smoke` | 5 min GPU | one 1-GPU cryoem job when it starts within 15 min, else one idle local GPU 1-3 (or the session's `RELAX_LOCAL_GPUS` subset) | the CPU fast guard and the merge-guard unit files; the fixed-state replays `k1_local_replay`, `k1_adaptive_replay`, `kclass_replay`; the GPU unit files for the changed paths that fit the budget by their recorded walls (the runner prints the deferred ones) |
| medium | `pixi run test-medium` | 1-2 h wall | one Slurm job, 3 GPUs | smoke's CPU items; the whole fast parity tier; the GPU unit sweep over `tests/unit`, `tests/integration`, `tests/ppca_abinitio`; native VDAM K=1 50k/256; a K=1 5k/128 auto-refine to convergence, scored by FSC against a RELION band |
| long | `pixi run test-long` | about 6 h on H100, 7 h on A100 | one Slurm job, 4 GPUs | the EM long tier (K=1 50k/256 standalone and seeded, native VDAM, K=4 50k/256) and the K=1 and K=4 100k/256 completions, against RELION repeat bands |
| regeneration | `pixi run regen-fixture-manifest`, `pixi run regen-pinned-fast` | - | - | the fixture manifest and the pinned outputs; only on the owner's instruction; each pinned entry records its commit, job, GPU model and date |

Options of `python scripts/run_test_tier.py submit <tier>`: `--run-root <dir>`, `--base <ref>` (default
`origin/main`; selects smoke's GPU files), `--where auto|slurm|local` (smoke), `--queue-wait-minutes`,
`--gpu-model any|a100|h100`, `--dry-run`, `--coarse-engine`. `plan <tier>` prints the items without running.

Every tier command freezes the checkout (HEAD and any uncommitted diff) into its run root, verifies the
fixture sets, builds the natives once and ends with `RECEIPT.json` (`scripts/write_test_receipt.py`: SHA,
tier, pass or fail, job, GPU model). Set `RELAX_TEST_RECEIPTS=<file>` to have each receipt appended there, and
cite the receipt in your report. The runner refuses to start when the natives were built from other sources
or when `relax` does not import from the frozen source.

**Pass criteria.** A tier passes when every item exits 0 and no parity or end-to-end case skipped. Quality is
judged by FSC: every fast parity case reaches its FSC-AUC floor and its minimum in-band shell FSC floor
against its RELION oracle, for every half or matched class, and keeps the mean per-particle |ΔPmax| under its
bound. The K=1 5k end-to-end run converges at RELION's iteration, records `final_all_data_grid_correct`, and
for both the unfiltered half-map average and the merged map stays within 0.002 of the RELION repeat band's
ground-truth FSC-AUC and reaches cross FSC-AUC 0.990 against every RELION run. The values are in
`tests/tiers/fsc_thresholds.json` (floor = 1 - 2 x max(RELAX deficit, RELION repeat deficit)); scoring is
`scripts/em_tier_fsc.py`. Each run also compares its `fsc.json` with the pinned outputs
(`tests/tiers/pinned_fast_cases.json`, `scripts/em_tier_pinned.py`). Map correlation is a diagnostic only.

**GPU model.** Pinned outputs are kept per GPU model and a run is compared only with the entry of its own
model. Anything that compares a control with a candidate numerically pins the same model on both
(`--gpu-model a100|h100`).

**GPU noise envelope.** Two runs of the same code on the same GPU model differ: racing reductions move the
printed metrics by about 1e-13 to 1e-6. `tests/tiers/gpu_noise_envelope.json` records, per fast-tier case and
metric, the largest same-code difference observed and a check limit of max(10 x that, 1e-9).
`python scripts/em_tier_noise_envelope.py check --control <basetemp> --candidate <basetemp>` reports each
difference against it. A difference inside needs no rerun; one outside is a real change, which the gates
judge. The envelope is measured on H100; across models the spread is larger.

**Fixtures.** The repository holds metrics, thresholds, pinned summaries and
`tests/fixtures/em_fixture_manifest.json` (per set: root, sizes, sha256, RELION command, version, seed). Bulk
data stays outside the repository at the roots the manifest names. Tests resolve it through
`tests/helpers/em_fixtures.py` and fail, not skip, when data is missing or a checksum differs.
`python tests/helpers/em_fixtures.py <set> ...` verifies sets before a submission.

**Measured long-tier walls** (one node, shared; not timing-controlled). `LONG_ARM_SECONDS` in the runner holds
the H100 values.

| Arm | H100 (job 14410744, 93ce1aa) | A100 (job 14375361, 319cd10) |
| --- | --- | --- |
| whole tier | 366 min | 425 min |
| K=4 100k/256 completion | 354 min | 413 min |
| K=1 100k/256 completion | 154 min | 197 min |
| K=4 50k/256 Class3D | 184 min | 91 min |
| K=1 50k/256 standalone | 105 min | 172 min |
| K=1 50k/256 RELION-seeded | 75 min | 201 min |
| EMPIAR-10097 hp3 replay (default / resident) | 31 / 11 min | - |
| native VDAM 50k | 4 min | 4 min |

## Landing

RELAX `main` is the integration branch. Agents merge to it themselves.

1. Rebase the branch onto current `origin/main`.
2. Run the tier for the change on the rebased commit ([which tier](AGENTS.md#which-tier)).
3. Push to `main` as a fast-forward. Never force-push.
4. Delete the branch, local and remote, unless the owner asked to keep it. If a branch is abandoned, say why
   in your report and delete it.

- Land each qualified piece when its tier passes, not at the end of the task. List your unmerged commits
  (SHA, subject, what blocks each) in your report.
- A rebase makes a new candidate that needs its tier again, with one exception: when every commit it moved
  over changes only documents, tests or scripts (nothing under `relax/`, no native source, no pixi manifest or
  lock), the CPU checks at the rebased head suffice.
- Validate forward: when the candidate passed its GPU tier at the previous head, the rebase is conflict-free,
  and main's new code commits touch neither the candidate's files nor its engine path, push after the CPU
  checks at the rebased head, run the GPU tier afterwards, and fix forward if it fails.
- Judge a candidate against its control by the gates in `tests/tiers/fsc_thresholds.json` and the noise
  envelope, never by bitwise equality.
- When the owner asks for a pull request instead, start its description with the problem and the resulting
  behaviour, then the evidence. Do not present replay agreement as an autonomous trajectory, or an average as
  per-class quality ([benchmark contracts](docs/development/benchmarks.md)).

## Commits

- Subject: `type(scope): a sentence that states the behaviour after the change`, with a type the history uses
  (`fix`, `perf`, `feat`, `refactor`, `test`, `docs`). Example: `refactor(noise): the shells-to-pixel-row
  expansion exists once`.
- Body: what changed and why, with its evidence (the trace of a deletion, job IDs, receipts).
- No narration of how the change was reached.

## Changing RECOVAR

RELAX imports RECOVAR at the commit pinned in `pixi.toml` (`[feature.pinned.pypi-dependencies]`). When RELAX
work needs RECOVAR to change, change RECOVAR; it must keep working without RELAX and must never import it.

1. Clone RECOVAR beside this checkout and use the `dev` environment, which installs `../recovar` editable:
   `pixi install -e dev`, then `pixi run -e dev python -c "import recovar; print(recovar.__file__)"`.
2. Branch from RECOVAR `origin/dev2`, commit the change with its focused tests, and merge it to `dev2` as a
   fast-forward; RECOVAR `dev2` takes direct merges. Do not push to RECOVAR `main` without the owner's word
   and its long test.
3. Keep RECOVAR's defaults and public API unchanged unless the owner decides otherwise.
4. Repin here: set the new commit in `pixi.toml`, regenerate `pixi.lock`, and land the RELAX side on `main`
   in the same batch.

## Code style

Ruff uses the 120-character line limit in `pyproject.toml`. Check the Python files you changed, passed
explicitly: `pixi run python -m ruff check <files>`. Do not format files you did not change.

## Instruction files

`AGENTS.md` is the single source in each directory that has a guide; `CLAUDE.md` beside it is a symbolic link
to it. Edit `AGENTS.md`. A rule lives in one guide; another guide refers to it. Run
`python scripts/check_agent_guides.py` after editing any of them.
