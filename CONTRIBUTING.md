# Contributing to RECOVAR

Make one reviewable change at a time. Establish an unchanged control before
structural or numerical work, and keep baseline failures visible.

## Package map

`core/` provides Fourier, CTF, geometry and forward/adjoint primitives;
`data_io/` loads and indexes particles; `reconstruction/` estimates the mean
and noise; `heterogeneity/` estimates covariance, PCA coordinates and volumes;
`em/` contains refinement and classification; `output/` serializes and analyzes
results; `commands/` orchestrates CLI workflows; `cuda/` supplies optional native
kernels; `simulation/` generates fixtures; `gui_v2/` provides the web interface.

The covariance pipeline runs dataset loading → mean/noise reconstruction →
covariance → PCA → embedding → kernel regression → output. Consult the scoped
source guide for FFT frames, packed layouts and normalization before changing
those boundaries. PPCA latent dimension and classification K are different axes.
The [contributor codebase map](docs/development/codebase.md) identifies the
separate pipeline PPCA, pose-refinement, K-class, VDAM and earlier EM entry
points, along with their state, kernel and diagnostic dependencies.

## Reproducible environment

From the intended checkout, use pixi and the committed lockfile:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
pixi install --frozen
pixi run python - <<'PYCODE'
import json
from pathlib import Path
import jax, recovar, relax
repo = Path.cwd().resolve()
assert Path(relax.__file__).resolve().is_relative_to(repo)
assert Path(jax.__file__).resolve().is_relative_to(repo / '.pixi/envs/default')
direct_url = next(Path(recovar.__file__).resolve().parents[1].glob('recovar-*.dist-info')) / 'direct_url.json'
print(relax.__file__, recovar.__file__, json.loads(direct_url.read_text())['vcs_info']['commit_id'])
print(jax.__version__, jax.devices())
PYCODE
```

The default environment installs the RECOVAR commit pinned in `pixi.toml`. To
change RECOVAR alongside relax, clone RECOVAR next to this checkout and use the
`dev` environment, which installs `../recovar` editable:

```bash
git clone https://github.com/ma-gilles/recovar ../recovar   # then check out the branch you work on
pixi install -e dev
pixi run -e dev python -c "import recovar; print(recovar.__file__)"
```

RECOVAR changes land on RECOVAR `dev2` first; relax then re-pins to that commit.

The empty GPU visibility above is for CPU setup. For GPU execution, start a
separate process with the assigned visibility and `JAX_PLATFORMS=cuda,cpu` before
imports. Never reuse a process that already initialized the wrong backend.
See [Della](docs/development/della.md) for physical-device and Slurm rules.

The optional fast-marching extension builds during installation when a compiler
is available. Custom CUDA requires a compatible local nvcc and this environment's
JAX FFI headers. Build explicitly into an exclusive run directory:

```bash
# Set RUN_ROOT to a new writable run directory before this command.
PIXI_PY="$(pixi run which python)"
PYTHON="$PIXI_PY" make -C relax/cuda LIB="$RUN_ROOT/librelax_cuda.so" all
export RELAX_CUDA_LIB="$RUN_ROOT/librelax_cuda.so"
sha256sum "$RELAX_CUDA_LIB"
```

Load a CUDA toolkit module (for example `module load cudatoolkit/12.8`) only
for the `make` step, in its own shell. With the module loaded, the pixi
environment's JAX reports that no CUDA-enabled jaxlib is installed and falls
back to the CPU; run Python without it.

This is relax's EM library. RECOVAR's own library is built from the installed
recovar package; `RECOVAR_CUDA_LIB` selects an explicit copy of it, and the
loaders refuse a library that lacks their own symbols.

Record source, lock, compiler, headers, loaded library path and library hash.
Verify the library hash before loading, immediately after loading, and after
each paired run. The loaders never build or rewrite an explicit `RELAX_CUDA_LIB`
or `RECOVAR_CUDA_LIB` path: they load it if it exists and exports the required
symbols, and otherwise stop with an error naming the file. Unpinned cache builds
are rebuilt only when the sha256 of their sources differs from the one recorded
beside the library (`<library>.sources.sha256`), not when source mtimes are
newer, and every build renames a temporary file over its target. A library built
directly with `make` records no digest; that is fine for a pinned path. For
qualification, place a freshly copied binary in a dedicated directory and make both the file and directory read-only
before loading. Record the actual loaded path; successful import alone does not
prove that the intended extension ran. Never rebuild a shared library while
another process uses it. End-user pip installation is described in
[installation](https://github.com/ma-gilles/recovar/blob/dev/docs/getting-started/installation.md).

## Validation during development

Read [test rules](tests/CLAUDE.md) before changing or selecting tests. Run the
smallest meaningful check first; advance after it passes. Use explicit backend
placement before pytest collection, which can initialize JAX. Give each run its
own `RECOVAR_JAX_CACHE_DIR` and `JAX_COMPILATION_CACHE_DIR`; RECOVAR can otherwise
reuse a shared cache despite `XDG_CACHE_HOME`. Record whether these caches began
empty or were warmed by a specified command.

| Scope | Starting check | Further qualification |
| --- | --- | --- |
| Pure helpers and reporting | `pixi run python -m pytest -v tests/unit/<affected_test>.py` on CPU | Real missing/invalid/duplicate input cases; affected callers |
| Dense/local EM | `pixi run test-em-fast-guard` | The [test tier](#test-tiers) for the change and the [EM ladder](relax/AGENTS.md) |
| Shared pipeline | Affected unit/integration tests | SPA, cryo-ET, outlier and downstream quality/performance under Slurm |
| GUI or docs | Applicable scoped checks | Build and relevant user workflow checks |

The dense/local fast guard first checks undefined names with the installed Ruff
before importing JAX or compiling tests. This check covers
`relax`; it is not repository-wide lint or scientific
qualification.

Use focused tests between edits. Group related changes into a frozen checkpoint
for broader CPU and applicable GPU checks; full long suites are publication or
milestone checks, not the default response to a small change. Reuse saved outputs
for report-only audits. Repeat a scientific run when the source, workload or an
unresolved failure requires it, and record that reason. The
[EM/VDAM scope](docs/development/em_status.md) records its current boundaries
and qualification gaps.

`pixi run test-fast` selects the repository's unit tier; do not assume every
unit test is tiny or independent of external fixtures. Long and GPU tests run
under Slurm. Record selected versus executed counts, skips and process exit
status. Preserve full logs; a truncated console tail is not a result archive.

CPU placement does not remove native dependencies: some unit tests require the
RELION extension even without a GPU. For those tests, preflight the required
exports from `relax.relion_bind._relion_bind_core` and record the loaded path
and binary hash; an isolated existing build can be selected with
`RECOVAR_RELION_BIND_BUILD_DIR`. Missing required exports are setup failures,
not passing or skipped checks. Report CPU-only, native-oracle, GPU and scientific
parity results separately; the EM fast guard does not qualify the other scopes.

Some legacy EM tests write ledgers beside baselines, and performance helpers
can auto-save hardware entries. Isolate their result-writing paths before
qualification. Do not overwrite established baselines as a side effect of a
benchmark. An optional local fixture skip is not accepted qualification.

## Test tiers

Four tiers, each one command. Pick the tier from the change; the root
[agent guide](CLAUDE.md#test-tiers) holds the same table.

| Change | Tier |
| --- | --- |
| docs, tests, scripts | CPU checks: `pixi run test-em-fast-guard`, the affected unit tests, `python scripts/check_agent_guides.py` |
| engine or numerical code | smoke |
| a numerical change | medium |
| a default flip, an engine replacement, a milestone | long |

| Tier | Command | Budget | Runs | Contents |
| --- | --- | --- | --- | --- |
| smoke | `pixi run test-smoke` | 5 min GPU | one 1-GPU cryoem job if it starts within 15 min, else one idle local GPU 1-3 (never GPU 0, chosen by UUID after nvidia-smi) | CPU fast guard and the merge-guard unit contracts; fixed-state replays `k1_local_replay` (local search), `k1_adaptive_replay` (global K1), `kclass_replay`; the GPU unit files for the changed paths that fit the 5 min budget by their recorded walls (the runner prints the rest, which the medium tier's sweep covers) |
| medium | `pixi run test-medium` | 1-2 h wall | one Slurm job, 3 GPUs | smoke's CPU items; the whole fast parity tier (10 cases); the GPU unit sweep (`tests/unit`, `tests/integration`, `tests/ppca_abinitio`) sharded longest-first; native VDAM K1 50k/256; a K1 5k/128 standalone auto-refine to convergence scored with FSC against a RELION band |
| long | `pixi run test-long` | about 7 h wall, 4 GPUs | one Slurm job, 4 GPUs | the EM long tier (K1 50k/256 standalone and seeded, native VDAM, K4 50k/256) and the K1 (standalone) and K4 100k/256 completions, masked and unmasked, against RELION repeat bands |
| baseline regeneration | `pixi run regen-fixture-manifest`, `pixi run regen-pinned-fast` | - | - | the fixture manifest and the pinned relax outputs. The initial pinned outputs come from a medium run whose pinned cases' own items all passed (other failed items are recorded with the pin); every later regeneration of them needs the user's explicit request. Each pinned entry records its source commit, job, GPU model and date |

Every tier command freezes the checkout (HEAD and any uncommitted diff) into
its run root under `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_test_tiers/`,
verifies the fixtures, builds the native libraries once
(`scripts/build_test_natives.sh`) and ends with a receipt
(`scripts/write_test_receipt.py`: SHA, tier, pass/fail, job id, GPU model). Record the
receipt in your handoff; set `RELAX_TEST_RECEIPTS=<file>` to have it appended
there. The medium tier also runs periodically on `main`.

**Native libraries match their sources.** `scripts/build_test_natives.sh` records in
`NATIVE.json` a digest of the native sources it built from (`relax/cuda`,
`relax/relion_bind`, the build script and the installed recovar commit). The tier runner
refuses to start when the run's natives were built from other sources, and every other tool
that freezes a candidate and reuses prebuilt natives runs
`python scripts/native_sources.py check <natives_dir> --root <candidate checkout>` first:
results on stale natives are not native-exact for their commit.

**Slurm sizing.** Related GPU work runs as one job with `--gres=gpu:N` and one
worker process per GPU (della-cryoem allows 16 running jobs but 32 GPUs per
user). Every tier job goes to the cryoem partition; `--queue general` is only
for when the user explicitly allows another partition. The long tier's
arms share a node, so its walls are not timing-controlled measurements. Measured
long tier (job 14375361, relax 319cd10, 4 A100s, one node): 425 min, bound by the K4
100k/256 completion (413 min); K1 completion 197 min, K1 50k standalone 172 min and
seeded 201 min, K4 50k 91 min, native VDAM 4 min. The planner's estimates are H100
walls, so on A100 the K1 arms take about twice as long. Tier and
benchmark jobs share nodes and take 8 CPUs and 128G per GPU, on any GPU model
unless the H100 matters. Memory above that is
sized from the job's measured peak RSS, and the script says so; nothing asks
for more than about 180G per GPU without that estimate. No job takes a whole node (`--exclusive`): a timing A/B runs
both arms in one `--gres=gpu:2` job on the same node, swapping GPUs between rounds.

**Pass criteria.** A tier passes when every item exits 0 and no parity or
end-to-end case skipped. The oracles are RELION fixed-state runs and pinned
relax outputs. Quality is judged by FSC. Every fast parity case must reach its
FSC-AUC floor and its minimum in-band shell FSC floor against its RELION oracle
for every half or matched class, and keep the mean per-particle |ΔPmax| under
its bound. The K1 5k end-to-end run must converge at RELION's iteration, record
`final_all_data_grid_correct`, and, for both the unfiltered half-map average and the merged
map, stay within 0.002 of the RELION repeat band's GT FSC-AUC and reach cross FSC-AUC
0.990 against every RELION run. The values are in
`tests/tiers/fsc_thresholds.json` (approved by the user 2026-09-24; floor =
1 − 2 × max(relax deficit, RELION repeat deficit)); scoring is
`scripts/em_tier_fsc.py`. Each run also compares its `fsc.json` with the pinned
relax outputs (`tests/tiers/pinned_fast_cases.json`, `scripts/em_tier_pinned.py`).
Pinned outputs are kept per GPU model, and a run is compared only with the
entry of its own model: anything that compares a control with a candidate
numerically pins the same GPU model on both (`--gpu-model a100|h100`). Map
correlation is a diagnostic only.

**GPU noise envelope.** Two runs of the same code on the same GPU model differ: racing
reductions move the printed metrics by about 1e-13 to 1e-6, so control and candidate are
judged by the gates above, never by bitwise equality. `tests/tiers/gpu_noise_envelope.json`
records, per fast-tier case and metric (the gate metrics and the ledger values), the largest
same-code difference observed (`max_abs_diff`) with its sources, and a check limit
`noise_limit` = max(10 × `max_abs_diff`, 1e-9), since each case has only a few same-code
pairs. Measured on H100 from seven pairs (three runs of 02b4cb3, speed's b1c40f9 repeats, the curres
v3/v4 control and candidate): replays move FSC-AUC by at most 1.4e-9 with Pmax identical; cold
starts, perturbreplay and K-class runs by at most 2.2e-7 (minimum shell FSC 3.4e-7, mean
|ΔPmax| 1.6e-6). Unit-level repeats agree (resident wsum_norm_correction 3.5e-8 over 40 runs).
`python scripts/em_tier_noise_envelope.py check --control <basetemp> --candidate <basetemp>`
reports each difference against it; a difference inside needs no rerun, and one outside is
a real change that the gates judge. The envelope is per GPU model (H100 so far); across
models the spread is larger (A100 vs H100 kclass_replay FSC-AUC 3.6e-6).

**Fixtures.** The repository keeps only small metrics, thresholds, pinned
summaries and `tests/fixtures/em_fixture_manifest.json` (per fixture set: root,
file sizes and sha256, the RELION command, version and seed, and generation
records). Bulk data lives under `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/`
(listed in its `README_KEEP.md`) or the curated `em_relion_proj`. Tests resolve
paths through `tests/helpers/em_fixtures.py` and fail, not skip, when data is
missing or a checksum differs.

## Documentation environment

Documentation builds use a separate locked environment with no scientific or
GPU dependencies. The default environment remains unchanged.

```bash
pixi install -e docs --locked
pixi run -e docs docs-build
```

For a local preview, run `pixi run -e docs mkdocs serve`. API references are
collected statically from source; building docs must not require importing JAX
or native extensions. Keep user-site packages disabled when invoking Python
directly, and record the docs lockfile identity with build evidence.

## Before pushing or creating a PR

relax holds no SPA/ET pipeline suites. When relax work changes recovar, the
recovar side follows recovar's own CONTRIBUTING before it is pushed to recovar
`dev2`: in a recovar checkout, `./scripts/run_tests_parallel.sh long-test` and
`pixi run python scripts/extract_regression_tables.py`, with their quality and
performance tables in the report. Include exact source identities, test
commands, Slurm IDs and linked logs.

EM-only changes follow [the EM contract](relax/AGENTS.md), including its
scoped suites and completion evidence, instead of unrelated SPA/ET suites.
A change spanning both scopes requires both sets of applicable checks when
covered by the task; do not infer repeated permission requirements from scope.

Explicit user authorization may allow publishing a draft checkpoint before
these publication prerequisites are complete. Record that authorization and
all missing or failed checks in the PR; retain the requested integration base
and frozen controls. Draft publication does not waive merge, quality or
performance acceptance gates.

Start a PR description with the problem and resulting behavior, then evidence.
Do not confuse replay agreement with an autonomous trajectory, or per-class
quality with an average. See [benchmark contracts](docs/development/benchmarks.md).

## Code style

Ruff uses the 120-character line limit in `pyproject.toml`. Check the Python
files changed by the task, without formatting unrelated files. For uncommitted
work, pass those files explicitly to `pixi run python -m ruff check` and
`pixi run python -m ruff format --check`. The helper
`scripts/check_changed_python.sh --base REV --head REV` compares committed
revisions; it does not validate unstaged changes. Keep existing pre-commit hooks
and their scoped checks in sync with the pinned development toolchain.

## Instruction maintenance

Keep root AGENTS/CLAUDE and EM AGENTS/CLAUDE mirrors identical. Scoped loader
files, such as PPCA's AGENTS.md, intentionally link to their substantive guide.
Keep durable invariants in guides, a short active state/next check in the program
board, and dated experiments in linked archives. Preserve superseded evidence
with its original source and an explicit historical label.

```bash
cmp AGENTS.md CLAUDE.md
cmp relax/AGENTS.md relax/CLAUDE.md
pixi run python scripts/check_agent_guides.py
```
