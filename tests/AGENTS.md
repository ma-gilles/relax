# tests/: rules for tests, gates and pinned files

The root guide applies. Tier budgets, contents and pass criteria are in CONTRIBUTING.md.

## What this directory owns

| Path | Holds |
| --- | --- |
| `unit/` (and `unit/initial_model/`, `unit/ppca_initial_model/`, `unit/ppca_refinement/`, `unit/test_relion_bind/`) | about 580 test files; "unit" is a location, not a size: some need a GPU, the RELION binding or external fixtures |
| `integration/`, `ppca_abinitio/` | parity replays (`integration/test_em_parity_fast.py`), end-to-end and long cases |
| `helpers/` | `float_compare.py`, `em_fixtures.py`, `natives.py`, `gpu_guard.py`, NumPy references, mocks |
| `tiers/` | `fsc_thresholds.json` (gates), `pinned_fast_cases.json` (pinned outputs per GPU model), `gpu_noise_envelope.json`, `gpu_file_seconds.json`, `gpu_path_map.json` |
| `baselines/` | `em_parity_completion_*.json` and `parity/` (pinned references); `relion_vs_relax_benchmarks.json` (a results ledger) |
| `fixtures/em_fixture_manifest.json` | per fixture set: root, sizes, sha256, RELION command and seed |

## Rules

1. Fix the code, not the test. Do not widen an `rtol`, an `atol`, a threshold or a flip fraction, and do not
   add a skip or an ignore, to make a test pass. Propose the change to the owner and wait.
2. No test requires bitwise or ULP-exact float equality, not even under `RELAX_EM_DETERMINISTIC_REDUCTIONS=1`.
   Compare floats with `helpers.float_compare.assert_matches` or `matches` (relative to the array's largest
   magnitude; defaults 1e-6 for float32, 1e-13 for float64). Pass a larger `rtol` only with the measured noise
   cited beside it. Tie-dependent discrete outputs use `flip_fraction`. Byte equality is only for data no float
   computation touches: an input that must stay unmodified, a file checksum.
3. `tiers/fsc_thresholds.json`, `tiers/pinned_fast_cases.json`, `baselines/em_parity_completion_*.json` and
   `baselines/parity/` change only on the owner's instruction; so do the regeneration commands
   `pixi run regen-fixture-manifest` and `pixi run regen-pinned-fast`. `baselines/relion_vs_relax_benchmarks.json`
   is updated when a benchmark is rerun; regenerate its pages with `python scripts/render_benchmark_table.py`.
4. Resolve external data through `helpers/em_fixtures.py`. A missing file or a checksum difference fails the
   test; it never skips.
5. Mark a test that needs the built RELION binding `requires_relion_bind`. It skips without the binding and
   stops the session under `--require-natives`, which the tiers pass.
6. Tests call behaviour. No new test reads source text or executes slices of it. An existing one is converted
   when the code it pins is extracted into something callable.
7. A test names the operation it exercises and passes that operation's operands. Do not write a builder that
   accepts every mode's operands and picks the operation: it keeps removed cases alive.
8. Declare what a comparison must contain: the metric and stage inventory, finite values, executed counts.
   A comparison over whatever keys survived proves nothing.
9. Write scores and ledgers under a unique run root, never beside a pinned file.

## Markers and options

`--strict-markers` is on. Used here: `unit`; `gpu` (run when a GPU is visible, or with `--run-gpu`);
`integration` (`--run-integration`); `slow` (`--run-slow`); `em_parity_long` (`--em-parity-long`);
`requires_relion_bind`. Registered but used by no test: `tiny_metrics`, `long_test`, `gpu_memory_matrix`, `io`.

`conftest.py` does three things to every session: outside Slurm, with `CUDA_VISIBLE_DEVICES` unset or naming
GPU 0, it forces the CPU and skips GPU tests (`helpers/gpu_guard.py`); it removes inherited run-control
variables such as `RELAX_TEST_RECEIPTS` (`RUN_CONTROL_ENV`); it seeds NumPy with 0 before each test.

## Tests for a change here

- The changed file: `pixi run python -m pytest -v tests/unit/<file>.py`.
- A changed helper: the files that import it, `grep -rl 'helpers.<name>\|import <name>' tests`.
- `pixi run test-em-fast-guard` when you touch `test_em_fast_guardrail.py`, `test_relion_replay_state.py`,
  `test_healpix_order_oracle.py` or `test_resolution_scheduling.py`. Keep their case inventory when you
  reorganize tests.
- A new or changed GPU test: `python scripts/run_test_tier.py plan smoke` must list it; then `pixi run test-smoke`.

## Pitfalls

- A GPU test the tier runner cannot recognise is in no tier's GPU sweep. `scripts/run_test_tier.py` selects files
  whose text matches `GPU_TEST_SIGNS` (`pytest.mark.gpu`, `@requires_..._gpu`, `_gpu_available(`,
  `custom_cuda_lib`, ...). `test_resident_pass2_driver.py` is selected through its `requires_resident_gpu`
  `skipif`; a file that skips on some other condition is counted as a CPU file.
- A green CPU run can be mostly skips. Read the skip reasons (`-ra` is on) before reporting a pass.
- `pixi run test-fast` is `pytest tests/ -q` over the whole tree. It is not a fast loop.
- `unit/test_em_deterministic_reductions.py` changes process-wide JAX state at import and must run in its own
  pytest process; the tier runner isolates it (`ISOLATE`).
- A subprocess that imports `relax` must use `conftest.repo_python_command` with `repo_subprocess_env()`
  (or `gpu_subprocess_env()` for a GPU child): the shared environment's editable install can resolve to
  another checkout, and `python scripts/x.py` does not put the repository root on `sys.path`.
- Tests of the refinement controller run it on the CPU stand-in engine (`helpers/tiny_refinement.py`:
  `run_tiny_refinement`, `CallTrace`, `frame_holds`). Some tests of `full_refinement.main` and of the engines
  still read source text; grep the name you move in `tests/` first, and convert such a test when its code
  becomes callable (rule 6).
- Pinned outputs are compared only within one GPU model, and `tiers/pinned_fast_cases.json` holds H100
  entries. Pin the model (`--gpu-model h100`) on both arms of a numerical comparison.
