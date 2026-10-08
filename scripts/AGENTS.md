# scripts/: tooling, launchers and analysis

The root guide applies. Slurm requests in a script follow the owner's global instruction file.

## What this directory owns

| Files | Role |
| --- | --- |
| `run_test_tier.py`, `write_test_receipt.py`, `build_test_natives.sh`, `native_sources.py`, `em_tier_fsc.py`, `em_tier_pinned.py`, `em_tier_bands.py`, `em_tier_noise_envelope.py`, `run_em_fast_guard.sh` | the test tiers: every qualification passes through these |
| `dev/doctor.py`, `check_agent_guides.py`, `report_refinement_structure.py`, `dev/ceilings.py` | read-only checks of the checkout; `dev/ceilings.py` is the slack every structure ceiling applies |
| `dev/fingerprint.py`, `dev/vdam_fingerprint.py`, `dev/refactor_verify.sh`, `dev/refactor_gate.sh`, `dev/refactor_module.sh`, `dev/check_mutation_anchors.py`, `dev/loop_kinds.py` | the move-only refactor check and its gate: `docs/development/refactor_procedure.md` |
| `render_benchmark_table.py`, `summarize_*_scorecard.py`, `summarize_em_k1_realdata_science_equivalence.py`, `masked_fsc.py`, `report_em_parity_progress.py`, `extract_em_parity_tables.py` | renderers of the tracked ledgers and scorecards |
| `build_em_fixture_manifest.py` | writes `tests/fixtures/em_fixture_manifest.json`; the owner's instruction only |
| `run_em_*_slurm.sh`, `run_em_*_slurm.py`, `run_vdam_*.sbatch`, `gpu_matrix/`, `polar/` | Slurm launchers |
| `lib/` | library modules that only scripts and tests import (moved out of `relax/`, which no command of it reaches); imported as `scripts.lib.<module>` |
| `run_*`, `analyze_*`, `audit_*`, `validate_*`, `compare_*`, `prepare_*` (about 180 files) | one-off experiment drivers and analyses; most are records of a past question |

## Rules

1. `dev/doctor.py` uses the standard library only, runs under any Python and changes nothing. Keep it so.
2. A renderer has a `--check` (or `--check-markdown`) mode that fails when the tracked page is stale. Change
   the JSON, rerun the renderer, commit both; never edit a rendered page by hand.
3. A script that other code imports is part of the test surface: about 160 test files import `scripts.<name>`, and
   `scripts/` has no `__init__.py`. Before renaming or deleting one, `grep -rn 'scripts.<name>\|scripts import <name>' tests scripts`.
4. A script run as `python scripts/x.py` has `scripts/` on `sys.path`, not the repository root. Insert the root
   before importing `relax` or `scripts.*`, as `run_test_tier.py` does.
5. A launcher reads its output and runtime roots from an environment variable or an option and uses a
   runtime root per job. What it creates is marked as the global file's data layout requires.
6. The tier sbatch is submitted with `--export=NONE`. A variable a tier item needs must be re-exported by name
   in `run_test_tier.py`; it is not inherited from your shell.

## Tests for a change here

- The script's own test when it has one: `tests/unit/test_run_test_tier.py`, `test_dev_doctor.py`,
  `test_check_agent_guides.py`, `test_refinement_structure_metrics.py`,
  `test_run_em_k1_robustness_matrix_slurm.py`, and so on (`ls tests/unit | grep <name>`).
- A renderer: run it with its check flag on the tracked file.
- A change to `run_test_tier.py`, `build_test_natives.sh` or `native_sources.py`:
  `python scripts/run_test_tier.py submit smoke --dry-run`, then a real `pixi run test-smoke`.
- The tier runner selects GPU tests that import a changed `scripts/*.py` module, as it does for `relax/`.

## Pitfalls

- Default roots name one account's directories. `run_em_k1_robustness_matrix_slurm.sh` defaults
  `EM_K1_MATRIX_SCRATCH_DIR` and `EM_K1_MATRIX_RUNTIME_ROOT` to a root the second account cannot write;
  `run_test_tier.py` defaults `RUN_BASE` likewise (`--run-root` overrides it). `pixi run doctor` lists each
  one with the variable to set.
- Thirteen `run_vdam_*.sbatch` files carry `--account=amits`, which only one account has. The global file
  names the account to use instead; pass it on the `sbatch` command line, which overrides the directive.
- `build_test_natives.sh` and `run_test_tier.py` call `<checkout>/.pixi/envs/default/bin/python` directly and
  default `RELION_SRC_DIR` to one fixed path. A worktree without `.pixi` cannot run a tier; `pixi run doctor`
  prints the fix.
- Launcher tests assert on the launcher's text. `tests/unit/test_run_em_k1_robustness_matrix_slurm.py` holds
  over a hundred `"..." in script` asserts; rewording a directive or a comment in the launcher fails them.
- Load the CUDA toolkit module only for a `make` step, in its own shell. With the module loaded, the pixi
  environment's JAX finds no CUDA-enabled jaxlib and falls back to the CPU.
- The smoke job requests one hour for a five-minute budget (`TIER_JOB` in `run_test_tier.py`). That is the
  cap for overruns, not the expected wall.
