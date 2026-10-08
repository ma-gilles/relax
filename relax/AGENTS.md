# relax/: package rules

The root guide applies. This file holds the rules of the package; `relax/refinement/`, `relax/sparse_pass2/`
and `relax/ppca_refinement/` have their own guides.

## Imports and ownership

- Package `__init__.py` files import nothing that executes. Import each name from the module that defines it:
  `refine_single_volume` from `relax.refinement.iteration_loop`, K-class execution from
  `relax.classification.k_class`, result types from `relax.classification.k_class_results`.
- The helper and diagnostic modules listed in `scripts/run_em_fast_guard.sh` must not load an execution module
  (controller, scoring, pass 2); `pixi run test-em-fast-guard` fails when they do.
- Production code does not import `relax.relion_bind`. Only `relax/relion_bind/`, `relax/diagnostics/`
  and `relax/commands/build_relion_bind.py` may
  (`tests/unit/test_relion_bind_production_imports.py`).
- Independent numerical references live in `tests/oracles/` (imported by tests as `oracles.<module>`); keep them
  independent of production code.
  Replay and score-audit modules belong in `relax/diagnostics/`.
- EM and VDAM Python APIs and command lines carry no backward-compatibility requirement: migrate callers,
  tests and documents in the same change and leave no forwarding wrapper. Stop at a stored format
  (snapshots, run files, result archives).
- A major policy is a typed option or a command-line flag, never an environment-only fork.
- Sub-agents and parallel work: `SUBAGENTS.md`, only when the owner has allowed them.

## Scientific rules

- Never enable double-precision scoring, projection or M-step by default, and never claim completion from a
  double-only run. Keep the deliberate higher-precision metadata, host arithmetic and accumulators as they
  are: no blanket narrowing.
- A gap that shrinks in double alone does not prove round-off: check serialization, casts, semantics and
  float32 bounds. Report the effective precision of scoring, projection, accumulation and M-step separately.
- Keep the sampler's canonical Euler angles as metadata and derive matrices from them. Never reconstruct known
  angles from rounded matrices. Carry their identity through selection, permutation and class-prior layouts.
  Missing canonical metadata is a question for the owner, not a silent conversion.
- Investigate a divergence by finding the first divergent iteration, half, class, particle, pass and state
  field, then replaying fixed state and candidates. When fixed-state arithmetic agrees, move one state boundary
  earlier. Confirm RELION's behaviour in its source or a dump, add a failing regression, make the smallest
  repair, and record negative findings
  ([procedure](../docs/development/em_parity_runbook.md#investigation-loop)).
- GPU score or Pmax gaps near `1e-4` are arithmetic-level parity. Investigate a reproducible `1e-3` gap or a
  systematic drift. Discrete choices outside the error band match exactly; a near-tie flip needs the measured
  competing scores and margins. The convergence iteration and finalization match exactly.
- Map-quality gates are shellwise FSC, FSC-AUC and the established FSC resolution summaries against ground
  truth and RELION. Correlation is a diagnostic. K=4 uses Hungarian matching and per-class results; never
  average away a poor class.
- Never label an intentional difference strict parity, and never tune until outputs agree.
- Every final all-data map is gridding-corrected, as RELION's reconstruct always is; there is no option to
  skip it.
- In `tests/oracles/iterations.py`, `run_halfset_em_iteration` reads `state.Ft_y` and `state.Ft_CTF` after
  `finish_up_M_step`; keep that order.

## RELION oracle

- The oracle is RELION 5.0.1. Its source commit, dump build and reference binaries are pinned in the
  [oracle rules](../docs/development/em_parity_runbook.md#relion-oracle-rules). Never rebuild the pinned
  binaries and never create another clone.
- RELAX's defaults are RELION's start-up methods and the RELION GUI's job defaults
  ([audit table](../docs/development/relion_defaults.md)). A run that reproduces a particular RELION command
  passes that command's values explicitly.
- Pin source, patched build, command, metadata, seed, subset and MPI layout, and hardware for every comparison.
- A restarted per-half capture fails closed unless the loaded noise is proved to match the target subset
  shellwise.
- Three K-class fast-parity cases need a dispatch schedule captured from the same oracle run and refuse without
  `--relion-dispatch-schedule`. That refusal is a missing fixture, not a parity failure.

## Validation and completion claims

- Before selecting tests for parity work, read the
  [validation ladder](../docs/development/em_parity_runbook.md#validation-ladder).

- Completion requires production-float32 K=1 and exactly K=4, each with at least 100k particles at 256x256 or
  larger, matched inputs, seeds, maps and masks, and RELAX/RELION pairs on the same GPU model. Small,
  historical or double-only results cannot satisfy it
  ([benchmark requirements](../docs/development/em_parity_runbook.md#benchmark-design-and-reporting)).
- A completion or real-data claim needs the full run, its ledger, and the artifact below updated. A short test
  does not substitute for an end-to-end run.
- Record a changed conclusion, its evidence and the next action in `docs/development/em_status.md`. Update
  `docs/math/em_parity_best_metrics.md` only for a completion attempt.

## Standing metric artifacts

Read the value your change touches before running, and write the new value back after.

| Artifact | Holds | Regenerate or check |
| --- | --- | --- |
| `docs/math/em_k1_realdata_science_equivalence_scorecard_v1.json` and `.md` | the EMPIAR real-data gate | `python scripts/summarize_em_k1_realdata_science_equivalence.py --check-markdown` |
| `docs/math/em_relion_parity_scorecard_v2.json`, `em_k4_class_fsc_auc_scorecard_v1.json`, `vdam_relion_parity_scorecard_v1.json` | the synthetic fixed suites (K=1 34-case, K=4 per class, VDAM 12-case) | `python scripts/summarize_*_scorecard.py` with `--check` |
| `docs/benchmarks/frozen_masks.json`, `masked_fsc_scores.json`, `masked_fsc.md` | one frozen mask per dataset and the masked FSC of every scored run; reporting only | `python scripts/masked_fsc.py score`, `collect`, `render --check` |
| `tests/baselines/relion_vs_relax_benchmarks.json` | the RELION-vs-RELAX resolution and wall-time ledger | `python scripts/render_benchmark_table.py --check` |
| `tests/baselines/em_parity_completion_*.json` | pinned completion references | compared by the completion tests; owner's instruction to change |
| `em_parity_quality_{fast,long}_ledger_*.json` beside each case's outputs | what a parity tier measured in one run | `python scripts/extract_em_parity_tables.py --ledger-root <run> --tier fast\|long` |
| all of the above | the consolidated panel | `python scripts/report_em_parity_progress.py --format markdown` |
