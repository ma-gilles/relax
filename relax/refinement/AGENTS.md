# relax/refinement/: Refine3D and Class3D controller

The root guide and `relax/AGENTS.md` apply. Code rules: `docs/development/refactor_rules.md`; the owner's
rulings and the full record: `docs/development/refactor_principles.md`.
A refactor slice follows `docs/development/refactor_procedure.md`; `docs/development/module_template.md` uses
this directory as its worked example; `docs/development/refinement_rules_status.md` lists what still falls short
of the rules (update it in the commit that closes an item).
Algorithm to code: `docs/math/relion_refinement_algorithm.md`.

## What this directory owns

| Module | Owns |
| --- | --- |
| `full_refinement.py` | `relax refine` and `relax class3d`: `run_from_command_line`, `main` (one 1,020-line function) |
| `command_options.py`, `refinement_options.py` | the flags, the RELION GUI job defaults, input-mode admission; the grouped options `refine_single_volume(options=...)` accepts |
| `iteration_loop.py` | `refine_single_volume`: the numbered-iteration controller for K=1 and for K classes |
| `iteration_planning.py`, `convergence.py`, `iteration_snapshot.py` | start-up state, sampling and convergence transitions, the state one iteration hands to the next |
| `expectation.py`, `expectation_batches.py`, `half_scoring.py`, `half_inputs.py`, `local_search_iteration.py`, `local_sampling.py`, `firstiter_cc.py` | preparing and running a half's expectation |
| `mean_helpers.py`, `noise_updates.py` | numbered prior estimation, reconstruction and noise updates |
| `finalization.py`, `final_sampling.py`, `final_reconstruction.py` | the final all-data pass |
| `run_files.py`, `result_files.py` | RELION's per-iteration run files and `--continue`; final archives and maps |
| `refinement_result.py` | what the controller and the final pass return (`RefinementResult` and its records); `archive_fields()` is the saved flat mapping |
| `optics_shapes.py`, `tomo_half.py`, `tomo_particles.py` | several optics groups with different pixel size or box; subtomogram particles |
| `particle_loading.py`, `startup_references.py`, `startup_noise.py`, `projector_preparation.py` | input loading and half sets, start-up maps, prior and noise, projector slabs |

## Rules

1. Keep the main sequence visible in the controller, in its existing order: sampling, expectation,
   reconstruction, mode decisions, state updates, the transition to finalization. Do not rearrange it.
2. Keep each convergence and sampling decision at its existing time, including decisions that read the
   preceding iteration's state.
3. An operation returns its result; the controller assigns state at the call site. Do not hide a write in a
   helper or an `apply` method.
4. Split K=1 and Class3D by contract, not by flag (code rule 6). `mean_helpers.py` has the pattern:
   `reconstruct_numbered_k1_halfmaps` and `reconstruct_numbered_class_maps` each spell out their sequence and
   call shared helpers that take no mode argument.
5. Keep genuine scientific alternatives separate. The K=1 and Class3D prior formulas do not become one formula
   to remove a conditional, and the final pass is not folded into the numbered iterations.
6. Define a flag and its default once, in `command_options.py`. A changed default updates
   `tests/unit/test_relion_gui_defaults.py` and `docs/development/relion_defaults.md`, and needs the long tier.
7. The structure ceilings in `docs/development/refinement_structure_metrics.json` are review signals with
   slack (owner ruling, 2026-10-05): within the slack the check warns and the report names the growth and
   its reason; above it the check fails. After a change that shrinks the code, lower them with
   `python scripts/report_refinement_structure.py --lower-ceilings docs/development/refinement_structure_metrics.json`.
   Raise one only by hand, with the reason in the commit message.

## Tests for a change here

- Always: `pixi run test-em-fast-guard` and
  `pixi run python -m pytest -q tests/unit/test_refinement_structure_metrics.py`.
- The CPU tests that name the module you changed:
  `grep -rl 'refinement.<module>\|refinement import .*<module>' tests/unit | xargs pixi run python -m pytest -q`.
- For a move-only change to the controller or its callees: `pixi run fingerprint check <base> --work-dir <scratch>`
  must report 0 differences in results, files and checkpoints; trace differences confined to log rows are
  allowed and are reported as such. `pixi run fingerprint cases` lists what runs and what it does not cover.
- Then `pixi run test-smoke`. `python scripts/run_test_tier.py plan smoke` prints the GPU files it selects for
  your diff and the ones it defers to the medium tier.
- A change to scoring, reconstruction, noise, priors or convergence is a numerical change: `pixi run test-medium`.

## Pitfalls

- `refine_single_volume` is 1,998 lines; its ceiling is in `docs/development/refinement_structure_metrics.json`.
  Up to the slack (5%) `report_refinement_structure.py --check` warns; beyond it
  `test_refinement_structure_metrics.py` fails. A warning is for the reviewer: extract or delete before you
  add, or say why the growth is needed.
- Tests of the controller run it: `tests/helpers/tiny_refinement.py` runs `refine_single_volume` on the CPU
  stand-in engine, and its `CallTrace` records the order, operands, results and nesting of the calls a test
  wraps (`keep_operands=False` for a lifetime test). `tests/unit/test_k1_mean_lifecycle.py` shows the
  pattern: the previous K=1 maps are released before the prior update and the reconstruction.
  `tests/helpers/tiny_main.py` runs `full_refinement.main` the same way (`run_tiny_main`), or stops it at the
  controller and returns what it was handed (`controller_inputs`); the fingerprint's `main_*` cases compare
  the whole command. No test reads its text; a test that needs a run fact the tiny data cannot produce
  (a frozen boundary, a follower topology) stands in for the one callee that provides it.
- RELION replay inputs the command reads (numbered STAR replay, final-only replay, K=1 initial state, Class3D
  initial translations) live in `relax/diagnostics/replay_inputs.py`, not in the command.
- A diagnostic environment variable of the controller is a field of `DiagnosticEnvironment`
  (`refinement_options`), read once when the options are built (`options.debug.environment`); do not add an
  `os.environ` read below the command.
- Buffer lifetimes follow code rule 3, not old release points. Releasing earlier is allowed; extending a
  lifetime is not; state any change and its peak-memory effect. A test holding a traced call's operands keeps
  them alive: trace with `keep_operands=False` or hold weak references.
- A flag that tells a callee how to read another argument may still have one live caller.
  `_reconstruct_volume_eager` keeps `tau_is_1d` because the final K=1 solve passes a full volume while the
  numbered operations pass shell curves. Trace every caller before removing such a flag.
- Stop a deletion at a stored format. `SplitHalfPrior.fsc_for_update` equals `fsc` for K=1 and stays because
  snapshots and the growth history carry it.
- `relax/refinement/full_refinement.py` maps to `tests/unit/test_refine_relion_mode.py` in
  `tests/tiers/gpu_path_map.json`. That file takes about 33 minutes on a GPU, so smoke defers it to medium.
- Extraction changes when arguments are evaluated. A value read under a guard in the controller may not exist
  in every branch; pass nothing for the branch that never had it instead of inventing a placeholder.
