# relax/refinement/: Refine3D and Class3D controller

The root guide and `relax/AGENTS.md` apply. Rules with their examples: `docs/development/refactor_principles.md`.
Algorithm to code: `docs/math/relion_refinement_algorithm.md`.

## What this directory owns

| Module | Owns |
| --- | --- |
| `full_refinement.py` | `relax refine` and `relax class3d`: `run_from_command_line`, `main` (one 1,767-line function) |
| `command_options.py`, `refinement_options.py` | the flags, the RELION GUI job defaults, input-mode admission; the grouped options `refine_single_volume(options=...)` accepts |
| `iteration_loop.py` | `refine_single_volume`: the numbered-iteration controller for K=1 and for K classes |
| `iteration_planning.py`, `convergence.py`, `iteration_snapshot.py` | start-up state, sampling and convergence transitions, the state one iteration hands to the next |
| `expectation.py`, `expectation_batches.py`, `half_scoring.py`, `half_inputs.py`, `local_search_iteration.py`, `local_sampling.py`, `firstiter_cc.py` | preparing and running a half's expectation |
| `mean_helpers.py`, `noise_updates.py` | numbered prior estimation, reconstruction and noise updates |
| `finalization.py`, `final_sampling.py`, `final_reconstruction.py` | the final all-data pass |
| `run_files.py`, `result_files.py` | RELION's per-iteration run files and `--continue`; final archives and maps |
| `optics_shapes.py`, `tomo_half.py`, `tomo_particles.py` | several optics groups with different pixel size or box; subtomogram particles |
| `particle_loading.py`, `startup_noise.py`, `projector_preparation.py` | input loading, start-up noise, projector slabs |

## Rules

1. Keep the main sequence visible in the controller, in its existing order: sampling, expectation,
   reconstruction, mode decisions, state updates, the transition to finalization. Do not rearrange it.
2. Keep each convergence and sampling decision at its existing time, including decisions that read the
   preceding iteration's state.
3. An operation returns its result; the controller assigns state at the call site. Do not hide a write in a
   helper or an `apply` method.
4. Share steps between K=1 and Class3D, not a sequence with a mode flag. `mean_helpers.py` has the pattern:
   `reconstruct_numbered_k1_halfmaps` and `reconstruct_numbered_class_maps` each spell out their sequence and
   call shared helpers that take no mode argument.
5. Keep genuine scientific alternatives separate. The K=1 and Class3D prior formulas do not become one formula
   to remove a conditional, and the final pass is not folded into the numbered iterations.
6. Define a flag and its default once, in `command_options.py`. A changed default updates
   `tests/unit/test_relion_gui_defaults.py` and `docs/development/relion_defaults.md`, and needs the long tier.
7. Stay under the structure ceilings in `docs/development/refinement_structure_metrics.json`. After a change
   that shrinks the code, lower them with
   `python scripts/report_refinement_structure.py --lower-ceilings docs/development/refinement_structure_metrics.json`.
   Raise one only by hand, with the reason in the commit message.

## Tests for a change here

- Always: `pixi run test-em-fast-guard` and
  `pixi run python -m pytest -q tests/unit/test_refinement_structure_metrics.py`.
- The CPU tests that name the module you changed:
  `grep -rl 'refinement.<module>\|refinement import .*<module>' tests/unit | xargs pixi run python -m pytest -q`.
- For a move-only change to the controller or its callees: `pixi run fingerprint check <base> --work-dir <scratch>`
  must report 0 differences. `pixi run fingerprint cases` lists what runs and what it does not cover.
- Then `pixi run test-smoke`. `python scripts/run_test_tier.py plan smoke` prints the GPU files it selects for
  your diff and the ones it defers to the medium tier.
- A change to scoring, reconstruction, noise, priors or convergence is a numerical change: `pixi run test-medium`.

## Pitfalls

- `refine_single_volume` is 2,307 lines and its ceiling is 2,307. One added line fails
  `test_refinement_structure_metrics.py`: extract or delete before you add.
- Tests pin the controller's source text. `tests/unit/test_k1_mean_lifecycle.py` asserts that `del init_volume`
  precedes `_snapshot_and_release_previous_k1_means(reference_model.maps)`, which precedes
  `estimate_split_half_prior(` and `reconstruct_numbered_k1_halfmaps(` in the text of `refine_single_volume`.
  Before renaming or moving a statement, `grep -rn '<name>' tests/` and update the pins in the same commit.
- Release points are behaviour. That test pins one: the previous K=1 means are copied to the host and released
  before the prior update and the reconstruction. Moving a release changes what is alive during later GPU work
  even when no value changes; it gets its own commit that says what differs at run time.
- A flag that tells a callee how to read another argument may still have one live caller.
  `_reconstruct_volume_eager` keeps `tau_is_1d` because the final K=1 solve passes a full volume while the
  numbered operations pass shell curves. Trace every caller before removing such a flag.
- Stop a deletion at a stored format. `SplitHalfPrior.fsc_for_update` equals `fsc` for K=1 and stays because
  snapshots and the growth history carry it.
- `relax/refinement/full_refinement.py` maps to `tests/unit/test_refine_relion_mode.py` in
  `tests/tiers/gpu_path_map.json`. That file takes about 33 minutes on a GPU, so smoke defers it to medium.
- Extraction changes when arguments are evaluated. A value read under a guard in the controller may not exist
  in every branch; pass nothing for the branch that never had it instead of inventing a placeholder.
