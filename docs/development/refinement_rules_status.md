# relax/refinement against the rules: verdict, exceptions and what remains

The agent-facing record of the open items of the refinement package against
[refactor_rules.md](refactor_rules.md) and [module_template.md](module_template.md). Close an item by
deleting it here in the commit that closes it; record a decision in the "Decided" section. Numbers are of
main on 2026-10-06. Gate and verify: `REFACTOR_MODULE=refinement` (`scripts/dev/refactor_module.sh`).

## Verdict (close-out, 2026-10-06)

**`relax/refinement` meets the code rules, with the recorded exceptions below.** Each exception names its
reason; the sections after this one hold the detail.

| Rule | Status |
| --- | --- |
| 1 numbers, 2 interfaces | Met: every move was fingerprint-identical (results, files, checkpoints, trace) and passed the GPU tiers; only logger names and accepted new option leaves differ. Archive keys are unchanged (`RefinementResult.archive_fields`). |
| 3 lifetimes | Met: the controller tests check required lifetimes (`frame_holds`, `keep_operands=False`). Releases that freed nothing are gone (2026-10-08); the M-step records are dropped in the end-of-iteration block, where their fields were. `RefinementResult` keeps the run's `RefinementHistory` (host curves, kilobytes). |
| 4 JAX | Met (audit below). Exceptions: the donating normalisation executable, the box-scale host staging and the end-of-iteration sync. |
| 5 decide once | Met for flags and the controller's diagnostics (`command_options` resolvers, `DiagnosticEnvironment`). The controller takes the command's options (no `RefinementOptions()` default inside) and its scoring precision is `options.precision` (owner review, 2026-10-07). Exceptions: the parity, replay and checkpoint records are built from start-up results; some environment reads stay where they are used (item 2), among them the engines' reads of the dense precision. |
| 6 contracts | Met: the controller's 24 mode tests select contract functions, mode-only steps or refusals (owner, 2026-10-04 and 2026-10-05); the final scoring loop and `DenseVariantPolicy.k_class_enabled` are variants. |
| 7 one owner | Met; no known duplicate. Exception: the pixel size is kept as the input scalar and as `ReconstructionSettings.voxel_size` (a float) on purpose (rule 1). |
| 8 config/state/results | Met: option records are frozen; the controller and the final pass return `RefinementResult`. Exception: the prior's per-shell outputs come from `relax.reconstruction.regularization_relion` as a dict (`return_details=True`); that module owns the type. |
| 9 transitions | Met (audit below). The expected-accuracy inputs, which steer sampling, are `options.expected_accuracy`, no longer under `debug`; each numbered half's per_half and significance writes are `finish_numbered_half`, a module-level step with explicit inputs (owner review, 2026-10-07). Exceptions: the local-search profile sink and the opt-in pass-2 diagnostic variants. |
| 10 signatures | Met: two groupings; every remaining function with ten or more parameters is accepted with its reason (table below). The two sentences added on 2026-10-07 (no record or option group unpacked into locals; no closures over controller state as callbacks) are met by `k1_maximization` (reads `SplitHalfPrior` where used), the two controllers (no option aliases) and the numbered halves (`NumberedHalfInputs`, `finish_numbered_half`). The M-step results are kept as records (mstep). The input source receives the scoring state as values (`ports.ScoringState`, `ScoringArrays`; deep2 c3), not thunks. Open against the 2026-10-08 extension: `dump_debug=partial(observer.noise_updated, ...)` (an observer callback handed to the noise update), and the positional unpacks of the state-swap 7-tuple and `join_half_accumulators_at_low_resolution`'s five values. |
| 11 layers | Met (audit below). Exceptions: `tomo_particles` (shared by engines and other workflows) and 62 private-name imports from `relax.helpers`, `relax.diagnostics` and `relax.relion`, until those modules are refactored. |
| 12 edges | Met: every command admission refuses with a message, each tested; the end-of-iteration sync no longer swallows device errors. |
| 13 tests | Met: no test reads controller or command source. Exception: four engine-core lint tests. |
| 14 evidence | Met: each landing listed what the fingerprint and the tiers do not cover (coverage limits below). |
| 15 ports | Partly met (2026-10-06, section below): the observers and the tiered replays go through the ports; the untiered comparison features and the run's telemetry do not yet. |

### Accepted, recorded exceptions

- The command builds the parity, replay and checkpoint option records from start-up results: they are
  results, not flags.
- Environment reads that stay where they are used (item 2 below), each with its reason.
- `refine_single_volume` stays one function (1,782 lines on 2026-10-07; owner, 2026-10-04). It holds no step with a
  mode flag.
- The prior's per-shell outputs stay a dict until `relax.reconstruction` is refactored.
- The coverage limits below (RELION run-directory fixture, local-search harness): costed, not built.

## Detail: the command, the environment and the controller loop

1. **The command still assembles three option records from start-up results (rules 5, 7).** The
   schedule, adaptive, batching, overlap, local-search, k_class and debug records are resolved in
   `command_options` (`resolve_*`, field tests in `tests/unit/test_option_resolvers.py`). The parity, replay
   and checkpoint records stay in `full_refinement.main`: their fields are start-up results (half sets,
   references, noise, poses, replay inputs, follower topology, the frozen boundary), not flags, so a
   resolver would be a function of ten or more of them. `main` is 1,020 lines.
2. **Environment reads below the boundary (rule 5).** The controller's remaining diagnostic switches and
   dump directories are one `DiagnosticEnvironment` (`EngineDebugOptions.environment`), read when the options
   are built; the final pass's two run variants are `FinalPassOptions`, read the same way. Dumps that watch
   the run are observers (`relax.refinement.ports.RunObserver`, rule 15), built at the command from flags
   and the environment (`relax.diagnostics.observers`): the intermediates, the parity capture and timings,
   the BPref accumulator captures and the noise-update terms. Still read where used, each with its reason:
   - `projector_preparation.prepare_scoring_projector`: `RELAX_RELION_PROJECTOR_CACHE_DIR` and
     `RELAX_RELION_PROJECTOR_DUMP_DIR`; `relax/helpers/expected_accuracy.py` calls it too, outside this
     package.
   - `local_search_iteration`: the x-half batch guard; `half_scoring`: hides the local engine's dump
     variables during the denominator pass; `expectation`: whether a BPref dump of the engines is armed. The
     variables belong to the engine packages, which read them.
   - `expectation_batches`: `RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS` (owned by
     `relax.helpers.dtype_policy`); `particle_loading`: `RELAX_USE_FLOAT64_SCORING` (the precision policy
     `relax.dense.scoring_policy.DENSE_PRECISION` reads it at import; particle loading runs at the command)
     and the RECOVAR native softmask switch, which is how the setting reaches RECOVAR.
   - The scoring and reconstruction route variants (x-half M-step, local adaptive pass-2 support, the
     support-width convergence gate, the reconstruction's stable windows and host inverse FFT) are one
     `ScoringVariants` (`RefinementOptions.variants`), read when the options are built; the engines below
     receive its values in their execution policies and `ReconstructionSettings.programs` (deep2 c1). The
     resident engine (`relax/sparse_pass2`) and the K-class engine still read the stable-windows switch
     themselves.
   - The dense precision: the engines receive `options.precision` (`DenseExecutionPolicy`,
     `LocalExecutionPolicy`, `BatchPlanner`, dtype arguments). `relax/parity/relion_replay_source.py` and
     `relax/diagnostics/iteration.py` still read `DENSE_PRECISION`, and `require_process_precision` refuses
     options whose precision differs, so the two cannot disagree.
3. **The controller loop (rule 6, shared steps).** `refine_single_volume` spans 1,782 lines (2026-10-07). Its mode
   decisions, read by the rules (2026-10-05): 24 tests of `k_class_enabled`. None makes a step take a
   mode flag; each either selects one of two contract functions (image size: `plan_class_image_size` or
   `plan_halfmap_image_size`; M-step: `class_maximization` or `k1_maximization`; direction priors,
   unregularized maps, resolution, the final pass), runs a step only one mode has (class weights from the
   posterior, the K=1 FSC record and growth latch), refuses an unsupported route (Class3D local search, the
   frozen-state check), or picks which mode's value a record or log carries. That is the shape the owner
   accepted on 2026-10-04 (the loop is not split; the steps are mode-free). What remains is glue: values
   computed in one place and read in several, kept as loose locals. Grouping them into typed records (rule
   8), one group per commit, judged by readability: done `PublishedAccuracy` (four locals: the latest
   expected-accuracy trials and per-class accuracies). Candidates, by the number of reads they would
   replace (the M-step records are done, 2026-10-08): Class3D's class state (assignments, previous assignments, mixture); RELION's growth
   latch (`incr_size`, `has_high_fsc_at_limit`, which the planners and the run files take apart). 29
   functions take ten or more parameters; the widest are `run_final_all_data` (28), `score_tomo_half` (26)
   and `build_archive_metadata` (20).

## Rule 15: what goes through the ports, and what does not yet (2026-10-06)

Through the ports (`relax/refinement/ports.py`), chosen by the command:
- `RunObserver` (`relax/diagnostics/observers.py`): the intermediates (`--save_intermediates_dir`), the parity
  capture and its stage timings (`RELAX_PARITY_DUMP_DIR`, `RELAX_PARITY_TIMING_DIR`, `--timing_dir`), the BPref
  accumulator captures (`RELAX_BPREF_PREJOIN_DUMP_DIR`, `RELAX_BPREF_ACCUM_DUMP_DIR`), the noise-update terms
  (`RELAX_NOISE_DEBUG_DUMP_DIR`), the Class3D image-size and M-step dumps (`RELAX_KCLASS_DUMP_DIR`) and the
  pre-mask maps (`RELAX_PREMASK_DUMP_DIR`).
- `InputSource` (`relax/parity/relion_replay_source.py`, `RelionReplay`): the numbered STAR replay
  (`--perturb_replay_relion_dir`), `--relion_init_dir`'s run_it000 slot and every other override slot, the
  cutoff, perturbation, optimiser and restart counters, the Class3D prior replay, the final pass's replayed
  state and sampling STAR (the final-only replay rides these call sites; its functions are unchanged).
  The frozen boundary (`--frozen-boundary-dir`: its sealed sampling state and scoring context, its
  `RefinementState` fields and the check that the scoring state is unchanged before the first iteration) and
  the state-swap probe (`--state-swap-*`: its snapshot, RELION references and swap) are `RelionReplay` fields;
  only the source calls `relax.diagnostics.frozen_boundary` and `state_swap_runtime`
  (`tests/unit/test_refinement_port_imports.py`). The follower topology of an MPI RELION run
  (`--relion-scale-followers`, the captured dispatch schedule `--relion-dispatch-schedule` and the follower-scale
  replay) is `RelionReplay.follower_topology`; the controller reads it as `source.follower_topology`. It is a
  tiered replay (medium's three K4 fast cases; a strict K>1 replay requires it) and stays (lead, 2026-10-06).
  Owner decision, 2026-10-06: these and the final-only
  replay are ported on the fingerprint and the CPU admission tests alone. **Runs from a real RELION run
  directory (a frozen boundary, a state-swap probe, a final-only replay) are not tested**: no tier runs
  them and no CPU fixture is a RELION run directory (see Known coverage limits).
- At the command, in `relax/parity` (deep2 c3): the admission of RELION oracle inputs
  (`oracle_admission.py`: the dispatch capture, the follower routing, the restart provenance), the start-up
  noise taken from elsewhere (`startup_noise_inputs.py`: a frozen boundary's, `--init_noise_from_npz`, the live
  K=1 estimate), what a frozen boundary replays (`RelionReplay.from_frozen_boundary`, with the start-up direction
  priors it keeps) and the archive's replay keys (`archive_provenance.py`). The source owns the sealed
  perturbation, a replayed translation grid at an unchanged order (`coarse_grids`), a sealed pass-1 width
  (`adaptive_coarse_size`), the sealed initial grid and the two replay predicates of the resume check; the
  controller passes the perturbation's HEALPix order, not the replayed sampling record.
- Run options, not diagnostics (rule 9): the local-search probe (`LocalSearchOptions.stop_after_local_search*`)
  and the final pass's after-the-cap and merged-reference variants (`FinalPassOptions`).

Retired (owner, 2026-10-06): the captured RELION projector (`--relion-projector-capture-dir`, `-manifest`,
`-iteration`; `relax/diagnostics/relion_projector_capture.py`, `scripts/parity/rebuild_relion_projector_capture.py`)
is deleted. Its code is at git tag `retired/captured-projector-20261006`, the last commit of main that has it;
the flags refuse and name the tag, and its archive and ledger keys keep the values a run without one wrote.
A frozen boundary's replay slots now stay empty (`empty_replay_slots`).

Not yet through the ports, each still read where it was:
- The engine telemetry, the profile histories and the setup timers: they are archive keys of the default
  run (`RefinementHistory`, `NumberedMetadata.setup_phase_seconds`).
- The significance and pass-2 single-half selectors (`RELAX_SIGNIFICANCE_DUMP_TARGET_HALF`,
  `RELAX_PASS2_DUMP_TARGET_HALF`) steer the run (one half only) and read the environment in the controller.
- The per-half E-step capture (`parity_dump.collect_e_step`) and the engines' own dump variables.
- Port-only parameters still threaded through the algorithm (REVIEW_DEEP #12; None or False natively):
  `NumberedState.prior_translations` -> `score_numbered_half(replay_prior_translations=)` ->
  `LocalPriorSpec.replay_prior_translations`; `source.sealed_sampling_state` -> `relion_direction_log_priors`,
  `prepare_final_half` and `iteration_trial_grid(sealed_grid=)`; `_validate_bpref_particle_order_scope` reads
  the replay's fields. `FinalSamplingSettings.sampling_star`/`sampling_star_source` (set only by the replay
  source) reach the archive through `RefinementResult`.
- The projector disk cache (`RELAX_RELION_PROJECTOR_CACHE_DIR`, deep2 M S43) is read in
  `prepare_scoring_projector`, which `relax/helpers/expected_accuracy.py` also calls.
- `parity.use_per_half_mean_variance` (deep2 O S2, checked): set only on a frozen boundary's fixed arm, it is a
  variant of the algorithm (each half scores against its own tau2), not a replaced input; it stays a run option.

## Rule 10: the functions with ten or more parameters

The target is the smallest coherent input, not a count. Grouped (2026-10-05): `score_tomo_half_in_loop` takes
`HalfScoringData` (24 -> 14), `build_archive_metadata` and `RunReport` take `RestartProvenance`. The rest are
accepted, with the reason:

| Function | Params | Accepted because |
| --- | --- | --- |
| `finalization.run_final_all_data` | 28 | Takes the controller's owners whole; the rest are run flags and perturbation values of different lifetimes that no record holds. |
| `tomo_half.score_tomo_half` | 26 | Engine boundary (rule 11: engines take arrays); `relax/vdam` calls it with raw arrays. |
| `result_files.build_archive_metadata` | 20 | One field each of many owners, written once into the archive; the replay provenance keys come as one dict (`relax/parity/archive_provenance.py`, deep2 c3). |
| `expectation.score_numbered_half` | 17 | Takes `NumberedHalfInputs`, the phase and the options; the rest are the iteration's shared operands, bound once by the controller. |
| `expectation_batches.prepare_half_batches` | 19 | Reads three fields of `RelionParityOptions` and two of `DenseVariantPolicy`: fields of a large object (rule 10). |
| `mean_helpers.estimate_class_priors` | 19 | Array operands and iteration scalars; `reference_model` would add a mutable owner. |
| `expectation.prepare_numbered_expectation` | 18 | No record covers more than two of its inputs. |
| `mean_helpers._reconstruct_volume_eager` | 18 | Six fields equal `ReconstructionSettings`'s at the production calls; those calls go through `ReconstructionSettings.reconstruct`, which forwards them. The solver keeps its raw-geometry signature for the 21 tests and the scripts that call it directly. |
| `convergence.update_iteration_convergence` | 16 | Separate inputs of one update; the controller passes the K=1/Class3D differences as values (`sampling_decision_now`, `class_change_fraction`) and applies the K=1 follower-counter reset itself. |
| `maximization.class_maximization`, `k1_maximization` | 16 | M-step operands plus iteration scalars; `per_half` would carry posterior arrays past their release (rule 3). |
| `SnapshotCapture.finish`, `finish_k1`, `finish_class` | 13-15 | One snapshot field per parameter, each from a different owner. |
| `iteration_planning.plan_halfmap_image_size`, `plan_class_image_size` | 10-14 | Scheduling inputs; `ImageGeometry` would change the pixel size's scalar type (rule 1). RELION's growth latch is a pair. |
| `expectation.prepare_final_half` | 13 | Operands plus settings. |
| `optics_shapes.shape_class_engine_inputs` | 13 | No production caller yet; kept for VDAM's multi-shape loop (`em_status.md`). |
| `local_sampling.prepare_numbered_local_sampling` | 11 | `CoarseGrids` would cover two fields. |
| `mean_helpers.join_half_accumulators_at_low_resolution` | 11 | Three fields equal `ReconstructionSettings`'s, but K=1 passes the raw pixel-size scalar the settings convert (rule 1). |
| `mean_helpers.estimate_class_prior`, `estimate_split_half_prior` | 10-11 | Already take `ReconstructionSettings`; the rest are operands. |
| `optics_shapes.prepare_optics` | 11 | `HalfSet` would cover two fields. |
| `TomoHalf.__init__`, `RunFileWriter.__init__`, `BatchPlanner.__call__` | 10-11 | A record's own constructor; writer policy beside `RunSettings`; a per-pass query with defaults. |

## Rules 4, 9 and 11: audit findings (2026-10-05)

- **Rule 4: met.** Two jit sites, both module-level or shape-cached, no side effects in traced code, no
  record crosses a transformation. Fixed: the end-of-iteration `block_until_ready` no longer swallows
  device errors. Accepted: the donating normalisation executable (`mean_helpers`, an int32 overflow of
  XLA's iFFT normalisation at box scale), the box-scale host staging and the end-of-iteration sync
  (memory boundaries), and about twelve once-per-iteration syncs for logs and history.
- **Rule 9: met, with exceptions.** Fixed: the six operations that update arguments in place now say so.
  The local-search probe (`stop_after_local_search*`, `LocalSearchOptions`) and the final pass's
  after-the-cap and merged-reference variants (`FinalPassOptions`) change the run, so they are run options,
  not diagnostics (2026-10-06). Accepted: `half_scoring` appends local-search profile rows to the list the
  controller hands its diagnostics policy (an observation sink nothing reads back); the `adaptive_pass2_*` switches and
  `diagnostic_score_only` of `LocalDiagnosticPolicy` are opt-in diagnostic variants of the pass-2 layout
  (default path unchanged). The prior's result is a dict named `details` (`prior_shells`, `ssnr_shells`)
  that the resolution and the run files read: it is the prior's output, not a diagnostics structure, and
  is listed with the result types (rule 8).
- **Rule 11: met, with exceptions.** Fixed: the controller no longer imports the command
  (`configure_half_image_preprocessing` moved to `half_inputs`); `InitialSampling` and `RestartProvenance`
  moved to `refinement_options`; `relax.diagnostics.relion_replay` imports two now-public `half_inputs`
  helpers. Accepted until those packages are refactored: `relax/refinement/tomo_particles.py` (numpy only)
  is imported by two engines and three other workflows, and its home moves with them; 62 imports of
  private names from `relax.helpers`, `relax.diagnostics` and `relax.relion` stay until those modules make
  them public.

## Fingerprint against a main older than 7237d091

On a main without 7237d091 (the CPU mock datasets' `original_image_indices_from_local(indices=None)`), 70 of the
fingerprint's 109 cases end in a TypeError at the final pass's expected-accuracy estimate (the final pass
raises since 48330782). A `fingerprint.py check` of a later head against such a main therefore exits 1: those
70 cases now complete. At 43ce3087 every output difference against main 34d9f53e was in one of those cases,
every other case was identical, and every commit was 0-diff against its parent (the case-by-case check:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/parity_ports_20261006/verify/logs/fpcheck_main.txt`).

## Which tier tests reach relax/refinement (2026-10-06)

The GPU items of `scripts/run_test_tier.py` that run the refinement controller (`refine_single_volume`, and
`run_final_all_data` where noted), with the paths each reaches. Fast cases are in
`tests/integration/test_em_parity_fast.py`; none sets an observer (intermediates or parity dump).

| Tier | Item | K | Start | Search | Final pass | Other paths |
|---|---|---|---|---|---|---|
| smoke | `k1_local_replay` | 1 | RELION it006 state through `scripts/run_multi_iter_parity.py` (`RelionReplaySource`) | local search, hp4 | no | the local-search regime |
| smoke | `k1_adaptive_replay` | 1 | RELION it003 os1 state, same script | global, adaptive os1 | no | |
| medium | `k1_replay` | 1 | RELION it003 state, same script | global, os0 | no | |
| medium | `k1_coldstart[standalone]`, `k1_os1_coldstart_standalone`, `k1_gui60_coldstart_standalone` | 1 | native cold start, 3 iterations | global hp3 os0; hp3 os1; hp2 os1 (60 A start) | no | |
| medium | `k1_coldstart[relion_seeded_debug]` | 1 | RELION half sets (`--relion_half_sets`) | global hp3 | no | |
| medium | `k1_perturbreplay` | 1 | `--perturb_replay_relion_dir` (numbered STAR replay), 3 iterations | global hp3 os0 | no | |
| medium | `k1_multioptics_coldstart`, `k1_multioptics_firstiter_cc` | 1 | native, `--init_volume` | global | no | two optics groups of different pixel size and box |
| medium | `kclass_coldstart` | 4 | `--perturb_replay_relion_dir`, 3 iterations | global hp2 os1 | no | Class3D; `--relion-dispatch-schedule` |
| medium | `kclass_nonadaptive_replay`, `kclass_strict_oversample_coldstart` | 4 | `--relion_init_dir` + `--perturb_replay_relion_dir` | global hp1 os0; hp1 os1 | no | Class3D; `--relion-dispatch-schedule` |
| medium | `e2e_k1_5k_standalone` (`tests/integration/test_em_tier_e2e.py`) | 1 | native cold start, to convergence (up to 25 iterations) | global, then local search | yes, asserted | the only tier item that runs the final pass of a converged run |
| medium | GPU sweep: `test_tomo_aberrations_refine.py`, `test_tomo_premultiplied_final_pass.py`, `test_class3d_multishape_smoke.py`, `test_non_finite_image_stops_before_output_gpu.py` | 1 / K | native | `--auto_local_healpix_order 2` (tomo) | tomo aberrations: yes (`RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER=1`) | tomography; several-shape Class3D; the non-finite refusal |

Not refinement: `kclass_replay` (smoke and medium) calls `relax.classification.k_class` directly; `vdam_k1_50k`
runs `relax.commands.initial_model` (relax/vdam); the guard and `cpu_merge_units` are CPU. The GPU unit sweep
also runs the refinement unit files marked GPU (`test_cuda_relion_translation.py`, `test_relion_functions.py`,
`test_resident_operand_avals.py`, `test_resident_tilt_scoring.py`, one test of `test_refine_relion_mode.py`).
Smoke additionally runs the GPU unit files a change touches, within its budget.

So smoke reaches the K=1 replay and local search only. A slice that touches cold start, Class3D, the
numbered STAR replay, `--relion_init_dir`, the dispatch schedule, multi-optics, tomography or the final pass
needs medium. No tier runs the frozen boundary, the state-swap probe or the final-only replay.

## Known coverage limits (recorded, not being built)

- **RELION run directories have no CPU fixture.** The fingerprint's `main_*` cases do not reach the frozen
  boundary, the RELION replays (`--relion_init_dir`, `--perturb_replay_relion_dir`, the final-only replay),
  or the state-swap probe: the tiny data cannot produce a RELION run directory. CPU
  tests stand in for the one callee that supplies such a fact (a stand-in boundary or follower topology).
  The GPU tiers run three of these paths end to end: the numbered STAR replay
  (`--perturb_replay_relion_dir`: medium `k1_perturbreplay` and the K4 fast cases, long `realdata_hp3_replay`),
  `--relion_init_dir` (medium `kclass_nonadaptive_replay` and `kclass_strict_oversample_coldstart`, long
  `k1_relion_seeded_debug`) and the follower dispatch schedule (the same K4 medium cases; a strict K>1 replay
  requires it). The frozen boundary, the state-swap probe and the final-only replay are passed by no tier: from the command they have CPU admission
  tests only; at the controller boundary the fingerprint's cases exercise the state-swap probe, the frozen
  scoring-state assertion, the sealed sampling state and per-iteration and final-only replay priors
  (parity audit, 2026-10-06). Cost of closing it: a tiny run-directory fixture
  written by RELION itself on the tiny data (`run_it000/001_{data,model,optimiser,sampling}.star`, half
  maps), checked in with its manifest, plus fingerprint cases using it: about two to three agent-days,
  mostly in making RELION's numbered files consistent with the stand-in engine's run.
- **Local search has no CPU harness.** The stand-in engine replaces the dense adaptive engine only; the
  resident local-search driver needs the x-half layout and its tables. Local search, tomography, symmetry
  other than C1 and multi-shape optics are therefore covered by the GPU tiers alone. Cost: a stand-in for
  the local driver at the `local_search_iteration` boundary seeded by its operands, as the dense one is:
  about two agent-days.
- **Source-reading tests outside the command.** Four local-name lint tests in
  `tests/unit/test_firstiter_cc_batch_budget.py` (engine cores) cannot be expressed as behaviour. Engine
  pins (k_class, sparse_pass2, significance, scripts) are outside this package.

## Decided

- **The final all-data scoring loop is one contract (rule 6).** It scores each half with the final sampling;
  K selects a variant (the Class3D pass-1 coarse size, `DenseVariantPolicy.k_class_enabled`, best-pose
  details, the profile label). Splitting it would duplicate about 440 lines for four one-line differences.
  The final reconstruction already makes the one mode decision with a block per mode. (Owner, 2026-10-05.)
- **`DenseVariantPolicy.k_class_enabled` stays a variant of one operation.** `half_scoring` branches on it
  twice below the controller; revisit only if K=1 and Class3D need different operands there. (Owner,
  2026-10-05.)
- **The line ceilings are 71 lines above their value, inside the slack.** The growth is the records and
  docstrings of the functions extracted from `main`; the next extraction should delete before it adds.
