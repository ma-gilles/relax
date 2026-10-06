# What in relax/refinement still falls short of the rules

The agent-facing record of the open items of the refinement package against
[refactor_rules.md](refactor_rules.md) and [module_template.md](module_template.md). Close an item by
deleting it here in the commit that closes it; record a decision in the "Decided" section. Numbers are of
the branch after main 7d8a1a1 (2026-10-05).

## Open, in the order they are worked

1. **The command still assembles three option records from start-up results (rules 5, 7).** The
   schedule, adaptive, batching, overlap, local-search, k_class and debug records are resolved in
   `command_options` (`resolve_*`, field tests in `tests/unit/test_option_resolvers.py`). The parity, replay
   and checkpoint records stay in `full_refinement.main`: their fields are start-up results (half sets,
   references, noise, poses, replay inputs, follower topology, the frozen boundary), not flags, so a
   resolver would be a function of ten or more of them. `main` is 1,020 lines.
2. **Environment reads below the boundary (rule 5).** The controller's diagnostic switches and dump
   directories are one `DiagnosticEnvironment` (`EngineDebugOptions.environment`), read when the options are
   built; add a new diagnostic variable there. Still read where used, each with its reason:
   - `projector_preparation.prepare_scoring_projector`: `RELAX_RELION_PROJECTOR_CACHE_DIR` and
     `RELAX_RELION_PROJECTOR_DUMP_DIR`; `relax/helpers/expected_accuracy.py` calls it too, outside this
     package.
   - `mean_helpers`: `RELAX_RELION_HOST_IRFFT`, `RELAX_RELION_HOST_FFT_WORKERS` and `SLURM_CPUS_PER_TASK`,
     execution-resource knobs of the host inverse FFT inside `_reconstruct_volume_eager` (18 parameters,
     several callers); threading them means a host-FFT policy record in `ReconstructionSettings`.
   - `local_search_iteration`: the x-half batch guard; `half_scoring`: hides the local engine's dump
     variables during the denominator pass; `expectation`: whether a BPref dump of the engines is armed. The
     variables belong to the engine packages, which read them.
   - `expectation_batches`: `RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS` (owned by
     `relax.helpers.dtype_policy`); `particle_loading`: `RELAX_USE_FLOAT64_SCORING` (the precision policy
     `relax.dense.scoring_policy.DENSE_PRECISION` reads it at import; particle loading runs at the command)
     and the RECOVAR native softmask switch, which is how the setting reaches RECOVAR.
3. **The controller loop (rule 6, shared steps).** `refine_single_volume` spans 1,992 lines. Its mode
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
   replace: the M-step results kept as records instead of unpacked into seven locals (`ClassMaximization`,
   `K1Maximization`); Class3D's class state (assignments, previous assignments, mixture); RELION's growth
   latch (`incr_size`, `has_high_fsc_at_limit`, which the planners and the run files take apart). 29
   functions take ten or more parameters; the widest are `run_final_all_data` (28), `score_tomo_half` (26)
   and `build_archive_metadata` (25).

## Known coverage limits (recorded, not being built)

- **RELION run directories have no CPU fixture.** The fingerprint's `main_*` cases do not reach the frozen
  boundary, the RELION replays (`--relion_init_dir`, `--perturb_replay_relion_dir`, the final-only replay),
  the state-swap probe or captured projectors: the tiny data cannot produce a RELION run directory. CPU
  tests stand in for the one callee that supplies such a fact (a stand-in boundary or follower topology);
  the paths themselves are checked by the GPU tiers only. Cost of closing it: a tiny run-directory fixture
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
