# What in relax/refinement still falls short of the rules

The agent-facing record of the open items of the refinement package against
[refactor_rules.md](refactor_rules.md) and [module_template.md](module_template.md). Close an item by
deleting it here in the commit that closes it; record a decision in the "Decided" section. Numbers are of
main 7d8a1a1 (2026-10-05).

## Open, in the order they are worked

1. **The command translates flags into options inline (rules 5, 7).** `full_refinement.main` is 1,103
   lines. Every concern it had is a call with its own tests (start-up maps, prior, noise, half sets, replay
   inputs, follower routing, captured projector, restart provenance, reports). What remains inline is the
   order of those calls and about 200 `args.` reads turned into the controller's option records; the
   `RefinementOptions(...)` call alone is 170 lines. Plan: one resolver per option group in
   `command_options` (schedule, parity, replay, local search, k_class, ...) with a test of its fields, so
   `main` passes records, not flags.
2. **Environment reads below the boundary (rule 5).** 17 reads in `relax/refinement/` outside the command
   modules: dump directories (`RELAX_KCLASS_DUMP_DIR`, `RELAX_BPREF_ACCUM_DUMP_DIR`,
   `RELAX_RELION_TAU2_DEBUG_DUMP_DIR`, `RELAX_PREMASK_DUMP_DIR`, `RELAX_RELION_PROJECTOR_DUMP_DIR`,
   `RELAX_RELION_PROJECTOR_CACHE_DIR`), precision diagnostics (`RELAX_USE_FLOAT64_SCORING`,
   `RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS`), host FFT workers (`RELAX_RELION_HOST_IRFFT`,
   `RELAX_RELION_HOST_FFT_WORKERS`, `SLURM_CPUS_PER_TASK`), the x-half batch guard and the native softmask
   switch. Each becomes a field of `EngineDebugOptions` or of the precision policy, read once at the command.
3. **The controller loop (rule 6, shared steps).** `refine_single_volume` spans 1,998 lines, 847 of them
   shared by both modes. Remaining work is to find contract splits or shared steps by the rules, not by line
   count. 29 functions take ten or more parameters; the widest are `run_final_all_data` (28),
   `score_tomo_half` (26) and `build_archive_metadata` (25).

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
