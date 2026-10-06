# relax/vdam against the rules: verdict, exceptions and what remains

The agent-facing record of the open items of the VDAM InitialModel package (`relax/vdam`, the gradient
optimiser behind `relax initial_model` and the PPCA start) against [refactor_rules.md](refactor_rules.md)
and [module_template.md](module_template.md). Close an item by deleting it here in the commit that closes
it; record a decision in the "Decided" section. Numbers are of main on 2026-10-06 after the refactor/vdam2 series.

## Verdict (2026-10-06)

**`relax/vdam` meets the code rules for its controller (the loop and the driver), with the recorded
exceptions below; rules 10 and 11 are partly met.** Each exception names its reason.

| Rule | Status |
| --- | --- |
| 1 numbers, 2 interfaces | Met: every move was fingerprint-identical (`scripts/dev/vdam_fingerprint.py`, 23 cases) and passed the GPU tiers. Three intended changes in failure paths, each its own commit with differences confined to its case: retired switches and `RELAX_USE_FLOAT64_SCORING` now refuse instead of being ignored, and a VDAM schedule with a NaN tau2 fudge (4 or 5 iterations) refuses (item 7). `run_native_options.json` keeps its format. |
| 3 lifetimes | Met: no buffer changed owner. `EstepSums` holds host sums only (kilobytes); the accumulators stay a list released after the M-step. |
| 4 JAX | Met: no jit region in the package. Recorded: `m_step._DEVICE_SOLVENT_MASK`, a module-global device mask kept for the same host mask object (one box-sized array for the process lifetime). |
| 5 decide once | Met for the switches the controller and the E-step configuration read: `NativeInitialModelOptions.environment` (`VdamEnvironment`: profile, cache clearing, the two expected-accuracy switches, `adaptive_fraction`, `subtract_projected_reference`; read when the options are built). Exceptions: item 1. |
| 6 contracts | Met: the optimizer is decided once (`VdamUpdate` or `MomentumSgdUpdate`, built by the driver). The SPA/subtomogram split of the E-step closure is one `if tomo:` with a block each (accepted, as refinement's final reconstruction). `mstep_compute_dtype` and `uniform_class_direction_prior` are variants of one operation. |
| 7 one owner | Met, with one exception: `do_grad` is computed by the loop and again by the E-step closure from the same state (item 2). |
| 8 config/state/results | Met for the model update: the E-step's operands are a frozen `EstepSums`; the driver returns `NativeInitialModelResult`. Exception: the E-step callback still returns `(accumulators, meta)`, with `meta` both its transport and its report (item 3). |
| 9 transitions | Met in the loop: model-state installs are statements of `run_vdam_iterations`; the sampling-counter install is `record_iteration`, no longer the output sink. Exceptions: item 3 (the E-step closure updates `particle_state` and `sampling_state` in place), and `relax.sgd_initial_model` reads and writes `meta`. |
| 10 signatures | Partly: `run_vdam_iterations` takes 30 parameters, as before: three optimizer arguments became `update`, and `environment` and `record_iteration` were added (item 4). |
| 11 layers | Partly: item 5. |
| 12 edges | Met: the unreachable and test-only fallbacks are gone (`retained_fallback`, the Pmax normalisation chain, the defaulted offset weight, partial noise sums, the missing M-step class mass); switches nothing honours refuse; a NaN tau2-fudge schedule refuses at the options edge (item 7). Accepted: "no sums at all keeps the model" (an empty subset), documented on `EstepSums`. |
| 13 tests | Met for the moves: the one ownership test whose names moved was converted (identity and a clean-import subprocess). Exception: `tests/unit/initial_model/test_refactor_invariants.py` still reads source (`inspect.getsource`) for ownership pins of code this refactor did not touch (item 6). |
| 14 evidence | Met: each commit lists its fingerprint result; coverage limits below. |

## Detail

1. **Environment reads that stay where they are used (rule 5).** Each is a diagnostic dump or a parity
   hook owned below the controller:
   - `dense_adapter._finish_relion_projector_class_inputs`: `RELAX_INITIAL_MODEL_PROJECTOR_DUMP_DIR`
     (called through the projector context, which has no options);
   - `estep_common._arrays_to_accumulators`: `RELAX_INITIAL_MODEL_ACCUM_DUMP_DIR`;
   - `native_sampling._estimate_native_sampling_accuracy`: the expected-accuracy dump directory and
     iterations;
   - `bootstrap_iref`: `RELAX_INITIAL_IREF_OVERRIDE` (replaces the start-up reference with RELION's);
   - `driver`: `RELAX_INITIALMODEL_IREF_REPLAY_TEMPLATE`, owned and read by
     `relax.diagnostics.vdam_mstep_replay`; the driver only refuses it with a float32 M-step;
   - `mstep_single_class._validate_mstep_precision_route`: refuses the native parity replay variables,
     which belong to `relax.diagnostics.vdam_native_mstep`;
   - `output._write_initial_run_metadata`: records the CUDA allocator and JAX cache variables as provenance.
   Moving them into `VdamEnvironment` costs plumbing through the projector context and the accuracy
   helpers. Not an exception: `RELAX_USE_FLOAT64_SCORING`, read once by `VdamEnvironment` only to refuse it.
2. **`do_grad` twice (rule 7).** The loop decides it before the schedule update; the E-step closure
   recomputes it with `schedules._native_initialmodel_do_grad` from the same `state` (same value: the
   iteration and `has_converged` do not change in between). Passing it needs the E-step callback contract
   to change (item 3).
3. **The E-step callback (rules 8, 9).** `ExpectationStepFn` returns `(accumulators, meta)`; the loop
   builds `EstepSums` from `meta` at once (`estep_common.estep_sums`, which refuses partial sums), and the
   rest of `meta` is the iteration's report. The closure built by `driver._native_expectation_step`
   updates `particle_state` (`_update_particle_state_from_estep_meta`) and `sampling_state` (the sampling
   plan and the change monitor) in place before it returns: run-state installs inside a callback. The
   next step is an E-step result type `(accumulators, sums, report)` and installs returned to the driver;
   about 20 test stubs return the tuple today.
4. **`run_vdam_iterations` has 30 parameters (rule 10).** Candidate record: the gradient schedule
   (`grad_ini_subset_size`, `grad_fin_subset_size`, `grad_ini_frac`, `grad_fin_frac`, `phase_lengths`,
   `grad_em_iters`, `tau2_fudge_arg`, `grad_stepsize`, `mu`), one concept of run lifetime. Other wide
   functions: `compute_bootstrap_iref` (22), `run_tomo_initial_model_estep` (16, engine boundary),
   `bootstrap_references` (16), `compute_subset_size` (13, RELION's arguments one for one).
5. **Layers (rule 11).** Production imports private diagnostics (`driver` imports
   `diagnostics.vdam_mstep_replay._maybe_replay_iteration_references`; `estep_meta_updates` imports
   `diagnostics.vdam_noise`), refinement internals (`refinement.optics_shapes`, `refinement.tomo_half`,
   `sparse_pass2.resident_pass2.stable_window_class_history`) and private names of
   `relion.initial_model_io`. `relax.sgd_initial_model` imports `relax.vdam.state`, and the loop imports
   it back (lazily, inside `MomentumSgdUpdate`). `relion_solvent_mask` has copies in
   `relax/relion/reference_initialization.py` and `relax/relion/initial_noise.py` (one formula, three
   homes).
6. **Source-reading tests (rule 13).** `test_refactor_invariants.py` pins owners with `inspect.getsource`
   (`_relion_round`, the M-step, sampling and adapter ownership tests); they pin code this refactor did not
   move, so they were kept. Its line budgets count file lines; they follow the slack policy (owner ruling
   2026-10-05) since adcb6c00. Budget warnings at this head: controller 2044 lines, above 1977 and within
   its slack 99 (`VdamEnvironment`, the two update records, `record_iteration` and the tau2 refusal, +67);
   input_output 1387, above 1385 and within 70 (the writers take the profile switch, +2). No budget was raised.
7. **The NaN tau2 fudge (rule 12, closed 2026-10-06).** With 4 or 5 iterations RELION's tau2-fudge sigmoid
   has length 0 and the fudge at iteration 1 is 0/0. It reached arithmetic: the M-step's data_vs_prior was
   NaN in every shell and the resolution update read it (the reference stayed finite through the FSC clamp).
   `NativeInitialModelOptions.validate_run` refuses such a VDAM schedule; momentum SGD and the diagnostic
   continuation are not refused. RELION itself runs the 0/0.

## Known coverage limits

- `vdam_fingerprint` stands in for the E-step engine (seeded by its operands). Not covered on a CPU:
  subtomograms (`--ios`), optics groups on several image shapes, the diagnostic optimiser continuation,
  RELION's CUDA image preprocessing; the GPU tiers (`vdam_k1_50k` in medium) cover the real engine on
  the default route only.
- `scripts/dev/refactor_cpu_unit_list.txt` holds 15 of the 85 files of `tests/unit/initial_model` and 6 of
  `tests/unit/ppca_initial_model`; the gate of this series ran both directories in full on the base and
  the head as an extra step.

## Decided

- **The optimizer is two update records, not a flag (rule 6, 2026-10-06).** The loop keeps the order of
  installs (M-step, priors, noise, Pmax, post-M-step hook, resolution) and calls the record at each
  point; SGD's resolution is its radius schedule.
- **Environment switches (owner, 2026-10-06).** `RELAX_HALF_SPECTRUM_SCORING`, `RELAX_SQUARE_WINDOW` and
  `RELAX_RECON_SQUARE_WINDOW` (dead: the E-step already scores the half spectrum, and the window keys are not
  forwarded to the engine), `RELAX_RANDOM_PERTURBATION` (its refusal names `--random-perturbation`) and
  `RELAX_INITIAL_MODEL_EXACT_RELION_PROJECTOR` stop relax at import (`relax/renamed_environment.json`).
  `RELAX_ADAPTIVE_FRACTION` and `RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE` stay as `VdamEnvironment`
  fields (`adaptive_fraction: float | None`, `subtract_projected_reference: bool`), with fingerprint cases.
- **Line budgets are review signals with slack (owner ruling, 2026-10-05).** `test_responsibility_loc_budget`
  uses the helpers of refinement's ceiling check; the budget numbers were not changed.
- **A NaN tau2-fudge schedule refuses (supervisor, 2026-10-06).** Item 7.
