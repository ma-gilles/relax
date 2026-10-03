# Refinement refactor review: current calling flow

## Integrated controller ownership changes

The completed operation ownership is integrated with GitHub main
`80f2b2b36cee01d110276d189a3b061c78746c86`. The thirteen code/guide commits through
`69abd77` retain their changes; the subsequent commit adds PPCA measurement
documentation only. The incoming work includes
pass-1 runner-up, exact-operand, normalization and dump-schema changes,
PPCA work and tomography safeguards. Incoming command/controller changes are
adapted to the existing refactor owners. Scientific sequencing and visible
updates remain unchanged by these interface migrations.

The first-CC margin now resolves in `command_options.resolve_firstiter_controls`,
then enters the existing parity settings at the original command boundary.
Numbered expectation and final dense execution consume those settings; the
batch planner reads the same value. `None` disables rescoring; `0.0` is enabled.
K1 CLI `auto` still means 4e-6 with firstiter_cc, while K4 `auto` stays off.
Particle-format optics admission stays immediately after the input STAR is read;
tomography uses main's qualified-feature set and preserves tilt versus particle
metadata indexing.

Focused CPU verification executed 538 cases: 536 passed, and two newly extended
staging expectations assumed the compact coarse planner also applied to enabled
rescoring. Main uses the generic coarse planner in that case. The corrected
nine-case staging suite passed, covering off, zero and the default margin.
The failed full-run receipt is preserved; its repair changes no numerical
production AST or tolerance. Five GPU cases were deselected for fresh GPU tiers.
Whole-source import lint and native source/import checks passed. The six
principal prior, reconstruction, noise, planning and half-input numerical owners
are byte-identical to the preceding implementation.

Earlier frozen `0b46b2` smoke **14909958** passed: 466 GPU cases, 251 controller
cases and all three required parity replays. One optional operand test lacked
its exact-CTF STAR fixture. Its medium **14909959** remains on its original
immutable source. The queued K1 pair **14910610** is preserved under a root-owned
synchronization hold to avoid redundant production work. These jobs do not
qualify the incoming engine changes. The unchanged native-source digest permits
reuse of the already built immutable libraries; runtime library/GPU checks still
run before and after new qualification.

[Current integration, receipts, preserved jobs and required gates](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/refactor_finish_sync_69abd77_20261003/HANDOFF.json)
track the new candidate. Fresh same-source smoke/medium, float32 K1 and exactly
K4 quality, characterized real-data confirmation, ordinary paired memory/speed,
the milestone long tier and qualified main delivery remain required. No
production quality or performance acceptance is claimed.



This is the current complete responsibility map and the actual remaining-phase
caller/implementations. Older sections below are source-specific review history;
their old pending buffer-expiry choices were resolved by the October 2 approval.

The remaining numbered Class3D prior aggregation, regularized reconstruction,
postprocessing, first-CC reporting taper and paired row/noise/accuracy preparation
are integrated in their existing responsibility owners. The controller retains
mode/source admission, explicit scientific state installation, scheduling/history
publication and the transition to final all-data. Three obsolete reconstruction
entry points and their maintained direct/indirect callers are migrated.

Unused temporary expiry after last use is explicitly authorized by the user on
October 2. Selected scientific products retain their required storage; unused
class originals, full CTF tables and taper/admission scratch may expire. Existing
casts, ordered numerical calls, JIT/donation boundaries and final consuming solves
are preserved. Final prior/reconstruction orchestration stays separate because
its replay, CTF, precision and release policies differ; shared mathematical
primitives retain one implementation.

The [complete calling flow and implementations](final_local_sampling_patch_review.md#integrated-controller-ownership-changes)
show the scientific order, producer/consumer ownership and actual caller together.
Source spans are 2752/1766 for numerical/command controllers;
these counts are review signals, not design acceptance. Current CPU checks are recorded above; earlier passing receipts describe their
own source only. The milestone is incomplete until the frozen float32 K1/exactly-K4
scientific, real-data, memory and matched-GPU speed gates pass and delivery to main
is verified. Existing comments, frozen candidates, controls and jobs are preserved.

### Complete scientific execution order

| Phase | Owner of implementation | Decisions and writes visible in orchestration |
| --- | --- | --- |
| Command options, manifests and metadata admission | `command_options`, `frozen_boundary_cli` | CLI precedence, source/mode admission and resolved controls |
| Particle loading and paired row frames | `particle_loading`, `input_particle_table` | Source choice, fresh RNG-order policy, diagnostic half admission, two dataset subsets |
| References, noise and restart restoration | `reference_initialization`, `startup_noise`, snapshot/replay owners | Source precedence, active state/reference/noise installation and restored array ownership |
| Permitted loop boundary | `convergence`, existing state/sampling policy | Decisions from the preceding iteration, cap and entry into finalization; no extra boundary after the cap |
| Sampling/window/projector preparation | `iteration_planning`, `sampling`, `expectation`, `projector_preparation` | Current mode and windows, sampling identity and projector replacement/release |
| Expectation/accumulation | `expectation`, `half_scoring`, dense/local/tomo engines | Half-specific operands, serial/overlap mode, ordered result installation and memory boundary |
| Prejoin diagnostics and accumulator adaptation | `diagnostics.reconstruction`, `mean_helpers` | Audit before combine/join; K1 low-frequency join and previous-map snapshot/release |
| Prior estimation | `mean_helpers.estimate_class_priors` / `estimate_split_half_prior` | Class history/scheduling before detail stacking; explicit shared/per-half tau2 updates |
| Regularized reconstruction and postprocessing | `mean_helpers.reconstruct_numbered_k1_halfmaps` / `reconstruct_numbered_class_maps` | One K1/Class3D dispatch with mode-specific accumulators and shell-curve priors, old-map clear, ready-map installation and retained numerator release |
| First-CC reporting taper | `mean_helpers` plus existing curve taper | Untapered reconstruction first; K1 tau2 publication or Class3D curve/history/scheduling before shell/detail taper; host staging afterwards |
| Posterior/direction, noise and correction updates | Existing prior/noise/normalization owners | Single-reference copying, explicit half/model state and follower-before-report order |
| Resolution, convergence and output capture | Resolution/convergence/history/snapshot/diagnostic owners | Observed versus scheduling resolution, live replay controls, completed-state capture/checkpoint order |
| End-of-iteration boundary | Controller | Readiness boundary and release before the next projector; no RNG/reduction/batch changes |
| Final all-data | `finalization`, `final_sampling`, `final_reconstruction` | Explicit final transition and numbered/setup publication; ordered final accumulator consumption |

### Ownership, precision and lifetime decisions

The actual `HalfSet` producer pairs dataset and scale corrections. Numbered CTF
adaptation consumes those paired owners at the same phase as before. Raw pixel
metadata remains an explicit reporting-taper operand because its scalar type can
affect host promotion; normalized geometry is not substituted blindly. Run-level
cutoff/filter settings come from their original producers. No new module, JIT,
GPU pass, dtype default, host/device transfer, formula or numerical fallback is
introduced by this final source package.

`ClassPriorAggregation` contains only consumed variance/shell/curve/detail/source
products. Detail aggregation remains after curve publication. Each regularized
operation returns ready maps; no diagnostic consumes `ReferenceModel` while local
maps are being postprocessed. E-step workers have already joined. Required K1
solve donation/completion and final merged→half1→half2 consuming order remain.
Selected accuracy CTF rows are an independent float64 copy; unused full tables and
validation-only identity vectors now expire at their operation boundary. JAX may
retain asynchronous operands independently. These are approved lifetime changes,
not a claim of measured memory or speed improvement.

A maintainer changing class prior-source admission can inspect the producer and
prior owner without opening command I/O, score engines or noise updates. A mask or
filter change needs the reconstruction settings and `_postprocess_numbered_maps`,
the one capture, filter and flatten sequence that both numbered map operations apply. A
half-order change needs the particle-table producer and input operation. Changes
to scientific phase order still require the controller and these exact operands.

### Reconstruction choice inventory

Plan step 1 of the first package. Every choice that reaches the two numbered
operations or the private solves beneath them, traced at `ce9c064` (line
numbers are that commit's) over the supported entry paths: numbered SPA and
tomography iterations (one controller, no modality branch in this code),
continuation and replay-installed state, and finalization. "Constant" means
every production producer supplies one value, so the other arm is reachable
only from tests.

| Choice | Producers | Consumers | Kind | Verdict |
| --- | --- | --- | --- | --- |
| Split-half maps versus combined class stack | `k_class_enabled = n_classes > 1`, `iteration_loop.py:452`, never reassigned; accumulators combined at `:1953-1955` (class) or joined at `:1957` (K1) | The one dispatch `iteration_loop.py:2103`; `_reconstruct_k1_maps` `mean_helpers.py:1332`, `_reconstruct_class_maps` `:1387`; premask writer layout `diagnostics/reconstruction.py:353` | scientific | live: both modes are production |
| Prior as shell curves versus full volume, numbered operations | K1: `estimate_split_half_prior` returns `details["prior_shells"]` for both halves (`mean_helpers.py:1318-1325`, always present, `regularization_relion.py:731`). Class3D: `estimate_class_prior` returns shells from the Iref power spectrum (`:481`) or from the diagnostic replay spectrum (`:472`, admitted and shape-checked as `(n_classes, n_shells)` at `diagnostics/relion_replay.py:68-103`), stacked at `:646`. No reassignment between `iteration_loop.py:2019`/`:2074` and the call. K1 replay `mean_variance` (`:1140-1155`) installs a full volume into `reference_model.tau2`, the scoring state, which is overwritten at `:2086` and is never the solve operand. Continuation restores the model state, not this operand | `tau_is_1d` and the `shells if not None else volume` fallback at `iteration_loop.py:2107-2115`, `:2122-2129`; forwarded by both operations and both private solves (`mean_helpers.py:1369`, `:1423`) | input validation | constant in production (always shells); the full-volume arm is reached only through the test builder (`tests/helpers/refinement_specs.py:70,80`) |
| Prior as shell curves versus full volume, eager solve | Numbered solves (shells); final Class3D (`final_reconstruction.py:177`, shells); final K1 half maps pass the full-volume prior with the default flag (`final_reconstruction.py:210-216`, from `compute_final_halfmap_prior`); unregularized and unfiltered solves pass no prior (`mean_helpers.py:1229`, `:1720`, `:1747`, `final_reconstruction.py:49`); four scripts | `_reconstruct_volume_eager` `mean_helpers.py:729`, stable reconstruction class `:686`, Stage A `:887`, `relion_functions_relion.py:234` | scientific operand layout | live: final K1 is a production full-volume caller, so the eager solve keeps the parameter |
| First-CC initial low-pass of the new references | `relion_firstiter_cc_this_iter`, `iteration_loop.py:963-965` (first numbered iteration of a fresh run with CC emulation); resolution from `parity.relion_firstiter_ini_high_angstrom` (`:524`) | Filter before flatten in both operations (`mean_helpers.py:1486`, `:1567`); closing log (`:1513`, `:1602`); reporting taper after the solve stays in the controller (`iteration_loop.py:2145-2202`) | scientific | live: iteration-dependent |
| Solvent flatten | `schedule.particle_diameter_ang`, a Python float or `None` (`full_refinement.py:591`, `scripts/run_multi_iter_parity.py:1617`); voxel size from `ImageGeometry`, a Python float (`helpers/resolution.py:36-39`, `iteration_loop.py:515`) | Radius and mask in both operations (`mean_helpers.py:1494-1507`, `:1581-1592`) | scientific | live. The K1 `float()` wrapping versus the bare class expression is not a case: the two differ only for `np.float32` scalars, which no production producer supplies |
| Flatten implementation | Mode | K1 `_apply_relion_solvent_flatten_k1` (`mean_helpers.py:2182`); Class3D inline iDFT, multiply, DFT (`:1594-1601`) | execution-storage | live and not interchangeable: at box scale the K1 implementation moves the result to the host and deletes the mask, which the class loop reuses for the next class |
| Host staging and large-box behaviour | Accumulator and box size: `_should_host_stage_large_relion_ifft` `mean_helpers.py:1897`; `_large_relion_solvent_mask_uses_compiled_builder` `:2118`; retained half-0 numerator from the low-resolution join (`iteration_loop.py:1957`) | Eager solve `:777-`; K1 host completion `:1379` and retained numerator `:1372-1383`; mask builder `:2150`; K1 flatten `:2193`; K1 mask handle release `:1511` | execution-storage | live: size-dependent, K1 only for completion, retained numerator and flatten staging |
| Premask capture | `RELAX_PREMASK_DUMP_DIR` read per slot (`mean_helpers.py:1472`, `:1553`) | `write_premask_mean` `diagnostics/reconstruction.py:341`, once per slot after all solves | diagnostic | live; the class operation writes the same stack for both slots |
| Class prior replay | `RELAX_KCLASS_REPLAY_TAU2` and the replay override (`diagnostics/relion_replay.py:68`) | `estimate_class_priors` `mean_helpers.py:568`, `:613` | diagnostic | live; changes the prior values, not their representation |
| Class operation returns two execution slots | `means = [shared_classes, shared_classes]` `mean_helpers.py:1548` | Both slots captured, filtered and flattened; the controller later re-aliases slot 1 to slot 0 | execution-storage | live; removal is deferred to the ownership package |
| Final versus numbered reconstruction | `finalization.py:606`, `:654`, `:678`, `:736`, `:757` | `final_reconstruction.py:38`, `:152`, `:190` call the eager solve directly | scientific | live and separate: finalization never calls the numbered operations |

What the package removed on this evidence, in the code shown below:

- `tau_is_1d` and the `shells if not None else volume` selection are gone from
  both numbered operations, both call sites and both private solves, whose only
  callers are those operations. The eager solve keeps the flag for final K1.
- The capture, filter and flatten sequence, the mask radius and the mask
  construction exist once, in `_postprocess_numbered_maps`. Its `class_axis`
  argument says whether a slot is one flat map or a class stack; it selects
  per-class filtering, the mask dtype source and the flatten implementation.
  The two flatten implementations stay distinct for the reason in the table.
  Solve, capture, filter, mask and flatten order, the mask handle's lifetime and
  the closing log are those of `ce9c064`.
- `ReconstructionSettings` converts `voxel_size` and
  `particle_diameter_angstrom` to Python floats when it is built, so the radius
  has one formula. Production arithmetic is unchanged; a caller that injects
  `np.float32` scalars now gets the double-precision radius in both modes.
- The `run_mean_reconstruction` test builder is removed; tests call the
  operation they exercise with its own operands.

Still present: the class operation postprocesses both execution slots, and
finalization repeats its K1 guards (plan step 4).

### Finalization mode-guard map

Plan step 4 of the first package. Every test of the K1/Class3D mode on the way
into and through finalization, traced at `ea003d7` (line numbers are that
commit's). `final_reconstruction.py` holds no mode test: each of its five
functions serves one mode and the caller chooses. `F` is `finalization.py`,
`L` is `iteration_loop.py`.

Reconstruction part of `run_final_all_data`, in execution order:

| Line | Test | What it guards | Kind |
| --- | --- | --- | --- |
| `F:591` | `not k_class_enabled` | Unfiltered half maps from the pre-join accumulators: two eager solves, each result moved to the host | K1 pre-join sequence |
| `F:613` | `not k_class_enabled and` join resolution set and positive | Low-resolution join in place (`preserve_inputs=False`), rebinding the four half locals | K1 pre-join sequence; the resolution test is an option, not the mode |
| `F:626` | `not k_class_enabled` | Clears the four accumulator slots of the pass collector `final_outs` | K1 pre-join sequence |
| `F:639` / `:676` | `if k_class_enabled` / `else` | Class3D: class weights, history record, previous-reference power prior, log. K1: prior from the joined halves' FSC, log | scientific policy |
| `F:704-708` | conditional expression | Data-versus-prior operand of the final resolution | scientific policy |
| `F:711` | value passed on | Resolution shell rule | scientific policy |
| `F:735` / `:748` | `if k_class_enabled` / `else` | Class3D: class maps, weighted merged map, final class assignments. K1: backprojection list, release of the six locals, consuming merged, half 1, half 2 solves | scientific policy |
| `F:790-791`, `:806` | conditional expressions, value passed on | Class fields of the result | result layout |

Nothing executes between the three K1 pre-join guards: no log, write or
release separates the unfiltered solves, the join and the collector clearing.
Shared work follows them at `F:632-638` (merged sums, `final_iter_fsc = None`,
half-axis metadata) and again at `F:699-734` (resolution update, two logs)
between the prior and the final solve, so those two mode decisions cannot join
the first without reordering or duplicating shared statements.

Before the reconstruction part: `F:157` defines the mode from `n_classes`;
`F:178` expected-accuracy class ids; `F:265` adaptive class pass-1 window;
`F:363`, `F:374` dense scoring variant and pose details; `F:535` profile
record. `F:519` tests `best_pose_translations is not None`, which the dense
class route leaves unset (`F:374`); it is a check on an engine result, not a
mode test. Admission and replay in the controller: `L:2852` into
`_should_run_final_all_data_iteration` (`F:94` rejects Class3D after the cap);
`L:2866`, `:2879-2880` class fields when no final pass runs; `L:2915` merged
reference diagnostic (K1); `L:2935` final reference substitution;
`L:2995-3025` replayed direction-prior layout; `L:3037`, `L:3055` logs;
`L:3051` `final_use_local = not k_class_enabled and state.do_local_search`.

| Entry path | Reaches the final pass | Guards exercised |
| --- | --- | --- |
| K1 native convergence at the top of a permitted iteration | yes | all three K1 pre-join guards; the join runs when `low_resol_join_halves_angstrom` (default 40) is positive; K1 arms of prior and solve; local or dense scoring from the converged state |
| Fixed iteration count exhausted, either mode | no (`L:2854` returns the numbered maps) | none |
| K1 after the cap with `RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER=1` | yes, diagnostic | as K1 native convergence |
| Replay-installed convergence (`relion_replay.py:964-999`) | yes, either mode | K1: as above. Class3D: as the next row |
| Class3D diagnostic final pass (injected or replayed convergence) | yes | skips the three K1 pre-join guards, including a positive join resolution; Class3D arms of prior and solve; dense scoring only |
| Tomography | K1 with local search only (`F:417` raises otherwise, so no Class3D) | as K1 native convergence; the intermediate capture at `F:543-579` precedes this code |

### Actual numbered prior, map and reporting flow

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 1992):

```python
        if k_class_enabled:
            class_priors = estimate_class_priors(
                previous_means,
                Ft_y_combined,
                Ft_ctf_combined,
                reconstruction_settings,
                half_denominators=(Ft_ctf_0, Ft_ctf_1),
                prior_tau2=reference_model.tau2,
                halves=halves,
                n_classes=n_classes,
                iteration=iteration,
                current_size=current_size,
                image_current_size=image_current_size,
                accumulator_shape=mstep_accumulator_shape,
                full_half_axis=mstep_full_half_axis,
                projector_power_spectrum=(
                    None
                    if not has_previous_iteration or projectors[0] is None
                    else projectors[0].power_spectrum
                ),
                iter_replay_override=iter_replay_override,
                replay=replay,
                scoring_dtype=scoring_dtype,
                started_at=_t_unreg_first,
                log=logger,
            )
            mean_signal_variance = class_priors.variance
            mean_signal_variance_shells = class_priors.shells
            data_vs_prior_iter = class_priors.data_vs_prior
            tau2_update_details_per_class = class_priors.details_per_class
            kclass_tau2_source = class_priors.source
            del class_priors
            history.data_vs_prior_trajectory.append(data_vs_prior_iter)
            previous_data_vs_prior_for_scheduling = data_vs_prior_iter
            tau2_update_details = _stack_class_tau2_update_details(tau2_update_details_per_class)
            del tau2_update_details_per_class
            logger.info(
                "Computed iter-%d Class3D tau2 from %s: %.1fs",
                iteration + 1,
                kclass_tau2_source,
                time.time() - _t_unreg_first,
            )
        else:
            mean_signal_variance_shells = None
            # Optional dump of post-join Ft_y, Ft_ctf for shell-by-shell parity
            # comparison against RELION's RELAX_MSTEP_DUMP_DIR. Activated by
            # RELAX_BPREF_ACCUM_DUMP_DIR. One npz per iteration.
            _bpref_accum_dir = os.environ.get("RELAX_BPREF_ACCUM_DUMP_DIR")
            if _bpref_accum_dir and _bpref_boundary_iteration_matches:
                reconstruction_diagnostics.write_bpref_accumulators(
                    _bpref_accum_dir,
                    stage="accum",
                    iteration=iteration,
                    current_size=current_size,
                    padding_factor=RECONSTRUCTION_PADDING_FACTOR,
                    grid_size=grid_size,
                    voxel_size=source_pixel_size_angstrom,
                    volume_shape=volume_shape,
                    accumulator_shape=mstep_accumulator_shape,
                    Ft_y_0=Ft_y_0,
                    Ft_y_1=Ft_y_1,
                    Ft_ctf_0=Ft_ctf_0,
                    Ft_ctf_1=Ft_ctf_1,
                )
            split_prior = estimate_split_half_prior(
                (Ft_y_0, Ft_y_1),
                (Ft_ctf_0, Ft_ctf_1),
                reconstruction_settings,
                current_size=current_size,
                accumulator_shape=mstep_accumulator_shape,
                full_half_axes=per_half.mstep_full_half_axis,
                do_solvent_fsc_correction=parity.do_solvent_fsc_correction,
                pixel_size_angstrom=source_pixel_size_angstrom,
                iteration=iteration,
                scoring_dtype=scoring_dtype,
                started_at=_t_unreg_first,
                log=logger,
            )
            current_iter_fsc = split_prior.fsc
            tau2_fsc_for_update = split_prior.fsc_for_update
            mean_signal_variance = split_prior.variance
            mean_signal_variance_per_half = split_prior.variance_per_half
            mean_signal_variance_shells_per_half = split_prior.shells_per_half
            tau2_update_details_per_half = split_prior.details_per_half
            # Diagnostics follow the half-1 model.star, matching the parity report.
            tau2_update_details = tau2_update_details_per_half[0]
            del split_prior
            logger.info(
                "tau2 update from THIS-iter FSC: old_max=%.4e new_max=%.4e half_max=(%.4e, %.4e)",
                float(jnp.max(jnp.abs(reference_model.tau2))),
                float(jnp.max(jnp.abs(mean_signal_variance))),
                float(jnp.max(jnp.abs(mean_signal_variance_per_half[0]))),
                float(jnp.max(jnp.abs(mean_signal_variance_per_half[1]))),
            )
        reference_model.tau2 = mean_signal_variance
        if not k_class_enabled:
            reference_model.tau2_per_half = _updated_mean_variance_per_half(
                reference_model.tau2,
                mean_signal_variance_per_half,
                use_per_half_mean_variance=parity.use_per_half_mean_variance,
            )
        else:
            reference_model.tau2_per_half = [reference_model.tau2, reference_model.tau2]

        # --- Free previous-iteration means to reclaim GPU memory ---
        # (previous_means already snapshotted earlier for FSC sign alignment)
        for k in range(2):
            reference_model.maps[k] = None

        # --- Now reconstruct the regularized means ---
        _t_recon = time.time()
        if k_class_enabled:
            reference_model.maps[:] = reconstruct_numbered_class_maps(
                Ft_y_combined,
                Ft_ctf_combined,
                mean_signal_variance_shells,
                reconstruction_settings,
                n_classes=n_classes,
                iteration=iteration,
                current_size=current_size,
                accumulator_volume_shape=mstep_accumulator_shape,
                relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
            )
        else:
            reference_model.maps[:] = reconstruct_numbered_k1_halfmaps(
                (Ft_y_0, Ft_y_1),
                (Ft_ctf_0, Ft_ctf_1),
                mean_signal_variance_shells_per_half,
                reconstruction_settings,
                iteration=iteration,
                current_size=current_size,
                accumulator_volume_shape=mstep_accumulator_shape,
                relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                retained_first_numerator=retained_Ft_y_0_device,
            )
        logger.info(
            "Regularized reconstruction (2 halves + flatten): %.1fs",
            time.time() - _t_recon,
        )
        retained_Ft_y_0_device = None


        # RELION reconstructs the first-iteration CC maps with the untapered
        # updateSSNRarrays tau2.  Only afterwards does
        # initialLowPassFilterReferences taper tau2/data_vs_prior for the
        # model state and reporting; that tapered spectrum is explicitly not
        # used in the reconstruction calculation (ml_optimiser.cpp:5296-5328).
        if (
            not k_class_enabled
            and relion_firstiter_cc_this_iter
            and parity.relion_firstiter_ini_high_angstrom is not None
        ):
            tapered_prior = taper_first_cc_k1_prior(
                mean_signal_variance_per_half,
                tau2_update_details_per_half,
                reconstruction_settings,
                pixel_size_angstrom=source_pixel_size_angstrom,
                scoring_dtype=scoring_dtype,
            )
            mean_signal_variance = tapered_prior.variance
            mean_signal_variance_per_half = tapered_prior.variance_per_half
            tau2_update_details_per_half = tapered_prior.details_per_half
            del tapered_prior
            reference_model.tau2 = mean_signal_variance
            reference_model.tau2_per_half = _updated_mean_variance_per_half(
                reference_model.tau2,
                mean_signal_variance_per_half,
                use_per_half_mean_variance=parity.use_per_half_mean_variance,
            )
            tau2_update_details = tau2_update_details_per_half[0]
            logger.info(
                "RELION iter-1 CC emulation: tapered post-reconstruction tau2/data-vs-prior "
                "with ini_high=%.2f A",
                float(parity.relion_firstiter_ini_high_angstrom),
            )
        elif relion_firstiter_cc_this_iter and parity.relion_firstiter_ini_high_angstrom is not None:
            # Class3D tapers each class's tau2_class and data_vs_prior_class the
            # same way (ml_optimiser.cpp:6389-6420). RELION's comment calls this
            # output only, but the next E-step gates each class's scale sums on
            # data_vs_prior_class > 3 (:10473), so the untapered curve let
            # shells past ini_high into iteration 2's scale correction. The
            # class tau2 volumes are recomputed from the Iref power next
            # iteration, so only the shell curves carry the taper.
            data_vs_prior_iter = _firstiter_cc_ini_high_tapered(
                data_vs_prior_iter,
                grid_size,
                source_pixel_size_angstrom,
                parity.relion_firstiter_ini_high_angstrom,
                filter_edgewidth=REFERENCE_FILTER_EDGE_SHELLS,
            )
            history.data_vs_prior_trajectory[-1] = data_vs_prior_iter
            previous_data_vs_prior_for_scheduling = data_vs_prior_iter
            tapered_prior = taper_first_cc_class_prior(
                mean_signal_variance_shells,
                tau2_update_details,
                reconstruction_settings,
                pixel_size_angstrom=source_pixel_size_angstrom,
            )
            mean_signal_variance_shells = tapered_prior.shells
            tau2_update_details = tapered_prior.details
            del tapered_prior
            logger.info(
                "RELION iter-1 CC emulation: tapered Class3D tau2/data-vs-prior with ini_high=%.2f A",
                float(parity.relion_firstiter_ini_high_angstrom),
            )
        if not k_class_enabled:
            # The K=1 tau2 volumes are read again only by the next M-step (the
            # resident E-step does not use them). Keep them on the host between
            # uses, as RELION keeps tau2 as a host spectrum: at box 800 the four
            # float32 volumes are 8 GB of the device floor (GPU census, bigbox
            # 14480607).
            (
                reference_model.tau2,
                reference_model.tau2_per_half,
                mean_signal_variance,
                mean_signal_variance_per_half,
            ) = _host_tau2_volumes(
                reference_model.tau2,
                reference_model.tau2_per_half,
                mean_signal_variance,
                mean_signal_variance_per_half,
            )
        _parity_dump.mark_stage(iteration, "recon")
```

### Complete numbered Class3D prior result and operation

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 514):

```python
class ClassPriorAggregation:
    """Class-axis priors, scheduling curves and detail rows for ordered publication.

    Variance and shell stacks use the RECOVAR frame. The controller publishes
    the data-vs-prior curve before aggregating detail rows, preserving RELION's
    numbered M-step order. No reference maps or unused class scratch are held.
    """

    variance: object
    shells: object
    data_vs_prior: np.ndarray
    details_per_class: list[dict]
    source: str
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 529):

```python
def estimate_class_priors(
    previous_half_maps,
    combined_numerators,
    combined_denominators,
    settings: ReconstructionSettings,
    *,
    half_denominators,
    prior_tau2,
    halves,
    n_classes,
    iteration,
    current_size,
    image_current_size,
    accumulator_shape,
    full_half_axis,
    projector_power_spectrum,
    iter_replay_override,
    replay,
    scoring_dtype,
    started_at,
    log,
) -> ClassPriorAggregation:
    """Prepare and aggregate numbered Class3D priors from the previous Iref.

    Own diagnostic replay admission, premultiplied-CTF adaptation, ordered class
    computation/capture and class-axis reductions. The controller installs the
    products and applies first-CC taper only after regularized reconstruction.
    See ``docs/math/relion_refinement_algorithm.md`` for the M-step ordering.
    """
    tau2_update_details_per_class = []
    mean_signal_variance_per_class = []
    mean_signal_variance_shells_per_class = []
    data_vs_prior_per_class = []
    # Dense RECOVAR accumulators live in the historical unnormalised
    # image frame: RELION BPref weight = Ft_ctf * N^4. Equivalently,
    # keep Ft_y/Ft_ctf in RECOVAR frame and scale RELION tau2 by N^4
    # before the Wiener solve. The same frame conversion is documented
    # in docs/math/ab_initio_initial_model_algorithm.md.
    kclass_tau2_frame_scale = float(settings.grid_size) ** 4
    replay_class_tau2, replay_tau2_enabled, kclass_tau2_source = replay_policy._class_tau2_replay(
        iteration=iteration,
        n_classes=n_classes,
        iter_replay_override=iter_replay_override,
        replay=replay,
        logger=log,
    )
    if iteration == 0:
        mean_variance_arr = jnp.asarray(prior_tau2)
        expected_shape = (n_classes, int(np.prod(settings.volume_shape)))
        if tuple(mean_variance_arr.shape) == expected_shape:
            log.info(
                "Class3D initial per-class tau2 volume available at iter=%d with shape=%s; "
                "M-step tau2 is recomputed from previous Iref power spectra",
                iteration + 1,
                tuple(mean_variance_arr.shape),
            )
    # CTF-premultiplied images: RELION's average CTF^2 correction of data_vs_prior
    # (setAverageCTF2; Class3D has no split halves and does not fix tau2).
    average_ctf2 = relion_ctf.premultiplied_average_ctf2(
        [half.dataset for half in halves],
        [half.scale_corrections for half in halves],
        image_current_size,
        settings.grid_size,
    )
    for class_idx in range(n_classes):
        log.info(
            "Class3D tau2 update start: iter=%d class=%d/%d current_size=%d source=%s spectrum=%s",
            iteration + 1,
            class_idx + 1,
            n_classes,
            int(current_size),
            kclass_tau2_source,
            "host transform" if projector_power_spectrum is None else "scoring projector",
        )
        class_prior = estimate_class_prior(
            previous_half_maps[0],
            combined_denominators,
            class_index=class_idx,
            settings=settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            full_half_axis=full_half_axis,
            frame_scale=kclass_tau2_frame_scale,
            projector_power_spectrum=projector_power_spectrum,
            replay_tau2_shells=replay_class_tau2 if replay_tau2_enabled else None,
            average_ctf2=average_ctf2,
        )
        mean_signal_variance_per_class.append(class_prior.variance)
        mean_signal_variance_shells_per_class.append(class_prior.shells)
        data_vs_prior_per_class.append(class_prior.data_vs_prior)
        tau2_update_details_per_class.append(class_prior.details)
        _kclass_dump_dir = os.environ.get("RELAX_KCLASS_DUMP_DIR")
        if _kclass_dump_dir:
            reconstruction_diagnostics.write_class_mstep(
                class_prior,
                numerators=combined_numerators,
                denominators=combined_denominators,
                half_denominators=half_denominators,
                references=previous_half_maps,
                settings=settings,
                output_dir=_kclass_dump_dir,
                class_index=class_idx,
                current_size=current_size,
                iteration=iteration,
                source=kclass_tau2_source,
                accumulator_shape=accumulator_shape,
                full_half_axis=full_half_axis,
                frame_scale=kclass_tau2_frame_scale,
            )
        log.info(
            "Class3D tau2 update done: iter=%d class=%d/%d elapsed=%.1fs",
            iteration + 1,
            class_idx + 1,
            n_classes,
            time.time() - started_at,
        )
    mean_signal_variance = jnp.stack(mean_signal_variance_per_class, axis=0)
    mean_signal_variance_shells = jnp.stack(mean_signal_variance_shells_per_class, axis=0)
    data_vs_prior_iter = np.stack(
        [np.asarray(dvp, dtype=scoring_dtype) for dvp in data_vs_prior_per_class],
        axis=0,
    )
    return ClassPriorAggregation(
        variance=mean_signal_variance,
        shells=mean_signal_variance_shells,
        data_vs_prior=data_vs_prior_iter,
        details_per_class=tau2_update_details_per_class,
        source=kclass_tau2_source,
    )
```

### Complete regularized reconstruction and reporting tapers

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1151):

```python
class ReconstructionSettings:
    """Run-level geometry, regularization, mask and initial-filter settings."""

    grid_size: int
    voxel_size: float
    volume_shape: tuple
    padding_factor: int
    projection_padding_factor: int
    minres_map: int
    width_mask_edge: int
    fmask_edge: int
    tau2_fudge: float
    particle_diameter_angstrom: float | None
    first_iteration_lowpass_angstrom: float | None

    def __post_init__(self):
        # Python floats, so the solvent-mask radius is the same double arithmetic for every caller.
        object.__setattr__(self, "voxel_size", float(self.voxel_size))
        if self.particle_diameter_angstrom is not None:
            object.__setattr__(self, "particle_diameter_angstrom", float(self.particle_diameter_angstrom))
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1338):

```python
def _reconstruct_k1_maps(
    numerators_by_half,
    denominators_by_half,
    tau_by_half,
    settings: ReconstructionSettings,
    *,
    current_size,
    accumulator_volume_shape,
    retained_first_numerator=None,
) -> list:
    """Reconstruct both K=1 halves while preserving RELION buffer lifetime."""

    if len(tau_by_half) != 2:
        raise ValueError("K=1 reconstruction tau2 requires exactly two halves")
    cs_int = int(current_size) if current_size is not None else None
    reconstructed_means = []
    retained_device_numerator = retained_first_numerator
    for k, (Ft_y_half, Ft_ctf_half, tau_half) in enumerate(
        zip(numerators_by_half, denominators_by_half, tau_by_half)
    ):
        # This RELION build uses double RFLOAT in BackProjector::reconstruct.
        # Keep the stored/controller tau2 state compact, but promote the
        # reconstruction operand so 1 / (padding_factor**3 * tau2) is not
        # rounded in float32 before it enters the Wiener denominator.
        reconstruction_tau = jnp.asarray(tau_half, dtype=jnp.float64)
        reconstructed = _reconstruct_volume_eager(
            Ft_ctf_half,
            Ft_y_half,
            settings.volume_shape,
            settings.padding_factor,
            tau=reconstruction_tau,
            tau2_fudge=settings.tau2_fudge,
            projection_padding_factor=settings.projection_padding_factor,
            minres_map=settings.minres_map,
            current_size=cs_int,
            accumulator_volume_shape=accumulator_volume_shape,
            tau_is_1d=True,
            preserve_output_precision=True,
            relion_filter_scale=float(settings.volume_shape[0] ** 4),
            **(
                {"retained_device_numerator": retained_device_numerator}
                if k == 0 and retained_device_numerator is not None
                else {}
            ),
        ).reshape(-1)
        reconstructed_means.append(
            _finish_host_staged_reconstruction(reconstructed, Ft_ctf_half, Ft_y_half)
        )
        if k == 0 and retained_device_numerator is not None:
            retained_device_numerator = None
            gc.collect()
    return reconstructed_means
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1392):

```python
def _reconstruct_class_maps(
    combined_numerators,
    combined_denominators,
    tau_by_class,
    settings: ReconstructionSettings,
    *,
    n_classes,
    iteration,
    current_size,
    accumulator_volume_shape,
):
    """Reconstruct the shared Class3D stack from combined accumulators."""

    _t_recon = time.time()
    cs_int = int(current_size) if current_size is not None else None
    shared_class_maps = []
    for class_idx in range(n_classes):
        logger.info(
            "Class3D reconstruction start: iter=%d class=%d/%d current_size=%s",
            iteration + 1,
            class_idx + 1,
            n_classes,
            cs_int,
        )
        class_map = _reconstruct_volume_eager(
            combined_denominators[class_idx],
            combined_numerators[class_idx],
            settings.volume_shape,
            settings.padding_factor,
            tau=tau_by_class[class_idx],
            tau2_fudge=settings.tau2_fudge,
            projection_padding_factor=settings.projection_padding_factor,
            minres_map=settings.minres_map,
            current_size=cs_int,
            accumulator_volume_shape=accumulator_volume_shape,
            tau_is_1d=True,
        ).reshape(-1)
        shared_class_maps.append(class_map)
        logger.info(
            "Class3D reconstruction done: iter=%d class=%d/%d elapsed=%.1fs",
            iteration + 1,
            class_idx + 1,
            n_classes,
            time.time() - _t_recon,
        )
    shared_classes = jnp.stack(shared_class_maps, axis=0)
    logger.info(
        "Class3D reconstruction stack complete: iter=%d classes=%d elapsed=%.1fs",
        iteration + 1,
        n_classes,
        time.time() - _t_recon,
    )
    return shared_classes
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1447):

```python
def _postprocess_numbered_maps(
    means,
    settings: ReconstructionSettings,
    *,
    n_classes,
    class_axis,
    iteration,
    current_size,
    relion_firstiter_cc_this_iter,
) -> list:
    """Capture, first-CC filter and solvent-flatten the two solved slots in turn.

    Each slot is one flat K1 half map, or a Class3D stack when ``class_axis``;
    the filter and the flatten then act on each class of the stack. The mask
    radius and construction are common. The flatten implementation is the one
    execution difference: K1 host-stages box-scale results and consumes its
    mask, while a class stack is flattened on the device with one mask.
    """
    for k in range(2):
        # Diagnostic: dump pre-mask Wiener output when env var set.
        _premask_dump = os.environ.get("RELAX_PREMASK_DUMP_DIR")
        if _premask_dump:
            from relax.diagnostics.reconstruction import write_premask_mean

            write_premask_mean(
                means[k], output_dir=_premask_dump, half_index=k, iteration=iteration,
                current_size=current_size, grid_size=settings.grid_size, voxel_size=settings.voxel_size,
                volume_shape=settings.volume_shape, n_classes=n_classes,
            )

        # RELION filters Iref inside maximizationOtherParameters, then calls
        # solventFlatten from the outer iteration loop.  These operations do
        # not commute: masking in real space after the Fourier low-pass adds a
        # small, deterministic high-shell tail.
        if relion_firstiter_cc_this_iter:
            if class_axis:
                means[k] = jnp.stack(
                    [
                        _apply_relion_initial_lowpass_filter(
                            means[k][class_idx],
                            settings.volume_shape,
                            settings.voxel_size,
                            settings.first_iteration_lowpass_angstrom,
                            filter_edgewidth=settings.fmask_edge,
                        )
                        for class_idx in range(n_classes)
                    ],
                    axis=0,
                )
            else:
                means[k] = _apply_relion_initial_lowpass_filter(
                    means[k],
                    settings.volume_shape,
                    settings.voxel_size,
                    settings.first_iteration_lowpass_angstrom,
                    filter_edgewidth=settings.fmask_edge,
                )
        if (
            settings.particle_diameter_angstrom is not None
            and settings.particle_diameter_angstrom > 0
        ):
            flatten_radius = settings.particle_diameter_angstrom / (2.0 * settings.voxel_size)
            solvent_mask = _make_relion_solvent_mask(
                settings.volume_shape,
                radius=flatten_radius,
                radius_p=flatten_radius + settings.width_mask_edge,
                offset=jnp.zeros(3),
                dtype=(means[k][0] if class_axis else means[k]).real.dtype,
            )
            if class_axis:
                flattened_classes = []
                for class_idx in range(n_classes):
                    vol_real = fourier_transform_utils.get_idft3(
                        means[k][class_idx].reshape(settings.volume_shape)
                    )
                    flattened_classes.append(
                        fourier_transform_utils.get_dft3(vol_real * solvent_mask).reshape(-1),
                    )
                means[k] = jnp.stack(flattened_classes, axis=0)
            else:
                means[k] = _apply_relion_solvent_flatten_k1(
                    means[k], solvent_mask, settings.volume_shape, half_index=k,
                )
                if _large_relion_solvent_mask_uses_compiled_builder(settings.volume_shape):
                    solvent_mask = None
    if relion_firstiter_cc_this_iter and settings.first_iteration_lowpass_angstrom is not None:
        logger.info(
            "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
            float(settings.first_iteration_lowpass_angstrom),
        )
    return means
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1540):

```python
def reconstruct_numbered_k1_halfmaps(
    numerators_by_half,
    denominators_by_half,
    tau_by_half,
    settings: ReconstructionSettings,
    *,
    iteration,
    current_size,
    accumulator_volume_shape,
    relion_firstiter_cc_this_iter,
    retained_first_numerator=None,
) -> list:
    """Solve two independent numbered K1 maps, then postprocess each half.

    Numerators, denominators and priors are ordered half pairs; each prior is
    a shell curve. The private solve frame owns promoted priors, host
    completion and the retained half-0 numerator boundary. Both solves finish
    before premask capture, initial filtering and solvent flattening. Return
    ready maps for installation.
    """
    means = _reconstruct_k1_maps(
        numerators_by_half, denominators_by_half, tau_by_half, settings,
        current_size=current_size, accumulator_volume_shape=accumulator_volume_shape,
        retained_first_numerator=retained_first_numerator,
    )
    return _postprocess_numbered_maps(
        means, settings, n_classes=1, class_axis=False, iteration=iteration, current_size=current_size,
        relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
    )
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1571):

```python
def reconstruct_numbered_class_maps(
    combined_numerators,
    combined_denominators,
    tau_by_class,
    settings: ReconstructionSettings,
    *,
    n_classes,
    iteration,
    current_size,
    accumulator_volume_shape,
    relion_firstiter_cc_this_iter,
) -> list:
    """Solve one numbered Class3D reference stack from combined partitions.

    Accumulators and priors have a leading class axis; each prior is a shell
    curve. All class solves finish before premask capture, initial filtering
    and solvent flattening.
    Return a two-entry list of particle-execution slots, not scientific
    halves. Each slot is captured, filtered and flattened in turn; the entries
    alias the shared stack when neither filtering nor flattening applies.
    """
    shared_classes = _reconstruct_class_maps(
        combined_numerators, combined_denominators, tau_by_class, settings,
        n_classes=n_classes, iteration=iteration, current_size=current_size,
        accumulator_volume_shape=accumulator_volume_shape,
    )
    means = [shared_classes, shared_classes]
    del shared_classes
    return _postprocess_numbered_maps(
        means, settings, n_classes=n_classes, class_axis=True, iteration=iteration, current_size=current_size,
        relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
    )
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1606):

```python
class K1ReportingPrior:
    """Tapered K1 shared/half volumes and their per-half reporting details."""

    variance: object
    variance_per_half: list
    details_per_half: list[dict]
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1614):

```python
def taper_first_cc_k1_prior(
    variance_per_half,
    details_per_half,
    settings: ReconstructionSettings,
    *,
    pixel_size_angstrom,
    scoring_dtype,
) -> K1ReportingPrior:
    """Taper K1 reporting priors after the untapered regularized reconstruction.

    Preserve incoming list/dict identity and the half -> prior/SSNR update
    order. The expanded taper and radial grid are temporary implementation
    arrays; model installation and shared/per-half policy stay with the caller.
    """
    tau2_taper = _firstiter_cc_ini_high_tau2_taper(
        len(details_per_half[0]["prior_shells"]),
        settings.grid_size,
        pixel_size_angstrom,
        settings.first_iteration_lowpass_angstrom,
        filter_edgewidth=settings.fmask_edge,
    )
    radial_shells = np.asarray(
        fourier_transform_utils.get_grid_of_radial_distances(
            settings.volume_shape,
            scaled=False,
            frequency_shift=0,
        ),
        dtype=np.int32,
    ).reshape(-1)
    radial_shells = np.minimum(radial_shells, len(tau2_taper) - 1)
    tau2_taper_volume = jnp.asarray(
        tau2_taper[radial_shells],
        dtype=scoring_dtype,
    )
    for half_idx in range(2):
        variance_per_half[half_idx] = (
            variance_per_half[half_idx] * tau2_taper_volume
        )
        for field in ("prior_shells", "ssnr_shells"):
            field_values = details_per_half[half_idx][field]
            details_per_half[half_idx][field] = field_values * jnp.asarray(
                tau2_taper,
                dtype=field_values.dtype,
            )
    variance = 0.5 * (
        variance_per_half[0] + variance_per_half[1]
    )
    return K1ReportingPrior(variance, variance_per_half, details_per_half)
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1665):

```python
class ClassReportingPrior:
    """Tapered Class3D shell stack and its aggregate prior/SSNR details."""

    shells: object
    details: dict
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1672):

```python
def taper_first_cc_class_prior(
    shells,
    details,
    settings: ReconstructionSettings,
    *,
    pixel_size_angstrom,
) -> ClassReportingPrior:
    """Taper class reporting shells after the caller publishes the tapered curve.

    The controller first tapers/publishes data-vs-prior for scheduling. This
    operation then adapts the shell stack and aggregate detail arrays in their
    existing order, preserving the detail mapping's identity.
    """
    def taper_shells(values):
        return _firstiter_cc_ini_high_tapered(
            values,
            settings.grid_size,
            pixel_size_angstrom,
            settings.first_iteration_lowpass_angstrom,
            filter_edgewidth=settings.fmask_edge,
        )

    shells = jnp.asarray(taper_shells(np.asarray(shells)))
    for field in ("prior_shells", "ssnr_shells"):
        details[field] = taper_shells(np.asarray(details[field]))
    return ClassReportingPrior(shells, details)
```

### Actual half-source selection, preparation, dataset subset and admission

[relax/refinement/full_refinement.py](../../relax/refinement/full_refinement.py) (line 689):

```python
    if args.relion_half_sets is not None:
        # Use RELION's half-set split from rlnRandomSubset
        logger.info("Loading RELION half-set assignments from %s", args.relion_half_sets)
        halfset_source = input_particle_table.read_relion_halfset_source(
            args.relion_half_sets,
            n_classes=args.n_classes,
            log=logger,
        )
        relion_particles = halfset_source.tables["particles"]
        relion_optics_image_sizes = halfset_source.optics_image_sizes
        relion_optics_pixel_sizes = halfset_source.optics_pixel_sizes
        use_fresh_auto_refine_order = _use_fresh_auto_refine_particle_order(
            args,
            frozen_boundary,
        )
        halfset_inputs = particle_loading.prepare_relion_halfset_inputs(
            our_particles,
            relion_particles,
            source_path=args.relion_half_sets,
            random_seed=args.seed if use_fresh_auto_refine_order else None,
            prepare_noise_order=(
                args.n_classes == 1
                and (relion_startup_noise_needed or use_relion_live_initial_noise)
            ),
            tomographic=tomo_run,
            image_grid_size=None if tomo_run else ds.grid_size,
            log=logger,
        )
        particle_layout = halfset_inputs.layout
        relion_fresh_initial_noise_source_rows = halfset_inputs.noise_source_rows
        relion_fresh_initial_noise_optics_group_ids = halfset_inputs.noise_optics_group_ids
        expected_accuracy_half1_ctf_params = halfset_inputs.accuracy_ctf_params
        del halfset_inputs
        if use_fresh_auto_refine_order:
            logger.info(
                "Applied RELION fresh paired AutoRefine particle order (mt19937) with effective seed %d; "
                "BPref will preserve this physical order",
                int(args.seed) + 1,
            )
    elif args.n_classes == 1:
        raise SystemExit(
            "K=1 auto-refine uses RELION's half sets: a fresh start rebuilds them from the "
            "input STAR (--relion-half-sets-from-input, the default); a RELION-seeded, "
            "replayed or frozen start needs --relion_half_sets"
        )
    else:
        particle_layout = input_particle_table.prepare_class3d_particle_layout(
            our_particles,
            n_particles=n_images,
            random_seed=args.seed,
            init_relion_iteration=args.init_relion_iteration,
        )
    if args.n_classes > 1 and relion_startup_noise_needed:
        (
            relion_fresh_initial_noise_source_rows,
            relion_fresh_initial_noise_optics_group_ids,
        ) = startup_noise.class3d_noise_order(our_particles)
        if not isinstance(our_star, dict) or "optics" not in our_star:
            raise SystemExit("Class3D RELION start-up noise needs an optics table in the particle STAR")
        class3d_noise_optics_pixel_sizes = np.asarray(
            our_star["optics"]["rlnImagePixelSize"],
            dtype=np.float64,
        )

    local_stop_requested = (
        bool(args.stop_after_local_search_profile)
        or bool(args.stop_after_local_search)
        or bool(args.stop_after_local_search_score_only)
    )
    if args.diagnostic_single_half:
        if not local_stop_requested:
            raise SystemExit(
                "--diagnostic_single_half is only valid with --stop_after_local_search, "
                "--stop_after_local_search_profile, or --stop_after_local_search_score_only"
            )
        if args.n_classes != 1:
            raise SystemExit("--diagnostic_single_half is K=1-only")
        logger.warning(
            "Diagnostic single-half local-search probe: running half 1 only (%d images); "
            "half 2 is empty. Do not use this for map/FSC quality.",
            int(particle_layout.half1_rows.size),
        )
        particle_layout = particle_layout._replace(half2_rows=np.empty(0, dtype=np.int64))

    resume_snapshot = None
    if args.continue_optimiser_star is not None:
        if shape_class_rows is not None:
            raise SystemExit("--continue does not support particle STARs with several image shapes yet")
        resume_snapshot = read_run_files(
            args.continue_optimiser_star,
            image_names=[str(name) for name in our_names],
            half_rows=[particle_layout.half1_rows, particle_layout.half2_rows],
        )
        logger.info(
            "Continuing after numbered iteration %d from %s",
            resume_snapshot.relion_iteration,
            args.continue_optimiser_star,
        )
        if int(args.max_iter) < int(resume_snapshot.relion_iteration):
            raise SystemExit(
                f"--max_iter {args.max_iter} is the last numbered iteration of the whole run "
                f"(RELION's --iter); the run files are already at iteration {resume_snapshot.relion_iteration}"
            )

    ds_half1 = ds.subset(particle_layout.half1_rows)
    ds_half2 = ds.subset(particle_layout.half2_rows)
    logger.info("Half-sets: %d + %d images", ds_half1.n_units, ds_half2.n_units)
    if frozen_boundary is not None:
        frozen_boundary_cli.validate_particle_half_inputs(
            frozen_boundary,
            image_names=our_names,
            half_rows=(particle_layout.half1_rows, particle_layout.half2_rows),
            image_shape=ds.image_shape,
            log=logger,
        )
```

### Complete half-row/noise/accuracy operation and result

[relax/refinement/particle_loading.py](../../relax/refinement/particle_loading.py) (line 46):

```python
class HalfsetParticleInputs(NamedTuple):
    """Selected half rows paired with their startup-noise and accuracy frames."""

    layout: "ParticleLayout"
    noise_source_rows: np.ndarray | None
    noise_optics_group_ids: np.ndarray | None
    accuracy_ctf_params: np.ndarray | None
```

[relax/refinement/particle_loading.py](../../relax/refinement/particle_loading.py) (line 55):

```python
def prepare_relion_halfset_inputs(
    our_particles,
    relion_particles,
    *,
    source_path,
    random_seed: int | None,
    prepare_noise_order: bool,
    tomographic: bool,
    image_grid_size: int | None,
    log,
) -> HalfsetParticleInputs:
    """Adapt a selected half-set source into scoring, noise and accuracy rows.

    Noise uses its pre-shuffle frame. Accuracy CTF rows index the RELION source
    table; tomography instead reads tilt-image CTFs from its dataset half.
    The unused full CTF source table expires after the selected copy is built.
    """
    from relax.relion import input_particle_table

    relion_fresh_initial_noise_source_rows = None
    relion_fresh_initial_noise_optics_group_ids = None
    expected_accuracy_half1_ctf_params = None
    particle_layout = input_particle_table.prepare_relion_halfset_layout(
        our_particles,
        relion_particles,
        random_seed=random_seed,
    )
    if prepare_noise_order:
        from relax.refinement import startup_noise

        (
            relion_fresh_initial_noise_source_rows,
            relion_fresh_initial_noise_optics_group_ids,
        ) = startup_noise.auto_refine_noise_order(our_particles, relion_particles)
    if not tomographic:
        # Subtomogram particles have one CTF per tilt image; their expected accuracy reads
        # those from the tomo half (relax.refinement.tomo_half).
        from recovar.data_io import metadata_readers

        relion_ctf_with_apix = metadata_readers.parse_ctf_from_star(
            source_path,
            image_grid_size,
        )
        expected_accuracy_half1_ctf_params = np.asarray(
            relion_ctf_with_apix[particle_layout.accuracy_particle_ids, 1:],
            dtype=np.float64,
        )
    log.info(
        "Using RELION half-set split: %d (subset=1) + %d (subset=2)",
        len(particle_layout.half1_rows),
        len(particle_layout.half2_rows),
    )
    return HalfsetParticleInputs(
        layout=particle_layout,
        noise_source_rows=relion_fresh_initial_noise_source_rows,
        noise_optics_group_ids=relion_fresh_initial_noise_optics_group_ids,
        accuracy_ctf_params=expected_accuracy_half1_ctf_params,
    )
```

### Complete frozen half-identity/shell admission

[relax/diagnostics/frozen_boundary_cli.py](../../relax/diagnostics/frozen_boundary_cli.py) (line 294):

```python
def validate_particle_half_inputs(
    frozen_boundary,
    *,
    image_names,
    half_rows,
    image_shape,
    log,
) -> None:
    """Admit active half identities and shell geometry against a sealed input.

    Comparisons preserve image/source/subset/half/local index relationships.
    Validation-only names and integer vectors expire at this boundary.
    """
    import numpy as np

    live_names_per_half = (
        np.asarray(image_names[half_rows[0]], dtype=str),
        np.asarray(image_names[half_rows[1]], dtype=str),
    )
    for half, (live_names, frozen_names) in enumerate(
        zip(live_names_per_half, frozen_boundary.image_names_per_half, strict=True),
        start=1,
    ):
        if not np.array_equal(live_names, frozen_names):
            raise SystemExit(
                "Frozen-boundary particle identity/order mismatch for "
                f"half {half}: live_rows={live_names.size}, "
                f"frozen_rows={frozen_names.size}"
            )
        frozen_half = half - 1
        expected_source_rows = np.asarray(
            half_rows[0] if frozen_half == 0 else half_rows[1],
            dtype=np.int64,
        )
        five_field_checks = {
            "source_row": (
                expected_source_rows,
                frozen_boundary.source_rows_per_half[frozen_half],
            ),
            "random_subset": (
                np.full(live_names.shape, half, dtype=np.int8),
                frozen_boundary.random_subsets_per_half[frozen_half],
            ),
            "half_index": (
                np.full(live_names.shape, frozen_half, dtype=np.int8),
                frozen_boundary.half_indices_per_half[frozen_half],
            ),
            "half_local_index": (
                np.arange(live_names.size, dtype=np.int64),
                frozen_boundary.half_local_indices_per_half[frozen_half],
            ),
        }
        for field_name, (expected, frozen) in five_field_checks.items():
            if not np.array_equal(expected, frozen):
                raise SystemExit(
                    "Frozen-boundary five-field identity mismatch for "
                    f"half {half} field {field_name}"
                )
    log.info(
        "Frozen-boundary five-field particle identities match the active half layout "
        "exactly: %d + %d",
        live_names_per_half[0].size,
        live_names_per_half[1].size,
    )
    expected_shells = int(image_shape[0] // 2 + 1)
    shell_shapes = {
        "fsc": frozen_boundary.fsc.shape,
        "half1_noise_radial": frozen_boundary.noise_radial_per_half[0].shape,
        "half2_noise_radial": frozen_boundary.noise_radial_per_half[1].shape,
    }
    unexpected_shell_shapes = {
        name: shape
        for name, shape in shell_shapes.items()
        if shape != (expected_shells,)
    }
    if unexpected_shell_shapes:
        raise SystemExit(
            "Frozen-boundary shell arrays do not match the active dataset: "
            f"expected={(expected_shells,)}, got={unexpected_shell_shapes}"
        )
```

### Final orchestration and retired interfaces

Final calls remain in [finalization.py](../../relax/refinement/finalization.py),
with numerical operations in [final_reconstruction.py](../../relax/refinement/final_reconstruction.py).
The [complete policy/lifetime and consumer audit](/scratch/gpfs/GILLES/mg6942/tmp/relax_finish_milestone_20261002/finalization_review/REVIEW.md)
explains the distinct numbered/final/VDAM semantics and shared primitives.
`reconstruct_k1_means`, `reconstruct_class_means` and
`postprocess_reconstructed_means` are retired, as is the single
`reconstruct_regularized_means` that replaced them. The controller calls
`reconstruct_numbered_k1_halfmaps` or `reconstruct_numbered_class_maps`, and each
test calls the operation it exercises with that operation's operands; the
`run_mean_reconstruction` test adapter is removed. Source/lifetime guards and
first-CC injection follow the actual owner. No baseline, scientific tolerance or independent reference changes.

<a id="shared-candidate-chunk-layout"></a>

## Earlier shared candidate chunk layout

The existing candidate-table owner now computes cell offsets for global scoring,
the local parent probe and local fine scoring. The two former production bodies
were identical. A maintainer changing padded-image layout can read the table and
chunk definitions plus this operation; neither scoring engine needs to be opened
to discover a second copy of the same layout rule. Engine-specific placement,
posterior kernels and accumulation remain with their respective drivers.

The moved body keeps int64 host arithmetic, capacity refusal and the final int32
layout. Padded image slots repeat the valid end offset. Callers retain their
existing placement, kernel sequence and buffer lifetimes. No new module, type,
wrapper, flag or JIT boundary is introduced. The unrelated unused private
`compact_candidates._quantize_up` is also removed; active capacity policy remains.

### Complete shared implementation

[relax/sparse_pass2/resident_candidates.py](../../relax/sparse_pass2/resident_candidates.py) (line 225):

```python
def chunk_segment_offsets(tables, chunk, *, n_fine_trans: int) -> np.ndarray:
    """Cell offsets by image slot; padded slots repeat the valid end offset."""
    image_capacity = int(chunk.image_capacity)
    n_valid_images = int(chunk.n_valid_images)
    offsets = np.full(image_capacity + 1, chunk.n_valid_rows * int(n_fine_trans), dtype=np.int64)
    starts = (
        np.asarray(
            tables.row_offsets[chunk.image_start : chunk.image_start + n_valid_images + 1],
            dtype=np.int64,
        )
        - int(chunk.row_start)
    ) * int(n_fine_trans)
    offsets[: n_valid_images + 1] = starts
    if int(offsets[-1]) > int(chunk.row_capacity) * int(n_fine_trans):
        raise ValueError("chunk segment offsets exceed the chunk's cell capacity")
    return offsets.astype(np.int32)
```

### Actual global chunk preparation

[relax/sparse_pass2/resident_pass2.py](../../relax/sparse_pass2/resident_pass2.py) (line 5239):

```python
    image_capacity = int(chunk.image_capacity)
    tables, host_chunk, local_chunk = _chunk_host_rows(tables, chunk)
    segment_offsets_np = chunk_segment_offsets(tables, local_chunk, n_fine_trans=n_fine_trans)
    image_row_start_np = segment_offsets_np.astype(np.int64)[:image_capacity] // int(n_fine_trans)
    image_row_count_np = (
        segment_offsets_np.astype(np.int64)[1:] - segment_offsets_np.astype(np.int64)[:-1]
    ) // int(n_fine_trans)
```

### Actual local parent and fine callers

[relax/sparse_pass2/resident_local_pass2.py](../../relax/sparse_pass2/resident_local_pass2.py) (line 1529):

```python
        del score_proj, ops
        scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
        segment_offsets_np = chunk_segment_offsets(tables, chunk, n_fine_trans=t)
        segment_offsets = jnp.asarray(segment_offsets_np, dtype=jnp.int32)
        log_z = em_cuda_kernels.sparse_pass2_segmented_log_z_f64(
            scores_flat, segment_offsets, n_valid_images_device
        )
```

[relax/sparse_pass2/resident_local_pass2.py](../../relax/sparse_pass2/resident_local_pass2.py) (line 1965):

```python
    # --- stage 4: segmented RELION float32 fine posterior -------------------
    segment_offsets_np = chunk_segment_offsets(tables, chunk, n_fine_trans=n_fine_trans)
    segment_offsets = jnp.asarray(segment_offsets_np, dtype=jnp.int32)
    log_z = cuda_backproject.sparse_pass2_segmented_log_z_f64(
        scores_flat, segment_offsets, n_valid_images_device
    )
```

### Checks and remaining work

All 198 selected CPU cases pass, zero skips. The source comparison proves that
both old bodies equal the shared body and that surviving engine operations differ
only at the three helper calls. The existing direct layout test now imports its
actual owner; numerical assertions, tolerances and baselines are unchanged.
GPU-only cases were excluded explicitly and are listed in the test inventory.
The [handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_shared_chunk_layout_20261002T214235Z/HANDOFF.json)
links exact source, commands, inventory and structural evidence.

The private-callgraph screen found no additional candidate in refinement, helpers
or diagnostics; the broader screen identified only the retired quantizer. A
substantial exact-body duplicate screen identified only these two layout bodies.
These are conservative static screens, not proof of complete dynamic reachability
or mathematical uniqueness. Independent numerical references remain independent.

The numerical controller remains 2,883 lines and command 1,860. Substantive
reconstruction extraction still needs the recorded scratch-lifetime decision.
Complete-flow design review precedes a new freeze and final production float32
K1/exactly-K4, real-data quality, peak memory and matched-GPU speed qualification.
Original work, frozen/control candidates and existing jobs remain preserved.

<a id="retired-private-capacity-planning-family"></a>

## Earlier retired private capacity-planning family

The batch planner contained a disconnected family describing a future whole-local
executor: two call/order descriptors, a generation token, a plan, its fingerprint
and a packing validator. Some members referenced each other, but no maintained
operation constructed or consumed the family. All six declarations and their
three unused import bindings are removed: 223 lines of protocol that a maintainer
no longer needs to distinguish from the live memory planner.

The audit screened 1,278 maintained Python, notebook, shell, configuration, JSON
and Markdown files, then checked registration and serialization in the owner.
It found no outside reference. The surviving live planner's AST is identical
after deleting only those declarations/imports. Existing dense, local, compact
and adaptive batch decisions, GPU occupancy calculations and shape budgets are
unchanged; independent numerical references and supported formats are preserved.

All 43 focused batch, expectation-planning and import cases pass, zero skips.
No tests, tolerances or baselines changed. The first lint run found the remaining
unused `numbers.Integral` import and stopped before tests; removing that binding
produced the passing receipt. Exact source, audit inventory and commands are in
the [handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_retired_capacity_planning_20261002T213032Z/HANDOFF.json).

This is a scoped retirement proof, not a whole-repository dead-code certificate.
The numerical controller remains 2,883 lines and command 1,860. Reconstruction
aggregation and post-reconstruction taper still need the pending scratch-lifetime
decision: extraction would expire currently retained arrays earlier. Full-flow
review and final production float32 K1/exactly-K4, real-data quality, peak memory
and matched-GPU performance remain open. No new GPU job, freeze or publication;
original work, frozen/control sources and existing jobs remain untouched.

The current image-preprocessing caller and implementation follow below.

<a id="half-image-preprocessing-ownership"></a>

## Earlier half image preprocessing ownership

The existing particle-input owner now configures half image backends and
scoring masks as one substantial operation. A maintainer changing backend
admission or mask units can read this operation and the backend API, leaving
sampling, expectation and reconstruction unopened. The controller exposes its
position before state initialization and keeps the same setup timing marker.
No context object, new module, result type or forwarding chain is introduced.

The operation walks SPA datasets, optics shape-class datasets or tomography
tilt images in the same order. Shape classes use their own pixel size; SPA and
tilt images retain the original reference pixel scalar. Backend registration,
selection, source-faithful refusal and mask installation run in the same order.
The CLI's separate mask-source admission remains with its existing private
helper; its precedence and fallback behavior are unchanged.

### Actual calling flow

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 531):

```python
configure_half_image_preprocessing(
    experiment_datasets,
    pixel_size_angstrom=source_pixel_size_angstrom,
    particle_diameter_angstrom=particle_diameter_ang,
    fourier_backend=parity.image_fourier_backend,
    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    log=logger,
)

_mark_setup_phase("mask_and_image_cache")

state = initialize_refinement_state(
    options,
    image_geometry,
    subtomogram=tomo_halves,
    dtype=scoring_dtype,
)
```

### Complete implementation

[relax/refinement/particle_loading.py](../../relax/refinement/particle_loading.py) (line 123):

```python
def configure_half_image_preprocessing(
    experiment_datasets,
    *,
    pixel_size_angstrom,
    particle_diameter_angstrom: float | None,
    fourier_backend: str,
    source_faithful_spectrum_norm: bool,
    log,
) -> None:
    """Configure half image backends and masks before refinement state is built.

    Shape classes use their own pixels; SPA and tilt-image masks keep the
    reference pixel scalar used by the existing refinement path.
    """
    multi_shape_halves = isinstance(experiment_datasets[0], MultiShapeHalf)
    # A half of several image shapes sets up each shape class's images, masked with
    # the class's own pixel size as RELION does.
    image_datasets = [
        dataset
        for half in experiment_datasets
        for dataset in (
            [c.dataset for c in half.classes]
            if isinstance(half, MultiShapeHalf)
            else [half.images] if isinstance(half, TomoHalf) else [half]
        )
    ]
    for ds in image_datasets:
        backend = _image_backend(ds)
        if backend is None:
            continue
        mask_pixel_size = ds.voxel_size if multi_shape_halves else pixel_size_angstrom
        if hasattr(backend, "set_relion_fourier_backend"):
            from relax.cuda import (
                kernels as _em_cuda_kernels,  # noqa: F401  (registers the relion_cuda preprocessor, relax split seam S2)
            )

            backend.set_relion_fourier_backend(fourier_backend)
        if source_faithful_spectrum_norm and getattr(backend, "relion_fourier_backend", None) not in (None, "relion_cuda"):
            # The fresh K=1 defaults score from RELION's CUDA image preprocessing;
            # fail here instead of inside the first sparse pass 2.
            raise ValueError(
                "fresh K=1 refinement defaults (source-faithful powerClass normalization and "
                "exact RELION BPref operands) require RELION CUDA image preprocessing; pass "
                "--image-fourier-backend relion_cuda or disable the fresh particle order"
            )
        if particle_diameter_angstrom is not None and particle_diameter_angstrom > 0:
            backend.set_relion_image_mask(
                pixel_size=mask_pixel_size,
                particle_diameter_ang=particle_diameter_angstrom,
                width_mask_edge_px=IMAGE_MASK_EDGE_PIXELS,
            )
            log.info(
                "RELION mode: image mask radius=%.1f px (particle_diameter=%.1f A, edge=%d px)",
                particle_diameter_angstrom / (2.0 * mask_pixel_size),
                particle_diameter_angstrom,
                IMAGE_MASK_EDGE_PIXELS,
            )
```

### Ownership, verification and limits

The removed controller aliases held only borrowed datasets, backends and scalar
metadata. SPA halves, `MultiShapeHalf.classes` and `TomoHalf.images` still own
every image dataset; each dataset's image source still owns its backend and
mask. No numerical scratch array is created by this operation and no array
root, donation, transfer or release boundary changes. The native registration
module remains rooted by Python's import system at the same runtime call site.

All 80 selected CPU cases pass with zero skips: 64
particle/input/optics/tomography/import cases and 16
affected controller cases. Existing tests and numerical assertions are unchanged;
no mirror test, scientific tolerance or baseline was added or changed. Structural
comparison inlines the actual implementation and verifies the original complete
controller operations, resolving only its six proven operands. Every existing
particle-input function/type remains unchanged. The extra host import dependency
and repeated fixed mode classification are recorded explicitly. Source lint and
diff checks pass. Exact commands, source and limits are in the
[handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_image_preprocessing_20261002T212207Z/HANDOFF.json).

Controller spans 2,883 lines and command 1,860. Phase ownership and complete-flow
review remain unfinished; line counts do not establish a finished design.
The earlier Class3D scratch-lifetime choice remains pending. Production float32
K1/exactly-K4, real-data quality, peak memory and matched-GPU speed await design
review and a new frozen source. No new GPU job, freeze, merge or publication;
existing work, sources, controls and jobs are preserved.

<a id="input-geometry-and-replay-boundary"></a>

## Earlier input geometry and replay boundary

Replay consumes the existing `ImageGeometry` rather than a generic dataset.
The same validated object now supplies fixed image shape and physical pixel
size. Its positive, finite pixel contract replaces two silent substitutions of
`1.0`, following the user's accepted invalid-metadata decision. Valid-input
translation arithmetic retains the old Python float value. Changing replay's
units no longer requires understanding the dataset's unrelated responsibilities.

The controller borrows `source_pixel_size_angstrom` directly from the input
dataset for existing host calculations whose scalar dtype affects NumPy
promotion. Normalized geometry and the original scalar are deliberately distinct.
The dataset remains rooted by the same experiment/half inputs. No new type,
function, module, copy, transfer, kernel or release boundary is introduced.
Fixed input geometry does not own changing Fourier windows or model support.

### Actual producer and boundary validation

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 443):

```python
volume_shape = experiment_datasets[0].volume_shape
# Keep the input scalar type for host arithmetic; geometry validates its value.
source_pixel_size_angstrom = experiment_datasets[0].voxel_size
image_geometry = ImageGeometry(
    image_shape=experiment_datasets[0].image_shape,
    pixel_size_angstrom=source_pixel_size_angstrom,
)
grid_size = image_geometry.box_size
```

[relax/helpers/resolution.py](../../relax/helpers/resolution.py) (line 29):

```python
class ImageGeometry:
    """Physical input-image grid; independent of Fourier windows and model support."""

    image_shape: tuple[int, int]
    pixel_size_angstrom: float

    def __post_init__(self):
        pixel_size = float(self.pixel_size_angstrom)
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(f"Particle pixel size must be finite and positive, got {pixel_size}")
        object.__setattr__(self, "pixel_size_angstrom", pixel_size)

    @property
    def box_size(self) -> int:
        return self.image_shape[0]
```

### Actual replay call and visible state publication

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 1147):

```python
replay_result = apply_iter_replay_overrides(
    iter_replay_override=iter_replay_override,
    perturb_replay_relion_dir=perturb_replay_relion_dir,
    perturb_replay_relion_prefix=perturb_replay_relion_prefix,
    init_relion_iteration=init_relion_iteration,
    iteration=iteration,
    state=state,
    cs=current_size,
    image_geometry=image_geometry,
    k_class_enabled=k_class_enabled,
    n_classes=n_classes,
    relion_half_inputs=halves,
    previous_best_rotations=previous_best_rotations,
    noise_model=noise_model,
    current_sigma_offset_angstrom=current_sigma_offset_angstrom,
    current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
    direction_priors=direction_priors,
    preserve_existing_direction_prior=replay.preserve_initial_direction_prior,
    sealed_sampling_state=sealed_sampling_state,
    dtype=scoring_dtype,
    symmetry=symmetry,
)
current_size = replay_result.cs
_replay_prior_translations = replay_result.prior_translations
_replay_meta = replay_result.replay_meta
previous_best_rotations = replay_result.previous_best_rotations
noise_model = replay_result.noise_model
replay_mean_variance = (
    None
    if iter_replay_override is None
    else iter_replay_override.get("mean_variance")
)
if replay_mean_variance is not None:
    replay_mean_variance = np.asarray(replay_mean_variance, dtype=np.float64).reshape(-1)
    expected_mean_variance_shape = tuple(reference_model.tau2.shape)
    if replay_mean_variance.shape != expected_mean_variance_shape:
        raise ValueError(
            "K=1 replay mean_variance shape mismatch: "
            f"expected {expected_mean_variance_shape}, "
            f"got {replay_mean_variance.shape}"
        )
    reference_model.tau2 = jnp.asarray(replay_mean_variance)
    logger.info("Replay override: K=1 tau2/mean_variance <- model.star")
current_sigma_offset_angstrom = replay_result.current_sigma_offset_angstrom
current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(
    replay_result.current_sigma_offset_angstrom_per_half
)
if replay_saved_healpix_order is not None:
    replay_saved_healpix_order = int(state.healpix_order)
if k_class_enabled and replay_result.class_weights is not None:
    class_weights = np.asarray(replay_result.class_weights, dtype=np.float64)
    class_log_priors = np.log(class_weights)
    logger.info(
        "Replay override: class priors <- direction-prior row sums (%s)",
        ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_weights)),
    )
```

### Replay consumers

The sealed branch below still installs the same sampling state and translations;
ordinary STAR replay likewise reads `image_geometry.pixel_size_angstrom` before
its existing range/grid calculation. Noise overrides retain their float32/device
adaptation and use `image_geometry.image_shape`. Conditions, mutation order,
exception scope and the rest of the replay implementation are unchanged.

[relax/diagnostics/relion_replay.py](../../relax/diagnostics/relion_replay.py) (line 1085):

```python
if sealed_sampling_state is not None:
    if int(iteration) != 0:
        raise ValueError("sealed frozen-boundary sampling currently owns exactly one iteration")
    _px = image_geometry.pixel_size_angstrom
    _relion_hp = int(sealed_sampling_state["healpix_order_original"])
    if state.max_healpix_order is not None and _relion_hp > int(state.max_healpix_order):
        raise ValueError(
            "sealed sampling HEALPix order exceeds runtime maximum: "
            f"sealed={_relion_hp} max={state.max_healpix_order}"
        )
    state.healpix_order = _relion_hp
    state.do_local_search = bool(state.auto_sampling and state.healpix_order >= state.auto_local_healpix_order)
    state.sigma_rot = np.deg2rad(float(sealed_sampling_state["sigma_rot_deg"]))
    state.sigma_psi = np.deg2rad(float(sealed_sampling_state["sigma_psi_deg"]))
    state.translation_range = float(sealed_sampling_state["offset_range_angstrom"]) / _px
    state.translation_step = float(sealed_sampling_state["offset_step_angstrom"]) / _px
    sealed_x = np.asarray(sealed_sampling_state["translations_x_angstrom"], dtype=np.float64)
    sealed_y = np.asarray(sealed_sampling_state["translations_y_angstrom"], dtype=np.float64)
    _replay_prior_translations = jnp.asarray(
        np.stack([sealed_x / _px, sealed_y / _px], axis=1),
        dtype=runtime_dtype,
    )
    cs = int(sealed_sampling_state["current_size"])
    _replay_meta = {
        "healpix_order": _relion_hp,
        "psi_step": float(sealed_sampling_state["psi_step_deg"]),
        "offset_range": float(sealed_sampling_state["offset_range_angstrom"]),
        "offset_step": float(sealed_sampling_state["offset_step_angstrom"]),
        "perturbation_factor": float(sealed_sampling_state["perturbation_factor"]),
        "random_perturbation": float(sealed_sampling_state["random_perturbation"]),
        "sealed_v3": True,
    }
    logger.info(
        "Frozen-boundary v3 owns sampling: consumer_iter=%d hp=%d current/coarse=%d/%d "
        "translations=%d rp=%+.12g",
        int(sealed_sampling_state["consumer_relion_iteration"]),
        _relion_hp,
        cs,
        int(sealed_sampling_state["coarse_size"]),
        int(_replay_prior_translations.shape[0]),
        float(sealed_sampling_state["random_perturbation"]),
    )
```

[relax/diagnostics/relion_replay.py](../../relax/diagnostics/relion_replay.py) (line 1442):

```python
_replay_noise = iter_replay_override.get("noise_variance")
if _replay_noise is not None:
    noise_model = noise_model_from_pixels(
        _replay_noise, image_geometry.image_shape, dtype=runtime_dtype,
    )
    logger.info("Replay override: sigma2_noise <- per-half model.star arrays")
```

### Verification and remaining work

All 185 selected CPU cases pass: 130 replay/resolution/startup/promotion cases,
one Class3D replay case and 54 affected controller cases, with zero skips.
Source/test import lint and diff checks pass. Structural comparison resolves
only the explicit metadata bindings and verifies unchanged scientific operations.
It records the deliberate invalid-input rejection and the setup read-order change:
raw pixel metadata is read before image shape, both fixed supported metadata.
No numerical assertion, tolerance or baseline changed. The first structural
checker attempt failed because its normalization missed older geometry-shape
reads; fixing that proof script alone produced the passing receipt.

Exact source, test inventory and limits are in the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_input_geometry_20261002T210920Z/HANDOFF.json).
The controller spans 2,917 lines and command 1,860; these are review signals,
not a completion claim. Representative phase ownership, the pending Class3D
scratch-lifetime choice and whole dead/shared-code audit remain open. Final
production float32 K1/exactly-K4, real-data quality, peak memory and matched-GPU
speed await complete-flow review and a new frozen source. Existing sources/jobs
are preserved; this package launches no GPU job or publication.

<a id="sealed-restart-runtime-adaptation"></a>

## Earlier sealed restart runtime adaptation

The existing diagnostic CLI owner now holds projector-only replay slots,
capture attachment and scoring-noise expansion alongside its admission and
source-binding operations. These four functions already performed substantial
adaptation or validation. They moved together; no new function, type, module,
forwarding layer, flag or numerical operation was introduced.

A change to the sealed restart's slot rules, captured-projector admission or
noise dtype now needs the diagnostic CLI/schema/capture owners. Normal command
admission, native sampling and refinement execution can remain unopened.
The controller still selects frozen versus trajectory replay, installs noise,
chooses capture paths, attaches the result at the same phase and validates the
completed projector-only slots. Its exception boundary, metadata publication,
initialization order and numerical buffer roots are unchanged.

### Producers and consumers

| Operand | Ownership and invariant |
| --- | --- |
| Sealed radial noise | Bundle loader supplies per-half host profiles; diagnostic adaptation expands them through the existing RECOVAR formula with float32 scoring input. The controller retains noise arrays and computes its existing reporting mean. |
| Replay slots | Diagnostic restart creates numbered/final slots with no fresh-process noise broadcast. Ordinary trajectory replay continues through its existing independent branch. |
| Captured projector | Diagnostic capture loader verifies the exact adjacent iteration, model support, manifest and slot before attaching it. Existing captured arrays and provenance return to the same controller locals. |
| Completed slot set | Diagnostic validation rejects missing slots, extra scientific overrides and misplaced/duplicate projectors before execution. No failure becomes a fallback. |

### Actual frozen-noise branch

The remaining NPZ/live/model noise branches continue unchanged in their existing
startup owner. This is the complete frozen branch at its original phase:

[relax/refinement/full_refinement.py](../../relax/refinement/full_refinement.py) (line 1235):

```python
if frozen_boundary is not None:
    noise_variance = frozen_boundary_cli.expand_boundary_noise(
        frozen_boundary.noise_radial_per_half,
        ds.image_shape,
    )
    initial_noise_radial = np.mean(
        np.stack(frozen_boundary.noise_radial_per_half, axis=0),
        axis=0,
    )
    logger.info(
        "Initial noise/tau2 state is owned by frozen boundary %s",
        frozen_boundary.source_dir,
    )
```

### Actual frozen replay branch

[relax/refinement/full_refinement.py](../../relax/refinement/full_refinement.py) (line 1537):

```python
if frozen_boundary is not None:
    replay_iteration_overrides = frozen_boundary_cli.projector_only_replay_slots(args.max_iter)
    logger.info(
        "Diagnostic frozen restart: local replay slot 0 is projector-only; "
        "sealed per-half scoring state suppresses process-start noise broadcast"
    )
```

### Complete capture admission, attachment and publication

[relax/refinement/full_refinement.py](../../relax/refinement/full_refinement.py) (line 1748):

```python
relion_projector_replay_slot = None
relion_projector_source_manifest_sha256 = None
relion_projector_capture_dir_resolved = None
relion_projector_capture_manifest_resolved = None
if args.relion_projector_capture_dir is not None:
    if args.perturb_replay_relion_dir is None:
        raise SystemExit(
            "--relion-projector-capture-dir requires --perturb_replay_relion_dir"
        )
    if args.relion_projector_capture_iteration is None:
        raise SystemExit(
            "--relion-projector-capture-dir requires "
            "--relion-projector-capture-iteration"
        )
    capture_dir = Path(args.relion_projector_capture_dir).expanduser().resolve()
    capture_manifest = (
        Path(args.relion_projector_capture_manifest).expanduser().resolve()
        if args.relion_projector_capture_manifest is not None
        else capture_dir
        / f"iter{int(args.relion_projector_capture_iteration)}_VALIDATED_SHA256SUMS"
    )
    try:
        relion_projector_replay_slot, projector_state = frozen_boundary_cli.attach_projector_capture(
            replay_iteration_overrides,
            capture_dir=capture_dir,
            manifest_path=capture_manifest,
            capture_iteration=args.relion_projector_capture_iteration,
            init_relion_iteration=args.init_relion_iteration,
            relion_replay_dir=args.perturb_replay_relion_dir,
            volume_shape=ds.volume_shape,
            n_classes=args.n_classes,
            validated_frozen_boundary_iteration=(
                None
                if frozen_boundary is None
                else frozen_boundary.completed_relion_iteration
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid captured RELION projector replay: {exc}") from exc
    relion_projector_source_manifest_sha256 = projector_state[
        "source_manifest_sha256"
    ]
    relion_projector_capture_dir_resolved = capture_dir
    relion_projector_capture_manifest_resolved = capture_manifest
elif (
    args.relion_projector_capture_manifest is not None
    or args.relion_projector_capture_iteration is not None
):
    raise SystemExit(
        "--relion-projector-capture-manifest/iteration require "
        "--relion-projector-capture-dir"
    )
if frozen_boundary is not None:
    frozen_boundary_cli.validate_projector_only_replay_slots(
        replay_iteration_overrides,
        projector_slot=relion_projector_replay_slot,
    )
```

### Existing operations with their diagnostic owner

[relax/diagnostics/frozen_boundary_cli.py](../../relax/diagnostics/frozen_boundary_cli.py) (line 280):

```python
def projector_only_replay_slots(max_iter: int) -> list[dict]:
    """Return projector-only replay slots for a sealed frozen restart.

    A restarted process must not reinterpret its local slot 0 as RELION's
    process-start iteration 0.  In particular, doing so broadcasts half-1
    sigma2_noise over half 2.  Scoring primitives owned by the boundary are
    instead supplied through its sealed initial state; only a separately
    sealed projector may be attached to these slots later.
    """

    count = int(max_iter) + 1
    if count < 2:
        raise ValueError("frozen replay requires at least one numbered iteration slot")
    return [{} for _ in range(count)]
```

[relax/diagnostics/frozen_boundary_cli.py](../../relax/diagnostics/frozen_boundary_cli.py) (line 296):

```python
def validate_projector_only_replay_slots(
    replay_slots: list[dict],
    *,
    projector_slot: int | None = None,
) -> None:
    allowed = {"relion_projector_state"}
    for slot_index, slot in enumerate(replay_slots):
        if slot is None:
            raise ValueError(f"frozen replay slot {slot_index} is missing")
        unexpected = sorted(set(slot) - allowed)
        if unexpected:
            raise ValueError(
                f"frozen replay slot {slot_index} is not projector-only: {unexpected}"
            )
    if projector_slot is not None:
        projector_slot = int(projector_slot)
        if projector_slot < 0 or projector_slot >= len(replay_slots):
            raise ValueError(f"frozen projector slot {projector_slot} is out of range")
        projector_slots = [
            index
            for index, slot in enumerate(replay_slots)
            if "relion_projector_state" in slot
        ]
        if projector_slots != [projector_slot]:
            raise ValueError(
                "frozen replay must contain exactly one projector in numbered "
                f"slot {projector_slot}; got {projector_slots}"
            )
        nonempty_other_slots = [
            index
            for index, slot in enumerate(replay_slots)
            if index != projector_slot and slot
        ]
        if nonempty_other_slots:
            raise ValueError(
                "frozen replay unused/final slots must be empty; got "
                f"{nonempty_other_slots}"
            )
```

[relax/diagnostics/frozen_boundary_cli.py](../../relax/diagnostics/frozen_boundary_cli.py) (line 336):

```python
def attach_projector_capture(
    replay_iteration_overrides,
    *,
    capture_dir,
    manifest_path,
    capture_iteration,
    init_relion_iteration,
    relion_replay_dir,
    volume_shape,
    n_classes,
    validated_frozen_boundary_iteration=None,
):
    """Attach one sealed live projector to its exact numbered replay slot."""

    from relax.relion.relion_metadata import read_relion_model_metadata

    capture_iteration = int(capture_iteration)
    init_relion_iteration = int(init_relion_iteration)
    frozen_iteration = (
        None
        if validated_frozen_boundary_iteration is None
        else int(validated_frozen_boundary_iteration)
    )
    if init_relion_iteration != 0 and frozen_iteration != init_relion_iteration:
        raise ValueError(
            "captured RELION projector replay currently requires an uninterrupted "
            "cold-start trajectory (init_relion_iteration=0); a later jump would "
            "reapply MPI process-start noise semantics without a validated frozen boundary"
        )
    if frozen_iteration is not None and capture_iteration != frozen_iteration + 1:
        raise ValueError(
            "frozen-boundary projector capture must represent the immediately following "
            f"numbered iteration: boundary={frozen_iteration}, capture={capture_iteration}"
        )
    replay_slot = capture_iteration - init_relion_iteration - 1
    if replay_iteration_overrides is None:
        raise ValueError("captured RELION projector requires trajectory replay overrides")
    if replay_slot < 0 or replay_slot >= len(replay_iteration_overrides):
        raise ValueError(
            "captured RELION projector iteration is outside the configured replay trajectory: "
            f"capture_iteration={capture_iteration}, init_relion_iteration={init_relion_iteration}, "
            f"replay_slots={len(replay_iteration_overrides)}"
        )
    existing = replay_iteration_overrides[replay_slot]
    if existing is None:
        raise ValueError(f"captured RELION projector replay slot {replay_slot} has no state override")
    if "relion_projector_state" in existing:
        raise ValueError(f"captured RELION projector replay slot {replay_slot} is already populated")

    relion_replay_dir = Path(relion_replay_dir).expanduser().resolve()
    model_candidates = (
        relion_replay_dir / f"run_it{capture_iteration:03d}_half1_model.star",
        relion_replay_dir / f"run_it{capture_iteration:03d}_model.star",
    )
    model_path = next((path for path in model_candidates if path.is_file()), None)
    if model_path is None:
        raise ValueError(
            "captured RELION projector has no matching replay control model: "
            + " or ".join(str(path) for path in model_candidates)
        )
    model_metadata = read_relion_model_metadata(model_path)
    current_size = int(model_metadata["current_image_size"])
    if current_size <= 0:
        raise ValueError(f"invalid captured-projector replay current size: {current_size}")

    capture_dir = Path(capture_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    projector_state = build_relion_projector_replay_state(
        capture_dir,
        manifest_path=manifest_path,
        iteration=capture_iteration,
        current_size=current_size,
        volume_shape=tuple(int(value) for value in volume_shape),
        n_classes=int(n_classes),
    )
    replay_iteration_overrides[replay_slot] = {
        **existing,
        "relion_projector_state": projector_state,
    }
    logger.info(
        "STRICT-PARITY: attached captured RELION Projector::data iteration=%d "
        "replay_slot=%d current_size=%d manifest=%s",
        capture_iteration,
        replay_slot,
        current_size,
        projector_state["source_manifest_sha256"],
    )
    return replay_slot, projector_state
```

[relax/diagnostics/frozen_boundary_cli.py](../../relax/diagnostics/frozen_boundary_cli.py) (line 426):

```python
def expand_boundary_noise(noise_radial_per_half, image_shape):
    """Expand sealed radial noise using the float32 scoring dtype.

    Frozen-boundary radial profiles are stored as float64 so their serialized
    RELION shell values remain lossless.  The captured physical boundary,
    however, scores with float32 full-image noise arrays.  With JAX x64
    enabled, an untyped ``jnp.asarray`` silently promotes the replay arrays to
    float64 and violates the boundary's dtype and byte-level identity.
    """
    from recovar.reconstruction import noise as recon_noise

    return [
        recon_noise.make_radial_noise(jnp.asarray(radial, dtype=jnp.float32), image_shape)
        for radial in noise_radial_per_half
    ]
```

### Checks and limits

The affected command, startup noise, frozen schema/finalization and import-owner
inventory ran 224 cases: 223 passed and one source guard from the preceding
expectation package still named its old local diagnostic owner. It now checks
construction in expectation and consumption through the shared phase, retaining
the same score-only/noise assertions. All three affected guards then pass, with
zero skips. Production source was unchanged for that repair. Existing numerical
tests still check float32 expansion, per-half identity, projector placement,
adjacent restart admission and atomic captured-state attachment. No scientific
tolerance or baseline changed; no extra tests were added for the body moves.

Structural comparison confirms identical bodies after their renames, unchanged
existing diagnostic functions and unchanged command functions/controller operations
after resolving four module-qualified calls. The diagnostic owner uses the same
logger label and imports no EM execution modules. Retired private entry names and
the command's capture-loader re-export are absent from maintained consumers.
Source/test lint and diff checks pass. The exact inventory, repaired failure,
source manifests and comparison are in the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_sealed_restart_owner_20261002T204921Z/HANDOFF.json).

Command `main()` remains 1,860 lines and the numerical controller 2,913. This
change establishes diagnostic ownership; it does not claim that those controllers
are finished. The complete numbered expectation flow follows below. Broader
controller ownership, the prior Class3D scratch-lifetime choice and final
production float32 K1/exactly-K4, real-data quality, memory and matched-GPU speed
remain open. Qualification waits for the finished representative flow and human
design review. Original work, frozen sources, controls and jobs are preserved.

<a id="numbered-expectation-preparation"></a>

## Earlier numbered expectation preparation

The expectation owner now binds the trial pose grid to the dense or local
sampling support and local diagnostic policy once, before the two halves run.
`NumberedExpectation` borrows those five related fields for this phase. It
contains no dataset, model, history, output collector or controller state.
Changing the dense coarse-grid binding or local diagnostic policy now requires
reading the producer and its sampling/policy types; reconstruction, correction
updates and command admission can remain unopened.

The controller still selects local/adaptive/first-CC modes, generates grids,
prepares projectors and explicitly adapts tomography's 3D sampling. Each half
still builds its own priors, batching and optics, then publishes its returned
poses and accumulators. Serial/overlap dispatch, preprocessing drain, support-count
combination and the end-of-iteration release remain visible below.

### Ownership and lifetime

| Operand | Producer and consumers | Constraint |
| --- | --- | --- |
| Trial grid | Existing perturbation/global-grid path; canonical poses, batching and scoring | Retain supplied arrays and Euler metadata, including deliberate higher precision. Device coarse rotations are a distinct scoring operand. |
| Dense/local sampling | `prepare_numbered_expectation`; both half scorers | Local sampling is the existing result, borrowed unchanged. Dense sampling binds the current trial/coarse grids and Fourier windows without array conversion. |
| Dense variant | Controller's visible mode choices; both half scorers | First-CC, Class3D and adaptive support decisions remain in orchestration. |
| Local diagnostics | Existing local parent-expansion policy reads and run debug settings | Preserve the parent-oversampling guard, read order, physical versus relative iteration IDs and borrowed profile list. |
| Half operands | Existing persistent particle, reference, noise and follower owners | Prepared inside the half worker; never retained in the shared phase. |
| Phase release | Existing end-of-iteration boundary | Clear the phase after map readiness and accumulator cleanup, before next-iteration projector allocation. No unused scientific scratch is added to emulate old locals. |

### Existing window, local and adaptive decisions

These operations remain in the controller. Coarse sizing continues to use the
captured incoming order, while local fine sampling uses the updated state.
Expected accuracy and angular-transition timing remain earlier in this loop.

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 1460):

```python
use_local = state.do_local_search and all(half.rotation_eulers is not None for half in halves)
if use_local and k_class_enabled:
    # Class3D keeps global searches: RELION switches to local searches from the
    # HEALPix order only under auto-refine (ml_optimiser.cpp:2541-2565, 3936-3938).
    raise RuntimeError("K>1 (Class3D) reached local angular searches; RELION never does")
```

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 1536):

```python
expectation_windows = plan_expectation_windows(
    scoring_current_size,
    image_geometry,
    model_pixel_size=model_pixel_size,
    optics_image_sizes=None if multi_shape_halves else optics_image_sizes,
    optics_pixel_sizes=optics_pixel_sizes,
    log=logger,
)
model_current_size_for_engine = expectation_windows.model_window_size
image_current_size = expectation_windows.image_size
cs_for_engine = expectation_windows.image_window_size
sigma_rot, sigma_psi = relion_local_search_sigmas(
    state.sigma_rot,
    state.sigma_psi,
    use_local=use_local,
    healpix_order=state.healpix_order,
    adaptive_oversampling=state.adaptive_oversampling,
)

# Angular step behind this iteration's pass-1 coarse size, when RELION's
# adaptive formula sets it (shape classes recompute their own from it).
if use_local:
    local_sampling = prepare_numbered_local_sampling(
        LocalSearchSettings(
            healpix_order=state.healpix_order + state.adaptive_oversampling,
            oversampling_order=int(state.adaptive_oversampling) if state.adaptive_oversampling > 0 else 0,
            sigma_rot=sigma_rot,
            sigma_psi=sigma_psi,
            symmetry=symmetry,
        ),
        sampling.TrialGrid(
            rotations=effective_rotations,
            rotation_eulers=effective_rotation_eulers,
            mstep_rotations=effective_mstep_rotations,
            translations=current_translations,
        ),
        base_translations=base_translations,
        image_window_size=cs_for_engine,
        model_support_size=model_current_size_for_engine,
        base_healpix_order=current_rotation_grid.healpix_order,
        coarse_size_healpix_order=coarse_size_healpix_order,
        perturbation=random_perturbation,
        model_pixel_size=model_pixel_size,
        original_model_size=grid_size,
        optics_image_sizes=optics_image_sizes,
        optics_pixel_sizes=optics_pixel_sizes,
        particle_diameter_angstrom=particle_diameter_ang,
        log=logger,
    )
else:
    local_sampling = None
direction_prior_healpix_order = _direction_prior_healpix_order_for_scoring(
    use_local=use_local,
    current_healpix_order=current_rotation_grid.healpix_order,
    state_healpix_order=state.healpix_order,
    adaptive_oversampling=state.adaptive_oversampling,
    local_search_order=local_sampling.search.healpix_order if use_local else None,
)
coarse_rotation_ids_for_scoring = (
    _sealed_sampling_rotation_ids(sealed_sampling_state)
    if sealed_sampling_state is not None and not use_local
    else None
)
if (
    coarse_rotation_ids_for_scoring is not None
    and coarse_rotation_ids_for_scoring.shape != (int(effective_rotations.shape[0]),)
):
    raise RuntimeError(
        "sealed captured rotation IDs do not match the directly materialized scorer grid"
    )

for _half_idx in range(2):
    half_direction_priors = relion_direction_log_priors_for_half(
        use_local=use_local,
        scoring_healpix_order=direction_prior_healpix_order,
        n_classes=n_classes,
        priors=direction_priors[_half_idx],
        sealed_sampling_state=sealed_sampling_state,
        dtype=scoring_dtype,
        log=logger,
        half_index=_half_idx,
        symmetry=symmetry,
    )
    direction_log_priors[_half_idx] = half_direction_priors

# --- Run E+M on each half-set ---
# Two modes: single-pass (adaptive_oversampling=0) or two-pass
# coarse/fine (adaptive_oversampling>=1).
significance = SignificanceStatistics()
use_adaptive = _should_use_adaptive_search(
    adaptive_oversampling=state.adaptive_oversampling,
    use_local=use_local,
    n_rotations=effective_rotations.shape[0],
    symmetry=symmetry,
)
# Track the rotation grids used for pose extraction.
# When adaptive oversampling is active, ha_k indices refer to the
# oversampled grid (from pass 2), not effective_rotations.
per_half = PerHalfOutputs()
hard_assignments = per_half.hard_assignments
class_assignments = per_half.class_assignments
class_posterior_per_half = per_half.class_posterior
class_full_posterior_per_half = per_half.class_full_posterior
max_posterior_per_half = per_half.max_posterior
rotation_posterior_per_half = per_half.rotation_posterior
class_rotation_posterior_per_half = per_half.class_rotation_posterior
pose_rotations = per_half.pose_rotations  # rotations to use with ha for poses
pose_rotation_eulers = per_half.pose_rotation_eulers
best_pose_rotations = per_half.best_pose_rotations
best_pose_rotation_eulers = per_half.best_pose_rotation_eulers
best_pose_translations = per_half.best_pose_translations
translation_search_bases = per_half.translation_search_bases
# Coarse-grid assignments for local search tracking (always indexed
# into effective_rotations, even when adaptive oversampling is used).
coarse_ha = per_half.coarse_ha
if use_adaptive:
    # --- TWO-PASS ADAPTIVE OVERSAMPLING (RELION parity) ---
    # Pass 1: coarse E-step at reduced resolution to find
    #         significant orientations.
    # Pass 2: oversampled E+M at full current_size for significant
    #         orientations only.

    coarse_image_plan = plan_adaptive_image_size(
        coarse_size_healpix_order,
        expectation_windows,
        image_geometry,
        particle_diameter_angstrom=particle_diameter_ang,
        optics_image_sizes=optics_image_sizes,
        optics_pixel_sizes=optics_pixel_sizes,
        sealed_sampling_state=sealed_sampling_state,
        log=logger,
    )
    coarse_size = coarse_image_plan.size
    coarse_cs = coarse_size if coarse_size < grid_size else None

    logger.info(
        "Adaptive oversampling: pass 1 at coarse_size=%s, "
        "pass 2 at current_size=%s (oversampling=%d, particle_diameter=%s)",
        coarse_cs,
        cs_for_engine,
        state.adaptive_oversampling,
        (f"{float(particle_diameter_ang):.1f} A" if particle_diameter_ang is not None else "box_size"),
    )
```

### Complete phase construction, half execution and dispatch

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 1760):

```python
numbered_variant = DenseVariantPolicy(
    firstiter_score_mode_this_iter=firstiter_score_mode_this_iter,
    firstiter_winner_take_all_this_iter=firstiter_winner_take_all_this_iter,
    k_class_enabled=k_class_enabled,
    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
    firstiter_coarse_current_size=coarse_cs if use_adaptive else None,
    firstiter_fine_current_size=cs_for_engine if use_adaptive else None,
    firstiter_log_label="" if use_adaptive else "(non-adaptive site) ",
    firstiter_updates_em_kwargs_ibs=bool(use_adaptive),
)
numbered_expectation = prepare_numbered_expectation(
    sampling.TrialGrid(
        rotations=effective_rotations,
        rotation_eulers=effective_rotation_eulers,
        mstep_rotations=effective_mstep_rotations,
        translations=current_translations,
    ),
    expectation_windows,
    local_sampling=local_sampling,
    variant=numbered_variant,
    use_adaptive=use_adaptive,
    base_translations=base_translations,
    current_healpix_order=current_rotation_grid.healpix_order,
    oversampling_order=state.adaptive_oversampling,
    translation_step=state.translation_step,
    random_perturbation=random_perturbation,
    adaptive_pass1_rotations=adaptive_pass1_rotations,
    coarse_rotation_ids=coarse_rotation_ids_for_scoring,
    coarse_angular_step_deg=coarse_image_plan.angular_step_deg if use_adaptive else None,
    options=options,
    iteration=iteration,
    numbered_relion_iteration=numbered_relion_iteration,
    collect_local_search_profile=collect_local_search_profile,
    local_profile_history=history.local_profile_history,
)
if tomo_halves:
    tomo_oversampling = int(state.adaptive_oversampling)
    tomo_coarse_size = local_sampling.coarse_image_window_size if use_local else coarse_cs
    numbered_tomo_sampling = TomoSampling(
        healpix_order=int(local_sampling.search.healpix_order) - tomo_oversampling if use_local else int(current_rotation_grid.healpix_order),
        oversampling_order=tomo_oversampling,
        offset_range_angst=float(state.translation_range) * image_geometry.pixel_size_angstrom,
        offset_step_angst=float(state.translation_step) * image_geometry.pixel_size_angstrom,
        random_perturbation=float(local_sampling.perturbation if use_local else random_perturbation),
        coarse_size=int(cryo.image_shape[0] if tomo_coarse_size is None else tomo_coarse_size),
        fine_size=int(cryo.image_shape[0] if cs_for_engine is None else cs_for_engine),
    )
else:
    numbered_tomo_sampling = None

def _run_half_estep(k):
    particle_half = halves[k]
    score_result = score_numbered_half(
        HalfScoringData(
            particles=particle_half,
            reference=reference_model.maps[k],
            mean_variance=reference_model.tau2_per_half[k],
            noise_variance=noise_model.variance_per_half[k],
            noise_radial=noise_model.radial_per_half[k] if not tomo_halves and particle_half.dataset.n_units else None,
            projector=projectors[k],
            scale_group_ids=follower_setup.scale_stats_group_ids_per_half[k],
            scale_group_count=follower_setup.scale_stats_group_count_per_half[k],
            scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
        ),
        numbered_expectation,
        tomo_sampling=numbered_tomo_sampling,
        direction_priors=direction_log_priors[k],
        class_log_priors=class_log_priors,
        sigma_offset_angstrom=_sigma_offset_for_half(
            current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half, k,
        ),
        batch_planner=batch_planner,
        image_geometry=image_geometry,
        padded_volume_shape=padded_volume_shape,
        multi_shape_halves=multi_shape_halves,
        options=options,
        replay_prior_translations=_replay_prior_translations,
        initial_class_assignments=k_class.first_iteration_seed_classes if seed_iteration else None,
        single_class_iteration=single_class_iteration,
        scoring_dtype=scoring_dtype,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        relion_translation_angle_scale=relion_translation_angle_scale,
        iteration=iteration,
        numbered_relion_iteration=numbered_relion_iteration,
    )
    per_half.translation_search_bases[k] = score_result.translation_search_base
    per_half.pose_rotations[k] = score_result.pose_rotations
    per_half.pose_rotation_eulers[k] = score_result.pose_rotation_eulers
    per_half.coarse_ha[k] = score_result.coarse_ha
    if particle_half.dataset.n_units != 0:
        score_result = _maybe_host_offload_half0_local_accumulators(
            half_index=k,
            use_local=use_local,
            k_class_enabled=k_class_enabled,
            score_result=score_result,
            log=logger,
        )
    per_half.update_from(k, score_result, dtype=scoring_dtype)
    record_numbered_half(
        score_result,
        particle_half,
        per_half,
        significance,
        profile_history=history.global_profile_history,
        iteration=iteration,
        image_window_size=cs_for_engine,
        healpix_order=current_rotation_grid.healpix_order,
        k_class_enabled=k_class_enabled,
    )

_overlap_active = _half_overlap_active(
    options.overlap.overlap_halves,
    diagnostic_half_indices=diagnostic_half_indices,
    log=logger,
)
if _overlap_active:
    _run_halves_overlapped(_run_half_estep, diagnostic_half_indices)
    k = diagnostic_half_indices[-1]
else:
    for k in diagnostic_half_indices:
        _run_half_estep(k)
if diagnostic_half_indices != (0, 1):
    raise RuntimeError(
        "targeted half-only significance diagnostic returned without writing its "
        "complete target set; refusing to continue with one half missing"
    )

Ft_y_0, Ft_y_1 = per_half.Ft_y
Ft_ctf_0, Ft_ctf_1 = per_half.Ft_ctf

# E-step + per-half M-step accumulators are now both populated.
_parity_dump.mark_stage(iteration, "e_step")
from relax.cuda.kernels import drain_relion_preprocess_checks
drain_relion_preprocess_checks()
significance.combine()
```

### Shared phase and preparation implementation

[relax/refinement/expectation.py](../../relax/refinement/expectation.py) (line 326):

```python
@dataclass(frozen=True, eq=False, kw_only=True)
class NumberedExpectation:
    """Trial pose grid and resolved sampling/policies shared by both numbered halves."""

    grid: TrialGrid
    sampling: LocalSampling | DenseSamplingSpec
    variant: DenseVariantPolicy
    use_adaptive: bool
    local_diagnostics: LocalDiagnosticPolicy | None
```

[relax/refinement/expectation.py](../../relax/refinement/expectation.py) (line 337):

```python
def prepare_numbered_expectation(
    grid: TrialGrid,
    windows: ExpectationWindows,
    *,
    local_sampling: LocalSampling | None,
    variant: DenseVariantPolicy,
    use_adaptive: bool,
    base_translations,
    current_healpix_order: int,
    oversampling_order: int,
    translation_step: float,
    random_perturbation: float,
    adaptive_pass1_rotations,
    coarse_rotation_ids,
    coarse_angular_step_deg,
    options: RefinementOptions,
    iteration: int,
    numbered_relion_iteration: int,
    collect_local_search_profile: bool,
    local_profile_history: list,
) -> NumberedExpectation:
    """Bind the numbered grid to dense/local support and local diagnostic policy.

    Dense scoring may use device coarse rotations while poses retain the supplied
    canonical trial rows.
    See ``docs/math/relion_refinement_algorithm.md#numbered-expectation-preparation``.
    """
    if local_sampling is not None:
        if local_sampling.search.oversampling_order > 0:
            local_adaptive_full_parent = _local_adaptive_pass2_full_parent_enabled()
            local_adaptive_rotation_only = _local_adaptive_pass2_rotation_only_enabled()
            local_adaptive_denominator_mode = (
                _local_adaptive_pass2_denominator_support_mode()
            )
        else:
            local_adaptive_full_parent = False
            local_adaptive_rotation_only = False
            local_adaptive_denominator_mode = None
        numbered_sampling = local_sampling
        numbered_local_diagnostics = LocalDiagnosticPolicy(
            iteration=iteration,
            debug_iteration=numbered_relion_iteration,
            save_intermediates_dir=options.debug.save_intermediates_dir,
            collect_local_search_profile=collect_local_search_profile,
            diagnostic_score_only=bool(options.debug.stop_after_local_search_score_only),
            local_profile_history=local_profile_history,
            adaptive_pass2_full_parent=local_adaptive_full_parent,
            adaptive_pass2_rotation_only=local_adaptive_rotation_only,
            adaptive_pass2_denominator_mode=local_adaptive_denominator_mode,
        )
    else:
        numbered_sampling = DenseSamplingSpec(
            effective_rotations=(
                adaptive_pass1_rotations
                if use_adaptive and adaptive_pass1_rotations is not None
                else grid.rotations
            ),
            current_translations=grid.translations,
            base_translations=base_translations,
            current_healpix_order=current_healpix_order,
            oversampling_order=oversampling_order,
            translation_step=translation_step,
            coarse_engine=options.adaptive.coarse_engine,
            random_perturbation=random_perturbation,
            cs_for_engine=windows.image_window_size,
            model_current_size_for_engine=windows.model_window_size,
            coarse_rotation_ids=coarse_rotation_ids,
            coarse_scoring_rotations=adaptive_pass1_rotations if int(oversampling_order) == 0 else None,
            coarse_angular_step_deg=coarse_angular_step_deg,
            symmetry=options.symmetry.point_group,
        )
        numbered_local_diagnostics = None
    return NumberedExpectation(
        grid=grid,
        sampling=numbered_sampling,
        variant=variant,
        use_adaptive=use_adaptive,
        local_diagnostics=numbered_local_diagnostics,
    )
```

### Half-specific computation and engine consumption

[relax/refinement/expectation.py](../../relax/refinement/expectation.py) (line 418):

```python
def score_numbered_half(
    half: HalfScoringData,
    phase: NumberedExpectation,
    *,
    tomo_sampling: TomoSampling | None,
    direction_priors: HalfDirectionLogPriors,
    class_log_priors,
    sigma_offset_angstrom,
    batch_planner: BatchPlanner,
    image_geometry: ImageGeometry,
    padded_volume_shape,
    multi_shape_halves: bool,
    options: RefinementOptions,
    replay_prior_translations,
    initial_class_assignments,
    single_class_iteration: bool,
    scoring_dtype,
    source_faithful_spectrum_norm: bool,
    relion_translation_angle_scale: float,
    iteration: int,
    numbered_relion_iteration: int,
) -> HalfScoreResult:
    """Build half-specific priors/batches and accumulate an empty, SPA or tomography half.

    Canonical pose grids and the applied translation base accompany the result.
    The controller owns publication, accumulator offloading and post-score capture.
    """
    sampling = phase.sampling
    particle_half = half.particles
    k = particle_half.index
    use_local = isinstance(sampling, LocalSampling)
    tomo_halves = tomo_sampling is not None
    k_class_enabled = phase.variant.k_class_enabled
    image_window_size = sampling.image_window_size if use_local else sampling.cs_for_engine
    model_support_size = sampling.model_support_size if use_local else sampling.model_current_size_for_engine
    symmetry = sampling.search.symmetry if use_local else sampling.symmetry
    coarse_size_step_deg = sampling.coarse_angular_step_deg
    particle_diameter_ang = options.schedule.particle_diameter_ang
    bpref_diagnostics.set_bpref_contribution_dump_context(
        iteration=iteration + 1,
        half=particle_half.index + 1,
    )
    bpref_device_signature_active = (
        _bpref_device_signature_active_for_numbered_half(
            iteration=iteration + 1,
            half=particle_half.index + 1,
        )
    )
    logger.info(
        "BPREF_DEVICE_SIGNATURE_ACTIVATION iteration=%d half=%d "
        "final_all_data=false active=%s",
        iteration + 1,
        particle_half.index + 1,
        str(bpref_device_signature_active).lower(),
    )
    previous_translations_k = particle_half.translations
    translation_search_base = relion_translation_search_base(
        previous_translations_k, dtype=scoring_dtype
    )
    half_batching = prepare_half_batches(
        particle_half.dataset,
        half.projector,
        planner=batch_planner,
        rotations=phase.grid.rotations if phase.use_adaptive or k_class_enabled else None,
        translations=phase.grid.translations,
        cs_for_engine=image_window_size,
        coarse_cs=phase.variant.firstiter_coarse_current_size if phase.use_adaptive else None,
        model_current_size_for_engine=model_support_size,
        use_adaptive=phase.use_adaptive,
        use_local=use_local,
        relion_firstiter_cc_this_iter=phase.variant.relion_firstiter_cc_this_iter,
        firstiter_winner_take_all_this_iter=phase.variant.firstiter_winner_take_all_this_iter,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
        image_fourier_backend=options.parity.image_fourier_backend,
        bpref_device_signature_active=bpref_device_signature_active,
        multi_shape_halves=multi_shape_halves,
        coarse_sizing=(coarse_size_step_deg, particle_diameter_ang) if phase.use_adaptive and multi_shape_halves else None,
    )
    # RELION translation priors: relion_half_translation_prior_inputs
    # documents the pdf_offset / wsum_sigma2_offset centers, the
    # cold-start engine center and the prior-grid selection.
    if tomo_halves:
        # A subtomogram half builds its 3D offset priors per particle (score_tomo_half).
        translation_prior_inputs = None
        trans_prior_center = local_trans_prior_center = trans_prior_center_for_engine = None
    else:
        translation_prior_inputs = relion_half_translation_prior_inputs(
            previous_translations_k,
            voxel_size=image_geometry.pixel_size_angstrom,
            base_translations=sampling.base_translations,
            current_translations=phase.grid.translations,
            dtype=scoring_dtype,
        )
        trans_prior_center = translation_prior_inputs.prior_center
        local_trans_prior_center = translation_prior_inputs.local_prior_center
        trans_prior_center_for_engine = translation_prior_inputs.engine_prior_center
    translation_log_prior = None
    if not use_local and not tomo_halves:
        if not k_class_enabled and trans_prior_center is None:
            # A fresh K1 half has implicit zero offsets, not a flat
            # pdf_offset. Native ACC applies the Gaussian even at
            # iteration 1; see relion_refinement_algorithm.md.
            trans_prior_center = np.zeros(2, dtype=scoring_dtype)
        translation_log_prior = make_relion_translation_log_prior(
            translation_prior_inputs.prior_translations,
            image_geometry.pixel_size_angstrom,
            sigma_offset_angstrom,
            trans_prior_center,
            offset_range_pixels=None,
            dtype=scoring_dtype,
        )
    if particle_half.dataset.n_units == 0:
        logger.info("Skipping E-step/M-step accumulation for empty half-%d dataset", particle_half.index + 1)
        n_shells = int(image_geometry.image_shape[0] // 2 + 1)
        n_rot_for_stats = int(
            rotation_grid_size(sampling.search.healpix_order, symmetry=symmetry) if use_local else phase.grid.rotations.shape[0]
        )
        empty_k1_x_half_mstep = (
            (not k_class_enabled)
            and (use_local or phase.use_adaptive)
            and _k1_relion_x_half_mstep_enabled()
        )
        empty_result = empty_half_result(
            volume_shape=particle_half.dataset.volume_shape if empty_k1_x_half_mstep else None,
            padded_volume_shape=padded_volume_shape,
            n_classes=batch_planner.n_classes,
            n_shells=n_shells,
            n_rotations=n_rot_for_stats,
            translation_dimension=3 if tomo_halves else phase.grid.translations.shape[1],
            image_window_size=image_window_size,
            model_support_size=model_support_size,
            use_x_half_mstep=empty_k1_x_half_mstep,
        )
        empty_result.coarse_ha = empty_result.ha
        empty_result.translation_search_base = translation_search_base
        return empty_result
    # The half's units' classes in RELION's seed iteration, by particle row (all class 0 in the CC
    # iteration before it).
    seed_classes_k = (
        np.asarray(initial_class_assignments)[
            particle_half.dataset._index_layout.original_image_indices_for_local(
                np.arange(particle_half.dataset.n_units)
            )
        ]
        if initial_class_assignments is not None
        else np.zeros(particle_half.dataset.n_units, dtype=np.int64)
        if single_class_iteration
        else None
    )
    if tomo_halves:
        score_result = _score_tomo_half_in_loop(
            particle_half.dataset,
            use_local=use_local,
            use_adaptive=phase.use_adaptive,
            volume=half.reference,
            noise_variance=half.noise_variance,
            relion_projector_half=None if half.projector is None else half.projector.data,
            relion_projector_r_max=None if half.projector is None else half.projector.r_max,
            sampling=tomo_sampling,
            local_search=(
                dict(
                    previous_eulers_deg=particle_half.rotation_eulers,
                    sigma_rot=sampling.search.sigma_rot,
                    sigma_psi=sampling.search.sigma_psi,
                )
                if use_local
                else None
            ),
            rotation_log_prior=direction_priors.rotation_log_prior,
            previous_translations=previous_translations_k,
            sigma_offset_angst=sigma_offset_angstrom,
            max_significants=options.adaptive.max_significants,
            unit_groups=particle_half.optics_group_ids,
            scale_corrections=particle_half.scale_corrections,
            group_ids=half.scale_group_ids,
            scale_correction_group_count=half.scale_group_count,
            scale_correction_data_vs_prior=half.scale_correction_data_vs_prior,
            reconstruction_current_size=model_support_size,
            symmetry=symmetry,
            class_log_priors=class_log_priors if k_class_enabled else None,
            class_rotation_log_prior=direction_priors.class_rotation_log_prior if k_class_enabled else None,
            unit_seed_classes=seed_classes_k,
            normalized_cc=phase.variant.firstiter_score_mode_this_iter == "normalized_cc",
        )
    elif use_local:
        local_optics = optics_shapes.prepare_optics(
            particle_half.dataset,
            noise_radial=half.noise_radial,
            coarse_step_deg=coarse_size_step_deg,
            particle_diameter_ang=particle_diameter_ang,
            previous_translations=previous_translations_k,
            sigma_offset_angstrom=sigma_offset_angstrom,
            base_translations=sampling.base_translations,
            current_translations=phase.grid.translations,
            with_log_prior=False,
            zero_cold_center=not k_class_enabled,
            dtype=scoring_dtype,
        )
        local_result = _score_half_local_in_bpref_scope(
            half=replace(half, mean_variance=None),
            sampling=sampling,
            priors=LocalPriorSpec(
                trans_prior_center=local_trans_prior_center,
                trans_prior_center_for_engine=trans_prior_center_for_engine,
                current_sigma_offset_angstrom=sigma_offset_angstrom,
                translation_search_base=translation_search_base,
                local_search_translation_prior_mode=(options.local_search.local_search_translation_prior_mode),
                replay_prior_translations=replay_prior_translations,
            ),
            batching=LocalBatchPolicy(
                max_significants=options.adaptive.max_significants,
                safe_batch_sizes=half_batching.safe_batch_sizes,
            ),
            execution=LocalExecutionPolicy(
                disc_type=options.disc_type,
                disable_adjoint_y=options.debug.disable_adjoint_y,
                disable_adjoint_ctf=options.debug.disable_adjoint_ctf,
                source_faithful_spectrum_norm=(source_faithful_spectrum_norm),
                relion_translation_angle_scale=(relion_translation_angle_scale),
            ),
            diagnostics=replace(
                phase.local_diagnostics, bpref_device_signature_active=bpref_device_signature_active,
            ),
            optics=local_optics,
        )
        if local_result.coarse_ha is None:
            local_result.coarse_ha = local_result.ha
        score_result = local_result

    else:
        # Shared dense half-scoring operands; the adaptive branch adds its
        # pass-1 grid and batch/size overrides.
        dense_optics = optics_shapes.prepare_optics(
            particle_half.dataset,
            noise_radial=half.noise_radial,
            coarse_step_deg=coarse_size_step_deg,
            particle_diameter_ang=particle_diameter_ang,
            previous_translations=previous_translations_k,
            sigma_offset_angstrom=sigma_offset_angstrom,
            base_translations=sampling.base_translations,
            current_translations=phase.grid.translations,
            with_log_prior=not use_local,
            zero_cold_center=not k_class_enabled,
            dtype=scoring_dtype,
        )
        dense_half = replace(half, image_seed_classes=seed_classes_k)
        dense_sampling = sampling
        dense_priors = DensePriorSpec(
            rotation_log_prior_k=direction_priors.rotation_log_prior,
            class_rotation_log_prior_k=direction_priors.class_rotation_log_prior,
            translation_log_prior=translation_log_prior,
            translation_search_base=translation_search_base,
            trans_prior_center_for_engine=trans_prior_center_for_engine,
            class_log_priors=class_log_priors,
        )
        dense_batching = DenseBatchPolicy(
            image_batch_size=options.batching.image_batch_size,
            max_significants=options.adaptive.max_significants,
            safe_batch_sizes=half_batching.safe_batch_sizes,
            significance_safe_batch_sizes=half_batching.significance_safe_batch_sizes,
            class_batch_overrides=half_batching.class_overrides,
            k_class_image_batch_size_override=(half_batching.fine_image_batch_size if phase.use_adaptive else None),
            k_class_rotation_block_size_override=(half_batching.fine_rotation_block_size if phase.use_adaptive else None),
            significance_image_batch_size_override=(half_batching.coarse_image_batch_size if phase.use_adaptive else None),
            significance_rotation_block_size_override=(half_batching.coarse_rotation_block_size if phase.use_adaptive else None),
        )
        dense_variant = phase.variant
        dense_execution = DenseExecutionPolicy(
            disc_type=options.disc_type,
            disable_adjoint_y=options.debug.disable_adjoint_y,
            disable_adjoint_ctf=options.debug.disable_adjoint_ctf,
            bpref_device_signature_active=bpref_device_signature_active,
            debug_iteration=numbered_relion_iteration,
            diagnostic_float64_pass2=_diagnostic_float64_pass2_matches(
                numbered_relion_iteration
            ),
            preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_translation_angle_scale=relion_translation_angle_scale,
        )
        dense_result = _score_half_dense_in_bpref_scope(
            dense_half,
            dense_sampling,
            dense_priors,
            dense_batching,
            dense_variant,
            dense_execution,
            dense_optics,
        )
        if not phase.use_adaptive or dense_result.pose_rotations is None:
            if dense_result.pose_rotations is None:
                dense_result.pose_rotations = phase.grid.rotations
            if dense_result.pose_rotation_eulers is None:
                dense_result.pose_rotation_eulers = phase.grid.rotation_eulers
        if dense_result.coarse_ha is None:
            dense_result.coarse_ha = dense_result.ha
        score_result = dense_result

        # --- Manifest dump for deterministic replay (Phase 0.1) ---
        if not phase.use_adaptive and options.debug.save_intermediates_dir is not None:
            _manifest_path = os.path.join(
                options.debug.save_intermediates_dir,
                f"manifest_iter{iteration}_half{k}.npz",
            )
            _manifest = {
                "effective_rotations": np.asarray(phase.grid.rotations),
                "coarse_scoring_rotations": _replay_manifest_array(
                    dense_sampling.coarse_scoring_rotations,
                ),
                "current_translations": np.asarray(phase.grid.translations),
                "rotation_log_prior": _replay_manifest_array(direction_priors.rotation_log_prior, dtype=np.float64),
                "translation_log_prior": _replay_manifest_array(translation_log_prior, dtype=np.float64),
                "image_corrections": _replay_manifest_array(
                    particle_half.image_corrections, dtype=np.float64,
                ),
                "scale_corrections": _replay_manifest_array(
                    particle_half.scale_corrections, dtype=np.float64,
                ),
                "image_pre_shifts": _replay_manifest_array(translation_search_base, dtype=np.float32),
                "absolute_previous_translations": _replay_manifest_array(
                    previous_translations_k, dtype=np.float32,
                ),
                "mean_vol_ft": np.asarray(half.reference),
                "mean_variance": np.asarray(half.mean_variance),
                "noise_variance": np.asarray(half.noise_variance),
                "current_size": np.int32(image_window_size) if image_window_size is not None else np.int32(-1),
                "half_spectrum_scoring": np.bool_(True),
                "use_float64_scoring": np.bool_(scoring_policy.DENSE_PRECISION.use_float64_scoring),
                "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                "reconstruction_padding_factor": np.int32(RECONSTRUCTION_PADDING_FACTOR),
                "score_with_masked_images": np.bool_(True),
                "perturbation_instance": np.float64(sampling.random_perturbation),
                "perturbation_factor": np.float64(options.parity.perturb_factor),
                "iteration": np.int32(iteration),
                "half_index": np.int32(k),
                "ave_Pmax": np.float64(float(np.mean(dense_result.em_stats.max_posterior_per_image))),
            }
            np.savez(_manifest_path, **_manifest)
            logger.info("Manifest dumped: %s", _manifest_path)

    score_result.translation_search_base = translation_search_base
    return score_result
```

### End-of-iteration memory boundary

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 2960):

```python
# End-of-iteration memory boundary.  The next iteration immediately
# pads each half-map to the projection grid; keeping previous
# backprojector accumulators or unregularized diagnostic maps live can
# make high-resolution runs OOM before the batch-size estimator can act.
try:
    jax.block_until_ready(reference_model.maps)
except Exception:
    pass
Ft_y_0 = Ft_y_1 = None
Ft_ctf_0 = Ft_ctf_1 = None
Ft_y_combined = Ft_ctf_combined = None
unreg_means = previous_means = None
mean_signal_variance_per_half = mean_signal_variance_shells_per_half = tau2_update_details_per_half = None
noise_stats_per_half = noise_stats_per_half_per_class = None
# Pass containers must not retain the previous grids while the next projector is built.
numbered_expectation = numbered_tomo_sampling = numbered_variant = None
if parse_env_true_flag("RELAX_RELION_CLEAR_JAX_CACHES_BETWEEN_ITERS"):
    jax.clear_caches()
```

### Checks and remaining work

The initial focused run passed 132 cases and found nine test issues: eight new
expectations confused the full model window with the image-window sentinel,
and one source guard still looked in the previous owner. The existing rule keeps
explicit full model support when particle support differs. After fixing those
test expectations and migrating the source guard, all 43 repair cases and 27
affected controller cases pass, with zero skips. The CPU EM guard also passes
all 108 cases. No production change was needed for those failures, and no
existing scientific assertion, tolerance or baseline was weakened.

Structural comparison resolves the phase fields and proves the same sampling
constructors, policy reads, scoring body and controller operations outside this
boundary. Pure borrowed metadata construction now precedes the local policy
reads; neither constructor computes arrays. Canonical/device-grid distinctions,
window sentinels, inactive metadata guards and the original release boundary
have focused tests. Exact commands, XML, source manifests, the failed-test
repairs and comparison limitations are in the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_numbered_expectation_preparation_20261002T202923Z/HANDOFF.json).

This introduces one result type and one substantive producer in the existing
expectation module, without a new module, forwarding method, flag, kernel or
JIT boundary. The scorer has 19 inputs, the producer 18; the 59-line half worker
captures 34 names. The numerical controller remains 2,913 lines and command
`main()` 1,860. These counts expose unfinished ownership work; they do not
establish readability or numerical acceptance. Broader controller work and the
pending Class3D scratch-lifetime decision remain. Full production float32 K1,
exactly K4, real-data quality, peak memory and matched-GPU speed still require
the completed, reviewed and newly frozen candidate. Existing frozen sources,
controls and jobs remain untouched.

<a id="command-oracle-admission-and-runtime-controls"></a>

## Earlier command oracle admission and runtime controls

The source selection and mapping excerpt below records the earlier interface.
The current particle-table owner combines them in
`prepare_particle_group_layout`, returning the authoritative source and computed
layout together. Dispatch admission and follower topology retain their separate
operations and the same consumers. The historical excerpt is retained as evidence.

The command-options owner now performs two substantial preparation operations:
verified dispatch admission and CLI/optimiser runtime-control resolution. They
replace inline verification and precedence logic in the command controller. A
change to the oracle manifest or the saved significant-support argument now
requires understanding the command-options operation and its existing RELION
format/formula owners; reference loading, reconstruction and iteration execution
can remain unopened.

The complete caller below retains authoritative group-source selection, the
strict Class3D replay decision, follower-topology preparation, frozen-aware
optimiser selection and visible installation of the resolved cap/CTF/filter
controls. Admission still runs before reference loading. The returned records
contain related configuration and provenance, never the controller's numerical
locals. No new module, forwarding method, flag, kernel or JIT boundary is added.

### Producers, consumers and lifetime

| Input or result | Producer | Consumers and retention |
| --- | --- | --- |
| Authoritative particle table and source | Existing `select_group_particle_source`, before admission | Group-layout identity, verified row hash and follower provenance use the same table; no row sorting or array conversion is introduced. |
| Strict replay mode | Visible `K>1` plus init/replay predicate | Dispatch admission and existing follower-topology admission. No missing state becomes a default schedule. |
| Verified dispatch | Existing schema loader and manifest/particle checks | Original `relion_dispatch_schedule` local remains the follower/archive operand. The paired verified roots support follower validation. Integer schedule arrays keep the original root; the extra record contains only that schedule and host paths. |
| Consumed optimiser | Existing frozen-aware selection after follower preparation | Runtime control resolution reads the same file and delegates the active cap to the existing RELION policy. |
| Runtime controls | Saved CTF/`ini_high`, CLI cap and first numbered iteration | Visible option installation, expected accuracy, post-first-CC filtering, archive and benchmark reporting. The same provenance dictionary reaches output. Transient metadata/text is host configuration, not numerical scratch. |

The significant-support resolver takes seven explicit inputs; the dispatch
operation takes the CLI request, selected table and already-decided strict mode.
It uses the CLI namespace for source discovery/admission only. Neither operation
receives the dataset, model, iteration history or GPU arrays.

### Complete source selection and command caller

[relax/refinement/full_refinement.py](../../relax/refinement/full_refinement.py) (line 1054):

```python
group_particle_source = input_particle_table.select_group_particle_source(
    halfset_particles=relion_particles,
    halfset_source=args.relion_half_sets,
    replay_dirs=(args.perturb_replay_relion_dir, args.relion_init_dir),
    init_relion_iteration=args.init_relion_iteration,
)
particle_groups = input_particle_table.resolve_particle_group_layout(
    our_particles,
    particle_layout.half1_rows,
    particle_layout.half2_rows,
    relion_particles=group_particle_source.particles,
)
strict_relion_scale_context = bool(
    args.n_classes > 1
    and (args.perturb_replay_relion_dir is not None or args.relion_init_dir is not None)
)
dispatch = command_options.load_verified_dispatch_schedule(
    args,
    group_particle_source.particles,
    strict_replay=strict_relion_scale_context,
)
relion_dispatch_schedule = dispatch.schedule
follower_topology = prepare_follower_topology(
    args.relion_scale_followers,
    relion_dispatch_schedule,
    particle_groups,
    strict_replay=strict_relion_scale_context,
    replay_path=args.relion_follower_scale_replay,
    oracle_dir=dispatch.oracle_dirs[0] if relion_dispatch_schedule is not None else None,
    random_seed=args.seed,
    init_relion_iteration=args.init_relion_iteration,
    max_iter=args.max_iter,
    group_source=group_particle_source.path,
    logger=logger,
)

optimiser_star = _relion_optimiser_star_for_runtime(
    args,
    frozen_boundary=frozen_boundary,
    fixed_diagnostic_source_paths=fixed_diagnostic_source_paths,
)
runtime_controls = command_options.resolve_relion_runtime_controls(
    optimiser_star,
    max_significants=args.max_significants,
    target_iteration=int(args.init_relion_iteration) + 1,
    firstiter_cc=bool(args.firstiter_cc),
    n_classes=int(args.n_classes),
    log=logger,
)
args.max_significants = runtime_controls.max_significants_resolution["active_max_significants"]
expected_accuracy_do_ctf_correction = runtime_controls.do_ctf_correction
relion_firstiter_ini_high_angstrom = runtime_controls.firstiter_ini_high_angstrom
```

### Verified dispatch result and operation

[relax/refinement/command_options.py](../../relax/refinement/command_options.py) (line 904):

```python
class VerifiedDispatchSchedule(NamedTuple):
    """Captured follower assignments paired with their verified oracle roots."""

    schedule: "RelionDispatchSchedule | None"
    oracle_dirs: list[Path]
```

[relax/refinement/command_options.py](../../relax/refinement/command_options.py) (line 911):

```python
def load_verified_dispatch_schedule(args, particles, *, strict_replay: bool) -> VerifiedDispatchSchedule:
    """Admit a CLI dispatch capture only against its oracle and particle order.

    Verify directory manifests before discovering the consumed optimiser and
    sampling files. See ``docs/math/relion_refinement_algorithm.md#command-admission``.
    """
    from relax.relion.relion_worker_scale import (
        load_relion_dispatch_schedule,
        relion_ordered_particle_sha256,
        verify_relion_dispatch_schedule_oracle,
    )

    relion_dispatch_schedule = None
    oracle_dirs = []
    if args.relion_dispatch_schedule is not None:
        if not strict_replay:
            raise SystemExit(
                "--relion-dispatch-schedule is strict K>1 RELION replay/init state only"
            )
        try:
            relion_dispatch_schedule = load_relion_dispatch_schedule(
                args.relion_dispatch_schedule
            )
            for candidate in (args.perturb_replay_relion_dir, args.relion_init_dir):
                if candidate is None:
                    continue
                resolved = Path(candidate).expanduser().resolve()
                if resolved not in oracle_dirs:
                    oracle_dirs.append(resolved)
                    verify_relion_dispatch_schedule_oracle(
                        relion_dispatch_schedule,
                        resolved,
                    )
            def _require_manifested_oracle_file(path, *, label):
                resolved_path = Path(path).expanduser().resolve()
                for oracle_dir in oracle_dirs:
                    try:
                        relative = resolved_path.relative_to(oracle_dir).as_posix()
                    except ValueError:
                        continue
                    if relative not in relion_dispatch_schedule.oracle_artifact_paths:
                        raise ValueError(
                            f"{label} is not included in the verified RELION oracle manifest: "
                            f"{resolved_path}"
                        )
                    return
                raise ValueError(
                    f"{label} must belong to a verified RELION oracle directory: {resolved_path}"
                )

            discovered_optimiser = find_relion_optimiser_star(args)
            if discovered_optimiser is not None:
                _require_manifested_oracle_file(
                    discovered_optimiser,
                    label="consumed RELION optimiser",
                )
            for oracle_dir in oracle_dirs:
                sampling_candidates = list(oracle_dir.glob("run_it*_sampling.star"))
                final_sampling = oracle_dir / "run_sampling.star"
                if final_sampling.exists():
                    sampling_candidates.append(final_sampling)
                for sampling_path in sampling_candidates:
                    _require_manifested_oracle_file(
                        sampling_path,
                        label="consumed RELION sampling state",
                    )
            observed_group_order = relion_ordered_particle_sha256(particles)
            if observed_group_order != relion_dispatch_schedule.particle_order_sha256:
                raise ValueError(
                    "authoritative RELION group/half-set particle order does not match "
                    "the dispatch schedule"
                )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Invalid --relion-dispatch-schedule: {exc}") from exc
    return VerifiedDispatchSchedule(relion_dispatch_schedule, oracle_dirs)
```

### Resolved controls and operation

[relax/refinement/command_options.py](../../relax/refinement/command_options.py) (line 988):

```python
class RelionRuntimeControls(NamedTuple):
    """Resolved CLI/optimiser controls and significant-support provenance."""

    do_ctf_correction: bool | None
    firstiter_ini_high_angstrom: float | None
    max_significants_resolution: dict
```

[relax/refinement/command_options.py](../../relax/refinement/command_options.py) (line 996):

```python
def resolve_relion_runtime_controls(
    optimiser_star,
    *,
    max_significants: int | None,
    target_iteration: int,
    firstiter_cc: bool,
    n_classes: int,
    log,
) -> RelionRuntimeControls:
    """Resolve consumed optimiser controls, CLI precedence and RELION defaults.

    ``target_iteration`` is the first numbered iteration of this invocation.
    See ``docs/math/relion_refinement_algorithm.md#command-admission``.
    """
    from relax.relion import relion_metadata

    expected_accuracy_do_ctf_correction = None
    relion_firstiter_ini_high_angstrom = None
    relion_optimiser_metadata = None
    if optimiser_star is not None:
        from relax.relion.relion_metadata import read_relion_optimiser_metadata

        relion_optimiser_metadata = read_relion_optimiser_metadata(optimiser_star)
        expected_accuracy_do_ctf_correction = relion_optimiser_metadata.get("do_correct_ctf")
        if expected_accuracy_do_ctf_correction is not None:
            expected_accuracy_do_ctf_correction = bool(expected_accuracy_do_ctf_correction)
            log.info(
                "RELION expected-accuracy CTF correction: %s (from %s)",
                expected_accuracy_do_ctf_correction,
                optimiser_star,
            )
        optimiser_text = Path(optimiser_star).read_text(errors="ignore")
        relion_firstiter_ini_high_angstrom = relion_metadata._parse_relion_cli_ini_high(optimiser_text)
        if firstiter_cc:
            if relion_firstiter_ini_high_angstrom is None:
                log.info(
                    "RELION firstiter_cc: no positive --ini_high found in %s",
                    optimiser_star,
                )
            else:
                log.info(
                    "RELION firstiter_cc: using --ini_high %.2f A from %s for post-iter1 low-pass",
                    float(relion_firstiter_ini_high_angstrom),
                    optimiser_star,
                )
    max_significants_resolution = None
    if optimiser_star is not None and relion_optimiser_metadata is not None:
        from relax.relion.relion_metadata import resolve_relion_runtime_max_significants

        optimiser_max_significants = relion_optimiser_metadata.get(
            "maximum_significants_arg"
        )
        if optimiser_max_significants is None:
            optimiser_max_significants = relion_metadata._load_relion_max_significants(optimiser_star)
            relion_optimiser_metadata = dict(relion_optimiser_metadata)
            relion_optimiser_metadata["maximum_significants_arg"] = (
                optimiser_max_significants
            )
        max_significants_resolution = resolve_relion_runtime_max_significants(
            override=max_significants,
            optimiser_metadata=relion_optimiser_metadata,
            target_iteration=target_iteration,
            do_firstiter_cc=firstiter_cc,
            n_classes=n_classes,
            reference_dimension=3,
        )
        max_significants = int(
            max_significants_resolution["active_max_significants"]
        )
        log.info(
            "RELION max_significants: saved_arg=%s active=%d source=%s "
            "do_grad=%s (from %s)",
            max_significants_resolution["maximum_significants_argument"],
            max_significants,
            max_significants_resolution["source"],
            max_significants_resolution["do_grad"],
            optimiser_star,
        )
    if max_significants is None:
        from relax.relion.relion_metadata import resolve_relion_runtime_max_significants

        # Without a RELION optimiser STAR, use relion_refine's own --maxsig default of -1
        # (ml_optimiser.cpp:1109) and its runtime resolution (ml_optimiser.cpp:3692-3699).
        max_significants_resolution = resolve_relion_runtime_max_significants(
            override=None,
            optimiser_metadata={"maximum_significants_arg": -1},
            target_iteration=target_iteration,
            do_firstiter_cc=firstiter_cc,
            n_classes=n_classes,
            reference_dimension=3,
        )
        max_significants_resolution["source"] = "relion_cli_default"
        max_significants = int(max_significants_resolution["active_max_significants"])
        log.info("RELION max_significants: --maxsig default -1 -> active %d", max_significants)
    elif max_significants_resolution is None:
        max_significants_resolution = {
            "maximum_significants_argument": None,
            "active_max_significants": int(max_significants),
            "source": "cli_override",
            "gradient_refine": False,
            "do_grad": False,
            "target_iteration": target_iteration,
        }

    return RelionRuntimeControls(
        expected_accuracy_do_ctf_correction,
        relion_firstiter_ini_high_angstrom,
        max_significants_resolution,
    )
```

### Verification and unfinished work

The first focused run passed 208 cases and encountered 12 new fixture setup
errors: the new manifest fixture used unsorted artifact names, which the existing
schema correctly refuses. Sorting that fixture repaired all 26 new admission
cases. The production source is identical between those runs; no existing
assertion, tolerance or baseline was weakened. The initial 194 existing cases
passed, including follower schedules, replay/command overrides, CLI defaults,
import ownership and archive metadata. Both runs had zero skips.

The new cases read real NPZ schedules, portable manifests and optimiser STAR
metadata. They cover deduplicated and relocated roots, altered manifests,
unmanifested numbered/final sampling, out-of-root optimiser files, changed
particle order/group labels, preserved exception scope, default/CLI/saved cap
precedence, gradient-mode active caps and legacy saved fields. A caller check
executes the actual command AST and verifies follower operand identity and
resolved option installation. Exact commands, XML, saved pre-edit source and
source manifests are in the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_command_admission_20261002T201344Z/HANDOFF.json).

Structural comparison resolves only proven input bindings and the moved empty
host-path list. Dispatch operations, runtime scalar computations and logging are
unchanged. All existing functions/types in both production files and controller
operations outside these two boundaries are unchanged. CLI scalar installation
now follows successful complete resolution; this is a configuration boundary,
not a change to numerical model-update timing. The existing CTF/filter/cap
values and output provenance are retained. This evidence does not establish
production quality or speed.

Command `main()` now spans 1,860 lines (previously 1,992); the numerical controller
still spans 2,949. The half worker still captures 38 names and the score operation
has 23 inputs. These are remaining ownership concerns, not acceptance criteria
based on counts. Full production K1/exactly-K4 quality, real-data quality, peak
memory and matched-GPU performance require the finished, reviewed, newly frozen
candidate. No new GPU job, merge or publication; previous sources/jobs remain
preserved. Class3D scratch lifetime still awaits the prior user decision.

<a id="correction-reporting-and-completed-iteration-capture"></a>

## Earlier correction reporting and completed-iteration capture

Correction reporting now has one small per-iteration result in its existing
normalization owner, reused by checkpoint averages and parity captures. Its
four fields replace four parallel reporting variables, with the same absent
per-half entries. It contains borrowed measurements only. The native numerical
update result stays separate because follower reports use rank 1's serialized
group scales while installed particle scales follow each particle's owner.

The existing diagnostic owner adapts the completed iteration to its capture
schema: scalar/array conversions, pose/particle list construction and the
existing warning policy now live in `dump_numbered_iteration`. The controller
retains full versus timing-only admission and the checkpoint-before-capture
sequence. A schema or capture-format change no longer requires inspecting
reconstruction, noise estimation or convergence internals. The normalization
admission and explicit correction installation remain visible below.

This adds one result type and one substantial diagnostic adaptation function
in existing modules, with no forwarding method, new module, flag or kernel.
The 18-input capture operation still exposes its dependencies. Existing model
fields are passed individually when only one is required; reconstruction
settings supply three immutable run fields. This is not a context containing
all controller locals.

### Producer, consumers and lifetime

| Operand | Producer and updates | Consumers and retention |
| --- | --- | --- |
| Native correction result | Existing numerical normalization producer | Visible native assignments or follower installation, then logging. Its original retained result and group-ID lifetimes are unchanged. |
| `NormScaleCorrectionReport` | New per-iteration report with absent entries; visible field assignments borrow native measurements and the selected native/rank-1 group report | Checkpoint average and completed-iteration capture. Default construction uses four Python lists; no numerical operation occurred between the old separate resets. |
| Pose/particle operands | Existing computed pose update and persistent halves | Capture keeps the original pose source and selected arrays, rather than regenerating metadata or reading a different pose frame. |
| Maps and average noise | Existing model fields | Passed as individual borrowed operands. No model object is retained by a new wrapper. |
| Fixed geometry/regularization | Existing reconstruction settings and raw dataset pixel size | Three settings fields are proven equal to the old locals. Pixel size retains its original scalar object/type; normalized geometry is not substituted. |

### Reporting result

[relax/relion/relion_normalization.py](../../relax/relion/relion_normalization.py) (line 33):

```python
@dataclass
class NormScaleCorrectionReport:
    """Per-half correction measurements for checkpoints and parity captures.

    Follower reporting uses rank 1's group scales with half 2 absent; installed
    particle corrections and follower state remain with their runtime owners.
    """

    group_scale_corrections_per_half: list = field(default_factory=lambda: [None, None])
    norm_corrections_per_half: list = field(default_factory=lambda: [None, None])
    avg_norm_correction_per_half: list = field(default_factory=lambda: [None, None])
    zero_norm_residual_counts: list = field(default_factory=lambda: [None, None])
```

### Complete noise update, correction policy and visible installation

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 2719):

```python
# RELION-style posterior-weighted noise update. Helper folds the
# K-class (shared) / K=1 (per-half) / firstiter_cc-skip variants;
# returns updated radial sigma2_noise + the unrolled
# ``noise_variance`` representation consumed by the engine.
noise_debug_dump = partial(
    _maybe_dump_noise_update_debug,
    iteration=iteration,
    current_size=current_size,
    image_shape=cryo.image_shape,
)
noise_update = update_posterior_noise_variance(
    noise_stats_per_half,
    noise_model,
    cryo.image_shape,
    k_class_enabled=k_class_enabled,
    firstiter_cc=relion_firstiter_cc_this_iter,
    dump_debug=noise_debug_dump,
)
noise_from_res = noise_update.noise_from_res
noise_from_res_per_half = noise_update.noise_from_res_per_half
noise_model = noise_update.model
if not relion_firstiter_cc_this_iter:
    _parity_dump.mark_stage(iteration, "noise_update")

correction_report = NormScaleCorrectionReport()
can_update_norm_scale = (
    noise_stats_per_half is not None
    and all(
        stats_k is not None
        and (
            getattr(stats_k, "wsum_norm_correction", None) is not None
            or int(experiment_datasets[_half_idx].n_units) == 0
        )
        for _half_idx, stats_k in enumerate(noise_stats_per_half)
    )
)
if follower_setup.follower_scale_state is not None and not can_update_norm_scale:
    raise RuntimeError(
        "Strict RELION follower-scale topology requires per-half norm/scale "
        "statistics at every numbered M-step"
    )
if can_update_norm_scale:
    group_ids_per_half = [
        np.zeros(int(experiment_datasets[_half_idx].n_units), dtype=np.int64)
        if group_ids_k is None
        else group_ids_k
        for _half_idx, group_ids_k in enumerate([particle_half.group_ids for particle_half in halves])
    ]
    norm_scale_update = prepare_norm_scale_update(
        noise_stats_per_half,
        halves,
        group_ids_per_half=group_ids_per_half,
        firstiter_cc=relion_firstiter_cc_this_iter,
        do_norm_correction=not tomo_halves,
        do_scale_correction=follower_setup.follower_scale_state is None,
        dtype=scoring_dtype,
        iteration=iteration,
        current_size=current_size,
    )
    if follower_setup.follower_scale_state is None:
        for half, images, scales in zip(
            halves,
            norm_scale_update.image_corrections_per_half,
            norm_scale_update.scale_corrections_per_half,
            strict=True,
        ):
            half.image_corrections = images
            half.scale_corrections = scales
        correction_report.group_scale_corrections_per_half = norm_scale_update.group_scale_corrections_per_half
    else:
        correction_report.group_scale_corrections_per_half = _update_relion_follower_corrections(
            follower_setup,
            noise_stats_per_half=noise_stats_per_half,
            norm_scale_update=norm_scale_update,
            relion_half_inputs=halves,
            relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
            dtype=scoring_dtype,
            logger=logger,
        )
    correction_report.norm_corrections_per_half = norm_scale_update.norm_corrections_per_half
    correction_report.avg_norm_correction_per_half = norm_scale_update.avg_norm_correction_per_half
    correction_report.zero_norm_residual_counts = norm_scale_update.zero_norm_residual_counts
    log_norm_scale_update(norm_scale_update, log=logger)
if follower_setup.follower_scale_state is not None:
    history.relion_scale_follower_scales_numbered_post_mstep_trajectory.append(
        np.asarray(follower_setup.follower_scale_state.scales, dtype=np.float64).copy()
    )

# Save per-iter per-shell sigma2 (after this iter's noise update) and
# the exact shell-wise tau2 ingredients used in the Wiener update.
history.record_noise_and_tau2(
    noise_from_res,
    noise_from_res_per_half,
    tau2_update_details,
    k_class_enabled=k_class_enabled,
)
```

### Complete checkpoint and full/timing-only capture boundary

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 2893):

```python
# --- RELION's run_itNNN files (ml_optimiser.cpp:3489) ---
checkpoint_writer = options.checkpoint.writer
if checkpoint_writer is not None and checkpoint_writer.due(numbered_relion_iteration):
    # The files hold incr_size/has_high_fsc_at_limit after this iteration's FSC
    # update, which the loop applies (idempotently) at the top of the next one.
    incr_size_after, high_fsc_after = relion_incr_size, relion_has_high_fsc_at_limit
    if not k_class_enabled:
        incr_size_after, high_fsc_after = update_relion_growth_state_from_fsc(
            _truncate_fsc_for_current_size_growth(
                tau2_fsc_for_update,
                current_size=current_size,
                grid_size=grid_size,
                dtype=scoring_dtype,
            ),
            current_size,
            incr_size=relion_incr_size,
            has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
        )
    snapshot = snapshot_capture.begin(
        numbered_relion_iteration,
        state,
        sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
        current_size=current_size,
        incr_size=incr_size_after,
        has_high_fsc_at_limit=high_fsc_after,
        random_perturbation=random_perturbation,
        acc_rot_per_class=model_acc_rot_per_class,
        acc_trans_per_class_angstrom=model_acc_trans_per_class,
    )
    snapshot = snapshot_capture.finish(
        snapshot,
        reference_model.maps,
        unreg_means,
        (
            mean_signal_variance_shells
            if k_class_enabled
            else [details["prior_shells"] for details in tau2_update_details_per_half]
        ),
        previous_data_vs_prior_for_scheduling,
        noise_model.radial_per_half,
        fsc=fsc,
        fsc_for_growth=None if k_class_enabled else tau2_fsc_for_update,
        class_weights=class_weights if k_class_enabled else None,
        direction_priors=direction_priors,
        half_inputs=halves,
        class_assignments=class_assignments if k_class_enabled else None,
        max_posterior=max_posterior_per_half,
        significant_counts=significance.per_half,
        avg_norm_correction=correction_report.avg_norm_correction_per_half,
    )
    checkpoint_writer(snapshot)

if _parity_dump.is_active():
    dump_numbered_iteration(
        iteration,
        init_relion_iteration=init_relion_iteration,
        state=state,
        current_size=current_size,
        sigma_offset_angstrom=current_sigma_offset_angstrom,
        random_perturbation=random_perturbation,
        settings=reconstruction_settings,
        pixel_size_angstrom=cryo.voxel_size,
        ave_pmax=ave_pmax,
        fsc=fsc,
        noise_variance=noise_model.average_variance,
        means=reference_model.maps,
        unfiltered_means=unreg_means,
        poses=pose_update.current,
        half_inputs=halves,
        corrections=correction_report,
        scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
        log=logger,
    )
elif _parity_dump.timing_is_active():
    try:
        _parity_dump.dump_timing_iteration(
            iteration=iteration,
            init_relion_iteration=int(init_relion_iteration),
            iteration_start=t0,
        )
    except Exception as exc:
        logger.warning("parity_dump.dump_timing_iteration failed at iter %d: %s", iteration, exc)
```

### Complete capture adaptation

[relax/diagnostics/iteration.py](../../relax/diagnostics/iteration.py) (line 45):

```python
def dump_numbered_iteration(
    iteration: int,
    *,
    init_relion_iteration: int,
    state: RefinementState,
    current_size: int,
    sigma_offset_angstrom,
    random_perturbation,
    settings: ReconstructionSettings,
    pixel_size_angstrom,
    ave_pmax,
    fsc,
    noise_variance,
    means,
    unfiltered_means,
    poses: tuple[ParticlePoses, ParticlePoses],
    half_inputs: tuple[HalfSet, HalfSet],
    corrections: NormScaleCorrectionReport,
    scale_correction_data_vs_prior,
    log,
) -> None:
    """Adapt completed-iteration operands to the parity capture schema.

    Capture admission and timing-only selection remain with the controller.
    See ``docs/math/relion_refinement_algorithm.md#diagnostic-capture``.
    """
    try:
        _parity_dump.dump_iteration(
            iteration=iteration,
            init_relion_iteration=int(init_relion_iteration),
            current_size=int(current_size),
            sigma_offset=float(sigma_offset_angstrom),
            translation_step=float(state.translation_step),
            translation_range=float(state.translation_range),
            random_perturbation=float(random_perturbation) if random_perturbation is not None else 0.0,
            random_perturbation_instance=int(state.perturbation_instance)
            if hasattr(state, "perturbation_instance")
            else 0,
            tau2_fudge=float(settings.tau2_fudge),
            voxel_size=pixel_size_angstrom,
            grid_size=int(settings.grid_size),
            volume_shape=tuple(settings.volume_shape),
            ave_pmax=float(ave_pmax),
            fsc=np.asarray(fsc, dtype=np.float64),
            sigma2_noise=np.asarray(noise_variance, dtype=np.float64),
            means=means,
            unreg_means=unfiltered_means,
            new_iter_best_rotation_eulers=[pose.eulers_deg for pose in poses],
            new_iter_best_translations=[pose.translations_pixels for pose in poses],
            image_corrections=[half.image_corrections for half in half_inputs],
            scale_corrections=[half.scale_corrections for half in half_inputs],
            group_ids=[half.group_ids for half in half_inputs],
            group_counts=[half.group_count for half in half_inputs],
            group_scale_corrections=corrections.group_scale_corrections_per_half,
            norm_corrections=corrections.norm_corrections_per_half,
            avg_norm_corrections=corrections.avg_norm_correction_per_half,
            zero_norm_residual_counts=corrections.zero_norm_residual_counts,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
        )
    except Exception as exc:
        log.warning("parity_dump.dump_iteration failed at iter %d: %s", iteration, exc)
```

### Verification and unfinished work

The initial focused run passed 88 of 93 cases. Five failures belonged to two
new fixture mistakes: the actual controller uses a while loop, and the file
writer emits per-half fields only when E-step snapshots exist. The fixtures now
select the actual loop and populate snapshots through the production capture API;
no numerical assertion or tolerance changed. All 13 new reporting/capture cases
then pass. All 25 affected controller cases pass (237 deselected), with zero
skips. Existing normalization, follower, checkpoint-lifetime and timing tests
passed in the initial run. See the
[package receipt](final_search_patch_status.md#current-correction-reporting-and-capture).

The structural proof inlines the actual capture operation, expands the actual
report defaults and resolves only proven-equal immutable settings reads. It
matches the original controller AST after bound comprehension variables are
renamed. Numerical operations, casts, mode conditions, state writes, exception
handlers and timing calls are not removed from the comparison. All pre-existing
normalization and diagnostic functions/types are unchanged. This is structural
and focused CPU evidence, not scientific or performance qualification.

The numerical controller still spans 2,949 lines; command `main()` remains
1,992. The score operation still has 23 inputs and its half worker captures 38
names. The representative design remains unfinished. Class3D scratch lifetime
still awaits the earlier user choice. Full production K1/exactly-K4, real-data
quality, peak-memory and matched-GPU performance remain unqualified until the
complete flow is finished, reviewed and frozen. No new GPU job, merge or
publication; previous sources, controls and jobs are preserved.

## Numbered half input ownership

The next pass traces three existing inputs to their producers before changing
any interface. The radial table now travels with the half's model operands;
coarse angular metadata travels with sampling; particle diameter is read from
the immutable run schedule already required by the operation. No new class,
function, module, flag, forwarding layer or numerical kernel was introduced.

| Input | Producer, owner and lifetime | Consumers and original constraints |
| --- | --- | --- |
| Radial noise | Persistent `NoiseModel.radial_per_half`; `HalfScoringData` borrows that same reference-grid host table | Local/dense optics preparation adapts it to image shape classes. The controller retains its tomography/empty-half guard; no array copy or cast. |
| Coarse angular step | Existing local sampling, or the adaptive image-size plan using the incoming sizing order; stored in the local/dense sampling result | Per-shape batch planning and optics sizing use the same scalar or None. No sampling order is inferred from a grid or recomputed. |
| Particle diameter | Immutable `options.schedule.particle_diameter_ang`, previously assigned verbatim to a controller local | The same batch/optics calculations; None retains the same box-size policy. No normalization or duplicate override. |

A maintainer changing per-half noise adaptation can now inspect half binding and
optics preparation without tracing an independent raw-noise argument through
the worker. Sampling size metadata stays with the grids it describes, including
local and dense routes. The operation's 23 inputs still need review; this is not
permission to package the remaining dependencies into a controller-local context.
The half worker still captures 38 names. Scoring, frame installation, offloading,
payload publication and recording remain explicit, followed by unchanged paired
execution and preprocessing drain. Scientific mode decisions and model updates
remain in orchestration.

### Existing operand and sampling owners

[relax/refinement/half_scoring.py](../../relax/refinement/half_scoring.py) (line 188):

```python
@dataclass(frozen=True, kw_only=True)
class HalfScoringData:
    """Persistent particle half and the model operands for one expectation."""

    particles: HalfSet
    reference: object
    noise_variance: object
    noise_radial: object | None = None
    mean_variance: object | None = None
    projector: PreparedProjector | None = None
    scale_group_ids: object | None = None
    scale_group_count: int | None = None
    scale_correction_data_vs_prior: object | None = None
    image_seed_classes: object | None = None
```

[relax/refinement/half_scoring.py](../../relax/refinement/half_scoring.py) (line 204):

```python
@dataclass(frozen=True, kw_only=True)
class DenseSamplingSpec:
    """Resolved dense sampling grids, oversampling and reconstruction sizes."""

    effective_rotations: object
    current_translations: object
    base_translations: object
    current_healpix_order: int
    oversampling_order: int
    translation_step: float
    random_perturbation: float
    cs_for_engine: int | None
    coarse_engine: str = "auto"
    model_current_size_for_engine: int | None = None
    coarse_angular_step_deg: float | None = None
    coarse_rotation_ids: object | None = None
    coarse_scoring_rotations: object | None = None
    symmetry: str = "C1"
```

### Complete current phase construction, caller and drain

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 1774):

```python
numbered_grid = sampling.TrialGrid(
    rotations=effective_rotations,
    rotation_eulers=effective_rotation_eulers,
    mstep_rotations=effective_mstep_rotations,
    translations=current_translations,
)
numbered_variant = DenseVariantPolicy(
    firstiter_score_mode_this_iter=firstiter_score_mode_this_iter,
    firstiter_winner_take_all_this_iter=firstiter_winner_take_all_this_iter,
    k_class_enabled=k_class_enabled,
    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
    firstiter_coarse_current_size=coarse_cs if use_adaptive else None,
    firstiter_fine_current_size=cs_for_engine if use_adaptive else None,
    firstiter_log_label="" if use_adaptive else "(non-adaptive site) ",
    firstiter_updates_em_kwargs_ibs=bool(use_adaptive),
)
if use_local:
    numbered_sampling = local_sampling
    numbered_local_diagnostics = LocalDiagnosticPolicy(
        iteration=iteration,
        debug_iteration=numbered_relion_iteration,
        save_intermediates_dir=debug.save_intermediates_dir,
        collect_local_search_profile=collect_local_search_profile,
        diagnostic_score_only=bool(debug.stop_after_local_search_score_only),
        local_profile_history=history.local_profile_history,
        adaptive_pass2_full_parent=local_adaptive_full_parent,
        adaptive_pass2_rotation_only=local_adaptive_rotation_only,
        adaptive_pass2_denominator_mode=local_adaptive_denominator_mode,
    )
else:
    numbered_sampling = DenseSamplingSpec(
        effective_rotations=(
            adaptive_pass1_rotations
            if use_adaptive and adaptive_pass1_rotations is not None
            else effective_rotations
        ),
        current_translations=current_translations,
        base_translations=base_translations,
        current_healpix_order=current_rotation_grid.healpix_order,
        oversampling_order=state.adaptive_oversampling,
        translation_step=state.translation_step,
        coarse_engine=adaptive.coarse_engine,
        random_perturbation=random_perturbation,
        cs_for_engine=cs_for_engine,
        model_current_size_for_engine=model_current_size_for_engine,
        coarse_rotation_ids=coarse_rotation_ids_for_scoring,
        coarse_scoring_rotations=adaptive_pass1_rotations if int(state.adaptive_oversampling) == 0 else None,
        coarse_angular_step_deg=coarse_size_step_deg,
        symmetry=symmetry,
    )
    numbered_local_diagnostics = None
if tomo_halves:
    tomo_oversampling = int(state.adaptive_oversampling)
    tomo_coarse_size = local_sampling.coarse_image_window_size if use_local else coarse_cs
    numbered_tomo_sampling = TomoSampling(
        healpix_order=int(local_sampling.search.healpix_order) - tomo_oversampling if use_local else int(current_rotation_grid.healpix_order),
        oversampling_order=tomo_oversampling,
        offset_range_angst=float(state.translation_range) * image_geometry.pixel_size_angstrom,
        offset_step_angst=float(state.translation_step) * image_geometry.pixel_size_angstrom,
        random_perturbation=float(local_sampling.perturbation if use_local else random_perturbation),
        coarse_size=int(cryo.image_shape[0] if tomo_coarse_size is None else tomo_coarse_size),
        fine_size=int(cryo.image_shape[0] if cs_for_engine is None else cs_for_engine),
    )
else:
    numbered_tomo_sampling = None

def _run_half_estep(k):
    particle_half = halves[k]
    score_result = score_numbered_half(
        HalfScoringData(
            particles=particle_half,
            reference=reference_model.maps[k],
            mean_variance=reference_model.tau2_per_half[k],
            noise_variance=noise_model.variance_per_half[k],
            noise_radial=noise_model.radial_per_half[k] if not tomo_halves and particle_half.dataset.n_units else None,
            projector=projectors[k],
            scale_group_ids=follower_setup.scale_stats_group_ids_per_half[k],
            scale_group_count=follower_setup.scale_stats_group_count_per_half[k],
            scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
        ),
        numbered_grid,
        sampling=numbered_sampling,
        tomo_sampling=numbered_tomo_sampling,
        direction_priors=direction_log_priors[k],
        class_log_priors=class_log_priors,
        sigma_offset_angstrom=_sigma_offset_for_half(
            current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half, k,
        ),
        batch_planner=batch_planner,
        image_geometry=image_geometry,
        padded_volume_shape=padded_volume_shape,
        use_adaptive=use_adaptive,
        multi_shape_halves=multi_shape_halves,
        variant=numbered_variant,
        options=options,
        local_diagnostics=numbered_local_diagnostics,
        replay_prior_translations=_replay_prior_translations,
        initial_class_assignments=k_class.first_iteration_seed_classes if seed_iteration else None,
        single_class_iteration=single_class_iteration,
        scoring_dtype=scoring_dtype,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        relion_translation_angle_scale=relion_translation_angle_scale,
        iteration=iteration,
        numbered_relion_iteration=numbered_relion_iteration,
    )
    per_half.translation_search_bases[k] = score_result.translation_search_base
    per_half.pose_rotations[k] = score_result.pose_rotations
    per_half.pose_rotation_eulers[k] = score_result.pose_rotation_eulers
    per_half.coarse_ha[k] = score_result.coarse_ha
    if particle_half.dataset.n_units != 0:
        score_result = _maybe_host_offload_half0_local_accumulators(
            half_index=k,
            use_local=use_local,
            k_class_enabled=k_class_enabled,
            score_result=score_result,
            log=logger,
        )
    per_half.update_from(k, score_result, dtype=scoring_dtype)
    record_numbered_half(
        score_result,
        particle_half,
        per_half,
        significance,
        profile_history=history.global_profile_history,
        iteration=iteration,
        image_window_size=cs_for_engine,
        healpix_order=current_rotation_grid.healpix_order,
        k_class_enabled=k_class_enabled,
    )

_overlap_active = _half_overlap_active(
    options.overlap.overlap_halves,
    diagnostic_half_indices=diagnostic_half_indices,
    log=logger,
)
if _overlap_active:
    _run_halves_overlapped(_run_half_estep, diagnostic_half_indices)
    k = diagnostic_half_indices[-1]
else:
    for k in diagnostic_half_indices:
        _run_half_estep(k)
if diagnostic_half_indices != (0, 1):
    raise RuntimeError(
        "targeted half-only significance diagnostic returned without writing its "
        "complete target set; refusing to continue with one half missing"
    )

Ft_y_0, Ft_y_1 = per_half.Ft_y
Ft_ctf_0, Ft_ctf_1 = per_half.Ft_ctf

# E-step + per-half M-step accumulators are now both populated.
_parity_dump.mark_stage(iteration, "e_step")
from relax.cuda.kernels import drain_relion_preprocess_checks
drain_relion_preprocess_checks()
significance.combine()
```

### Complete expectation operation

[relax/refinement/expectation.py](../../relax/refinement/expectation.py) (line 320):

```python
def score_numbered_half(
    half: HalfScoringData,
    grid: TrialGrid,
    *,
    sampling: LocalSampling | DenseSamplingSpec,
    tomo_sampling: TomoSampling | None,
    direction_priors: HalfDirectionLogPriors,
    class_log_priors,
    sigma_offset_angstrom,
    batch_planner: BatchPlanner,
    image_geometry: ImageGeometry,
    padded_volume_shape,
    use_adaptive: bool,
    multi_shape_halves: bool,
    variant: DenseVariantPolicy,
    options: RefinementOptions,
    local_diagnostics: LocalDiagnosticPolicy | None,
    replay_prior_translations,
    initial_class_assignments,
    single_class_iteration: bool,
    scoring_dtype,
    source_faithful_spectrum_norm: bool,
    relion_translation_angle_scale: float,
    iteration: int,
    numbered_relion_iteration: int,
) -> HalfScoreResult:
    """Build half-specific priors/batches and accumulate an empty, SPA or tomography half.

    Canonical pose grids and the applied translation base accompany the result.
    The controller owns publication, accumulator offloading and post-score capture.
    """
    particle_half = half.particles
    k = particle_half.index
    use_local = isinstance(sampling, LocalSampling)
    tomo_halves = tomo_sampling is not None
    k_class_enabled = variant.k_class_enabled
    image_window_size = sampling.image_window_size if use_local else sampling.cs_for_engine
    model_support_size = sampling.model_support_size if use_local else sampling.model_current_size_for_engine
    symmetry = sampling.search.symmetry if use_local else sampling.symmetry
    coarse_size_step_deg = sampling.coarse_angular_step_deg
    particle_diameter_ang = options.schedule.particle_diameter_ang
    bpref_diagnostics.set_bpref_contribution_dump_context(
        iteration=iteration + 1,
        half=particle_half.index + 1,
    )
    bpref_device_signature_active = (
        _bpref_device_signature_active_for_numbered_half(
            iteration=iteration + 1,
            half=particle_half.index + 1,
        )
    )
    logger.info(
        "BPREF_DEVICE_SIGNATURE_ACTIVATION iteration=%d half=%d "
        "final_all_data=false active=%s",
        iteration + 1,
        particle_half.index + 1,
        str(bpref_device_signature_active).lower(),
    )
    previous_translations_k = particle_half.translations
    translation_search_base = relion_translation_search_base(
        previous_translations_k, dtype=scoring_dtype
    )
    half_batching = prepare_half_batches(
        particle_half.dataset,
        half.projector,
        planner=batch_planner,
        rotations=grid.rotations if use_adaptive or k_class_enabled else None,
        translations=grid.translations,
        cs_for_engine=image_window_size,
        coarse_cs=variant.firstiter_coarse_current_size if use_adaptive else None,
        model_current_size_for_engine=model_support_size,
        use_adaptive=use_adaptive,
        use_local=use_local,
        relion_firstiter_cc_this_iter=variant.relion_firstiter_cc_this_iter,
        firstiter_winner_take_all_this_iter=variant.firstiter_winner_take_all_this_iter,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
        image_fourier_backend=options.parity.image_fourier_backend,
        bpref_device_signature_active=bpref_device_signature_active,
        multi_shape_halves=multi_shape_halves,
        coarse_sizing=(coarse_size_step_deg, particle_diameter_ang) if use_adaptive and multi_shape_halves else None,
    )
    # RELION translation priors: relion_half_translation_prior_inputs
    # documents the pdf_offset / wsum_sigma2_offset centers, the
    # cold-start engine center and the prior-grid selection.
    if tomo_halves:
        # A subtomogram half builds its 3D offset priors per particle (score_tomo_half).
        translation_prior_inputs = None
        trans_prior_center = local_trans_prior_center = trans_prior_center_for_engine = None
    else:
        translation_prior_inputs = relion_half_translation_prior_inputs(
            previous_translations_k,
            voxel_size=image_geometry.pixel_size_angstrom,
            base_translations=sampling.base_translations,
            current_translations=grid.translations,
            dtype=scoring_dtype,
        )
        trans_prior_center = translation_prior_inputs.prior_center
        local_trans_prior_center = translation_prior_inputs.local_prior_center
        trans_prior_center_for_engine = translation_prior_inputs.engine_prior_center
    translation_log_prior = None
    if not use_local and not tomo_halves:
        if not k_class_enabled and trans_prior_center is None:
            # A fresh K1 half has implicit zero offsets, not a flat
            # pdf_offset. Native ACC applies the Gaussian even at
            # iteration 1; see relion_refinement_algorithm.md.
            trans_prior_center = np.zeros(2, dtype=scoring_dtype)
        translation_log_prior = make_relion_translation_log_prior(
            translation_prior_inputs.prior_translations,
            image_geometry.pixel_size_angstrom,
            sigma_offset_angstrom,
            trans_prior_center,
            offset_range_pixels=None,
            dtype=scoring_dtype,
        )
    if particle_half.dataset.n_units == 0:
        logger.info("Skipping E-step/M-step accumulation for empty half-%d dataset", particle_half.index + 1)
        n_shells = int(image_geometry.image_shape[0] // 2 + 1)
        n_rot_for_stats = int(
            rotation_grid_size(sampling.search.healpix_order, symmetry=symmetry) if use_local else grid.rotations.shape[0]
        )
        empty_k1_x_half_mstep = (
            (not k_class_enabled)
            and (use_local or use_adaptive)
            and _k1_relion_x_half_mstep_enabled()
        )
        empty_result = empty_half_result(
            volume_shape=particle_half.dataset.volume_shape if empty_k1_x_half_mstep else None,
            padded_volume_shape=padded_volume_shape,
            n_classes=batch_planner.n_classes,
            n_shells=n_shells,
            n_rotations=n_rot_for_stats,
            translation_dimension=3 if tomo_halves else grid.translations.shape[1],
            image_window_size=image_window_size,
            model_support_size=model_support_size,
            use_x_half_mstep=empty_k1_x_half_mstep,
        )
        empty_result.coarse_ha = empty_result.ha
        empty_result.translation_search_base = translation_search_base
        return empty_result
    # The half's units' classes in RELION's seed iteration, by particle row (all class 0 in the CC
    # iteration before it).
    seed_classes_k = (
        np.asarray(initial_class_assignments)[
            particle_half.dataset._index_layout.original_image_indices_for_local(
                np.arange(particle_half.dataset.n_units)
            )
        ]
        if initial_class_assignments is not None
        else np.zeros(particle_half.dataset.n_units, dtype=np.int64)
        if single_class_iteration
        else None
    )
    if tomo_halves:
        score_result = _score_tomo_half_in_loop(
            particle_half.dataset,
            use_local=use_local,
            use_adaptive=use_adaptive,
            volume=half.reference,
            noise_variance=half.noise_variance,
            relion_projector_half=None if half.projector is None else half.projector.data,
            relion_projector_r_max=None if half.projector is None else half.projector.r_max,
            sampling=tomo_sampling,
            local_search=(
                dict(
                    previous_eulers_deg=particle_half.rotation_eulers,
                    sigma_rot=sampling.search.sigma_rot,
                    sigma_psi=sampling.search.sigma_psi,
                )
                if use_local
                else None
            ),
            rotation_log_prior=direction_priors.rotation_log_prior,
            previous_translations=previous_translations_k,
            sigma_offset_angst=sigma_offset_angstrom,
            max_significants=options.adaptive.max_significants,
            unit_groups=particle_half.optics_group_ids,
            scale_corrections=particle_half.scale_corrections,
            group_ids=half.scale_group_ids,
            scale_correction_group_count=half.scale_group_count,
            scale_correction_data_vs_prior=half.scale_correction_data_vs_prior,
            reconstruction_current_size=model_support_size,
            symmetry=symmetry,
            class_log_priors=class_log_priors if k_class_enabled else None,
            class_rotation_log_prior=direction_priors.class_rotation_log_prior if k_class_enabled else None,
            unit_seed_classes=seed_classes_k,
            normalized_cc=variant.firstiter_score_mode_this_iter == "normalized_cc",
        )
    elif use_local:
        local_optics = optics_shapes.prepare_optics(
            particle_half.dataset,
            noise_radial=half.noise_radial,
            coarse_step_deg=coarse_size_step_deg,
            particle_diameter_ang=particle_diameter_ang,
            previous_translations=previous_translations_k,
            sigma_offset_angstrom=sigma_offset_angstrom,
            base_translations=sampling.base_translations,
            current_translations=grid.translations,
            with_log_prior=False,
            zero_cold_center=not k_class_enabled,
            dtype=scoring_dtype,
        )
        local_result = _score_half_local_in_bpref_scope(
            half=replace(half, mean_variance=None),
            sampling=sampling,
            priors=LocalPriorSpec(
                trans_prior_center=local_trans_prior_center,
                trans_prior_center_for_engine=trans_prior_center_for_engine,
                current_sigma_offset_angstrom=sigma_offset_angstrom,
                translation_search_base=translation_search_base,
                local_search_translation_prior_mode=(options.local_search.local_search_translation_prior_mode),
                replay_prior_translations=replay_prior_translations,
            ),
            batching=LocalBatchPolicy(
                max_significants=options.adaptive.max_significants,
                safe_batch_sizes=half_batching.safe_batch_sizes,
            ),
            execution=LocalExecutionPolicy(
                disc_type=options.disc_type,
                disable_adjoint_y=options.debug.disable_adjoint_y,
                disable_adjoint_ctf=options.debug.disable_adjoint_ctf,
                source_faithful_spectrum_norm=(source_faithful_spectrum_norm),
                relion_translation_angle_scale=(relion_translation_angle_scale),
            ),
            diagnostics=replace(
                local_diagnostics, bpref_device_signature_active=bpref_device_signature_active,
            ),
            optics=local_optics,
        )
        if local_result.coarse_ha is None:
            local_result.coarse_ha = local_result.ha
        score_result = local_result

    else:
        # Shared dense half-scoring operands; the adaptive branch adds its
        # pass-1 grid and batch/size overrides.
        dense_optics = optics_shapes.prepare_optics(
            particle_half.dataset,
            noise_radial=half.noise_radial,
            coarse_step_deg=coarse_size_step_deg,
            particle_diameter_ang=particle_diameter_ang,
            previous_translations=previous_translations_k,
            sigma_offset_angstrom=sigma_offset_angstrom,
            base_translations=sampling.base_translations,
            current_translations=grid.translations,
            with_log_prior=not use_local,
            zero_cold_center=not k_class_enabled,
            dtype=scoring_dtype,
        )
        dense_half = replace(half, image_seed_classes=seed_classes_k)
        dense_sampling = sampling
        dense_priors = DensePriorSpec(
            rotation_log_prior_k=direction_priors.rotation_log_prior,
            class_rotation_log_prior_k=direction_priors.class_rotation_log_prior,
            translation_log_prior=translation_log_prior,
            translation_search_base=translation_search_base,
            trans_prior_center_for_engine=trans_prior_center_for_engine,
            class_log_priors=class_log_priors,
        )
        dense_batching = DenseBatchPolicy(
            image_batch_size=options.batching.image_batch_size,
            max_significants=options.adaptive.max_significants,
            safe_batch_sizes=half_batching.safe_batch_sizes,
            significance_safe_batch_sizes=half_batching.significance_safe_batch_sizes,
            class_batch_overrides=half_batching.class_overrides,
            k_class_image_batch_size_override=(half_batching.fine_image_batch_size if use_adaptive else None),
            k_class_rotation_block_size_override=(half_batching.fine_rotation_block_size if use_adaptive else None),
            significance_image_batch_size_override=(half_batching.coarse_image_batch_size if use_adaptive else None),
            significance_rotation_block_size_override=(half_batching.coarse_rotation_block_size if use_adaptive else None),
        )
        dense_variant = variant
        dense_execution = DenseExecutionPolicy(
            disc_type=options.disc_type,
            disable_adjoint_y=options.debug.disable_adjoint_y,
            disable_adjoint_ctf=options.debug.disable_adjoint_ctf,
            bpref_device_signature_active=bpref_device_signature_active,
            debug_iteration=numbered_relion_iteration,
            diagnostic_float64_pass2=_diagnostic_float64_pass2_matches(
                numbered_relion_iteration
            ),
            preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_translation_angle_scale=relion_translation_angle_scale,
        )
        dense_result = _score_half_dense_in_bpref_scope(
            dense_half,
            dense_sampling,
            dense_priors,
            dense_batching,
            dense_variant,
            dense_execution,
            dense_optics,
        )
        if not use_adaptive or dense_result.pose_rotations is None:
            if dense_result.pose_rotations is None:
                dense_result.pose_rotations = grid.rotations
            if dense_result.pose_rotation_eulers is None:
                dense_result.pose_rotation_eulers = grid.rotation_eulers
        if dense_result.coarse_ha is None:
            dense_result.coarse_ha = dense_result.ha
        score_result = dense_result

        # --- Manifest dump for deterministic replay (Phase 0.1) ---
        if not use_adaptive and options.debug.save_intermediates_dir is not None:
            _manifest_path = os.path.join(
                options.debug.save_intermediates_dir,
                f"manifest_iter{iteration}_half{k}.npz",
            )
            _manifest = {
                "effective_rotations": np.asarray(grid.rotations),
                "coarse_scoring_rotations": _replay_manifest_array(
                    dense_sampling.coarse_scoring_rotations,
                ),
                "current_translations": np.asarray(grid.translations),
                "rotation_log_prior": _replay_manifest_array(direction_priors.rotation_log_prior, dtype=np.float64),
                "translation_log_prior": _replay_manifest_array(translation_log_prior, dtype=np.float64),
                "image_corrections": _replay_manifest_array(
                    particle_half.image_corrections, dtype=np.float64,
                ),
                "scale_corrections": _replay_manifest_array(
                    particle_half.scale_corrections, dtype=np.float64,
                ),
                "image_pre_shifts": _replay_manifest_array(translation_search_base, dtype=np.float32),
                "absolute_previous_translations": _replay_manifest_array(
                    previous_translations_k, dtype=np.float32,
                ),
                "mean_vol_ft": np.asarray(half.reference),
                "mean_variance": np.asarray(half.mean_variance),
                "noise_variance": np.asarray(half.noise_variance),
                "current_size": np.int32(image_window_size) if image_window_size is not None else np.int32(-1),
                "half_spectrum_scoring": np.bool_(True),
                "use_float64_scoring": np.bool_(scoring_policy.DENSE_PRECISION.use_float64_scoring),
                "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                "reconstruction_padding_factor": np.int32(RECONSTRUCTION_PADDING_FACTOR),
                "score_with_masked_images": np.bool_(True),
                "perturbation_instance": np.float64(sampling.random_perturbation),
                "perturbation_factor": np.float64(options.parity.perturb_factor),
                "iteration": np.int32(iteration),
                "half_index": np.int32(k),
                "ave_Pmax": np.float64(float(np.mean(dense_result.em_stats.max_posterior_per_image))),
            }
            np.savez(_manifest_path, **_manifest)
            logger.info("Manifest dumped: %s", _manifest_path)

    score_result.translation_search_base = translation_search_base
    return score_result
```

### Checks and remaining work

The first focused command ran 125 cases: 122 passed and three failed in a final
preparation fixture importing the numbered fixture's retired argument dictionary.
That indirect fixture now reads the half owner, preserving all its independent
numerical frame assertions. All three final preparation cases subsequently pass.
All 45 affected controller cases pass (217 deselected), with zero skips. Eight
new cases check scalar/table identity in batch/optics consumers and execute the
actual controller binding with unused radial metadata unavailable.

The structural proof resolves only the three proven-equal owner bindings and
confirms the original controller and score-operation AST. All scoring adapters
and all other expectation functions/types are unchanged. Production precision,
casts, reduction order, grid allocations and state updates are not normalized
out of the comparison. Source manifests distinguish the two checked commands
and include untracked files. See the
[package receipt](final_search_patch_status.md#earlier-half-input-ownership).

The controller still spans 2,968 lines and command `main()` 1,992. Its half worker
spans 63 lines, captures 38 names and calls a 23-input operation; phase ownership
and the complete representative design are unfinished. Class3D scratch lifetime
still awaits the earlier user choice. Full production K1/exactly-K4, real-data
quality, memory and matched-GPU performance remain unqualified. Earlier frozen
sources, jobs and their receipts are preserved for their own source.

## Half-expectation recording and support counts

This section records the preceding recording package and its checked source,
before the input-ownership changes shown above. Its counters and recorder remain
unchanged; use the latest caller above for the current score interface.

The existing expectation owner now separates half scoring from recording its
profile, support counts and E-step captures. The controller still constructs
phase/mode settings, scores the half, records its sampling frame, offloads eligible
half-0 local accumulators and publishes the payload. It invokes recording last
inside the same half worker. Its serial/overlap dispatch and preprocessing drain
remain at the same points before support-count combination and reconstruction.

A capture or support-count change can be reviewed in the recorder and its count
owner without opening translation-prior construction, adaptive engine selection,
projector preparation, reconstruction, noise or convergence implementations.
The actual controller below shows when these operations run and which input
owners they consume. The phase interface is not yet the target: the half worker
still captures 40 names, down from 42. Moving recording does not resolve those
remaining phase dependencies.

### Inputs, consumers and lifetime

| Input or result | Producer and lifetime | Consumers and updates |
| --- | --- | --- |
| `HalfScoreResult` | Existing numbered half expectation | Controller publishes metadata/payload and performs existing offloading; recorder reads the same retained arrays |
| `PerHalfOutputs` | Fresh phase slots; existing class, pose and accumulator metadata | Explicit controller writes precede captures; reconstruction and pose updates consume them later |
| `SignificanceStatistics` | Fresh per-expectation collection | Recorder casts supplied counts to int32 and fills per-half slots and the two existing parts lists; combination happens after preprocessing drain |
| Recorded versus convergence counts | Combined once, in the original two-list append/concatenation order | History/profile output uses recorded counts; native convergence uses its separate aggregate; Class3D leaves it absent |
| Per-half counts | Original converted arrays, retained by the same slots and parts lists | Particle-state capture and checkpoint finishing read them at their original points |
| Profile/capture metadata | Half identity, current engine window and base grid order | Recorder preserves the existing profile numbering and capture schema, including the empty-half route |

Absent counts and supplied empty counts are distinct. Empty particle halves keep
the zero E-step capture but do not record a profile, flush a device panel or
contribute support counts. Native and Class3D support policies remain distinct.
The two parts lists are not replaced by sorting half slots: overlap can change
arrival order, and the original recording/convergence append sequences remain.

The existing particle-state index helper falls back to identity indices when the
layout is absent. E-step capture instead retains its existing None-on-error
semantics. This package does not substitute one metadata contract for the other.

### Actual construction, scoring, publication and dispatch

This extract follows prior/window planning. It includes the mode decision,
output-slot construction, adaptive window/projector preparation, dense/local/tomo
settings, half worker and dispatch. Reconstruction starts after the preprocessing
drain and support combination. Offloading and output installation stay visible.

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 1622):

```python
# --- Run E+M on each half-set ---
# Two modes: single-pass (adaptive_oversampling=0) or two-pass
# coarse/fine (adaptive_oversampling>=1).
significance = SignificanceStatistics()
use_adaptive = _should_use_adaptive_search(
    adaptive_oversampling=state.adaptive_oversampling,
    use_local=use_local,
    n_rotations=effective_rotations.shape[0],
    symmetry=symmetry,
)
# Track the rotation grids used for pose extraction.
# When adaptive oversampling is active, ha_k indices refer to the
# oversampled grid (from pass 2), not effective_rotations.
per_half = PerHalfOutputs()
hard_assignments = per_half.hard_assignments
class_assignments = per_half.class_assignments
class_posterior_per_half = per_half.class_posterior
class_full_posterior_per_half = per_half.class_full_posterior
max_posterior_per_half = per_half.max_posterior
rotation_posterior_per_half = per_half.rotation_posterior
class_rotation_posterior_per_half = per_half.class_rotation_posterior
pose_rotations = per_half.pose_rotations  # rotations to use with ha for poses
pose_rotation_eulers = per_half.pose_rotation_eulers
best_pose_rotations = per_half.best_pose_rotations
best_pose_rotation_eulers = per_half.best_pose_rotation_eulers
best_pose_translations = per_half.best_pose_translations
translation_search_bases = per_half.translation_search_bases
# Coarse-grid assignments for local search tracking (always indexed
# into effective_rotations, even when adaptive oversampling is used).
coarse_ha = per_half.coarse_ha
if use_adaptive:
    # --- TWO-PASS ADAPTIVE OVERSAMPLING (RELION parity) ---
    # Pass 1: coarse E-step at reduced resolution to find
    #         significant orientations.
    # Pass 2: oversampled E+M at full current_size for significant
    #         orientations only.

    coarse_image_plan = plan_adaptive_image_size(
        coarse_size_healpix_order,
        expectation_windows,
        image_geometry,
        particle_diameter_angstrom=particle_diameter_ang,
        optics_image_sizes=optics_image_sizes,
        optics_pixel_sizes=optics_pixel_sizes,
        sealed_sampling_state=sealed_sampling_state,
        log=logger,
    )
    coarse_size = coarse_image_plan.size
    coarse_size_step_deg = coarse_image_plan.angular_step_deg
    coarse_cs = coarse_size if coarse_size < grid_size else None

    logger.info(
        "Adaptive oversampling: pass 1 at coarse_size=%s, "
        "pass 2 at current_size=%s (oversampling=%d, particle_diameter=%s)",
        coarse_cs,
        cs_for_engine,
        state.adaptive_oversampling,
        (f"{float(particle_diameter_ang):.1f} A" if particle_diameter_ang is not None else "box_size"),
    )

# D.2: per-class noise stats (K-tuple of NoiseStats per half) for the
# per-class sigma_offset C1 update at end-of-iter. K=1 paths leave
# this None; K-class paths populate from k_class_result.noise_stats.
noise_stats_per_half = per_half.noise_stats
noise_stats_per_half_per_class = per_half.noise_stats_per_class

projectors = [None, None]
captured_projector_state = replay_result.relion_projector_state
selected_gemm_global = adaptive.coarse_engine in {"gemm_hybrid", "gemm_dense"}
if captured_projector_state is not None and not (use_local or use_adaptive or selected_gemm_global):
    raise RuntimeError(
        "captured RELION Projector::data was supplied but this iteration has no projector scoring path"
    )
if use_local or use_adaptive or selected_gemm_global:
    projector_t0 = time.time()
    if captured_projector_state is not None:
        projectors = _validate_captured_relion_projector_for_iteration(
            captured_projector_state,
            current_size=model_current_size_for_engine,
            volume_shape=volume_shape,
            padding_factor=PROJECTION_PADDING_FACTOR,
            n_classes=n_classes,
        )
        logger.info(
            "RELION mode: using captured exact Projector::data at current_size=%s "
            "r_max=%s manifest=%s",
            model_current_size_for_engine,
            None if projectors[0] is None else projectors[0].r_max,
            captured_projector_state.source_manifest_sha256,
        )
    else:
        for half in halves:
            if half.dataset.n_units == 0:
                logger.info(
                    "RELION mode: skipping Projector::data build for empty half-%d dataset",
                    half.index + 1,
                )
                continue
            projector = prepare_scoring_projector(
                reference_model.maps[half.index],
                volume_shape=volume_shape,
                current_size=model_current_size_for_engine,
                padding_factor=PROJECTION_PADDING_FACTOR,
                n_classes=n_classes,
                reusable=shared_projector_half1 if half.index == 0 else None,
                real_references=(
                    initial_real_references_by_half[half.index]
                    if iteration == 0
                    else None
                ),
                dump_label=f"iter{iteration:03d}_half{half.index}",
            )
            projectors[half.index] = projector
        logger.info(
            # The slab dtype decides whether pass-2 projection runs on
            # the native texture projector or the vmapped JAX fallback
            # (_relion_projector_texture_enabled requires complex64),
            # so record it rather than leaving the path implicit.
            "RELION mode: built exact Projector::data for scoring at current_size=%s r_max=%s "
            "dtype=%s in %.2fs",
            model_current_size_for_engine,
            None if projectors[0] is None else projectors[0].r_max,
            None if projectors[0] is None else projectors[0].data.dtype,
            time.time() - projector_t0,
        )

# Freeze the exact iteration-start curve used by RELION's scale XA/AA
# shell gate.  The scheduling variable is updated again after the
# reconstruction, before parity diagnostics are written.
scale_correction_data_vs_prior_this_iter = previous_data_vs_prior_for_scheduling

diagnostic_half_indices = _significance_dump_half_indices(
    numbered_iteration=numbered_relion_iteration,
    n_classes=n_classes,
    experiment_datasets=experiment_datasets,
)
# The two halves are independent inside the E-step. Extracting one
# half's work into a function changes neither what runs nor its
# order; it makes the two callable independently, which is what the
# overlap option uses. Serial dispatch stays the default.
if use_local:
    if local_sampling.search.oversampling_order > 0:
        local_adaptive_full_parent = _local_adaptive_pass2_full_parent_enabled()
        local_adaptive_rotation_only = _local_adaptive_pass2_rotation_only_enabled()
        local_adaptive_denominator_mode = (
            _local_adaptive_pass2_denominator_support_mode()
        )
    else:
        local_adaptive_full_parent = False
        local_adaptive_rotation_only = False
        local_adaptive_denominator_mode = None

numbered_grid = sampling.TrialGrid(
    rotations=effective_rotations,
    rotation_eulers=effective_rotation_eulers,
    mstep_rotations=effective_mstep_rotations,
    translations=current_translations,
)
numbered_variant = DenseVariantPolicy(
    firstiter_score_mode_this_iter=firstiter_score_mode_this_iter,
    firstiter_winner_take_all_this_iter=firstiter_winner_take_all_this_iter,
    k_class_enabled=k_class_enabled,
    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
    firstiter_coarse_current_size=coarse_cs if use_adaptive else None,
    firstiter_fine_current_size=cs_for_engine if use_adaptive else None,
    firstiter_log_label="" if use_adaptive else "(non-adaptive site) ",
    firstiter_updates_em_kwargs_ibs=bool(use_adaptive),
)
if use_local:
    numbered_sampling = local_sampling
    numbered_local_diagnostics = LocalDiagnosticPolicy(
        iteration=iteration,
        debug_iteration=numbered_relion_iteration,
        save_intermediates_dir=debug.save_intermediates_dir,
        collect_local_search_profile=collect_local_search_profile,
        diagnostic_score_only=bool(debug.stop_after_local_search_score_only),
        local_profile_history=history.local_profile_history,
        adaptive_pass2_full_parent=local_adaptive_full_parent,
        adaptive_pass2_rotation_only=local_adaptive_rotation_only,
        adaptive_pass2_denominator_mode=local_adaptive_denominator_mode,
    )
else:
    numbered_sampling = DenseSamplingSpec(
        effective_rotations=(
            adaptive_pass1_rotations
            if use_adaptive and adaptive_pass1_rotations is not None
            else effective_rotations
        ),
        current_translations=current_translations,
        base_translations=base_translations,
        current_healpix_order=current_rotation_grid.healpix_order,
        oversampling_order=state.adaptive_oversampling,
        translation_step=state.translation_step,
        coarse_engine=adaptive.coarse_engine,
        random_perturbation=random_perturbation,
        cs_for_engine=cs_for_engine,
        model_current_size_for_engine=model_current_size_for_engine,
        coarse_rotation_ids=coarse_rotation_ids_for_scoring,
        coarse_scoring_rotations=adaptive_pass1_rotations if int(state.adaptive_oversampling) == 0 else None,
        symmetry=symmetry,
    )
    numbered_local_diagnostics = None
if tomo_halves:
    tomo_oversampling = int(state.adaptive_oversampling)
    tomo_coarse_size = local_sampling.coarse_image_window_size if use_local else coarse_cs
    numbered_tomo_sampling = TomoSampling(
        healpix_order=int(local_sampling.search.healpix_order) - tomo_oversampling if use_local else int(current_rotation_grid.healpix_order),
        oversampling_order=tomo_oversampling,
        offset_range_angst=float(state.translation_range) * image_geometry.pixel_size_angstrom,
        offset_step_angst=float(state.translation_step) * image_geometry.pixel_size_angstrom,
        random_perturbation=float(local_sampling.perturbation if use_local else random_perturbation),
        coarse_size=int(cryo.image_shape[0] if tomo_coarse_size is None else tomo_coarse_size),
        fine_size=int(cryo.image_shape[0] if cs_for_engine is None else cs_for_engine),
    )
else:
    numbered_tomo_sampling = None

def _run_half_estep(k):
    particle_half = halves[k]
    score_result = score_numbered_half(
        HalfScoringData(
            particles=particle_half,
            reference=reference_model.maps[k],
            mean_variance=reference_model.tau2_per_half[k],
            noise_variance=noise_model.variance_per_half[k],
            projector=projectors[k],
            scale_group_ids=follower_setup.scale_stats_group_ids_per_half[k],
            scale_group_count=follower_setup.scale_stats_group_count_per_half[k],
            scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
        ),
        numbered_grid,
        sampling=numbered_sampling,
        tomo_sampling=numbered_tomo_sampling,
        direction_priors=direction_log_priors[k],
        class_log_priors=class_log_priors,
        sigma_offset_angstrom=_sigma_offset_for_half(
            current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half, k,
        ),
        noise_radial=noise_model.radial_per_half[k] if not tomo_halves and particle_half.dataset.n_units else None,
        batch_planner=batch_planner,
        image_geometry=image_geometry,
        coarse_size_step_deg=coarse_size_step_deg,
        particle_diameter_ang=particle_diameter_ang,
        padded_volume_shape=padded_volume_shape,
        use_adaptive=use_adaptive,
        multi_shape_halves=multi_shape_halves,
        variant=numbered_variant,
        options=options,
        local_diagnostics=numbered_local_diagnostics,
        replay_prior_translations=_replay_prior_translations,
        initial_class_assignments=k_class.first_iteration_seed_classes if seed_iteration else None,
        single_class_iteration=single_class_iteration,
        scoring_dtype=scoring_dtype,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        relion_translation_angle_scale=relion_translation_angle_scale,
        iteration=iteration,
        numbered_relion_iteration=numbered_relion_iteration,
    )
    per_half.translation_search_bases[k] = score_result.translation_search_base
    per_half.pose_rotations[k] = score_result.pose_rotations
    per_half.pose_rotation_eulers[k] = score_result.pose_rotation_eulers
    per_half.coarse_ha[k] = score_result.coarse_ha
    if particle_half.dataset.n_units != 0:
        score_result = _maybe_host_offload_half0_local_accumulators(
            half_index=k,
            use_local=use_local,
            k_class_enabled=k_class_enabled,
            score_result=score_result,
            log=logger,
        )
    per_half.update_from(k, score_result, dtype=scoring_dtype)
    record_numbered_half(
        score_result,
        particle_half,
        per_half,
        significance,
        profile_history=history.global_profile_history,
        iteration=iteration,
        image_window_size=cs_for_engine,
        healpix_order=current_rotation_grid.healpix_order,
        k_class_enabled=k_class_enabled,
    )

_overlap_active = _half_overlap_active(
    options.overlap.overlap_halves,
    diagnostic_half_indices=diagnostic_half_indices,
    log=logger,
)
if _overlap_active:
    _run_halves_overlapped(_run_half_estep, diagnostic_half_indices)
    k = diagnostic_half_indices[-1]
else:
    for k in diagnostic_half_indices:
        _run_half_estep(k)
if diagnostic_half_indices != (0, 1):
    raise RuntimeError(
        "targeted half-only significance diagnostic returned without writing its "
        "complete target set; refusing to continue with one half missing"
    )

Ft_y_0, Ft_y_1 = per_half.Ft_y
Ft_ctf_0, Ft_ctf_1 = per_half.Ft_ctf

# E-step + per-half M-step accumulators are now both populated.
_parity_dump.mark_stage(iteration, "e_step")
from relax.cuda.kernels import drain_relion_preprocess_checks
drain_relion_preprocess_checks()
significance.combine()
```

### Complete recording operation and count owner

The recorder consumes already-published data and an individual profile-history
list, rather than a controller context or entire history object. Its nine inputs
are the score payload, half identity, published slots, count owner and five
profile/mode values. The count methods own conversion and aggregation; neither
adds another scoring pass or JIT boundary. The original scorer implementations
and large-array owners remain unchanged.

[relax/refinement/expectation.py](../../relax/refinement/expectation.py) (line 58):

```python
@dataclass
class SignificanceStatistics:
    """Half counts and aggregates in recording order for one expectation.

    See ``docs/math/relion_refinement_algorithm.md#iteration-convergence-policy``.
    """

    per_half: list = field(default_factory=lambda: [None, None], init=False)
    recorded: np.ndarray | None = field(default=None, init=False)
    convergence: np.ndarray | None = field(default=None, init=False)
    _recorded_parts: list = field(default_factory=list, init=False, repr=False)
    _convergence_parts: list = field(default_factory=list, init=False, repr=False)

    def record(self, half_index: int, counts, *, for_convergence: bool) -> None:
        if counts is not None:
            counts = np.asarray(counts, dtype=np.int32)
            self.per_half[half_index] = counts
            self._recorded_parts.append(counts)
            if for_convergence:
                self._convergence_parts.append(counts)

    def combine(self) -> None:
        """Combine after both halves and preprocessing checks have completed."""
        if self._convergence_parts:
            self.convergence = np.concatenate(self._convergence_parts, axis=0)
        if self._recorded_parts:
            self.recorded = np.concatenate(self._recorded_parts, axis=0)
```

[relax/refinement/expectation.py](../../relax/refinement/expectation.py) (line 87):

```python
def record_numbered_half(
    score_result: HalfScoreResult,
    particle_half: HalfSet,
    per_half: PerHalfOutputs,
    significance: SignificanceStatistics,
    *,
    profile_history: list,
    iteration: int,
    image_window_size: int | None,
    healpix_order: int,
    k_class_enabled: bool,
) -> None:
    """Record an already-published half's profile, support counts and captures.

    Empty halves keep the existing zero capture without a profile or device
    panel flush. Scoring and accumulator offloading remain with the caller.
    """
    k = particle_half.index
    if particle_half.dataset.n_units == 0:
        _parity_dump.collect_e_step(
            half=k,
            em_stats=score_result.em_stats,
            hard_assignment=score_result.ha,
            coarse_hard_assignment=per_half.coarse_ha[k],
            noise_stats=score_result.noise_stats,
            Ft_y=score_result.Ft_y,
            Ft_ctf=score_result.Ft_ctf,
            pose_rotation_eulers=per_half.pose_rotation_eulers[k],
            best_pose_rotation_eulers=per_half.best_pose_rotation_eulers[k],
            best_pose_translations=per_half.best_pose_translations[k],
            translation_search_base=per_half.translation_search_bases[k],
            original_image_indices=np.zeros(0, dtype=np.int64),
        )
        return

    _record_score_profile(
        profile_history,
        score_result,
        phase="iteration",
        iteration=iteration,
        relion_iteration=iteration + 1,
        half_index=k,
        current_size=image_window_size,
        healpix_order=healpix_order,
        k_class_enabled=k_class_enabled,
    )
    significance.record(k, score_result.significant_counts, for_convergence=not k_class_enabled)
    bpref_diagnostics.flush_selected_bpref_device_panel(iteration_index=iteration, half_index=k)

    try:
        original_image_indices = np.asarray(
            particle_half.dataset._index_layout.original_image_indices_for_local(
                np.arange(particle_half.dataset.n_images, dtype=np.int32)
            ),
            dtype=np.int64,
        )
    except Exception:
        original_image_indices = None
    _parity_dump.collect_e_step(
        half=k,
        em_stats=score_result.em_stats,
        hard_assignment=score_result.ha,
        coarse_hard_assignment=per_half.coarse_ha[k],
        noise_stats=per_half.noise_stats[k],
        Ft_y=score_result.Ft_y,
        Ft_ctf=score_result.Ft_ctf,
        pose_rotation_eulers=per_half.pose_rotation_eulers[k],
        best_pose_rotation_eulers=per_half.best_pose_rotation_eulers[k],
        best_pose_translations=per_half.best_pose_translations[k],
        translation_search_base=per_half.translation_search_bases[k],
        original_image_indices=original_image_indices,
    )
```

### Actual convergence and checkpoint consumers

Pose publication, noise estimation and correction updates still precede this
completed-iteration convergence call. It consumes the same native support
aggregate as before. Checkpoint finishing still follows convergence and sigma
updates; it reads per-half counts in physical half order. Model writes and
convergence decision timing remain in their established locations.

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 2816):

```python
# --- Update convergence state ---
# This checks assignment changes, resolution stalls, and may trigger
# angular step refinement or convergence.
state, accuracy_replay = update_iteration_convergence(
    state,
    pose_comparison,
    options,
    image_geometry=image_geometry,
    iteration=iteration,
    native_sampling_boundary=native_sampling_boundary,
    scheduling_resolution_shell=resolution_estimate.scheduling_shell,
    replay_dir=perturb_replay_relion_dir,
    translations=current_translations,
    current_assignments=current_combined_ha,
    previous_assignments=previous_combined_ha,
    current_classes=current_combined_classes,
    previous_classes=previous_combined_classes,
    max_posterior=combined_max_posterior,
    ave_pmax=ave_pmax,
    significant_counts=significance.convergence,
    exact_acc_rot=exact_acc_rot_this_iter,
    exact_acc_trans=exact_acc_trans_this_iter,
    log=logger,
)
iter_acc_rot = accuracy_replay.acc_rot
iter_acc_trans = accuracy_replay.acc_trans
```

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 2893):

```python
# --- RELION's run_itNNN files (ml_optimiser.cpp:3489) ---
checkpoint_writer = options.checkpoint.writer
if checkpoint_writer is not None and checkpoint_writer.due(numbered_relion_iteration):
    # The files hold incr_size/has_high_fsc_at_limit after this iteration's FSC
    # update, which the loop applies (idempotently) at the top of the next one.
    incr_size_after, high_fsc_after = relion_incr_size, relion_has_high_fsc_at_limit
    if not k_class_enabled:
        incr_size_after, high_fsc_after = update_relion_growth_state_from_fsc(
            _truncate_fsc_for_current_size_growth(
                tau2_fsc_for_update,
                current_size=current_size,
                grid_size=grid_size,
                dtype=scoring_dtype,
            ),
            current_size,
            incr_size=relion_incr_size,
            has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
        )
    snapshot = snapshot_capture.begin(
        numbered_relion_iteration,
        state,
        sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
        current_size=current_size,
        incr_size=incr_size_after,
        has_high_fsc_at_limit=high_fsc_after,
        random_perturbation=random_perturbation,
        acc_rot_per_class=model_acc_rot_per_class,
        acc_trans_per_class_angstrom=model_acc_trans_per_class,
    )
    snapshot = snapshot_capture.finish(
        snapshot,
        reference_model.maps,
        unreg_means,
        (
            mean_signal_variance_shells
            if k_class_enabled
            else [details["prior_shells"] for details in tau2_update_details_per_half]
        ),
        previous_data_vs_prior_for_scheduling,
        noise_model.radial_per_half,
        fsc=fsc,
        fsc_for_growth=None if k_class_enabled else tau2_fsc_for_update,
        class_weights=class_weights if k_class_enabled else None,
        direction_priors=direction_priors,
        half_inputs=halves,
        class_assignments=class_assignments if k_class_enabled else None,
        max_posterior=max_posterior_per_half,
        significant_counts=significance.per_half,
        avg_norm_correction=avg_norm_corrections_for_dump,
    )
    checkpoint_writer(snapshot)
```

History and the existing local-profile early return read `significance.recorded`;
particle-state capture reads `significance.per_half`. The inlined-controller proof
covers those consumers as well as the complete calling flow above.

### Evidence and remaining work

The initial focused inventory passed 127 cases and found one stale checkpoint
fixture still supplying the retired local name. That fixture now supplies the
count owner, retaining its weak-reference check that old snapshot maps expire
before new copies. All 19 checkpoint cases then pass, covering the remaining
failed case and the previously passing checkpoint cases. All 45 affected
controller cases pass (217 deselected); there are zero skips. Fifteen new cases
cover absent/empty counts, native/Class3D policy, half identity, metadata refusal,
profile/count/panel/capture order and borrowed accumulator identity.

Structural comparison inlines the actual recorder and count methods into the
controller and obtains exact AST equality after equivalent empty/nonempty
routing, five count bindings and two redundant accumulator aliases are restored.
Numerical calculations, casts, model writes, reductions, capture arguments and
synchronization are not removed from the comparison. All four pre-existing
expectation functions are unchanged. See the
[package receipt](final_search_patch_status.md#earlier-half-expectation-recording)
for commands, source manifests, the repaired failure and reproduction.

The numbered controller still spans 2,969 lines and command `main()` 1,992.
Publication is clearer, but phase construction and the 40-name half closure
remain unfinished. Class3D prior aggregation still awaits the earlier user
buffer-lifetime choice. No new module, numerical kernel, GPU job, publication or
scientific/performance acceptance was added. Complete the representative flow
and obtain design review before freezing it for production K1/exactly-K4 quality,
peak-memory and matched-GPU performance qualification.

## Reconstruction settings and visible model updates

This package resolves repeated run invariants in the existing reconstruction
owner. `ReconstructionSettings` now also carries regularization strength, particle
diameter and the initial reference filter. It is constructed once from resolved
options and consumed by numbered/final prior estimation, MAP reconstruction,
unregularized maps, postprocessing and Class3D captures. It adds no module,
function, forwarding layer or numerical kernel.

For a change to a reconstruction setting or mask, a maintainer can inspect its
input binding and the reconstruction owner without tracing parallel overrides
through scoring, batching, particle loading or convergence. The underlying
RECOVAR solve remains unchanged. A change to scientific order still belongs in
the controller and requires reviewing the flow below.

### Ownership and remaining explicit inputs

| Input | Producer and lifetime | Consumers and updates |
| --- | --- | --- |
| Geometry, padding, map floor, regularization, particle mask and initial filter | Resolved run options and input geometry; one frozen run-level settings object | Prior estimation, regularized/unregularized reconstruction, postprocessing and captures; no per-call override |
| Current image window and accumulator layout | This expectation and its output metadata | Each prior/reconstruction call at its current phase; retained explicitly |
| Prior arrays and radial shells | Current FSC/half weights or preceding class references/replay | MAP operands; controller installs model updates and retains CC taper/host parking |
| First-iteration CC decision | Numbered iteration policy | Postprocessing and later reporting taper; separate from the run's filter setting |
| Solvent-FSC pixel size | Existing dataset metadata scalar | Existing radius calculation; retained separately because normalized image geometry can change its scalar type |

The pixel-size duplication is a concrete unresolved numerical contract, not a
pattern to copy. Replacing the NumPy metadata scalar with the geometry's Python
float can change promotion in a radius calculation. Trace that boundary and
qualify its arithmetic before removing the last separate argument.

### Construction and actual numbered calling flow

The controller finishes expectation, resolves accumulator shape/precision and
checks the optional finite guard before this extract. It then combines Class3D
accumulators or joins K1 low frequencies, snapshots preceding references, selects
the correct prior policy, installs tau2, releases old maps, reconstructs and
postprocesses. Untapered tau2 is used for MAP before first-CC reporting taper.
K1 host parking follows the taper. Posterior/direction updates and convergence
continue afterwards at their existing positions.

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1000):

```python
@dataclass(frozen=True, kw_only=True)
class ReconstructionSettings:
    """Run-level geometry, regularization, mask and initial-filter settings."""

    grid_size: int
    voxel_size: object
    volume_shape: tuple
    padding_factor: int
    projection_padding_factor: int
    minres_map: int
    width_mask_edge: int
    fmask_edge: int
    tau2_fudge: float
    particle_diameter_angstrom: float | None
    first_iteration_lowpass_angstrom: float | None
```

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 507):

```python
reconstruction_settings = ReconstructionSettings(
    grid_size=grid_size,
    voxel_size=image_geometry.pixel_size_angstrom,
    volume_shape=volume_shape,
    padding_factor=RECONSTRUCTION_PADDING_FACTOR,
    projection_padding_factor=PROJECTION_PADDING_FACTOR,
    minres_map=RELION_MINRES_MAP,
    width_mask_edge=IMAGE_MASK_EDGE_PIXELS,
    fmask_edge=REFERENCE_FILTER_EDGE_SHELLS,
    tau2_fudge=parity.tau2_fudge,
    particle_diameter_angstrom=schedule.particle_diameter_ang,
    first_iteration_lowpass_angstrom=parity.relion_firstiter_ini_high_angstrom,
)
```

[relax/refinement/iteration_loop.py](../../relax/refinement/iteration_loop.py) (line 2114):

```python
retained_Ft_y_0_device = None
# RELION's --low_resol_join_halves averages the low-resolution shells of
# the K=1 half accumulators before the Wiener solve; see
# join_half_accumulators_at_low_resolution for the rationale and cap.
if k_class_enabled:
    Ft_y_combined = _combine_optional_half_accumulators(Ft_y_0, Ft_y_1, label="Ft_y")
    Ft_ctf_combined = _combine_optional_half_accumulators(Ft_ctf_0, Ft_ctf_1, label="Ft_ctf")
elif parity.low_resol_join_halves_angstrom is not None and parity.low_resol_join_halves_angstrom > 0:
    Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1, retained_Ft_y_0_device = join_half_accumulators_at_low_resolution(
        (Ft_y_0, Ft_y_1),
        (Ft_ctf_0, Ft_ctf_1),
        accumulator_volume_shape=mstep_accumulator_shape,
        grid_size=grid_size,
        voxel_size=cryo.voxel_size,
        padding_factor=RECONSTRUCTION_PADDING_FACTOR,
        low_resolution_angstrom=parity.low_resol_join_halves_angstrom,
        pixel_resolutions=history.pixel_resolutions,
        current_resolution=getattr(state, "current_resolution", float("inf")),
        preserve_inputs=False,
        return_retained_first_numerator=True,
    )

# --- RELION-exact M-step ordering ---
# K=1 stays on RELION's split-half auto-refine path
# (compareTwoHalves -> updateSSNRarrays -> reconstruct).
# K>1 switches to RELION Class3D semantics:
#   1. combine the two half accumulators per class
#   2. carry the previous Iref power spectrum forward as tau2
#   3. run one Wiener solve per class
#
# Snapshot the previous-iter means BEFORE the reconstruction so sign
# alignment has a reference at iter 1.
if k_class_enabled:
    # K-class 256px maps are large enough that materializing both
    # previous class stacks on the host immediately after pass 2 can
    # SIGBUS under Slurm/tmp quota pressure.  JAX arrays are immutable;
    # keep device references here and let the later per-class tau2/sign
    # code transfer only the slices it actually needs.
    previous_means = [jnp.asarray(mean) if mean is not None else None for mean in reference_model.maps]
else:
    previous_means = _snapshot_and_release_previous_k1_means(reference_model.maps)

_t_unreg_first = time.time()
if k_class_enabled:
    tau2_update_details_per_class = []
    mean_signal_variance_per_class = []
    mean_signal_variance_shells_per_class = []
    data_vs_prior_per_class = []
    # Dense RECOVAR accumulators live in the historical unnormalised
    # image frame: RELION BPref weight = Ft_ctf * N^4. Equivalently,
    # keep Ft_y/Ft_ctf in RECOVAR frame and scale RELION tau2 by N^4
    # before the Wiener solve. The same frame conversion is documented
    # in docs/math/ab_initio_initial_model_algorithm.md.
    kclass_tau2_frame_scale = float(grid_size) ** 4
    replay_class_tau2, replay_tau2_enabled, kclass_tau2_source = replay_policy._class_tau2_replay(
        iteration=iteration,
        n_classes=n_classes,
        iter_replay_override=iter_replay_override,
        replay=replay,
        logger=logger,
    )
    if iteration == 0:
        mean_variance_arr = jnp.asarray(reference_model.tau2)
        expected_shape = (n_classes, int(np.prod(volume_shape)))
        if tuple(mean_variance_arr.shape) == expected_shape:
            logger.info(
                "Class3D initial per-class tau2 volume available at iter=%d with shape=%s; "
                "M-step tau2 is recomputed from previous Iref power spectra",
                iteration + 1,
                tuple(mean_variance_arr.shape),
            )
    # CTF-premultiplied images: RELION's average CTF^2 correction of data_vs_prior
    # (setAverageCTF2; Class3D has no split halves and does not fix tau2).
    average_ctf2 = relion_ctf.premultiplied_average_ctf2(
        experiment_datasets, [half.scale_corrections for half in halves], image_current_size, grid_size
    )
    for class_idx in range(n_classes):
        logger.info(
            "Class3D tau2 update start: iter=%d class=%d/%d current_size=%d source=%s spectrum=%s",
            iteration + 1,
            class_idx + 1,
            n_classes,
            int(current_size),
            kclass_tau2_source,
            "host transform" if not has_previous_iteration or projectors[0] is None or projectors[0].power_spectrum is None else "scoring projector",
        )
        class_prior = estimate_class_prior(
            previous_means[0],
            Ft_ctf_combined,
            class_index=class_idx,
            settings=reconstruction_settings,
            current_size=current_size,
            accumulator_shape=mstep_accumulator_shape,
            full_half_axis=mstep_full_half_axis,
            frame_scale=kclass_tau2_frame_scale,
            projector_power_spectrum=(
                None
                if not has_previous_iteration or projectors[0] is None
                else projectors[0].power_spectrum
            ),
            replay_tau2_shells=replay_class_tau2 if replay_tau2_enabled else None,
            average_ctf2=average_ctf2,
        )
        mean_signal_variance_per_class.append(class_prior.variance)
        mean_signal_variance_shells_per_class.append(class_prior.shells)
        data_vs_prior_per_class.append(class_prior.data_vs_prior)
        tau2_update_details_per_class.append(class_prior.details)
        _kclass_dump_dir = os.environ.get("RELAX_KCLASS_DUMP_DIR")
        if _kclass_dump_dir:
            reconstruction_diagnostics.write_class_mstep(
                class_prior,
                numerators=Ft_y_combined,
                denominators=Ft_ctf_combined,
                half_denominators=(Ft_ctf_0, Ft_ctf_1),
                references=previous_means,
                settings=reconstruction_settings,
                output_dir=_kclass_dump_dir,
                class_index=class_idx,
                current_size=current_size,
                iteration=iteration,
                source=kclass_tau2_source,
                accumulator_shape=mstep_accumulator_shape,
                full_half_axis=mstep_full_half_axis,
                frame_scale=kclass_tau2_frame_scale,
            )
        logger.info(
            "Class3D tau2 update done: iter=%d class=%d/%d elapsed=%.1fs",
            iteration + 1,
            class_idx + 1,
            n_classes,
            time.time() - _t_unreg_first,
        )
    mean_signal_variance = jnp.stack(mean_signal_variance_per_class, axis=0)
    mean_signal_variance_shells = jnp.stack(mean_signal_variance_shells_per_class, axis=0)
    data_vs_prior_iter = np.stack(
        [np.asarray(dvp, dtype=scoring_dtype) for dvp in data_vs_prior_per_class],
        axis=0,
    )
    history.data_vs_prior_trajectory.append(data_vs_prior_iter)
    previous_data_vs_prior_for_scheduling = data_vs_prior_iter
    tau2_update_details = _stack_class_tau2_update_details(tau2_update_details_per_class)
    logger.info(
        "Computed iter-%d Class3D tau2 from %s: %.1fs",
        iteration + 1,
        kclass_tau2_source,
        time.time() - _t_unreg_first,
    )
else:
    mean_signal_variance_shells = None
    # Optional dump of post-join Ft_y, Ft_ctf for shell-by-shell parity
    # comparison against RELION's RELAX_MSTEP_DUMP_DIR. Activated by
    # RELAX_BPREF_ACCUM_DUMP_DIR. One npz per iteration.
    _bpref_accum_dir = os.environ.get("RELAX_BPREF_ACCUM_DUMP_DIR")
    if _bpref_accum_dir and _bpref_boundary_iteration_matches:
        _save_bpref_accumulators(
            _bpref_accum_dir,
            stage="accum",
            iteration=iteration,
            current_size=current_size,
            padding_factor=RECONSTRUCTION_PADDING_FACTOR,
            grid_size=grid_size,
            voxel_size=cryo.voxel_size,
            volume_shape=volume_shape,
            accumulator_shape=mstep_accumulator_shape,
            Ft_y_0=Ft_y_0,
            Ft_y_1=Ft_y_1,
            Ft_ctf_0=Ft_ctf_0,
            Ft_ctf_1=Ft_ctf_1,
        )
    split_prior = estimate_split_half_prior(
        (Ft_y_0, Ft_y_1),
        (Ft_ctf_0, Ft_ctf_1),
        reconstruction_settings,
        current_size=current_size,
        accumulator_shape=mstep_accumulator_shape,
        full_half_axes=per_half.mstep_full_half_axis,
        do_solvent_fsc_correction=parity.do_solvent_fsc_correction,
        pixel_size_angstrom=cryo.voxel_size,
        iteration=iteration,
        scoring_dtype=scoring_dtype,
        started_at=_t_unreg_first,
        log=logger,
    )
    current_iter_fsc = split_prior.fsc
    tau2_fsc_for_update = split_prior.fsc_for_update
    mean_signal_variance = split_prior.variance
    mean_signal_variance_per_half = split_prior.variance_per_half
    mean_signal_variance_shells_per_half = split_prior.shells_per_half
    tau2_update_details_per_half = split_prior.details_per_half
    # Diagnostics follow the half-1 model.star, matching the parity report.
    tau2_update_details = tau2_update_details_per_half[0]
    del split_prior
    logger.info(
        "tau2 update from THIS-iter FSC: old_max=%.4e new_max=%.4e half_max=(%.4e, %.4e)",
        float(jnp.max(jnp.abs(reference_model.tau2))),
        float(jnp.max(jnp.abs(mean_signal_variance))),
        float(jnp.max(jnp.abs(mean_signal_variance_per_half[0]))),
        float(jnp.max(jnp.abs(mean_signal_variance_per_half[1]))),
    )
reference_model.tau2 = mean_signal_variance
if not k_class_enabled:
    reference_model.tau2_per_half = _updated_mean_variance_per_half(
        reference_model.tau2,
        mean_signal_variance_per_half,
        use_per_half_mean_variance=parity.use_per_half_mean_variance,
    )
else:
    reference_model.tau2_per_half = [reference_model.tau2, reference_model.tau2]

# --- Free previous-iteration means to reclaim GPU memory ---
# (previous_means already snapshotted earlier for FSC sign alignment)
for k in range(2):
    reference_model.maps[k] = None

# --- Now reconstruct the regularized means ---
_t_recon = time.time()
if k_class_enabled:
    class_tau = (
        mean_signal_variance_shells
        if mean_signal_variance_shells is not None
        else mean_signal_variance
    )
    shared_class_means = reconstruct_class_means(
        Ft_y_combined,
        Ft_ctf_combined,
        class_tau,
        reconstruction_settings,
        n_classes=n_classes,
        iteration=iteration,
        current_size=current_size,
        accumulator_volume_shape=mstep_accumulator_shape,
        tau_is_1d=mean_signal_variance_shells is not None,
    )
    reference_model.maps[0] = shared_class_means
    reference_model.maps[1] = shared_class_means
else:
    tau_by_half = (
        mean_signal_variance_shells_per_half
        if mean_signal_variance_shells_per_half is not None
        else mean_signal_variance_per_half
    )
    reference_model.maps[:] = reconstruct_k1_means(
        (Ft_y_0, Ft_y_1),
        (Ft_ctf_0, Ft_ctf_1),
        tau_by_half,
        reconstruction_settings,
        current_size=current_size,
        accumulator_volume_shape=mstep_accumulator_shape,
        tau_is_1d=mean_signal_variance_shells_per_half is not None,
        retained_first_numerator=retained_Ft_y_0_device,
    )
postprocess_reconstructed_means(
    reference_model.maps,
    reconstruction_settings,
    n_classes=n_classes,
    iteration=iteration,
    current_size=current_size,
    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
)
logger.info(
    "Regularized reconstruction (2 halves + flatten): %.1fs",
    time.time() - _t_recon,
)
retained_Ft_y_0_device = None


# RELION reconstructs the first-iteration CC maps with the untapered
# updateSSNRarrays tau2.  Only afterwards does
# initialLowPassFilterReferences taper tau2/data_vs_prior for the
# model state and reporting; that tapered spectrum is explicitly not
# used in the reconstruction calculation (ml_optimiser.cpp:5296-5328).
if (
    not k_class_enabled
    and relion_firstiter_cc_this_iter
    and parity.relion_firstiter_ini_high_angstrom is not None
):
    tau2_taper = _firstiter_cc_ini_high_tau2_taper(
        len(tau2_update_details_per_half[0]["prior_shells"]),
        grid_size,
        cryo.voxel_size,
        parity.relion_firstiter_ini_high_angstrom,
        filter_edgewidth=REFERENCE_FILTER_EDGE_SHELLS,
    )
    radial_shells = np.asarray(
        fourier_transform_utils.get_grid_of_radial_distances(
            volume_shape,
            scaled=False,
            frequency_shift=0,
        ),
        dtype=np.int32,
    ).reshape(-1)
    radial_shells = np.minimum(radial_shells, len(tau2_taper) - 1)
    tau2_taper_volume = jnp.asarray(
        tau2_taper[radial_shells],
        dtype=scoring_dtype,
    )
    for half_idx in range(2):
        mean_signal_variance_per_half[half_idx] = (
            mean_signal_variance_per_half[half_idx] * tau2_taper_volume
        )
        for field in ("prior_shells", "ssnr_shells"):
            field_values = tau2_update_details_per_half[half_idx][field]
            tau2_update_details_per_half[half_idx][field] = field_values * jnp.asarray(
                tau2_taper,
                dtype=field_values.dtype,
            )
    mean_signal_variance = 0.5 * (
        mean_signal_variance_per_half[0] + mean_signal_variance_per_half[1]
    )
    reference_model.tau2 = mean_signal_variance
    reference_model.tau2_per_half = _updated_mean_variance_per_half(
        reference_model.tau2,
        mean_signal_variance_per_half,
        use_per_half_mean_variance=parity.use_per_half_mean_variance,
    )
    tau2_update_details = tau2_update_details_per_half[0]
    logger.info(
        "RELION iter-1 CC emulation: tapered post-reconstruction tau2/data-vs-prior "
        "with ini_high=%.2f A",
        float(parity.relion_firstiter_ini_high_angstrom),
    )
elif relion_firstiter_cc_this_iter and parity.relion_firstiter_ini_high_angstrom is not None:
    # Class3D tapers each class's tau2_class and data_vs_prior_class the
    # same way (ml_optimiser.cpp:6389-6420). RELION's comment calls this
    # output only, but the next E-step gates each class's scale sums on
    # data_vs_prior_class > 3 (:10473), so the untapered curve let
    # shells past ini_high into iteration 2's scale correction. The
    # class tau2 volumes are recomputed from the Iref power next
    # iteration, so only the shell curves carry the taper.
    def taper_shells(values):
        return _firstiter_cc_ini_high_tapered(
            values,
            grid_size,
            cryo.voxel_size,
            parity.relion_firstiter_ini_high_angstrom,
            filter_edgewidth=REFERENCE_FILTER_EDGE_SHELLS,
        )

    data_vs_prior_iter = taper_shells(data_vs_prior_iter)
    history.data_vs_prior_trajectory[-1] = data_vs_prior_iter
    previous_data_vs_prior_for_scheduling = data_vs_prior_iter
    mean_signal_variance_shells = jnp.asarray(taper_shells(np.asarray(mean_signal_variance_shells)))
    for field in ("prior_shells", "ssnr_shells"):
        tau2_update_details[field] = taper_shells(np.asarray(tau2_update_details[field]))
    logger.info(
        "RELION iter-1 CC emulation: tapered Class3D tau2/data-vs-prior with ini_high=%.2f A",
        float(parity.relion_firstiter_ini_high_angstrom),
    )
if not k_class_enabled:
    # The K=1 tau2 volumes are read again only by the next M-step (the
    # resident E-step does not use them). Keep them on the host between
    # uses, as RELION keeps tau2 as a host spectrum: at box 800 the four
    # float32 volumes are 8 GB of the device floor (GPU census, bigbox
    # 14480607).
    (
        reference_model.tau2,
        reference_model.tau2_per_half,
        mean_signal_variance,
        mean_signal_variance_per_half,
    ) = _host_tau2_volumes(
        reference_model.tau2,
        reference_model.tau2_per_half,
        mean_signal_variance,
        mean_signal_variance_per_half,
    )
_parity_dump.mark_stage(iteration, "recon")
```

### Actual final calling flow

Final expectation returns the established accumulator metadata. Unfiltered K1
maps are reconstructed before joining mutates the low frequencies. Final prior
policy differs from numbered policy, and the controller updates resolution before
MAP. Class weights and final products remain visible here.

[relax/refinement/finalization.py](../../relax/refinement/finalization.py) (line 587):

```python
final_reconstruct_t0 = time.time()
final_unfiltered_means_for_output = None
if not k_class_enabled:
    # RELION writes run_half{1,2}_class001_unfil.mrc from the converged half
    # BackProjectors saved before joinTwoHalvesAtLowResolution mutates their
    # low-frequency voxels, with do_map=false.  Keep this separate from the
    # Wiener-regularized half maps below so parity audits compare like
    # products.  Default RELION refinement sets BackProjector's
    # skip_gridding=true, so reconstruct uses the radial denominator floor
    # and direct division path before its final real-space gridding
    # correction and always applies softMaskOutsideMap inside
    # windowToOridimRealSpace; do_map=false only omits the tau2 prior.
    # They are reconstructed first, from the pre-join arrays, so the join
    # below can update those arrays in place instead of copying them: at
    # EMPIAR-10202's full box a copy of both halves' accumulators is 49 GB
    # of host memory (bigbox 14592943, 14594535). Each map is kept on the
    # host so the device holds only the reconstruction in progress.
    final_unfiltered_means_for_output = final_reconstruction.reconstruct_unfiltered_halfmaps(
        (final_Ft_y_0, final_Ft_y_1),
        (final_Ft_ctf_0, final_Ft_ctf_1),
        settings=reconstruction_settings,
        current_size=final_current_size,
        accumulator_shape=final_mstep_accumulator_shape,
    )
if not k_class_enabled and parity.low_resol_join_halves_angstrom is not None and parity.low_resol_join_halves_angstrom > 0:
    final_Ft_y_0, final_Ft_y_1, final_Ft_ctf_0, final_Ft_ctf_1 = join_half_accumulators_at_low_resolution(
        (final_Ft_y_0, final_Ft_y_1),
        (final_Ft_ctf_0, final_Ft_ctf_1),
        accumulator_volume_shape=final_mstep_accumulator_shape,
        grid_size=grid_size,
        voxel_size=image_geometry.pixel_size_angstrom,
        padding_factor=RECONSTRUCTION_PADDING_FACTOR,
        low_resolution_angstrom=parity.low_resol_join_halves_angstrom,
        pixel_resolutions=history.pixel_resolutions,
        current_resolution=state.current_resolution,
        preserve_inputs=False,
    )
if not k_class_enabled:
    # The unfiltered maps are made; drop the pass outputs' references so the
    # pre-join accumulators live only as long as the joined ones do.
    final_outs.Ft_y[0] = final_outs.Ft_y[1] = None
    final_outs.Ft_ctf[0] = final_outs.Ft_ctf[1] = None

final_ft_y = final_Ft_y_0 + final_Ft_y_1
final_ft_ctf = final_Ft_ctf_0 + final_Ft_ctf_1
final_iter_fsc = None
final_mstep_full_half_axis = _resolve_mstep_full_half_axis(
    final_outs.mstep_full_half_axis,
    default_axis=-1,
)
if k_class_enabled:
    class_weights = _class_weights_from_posterior(
        final_outs.class_posterior,
        n_classes,
        class_weights,
    )
    history.record_class_weights(
        class_weights,
        _class_weights_from_posterior(
            final_outs.class_full_posterior,
            n_classes,
            class_weights,
        ),
    )
    _t_final_tau2 = time.time()
    final_class_priors = final_reconstruction.compute_final_class_priors(
        final_ft_ctf,
        final_join_means[0],
        projector=final_projectors[0],
        n_classes=n_classes,
        settings=reconstruction_settings,
        current_size=final_current_size,
        accumulator_shape=final_mstep_accumulator_shape,
        full_half_axis=final_mstep_full_half_axis,
    )
    final_mean_variance = final_class_priors.variance
    final_mean_variance_shells = final_class_priors.shells
    final_data_vs_prior = final_class_priors.data_vs_prior
    final_tau2_update_details = final_class_priors.details
    logger.info(
        "RELION final all-data Class3D tau2 from Iref power spectra: old_max=%.4e new_max=%.4e "
        "dvp_shell_1=%.4f wall=%.1fs",
        float(jnp.max(jnp.abs(reference_model.tau2))),
        float(jnp.max(jnp.abs(final_mean_variance))),
        float(np.asarray(final_data_vs_prior)[0, 1]) if np.asarray(final_data_vs_prior).shape[-1] > 1 else float("nan"),
        time.time() - _t_final_tau2,
    )
else:
    _t_final_tau2 = time.time()
    final_halfmap_prior = final_reconstruction.compute_final_halfmap_prior(
        (final_Ft_y_0, final_Ft_y_1),
        (final_Ft_ctf_0, final_Ft_ctf_1),
        settings=reconstruction_settings,
        current_size=final_current_size,
        accumulator_shape=final_mstep_accumulator_shape,
        full_half_axis=final_mstep_full_half_axis,
        scoring_dtype=scoring_dtype,
    )
    final_iter_fsc = final_halfmap_prior.fsc
    final_mean_variance = final_halfmap_prior.variance
    final_tau2_update_details = final_halfmap_prior.details
    logger.info(
        "RELION final all-data tau2 from joined FSC: old_max=%.4e new_max=%.4e "
        "fsc_shell_1=%.4f wall=%.1fs",
        float(jnp.max(jnp.abs(reference_model.tau2))),
        float(jnp.max(jnp.abs(final_mean_variance))),
        float(np.asarray(final_iter_fsc)[1]) if np.asarray(final_iter_fsc).size > 1 else float("nan"),
        time.time() - _t_final_tau2,
    )

# RELION calls updateCurrentResolution after the final all-data
# iteration too (ml_optimiser_mpi.cpp:4329), from that iteration's
# whole-data DVP, so rlnCurrentResolution reports the final half-map FSC
# at 0.143 rather than the last split-half iteration's 0.5 crossing.
# Nothing after this point schedules on the resolution.
final_dvp = (
    final_data_vs_prior
    if k_class_enabled
    else np.asarray(final_tau2_update_details["ssnr_shells"], dtype=scoring_dtype)
)
final_res_shell = relion_current_resolution_shell(
    final_dvp,
    k_class_enabled=k_class_enabled,
    current_size=final_current_size,
    grid_size=grid_size,
    dtype=scoring_dtype,
)
state.previous_resolution = state.current_resolution
state.current_resolution = shell_index_to_resolution_angstrom(
    final_res_shell, image_geometry.box_size, image_geometry.pixel_size_angstrom
)
logger.info(
    "RELION final all-data current resolution: shell=%d res=%.2f A (last split-half iteration %.2f A)",
    int(final_res_shell),
    state.current_resolution,
    state.previous_resolution,
)

# Reconstruct the final volume from the COMBINED Ft_y/Ft_ctf accumulators
# at the full Nyquist resolution. Skip the join_halves step (we're already
# combining the two halves into one dataset for this final iter).
logger.info(
    "RELION final all-data reconstruction start: current_size=%d n_classes=%d",
    final_current_size,
    n_classes,
)
if k_class_enabled:
    final_maps = final_reconstruction.reconstruct_final_class_maps(
        final_ft_y,
        final_ft_ctf,
        final_mean_variance_shells,
        class_weights=class_weights,
        n_classes=n_classes,
        settings=reconstruction_settings,
        current_size=final_current_size,
        accumulator_shape=final_mstep_accumulator_shape,
    )
    final_class_means = final_maps.halves[0]
    class_assignments = final_outs.class_assignments
else:
    final_class_means = None
    final_backprojections = [
        (final_ft_ctf, final_ft_y),
        (final_Ft_ctf_0, final_Ft_y_0),
        (final_Ft_ctf_1, final_Ft_y_1),
    ]
    del final_ft_ctf, final_ft_y
    del final_Ft_ctf_0, final_Ft_y_0, final_Ft_ctf_1, final_Ft_y_1
    final_maps = final_reconstruction.reconstruct_final_halfmaps(
        final_backprojections,
        final_mean_variance,
        settings=reconstruction_settings,
        current_size=final_current_size,
        accumulator_shape=final_mstep_accumulator_shape,
    )
merged_mean = final_maps.merged
final_means_for_output = final_maps.halves
logger.info(
    "RELION final all-data reconstruction done: wall=%.1fs",
    time.time() - final_reconstruct_t0,
)
```

### Complete reconstruction consumers

These are the existing operations with their migrated interfaces. K1
half-specific radial priors still promote to the established reconstruction
precision, retained numerator donation is unchanged, and filtering still precedes
solvent flattening. Array creation, transfer, synchronization and deletion stay
in their original operations.

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 445):

```python
def estimate_class_prior(
    references,
    denominators,
    *,
    class_index,
    settings: ReconstructionSettings,
    current_size,
    accumulator_shape,
    full_half_axis,
    frame_scale,
    projector_power_spectrum=None,
    replay_tau2_shells=None,
    average_ctf2=None,
) -> ClassPriorEstimate:
    """Estimate a class prior from Iref power or diagnostic replay, then its weights.

    Index inside the operation so reference/projector views are created only
    on the native path and weight views follow prior estimation.
    """
    if replay_tau2_shells is not None:
        shells = jnp.asarray(replay_tau2_shells[class_index], dtype=jnp.float32)
        from recovar import utils

        variance = jnp.asarray(
            utils.make_radial_image(shells, settings.volume_shape, extend_last_frequency=True),
            dtype=jnp.float32,
        ).reshape(-1)
        relion_shells = shells / jnp.asarray(frame_scale, dtype=shells.dtype)
    else:
        variance, relion_shells, shells = _class_tau2_from_iref_power_spectrum(
            references[class_index], settings.volume_shape,
            padding_factor=settings.padding_factor,
            current_size=current_size,
            frame_scale=frame_scale,
            projector_power_spectrum=(
                None if projector_power_spectrum is None else projector_power_spectrum[class_index]
            ),
        )
    weight_shells = regularization_relion._compute_relion_weight_shell_stats(
        denominators[class_index], settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        shell_rounding="round",
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
    )
    data_vs_prior, details = _class_tau2_update_details(
        denominators[class_index], shells, weight_shells, settings.volume_shape,
        padding_factor=settings.padding_factor,
        tau2_fudge=settings.tau2_fudge,
        current_size=current_size,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
        average_ctf2=average_ctf2,
    )
    return ClassPriorEstimate(
        variance=variance, shells=shells, relion_shells=relion_shells,
        data_vs_prior=data_vs_prior, details=details, weight_shells=weight_shells,
    )
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1017):

```python
@dataclass(frozen=True)
class SplitHalfPrior:
    """K1 scoring/reconstruction priors and raw versus corrected FSC."""

    variance: object
    variance_per_half: list
    shells_per_half: list
    fsc: object
    fsc_for_update: object
    details_per_half: list[dict]
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1029):

```python
def estimate_split_half_prior(
    numerators,
    denominators,
    settings: ReconstructionSettings,
    *,
    current_size,
    accumulator_shape,
    full_half_axes,
    do_solvent_fsc_correction,
    pixel_size_angstrom,
    iteration,
    scoring_dtype,
    started_at,
    log,
) -> SplitHalfPrior:
    """Estimate independent half priors from this expectation's shared FSC.

    Raw FSC is reported; solvent-corrected FSC, when enabled, drives tau2 and
    size growth. The two halves retain their own Fourier weights.
    See ``docs/math/relion_refinement_algorithm.md`` for the M-step ordering.
    """

    current_iter_fsc = regularization_relion.compute_relion_fsc_from_backprojector(
        numerators[0],
        numerators[1],
        denominators[0],
        denominators[1],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        accumulator_volume_shape=accumulator_shape,
        output_dtype=scoring_dtype,
    )
    log.info(
        "Computed iter-%d FSC for tau2 (RELION backprojector path): %.1fs",
        iteration + 1,
        time.time() - started_at,
    )
    raw_backprojector_fsc = current_iter_fsc
    tau2_fsc_for_update = current_iter_fsc
    if (
        do_solvent_fsc_correction
        and settings.particle_diameter_angstrom is not None
        and settings.particle_diameter_angstrom > 0
    ):
        from recovar.core import mask as _mask

        _t_solvent_fsc = time.time()
        unfiltered_half_maps = []
        for Ft_ctf_half, Ft_y_half in ((denominators[0], numerators[0]), (denominators[1], numerators[1])):
            unfiltered_real = _reconstruct_volume_eager(
                Ft_ctf_half,
                Ft_y_half,
                settings.volume_shape,
                settings.padding_factor,
                tau=None,
                tau2_fudge=settings.tau2_fudge,
                projection_padding_factor=settings.projection_padding_factor,
                # RELION's BackProjector::reconstruct(do_map=false)
                # still calls softMaskOutsideMap inside
                # windowToOridimRealSpace.
                use_spherical_mask=True,
                minres_map=settings.minres_map,
                current_size=int(current_size),
                return_real_space=True,
                accumulator_volume_shape=accumulator_shape,
            )
            unfiltered_real = np.asarray(
                jnp.asarray(unfiltered_real).reshape(settings.volume_shape),
                dtype=np.float64,
            ).real
            unfiltered_half_maps.append(unfiltered_real)

        flatten_radius = settings.particle_diameter_angstrom / (2.0 * pixel_size_angstrom)
        solvent_mask = np.asarray(
            _mask.raised_cosine_mask(
                settings.volume_shape,
                radius=flatten_radius,
                radius_p=flatten_radius + settings.width_mask_edge,
                offset=jnp.zeros(3),
                dtype=jnp.float64,
            ),
            dtype=np.float64,
        )
        tau2_fsc_for_update, solvent_fsc_details = regularization_relion.compute_relion_solvent_corrected_true_fsc(
            unfiltered_half_maps[0],
            unfiltered_half_maps[1],
            solvent_mask,
            current_size=int(current_size),
            rng_seed=int(1775735620 + iteration),
            return_details=True,
        )
        randomize_at = int(solvent_fsc_details["randomize_at"])
        probe_shell = max(1, randomize_at) if randomize_at > 0 else min(len(solvent_fsc_details["fsc_true"]) - 1, 1)
        corrected_shell = min(len(solvent_fsc_details["fsc_true"]) - 1, max(probe_shell, randomize_at + 2))
        log.info(
            "Computed iter-%d solvent-corrected true FSC for tau2: randomize_at=%d "
            "raw_fsc[%d]=%.4f masked=%.4f random_masked=%.4f true=%.4f; "
            "formula_shell[%d]: masked=%.4f random_masked=%.4f true=%.4f elapsed=%.1fs",
            iteration + 1,
            randomize_at,
            probe_shell,
            float(np.asarray(raw_backprojector_fsc)[probe_shell]),
            float(solvent_fsc_details["fsc_masked"][probe_shell]),
            float(solvent_fsc_details["fsc_random_masked"][probe_shell]),
            float(solvent_fsc_details["fsc_true"][probe_shell]),
            corrected_shell,
            float(solvent_fsc_details["fsc_masked"][corrected_shell]),
            float(solvent_fsc_details["fsc_random_masked"][corrected_shell]),
            float(solvent_fsc_details["fsc_true"][corrected_shell]),
            time.time() - _t_solvent_fsc,
        )
    elif do_solvent_fsc_correction:
        log.warning(
            "RELION solvent FSC correction requested but particle_diameter_ang is unset; using raw FSC for tau2"
        )

    # RELION calls BackProjector::updateSSNRarrays independently for each
    # half-map BPref.  The gold-standard FSC is shared, but sigma2/tau2
    # come from each half's own Fourier weight outside the joined shells.
    tau2_update_details_per_half = []
    mean_signal_variance_per_half = []
    for half_idx, Ft_ctf_half in enumerate((denominators[0], denominators[1])):
        full_half_axis = full_half_axes[half_idx]
        mean_signal_variance_k, _, tau2_update_details_k = regularization_relion.compute_relion_tau2_from_weights(
            Ft_ctf_half,
            Ft_ctf_half,
            tau2_fsc_for_update,
            settings.volume_shape,
            tau2_fudge=settings.tau2_fudge,
            padding_factor=settings.padding_factor,
            r_max=current_size // 2,
            return_details=True,
            full_half_axis=-1 if full_half_axis is None else int(full_half_axis),
            accumulator_volume_shape=accumulator_shape,
            output_dtype=scoring_dtype,
        )
        mean_signal_variance_per_half.append(mean_signal_variance_k)
        tau2_update_details_per_half.append(tau2_update_details_k)
    mean_signal_variance_shells_per_half = [
        details["prior_shells"] for details in tau2_update_details_per_half
    ]
    mean_signal_variance = 0.5 * (mean_signal_variance_per_half[0] + mean_signal_variance_per_half[1])
    return SplitHalfPrior(
        variance=mean_signal_variance,
        variance_per_half=mean_signal_variance_per_half,
        shells_per_half=mean_signal_variance_shells_per_half,
        fsc=current_iter_fsc,
        fsc_for_update=tau2_fsc_for_update,
        details_per_half=tau2_update_details_per_half,
    )
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1182):

```python
def reconstruct_k1_means(
    numerators_by_half,
    denominators_by_half,
    tau_by_half,
    settings: ReconstructionSettings,
    *,
    current_size,
    accumulator_volume_shape,
    tau_is_1d,
    retained_first_numerator=None,
) -> list:
    """Reconstruct both K=1 halves while preserving RELION buffer lifetime."""

    if len(tau_by_half) != 2:
        raise ValueError("K=1 reconstruction tau2 requires exactly two halves")
    cs_int = int(current_size) if current_size is not None else None
    reconstructed_means = []
    retained_device_numerator = retained_first_numerator
    for k, (Ft_y_half, Ft_ctf_half, tau_half) in enumerate(
        zip(numerators_by_half, denominators_by_half, tau_by_half)
    ):
        # This RELION build uses double RFLOAT in BackProjector::reconstruct.
        # Keep the stored/controller tau2 state compact, but promote the
        # reconstruction operand so 1 / (padding_factor**3 * tau2) is not
        # rounded in float32 before it enters the Wiener denominator.
        reconstruction_tau = jnp.asarray(tau_half, dtype=jnp.float64)
        reconstructed = _reconstruct_volume_eager(
            Ft_ctf_half,
            Ft_y_half,
            settings.volume_shape,
            settings.padding_factor,
            tau=reconstruction_tau,
            tau2_fudge=settings.tau2_fudge,
            projection_padding_factor=settings.projection_padding_factor,
            minres_map=settings.minres_map,
            current_size=cs_int,
            accumulator_volume_shape=accumulator_volume_shape,
            tau_is_1d=tau_is_1d,
            preserve_output_precision=True,
            relion_filter_scale=float(settings.volume_shape[0] ** 4),
            **(
                {"retained_device_numerator": retained_device_numerator}
                if k == 0 and retained_device_numerator is not None
                else {}
            ),
        ).reshape(-1)
        reconstructed_means.append(
            _finish_host_staged_reconstruction(reconstructed, Ft_ctf_half, Ft_y_half)
        )
        if k == 0 and retained_device_numerator is not None:
            retained_device_numerator = None
            gc.collect()
    return reconstructed_means
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1237):

```python
def reconstruct_class_means(
    combined_numerators,
    combined_denominators,
    tau_by_class,
    settings: ReconstructionSettings,
    *,
    n_classes,
    iteration,
    current_size,
    accumulator_volume_shape,
    tau_is_1d,
):
    """Reconstruct the shared Class3D stack from combined accumulators."""

    _t_recon = time.time()
    cs_int = int(current_size) if current_size is not None else None
    shared_class_maps = []
    for class_idx in range(n_classes):
        logger.info(
            "Class3D reconstruction start: iter=%d class=%d/%d current_size=%s",
            iteration + 1,
            class_idx + 1,
            n_classes,
            cs_int,
        )
        class_map = _reconstruct_volume_eager(
            combined_denominators[class_idx],
            combined_numerators[class_idx],
            settings.volume_shape,
            settings.padding_factor,
            tau=tau_by_class[class_idx],
            tau2_fudge=settings.tau2_fudge,
            projection_padding_factor=settings.projection_padding_factor,
            minres_map=settings.minres_map,
            current_size=cs_int,
            accumulator_volume_shape=accumulator_volume_shape,
            tau_is_1d=tau_is_1d,
        ).reshape(-1)
        shared_class_maps.append(class_map)
        logger.info(
            "Class3D reconstruction done: iter=%d class=%d/%d elapsed=%.1fs",
            iteration + 1,
            class_idx + 1,
            n_classes,
            time.time() - _t_recon,
        )
    shared_classes = jnp.stack(shared_class_maps, axis=0)
    logger.info(
        "Class3D reconstruction stack complete: iter=%d classes=%d elapsed=%.1fs",
        iteration + 1,
        n_classes,
        time.time() - _t_recon,
    )
    return shared_classes
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1293):

```python
def postprocess_reconstructed_means(
    means,
    settings: ReconstructionSettings,
    *,
    n_classes,
    iteration,
    current_size,
    relion_firstiter_cc_this_iter,
) -> None:
    """Apply premask capture, first-iteration filtering and solvent flattening.

    ``width_mask_edge`` is the real-space mask edge (RELION ``--maskedge``).
    ``fmask_edge`` is the Fourier edge of ``initialLowPassFilterReferences``;
    preserving their distinct units is part of the reconstruction contract.
    """

    for k in range(2):
        # Diagnostic: dump pre-mask Wiener output when env var set.
        _premask_dump = os.environ.get("RELAX_PREMASK_DUMP_DIR")
        if _premask_dump:
            from relax.diagnostics.reconstruction import write_premask_mean

            write_premask_mean(
                means[k], output_dir=_premask_dump, half_index=k, iteration=iteration,
                current_size=current_size, grid_size=settings.grid_size, voxel_size=settings.voxel_size,
                volume_shape=settings.volume_shape, n_classes=n_classes,
            )

        # RELION filters Iref inside maximizationOtherParameters, then calls
        # solventFlatten from the outer iteration loop.  These operations do
        # not commute: masking in real space after the Fourier low-pass adds a
        # small, deterministic high-shell tail.
        if relion_firstiter_cc_this_iter:
            if n_classes > 1:
                means[k] = jnp.stack(
                    [
                        _apply_relion_initial_lowpass_filter(
                            means[k][class_idx],
                            settings.volume_shape,
                            settings.voxel_size,
                            settings.first_iteration_lowpass_angstrom,
                            filter_edgewidth=settings.fmask_edge,
                        )
                        for class_idx in range(n_classes)
                    ],
                    axis=0,
                )
            else:
                means[k] = _apply_relion_initial_lowpass_filter(
                    means[k],
                    settings.volume_shape,
                    settings.voxel_size,
                    settings.first_iteration_lowpass_angstrom,
                    filter_edgewidth=settings.fmask_edge,
                )
        if (
            settings.particle_diameter_angstrom is not None
            and settings.particle_diameter_angstrom > 0
        ):
            flatten_radius = (
                float(settings.particle_diameter_angstrom) / (2.0 * float(settings.voxel_size))
                if n_classes == 1 else settings.particle_diameter_angstrom / (2.0 * settings.voxel_size)
            )
            solvent_mask = _make_relion_solvent_mask(
                settings.volume_shape,
                radius=flatten_radius,
                radius_p=flatten_radius + settings.width_mask_edge,
                offset=jnp.zeros(3),
                dtype=(means[k].real.dtype if n_classes <= 1 else means[k][0].real.dtype),
            )
            if n_classes > 1:
                flattened_classes = []
                for class_idx in range(n_classes):
                    vol_real = fourier_transform_utils.get_idft3(
                        means[k][class_idx].reshape(settings.volume_shape)
                    )
                    flattened_classes.append(
                        fourier_transform_utils.get_dft3(vol_real * solvent_mask).reshape(-1),
                    )
                means[k] = jnp.stack(flattened_classes, axis=0)
            else:
                means[k] = _apply_relion_solvent_flatten_k1(
                    means[k], solvent_mask, settings.volume_shape, half_index=k,
                )
                if _large_relion_solvent_mask_uses_compiled_builder(settings.volume_shape):
                    solvent_mask = None
    if relion_firstiter_cc_this_iter and settings.first_iteration_lowpass_angstrom is not None:
        logger.info(
            "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
            float(settings.first_iteration_lowpass_angstrom),
        )
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1391):

```python
def reconstruct_unregularized_k1_halfmaps(
    Ft_y_per_half,
    Ft_ctf_per_half,
    settings: ReconstructionSettings,
    *,
    accumulator_volume_shape=None,
) -> list:
    """Reconstruct each K=1 half from its own unregularized accumulator."""

    return [
        _reconstruct_volume_eager(
            Ft_ctf_half,
            Ft_y_half,
            settings.volume_shape,
            settings.padding_factor,
            tau=None,
            tau2_fudge=settings.tau2_fudge,
            projection_padding_factor=settings.projection_padding_factor,
            minres_map=settings.minres_map,
            accumulator_volume_shape=accumulator_volume_shape,
        )
        for Ft_ctf_half, Ft_y_half in zip(Ft_ctf_per_half, Ft_y_per_half)
    ]
```

[relax/refinement/mean_helpers.py](../../relax/refinement/mean_helpers.py) (line 1416):

```python
def reconstruct_unregularized_class_means(
    Ft_y_combined,
    Ft_ctf_combined,
    settings: ReconstructionSettings,
    n_classes,
    *,
    accumulator_volume_shape=None,
) -> list:
    """Reconstruct the shared K-class stack from combined accumulators."""

    unreg_shared = jnp.stack(
        [
            _reconstruct_volume_eager(
                Ft_ctf_combined[class_idx],
                Ft_y_combined[class_idx],
                settings.volume_shape,
                settings.padding_factor,
                tau=None,
                tau2_fudge=settings.tau2_fudge,
                projection_padding_factor=settings.projection_padding_factor,
                minres_map=settings.minres_map,
                accumulator_volume_shape=accumulator_volume_shape,
            ).reshape(-1)
            for class_idx in range(n_classes)
        ],
        axis=0,
    )
    return [unreg_shared, unreg_shared]
```

[relax/refinement/final_reconstruction.py](../../relax/refinement/final_reconstruction.py) (line 1):

```python
"""Prior estimation and map reconstruction for the final all-data expectation.

The controller applies resolution and class-weight updates between these steps.
See ``docs/math/relion_refinement_algorithm.md`` for the reconstruction order.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from relax.reconstruction import regularization_relion
from relax.refinement import mean_helpers
from relax.refinement.mean_helpers import ReconstructionSettings


@dataclass(frozen=True)
class HalfmapPrior:
    variance: object
    fsc: object
    details: dict


@dataclass(frozen=True)
class ClassPriors:
    variance: object
    shells: object
    data_vs_prior: np.ndarray
    details: dict


@dataclass(frozen=True)
class FinalMaps:
    merged: object
    halves: list


def reconstruct_unfiltered_halfmaps(
    numerators,
    denominators,
    *,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
) -> list:
    """Reconstruct host maps before low-frequency joining modifies the halves."""
    return [
        np.asarray(
            mean_helpers._reconstruct_volume_eager(
                half_ctf,
                half_y,
                settings.volume_shape,
                settings.padding_factor,
                tau=None,
                tau2_fudge=settings.tau2_fudge,
                projection_padding_factor=settings.projection_padding_factor,
                minres_map=settings.minres_map,
                current_size=current_size,
                accumulator_volume_shape=accumulator_shape,
                use_spherical_mask=True,
                grid_correct=True,
            ).reshape(-1)
        )
        for half_ctf, half_y in zip(denominators, numerators, strict=True)
    ]


def compute_final_halfmap_prior(
    numerators,
    denominators,
    *,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
    full_half_axis: int,
    scoring_dtype,
) -> HalfmapPrior:
    """Estimate the whole-data prior from the joined halves' backprojector FSC."""
    fsc = regularization_relion.compute_relion_fsc_from_backprojector(
        numerators[0],
        numerators[1],
        denominators[0],
        denominators[1],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        accumulator_volume_shape=accumulator_shape,
        output_dtype=scoring_dtype,
        full_is_hermitian=True,
    )
    variance, _, details = regularization_relion.compute_relion_tau2_from_weights(
        denominators[0],
        denominators[1],
        fsc,
        settings.volume_shape,
        tau2_fudge=settings.tau2_fudge,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        is_whole_instead_of_half=True,
        return_details=True,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
        weight_combination="sum",
        output_dtype=scoring_dtype,
    )
    return HalfmapPrior(variance=variance, fsc=fsc, details=details)


def compute_final_class_priors(
    denominator,
    references,
    *,
    projector,
    n_classes: int,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
    full_half_axis: int,
) -> ClassPriors:
    """Estimate each class prior from its reference and merged backprojection."""
    frame_scale = float(settings.grid_size) ** 4
    variances = []
    shells = []
    data_vs_prior = []
    details = []
    for class_idx in range(n_classes):
        prior = mean_helpers.estimate_class_prior(
            references,
            denominator,
            class_index=class_idx,
            settings=settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            full_half_axis=full_half_axis,
            frame_scale=frame_scale,
            projector_power_spectrum=None if projector is None else projector.power_spectrum,
        )
        variances.append(prior.variance)
        shells.append(prior.shells)
        data_vs_prior.append(prior.data_vs_prior)
        details.append(prior.details)
    return ClassPriors(
        variance=jnp.stack(variances, axis=0),
        shells=jnp.stack(shells, axis=0),
        data_vs_prior=np.stack(
            [np.asarray(value, dtype=np.float32) for value in data_vs_prior], axis=0,
        ),
        details=mean_helpers._stack_class_tau2_update_details(details),
    )


def reconstruct_final_class_maps(
    numerator,
    denominator,
    prior_shells,
    *,
    class_weights,
    n_classes: int,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
) -> FinalMaps:
    """Reconstruct class maps and their posterior-weighted merged map."""
    class_means = jnp.stack(
        [
            mean_helpers._reconstruct_volume_eager(
                denominator[class_idx],
                numerator[class_idx],
                settings.volume_shape,
                settings.padding_factor,
                tau=prior_shells[class_idx],
                tau2_fudge=settings.tau2_fudge,
                projection_padding_factor=settings.projection_padding_factor,
                minres_map=settings.minres_map,
                current_size=current_size,
                accumulator_volume_shape=accumulator_shape,
                tau_is_1d=True,
            ).reshape(-1)
            for class_idx in range(n_classes)
        ],
        axis=0,
    )
    merged = jnp.sum(
        jnp.asarray(class_weights, dtype=class_means.real.dtype)[:, None] * class_means,
        axis=0,
    )
    return FinalMaps(merged=merged, halves=[class_means, class_means])


def reconstruct_final_halfmaps(
    backprojections: list,
    prior,
    *,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
) -> FinalMaps:
    """Consume merged/half backprojections in order, keeping maps on the host.

    Each slot is a (denominator, numerator) pair: merged, half 1, half 2.
    Clearing a slot before reconstruction and releasing its arrays afterwards
    prevents retaining the large buffers after their last use.
    """
    maps = []
    for index in range(3):
        denominator, numerator = backprojections[index]
        backprojections[index] = None
        maps.append(
            np.asarray(
                mean_helpers._reconstruct_volume_eager(
                    denominator,
                    numerator,
                    settings.volume_shape,
                    settings.padding_factor,
                    tau=prior,
                    tau2_fudge=settings.tau2_fudge,
                    projection_padding_factor=settings.projection_padding_factor,
                    minres_map=settings.minres_map,
                    current_size=current_size,
                    accumulator_volume_shape=accumulator_shape,
                ).reshape(-1)
            )
        )
        del denominator, numerator
    return FinalMaps(merged=maps[0], halves=maps[1:])
```

### Class3D capture consumer

The capture already needs reconstruction settings for geometry and floor-shell
statistics. Its regularization field now comes from that same owner. The NPZ
keys, `float64` metadata serialization and per-class capture timing are unchanged.

[relax/diagnostics/reconstruction.py](../../relax/diagnostics/reconstruction.py) (line 58):

```python
def write_class_mstep(
    prior: ClassPriorEstimate,
    *,
    numerators,
    denominators,
    half_denominators,
    references,
    settings: ReconstructionSettings,
    output_dir,
    class_index,
    current_size,
    iteration,
    source,
    accumulator_shape,
    full_half_axis,
    frame_scale,
):
    """Capture a class prior and its reconstruction operands in the M-step NPZ."""
    reconstruct_floor_stats_k = regularization_relion._compute_relion_weight_shell_stats(
        denominators[class_index],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        shell_rounding="floor",
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
    )
    import pathlib

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    _preserve_kclass_dump_dtype = os.environ.get("RELAX_KCLASS_DUMP_PRESERVE_DTYPE", "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }
    dump_dtype = None if _preserve_kclass_dump_dtype else np.complex64
    np.savez(
        pathlib.Path(output_dir) / f"recovar_kclass_mstep_it{iteration + 1:03d}_c{class_index + 1:02d}.npz",
        iteration=np.int32(iteration + 1),
        class_index=np.int32(class_index + 1),
        current_size=np.int32(current_size),
        padding_factor=np.int32(settings.padding_factor),
        grid_size=np.int32(settings.grid_size),
        mstep_accumulator_shape=np.asarray(accumulator_shape, dtype=np.int32),
        mstep_full_half_axis=np.int32(full_half_axis),
        tau2_fudge=np.float64(settings.tau2_fudge),
        tau2_frame_scale=np.float64(frame_scale),
        previous_mean=np.asarray(references[0][class_index], dtype=np.complex64),
        previous_mean_half0=np.asarray(references[0][class_index], dtype=np.complex64),
        previous_mean_half1=np.asarray(references[1][class_index], dtype=np.complex64),
        Ft_y_combined=np.asarray(numerators[class_index], dtype=dump_dtype),
        Ft_ctf_0=(
            np.asarray(half_denominators[0][class_index], dtype=dump_dtype)
            if half_denominators[0] is not None
            else np.empty(0, dtype=np.complex64)
        ),
        Ft_ctf_1=(
            np.asarray(half_denominators[1][class_index], dtype=dump_dtype)
            if half_denominators[1] is not None
            else np.empty(0, dtype=np.complex64)
        ),
        Ft_ctf_combined=np.asarray(denominators[class_index], dtype=dump_dtype),
        dump_preserve_dtype=np.int32(int(_preserve_kclass_dump_dtype)),
        tau2_shells=np.asarray(prior.shells, dtype=np.float64),
        tau2_shells_relion=np.asarray(prior.relion_shells, dtype=np.float64),
        tau2_source=np.asarray(source),
        sigma2_shells=np.asarray(
            jnp.where(
                prior.weight_shells["avg_weight_shells"] > 0,
                1.0 / (settings.padding_factor**3 * prior.weight_shells["avg_weight_shells"]),
                0.0,
            ),
            dtype=np.float64,
        ),
        avg_weight_shells=np.asarray(prior.weight_shells["avg_weight_shells"], dtype=np.float64),
        shell_sum=np.asarray(prior.weight_shells["shell_sum"], dtype=np.float64),
        shell_count=np.asarray(prior.weight_shells["shell_count"], dtype=np.float64),
        reconstruct_floor_avg_weight_shells=np.asarray(
            reconstruct_floor_stats_k["avg_weight_shells"],
            dtype=np.float64,
        ),
        reconstruct_floor_shell_count=np.asarray(
            reconstruct_floor_stats_k["shell_count"],
            dtype=np.float64,
        ),
        data_vs_prior=np.asarray(prior.data_vs_prior, dtype=np.float64),
    )
```

### Checks and the next unresolved boundary

58 focused reconstruction/prior/lifecycle/capture cases and 19 affected controller
cases pass. After the final capture-interface migration, all 11 capture cases
pass again. Both runs have zero skips and unchanged source identities; 243
unselected controller cases were not rerun. Import lint and diff checks pass.
A comparison against the hash-verified preceding source checks 64 existing
function bodies after only enumerated settings/interface substitutions and the
retired final scalar binding. Numerical expressions, casts, array operations,
model writes, synchronization, buffer deletion and ordering are not normalized
away. These checks do not qualify GPU quality, peak memory or throughput.
See the [package receipt](final_search_patch_status.md#earlier-reconstruction-settings-cleanup).

A dead-code follow-up removes only the unused command
`_validate_initial_noise_radial`, already uncalled on cached main. Whole-repository
name screening found no retained consumer; the command AST is unchanged apart
from that definition. All 33 affected startup-noise, CLI and import checks pass.
Active noise loading, estimation and validation behavior is unchanged.

The controller still spans 3,037 lines, and command `main()` 1,992. This is an
ownership improvement within the unfinished representative example. Class3D
prior aggregation remains inline: its per-class variance arrays and the last
class result survive after the stacked outputs are created. Extracting a stage
returning only the stacked results would release those arrays earlier. The
pending user question asks whether unused scratch may expire after its last use,
with final memory/performance qualification, or every existing lifetime must
remain identical. No unused-buffer context or extra collector was introduced.
The memory-lifetime constraint comes from the [root contract](../../AGENTS.md).

Finish the representative flow and obtain its design review before freezing a
new candidate for the full K1/exactly-K4 scientific and matched-GPU performance
gates. Existing frozen sources and jobs remain evidence for their own source.

## Numbered sampling policy and Fourier windows

This package gives host sampling decisions clear owners without moving grid
execution or model updates. It uses the existing convergence and iteration
planning modules. Changing perturbation replay precedence or a scoring window
no longer requires editing the numbered controller's reconstruction, noise or
finalization implementation.

### Inputs, consumers and lifetime

| Input or result | Producer and lifetime | Consumers and updates |
| --- | --- | --- |
| Sampling state and completed-iteration flag | Startup/resume and preceding M-step; persistent state | Accuracy updates precede `advance_expectation_sampling`; controller explicitly replaces `state` and records scheduling |
| Native/replay boundary | Controller's replay-expiry decision on each iteration | Earlier convergence admission and angular-transition policy use the same `uses_native_auto_refine` rule; Class3D excludes native transitions |
| Explicit HEALPix schedule | Validated immutable adaptive configuration | Angular policy selects the indexed order instead of native advancement |
| Replay perturbation metadata | This iteration's replay result or sealed boundary | Sealed values bypass seed reconstruction; STAR values retain existing precision/restart helpers |
| Previous perturbation and RNG | Startup/resume, updated once per expectation | Native resolution advances the existing RNG; replay and disabled perturbation leave it untouched; controller installs the scalar |
| Model size and image geometry | Image-size planning/replay and fixed input metadata | `ExpectationWindows` holds only widths; properties adapt the image/model cutoff for existing engine interfaces |
| Optics sizes and pixels | Validated run metadata | Single-shape window planning remaps image support; multi-shape callers leave that remapping with each shape class |
| Incoming coarse order | Captured before replay, accuracy and angular update | `plan_adaptive_image_size` sizes pass 1; current order still selects trial grids |
| `CoarseImageSize` | Conditional global-adaptive planning; two computed scalars | Engine pass-1 width, shape-specific sizing and batch planning consume the result |

The new results contain no device arrays, references, accumulators or copied
model state. The retained canonical Euler rows, current/base translation arrays,
trial grid, projector aliases and phase release sites remain in the controller.
No JIT, transfer, synchronization or reduction boundary is added.

The matrix-only perturbation fallback was unreachable: its tested value was
already produced by `np.asarray`. The surviving Euler trial construction,
rotation/translation regeneration and device coarse-rotation block match the
previous frozen candidate structurally. This deletion does not retire a
reachable matrix conversion API elsewhere.

### Actual calling flow, including accuracy and grid construction

Before this contiguous extract, the controller checks preceding-iteration
convergence, computes image support, captures `coarse_size_healpix_order`, applies
replay/state-swap updates and verifies any frozen boundary. The extract includes
the accuracy producer, angular transition, grid rebuilding, perturbation
application, optics/local windows, direction-prior preparation and adaptive
window decision. After it, the controller prepares projectors and selects half
expectation; reconstruction, noise and completed-iteration convergence follow.
The controller is still unfinished; this is the actual flow to review, not a
proposed replacement loop.

```python
        # Half 1's projector of this iteration's references, built for the
        # expected-accuracy estimate and reused by the scoring projector setup
        # below: RELION computes each class's projector once per iteration.
        shared_projector_half1 = None
        exact_acc_rot_this_iter = None
        exact_acc_trans_this_iter = None
        exact_acc_rot_per_class_this_iter = None
        exact_acc_trans_per_class_this_iter = None
        exact_accuracy_class_counts_this_iter = None
        exact_accuracy_status_this_iter = "skipped_firstiter_cc"
        should_estimate_exact_accuracy = not relion_firstiter_cc_this_iter
        if native_sampling_boundary and should_estimate_exact_accuracy:
            previous_eulers_half1 = halves[0].rotation_eulers
            if expected_accuracy_trial_order is None or previous_eulers_half1 is None:
                exact_accuracy_status_this_iter = "unavailable_inputs"
                state.acc_rot = float("inf")
                state.acc_trans = float("inf")
                logger.warning(
                    "RELION exact expected accuracy unavailable at iteration %d; "
                    "convergence remains fail-closed",
                    iteration + 1,
                )
            else:
                accuracy_class_ids = _expected_accuracy_class_ids(
                    class_assignments[0],
                    k_class_enabled=k_class_enabled,
                    n_units=experiment_datasets[0].n_units,
                )
                if has_previous_iteration and replay_result.relion_projector_state is None:
                    # Iteration 1 may project the initial real references and a
                    # replay may supply a captured projector; both keep their own.
                    shared_projector_size = min(int(current_size), int(grid_size))
                    shared_projector_half1 = ProjectorReuse(
                        references=reference_model.maps[0],
                        current_size=shared_projector_size,
                        image_box_size=grid_size,
                        projector=prepare_scoring_projector(
                            reference_model.maps[0],
                            volume_shape=volume_shape,
                            current_size=shared_projector_size,
                            padding_factor=PROJECTION_PADDING_FACTOR,
                            n_classes=n_classes,
                            dump_label=f"iter{iteration:03d}_half0",
                        ),
                    )
                try:
                    accuracy = expected_accuracy_inputs.estimate(
                        projector_data=None if shared_projector_half1 is None else shared_projector_half1.projector.data,
                        reference_fourier=reference_model.maps[0],
                        best_eulers_deg=previous_eulers_half1,
                        class_ids=accuracy_class_ids,
                        class_weights=class_weights,
                        sigma2_noise_native=noise_model.radial_per_half[0],
                        current_image_size=current_size,
                    )
                    exact_acc_rot_this_iter = float(accuracy.acc_rot)
                    exact_acc_trans_this_iter = float(accuracy.acc_trans_angstrom)
                    exact_acc_rot_per_class_this_iter = np.asarray(
                        accuracy.acc_rot_per_class,
                        dtype=np.float64,
                    ).copy()
                    exact_acc_trans_per_class_this_iter = np.asarray(
                        accuracy.acc_trans_per_class_angstrom,
                        dtype=np.float64,
                    ).copy()
                    exact_accuracy_class_counts_this_iter = np.asarray(
                        accuracy.class_counts,
                        dtype=np.int64,
                    ).copy()
                    expected_accuracy_trial_local_indices = np.asarray(
                        accuracy.trial_local_indices,
                        dtype=np.int64,
                    ).copy()
                    expected_accuracy_trial_particle_ids = np.asarray(
                        accuracy.trial_particle_ids,
                        dtype=np.int64,
                    ).copy()
                    exact_accuracy_status_this_iter = "ok"
                    model_acc_rot_per_class = exact_acc_rot_per_class_this_iter.copy()
                    model_acc_trans_per_class = exact_acc_trans_per_class_this_iter.copy()
                    state.acc_rot = exact_acc_rot_this_iter
                    state.acc_trans = exact_acc_trans_this_iter
                    logger.info(
                        "RELION exact expected accuracy: acc_rot=%.3f deg, acc_trans=%.4f A "
                        "(trials=%d, first_particle_ids=%s)",
                        exact_acc_rot_this_iter,
                        exact_acc_trans_this_iter,
                        int(accuracy.trial_local_indices.size),
                        accuracy.trial_particle_ids[:5].tolist(),
                    )
                except Exception as exc:
                    exact_accuracy_status_this_iter = f"error:{type(exc).__name__}:{exc}"
                    state.acc_rot = float("inf")
                    state.acc_trans = float("inf")
                    logger.warning(
                        "RELION exact expected-accuracy estimation failed at iteration %d; "
                        "convergence remains fail-closed: %s",
                        iteration + 1,
                        exc,
                    )

        # Accuracy and the preceding iteration's stall counters select this
        # expectation's grid; completed-iteration updates remain after M-step.
        state = advance_expectation_sampling(
            state,
            adaptive,
            iteration=iteration,
            has_previous_iteration=has_previous_iteration,
            native_sampling_boundary=native_sampling_boundary,
            n_classes=n_classes,
            log=logger,
        )

        history.record_scheduling(
            current_size,
            state.healpix_order,
            float(current_sigma_offset_angstrom),
            _copy_optional_float_pair(current_sigma_offset_angstrom_per_half),
        )
        scoring_current_size = int(current_size)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===",
            iteration + 1,
            schedule.max_iter,
            scoring_current_size,
            state.healpix_order,
            state.do_local_search,
        )

        # --- Angular step refinement: regenerate rotation grid if needed ---
        # When update_refinement_state incremented healpix_order, we need
        # a new rotation grid at the finer level.
        # IMPORTANT: At order >= 5, the full grid has 2.4M+ rotations which
        # OOMs the GPU.  Instead, keep the order-4 grid as the "base" and
        # rely on local search + oversampling to achieve finer angular steps.
        # The order is still tracked for sigma calculation.
        if state.healpix_order != current_rotation_grid.healpix_order:
            new_order = _exhaustive_grid_order_for_state(state)
            if new_order != current_rotation_grid.healpix_order:
                logger.info(
                    "Regenerating rotation grid: order %d -> %d",
                    current_rotation_grid.healpix_order,
                    new_order,
                )
                current_rotation_grid = sampling.relion_scoring_rotation_grid(
                    new_order, dtype=scoring_dtype,
                    symmetry=symmetry,
                )
            else:
                logger.info(
                    "Angular step refined to order %d (exhaustive grid stays at order %d — local search handles finer sampling)",
                    state.healpix_order,
                    current_rotation_grid.healpix_order,
                )

            # Regenerate translation grid based on updated parameters
            base_translations = sampling._relion_base_translation_grid(
                state.translation_range,
                state.translation_step,
                n_classes=n_classes,
                voxel_size=cryo.voxel_size,
            )
            current_translations = jnp.asarray(base_translations, dtype=scoring_dtype)
            logger.info(
                "New grid: %d rotations, %d translations (range=%.1f, step=%.1f)",
                current_rotation_grid.rotations.shape[0],
                current_translations.shape[0],
                state.translation_range,
                state.translation_step,
            )
        elif perturb_replay_relion_dir is not None and sealed_sampling_state is None:
            # Translation params may have changed under replay without an
            # hp_order bump. Regenerate the translation grid to match RELION.
            _new_t_source = sampling._relion_base_translation_grid(
                state.translation_range,
                state.translation_step,
                n_classes=n_classes,
                voxel_size=cryo.voxel_size,
            )
            _new_t = jnp.asarray(_new_t_source, dtype=scoring_dtype)
            if _new_t.shape != base_translations.shape or not jnp.allclose(
                _new_t,
                np.asarray(base_translations, dtype=scoring_dtype),
            ):
                current_translations = _new_t
                base_translations = _new_t_source
                logger.info(
                    "Replay: regenerated translation grid: %d translations (range=%.2f px, step=%.2f px)",
                    current_translations.shape[0],
                    state.translation_range,
                    state.translation_step,
                )

        # --- Local angular search bookkeeping ---
        # Once RELION enters local search, each image should search around its
        # own previous orientation on the true current HEALPix order. Use the
        # exact rotations selected in the previous iteration, not the nearest
        # snapped grid indices.
        effective_rotations = current_rotation_grid.rotations
        effective_rotation_eulers = np.asarray(
            current_rotation_grid.rotation_eulers,
            dtype=scoring_dtype,
        )
        effective_mstep_rotations = None
        adaptive_pass1_rotations = None
        direction_log_priors = [None, None]
        use_local = state.do_local_search and all(half.rotation_eulers is not None for half in halves)
        if use_local and k_class_enabled:
            # Class3D keeps global searches: RELION switches to local searches from the
            # HEALPix order only under auto-refine (ml_optimiser.cpp:2541-2565, 3936-3938).
            raise RuntimeError("K>1 (Class3D) reached local angular searches; RELION never does")
        adaptive_pass1_source_eulers = np.asarray(effective_rotation_eulers, dtype=np.float64)
        # --- Apply RELION SamplingPerturbation to the trial grid for this iter ---
        # healpix_sampling.cpp:1909-1934 (rotations) + 1810-1820 (translations)
        # Perturbation is a rigid rotation of SO(3): A := A @ R_perturb applied
        # AFTER oversampling. At adaptive_oversampling=0 (os0 RELION runs),
        # the coarse grid IS the trial grid so we apply directly here.
        random_perturbation = resolve_numbered_perturbation(
            random_perturbation,
            parity,
            iteration=iteration,
            init_relion_iteration=init_relion_iteration,
            replay_metadata=_replay_meta,
            replay_dir=perturb_replay_relion_dir,
            rng=perturb_rng,
            log=logger,
        )
        if _replay_meta is not None or parity.perturb_factor > 0:
            # Use RELION's actual hp_order when replaying (recovar's current
            # grid order may be capped at MAX_FULL_GRID_ORDER=4 for memory).
            _angsamp_order = int(_replay_meta["healpix_order"]) if _replay_meta is not None else current_rotation_grid.healpix_order
            angsamp_deg = relion_angular_sampling_deg(_angsamp_order, adaptive_oversampling=0)
            trial_grid = sampling._perturbed_trial_grid(
                rotation_eulers=effective_rotation_eulers,
                mstep_source_eulers=sampling._relion_mstep_source_eulers(
                    effective_rotation_eulers,
                    _angsamp_order,
                    use_grid_eulers=sealed_sampling_state is not None,
                    symmetry=symmetry,
                ),
                base_translations=base_translations,
                translation_step=float(state.translation_step),
                random_perturbation=random_perturbation,
                angular_sampling_deg=angsamp_deg,
                dtype=scoring_dtype,
            )
            effective_rotations = trial_grid.rotations
            effective_rotation_eulers = trial_grid.rotation_eulers
            effective_mstep_rotations = trial_grid.mstep_rotations
            current_translations = trial_grid.translations
        # RELION's coarse device geometry also applies at OS0. Keep this
        # separate from host fine/M-step geometry; see docs/math/zero_coarse_geometry.md.
        if not use_local and (
            int(state.adaptive_oversampling) > 0
            or (
                int(state.adaptive_oversampling) == 0
                and n_classes == 1
                and firstiter_score_mode_this_iter == "gaussian"
                and not firstiter_winner_take_all_this_iter
                and not scoring_policy.DENSE_PRECISION.use_float64_scoring
            )
        ):
            adaptive_pass1_order = (
                int(_replay_meta["healpix_order"])
                if _replay_meta is not None
                else int(current_rotation_grid.healpix_order)
            )
            adaptive_pass1_use_float64 = bool(scoring_policy.DENSE_PRECISION.use_float64_scoring)
            adaptive_pass1_rotations = _relion_adaptive_pass1_rotations(
                adaptive_pass1_source_eulers,
                random_perturbation if (_replay_meta is not None or parity.perturb_factor > 0) else 0.0,
                relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=0),
                use_float64=adaptive_pass1_use_float64,
            )
            if adaptive_pass1_rotations is not None:
                logger.info(
                    "RELION adaptive pass 1: using %s-built coarse scorer rotations; "
                    "fine/M-step rotations remain host-generated",
                    "double-precision CUDA" if adaptive_pass1_use_float64 else "CUDA",
                )
        # First-iteration CC scores the full translation grid before choosing
        # its single winning pose (ml_optimiser.cpp:9181-9207).
        expectation_windows = plan_expectation_windows(
            scoring_current_size,
            image_geometry,
            model_pixel_size=model_pixel_size,
            optics_image_sizes=None if multi_shape_halves else optics_image_sizes,
            optics_pixel_sizes=optics_pixel_sizes,
            log=logger,
        )
        model_current_size_for_engine = expectation_windows.model_window_size
        image_current_size = expectation_windows.image_size
        cs_for_engine = expectation_windows.image_window_size
        sigma_rot, sigma_psi = relion_local_search_sigmas(
            state.sigma_rot,
            state.sigma_psi,
            use_local=use_local,
            healpix_order=state.healpix_order,
            adaptive_oversampling=state.adaptive_oversampling,
        )

        # Angular step behind this iteration's pass-1 coarse size, when RELION's
        # adaptive formula sets it (shape classes recompute their own from it).
        coarse_size_step_deg = None
        if use_local:
            local_sampling = prepare_numbered_local_sampling(
                LocalSearchSettings(
                    healpix_order=state.healpix_order + state.adaptive_oversampling,
                    oversampling_order=int(state.adaptive_oversampling) if state.adaptive_oversampling > 0 else 0,
                    sigma_rot=sigma_rot,
                    sigma_psi=sigma_psi,
                    symmetry=symmetry,
                ),
                sampling.TrialGrid(
                    rotations=effective_rotations,
                    rotation_eulers=effective_rotation_eulers,
                    mstep_rotations=effective_mstep_rotations,
                    translations=current_translations,
                ),
                base_translations=base_translations,
                image_window_size=cs_for_engine,
                model_support_size=model_current_size_for_engine,
                base_healpix_order=current_rotation_grid.healpix_order,
                coarse_size_healpix_order=coarse_size_healpix_order,
                perturbation=random_perturbation,
                model_pixel_size=model_pixel_size,
                original_model_size=grid_size,
                optics_image_sizes=optics_image_sizes,
                optics_pixel_sizes=optics_pixel_sizes,
                particle_diameter_angstrom=particle_diameter_ang,
                log=logger,
            )
            coarse_size_step_deg = local_sampling.coarse_angular_step_deg
        else:
            local_sampling = None
        direction_prior_healpix_order = _direction_prior_healpix_order_for_scoring(
            use_local=use_local,
            current_healpix_order=current_rotation_grid.healpix_order,
            state_healpix_order=state.healpix_order,
            adaptive_oversampling=state.adaptive_oversampling,
            local_search_order=local_sampling.search.healpix_order if use_local else None,
        )
        coarse_rotation_ids_for_scoring = (
            _sealed_sampling_rotation_ids(sealed_sampling_state)
            if sealed_sampling_state is not None and not use_local
            else None
        )
        if (
            coarse_rotation_ids_for_scoring is not None
            and coarse_rotation_ids_for_scoring.shape != (int(effective_rotations.shape[0]),)
        ):
            raise RuntimeError(
                "sealed captured rotation IDs do not match the directly materialized scorer grid"
            )

        for _half_idx in range(2):
            half_direction_priors = relion_direction_log_priors_for_half(
                use_local=use_local,
                scoring_healpix_order=direction_prior_healpix_order,
                n_classes=n_classes,
                priors=direction_priors[_half_idx],
                sealed_sampling_state=sealed_sampling_state,
                dtype=scoring_dtype,
                log=logger,
                half_index=_half_idx,
                symmetry=symmetry,
            )
            direction_log_priors[_half_idx] = half_direction_priors

        # --- Run E+M on each half-set ---
        # Two modes: single-pass (adaptive_oversampling=0) or two-pass
        # coarse/fine (adaptive_oversampling>=1).
        iter_sig_counts = None
        iter_sig_count_parts: list[np.ndarray] = []
        iter_recorded_sig_counts = None
        iter_recorded_sig_count_parts: list[np.ndarray] = []
        iter_significant_counts_per_half = [None, None]
        use_adaptive = _should_use_adaptive_search(
            adaptive_oversampling=state.adaptive_oversampling,
            use_local=use_local,
            n_rotations=effective_rotations.shape[0],
            symmetry=symmetry,
        )
        # Track the rotation grids used for pose extraction.
        # When adaptive oversampling is active, ha_k indices refer to the
        # oversampled grid (from pass 2), not effective_rotations.
        per_half = PerHalfOutputs()
        hard_assignments = per_half.hard_assignments
        class_assignments = per_half.class_assignments
        class_posterior_per_half = per_half.class_posterior
        class_full_posterior_per_half = per_half.class_full_posterior
        max_posterior_per_half = per_half.max_posterior
        rotation_posterior_per_half = per_half.rotation_posterior
        class_rotation_posterior_per_half = per_half.class_rotation_posterior
        pose_rotations = per_half.pose_rotations  # rotations to use with ha for poses
        pose_rotation_eulers = per_half.pose_rotation_eulers
        best_pose_rotations = per_half.best_pose_rotations
        best_pose_rotation_eulers = per_half.best_pose_rotation_eulers
        best_pose_translations = per_half.best_pose_translations
        translation_search_bases = per_half.translation_search_bases
        # Coarse-grid assignments for local search tracking (always indexed
        # into effective_rotations, even when adaptive oversampling is used).
        coarse_ha = per_half.coarse_ha
        if use_adaptive:
            # --- TWO-PASS ADAPTIVE OVERSAMPLING (RELION parity) ---
            # Pass 1: coarse E-step at reduced resolution to find
            #         significant orientations.
            # Pass 2: oversampled E+M at full current_size for significant
            #         orientations only.

            coarse_image_plan = plan_adaptive_image_size(
                coarse_size_healpix_order,
                expectation_windows,
                image_geometry,
                particle_diameter_angstrom=particle_diameter_ang,
                optics_image_sizes=optics_image_sizes,
                optics_pixel_sizes=optics_pixel_sizes,
                sealed_sampling_state=sealed_sampling_state,
                log=logger,
            )
            coarse_size = coarse_image_plan.size
            coarse_size_step_deg = coarse_image_plan.angular_step_deg
            coarse_cs = coarse_size if coarse_size < grid_size else None

            logger.info(
                "Adaptive oversampling: pass 1 at coarse_size=%s, "
                "pass 2 at current_size=%s (oversampling=%d, particle_diameter=%s)",
                coarse_cs,
                cs_for_engine,
                state.adaptive_oversampling,
                (f"{float(particle_diameter_ang):.1f} A" if particle_diameter_ang is not None else "box_size"),
            )

```

### Angular-transition owner

```python
def uses_native_auto_refine(*, native_sampling_boundary: bool, n_classes: int) -> bool:
    """Class3D reports accuracy but never advances auto-refine or converges."""
    return native_sampling_boundary and n_classes == 1


def advance_expectation_sampling(
    state: RefinementState,
    adaptive: AdaptiveOptions,
    *,
    iteration: int,
    has_previous_iteration: bool,
    native_sampling_boundary: bool,
    n_classes: int,
    log: logging.Logger,
) -> RefinementState:
    """Choose the next angular grid after accuracy, before expectation.

    Native auto-refine uses the preceding iteration's stall counters. An
    explicit HEALPix schedule takes precedence, including on the first iteration.
    See ``docs/math/relion_refinement_algorithm.md#iteration-convergence-policy``.
    """
    if adaptive.relion_healpix_orders is not None:
        target_order = int(adaptive.relion_healpix_orders[iteration])
        state = _apply_relion_healpix_order_oracle(
            state, target_order, iteration_number=iteration + 1,
        )
        log.info(
            "HEALPix-order oracle: iteration %d using healpix_order=%d",
            iteration + 1, target_order,
        )
    elif has_previous_iteration and uses_native_auto_refine(
        native_sampling_boundary=native_sampling_boundary, n_classes=n_classes,
    ):
        state = update_angular_sampling(state)
    return state
```

### Perturbation and Fourier-window owners

```python
def resolve_numbered_perturbation(
    previous_perturbation: float,
    parity: RelionParityOptions,
    *,
    iteration: int,
    init_relion_iteration: int,
    replay_metadata,
    replay_dir: str | None,
    rng,
    log: logging.Logger,
) -> float:
    """Resolve sealed, STAR-replayed or native perturbation, in that order.

    Only native sampling advances the run RNG. Replay may instead reconstruct
    the seeded sequence from a restart iteration. See
    ``docs/math/relion_refinement_algorithm.md#2-sampling-grids-and-units``.
    """
    if replay_metadata is not None:
        if replay_metadata.get("sealed_v3", False):
            perturbation = float(replay_metadata["random_perturbation"])
            source = "sealed_frozen_boundary_v3"
        else:
            relion_iteration = init_relion_iteration + iteration + 1
            restart_iteration = _perturbation_restart_state_iteration(
                parity.perturb_replay_restart_state_iterations, relion_iteration,
            )
            perturbation, source = _resolve_replay_random_perturbation(
                star_value=float(replay_metadata["random_perturbation"]),
                perturbation_factor=float(replay_metadata["perturbation_factor"]),
                relion_iteration=relion_iteration,
                replay_dir=str(replay_dir),
                replay_prefix=parity.perturb_replay_relion_prefix,
                explicit_seed=parity.perturb_seed,
                precision_mode=str(parity.perturb_replay_precision),
                restart_state_iteration=restart_iteration,
            )
        log.info(
            "Perturbation replay: iter=%d rp=%+.12g pf=%.3f relion_hp_order=%d source=%s",
            iteration + 1, perturbation,
            float(replay_metadata["perturbation_factor"]),
            int(replay_metadata["healpix_order"]), source,
        )
        return perturbation
    if not parity.perturb_factor > 0:
        return previous_perturbation
    relion_iteration = init_relion_iteration + iteration + 1
    perturbation, seed = sampling._advance_relion_perturbation(
        previous_perturbation,
        perturb_factor=parity.perturb_factor,
        perturb_seed=parity.perturb_seed,
        relion_iteration=relion_iteration,
        rng=rng,
    )
    if seed is not None:
        log.info(
            "Perturbation advance: iter=%d relion_iter=%d seed=%d rp=%+.5f",
            iteration + 1, relion_iteration, seed, perturbation,
        )
    else:
        log.info("Perturbation advance: iter=%d rp=%+.5f", iteration + 1, perturbation)
    return perturbation


@dataclass(frozen=True, kw_only=True)
class ExpectationWindows:
    """Model and particle Fourier widths for one numbered expectation."""

    model_size: int
    image_size: int
    image_box_size: int

    @property
    def image_window_size(self) -> int | None:
        return self.image_size if self.image_size < self.image_box_size else None

    @property
    def model_window_size(self) -> int | None:
        # Engines otherwise reuse the image cutoff for a None model window.
        if self.model_size < self.image_box_size or self.model_size != self.image_size:
            return self.model_size
        return None


def plan_expectation_windows(
    current_size: int,
    image_geometry: ImageGeometry,
    *,
    model_pixel_size: float,
    optics_image_sizes,
    optics_pixel_sizes,
    log: logging.Logger,
) -> ExpectationWindows:
    """Remap single-shape optics while keeping model support independent.

    Shape-class callers omit optics image sizes: each class remaps its own
    support. See ``docs/math/relion_refinement_algorithm.md#5-accumulation-reconstruction-and-parameter-updates``.
    """
    image_size = current_size
    if optics_image_sizes is not None:
        remapped = relion_optics_image_current_sizes(
            current_size,
            model_ori_size=image_geometry.box_size,
            model_pixel_size=model_pixel_size,
            optics_image_sizes=optics_image_sizes,
            optics_pixel_sizes=optics_pixel_sizes,
        )
        unique_sizes = np.unique(remapped)
        if unique_sizes.size != 1:
            raise NotImplementedError(
                "K=1 parity currently requires all optics groups to share one remapped "
                f"image current size; got {remapped.tolist()}",
            )
        image_size = int(unique_sizes[0])
    if image_size != current_size:
        log.info(
            "RELION optics current-size remap: model_current_size=%d "
            "image_current_size=%d model_pixel_size=%.9g",
            current_size, image_size, model_pixel_size,
        )
    return ExpectationWindows(
        model_size=current_size,
        image_size=image_size,
        image_box_size=image_geometry.box_size,
    )


@dataclass(frozen=True, kw_only=True)
class CoarseImageSize:
    """Pass-1 width and the pre-update angular step used to size it."""

    size: int
    angular_step_deg: float


def plan_adaptive_image_size(
    pre_update_healpix_order: int,
    windows: ExpectationWindows,
    image_geometry: ImageGeometry,
    *,
    particle_diameter_angstrom: float | None,
    optics_image_sizes,
    optics_pixel_sizes,
    sealed_sampling_state,
    log: logging.Logger,
) -> CoarseImageSize:
    """Size pass 1 from incoming sampling, then admit an exact sealed width.

    The fine grid can already have advanced to another order. See
    ``docs/math/relion_refinement_algorithm.md#5-accumulation-reconstruction-and-parameter-updates``.
    """
    angular_step_deg = healpix_angular_step(pre_update_healpix_order)
    coarse_size = compute_coarse_image_size(
        angular_step_deg,
        float(optics_pixel_sizes[0]) if optics_pixel_sizes is not None else image_geometry.pixel_size_angstrom,
        int(optics_image_sizes[0]) if optics_image_sizes is not None else image_geometry.box_size,
        particle_diameter=particle_diameter_angstrom,
    )
    coarse_size = clamp_relion_coarse_image_size(
        coarse_size,
        windows.image_size if windows.image_window_size is not None else None,
        image_geometry.box_size,
    )
    if sealed_sampling_state is not None:
        coarse_size = int(sealed_sampling_state["coarse_size"])
        if coarse_size > windows.model_size:
            raise ValueError(
                "sealed sampling coarse_size exceeds active current_size: "
                f"coarse={coarse_size} current={windows.model_size}"
            )
        log.info("Frozen-boundary v3 directly owns adaptive pass-1 coarse_size=%d", coarse_size)
    return CoarseImageSize(size=coarse_size, angular_step_deg=angular_step_deg)
```

### Verification and remaining work

71 focused policy/caller-window cases, 87 affected controller cases and 108 CPU
guards pass. After removing the unreachable fallback, 11 targeted trial-grid,
caller and import checks pass on unchanged source during the receipt. Structural
comparison verifies the surviving numerical grid blocks and the controller's
phase order against the previous frozen source. Four initial new-fixture errors
and one subsequent incomplete-counter fixture were repaired without changing
production math or tolerance. See the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_numbered_sampling_policy_20261002T173238Z/HANDOFF.json)
for exact source identities, commands and logs.

The numbered controller remains 3,052 lines and the command 1,992 lines. Sampling
phase construction, the expectation publication closure, Class3D scratch
lifetime and finalization inputs remain design work. Finish the representative
code and review it before freezing a new production candidate. No new production
run, GPU qualification, speed result, merge or publication is claimed. Earlier
frozen candidates and job packets remain evidence for their original source.

## Controller scope and remaining coupling

The current candidate's command `main()` is 1,992 lines in a 2,762-line
`full_refinement.py`. `refine_single_volume()` is 3,052 lines; its numbered loop
alone spans 2,204 lines. These remain unfinished controllers. A smaller file
does not establish that a responsibility has a usable boundary.

The original shared workspace still has the earlier 4,580-line command file
and 2,569-line `main()`. Current edits live in the isolated integration checkout
identified in the package handoff; earlier source snapshots remain unchanged.

| Actual flow | Existing ownership | Implementation still embedded in orchestration |
| --- | --- | --- |
| Command startup | Command options, particle loading/row layout, geometry, noise and iteration-zero model replay owners | Reference/class startup, replay and frozen-boundary adaptations |
| Numbered admission and sampling | Planning and local-sampling operations | Preceding-iteration convergence decision, replay expiry, grid rebuilding and phase construction |
| Half expectation | `expectation.py` produces complete score/pose results | Serial/overlap selection, 123-line publication closure and explicit model updates |
| Reconstruction and priors | Mean/reconstruction operations and independent prior estimators | 160-line Class3D aggregation branch, retained class scratch and reference replacement |
| Parameter/convergence updates | Noise, normalization, pose and convergence owners | State installation, history, diagnostics and checkpoint scheduling |
| Finalization and publication | Final sampling, scoring/reconstruction controller and file owners | Final admission/replay handoff and command reporting |

The next substantial boundaries should address those embedded implementations.
The controller must continue to show the execution order, important mode choices,
state writes and transition to finalization. Convergence at the top of iteration
N uses the completed statistics from N-1; moving that check would change the
algorithm. Each proposed extraction needs a producer/consumer and buffer-lifetime
audit. Collecting every local into a context object would leave the coupling.

Current editable source incorporates cached main `a9e0668`. The fast-forward
preserved every refactor edit; six focused incoming crop/window cases and 108
EM CPU guards pass. See the [integration handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T154930Z/HANDOFF.json).
The projector implementation below is unchanged. Its [earlier frozen package](/scratch/gpfs/GILLES/mg6942/tmp/relax_projector_reuse_owner_20261002T134633Z/HANDOFF.json)
passed 79 affected CPU cases and 96 old/new preparation-stage comparisons.
Its prepared GPU jobs and admitted real-data packet remain source-specific to
`7412fd8`. Incoming resident-projection code requires fresh GPU qualification;
latest quality/performance and the completed controller review remain open.

## Projector preparation and accuracy reuse

The existing projector owner now makes the reuse decision as well as preparing
the slabs. A cache/reuse change can be reviewed with this operation and its
accuracy producer, without opening reconstruction, noise updates, convergence
internals or finalization. The controller still exposes when preparation runs,
which replay/mode is active, half identity and explicit installation.

### Producer-owned identity and lifetime

`PreparedProjector` is unchanged. The former positional reuse tuple now has named
fields at its actual producer; the binding is used by both accuracy and scoring.
It borrows the same reference array for the same lifetime. Its additional scalar
`image_box_size` records fixed image support: the existing reuse comparison treats
`current_size=None` as the full image width, whereas a cold transform treats it as
the volume width. These can differ. Retain that distinction rather than equating
output volume geometry with input images.

```python
@dataclass(frozen=True, kw_only=True)
class PreparedProjector:
    """Projection slabs, Fourier cutoff and power spectra from one transform."""

    data: object
    r_max: int
    power_spectrum: object | None = None


@dataclass(frozen=True, kw_only=True)
class ProjectorReuse:
    """An accuracy projector tied to its reference identity and image support."""

    references: object
    current_size: int
    image_box_size: int
    projector: PreparedProjector
```

The following is the actual producer inside the native expected-accuracy branch,
after replay/state-swap handling. Initial real references and captured replay
projectors retain their existing paths:

```python
                if has_previous_iteration and replay_result.relion_projector_state is None:
                    # Iteration 1 may project the initial real references and a
                    # replay may supply a captured projector; both keep their own.
                    shared_projector_size = min(int(current_size), int(grid_size))
                    shared_projector_half1 = ProjectorReuse(
                        references=reference_model.maps[0],
                        current_size=shared_projector_size,
                        image_box_size=grid_size,
                        projector=prepare_scoring_projector(
                            reference_model.maps[0],
                            volume_shape=volume_shape,
                            current_size=shared_projector_size,
                            padding_factor=PROJECTION_PADDING_FACTOR,
                            n_classes=n_classes,
                            dump_label=f"iter{iteration:03d}_half0",
                        ),
                    )
```

The actual accuracy consumer, before accuracy results are installed, is:

```python
                    accuracy = expected_accuracy_inputs.estimate(
                        projector_data=None if shared_projector_half1 is None else shared_projector_half1.projector.data,
                        reference_fourier=reference_model.maps[0],
                        best_eulers_deg=previous_eulers_half1,
                        class_ids=accuracy_class_ids,
                        class_weights=class_weights,
                        sigma2_noise_native=noise_model.radial_per_half[0],
                        current_image_size=current_size,
                    )
```

### Complete numbered preparation caller

The list reset still occurs before replacement builds. Keep the explicit
`projector = ...` assignment: Python retains the previous last-projector alias
while evaluating the right-hand side and releases it at assignment. A function
returning the whole replacement list would shift both boundaries. Iteration is
over the persistent `HalfSet` owners; their indexes select reference and output
slots. Captured-state validation remains one atomic operation.

```python
        projectors = [None, None]
        captured_projector_state = replay_result.relion_projector_state
        selected_gemm_global = adaptive.coarse_engine in {"gemm_hybrid", "gemm_dense"}
        if captured_projector_state is not None and not (use_local or use_adaptive or selected_gemm_global):
            raise RuntimeError(
                "captured RELION Projector::data was supplied but this iteration has no projector scoring path"
            )
        if use_local or use_adaptive or selected_gemm_global:
            projector_t0 = time.time()
            if captured_projector_state is not None:
                projectors = _validate_captured_relion_projector_for_iteration(
                    captured_projector_state,
                    current_size=model_current_size_for_engine,
                    volume_shape=volume_shape,
                    padding_factor=PROJECTION_PADDING_FACTOR,
                    n_classes=n_classes,
                )
                logger.info(
                    "RELION mode: using captured exact Projector::data at current_size=%s "
                    "r_max=%s manifest=%s",
                    model_current_size_for_engine,
                    None if projectors[0] is None else projectors[0].r_max,
                    captured_projector_state.source_manifest_sha256,
                )
            else:
                for half in halves:
                    if half.dataset.n_units == 0:
                        logger.info(
                            "RELION mode: skipping Projector::data build for empty half-%d dataset",
                            half.index + 1,
                        )
                        continue
                    projector = prepare_scoring_projector(
                        reference_model.maps[half.index],
                        volume_shape=volume_shape,
                        current_size=model_current_size_for_engine,
                        padding_factor=PROJECTION_PADDING_FACTOR,
                        n_classes=n_classes,
                        reusable=shared_projector_half1 if half.index == 0 else None,
                        real_references=(
                            initial_real_references_by_half[half.index]
                            if iteration == 0
                            else None
                        ),
                        dump_label=f"iter{iteration:03d}_half{half.index}",
                    )
                    projectors[half.index] = projector
                logger.info(
                    # The slab dtype decides whether pass-2 projection runs on
                    # the native texture projector or the vmapped JAX fallback
                    # (_relion_projector_texture_enabled requires complex64),
                    # so record it rather than leaving the path implicit.
                    "RELION mode: built exact Projector::data for scoring at current_size=%s r_max=%s "
                    "dtype=%s in %.2fs",
                    model_current_size_for_engine,
                    None if projectors[0] is None else projectors[0].r_max,
                    None if projectors[0] is None else projectors[0].data.dtype,
                    time.time() - projector_t0,
                )
```

### Complete preparation operation

Reuse is resolved before host conversion, lazy imports or cache/dump access.
Existing cold transformation, layout, casts, cache version and serialization are
preserved. In-memory reuse still skips cache/dump work as the preceding controller
branch did. The captured validator remains alongside this operation and is unchanged.
Finalization calls the same builder with no reuse binding; no final ordering changes.

```python
def prepare_scoring_projector(
    references,
    *,
    volume_shape,
    current_size: int | None,
    padding_factor: int,
    n_classes: int,
    real_references=None,
    dump_label: str | None = None,
    reusable: ProjectorReuse | None = None,
) -> PreparedProjector:
    """Prepare RELION ``Projector::data`` slabs from current Fourier references.

    The returned ``power_spectrum`` (float64
    ``[K, ori_size // 2 + 1]``) is each class's corrected power spectrum from
    the same transform, as ``computeFourierTransformMap`` returns it; Class3D
    derives its tau2 from it. It is ``None`` when the slabs come from a
    projector cache entry written without it.

    The slabs come from the device projector setup
    (:func:`relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps_and_power`);
    RELION's own transform is a test oracle only.
    """

    if reusable is not None:
        reuse_size = reusable.image_box_size if current_size is None else int(current_size)
        if reusable.references is references and reusable.current_size == reuse_size:
            return reusable.projector

    from recovar.core import fourier_transform_utils as ftu

    from relax.relion.relion_projector_setup import reference_to_relion_projector_half_maps_and_power

    refs_ft = np.asarray(references)
    if int(n_classes) == 1 and refs_ft.ndim == 1:
        refs_ft = refs_ft[None, :]
    if refs_ft.ndim != 2 or int(refs_ft.shape[0]) != int(n_classes):
        raise ValueError(
            "references must be a flat reference or a per-class reference array; "
            f"got shape {refs_ft.shape} for n_classes={n_classes}",
        )
    refs_real_override = None
    if real_references is not None:
        refs_real_override = np.asarray(real_references, dtype=np.float64)
        expected_shape = (int(n_classes),) + tuple(int(value) for value in volume_shape)
        if refs_real_override.shape != expected_shape:
            raise ValueError(
                "real_references must have one real-space volume per class; "
                f"got {refs_real_override.shape}, expected {expected_shape}",
            )
    resolved_current_size = int(current_size) if current_size is not None else int(volume_shape[0])
    cache_dir = os.environ.get("RELAX_RELION_PROJECTOR_CACHE_DIR", "").strip()
    cache_path = None
    if cache_dir:
        refs_for_hash = np.ascontiguousarray(
            refs_ft if refs_real_override is None else refs_real_override
        )
        hasher = hashlib.sha256()
        # v3: one transform (the device setup); v2 keys named the backend, v1
        # entries could mix backends. Older entries are unreachable, which costs
        # one rebuild and is the safe direction.
        hasher.update(b"recovar-relion-projector-cache-v3")
        hasher.update(b"fourier-reference" if refs_real_override is None else b"real-reference")
        hasher.update(str(refs_for_hash.dtype).encode("utf-8"))
        hasher.update(np.asarray(refs_for_hash.shape, dtype=np.int64).tobytes())
        hasher.update(np.asarray(volume_shape, dtype=np.int64).tobytes())
        cache_params = np.asarray(
            [resolved_current_size, int(padding_factor), int(n_classes)],
            dtype=np.int64,
        )
        hasher.update(cache_params.tobytes())
        hasher.update(refs_for_hash.view(np.uint8))
        cache_path = os.path.join(cache_dir, f"projector_{hasher.hexdigest()[:24]}.npz")
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    projector_half = np.asarray(cached["projector_half"])
                    projector_r_max = int(np.asarray(cached["projector_r_max"]))
                    projector_power = (
                        np.asarray(cached["projector_power"]) if "projector_power" in cached.files else None
                    )
                    if (
                        int(np.asarray(cached["current_size"])) != resolved_current_size
                        or int(np.asarray(cached["padding_factor"])) != int(padding_factor)
                        or int(np.asarray(cached["n_classes"])) != int(n_classes)
                        or tuple(np.asarray(cached["volume_shape"], dtype=np.int64).tolist()) != tuple(volume_shape)
                    ):
                        raise ValueError("metadata mismatch")
                logger.info("RELION mode: loaded cached Projector::data from %s", cache_path)
                return PreparedProjector(data=projector_half, r_max=projector_r_max, power_spectrum=projector_power)
            except Exception as exc:
                logger.warning("Ignoring unreadable RELION projector cache %s: %s", cache_path, exc)
    if refs_real_override is None:
        refs_real = []
        for class_index in range(int(n_classes)):
            ref_ft = jnp.asarray(refs_ft[class_index]).reshape(volume_shape)
            refs_real.append(np.asarray(ftu.get_idft3(ref_ft)).real)
        refs_real = np.asarray(refs_real, dtype=np.float64)
    else:
        refs_real = refs_real_override
    projector_half, projector_power, projector_r_max = reference_to_relion_projector_half_maps_and_power(
        refs_real,
        current_size=resolved_current_size,
        padding_factor=int(padding_factor),
        # Refinement consumes complex128, which its own log line reports; the
        # setup narrows to complex64 (the InitialModel consumer) unless told otherwise.
        projector_data_dtype="complex128",
    )
    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(os.path.join(cache_dir, "SAFE_TO_DELETE"), "a", encoding="utf-8"):
                pass
            tmp_path = f"{cache_path}.{os.getpid()}.tmp.npz"
            np.savez(
                tmp_path,
                projector_half=np.asarray(projector_half),
                projector_r_max=np.int64(projector_r_max),
                projector_power=np.asarray(projector_power),
                current_size=np.int64(resolved_current_size),
                padding_factor=np.int64(padding_factor),
                volume_shape=np.asarray(volume_shape, dtype=np.int64),
                n_classes=np.int64(n_classes),
            )
            os.replace(tmp_path, cache_path)
            logger.info("RELION mode: saved Projector::data cache to %s", cache_path)
        except Exception as exc:
            logger.warning("Could not write RELION projector cache %s: %s", cache_path, exc)
    dump_dir = os.environ.get("RELAX_RELION_PROJECTOR_DUMP_DIR")
    if dump_dir:
        label = dump_label or "projector"
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        os.makedirs(dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(dump_dir, f"{safe_label}_relion_projector_half.npz"),
            projector_half=np.asarray(projector_half),
            reference_real=np.asarray(refs_real),
            projector_r_max=np.int64(projector_r_max),
            current_size=np.int64(resolved_current_size),
            padding_factor=np.int64(padding_factor),
            volume_shape=np.asarray(volume_shape, dtype=np.int64),
            n_classes=np.int64(n_classes),
        )
    return PreparedProjector(data=projector_half, r_max=projector_r_max, power_spectrum=projector_power)
```

### Checks and remaining review

79 affected CPU cases and 108 guards passed. Two new full-controller tests check
the original release boundaries/reuse identity and empty-half handling. The independent
old/new stage audit executes 96 K1/K4 combinations: full/explicit windows,
initial/continued runs, nonempty/empty second halves and absent/hit/window-miss/
identity-miss reuse. It checks transform calls, layouts, precision, returned values
and borrowed identity. These checks establish this structural boundary; GPU smoke,
production scientific quality, timing and peak RSS remain open.

The complete caller intentionally retains scientific admission and updates. This
package does not complete the 3,184-line numerical controller. Class3D aggregation
and broader sampling preparation still have separate lifecycle/RNG questions.
Human review of the complete producer and consumer precedes wider automation.

## Paired-half expectation execution ownership

Expectation owns the execution machinery below. It borrows the controller's
existing callback until both threads have joined; the callback still scores and
publishes each half into its established disjoint slots. The guard reads the same
environment at invocation, errors remain in half order, and the device budget share
is restored in the same `finally` block. Private helpers remain with the operation.
No new module, result class, forwarding wrapper or retained scratch is introduced.

The controller keeps its mode decision and execution order:

```python
        _overlap_active = _half_overlap_active(
            options.overlap.overlap_halves,
            diagnostic_half_indices=diagnostic_half_indices,
            log=logger,
        )
        if _overlap_active:
            _run_halves_overlapped(_run_half_estep, diagnostic_half_indices)
            k = diagnostic_half_indices[-1]
        else:
            for k in diagnostic_half_indices:
                _run_half_estep(k)
```

The complete execution operation in `expectation.py` is:

```python
_BPREF_DUMP_ENV_VARS = (
    "RELAX_BPREF_MEMBERSHIP_DUMP_DIR",
    "RELAX_BPREF_CONTRIBUTION_DUMP_DIR",
    "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
)


def _half_overlap_active(requested: bool, *, diagnostic_half_indices, log) -> bool:
    """Decide whether the two halves' E-steps may run concurrently.

    Refuses rather than degrades. Each guard is a place where running the two
    halves at once would change what is observed, not merely when:

    * a subset of halves is being scored, so there is nothing to overlap;
    * a BPref dump is armed. The dump context is process-global and is set per
      half at the top of the half's work, so two halves in flight would write
      each other's context. The dump is a diagnostic, so the overlap yields.
    """

    if not requested:
        return False
    if tuple(diagnostic_half_indices) != (0, 1):
        log.info("Half overlap off: scoring halves %s, not both", tuple(diagnostic_half_indices))
        return False
    armed = [name for name in _BPREF_DUMP_ENV_VARS if os.environ.get(name, "").strip()]
    if armed:
        log.info("Half overlap off: a BPref dump is armed (%s)", ", ".join(armed))
        return False
    log.info("Half overlap ON: the two halves' E-steps run in one thread each")
    return True


def _run_halves_overlapped(run_half, diagnostic_half_indices) -> None:
    """Run each half's E-step in its own thread and re-raise in half order.

    Kernels still serialise on JAX's single compute stream, so the device order
    within a half is unchanged and the two halves' kernels cannot interleave
    mid-kernel. What overlaps is host work: one half's dispatch and operand
    preparation proceed while the other's kernels run.

    The halves write disjoint state, each indexed by its own half, so no
    accumulator is shared. Exceptions are collected and re-raised in half order
    so a failure reports the same way it would have when the halves ran one
    after another.
    """

    import threading

    from relax.sparse_pass2.sparse_pass2_budget import set_concurrent_device_shares

    errors: dict[int, BaseException] = {}

    def _target(half_index):
        try:
            run_half(half_index)
        except BaseException as exc:  # re-raised below, in half order
            errors[half_index] = exc

    # The name is a process-level label; CPython does not push it to the OS, so
    # a profile shows these threads unnamed. Naming them through libc was tried
    # and removed: ctypes defaults a return type to int, which truncates a
    # 64-bit pthread handle, and the truncated handle segfaults. A profiling
    # convenience is not worth a crash in the driver. Profiles identify the two
    # half threads by dispatch volume instead, which is unambiguous: in the
    # order-1 trace they issued 514,849 and 496,720 CUDA calls against 7,500
    # for the next busiest thread.
    threads = [
        threading.Thread(target=_target, args=(int(k),), name=f"em-half-{int(k)}")
        for k in diagnostic_half_indices
    ]
    # Every pass-2 cache budget is a fraction of the device, written for one
    # worker at a time. Both halves sizing against the whole device is what
    # made the order-3 overlap fail with RESOURCE_EXHAUSTED while building the
    # second half's projection cache, so each is told it owns its share.
    previous_shares = set_concurrent_device_shares(len(threads))
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        set_concurrent_device_shares(previous_shares)
    for k in diagnostic_half_indices:
        if int(k) in errors:
            raise errors[int(k)]
```

The large numerical controller is still 3,194 lines. Moving this machinery does
not make that body the desired final design. Its existing function AST, publication
closure, convergence timing and array aliases are unchanged.

The reviewed whole-projector-list extraction would release the old list after
building replacements rather than before, and would release the controller's last
`projector` alias earlier. A per-half operation can leave that assignment visible;
trace it before choosing its interface. Class3D aggregation likewise retains
per-class arrays alongside stacked results. These require explicit lifecycle
decisions and matched-GPU memory/timing evidence; an otherwise unused retention
field is not an accepted design.

## Main integration and Class3D shape results

The incoming code carries Class3D summaries through shared per-half output slots.
The refactor already returns explicit score results. The adaptation keeps those
results: each shape's `HalfScoreResult.classes` is retained when K-class scoring
needs it, and `merge_class_results()` returns one merged `ClassScoreSummary`.
The existing controller installs that result once. The one-shape route and its
numerical kernels keep the incoming projection/reference-grid formulas.

For K1, the previously unused class summaries are still released before scoring
the next shape; the existing weak-reference test remains in the combined suite.
No list of all prepared shape owners is added. Class assignments return to particle
order, posterior sums add in the established float64 host precision, and class
noise uses the existing reference-shell remapping. A maintainer of this merge can
leave numbered convergence, reconstruction and pose-source loading unopened.

This private operation and its caller remain in the existing optics owner:

```python
def _merge_class_scores(summaries, classes, n_half, ref_box):
    """Merge class assignments, posterior sums and noise into one half's result."""

    from relax.dense.score_outputs import ClassScoreSummary

    if all(summary is None for summary in summaries):
        return None
    if any(summary is None for summary in summaries):
        raise ValueError("class statistics are missing for some shape classes")
    per_class = [summary.noise_stats for summary in summaries]
    if all(stats is None for stats in per_class):
        noise_stats = None
    else:
        if any(stats is None for stats in per_class) or len({len(stats) for stats in per_class}) != 1:
            raise ValueError("per-class noise statistics are missing for some shape classes")
        noise_stats = [
            _merge_noise_stats([stats[c] for stats in per_class], classes, n_half, ref_box)
            for c in range(len(per_class[0]))
        ]
    return ClassScoreSummary(
        assignments=place_by_index([summary.assignments for summary in summaries], classes, n_half),
        mstep_mass=_sum([np.asarray(summary.mstep_mass, dtype=np.float64) for summary in summaries]),
        evidence_mass=_sum([np.asarray(summary.evidence_mass, dtype=np.float64) for summary in summaries]),
        rotation_mass=_sum([np.asarray(summary.rotation_mass, dtype=np.float64) for summary in summaries]),
        noise_stats=noise_stats,
    )
```

## Startup pose source ownership

Initial pose selection, loading, norm-correction preparation and source provenance
now belong together in `relion/input_poses.py`. The command calls that operation
at the original boundary after perturbation setup and before the state-swap probe.
It retains visible replay configuration and frozen-correction selection.

`InitialPoses` carries the actual existing seed payload, computed corrections and
its `PoseProvenance`. The payload has source-specific metadata (including input
norms); keeping it complete preserves its array aliases and retained references.
The provenance is produced after selection/validation and reused by profile and
NPZ reports. It is not assembled later from disconnected controller locals.
A maintainer changing a pose source can leave numbered expectation,
reconstruction, noise updates and convergence unopened.

| Source handling | Preserved behavior |
| --- | --- |
| Explicit input-STAR request | Reject incompatible restart/diagnostic/half-set modes before loading |
| Frozen boundary | Wins over diagnostic NPZ; copy half-list structure and borrow its arrays |
| Diagnostic NPZ | Select the requested numbered/final poses with existing dtype and error rules |
| Fresh Class3D replay origins | Seed translations only; retain the supplied half arrays |
| Production input STAR | K1 reads canonical angles/origins/norms; Class3D reads origins only |
| No selected source | Keep the established unseeded/unit-correction representation |

The operation has 18 explicit inputs because it admits several established source
modes. Its interface remains a review point; another settings bag would not fix
ownership. This package adds one operation and two small result types, with no new
module or forwarding layer. Existing eight pose helpers and six file helpers stay
unchanged. Redundant casts of established integer configuration/array dimensions
and boolean operands are removed from the new operation; numerical casts stay.

414 affected CPU cases pass, including 34 new pose-source cases. All 504 actual
previous/current selection cases match (344 successes, 160 matching refusals),
including logging, dtype/layout and 80 borrowed-array checks. Eight archive
comparisons and stored-format roundtrips pass. Full-command expansion matches;
numerical/scoring/reconstruction owners are byte unchanged. The 108 CPU guards
pass on the unchanged engine path. Exact source/version scope and initial lint
and comparison-harness repairs are in the
[pose-source handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_initial_pose_owner_20261002T112730Z/HANDOFF.json).
GPU smoke, production K1/exactly K4/real-data quality and repeated speed/memory
qualification remain open. Both long controllers remain unfinished.

### Actual preparation and replay consumer

The `ReplayState` constructor below is the replay argument of the command's
`RefinementOptions` construction.

```python
initial_poses = input_poses.prepare_initial_poses(
    our_particles,
    particle_layout=particle_layout,
    relion_halfset_particles=relion_particles,
    pixel_size_angstrom=ds.voxel_size,
    data_dir=args.data_dir,
    requested_source=args.initial_pose_source,
    n_classes=args.n_classes,
    init_relion_iteration=args.init_relion_iteration,
    has_relion_half_sets=args.relion_half_sets is not None,
    diagnostic_single_half=args.diagnostic_single_half,
    frozen_boundary=frozen_boundary,
    poses_npz_path=args.init_previous_best_poses_npz,
    pose_iteration=args.init_previous_best_poses_iter,
    has_replay_pose_source=(
        args.relion_init_dir is not None or args.perturb_replay_relion_dir is not None
    ),
    class3d_translations=kclass_firstiter_translations,
    class3d_translation_path=kclass_firstiter_translation_path,
    log=logger,
)

replay=ReplayState(
    init_reference_real=None if resume_snapshot is not None else init_reference_real_for_projector,
    init_refinement_state_fields=(
        None if frozen_boundary is None else frozen_boundary.refinement_state_fields
    ),
    replay_iteration_overrides=replay_iteration_overrides,
    final_replay_override=final_replay_override,
    final_replay_reference_maps=final_replay_reference_maps,
    final_replay_source_iteration=final_replay_source_iteration,
    init_group_ids=list(particle_groups.group_ids_per_half),
    init_group_count=particle_groups.n_groups,
    relion_scale_follower_count=follower_topology.n_followers,
    relion_scale_follower_owners_by_iteration=follower_topology.owners_by_iteration,
    relion_scale_reduction_mode=follower_topology.reduction_mode,
    relion_follower_scale_replay=follower_topology.replay,
    init_relion_optics_group_count=particle_groups.n_optics_groups,
    init_previous_best_translations=(
        None
        if initial_poses.poses is None
        else initial_poses.poses["previous_best_translations"]
    ),
    init_previous_best_rotation_eulers=(
        None
        if initial_poses.poses is None
        else initial_poses.poses["previous_best_rotation_eulers"]
    ),
    init_image_corrections=(
        initial_poses.image_corrections if frozen_boundary is None else frozen_boundary.image_corrections
    ),
    init_scale_corrections=(
        initial_poses.scale_corrections if frozen_boundary is None else frozen_boundary.scale_corrections
    ),
    init_direction_prior=(
        None if frozen_boundary is None else frozen_boundary.direction_prior_per_half
    ),
    preserve_initial_direction_prior=frozen_boundary is not None,
),
```

Profile output reads `initial_poses.provenance.requested_source`,
`resolved_source` and `sha256`. Archive publication passes that same provenance,
whose four scalar fields are serialized by the formatter shown below.

### Result types and complete operation

```python
class PoseProvenance(NamedTuple):
    """Requested and selected startup pose source, reused by run reports."""

    requested_source: str
    resolved_source: str
    path: Path | None
    sha256: str | None

class InitialPoses(NamedTuple):
    """Selected pose payload and corrections in the established half-local order."""

    poses: dict | None
    image_corrections: list[np.ndarray] | None
    scale_corrections: list[np.ndarray] | None
    provenance: PoseProvenance

def prepare_initial_poses(
    input_particles,
    *,
    particle_layout: "ParticleLayout",
    relion_halfset_particles,
    pixel_size_angstrom: float,
    data_dir,
    requested_source: str,
    n_classes: int,
    init_relion_iteration: int,
    has_relion_half_sets: bool,
    diagnostic_single_half: bool,
    frozen_boundary,
    poses_npz_path,
    pose_iteration,
    has_replay_pose_source: bool,
    class3d_translations,
    class3d_translation_path,
    log: Logger,
) -> InitialPoses:
    """Select and load startup poses, their corrections and source provenance.

    See ``docs/math/relion_refinement_algorithm.md#startup-particle-state-and-norm-corrections``.
    """
    has_competing_initial_pose_source = (
        frozen_boundary is not None
        or poses_npz_path is not None
        or has_replay_pose_source
    )
    try:
        use_input_star_pose_seed = _resolve_input_star_pose_seed(
            requested_source,
            n_classes=n_classes,
            init_relion_iteration=init_relion_iteration,
            has_relion_half_sets=has_relion_half_sets,
            has_competing_pose_source=has_competing_initial_pose_source,
            diagnostic_single_half=diagnostic_single_half,
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid initial pose source: {exc}") from exc

    resolved_initial_pose_source = "diagnostic_replay" if has_competing_initial_pose_source else "none"
    initial_pose_source_path = None
    initial_pose_source_sha256 = None
    init_previous_best_poses = None
    initial_image_corrections = None
    initial_scale_corrections = None
    if frozen_boundary is not None:
        init_previous_best_poses = {
            "iteration": f"{frozen_boundary.completed_relion_iteration - 1:03d}",
            "previous_best_rotation_eulers": list(
                frozen_boundary.previous_best_rotation_eulers
            ),
            "previous_best_translations": list(
                frozen_boundary.previous_best_translations
            ),
        }
    elif poses_npz_path is not None:
        init_previous_best_poses = iteration_history._load_init_previous_best_poses_npz(
            poses_npz_path,
            pose_iteration,
        )
        log.info(
            "Diagnostic local-search seed: loaded previous best poses from %s (iter=%s; half sizes=%s)",
            poses_npz_path,
            init_previous_best_poses["iteration"],
            [
                arr.shape[0]
                for arr in init_previous_best_poses["previous_best_rotation_eulers"]
            ],
        )

    elif class3d_translations is not None:
        init_previous_best_poses = {
            "iteration": "000_translation_only",
            "previous_best_rotation_eulers": [None, None],
            "previous_best_translations": class3d_translations,
        }
        resolved_initial_pose_source = "relion_run_it000_translations"
        initial_pose_source_path = class3d_translation_path
        initial_pose_source_sha256 = _sha256_file(initial_pose_source_path)
        log.info(
            "Production fresh Class3D translation initialization: source=%s "
            "sha256=%s half_sizes=%s (orientations intentionally unset)",
            initial_pose_source_path,
            initial_pose_source_sha256,
            [arr.shape[0] for arr in class3d_translations],
        )
    elif use_input_star_pose_seed and n_classes > 1:
        input_pose_path = (Path(data_dir) / "particles.star").resolve()
        try:
            init_previous_best_poses = _load_input_star_class3d_translations(
                input_particles,
                particle_layout.half1_rows,
                voxel_size=pixel_size_angstrom,
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid input-STAR Class3D origin initialization: {exc}") from exc
        resolved_initial_pose_source = "input_star_translations"
        initial_pose_source_path = input_pose_path
        initial_pose_source_sha256 = _sha256_file(input_pose_path)
        log.info(
            "Fresh Class3D translation initialization: source=%s sha256=%s "
            "translation_units=%s particles=%d (orientations intentionally unset)",
            input_pose_path,
            initial_pose_source_sha256,
            init_previous_best_poses["translation_units"],
            init_previous_best_poses["previous_best_translations"][0].shape[0],
        )
    elif use_input_star_pose_seed:
        input_pose_path = (Path(data_dir) / "particles.star").resolve()
        try:
            init_previous_best_poses = _load_input_star_previous_best_poses(
                input_particles,
                relion_halfset_particles,
                particle_layout.half1_rows,
                particle_layout.half2_rows,
                voxel_size=pixel_size_angstrom,
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid input-STAR pose initialization: {exc}") from exc
        initial_image_corrections, initial_scale_corrections = _initial_corrections_from_norm(
            init_previous_best_poses["norm_corrections"],
        )
        resolved_initial_pose_source = "input_star"
        initial_pose_source_path = input_pose_path
        initial_pose_source_sha256 = _sha256_file(input_pose_path)
        log.info(
            "Production fresh-run pose initialization: source=%s sha256=%s "
            "translation_units=%s half_sizes=%s norm_corrections=%s",
            input_pose_path,
            initial_pose_source_sha256,
            init_previous_best_poses["translation_units"],
            [
                arr.shape[0]
                for arr in init_previous_best_poses["previous_best_rotation_eulers"]
            ],
            "unit" if initial_image_corrections is None else "from input rlnNormCorrection",
        )

    return InitialPoses(
        poses=init_previous_best_poses,
        image_corrections=initial_image_corrections,
        scale_corrections=initial_scale_corrections,
        provenance=PoseProvenance(
            requested_source=requested_source,
            resolved_source=resolved_initial_pose_source,
            path=initial_pose_source_path,
            sha256=initial_pose_source_sha256,
        ),
    )
```

## Command configuration and archive metadata

Startup pass orders and their coarse refinement limit now have one CLI
configuration producer. `InitialSampling` holds coarse/fine orders and the
resolved maximum/source. It is reused by logging, refinement options, run files,
the NPZ archive and the benchmark ledger. Dynamic `RefinementState` remains the
numbered controller's mutable scientific state; the startup record is unchanged
throughout the run. No object is assembled from controller locals at publication.

Archive metadata conversion and optional follower/dispatch fields now belong to
the existing `result_files.py` owner. The command supplies resolved inputs and
still selects profile-only output, writes the archive, collects timing, writes
the ledger, exports maps and prints its summary in the original order. The old
two private sampling-policy helpers are retired with maintained callers/tests
migrated. The existing archive writer and its other helpers are unchanged.

A maintainer changing a saved startup field can inspect its producer and the
archive formatter while leaving expectation, reconstruction and convergence
unopened. Existing complete symmetry provenance supplies the formatter's symmetry
fields; the follower input is only the replay field it consumes. The formatter
returns the actual variable-key stored payload, retained by the command through
reporting. It creates the same arrays at the same point and adds no model buffers,
GPU pass, synchronization, JIT boundary or runtime precision flag.

The formatter's 28-parameter interface remains a review point. Further grouping
needs source-provenance ownership at the validated pose/projector/restart
producers, with reuse by profile/archive/ledger consumers. A temporary object
containing those locals would conceal the existing coupling. No production module was added. Startup pose ownership now adds one substantive
operation and two small result types in the existing input-pose owner.

### Actual producer and publication calls

These are actual command statements. The first precedes rotation/translation
grid construction; the final two follow refinement, writer completion and the
profile-only return. Existing timing/ledger/map/summary operations follow them.

```python
initial_sampling = command_options.resolve_initial_sampling(
    args.healpix_order,
    args.adaptive_oversampling,
    n_classes=args.n_classes,
    max_healpix_order=args.max_healpix_order,
)

save_dict = build_archive_metadata(
    result,
    args=args,
    dataset=ds,
    effective_tau2_fudge=effective_tau2_fudge,
    follower_replay=follower_topology.replay,
    frozen_boundary=frozen_boundary,
    initial_sampling=initial_sampling,
    initial_pose_source=initial_poses.provenance,
    max_significants_resolution=max_significants_resolution,
    n_images=n_images,
    n_rotations=n_rotations,
    n_translations=translations.shape[0],
    optimizer_seed_source=optimizer_seed_source,
    particle_diameter_ang=particle_diameter_ang,
    particle_layout=particle_layout,
    perturb_replay_restart_provenance_path=perturb_replay_restart_provenance_path,
    perturb_replay_restart_provenance_sha256=perturb_replay_restart_provenance_sha256,
    perturb_replay_restart_state_iterations=perturb_replay_restart_state_iterations,
    relion_dispatch_schedule=relion_dispatch_schedule,
    relion_projector_capture_dir_resolved=relion_projector_capture_dir_resolved,
    relion_projector_capture_manifest_resolved=relion_projector_capture_manifest_resolved,
    relion_projector_replay_slot=relion_projector_replay_slot,
    relion_projector_source_manifest_sha256=relion_projector_source_manifest_sha256,
    state_swap_probe=state_swap_probe,
    symmetry_provenance=symmetry_provenance,
    tau2_fudge_source=tau2_fudge_source,
    total_time=total_time,
    use_fresh_auto_refine_order=use_fresh_auto_refine_order,
)

archive_report = write_refinement_archive(
    result,
    out_path=os.path.join(args.output, "refinement_results.npz"),
    metadata=save_dict,
    half_indices=(particle_layout.half1_rows, particle_layout.half2_rows),
    n_images=n_images,
    skip_large_outputs=args.skip_large_outputs,
)
```

### Complete startup sampling configuration

The parser supplies integers. Validation remains at configuration construction,
and consumers read the resolved fields without repeating coercions. The limit
applies to coarse order; adaptive oversampling still determines the fine order.

```python
class InitialSampling(NamedTuple):
    """Startup pass orders and the resolved refinement limit with its source."""

    coarse_order: int
    fine_order: int
    max_order: int | None
    max_order_source: str

def resolve_initial_sampling(
    healpix_order: int,
    adaptive_oversampling: int,
    *,
    n_classes: int,
    max_healpix_order: int | None,
) -> InitialSampling:
    """Resolve CLI pass orders and Class3D's fixed or auto-refine's uncapped limit."""
    if healpix_order < 0:
        raise ValueError(f"healpix_order must be non-negative, got {healpix_order}")
    if adaptive_oversampling < 0:
        raise ValueError(f"adaptive_oversampling must be non-negative, got {adaptive_oversampling}")
    if max_healpix_order is None:
        if n_classes > 1:
            max_order = healpix_order
            source = "RELION Class3D fixed --healpix_order"
        else:
            max_order = None
            source = "K=1 auto-refine, uncapped as RELION"
    else:
        if max_healpix_order < healpix_order:
            raise ValueError(
                "max_healpix_order must be >= healpix_order "
                f"({max_healpix_order} < {healpix_order})",
            )
        max_order = max_healpix_order
        source = "explicit CLI"
    return InitialSampling(
        coarse_order=healpix_order,
        fine_order=healpix_order + adaptive_oversampling,
        max_order=max_order,
        max_order_source=source,
    )
```

### Complete archive metadata formatter

Stored casts express the existing NPZ schema; their precision does not enable
double EM execution. Optional diagnostics keep their existing empty/NaN/-1
representations and field presence. Half-row arrays retain identity and order.

```python
def build_archive_metadata(
    result,
    *,
    args,
    dataset,
    effective_tau2_fudge,
    follower_replay,
    frozen_boundary,
    initial_sampling: "InitialSampling",
    initial_pose_source: "PoseProvenance",
    max_significants_resolution,
    n_images,
    n_rotations,
    n_translations,
    optimizer_seed_source,
    particle_diameter_ang,
    particle_layout,
    perturb_replay_restart_provenance_path,
    perturb_replay_restart_provenance_sha256,
    perturb_replay_restart_state_iterations,
    relion_dispatch_schedule,
    relion_projector_capture_dir_resolved,
    relion_projector_capture_manifest_resolved,
    relion_projector_replay_slot,
    relion_projector_source_manifest_sha256,
    state_swap_probe,
    symmetry_provenance,
    tau2_fudge_source,
    total_time,
    use_fresh_auto_refine_order,
) -> dict:
    """Format startup/source metadata for the refinement NPZ archive.

    Diagnostic provenance, defaults and half-row identity retain their stored
    schema. Numerical result arrays are appended by ``write_refinement_archive``.
    """
    save_dict = {
        "symmetry_label": np.asarray(symmetry_provenance["label"]),
        "symmetry_family": np.asarray(symmetry_provenance["family"]),
        "symmetry_operator_count": np.int64(symmetry_provenance["operator_count"]),
        "symmetry_operator_sha256": np.asarray(symmetry_provenance["operator_sha256"]),
        "symmetry_relion_point_group": np.int64(symmetry_provenance["relion_point_group"]),
        "symmetry_relion_point_group_order": np.int64(symmetry_provenance["relion_point_group_order"]),
        "initial_pose_source_requested": np.asarray(initial_pose_source.requested_source),
        "initial_pose_source_resolved": np.asarray(initial_pose_source.resolved_source),
        "initial_pose_source_path": np.asarray(str(initial_pose_source.path or "")),
        "initial_pose_source_sha256": np.asarray(initial_pose_source.sha256 or ""),
        "relion_fresh_particle_order_applied": np.bool_(use_fresh_auto_refine_order),
        # The seed actually used (RELION's default -1 takes the time) and where it came from.
        "random_seed": np.int64(args.seed),
        "random_seed_source": np.asarray(optimizer_seed_source),
        "current_sizes": np.array(result["current_sizes"]),
        "pixel_resolutions": np.array(result["pixel_resolutions"]),
        "wall_times": np.array(result["wall_times"]),
        "total_time": total_time,
        "n_iterations": args.max_iter,
        "healpix_order": args.healpix_order,
        "coarse_healpix_order": initial_sampling.coarse_order,
        "finest_healpix_order": initial_sampling.fine_order,
        "max_healpix_order": -1 if initial_sampling.max_order is None else initial_sampling.max_order,
        "max_healpix_order_source": np.asarray(initial_sampling.max_order_source),
        "n_rotations": n_rotations,
        "n_translations": n_translations,
        "n_images": n_images,
        "image_shape": np.array(dataset.image_shape),
        "volume_shape": np.array(dataset.volume_shape),
        "voxel_size": dataset.voxel_size,
        "adaptive_oversampling": args.adaptive_oversampling,
        "max_significants": args.max_significants,
        "max_significants_argument": (
            np.nan
            if max_significants_resolution["maximum_significants_argument"] is None
            else int(max_significants_resolution["maximum_significants_argument"])
        ),
        "max_significants_source": np.asarray(
            str(max_significants_resolution["source"])
        ),
        "max_significants_do_grad": np.bool_(
            bool(max_significants_resolution["do_grad"])
        ),
        "offset_sigma_angstrom": args.offset_sigma_angstrom,
        "tau2_fudge": np.float64(effective_tau2_fudge),
        "tau2_fudge_source": np.asarray(tau2_fudge_source),
        "particle_diameter_ang": (np.float64(particle_diameter_ang) if particle_diameter_ang is not None else np.nan),
        "firstiter_cc_effective": np.bool_(bool(args.firstiter_cc)),
        "half1_indices": particle_layout.half1_rows,
        "half2_indices": particle_layout.half2_rows,
        "perturb_replay_restart_state_iterations": np.asarray(
            perturb_replay_restart_state_iterations,
            dtype=np.int64,
        ),
        "perturb_replay_restart_provenance_path": np.asarray(
            ""
            if perturb_replay_restart_provenance_path is None
            else str(perturb_replay_restart_provenance_path)
        ),
        "perturb_replay_restart_provenance_sha256": np.asarray(
            perturb_replay_restart_provenance_sha256 or ""
        ),
        "relion_projector_replay_slot": np.int64(
            -1 if relion_projector_replay_slot is None else relion_projector_replay_slot
        ),
        "relion_projector_source_manifest_sha256": np.asarray(
            relion_projector_source_manifest_sha256 or ""
        ),
        "relion_projector_capture_dir": np.asarray(
            ""
            if relion_projector_capture_dir_resolved is None
            else str(relion_projector_capture_dir_resolved)
        ),
        "relion_projector_capture_manifest": np.asarray(
            ""
            if relion_projector_capture_manifest_resolved is None
            else str(relion_projector_capture_manifest_resolved)
        ),
        "frozen_boundary_dir": np.asarray(
            "" if frozen_boundary is None else str(frozen_boundary.source_dir)
        ),
        "frozen_boundary_manifest_sha256": np.asarray(
            "" if frozen_boundary is None else frozen_boundary.source_manifest_sha256
        ),
        "frozen_boundary_sha256": np.asarray(
            "" if frozen_boundary is None else frozen_boundary.boundary_sha256
        ),
        "frozen_boundary_completed_relion_iteration": np.int64(
            -1 if frozen_boundary is None else frozen_boundary.completed_relion_iteration
        ),
        "state_swap_probe_target_relion_iteration": np.int64(
            -1
            if state_swap_probe is None
            else int(state_swap_probe["target_relion_iteration"])
        ),
        "state_swap_probe_loop_index": np.int64(
            -1 if state_swap_probe is None else int(state_swap_probe["iteration"])
        ),
        "state_swap_probe_variant": np.asarray(
            "" if state_swap_probe is None else str(state_swap_probe["variant"])
        ),
        "state_swap_probe_replay_relion_references": np.bool_(
            False
            if state_swap_probe is None
            else bool(state_swap_probe["replay_relion_references"])
        ),
        "state_swap_probe_applied_relion_iterations": np.asarray(
            result.get("state_swap_probe_applied_relion_iterations", []),
            dtype=np.int64,
        ),
        "state_swap_probe_replay_override_keys": np.asarray(
            [] if state_swap_probe is None else state_swap_probe["replay_override_keys"],
            dtype=np.str_,
        ),
        "state_swap_probe_required_replay_override_keys": np.asarray(
            []
            if state_swap_probe is None
            else state_swap_probe["required_replay_override_keys"],
            dtype=np.str_,
        ),
    }
    if follower_replay is not None:
        save_dict["relion_follower_scale_replay_iterations"] = np.asarray(
            follower_replay.relion_iterations,
            dtype=np.int64,
        )
        save_dict["relion_follower_scale_replay_source"] = np.asarray(
            follower_replay.source
        )
        save_dict["relion_follower_scale_replay_oracle_id"] = np.asarray(
            follower_replay.oracle_id
        )
        save_dict["relion_follower_scale_replay_boundary"] = np.asarray(
            follower_replay.boundary
        )
        save_dict["relion_follower_scale_replay_source_artifacts"] = np.asarray(
            follower_replay.source_artifact_relative_paths
        )
    if relion_dispatch_schedule is not None:
        save_dict["relion_dispatch_oracle_id"] = np.asarray(
            relion_dispatch_schedule.oracle_id
        )
        save_dict["relion_dispatch_oracle_manifest_sha256"] = np.asarray(
            relion_dispatch_schedule.oracle_manifest_sha256
        )
        save_dict["relion_dispatch_particle_order_sha256"] = np.asarray(
            relion_dispatch_schedule.particle_order_sha256
        )
    return save_dict
```

### Checked source and remaining qualification

280 affected CPU cases, 37 focused cases, 108 CPU guards and four import/structure
contracts pass. The 48 actual old/new archive cases preserve every key, value,
dtype, shape, optional group and insertion order. All 448 old/new sampling cases
match, including 262 matching refusals. Complete command expansion preserves its
surrounding AST and 27 remaining helpers. All existing functions in the two
owners match their previous AST; numerical/reconstruction/noise sources are byte
unchanged. New roundtrips cover uneven/permuted halves and optional diagnostics.
Two indirect source-location tests were migrated after the first broad run.

Exact commands, source inventories and failure corrections are in the
[package handoff](/scratch/gpfs/GILLES/mg6942/tmp/relax_command_metadata_owner_20261002T101901Z/HANDOFF.json). Source is based on cached main `35430e1`;
remote refresh is unverified. Original work/comments and every previous frozen
source/job remain preserved. GPU driver access fails and Slurm sockets are denied;
the older uncertain submission must be inspected before retrying. Latest GPU,
production float32 K1/exactly K4/real-data quality and matched-GPU speed/peak RSS
remain unqualified. The large controllers remain unfinished.

## Iteration-zero model replay

The command previously mixed model-file selection, NPZ/live/STAR noise
precedence, MPI startup broadcast, class prior expansion and optimiser overrides.
Those operations now belong together in `diagnostics/initial_model_replay.py`.
The former metadata reader is retired; its source selection and numerical tests
were migrated. Noise broadcast uses the existing `initial_noise` implementation;
radial expansion and noise construction keep their RECOVAR owner.

A developer changing replay file selection or optimiser precedence can inspect
this owner and its source parsers. Expectation, reconstruction, noise estimation
and the numerical loop can remain unopened. The command still selects replay,
installs noise and priors visibly, then applies the scalar overrides before
constructing refinement options. The whole numbered controller is byte unchanged.

`ModelStar` pairs a source path with its parsed tables. `InitialModelReplay`
keeps ordered model files and their explicit shared/half-specific identity;
its reference is the first model, with no duplicated reference-table/path fields.
That source is reused by noise, prior and control operations. `NoiseReplay`
contains computed scoring variance and the raw spectra/frame scale needed to
report its installed source. It is deleted after reporting, before the prior
allocation. `InitialModelControls` names two computed startup overrides.
These are source relationships and results, not bags of controller inputs.

### Actual replay decision, updates and reporting

This contiguous source follows initial noise/prior bootstrapping and the
separate diagnostic live-noise estimator. The earlier source-selection branch
guarantees that `initial_noise_radial` is an explicit NPZ array or `None` when
this gate admits replay. Passing that value directly preserves the existing
NPZ-before-live-before-STAR precedence. Fresh/resume/frozen policy follows this
block in its original order.

```python
    if args.relion_init_dir is not None and frozen_boundary is None:
        initial_model = initial_model_replay.read_initial_model(
            args.relion_init_dir,
            n_classes=args.n_classes,
        )
        replayed_noise = initial_model_replay.prepare_noise(
            initial_model,
            grid_size=ds.grid_size,
            image_shape=ds.image_shape,
            explicit_noise_radial=initial_noise_radial,
            live_sigma2=relion_live_initial_sigma2,
            log=logger,
        )
        noise_variance = replayed_noise.variance
        initial_model_replay.log_noise_source(replayed_noise, log=logger)
        del replayed_noise
        mean_variance = initial_model_replay.prepare_prior(
            initial_model,
            n_classes=args.n_classes,
            grid_size=ds.grid_size,
            volume_shape=ds.volume_shape,
        )
        if args.n_classes > 1:
            logger.info(
                "STRICT-PARITY: replaced bootstrapped per-class tau2 with RELION it000 spectra (K=%d)",
                args.n_classes,
            )
        else:
            logger.info("STRICT-PARITY: replaced bootstrapped tau2 with RELION it000 spectrum (K=1)")
        relion_init_tau2_fudge, relion_init_sigma_offset_angstrom = initial_model_replay.read_controls(
            initial_model,
            log=logger,
        )

```

The noise assignment precedes reporting, and prior replacement precedes its
report and optimiser parsing. Those positions release the previous large
arrays at the original boundaries. The temporary noise result must not retain
its scoring arrays across a later controller replacement. Parsed model tables
retain their original command lifetime; small radial scratch and source text
expire after their last diagnostic use.

### Complete source owner and numerical operations

```python
"""Iteration-zero model replay: file identity, noise, priors and optimiser controls.

The command selects replay and installs each result at its existing boundary.
See ``docs/math/relion_refinement_algorithm.md#iteration-zero-model-replay``.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.relion import relion_metadata
from relax.relion.initial_noise import (
    read_relion_single_optics_sigma2_noise,
    relion_mpi_process_start_scoring_noise_pair,
)


class ModelStar(NamedTuple):
    path: Path
    tables: dict


@dataclass(frozen=True)
class InitialModelReplay:
    models: list[ModelStar]
    source: str

    @property
    def reference(self) -> ModelStar:
        return self.models[0]


class NoiseReplay(NamedTuple):
    """Scoring variance and the RELION-frame spectra used to report its source."""

    variance: jnp.ndarray | list[jnp.ndarray]
    sigma2_per_model: list[np.ndarray]
    frame_scale: int


class InitialModelControls(NamedTuple):
    tau2_fudge: float | None
    sigma_offset_angstrom: float | None


def read_initial_model(directory, *, n_classes: int) -> InitialModelReplay:
    """Prefer the shared model; only K1 can replay a half-specific model pair."""
    import starfile

    directory = Path(directory)
    shared_path = directory / "run_it000_model.star"
    if shared_path.exists():
        return InitialModelReplay(
            models=[ModelStar(shared_path, starfile.read(str(shared_path)))],
            source="shared",
        )

    half_paths = [directory / "run_it000_half1_model.star", directory / "run_it000_half2_model.star"]
    if n_classes == 1 and all(path.exists() for path in half_paths):
        return InitialModelReplay(
            models=[ModelStar(path, starfile.read(str(path))) for path in half_paths],
            source="half-specific",
        )

    expected = [shared_path, *half_paths]
    missing = [str(path) for path in expected if not path.exists()]
    raise SystemExit(
        "--relion_init_dir given but no compatible iter-0 model STAR was found; "
        f"missing candidates: {', '.join(missing)}",
    )


def prepare_noise(
    model,
    *,
    grid_size,
    image_shape,
    explicit_noise_radial,
    live_sigma2,
    log,
) -> NoiseReplay:
    """Resolve NPZ/live/STAR precedence and the MPI follower-1 broadcast.

    Explicit radial noise is in RECOVAR's image frame; live sigma2 is in
    RELION's frame. Missing model spectra remain an error even with an override.
    """
    from recovar.reconstruction import noise as recon_noise

    frame_scale = grid_size**4
    sigma2_per_model = [
        read_relion_single_optics_sigma2_noise(
            source.tables,
            context=f"RELION iteration-0 model {index + 1}",
        )
        for index, source in enumerate(model.models)
    ]
    if any(sigma2 is None for sigma2 in sigma2_per_model):
        raise ValueError("RELION iteration-0 model is missing rlnSigma2Noise")
    if explicit_noise_radial is not None:
        explicit_sigma2 = np.asarray(explicit_noise_radial, dtype=np.float64) / float(frame_scale)
        sigma2_per_model = [explicit_sigma2.copy() for sigma2 in sigma2_per_model]
        log.info(
            "STRICT-PARITY: explicit --init-noise-from-npz overrides rounded "
            "RELION iteration-0 rlnSigma2Noise values",
        )
    elif live_sigma2 is not None:
        sigma2_per_model = [np.asarray(live_sigma2, dtype=np.float64).copy() for sigma2 in sigma2_per_model]
    if model.source == "half-specific":
        sigma2_per_model = relion_mpi_process_start_scoring_noise_pair(
            sigma2_per_model[0], sigma2_per_model[1], split_random_halves=True,
        )
        log.info(
            "STRICT-PARITY: emulating RELION MPI rank-1 sigma2_noise broadcast "
            "for both AutoRefine half-sets",
        )
    if len(sigma2_per_model) == 1:
        sigma2 = sigma2_per_model[0]
        noise_radial = jnp.asarray(sigma2 * frame_scale)
        variance = recon_noise.make_radial_noise(noise_radial, image_shape)
    else:
        variance = [
            recon_noise.make_radial_noise(jnp.asarray(sigma2 * frame_scale), image_shape)
            for sigma2 in sigma2_per_model
        ]
    return NoiseReplay(variance, sigma2_per_model, frame_scale)


def log_noise_source(noise, *, log):
    """Report the installed noise without retaining its scoring arrays."""
    if len(noise.sigma2_per_model) == 1:
        sigma2 = noise.sigma2_per_model[0]
        log.info(
            "STRICT-PARITY: replaced bootstrapped sigma2_noise with RELION it000 "
            "shared spectrum (× N^4=%.3e). RELION shape=%s, head=%s",
            float(noise.frame_scale),
            sigma2.shape,
            np.asarray(sigma2[:5]),
        )
    else:
        log.info(
            "STRICT-PARITY: replaced bootstrapped sigma2_noise with RELION it000 "
            "per-half spectra (× N^4=%.3e). half1 shape=%s head=%s half2 shape=%s head=%s",
            float(noise.frame_scale),
            noise.sigma2_per_model[0].shape,
            np.asarray(noise.sigma2_per_model[0][:5]),
            noise.sigma2_per_model[1].shape,
            np.asarray(noise.sigma2_per_model[1][:5]),
        )


def prepare_prior(model, *, n_classes: int, grid_size, volume_shape):
    """Expand the reference model's class tau2 shells in RECOVAR's image frame."""
    frame_scale = grid_size**4
    tables = model.reference.tables
    if n_classes > 1:
        per_class_tau2 = []
        for class_index in range(n_classes):
            table = tables[f"model_class_{class_index + 1}"]
            column = "rlnReferenceTau2" if "rlnReferenceTau2" in table.columns else "rlnReferenceSigma2"
            per_class_tau2.append(np.asarray(table[column], dtype=np.float64) * frame_scale)
        return jnp.stack(
            [
                jnp.asarray(utils.make_radial_image(shells, volume_shape, extend_last_frequency=True)).reshape(-1)
                for shells in per_class_tau2
            ],
            axis=0,
        )
    table = tables["model_class_1"]
    column = "rlnReferenceTau2" if "rlnReferenceTau2" in table.columns else "rlnReferenceSigma2"
    tau2 = np.asarray(table[column], dtype=np.float64) * frame_scale
    return jnp.asarray(utils.make_radial_image(tau2, volume_shape, extend_last_frequency=True)).reshape(-1)


def read_controls(model, *, log) -> InitialModelControls:
    """Prefer model tau2 fudge; read offset sigma only from the optimiser."""
    sigma_offset_angstrom = None
    model_text = model.reference.path.read_text()
    tau2_fudge = relion_metadata._parse_relion_tau2_fudge(model_text)
    if tau2_fudge is not None:
        log.info("STRICT-PARITY: tau2_fudge from RELION it000 model.star: %.3f", tau2_fudge)
    optimiser_path = model.reference.path.parent / "run_it000_optimiser.star"
    if optimiser_path.exists():
        optimiser_text = optimiser_path.read_text()
        if tau2_fudge is None:
            tau2_fudge = relion_metadata._parse_relion_tau2_fudge(optimiser_text)
            if tau2_fudge is not None:
                log.info("STRICT-PARITY: --tau2_fudge override from RELION it000 optimiser: %.3f", tau2_fudge)
        match = re.search(r"_rlnSigmaOffsetsAngst\s+(\S+)", optimiser_text)
        if match is not None:
            sigma_offset_angstrom = float(match.group(1))
            log.info("STRICT-PARITY: --offset_sigma_angstrom override from RELION it000: %.3f Å", sigma_offset_angstrom)
    return InitialModelControls(tau2_fudge, sigma_offset_angstrom)
```

### Evidence and remaining work

46 new regressions cover source precedence, physical frames, the MPI broadcast,
K1/exactly K4 prior layouts, fallback/preferred columns, model/optimiser controls,
missing-noise refusals and the actual caller's array-release/gate boundaries.
The affected command/startup/frozen suite passes 308 cases; 177 focused cases,
108 CPU fast guards and four import/structure contracts pass. The new independent
NumPy reference was corrected to use rounded startup shells; production and
tolerances were unchanged.

Compiling the actual previous controller block and reader passes 456 comparisons:
432 captures and 24 matching refusals. Numerical call operands, shapes, dtypes,
order, text reads and log payloads match. The remaining command statements and
all 29 command helpers match their previous AST; the remaining 23 metadata
functions and the shared broadcast module are unchanged. See the [immutable
receipt](/scratch/gpfs/GILLES/mg6942/tmp/relax_initial_model_replay_20261002T091611Z/HANDOFF.json) for exact source, commands and initial failure evidence.

This does not qualify production GPU behavior, K1/K4/real-data quality or repeated
matched-GPU performance and peak RSS. Those gates remain open. The command still
spans 2,209 lines and the numerical controller 3,194 lines. Remaining reference
loading and Class3D aggregation require deliberate lifetime treatment; this is
not the completed representative controller or approval for broad automation.

## Checkpoint array capture

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T090332Z/source/docs/development/final_local_sampling_patch_review.md#checkpoint-array-capture).

### Actual scheduled capture and publication

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T090332Z/source/docs/development/final_local_sampling_patch_review.md#actual-scheduled-capture-and-publication).

### Complete capture owner and existing result construction

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T090332Z/source/docs/development/final_local_sampling_patch_review.md#complete-capture-owner-and-existing-result-construction).

### Checks and open qualification

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_main_sync_20261002T090332Z/source/docs/development/final_local_sampling_patch_review.md#checks-and-open-qualification).

## Iteration convergence policy

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#iteration-convergence-policy).

### Actual preceding-iteration decision

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#actual-preceding-iteration-decision).

### Actual update and following consumers

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#actual-update-and-following-consumers).

### Complete operation and result

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-operation-and-result).

### Ownership and runtime review

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#ownership-and-runtime-review).

### Verification and remaining design work

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#verification-and-remaining-design-work).

## Startup sampling state

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#startup-sampling-state).

### Actual controller and following initialization

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#actual-controller-and-following-initialization).

### Complete startup operation

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-startup-operation).

### Source precedence and lifetime

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#source-precedence-and-lifetime).

### Checks and outstanding qualification

[Complete previously reviewed source and calling flow](/scratch/gpfs/GILLES/mg6942/tmp/relax_iteration_convergence_20261002T080454Z/source/candidate/docs/development/final_local_sampling_patch_review.md#checks-and-outstanding-qualification).

## Previous reviewed responsibilities

Earlier complete examples are preserved in the immutable particle-layout review.
The headings below retain existing review links and their source-specific evidence.

## Particle row layout

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#particle-row-layout).

### Actual preparation and input consumers

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#actual-preparation-and-input-consumers).

### Complete row-layout operations and result

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-row-layout-operations-and-result).

### Expected accuracy and output consumption

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#expected-accuracy-and-output-consumption).

### Identity frames, lifetime and visible decisions

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#identity-frames-lifetime-and-visible-decisions).

### Evidence and remaining qualification

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#evidence-and-remaining-qualification).

## Previous reviewed responsibilities

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#previous-reviewed-responsibilities).

## Particle input preparation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#particle-input-preparation).

### Complete controller prefix and result consumption

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-controller-prefix-and-result-consumption).

### Complete loading owner

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-loading-owner).

### Shared optimiser discovery

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#shared-optimiser-discovery).

### Ownership, arrays and validation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#ownership-arrays-and-validation).

## Previous reviewed responsibilities

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#previous-reviewed-responsibilities).

## Particle pose transition

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#particle-pose-transition).

### Ownership, order and lifetime

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#ownership-order-and-lifetime).

### Existing preceding-iteration convergence decision

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#existing-preceding-iteration-convergence-decision).

### Complete surrounding post-reconstruction flow

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-surrounding-post-reconstruction-flow).

### Complete pose result types and operations

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-pose-result-types-and-operations).

### Evidence and remaining work

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#evidence-and-remaining-work).

## Iteration resolution observations

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#iteration-resolution-observations).

## Earlier reviewed boundaries

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#earlier-reviewed-boundaries).

## Command-option ownership

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#command-option-ownership).

## Particle-table source and group-layout ownership

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#particle-table-source-and-group-layout-ownership).

## Command follower-topology preparation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#command-follower-topology-preparation).

## Particle-pose interpretation and visible state update

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#particle-pose-interpretation-and-visible-state-update).

## Scope and ownership

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#scope-and-ownership).

## Argument audit

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#argument-audit).

## Review decision and surrounding controller

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#review-decision-and-surrounding-controller).

## Precision and dense engine defaults

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#precision-and-dense-engine-defaults).

## Expectation batch preparation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#expectation-batch-preparation).

## Normalization and scale update ownership

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#normalization-and-scale-update-ownership).

## Numbered image-size planning

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#numbered-image-size-planning).

## Numbered split-half FSC and prior estimation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#numbered-split-half-fsc-and-prior-estimation).

## Startup noise preparation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#startup-noise-preparation).

## Frozen-boundary CLI admission

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#frozen-boundary-cli-admission).

## Command result publication

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#command-result-publication).

## Final half priors and optics on the scoring frame

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#final-half-priors-and-optics-on-the-scoring-frame).

## Numbered half expectation and explicit publication

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#numbered-half-expectation-and-explicit-publication).

## Numbered sampling and empty expectation preparation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#numbered-sampling-and-empty-expectation-preparation).

## Complete numerical scoring results

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-numerical-scoring-results).

## Persistent references and shared class-prior estimation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#persistent-references-and-shared-class-prior-estimation).

## Persistent noise-model lifecycle

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#persistent-noise-model-lifecycle).

## Persistent direction-prior ownership

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#persistent-direction-prior-ownership).

## Current operation: optics preparation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#current-operation-optics-preparation).

## Optics preparation result types

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#optics-preparation-result-types).

## Complete optics preparation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-optics-preparation).

## Actual final optics caller

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#actual-final-optics-caller).

## Dense class adaptation and consumption

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#dense-class-adaptation-and-consumption).

## Local class adaptation and consumption

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#local-class-adaptation-and-consumption).

## Existing shared image and geometry remapping

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#existing-shared-image-and-geometry-remapping).

## 1. Particle ownership

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#1-particle-ownership).

## Initialize once at the input boundary

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#initialize-once-at-the-input-boundary).

## Controller initialization

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#controller-initialization).

## Restart updates the resident halves

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#restart-updates-the-resident-halves).

## Numbered pose update

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#numbered-pose-update).

## Numbered correction update

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#numbered-correction-update).

## 2. Shared local/dense scoring operands

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#2-shared-localdense-scoring-operands).

## Prepared projector result

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#prepared-projector-result).

## Shared optics

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#shared-optics).

## Dense batch configuration

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#dense-batch-configuration).

## Local execution configuration

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#local-execution-configuration).

## Dense execution configuration

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#dense-execution-configuration).

## Local prior inputs

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#local-prior-inputs).

## Local diagnostics

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#local-diagnostics).

## Dense sampling inputs

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#dense-sampling-inputs).

## Dense prior inputs

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#dense-prior-inputs).

## Dense mode selection

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#dense-mode-selection).

## 3. Final input boundary

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#3-final-input-boundary).

## 4. Physical image geometry

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#4-physical-image-geometry).

## Read physical geometry at entry

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#read-physical-geometry-at-entry).

## Complete local sampling module

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-local-sampling-module).

## Complete final sampling producer

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#complete-final-sampling-producer).

## Rotation-grid ownership and remaining controller work

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#rotation-grid-ownership-and-remaining-controller-work).

## 5. Final all-data orchestration boundary

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#5-final-all-data-orchestration-boundary).

## 6. Complete final all-data controller

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#6-complete-final-all-data-controller).

## 7. Final reconstruction implementation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#7-final-reconstruction-implementation).

## 8. Fixed refinement geometry defaults

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#8-fixed-refinement-geometry-defaults).

## 9. Shared reference initialization

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#9-shared-reference-initialization).

## Shared adaptive grid result

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#shared-adaptive-grid-result).

## Shared adaptive grid preparation

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#shared-adaptive-grid-preparation).

## Implementation and evidence

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#implementation-and-evidence).

### Class3D capture ownership

[Preserved complete example](/scratch/gpfs/GILLES/mg6942/tmp/relax_particle_layout_20261002T070932Z/source/candidate/docs/development/final_local_sampling_patch_review.md#class3d-capture-ownership).
