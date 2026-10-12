# relax/scoring pass 1 (significance) against the rules: what remains

The agent-facing record of the open items of the coarse pass of the adaptive E-step
(`relax/scoring/significance.py`, with the names `relax/scoring/tomo_coarse.py` imports from it) against
[refactor_rules.md](refactor_rules.md) and [module_template.md](module_template.md). Close an item by deleting it
here in the commit that closes it; record a decision in the "Decided" section. "Before" numbers are of main (6610f54d) on
2026-10-08. Gate and verify: `REFACTOR_MODULE=scoring` (`scripts/dev/refactor_module.sh`).

## Measured

| Measure | Before (main 6610f54d) | Now |
| --- | --- | --- |
| `_compute_k_class_significance_batched` | 1,819 lines, 46 parameters, 283 locals | 9 lines, 4 parameters and the options of `Pass1Request` |
| Planning (`plan_pass1`) and the loop (`run_pass1`) | inside that function | 205 and 71 lines, one parameter each; the route, the program and support plans and the batch's two steps are functions of their own modules |
| `significance.py` | 2,660 lines | 106 lines |
| `_publish_batch` | 264 lines, nested; captures 51 enclosing variables, unpacks a 34-field tuple | `pass1_publish.publish_batch` (16 lines), explicit inputs, `BatchOutputs` record, five functions by what each writes |
| Largest function of `relax/scoring` | 1,819 lines | 465 lines (`tomo_coarse.particle_coarse_supports`, the subtomogram coarse pass); pass 1's largest is `plan_pass1` |
| Largest parameter list of `relax/scoring` | 46 | 23 (`particle_coarse_supports`); pass 1's widest is `assemble_relion_exact_coarse_gaussian_operands` with 19 (20 until its float64 arm went), then `_pass1_block_update` with 14 (nine static settings are one `ProgramStatics`) |
| Parameters never read | `disc_type`, `do_gridding_correction`; `means` read only for `n_classes` | deleted |
| `relax/scoring` production lines (physical / nonblank) | 7,632 / 6,619 | 7,120 / 6,022, below the baseline: the layout work (2026-10-12) moved modules between packages, and the 2026-10-12 round deleted 205 net lines of `relax/scoring` |
| Ceilings | `docs/development/scoring_structure_metrics.json` (the totals on that date) | lowered to the measured values on 2026-10-12 with `report_refinement_structure.py --lower-ceilings`: span 465, parameters 23, large-argument functions 15, lines 7,120 / 6,022 (very-large-argument functions: 1, as before) |

## Coverage

`scripts/dev/pass1_fingerprint.py` calls `_compute_k_class_significance_batched` itself, on the CPU exact-operand
harness (`tests/helpers/exact_pass1_harness.py`), for 69 cases: K=1 and K=2, every prior shape, windows (radial,
square, at the box, quantized physical), the padded tail batch, an optics-group noise table, corrections and
pre-shifts, the cached and uncached projection program, the generic and the float32 support route, the support
audit, the score dump (the host-mask route), the first-iteration normalized CC (with the tree rescore, and with poses whose scores differ) and 13 refusals. Each case records
the six results, the files a dump wrote, and an ordered trace of log records and of the calls into the stand-in
kernels. 33 deliberate mutations of pass 1 are each detected (`selftest`), and `check_mutation_anchors.py`
verifies their anchors. The refinement fingerprint cannot see inside pass 1: it replaces the function by a recorder.

Each stage also has direct CPU tests (the fingerprint is a dev tool, not a test): `tests/unit/test_pass1_request.py`
(the request's refusals), `test_pass1_planning.py` (the window, priors, blocks, support plan, route and the planner's
refusals) and `test_pass1_run.py` (the batch steps, publish, the order of the loop, the typed result). They pin the
deferred-publish rule that a waiting batch holds none of its large arrays, and the order operands, publish, program.
They were checked against 13 deliberate breaks of the stage code, each caught.

Not covered by it (the GPU tiers cover them): the tree rescore's kernel and its CUDA gates (the cases replace them by a
deterministic stand-in, as `tests/unit/test_refine_relion_mode.py` does, so the selection around the kernel is covered);
RELION's CUDA preprocessing, translation kernel and texture projector (stand-ins); real CTFs and noise weighting (unit values);
GPU operation order, peak memory and lifetimes; the overlap of a batch's host read-back with the next batch's
device scoring (`defer_publish`).

## Open items

1. **The planner is still a sequence of steps (rules 6, 10, 11).** `plan_pass1` (205 lines) decides the route once
   (`plan_gaussian_route` and `plan_cc_route` in `pass1_route.py`, one `RoutePlan`) and delegates the program and support
   plans (`plan_score_program`, `plan_support`); the refusals that read only the request's fields are `Pass1Request`'s own,
   and the exact-operand refusal (`_require_exact_pass1_operands`, which includes double precision) is its first statement.
   What remains inline is the normalization of the request's fields, the window and the construction of the output,
   dump and batch-input records. `coarse_rotation_ids` and the healpix order are validated by the request for every route
   although only the normalized-CC tree rescore reads them (it resolves them itself). The callers (`k_class.py`, `scripts/run_k_class_parity.py`, about 30 tests)
   still pass keywords that `_compute_k_class_significance_batched` turns into a `Pass1Request`; that entry is the seam
   they patch and the fingerprint calls, so it stays until they build the request.
2. **The environment steers pass 1 below the boundary (rule 5).** Read in `pass1_plan.py`:
   `RELAX_K1_COARSE_ROTATED_RADIUS`, `RELAX_COARSE_PAD_FINAL_IMAGE_BATCH` (also the field `pad_final_image_batch`: two
   owners), `RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP` (its only effect is a refusal; the option also runs
   through `k_class.py`, `expectation_batches.py` and the refinement controller, so it is not pass 1's alone) and
   `RELAX_COARSE_GEMM_FLOAT64` (read once per pass, then passed to `plan_score_program`); in `pass1_route.py`
   `RECOVAR_K1_RELION_F32_COARSE_SUPPORT` (selects the generic support route; its name is `K1_RELION_F32_COARSE_SUPPORT_ENV`
   in `coarse_operands.py`, beside its one reader); the projection-cache variables (`gaussian_plan.py`: they change
   `rotation_block_size`); in `pass1_assembly.py` the support-audit variables; in `pass1_dump.py` the dump target list.
3. **A diagnostic still steers execution (rule 9).** The dumps and the support audit are a `Pass1Probe` the request carries
   (`probes.py`, layout S15): pass 1 imports no implementation. A dump batch still switches off device compaction and
   deferred publishing (the results are the same).
4. **Pass 1 imports `relax/fine_pass`, by design and without private names.** The significant-sample encodings it hands to
   pass 2 live there since layout S13, and five RELION operand helpers it shares with pass 2 are public
   (`relion_cuda_corr_img_from_rfloat_ctf`, `relion_cuda_corr_img_from_native_noise_variance`,
   `relion_cuda_pixel_correction_from_rfloat_ctf`, `relion_cuda_fine_full_to_compact_lookup`,
   `relion_translation_angles_f64`). Whether those helpers belong in a package both passes sit above (`relax/relion`) is a
   layout question for the owner of the layout work.

## Noticed, not changed (a refactor does not fix behaviour)

- With the tail-batch padding on (the default), `maybe_dump_tree_rescore_batch` indexes the padded ambiguous rows into
  the unpadded batch indices and raises an `IndexError` when a tree-rescore dump is requested for a half whose last
  batch is short. A diagnostic only; the fingerprint's dump case uses a half without a tail.

## Waiting for the owner

- Whether `RECOVAR_K1_RELION_F32_COARSE_SUPPORT=0` (the generic Gaussian support route, set only by tests) is retired
  as a production capability, kept as a named option, or moved to `tests/oracles` as an independent reference.
- Whether the legacy source-pixel disk (`RELAX_K1_COARSE_ROTATED_RADIUS=0`) stays an option.
- Whether `RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP` and the texture-interpolation keyword of pass 1
  (`relion_projector_texture_interp`, `coarse_relion_projector_texture_interp` in `k_class.py`) are retired: the manual
  fallback they selected was removed on 2026-10-02, so `False` can only be refused.
- Whether the subtomogram coarse pass, `tomo_coarse.particle_coarse_supports` (465 lines, 23 parameters, the largest
  function and parameter list of `relax/scoring`), is part of this refactor. It has no characterization harness yet.

Retiring the generic Gaussian support route (the first question) would also delete the generic route's RELION
normalization branch (`_coarse_max_posterior_for_host`, two `SupportResult` and `BatchOutputs` fields, their publish
branch and one import from `relax/fine_pass`).

## Decided

- 2026-10-09 (the owner, on PR #54): the two line-count ceilings of `scoring_structure_metrics.json` were raised by hand
  to the measured values (2066614d), and the span, parameter and large-argument ceilings lowered to the measured values.
  Later slices lower a ceiling with `report_refinement_structure.py --lower-ceilings` and never raise one.
- 2026-10-09 (the PR #54 post-merge review; every later commit follows it): pass-1 modules import no other module's
  `_private` name (`smell_check` private-import: make the name public, or move its user); they keep no read-only alias
  of a request field and no record field nothing reads (`4b416d8e`); a module with more than 20 names it imports and uses
  once is split or calls through the owning module (single-use-import, a warning). A slice runs `scripts/dev/smell_check.py`
  and the full unit list, not only the pass-1 tests, before it is pushed; the findings it introduces are not baselined.
- 2026-10-12 (rule 7, the code behind pass 1's own refusals): pass 1 is float32. `plan_pass1` refuses
  `use_float64_scoring` and `use_float64_projections` before it uploads the projector, so the float64 arm of RELION's exact
  coarse operands (`relion_exact_coarse_operands`, `assemble_relion_exact_coarse_gaussian_operands`), the complex128 upload
  and the float64 `score_real_dtype` are deleted; `RELAX_COARSE_GEMM_FLOAT64` is the one labelled double-precision
  diagnostic left in pass 1. The coarse GEMM's resource gate, which checked `block * row_bytes` against a budget raised to
  at least that product, is deleted with its record. Test-only definitions went too: a layout mask, and two dense reference
  scorers moved to `tests/helpers/dense_block_scores.py`.
