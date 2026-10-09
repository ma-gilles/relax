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
| Planning (`plan_pass1`) and the loop (`run_pass1`) | inside that function | 206 and 76 lines, one parameter each; the route, the program and support plans and the batch's two steps are functions of their own modules |
| `significance.py` | 2,660 lines | 111 lines |
| `_publish_batch` | 264 lines, nested; captures 51 enclosing variables, unpacks a 34-field tuple | `pass1_publish.publish_batch` (12 lines), explicit inputs, `BatchOutputs` record, five functions by what each writes |
| Largest function of `relax/scoring` | 1,819 lines | 466 lines (`tomo_coarse.particle_coarse_supports`) |
| Largest parameter list of `relax/scoring` | 46 | 23 (`particle_coarse_supports`); pass 1's widest is `_pass1_block_update` with 14 (nine static settings are one `ProgramStatics`) |
| Parameters never read | `disc_type`, `do_gridding_correction`; `means` read only for `n_classes` | deleted |
| `relax/scoring` production lines (physical / nonblank) | 7,632 / 6,619 | 8,968 / 7,640: the cost of 19 new modules (headers, records, docstrings); 67 / 27 above the line-count ceilings the owner set on 2026-10-09 (8,901 / 7,613), inside their 5% slack |
| Ceilings | `docs/development/scoring_structure_metrics.json` (the totals on that date) | the span, parameter and large-argument ceilings are at the measured values; the very-large-argument count is 1 |

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

1. **The planner is still a sequence of steps (rules 6, 10, 11).** `plan_pass1` (206 lines) decides the route once
   (`plan_gaussian_route` and `plan_cc_route` in `pass1_route.py`, one `RoutePlan`) and delegates the program and support
   plans (`plan_score_program`, `plan_support`); the refusals that read only the request's fields are `Pass1Request`'s own.
   What remains inline is the normalization of the request's fields, the window and the construction of the output,
   dump and batch-input records. `coarse_rotation_ids` and the healpix order are validated for every route although only
   the normalized-CC tree rescore reads them. The callers (`k_class.py`, `scripts/run_k_class_parity.py`, about 30 tests)
   still pass keywords that `_compute_k_class_significance_batched` turns into a `Pass1Request`; that entry is the seam
   they patch and the fingerprint calls, so it stays until they build the request.
2. **The environment steers pass 1 below the boundary (rule 5).** Read in `pass1_plan.py`:
   `RELAX_K1_COARSE_ROTATED_RADIUS`, `RELAX_COARSE_PAD_FINAL_IMAGE_BATCH` (also the field `pad_final_image_batch`: two
   owners), `RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP` (its only effect is a refusal) and
   `RELAX_COARSE_GEMM_FLOAT64` (read once per pass, then passed to `plan_score_program`); in `pass1_route.py`
   `RECOVAR_K1_RELION_F32_COARSE_SUPPORT` (selects the generic support route); the projection-cache variables
   (`gaussian_plan.py`: they change `rotation_block_size`); in `pass1_assembly.py` the support-audit variables; in
   `pass1_dump.py` the dump target list.
3. **A diagnostic steers execution (rule 9).** A dump batch switches off device compaction and deferred publishing.
   The dump and the audit are imported by `pass1_publish.py`, `pass1_dump.py`, `pass1_assembly.py` and
   `tree_rescore.py` (the allowlist of `tests/unit/test_refinement_port_imports.py` names the four edges; the two
   edges of `significance.py` it replaced are gone).
4. **Layer leak (rule 11).** Pass 1 imports `relax/sparse_pass2` modules (private names among them:
   `_relion_f32_fine_posterior`, `_relion_cuda_fine_full_to_compact_lookup`).

## Noticed, not changed (a refactor does not fix behaviour)

- With the tail-batch padding on (the default), `maybe_dump_tree_rescore_batch` indexes the padded ambiguous rows into
  the unpadded batch indices and raises an `IndexError` when a tree-rescore dump is requested for a half whose last
  batch is short. A diagnostic only; the fingerprint's dump case uses a half without a tail.

## Waiting for the owner

- Whether `RECOVAR_K1_RELION_F32_COARSE_SUPPORT=0` (the generic Gaussian support route, set only by tests) is retired
  as a production capability, kept as a named option, or moved to `tests/oracles` as an independent reference.
- Whether the legacy source-pixel disk (`RELAX_K1_COARSE_ROTATED_RADIUS=0`) stays an option.
- Whether dumps and the support audit become a `RunObserver` hook within this refactor.

Retiring the generic Gaussian support route (the first question) would also delete the generic route's RELION
normalization branch (`_coarse_max_posterior_for_host`, two `SupportResult` and `BatchOutputs` fields, their publish
branch and one private import from `relax/sparse_pass2`): about 45 of the 67 lines by which the package is over its
line-count ceiling. The rest is the headers of the modules added in the last two rounds (`pass1_route.py`,
`pass1_step.py`).

## Decided

- 2026-10-09 (the owner, on PR #54): the two line-count ceilings of `scoring_structure_metrics.json` were raised by hand
  to the measured values (2066614d), and the span, parameter and large-argument ceilings lowered to the measured values.
  Later slices lower a ceiling with `report_refinement_structure.py --lower-ceilings` and never raise one.
- 2026-10-09 (the PR #54 post-merge review; every later commit follows it): pass-1 modules import no other module's
  `_private` name (`smell_check` private-import: make the name public, or move its user); they keep no read-only alias
  of a request field and no record field nothing reads (`4b416d8e`); a module with more than 20 names it imports and uses
  once is split or calls through the owning module (single-use-import, a warning). A slice runs `scripts/dev/smell_check.py`
  and the full unit list, not only the pass-1 tests, before it is pushed; the findings it introduces are not baselined.
