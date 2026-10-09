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
| Planning (`plan_pass1`) and the loop (`run_pass1`) | inside that function | 404 and 173 lines, one parameter each |
| `significance.py` | 2,660 lines | 218 lines |
| `_publish_batch` | 264 lines, nested; captures 51 enclosing variables, unpacks a 34-field tuple | `pass1_publish.publish_batch`, explicit inputs, `BatchOutputs` record |
| Largest function of `relax/scoring` | 1,819 lines | 466 lines (`tomo_coarse.particle_coarse_supports`) |
| Largest parameter list of `relax/scoring` | 46 | 23 (`particle_coarse_supports`); pass 1's widest is `_pass1_block_update` with 22 |
| Parameters never read | `disc_type`, `do_gridding_correction`; `means` read only for `n_classes` | deleted |
| `relax/scoring` production lines (physical / nonblank) | 7,632 / 6,619 | 8,901 / 7,613: the cost of 17 new modules (headers, records, docstrings) |
| Ceilings | `docs/development/scoring_structure_metrics.json` (the totals on that date) | the two line-count ceilings are exceeded (see "Waiting for the owner") |

## Coverage

`scripts/dev/pass1_fingerprint.py` calls `_compute_k_class_significance_batched` itself, on the CPU exact-operand
harness (`tests/helpers/exact_pass1_harness.py`), for 69 cases: K=1 and K=2, every prior shape, windows (radial,
square, at the box, quantized physical), the padded tail batch, an optics-group noise table, corrections and
pre-shifts, the cached and uncached projection program, the generic and the float32 support route, the support
audit, the score dump (the host-mask route), the first-iteration normalized CC (with the tree rescore, and with poses whose scores differ) and 13 refusals. Each case records
the six results, the files a dump wrote, and an ordered trace of log records and of the calls into the stand-in
kernels. 33 deliberate mutations of pass 1 are each detected (`selftest`), and `check_mutation_anchors.py`
verifies their anchors. The refinement fingerprint cannot see inside pass 1: it replaces the function by a recorder.

Not covered by it (the GPU tiers cover them): the tree rescore's kernel and its CUDA gates (the cases replace them by a
deterministic stand-in, as `tests/unit/test_refine_relion_mode.py` does, so the selection around the kernel is covered);
RELION's CUDA preprocessing, translation kernel and texture projector (stand-ins); real CTFs and noise weighting (unit values);
GPU operation order, peak memory and lifetimes; the overlap of a batch's host read-back with the next batch's
device scoring (`defer_publish`).

## Open items

1. **The planner is one 404-line function (rules 6, 10, 11).** `plan_pass1` decides the route once and builds each
   stage's plan, but it is still a sequence of 15 steps with the options read as `request.<field>` (ten rebound or
   formatted fields keep a local name). Split it by contract into the Gaussian and the normalized-CC route planners, and
   resolve the options once at the boundary: the callers (`k_class.py`, `scripts/run_k_class_parity.py`, tests) still pass
   keywords that `_compute_k_class_significance_batched` turns into a `Pass1Request`; they would build the request.
2. **The environment steers pass 1 below the boundary (rule 5).** Read in `pass1_plan.py`:
   `RECOVAR_K1_RELION_F32_COARSE_SUPPORT` (selects the generic support route), `RELAX_K1_COARSE_ROTATED_RADIUS`,
   `RELAX_COARSE_PAD_FINAL_IMAGE_BATCH` (also the field `pad_final_image_batch`: two owners),
   `RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP` (its only effect is a refusal), the projection-cache variables
   (`gaussian_plan.py`: they change `rotation_block_size`), and `RELAX_COARSE_GEMM_FLOAT64` (`pass1_plan.py`, read once
   per pass); in `pass1_assembly.py` the support-audit variables; in `pass1_dump.py` the dump target list.
3. **A diagnostic steers execution (rule 9).** A dump batch switches off device compaction and deferred publishing.
   The dump and the audit are imported by `pass1_publish.py`, `pass1_dump.py`, `pass1_assembly.py` and
   `tree_rescore.py` (the allowlist of `tests/unit/test_refinement_port_imports.py` names the four edges; the two
   edges of `significance.py` it replaced are gone).
4. **`full_stats` is a dict (rules 8, 10).** The six-tuple is now `Pass1Result` (named fields, same positions); the
   statistics stay a dict whose keys `k_class.py`, `scripts/run_k_class_parity.py`, `global_winner_summary.py` and
   `_full_stats_from_subset` read.
5. **Layer leak (rule 11).** Pass 1 imports `relax/sparse_pass2` modules (private names among them:
   `_relion_cuda_powerclass_highres_xi2_half`, `_relion_translation_angles_f32`, `_relion_f32_fine_posterior`,
   `_relion_cuda_fine_full_to_compact_lookup`).
6. **The score-program kernels take 22, 22 and 19 parameters** (`pass1_program.py`): nine static settings travel as
   keywords through three functions.

## Noticed, not changed (a refactor does not fix behaviour)

- With the tail-batch padding on (the default), `_maybe_dump_tree_rescore_batch` indexes the padded ambiguous rows into
  the unpadded batch indices and raises an `IndexError` when a tree-rescore dump is requested for a half whose last
  batch is short. A diagnostic only; the fingerprint's dump case uses a half without a tail.

## Waiting for the owner

- Whether `RECOVAR_K1_RELION_F32_COARSE_SUPPORT=0` (the generic Gaussian support route, set only by tests) is retired
  as a production capability, kept as a named option, or moved to `tests/oracles` as an independent reference.
- Whether the legacy source-pixel disk (`RELAX_K1_COARSE_ROTATED_RADIUS=0`) stays an option.
- Whether dumps and the support audit become a `RunObserver` hook within this refactor.
- Whether the two line-count ceilings of `scoring_structure_metrics.json` are raised by hand (the split added 17 modules
  and 1,269 lines of headers, records and docstrings; the largest function fell from 1,819 to 466 lines and the widest
  parameter list from 46 to 23), or the package is trimmed to fit.

## Decided

Nothing yet.
