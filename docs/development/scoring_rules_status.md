# relax/scoring pass 1 (significance) against the rules: what remains

The agent-facing record of the open items of the coarse pass of the adaptive E-step
(`relax/scoring/significance.py`, with the names `relax/scoring/tomo_coarse.py` imports from it) against
[refactor_rules.md](refactor_rules.md) and [module_template.md](module_template.md). Close an item by deleting it
here in the commit that closes it; record a decision in the "Decided" section. Numbers are of main (6610f54d) on
2026-10-08. Gate and verify: `REFACTOR_MODULE=scoring` (`scripts/dev/refactor_module.sh`).

## Measured, before the first slice

| Measure | Value |
| --- | --- |
| `significance.py` | 2,660 lines |
| `_compute_k_class_significance_batched` | 1,819 lines, 46 parameters, 283 locals |
| `_publish_batch` (nested) | 264 lines; captures 51 enclosing variables, writes one with `nonlocal`, unpacks a 34-field tuple |
| Parameters never read | `disc_type`, `do_gridding_correction`; `means` is read only for `n_classes` |
| Environment reads below the options boundary | 7 groups (item 3) |
| Ceilings | `docs/development/scoring_structure_metrics.json` (the totals of `relax/scoring` on that date) |

## Coverage

`scripts/dev/pass1_fingerprint.py` calls `_compute_k_class_significance_batched` itself, on the CPU exact-operand
harness (`tests/helpers/exact_pass1_harness.py`), for 60 cases: K=1 and K=2, every prior shape, windows (radial,
square, at the box, quantized physical), the padded tail batch, an optics-group noise table, corrections and
pre-shifts, the cached and uncached projection program, the generic and the float32 support route, the support
audit, the score dump (the host-mask route), the first-iteration normalized CC and 13 refusals. Each case records
the six results, the files a dump wrote, and an ordered trace of log records and of the calls into the stand-in
kernels. 27 deliberate mutations of pass 1 are each detected (`selftest`), and `check_mutation_anchors.py`
verifies their anchors. The refinement fingerprint cannot see inside pass 1: it replaces the function by a recorder.

Not covered by it (the GPU tiers cover them): the tree rescore (CUDA only: its CPU refusal is a case); RELION's CUDA
preprocessing, translation kernel and texture projector (stand-ins); real CTFs and noise weighting (unit values);
GPU operation order, peak memory and lifetimes; the overlap of a batch's host read-back with the next batch's
device scoring (`defer_publish`).

## Open items

1. **One function holds nine jobs (rules 6, 10, 11).** Planning and validation, projection closures, per-image
   output buffers, operand preparation, the score-program driver, the tree rescore, the posterior and support
   stage, publishing, and the post-loop assembly are one body. Mode (Gaussian or normalized CC) is a flag tested about
   ten times, with `coarse_gaussian_*` and `exact_cc_*` locals set on one route only.
2. **A 34-field tuple and a closure instead of records (rule 10).** Twelve of the fields are filled only for a dump
   batch; the `None`s are how a waiting batch avoids holding the large score and operand arrays (rule 3).
3. **The environment steers pass 1 below the boundary (rule 5).** `RECOVAR_K1_RELION_F32_COARSE_SUPPORT` (selects the
   generic support route), `RELAX_K1_COARSE_ROTATED_RADIUS`, `RELAX_COARSE_PAD_FINAL_IMAGE_BATCH` (also the parameter
   `pad_final_image_batch`: two owners), `RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP` (its only effect is a
   refusal), `RELAX_DENSE_MEANS_SCALE`, the projection-cache variables (they change `rotation_block_size`), and the
   support-audit and dump variables.
4. **A diagnostic steers execution (rule 9).** A dump batch switches off device compaction and deferred publishing;
   `significance.py` imports `relax.diagnostics` directly.
5. **Dead inputs and arms (rule 7).** `disc_type`, `do_gridding_correction`, `means`; the arms that
   `_require_exact_pass1_operands` makes unreachable.
6. **Results as a dict and a positional 6-tuple (rules 8, 10).**
7. **Layer leak (rule 11).** Pass 1 imports four `relax/sparse_pass2` modules (private names among them:
   `_relion_cuda_powerclass_highres_xi2_half`, `_relion_translation_angles_f32`, `_relion_f32_fine_posterior`,
   `_relion_cuda_fine_full_to_compact_lookup`).
8. **A source-reading test pins the function (rule 13).** `tests/unit/test_adaptive_oversampling.py`
   (`_production_batch_size`).

## Waiting for the owner

- Whether `RECOVAR_K1_RELION_F32_COARSE_SUPPORT=0` (the generic Gaussian support route, set only by tests) is retired
  as a production capability, kept as a named option, or moved to `tests/oracles` as an independent reference.
- Whether the legacy source-pixel disk (`RELAX_K1_COARSE_ROTATED_RADIUS=0`) stays an option.
- Whether dumps and the support audit become a `RunObserver` hook within this refactor.

## Decided

Nothing yet.
