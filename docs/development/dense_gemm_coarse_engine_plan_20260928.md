# Dense GEMM coarse engine — integration and qualification plan

## Final experiment decision (September 30, 2026)

The integrated `gemm_dense` route is retained only as an **experimental,
off-by-default option**. It evaluates every configured class, rotation and
translation without adaptive pruning. The production default stays `auto`;
selecting `gemm_dense` explicitly logs its unqualified status. The separately
qualified `gemm_hybrid` still prunes fine support and is a different algorithm.

The matched same-source A100 K4/5k/128 os1 one-update comparison (job 14694770,
source `efc18e4f`) found 37.64 s warm whole-process for `auto` and 37.55 s
for cached full dense, an effective tie. Minimum per-class FSC-AUC against the
RELION iteration-2 oracle was 0.999999993 versus 0.963218, and mean absolute
Pmax error was 7.97e-6 versus 0.1132. This is an intentionally different
no-pruning posterior, but its ground-truth and multi-iteration quality remain
unqualified. The earlier isolated GEMM gains must not be quoted as production
speedups. The detailed K4 result is the scratch artifact
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_class_batch_experiment_20260929/END_TO_END_RESULTS.md`;
the source and job ID above identify the frozen run.
The remaining sections document the historical integration plan and earlier
source checkpoints, not current qualification claims.

User request: make the dense engine a selectable coarse engine across standard EM,
VDAM and K>1, and determine robustness across resolution. This is a new production
integration package, beyond the completed fixed-state K1 experiment. The measured
6.55x process gain at one early checkpoint is evidence, not a universal promise.

## Current checkpoint and evidence

The integration branch is `codex/dense-gemm-coarse-engine-20260928` in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relax_dense_gemm_experiment_20260928`.
Its production code at `e2080b5a` is based on current main `a2f87236`; the next
commit `297b55b1` strengthens a CPU assertion without changing execution. The
pinned recovar commit is `5514ac6e`, and the native source digest is `3dbcd1e4`.
Run outputs and immutable source/native identities are indexed in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dense_gemm_coarse_engine_20260928/STATUS.md`.

Both public selections are implemented across global EM and VDAM, including K>1:
`gemm_hybrid` uses exact coarse GEMM with the canonical resident fine pass and its
existing support pruning; `gemm_dense` evaluates every class/rotation/translation
on the configured expanded grid with exact two-pass per-image normalization and
no pruning. `auto` remains the production default. The lagged-normalizer
one-pass experiment overflowed and is not a selectable production engine.
Translation tiling was measured with serial, mixed and full expansion; on the
recorded box-128 A100 workload, a full image-side U=29 tile beat a projection-
side tile, while Q around 3072 beat Q around the 100-image batch. The planner
reduces B/Q/U when the memory estimate demands it; these choices are not
universal throughput claims.

A second synthetic-operand probe at box 256/current-size 128, B100/R768/T29,
again favored full image-side expansion: Q768/U29 34.13 ms versus projection-
side U29 39.88 ms, mixed U7 58.52 ms and serial U1 76.94 ms. Q96/U29 took
79.27 ms and changed the synthetic volume numerator by 3.22e-4 relative L2
on absolute scores near -3.3e5; it is a negative numerical diagnostic, not a
quality pass. These are warmed exact two-sweep kernel timings that exclude
preprocessing, staging, full statistics and reconstruction. The complete
record is `box256_tile_probe_v31/SUMMARY.md` under the artifact root.

The new dense-versus-resident fixed-state K1/K4 tests use the user's approved
existing float32 reference gates; production FSC gates and baselines have not
changed. Box-128 K1 full-grid tests at R36,864/T29 pass the score, BPref,
noise and scale gates. The selected hybrid's first-iteration CC/top-two route
also passed one-iteration K1 5k/128 and 50k/256 CUDA runs: against the default
route, all best rotations, translations and per-image Pmax agree, with 50k
unfiltered half-map relative L2 near `2.2e-7`. These are route checks, not
multi-iteration science gates.

The clean selected-hybrid smoke tier passed on A100 (job `14660387`, source
`e2080b5a`, receipt `tier_smoke_hybrid_v27/RECEIPT.json`). Medium passed all
items in 30.2 minutes on three A100s at test-only head `297b55b1` (job
`14660797`, receipt `tier_medium_hybrid_v28/RECEIPT.json`). The required long
tier started after that pass as job `14661067`, on one H100 node with four GPUs.
Earlier medium at `967d1aa2` passed, but its long run exposed the now-repaired
CC/top-two route conflict. No current long or 100k completion pass exists yet.

Matched A100 timings bound the performance claim. K2 first-iteration CC at
5k/128 was 1.49x faster warm with hybrid, K4 Gaussian 1.07x faster, and K1
Gaussian showed no process gain. The no-pruning full dense path was 2.66x
slower warm than `auto` on K1 5k/128 os0 at current size 80, and its
cross-engine FSC-AUC `0.997783` missed the unchanged `0.99987` floor; `auto`
passed. Full dense had slightly higher GT FSC-AUC in that one case, which
cannot establish a general scientific advantage. The earlier 6.55x controlled
fixed-spectrum benchmark did not include this production path and must not
be extrapolated to full refinement. Exact summaries are
`paired_hybrid_perf_v20/SUMMARY.md`, `paired_k4_gaussian_perf_v24/SUMMARY.md`
and `paired_k1_os0_fullgrid_v23/SUMMARY.md` under the artifact root above.

Publication remains conditional on the required medium/long science gates.
The full dense option needs a separate quality and speed decision. Its
no-pruning posterior has different support from RELION, but the cause of the
measured FSC gap is not yet isolated. A passing hybrid tier will not qualify
that option. Do not describe hybrid as a no-pruning engine.

## Required behavior

1. Add an explicit typed CLI/config coarse-engine selection, same meaning in EM
   and VDAM, persisted in resolved configuration and actual execution records.
   Existing engine remains default. Selecting GEMM must execute GEMM, or explain
   an unsupported configuration explicitly; no silent fallback.
2. Evaluate two explicit strategies: `gemm_hybrid` uses the current production
   coarse GEMM scorer followed by its existing fine engine and support policy;
   `gemm_dense` uses the exact dense implementation over the complete configured
   fine grid. The user explicitly requested both on clarification. Report differing
   candidate counts and any existing hybrid pruning; do not call it full dense.
   If hybrid equals current `auto`, reuse and identify that implementation rather
   than duplicating code or implying a new scoring speedup.
   For `gemm_dense`, integrate the exact two-pass implementation. All images remain resident across
   both candidate sweeps. Every configured global class/rotation/translation, including the complete
   oversampled grid when enabled, is included; padding is masked, with no adaptive support pruning.
3. Extend to K classes with one joint per-image partition function:
   Z[i] = sum over (class, rotation, translation) exp(score + all applicable priors).
   Never normalize classes independently. Maintain class-specific numerator/weight
   volumes and native class occupancy/direction/pose statistics.
4. Preserve canonical preprocessing, CTF/noise/scale/norm conventions, firstiter_cc,
   Fourier frames, native backprojection and reconstruction/gridding correction.
   Use existing sufficient-statistics owners for noise, scales, normalization,
   Wavg, Pmax, best poses and priors. No copied variant of a RECOVAR formula.
   First-iteration normalized CC scores the complete configured grid but retains
   canonical winner-only reconstruction and Pmax=1. Record evaluated candidate
   counts separately from reconstruction support; this is not adaptive pruning.
   Ties resolve by canonical class/rotation/translation order across tile sizes.
   Hybrid records report exact evaluated support and leave the post-fine-prune
   reconstruction count explicitly unknown when its existing owner does not
   expose it; dense evaluated and reconstruction counts must both be known.
5. Preserve distinct EM/VDAM controllers: subset identity, learning rate, momentum,
   class scheduling, restart metadata and resolution transitions stay with owners.
   The requested option applies to global/coarse execution; local/fine scheduling
   is a separate existing responsibility. Explicitly document the dispatch boundary.
6. Plan B/Q/U/K storage from a memory budget and current geometry, not the single
   benchmark window56. Exercise full, mixed and serial translations through the
   existing tile mechanism; keep score/accumulation F32/C64. Reuse reference/volume
   buffers when geometry permits and rebuild bounded caches when it changes.
7. Previous-iteration normalization remains an experimental diagnostic because it
   overflowed. It is not the robust engine's default or a fallback.

## Work sequence and stop conditions

A. Map both production callers and all required output/state fields; inventory
   missing statistics and existing numerical owners. Root chooses a minimal
   shared integration seam and grants a disjoint source set to one writer.
B. Implement selection and K-aware exact arithmetic with focused CPU/reference
   tests. Prove joint normalization, class-prior effects, uneven occupancies,
   class permutation behavior, padded tails and input validation. Verify actual
   route selection independently for EM and VDAM.
C. Short fixed-state GPU comparisons: K1 and exactlyK4, multiple window/box sizes,
   non-divisible B/Q/T and both translation layouts. Compare scores, posterior
   mass, all sufficient statistics, poses outside near ties and accumulators.
   Diagnose the first divergent field; no tolerances/baselines changed.
D. Short end-to-end EM and VDAM runs with changing current_size, noise/scales,
   priors and references; K4 uses Hungarian class matching and per-class reports.
   Demonstrate whole workflows rather than only a benchmark adapter.
E. Required smoke/medium and production-integration long/completion qualification,
   using existing K1 and exactlyK4 >=100k/256 gates and real-particle confirmation
   according to scoped repository contracts. First verify curated inputs exist;
   missing cases remain missing, not green skips. Update standing evidence.
F. After quality gates, paired Slurm timing/memory/residency across the validated
   resolution matrix with matched GPU model and symmetric load. Report cold setup,
   warm work and whole process separately; include any regressions or limits.
   The v9 source residency audit confirms that each image batch spans both sweeps
   on device and volume accumulators carry across batches. Before timing claims,
   profile likely projector texture restaging per tile and new JIT factories per
   adapter invocation, then reuse the canonical texture/program owners where
   measurements establish a cost. Check actual buffer aliases, allocation peaks
   and H2D/D2H/D2D traffic; source-level residency alone is insufficient.
G. Publish qualified pieces to current main, remove finished branches, document
   CLI usage, supported domain, resolution coverage, timings and negative results.

## Validation constraints

No GPU0 local workloads. Local short checks require immediate idle-GPU UUID check;
long, multi-iteration and paired final work use cryoem Slurm, right-sized requests,
no exclusive nodes. Root authorizes exact job packets; frozen candidates stay
immutable. No new source build/GPU job until source scope and first cheap check
are concrete. Preserve float32 production, existing scientific gates and defaults.

Completion requires implemented selectable behavior in both workflows, K>1
including K4 proof, actual resolution-change coverage, required scientific gates,
matched performance/quality report and qualified delivery. A CLI flag or one K1
benchmark alone is not completion. Run records, errors and live job IDs go in
STATUS.md and a concise registry; use scripts for monitoring and summaries.
