# Dense GEMM coarse engine — integration and qualification plan

User request: make the dense engine a selectable coarse engine across standard EM,
VDAM and K>1, and determine robustness across resolution. This is a new production
integration package, beyond the completed fixed-state K1 experiment. The measured
6.55x process gain at one early checkpoint is evidence, not a universal promise.

## Baseline and ownership

Start clean from main ecf60ff5ba011f92d72bb98a30f516fe8c8bf7f1 in
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relax_dense_gemm_experiment_20260928,
branch codex/dense-gemm-coarse-engine-20260928. Previous experiment and pinned
controls are immutable. Root owns design, scientific acceptance, resource allocation,
tracked documentation and integration. Fresh Sol/high dense_coarse_integration is
production source writer after root approves its exact file set. Sol/medium
dense_coarse_validation owns the reserved tier/harness files. Source edits are
serialized. The first core/harness checkpoint was refreshed without conflicts to
main abfaa0ad6f056550157afd794943c5157040393f; its 21 focused CPU checks pass.
The next adapter checkpoint passes 28 focused CPU checks, including native
preprocessing operand forwarding. Initial local GPU attempts exposed fixture
setup failures before candidate execution. The first executable comparison (v3)
reported different hard-assignment IDs; v4 proved that global winning poses
agreed and exposed a local-slot versus global-index output bug, now repaired.
The live checkout subsequently refreshed conflict-free to main
30a4a0fcae8519da13d3b58ff8ae62a8d6ae93b3, including resident memory fixes;
84 focused CPU checks pass there. Frozen debugging candidates are unchanged.

The user approved existing float32 reference gates for the new dense/resident
comparison only; production and FSC gates are unchanged. The v5 native K1
full-grid comparison passes. K4 poses and backprojection accumulators pass,
but noise/norm metadata qualification remains open. The v7 capture traces a
resident posterior-mass undercount to its float32 row scatter; an isolated A100
bucketed-reduction probe reduces mass relative L2 error from 1.695e-6 to
4.86e-8 without changing arithmetic dtype. A bounded-memory repair is under
review; this probe does not qualify production performance.

The v8 first-iteration CC diagnostic identifies a separate operand bug: the
dense adapter supplies Gaussian pixel weights to CC scoring. Projections match
and same-operand scorers agree within 1.79e-6, but the wrong weights change 9/12
winning poses. The v9 attempt to preserve the unshifted weights did not fix it:
canonical CC weights are actually constructed later in the direct-scoring
preparation path. Reusing that canonical step is the next correction. The v9
gradient case, with valid distinct windows and two reconstruction groups, agrees
on poses and passes both raw backprojection group gates; its norm statistic
remains outside the approved gate (1.291e-6 relative L2 versus 1e-6). These focused
checks do not replace smoke, medium, long, EM/VDAM trajectory or resolution
qualification. The older 6.55x result excludes this production statistics adapter.
Immutable captures and current results are indexed in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dense_gemm_coarse_engine_20260928/STATUS.md`.

The v10 native checkpoint passes Gaussian K1 and CC. Canonical CC correlation
construction is now shared by the two preparation paths. Bucketed SPA posterior
reduction fixes the support-mass discrepancy with explicit scratch accounting;
scalar and ET reductions are unchanged. The balanced K4 case passes saved
numerical gates with all four classes populated, but its final total-mass
assertion still needed the already-approved explicit float32 tolerance.
Skewed K4 and gradient norm errors remain 1.054e-6 and 1.326e-6 relative L2.
Saved component audits locate about 99% of the residual difference in aggregated
A2. A test-only row capture is next to distinguish differing inputs, scalar
scatter and outer-loop accumulation before any further numerical change.

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
