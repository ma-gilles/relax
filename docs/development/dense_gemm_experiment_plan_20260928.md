# Dense resident GEMM EM experiment

Status: exact/lagged CUDA-backed cores, real-operand tile sweeps, matched H100
native comparisons, residency traces and three-iteration reconstruction comparisons
are complete. Exact mode closely agrees with the native dense control; lagged
absolute normalization overflows at iteration two. Smoke and medium tiers pass.
The final shared raw-batch staging change passes full/tail real GPU operand checks;
raw-image uploads fall from two to one. The branch includes main df5d0118. Final
medium validation of this shared-owner change is in progress. Full dense support without adaptive pruning is the requested endpoint.
User decisions (2026-09-28): K=1 first, fixed dense pose grid, no coarse pruning;
about 100 resident images initially; exact two rotation sweeps while images stay
resident; lagged one sweep after one exact initialization iteration. Compare
translation expansion on images and on projections. A subsequent explicit user
instruction requires three translation batching regimes: all translations at
once, one at a time, and intermediate translation tiles. Tune image/rotation
capacity jointly under the same memory budget. Production defaults stay
unchanged. The lagged algorithm is an explicitly requested approximation.

## Source, ownership and delivery

- Active base and immutable control: fetched `origin/main`
  `e2401c4c53182f6aed4e12b060b28257ef86894f` (2026-09-28 10:09 EDT);
  control checkout
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relax_dense_gemm_control_20260928`.
  It is clean; empty diff SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Implementation worktree:
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relax_dense_gemm_experiment_20260928`,
  branch `codex/dense-gemm-experiment-20260928`, current base
  `df5d01182b9ecae70cca92a6f767ea9592452cee`. The branch fast-forwarded from
  the initial e240 control before v2 qualification; those intervening main
  commits did not overlap the K1 engine, native sources or environment.
  Subsequent integration from e0f50017 to df5d0118 preserved all 16 dirty/untracked
  experiment file hashes; backup `pre_main_d2277_backup/`. Transitive import checks,
  104 focused CPU tests, fixture inspection and the 102-test fast guard pass.
  Frozen v2 performance/quality evidence remains on e0f50017; new medium14632505
  covers the current shared operand owner and upstream integration.
- The original discussion checkout `relax_controls_independent_20260928` at
  `e8c14ecd2309d85f54b9167c0448d38be539a575` remains untouched, but is not the
  experiment's timing/scientific control. On the user's request to stay close
  to main, the new feature branch was moved to freshly fetched main before any
  validation. Its three untracked implementation/test files were backed up and
  their hashes matched before/after the move. No old branch-only commits were
  carried over. Backup: artifact root `pre_main_base_files.tar.gz`.
- Recheck origin/main at review checkpoints and before qualification. Integrate
  upstream deliberately while no jobs read a live source tree, adapt changed
  APIs, refresh frozen environment/native libraries, and follow the repository's
  revalidation rules. Never mutate a frozen running candidate to chase main.
- Disposable results:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dense_gemm_experiment_20260928`
  (contains `SAFE_TO_DELETE`). Preserve fixtures and all other runs.
- Primary owns this plan, resource/job decisions, scientific acceptance, review
  and integration. One GPT-6 Sol agent at high reasoning owns implementation.
- Initial writable files for Sol: new `relax/dense/gemm_experiment.py`,
  `relax/dense/gemm_experiment_kernels.py`,
  `scripts/benchmark_dense_gemm_em.py`,
  `tests/unit/test_dense_gemm_experiment.py`,
  `tests/integration/test_dense_gemm_experiment_gpu.py`, and artifact files
  beneath the result root. Request a concrete ownership expansion from the
  primary before editing other files. Do not modify the control, defaults,
  fixtures, baselines, shared binaries or standing scorecards.
- Sol maintains `IMPLEMENTATION_HANDOFF.md` in the artifact root: current
  source/diff fingerprint, files, commands, executed checks, results, unresolved
  questions, unmerged commits, job IDs and next action. No repeated history.
- First review is an executable opt-in experiment and evidence, not a default
  replacement. Subsequent qualified integration follows root delivery rules.

## Falsifiable questions

1. With identical inputs, candidates and float32 numerical conventions, does
   dense GEMM match direct CUDA scores, posterior statistics and reconstruction
   to the approved accuracy bands while reducing total device time?
2. Which translation side and translation tile size minimize score plus M-step
   time at a given memory budget, rather than just scoring time? The full-T
   expansion may lose by shrinking the feasible image batch.
3. How much wall time does lagged normalization save, and how much does it
   perturb per-image mass, reconstructions and multi-iteration map quality?
4. Does the executed device program reuse bounded storage and avoid host/device
   transfers and host-controlled score decisions within each resident batch?

## Existing code to audit and reuse

| Responsibility | Existing owner | Reuse boundary |
| --- | --- | --- |
| Exact coarse operands and translation convention | `relax/relion/relion_coarse_operands.py` | Same masks, CTF, norm/scale, image offsets, Fourier window and high-frequency score constant; avoid redundant generic preprocessing |
| Project once, score many images | `relax/scoring/coarse_gaussian_gemm.py`, `_score_relion_coarse_gaussian_gemm_macro` | Existing projection callback and score operand contract; no hybrid selection/pruning in this experiment |
| GEMM score arithmetic | `relax/scoring/scoring.py`, `_relion_coarse_gaussian_gemm_scores*` | Real/complex float32 GEMMs and absolute score restoration; inspect inherited casts rather than assume |
| Dense sufficient-statistic GEMMs | `relax/dense/dense_big_jit.py`, `_mstep_half_sums` | Sum images and translations to one slice per rotation, then backproject; avoid copying a second variant of shared formulas |
| Projection cache admission | `relax/helpers/projection_cache.py` | Exact retained/build/scratch byte estimates, bounded row construction, alias accounting |
| RELION projector state | `relax/relion/relion_projector_setup.py`, `relax/helpers/projection.py`, CUDA wrappers | Reuse float32 texture projector, coordinate/half-spectrum conventions and texture lifetime |
| Resident loop and statistics patterns | `relax/sparse_pass2/resident_pass2.py` | Fixed capacities, device loops, donation and persistent accumulators; do not import sparse support machinery just to run dense GEMMs |
| Native backprojection | `relax/helpers/adjoint.py`, `relax/helpers/half_volume_mstep.py`, pinned RECOVAR CUDA | Existing RELION x-half interpolation/scatter and final accumulator conversion |
| VDAM map update | `relax/vdam/mstep_single_class.py`, `relax/vdam/m_step.py` | Existing residual and optimizer/moment semantics; do not implement a new VDAM update |
| Focused reference coverage | `tests/unit/test_coarse_gaussian_gemm_macro.py`, dense/local tests | Independent direct-square reference, translation/phase, tail padding, geometry and dtype checks |

The existing GEMM macro is a useful primitive, not proof the requested resident
algorithm already exists. The existing significance driver can materialize all
scores and recompute redundant operands; do not route through it indiscriminately.
If a reusable primitive needs a small interface change, propose that exact edit
to the primary. Shared RECOVAR changes belong in RECOVAR dev2 with repinning,
never a copied or patched local formula.

## Numerical contract and two translation layouts

For one class, let `X[i,p]` be the corrected Fourier image, `A[r,p]` the
projection, `C[i,p]` the effective real CTF including the established scale,
and `v[i,p]` the selected Fourier-metric/inverse-noise weight. Let
`phase[t,p]` be the existing image-translation phase. The schematic absolute
log score is

```text
s[i,r,t] = -0.5 sum_p v[i,p] |C[i,p] A[r,p] - X[i,p] phase[t,p]|^2
           + log_prior[i,r,t] + established out-of-window image constant
```

Implement with the existing operand convention, including any CTF-divided
representation and zero-CTF handling; the equation is not permission to change
serialization, metric, scale or masking semantics. Score and reconstruction
images can have different masks/windows: name and retain both operands.

Image-expanded path:

```text
D[(i,t),p] = v[i,p] C[i,p] X[i,p] phase[t,p]
cross[(i,t),r] = Re(conj(D) @ A.T)
power[i,r] = (v C^2)[i,p] @ |A[r,p]|^2.T
```

Projection-expanded path:

```text
A_shift[(r,t),p] = A[r,p] conj(phase[t,p])
cross[i,(r,t)] = Re(conj(v C X)[i,p] @ A_shift.T)
power[i,r] = (v C^2)[i,p] @ |A[r,p]|^2.T
```

The power GEMM is not expanded over translations. Compare complex64 GEMM with
packing real and imaginary parts into real float32 GEMMs when profitable.
Packing, phase construction, transpose/copy and workspace costs count in timings.
Require float32 multiply/accumulate semantics; do not silently enable TF32,
FP16/BF16, float64 scoring or a direct-score fallback. Inspect the actual compiled
lowering/backend and record the effective precision. CUDA libraries may be
called through compiled JAX; Python must not dispatch each rotation tile in the
steady-state implementation. If compiler lowering fails the performance or
buffer-reuse contract, propose a narrow native cuBLAS/FFI implementation.

For reconstruction, define separate native-compatible weighted image `Drec`
and `Hrec = C^2/noise` operands. Preserve metric placement at the adjoint.
With posterior `q[i,r,t]`, image expansion computes

```text
U[r,p] = q.transpose(r,i,t).reshape(R,B*T) @ Drec_shifted.reshape(B*T,P)
W[r,p] = q.sum(t).T @ Hrec
```

Projection expansion computes

```text
U_by_translation[(r,t),p] = q.reshape(B,R*T).T @ Drec
U[r,p] = sum_t phase[t,p] * U_by_translation[r,t,p]
W[r,p] = q.sum(t).T @ Hrec
```

This reconstruction phase is essential. Validate its sign and conjugation with
nonzero asymmetric translations and complex images. Never construct a
`B*R*T*P` tensor. Backproject `Q` summed slices per tile using the existing CUDA
adjoint into persistent numerator/weight volumes. For VDAM, aggregate the
corresponding CTF-projection residual with the same posterior and reuse the
established optimizer. Plain EM and VDAM must be distinctly labeled.

## Resident execution and normalizers

```python
allocate_fixed_capacity_buffers_and_compile()
for iteration:
    prepare_projector_once(reference)  # reference fixed for entire iteration
    zero_shared_accumulators()
    for batch in stable_particle_batches:
        upload_raw_images_and_metadata_into_batch_storage()
        prepare_score_and_reconstruction_operands_on_gpu()
        if exact or iteration == 0:
            current_norm = empty_max_logsum_pair_per_image()
            for tile in device_rotation_loop:  # first complete sweep
                project_reference_for_rotation_tile(tile)
                for translations in device_translation_tile_loop:
                    score = gemm_score(tile, translations)
                    current_norm = merge_max_logsum_pair(current_norm, score)
            norm_to_use = current_norm
            # Same resident images, second complete candidate sweep.
            for tile in device_rotation_loop:
                project_reference_for_rotation_tile(tile)
                zero_2d_slice_sums()
                for translations in device_translation_tile_loop:
                    score = gemm_score(tile, translations)
                    q = exp((score - norm_to_use.maximum) - norm_to_use.log_shifted_sum)
                    add_gemm_weighted_sums_to_2d_slices(q, tile, translations)
                native_backproject_summed_slices(tile)
        else:
            current_norm = empty_max_logsum_pair_per_image()
            norm_to_use = previous_norm[particle_ids]
            for tile in device_rotation_loop:
                project_reference_for_rotation_tile(tile)
                zero_2d_slice_sums()
                for translations in device_translation_tile_loop:
                    score = gemm_score(tile, translations)
                    current_norm = merge_max_logsum_pair(current_norm, score)
                    q = exp((score - norm_to_use.maximum) - norm_to_use.log_shifted_sum)
                    add_gemm_weighted_sums_to_2d_slices(q, tile, translations)
                native_backproject_summed_slices(tile)
        store_next_norm_pair_by_stable_particle_id(current_norm)
    finalize_map_and_update_state_on_gpu_where_existing_api_allows()
    swap_previous_and_next_norm_tables()
```

- Normalize over all dense rotations/translations for an image, independent of
  tile boundaries. Exact mode uses stable online logsumexp and two candidate
  sweeps while the same images remain resident; no full score table is required.
- Represent each absolute normalizer by two float32 values, `m=max(score)`
  and `l=log(sum(exp(score-m)))`, so `logZ=m+l` mathematically. Do not collapse
  that pair before computing posterior weights. On the preserved SO3/box64
  diagnostic, scalar float32 logZ near -20,000 introduced exact mass error up
  to 0.00185 even with the same scores. Merge tile pairs with
  `m_new=max(m,m_tile)` and
  `l_new=logaddexp(l+(m-m_new), l_tile+(m_tile-m_new))`. Guard empty/padded rows.
  This is a representation change that preserves the absolute score gauge,
  including when the previous iteration's pair is used in lagged mode.
- Lagged mode uses one exact initialization iteration, then one candidate sweep
  using previous-iteration logZ. Keep old/new tables separate until the iteration
  ends. Particle reordering must not change which previous normalizer is used.
- Track `delta_logZ = current_logZ - previous_logZ` and mass
  `exp(delta_logZ)`. Never renormalize lagged weights after accumulation or silently
  substitute exact weights: that would change the requested experiment.
- Use consistent absolute score offsets, priors and quadrature convention across
  iterations. Pose-independent terms cannot be discarded independently in the
  two iterations. Freeze pose grid, window, noise/scale policy and conventions for
  the first controlled test. Mark any later policy change explicitly.
- If lagged weights overflow or an image has no finite candidates, record the
  failure and image identity; no clipping, NaN replacement or hidden fallback.
  Stable evaluation must not disguise large true mass ratios.
- Tail batches, final rotation tiles and final translation tiles use fixed
  capacities plus valid masks. Translation-tile size is an explicit experiment
  parameter `U`: `U=T` fully expanded, `U=1` serial, and `1<U<T` mixed. All
  normalizers cover the full grid, not one translation tile. Maintain one
  numerator/weight slice sum per rotation tile across all translation tiles so
  backprojection occurs once per rotation tile. Compare loop-order/cache choices
  only with their phase regeneration and backprojection work counted.
  Padding contributes neither posterior mass nor reconstruction. Avoid shape
  specialization for each tail. Keep previous/next normalizers device-resident
  over the full dataset when affordable (small scalar tables).
- No scalar device reads, NumPy conversions, host callbacks, result downloads,
  per-tile synchronization or Python data-dependent decisions inside a batch.
  Transfer raw data once per batch; defer diagnostics to declared boundaries.
- Keep reference/projector and accumulators resident across batches. Reuse
  textures, phase tables and other geometry for their correct lifetimes.
  Projection cache, if admitted, is invalidated after reference update.
- Preallocate bounded workspace and retain library handles/workspaces. Compiled
  buffer donation/aliasing is a means, not proof of one-time allocation. Inspect
  buffer assignments and a warmed GPU trace. Report any remaining allocations
  or CPU map-update boundary honestly, and remove it before a full-residency claim.

## Memory planning and tuning

Budget live arrays, including source and padded Fourier volumes, projector
textures, both 3D accumulators, old/new normalizers, raw/FFT/masked/reconstruction
images, image/projection translation expansion, score and posterior scratch,
summed slices, GEMM workspace and compiler temporaries. Float32 is 4 bytes and
complex64 is 8 bytes; full volume shapes include interpolation padding.

Leading tile sizes are `8*B*U*P` for image expansion or `8*Q*U*P` for projection
expansion, `4*B*Q*U` per real score/posterior tile, `8*Q*P` numerator and
`4*Q*P` denominator slices. Projection-expanded accumulation can require another
`8*Q*U*P` temporary. Full expansion uses `U=T`. Account for lifetime overlap
rather than add only outputs. Holding translated images across every rotation
tile trades additional storage against phase regeneration; report this choice.

Start `B=100`; explore a bounded set such as 32, 64, 100, 128, 256 after the
first profile. Select a few aligned rotation tile sizes from the actual memory
budget. Use T=1, T=9 and one representative larger translation grid; representative
coarse windows/boxes and increasing rotation grids. Do not launch a Cartesian
explosion of benchmark jobs. First select viable layouts/tiles with short probes,
then test winners on realistic regimes. Include `U=1`, `U=T`, and fitting
intermediate sizes (for T=9, start with U=1,3,9). Optimize B, Q and U jointly;
do not keep B artificially fixed when a serial/mixed version frees memory.
Provide both fixed-shape comparisons and best-under-equal-budget comparisons.

User steering: roughly equal numbers of images and projections is the first
GEMM-efficiency hypothesis; more images may win if backprojection dominates.
Start near Q=B, then include balanced expanded dimensions: image-side score
GEMM outputs `(B*U, Q)`, projection-side outputs `(B, Q*U)`. Test Q near B*U
and Q*U near B where viable. This is a hypothesis, not a requirement that
every GEMM be square: Fourier pixel count, alignment, layout, power GEMM and
weighted-sum GEMMs also affect utilization. Measure rather than assume.
At fixed R, increasing B reduces the number of image batches and hence the
number of backprojected summed slices. Report native adjoint launch count,
backprojected slice count, score throughput and total throughput together.
If the measured critical path is backprojection, prioritize larger B and
translation tiling that makes it fit before micro-optimizing score GEMMs.

Compare streaming projection and a budgeted per-iteration cache only when the
cache has a plausible benefit.

## Implementation packages and review gates

1. **Inventory and mathematical seam.** Return exact reusable functions, dtype
   and geometry contracts, existing environment/fixture provenance, proposed
   buffer layout and any required ownership expansion. Confirm no source
   duplication is needed. Prepare the first executable core in assigned files.
2. **Exact resident core.** Implement both translation sides with full, serial
   and mixed translation tiling, tiled logsumexp,
   GEMM weighted sums and native projector/adjoint composition, padded capacities
   and output/provenance records. Independent small direct-square/dense reference
   checks plus actual GPU kernel checks precede tuning.
3. **Lagged core.** Add stable-ID old/new normalizers, exact bootstrap, mass
   diagnostics and failure reporting. Demonstrate equality with exact weights
   when supplied old logZ equals current logZ; demonstrate predicted mass ratios
   when deliberately offset. Keep actual iteration state transitions testable.
4. **Harness and residency verification.** Reproducible CLI with explicit modes,
   input identities, seeds, grid/window, precision, backend, tile sizes, warmup,
   repeats and separate quality/performance outputs. Integrate real CUDA
   projection/backprojection, not just precomputed random GEMMs. Trace one warmed
   batch; report GEMM lowering, device transfers, allocations and launch gaps.
5. **Controlled performance and trajectories.** Compare the twelve structural
   combinations (exact/lagged x image/projection side x full/serial/mixed
   translation batching), plus production CUDA, using staged tuning rather than
   a wasteful complete B/Q/U sweep. Distinguish
   matched dense full-grid operator comparison from production adaptive search:
   different candidate counts are not an implementation-only speedup. Reuse
   captured matched inputs/state and existing reconstruction/finalization.
6. **Qualification and integration.** Primary reviews scientific and performance
   evidence. Run affected fast guard, smoke and medium tiers for numerical code;
   changing a production default would additionally require long qualification
   and is outside this initial default-off experiment. K1 fixed-grid results are
   not K4/production-completion claims. No baseline/tolerance changes.

## Required evidence

- Exact GEMM versus direct float32 CUDA: absolute scores, error distribution,
  competing score margins for changed winners, logZ, posterior sums/Pmax,
  reconstruction numerator/weight volumes, native final maps and FSC. Isolate
  expanded-distance cancellation from geometry/CTF/translation bugs. Use an
  independent double diagnostic when needed, never a double production fix.
- Both translation layouts: asymmetric translations, heterogeneous image CTF,
  zeros in CTF, nonuniform priors, score/reconstruction masks, padding/tail
  invariance, rotation/translation tile-size invariance including nondivisible
  translation tails, fixed-volume repeated iterations, lagged
  table identity/permutation checks and finite-value failures.
- Lagged trajectories: distribution/quantiles/max of delta-logZ and image mass,
  total mass, map/update norm and FSC per iteration, convergence behavior,
  failure rate and time to a stated quality. Speed alone is not acceptance of
  the approximation. Use existing FSC gates for any quality-equivalence claim.
- Timings: cold compile/init, image load/upload/preprocessing, projection, score
  GEMMs, normalization, weighted-sum GEMMs, phase/reorder work, backprojection,
  map update and synchronized complete iteration/process time. Do not sum
  overlapping stages. Include setup and cache-generation cost.
- Performance: repeated warmed timings and spread, particles/s, peak device
  memory with method/interval, host RSS, tile sizes and actual candidate count;
  native library/build hashes, source/diff/untracked hashes, GPU model/UUID,
  environment and explicit dispatch. No performance assertions from source alone.
- Standing scorecards are read before scientific runs; primary updates applicable
  artifacts only with executed evidence. Store this new experiment's complete
  matrix in its own JSON/Markdown report; missing measurements remain missing.

## Resources, stopping conditions and immediate delegation

The Sol agent starts with code, CPU checks and at most 15 minutes total of
short focused local GPU work, each launch bounded to 5 minutes. Check nvidia-smi
immediately before every launch; only idle physical GPUs 1-3 by UUID, never GPU0.
No local multi-iteration trajectories. Initial engineering review checkpoint:
90 minutes, or earlier on a blocking API/geometry/numerical finding. This is a
checkpoint, not an automatic cancellation of useful work.

The primary owns Slurm submissions and the job registry. Sol may prepare scripts
but does not submit jobs until assigned an explicit job package. Paired timing
uses one cryoem job, two matching GPUs, symmetric 8-12 CPUs/GPU, normally 128G
RAM/GPU, no exclusive nodes, realistic walltime and immutable sources. Preserve
scheduler CUDA visibility. Medium/long scientific work stays on Slurm. Never
change matched hardware or churn queued jobs to chase availability.

The implementation worktree's frozen pixi environment has been refreshed and
verified. Do not modify it while frozen snapshots share its environment link.
Relocated snapshots run canonical tier entry points through their frozen Python
interpreter; the pixi task wrapper otherwise tries to resolve a sibling dev
RECOVAR dependency that is absent at the relocated path.
Verify relax path, the new pinned RECOVAR commit
`5514ac6e2cb63ce1e9d1d88662f80dc4a4a8e70c`, JAX path and actual GPU backend before
scientific checks. An environment in another checkout is discovery evidence,
not authorization to run the wrong editable installation. Explicitly build and
identify exclusive native libraries before GPU qualification.

Escalate ownership needs or confirmed score errors to the primary with a minimal
reproducer. Never widen a tolerance, substitute double, prune dense candidates,
silently fall back to production direct scoring, or claim an unmeasured fast
path. The agent does not push, merge, alter shared setup or spawn further agents.
