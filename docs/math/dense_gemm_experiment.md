# Resident dense GEMM K1 experiment

This opt-in experiment holds one image batch on the GPU while streaming a fixed,
dense rotation/translation grid. It reuses native CUDA RELION projection and
BPref backprojection. It does not change production defaults or implement VDAM's
optimizer. The fixed-grid experiment is separate from production adaptive support
selection.

The implementation is [`make_batch_program`](../../relax/dense/gemm_experiment.py),
with tile arithmetic in [`gemm_experiment_kernels.py`](../../relax/dense/gemm_experiment_kernels.py).
The real-data adapter is
[`benchmark_dense_gemm_trajectory.py`](../../scripts/benchmark_dense_gemm_trajectory.py).

The selectable production EM/VDAM full-grid route is
[`run_dense_gemm_full_grid`](../../relax/dense/gemm_coarse_engine.py), using the
joint-class two-sweep program
[`make_joint_k_batch_program`](../../relax/dense/gemm_experiment.py). It does not
prune configured poses. The experimental trajectory adapter described above
has a different lifetime and benchmark scope.

## Scores and the per-image denominator

Let `i` index images, `r` rotations, `t` translations and `p` Fourier pixels.
The canonical operand assembler supplies a corrected complex score image
`X[i,p]`, pixel weight `v[i,p]`, and out-of-window term `d[i]`. For a projection
`A[r,p]` and image-translation phase `phi[t,p]`, the absolute Gaussian log score is

```text
s[i,r,t] = Re sum_p conj(A[r,p]) v[i,p] X[i,p] phi[t,p]
           - 0.5 sum_p v[i,p] |A[r,p]|²
           - 0.5 sum_p v[i,p] |X[i,p] phi[t,p]|²
           - d[i] + rotation_prior[i,r] + translation_prior[i,t]

Z[i] = sum_(all r,t) exp(s[i,r,t])
posterior[i,r,t] = exp(s[i,r,t]) / Z[i]
```

`Z` is per image, over all matches. The volume and weight accumulators are shared
across images. Exact mode obtains every image's denominator before accumulating
its batch; it does not need one volume per image.

Do not replace canonical image/CTF/noise conventions with a generic Euclidean
metric. The existing
[`relion_coarse_operands.py`](../../relax/scoring/coarse_operands.py) owns
corrections, masks, Fourier weights, zero-CTF handling and `d[i]`.

The implementation stores a float32 pair `(m[i], l[i])`, where
`m = max(s)` and `l = log(sum(exp(s-m)))`. It evaluates posterior weights as
`exp((s-m)-l)` without first rounding `m+l` to one float32 value. Pairs from tiles
merge using `m_new=max(m_a,m_b)` and
`l_new=logaddexp(l_a+m_a-m_new, l_b+m_b-m_new)`.

## Two passes and the lagged approximation

```python
allocate_reference_texture_and_persistent_tables()
for iteration:
    refresh_reference_texture(reference)        # fixed during this iteration
    zero_shared_numerator_and_weight()
    for image_batch in particles:
        stage_and_prepare_images_on_gpu()       # B images before candidate sweeps
        mode = "exact" if iteration == 0 else requested_mode

        if mode == "exact":
            norm = empty_pairs(B)
            for rotations in device_rotation_tiles(Q):
                projections = project_reference(rotations)
                for translations in device_translation_tiles(U):
                    scores = gemm_score(projections, translations)
                    norm = merge_per_image_pairs(norm, scores)
        else:
            norm = previous_iteration_pairs[stable_particle_ids]
            next_norm = empty_pairs(B)

        # Same resident images; this is the second sweep only in exact mode.
        for rotations in device_rotation_tiles(Q):
            projections = project_reference(rotations)
            zero_2d_slice_sums()
            for translations in device_translation_tiles(U):
                scores = gemm_score(projections, translations)
                if mode == "lagged":
                    next_norm = merge_per_image_pairs(next_norm, scores)
                weights = exp_using_pair(scores, norm)
                gemm_sum_images_and_translations_into_slices(weights)
            native_backproject_Q_summed_slices_into_shared_volumes()

        save_next_pairs_by_stable_particle_id()
    check_failure_flags_and_reconstruct_reference()
    swap_previous_and_next_normalizer_tables()
```

There are two candidate sweeps in exact mode and one after bootstrap in lagged
mode. Neither candidate sweep reloads images from the host. Reference textures
and accumulated volumes stay on the GPU across image batches. Diagnostics are read at iteration
boundaries in the trajectory harness.

Lagged mode uses the previous iteration's **absolute** per-image denominator.
Consequently its total image weight is
`exp(logZ_current[i] - logZ_previous[i])`, generally different from one. It also
computes the next denominator during that sweep. There is no clamp, posterior
renormalization or silent exact fallback; invalid weights are reported.

## Translation layouts and accumulation

For an image batch `B`, rotation tile `Q` and translation tile `U`:

- Image expansion packs translated weighted images as `(B*U, 2*Ps)` and
  projections as `(Q, 2*Ps)` for a real float32 score GEMM.
- Projection expansion packs translated projections as `(Q*U, 2*Ps)` and images
  as `(B, 2*Ps)`. The phase on projections is conjugated.
- The translation-independent projection-power GEMM has shape `(B,Q)` and is
  outside the translation loop.
- `U=T`, `U=1` and intermediate `U` implement full, serial and mixed expansion.
  Padding masks prevent tail candidates from contributing posterior mass.

Reconstruction uses its own pixel window and native-compatible operands:
`D[i,p]` is the weighted reconstruction image and `H[i,p]` its CTF²/noise weight.
For each rotation tile, GEMMs compute

```text
slice_num[r,p] = sum_(i,t) posterior[i,r,t] D[i,p] phi_rec[t,p]
rotation_mass[i,r] = sum_t posterior[i,r,t]
slice_weight[r,p] = sum_i rotation_mass[i,r] H[i,p]
```

For image expansion, the numerator packs the real and imaginary parts of the
translated `D` side by side, then uses one full-precision float32 GEMM of shape
`(Q,B*U) @ (B*U,2*Pr)`. Splitting its output recovers the complex slices;
the posterior operand is real, so no complex multiplication is needed.

The denominator GEMM and one native Q-slice backprojection follow all translation
tiles. No tensor with shape `(B,Q,T,P)` is formed. Score and reconstruction phases
use their respective coordinate layouts; BPref adjoint indices are a separate
native layout. The real-data adapter reuses the canonical Hermitian finalizer
and regularized map solver, not a numerator/denominator division.

## Production full-grid statistics

The EM/VDAM adapter also needs per-image noise, normalization and Wavg statistics.
The single-optics, RFLOAT-CTF route in
[`make_statistics_callbacks`](../../relax/dense/gemm_coarse_statistics.py)
uses the full posterior tile `w[i,r,t]` without selecting significant poses.
For reconstruction-window projection `P[r,p]`, it forms two expectations:

```text
E[i,t,p] = sum_r w[i,r,t] P[r,p]          # (B*T,Q) @ (Q,P) GEMM
M[i,p]   = sum_r (sum_t w[i,r,t]) |P[r,p]|²  # (B,Q) @ (Q,P) GEMM
```

If `Y[i,t,p]` is the canonically translated noise image, `v[p]` its noise
variance and `h[i,p]` the prepared CTF²/noise operand, the row-summed noise
terms are `A2[i,p] = M[i,p] h[i,p] v[p]` and
`XA[i,p] = v[p] sum_t Re(E[i,t,p] conj(Y[i,t,p]))`. The established shell and
image reducers consume their sums. For Wavg, let `X[i,t,p]` be the native
translated raw image and `c[i,p]` the raw CTF times the image scale. Its exact
pixels receive `xa_raw = c sum_t Re(E conj(X))`, `aa_raw = c² M`, and
`diff2 = aa_raw - 2 xa_raw + sum_t (sum_r w[i,r,t]) |X[i,t,p]|²`; the existing
scale divisions and rectangle power convention then apply. K classes share one
joint normalizer and fold their Wavg masks class by class. Full-precision
float32 GEMMs avoid TF32; existing higher-precision noise metadata remains in
its established dtype. This changes reduction order, so the resident-reference
float32 gates and trajectory checks remain required.

For multiple optics groups or non-RFLOAT Wavg operands, the adapter keeps the
resident row-statistics callbacks. In both routes the score grid and posterior
support remain full; an exactly zero float32 posterior may skip pixel work.

For K classes, the exact denominator is joint over `(class, rotation,
translation)` for each image. The optional class-batched first sweep projects
each class in turn, stacks their `K*Q` slices, and scores that stack against the
same translated image batch in one GEMM. It retains separate class normalizers
and best poses, then merges the normalizers across K. The second sweep still
visits classes in order so Wavg masks, statistics, and the native adjoint follow
their established reduction schedule. With score caching enabled, it reads the
first-sweep float32 scores instead of projecting and scoring again in the second
sweep; the posterior and reconstruction work are otherwise unchanged. The cache
contains `K*B*R_padded*T_padded` scores, not per-image volume accumulators. The
adapter admits that cache only within its doubled tile-workspace memory budget.

## Buffer ownership and measurement boundaries

The table below describes the experimental trajectory adapter. The production
full-grid adapter in [`_memory_tiles`](../../relax/dense/gemm_coarse_engine.py)
starts with at most 256 images per batch, up to 3,072 rotations per tile and all
translations per tile. It reduces the translation, image and rotation tiles as
needed until fixed storage plus twice its estimated tile workspace fits 60% of
available device memory. Its reference and volume accumulators remain on device
across image batches; each EM iteration creates its own adapter invocation.

| Storage | Extent | Lifetime |
| --- | --- | --- |
| Score and reconstruction images/weights | `B*Ps`, `B*Pr` | Resident batch |
| Translation-expanded images or projections | `B*U*P` or `Q*U*P` | Compiled tile workspace |
| Scores/posteriors | `B*Q*U` | Compiled tile workspace |
| Projection numerator/weight slices | `Q*Pr` | Rotation tile |
| Shared numerator and weight volumes | One complex64 and one float32 volume | All batches of an iteration; donated between batch calls |
| Old/new normalizer pairs and flags | Per stable particle ID | Iterations |
| Reference texture plus optional refresh staging | Projector capacity | Reused texture handle across iterations |
| Exact CTF rows of one request | `float64[B, requested pixels]` on the device | The request; evaluated again at the next one |

The source-precision CTF rows are intentionally float64; EM scores, projections
and accumulation remain float32/complex64. The rows are evaluated per request
by `relion_ctf._relion_ctf_program` and kept by no cache (relax#39).

The final real-data adapter fetches and stages each raw batch once and passes it
to both canonical operand owners. The optional `staged_batch` input of
[`prepare_resident_half_operands`](../../relax/fine_pass/resident_operands.py)
validates row identities and shapes and pads short batches on device. Full/tail
real GPU operand comparisons pass existing float bands. The existing lazy loader
uses float16 raw storage; preprocessing promotes to float32, and scoring and
accumulation remain float32/complex64. A B100 raw upload is 3,276,800 bytes; the
whole-process trace shows that upload count falling from two to one. The earlier
frozen timing/quality adapter still used two raw preparation fetches. Masked score
and unmasked reconstruction FFT paths remain distinct.

Warmed batch traces can establish absence of CUDA allocation APIs and host/device
copies in that region. They do not prove that XLA makes no allocator-pool
suballocations. Image staging, diagnostics, map reconstruction, compilation and
iteration-boundary texture refresh are separate costs. Refresh has an explicit
completion dependency and synchronization; it is not a zero-transfer iteration.

## Experiment status

The fixture has 128×128 source images at 4.25 Å/pixel, a 56-pixel Fourier window,
1,624 score coefficients and 1,227 reconstruction coefficients. Its C1 HEALPix
order-3 grid has 768 directions × 48 in-plane angles (7.5° psi step). The 29 shifts
form a unit-original-pixel lattice inside a three-pixel radius. All selected
matches are evaluated densely; there is no adaptive pruning. Warm timing uses
a deterministic 6,144-orientation subset, while the reconstruction comparison
uses all 36,864 orientations.


The 2026-09-28 H100 experiment compared three exact iterations on all 5,000
particles, 36,864 rotations and 29 translations, with fixed checkpoint
noise/tau/priors. Native-versus-GEMM half-map FSC over shells 1–28 had AUC at least
0.999999995 and every shell at least 0.999999967. Worst complex-map relative L2 was
3.27e-5. These describe the controlled comparison, not stock AutoRefine parity.

Matched H100 warm measurements use two swapped GPU rounds and seven repeats per
round, with exact two-pass normalization on the same dense R=6,144/T=29 support:

| Images B | Rotation tile Q | Translation side | Translation tile U | Exact ms |
| ---: | ---: | --- | ---: | ---: |
| 100 | 100 | image | 29 | 32.927 |
| 100 | 768 | image | 1 | 46.424 |
| 100 | 768 | image | 7 | 22.568 |
| 100 | 768 | image | 29 | 14.648 |
| 100 | 768 | projection | 29 | 20.325 |
| 100 | 3072 | image | 29 | 13.473 |
| 256 | 3072 | image | 29 | 26.459 |
| 512 | 3072 | image | 29 | 48.300 |

The corrected `resident_cuda` control uses the current SPA F32 GEMM scorer,
shares projected slices across images, and streams all image/orientation rows
through the production CUDA translate-sum and indexed adjoint. Job14634396 used
two matched H100s with swapped roles, two warmups and seven measured repetitions.
Both engines evaluated every R6144/T29 hypothesis in two sweeps:

| Images B | Current SPA row control, ms | Collapsed-slice GEMM, ms | Ratio |
| ---: | ---: | ---: | ---: |
| 100 | 271.23 / 271.64 | 13.375 / 13.412 | 20.27× |
| 512 | 1338.49 / 1338.08 | 48.023 / 48.084 | 27.85× |

Each pair lists the two GPU assignments. The best tested control used Q768 and
2048 rows per accumulation block; GEMM used Q3072 and full image-side translation
expansion. All twelve tile comparisons passed the exploratory operator checks;
worst numerator/denominator relative L2 errors were 2.80e-5/1.51e-5. Sampled
process memory stayed below the experiment's 4 GiB budget (largest 2956 MiB).
These warmed timings exclude staging, compilation and reconstruction. They compare
fully dense schedules; they do not estimate a normal adaptive production run.
The paired full-grid run 14634727 evaluated all 5000 particles with R36864/T29
for three exact iterations. Process wall times were 271.34 s for the row control
and 41.43 s for GEMM (6.55×). Process time includes launch/import, preparation,
compilation, transfers, reconstruction/FSC, reference refresh and output saving;
it excludes preflight, paired barriers and the separate profiler runs.
Worst Fourier-map relative L2 difference was 2.75e-5; cross-map FSC AUC over shells
1–28 was at least 0.99999999788, the largest half-map FSC difference was 1.55e-6,
and the largest ground-truth FSC AUC difference was 1.67e-7. These describe this
fixed-grid/noise/tau comparison; they are not a stock AutoRefine acceptance row.

Both warmed B100 profiles had no H2D/D2H copies, CUDA allocation/free calls or
explicit host synchronization. Across three control calls, indexed backprojection
alone used 690.5 ms of GPU kernel time in 1824 launches. The experiment's gain largely
comes from summing images before backprojection; both scorers already use GEMM.
Full-process B512 profiling measured traced allocation peaks of 1368.33 MiB for
the row control and 2392.32 MiB for GEMM; sampled process peaks were 1960/2984 MiB.
Traces exclude untraced driver/context overhead; 200 ms samples can miss transients.
Setup and teardown do allocate, transfer and synchronize; the warm-region claim
does not apply to the whole process or exclude GPU allocator pool suballocations.

A tiny B4/R8/T3 pilot failed its exploratory raw-FSC bound (shell15 delta 0.002539).
Canonical voxel analysis isolated one zero-versus-tiny-positive downsampled weight
per half; unregularized division amplified the difference. The failed check remains
recorded. No tolerance, denominator clamp or production formula was changed; the
full-grid comparison above uses the unchanged canonical FSC calculation.

The historical `native` control invokes CUDA primitives in this experiment's
two-sweep dense schedule. Its fused score/projector is used in tomography and
repeats projection across images; its ordered per-particle adjoint comes from the
deprecated physical-grid route. Current SPA CUDA shares projections and batches
row adjoints differently. Thus this historical ratio is not a current SPA-resident
performance comparison. A control using the current SPA scoring and row-accumulation owners while
retaining all dense matches is implemented and measured above. No adaptive pruning is introduced.

The historical control backprojects per-image/orientation rows, whereas GEMM sums
images before backprojection. Its ratio includes this accumulation reorganization
and projection sharing as well as GEMM scoring.

The best measured native control within the chosen 4 GiB process comparison
budget used B100/Q256: 338.167 ms versus 13.441 ms for GEMM B100/Q3072, about
25.2×. A larger native Q768 reached 319.472 ms but sampled 6,948 MiB; the budget
is an experiment constraint, not the H100 capacity. These warm batch timings
exclude preprocessing, compilation and map reconstruction. The separate full
trajectory timing includes raw batch fetch/upload, preprocessing, diagnostic
copies and map reconstruction; it excludes cross-half FSC, reference refresh
and final output serialization. It is not a whole-iteration timer.

Full-process allocation traces measured GEMM B512/Q3072 at 2,392.32 MiB traced
and 2,984 MiB sampled process memory. Native B100/Q256 measured 2,290.57/2,880 MiB;
native B512/Q32 measured 2,418.54/3,016 MiB. Traced allocation totals exclude
untraced driver/context overhead, and 200 ms sampling can miss transients.
Five warmed GEMM configurations had no host/device copies, CUDA allocation/free
calls or explicit host synchronization in the captured region. At B100/Q768/U29,
GEMMs accounted for 64.8% of summed kernel time and backprojection for 16.1%.

Lagged mode failed at iteration 2: 438 of 2,515 first-half images had overflowing
posterior mass. The finite masses followed the stated logZ-ratio identity. The
approximation is therefore unsuitable for this checkpoint/update sequence in
its requested form; exact mode remains the valid reference for this experiment.

Commands, job/source/native hashes, timing matrices, memory traces and limitations
are recorded in the [development plan](../development/dense_gemm_experiment_plan_20260928.md)
and its external evidence directory. The experiment does not change scientific
tolerances, production defaults, or baseline data.

## Reproduce the dense comparison

Run these inside an assigned H100 Slurm allocation in `cryoem`, using the frozen
checkout environment and explicitly built, source-matched native libraries as
specified in [CONTRIBUTING](../../CONTRIBUTING.md). Preserve scheduler GPU
visibility. The fixture defaults resolve the 5,000-particle iteration-1 checkpoint;
`--data-dir` and `--model-dir` can select another compatible checkpoint.

```bash
# Exact GEMM, with full image-side translation expansion.
.pixi/envs/default/bin/python scripts/benchmark_dense_gemm_trajectory.py \
  --benchmark-batch --engine gemm --mode exact \
  --images 100 --rotations 6144 --translations 29 \
  --rotation-tile 3072 --translation-tile 29 --translation-side image \
  --warmup 2 --repeats 7 --save-arrays --output /path/to/gemm.json

# Same dense matches, current SPA GEMM scorer and CUDA row-accumulation control.
.pixi/envs/default/bin/python scripts/benchmark_dense_gemm_trajectory.py \
  --benchmark-batch --engine resident_cuda --mode exact \
  --images 100 --rotations 6144 --translations 29 \
  --rotation-tile 768 --row-tile 2048 --translation-tile 29 --translation-side image \
  --warmup 2 --repeats 7 --save-arrays --output /path/to/resident_cuda.json

# Full-grid exact changing-reference trajectory on all particles in both halves.
.pixi/envs/default/bin/python scripts/benchmark_dense_gemm_trajectory.py \
  --engine gemm --mode exact --iterations 3 --half-particles 5000 \
  --images 100 --rotations 36864 --translations 29 \
  --rotation-tile 3072 --translation-tile 29 --translation-side image \
  --save-arrays --output /path/to/trajectory.json
```

For the companion full-grid control, use `--engine resident_cuda`,
`--rotation-tile 768` and `--row-tile 2048`, keeping the other trajectory settings.

For the layout sweep, vary `--translation-side image|projection` and
`--translation-tile 1|7|29`, then B and Q. For the requested approximation, use
`--mode lagged`: changing-reference trajectories always bootstrap exactly and
retain partial failure diagnostics on overflow. Warm lagged batch timing instead
uses a fixed-reference exact bootstrap. Compare paired arms on matched hardware
with swapped GPU assignments; a single invocation is not the paired timing study.
