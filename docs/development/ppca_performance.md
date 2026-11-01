# PPCA ab initio: per-update speed (2026-11)

A record of the speed work on `relax ppca_initial_model` (both optimizers) of 2026-10-29 to 2026-11-01, for
whoever works on its speed next. The general guide is [performance_design.md](performance_design.md).

## Setting

- Workload: the 3000-update box-128 schedule of the PPCA sweeps (`[first update, shell, HEALPix order]`
  rising linearly to shell 63 at update 2500; exhaustive HEALPix 1, 2, 3, then HEALPix 3 with
  `--oversampling 1` from update 1732), q = 10, 300-image balanced batches, shifts ±4 px in 2 px steps.
- Stage mix of those 3000 updates: HEALPix 1 49, HEALPix 2 576, HEALPix 3 1106, HEALPix 3 + oversampling
  1269.
- Timing: one A100-SXM4-80GB per job, both arms of a comparison in the same job, mean seconds per update
  after two warm-up updates. HEALPix 3 and oversampled cases start from late, aligned box-128 checkpoints, so
  the posterior (and the oversampled pass 2's work) is that of a real run.

## Result

Main before this work (347050e7) against the final head (43deafe9), job 16579282:

| Stage (box 128) | momentum SGD main | head | VDAM main | head |
| --- | ---: | ---: | ---: | ---: |
| HEALPix 1, shell 11 | 0.428 | 0.261 | 0.533 | 0.323 |
| HEALPix 2, shell 23 | 0.499 | 0.300 | 0.638 | 0.428 |
| HEALPix 3, shell 46 | 1.076 | 0.660 | 1.252 | 0.819 |
| HEALPix 3 + oversampling, shell 63 | 1.023 | 0.602 | 3.455 | 1.920 |
| 3000 updates (stage mix), minutes | 46.6 | 28.0 | 102.7 | 60.1 |

The VDAM oversampled row starts from fresh moments and reaches `--maxsig` (100 significant samples per image)
within a few updates; real VDAM runs keep 5-40 per image, so its main and head times, and the VDAM totals,
are upper bounds. Box 64 HEALPix 3 (shell 31): momentum SGD 0.94 -> 0.79 s per update.

## What changed

| Commit | Change | Gain | Numbers |
| --- | --- | --- | --- |
| `6556a1ae` | window projector templated on the basis size: components and pair products in registers | 512-rotation block at shell 63: 6.4 -> 3.6 ms | bitwise |
| `929673e2` | the stream stacks the device model; one host copy of theta per update; the spherical support cached; VDAM coverage from the packed trace | 0.1-0.3 s per update | same values |
| `83a64a70` | latent epilogue: two (rotation, image) pairs per warp when there are at most 16 translations | block 0.78 -> 0.51 ms | bitwise |
| `d7e86339` | the update's log line (host SVD of the loadings, model powers) written on a background thread | about 0.15 s per update | same values |
| `9b0a7b32` | projector reads the voxel rows of a zero-padded even-width model two components per load | block at shell 63: 3.6 -> 2.6 ms; oversampled VDAM -17% | bitwise |
| `43deafe9` | GPU tiles padded to 152 images, so the per-image GEMMs run aligned kernels | HEALPix 3 -4% | GEMM order (below) |

Acceptance: 50-update traces against main run twice (same seed and batches) stay at main's own run-to-run
level on box 64 (about 1e-7 to 1e-5 relative), box-128 HEALPix 1 and 3, and oversampled runs (jobs 16471876,
16472356, local runs of 2026-10-31, 16525861). The padding commit changes the GEMMs' accumulation order;
under the owner's rule for PPCA (no bitwise requirement, algorithm and precision class unchanged) it also
passed end to end: C box 64 noise 1 with a defocus spread, 3000 updates, seeds 11 and 12, VDAM and momentum
SGD, same pose error, fraction within 10°, state FSC, and k-means ARI within the spread of the k-means
initialisation alone (0.06 on the same latents; jobs 16526040, 16526041).

## Tried and set aside

| Change | Why not |
| --- | --- |
| TF32 metric split stacked along the contraction | -3.6% for VDAM at HEALPix 3 before padding; with padding +24% slower, and 50-update traces drift to 3e-4 |
| loading singular values from a device QR | the host SVD already overlaps on the log thread; changes a logged value |
| projecting each distinct coarse rotation of an oversampled job chunk once | bitwise, but no faster: those projections are bound by writes and gather latency, not by arithmetic |
| reading the next oversampled tile ahead (reader thread, or during pass 1 on the main thread) | about 1% and 0% |
| a tile's rotation blocks in one `fori_loop` program | no gain: Python dispatch is not the bottleneck |
| padding the tiles for the score GEMM | already aligned; the score and Gram GEMMs run near the TF32 peak (about 120 TFLOPS) |

## Where the time goes now

- HEALPix 3 (exhaustive): about 0.25 s of 0.66 in the two TF32 GEMMs, near peak; then the projector and the
  latent epilogue.
- HEALPix 3 + oversampling, momentum SGD: about 0.28 s of 0.60 on the GPU; the job projections of pass 2
  0.08 s; the rest is host dispatch and the synchronisations of the per-tile significance steps.
- HEALPix 1 and 2: host-bound (the GPU works about 20 ms of 0.26 s).
- Remaining ideas, each about 5% or less: one jitted program for a tile's image preprocessing (PPCA only),
  a fused kernel that forms an oversampled job's latent Gram and inner products without writing the pair
  products (more for VDAM near `--maxsig`), and the epilogue's memory layout.
