# RELION instrumentation patches

## RELION fixes for the reference build

Unlike the diagnostic patches below, these change RELION's production behaviour. Each one fixes a RELION 5.0.1
defect recorded as a `relion-bug` issue on the relax repository, for the patched RELION build that serves as
the GPU reference. relax itself follows RELION's intended method, so it needs no change for them.

### 0001-AccProjectorPlan-Class3D-s-predefined-coarse-plans-s.patch (relax#12)

**What it fixes.** In classification (no `--auto_refine`, `--skip_align`, `--skip_rotate`, orientational prior
or tomography data), the accelerated paths (CUDA, HIP, SYCL, CPU-acc) score pass 1 with predefined coarse
projector plans. Those plans are set up without `MBL`/`MBR`, so pass 1 ignores the optics group's anisotropic
magnification (`ObservationModel::applyAnisoMag`) and its scale difference (another pixel size or box,
`applyScaleDifference`). Pass 2, auto-refine and the non-accelerated CPU path apply both. The patch makes each
backend's `setupFixedSizedObjects` use on-the-fly plans whenever any optics group's
`applyScaleDifference(applyAnisoMag(I3))` is not the identity, so pass 1 also takes the magnification and
scale difference into account.

**Base.** RELION commit `b252392` from the reference-build chain: RELION 5.0.1 `f2c1a38`, then `8bc2ab1`
(the MPI pieced pack carries every scale group's sums, relax#1), then `b252392` (the device backprojector
accumulates in double, relax#4). The patch is commit `b8153b6` on top of that chain. Its SHA-256 is
`af57ed54658a8c7a8ff22dc7febd73839209da0ea1e121e6957844f19d01c5aa`.

```bash
git -C /absolute/path/to/relion rev-parse HEAD   # b252392...
git -C /absolute/path/to/relion apply --check \
  /absolute/path/to/relax/docs/patches/0001-AccProjectorPlan-Class3D-s-predefined-coarse-plans-s.patch
git -C /absolute/path/to/relion am \
  /absolute/path/to/relax/docs/patches/0001-AccProjectorPlan-Class3D-s-predefined-coarse-plans-s.patch
```

**Evidence (relax#12).** Fixture `optics_mag_k2_10k256_20260930` (MagMat [[1.015, 0.004], [0.004, 0.99]]),
seed 42, `--firstiter_cc`, on the `b252392` build:

- The iteration-1 poses of RELION Class3D K=2's class-1 particles differ from RELION auto-refine K=1 (same
  reference and data) for 19.4% of particles, at the coarse level. relax, which applies the magnification in
  every pass, equals RELION K=1 at 100%, and relax K=2's class-1 particles equal K=1 at 100%.
- Without magnification (the even-Zernike and beam-tilt K=2 fixtures), relax and RELION Class3D agree at 100%
  at iteration 1.
- After 25 iterations the masked GT FSC-AUC was relax 0.208381 against RELION 0.213430-0.215135 (four runs).

**Qualification.** Not yet built or qualified. The first build was stopped on 2026-10-01, when the user decided
to qualify Class3D magnification against stock RELION run on the CPU instead. Its `getAllSquaredDifferences`
applies both terms in both passes. On 2026-10-06 the user approved fixing confident RELION defects in our own
build. benchw is building and qualifying this patch as the GPU reference: pass-1 poses against auto-refine K=1,
and final masked GT FSC-AUC against the stock CPU runs.

## Case-22 coarse component and operand series

The numbered `0001`--`0005` patches form a diagnostic-only series on top of
detached RELION commit
`bc319d0b3ca063de4a9c8b66da6e5b4d9f618630`:

1. `0001` captures raw coarse diff2, production weights, significance, and
   separately expanded reference-norm/cross components.
2. `0002` widens only those passive component accumulators to FP64 while
   leaving production float32 diff2 unchanged.
3. `0003` captures a bounded set of live projected references, corrected
   Fourier images, correction weights, Euler matrices, and translations.
4. `0004` adds the exact GPU `translatePixel` outputs and CUDA coarse-launch
   topology needed to replay the original float32 squared-difference path
   directly.
5. `0005` copies the live Euler matrices from device memory instead of an
   unsynchronised host-side `AccPtr` buffer.

Apply them in order.  Exact SHA-256 values are
`84a41aa04d8eae11512fe7728b482eb539844c06ac5be62e39218ee863e17631`,
`5d946b88ed75a46d2ecbe6bdf037414fd4581ae61337b91e691d7ae08f3e7be0`,
`a00ad73ac496be4b2cc0513ee7aa2fd0dd8de137927db66b4d80420d0b06ad1e`,
`3d090744381306bdccc3be641834909286355f2bc15abc707053ad48d95f3b21`,
and
`c7a27cb9467103b4cea840ce7a36c9bfd11ad1a46263f824d2baa5d04d8f5e0c`.
All paths are default-off and bounded by the existing capture particle/byte
caps.  They do not modify a production score, weight, projector, map, or
model buffer.

## relion_bpref_membership_chunked_bc319d0.patch

Adds a compact, passive `RELION_BPM_CAPTURE_*` diagnostic that copies the
exact float32 fine-posterior table and rotation identities for an explicit
particle-ID cohort after the untouched production backprojection launch.
Unlike the pre-scatter row capture below, its storage is proportional to
orientation/translation hypotheses rather than hypothesis/pixel products.
The same patch retains the checked explicit part-ID filter and bounded
pre-scatter capture.

Apply only to detached RELION commit
`bc319d0b3ca063de4a9c8b66da6e5b4d9f618630`:

```bash
git apply --check \
  /absolute/path/to/recovar/docs/patches/relion_bpref_membership_chunked_bc319d0.patch
git apply \
  /absolute/path/to/recovar/docs/patches/relion_bpref_membership_chunked_bc319d0.patch
```

The exact patch SHA-256 is
`30c2d2f7d7bdd34312ed792b86cdc1aaf3976b4ffe8cd64828def0add1f79a76`.
It is diagnostic-only and must not be treated as a production RELION change.

## relion_bpref_prescatter_chunked_capture_bc319d0.patch

Combines the explicit `RELION_BPRE_CAPTURE_PART_IDS` filter with a required
`RELION_BPRE_CAPTURE_DEVICE_BYTES` cap for passive BPref pre-scatter capture.
The diagnostic walks orientation rows in ordered chunks, preserves each row's
global `orientation_local` identity, and leaves production backprojection
unchanged. It is based on RELION commit
`bc319d0b3ca063de4a9c8b66da6e5b4d9f618630`.

The patch SHA-256 is
`1a9680d93ae6ab0577a7901999dca464c7929ed10b36c36744fc87672889668f`.
The case-22 physical-iteration-2 one-particle probe used a 512 MiB requested
device cap. Its 145,568 orientations were captured as 19 ordered chunks
instead of one 10,131,532,800-byte temporary allocation.

Apply this combined patch directly to a clean matching RELION tree. Do not
apply the part-ID-only patch first:

```bash
git -C /absolute/path/to/relion rev-parse HEAD
git -C /absolute/path/to/relion apply \
  /absolute/path/to/recovar/docs/patches/relion_bpref_prescatter_chunked_capture_bc319d0.patch
```

## relion_bpref_prescatter_part_id_filter_bc319d0.patch

Adds an optional, fail-closed `RELION_BPRE_CAPTURE_PART_IDS` CSV filter to
the passive BPref pre-scatter capture instrumentation based on RELION commit
`bc319d0b3ca063de4a9c8b66da6e5b4d9f618630`. Unset preserves the existing
all-particle diagnostic behavior. When set, the explicit ID count must equal
`RELION_BPRE_CAPTURE_EXPECTED_PARTICLES`, malformed or duplicate identities
fail closed, and unselected particles return before passive capture
allocation. Production backprojection is unchanged.

The patch SHA-256 is
`82e79e3e07079e553280e2089d2fc5c4887fb43a27c032ee6df3228eb789bd21`.
It is used by the frozen case-22 physical-iteration-2 bounded cohort.

Apply it only to the matching instrumented RELION tree:

```bash
git -C /absolute/path/to/relion rev-parse HEAD
git -C /absolute/path/to/relion apply \
  /absolute/path/to/recovar/docs/patches/relion_bpref_prescatter_part_id_filter_bc319d0.patch
```

## relion_ml_optimiser_debug_dump.patch

Adds `RECOVAR_DEBUG_DUMP_DIR`-gated dumps to
`/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser.cpp` at two points:

- **Bootstrap reconstruct** (line 3264, `setSigmaNoiseEstimatesAndSetAverageImage`):
  dumps `wsum_model.BPref[iclass].data / .weight` and `mymodel.Iref[iclass]`
  before/after the reconstruct call. Also dumps `wsum_model.current_size`
  trace at the bootstrap entry point (line ~2946).

- **Iter-1 VDAM M-step** (line ~5234, `reconstructGrad` branch): dumps
  `Iref[iclass]` before + after `reconstructGrad`, the BPref accumulator
  `data / weight`, and a meta file with `grad_current_stepsize`,
  effective stepsize, `tau2_fudge_factor`, `min_resol_shell`,
  BackProjector `pad_size / r_max / skip_gridding`.

## How to apply

```bash
cd /scratch/gpfs/GILLES/mg6942/relion
git apply /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/docs/patches/relion_ml_optimiser_debug_dump.patch
cd build_patched && make -j16
```

## How to run

```bash
export RECOVAR_DEBUG_DUMP_DIR=/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_debug_dump
mkdir -p $RECOVAR_DEBUG_DUMP_DIR
/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine \
    --o out/run --iter 1 --grad --denovo_3dref --i particles.star \
    [...exact fixture args...] --random_seed 1776701668
```

The historical fixture comparison achieved **CC = 0.999313** against the
same RELION build (machine-precision parity modulo FFTW planner
non-determinism). Its unavailable April fixture suite and binary reader are
preserved in the private experiment archive at revision `2a43af1`.

## File format

Each `.bin` dump is:

```
int64 nz
int64 ny
int64 nx
<nz * ny * nx elements>   # complex128 (if 16 bytes/elem) or float64 (if 8)
```

The archived `test_bootstrap_iref_fixture.py` contains the matching Python reader.

## Why not vendor the patched RELION

Keeping the patch as a reviewable diff means:
1. Regular RELION releases can be tracked and the patch rebased.
2. We don't ship a fork — users apply the patch locally if they need to
   regenerate dumps (e.g. for a new fixture or different RELION version).
3. Keeps the diff minimal and easy to read: dump code is env-gated, off
   by default.
