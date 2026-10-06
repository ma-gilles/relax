# GPU compatibility

relax runs on NVIDIA GPUs from compute capability 6.0 (P100) to 9.0 (H100), on 16 to 80 GB of device
memory. This page records the policy for code that adapts to the card, how compatibility is tested, the
measured matrix, and the issues found. The user-facing summary is in the
[user guide](../user_guide.md#supported-gpus).

## Policy for small-card adaptation

Adapting to every GPU size must not make relax slower, less correct or harder to read on the large cards
(user, 2026-10-03). Every change that sizes work from device memory shows all four of these:

1. **A no-op on 80 GB cards.** On an 80 GB H100 or A100 the chosen sizes and routes are the same as before
   the change. Show it with a unit test that asserts the plan at 80 GB, and with a same-node wall A/B
   (both arms in one `--gres=gpu:2` job, GPUs swapped between rounds) that agrees within 2%.
2. **Invariant results.** Results do not depend on the tile or batch size beyond float noise. Show it with
   a test that compares two sizes.
3. **One code path.** Adaptation changes sizes only, never the algorithm. A host-versus-device fallback
   lives in one helper with a documented threshold, not in per-GPU branches.
4. **Small.** The net line count and complexity stay small. Extend the existing planners
   (`relax/helpers/batch_planning.py`, `relax/sparse_pass2/sparse_pass2_budget.py`) rather than add knobs.
   A budget is a share of `device_available_bytes(...)`, which counts state already resident on the device.

## How compatibility is tested

`scripts/gpu_matrix/` runs a fixed set of short end-to-end cells (`cells.py`) on one GPU each:

- `run_cell.py` runs `python -m relax.commands.<cmd>` as a user would. It records completion,
  out-of-memory errors, the wall time and the peak GPU memory (nvidia-smi compute-apps, summed over the
  process tree and sampled every second).
- `--emulate-gb X` makes an 80 GB card behave like an X GB card. A ballast process holds the rest of the
  device, so the free memory relax probes matches the small card. `XLA_PYTHON_CLIENT_MEM_FRACTION` is set
  to the pool limit the small card would get, which is recovar's 0.90, or relax's projector-texture
  reserve for `refine` and `class3d`, scaled to the device total.
- `della_cells.sbatch` runs cells on Della (one worker per GPU). `polar_cells.sbatch` runs them on Polar
  through `scripts/polar/submit.py --sbatch-option=--gres=gpu:<model>:1`.
- `score_cells.py` scores a set of runs against ground truth and against a reference GPU, using the same
  scorers as the benchmarks.

`run_cell.py` also records live memory from inside the relax process (`live_memory/sitecustomize.py`
reads `memory_stats()` once a second after the backend starts): `live_peak_mib` is the allocator's
`peak_bytes_in_use`, and `live_stages` gives, per refinement iteration or PPCA stage, the largest sampled
`bytes_in_use` and the cumulative peak at the stage's end, so a planner's counted bytes can be compared
with what was live. The sampler runs inside the relax process, so it is off (`MATRIX_LIVE_MEMORY=0`, or
`run_cell.py --no-live-memory`) in any run whose wall is quoted, such as an A/B or a benchmark;
`cell.json` records `live_memory_hook`. The reserved peak (`peak_gpu_mib`) is the other figure: the CUDA context, the XLA pool's reserved regions and any
allocation outside the pool, without the emulation's ballast. It is not the live-buffer peak
(`memory_stats()["peak_bytes_in_use"]`). The XLA pool grows in regions and never shrinks, so the peak
follows the card (about 13 to 15 GB on 16 GB cards, 58 to 66 GB on 80 GB cards) and is not a minimum
requirement.

The matrix cells and the benchmark arms run JAX's default allocator mode, one preallocated pool, which
is what `relax` gives a user; until 2026-10-04 they ran with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The
test tiers keep `false` (several processes on one GPU). With preallocation the reserved peak is the pool
limit from the first second, so `live_peak_mib` is the figure to read.

## Matrix

Fixtures: SPA 5k/128 (`k1_5k128`, `k2_5k128`, `k4_5k128`); box 256 (`k1_50k256`, `k4_50k256`); tomography
`cryoet_s1` depthfix (Refine3D, 1200 particles by 41 tilts), `et13_k2conf` (Class3D), `et09_box64` and
`et15_k2conf_box64_ogtomo` (VDAM), and `cryoet_ppca_k3conf_box64` (PPCA). Walls are in seconds, cold
compile cache, one seed (relax 4f1f49b unless noted).

| Workflow | H100 80 GB (sm_90) | A100 80 GB (sm_80) | P100 16 GB (sm_60) | V100 16 GB (sm_70) | A100 40 GB (sm_80) |
| --- | --- | --- | --- | --- | --- |
| Refine3D K1 5k/128 | 417 | 560 | 1504 | pending: hardware busy | pending: hardware busy |
| Class3D K2 5k/128 | 161 | 138 | 219 | pending: hardware busy | pending: hardware busy |
| Class3D K4 5k/128 | 162 | 247 | 306 | pending: hardware busy | pending: hardware busy |
| VDAM K1 5k/128, 200 iterations | 283 | 319 | 597 | pending: hardware busy | pending: hardware busy |
| VDAM K4 5k/128, 200 iterations | 482 | 686 | 2043 | pending: hardware busy | pending: hardware busy |
| Tomo Refine3D s1 | 1613 | 2697 | 5499 (main 5b44d0b) | pending: hardware busy | pending: hardware busy |
| Tomo Class3D et13, 3 iterations | 346 | 415 | 724 (main 5b44d0b) | pending: hardware busy | pending: hardware busy |
| Tomo VDAM K1 et09, 10 iterations | 115 | 119 | 216 | pending: hardware busy | pending: hardware busy |
| Tomo VDAM K2 et15, 10 iterations | 121 | 139 | 185 | pending: hardware busy | pending: hardware busy |
| PPCA tomo VDAM / SGD, 24 iterations (main 7b1e4da; Polar cards main 23c3ebd, live-memory sampler on) | 201 / 200 | — | 3857 / 3592 | 2184 / 2134 | out of memory at radius 32 (open issue 2) |

Agreement with the H100 run of the same commit and seed:

| Workflow | Metric | A100 80 GB | P100 16 GB | Same-GPU spread |
| --- | --- | --- | --- | --- |
| Refine3D K1 5k/128 | merged-map GT FSC-AUC (H100 0.596004) | 0.595999, cross 0.999992 | 0.596001, cross 0.999999 | — |
| Tomo Refine3D s1 | merged-map GT FSC-AUC (H100 0.951954) | 0.951937, cross 0.99997 | 0.951955, cross 0.999999 | — |
| Class3D K2 / K4 5k/128 | masked GT FSC-AUC, class accuracy | identical to 6 digits | identical to 6 digits | — |
| Tomo Class3D et13 | mean GT FSC-AUC (H100 0.356106) | 0.356105 | — | — |
| VDAM K1 5k/128 | masked GT FSC-AUC (H100 0.0842) | 0.0849, cross 0.966 | 0.0931, cross 0.918 | seeds 41/53: 0.345 / 0.131 |
| VDAM K4 5k/128 | weighted masked GT FSC-AUC (H100 0.241) | 0.236, cross 0.926 | 0.256 / 0.214 / 0.235, cross 0.882 / 0.834 / 0.856 | H100 repeats 0.263 / 0.275, cross 0.934 / 0.918; P100 repeats cross the P100 run 0.906 / 0.914 |
| Tomo VDAM K1 et09, 10 iterations | masked GT FSC-AUC (H100 0.3926) | 0.3926, cross 1.000 | 0.3804, cross 0.997 | seeds 2/3: 0.512 / 0.308 |
| PPCA tomo (TF32 on both) | log-likelihood, loading subspace | within 2e-7 relative, cosines ≥ 0.9999998 | — | — |

The deterministic workflows (Refine3D, Class3D) agree across GPU models to 1e-5 or better. The stochastic
ones (VDAM, 200 iterations or a stopped run) reach the same ground-truth quality on every GPU, within the
spread of repeats and seeds on one GPU. Their maps agree within one architecture (repeats: 0.92 to 0.93 on
H100, 0.91 on P100) more closely than across sm_60 and sm_90 (0.83 to 0.88): an architecture-level float
shift (reduction and atomic order in the custom kernels; production EM matmuls run at HIGHEST precision, so
not TF32), not P100 instability. Different seeds agree at 0.42 to 0.46.

### Small-memory emulation

Every cell below completes on an 80 GB card emulating the given size (relax 4f1f49b and c8ac3e6, the
fixes below applied where noted):

| Cell | 16 GB | 24 GB | 32 GB | 40 GB |
| --- | --- | --- | --- | --- |
| Refine3D K1 5k/128 | 832 | 680 | 581 | 490 |
| Class3D K2 / K4 5k/128 | 163 / 183 | 109 / 134 | — | — |
| VDAM K1 / K4 5k/128 | 314 / 872 | 260 / 795 | — / 918 | — / 506 |
| Class3D K4 50k/256, 4 iterations | 340 | 258 | 237 | 237 |
| VDAM K1 50k/256, 8 iterations | 878 | 746 | 396 | 403 |
| Tomo Refine3D s1 | 1650 (main 5b44d0b) | — | — | — |
| Tomo Class3D et13 / VDAM K1 / VDAM K2 | 275 (main 5b44d0b) / 138 / 156 | — | — | — |
| PPCA tomo VDAM / SGD, default tile | 183 / 183 (main 23c3ebd, live-memory sampler on) | — | — | — |
| Refine3D K1 50k/256, 15 iterations | 16800 (main 7b1e4da) | — | — | — |

## Minimum compute capability

relax runs on compute capability 6.0 when its native libraries are built for it; P100 runs every workflow
above. The default build targets 7.0 to 9.0, plus PTX for later cards.
`scripts/build_test_natives.sh` takes `RELAX_NATIVE_CUDA_ARCH` for other targets. A library without the
card's architecture is refused when relax first loads it: recovar's preflight reads the targets from the
library's fat binary, with or without `cuobjdump`, and checks every library (recovar c60e3c26f). Before
that it failed mid-run with CUDA error 209.

## Issues

Fixed:

| Issue | Fix | Evidence |
| --- | --- | --- |
| Tomo coarse pass ran out of memory on 16 GB (6.4 to 7.6 GiB batch): fixed 2 GiB budget at 8 B a pixel, no device cap | `_coarse_batches` counts the bytes of the projection path that serves and is capped at a quarter of `device_available_bytes` (dbbb49c) | 16 GB emulation and P100 complete, GT FSC-AUC equal to H100 to 1e-6 |
| Final tau2 shell statistics ran out of memory at box 256 on 16 GB, after 15 iterations | `_shell_stats_on_host` sends them to the host when their device arrays exceed `_SHELL_STATS_DEVICE_SHARE` of the free memory (29e862a, f4ea273) | host and device equal at 512³; the failing case, Refine3D K1 50k/256 for 15 iterations at 16 GB, completes on main 7b1e4da (Slurm 14937566, 16800 s) |
| Preflight skipped without `cuobjdump`; one library's verdict exempted the next | fat-binary reader, per-library verdict (recovar c60e3c26f) | P100 and H100 refuse an unbuilt architecture in 4 to 6 s |
| A native build started inside a recovar checkout compiled that checkout | builds run in their output directory, and natives record the recovar kernel sources they compiled; `native_sources.py check` refuses any other (adfa7f8, 5d69a51) | unit test with a shadowing checkout |
| `relax ppca_initial_model` failed outside a git checkout | source identity records `head: None` (302748d3, ppcaspeed) | Polar runs start |
| A relax command or script started inside a recovar checkout imported that checkout's recovar | importing relax refuses a recovar outside the installed package or editable checkout; `RELAX_ALLOW_SHADOWED_RECOVAR=1` permits it (relax/__init__.py `_reject_shadowed_recovar`) | unit and subprocess tests (`python -m relax.commands.*` and `python -c "import relax"`) with a shadowing package |
| PPCA tomo at the default tile ran out of memory on 16 GB cards at the first radius-31 iteration (tile reader, then `_score_tile`) | the tile planner counts the block programs from XLA's memory analysis and the next tile's reader (main 06266b9b, ppcaspeed) | 16 GB emulation and a real P100 complete; P100 live peak 11.96 GiB against 12.33 GiB counted at r31. Unchanged on main 23c3ebd (P100 11.96 GiB, emulation 11.95 GiB against 12.36 GiB counted): the two later kept-buffer fixes (b3057db, 23c3ebd) leave the walls and the live peak of this fixture as they were at d1ba3e83 |
| EMPIAR-10202 (box 800) died in the final all-data pass after 6 h 17 min: the 15.35 GiB projector texture, a CUDA allocation outside XLA's pool, did not fit once the pool had grown (relax#17). The benchmark runner imported relax before naming the command, so the texture reserve was skipped and the pool limit stayed 0.90 | `reserve_for_refinement` is the one call an entry point makes before the backend starts, and the refinement driver refuses at start-up a backend whose pool limit leaves no room for the run's texture (`require_projector_texture_reserve`, main 7242fae). The reserve changes the pool fraction above box 548 on an 80 GB card, above box 320 on 40 GB, and for every box on 16 GB | probe on an H100: texture fails at 0.90 after a 58 GiB pool, succeeds at 0.77; same-node pair from iteration 24 (14986821): without the reserve the final pass fails, with it the run completes (iteration 1594 s against 1450 s, final pass 1797 s); on a P100 an in-process caller that skips the reserve is refused in seconds, and the reserved caller and the command line run |
| EMPIAR-10202 (box 800) in one process needed 502 GB of host memory (job 15036604) and was killed at 430G in the final half-map FSC (15003773) | the iteration boundary releases the pass outputs and the run-files snapshot, the rotation-posterior history is kept only for diagnostic runs (9360444d), the final prior keeps two host copies fewer (69db1aa3), and the expected-accuracy slab streams in chunks of at most 512 MiB (63678028) | whole run on relax 1af55661 plus the accuracy fix (job 15104150, H100): host peak 433.6 GB (VmHWM 423,390,852 KiB), reached in the final reconstruction's padded inverse FFT (1600³); iteration ends 209 GB (iteration 1) to 290 GB (iteration 26). RELION 5.0.1 used about 449 to 475 GB summed over its 3 MPI ranks (sacct TRESUsageInTot, jobs 14640954, 14475516). Request at least 480G for box 800 |

Open:

1. The V100 and A100 40 GB rows other than PPCA are pending: hardware busy (Polar's V100 and A100 nodes run
   multi-day workloads). Their cells stay queued and fill in when the nodes free up. Meanwhile the P100 (16 GB,
   an older architecture than V100), the A100 80 GB (sm_80) and the 16 to 40 GB emulation cover them.
2. PPCA tomo runs out of memory on a real A100 40 GB at main 23c3ebd (Polar 413542, both optimizers, after
   167 s and 119 s): at radius 32 the tile reader (`load_tilt_tile`) fails to allocate 8.48 GiB with 9.2 GiB
   in use of a 36.4 GiB pool. The byte count is right; with preallocation off the pool had grown in small
   regions and had no contiguous block (ppcaspeed, reproduced on an H100 limited to that pool). Fixed on
   main c6c35155: each new tile plan allocates and releases one block of its counted bytes, so the pool
   grows by one region of that size. Real-card reruns at 0b71d11d are running.
3. `relax refine` at box 800 with `XLA_PYTHON_CLIENT_PREALLOCATE=false` runs out of memory at iteration 13
   (relax#20): an 18.54 GiB probe buffer, counted by the plan (23.08 GiB of a 29.67 GiB budget), finds no
   block in a 60.9 GiB pool that thirteen iterations grew in regions of at most 16 GiB. JAX's default mode
   is the supported one above the reserve threshold; the error carries a note naming the setting. The whole
   run in the default mode is the gate (Slurm 15003773).

## Policy audit of the 2026-10-03 landings

| Change | 1. 80 GB no-op | 2. size-invariant | 3. one path | 4. small |
| --- | --- | --- | --- | --- |
| Tomo coarse batch sizing (dbbb49c) | on main the coarse pass takes the texture path (d28c258), where the 80 GB plan is the old one (`test_coarse_batch_plan_on_an_80gb_card_is_the_fixed_2gib_plan`); wall A/B below. History: between dbbb49c and d28c258 the complex128 fallback made 80 GB batches 4 times smaller (the old plan undercounted that path's 32 B a pixel) | `test_k_class_particles_cut_their_weights_over_every_class_jointly` (one flush versus one per particle); 16 GB run equals H100 | sizes only | relax/ +72 -18, one shared `_device_share_bytes` |
| tau2 host routing (29e862a, f4ea273) | routing test asserts the device route at 80 GB; wall A/B below | host versus device equality test, exact at 512³ | one helper, named threshold | relax/ +46 -11 |
| PPCA tile planner (ppcaspeed, 625259f8 to 06266b9b) | on H100 the plan is the requested 150 at every stage; plan test and wall A/B with ppcaspeed | end to end on H100 (main 7b1e4da): 16 versus 150 particles a tile differ by no more than two runs at 150 (mean correlation 1 − 4e-8 versus 1 − 3e-8, loading cosines ≥ 0.9999996 versus ≥ 0.9999997, log-likelihood within 2e-8 relative); unit tests with ppcaspeed | sizes only | planner in `full_row_stream.py` |

80 GB wall A/B (Slurm 14940497, one H100 node, two rounds with the GPUs swapped, cold compile caches):
main 7b1e4da against main with dbbb49c, 29e862a and f4ea273 reverted.

| Cell | 7b1e4da (rounds 1, 2) | reverted (rounds 1, 2) | Mean difference |
| --- | --- | --- | --- |
| Refine3D K1 5k/128 | 433.4, 419.2 s | 443.7, 419.3 s | −1.2% |
| Tomo Refine3D s1 | 1133.6, 1264.8 s | 1132.3, 1257.2 s | +0.4% |

Both are within 2%. The batch sizes and chunk counts in the two arms' logs differ only as much as two runs
of one arm do; the tomo pass-2 chunk counts follow the data-dependent significant rows.
