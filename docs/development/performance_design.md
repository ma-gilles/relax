# Performance design

Read this guide for speed design and review, not unrelated changes. Existing
requirements remain in the [EM contract](../../relax/AGENTS.md),
[benchmark contract](benchmarks.md), [Della runbook](della.md) and
[CUDA contract](https://github.com/ma-gilles/recovar/blob/dev2/recovar/cuda/CLAUDE.md).
Current user instructions take precedence over these guides.

**Design goal:** deliver meaningful speed while keeping code easy to maintain
and modify, including through agentic development. A human should be able to
read and understand the architecture: stage responsibilities, module boundaries,
data flow and numerical ownership. Agent-written changes must remain human
reviewable. Prefer explicit interfaces, cohesive components and focused diffs
that make future changes easy to reason about.

## Optimize the complete workload

Judge designs by total runtime and peak memory on realistic datasets. Measure
startup, compilation, preparation and repeated computation across early, middle,
late and final iterations. Final claims require the applicable full-run gates.

**Longer JAX compilation in the first few iterations is acceptable when it has
little effect on realistic full runs.** Do not replace clear JAX code merely
to improve a tiny test or the first iteration. Compare cold-process time and
warmed throughput, record compilation-cache reuse, and estimate the break-even
workload. Repeated specialization matters when its cumulative cost materially
affects the full run; compilation count alone is not a failure.

Before editing, identify the measured bottleneck and its maximum possible
whole-run benefit. Distinguish CPU work, I/O, compilation, synchronization,
transfers and GPU computation; an idle GPU alone does not justify CUDA.
Exclude profiler overhead from speed claims. Budget retained arrays, construction
temporaries, other live buffers and allocator headroom.

## Iterate without hours-long experiments

1. Use existing relax and RELION logs to locate expensive early, middle or late
   iterations; separate initialization and finalization.
2. Replay a checkpoint through one iteration, half or stage. Verify the stopping
   boundary. Preserve precision, particle order, candidates, window sizes and
   execution route; subsets can change these.
3. Profile comparable stages in both codes, separating first-use compilation
   from repeated work. Mismatched probes explain mechanisms, not RELION ratios.
4. Change one bottleneck; run a focused numerical check and the same short replay
   against an immutable control. Run probes on Slurm under the current device
   rules (local GPUs only for the short smoke tier when the queue is busy), and
   matched Slurm pairs for long comparisons.
5. Check representative affected regimes. Reserve full dataset qualification
   for promising candidates, using uninstrumented matched timing runs. Short
   tests accelerate this loop; they do not replace scientific gates.

## Measure against RELION fairly

A speed ratio is only meaningful when both arms do the same work under the same
conditions. Mismatched setups have produced ratios that were wrong by 3-5x.

- **One job, same hardware, own CPUs.** Run RELION and relax in one Slurm job
  with one GPU each (`--gres=gpu:2`), same GPU model, and give each arm its own
  step and CPUs (`srun --exact -c N`). Arms sharing one task's CPUs inflated
  relax's VDAM wall about 3.5x; arms on different nodes or dates are not paired.
- **Same image I/O on both arms.** Use RELION's default reading, or stage both to
  node-local disk (`--scratch_dir`). `--preread_images` made RELION 4-6x slower
  per iteration on EMPIAR-10073, because every MPI follower holds the whole stack.
- **Same command and defaults.** Match RELION GUI defaults ([relion_defaults](relion_defaults.md)),
  including `--ini_high`, sampling and `--firstiter_cc`; a speed or quality gap
  under different settings says nothing about the engines.
- **Report the per-iteration split**, not only the wall: global iterations, local
  iterations and the final all-data pass behave differently, and the whole-run
  ratio hides where the time goes.

## Lessons from the device-resident K=1 engine

The resident engine took K=1 auto-refine from about 3-5x RELION's wall to
0.65-1.09x at equal quality (evidence: the timing-controlled rows in
[the benchmark tables](../benchmarks/relion_vs_relax.md) and
[em_status](em_status.md)). What mattered:

- **Keep candidates, accumulators and statistics on the device across a pass.**
  The previous engine's cost was host round trips, per-bucket eager dispatch and
  per-particle launches, not GPU arithmetic.
- **Use RELION's own GPU arithmetic path.** RELION's float32 texture projection
  was both faster and closer to RELION than a complex128 JAX fallback. Check which
  path actually ran (log it); a silent fallback cost a 2x slowdown and a 1e-4
  accuracy gap before anyone noticed.
- **Stream large per-iteration caches.** Stream what does not fit (chunk-local
  projections at high angular sampling) instead of caching it, as RELION does.
- **Budget memory from what is actually live.** Count padded copies and operands
  allocated after the free-memory reading, and read headroom from the JAX
  allocator pool, not only the driver's free memory: a grown pool looks "used" to
  `nvidia-smi`. A one-iteration replay starts with an empty pool, so validate
  memory budgets on a full run.
- **Keep shapes stable.** Recompilation for every new image count or current size
  dominated K>1 VDAM (about 21k compiles, 40% of the wall); a persistent compile
  cache does not help when shapes keep changing. Fixed capacity ladders, host-side
  slicing and fused gather programs remove it.
- **Remove per-call synchronization.** Rebuilding a texture and synchronizing the
  stream on every call left the GPU idle half the time; reuse stream-owned
  resources.
- **Default-mode fallbacks must be visible.** When a fast path falls back to an
  older engine, log the reason, record the engine used per iteration in the
  results, and make tests assert the intended engine ran.

## Choose the implementation layer

| Layer | Prefer it when |
| --- | --- |
| Python | High-level control, I/O policy and scheduling stay clear and their measured cost is modest. |
| JAX | Batched tensor operations express the math clearly and deliver suitable full-run speed and memory use. This is the default numerical layer. |
| C++ | Measured CPU preparation, irregular loops or repeated dispatch justify native execution or ownership of a larger operation. |
| CUDA through JAX FFI or C++ | Measured launch, layout, fusion, reduction or memory-traffic costs justify explicit GPU control. |

Prefer simple reuse, batching or transfer fixes when sufficient. Do not require
speculative JAX rewrites before a justified native implementation. C++ should
own meaningful work, not merely sit between Python and CUDA. Specify device,
stream, buffer lifetime, synchronization and errors at native boundaries.

Keep shared numerical code in its owning project: RECOVAR changes belong in
RECOVAR. Preserve production float32, deliberate higher-precision operations
and scientific semantics. Speed does not justify weaker numerical gates.

## Prefer maintainable speed

**Prefer simpler code over a modest speedup that substantially increases
maintenance cost.** Justify exceptions with meaningful full-run runtime or
memory savings; there is no universal percentage threshold. Consider expected
usage, build burden, debugging and validation cost, as well as the measured gain.

Prefer **one production path per supported backend** after a replacement is
validated. Preserve required CPU support and useful independent test references;
avoid maintaining two selectable production implementations solely for ongoing
benchmarking. Temporary comparison paths should have a removal condition.

Leave a short decision record alongside the change:

- Bottleneck, representative workload and immutable evidence.
- Chosen layer, simpler alternative and expected whole-run benefit.
- Compilation amortization, peak-memory estimate and numerical checks.
- Measured result, limitations, and when to reject or revisit the design.

Link detailed run artifacts instead of copying logs into this guide. Negative
results are useful evidence; revert complexity that does not earn its cost.
