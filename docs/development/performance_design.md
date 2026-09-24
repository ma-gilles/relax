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
   against an immutable control. Use available local GPUs for bounded probes
   under current device rules; matched Slurm pairs for long comparisons.
5. Check representative affected regimes. Reserve full dataset qualification
   for promising candidates, using uninstrumented matched timing runs. Short
   tests accelerate this loop; they do not replace scientific gates.

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
