# relax/sparse_pass2/: the device-resident pass-2 engine

The root guide and `relax/AGENTS.md` apply. This is the fine (oversampled) pass of the E-step and the M-step
accumulation it feeds. It is the only pass-2 engine and it needs a CUDA GPU.

## What this directory owns

| Module | Owns |
| --- | --- |
| `dispatch.py` | `compute_pass2_stats_sparse`: the one entry the controllers call; it routes to the resident drivers |
| `resident_pass2.py` | the drivers `compute_pass2_stats_resident`, `compute_k_class_pass2_stats_resident`, `compute_tilt_pass2_stats_resident`; the configuration gate `require_resident_production_configuration`; chunk memory planning |
| `resident_local_pass2.py`, `resident_local_layout.py` | local-search pass 2 and its adapter from `relax/local/local_layout.py` |
| `resident_candidates.py`, `resident_significance.py` | the candidate tables and their fixed-capacity chunks; device compaction of the coarse significance mask |
| `resident_operands.py`, `resident_scoring.py`, `resident_statistics.py`, `resident_tilts.py` | per-image operands kept on the device; the scoring stage; the float64 device accumulators; tilt images of subtomogram particles |
| `sparse_pass2_*.py` | shared pieces the drivers import: window, budgets, scoring, posterior, projection blocks, Wavg, adjoint, and `sparse_pass2_policy.py` (the environment switches) |
| `compile_ahead.py`, `engine_record.py` | compiling predictable programs on a helper thread; the per-pass engine record in the results |

## Rules

1. There is no fallback engine. A configuration the drivers do not implement raises
   `ResidentConfigurationUnsupported` from the gate, before any device work. Implement the configuration and
   extend the gate; never route around it.
2. Scoring and the fine posterior are float32. The statistics accumulators in `resident_statistics.py` are
   float64 on purpose; do not narrow them.
3. Compare this engine with the NumPy RELION E-step reference (`tests/unit/test_resident_relion_reference.py`)
   and with the same-source noise band, never with bitwise equality. Chunk partials change reduction order.
4. Keep program shapes static. Chunks are padded to a capacity ladder and the pixel axis is walked in
   fixed-size row blocks, so each program compiles once per capacity class. A shape that depends on the data
   recompiles on every chunk.
5. Give an environment switch one reader function, asked once per pass, as `sparse_pass2_policy.py` does; do
   not read the same variable in a second place. A new major policy is an option, not a variable.
6. A change to `relax/cuda/` used here is a native-source change: the tier rebuilds the natives, and results
   from a library built from other sources do not count (`python scripts/native_sources.py check`).

## Tests for a change here

- CPU, seconds: `pixi run python -m pytest -q tests/unit/test_resident_candidates.py
  tests/unit/test_resident_statistics.py tests/unit/test_resident_operands.py` (the last one's five GPU cases
  skip). They cover the host tables, the accumulator arithmetic and the operand layout only.
- GPU: `pixi run test-smoke`. It runs the fixed-state replays `k1_local_replay`, `k1_adaptive_replay` and
  `kclass_replay`, which all pass through this engine, plus the `tests/unit/test_resident_*.py` and
  `test_sparse_pass2_*.py` files that import the module you changed.
- On one idle local GPU 1-3, for a single file: set `CUDA_VISIBLE_DEVICES` to that GPU's UUID, then
  `pixi run python -m pytest -v tests/unit/test_resident_pass2_driver.py`.
- Any change to scores, posteriors, accumulators or reduction order is a numerical change: `pixi run test-medium`.

## Pitfalls

- A CPU run of the driver tests passes by skipping. `test_resident_pass2_driver.py` marks its GPU cases with
  `requires_resident_gpu`, a `skipif`, so a green CPU run has not exercised the scoring, posterior, Wavg or
  backprojection stages. Read the skip count.
- `_resident_pass2` is one function of 2,124 lines with 80 parameters, and `_run_resident_chunk` takes 71. An
  argument is threaded through `dispatch.py`, the gate and the driver; grep its name in all three.
- `dispatch.py` logs under the logger name `relax.helpers.oversampling`, not its own module name, because
  run-log collectors read that category. Do not "fix" it.
- The module docstring of `resident_pass2.py` still names `recovar.em.sparse_pass2.*` and
  `recovar.cuda_backproject.*`. The modules are `relax.sparse_pass2.*` and the kernels are in `relax/cuda/`.
- The translation is applied per pixel inside the fused scoring kernel. It differs from a pre-shifted tile
  only on the `ky = -N/2` Nyquist row of an unwindowed half image; the gate refuses the unwindowed case
  instead of handling it.
- `compile_ahead.py` jobs carry shapes and dtypes, never arrays, and a failed warm-up is dropped, not raised.
  A warm-up that changes what a run computes is a bug.
- A buffer that outlives its chunk is held on the device across the next chunk's allocation. Release at the
  existing boundary when you regroup results.
