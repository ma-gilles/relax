# Extending relax: a checklist for coding agents

How to add a feature to relax, usually a RELION behaviour relax does not have yet. Each step links to the
page that owns the details; follow those pages, this one only gives the order.

## 1. Read first

- [ ] [AGENTS.md](../AGENTS.md): the rules (float32, RELION 5.0.1 by default, no tolerance changes, no
      silent fallbacks), the tier table and the code map. Read the `AGENTS.md` of every directory you edit.
- [ ] [CONTRIBUTING.md](../CONTRIBUTING.md): environment, natives, test tiers, landing, commits.
- [ ] The algorithm page for the area: [Refine3D/Class3D](math/relion_refinement_algorithm.md),
      [InitialModel](math/ab_initio_initial_model_algorithm.md), or the other pages in [docs/math](math/).
- [ ] [RELION defaults](development/relion_defaults.md) and [feature gaps](development/relion_feature_gaps.md).
- [ ] Work in your own worktree, off current `main`:

```bash
git fetch origin
git worktree add -b <you>/<feature> ../relax-<feature> origin/main
cd ../relax-<feature> && pixi install && pixi run doctor
```

## 2. Define the reference and the scope

- [ ] The oracle is RELION 5.0.1 ([the pinned oracle](development/em_parity_runbook.md#the-pinned-oracle-relion-501)).
      Cite the RELION source file and lines for every behaviour you reproduce.
- [ ] Write down the supported configurations (program, K, symmetry, optics, SPA or tomo, GPUs).
- [ ] Refuse every unsupported combination with an error that names it. Never fall back silently.
- [ ] A behaviour that differs from RELION on purpose is a named opt-in option (AGENTS.md rule 2).

## 3. Map the feature onto what exists

- [ ] Before writing code, list the relax and RECOVAR pieces the RELION code path uses: projector and
      backprojector, coarse and local search, scoring, reconstruction, noise and priors, STAR and map I/O
      (`relax/io/`, RELION sign and axis convention).
- [ ] Reuse them. Keep the new code small, and put each mode decision in one place.
- [ ] A change to shared RECOVAR code goes into RECOVAR and is repinned here
      ([Changing RECOVAR](../CONTRIBUTING.md#changing-recovar)); never copy or wrap it.

## 4. Build test data with known ground truth

- [ ] Generate synthetic datasets with RECOVAR's simulator
      ([API]](https://ma-gilles.github.io/recovar/reference/api/simulation/)), as relax's scripts do:
  - `scripts/prepare_relion_parity_benchmark.py`: K=1 SPA (`recovar.simulation.simulator.generate_synthetic_dataset`).
  - `scripts/prepare_pdb_k1_relion_sanity_benchmark.py`: maps from atomic models
    (`recovar.simulation.trajectory_generation.generate_trajectory_volumes`).
  - `scripts/prepare_cryobench_pdb_multiclass_relion_parity_benchmark.py`: several states, outliers
    (`recovar.simulation.simulate_scattering_potential`).
- [ ] Keep the ground truth: poses, shifts, states, CTF parameters, noise level, seeds.
- [ ] Start tiny and deterministic (few particles, small box, fixed seed), then go to realistic sizes.
- [ ] Use a realistic defocus spread (`dataset_params_option="dataset1"` draws CTFs from a real dataset's
      parameters); add contrast and noise variation, outliers and preferred orientations (`"kent"`)
      when the feature touches them.

## 5. Parity method

- [ ] Same inputs, random seeds and half sets in RELION and relax.
- [ ] Compare intermediate quantities in order and stop at the first divergence: geometry and conventions,
      then operands (CTF, masks, noise), scores, posteriors and poses, backprojection, maps, then the
      schedule (resolution, current size, convergence). See the
      [investigation loop](development/em_parity_runbook.md#investigation-loop) and
      [parity notes](math/relion_parity_agent_notes.md).
- [ ] The acceptance band is RELION's own same-seed repeat spread, not a number you pick
      ([benchmark contracts](development/benchmarks.md)).
- [ ] Never tune the final map toward RELION's. Fix the first quantity that differs.
- [ ] A RELION behaviour that looks like a bug: open an issue (label `relion-bug`) citing the source lines,
      and reproduce it on the default path.

## 6. Tests

- [ ] Unit tests for geometry and conventions (axis order, sign, Euler angles, FFT layout).
- [ ] A fixed-state oracle replay: one iteration from a saved RELION state, compared with RELION's next
      iteration (as `tests/integration/test_em_parity_fast.py` does).
- [ ] Then an end-to-end run on the synthetic data, scored against the ground truth and against RELION.
- [ ] Compare floats with `tests/helpers/float_compare.py`; keep integers and discrete logic exact.
- [ ] Never loosen a tolerance, threshold or baseline (AGENTS.md rule 3). A test that fails for a real
      reason stays failing until the code is fixed.
- [ ] Climb the [validation ladder](development/em_parity_runbook.md#validation-ladder) one rung at a time.

## 7. Land

- [ ] Chain fingerprints, then the smoke and medium tiers, then the landing gate, then `land_fp.sh`, then a
      fast-forward push: [CONTRIBUTING.md, Landing](../CONTRIBUTING.md#landing).

## 8. Report

- [ ] Record benchmark results as rows of
      [`tests/baselines/relion_vs_relax_benchmarks.json`](../tests/baselines/relion_vs_relax_benchmarks.json),
      rendered by `python scripts/render_benchmark_table.py` into the
      [table](benchmarks/relion_vs_relax.md) and [provenance page](benchmarks/relion_vs_relax_provenance.md):
      the exact commands, sources, inputs, seeds, hardware and resolution definitions.
- [ ] Keep four statuses apart: implemented, executed, quality-accepted and performance-qualified. A run
      that completed is not accepted; an accepted result is not a speed claim unless the arms are matched.
