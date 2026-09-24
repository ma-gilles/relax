# EM validation and oracle runbook

Read the section needed for the current test, submission or scientific review.
The [EM contract](../../relax/AGENTS.md) contains always-applicable rules.
These procedures remain mandatory when their scope applies; moving them here
does not waive a gate. Current work is in [EM status](em_status.md).

## Validation Ladder

Use the cheapest sufficient rung and advance only after it passes:

1. one/few-particle fixed-state dump replay;
2. focused unit test for the changed helper/path;
3. CPU fast guard: `pixi run test-em-fast-guard`;
4. GPU fast parity: `pixi run test-em-parity-fast`;
5. 5k/128 end-to-end K=1 or K-class smoke;
6. 10k-50k robustness cells at 128/256;
7. 100k/256 K=1 and K=4 completion pair, with RECOVAR and RELION for each pair
   run on the same GPU model.

Rungs 3-4 are the smoke and medium [test tiers](../../CONTRIBUTING.md#test-tiers)
(`pixi run test-smoke`, `pixi run test-medium`), and rung 7 is part of the long tier
(`pixi run test-long`); the change decides the tier.

During normal iteration, run the whole fast parity tier at most once every 3-4 hours
unless fixing that tier, changing its path, or doing final validation.
Prefer the directly affected test between tier runs.

relax has no SPA/ET pipeline suites; `recovar`'s own runners stay in the
recovar repository. When relax work changes recovar (the root guide's "Changing
recovar from relax work"), run recovar's applicable qualification in a recovar
checkout: `pixi run test-full`, `./scripts/run_tests_parallel.sh long-test` and
`scripts/extract_regression_tables.py`. Never run them for relax-only changes.

The EM long tier is Slurm-only. `pixi run test-long` runs it together with the
100k/256 completions; on its own:

```bash
./scripts/run_em_parity_long_slurm.sh
```

Completion evidence must use both K=1 and K=4 (exactly K=4, not a proxy), at least 100k particles,
at least 256x256 images, identical inputs/seeds/initial maps/masks,
and the same GPU class for RECOVAR and RELION. Completion runs are milestone
evidence, not edit-loop tests.

## Environment, GPU, And Scratch

Use the frozen pixi environment and import-provenance checks in
[CONTRIBUTING.md](../../CONTRIBUTING.md). Select CPU or allocated GPU visibility
before Python imports. Build custom CUDA explicitly for GPU qualification and
protect the recorded binary from runtime rebuilds as described there.

Before a short local GPU check, run `nvidia-smi` and do not use a device already
used by another person or process. On the user's four-GPU development machine,
leave physical GPU 0 free and use only idle physical GPUs 1, 2 or 3, at most three
in total. Set visibility by GPU UUID before Python imports or pytest collection.
On Slurm nodes, preserve the scheduler's allocation.
Use Slurm for multi-iteration, long, or contention-sensitive GPU work; cluster
jobs may be submitted broadly and allowed to queue. Compare RECOVAR and RELION
on the same GPU model within each timing pair; no single GPU architecture is
the universal oracle. Every sbatch job must set
`PYTHONNOUSERSITE=1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, unset contaminating
Python/conda variables, and create per-job runtime roots:

```bash
RUN_ID="${SLURM_JOB_ID:-manual}"
export TMPDIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/$RUN_ID/tmp"
export PIXI_HOME="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/$RUN_ID/pixi_home"
export RATTLER_CACHE_DIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/$RUN_ID/rattler_cache"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"
```

Put bulky disposable runs under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/<dated-run-name>/` and create a
`SAFE_TO_DELETE` marker at the run root. Do not put long-lived matrices under
the shared `_agent_scratch` roots. Keep long-lived EM source checkouts under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/`, not the quota-constrained
GILLES project filesystem. Preserve curated fixtures in place.

Particle images are read the way RELION reads them
([`relax/helpers/particle_io.py`](../../relax/helpers/particle_io.py)). Auto-refine,
Class3D and InitialModel stream batches from the original stacks by default, as the
RELION GUI does (no pre-read, no scratch). `--preread_images` holds every particle
in host memory. `--scratch_dir DIR` copies the referenced stack files to `DIR` at
start-up, checks the free space first and keeps `--keep_free_scratch` GB free,
reads from the copy and removes it at exit. On della, pass
`--scratch_dir /tmp`: `/tmp` inside a job is a private node-local NVMe xfs mount that
Slurm cleans up (28 TB on the cryoem H100 nodes, 5.9 TB on the A100 nodes, about 6.5 GB/s
direct reads and 6,400-6,800 random particle reads per second). The EMPIAR-10097 stack
lives on `/projects`, which is NFS: about 60 random particle reads per second, and 0.5 GB/s
for an uncached sequential copy. The exported `TMPDIR` above is GPFS, so recovar's implicit
`TMPDIR` staging never applies, and relax turns it off anyway unless `--scratch_dir`
is given. For speed comparisons, pass the same mode to both engines. RELION
takes the same flags.

## RELION Oracle Rules

### The pinned oracle: RELION 5.0.1

Every RELION reference, dump and score in this program comes from RELION
**5.0.1**. There is no pre-5 build in play anywhere, and none should be
introduced.

| Role | Identity |
| --- | --- |
| Source read when checking RELION behaviour | `/scratch/gpfs/GILLES/mg6942/relion`, version 5.0.1, git `f2c1a38` (2026-02-27) |
| Env-gated dump build (`RELION_DUMP_*`) | `/scratch/gpfs/GILLES/mg6942/relion/build_patched/`, compiled from that source |
| Reference refinements | module `relion/5.0.1/gcc-11.5.0-gpu`, and `/projects/MOLBIO/local/relion-5.0.1-gcc-11.5.0-cuda-12.6-rhel9-arch80/bin/relion_refine_mpi` |
| Sealed `relion_postprocess` used for every scored FSC | 5.0.1 |

The dump build prints a "RELION 3.1." line in some messages. That is legacy
text about column names renamed in 3.1, from `motioncorr_runner.cpp` and
`local_symmetry.cpp`, not a version stamp.

**Particle ordering must be the modern one.** 5.0.1 randomises the two
half-set orders in `Experiment::randomiseParticlesOrder` (`src/exp_model.cpp`)
with `std::mt19937` seeded by `random_seed + iter`, followed by a stable sort
on numeric optics group. Commit `f2c1a38` is where that replaced the older
libc `rand` / `std::random_shuffle` path. relax implements only the `mt19937`
order (the libc order and `--relion-particle-shuffle` were removed on
2026-09-24). The MOLBIO module build
(`relion/5.0.1/gcc-11.5.0-gpu`, whose STAR headers say `version 5.0.1`
without a commit) still has the libc path, so oracles written by it
(`em_fixtures/k4_fast_oracles/*`, the K4 100k dispatch oracle, the symmetry
matrix) are in the older order. With the wrong order the processing order
and the expected-accuracy trial particles differ from the oracle's, so
per-particle and half-map comparisons against it do not mean what they appear
to. This is measurable, not theoretical. On EMPIAR-10073 the legacy ordering
gives an unmasked resolution of 6.650052 A while the modern ordering gives
6.567953 A, which is exactly RELION's value on both of its repeats.

- Pin and record the RELION source commit, patched-build identity, complete
  command, STAR metadata, GPU model, MPI layout, and seed. Do not trust help
  text for GUI defaults; inspect `pipeline_jobs.cpp` and output model STARs.
- Fail closed on mid-trajectory restarted per-half captures. RELION MPI
  initialization can broadcast rank-1 `sigma2_noise` to every follower and
  overwrite a loaded half-2 curve. Either capture the trajectory
  uninterrupted or record the target random subset and prove shellwise that
  `CTF^2 * group_scale^2 / corr_img` matches that subset's
  previous-iteration model STAR before attributing any score difference.
- Use the shared env-gated dump build under
  `/scratch/gpfs/GILLES/mg6942/relion/build_patched/`; do not create another
  RELION clone. Coordinate before editing or rebuilding this shared resource.
- Load RELION MRCs with `recovar.utils.helpers.load_relion_volume`; the frame
  convention is `vol_recovar = -transpose(vol_relion, (2, 1, 0))`.
- `--healpix_order` means the coarse pass-1 order. Adaptive oversampling is
  applied after it.
- Auto-refine uses `tau2_fudge=1`; 3D classification and InitialModel use 4.
  Verify `_rlnTau2FudgeFactor` in the model STAR.
- GUI auto-refine includes `--firstiter_cc`. Strict oracle mode must reproduce
  its hard winner and pass-2 routing semantics. Quality mode may differ only as
  an explicit, measured policy decision.
- Current-size BPref half joins use the explicit RELION padding factor.
- K-class quality claims use the RELION x-half/current-size BPref path. Native
  half-volume K-class accumulation is diagnostic unless explicitly selected.
- Do not force K-class final-all-data after non-convergence. The strict-parity
  target specifies final gridding correction on. The reviewed PR158 source
  actually defaults it off; preserve that implementation during cleanup and
  record the effective setting. Resolving this scientific-policy discrepancy
  requires a separate, explicitly qualified change. Do not label the off path
  as satisfying the on-policy contract.
- Preserve shared contracts: `run_halfset_em_iteration` reads `state.Ft_y` and
  `state.Ft_CTF` after `finish_up_M_step`.

Detailed source findings and dump variables belong in
`docs/math/relion_parity_agent_notes.md`, not in this contract.

### Standalone K1 auto-refine launch

A K1 auto-refine that counts as qualification evidence reads only what
`relion_refine` reads: `<data_dir>/particles.star`, its stacks, the reference
map and the values on the RELION command line. For a real-data run the input
STAR may be a RELION InitialModel/VDAM `run_itNNN_data.star`, since RELION's
own auto-refine starts from it. The RELION auto-refine run is read only
afterwards, for comparison. Map the RELION command to `run_full_refinement.py`
as follows. A fresh K1 run given no RELION output is standalone by default.
relax's defaults are RELION's start-up methods and the RELION GUI's job
defaults ([audit](relion_defaults.md)), not relion_refine's command-line
defaults, so pass every value of the RELION command being reproduced,
including the `--no-...` forms where that command omits a GUI option.

| RELION | relax |
| --- | --- |
| `--split_random_halves --random_seed S` | `--seed S`; the half sets from the input are the default (input `rlnRandomSubset` if every row has one, else glibc `srand(S)`/`rand()%2+1` in micrograph order; groups from `rlnGroupName`/micrograph). Without `--seed` the time is used, as RELION does |
| 5.0.1 f2c1a3 particle order | default (mt19937) |
| start-up noise from the images | default (the only estimator) |
| `--particle_diameter D` | `--particle_diameter_ang D` (default 200, the GUI's) |
| `--ini_high H` / no `--ini_high` | `--init_resolution H` (the low-pass is on by default) / `--no-apply-initial-lowpass` |
| `--firstiter_cc` / none | default / `--no-firstiter_cc` |
| `--healpix_order`, `--offset_range`, `--offset_step` | same names with underscores |
| `--oversampling 1` / `0` | `--adaptive_oversampling 1` / `0` |
| `--tau2_fudge` (auto-refine default 1) | `--tau2_fudge 1.0` |
| `--perturb 0.5` (default) | `--perturb_factor 0.5` |
| `--maxsig` (default -1) | default, resolved as RELION does |
| `--offset 10` (default) | default `--offset_sigma_angstrom 10` |

Start-up norm corrections come from the input `rlnNormCorrection` and tau2
and `data_vs_prior` from `initialiseDataVersusPrior` on the low-passed
reference (see [the start-up state](../math/relion_refinement_algorithm.md)).
`--image-fourier-backend relion_cuda`, which the fresh K1 start requires, is the K1 default.

Harness entry points: `K1_TRAJECTORY_MODE=standalone` (the default) in
`scripts/run_em_completion_bench_slurm.sh`, the `[standalone]` cases of
`test_em_parity_fast_k1_coldstart` and `test_em_parity_long_k1_full`.

Debug-only starts, labelled as such wherever they appear: `--relion_init_dir`,
`--relion_half_sets <run_it000_data.star or a STAR carrying RELION's split>`,
`--relion_optimiser`, `--perturb_replay_relion_dir`, the completion modes
`autonomous` (RELION-seeded run_it000) and `relion-replay`, and the
`[relion_seeded_debug]` test cases. They pin RELION state for first-divergence
hunts and fixed-state replays; they are not standalone evidence.

## Benchmark Design And Reporting

High-resolution completion fixtures must come from target-grid PDB/mmCIF
scattering-potential volumes, not upsampled legacy 64^3 assets. Record source
coordinates, grid/voxel size, B-factor, noise model/level, CTF, class balance,
angle distribution, contrast/noise-scale variation, translations, outliers,
normalization, and seed.

Use `scripts/prepare_pdb_k1_relion_sanity_benchmark.py` for the canonical K=1
fixture and `scripts/prepare_cryobench_pdb_multiclass_relion_parity_benchmark.py`
for the canonical K=4 fixture. A K=15 run is useful stress coverage but is not the K=4
completion gate.

Broad quality claims require a matrix across dataset family, SNR/noise model,
K, class balance, uniform/preferred orientation distributions, CTF/no-CTF,
contrast/noise scale, translations, junk/outliers, seed, grid size, and
particle count. Use small cells to find failures; reserve 100k/256 runs for
milestone confirmation. Close synthetic K=1 trajectory parity first, then run
at least one well-characterized real-particle confirmation before K=4.

Complete aggregate state is compared every iteration. Candidate score surfaces
may use stratified sampling at scale, but automatically dump and investigate
every particle with a discrete, posterior, or convergence-relevant mismatch.

Every reported run includes:

- commit, dirty fingerprint, exact commands and environment overrides;
- fixture and RELION oracle identities;
- Slurm job IDs, node/GPU, logs, artifact root, `SAFE_TO_DELETE` status;
- FSC/FSC-AUC versus GT and RELION, Pmax, pose/translation, and K=4 class
  metrics as applicable;
- end-to-end and per-stage time, throughput, peak memory, compilation/warmup
  treatment, batch/microbatch sizes;
- comparison to the accepted run with every delta labeled better, worse, or same;
  use mixed or not measured only when no single directional label is valid.

Keep the active conclusion, evidence state and next check in
`docs/development/em_status.md`. Preserve detailed dated evidence in the linked
program/notes archives; update `docs/math/em_parity_best_metrics.md` only for
completion attempts. Do not paste large run histories into this contract.

## Investigation Loop

Use this order for quality bugs:

1. Find the first divergent iteration, half, class, particle, pass, and state
   field. Do not debug only the final map.
2. Replay the same fixed RELION state and candidate set. Compare raw scores,
   probabilities, best pose/class/translation, and accumulators.
3. If fixed-state arithmetic agrees, move one state boundary earlier. Treat
   the issue as trajectory history rather than changing the E-step kernel.
4. Confirm the relevant behavior in RELION source or an env-gated RELION dump.
5. Add a focused regression that fails for the demonstrated reason.
6. Make the smallest correctness change, rerun the focused case, then climb
   the validation ladder.
7. Record the result, including null or negative findings, before moving on.

Keep algorithmic changes separate from performance changes. A batching,
microbatch-cap, scheduling, layout, fusion, or precision change is
performance-only until equivalence against the accepted path is demonstrated.

For deep parity work, capture enough state to locate first divergence: raw
scores and posterior probabilities, all pass-1/pass-2/local candidates, best
pose/class/translation after each pass, priors, masks, noise accumulators,
`Ft_y`, `Ft_CTF`, BPref data/weight, maps, FSC, tau2, data-vs-prior,
current-size/resolution state, convergence state, and stage timings.
