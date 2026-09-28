# Coarse K3 momentum SGD: implementation and first comparison

The opt-in optimizer supplies a scalar-curvature momentum map update and a
discounted noise estimator through the existing RELAX engine. Native VDAM
remains the default. Positivity is off. The method follows the 2017 cryoSPARC
paper with the documented adaptations in [the formulation](../math/cryosparc_sgd.md);
it does not reproduce current proprietary cryoSPARC software.

## Outcome and interpretation

Matched fixed-HP3 runs recover state 2 well in both seeds. Neither optimizer
reliably separates states 0 and 1. SGD is higher on every per-state FSC in these
two seeds, but outperforming VDAM is not the acceptance requirement. The target
is a correct implementation and robust behavior under controlled perturbations.
These results support implementation validity and repeatability on one fixture;
they do not establish broad robustness or three-state recovery.

FSC values are unmasked mean FSC over shells 1–15 after independent rigid/hand
fits and one-to-one class matching to the three exact transformed GT volumes.
GT is evaluation-only.

| Seed | VDAM GT0 | SGD GT0 | VDAM GT1 | SGD GT1 | VDAM GT2 | SGD GT2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | .190189 | .313329 | .164577 | .301398 | .884294 | .977046 |
| 12 | .154521 | .293602 | .173180 | .281066 | .891961 | .978009 |

The [standing scorecard](../math/cryosparc_sgd_science_scorecard.md) retains all
three rounds, including failed small-grid starts. Correctly assigned state 2
particles have median angular error 3.7 degrees in SGD and 3.9–4.0 degrees in VDAM;
SGD state 2 recall is .935/.990 versus .997/.998. States 0/1 have poor poses.
Class occupancy is a separate diagnostic and is not inferred from the fixed
uniform STAR prior.

Registered, equally band-limited map panels:
[seed 11](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/analysis_full_r3/seed11_registered_3row_band15.png),
[seed 12](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/analysis_full_r3/seed12_registered_3row_band15.png).
Each panel uses its own 2nd–98th percentile display scale, so compare morphology,
not absolute amplitude. Full [FSC curves and round comparison](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/analysis_full_r3/r2_vs_r3_comparison.md)
and [timing/protocol audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/analysis_full_r3/timing_summary.md) remain beside the runs.

## Matched protocol and measured cost

20,000 balanced ribosome particles, box 64, pixel size 6 Angstrom, simulator noise
variance multiplier 1, measured probe SNR .0322, known CTF, unit contrast,
zero translations and uniform orientations. Signed GT uses solvent correction
 a=.8/B=2000. Both arms use K3, seeds 11/12, 600 updates, batch 300 throughout,
oversampling 0, fixed HEALPix 3 (36,864 rotations), 29 shifts, radius schedule
15x100,24x200,31x300, uniform class/direction priors and production float32.
SGD learning rate .4 is fixed. Initial maps and sampled particle/halfset hashes
match; seed 12 swaps arms between the two A100 GPUs. This is a controlled native
VDAM configuration, not its default adaptive schedule.

Whole-update medians are .86–.88s at radius 15, 1.08–1.10s at 24 and 1.35–1.36s at 31.
All four 600-update arms take 770–776s including startup and checkpoint output;
there is no material measured difference between optimizers. Job 14629884 used
1555s on two A100s, .86389 allocated GPU-hours. The detailed audit separates
first-use compilation, checkpoint writes and startup. The separate instrumented
posterior dump is excluded from timing.

## Validation and source identity

Science source: `f81e815e33b160511526a580e48905bb3c8ae229`, frozen at
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relax_cryosparc_sgd_r3_frozen_20260928`.
Integration source: `ae1165193c3d75afa64ffb1d6524e85b75237538`, frozen at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/tier_integration_r4/src`.
RECOVAR pin: `5514ac6e2cb63ce1e9d1d88662f80dc4a4a8e70c`.
Native-source fingerprint: `211d2f14684542eaf652f1a107875f2eb1cd204735de09678f620f01496e3032`.
[NATIVE.json](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/native_e2401c4c/NATIVE.json) records binary identities.
The intervening main rebase removes an unused exact-local VDAM path; active
adaptive functions and moved numerical helpers were checked for AST equality.
Later tomography-only upstream changes are outside this execution path.

- Independent map-gradient/scaling check, known-target descent, noise statistics,
  duplicated-batch and empty-class tests passed. The gradient check includes
  projector gridding correction and native FFT coordinate scaling; relative
  error was approximately 1.5e-7. The update is a fixed-responsibility
  reconstruction surrogate, not the exact masked marginal-likelihood gradient.
- Medium job 14629153 on f81 passed 27/27 items: 10,938 tests, zero failures,
  five optional unit skips, no required parity skip. Twelve approved FSC/Pmax
  cases passed; K1 5k converged at 12 iterations and stayed in the RELION band.
  [Extracted tables](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/tier_medium_r2/extracted_fast_tables.md).
  The K4 nonadaptive replay has no approved FSC floor; its recorded .66139 AUC
  is not a passing FSC claim. A100 pinned-output comparison was not configured.
- First medium job 14627542 failed setup/stub/import/inventory/report checks;
  these were repaired without changing scientific tolerances. Its failure is
  retained in the registry. Subsequent numerical checks passed.
- Fresh integration job 14632067 completed the smoke tier on ae116519 and
  swapped two-seed A100 20-update source comparisons. Cross-source map FSC AUC
  is at least .999997, largest latest-batch mean absolute Pmax difference
  1.40e-5. Two of 300 seed 11 latest poses differ; saved outputs lack competing
  winner margins, so their individual cause is unresolved. Agreement is close,
  not bitwise. A same-source repeat provides context, not a new tolerance.
  [Detailed source comparison](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/integration_pilot_r4/source_invariance_report.json).
- Focused post-rebase CPU checks passed. Existing upstream whole-file formatting
  and five Ruff import-order findings were identified as unchanged baseline
  findings; unrelated files were not reformatted.

Documentation mirrors, added local links, JSON identity and whitespace checks pass.
The documentation build command cannot run because this checkout has no tracked
`mkdocs.yml`; the failed build log is preserved with the run artifacts.

## Evaluation corrections and remaining limits

The original pose report transposed the native Euler matrix incorrectly.
An independent compiled native-projector check against saved simulated images
established the corrected convention. Clean-image median cosine is .9945 with
the native convention and .0386 with the old transpose. A second reporting fix
scores classes containing 1–19 particles in primary pose metrics. The optional
particle-fitted-frame diagnostic still needs 20. Both changes leave map fits and
FSC unchanged. The [v4 report](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/analysis_pose_bridge/comparison_pose_v4.md)
and `evaluation_pose_v4.json` files are authoritative. All 12 runs cover 19,996
of 20,000 saved particles; four were unvisited and have class 0. STAR poses/classes
are last-visited minibatch estimates, not fresh final-map inference. Shift error
is not qualified because its map-registration/origin boundary remains to be checked.

The inherited coarse support cap is 300 joint hypotheses per image. Pass2
renormalizes within that support, so metadata named `full` describes only that
selected support. The [iteration 1 mass diagnostic](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/analysis_coarse_mass/REPORT.md)
measures .41–.53 percent retained mass for SGD and 3.44–11.44 percent for VDAM on
four sampled images. This is a significant shared approximation; it is not
proof of why a particular state fails. No support-cap override has been added.
General SGD restart is not implemented because momentum/noise accumulators are
not serialized for continuation.

## Reproduction and next experiment

The original paired script and wrapper are immutable. To repeat into a NEW
output directory, use a new dated root and preserve the frozen source/fixture:

```bash
RUN_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_repeat_YYYYMMDD
mkdir "$RUN_ROOT"
touch "$RUN_ROOT/SAFE_TO_DELETE"
sed "s|^OUTPUT_ROOT=.*|OUTPUT_ROOT=$RUN_ROOT/full|" \
  /scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/jobs/full_r3.sbatch > "$RUN_ROOT/full.sbatch"
bash -n "$RUN_ROOT/full.sbatch"
squeue -u "$USER"
sbatch --time=00:35:00 --output="$RUN_ROOT/full-%j.log" "$RUN_ROOT/full.sbatch"
```

The wrapper refuses existing nonempty arm directories and verifies native and
Python import identities. Evaluation scripts and full invocation manifests are
in `/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/protocol_r3` and `protocol_pose_v4`; reruns need new evaluation output
paths as well. Do not overwrite the saved protocols, runs or evaluation files.

The next user-requested package freezes this algorithm/settings and compares
noise variance factors .5/2, preferred orientations, and nonzero shifts, each
with two matched optimizer seeds. Results will live at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_robustness_20260928`.
No claim of robustness is made before those comparisons. Posterior-support
sensitivity is a separate follow-up so it cannot confound this matrix.

All jobs from this first package are terminal; job 14627542 failed as recorded,
all subsequent required checks completed. [Job registry](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/JOBS.json)
records exact status and allocated cost: 5.44861 GPU-hours
including the failed qualification, pilots and diagnostics.
