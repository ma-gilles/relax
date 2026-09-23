# VDAM/PPCA three-state pilot scorecard

> Historical RECOVAR-source evidence. No RELAX-source 5k PPCA final quality or paired runtime is yet measured.

This pilot uses 20,000 particles at box 64 and does not assess the K1/K4 completion gates.
No numerical recovery threshold has been selected.

Training manifest SHA256: `0bc91e4b98e4739cac378897dfd50afbada2fa2fba62fed91286db2be888613f`. STAR SHA256: `b7c24bfb5d1ec8c0e877ba078ac9883148d430cb8fe8eefd6b40b4f3b7b3a696`.

| Evidence | Current result |
| --- | --- |
| Affected CPU / EM guard / final delta | 170 / 99 / 47 passed |
| PPCA GPU tests | 65 passed |
| Corrected projector CPU tests | 49 passed |
| CPU resume | exact |
| GPU bitwise resume | failed; controlled same-state GPU repeat differs in fine backprojection before loading, while checkpoint roundtrip is exact |
| Controlled GPU checkpoint replay | job 14307910: checkpoint fields exact; first differing fine backprojection; same-state repeat and restored continuation both vary on GPU; fixed-statistics relative L2 below existing 2e-5 float32 bound |
| No-custom-CUDA replay | job 14294639: completed diagnostic; bitwise mismatch persists with custom CUDA disabled (theta max 7.07e-9, first 1.49e-8, second 2.26e-6, noise 4.55e-13) |
| Late grid feasibility | job 14292869: completed on 12-particle cold-start diagnostic, A100 80GB; 790.1 s for 12 particles, 16883 MiB peak, 1195680 fine rotations |
| Matched A100 continuation | job 14293635: completed on A100 80GB; iterations 2 and 3; iteration 2 756.8 s / 532136 fine rotations; iteration 3 124.9 s / 131784 fine rotations |
| Coarse-only throughput | job 14294837: 24.7 s for 48 pilot particles (72173 selected samples); fixed tiny-trained state |
| Representative box64 update | job 14307408: 2,000 late-stage particles in 972.8 s; 2453553 coarse retained, 8122176 fine rotations; 2583 MiB peak sampled |
| Final embeddings | job 14307696: 2,000 particles in 420.5 s; matched 8-particle warm complete/embedding-only 9.55/3.18 s, bitwise-equal embeddings |
| Final embedding cold/warm | job 14307750: 2,000 particles 402.2/273.0 s; identical embedding/ID hashes |
| First trajectory feasibility | about 29.6 A100 GPU hours including final update and embeddings; fixed-state scaling, not a performance bound; includes final all-20k update and final all-particle embeddings; evolving support unmeasured |
| K3 tiny one-iteration diagnostic | job 14293103: completed; mixed diagnostic: float32 scoring, complex64 projection consumption and float32 M-step; corrected projector gridding/FFT/power actually float64 before cast; native double bootstrap bypassed by seed-map override |
| K3 tiny corrected float32 route | job 14295622: completed on assigned GPU; diagnostic one iteration on 12 particles; seed maps exact, projector StableHLO f64 ops 0 |
| End-to-end evaluator on 12-particle fixture | job 14294234: completed; 3 per-state rows, diagnostic only |
| Evaluator with corrected float32 K3 maps | job 14295823: completed; 3 per-state rows using corrected float32 K3 maps, diagnostic only; not threshold-qualified |
| PPCA seed 11, 20k | not measured |
| K3 seeds 21–23, 20k | not measured |
| PPCA seeds 21–23, 20k | not measured |
| Per-state scientific quality | 0 measured rows |

Scientific acceptance: **not threshold-qualified; no completed recovery measurements**.

## Separate 2,000-particle exploratory pilot

The [small-pilot handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_small_pilot_handoff.md) records the original pair and confirmed input-sign repair. These are separate from the 20k rows above.

Seed 11; 60 iterations; state counts [679, 656, 665]. Training manifest SHA256: `bed5a2647f2e071bd20700dfb0a5490aa7d819c7138b4d3520a60129ee20aca6`. Subset-index SHA256: `f865f13721dd70737e4d7a12acf83382aa590e8b07b2a8048761e8cdd6f6b802`.

First-pair shared-frame active FSC AUCs: PPCA [0.1421, 0.1215, 0.0723]; K3 [0.29, 0.4209, 0.3128]. The PPCA input sign was wrong; these are diagnostic only. The corrected pair completed all-particle final updates and PPCA embeddings.

| State | Corrected PPCA shared-frame AUC | Corrected K3 shared-frame AUC |
| --- | ---: | ---: |
| 0 | 0.4555 | 0.2899 |
| 1 | 0.6011 | 0.4219 |
| 2 | 0.2512 | 0.3134 |

Original package allocation: 1.594 A100 GPU hours, including probes, tests and the repaired follow-up.

### Current per-state shape comparison

The user clarified that common frame/hand is diagnostic, not a recovery requirement. The [six-column panels and protocol](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_visual_comparison.md) use independent rigid rotation/translation/reflection for both methods and Hungarian matching after all nine K3/GT fits. Older shared-frame assignments above remain historical evidence.

| State | PPCA active FSC AUC | K3 active FSC AUC | K3 class, 1-based | K3 mass |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.9875 | 0.4168 | 3 | 0.9% |
| 1 | 0.9863 | 0.9474 | 2 | 66.8% |
| 2 | 0.9851 | 0.9846 | 1 | 32.3% |

CPU job 14315426: completed, no retraining or GPU allocation. FSC uses shells 1–8 (48 Å nominal cutoff). PPCA and K3 state-0 local fits exhausted their evaluation budgets; full fit flags and all-pair scores remain in the [report](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_ppca_visual_comparison_20260923/comparison/report.json) (SHA256 `90cdfd0914455d726f166c80fba3bffdc7c746591b49e6660f81721f1f0a085b`). PPCA uses evaluation-label coordinate averages; K3 learns its classes. The comparison does not establish a general method ranking.

Scientific interpretation: supports recovery of three coarse PPCA shapes in this seed/subset after independent rigid fitting; K3 has two strong matches and a nearly empty class; shared frame/hand is diagnostic; no agreed recovery threshold or 20k approval.
