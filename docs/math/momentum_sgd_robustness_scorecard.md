# K3 Momentum SGD robustness scorecard

**Status: condition-dependent recovery; general robustness is not established.** The [report](../benchmarks/momentum_sgd_robustness_20260928.md) explains the protocol and limitations. This table preserves every condition, seed, and GT state, including poor outcomes. The [standing JSON](momentum_sgd_robustness_scorecard.json) records all 16 arm-level results: per-state FSC, correct-state pose and recall, calibrated shifts, pass-2 class fractions, class-pair FSC, noise range, maps, timing, source and input hashes. Full FSC curves remain in the frozen analysis (`/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_robustness_20260928/analysis_conditions/combined_scorecard.json`; outside the repository).

The frozen run records use the historical arm ID `cryosparc_sgd`; “Momentum SGD” is the current display name. FSC is the mean of signed, independently rigid/hand-registered map FSC over shells 1–15. Values are GT states **0 / 1 / 2**. All-particle pose medians come from the last visited minibatch STAR poses; they are not a final-map inference. Recall is the fraction of a GT state assigned to its matched class.

| Condition | Seed | Native FSC 0/1/2 | Momentum FSC 0/1/2 | Native pose ° | Momentum pose ° | Native recall 0/1/2 | Momentum recall 0/1/2 | Momentum/native wall |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: |
| ½ noise variance | 11 | 0.151/0.865/0.952 | 0.541/0.905/0.705 | 4.3 | 5.1 | 0.001/0.997/0.999 | 0.992/0.998/0.379 | 1.005 |
| ½ noise variance | 12 | 0.163/0.882/0.955 | 0.965/0.838/0.591 | 4.1 | 4.0 | 0.001/0.996/0.998 | 0.994/0.958/0.117 | 0.993 |
| 2× noise variance | 11 | 0.202/0.224/0.233 | 0.242/0.253/0.281 | 131.5 | 132.7 | 0.001/0.002/0.994 | 0.448/0.182/0.419 | 0.998 |
| 2× noise variance | 12 | 0.197/0.170/0.251 | 0.240/0.236/0.240 | 131.8 | 131.2 | 0.002/0.004/0.992 | 0.211/0.310/0.553 | 0.998 |
| Preferred angles | 11 | 0.185/0.174/0.701 | 0.554/0.917/0.955 | 98.8 | 5.1 | 0.001/0.001/0.991 | 0.965/0.992/0.999 | 1.006 |
| Preferred angles | 12 | 0.208/0.193/0.799 | 0.665/0.912/0.948 | 97.3 | 5.2 | 0.002/0.003/0.988 | 0.949/0.981/0.996 | 1.002 |
| Nonzero shifts | 11 | 0.207/0.186/0.239 | 0.243/0.318/0.239 | 131.6 | 129.7 | 0.003/0.001/0.997 | 0.254/0.567/0.080 | 1.001 |
| Nonzero shifts | 12 | 0.162/0.177/0.246 | 0.227/0.271/0.251 | 132.4 | 127.7 | 0.001/0.001/0.998 | 0.120/0.836/0.099 | 1.003 |

The shifted-particle condition has all-assigned calibrated median shift errors of **4.341/4.235 px** for native and **3.156/3.479 px** for Momentum SGD in seeds 11/12. Its all-assigned angular medians remain 128–132°, so the shift values do not establish successful joint pose and shift recovery. Correct-class, pose <10° shift subsets contain only 3–6 particles per arm in this condition; their medians and counts are in JSON.

Pass-2 posterior class fractions are normalized only **after the inherited coarse significant-support selection** (cap 300). They are not unpruned coarse occupancy or the uniform class prior written to STAR. No class map is an exact duplicate, but that alone does not rule out poor separation; the JSON retains class-pair FSC and individual occupancy. The centering diagnostic (`/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_robustness_20260928/analysis_conditions/centering_diagnostic.md`; outside the repository) is a raw-array **density-energy centroid** (`Σ index·V² / ΣV²`), not a physical center of mass or a quality gate.

All 16 arms finished 600 float32 updates without nonfinite map or noise failures. Initial maps matched inside each condition/seed pair, saved particle/halfset hashes matched at twelve 50-update checkpoints, and the A100 arm placement was swapped for seed 12. Per-update hashes between saved checkpoints were not retained. The scientific source was frozen `ae1165193c3d75afa64ffb1d6524e85b75237538`; the name-only branch was not used to produce these maps. The [uniform noise-1 r3 scorecard](momentum_sgd_science_scorecard.md) used source `f81e815e33b160511526a580e48905bb3c8ae229` and is context, not a source-matched causal contrast.
