# K3 momentum SGD science scorecard

Scientific status: **partial state recovery**; three-state recovery and broad robustness are not established. The original run scorecard labels this “experimental partial, not accepted”; that describes full scientific recovery, not a failure of the implementation checks. See the [validation report](../benchmarks/momentum_sgd_coarse_20260928.md). FSC and pose values below are raw per-seed results; there is no passing cutoff. Pose metrics use the v4 fixed map-frame scorer: classes with 1–19 saved particles count in primary coverage, while the optional fitted-frame diagnostic still needs 20. STAR poses/classes are last-visited minibatch values. Conditional pose applies only to correctly assigned GT-state particles, so read it with recall and unconditional medians. R1 pass2 posterior class fractions were not saved; its model STAR class fractions are learned priors, not measured occupancy. R2/r3 posterior fractions are pass2 class mass **after significant coarse selection**, not global full coarse-posterior occupancy. The engine metadata suffix `full` means full pass2 distribution before M-step retention; coarse-discarded mass was not saved.

The table uses “Momentum SGD” as the current display name. The immutable r1–r3
JSON records retain their historical optimizer identifier `cryosparc_sgd`.

| Round | Seed | Arm | FSC GT0/1/2 | Full pose median ° (n) | Fixed600 pose median ° (n) | State2 conditional median ° (n; recall) | Pass2 fractions after coarse selection | Wall s | GPU job |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | --- |
| r1 | 11 | native | 0.156/0.234/0.209 | 131.8 (19996/20000) | 129.8 (599/600) | 134.9 (122; 0.018) | NA | 337.1 | 14627612 |
| r1 | 11 | Momentum SGD | 0.178/0.239/0.220 | 130.6 (19996/20000) | 128.6 (599/600) | 130.9 (4965; 0.745) | NA | 348.7 | 14627612 |
| r1 | 12 | native | 0.194/0.232/0.212 | 130.8 (19996/20000) | 132.7 (600/600) | 130.4 (4953; 0.743) | NA | 289.1 | 14627612 |
| r1 | 12 | Momentum SGD | 0.025/0.197/0.065 | 130.2 (19996/20000) | 132.2 (600/600) | NA (0; 0.000) | NA | 284.5 | 14627612 |
| r2 | 11 | native | 0.188/0.216/0.253 | 130.1 (19996/20000) | 126.3 (599/600) | 129.7 (6626; 0.994) | [0.000,1.000,0.000] | 254.8 | 14628694 |
| r2 | 11 | Momentum SGD | 0.232/0.287/0.534 | 123.7 (19996/20000) | 119.9 (599/600) | 39.9 (2187; 0.328) | [0.517,0.097,0.386] | 254.4 | 14628694 |
| r2 | 12 | native | 0.165/0.223/0.238 | 130.1 (19996/20000) | 131.1 (600/600) | 128.0 (6639; 0.996) | [0.000,0.000,1.000] | 251.1 | 14628694 |
| r2 | 12 | Momentum SGD | 0.232/0.246/0.673 | 106.3 (19996/20000) | 108.8 (600/600) | 10.8 (3650; 0.548) | [0.108,0.842,0.050] | 253.4 | 14628694 |
| r3 | 11 | native | 0.190/0.165/0.884 | 84.4 (19996/20000) | 82.3 (599/600) | 4.0 (6649; 0.997) | [0.000,1.000,0.000] | 774.5 | 14629884 |
| r3 | 11 | Momentum SGD | 0.313/0.301/0.977 | 83.1 (19996/20000) | 78.5 (599/600) | 3.7 (6233; 0.935) | [0.270,0.723,0.007] | 776.2 | 14629884 |
| r3 | 12 | native | 0.155/0.173/0.892 | 58.2 (19996/20000) | 67.2 (600/600) | 3.9 (6651; 0.998) | [1.000,0.000,0.000] | 774.0 | 14629884 |
| r3 | 12 | Momentum SGD | 0.294/0.281/0.978 | 91.2 (19996/20000) | 92.2 (600/600) | 3.7 (6602; 0.990) | [0.683,0.303,0.014] | 770.1 | 14629884 |

All three state-specific conditional scores, assignments, hashes, and full/fixed600 coverage are in [science_scorecard.json](momentum_sgd_science_scorecard.json). [R3 registered comparison](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/analysis_full_r3/r2_vs_r3_comparison.md) provides matched map panels and FSC curves. The [separate coarse-mass diagnostic](/scratch/gpfs/CRYOEM/gilleslab/em_work/cryosparc_sgd_coarse_20260928/analysis_coarse_mass/REPORT.md) has its own instrumented runtime and is excluded from paired timing.
