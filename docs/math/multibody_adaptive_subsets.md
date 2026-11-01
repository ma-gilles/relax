# Adaptive subset reconstruction on the multi-body embedding

[`relax/multibody/subsets.py`](../../relax/multibody/subsets.py) and
[`scripts/multibody_adaptive_subsets.py`](../../scripts/multibody_adaptive_subsets.py) reconstruct one body from the
particles near a target point of the multi-body motion embedding, at many bandwidths, and let RECOVAR choose among
them. Nothing is realigned.

## Inputs

- A finished multi-body run, RELION's or relax's: `run_data.star` (every particle's body poses) and the last numbered
  iteration's half maps of every body (the references of the final pass; a body fixed earlier keeps its last maps).
- `relion_flex_analyse --PCA_orient --write_pca_projections` of that run (relax's `run_model.star` is a RELION model
  STAR, so the same executable reads it). Scores are joined to the consensus particle table by image name, refusing
  duplicate, missing or extra names. The scores are used as exported: the flex analyser's weighted, centred,
  unwhitened PCA of the residual body rotations and translations.
- Chosen components, a target point `z*`, the target body and the number of candidates.

## Procedure

1. Distances `d_i = ||z_i - z*||` (Euclidean, not squared) in the chosen components. Thresholds `h_1 < ... < h_K`
   common to both halves: from the smallest keeping 50 particles in each half to the largest distance (the
   all-particle endpoint), evenly spaced in the number of particles kept (every distinct value for a discrete
   coordinate). A subset is `{i : d_i <= h}` (`np.digitize(..., right=True)`), with each particle in its own half.
2. Each half's images of the target body with the other bodies subtracted at their final poses, with that half's own
   maps (the multi-body subtraction, `subtraction.subtracted_body_images`), norm-corrected and unmasked.
3. The target body's composed poses `A O^T A_rot90 R O` and refined offsets, a RECOVAR dataset in the body frame
   (centred on the body's centre of mass); the noise is the mean of the two halves' spectra of the maps' iteration.
4. `recovar.heterogeneity.heterogeneity_volume.make_volumes_kernel_estimate_local` with the `"square"` kernel: disjoint
   bins accumulated once and summed cumulatively, every candidate in both halves, the narrowest subset as the
   opposite-half target, and the `locshellmost_likely` selector and recombination, with the settings of RECOVAR's
   external-embedding wrapper (`compute_and_save_reweighted`: `locres_sampling = locres_maskrad = box * voxel / 20`, no
   mask edge).

Candidates are new reconstructions from the selected images at fixed body poses: point-pose estimates, not RELION's
soft-posterior body map. `--oracle-coordinates` (the simulated twist) and `--oracle-poses` (the simulated body poses)
are labelled diagnostics on the synthetic benchmark.

Checks: the cumulative candidates equal direct reconstructions of the same subsets to 2.5e-7 relative; the
all-particle candidate agrees with the multi-body body map at masked FSC-AUC 0.9995-0.9998.

## Synthetic results

`scripts/prepare_multibody_benchmark.py --deformation`: body 2 of the two-body 5nrl benchmark twists by up to
+-15 degrees (17 states) and a 35 A lobe of it hinges by `delta`: tied to the twist (case B), independent of it
(case C, 3 levels per twist) or absent (case A). 10000 particles. Multi-body inputs: RELION 5.0.1 (or relax) with
`--offset_range` and `--offset_step` divided by the pixel size (undoing relax issue #132) and `--sigma_ang 3.75`;
with the GUI command's settings the bodies do not follow the motion on these data. PC1 correlates with the true twist
at -0.998.

Masked FSC-AUC against the target state's body, in the region where it differs from the undeformed body
(`scripts/evaluate_adaptive_subsets.py`):

| Case, noise | Target | narrow | all | best (n) | selected |
| --- | --- | --- | --- | --- | --- |
| B, 0.1 | twist -15 | 0.717 | 0.817 | 0.950 (1723) | 0.947 |
| B, 0.1 | twist -11.25 | 0.662 | 0.893 | 0.973 (4143) | 0.961 |
| B, 0.1 | twist 0 | 0.623 | 0.984 | 0.984 (all) | 0.982 |
| B, 0.1 | twist +15 | 0.668 | 0.810 | 0.952 (2326) | 0.948 |
| B, 0.3 | twist -15 | 0.554 | 0.797 | 0.902 (2735) | 0.894 |
| B, 0.3 | twist -11.25 | 0.532 | 0.869 | 0.940 (4347) | 0.930 |
| C, 0.1 | twist -7.5, delta -15 | 0.580 | 0.851 | 0.851 | 0.850 |

In case A the all-particle map is best (0.984) and the selection keeps it within 0.003. The selection stays within
0.012 of the best candidate everywhere; a single bandwidth chosen from the whole-volume criterion picks all particles,
because the unchanging body dominates it. Case C shows the limit: a motion coordinate carries no information about a
deformation independent of the motion. relax-input and RELION-input runs agree within 0.006. The upstream poses and
PCA weights are data-derived, so the opposite-half criterion is not exactly unbiased; these results test its use
against known truth.
