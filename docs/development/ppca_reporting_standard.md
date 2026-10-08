# PPCA run reporting standard (owner rule, 2026-10-07)

Every PPCA run and seed that is reported (suite, realistic fixtures, real data such as EMPIAR-10076)
gets the outputs below. A table of numbers alone is not a result.

## Required per run and seed

- **Volume trajectories.** Signed mean and every PC at initialization, early checkpoints and the
  final update. Show orthogonal central slices. Keep contrast fixed across time within each channel,
  and state each channel's scale.
- **Latent trajectories.** Re-infer the same fixed particle subset at every checkpoint. Show
  coordinate-pair scatters coloured by evaluation labels when they exist. For large q add pooled
  PCA-2 with the retained variance. For q = 1 use histograms.
- **Final embeddings.** Infer **all particles with the final model**, keeping the original particle
  IDs. Export the latents and scatter/density plots. For real data also make a CPU UMAP; document any
  landmark fitting and check that label ordering matches particle IDs.
- **Metrics against updates / image presentations.**
  - Synthetic: per-state FSC, latent R^2 and calibrated pose errors.
  - All runs: noise, mean and PC power, update size and timing.
  - VDAM: plot the mean gates and the PC gates separately.
  - Keep common-frame FSC and independently aligned per-state FSC apart.
- **Downloads and report.** Signed float32 MRC files of the means and PCs, the embeddings, the
  manifests and ZIP archives. Include partial and failed runs. Put a compact table with plot links (a Markdown page with image paths, not HTML) on
  the dataset's results page and link it from the synthetic and real-data index.

## Interpretation

Ground truth is for evaluation only. Plot sharpness, spectral power and UMAP clusters on their own do
not show that a structure or state has been recovered.

## Existing recipes (experiment-specific; adapt their paths and rank assumptions)

- Synthetic scoring and plotting conventions:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_pose_artifact_controls_20261003/analysis/README.md`
- All-particle embeddings, volume exports and the real-data UMAP recipe:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_challenge25_20261003/evaluation_salvage_20261004/HANDOFF.md`
- Plotting script:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_challenge25_20261003/evaluation_salvage_20261004/plot.py`
