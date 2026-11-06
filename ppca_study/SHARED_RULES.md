# Shared rules for comparing PPCA ab-initio methods

For any agent comparing PPCA ab-initio variants with ours, so that results from different branches are directly
comparable. Edit it if something needs changing, and say what you changed in the commit message.

Paths are on Della. Datasets and scoring code are readable by the cryoem group and by ry2228. Write your own outputs
under your own scratch folder (for example `/scratch/gpfs/CRYOEM/gilleslab/ry2228/...`), never into ours.

## 1. Datasets

Synthetic sets, all under `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_suite_b150_20261102/datasets/`. Use the
box-64 versions (`__b128crop64`, 6 Å/px): Fourier crops of the box-128 builds with the field of view kept.

| set | what it is | noise | latent dim q |
|---|---|---|---|
| `C__realistic__noise{1,3}` | 11 related states | 1 and 3 | 10 |
| `H1__realistic__noise3` | one state (a pose-only check) | 3 | 10 |
| `Ribosembly__realistic__noise3` | 16 assembly states | 3 | 10 |
| `D{10,20,40}__realistic__noise1` | one distinctive protein at 10/20/40 % plus 99 background proteins | 1 | 2 |
| `T10__realistic__noise1` | 10 unrelated proteins | 1 | 9 |
| `C__realistic_shift__noise{1,3}`, `Ribosembly__realistic_shift__noise3` | the same with in-plane shifts, 2.5 px sd (15 Å) | | as above |

- All sets: no solvent correction, uniform rotations, B 150.
- Each set's `evaluation/` folder holds the truth (labels, rotations, maps). Use it for scoring only, never in training.
- The `_solv`, `_pref`, `_b0`, `_shift6A`, `_shift30A` and `_shiftcom` variants are diagnostics, not comparison sets.
- On the D sets, judge the distinctive protein, not the 100-protein average.

## 2. Protocol

- Random start; seed 11 for a first comparison, seeds 12 and 13 before any claim about the D sets, which are seed-sensitive.
- 3000 updates of 300 images (150 + 150 halves).
- Box-64 stages:
  - HEALPix order 1 to update 125, order 2 to 1625, order 3 (7.5°) from 1626.
  - Band radius +1 shell every 125 updates, from shell 11 to shell 31 at update 2500.
- Oversampling 0 unless oversampling is what you are testing.
- Write down the pass rule before you run (in a PREDECLARED file next to the runs). Label any use of the truth beyond
  scoring as "oracle".

## 3. Scoring

Use metric suite **v3.2**: `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/tools/metrics_suite_v3_2/`.
`REVISIONS.json` there explains each revision. v3.2 fixes a v3.1 bug where frames of pseudo-symmetric proteins could
land 180° off. Don't use earlier revisions or the older per-state-FSC scorer for comparisons.

**How to run** (copy the scripts; don't run them in our folders):
- Templates are in `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/metrics_suite_v3_1_bestv1_20261105/`:
  `entries.tsv`, `gpu.sbatch` + `worker.sh` (GPU stage), `score_v32.sbatch` (CPU scoring).
- `entries.tsv` has one line per row: `name`, checkpoint path, dataset path, grid. The grid is `protocol`, or the
  shift-sized grid for the shifted sets.
- Stage 1, GPU: inference and reconstruction per row, about 5-20 minutes per row on an A100.
  - Shifted sets use the wider shift grid and take about 2.7 h per row.
  - Make the job fail if the worker stops with work remaining (exit 3), and start scoring only `afterok`.
- Stage 2, CPU: about 25 minutes per C/T10 row, up to 3 h per D row.
- **Input:** the suite reads PPCA checkpoints in our driver's format (`checkpoint_NNNN.npz` with `theta`). It also has
  an explicit-poses-plus-latents mode and a decoded-maps mode. If your branch writes a different format, convert to
  one of these rather than changing the suite.

## 4. What to report (fixed; don't add or rename metrics)

For every run, from the suite report:
- **Map scores:** decoded-class / per-image / BP-class FSC-AUC (shells 1-31), and BP gain.
- **References next to every map score:**
  - mean only (`metric1.references.mean_only`);
  - the true model (`true_<set>` row).
  
  A score is only meaningful between those two. For example, on C mean only is .837, so .83 there means no
  heterogeneity was recovered.
- **Pose:** the suite pose, median / p90, in the PPCA-map and BP-map frames.
- **Latents:** N_struct, and II z→E.
- **D sets:** the distinctive-protein and population-weighted lines. **T10:** the per-protein decoded scores.
- **Tables:** `tools/report_tools/row_table.py REPORT.json ...` prints rows in this form.
- **Figures:** raw PC volumes with their size relative to the mean; latent scatters in PC coordinates (not only UMAP).

Not headline numbers:
- The direct-fit pose, which uses the truth to set the frame. It's a diagnostic; label it as one.
- `summary.json` pose and the old scorer's energy, which have misled us before.

## 5. Real data

No ground truth, so all of this is descriptive:
- mean-map FSC against the published map (11846: use a map made from PDB 8EKI, not a back-projection at published poses);
- pose agreement with the published refinement (diagnostic);
- raw PCs with their scale;
- latent scatters in PC coordinates and an all-particle UMAP (state the settings);
- MRC exports of the mean and of mean ± 2 sd along each PC.

Say whether a run used published translations (pre-centring) or ran fully ab initio.

## 6. Known traps

- **Our driver caps runs at 6000 updates** unless you pass `--iterations`.
- **Our driver doesn't save the PC gate's moving averages in checkpoints,** so a resumed run differs from an
  uninterrupted one by about 3% over 500 updates. Run uninterrupted, or label the resume.
- **One seed is not enough on D10/D20:** the same method passes some seeds and fails others.
