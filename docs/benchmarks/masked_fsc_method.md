# Frozen masks and masked FSC

Masked FSC numbers are reported next to the unmasked ones for every scored real
and synthetic dataset. They are reporting only: no pass/fail gate reads them, and
changing a gate is a separate decision. Results: [masked FSC table](masked_fsc.md).
Implementation: `scripts/masked_fsc.py`; tests: `tests/unit/test_masked_fsc.py`.

## One frozen mask per dataset

Each dataset (and symmetry) has exactly one mask, generated once and used for
every arm, RELION and relax alike. Masks are never recomputed per run. The
registry [`frozen_masks.json`](frozen_masks.json) maps a dataset key to the
mask's `MASK.json` and SHA-256. `masked_fsc.py score` refuses a dataset that has
no registered mask, or whose mask file no longer matches its hash; it never
falls back to an unmasked number. `register` accepts only a mask whose
reproduction has been verified, and never replaces a registered mask.

Masks are curated artifacts (no `SAFE_TO_DELETE`) under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/<dataset>/masks/<key>/`, each
beside its `MASK.json`: exact commands, recipe parameters, source maps with
SHA-256, box, pixel size, RELION binaries with SHA-256 and version, mask
file/payload/header hashes, the sanity result and the reproduction check.

## Recipe

Masks are made with RELION's own tools from the build the references used
(`/scratch/gpfs/GILLES/mg6942/relion/build_patched`, RELION 5.0.1, read-only):

1. Source map: for real data the RELION reference run's final merged map; for
   synthetic data the ground-truth map (mean of the class maps for a
   multi-class fixture), taken from the relax-frame file because the
   simulator's RELION-frame maps carry negative density. Both file frames share
   one voxel layout (relax's file array is the negated RELION file array), so
   the mask applies to either.
2. `relion_image_handler --lowpass 15 --angpix <pixel>`.
3. Threshold: `median + 0.05 * (99.99th percentile - median)` of the low-passed
   map (a robust fraction of the positive density, insensitive to map scale).
4. `relion_mask_create --ini_threshold <t> --extend_inimask round(7 A / pixel)
   --width_soft_edge round(11 A / pixel)`: 5 and 8 pixels at 1.31-1.40 A/px, 3
   and 5 at 2.125 A/px, 2 and 3 at 4.25 A/px.
5. RELION writes a timestamp into MRC label 0 (header bytes 224-303); generated
   masks carry a fixed label instead, so regeneration reproduces the file's
   SHA-256 exactly.

This is the recipe of the 2026-08-31 RELION masked-FSC evidence for EMPIAR
10073/10345/10097; the regenerated masks have voxel payloads identical to those
masks.

Where the RELION reference already used a mask, that exact file is adopted with
the command that made it: HCN1 EMPIAR-10081 (C4; `--lowpass 15 --ini_threshold
0.005 --extend_inimask 3 --width_soft_edge 6`, RELION masked 3.70 A) and
EMPIAR-10202 set 6 (I1; the publication's 75-150 A radial shell, `--denovo`, 8
px edge). An adopted file keeps RELION's timestamp label, so its reproduction
check compares the voxel payload and the header outside label 0.

`masked_fsc.py verify-mask MASK.json --record` regenerates a mask in a scratch
directory and records the comparison. Sanity checks, recorded per mask: the mask
is within [0, 1], does not touch the box edge, has a soft edge, covers at most
half the box, and leaves none of the unfiltered source map's density above 3
sigma at weight below 0.5 (no clipped density).

## Masked metrics

Masked half-map resolution uses relion_postprocess, so it compares with RELION's
reported masked resolution. Both arms' unfiltered half maps are postprocessed
with the frozen mask and one protocol: `--force_mask --skip_fsc_weighting
--low_pass 0 --randomize_at_fsc 0.8 --random_seed 42`. `--force_mask` stops
RELION from reporting an unmasked resolution when the mask looks unhelpful.
relax maps are negated into the RELION file frame first.

- Masked resolution: `rlnFinalResolution`, the phase-randomisation-corrected
  masked FSC at 0.143 in RELION's convention (the last shell before the first
  value below 0.143). The scorecard's sustained three-shell 1/7 crossing of the
  same curve is recorded beside it; older tables quoted that value.
- Masked FSC-AUC: normalised trapezoid of the corrected masked FSC over the
  real-data scorecard's jointly resolved band (shell 1 up to the earlier of the
  two arms' sustained unmasked half-map crossings), the same band as the
  unmasked scorecard AUC, which is reported beside it.
- Cross-engine masked FSC: both merged maps (and each half pair) multiplied by
  the frozen mask, FSC-AUC over the same band. Ground-truth masked FSC for
  synthetic data is computed the same way against the RELION-frame ground truth.
  Both are null when the maps do not share a frame, detected by a correlation
  inside the mask below 0.5 (for example a standalone run in a different pose
  basin); they are not aligned.

A row with no half maps on disk is masked-null with the reason recorded. Class3D
rows have no gold-standard half maps, so masked half-map resolution does not
apply to them.

## Commands

```bash
python scripts/masked_fsc.py make-mask --dataset KEY --symmetry C1 --source-map MAP \
  --source-frame relion --source-role "RELION reference final merged map" --pixel-size A --out-dir DIR
python scripts/masked_fsc.py verify-mask DIR/MASK.json --record
python scripts/masked_fsc.py register DIR/MASK.json
python scripts/masked_fsc.py score --dataset KEY --label NAME \
  --relion-maps MERGED HALF1 HALF2 --relax-maps final_merged.mrc final_half1_unfil.mrc final_half2_unfil.mrc \
  [--gt-map GT_RELION_FRAME.mrc] [--provenance PROVENANCE.json] --out-dir RUN
python scripts/masked_fsc.py collect RUN/masked_fsc.json
python scripts/masked_fsc.py render            # --check in CI
```

`--provenance` records a JSON object as the row's provenance and the rendered page lists it. Rows whose relax
maps were transformed before scoring use it: the 2026-09-24 rows scored runs whose final all-data pass skipped
RELION's gridding correction (`final_all_data_grid_correct` false in `refinement_results.npz`) on a copy of
`final_merged.mrc` divided by RELION's radial sinc^2 (the unfiltered halves were always corrected), which equals
the corrected reconstruction to float32 rounding
(`tests/unit/test_relion_functions.py::test_final_gridding_correction_equals_post_hoc_division_of_uncorrected_map`).
