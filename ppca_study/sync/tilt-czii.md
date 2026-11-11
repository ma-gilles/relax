# tilt-czii (CZII Biohub HPC)

## 2026-11-11 ~18:10 UTC

- NOTE: access. CZII Biohub HPC, Slurm, H100 / H200 nodes, 4 GPUs at a time. Data: real cryo-ET tilt series of a
  phantom sample with five annotated species (thyroglobulin, beta-galactosidase, apo-ferritin, beta-amylase, ribosome),
  as RELION 5 2D-stack subtomograms (box 128, 3.37 Å/px, about 28 tilts per particle). This includes the annotated
  per-species particle sets and a 70,032-pick thyroglobulin template-matching set with heavy junk (4,620 of the picks
  sit on an annotated thyroglobulin, 2,484 on another annotated species). Plus matched synthetic tilt-series sets built
  from those maps. The data are not public. No metric suite v3.2 here and none of the Della synthetic sets; the
  synthetic suite is not my focus (see below), so I am not asking for it yet.
- NOTE: focus. Tilt-series (subtomogram) PPCA in relax (`relax ppca_initial_model --ios`, algorithm doc section 16)
  on real cryo-ET data: species mixes, junk, the 70k pick set. One pose (R, 3D t) and one latent per particle, shared
  by its tilts; tilt geometry fixed from the tilt-series alignment (same as RELION Refine3D / Class3D on 2D stacks).
- NOTE: low-frequency runaway on real tilt data, and a fix on a feature branch. VDAM-simple
  (`--optimizer vdam --no-vdam-second-moment`, box 128, physical band to 12.4 Å, `--q 4`, batch 300, ±4 px step 2
  shifts, seed 11) on a real ~9k-particle thyroglobulin / beta-galactosidase mix: mean power flat to about update 1100,
  then +3 orders of magnitude concentrated at the zero frequency of the mean; the run stops at ~1350 (indefinite metric
  or a nonpositive noise variance in shells 0-1). An independent user hit the same failure with the same command. It
  appears on ~9k particles and not on 70k because VDAM's step factor depends on the set size (about .55 vs .09 at
  update 1500). Fix: `--likelihood-min-shell K` (image shells below K get no likelihood weight; model and noise held
  there), branch `claude/ppca-likelihood-min-shell` (fead10da, = relax main + the tilt mixed-tile performance work +
  this flag; 6 unit tests). With K = 2 the same mix runs through update 2000.
- RESULT (real, thyroglobulin, poses against the annotated poses as a DIAGNOSTIC; thyroglobulin is C2 here, errors to
  the nearest copy; box 128, seed 11, `--q 4`, ±4 px step 2 shifts, oversampling 0):
  - 4,620 annotated picks, 0 / 20 / 50% added real-background junk, VDAM-simple + `--likelihood-min-shell 2`,
    update 2000: median 7.2° / 7.2° / 7.2°, 62-63% < 10°. 80% junk, update 1000: 11.9°, 41% < 10°.
  - All 70,032 picks, VDAM-simple without the fix, update 1400 (HEALPix order 2 only): 12.4° on the annotated subset,
    40% < 10°.
  - Junk score (per-tilt log-likelihood ratio) on real junk: AUC .65-.75; on synthetic junk .96-.98.
  - Latents, 70k run: annotated thyroglobulin forms its own UMAP lobe and other-species picks a separate cluster;
    an oracle 5-fold logistic regression on z gives AUC .83 for thyroglobulin vs the rest (pmax .65). Oracle = labels
    used for scoring only.
- RESULT (real, maps by back-projection at the recovered poses, half-map FSC 0.143 / FSC 0.143 vs the published
  average): annotated thyroglobulin, centred, published poses (ceiling) 9.6 Å / Nyquist; the 70k picks with the
  TRUE annotated rotations and no shift 28.7 / 25.4 Å; the same picks with PPCA poses 28.7 / 25.4-28.7 Å at 0-80%
  junk. So on these picks the map limit is the pick stacks, not PPCA. The recovered 3D shifts carry no information on
  the picks' true offsets (median 31 Å; R² .001 in any frame). Adding the true offsets made the map worse (33 Å), so
  the stack geometry is being checked before any wide-window run.
- NOTE: real-data memory. On the 70k set (2.0M tilt images) every run and reconstruction so far has run out of host
  memory while loading (128 / 256 GB), with the image preread on or off. `load_tilt_training` passes `lazy=False`
  to the dataset loader. Being profiled; a lazy-loading fix would go on the feature branch above.
- CLAIM: all 70,032 picks, VDAM-simple + `--likelihood-min-shell 2`, `--q 4` and `--q 10`, seed 11, 24 h each
  (expected to reach update ~2300), after the memory fix. Start 2026-11-11/12.
- CLAIM: junk separation from the latents (truth-free selections from z, scored against the annotation), and the
  stack-geometry check on the 70k picks.
- NOTE: not claiming any synthetic shifted-set work; the real 70k picks are as-picked with ~31 Å offsets, so the
  shift program's wide held-prior window is the obvious next arm here once the geometry check passes.
