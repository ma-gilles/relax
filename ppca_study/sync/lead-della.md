# lead-della (Princeton Della + Polar)

## 2026-11-12 ~03:30 UTC

- RESULT (CAHRA dataset 1, scored against the released truth, EMPIAR-13973; fixed 200k subset): relax K-class
  ab initio at RELION InitialModel defaults (trained on the 200k) beats every PPCA arm on clustering: ARI .167 / .131
  / .107 at K 4 / 6 / 10 vs .083 for our best PPCA arm (images masked). GDH separates at every K, β-amylase partly,
  ADH only at K 10; aldolase and IgG never get a class. ARI is still rising at iteration 200; 600-iteration runs are
  queued. PPCA latents separate GDH and β-amylase (AUC .87-.91), not the other four (.68-.75).
- RESULT (synthetic): Ribosembly_shift n3, VDAM then SGD stage 3: .797 (mean only .780, true .868), the first shifted
  n3 row above mean only, still partial (66% < 10°). BEST_PRACTICE.md v1.73.

## 2026-11-12 ~01:00 UTC

- RESULT (synthetic shifted sets): relax K-class ab initio at RELION InitialModel defaults (K 1 / 4, seed 11) fails
  all 7 shifted sets (poses 97-126° except C_shift n1 K1 55°); the default ±6 px / 2 px search never recovers the
  offsets (error stays > 2 px from a 2.9 px start). relax matches real RELION 5 here (C_shift n1 K1 54.8° vs 54.9°),
  and both solve the centred twins (2.8-3.6°). BEST_PRACTICE.md v1.72.
- NOTE: v1.71 corrected. The C_shift pre-centred PPCA row (.662) is a hand split across states, not a centring bug.
- CLAIM (update): CAHRA dataset 1 against its released truth (EMPIAR-13973): per-species oracle references and a
  classify-against-known-maps ceiling, then single-factor PPCA arms and relax K-class K 4/6/10.

## 2026-11-12 ~00:10 UTC

- RESULT (synthetic shifted sets, suite v3.2, seed 11 unless noted; refs mean only / true): "VDAM-simple, then SGD
  stage 3 (all particles, ±4 px recentred)" is the best known route on C_shift n1 .887 (.837/.979), C_shift n3 .857
  (.837/.948), D20_shift .936 (.849/.991), Ribosembly_shift n1 .878 (.780/.937). The D10_shift subset route needs
  the 10% subset: at 20-30% only 1 of 3 seeds poses. D5_shift fails at every setting. Details BEST_PRACTICE.md v1.71.
- NOTE: always report pose next to decoded score; one D5_shift row decodes .870 with a 89.5° pose median.
- CLAIM: EMPIAR-11846 and 10097 with the consensus map held at full resolution for the whole run, RELION-style image
  masking, and a RELION-style refinement control (realshift, addendum 10). CAHRA dataset 1 scored against its released
  ground truth (EMPIAR-13973 source.star species labels).

## 2026-11-11 ~23:30 UTC

- NOTE (owner decision): EMPIAR-10180 dropped as a shift test case. Its shared offset is a one-off of the deposited
  stack (fixed in the image frame on every micrograph; never re-extracted centred). The 10180 fixed-mean tests are
  closed: design 2 (full-resolution search from update 1) gives 7.0°, 69% < 10°, the same as design 1.
  BEST_PRACTICE.md v1.70. Real-data shift work moves to other datasets.

## 2026-11-11 ~22:45 UTC

- RESULT (real, EMPIAR-10180 as picked, box 64, seed 11, 3000 updates; pose = direct fit to the RELION consensus,
  DIAGNOSTIC): global recentring solves it truth-free. One constant offset from the mean particle image's centre of
  mass (57.5 / 58.3 Å; consensus 55.4 / 55.2 Å) is removed from every image, then recipe v2 with the wide window
  (±6 px, prior held 20 Å). Mean-map FSC .958 (as picked .274, wide window alone .599, pre-centred .973); pose 5.3°,
  77% < 10° (pre-centred 4.3°, 91%). PCs match the pre-centred run's (|PC|/|mean| .16 .14 .12; top-3 subspace
  cosines .95 .92 .85). The ~20% miss tail is the window edge: miss rate .06 at <2 px residual offset after
  recentring, .50 at 8-10 px, .85 beyond 10 px. Holding the mean at the consensus map (±18 px grid then local
  recentring) is worse (7.0°, 69%) and puts a mean-mismatch mode into PC1. Details: BEST_PRACTICE.md v1.69.
  Relevant to tilt-czii's 70k picks (~31 Å offsets): check first whether the picks share a constant offset.
- DONE: BEST_PRACTICE.md v1.69.

## 2026-11-11 ~20:45 UTC

- RE [flatiron-filament] (ASK 17:30, reporting format): yes, report here in your own format. SHARED_RULES section 1
  (suite v3.2, mean-only / true references) governs single-particle suite rows only. For your rows use predeclared
  pass rules written before the run, synthetic truth tables, and the null you already show (bare-core gain
  distribution); state the relax sha and flags per row. Keep it methods + numbers, as you do (no data identifiers
  from the unpublished set). The only cryo-ET handoff is `best_v1/CRYOET_PROMPT.md` (recipe pointers, not a protocol);
  the closest peer is tilt-czii (relax `--ios`), so post cross-cutting tilt code notes as NOTE lines both can see.
  Keep your relax changes on your feature branch; nothing goes into `best_v1/` / `best_v2/` without a RECIPE line.
- RE [tilt-czii] (ASK 20:20, relax-native recipe v2): yes, useful. CLAIM: one C n1 row, seed 11, relax
  `claude/ppca-likelihood-min-shell` a5105206 with your flags vs the driver row (`best_v2/drivers/coupled_arm_v2.py`
  8294284e, same seed, same set), suite v3.2 on both; pass = per-update loss and step traces match to float tolerance
  through update 1626 and suite cells within seed noise. Queued behind the 10180 arms (cryoem GPUs busy); result here.
  Two things to check meanwhile, from our side: (1) the driver's flags you did not list (`--sgd-pc-gate-floor` is
  there; also the PC-only prior and the explicit data-measured diameter: relax `--diameter` sets the 3D support only,
  too small or too large broke two of our real sets); (2) on "adding the true offsets made the map worse": on
  EMPIAR-10180 our first offset fit came out with the opposite sign (a sign and px-vs-Å convention, scale -4.55 before
  the fix), so check sign and units before concluding the stack geometry is off.
- NOTE: no new results from lead-della since 18:30; 10180 fixed-mean arms (F1s, C) are running, F2s waits on F1s.

## 2026-11-11 ~18:30 UTC

- DONE: `best_v2/` added: recipe v2 code with RECIPE_v2.md (exact flags per use case), drivers byte-identical to the
  runs behind the reported rows (`SHA256SUMS`), stage-2 subset tools and example arms. Known gaps, stated in RECIPE_v2.md:
  the D10_shift stage-1 run at seed 11 used an earlier driver build that is no longer on disk (seeds 12-13 and all later
  stages used `coupled_arm_shift.py` a3cc9708); `stage2/infer_all.py` imports the metric suite from a Della path;
  `coupled_arm_fft2_ablate.py` imports an FFT engine from a Della path only when `--fft-sched` is used.

## 2026-11-11 ~18:00 UTC

- NOTE: sync protocol set up (`../SYNC.md`). Doc and code here are the standard; BEST_PRACTICE.md is at v1.68.
- NOTE: code on this branch is behind. `best_v1/` holds the recipe v1 driver (11-05). The recipe v2 driver and the
  shifted-set route drivers are being assembled into `best_v2/` (with RECIPE_v2.md, exact flags and shas); expected
  2026-11-11/12. Until then, recipe v2 = `best_v1/coupled_arm.py` flags plus `--fixed-step 0.10 --os-from-update 1626
  --os-rotations-only` and an explicit, data-measured particle diameter; see BEST_PRACTICE.md section 0.
- CLAIM: EMPIAR-10180 fixed-mean tests (mean = consensus map held for the whole run; full ±18 px grid then local
  recentring; full-resolution-search variant; pre-centred control) and a truth-free global recentring arm (10180's
  picks share a ~55 Å image-plane offset; the mean particle image's centre of mass recovers it to 0.3 px). Running.
- CLAIM: suite scoring of the shifted-set rows (D5/D10/D20_shift, Ribosembly_shift n1/n3, C_shift stage-3 rows) and
  the PPCA-on-pre-centred C_shift row that decides whether our C_shift gap to cryoDRGN (.840 vs .942) is offsets or
  model. Running.
- CLAIM: relax K-class ab-initio baselines (relax defaults, K=1/K=4) on the 7 shifted sets. Running.
- RESULT (summary of v1.66-1.68, details in BEST_PRACTICE.md change log): D10_shift solved by the three-stage route
  (suite .907-.910 at seeds 11-13; mean only .669, true .998); D5_shift signal-limited (both optimizers fail with and
  without selection); real data: the wide held-prior window closes most of the as-picked gap on 10073 / 10345 / 10076,
  not on 10097 / 11846 / 10180.
- NOTE: on 10345 the as-picked runs' PC1 is about c (R.mean - mean) (42-49% of the PC; R 134-158°): part of the
  particles are fitted to a rotated copy of the mean. The pre-centred run's PC1 is mostly the plain mean (amplitude).
- RESULT (2026-11-12, BEST_PRACTICE v1.74): stage-3 SGD does NOT transfer to real data. Seed 11, as picked, mean-map
  FSC 1-31 vs the consensus map; refs are W (wide window, best as picked) and B (pre-centred ceiling); no mean-only /
  true refs on real data. VW = VDAM-simple + wide window (best_v2/drivers/coupled_arm_fft2_ablate.py 0037f627);
  S3 = SGD stage 3 seeded from VW (coupled_arm_shift.py a3cc9708 + seed_offsets.py f0d810ae).
  10073 VW .942 / S3 .956 (W .963, B .973); 10345 .894 / .908 (W .916, B .930); 10076 .914 / .941 (W .947, B .949);
  10097 .829 / .792 (W .709, B .853). Rule needs 3 PASS, got 1 PASS / 3 TIE. VW alone is the best as-picked 10097 row
  (closes 83% of the W-B gap); there its PC1 is a per-image grey-level mode (z'1 vs image mean, Spearman .96).
- NOTE: cryoDRGN-AI C_shift seeds 12/13 show no hand split (93.5% / 92.5% < 10 deg in one frame); the seed-11 split
  was seed-specific. Suite rows pending.
- RESULT (2026-11-12, BEST_PRACTICE v1.75): EMPIAR-11846 as picked defeats a held full-resolution mean (selection
  particles 14.4 deg, the other 12,900 at 123 deg) and real RELION 5.1.1 auto-refine from the published map at 40 A
  (.240 / 129 deg; pre-centred ceiling .679 / 14.0 deg): it needs re-extraction. An image mask (RELION-style soft circle)
  helps the wide window on 10097 (.709 -> .768) and is neutral elsewhere; VDAM-simple wide stays best on 10097 (.829).
- RESULT (diagnostic, suite pending): relax K=1 at defaults breaks between 5 and 10 A offset sd on C_shift (pose 9.3 deg
  at 5 A, 25.8 at 10 A, 117 at 22 A); our VDAM-simple training pose stays ~5 deg to 22 A.
- NOTE: coupled_arm_v2.py stops logging training rotations once oversampling starts (update 1626 in v2os1); read
  v2os1 poses from the suite only.
- RESULT (2026-11-12, BEST_PRACTICE v1.76): CAHRA relax K-class at 600 iterations levels off at ARI .15-.17 (K4 .174,
  K6 .161, K8 .150; best PPCA .083); class maps are mask-filling speckled balls, not proteins, so the separation is
  from coarse 2D features. Species were imaged on separate micrographs, so micrograph-level features can leak labels.
- RESULT (2026-11-12, BEST_PRACTICE v1.77): CAHRA oracle RECOVAR (poses from per-species refinements, box 64) separates
  the six species (macro-AUC .999) vs our PPCA .791: resolution and the linear model are not the limit, pose/frame is.
  Template re-picking: NO-GO (single-particle detection ~2 sigma; even oracle templates fail the null test).
- NOTE: relax PPCA CUDA kernels cap q at 16 (basis size P = q+1 <= 17).
- RESULT (2026-11-12, BEST_PRACTICE v1.78): shift ladder on C_shift n1, suite decoded / pose (mean only .837, true 1.000):
  SGD recipe v2 .869 / 5.1 deg at 5 A, .846 / 5.1 at 10 A, .801 / 6.3 at 15 A; VDAM-simple .875 / 4.6 at 5 and 10 A.
  CAHRA: label-free classification against six per-species reference maps gives ARI .211 (54% own class) - barely above
  relax K-class (.174); the oracle-reference ceiling is low, partly from junk-blurred references.
