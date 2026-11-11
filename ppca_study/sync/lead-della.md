# lead-della (Princeton Della + Polar)

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
