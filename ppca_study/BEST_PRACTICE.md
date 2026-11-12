# PPCA ab initio: current best practice

Version 1, 2026-11-04. Branch `best-practice` of `ma-gilles/ppca_study` (private).

This is the standard way we run, score and report PPCA ab-initio work. Read it before starting anything. It changes
only by an owner decision, recorded in the change log at the bottom with the date and the evidence. Anything that
departs from it is an experiment: label it, predeclare it, and compare it against the standard.

Status words:
- **STANDARD**: use it as written.
- **REFERENCE**: the thing to beat.
- **OPEN**: not settled; the experiment that will settle it is named.

## 0. Current best recipe (v2 decisions 2026-11-08; see change log v1.61)

v2 = v1 (SGD, per-voxel H^-1 direction, PC gate floor .25, uniform orientation prior, VDAM step decay, box-64 stages, ±2 px shifts for centred data) with: a fixed step T x H^-1 G (T about .08-.10, no lr matching); an explicit, data-measured particle diameter (required). Options, not default: FFT all-shift search for uncentred data; slow start. Not adopted: final RECOVAR solve.

### Previous: recipe v1 (2026-11-05)

`best_v1/RECIPE.md` gives the full recipe, code and evidence. In short: plain SGD (momentum_sgd) with the per-voxel H⁻¹ direction, the PC gate with a .25 floor, a uniform orientation prior and the VDAM step decay; translations from the exhaustive absolute grid (±48 Å, 12 Å then 6 Å steps, prior centred at 0 with σ held at 20 Å); box-64 stages with no oversampling. **Validation (2026-11-06, suite v3.1):** the exhaustive grid is rejected for SGD. It loses on C n3 (.832 against .897), Ribosembly (.164, no poses), T10, D20, D40 and shifted C n1, and holds only on C n1 and H1. v1b (SGD + relrfine) is being scored. Oversampling from HEALPix 3 passes its predeclared test on C n1 and C n3 (+.018 / +.015 decoded), a candidate for v2. Real data: 10076 recovers the structure (FSC .80). 11846: the first reference map was wrong (a pose-convention mismatch in our back-projection); against the deposited model 8EKI, relax ab initio on the pre-centred selection scores .82 and our SGD PPCA .26-.29 (partial). 10180 is being scored.

## 1. Datasets

STANDARD. Suite `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_suite_b150_20261102/datasets/` (read-only; README.md
there documents every set and its checks).

- **Rules for new synthetic sets:**
  - No solvent correction, uniform views, B-factor 150.
  - Add complexities back one at a time, each as a variant identical to its base except that one factor.
- **Sets, with the q we run:**

  | Set | Contents | q |
  |---|---|---|
  | C | 11 states | 10 |
  | H1 | one map (homogeneous control) | 10 |
  | T10 | 10 unrelated Tomotwin proteins, equal shares | 9 |
  | D10 / D20 / D40 | one distinctive Tomotwin protein (vol0042) in 10 / 20 / 40% of particles, the other 99 proteins share the rest | 2 |
  | Ribosembly | 16 CryoBench maps, paper weights | 10 |

- **Noise and box:** noise levels 1 and 3. Box 128 and the box-64 crop (`__b128crop64`).
- **Development box is 64:** method decisions are made at box 64. Box 128 is for final and high-resolution runs.
- **Diagnostic-only variants**, never for method comparisons: `C__realistic_{solv,pref,b0}__noise1` (one old setting put back each). The shifted variants `C__realistic_shift__noise{1,3}` (shifts N(0, 2.5² px) at box 64, truncated at 3σ) are for translation work only.
- **Known gaps:** no shifts in the main sets (being addressed). Every set exists at noise 1 and 3, but runs so far cover only H1 and Ribosembly at noise 3 and T10 and D at noise 1.

## 2. Reference method: VDAM-simple

REFERENCE. The from-scratch method every new method is compared against.

    coupled_arm.py --source <relax> --manifest <set>/training/manifest.json --seed 11 --stop-after 3000 --q <q> \
      --shift-step 1 --shift-range 2 --vdam-ko noadapt --eval-every 500 --milestones 25,50,...,500,750,...,3000 \
      -- --optimizer vdam --stages <box-64 stages, section 4> --oversampling 0

- relax source: `/scratch/gpfs/CRYOEM/gilleslab/em_work/ws2_gated_sgd_20261006/relax`.
- Driver: `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round17_simple_sgd_20261102/tools/coupled_arm.py`.
- VDAM keeps its learned direction prior; on uniform data it stays uniform.
- **Direction ablation (2026-11-05, one seed):** VDAM with a per-frequency diagonal direction (G / diag H_f) matches full H⁻¹ on poses and frames on C n1 (4.0°, 100%) and C n3 (4.5°, 96%), with PC sizes nearer truth at the high shells. The block direction (no mean-PC cross terms) also matches on poses. A scalar direction fails C n1 (61°, mirrored states). So the per-frequency scaling is what matters, not the coupling. Suite v3 scores (seed 11; decoded / pose median / N_struct / II z→E): C n1 full .895 / 3.9° / .939 / .82, block .904 / 3.9° / .940 / .79, diagonal .912 / 4.1° / .935 / .65; C n3 full .859 / 4.6° / .762 / .98, block .864 / 4.6° / .768 / .97, diagonal .876 / 4.6° / .770 / .92. Diagonal gives the best decoded score but a lower II z→E. Seed 12 reproduces it (C n1 .913, C n3 .874, v3.1), so for VDAM the diagonal direction holds at two seeds.
- **Which parts matter** (round-17 ablation): the half-set gate and the per-frequency H⁻¹ direction are essential. The late step decay sets the PC size. The fudge ramp and direction prior matter little, and a constant fudge of 4 is as good as the ramp. Dropping the moving average kept poses but killed the PCs on C n3 (PC/mean about 0); it was harmless on C n1 and H1.
- **Current numbers** (suite v2, decoded class; pose median):

  | Set | VDAM-simple |
  |---|---|
  | C n1 | .895 (3.9°) |
  | C n3 | .859 (4.6°) |
  | H1 n3 | .971 (4.2°) |
  | Ribosembly n3 | .794 (4.5°) |
  | T10 n1 | **fails** (.275) |
  | D distinctive protein | .787 / .914 / .942 |

## 3. SGD standard (in development; not yet a replacement for VDAM)

- **STANDARD for every SGD arm:**
  - uniform direction prior (owner 2026-11-03);
  - separate curvature scalars for the mean and the PCs, `--sgd-curv split` (owner 2026-11-04);
  - no momentum: omit `--momentum` for split arms (`--momentum 0` installs a wrapper that multiplies lr by 10);
  - VDAM's step-decay schedule (`--lr-schedule file:`), with lr matched at update 1 so the mean step equals the shared-scalar step.
- **Why:** with a learned prior, early wrong poses collapse the prior onto a few directions and lock in. With one shared curvature scalar, the PC steps are far too large and the PCs absorb pose error.
- **Known failure:** on the new data without solvent correction, SGD can lock individual states into a mirrored or rotated frame. Seen with the shared-scalar recipe (uniform + decay + oracle shrink, step 1e): 4 of 11 states mirrored on C n1, 3 on the b0 variant, and 2 states rotated about 40° on the pref variant. VDAM-simple on C n1 stays single-framed. Always run the per-state frame check (section 5).
- **Leading SGD candidate (step 1g, 2026-11-04, one seed, old-scorer diagnostics; suite pending):**
  - recipe: uniform direction prior + per-frequency H⁻¹ direction + ρ_pc gate on the PCs + VDAM step decay; no fudge, no moving average, no momentum, no oracle;
  - C n1: 4.1° (100% within 10°), all states in one frame, FSC .968, energy .80, PC/mean .16 (VDAM-simple: 3.9°, 100%, energy .92);
  - C n3: 4.5° (97%), one frame, energy .63 (VDAM .89);
  - without the gate, C n1 mirrors 6 states; without H⁻¹, C n1 is slow (74%);
  - **batch 3 (two seeds on C; H1, Ribosembly, D40):** poses and frames match VDAM-simple on every set and seed (C n1 100%, C n3 97%, H1 99%, D40 distinctive 100%, Ribosembly 79% vs 82%);
  - **energy (old scorer) is low and varies with seed:** C n1 .80 / .33 at seeds 11 / 12. **But in suite v2, hinvgate C n1 decoded is .905 at seed 11 and .907 at seed 12, against VDAM-simple's .895.** So the old scorer's energy is not a reliable guide here. On C n3, the best SGD arms (udecayshrink, splitshrink, hinvdecay) reach .900-.901 decoded against VDAM's .859;
  - **Suite (v3) for the gate schedule:**
    - hinvgate with gate off after 500: decoded C n1 .918, C n3 .900;
    - floor .25: .914, .897;
    - gate on throughout: .905-.907, .863-.866.
  - **Off the C sets, gate-off-500** helps Ribosembly (85% within 10°), but overshoots the high-shell PCs on D40, adds spurious power on H1, and diverges on T10.
  - **Caveat:** switching the gate off after 500 DIVERGES on T10: the PCs run away and the run crashes at update 909. T10 needs PC control for the whole run, so gate-off-500 is not a universal default. A permanent higher floor (.25) is being tested.
  - **The gate schedule matters:** switching the gate off (φ = 1) after update 500 keeps C n1 single-framed (4.2°, 100%) and recovers energy (C n1 .90, C n3 .898 against .63), with high-shell PCs nearer truth (one seed, 2026-11-05). The gate's job is the first 500 updates. A floor of .25 is nearly as good (C n3 .848).
  - **SGD direction (step 1h, 2026-11-05, two seeds):** a diagonal direction + floor .25 passes pose and frame on C at both seeds (C n1 4.9-5.0°, 99%; C n3 5.1°, 95.4-96%, a thin margin), but it ends the 16-31 band at 1.6-1.8× truth, is about 0.5° worse than H⁻¹ and converges more slowly. **Decision (lead, 2026-11-05): SGD keeps H⁻¹ + ρ_pc floor .25 for the runs on the other datasets (D10, D20, H1, Ribosembly, D40).** This changes the predeclared rule, which waited for the diagonal decoded scores. Those scores still get recorded; they are context only. Diagonal suite rows (v3.1, two seeds): C n1 .882 / .886, C n3 .888 / .889, against H⁻¹ + floor .25's .914 / .897. Diagonal fails the amended C rule at both seeds and both noise levels. Gate-off-500 at seed 12 reproduces seed 11 within .001 (C n1 .918, C n3 .899), but it stays ruled out as a default (T10 crash, D40 overshoot). On T10 from scratch (s11), both floor .25 directions are stable to 3000; diagonal poses 5 of 10 proteins, H⁻¹ 3 of 10. (For VDAM, by contrast, the diagonal direction is enough; see section 2.)
  - **OPEN:** gate-off-500 at seed 12 and on H1, Ribosembly and D40 (suite pending); H⁻¹ + floor .25 on D10, D20, H1, Ribosembly and D40.
- **Known risk with split curvature** (t10solve, 2026-11-04; T10 from an oracle start, constant lr): the PC power can run away (PC/mean 3 at update 500, 427 at 810).
  - c_PC scales with E[zz^T], which shrinks as W grows (the W·z scale ambiguity), so the PC step keeps growing.
  - **Update:** it is not specific to split. Shared-scalar SGD without PC control also runs away on T10 (PC/mean 14 at update 1000), and split runs away on C n1/n3 too (PC/mean 692 / 32).
  - The ρ_pc gate held PC/mean at .8-.9 on T10 from an oracle start (8 of 10 states solved). Treat SGD without a PC control as unstable.
  - Watch c_PC, the effective PC step and PC/mean in every SGD run.

## 4. Sampling schedule

- **STANDARD, box 64:** `--stages [[1,11,1],[126,12,2],[251,13,2],...,[1501,23,2],[1626,24,3],...,[2500,31,3]]`, i.e. (first update, band radius in shells, HEALPix order).
  - Order 1 (30°) to update 125, order 2 (15°) from 126, order 3 (7.5°) from 1626.
  - Radius +1 shell every 125 updates, to shell 31 at update 2500, held to 3000.
  - No oversampling.
  - The pose floor is about 3.6° (median distance to the nearest grid point), so differences below about 1° are invisible at box 64.
- **STANDARD, box 128:** the same stages to update 2500, then +1 shell every 125 updates to Nyquist (63) at 6500. Shifts ±4 px step 2.
  - **OPEN:** HEALPix stays at order 3 with no oversampling above shell 31, which undersamples the high shells. Oversampling or order 4 above about shell 32 is to be tested after the shift work.
- **Shifts, current:** a zero-centred circular grid ±2 px step 1 at box 64 (13 shifts), no stored per-particle offsets.
  - **OPEN:** this has never been tested on shifted data. Settled by the shift-scheduling study (`ppca_coupled_20261026/shift_sched_20261104`).
  - **Leading candidate (2026-11-05): "relrfine"**, which is RELION's rule R with a 1 px (6 Å) step from HEALPix 3, keeping the ±6 px window. On C_shift n3 its direct fit is 5.64° (81.5%), FSC .894, shift error 0.51 px median (R: 6.84°, 0.89 px; T1: 5.81°).
    - R in Å is box-size invariant: box 128 matches box 64 in pose, shift error in Å and learned σ.
    - R passes the no-cost test on new C n3 in suite v2 (decoded .858 against .859).
    - **Suite v2 on C_shift n3:** decoded .833, pose 6.4° (p90 25°). That equals the exhaustive TFULL-T1 (.834, 6.4°, p90 25°) and beats T1 (.824, p90 54°). The remaining gap to the unshifted control (.859, 4.55°) is the same for the exhaustive search, so it comes from the shifts, not the rule.
    - **Wave F (direct fit, one seed):**
      - Ribosembly n3 4.42° (control 4.49°, passes);
      - new C n3 4.54° (control 4.49°, passes on pose; suite pending);
      - C_shift n1 4.16°, 98.9%, shift error 0.40 px, the best arm at n1 (control 3.89°);
      - **known cost:** old-suite C n3 8.74° against 7.38°. That set locks late and has preferred views and solvent, which are non-standard; every shift-searching rule loses there.
    - **Suite v3, new C n3 (unshifted):** relrfine .858 decoded, 4.5° (current rule .859, 4.55°). No cost on new-suite data.
    - **Other shift sizes and cases (2026-11-05, direct fit, one seed):**
      - C n3 6 Å: relrfine 5.5° = current rule 5.5°; both are about +1° on unshifted. No suite rows.
      - C n3 30 Å (diagnostic): relrfine 7.6° (training-time); the current rule fails (127°).
      - C_shift n1 suite v3: decoded .874 against unshifted .895 (−.021, at the .02 limit); pose +0.7°.
      - Off-centre map (C n3, 15 Å shifts plus a common 20 Å offset): relrfine 5.66° against 5.64° without the offset; learned σ within 3% of truth; the map carries the offset. Passes.
      - **Known failure: shifted Ribosembly n3 (15 Å).** relrfine 108° and the current rule 112° never lock. The exhaustive ±8 px window poses (6.3°, 60%), and so does the true-shift oracle (5.3°, 67%). So the stored-offset window, which follows wrong early offsets, is the problem on this set, not the data.
    - **absearly** (absolute zero-centred ±48 Å window until HEALPix 3, then relrfine): fails shifted Ribosembly (52.5°, 42.5%; direct fit, diagnostic), no cost on C_shift n3 (5.67°). It switches to stored offsets before the poses lock (even the exhaustive window locks only at updates 2000-2500).
    - **abslate** (the exhaustive window with 12 Å steps throughout, no 6 Å refinement): poses shifted Ribosembly (7.9°, 53%; exhaustive 6.3°, 60%) but costs 1.2° on shifted C n3 (6.9° against relrfine's 5.6°). It fails the predeclared rule. The absolute re-search is what lets Ribosembly lock; the late 6 Å step is what C needs for precision. So the full exhaustive grid stays in best v1.
  - **Earlier candidate: RELION's rule "R".**
    - Window ±6 px step 2 at box 64 (±36 Å, 12 Å), fixed.
    - Stored offsets.
    - Learned σ starting at 10 Å, floor √2 Å, prior centred at 0.
    - Its learned σ grows back to the true 2.5 px on shifted C (n1 and n3) and collapses to the floor on unshifted C, so the data release the shifts with no schedule.
    - T1 (fixed 20 Å) fails on unshifted old C n3 (120°) and costs on Ribosembly. The current ±2 rule fails on shifted C (suite decoded .738, pose p90 157°).
  - **Shift-study facts (shiftsched, transeval; one seed, C at box 64):**
    - **Visits:** with minibatch 300 over 3000 updates, a particle is seen about 0.4 / 4.5 / 4.1 times at HEALPix orders 1 / 2 / 3.
    - **Stock learned σ is biased:** relax's M-step, σ² = E_post|t|²/2, collapses under a weak likelihood on narrow or coarse grids (±2 step 1: 1.67 → 0.64 px in 100 updates; 3.1 px steps: to 0.24 px). The exact grid M-step (--offset-mstep grid, zero-centred grids only) removes the bias.
    - **Insertion is verified:** from a true start, the shifted map tracks its unshifted twin.
    - **Late step size matters:** the arc-tied 3 px late step (T0) costs about 3° and a third of the energy against 1-2 px schemes on shifted C n3. That is the step, not the window.
    - **On shifted C n3 (suite v2), no scheme reaches the unshifted control** (.859 / 4.55°). T1 .824 / 6.50°, exhaustive TFULL-T1 .834 / 6.40° (p90 25° vs T1's 54°), the current ±2 rule .738 / 14.7°.
    - **On shifted C n1 (direct fit):** T1 4.23°, TFULL-T1 4.25°, relr 4.60°, the current rule 5.61°, against the control's 3.89°.
  - **Earlier proposal "T1"**, no longer the best guess (it fails on unshifted old C n1/n3 and Ribosembly), (owner 2026-11-04; use it for new work with shifts, broad evaluation in `ppca_coupled_20261026/trans_eval_20261104`):
    - steps of 12 Å (2 px at box 64) until HEALPix order 3, then 6 Å (1 px); at box 128 that is 4 px then 2 px;
    - ±3 steps (29 points) throughout, centred on the stored best offset;
    - Gaussian σ = 20 Å on the total offset, centred at zero. **STANDARD (owner): the translation prior is always centred at zero on the total offset. Only the search window may follow the stored offset.**
    - integer offsets.
    - The reference arm TFULL uses the same steps in a fixed, zero-centred ±8 px window.
  - **Earlier proposal "T0"**, kept as a comparison arm:
    - **Prior:** a Gaussian on the total offset, σ = 20 Å, centred at zero. The RELION EM-like fitted σ is an option.
    - **Rule (STANDARD for all sampling rules):** image-size invariant, tied to the HEALPix sampling. Define steps in Å from the angular step, windows in steps, priors in Å, and round to whole pixels only at the end.
    - **Grid:** fixed and coarse at the start, refined with HEALPix, keeping the same stages and 3000 updates. step_Å = θ(order) × D/2 (RELION's rotated distance, D = manifest particle diameter); step_px = max(1, round(step_Å / pixel)). Window ±3 steps (29 circular points), as in RELION's InitialModel default of 6 px range / 2 px step.
    - **At box 64** (48 px diameter): steps of 13 / 6 / 3 px (75 / 38 / 19 Å) for orders 1 / 2 / 3. At box 128 the same rule gives 25 / 13 / 6 px.
    - **The late 3 px step at box 64 is the angular grid's own coarseness:** order 3 (7.5°) moves the particle edge by 3 px, which is also why the pose floor is about 3.6°. Finer late sampling (oversampling or HEALPix 4) must refine angles and shifts together.
    - **Centring:** each particle's best previous offset is stored and centres its next window.
    - **Steps:** never half-pixel shifts; integer steps and offsets.
    - **Search around T0:** step multiplier, window size (±2 / ±3 / ±4 steps), prior σ (10 / 20 / 40 Å or learned), zero-centred vs stored, and a final 1 px phase.
    - **Reference arm TFULL** (owner): every rotation in a fixed, zero-centred big window (±12 px at box 64), with the same steps and no stored offsets. It measures what the efficient centring and window cost.

## 5. Scoring

STANDARD. Metric suite v2 (run by the metrics agent; GPU stage then CPU scoring, chained). No new or renamed metrics
without the owner's OK.

- **Revisions:**
  - **Current (v3.1, 2026-11-05):** `tools/metrics_suite_v3_1`, msuite_score `83dc8ee9`, msuite_data `e69374a1`. It is v3 plus a cross-seeded refit for frames whose routes still disagree after the v3 refit, keeping the better objective. Other rows are identical.
  - **v3:** `tools/metrics_suite_v3`, with msuite_score `bd78b912`, msuite_data `43ccc732`, msuite_core `23a6a294`, msuite_infer `fcde49b6`.
    - Adds the flag-and-refit: a frame whose two routes disagree by more than 10°, or whose objective is under .5 while the other route's is over .9, is refit from 10 coarse seeds plus a centre-of-mass seed.
    - Adds a `rotation_ambiguous` status: the same-hand runner-up is within .02 of the winner and at least 10° away. It changes no score.
    - Acceptance: unflagged rows are unchanged (largest difference 2.3e-10), and cdrgn H1 recovers (BP .344 → .990).
    - The shifted-set translation grid is sized from the shift spread (±5 or ±9 step 1).
  - v2 (`4b4742e0` / `e8960944`, msuite_data `6940dce1`) is still valid for unflagged rows.
  - The round-17 28-entry table, Ribosembly and D10 were scored at msuite_score `c90f5017`. Its grid and explicit-input outputs are identical; 4b4742e0 only adds the decoded-map input.
  - Notes: `tools/metrics_suite_v1/REVISIONS.json`.
- **Headline columns:**
  - decoded-class FSC-AUC and per-image FSC-AUC;
  - BP class, its single random subset, and gain = BP − random;
  - pose median / p90 after the PPCA-map frame and after the BP-map frame (same raw MAP poses), with the nearest-grid reference;
  - N_struct;
  - information imbalance in both directions (z to E, E to z). No k-means ARI.
- **Map score definition:**
  - Normalised FSC-AUC: a signed trapezoid divided by S − 1, where 1 = exact. Not CryoBench's integral.
  - Band: shells 1 to box/2 − 1. At box 64 that is shells 1-31. At box 128 it is 1-63, with 1-31 reported beside it and 32-63 separately.
  - One truth-only whole-particle mask; equal-class macro means.
- **References next to every map score** (references, not ceilings), from that dataset:
  - mean only;
  - rank-q noiseless;
  - true-pose class BP;
  - the true model with noisy inference.
- **Pose:**
  - Suite v2's pose is the headline.
  - `direct_fit.json` (old pipeline; it fits the frame from inferred against true poses, so it uses the truth) is a quick-check and compatibility diagnostic only. Never quote `summary.json`'s pose.
- **Frames in the suite:**
  - Frames are fitted on calibration particles only (20%, stratified by class and view) and scored on the other 80%.
  - Related states (C, H1, Ribosembly): one frame per dataset and route. Unrelated proteins (D, T10): one frame per protein.
  - Multi-start search, 16 starts per hand. Hand disagreement between the two routes is flagged. Per-state hands are never fitted.
- **Per-state frame check** (diagnostic, `round17 step1f/per_state_pose.py`): for related-state sets, split the pose by true state to catch mixed-hand solutions. The old scorer's per-state FSC fits a hand per state and hides them, so it is compatibility only.
- **D sets:**
  - Report the distinctive-protein column, and the population-weighted values beside it.
  - The equal-class macro is about .2 even for the true model (.225 / .230 / .224).
  - The all-protein pose and frame numbers are unreliable: 40-60 of the 100 background-protein frames disagree in hand between routes, even for the true models.
- **Input modes:**
  - PPCA checkpoints (grid);
  - explicit poses plus given latents, with the estimator named in the row (route a, RECOVAR);
  - decoded maps (cryoDRGN).
- **Known suite bug (found 2026-11-06, v3.1):** for a pseudo-symmetric protein the per-protein frame search can settle on the winner's ~180° partner. That partner fits the low shells well (objective .86-.88), so no flag fires, but it loses at full band. Seen on the D distinctive protein 42: the suite gives D20/D40 decoded .608-.625, while the frame its own objective prefers gives .90-.94. Fixed in suite v3.2 (accepted 2026-11-06): six extra 180° partner starts about axes through the target's centre of mass (3 box axes, 3 inertia axes), truth-free; a partner replaces the winner only if the objective gains more than 1e-3. Acceptance: C n1/n3, H1 and true_T10 are unchanged; T10 union protein 5 is fixed (.778 → .949, row decoded .779 → .796); the true-model D distinctive frames are unchanged, while 60-65 of the poorly determined D background frames move (population-weighted +.003). Use v3.2 for all new scoring; quoted D and T10 rows are being rescored.
- **Oracle arms** (any use of the truth beyond scoring) are labelled "oracle" everywhere.

## 6. Baselines

STANDARD rows per dataset. Box 64 (`__b128crop64`), seed 11, b150 sets. Table:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/b150_baselines_20261103/results/b150_table.md`. True-model rows are in the
metrics suite folders (`true_<set>` under `ppca_coupled_20261026/metrics_suite_v2_*`).

- **Homogeneous refinement (route a):**
  - Settings: relax initial_model --K 1 at defaults, then relax refine (HEALPix 2 to 3, no oversampling, offsets ±6 px step 1 at box 64, at most 30 iterations).
  - Map sign: flipped for scoring (RELION convention: protein-dark stacks). Checked by an inverted-stack control.
  - Status: poses found on H1 n3, C n3 and Ribosembly n3 only; it fails on C n1, D10, D20, D40 and T10 (one seed).
- **RECOVAR at route-(a) poses:**
  - Settings: unmodified pipeline, `--mask from_halfmaps`, zdim = the set's q, `--no-downsample`. Latents are RECOVAR's embedding divided by sqrt(s).
  - Run only where route (a) found poses (median under about 10°); otherwise the row says so.
- **RECOVAR at true poses:** the ceiling. Same settings.
  - D sets: q = 2 as standard. More PCs barely move the distinctive protein (D10 .795 at q 2, .805 at q 20).
- **One cryoDRGN: v3.**
  - Settings: cryoDRGN 3.4.4 abinit_het, CryoBench App. B settings (zdim 8, 30 epochs). Scored through the decoded-map input, sign checked.
  - Status: it finds poses on C n1 only (2.8°) and fails on C n3, Ribosembly, T10 and D40. A weak baseline on most sets.
- **cryoDRGN-AI:** not continued. It was worse on C n1 (5.8° vs 2.8°), failed on Ribosembly and D20, and takes about 8 h vs 3 h. Rerunning 0.3.3b0 needs `--cfgs amp=False`. The rows for C n1, Ribosembly and D20 are kept as "supplementary".
- **Scoring:** suite v2 as in section 5. Non-PPCA models enter through the explicit-pose input (route a, RECOVAR) or the decoded-map input (cryoDRGN).

## 7. Reporting

STANDARD for every report:
- Plain language, numbers first, verified before sending.
- Full absolute paths; the number of seeds stated.
- SVD-sorted raw PCs with σ and |PC|/|mean|.
- Latents in PC coordinates (UMAP only in addition).
- Corrections reported plainly.

## 8. How experiments run

STANDARD:
- One seed (11) first, few arms. Predeclare hypotheses and pass rules in PREDECLARED.txt before running.
- Each workstream gets a dated folder under `/scratch/gpfs/CRYOEM/gilleslab/em_work/` with SAFE_TO_DELETE, TASK.md and HANDOFF.json (kept current).
- At most 2 GPUs per agent. Ask before more than about 4 GPU-hours in one job. Cluster rules are in the agents' CLAUDE.md.
- Check that jobs advance in their logs, not just their state. Chain scoring jobs to the GPU stage.

## 9. Open questions (current work)

| Question | Where | Status |
|---|---|---|
| SGD PC control; SGD + H⁻¹ | round17 step 1g | running |
| Shifted data (owner top priority 2026-11-09) | robust_20261107/SHIFT_FINDINGS.md (shiftdiag), realshift_20261109, relionshift_20261109, ribodeep_20261109, relaxab_20261110 | D10_shift solved truth-free by the three-stage route (v1.66; suite-confirmed v1.67). D5_shift and Ribosembly_shift n3 open (best Ribo_shift n3 row: VDAM-simple 6000 updates .755, mean only .780). Real data: the wide held-prior window closes most of the as-picked gap on 10073, 10345 and 10076, not on 10097, 11846 or 10180 (v1.67). 10180 solved truth-free by global recentring (v1.69). Running: relax K-class defaults on the shifted sets, full-set VDAM on D5_shift |
| D-set seed dependence: does the SGD candidate flip the distinctive protein into a second frame on some seeds? (abinit_study handoff, 2026-11-05: many SGD arms at ~50% on D40 at one seed of three) | round17 step1h_sgddiag, seeds 12-13 on D10/D20/D40 | Answered 2026-11-05: yes. The candidate fails D10 at 2 of 3 seeds and D20 at 1 of 3; VDAM fails D10 at 3 of 3. Failures are scattered mis-poses, not a clean second frame. The noise anneal (σ² × 3 → 1 over updates 1-500) is not adopted: it rescues 2 of 3 failures but makes D10 s13 worse and costs Ribosembly 14 points and T10 one protein (no harm on C, D40, H1). Seed selection by held-out likelihood fails, both mean and paired per-image (spread < 1 nat/image). PC-shrink test on the 3 failing seeds: floor .5 rescues 1, gate-off-500 rescues 2 (crashes T10); biased by seed selection. Fresh-seed test (seeds 14-16, direct fit, diagnostic): floor .25 passes 3 of 6, floor .5 passes 4 of 6, a tie under the predeclared rule (≥3 needed). Floor .25 stays. Pooled floor .25 over seeds 11-16: D10 2 of 6, D20 4 of 6. D10/D20 remain seed-sensitive with no fix found |
| Off-centre particles (common image-plane offset) | realshift_20261109 (10180) | Found on real data (v1.67): 10180's picks share a 55 Å image-plane offset that no volume shift can absorb; the mean particle image's centre of mass recovers it truth-free. Global recentring + wide window solves it (v1.69: .958 vs pre-centred .973). CLOSED (owner, v1.70): a one-off of this deposited stack (never re-extracted centred), not pursued further |
| T10 | t10_solve_20261104; transfer_20261105 | Best row: union10hinv (K-class SGD → per-class ab initio, seeds by held-out likelihood → union selection → SGD H⁻¹ + gate floor .05), suite decoded .779 / BP .826 / N_struct .968 (mean only .382, true .996). **Correction 2026-11-05:** not fully truth-free. In its history, 3 classes kept their seed-11 maps because the truth showed them posed. The frozen, truth-free version is being rerun on T10 as a control, then on D10, D20 and a new 10-protein set T10b (transfer test). **From scratch, no class step, with gate floor .25 (suite v3.1):** 8 random starts to update 500, best by held-out likelihood (seed 12), continued with the diagonal direction: decoded .760, BP .770, N_struct .951. Continued with H⁻¹: .695 / .746 / .948. Single wrong-protein start (seed 11): .633; ten wrong proteins: .614. The older from-scratch rows (.28-.41) used floor .05, plain SGD or VDAM. The wrong-protein start gives no gain over a random start (direct fit at seed 12: 9.5° vs 8.9°). The simple multi-start pipeline is in the transfer test. |
| Box-128 high-shell sampling | after shifts | not started |
| Ribosembly gap (VDAM .794 vs true .932) | after step 1g | the pose tail (states 0-2, the three smallest assemblies) is a signal limit: the true model mis-poses them equally (median 44-102°). So the map gap is on the PC/state-map side, not poses (2026-11-04) |
| Noise-level gaps (H1/Ribo n1, T10/D n3) | with the winning recipe | not started |

## Change log

- 2026-11-04 v1. First version. Collects the decisions of 2026-11-02 to 11-04:
  - new dataset rules;
  - VDAM-simple as reference;
  - uniform direction prior for SGD (11-03);
  - split mean/PC curvature for SGD (11-04);
  - direct-fit pose;
  - per-state frame check;
  - D distinctive column;
  - one cryoDRGN baseline (v3).
- 2026-11-04 v1.1. Section 6: baseline settings and status filled in (drgnbench, checked against b150_table.md).
- 2026-11-04 v1.2 (ppcabase corrections). All sets exist at both noise levels; the gap is runs. Momentum flag note for split arms. Moving-average ablation effect. Frame-failure cases attributed to the step-1e recipe.
- 2026-11-04 v1.3 (ppcametrics corrections). Section 5 rewritten:
  - suite v2's map-frame pose (PPCA and BP frames) is the headline pose; direct_fit is a diagnostic only;
  - full headline column list, band and AUC definition, references, revisions, how frames are fitted, D-set caveats, input modes.
- 2026-11-04 v1.4. Section 4: the owner's proposed base translation schedule T0 recorded (under evaluation, not yet STANDARD).
- 2026-11-04 v1.5 (owner). Sampling rules must be image-size invariant, tied to HEALPix. T0 window widened from ±1 to ±3 steps (RELION InitialModel uses 6 px range / 2 px step). TFULL reference arm added.
- 2026-11-09 v1.64. ADOPT oversampling (os1: from HEALPix 3, each kept rotation split into 8 at 3.75 deg, rotations only) on recipe v2 (--fixed-step .10), by the predeclared rule (mean os1-v2 over s11/s12 >= +.01 on >= 2 of 4 sets, none < -.01): C n1 +.019 (.937/.721 vs .915/.706), C n3 +.018 (.916/.917 vs .899/.899), Ribosembly +.022 (.860/.861 vs .838/.839), T10 -.003 (.594/.556 vs .601/.555). Cost: os1/v2 time per update after 1626 = .68-1.05 (cheaper on C n3, Ribosembly, T10). Note: C n1 seed 12 fails for both arms (.706/.721, below mean-only .837). D10 with realistic shifts (shiftdiag, seeds 11-13): search-off fails (111-118 deg), late search from 1626 fails (123-131 deg) and also breaks unshifted D10 (96 deg); the ORACLE pre-centred copy and fixed true offsets pose (3.7-3.8 deg). A 10% state needs centred particles to pose, and shift search before the model forms locks onto a wrong model: open problem.
- 2026-11-08 v1.63. Correction to v1.62: the slow-start gains were a step-size confound. The driver's step is lr x k with k recomputed in-run at update 1; a shell-8 start doubles k, so E2/E2a ran at T .18-.27 vs .10 (also the old E1 resume, T .18). New driver flag --fixed-step T (sha 8294284e) sets the step directly. Clean T10 test (seeds 11-13): slow marching at T .10 .557/.579/.571 (mean .569, plain .624); plain stages at T .21 .744/.640/.626 (mean .670, within .02 of E2a's .686). So the step is the active ingredient and slow marching at matched step hurts (C n1 .905 vs .915, C n3 .890 vs .899, Ribosembly .825 vs .838). Slow start is dropped. Very mixed samples (T10) and D5 subsets (0/5 at T .05, 4/5 at .10, 5/5 at .15) want a larger step, about .15-.2; the canonical sets were tested at .10. D10 with realistic 15/30 A shifts: every strategy fails (108-133 deg); the ORACLE pre-centred copy passes (3.8 deg), so shift learning is the failure, not the data.
- 2026-11-08 v1.62. Recipe v2 validation (suite v3.2, ablate VALIDATION_V2.md): centred sets all hold vs their matched SGD rows (C n1 .915, C n3 .899, T10 .601, Ribosembly .838, H1 .980; D20 2/3 above bar; D40 .961-.963 = hinv25, bar .970 unreached by any method). Shifted sets with the FFT option: C_shift n1 .689 FAIL (36% partner-pose tail), C_shift n3 .834 (at mean-only), Ribosembly_shift unposed. On synthetic centred sets the fixed step is neutral (references already ran at T .06-.12); its gain is on real data (10076, 11846). Also: slow start transfers to T10b (3/3 pass, mean .786 > union .756) and beats plain on all 8 T10 seeds (mean .671 vs .583, 2/8 pass); D5 first solved on a subset by latent-outlier selection (4/5 seeds incl. 2 blind; a full-data run from the subset model still loses it).
- 2026-11-08 v1.61 (owner decisions on recipe v2). (A) ADOPT a fixed step: the update is T x H^-1 G per voxel (H^-1 G = the batch's per-voxel Newton direction), T about .08-.10 (C n1 spikes to ratio 2.96 at .10, so .08 is the safer default). This replaces the update-1 lr match; the old factor k (= |G/c| / |H^-1 G| at update 1, a conversion from the earlier split-curvature step) cancels and is dropped: set T directly. Evidence: 11846 .29 (old rule, T .011) -> .75-.79 (T .04-.20), .71 diverges; 10076 .80 -> .93; at T .10 the canonical sets hold (C n1 .915, C n3 .899, T10 .601, Ribosembly .838, H1 .980; D20 2/3). (B) ADOPT: particle diameter is a REQUIRED explicit input (it is the 3D support mask; relax never masks images). Measure it from the data plus a margin for pick offsets; if unknown, about 80% of the box. Too small ruined HA/TRPV1 v1 (150 A); too large let the challenge set fit neighbours (262 A). (C) OPTION, not default: the FFT all-shift search with learned centre/sigma and a late switch to local windows. It solved real uncentred 10180 (.961) but fails synthetic shifted sets (C_shift n1 .689 with a 36% partner-pose tail, n3 at mean-only, Ribosembly_shift unposed) and drifts to neighbours on crowded HA picks. Needs an automatic, robust version before it becomes default. (D) NOTE, not default: slow start (half step + start at shell 8 for the first 750 updates) improved T10 (mean .685 vs .583, 3 seeds); being tested on more seeds, T10b and other sets, plus single-factor ablation. (E) NOT adopted: a final RECOVAR solve at fixed poses; used only for baselines (homogeneous + RECOVAR). Next: D5 exploration (all techniques 0/5 so far).
- 2026-11-07 v1.60. Candidate C (VDAM halfset gate applied inside the H^-1 solve, PCs only) is unstable on synthetic C n1: diverged at update 1608 at T .10 ("nonfinite metric"); step spikes 6-7x at T .0437 and .07; on 11846 at T .10 .788/.791 (< .80); 10076 at T .10 .930. Cause: shrinkage -(1-phi) diag(H) theta inside the solve is amplified along ill-conditioned directions (cond 150-300). Candidate D (shrinkage applied outside the solve, VDAM-exact: theta_PC += T[phi (H^-1 G)_PC - (1-phi) theta_PC], mean ungated) ran C n1 at T .10 to 3000 stably (step ratio 1.40). D suite rows and real-data runs pending.
- 2026-11-07 v1.59. Transfer test final (suite v3.2, one seed each): the frozen truth-free union pipeline transfers to T10-like sets: T10 control .791 (7/10 proteins), fresh T10b .756 vs bar .695 (6/10; plain SGD .628, 3/10). On D sets it matches but does not beat the best single SGD seed (D10 .935 vs bar .924; D20 .954 = bar). Fails everywhere at noise 3 (T10 n3: best .543, union .489, bar .678; true model .973). Lost proteins separate in latent space, so failures are per-protein pose/frame, not classification.
- 2026-11-07 v1.58. Recipe-v2 candidates (lr = T/k_update1). Candidate C (T .0437 + VDAM halfset gate on PCs only, mean ungated): 10076 .936, 11846 pre-centred s11 .803 / s12 .804 (seed-robust): passes the real-data rule. Candidate A (no gate): 10076 .930 but 11846 .787/.652 and synthetic C n1 suite decoded .829, below mean-only .837 (N_struct .48): fails. Candidate B (full gate): 11846 .822/.823, C n1 .893 / BP .960 / II .947, but 10076 .606: fails. A at T .10 on 10076: .928. Proposed v2: T = .10 + PC-only gate; validation of C at T .10 (10076, 11846 s11/s12, C n1, D10/D5/T12 sweep) running.
- 2026-11-07 v1.57. 10180 uncentred, global FFT search + learned centre then travelling windows: PASSES the predeclared rule. Final (update 6600) mean-map FSC vs consensus .961 (consensus-centred run .963), above 0.143 to Nyquist; scorer pose on images pre-shifted by the stored offsets: median 5.3 deg, 78.9% < 10 deg (p90 134 deg, ~20% flipped/ambiguous). Learned offsets: for particles with pose < 10 deg, median 0.51 px from consensus, 93.5% within 1 px (after the fitted map-offset term).
- 2026-11-07 v1.56. Uncentred 10180 SOLVED without consensus offsets by global FFT shift search with a learned prior centre + learned sigma (start 80 A, floor 1 px) for updates 1-3575, then per-particle travelling windows (+-4 px around each particle's stored offset, local prior): mean-map FSC vs consensus .946 at update 5500 (default settings .166, homogeneous route .377, consensus-centred .963); training-time pose diagnostic 77% < 10 deg, median 5.6 deg at 6600. The learned centre found the stack's constant offset by update 100 (about 61 A per axis vs consensus 55). Not solved: 11846 all picks (.280 vs 8EKI; sigma never shrank; 22 A box margins). Synthetic: C_shift n3 fails with both priors (sigma held .803/p90 135; learned .832/p90 63); D20 s11 with learned sigma .953 distinctive (pass). Formal pose score pending.
- 2026-11-06 v1.55. Confirmation of the fixed-step rule (lr = T / k_update1, T = lr*k = .0437; k from a 1-update probe of the identical arm). EMPIAR-10076 (mean-map FSC vs consensus): candidate A, SGD fixed step, .930 (old-rule SGD .804, VDAM .887, RECOVAR at VDAM poses .940): FIXES. Candidate B, same + VDAM halfset gate, .606: HURTS (the gate shrinks the mean late, power moves into the PCs). 11846 pre-centred seed 12, A: .652 (seed 11 .787; spread .135). Step bracket on 11846 s11: .04-.20 flat within one-seed noise, .71 runs away. Recommended rule: T = .05 (range .04-.10), no gate. Synthetic C n1 check pending.
- 2026-11-06 v1.54. Uncentred real stacks defeat both routes: 10180 deposited stack, mean-map FSC vs consensus: PPCA SGD .166, homogeneous ab initio -> refine -> RECOVAR .377 (about 1% of poses < 10 deg for both; refine's travelling offset search gets further than PPCA's fixed zero-centred window); 11846 all picks: .264 vs .269. Consensus/published poses -> RECOVAR .994 / .753.
- 2026-11-06 v1.53. Suite v3.2, seed 11, SGD translations: v1b (relrfine) REJECTED. Decoded vs old +-2 px: C n1 .914 = .914; C n3 .842 vs .897; T10 .545 vs .594; Ribosembly .191 vs .836 (never locks); C_shift n1 .846 (rule >= .874, VDAM relrfine); C_shift n3 .819 (below mean-only .837, as v1 .816). For SGD on unshifted sets any translation freedom beyond +-2 px loses (v1 exhaustive grid and relrfine alike). The +-2 px grid stays the SGD translation rule for centred data; no SGD rule yet matches VDAM relrfine on shifted data (FFT all-shift arms pending). Oracle (true shifts) confirmed in the suite on D10 distinctive: s12 .918 vs control .409, s13 .921 vs .695, s11 .914 vs .924.
- 2026-11-06 v1.52. RECOVAR at PPCA MAP poses vs homogeneous ab initio -> refine -> RECOVAR (mean-map FSC; 11846 vs 8EKI): 10076 SGD .926 / VDAM .940 vs .946; 10180c .966 / .976 vs .975; 11846 pre-centred VDAM .801, SGD lr x4 .783 vs .797 (SGD at the old lr .291: its poses are wrong). With good poses, PPCA + a final RECOVAR pass matches the two-step route; the remaining gap in PPCA's own map is the estimator.
- 2026-11-06 v1.51. VDAM-vs-SGD ablation on 11846 pre-centred (one seed, FSC vs 8EKI at 3000; SGD .288, VDAM .762). Step size decides SGD: effective step lr*k .011 -> .288, .0437 -> .787, .10 -> .766, .71 -> collapses after update 1000 (mean amplitude runs away). With the step fixed at .0437, adding VDAM's shrinkage gate gives .822 (best). Orientation prior and momentum: no effect on either side. Small initial PCs hurt once the step is fixed (.622). In VDAM the PC gate is essential (removing it diverges; removing all gating locks wrong poses); the mean gate is not. Confirmation on 10076 and synthetic C pending.
- 2026-11-06 v1.50. 10076 (all particles, uncentred), mean-map FSC vs the RELION map, shells 1-31: PPCA model means SGD .804 / VDAM .887; RECOVAR run at the PPCA MAP poses .926 (SGD) / .940 (VDAM); homogeneous ab initio -> refine -> RECOVAR .946; published poses -> RECOVAR .996. Most of the map gap to the two-step route is the estimator (PPCA trained mean vs a full reconstruction at fixed poses), not the poses. A final reconstruction/RECOVAR pass at PPCA poses is worth adding to the pipeline.
- 2026-11-06 v1.49. EMPIAR-10180 uncentred deposited stack, default settings (SGD recipe, +-48 A window, sigma held 20 A, 6600 updates, lr re-matched .155): FAILS. Mean-map FSC vs consensus .166 (consensus-centred run .963); pose vs consensus 120 deg median, 0% < 10 deg. Only 12% of consensus offsets fall inside the window; learned MAP shifts correlate .21 with consensus; window-edge fraction stays <= .03. The default zero-centred window cannot handle stacks with large systematic offsets (here about 63 A per axis).
- 2026-11-06 v1.48. 11846 pre-centred, FSC vs 8EKI (shells 1-31): SGD with lr x4 (.707 instead of the matched .177; only change) reaches .787 at 3000 (VDAM .762, SGD .288); at 1000 .505 vs VDAM .511. The SGD failure on 11846 was the step size. The update-1 lr match is in split-curvature units: the effective step along the H^-1 direction (lr*k) was .011 on 11846 vs .059 on synthetic C n1 vs about .71 for VDAM. The lr rule needs revising (open).
- 2026-11-06 v1.47. Roey tests at seed 11 (direct-fit diagnostics): momentum 0.9 is INVALID, not a fail. The driver's --sgd-momentum (79fb4e29) applied the PC gate to the whole velocity each update (steady-state PC step about 8x too small at rho .25) and restarted velocity at zero on new shells; a corrected arm is in the VDAM-vs-SGD ablation. HEALPix 3 from update 126: C decoded +.002/+.005, T10 and D20 neutral, about 2x cost in updates 126-1625; suite rows pending.
- 2026-11-06 v1.46. EMPIAR-11846, pre-centred selection (uses published translations), against a map made from PDB 8EKI (FSC shells 1-31): VDAM-simple PPCA .762 at 3000 (above 0.143 to Nyquist, 14 Å) vs our SGD recipe .288 on identical input; relax homogeneous InitialModel .819. SGD fails on this real set where VDAM works. Pose vs published (direct fit, diagnostic): VDAM 6.7° median, 66% < 10°.
- 2026-11-06 v1.45. ORACLE, direct-fit diagnostic (suite rows pending): with translations fixed at the truth (no shift search), SGD poses all 6 D10/D20 runs at seeds 11-13, including the 3 that fail with the ±2 px grid (D10 s12 22.7° → 3.6°, D10 s13 5.4° → 3.8°, D20 s11 10.1° → 3.8°). Shift freedom is one cause of D seed sensitivity. Runs: best_v1_20261105/roey/.
- 2026-11-06 v1.44. Suite v3.2 accepted (pseudo-symmetric frame fix).
- 2026-11-06 v1.43. 11846 correction: our published-pose reference was wrong; scored against the PDB 8EKI model instead.
- 2026-11-06 v1.42. Suite bug: pseudo-symmetric per-protein frames can land 180° off; D/T10 rows possibly low; v3.2 fix in progress.
- 2026-11-06 v1.41. Exhaustive grid rejected for SGD (suite); oversampling passes on C; wrong-protein T10 start worse than random at seed 12 (.641 vs .695).
- 2026-11-06 v1.40. best v1 validation: the exhaustive grid hurts SGD on several sets; v1b (relrfine) running; 11846 collapses.
- 2026-11-05 v1.39. T10 from scratch with floor .25: multi-start + held-out-likelihood pick + diagonal continuation reaches decoded .760 (union start .779), no class step.
- 2026-11-05 v1.38. abslate fails (C_shift +1.2°); the exhaustive grid with the 6 Å late step stays in best v1.
- 2026-11-05 v1.37. Fresh-seed D test: floor .5 vs .25 is a tie (4/6 vs 3/6); floor .25 stays.
- 2026-11-05 v1.36. T10: best row partly used the truth (seed choice for 3 classes); frozen truth-free control and transfer test running; wrong-protein start no gain.
- 2026-11-05 v1.35 (owner). Current best recipe v1 (SGD candidate + exhaustive translation grid) pushed in best_v1/, validation running.
- 2026-11-05 v1.34. absearly fails shifted Ribosembly; abslate and the fresh-seed D test submitted.
- 2026-11-05 v1.33. D seed dependence answered; anneal not adopted; held-out-LL seed selection fails on D.
- 2026-11-05 v1.32. Shifts: relrfine passes the off-centre map; fails shifted Ribosembly, where the exhaustive window poses.
- 2026-11-05 v1.31. Step-1h suite rows: diagonal SGD fails C at two seeds; diagonal VDAM reproduces at seed 12.
- 2026-11-05 v1.30. Diagonal SGD fails the C n1 suite rule (.882); gate-off-500 reproduces at s12; floor .25 stable on T10.
- 2026-11-05 v1.29. Open questions: D-set seed dependence and off-centre map, from the abinit_study handoff (copied to ppca_coupled_20261026/abinit_study_import_20261105).
- 2026-11-05 v1.28. SGD direction: H⁻¹ kept over diagonal (diagonal overshoots high-shell PCs); beyond-C runs released with H⁻¹ + floor .25.
- 2026-11-05 v1.27. VDAM direction ablation suite v3 rows: diagonal ≥ full H⁻¹ on decoded score (+.017), lower II z→E.
- 2026-11-05 v1.26. Suite v3.1. Gate-schedule suite rows. relrfine passes the no-cost test on new C n3.
- 2026-11-05 v1.25. Gate-off-500 diverges on T10; not universal.
- 2026-11-05 v1.24. Suite v2: H⁻¹ + gate decoded .905 / .907 on C n1 at two seeds (VDAM .895); the old-scorer energy was misleading.
- 2026-11-05 v1.23. relrfine equals the exhaustive search on shifted C n3 in the suite.
- 2026-11-05 v1.22. VDAM direction ablation: a diagonal per-frequency direction is enough for poses (one seed).
- 2026-11-05 v1.21. relrfine wave F: passes Ribosembly, new C n3 and C_shift n1; known cost on old-suite C n3.
- 2026-11-05 v1.20. Section 4: shift-study facts (visit counts, stock σ bias, insertion verified, late step size, n1/n3 results).
- 2026-11-05 v1.19. Scoring: suite v3 lands (search robustness fix + rotation_ambiguous status; no metric change).
- 2026-11-05 v1.18. SGD: gate off after 500 recovers energy (one seed). T10: 6 of 10 posed, N_struct up to .890.
- 2026-11-05 v1.17. Shifts: relrfine (R + 1 px late step) is the leading candidate; R is size-invariant and costs nothing on new C n3 in the suite.
- 2026-11-05 v1.16. T10: held-out-likelihood seed selection poses 7 of 10 proteins without the truth; N_struct .817.
- 2026-11-05 v1.15. H⁻¹ + gate holds poses and frames on all sets at two seeds; energy is low and seed-dependent.
- 2026-11-04 v1.14. T10 status: partly solved (3 of 10); pseudo-symmetry is the obstacle.
- 2026-11-04 v1.13. Section 3: SGD + H⁻¹ + ρ_pc gate (no fudge, moving average or momentum) matches VDAM poses on C n1 and n3 (step 1g, one seed).
- 2026-11-04 v1.12. Shifts: RELION's learned-σ rule R becomes the leading candidate; T1 is demoted (transeval R1, shiftsched, ppcametrics).
- 2026-11-04 v1.11. Ribosembly: the pose tail is a signal limit shared with the true model; the map gap is in the PCs/state maps.
- 2026-11-04 v1.10. PC runaway is not split-specific; SGD without a PC control is unstable; the ρ_pc gate held on T10.
- 2026-11-04 v1.9. Section 3: known runaway risk of split curvature without PC control (t10solve evidence).
- 2026-11-04 v1.8 (owner). The translation prior is always centred at zero on the total offset; only the window follows the stored offset.
- 2026-11-04 v1.7 (owner). T1 is the current best guess for shifts; broad evaluation given to agent transeval (several shift sizes, the old suite, the cost on unshifted data, T1 vs exhaustive, option search).
- 2026-11-04 v1.6 (owner). New base shift schedule T1: 12 Å steps until HEALPix 3, then 6 Å, ±3 steps, 29 points, stored-best centring, 20 Å prior. T0 kept as a comparison arm.
- 2026-11-09 v1.65. Shifts, masking, latent-outlier route (all seed 11 unless stated; pose = direct fit, DIAGNOSTIC).
  - Shift mechanism (shiftdiag F4-F9). With the true model, 15 Å offsets are measurable to < .5 px from ~50 Å (P1); orientation-free templates are worse than no correction. ORACLE: true offsets for the 10% state only (background left at 0) pose it at 4.0°; a true-model start with the normal ±4 px recentred search locks offsets (1.96 → .45 px) and poses it (4.3°). So only the minority's offsets matter and the search works given a model; from a random start the 10% state never forms under the blur. Fraction curve at 15 Å (recipe v2, s11-13): D20_shift passes 3/3 (3.9-4.0°), D10_shift and D5_shift fail 3/3. Convergence curves (shiftfig): centred D10 poses state 42 at updates 750-1250 with offsets still 0; the shifted run never forms it; training-log signals (pmax, log-likelihood) do not distinguish the two runs.
  - Fail on D10_shift (113-131°): damping .3, RELION local-range rule, per-particle pmax gate, tight-then-learned prior, band start at shell 8, band-scheduled σ, model COM recentring, exhaustive ±5 px grid, our VDAM relrfine. None harms C_shift n1 (damp3 5.2° vs base 5.3°).
  - Real data: per-axis offset sd on as-picked stacks 6-12 Å (curated), 15-18 Å (fresh picks), 38-40 Å (11846, 10180); synthetic 15 Å sets are typical in Å and harsh relative to particle size. EMPIAR-10073 as picked vs pre-centred by consensus offsets: mean-map FSC .913 vs .973 (predeclared rule fires); pose error grows with the particle's offset beyond the ±2 px window (5° to 25° at 30-50 Å).
  - RELION-style image masking implemented in relax (local branch ppca-image-mask facc181a; pre-shift, soft circle, noise or zero fill; off-path within GPU noise). No effect on synthetic data: C n1 decoded .915 vs .915; D10_shift masked arms fail like unmasked. Default stays off; real-data test (CAHRA) running.
  - Latent-outlier route (top 10% |z|, retrain at T .15): unshifted D5 s16-18 3/3, D10 s11-15 5/5, D20 s11-15 5/5 pass the screen (suite pending). Step-matched control (plain base at T .15) fails all 3 seeds where the base failed, so the gain is the selection. The 10% fraction was chosen knowing D5's 5% share; a χ² tail rule does not estimate the minority size (the background is not χ²₂ under std, MAD or MCD scaling). Fraction sweep 3-30% running. On D5_shift the route fails (|z| AUC .72-.77 vs .93-.98 unshifted).
- 2026-11-09 v1.66 (evening). Shift program results; candidates, not yet adopted (seed 11 unless stated; pose = direct fit, DIAGNOSTIC).
  - Real data, as picked vs pre-centred by consensus offsets (mean-map FSC vs consensus): 10073 .913 vs .973, 10345 .700 vs .930, 10097 .670 vs .853, 11846 .228 vs .679, so shifts cost the current recipe on 4/4 sets. A zero-centred wide window (about ±45-50 Å, offset prior held at 20 Å; the rule real_bestv1 used on 10076) closes the gap on moderate shifts: 10073 .963, 10345 .916 (needs a GPU per arm at ±8 px, 69 GB). RELION-style recentring (±2 px around the stored offset): 10073 .963, 10345 .866. VDAM + offset refinement on 10345: .893. Nothing rescues 11846 (5.5 px spread): W .201, R .225. A fixed-mean start (published map at 30 Å, frozen to 1626) lifts 11846's beyond-start FSC to .29 (B .42) but not the poses. Polishing an as-picked model with wide shifts recovers about two-thirds of the gap, so the window matters during training. Candidate for real data: the wide held-prior window.
  - D10_shift solved truth-free on all particles at 3/3 seeds (3.8-4.0°, FSC .94-.97): stage 1, recipe v1b base with offsets moving only on visits with pmax ≥ .5 (gate5); stage 2, top-10% |z| subset retrained at T .15 (4.3-4.9°); stage 3, all particles from the subset model with the ±4 px recentred search (the protein is lost early and relearned by update 2000). About 20 single-run schemes failed. Generality: D20_shift passes, centred D10 passes (no regression), D5_shift fails (purity .285 but only ~2,850 copies).
  - Synthetic C/Ribosembly: on C_shift the FFT-centring phase grows only PC1; C_shift n1 v2fft splits into two hand families (4 of 11 states mirrored; own-hand decoded .912 vs .689 single-hand; the true C states are strongly chiral at box 64, so this is a real defect). Recentring from update 1 poses C_shift n1 (5.3°, ARI .97) and n3 (6.9°, ARI .26); late recentring is worse. Ribosembly_shift n3 fails every rule tried with today's recipe, including the 10076 wide rule; it has 2.5x less SNR than real 10076 and a 6-fold mass range. A truth-free hand check finds no mixed hand in 8 real runs.
  - Baselines: RELION 5 fails centred D10. cryoDRGN v3 on consensus poses is unaffected by shifts on C.
  - RELION-style image masking: no effect on synthetic data; on CAHRA (neighbours in the box) shifts localise (~10 Å vs 50-80 Å), the mean centres and compacts, the top clusters reconstruct better (FSC .74-.78 vs .64-.67). The σ-held control that separates the mask's centring-prior effect is pending.
  - Recipe v2's PC gate (half-batch gradient correlation) sits at its .25 floor from about update 100, so in practice it is a constant 0.25x PC step. Roey's power + full-data gate fix is being tested on VDAM-simple (gatefix_20261109).
  - Latent-outlier fraction: 10% is the smallest reliable fraction (3-5% subsets become unstable at the fine-resolution stages even with matched passes; 5% passes 3/4 with matched passes); a χ² tail rule does not estimate the minority size.
  - Box-128 oversampling (os1 vs v2): no divergence, cap irrelevant, 20-38% cheaper; scores pending.

- 2026-11-11 v1.67. Shift program results of 11-10 and 11-11 (seed 11; real-data pose is a DIAGNOSTIC against the consensus poses).
  - Roey's PC power gate on VDAM-simple (gatefix addendum 2, suite v3.2, 3000-update schedule): gate vs same-schedule control W2 T8 L1 over 11 rows, neutral on C_shift (n1 .880 vs .878, n3 .842 vs .841). Not adopted. The 3000-update schedule is itself fragile: Ribosembly_shift n3 .236 (gate) and .292 (control) against .755 for VDAM on 6000 updates (mean only .780); D10 control .173 against .790.
  - VDAM-simple vs SGD v2 on shifted sets is set-dependent: VDAM ahead on C_shift n1 (.873 vs .840) and Ribosembly_shift n3 (.755 vs .164); neither solves D10_shift or D5_shift alone. No switch.
  - D10_shift three-stage route confirmed on the suite: stage 3 at seeds 11-13 decoded .907 / .910 / .910 (mean only .669, true .998), pose 4.1°, 97% < 10°. D5_shift still fails: subset screens 86-104° (direct fit); VDAM on the subset fails (95°); full-set VDAM pending. Unshifted D5 subset route: frame check passes at 8/8 seeds.
  - Fixed mean (owner idea; fixedmean_20261109), design "start map low-passed to 30 Å, held to update 1626" (full or band-limited hold): closed. 10345 (consensus map): .921 vs the wide window .916 and pre-centred .930, same poses as the wide window, so no gain. 11846: every variant fails (published map .562 / 111°; independent 8EKI model .488 / 123°; with recentring .262). Ribosembly_shift n3 with the TRUE mean: 5.3°, 67% < 10° (state FSC .887, mean only .808) where the same recipe without it never locks (~123°); D10_shift with the true mean fails (133°). Redesigned test on 10180 running: mean = consensus at box 64 held for the whole run, ±18 px grid then local recentring, plus a full-resolution-search variant and a pre-centred control.
  - 10180's as-picked offsets carry a constant image-plane offset (source star mean +55 Å per axis, sd 40 Å; a 3D shift fit explains none of it). No volume shift can absorb it, and a zero-centred ±6 px window reaches 21% of particles (±18 px: 99%). The mean particle image's centre of mass recovers it truth-free (about 6 px after a ~1 px baseline). Global-recentring arm (constant removed, then the wide window) running.
  - EMPIAR-10076 twin (100k particles, box 64): as picked .794 / 41.8°, wide ±7 px .947 / 8.9°, pre-centred .949 / 5.1°; the wide window closes 99% of the gap. The earlier rows (6k particles, ±10 px; best-v1 ±15 px) had suggested no centring effect; superseded.
  - Real-data pose: next to the all-particle median, report a confident-half pose (particles with RELION MaxValueProbDistribution at or above the dataset median). On 10076 the pose misses are almost all particles RELION was unsure of (wide-window miss rate .97 in the lowest decile, .01 in the top); confident half: as picked 5.2° / 75%, wide 4.0° / 92%, pre-centred 3.9° / 95%. 10345 and 10097 keep a real gap on the confident half (pre-centred vs wide 86 vs 72%, 70 vs 42%). Pre-centred stacks use RELION's offsets, so they are not independent of RELION on its uncertain particles.
  - Wide window across real sets (FSC as picked / best wide-window variant / pre-centred): 10073 .913 / .961 / .973; 10345 .700 / .916 / .930; 10097 .670 / .709 / .853; 11846 .228 / fails / .679; 10180 .274 / .599 / .973; 10076 .794 / .947 / .949. Stays the candidate for real data.
  - RELION 5 on correctly signed inverted stacks: shifts cost RELION too (refine C 2.8° / FSC .991 vs C_shift 54.9° / .749; Class3D K=4 6.4° / ARI .258 vs 18.3° / .060).
  - Closed: box-128 oversampling arm p4b (.414 vs v2os1 .772, mean only .828). Ribosembly_shift n1: the suite set is canonical; ribodeep's builder copy is labelled "rd-built".
- 2026-11-11 v1.68 (afternoon). Shifted-set stage-3 round (ribodeep harvest 2; direct fit = DIAGNOSTIC, suite rows scoring).
  - D5_shift is signal-limited: VDAM-simple on all particles never locks (distinctive-protein training pose 109-117° from update 500 to 3000, direct fit 99.9°, 3% < 10°), like the SGD subset route (84-90°, stage 3 87.6°). Selection is not what fails it. Closed as out of reach for the current model at 5% / 15 Å shifts.
  - SGD stage 3 (all particles from a posed model, ±4 px recentred) never loses an existing pose: D10_shift kept at 4.1-4.2° (suite .907-.910 at s11-13), Ribosembly_shift n3 from the VDAM model 8.7° → 5.9°, 52 → 62% < 10° (direct-fit FSC .847, partial), Ribo_shift n1 a tie (.917 vs .916 direct-fit FSC).
  - VDAM as stage 3 destroys a good minority model (D10_shift 116° by update 500; Ribo_shift n3 loses the bootstrap and re-forms worse, 7.2°). Not used as stage 3.
  - Best known per shifted set: C_shift = VDAM-simple (n1 .873, n3 .834) pending the stage-3 SGD suite rows; D10/D20_shift = subset route + SGD stage 3; Ribo_shift n3 = VDAM then SGD stage 3 (partial); D5_shift unsolved.
  - cryoDRGN v3 on external consensus poses under suite v3.2: C_shift n1 and C n1 both .942 decoded (PPCA SGD v2 .840 / .915; mean only .837, true .979). Part of the gap is pose accuracy (continuous poses ~3.0° vs our 4.4-5.4°; random-subset BP .804 vs .766). Latent neighbourhoods favour PPCA (metric3 .85 / .93 vs .59 / .60). PPCA on C_shift pre-centred with those consensus offsets is scoring; it decides offsets vs model.
- 2026-11-11 v1.69 (evening). EMPIAR-10180 addendum 9 (realshift; seed 11, 3000 updates, box 64; pose = direct fit to the consensus poses, DIAGNOSTIC; confident half = RELION MaxValueProbDistribution >= the dataset median).
  - Global recentring SOLVES as-picked 10180 truth-free: one constant offset from the mean particle image's centre of mass (57.5 / 58.3 Å; consensus mean 55.4 / 55.2 Å) removed from every image, then the wide-window recipe (W). Mean-map FSC .958 (as picked .274, wide window alone .599, pre-centred .973); pose 5.3° / 77% < 10° (confident half 4.9° / 87%; pre-centred 4.3° / 91%, 4.1° / 97%). PC sizes match the pre-centred run (|PC|/|mean| .16 .14 .12 vs .16 .14 .11; as picked .28, wide window .35), and its top-3 PCs match the pre-centred run's (subspace cosines .95 .92 .85; PC1 vs PC1 .90). Same answer as the FFT learned-centre route of v1.56/1.57 (.961, 5.3°, 79%) with a one-shot, truth-free preprocessing step. Candidate for real sets whose picks share an offset; test on the next such set before adopting.
  - Miss tail (~20% of particles >= 30° off; pre-centred 5%) = the window edge after recentring: miss rate .06 / .08 / .14 / .26 / .50 / .85 for residual offsets 0-2 / 2-4 / 4-6 / 6-8 / 8-10 / >10 px (13.6% lie beyond ±8 px). It also concentrates in RELION-uncertain particles (lowest MaxValueProb quartile .42 vs top .08).
  - Owner design 1 (mean held at the consensus map for the whole run, ±18 px / 2 px grid, then ±3 / 1 px recentring from update 1001 seeded by each particle's wide-grid MAP; F1s, driver 35e9277f): poses at 7.0° / 69% < 10° (confident half 6.3° / 79%), already 8° / 61% by update 250 and little better after. Worse than learning the mean after global recentring. Its misses are particles whose wide-grid offset was wrong (median 8.3 px off vs 0.5 px for hits); recentring cannot rescue them. The first F1 run's drift (batch mean offset 55 → 48 Å) came from unvisited particles seeded at zero (with-replacement batches leave 22% unvisited by update 500); seeding them with the batch mean removes the drift (57.5 → 57.9 Å) but barely changes the pose tail (F1 vs F1s misses Jaccard .91). Holding the mean fixed puts a whole-particle mean-mismatch mode into PC1 (|PC1|/|mean| .28-.32), on the pre-centred stack too (control C: poses 4.3° / 90%, same as pre-centred, so the code is sound). Not a route. Owner design 2 (full-resolution search from update 1, F2s) is running.
- 2026-11-11 v1.70 (owner decision). EMPIAR-10180's shared offset is a property of the deposited stack: every micrograph carries it (per-micrograph mean sd 3.3 px vs the 32.7 px offset), it is fixed in the image frame (rotating by psi averages it to ~0), and the star's offset priors equal the offsets, so the particles were never re-extracted centred. Owner: "not so interested in this failure mode if it's just a one off". 10180 is dropped as a shift test case; addendum 9 closed (design 2, full-resolution search from update 1 with the mean held: 7.0°, 69% < 10°, same as design 1). Global recentring stays in the notes as a preprocessing option for stacks whose mean particle image is off-centre, not part of the recipe. Real-data shift work continues on other datasets.
- 2026-11-11 v1.71 (evening). Suite v3.2 rows landed (closer; decoded / bp; suite pose median / p90, % < 10; refs mean only / true model).
  - New best on C_shift (ribodeep stage-3 SGD from the VDAM model, "s3hi"): n1 .887 / .906, 4.5° / 7.3, 99% (VDAM-simple .873, SGD v2 .840; refs .837 / .979; cryoDRGN v3 on external consensus poses .942); n3 .857, 5.7° / 13.7, 83% (VDAM .834; refs .837 / .948). D20_shift s3hi .936, 3.9°, 99% (refs .849 / .991). Ribosembly_shift n1 (suite set) stage-3-from-baseline .878 / .854, 4.1°, 98% (refs .780 / .937). So "VDAM, then SGD stage 3" is the best known route on C_shift, D20_shift and Ribo_shift n1; Ribo_shift n3 and D5/D10 rows still landing.
  - Subset fraction matters: the D10_shift subset route at 20% / 30% subsets (zb4) poses only at seed 13 (f20 .793, 4.6°, 93%; f30 .593, 59%); seeds 11-12 fail at both (1-29% < 10). The 10% subset route stays the standard (3/3 seeds, v1.67). D20_shift is robust to the fraction (f20 / f30 at 3 seeds .894-.904, 98-99%). D5_shift fails at every fraction (frame-check oracle medians 101-119°, real failures); VDAM on the 20% subset also fails (95°, direct fit).
  - Decoded score can mislead: rd_s3hi on D5_shift decodes .870 / .907 with pose 89.5° (12% < 10; oracle frame 72.9°). Report pose next to decoded, always.
  - RELION 5 on Ribosembly n3, correctly signed stacks: Class3D fails even centred (42.9°, ARI .127, best-class FSC .660), so this set is beyond RELION regardless of shifts; shifts still break RELION's K=1 InitialModel (3.6° → 114.5°).
  - T7late on Ribo_shift n3 fails (.172, 122°).
  - PPCA on C_shift pre-centred with the external consensus poses' offsets scored .662 / 116° (below mean only .837 and below as-picked .840). Not a centring bug (sign, units and the centre-of-mass check all pass; training offsets settle to 0.25 px rms vs 3.57 raw): the run split its 11 states between the two hands (6 in one frame at 4.8° / 93% < 10°, 5 in the mirror frame at 4.6° / 93%), so a single-frame suite registration fails. A per-hand rescore (with own-hand baselines for the raw C_shift and C n1 rows) decides whether pre-centring lifts PPCA; one seed cannot say whether pre-centring caused the split.
- 2026-11-11 v1.72 (night). relax K-class ab initio at RELION InitialModel defaults (relaxab; K = 1 and 4, seed 11, 200 iterations, 6 px / 2 px offset search) on the 7 shifted sets: it fails every one. Pose after one global frame, K1 / K4: C_shift n1 54.8° / 114.4°; C_shift n3 114.5° / 125.1°; Ribo_shift n1 100.1° / 96.7°; Ribo_shift n3 114.0° / 119.9°; D5 / D10 / D20_shift distinctive 116-126° except D20 K4 35.6° (20% < 10). Offset error never drops below ~2.1 px (start 2.9 px); K4 ARI <= .06. relax matches real RELION 5 InitialModel on the same stack (C_shift n1 K1 54.8° vs 54.9°), so the failure is the default search under shifts, in both programs. There is no shifted set where K-class at defaults works and our PPCA fails: our VDAM-simple / stage-3 route poses C_shift n1/n3, Ribo_shift n1 and D20_shift; both fail D5_shift (and D10_shift without the subset route). Map-level readouts still running.
- 2026-11-11 v1.73 (night). More suite v3.2 rows (closer; decoded / bp; pose median / p90, % < 10; refs mean only / true).
  - Ribosembly_shift n3: stage-3 SGD from the VDAM model .797 / .715, 5.6° / 143, 66% (refs .780 / .868), the first shifted Ribo n3 row above mean only (VDAM-simple .755; stage-3 VDAM .764). Still partial: a third of particles unposed.
  - Ribosembly_shift n1: stage-3 SGD .880, 4.1°, 98% (recipe-v2 baseline .878; VDAM .841): stage 3 ties the baseline here.
  - D5_shift fails in all 6 subset-route rows (20% / 30% subsets, seeds 11-13; best .518, 7% < 10; oracle-frame medians 101-119°).
  - C_shift pre-centred (s11) own-hand rescore: .827 vs .910 for the raw C_shift row under the same own-hand scoring, so this seed is uninformative about offsets; seeds 12-13 running.
- 2026-11-12 v1.74. Real-data transfer test of stage-3 SGD (ribodeep Addendum 7; seed 11, as picked, 3000 updates; mean-map FSC 1-31 vs the consensus map; pose = C3-aware direct fit, a labelled diagnostic). Real data has no mean-only / true-model refs; the refs are the tracker's W (best as picked) and B (pre-centred ceiling). Rows VW = VDAM-simple with the wide window (driver coupled_arm_fft2_ablate.py 0037f627); S3 = SGD stage 3 seeded with VW's poses and offsets (coupled_arm_shift.py a3cc9708 + seed_offsets.py f0d810ae).
  - 10073: VW .942, S3 .956 | W .963, B .973 (TIE).
  - 10345: VW .894, S3 .908 | W .916, B .930 (TIE).
  - 10076: VW .914, S3 .941 | W .947, B .949 (TIE).
  - 10097 (HA): VW .829 (11.0°), S3 .792 (13.7°) | W .709 (23.6°), B .853 (PASS for S3 at +.083, but VW alone is higher). VW closes 83% of the W-B gap: the best as-picked HA row so far.
  - Decision: stage 3 does not transfer by the predeclared rule (1 PASS / 3 TIE, rule needs 3 PASS); it stays the synthetic shifted-set recipe only. For real as-picked data keep W as the default; run VW as well where W leaves a large gap to B (10097). On 10097 VW's PC1 is a per-image grey-level (DC) mode (z'1 vs image mean, Spearman .96; vs norm -.20, defocus .05), and S3 adds high-frequency PCs 3+.
  - cryoDRGN-AI C_shift seeds 12/13 (training poses, diagnostic): no hand split (93.5% / 92.5% < 10°, every state .83-.96); the seed-11 split (v1.71 correction) was seed-specific. Suite decoded rows pending (rule: vs .840 single-frame, offsets matter at >= .87).
- 2026-11-12 v1.75. Hard real sets: held full-resolution mean, image mask, real RELION refine (realshift Addendum 10; seed 11, update 3000). Pose = direct fit to the consensus, a diagnostic (11846 vs cryoDRGN-AI poses; 10097 C3-aware); for held-mean arms the FSC only checks the held map against its source, so they have no map score.
  - 11846 (refs: as picked .228 / 128°; wide window .201 / 127°; pre-centred .679 / 14.0°, selection 7.3°). Held published map (M1) 99.7°, 18% < 10: selection particles 14.4°, the other 12,900 at 123°. The held map fixes only the particles it was built from, and it is not truth-free. + image mask (M3) 98.8°: no change. Wide window + image mask (M2) .218 / 126°: no change from W. Real RELION 5.1.1 auto-refine from the published map at 40 Å: .240 / 129°, selection 128°; the positive control fails, so this as-picked stack defeats standard refinement too (RELION's offset window ±5 px vs consensus sd 5.5 px). 11846 needs re-extraction, not a training fix.
  - 10097 (refs: wide window .709 / 23.6°; pre-centred .853 / 8.1°). Held map 13.7° (+ mask 13.6°): partial, misses flat over offset bins. Wide window + image mask .768 / 17.2° (mask helps W by the >= 5° rule). VDAM-simple with the wide window (v1.74) stays best: .829 / 11.0°.
  - Decision: the image mask (soft circle, RELION edge and noise fill, diameter = particle size; relaxmask facc181a) is optional for W on real data: it helped 10097 and was neutral on 11846; VDAM wide + mask is untested. Held-mean arms stay oracle diagnostics.
- Shift ladder on C_shift (gatefix; suite rows pending). relax K = 1 at defaults (its own analysis, global frame + hand): pose median 2.8 / 9.3 / 25.8 / 54.8 / 117° at offset sd 0 / 5 / 10 / 15 / 22 Å, map vs true mean .991 / .947 / .840 / – / .599. It breaks between 5 and 10 Å and barely recovers offsets even at 5 Å (error .82 px vs true .98 px). This explains the real-data K-class failures (real sd 6-18 Å typical). Our VDAM-simple training pose stays 4.7-5.1° up to 22 Å (diagnostic).
  - Caveat: coupled_arm_v2.py (8294284e) does not log training rotations once oversampling starts (--os-from-update 1626), so v2os1 training-pose trajectories are frozen after 1626 although training keeps improving (centred C n1: logged 8.2° vs suite 4.0°). Read v2os1 poses from the suite only; offsets are logged throughout.
- relax K-class map fits on the shifted sets (relaxab; true-mean FSC; refs C .837 mean only / 1.000 true, Ribo .780 / .980): C_shift n1 K1 .768 (real RELION .749), K4 best class .662; C_shift n3 .595 / .481; Ribo_shift n1 .254 / .625; Ribo_shift n3 .186 / .198. All below mean only, consistent with the failed poses (v1.72).
- 2026-11-12 v1.76. CAHRA (EMPIAR-13973, 200k subset, truth labels for scoring only): relax K-class at 600 iterations (relaxab). ARI / NMI: K4 .174 / .231 (200 it .167), K6 .161 / .226 (.131), K8 .150 / .209; every run levels off by iteration ~450, so longer runs will not help. Best PPCA arm ARI .083. Per species unchanged: GDH one clean class (K4 recall .70, purity .76), beta-amylase shares a class with IgG, ADH <= .37 pure, aldolase and IgG never separate, extra classes split PKM2. The class maps are not proteins: mask-filling speckled balls (Rg 41-54 A vs 20-44 A for the particles); the GDH class is a half-moon with a straight edge. So K-class sorts on coarse 2D features. Caution for all CAHRA scores: each species sits on its own micrographs, so micrograph-level features (ice, defocus, carbon edge) leak species identity; the per-species reference maps (cahra O1/O3, running) will set the structural ceiling.
  - Leak check (relaxab, 5-fold logistic regression on the 200k): CTF metadata + outer-ring image stats predict species at macro-AUC .684, accuracy .438 (chance .404; defocus carries most of it, one session per species). K4-600 as a classifier: accuracy .518; PPCA latents macro-AUC .77-.79. Both carry signal beyond metadata, by a modest margin. The GDH half-moon is not an image or border feature (no more in-box edge signal, same distance to the frame edge); most likely a mask-clipped density from wrong poses.
- 2026-11-12 v1.77. CAHRA (EMPIAR-13973, 200k subset; truth labels for scoring only).
  - Oracle RECOVAR (cahra O2): poses fixed from per-species refinements against each species' own map, box 64. Species from latents: macro-AUC .999, accuracy .975 at zdim 10 (zdim 4: GMM K = 6 ARI .816). Best PPCA: macro-AUC .791, ARI .083; relax K4-600 ARI .174; metadata-only leak baseline: accuracy .438, AUC .684. A capacity result, not a ceiling: the oracle frame encodes the label (junk included). Box 64 and a linear model are not the limit; our gap is the pose search and one shared frame for six shapes. Label-free ceiling = O3 (classification against the six maps), pending.
  - Re-picking by template matching (harvest pilot, 504 micrographs + 32 phase-randomised nulls): NO-GO. All local maxima recall 31% of the original picks (11 Å centring error); with a fixed matcher (CTF-weighted templates, 110 Å normalisation circle), the original picks score below the null maxima for blob, PPCA and oracle templates alike (<= 0.8% above the null p99). Single-particle detection is ~2 sigma (beta-amylase ~0.1) on the released F30 sums (30 of 41 e/A^2), so template quality is not the limit. The picks themselves are real and centred (patch averages). Re-picking via our means is dropped for small proteins; scoring/recentring existing picks is a proposal only.
  - Engine limit: relax's PPCA CUDA kernels accept q <= 16 (ppca_moment_backproject.cuh, basis size P = q+1 <= 17); the CAHRA q20 arm runs as q16.
- 2026-11-12 v1.78.
  - Shift ladder on C_shift n1, suite v3.2 rows (decoded / pose median, fraction < 10; refs mean only .837 / true 1.000; centred C n1 SGD .915 / 4.35°). SGD recipe v2 (S): 5 Å .869 / 5.05°, .97; 10 Å .846 / 5.14°, .95; 15 Å .801 / 6.31°, .76 (below mean only). VDAM-simple (V): 5 Å .875 / 4.64°, .97; 10 Å .875 / 4.66°, .97 (22 Å pose only, 5.1° training). relax K=1 at defaults breaks between 5 and 10 Å (v1.75). Reading: our VDAM-simple is flat to 10 Å; SGD v2 degrades from 10 Å and falls below mean only at 15 Å; both lose ~.04 decoded vs centred already at 5 Å, so shifts cost heterogeneity accuracy even when poses hold.
  - CAHRA label-free classification against the six per-species RELION maps (cahra O3; one iteration, global search, box 64): ARI .211, NMI .254 (F30 .243 / .279); 54% of particles in their own species' class (relax K4-600 .518 as a classifier; metadata-only .438). Own-class recall GDH .78, beta-amylase .72, ADH .56, IgG .48, PKM2 .45, aldolase .44; main confusion among the similar-sized tetramers (25% of PKM2 in the aldolase class, 13% in ADH). Only the GDH and ADH maps were clean, so this is a lower bound; map cleanup and a 10-iteration local follow-up (O3b) running. Even with oracle references, single-particle species assignment on CAHRA is hard.
  - cryoDRGN-AI on truth-free pre-centred C_shift, seeds 12/13 (no hand split in training): suite decoded .874 / .871, just over the predeclared .87 line ("offsets matter"), but suite pose is 56°, 42% < 10 against 93% < 10 in training. PROVISIONAL: the gap probably comes from suite inference on the uncentred stack; own-hand rescore pending before this counts.
  - CORRECTION to v1.75 (11846): the real-RELION refine control (.240 / 129°) is WITHDRAWN as a setup bug. Its input stack was contrast-inverted for RELION (relionctl s0 check: relion_reconstruct of that stack with the cryoDRGN-AI consensus poses gives a clean map that is the exact negative of the reference, unaligned FSC -0.70, +0.72 after a sign flip; pose/offset conventions verified). The original as-picked stack is already protein-white in RELION's convention; negating the published map was right, inverting the images was wrong. "Even RELION fails on 11846" and "11846 needs re-extraction" no longer follow; the fixed controls (all 100k, published selection only, pre-centred) are running. The PPCA rows (held map, mask) are unaffected.
- 2026-11-12 v1.79. EMPIAR-11846 real-RELION refine controls redone with the correct contrast sign (relionctl; RELION 5.1.1 auto-refine from the published map at 40 Å, box 64; pose = direct fit to the cryoDRGN-AI poses, diagnostic; pass = median <= 20° with >= 30% < 10°).
  - All 100k as picked: 121°, 12% < 10, map vs published .63 (18.6 Å): fails. Published-selection particles only (35,527): 102°, 26% < 10, map .71 (14.9 Å): fails. Pre-centred stack (100k): 13.2°, 45% < 10 (selection 5.6°, 67%), map .79 (14.4 Å): passes. All three maps resemble the published map.
  - By true offset (selection run): 6-9° and 3-5 Å offset error up to 40 Å; 102° at 40-50 Å, 120° at 50-60, 131° at 60-80, 143° beyond. RELION's recovered offsets stall at ~22-26 Å. Most 11846 picks are 40-80 Å off-centre.
  - Reading: 11846 as picked needs recentring before refinement; standard refinement, even from the correct map, does not reach offsets beyond ~40 Å. Junk is not the cause (the selection-only run fails the same way). This restores, on a valid control, the v1.75 conclusion withdrawn earlier today.
- 2026-11-12 v1.80. Real RELION InitialModel on C_shift n1 (relionctl; K=1, 200 iterations; pose global frame + hand vs truth; refs mean only .837 / true 1.000 for the suite, K=1 map FSC vs true mean here).
  - Defaults on 5.1.1: 55.1°, 7.8% < 10, offset error 2.1 px (true shifts 2.9 px median), map .768; 5.0.0 defaults 54.9°: the version is not a factor.
  - Wider initial offset window (8 px = 48 Å vs 6 px = 36 Å): identical (55.1°, map .768). VDAM's auto-sampling shrinks the offset window to ~9 Å by iteration 10 and ~7 Å by iteration 50, after which it is a local travelling search, so the initial range does not matter. A user setting does not fix RELION ab initio on 15 Å offsets; the standard 2D-classification + recentring route is being scored. relax K=1 matches RELION at 15 Å (54.8°).

