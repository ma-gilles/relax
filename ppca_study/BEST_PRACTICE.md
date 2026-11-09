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
- Code changes go in a copy of the driver or on a branch. Never land on relax main except through the shared landing tools. RECOVAR changes need the owner's OK.

## 9. Open questions (current work)

| Question | Where | Status |
|---|---|---|
| SGD PC control; SGD + H⁻¹ | round17 step 1g | running |
| Shift scheduling on shifted data | shift_sched_20261104 | data building |
| D-set seed dependence: does the SGD candidate flip the distinctive protein into a second frame on some seeds? (abinit_study handoff, 2026-11-05: many SGD arms at ~50% on D40 at one seed of three) | round17 step1h_sgddiag, seeds 12-13 on D10/D20/D40 | Answered 2026-11-05: yes. The candidate fails D10 at 2 of 3 seeds and D20 at 1 of 3; VDAM fails D10 at 3 of 3. Failures are scattered mis-poses, not a clean second frame. The noise anneal (σ² × 3 → 1 over updates 1-500) is not adopted: it rescues 2 of 3 failures but makes D10 s13 worse and costs Ribosembly 14 points and T10 one protein (no harm on C, D40, H1). Seed selection by held-out likelihood fails, both mean and paired per-image (spread < 1 nat/image). PC-shrink test on the 3 failing seeds: floor .5 rescues 1, gate-off-500 rescues 2 (crashes T10); biased by seed selection. Fresh-seed test (seeds 14-16, direct fit, diagnostic): floor .25 passes 3 of 6, floor .5 passes 4 of 6, a tie under the predeclared rule (≥3 needed). Floor .25 stays. Pooled floor .25 over seeds 11-16: D10 2 of 6, D20 4 of 6. D10/D20 remain seed-sensitive with no fix found |
| Off-centre map (common centre-of-mass offset) with the zero-centred shift prior | trans_eval_20261104 | queued |
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
