# RELION 5 features relax does not support yet

This is the roadmap for the RELION 5 Refine3D, Class3D and InitialModel options and behaviours that relax
lacks, for single particles and for subtomograms. Strike a row off by adding the commit that lands it and
the feature check that qualifies it. Current status of what relax does support is in
[em_status.md](em_status.md).

## Method

Recorded 2026-10-04 at relax main 78a37c6. RELION 5.0.1's `relion_refine` argument list
(`ml_optimiser.cpp`, `MlOptimiser::parseInitial`, 190 options) and the GUI job tabs were compared with the
option lists of `relax refine`, `relax class3d` and `relax initial_model` and with relax's explicit
refusals. An option that is absent from relax's command line is counted as unsupported; the options were
not each run. "Use" is an estimate of how often RELION users set the option, from the GUI defaults and
common practice; it is not a measurement. Rows marked (verify) are ones where relax has related code and
the exact coverage needs a run before the row is trusted.

## A. Common in everyday processing

| # | RELION feature (option / GUI field) | Jobs | relax today | Use | What support takes | Landed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Reference (solvent) mask, `--solvent_mask`; with it "Use solvent-flattened FSCs", `--solvent_correct_fsc` | Refine3D, Class3D, subtomogram | spherical mask from the particle diameter only | very common: nearly every final Refine3D and every focused job | read the mask, apply it in the M-step's solvent flatten and to the start-up reference; a solvent-corrected FSC for the resolution estimate (relax deleted its unused, never-produced option in 765d36d). Medium. | |
| 2 | Classification without alignment, `--skip_align` (GUI "Perform image alignment: No") | Class3D, subtomogram Class3D | not available | very common: focused classification after a consensus refinement | an E-step over classes only, at the input poses and offsets; pass 2 and the M-step unchanged. Small to medium. | |
| 3 | Local angular searches from the start, `--sigma_ang` (GUI "Perform local angular searches", "Local angular search range"), and Refine3D started at a local sampling from input priors | Class3D, Refine3D, subtomogram | Refine3D reaches local searches through auto-sampling; a Class3D local search with a prior width from the input STAR is not a command-line option (verify what `--initial-pose-source input-star` covers) | common: second-round refinements, focused refinement, classification with local searches | the local-search engine exists for K=1; needs the K>1 route and priors read from `rlnAngle*Prior` or the input angles. Medium. | |
| 4 | Limit resolution in the E-step, `--strict_highres_exp` (GUI "Limit resolution E-step to") | Class3D | not available | common in Class3D | cap the current image size from a resolution in Angstrom. Small. | |
| 5 | More than one GPU, MPI (`relion_refine_mpi`, `--gpu 0:1`) | all | one process, one GPU | common on clusters for large data sets | split the halves or the batches over devices; a design item, not a port. Large. | |
| 6 | Blush regularisation, `--blush` | Refine3D, Class3D, InitialModel, subtomogram | not available | RELION 5's headline feature; growing, mostly for small or hard targets | call RELION's Blush network on the half maps in the M-step (RELION does it through its external-reconstruct hook, row 22). Medium, plus a PyTorch dependency. | |
| 7 | Helical reconstruction (`--helix`, the `--helical_*` family, `--bimodal_psi`, tube masks, symmetry search) | all | not available (point groups only) | a whole user community; none for everyone else | helical symmetry in the reconstruction, priors along the tube, segment handling, helical masks. Large. | |

## B. Used regularly, by fewer jobs

| # | RELION feature | Jobs | relax today | Use | What support takes | Landed |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | Fast subsets, `--fast_subsets` | Class3D | not available | occasional, large data sets | RELION's subset schedule for the first iterations. Small. | |
| 9 | "Use finer angular sampling faster", `--auto_resol_angles --auto_ignore_angles` | Refine3D | the convergence helper knows the rule; no command-line option (verify) | occasional | expose and qualify. Small. | |
| 10 | Ignore CTFs until the first peak, `--ctf_intact_first_peak` | all three (a GUI field in each) | not available | occasional, mainly InitialModel and early classification | a variant of the exact CTF operand. Small. | |
| 11 | Skip padding, `--pad 1` | Refine3D, Class3D | fixed at padding 2 (InitialModel takes 1 or 2) | occasional, to save memory | padding 1 through the projector and backprojector sizes; touches the stable-window classes. Medium. | |
| 12 | Symmetry relaxation, `--relax_sym` | Refine3D, Class3D | not available | occasional (pseudo-symmetric complexes) | sample the symmetry mates in the E-step prior. Medium. | |
| 13 | Solvent filled with noise instead of zeros (GUI "Mask individual particles with zeros: No") | all | zero masking only | occasional (the GUI default is zeros) | RELION's random-noise fill in the preprocess kernel. Small to medium. | |
| 14 | Allow coarser sampling, `--allow_coarser_sampling` | Class3D | not available | occasional | the sampling-coarsening rule from the accuracies. Small. | |
| 15 | InitialModel optics: anisotropic magnification; optics groups on several image grids | InitialModel | refused (premultiplied images, odd and even aberrations are supported) | occasional (merged data sets) | magnification in the VDAM scorer; the several-shape loop exists for Class3D and is not wired for VDAM. Medium. | |
| 16 | Subtomograms: InitialModel with symmetry (relax runs the subtomogram VDAM in C1), InitialModel optics beyond premultiplied images, magnification, a reference mask (row 1), no alignment (row 2) | subtomogram | refused or absent | common for rows 1 and 2; occasional for the rest | per-tilt versions of the single-particle paths. Small to medium each. | |
| 17 | Continuing a run, `--continue` | InitialModel; Refine3D and Class3D on several image shapes | `relax refine` and `relax class3d` continue from their run files (`--continue <output>/run_itNNN_optimiser.star`, written every iteration by default), except for particle STARs with several image shapes, which are refused; subtomogram runs (verify). `relax initial_model` has only a diagnostic one-iteration continuation, no user-facing resume | common on clusters with wall-time limits | read the VDAM state back from its written iteration (model, gradient moments, sampling) and rebuild the E-step input; the several-shape resume needs the per-shape state in the run files. Medium. | |

## C. Rare or legacy

| # | RELION feature | Use | Note | Landed |
| --- | --- | --- | --- | --- |
| 18 | `--solvent_mask2`, `--lowpass_mask` with `--lowpass`, a fixed tau spectrum `--tau`, `--tau2_fudge_scheme` | rare | small each once row 1 exists | |
| 19 | `--local_symmetry` | rare | medium | |
| 20 | `--skip_rotate`, `--limit_tilt`, `--offset_range_x/y/z`, `--psi_step`, `--sigma_rot/tilt/psi` | rare | sampling variants, small each | |
| 21 | `--ctf_phase_flipped`, `--only_flip_phases`, `--pad_ctf`, `--ctf_uncorrected_ref`, running without `--ctf` | rare | small each | |
| 22 | `--no_norm`, `--no_scale`, `--fix_sigma_noise`, `--fix_sigma_offset`, `--always_cc`, `--incr_size`, `--coarse_size`, `--adaptive_fraction` (relax: an environment variable only), `--nr_parts_sigma2noise`, `--NN`, `--dont_skip_gridding`, `--center_classes`, `--skip_maximize`, `--external_reconstruct` (needed by row 6), `--abort_at_resolution`, `--failsafe_threshold` | rare | expert switches; most are one constant each | |
| 23 | Self-organising map and class replacement in gradient classification (`--som*`, `--class_inactivity_threshold`) | rare | medium | |
| 24 | RELION 4 style 3D subtomogram volumes with 3D CTFs (`--normalised_subtomo`, `--ctf3d_not_squared`, `--skip_subtomo_multi`) | legacy: RELION 5 uses 2D stacks, which relax supports | not planned | |
| 25 | `--reuse_scratch`, `--keep_scratch`, `--onthefly_shifts`, `--no_parallel_disc_io`, `--free_gpu_memory`, the SYCL and CPU back ends | rare or not applicable | computation options | |

## Supported already, for orientation

Auto-refine with split halves and `--low_resol_join_halves`, `--firstiter_cc`, adaptive oversampling,
point-group symmetry, `--continue` for Refine3D and Class3D (row 17 lists its limits), `--preread_images`
and `--scratch_dir`, several optics groups including other pixel sizes and boxes (Refine3D, Class3D),
CTF-premultiplied images, odd and even aberrations, anisotropic magnification (Refine3D, Class3D), RELION 5
2D-stack subtomograms in Refine3D, Class3D and InitialModel, VDAM InitialModel with K classes, symmetry
applied at the end, flattened solvent and zero masking.

Other RELION programs are outside this list: 2D classification, multi-body refinement, CTF refinement,
polishing and post-processing.

## Suggested order

By users reached per unit of work: rows 2, 4 and 1 first (focused classification and masked refinement
are the two most common things a RELION user cannot do in relax today), then row 3, then the subtomogram
counterparts of rows 1 and 2 (row 16), then row 17's InitialModel resume, then Blush (row 6) and several
GPUs (row 5) as the two larger projects; helices (row 7) only if that community is a target.
