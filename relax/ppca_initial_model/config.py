"""Explicit provisional pilot policy (implementation plan sections 3 and 5)."""

from dataclasses import dataclass

import numpy as np

from relax.steps.sgd_schedule import (
    DEFAULT_GRAD_EM_ITERS,
    compute_phase_lengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
    default_subset_sizes_for_3d_initial_model,
)


@dataclass(frozen=True)
class Config:
    q: int = 2
    iterations: int = 200
    seed: int = 11
    stages: tuple = ((1, 4, 1), (61, 8, 2), (111, 16, 3), (161, 32, 3))
    # Adaptive oversampling: 0 scores every stage's full pose grid; 1 runs the last stage with RELION's two passes
    # (each image's significant coarse samples, then their children; relax/ppca_refinement/oversampled_stream.py)
    # and the earlier stages dense (iteration_loop.oversampled_update).
    oversampling: int = 0
    # With oversampling 1: the first update that oversamples (None: the last stage's first update), and whether the
    # children are rotations only (the shifts keep the stage's grid).
    oversampling_from_update: int | None = None
    oversampling_rotations_only: bool = False
    # The adaptive fraction of the posterior mass the significant samples hold (RELION's --adaptive_fraction).
    target_mass: float = 0.999
    # Significant coarse samples kept per image at most (RELION's --maxsig: 100 per class for gradient runs).
    max_significant: int = 100
    shift_range: float = 6
    shift_step: float = 2
    image_batch_size: int = 150
    rotation_block_size: int = 512
    stream_full_fine_rows: bool = False
    # The default engine: the GPU stream over the full pose grid of each stage. False selects the
    # host-mask dense engine (q <= 2 only), the float32 reference of the stream's tests.
    stream_coarse_recompute: bool = True
    fine_devices: int = 1
    stochastic_batch_size: int | None = None
    stochastic_all_iterations: bool = False
    balanced_stochastic_halves: bool = False
    checkpoint_interval: int = 1
    skip_final_embeddings: bool = False
    optimizer: str = "vdam"
    # The multiplier of G / c, with c the band maximum of the metric trace over all 1 + q channels: rates comparable
    # across q are sgd_learning_rate / (1 + q) (docs/math/ppca_momentum_sgd.md).
    sgd_learning_rate: float = 0.4
    # Momentum B of the velocity B v + (1 - B) sgd_learning_rate G / c; 0 is plain SGD.
    sgd_momentum: float = 0.9
    # VDAM divides its direction by the root of the half-set second moment; False keeps the rest of VDAM without it.
    vdam_second_moment: bool = True
    # Momentum SGD variants (docs/development/ppca_optimizer_axes.md, "Gated momentum SGD"); the defaults are
    # the plain step. sgd_halfset_gate applies VDAM's half-set shell gate to the step (the direction
    # gate G - (1 - gate) diag(H) theta; the run then keeps VDAM's half-set moments as well).
    # sgd_step_scale: "trace" divides by the band maximum of the backprojected metric trace sum_p w_p tr LHS_p,
    # "exact_diagonal" by that of the exact projection-Hessian diagonal sum_p w_p^2 tr LHS_p (GPU streams).
    sgd_halfset_gate: bool = False
    sgd_step_scale: str = "trace"
    # Noise annealing: the E-step of update t uses the noise estimate times
    # 1 + (noise_anneal_factor - 1) max(0, 1 - t / noise_anneal_updates); 1.0 and 0 leave it unchanged.
    noise_anneal_factor: float = 1.0
    noise_anneal_updates: int = 0
    # Precision of the streamed engine's four GEMMs: "auto" (the default: tf32 on GPUs of compute
    # capability 8.0+, fp32 elsewhere), "tf32" (TF32 tensor-core products, float32 accumulation) or
    # "fp32" (exact float32 products). Section 14 of docs/math/vdam_ppca_algorithm.md has the evidence.
    gemm_precision: str = "auto"
    # Particle images in host memory: "auto" (the default) reads the whole stack once at start-up
    # when it takes at most PREREAD_MEMORY_FRACTION of the job's memory, otherwise reads each tile
    # from disk; "on" and "off" force either. A runtime setting, like gemm_precision.
    preread_images: str = "auto"
    # Pass 2 skips the pose rows on which every image of a tile has less posterior mass than this
    # (0: visit every row), when that leaves at most half the rows. A runtime setting;
    # full_row_kernels._pass2_rows has the bound; section 14 of docs/math/vdam_ppca_algorithm.md the evidence.
    pass2_mass_floor: float = 1e-10
    # Subtomogram particles from HEALPix order 2: score pass 1 on the order hp - 1 grid first and the stage's grid only
    # on the rows whose nearest coarse rotation is a particle's significant coarse rotation (those of its poses
    # holding target_mass, at most max_significant) or a neighbouring cell of one (iteration_loop._expectation).
    # A runtime setting, like pass2_mass_floor; off by default.
    tilt_coarse_pass: bool = False
    # With tilt_coarse_pass: the coarse grid's HEALPix order for every stage above it (None: each stage's hp - 1).
    # 2 gives order 3 stages a coarse pass at order 2 (one level, as RELION's oversampling 1) and order 4 stages
    # one at order 2 too (two levels, as RELION's oversampling 2). A runtime setting, like tilt_coarse_pass.
    tilt_coarse_order: int | None = None
    # Newton SGD (docs/math/ppca_momentum_sgd.md, "Newton SGD"; momentum SGD with momentum 0). sgd_direction "coupled" steps
    # along the per-frequency H_f^-1 G_f, scaled at the first update to the split-curvature step's norm.
    # sgd_pc_gate_floor F multiplies the PC step per shell by clip(rho, F, 1), rho the moving correlation of the
    # two half-batch PC gradients (None: no gate). sgd_lr_schedule: a .npy of per-update steps s; the rate of
    # update t is sgd_learning_rate s(t) / s(1).
    sgd_direction: str = "trace"
    sgd_pc_gate_floor: float | None = None
    sgd_lr_schedule: str | None = None
    # sgd_lr_match M: the base rate is set at the first update to M c_mean / max tr H, so the mean's step equals
    # the shared-trace step at rate M there (None: sgd_learning_rate).
    sgd_lr_match: float | None = None
    # sgd_fixed_step T (coupled direction): the step is T s(t) / s(1) times the damped Newton direction H^-1 G, s the
    # sgd_lr_schedule's steps (T throughout without one); no update-1 norm match (k) and no learning rate.
    sgd_fixed_step: float | None = None
    # Newton SGD stabilisers (opt-in). sgd_pc_min_shell N: no PC step on shells below N (1 freezes the PCs' DC term).
    # sgd_eig_floor_rel r: floor every frequency's metric eigenvalues at r times the band's largest eigenvalue too.
    sgd_pc_min_shell: int = 0
    sgd_eig_floor_rel: float | None = None
    # sgd_mean_gate_floor F: multiply the mean's step per shell by clip(rho, F, 1), rho the PC gate's moving
    # half-batch gradient correlation computed on the mean channel.
    sgd_mean_gate_floor: float | None = None
    # likelihood_min_shell K > 0 (subtomogram particles): no likelihood weight on image shells below K
    # (tomo.free_low_shells): the images are zeroed there, the model's mean and PCs held at zero on 3D shells below
    # K and those noise shells fixed. For low-frequency content shared by a particle's tilts that is not the
    # particle (per-image offsets and gradients, ice, surrounding density); K = ceil(box / particle diameter in px)
    # keeps every shell finer than the particle.
    likelihood_min_shell: int = 0
    # Keep the orientation prior uniform (the learned direction prior is never used or kept).
    uniform_direction_prior: bool = False
    # Translation search per update: ((first update, range px, step px), ...) replacing shift_range and
    # shift_step from each entry's first update on (empty: those two throughout). offset_sigma_hold_px holds the
    # zero-centred Gaussian offset prior's sigma per axis at this value in pixels (None: the learned sigma).
    shift_schedule: tuple = ()
    offset_sigma_hold_px: float | None = None
    # Filament band (relax/ppca_initial_model/filament_band.py): with band_length_ang set, each particle is searched
    # in its own frame (rlnAngle* composed into the projections) over rotations about +z every band_phi_step_deg and
    # shifts along it every band_axial_step_ang within +-band_length_ang / 2, with uniform priors, and the model is
    # supported on a sphere of attachment_radius_ang at (attachment_center_ang, 0, 0). None: the full grid.
    band_length_ang: float | None = None
    band_axial_step_ang: float = 15.0
    band_phi_step_deg: float = 5.0
    attachment_center_ang: float = 180.0
    attachment_radius_ang: float = 60.0
    # The support leaves out the cylinder of this radius about the axis (FilamentBand.core_exclusion_radius_ang).
    attachment_exclude_radius_ang: float | None = None
    # Prior probability that a particle holds the model, against an explicit empty component
    # (relax/ppca_refinement/occupancy.py); None: every particle holds it. Needs the band's full grid.
    occupancy: float | None = None
    # A RELION-convention map of a fixed core in each particle's frame, removed from every tilt image before the PPCA
    # sees it (relax/ppca_initial_model/fixed_core.py): the model then holds what is left. None: nothing removed.
    fixed_core: str | None = None
    # "particle": each particle's core scale is fitted (least squares over its tilts, white metric) before the core
    # is removed - a core map's absolute scale and the per-filament contrast are unknown; "basis": the scale plus
    # the core's radial-dilation and transverse-shift directions (fixed_core.core_basis_volumes), which absorb small
    # core shape errors that a scale leaves at the wall; "none": scale 1.
    fixed_core_scale: str = "particle"
    # The box the fixed core is projected in (fixed_core.core_prediction): the filament continues past the image box
    # and the CTF acts on all of it, so the core (axially uniform: fixed_core.extend_axial_core) is extended to this
    # box, projected and CTF-weighted there and cropped. None: twice the image box.
    fixed_core_box: int | None = None
    # Each image's mean is free (its DC pixel is dropped from the core fit and the residual): RELION's background
    # normalisation subtracts a ring mean that includes the filament's own projection.
    fixed_core_free_offset: bool = True
    # "z" / "cylinder": after the core, the residual's mean in the particle frame, averaged along (and about) the
    # filament (tomo.estimate_residual_background), joins the core as a second fitted component, so a leftover that
    # every particle shares is not fitted as an attachment in every window. "none": the core alone.
    fixed_core_background: str = "none"
    # With fixed_core_scale "particle": the core's spectral shape is fitted to the data (one gain per image shell,
    # tomo.estimate_core_spectrum), the map filtered by it (fixed_core.filter_core_spectrum) and the scales refitted.
    fixed_core_spectrum: bool = False
    # An .npz of each particle's neighbouring copies of the core (fixed_core.PlacedNeighbours; make it with the
    # project's make_neighbours.py from the particles STAR): their sum joins the core as a second component with
    # its own coefficient per particle. None: the core alone.
    fixed_core_neighbours: str | None = None
    # A RELION-convention map that replaces the random initial mean (times initial_mean_scale, masked like the
    # seed); the loadings stay random. None: the random seed.
    initial_mean: str | None = None
    initial_mean_scale: float = 1.0

    def band(self):
        """The :class:`FilamentBand` search, or None for the full grid."""
        if self.band_length_ang is None:
            return None
        from relax.ppca_initial_model.filament_band import FilamentBand

        return FilamentBand(
            self.band_length_ang,
            self.band_axial_step_ang,
            self.band_phi_step_deg,
            self.attachment_center_ang,
            self.attachment_radius_ang,
            self.attachment_exclude_radius_ang,
        )

    def __post_init__(self):
        if self.band_length_ang is not None:
            self.band()  # validates the band
            if self.oversampling or not self.stream_coarse_recompute or self.shift_schedule or self.tilt_coarse_pass:
                raise ValueError(
                    "The filament band searches its own full grid: oversampling 0, the streamed engine, "
                    "no shift schedule, no tilt coarse pass"
                )
        if self.fixed_core_neighbours is not None and self.fixed_core_scale != "particle":
            raise ValueError("fixed_core_neighbours needs fixed_core_scale 'particle'")
        if not np.isfinite(self.initial_mean_scale) or self.initial_mean_scale == 0:
            raise ValueError("initial_mean_scale must be finite and nonzero")
        if self.fixed_core_spectrum and self.fixed_core_scale != "particle":
            raise ValueError("fixed_core_spectrum needs fixed_core_scale 'particle'")
        if self.fixed_core_background not in ("none", "z", "cylinder"):
            raise ValueError(f"fixed_core_background must be none, z or cylinder, got {self.fixed_core_background}")
        if self.fixed_core_scale not in ("particle", "basis", "none"):
            raise ValueError(f"fixed_core_scale must be particle, basis or none, got {self.fixed_core_scale}")
        if self.occupancy is not None:
            if self.band_length_ang is None:
                raise ValueError("An occupancy prior needs the filament band's full grid")
            if not 0 < self.occupancy <= 1:
                raise ValueError(f"Occupancy prior must be in (0, 1], got {self.occupancy}")
        if isinstance(self.q, bool) or not isinstance(self.q, (int, np.integer)) or self.q <= 0 or self.seed <= 0 or self.iterations <= 0:
            raise ValueError("PPCA requires positive integer q, seed and iteration count")
        if not self.stages or self.stages[0][0] != 1:
            raise ValueError("Stages must begin at iteration 1")
        if any(isinstance(hp, bool) or not isinstance(hp, (int, np.integer)) for _, _, hp in self.stages):
            # A fractional order built the truncated order's rotation grid but sized pass 1 from the fraction.
            raise ValueError(f"Stage HEALPix orders must be integers, got {[hp for _, _, hp in self.stages]}")
        if list(self.stages) != sorted(self.stages) or any(r <= 0 or hp < 0 for _, r, hp in self.stages):
            raise ValueError("Invalid radius/HEALPix schedule")
        if not 0 < self.target_mass <= 1 or self.oversampling < 0:
            raise ValueError("Invalid pose support")
        if self.oversampling > 1 or self.stream_full_fine_rows or self.fine_devices > 1:
            raise ValueError(
                "PPCA supports oversampling 0 and 1; the --stream-full-fine-rows / --fine-devices fine pass is "
                "retired (section 14 of docs/math/vdam_ppca_algorithm.md)"
            )
        if self.tilt_coarse_order is not None and (
            isinstance(self.tilt_coarse_order, bool) or not isinstance(self.tilt_coarse_order, (int, np.integer))
            or self.tilt_coarse_order < 1
        ):
            raise ValueError(f"tilt_coarse_order must be a positive integer, got {self.tilt_coarse_order}")
        if self.max_significant <= 0:
            raise ValueError("max_significant must be positive")
        if min(self.image_batch_size, self.rotation_block_size, self.checkpoint_interval) <= 0:
            raise ValueError("Batch/checkpoint sizes must be positive")
        if self.fine_devices < 1:
            raise ValueError("fine_devices must be positive")
        if self.oversampling and not self.stream_coarse_recompute:
            raise ValueError("Adaptive oversampling runs on the streamed engine (stream_coarse_recompute)")
        if self.q > 2 and not self.stream_coarse_recompute:
            raise ValueError("Higher-rank InitialModel requires the streamed coarse recompute engine")
        if self.stochastic_batch_size is not None and self.stochastic_batch_size <= 0:
            raise ValueError("Stochastic batch size must be positive")
        if self.stochastic_all_iterations and self.stochastic_batch_size is None:
            raise ValueError("Stochastic all iterations requires a fixed stochastic batch size")
        if self.balanced_stochastic_halves and (
            not self.stochastic_all_iterations
            or self.stochastic_batch_size is None
            or self.stochastic_batch_size < 2
            or self.stochastic_batch_size % 2
        ):
            raise ValueError("Balanced stochastic halves require an even fixed batch on all iterations")
        if self.shift_range < 0 or self.shift_step <= 0:
            raise ValueError("Shift range must be nonnegative and step positive")
        if self.optimizer not in ("vdam", "momentum_sgd"):
            raise ValueError("Optimizer must be vdam or momentum_sgd")
        if self.optimizer == "vdam":
            # RELION's tau2-fudge sigmoid has length grad_inbetween_iter // 4; for 4 and 5 iterations that is 0 and
            # the fudge at iteration grad_ini_iter is 0/0 = NaN (kept for RELION parity in relax.vdam.schedules).
            # The VDAM update multiplies the fudge into every gate, so such a run cannot produce a finite model.
            phases = compute_phase_lengths(self.iterations)
            unusable = [
                i for i in range(1, self.iterations + 1) if not np.isfinite(compute_tau2_fudge(i, phases, True, 3))
            ]
            if unusable:
                raise ValueError(
                    f"VDAM's tau2-fudge schedule for {self.iterations} iterations is not finite at iteration(s) "
                    f"{unusable} (RELION's sigmoid length grad_inbetween_iter // 4 is 0); choose another iteration "
                    "count or optimizer momentum_sgd"
                )
        if not np.isfinite(self.sgd_learning_rate) or self.sgd_learning_rate <= 0:
            raise ValueError("Momentum SGD learning rate must be finite and positive")
        if not 0 <= self.sgd_momentum < 1:
            raise ValueError("Momentum SGD momentum must be in [0, 1)")
        if self.optimizer != "momentum_sgd" and self.sgd_momentum != 0.9:
            raise ValueError("The SGD momentum is a momentum SGD option")
        if self.optimizer != "vdam" and not self.vdam_second_moment:
            raise ValueError("Turning off the second-moment scaling is a VDAM option")
        if self.sgd_step_scale not in ("trace", "exact_diagonal"):
            raise ValueError("Momentum SGD step scale must be trace or exact_diagonal")
        if self.optimizer != "momentum_sgd" and (self.sgd_halfset_gate or self.sgd_step_scale != "trace"):
            raise ValueError("The half-set gate and the exact-diagonal step scale are momentum SGD options")
        if self.sgd_step_scale == "exact_diagonal" and not self.stream_coarse_recompute:
            raise ValueError("The exact-diagonal step scale runs on the streamed engine (stream_coarse_recompute)")
        if isinstance(self.noise_anneal_updates, bool) or not isinstance(self.noise_anneal_updates, (int, np.integer)):
            raise ValueError("Noise annealing updates must be an integer")
        if not np.isfinite(self.noise_anneal_factor) or self.noise_anneal_factor < 1 or self.noise_anneal_updates < 0:
            raise ValueError("Noise annealing needs a finite factor of at least 1 and a nonnegative update count")
        if (self.noise_anneal_factor == 1) != (self.noise_anneal_updates == 0):
            raise ValueError("Noise annealing needs both a factor above 1 and a positive update count, or neither")
        if self.gemm_precision not in ("auto", "fp32", "tf32"):
            raise ValueError("GEMM precision must be auto, fp32 or tf32")
        if self.gemm_precision == "tf32" and not self.stream_coarse_recompute:
            raise ValueError("TF32 GEMMs apply to the streamed engines only")
        if self.preread_images not in ("auto", "on", "off"):
            raise ValueError("Image preread must be auto, on or off")
        if not 0 <= self.pass2_mass_floor < 1:
            raise ValueError("The pass-2 mass floor must be in [0, 1)")
        if self.sgd_direction not in ("trace", "coupled"):
            raise ValueError("The SGD direction must be trace or coupled")
        newton = self.newton_sgd()
        if self.sgd_lr_match is not None and not (np.isfinite(self.sgd_lr_match) and self.sgd_lr_match > 0):
            raise ValueError("The matched learning-rate factor must be finite and positive")
        if newton and (self.optimizer != "momentum_sgd" or self.sgd_momentum != 0 or self.sgd_halfset_gate):
            raise ValueError("The coupled direction, PC gate and learning-rate schedule need momentum SGD with momentum 0 and no half-set gate")
        if newton and self.sgd_step_scale != "trace":
            raise ValueError("The coupled direction, PC gate and learning-rate schedule use the trace step scale")
        if self.sgd_pc_gate_floor is not None and not 0 < self.sgd_pc_gate_floor <= 1:
            raise ValueError("The PC gate floor must be in (0, 1]")
        if self.sgd_pc_min_shell < 0 or (self.sgd_eig_floor_rel is not None and not 0 < self.sgd_eig_floor_rel < 1):
            raise ValueError("The PC minimum shell must be >= 0 and the relative eigenvalue floor in (0, 1)")
        if (self.sgd_pc_min_shell or self.sgd_eig_floor_rel is not None) and self.sgd_direction != "coupled":
            raise ValueError("The PC minimum shell and relative eigenvalue floor need the coupled direction")
        if self.sgd_fixed_step is not None and not (
            np.isfinite(self.sgd_fixed_step) and self.sgd_fixed_step > 0 and self.sgd_direction == "coupled"
            and self.sgd_lr_match is None
        ):
            raise ValueError("The fixed step must be finite and positive, with the coupled direction and no matched rate")
        if (self.oversampling_from_update is not None or self.oversampling_rotations_only) and not self.oversampling:
            raise ValueError("--oversampling-from-update and --oversampling-rotations-only need --oversampling 1")
        if self.oversampling_from_update is not None and self.oversampling_from_update < 1:
            raise ValueError("The first oversampled update must be >= 1")
        if self.likelihood_min_shell < 0:
            raise ValueError("The likelihood minimum shell must be >= 0")
        if self.sgd_mean_gate_floor is not None and (
            not 0 < self.sgd_mean_gate_floor <= 1 or self.sgd_pc_gate_floor is None
        ):
            raise ValueError("The mean gate floor must be in (0, 1] and needs --sgd-pc-gate-floor")
        if self.shift_schedule:
            entries = [tuple(e) for e in self.shift_schedule]
            if entries[0][0] != 1 or entries != sorted(entries) or any(len(e) != 3 or e[1] < 0 or e[2] <= 0 for e in entries):
                raise ValueError("The shift schedule is sorted (first update, range >= 0, step > 0) entries from update 1")
        if self.offset_sigma_hold_px is not None and not (np.isfinite(self.offset_sigma_hold_px) and self.offset_sigma_hold_px > 0):
            raise ValueError("The held offset sigma must be finite and positive")

    def newton_sgd(self):
        """Whether the update is the Newton SGD step (sgd_recipe.NewtonSGD) rather than the momentum step."""
        return (
            self.sgd_direction != "trace"
            or self.sgd_pc_gate_floor is not None
            or self.sgd_lr_schedule is not None
            or self.sgd_lr_match is not None
            or self.sgd_fixed_step is not None
        )

    def shifts(self, iteration):
        """``(shift_range, shift_step)`` in pixels for update ``iteration``."""
        if not self.shift_schedule:
            return self.shift_range, self.shift_step
        entry = [e for e in self.shift_schedule if e[0] <= max(iteration, 1)][-1]
        return float(entry[1]), float(entry[2])

    def keeps_halfset_moments(self):
        """Whether the run keeps VDAM's half-set moments: VDAM, and momentum SGD with the half-set gate."""
        return self.optimizer == "vdam" or self.sgd_halfset_gate

    def noise_inflation(self, iteration):
        """The factor on the noise estimate in the E-step of update ``iteration`` (1.0 without annealing)."""
        if self.noise_anneal_updates == 0:
            return 1.0
        return 1.0 + (self.noise_anneal_factor - 1.0) * max(0.0, 1.0 - iteration / self.noise_anneal_updates)

    def step_factor(self, iteration, n_images):
        """``min(1, count / scheduled count)``: the factor on VDAM's step for a batch below its own subset.

        VDAM's step and subset schedules together set how many particles its moving average spans
        (about subset / step). A fixed ``stochastic_batch_size`` below VDAM's subset size shortens that
        window; scaling the step by the batch's share of the scheduled subset restores it
        (docs/math/vdam_ppca_algorithm.md section 17). It is one for VDAM's own schedule.
        """
        phases = compute_phase_lengths(self.iterations)
        first, last = default_subset_sizes_for_3d_initial_model(n_images)
        scheduled = compute_subset_size(
            iteration, phases, first, last, n_images, self.iterations, grad_em_iters=DEFAULT_GRAD_EM_ITERS
        )
        scheduled = n_images if scheduled < 0 or iteration == self.iterations else min(scheduled, n_images)
        return min(1.0, self.schedule(iteration, n_images)[0] / scheduled)

    def stage(self, iteration):
        return next((r, hp) for start, r, hp in reversed(self.stages) if start <= iteration)

    def schedule(self, iteration, n_images):
        phases = compute_phase_lengths(self.iterations)
        first, last = default_subset_sizes_for_3d_initial_model(n_images)
        count = compute_subset_size(
            iteration, phases, first, last, n_images, self.iterations, grad_em_iters=DEFAULT_GRAD_EM_ITERS
        )
        if self.stochastic_batch_size is not None and (iteration != self.iterations or self.stochastic_all_iterations):
            count = self.stochastic_batch_size
        if (iteration == self.iterations and not self.stochastic_all_iterations) or count < 0:
            count = n_images
        return (
            min(count, n_images),
            compute_stepsize(iteration, phases, True, 3),
            compute_tau2_fudge(iteration, phases, True, 3),
        )
