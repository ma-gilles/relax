# Coarse geometry at zero oversampling

RELION's accelerated pass1 builds scorer matrices with device
`make_eulers_3D`, including when adaptive oversampling is zero. Fine scoring
and weighted reconstruction use the distinct host Euler matrix path.
They represent the same intended rotations but differ in float32 arithmetic;
texture interpolation can amplify these small matrix differences.

`refine_single_volume` in
`recovar/em/refinement/iteration_loop.py` uses the existing
`sampling._relion_adaptive_pass1_rotations` with canonical source Euler rows
and the separate perturbation matrix. The OS0 result is transported as
`coarse_scoring_rotations`, not substituted for `effective_rotations`.
`half_scoring._score_half_dense` applies it only to the coarse operand of
the soft-Gaussian F32 K1 sparse x-half engine. Fine grids, pose metadata and
M-step rotations retain their original values. CPU fallback remains unchanged.
Nonzero oversampling, local search, firstiter_cc/hard first iteration,
K4, and diagnostic-double behavior are not expanded by this repair.

The stack6 matrix-only discriminator on frozen ac9 and the actual-source
replay on peer commit 84fb5867 agree: centered raw-score maximum absolute
gap versus native falls from .01166 to .001230, and support flips fall
from five to three. The count remains 52884 versus native 52883. The remaining scoring/prior boundary is separate; this change
does not qualify strict parity, full trajectories, FSC or speed.

`tests/unit/test_zero_coarse_geometry.py` executes the production eligibility,
transport and engine-operand expressions with controlled sentinels. It does
not replace an actual-source GPU replay. Native capture and diagnostic receipts
are linked from the coordination status in `docs/development/em_status.md`.

## Images on another grid or magnified

For an optics group whose images need a projection matrix other than the rotation, RELION's left matrix is
`MBL = applyScaleDifference(applyAnisoMag(I)) = s inv(M3)` (acc_ml_optimiser_impl.h:1602-1617, obs_model.cpp:1309-1340),
and each matrix path applies it in its own precision:

- pass 1 (`AccProjectorPlan::setup`, `make_eulers_3D`): `MBL (A R)` in float32 from float Euler angles, inverted
  with the float32 adjugate, since with a left matrix the inverse is not the transpose
  ([`relion_device_projection_rotations`](../../relax/sampling/__init__.py)); the tilt images' `MBL` use the same kernel;
- fine pass, weighted sums and M-step (`generateEulerMatrices`): `MBL A R` and its inverse in double, cast to
  float once ([`projection_rotations`](../../relax/relion/optics_aberrations.py)).

relax builds every projected set of rows from its own source by the rule of the path that built it
([`project_pass2_rotations`](../../relax/sampling/oversampling.py), [`project_rows`](../../relax/sampling/__init__.py)):
the device rows from their source Euler rows, the host rows from their float64 matrices, which must reproduce the
rows handed in (`RotationProvenanceError` otherwise, one check per grid build). Composing after the float32 cast
moved 80% of the order-2 magnified matrices by an ulp and flipped two near-tie iteration-1 poses in 10,000 on the
magnification fixture (relax#24); with the two rules relax's iteration 1 equals RELION's to float noise
(`tests/unit/test_projection_rotation_rules.py`).
