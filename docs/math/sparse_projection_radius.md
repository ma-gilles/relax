# Sparse projection radius propagation

The generic sparse projection path must forward the requested `max_r` to the
same projector used by coarse scoring. The radius is not merely a final pixel
selection: the CUDA texture implementation also derives its compact texture
extent and origin from the radius. Gathering the same image pixels afterward
does not make a projection computed with a different radius equivalent.

Implementation: [`_compute_sparse_pass2_projections_block`](../../relax/sparse_pass2/sparse_pass2_projection_blocks.py).
The helper also reads `max_r` to infer a supplied RELION projector's output crop,
but must not consume it before the generic call. An explicit `None` and an
omitted argument retain their distinct underlying-projector meanings. Explicit
RELION output size remains independent of the model radius.

For fixed image operand \(I\), weights \(w\), and projections \(P_f,P_c\), the
projection-only change in Gaussian log score is

\[
\Delta s=\Re\sum_p w_p\overline{I_p}(P_{f,p}-P_{c,p})
-\tfrac12\sum_p w_p(|P_{f,p}|^2-|P_{c,p}|^2).
\]

This identity permits an offline diagnostic without changing production precision.
The September 13 four-particle cold-start discriminator found identical padded
volume and rotation inputs but differing coarse/pass-2 projections. For particle
1433, projection differences predicted score RMS0.101721 versus observed0.101761;
the remaining RMS was6.97e-5. A matched missing-radius intervention reduced its
Pmax from0.103265554 to0.095389739 (native STAR0.095284). This is a scoped diagnostic,
not strict-state, full-trajectory or speed acceptance. Existing mixed-precision
stages were not changed or qualified as all-float32.

Exact forwarding regressions, including whole/chunked/windowed generic paths and
supplied-RELION crop behavior, are in
[`test_sparse_projection_radius.py`](../../tests/unit/test_sparse_projection_radius.py).
Current qualification and pinned artifacts are linked from
[`em_status.md`](../development/em_status.md).

For the strict RELION texture path, the native kernel owns image clipping.
It clamps the model radius to the active image radius, rotates in float32,
and truncates squared radius to an integer before comparison. An exact disk
in source-pixel coordinates can erase valid rounded-shell samples and is not
applied by default. The embedding helpers preserve the kernel's values.
Capacity-backed projections forward the active image radius independently of
the allocated image extent. Explicit exact-disk masking remains a diagnostic
option. The compact coarse certificate and rescorer share this canonical
radius rule by default when the texture route is available.

Implementation: [`_project_relion_projector_texture`](../../relax/helpers/projection.py).
Boundary regressions live in `tests/unit/test_coarse_rotated_radius.py` and
`tests/unit/test_cuda_relion_fine_diff2.py`; their focused results do not replace
matched trajectory qualification.

## Image windows wider than the model sphere

An optics group on a coarser grid than the reference (scale \(1<s<\sqrt2\)) gets
an image window of about \(s\) times the reference window, wider than
\(2\,r_{\max}\). RELION's kernels clamp \(\mathrm{maxR}=\min(r_{\max},\,\mathrm{imgX}-1)\)
and relabel the rows beyond it before projecting them. The coarse diff2 kernel
labels them negative. The fine diff2 and weighted-sum kernels move them to
\(x=\mathrm{maxR}\), except for the negative rows from \(\mathrm{imgY}-\mathrm{maxR}\).
Each relabelled pixel then lies outside the rotated sphere, so RELION's reference there
is zero. This holds even where the pixel's own rotated radius is inside the sphere,
at \(|k|/s<r_{\max}\). Relax zeros those rows after projection. The coarse kernel zeros
rows \(\mathrm{maxR}<\text{label}\le\mathrm{imgY}/2\); the fine kernels zero rows
\(|\text{label}|>\mathrm{maxR}\). The fused coarse CUDA scorer wraps its rows at maxR
as RELION does, for both the projection and the image shift. The coarse zeros are
RELION's only when the window is at least about \(2s\,\mathrm{maxR}\). A coarse
window strictly between \(2\,\mathrm{maxR}\) and that bound makes RELION score the
wrapped rows with nonzero references
([`coarse_rows_wrap_inside`](../../relax/helpers/optics_scale.py)). The non-fused
coarse projection and the local parent pass refuse that band; only the fused scorer
reproduces it. Before this rule, the relabelled rows held nonzero reference values
and the other-grid group's Pmax moved by about 0.01 per particle. Groups at
\(s\ge\sqrt2\), where the moved fine pixel can fall inside the sphere, are refused.
The local search passes `projection_relion_kernel="coarse"` for its parent pass (RELION's
pass 1). Projections that name no kernel get the fine rule. For unscaled rotations that rule
changes nothing, because those rows already lie outside the sphere.

Implementation: [`relion_kernel_zero_rows`](../../relax/helpers/projection.py), passed
through `compute_relion_projector_projections_block(relion_kernel=...)`, and the fused
coarse scorer body `relax/cuda/relion_coarse_diff2_projector_body.inc`. The regression is
`tests/unit/test_relion_kernel_rows.py`, which uses an independent scalar model of the
RELION kernels.

## Backprojection support under anisotropic magnification

RELION backprojects an image pixel \(k=(x,y)\) of matrix \(A\) when two conditions hold. Its
weight is nonzero, which bounds the image radius by the rounded support
\(\mathrm{ROUND}(|k|)\le \mathrm{cs}/2\): `Minvsigma2` is zero beyond it
(ml_optimiser.cpp:6894-6901), and the weight carries it. Its rotated radius is inside the model
sphere, \(|A^{-1}(x,y,0)^T|\le r_{\max}\) (acc/cuda/cuda_kernels/BP.cuh:322,
`BackProjector::backproject2Dto3D`). recovar's adjoint kernel clips the image radius
\(|k|\le \text{max\_r}\) and repeats the cut on the rotated radius.

The image clip alone is RELION's rule when \(A^{-1}\) restricted to the image plane is a
multiple of an isometry. That covers one grid, optics groups on another pixel size or box
(\(A=R/s\), image radius \(r_{\max}s\), `ReferenceSphereClip` with no reference radius) and
unmagnified tilt images (\(A=A_{\mathrm{proj}}R\)). With an anisotropic magnification
\(A=M_3^{-1}A_{\mathrm{proj}}R\) (`ObservationModel::applyAnisoMag`), the rotated radius is
\(|Mk|\). It ranges over \([\sigma_{\min}(M),\sigma_{\max}(M)]\,|k|\), so no image radius
reproduces the rule. In a band around \(r_{\max}\), pixels outside the image radius are
inside the sphere and pixels inside it are outside. The band holds the outer shell of every
M-step; its FSC moved by 0.7% at iteration 1 of the subtomogram fixture `eto_mag`.

When any optics group's `rlnMagMat` has unequal singular values
(`magnification_is_anisotropic`, decided once from the optics table), the adjoint keeps the
rounded image support (\(\text{max\_r}=r_{\max}+1/2\)). It masks each row's pixels by
\(|A^{-1}k|\le r_{\max}\) before the kernel, and the recon window keeps the rounded support
(`recon_exact_radius=False`). The forward projector already clips on the rotated radius with
RELION's integer-truncated \(r^2\) (relax/cuda/relion_scoring.cuh:464).

Optics groups with different `rlnMagMat` (relax#48) are split into shape classes like groups on
other grids ([`optics_shape_class_rows`](../../relax/refinement/optics_shapes.py)), so each class
applies one matrix and the mask above reads that class's \(A=M_3^{-1}A_{\mathrm{proj}}R/s\). A
class on another grid with anisotropic magnification keeps \(\text{max\_r}=r_{\max}+1/2\) when
\(s\le 1\): its recon window holds its rounded support, which lies inside that radius. A class with
\(s>1\) and anisotropic magnification is refused.

Implementation: [`ReferenceSphereClip`, `mstep_adjoint_max_r` and `rotated_radius_mask`](../../relax/helpers/adjoint.py),
selected by `dataset_magnification_is_anisotropic` in the resident and local engines. Regression:
`tests/unit/test_mstep_rotated_radius_clip.py`. Its CPU half checks relax's support against the rule
above for a symmetric and an asymmetric `rlnMagMat`. Its GPU half checks relax's adjoint against
RELION's BackProjector through the binding.
