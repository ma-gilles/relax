# Grid ownership and working resolution

Proposed September 30, 2026 after user review; not yet accepted or implemented.
Inspected source: `f17505a220769e488f4cf3eceb33ef8a7831ea27`. No numerical
imports or tests. A refresh of main was attempted but `.git/FETCH_HEAD` is
read-only under the current sandbox. This proposal supersedes the geometry
assumptions in the earlier final-search examples, not the agreed principles.

## Findings from the current implementation

1. `refine_single_volume` takes `volume_shape` and `grid_size` from its first
   dataset at entry (`iteration_loop.py:718–720`). Those local values are not
   reassigned during the loop. The volume values change; the full model grid
   represented by these values does not. Final scoring uses `grid_size`
   (`:5047`). This is a property of this implementation, not a universal rule
   that model grids can never change.
2. `current_size` is scheduled, quantized, and sometimes replayed between
   iterations (`:1403–1663`). It describes working frequency support in model
   coordinates; it is not automatically the original image shape or the output
   volume shape.
3. Particle windows and model support are explicitly separated at
   `:2137–2170`. Model support is remapped to particle windows using each optics
   grid. The existing `None` convention is overloaded: full model support can
   accidentally become the particle cutoff unless stated explicitly.
4. `helpers/fourier_window.py` implements support restrictions using original
   frequency coordinates and gathered/scattered indices. A smaller working
   array does not imply a newly sampled physical image grid.
5. `helpers/half_volume_mstep.py:118` derives a padded, odd backprojector shape
   from model shape, support, and padding. This temporary accumulator shape is
   neither the input image shape nor the output volume shape.
6. `MultiShapeHalf` presents reference geometry through attributes named
   `image_shape`, `volume_shape`, and `voxel_size`, while its constituent image
   datasets have their own geometry (`refinement/optics_shapes.py:51`).
   `_ReferenceGridView` at `:297` overrides a dataset's volume shape to make it
   look like the model grid to engines. This is evidence of mixed ownership.
7. The K1 loader reads model pixel size from the reference MRC and image pixel
   sizes from particle metadata (`full_refinement.py:3763–3784`). The sources
   must remain distinct: `helpers/resolution.py:84` documents how rounding can
   produce a 58-pixel particle window for 56-pixel model support even when the
   nominal pixel sizes look equal.

See [iteration loop](../../relax/refinement/iteration_loop.py),
[Fourier windows](../../relax/helpers/fourier_window.py),
[accumulator geometry](../../relax/helpers/half_volume_mstep.py),
[optics grouping](../../relax/refinement/optics_shapes.py),
[resolution helpers](../../relax/helpers/resolution.py), and
[input loading](../../relax/refinement/full_refinement.py).

## Proposed ownership

Separate three questions: what physical samples mean, which frequencies this
pass uses, and how the implementation stores the arrays.

| Owner | Authoritative information | Lifetime |
| --- | --- | --- |
| Input image group | Logical real-space image shape and physical pixel size from its source metadata | Fixed for that loaded input representation |
| Current model | Logical real-space volume grid and physical voxel size, alongside model estimates | Stable in today's loop; replaced explicitly if genuine model resampling is introduced |
| Pass specification | Model support width and each image group's working Fourier window | Recomputed when the pass changes |
| Engine workspace | Packed axes, padded projector/accumulator shapes, gathered indices | Derived for the selected pass/backend; released according to current memory policy |

Keep angular/translation search grids separate from physical image/volume grids.
Particle diameter is a physical model/masking parameter, not grid geometry.
Symmetry is a model/search constraint, not a pixel-grid property.

Do not store independent copies of `original_size`, `grid_size`, and `shape[0]`
when all mean the same grid width. A derived property may give that width a
readable name. Conversely, do not deduplicate image and model pixel sizes just
because their values currently agree. They describe different data and have
different provenance.

## Minimal proposed records

These are interface sketches, not production classes. Reuse existing owning
containers where possible; no new hierarchy, geometry manager, or abstract base
class is proposed. Begin with isotropic grids supported by the affected path;
do not imply support for arbitrary anisotropic geometry.

```python
@dataclass(frozen=True, kw_only=True)
class ImageGrid:
    shape: tuple[int, int]       # Logical real-space dimensions, not rFFT storage
    pixel_size_angstrom: float  # Validated finite and positive


@dataclass(frozen=True, kw_only=True)
class VolumeGrid:
    shape: tuple[int, int, int]  # Logical real-space dimensions
    voxel_size_angstrom: float  # Preserve the model source's precision


@dataclass(frozen=True, kw_only=True)
class PassBand:
    model_support_size: int
    image_window_sizes: tuple[int, ...]  # One per established image-group order
```

`model.grid` would hold a `VolumeGrid`; each existing image group would expose
an `ImageGrid`. These are composition, not wrappers forwarding every dataset
attribute. A pass stores both resolved sizes because the model cutoff and
image windows have different meanings, not because they are unrelated knobs.
Construct the pass only through its planning function; validate group alignment
and bounds once there. Carry existing group identities without resorting rows.

Use explicit full widths in the new pass representation. Translate to existing
engine `None` conventions at a single controlled boundary until consumers are
migrated. A full physical grid does not need a fabricated default pixel size.

Do not replace integer working sizes with a floating-point angstrom cutoff
during this structural refactor: converting back can change `ceil`, clamping,
and selected shells. Keep existing formulas and evaluation order. Physical
resolution can be reported as a derived quantity.

## Concrete example

Suppose an image group is 256 pixels wide at 1.5 angstroms/pixel, and the current
model grid is 256 cubed at 1.5 angstroms/voxel. A pass with model support size
64 can use a 64-wide particle Fourier window. The full model remains 256 cubed;
the output's voxel size remains 1.5 angstroms. Do not mutate its grid to 64 cubed
at 1.5 angstroms merely because the working calculation is smaller.

A second image group of 128 pixels at 3 angstroms/pixel has the same physical
field of view. The existing remapping formula can also select a 64-wide Fourier
window, despite different full image shapes and pixel sizes. Preserve the
actual per-group rounding and clamping even in nominally equivalent cases.

If an operation actually converts the full 256-cubed model to a 128-cubed
real-space representation while preserving field of view, it needs a new model
grid with voxel size 3 angstroms. That would be an explicit resampling operation
with numerical qualification, not a hidden consequence of changing pass size.
No new resampling feature is part of this proposal.

## Proposed calling flow and where inputs originate

At loading, the reference map provides model grid metadata, while the existing
particle loader provides each image group's metadata. Validate both sources;
remove silent nonpositive/nonfinite pixel-size substitutions. Do not silently
switch a formula from image pixel size to model voxel size during migration.
Some current paths use reference optics metadata rather than an MRC-derived
model spacing; preserve and explicitly trace each path's current source.

```python
# At the input boundary; names denote existing loaded source values.
model_grid = VolumeGrid(
    shape=reference_volume_shape,
    voxel_size_angstrom=reference_voxel_size,
)
image_grids = tuple(
    ImageGrid(shape=group.image_shape, pixel_size_angstrom=group.voxel_size)
    for group in input_image_groups
)

# Existing scheduling computes the model support width; no formula change.
pass_band = plan_pass_band(
    model_grid=model_grid,
    image_grids=image_grids,
    model_support_size=scheduled_model_support_size,
)

# Preparation receives physical grids and working support separately.
search = prepare_final_search(
    sampling_state,
    model_grid=model_grid,
    image_grids=image_grids,
    pass_band=pass_band,
    particle_diameter_angstrom=particle_diameter_angstrom,
)
```

This is a proposed boundary, not a replacement for all arguments of the local
search function. Existing source-specific replay, canonical Euler, precision,
and materialization inputs still require the established branch inventory.
`plan_pass_band` resolves the existing model-to-optics size rules; it does not
invent a second copy of those formulas. Coarse and fine passes may have distinct
bands. Engine layout builders consume the grids and band and retain the existing
padding, shape bucketing, packed-axis and allocation behavior.

## What should happen first

Withdraw `SearchImageSizing`/`FinalSearchImageSizing` as the preferred design:
they bundle immutable provenance, changing support, and particle diameter into
one temporary bag. Do not first introduce those types and later undo them.

Before source extraction, make a field mapping for the bounded final-search
path: every original image/model size, pixel size, Fourier cutoff, and workspace
shape must have a documented source and consumer. It must cover ordinary SPA,
per-optics geometry, replay, and the tomography adapter. Then replace the
`cryo` alias's geometry access with the explicit correct owner while retaining
particle loading/batching through current RECOVAR interfaces. Migrating engine
dataset dependencies beyond that is a separate cohesive package.

Validate valid-input equivalence with the existing scientific gates and
targeted checks for unequal model/image spacing, exact rounding boundaries,
multiple image shapes, deferred local grids, and final full-support behavior.
Test rejected invalid metadata separately. Do not widen tolerances or infer
success from matching shapes alone. Preserve current lifetimes: grid metadata
must not retain temporary arrays or accumulate workspaces across iterations.

The decision for user review is the ownership split, not these provisional
class names. Once accepted, show the complete revised local/global preparation
with the field mapping, before implementation.
