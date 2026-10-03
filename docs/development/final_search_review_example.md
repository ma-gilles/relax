# Final search: worked review example

Superseded for geometry design by the
[grid ownership proposal](grid_ownership_proposal.md). Retained as the reviewed
example that exposed duplicated sizes and mixed responsibilities. Do not
implement its `SearchImageSizing` design.

User review correction: the proposed pixel-size fallback below is rejected.
The implementation should validate physical geometry once at the input boundary
and use its positive pixel size directly. The original fallback remains in the
before excerpt only as evidence of behavior being retired for invalid inputs.
The `cryo` alias also needs replacement by an explicit geometry dependency.

Proposed code only; no production edits or numerical validation. Source:
`f17505a`. This is a complete small example from the global part of the annotated
final-search section. It exposes input construction and the consumer as well as
the helper, so we can judge the cost of abstraction. The
[larger proposal](final_search_refactor_proposal.md) covers local search.

## Before

The two pass sizes are loose variables initialized before the local/global
branch. Inside the global branch, the current sizing calculation is:

```python
final_adaptive_pass1_current_size = None
final_adaptive_pass2_current_size = None
if k_class_enabled and int(state.adaptive_oversampling) > 0:
    final_coarse_size = compute_coarse_image_size(
        healpix_angular_step(final_current_healpix_order),
        cryo.voxel_size if cryo.voxel_size > 0 else 1.0,
        grid_size,
        particle_diameter=particle_diameter_ang,
    )
    final_adaptive_pass1_current_size = clamp_relion_coarse_image_size(
        final_coarse_size,
        final_current_size,
        grid_size,
    )
    final_adaptive_pass2_current_size = final_current_size
```

Later these values become `firstiter_coarse_current_size` and
`firstiter_fine_current_size` in the existing `DenseVariantPolicy` construction.
Logging is omitted from this excerpt; retain its existing relative position.

## Proposed definitions and complete calculation

This alternative uses a named result for pass sizes. The local/global results
remain distinct; this small global result deliberately does not introduce two
more classes for adaptive versus ordinary global scoring.

```python
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SearchImageSizing:
    """Image widths in pixels; physical lengths in angstroms."""

    original_size: int
    current_size: int
    pixel_size_angstrom: float
    particle_diameter_angstrom: float | None


@dataclass(frozen=True, kw_only=True)
class GlobalSearchSizes:
    """Both fields absent for ordinary scoring, present for adaptive scoring."""

    coarse_size: int | None
    fine_size: int | None


def prepare_final_global_search(
    *,
    healpix_order: int,
    oversampling_order: int,
    k_class_enabled: bool,
    image_sizing: SearchImageSizing,
) -> GlobalSearchSizes:
    if not (k_class_enabled and int(oversampling_order) > 0):
        return GlobalSearchSizes(coarse_size=None, fine_size=None)

    coarse_size = compute_coarse_image_size(
        healpix_angular_step(healpix_order),
        image_sizing.pixel_size_angstrom,
        image_sizing.original_size,
        particle_diameter=image_sizing.particle_diameter_angstrom,
    )
    coarse_size = clamp_relion_coarse_image_size(
        coarse_size,
        image_sizing.current_size,
        image_sizing.original_size,
    )
    return GlobalSearchSizes(
        coarse_size=coarse_size,
        fine_size=image_sizing.current_size,
    )
```

The three numerical helper dependencies retain their existing implementations.
No new validation or cast is introduced in this review example. If this type
is accepted, enforce the paired-present/paired-absent result invariant at its
construction boundary. Do not invent new restrictions on existing geometry.

## Proposed caller, including input construction

```python
image_sizing = SearchImageSizing(
    original_size=grid_size,
    current_size=final_current_size,
    pixel_size_angstrom=geometry.pixel_size_angstrom,
    particle_diameter_angstrom=particle_diameter_ang,
)
search_sizes = prepare_final_global_search(
    healpix_order=final_current_healpix_order,
    oversampling_order=state.adaptive_oversampling,
    k_class_enabled=k_class_enabled,
    image_sizing=image_sizing,
)
```

Here `geometry` means validated input geometry; its construction is part of the
next revision, not an existing object in the current implementation.
Keep this within the existing global branch initially. Share geometry
construction with the local branch only after checking supported inputs and
access timing. The existing global log precedes sizing, and the adaptive-size
log follows it when sizes are present. No hidden recorder is needed to review
the computational interface.

## Proposed consumer

At the existing `DenseVariantPolicy(...)` call, replace precisely these two
keyword arguments; the rest of the policy and scoring call remain in place:

```python
firstiter_coarse_current_size=search_sizes.coarse_size,
firstiter_fine_current_size=search_sizes.fine_size,
```

The same immutable result is used for both halves. There is no array conversion,
copy, state mutation, or additional numerical calculation in the containers.

## What to judge

This is more total code than the original small branch. It is worthwhile only
if image sizing is a coherent concept shared with local preparation and the
named result clarifies the main flow and its consumers. For the global branch
alone, a helper returning named sizes could be enough. The larger local branch
has more related inputs and substantially more to gain from grouping.

Review the caller, definitions, and consumer together: can you follow where
each value comes from, what calculation owns it, and where it goes? We should
not count a shorter caller as success if the additional types make that harder.

This is a review alternative, not an amendment to the agreed preference for
distinct modes with genuinely different required data. It makes the tradeoff
around two optional sizes explicit rather than silently treating every mode as
requiring another class. The implementation and final naming await review.
