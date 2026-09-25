# Resident pass 2: rows, segments and classes

The device-resident sparse pass 2 ([`resident_pass2.py`](../../relax/sparse_pass2/resident_pass2.py))
is being generalized from one class and one image per posterior to RELION's full hidden space. Three
workstreams share this layout, so each one changes only its own side:

- K>1 Class3D (ressym): the class axis;
- cryo-ET subtomograms, S4.2 (cryoet): several images per particle;
- VDAM K>1 on the resident engine (vdamres): builds on the class axis.

## The posterior side: units, hypothesis rows, cells

| Term | Meaning | RELION |
| --- | --- | --- |
| unit `u` | the posterior segment: one particle (SPA: one image) | `part_id` |
| hypothesis row `h = (u, k, r)` | class `k`, fine rotation `r` of unit `u` | `iclass`, `iorient` |
| cell `(h, t)` | one posterior entry, fine translation `t` of the unit (2D for SPA, 3D for tomography) | one `exp_Mweight` entry |

Rows are unit-major, then class-major, then in the single-class row order (the RELION parent execution
order when the float32 fine posterior needs it). Within a unit this is RELION's hidden-space index
`iorientclass = iclass * nr_dir * nr_psi + iorient` (ml_optimiser.cpp:8406, :8450). One unit's cells are one
contiguous segment (`segment_offsets`), so the segmented log-Z and posterior kernels give RELION's joint
minimum, sum and significance over classes and poses without knowing about classes (ml_optimiser.cpp:8411,
:9225, :9602-9660).

[`ResidentCandidateTables`](../../relax/sparse_pass2/resident_candidates.py) holds the hypothesis rows:

- `row_offsets`: the rows of each unit;
- `row_unit`: the row's unit;
- `row_class`: the row's class, or `None` for K=1;
- `row_fine_rot`: the row's fine rotation;
- `row_parent_local`: an index into the unit's parent bitsets;
- `row_log_prior`: the rotation prior plus `log pdf_class[k]`, as the compact engine folds it (k_class.py:539).

Coarse-translation bitsets are per (unit, class, parent). `merge_class_tables` builds the K-class table from K
single-class tables, one per class's own significant support.

## The scoring side: image rows

A hypothesis row is scored by every image of its unit. Both passes do this: the coarse pass sums over a unit's
images before its significance, so the coarse bitsets per (unit, class, parent) come from image-summed coarse
`diff2`, and the fine pass hands the same sum to the posterior. Both passes score image rows and hand
hypothesis-row `diff2` to the posterior side.

The scoring side (cryoet) holds `unit_image_offsets` (CSR over units) and the per-image data: the projection
matrix `L_i` (the tilt rotation composed with the optics magnification; RELION `getRotationMatrix(part_id,
img_id)`, acc_ml_optimiser_impl.h:1037, :1603-1605), the per-image CTF and dose, the optics group and the noise
row. For SPA every unit has one image and `L_i = I`, so the SPA path is the one-image case, with no separate
branch.

Image rows `j = (image i of u, h)` are enumerated, not stored: the scoring loop runs over a unit's images,
evaluates `L_i R` with the image's phases, and accumulates into `diff2(h, t)` in RELION's image order (the float
sum order matters for parity). For tomography `t` is a 3D translation of the unit (ml_optimiser.cpp:2595, :7031);
each image applies its projected 2D shift `L_i t`, with no rounding of the old offset (ml_optimiser.cpp:7259,
:7289), so the per-image phases are a scoring-side function of `(i, t)`. The sum over images comes before the
per-unit minimum, the weights and the significance (acc_ml_optimiser_impl.h:1589-1662). The interface between
the two sides is `diff2(h, t)` over hypothesis rows, `[C_H, T]`; the posterior and the statistics never see
images.

## Projections, M-step and statistics

- **Projections.** For SPA they are per (class, rotation). A chunk row carries the projection id
  `class * n_fine_rot + rotation` into class-stacked caches and grids (M-step rotations tiled, coarse parents
  offset to `class * n_coarse_rot + parent`), or into streamed slots projected class by class. For tomography
  they are per (image, class, rotation).
- **M-step.** Each scored (image, h, t) backprojects with weight `posterior(h, t)`, with the image's own
  `L_i R`, CTF and dose, into class `row_class`'s accumulator (RELION's `BPref[iclass]`, ml_optimiser.cpp:10826). The M-step visits a chunk's rows
  class-major (`mstep_row_order`), so each class's blocks write one accumulator; a block that straddles two
  classes is run once per class, and its rows of the other class get no weight. The per-image Wavg, noise and
  norm partials run on across classes.
- **Accumulator slots (VDAM).** The M-step accumulator axis is `[n_slots]`, slot
  `a = class + K * slot_offset[unit]`: `slot_offset` is 0 for Class3D and auto-refine and the pseudo-halfset
  `part_id % 2` for VDAM, whose BPref slot is `iclass + (part_id % 2) * K` (acc_ml_optimiser_impl.h:4800-4804).
  Projections and scoring stay per class; only the M-step order key and the number of accumulators change
  (`_chunk_class_layout`, the per-class BPref tuples). vdamres adds the offset.
- **Statistics.**
  - `rotation_posterior_sums` is `[K, n_coarse_rot]`, and class posterior sums are the pruned M-step mass per
    class, `[K]` (`thr_wsum_pdf_class`, ml_optimiser.cpp:10497).
  - Per-class evidence, best score and best cell are reductions over each (unit, class) sub-segment, which is
    contiguous inside the unit's segment: the segmented log-Z kernel and the first maximum in row order.
  - Noise is one total over classes per optics group (ml_optimiser.cpp:10470, :11010). The scoring side owns
    the per-image indexing of the noise and power sums; the posterior side supplies only `posterior(h, t)`. The
    per-particle divisions of tomography are cryoet's (acc_ml_optimiser_impl.h:4139-4145).

## Implementation

[`compute_k_class_pass2_stats_resident`](../../relax/sparse_pass2/resident_pass2.py) runs the class axis;
`compute_pass2_stats_resident` is its one-class case and keeps the compact engine's signature. The K-class
output has the field names of the exact-local engine's class-segmented output, so
`k_class._class_segmented_em_result` builds the K-class result from either. Under `RELAX_SPARSE_PASS2_RESIDENT`
the fused K-class pass runs on it with the K=1 production arithmetic, since RELION's does not depend on the class
count. The first iteration's `--firstiter_cc` winner, the zero-oversampling reuse and several optics groups are
still K=1 only.
