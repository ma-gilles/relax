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
| cell `(h, t)` | one posterior entry, fine translation `t` | one `exp_Mweight` entry |

Rows are unit-major, then class-major, then in the single-class row order (the RELION parent execution
order when the float32 fine posterior needs it). Within a unit this is RELION's hidden-space index
`iorientclass = iclass * nr_dir * nr_psi + iorient` (ml_optimiser.cpp:8406, :8450). One unit's cells are one
contiguous segment (`segment_offsets`), so the segmented log-Z and posterior kernels give RELION's joint
minimum, sum and significance over classes and poses without knowing about classes (ml_optimiser.cpp:8411,
:9225, :9602-9660).

[`ResidentCandidateTables`](../../relax/sparse_pass2/resident_candidates.py) holds the hypothesis rows:

- `row_offsets`: the rows of each unit;
- `row_image`: the row's unit (named for the SPA case);
- `row_class`: the row's class, or `None` for K=1;
- `row_fine_rot`: the row's fine rotation;
- `row_parent_local`: an index into the unit's parent bitsets;
- `row_log_prior`: the rotation prior plus `log pdf_class[k]`, as the compact engine folds it (k_class.py:539).

Coarse-translation bitsets are per (unit, class, parent). `merge_class_tables` builds the K-class table from K
single-class tables, one per class's own significant support.

## The scoring side: image rows

A hypothesis row is scored by every image of its unit. For SPA the image rows are the hypothesis rows (one
image per unit), so nothing changes there.

For tomography (S4.2) the scoring stage evaluates image rows `j = (image i of u, h)` with per-image matrices
`L_i R` and phases, then segment-sums `diff2` over the unit's images into `diff2(h, t)`. That sum comes before
the per-unit minimum, the weights and the significance (acc_ml_optimiser_impl.h:1589-1662). The interface
between the two sides is `diff2(h, t)` over hypothesis rows, `[C_H, T]`; the posterior and the statistics never
see images.

## Projections, M-step and statistics

- **Projections.** For SPA they are per (class, rotation). A chunk row carries the projection id
  `class * n_fine_rot + rotation` into class-stacked caches and grids (M-step rotations tiled, coarse parents
  offset to `class * n_coarse_rot + parent`), or into streamed slots projected class by class. For tomography
  they are per (image, class, rotation).
- **M-step.** Each scored (image, h, t) backprojects into class `row_class`'s accumulator with weight
  `posterior(h, t)` (RELION's `BPref[iclass]`, ml_optimiser.cpp:10826). The M-step visits a chunk's rows
  class-major (`mstep_row_order`), so each class's blocks write one accumulator; a block that straddles two
  classes is run once per class, and its rows of the other class get no weight. The per-image Wavg, noise and
  norm partials run on across classes.
- **Statistics.**
  - `rotation_posterior_sums` is `[K, n_coarse_rot]`, and class posterior sums are the pruned M-step mass per
    class, `[K]` (`thr_wsum_pdf_class`, ml_optimiser.cpp:10497).
  - Per-class evidence, best score and best cell are reductions over each (unit, class) sub-segment, which is
    contiguous inside the unit's segment: the segmented log-Z kernel and the first maximum in row order.
  - Noise is one total over classes per optics group (ml_optimiser.cpp:10470, :11010). The per-particle
    divisions of tomography are cryoet's (acc_ml_optimiser_impl.h:4139-4145).

## Implementation

[`compute_k_class_pass2_stats_resident`](../../relax/sparse_pass2/resident_pass2.py) runs the class axis;
`compute_pass2_stats_resident` is its one-class case and keeps the compact engine's signature. The K-class
output has the field names of the exact-local engine's class-segmented output, so
`k_class._class_segmented_em_result` builds the K-class result from either. Under `RELAX_SPARSE_PASS2_RESIDENT`
the fused K-class pass runs on it with the K=1 production arithmetic, since RELION's does not depend on the class
count. The first iteration's `--firstiter_cc` winner, the zero-oversampling reuse and several optics groups are
still K=1 only.
