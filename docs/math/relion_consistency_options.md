# RELION-consistency options

relax reproduces RELION 5.0.1, including the places where RELION's own arithmetic is not
self-consistent. Each such place has an opt-in option that applies the consistent rule, so its
effect on a refinement can be measured. Every default is RELION's rule; with every option at its
default relax runs the same operations as before the options existed.

The options are the fields of `RelionConsistencyOptions`
(`relax/refinement/refinement_options.py`) and flags of `relax refine` / `relax class3d`.
`require_consistency_route` and `resolve_consistency_options`
(`relax/refinement/command_options.py`) refuse a non-default value wherever it would not be
honoured:

- subtomogram particles and optics groups on several image shapes (their scorers are not reached);
- a run seeded from, replaying or frozen at RELION's own state (`--relion_init_dir`,
  `--perturb_replay_relion_dir`, `--frozen-boundary-dir`, `--init_relion_iteration`, replay
  overrides, captured projectors, state swaps): those statistics, references and projectors were
  computed with RELION's rules;
- a continuation (`--continue`) whose run files were written with other option values: the run
  files record the non-default options (`relax_consistency_<name>` in the optimiser STAR).

`relax initial_model` (VDAM) and the PPCA commands do not take the options.

## `--gridding_kernel {radial,separable}`

Trilinear interpolation in the padded 3-D transform multiplies the real-space map by the
transform of the interpolation kernel, the per-axis product
`prod_i sinc^2(x_i / (pf N))`. RELION corrects with the radial window
`sinc^2(|x| / (pf N))` instead (`Projector::griddingCorrect`, `projector.cpp:595-628`, and the same
window in `BackProjector::reconstruct`). The two agree on the axes; off the axes the radial window
is smaller, so RELION over-corrects towards the box corners (3.1% at the corner of an 8-voxel box
with padding 2).

`separable` divides by the per-axis product at every K=1 site:

| Site | Function |
| --- | --- |
| scoring projector, every numbered iteration and the final pass | `prepare_scoring_projector` -> `setup_relion_projector_on_host` |
| expected-accuracy projector | `Half1AccuracyInputs.estimate` -> `_projector_data` |
| regularized and unregularized half maps, final unfiltered, merged and half maps | `_reconstruct_volume_eager` (device route and both large host-staged finishes) |

A separable projector-cache entry has its own key. Class3D refuses the option: its tau2 is the
power spectrum of the radially corrected reference.
