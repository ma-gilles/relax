# Multi-body refinement

`relax multibody` ([`relax/multibody`](../../relax/multibody)) follows `relion_refine --continue <consensus> --multibody_masks
bodies.star` at RELION f2c1a38. This note lists what RELION does and where relax does the same.

## Start

The run continues a converged split-half auto-refine (`MlOptimiser::initialise` with `do_initialise_bodies`,
ml_optimiser.cpp:2122-2190). Each body gets the consensus half maps times its mask, centred on the mask's centre of mass
rounded to integers (`com_bodies`), and starts at residual angles `(0, 90, 0)` with zero residual offset. The body's
orientation `O = alignWithZ(com_other - com_body)` points to the body it rotates relative to
([`bodies.py`](../../relax/multibody/bodies.py)).

- `--offset_range` and `--offset_step` are multiplied by the pixel size twice on continuation (`parseContinue`
  and `initialiseGeneral`): `3` px at 4.25 Å gives a 54.19 Å range.
- The sampling perturbation of iteration 0 is the consensus value plus one draw of seed 1; iteration `n` then
  advances it with seed `random_seed + n`.
- Both halves start from half 1's noise spectrum. This reproduces RELION issue #7
  ([relax #7](https://github.com/ma-gilles/relax/issues/7)): the first follower broadcasts its noise to every rank,
  also on a continuation (ml_optimiser_mpi.cpp:748-758). It differs from relax's own K=1 `--continue`, which keeps
  each half's noise ([§8 of the refinement notes](relion_refinement_algorithm.md)); a multi-body run always starts
  from a consensus, so it follows RELION.
- The noise update divides each shell by its full-image pixel count (`Npix_per_shell`), so the shell at the current
  size's Nyquist, partly summed, dips, as in RELION and relax's K=1 default.
- `sigma2_rot` and `sigma2_psi` stay the consensus values unless `--sigma_ang` is given. After a fine consensus
  run they are narrower than the order-4 grid, so each particle keeps one parent direction: the first at the
  smallest angle in grid order (`best_idir`).

## Overlapping masks

Masks may overlap (`MlModel::initialiseBodies`, ml_model.cpp:1363-1456). By default, when body `o` is subtracted
from body `i`'s images it is projected with `m_o - m_o (m_i + m_p)`, where `m_p` sums the masks already chosen for
the bodies before `o`, an extra projector used only for that subtraction; the body's own reference, FSC mask,
centre and radius keep its mask ([`bodies.subtraction_masks`](../../relax/multibody/bodies.py)).
`--multibody_norm_overlap` instead divides every mask by the masks' sum where it exceeds one, after the centres and
radii are taken, and uses those masks everywhere.

## Iteration

For each body `b` in turn ([`refine.score_bodies`](../../relax/multibody/refine.py)), each half:

1. **Subtraction** ([`subtraction.py`](../../relax/multibody/subtraction.py)). The image is shifted by
   `round(consensus offset - A_consensus com_b)` (the fraction is dropped, as RELION drops it). The other bodies
   are projected at their composed poses `A Oᵀ A_rot90 R O`, with `A_rot90 = rotation3DMatrix(90, 'Y')`, multiplied
   by RELION's own CTF `getCTF` (the sign is the opposite of relax's exact CTF rows), shifted to their projected
   centres plus their refined offsets, divided by the norm correction and subtracted.
2. **Search** ([`estep.py`](../../relax/multibody/estep.py), [`layout.py`](../../relax/multibody/layout.py)). This is
   an auto-refine local search around the previous residual pose with zero translation prior. The body prior
   (`sigma_tilt_from_ninety`, `sigma_psi_from_zero`, cut at 3σ) multiplies the parents' priors, and every scored
   row is composed as `MBL R MBR`. The refined body offset is applied to the image; the new offset is that offset
   plus the best translation. A body is scored against the poses the earlier bodies of the same iteration have
   just written.
3. **Maximisation** ([`mstep.py`](../../relax/multibody/mstep.py)). The two half accumulators of the body are joined
   below `--low_resol_join_halves`. The FSC is the solvent-corrected FSC of the unregularised half maps in the body
   mask, read in RELION's file convention, since the phase-randomised shells make it depend on the sign. Its phase
   randomisation advances the leader's `rand()` stream seeded with the run seed. Each half then gets its prior
   from that FSC, the maps are solved in the body frame and moved back by `+com`, with no wrap, mask or solvent
   flatten.

The shared values follow the K=1 rules, applied per body where RELION keeps them per body:

- `ave_Pmax` is half 1's `n_refined * sum(body-0 Pmax) / sum(body masses)`.
- The noise spectra are summed over the bodies.
- `incr_size` and `has_high_fsc_at_limit` are updated from each body's FSC in turn.
- A body is frozen when `old_step < 0.9 * acc_rot[b]`.
- The norm, scale and translation sigma are not updated.

Pass 1 of the adaptive search scores at RELION's `image_coarse_size`, which depends on the consensus
`rlnParticleDiameter`.

## Last iteration

At convergence RELION runs one more iteration with the halves joined: the unfiltered half maps
(`run_half<h>_body<BBB>_unfil.mrc`), the masked FSC and one map per body from the summed accumulators with the
whole-data prior (`run_body<BBB>.mrc`).

## Agreement with RELION

Multi-body parity references must come from a RELION build whose GPU back-projection accumulates in double
precision (bpdouble): stock RELION 5.0.1 sums an iteration's back-projection in float32 and loses low-shell weight
(relax#65, relax#92; see "Exact float32 accumulation" in [the algorithm map](relion_refinement_algorithm.md)). In
the final iteration on EMPIAR-10180 (327k particles, four bodies), the worst body kept 0.85 of its weight at DC.
Against a bpdouble build, relax's final maps there agree to 5e-5 per shell over shells 1-40.

On two simulated 5000-image datasets (5nrl, body 2 twisted about an axis through a pivot,
`scripts/prepare_multibody_benchmark.py`), relax and RELION 5.0.1 were run on the same GPU model (A100-SXM4-80GB):

| Dataset | Iterations | Schedule | Body poses identical | Final maps (corr) |
| --- | --- | --- | --- | --- |
| noise 1, ±8° | 5 + final | identical | 99.6% (it1), 97.8-99% (it2-3), 96.5% / 96.8% (final) | 0.9975 / 0.9996 |
| noise 0.1, ±15° | 15 + final | identical (sizes, resolution, accuracies, freezing, HEALPix order, perturbation) | 99.8% (it1), 99-99.3% (it2-3) | 0.9986 / 0.9924 |

On the noise-0.1 set, both programs give the same ground-truth scores. Body 1's motion error is 1.01°, and its
masked FSC area against the true body map is 0.965 in both. Body 2's recovered twist correlates with the true twist
at 0.79. [`scripts/compare_multibody_relion.py`](../../scripts/compare_multibody_relion.py) and
[`scripts/evaluate_multibody_gt.py`](../../scripts/evaluate_multibody_gt.py) compute these.
[`scripts/replay_multibody_iteration.py`](../../scripts/replay_multibody_iteration.py) replays one RELION iteration
from RELION's own state.
