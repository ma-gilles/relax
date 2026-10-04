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

## `--shell_pair_counting {relion,once}`

RELION keeps Fourier volumes as the half `x >= 0` and its shell statistics loop over every stored
entry with weight 1. A Hermitian pair `(k, -k)` with `kx != 0` is stored once; on the plane
`kx = 0` both members are stored, so those pairs count twice, in the sum and in the count alike.
The shell statistic is then a reweighted mean that leans towards the `kx = 0` plane; it equals the
mean over the whole Fourier grid only when the quantity is isotropic.

`once` weights the entries whose mate is stored too by 1/2 (the `kx = 0` plane, an even grid's
Nyquist plane, and the origin, which is its own mate), which makes every sum and count exactly
half the one over the whole Fourier grid, every voxel once. Both public layouts of an accumulator
give that statistic: the full layout expanded from RELION's x-half, and the native packed half
that accumulators of 200M voxels or more are repacked to (`_pair_once_weights`,
`relax/reconstruction/regularization_relion.py`).

| RELION loop | relax function | Feeds |
| --- | --- | --- |
| `BackProjector::updateSSNRarrays` (`backprojector.cpp:1066-1090`, `1129-1190`) | `_compute_relion_weight_shell_stats` (device, host and native-grid branches), through `compute_relion_tau2_from_weights` and `compute_data_vs_prior` | K=1 sigma2 and tau2 = SSNR sigma2; Class3D data_vs_prior; the current-size and resolution scheduling |
| `calculateDownSampledFourierShellCorrelation` (`backprojector.cpp:995-1039`) | `compute_relion_fsc_from_backprojector` (streamed packed-half and full-expansion paths) | the gold-standard half-map FSC of every numbered iteration and of the final pass |
| `Projector::computeFourierTransformMap` power spectrum (`projector.cpp:509-530`) | `_mask_and_shell_power`, through `compute_relion_tau2_from_iref_power_spectrum` | the Class3D tau2 of every iteration and of the final pass (under `once` the class spectrum is built separately; the scoring projector's own spectrum keeps RELION's counting) |
| `getSpectrum` of the start-up reference (`fftw.cpp:1010-1039`, `ml_model.cpp:1588`) | `_relion_power_spectrum_3d` / `_whole_grid_power_spectrum_3d`, through `relion_initial_tau2_and_data_vs_prior` | the start-up tau2 and data_vs_prior |

Not changed: the 1/1000 weight floor inside the reconstruction (`BackProjector::reconstruct`,
`backprojector.cpp:1515-1574`) is computed in RECOVAR (`_relion_reconstruct_floor_volume`) and keeps
RELION's counting. It only replaces weights below a thousandth of their shell mean, so the two
countings give the same map unless a voxel sits within the difference of the two shell means of
that threshold. `getFSC` on real-space maps has no caller on the refinement path.

## `--noise_shell_count {relion,summed}`

The M-step turns the noise sums of an expectation into
`sigma2_noise[s] = wsum[s] / (2 sumw Npix_per_shell[s])` (`ml_optimiser.cpp:5246-5285`).
`Npix_per_shell` counts the pixels of every shell on the full image (`:5717-5730`). The sums of
shells up to `current_size / 2` run on the cropped image (`windowFourierTransform`, rows
`-(cs/2 - 1) .. +cs/2`, `fftw.h:807-856`), which has no row `-cs/2`; higher shells come from the full
image (`power_img`). Below the box the pixels `(jp >= 1, ip = -cs/2)` of shell `cs/2` are therefore
counted and never summed, and `sigma2_noise[cs/2]` is low by that fraction: 52 of 56 pixels at box
64 and current size 32 (-7.1%), 89 of 94 at 128/64 (-5.3%), 212 of 220 at 256/128 (-3.6%). Every
other shell, and every shell at the box, is exact.

`summed` divides each shell by the pixels the expectation summed
(`summed_noise_pixels_per_shell`, `relax/reconstruction/noise_relion.py`), through
`update_posterior_noise_variance` for the K=1 per-half, the Class3D shared and the per-optics-group
updates; the numbered iterations pass the expectation's image current size. The sums themselves
and the kernels that accumulate them are unchanged. Not changed: RELION's average CTF^2 of
CTF-premultiplied images divides by the same `Npix_per_shell` and keeps it.

## `--initial_noise_pair_counting {relion,once}`

The start-up noise spectrum (`calculateSumOfPowerSpectraAndAverageImage`,
`ml_optimiser.cpp:2968-2977`, and `getSpectrum` of the average image, `fftw.cpp:1010-1039`) averages
the power of every stored pixel of each image's FFTW half. The Hermitian pairs of the `kx = 0`
column (and of an even image's Nyquist column) have both members stored and count twice, in the sum
and the count. Every later `sigma2_noise` update counts each pair of `kx = 0` once
(`!(jp == 0 && ip < 0)`), so the first spectrum and the later ones are different shell means.

`once` takes the shell mean over the whole 2-D transform instead
(`whole_transform_power_spectrum`, `relax/refinement/startup_noise.py`), for the image spectra and
for the average image's spectrum. The command applies it when it estimates the start-up noise
(`prepare_startup_noise` -> `estimate_startup_sigma2` -> `compute_avg_unaligned_and_sigma2`), for
K=1 and Class3D; the start-up tau2 and data_vs_prior follow from that noise.

## `--nyquist_column_counting {relion,once}`

RELION's per-image sums run over the FFTW half image and drop the redundant half of the `kx = 0`
column with `!(jp == 0 && ip < 0)` (`MlOptimiser::updateImageSizeAndResolutionPointers`, where
`Mresol_fine`, `Mresol_coarse` and `Npix_per_shell` are filled, `ml_optimiser.cpp:5717-5730`). On a
full-size image of even size `N` the last stored column `jp = N/2` is its own mirror: `(N/2, ip)`
and `(N/2, -ip)` are one Hermitian pair, both stored, and the rule drops neither. Those pairs count
twice where every other pair counts once, in the Gaussian score, the image power, the noise, norm
and scale sums and in `Npix_per_shell`. A cropped image has no such column (its last column is
`jp = cs/2` of a full-size row that has a distinct mate), so only expectations at the box are
affected, and there only shells `>= N/2`, which lie outside the reference sphere.

`once` leaves out the members `jp = N/2, ip < 0` (`redundant_nyquist_column_pixels`,
`relax/helpers/half_spectrum.py`), through arrays built in Python only:

| Quantity | Function |
| --- | --- |
| Gaussian score weights of pass 1, pass 2, the first-iteration tree rescore and local search | `make_scoring_half_image_weights` (`_compute_k_class_significance_batched`, `_pass2_half_weights`) |
| image power of the coarse pass (the high-shell constant of the squared difference) | the processed image of `_compute_k_class_significance_batched` |
| noise, image-power, norm-correction and scale sums of pass 2 and local search | the image the noise statistics read in `prepare_unshifted_bucket_operands` |
| `Npix_per_shell` of the noise update | `make_relion_noise_shell_indices_half`, through `normalize_wsum_to_sigma2_noise` |

The first-iteration normalized cross-correlation applies no mask at all and is not changed by this
option. The expected-accuracy estimate is not changed either: the reference is zero on those
pixels. Refused: CTF-premultiplied images (RELION's average CTF^2 divides by `Npix_per_shell`),
the experimental `gemm_dense` coarse engine, subtomograms and full-grid GEMM passes.

## `--firstiter_cc_support {relion,gaussian}`

The first iteration of a refinement from a low-resolution reference scores by normalized
cross-correlation (`--firstiter_cc`). RELION's CC kernels loop over every pixel of the cropped FFTW
rectangle with no mask: both copies of the `kx = 0` column, the DC pixel and the corners beyond
shell `cs/2` enter the cross term, the reference power and the image power `Xi2`. Every Gaussian
iteration instead scores on `Mresol_fine > 0`: shells `1 .. cs/2`, without `jp = 0, ip < 0` and
without DC. The first iteration therefore picks its poses on a different set of pixels, with the
`kx = 0` column at double weight, from all later ones.

`gaussian` scores the CC iteration on the Gaussian support (`gaussian_support_weights`,
`relax/helpers/half_spectrum.py`), through arrays built in Python only:

| Quantity | Function |
| --- | --- |
| weights of the cross term and the reference power, pass 1, the tree rescore and pass 2 | `make_scoring_half_image_weights(firstiter_cc_support_size=...)` (`_compute_k_class_significance_batched`, `_pass2_half_weights`) |
| image power of pass 1 | `_relion_cc_inverse_power_from_processed(power_weights=...)` |
| image power `Xi2` of pass 2 | `prepare_unshifted_bucket_operands(cc_power_weights=...)` |

The noise, norm and scale sums of the CC iteration and its backprojection are RELION's (they are
not part of the score). Refused: a run without a CC iteration (`--no-firstiter_cc`), the
experimental `gemm_dense` coarse engine, subtomograms and full-grid GEMM passes.
