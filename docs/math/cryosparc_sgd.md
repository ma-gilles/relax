# Coarse K-class momentum SGD

This opt-in InitialModel optimizer is inspired by the ab-initio method in
[Punjani et al. (2017)](https://www.nature.com/articles/nmeth.4169), particularly
equations 8–11 of its
[supplement](https://www.cs.toronto.edu/~fleet/research/Papers/cryoSPARC-Suppl-1.pdf).
It is an experimental alternative to RELAX's native VDAM update, not a
reproduction of the proprietary cryoSPARC implementation. Native VDAM remains
the default. Positivity is disabled; signed maps are allowed.

Implementation owners are [the optimizer](../../relax/sgd_initial_model/optimizer.py),
[noise estimation](../../relax/sgd_initial_model/noise.py), and
[the shared iteration loop](../../relax/vdam/iteration_loop.py). Mathematical
and option contracts are tested in
[test_sgd_initial_model.py](../../tests/unit/initial_model/test_sgd_initial_model.py).

## Shared inference and controlled differences

The model contains K independent volumes and latent class, rotation, and shift.
The existing engine computes joint class/pose responsibilities and backprojects
posterior-weighted image residuals. It retains the native CTF handling, image
units, class/direction probabilities, shift prior, projection interpolation,
and significant-support policy. Ground-truth maps, classes, and poses are
evaluation inputs only.

The candidate requires oversampling zero, a fixed HEALPix order, and an explicit
Fourier radius schedule. With oversampling zero, the engine's second computational
pass gathers sufficient statistics on the same coarse hypotheses; it creates no
finer pose samples. A fixed grid does not imply a fixed Fourier bandwidth.
`--stochastic-all-iterations` keeps the fixed minibatch at the last iteration,
where native K>1 InitialModel would otherwise use all particles.

The native engine scores masked images and reconstructs from raw images.
Consequently, its backprojected residual describes a reconstruction surrogate
with responsibilities held fixed. It must not be described as the exact
derivative of the marginal masked-image likelihood. The two arms retain this
same convention so that the comparison tests the update rule and noise model.

## Map update

Let G_k be the pooled residual backprojection for class k, and H_k its pooled
diagonal curvature estimate. Both use joint class/pose responsibility, including
the class mass; neither is separately normalized by class occupancy. Two
pseudo-halfsets are retained for compatibility with the shared engine and then
summed. They do not produce a disagreement gate in the candidate.

One scalar per class controls the step:

\[
L_k = \max_{f\text{ in active band}} H_k(f),\qquad
d_{k,t}=0.9d_{k,t-1}+0.1\eta\,\mathcal{T}(G_k/L_k).
\]

Here \(\mathcal{T}\) denotes the Fourier-to-map update conversion, including
the implementation's axis, FFT normalization, and projector conventions.
The shared projector applies a real-space sinc-squared gridding correction
before its Fourier transform; the map-space adjoint applies that same
real-space multiplier after the inverse transform. Reference, momentum, and
gradient are projected to the active band. The final support mask can introduce
power outside that band, which the next update removes.

The native uncorrected transform is the padded DFT divided by \(N^2\), while
the inverse update divides an unnormalized inverse transform by \(p^3N\),
where N is the original box and p the padding factor. Thus the inverse update
is \(N/p^3\) times the Euclidean adjoint in the full Hermitian Fourier inner
product. This is the native coordinate scaling of the step, not a fitted
normalization constant. Gradient tests must include the same active support
and Hermitian ownership.

The initialized momentum is zero. An unsupported class receives no update.
Multiplying both G and H by a common minibatch normalization leaves the step
unchanged. Maximum gridded diagonal curvature is an approximate metric, not a
claim that the full interpolated Hessian's spectral norm has been computed.

The map update retains spherical support and the prescribed bandwidth. It has
no positivity projection, explicit tau-squared volume penalty, VDAM shell gate,
VDAM disagreement second moment, or class-occupancy step multiplier. Projector
power and tau-squared metadata used elsewhere in the native infrastructure do
not constitute a candidate volume prior.

## Noise update

Estimate the initial background pixel variance from particle-image corners.
Convert it through the actual scoring mask, which blends pixels with the
weighted background mean, to a radial Fourier variance \(c_s\). This conversion
is essential: raw-image white variance is not the variance of masked images.
The result uses native per-component noise units, including the image FFT
normalization. No simulator noise value is read.

For the posterior residual variance estimate \(v_{t,s}\) from a batch of
effective mass \(M_t\), accumulate:

\[
A_{t,s}=\gamma A_{t-1,s}+M_t v_{t,s},\qquad
B_{t,s}=\gamma B_{t-1,s}+M_t,\qquad \gamma=0.9999.
\]

Only observed shells accrue data and mass. Unobserved-shell accumulators decay.
With \(w_t=2500\gamma^t\), set

\[
\sigma^2_{t,s} =
\frac{A_{t,s}+50c_s+8w_t c_s}{B_{t,s}+50+w_t}.
\]

Before the first update, this is the mean of the two priors, approximately
7.86 times the background variance. The persistent white prior has mass50;
the inflated initial prior has mass2500 and eight times the background
variance. Data progressively reduce its influence.

The supplement's printed noise equation and its definition of the residual
average are ambiguous about normalization. This implementation explicitly
stores a residual numerator and a matching observation count. It also uses
observation-space counts, rather than the supplement's CTF-squared exposure.
These are documented adaptations to the existing RELAX residual statistics.
The native VDAM arm retains its existing noise estimator.

## Scientific comparison and limits

Use identical data, seed maps, particle order, batch size, coarse grid,
significant-support settings, bandwidth schedule, masks, and GPU model. Record
the initialization and subset hashes, effective precision, native-library
identity, and whole-update time. Shared overrides apply to both optimizers;
they do not alter the default VDAM path.

The first fixture is the existing signed, three-state ribosome simulation:
20,000 particles, box64, pixel size6 Angstrom, noise_level1, known CTF and unit
contrast. The initial comparison uses K3 and seeds11/12. Report per-state fitted
FSC curves and mean FSC over shells1–15, class occupancies, global-frame pose
errors, and maps. Confidence alone is not an alignment metric. Two failed
controls cannot establish candidate equivalence; additional seeds or a more
informative control are then necessary.

The fixed coarse-grid baseline must be rerun. Earlier successful native K3
results used a different sampling schedule and provide context only. This
document specifies the method; measured scientific acceptance and timing belong
to the associated immutable run receipts and comparison report.

The trajectory files are evaluation artifacts. General restart of the
candidate's momentum and running noise statistics is not implemented; native
VDAM continuation cannot be used to resume this optimizer.
