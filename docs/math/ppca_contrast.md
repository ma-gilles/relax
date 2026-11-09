# Per-particle contrast in the pose-marginal PPCA

Status: derivation (section 1-6), implementation and evidence are in
[the progress log](../development/ppca_contrast_progress.md). The unit-contrast model of
[vdam_ppca_algorithm.md](vdam_ppca_algorithm.md) (section 4 and 16.10) is the special case `tau = 0`.

## 1. Model

Particle `i` has pose `phi = (R, t)`, a latent `z in R^q`, a contrast `c` and an image `y` (Fourier
coefficients inside the score window, noise-whitened by `D = Sigma^-1`, diagonal):

```
y = c A_phi (mu + W z) + eps,    z ~ N(0, I_q),    eps ~ N(0, Sigma),    c ~ p(c).
```

`A_phi` is the CTF-weighted slice at the pose and the translation phase. The contrast `c` multiplies the
whole signal and not the noise, as in the simulator (`recovar.simulation.simulator.simulate_data`,
`per_image_contrast`). A subtomogram particle has one `c` for all its tilts: the stream sums the tilt
frames inside each GEMM, so the statistics below are already per particle.

## 2. Per-pose statistics

With `a = A mu`, `B = A W` and `<u, v> = Re u^* D v`, the stream's two GEMMs
([full_row_kernels.py](../../relax/ppca_refinement/full_row_kernels.py), `_latent_block`) give for every
pose

```
s = <y, y>,  t = <a, y>,  nu = <a, a>,      g = B^* D y (q),  h = B^* D a (q),  H = B^* D B (q x q).
```

`t` and `g` depend on the pose and the translation, `nu`, `h`, `H` only on the rotation and the image.
Given `c` the log-likelihood is quadratic in `z`:

```
-2 log p(y | phi, c, z) = r(c) - 2 c z^T (g - c h) + c^2 z^T H z + const,     r(c) = s - 2 c t + c^2 nu.
```

## 3. Posterior given the pose

Integrating `z` for fixed `c`, with `M(c) = I + c^2 H` and `b(c) = c (g - c h)`:

```
log p(y | phi, c) = -1/2 [ r(c) - b^T M^-1 b + log det M ] + const,
z | y, phi, c ~ N( M(c)^-1 b(c),  M(c)^-1 ).
```

The pose posterior needs `p(y | phi) = int p(c) p(y | phi, c) dc`, and the pose weights `gamma(phi) = p(phi | y)`
are the normalized `pi(phi) p(y | phi)`.

## 4. What the M-step needs

The part of the expected complete-data log-likelihood that depends on `Theta = [mu, W]` is
`-1/2 E || y - c A Theta alpha ||_D^2` with `alpha = (1, z)`. Expanding the square, it depends on the
posterior only through two moments per pose:

```
a~(phi) = E[ c alpha | y, phi ]      (1 + q numbers):  E[c], E[c z]
G~(phi) = E[ c^2 alpha alpha^T | y, phi ]  (tri(1 + q) numbers):  E[c^2], E[c^2 z], E[c^2 z z^T]
```

and `gamma(phi)`. The stream's M-step already consumes exactly these shapes: pass 2 forms `weights =
gamma a~` `(P, R, B, T)` for the RHS GEMM and `sums = sum_t gamma G~` `(tri(P), R, B)` for the metric GEMM
([full_row_kernels.py](../../relax/ppca_refinement/full_row_kernels.py), `moment_block`). With `c = 1`,
`a~ = (1, E z)` and `G~ = E alpha alpha^T`, the present engine. Nothing downstream of the per-pose
latent stage changes: the backprojection, the residual gradient, the metric `lhs_tri`, the noise
estimate (`E || y - c A Theta alpha ||^2` is the same expression in `a~`, `G~`) and both optimizers read
the statistics unchanged.

So the intuition "the posterior first and second moment of `c`" is the `(0)` and `(0, 0)` entries of
`a~` and `G~`. They are not enough for `Theta`: `c` and `z` are coupled in the posterior, and the
loadings need `E[c z]`, `E[c^2 z]` and `E[c^2 z z^T]`. The moments of `c` alone suffice when `z` has no
influence on `c` (`q = 0`, or no active loadings).

The prior's hyperparameters need only moments of `c`: for `c ~ N(1, tau^2)`, the EM update is
`tau^2 = mean_i E[(c_i - 1)^2 | y_i]`, with `E[c] = a~_0` and `E[c^2] = G~_00` summed over poses with weights
`gamma`. The prior mean stays 1: the scale of `mu` absorbs any other value.

## 5. Which prior gives exact statistics

**`q = 0`.** `r(c)` is quadratic in `c`: a Gaussian prior is conjugate, `E[c]` and `Var[c]` are closed
forms of `t, nu`, and the pose evidence is a Gaussian integral.

**`q >= 1`, contrast on the whole signal (the physical model).** In the eigenbasis of `H` (eigenvalues
`lambda_k`, `g_k`, `h_k` the rotated vectors)

```
log p(y | phi, c) = -1/2 [ r(c) - sum_k c^2 (g_k - c h_k)^2 / (1 + c^2 lambda_k) + sum_k log(1 + c^2 lambda_k) ].
```

The `log(1 + c^2 lambda_k)` and the rational terms stop any standard prior from giving a closed-form
integral: there is no conjugate prior. The `c`-integral is one-dimensional and smooth, so it is done by
quadrature on a grid of contrast values `c_k` with prior weights `omega_k`; each node is one latent solve
with `M(c_k)`, and `a~`, `G~` are the node-weighted sums of the per-node posterior moments. This is
"model B" below. RECOVAR implements it on the six statistics as
`recovar.ppca.contrast_posterior.solve_latent_posterior` (`contrast_mode="marginalize"`; known pose,
documented in RECOVAR's `docs/math/contrast_in_ppca_em.md`).

**Contrast on the mean only ("model A", exact).** Let `c = 1 + delta`, `delta ~ N(0, tau^2)`, and drop the
product `delta * A W z` of the physical model:

```
y = A [ (1 + delta) mu + W z ] + eps.
```

With `e = delta / tau ~ N(0, 1)` this is the ordinary PPCA with `q + 1` latents `(e, z)` and loadings
`[tau mu, W]`: the first loading is the mean itself. Everything is Gaussian, so `a~`, `G~` and the pose
evidence are exact and need no quantization. With `b_A = (tau (t - nu), g - h)` and
`H_A = [[tau^2 nu, tau h^T], [tau h, H]]` the score is
`-1/2 [ s - 2 t + nu - b_A^T (I + H_A)^-1 b_A + log det (I + H_A) ]`, the posterior of `(e, z)` is
`N((I + H_A)^-1 b_A, (I + H_A)^-1)` and the contrast estimate is `1 + tau E[e]`.

In terms of the physical model, model A uses `u = c z` as its heterogeneity coordinate and a prior on `u`
that does not depend on `c` (the physical model has `u | c ~ N(0, c^2 I)`). The difference is a rescaling
of the prior covariance of `u` by `c^2`, which matters only along directions the likelihood does not
determine. The loadings then absorb `sqrt(E c^2)` and the latent `u` keeps the contrast scale of its
particle: `u / (1 + tau E[e])` estimates `z`. How accurate this is against model B is measured in the
progress log.

## 6. Model A inside the engine

The stream needs no new kernel: the model volumes passed to it are `[mu, W, tau mu]` with `q + 1` loadings.
The statistics come back for `P = q + 2` columns; the tied columns are folded back with the chain rule
(`theta_aug = T theta`, `T` maps `[mu, W]` to `[mu, W, tau mu]`):

```
residual_gradient  ->  T^T residual_gradient,       lhs_tri  ->  T^T LHS T,       rhs (dense engine)  ->  T^T rhs.
```

The metric `T^T LHS T` is the exact Hessian of the expected objective, so the VDAM and momentum-SGD
updates see a consistent gradient and metric of `[mu, W]`. `tau` is updated from the batch with the
second moment `E[e^2]`, the `(q + 1, q + 1)` diagonal entry of `sum_t gamma E[alpha alpha^T]`, which pass 2
already holds per rotation and image. Cost: one more channel in the inner-product GEMM, and `tri(P + 1)`
instead of `tri(P)` packed moment channels in the Gram and metric GEMMs and the elementwise stages, a factor
`(q + 3) / (q + 1)` on those (1.18 at `q = 10`, 1.67 at `q = 1`).

Model B needs a per-pose latent solve per node and the per-node posterior in pass 2 (`kept` holds `t, g`
instead of the latent means): `C` nodes cost `C` times the elementwise stages of passes 1 and 2, not the
GEMMs.

## 7. Behaviour to expect

- Early in a run the model is a blob and the poses are wrong: an image that matches the model poorly can
  explain its mismatch with a contrast far from 1. The prior bounds this (`delta ~ N(0, tau^2)`), but an
  estimated `tau` can inflate in the same way; section 16.10 of
  [vdam_ppca_algorithm.md](vdam_ppca_algorithm.md) records that failure for a point estimate. A fixed
  small `tau`, or switching the contrast on only after the poses settle, are the levers; the progress log
  records which one is used and why.
- The per-pose posterior of `c` is narrow when the image has signal: the integrated signal-to-noise `nu` of
  a particle is of order 10^2, so `sd(c | y, phi)` is about `nu^-1/2`, a few percent to ten percent. A
  quadrature grid built on the prior alone would then miss the likelihood; model B needs nodes adapted to
  each pose.

## 8. Measured: model A against the exact model B

[model_a_vs_exact.py](../../scripts/ppca_contrast/model_a_vs_exact.py) draws images from the physical model
(`y = c (a + B z) + noise`, white noise, one known pose, `nu = 200`, three latents, 1,500 images), computes the exact
posterior of model B by quadrature (801 nodes on `1 +- 6 sd`, `z` in closed form), model A's closed form and the
unit-contrast model, and also ranks the true pose against 12 wrong poses whose images correlate with the true one by
0.95 (a neighbouring pose of the grid). Loading-to-mean energy ratio `r`, contrast sd `sd`:

| r | sd | corr(c_A, c_B) | sd of c given y, A / B | R^2 of z: B / A over c / unit | pose margin (nats): unit / A / B | true pose wins: unit / A / B |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.30 | 0.999 | 0.069 / 0.068 | 0.839 / 0.816 / 0.807 | 3.31 / 4.20 / 4.21 | 0.779 / 0.777 / 0.777 |
| 0.3 | 0.30 | 0.999 | 0.069 / 0.068 | 0.933 / 0.926 / 0.878 | 4.71 / 5.63 / 5.66 | 0.824 / 0.823 / 0.827 |
| 1.0 | 0.30 | 0.999 | 0.069 / 0.068 | 0.975 / 0.972 / 0.906 | 10.19 / 11.14 / 11.16 | 0.889 / 0.896 / 0.895 |
| 3.0 | 0.30 | 0.999 | 0.069 / 0.068 | 0.988 / 0.986 / 0.914 | 27.35 / 28.31 / 28.34 | 0.945 / 0.953 / 0.953 |
| 1.0 | 0.15 | 0.998 | 0.064 / 0.063 | 0.981 / 0.981 / 0.965 | 9.81 / 9.98 / 9.99 | 0.955 / 0.956 / 0.957 |
| 0.3 | 0 (model sd 0.15) | 0.987 | 0.064 / 0.063 | 0.950 / 0.950 / 0.953 | 4.64 / 4.62 / 4.63 | 0.915 / 0.917 / 0.917 |

- Model A's contrast posterior is the exact one to three digits in correlation and in width, at every ratio:
  the dropped term `delta * A W z` does not change what the image says about `c`.
- The latent of model A, divided by its contrast estimate, recovers model B's accuracy to 0.01 (0.023 where the
  loadings are weak, `r = 0.1`); the unit-contrast latent loses 0.05 to 0.07 of R^2 at sd 0.3, and model A's
  undivided latent `u` is as bad as the unit one, so the division is what carries the gain.
- The pose evidence of models A and B agree to 0.5 percent. Modelling the contrast adds 0.2 to 1.0 nats of
  margin at sd 0.3; the fraction of images whose true pose wins moves by less than 0.01, so in this toy the
  contrast changes how confident the pose posterior is more than which pose wins.
- With no contrast spread in the data and a model prior sd of 0.15, nothing changes (margins within 0.5 percent,
  same latent accuracy): modelling an absent contrast is harmless at a known pose.

Model B is therefore not built into the stream. It stays the reference of these tables.

## 9. The scale gauge, and why the sd is estimated from a centred moment

Model A has a flat direction in the likelihood: scaling the mean volume by `a` and shifting the contrast
latent so that `(1 + sd e) mu` is unchanged fits every image equally well. Only the prior `e ~ N(0, 1)` pins it,
with a pull of `1 / sd^2` against the data's `nu` per image, which is weak at `sd = 0.3` (`nu` about 200: the
prior holds 5 percent of the precision). Any force along the valley then moves the mean volume: VDAM's
regularisation shrinks the mean each update, and the contrast latent compensates, per image, at almost no cost.

Round 1 of the experiments (progress log) updated the sd from the uncentred second moment, `sd^2 <- sd^2 E[e^2]`.
That counts the global gain, the mean of `e` over the images, as spread; the sd inflates, the prior pin weakens
(it goes as `1 / sd^2`) and the gain moves further. Observed on C n1 at box 64, 100,000 particles, seed 11:

- VDAM: the sd reached its cap (0.5) by iteration 100 to 200 on all three datasets, including the unit-contrast
  control (true sd 0); the mean contrast estimate drifted to 1.8 and the mean volume's power fell to 0.28 of the
  baseline's at iteration 1000 (amplitude 0.53, the inverse of the gain the latent had taken). The spread of the
  estimate about its own mean was right (0.30 of the mean at true sd 0.30; 0.08 at true sd 0).
- Momentum SGD: the sd estimate was good for the first 500 iterations (0.29-0.30 at true 0.30; falling to 0.07
  at true 0), then the sd rose and the mean volume's power ran away (0.56 at iteration 1000, 1.8e4 at 2000, true
  sd 0.30) until the curvature check stopped the run.

The sd update therefore uses the second moment about the mean, `sd^2 <- sd^2 (mean E[e^2] - (mean E[e])^2)`: it is
the EM update for a prior with a free mean, applied while the E-step keeps the prior mean at 0. The sd measures
the spread of the contrast; the global gain does not feed it, and the prior pin stays as strong as the spread
says. Variants that remove the freedom altogether are tested alongside it: a fixed sd, and switching the contrast
on only after the mean volume's amplitude has settled (`--contrast-start`). A hard recentring (move the mean of
`e` into the mean volume every update, `mu <- (1 + sd mean(e)) mu`, `sd <- sd / (1 + sd mean(e))`) is the next
step if these are not enough; it makes the mean volume carry the whole gain, as it does without contrast.
