# From VDAM to ab-initio PPCA: algorithm and scientific decisions

> Migration note (2026-09-23): implementation links point to RELAX. Historical pilot measurements below predate the RELAX port; they do not qualify the migrated source.

**Discussion draft, September 22, 2026. No new algorithm is implemented by
this document.** Statements marked *current* describe inspected source;
*derived* statements follow from the stated model; *proposed* choices remain
open for discussion. The aim is one PPCA model learned without supplied poses
or reference volumes, followed later by mixtures of PPCA models.

The user subsequently requested an executable handoff and accepted the random
seed-map construction below. See the [implementation plan](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_implementation_plan.md)
for work packages, proposed numerical defaults and validation, and the
[short execution handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_handoff.md) for the next agent.
This document's derivations remain the scientific context; the plan explicitly
distinguishes user decisions from bounded implementation assumptions.

## 1. Scope and source identity

The starting description is the user's
[VDAM algorithm document](/scratch/gpfs/CRYOEM/gilleslab/mg6942/vdam_dev_20260919/recovar_vdam_cap/docs/math/vdam_algorithm.md),
read at source commit `983edce72690a144b9fb26b43f10c28b3c18525d`.
Its SHA-256 is
`e834650772cf71387a901d2c1973fd74a343e25fc4592f6652bc6517087d7010`.
It is a useful description of a separate, advancing VDAM checkout, not a
guarantee about every source version or backend.

Here, *current* means the source in `codex/vdam-ppca` at
`ca440f5e4a84106b77613434ee9d7af8bf2b8b3e`, based on reconciliation
`f078ac1be64a21c2b478ef34b9f68db19920e23e`. Code links below refer to this
checkout. Recheck these claims when incorporating later reconciliation work.
The [workstream and historical recovery](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca.md) records
the recovered PPCA branches and the policy for following reconciliation.

### Already decided with the user

| Topic | Decision |
| --- | --- |
| Model | One mean volume and a low-rank loading matrix; mixtures later. |
| Starting information | No supplied poses or reference volumes. |
| Initialization | User accepted three VDAM-style random seed maps, their average and two normalized contrasts as mean/loadings. PPCA starts immediately; no preceding refinement. The suggested 10% loading-power ratio was not selected. |
| First optimizer | Coupled 3x3 mean/loading block per pseudo-halfset, vector first moments and shared adaptive scaling for the loading vector; no independent per-PC normalization. Numerical floors and normalization still need to be specified. |
| Gradient-disagreement shrinkage | Working direction from the latest discussion: retain it in the first PPCA version, and flag its interpretation and behavior for follow-up. The PPCA extension is still a proposal. |
| Loading regularization | Separate mean/loading shell gates, one loading gate shared across PCs, same initial VDAM strength schedule for both; no extra volume prior initially. |
| Pose/resolution schedule | Global PPCA pose scoring and a controlled coarse-to-fine schedule, same bandwidth for mean and loadings; initially reuse VDAM step-size and strength schedules. |
| Noise and contrast | Known CTF and unit contrast; infer the noise spectrum from particles, including its initially overestimated signal contribution. Do not supply the simulator's true noise to training. |
| First experiment | Three Ribosembly states, about 20k particles, box 64, balanced populations, simulator `noise_level=0.01`. Independent K=3 VDAM and mean-plus-two-loadings PPCA; class-averaged inferred coordinates for evaluation. |
| Repeats | One debugging initialization, followed by three independent initializations on the same dataset before drawing conclusions. |

The execution plan supplies proposed defaults for batch sizes, numerical floors
and grid milestones. They are explicit assumptions, not additional user
decisions or validated settings. The map triplet and acceptance thresholds
remain to be pinned. Production computations target float32/complex64; existing
deliberate metadata precision remains a separate contract.

## 2. What changes in one iteration?

VDAM's K-class model is a probabilistic mixture over class and pose. It is not
literally k-means. PPCA replaces the discrete class variable by a continuous
Gaussian coordinate, while still integrating over unknown poses.

| Stage | Current K-class VDAM | One PPCA model |
| --- | --- | --- |
| Model state | K reference maps, class/direction probabilities, scalar reconstruction weights and moments per map | Mean `mu`, loading volumes `W`, pose probabilities, coupled mean/loading statistics; optimizer state depends on our choice |
| Particle subset | Scheduled subsets, optionally split into two pseudo-halfsets | Retain pseudo-halfsets for gradient disagreement; batch sizes and normalization remain open |
| Pose scoring | Residual against each class projection | Marginal likelihood after analytically integrating over the Gaussian coordinate |
| Posterior | Joint class/pose weights | Pose weights and conditional first/second latent moments |
| Backprojection | Residual and scalar curvature per class | A vector RHS and a coupled `(q+1)` by `(q+1)` reconstruction block |
| Update | Preconditioned residual, two moment estimates, shell shrinkage | Retain gradient-disagreement shrinkage; define coupled directions and basis-invariant adaptation |
| Regularization | Reference power/SSNR bookkeeping and gradient-disagreement shrinkage have distinct roles | Use one shell shrinkage factor across loading directions; any additional explicit prior needs a separate decision |
| Pose/resolution schedule | Class-map accuracy estimates, scalar data/prior shell ratios | Geometry can be reused; heterogeneity-aware accuracy and resolution criteria need a definition |
| Outputs | K maps and class/pose estimates | Mean, loadings, latent prior convention, and final pose-marginal particle coordinates |

For `q=2`, PPCA has three volume channels, but they are never three competing
classes. Its gridded statistics have three RHS channels and six independent
symmetric LHS channels, versus three RHS and three scalar LHS channels for
K=3. Reuse of projection primitives does not imply reuse of the class model.

## 3. Current VDAM: likelihood, metric and update

Let `i` index particles, `k` classes, `phi=(R,t)` poses, `f` Fourier voxels,
`s(f)` radial shells, and `j` iterations. Write `A_i,phi` for the projection,
CTF and shift operator and `D_i` for image-noise precision, with the actual
Fourier weighting convention included in the inner product.

### 3.1 Class/pose posterior and residual statistics

The model is

\[
y_i=A_{i\phi}\mu_k+\epsilon_i,
\qquad
\gamma_{ik\phi}\ \propto\ p(k,\phi)
\exp\{-\tfrac12\|y_i-A_{i\phi}\mu_k\|_{D_i}^2\}.
\]

In the native InitialModel path, the class/direction distribution carries the
class mass; adding an extra class prior would count it twice. Scoring and
reconstruction image masking are also distinct in the current workflow.

For pseudo-halfset `h`, the conceptual statistics are

\[
R_{kh}=\sum_{i\in h,\phi}\gamma_{ik\phi}
A_{i\phi}^*D_i(y_i-A_{i\phi}\mu_k),
\qquad
C_{kh}\approx\operatorname{diag}
\sum_{i\in h,\phi}\gamma_{ik\phi}A_{i\phi}^*D_iA_{i\phi}.
\]

The code stores gridded versions, with its interpolation, FFT and padding
normalizations. `C` is the diagonal reconstruction metric, not the exact
marginal-likelihood Hessian. Interpolated projections generally produce
off-voxel normal-operator terms that diagonal gridding does not retain.

Sources: [E-step](../../relax/vdam/adaptive_estep.py),
[layouts](../../relax/vdam/layout.py),
[state](../../relax/vdam/state.py).

### 3.2 Preconditioning and moments

For each class and valid Fourier voxel, suppressing `k,f`, the current
pseudo-halfset update is

\[
d_h=\frac{R_h}{\max(1,C_h)},\qquad
m_h\leftarrow0.9m_h+0.1d_h,
\]
\[
v\leftarrow0.999v+0.001\,
\frac{|d_1-d_0|^2}{|(d_0+d_1)/2|^2+10^{-12}},
\qquad
u=\frac{(m_0+m_1)/2}{\sqrt v+10^{-12}}.
\]

First moments initialize from the new direction when the native initialization
test triggers, rather than always starting with the displayed EMA. The native
test uses a sequential sum of real components; the device interface carries
explicit initialization certificates. Second moments start at one. This is
not standard Adam's second moment of the gradient, and there is no standard
Adam bias correction.

The two halves share the current model and posterior machinery. They estimate
update disagreement; they are not independently reconstructed gold-standard
half maps. In the inspected subset code, routing follows the internal particle
`part_id % 2`, carried through shuffling and optics sorting, not simply the
position in a newly shuffled subset. Also, averaging two separately
preconditioned directions is generally different from preconditioning pooled
statistics once.

Sources: [device transaction](../../relax/relion/relion_vdam_mstep.py),
[native transaction wrapper](../../relax/vdam/mstep_single_class.py),
[subset ordering](../../relax/vdam/subset_schedule.py).

### 3.3 The gradient-disagreement shrinkage gate

Let `F_k` be the uncorrected projector transform of the current reference.
The reconstruction uses shell **amplitudes**

\[
\nu_{ks}=\langle|m_{k0}-m_{k1}|\rangle_s,
\qquad a_{ks}=\langle|F_k|\rangle_s,
\qquad \rho_{ks}=2\,\mathrm{fudge}\,a_{ks}/\nu_{ks}
\quad(\nu_{ks}>0),
\]
\[
\varphi_{ks}=\min\{1,\max[\rho_{ks}/(1+\rho_{ks}),
\mathrm{fsc\_reconstruct}_{ks}]\},
\]
\[
F_k\leftarrow F_k+eta_k
\{\varphi_k u_k-(1-\varphi_k)F_k\},
\qquad
\eta_k=\eta_j\{1-\exp[-(3K+10)\,p_k]\}.
\]

Empty shells are handled separately. **At exactly `nu=0`, the inspected
device code sets `rho=0`**, rather than taking the limit from positive `nu`.
Thus identical half directions do not by themselves imply `varphi=1` in that
branch. `fsc_reconstruct` is a separate input, initially zero in the native
driver; the symbol is not evidence that independent half-map FSC is estimated
on each iteration. Classes with no mass or usable half-0 weight are skipped.

This is a Wiener-like algebraic form, but its amplitude ratio and adaptive
gradient statistics do not by themselves establish a Wiener-optimal estimator
or an exact MAP update under a stated volume prior. That distinction matters
when deciding how to regularize PPCA. We should not assume that adding an
explicit Gaussian prior and retaining this gate gives the same objective.

### 3.4 Reference power, SSNR and resolution are a different mechanism

The driver refreshes `tau2_class` from the current projector power before the
E-step. The M-step's SSNR bookkeeping uses that spectrum and half-0 weights;
it does not replace it with the gradient-amplitude estimate above.
For a populated shell and positive `tau2`, with padding factor `p`, the
implemented quantities include

\[
\sigma^2_{\rm recon}(s)=
\frac{\#s}{\sum_{f\in s}p^3C_0(f)},
\qquad
\mathrm{data\_vs\_prior}(s)=
\left\langle p^3\,\mathrm{fudge}\,\tau_s^2 C_0(f)\right\rangle_s.
\]

The first is an inverse average weight, not the average inverse weight or a
general proof of reconstruction MSE with uncertain poses. There are additional
zero-prior and empty-shell branches. The scalar data/prior crossing drives
resolution estimates; current-size selection includes frequency headroom.
Neither quantity can simply be reused as a PPCA loading variance.

Sources: `refresh_tau2_from_projector_power` and scheduling in
[iteration_loop.py](../../relax/vdam/iteration_loop.py), and
`relion_vdam_m_step_device` in the transaction linked above.

## 4. Derived PPCA likelihood and pose posterior

The proposed model, without contrast variation for now, is

\[
y_i=A_{i\phi}(\mu+Wz_i)+\epsilon_i,
\qquad z_i\sim\mathcal N(0,I_q),
\qquad\epsilon_i\sim\mathcal N(0,\Sigma_i).
\]

The coordinates are real; the Fourier representations of the real mean and
loadings are Hermitian. `W` carries the heterogeneity scale. Define

\[
r=y_i-A_{i\phi}\mu,\quad B=A_{i\phi}W,\quad D=\Sigma_i^{-1},
\quad M=I_q+\operatorname{Re}(B^*DB),\quad
b=\operatorname{Re}(B^*Dr).
\]

Here all products use the consistent real-image Fourier metric, including
Hermitian multiplicities when working in packed half Fourier storage. Then

\[
C_{i\phi}=M^{-1},\qquad m_{i\phi}=C_{i\phi}b,
\qquad E[zz^T\mid y_i,\phi]=C_{i\phi}+m_{i\phi}m_{i\phi}^T,
\]
\[
\ell_{i\phi}=-\tfrac12
\{r^*Dr-b^TM^{-1}b+\log\det M\},
\qquad
\gamma_{i\phi}=
\frac{p(\phi)e^{\ell_{i\phi}}}{\sum_{\phi'}p(\phi')e^{\ell_{i\phi'}}}.
\]

Pose quadrature weights belong in `p(phi)`. Terms constant over poses were
omitted from `ell`; the noise determinant cannot be omitted when optimizing
noise. The determinant of `M` penalizes unexplained loading capacity. Keeping
only the best coordinate fit or only `m m^T` changes the PPCA likelihood.

With a specified volume penalty `R` and fixed noise/prior hyperparameters,
the objective we would maximize is

\[
\mathcal J(\mu,W)=\sum_i\log\left[
\sum_\phi p(\phi)\,p(y_i\mid\phi,\mu,W)\right]-\mathcal R(\mu,W).
\]

Each stochastic iteration recomputes the batch's posteriors at the current
model. Updating a prior spectrum or a support rule adds another state update;
it is not automatically ascent on this same fixed objective.

This analytic core already exists in
[`compute_ppca_pose_scores_and_moments_no_contrast`](https://github.com/ma-gilles/recovar/blob/162dc9477a64888beb53af757649f141034c3cbd/recovar/ppca/pose_marginal.py)
and the [PPCA engine](../../relax/ppca_refinement/engine.py).
See the [existing refinement derivation](ppca_refinement.md) for its current
implementation conventions. Its existence is not evidence of successful
pose-free cold-start reconstruction.

## 5. Derived coupled reconstruction statistics

Write `Theta=[mu,W]`, `a=[1;z]`, and

\[
\alpha=E[a]=\begin{bmatrix}1\\m\end{bmatrix},\qquad
G=E[aa^T]=
\begin{bmatrix}1&m^T\\m&C+mm^T\end{bmatrix}.
\]

For fixed E-step posteriors, the exact reconstruction surrogate has data RHS
and normal operator

\[
\mathcal B=\sum_{i,\phi}\gamma_{i\phi}A_{i\phi}^*D_i y_i\alpha_{i\phi}^T,
\qquad
\mathcal H[\Theta]=\sum_{i,\phi}\gamma_{i\phi}
A_{i\phi}^*D_iA_{i\phi}\Theta G_{i\phi}.
\]

The data ascent direction before preconditioning is `B - H[Theta]`.
In particular, the loading gradient is

\[
g_W=\sum_{i,\phi}\gamma_{i\phi}
\left[A^*D(r-Bm)m^T-A^*DAW C\right].
\]

The `C` term is essential: posterior uncertainty contributes to curvature and
to the loading update. Mean/loading and loading/loading cross terms also
remain, even though the prior is isotropic in latent coordinates.

The existing gridded approximation gives, for each Fourier voxel, a vector
`b_f` and a coupled block `L_f`. With `theta_f=[mu_f;W_f1;...;W_fq]` and
prior precision `Lambda_f`,

\[
g_f=b_f-(L_f+\Lambda_f)\theta_f.
\]

The [augmented solver](https://github.com/ma-gilles/recovar/blob/162dc9477a64888beb53af757649f141034c3cbd/recovar/ppca/augmented_mstep.py) implements this
kind of block system. The block curvature is expected **complete-data**
curvature; it is not generally the observed marginal-likelihood Hessian.
The per-voxel solve additionally approximates the spatial normal operator.
For the selected VDAM controller, the implementation plan calls for direct
backprojection of the expected image residual, retaining the exact interpolated
normal action in the gradient. Use the gridded block as a metric. The
per-voxel `b-L*theta` expression is a surrogate-gradient comparator and is not
generally interchangeable with that residual gradient.

## 6. Working direction: retain VDAM gradient-disagreement shrinkage

The user favors retaining the VDAM shrinkage mechanism for the first PPCA
version and marking it for follow-up. There is no result here showing it to
be worse for PPCA than for K>1 VDAM. The shared-model and adaptive-noise
caveats also occur in VDAM; they do not by themselves justify discarding it.
The PPCA-specific task is to extend it to coupled loadings without dependence
on an arbitrary latent basis. Section 6.2 describes that candidate extension.

### 6.1 Explicit-prior comparator, not the current starting recommendation

For a uniform batch of `n_b` out of `N` particles, a full-sum objective gives

\[
H_f=(N/n_b)L_{b,f}+\Lambda_f,\qquad
g_f=(N/n_b)b_{b,f}-H_f\theta_f,
\]
\[
d_f=H_f^{-1}g_f,\qquad\theta_f\leftarrow\theta_f+\eta_j d_f.
\]

This is an **alternative for comparison**, not the selected starting direction.
For `q=2` the coupled solve is
only 3 by 3 per voxel. It respects the latent-basis symmetry with the prior
below. A common scalar step without this block inverse is an even simpler
alternative, but may be poorly conditioned across frequencies and channels.
Numerical damping, if necessary, must be stated separately from a scientific
prior; a spectral eigenvalue floor or scalar identity damping preserves the
symmetry. Elementwise manipulation of a block does not generally do so.

With full data and `eta=1`, the displayed, undamped formula equals the joint
block M-step before masking or grid corrections. On a batch it is a damped
batch M-step, giving a useful algebraic check. It is not the full VDAM moment
algorithm. Moreover, a batch-dependent inverse makes the preconditioned
direction generally biased relative to the full-data preconditioned direction;
we should not claim ordinary unbiased-SGD convergence from the notation.

Scale **both** data statistics by `N/n_b` and apply the prior once. If using
a mean likelihood instead, change the prior convention accordingly. Otherwise
increasing the batch size silently changes the amount of regularization.
If treating each pseudo-halfset as a full-data estimator, use its own count.

### 6.2 Proposed PPCA extension of the retained VDAM mechanism

The scalar VDAM denominator would become a coupled metric. First moments can
remain vectors. Independent coordinatewise second moments cannot simply be
copied, because a latent rotation mixes coordinates.

The selected invariant adaptation keeps a scalar second moment for the mean
and **one shared scalar for the loading vector at each Fourier voxel**. For
loading directions `d_Wh`, a candidate disagreement ratio is

\[
\frac{\|d_{W1}-d_{W0}\|_2^2}
{\|(d_{W0}+d_{W1})/2\|_2^2+\varepsilon}.
\]

A candidate shell amplitude gate uses the loading-vector norms:

\[
a_{Ws}=\left\langle\|W_f\|_2/\sqrt q\right\rangle_s,\qquad
\nu_{Ws}=\left\langle\|m_{W0,f}-m_{W1,f}\|_2/\sqrt q\right\rangle_s,
\]
\[
\rho_{Ws}=2\,\mathrm{fudge}\,a_{Ws}/\nu_{Ws},\qquad
\varphi_{Ws}=\rho_{Ws}/(1+\rho_{Ws})\quad(\nu_{Ws}>0),
\]
\[
W_f\leftarrow W_f+\eta_j
\{\varphi_{Ws}u_{W,f}-(1-\varphi_{Ws})W_f\}.
\]

Here W and its directions are in the same reconstruction Fourier coordinates.
The same scalar `varphi_Ws` multiplies every loading direction; the mean can
retain its own scalar VDAM gate. The norms are unchanged under `W -> WQ`.
The displayed gate has no added FSC floor; an independent reliability floor
and the exactly-zero-disagreement branch in section 3.3 need explicit choices.
These are symmetry-preserving proposals, **not derived optimal noise estimators**.
Whether to compare directions preconditioned separately, or in a shared pooled
metric, is another choice: their disagreement has different interpretations.

The first candidate should not automatically add the explicit prior in 6.1
on top of this shrinkage. That would introduce another regularizer. A distinct
mean gate and loading gate also need not commute with the coupled metric.
We must specify the resulting objective or clearly label a heuristic; neither
the scalar VDAM update nor the fixed-pose PPCA prior settles this automatically.

For the likelihood-only coupled direction, the execution plan uses
`d_h = P(L_h)^-1 r_h`, with the directly accumulated expected residual gradient
`r_h` from section 5. P uses a basis-invariant positive spectral floor,
extending the role of VDAM's scalar `max(1,C_h)`. The floor,
units and halfset scaling remain to be pinned. A numerical metric floor is not
an additional volume prior. A random batch inverse and separate half metrics
still do not yield an unbiased full-data preconditioned gradient.

### 6.3 Batch size and follow-up questions

The user asked to inspect the existing VDAM schedule before deciding the PPCA
batch size. The previously suggested fixed 2,048-particle batch is not selected;
section 10 gives the VDAM schedule for the now-selected 20k-particle fixture.
Define batch size as the total count across both pseudo-halfsets, and record both
actual half counts and their information/coverage. Unlike K-class VDAM,
particles are not assigned to competing loading columns; their latent moments
weight coupled directions, so equal counts do not guarantee equal information.

Batch size changes the measured disagreement and therefore the shrinkage.
Under an ideal fixed-model, independent finite-variance approximation,
sampling fluctuations fall as the inverse square root of the half-batch size;
adapted moments, varying poses and separate metrics complicate that picture.
Normalize directions consistently before comparing halves. Do not assume a
change in batch size is only a throughput change.

Flag for follow-up: whether this gate suppresses emerging heterogeneity at
cold start; sensitivity to batch size; whether a strong loading subspace hides
a weak or collapsed mode under the shared gate; and behavior at zero or very
small disagreement. Record shell gates, loading power and basis-invariant
loading singular values. After the first experiment, use matched-initialization
comparisons to distinguish gate effects from changes in metric or pose support.
Agreement between halves measures stability, not correctness of the recovered
structure. That limitation is shared with K>1 VDAM.

## 7. Ambiguity: what should and should not change the result?

### 7.1 Latent rotations

For real orthogonal `Q`,

\[
W'=WQ,\qquad z'=Q^Tz
\]

leaves every represented volume and the marginal covariance unchanged.
With `S=diag(1,Q)`, Fourier coefficient columns transform as
`theta'=S^T theta`. The coupled statistics transform as
`L'=S^T L S`, `b'=S^T b`. A prior
`Lambda=diag(lambda_mu,lambda_W I_q)` commutes with this transformation, so
the proposed update satisfies `d'=S^T d`. This is the concrete ambiguity
criterion: equivalent initial models should produce equivalent updated models.

A vector first moment transforms the same way. A vector of elementwise squared
moments generally does not; retaining a fixed working basis alone does not
remove the dependence on the initial basis choice. Shared scalar or full
matrix adaptation are alternatives. Do not normalize each loading column or
whiten W as a harmless reparameterization: with `z~N(0,I)`, general scaling
changes `WW*` and therefore changes the model.

There is also no free latent-origin shift under the fixed zero-mean Gaussian
prior. Recentring posterior coordinates requires accounting for the mean and
prior, rather than silently treating it as the same model.

### 7.2 Physical alignment and pose/heterogeneity ambiguity

A global physical rotation, origin change or allowed handedness change must
act consistently on the mean, all loadings and the corresponding pose
convention. It is different from mixing loading columns by Q. Never align
each loading volume independently in physical space.

The current [rigid registration](../../relax/diagnostics/gt_registration.py)
returns a transform and registration receipt; the earlier
[rotation-grid registration](../../relax/diagnostics/gt_metrics.py) is
also available. Independent model comparisons may need physical registration
followed by orthogonal latent alignment. Column sign matching alone is not
enough. The two pseudo-half gradients at one shared model are already in the
same latent basis and need no independent Procrustes fit before subtraction.

Small rotations, translations and contrast errors can also be partly
represented by loading volumes. This is a scientific identifiability issue,
not just a plotting problem. The first experiment should monitor these modes;
explicitly projecting them out would be an additional modeling decision.

## 8. Regularization: separate mean and loading gates

### 8.1 Selected starting strategy for mean versus PCs

Use two shell-dependent shrinkage factors: `varphi_mu(s)` for the mean and
`varphi_W(s)` shared across all loading directions. The mean and loadings have
different evidence and uncertainty. Sharing one gate across both would let a
strong mean protect poorly supported heterogeneity; giving each loading its
own gate would introduce latent-basis dependence.

| Component | Signal amplitude | Disagreement amplitude | Applied shrinkage |
| --- | --- | --- | --- |
| Mean | Shell average of the mean Fourier amplitude | Shell average of the absolute difference between the two mean first moments | One scalar per shell for the mean |
| Loadings | Shell average of the loading-vector norm, divided by `sqrt(q)` | Shell average of the norm of the difference between loading first moments, divided by `sqrt(q)` | One scalar per shell shared by every loading |

Use VDAM's ratio form in sections 3.3 and 6.2 for the respective gates. The
proposed updates, in the reconstruction Fourier coordinates, are

\[
\mu_f\leftarrow\mu_f+\eta_j
\{\varphi_{\mu s}u_{\mu,f}-(1-\varphi_{\mu s})\mu_f\},
\qquad
W_f\leftarrow W_f+\eta_j
\{\varphi_{Ws}u_{W,f}-(1-\varphi_{Ws})W_f\}.
\]

The directions come from the coupled mean/loading system, retaining cross
terms and latent posterior covariance. Separate shrinkage does not mean
separate independent likelihood solves. The distinction between complete-data
curvature and an exact marginal Hessian also remains.

**Selected initial strength policy:** use the same VDAM fudge schedule for
both gates, letting their separate amplitude/disagreement estimates determine
the actual shrinkage. This introduces no new relative strength parameter at
the outset. It does not imply equal shrinkage or establish that the scalar
calibration transfers optimally to a vector norm. Separate mean/loading
strength schedules are a follow-up option if the first experiment warrants
them. Initialization amplitude, batch counts and the metric floor must still
be pinned before numerical values have a reproducible meaning.

Do not automatically add a Gaussian volume penalty on top of these gates.
The latent prior `z ~ N(0,I)` and the marginal likelihood's determinant term
remain part of PPCA regardless; they are distinct from an additional prior
on W. Spectral bookkeeping for resolution also remains separate from this
update rule, and needs the heterogeneity-aware design discussed in section 9.

Monitor both gates and loading singular values during cold start. In
particular, confirm that weak initial W can grow and that a reliable first
mode does not conceal collapse of the second. The selected strategy is not
yet an implemented or validated regularization rule.

### 8.2 Explicit-prior comparator and spectrum interpretation

The retained shrinkage mechanism in section 6.2 implements the agreed
direction-independent pattern without requiring an additional explicit prior.
For the alternative objective-based comparator, a penalty over the full Fourier
grid would be

\[
\mathcal R(\mu,W)=\tfrac12\sum_f
\left[\lambda_\mu(s(f))|\mu_f|^2+
\lambda_W(s(f))\sum_{a=1}^{q}|W_{fa}|^2\right].
\]

Packed storage must use the equivalent Fourier metric. The loading term is
the user's chosen symmetry: one precision per frequency shared across latent
directions. Frequency-dependent isotropic regularization does **not** require
diagonalizing the likelihood curvature or dropping cross terms. The mean
spectrum may be different; its specific treatment remains to be chosen.

For `z~N(0,I)`, the heterogeneity power at a Fourier voxel is

\[
E_z|[Wz]_f|^2=\sum_a|W_{fa}|^2.
\]

For that explicit-prior alternative, a shell total variance `d_s` suggests a per-loading scale proportional
to `d_s/q`, with a documented overall strength and floor. This is a candidate
parameterization, not yet a valid estimator from pose-free particles. Using
current loading power alone can create a feedback loop: early shrinkage lowers
the estimated prior variance, which causes further shrinkage. A frozen initial
spectrum or a delayed/controlled update are alternatives to discuss.

The [variance-prior notes](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/math/ppca_variance_prior_notes.md) distinguish physical
signal variance from RELION-style reconstruction tau. Existing fixed-pose
estimators depend on poses and cannot be carried into ab-initio initialization
unchanged. The [mean-prior adapter](../../relax/ppca_refinement/mean_regularization.py)
also shows that RELION's mean precision need not be a literal `1/tau` in every
branch. Prior units, FFT normalization and particle-count normalization must
be pinned before choosing numerical strengths.

Keep four different quantities separate:

| Quantity | Meaning |
| --- | --- |
| Image noise `Sigma_i` | Observation covariance in scoring and reconstruction |
| Volume/loading prior variance | Prior scale on unknown model coefficients |
| Reconstruction inverse weight | Approximate uncertainty from accumulated coverage/curvature |
| Pseudo-half gradient disagreement | Noise proxy for a stochastic optimization direction |

None can be substituted for another solely because it is called “variance”.

## 9. Pose search and resolution growth

### Current VDAM

The workflow uses global rotation/translation grids with coarse-to-fine
oversampling, posterior support selection and a finer reconstruction pass.
The usual support target is 0.999 mass with a coarse cap proportional to K
(`100*K`); a cap and threshold ties can alter the retained mass. The zero-
oversampling K=1 path and oversampled path have different normalization
boundaries, so “renormalize retained poses” is not an adequate universal
description. Canonical sampler Euler angles travel with candidate identity.

The driver estimates expected angular/translation accuracy from class-map
projection changes. Sampling updates occur periodically and are coupled to
resolution progress. Resolution estimates use scalar class data/prior curves;
current-size selection allows roughly ten shells of headroom and is not a
general monotone-resolution guarantee.

Sources: [sampling](../../relax/vdam/native_sampling.py),
[E-step](../../relax/vdam/adaptive_estep.py),
[schedules](../../relax/vdam/schedules.py), and the iteration loop.

### PPCA changes and decisions

Reuse geometric grids, canonical Euler metadata, projection and candidate
storage. Rank candidates using the **PPCA marginal likelihood**, not the mean
projection score or the residual at a best-fit latent coordinate. A pose that
looks poor for the mean may explain a valid heterogeneous state.

`q+1` is not a class count and does not justify a `100*(q+1)` cap. Initially
use generous support and measure discarded posterior mass against small dense
references. Retained pose normalization and truncation are part of the model
approximation and must be specified explicitly.

A mean-only angular-accuracy estimate ignores how W changes the score and
can miss informative variable regions. Likewise, one scalar FSC or shell
ratio for the mean cannot qualify all loading directions. Possible block
diagnostics include the eigenvalues of prior-whitened loading curvature,
after accounting for mean/loading coupling, plus gauge-invariant loading
power and reconstructed-state quality. Their threshold is not decided.

**Selected first pilot simplification:** use a documented, conservative
low-resolution bandwidth and pose-grid schedule while validating the PPCA
posterior, then compare adaptive growth. This still uses unknown global poses
and active PPCA throughout. It does not establish a production resolution
policy, and it need not force the independently characterized VDAM baseline
to abandon its native schedule.

## 10. Noise, masks, initialization and scheduling

### Noise and other nuisance variables

Current VDAM updates noise and pose/class distributions with subset smoothing;
the usual EMA coefficient is 0.9 on subsets and zero on full-data updates.
The current owner is
[estep_meta_updates.py](../../relax/vdam/estep_meta_updates.py).
PPCA needs expected residual energy, not only the residual at its posterior
mean. For pixel `p` at a pose,

\[
E[|y_p-(A\mu)_p-(Bz)_p|^2\mid y,\phi]
=|r_p-(Bm)_p|^2+(BCB^*)_{pp}.
\]

Average this over pose weights and shells with the chosen noise convention.
Dropping the covariance term underestimates residual energy. Unknown contrast
adds another ambiguity with heterogeneity and should have an explicit model.
The user selected **inferred noise**, known CTF and unit contrast for the first
fixture. An artificially fixed true noise spectrum would remove an important
part of the intended early-iteration behavior.

The inspected [initial-noise estimator](../../relax/relion/initial_noise.py)
uses up to 1,000 images per optics group by default and forms, in its Fourier
normalization,

\[
\widehat\sigma^2_0(s)=\tfrac12
\left[\langle|F y_i|^2\rangle_{i,s}
-\langle|F\overline y|^2\rangle_s\right],
\]

with nonpositive-shell repair. Because the images are unaligned, pose and
structural variation contribute to this estimate. That explains a tendency
to overestimate observation noise initially; it is not a guarantee of an
upper bound in every shell. No additional inflation multiplier or forced
monotonic decrease is implied by the inspected code.

Recommendation: initialize PPCA noise using this particle-based estimator,
then update from the PPCA expected residual above, with VDAM's 0.9 EMA on
subsets and unsmoothed update on an all-data iteration. Preserve the current
iteration timing and audit Fourier units. The known simulator noise is an
evaluation diagnostic only. Monitor whether early loading growth and noise
reduction compete; do not remove the posterior-covariance term to force the
noise downward.

### Masking and Fourier conventions

Current VDAM scores masked images and can reconstruct from unmasked images;
the current PPCA path has its own scoring-mask and Hermitian-weight options.
RELION-style stored-half weighting and PPCA's real-image inner product must
be compared with their noise/FFT conventions, not silently interchanged.
Specify whether the new objective uses one coherent observation model or a
documented scoring/reconstruction approximation. Existing scalar scores are
not enough to establish equivalence.

A common real-space solvent mask applied to the mean and every loading
preserves latent rotational symmetry, but post-update masking is still a
separate operation from a spectral prior. Independent clipping/positivity
constraints on loading columns do not preserve the PPCA model. A nonzero
Gaussian loading produces unbounded fluctuations, so requiring every possible
`mu+Wz` to be positive would require changing the distribution or constraints.
Keep existing grid-correction defaults explicit and separate from this choice.
Current VDAM solvent handling lives in
[m_step.py](../../relax/vdam/m_step.py).

### Cold start

The user accepts random initialization and suggests using the procedure that
initializes K-class VDAM. This permits its seeding procedure, not a preceding
VDAM refinement run. The earlier suggested fixed 10% loading-power ratio was
not selected.

Current [bootstrap_iref.py](../../relax/vdam/bootstrap_iref.py) and its
[C++ implementation](../../relax/relion_bind/initialmodel_bind.cpp) do:

1. Use roughly 1,000 particles by default, assign random Euler angles, and
   route particles round-robin to K initial references. This does not use
   class labels or inferred poses.
2. Backproject with CTF handling at low resolution, initially around shell
   `round(0.07 * box_size)`.
3. Construct positive/negative random blob fields, combine them as
   `positive - negative/2`, and rescale to the preceding reference's standard
   deviation; low-pass and apply a soft solvent mask.

Thus the references are random, with data-derived scales. The same procedure
serves K=4 and other K values. Reusing its scientific construction does not
authorize silently inheriting the native binding's double computation as the
new PPCA production path.

**User-selected PPCA mapping:** generate three random seed maps `v1,v2,v3` by this
procedure, then use

\[
\mu_0=(v_1+v_2+v_3)/3,\qquad
W_{0,1}=(v_1-v_2)/\sqrt6,\qquad
W_{0,2}=(v_1+v_2-2v_3)/\sqrt{18}.
\]

This requires no fitted PCA or pose refinement. It makes `W0 W0*` equal to the
three seed maps' empirical covariance with denominator three, giving a
documented scale without the arbitrary 10% rule. The user accepted this
construction for the implementation plan. It uses no GT maps.

`W=0` is an absorbing EM stationary point: `B=0`, `m=0`, and the loading
gradient vanishes. Nonzero random W avoids that exact trap, but is not a
guarantee of growth or recovery. Too small a W may collapse under the prior;
too large a W can explain away pose errors. This scale is a scientific choice,
not an incidental initializer constant.

### Batch, step and stopping schedules

The inspected VDAM defaults use 200 iterations in initial/middle/final phases
of 60/100/40, growing subsets from a clamped 0.5% toward a clamped 10% of the
data, a step schedule roughly 0.9 to 0.5, and a fudge schedule 1 to 4. K>1
has a final full-data iteration. Native completion is schedule-driven; these
constants are not a convergence proof.

For the selected 20,000-particle fixture, the current VDAM defaults give:

| Iterations (1-based) | Total particles | Approximate particles per pseudo-halfset |
| --- | ---: | ---: |
| 1–60 | 200 | 100 |
| 61–159 | 218 through 1,982, increasing by 18 per iteration | 109 through 991 |
| 160–199 | 2,000 | 1,000 |
| 200, K>1 | 20,000 | 10,000 |

The general initial count is `clamp(round(0.005*N), 200, 5000)` and the final
subset count is `clamp(round(0.1*N), 1000, 50000)`. These are **total** counts,
not per-class counts. Actual pseudo-half counts follow particle identity.
Sources: `default_subset_sizes_for_3d_initial_model` and `compute_subset_size`
in [schedules.py](../../relax/vdam/schedules.py).

Recommendation: begin with this batch schedule, including an explicitly
specified final all-data update for PPCA. The user requested this comparison
before selecting the schedule; it supersedes the fixed-batch recommendation,
but is not yet a recorded user decision. The mean/PC channels do not turn
the PPCA model into K=3 for other scheduling or class-prior rules.

The user accepted initially reusing VDAM's step-size and strength schedules,
with a controlled coarse-to-fine pose/bandwidth schedule. Their quality for
PPCA still needs validation. Regardless of
whether the optimizer ends with a full-data update, the requested evaluation
requires a fresh all-particle embedding pass at the final model.

The current InitialModel CLI still exposes inherited native/float64 M-step
defaults. That is a source fact, not our production precision policy. The
PPCA pilot and its VDAM baseline need explicit effective precision for scoring,
projection, accumulation and updates; a double-only run cannot qualify them.
No default is changed in this documentation package.
See [initial_model.py](../../relax/commands/initial_model.py) for the inspected
CLI defaults and [native_options.py](../../relax/vdam/native_options.py)
for runtime routing.

## 11. First experiment: three states, K=3 versus q=2

Use one synthetic particle set generated from three volumes in a common
physical frame. Run K=3 VDAM to characterize the fixture, then run independent
PPCA with `q=2` from its random initialization. Neither training workflow
receives true poses, true class labels or GT maps. VDAM maps/poses are not
PPCA initialization. Training exports must not accidentally retain GT pose
metadata used by a local pose grid or initializer.

Selected: about 20,000 particles, three balanced populations, box 64, known
CTF, unit contrast, inferred reconstruction noise, unknown random orientations
and small shifts. Use the simulator's **`noise_level=0.01`**, as explicitly
clarified by the user; do not reinterpret this as SNR or noise standard
deviation. Run one debugging initialization, then three independent
initializations on the same dataset. Multi-iteration runs belong on Slurm.

The located source bank is
[`~/mytigress/cryobench2/Ribosembly/vols/128_org`](/home/mg6942/mytigress/cryobench2/Ribosembly/vols/128_org).
It contains 16 MRC maps, all with 128-cubed dimensions, 3-Angstrom voxels and
zero header origins. The corresponding PDBs are in the sibling `pdbs/`
directory. This is a header inventory, not proof of structural registration or
of an easy, distinguishable triplet. Select three maps after checking their
common frame and differences; downsample to 64 with appropriate antialiasing
and preserve the physical field of view. Exact map IDs and hashes remain open.

The [simulator](https://github.com/ma-gilles/recovar/blob/162dc9477a64888beb53af757649f141034c3cbd/recovar/simulation/simulator.py) defines its initial
Fourier noise variance as
`get_noise_model(noise_model, grid_size) / 50000 * noise_level`, with subsequent
documented volume/image normalization. Its source-map normalization and selected
noise model also affect the measured SNR. The related fixture CLI's `--snr`
alias does not turn this parameter into a literal SNR. Record the exact noise
model, normalization, final noise spectrum and measured SNR in the fixture
receipt. Keep true noise, poses and labels available only to evaluation.

Unit contrast also requires disabling simulated contrast spread and avoiding
unaccounted per-particle rescaling during data export. Prefer a documented
common scale for the first fixture. Pin the exact particle counts per class
(20,000 is not divisible by three), map IDs, seeds, CTF/shift distribution and
normalization before launching; no fixture has been generated in this package.

Three common-frame maps have affine rank at most two, so a mean plus two
loadings can represent them. A three-point latent population is nevertheless
not Gaussian. Posterior shrinkage and model mismatch may affect reconstructed
class centroids even when the learned subspace can represent all three maps.

At the final PPCA model, compute the pose-marginal coordinate of every particle:

\[
\widehat z_i=\sum_\phi\gamma_{i\phi}m_{i\phi},\qquad
\overline z_c=\frac1{|I_c|}\sum_{i\in I_c}\widehat z_i,\qquad
\widehat V_c=\widehat\mu+\widehat W\overline z_c.
\]

`I_c` is defined by true class labels **only during evaluation**. Do not use
coordinates conditioned on true poses, coordinates at only the best pose, or
stale coordinates from earlier minibatches. GT-fitted least-squares coordinates
measure oracle representability and may be reported separately, never as the
primary recovery result. Jointly rotating W and the inferred coordinates leaves
the reconstructed states unchanged; no latent alignment is needed for this
formula.

For each state, report shellwise FSC, FSC-AUC and established resolution
summaries for PPCA versus GT, VDAM versus GT, and PPCA versus matched VDAM.
Match the K=3 permutation one-to-one with Hungarian assignment, using a stated
FSC-based matching score. Save masks and all registration transforms. Show
each state's result, including failed states, rather than only averages.

Per-state rigid registration is appropriate for map-quality comparison because
K-class maps can have independent frames. Also evaluate the entire PPCA model
under a single common physical registration to reveal inconsistent relative
geometry. Do not rotate individual loading columns independently or feed GT
registration back into training. Handle handedness consistently and record it.

Alongside map quality, record class counts, latent centroids/spread, loading
power, pose uncertainty, support mass, prior/data scales and objective traces.
Separately fitting GT maps to the learned subspace can help distinguish a
representation failure from an embedding failure. Quantitative PPCA acceptance
thresholds remain open; the first experiment characterizes feasibility. It
does not replace the reconciliation programme's K1/exactly-K4 qualification.

## 12. Decisions to discuss before writing the controller

| Order | Question | Current recommendation / reason |
| --- | --- | --- |
| 1 | What batch schedule? | The execution plan adopts VDAM's 200-to-2,000 schedule for 20k particles, with a final full-data update, as an explicit plan assumption. |
| 2 | How to map random VDAM seeds to PPCA? | Selected by the user: three initial random blob maps, their mean and two normalized contrasts. This is initialization only; no VDAM refinement. |
| 3 | Which numerical details of the accepted optimizer? | Coupled halfset blocks, shared loading adaptation and separate mean/loading gates are selected. Specify spectral floors, halfset scaling and zero-disagreement behavior; retain the gate's scientific follow-up flag. |
| 4 | Which observation and grid conventions? | Noise inference, known CTF, unit contrast and controlled coarse-to-fine search are selected. Pin Fourier weighting, masks, initial-noise normalization and exact grid/bandwidth milestones. |
| 5 | What exact three-state fixture and pass criteria? | 20k particles, box 64, Ribosembly source bank, simulator noise_level=0.01 and repeat policy are selected. Pin the three map IDs, noise model, seeds and thresholds; measure SNR without reinterpreting noise_level. |

No need to settle mixture models or high-dimensional optimizer tensors to
answer these first questions. Conversely, regularization, initialization scale and
pose score cannot remain implicit implementation choices: they change the
scientific experiment.

## 13. Checks once the design is selected

Before an ab-initio run, small independent float32 checks should establish:

1. Marginal scores and latent moments against dense Gaussian algebra, including
   determinant and Fourier weighting; `q=0` and `W=0` limits.
2. Coupled gradient agreement, the scalar limit of the proposed VDAM extension,
   and a unit-step/block-M-step identity only for the explicit-prior comparator
   with adaptive moments, shrinkage gates and postprocessing disabled.
3. Equivalent results under a nontrivial orthogonal latent rotation, including
   prior and any optimizer state; sign-only tests are insufficient.
4. Consistent statistic normalization across batch sizes, measured effects on
   disagreement/shrinkage, nonzero cross terms, correct residual covariance in
   a noise update, and quantified pose-support loss. For an explicit prior,
   additionally check that its declared strength does not change with batching.
5. Forward/adjoint and gridding conventions, separating exact operator checks
   from the approximate voxelwise reconstruction system.

These are proposed checks, not executed results. No numerical implementation,
science qualification or performance claim is attached to this document.

## 14. First implementation numerical boundary (September 22, 2026)

The new controller is [ppca_initial_model/iteration_loop.py](../../relax/ppca_initial_model/iteration_loop.py).
Implementation validation and recovery experiments remain distinct. The
[execution plan](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_implementation_plan.md) is still the
scientific contract; runnable code alone does not establish recovery.

- [noise.py](../../relax/ppca_initial_model/noise.py) uses coefficient
  variance `E|FFT(epsilon)|^2`. With the unnormalized image FFT, conversion from
  the initial RELION per-component spectrum is `2*N^4*sigma2`. Independent
  real Gaussian covariance and packed FFT tests include DC and Nyquist.
- [residual_statistics.py](../../relax/ppca_refinement/residual_statistics.py)
  forms the expected image residual before backprojection. The half-image
  adjoint supplies conjugate scatters. Applying RELION's additional x=0
  accumulator summation to this gradient doubled plane contributions and
  failed an off-grid directional derivative check; the direct gradient does
  not apply that operation. Existing refinement accumulator enforcement is
  preserved.
- The new full-real-observation mode uses the same exact spherical pixel
  support, including DC, for scoring and reconstruction. Raw image power is
  retained outside that support for the expected-noise update. Legacy PPCA
  observation defaults are unchanged.
- The new path explicitly uses full float32 matrix products (`highest` JAX
  matmul precision). Default GPU contractions failed block-size and gradient
  checks. Full float32 passed without FP64 arithmetic or changed tolerances.
- [update.py](../../relax/ppca_initial_model/update.py) converts VDAM's
  native BPref floor 1 to `0.5/N^4` in these coefficient-noise statistics.
  Directions are in raw-FFT model units, so the squared-direction epsilon is
  `N^4*1e-12`; the dimensionless adaptive denominator epsilon remains `1e-12`.
  No dataset/minibatch mass multiplier or additional volume prior is inserted.
- First-moment coverage flags are independent of latent coordinates. The
  loading second moment and shell gate each use one scalar for all loading
  columns. Exactly zero disagreement retains the specified zero-rho branch.
- Shift variance starts at VDAM's 100 square Angstroms and has its 2 square
  Angstrom minimum, converted to pixel-squared units. The PPCA Gaussian prior
  uses consistent pixel units; it does not reproduce the native parity path's
  documented mixed-unit offset arithmetic. Orientation probabilities reset
  uniformly on a grid change and otherwise use the same subset smoothing.
- The first controller evaluates the shared `LocalHypothesisLayout` rows with
  the extracted dense statistics routine. It preserves all fine candidates
  from the 0.999 coarse support and is deliberately unqualified for performance.
  The maximum requested radius 32 at box 64 is clipped to radius 31 under the
  existing unpaired-Nyquist convention, and the effective radius is recorded.
- Fine orientations are children of coarse orientations and do not depend on
  the image, so opt-in streamed rows share one fine rotation grid.
  [full_row_stream.py](../../relax/ppca_refinement/full_row_stream.py) scores,
  per image tile, the sorted union of the images' supported rows. It uploads
  the tile's coarse support once and forms each fine pose prior on the device
  as the coarse parent's support plus the rotation and translation log-priors,
  so a row outside an image's own support is `-inf` for that image and adds
  exact zeros. A fixed-capacity row table padded with a masked sentinel row
  keeps one block shape for every support pattern. Each rotation block is one
  jitted program per pass with no host round trip. The InitialModel controller
  now runs only the one-parent case, the full grid (oversampling 0; see the
  rejected oversampling note below); the stream keeps general coarse parents,
  which its unit tests exercise.
  The scores and moments are those of the host-mask dense routine, evaluated
  in another float32 order (below). Unit tests compare both routines against
  the independent local layout, and the general-rank test checks that the
  streamed engine is no further from a float64 evaluation than the host-mask
  routine.
  With one score and reconstruction window, the expected residual and its
  noise correction are linear in the block's M-step images
  `R = sum gamma alpha Y1` and `L = sum gamma G CTF^2/sigma^2`: the residual is
  `R_p - sum_q L_pq A_q` for projections `A`, and the correction is
  `sum_r [sum_pq L_pq Re(conj(A_p) A_q) - 2 sum_p Re(R_p conj(A_p))]`
  ([residual_statistics_from_moment_images](../../relax/ppca_refinement/residual_statistics.py)).
  The streamed engine uses this form instead of repeating the per-pose
  contractions; a float32 and float64 test checks it against the direct form.
  Performance record (September 25, 2026): the host-mask route spent about
  74% of each r16/HP3 16-image tile idle on host mask/prior construction and
  about 54k eager launches. One paired A100 CP113-to-114 update (Slurm job
  14423310, `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_speed_20260925/devres/qual/`)
  took 761 s with the frozen host-mask stream and 189 s with this engine at
  R512, passing every scoped PPCA gate (LHS relL2 1.3e-7, gradient 5.3e-7,
  direction 6.5e-7, exact selected IDs and noise). The tile is now mostly
  GPU-bound (adjoints and contractions); larger rotation blocks are a further
  measured config choice, not a default change.
- GEMM-shaped streamed engine (October 1, 2026). Pass 1 forms, per rotation
  block, every pose's inner products `Re<Y1_bt, A_rp>` (``t_mx`` and ``g_zx``)
  as one real GEMM of the `[Re; Im]` projections against the shifted images,
  and the CTF/noise-weighted products `Re(conj(A_i) A_j)` (``nu_mm``, ``h_zm``,
  ``H_zz``) as a second; the score uses the unrolled Cholesky factor `L` of
  `I + H_zz`: `-0.5 [nu_mm - 2 t_mx - |L^-1 b|^2 + log det(I + H_zz)]`.
  Pass 1 keeps the tile's scores, latent means `L^-T L^-1 b` and covariances
  `L^-T L^-1` on the device (component-major, the projector's and
  backprojector's layout), so pass 2 recomputes nothing: the posterior weights
  and kept moments form the M-step images with two more GEMMs. Every float32
  weight is used; on the live 10076 and 11-state updates essentially every
  (image, rotation) pair has a nonzero float32 weight, so skipping zero weights
  saves nothing. The streamed statistics carry no RHS volume (`rhs` is `None`):
  both optimizers read only the LHS metric and the direct residual gradient.
  Momentum SGD reads only the metric trace, so its streams backproject the
  trace channel alone (`TracePPCAStats`, `ppca_momentum_sgd.md`).
  On GPU the metric and residual images of a block are backprojected together
  into one voxel-major half volume of 32-float voxel rows
  ([ppca_moment_scatter_f32](../../relax/cuda/kernels.py)): each lane
  of a warp forms its own pixel's expected residual and noise correction from
  the block's moment images and projections, and expands the pixel's trilinear
  and Hermitian-partner targets, as RECOVAR's windowed adjoint does; then the
  warp adds one channel per lane, so each target costs one warp-wide atomic on one 128-byte
  row instead of one scattered atomic per channel. Every channel is RECOVAR's
  adjoint up to the float32 order of each voxel's sum; on one HP4 block both are
  3.0e-6 from a float64 adjoint. The 10076 block's backprojection drops from
  2.9 ms to 0.75 ms on A100. Other platforms use RECOVAR's adjoint per channel
  type and the XLA residual statistics.
  The controller streams both pseudo-halves through one prepared model,
  dispatching each tile before finishing the previous one, with one reused
  pose-kept buffer. Paired local A100 replays of the live checkpoints
  (`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_ppca_dense_speed_20261001/`):
  10076 q4/HP4 from about 42 s to 16 s per update and 11-state q10/HP3 from
  about 15.5 s to 6.3 s, every scoped PPCA gate passing; the two large GEMMs
  run at 88-94% of A100 float32 peak. 3xTF32 contractions were measured and
  rejected: no closer to float64 than float32 on the 11-state statistics.
- Fused GPU stages (October 2, 2026). On GPU streams the elementwise work
  around the four GEMMs runs in relax CUDA kernels
  ([ppca_stream.cuh](../../relax/cuda/ppca_stream.cuh)); the XLA formulation above
  stays the CPU path and the float64 reference of the tests. The window
  projector ([ppca_window_project_f32](../../relax/cuda/kernels.py)) evaluates
  all `P` components of a voxel-major model copy at once, at the score-window
  pixels only, into the planar `[Re | Im]` GEMM operand and the packed products
  `Re(conj(A_i) A_j)` of the Gram GEMM; it replaces RECOVAR's per-component
  projection loop, whose XLA loop predicate synchronized the host every block.
  The latent epilogue ([ppca_latent_epilogue_f32](../../relax/cuda/kernels.py))
  writes the scores, latent means and covariances in place into the kept
  buffers, and per (rotation, image) the maximum score over translations, the
  first maximizing translation and `sum_t exp(score - maximum)`; the tile
  normalization reduces these partials
  ([_normalize_partials](../../relax/ppca_refinement/full_row_stream.py)), with the
  same first-maximum order, instead of every kept score. The posterior
  preparation ([ppca_posterior_prep_f32](../../relax/cuda/kernels.py)) forms the
  weights `gamma [1, E z]` and the translation sums of `gamma E[a a^T]`; the
  embedding, rotation mass and latent trace follow from those sums. The GEMM
  window is padded with zero pixels to a multiple of four so every GEMM row is
  16-byte aligned. Posterior-weight diagnostics (rotation mass, top posterior)
  differ from the host-mask routine by a few 1e-6 because float32 scores carry
  rounding relative to their magnitude; the fused order is the closer of the two
  to float64. On A100 the four GEMMs are now about 84% of the busy device time
  of a 10076 update. Reading image tiles ahead on a worker thread (within an
  update, and the next update's first tile) was measured and dropped: no
  steady-state change in paired A100 updates. 16-byte vector atomics in the
  Hopper scatter were dropped too: 4% of the scatter, no update-level change.
  The four stream GEMMs (score, Gram, RHS and LHS images) take one precision
  setting (`Config.gemm_precision`, `--ppca-gemm-precision`, resolved by
  [resolve_gemm_precision](../../relax/ppca_refinement/full_row_stream.py)):
  `tf32` is one TF32 tensor-core pass with float32 accumulation, `fp32` the full
  float32 products (cuBLAS SIMT FFMA kernels), and `auto`, the default, is tf32 on
  GPUs of compute capability 8.0 or later and fp32 elsewhere (V100, CPU). The
  resolved value is in every update's log record and the stream diagnostics; the
  checkpoint configuration keeps the requested one. Precision is a runtime setting,
  not part of the checkpoint's configuration identity: a run resumes under any
  value (checkpoints from before the setting count as fp32), and the update
  records then carry `resumed_from_gemm_precision`. TF32 leaves
  the float32 equivalence tests (a bug check for the fp32 path) and was adopted on
  end-to-end science (October 2, 2026; H100 job 14880649, three selection seeds
  per arm, harness `relax_ppca_dense_speed_20261001/harness5`): per-state
  shared-frame FSC over shells 1-15 against GT, pose median, and latent
  between-state R^2 on the fixed 1100-particle eleven-state evaluation subset.

  | Eleven-state q10 arm, end of run | fp32 (seeds 101/102/103) | tf32 (seeds 101/102/103) |
  | --- | --- | --- |
  | VDAM from GT, 150 updates: state FSC | .787 / .828 / .772 | .768 / .833 / .777 |
  | same: worst state FSC | .618 / .639 / .609 | .589 / .653 / .610 |
  | same: pose median, fraction under 10 deg | 4.24 / 4.31 / 4.46 deg, .845-.854 | 4.22 / 4.26 / 4.44 deg, .850-.863 |
  | same: latent R^2 | .118 / .126 / .126 | .120 / .115 / .127 |
  | SGD from GT, 150 updates: state FSC | .967 / .967 / .967 | .967 / .967 / .967 |
  | same: pose median | 3.81 / 3.80 / 3.85 deg | 3.81 / 3.83 / 3.85 deg |
  | VDAM from CP4000, 300 updates: state FSC | .156 / .144 / .092 | .124 / .133 / .124 |
  | SGD from CP4000, 300 updates: state FSC | .244 / .246 / .246 | .244 / .246 / .246 |

  The GT start scores .983 state FSC, 3.8 deg pose median and latent R^2 .584; VDAM
  drifts from it under both precisions by the same amount (its stochastic step),
  SGD holds it. Same-seed fp32/tf32 final mean maps agree more closely (FSC .95,
  VDAM from GT; .9999 for 10076 VDAM) than fp32 seeds agree with each other (.79;
  .91), and the last-50-update log-likelihoods match to four significant figures.
  Per update on H100, tf32 is 1.89x faster for 10076 VDAM (4.81 to 2.54 s) and
  1.56-1.75x for the eleven-state arms; on A100 2.37x for 10076 VDAM (11.4 to 4.81 s).
- A per-shell separable VDAM metric was tried and rejected (October 2, 2026). It
  would cut the streamed scatter at `P = 11` from 88 to 23 channels: backproject
  only the metric trace `t(v)` (the momentum-SGD channel) and take the PxP
  structure per shell from the image-domain pair sums
  `S_s = sum_(r,b) sum_t gamma E[a a^T] sum_(f in s) CTF_b(f)^2 / sigma^2(f)`, so that
  `M(v) = t(v) S_s(v) / tr S_s(v)` needs one eigendecomposition per shell.
  The real minibatch metric is far from separable. Each voxel sees the few
  images whose central slices pass through it, so on a 300-image eleven-state
  batch the per-voxel unit-trace metric differs from its shell model by 60%
  (trace-weighted relative Frobenius error). A per-voxel diagonal with
  per-shell correlations still differs by 56%. The resulting direction
  differs from the full one by 35% (shells 0-4) to 120% (shells 25-29).
  From the GT start, with 150 VDAM updates and seeds 101/102/103 on H100
  (jobs 14896113 and 14896230), the full metric scored state FSC
  .769 / .826 / .768, pose median 4.2-4.4 deg and latent R^2 .12-.13. The
  separable metric scored .10 / .08 / .10, pose median 131-133 deg and
  R^2 .01 (after 50 updates it was already at .73 against .82). The
  update time fell only from 0.76 s to 0.60 s. Evidence: the harness
  `relax_ppca_dense_speed_20261001/harness5` (`metric_queue.py`,
  `metric_separability.py`). VDAM keeps the full per-voxel metric.
- Oversampling 1 on the streamed engine was measured and rejected (October 3,
  2026). In that path, a pass 1 over the coarse grid keeps each image's 0.999
  posterior mass, and pass 2 scores the 2x-finer children (HEALPix N+1, half
  the shift step) with `--stream-full-fine-rows`. A prototype ran that coarse
  pass on the streamed engine, with the same significant sets as the host-mask
  coarse pass. It still did not pay against the dense grid (median A100
  update, same start and selection seed):

  | Workload | Dense | os1, fine tiles of 16 / 32 / 64 / 150 images |
  | --- | --- | --- |
  | eleven-state GT CP4000, HP2 -> HP3 | 1.43 s | 3.37 / 2.65 / 2.67 / 5.2-7.8 s |
  | 10076 CP500, HP3 -> HP4 | 3.23 s | - / 9.02 s / - / out of memory |

  Three effects cancel the pruning:
  1. At 10076 the posterior has a heavy tail: pmax is about .44, but 0.999
     of the mass needs 68-87 thousand coarse poses per image.
  2. On the eleven-state data the per-image supports are small (19-160
     poses), but a tile scores the union of its images' rows. At 150 images
     that is about 70% of the grid, and smaller tiles lose to per-tile
     overhead.
  3. The oversampled shift grid makes each scored row cost 116 shifts
     instead of 29.

  For the current controller's schedule, the dense HP 3/4 grid stays the
  default. Evidence is in
  `relax_ppca_dense_speed_20261001/jobs/local_os1_smoke_20261002`.
  [Config](../../relax/ppca_initial_model/config.py) and the
  `relax ppca_initial_model` command therefore default to the dense stream:
  oversampling 0, `stream_coarse_recompute`, image batch 150 and rotation
  block 512, as in the live runs. They refuse oversampling > 0, together with
  its `--stream-full-fine-rows` and `--fine-devices` fine pass.
  `--no-stream-coarse-recompute` selects the host-mask dense engine (q <= 2),
  which is the stream's test reference.
  The image batch is an upper bound on the particles in a tile. A tile holds
  fewer when its image-proportional buffers would take more than 40% of the
  device memory still available after the stream's upload
  ([plan_tile_images](../../relax/ppca_refinement/full_row_stream.py)): the
  kept pass-1 buffers over the row table, every tilt frame's operands and one
  block's GEMM outputs. Each update records the planned size as `tile_images`.
  A subtomogram particle counts all of its tilts.
- The fine pose scores (blocked and factor-once) are assembled without the
  pose-invariant image energy: `-y_norm/2` is the same for every pose of an
  image (about `1e3` here) and cancels in every posterior, but in float32 it
  sets the rounding of the pose-dependent score, which sharp posteriors turn
  into weight differences. It is carried as `score_offset`
  ([pose_invariant_score_offset](../../relax/ppca_refinement/engine.py)) and
  added back only to absolute values (log-likelihood, reported top scores).
- Every PPCA M-step volume sum (RHS, LHS, residual gradient; the streamed
  engine forms no RHS) backprojects each rotation block into zero volumes and
  adds it to the running sum with Kahan compensation ([compensated_add](../../relax/ppca_refinement/engine.py)): the
  streamed engine, the host-mask dense accumulation and exact-local PPCA. A
  single float32 atomic accumulator rounds away posterior-tail contributions
  once voxels grow: against a float64 accumulation of the same block images,
  one CP110 full-row tile had RHS relL2 4.7e-4 uncompensated and 1.2e-7
  compensated, and one sharp o1_r16 tile had LHS 1.3e-4 and 1.4e-6. The
  RECOVAR adjoint kernel is unchanged; it accumulates one block into the
  volume it is given, and the block sum is the caller's. Tile merges and the
  bootstrap initialization add a few volumes of comparable magnitude and stay
  plain float32 sums. After this change lands, its head replaces the frozen
  contiguous-mask source as the control for PPCA scoped-gate pairs; pairs
  against the older uncompensated control measure that control's own float32
  accumulation error.
- Decision record (September 25, 2026), stable shapes: the last image tile of
  each half is not padded to 16 images. A new tile image count compiles the
  block programs once per process, measured at 1.7 s (O2x CP60, r16/HP2) and
  3.6 s (o1_r16 CP111, r16/HP3) per half; at most 16 counts occur per process
  (about 50 s per process, below 0.5% of a T200 run) and the persistent
  compilation cache carries across restarts. Revisit if tiles grow or
  compilation per count grows.
- After the final all-particle update,
  [compute_dense_ppca_embeddings](../../relax/ppca_refinement/dense_dataset.py)
  uses the same fine pose scores, latent means, candidate support and sequential
  posterior normalizer to compute the pose-marginal coordinates in Section 4.
  It omits M-step adjoints and expected-noise statistics, which the final
  embedding pass does not consume. The update iterations retain the full
  statistics path and its original reduction order.

The [fixture command](../../scripts/prepare_vdam_ppca_fixture.py) uses the
existing simulator's MRC/Fourier path, white `noise_level/50000` spectrum and
one recorded volume normalization. It omits the optional common image-power
rescale and all per-particle normalization. The standard simulator's zero
shift distribution is explicit. Source maps and truth stay in the evaluator
bundle; training manifests contain neutral poses required by the existing
loader, known CTFs and stable particle IDs. Simulator arithmetic retains its
existing float64 noise RNG; stored particles and the new production model are
float32/complex64. Simulation precision is not EM production precision.

The [training-only STAR bridge](../../scripts/prepare_vdam_ppca_star.py)
transcribes the fixture's hashed particle/CTF inputs for the independent K3
VDAM run. The [seed exporter](../../scripts/export_vdam_ppca_seed_maps.py)
inverts the q2 initial mean/loading basis at checkpoint zero and records the
three float32 RELION-frame maps. The VDAM override skips its native double
bootstrap when these maps are supplied; later projection and M-step backends
must still be selected explicitly for production float32 execution.
VDAM has one projector setup
([`reference_to_relion_projector_half_maps_and_power`](../../relax/relion/relion_projector_setup.py)): the device
FFT in double, narrowed to the complex64 slab RELION's GPU projector holds as a
float texture, with the tau2 shell power kept in double. It does not follow the
M-step dtype, and `--projector-setup-backend` is gone from InitialModel; a
float32 run therefore scores float32 textures built from a double setup, as
RELION does.
The earlier tiny K3 job 14293103 used float64 corrected projector preparation
despite float32 scoring, accumulation and M-step flags, so it is a mixed-precision
diagnostic and must not be counted as final float32 evidence.

The [pilot evaluator](../../scripts/evaluate_vdam_ppca_pilot.py) uses true
labels only after training: it averages final pose-marginal coordinates by
state, forms `mu + W @ z_bar`, fits one shared PPCA frame to the GT mean, and
uses a separate shared VDAM frame for Hungarian class matching. Per-state
rigid fits were originally labeled diagnostics. It reports raw and common-mask FSC
curves, AUC and resolution summaries without a post-hoc recovery threshold.
The current measurements and missing cells live in the
[pilot scorecard](vdam_ppca_pilot_scorecard_v1.md).

## 15. Small 2,000-particle exploratory pilot

The [small-pilot plan](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_small_pilot_plan.md) fixes a
subset of the existing simulator fixture before training. An opt-in
`stochastic_batch_size=200` in
[PPCA Config](../../relax/ppca_initial_model/config.py) replaces only the
non-final subset count; iteration 60 still uses every particle. The VDAM
comparison has opt-in fixed non-final subset and low-resolution Fourier and
angular caps in the PPCA-owned
[VdamPilotControls](../../relax/ppca_initial_model/vdam_controls.py), which the
VDAM controller holds as `pilot_controls` (`None` is native VDAM).
Native frequency and angular adaptation remains active within those caps, so
its effective per-iteration support must be reported rather than assumed equal
to the PPCA milestones. Defaults for either method are unchanged.

The [PPCA training loader](../../relax/commands/ppca_initial_model.py)
preserves the simulator particle sign when the manifest declares unit
contrast and `image_multiplier=1`. The subset MRC stack is stored in that
sign convention, and the K3 STAR loader consumes it without inversion.
Setting `uninvert_data=True` in PPCA changed every processed Fourier image
to its negative before scoring; this was corrected after the first small
pilot pair identified the exact first differing observation array. A loader
regression checks the processed image against the stored MRC pixels.

The [pilot evaluator](../../scripts/evaluate_vdam_ppca_pilot.py) retains its
full shellwise FSC and full-band AUC. When an active radius is declared, shared
and individual rigid fits use both maps low-passed to that radius; the active
FSC summary uses shells 1 through the radius after the same common low-pass
and mask for both methods. A curve that has not crossed a threshold by the
cutoff reports the cutoff resolution as censored, without claiming information
beyond the trained band.

### Updated shape-comparison policy, September 23

The user clarified that a unique common frame/hand across different states is
not a necessary definition of recovery. The
[visual comparison](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_visual_comparison.md) therefore
uses independent rigid rotation/translation/reflection for both methods and
matches K3 classes only after fitting all nine class/GT pairs. The
[comparison script](../../scripts/plot_vdam_ppca_comparison.py) preserves the
original shared-frame results as diagnostics and exports three projections and
three central slices of GT, unaligned and aligned maps. It does not alter the
training model or transform its basis. Similar-volume alignment can reduce
loading power, but the present shrinkage rule does not guarantee a unique
relative frame. Common-frame scores address that separate question; they are
not required for the exploratory per-state shape comparison.

### General-rank initialization

[`seed_maps_to_model`](../../relax/ppca_initial_model/initialization.py) uses K=q+1
independent data-only random bootstrap seed maps. The mean is their average,
and contrast j is (sum of the first j seed maps minus j times seed map j)
divided by sqrt(K j (j+1)), for j=1,...,q. These scaled Helmert rows satisfy
WW^T = (1/K) sum_k (v_k-mu)(v_k-mu)^T, so Gaussian latent prior covariance
retains the empirical seed-map covariance at arbitrary rank. The q=2 branch
keeps the previous reduction order exactly. No ground-truth label, map or pose
enters the random-angle round-robin bootstrap. Config q defaults to two;
changing rank does not change the optimizer schedule or observation model.

## 16. Subtomogram (cryo-ET) PPCA: design (October 2, 2026)

*Implemented October 2, 2026; first science in 16.9.* The aim is the pose-free PPCA initial model of
sections 4-15 for RELION 5 subtomograms (2D tilt stacks), with both optimizers,
on the streamed engine of section 14. Tomographic input, geometry, CTF and dose
follow the subtomogram Refine3D/VDAM path that relax qualifies against RELION
([tomo_particles.py](../../relax/refinement/tomo_particles.py),
[tomo_half.py](../../relax/refinement/tomo_half.py),
[tomo_input.py](../../relax/relion/tomo_input.py)). RECOVAR's known-pose
cryo-ET PPCA (shared latent and contrast per particle) is prior art for the
per-particle shared quantities only.

### 16.1 Observation model

Particle `i` has visible tilt images `k in V_i`, each with RELION's projection
matrix `Aproj_ik` (tilt-series rotation times the subtomogram matrix), its own
depth-corrected CTF with `rlnCtfScalefactor` and cumulative-dose damping
(RELION's exact CTF rows, `relion_ctf`), written `C_ik`. The particle has one
pose `phi = (R, t)` with `t` a 3D shift in pixels, one latent `z_i`, and unit
contrast (as SPA PPCA's first version). Tilt `k` is sliced at the rotation
`Aproj_ik R` (in RECOVAR's convention this is the matrix product of the two
RELION matrices; `_relion_mstep_rotations_from_eulers` with `left_matrices`
returns the same) and shifted by `s_ik(t) = [Aproj_ik t]_{1,2}`
(`tilt_image_shifts`):

\[
y_{ik}=A_{ik\phi}(\mu+Wz_i)+\epsilon_{ik},\qquad
A_{ik\phi}=C_{ik}\,S_{s_{ik}(t)}\,P(\mathrm{Aproj}_{ik}R),\qquad
z_i\sim\mathcal N(0,I_q).
\]

Noise is independent across tilt images, with spectrum `Sigma_g` in the
coefficient units of section 14 for the particle's optics group `g` (one per
tomogram on RELION 5 imports), as in RELION; dose enters through the CTF, not the
noise. Tiles hold particles of one tilt group and one noise group (16.4), and the
controller runs one stream per noise group, whose noise enters that stream's
operands; no engine change is needed. Until October 3, 2026 the first version
used one spectrum for every tilt image.

### 16.2 The latent is shared: sum the tilts before integrating it out

Stacking the visible tilts, `y_i = [y_ik]_k` and `A_{i phi} = [A_{ik phi}]_k`,
section 4 applies unchanged with block-diagonal `D_i`. With
`r_k = y_ik - A_ik mu` and `B_k = A_ik W`,

\[
M=I_q+\sum_{k\in V_i}\operatorname{Re}(B_k^*DB_k),\qquad
b=\sum_{k\in V_i}\operatorname{Re}(B_k^*Dr_k),\qquad
\ell_{i\phi}=-\tfrac12\Big\{\sum_k r_k^*Dr_k-b^TM^{-1}b+\log\det M\Big\}.
\]

The five inner-product statistics of the streamed engine (`t_mx`, `g_zx`,
`nu_mm`, `h_zm`, `H_zz`) are sums over tilts; the Cholesky factor, score,
latent mean `M^-1 b` and covariance `M^-1` are formed once per particle and
pose. Summing per-tilt PPCA scores instead would give each tilt its own `z`
(a different model with `|V_i|` log-determinants). As in SPA, the Gram terms do
not depend on `t`, since shifts are unimodular phases. The posterior over poses
is one softmax per particle; the rotation and 3D translation priors are
per particle.

### 16.3 Statistics, noise and optimizers

With the particle's `gamma`, `alpha` and `G` (section 5), every visible tilt
backprojects at its own rotation `Aproj_ik R`: the LHS metric is
`sum_i sum_phi gamma sum_k P^*(Aproj_ik R)[C_ik^2/Sigma] ⊗ G` and the direct
residual gradient is the adjoint of each tilt's expected residual image,
`gamma [alpha_a Y1_k - sum_b G_ab (C^2/Sigma) A_b]`, exactly the per-pose
moment images of section 14 with one image per tilt. These are the same
`AugmentedPPCAStats` / `TracePPCAStats` volumes, so VDAM's coupled update and
momentum SGD are used unchanged. Batches, pseudo-halves (stable particle id
parity) and the subset schedule count particles, not tilt images.

The noise update averages the expected residual power (with the posterior
covariance term of section 10) over the visible tilt images of each noise group:
each tilt image is one noise observation in its group's denominator, and a group
with no particle in a minibatch keeps its spectrum. The initial spectra use the
unaligned estimator over about 1,000 tilt images of each group (RELION's
per-group start-up count). The offset variance is
`sum gamma |t|^2 / (3 count)` in 3D, with the same floor and 0.9 subset
smoothing as SPA.

### 16.4 Mapping onto the streamed GEMMs

A **tilt group** is a set of particles whose images come from one fixed list of
frame matrices `{Aproj_f}_{f=1..K}`: the particles of one tomogram when they
share the subtomogram matrix (the simulator writes none, so every particle of a
tomogram does). Image tiles are drawn from one group. A particle's invisible
frames get zero operands, which add exact zeros to every GEMM and to the scatter.
Particles with individual subtomogram orientations form groups of one, which is
correct but shares no projections.

The tilt index joins the GEMM contraction axis, ordered `(frame, [Re | Im], pixel)`:

| Operand | SPA (section 14) | Subtomogram tile |
| --- | --- | --- |
| projections (window projector, `with_products`) | `(P R, 2F)`, rows `R_r` | `(P R, K 2F)`, rows `Aproj_f R_r`, frame minor |
| `Y1` | `(2F, B T)` | `(K 2F, B T)`: `[Re; Im](C_bf y_bf e^{-2 pi i x . s_bf(t)/N} / Sigma)`, zero if `f` not visible |
| `ctf2` | `(F, B)` | `(K F, B)`: `C_bf^2 / Sigma`, zero if not visible |
| pair products (Gram GEMM) | `(tri R, F)` | `(tri R, K F)` |
| pass 2 `rhs_parts`, `lhs_images` | `(P, R, 2F)`, `(tri, R, F)` | `(P, R K, 2F)`, `(tri, R K, F)` |
| moment scatter rotations | `R` | `R K` (`Aproj_f R_r`) |

The GEMM outputs keep their SPA shapes, `inner (P, R, B, T)` and
`gram (tri, R, B)`, so the latent epilogue, tile normalization and posterior
preparation are reused as they are. Pass 2 reshapes the same two GEMM outputs
to `R K` rows, and the moment scatter backprojects them with the expanded
rotations. A one-frame group with `Aproj = I` and a 2D shift grid is exactly the
SPA tile. Per-image 3D-to-2D phases replace SPA's shared translation table when
`Y1` is built (`tilt_translation_angles` geometry, one table per tilt image).

Cost per tile and pass is `2 P R (K 2F)(B T)` GEMM flops, `K` times an SPA tile
of the same `B T`, plus `R K` window projections shared by the tile's `B`
particles. Example: `K = 41`, radius 31 at box 64 (`2F` about 3000), HEALPix 3
(`R = 36864`), the 3D grid of radius 2 and step 1 px (`T = 33`): about `9e11`
flops per particle and pass, about `2e15` for a 1,000-particle all-data update
(seconds on an H100 in TF32). Early stages (radius 4-16, HEALPix 1-2) are
10-1000 times cheaper. Device memory per tile: `Y1` is `4 K 2F B T` bytes
(130 MB at `B = 8`) and the kept scores and moments `(1 + q) R B T` floats
(117 MB at `q = 2`).

### 16.5 Initialization and pose search

The `q + 1` random seed maps and their Helmert mapping (section 15) are reused.
The bootstrap draws random particles until about 1,000 tilt images are read,
gives each particle one random rotation and backprojects each visible tilt at
`Aproj_ik R` with its exact CTF, round-robin over the seed maps by particle.

The first version scores the full rotation grid of each stage with
`--oversampling 0 --stream-coarse-recompute` (the full-row stream with one
coarse parent). The oversampled path needs a coarse significance pass; the
SPA one runs the host-mask dense engine, which has no tilt axis. A coarse pass
through the same tilt stream is the follow-up, if the full grid is too slow.

### 16.6 First-version limits

Approved for the first version (October 2, 2026), each to be lifted separately:

1. Lifted October 3, 2026: one noise spectrum per optics group (`state.noise`
   is `(G, S)` for subtomogram particles).
2. Opt-in since October 3, 2026: a per-particle contrast point estimate (16.10).
   Unit contrast remains the default.
3. Full rotation grid per stage only (`--oversampling 0
   --stream-coarse-recompute`); no coarse significance pass.

### 16.7 Implementation and checks

- [full_row_stream.py](../../relax/ppca_refinement/full_row_stream.py): a tile may
  carry `K` frame matrices (`_TileArrays.frames`); projection and scatter
  rotations expand block rows by frames (`_frame_rotations`); operands and pass-2
  images flatten `(frame, pixel)`; the tile reader is a stream argument
  (`tile_loader`); the noise denominator counts observed images. Single-particle
  tiles keep their shapes and arithmetic.
- [tomo.py](../../relax/ppca_initial_model/tomo.py): `TiltParticles` (tilt groups
  from `TomoDataset`), `tilt_tiles`, the tilt tile reader `load_tilt_tile` (exact
  RELION CTF rows with dose, per-image phases of the 3D shifts) and
  `initialize_tilts`.
- `TomoDataset` records each image's frame and each particle's tomogram.
- [iteration_loop.py](../../relax/ppca_initial_model/iteration_loop.py) routes
  `TiltParticles` through these with RELION's 3D shift grid and a 3D offset
  variance; `relax ppca_initial_model --ios optimisation_set.star
  --particle-diameter D --oversampling 0 --stream-coarse-recompute` trains on
  the images, CTFs and tilt geometry only.

Checks ([test_tomo_ppca.py](../../tests/unit/ppca_initial_model/test_tomo_ppca.py)):
the tilt tile, and the controller's per-noise-group expectation (two groups with
different spectra), against a brute-force float64 joint Gaussian that stacks each
particle's tilts with one latent and evaluates `log N(y; A mu, A W W^T A^T +
D^-1)` from the dense covariance (box 8, `q = 2`, three particles, four frames,
one hidden tilt, five rotations, seven 3D shifts): log-likelihood, embeddings,
rotation mass, offset moment, LHS, residual gradient and noise sums agree to
4.0e-6 of their scale (three problem seeds; float32 engine, CPU and GPU). Summed
per-tilt marginals differ from it by more than 1e-3 relative; a hidden tilt equals
a zero tilt; one identity frame equals the single-particle tile; both optimizers
run through the controller. On the k3conf fixture (16.8) with the ground-truth
model, the stream puts the top pose at the GT rotation for 45 of 45 particles and
at the GT 3D offset (not its negative) for all 28 with offsets above 0.3 px, and
GT-pose embeddings separate the three states for every particle.

### 16.8 Science plan

Judged against ground truth with seed-to-seed spread as the noise band (there
is no RELION reference): per-state FSC against GT after rigid registration
(class-mean states `mu + W zbar_c` from final pose-marginal embeddings, section
11), state recovery from the embeddings, and pose error against GT after one
global alignment. Baseline: subtomogram VDAM K-class with K equal to the state
count (`relax initial_model --ios`). Fixture: `generate_relion5_tomo_dataset`,
one optics group per tomogram, unit contrast, total B 30-50 A^2 set once by the
preset, multiple states (the et15 5nrl path or a three-state path), with pixel
size and SNR chosen so that a true-pose reconstruction is not Nyquist-limited.
Small decisive runs first: a few hundred particles, a GT-initialized arm per
optimizer (it must hold GT), then random initializations.

### 16.9 First science on the k3conf fixture (October 2-3, 2026)

Fixture `em_fixtures/cryoet_ppca_k3conf_box64_20261002` (README there): 399
particles, three 5nrl states at 0/20/40 deg (133 each), four tomograms with one
optics group each, 41 tilts with 5% hidden, box 64 at 8.5 A, total B 40, unit
contrast, simulator `snr` 0.002. True-pose half maps cross FSC 0.143 at shells
20.7-22.9 of 32, so the fixture is not Nyquist-limited. Every arm: `q = 2`,
`--oversampling 0 --stream-coarse-recompute`, the default stage and batch schedule
(200 updates), 3D shifts within 2 px at 1 px, TF32 GEMMs, A100. GT arms start from
the GT state maps (mean and Helmert contrasts on the images' greyscale) at update
160 and run the last 40. The baseline is relax's subtomogram VDAM InitialModel,
`K = 3`, with the etbench command. Evaluation (truth used only here): state `c`
is `mu + W zbar_c` with `zbar_c` the mean final embedding of GT state `c`
(section 11); masked FSC-AUC after rigid registration of each state (both hands);
state specificity is the mean over states of `AUC(c, c) - mean AUC(c, other)` in
one common frame (the GT maps score 0.328); latent accuracy is the nearest-GT-centroid
and k-means (Hungarian) agreement of the embeddings; pose error is each particle's
angle from the chordal mean of `A_est^T A_gt`. Seeds 11/12/13 for the random arms.
Jobs 14897752 and 14900962 (PPCA), 14896308 (baseline);
evidence in `em_work/relax_ppca_cryoet_20261002/` (`HANDOFF.json`).

| Arm | State FSC-AUC (mean) | State specificity | Pose median, fraction < 10 deg | Latent nearest-centroid / k-means |
| --- | --- | --- | --- | --- |
| GT init, momentum SGD | .945 | .347 | 3.67 deg, 1.000 | .990 / .990 |
| GT init, VDAM | .750 | .211 | 3.65 deg, .997 | .977 / .975 |
| random init, VDAM (3 seeds) | .723 / .718 / .713 | .122 / .137 / .127 | 4.8 / 4.8 / 5.1 deg, .95 / .96 / .94 | .91 / .92 / .90 ; .90 / .61 / .90 |
| random init, momentum SGD (3 seeds) | .749 / .756 / .760 | .025 / .050 / .036 | 4.3 / 4.6 / 4.0 deg, .997 / .992 / .997 | .65 / .72 / .63 ; .51 / .72 / .48 |
| VDAM K-class, K = 3 (seed 1) | .134 / .388 / .110 (classes) | - | 27.0 deg, .11 | one class holds 99.5% |

Conclusions so far. The subtomogram likelihood is right: from GT, momentum SGD
holds the states (FSC-AUC .945, specificity at the GT value), poses at the HEALPix 3
spacing and a 99% separable latent. From random seed maps, both optimizers find
the poses (4-5 deg median against 27 deg for K-class VDAM, which collapses to one
class on this fixture). VDAM recovers about 40% of the GT state specificity and a
90% separable latent; momentum SGD's maps stay close to the mean (specificity
.03-.05). VDAM also moves away from GT at full resolution (loading power
overshoots, FSC falls near shell 20), as in the single-particle eleven-state
comparison (section 14). The baseline has one seed. Stock RELION 5.0.1 subtomogram VDAM K=3 (seed 1, non-MPI, one H100)
collapses on this fixture too: class populations .005/.972/.023 at iteration 133, resolution stuck
at 24.7 A (`em_work/cryoet_vdam_20261001/relion/cryoet_ppca_k3conf/vdam_k3_mpiscale/seed1/r1`), so
the collapse is the K-class algorithm on this fixture, not a relax difference. GT state FSC-AUC of the GT mean
map is .87-.90 unregistered, so map AUC alone does not show heterogeneity.

### 16.10 Per-particle contrast point estimate (opt-in)

`Config.contrast_estimate` (`--ppca-contrast-estimate`) gives particle `i` a
contrast `c_i` shared by its tilts, `y_ik = c_i A_ik(mu + W z_i) + epsilon_ik`, as a
fixed scale refit after every E-step outside the stream
([contrast.py](../../relax/ppca_initial_model/contrast.py)), in the manner of
RELION's scale corrections. For each particle of the minibatch, the model
`mu + W E[z_i]` is projected at its most probable pose: one projection per image,
at `Aproj_k R` and shift `[Aproj_k t]` for tilts. The least-squares scale of its
images under the scoring metric `D` (half-spectrum weights over the noise, inside
the stage's Fourier radius), with a Gaussian prior of sd `contrast_prior_sd`
(0.3) about 1, is

\[
c_i=\frac{\sum_k\operatorname{Re}\langle m_{ik},y_{ik}\rangle_D+s^{-2}}
{\sum_k\|m_{ik}\|_D^2+s^{-2}},
\]

clamped to `contrast_range` (0.5-2). The next E-step reads it through the
stream's per-image scale (`ScoringConfig.image_scale_corrections`), which
multiplies the CTF: score and reconstruction images by `c`, CTF^2 weights by `c^2`.
The residual statistics are then those of the scaled model. Single particles use
the same estimator and stream path. The estimate costs one projection per image
and no GEMM. Checks
([test_tomo_ppca.py](../../tests/unit/ppca_initial_model/test_tomo_ppca.py)): it
recovers known scales of noise-free tilt and single-particle images to 1e-4. The
stream with fixed per-particle scales agrees with the brute-force joint Gaussian of
scaled particles to the 2e-5 band, and the single-particle stream equals the
identity-frame tilt stream under the same scales.

A marginalized alternative was implemented and tested first: a contrast grid
whose values were extra tile images normalized jointly with the pose. It costs
`C` times the GEMM work. The two were compared on the contrast variant of the
fixture (`em_fixtures/cryoet_ppca_k3conf_contrast15_box64_20261003`, contrast sd
0.144). Each arm was GT-initialized momentum SGD for the last 40 updates (H100,
job 14912480, `em_work/relax_ppca_cryoet_20261002/science6_14912480`):

| Arm | Contrast vs truth (Pearson r) | Estimated sd | State FSC-AUC | Specificity | Pose median | Latent nearest-centroid | Seconds per update |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no contrast | - | - | .943 / .942 / .948 | .344 | 3.68 deg | .990 | 19.4 |
| grid 0.8 / 1 / 1.25 (posterior mean) | .754 | .125 | .944 / .940 / .948 | .338 | 3.70 deg | .992 | 25.1 |
| point estimate | .825 | .150 | .944 / .940 / .949 | .337 | 3.70 deg | .992 | 20.1 |

The point estimate recovers the contrast better than the three-value grid at
almost no cost, and the maps, poses and latent agree for all three arms. The grid
was removed. From the GT start the contrast does not change the reconstruction
on this fixture.

Random start, VDAM seed 11 on the same fixture, rebased code (A100, job 14917508,
batch 150):

| Arm | State FSC-AUC | Specificity | Pose median, < 10 deg | Latent nearest-centroid | Contrast r / sd |
| --- | --- | --- | --- | --- | --- |
| no contrast | .715 / .752 / .711 | .129 | 4.91 deg, .935 | .902 | - |
| point estimate | .721 / .745 / .734 | .127 | 4.60 deg, .962 | .862 | .743 / .248 |

With one seed the two arms are within the earlier three-seed spread (16.9). The
estimate tracks the contrast (r .74) but overstates its spread (sd .25 against
.144). The option stays off by default.

