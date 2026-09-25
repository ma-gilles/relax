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

Sources: [E-step](../../relax/vdam/sparse_pass2_estep.py),
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
[E-step](../../relax/vdam/sparse_pass2_estep.py),
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
- When every image retains every coarse orientation, opt-in streamed full rows
  share one fine rotation grid per image tile.
  [full_row_stream.py](../../relax/ppca_refinement/full_row_stream.py) uploads
  the tile's coarse support once and forms each fine pose prior on the device
  as the coarse parent's support plus the rotation and translation log-priors.
  Each rotation block is one jitted program per pass with no host round trip.
  Scores, moments, the float32 centered normalizer, the first-maximum top pose
  and block accumulation order are those of the host-mask dense routine.
  Unit tests compare both routines against the independent local layout.
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
The corrected JAX projector now accepts an explicit float32 compute dtype for
gridding correction, FFT and shell power. The VDAM driver passes its selected
M-step compute dtype through the per-iteration projector and E-step fallback.
Only the JAX projector setup follows it (`--projector-setup-backend jax`); the
native setup stays float64, so a float32 run with the native setup is a
mixed-precision diagnostic
([`_projector_setup_dtype`](../../relax/vdam/dense_adapter.py)).
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
