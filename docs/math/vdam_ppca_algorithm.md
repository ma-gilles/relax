# From VDAM to ab-initio PPCA: algorithm and scientific decisions

> Migration note (2026-09-23): implementation links point to RELAX. Historical pilot measurements below predate the RELAX port; they do not qualify the migrated source.

**Discussion draft, September 22, 2026. No new algorithm is implemented by
this document.** Statements marked *current* describe inspected source;
*derived* statements follow from the stated model; *proposed* choices remain
open for discussion. The aim is one PPCA model learned without supplied poses
or reference volumes, followed later by mixtures of PPCA models.

The user subsequently requested an executable handoff and accepted the random
seed-map construction below. See the implementation plan (`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_implementation_plan.md`; outside the repository)
for work packages, proposed numerical defaults and validation, and the
short execution handoff (`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_handoff.md`; outside the repository) for the next agent.
This document's derivations remain the scientific context; the plan explicitly
distinguishes user decisions from bounded implementation assumptions.

## 1. Scope and source identity

The starting description is the user's
VDAM algorithm document (`/scratch/gpfs/CRYOEM/gilleslab/mg6942/vdam_dev_20260919/recovar_vdam_cap/docs/math/vdam_algorithm.md`; outside the repository),
read at source commit `983edce72690a144b9fb26b43f10c28b3c18525d`.
Its SHA-256 is
`e834650772cf71387a901d2c1973fd74a343e25fc4592f6652bc6517087d7010`.
It is a useful description of a separate, advancing VDAM checkout, not a
guarantee about every source version or backend.

Here, *current* means the source in `codex/vdam-ppca` at
`ca440f5e4a84106b77613434ee9d7af8bf2b8b3e`, based on reconciliation
`f078ac1be64a21c2b478ef34b9f68db19920e23e`. Code links below refer to this
checkout. Recheck these claims when incorporating later reconciliation work.
The workstream and historical recovery (`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca.md`; outside the repository) records
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

The variance-prior notes (`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/math/ppca_variance_prior_notes.md`; outside the repository) distinguish physical
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
`~/mytigress/cryobench2/Ribosembly/vols/128_org` (`/home/mg6942/mytigress/cryobench2/Ribosembly/vols/128_org`; outside the repository).
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
execution plan (`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_implementation_plan.md`; outside the repository) is still the
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
  14423310, `em_fixtures/ppca_evidence_20261003/em_work/ppca_speed_20260925/devres/qual/paired_cp114_devres_14423310`)
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
  (`em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_dense_speed_20261001/jobs/local_final_20261001`):
  10076 q4/HP4 from about 42 s to 16 s per update and 11-state q10/HP3 from
  about 15.5 s to 6.3 s, every scoped PPCA gate passing; the two large GEMMs
  run at 88-94% of A100 float32 peak. 3xTF32 contractions were measured and
  rejected: no closer to float64 than float32 on the 11-state statistics
  (`em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_dense_speed_20261001/jobs/local_a100_20261001`).
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
  per arm, harness `em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_dense_speed_20261001/harness5`): per-state
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
  `em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_dense_speed_20261001/harness5` (`metric_queue.py`,
  `metric_separability.py`) and the runs under `jobs/slurm_sepmetric_gt_h100_20261002`
  and `jobs/slurm_sepmetric2_gt_h100_20261002` beside it. VDAM keeps the full per-voxel metric.
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
  `em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_dense_speed_20261001/jobs/local_os1_smoke_20261002`.
  [Config](../../relax/ppca_initial_model/config.py) and the
  `relax ppca_initial_model` command therefore default to the dense stream:
  oversampling 0, `stream_coarse_recompute`, image batch 150 and rotation
  block 512, as in the live runs. They refuse that tile-union fine pass
  (`--stream-full-fine-rows`, `--fine-devices`) and oversampling above 1;
  oversampling 1 is the per-image scheme described below, opt-in.
  `--no-stream-coarse-recompute` selects the host-mask dense engine (q <= 2),
  which is the stream's test reference.
  The image batch is an upper bound on the particles in a tile. A tile holds
  fewer when its counted device bytes exceed the memory still available after
  the stream's upload, less 10% of the device for fragmentation and
  uncounted temporaries
  ([plan_tile_images](../../relax/ppca_refinement/full_row_stream.py),
  `TILE_FRAGMENTATION_HEADROOM`). The count includes:
  - the kept pass-1 rows over the row table and the moment accumulators, from
    their shapes;
  - the larger of the score and moment block programs' own memory (temporaries
    plus outputs, less donated inputs), from XLA's compiled memory analysis of
    each program at the planned shapes on the device, cached per shape
    (`tile_program_bytes`);
  - the current tile's resident operands and the next tile's reader peak,
    because tiles are read ahead while the current one runs. The subtomogram
    reader reports that peak from the compiled memory analysis of its operand
    program (`load_tilt_tile.operand_bytes`) and its frames per particle
    (`max_frames`); the single-particle reader counts its arrays. A custom
    reader without both attributes is refused rather than planned as single
    particles.

  A finished tile is released before the next one is read, and a tile of
  another size releases the previous tile's kept buffer before allocating its
  own (two live buffers had run a dense HP4 eleven-state tile pair out of memory
  on an 80 GB card: 101 and 49 images). The tiles of one accumulate call run
  largest first, so the large kept buffer is allocated once on unfragmented
  memory; results return in the given order. That ordering is a precaution, not
  a proven cure: the allocation failure it answers (16 GiB for the compacted
  pass-2 means, update 95 of a dense HP4 eleven-state arm on an H100, Slurm
  14948876) happened once in three runs of that arm without the change and did
  not recur in reruns with or without it (14960561, 14961834). Tiles are read
  ahead only within one accumulate call, and the controller makes one call
  per noise group. When every call of an update holds one tile at the plan
  without read-ahead, that plan is used and the next tile's reader is not
  counted (`tiles_per_call`); otherwise it is. A stage is planned
  once, from its shapes, and the plan is logged ("PPCA tile plan", in the
  command's run.log). Each update records the planned size as `tile_images`.

  A plan counts bytes; the allocator hands out a block only from one
  contiguous region of its pool. Without preallocation
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, the setting of every test tier and
  benchmark) the pool grows by regions sized by the requests so far, so the
  small early stages left a pool of small regions in which the last stage's
  8.48 GiB tilt-reader block found no room, with 9.2 GiB in use of a 36.4 GiB
  pool (k3conf cryo-ET on an A100 40 GB, Polar 413542; the reader's count was
  right: 4.83 GiB counted against 4.87 GiB measured at 33 shifts). Each new
  plan therefore allocates and releases one block of its counted bytes, or of
  all the allocator may still take when that is less
  ([reserve_plan_region](../../relax/ppca_refinement/full_row_stream.py)), so
  the pool grows by one region of at least that size before the stage runs. A
  preallocated pool has nothing left to take, and the plans do not change
  (the same tile sizes and counted bytes at every stage with and without it).
  On an H100 with the allocator limited to the 40 GB card's pool (Slurm
  14993921, `em_work/relax_ppca_dense_speed_20261001/jobs/slurm_pool_region_emulation_a4ec984b`)
  main fails at the first radius-32 update and the fix completes the 24
  updates in 175 s; limited to a 16 GB card's pool it completes in 193 s, and
  on the full pool in 141 s with and without it. The full-pool runs with and
  without it differ as two runs of main do (5e-5 relative after one update in
  both pairs, Slurm 14995021): the engine is not bitwise repeatable.
  On an emulated 16 GB H100 (job 14939163,
  `em_fixtures/ppca_evidence_20261003/em_work/relax_gpuport_20261003/della_ppca_fb33c09`; r31/HP3 cryo-ET, 41 tilts, batch
  150) the r31 stage plans 33 particles (block programs 2.13 GiB) and SGD and
  VDAM complete at a 15.3-15.6 GiB nvidia-smi peak (the pool's reserved memory),
  where the count without the block programs had run out of memory in the score
  block. Live allocator peaks match the count: three pipelined r31 tiles of 139
  particles peak at 12.80 GiB against 12.94 GiB counted, and 33-particle tiles
  at 4.83 against 4.76 GiB (A100).
  A CPU stream is planned the same way, from the host memory this process may
  use (physical memory capped by the job's cgroup limit, less its resident set)
  and the compiled CPU programs. There the moment program's XLA adjoint holds one
  half volume per block image and channel (about 3.5 TB at box 128, 41 tilts and
  rotation block 512), so on either device a stage whose one-image tile does not
  fit is refused with the rotation block to shrink, before anything is allocated.
  On an 80 GB card the plan is the requested 150 particles at every k3conf and
  EMPIAR-10076 stage (`test_an_80_gb_card_keeps_the_requested_tile_at_every_stage`),
  and the controller's statistics do not depend on the tile size beyond float32
  reduction order (`test_controller_statistics_do_not_depend_on_the_tile_size`).
- Adaptive oversampling, order 1 (October 3, 2026; `--oversampling 1`, opt-in;
  oversampling 0 stays the default; since October 4 it applies to the last
  stage's updates only, see "Schedule" at the end of this item). This is RELION's two-pass
  scheme with per-image work, and replaces the tile-union design rejected above
  ([oversampled_stream](../../relax/ppca_refinement/oversampled_stream.py)).
  Pass 1 is the dense stream at the stage's order N over every rotation and
  shift. Each image's coarse (rotation, translation) samples are sorted by
  posterior weight. The largest are kept until their sum exceeds the adaptive
  fraction (0.999) of the image's mass, at most `max_significant` of them (RELION's
  `--maxsig`, 100 x classes for gradient runs, so 100 for PPCA), together with every
  weight at least the last one counted (`ml_optimiser.cpp:9602-9716`). Pass 2 scores
  the children of each kept sample only: 8 child rotations (HEALPix N+1 directions
  and half psi steps) times 4 child shifts in 2D or 8 in 3D (half steps), RELION's
  oversampled grid, each with its coarse parent's rotation and translation
  log-priors (RELION evaluates both at the coarse sample, `ml_optimiser.cpp:9434`).
  The posterior over those poses and the latent variable, and every statistic, are
  the dense stream's restricted to them; the rotation mass (for the direction
  prior) is summed into the order-N parents. Each kept sample is one job (image,
  coarse rotation, coarse translation). A tile's samples have image-major slots,
  `max_significant` per image; only the used ones are scored, in chunks that write
  their results at their slots, so every pass-2 program has one shape per tile size.
  Pass 2 then accumulates each image's significant fine samples only: the same rule
  over the child weights (the largest until their sum exceeds the adaptive fraction,
  and every weight at least the last one counted; RELION's `exp_significant_weight`
  in `storeWeightedSums`), with the image's normalization over every scored child.
  The M-step images are formed and backprojected per kept (image, child rotation)
  row; a child rotation none of whose shifts is significant for its image is not
  visited. Each update records the share of scored rows it accumulated
  (`accumulated_fine_row_fraction`: about 0.09 at the eleven-state GT checkpoint,
  0.22 from its random start, 0.61 at the EMPIAR-10076 update-500 checkpoint,
  whose posteriors are flat).
  The tests (`test_oversampled_stream.py`) compare the result, with every scored
  child accumulated (fine fraction 1), with the dense stream over the whole child
  grid, with each image's kept samples as its coarse support. They match within
  float32 reduction order, on CPU and on GPU, for single particles (metric and
  trace-only) and tilt series, also in padded tiles. With fraction 1 and no cap,
  every coarse sample is kept and the result is the dense child grid. The fine
  rule is checked against RELION's loop in float64 on the engine's job scores: the
  kept mass per coarse rotation, the embeddings and the accumulated rows.
  As in RELION, pass 1 runs on a smaller image window than pass 2: the coarse
  image size the coarse angular step resolves, `2 ceil(pixel ori_size /
  (step/360 pi diameter / 1.2))`, at most the stage's
  ([compute_coarse_image_size](../../relax/helpers/resolution.py), with RELION's
  clamp; 50 against 62 pixels at the eleven-state HP3 stage). A tile's images are
  read once, unshifted at the pass-2 window: the pass-1 operands are their pixels
  inside the pass-1 window times the coarse shifts' phase factors, and a job's
  child-shifted images are formed in its program from the same operands and the
  reader's phase factors (`shift_phases`), so a tile never holds its images at
  every child translation. The jobs per pass-2 program go through the planner
  (`plan_job_chunk`: at most 1024, halved until the job programs' compiled memory
  fits, as the tile size does), and at most as many as one launch of the CUDA
  projector and scatter takes (65535 rotations: jobs x 8 children x tilts, so 128
  jobs at 41 tilts). A tile runs whole chunks of the planned size, then of 1/4 and
  1/16 of it.
  Science against the dense grids (October 3-4, 2026; eleven-state fixture, 150
  updates from the GT checkpoint, one evaluator at HEALPix 4; state FSC / latent
  R^2 / GT power captured / seconds per update on an A100):
  dense HP3 0.957 / 0.728 / 0.908 / 1.45; dense HP4 0.965 / 0.764 / 0.927 / 7.29;
  oversampling 1 at HP3 with the default 2 px shift step 0.950 / 0.701 / 0.873 /
  0.74; with a 1 px shift step 0.964 / 0.756 / 0.919 / 1.26 (dense HP3 at that
  step: 0.958 / 0.727 / 0.909 / 2.60). The seed spread of the 2 px arms is 0.001
  in FSC and 0.003 in R^2 (three seeds; the 1 px and HP4 arms are one or two).
  The 2 px result is below dense HP3 because this fixture's shifts are all exactly
  zero: zero is a node of the dense grid, but RELION's child shifts sit a quarter
  step on either side of their parent (0.5 px here) and never on it, so every
  image is fitted 0.71 px off (the recorded offset variance is 0.25 px^2 per
  axis). Halving the step halves that offset and brings oversampling 1 to within
  0.001 FSC and 0.008 R^2 of dense HP4 at 0.17 of its time.
  On the continuous-shift twin of that fixture
  (`em_fixtures/ppca_elevenstate_contshift_box64_20261004`: Gaussian shifts, sd 2 px,
  truncated at 5 px; same maps, labels, rotations and noise) the order reverses.
  From the GT start with the true shift-prior variance, 150 updates, three seeds,
  evaluator at HEALPix 4 with a 1 px shift grid (state FSC / latent R^2 / GT power
  captured / pose median in degrees / share of poses within 10 degrees / seconds
  per update): dense HP3 0.931-0.933 / 0.559-0.565 / 0.878-0.882 / 2.86-2.94 /
  0.902-0.908 / 1.43; oversampling 1 at the default 2 px step 0.948-0.949 /
  0.594 / 0.909-0.912 / 2.81-2.83 / 0.910-0.912 / 0.92; the GT start itself 0.981 /
  0.570 / 0.999 / 2.75 / 0.934. Oversampling 1 is better than dense HP3 on every
  column on every seed, at 0.64 of its time. One seed each for reference: dense
  HP4 0.938 / 0.530 / 0.881 / 2.93 / 0.904 at 15.1 s per update (no pass-2 row is
  skipped under continuous shifts), oversampling 1 with a 1 px step 0.955 / 0.654 /
  0.924 / 2.84 / 0.916 at 1.35 s. The gate fixed
  beforehand (oversampling 1 inside the dense HP3 seed range widened by its width,
  or better, on each metric) passes. Two cautions from this fixture. The
  evaluator must resolve the shifts: with its 2 px inference grid the same
  checkpoints score 5.4-5.5 against 5.5-5.6 degrees and the GT start only R^2 0.25
  and 6.0 degrees. And a GT start made on the zero-shift fixture carries the floor
  of the shift-prior variance (0.056 px^2), under which the dense 2 px grid cannot
  leave zero shift (a node costs 36 nats) while oversampling's children, which
  take their parent's prior, can: that start must be given the fixture's variance
  (`harness10/make_contshift_gt_init.py`) or the comparison is decided by the prior
  (evidence: `jobs/local_gate2_contshift_v2_a100_277749fe`,
  `jobs/local_gate2_contshift_eval1px_a100_20261004` in the evidence tree below). The fine
  significance rule changes none of these scores: the engine before it gives
  0.950 / 0.701 / 0.873 per seed. On the cryo-ET k3conf fixture (default schedule,
  three seeds, H100) oversampling 1 reaches the dense HP4 last stage on state
  masked AUC and latent R^2 at the dense HP3 run time: from the random start
  0.761-0.765 and 0.79-0.84 against 0.751-0.757 and 0.77-0.80 (dense HP3
  0.711-0.725 and 0.62-0.67), pose medians 4.1-5.3 degrees against 3.4-4.2 (dense
  HP3 4.1-4.9); from the GT start 0.949 / 0.939 / 2.05 degrees against 0.948 /
  0.944 / 1.99 (dense HP3 0.945 / 0.896 / 2.38). Evidence, under
  `em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_dense_speed_20261001/jobs/`:
  `local_os1_science_e11_a100_dd709820`, `local_os1_science_refs_a100_20261003`,
  `local_os1_science_shiftstep_a100_20261003`, `slurm_os1_science_e11_h100_20261003`
  (14948876), `slurm_os1fine_science_e11_h100_20261003_40994eed` (14954551),
  `slurm_os1fine_science_tomo_h100_20261003_40994eed` (14954552); tables by
  `harness9/summarize_os1_science.py`.
  Tried and rejected for pass 1 (October 3, 2026): scoring it with the mean alone
  (the components zeroed, which would cut its GEMMs by the basis size). The
  samples that score selects hold, under the full model, 0.94 of an image's mass on
  average at the eleven-state GT checkpoint (5th percentile 0.43), 0.15 from the
  random start at update 4000 and 0.39 at the EMPIAR-10076 update-500 checkpoint,
  against 0.9985, 0.994 and 0.82 for the full model's own selection at the same
  cap; taking its 1000 largest samples instead reaches 0.30 and 0.62 in the last
  two. Once the components carry structure, the latent decides which poses matter
  (`harness9/pass1_meanonly_probe.py` and `jobs/local_meanonly_probe_dd709820` of
  the same evidence tree).
  Grouping jobs by coarse rotation to share projections was measured too: an
  update's jobs touch nearly as many coarse rotations as there are jobs (96% at
  the eleven-state checkpoint, 75% at EMPIAR-10076), so there is nothing to share.
  `--maxsig` caps the kept samples (100). At the EMPIAR-10076 update-500
  checkpoint, whose posteriors are flat, the cap leaves the kept samples 0.82 of
  an image's mass on average (38% of the images below 0.99): oversampling 1 at
  such a stage is a coarser approximation than its adaptive fraction says, and
  the cap warning below reports it. Each update records, in
  `iterations.jsonl` under `oversampling`, the two window sizes, the kept
  samples per image (median, mean), the share of images the cap stopped short of
  the fraction, the posterior mass those images hold (mean, 5th percentile,
  minimum) and the share of the batch both capped and below 0.99 of its mass.
  When that share exceeds 5%, the update logs a warning suggesting a larger
  `--maxsig`. On the eleven-state GT checkpoint (HP3, A100) the median image
  keeps 2 samples (mean 12); 7% are capped, holding 0.988 of their mass on average
  (5th percentile 0.951).
  The warning reports coverage, not a wrong result. On the continuous-shift
  fixture a third of the images (34-37%) are capped below 0.99 of their mass at
  the cap of 100 (capped images keep 0.96 on average) and oversampling 1 is still
  the better result against both dense grids (above). What the count distribution
  looks like everywhere: the median image needs 1 to 9 samples; a minority needs
  hundreds (kept mass at caps 100 / 400 / 1600: eleven-state GT 0.9988 / 0.9995 /
  0.9997; random start 0.9946 / 0.9985 / 0.9994; EMPIAR-10076 update 500 0.880 /
  0.900 / 0.922, where a quarter of the images want more than 16,000 samples;
  cryo-ET k3conf update 190 needs one sample per image).
  Tried and rejected (October 4, 2026): spending the cap as a per-tile budget (one
  common cap per tile, the largest whose total stays within 100 samples per image,
  up to 1600 per image), a fixed cap of 400, and a dense fallback for tiles with
  more than 5% of their images capped below 0.99. On the continuous-shift fixture
  (three seeds; 2 px evaluator) fixed 100, fixed 400, budget 100 and budget 50
  give the same state FSC (0.927-0.929), latent R^2 (0.28-0.295) and captured
  power (0.91) at 0.92, 1.58, 1.13 and 0.89 s per update; fixed 400 improves the
  pose median by about 0.05 degree. The fallback with that trigger took every
  tile there, returning dense HP3's (worse) result at 1.84 s, and took every tile
  on EMPIAR-10076 at updates 500 and 2550, costing dense HP3 plus pass 1 (0.76-0.78
  against 0.55-0.57 s). The fixed cap of 100 stays; the share of images below
  0.99 is not a trigger that separates the stages where oversampling 1 loses from
  those where it wins (evidence, same tree: `jobs/local_maxsig_curve_a100_277749fe`,
  `jobs/local_os1_science_maxsig400_a100_277749fe`,
  `jobs/local_budget_cap_a100_a23f3234`, `jobs/local_budget_cap_a100_1f7f8747`).
  Schedule (October 4, 2026). `--oversampling 1` runs the earlier stages dense and
  adaptive oversampling in the last stage only
  ([oversampled_update](../../relax/ppca_initial_model/iteration_loop.py): updates
  from the last stage's first one). Early models give flat posteriors, where the
  cap of 100 drops mass (cryo-ET k3conf kept mass of capped images at radius 4 /
  8 / 16: 0.55 / 0.84 / 0.95) and the dense pass is cheap. Tested from a
  consensus-mean start on the 20,000-particle continuous-shift fixtures (updates
  111-200, seeds 11 and 12, HP3 1 px evaluator; state FSC / latent R^2 / pose
  median in degrees, per seed):

  | Noise | Dense throughout | Oversampling 1 throughout | Dense, then oversampling 1 in the last stage |
  | --- | --- | --- | --- |
  | 0.25 | 0.923, 0.926 / 0.263, 0.278 / 4.09, 4.08 | 0.953, 0.954 / 0.373, 0.364 / 3.83, 3.89 | 0.954, 0.954 / 0.451, 0.436 / 3.90, 3.89 |
  | 1 | 0.822, 0.823 / 0.087, 0.085 / 5.49, 5.51 | 0.865, 0.865 / 0.267, 0.257 / 4.64, 4.61 | 0.868, 0.867 / 0.246, 0.258 / 4.58, 4.64 |

  A third arm, built and not shipped, chose per stage from the first update's
  capped mass (dense when more than 5% of a tile's images were capped and their
  mean kept mass was below 0.9). It chose oversampling at both stages here and
  matched "throughout", so its dense decisions have no scored comparison
  (`jobs/local_schedule_consensus_a100_f68b866a`).

  Per-state signal (note added October 5, 2026; re-score in section 17.11). At a fixed
  noise level the eleven states carry different power: relative to state 5, states 0-10
  hold .30 / .40 / .50 / .56 / .79 / 1.0 / .88 / .91 / 1.09 / 1.33 / 1.73, and states 0-1
  are also the least like the GT average (correlation .56 / .60, against .74-.88). At
  noise 1 their per-particle SNR is 2.5-6 times below the other states and they cannot be
  aligned: within-state pose medians of 93-120 degrees in every arm (dense and every
  oversampling arm, both seeds), against 4.0-5.4 at noise .25. This is a property of the
  fixture, not a defect. The GT maps have no pseudo-symmetry (self-correlation under
  90 and 180 degree turns at most .02), the wrong poses are spread over all angles and
  axes, and the latent places these particles at the low-mass end (states 0-3) whether
  or not their pose is right. Oversampling does matter at the margin: state 2 (power .50)
  fails dense (52-54 degrees) and aligns with oversampling 1 in every variant (5.9-7.2
  degrees). Analysis: `spa_state01_20261005/state01.py` in the run root of section 16.

  Cost, one H100 with nothing else on the device, arms back to back, second run
  of each arm (Slurm 14992520, `jobs/slurm_laststage_walls_h100_6b9ab85b`). Seed
  11, last stage = updates 161-199 at 2,000 images per update; seconds per update:

  | Arm | Whole run (s) | Last stage, mean | Last stage, tile shape compiled before (median, 16 updates) | Last stage, new tile shape (median, 23 updates) | Samples per image (mean / median) | Images capped below 0.99 |
  | --- | --- | --- | --- | --- | --- | --- |
  | noise 0.25 dense | 681 | 6.3 | 2.7 | 8.1 | | |
  | noise 0.25 last-stage oversampling | 795 | 9.3 | 2.1 | 13.4 | 2.0 / 1 | 0% |
  | noise 1 dense | 771 | 8.0 | 4.6 | 10.3 | | |
  | noise 1 last-stage oversampling | 971 | 13.6 | 3.5 | 20.0 | 54.5 / 26 | 34% |

  On the eleven-state continuous-shift GT start (60 updates of 300 images, one
  tile shape) dense HP3 takes 0.76 s per update and oversampling 1 0.56 s (59
  samples per image, 35% capped below 0.99). On EMPIAR-10499 at radius 32 (H100,
  1,847 particles per update, ppcaet) VDAM takes 83.3 s dense (one run, Slurm
  14957154) and 75.0-77.4 s with oversampling 1 (four runs, 14963878 and
  14974315; 5.6-6.0 samples per image, 0.45-0.8% capped below 0.99); momentum SGD
  78.3-83.8 s dense and 80.9 s with oversampling 1 (7.1-7.3 samples, 1.8-2.1%).

  Once a tile shape is compiled, oversampling 1 is the cheaper update in every
  row but SGD on 10499, where it is level. The single-particle runs are slower end
  to end because 23 of the 39 last-stage updates compile a new tile shape, in
  both arms: the half-set split leaves a remainder tile of a different size
  nearly every update (47 sizes in the stage), the single-particle reader does
  not pad tiles to fixed sizes as the tilt reader does, and a new shape costs
  the oversampled programs more to compile (13-20 s against 8-10 s; 25-38 s
  against the same on a cold compile cache).

  Single-particle tile padding (October 5, 2026). The single-particle reader now
  pads a tile with zero images to its planned size bucket
  ([_load_tile](../../relax/ppca_refinement/full_row_stream.py), `n_real` in the
  layout), as the tilt reader does. Padding images have zero CTF and take no
  posterior mass, and the padded size never exceeds the planned tile, so the
  memory plan is unchanged. Same fixtures and seed, one H100 (Slurm 15004428,
  `jobs/slurm_sp_padding_walls_h100_cd0d432f`), each source on its own empty
  compile cache, every arm run cold and then warm; updates 111-200, last stage
  161-200, seconds:

  | Arm | Whole run, before (cold / warm) | Whole run, padded (cold / warm) | Last stage, before (warm) | Last stage, padded (warm) | Updates compiling a new tile shape, before / padded (whole run; last stage) |
  | --- | --- | --- | --- | --- | --- |
  | noise 0.25 dense | 623 / 620 | 324 / 321 | 250 | 141 | 67 / 5; 18 / 0 |
  | noise 0.25 last-stage oversampling | 989 / 767 | 342 / 311 | 397 | 136 | 73 / 7; 24 / 2 |
  | noise 1 dense | 703 / 702 | 412 / 412 | 339 | 244 | 67 / 5; 18 / 0 |
  | noise 1 last-stage oversampling | 1299 / 966 | 386 / 368 | 602 | 200 | 73 / 7; 24 / 2 |

  The persistent cache holds 17 programs after the padded runs against 242
  before. The eleven-state GT start already ran one full tile shape and is
  unchanged (dense HP3 0.74 s, oversampling 1 0.53 s per update). Final
  checkpoints differ from the unpadded ones by no more than the unpadded
  cold-against-warm repeat does (relative L2 of the loadings, second moments and
  noise, all six arms; `harness10/sp_padding_table.py` in the run root). One
  difference is unexplained and not chased: the padded oversampling 1 GT start
  differs between its cold and warm runs by 2.6e-3 in the loadings, against
  1.4e-4 for padded against unpadded (both cold) and 1.3e-4 for the unpadded
  repeat. With
  padding, last-stage oversampling costs the same as dense end to end or less.
  The allocator limited to a 16 GB card's pool and to a 40 GB card's
  (Slurm 15004433) runs both single-particle stages with full tiles and the
  k3conf cryo-ET cell. The k3conf cell's out-of-memory failure on a real P100
  (Polar 413550, a 2.13 GiB block-program allocation at radius 32) was pool
  fragmentation under `XLA_PYTHON_CLIENT_PREALLOCATE=false`, not a miscount: the
  pool had grown to its limit in regions of 1, 4 and 8.92 GiB, and 9.65 GiB was
  in use when the request failed. On an A100 limited to the same pool the
  identical plans run with preallocation off and on, and the counted 12.33 GiB
  lies 3% above the measured peak of 11.94-11.96 GiB
  (`jobs/local_p100pool_a100_20261005`).
- Pass-2 row skip (October 3, 2026; default floor 1e-10, `--ppca-pass2-mass-floor`).
  After pass 1, the stream reads each pose row's largest per-image posterior mass
  in the tile from the epilogue partials
  ([_pass2_rows](../../relax/ppca_refinement/full_row_stream.py)). When at most half
  the rows reach the floor, pass 2 (weights, M-step GEMMs, projections and moment
  scatter) runs on those rows only. They are gathered into a power-of-two block
  buffer, and their rotation masses return to the pass-1 positions. Each image then
  loses less than `rows x floor` of its posterior mass. Otherwise pass 2 visits every
  row in place and nothing is dropped. Each update records the floor and
  `pass2_row_fraction`.
  The 50% threshold exists because compacting perturbs a trajectory even when it
  keeps nearly every row. A snapshot that compacted at any kept share (H100 job
  14919588, `em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_dense_speed_20261001/jobs/slurm_rowskip_gt_h100_20261003`; GT-started eleven-state VDAM, 150 updates, seeds 101/102/103) visited
  97-99.9% of rows and moved state FSC from .771 / .832 / .766 to
  .767 / .825 / .748 and latent R^2 from .118 / .133 / .131 to .111 / .123 / .132.
  With the threshold, these SPA tiles run in place and their output equals no
  skip (10076 and eleven-state walls and logL unchanged).
  Late cryo-ET posteriors are sharp and tile-coherent. From a random-start k3conf
  checkpoint at r31/HP3 (pmax 1, batch 150), pass 2 visits 0.27% of the rows and an
  A100 update falls from 9.5-10.1 s to 4.4-4.7 s; logL agrees to seven significant
  figures. Science (H100 job 14919848, floor 0 / 1e-10):
  - GT-started SGD: FSC-AUC .9446 / .9446, pose median 3.67 / 3.67 deg.
  - Random-start VDAM: FSC-AUC .7250 / .7246, specificity .123 / .120,
    latent R^2 .659 / .654, pose median 4.95 / 4.98 deg.
  The random-start run's wall fell from 1064 s to 952 s, all of it in the HP3
  stages.
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

The small-pilot plan (`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_small_pilot_plan.md`; outside the repository) fixes a
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
visual comparison (`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_ppca_20260922/docs/development/vdam_ppca_visual_comparison.md`; outside the repository) therefore
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
2. Unit contrast. A per-particle contrast estimate was implemented, tested and
   removed on October 3, 2026 (16.10).
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
evidence in `em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_cryoet_20261002/` (`HANDOFF.json`;
the curated copy of the run root, with a README and sha256 manifest).

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
at 24.7 A (`em_fixtures/ppca_evidence_20261003/em_work/cryoet_vdam_20261001/relion/cryoet_ppca_k3conf/vdam_k3_mpiscale/seed1/r1`), so
the collapse is the K-class algorithm on this fixture, not a relax difference. GT state FSC-AUC of the GT mean
map is .87-.90 unregistered, so map AUC alone does not show heterogeneity.

Baseline on the current defaults (October 3, 2026). Since the runs above, the
streamed engine gained memory-planned tiles of 150 particles with rotation blocks of
512, padded tile sizes (at most five compiled shapes per stage), a pass-2 skip of
pose rows without posterior mass (floor 1e-10), one fused operand program per tilt
tile and the batch-share VDAM step (section 17; its factor is one on all 200 updates
here). Random-start VDAM, seeds 11/12/13, relax main `ab91c0f`, every default, one
H100 per seed, cold compilation (no compile cache; job 14937827,
`em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_cryoet_20261002/baseline_14937827`):

| Random-start VDAM | State FSC-AUC (mean) | State specificity | Pose median, fraction < 10 deg | Latent nearest-centroid / k-means |
| --- | --- | --- | --- | --- |
| earlier code (table above) | .723 / .718 / .713 | .122 / .137 / .127 | 4.8 / 4.8 / 5.1 deg, .95 / .96 / .94 | .91 / .92 / .90 ; .90 / .61 / .90 |
| main `ab91c0f` defaults | .724 / .725 / .711 | .122 / .145 / .112 | 5.0 / 4.6 / 5.3 deg, .95 / .97 / .91 | .92 / .94 / .91 ; .92 / .92 / .92 |

The three-seed ranges overlap on every metric (mean FSC-AUC .720 against .718,
specificity .126 against .129), so the speed work moved no science. The seeds
follow different trajectories under the new code, so seed-by-seed differences are
run-to-run variation. Training wall per seed, in seconds per stage (r4/HP1, r8/HP2,
r16/HP3, r31/HP3; 60/50/50/40 updates):

| Code | r4/HP1 | r8/HP2 | r16/HP3 | r31/HP3 | Training | With final pose pass |
| --- | --- | --- | --- | --- | --- | --- |
| before the planner, row skip and padding (seed 11, H100, job 14919848) | 474 | 315 | 91 | 184 | 1064 | 1091 |
| main `ab91c0f` (seeds 11/12/13) | 51-53 | 36-42 | 55 | 102 | 245-252 | 273-280 |

The run is 4.2 times faster. No stage dominates; the full-resolution stage
(r31/HP3) takes 41% of the training wall, so a coarse significance pass for the
tilt stream (limit 3 in 16.6) is not needed on this fixture.

### 16.10 Per-particle contrast: tested and removed

Two per-particle contrast models were implemented and tested in October 2026.
Both gave particle `i` a contrast `c_i` shared by its tilts,
`y_ik = c_i A_ik(mu + W z_i) + epsilon_ik`. Neither improved the maps or the
latent, so unit contrast remains the model and the code was removed.

- A marginalized grid: the contrast values were extra tile images, normalized
  jointly with the pose. It costs `C` times the GEMM work.
- A point estimate refit after every E-step outside the stream, in the manner of
  RELION's scale corrections. It is the least-squares scale of the particle's
  images against `mu + W E[z_i]` at its most probable pose, under the scoring
  metric. At first it used a fixed Gaussian prior about 1; later it used the
  posterior mean under a prior whose variance was estimated from the particles
  (`var(a/b) - mean(1/b)` for the per-particle numerators `a` and denominators
  `b`). The stream then used it as a fixed per-image scale of the CTF.

On the contrast-sd .144 variant of the fixture
(`em_fixtures/cryoet_ppca_k3conf_contrast15_box64_20261003`), from the GT start
(momentum SGD, last 40 updates, job 14912480), the following results were
measured. Values in the AUC column are state FSC-AUC.

| Arm | Contrast r | Estimated sd | AUC | Latent nearest-centroid |
| --- | --- | --- | --- | --- |
| unit contrast | - | - | .943 / .942 / .948 | .990 |
| grid 0.8 / 1 / 1.25 | .754 | .125 | .944 / .940 / .948 | .992 |
| point estimate | .825 | .150 | .944 / .940 / .949 | .992 |

The grid was dropped for the cheaper and more accurate point estimate. The
decisive test used a larger spread: the contrast-sd .29 variant
(`em_fixtures/cryoet_ppca_k3conf_contrast30_box64_20261003`, kept as a stress
case). The point estimate was shrunk with the estimated prior variance (job
14920680, H100, one seed per arm). The criterion for keeping it was a gain larger
than the earlier three-seed half-range (mean state FSC-AUC .005, latent
nearest-centroid .0115) on one metric, with no larger loss elsewhere.

| Arm | Mean AUC | Latent nearest-centroid | Pose median, < 10 deg | Contrast r / sd |
| --- | --- | --- | --- | --- |
| GT start, unit contrast | .9470 | .980 | 3.66 deg, .987 | - |
| GT start, estimate | .9496 | .983 | 3.68 deg, .987 | .945 / .258 |
| random VDAM s11, unit, update 190 | .7226 | .875 | 5.7 deg, .875 | - |
| random VDAM s11, estimate, update 190 | .6755 | .752 | 24.4 deg, 0 | .887 / .415 |

From the GT start, the estimate recovers the contrast (true sd .288) but ties on
every metric. From a random start, it loses .047 AUC and .123 latent accuracy,
and its poses do not converge. The pose error is the spread after the best global
rotation has been removed. Before the poses settle, a misaligned particle matches
the model poorly and receives a low contrast. The estimated spread is therefore
inflated (.415), and the down-weighted particles cannot correct their poses: the
estimate feeds back on pose error. The unit-contrast random arm stopped at update
199 with an indefinite metric. That failure is the r31 high-shell loading
overshoot of VDAM (section 17), which also occurs on the fixture without contrast
spread; it was not caused by the unmodelled contrast.

A future contrast model would need to start only after the poses settle. It must
also be judged from random starts, not from the GT start.

### 16.11 Real data: EMPIAR-10499 (October 3-4, 2026)

Data. EMPIAR-10499, *M. pneumoniae* 70S ribosomes in situ: 18,466 particles from
64 tilt series, about 41 tilts each, box 128 at 3.906 A (fixture
`em_fixtures/empiar_10499_relion5`; RELION 5 Refine3D of it reaches 7.94 A). There
is no ground truth. The references are RELION's Refine3D poses and map, the
RECOVAR cryo-ET analysis of the same particles at fixed Warp/M poses (mean,
eigenvolumes and embeddings), and the 1,811 particles that a cryoDRGN-ET filtering
removed (junk and free 50S together). The fixture's particle rows are sorted by
tilt series and differ from the `M_particles.star` order of those references for
35% of the particles; scores map them by particle name.

Geometry. Warp/M refined each particle's tilt geometry, so every particle is its
own tilt group and tiles would hold one particle. The runs use a regrouped copy of
the project in which each particle's per-tilt rotation is replaced by its tilt
series' mean rotation for that tilt (64 groups; residual angle median 1.91 deg, p90
4.99 deg, growing with tilt). With poses fixed, the mean map is the same under
both geometries at 31 A (masked FSC 1.000 on every shell).

Runs. `q = 4`, random start, VDAM, `--oversampling 0 --stream-coarse-recompute`,
the default stages (radius 4/8/16/32 at HEALPix 1/2/3/3; 200 updates, of about
200 particles up to radius 16 and of 1,847, a tenth of the data, at radius 32, the
last over all particles; radius 32 is 15.6 A), 3D shifts within 2 px at 1 px, particle diameter
300 A. A pilot (seed 11) stopped at update 165 on the metric check: one frequency
row outside the stage radius had a near rank-1 metric at float32 roundoff. The
coupled direction is now solved and checked on the update's support only (16ca4e5,
40cf41d). The pilot continued from update 160 at 40cf41d; seeds 11, 12 and 13 then
ran fresh at 40cf41d. Scores, after one rigid registration of the mean map to
RELION's (both hands tried): the angle of each particle's top pose from RELION's
after the best global rotation; the masked FSC against RELION's map (mask
`ET_ribo_mask.mrc` with a 2 px Gaussian edge) of the model mean and of the
population average `mu + W zbar`, `zbar` the mean posterior latent; the
cross-validated AUC of the removed set from the latent (logistic regression; the
RECOVAR embedding with four dimensions scores .81); the canonical correlations of
the latent with the RECOVAR embedding and between seeds.

| Run | Pose median, fraction < 10 deg | Masked FSC, model mean, shells 8 / 16 / 24 / 28 | Masked FSC, `mu + W zbar` | Loading / mean power | Removed-set AUC | Canonical correlations with RECOVAR |
| --- | --- | --- | --- | --- | --- | --- |
| pilot, seed 11 | 4.07 deg, .961 | .991 / .986 / .939 / .904 | .989 / .962 / .902 / .854 | .22 | .787 | .88 .49 .07 .02 |
| seed 11 | 4.09 deg, .955 | .958 / .958 / .912 / .864 | .990 / .962 / .902 / .849 | .39 | .787 | .87 .55 .04 .01 |
| seed 12 | 3.96 deg, .963 | .988 / .989 / .966 / .937 | .990 / .968 / .925 / .889 | .11 | .796 | .88 .42 .02 .00 |
| seed 13 | 3.94 deg, .963 | .992 / .989 / .969 / .942 | .989 / .973 / .937 / .894 | .11 | .798 | .90 .37 .01 .00 |

Shell `s` is 500 A / `s`. Every run has RELION's hand. No mean map crosses FSC 0.5
inside the band, so the maps are limited by the 15.6 A band. The RECOVAR mean at
fixed poses scores .997-1.000 masked on every shell.

Seed agreement, against thresholds fixed before the three seeds ran: the pose
median stays within 0.5 deg across seeds (range 0.14); the removed-set AUC within
.02 (range .012); the first canonical correlation between two seeds' latents is
above .8 (.868, .869, .890). All three hold. The second canonical correlation
between seeds is .38, .28 and .23, and the third and fourth are below .06: at
`q = 4` one latent direction is reproducible on this dataset. It is the direction
shared with the RECOVAR embedding, and it separates the removed set. Top poses of
two seeds differ by 5.2-5.3 deg (median, after the best global rotation).

Open: the posterior latents are not zero-mean. The mean posterior latent is up to
1.7 prior standard deviations from zero (seed 12, first coordinate), so part of
the average map sits in the loadings and the model mean is not the population
average. At low resolution the population average agrees with RELION in every run
(.989-.998 masked at shells 4-8) where the model mean spreads from .958 to .992 at
shell 8. Beyond 30 A the loadings are noisier than the mean, and the model mean is
the better map; there the seeds differ (masked FSC .912 to .969 at 21 A), and the
run with the most loading power has the worst mean map. The pilot and the fresh
seed 11 use the same minibatches but differ from the first update (the
support-only solve changes the edge-shell rows), so they count as two draws.

Poses or data model. At update 160 (radius 16) the pilot's mean scored .972 masked
on shells 8-12. For 2,000 random particles scored at RELION's pose only, the mean
M-step of the same model gives .992; `relion_reconstruct` of the same particles
against RELION's full map gives .996. Pose error on the 7.5 deg grid is therefore
the larger part. A smaller difference remains with poses fixed: against
`relion_reconstruct` of the same particles at the same poses, the fixed-pose mean
differs by a smooth field at shells 0-4 (beyond 125 A), mostly outside the particle
(correlation .99 inside 30 px of the box centre, .72 at 40-50 px, where two RELION
reconstructions agree at .99). It is not the padding (RELION at padding 1 and 2
agree), the per-series noise weighting, the regrouped geometry, the pose source or
the image loading, and the fitted loadings explain it no better than chance.

Identified (October 5, 2026): two terms, neither a defect of the tilt-stream
statistics. Same 200 random particles, Warp's poses, CPU, `relion_reconstruct` used
only as a debugging oracle (`empiar10499/residual_terms_20261005/` in the run root of
this section; per-shell power ratio normalised to shells 1-3, shell k is 500/k A).

| Comparison | Power ratio, shells 5 / 10 / 15 | Unmasked FSC, shells 1 / 2 | Low-pass (shells 1-4) correlation at 40-50 / 50-64 px |
| --- | --- | --- | --- |
| PPCA (RELION 5 dose damping) against RELION (SPA star, Warp's `rlnCtfBfactor`) | 1.14 / 1.54 / 2.35 | .955 / .968 | .73 / .65 |
| PPCA against RELION, no dose damping in either | 1.04 / 1.04 / 1.07 | .955 / .968 | .73 / .65 |
| RELION, Warp's B against no B | .99 / .95 / .88 | 1.000 / 1.000 | 1.00 / 1.00 |
| PPCA mean, 2 / 4 / 8 sweeps of the same statistics, against RELION (no damping) | 1.12 / 1.20 / 1.28; 1.20 / 1.53 / 1.74; 1.29 / 1.96 / 2.38 | .968 / .978; .979 / .985; .977 / .981 | .84 / .75; .93 / .85; .97 / .89 |
| RELION, padding 1 or nearest neighbour against padding 2 (no damping) | | .978-1.000 | .99-1.00 / .97-1.00 |

1. The high-shell amplitude difference is the oracle's damping. The SPA STAR carries
Warp's per-tilt `rlnCtfBfactor` (-6.52 - 4 x dose), and RELION's CTF envelope is
`exp(-B k^2 / 4)` (`src/ctf.h`), so these negative values amplify the high-dose tilts
instead of damping them. relax damps by dose as `relion_refine` does for tomo images
(`relion_tomo_damping`). With damping removed from both, the power ratio is .92-1.07
to shell 15.
2. The low-frequency field outside the particle is the diagnostic itself. The
fixed-pose mean is one diagonal step `m0 + g / a` from almost zero. Near the origin the
projection normal matrix is far from diagonal, so one step does not reach the
least-squares map. Repeating the step with the same statistics moves the low shells
toward RELION's map (correlation at 50-64 px .65 to .89 in 8 sweeps; DC power ratio
2.7 to 1.1). Unregularised, the higher shells then gain noise power (unmasked FSC
falls, masked stays at .996-.997). Training updates the mean every step, so its maps
do not carry this one-step error. The pose-error reading above stands.

Oversampling 1 in the last stage. Each run's update-160 checkpoint continued
through radius 32 with `--oversampling 1` (277749f; oversampling 0 gives the same
numbers at 277749f and 40cf41d). Both models of a run were scored with a dense
HEALPix 4 pose pass over the same 4,000 random particles. Rule fixed before the
three fresh seeds ran: oversampling 1 is better on the pose median and on the masked
FSC of the model mean at shells 24 and 28 in every seed.

| Run | Pose median, oversampling 0 / 1 | Masked FSC shells 24 / 28, model mean, 0 | same, 1 | `mu + W zbar`, 0 | `mu + W zbar`, 1 |
| --- | --- | --- | --- | --- | --- |
| pilot, seed 11 | 3.32 / 2.85 deg | .939 / .904 | .960 / .929 | .902 / .854 | .948 / .921 |
| seed 11 | 3.35 / 2.95 deg | .912 / .864 | .918 / .879 | .902 / .849 | .969 / .950 |
| seed 12 | 3.16 / 2.67 deg | .966 / .937 | .968 / .945 | .925 / .889 | .962 / .943 |
| seed 13 | 3.20 / 2.67 deg | .969 / .942 | .974 / .956 | .937 / .894 | .953 / .926 |

The rule holds in the three fresh seeds (the pilot was the trial that suggested
it). The pose median improves by 0.4-0.5 deg; the model mean by .002-.006 at shell
24 and .008-.015 at shell 28; the population average by .02-.07 and .03-.10.
The fraction of poses beyond 10 deg does not change (.03-.05). The radius 32 update
takes 76-77 s against 83 s (H100); 1.5% of particles reach the 100-sample cap.

Momentum SGD (learning rate 1.6, seeds 11 and 12, otherwise as the VDAM seeds,
40cf41d). Stated before the runs: on the simulated fixture SGD gave a sharper mean
map and a less reliable separation than VDAM (section 16.9).

| Run | Pose median, fraction < 10 deg | Masked FSC `mu + W zbar`, shells 8 / 16 / 24 / 28 | Loading / mean power | Removed-set AUC | Canonical correlations with RECOVAR | between the two seeds |
| --- | --- | --- | --- | --- | --- | --- |
| SGD, seed 11 | 3.99 deg, .960 | .996 / .994 / .972 / .939 | 1.79 | .780 | .93 .77 .35 .02 | .94 .90 .47 .08 |
| SGD, seed 12 | 3.98 deg, .960 | .996 / .993 / .969 / .937 | 1.62 | .787 | .93 .81 .27 .01 | |
| VDAM, seed 11 | 4.09 deg, .955 | .990 / .962 / .902 / .849 | .39 | .787 | .87 .55 .04 .01 | .87 .38 .02 .01 |
| VDAM, seed 12 | 3.96 deg, .963 | .990 / .968 / .925 / .889 | .11 | .796 | .88 .42 .02 .00 | |

The map prediction holds: SGD's population average is closer to RELION at every
shell from 16 on, and level with the best VDAM model means of the first table
(.969 / .942 at shells 24 / 28). The
separation prediction does not hold here: the removed-set AUC is the same, and
SGD's latent has two directions shared between seeds (.94, .90) and with the
RECOVAR embedding (.93, .77-.81), where VDAM has one. In the SGD models the mean
is not a map of the particle: the loadings hold 1.6-1.8 times the mean's power, the
mean posterior latent is up to 1.4 prior standard deviations from zero with a
spread of .4 in that coordinate, and the model mean alone scores .19-.29 masked
at shells 4-8. The SGD models were therefore registered to RELION by their
population average. Two seeds; SGD is seed-fragile on the simulated fixture at
this learning rate, so this is not yet a default.

A third SGD seed and oversampling 1 for SGD (job 14981703), with the VDAM rules
fixed beforehand. Seed agreement over SGD seeds 11, 12 and 13 holds: pose median
3.99 / 3.98 / 3.96 deg (range .03), removed-set AUC .780 / .787 / .789 (range
.008), first canonical correlation between seeds .941 / .928 / .938. The second is
.896 / .838 / .879 (VDAM .38 / .28 / .23) and the third .47 / .45 / .31. The
population averages agree across seeds within .005 (masked FSC .969-.974 at shell
24, .937-.941 at shell 28); the loading-to-mean power is 1.79 / 1.62 / 2.12.
Oversampling 1 in the last stage, from the update-160 checkpoints of seeds 11 and
12, same evaluator as for VDAM:

| SGD run | Pose median, oversampling 0 / 1 | Masked FSC `mu + W zbar`, shells 20 / 24 / 28 / 31, oversampling 0 | same, oversampling 1 |
| --- | --- | --- | --- |
| seed 11 | 2.60 / 2.29 deg | .988 / .972 / .939 / .842 | .996 / .992 / .981 / .900 |
| seed 12 | 2.65 / 2.29 deg | .986 / .969 / .937 / .838 | .994 / .989 / .978 / .891 |

The rule holds on both seeds. Under this evaluator SGD at oversampling 0 already
has better poses than VDAM with oversampling 1 (2.67-2.95 deg). For SGD the
radius 32 update is slightly slower with oversampling 1 (80.9 s against 78-79 s);
2.7-3.0% of particles reach the 100-sample cap.

Wall, one seed on an H100: reading the images 3.5-4.5 min (47 GB from 18,466 stack
files, one thread; a parallel reader would shorten it); radius 4, 8 and 16 stages
495, 524 and 1,104 s; radius 32 3,580 s; the final pose pass and embeddings 8 min;
about 1 h 50 in all. On an A100 the radius 32 update takes 135 s and a seed 2 h 50.

Jobs 14944779 and 14957154 (pilot), 14963424 and 14970619 (seeds), 14963878 and
14974315 (oversampling), 14974370 and 14981703 (momentum SGD), 14949391 (update-160
pose pass). Evidence in
`em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_cryoet_20261002/empiar10499/`.


### 16.12 Oversampling 1 on the SNR .008 cell: VDAM with oversampling 1 is the cryo-ET configuration (October 5, 2026)

Does oversampling 1 in the last stage (dense pose grids in the earlier stages,
RELION's two passes in the last; relax 0e0a8c3, which contains the change) remove
momentum SGD's pose weakness on the SNR .008 cell of section 17.3? Fixture
`cryoet_ppca_k3conf_snr0.008_box64_20261003` (399 subtomograms, box 64, q = 2),
seeds 14-18, 200 updates, momentum SGD at learning rate 1.6 and VDAM, each with
oversampling 0 and 1: 20 arms on one A100, one after another.

Rule, written into the job script on October 4 before any arm ran
(`jobs/snr008_os1.sbatch`): SGD with oversampling 1 has at most 1 bad seed of 5
(nearest-GT-centroid accuracy below .8 or pose median above 10 degrees, section
17.3's definition, on the driver's HEALPix 3 pose pass) and a pose median, median
over the seeds, no worse than VDAM's at oversampling 0. That first submission
(job 15005021) stopped at start-up on a driver error (the oversampling override was
passed twice) before any training; the rerun used the same rule, recipe and seeds.

Median over seeds 14-18; map is the mean state masked FSC-AUC, nn the
nearest-GT-centroid accuracy, pose the median pose error on the HEALPix 3 pass
(HEALPix 4 pass in brackets); training wall on the A100.

| Configuration | Map | nn | Pose (deg) | Bad seeds | Training (s) |
| --- | --- | --- | --- | --- | --- |
| SGD 1.6, oversampling 0 | .872 | .975 | 7.62 (8.03) | 2 of 5 (15, 18) | 418 |
| SGD 1.6, oversampling 1 | .902 | .997 | 7.66 (7.79) | 2 of 5 (15, 18) | 356 |
| VDAM, oversampling 0 | .850 | .962 | 6.07 (5.56) | 1 of 5 (18) | 428 |
| VDAM, oversampling 1 | .890 | .992 | 5.97 (5.37) | 1 of 5 (18) | 367 |

| Question | Reading |
| --- | --- |
| Does the SGD rule hold? | No, on both parts: 2 bad seeds, and a pose median of 7.66 against VDAM's 6.07. SGD seed 15 loses states and poses with or without oversampling (pose 36 degrees, nn .73). |
| What does oversampling 1 change? | The map rises on every paired seed (SGD +.025 to +.035, VDAM +.040 to +.046); nn is equal or higher except VDAM seed 16 (.942 to .940); each seed's pose median moves by at most 0.1 degree, so no bad seed is rescued. Training is 14-15% shorter. |
| Is the run comparable with section 17.3? | Yes: the oversampling 0 pose medians reproduce ppcaopt's H100 rows (SGD 6.9 / 38.0 / 7.6 / 4.8 / 10.8, VDAM 6.1 / 5.1 / 9.1 / 5.9 / 13.4 degrees, seeds 14-18) to within 0.05 degree on the A100. |

Recommendation: VDAM with oversampling 1 for cryo-ET PPCA. Oversampling 1 wins the
map and the latent for both optimizers at lower cost, as it did on EMPIAR-10499
(section 16.11), and VDAM keeps fewer bad seeds and better poses than SGD; its map
is .012 below SGD's with oversampling 1.

Limit: one simulated cell of 399 particles. Seed 18 is bad in every configuration
(pose 10.8-13.4 degrees while nn stays at .99 or above): it is a property of that
seed's pose basin, not of the configuration, and is open.

Amendment, same day: seed 18 is a state-frame offset, and the pose metric above is
confounded. The pose error removes one rotation for the whole population (the chordal
mean of `A_est^T A_gt`). The fixture's states are 5nrl `path_symmetric` at 0 / 20 / 40
degrees (subunits B and Db turn against Ab; the GT poses hold Ab fixed), and the model
may reconstruct each state in its own frame along that hinge. Removing one rotation per
GT state instead (the within-state median; `scripts/pose_metrics.py` in the run root,
reported by `evaluate_arm.py` as `poses.within_state` beside the unchanged global
metric): inside every state of all 20 arms, seed 18 included, the median is 3.7-4.5
degrees (HEALPix 3) and 2.0-2.7 (HEALPix 4). States 0 and 2 sit on either side of state
1 about nearly one axis: 15 / 1 / 14 degrees from the global frame in seed 18, 5-9 in
the others. No arm is mirrored (other hand about 115 degrees). The state maps are
registered state by state and are unaffected.

| Configuration | Pose, global (HEALPix 3 / 4) | Pose, within-state (HEALPix 3 / 4) | Bad seeds, global metric | Bad seeds, within-state metric |
| --- | --- | --- | --- | --- |
| SGD 1.6, oversampling 0 | 7.62 / 8.03 | 3.77 / 2.36 | 2 of 5 (15, 18) | 1 of 5 (15: nn .68) |
| SGD 1.6, oversampling 1 | 7.66 / 7.79 | 3.72 / 2.05 | 2 of 5 (15, 18) | 1 of 5 (15: nn .73) |
| VDAM, oversampling 0 | 6.07 / 5.56 | 3.92 / 2.45 | 1 of 5 (18) | 0 of 5 |
| VDAM, oversampling 1 | 5.97 / 5.37 | 3.84 / 2.06 | 1 of 5 (18) | 0 of 5 |

Both verdicts are recorded. The rule as registered, on the global metric, failed. On
the within-state metric it would pass: 1 bad seed, and a pose median of 3.72 against
VDAM's 3.92. Oversampling 1 does improve poses (within-state HEALPix 4 median 2.36 to
2.05 for SGD, 2.45 to 2.06 for VDAM), so its support is stronger than the table above
shows. VDAM with oversampling 1 stays the recommendation, on seed reliability (0 of 5
against SGD's seed 15, whose states do not separate) rather than on poses, which are
tied; SGD with oversampling 1 has the higher map (.902 against .890). Re-scoring of
earlier sections: 17.11.

Script `snr008_os1/run_local_gpu3.sh` (driver `snr008_os1/et_run_hp4.py`, table
`snr008_os1/table_local.py`). Evidence in
`em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_cryoet_20261002/snr008_os1/local_a100_20261005/`
(`table.json`, per-arm `eval.json`).


## 17. VDAM drift from a ground-truth start and the batch-share step (October 3, 2026)

Started from the ground-truth model, momentum SGD holds it and VDAM does not: on
the eleven-state fixture (q = 10, a fixed 300-particle batch on every update) the
state FSC falls from .983 to .77-.83 in 150 updates (section 14 table). The first
quantity to move is the loading power (13x in 10 updates, with latent posterior
covariance trace 3.3 to 0.8); noise, offset variance, step and fudge stay constant.

Mechanism. The coupled direction of section 6.2 makes `theta + d_h` the
unregularized M-step of pseudo-half `h`. With 150 images per half its loading
power is 6-330x the GT loadings', its noise power scales as 1/N, and a pooled
1800-image M-step at GT is unbiased in scale: GT is close to the full-data maximum
likelihood. VDAM's update moves `theta` toward the gated moving average of these
M-steps with step .5, so its moving average spans about `subset / step`
particles. VDAM's own schedule uses late subsets of 10% of the data (10,000 here),
a window of about 20,000 particles; the fixed 300-particle batch shrinks it to
about 600. The gate does not compensate: with fudge 4 it stays at .8-.9 where the
half-M-step Wiener factor is .1-.5. Once noise enters the loadings, the next
M-step reproduces 60-75% of it, because the E-step fits the latent coordinates to
it. Momentum SGD's per-voxel step is `lr * trace / max trace`, .005-.4 of a Newton
step, so it barely moves the noise-dominated shells. Evidence:
`em_work/relax_ppca_vdamdrift_20261003` (`HANDOFF.json`, replays
`analysis/diag_directions.py`, jobs 14907807 and 14907965).

Rule. The PPCA VDAM step is multiplied by
[`Config.step_factor`](../../relax/ppca_initial_model/config.py), `min(1, count /
scheduled count)`, where the scheduled count is VDAM's own subset size for the
iteration (`compute_subset_size`, all particles on the final iteration). The factor
is exactly one on VDAM's own schedule, so default runs are unchanged; only runs
whose `stochastic_batch_size` is below VDAM's subset take smaller steps, restoring
VDAM's particle window. Every update logs it as `vdam_step_factor`. The native
VDAM InitialModel is not affected.

| Eleven-state GT start, 150 tf32 updates, seeds 101/102/103 | State FSC | Worst state | Latent R^2 |
| --- | --- | --- | --- |
| VDAM (factor 1) | .768/.833/.777 | .59-.65 | .12 |
| step x0.1 | .928/.927/.926 | .80 | .46 |
| step x0.03 | .964/.963/.964 | .89 | .59-.60 |
| batch-share rule (factor .038-.040 here; seed 101 at update 4125) | .959/.956/.956 | .87 | .56-.58 |
| momentum SGD | .967 | .90 | .40 |

The state FSC rises monotonically as the step falls (x0.1 .927, x.038-.040 .956, x0.03
.964). The rule's factor is .038-.040 here because these updates sit in VDAM's middle
phase, where its subset is still growing (7,467-7,942 particles); in the final phase
it is 300/10,000 = .03.

Under the rule the state FSC still declines slowly and levels off: seed 101 over
450 updates reads .970, .959, .950, .942, .938, .935, .934 every 75 updates
(latent R^2 .60 to .48). On the cryo-ET random start (seed 11, default schedule)
the factor is one on all 200 updates and the run matches the control (state
FSC-AUC .723, specificity .120, latent nearest-centroid .910). Jobs 14908906,
14910785, 14914173, 14919430.

Rejected. A Wiener shell gate and a gate-free update (both a multiplicative
shrinkage of the M-step output, which compounds through EM: the loading fixed
point `W = phi M(W)` shrinks or collapses) and an empirical-Bayes per-shell
Gaussian prior on the VDAM directions (the prior's precision exceeds the batch
metric at weak shells and the first-moment average overshoots: the mean's
high-frequency FSC against GT turns negative on the cryo-ET start). Removing
the gate on the cryo-ET random start lowered specificity from .12-.14 to .08 and
nearest-centroid accuracy from .90-.92 to .84, so the gate stays. The cryo-ET GT
start drifts on VDAM's own schedule: with 399 particles the full-data maximum
likelihood itself fits noise beyond shell 19 (per-coefficient loading SNR .03-.2),
and SGD's GT hold there is the slowness of its high-shell steps. That needs a
regularizer, not a step rule: the same shell prior under momentum SGD (lr 1.2)
scored FSC-AUC .993 from the cryo-ET GT start but collapsed from a random start
(.04, job 14915796).

High-shell overshoot on the cryo-ET random start (October 3, 2026). When the stage
radius jumps from 16 to 31, VDAM's mean power at shells 20-28 grows to several times
its shell-2 power within 15 updates (total mean power about 10x), on the k3conf and
contrast-sd-.3 fixtures alike; it is not contrast absorbed into the loadings (their
alignment with the mean is the same on both fixtures). One such run stopped at update
199 of 200 on the metric check; resumed from update 190 in tf32 and fp32 it did not
recur (bulk metric `lambda_min / lambda_max >= .0045`; the only indefinite voxels are
denormal spill at shells 32-33, far inside the check's bound, identical in the CUDA
scatter and the XLA adjoint). A failing check now saves the state and the offending
statistics (`failure_before_<iteration>.npz`, `failure_metric_<iteration>_half<h>.npz`).

Tested and rejected: an update radius limited to the shells the data support (pseudo-
half M-step SNR scaled to all particles at least one, plus two shells, RELION's
`data_vs_prior > 1` with headroom). It removed the overshoot (peak mean power
1.2-2.7e6 against 5.4-8.1e6) but lost map quality, because the high shells carry weak
signal: shrink them, do not truncate them. Seed 11, batch 150, against main at
c9d0a11 (jobs 14930390 and 14930399):

| Cryo-ET arm | Radius | FSC-AUC | Specificity | Latent nearest-centroid |
| --- | --- | --- | --- | --- |
| contrast-sd-.3 random start, fixed stages | 31 | .730 | .108 | .867 |
| same, data-supported radius | 20 | .598 | .093 | .897 |
| k3conf random start, fixed stages | 31 | .724 | .116 | .910 |
| same, data-supported radius | 24 | .658 | .108 | .920 |
| k3conf GT start, fixed stages | 31 | .750 | .211 | .977 |
| same, data-supported radius | 21-27 | .724 | .176 | .987 |

| Masked state FSC against GT, shell | 26 | 27 | 28 | 29 | 30 | 31 |
| --- | --- | --- | --- | --- | --- | --- |
| k3conf random start, fixed stages | .40 | .39 | .36 | .31 | .30 | .28 |
| same, data-supported radius | .07 | -.01 | -.03 | -.03 | -.01 | .00 |

The eleven-state runs kept radius 31 under the rule (full-data SNR well above one)
and scored as without it (.956 for all three seeds).

Support-only solve (October 3, 2026). The coupled direction of section 6.2 is solved
and checked only on the update's own support, the rows
[`bandlimit_and_mask`](../../relax/ppca_initial_model/initialization.py) keeps (radius
at most the stage radius); rows outside get a zero direction and no coverage, so their
moments stay uninitialized until a later stage reaches them
([`coupled_direction`](../../relax/ppca_initial_model/update.py)). Outside the support
the metric holds only interpolation spill, and a spill row can sit at the float floor:
the EMPIAR-10499 pilot stopped at update 165, four updates into the radius-32 stage,
on one row of 1,064,960 at radius 33.3 whose near-rank-one 5x5 metric had largest
eigenvalue 2.1e-9 (floor 1.86e-9) and smallest -1.6e-14 against the bound 7.9e-15;
nothing was non-finite. The update-199 stop above is the same class. The 32-eps check
is unchanged on the support. Tests: `tests/unit/ppca_initial_model/test_coupled_direction.py`
(that row outside the support passes, inside it still stops the update) and
`test_metric_is_positive_semidefinite_on_every_row` in `test_tomo_ppca.py`.

### 17.1 Momentum SGD against VDAM: scope, sources and definitions (October 3-4, 2026)

Decision: VDAM stays the default for cryo-ET (200 updates) and for SPA; momentum SGD
stays opt-in with `sgd_learning_rate` 0.4 unchanged. No candidate below is on main
except where a commit is named. Evidence (outputs, arm lists, tables, analysis scripts):
`em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_vdamdrift_20261003`, under
`jobs/matrix` (learning-rate matrix) and `jobs/matrix2` (everything after it;
`NOTES.txt`, `pooled_cells.txt`, `parta_table.txt`, `long_gauge_report.txt`,
`gauge2_report.txt`, `gauge3_report.txt`, `mean_latent.txt`, and the candidates'
code as `patches/gauge_candidates_0880fee_to_8d9354f.patch`).

| Item | Value |
| --- | --- |
| Source | relax 0880fee for training; candidates are wip commits on it. The longer-schedule and gauge runs include the support-only solve (cherry-pick of main 16ca4e5); two VDAM control arms of job 14970637 stopped on the earlier metric check and were rerun with it (job 14973904) |
| Cryo-ET cells | k3conf (`cryoet_ppca_k3conf_box64_20261002`), contrast sd .3 (`..._contrast30_box64_20261003`), SNR .008 and SNR .0005 (`..._snr{0.008,0.0005}_box64_20261003`); 399 subtomograms, q = 2, random start, default stages |
| SPA cells | eleven-state GT start (100k fixture, 150 updates of 300 particles); eleven-state 20k at noise .25 / 1 / 4 (`ppca_elevenstate20k_noise*_20261003`), q = 10 |
| Cryo-ET metrics | masked state FSC-AUC ("map"), state specificity, latent nearest-GT-centroid accuracy ("nn"), k-means accuracy, between-state R^2, pose median error |
| SPA metrics | state FSC shells 1-15 (mean and worst state), latent R^2, pose median, GT heterogeneity power captured |
| Bad seed | nn below .8 or pose median above 10 degrees (global pose metric; at SNR .008 it counts state-frame offsets as pose errors, see 17.11) |
| Loading-power flag | largest loading power after the last stage starts divided by its value at that update, above 10 |
| Pairing | differences are per seed against the named control, then averaged |

Every rule below was fixed before its results. One seed is not a conclusion; seed
counts are in each table.

### 17.2 Learning-rate matrix (jobs 14947237, 14947238, 14947401; 114 arms, none failed)

Cryo-ET, seed means: map / specificity / nn / k-means / R^2 / pose (degrees).

| Cell | lr .8 | lr 1.6 | lr 2.4 | lr 3.2 | lr 4.8 | VDAM |
| --- | --- | --- | --- | --- | --- | --- |
| k3conf random, seeds 11-13 | .768 / .106 / .850 / .819 / .69 / 4.4 | .771 / .142 / .967 / .962 / .77 / 4.5 | .765 / .131 / .940 / .940 / .67 / 4.9 | .755 / .125 / .931 / .808 / .58 / 5.2 | .734 / .084 / .891 / .708 / .56 / 8.9 | .720 / .126 / .910 / .905 / .64 / 4.9 |
| contrast random, seeds 11-12 | .785 / .113 / .907 / .817 / .55 / 4.8 | .775 / .109 / .911 / .766 / .55 / 5.0 | .768 / .105 / .881 / .640 / .48 / 5.7 | .746 / .090 / .743 / .623 / .33 / 22.9 | .754 / .099 / .858 / .576 / .40 / 5.6 | .729 / .113 / .861 / .593 / .48 / 5.3 |
| SNR .008 random, seeds 11-12 | .869 / .109 / .995 / .994 / .95 / 5.3 | .845 / .102 / .877 / .882 / .83 / 7.4 | .880 / .128 / .982 / .982 / .90 / 7.6 | .882 / .130 / .982 / .980 / .89 / 7.5 | .879 / .093 / .976 / .975 / .88 / 9.5 | .830 / .141 / .985 / .860 / .77 / 6.0 |
| k3conf GT start, one seed | .881 / .302 / .992 / .992 / .89 / 3.7 | .817 / .244 / .990 / .990 / .87 / 3.6 | .794 / .225 / .990 / .990 / .87 / 3.6 | .785 / .220 / .990 / .990 / .86 / 3.6 | .787 / .224 / .985 / .982 / .84 / 3.6 | .750 / .211 / .977 / .977 / .82 / 3.6 |
| SNR .0005 random, seeds 11-12 | void | void | void | void | void | void |

SPA eleven-state GT start, seeds 101-103: state FSC / worst state / R^2 / pose.

| lr .8 | lr 1.6 | lr 2.4 | lr 3.2 | lr 4.8 | VDAM |
| --- | --- | --- | --- | --- | --- |
| .973 / .924 / .40 / 3.8 | .960 / .886 / .40 / 3.9 | .948 / .853 / .40 / 3.9 | .935 / .820 / .40 / 4.0 | .905 / .748 / .38 / 4.1 | .956 / .872 / .56 / 3.8 |

| Row | Reading |
| --- | --- |
| Rates above 1.6 | Worse on the GT holds, contrast and k3conf (k-means, R^2, pose); no evidence for going higher. |
| GT starts | The lower rate holds ground truth better, monotonically; VDAM has the higher latent R^2 on SPA (.56 against .40). |
| SNR .0005 (void) | Unsolvable at this budget: every arm at chance (map .12-.14, pose 118-130 degrees, nn .34-.40). |
| SPA 20k random start, noise .25 / 1 / 4 (void) | No optimizer converges from a fully random eleven-state start in 200 updates: state FSC .19-.29, pose 108-133 degrees in 35 of 36 arms; one VDAM seed at noise .25 reaches 53 degrees (state FSC .40). |
| Loading-power flag | Fires on all 11 VDAM cryo-ET random-start arms (13-63) and on no SGD arm (at most 4.8); it is the overshoot at the radius 16 to 31 jump described above, with nothing non-finite. |

### 17.3 Seed study and pooled counts (jobs 14963637-39, 14970636-37, 14973904)

Seeds 11-13 were easy: on seeds 14-18, with the same code and configuration, every
SGD rate separates states worse on k3conf and contrast. Pooled over seeds 11-18,
200 updates: map / nn mean (minimum) / pose median, and bad seeds.

| Cell | SGD .8 | SGD 1.2 (seeds 14-18) | SGD 1.6 | VDAM |
| --- | --- | --- | --- | --- |
| k3conf | .761 / .732 (.42) / 4.3; 5 of 8 | .758 / .734 (.59) / 4.3; 3 of 5 | .764 / .919 (.72) / 4.5; 1 of 8 | .719 / .896 (.86) / 5.4; 0 of 8 |
| contrast | .772 / .739 (.50) / 4.8; 5 of 7 | .751 / .734 (.53) / 5.1; 3 of 5 | .768 / .803 (.54) / 4.8; 3 of 7 | .717 / .784 (.62) / 5.6; 4 of 7 |
| SNR .008 | .865 / .989 (.98) / 5.4; 0 of 7 | .866 / .988 (.98) / 5.8; 1 of 5 | .863 / .909 (.67) / 7.6; 3 of 7 | .843 / .967 (.92) / 6.5; 0 of 7 |

| Comparison, paired by seed | Reading |
| --- | --- |
| lr .8 against 1.6 | nn lower by .187 on k3conf (1.6 better on 7 of 8 seeds) and .064 on contrast; nn higher by .080 on SNR .008, where 1.6 loses states or poses on 2-3 of 7 seeds. No constant rate is stable in all three cells. |
| VDAM against SGD 1.6 | VDAM's map is lower by .045 (k3conf, 8 of 8 seeds), .052 (contrast, 7 of 7) and .019 (SNR .008); its bad seeds are no more frequent except on contrast (4 against 3) and are milder (nn .62-.79 against .54-.80, no pose loss). |

### 17.4 Rate decaying over the last stage: tested, not adopted

Candidate: `sgd_learning_rate` 1.6 decaying geometrically to .4 over updates 161-200
(seeds 14-18, three cells, and the k3conf GT start).

| Pre-stated rule | Result |
| --- | --- |
| k3conf nn at least .96 with no slow seed | No: .72-.98, one slow seed (the constant rate does the same on these seeds). |
| k3conf GT-start map at least .86 | .8598 (lr 1.6 .817, lr .8 .881). |
| SNR .008 pose at most 6 degrees | No: 7.6, identical to lr 1.6 seed by seed. |

Paired against constant 1.6: map +.009 on k3conf (5 of 5 seeds), +.004 on contrast,
-.004 on SNR .008; nn unchanged. Reading: separation and poses are decided before
update 161, so a last-stage decay only polishes the map.

### 17.5 When separation appears (k3conf, lr .8 and 1.6, seeds 11-18)

Pose and latent pass at saved checkpoints (64 passes, evaluations 14970977-84);
means over the eight seeds, chance nn = .33.

| | update 60 | 110 | 130 | 140 | 160 | 200 |
| --- | --- | --- | --- | --- | --- | --- |
| lr .8: nn | .37 | .36 | | | .60 | .73 |
| lr .8: pose median | 122 | 104 | | | 4.4 | 4.3 |
| lr 1.6: nn | .36 | .37 | .52 | .62 | .84 | .92 |
| lr 1.6: pose median | 122 | 28 | 4.5 | 4.5 | 4.5 | 4.5 |

| Finding | Number |
| --- | --- |
| No separation and, at lr .8, no poses in the radius-4 and radius-8 stages | nn .33-.40, R^2 at most .02 through update 110 |
| nn at update 160 predicts the final nn | rank correlation .95; one threshold separates the 6 bad arms from the 10 good ones |
| nn at updates 60 and 110 does not | rank correlation .45 and -.14 |
| Best logged predictor | loading-to-mean power at updates 135-160 (rank correlation .58-.59 on k3conf, .49-.51 on contrast) |

Reading: a bad seed is a slow seed. It locks its poses late, has less of the
radius-16 stage left to grow its loadings, and is still improving at update 200.

### 17.6 Longer schedule (300 updates, stage starts 1 / 91 / 166 / 241; seeds 14-18)

Jobs 14973963, 14973966, 14981363. Rule: SGD has at most 1 bad seed of 5 per cell and
no more than VDAM. Bad seeds / map / nn (minimum) / pose median, mean over seeds.

| Cell | SGD 1.6, 200 | SGD 1.6, 300 | VDAM, 200 | VDAM, 300 | Rule |
| --- | --- | --- | --- | --- | --- |
| k3conf | 1 / .760 / .890 (.72) / 4.7 | 0 / .761 / .941 (.87) / 5.7 | 0 / .718 / .888 (.86) / 5.8 | 0 / .727 / .895 (.87) / 5.5 | met |
| contrast | 3 / .766 / .760 (.54) / 4.8 | 0 / .770 / .888 (.85) / 5.0 | 4 / .712 / .753 (.62) / 6.2 | 1 / .728 / .839 (.78) / 5.7 | met |
| SNR .008 | 2 / .869 / .922 (.67) / 13.6 | 3 / .869 / .974 (.97) / 9.6 | 1 / .848 / .965 (.94) / 7.9 | 0 / .843 / .958 (.92) / 6.0 | not met |

| Question | Reading |
| --- | --- |
| Does a longer schedule fix SGD's slow seeds? | Yes at the default SNR (every seed bad at 200 updates is good at 300); at SNR .008 states are found on every seed but the pose median sits at 8-11 degrees against VDAM's 5-8. |
| Does it make VDAM as sharp as SGD? | No: its map gains .009 (k3conf) and .016 (contrast), about a fifth of the gap, and the loading-power ratio barely moves (58 to 47, 35 to 35). |
| What else changes for VDAM? | Its scale drifts: loading-to-mean power .45 to .18 (k3conf), .29 to .13 (contrast), .28 to .04 (SNR .008), with the posterior-mean latent sd rising from 1.7-2.0 to 2.7-4.2 (prior 1). |

### 17.7 SGD against VDAM: summary and recommendation

| Case | SGD lr 1.6 | VDAM |
| --- | --- | --- |
| Cryo-ET map, every solvable cell, 200 or 300 updates | higher by .02-.05 on nearly every seed | |
| Cryo-ET seed reliability, 200 updates | 1 of 8, 3 of 7, 3 of 7 bad seeds, some severe | 0 of 8, 4 of 7, 0 of 7, all mild |
| Cryo-ET poses at SNR .008 | 8-11 degrees at 300 updates, loss on 2 of 7 seeds at 200 | 5-8 degrees |
| SPA, GT start | state FSC .960, R^2 .40 | state FSC .956, R^2 .56 |
| SPA, consensus-mean start (job 14963640), state FSC at noise .25 / 1 | .873 / .830 (rising with the rate, still rising at 1.6) | .963 / .895 |
| SPA, fully random start | no convergence at any rate | the only arm that starts to converge |
| Real data | section 16.11 (EMPIAR-10499, SGD lr 1.6 against VDAM on the same seeds) | |

Recommendation: VDAM is the default for both modalities. SGD has the better maps
when it works but is not robust across seeds at 200 updates or in poses at high SNR,
and no single rate serves the default-SNR and high-SNR cells. The consensus-mean start
(mean = average of the GT states, random loadings, entered at update 111) converges at
noise .25 and 1 for both optimizers and at noise 4 for neither (poses 118-123 degrees).

### 17.8 The mean posterior latent is not zero: report mu + W zbar

For each particle the posterior latent mean is `W^H A^H C^-1 (y - A mu)`, with `C` the
marginal covariance, so the sum of the posterior means over particles equals `W^H g`
with `g` the gradient of the marginal log-likelihood with respect to the mean. It is
zero only at a stationary mean. Neither optimizer reaches one: momentum SGD moves the
mean a small fraction of a Newton step per update, and VDAM's update `theta + step
(gate x direction - (1 - gate) theta)` holds the mean away from its stationary point
wherever the gate is below one.

Norm of the mean posterior latent (prior sd 1), and the power of `W zbar` relative to
the model mean's power over all shells (seeds 11-18, 200 updates;
`analysis/mean_latent.py`).

| Cell | SGD lr .8 | SGD lr 1.6 | VDAM |
| --- | --- | --- | --- |
| k3conf | 1.21; .75 | .56; .18 | 1.12; .37 |
| contrast | 1.06; .66 | .47; .14 | 2.14; .80 |
| SNR .008 | .99; .52 | .46; .11 | .55; .06 |

| Property | Value |
| --- | --- |
| Size against its sampling error | 5-15 standard errors in almost every run; the direction differs by seed |
| VDAM on the SPA consensus-mean start | .73 (noise .25) and .53 (noise 1); .14 at the GT start |
| Tracks bad seeds? | No consistent sign (correlation with final nn between -.92 and +.68 across cells) |
| Effect on the model mean, k3conf seeds 14-18, masked FSC against the GT average, shells 1-8 | SGD 1.6: .934 for `mu`, .990 for `mu + W zbar`; VDAM: .954 and .986 |

Consequence: the population-average map of a trained model is `mu + W zbar`, with
`zbar` the mean posterior latent, not `mu` alone; registration and average-map scores
use it (`analysis/mean_vs_gt.py`; the same on real data in section 16.11).

### 17.9 Gauge steps: three candidates, none adopted

The transformation `mu -> mu + W c`, `z -> z - c` leaves the data term unchanged and
`c = zbar` maximises the prior term; likewise `W -> W S^(1/2)`, `z -> S^(-1/2) z` for
the latent scale. Rule 1 (mean gauge against a same-snapshot control, paired by seed,
every point in every cell): cryo-ET map change at least -.005, nn change at least -.02
with no more bad seeds, pose change at most +.3 degrees, model-mean FSC against the GT
average on shells 1-8 at least -.002, final mean-latent norm below .2; SPA state FSC at
least -.005, worst state at least -.01, R^2 at least -.02, pose at most +.3 degrees, GT
power at least -.02, mean-latent norm below .2.

| Candidate | Cells run | Outcome | Misses |
| --- | --- | --- | --- |
| Full step every update, momentum SGD lr 1.6 (job 14974448) | k3conf | closed | map -.022, nn -.120 (bad seeds 1 to 2), pose +4.9 degrees |
| Full step every update, VDAM (14974448, 14981363, 14981364) | all six | not adopted | SNR .008 map -.010; SPA GT-start worst state -.015 |
| Mean plus scale gauge, VDAM, against the mean gauge alone (14981363, 14981364) | five | closed in this form | SNR .008 pose +10.4 degrees and bad seeds 0 to 3; SPA consensus noise .25 state FSC -.015, GT power -.083; noise 1 state FSC -.007; GT start GT power -.030 |
| Step scaled by the noise-subtracted gain `(1 - v / |c|^2)_+`, `v` the sampling variance of the batch mean latent, VDAM (14992588, 14992589) | all six | closed (one miss) | SPA GT-start worst state -.0115 against the limit -.010, 3 of 3 seeds |

Gated step against the control, cell by cell (VDAM).

| Cell | map or state FSC | worst state | nn or R^2; bad seeds | pose | mean FSC shells 1-8, or GT power | final mean latent |
| --- | --- | --- | --- | --- | --- | --- |
| k3conf, seeds 14-18 | +.004 | | +.012; 0 to 0 | -.66 | +.036 (.954 to .990) | .12 |
| contrast, seeds 14-18 | +.025 | | +.148; 3 to 0 | -1.37 | +.053 (.937 to .990) | .16 |
| SNR .008, seeds 14-18 | +.000 | | .000; 1 to 0 | -1.32 | +.039 (.949 to .988) | .15 |
| SPA consensus, noise .25, seeds 11-12 | +.005 | +.003 | -.001 | +.00 | +.011 | .04 |
| SPA consensus, noise 1, seeds 11-12 | +.004 | +.007 | +.002 | -.04 | +.007 | .07 |
| SPA GT start, seeds 101-103 | -.0014 | -.0115 | -.0015 | -.01 | +.0003 | .06 |

| Observation | Number |
| --- | --- |
| Why the full step hurts at the SPA GT start | the batch mean latent's noise is about .18 against an offset drift of about .005 per update (300-particle batches, q = 10) |
| Why the gated step still misses there | its gain averages .14-.20, so part of that noise is still applied |
| Mean map with the gated step, cryo-ET shells 17-31 against the GT average | k3conf .39 to .54, contrast .16 to .56, SNR .008 .53 to .62 |
| Scale gauge, cryo-ET random starts | posterior-mean latent sd 2.0-2.2 to 1.01-1.02; loading-power ratio 30 to 8.5 (contrast), 10 to 4.6 (SNR .008); contrast bad seeds 3 to 0 |
| Scale gauge, SPA GT start | state FSC +.014, worst state +.063, R^2 +.081, but GT power -.030 |
| Scale criterion on SPA | "posterior-mean latent sd within .8-1.2" was mis-specified: the SPA controls sit at .70-.89 because the posterior covariance carries the rest |

Reading: the mean gauge improves the model mean in every cell and never hurts
separation, but a correction driven by a noisy batch estimate degrades a model that is
already at ground truth; the offset of 17.8 therefore stays a documented property.

### 17.10 Scale-preserving loading shrinkage: tested, rejected; the scale drift costs no accuracy (October 5, 2026)

VDAM's update `theta + step (gate x direction - (1 - gate) theta)` has the fixed point
`theta_M / (1 + 1/rho)` per coefficient, so the decay term `-(1 - gate) theta` shrinks
the loadings and the latents absorb the lost scale (17.6). Candidate (VDAM only, one
scalar per update, relax commit 238616c on 0880fee, not merged): after the standard update,
rescale the loadings so that their information-weighted norm `sum_f h_f |W_f|^2` (with `h`
the batch metric's mean-mean diagonal) equals its value without the decay term. Jobs
15004947 (H100), 15004948 (A100), evals 15004949; table `jobs/matrix2/sp_report.txt`
in the run root of section 17.

Candidate against the same-snapshot VDAM control, paired by seed (cryo-ET: seeds 14-18;
bad seed = nn below .8 or median pose above 10 degrees).

| Cell | map | nn | bad seeds | pose | loading-to-mean power | posterior-mean latent sd |
| --- | --- | --- | --- | --- | --- | --- |
| k3conf, 200 updates | -.031 | -.132 | 0 to 3 | +10.4 | .34 to 188 | 1.91 to .08 |
| k3conf, 300 updates | -.056 | -.202 | 0 to 4 | +9.5 | .18 to 939 | 2.67 to .04 |
| contrast, 200 updates | +.019 | +.134 | 3 to 0 | -0.5 | .28 to 195 | 1.99 to .08 |
| SNR .008, 200 updates | -.013 | -.031 | 1 to 3 | +11.2 | .28 to 201 | 1.97 to .09 |
| SNR .008, 300 updates | -.016 | -.053 | 0 to 5 | +15.4 | .04 to 194 | 4.19 to .10 |

| SPA cell | state FSC | worst state | R^2 | GT power |
| --- | --- | --- | --- | --- |
| Consensus-mean start, noise .25, seeds 11-12 | +.0005 | +.001 | -.046 | +.002 |
| Consensus-mean start, noise 1, seeds 11-12 | -.0015 | +.001 | -.056 | -.008 |
| GT start, seeds 101-103 | +.012 | +.040 (3 of 3 up) | -.054 | +.004 |

Why it fails: the scalar is 1.04-1.05 on every update and never settles, so the drift
is reversed without bound (loadings grow, latent sd collapses to .04-.10, model-mean FSC
against the GT average on shells 1-8 falls by .07-.20 in every cryo-ET cell). The decay
term is the only part of the update that bounds `|W|`; cancelling its effect on scale
leaves the `W`/`z` trade-off free in the other direction.

Does the drift itself cost accuracy? VDAM controls at the end of the 200-update schedule
against the end of the 300-update schedule (stage starts 1 / 91 / 166 / 241), same seeds
14-18, paired (`analysis/drift_harm.py`, `jobs/matrix2/drift_harm.txt`; the two schedules
differ in stage lengths as well as in count, and the 200 arms come from the gauge-control
jobs 14974448 and 14981363, the 300 arms from 14973963, 14973966 and 14981363).

| Cell | latent sd | map | nn | pose | bad seeds | k-means accuracy | R^2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k3conf | 1.91 to 2.67 | +.011 (5 of 5 up) | -.001 | -.23 | 0 to 0 | +.021 | +.051 |
| contrast | 1.99 to 2.73 | +.023 (5 of 5 up) | +.134 | -.70 | 3 to 1 | +.009 | +.108 |
| SNR .008 | 1.97 to 4.19 | -.005 (2 of 5 up) | -.007 | -1.90 | 1 to 0 | -.048 | -.103 |

Reading: on the gated metrics (map, nn, pose, bad seeds) accuracy does not get worse
while the latent sd grows from 2 to 3-4; the one change that falls is SNR .008's latent
geometry (k-means -.048 and R^2 -.103, 4 of 5 seeds down), which is where the drift is
largest. The drift is therefore treated as a trade-off between the loadings and the
latents that VDAM's decay term sets, not as an accuracy defect; the item is closed with
no option added.

### 17.11 Pose medians count state-frame offsets: re-scored (October 5, 2026)

The pose median of 17.1-17.10 removes one rotation for the whole population. Section
16.12's amendment shows that the model may hold each conformation in its own frame
(the k3conf states differ by a hinge rotation of subunits), so that metric adds each
state's frame offset to the alignment error. Every cryo-ET arm with a saved pose pass
was re-scored from its existing `poses.npz` with one rotation removed per GT state
(the within-state median), on CPU, with no new training (`scripts/rescore_poses_et.py`,
`scripts/rescore_poses_spa.py` in the run root of section 16; outputs under
`rescore_poses_20261005/`; copies in
`em_fixtures/ppca_evidence_20261003/em_work/relax_ppca_cryoet_20261002/`). The global metric of the re-score reproduces every pose
number quoted below to the printed precision.

Size of the confound. On k3conf and contrast the per-seed largest state offset has a
median of 2.4-5.3 degrees in most cells and the within-state median is .1-1.3 degrees
below the global one. On SNR .008 the offsets are 6-9 degrees (up to 74-126 in some arms), and the
within-state medians of every optimizer and variant fall to 3.6-4.4 degrees. Bad-seed
sets change only where noted below.

| Section, comparison (SNR .008 unless named) | As published (global) | Within-state | Verdict |
| --- | --- | --- | --- |
| 17.3 pooled bad seeds, SGD 1.6 / 1.2 / .8 / VDAM | 3 of 7 / 1 of 5 / 0 of 7 / 0 of 7 | 2 of 7 / 0 of 5 / 0 of 7 / 0 of 7 | SGD 1.6's remaining bad seeds (11-12 matrix seed 12, seed 15) are latent failures (nn .75 and .67); stands |
| 17.6 pose, SGD 1.6 at 200 / 300, VDAM at 200 / 300 (mean over seeds 14-18) | 13.6 / 9.6 / 7.9 / 6.0 | 3.89 / 3.82 / 3.86 / 3.86 | the rule "not met" stands at 200 updates (SGD 1 bad seed against VDAM's 0); at 300 it would be met (0 against 0); the reading "SGD's poses 8-11 against VDAM's 5-8 degrees" is withdrawn: poses are tied |
| 17.7 "cryo-ET poses at SNR .008" | SGD 8-11, VDAM 5-8 degrees | tied, 3.8-3.9 | VDAM stays the default on seed reliability; the pose part of that row no longer supports it |
| 17.2 rates above 1.6, pose (k3conf 4.8 / contrast 3.2 / SNR .008 2.4-4.8) | 8.9 / 22.9 / 7.5-9.5 | 5.1 / 5.7 / 3.6-3.7 (lr 1.6 k3conf: 4.0) | "worse at higher rates" stands on k-means and R^2; the pose evidence for it is small (k3conf, contrast) or absent (SNR .008) |
| 17.9 mean plus scale gauge against mean gauge | pose +10.4, bad seeds 0 to 3 | pose -.14, 0 to 0 | its SNR .008 miss disappears; still closed by its SPA misses |
| 17.9 gated step against control | pose -1.32, bad seeds 1 to 0 | pose +.01, 0 to 0 | the pose gain was frame; the rule (at most +.3) still holds |
| 17.10 loading shrinkage, 200 / 300 updates | pose +11.2 / +15.4, bad seeds 1 to 3 / 0 to 5 | pose +.97 / +1.92, 0 to 0 / 0 to 1 | rejection stands on k3conf (within-state pose +5.8 / +8.8, bad seeds 0 to 2 / 0 to 4) and on the latent collapse |
| 17.10 drift, 200 against 300 updates | pose -1.90, bad seeds 1 to 0 | pose .00, 0 to 0 | "does not get worse" stands |

SPA, eleven-state 20k consensus-mean start (ppcaspeed's oversampling table, section
"Schedule" above; scorer `harness10/score_contshift_1px.py`): the global chordal-mean
median matches the scorer's map-frame median within .4 degree, and the within-state
median equals it within .5 degree in all 16 arms; at noise 1 states 0 and 1 have
within-state medians of 93-120 degrees, which is a failure inside those states, not a
frame offset. No SPA verdict moves.

From October 5 `evaluate_arm.py` reports both metrics; cryo-ET pose comparisons quote
the within-state median, and the global one for continuity.

