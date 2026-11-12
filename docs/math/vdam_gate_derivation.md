# The VDAM shrinkage gate: what RELION computes, what it is meant to estimate, and what is changed here

This note covers VDAM in general (a single volume or K classes; nothing here is specific to PPCA):

1. the update of the VDAM paper (equations 8–12) and the place where the FSC enters it (sections 2–3);
2. how RELION implements the gate that stands in for that FSC (section 4);
3. why the disagreement of the two half-batch preconditioned gradients measures a noise variance, and why
   $\rho/(1+\rho)$ is $P/(P+V)$ (signal power over signal power plus noise variance), which is what an FSC estimates
   (sections 5–6);
4. why `relax ppca_initial_model` changes RELION's gate in three places, and what each change corrects (section 7).

Sources. Paper: Kimanius, Dong, Sharov, Nakane and Scheres, "New tools for automated cryo-EM single-particle
analysis in RELION-4.0", *Biochem. J.* 478:4169–4185 (2021), doi 10.1042/BCJ20210708. Equation numbers and
formulas are those of its Methods, read from the Europe PMC full-text XML (PMC8786306); the PDF was not
consulted. RELION: 5.0.1, `src/backprojector.cpp` (`reweightGrad`, `getFristMoment`, `getSecondMoment`,
`applyMomenta`, `reconstructGrad`) and `src/ml_optimiser.cpp` (`storeWeightedSums` and the calls of the methods
above). Code: `stochastic_update`, `advance_noise_weight` and `full_data_disagreement_scale` in
[update.py](../../relax/ppca_initial_model/update.py), and `relion_vdam_m_step_device` in
[relion_mstep.py](../../relax/vdam/relion_mstep.py), which ports RELION's update. The scientific decisions around
the gate are in section 3.3 and sections 18.16–18.17 of the [VDAM–PPCA algorithm note](vdam_ppca_algorithm.md).

## 0. The result in one chain

$$
\underbrace{\frac{F}{F+\tau^{-2}}}_{\text{EM weight (eq. 11)}}
=\frac{P}{P+V}\ \ \text{with}\ P=\tau^2,\ V=F^{-1}
\quad\longrightarrow\quad
\underbrace{\mathrm{FSC}}_{\text{eq. 12}}=\frac{P}{P+V_{\mathrm{half}}}
\quad\longleftarrow\quad
\underbrace{\frac{\rho}{1+\rho}}_{\text{gate}},\ \ \rho=\frac{2\,\mathrm{fudge}\,\hat P}{\nu},\ \ \mathbb E\,\nu=2V.
$$

Every term is a signal-to-noise ratio mapped through $\mathrm{SNR}/(1+\mathrm{SNR})$. The gate measures the noise $V$
from the disagreement of two half-batch gradients instead of from a model ($F$, $\tau$) or from separately
reconstructed half maps (the paper's FSC). RELION's gate and the one in this repository both have the form
$\rho/(1+\rho)$ with $\rho$ built from that disagreement. They differ in the statistic and in which noise the
disagreement is taken to be (section 7).

## 1. Notation

Per class, per Fourier voxel; the class and voxel indices are suppressed. All quantities are complex, with the
Hermitian and padding conventions of the code left out.

| Symbol | Meaning | Code |
| --- | --- | --- |
| $v$ | current estimate of the volume | `theta`, the projector of the reference |
| $x_i$, $H_i^q$, $\sigma^{-2}$ | image, operator (CTF, projection, shift) at the hidden state $q$, noise precision | |
| $\gamma_i(q)$ | posterior of $q$ for particle $i$ at the current model | responsibilities |
| $B$, $\mathcal F$ | $\sum_i\!\int\gamma_i(q)\,H^*\sigma^{-2}x_i\,dq$, $\sum_i\!\int\gamma_i(q)\,H^*\sigma^{-2}H\,dq$ | RHS, LHS |
| $F$ | diagonal of $\mathcal F$ (paper eq. 6) | the weight, `weight` |
| $R$ | residual backprojection $\sum_i\!\int\gamma_i(q)\,H^*\sigma^{-2}(x_i-Hv)\,dq=B-\mathcal Fv$ | `data`, `residual_gradient` |
| $\tau^2$ | prior variance of $v$ in the shell | `tau2` |
| $\eta$ | learning rate | `grad_stepsize`, `step` |
| $P$, $V$ | signal power and noise variance of a reconstruction (section 6); capital $V$ is not the volume $v$ | |
| $b$, $N$ | particles in a mini-batch, in the data set | |

## 2. VDAM as a preconditioned gradient step (paper equations 6–11)

**2.1 The EM step.** For fixed posteriors, the expected log posterior of the volume is

$$
Q(v)=-\tfrac12\sum_i\!\int \gamma_i(q)\,\|x_i-H_i^qv\|^2_{\sigma^{-2}}\,dq-\tfrac12\,v^*\tau^{-2}v
=\operatorname{Re}(B^*v)-\tfrac12\,v^*(\mathcal F+\tau^{-2})v+\text{const}.
$$

Its maximizer is the regularized EM step, which with the diagonal metric of the paper is

$$
v_{\mathrm{EM}}=(\mathcal F+\tau^{-2})^{-1}B\;\approx\;\frac{B}{F+\tau^{-2}}
$$

(the left side of eq. 9).

**2.2 The gradient.** By Fisher's identity the gradient of the marginal log likelihood at the current model
equals the gradient of $Q$ at the current $v$. The data part is $B-\mathcal Fv=R$. The paper's $G$ is its negative,

$$
G=\mathcal F v-B=-R .
$$

Equations 7–11 add the prior term $\tau^{-2}v$ explicitly, so $G$ must be the data term alone; if it included the
prior, the prior would be counted twice. Eq. 7 is plain gradient ascent on the log posterior,

$$
v\leftarrow v-\eta\,(G+\tau^{-2}v)=v+\eta\,(R-\tau^{-2}v),
$$

with the identity as metric. $G$ is accumulated as the backprojected residual, $G=-R$.

**2.3 The metric (eqs. 8–10).** Replace the identity by a positive definite $D$: $v\leftarrow v-\eta D(G+\tau^{-2}v)$.
Eq. 9 requires that one full step, $\eta=1$, reproduce the EM step for every $v$ and $B$. With the diagonal
metric $G+\tau^{-2}v=(F+\tau^{-2})v-B$,

$$
\frac{B}{F+\tau^{-2}}=v-D\,[(F+\tau^{-2})v-B]
\;\Longleftrightarrow\;
D\,(F+\tau^{-2})=1\ \text{ and }\ D\,B=\frac{B}{F+\tau^{-2}}
\;\Longrightarrow\;
D=\operatorname{diag}\{(F+\tau^{-2})^{-1}\}\quad\text{(eq. 10)}.
$$

$D$ is positive definite because $F\ge0$ and $\tau^{-2}>0$. It is the inverse of the diagonal Hessian of $-Q$, so
the step is a diagonal Newton step on the EM surrogate. The equality with EM holds exactly only when $\mathcal F$ is
diagonal.

*Printed sign of eq. 8.* As encoded in the XML, eq. 8 reads $+\eta D[G-\tau^{-2}v]$, while eqs. 7, 9 and 11 use
$-\eta(G+\tau^{-2}v)$. Only the second form gives EM at $\eta=1$; this note uses it.

**2.4 Equation 11 is a relaxation toward the Wiener-filtered reconstruction.** Define the unregularized
reconstruction and the Wiener weight

$$
\hat\theta=v-\frac{G}{F}=\frac{B}{F},\qquad W=\frac{F}{F+\tau^{-2}}=\frac{\tau^2F}{1+\tau^2F}.
$$

Then $(G+\tau^{-2}v)/(F+\tau^{-2})=[F(v-\hat\theta)+\tau^{-2}v]/(F+\tau^{-2})=v-W\hat\theta$, so eq. 11 is

$$
v^{(n+1)}=v-\eta\,\frac{G+\tau^{-2}v}{F+\tau^{-2}}=(1-\eta)\,v+\eta\,W\,\hat\theta .
$$

$\eta=1$ is EM; a mini-batch uses $\eta<1$ because $B$ and $F$ come from a few particles. $\hat\theta$ is what the batch
alone would reconstruct without a prior, and $W$ is the Wiener filter for it, by the following argument. With known
poses, $\hat\theta=v_{\mathrm{true}}+F^{-1}\sum H^*\sigma^{-2}\varepsilon$, and the noise term has covariance
$F^{-1}\mathcal F F^{-1}$, whose diagonal is $F^{-1}$. So $V=1/F$ is the noise variance of $\hat\theta$, $P=\tau^2$ is
the signal variance, and

$$
W=\frac{\tau^2F}{1+\tau^2F}=\frac{P/V}{1+P/V}=\frac{P}{P+V}.
$$

## 3. Where the FSC appears (paper equation 12)

Write the preconditioned data gradient $\hat G=G/(F+\epsilon_1)$, with $\hat G=v-\hat\theta$ for $\epsilon_1\to0$, and use
$(G+\tau^{-2}v)/(F+\tau^{-2})=W\hat G+(1-W)v$ (expand $G=F\hat G$). Eq. 11 is then

$$
v^{(n+1)}=v-\eta\,\big[\,W\,\hat G+(1-W)\,v\,\big].
$$

The update is a convex combination of two preconditioned directions:

- the data gradient $\hat G$, with weight $W$;
- the pull toward zero from the prior, $\tau^{-2}v/(F+\tau^{-2})=(1-W)v$, with weight $1-W$.

**Equation 12 replaces $W$ by the FSC:**

$$
v^{(n+1)}=v-\eta\,\Big[\mathrm{FSC}\cdot\frac{G}{F+\epsilon_1}+(1-\mathrm{FSC})\,v\Big]
=(1-\eta)\,v+\eta\,\mathrm{FSC}\cdot\hat\theta .
$$

Limits. At $\mathrm{FSC}\to1$ the step is a pure data step, $v\leftarrow(1-\eta)v+\eta\hat\theta$. At $\mathrm{FSC}\to0$ it
is $v\leftarrow(1-\eta)v$, which removes frequencies where the data say nothing. The FSC multiplies the preconditioned
gradient and its complement multiplies the current estimate.

Why this substitution is natural. $W$ and the FSC have the same form, $\mathrm{SNR}/(1+\mathrm{SNR})$ (sections 2.4 and
6.1). $W$ needs the prior variance $\tau^2$ and trusts $F$ as the noise variance of the reconstruction, which ignores
pose uncertainty. The FSC is a measured SNR and needs neither. (This is my reading; the paper states only that the
FSC is used as a signal-to-noise estimate for each voxel and that it comes from two separately reconstructed halves,
each from one half of the data set.)

Why $P/(P+V)$ is also the scalar a shrinkage should use. For an estimate $y=\theta^*+\text{noise}$ with noise variance
$V_y$ and a gain $g$, the mean square error of $g\,y$ against a signal of power $P$ is $(1-g)^2P+g^2V_y$, minimized at
$g=P/(P+V_y)$.

**Equations 13–14 (moments).** Each half $h$ keeps a first moment $m_h\leftarrow\beta_1m_h+(1-\beta_1)\hat G_h$,
$\beta_1=0.9$, and a second moment $u$ shared by the halves with $\beta_2=0.999$; the update is eq. 12 with $\hat G$
replaced by $m_h/(\sqrt u+\epsilon_2)$ and no prior term inside the averages. Section 4 gives RELION's version.

## 4. The gate as RELION implements it

The paper's FSC comes from two separately reconstructed half maps. In an InitialModel (VDAM) run RELION does not
reconstruct two half maps. `grad_pseudo_halfsets = gradient_refine && !do_split_random_halves` splits each
mini-batch into two pseudo-halves (by the parity of the particle id; see
[section 3.2](vdam_ppca_algorithm.md#32-preconditioning-and-moments) of the algorithm note), and both halves share the
current reference. `reconstructGrad` is called with `use_fsc = do_split_random_halves`, which is false there, so it
forms an FSC-like weight from the halves. (When a refinement splits random halves, `use_fsc` is true and the
FSC of the half maps, `fsc_halves_class`, is used directly; that path is not covered here.)

**4.1 One iteration.** Methods of `BackProjector`, called from `MlOptimiser`, in this order, for each class and each
pseudo-half $h\in\{0,1\}$. All sums run over the voxels with $r^2<\mathrm{ROUND}(r_{\max}\cdot\text{padding})^2$.

1. `storeWeightedSums` backprojects the residual image $x_i-H_iv$ (`Fimg_shift_nomask − Frefctf`, the image minus the
   CTF-weighted projection of the current reference) into `data` and accumulates the matching weights into `weight`:
   $R_h$ and $F_h$.
2. `reweightGrad`: $d_h=R_h/\max(1,F_h)$. This is $-\hat G_h$, the ascent sign, with the floor 1 in the role of
   $\epsilon_1$.
3. `getFristMoment` (sic), $\lambda=0.9$: $m_h\leftarrow0.9\,m_h+0.1\,d_h$; while the moment is still all zero (the
   first iteration) $m_h\leftarrow d_h$.
4. `getSecondMoment`, $\lambda=0.999$: $u\leftarrow0.999\,u+0.001\,|d_1-d_0|^2/(|(d_0+d_1)/2|^2+10^{-12})$.
5. `applyMomenta`: `data` $\leftarrow\bar m/(\sqrt u+10^{-12})$ with $\bar m=(m_0+m_1)/2$, and, per shell $s$,
   `mom1_noise_power`$_s=\nu_s=\big\langle|m_0-m_1|\big\rangle_s$, the mean over the shell of the amplitude
   `sqrt(norm(diff))`.
6. `reconstructGrad`, with $\hat v$ the Fourier transform of the current reference as the E-step makes it, without the
   gridding correction (`computeFourierTransformMap(..., false)`):

$$
a_s=\big\langle|\hat v|\big\rangle_s,\qquad
\rho_s=\frac{2\,T\,a_s}{\nu_s},\qquad
\varphi_s=\min\big\{1,\ \max[\rho_s/(1+\rho_s),\ \mathrm{fsc\_spectrum}_s]\big\},
$$

where $T$ is `tau2_fudge` and `fsc_spectrum` is an input floor (zero in the InitialModel driver, according to section
3.3 of the algorithm note; I did not trace it in RELION). A shell with $\nu_s\le0$ gets $\rho_s=0$; before the
noise power exists, $\varphi_s=1$. Then `tau2_fudge` is set to 1, and each voxel is updated with
$\eta=\texttt{grad\_current\_stepsize}\cdot(1-e^{-(3K+10)\,p_k})$ ($K$ classes, $p_k$ the class fraction):

$$
v\leftarrow v+\eta\,\Big\{\varphi_s\,\frac{\bar m}{\sqrt u+10^{-12}}-(1-\varphi_s)\,v\Big\}.
$$

This is eq. 14 with $\varphi$ in place of the FSC.

**4.2 What the gate is, and is not.** It is a function of the ratio of two shell means of amplitudes: that of the
current reference ($a_s$) and that of the difference of the two half first moments ($\nu_s$). The first moments are
running averages over past iterations (step 3). The gate does not depend on the number of particles $b$ of the
mini-batch, on the size $N$ of the data set, or on how many iterations the moments average. The quantity
`mom1_noise_power` is named a power but is a mean amplitude. Sections 5 and 6 say what the ratio of two such means
measures if the means are taken of power; section 7 says what the amplitudes and the missing scale change.

## 5. Why the half disagreement is a noise variance

**5.1 The direction is the reconstruction minus the model.** For half $h$, with $\mathcal F_h$ its normal operator and
$B_h$ its right-hand side, the gradient is $G_h=\mathcal F_h v-B_h$, so

$$
\mathrm{LHS}^{-1}\,\mathrm{grad}=\mathcal F_h^{-1}G_h=v-\mathcal F_h^{-1}B_h=v-\hat\theta_h .
$$

The direction of step 2 is the negative of this, $d_h=\hat\theta_h-v$ (`relion_vdam_m_step_device` forms it as
$R/\max(1,F)$; the PPCA `coupled_direction` solves the same system with a $(1+q)\times(1+q)$ block per voxel).

**5.2 The model cancels exactly in the difference.** Both halves use the same $v$ of the same iteration. Both
reconstructions estimate the same underlying object $\theta^*$ (the true signal, plus any bias the common model puts
into the reconstruction), each with its own noise:

$$
\hat\theta_h=\theta^*+e_h,\qquad \mathbb E\,e_h=0,\quad \mathbb E|e_h|^2=V_h,\quad e_0\perp e_1 .
$$

The halves hold disjoint particles, so $e_0$ and $e_1$ are independent given the model. Then

$$
d_0-d_1=\hat\theta_0-\hat\theta_1=e_0-e_1,
\qquad
\mathbb E\,|d_0-d_1|^2=V_0+V_1=2V .
$$

The model $v$, the signal $\theta^*$ and any common bias subtract out by algebra, not by statistics. This is why the
difference measures noise whatever the model quality, while the sum $d_0+d_1=2(\theta^*-v)+e_0+e_1$ carries the
signal. The same cancellation holds for the first moments across iterations: $m_h=\sum_jc_jd_h^{(j)}$, so
$m_0-m_1=\sum_jc_j(e_0^{(j)}-e_1^{(j)})$ although $v$ moved between the iterations.

**5.3 What $V$ is.** For known poses $V=1/F$ (section 2.4), and it scales as the inverse number of particles. The
measured difference is the empirical variance of the direction actually applied. It includes the effect of pose
uncertainty on $\hat\theta_h$, which $1/F$ ignores, and excludes bias common to both halves. That is the reason to
measure $V$ from the difference.

**5.4 The first moments average noise down.** RELION's $m_h\leftarrow0.9\,m_h+0.1\,d_h$ adds up independent noise with
coefficients $c_j=(1-l)\,l^{t-j}$, $l=0.9$, so in steady state with constant batches
$\mathbb E|m_0-m_1|^2=2eV_b$, with $e=(1-l)/(1+l)=0.0526$ and $V_b$ the noise variance of one half-batch direction
(a reconstruction from $b/2$ particles). At the first iteration $m_h=d_h$ and there is no reduction.

**5.5 Where the identity fails.** These follow from the algebra above and are not measured here.

- *The preconditioner must be the normal operator.* In general $d_h=(\mathcal F_h/F_h)(\theta^*-v)+F_h^{-1}(\text{noise})$.
  If $\mathcal F_h\ne\operatorname{diag}\mathcal F_h$ (interpolation couples voxels) or the floor $\max(1,F)$ is active, the
  first term differs between halves by a multiple of the model error $\theta^*-v$ and adds to the measured
  disagreement.
- *Independence needs disjoint halves, and independent batches for the average.* A particle stays in its half and
  recurs in later batches, so the noise of a half's first moment is correlated across iterations (section 7.6).
- *Shared bias is invisible.* A bias common to both halves (a wrong model that both halves fit) counts as signal.

## 6. $\rho/(1+\rho)=P/(P+V)$: the gate as an FSC estimate

**6.1 The FSC.** Two independent half reconstructions of the same object, $A_0=S+e_0$, $A_1=S+e_1$, with
$\mathbb E|S|^2=P$ and $\mathbb E|e_h|^2=V$, give $\mathbb E[A_0A_1^*]=P$ and $\mathbb E|A_0|^2=\mathbb E|A_1|^2=P+V$. The
expected shell FSC is

$$
\mathrm{FSC}=\frac{\operatorname{Re}\langle A_0A_1^*\rangle}{\sqrt{\langle|A_0|^2\rangle\langle|A_1|^2\rangle}}=\frac{P}{P+V}.
$$

With $\mathrm{SNR}=P/V$ this is $\mathrm{SNR}/(1+\mathrm{SNR})$, the standard relation between FSC and SNR of a half map.

**6.2 The gate, with shell means of power.** Take $\mathrm{signal}=\hat P\approx P$ and $\mathrm{disagreement}=\nu$
with $\mathbb E\,\nu=2V$ (section 5.2). Then

$$
\rho=\frac{2\,\mathrm{fudge}\,\hat P}{\nu}\;\to\;\mathrm{fudge}\,\frac{P}{V},
\qquad
\frac{\rho}{1+\rho}=\frac{\mathrm{fudge}\,P}{\mathrm{fudge}\,P+V}.
$$

The factor 2 in $\rho$ is the 2 of $\mathbb E|e_0-e_1|^2=2V$. At $\mathrm{fudge}=1$ the gate is $P/(P+V)$, which is the FSC
of halves with noise $V$ (6.1) and the Wiener weight $W$ of section 2.4. $\mathrm{fudge}$ multiplies the SNR, so it acts
as $\tau^2\to\mathrm{fudge}\cdot\tau^2$, which is the role of RELION's regularization parameter $T$.

**6.3 What the proxy does not give.** $\hat P$ is the shell mean of $|v|^2$ (or of $|v|$), not of $|\theta^*|^2$. Noise
left in $v$ raises it, and the shrinkage applied by earlier gates lowers it; the net bias is not derived here. The FSC
uses the cross spectrum $\operatorname{Re}\langle A_0A_1^*\rangle$, which is unbiased for $P$. The halves share the
model, so $\nu$ is the noise of the updates and not of independently reconstructed maps (section 5.5). The gate is
the FSC only under those idealizations. Sections 4 and 6.2 differ in two respects, the statistic and the noise that
$\nu$ stands for; they are the subject of section 7.

## 7. The changes in this repository, and why each is needed

`stochastic_update` keeps RELION's structure (section 4: the same first and second moments, the same update
$v\leftarrow v+\eta\{\varphi g-(1-\varphi)v\}$, the same form $\rho/(1+\rho)$) and changes how $\rho$ is computed. It has no
`fsc_reconstruct` floor. In the code: `first = 0.9 first + 0.1 directions` (the first iteration:
`first = directions`), $\mathrm{signal}$ = shell mean of $|v|^2$, $\mathrm{disagreement}$ = shell mean of
$|\mathrm{first}_0-\mathrm{first}_1|^2$ times `disagreement_scale`, `rho = 2 fudge signal / disagreement`,
`gate = rho/(1+rho)`. The changes are not options: the owner made them the behavior on November 6, 2026 (algorithm
note, sections 3.3 and 18.16).

| | RELION | here |
| --- | --- | --- |
| statistic | shell means of amplitudes, $\langle\lvert v\rvert\rangle$ and $\langle\lvert m_0-m_1\rvert\rangle$ | shell means of power, $\langle\lvert v\rvert^2\rangle$ and $\langle\lvert m_0-m_1\rvert^2\rangle$ |
| sample size | disagreement of half-batches, $b/2$ particles per half | multiplied by $b/N$: the noise of full-data halves, $N/2$ per half |
| averaging | disagreement of first moments, noise reduced by $e$ | divided by $e$ (the exact sum of squared coefficients) |

**7.1 The target.** Equation 12 uses the FSC of two half maps of the full data set (section 3). By 6.1 that is
$P/(P+V_{\mathrm{full}})$, where $V_{\mathrm{full}}$ is the noise variance of a half map from $N/2$ particles. The gate
therefore has to produce $\rho=P/V_{\mathrm{full}}$ at $\mathrm{fudge}=1$. Write $V_b$ for the noise variance of one
half-batch direction ($b/2$ particles). Noise variance scales as the inverse number of particles, so
$V_{\mathrm{full}}=V_b\,b/N$.

**7.2 Power instead of amplitude.** For a circular complex Gaussian $z$ with $\mathbb E|z|^2=\sigma^2$,
$\mathbb E|z|=(\sqrt\pi/2)\,\sigma$. If the signal is Gaussian too, and $V$ is the noise variance of each half's
direction (so that the power version has $\rho=P/V$), RELION's ratio of shell means of amplitudes is

$$
\rho_{\mathrm{amp}}=\frac{2\langle\lvert v\rvert\rangle}{\langle\lvert m_0-m_1\rvert\rangle}=\sqrt{\frac{2P}{V}}
\qquad\text{instead of}\qquad\rho=\frac{P}{V}.
$$

The amplitude gate follows about the square root of the power SNR. With the same noise $V$ in both:

| $P/V$ | FSC $P/(P+V)$ | power gate | amplitude gate $\rho_{\mathrm{amp}}/(1+\rho_{\mathrm{amp}})$ |
| --- | --- | --- | --- |
| 0.02 | 0.020 | 0.020 | 0.167 |
| 1 | 0.500 | 0.500 | 0.586 |
| 10 | 0.909 | 0.909 | 0.817 |

The two agree only where $\sqrt{2P/V}=P/V$, at $P/V=2$. Below that the amplitude gate is too open (0.167 against
0.020), above it too closed (0.817 against 0.909). The power gate is the FSC for independent noise, which is why
`stochastic_update` takes shell means of power.

**7.3 From the batch to the full data ($b/N$).** A half-batch holds $b/2$ particles and a full-data half $N/2$, so
$V_{\mathrm{full}}=V_b\,b/N$: the disagreement of two half-batches is $N/b$ times that of two full-data halves.
RELION's gate does not know $b$ or $N$ (section 4.2), so it reads the noise of a mini-batch as if it were the noise
of the reconstruction the FSC describes, and underestimates the SNR by $N/b$ (for $N=100{,}000$ and $b=300$, by 333).

**7.4 The running average ($1/e$).** The disagreement RELION measures is that of the first moments, which average
noise down (section 5.4): $\mathbb E\nu=2eV_b$ in steady state, $e=0.0526$. This is $e$ times the noise of one
half-batch direction, so it partly offsets 7.3. The correct combined factor on the measured disagreement is the ratio
of the target $2V_{\mathrm{full}}=2V_b\,b/N$ to the measurement $2eV_b$:

$$
\text{scale}=\frac{b/N}{e}\qquad(0.057\ \text{for}\ b=300,\ N=100{,}000).
$$

The code does not use the steady-state $e$. Each iteration $t$ adds the noise of a direction from $\mathrm{count}_t$
particles (per half $\mathrm{count}_t/2$; the half difference has noise power $4s^2/\mathrm{count}_t$ for a particle
noise power $s^2$), with the coefficients of the moment, so the measured disagreement is $s^2\cdot\mathrm{weight}$ with

$$
\mathrm{weight}_t=0.9^2\,\mathrm{weight}_{t-1}+0.1^2\,\frac{4}{\mathrm{count}_t},\qquad
\mathrm{weight}_1=\frac{4}{\mathrm{count}_1},
$$

(`advance_noise_weight`). The full-data target is $2V_{\mathrm{full}}=4s^2/N$, so the scale is
$4/(N\cdot\mathrm{weight})$ (`full_data_disagreement_scale`). For a constant batch $\mathrm{weight}\to4e/b$ and this is
$(b/N)/e$. The exact sum also covers what the steady state does not: the first iterations (the moment is the
direction itself), a batch size that changes during a run, and the all-data last iteration. A voxel covered for the
first time starts from a raw direction; the code counts every voxel as averaged since iteration 1.

**7.5 The three changes together.** Take $b=300$, $N=100{,}000$, so the measured noise is $eN/b=17.5$ times
$V_{\mathrm{full}}$ and the scale is $0.057$. For a shell whose full-data half maps have $P/V_{\mathrm{full}}=s$:

| $s=P/V_{\mathrm{full}}$ | FSC of full-data halves | RELION (amplitude, measured noise) | power, measured noise | amplitude, scaled | here (power, scaled) |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 0.020 | 0.046 | 0.001 | 0.166 | 0.020 |
| 1 | 0.500 | 0.253 | 0.054 | 0.586 | 0.501 |
| 10 | 0.909 | 0.516 | 0.363 | 0.817 | 0.909 |

Only the last column is the FSC. RELION's gate, with a noise 17.5 times too large, is below the FSC except in the
weakest shells; the square root in the amplitude statistic reduces that factor to 4.2 in $\rho$ and so partly hides it.
Changing only the statistic closes the gates further (third column); changing only the scale opens them past the FSC
at low and middle SNR (fourth). This is consistent with the first experiment of section 18.16 of the algorithm
note, where the power gate with the measured noise did worse than RELION's amplitude gate (C n3 ARI .637 against
.895; T10, 1 of 10 states solved against 3) and the power gate with the full-data scale did not (C n3 ARI .924; T10,
4 or 5 states). That consistency is not a proof that the closed gate is what repaired those runs.

**7.6 What the changes do not fix, and the evidence.**

- *Independent batches.* Section 7.4 assumes the noise of different iterations' batches is independent. Batches are
  drawn from the same $N$ particles. If each iteration draws its batch independently from the data set, so that each
  particle enters each batch with probability $b/N$ (an idealization of the epoch-wise sampling), the
  variance of a half's first moment is $1+\frac{1-e}{e}\frac bN=1+18\,\frac bN$ times the independent-batch value, so
  the scaled disagreement is that many times the full-data noise: 1.05 for $b=300$, $N=100{,}000$; 19 when $b=N$. A
  scale that accounts for it, per particle, is on branch `ppca_gate_overlap` (section 18.18 of the algorithm note
  there); it is not merged here.
- *The signal estimate and shared bias.* $\hat P=\langle\lvert v\rvert^2\rangle$ and the shared-model disagreement are the same
  proxies as in RELION (sections 5.5 and 6.3).
- *The window of $v$.* The scale treats $v$ as a full-data estimate; $v$ averages the first moments with the step
  $\eta$, so its own noise window is shorter early in a run (algorithm note, section 3.3).
- *The evidence is mixed.* In section 18.16 (six synthetic datasets, 3000 iterations, $b=300$, the amplitude gate as
  RELION): the power gate with the full-data scale matches the amplitude gate on pose, ARI and R2 on C n1 and C n3,
  and is the only arm that improves T10 (4 to 5 of 10 states against 3, pose median 23.7 against 48.3 degrees). It has
  the worst loading subspace on C n1 and C n3 (49 and 39 degrees against 37 and 27) and the lowest R2 on D40 (.440
  against .616). One seed on H1, D40 and D20. The decision of November 6, 2026 was the owner's, on that mixed record.

## 8. Checks

A one-off simulation (numpy, complex Gaussian signal of power $P$, two halves $\hat\theta_h=S+e_h$ with $V=1$, a shared
model $v=0.3\,S+\text{noise}$, $4\cdot10^5$ voxels per shell, seed 0, not tracked in the repository; the gates use the
true $P$ as the signal) gave:

| $P/V$ | $\mathbb E\nu/(2V)$ | measured FSC | power gate | amplitude gate |
| --- | --- | --- | --- | --- |
| 0.02 | 0.9989 | 0.0196 | 0.0196 | 0.1668 |
| 1 | 0.9979 | 0.5003 | 0.5005 | 0.5859 |
| 10 | 0.9985 | 0.9091 | 0.9092 | 0.8173 |

The disagreement is $2V$ whatever the shared model is, the power gate equals the measured FSC, and the amplitude gate
equals $\dfrac{\sqrt{2P/V}}{1+\sqrt{2P/V}}$. An exponential average with $l=0.9$ of unit-variance noise has variance
0.0528 against $e=0.0526$.

A second simulation (also untracked: $3\cdot10^5$ voxels, seed 1, 150 iterations, $b=300$, $N=100{,}000$,
independent half-batch noise of power $V_{\mathrm{full}}N/b$, first moments $0.9\,\mathrm{first}+0.1\,d$, the model equal
to the signal, the scale from the weight recursion of 7.4) gave the table of 7.5, with scale 0.0570 against the closed
form $(b/N)/e=0.0570$. The closed forms of 7.5 (evaluated with a one-line calculation from $\rho=\sqrt{2s/17.54}$, $s/17.54$,
$\sqrt{2s}$ and $s$, each through $\rho/(1+\rho)$) agree with it to the third decimal.

The tracked checks are in
[test_gate_statistic.py](../../tests/unit/ppca_initial_model/test_gate_statistic.py):
`test_the_gate_estimates_the_half_map_fsc` (power gate against $s/(1+s)$),
`test_the_noise_weight_is_the_sum_of_squared_moment_coefficients_over_the_batches` (7.4),
`test_for_a_constant_batch_the_scale_is_the_batch_share_over_the_running_average_noise_reduction` ($(b/N)/e$) and
`test_the_scaled_disagreement_recovers_the_noise_of_two_full_data_half_maps_through_the_update` (the scale through
`stochastic_update`). No experiment here compares the gate with the FSC of independently reconstructed half maps on
real data.
