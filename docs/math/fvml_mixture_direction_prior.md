# Fitting a mixture-of-FvML viewing-direction prior by a nested EM

**Status (2026-11-20, branch `fvml_refinement`).** Derived, numerically checked (section 10), implemented (section
8) with a point-atom and a cell-mass fit (section 7), wired into K=1 refinement as an opt-in prior,
`relax refine --fvml-direction-prior N` (section 9), and benchmarked on the pose-prior sweep (section 10). The implementation is
[`relax/sampling/fvml_mixture.py`](../../relax/sampling/fvml_mixture.py) (the EM) and
[`relax/sampling/fvml_direction_prior.py`](../../relax/sampling/fvml_direction_prior.py) (the refinement's fit), with the
input source [`relax/parity/fvml_prior_source.py`](../../relax/parity/fvml_prior_source.py); its tests are
[`tests/unit/test_fvml_mixture_em.py`](../../tests/unit/test_fvml_mixture_em.py) and
[`tests/unit/test_fvml_direction_prior.py`](../../tests/unit/test_fvml_direction_prior.py). The document is self-contained: it
formalizes the mixture-of-FvML pose prior proposed in the team's note on EM for cryo-EM reconstruction (May 2026, a PDF
outside the repository) but does not require reading it.

## 1. Summary

In EM for cryo-EM, the prior over particle orientations has its own hyperparameters, learned in the M-step. RELION and
RELAX learn the prior over *viewing directions* as a histogram on a direction grid. This document replaces the
histogram by a smooth, low-dimensional model, a uniform component plus a few von Mises–Fisher ("FvML") lobes on the sphere,
and shows how to fit it.

- **The fit is an EM inside the M-step (a sub-EM), exact and closed form.** The M-step objective for the viewing-direction
  prior is the log-likelihood of a mixture on the sphere whose data are the posterior masses over directions. The
  weights, mean directions and concentrations update in closed form, except each concentration $\kappa_j$, which is
  the root of a strictly monotone scalar equation (section 6).
- **Only a mass per direction is needed.** The in-plane angle, the shifts and the number of particles drop out; the
  input is one mass per direction of the grid, the quantity `collapse_rotation_posterior_to_direction_prior` in
  [`relax/sampling/orientation_priors.py`](../../relax/sampling/orientation_priors.py) forms before it normalizes.
- **It is a valid inner loop of the outer EM.** Any number $s\ge1$ of sub-iterations from a warm start keeps the outer EM's
  monotone increase of the log-posterior (a generalized EM); $s=1$ is the ordinary EM on the model augmented with a component
  label (section 6.2).
- **Three things the objective does not give for free.**
  1. Without a bound on the concentrations the objective is unbounded above: a lobe collapses onto one direction atom
     (section 6.3). In a toy case with one dominant atom EM reached the cap in four iterations (section 10), so the
     problem needs a cap or a prior on $\kappa$, tied to the direction grid (section 7).
  2. The maximum is local, so initialization matters (sections 8 and 10, measured).
  3. Atoms sit at grid nodes, which biases $\kappa$ low by $1/(1+\kappa\delta^2/12)$ ($\delta$ the node spacing) and leaves a
     quadrature error in the density's normalization (section 7).
- **Two generalizations are included**: a finite set of orthogonal maps $\mathcal G$ acting on the sphere (antipodal
  symmetry is $\{I,-I\}$; a point group such as C3 is the same construction), and Dirichlet and exponential hyperpriors on
  the weights and the concentrations.

## 2. Formulas at a glance

Notation (details in sections 3–5). The data are *atoms* $(v_m,\rho_m)_{m=1}^{D}$: unit vectors $v_m\in S^2$ (a direction
grid) and masses $\rho_m\ge0$ (the posterior mass the E-step puts on direction $v_m$, summed over particles), $N=\sum_m\rho_m$.
The mixture has $n_r$ components: component 1 is uniform with weight $\pi_1$; components $j=2,\dots,n_r$ are lobes with weight $\pi_j$,
mean direction $\eta_j\in S^2$ and concentration $\kappa_j\ge0$ (the code indexes from 0). $\mathcal G$ is a finite set of orthogonal
$3\times3$ maps, $n_g=|\mathcal G|$, default $\{I\}$.

$$
f(v;\eta,\kappa)=\frac{\kappa}{4\pi\sinh\kappa}\,e^{\kappa\,\eta^\top v},\qquad
\log f=\log\kappa-\log2\pi-\log\!\big(1-e^{-2\kappa}\big)+\kappa(\eta^\top v-1) \tag{E1}
$$

$$
\lambda(v)=\frac{\pi_1}{4\pi}+\sum_{j=2}^{n_r}\pi_j\,\frac1{n_g}\sum_{g\in\mathcal G}f(v;\,g\eta_j,\kappa_j),\qquad \sum_j\pi_j=1 \tag{E2}
$$

$$
Q_{\rm rot}(\Theta)=\sum_{m=1}^{D}\rho_m\log\lambda(v_m)+\log p(\Theta)\qquad\text{(the objective maximized by the sub-EM)} \tag{E3}
$$

E-step, with $a_1=\pi_1/4\pi$ and $a_{(j,g)}(v)=\pi_j\,n_g^{-1}f(v;g\eta_j,\kappa_j)$:

$$
r_{m1}=\frac{a_1}{\lambda(v_m)},\qquad r_{mjg}=\frac{a_{(j,g)}(v_m)}{\lambda(v_m)} \tag{E4}
$$

Sufficient statistics and M-step (Dirichlet$(a_1,\dots,a_{n_r})$ hyperprior on $\pi$, $a_j=1$ flat; $\kappa_j\le\kappa_{\max}$):

$$
N_1=\sum_m\rho_m r_{m1},\quad N_j=\sum_m\rho_m\sum_g r_{mjg},\quad
S_j=\sum_m\rho_m\sum_g r_{mjg}\,g^{\top}v_m,\quad \bar R_j=\frac{\lVert S_j\rVert}{N_j} \tag{E5}
$$

$$
\pi_j=\frac{N_j+a_j-1}{N+\sum_l(a_l-1)},\qquad \eta_j=\frac{S_j}{\lVert S_j\rVert},\qquad
\kappa_j=\min\{A^{-1}(\bar R_j),\kappa_{\max}\},\qquad A(\kappa)=\coth\kappa-\frac1\kappa \tag{E6}
$$

$$
A'(\kappa)=\frac1{\kappa^2}-\frac1{\sinh^2\kappa}\in(0,\tfrac13],\qquad
A^{-1}(\bar R)\ \text{by Newton from}\ \kappa_0=\frac{\bar R(3-\bar R^2)}{1-\bar R^2} \tag{E7}
$$

Exponential hyperprior $p(\kappa_j)\propto e^{-\kappa_j/\kappa_0}$ instead of the cap:

$$
A(\kappa_j)=\bar R_j-\frac1{\kappa_0N_j}\qquad(\kappa_j=0\ \text{if the right side is}\le0) \tag{E8}
$$

Grid rules, with $\delta=\sqrt{4\pi/D}$ the node spacing of a whole-sphere equal-area grid of $D$ nodes:

$$
\kappa_{\max}=\frac1{(c\,\delta)^2}\ (\text{lobe width}\ \sigma=\kappa^{-1/2}\ge c\,\delta),\qquad
\kappa_{\rm fit}\approx\frac{\kappa}{1+\kappa\delta^2/12} \tag{E9}
$$

## 3. Setting: EM for cryo-EM and where the pose prior enters

**Model.** A dataset is $n$ images $y_1,\dots,y_n$ of copies of one particle in random orientations. Each image is a noisy
2-D projection of an unknown 3-D density $V$, seen after a rotation $R_i\in SO(3)$ and an in-plane shift $t_i\in\mathbb R^2$ and
filtered by the known contrast transfer function (CTF). In Fourier space,

$$
y_i=\mathcal P(\phi_i)\,V+\varepsilon_i,\qquad \phi_i=(R_i,t_i),
$$

where $\mathcal P(\phi)$ is the known linear operator that rotates $V$, takes the central slice (Fourier slice theorem), applies the
phase factor of the shift and the CTF, and $\varepsilon_i$ is independent complex Gaussian noise. This gives a Gaussian
density $p(y_i\mid\phi,V,\Theta_0)$, with noise hyperparameters $\Theta_0$. The poses $\phi_i$ are unknown: they are latent
variables, independent across particles, with a prior $p(\phi\mid\Theta_\phi)$ that does not depend on $V$. The volume has its own prior
$p(V\mid\Theta_V)$. Integrating the poses out, particle $i$ has marginal likelihood
$p(y_i\mid V,\Theta)=\int p(y_i\mid\phi,V,\Theta_0)\,p(\phi\mid\Theta_\phi)\,d\phi$, and the goal is the
maximum a posteriori estimate of $(V,\Theta)$, $\Theta=(\Theta_0,\Theta_V,\Theta_\phi)$:

$$
L(V,\Theta)=\sum_{i=1}^n\log p(y_i\mid V,\Theta)+\log p(V\mid\Theta_V)+\log p(\Theta),\qquad p(\Theta)=p(\Theta_0)\,p(\Theta_V)\,p(\Theta_\phi).
$$

**EM.** Given the current estimates $(V^{(k)},\Theta^{(k)})$, the E-step computes each particle's posterior over poses (its
*responsibilities*),

$$
w_i(\phi)=\frac{p(y_i\mid\phi,V^{(k)},\Theta_0^{(k)})\,p(\phi\mid\Theta_\phi^{(k)})}{\int p(y_i\mid\phi',V^{(k)},\Theta_0^{(k)})\,p(\phi'\mid\Theta_\phi^{(k)})\,d\phi'} ,
$$

and the M-step maximizes the expected complete-data log-posterior

$$
Q(V,\Theta)=\sum_{i=1}^n\int w_i(\phi)\,\log p(y_i,\phi\mid V,\Theta)\,d\phi+\log p(V\mid\Theta_V)+\log p(\Theta).
$$

*Why this works.* For any weights $w_i$ that are probability densities, Jensen's inequality gives
$\log\int p(y_i,\phi\mid V,\Theta)\,d\phi\ge\int w_i\log\frac{p(y_i,\phi\mid V,\Theta)}{w_i}\,d\phi$, with equality when $w_i$ is the posterior at
$(V,\Theta)$. Hence $L(V,\Theta)\ge Q(V,\Theta)+\mathcal E$, with $\mathcal E=-\sum_i\int w_i\log w_i$ independent of $(V,\Theta)$ and equality at the current
estimate: any $(V,\Theta)$ with larger $Q$ has larger $L$ (Dempster, Laird and Rubin 1977).

**The pose-prior part decouples.** $\log p(y_i,\phi\mid V,\Theta)=\log p(y_i\mid\phi,V,\Theta_0)+\log p(\phi\mid\Theta_\phi)$, the likelihood depends on
$\Theta_0$ only, the volume prior on $\Theta_V$ only, the pose prior on $\Theta_\phi$ only. So $Q=Q_{\rm vol}(V,\Theta_0,\Theta_V)+Q_{\rm pose}(\Theta_\phi)$, maximized separately.
$Q_{\rm vol}$ gives the usual volume and noise updates and is not discussed here. The pose-prior objective is

$$
Q_{\rm pose}(\Theta_\phi)=\sum_{i=1}^n\int w_i(\phi)\,\log p(\phi\mid\Theta_\phi)\,d\phi+\log p(\Theta_\phi). \tag{E10}
$$

In an implementation $w_i$ lives on a quadrature of the pose space (a grid of viewing directions on the sphere, in-plane angles and
translations), $w_i\to\{w_{i,s}\}_s$ with $\sum_sw_{i,s}=1$; RELION-style implementations also drop negligible weights.

**The viewing-direction prior.** The prior factorizes into a shift part and a rotation part,
$p(\phi\mid\Theta_\phi)=p(t\mid\Theta_{21})\,p(R\mid\Theta_{22})$. The shift prior (typically a Gaussian) is not discussed.
For the rotation, assume it is *in-plane uniform*: composing $R$ with a rotation about the viewing axis does not change its density
(equivalently, one may augment the dataset with randomly rotated copies of each image). Then the prior depends on $R$ only through the
viewing direction $v=Re_3\in S^2$, $e_3=(0,0,1)^\top$:

$$
p(R\mid\Theta_{22})\,dR=\lambda(v\mid\Theta_{22})\,\frac{d\psi}{2\pi}\,dS(v), \tag{E11}
$$

where $\psi$ is the in-plane angle, $dS$ the surface measure of $S^2$ (total $4\pi$) and $\lambda$ a probability density on the sphere. Sampling a
rotation is then: draw $v\sim\lambda$, draw $\psi$ uniformly, and compose. Which axis of the rotation matrix is called the viewing direction is a convention of
the caller: the optimizer below only sees unit vectors, which must be the grid points on which the E-step accumulates its masses.

**What RELION does, and what is proposed.** RELION's learned `pdf_direction` is a histogram of $\lambda$ on the HEALPix direction grid: each iteration it is set to the posterior
mass per direction (the maximizer of the objective below over all probability vectors, section 4) and it is reset to uniform whenever the angular sampling order
changes. The proposal is to restrict $\lambda$ to a mixture of a uniform component and a few FvML lobes, a smooth model with $4(n_r-1)$ free parameters ($n_r-1$ weights and, per lobe, two for the direction and one for the concentration) that is a function on the
sphere and not on a grid.

**Assumptions used below.** (A1) The hyperprior factorizes further, $p(\Theta_\phi)=p(\Theta_{21})\,p(\Theta_{22})$, so that the shift and rotation parts of (E10) separate.
(A2) The rotation prior is in-plane uniform, (E11). (A3) $\lambda$ is the mixture of section 5 with a given number $n_r$ of components, the first one uniform. (A4) The weights $w_i$ are the
exact posterior or a quadrature of it.

## 4. Reduction to a weighted mixture likelihood

**Lemma 1.** Under A1 and A2, $Q_{\rm pose}=Q_{\rm shift}(\Theta_{21})+Q_{\rm rot}(\Theta_{22})$, where $Q_{\rm shift}(\Theta_{21})=\sum_i\int w_i\log p(t\mid\Theta_{21})\,d\phi+\log p(\Theta_{21})$ and

$$
Q_{\rm rot}(\Theta_{22})=\int_{S^2}\log\lambda(v\mid\Theta_{22})\,\rho(dv)+\log p(\Theta_{22})+\text{const},\qquad
\rho:=\sum_{i=1}^n\rho_i,\quad \rho_i:=\big(\phi\mapsto Re_3\big)_{\#}w_i .
$$

*Proof.* $\log p(\phi\mid\Theta_\phi)=\log p(t\mid\Theta_{21})+\log p(R\mid\Theta_{22})$, so (E10) splits (A1 for the hyperprior). By (E11) the density of
$p(R\mid\Theta_{22})\,dR$ with respect to the $\Theta$-free measure $dS(v)\,d\psi/2\pi$ is $\lambda(Re_3\mid\Theta_{22})$; changing to the measure $dR$ adds a constant that does not depend on $\Theta$.
The integrand $\log\lambda(Re_3)$ depends on $\phi=(R,t)$ only through $v=Re_3$, so integrating against $w_i$ equals integrating $\log\lambda$ against the pushforward
$\rho_i$ of $w_i$ to the sphere. $\square$

$\rho$ is a nonnegative measure on $S^2$ of total mass $n$: the aggregated posterior over viewing directions, with the shifts and in-plane angles integrated out. For a
$\rho$ with a density, maximizing $Q_{\rm rot}$ without a hyperprior is minimizing $\mathrm{KL}(\rho/n\,\Vert\,\lambda_\Theta)$: the M-step is the moment projection of the aggregated direction posterior onto the
mixture family.

**Atoms.** If the quadrature nodes $\phi_{i,s}$ have viewing directions on a grid $\{v_m\}_{m=1}^{D}$ (HEALPix pixel centres), then $\rho=\sum_m\rho_m\delta_{v_m}$ with
$\rho_m=\sum_i\sum_{s:\,\mathrm{dir}(s)=m}w_{i,s}$ and (E3) follows, with $N=\sum_m\rho_m=n$. This is a weighted mixture log-likelihood with $D$ points, whatever $n$ is.
Without any restriction on $\lambda$ other than being a probability vector $p_m$ on the grid, $\sum_m\rho_m\log p_m$ is maximized by $p_m=\rho_m/N$: this is RELION's histogram. The mixture is a
smooth parametric restriction of it, $p_m\propto\lambda(v_m)$.

## 5. The model

**The FvML density.** On the sphere, take the density proportional to $e^{\kappa\,\eta^\top v}$: it peaks at the mean direction $\eta\in S^2$ and $\kappa\ge0$ controls how sharply.
Its normalizing constant follows from putting $\eta$ at the pole: with $x=\eta^\top v=\cos\theta$, $dS=dx\,d\varphi$, so

$$
\int_{S^2}e^{\kappa\eta^\top v}\,dS(v)=2\pi\int_{-1}^{1}e^{\kappa x}\,dx=\frac{4\pi\sinh\kappa}{\kappa},
$$

which gives $f$ of (E1); $\kappa=0$ is the uniform density $1/4\pi$. For large $\kappa$, $\kappa(\cos\theta-1)\approx-\kappa\theta^2/2$, so the lobe is close to an isotropic Gaussian of standard
deviation $\sigma=\kappa^{-1/2}$ radians in the tangent plane.

Two facts about $f$ drive everything below. Under $f$ the cosine $x=\eta^\top v$ has density $\propto e^{\kappa x}$ on $[-1,1]$ (for uniform $v$, $x$ is uniform by Archimedes' theorem), so its mean is

$$
A(\kappa):=\mathbb E_\kappa[x]=\frac{\int xe^{\kappa x}dx}{\int e^{\kappa x}dx}=\frac{d}{d\kappa}\log\frac{\sinh\kappa}{\kappa}=\coth\kappa-\frac1\kappa,\tag{E12}
$$

the *Langevin function*; and the variance of $x$ is $A'(\kappa)=1/\kappa^2-1/\sinh^2\kappa>0$, so $A$ is a strictly increasing bijection $[0,\infty)\to[0,1)$.
For numerical work use $\sinh\kappa=\tfrac12e^\kappa(1-e^{-2\kappa})$, which gives the second form of (E1): it does not overflow (checked up to $\kappa=10^8$) and tends to $-\log4\pi$ as $\kappa\to0$.

**The mixture and the symmetry maps.** (E2) is a weighted sum of one uniform density and $n_r-1$ lobes, each lobe averaged over a finite set $\mathcal G\subset O(3)$ of orthogonal maps (default $\{I\}$; then (E2) is the plain
mixture). For a *group* $\mathcal G$, $\lambda$ is invariant, $\lambda(hv)=\lambda(v)$, because $f(hv;g\eta,\kappa)=f(v;h^{-1}g\eta,\kappa)$ and $g\mapsto h^{-1}g$ permutes $\mathcal G$. Two uses:

- *Antipodal symmetry*, $\mathcal G=\{I,-I\}$, imposes $\lambda(v)=\lambda(-v)$ (one may want this when the dataset is augmented with reflected images, which correspond to opposite viewing directions). Then $\pi_j$ is the weight of the *pair* of lobes $\pm\eta_j$, each carrying $\pi_j/2$,
  so that the weights still sum to one.
- *Point-group symmetry* of the particle (for example C3): the viewing directions are only defined up to the group, and the E-step works on an asymmetric unit. Because $\lambda$ is invariant, fitting the atoms of the asymmetric unit is
  the same as fitting all $n_g$ copies of each atom at mass $\rho_m/n_g$ (checked, section 10).

## 6. The sub-EM

**Latent state.** Introduce for each atom a latent label $k\in\mathcal K=\{1\}\cup\{(j,g):j\ge2,\,g\in\mathcal G\}$ with joint weights $a_k(v)$ as in (E4), so $\lambda=\sum_{k}a_k$ and the posterior of the label is the responsibility $r_{mk}=a_k(v_m)/\lambda(v_m)$ of (E4)
(computed as a softmax in the log domain).

**Lemma 2 (minorizer).** For any parameters $\Theta'$ (where the responsibilities are evaluated) and all $\Theta$,
$Q_{\rm rot}(\Theta)\ge B(\Theta\mid\Theta')+H(\Theta')$, with equality at $\Theta=\Theta'$, where

$$
B(\Theta\mid\Theta')=\sum_m\rho_m\sum_{k}r_{mk}(\Theta')\,\log a_k(v_m;\Theta)+\log p(\Theta),\qquad H(\Theta')=-\sum_m\rho_m\sum_kr_{mk}\log r_{mk}.
$$

*Proof.* For positive numbers $a_k$ and a probability vector $r$, Jensen's inequality gives $\log\sum_ka_k\ge\sum_kr_k\log(a_k/r_k)$, with equality iff $r_k\propto a_k$. Apply it at each atom with $r=r_{m\cdot}(\Theta')$, sum with weights
$\rho_m$, and add the hyperprior, which Jensen does not touch. $\square$

**Proposition 3 (M-step).** Let the hyperprior on $\pi$ be Dirichlet$(a_1,\dots,a_{n_r})$ with $a_j\ge1$ ($a_j=1$ is flat), let each $\kappa_j$ be restricted to $[0,\kappa_{\max}]$ (a uniform hyperprior on that interval) and the rest flat.
Then the maximizer of $B(\cdot\mid\Theta')$ is (E6) with the statistics (E5), where $A^{-1}(1):=\infty$, $\kappa_j=0$ if $\bar R_j=0$ (then $\eta_j$ is arbitrary), and $(\eta_j,\kappa_j)$ is left unchanged if $N_j=0$.

*Derivation.* The terms of $B$ split by parameter.

- *Weights.* They are $\sum_jN_j\log\pi_j+\sum_j(a_j-1)\log\pi_j$ (data plus Dirichlet), maximized on the simplex $\sum\pi_j=1$ by a Lagrange multiplier: $\pi_j\propto N_j+a_j-1$, the first formula of (E6).
- *Directions and concentrations.* The terms of $B$ in $(\eta_j,\kappa_j)$ are $N_j[\log\kappa-\log(4\pi\sinh\kappa)]+\kappa\,\eta_j^\top S_j$, using $(g\eta)^\top v=\eta^\top g^\top v$ (a lobe's images act by moving the datum into the lobe's frame, which is where $g^\top v$ in $S_j$ comes
  from). For fixed $\kappa>0$ the Cauchy–Schwarz inequality maximizes $\eta_j^\top S_j$ at $\eta_j=S_j/\lVert S_j\rVert$, with value $\lVert S_j\rVert$.
- *Concentration.* What remains is $\varphi(\kappa)=N_j[\kappa\bar R_j-\log(\sinh\kappa/\kappa)]$ + const, $\bar R_j=\lVert S_j\rVert/N_j\in[0,1]$ the mean cosine between the data and the mean direction. Since $\log(\sinh\kappa/\kappa)=\log\int_{-1}^{1}e^{\kappa x}dx/2$ is a cumulant generating function it is strictly convex, so
  $\varphi$ is strictly concave with $\varphi'(\kappa)=N_j(\bar R_j-A(\kappa))$ by (E12). The maximizer on $[0,\kappa_{\max}]$ is the root of $A(\kappa)=\bar R_j$, clipped: $\min\{A^{-1}(\bar R_j),\kappa_{\max}\}$. $\square$

The $(\eta_j,\kappa_j)$ block is therefore maximized *exactly*, not merely improved, and the update is moment matching: $\kappa_j$ makes the lobe's mean cosine $A(\kappa_j)$ equal to the data's $\bar R_j$.

**Solving $A(\kappa)=\bar R$.** Newton's method with the slope $A'$ of (E7), started from the $p=3$ approximation of Banerjee et al. (2005), $\kappa_0=\bar R(3-\bar R^2)/(1-\bar R^2)$, and a bisection bracket as safeguard.
$A$ and $A'$ are evaluated by their series below $\kappa=0.1$ ($A=\kappa/3-\kappa^3/45+2\kappa^5/945-\kappa^7/4725$, $A'=1/3-\kappa^2/15+2\kappa^4/189-\kappa^6/675$) and by the closed forms otherwise. Accuracy is in section 10.

**Other hyperpriors.** An exponential prior $p(\kappa_j)\propto e^{-\kappa_j/\kappa_0}$ replaces the cap and adds $-\kappa/\kappa_0$ to $\varphi$, which shifts the target of the equation to $A(\kappa)=\bar R_j-1/(\kappa_0N_j)$ (E8): still one scalar root, and it keeps the objective
bounded (section 6.3) without a grid-dependent cap.

### 6.1 Monotone ascent

One E-step plus one M-step gives $\Theta^+$ with $Q_{\rm rot}(\Theta^+)\ge B(\Theta^+\mid\Theta')+H(\Theta')\ge B(\Theta'\mid\Theta')+H(\Theta')=Q_{\rm rot}(\Theta')$ (Lemma 2, then Proposition 3). The sequence $Q_{\rm rot}(\Theta^s)$ is non-decreasing.

### 6.2 Compatibility with the outer EM

The outer EM needs only $Q_{\rm pose}(\Theta_\phi^{(k+1)})\ge Q_{\rm pose}(\Theta_\phi^{(k)})$ to raise $L$: by section 3, any $(V,\Theta)$ with larger $Q$ has larger $L$ (a generalized EM, Dempster et al. 1977), and $Q_{\rm pose}$ can be improved without
touching $Q_{\rm vol}$. So **any $s\ge1$ sub-iterations from the warm start $\Theta_{22}^{(k)}$ are valid**; the sub-EM need not converge. With $s=1$ the scheme is the ordinary EM on the model augmented with a component label $z_i$ per particle:
the joint posterior of $(\phi_i,z_i)$ is $w_i(\phi)\,r_z(Re_3)$ and its expected complete-data log-posterior is $B(\cdot\mid\Theta^{(k)})$.

### 6.3 The unregularized objective is unbounded

**Proposition 4.** Let $\rho$ have an atom $v_a$ with $\rho_a>0$, let $n_r\ge2$, and take $\pi=(\tfrac12,\tfrac12,0,\dots)$, $\eta_2=v_a$, $\kappa_2=\kappa$. Then
$Q_{\rm rot}\ge\rho_a\log\frac{\kappa}{4\pi n_g}+(N-\rho_a)\log\frac1{8\pi}\to\infty$ as $\kappa\to\infty$.

*Proof.* $f(v_a;v_a,\kappa)=\kappa/(2\pi(1-e^{-2\kappa}))\ge\kappa/2\pi$ and the $g=I$ term gives $\lambda(v_a)\ge\pi_2\kappa/(2\pi n_g)$; every other atom has $\lambda\ge\pi_1/4\pi$ from the uniform component. $\square$

Quadrature always produces atoms, so $\sup Q_{\rm rot}=+\infty$ and the maximum-likelihood estimate does not exist; the iteration is attracted to the singularity (section 10: $\kappa$ goes $200\to340\to1.5\!\times\!10^{3}\to1.1\!\times\!10^{6}\to$ cap in four
iterations from a start on the heaviest atom). The remedy is part of the model, through the hyperprior $\log p(\Theta)$: the interval $\kappa\le\kappa_{\max}$ or the exponential prior. Section 7 gives a rule for the cap.

### 6.4 Convergence

With $\kappa_j\le\kappa_{\max}$ the parameter space, in the smooth coordinates $\theta_j=\kappa_j\eta_j$ ($\lVert\theta_j\rVert\le\kappa_{\max}$) and $\pi$ on the simplex, is compact, and $Q_{\rm rot}$ is continuous with values in $[-\infty,\infty)$.
Then $Q_{\rm rot}(\Theta^s)$ converges to a finite limit, and every limit point of $\Theta^s$ is a fixed point of the EM map and satisfies the first-order (KKT) conditions of the capped problem (Wu 1983; the gradients of $B(\cdot\mid\Theta')$ and $Q_{\rm rot}$
agree at $\Theta'$, so a fixed point is stationary). **Not guaranteed:** convergence to a global maximum, to a local maximum rather than a saddle, or of the parameter sequence itself. The likelihood is multimodal and invariant under permuting
components $2,\dots,n_r$ and under replacing $\eta_j$ by $g\eta_j$; compare fits only after sorting components.

### 6.5 Practical behaviour

A component started away from any concentration of mass loses its weight geometrically: in four inspected random starts $\pi_j$ fell to $\approx10^{-15}$ within 22 iterations and the fit ended at the uniform-only solution, a fixed point from which
it does not recover in practice. (A component with $N_j=0$ exactly keeps its $\eta_j,\kappa_j$ and gets $\pi_j=0$ unless the Dirichlet pseudo-count keeps it positive.) An over-specified $n_r$ converges slowly (extra components settle at $\kappa\approx0.2$–$0.3$ with 2–3 % of the
mass and a nearly flat objective). Neither affects the guarantees above.

## 7. Discretization

The atoms are grid nodes, but (E3) evaluates a *density* at them. Three measured consequences (section 10):

1. **Normalization.** $\sum_mf(v_m)\,\Delta S$ (with $\Delta S=4\pi/D$) differs from 1: HEALPix pixel centres are not a quadrature rule. The worst error over 40 random $\eta$ is $\le10^{-3}$ at order 5–6 once
   $\sigma/\delta\ge0.75$, $6\times10^{-3}$–$3\times10^{-2}$ at order 4 and $4\times10^{-3}$–$8\times10^{-3}$ at order 3 for $\sigma/\delta\in[0.75,3]$, and $0.5$–$0.8$ at $\sigma/\delta=0.3$. A pmf handed to an E-step must be normalized
   over the grid, as `pdf_direction` is.
2. **Bias of $\kappa$.** For a narrow lobe the density is nearly Gaussian with variance $1/\kappa$ per tangent axis. Mass quantized to node centres adds the variance of a uniform cell, about $\delta^2/12$ per axis (a cell of side $\delta$; HEALPix cells are equal-area, not square), and independent variances add
   (Sheppard's correction for grouped data, in reverse): $1/\kappa_{\rm fit}\approx1/\kappa+\delta^2/12$, which is (E9). Measured ratios agree to 1 % for $\kappa\delta^2\le2$ and to 1.5 % at 4. Beyond that the formula overstates the bias
   (single runs: fitted/true 0.76 at $\kappa\delta^2=8.2$ and 0.43 at 30.7, against 0.60 and 0.28 predicted), and the fit no longer resolves the width.
3. **Cap.** A lobe narrower than the node spacing is resolved by one or two atoms, the situation of Proposition 4. Requiring $\sigma=\kappa^{-1/2}\ge c\,\delta$ gives $\kappa_{\max}=1/(c\delta)^2$ (E9). Starting points to test: $c=1$ ($\kappa_{\max}=1/\delta^2$, about 244 at
   HEALPix order 4 and 978 at order 5) and $c=\tfrac12$ ($4/\delta^2$), which follows narrower peaks at the price of a larger bias. These are heuristics, not derived thresholds.

**The cell-mass (grouped-data) likelihood.** Treating $\rho_m$ as the mass of the whole cell $m$ instead of a point gives the objective
$\sum_m\rho_m\log P_m$ with $P_m=\int_{{\rm cell}\,m}\lambda\,dS$. With the cell integral replaced by the midpoint rule on $C$ equal-area children $v_{m,1..C}$ of a finer grid,
$P_m\approx\frac{4\pi}{DC}\sum_c\lambda(v_{m,c})=\frac{4\pi}{DC}\sum_c\sum_ka_k(v_{m,c})$, this is again a latent-variable model, now with the latent pair (child $c$, component $k$). Its EM is the M-step (E6) on the children,
with the masses and responsibilities of the E-step
$$
r_{m,c,k}=\frac{a_k(v_{m,c})}{\sum_{c'}\sum_{k'}a_{k'}(v_{m,c'})},\qquad\text{child mass }\rho_m\sum_kr_{m,c,k},
$$
i.e. each cell's mass is spread over its children in proportion to the current density, then over the components. Ascent and the other properties of section 6 carry over unchanged. Two consequences, both measured (section 10):
the $\kappa$ bias of (2) shrinks to the bias at the *children's* spacing, $1/(1+\kappa\delta^2/(12C))$, and with $C=1$ the fit is exactly the point-atom fit (the objective shifted by $N\log(4\pi/D)$). The objective is still unbounded (a lobe can
collapse onto one child), so the cap stays; with a cap of $c=1$ on the fit grid and $C=16$ the remaining bias is $1/(1+1/192)$. The cost is $C$ times that of the point fit.

## 8. Algorithm and implementation

```
input   atoms (v_m, rho_m), m=1..D        # unit vectors (D,3), masses (D,), one half-set / one class
        n_r, group G (default {I}), kappa_max, Dirichlet a_j (default 1), s_max, tol
init    Theta^0 = previous Theta (warm start)  or  weighted spherical k-means++ (below)
for s = 0, 1, ..., s_max-1:
    E:  log a_k(v_m) for all k;  log lambda_m = logsumexp_k;  r_mk = exp(log a_k - log lambda_m)
        Q_s = sum_m rho_m log lambda_m
    M:  N_j, S_j; pi_j; eta_j = S_j/|S_j|; kappa_j = min(A^{-1}(|S_j|/N_j), kappa_max)
    stop if Q_s - Q_{s-1} <= tol * sum_m rho_m        # tol in nats per particle
output  Theta, [Q_s]                      # the history is part of the result
```

- **Cold start.** Weighted spherical k-means++ seeding ($n_r-1$ centres drawn with probability $\propto\rho_m(1-\max\cos)^2$, the cosine taken to the nearest seed and best over the symmetry maps), ten Lloyd steps on the cosine, then $\eta_j$ the cluster's resultant direction, $\kappa_j=A^{-1}(\bar R_j)$ of the cluster and
  $\pi$ from the cluster masses with $\pi_1=0.2$ (the choices of the checks in section 10). An empty cluster keeps its seed with a small weight share. Run several seeds and keep the highest $Q$; the measured success rates are in section 10.
- **Cost.** $O(D\,n_r\,n_g)$ per sub-iteration. Measured (4 threads, 50 iterations, one run each): 3.4 ms per iteration at $D=12\,288$ (order 5, $n_r=4$); 20.9 ms at $D=49\,152$ (order 6, $n_r=6$) and 27.7 ms at $n_r=10$; 69.6 ms at order 6, $n_r=6$ with C3 (three maps).
- **Implementation** ([`relax/sampling/fvml_mixture.py`](../../relax/sampling/fvml_mixture.py)): `FvmlMixture` holds $\Theta_{22}$ and the symmetry maps; `init_fvml_mixture` is the cold start above; `fit_fvml_mixture` runs the loop and returns a `FitResult` (mixture, the objective at every iterate, whether it converged); `fit_fvml_mixture_cells` is the cell-mass fit of section 7 on `(D, C, 3)` child directions, with the same M-step and loop;
  `log_density` evaluates $\log\lambda$; `inverse_langevin` solves $A(\kappa)=\bar R$; `kappa_max_for_grid` is the cap rule (E9) with $c$ the minimum width in node spacings, $c=0.25$ by default. The fit's `kappa_max` has no default (an uncapped fit is unbounded), the Dirichlet concentration is one scalar,
  the returned objective includes the Dirichlet term so it is the quantity that never decreases, and the loop raises if it does. The maps may be any finite set of orthogonal matrices; only a group makes $\lambda$ invariant.
- **Precision.** Host float64, cast once at the boundary, as `orientation_priors.py` does for the other priors; a 3-vector and a few scalars per component do not justify float32, and $A^{-1}$ near $\bar R\to1$ loses $\approx10^{-16}/(1-\bar R)$ in relative accuracy.

## 9. Interface to the refinement loop

- **Input.** One half-set's direction masses $\rho_m$ on the grid of its E-step: per half, because the gold-standard halves keep separate priors. With the flat Dirichlet ($\alpha=1$) the masses' scale does not matter, so the refinement passes the normalized sum (RELION's learned `pdf_direction`), and the stopping rule stays per particle.
- **Output.** $\Theta_{22}$. The E-step needs $\log\lambda(v)$ at its node directions, up to a constant for K=1 (the constant cancels in each particle's normalization); K>1 would need the per-class normalization over the grid. Because $\lambda$ is a function
  on the sphere and not on a grid, a prior learned at one HEALPix order can be evaluated at the next one: RELION and RELAX reset `pdf_direction` to uniform at every order change (`relion_direction_log_priors_for_half`), which this prior does not
  need.
- **As implemented** (`relax refine --fvml-direction-prior N`, K=1, C1, single-particle, one GPU; 2026-11-19). After every numbered M-step, [`FvmlDirectionPriorSource.learned_direction_priors`](../../relax/parity/fvml_prior_source.py) fits each half's learned `pdf_direction` with [`fit_direction_prior`](../../relax/sampling/fvml_direction_prior.py): the cell-mass fit on the masses summed to order $\min(\text{order},5)$, cells of 16 children, $c=0.25$, a half's first fit the best of five k-means++ starts and every later one warm-started from its previous fit (the settings section 10 and its replays support; five lobes is the owner's choice of 2026-11-19, from the replays). The half's prior is then the mixture's cell probabilities, which every later expectation of the half scores with, on whatever order it samples, through `InputSource.direction_log_prior`, as the fixed prior does: each global search, each local search (which keeps its three-width selection of orientations and weights them by this prior alone, without the Gaussian log priors; the owner's choice of 2026-11-19) and the final pass. The first iteration has no fit and scores as RELION's does. Each half's `run_itNNN_half{1,2}_model.star` holds the cell probabilities as `rlnOrientationDistribution` and the fit in two blocks, `data_relax_direction_prior_fvml` (lobes, fit order, cap, cold or warm start, EM iterations, objective per particle, seconds) and `data_relax_direction_prior_fvml_components` (component 0 the uniform one; weight, mean direction as RELION's rot and tilt and as a unit vector, $\kappa$); `--continue` reads the fit back and warm-starts from it.
- **Approximation in the E-step.** RELION-style significance truncation makes $\rho$ the truncated posterior, so the outer EM guarantee of section 6.2 is as approximate as it is for the existing nonparametric update.
- **Number of components.** $n_r$ is fixed here. A Dirichlet-process mixture (a countably infinite number of lobes) is a separate derivation and is not covered.

## 10. Numerical verification

Throwaway scripts in the session scratchpad (not kept), CPU, float64, NumPy 2.2.6 / SciPy 1.16.2 / healpy 1.18.0 of the project's default pixi environment, HEAD `e1f0c291` on `pose_prior`; they were written before the implementation and are not
the code under test. Single seeded runs, no repetition beyond what is stated. The unit tests (`pixi run python -m pytest -v tests/unit/test_fvml_mixture_em.py`, 32 tests, about 5 s) pin the density and its
normalization, $A^{-1}$, the M-step against a numerical maximizer (plain, antipodal, and a non-group pair of maps), monotone ascent and the Dirichlet floor, recovery of the generating mixture, the sector-versus-all-copies identity, antipodal recovery, the cap and the unboundedness
slope, the initialization and input validation. They do **not** pin the normalization table, the $\kappa$-bias table, the initialization success rates or the singular trajectory below; those are recorded here only. Mutating the code (the $\kappa$ update,
the group transform, the uniform normalization, the Dirichlet term, the mean normalization) fails the suite.

**Density and $A^{-1}$.** The stable $\log f$ agrees with the direct formula to $1.4\times10^{-14}$ at $\kappa=300$ and stays finite at $\kappa=10^{8}$, where the direct formula overflows. Newton's $A^{-1}$ against a 60-digit root: relative error $\le1.1\times10^{-15}$
for $\bar R\le0.9$, $6\times10^{-15}$ at $\kappa=100$, $5.5\times10^{-13}$ at $\kappa=10^{4}$, $5.6\times10^{-11}$ at $\kappa=10^{6}$ (the representation of $\bar R$ limits it from there).

**M-step is the exact maximizer of $B$.** On random atoms ($D=400$, $n_r=4$, random responsibilities from random $\Theta'$), $B(\text{closed form})-\max B(\text{12 BFGS starts})$ is $+1.2\times10^{-10}$ (plain), $+1.6\times10^{-11}$ (antipodal), $0$ (C3).

**Ascent and stationarity.** Data: $n=10^5$ draws from $0.25\,\text{uniform}+0.40\,\mathrm{FvML}(e_z,25)+0.35\,\mathrm{FvML}(e_x,120)$ binned at HEALPix $N_{\rm side}=16$ ($D=3072$, $\delta=3.67^\circ$), $\kappa_{\max}=1/\delta^2=244$. Smallest per-iteration change of $Q$ over all runs:
$\ge0$ in every configuration ($+1.0\times10^{-10}$ for $n_r=3$, $+5.8\times10^{-6}$ for $n_r=4$). Best fit for $n_r=3$: $\pi=(0.249,0.400,0.351)$, $\kappa=(24.5,113.8)$, $\lvert\eta_z\rvert=1.000$, $\lvert\eta_x\rvert=1.000$ (the $\kappa=120$ value is the expected shift of (E9),
$120/(1+120\delta^2/12)=115$); $Q=-79\,074.12$ against $-79\,122.90$ at the generating parameters. Finite-difference gradient of $Q$ at that fit in $(\log\pi,\eta,\log\kappa)$: $\max\lvert\partial Q/\partial x\rvert=2.4\times10^{-3}$, cap inactive.
$n_r=4$ (one extra component): $Q=-79\,072.05$, still moving after 5000 iterations (extra component at $\kappa\approx0.3$, $\pi\approx0.02$).

**Initialization** ($n_r=3$, same data, 40 starts each, cap $1/\delta^2$, 3000 iterations): random $\eta$, $\kappa\in[1,200]$, random $\pi$: 6 starts within 1 nat of the best optimum, 6 stuck at the uniform-only solution ($Q=-N\log4\pi=-253\,102.42$), 28 at other local optima
(the poorest of them at $Q\approx-1.95\times10^{5}$). In four inspected uniform-only random starts each component began with $N_j=250$–$875$ of the $10^5$ mass, and $\pi_j$ fell to $\approx10^{-15}$ within 21–22 iterations. k-means++ starts: 31 within 1 nat of the best, 0 uniform-only, 9 at one poorer
optimum ($Q=-207\,636$). The optimum is local; use several seeds.

**Unboundedness.** HEALPix $N_{\rm side}=8$, all atoms mass 1 except one of mass 50, component on that atom, $\kappa_{\max}=10^{12}$, start $\kappa=200$:

| iteration | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| $\kappa$ | $2\!\times\!10^{2}$ | $3.4\!\times\!10^{2}$ | $1.5\!\times\!10^{3}$ | $1.1\!\times\!10^{6}$ | $10^{12}$ (cap) | $10^{12}$ |
| $Q$ | $-2303.2$ | $-1923.3$ | $-1857.1$ | $-1526.2$ | $-839.76$ | $-839.76$ |

**Symmetry.** C3 about $z$: the objective on the atoms of one sector equals the objective on all three copies at mass $\rho/3$ ($-482.9338196135$ both), and 25 EM steps agree to $\lvert\Delta\kappa\rvert\le2.4\times10^{-12}$, $\lvert\Delta\pi\rvert\le4.4\times10^{-16}$, same $\eta$ modulo
the group. Antipodal: $4\times10^{4}$ draws from $0.3\,\text{uniform}+0.7\,\mathrm{FvML}(\eta_0,40)$ with random sign: the $\mathcal G=\{I,-I\}$ fit gives $\pi=(0.305,0.695)$, $\kappa=39.6$, $\lvert\eta\cdot\eta_0\rvert=1.0000$, $Q=-45\,984$; the plain fit with the same $n_r$ (one lobe) gives
$Q=-77\,140$.

**Discretization.** Normalization error $\max_\eta\lvert\sum_mf(v_m)\Delta S-1\rvert$ (40 random $\eta$):

| order ($\delta$) | $\sigma/\delta=0.3$ | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 ($7.33^\circ$) | $6.1\!\times\!10^{-1}$ | $3.5\!\times\!10^{-2}$ | $5.2\!\times\!10^{-3}$ | $3.7\!\times\!10^{-3}$ | $8.1\!\times\!10^{-3}$ | $7.1\!\times\!10^{-3}$ | $3.8\!\times\!10^{-3}$ |
| 4 ($3.67^\circ$) | $7.7\!\times\!10^{-1}$ | $7.0\!\times\!10^{-2}$ | $3.2\!\times\!10^{-2}$ | $3.1\!\times\!10^{-2}$ | $1.9\!\times\!10^{-2}$ | $1.2\!\times\!10^{-2}$ | $5.6\!\times\!10^{-3}$ |
| 5 ($1.83^\circ$) | $6.9\!\times\!10^{-1}$ | $5.4\!\times\!10^{-2}$ | $9.0\!\times\!10^{-4}$ | $7.9\!\times\!10^{-4}$ | $5.8\!\times\!10^{-4}$ | $4.4\!\times\!10^{-4}$ | $7.8\!\times\!10^{-4}$ |
| 6 ($0.92^\circ$) | $5.2\!\times\!10^{-1}$ | $4.7\!\times\!10^{-2}$ | $6.0\!\times\!10^{-4}$ | $3.3\!\times\!10^{-4}$ | $2.7\!\times\!10^{-4}$ | $2.1\!\times\!10^{-4}$ | $1.5\!\times\!10^{-4}$ |

Bias of $\kappa$ (HEALPix $N_{\rm side}=32$, $3\times10^{5}$ draws per run, mean of 5 runs, one component): fitted/true against $1/(1+\kappa\delta^2/12)$: $\kappa=20$: 0.999 / 0.998; 100: 0.991 / 0.992; 400: 0.966 / 0.967; 1000: 0.921 / 0.921; 2000: 0.854 / 0.854; 4000: 0.735 / 0.746. Beyond that range, single runs ($N_{\rm side}=32$,
$2\times10^{5}$ draws, $\kappa_{\max}=10^{5}$): $\kappa=8000$ ($\kappa\delta^2=8.2$): 0.76 against 0.60 predicted; $\kappa=30\,000$ (30.7): 0.43 against 0.28.

**Cell-mass fit** (section 7; `fit_fvml_mixture_cells`, same data and starts as the point fit, $N_{\rm side}=32$, $3\times10^5$ draws, mean of 3 runs, one component, fitted/true for $C=4,16,64$ children against the point fit):
$\kappa=100$ ($\kappa\delta^2=0.10$): 0.997, 0.999, 0.999 against 0.990; $\kappa=1000$ (1.02): 0.978, 0.994, 0.998 against 0.916; $\kappa=4000$ (4.09): 0.918, 0.977, 0.993 against 0.742; $\kappa=8000$ (8.18): 0.847, 0.956, 0.988 against 0.622. The residual
follows quantization at the children's spacing, $1/(1+\kappa\delta^2/(12C))$ (0.979 predicted for $\kappa=4000$, $C=16$). With $C=1$ the fit equals the point fit and its objective is the point objective plus $N\log(4\pi/D)$ (unit test). On the three-component data above,
$n_r=3$, best of three k-means++ starts, $C=16$: $\kappa=(24.8,120.1)$, $-0.9\,\%$ and $+0.06\,\%$, where the point fit is $-2.6\,\%$ and $-3.7\,\%$ low.

**Simulation against the histogram** ([`scripts/fvml_prior_simulation/`](../../scripts/fvml_prior_simulation/Snakefile), run 2026-11-17 13:55–14:47 locally on 20 cores, root `/scratch/gpfs/CRYOEM/gilleslab/ry2228/data/fvml_prior_simulation/run2`, snapshot
`sim_20261117_1353` = HEAD `b4319a3c` plus the uncommitted simulation and cell fit, `git diff HEAD | sha256sum` `eb880722…986b`). Particles are drawn from a target, binned at HEALPix orders 3–5 and estimated by the histogram (RELION's update), the point fit and the cell fit
($C=16$); best of 5 k-means++ starts, $c=1$ ($\kappa_{\max}$ 61 / 244 / 978), stop at a gain below $10^{-7}$ nats per particle or 1000 iterations; 5 replicates per count; score the Jensen–Shannon distance (bits) to the target on the fit grid. Targets: three known mixtures,
the closed-form Kent populations of the pose-prior benchmark's mild and strong caps, the final learned `pdf_direction` of three benchmark refinements (100k particles; mild cap, 66 % mild cap, strong cap) and the C3-expanded RELION poses of 130k EMPIAR-10096 particles; 5 lobes plus uniform unless the truth's count is known.
Median JS(cell fit) / JS(histogram) on the same draw:

| target | order | $10^3$ | $10^4$ | $10^5$ | $10^6$ |
| --- | --- | --- | --- | --- | --- |
| Kent population, mild | 3 / 4 / 5 | 0.038 / 0.015 / 0.006 | 0.030 / 0.009 / 0.003 | 0.025 / 0.006 / 0.002 | 0.099 / 0.027 / 0.006 |
| Kent population, strong | 3 / 4 / 5 | 0.069 / 0.022 / 0.010 | 0.028 / 0.008 / 0.003 | 0.046 / 0.014 / 0.004 | 0.19 / 0.058 / 0.016 |
| three lobes ($\kappa$ 8, 60, 300) | 3 / 4 / 5 | 0.32 / 0.013 / 0.006 | 2.6 / 0.019 / 0.001 | 28 / 0.19 / 0.001 | 293 / 2.0 / 0.001 |
| learned prior, mild-cap run | 3 / 4 / 5 | 0.081 / 0.058 / 0.060 | 0.27 / 0.28 / 0.23 | 2.4 / 2.4 / 1.8 | 23 / 25 / 19 |
| learned prior, strong-cap run | 3 / 4 / 5 | 0.13 / 0.15 / 0.12 | 0.52 / 0.59 / 0.44 | 5.5 / 4.7 / 2.5 | 58 / 53 / 25 |
| EMPIAR-10096 poses, C3 | 3 / 4 / 5 | 0.17 / 0.093 / 0.077 | 1.2 / 0.36 / 0.22 | 13 / 3.7 / 1.4 | 136 / 40 / 16 |

- Smooth targets that are not FvML mixtures (the Kent populations) are fitted to $1$–$2\times10^{-5}$ bits without sampling noise, so the fit stays closer than the histogram up to $10^6$ particles.
- With $c=1$ a lobe narrower than the grid spacing is held at the cap (the $\kappa=300$ lobe at 61 on order 3), the fit's floor is then 0.04–0.06 bits and the histogram overtakes it from about $3\times10^3$ particles (order 3) or $10^5$–$10^6$ (order 4).
- On learned priors and real poses the fit wins below $1$–$6\times10^4$ particles. For the mild-cap run its floor (0.0031 / 0.0115 / 0.0302 bits at orders 3 / 4 / 5) equals the learned prior's own distance to the Kent population it estimates (0.0032 / 0.0115 / 0.0302): the fit returns the population and the floor is the learned prior's
  noise (31 % of its order-5 cells are empty). For the strong-cap run the learned prior is 0.026 / 0.047 / 0.068 bits from its population and the fit's floor 0.008 / 0.029 / 0.053.
- Point and cell fits agree to about 1 % in this table except where lobes are narrow for the grid, where the cell fit's floor is 5–10 times lower (strong cap at order 3, the known mixtures at order 5). Cost per start at order 5: about 1 s (point) and 19 s (cell); 87 % of fits met the gain criterion before 1000 iterations,
  the rest are on the learned-prior targets.

**Lobe count, cap width and held-out choice** (run 2026-11-17 15:16–21:11 locally on 20 cores, root `/scratch/gpfs/CRYOEM/gilleslab/ry2228/data/fvml_prior_simulation/run3`, snapshot `sim_20261117_1514` = HEAD `b4319a3c` plus the uncommitted simulation and library,
`git diff HEAD | sha256sum` `84a5d1a6…f243`). Same protocol as above with 3 replicates at $10^3$, $10^4$, $10^5$ particles and no draws. Settings: 5, 10 and 20 lobes at $c=1$ (targets without a known count), and $c=1, 0.5, 0.25$ with 5 lobes or the true count.
Each trial also draws a second, independent sample of the same size and scores every estimate by its mean log cell probability under that sample (held-out score). Median JS (bits) of the cell fit at order 4, $c=1$:

| target | lobes | $10^3$ | $10^4$ | $10^5$ | no draws | histogram $10^5$ |
| --- | --- | --- | --- | --- | --- | --- |
| Kent population, mild | 5 / 10 / 20 | 0.0042 / 0.0095 / 0.019 | 0.00034 / 0.0008 / 0.0015 | 3.4e-5 / 6.1e-5 / 1.0e-4 | 1.1e-5 / 1.2e-5 / 1.4e-5 | 0.0057 |
| learned prior, mild-cap run | 5 / 10 / 20 | 0.015 / 0.020 / 0.028 | 0.012 / 0.012 / 0.011 | 0.011 / 0.011 / 0.0094 | 0.011 / 0.011 / 0.0097 | 0.0048 |
| learned prior, strong-cap run | 5 / 10 / 20 | 0.030 / 0.035 / 0.035 | 0.029 / 0.021 / 0.018 | 0.029 / 0.020 / 0.017 | 0.029 / 0.020 / 0.017 | 0.0061 |
| EMPIAR-10096 poses, C3 | 5 / 10 / 20 | 0.027 / 0.023 / 0.027 | 0.022 / 0.017 / 0.013 | 0.022 / 0.017 / 0.012 | 0.022 / 0.017 / 0.011 | 0.0059 |

- More lobes help only on targets that are not smooth: on the Kent populations 5 lobes is best at every finite draw and all counts share the floor; on the strong-cap learned prior and EMPIAR-10096 the floor falls by 40–50 % from 5 to 20 lobes; the mild-cap floor is the learned prior's
  noise and barely moves. The histogram still wins on these four targets at $10^5$ particles.
- The held-out score picks the setting with the lowest median JS in 98 % of the 270 (target, order, count, fit, choice) cases; where it misses, the pick is at most 1.05 times farther. It picks 5 lobes on smooth targets and at $10^3$ particles, and 20 lobes on the rough targets from
  $10^4$. In refinement the two half-sets can play the sample and held-out roles.
- Known mixtures without draws at order 3: $c=1$ leaves 0.040 / 0.056 bits (three / five lobes), $c=0.5$ 0.0007 / 0.002, $c=0.25$ $1\times10^{-5}$ / $2\times10^{-5}$; at order 4, $c\le0.5$ lowers the floor from $10^{-3}$ to $10^{-6}$. On the other targets with 5 lobes the three caps give
  the same fit, except EMPIAR-10096 at order 3 (0.018 to 0.009). No overfitting from the looser cap was seen, even at $10^3$ particles; a loose cap with 10–20 lobes was not run (at $c=1$, 20-lobe fits hold a median 1–4 lobes at the cap).
- With $c\le0.5$ the point fit's quantization bias shows: its floor on the known mixtures at order 4 is $2.7\times10^{-4}$ / $4.1\times10^{-4}$ bits against $1.0\times10^{-6}$ / $1.7\times10^{-6}$ for the cell fit.
- From a cold start the cost grows with the lobe count: at order 5 the 20-lobe cell fit takes a median 15 min for 5 starts (point fit 52 s), and 57 % reach the 1000-iteration limit.

**Lobe count at $c=0.25$ and refinement replays** (run 2026-11-18 15:57–21:34 locally on 20 cores, root `/scratch/gpfs/CRYOEM/gilleslab/ry2228/data/fvml_prior_simulation/run4`, snapshot `sim_20261118_1556` = HEAD `b4319a3c` plus the uncommitted simulation, replay and library,
`git diff HEAD | sha256sum` `f1a02a1b…e288`). Static trials as in run 3 with 10 and 20 lobes at $c=0.25$ (5 lobes from run 3, same trial code). They repeat run 3's result: on the strong-cap learned prior and EMPIAR-10096 the order-4 floor falls from 0.029 to 0.0095 and from 0.022 to 0.0071 bits with 20 lobes,
on the Kent populations 10 and 20 lobes are 2–4 times farther at finite draws, and the held-out score picks the best count in 99 % of 108 cases (worst 1.02 times farther).

Replays (`replay_refinement.py`): three benchmark refinements (100k particles; mild cap, 66 % mild cap, strong cap), each half-set on its own, every iteration. The half's max-posterior directions (`run_itNNN_data.star`) are binned at the iteration's prior order (at most 5) and fitted with the cell fit;
a fixed random 10 % or 20 % of the half is held out, 5, 10 and 20 lobes are fitted on the rest, the best held-out count is refitted on the whole half. Warm: each candidate starts from its own previous fit. Cold: best of 5 k-means++ starts ($c=0.25$, 20 % only). Score: JS distance (bits) to the
population the particles were drawn from, at the iteration's order (6 at the end). Medians over both halves and the last 3 iterations, $c=0.25$, 20 % held out, warm:

| run | chosen, refit | 5 lobes | 10 lobes | 20 lobes | RELION learned prior | histogram of the poses |
| --- | --- | --- | --- | --- | --- | --- |
| mild cap | 0.0070 | 0.0013 | 0.0023 | 0.0069 | 0.107 | 0.142 |
| 66 % mild cap | 0.0058 | 0.0017 | 0.0036 | 0.0057 | 0.160 | 0.223 |
| strong cap | 0.042 | 0.025 | 0.032 | 0.042 | 0.080 | 0.095 |

- The fit is 15–30 times closer to the population than RELION's learned prior on the mild runs and 2 times on the strong run; a fixed 5-lobe fit 60–100 and 3 times.
- The held-out score from the same half picks 20 lobes in 96 % of 540 (iteration, setting) cases, while the 5-lobe candidate is the closest to the population in 98 %. The cause is in the data, not the score: pose estimation leaves small clumps that the population does not have
  (66 % run, half 1, iteration 16: the warm 5-lobe fit spends two lobes of weight 0.003 at $\kappa$ 1550 and 15200, cap 15600; within 2° of them the estimated poses hold 0.39 % and 0.31 % of the half against 0.17 % and 0.15 % under the population), and the held-out particles share them
  (20 against 5 lobes: +0.01 to +0.05 nats per held-out particle). Neither the 10 % share nor $c=1$ changes the choice; $c=1$ still has 2–4 lobes at its cap.
- Warm fits reach a higher objective than the best cold start at late iterations (+0.003 to +0.015 nats per particle) and are about 10 times farther from the population (5 lobes: 0.0014 against 0.0001 bits): the cold starts miss the clumps. Cold's lower distance is overfitting seen from
  the other side, not a better optimizer.
- Cost at order 5, one core (numpy), median per refinement iteration: warm 110 s (5 / 10 / 20 lobes 4 / 12 / 45 s, refit 37 s; 45 / 83 / 204 EM iterations), cold 1800 s. Keeping every candidate's mixture costs about 100 numbers each.
- Caveat: the replay fits max-posterior poses; the sub-EM inside refinement would see each particle's posterior over directions, which is smoother. The run files do not keep it, so the replay cannot test whether the clumps survive there.

**In refinement: the pose-prior sweep** (`relax refine --fvml-direction-prior 5`, Slurm jobs 17934685–17934696, 2026-11-19/20; snapshot `fvml_refinement_20261119_1024` = HEAD `d2dff905` plus this change, `git diff HEAD | sha256sum` `87c25f25…c95a`;
root `/scratch/gpfs/CRYOEM/gilleslab/ry2228/data/pose_prior_benchmark/sweep_mild_box256`, variant `fvml` of `scripts/pose_prior_benchmark/config.yaml`). The 12 sweep cells with 100k particles (box 256; Kent share 0, 0.33, 0.66, 1 of the mild cap mixed with uniform directions; noise 0.5, 1, 2) are refined with the
same start map, settings and seed as the sweep's other variants: RELION's learned prior, the true prior fixed and the uniform prior fixed (`--fixed-direction-prior`). Aligned ground-truth FSC area (`gt_fsc_auc_aligned` of `compare_variants.py`) and, last, the area of the map from the true poses:

| Kent share | noise | learned | fixed uniform | fixed true | FvML | true poses |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5 / 1 / 2 | 0.7552 / 0.5819 / 0.1735 | 0.7556 / 0.5874 / 0.1755 | – | 0.7554 / 0.5872 / 0.1746 | 0.7846 / 0.7111 / 0.6345 |
| 0.33 | 0.5 / 1 / 2 | 0.7503 / 0.5827 / 0.1691 | 0.7528 / 0.5866 / 0.1704 | 0.7561 / 0.5897 / 0.1726 | 0.7560 / 0.5890 / 0.1708 | 0.7819 / 0.7083 / 0.6316 |
| 0.66 | 0.5 / 1 / 2 | 0.7448 / 0.5663 / 0.1697 | 0.7471 / 0.5653 / 0.1695 | 0.7475 / 0.5774 / 0.1763 | 0.7475 / 0.5751 / 0.1712 | 0.7725 / 0.6981 / 0.6213 |
| 1 | 0.5 / 1 / 2 | 0.7365 / 0.5632 / 0.1725 | 0.7401 / 0.5549 / 0.1724 | 0.7409 / 0.5768 / 0.1919 | 0.7446 / 0.5733 / 0.1753 | 0.7462 / 0.6705 / 0.5934 |

- FvML is at or above the learned prior in all 12 cells (+0.0002 to +0.0101) and lowers the median direction error in 11 (at Kent share 1: 0.765 to 0.736°, 1.96 to 1.84°, 79.0 to 77.4°). At noise 0.5 it ties the fixed true prior (better at Kent share 1); at noise 1 it gets 74–90 % of the fixed true prior's
  gain over the learned prior. On uniform directions it ties the fixed uniform prior within 0.0009 (one seed per variant: a scale for run-to-run differences), where the learned prior loses up to 0.0055.
- At noise 2 every run but the fixed true prior stays near failure (median direction error about 80°): FvML adds 0.001–0.003, the fixed true prior 0.004–0.019. A prior fitted to wrong poses follows them: the uniform weight falls to about 0 in every cell, the uniform one included, once the
  search reaches order 3.
- The final prior's JS distance to the population (the final model STAR's `rlnOrientationDistribution`): FvML $6\times10^{-5}$–$10^{-3}$ bits at noise 0.5 and 0.0008–0.006 at noise 1, against 0.10–0.17 and 0.042–0.053 for the learned prior (9–3000 times closer); at noise 2 both are 0.01–0.3.
- Fits: under one minute per run in total (both halves, every iteration; 0–4 of 24–36 fits stop at the 1000-iteration limit). Lobes stay broad ($\kappa$ 1–20) except one small lobe (final weight 0.007–0.011) in the noise-1 cells at Kent share 0.66 and 1 that peaks at $\kappa$ 800–900 mid-run. The FvML jobs' wall
  times (1.2–2.4 times the learned runs') are not the method's: iteration 1, which has no fit yet, is already about 5 times slower in host-side stages (operand preparation 241 s against 28 s at Kent share 1, noise 2) while 12 jobs shared nodes. Run time was not measured cleanly.

## 11. Open decisions

1. The cap or hyperprior on $\kappa$. Decided (2026-11-17): the cap $\kappa_{\max}=1/(c\delta)^2$, not the exponential prior (E8), which needs its own tuning. Decided (2026-11-18), from the simulation of section 10: $c=0.25$ with the cell-mass fit is the default.
2. Point atoms versus the cell-mass likelihood of section 7: both are implemented and compared in a simulation (`scripts/fvml_prior_simulation/`); the cell-mass fit is the default (2026-11-18), since a cap looser than one spacing exposes the point fit's quantization bias.
3. How $n_r$ is chosen: five lobes plus the uniform component by default, the true count when the target is a known mixture. The held-out score (section 10) picks the count well on i.i.d. draws. Decided (2026-11-18): the held-out particles are a fixed 10–20 % of the same half-set, never the other half-set, so the two halves stay independent. Open: in the refinement replays this choice always takes the
   largest count, which fits pose-estimation clumps and is farther from the population than 5 lobes; whether to fix the count, penalize it, widen the cap or fit the posterior masses instead. Warm starts for every candidate work (keep each candidate's mixture; 16 times cheaper than cold).
4. Integration (2026-11-19): five lobes, fixed, warm-started, acting in every search including local search (section 9); the module moved to `relax/sampling/`. Benchmarked (2026-11-20, section 10) on the 12 sweep cells with 100k particles: at or above RELION's learned prior everywhere, near the fixed true prior at noise $\le 1$. Open: the 50k cells, a clean run-time measurement, and seeds to put a
   spread on the differences.

## References

- Dempster, A. P., Laird, N. M., Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *J. R. Stat. Soc. B* 39(1):1–38.
- Wu, C. F. J. (1983). On the convergence properties of the EM algorithm. *Ann. Statist.* 11(1):95–103, doi 10.1214/aos/1176346060.
- Banerjee, A., Dhillon, I. S., Ghosh, J., Sra, S. (2005). Clustering on the unit hypersphere using von Mises–Fisher distributions. *J. Mach. Learn. Res.* 6:1345–1382 (EM for vMF mixtures and the $\kappa_0$ approximation; their data are unweighted).
- Rossi, F., Barbaro, F. (2022). Mixture of von Mises–Fisher distribution with sparse prototypes. *Neurocomputing* 501:41–74 (a mixture-of-vMF reference cited by the team's note; context only, not used for the derivation above).

(Bibliographic details of the first three were checked against the publishers' or JMLR's listings on 2026-11-06; Rossi and Barbaro is quoted from the note's reference list.)
