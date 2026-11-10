# Fitting a mixture-of-FvML viewing-direction prior by a nested EM

**Status (2026-11-06, branch `pose_prior`).** Derived, numerically checked (section 10) and implemented standalone
(section 8). Nothing in the refinement loop calls the implementation; section 9 says what an integration needs. The
implementation is [`scripts/lib/fvml_mixture_em.py`](../../scripts/lib/fvml_mixture_em.py), placed there because no
command of `relax/` reaches it, and its tests are
[`tests/unit/test_fvml_mixture_em.py`](../../tests/unit/test_fvml_mixture_em.py). The document is self-contained: it
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

An alternative that removes (2) and (3) is the grouped-data likelihood $\prod_m(\int_{{\rm cell}\,m}\lambda)^{\rho_m}$: its EM is the same M-step with each cell represented by child atoms of a finer grid, and an E-step that redistributes the cell's mass over its children in proportion to the current model,
$r_{m,c,k}=a_k(v_c)/\sum_{c'\in m}\sum_{k'}a_{k'}(v_{c'})$. It is *not* implemented and *not* checked numerically here; it is listed in section 11.

## 8. Algorithm and implementation

```
input   atoms (v_m, rho_m), m=1..D        # unit vectors (D,3), masses (D,), one half-set / one class
        n_r, group G (default {I}), kappa_max, Dirichlet a_j (default 1), s_max, tol
init    Theta^0 = previous Theta (warm start)  or  weighted spherical k-means++ (below)
for s = 0, 1, ..., s_max-1:
    E:  log a_k(v_m) for all k;  log lambda_m = logsumexp_k;  r_mk = exp(log a_k - log lambda_m)
        Q_s = sum_m rho_m log lambda_m
    M:  N_j, S_j; pi_j; eta_j = S_j/|S_j|; kappa_j = min(A^{-1}(|S_j|/N_j), kappa_max)
    stop if Q_s - Q_{s-1} <= tol * |Q_s|
output  Theta, [Q_s]                      # the history is part of the result
```

- **Cold start.** Weighted spherical k-means++ seeding ($n_r-1$ centres drawn with probability $\propto\rho_m(1-\max\cos)^2$, the cosine taken to the nearest seed and best over the symmetry maps), ten Lloyd steps on the cosine, then $\eta_j$ the cluster's resultant direction, $\kappa_j=A^{-1}(\bar R_j)$ of the cluster and
  $\pi$ from the cluster masses with $\pi_1=0.2$ (the choices of the checks in section 10). An empty cluster keeps its seed with a small weight share. Run several seeds and keep the highest $Q$; the measured success rates are in section 10.
- **Cost.** $O(D\,n_r\,n_g)$ per sub-iteration. Measured (4 threads, 50 iterations, one run each): 3.4 ms per iteration at $D=12\,288$ (order 5, $n_r=4$); 20.9 ms at $D=49\,152$ (order 6, $n_r=6$) and 27.7 ms at $n_r=10$; 69.6 ms at order 6, $n_r=6$ with C3 (three maps).
- **Implementation** ([`scripts/lib/fvml_mixture_em.py`](../../scripts/lib/fvml_mixture_em.py)): `FvmlMixture` holds $\Theta_{22}$ and the symmetry maps; `init_fvml_mixture` is the cold start above; `fit_fvml_mixture` runs the loop and returns a `FitResult` (mixture, the objective at every iterate, whether it converged);
  `log_density` evaluates $\log\lambda$; `inverse_langevin` solves $A(\kappa)=\bar R$; `kappa_max_for_grid` is the cap rule (E9) with $c$ the minimum width in node spacings. `kappa_max` has no default (an uncapped fit is unbounded), the Dirichlet concentration is one scalar,
  the returned objective includes the Dirichlet term so it is the quantity that never decreases, and the loop raises if it does. The maps may be any finite set of orthogonal matrices; only a group makes $\lambda$ invariant.
- **Precision.** Host float64, cast once at the boundary, as `orientation_priors.py` does for the other priors; a 3-vector and a few scalars per component do not justify float32, and $A^{-1}$ near $\bar R\to1$ loses $\approx10^{-16}/(1-\bar R)$ in relative accuracy.

## 9. Interface to the refinement loop (not implemented)

- **Input.** One half-set's (and class's) direction masses $\rho_m$ on the grid of its E-step and the grid's unit vectors; per half, because the gold-standard halves keep separate priors. The Dirichlet pseudo-counts are in units of particles, so $N$ (not the normalized sum) is passed.
- **Output.** $\Theta_{22}$. The E-step needs $\log\lambda(v)$ at its node directions, up to a constant for K=1 (the constant cancels in each particle's normalization); K>1 needs the per-class normalization over the grid. Because $\lambda$ is a function
  on the sphere and not on a grid, a prior learned at one HEALPix order can be evaluated at the next one: RELION and RELAX reset `pdf_direction` to uniform at every order change (`relion_direction_log_priors_for_half`), which this prior would not
  need, and local search could evaluate $\log\lambda(Re_3)$ at any rotation.
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

## 11. Open decisions

1. The cap or hyperprior on $\kappa$: the interval $\kappa_{\max}=1/(c\delta)^2$ with which $c$ (section 7), or the exponential prior with a mean $\kappa_0$ (E8).
2. Point atoms (this document) versus the grouped-data likelihood of section 7, which removes the $\kappa$ bias and the grid-tied cap.
3. How $n_r$ is chosen; a Dirichlet-process mixture is a separate derivation.
4. Integration: when the prior is wired into the refinement loop the module moves from `scripts/lib/` to `relax/sampling/` and the links in this document move with it.

## References

- Dempster, A. P., Laird, N. M., Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *J. R. Stat. Soc. B* 39(1):1–38.
- Wu, C. F. J. (1983). On the convergence properties of the EM algorithm. *Ann. Statist.* 11(1):95–103, doi 10.1214/aos/1176346060.
- Banerjee, A., Dhillon, I. S., Ghosh, J., Sra, S. (2005). Clustering on the unit hypersphere using von Mises–Fisher distributions. *J. Mach. Learn. Res.* 6:1345–1382 (EM for vMF mixtures and the $\kappa_0$ approximation; their data are unweighted).
- Rossi, F., Barbaro, F. (2022). Mixture of von Mises–Fisher distribution with sparse prototypes. *Neurocomputing* 501:41–74 (a mixture-of-vMF reference cited by the team's note; context only, not used for the derivation above).

(Bibliographic details of the first three were checked against the publishers' or JMLR's listings on 2026-11-06; Rossi and Barbaro is quoted from the note's reference list.)
