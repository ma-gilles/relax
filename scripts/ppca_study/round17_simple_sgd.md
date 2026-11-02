# Simple SGD with a PC prior (owner design, 2026-11-02)

Goal: a simpler replacement for VDAM. No half-set gate, no fudge, no Newton step, no moving average in the update.

## Update
    mu <- mu + lr * G_mu / c
    W  <- W  + lr * G_W  / c
    W  <- W / (1 + lr * (b/N) / (c * tau2))     per shell

Prior on PCs only, none on the mean. G = batch gradient, c = the current SGD scalar, b/N = batch fraction, momentum 0.
tau2 per shell = true total PC power at that shell / q (relax ppca_refinement.initialization.loading_row_norm_variance_prior,
RECOVAR make_gt_prior_from_variance_total). Written as a shrink so small tau2 at high shells can't overshoot.

## Logged only, never used in the update (step 1)
    m <- .9 m + .1 G        per coefficient, complex, mean and every PC
    A(s) = shell mean of |m|^2 (PCs: sum over PCs first); B(s) same with G
    v(s) = A(s) / (B(s)/19) - 1
Read only after 20-30 updates in a new shell. Also log per-shell <W, G>, PC power, mean power.

## Steps (one seed, new sets in ppca_suite_b150_20261102)
1. Oracle shrink, current fixed schedule. C noise 1, C noise 3, H1 noise 3.
   H1 has no true PCs, so use C's tau2 there; tau2 = 0 would pass trivially.
2. Oracle shrink, marching rule from the v curves (rule set after seeing them).
3. Estimated tau2 in place of the oracle.
Fallback for marching: FSC from running rhs per even/odd half, mean channel, no lhs (uniform views).

## Pass, step 1
H1: PC power near zero. C noise 1: PC/mean near .44 (shells 1-31), energy >= .46. C noise 3: poses hold, PCs survive.
Report with the agreed suite, raw PCs and PC-coordinate latent scatters.
