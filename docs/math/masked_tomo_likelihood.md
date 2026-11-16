# Masked-reconstruction likelihood for crowded subtomograms (pilot backend)

Opt-in experimental backend, `relax/masked_tomo/`. It is not wired into `relax refine` or
`relax ppca_initial_model`; the crowding pilot drives it (`masked-tomo/scripts/run_pilot.py` in the project).
No default changes.

## 1. Two measurements of one particle

Particle `i` has `K` tilts with RECOVAR tilt rotations `rt_k` (identity particle pose), CTF rows `C_k` and
a detector noise spectrum `Sigma` (per Fourier coefficient). A pose is a rotation `Q` (row rotation
`rt_k Q`) and a 3D shift `u` in working pixels; tilt `k` sees the 2D shift `t_k(u) = (rt_k u)_{1,2}`.

**A, image likelihood** (section 16.1 of `vdam_ppca_algorithm.md`): on the target-box crops,
`y_k = C_k S_{t_k(u)} P(rt_k Q) x + e_k`, `e_k ~ N(0, Sigma)`.

**B, masked reconstruction.** The larger detector patch `y` (all objects of the scene) is mapped by one fixed
linear operator per particle and stage,

```text
D = A^* Sigma^-1 C^2            (backprojected weights on the scene grid, Fourier diagonal)
L y = M F^-1 (D + lam)^-1/2 A^* Sigma^-1 C y
```

`A^*` is RECOVAR's trilinear backprojection at `rt_k` onto the `S^3` scene grid (no particle pose), `F^-1` the
centred inverse DFT, `M` a fixed binary sphere around the picking centre (radius 15 working px, generous: target
radius 11.5 plus the shift search) on a crop of the scene. `lam = lam_rel * mean(D)` over the observed band.
The brief's general form `L = M R A^* Sigma^-1` is used with `R = (D + lam)^-1/2` instead of
`(H + lam)^-1`: any fixed invertible `R` gives a valid measurement; this one whitens the noise so the masked
covariance is close to a multiple of the identity.

Model: `V = L y | x, g ~ N(F_g x, C)`, `F_g = L A_target,g`, `C = L Sigma L^*`. Neighbour density that leaks
through the mask is **not** in `C`: it is treated as unmodelled signal (the measurement's misspecification).

## 2. PPCA score

With `x = mu + W z`, `r = V - F_g mu`, `B = F_g W`, the score, posterior mean and covariance are those of the
brief (`K = I + B^T C^+ B`, `d = B^T C^+ r`, score `-1/2 [r^T C^+ r - d^T K^-1 d + logdet K]`). Both arms hand
`inner_p = <f_p, data>_w` and `gram_pq = <f_p, f_q>_w` to RELAX's own epilogue
(`relax.ppca_refinement.full_row_kernels.latent_scores`). For B the Gram matrix depends on the shift (the mask
breaks translation invariance), so the epilogue runs on the flattened (rotation, shift) grid.

## 3. Labelled approximations in B

1. **Mean:** `A^* Sigma^-1 A ~ D` (Fourier diagonal), so `F_g x ~ M F^-1 D (D + lam)^-1/2 T_g x`, with `T_g x`
   the template's DFT resampled at `k Q` (trilinear, from a 2x zero-padded, gridding-corrected grid) times the
   3D phase of `u`. Tiny-case error against the exact `L A_g x`: 18% (5 tilts; `tests/unit/test_masked_tomo.py`),
   against about 160% for a transposed rotation or negated shift.
2. **Covariance:** `C ~ c I` on the mask, `c` the Monte Carlo mean voxel variance of `L e` (2 draws per particle).
   Tiny-case score differences against the exact `C^+`: see the test report (`white_cov_*`).
3. The data-side map `L` is exact (no approximation); its covariance is checked by Monte Carlo against the dense
   `L Sigma L^*`.

Exact checks (dense, float64, tiny scene): adjointness of the projector pair (float32 coordinates: 1.3e-7),
PPCA score and gradients against a dense Gaussian density (2e-8 of the score range; 3e-11 and 2e-8 for the
gradients), and the unmasked, uncropped measurement with exact `C^+` reproducing the image likelihood up to a
data constant for targets in the range of the scene operator.

## 4. M-step

Fixed poses: `sum_i J_i^T J_i X G_i + X / tau^2 = sum_i J_i^T d_i g_i^T` on a fixed support sphere, solved by
CG, with `J_i` the whitened forward map of either arm (`J^T` by `jax.vjp`), `g_i = [1, E z_i]`,
`G_i = E[[1; z_i][1; z_i]^T]` (posterior covariance included). The prior `tau^2` is the same for both arms.
