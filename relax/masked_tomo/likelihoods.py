"""Two measurement models of one subtomogram particle, sharing pose candidates, priors and the PPCA epilogue.

Arm A (``ImageArm``) is RELAX's sub-tilt image likelihood (``docs/math/vdam_ppca_algorithm.md`` section 16.1):
tilt ``k`` of a particle at pose ``(Q, u)`` is ``C_k S_{t_k(u)} P(rt_k Q) x`` with independent Gaussian noise of
spectrum ``Sigma`` per Fourier coefficient. Arm B (``MaskedArm``) observes the particle through a fixed linear
map of the larger scene patch: ``V = M F^-1 (D + lam)^-1/2 A^* Sigma^-1 y`` (whitened scene backprojection,
cropped and masked around the target), with mean ``F_g x = M F^-1 (D + lam)^-1/2 A^* Sigma^-1 A_g x`` and
covariance ``C = L Sigma L^*``. Two labelled approximations make B tractable at pilot size (section 3 of the
math note): ``A^* Sigma^-1 A ~ D`` (the backprojected CTF^2/Sigma weights, Fourier diagonal) in the mean, and
``C ~ c I`` on the mask, with ``c`` the per-particle mean voxel variance of ``V`` under detector noise
(Monte Carlo). Both arms hand ``inner = Re<f_p, data>_w`` and ``gram = Re<f_p, f_q>_w`` to RELAX's own
latent epilogue (:func:`relax.ppca_refinement.full_row_kernels.latent_scores`), so the PPCA score, posterior
mean and covariance (with the log determinant) are the same code for both.

Conventions (checked in ``tests/unit/test_masked_tomo.py``): a pose is a RECOVAR rotation ``Q`` (row
rotation ``rt_k @ Q``, ``rt_k`` the tilt rotation of an identity-pose particle) and a 3D shift ``u`` in working
pixels on the scene grid axes; tilt ``k`` sees the 2D shift ``(rt_k u)[:2]``. Volumes are real ``(n, n, n)``
arrays on the working grid (the centred inverse DFT of the central ``n``-crop of a bin-1 DFT).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import recovar.core.fourier_transform_utils as ftu
from recovar.core import slicing
from recovar.simulation import relion_tomo

from relax.ppca_refinement.full_row_kernels import latent_scores

F32, C64 = jnp.float32, jnp.complex64


# ----------------------------------------------------------------------------------------------- grids
def gridding_correction(box):
    """Real-space ``1 / sinc^2`` per axis for a trilinear Fourier interpolation on a ``box`` grid."""
    x = (np.arange(box) - box // 2) / box
    w = np.sinc(x) ** 2
    return (1.0 / (w[:, None, None] * w[None, :, None] * w[None, None, :])).astype(np.float32)


def sphere(box, radius):
    g = np.indices((box,) * 3) - box // 2
    return (np.sum(g**2, 0) <= radius**2).astype(np.float32)


def plane_lattice(n):
    """``(n*n, 2)`` scaled 2D frequencies of RECOVAR's translation phase (``exp(-2 pi i lattice . t)``)."""
    return np.asarray(ftu.get_k_coordinate_of_each_pixel((n, n), voxel_size=1, scaled=True), np.float32)


def grid3(box):
    """``(box**3, 3)`` integer frequency coordinates, component ``j`` along array axis ``j``."""
    return np.asarray(ftu.get_k_coordinate_of_each_pixel_3d((box,) * 3, voxel_size=1, scaled=False), np.float32)


def radius2(box):
    return np.sum(grid3(box) ** 2, -1).reshape((box,) * 3)


def pad_ft(x, box, *, gridding=True):
    """Centred DFT of a real ``(n,n,n)`` volume zero-padded to ``box`` (optionally gridding-corrected)."""
    n = x.shape[-1]
    lo = (box - n) // 2
    big = jnp.zeros(x.shape[:-3] + (box,) * 3, x.dtype).at[..., lo : lo + n, lo : lo + n, lo : lo + n].set(x)
    if gridding:
        big = big * gridding_correction(box)
    return ftu.get_dft3(big).astype(C64)


def trilinear(vol, coords):
    """Trilinear gather of a ``(G,G,G)`` complex grid at ``(m, 3)`` index coordinates (zero outside).

    ``vol`` may carry trailing channels ``(G,G,G,C)`` (e.g. every PPCA component): one gather per corner then
    fetches all channels, with the same per-channel arithmetic; the result is ``(m, C)``."""
    G = vol.shape[0]
    extra = (1,) * (vol.ndim - 3)
    b = jnp.floor(coords)
    f = coords - b
    b = b.astype(jnp.int32)
    out = 0
    for d0 in (0, 1):
        for d1 in (0, 1):
            for d2 in (0, 1):
                i = b + jnp.array([d0, d1, d2])
                w = (
                    (f[:, 0] if d0 else 1 - f[:, 0])
                    * (f[:, 1] if d1 else 1 - f[:, 1])
                    * (f[:, 2] if d2 else 1 - f[:, 2])
                )
                ok = jnp.all((i >= 0) & (i < G), -1)
                v = vol[jnp.clip(i[:, 0], 0, G - 1), jnp.clip(i[:, 1], 0, G - 1), jnp.clip(i[:, 2], 0, G - 1)]
                out = out + jnp.where(ok, w, 0).reshape(w.shape + extra) * v
    return out


def fourier_crop2(images, n):
    """Central ``n x n`` crop of the centred 2D DFT of real images ``(..., N, N)`` (complex64)."""
    N = images.shape[-1]
    lo = (N - n) // 2
    return np.asarray(ftu.get_dft2(jnp.asarray(images, jnp.float64)))[..., lo : lo + n, lo : lo + n].astype(np.complex64)


def fourier_crop3(volume, n):
    N = volume.shape[-1]
    lo = (N - n) // 2
    ft = np.asarray(ftu.get_dft3(jnp.asarray(volume, jnp.float64)))[lo : lo + n, lo : lo + n, lo : lo + n]
    return np.real(np.asarray(ftu.get_idft3(jnp.asarray(ft)))).astype(np.float32)


def shell_index(n):
    g = plane_lattice(n) * n
    return np.round(np.sqrt(np.sum(g**2, -1))).astype(int)


def ctf_rows(ctf_params, n, pixel):
    """RELION tomo CTF (with dose weight and CTF scale) of every row on an ``n x n`` working image."""
    shape = ctf_params.shape[:-1]
    c = relion_tomo.relion_tomo_ctf(jnp.asarray(ctf_params.reshape(-1, ctf_params.shape[-1])), (n, n), pixel)
    return np.asarray(c, np.float32).reshape(*shape, n * n)


# ----------------------------------------------------------------------------------------------- noise
def unaligned_noise(real_images, radius):
    """RELION-style start-up noise: shell means of ``|DFT|^2`` of images with the particle disc zeroed,
    divided by the kept pixel fraction. ``real_images`` ``(m, n, n)``; returns ``(n//2 + 1,)``."""
    m, n, _ = real_images.shape
    y, x = np.indices((n, n)) - n // 2
    keep = (x**2 + y**2) > radius**2
    ft = np.asarray(ftu.get_dft2(jnp.asarray(real_images * keep, jnp.float64))).reshape(m, -1)
    power = np.mean(np.abs(ft) ** 2, 0) / keep.mean()
    s = shell_index(n)
    out = np.array([power[s == i].mean() for i in range(n // 2 + 1)])
    return out


def noise_map(shell_noise, n_from, n_to):
    """The shell spectrum of ``n_from``-pixel images as a per-coefficient variance on an ``n_to`` grid of the
    same pixel size: interpolated by physical frequency and scaled by the coefficient count ``(n_to/n_from)^2``."""
    r = np.sqrt(np.sum((plane_lattice(n_to) * n_to) ** 2, -1)) * n_from / n_to
    v = np.interp(r, np.arange(shell_noise.size), shell_noise)
    return (v * (n_to / n_from) ** 2).astype(np.float32)


# ----------------------------------------------------------------------------------------------- data
@dataclasses.dataclass
class Particles:
    """Working-size data of one condition. ``images`` ``(P, K, n*n)`` and ``patches`` ``(P, K, S*S)`` are centred
    DFTs; ``rt`` ``(P, K, 3, 3)`` tilt rotations; ``ctf_img``/``ctf_patch`` CTF rows; ``noise_img``/``noise_patch``
    per-coefficient noise variance; ``pixel`` the working pixel size (A)."""

    images: np.ndarray
    patches: np.ndarray | None
    rt: np.ndarray
    ctf_img: np.ndarray
    ctf_patch: np.ndarray | None
    noise_img: np.ndarray
    noise_patch: np.ndarray | None
    shell_noise: np.ndarray
    n: int
    S: int
    pixel: float


def prepare(crops, patches, rots, ctf_params, voxel_size, n, S, noise_radius_px):
    """Fourier-crop bin-1 ``crops`` ``(P,K,N,N)`` and ``patches`` ``(P,K,M,M)`` to ``n`` and ``S`` and estimate the
    unaligned noise spectrum on the working crops (radius in working pixels)."""
    P, K, N, _ = crops.shape
    pixel = voxel_size * N / n
    images = fourier_crop2(crops.reshape(P * K, N, N), n)
    real = np.real(np.asarray(ftu.get_idft2(jnp.asarray(images, jnp.complex128)))).astype(np.float32)
    shell = unaligned_noise(real, noise_radius_px)
    out = Particles(
        images=images.reshape(P, K, n * n),
        patches=None,
        rt=np.asarray(rots, np.float32),
        ctf_img=ctf_rows(ctf_params, n, pixel),
        ctf_patch=None,
        noise_img=noise_map(shell, n, n),
        noise_patch=None,
        shell_noise=shell,
        n=n,
        S=S,
        pixel=pixel,
    )
    if patches is not None:
        M = patches.shape[-1]
        if not np.isclose(voxel_size * M / S, pixel):
            raise ValueError("patch and crop working pixel sizes differ")
        out.patches = fourier_crop2(patches.reshape(P * K, M, M), S).reshape(P, K, S * S)
        out.ctf_patch = ctf_rows(ctf_params, S, pixel)
        out.noise_patch = noise_map(shell, n, S)
    return out


# ----------------------------------------------------------------------------------------------- arm A
class ImageArm:
    """RELAX's sub-tilt image likelihood on the working ``n x n`` crops."""

    name = "image"

    def __init__(self, data: Particles, support_radius):
        self.d = data
        n = data.n
        self.n = n
        self.G = 2 * n
        self.lattice = jnp.asarray(plane_lattice(n))
        self.support = jnp.asarray(sphere(n, support_radius))
        r2 = np.sum((plane_lattice(n) * n) ** 2, -1)
        self.disc = jnp.asarray((r2 <= (n // 2 - 1) ** 2).astype(np.float32))  # RECOVAR's default max_r disc
        self.band = self.disc

    def particle(self, i):
        d = self.d
        w = self.band / d.noise_img
        return {
            "rt": jnp.asarray(d.rt[i]),
            "ctf": jnp.asarray(d.ctf_img[i]),
            "yw": jnp.asarray(d.images[i] * w, C64),
            "w": jnp.asarray(w, F32),
        }

    def volumes_ft(self, X):
        """``(P, G^3)`` padded, gridding-corrected DFTs of real volumes ``(P, n, n, n)``."""
        return pad_ft(X * self.support, self.G).reshape(X.shape[0], -1)

    def slice(self, v, rot_rows):
        """``(m, n*n)`` slices of one padded DFT ``v`` (``G = 2n`` grid) at ``(m, 3, 3)`` rotations, inside the
        projector's disc. RECOVAR infers the oversampling ``G / n`` on both its CUDA and JAX paths (recovar a9333ddf5)."""
        n, G = self.n, self.G
        img = slicing.slice_volume(v, rot_rows, (n, n), (G, G, G), "linear_interp", max_r=None)
        return img * self.disc

    def project(self, vft, rot_rows):
        """``(P, m, n*n)`` slices of padded DFTs at ``(m, 3, 3)`` rotations."""
        return jax.vmap(lambda v: self.slice(v, rot_rows))(vft)

    def phases(self, rt, us):
        """``(T, K, n*n)`` translation phases of 3D shifts ``us`` ``(T, 3)`` on every tilt."""
        t2 = jnp.einsum("kab,tb->tka", rt, us)[..., :2]
        return jnp.exp(-2j * jnp.pi * jnp.einsum("pa,tka->tkp", self.lattice, t2)).astype(C64)

    def stats(self, vft, part, Qs, us):
        """``inner (P, R, T)`` and ``gram (tri, R)`` for rotations ``Qs (R,3,3)`` and shifts ``us (T,3)``."""
        rt = part["rt"]
        K = rt.shape[0]
        R = Qs.shape[0]
        rows = jnp.einsum("kab,rbc->rkac", rt, Qs).reshape(R * K, 3, 3)
        proj = self.project(vft, rows).reshape(vft.shape[0], R, K, -1) * part["ctf"][None, None]
        # inner = Re sum conj(C ph P x) w y = Re sum conj(C P x) (conj(ph) w y)
        z = jnp.conj(self.phases(rt, us)) * part["yw"][None]
        inner = jnp.einsum("prkx,tkx->prt", jnp.conj(proj), z).real
        wproj = proj * part["w"][None, None, None]
        P = vft.shape[0]
        first, second = np.triu_indices(P)
        gram = jnp.stack([jnp.einsum("rkx,rkx->r", jnp.conj(proj[i]), wproj[j]).real for i, j in zip(first, second)])
        return inner, gram


# ----------------------------------------------------------------------------------------------- arm B
# Shift sets larger than this use the FFT-correlation path in MaskedArm.stats_multi (2 T Cc^3 pre-rolled copies and
# products scale with T; the correlations do not). Smaller sets use the products, which are faster there and match
# the reference to float32 on the GPU.
FFT_SHIFTS_ABOVE = 256


class MaskedArm:
    """Masked whitened scene reconstruction around the target (see the module docstring)."""

    name = "masked"

    def __init__(self, data: Particles, support_radius, mask_radius, *, lam_rel=0.1, crop=40, window=None,
                 n_mc=2, seed=0, dof_correction=True):
        d = self.d = data
        n, S = data.n, data.S
        self.n, self.S, self.G, self.Cc = n, S, 2 * n, crop
        self.support = jnp.asarray(sphere(n, support_radius))
        lo = (crop - 2 * int(np.ceil(mask_radius))) // 2
        if lo < 3:
            raise ValueError("crop too small for the mask and the shift search")
        self.mask = jnp.asarray(sphere(crop, mask_radius))
        self.k3 = jnp.asarray(grid3(S))
        # Template side: the periodic box of the predicted mean (the scene grid here; KernelBoxMaskedArm shrinks it).
        self.N, self.k3t = S, self.k3
        self._kaxes = jnp.asarray(np.stack([grid3(S).reshape(S, S, S, 3)[:, 0, 0, 0], grid3(S).reshape(S, S, S, 3)[0, :, 0, 1],
                                            grid3(S).reshape(S, S, S, 3)[0, 0, :, 2]]))
        self.lam_rel = lam_rel
        self.window = window  # None, or the side (working px) of a central detector window (matched field)
        self.max_r = S // 2 - 1
        self.k_ok = jnp.asarray((radius2(S) <= self.max_r**2).reshape(-1))
        # Template frequencies where the filter can be non-zero (the projector's ball; the filter is zero elsewhere by
        # construction): the trilinear gather runs only there. None means the whole template box.
        self.support_idx = jnp.asarray(np.nonzero(np.asarray(self.k_ok))[0].astype(np.int32))
        self.prefilter_k = None  # opt-in: shortlist size of the Fourier prefilter in pilot.search (not the last stage)
        self.seed = seed
        self.n_mc = n_mc
        self.dof_correction = dof_correction
        self._prep = {}

    # ---- fixed linear preprocessing (per particle, computed once) ----
    def _backproject(self, values, rt):
        S = self.S
        return slicing.adjoint_slice_volume(values, rt, (S, S), (S, S, S), "linear_interp").reshape(-1)

    def _window(self, ft_images):
        if self.window is None:
            return ft_images
        S = self.S
        real = ftu.get_idft2(ft_images.reshape(-1, S, S))
        y, x = np.indices((S, S)) - S // 2
        win = ((np.abs(x) < self.window // 2) & (np.abs(y) < self.window // 2)).astype(np.float32)
        return ftu.get_dft2(real * win).reshape(ft_images.shape)

    def _whiten(self, b, D, lam):
        return b / jnp.sqrt(D + lam)

    def _to_crop(self, qft, box=None):
        """Central ``Cc^3`` crop of the real part of the centred inverse DFT of ``(..., box^3)`` spectra (default
        the scene grid ``S``)."""
        S, Cc = box or self.S, self.Cc
        q = jnp.real(ftu.get_idft3(qft.reshape(qft.shape[:-1] + (S, S, S))))
        lo = (S - Cc) // 2
        return q[..., lo : lo + Cc, lo : lo + Cc, lo : lo + Cc]

    def weights(self, ctf, rt):
        """``D`` (backprojected ``CTF^2 / Sigma``, zero outside the projector's disc) and ``lam``."""
        winv = jnp.asarray(1.0 / self.d.noise_patch)
        D = jnp.real(self._backproject((ctf**2 * winv).astype(C64), rt))
        D = jnp.where(self.k_ok, D, 0.0)
        # lam: lam_rel times the mean weight over the observed band (D > 0 inside the projector's disc).
        observed = self.k_ok & (D > 0)
        lam = self.lam_rel * jnp.sum(jnp.where(observed, D, 0.0)) / jnp.sum(observed)
        return D, lam

    def measure(self, y_ft, ctf, rt, D, lam):
        """The fixed linear map ``L``: patch DFTs ``(K, S*S)`` -> whitened scene backprojection on the crop
        ``(Cc, Cc, Cc)`` (the mask is applied by the caller)."""
        winv = jnp.asarray(1.0 / self.d.noise_patch)
        y = self._window(y_ft)
        b = jnp.where(self.k_ok, self._backproject(ctf * y * winv, rt), 0)
        return self._to_crop(self._whiten(b, D, lam))

    def noise_sample(self, key, K):
        """Detector noise DFTs ``(K, S*S)`` with the noise model's spectrum ``noise_patch``."""
        S = self.S
        e = jax.random.normal(key, (K, S, S), F32)
        return ftu.get_dft2(e).reshape(K, -1) * jnp.sqrt(jnp.asarray(self.d.noise_patch) / S**2)

    def particle_from(self, rt, ctf, D, lam):
        """The template-side operands of a particle with these weights (no data)."""
        return {"filt": jnp.where(self.k_ok, D / jnp.sqrt(D + lam), 0.0).astype(F32), "rt": rt}

    def prepare_particle(self, i):
        if i in self._prep:
            return self._prep[i]
        d = self.d
        rt = jnp.asarray(d.rt[i])
        ctf = jnp.asarray(d.ctf_patch[i])
        D, lam = self.weights(ctf, rt)
        V = self.measure(jnp.asarray(d.patches[i]), ctf, rt, D, lam)
        # c: Monte Carlo mean voxel variance of V on the mask under the detector noise model.
        key = jax.random.PRNGKey(self.seed * 100003 + i)
        var = 0.0
        for _ in range(self.n_mc):
            key, sub = jax.random.split(key)
            Ve = self.measure(self.noise_sample(sub, rt.shape[0]), ctf, rt, D, lam)
            var = var + jnp.sum(self.mask * Ve**2) / jnp.sum(self.mask)
        c = var / self.n_mc
        if self.dof_correction:
            c = c * dof_ratio(D, lam, self.k_ok)
        out = {"rt": rt, "V": (V * self.mask).astype(F32), "c": c, "lam": lam, "D": D,
               "filt": self.particle_from(rt, ctf, D, lam)["filt"]}
        self._prep[i] = out
        return out

    def particle(self, i):
        p = self.prepare_particle(i)
        return {"V": p["V"], "c": p["c"], "filt": p["filt"]}

    # ---- template side ----
    def volumes_ft(self, X):
        G = self.G
        return pad_ft(X * self.support, G)  # (P, G, G, G)

    def filtered(self, vft, Q, u, filt):
        """Masked-crop mean ``F_g x`` of every component: ``(P, Cc, Cc, Cc)`` (before the mask)."""
        return self._filtered_channels(jnp.moveaxis(vft, 0, -1), Q, u, filt)

    def _filtered_channels(self, vol, Q, u, filt):
        """:meth:`filtered` of channels-last padded DFTs ``vol (G, G, G, P)``: one trilinear gather per corner
        fetches every component, and the ``P`` inverse DFTs run as one batch."""
        return self._to_crop(self._spectrum(self._gather(vol, Q), u, filt), self.N)

    def _support_k(self):
        return self.k3t if self.support_idx is None else self.k3t[self.support_idx]

    def _gather(self, vol, Q):
        """Trilinear samples ``(K, P)`` of channels-last ``vol`` at the rotated template frequencies on the support."""
        S, G = self.N, self.G
        # Precision.HIGHEST: at default precision a GPU dot may take TF32 operands (coordinates off by ~1e-3 relative).
        coords = jnp.matmul(self._support_k(), Q, precision=jax.lax.Precision.HIGHEST)
        return trilinear(vol, coords * (G / S) + G // 2)

    def _spectrum(self, t, u, filt):
        """``(P, N^3)`` template spectra ``t * phase(u) * filt``, zero off the support."""
        S = self.N
        phase = jnp.exp(-2j * jnp.pi * jnp.matmul(self._support_k(), u, precision=jax.lax.Precision.HIGHEST) / S)
        if self.support_idx is None:
            return jnp.moveaxis(t * phase[:, None] * filt[:, None], -1, 0)
        f = t * phase[:, None] * filt[self.support_idx][:, None]
        return jnp.zeros((t.shape[1], S**3), f.dtype).at[:, self.support_idx].set(
            f.T, indices_are_sorted=True, unique_indices=True)

    def stats(self, vft, part, Qs, us):
        """``inner (P, R, T)`` and ``gram (tri, R, T)``; shifts ``us`` must be ``delta + integer`` with one
        sub-pixel ``delta`` shared by all of them (``us - delta`` integers within ``+-(Cc/2 - mask)``)."""
        delta = us[0] - jnp.round(us[0])
        ints = jnp.round(us - delta).astype(jnp.int32)
        return self.stats_multi(vft, part, Qs, delta[None], ints, jnp.zeros(us.shape[0], jnp.int32))

    def stats_multi(self, vft, part, Qs, deltas, ints, gid):
        """``inner (P, R, T)`` and ``gram (tri, R, T)`` for shifts ``deltas[gid] + ints``: ``deltas (D, 3)``
        sub-pixel parts (Fourier phase), ``ints (T, 3)`` integer parts within ``+-(Cc/2 - mask)`` (a circular
        roll of the crop), ``gid (T,)`` the sub-pixel part of each shift.

        The trilinear gather of a rotation is shared by its ``D`` sub-pixel parts. With ``f`` rolled by an integer
        ``m`` and the 0/1 mask ``M``, ``inner_p(m) = sum_x M V roll(f_p, m) = sum_y f_p(y) roll(M V, -m)(y)`` and
        ``gram_ab(m) = sum_x M roll(f_a f_b, m) = sum_y (f_a f_b)(y) roll(M, -m)(y)``: an exact re-indexing, so the
        shifts are float32 (``Precision.HIGHEST``) matrix products against the rolled data and mask."""
        mask = self.mask
        c = part["c"]
        V = part["V"] / c
        P, D, T = vft.shape[0], deltas.shape[0], ints.shape[0]
        first, second = np.triu_indices(P)

        def rolled(a):
            return jax.vmap(lambda m: jnp.roll(a, shift=(-m[0], -m[1], -m[2]), axis=(0, 1, 2)).reshape(-1))(ints)

        vol = jnp.moveaxis(vft, 0, -1)
        S, G = self.N, self.G
        filt = part["filt"]

        def per_rot(Q):
            t = self._gather(vol, Q)  # (K, P)

            def per_delta(u):
                return self._to_crop(self._spectrum(t, u, filt), S)

            return jax.vmap(per_delta)(deltas)  # (D, P, Cc, Cc, Cc)

        if T > FFT_SHIFTS_ABOVE:
            # Large shift sets: every integer shift comes from FFT cross-correlations on the crop, sum_x V(x) f(x - m)
            # and sum_y M(y + m) (f_a f_b)(y), per sub-pixel part. These are circular on the crop, i.e. exactly the
            # rolls of the product path below, for any crop (a periodic shift when the crop is the whole box).
            # Complex FFTs: the real-FFT version differed from the product path by ~1e-3 relative on the GPU (jobs
            # 7482932, 7483092; CPU 1e-7), the complex one agrees to ~6e-7 (masked-tomo check before GPU 12).
            f = jax.vmap(per_rot)(Qs)  # (R, D, P, Cc, Cc, Cc)
            inner, gram = shift_correlations(f, mask * V, mask, ints)  # (R, D, P, T), (R, D, tri, T)
            gram = gram / c
            sel = gid[None, None, None, :]
            inner = jnp.take_along_axis(inner, jnp.broadcast_to(sel, (inner.shape[0], 1) + inner.shape[2:]), axis=1)[:, 0]
            gram = jnp.take_along_axis(gram, jnp.broadcast_to(sel, (gram.shape[0], 1) + gram.shape[2:]), axis=1)[:, 0]
            return jnp.transpose(inner, (1, 0, 2)), jnp.transpose(gram, (1, 0, 2))
        Vs, Ms = rolled(mask * V), rolled(mask)  # (T, Cc^3)
        f = jax.vmap(per_rot)(Qs)
        f = f.reshape(f.shape[:3] + (-1,))  # (R, D, P, X)
        hi = jax.lax.Precision.HIGHEST
        if D == 1:
            inner = jnp.einsum("rpx,tx->prt", f[:, 0], Vs, precision=hi)
            gram = jnp.einsum("rkx,tx->krt", f[:, 0, first] * f[:, 0, second], Ms, precision=hi) / c
            return inner, gram

        def pick(a):  # (k, R, D, T) -> (k, R, T): each shift reads its own sub-pixel part
            return jnp.take_along_axis(a, gid[None, None, None, :], axis=2)[:, :, 0]

        inner = pick(jnp.einsum("rdpx,tx->prdt", f, Vs, precision=hi))
        gram = pick(jnp.einsum("rdkx,tx->krdt", f[:, :, first] * f[:, :, second], Ms, precision=hi)) / c
        return inner, gram


def _fourier_stats(arm, vft, part, Qs, axes):
    """``inner (P, R, T)`` and the unmasked ``gram (tri, R)`` of ``arm`` (a ``MaskedArm``) on a Cartesian shift grid
    ``axes (3, Ta)`` (``T = Ta^3`` in ``pilot.shift_grid`` order), with no inverse DFT per rotation.

    ``inner`` is exact for every shift inside the crop's valid range (sub-pixel included): with ``a = M V / c`` on
    the template box and ``F_u = filt x^_Q e^{-2 pi i k.u / N}``, ``inner(u) = sum_x a Re F^-1[F_u] =
    Re sum_k conj(A(k)) F_u(k) / N^3``, evaluated as a separable partial DFT at the grid's ``Ta`` points per axis.
    ``gram`` drops the mask, ``sum_k Re F_a conj(F_b) / N^3`` (shift-independent): the APPROXIMATION of the opt-in
    prefilter (the template energy outside the mask and its shift dependence are ignored)."""
    S, Cc = arm.N, arm.Cc
    c = part["c"]
    lo = (S - Cc) // 2
    a = jnp.zeros((S, S, S), F32).at[lo : lo + Cc, lo : lo + Cc, lo : lo + Cc].set(arm.mask * part["V"] / c)
    conjA = jnp.conj(ftu.get_dft3(a.astype(C64))).reshape(-1)
    kax = arm._kaxes  # (3, S) integer frequency of each index along each axis
    E = jnp.exp(-2j * jnp.pi * kax[:, :, None] * axes[:, None, :] / S).astype(C64)  # (3, S, Ta)
    vol = jnp.moveaxis(vft, 0, -1)
    P = vft.shape[0]
    first, second = np.triu_indices(P)
    hi = jax.lax.Precision.HIGHEST
    zero = jnp.zeros(3, F32)

    def per_rot(Q):
        F = arm._spectrum(arm._gather(vol, Q), zero, part["filt"])  # (P, S^3)
        g = (conjA[None] * F).reshape(P, S, S, S)
        g = jnp.einsum("pxyz,xi->piyz", g, E[0], precision=hi)
        g = jnp.einsum("piyz,yj->pijz", g, E[1], precision=hi)
        g = jnp.einsum("pijz,zl->pijl", g, E[2], precision=hi)
        gram = jnp.real(jnp.sum(F[first] * jnp.conj(F[second]), -1)) / S**3
        return jnp.real(g).reshape(P, -1) / S**3, gram

    inner, gram = jax.vmap(per_rot)(Qs)  # (R, P, T), (R, tri)
    return jnp.transpose(inner, (1, 0, 2)), gram.T


class KernelBoxMaskedArm(MaskedArm):
    """Arm B with the data side on the full scene field and the template side on a small periodic box
    (``masked-tomo/notes/TWO_RES.md``, Option A; opt-in, ``MaskedArm`` is unchanged).

    ``V``, ``D``, ``lam`` and the Monte Carlo ``c`` are ``MaskedArm``'s on the scene grid ``S``. The predicted mean
    ``F_g x = M C (h_S (*)_S x_g)``, ``h_S = F_S^-1 [D / sqrt(D + lam)]``, reads ``h_S`` only at offsets
    ``|d|_inf <= r_mask + r_support + |shift|``; it is evaluated as ``M C (h_N (*)_N x_g)`` with ``h_N`` the central
    ``N^3`` box of ``h_S`` (its DFT replaces the per-particle filter). This is exact for the masked crop while that
    offset stays below ``N / 2``; the one other difference from ``MaskedArm`` is that the target spectrum is
    interpolated at the ``N``-grid frequencies. ``N = S`` is ``MaskedArm``. ``kernel_tail`` (in every prepared
    particle) is the fraction of ``||h_S||^2`` outside the central ``N^3`` box (labelled diagnostic for ``N`` below
    the exact size)."""

    name = "masked_kb"

    def __init__(self, data: Particles, support_radius, mask_radius, *, template_box, **kw):
        super().__init__(data, support_radius, mask_radius, **kw)
        N = int(template_box)
        if N > self.S or self.Cc > N or N % 2:
            raise ValueError("template box must be even, at most the scene grid and at least the crop")
        self.N, self.k3t = N, jnp.asarray(grid3(N))
        self.support_idx = None  # the kernel's spectrum on the N grid is not confined to the projector's ball
        g = grid3(N).reshape(N, N, N, 3)
        self._kaxes = jnp.asarray(np.stack([g[:, 0, 0, 0], g[0, :, 0, 1], g[0, 0, :, 2]]))
        # Largest integer shift (per axis) for which the kernel box is exact: r_mask + r_support + s < N / 2.
        self.exact_shift = N // 2 - 1 - int(np.ceil(mask_radius)) - int(np.ceil(support_radius))

    def kernel(self, filt):
        """``(N^3,)`` centred DFT of the central ``N^3`` box of ``h_S = F_S^-1 filt``, and the tail fraction."""
        S, N = self.S, self.N
        cdt = jnp.result_type(filt.dtype, C64)  # complex64 in production (float32 filter)
        h = ftu.get_idft3(filt.reshape(S, S, S).astype(cdt))  # offset 0 at index S // 2
        lo = (S - N) // 2
        box = h[lo : lo + N, lo : lo + N, lo : lo + N]
        total = jnp.sum(jnp.abs(h) ** 2)
        tail = 1.0 - jnp.sum(jnp.abs(box) ** 2) / total
        return ftu.get_dft3(box).reshape(-1).astype(cdt), tail

    def prepare_particle(self, i):
        out = super().prepare_particle(i)
        if "kernel_tail" not in out:
            out["filt"], out["kernel_tail"] = self.kernel(out["filt"])
        return out


def shift_correlations(f, MV, M, ints):
    """All circular integer shifts ``m`` of ``inner_p(m) = sum_x MV(x) f_p(x - m)`` and
    ``gram_ab(m) = sum_x M(x) f_a(x - m) f_b(x - m)`` by complex FFT cross-correlation (in ``f``'s precision),
    read at ``ints (T, 3)``. ``f`` is ``(..., P, C, C, C)``; returns ``(..., P, T)`` and ``(..., tri(P), T)``."""
    C = f.shape[-1]
    P = f.shape[-4]
    first, second = np.triu_indices(P)
    cdt = jnp.result_type(f.dtype, jnp.complex64)
    ax = (-3, -2, -1)
    idx = jnp.mod(ints, C)
    FV = jnp.fft.fftn(MV.astype(cdt))
    FM = jnp.fft.fftn(M.astype(cdt))
    inner = jnp.real(jnp.fft.ifftn(jnp.conj(jnp.fft.fftn(f.astype(cdt), axes=ax)) * FV, axes=ax))
    prods = f[..., first, :, :, :] * f[..., second, :, :, :]
    gram = jnp.real(jnp.fft.ifftn(jnp.conj(jnp.fft.fftn(prods.astype(cdt), axes=ax)) * FM, axes=ax))
    return inner[..., idx[:, 0], idx[:, 1], idx[:, 2]], gram[..., idx[:, 0], idx[:, 1], idx[:, 2]]


def dof_ratio(D, lam, k_ok):
    """In-band frequencies per effective degree of freedom of the whitened noise, ``N_band sum s^2 / (sum s)^2``
    with ``s = D / (D + lam)`` its spectrum over the projector's band (``1 / observed fraction`` when ``s`` is 0
    or 1). ``C ~ c I`` with the per-voxel variance ``c`` over-weights band-limited residuals by about this factor;
    scaling ``c`` by it is the labelled, partial temperature correction of section 3 of the math note. Dense tiny
    case (tests/unit/test_masked_tomo.py): approximate/exact score slope 1.86 (41 tilts) and 2.12 (5 tilts) before,
    1.52 and 1.28 after this correction: B's scores stay about 1.3-1.5x overconfident."""
    sp = jnp.where(k_ok, D / (D + lam), 0.0)
    return jnp.sum(k_ok) * jnp.sum(sp**2) / jnp.sum(sp) ** 2


# ----------------------------------------------------------------------------------------------- scores
def scores_from_stats(inner, gram, translation_dependent):
    """PPCA scores, latent means and covariances on the ``(R, T)`` grid with RELAX's latent epilogue.

    ``inner`` ``(P, R, T)``; ``gram`` ``(tri, R)`` (translation-free, arm A) or ``(tri, R, T)`` (arm B).
    Returns score ``(R, T)``, mean ``(q, R, T)`` and packed covariance ``(R, T, tri(q))``.
    """
    P, R, T = inner.shape
    if translation_dependent:
        inner_ = inner.reshape(P, R * T, 1, 1)
        gram_ = gram.reshape(gram.shape[0], R * T, 1)
        prior = jnp.zeros((1, R * T, 1), inner.dtype)
        s, m, c = latent_scores(inner_, gram_, prior)
        return s.reshape(R, T), m.reshape(P - 1, R, T), c.reshape(R, T, -1)
    s, m, c = latent_scores(inner[:, :, None, :], gram[:, :, None], jnp.zeros((1, R, T), inner.dtype))
    return s[:, 0, :], m[:, :, 0, :], jnp.broadcast_to(c[:, 0, None, :], (R, T, c.shape[-1]))


def unpack_tri(c, q):
    out = np.zeros(c.shape[:-1] + (q, q))
    for k, (i, j) in enumerate(zip(*np.triu_indices(q))):
        out[..., i, j] = out[..., j, i] = c[..., k]
    return out


# ----------------------------------------------------------------------------------------------- forward maps for the M-step
def image_forward(arm: ImageArm, part, Q, u):
    """Real-linear map ``x -> sqrt(w) C S P x`` stacked ``[Re, Im]`` for one particle at one pose, and its data."""
    rt = part["rt"]
    rows = jnp.einsum("kab,bc->kac", rt, Q)
    ph = arm.phases(rt, u[None])[0]
    sw = jnp.sqrt(part["w"])
    data = part["yw"] / jnp.where(part["w"] > 0, sw, 1.0)

    def f(x):
        v = arm.volumes_ft(x[None])[0]
        p = slicing.slice_volume(v, rows, (arm.n, arm.n), (arm.G,) * 3, "linear_interp") * part["ctf"] * ph * sw
        return jnp.concatenate([p.real.reshape(-1), p.imag.reshape(-1)])

    return f, jnp.concatenate([data.real.reshape(-1), data.imag.reshape(-1)]).astype(F32)


def masked_forward(arm: MaskedArm, part, Q, u):
    sc = 1.0 / jnp.sqrt(part["c"])

    def f(x):
        v = arm.volumes_ft(x[None])
        return (arm.filtered(v, Q, u, part["filt"])[0] * arm.mask * sc).reshape(-1)

    return f, (part["V"] * sc).reshape(-1).astype(F32)

