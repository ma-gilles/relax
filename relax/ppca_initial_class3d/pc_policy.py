"""How the mixture's principal components enter the model.

Four policies, each opt-in through :class:`relax.ppca_initial_class3d.config.Config`:

* Gradual PCs (``pc_start_fraction``). The run starts with no loading (q = 0). From update
  ``round(pc_start_fraction * iterations) + 1`` one PC opens at a time, at evenly spaced updates up to
  ``round(pc_end_fraction * iterations) + 1``, so each count of PCs runs for several updates. At the end of the
  phase of PC ``j`` (the update before PC ``j + 1`` would open; for the last PC, one spacing after its opening)
  every class with ``j`` open PCs tests its PC ``j``: the update's minibatch is scored a second time by the same
  model with that PC closed (its column zero and ``w_j zbar_j`` moved into the mean, so the closed model keeps
  the batch's average structure and loses only the variation along the PC), and the class's gain is the mean,
  over the batch's particles weighted by their probability of the class, of each particle's log-evidence with
  the PC minus without it. The evidence integrates the latent, so a PC that explains nothing lowers it by its
  share of ``log det(I + W^T P^T S^-1 P W)``. PC 1 stays if its gain is positive; PC ``j > 1`` stays if its gain
  is positive and at least ``PC_GAIN_SHARE`` of the class's PC-1 gain; no PC stays in a class holding less than
  ``MIN_CLASS_PARTICLES`` of the batch. A PC that does not stay is closed again and the class opens no further
  PC: one more PC made little difference. Each class keeps its own count.

  A loading column that is exactly zero is a fixed point of the E- and M-step (its latent posterior mean is
  zero, so its residual gradient and VDAM direction are zero), so the engine runs unchanged with ``q`` columns
  and the closed ones are held at zero. An opening column starts as a smooth random field (low-passed at the
  seed model's bootstrap radius, ``OPEN_RADIUS_FRACTION`` of the box) inside the support, orthogonal to the
  class mean and its open loadings, with ``PC_OPEN_NORM_SHARE`` of the mean's norm: small, so the data grow it.
  Its VDAM first moments start at the opening update's directions, as every channel's do in a run's first
  update (iteration_loop.update_models), so it is not gated back to zero.
* Gradual PC band (``pc_band_test``). The loadings follow the stage radius up as the mean does, but every rise
  is tested: at the last update of each later stage (for the last stage, the run's second-to-last update) every
  class with open PCs and an untested band scores the minibatch a second time with its loadings cut at the
  previous stage's radius (``W_high zbar`` moved into the mean, as above), and keeps the new shells if their gain
  is positive and the class holds at least ``MIN_CLASS_PARTICLES`` of the batch. Otherwise its loadings stay at
  the previous radius for the rest of the run while the mean goes on: there the M-step solves the mean alone
  (its coupling to the loadings removed from the normal equations) and holds the loadings and their VDAM first
  moments at zero. This targets loadings that turn into high-frequency noise once the radius opens.
* Latent centering (``center_latents``). After each M-step the class mean absorbs ``W zbar``, the batch's
  responsibility- and CTF-weighted mean latent of the class (read from the augmented normal equations), so the
  loadings describe variation about the mean and the latent prior ``N(0, I)`` holds on average.
* Rigid-motion projection (``project_rigid_motions``). After each M-step every open loading loses its
  components along the six first-order rigid motions of the class mean (three translations, three rotations
  about the box centre), so a loading cannot describe a small shift or rotation of the particle as structural
  variation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.triangular import pack_upper_tri, unpack_tri_to_full

from relax.ppca_initial_model.initialization import bandlimit_and_mask

PC_GAIN_SHARE = 0.1  # a later PC must add at least this share of the class's PC-1 log-evidence gain
MIN_CLASS_PARTICLES = 5.0  # a class with less responsibility mass in the batch cannot keep a tested PC
PC_OPEN_NORM_SHARE = 0.1  # an opening PC's norm, as a share of the class mean's
OPEN_RADIUS_FRACTION = 0.07  # an opening PC's low-pass radius, as a share of the box: the seed's bootstrap radius


def pc_open_updates(config) -> tuple[int, ...]:
    """The update at which each PC (1..q) may open: all at update 1 unless ``pc_start_fraction`` is set."""
    q, n = int(config.q), int(config.iterations)
    if config.pc_start_fraction is None:
        return (1,) * q
    start = int(round(float(config.pc_start_fraction) * n)) + 1
    end = int(round(float(config.pc_end_fraction) * n)) + 1
    if q == 1:
        return (start,)
    return tuple(int(round(start + j * (end - start) / (q - 1))) for j in range(q))


def scheduled_pcs(config, iteration: int) -> int:
    """How many PCs the schedule allows in ``iteration`` (before any class's test closed one)."""
    return int(sum(1 for opening in pc_open_updates(config) if opening <= iteration))


def pc_test_updates(config) -> dict[int, int]:
    """``{update: j}``: the update at which each class tests its PC ``j`` (the last update of that PC's phase)."""
    if config.pc_start_fraction is None:
        return {}
    openings = pc_open_updates(config)
    tests = {}
    for j, opening in enumerate(openings, start=1):
        if j < len(openings):
            end = openings[j] - 1
        else:
            spacing = openings[-1] - openings[-2] if len(openings) > 1 else max(2, int(config.iterations) - opening)
            end = opening + spacing - 1
        if opening < end <= int(config.iterations):
            tests[end] = j
    return tests


def pc_band_test_updates(config, grid_size: int) -> dict[int, tuple[int, int]]:
    """``{update: (previous radius, radius)}``: the update at which each class tests the shells its loadings gained
    when the stage radius last rose (module docstring); radii clamped to the box as the controller clamps them."""
    if not config.pc_band_test:
        return {}
    limit = int(grid_size) // 2 - 1
    stages = [(int(start), min(int(radius), limit)) for start, radius, _ in config.stages]
    tests = {}
    for i in range(1, len(stages)):
        start, radius = stages[i]
        previous = stages[i - 1][1]
        end = stages[i + 1][0] - 1 if i + 1 < len(stages) else int(config.iterations) - 1
        if radius > previous and start < end <= int(config.iterations):
            tests[end] = (previous, radius)
    return tests


def class_pc_radii(band_caps, n_classes: int, radius: int) -> list:
    """Each class's loading radius in an update at stage radius ``radius``: None (the stage radius) unless a band
    test held its loadings at a smaller one."""
    caps = band_caps if band_caps is not None else [None] * n_classes
    return [None if cap is None or int(cap) >= int(radius) else int(cap) for cap in caps]


def class_active_pcs(config, iteration: int, caps) -> list[int]:
    """Each class's PC count in ``iteration``: the schedule's, at most the class's cap."""
    scheduled = scheduled_pcs(config, iteration) if config.pc_start_fraction is not None else int(config.q)
    return [min(scheduled, int(cap)) for cap in caps]


def _to_real(spectra, volume_shape):
    """``(n_half, C)`` half spectra -> ``(C, N, N, N)`` real volumes, as :func:`bandlimit_and_mask`."""
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    return np.asarray(ftu.get_idft3_real(jnp.asarray(spectra).T.reshape((-1,) + half_shape), volume_shape))


def _to_spectra(volumes):
    """``(C, N, N, N)`` real volumes -> ``(n_half, C)`` half spectra."""
    volumes = jnp.asarray(np.asarray(volumes, np.float32))
    return ftu.get_dft3_real(volumes).reshape(volumes.shape[0], -1).T


def close_inactive(theta, moments, active):
    """Hold the loadings of each class's closed PCs (and their VDAM first moments) at exactly zero."""
    theta = jnp.asarray(theta)
    closed = []
    for k, history in enumerate(moments):
        theta = theta.at[k, :, 1 + int(active[k]):].set(0)
        first = jnp.asarray(history.first).at[..., 1 + int(active[k]):].set(0)
        closed.append(type(history)(first, history.second, history.initialized))
    return theta, tuple(closed)


def open_pcs(theta, config, *, previously_active, active, volume_shape, radius: int, mask):
    """Start the loadings of each class's PCs ``previously_active[k] + 1 .. active[k]`` (module docstring).

    The random field of class ``k``, PC ``j`` is drawn from ``default_rng([seed, k, j])``, so an opening repeats
    exactly when a run resumes from the checkpoint before it.
    """
    theta = jnp.asarray(theta)
    if all(int(a) <= int(p) for a, p in zip(active, previously_active)):
        return theta
    n = int(volume_shape[0])
    updated = []
    for k, model in enumerate(theta):
        model = np.array(model)
        before, after = int(previously_active[k]), int(active[k])
        if after > before:
            real = _to_real(model[:, : 1 + before], volume_shape)  # mean and the open loadings
            basis = []
            for volume in real:
                vector = volume.reshape(-1).astype(np.float64)
                for b in basis:
                    vector -= b * (b @ vector)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    basis.append(vector / norm)
            scale = PC_OPEN_NORM_SHARE * float(np.linalg.norm(real[0]))
            low_pass = min(int(radius), max(1, int(np.floor(OPEN_RADIUS_FRACTION * n + 0.5))))
            for j in range(before, after):
                rng = np.random.default_rng([int(config.seed), k, j + 1])
                field = rng.standard_normal((n, n, n)).astype(np.float32)
                smooth = np.asarray(bandlimit_and_mask(_to_spectra(field[None]), volume_shape, low_pass, mask))
                vector = _to_real(smooth, volume_shape)[0].reshape(-1).astype(np.float64)
                for b in basis:
                    vector -= b * (b @ vector)
                vector /= np.linalg.norm(vector)
                basis.append(vector)
                column = (vector * scale).reshape(n, n, n)
                model[:, 1 + j] = np.asarray(_to_spectra(column[None]))[:, 0]
        updated.append(model)
    return jnp.asarray(np.stack(updated), theta.dtype)


def close_tested_pc(theta, classes, pc: int, centers):
    """The model of the PC test (module docstring): PC ``pc`` of every class in ``classes`` closed, its column
    zero and ``w zbar`` (``centers[k]``: the class's mean latent in the batch) moved into the class mean."""
    theta = jnp.asarray(theta)
    for k in classes:
        weight = jnp.asarray(float(centers[k][pc - 1]), theta.dtype)
        theta = theta.at[k, :, 0].add(weight * theta[k, :, pc]).at[k, :, pc].set(0)
    return theta


def truncate_pc_band(theta, classes, radius: int, radii, centers):
    """The model of the band test (module docstring): the loadings of every class in ``classes`` cut at ``radius``
    (zero where ``radii`` exceeds it), ``W_high zbar`` (``centers[k]``: the class's mean latent) moved into the
    class mean."""
    theta = jnp.asarray(theta)
    high = jnp.asarray(np.asarray(radii) > radius)[:, None]
    for k in classes:
        loadings = theta[k, :, 1:]
        cut = jnp.where(high, loadings, 0)
        theta = theta.at[k, :, 0].add(cut @ jnp.asarray(np.asarray(centers[k]), theta.dtype))
        theta = theta.at[k, :, 1:].set(loadings - cut)
    return theta


def limit_pc_band(theta, moments, pc_radii, radii):
    """Hold each class's loadings, and their VDAM first moments, at zero beyond its loading radius (None: none)."""
    theta = jnp.asarray(theta)
    limited = []
    for k, history in enumerate(moments):
        if pc_radii[k] is None:
            limited.append(history)
            continue
        high = jnp.asarray(np.asarray(radii) > pc_radii[k])
        theta = theta.at[k, :, 1:].set(jnp.where(high[:, None], 0, theta[k, :, 1:]))
        loading = jnp.arange(theta.shape[2]) >= 1
        first = jnp.where(high[None, :, None] & loading[None, None, :], 0, jnp.asarray(history.first))
        limited.append(type(history)(first, history.second, history.initialized))
    return theta, tuple(limited)


def uncouple_mean(lhs_tri, q: int, high):
    """Packed normal equations with the mean's coupling to the loadings removed at the ``high`` frequencies, where
    the loadings stay zero: there the mean's direction is its own, as in a model without loadings."""
    full = unpack_tri_to_full(jnp.asarray(lhs_tri), q + 1)
    index = jnp.arange(q + 1)
    cross = (index[:, None] == 0) != (index[None, :] == 0)
    full = jnp.where(jnp.asarray(high)[:, None, None] & cross[None], 0, full)
    return pack_upper_tri(full)


def evidence_gains(stats, without, n_classes: int):
    """Per class: the tested PC's mean log-evidence gain over the batch's particles, each weighted by its
    probability of the class in ``stats`` (the update's E-step, with the PC), from its log-evidence there minus
    ``without[particle]`` (the batch scored with the PC closed); and the class's probability mass in the batch."""
    with_pc = {}
    for half in stats:
        for ids, log_likelihood in half.image_log_likelihood:
            with_pc.update(zip(np.asarray(ids).tolist(), np.asarray(log_likelihood, np.float64).tolist()))
    weighted, mass = np.zeros(n_classes), np.zeros(n_classes)
    for half in stats:
        for ids, probabilities, *_ in half.membership_tiles:
            for particle, row in zip(np.asarray(ids).tolist(), np.asarray(probabilities, np.float64)):
                weighted += row * (with_pc[particle] - without[particle])
                mass += row
    return np.where(mass > 0, weighted / np.where(mass > 0, mass, 1.0), 0.0), mass


def keep_tested_pc(pc: int, gain: float, first_gain, mass: float) -> bool:
    """The PC test (module docstring)."""
    if mass < MIN_CLASS_PARTICLES or not gain > 0:
        return False
    if pc == 1:
        return True
    return first_gain is not None and gain >= PC_GAIN_SHARE * float(first_gain)


def keep_band(gain: float, mass: float) -> bool:
    """The band test (module docstring)."""
    return mass >= MIN_CLASS_PARTICLES and gain > 0


def variance_shares(model, latent_second_moment, active: int):
    """Shares of the class's heterogeneity variance along its principal directions, largest first: eigenvalues
    of ``S^1/2 Re(W^H W) S^1/2`` with ``S`` the latents' weighted second moment."""
    if active < 1:
        return []
    w = np.asarray(model)[:, 1: 1 + active]
    gram = np.real(w.conj().T @ w)
    s = np.asarray(latent_second_moment, np.float64)[:active, :active]
    values, vectors = np.linalg.eigh((s + s.T) / 2)
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0))) @ vectors.T
    eig = np.sort(np.linalg.eigvalsh(root @ gram @ root))[::-1]
    total = float(np.sum(np.maximum(eig, 0)))
    return [float(max(e, 0) / total) if total > 0 else 0.0 for e in eig]


def latent_moments(stats, q: int, support):
    """Per class, the batch's mean latent and latent second moment: the augmented normal equations' ``[0, 1:]``
    row and ``[1:, 1:]`` block over their ``[0, 0]`` entry, summed over the pseudo-halves and the update's support
    (responsibility- and CTF/noise-weighted)."""
    support = jnp.asarray(np.asarray(support, bool))
    means, seconds = [], []
    for k in range(len(stats[0].components)):
        cross, block, total = 0.0, 0.0, 0.0
        for half in stats:
            lhs = jnp.real(unpack_tri_to_full(half.components[k].lhs_tri, q + 1))
            cross = cross + jnp.sum(jnp.where(support[:, None], lhs[:, 0, 1:], 0), axis=0)
            block = block + jnp.sum(jnp.where(support[:, None, None], lhs[:, 1:, 1:], 0), axis=0)
            total = total + jnp.sum(jnp.where(support, lhs[:, 0, 0], 0))
        scale = max(float(total), np.finfo(np.float32).tiny)
        means.append(np.asarray(cross) / scale)
        seconds.append(np.asarray(block) / scale)
    return np.stack(means).astype(np.float32), np.stack(seconds).astype(np.float32)


def center_latents(theta, centers):
    """Move ``W zbar`` into each class mean: the model describes the same images with latents centred on zero."""
    theta = jnp.asarray(theta)
    shift = jnp.einsum("kfq,kq->kf", theta[..., 1:], jnp.asarray(centers, theta.dtype))
    return theta.at[..., 0].add(shift)


def rigid_motion_basis(mean_volume, mask):
    """Orthonormal basis ``(6, N³)`` of the first-order rigid motions of ``mean_volume`` inside ``mask``:
    the three translations (gradients) and three rotations about the box centre ``(r x grad)``."""
    volume = np.asarray(mean_volume, np.float64)
    n = volume.shape[0]
    spectrum = np.fft.fftn(volume)
    frequencies = np.meshgrid(*(np.fft.fftfreq(n),) * 3, indexing="ij")
    gradient = [np.fft.ifftn(2j * np.pi * f * spectrum).real for f in frequencies]
    coordinates = np.meshgrid(*(np.arange(n) - n // 2,) * 3, indexing="ij")
    fields = list(gradient)
    for a, b in ((1, 2), (2, 0), (0, 1)):
        fields.append(coordinates[a] * gradient[b] - coordinates[b] * gradient[a])
    weights = np.asarray(mask, np.float64)
    matrix = np.stack([(field * weights).reshape(-1) for field in fields], axis=1)
    q_matrix, r_matrix = np.linalg.qr(matrix)
    keep = np.abs(np.diag(r_matrix)) > 1e-8 * np.max(np.abs(np.diag(r_matrix)))
    return q_matrix[:, keep].T


def project_rigid_motions(theta, active, *, volume_shape, radius: int, mask):
    """Remove from each class's open loadings their components along the class mean's rigid motions.

    Returns the projected models and, per class, the fraction of the loadings' energy removed.
    """
    theta = jnp.asarray(theta)
    updated, removed = [], []
    for k, model in enumerate(theta):
        model = np.array(model)
        count = int(active[k])
        if count < 1:
            updated.append(model)
            removed.append(0.0)
            continue
        real = _to_real(model[:, : 1 + count], volume_shape)
        basis = rigid_motion_basis(real[0], mask)
        loadings = real[1:].reshape(count, -1).astype(np.float64)
        before = float(np.sum(loadings ** 2))
        projected = loadings - (loadings @ basis.T) @ basis
        after = float(np.sum(projected ** 2))
        removed.append(0.0 if before == 0 else 1.0 - after / before)
        spectra = np.asarray(_to_spectra(projected.reshape((count,) + tuple(volume_shape))))
        model[:, 1: 1 + count] = np.asarray(bandlimit_and_mask(jnp.asarray(spectra), volume_shape, radius, mask))
        updated.append(model)
    return jnp.asarray(np.stack(updated), theta.dtype), removed
