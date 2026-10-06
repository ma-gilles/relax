"""Model and particle-state updates from the InitialModel E-step's sums and metadata.

RELION's ``MlOptimiser::maximization`` noise (sigma2) and class-probability
(pdf_class) updates for the native VDAM InitialModel, computed from the
E-step accumulator metadata. Optional reports live in diagnostics.vdam_noise.
``iteration_loop`` calls these once per iteration.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from relax.diagnostics import vdam_noise
from relax.helpers.orientation_priors import relion_round_away_from_zero
from relax.vdam.estep_common import EstepSums
from relax.vdam.schedules import DEFAULT_GRAD_MU
from relax.vdam.state import InitialModelState, NativeParticleState

MIN_SIGMA2_OFFSET_ANGSTROM2: float = 2.0


def _my_mu(mu: float, do_grad: bool, subset_size: int) -> float:
    my_mu = float(mu) if do_grad and subset_size != -1 else 0.0
    if my_mu < 0.0 or my_mu > 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")
    return my_mu


def with_uniform_class_direction_priors(
    state: InitialModelState, *, n_directions: int | None = None
) -> InitialModelState:
    """Set the joint class/direction prior to 1/(K*D), including restarts."""
    if n_directions is None:
        direction = np.asarray(state.pdf_direction)
        if direction.ndim != 2 or direction.shape[0] != state.K:
            raise ValueError("uniform class/direction prior requires a (K, D) direction grid")
        n_directions = int(direction.shape[1])
    if int(n_directions) < 1:
        raise ValueError("uniform class/direction prior requires at least one direction")
    return replace(
        state,
        pdf_class=np.full(state.K, 1.0 / float(state.K), dtype=np.float64),
        pdf_direction=np.full((state.K, n_directions), 1.0 / float(state.K * n_directions), dtype=np.float64),
    )


def update_noise_from_estep(
    state: InitialModelState,
    sums: EstepSums,
    *,
    do_grad: bool,
    mu: float = DEFAULT_GRAD_MU,
    report: dict | None = None,
) -> InitialModelState:
    """Update ``sigma2_noise`` from E-step weighted sums (engine units → RELION /N⁴).

    ``report`` (the E-step's meta) is only what a non-finite sum dumps (diagnostics.vdam_noise).

    One optics group's sums are ``[n]`` with a scalar ``noise_sumw``; several groups give
    ``[G, n]`` sums and ``[G]`` weights, and each group with noise sums is updated on its own
    (``maximizationOtherParameters``, ml_optimiser.cpp:6316-6372: a group whose sums are zero
    keeps its spectrum).
    """
    if sums.noise is None:
        return state
    wsum_sigma2_noise, wsum_img_power = sums.noise.wsum_sigma2_noise, sums.noise.wsum_img_power
    wsum_noise_a2, wsum_noise_xa, noise_sumw = sums.noise.wsum_noise_a2, sums.noise.wsum_noise_xa, sums.noise.sumw
    total_sumw = float(np.sum(noise_sumw))
    if total_sumw <= 0.0 or not np.all(np.isfinite(noise_sumw)):
        return state
    my_mu = _my_mu(mu, do_grad, state.subset_size)

    if wsum_sigma2_noise.shape != wsum_img_power.shape:
        raise ValueError(
            f"wsum_sigma2_noise and wsum_img_power shape mismatch: {wsum_sigma2_noise.shape} vs {wsum_img_power.shape}"
        )
    expected_shells = int(state.ori_size) // 2 + 1
    new_sigma2 = np.asarray(state.sigma2_noise, dtype=np.float64).copy()
    if new_sigma2.ndim != 2 or new_sigma2.shape[1] != expected_shells:
        raise ValueError(f"sigma2_noise must have shape (G, {expected_shells}), got {new_sigma2.shape}")
    n_groups = int(new_sigma2.shape[0])
    per_group = wsum_sigma2_noise.ndim == 2
    expected = (n_groups, expected_shells) if per_group else (expected_shells,)
    if wsum_sigma2_noise.shape != expected or (per_group and noise_sumw.shape != (n_groups,)):
        raise ValueError(
            f"noise weighted sums must have shape {expected} with {n_groups if per_group else 'one'} weight(s), "
            f"got {wsum_sigma2_noise.shape} and {noise_sumw.shape}"
        )
    if not per_group and n_groups != 1:
        raise ValueError(f"{n_groups} optics groups need per-group noise sums")
    if not np.all(np.isfinite(wsum_sigma2_noise)) or not np.all(np.isfinite(wsum_img_power)):
        summaries = [
            vdam_noise._array_finite_summary("wsum_sigma2_noise", wsum_sigma2_noise),
            vdam_noise._array_finite_summary("wsum_img_power", wsum_img_power),
            f"noise_sumw={noise_sumw!r}",
        ]
        if dump_path := vdam_noise._dump_noise_failure_meta(state, report or {}, summaries):
            summaries.append(f"dump={dump_path}")
        raise ValueError("noise weighted sums must be finite: " + "; ".join(summaries))

    from relax.reconstruction import noise_relion

    wsum_rows = wsum_sigma2_noise.reshape(-1, expected_shells)
    power_rows = wsum_img_power.reshape(-1, expected_shells)
    sumw_rows = noise_sumw.reshape(-1)
    new_state = replace(state)
    for g in range(n_groups):
        if np.sum(wsum_rows[g]) == 0.0:
            continue
        if sumw_rows[g] <= 0.0:
            raise ValueError(f"optics group {g + 1} has noise sums but no weight")
        shape = (int(state.ori_size), int(state.ori_size))
        wsum_g = noise_relion.normalize_wsum_to_sigma2_noise(wsum_rows[g], power_rows[g], float(sumw_rows[g]), shape, apply_floors=False)
        # RELION blends, then applies its floors (ml_optimiser.cpp:5255-5282): a shell without data decays by mu.
        new_sigma2[g] = noise_relion.apply_relion_sigma2_floors(
            new_sigma2[g] * my_mu + (1.0 - my_mu) * np.asarray(wsum_g, dtype=np.float64) / float(shape[0] ** 4),
            ctf_premultiplied=sums.average_ctf2 is not None,
        )
        if not np.all(np.isfinite(new_sigma2[g])) or np.any(new_sigma2[g] <= 0.0):
            raise ValueError("updated sigma2_noise must be positive and finite")
    new_state.sigma2_noise = new_sigma2
    vdam_noise._maybe_dump_noise_update_boundary(
        state,
        new_state,
        wsum_sigma2_noise=wsum_sigma2_noise,
        wsum_img_power=wsum_img_power,
        noise_sumw=float(total_sumw) if not per_group else noise_sumw,
        wsum_noise_a2=wsum_noise_a2,
        wsum_noise_xa=wsum_noise_xa,
    )
    return new_state


def update_probabilities_from_estep(
    state: InitialModelState,
    sums: EstepSums,
    *,
    do_grad: bool,
    mu: float = DEFAULT_GRAD_MU,
    uniform_class_direction_prior: bool = False,
) -> InitialModelState:
    """``MlOptimiser::maximizationOtherParameters`` for pdf_class / pdf_direction / sigma2_offset."""
    class_sums = sums.class_mass
    if class_sums is None:
        return state
    if class_sums.shape != (state.K,):
        raise ValueError(f"class_posterior_sums must have shape ({state.K},), got {class_sums.shape}")
    if not np.all(np.isfinite(class_sums)) or np.any(class_sums < 0.0):
        raise ValueError("class_posterior_sums must be non-negative and finite")
    sum_weight = float(np.sum(class_sums))
    if sum_weight <= 0.0:
        return state
    my_mu = _my_mu(mu, do_grad, state.subset_size)

    new_state = replace(state)
    if not uniform_class_direction_prior:
        new_pdf_class = np.asarray(state.pdf_class, dtype=np.float64) * my_mu
        new_pdf_class += (1.0 - my_mu) * class_sums / sum_weight
        pdf_class_sum = float(np.sum(new_pdf_class))
        if pdf_class_sum > 0.0:
            new_pdf_class /= pdf_class_sum
        new_state.pdf_class = new_pdf_class

    direction_sums = sums.direction_mass
    if uniform_class_direction_prior:
        if direction_sums is not None:
            if direction_sums.ndim != 2 or direction_sums.shape[0] != state.K:
                raise ValueError("class_direction_posterior_sums must have shape (K, n_directions)")
            if not np.all(np.isfinite(direction_sums)) or np.any(direction_sums < 0.0):
                raise ValueError("class_direction_posterior_sums must be non-negative and finite")
            n_directions = int(direction_sums.shape[1])
        elif state.pdf_direction is not None and np.asarray(state.pdf_direction).ndim == 2:
            n_directions = int(np.asarray(state.pdf_direction).shape[1])
        else:
            raise ValueError("uniform class/direction prior requires a direction grid")
        if n_directions < 1:
            raise ValueError("uniform class/direction prior requires at least one direction")
        new_state = with_uniform_class_direction_priors(new_state, n_directions=n_directions)
    elif direction_sums is not None and state.pdf_direction is not None:
        if direction_sums.ndim != 2 or direction_sums.shape[0] != state.K:
            raise ValueError(
                f"class_direction_posterior_sums must have shape ({state.K}, n_directions), got {direction_sums.shape}"
            )
        if not np.all(np.isfinite(direction_sums)) or np.any(direction_sums < 0.0):
            raise ValueError("class_direction_posterior_sums must be non-negative and finite")
        pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64)
        if pdf_direction.shape != direction_sums.shape:
            # RELION resizes pdf_direction to the new sampling.NrDirections()
            # and fills it uniformly when angular sampling changes.
            pdf_direction = np.full(direction_sums.shape, 1.0 / float(state.K * direction_sums.shape[1]))
        new_pdf_direction = pdf_direction * my_mu
        new_pdf_direction += (1.0 - my_mu) * direction_sums / sum_weight
        new_state.pdf_direction = new_pdf_direction

    wsum_sigma2_offset = sums.offset_wsum
    if wsum_sigma2_offset is not None:
        if not np.isfinite(wsum_sigma2_offset) or wsum_sigma2_offset < 0.0:
            raise ValueError("wsum_sigma2_offset must be non-negative and finite")
        sigma2_offset_sumw = sums.offset_sumw
        if not np.isfinite(sigma2_offset_sumw) or sigma2_offset_sumw <= 0.0:
            raise ValueError("sigma2_offset_sumw must be positive and finite")
        sigma2_offset = float(state.sigma2_offset) * my_mu
        # RELION divides by 2*sum_weight for 2D particle translations, 3*sum_weight for
        # subtomograms' 3D offsets (ml_optimiser.cpp:5222-5231).
        # Its sum_weight is accumulated from the same significant-pruned
        # reconstruction weights as wsum_sigma2_offset, rather than from the
        # unpruned per-image class responsibilities.
        offset_dims = float(sums.offset_dims)
        sigma2_offset += (1.0 - my_mu) * wsum_sigma2_offset / (offset_dims * sigma2_offset_sumw)
        new_state.sigma2_offset = max(float(sigma2_offset), MIN_SIGMA2_OFFSET_ANGSTROM2)

    return new_state


def _ensure_field(arr: np.ndarray | None, shape: tuple, dtype, fill=0) -> np.ndarray:
    if arr is None or arr.shape != shape:
        return np.full(shape, fill, dtype=dtype) if fill != 0 else np.zeros(shape, dtype=dtype)
    return arr


def _update_particle_state_from_estep_meta(
    particle_state: NativeParticleState,
    meta: dict,
    translations: np.ndarray,
) -> None:
    selected = meta.get("selected_particle_ids")
    if selected is None:
        return
    ids = np.asarray(selected, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return
    N = particle_state.translation_offsets.shape[0]
    if np.any(ids < 0) or np.any(ids >= N):
        raise ValueError("selected_particle_ids contains entries outside the particle state table")
    particle_state.visited = _ensure_field(particle_state.visited, (N,), bool)
    particle_state.visited[ids] = True

    if (pose := meta.get("pose_assignments")) is not None:
        assignments = np.asarray(pose, dtype=np.int64).reshape(-1)
        trans = np.asarray(translations, dtype=np.float64)
        translation_ids = np.mod(assignments, int(trans.shape[0]))
        base = relion_round_away_from_zero(particle_state.translation_offsets[ids])
        particle_state.translation_offsets[ids] = base + trans[translation_ids, :2]
        particle_state.pose_assignments = _ensure_field(particle_state.pose_assignments, (N,), np.int32, -1)
        particle_state.pose_assignments[ids] = assignments.astype(np.int32, copy=False)

    if (offsets := meta.get("tomo_offsets_px", meta.get("image_offsets_px"))) is not None:
        # Subtomogram particles (3D) and images on several shapes (rounded in each image's own
        # pixels): RELION's new offset, the rounded old one plus the winning shift.
        particle_state.translation_offsets[ids] = np.asarray(offsets, dtype=np.float64)

    if (rot := meta.get("best_pose_rotations")) is not None:
        particle_state.best_pose_rotations = _ensure_field(particle_state.best_pose_rotations, (N, 3, 3), np.float32)
        particle_state.best_pose_rotations[ids] = np.asarray(rot, dtype=np.float32)

    source_eulers = meta.get("best_pose_eulers_deg")
    if rot is not None or source_eulers is not None:
        particle_state.best_pose_eulers_valid = _ensure_field(particle_state.best_pose_eulers_valid, (N,), bool, False)
        particle_state.best_pose_eulers_valid[ids] = False
    if source_eulers is not None:
        eulers = np.asarray(source_eulers)
        if eulers.dtype != np.float64 or eulers.shape != (ids.size, 3) or not np.all(np.isfinite(eulers)):
            raise ValueError("source Euler metadata must be finite float64 [selected_particles, 3]")
        particle_state.best_pose_eulers_deg = _ensure_field(particle_state.best_pose_eulers_deg, (N, 3), np.float64)
        particle_state.best_pose_eulers_deg[ids] = eulers
        valid = np.asarray(meta.get("best_pose_eulers_valid", np.ones(ids.size, dtype=bool)))
        if valid.dtype != bool or valid.shape != (ids.size,):
            raise ValueError("source Euler validity must be boolean [selected_particles]")
        particle_state.best_pose_eulers_valid[ids] = valid

    if (bt := meta.get("best_pose_translations")) is not None:
        particle_state.best_pose_translations = _ensure_field(particle_state.best_pose_translations, (N, 2), np.float32)
        particle_state.best_pose_translations[ids] = np.asarray(bt, dtype=np.float32)

    if (rid := meta.get("best_pose_rotation_ids")) is not None:
        particle_state.best_pose_rotation_ids = _ensure_field(particle_state.best_pose_rotation_ids, (N,), np.int32, -1)
        particle_state.best_pose_rotation_ids[ids] = np.asarray(rid, dtype=np.int32).reshape(-1)
        particle_state.best_pose_rotation_orders = _ensure_field(
            particle_state.best_pose_rotation_orders, (N,), np.int32, -1
        )
        particle_state.best_pose_rotation_orders[ids] = int(meta.get("healpix_order", 0)) + int(
            meta.get("oversampling", 0)
        )

    if (cls := meta.get("class_assignments")) is not None:
        particle_state.class_assignments[ids] = np.asarray(cls, dtype=np.int32).reshape(-1)

    if (pmax := meta.get("max_posterior_per_image")) is not None:
        particle_state.max_posterior[ids] = np.asarray(pmax, dtype=np.float32).reshape(-1)

    if (dll := meta.get("log_likelihood_contribution")) is not None:
        particle_state.log_likelihood_contribution = _ensure_field(particle_state.log_likelihood_contribution, (N,), np.float64, 0.0)
        particle_state.log_likelihood_contribution[ids] = np.asarray(dll, dtype=np.float64).reshape(-1)
    if (nsig := meta.get("significant_counts")) is not None:
        particle_state.significant_counts = _ensure_field(particle_state.significant_counts, (N,), np.int32, 0)
        particle_state.significant_counts[ids] = np.asarray(nsig, dtype=np.int32).reshape(-1)


def relion_log_likelihood_contributions(log_evidence, *, sigma2_noise, groups, n_images, ori_size: int, current_size: int):
    """RELION's per-particle dLL, ``log(sum_weight) - min_diff2 - logsigma2`` (ml_optimiser.cpp:9029-9058).

    ``log_evidence`` is the E-step's ``log(sum_weight) - min_diff2`` per particle; ``logsigma2`` sums
    ``log(2 pi sigma2_noise[group][ires])`` over the current-size ``Mresol_fine`` pixels with ``ires > 0``
    once per image of the particle (``n_images``: 1, or a subtomogram's tilt images).
    """
    from relax.relion.relion_ctf import _fftw_shell_labels

    shells = _fftw_shell_labels(int(ori_size), int(current_size), centered_rows=False)
    sigma2 = np.atleast_2d(np.asarray(sigma2_noise, dtype=np.float64))
    shells = shells[(shells > 0) & (shells < sigma2.shape[1])]
    logsigma2 = np.log(2.0 * np.pi * sigma2[:, shells]).sum(axis=1)
    return np.asarray(log_evidence, np.float64) - np.asarray(n_images, np.float64) * logsigma2[np.asarray(groups, np.int64)]
