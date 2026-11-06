"""Plain SGD along the per-frequency Newton direction, with a half-batch PC gate (docs/math/ppca_momentum_sgd.md, "Newton SGD").

The step of update ``t`` is ``lr(t) k H_f^-1 G_f`` on every active Fourier voxel ``f``: ``G`` sums the two
half-batch residual gradients and ``H_f`` the two half-batch ``(1+q) x (1+q)`` Gauss-Newton metrics. ``k`` is fixed
at the first update so that the step's norm equals that of the split-curvature step ``lr G / c`` there, with
``c = (c_mean, c_pc, ..., c_pc)`` the band maxima of the mean's metric diagonal and of the PCs' mean diagonal.
The PC columns of the step are then multiplied, per shell, by ``clip(rho, floor, 1)``, ``rho`` the correlation of
the two half-batch PC gradients over the shell, each sum an exponential moving average over about 20 updates.
"""

import json

import jax.numpy as jnp
import numpy as np
from recovar.ppca.triangular import unpack_tri_to_full

from relax.ppca_initial_model.sgd_update import metric_diagonal

GATE_EMA_WEIGHT = 1 / 20


def split_curvature_step(gradient, diagonal, active, learning_rate, floor):
    """``learning_rate G / c`` and ``(c_mean, c_pc)``: ``c_mean`` the band maximum of ``H[0, 0]``, ``c_pc`` of
    ``tr H[1:, 1:] / q`` (one scalar for every PC keeps the step invariant to rotating the PCs)."""
    active = jnp.asarray(active, bool)
    diagonal = jnp.asarray(diagonal)
    q = diagonal.shape[1] - 1
    c_mean = jnp.maximum(jnp.max(jnp.where(active, diagonal[:, 0], 0.0)), jnp.float32(floor))
    c_pc = jnp.maximum(jnp.max(jnp.where(active, jnp.sum(diagonal[:, 1:], axis=1) / q, 0.0)), jnp.float32(floor))
    scale = jnp.concatenate([c_mean[None], jnp.repeat(c_pc, q)]).astype(jnp.float32)
    step = jnp.float32(learning_rate) * jnp.where(active[:, None], jnp.asarray(gradient), 0) / scale[None, :]
    return step, float(c_mean), float(c_pc)


def damped_newton_direction(lhs_tri, gradient, active, floor, relative_floor=None):
    """``H_f^-1 G_f`` per active frequency, eigenvalues floored at ``floor``.

    A materially indefinite ``H_f`` (smallest eigenvalue below ``-32 eps max|eigenvalue|``) is damped instead of
    refused: ``floor - smallest eigenvalue`` is added to all its eigenvalues. Returns the direction and a record of
    the damping and of the floored eigenvalues.
    """
    gradient = jnp.asarray(gradient)
    active = jnp.asarray(active, bool)
    p = gradient.shape[1]
    floor_ = np.float32(floor)
    metric = unpack_tri_to_full(jnp.asarray(lhs_tri), p)
    metric = jnp.where(active[:, None, None], (metric + metric.swapaxes(-1, -2)) * 0.5, 0)
    values, vectors = jnp.linalg.eigh(metric)
    if relative_floor is not None:
        # A floor relative to the band's stiffest direction bounds every step at 1 / (r max eigenvalue).
        top = float(jnp.max(jnp.where(active, values[:, -1], 0.0)))
        floor_ = np.float32(max(float(floor_), relative_floor * top))
    bound = 32 * np.finfo(np.float32).eps * jnp.maximum(jnp.max(jnp.abs(values), axis=-1, keepdims=True), floor_)
    damping = jnp.where(values[:, :1] < -bound, floor_ - values[:, :1], 0.0)
    damped = jnp.maximum(values + damping, floor_)
    projected = jnp.einsum("fji,fj->fi", vectors, jnp.where(active[:, None], gradient, 0))
    direction = jnp.where(active[:, None], jnp.einsum("fij,fj->fi", vectors, projected / damped), 0)
    host_active = np.asarray(active)
    host_values = np.asarray(values)[host_active]
    host_damping = np.asarray(damping)[host_active, 0]
    if not np.all(np.isfinite(host_values)) or not bool(jnp.all(jnp.isfinite(direction))):
        raise ValueError("Nonfinite PPCA metric or Newton direction")
    return direction.astype(gradient.dtype), {
        "indefinite_frequencies": int(np.sum(host_damping > 0)),
        "max_damping": float(host_damping.max()) if host_damping.size else 0.0,
        "minimum_eigenvalue": float(host_values[:, 0].min()) if host_values.size else 0.0,
        "floored_eigenvalues": int(np.sum(host_values < floor_)),
        "eigenvalue_floor": float(floor_),
    }


class PCGate:
    """The per-shell PC step factor ``clip(rho, floor, 1)`` from the two half-batch PC gradients.

    ``rho = EMA(Re <g0, g1>) / sqrt(EMA |g0|^2 EMA |g1|^2)`` per shell, sums over the shell's frequencies and the PC
    channels, EMA weight 1/20. The mean is never gated.
    """

    def __init__(self, floor, n_shells, state=None):
        if not 0 < floor <= 1:
            raise ValueError("The PC gate floor must be in (0, 1]")
        self.floor = float(floor)
        self.n_shells = int(n_shells)
        self.sums = None if state is None else {key: jnp.asarray(value) for key, value in state.items()}

    def update(self, half_gradients, shells):
        if len(half_gradients) != 2:
            raise ValueError("The PC gate needs the two half-batch gradients")
        g0, g1 = (jnp.asarray(g)[:, 1:] for g in half_gradients)
        shells = jnp.asarray(shells)

        def per_shell(values):
            return jnp.zeros(self.n_shells, jnp.float32).at[shells].add(values.astype(jnp.float32))

        new = {
            "num": per_shell(jnp.real(jnp.sum(jnp.conj(g0) * g1, axis=1))),
            "d0": per_shell(jnp.sum(jnp.abs(g0) ** 2, axis=1)),
            "d1": per_shell(jnp.sum(jnp.abs(g1) ** 2, axis=1)),
        }
        w = jnp.float32(GATE_EMA_WEIGHT)
        self.sums = new if self.sums is None else {k: (1 - w) * self.sums[k] + w * new[k] for k in new}
        return self.raw()

    def raw(self):
        return self.sums["num"] / jnp.sqrt(jnp.maximum(self.sums["d0"] * self.sums["d1"], 1e-30))

    def factors(self):
        return jnp.clip(self.raw(), self.floor, 1.0)

    def state(self):
        return {key: np.asarray(value) for key, value in self.sums.items()}


class LearningRateSchedule:
    """``lr(t) = lr0 s(t) / s(1)`` from a per-update step array ``s`` (index ``t - 1``; held after its end).

    ``lr0`` is the configured rate, or, when the run matches it, set at the first update (:meth:`NewtonSGD.step`)."""

    def __init__(self, base, path):
        self.base = None if base is None else float(base)
        self.path = path
        self.steps = None if path is None else np.asarray(np.load(path), np.float64).reshape(-1)
        if self.steps is not None and (self.steps.size == 0 or not np.all(np.isfinite(self.steps)) or self.steps[0] <= 0):
            raise ValueError(f"The learning-rate schedule {path} must be a finite array with a positive first entry")

    def __call__(self, t):
        if self.steps is None:
            return self.base
        return self.base * float(self.steps[min(t, self.steps.size) - 1] / self.steps[0])


class NewtonSGD:
    """One recipe step per update; ``k`` and the gate's moving averages are its state (saved with checkpoints)."""

    def __init__(self, config, n_shells, state=None):
        base = config.sgd_learning_rate if config.sgd_lr_match is None else None
        if config.sgd_lr_match is not None and state is not None and state.get("lr0") is not None:
            base = float(state["lr0"])
        self.lr = LearningRateSchedule(base, config.sgd_lr_schedule)
        self.lr_match = config.sgd_lr_match
        self.k = None if state is None or state["k"] is None else float(state["k"])
        gate_state = None if state is None else {key[5:]: state[key] for key in state if key.startswith("gate_")}
        self.gate = (
            PCGate(config.sgd_pc_gate_floor, n_shells, gate_state or None)
            if config.sgd_pc_gate_floor is not None
            else None
        )
        self.direction = config.sgd_direction
        self.pc_min_shell = int(config.sgd_pc_min_shell)
        self.relative_floor = config.sgd_eig_floor_rel

    def step(self, stats, active, shells, iteration, floor, q):
        """The update ``(n_frequency, 1+q)`` and its diagnostics."""
        gradient = sum(jnp.asarray(s.residual_gradient) for s in stats)
        lhs = sum(jnp.asarray(s.lhs_tri) for s in stats)
        diagonal = metric_diagonal(lhs, q + 1)
        if self.lr.base is None:
            # Matched rate: the mean's step G / c_mean equals the shared-trace step M G / max tr H at this update.
            _, c_mean, _ = split_curvature_step(gradient, diagonal, active, 1.0, floor)
            trace_max = float(jnp.max(jnp.where(jnp.asarray(active, bool), jnp.sum(diagonal, axis=1), 0.0)))
            self.lr.base = self.lr_match * c_mean / max(trace_max, floor)
        lr = self.lr(iteration)
        split, c_mean, c_pc = split_curvature_step(gradient, diagonal, active, lr, floor)
        diagnostics = {"learning_rate_used": lr, "learning_rate_base": self.lr.base, "c_mean": c_mean, "c_pc": c_pc}
        if self.direction == "coupled":
            direction, info = damped_newton_direction(lhs, gradient, active, floor, self.relative_floor)
            if self.k is None:
                self.k = float(jnp.linalg.norm(split) / jnp.maximum(lr * jnp.linalg.norm(direction), 1e-30))
            step = jnp.float32(lr * self.k) * direction
            diagnostics.update(info, coupled_k=self.k)
        else:
            step = split
        if self.gate is not None:
            self.gate.update([s.residual_gradient for s in stats], shells)
            factors = self.gate.factors()
            step = jnp.asarray(step).at[:, 1:].multiply(factors[jnp.asarray(shells)][:, None].astype(step.dtype))
            top = int(np.max(np.where(np.asarray(active), np.asarray(shells), 0)))
            diagnostics["pc_gate_shell"] = [round(float(x), 4) for x in np.asarray(factors)[: top + 1]]
            diagnostics["pc_gate_raw_shell"] = [round(float(x), 4) for x in np.asarray(self.gate.raw())[: top + 1]]
        if self.pc_min_shell:
            keep = (jnp.asarray(shells) >= self.pc_min_shell)[:, None].astype(step.dtype)
            step = jnp.asarray(step).at[:, 1:].multiply(keep)
        diagnostics["step_l2"] = float(jnp.linalg.norm(step))
        return step.astype(gradient.dtype), diagnostics

    def save(self, path):
        record = {
            "k": np.float64(np.nan if self.k is None else self.k),
            "lr0": np.float64(np.nan if self.lr.base is None else self.lr.base),
        }
        if self.gate is not None and self.gate.sums is not None:
            record.update({f"gate_{key}": value for key, value in self.gate.state().items()})
        np.savez(path, **record)

    @staticmethod
    def load(path):
        with np.load(path) as saved:
            state = {key: saved[key] for key in saved.files}
        for key in ("k", "lr0"):
            if key in state and np.isnan(state[key]):
                state[key] = None
        return state


def describe(config):
    return json.dumps(
        {
            "direction": config.sgd_direction,
            "pc_gate_floor": config.sgd_pc_gate_floor,
            "learning_rate": config.sgd_learning_rate if config.sgd_lr_match is None else f"matched x{config.sgd_lr_match}",
            "lr_schedule": config.sgd_lr_schedule,
            "uniform_direction_prior": config.uniform_direction_prior,
            "shift_schedule": config.shift_schedule,
            "offset_sigma_hold_px": config.offset_sigma_hold_px,
        }
    )
