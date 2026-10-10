"""The VDAM noise-boundary capture and the non-finite noise-sum dump, written by the command's observer
(``relax.diagnostics.vdam_observers.VdamDumpObserver``)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from relax.vdam.state import InitialModelState


def dump_noise_failure_meta(dump_root: str, state: InitialModelState, meta: dict, summaries: Sequence[str]) -> str:
    """Write the E-step meta's noise sums of an iteration whose sums are not finite; returns the file."""
    path = Path(dump_root)
    path.mkdir(parents=True, exist_ok=True)
    dump_path = path / f"noise_failure_iter_{getattr(state, 'iter', 'unknown')}.npz"
    payload: dict[str, np.ndarray] = {
        "state_iter": np.asarray([getattr(state, "iter", -1)], dtype=np.int64),
        "state_subset_size": np.asarray([getattr(state, "subset_size", -1)], dtype=np.int64),
        "state_ori_size": np.asarray([getattr(state, "box_size", -1)], dtype=np.int64),
        "summaries": np.asarray(list(summaries), dtype=str),
    }
    for key, value in sorted(meta.items()):
        if not any(token in key for token in ("noise", "wsum")):
            continue
        try:
            payload[key] = np.asarray(value)
        except Exception:
            payload[f"{key}_repr"] = np.asarray([repr(value)], dtype=str)
    np.savez_compressed(dump_path, **payload)
    return str(dump_path)


def dump_noise_update_boundary(
    dump_root: str,
    state: InitialModelState,
    updated_state: InitialModelState,
    *,
    wsum_sigma2_noise: np.ndarray,
    wsum_img_power: np.ndarray,
    noise_sumw: float,
    wsum_noise_a2: np.ndarray | None,
    wsum_noise_xa: np.ndarray | None,
) -> str:
    """Write one iteration's VDAM noise sufficient statistics and spectra; returns the file."""

    from relax.relion.metadata import _relion_half_plane_shell_counts

    dump_dir = Path(dump_root)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"initialmodel_noise_update_it{int(state.iter):03d}.npz"
    if dump_path.exists():
        raise ValueError(f"refusing to overwrite {dump_path}")
    n4 = float(int(state.box_size) ** 4)
    old_noise = np.asarray(state.sigma2_noise, dtype=np.float64)[0] * n4
    new_noise = np.asarray(updated_state.sigma2_noise, dtype=np.float64)[0] * n4
    residual = np.asarray(wsum_sigma2_noise, dtype=np.float64)
    image_power = np.asarray(wsum_img_power, dtype=np.float64)
    payload = {
        "schema": np.asarray("recovar.initialmodel.noise_update_boundary.v1"),
        "iteration": np.asarray([int(state.iter)], dtype=np.int32),
        "current_size": np.asarray([int(state.current_size)], dtype=np.int32),
        "image_shape": np.asarray([int(state.box_size), int(state.box_size)], dtype=np.int32),
        "relion_half_plane_shell_counts": _relion_half_plane_shell_counts((int(state.box_size), int(state.box_size))),
        "half0_wsum_sigma2_noise": residual,
        "half0_wsum_img_power": image_power,
        "half0_wsum_total": residual + image_power,
        "half0_sumw": np.asarray([float(noise_sumw)], dtype=np.float64),
        "half0_previous_sigma2_noise": old_noise,
        "half0_sigma2_noise": new_noise,
    }
    if wsum_noise_a2 is not None or wsum_noise_xa is not None:
        if wsum_noise_a2 is None or wsum_noise_xa is None:
            raise ValueError("noise split diagnostics require both wsum_noise_a2 and wsum_noise_xa")
        noise_a2 = np.asarray(wsum_noise_a2, dtype=np.float64)
        noise_xa = np.asarray(wsum_noise_xa, dtype=np.float64)
        if noise_a2.shape != residual.shape or noise_xa.shape != residual.shape:
            raise ValueError("noise split diagnostics must match the noise shell topology")
        payload["half0_wsum_noise_a2"] = noise_a2
        payload["half0_wsum_noise_xa"] = noise_xa
    np.savez_compressed(dump_path, **payload)
    return str(dump_path)
