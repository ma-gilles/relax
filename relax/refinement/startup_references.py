"""The start-up references of a refinement: the maps relion_refine reads with --ref, in the loop's layout.

References are RELION-convention maps; ``load_relion_volume`` puts them in the internal frame
(``relax.helpers.map_io``), and ``get_dft3`` gives the centered Fourier volume. NEVER use raw
``mrcfile.open`` + ``np.fft.fftn(np.fft.ifftshift(...))`` here: that produces a Fourier volume with the
right values but at WRONG array indices (DC at corner instead of center), so ``slice_volume`` reads Nyquist
as if it were DC and projections are off by ~2400x in amplitude at low frequencies.

With a start-up low-pass (``--apply-initial-lowpass``), each map is filtered as RELION's
``initialLowPassFilterReferences`` (ml_optimiser.cpp:3556) filters ``mymodel.Iref`` at start-up, gated only on
``ini_high > 0``. The command chooses the precision (``real_dtype``/``complex_dtype``) and the maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu

from relax.relion import relion_metadata
from relax.relion.geometry import REFERENCE_FILTER_EDGE_SHELLS


@dataclass(frozen=True)
class StartupReferences:
    """The references a run starts from.

    ``fourier``: the loop's start-up volume, ``(V,)`` for K=1, ``(K, V)`` for Class3D, ``(2, V)`` (one per half)
    from a frozen boundary. ``prior_source``: the Fourier volume the start-up tau2 is bootstrapped from (the
    reference, class 1, or the halves' mean). ``real_for_projector``: the real-space maps handed to the first
    projector (float64) when that handoff is on, else None. ``reference_real`` (K=1) and
    ``class_references_real`` (Class3D, a list): the float64 real maps of RELION's start-up data-vs-prior.
    ``model_pixel_size``: the K=1 map header's pixel size (None otherwise).
    """

    fourier: np.ndarray
    prior_source: np.ndarray
    real_for_projector: np.ndarray | None = None
    reference_real: np.ndarray | None = None
    class_references_real: list | None = None
    model_pixel_size: float | None = None


def _initial_lowpass_real(volume_real, volume_shape, voxel_size, ini_high):
    from relax.relion.reference_initialization import initial_low_pass_filter_references

    filtered = initial_low_pass_filter_references(
        np.asarray(volume_real, dtype=np.float64)[None, ...],
        ori_size=int(volume_shape[0]),
        pixel_size=float(voxel_size),
        ini_high_ang=float(ini_high),
        filter_edgewidth=float(REFERENCE_FILTER_EDGE_SHELLS),
    )[0]
    return np.asarray(filtered, dtype=np.float64)


def frozen_boundary_references(frozen_boundary, *, complex_dtype) -> StartupReferences:
    """The per-half Fourier references a frozen boundary owns; the start-up tau2 uses their mean."""
    halves = np.stack(frozen_boundary.means, axis=0)
    merged = np.mean(halves.astype(np.complex128), axis=0).astype(complex_dtype)
    return StartupReferences(fourier=halves, prior_source=merged)


def load_k1_reference(
    path,
    *,
    volume_shape,
    ini_high: float | None,
    real_for_projector: bool,
    real_dtype,
    complex_dtype,
    log: logging.Logger,
) -> StartupReferences:
    """Load the K=1 reference map ``path``, low-pass filtered at ``ini_high`` in the map's own pixel size."""
    from recovar.utils.helpers import load_relion_volume

    init_vol_real = load_relion_volume(path).astype(real_dtype)
    model_pixel_size = relion_metadata._read_relion_mrc_model_pixel_size(path)
    if not np.isfinite(model_pixel_size) or model_pixel_size <= 0.0:
        raise SystemExit(f"Initial RELION reference has invalid voxel size {model_pixel_size}: {path}")
    assert init_vol_real.shape == volume_shape, f"Volume shape mismatch: {init_vol_real.shape} vs {volume_shape}"
    reference_for_projector = None
    if ini_high is not None:
        # RELION filters ``mymodel.Iref`` in model coordinates.  The
        # particle STAR optics pixel size can be a rounded serialization
        # (for example 1.416667 versus the MRC header ratio
        # 544.0 / 384 = 1.4166666666666667 A/px),
        # which is enough to flip marginal firstiter-CC winners.
        filtered_real = _initial_lowpass_real(init_vol_real, volume_shape, model_pixel_size, ini_high)
        if real_for_projector:
            reference_for_projector = filtered_real
        reference_real = filtered_real
        init_vol_real = filtered_real.astype(real_dtype, copy=False)
        log.info(
            "Applied RELION initialLowPassFilterReferences to init reference: ini_high=%.2f A, fmask_edge=%d shells",
            ini_high, REFERENCE_FILTER_EDGE_SHELLS,
        )
    else:
        reference_real = np.asarray(init_vol_real, dtype=np.float64)
        if real_for_projector:
            reference_for_projector = reference_real
    init_vol_ft = np.array(ftu.get_dft3(jnp.asarray(init_vol_real))).astype(complex_dtype).reshape(-1)
    log.info(
        "Initial volume loaded from %s: shape=%s model_pixel_size=%.9g A/px",
        path,
        init_vol_real.shape,
        model_pixel_size,
    )
    return StartupReferences(
        fourier=init_vol_ft,
        prior_source=init_vol_ft,
        real_for_projector=reference_for_projector,
        reference_real=reference_real,
        model_pixel_size=model_pixel_size,
    )


def load_class_references(
    paths,
    *,
    volume_shape,
    voxel_size,
    ini_high: float | None,
    real_for_projector: bool,
    real_dtype,
    complex_dtype,
    log: logging.Logger,
) -> StartupReferences:
    """Load one map per class, low-pass filtered at ``ini_high`` in the particles' pixel size ``voxel_size``."""
    from recovar.utils.helpers import load_relion_volume

    per_class_ft = []
    per_class_real_for_projector = []
    class_references_real = []
    for k, p in enumerate(paths):
        vol_real = np.asarray(load_relion_volume(p)).astype(real_dtype)
        assert vol_real.shape == volume_shape, (
            f"Class {k + 1} volume shape mismatch at {p}: {vol_real.shape} vs {volume_shape}"
        )
        if ini_high is not None:
            filtered_real = _initial_lowpass_real(vol_real, volume_shape, voxel_size, ini_high)
            if real_for_projector:
                per_class_real_for_projector.append(filtered_real)
            class_references_real.append(np.asarray(filtered_real, dtype=np.float64))
            vol_real = filtered_real.astype(real_dtype, copy=False)
        else:
            class_references_real.append(np.asarray(vol_real, dtype=np.float64))
            if real_for_projector:
                per_class_real_for_projector.append(np.asarray(vol_real, dtype=np.float64))
        vol_ft = np.array(ftu.get_dft3(jnp.asarray(vol_real))).astype(complex_dtype).reshape(-1)
        per_class_ft.append(vol_ft)
        log.info("Class %d initial volume loaded from %s", k + 1, p)
    if ini_high is not None:
        log.info(
            "Applied RELION initialLowPassFilterReferences to %d init references: ini_high=%.2f A, fmask_edge=%d shells",
            len(paths), ini_high, REFERENCE_FILTER_EDGE_SHELLS,
        )
    # Stack to (K, V); refine_single_volume._normalize_initial_means handles the per-half broadcast.
    return StartupReferences(
        fourier=np.stack(per_class_ft, axis=0),
        # The K-class start-up prior uses class 1 as the representative (a single spectrum).
        prior_source=per_class_ft[0],
        real_for_projector=np.stack(per_class_real_for_projector, axis=0) if real_for_projector else None,
        class_references_real=class_references_real,
    )
