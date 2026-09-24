"""Diagnostic: run the compact engine alongside a resident pass and save both outputs.

``RELAX_SPARSE_PASS2_RESIDENT_SHADOW_DIR=<dir>`` makes every in-scope resident pass 2
also run ``compute_pass2_stats_sparse_bucketed`` on the same arguments and write the
per-image outputs of both engines to ``<dir>/pass2_shadow_<n>.npz``. The resident
result is returned unchanged. Used to localise where the two engines' posteriors
first differ on a real iteration; it doubles the pass cost and is never on by default.
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SHADOW_DIR_ENV = "RELAX_SPARSE_PASS2_RESIDENT_SHADOW_DIR"
_counter = itertools.count()


def shadow_dir() -> Path | None:
    value = os.environ.get(SHADOW_DIR_ENV, "").strip()
    return Path(value) if value else None


def _host(value):
    return None if value is None else np.asarray(value)


def _fields(prefix, out):
    arrays = {
        f"{prefix}_hard_assignment": _host(out.hard_assignment),
        f"{prefix}_best_rotation_indices": _host(out.best_rotation_indices),
        f"{prefix}_best_translations": _host(out.best_translations),
        f"{prefix}_Ft_y": _host(out.Ft_y),
    }
    stats = out.relion_stats
    if stats is not None:
        for name in ("log_evidence_per_image", "best_log_score_per_image", "max_posterior_per_image"):
            arrays[f"{prefix}_{name}"] = _host(getattr(stats, name))
    noise = out.noise_stats
    if noise is not None:
        for name in ("wsum_sigma2_noise", "wsum_norm_correction", "wsum_img_power", "sumw"):
            arrays[f"{prefix}_{name}"] = _host(getattr(noise, name, None))
    return {k: v for k, v in arrays.items() if v is not None}


def run_resident_with_compact_shadow(resident_impl, compact_impl, open_texture, *args, **kwargs):
    """Run ``resident_impl``, then ``compact_impl`` on the same inputs, and save both."""

    resident = resident_impl(*args, **kwargs)
    compact_kwargs = dict(kwargs)
    texture = open_texture()
    if texture is not None:
        compact_kwargs["relion_projector_half"] = None
        compact_kwargs["relion_projector_texture"] = texture
    try:
        compact = compact_impl(*args, **compact_kwargs)
    finally:
        if texture is not None:
            texture.close()
    out_dir = shadow_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pass2_shadow_{next(_counter):03d}.npz"
    np.savez(path, **_fields("resident", resident), **_fields("compact", compact))
    logger.info("Resident pass-2 shadow: compact engine outputs saved beside the resident ones in %s", path)
    return resident
