"""RELION's resolution and image-size criteria: pure functions of the FSC and data-vs-prior curves.

``resolution_from_data_vs_prior`` is RELION's current-resolution shell (with ``RELION_MINRES_MAP``, its
``--minres_map`` default); ``fsc_to_relion_ssnr`` turns an FSC into the data-vs-prior ratio;
``compute_relion_incr_size_from_fsc``, ``update_relion_growth_state_from_fsc`` and
``compute_current_size_relion`` are the image-size growth rule. The definitions moved verbatim from
``relax.reconstruction.regularization``; none reads a map or the backprojector.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from recovar import jax_config

# RELION's --minres_map default (relion/src/ml_optimiser.cpp:1298): the
# smallest shell updateCurrentResolution may report, and the lowest shells
# that MAP reconstruction leaves without the Wiener prior term.
RELION_MINRES_MAP = 5


def resolution_from_data_vs_prior(
    data_vs_prior,
    *,
    box_size=None,
    allow_high_res_recovery=False,
    recovery_margin_shells=3,
    minres_map=RELION_MINRES_MAP,
):
    """RELION ``updateCurrentResolution`` shell for one data_vs_prior curve.

    Mirrors relion/src/ml_optimiser.cpp:6776-6819. Scans shells
    ``1 <= ires < ori_size // 2`` (skipping DC) and returns the last shell
    before ``data_vs_prior < 1``; a curve that never drops returns
    ``ori_size // 2 - 1``, one shell below Nyquist, as RELION does.

    When ``allow_high_res_recovery`` is enabled, mimic RELION's check for
    split-half auto-refine: scanning down from ``ori_size // 2 - 1``, if the
    curve rises above 1.0 again more than ``recovery_margin_shells`` shells
    past the first crossing, keep the later shell. This handles the
    phase-randomization / tight-mask artefact check in RELION.

    The result is never below ``minres_map`` (``maxres = XMIPP_MAX(maxres,
    minres_map)``, ml_optimiser.cpp:6819). RELION floors the class maximum;
    flooring each class first gives the same maximum.

    Parameters
    ----------
    data_vs_prior : array-like, shape (n_shells,)
        Per-shell data_vs_prior ratio from :func:`compute_data_vs_prior`.
    box_size : int, optional
        Box size. Defaults to ``2 * (n_shells - 1)``, RELION's
        ``ori_size / 2 + 1`` shell layout.
    allow_high_res_recovery : bool, optional
        Enable RELION's high-resolution recheck.
    recovery_margin_shells : int, optional
        Minimum number of shells by which the recovered high-resolution shell
        must exceed the first crossing.
    minres_map : int, optional
        RELION ``--minres_map``.

    Returns
    -------
    int
        Shell index of the resolution limit.
    """
    dvp = np.asarray(data_vs_prior)
    limit = len(dvp) - 1 if box_size is None else min(int(box_size) // 2, len(dvp))
    ires = 1
    while ires < limit and not dvp[ires] < 1.0:
        ires += 1
    maxres = ires - 1

    if allow_high_res_recovery:
        recovered = maxres
        for ires2 in range(limit - 1, maxres - 1, -1):
            if dvp[ires2] > 1.0:
                recovered = ires2
                break
        if recovered > maxres + int(recovery_margin_shells):
            maxres = recovered

    return max(maxres, int(minres_map))


def fsc_to_relion_ssnr(fsc, tau2_fudge=1.0, is_whole_instead_of_half=False):
    """Convert an FSC curve to RELION's data-vs-prior / SSNR curve.

    In gold-standard auto-refine, RELION updates ``data_vs_prior`` from the
    half-map FSC by converting each shell's FSC into an SSNR value. The shell
    where this curve drops below ``1`` is the same shell where the FSC drops
    below ``0.5``.
    """
    fsc = jnp.asarray(fsc)
    epsilon = jax_config.FSC_ZERO_THRESHOLD
    myfsc = jnp.maximum(fsc, epsilon)
    if is_whole_instead_of_half:
        myfsc = jnp.sqrt(2.0 * myfsc / (myfsc + 1.0))
    myfsc = jnp.minimum(myfsc, 1.0 - epsilon)
    return tau2_fudge * myfsc / (1.0 - myfsc)


def first_shell_below_threshold(values, threshold):
    """Return the first shell index below ``threshold``.

    RELION's shell scans start at shell 1 (skipping DC). If no shell drops
    below the threshold, return the last available shell.
    """
    arr = np.asarray(values)
    for i in range(1, len(arr)):
        if arr[i] < threshold:
            return i
    return len(arr) - 1


def compute_relion_incr_size_from_fsc(fsc, default=10):
    """RELION auto-refine shell-growth heuristic from the current FSC curve.

    RELION enlarges ``incr_size`` to at least ``fsc0143 - fsc05 + 5`` after the
    half-map comparison, where ``fsc05`` and ``fsc0143`` are the first shells
    where the FSC drops below 0.5 and 0.143, respectively.
    """
    fsc05 = first_shell_below_threshold(fsc, 0.5)
    fsc0143 = first_shell_below_threshold(fsc, 0.143)
    return max(int(default), int(fsc0143 - fsc05 + 5))


def update_relion_growth_state_from_fsc(
    fsc,
    current_size,
    *,
    incr_size=10,
    has_high_fsc_at_limit=False,
):
    """Update RELION's sticky current-size growth state from the FSC curve.

    RELION keeps ``incr_size`` as a non-decreasing value across iterations and
    only flips ``has_high_fsc_at_limit`` from false to true once. This helper
    mirrors the MPI auto-refine update in ``ml_optimiser_mpi.cpp``.
    """
    fsc = np.asarray(fsc)
    next_incr_size = compute_relion_incr_size_from_fsc(fsc, default=int(incr_size))

    if len(fsc) == 0:
        return next_incr_size, bool(has_high_fsc_at_limit)

    limit_shell = min(max(int(current_size) // 2 - 1, 0), len(fsc) - 1)
    high_fsc_now = bool(float(fsc[limit_shell]) > 0.2)
    return next_incr_size, bool(has_high_fsc_at_limit or high_fsc_now)


def compute_current_size_relion(resolution_shell, box_size, ave_Pmax=0.0, has_high_fsc_at_limit=False, incr_size=10):
    """Compute the next current_size using RELION's growth logic.

    RELION grows current_size beyond the current resolution limit.  If
    the average maximum posterior probability (``ave_Pmax``) exceeds 0.1
    AND the FSC is still high at the resolution limit, the jump is 25%
    of ``ori_size / 2`` (aggressive growth).  Otherwise the jump is
    ``incr_size`` shells (conservative).

    The result is clamped to ``ori_size``.

    Parameters
    ----------
    resolution_shell : int
        Current resolution shell index (e.g. from
        :func:`resolution_from_data_vs_prior` or FSC-based estimate).
    box_size : int
        Original image size in pixels (diameter, e.g. 128).
    ave_Pmax : float
        Average of the per-image maximum posterior probability.
        Typical range 0-1; early iterations have low values.
    has_high_fsc_at_limit : bool
        True if the FSC is still significantly above 0 at the current
        resolution limit (indicating the data supports higher resolution).
    incr_size : int
        Default shell increment when conditions for aggressive growth
        are not met.

    Returns
    -------
    int
        New current_size in pixels (diameter).
    """
    maxres = resolution_shell
    if ave_Pmax > 0.1 and has_high_fsc_at_limit:
        maxres += round(0.25 * box_size / 2)
    else:
        maxres += incr_size
    return min(2 * maxres, box_size)
