"""Opt-in finite checks for the pass-2 M-step operands and accumulators (P4-D).

A second compact control run crashed at RELION iteration 15 with a
non-finite ``wsum_norm_correction``. Its saved accumulators localise the damage
but not its source: ``Ft_ctf`` for half 2 holds 2.35 M ``+inf`` entries and
``Ft_y`` follows to NaN, while half 1 is clean apart from the radius <= 18 ball
that ``joinTwoHalvesAtLowResolution`` copies across from half 2. The sums cannot
say which row produced the first ``inf``, because a sum hides its terms.

This module adds that missing step. The per-bucket checks are off unless
``RELAX_EM_FINITE_CHECK=1`` (``EngineTuning.finite_check``, read once by the command; the callers run a check
only when it is not ``"off"``): the clean path reduces on the device and pulls
back one boolean per field, so it costs a reduction and a synchronisation
rather than a transfer, but it still serialises the pass it watches. An arm
with it set is a diagnostic arm and never a timing arm. The once-per-iteration
accumulator guard (:func:`check_half_accumulators`) is the exception: it is on
by default, see ``relax.reconstruction.accumulator_guard.half_accumulator_guard_mode``.

On the first non-finite value it reports the half, the bucket, the field, how
many entries are bad, the first bad flat index, and the operands at that index,
then raises unless ``RELAX_EM_FINITE_CHECK_WARN=1`` (``EngineTuning.finite_check == "warn"``; the checks take
``warn_only``) asks it to keep going and collect every occurrence.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "FiniteCheckError",
    "check_arrays",
    "check_bundle",
    "check_per_image",
    "check_posterior_bounds",
    "describe_context",
    "check_half_accumulators",
]

# ``reconstruction_probs`` are ``raw_weight / sum_weight`` where the numerator
# is one of the terms of the denominator, so every entry is at most one and a
# whole image's entries sum to at most one.  A value above that bound cannot be
# a rounding artefact; it means the denominator did not come from the weights
# it normalises.  One part in a thousand is far outside float32 rounding for a
# sum of ~600 positive terms and far inside the failure this hunts.
POSTERIOR_UPPER_BOUND = 1.0 + 1e-3


class FiniteCheckError(AssertionError):
    """A pass-2 operand or accumulator held a non-finite value."""


def _array_all_finite(array):
    import jax.numpy as jnp

    if jnp.iscomplexobj(array):
        return jnp.isfinite(array.real).all() & jnp.isfinite(array.imag).all()
    return jnp.isfinite(array).all()


@lru_cache(maxsize=None)
def _device_all_finite_program():
    import jax

    # One program per shape: eager, the check was up to five single-primitive
    # programs compiled again at every new reconstruction size.
    return jax.jit(_array_all_finite)


def _device_all_finite(array):
    return _device_all_finite_program()(array)


@lru_cache(maxsize=None)
def _device_each_finite_program():
    import jax
    import jax.numpy as jnp

    def each_finite(arrays):
        return jnp.stack([_array_all_finite(array) for array in arrays])

    # One program for all of an iteration's accumulators: a program per array
    # recompiled each of them at every new reconstruction size.
    return jax.jit(each_finite)


def _non_finite_names(arrays: dict) -> list:
    """The names in ``arrays`` whose value holds a non-finite entry, in order.

    The floating device arrays are reduced in one program and read back as one
    boolean vector; a value ``jnp.asarray`` cannot take goes through
    :func:`_all_finite`'s host route.
    """

    import jax.numpy as jnp

    names, device = [], []
    offenders = set()
    for name, value in arrays.items():
        if value is None:
            continue
        try:
            array = jnp.asarray(value)
        except Exception:  # pragma: no cover - host arrays and exotic dtypes
            if not _all_finite(value):
                offenders.add(name)
            continue
        if array.dtype.kind in "fgc":
            names.append(name)
            device.append(array)
    if device:
        finite = np.asarray(_device_each_finite_program()(tuple(device)))
        offenders.update(name for name, ok in zip(names, finite) if not ok)
    return [name for name in arrays if name in offenders]


def _all_finite(value) -> bool:
    """Whether every entry is finite, reduced where the array already lives.

    Only the resulting scalar crosses the device boundary, so the check costs a
    reduction and one synchronisation rather than a full transfer.
    """

    try:
        import jax.numpy as jnp

        array = jnp.asarray(value)
        if array.dtype.kind not in "fgc":
            return True
        return bool(_device_all_finite(array))
    except Exception:  # pragma: no cover - host arrays and exotic dtypes
        array = np.asarray(value)
        if array.dtype.kind not in "fgc":
            return True
        return bool(np.isfinite(array).all())


def describe_context(**fields) -> str:
    """Render the loop coordinates a report needs, skipping absent ones."""

    return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)


def _summarise(name, value, bad_index):
    """One field's shape, dtype, extremes and the value at ``bad_index``."""

    array = np.asarray(value)
    flat = array.reshape(-1)
    finite = np.isfinite(flat) if flat.dtype.kind != "c" else (
        np.isfinite(flat.real) & np.isfinite(flat.imag)
    )
    n_bad = int(flat.size - finite.sum())
    finite_values = flat[finite]
    extremes = ""
    if finite_values.size:
        magnitude = np.abs(finite_values)
        extremes = f" finite|max|={magnitude.max():.6g} finite|min|={magnitude.min():.6g}"
    at = ""
    if bad_index is not None and bad_index < flat.size:
        at = f" value_at_first_bad={flat[bad_index]!r}"
    return (
        f"{name}: shape={array.shape} dtype={array.dtype} nonfinite={n_bad}/{flat.size}"
        f"{extremes}{at}"
    )


def check_arrays(stage: str, arrays: dict, *, warn_only: bool, context: str = "", operands: dict | None = None):
    """Check ``arrays`` for non-finite entries; report the first offender (logged under ``warn_only``, else raised).

    ``arrays`` are the values under test; ``operands`` are additional arrays to
    describe at the offending index, so the report says not only which sum went
    non-finite but what it was built from. Only a field that fails the device
    reduction is brought to the host, and only to build the report.
    """

    # Fast path: reduce on whatever device holds the array and pull back one
    # boolean per field. The M-step sums are [images, rotations, pixels]; moving
    # them to the host every bucket would cost more than the pass being watched.
    suspect = [name for name, value in arrays.items() if value is not None and not _all_finite(value)]
    if not suspect:
        return None

    first_bad = None
    offender = None
    for name in suspect:
        flat = np.asarray(arrays[name]).reshape(-1)
        finite = (
            np.isfinite(flat.real) & np.isfinite(flat.imag)
            if flat.dtype.kind == "c"
            else np.isfinite(flat)
        )
        bad = int(np.flatnonzero(~finite)[0])
        if first_bad is None or bad < first_bad:
            first_bad, offender = bad, name

    if offender is None:
        return None

    lines = [f"non-finite value in pass-2 stage {stage!r}: {context}"]
    lines.append(f"  first offender: {offender} at flat index {first_bad}")
    for name, value in arrays.items():
        if value is not None:
            lines.append("  " + _summarise(name, value, first_bad))
    for name, value in (operands or {}).items():
        if value is not None:
            lines.append("  operand " + _summarise(name, value, None))
    message = "\n".join(lines)
    if warn_only:
        logger.error("%s", message)
        return message
    raise FiniteCheckError(message)


def check_per_image(stage: str, arrays: dict, *, warn_only: bool, image_ids=None, context: str = ""):
    """Check per-image statistics and name the offending image ids (logged under ``warn_only``, else raised).

    ``wsum_norm_correction`` is the statistic that failed in that control run, and
    it is per image, so the report is only useful if it says which particles.
    """

    ids = None if image_ids is None else np.asarray(image_ids).reshape(-1)
    reports = []
    for name, value in arrays.items():
        if value is None or _all_finite(value):
            continue
        flat = np.asarray(value).reshape(-1)
        if flat.dtype.kind not in "fgc":
            continue
        finite = (
            np.isfinite(flat.real) & np.isfinite(flat.imag)
            if flat.dtype.kind == "c"
            else np.isfinite(flat)
        )
        if bool(finite.all()):
            continue
        bad = np.flatnonzero(~finite)
        named = bad if ids is None or ids.size != flat.size else ids[bad]
        reports.append(
            f"  {name}: {bad.size}/{flat.size} non-finite, "
            f"first rows {bad[:8].tolist()}, image ids {np.asarray(named[:8]).tolist()}"
        )
    if not reports:
        return None
    message = "\n".join([f"non-finite per-image statistic in stage {stage!r}: {context}", *reports])
    if warn_only:
        logger.error("%s", message)
        return message
    raise FiniteCheckError(message)


def check_posterior_bounds(stage: str, probs, *, warn_only: bool, context: str = "", image_ids=None):
    """Assert the posterior bound that the M-step operands inherit (logged under ``warn_only``, else raised).

    ``probs`` is ``[images, ...]``.  Both the largest entry and each image's
    total must stay at or below one.  This fires on a denominator that is only
    mildly wrong, long before the product overflows float32, so a repeat that
    never reaches ``inf`` still shows whether the bound is being broken.
    """

    if probs is None:
        return None

    import jax.numpy as jnp

    array = jnp.asarray(probs)
    if array.dtype.kind not in "fg":
        return None
    rows = array.reshape(array.shape[0], -1)
    finite_rows = jnp.where(jnp.isfinite(rows), rows, jnp.float32(0.0))
    row_sums = jnp.sum(finite_rows, axis=1, dtype=jnp.float64)
    entry_max = float(jnp.max(finite_rows)) if rows.size else 0.0
    row_sum_max = float(jnp.max(row_sums)) if rows.size else 0.0

    if entry_max <= POSTERIOR_UPPER_BOUND and row_sum_max <= POSTERIOR_UPPER_BOUND:
        return None

    sums = np.asarray(row_sums)
    offenders = np.flatnonzero(sums > POSTERIOR_UPPER_BOUND)
    if offenders.size == 0:
        offenders = np.flatnonzero(np.asarray(jnp.max(finite_rows, axis=1)) > POSTERIOR_UPPER_BOUND)
    ids = None if image_ids is None else np.asarray(image_ids).reshape(-1)
    named = offenders if ids is None or ids.size != sums.size else ids[offenders]
    message = (
        f"posterior bound broken in stage {stage!r}: {context}\n"
        f"  entry_max={entry_max:.6g} image_sum_max={row_sum_max:.6g} "
        f"bound={POSTERIOR_UPPER_BOUND}\n"
        f"  offending rows {offenders[:8].tolist()} image ids "
        f"{np.asarray(named[:8]).tolist()} sums "
        f"{[float(x) for x in sums[offenders[:8]]]}"
    )
    if warn_only:
        logger.error("%s", message)
        return message
    raise FiniteCheckError(message)


def check_bundle(
    stage: str,
    arrays: dict,
    *,
    warn_only: bool,
    posterior=None,
    posterior_name: str = "posterior",
    dominated_by: tuple | None = None,
    context: str = "",
    image_ids=None,
    operands: dict | None = None,
):
    """Check several arrays and the posterior bound with one synchronisation.

    ``check_arrays`` plus ``check_posterior_bounds`` cost one
    device synchronisation each, and a bucket has half a dozen of them; at 200
    to 350 buckets a half that serialises the pass being watched and costs
    about four times the uninstrumented wall, which buys four times less
    exposure per GPU-hour on a defect measured at 1 in 671 half-iterations.

    Here every reduction is issued asynchronously and only the stacked scalars
    cross the boundary, so a clean bucket pays one synchronisation for the
    whole set. The expensive per-array host analysis runs only when a scalar
    says something is wrong, and then it is free to synchronise again. A report is logged under ``warn_only``,
    else raised.
    """

    import jax.numpy as jnp

    names = [name for name, value in arrays.items() if value is not None]
    scalars = []
    for name in names:
        array = jnp.asarray(arrays[name])
        if array.dtype.kind not in "fgc":
            scalars.extend([jnp.float64(0.0), jnp.float64(1.0)])
            continue
        magnitude = jnp.abs(array)
        finite = jnp.isfinite(magnitude)
        scalars.append(jnp.max(jnp.where(finite, magnitude, jnp.zeros((), magnitude.dtype))).astype(jnp.float64))
        scalars.append(jnp.all(finite).astype(jnp.float64))

    has_posterior = posterior is not None
    if has_posterior:
        rows = jnp.asarray(posterior)
        rows = rows.reshape(rows.shape[0], -1)
        finite_rows = jnp.where(jnp.isfinite(rows), rows, jnp.zeros((), rows.dtype))
        scalars.append(jnp.max(finite_rows).astype(jnp.float64))
        scalars.append(jnp.max(jnp.sum(finite_rows, axis=1, dtype=jnp.float64)))

    if not scalars:
        return None
    host = np.asarray(jnp.stack(scalars))  # the single synchronisation

    offenders = []
    for index, name in enumerate(names):
        maximum = float(host[2 * index])
        all_finite = bool(host[2 * index + 1] > 0.5)
        if not all_finite:
            offenders.append(name)

    entry_max = row_sum_max = None
    if has_posterior:
        entry_max = float(host[-2])
        row_sum_max = float(host[-1])

    messages = []
    # Exact reduction invariants, free because the maxima are already here.
    # ``sum_t probs <= 1`` and ``probs >= 0`` make each M-step sum bounded by
    # the largest operand it reduces, whatever the magnitudes. This catches an
    # element-level fault inside the reduction that never appears in any
    # materialised input array, which a finiteness test cannot see until the
    # accumulated value actually overflows.
    maxima = {name: float(host[2 * index]) for index, name in enumerate(names)}
    for reduced, operand, slack in dominated_by or ():
        factors = (operand,) if isinstance(operand, str) else tuple(operand)
        if reduced not in maxima or any(f not in maxima for f in factors):
            continue
        product = 1.0
        for factor in factors:
            product *= maxima[factor]
        limit = product * (1.0 + slack)
        if maxima[reduced] > limit:
            shown = " * ".join(f"max |{f}|" for f in factors)
            broken = (
                f"operand bound broken in stage {stage!r}: {context}\n"
                f"  max |{reduced}| = {maxima[reduced]:.6g} exceeds "
                f"(1+{slack:g}) * {shown} = {limit:.6g}\n"
                f"  {reduced} is built elementwise or by a posterior-weighted "
                f"sum from those, with non-negative weights summing to at most "
                f"one, so this cannot be rounding"
            )
            if warn_only:
                logger.error("%s", broken)
                messages.append(broken)
            else:
                raise FiniteCheckError(broken)
    if offenders:
        messages.append(
            check_arrays(
                stage,
                {name: arrays[name] for name in names},
                warn_only=warn_only,
                context=context,
                operands=operands,
            )
        )
    if has_posterior and (
        entry_max > POSTERIOR_UPPER_BOUND or row_sum_max > POSTERIOR_UPPER_BOUND
    ):
        messages.append(
            check_posterior_bounds(
                posterior_name,
                posterior,
                warn_only=warn_only,
                context=context,
                image_ids=image_ids,
            )
        )
    return [m for m in messages if m] or None


def check_half_accumulators(accumulators: dict, *, mode: str, context: str = ""):
    """Fail where the damage is, not four stages later.

    That compact control run wrote a non-finite BPref accumulator, then
    reconstructed and saved two all-NaN half maps, updated tau2 from an FSC of
    [1, 0, 0, ...], printed a healthy noise spectrum, and only then raised on a
    non-finite ``wsum_norm_correction``. Nothing in that sequence says which
    accumulator went wrong or in which half.

    Unlike the per-bucket checks this costs one reduction per accumulator and
    one synchronisation per iteration, against a pass that takes tens of
    seconds, so it is affordable in production and runs by default in raise
    ``mode`` (``accumulator_guard.half_accumulator_guard_mode``; ``RELAX_EM_BPREF_FINITE_GUARD=0``
    opts out, ``warn`` logs and continues). Turning it off restores the
    earlier behaviour of failing four stages later on ``wsum_norm_correction``.
    """

    if mode == "off":
        return None
    offenders = _non_finite_names(accumulators)
    if not offenders:
        return None
    lines = [f"non-finite BPref accumulator before reconstruction: {context}"]
    for name in offenders:
        lines.append("  " + _summarise(name, accumulators[name], None))
    message = "\n".join(lines)
    if mode == "warn":
        logger.error("%s", message)
        return message
    raise FiniteCheckError(message)
