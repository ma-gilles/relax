"""Shared source-precision RELION CTF operands for coarse, local and sparse EM.

The source STAR and native CTF cache retain their original per-process lifetime.
Host callers can gather planned pixels before stacking; the device accessor
places the full operand once. Native RFLOAT values, centered half-spectrum
coordinates, signs and caller-selected cast boundaries are preserved.
"""

from __future__ import annotations

import collections
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from recovar.data_io.starfile import star_column

from relax.helpers.batch_fetch import original_image_indices
from relax.relion.tomo_input import fftw_half_freq_sq, relion_tomo_damping

_RELION_EXACT_CTF_SOURCE_CACHE: dict[tuple[str, tuple[int, int]], dict] = {}

# Assembled-operand memo. The per-image rows below are already cached, but the
# stack that turns them into one operand is rebuilt on every call: pass 2 asks
# for a whole half (4966 rows of 33024 float64 = 1.31 GB of host memcpy, 250 ms
# per half per iteration), and the coarse pass asks for one image batch of
# planned columns. Both requests repeat with identical inputs every iteration,
# so the assembled array is a pure function of the key and the memo returns the
# very same object: bit-for-bit by construction, not by re-derivation.
_EXACT_CTF_RESULT_CACHE: "collections.OrderedDict[tuple, tuple]" = collections.OrderedDict()
_EXACT_CTF_RESULT_BYTES = 0
_EXACT_CTF_CACHE_GB_ENV = "RELAX_RELION_EXACT_CTF_CACHE_GB"


def _exact_ctf_result_cache_budget_bytes() -> int:
    """Host bytes the assembled-operand memo may hold; 0 disables it.

    The default is 4 GB since the P4-I merge, which holds the two 1.31 GB
    whole-half operands of the 256x256 fixture and costs about 2.5 GB of host
    memory for 2.0 s of a steady hp3 iteration. Set the variable to 0 to
    restore the unmemoized assembly, which stays the oracle.

    The memo trades host memory for the repeated stack: a whole-half operand is
    1.31 GB at 256x256, so a budget below that disables the pass-2 entry while
    still holding the much smaller per-batch coarse entries.
    """

    token = os.environ.get(_EXACT_CTF_CACHE_GB_ENV, "4").strip()
    try:
        budget = float(token)
    except ValueError as exc:
        raise ValueError(f"{_EXACT_CTF_CACHE_GB_ENV} must be a number of gigabytes, got {token!r}") from exc
    if budget < 0:
        raise ValueError(f"{_EXACT_CTF_CACHE_GB_ENV} must not be negative, got {token!r}")
    return int(budget * (1024 ** 3))


def _exact_ctf_result_key(source_path, image_shape, original_indices, pixel_indices):
    """A content key for the assembled operand, with the arrays kept for verification."""

    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(original_indices).view(np.uint8))
    if pixel_indices is None:
        pixel_tag = b"none"
    else:
        pixel_tag = np.ascontiguousarray(pixel_indices).view(np.uint8).tobytes()
        digest.update(b"|pixels|")
        digest.update(pixel_tag)
    return (
        str(source_path),
        tuple(int(size) for size in image_shape),
        int(original_indices.size),
        None if pixel_indices is None else (str(pixel_indices.dtype), int(pixel_indices.size)),
        digest.digest(),
    )


def _exact_ctf_result_lookup(key, original_indices, pixel_indices):
    """Return the memoized operand for an exactly equal request, else None."""

    entry = _EXACT_CTF_RESULT_CACHE.get(key)
    if entry is None:
        return None
    cached_indices, cached_pixels, result = entry
    # A digest match is not trusted on its own: compare the actual selectors, so
    # a collision can never serve one image set's CTFs for another's.
    if not np.array_equal(cached_indices, original_indices):
        return None
    if (cached_pixels is None) != (pixel_indices is None):
        return None
    if cached_pixels is not None and not np.array_equal(cached_pixels, pixel_indices):
        return None
    _EXACT_CTF_RESULT_CACHE.move_to_end(key)
    return result


def _exact_ctf_result_store(key, original_indices, pixel_indices, result):
    """Memoize one assembled operand, evicting least-recently-used entries."""

    global _EXACT_CTF_RESULT_BYTES

    budget = _exact_ctf_result_cache_budget_bytes()
    if budget <= 0 or result.nbytes > budget:
        return result
    # The memo hands the same object to every caller, so make it read-only: no
    # caller mutates it today (each wraps it in np.asarray and uploads), and a
    # future one must not silently corrupt a shared operand.
    result.setflags(write=False)
    _EXACT_CTF_RESULT_CACHE[key] = (
        np.array(original_indices, copy=True),
        None if pixel_indices is None else np.array(pixel_indices, copy=True),
        result,
    )
    _EXACT_CTF_RESULT_BYTES += result.nbytes
    while _EXACT_CTF_RESULT_BYTES > budget and len(_EXACT_CTF_RESULT_CACHE) > 1:
        _, evicted = _EXACT_CTF_RESULT_CACHE.popitem(last=False)
        _EXACT_CTF_RESULT_BYTES -= evicted[2].nbytes
    return result


def clear_exact_ctf_result_cache() -> None:
    """Drop the assembled-operand memo (tests and long-lived processes)."""

    global _EXACT_CTF_RESULT_BYTES

    _EXACT_CTF_RESULT_CACHE.clear()
    _EXACT_CTF_RESULT_BYTES = 0


def _relion_exact_ctf_source_star(experiment_dataset) -> Path:
    """Resolve the immutable source STAR for exact RELION CTF evaluation."""

    source_star = os.environ.get("RELAX_K1_RELION_EXACT_CTF_STAR", "").strip()
    if not source_star:
        dataset_source = getattr(experiment_dataset, "particles_file", None)
        if dataset_source and Path(dataset_source).suffix.lower() == ".star":
            source_star = str(dataset_source)
    if not source_star:
        raise ValueError(
            "exact RELION operands require a STAR-backed dataset or "
            "RELAX_K1_RELION_EXACT_CTF_STAR"
        )
    return Path(source_star).expanduser().resolve()


def _relion_ctf_threads() -> int:
    """Worker threads for the host CTF evaluation: the CPUs this process may use."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux
        return max(1, os.cpu_count() or 1)


def _relion_ctf_column(cache, name: str, default: float | None, *, optics_only: bool = False) -> np.ndarray:
    """One CTF parameter for every particle, resolved as ``CTF::readValue`` resolves it."""

    columns = cache.setdefault("columns", {})
    if name in columns:
        return columns[name]
    particles = cache["particles"]
    groups = np.asarray(star_column(particles, "rlnOpticsGroup", required=True), dtype=np.int64)
    values = None if optics_only else star_column(particles, name)
    if values is not None:
        column = np.asarray(values, dtype=np.float64)
    else:
        by_group = {}
        for group, optics in cache["optics"].items():
            found = [float(optics[key]) for key in (name, f"_{name}") if key in optics]
            if found:
                by_group[group] = found[0]
            elif default is None:
                raise KeyError(f"RELION optics group {group} has no {name}")
            else:
                by_group[group] = float(default)
        column = np.asarray([by_group[int(group)] for group in groups], dtype=np.float64)
    columns[name] = column
    return column


def _premultiplied_particles(cache) -> np.ndarray:
    """Per particle: its optics group stores CTF-premultiplied images (``rlnCtfDataAreCtfPremultiplied``)."""

    column = cache.get("premultiplied")
    if column is None:
        values = _relion_ctf_column(cache, "rlnCtfDataAreCtfPremultiplied", 0.0, optics_only=True)
        column = np.asarray(values != 0.0, dtype=bool)
        cache["premultiplied"] = column
    return column


def premultiplied_ctf_rows(experiment_dataset, image_indices, image_shape) -> np.ndarray | None:
    """Which of these images are CTF-premultiplied, or ``None`` when no optics group is.

    RELION reads ``rlnCtfDataAreCtfPremultiplied`` per optics group
    (``ObservationModel::getCtfPremultiplied``). For such an image the exact CTF
    rows of this module already hold RELION's ``Fctf = CTF^2``
    (:func:`_evaluate_exact_ctf_rows`); the backprojection operands then differ
    as well (:func:`relax.sparse_pass2.sparse_pass2_bucket_io.premultiplied_bpref_weights`).
    ``None`` keeps every ordinary dataset on its unchanged programs. A dataset
    without a source STAR has no optics table and so no premultiplied images.
    """

    try:
        _, cache = _exact_ctf_source_cache(experiment_dataset, image_shape)
    except ValueError:
        return None
    flags = _premultiplied_particles(cache)
    if not flags.any():
        return None
    original_indices = np.asarray(
        original_image_indices(experiment_dataset, np.asarray(image_indices, dtype=np.int64)), dtype=np.int64
    )
    return flags[original_indices]


def dataset_has_premultiplied_ctf(experiment_dataset, image_shape) -> bool:
    """Whether any optics group of the dataset's source STAR stores CTF-premultiplied images."""

    try:
        _, cache = _exact_ctf_source_cache(experiment_dataset, image_shape)
    except ValueError:
        return False
    return bool(_premultiplied_particles(cache).any())


def _fftw_shell_labels(size: int, window: int, *, centered_rows: bool) -> np.ndarray:
    """RELION's ``Mresol_fine`` of a ``window``-pixel current size on a ``size``-pixel half grid.

    Per half-grid pixel ``(ip, jp)``: ``ROUND(|(ip, jp)|)`` for the pixels of the
    ``window``-sized FFTW transform (rows ``ip`` in ``[window//2 + 1 - window, window//2]``,
    ``jp <= window//2``) with ``ires < window//2 + 1``, and -1 for the rest and for the
    redundant half of the ``jp = 0`` column (``ip < 0``; ml_optimiser.cpp:6958-6968).
    ``centered_rows`` lays the rows out as RECOVAR's half images (fftshifted; its row
    ``-size/2`` is FFTW's ``+size/2``), otherwise in FFTW order. Flattened.
    """

    rows = np.arange(size)
    if centered_rows:
        ip = rows - size // 2
        ip = np.where(ip == -(size // 2), size // 2, ip)
    else:
        ip = np.where(rows <= size // 2, rows, rows - size)
    jp = np.arange(size // 2 + 1)
    ip, jp = np.meshgrid(ip, jp, indexing="ij")
    ires = np.floor(np.sqrt(ip * ip + jp * jp) + 0.5).astype(np.int64)
    inside = (ip >= window // 2 + 1 - window) & (ip <= window // 2) & (jp <= window // 2)
    keep = inside & (ires < window // 2 + 1) & ~((jp == 0) & (ip < 0))
    return np.where(keep, ires, -1).reshape(-1)


def premultiplied_ctf2_shell_sums(experiment_dataset, image_indices, image_shape, window: int, *, chunk: int = 1024):
    """Per image, RELION's ``Fctf`` summed over each ``Mresol_fine`` shell of the current size.

    ``Fctf`` is the image's exact CTF row (CTF^2 for a premultiplied optics group); the
    shells are those of :func:`_fftw_shell_labels` at ``window``. Returns an
    ``(images, window // 2 + 1)`` float64 array, the per-image term RELION adds to
    ``wsum_model.sumw_ctf2`` for premultiplied images (acc_ml_optimiser_impl.h, the
    ``getCtfPremultiplied`` block of storeWeightedSums).
    """

    size = int(image_shape[0])
    labels = _fftw_shell_labels(size, int(window), centered_rows=True)
    pixels = np.flatnonzero(labels >= 0)
    order = np.argsort(labels[pixels], kind="stable")
    pixels = pixels[order]
    shells = labels[pixels]
    starts = np.flatnonzero(np.r_[True, shells[1:] != shells[:-1]])
    image_indices = np.asarray(image_indices, dtype=np.int64)
    sums = np.zeros((image_indices.size, int(window) // 2 + 1), dtype=np.float64)
    for begin in range(0, image_indices.size, int(chunk)):
        block = image_indices[begin : begin + int(chunk)]
        # RECOVAR's frame holds -Fctf (relion_ctf._evaluate_exact_ctf_rows).
        rows = -np.asarray(
            _relion_exact_ctf_half_from_source_star_host(experiment_dataset, block, image_shape, pixel_indices=pixels),
            dtype=np.float64,
        )
        sums[begin : begin + block.size, shells[starts]] = np.add.reduceat(rows, starts, axis=1)
    return sums


def premultiplied_average_ctf2(experiment_datasets, scale_corrections, window: int, ori_size: int):
    """RELION's ``setAverageCTF2`` (ml_optimiser.cpp:5697-5740), or ``None`` without premultiplied images.

    ``avgctf2[ires] = sum_images max(0.001, scale) * sum_{shell} Fctf / (N * Npix_per_shell[ires])``
    over every image of ``experiment_datasets`` (one per half), the numerator only over
    CTF-premultiplied images and the denominator over all of them (each image's weights
    sum to one, so ``sumw_group`` counts images). ``scale_corrections`` holds each
    half's per-image scale correction of this iteration's E-step (``None``: 1).
    ``window`` is the E-step's image current size, ``ori_size`` the model's box. RELION
    uses it without split halves and with tau2 not fixed (Class3D and InitialModel):
    it divides ``invtau2`` by ``avgctf2`` in ``BackProjector::updateSSNRarrays``
    (backprojector.cpp:1277-1279), which scales ``data_vs_prior`` by ``avgctf2``.
    """

    n_shells = int(ori_size) // 2 + 1
    numerator = np.zeros(n_shells, dtype=np.float64)
    n_images = 0
    found = False
    for dataset, scales in zip(experiment_datasets, scale_corrections):
        count = int(dataset.n_units)
        n_images += count
        image_shape = tuple(int(v) for v in dataset.image_shape)
        flags = premultiplied_ctf_rows(dataset, np.arange(count), image_shape)
        if flags is None:
            continue
        found = True
        indices = np.flatnonzero(flags)
        weights = np.ones(count) if scales is None else np.asarray(scales, dtype=np.float64).reshape(-1)
        sums = premultiplied_ctf2_shell_sums(dataset, indices, image_shape, window)
        term = np.maximum(0.001, weights[indices]) @ sums
        numerator[: min(n_shells, term.size)] += term[:n_shells]
    if not found:
        return None
    npix = np.bincount(
        (labels := _fftw_shell_labels(int(ori_size), int(ori_size), centered_rows=False))[labels >= 0],
        minlength=n_shells,
    )[:n_shells].astype(np.float64)
    denominator = n_images * npix
    return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), numerator)


def require_no_premultiplied_ctf(experiment_dataset, image_indices, image_shape, *, where: str) -> None:
    """Refuse CTF-premultiplied images on a path that backprojects them as ordinary ones."""

    flags = premultiplied_ctf_rows(experiment_dataset, image_indices, image_shape)
    if flags is not None and flags.any():
        raise NotImplementedError(
            f"{where} does not implement CTF-premultiplied images; they run on the resident sparse pass 2"
        )


def _refuse_generic_ctf(ctf_params, image_shape, voxel_size, *, half_image=False, **kwargs):
    raise NotImplementedError(
        "CTF-premultiplied images, even Zernike aberrations and magnification need RELION's exact CTF "
        "operands; this path evaluates the dataset's generic CTF, which knows none of them"
    )


def refuse_generic_ctf_for_optics(experiment_dataset) -> bool:
    """Make the dataset's generic CTF evaluator raise when its CTF needs the optics table.

    The generic evaluator (``ForwardModelConfig.compute_ctf``) returns the plain
    CTF. CTF-premultiplied images (scored with ``CTF^2`` and their own BPref
    weights), even Zernike terms and anisotropic magnification are carried only by
    the exact operands of this module
    (:func:`relax.relion.optics_aberrations.dataset_needs_exact_ctf`), so every other
    use fails closed instead of silently ignoring them. Returns whether the evaluator
    was replaced.
    """

    from recovar import core

    from relax.relion.optics_aberrations import dataset_needs_exact_ctf

    if not dataset_needs_exact_ctf(experiment_dataset):
        return False
    experiment_dataset._ctf_evaluator = core.as_ctf_evaluator(_refuse_generic_ctf)
    return True


def _relion_ctf_batch_params(cache, original_indices: np.ndarray) -> np.ndarray:
    """``get_ctf_images_batch`` rows for these particles, in the get_ctf_image argument order."""

    idx = np.asarray(original_indices, dtype=np.int64)
    names = (
        ("rlnDefocusU", None, False),
        ("rlnDefocusV", None, False),
        ("rlnDefocusAngle", None, False),
        ("rlnVoltage", None, True),
        ("rlnSphericalAberration", None, True),
        ("rlnAmplitudeContrast", None, True),
        ("rlnCtfBfactor", 0.0, False),
        ("rlnImagePixelSize", None, True),
        ("rlnPhaseShift", 0.0, False),
        ("rlnCtfScalefactor", 1.0, False),
    )
    return np.ascontiguousarray(
        np.stack([_relion_ctf_column(cache, n, d, optics_only=o)[idx] for n, d, o in names], axis=1)
    )


def _exact_ctf_source_cache(experiment_dataset, image_shape):
    """The source STAR, its parsed tables and the per-particle CTF row block."""

    source_path = _relion_exact_ctf_source_star(experiment_dataset)
    cache_key = (str(source_path), tuple(int(size) for size in image_shape))
    cache = _RELION_EXACT_CTF_SOURCE_CACHE.get(cache_key)
    if cache is None:
        from recovar.data_io.starfile import read_star

        particles, optics = read_star(str(source_path))
        if optics is None:
            raise ValueError(f"RELION source STAR has no optics table: {source_path}")
        optics_ids = np.asarray(star_column(optics, "rlnOpticsGroup", required=True), dtype=np.int64)
        if np.unique(optics_ids).size != optics_ids.size:
            raise ValueError(f"RELION source STAR has duplicate optics groups: {source_path}")
        # Cached CTF rows live in one 2-D block rather than a dict of rows, so a
        # batch is gathered with two vectorized indexing operations instead of a
        # Python loop per particle. `slots` maps a particle's original index to its
        # row in that block, with -1 meaning "not evaluated yet"; the block has a
        # row for every particle and is allocated once.
        cache = {
            "particles": particles,
            "optics": {
                int(group): optics.iloc[row]
                for row, group in enumerate(optics_ids)
            },
            # Per-tilt rows of RELION tomo particles (tomo_input.flatten_relion5_tomo).
            "tomo": star_column(particles, "rlnMicrographPreExposure") is not None,
            "slots": np.full(len(particles), -1, dtype=np.int64),
            "rows": None,
            "n_cached": 0,
        }
        _RELION_EXACT_CTF_SOURCE_CACHE[cache_key] = cache

    return source_path, cache


# Particles per CTF evaluation chunk: 8 rows of a box-256 half grid are 2.1 MB of float64,
# so a chunk and its scratch stay in one core's cache.
_CTF_ROW_CHUNK = 8


def _relion_ctf_grid(size: int, extent: float, mag):
    """The pixel coordinates of ``CTF::getCTF`` on the FFTW half grid, shared by every particle.

    ``x``, ``y``, ``u^2 = x^2 + y^2`` and ``u^4``, each flattened to ``size * (size // 2 + 1)``. getFftwImage (ctf.cpp:398-452): row i is frequency i up
    to size / 2, i - size after, in units of 1 / (size * angpix); getCTF (ctf.h:184-257)
    applies the magnification first.
    """

    rows = np.arange(size)
    y0 = (np.where(rows <= size // 2, rows, rows - size) / extent)[:, None]
    x0 = (np.arange(size // 2 + 1) / extent)[None, :]
    if mag is None:
        x, y = np.broadcast_to(x0, (size, size // 2 + 1)), np.broadcast_to(y0, (size, size // 2 + 1))
    else:
        x, y = mag[0, 0] * x0 + mag[0, 1] * y0, mag[1, 0] * x0 + mag[1, 1] * y0
    x, y = np.ascontiguousarray(x).reshape(-1), np.ascontiguousarray(y).reshape(-1)
    u2 = x * x + y * y
    return x, y, u2, u2 * u2


def _relion_ctf_coefficients(params):
    """Per-particle constants of gamma, ``(Axx, 2 Axy, Ayy, K1, K2, K5, K3)``, from CTF::initialise (ctf.cpp:211-262)."""

    du, dv, angle, voltage, cs, q0 = (params[:, i] for i in range(6))
    phase = params[:, 8]
    volts = voltage * 1e3
    lam = 12.2643247 / np.sqrt(volts * (1.0 + volts * 0.978466e-6))
    k1 = np.pi / 2 * 2 * lam
    k2 = np.pi / 2 * (cs * 1e7) * lam * lam * lam
    k3 = np.arctan(q0 / np.sqrt(1 - q0 * q0))
    k5 = phase * np.pi / 180
    az = angle * np.pi / 180
    sin_az, cos_az = np.sin(az), np.cos(az)
    # A = Qt D Q, Q = [[cos, sin], [-sin, cos]], D = diag(-defU, -defV).
    axx = cos_az * cos_az * -du + sin_az * sin_az * -dv
    axy = cos_az * sin_az * -du - sin_az * cos_az * -dv
    ayy = sin_az * sin_az * -du + cos_az * cos_az * -dv
    return axx, 2.0 * axy, ayy, k1, k2, k5, k3


def _relion_ctf_chunks(params, image_size: int, pixel_size: float, gamma_offset, mag_matrix, *, out=None, finish=None):
    """Evaluate ``CTF::getFftwImage`` rows in chunks over host threads.

    A chunk of rows ``start:stop`` is computed into ``out[start:stop]`` (``(N, size * (size // 2 + 1))``
    float64) or, without ``out``, into per-thread scratch; ``finish(start, stop, ctf)``, when
    given, consumes it in the same worker. The pixel coordinates are computed once per call and
    every chunk operation writes in place, in RELION's order of operations; NumPy releases
    the GIL inside each operation, so the chunks run in parallel.
    """

    params = np.ascontiguousarray(params, dtype=np.float64).reshape(-1, 9)
    size = int(image_size)
    x, y, u2, u4 = _relion_ctf_grid(size, float(size) * float(pixel_size), mag_matrix)
    offset = None if gamma_offset is None else np.asarray(gamma_offset, dtype=np.float64).reshape(1, -1)
    axx, axy2, ayy, k1, k2, k5, k3 = (c[:, None] for c in _relion_ctf_coefficients(params))
    bfactor, neg_scale = params[:, 6, None], -params[:, 7, None]
    n = params.shape[0]
    local = threading.local()

    def fill(start):
        stop = min(start + _CTF_ROW_CHUNK, n)
        rows = slice(start, stop)
        if not hasattr(local, "tmp"):
            local.tmp = np.empty((_CTF_ROW_CHUNK, x.size))
            local.ctf = None if out is not None else np.empty((_CTF_ROW_CHUNK, x.size))
        tmp = local.tmp[: stop - start]
        ctf = out[rows] if out is not None else local.ctf[: stop - start]
        # gamma = K1 * (Axx * X * X + 2.0 * Axy * X * Y + Ayy * Y * Y) + K2 * u4 - K5 - K3 (ctf.h:209-217).
        np.multiply(axx[rows], x, out=ctf)
        ctf *= x
        np.multiply(axy2[rows], x, out=tmp)
        tmp *= y
        ctf += tmp
        np.multiply(ayy[rows], y, out=tmp)
        tmp *= y
        ctf += tmp
        ctf *= k1[rows]
        ctf += np.multiply(k2[rows], u4, out=tmp)
        ctf -= k5[rows]
        ctf -= k3[rows]
        if offset is not None:
            ctf += offset
        np.sin(ctf, out=ctf)
        # do_damping (relion_refine always damps): the B-factor envelope exp(K4 u2), K4 = -Bfac / 4,
        # before the scale (ctf.h:219-246). A dose-weighted (tomo) image is damped by its dose
        # instead (relion_tomo_damping) and passes Bfac 0, for which the envelope is exactly 1,
        # so chunks without a B-factor skip it.
        if np.any(bfactor[rows]):
            ctf *= np.exp(np.multiply(-bfactor[rows] / 4.0, u2, out=tmp), out=tmp)
        ctf *= neg_scale[rows]  # CTF = -sin(gamma) * envelope * scale
        # |CTF| >= 1e-8 with SGN(0) = 1 (ctf.h:250-253, macros.h:143).
        small = np.abs(ctf, out=tmp) < 1e-8
        if small.any():
            ctf[small] = np.where(ctf[small] >= 0, 1e-8, -1e-8)
        if finish is not None:
            finish(start, stop, ctf)

    with ThreadPoolExecutor(max_workers=_relion_ctf_threads()) as pool:
        list(pool.map(fill, range(0, n, _CTF_ROW_CHUNK)))


def relion_ctf_fftw_half(params, image_size: int, pixel_size: float, *, gamma_offset=None, mag_matrix=None, finish=None):
    """RELION's ``CTF::getFftwImage`` for particles of one optics group, ``(N, size, size // 2 + 1)`` float64.

    relax's own host implementation of the CTF relion_refine evaluates per particle
    (``setValuesByGroup`` + ``getFftwImage``, ml_optimiser.cpp:6461-6484) in RELION's
    double precision, with its B-factor damping and without CTF padding.
    ``params`` rows are defU, defV, defAng (degrees), voltage (kV), Cs (mm), Q0, the
    particle's CTF B-factor (rlnCtfBfactor, A^2), scale (rlnCtfScalefactor), phase shift
    (degrees). ``gamma_offset`` (the group's even Zernike phase on this grid,
    ``ObservationModel::getGammaOffset``) and ``mag_matrix`` (its 2x2 anisotropic
    magnification) are the optics-table terms. No RELION code runs here: the binding's
    ``get_ctf_images_batch`` / ``optics_ctf_images_batch`` are the test oracles
    (tests/unit/test_relion_ctf_formula.py). Rows are split over host threads. With
    ``finish``, each chunk of rows is handed to ``finish(start, stop, ctf)`` in its worker
    thread, ``ctf`` flattened to ``(stop - start, size * (size // 2 + 1))`` in per-thread
    scratch, and nothing is returned.
    """

    params = np.ascontiguousarray(params, dtype=np.float64).reshape(-1, 9)
    size = int(image_size)
    mag = None if mag_matrix is None else np.asarray(mag_matrix, dtype=np.float64)
    if finish is not None:
        _relion_ctf_chunks(params, size, pixel_size, gamma_offset, mag, finish=finish)
        return None
    out = np.empty((params.shape[0], size * (size // 2 + 1)), dtype=np.float64)
    _relion_ctf_chunks(params, size, pixel_size, gamma_offset, mag, out=out)
    return out.reshape(-1, size, size // 2 + 1)


def _optics_group_ctf_geometry(cache, group: int, size: int):
    """``(gamma_offset, mag_matrix)`` of one optics group on a ``size`` grid, each None when absent.

    The even Zernike gamma offset is ``ObservationModel::getGammaOffset`` on the FFTW
    half grid (obs_model.cpp:1263-1307), evaluated by relax
    (:func:`relax.relion.optics_aberrations.zernike_phase_fftw_half`).
    """

    from relax.relion import optics_aberrations as oa

    geometry = cache.setdefault("group_geometry", {})
    key = (int(group), int(size))
    if key not in geometry:
        row = cache["optics"][int(group)]
        labels = {str(label).lstrip("_") for optics in cache["optics"].values() for label in optics.keys()}
        mag = oa.optics_group_mag_matrix(row) if any(label.startswith("rlnMagMat") for label in labels) else None
        even = oa.parse_relion_vector(oa._optics_value(row, "rlnEvenZernike", "[]"))
        gamma = None
        if any(even):
            box = int(oa._optics_value(row, "rlnImageSize", size))
            if box != size:
                raise ValueError(f"even Zernike terms need the optics group's box {box}, got an image of {size}")
            gamma = oa.zernike_phase_fftw_half(
                even, oa.even_index_to_mn, size, float(oa._optics_value(row, "rlnImagePixelSize")), box, mag
            )
        geometry[key] = (gamma, mag)
    return geometry[key]


def _evaluate_exact_ctf_rows(cache, original_indices, image_h: int, image_w: int) -> np.ndarray:
    """Evaluate the particles' missing CTF rows into the block; return their row slots."""

    slots = cache["slots"]
    missing = np.unique(original_indices[slots[original_indices] < 0])
    if missing.size:
        # RELION's CTF of every missing particle, evaluated by relax on the host
        # (relion_ctf_fftw_half), one call per optics group: its pixel size, even
        # Zernike gamma offset and anisotropic magnification (ml_optimiser.cpp:6461-6484).
        if image_h != image_w:
            raise ValueError("RELION's CTF rows need square images")
        groups = np.asarray(star_column(cache["particles"], "rlnOpticsGroup", required=True), dtype=np.int64)[missing]
        # Each optics group fills one contiguous run of new cache rows.
        missing = missing[np.argsort(groups, kind="stable")]
        groups = np.sort(groups, kind="stable")
        n_new = int(missing.size)
        params = _relion_ctf_batch_params(cache, missing)
        tomo = cache.get("tomo", False)
        if tomo:
            dose = _relion_ctf_column(cache, "rlnMicrographPreExposure", None)[missing]
            dose_bfactor = _relion_ctf_column(cache, "rlnCtfBfactorPerElectronDose", 0.0)[missing]
        premultiplied = _premultiplied_particles(cache)[missing]
        rows = cache["rows"]
        used = cache["n_cached"]
        width = image_h * (image_w // 2 + 1)
        if rows is None:
            # One block with a row for every particle of the source STAR; pages are
            # touched only as rows are evaluated. Growing by doubling recopied the
            # whole cache (17.5 s of a noise1 50k VDAM run, job 14523070).
            rows = np.empty((slots.size, width), dtype=np.float64)
        shift = image_h // 2            # np.fft.fftshift is np.roll(x, image_h // 2)
        split = image_h - shift
        block = rows[used : used + n_new].reshape(n_new, image_h, image_w // 2 + 1)
        for group in np.unique(groups):
            first, last = np.searchsorted(groups, [group, group + 1])
            pixel = np.unique(params[first:last, 7])
            if pixel.size != 1:
                raise ValueError(f"optics group {int(group)} has several pixel sizes")
            gamma, mag = _optics_group_ctf_geometry(cache, int(group), image_h)
            columns = params[first:last][:, [0, 1, 2, 3, 4, 5, 6, 9, 8]]  # ..., Q0, Bfac, scale, phase shift
            if tomo:
                columns[:, 6] = 0.0  # dose-weighted: damped by the dose below, as RELION does for dose >= 0
                freq_sq = fftw_half_freq_sq(image_h, image_w, float(pixel[0]))

            def finish(start, stop, ctf, first=first):
                chunk = slice(first + start, first + stop)
                ctf = ctf.reshape(stop - start, image_h, image_w // 2 + 1)
                if tomo:
                    # A RELION tomo image (one row per particle-tilt): relion_refine damps its
                    # CTF by the tilt's cumulative dose (tomo_input.relion_tomo_damping). RELION
                    # multiplies before the scale factor and before the |CTF| >= 1e-8 floor
                    # (src/ctf.h:219-253); applying it here reorders one product and moves the
                    # floor, both below 1e-8 absolute.
                    ctf *= relion_tomo_damping(freq_sq, dose[chunk], dose_bfactor[chunk])
                if premultiplied[chunk].any():
                    # CTF-premultiplied images: RELION squares its CTF image right after
                    # getFftwImage and then uses the ordinary scoring kernels with it
                    # (ml_optimiser.cpp:6486-6492, acc_ml_optimiser_impl.h:840-847).
                    square = premultiplied[chunk]
                    ctf[square] = ctf[square] * ctf[square]
                # RELION/FFTW stores y in standard order and uses the opposite CTF
                # sign from RECOVAR's forward-model convention: one roll and negation
                # of the row axis, written straight into the cache block.
                np.negative(ctf[:, split:], out=block[chunk, :shift])
                np.negative(ctf[:, :split], out=block[chunk, shift:])

            relion_ctf_fftw_half(columns, image_h, float(pixel[0]), gamma_offset=gamma, mag_matrix=mag, finish=finish)
        cache["rows"] = rows
        cache["n_cached"] = used + n_new
        slots[missing] = np.arange(used, used + n_new, dtype=np.int64)

    batch_slots = slots[original_indices]
    if np.any(batch_slots < 0):
        raise RuntimeError("a requested RELION CTF row was not evaluated")
    return batch_slots


def _relion_exact_ctf_half_from_source_star_host(
    experiment_dataset,
    image_indices,
    image_shape,
    *,
    pixel_indices=None,
):
    """Evaluate source-precision SPA CTFs into one host-native operand.

    The result uses RECOVAR's centered-y half-spectrum coordinates and sign.
    The source STAR is mandatory because the ordinary dataset metadata has
    already been rounded to float32 before pass 2.  RELION's binding and the
    source cache are host-native; callers that must pad on the image axis use
    this helper so they place the final operand exactly once. Optional host
    pixel indices gather the requested columns before stacking full CTF rows;
    their order and duplicates are preserved without changing source precision.
    """

    source_path, cache = _exact_ctf_source_cache(experiment_dataset, image_shape)
    original_indices = original_image_indices(
        experiment_dataset,
        np.asarray(image_indices, dtype=np.int64),
    )
    image_h, image_w = (int(size) for size in image_shape)
    if image_h != image_w:
        raise ValueError("exact RELION CTF replay currently requires square images")
    if pixel_indices is not None:
        if not isinstance(pixel_indices, np.ndarray):
            raise TypeError("CTF pixel indices must already be a host NumPy array")
        if pixel_indices.ndim != 1 or pixel_indices.dtype.kind not in "iu":
            raise ValueError("CTF pixel indices must be a one-dimensional integer array")
        if np.any(pixel_indices < 0) or np.any(pixel_indices >= image_h * (image_w // 2 + 1)):
            raise ValueError("CTF pixel indices are outside the full half-spectrum")
    original_indices = np.asarray(original_indices, dtype=np.int64)
    cache_key = _exact_ctf_result_key(source_path, image_shape, original_indices, pixel_indices)
    memoized = _exact_ctf_result_lookup(cache_key, original_indices, pixel_indices)
    if memoized is not None:
        return memoized

    batch_slots = _evaluate_exact_ctf_rows(cache, original_indices, image_h, image_w)
    rows = cache["rows"]
    # Gather rows and the planned columns in one step, so the intermediate is the
    # size of the result rather than of the full half-spectrum.
    gathered = (
        rows[batch_slots]
        if pixel_indices is None
        else rows[np.ix_(batch_slots, np.asarray(pixel_indices, dtype=np.int64))]
    )
    assembled = np.asarray(gathered, dtype=np.float64)
    return _exact_ctf_result_store(cache_key, original_indices, pixel_indices, assembled)


def _relion_exact_ctf_half_from_source_star(
    experiment_dataset,
    image_indices,
    image_shape,
    *,
    pixel_indices=None,
):
    """Return the source-precision CTF rows of these images on the JAX device.

    The binary64 rows are evaluated once per particle on the host
    (:func:`_evaluate_exact_ctf_rows`) and served from a bounded device row
    cache (:func:`_exact_ctf_device_rows`): a request uploads only the rows the
    cache does not hold and is one device gather, optionally of planned pixel
    columns, whose order and duplicates are preserved. The host path copied and
    uploaded every requested row on each call, 72.6 s of main-thread gather in a
    noise1 50k VDAM run (py-spy, job 14523070).
    """

    image_h, image_w = (int(size) for size in image_shape)
    if image_h != image_w:
        raise ValueError("exact RELION CTF replay currently requires square images")
    width = image_h * (image_w // 2 + 1)
    if pixel_indices is not None:
        pixel_indices = np.asarray(pixel_indices)
        if pixel_indices.ndim != 1 or pixel_indices.dtype.kind not in "iu":
            raise ValueError("CTF pixel indices must be a one-dimensional integer array")
        if np.any(pixel_indices < 0) or np.any(pixel_indices >= width):
            raise ValueError("CTF pixel indices are outside the full half-spectrum")
    _, cache = _exact_ctf_source_cache(experiment_dataset, image_shape)
    original_indices = np.asarray(
        original_image_indices(experiment_dataset, np.asarray(image_indices, dtype=np.int64)), dtype=np.int64
    )
    batch_slots = _evaluate_exact_ctf_rows(cache, original_indices, image_h, image_w)
    block, device_rows = _exact_ctf_device_rows(cache, batch_slots, width)
    return _gather_ctf_rows(block, device_rows, None if pixel_indices is None else pixel_indices.astype(np.int32))


@jax.jit
def _gather_ctf_rows(block, rows, pixel_indices):
    """``block[rows]``, or ``block[rows][:, pixel_indices]``, as one program.

    Loose advanced indexing builds its gather on the host at every call: 3.1 s of a 60 s window of late
    10k VDAM iterations (py-spy, 510 coarse batches).
    """

    if pixel_indices is None:
        return jnp.take(block, rows, axis=0)
    return block[rows[:, None], pixel_indices[None, :]]


_EXACT_CTF_DEVICE_GB_ENV = "RELAX_RELION_EXACT_CTF_DEVICE_GB"
# The cache takes this share of what the allocator can still hand out (plus what
# it already holds), so the passes that plan from free memory keep the rest.
_EXACT_CTF_DEVICE_FREE_FRACTION = 0.25
# Shrink only below this share of the current capacity, so small swings in free
# memory do not rebuild the cache.
_EXACT_CTF_DEVICE_SHRINK_BELOW = 0.75
# Uploads go up in power-of-two row chunks of at most this many rows: few scatter
# shapes compile and no padded rows travel (a box-800 row is 2.57 MB).
_EXACT_CTF_DEVICE_UPLOAD_ROWS = 256


def _exact_ctf_device_budget_bytes(held_bytes: int) -> int:
    """Device bytes the CTF row cache may hold: ``RELAX_RELION_EXACT_CTF_DEVICE_GB``, else
    ``_EXACT_CTF_DEVICE_FREE_FRACTION`` of the allocator's available bytes now plus the
    ``held_bytes`` the cache already occupies (``device_available_bytes``); the held
    bytes alone when the device reports nothing."""

    token = os.environ.get(_EXACT_CTF_DEVICE_GB_ENV, "").strip()
    if token:
        try:
            budget = float(token)
        except ValueError as exc:
            raise ValueError(f"{_EXACT_CTF_DEVICE_GB_ENV} must be a number of gigabytes, got {token!r}") from exc
        if budget <= 0:
            raise ValueError(f"{_EXACT_CTF_DEVICE_GB_ENV} must be positive, got {token!r}")
        return int(budget * (1024**3))
    from relax.sparse_pass2.sparse_pass2_budget import (
        _device_free_memory_bytes,
        _jax_allocator_free_memory_bytes,
        _jax_allocator_pool_free_bytes,
        device_available_bytes,
    )

    available = device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(), _jax_allocator_pool_free_bytes()
    )
    if available is None:
        return int(held_bytes)
    return int((float(available) + held_bytes) * _EXACT_CTF_DEVICE_FREE_FRACTION)


def release_exact_ctf_device_cache() -> None:
    """Free every device CTF row cache; the next request rebuilds one at the budget then.

    A pass whose fixed allocations need the memory the cache holds (for example a
    box-800 reconstruction accumulator) calls this before allocating them.
    """

    for cache in _RELION_EXACT_CTF_SOURCE_CACHE.values():
        cache.pop("device_cache", None)


def ensure_device_headroom(n_bytes: int) -> bool:
    """Free the device CTF row caches only if the device cannot hand out ``n_bytes`` now.

    For a pass about to allocate large fixed buffers (a box-800 reconstruction
    accumulator): at 256 px it never fires, so the cache keeps its rows across
    passes. Returns whether it released anything. Call it once before the
    allocation, not per chunk.
    """

    from relax.sparse_pass2.sparse_pass2_budget import (
        _device_free_memory_bytes,
        _jax_allocator_free_memory_bytes,
        _jax_allocator_pool_free_bytes,
        device_available_bytes,
    )

    available = device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(), _jax_allocator_pool_free_bytes()
    )
    held = any("device_cache" in cache for cache in _RELION_EXACT_CTF_SOURCE_CACHE.values())
    if not held or available is None or available >= n_bytes:
        return False
    release_exact_ctf_device_cache()
    return True


def _exact_ctf_device_capacity(n_particles: int, request_rows: int, row_bytes: int, held_rows: int) -> int:
    """Rows the cache should hold: the budget's rows, at least the request, at most one per particle."""

    budget_rows = _exact_ctf_device_budget_bytes(held_rows * row_bytes) // row_bytes
    return int(min(n_particles, max(request_rows, budget_rows)))


def _exact_ctf_device_rows(cache, batch_slots: np.ndarray, width: int):
    """The device row cache and the cache rows of ``batch_slots``.

    The cache holds as many rows as :func:`_exact_ctf_device_budget_bytes` allows,
    re-read at every request, and never fewer than the request needs nor more than
    one per particle. It is rebuilt empty at the budget when a request does not fit
    or the budget falls below ``_EXACT_CTF_DEVICE_SHRINK_BELOW`` of its capacity.
    Rows a request needs and the cache lacks replace rows in clock order, skipping
    rows the same request reads, and go up in place. It grows only when a request
    does not fit (the first request of a pass, as batch sizes are fixed), never
    because memory freed up, so it does not race a chunk loop's next allocation.
    """

    import jax

    state = cache.get("device_cache")
    n_particles = cache["slots"].size
    row_bytes = width * 8
    requested = np.unique(batch_slots)
    capacity = 0 if state is None else state["block"].shape[0]
    target = _exact_ctf_device_capacity(n_particles, requested.size, row_bytes, capacity)
    if capacity < requested.size or target < _EXACT_CTF_DEVICE_SHRINK_BELOW * capacity:
        cache.pop("device_cache", None)
        state = None  # drop the old block before allocating the new one
        state = {
            "block": jnp.zeros((target, width), dtype=jnp.float64),
            "row_of_slot": np.full(n_particles, -1, dtype=np.int64),
            "slot_of_row": np.full(target, -1, dtype=np.int64),
            "clock": 0,
        }
        cache["device_cache"] = state
    capacity = state["block"].shape[0]
    missing = requested[state["row_of_slot"][requested] < 0]
    if missing.size:
        keep = np.zeros(capacity, dtype=bool)
        keep[state["row_of_slot"][requested[state["row_of_slot"][requested] >= 0]]] = True
        order = (np.arange(capacity) + state["clock"]) % capacity
        free = order[~keep[order]][: missing.size]
        evicted = state["slot_of_row"][free]
        state["row_of_slot"][evicted[evicted >= 0]] = -1
        state["row_of_slot"][missing] = free
        state["slot_of_row"][free] = missing
        state["clock"] = int((free[-1] + 1) % capacity)
        start = 0
        while start < missing.size:
            size = min(_EXACT_CTF_DEVICE_UPLOAD_ROWS, 1 << int(np.log2(missing.size - start)))
            state["block"] = _scatter_exact_ctf_rows(
                state["block"],
                jax.device_put(free[start : start + size].astype(np.int32)),
                jax.device_put(cache["rows"][missing[start : start + size]]),
            )
            start += size
    return state["block"], state["row_of_slot"][batch_slots]


def _scatter_exact_ctf_rows(block, positions, values):
    import jax

    global _scatter_exact_ctf_rows_program
    if _scatter_exact_ctf_rows_program is None:
        _scatter_exact_ctf_rows_program = jax.jit(
            lambda block, positions, values: block.at[positions].set(values), donate_argnums=0
        )
    return _scatter_exact_ctf_rows_program(block, positions, values)


_scatter_exact_ctf_rows_program = None


