"""Shared source-precision RELION CTF operands for coarse, local and sparse EM.

The source STAR's parsed tables keep their per-process lifetime; the CTF rows themselves are evaluated on
demand, at the requested pixels, by one float64 JAX program (:func:`_relion_ctf_program`, which documents
the last-ulp difference from RELION's CPU values) and kept by no cache, as RELION evaluates each particle's
CTF in every expectation. Centered half-spectrum coordinates, signs and caller-selected cast boundaries
are preserved.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from recovar.data_io.starfile import star_column

from relax.fourier.shells import shell_of_radius_sq
from relax.io.batch_fetch import original_image_indices
from relax.relion.tomo_input import relion_tomo_damping

_RELION_EXACT_CTF_SOURCE_CACHE: dict[tuple[str, tuple[int, int]], dict] = {}


def dataset_optics_source_star(experiment_dataset) -> Path | None:
    """The dataset's RELION source STAR, or ``None`` for a dataset built in memory.

    The optics table (CTF-premultiplied, aberrated or magnified groups) is read from this
    STAR. ``particles_file=None`` declares a dataset built from arrays, which has no
    optics table: ``None``, unless the dataset flags its images CTF-premultiplied
    (``premultiplied_ctf``), which only the STAR's exact CTF rows can serve, so that
    raises. A dataset that names a non-STAR source raises, as its optics are unknown;
    an object that cannot name a source at all (``MultiShapeHalf``: ask its shape
    classes' datasets) raises AttributeError.
    """

    source_star = os.environ.get("RELAX_K1_RELION_EXACT_CTF_STAR", "").strip()
    if source_star:
        return Path(source_star).expanduser().resolve()
    dataset_source = experiment_dataset.particles_file
    if dataset_source is None:
        if getattr(experiment_dataset, "premultiplied_ctf", False):
            raise ValueError(
                f"{type(experiment_dataset).__name__} is built in memory (particles_file=None) with CTF-premultiplied "
                "images; RELION's premultiplied operands are evaluated from a particle STAR's optics table"
            )
        return None
    if Path(dataset_source).suffix.lower() != ".star":
        raise ValueError(
            f"{type(experiment_dataset).__name__} is not a STAR-backed dataset (particles_file={dataset_source!r}): its RELION "
            "optics table, and so its CTF-premultiplied, aberrated or magnified optics groups, are unknown; load it "
            "from a particle STAR or set RELAX_K1_RELION_EXACT_CTF_STAR"
        )
    return Path(dataset_source).expanduser().resolve()


def _relion_exact_ctf_source_star(experiment_dataset) -> Path:
    """The source STAR exact RELION CTF evaluation reads (:func:`dataset_optics_source_star`); a dataset
    built in memory has none and raises."""

    source_star = dataset_optics_source_star(experiment_dataset)
    if source_star is None:
        raise ValueError(
            f"exact RELION CTF operands need a STAR-backed dataset; {type(experiment_dataset).__name__} is built in "
            "memory (particles_file=None)"
        )
    return source_star


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
    (:func:`_exact_ctf_rows`); the backprojection operands then differ
    as well (:func:`relax.fine_pass.bucket_io.premultiplied_bpref_weights`).
    ``None`` keeps every ordinary dataset on its unchanged programs, as it does a dataset
    built in memory; a dataset whose optics are unknown raises (:func:`dataset_optics_source_star`).
    """

    if dataset_optics_source_star(experiment_dataset) is None:
        return None
    _, cache = exact_ctf_source_cache(experiment_dataset, image_shape)
    flags = _premultiplied_particles(cache)
    if not flags.any():
        return None
    original_indices = np.asarray(
        original_image_indices(experiment_dataset, np.asarray(image_indices, dtype=np.int64)), dtype=np.int64
    )
    return flags[original_indices]


def dataset_has_premultiplied_ctf(experiment_dataset, image_shape) -> bool:
    """Whether any optics group of the dataset's source STAR stores CTF-premultiplied images.

    A dataset built in memory has none; one whose optics are unknown raises (:func:`dataset_optics_source_star`).
    """

    if dataset_optics_source_star(experiment_dataset) is None:
        return False
    _, cache = exact_ctf_source_cache(experiment_dataset, image_shape)
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
    ires = shell_of_radius_sq(ip * ip + jp * jp, rule="half_up")
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
        # RECOVAR's frame holds -Fctf (relion_ctf._exact_ctf_rows).
        rows = -np.asarray(
            relion_exact_ctf_half_from_source_star_host(experiment_dataset, block, image_shape, pixel_indices=pixels),
            dtype=np.float64,
        )
        sums[begin : begin + block.size, shells[starts]] = np.add.reduceat(rows, starts, axis=1)
    return sums


def premultiplied_average_ctf2(parts, box_size: int, sumw: float):
    """RELION's ``setAverageCTF2`` (ml_optimiser.cpp:4885-4926), or ``None`` without premultiplied images.

    ``avgctf2[ires] = sum_images max(0.001, scale) * sum_{shell} Fctf / (sumw * Npix_per_shell[ires])``,
    the numerator over the CTF-premultiplied images of ``parts`` and the denominator RELION's
    ``sum_igroup wsum_model.sumw_group[igroup] * Npix_per_shell`` (ml_optimiser.cpp:4909-4912).
    ``sumw`` is that sum: each image adds the posterior weights of its significant fine samples
    (``thr_sumw_group += p_weights[n]``, acc_ml_optimiser_impl.h:2842), which keep the top
    ``adaptive_fraction`` of its mass (:2522), so it is below the image count at soft posteriors.
    Every caller passes the E-step's noise ``sumw``, the same sums the noise update divides by. For
    subtomograms ``parts`` holds the tilt images (RELION's numerator loops ``sp.nr_images``, :3590-3600)
    while ``sumw`` adds once per particle, so the average is a particle's CTF^2 summed over its tilts.
    Each part is ``(dataset, scales, window, group_scale)``: a STAR-backed dataset on one image
    grid, its per-image scale corrections of this iteration's E-step (``None``: 1), its image
    current size and its scale difference ``s_g``. A part's shell sums (``Mresol_fine`` of its
    own window) add onto the model shells ``ROUND(i / s_g)``, as storeWeightedSums adds
    ``thr_wsum_ctf2`` into ``sumw_ctf2`` (acc_ml_optimiser_impl.h:3590-3626); ``box_size`` is the
    model's ``ori_size``. RELION uses it without split halves and with tau2 not fixed (Class3D
    and InitialModel): it divides ``invtau2`` by ``avgctf2`` in
    ``BackProjector::updateSSNRarrays`` (backprojector.cpp:1143-1145), which scales
    ``data_vs_prior`` by ``avgctf2``.
    """

    from relax.relion.optics_scale import add_group_shells_to_reference

    n_shells = int(box_size) // 2 + 1
    numerator = np.zeros(n_shells, dtype=np.float64)
    found = False
    for dataset, scales, window, group_scale in parts:
        count = int(dataset.n_units)
        image_shape = tuple(int(v) for v in dataset.image_shape)
        flags = premultiplied_ctf_rows(dataset, np.arange(count), image_shape)
        if flags is None:
            continue
        found = True
        indices = np.flatnonzero(flags)
        weights = np.ones(count) if scales is None else np.asarray(scales, dtype=np.float64).reshape(-1)
        sums = premultiplied_ctf2_shell_sums(dataset, indices, image_shape, int(window))
        numerator = add_group_shells_to_reference(numerator, np.maximum(0.001, weights[indices]) @ sums, group_scale)
    if not found:
        return None
    if not float(sumw) > 0.0:
        raise ValueError(f"the average CTF^2 of premultiplied images needs the E-step's sumw, got {sumw}")
    npix = np.bincount(
        (labels := _fftw_shell_labels(int(box_size), int(box_size), centered_rows=False))[labels >= 0],
        minlength=n_shells,
    )[:n_shells].astype(np.float64)
    denominator = float(sumw) * npix
    return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), numerator)


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


def exact_ctf_source_cache(experiment_dataset, image_shape):
    """The source STAR and its parsed tables."""

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
        # The parsed tables only; CTF rows are evaluated per request (_exact_ctf_rows).
        cache = {
            "particles": particles,
            "optics": {
                int(group): optics.iloc[row]
                for row, group in enumerate(optics_ids)
            },
            # Per-tilt rows of RELION tomo particles (tomo_input.flatten_relion5_tomo).
            "tomo": star_column(particles, "rlnMicrographPreExposure") is not None,
        }
        _RELION_EXACT_CTF_SOURCE_CACHE[cache_key] = cache

    return source_path, cache


def _relion_ctf_constants(params) -> np.ndarray:
    """Per-particle constants of ``CTF::getCTF``, ``(n, 9)`` float64 on the host.

    Columns ``(Axx, 2 Axy, Ayy, K1, K2, K5, K3, K4, -scale)`` from CTF::initialise (ctf.cpp:211-262), with
    ``K4 = -Bfac / 4``. ``params`` rows are :func:`relion_ctf_fftw_half`'s. Nine scalars per particle: the
    per-pixel formula is :func:`_relion_ctf_program` alone.
    """

    params = np.ascontiguousarray(params, dtype=np.float64).reshape(-1, 9)
    du, dv, angle, voltage, cs, q0, bfactor, scale, phase = (params[:, i] for i in range(9))
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
    return np.stack([axx, 2.0 * axy, ayy, k1, k2, k5, k3, -bfactor / 4.0, -scale], axis=1)


@functools.partial(jax.jit, static_argnames=("size", "centered_rows"))
def _relion_ctf_program(constants, pixels, extent, one, mag, offset, dose, square, *, size: int, centered_rows: bool):
    """RELION's ``Fctf`` of ``n`` particles at ``P`` pixels, float64 ``(n, P)``: relax's one CTF formula.

    ``CTF::getCTF`` (`ctf.h <https://github.com/3dem/relion/blob/5.0.1/src/ctf.h#L184-L257>`_) on the
    pixels of ``CTF::getFftwImage``
    (`ctf.cpp <https://github.com/3dem/relion/blob/5.0.1/src/ctf.cpp#L398-L452>`_), with RELION's
    damping and without CTF padding, as relion_refine evaluates it per particle
    (`ml_optimiser.cpp <https://github.com/3dem/relion/blob/5.0.1/src/ml_optimiser.cpp#L6461-L6492>`_).

    ``constants`` are the particles' :func:`_relion_ctf_constants`. ``pixels`` are flat indices into the
    ``size x (size // 2 + 1)`` half grid (None: every pixel, in order): FFTW rows, or RECOVAR's centred
    rows with ``centered_rows`` (its row ``r`` is FFTW's ``(r + size - size // 2) % size``), which also
    returns RECOVAR's sign, the opposite of RELION's. ``extent`` is ``size * angpix``. The optional terms
    are None when absent: ``mag``, the optics group's 2x2 anisotropic magnification, applied to the
    frequency first (ctf.h:189-197); ``offset``, its even Zernike phase on the whole FFTW half grid,
    flattened (``ObservationModel::getGammaOffset``); ``dose``, per particle the tilt's cumulative dose and
    B-factor per electron dose (:func:`relax.relion.tomo_input.relion_tomo_damping`); ``square``, per
    particle whether its optics group stores CTF-premultiplied images, whose ``Fctf`` RELION squares right
    after getFftwImage (ml_optimiser.cpp:6486-6492).

    Float64 by explicit dtypes, in RELION's order of operations (ctf.h:209-253): gamma, ``sin``, the
    B-factor envelope, the scale, the ``|CTF| >= 1e-8`` floor. Every product that feeds a sum is first
    multiplied by ``one``, the run-time value 1.0: that product is exact, so wherever a backend contracts
    ``a * b * one + c`` into a fused multiply-add (XLA's CPU backend does, through an optimization barrier
    too) the result is still the twice-rounded ``fl(fl(a * b) + c)`` and gamma is the double the
    source expression gives, one operation at a time (a compiled RELION may itself contract: its gamma
    then differs by one rounding, about 1e-13 at the edge of the grid). The dose damping multiplies after the scale and the floor, where RELION multiplies before
    them (ctf.h:219-253): one product reordered and the floor moved, both below 1e-8 absolute.

    Last-ulp difference from RELION: ``sin`` and ``exp`` (and the dose damping, its ``pow`` and one
    unguarded sum) are XLA's float64 routines on the device that runs this program, where RELION's CPU
    code calls glibc's libm (RELION's GPU code evaluates no CTF: it uploads the CPU image). Both are
    accurate to about one unit in the last place but round differently, so a value may differ from
    RELION's CPU double by a relative 1e-16 to 1e-15, and from one device type to another by as much. A
    float32 operand derived from these rows changes only where the double sat within that distance of a
    float32 rounding boundary: 5 of the 9.8e9 values of EMPIAR-10202's 30515 rows at box 800, against
    the glibc evaluation (relax#39).
    """

    width = size // 2 + 1
    if pixels is None:
        pixels = jnp.arange(size * width, dtype=jnp.int32)
    row, column = pixels // width, pixels % width
    if centered_rows:
        row = (row + (size - size // 2)) % size
    if offset is not None:
        offset = offset[row * width + column]
    # getFftwImage: row i is frequency i up to size / 2, i - size after, in units of 1 / (size * angpix).
    x = column.astype(jnp.float64) / extent
    y = jnp.where(row <= size // 2, row, row - size).astype(jnp.float64) / extent
    if mag is not None:
        x, y = (
            mag[0, 0] * x * one + mag[0, 1] * y * one,
            mag[1, 0] * x * one + mag[1, 1] * y * one,
        )
    # Do not remove ``* one``. Without it XLA's CPU backend contracts a product and the sum it feeds into one
    # fused multiply-add (an optimization barrier does not stop it): gamma moves by one ulp, the CTF by up to
    # 3e-7 relative next to a zero, and 22 of 1.2e7 float32 casts of box-800 rows differed from the glibc
    # evaluation (0 with it, on the CPU backend and on a GPU). ``one`` is a run-time 1.0, so the compiler
    # keeps the product; it is exact, so a contraction of ``p * one + c`` returns ``fl(p + c)``.
    u2 = x * x * one + y * y * one
    u4 = u2 * u2
    axx, axy2, ayy, k1, k2, k5, k3, k4, neg_scale = (constants[:, i, None] for i in range(9))
    # gamma = K1 * (Axx * X * X + 2.0 * Axy * X * Y + Ayy * Y * Y) + K2 * u4 - K5 - K3 (ctf.h:209-217).
    gamma = axx * x * x * one + axy2 * x * y * one
    gamma = gamma + ayy * y * y * one
    gamma = gamma * k1 * one + k2 * u4 * one
    gamma = gamma - k5 - k3
    if offset is not None:
        gamma = gamma + offset
    # do_damping: the B-factor envelope exp(K4 u2) before the scale (ctf.h:219-246); a particle without a
    # B-factor multiplies by exp(-0.0) = 1. CTF = -sin(gamma) * envelope * scale.
    ctf = jnp.sin(gamma) * jnp.exp(k4 * u2) * neg_scale
    # |CTF| >= 1e-8 with SGN(0) = 1 (ctf.h:250-253, macros.h:143).
    ctf = jnp.where(jnp.abs(ctf) < 1e-8, jnp.where(ctf >= 0, 1e-8, -1e-8), ctf)
    if dose is not None:
        ctf = ctf * relion_tomo_damping(u2, dose[:, 0], dose[:, 1])
    if square is not None:
        ctf = jnp.where(square[:, None], ctf * ctf, ctf)
    return -ctf if centered_rows else ctf


# Elements of one CTF program call: 128 MB of float64 result, and a few times that while it runs.
_CTF_BLOCK_ELEMENTS = 1 << 24
# Particles of one CTF program call at most.
_CTF_BLOCK_ROWS = 256


def _relion_ctf_rows(
    constants, size: int, extent: float, *, pixels=None, mag=None, offset=None, dose=None, square=None, centered_rows=False
):
    """:func:`_relion_ctf_program` of every row of ``constants``, a device ``(n, P)`` float64 array.

    Evaluated now on the default JAX device (the CPU backend when that is the default) and kept nowhere.
    The particles go through the program in blocks of a power of two of rows, a short block padded with
    its last particle, so one request compiles a few row counts per pixel count; a block holds at most
    :data:`_CTF_BLOCK_ELEMENTS` values. ``dose`` is ``(n, 2)`` and ``square`` ``(n,)`` (see the program).
    """

    if not jax.config.jax_enable_x64:
        raise RuntimeError("RELION's CTF is evaluated in float64; JAX x64 must be enabled (recovar enables it)")
    constants = np.ascontiguousarray(constants, dtype=np.float64).reshape(-1, 9)
    n = constants.shape[0]
    n_pixels = size * (size // 2 + 1) if pixels is None else int(np.asarray(pixels).size)
    if n == 0:
        return jnp.zeros((0, n_pixels), dtype=jnp.float64)
    cap = max(1, min(_CTF_BLOCK_ROWS, _CTF_BLOCK_ELEMENTS // max(1, n_pixels)))
    cap = 1 << (cap.bit_length() - 1)
    shared = (
        None if pixels is None else jnp.asarray(pixels, dtype=jnp.int32),
        np.float64(extent),
        np.float64(1.0),
        None if mag is None else jnp.asarray(mag, dtype=jnp.float64).reshape(2, 2),
        None if offset is None else jnp.asarray(offset, dtype=jnp.float64).reshape(-1),
    )
    blocks = []
    for start in range(0, n, cap):
        rows = min(cap, n - start)
        padded = 1 << (rows - 1).bit_length()
        take = np.minimum(np.arange(start, start + padded), n - 1)
        block = _relion_ctf_program(
            jnp.asarray(constants[take]),
            *shared,
            None if dose is None else jnp.asarray(np.asarray(dose, dtype=np.float64)[take]),
            None if square is None else jnp.asarray(np.asarray(square, dtype=bool)[take]),
            size=size,
            centered_rows=bool(centered_rows),
        )
        blocks.append(block if padded == rows else jax.lax.slice_in_dim(block, 0, rows))
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=0)


def relion_ctf_fftw_half(params, box_size: int, pixel_size: float, *, gamma_offset=None, mag_matrix=None):
    """RELION's ``CTF::getFftwImage`` for particles of one optics group, ``(N, size, size // 2 + 1)`` float64.

    The CTF relion_refine evaluates per particle (``setValuesByGroup`` + ``getFftwImage``,
    ml_optimiser.cpp:6461-6484), by relax's one CTF program (:func:`_relion_ctf_program`; its last-ulp
    note applies). ``params`` rows are defU, defV, defAng (degrees), voltage (kV), Cs (mm), Q0, the
    particle's CTF B-factor (rlnCtfBfactor, A^2), scale (rlnCtfScalefactor), phase shift (degrees).
    ``gamma_offset`` (the group's even Zernike phase on this grid, ``ObservationModel::getGammaOffset``)
    and ``mag_matrix`` (its 2x2 anisotropic magnification) are the optics-table terms. No RELION code runs
    here: the binding's ``get_ctf_images_batch`` / ``optics_ctf_images_batch`` are the test oracles
    (tests/unit/test_relion_ctf_formula.py). Returns a host array the caller owns.
    """

    size = int(box_size)
    rows = _relion_ctf_rows(
        _relion_ctf_constants(params), size, float(size) * float(pixel_size), mag=mag_matrix, offset=gamma_offset
    )
    return np.array(rows).reshape(-1, size, size // 2 + 1)


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
            box_size = int(oa._optics_value(row, "rlnImageSize", size))
            if box_size != size:
                raise ValueError(f"even Zernike terms need the optics group's box {box_size}, got an image of {size}")
            gamma = oa.zernike_phase_fftw_half(
                even, oa.even_index_to_mn, size, float(oa._optics_value(row, "rlnImagePixelSize")), box_size, mag
            )
        geometry[key] = (gamma, mag)
    return geometry[key]


def _ctf_particle_table(cache) -> dict:
    """Per particle of the source STAR, the scalars its CTF rows need; built once with the parsed tables.

    ``groups`` (optics group), ``constants`` (:func:`_relion_ctf_constants`; a tomo row has no B-factor: it
    is damped by its dose, as RELION does for dose >= 0), ``dose`` (``(N, 2)``, tomo only),
    ``premultiplied``, and ``pixel_size`` per optics group.
    """

    table = cache.get("particle_table")
    if table is None:
        groups = np.asarray(star_column(cache["particles"], "rlnOpticsGroup", required=True), dtype=np.int64)
        params = _relion_ctf_batch_params(cache, np.arange(groups.size))
        columns = params[:, [0, 1, 2, 3, 4, 5, 6, 9, 8]]  # ..., Q0, Bfac, scale, phase shift
        dose = None
        if cache["tomo"]:
            columns[:, 6] = 0.0
            dose = np.stack(
                [
                    _relion_ctf_column(cache, "rlnMicrographPreExposure", None),
                    _relion_ctf_column(cache, "rlnCtfBfactorPerElectronDose", 0.0),
                ],
                axis=1,
            )
        pixel_size = {}
        for group in np.unique(groups):
            pixel = np.unique(params[groups == group, 7])
            if pixel.size != 1:
                raise ValueError(f"optics group {int(group)} has several pixel sizes")
            pixel_size[int(group)] = float(pixel[0])
        table = {
            "groups": groups,
            "constants": _relion_ctf_constants(columns),
            "dose": dose,
            "premultiplied": _premultiplied_particles(cache),
            "pixel_size": pixel_size,
        }
        cache["particle_table"] = table
    return table


def _exact_ctf_rows(
    cache,
    original_indices,
    image_h: int,
    image_w: int,
    pixel_indices=None,
    *,
    centered_rows=True,
    square_premultiplied=True,
):
    """The particles' RELION ``Fctf`` at ``pixel_indices``, evaluated now on the default JAX device, ``(n, P)``.

    Row ``i`` is ``original_indices[i]``'s (order and duplicates kept). Per optics group: its pixel size,
    even Zernike gamma offset and anisotropic magnification (ml_optimiser.cpp:6461-6484); a tomo row is
    damped by its tilt's dose; a CTF-premultiplied image gets ``CTF^2`` when ``square_premultiplied`` (the
    scoring CTF; RELION's start-up bootstrap keeps the plain CTF). With ``centered_rows`` the pixels and the
    result are in RECOVAR's frame, centred rows and the opposite sign; otherwise RELION's FFTW frame and sign.

    RELION evaluates each particle's CTF in every expectation and keeps it only for that particle's passes
    (``getFourierTransformsAndCtfs``); nothing here outlives the call either. A cross-iteration row cache
    held every particle's full float64 row, 73 GiB at EMPIAR-10202's box 800 (relax#39).
    """

    if image_h != image_w:
        raise ValueError("RELION's CTF rows need square images")
    original_indices = np.asarray(original_indices, dtype=np.int64)
    table = _ctf_particle_table(cache)
    groups = table["groups"][original_indices]
    constants = table["constants"][original_indices]
    dose = None if table["dose"] is None else table["dose"][original_indices]
    square = table["premultiplied"][original_indices] if square_premultiplied else None
    if square is not None and not square.any():
        square = None

    def evaluate(group, members):
        gamma, mag = _optics_group_ctf_geometry(cache, int(group), image_h)
        return _relion_ctf_rows(
            constants[members],
            image_h,
            float(image_h) * table["pixel_size"][int(group)],
            pixels=pixel_indices,
            mag=mag,
            offset=gamma,
            dose=None if dose is None else dose[members],
            square=None if square is None else square[members],
            centered_rows=centered_rows,
        )

    present = np.unique(groups)
    if present.size == 1:
        return evaluate(present[0], slice(None))
    n_pixels = image_h * (image_w // 2 + 1) if pixel_indices is None else int(np.asarray(pixel_indices).size)
    rows = jnp.zeros((original_indices.size, n_pixels), dtype=jnp.float64)
    for group in present:
        members = np.flatnonzero(groups == group)
        rows = rows.at[jnp.asarray(members)].set(evaluate(group, members))
    return rows


def relion_fftw_ctf_rows(experiment_dataset, image_indices, image_shape, *, square_premultiplied: bool = True):
    """These images' RELION ``Fctf`` rows, ``[n, N, N // 2 + 1]`` float64 host array on the FFTW half grid in
    RELION's sign (:func:`_exact_ctf_rows`); ``square_premultiplied=False`` keeps premultiplied groups' plain
    CTF, as RELION's start-up bootstrap uses it (ml_optimiser.cpp:3036-3051).
    """

    image_h, image_w = (int(v) for v in image_shape)
    _, cache = exact_ctf_source_cache(experiment_dataset, (image_h, image_w))
    original = original_image_indices(experiment_dataset, np.asarray(image_indices, dtype=np.int64))
    rows = _exact_ctf_rows(
        cache, original, image_h, image_w, centered_rows=False, square_premultiplied=square_premultiplied
    )
    return np.array(rows).reshape(-1, image_h, image_w // 2 + 1)


def _checked_pixel_indices(pixel_indices, image_shape):
    image_h, image_w = (int(size) for size in image_shape)
    if image_h != image_w:
        raise ValueError("exact RELION CTF replay currently requires square images")
    if pixel_indices is None:
        return None
    pixel_indices = np.asarray(pixel_indices)
    if pixel_indices.ndim != 1 or pixel_indices.dtype.kind not in "iu":
        raise ValueError("CTF pixel indices must be a one-dimensional integer array")
    if np.any(pixel_indices < 0) or np.any(pixel_indices >= image_h * (image_w // 2 + 1)):
        raise ValueError("CTF pixel indices are outside the full half-spectrum")
    return pixel_indices


def relion_exact_ctf_half_from_source_star(experiment_dataset, image_indices, image_shape, *, pixel_indices=None):
    """These images' source-precision RELION CTF rows on the JAX device, ``(n, P)`` float64.

    RECOVAR's centred-y half-spectrum coordinates and sign; the source STAR is mandatory because the ordinary
    dataset metadata has already been rounded to float32. Optional ``pixel_indices`` select the columns (order
    and duplicates kept), and only those are evaluated. Evaluated now by :func:`_exact_ctf_rows`.
    """

    pixel_indices = _checked_pixel_indices(pixel_indices, image_shape)
    _, cache = exact_ctf_source_cache(experiment_dataset, image_shape)
    original = original_image_indices(experiment_dataset, np.asarray(image_indices, dtype=np.int64))
    image_h, image_w = (int(size) for size in image_shape)
    return _exact_ctf_rows(cache, original, image_h, image_w, pixel_indices)


def relion_exact_ctf_half_from_source_star_host(experiment_dataset, image_indices, image_shape, *, pixel_indices=None):
    """:func:`relion_exact_ctf_half_from_source_star` as a host array the caller owns, for callers that place
    the operand themselves (padding on the image axis). The pixel indices must be a host array already: a
    device array here would be read back at every call."""

    if pixel_indices is not None and not isinstance(pixel_indices, np.ndarray):
        raise TypeError("CTF pixel indices must already be a host NumPy array")
    return np.array(
        relion_exact_ctf_half_from_source_star(
            experiment_dataset, image_indices, image_shape, pixel_indices=pixel_indices
        )
    )
