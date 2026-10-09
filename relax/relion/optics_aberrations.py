"""RELION optics-group aberrations: beam tilt and odd Zernike phase (``ObservationModel``).

RELION 5.0.1 turns an optics group's ``rlnBeamTiltX/Y`` into odd Zernike
coefficients (``TiltHelper::insertTilt``, tilt_helper.cpp:557-574) and adds them to
``rlnOddZernike`` (obs_model.cpp:150-172). ``ObservationModel::getPhaseCorrection``
(obs_model.cpp:1215-1261) evaluates their phase on the image's FFTW half grid, and
``demodulatePhase`` (obs_model.cpp:598-626) multiplies every particle image by its
complex conjugate before scoring and backprojection (ml_optimiser.cpp:6285, 6418;
acc_ml_optimiser_impl.h:538, 859). The references are not modulated.

The phase is a property of the image pixels, so relax applies it where the images
are Fourier transformed (:func:`relax.helpers.preprocessing.process_half_image`),
which every scoring and backprojection path reads.
"""

from __future__ import annotations

import math
import re

import numpy as np
from recovar.data_io.starfile import star_column

from relax.helpers.batch_fetch import original_image_indices

# The phase tables of one source STAR and image shape: {(source, shape): {group: row or None}}.
_ODD_PHASE_CACHE: dict = {}


def relion_wavelength(voltage_kv: float) -> float:
    """Electron wavelength in Angstrom, as ObservationModel computes it (obs_model.cpp:144-146)."""

    volts = float(voltage_kv) * 1e3
    return 12.2643247 / math.sqrt(volts * (1.0 + volts * 0.978466e-6))


def parse_relion_vector(value) -> list[float]:
    """A RELION vector label (``[a,b,c]``) as floats; empty for a missing or empty value."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return [float(token) for token in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(value))]


def insert_beam_tilt(odd_coefficients, tilt_x, tilt_y, cs_mm, wavelength) -> list[float]:
    """``TiltHelper::insertTilt`` (tilt_helper.cpp:557-574): beam tilt (mrad) as odd Zernike terms."""

    coefficients = list(float(c) for c in odd_coefficients)
    if len(coefficients) <= 5:
        coefficients += [0.0] * (5 - len(coefficients))
    scale = float(cs_mm) * 20000 * float(wavelength) * float(wavelength) * 3.141592654
    z3x = -scale * float(tilt_x) / 3.0
    z3y = -scale * float(tilt_y) / 3.0
    coefficients[1] += 2.0 * z3x
    coefficients[0] += 2.0 * z3y
    coefficients[4] += z3x
    coefficients[3] += z3y
    return coefficients


def odd_index_to_mn(index: int) -> tuple[int, int]:
    """``Zernike::oddIndexToMN`` (Zernike.cpp)."""

    k = int((math.sqrt(1 + 4 * index) - 1.0) / 2.0)
    i0 = k * k + k
    n = 2 * k + 1
    return 2 * (index - i0) - n, n


def even_index_to_mn(index: int) -> tuple[int, int]:
    """``Zernike::evenIndexToMN`` (Zernike.cpp)."""

    k = int(math.sqrt(float(index)))
    return 2 * (index - k * k - k), 2 * k


def zernike_radial(m: int, n: int, rho):
    """``Zernike::R``: the radial polynomial, zero when ``n - m`` is odd."""

    if m > n:
        raise ValueError(f"Zernike radial polynomial needs m <= n, got m={m}, n={n}")
    rho = np.asarray(rho, dtype=np.float64)
    if (n - m) % 2 == 1:
        return np.zeros_like(rho)
    out = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coefficient = (1 - 2 * (k % 2)) * math.factorial(n - k)
        coefficient /= math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
        out = out + coefficient * rho ** (n - 2 * k)
    return out


def zernike_cartesian(m: int, n: int, x, y):
    """``Zernike::Z_cart``: ``R(|m|, n, rho)`` times ``cos(m phi)`` (m >= 0) or ``sin(-m phi)``."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho = np.sqrt(x * x + y * y)
    phi = np.where((x == 0) & (y == 0), 0.0, np.arctan2(y, x))
    if m >= 0:
        return zernike_radial(m, n, rho) * np.cos(m * phi)
    return zernike_radial(-m, n, rho) * np.sin(-m * phi)


def zernike_phase_fftw_half(coefficients, index_to_mn, size: int, angpix: float, box_size: int, mag_matrix=None):
    """``sum_i c_i Z_i(M k)`` on RELION's FFTW half grid ``(size, size // 2 + 1)``.

    The grid of ``getPhaseCorrection`` / ``getGammaOffset`` (obs_model.cpp:1215-1307): row
    ``y`` is frequency ``y`` below ``size // 2`` and ``y - size`` from there on, in units of
    ``1 / (angpix * box_size)``, and the anisotropic magnification ``M`` acts on ``(x, y)``.
    """

    half = size // 2 + 1
    extent = float(angpix) * float(box_size)
    xs = np.arange(half, dtype=np.float64) / extent
    rows = np.arange(size)
    ys = np.where(rows < half - 1, rows, rows - size).astype(np.float64) / extent
    x0, y0 = np.meshgrid(xs, ys, indexing="xy")
    matrix = np.eye(2) if mag_matrix is None else np.asarray(mag_matrix, dtype=np.float64)
    xx = matrix[0, 0] * x0 + matrix[0, 1] * y0
    yy = matrix[1, 0] * x0 + matrix[1, 1] * y0
    phase = np.zeros((size, half), dtype=np.float64)
    for index, coefficient in enumerate(coefficients):
        m, n = index_to_mn(index)
        phase += float(coefficient) * zernike_cartesian(m, n, xx, yy)
    return phase


def _optics_value(optics_row, name, default=None):
    for key in (name, f"_{name}"):
        if key in optics_row:
            return optics_row[key]
    if default is None:
        raise KeyError(f"optics group has no {name}")
    return default


def optics_group_mag_matrix(optics_row) -> np.ndarray:
    """``rlnMagMat00..11`` as a 2x2 matrix; a table with some of them keeps identity for the rest."""

    matrix = np.eye(2)
    for (i, j), default in (((0, 0), 1.0), ((0, 1), 0.0), ((1, 0), 0.0), ((1, 1), 1.0)):
        matrix[i, j] = float(_optics_value(optics_row, f"rlnMagMat{i}{j}", default))
    return matrix


def optics_group_odd_coefficients(optics_row, *, has_odd: bool, has_tilt: bool) -> list[float]:
    """An optics group's odd Zernike coefficients after beam-tilt insertion (obs_model.cpp:150-172)."""

    def get(name, default=None):
        for key in (name, f"_{name}"):
            if key in optics_row:
                return optics_row[key]
        return default

    coefficients = parse_relion_vector(get("rlnOddZernike")) if has_odd else []
    if has_tilt:
        if not has_odd:
            coefficients = [0.0] * 6
        coefficients = insert_beam_tilt(
            coefficients,
            float(get("rlnBeamTiltX", 0.0)),
            float(get("rlnBeamTiltY", 0.0)),
            float(get("rlnSphericalAberration")),
            relion_wavelength(float(get("rlnVoltage"))),
        )
    return coefficients


def _source_tables(experiment_dataset, image_shape):
    from relax.relion.relion_ctf import exact_ctf_source_cache

    return exact_ctf_source_cache(experiment_dataset, image_shape)


def _odd_phase_tables(experiment_dataset, image_shape):
    """Per optics group: the demodulation factor ``exp(-i phase)`` in RECOVAR's centered half layout, or None."""

    try:
        source_path, cache = _source_tables(experiment_dataset, image_shape)
    except ValueError:
        return None
    key = (str(source_path), tuple(int(s) for s in image_shape))
    tables = _ODD_PHASE_CACHE.get(key)
    if tables is not None:
        return tables
    optics = cache["optics"]
    labels = {str(label).lstrip("_") for row in optics.values() for label in row.keys()}
    has_odd = "rlnOddZernike" in labels
    has_tilt = "rlnBeamTiltX" in labels or "rlnBeamTiltY" in labels
    has_mag = any(label.startswith("rlnMagMat") for label in labels)
    size = int(image_shape[0])
    if int(image_shape[1]) != size:
        raise ValueError("RELION's odd aberrations need square images")
    tables = {}
    for group, row in optics.items():
        coefficients = optics_group_odd_coefficients(row, has_odd=has_odd, has_tilt=has_tilt)
        if not any(coefficients):
            tables[group] = None
            continue
        box_size = int(_optics_value(row, "rlnImageSize", size))
        angpix = float(_optics_value(row, "rlnImagePixelSize"))
        phase = zernike_phase_fftw_half(
            coefficients, odd_index_to_mn, size, angpix, box_size, optics_group_mag_matrix(row) if has_mag else None
        )
        # RECOVAR's centered half rows are the FFTW rows rolled by size // 2
        # (relion_ctf._evaluate_exact_ctf_rows); the image values are otherwise RELION's.
        tables[group] = np.exp(-1j * np.fft.fftshift(phase, axes=0)).reshape(-1)
    _ODD_PHASE_CACHE[key] = tables
    return tables


def _odd_demodulation_table(experiment_dataset, image_shape):
    """``(table, row_of_group)``: one ``exp(-i phase)`` row per aberrated group plus a row of ones, or None."""

    tables = _odd_phase_tables(experiment_dataset, image_shape)
    if tables is None or all(row is None for row in tables.values()):
        return None
    source_path, _ = _source_tables(experiment_dataset, image_shape)
    key = ("device", str(source_path), tuple(int(s) for s in image_shape))
    cached = _ODD_PHASE_CACHE.get(key)
    if cached is None:
        import jax.numpy as jnp

        n_pixels = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
        groups = sorted(tables)
        rows = [np.ones(n_pixels, dtype=np.complex128)] + [
            tables[g] for g in groups if tables[g] is not None
        ]
        row_of_group = {}
        next_row = 1
        for g in groups:
            if tables[g] is None:
                row_of_group[g] = 0
            else:
                row_of_group[g] = next_row
                next_row += 1
        cached = (jnp.asarray(np.stack(rows)), row_of_group)
        _ODD_PHASE_CACHE[key] = cached
    return cached


def odd_demodulation_rows(experiment_dataset, image_indices, n_rows=None):
    """``exp(-i phase)`` rows ``[n_rows, P]`` of these images (ones on padding rows), or None.

    For programs that demodulate inside a compiled image pass instead of through
    :func:`demodulate_odd_aberrations`.
    """

    from relax.relion.relion_ctf import _relion_exact_ctf_source_star

    try:
        _relion_exact_ctf_source_star(experiment_dataset)
    except ValueError:
        return None
    image_shape = tuple(int(s) for s in experiment_dataset.image_shape)
    found = _odd_demodulation_table(experiment_dataset, image_shape)
    if found is None:
        return None
    import jax.numpy as jnp

    table, row_of_group = found
    _, cache = _source_tables(experiment_dataset, image_shape)
    groups = np.asarray(star_column(cache["particles"], "rlnOpticsGroup", required=True), dtype=np.int64)
    original = np.asarray(
        original_image_indices(experiment_dataset, np.asarray(image_indices, dtype=np.int64)), dtype=np.int64
    )
    rows = np.zeros(int(len(original) if n_rows is None else n_rows), dtype=np.int32)
    rows[: len(original)] = [row_of_group[int(g)] for g in groups[original]]
    return jnp.take(table, jnp.asarray(rows), axis=0)


def demodulate_odd_aberrations(experiment_dataset, processed_half, image_indices):
    """RELION's ``demodulatePhase``: multiply each processed image by ``exp(-i phase)`` of its group.

    ``processed_half`` is ``[B, P]`` in RECOVAR's centered half layout for the
    dataset's image shape; ``image_indices`` are its ``B`` dataset images. The
    product is taken in complex128, as RELION's RFLOAT image, and returned in the
    input dtype. Datasets without odd aberrations return the input unchanged.
    """

    from relax.relion.relion_ctf import _relion_exact_ctf_source_star

    try:
        _relion_exact_ctf_source_star(experiment_dataset)
    except ValueError:
        return processed_half  # no source STAR, so no optics table
    image_shape = tuple(int(s) for s in experiment_dataset.image_shape)
    found = _odd_demodulation_table(experiment_dataset, image_shape)
    if found is None:
        return processed_half
    if image_indices is None:
        raise NotImplementedError(
            "beam tilt / odd Zernike aberrations: this image preprocessing path does not pass its "
            "image indices, so it cannot demodulate the images"
        )
    import jax.numpy as jnp

    processed = jnp.asarray(processed_half)
    if processed.shape[0] != len(image_indices):
        raise ValueError(f"{processed.shape[0]} processed images for {len(image_indices)} image indices")
    factors = odd_demodulation_rows(experiment_dataset, image_indices).reshape(processed.shape)
    return (processed.astype(jnp.complex128) * factors).astype(processed.dtype)


def relax_projection_magnification(mag_matrix) -> np.ndarray:
    """relax's left factor for RELION's ``applyAnisoMag`` (obs_model.cpp:1309-1330).

    RELION projects and backprojects with ``inv(M3) A``, ``M3`` the 2x2 magnification
    embedded in a 3x3 identity (acc_ml_optimiser_impl.h:1098, 1721, 3221). relax's
    projection matrices are ``inv(A)^T`` (``sampling._relion_mstep_rotations_from_eulers``),
    so the magnified ones are ``inv(inv(M3) A)^T = M3^T inv(A)^T``: ``M3^T`` on the left.
    """

    matrix = np.eye(3)
    matrix[:2, :2] = np.asarray(mag_matrix, dtype=np.float64)
    return matrix.T


def dataset_optics_mag_matrices(experiment_dataset) -> dict | None:
    """Each optics group's 2x2 ``rlnMagMat`` among the dataset's own images, or None without magnification.

    Only the groups the dataset's images belong to count: a shape class
    (:func:`relax.refinement.optics_shapes.optics_shape_class_rows`) holds one
    magnification matrix even when the source STAR has several. A dataset without a
    RELION source STAR or without magnification columns has none.
    """

    from recovar.data_io.starfile import star_column

    from relax.helpers.batch_fetch import original_image_indices
    from relax.relion.relion_ctf import _relion_exact_ctf_source_star

    try:
        _relion_exact_ctf_source_star(experiment_dataset)
    except ValueError:
        return None
    _, cache = _source_tables(experiment_dataset, tuple(int(s) for s in experiment_dataset.image_shape))
    labels = {str(label).lstrip("_") for row in cache["optics"].values() for label in row.keys()}
    if not any(label.startswith("rlnMagMat") for label in labels):
        return None
    groups = np.asarray(star_column(cache["particles"], "rlnOpticsGroup", required=True), dtype=np.int64)
    n_units = getattr(experiment_dataset, "n_units", None)
    if n_units is not None:
        groups = groups[original_image_indices(experiment_dataset, np.arange(int(n_units), dtype=np.int64))]
    return {int(group): optics_group_mag_matrix(cache["optics"][int(group)]) for group in np.unique(groups)}


def dataset_projection_magnification(experiment_dataset) -> np.ndarray | None:
    """The dataset's :func:`relax_projection_magnification`, or None without magnification.

    The projection matrices are transformed per scoring call, so the dataset's images
    must share one magnification matrix. Optics groups with different matrices are
    split into shape classes, one matrix each, as groups on other grids are
    (:func:`relax.refinement.optics_shapes.optics_shape_class_rows`, relax#48).
    """

    matrices = dataset_optics_mag_matrices(experiment_dataset)
    if not matrices:
        return None
    first = next(iter(matrices.values()))
    if any(not np.array_equal(m, first) for m in matrices.values()):
        raise ValueError(
            "this dataset's images span optics groups with different magnification matrices "
            f"({sorted(matrices)}); they must be split into shape classes (optics_shape_class_rows)"
        )
    if np.array_equal(first, np.eye(2)):
        return None
    return relax_projection_magnification(first)


def dataset_needs_exact_ctf(experiment_dataset) -> bool:
    """Whether the dataset's CTF is only right in RELION's exact rows (premultiplied, even Zernike, magnification).

    The generic CTF evaluator knows none of these optics-table features.
    """

    from relax.relion.relion_ctf import _relion_exact_ctf_source_star, dataset_has_premultiplied_ctf

    try:
        _relion_exact_ctf_source_star(experiment_dataset)
    except ValueError:
        return False
    image_shape = tuple(int(s) for s in experiment_dataset.image_shape)
    if dataset_has_premultiplied_ctf(experiment_dataset, image_shape):
        return True
    _, cache = _source_tables(experiment_dataset, image_shape)
    for row in cache["optics"].values():
        if any(parse_relion_vector(_optics_value(row, "rlnEvenZernike", "[]"))):
            return True
    return dataset_projection_magnification(experiment_dataset) is not None


def dataset_magnification_is_anisotropic(experiment_dataset) -> bool:
    """Whether any optics group of the dataset has an anisotropic ``rlnMagMat``.

    The M-step adjoint then clips on RELION's rotated radius
    (:func:`relax.helpers.adjoint.magnification_is_anisotropic`, ``ReferenceSphereClip``).
    A dataset without a RELION source STAR or without magnification columns has none.
    """

    from relax.helpers.adjoint import magnification_is_anisotropic

    matrices = dataset_optics_mag_matrices(experiment_dataset)
    return bool(matrices) and magnification_is_anisotropic(matrices.values())


def projection_rotations(rotations, scale: float, magnification=None, *, dtype=np.float32):
    """Host projection matrices for the images' optics (RELION applyAnisoMag, applyScaleDifference).

    The projector samples the reference at image pixel ``k`` times the matrix, so a grid
    ``s`` times coarser in reference voxels per image pixel divides the matrix by ``s``.
    ``magnification`` is the left factor of an anisotropic magnification
    (:func:`relax_projection_magnification`), None without.

    This is RELION's host rule (``generateEulerMatrices``, acc_helper_functions_impl.h: ``L * A * R``
    and its inverse in double, cast to XFLOAT once), used for the fine pass, the weighted sums and
    every host-built row: ``rotations`` are the rows' double-precision matrices, composed here in
    float64 and cast to ``dtype`` once. Composing after a float32 cast moves most magnified matrices
    by an ulp and flips near-tie poses (relax#24). The device-built pass-1 rows follow
    :func:`relax.sampling.relion_device_projection_rotations` instead.
    """

    if rotations is None or (scale == 1.0 and magnification is None):
        return rotations
    rotations = np.asarray(rotations)
    if rotations.dtype != np.float64:
        raise TypeError(f"projected rows are composed from their float64 matrices, got {rotations.dtype}")
    if magnification is not None:
        rotations = np.einsum("ij,njk->nik", np.asarray(magnification, dtype=np.float64), rotations)
    if scale != 1.0:
        rotations = rotations / float(scale)
    return rotations.astype(dtype)


def relion_projection_left_matrix(scale: float, magnification=None) -> np.ndarray:
    """RELION's left matrix of an image grid, ``applyScaleDifference(applyAnisoMag(I))`` = ``s inv(M3)``.

    RELION's pass-1 plan sends it to ``make_eulers_3D`` as ``MBL`` (acc_ml_optimiser_impl.h:1602-1617).
    ``magnification`` is relax's left factor ``M3^T`` (:func:`relax_projection_magnification`).
    """

    m3 = np.eye(3) if magnification is None else np.asarray(magnification, dtype=np.float64).T
    return float(scale) * np.linalg.inv(m3)


def reported_rotations(rotations, scale: float, magnification=None):
    """Undo :func:`projection_rotations` on the engine's best poses: poses are reported unmagnified."""

    rotations = np.asarray(rotations)
    if scale != 1.0:
        rotations = rotations * float(scale)
    if magnification is not None:
        inverse = np.linalg.inv(np.asarray(magnification, dtype=np.float64))
        rotations = np.einsum("ij,njk->nik", inverse, rotations).astype(rotations.dtype)
    return rotations


def expected_accuracy_optics(experiment_dataset, trial_local_indices):
    """The expected-accuracy binding's optics inputs for these trial particles, or None.

    ``trial_ctf``: each trial particle's RELION ``Fctf`` on the FFTW half grid of its
    image (relax's exact CTF rows, :mod:`relax.relion.relion_ctf`: CTF^2 for a
    premultiplied group, even Zernike terms, magnification); ``projection_left``:
    ``applyAnisoMag``'s ``inv(M3)``, or absent. None when the dataset's CTF needs no
    optics table (:func:`dataset_needs_exact_ctf`).
    """

    from relax.relion.relion_ctf import relion_exact_ctf_half_from_source_star_host

    if not dataset_needs_exact_ctf(experiment_dataset):
        return None
    size, width = (int(s) for s in experiment_dataset.image_shape)
    rows = np.asarray(
        relion_exact_ctf_half_from_source_star_host(
            experiment_dataset, np.asarray(trial_local_indices, dtype=np.int64), (size, width)
        ),
        dtype=np.float64,
    ).reshape(-1, size, width // 2 + 1)
    # Undo RECOVAR's frame (centered rows, opposite sign) for RELION's.
    inputs = {"trial_ctf": np.ascontiguousarray(-np.fft.ifftshift(rows, axes=1))}
    magnification = dataset_projection_magnification(experiment_dataset)
    if magnification is not None:
        inputs["projection_left"] = np.linalg.inv(np.asarray(magnification).T)
    return inputs
