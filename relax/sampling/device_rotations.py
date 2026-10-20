"""RELION's device-built scoring rotations.

The pass-1 rotation matrices RELION builds on the GPU from Euler rows and the sampling perturbation
(``AccProjectorPlan::setup``), their record :class:`DevicePass1Source`, and the provenance check that rows handed to
a projection are the rows their stated source builds (:func:`require_same_rows`, :func:`project_rows`).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.sampling import healpix


def _relion_device_scoring_rotations_f32(
    eulers_deg: np.ndarray,
    right_matrix: np.ndarray | None = None,
) -> np.ndarray | None:
    """Reproduce RELION's CUDA ``make_eulers_3D`` arithmetic.

    RELION's adaptive pass-1 ``AccProjectorPlan`` builds scorer matrices on
    the device. Fine scoring and weighted-sum backprojection instead use the
    host ``generateEulerMatrices(..., inverse=true)`` path represented by
    :func:`relion_mstep_rotations_from_eulers`. Callers must therefore keep
    coarse scorer matrices separate from fine/M-step matrices.

    Return ``None`` on CPU so CPU-only tools retain the existing NumPy path.
    A GPU RELION-parity run is deliberately fail-closed if the custom CUDA
    implementation is unavailable; silently using the nearby NumPy result
    would make an exact-parity run claim arithmetic it did not execute.
    """

    if jax.default_backend() != "gpu":
        return None

    from relax.cuda import rotations as cuda_rotations

    eulers_f32 = np.asarray(eulers_deg, dtype=np.float32).reshape(-1, 3)
    do_right = right_matrix is not None
    if right_matrix is None:
        right_f32 = np.eye(3, dtype=np.float32)
    else:
        right_f32 = np.asarray(right_matrix, dtype=np.float32)
        if right_f32.shape != (3, 3):
            raise ValueError(f"right_matrix must have shape (3, 3), got {right_f32.shape}")
    rotations = cuda_rotations.relion_make_scoring_rotations_f32(
        jnp.asarray(eulers_f32),
        jnp.asarray(right_f32),
        do_right=do_right,
    )
    return np.asarray(jax.device_get(rotations), dtype=np.float32)


def _relion_device_scoring_rotations_f64(
    eulers_deg: np.ndarray,
    right_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce RELION's ``ACC_DOUBLE_PRECISION`` ``make_eulers_3D`` arithmetic.

    Under ``ACC_DOUBLE_PRECISION``, ``AccProjectorPlan::setup`` builds these
    coarse-scorer matrices with ``XFLOAT=double`` throughout
    (``acc_projector_plan_impl.h``: the ``RFLOAT`` euler angles from
    ``getOrientations`` are copied straight into the ``AccPtr<XFLOAT>``
    workspace with no float32 cast, and ``acc_make_eulers_3D`` runs its
    ``sincos``/matrix-construction arithmetic in that same ``XFLOAT``). On a
    GPU this must execute in CUDA: device ``sincos`` and multiply/add ordering
    can differ by the last bits from NumPy/libm and BLAS on the host. CPU-only
    callers retain the NumPy equivalent as a portability fallback.
    """

    eulers_f64 = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    do_right = right_matrix is not None
    if right_matrix is None:
        right_f64 = np.eye(3, dtype=np.float64)
    else:
        right_f64 = np.asarray(right_matrix, dtype=np.float64)
        if right_f64.shape != (3, 3):
            raise ValueError(f"right_matrix must have shape (3, 3), got {right_f64.shape}")

    if jax.default_backend() == "gpu":
        from relax.cuda import rotations as cuda_rotations

        rotations = cuda_rotations.relion_make_scoring_rotations_f64(
            jnp.asarray(eulers_f64),
            jnp.asarray(right_f64),
            do_right=do_right,
        )
        return np.asarray(jax.device_get(rotations), dtype=np.float64)

    a = healpix.euler_angles_to_matrix(eulers_f64)
    return a @ right_f64 if do_right else a


def relion_adaptive_pass1_rotations(
    source_eulers_deg: np.ndarray,
    random_perturbation: float,
    angular_sampling_deg: float,
    *,
    use_float64: bool = False,
    left_matrices: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build exact RELION matrices for adaptive coarse scoring only.

    ``AccProjectorPlan::setup`` sends the unperturbed Euler rows and, when
    active, a host-generated right perturbation matrix to
    ``acc_make_eulers_3D``. Under RELION's default single-precision ACC
    build this is ``XFLOAT=float`` and differs by a few float32 ulps from the
    host inverse matrices used by RELION's fine and weighted-sum paths;
    ``use_float64=True`` instead reproduces ``ACC_DOUBLE_PRECISION``, where
    this construction stays double throughout (see
    :func:`_relion_device_scoring_rotations_f64`). On CPU, the float32 path
    returns ``None`` so callers retain the existing host implementation; the
    float64 path runs the corresponding CUDA specialization on GPU and uses
    an equivalent NumPy fallback only for CPU callers.
    """

    right_matrix = pass1_right_matrix(random_perturbation, angular_sampling_deg)
    if left_matrices is not None:
        return _relion_device_scoring_rotations_left_f32(source_eulers_deg, right_matrix, left_matrices)
    if use_float64:
        return _relion_device_scoring_rotations_f64(source_eulers_deg, right_matrix)
    return _relion_device_scoring_rotations_f32(source_eulers_deg, right_matrix)


def pass1_right_matrix(random_perturbation: float, angular_sampling_deg: float) -> np.ndarray | None:
    """RELION's host perturbation matrix of the pass-1 rows (float64 ``[3, 3]``), or None when it is not applied."""

    if abs(float(random_perturbation)) < 1e-12:
        return None
    perturbation_deg = float(random_perturbation) * float(angular_sampling_deg)
    return healpix.euler_angles_to_matrix(
        np.asarray([[perturbation_deg, perturbation_deg, perturbation_deg]], dtype=np.float64)
    )[0]


def pass1_rotations_left_f32(eulers_f32, right_f32, left_matrices) -> jax.Array:
    """:func:`relion_adaptive_pass1_rotations` with left matrices, traceable: the float32 Euler rows ``[N, 3]``, the
    float32 right matrix (``pass1_right_matrix`` cast to float32, or None) and the left matrices ``[B, 3, 3]``.

    A program that builds rows of every iteration's perturbation takes it as this array, not as a static float
    (one compile per value, relax#115). The kernel sees the same float32 values either way. CUDA only.
    """

    from relax.cuda import rotations as cuda_rotations

    return cuda_rotations.relion_make_scoring_rotations_left_f32(
        jnp.asarray(eulers_f32, dtype=jnp.float32).reshape(-1, 3),
        jnp.eye(3, dtype=jnp.float32) if right_f32 is None else jnp.asarray(right_f32, dtype=jnp.float32),
        jnp.asarray(left_matrices, dtype=jnp.float32).reshape(-1, 3, 3),
        do_right=right_f32 is not None,
    )


class DevicePass1Source(NamedTuple):
    """The inputs of RELION's device-built pass-1 rows (:func:`relion_adaptive_pass1_rotations`)."""

    source_eulers_deg: np.ndarray
    random_perturbation: float
    angular_sampling_deg: float
    use_float64: bool


class RotationProvenanceError(RuntimeError):
    """Rows handed to a projection are not the rows their stated RELION source builds."""


# A rebuilt row must reproduce the row it replaces to float32 rounding (about 1e-7); a wrong
# index or source is off by O(1e-2) or more.
_PROVENANCE_ATOL = 1e-5


# The check compares an evenly strided sample of at most this many rows: a wrong source, perturbation
# or row order shows on any row, and the check stays a few milliseconds on a 10M-row local layout.
_PROVENANCE_SAMPLE_ROWS = 65536


def require_same_rows(rebuilt, rows, what: str) -> None:
    """Raise :class:`RotationProvenanceError` unless ``rebuilt`` reproduces ``rows`` (one check per grid build)."""

    if np.shape(rebuilt) != np.shape(rows):
        raise RotationProvenanceError(f"{what}: rebuilt {np.shape(rebuilt)} rows for {np.shape(rows)}")
    stride = max(1, np.shape(rows)[0] // _PROVENANCE_SAMPLE_ROWS) if np.ndim(rows) else 1
    sample = np.abs(np.asarray(rebuilt[::stride], dtype=np.float64) - np.asarray(rows[::stride], dtype=np.float64))
    error = float(np.max(sample)) if sample.size else 0.0
    if not error <= _PROVENANCE_ATOL:
        raise RotationProvenanceError(f"{what}: rebuilt rows differ from the handed rows by {error:.3g}")


def relion_device_projection_rotations(source: DevicePass1Source, rows, scale: float, magnification=None):
    """Pass-1 rows of images on another grid or magnified, as RELION's device builds them.

    RELION's pass-1 plan hands ``MBL = s inv(M3)`` to ``make_eulers_3D``, which forms ``L (A R)`` in
    float32 from float Euler angles and inverts it with the float32 adjugate, since with a left matrix
    the inverse is not the transpose (acc_projector_plan_impl.h:151-330, helper.cuh). The tilt images'
    ``MBL`` go through the same kernel (:func:`_relion_device_scoring_rotations_left_f32`). ``rows`` are
    the unprojected rows built from ``source``; they are checked against it. Under ACC double
    precision the device chain stays double, which the host rule reproduces
    (:func:`relax.relion.optics_aberrations.projection_rotations`). Returns float32 rows (float64
    for the double source).
    """

    from relax.relion.optics_aberrations import (
        projection_rotations,
        relion_projection_left_matrix,
        relion_projection_optics,
    )

    scale, magnification = relion_projection_optics(scale, magnification)
    if scale == 1.0 and magnification is None:
        return rows
    if source.use_float64:
        rows64 = np.asarray(rows, dtype=np.float64)
        return projection_rotations(rows64, scale, magnification, dtype=np.float64)
    plain = relion_adaptive_pass1_rotations(
        source.source_eulers_deg, source.random_perturbation, source.angular_sampling_deg
    )
    require_same_rows(plain, rows, "device pass-1 rows")
    left = relion_projection_left_matrix(scale, magnification)
    return np.asarray(
        relion_adaptive_pass1_rotations(
            source.source_eulers_deg,
            source.random_perturbation,
            source.angular_sampling_deg,
            left_matrices=left[None],
        )
    )[0]


def project_rows(rows, scale: float, magnification=None, *, device_source=None, host_rows=None, what: str = "rows"):
    """Rows of images on another grid or magnified, by the RELION path that builds them.

    ``device_source`` (:class:`DevicePass1Source`): device-built pass-1 rows
    (:func:`relion_device_projection_rotations`). ``host_rows``: a callable returning the rows'
    float64 matrices, which must reproduce ``rows`` (checked once per call); they are composed in
    float64 and cast to the rows' dtype once (:func:`relax.relion.optics_aberrations.projection_rotations`).
    Exactly one is given. Unprojected images keep ``rows``.
    """

    from relax.relion.optics_aberrations import projection_rotations, relion_projection_optics

    scale, magnification = relion_projection_optics(scale, magnification)
    if rows is None or (scale == 1.0 and magnification is None):
        return rows
    if (device_source is None) == (host_rows is None):
        raise ValueError("projected rows need exactly one of a device source and host rows")
    if device_source is not None:
        return relion_device_projection_rotations(device_source, rows, scale, magnification)
    rebuilt = np.asarray(host_rows(), dtype=np.float64)
    require_same_rows(rebuilt, rows, what)
    return projection_rotations(rebuilt, scale, magnification, dtype=np.asarray(rows).dtype)


def _relion_device_scoring_rotations_left_f32(eulers_deg, right_matrix, left_matrices) -> jax.Array | None:
    """Tilt images' coarse scorer matrices, ``[B, N, 3, 3]``: ``make_eulers_3D`` with one left matrix per image.

    RELION's pass-1 plan of a tilt image passes ``MBL`` (its ``Aproj`` times the optics scale)
    to the device as float (acc_ml_optimiser_impl.h:1086-1116); with a left matrix the inverse
    is the float32 adjugate, not the transpose. ``None`` on CPU, like the SPA path. The float32
    matrices stay on the device (the coarse pass consumes them there; no host round trip).
    """

    if jax.default_backend() != "gpu":
        return None

    from relax.cuda import rotations as cuda_rotations

    # A device array of Euler rows stays on the device (the exact cut gathers its rows there).
    eulers_f32 = (
        eulers_deg.astype(jnp.float32) if isinstance(eulers_deg, jax.Array) else np.asarray(eulers_deg, dtype=np.float32)
    ).reshape(-1, 3)
    right_f32 = np.eye(3, dtype=np.float32) if right_matrix is None else np.asarray(right_matrix, dtype=np.float32)
    rotations = cuda_rotations.relion_make_scoring_rotations_left_f32(
        jnp.asarray(eulers_f32),
        jnp.asarray(right_f32),
        (
            left_matrices.astype(jnp.float32)
            if isinstance(left_matrices, jax.Array)
            else jnp.asarray(np.asarray(left_matrices, dtype=np.float32))
        ).reshape(-1, 3, 3),
        do_right=right_matrix is not None,
    )
    return rotations
