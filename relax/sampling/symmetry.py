"""RELION-compatible non-helical rotational point groups.

RELION describes an equivalent orientation as ``E' = L E R``.  Proper
rotational point groups currently have identity ``L`` operators, but this
module deliberately preserves both matrices so the convention remains
explicit and can be checked against RELION's source implementation.

Only proper rotations are public here: cyclic, dihedral, tetrahedral,
octahedral, and RELION's implemented icosahedral axis conventions.  Mirror,
inversion, and roto-reflection groups are point groups too, but they are not
rotational symmetries and require separate handedness tests before use in the
refinement engine.
"""

from __future__ import annotations

import functools
import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np

_CYCLIC = re.compile(r"C([0-9]{1,2})", re.IGNORECASE)
_DIHEDRAL = re.compile(r"D([0-9]{1,2})", re.IGNORECASE)
_ICOSAHEDRAL = {"I", "I1", "I2", "I3", "I4"}
_IMPROPER_PREFIXES = (
    "CI",
    "CS",
    "S",
)


@dataclass(frozen=True)
class RelionRotationalSymmetry:
    """Canonical description of one proper RELION point group."""

    label: str
    family: str
    order: int | None
    operator_count: int


def parse_rotational_symmetry(label: str) -> RelionRotationalSymmetry:
    """Validate and canonicalize a proper rotational RELION symmetry label.

    Supported labels are ``C1``--``C99``, ``D1``--``D99``, ``T``, ``O``,
    and ``I``/``I1``--``I4``.  RELION treats ``I`` as an alias for ``I2``;
    the canonical label returned here is therefore ``I2``.
    """

    if not isinstance(label, str) or not label.strip():
        raise ValueError("symmetry must be a nonempty RELION point-group label")
    normalized = label.strip().upper()

    cyclic = _CYCLIC.fullmatch(normalized)
    if cyclic is not None:
        order = int(cyclic.group(1))
        if not 1 <= order <= 99:
            raise ValueError(f"cyclic symmetry order must be in [1, 99], got {order}")
        return RelionRotationalSymmetry(f"C{order}", "cyclic", order, order)

    dihedral = _DIHEDRAL.fullmatch(normalized)
    if dihedral is not None:
        order = int(dihedral.group(1))
        if not 1 <= order <= 99:
            raise ValueError(f"dihedral symmetry order must be in [1, 99], got {order}")
        return RelionRotationalSymmetry(f"D{order}", "dihedral", order, 2 * order)

    if normalized == "T":
        return RelionRotationalSymmetry("T", "tetrahedral", None, 12)
    if normalized == "O":
        return RelionRotationalSymmetry("O", "octahedral", None, 24)
    if normalized in _ICOSAHEDRAL:
        canonical = "I2" if normalized == "I" else normalized
        return RelionRotationalSymmetry(canonical, "icosahedral", None, 60)

    if normalized in {"I5", "I5H"}:
        raise ValueError(f"RELION recognizes {normalized} but does not implement it")
    if (
        normalized.startswith(_IMPROPER_PREFIXES)
        or normalized.endswith(("H", "V"))
        or normalized in {"TD", "TH", "OH", "IH", "I1H", "I2H", "I3H", "I4H"}
    ):
        raise ValueError(
            f"{normalized} contains mirror, inversion, or roto-reflection operators; "
            "it is not a proper rotational symmetry"
        )
    raise ValueError(f"unsupported RELION rotational symmetry label: {label!r}")


def canonicalize_rotational_symmetry(label: str) -> str:
    """Return the canonical RELION label for a proper rotational group."""

    return parse_rotational_symmetry(label).label


def is_identity_symmetry(label: str) -> bool:
    """Return whether ``label`` is the trivial group C1."""

    return canonicalize_rotational_symmetry(label) == "C1"


# RELION symmetries.h point-group codes and SymList::fill_symmetry_class
# generator axes (symmetries.cpp:606-767), written as RELION writes them.
_PG_CN, _PG_DN, _PG_T, _PG_O, _PG_I, _PG_I1, _PG_I2, _PG_I3, _PG_I4 = 202, 206, 209, 212, 214, 216, 217, 218, 219
_ICOSAHEDRAL_GENERATORS = {
    "I2": ((2, (0.0, 0.0, 1.0)), (5, (0.525731114, 0.0, 0.850650807)), (3, (0.0, 0.356822076, 0.934172364))),
    "I1": (
        (2, (1.0, 0.0, 0.0)),
        (5, (0.85065080702670, 0.0, -0.5257311142635)),
        (3, (0.9341723640, 0.3568220765, 0.0)),
    ),
    "I3": (
        (2, (-0.5257311143, 0.0, 0.8506508070)),
        (5, (0.0, 0.0, 1.0)),
        (3, (-0.4911234778630044, 0.3568220764705179, 0.7946544753759428)),
    ),
    "I4": (
        (2, (0.5257311143, 0.0, 0.8506508070)),
        (5, (0.8944271932547096, 0.0, 0.4472135909903704)),
        (3, (0.4911234778630044, 0.3568220764705179, 0.7946544753759428)),
    ),
}
_ICOSAHEDRAL_CODES = {"I2": _PG_I2, "I1": _PG_I1, "I3": _PG_I3, "I4": _PG_I4}
# XMIPP_EQUAL_ACCURACY for RELION's double-precision CPU build (macros.h:115).
_EQUAL_ACCURACY = 1e-6


def _point_group_and_generators(parsed: RelionRotationalSymmetry):
    """``SymList::isSymmetryGroup`` code and order plus the ``rot_axis`` generators."""

    if parsed.family == "cyclic":
        return _PG_CN, parsed.order, ((parsed.order, (0.0, 0.0, 1.0)),)
    if parsed.family == "dihedral":
        generators = ((2, (1.0, 0.0, 0.0)),)
        if parsed.order > 1:
            generators = ((parsed.order, (0.0, 0.0, 1.0)),) + generators
        return _PG_DN, parsed.order, generators
    if parsed.family == "tetrahedral":
        return _PG_T, -1, ((3, (0.0, 0.0, 1.0)), (2, (0.0, 0.816496, 0.577350)))
    if parsed.family == "octahedral":
        return _PG_O, -1, ((3, (0.5773502, 0.5773502, 0.5773502)), (4, (0.0, 0.0, 1.0)))
    return _ICOSAHEDRAL_CODES[parsed.label], -1, _ICOSAHEDRAL_GENERATORS[parsed.label]


def _matmul4(a, b):
    """``Matrix2D::operator*``: each entry accumulates from zero over k in order."""

    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            total = 0.0
            for k in range(4):
                total += a[i][k] * b[k][j]
            out[i][j] = total
    return out


def _transpose4(a):
    return [[a[j][i] for j in range(4)] for i in range(4)]


def _set_small_values_to_zero(a):
    return [[0.0 if abs(value) < _EQUAL_ACCURACY else value for value in row] for row in a]


def _align_with_z(axis):
    """``alignWithZ`` (transformations.cpp:137-179), homogeneous 4x4."""

    module = math.sqrt(0.0 + axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if abs(module) > _EQUAL_ACCURACY:
        inverse = 1.0 / module
        x, y, z = axis[0] * inverse, axis[1] * inverse, axis[2] * inverse
    else:
        x = y = z = 0.0
    result = [[0.0] * 4 for _ in range(4)]
    result[3][3] = 1.0
    proj_mod = math.sqrt(y * y + z * z)
    if proj_mod > _EQUAL_ACCURACY:
        result[0][0] = proj_mod
        result[0][1] = -x * y / proj_mod
        result[0][2] = -x * z / proj_mod
        result[1][1] = z / proj_mod
        result[1][2] = -y / proj_mod
        result[2][0], result[2][1], result[2][2] = x, y, z
    else:
        result[0][2] = -1.0 if x > 0 else 1.0
        result[1][1] = 1.0
        result[2][0] = 1.0 if x > 0 else -1.0
    return result


def _rotation_about_axis(angle_deg, axis):
    """``rotation3DMatrix(ang, axis, R)``: ``A^T Rz A`` (transformations.cpp:184-193)."""

    angle = angle_deg * math.pi / 180
    cosine, sine = math.cos(angle), math.sin(angle)
    rz = [[cosine, -sine, 0.0, 0.0], [sine, cosine, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    align = _align_with_z(axis)
    return _matmul4(_matmul4(_transpose4(align), rz), align)


def _is_identity(a, size):
    return all(abs(a[i][j] - (1.0 if i == j else 0.0)) <= _EQUAL_ACCURACY for i in range(size) for j in range(size))


def _equal(a, b):
    return all(abs(a[i][j] - b[i][j]) <= _EQUAL_ACCURACY for i in range(4) for j in range(4))


def _found_not_tried(tried, size):
    """``found_not_tried`` (symmetries.cpp:243-270): the next untried ``(i, j)`` pair."""

    i = j = n = 0
    while n != size:
        if not tried[i][j]:
            return i, j
        if i != n:
            i += 1
        else:
            j -= 1
            if j == -1:
                n += 1
                j = n
                i = 0
    return None


def _symlist_matrices(generators):
    """``SymList::read_sym_file`` for ``rot_axis`` lines then ``compute_subgroup``.

    Returns RELION's ordered ``(L, R)`` 4x4 list without the implicit identity.
    """

    identity = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    left, right, chain = [], [], []
    for fold, axis in generators:
        increment = 360.0 / fold
        angle = increment
        for _ in range(1, fold):
            rotation = _set_small_values_to_zero(_rotation_about_axis(angle, axis))
            left.append(identity)
            right.append(_transpose4(rotation))
            chain.append(1)
            angle += increment
    tried = [[0] * len(left) for _ in range(len(left))]
    while (pair := _found_not_tried(tried, len(tried))) is not None:
        i, j = pair
        tried[i][j] = 1
        new_left = _matmul4(left[i], left[j])
        new_right = _matmul4(right[i], right[j])
        if _is_identity(new_left, 4) and _is_identity(new_right, 3):
            continue
        if any(_equal(new_left, left[m]) and _equal(new_right, right[m]) for m in range(len(left))):
            continue
        left.append(_set_small_values_to_zero(new_left))
        right.append(_set_small_values_to_zero(new_right))
        chain.append(chain[i] + chain[j])
        for row in tried:
            row.append(0)
        tried.append([0] * (len(tried) + 1))
    return left, right


@functools.lru_cache(maxsize=None)
def _operators_float64(canonical_label: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """RELION's ordered ``SymList`` operators with the identity prepended.

    Ports ``SymList::isSymmetryGroup``, ``fill_symmetry_class``, ``read_sym_file``
    and ``compute_subgroup`` (RELION 5.0.1 symmetries.cpp) for the proper groups;
    C1 is the identity alone, as RELION never builds a list for it.
    """

    parsed = parse_rotational_symmetry(canonical_label)
    point_group, point_group_order, generators = _point_group_and_generators(parsed)
    left = [np.eye(3)]
    right = [np.eye(3)]
    if parsed.label != "C1":
        sym_left, sym_right = _symlist_matrices(generators)
        left += [np.asarray(matrix, dtype=np.float64)[:3, :3] for matrix in sym_left]
        right += [np.asarray(matrix, dtype=np.float64)[:3, :3] for matrix in sym_right]
    left = np.stack(left)
    right = np.stack(right)
    if left.shape[0] != parsed.operator_count:
        raise RuntimeError(
            f"RELION {canonical_label} closure gave {left.shape[0]} operators; expected {parsed.operator_count}"
        )
    return left, right, point_group, point_group_order


def relion_symmetry_operators(
    label: str,
    *,
    dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered ``(L, R)`` operators, with identity first.

    The matrices follow RELION's ``SymList`` construction (:func:`_operators_float64`).
    Copies are returned so callers cannot mutate the cached source arrays.
    """

    canonical = canonicalize_rotational_symmetry(label)
    left, right, _, _ = _operators_float64(canonical)
    target = np.dtype(dtype)
    return left.astype(target, copy=True), right.astype(target, copy=True)


def relion_point_group_code(label: str) -> tuple[int, int]:
    """Return RELION's integer ``(point_group, order)`` pair."""

    canonical = canonicalize_rotational_symmetry(label)
    _, _, point_group, point_group_order = _operators_float64(canonical)
    return point_group, point_group_order


def rotational_operators(label: str, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    """Return the proper 3-D rotation operators used for reconstruction."""

    left, right = relion_symmetry_operators(label, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(left, identity[None, :, :], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"RELION {label} unexpectedly contains non-identity left operators")
    determinants = np.linalg.det(right)
    if not np.allclose(determinants, 1.0, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"RELION {label} contains an improper reconstruction operator")
    return right.astype(np.dtype(dtype), copy=True)


def symmetry_operator_sha256(label: str, *, decimals: int = 12) -> str:
    """Hash the ordered RELION operator convention deterministically."""

    left, right = relion_symmetry_operators(label, dtype=np.float64)
    stacked = np.stack([left, right], axis=0)
    rounded = np.round(stacked, decimals=decimals)
    rounded[np.abs(rounded) < 0.5 * 10.0 ** (-decimals)] = 0.0
    canonical = np.ascontiguousarray(rounded.astype("<f8", copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


__all__ = [
    "RelionRotationalSymmetry",
    "canonicalize_rotational_symmetry",
    "is_identity_symmetry",
    "parse_rotational_symmetry",
    "relion_point_group_code",
    "relion_symmetry_operators",
    "rotational_operators",
    "symmetry_operator_sha256",
]
