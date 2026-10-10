"""RELION's projector-plan matrices in NumPy, independent of relax's builders.

``AccProjectorPlan::setup`` hands the plan's Euler angles (XFLOAT), the perturbation matrix ``R`` and the optics'
left matrix ``L`` to ``cuda_kernel_make_eulers_3D<invert=true, doL, doR>`` (acc_projector_plan_impl.h:158-171,
246-250, 310-392; the kernel is acc/cuda/cuda_kernels/helper.cuh:714-829). This module repeats the kernel's
statements in float32, in its order: degrees to radians, ``sincosf``, the nine products of ``A``, ``B = A R``,
``B = L B``, and the inverse (the adjugate over the determinant with a left matrix, the transpose without).
relax's rows are the transpose of that inverse (its projection convention), so that is what is returned.

A GPU contracts some of these multiply-adds and its ``sincosf`` is not NumPy's, so the device's rows are not
these bit for bit: on an H100 they differ by at most 2 float32 units without a left matrix and 3 with one
(tests/unit/test_projection_rotation_rules.py records the distribution).
"""

import numpy as np

F32 = np.float32


def relion_plan_rows_f32(eulers_deg, right_matrix=None, left_matrix=None) -> np.ndarray:
    """``[N, 3, 3]`` float32 rows of ``eulers_deg`` (rot, tilt, psi), with ``right_matrix`` and ``left_matrix``
    as RELION's ``R`` and ``MBL`` (None: absent, as the kernel's ``doR`` and ``doL`` template arguments)."""

    angles = np.asarray(eulers_deg, dtype=F32).reshape(-1, 3)
    a, b, g = (angles[:, i] * F32(np.pi) / F32(180.0) for i in range(3))
    sa, ca, sb, cb, sg, cg = np.sin(a), np.cos(a), np.sin(b), np.cos(b), np.sin(g), np.cos(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    matrix = np.stack(
        [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb, -sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb, sc, ss, cb],
        axis=1,
    ).astype(F32)

    def product(left, right):
        out = np.zeros_like(matrix)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    out[:, i * 3 + j] += left[:, i * 3 + k] * right[:, k * 3 + j]
        return out

    n = matrix.shape[0]
    if right_matrix is not None:
        matrix = product(matrix, np.broadcast_to(np.asarray(right_matrix, dtype=F32).reshape(1, 9), (n, 9)))
    if left_matrix is None:
        # Without a left matrix the inverse is the transpose; relax's row is its transpose: the matrix itself.
        return matrix.reshape(n, 3, 3)
    m = product(np.broadcast_to(np.asarray(left_matrix, dtype=F32).reshape(1, 9), (n, 9)), matrix)
    det = (
        m[:, 0] * (m[:, 4] * m[:, 8] - m[:, 7] * m[:, 5])
        - m[:, 1] * (m[:, 3] * m[:, 8] - m[:, 6] * m[:, 5])
        + m[:, 2] * (m[:, 3] * m[:, 7] - m[:, 6] * m[:, 4])
    )
    inverse = np.stack(
        [
            (m[:, 4] * m[:, 8] - m[:, 7] * m[:, 5]) / det,
            (m[:, 7] * m[:, 2] - m[:, 1] * m[:, 8]) / det,
            (m[:, 1] * m[:, 5] - m[:, 4] * m[:, 2]) / det,
            (m[:, 5] * m[:, 6] - m[:, 8] * m[:, 3]) / det,
            (m[:, 8] * m[:, 0] - m[:, 2] * m[:, 6]) / det,
            (m[:, 2] * m[:, 3] - m[:, 5] * m[:, 0]) / det,
            (m[:, 3] * m[:, 7] - m[:, 6] * m[:, 4]) / det,
            (m[:, 6] * m[:, 1] - m[:, 0] * m[:, 7]) / det,
            (m[:, 0] * m[:, 4] - m[:, 3] * m[:, 1]) / det,
        ],
        axis=1,
    ).astype(F32)
    return inverse.reshape(n, 3, 3).transpose(0, 2, 1)
