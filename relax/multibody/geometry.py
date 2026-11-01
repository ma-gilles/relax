"""Per-particle body geometry: composed projection matrices, projected centres and image shifts.

Everything follows ``getFourierTransformsAndCtfs`` of RELION's accelerated multi-body path
(acc_ml_optimiser_impl.h:70-100, 880-998 at f2c1a38) in RELION's conventions: Euler matrices from
``Euler_angles2matrix``, offsets in pixels, and an image shift ``v`` moving image content by ``+v`` (RELION's
``translate2D`` and ``shiftImageInFourierTransform``). Relax's projection rows are these RELION matrices
(``relax.sampling.grids.relion_mstep_rotations_from_euler_matrices``), so a composed body matrix is a relax row too.

For particle ``i`` with consensus pose ``A_i`` and offset ``o_i`` and body ``b`` with orientation ``O_b``, centre
``c_b`` and refined residual pose ``(R_ib, f_ib)``:

- the body's projection matrix is ``A_i O_b^T A_rot90 R_ib O_b``;
- its projected centre is ``p_ib = (A_i c_b)_xy``;
- its image is translated by the integer ``r_ib = ROUND(o_i - p_ib)``, then, after the other bodies are
  subtracted and the image is masked, by the fractional ``f_ib`` in Fourier space;
- another body ``q`` sits in that image at ``s_ibq = p_iq - o_i - f_iq + r_ib``.
"""

from __future__ import annotations

import numpy as np

from relax.multibody.bodies import A_ROT90, Body
from relax.relion.macros import relion_round_array
from relax.sampling.healpix import euler_angles_to_matrix


def consensus_matrices(consensus_eulers_deg) -> np.ndarray:
    """``[N, 3, 3]`` RELION matrices of the consensus (whole-particle) poses."""

    return euler_angles_to_matrix(np.asarray(consensus_eulers_deg, dtype=np.float64))


def body_left_right(consensus: np.ndarray, body: Body) -> tuple[np.ndarray, np.ndarray]:
    """RELION's ``MBL = A O^T A_rot90`` (``[N, 3, 3]``) and ``MBR = O`` (``[3, 3]``) of a body."""

    return consensus @ (body.orientation.T @ A_ROT90), body.orientation


def body_matrices(consensus: np.ndarray, body: Body, residual_eulers_deg) -> np.ndarray:
    """``[N, 3, 3]`` projection matrices ``A O^T A_rot90 R(residual) O`` of one body."""

    left, right = body_left_right(consensus, body)
    return left @ euler_angles_to_matrix(np.asarray(residual_eulers_deg, dtype=np.float64)) @ right


def projected_centers(consensus: np.ndarray, body: Body) -> np.ndarray:
    """``[N, 2]`` projected centre ``(A c)_xy`` of a body in each image, pixels."""

    return (consensus @ body.com)[:, :2]


def integer_image_shifts(consensus_offsets_px, projected: np.ndarray) -> np.ndarray:
    """``[N, 2]`` integer shift ``ROUND(o - p)`` applied to each image before the body is scored."""

    return relion_round_array(np.asarray(consensus_offsets_px, dtype=np.float64) - projected)


def other_body_shifts(projected_other, consensus_offsets_px, refined_other_px, integer_shifts) -> np.ndarray:
    """``[N, 2]`` position ``p_q - o - f_q + r_b`` of another body's centred projection in a body's image."""

    return (
        np.asarray(projected_other, dtype=np.float64)
        - np.asarray(consensus_offsets_px, dtype=np.float64)
        - np.asarray(refined_other_px, dtype=np.float64)
        + np.asarray(integer_shifts, dtype=np.float64)
    )


def body_mask_radius_px(body: Body, offset_range_angstrom: float, pixel_size: float) -> float:
    """The particle mask radius of a body's images: ``(max_radius + offset_range) / pixel_size``
    (acc_ml_optimiser_impl.h:550-552)."""

    return (body.max_radius_angstrom + float(offset_range_angstrom)) / float(pixel_size)
