"""The Fourier rows pass 1 scores and their weights: the window of the current size, fixed before the first batch."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from relax.helpers.fourier_window import make_fourier_window_spec
from relax.helpers.half_spectrum import make_scoring_half_image_weights


@dataclass(frozen=True)
class ScoringWindow:
    """The rows of the half spectrum a pass scores, and what weights them.

    ``half_weights`` weight every row of the half spectrum; ``score_half_weights`` the scored rows (the weights of the
    window's rows when the pass uses a window). ``window_spec`` describes the window; ``window_indices`` are its rows
    when ``use_window`` (``None``: every row). ``score_size`` is the current size the pass scores at (the image size
    when the call gives none). ``cc_gaussian_support`` says that a normalized-CC pass takes its image power on the
    Gaussian support of the current size (``firstiter_cc_support="gaussian"``).
    """

    half_weights: Any
    score_half_weights: Any
    window_spec: Any
    use_window: bool
    window_indices: Any
    score_size: int
    cc_gaussian_support: bool


def plan_scoring_window(
    image_shape,
    n_half: int,
    current_size,
    *,
    score_mode: str,
    half_spectrum_scoring: bool,
    square_window: bool,
    window_at_box: bool,
    nyquist_column_counting: str,
    firstiter_cc_support: str,
) -> ScoringWindow:
    """The window and weights of a ``"gaussian"`` or ``"normalized_cc"`` pass over images of ``image_shape``.

    ``n_half`` is the half spectrum's pixel count. The normalized CC scores a square window with its DC row; the
    Gaussian score may keep RELION's radial window at the box (``window_at_box``).
    """

    cc_gaussian_support = score_mode == "normalized_cc" and firstiter_cc_support == "gaussian"
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=score_mode != "normalized_cc",
        nyquist_column_counting=nyquist_column_counting,
        firstiter_cc_support_size=(
            (image_shape[0] if current_size is None else current_size) if cc_gaussian_support else None
        ),
    )
    window_spec_kwargs = {}
    if score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=False,
        # RELION's radial window at the box too (Gaussian scoring; the normalized-CC score
        # keeps its rectangular window).
        window_at_box=bool(window_at_box) and score_mode != "normalized_cc",
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    return ScoringWindow(
        half_weights=half_weights,
        score_half_weights=window_spec.score_values(half_weights) if use_window else half_weights,
        window_spec=window_spec,
        use_window=use_window,
        window_indices=window_spec.score_indices if use_window else None,
        score_size=int(image_shape[0]) if current_size is None else int(current_size),
        cc_gaussian_support=cc_gaussian_support,
    )


def coarse_kernel_window(score_size: int, projector_r_max, rotations):
    """The window RELION's coarse kernel shifts the rows beyond ``r_max`` at, or ``None`` when it shifts none.

    RELION's coarse kernel projects and shifts the rows beyond maxR at ``i - window`` inside
    the model sphere only for a window between 2 r_max and about 2 s r_max (an optics group on
    a coarser grid: its rotations carry 1 / s); the exact coarse operands then shift them there.
    Call it with a supplied projector (``projector_r_max`` is its radius).
    """

    if score_size // 2 > int(projector_r_max):
        from relax.helpers.optics_scale import coarse_rows_wrap_inside

        rotation_scale = 1.0 / float(np.linalg.norm(np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)[0, 0]))
        if coarse_rows_wrap_inside(score_size, int(projector_r_max), rotation_scale):
            return score_size
    return None
