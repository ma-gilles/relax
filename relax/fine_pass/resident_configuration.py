"""The resident pass-2 configuration gate.

:func:`require_resident_production_configuration` checks, before any device work, that a pass asks for the
production configuration the resident drivers implement (RELION x-half M-step, exact RELION fine Gaussian
scoring, float32 fine posterior, fine M-step prune, float32 scoring, no diagnostics or dumps), and raises
:class:`~relax.fine_pass.policy.ResidentConfigurationUnsupported` naming the first piece that is missing.
There is no other pass-2 engine to fall back to.
"""

from __future__ import annotations

import numpy as np

import relax.runtime.engine_tuning
from relax.fine_pass.policy import ResidentConfigurationUnsupported

# The diagnostic dumps and arms the resident drivers refuse (EngineTuning.resident_diagnostics).
DIAGNOSTIC_DIR_ENVS = relax.runtime.engine_tuning.RESIDENT_DIAGNOSTIC_DIR_ENVS
DIAGNOSTIC_FLAG_ENVS = relax.runtime.engine_tuning.RESIDENT_DIAGNOSTIC_FLAG_ENVS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResidentConfigurationUnsupported(
            f"The device-resident sparse pass 2 does not implement this configuration: {message}."
        )


def require_resident_production_configuration(**kwargs) -> None:
    """Raise :class:`NotImplementedError` unless this is the production path.

    Every check names the specific missing piece rather than reporting a
    generic refusal, so a caller that trips one knows which behaviour the
    resident driver would have to grow.
    """

    require(bool(kwargs["relion_x_half_mstep"]), "the RELION x-half M-step is required")
    # The resident driver builds its own texture from relion_projector_half; a
    # caller that supplies a persistent one gets a named refusal, not a drop.
    require(
        kwargs["relion_projector_texture"] is None,
        "the resident driver projects from relion_projector_half, not a caller's persistent texture",
    )
    score_mode = kwargs["relion_firstiter_score_mode"]
    require(
        (score_mode == "gaussian" and bool(kwargs["relion_exact_fine_gaussian"]))
        or (score_mode == "normalized_cc" and bool(kwargs.get("relion_exact_fine_normalized_cc"))),
        "exact RELION fine Gaussian scoring, or RELION's literal fine normalized-CC "
        "reduction for --firstiter_cc, is required "
        f"(got relion_exact_fine_gaussian={kwargs['relion_exact_fine_gaussian']!r}, "
        f"relion_exact_fine_normalized_cc={kwargs.get('relion_exact_fine_normalized_cc')!r}, "
        f"score_mode={score_mode!r})",
    )
    require(not bool(kwargs["use_float64_scoring"]), "float64 scoring is a diagnostic mode")
    # RELION's --firstiter_cc iteration scores with normalized CC and keeps only
    # the best weight (ml_optimiser.cpp:9266-9293); a Gaussian pass never does.
    require(
        bool(kwargs["relion_firstiter_winner_take_all"]) == (score_mode == "normalized_cc"),
        "winner-take-all goes with the --firstiter_cc normalized-CC pass and only with it "
        f"(winner_take_all={bool(kwargs['relion_firstiter_winner_take_all'])!r}, "
        f"score_mode={score_mode!r})",
    )
    require(not bool(kwargs["return_score_log_z_only"]), "score-logZ-only passes are score-only")
    require(bool(kwargs["accumulate_noise"]), "the production pass accumulates noise statistics")
    # The K=1 adaptive route always hands the M-step call an
    # ``normalization_other_score_log_z`` built from the *other* classes'
    # log-Z (k_class.py::_run_sparse_k_class_adaptive_pass2). At K=1 there are
    # no other classes, so that vector is all -inf and the compact engine's
    # own arithmetic collapses to its unnormalized branch: logaddexp(x, -inf)
    # is x, the reported log-evidence and score log-Z are taken from
    # ``local_score_log_z`` rather than from the combined value, and the
    # float32 reconstruction weights never read it at all. Accept exactly that
    # degenerate vector, which is the production K=1 case, and refuse any
    # finite entry, which would genuinely mix classes.
    other_log_z = kwargs["normalization_other_score_log_z"]
    other_log_z_is_degenerate = other_log_z is not None and bool(
        np.all(np.asarray(other_log_z) == -np.inf)
    )
    require(
        kwargs["normalization_log_z"] is None,
        "an externally supplied log-Z belongs to the K-class engine",
    )
    require(
        other_log_z is None or other_log_z_is_degenerate,
        "a finite cross-class score normalization belongs to the K-class engine",
    )
    coarse_sum_weight = kwargs["relion_f32_normalization_sum_weight"]
    coarse_winner = kwargs["relion_coarse_hard_assignment"]
    coarse_max_posterior = kwargs.get("relion_coarse_max_posterior")
    require(
        (coarse_sum_weight is None) == (coarse_winner is None) == (coarse_max_posterior is None),
        "the zero-oversampling pass reuses the coarse normalization sum, winner and Pmax "
        "together, as the K=1 adaptive route supplies them",
    )
    require(
        coarse_sum_weight is None or int(kwargs.get("oversampling_order", 0)) == 0,
        "the coarse normalization sum is reused only at zero oversampling "
        "(acc_ml_optimiser_impl.h:2868)",
    )
    # ``preserve_bpref_particle_order`` is the production setting, and on its
    # own it forces one BPref launch per particle. The production run pairs it
    # with the soft-posterior block prototype, which turns those launches back
    # into one block launch per bucket; that is the semantics the resident
    # driver reproduces with one launch per row block. Without the prototype
    # the compact engine really would launch per particle, so refuse.
    require(
        (not bool(kwargs["preserve_bpref_particle_order"]))
        or bool(kwargs["soft_posterior_block_bpref"]),
        "strict per-particle BPref launches are not implemented; set "
        "RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF=1 (the production "
        "setting) so BPref accumulates per block, or clear "
        "preserve_bpref_particle_order",
    )
    require(
        kwargs["fine_rotations_override"] is not None
        and kwargs["fine_rotation_parent_override"] is not None,
        "the resident driver gathers rotations from the caller's fine grid, "
        "so fine_rotations_override and fine_rotation_parent_override are required",
    )
    require(
        bool(kwargs["use_window"]),
        "the resident driver scores through the RELION current-size window; "
        "a full-half pass would include the ky=-N/2 Nyquist row, where the "
        "fused-translate and pre-shifted scorers are known to differ",
    )
    require(
        bool(kwargs["projection_cache_available"]),
        "the resident scoring stage gathers cached fine-rotation projections; "
        "the per-iteration projection cache is disabled or did not fit its budget",
    )
    require(
        bool(kwargs["relion_wavg_atomic_scale_aa"]),
        "the resident statistics stage consumes the RELION atomic Wavg triplet",
    )
    require(
        bool(kwargs["relion_wavg_atomic_direct_noise"]),
        "the resident statistics stage uses RELION's direct low-shell residual",
    )
    require(
        not bool(kwargs["relion_wavg_atomic_direct_norm"]),
        "the direct per-particle Wavg norm arm is a stopped diagnostic",
    )
    for name in kwargs["requested_diagnostics"]:
        require(
            name not in DIAGNOSTIC_DIR_ENVS,
            f"the diagnostic dump {name} is set; the resident driver emits no dumps",
        )
        require(False, f"the diagnostic flag {name} is set; the resident driver has no such arm")
