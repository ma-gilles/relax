"""Configuration admission of the resident local-search pass 2: refuses what the driver does not implement.

``require_resident_local_configuration`` raises :class:`~relax.fine_pass.policy.ResidentConfigurationUnsupported`
unless the call is the K=1 or Class3D local fine pass 2, or its pass-1 parent probe, in production shape, with
no diagnostic arm requested. ``require_local_support`` is the single refusal the checks share. Checked before
any device work; imports nothing from the driver (``resident_pass2``).
"""

from __future__ import annotations

import relax.runtime.engine_tuning
from relax.fine_pass.policy import ResidentConfigurationUnsupported


def require_local_support(condition: bool, message: str) -> None:
    if not condition:
        raise ResidentConfigurationUnsupported(
            "The device-resident local-search pass 2 does not implement this configuration: "
            f"{message}. There is no other local fine-pass engine."
        )


def require_resident_local_configuration(**kwargs) -> None:
    """Raise unless this is the K=1 local fine pass 2, or its pass-1 parent probe, in production shape."""

    if bool(kwargs["score_only"]):
        _require_parent_probe_configuration(**kwargs)
        return
    require_local_support(
        not (bool(kwargs["disable_adjoint_y"]) or bool(kwargs["disable_adjoint_ctf"])),
        "a score-only pass has no M-step to make resident",
    )
    require_local_support(bool(kwargs["mstep_relion_x_half"]), "the RELION x-half M-step is required")
    require_local_support(bool(kwargs["accumulate_noise"]), "the production pass accumulates noise statistics")
    require_local_support(
        kwargs["max_significants"] is None or int(kwargs["max_significants"]) <= 0,
        "a maximum_significants cap on the fine support is not in the segmented "
        "posterior's contract",
    )
    # Pass 2 of an oversampled search reconstructs from and accumulates the
    # pruned posterior; the zero-oversampling route keeps every weight for both
    # (RELION's symbolic second pass sets significant_weight to the minimum
    # weight: acc_ml_optimiser_impl.h:3590). One posterior serves both.
    require_local_support(
        bool(kwargs["stats_use_reconstruction_probs"]) == bool(kwargs["reconstruct_significant_only"]),
        "the resident statistics stage accumulates the reconstruction posterior, so the "
        "statistics and the reconstruction must use the same (pruned or complete) weights",
    )
    require_local_support(
        not bool(kwargs["use_float64_scoring"]) and not bool(kwargs["use_float64_projections"]),
        "float64 local search is a diagnostic mode",
    )
    require_local_support(
        bool(kwargs["relion_exact_score_translation"]),
        "the fused translate-and-score kernel needs RELION translation angles",
    )
    require_local_support(bool(kwargs["half_spectrum_scoring"]), "RELION half-spectrum scoring is required")
    require_local_support(
        kwargs["relion_projector_half"] is not None
        and kwargs["relion_projector_r_max"] is not None,
        "the resident row projection uses the RELION PPref projector",
    )
    require_local_support(
        kwargs["normalization_log_evidence"] is None,
        "an externally supplied normalizer belongs to the broad-denominator probe",
    )
    require_local_support(
        not bool(kwargs["return_reconstruction_sample_indices"]),
        "significant-sample capture belongs to the pass-1 parent probe",
    )
    require_local_support(
        kwargs["group_ids"] is not None,
        "the resident statistics stage accumulates RELION's group scale terms",
    )
    require_local_support(
        bool(kwargs["relion_wavg_atomic_scale_aa"]),
        "the resident statistics stage consumes the RELION atomic Wavg triplet",
    )
    # RELION scores its radial support at every current size, the final
    # all-data iteration's full box included (updateImageSizeAndResolutionPointers
    # cuts the FFTW rectangle's corners); both local engines take that window at
    # the box (window_at_box, aa03fd2), and the RELION Wavg rectangle requires it.
    require_local_support(
        bool(kwargs["use_window"]),
        "the resident local pass scores RELION's radial window; a full-rectangle "
        "support is not RELION's scoring support and is not implemented here",
    )
    _refuse_requested_diagnostics(kwargs["requested_diagnostics"])


def _refuse_requested_diagnostics(requested_diagnostics) -> None:
    """Refuse the first diagnostic dump or arm that is set (``EngineTuning.resident_diagnostics``)."""

    for name in requested_diagnostics:
        require_local_support(
            name not in relax.runtime.engine_tuning.RESIDENT_DIAGNOSTIC_DIR_ENVS,
            f"the diagnostic dump {name} is set; this driver emits no dumps",
        )
        require_local_support(False, f"the diagnostic flag {name} is set; this driver has no such arm")


def _require_parent_probe_configuration(**kwargs) -> None:
    """Raise unless this is RELION's local pass 1 (the parent probe) in production shape."""

    require_local_support(
        bool(kwargs["return_reconstruction_sample_indices"]),
        "a score-only local pass is the parent probe, which returns its significant samples",
    )
    # The probe keeps RELION's adaptive-fraction support; the segmented float32
    # posterior has no maximum_significants cap, and RELION's default (--maxsig
    # -1, ml_optimiser.cpp:1109) applies none. A cap stays on the exact engine.
    require_local_support(
        kwargs["max_significants"] is None or int(kwargs["max_significants"]) <= 0,
        "a maximum_significants cap on the pass-1 support is not in the segmented "
        "posterior's contract",
    )
    require_local_support(
        not bool(kwargs["use_float64_scoring"]) and not bool(kwargs["use_float64_projections"]),
        "float64 local search is a diagnostic mode",
    )
    require_local_support(
        bool(kwargs["relion_exact_score_translation"]),
        "the fused translate-and-score kernel needs RELION translation angles",
    )
    require_local_support(bool(kwargs["half_spectrum_scoring"]), "RELION half-spectrum scoring is required")
    require_local_support(
        kwargs["relion_projector_half"] is not None
        and kwargs["relion_projector_r_max"] is not None,
        "the resident row projection uses the RELION PPref projector",
    )
    require_local_support(
        kwargs["normalization_log_evidence"] is None,
        "an externally supplied normalizer belongs to the broad-denominator probe",
    )
    require_local_support(bool(kwargs["use_window"]), "the probe scores RELION's current-size window")
    _refuse_requested_diagnostics(kwargs["requested_diagnostics"])
