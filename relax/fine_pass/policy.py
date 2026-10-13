"""Execution-policy switches of the sparse bucketed pass 2.

The environment-gated opt-ins and per-pass policy values (BPref processing
order and execution buckets, native weighted sums, fused M-step noise, the
RELION wavg direct modes, projection cache, windowed prepare and translation
tiles, compact-pair execution and bucket coalescing), and the resident pass-2
switches (per-chunk timing, resident per-image operands and their verification
arm, the kernel's ctf_probs, the jitted stage glue and the carry-aval probe).
Each reader is the single owner of its variable; the pass-2 kernels ask these
once per pass.
"""

from __future__ import annotations

from relax.runtime.env_flags import parse_env_flag

_RELION_WAVG_ATOMIC_SCALE_AA_ENV = "RELAX_RELION_WAVG_ATOMIC_SCALE_AA"


_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV = (
    "RELAX_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL"
)


_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV = (
    "RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY"
)


class ResidentConfigurationUnsupported(NotImplementedError):
    """A device-resident K=1 driver does not implement this pass's configuration.

    Raised by the resident drivers' configuration checks, which run before any
    device work. It stops the run: there is no other pass-2 engine to fall back to.
    """


_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV = "RELAX_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE"


_BPREF_REVERSE_PHYSICAL_ORDER_ENV = "RELAX_K1_BPREF_REVERSE_PHYSICAL_ORDER"


_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV = "RELAX_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE"


_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV = (
    "RELAX_K1_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT"
)


_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV = (
    "RELAX_K1_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE"
)






def _relion_wavg_direct_modes(
    *,
    accumulate_noise: bool,
    scale_groups_available: bool,
    scale_aa_enabled: bool,
    direct_noise_only_default: bool = False,
) -> tuple[bool, bool]:
    """Resolve the stopped direct-Wavg noise/norm factorial arms.

    ``DIRECT_RESIDUAL`` preserves the existing coupled treatment: the native
    Wavg ``diff2`` stream supplies both shell noise and per-particle norm.
    ``DIRECT_NOISE_ONLY`` supplies only shell noise, leaving normalization on
    the production algebraic path.  The latter isolates the already-localized
    radial-noise boundary without silently changing a second state variable.
    """

    direct_residual_requested = bool(
        accumulate_noise
        and parse_env_flag(
            _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV,
            default=False,
        )
    )
    direct_noise_only_requested = bool(
        accumulate_noise
        and parse_env_flag(
            _RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV,
            default=direct_noise_only_default,
        )
    )
    if direct_residual_requested and direct_noise_only_requested:
        raise ValueError(
            f"{_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV}=1 and "
            f"{_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV}=1 are mutually exclusive"
        )
    direct_noise = direct_residual_requested or direct_noise_only_requested
    # Fresh iteration 1 intentionally has no scale-group accumulator.  The
    # established coupled diagnostic is dormant there and activates once the
    # scale state exists; preserve that lifecycle for the isolated arm.
    if direct_noise and not scale_groups_available:
        return False, False
    if direct_noise and not scale_aa_enabled:
        requested_name = (
            _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV
            if direct_residual_requested
            else _RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV
        )
        raise ValueError(
            f"{requested_name}=1 requires "
            f"{_RELION_WAVG_ATOMIC_SCALE_AA_ENV}=1 and scale groups"
        )
    return direct_noise, direct_residual_requested


def _projection_cache_enabled_for_pass(
    *,
    fine_rotations_override,
    dump_pass2_operands: bool,
    requested: bool | None,
) -> bool:
    """Resolve the diagnostic projection-cache override without changing defaults; ``requested`` is
    ``EngineTuning.pass2_projection_cache`` (None: auto)."""

    if fine_rotations_override is None:
        return False
    if requested is None:
        # Preserve the currently qualified paths while cache-on/cache-off is
        # adjudicated: production uses the cache and operand dumps do not.
        return not bool(dump_pass2_operands)
    return requested


# Attribution only, default off. Logs one line per chunk with its occupancy,
# its M-step block count and a device-synchronised wall, and counts the T7
# offsets readbacks. The synchronisation perturbs the wall, so an arm with
# this set is a diagnostic arm and never a timing arm.


# T16: prepare the per-image operands once per half and keep them resident, and
# take the M-step's weighted sums with T15's flat-row translate-and-sum kernel
# instead of a gathered ``[images, translations, pixels]`` tile. Default on;
# ``RELAX_SPARSE_PASS2_RESIDENT_OPERANDS=0`` selects the per-chunk
# ``_prepare_bucket_io`` preparation and the XLA tile reduction, which stay as
# the oracle both forms are compared against.
_RESIDENT_OPERANDS_ENV = "RELAX_SPARSE_PASS2_RESIDENT_OPERANDS"


# Diagnostic, default off. For the first chunk of a half it also runs the
# per-chunk preparation and checks, on that chunk's real operands, that
# translating the resident per-image arrays reproduces the pre-shifted tiles
# bitwise and that the kernel's weighted sums equal the XLA reduction. It
# doubles that chunk's preparation cost, so an arm with it set is a diagnostic
# arm, never a timing arm.
_RESIDENT_OPERANDS_VERIFY_ENV = "RELAX_SPARSE_PASS2_RESIDENT_OPERANDS_VERIFY"


# Take ``ctf_probs`` from the translate-and-sum kernel's fourth output instead
# of the XLA statement. Measurement only: see
# ``resident_block_weighted_sums_kernel`` for why it is not the default.
_KERNEL_CTF_PROBS_ENV = "RELAX_SPARSE_PASS2_RESIDENT_KERNEL_CTF_PROBS"


# P3-A: dispatch the per-stage chunk loop's three stages as jitted programs
# keyed on the capacity class instead of as loose eager operations. Default on.
# ``RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT=0`` restores the loose dispatch,
# which stays the oracle every bitwise comparison of this change is made
# against. The stage bodies are the same functions in both settings, so the
# flag changes only where the JIT boundary sits.
_RESIDENT_GLUE_JIT_ENV = "RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT"


# Diagnostic, default off. Checks the statically computed M-step carry avals
# against a ``jax.eval_shape`` probe of the same block stages, once per
# capacity class. The probes are what this change removes from the chunk loop;
# the flag exists so a test, or a suspicious run, can prove the arithmetic
# still agrees with them.
_CARRY_AVAL_PROBE_ENV = "RELAX_SPARSE_PASS2_RESIDENT_CARRY_AVAL_PROBE"


def kernel_ctf_probs_enabled() -> bool:
    """Whether the M-step's ``ctf_probs`` comes from the kernel's fourth output."""

    return parse_env_flag(_KERNEL_CTF_PROBS_ENV, default=False)


def resident_operands_verify_enabled() -> bool:
    """Whether to check the first chunk of a half against the per-chunk path."""

    return parse_env_flag(_RESIDENT_OPERANDS_VERIFY_ENV, default=False)


def resident_operands_requested() -> bool:
    """Whether the per-image operands are prepared once per half (T16).

    Default on. ``RELAX_SPARSE_PASS2_RESIDENT_OPERANDS=0`` keeps the per-chunk
    ``_prepare_bucket_io`` preparation and the XLA tile reduction, which are the
    oracle for every bitwise comparison of the new path.
    """

    return parse_env_flag(_RESIDENT_OPERANDS_ENV, default=True)


def resident_glue_jit_enabled() -> bool:
    """Whether the chunk loop's three stages are dispatched as jitted programs.

    Default on. With the flag off every stage runs as the loose sequence of
    eager operations the per-stage path used before P3-A: the same functions,
    the same order, only without the enclosing ``jax.jit``. That form is the
    oracle for the bitwise comparison, so it is kept rather than deleted.
    """

    return parse_env_flag(_RESIDENT_GLUE_JIT_ENV, default=True)


def carry_aval_probe_enabled() -> bool:
    """Whether to check the static carry avals against a shape probe."""

    return parse_env_flag(_CARRY_AVAL_PROBE_ENV, default=False)
