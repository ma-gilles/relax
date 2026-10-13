"""Execution-policy values of the sparse bucketed pass 2.

The resident configuration refusal, the BPref execution-order variable names, the per-pass projection-cache
resolution and the resident pass 2's jitted stage glue switch. The other pass-2 switches are read once by the
commands (relax.runtime.engine_tuning.EngineTuning) or retired.
"""

from __future__ import annotations

from relax.runtime.env_flags import parse_env_flag


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


# P3-A: dispatch the per-stage chunk loop's three stages as jitted programs
# keyed on the capacity class instead of as loose eager operations. Default on.
# ``RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT=0`` restores the loose dispatch,
# which stays the oracle every bitwise comparison of this change is made
# against. The stage bodies are the same functions in both settings, so the
# flag changes only where the JIT boundary sits.
_RESIDENT_GLUE_JIT_ENV = "RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT"


def resident_glue_jit_enabled() -> bool:
    """Whether the chunk loop's three stages are dispatched as jitted programs.

    Default on. With the flag off every stage runs as the loose sequence of
    eager operations the per-stage path used before P3-A: the same functions,
    the same order, only without the enclosing ``jax.jit``. That form is the
    oracle for the bitwise comparison, so it is kept rather than deleted.
    """

    return parse_env_flag(_RESIDENT_GLUE_JIT_ENV, default=True)
