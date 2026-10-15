"""Execution-policy values of the sparse bucketed pass 2.

The resident configuration refusal and the per-pass projection-cache resolution. The pass-2 switches are read
once by the commands (relax.runtime.engine_tuning.EngineTuning) or retired.
"""

from __future__ import annotations


class ResidentConfigurationUnsupported(NotImplementedError):
    """A device-resident K=1 driver does not implement this pass's configuration.

    Raised by the resident drivers' configuration checks, which run before any
    device work. It stops the run: there is no other pass-2 engine to fall back to.
    """


def projection_cache_enabled_for_pass(
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
