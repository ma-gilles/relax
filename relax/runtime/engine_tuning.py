"""How the scoring and reconstruction engines execute, read once from the environment (code rule 5).

:class:`EngineTuning` holds the engines' execution settings that come from environment variables: memory
budgets, chunk and batch sizes, host or device placement and warm-up. None of them changes a computed value.
:meth:`EngineTuning.from_environ` reads them through ``relax.runtime.env_flags``; the commands call it once
(``RefinementEnvironment.engine``, ``VdamEnvironment.engine``) and the engines receive the record as an argument.
The default instance is the unset environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from relax.runtime.compile_ahead import (
    COMPILE_AHEAD_ENV,
    COMPILE_AHEAD_MAX_ENV,
    CompileAheadConfig,
    parse_compile_ahead_config,
)
from relax.runtime.env_flags import read_env_text

DEFAULT_PREPARE_IMAGE_BATCH = 256


@dataclass(frozen=True, kw_only=True)
class EngineTuning:
    """The engines' execution settings (no effect on any value).

    ``compile_ahead`` (RELAX_EM_COMPILE_AHEAD, RELAX_EM_COMPILE_AHEAD_MAX_PROGRAMS;
    :func:`relax.runtime.compile_ahead.parse_compile_ahead_config`): the resident pass 2's helper-thread warm-up.
    ``pass2_projection_cache`` (RELAX_SPARSE_PASS2_PROJECTION_CACHE, :func:`parse_projection_cache_request`): the
    resident pass 2's projection cache, None for its default (on, except for an operand dump).
    ``resident_operand_max_bytes`` (RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES, :func:`parse_positive_override`):
    the resident operands' byte budget, None for the allocator-derived one. ``prepare_image_batch``
    (RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH, :func:`parse_positive_override`; default
    :data:`DEFAULT_PREPARE_IMAGE_BATCH`): images per resident operand preparation call (no value depends on it).
    """

    compile_ahead: CompileAheadConfig = field(default_factory=CompileAheadConfig)
    pass2_projection_cache: bool | None = None
    resident_operand_max_bytes: int | None = None
    prepare_image_batch: int = DEFAULT_PREPARE_IMAGE_BATCH

    @classmethod
    def from_environ(cls) -> EngineTuning:
        return cls(
            compile_ahead=parse_compile_ahead_config(
                read_env_text(COMPILE_AHEAD_ENV, ""), read_env_text(COMPILE_AHEAD_MAX_ENV, ""),
            ),
            pass2_projection_cache=parse_projection_cache_request(read_env_text(PROJECTION_CACHE_ENV)),
            resident_operand_max_bytes=parse_positive_override(
                "RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES",
                read_env_text("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES", ""),
            ),
            prepare_image_batch=parse_positive_override(
                "RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH",
                read_env_text("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH", ""),
            ) or DEFAULT_PREPARE_IMAGE_BATCH,
        )


def parse_positive_override(name: str, raw: str) -> int | None:
    """A positive integer override (stripped): blank is None; ``int()`` refuses a non-integer, and a value that is
    not positive is refused naming ``name``."""
    raw = raw.strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


PROJECTION_CACHE_ENV = "RELAX_SPARSE_PASS2_PROJECTION_CACHE"


def parse_projection_cache_request(raw: str | None) -> bool | None:
    """RELAX_SPARSE_PASS2_PROJECTION_CACHE: unset, blank or ``auto`` is None; 1/true/yes/on True; 0/false/no/off
    False (any case, stripped); anything else is refused."""
    mode = "auto" if raw is None or raw.strip() == "" else raw.strip().lower()
    if mode == "auto":
        return None
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{PROJECTION_CACHE_ENV} must be 'auto', 'on', or 'off', got {raw!r}")
