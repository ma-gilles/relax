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


@dataclass(frozen=True, kw_only=True)
class EngineTuning:
    """The engines' execution settings (no effect on any value).

    ``compile_ahead`` (RELAX_EM_COMPILE_AHEAD, RELAX_EM_COMPILE_AHEAD_MAX_PROGRAMS;
    :func:`relax.runtime.compile_ahead.parse_compile_ahead_config`): the resident pass 2's helper-thread warm-up.
    """

    compile_ahead: CompileAheadConfig = field(default_factory=CompileAheadConfig)

    @classmethod
    def from_environ(cls) -> EngineTuning:
        return cls(
            compile_ahead=parse_compile_ahead_config(
                read_env_text(COMPILE_AHEAD_ENV, ""), read_env_text(COMPILE_AHEAD_MAX_ENV, ""),
            ),
        )
