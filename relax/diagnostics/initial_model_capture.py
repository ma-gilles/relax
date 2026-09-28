"""Default-off captures for InitialModel K-class localization.

Each records a quantity no output file carries, writes only when its environment
variable names a directory, and changes no production value.
"""

from __future__ import annotations

import os

_KCLASS_STATS_DUMP_ENV = "RELAX_VDAM_KCLASS_STATS_DUMP_DIR"


def k_class_statistics_capture_enabled() -> bool:
    """Whether the K-class statistics capture is on, for producers of its inputs.

    The engine buffer and the per-class driver copy of the pre-cast normalizer
    feed only this module, so both ask here rather than reading the environment
    separately. With the capture off neither is allocated, pulled or copied.
    """
    return bool(os.environ.get(_KCLASS_STATS_DUMP_ENV))




