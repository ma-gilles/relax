"""The ``EngineTuning`` values the resident engine tests run with, passed to the engines as the command would."""

from dataclasses import replace

from relax.runtime.engine_tuning import EngineTuning

# The global resident pass's production capacity ladders and M-step block rows.
RESIDENT_PRODUCTION_ENGINE = replace(
    EngineTuning(), resident_row_capacities=(256, 1024, 4096), resident_image_capacities=(4, 16, 64), mstep_block_rows=128,
)
# The local resident pass's capacity ladders for the small test fixtures.
LOCAL_RESIDENT_ENGINE = replace(EngineTuning(), local_row_capacities=(64, 256, 1024), local_image_capacities=(2, 4, 8))
