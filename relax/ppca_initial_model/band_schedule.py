"""The physical band rule of ab initio PPCA: the band grows in frequency, not in pixels, so it does not depend on the box."""

import math

BAND_START_ANGSTROM = 35.0
BAND_GROWTH_ANGSTROM = 384.0  # the band gains 1 / BAND_GROWTH_ANGSTROM per step
BAND_STEP_UPDATES = 125
# Steps first update 125 k + 1 before this update and 125 k from it on, as the measured schedules were written.
BAND_STEP_ALIGNED_FROM = 2500
COARSE_ORDER_LIMIT_ANGSTROM = 32.0  # HEALPix order 1 above, 2 down to the next limit
FINE_ORDER_LIMIT_ANGSTROM = 16.0  # HEALPix order 3 at and below
# A band that stops above FINE_ORDER_LIMIT_ANGSTROM takes order 3 from here instead.
EARLY_FINE_ORDER_ANGSTROM = 20.0


def _nearest_shell(radius):
    return math.floor(radius + 0.5)


def physical_band_stages(grid_size, voxel_size, target_resolution=None):
    """``Config.stages`` of the physical band rule for a box of ``grid_size`` voxels of ``voxel_size`` Angstrom.

    With ``L = grid_size voxel_size``, the band starts at 35 A and gains ``1/(384 A)`` every 125 updates, on the
    nearest shell ``L f``, up to ``target_resolution`` (Angstrom) or Nyquist - 1 if that comes first. HEALPix order
    (at the stage's ``L / shell``): 1 above 32 A, 2 down to but excluding 16 A, 3 from there, or 3 from 20 A if the
    band stops above 16 A. Steps that reach neither a new shell nor a new order start no stage.
    """
    box = grid_size * voxel_size
    last = grid_size // 2 - 1
    if target_resolution is not None:
        last = min(last, _nearest_shell(box / target_resolution))
    if last <= 0:
        raise ValueError(f"No shell at the target resolution {target_resolution} A of a {box} A box")
    fine_from = EARLY_FINE_ORDER_ANGSTROM if box / last > FINE_ORDER_LIMIT_ANGSTROM else FINE_ORDER_LIMIT_ANGSTROM

    def order(shell):
        resolution = box / shell
        if resolution > COARSE_ORDER_LIMIT_ANGSTROM:
            return 1
        return 3 if resolution <= fine_from else 2

    stages = []
    step = 0
    while not stages or stages[-1][1] < last:
        shell = min(last, _nearest_shell(box * (1 / BAND_START_ANGSTROM + step / BAND_GROWTH_ANGSTROM)))
        if not stages or (shell, order(shell)) != tuple(stages[-1][1:]):
            first = BAND_STEP_UPDATES * step
            stages.append((first + 1 if first < BAND_STEP_ALIGNED_FROM else first, shell, order(shell)))
        step += 1
    return tuple(stages)
