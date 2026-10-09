"""Scientific signal selection and startup precedence in image-size planning."""

from unittest.mock import Mock

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.run_options import stand_in

from relax.refinement.iteration_planning import (
    plan_class_image_size,
    plan_halfmap_image_size,
    plan_initial_image_size,
)
from relax.refinement.refinement_options import StartState

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["production-f32", "diagnostic-f64"])
def test_initial_ini_high_precedes_fsc_and_retains_growth(dtype):
    plan = plan_initial_image_size(
        stand_in.options(
            schedule=stand_in.schedule(init_current_size=32), start=StartState(init_fsc=np.zeros(65, dtype=dtype)),
            parity=stand_in.parity(relion_firstiter_ini_high_angstrom=30.),
        ),
        box_size=128, pixel_size_angstrom=4.25, incr_size=16,
        has_high_fsc_at_limit=True, dtype=dtype, log=Mock(),
    )
    # ini_high sets shell 18; this startup path adds the configured 16 shells.
    assert plan.size == 68
    assert plan.data_vs_prior is None
    assert plan.incr_size == 16 and plan.has_high_fsc_at_limit


@pytest.mark.parametrize("restart_iteration", [0, 2])
def test_initial_fallback_retains_its_distinct_default_growth(restart_iteration):
    plan = plan_initial_image_size(
        stand_in.options(schedule=stand_in.schedule(init_current_size=32, init_relion_iteration=restart_iteration)),
        box_size=128, pixel_size_angstrom=4.25,
        incr_size=16, has_high_fsc_at_limit=True, dtype=np.float32, log=Mock(),
    )
    # Without ini_high/FSC, the existing bootstrap uses its own ten-shell default.
    assert plan.size == 52
    assert plan.data_vs_prior is None
    assert plan.incr_size == 16 and plan.has_high_fsc_at_limit


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["production-f32", "diagnostic-f64"])
def test_initial_fsc_updates_growth_and_scheduling_curve(dtype):
    fsc = np.full(65, .3, dtype=dtype)
    fsc[:10] = .8
    original = fsc.tobytes()
    plan = plan_initial_image_size(
        stand_in.options(
            schedule=stand_in.schedule(init_current_size=40), start=StartState(init_fsc=fsc, init_ave_Pmax=.2),
        ),
        box_size=128, pixel_size_angstrom=4.25,
        incr_size=6, has_high_fsc_at_limit=False, dtype=dtype, log=Mock(),
    )
    # Truncation at shell 20 makes incr_size 21 - 10 + 5 = 16.
    # The high-limit latch selects the 16-shell aggressive jump from shell 9.
    assert plan.size == 50
    assert plan.incr_size == 16 and plan.has_high_fsc_at_limit
    assert plan.data_vs_prior.shape == (65,) and plan.data_vs_prior.dtype == dtype
    assert_matches(plan.data_vs_prior[:10], np.full(10, 4, dtype=dtype))
    assert_matches(plan.data_vs_prior[10:21], np.full(11, 3 / 7, dtype=dtype))
    assert fsc.tobytes() == original


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["production-f32", "diagnostic-f64"])
def test_halfmap_resolution_and_growth_use_their_separate_signals(dtype):
    raw_fsc = np.full(65, .3, dtype=dtype)
    raw_fsc[:10] = .8
    growth_fsc = np.full(65, .05, dtype=dtype)
    corrected_prior = np.full(65, .5, dtype=dtype)
    corrected_prior[:12] = 2
    originals = [a.tobytes() for a in (raw_fsc, growth_fsc, corrected_prior)]
    plan = plan_halfmap_image_size(
        [raw_fsc], growth_fsc_history=[growth_fsc], restart=None, data_vs_prior=corrected_prior,
        previous_size=40, box_size=128, pixel_size_angstrom=4.25,
        incr_size=3, has_high_fsc_at_limit=False, ave_pmax=.9,
        completed_relion_iteration=2, parity=stand_in.parity(), dtype=dtype, log=Mock(),
    )
    assert plan.resolution_shell == 11
    assert plan.incr_size == 5 and not plan.has_high_fsc_at_limit
    assert plan.size == plan.raw_size == 32
    assert_matches(plan.data_vs_prior[:21], corrected_prior[:21])
    assert_matches(plan.data_vs_prior[21:], np.zeros(44, dtype=dtype))
    assert [a.tobytes() for a in (raw_fsc, growth_fsc, corrected_prior)] == originals


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["production-f32", "diagnostic-f64"])
@pytest.mark.parametrize("n_classes", [2, 4])
def test_class_image_size_uses_best_class_without_high_resolution_recovery(dtype, n_classes):
    curves = np.full((n_classes, 65), .5, dtype=dtype)
    curves[:, :8] = 2
    curves[-1, :13] = 2
    curves[-1, 19] = 2  # A separated high-shell spike must not trigger K1 recovery.
    original = curves.tobytes()
    plan = plan_class_image_size(
        curves, previous_size=40, box_size=128, pixel_size_angstrom=4.25,
        incr_size=6, ave_pmax=.9, completed_relion_iteration=2,
        parity=stand_in.parity(), dtype=dtype, log=Mock(),
    )
    assert plan.resolution_shell == 12
    assert plan.size == plan.raw_size == 36
    assert plan.resolution_shells_per_class.dtype == np.int32
    np.testing.assert_array_equal(plan.resolution_shells_per_class, [7] * (n_classes - 1) + [12])
    assert_matches(plan.raw_data_vs_prior, curves)
    assert_matches(plan.data_vs_prior[:, :21], curves[:, :21])
    assert_matches(plan.data_vs_prior[:, 21:], np.zeros((n_classes, 44), dtype=dtype))
    assert curves.tobytes() == original


@pytest.mark.parametrize("completed_iteration", [0, 1, 2])
def test_firstiter_cc_override_uses_the_completed_iteration(completed_iteration):
    curves = np.full((4, 65), .5, dtype=np.float32)
    curves[:, :13] = 2
    plan = plan_class_image_size(
        curves, previous_size=40, box_size=128, pixel_size_angstrom=4.25,
        incr_size=6, ave_pmax=.9, completed_relion_iteration=completed_iteration,
        parity=stand_in.parity(emulate_relion_firstiter_cc=True, relion_firstiter_ini_high_angstrom=30.),
        dtype=np.float32, log=Mock(),
    )
    assert plan.resolution_shell == (18 if completed_iteration == 1 else 12)
    assert plan.size == (48 if completed_iteration == 1 else 36)
    np.testing.assert_array_equal(plan.resolution_shells_per_class, [12] * 4)
