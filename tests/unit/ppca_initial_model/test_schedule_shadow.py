"""Hand-derived expectations for the schedule criteria of scripts/analyze_ppca_schedule.py."""

import math

import numpy as np
import pytest

from scripts import analyze_ppca_schedule as shadow

pytestmark = pytest.mark.unit

# The ab-initio fixture: box 64 at 6 A, particle diameter 288 A.
GEOMETRY = shadow.Geometry(box=64, voxel_size=6.0, diameter_ang=288.0)


def test_relion_step_is_crowther_count():
    """24 A and a 288 A particle: ceil(pi * 12) = 38 angular steps of 360 / 38 degrees."""
    assert shadow.relion_step_deg(24.0, 288.0) == pytest.approx(360.0 / 38)
    assert shadow.relion_step_deg(48.0, 288.0) == pytest.approx(360.0 / 19)


def test_heterogeneity_step_follows_resolution_over_kappa_radius():
    """d / (kappa R) radians with R the particle radius: 24 / (3.5 * 144) rad at 24 A."""
    assert shadow.heterogeneity_step_deg(24.0, 288.0, 3.5) == pytest.approx(math.degrees(24 / (3.5 * 144)))
    assert shadow.heterogeneity_step_deg(48.0, 288.0, 3.5) == pytest.approx(
        2 * shadow.heterogeneity_step_deg(24.0, 288.0, 3.5)
    )


@pytest.mark.parametrize(
    "step, order",
    [(30.0, 1), (15.0, 2), (14.99, 3), (7.5, 3), (7.49, 4), (3.75, 4), (1.875, 5)],
)
def test_order_for_step_is_coarsest_order_at_most_the_step(step, order):
    """HEALPix steps are 60 / 2^order degrees: 30, 15, 7.5, 3.75, 1.875 for orders 1 to 5."""
    assert shadow.order_for_step(step) == order


def test_order_for_step_refuses_a_step_no_order_reaches():
    with pytest.raises(ValueError, match="at most"):
        shadow.order_for_step(0.01)


@pytest.mark.parametrize(
    "radius, relion_order, het_order",
    [(4, 1, 3), (8, 2, 4), (16, 3, 5)],
)
def test_criteria_on_fixture_geometry(radius, relion_order, het_order):
    """At 96, 48 and 24 A (shells 4, 8, 16 of box 64 at 6 A) RELION's rule needs orders 1, 2, 3 (30, 15, 7.5
    degrees against 19, 19 and 9.5 degree steps) and the heterogeneity rule orders 3, 4, 5."""
    d = shadow.resolution_ang(radius, GEOMETRY)
    assert shadow.order_for_step(shadow.relion_step_deg(d, GEOMETRY.diameter_ang)) == relion_order
    assert shadow.order_for_step(shadow.heterogeneity_step_deg(d, GEOMETRY.diameter_ang, 3.5)) == het_order


def test_gate_edge_is_the_shell_before_the_first_one_below_tau():
    gate = np.array([0.99, 0.95, 0.92, 0.4, 0.95, 0.97])
    assert shadow.gate_edge(gate, 5, 0.9) == 2
    assert shadow.gate_edge(gate, 5, 0.99) == 0  # shell 1 is already below: RELION's first-shell break
    assert shadow.gate_edge(gate, 2, 0.9) == 2  # none below within the radius


def test_last_trajectory_keeps_the_last_run_and_applies_resumes():
    lines = [{"iteration": i, "tag": "a"} for i in (1, 2, 3)]
    lines += [{"iteration": i, "tag": "b"} for i in (1, 2, 3, 4)]
    lines += [{"iteration": i, "tag": "c"} for i in (3, 4, 5)]  # a resume from iteration 3
    run = shadow.last_trajectory(lines)
    assert [(r["iteration"], r["tag"]) for r in run] == [(1, "b"), (2, "b"), (3, "c"), (4, "c"), (5, "c")]


def _record(radius, order, entropies):
    gates = np.full((radius + 1, 2), 0.95)
    gates[radius, 1] = 0.3
    return {
        "iteration": 7,
        "radius": radius,
        "healpix_order": order,
        "gates": gates.tolist(),
        "half_counts": [100, 120],
        "step": 0.9,
        "fudge": 1.0,
        "posterior": [{"pose_entropy_mean": e} for e in entropies],
    }


def test_schedule_row_reports_what_ran_and_what_each_criterion_asks():
    row = shadow.schedule_row(_record(8, 3, [4.0, 6.0]), GEOMETRY, kappa=3.5, tau=0.9)
    assert (row.iteration, row.radius, row.healpix_order, row.batch) == (7, 8, 3, 220)
    assert row.resolution_ang == pytest.approx(48.0)
    assert (row.rel_order, row.het_order) == (2, 4)
    assert row.pose_entropy == pytest.approx(5.0)
    assert (row.edge_mean, row.edge_load) == (8, 7)  # the loading gate is below tau at the edge shell only


def test_schedule_row_without_pose_entropy_is_nan():
    record = _record(4, 1, [None])
    assert math.isnan(shadow.schedule_row(record, GEOMETRY, kappa=3.5, tau=0.9).pose_entropy)
