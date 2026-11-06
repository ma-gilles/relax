"""The ab initio PPCA versions as options: VDAM without second-moment scaling, plain SGD and the physical band rule."""

import argparse

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.commands.ppca_initial_model import add_args, resolve_sgd_options
from relax.ppca_initial_model.band_schedule import physical_band_stages
from relax.ppca_initial_model.checkpoint import _identity_config
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.iteration_loop import _update_rule
from relax.ppca_initial_model.sgd_update import momentum_step
from relax.ppca_initial_model.update import advance_moments, empty_moments, shell_gates, stochastic_update

pytestmark = pytest.mark.unit

# The measured schedule at L = 384 A (box 128 at 3 A), to Nyquist - 1; the 12.4 A schedule is its first 21 stages.
MEASURED_STAGES = (
    (1, 11, 1), (126, 12, 2), (251, 13, 2), (376, 14, 2), (501, 15, 2), (626, 16, 2), (751, 17, 2), (876, 18, 2),
    (1001, 19, 2), (1126, 20, 2), (1251, 21, 2), (1376, 22, 2), (1501, 23, 2), (1626, 24, 3), (1751, 25, 3),
    (1876, 26, 3), (2001, 27, 3), (2126, 28, 3), (2251, 29, 3), (2376, 30, 3), (2500, 31, 3), (2625, 32, 3),
    (2750, 33, 3), (2875, 34, 3), (3000, 35, 3), (3125, 36, 3), (3250, 37, 3), (3375, 38, 3), (3500, 39, 3),
    (3625, 40, 3), (3750, 41, 3), (3875, 42, 3), (4000, 43, 3), (4125, 44, 3), (4250, 45, 3), (4375, 46, 3),
    (4500, 47, 3), (4625, 48, 3), (4750, 49, 3), (4875, 50, 3), (5000, 51, 3), (5125, 52, 3), (5250, 53, 3),
    (5375, 54, 3), (5500, 55, 3), (5625, 56, 3), (5750, 57, 3), (5875, 58, 3), (6000, 59, 3), (6125, 60, 3),
    (6250, 61, 3), (6375, 62, 3), (6500, 63, 3),
)


def _args(*argv):
    parser = argparse.ArgumentParser()
    add_args(parser)
    return parser.parse_args(["--particle-diameter", "300", "-o", "out", *argv])


def test_band_rule_reproduces_the_measured_schedules():
    assert physical_band_stages(128, 3.0) == MEASURED_STAGES
    assert physical_band_stages(128, 3.0, 12.4) == MEASURED_STAGES[:21]
    # The stages pass Config's checks as they are.
    assert Config(stages=physical_band_stages(128, 3.0)).stage(1626) == (24, 3)


def test_band_rule_grows_in_physical_frequency():
    """The same frequency at the same update on any box: half the box, half the shells."""
    small, large = physical_band_stages(128, 3.0), physical_band_stages(256, 3.0)
    assert large[0] == (1, 22, 1)
    for first, shell, order in small[:-1]:
        wide = next(stage for stage in reversed(large) if stage[0] <= first)
        assert abs(wide[1] - 2 * shell) <= 2
        assert wide[0] == first
    assert large[-1][1] == 127


def test_band_rule_healpix_orders():
    def orders(stages):
        return {shell: order for _, shell, order in stages}

    # 1 above 32 A; 2 down to 16 A, not at it; 3 below.
    full = orders(physical_band_stages(128, 3.0))
    assert (full[11], full[12], full[23], full[24]) == (1, 2, 2, 3)
    # A band stopping above 16 A takes order 3 from 20 A: shell 19 is 20.2 A, shell 20 is 19.2 A.
    short = orders(physical_band_stages(128, 3.0, 18.0))
    assert max(short) == 21
    assert (short[19], short[20], short[21]) == (2, 3, 3)
    # A band stopping at 16 A does not.
    assert orders(physical_band_stages(128, 3.0, 16.0))[23] == 2


def test_band_rule_stops_at_nyquist_minus_one_and_skips_empty_steps():
    stages = physical_band_stages(32, 3.0)  # L = 96 A: a quarter of a shell per step
    assert stages[0][1:] == (3, 2) and stages[-1][1] == 15
    assert [s[1] for s in stages] == list(range(3, 16))
    assert np.all(np.diff([s[0] for s in stages]) >= 250)
    # A target past Nyquist stops at Nyquist - 1; one coarser than the start is the one stage.
    assert physical_band_stages(32, 3.0, 1.0) == stages
    assert physical_band_stages(128, 3.0, 50.0) == ((1, 8, 1),)


def test_second_moment_off_keeps_vdam_but_the_division():
    rng = np.random.default_rng(1)
    theta = jnp.asarray((rng.normal(size=(40, 3)) + 1j * rng.normal(size=(40, 3))).astype(np.complex64))
    directions = jnp.asarray((rng.normal(size=(2, 40, 3)) + 1j * rng.normal(size=(2, 40, 3))).astype(np.complex64))
    coverage = jnp.asarray(rng.random((2, 40)) > 0.2)
    shells = (np.arange(40) // 8).astype(np.int32)
    moments = advance_moments(empty_moments(theta), directions * 0.5, coverage, image_size=16)
    kwargs = dict(step=0.3, fudge=2.0, image_size=16)

    on, moments_on, info_on = stochastic_update(theta, moments, directions, coverage, shells, **kwargs)
    off, moments_off, info_off = stochastic_update(
        theta, moments, directions, coverage, shells, second_moment=False, **kwargs
    )

    # Written from the definition: the averaged first moment, gated, with no division.
    active = np.any(np.asarray(coverage), axis=0)
    first = np.asarray(moments_off.first)
    gate = np.asarray(shell_gates(theta, first, active, shells, fudge=2.0).per_frequency(shells, 2))
    theta_host = np.asarray(theta)
    expected = theta_host + 0.3 * (gate * (first[0] + first[1]) * 0.5 - (1 - gate) * theta_host)
    expected[~active] = theta_host[~active]
    assert_matches(np.asarray(off), expected.astype(np.complex64))
    assert np.max(np.abs(np.asarray(off) - np.asarray(on))) > 1e-3
    # The moments and gates advance as with the scaling.
    assert_matches(np.asarray(moments_off.second), np.asarray(moments_on.second))
    assert_matches(info_off["gates"], info_on["gates"])
    # The record that shows it: the divisor the update used.
    assert info_on["second_moment_scaling"] and not info_off["second_moment_scaling"]
    assert_matches(np.float32(info_off["direction_divisor_mean"]), np.ones(2, np.float32))
    assert np.min(np.abs(np.float32(info_on["direction_divisor_mean"]) - 1)) > 1e-3


def _sgd_problem():
    rng = np.random.default_rng(2)
    theta = (rng.normal(size=(30, 3)) + 1j * rng.normal(size=(30, 3))).astype(np.complex64)
    gradient = (rng.normal(size=(30, 3)) + 1j * rng.normal(size=(30, 3))).astype(np.complex64)
    previous = (rng.normal(size=(30, 3)) + 1j * rng.normal(size=(30, 3))).astype(np.complex64)
    trace = rng.uniform(1, 5, 30).astype(np.float32)
    active = rng.random(30) > 0.2
    return theta, previous, gradient, trace, active


def test_momentum_zero_is_plain_sgd():
    theta, previous, gradient, trace, active = _sgd_problem()
    rate = 3.25
    updated, velocity, info = momentum_step(
        theta, previous, gradient, trace, active, learning_rate=rate, floor=1e-7, momentum=0.0
    )
    expected = np.where(active[:, None], gradient, 0) * rate / np.max(trace[active])
    assert_matches(np.asarray(velocity), expected.astype(np.complex64))
    assert_matches(np.asarray(updated), (theta + expected).astype(np.complex64))
    assert info["momentum"] == 0 and info["velocity_carried_l2"] == 0


def test_default_momentum_is_the_point_nine_step():
    theta, previous, gradient, trace, active = _sgd_problem()
    _, velocity, info = momentum_step(theta, previous, gradient, trace, active, learning_rate=0.4, floor=1e-7)
    carried = 0.9 * np.where(active[:, None], previous, 0)
    expected = carried + 0.1 * 0.4 * np.where(active[:, None], gradient, 0) / np.max(trace[active])
    assert_matches(np.asarray(velocity), expected.astype(np.complex64))
    assert info["velocity_carried_l2"] > 0


def test_options_reach_the_config():
    vdam_simple = resolve_sgd_options(_args("--ios", "s.star", "--no-vdam-second-moment"))
    assert vdam_simple == {"vdam_second_moment": False}
    assert not Config(**vdam_simple).vdam_second_moment
    plain = resolve_sgd_options(
        _args("--ios", "s.star", "--optimizer", "momentum_sgd", "--sgd-momentum", "0", "--sgd-learning-rate", "3.25")
    )
    assert plain == {"sgd_learning_rate": 3.25, "sgd_momentum": 0.0}
    assert Config(optimizer="momentum_sgd", **plain).sgd_momentum == 0.0


def test_each_option_refuses_the_other_optimizer():
    with pytest.raises(ValueError, match="--sgd-momentum need --optimizer momentum_sgd"):
        resolve_sgd_options(_args("--sgd-momentum", "0"))
    with pytest.raises(ValueError, match="--no-vdam-second-moment needs --optimizer vdam"):
        resolve_sgd_options(_args("--optimizer", "momentum_sgd", "--no-vdam-second-moment"))
    with pytest.raises(ValueError, match="SGD momentum is a momentum SGD option"):
        Config(sgd_momentum=0.0)
    with pytest.raises(ValueError, match="second-moment scaling is a VDAM option"):
        Config(optimizer="momentum_sgd", vdam_second_moment=False)
    with pytest.raises(ValueError, match=r"momentum must be in \[0, 1\)"):
        Config(optimizer="momentum_sgd", sgd_momentum=1.0)


def test_defaults_stay_out_of_the_checkpoint_identity():
    """A checkpoint written before these options exists at their defaults; the options themselves change the identity."""
    identity = _identity_config(Config().__dict__)
    assert "sgd_momentum" not in identity and "vdam_second_moment" not in identity
    assert _identity_config(Config(vdam_second_moment=False).__dict__)["vdam_second_moment"] is False


def test_update_rule_lines_name_what_is_off():
    assert _update_rule(Config(vdam_second_moment=False)) == "VDAM, second-moment scaling off"
    line = _update_rule(Config(q=4, optimizer="momentum_sgd", sgd_momentum=0.0, sgd_learning_rate=3.25))
    assert line == "momentum SGD, momentum 0 (off), learning rate 3.25 = 0.65 (1 + q)"
