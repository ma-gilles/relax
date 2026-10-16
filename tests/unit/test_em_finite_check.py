"""Unit tests for the P4-D pass-2 finite and bound checks.

The checks exist because job 14178118's compact control overflowed its BPref
accumulators from operands that are provably bounded, and because a
large-but-finite corruption raises nothing until the accumulated value reaches
infinity. Each test below pins one of the properties that makes the checks
useful: they are silent unless asked for, they name the first offending field
and index, they fire on a bound that arithmetic cannot break, and they can be
asked to record every occurrence instead of dying on the first.
"""

import logging

import numpy as np
import pytest

from relax.numerics import finite_check
from relax.runtime.engine_tuning import FINITE_CHECK_WARN_ENV


def test_check_arrays_names_the_first_offender_and_its_operands():
    clean = np.array([3.0, 4.0])
    poisoned = np.array([1.0, 2.0, np.nan, np.inf])
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_arrays(
            "mstep-sums",
            {"clean": clean, "poisoned": poisoned},
            warn_only=False,
            context="half=2 bucket=17",
            operands={"operand": clean},
        )
    message = str(excinfo.value)
    assert "mstep-sums" in message and "half=2 bucket=17" in message
    assert "first offender: poisoned at flat index 2" in message
    assert "nonfinite=2/4" in message
    assert "operand operand" in message


def test_check_per_image_names_the_offending_image_ids():
    statistic = np.array([1.0, np.nan, 3.0, np.inf])
    ids = np.array([100, 101, 102, 103])
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_per_image(
            "norm", {"wsum_norm_correction": statistic}, warn_only=False, image_ids=ids,
        )
    message = str(excinfo.value)
    assert "2/4 non-finite" in message
    assert "[101, 103]" in message


def test_posterior_bound_passes_a_pruned_posterior_and_fails_a_broken_one():
    # RELION prunes to the adaptive fraction, so a healthy image total sits
    # just under one and must not be reported.
    pruned = np.array([[0.5, 0.3, 0.199], [0.999, 0.0, 0.0]], dtype=np.float32)
    assert finite_check.check_posterior_bounds("p", pruned, warn_only=False, image_ids=np.array([7, 8])) is None

    # A stale or uninitialised padded row shows up here: the total is taken
    # over every row and translation of the full array.
    broken = pruned.copy()
    broken[1, 1] = 0.5
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_posterior_bounds("p", broken, warn_only=False, image_ids=np.array([7, 8]))
    assert "image ids [8]" in str(excinfo.value)


def test_reduction_bound_fires_on_a_sum_larger_than_its_operand():
    operand = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    within = {"summed": operand * 0.9, "shifted": operand}
    bound = (("summed", "shifted", 1e-3),)
    assert finite_check.check_bundle("m", within, warn_only=False, dominated_by=bound) is None

    beyond = {"summed": operand * 1.5, "shifted": operand}
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_bundle("m", beyond, warn_only=False, dominated_by=bound)
    assert "max |summed| = 6" in str(excinfo.value)


def test_operand_bound_uses_the_product_of_several_factors():
    ctf = np.array([[0.5, -0.8], [0.9, 0.2]], dtype=np.float32)
    inverse_noise = np.array([1e-4, 2e-4], dtype=np.float32)
    weighted = ctf * inverse_noise[None, :]
    bound = (("weighted", ("ctf", "inverse_noise"), 1e-3),)
    arrays = {"ctf": ctf, "inverse_noise": inverse_noise, "weighted": weighted}
    assert finite_check.check_bundle("io", arrays, warn_only=False, dominated_by=bound) is None

    arrays["weighted"] = weighted.copy()
    arrays["weighted"][1, 1] = np.float32(1e3)
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_bundle("io", arrays, warn_only=False, dominated_by=bound)
    message = str(excinfo.value)
    assert "max |ctf| * max |inverse_noise|" in message


def test_warn_mode_records_every_occurrence_instead_of_dying(caplog):
    poisoned = np.array([1.0, np.inf])
    with caplog.at_level(logging.ERROR, logger=finite_check.__name__):
        first = finite_check.check_arrays("s", {"a": poisoned}, warn_only=True)
        second = finite_check.check_posterior_bounds("p", np.array([[2.0]], dtype=np.float32), warn_only=True)
    assert first is not None and second is not None
    assert len(caplog.records) == 2


def test_half_accumulator_guard_is_on_by_default_and_names_the_half():
    """The cheap once-per-iteration guard runs in production unless opted out."""
    from relax.refinement.refinement_options import RefinementEnvironment

    assert RefinementEnvironment().half_accumulator_guard == "raise"

    poisoned = {"Ft_ctf_1": np.array([1.0, np.inf]), "Ft_y_0": np.array([1.0, 2.0])}
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_half_accumulators(poisoned, mode="raise", context="relion_iteration=15")
    message = str(excinfo.value)
    assert "relion_iteration=15" in message
    assert "Ft_ctf_1" in message and "nonfinite=1/2" in message
    # The clean half is not named, so the report points at the damage.
    assert "Ft_y_0" not in message
    # Clean accumulators cost one reduction each and report nothing.
    assert finite_check.check_half_accumulators({"Ft_ctf_1": np.array([1.0, 2.0])}, mode="raise") is None


def test_half_accumulator_guard_opt_out_and_warn_modes(caplog):
    """The ``off`` mode checks nothing; ``warn`` reports and keeps going."""
    poisoned = {"Ft_ctf_1": np.array([1.0, np.inf])}
    assert finite_check.check_half_accumulators(poisoned, mode="off") is None
    with caplog.at_level(logging.ERROR, logger=finite_check.__name__):
        message = finite_check.check_half_accumulators(poisoned, mode="warn", context="relion_iteration=15")
    assert message is not None and "Ft_ctf_1" in message and "relion_iteration=15" in message
    assert len(caplog.records) == 1


def test_the_command_reads_the_guard_mode_into_the_options(monkeypatch, tmp_path):
    """``relax refine`` reads RELAX_EM_BPREF_FINITE_GUARD once, into ``options.environment``."""
    from helpers.tiny_main import controller_inputs

    from relax.reconstruction.accumulator_guard import HALF_ACCUMULATOR_GUARD_ENV

    monkeypatch.delenv(FINITE_CHECK_WARN_ENV, raising=False)
    monkeypatch.setenv(HALF_ACCUMULATOR_GUARD_ENV, "warn")
    inputs = controller_inputs(monkeypatch, tmp_path, "refine")
    assert inputs["options"].environment.half_accumulator_guard == "warn"


def test_the_controller_hands_the_options_guard_mode_to_the_guard(monkeypatch):
    """The controller passes ``options.environment``'s mode, not the environment, to the pre-join guard."""
    from helpers.tiny_refinement import run_tiny_refinement

    from relax.reconstruction import accumulator_guard
    from relax.reconstruction.accumulator_guard import HALF_ACCUMULATOR_GUARD_ENV
    from relax.refinement.refinement_options import RefinementEnvironment

    monkeypatch.delenv(HALF_ACCUMULATOR_GUARD_ENV, raising=False)
    modes = []
    monkeypatch.setattr(
        accumulator_guard.finite_check, "check_half_accumulators",
        lambda accumulators, *, mode, context: modes.append(mode),
    )
    run_tiny_refinement(
        monkeypatch, max_iter=1, final_after_max_iter=False,
        environment=RefinementEnvironment(half_accumulator_guard="warn"),
    )
    assert modes == ["warn"]
