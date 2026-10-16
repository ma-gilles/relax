"""``ScoringVariants.from_environ``: each scoring-variant variable the command reads once, and how it is parsed.

The command reads these variables once, into ``ScoringVariants`` (code rule 5); every other test builds the
record with explicit values, usually ``helpers.run_options.cli.variants(...)`` (rule 22). These are the record's
parse tests; its guard against the unset environment is in ``test_refinement_environment.py``.
"""

import pytest

from relax.refinement.refinement_options import ScoringVariants
from relax.steps import scoring_policy

pytestmark = pytest.mark.unit


def test_local_adaptive_pass2_defaults_to_relion_pruned_parent(monkeypatch):
    monkeypatch.delenv(scoring_policy.LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, raising=False)

    def full_parent():
        return ScoringVariants.from_environ().local_adaptive_pass2.full_parent

    assert full_parent() is False

    monkeypatch.setenv(scoring_policy.LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, "1")
    assert full_parent() is True

    monkeypatch.setenv(scoring_policy.LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, "0")
    assert full_parent() is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("rotation_only", "rotation_only"),
        ("rotation", "rotation_only"),
        ("full_parent", "full_parent"),
        ("1", None),
        ("default", None),
    ],
)
def test_local_adaptive_pass2_denominator_support_mode(monkeypatch, value, expected):
    monkeypatch.setenv("RELAX_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT", value)

    assert ScoringVariants.from_environ().local_adaptive_pass2.denominator_mode == expected


def test_kclass_mstep_defaults_to_relion_x_half_with_full_volume_escape_hatch(monkeypatch):
    """K-class quality parity should use RELION x-half BPref accumulators by default."""

    monkeypatch.delenv("RELAX_K_CLASS_RELION_X_HALF_MSTEP", raising=False)
    assert ScoringVariants.from_environ().k_class_relion_x_half_mstep is True

    monkeypatch.setenv("RELAX_K_CLASS_RELION_X_HALF_MSTEP", "0")
    assert ScoringVariants.from_environ().k_class_relion_x_half_mstep is False

    monkeypatch.setenv("RELAX_K_CLASS_RELION_X_HALF_MSTEP", "1")
    assert ScoringVariants.from_environ().k_class_relion_x_half_mstep is True


def test_k1_relion_x_half_mstep_defaults_on_with_escape_hatch(monkeypatch):
    """K=1 adaptive RELION mode should use x-half BPref layout by default."""

    monkeypatch.delenv(scoring_policy.K1_RELION_X_HALF_MSTEP_ENV, raising=False)
    monkeypatch.setattr(scoring_policy, "k1_relion_x_half_mstep_default_available", lambda: True)
    assert ScoringVariants.from_environ().k1_relion_x_half_mstep is True

    monkeypatch.setenv(scoring_policy.K1_RELION_X_HALF_MSTEP_ENV, "0")
    assert ScoringVariants.from_environ().k1_relion_x_half_mstep is False

    monkeypatch.setenv(scoring_policy.K1_RELION_X_HALF_MSTEP_ENV, "1")
    assert ScoringVariants.from_environ().k1_relion_x_half_mstep is True

    monkeypatch.setenv(scoring_policy.K1_RELION_X_HALF_MSTEP_ENV, "invalid")
    assert ScoringVariants.from_environ().k1_relion_x_half_mstep is True


def test_k1_relion_x_half_mstep_default_disables_when_cuda_unavailable(monkeypatch):
    """The default must not request CUDA-only x-half adjoints on CPU tests."""

    monkeypatch.delenv(scoring_policy.K1_RELION_X_HALF_MSTEP_ENV, raising=False)
    monkeypatch.setattr(scoring_policy, "k1_relion_x_half_mstep_default_available", lambda: False)
    assert ScoringVariants.from_environ().k1_relion_x_half_mstep is False

    monkeypatch.setenv(scoring_policy.K1_RELION_X_HALF_MSTEP_ENV, "1")
    assert ScoringVariants.from_environ().k1_relion_x_half_mstep is True


@pytest.mark.parametrize(("value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_approx_acc_rot_for_convergence_is_read_once(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("RELAX_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE", raising=False)
    else:
        monkeypatch.setenv("RELAX_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE", value)
    assert ScoringVariants.from_environ().approx_acc_rot_for_convergence is expected
