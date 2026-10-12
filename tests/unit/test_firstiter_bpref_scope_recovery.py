"""Final-Q native first-iteration scope guards at the current owner."""
import pytest

from relax.refinement.firstiter_bpref import _relion_firstiter_fused_bpref_enabled


def test_firstiter_fused_bpref_defaults_only_inside_complete_fresh_k1_guard(monkeypatch):
    monkeypatch.delenv("RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", raising=False)
    kwargs = dict(
        fresh_k1_guard=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        relion_exact_bpref_operands=True,
        use_relion_x_half_mstep=True,
        score_only=False,
        first_numbered_iteration=True,
    )

    assert _relion_firstiter_fused_bpref_enabled(**kwargs)
    assert not _relion_firstiter_fused_bpref_enabled(
        **{**kwargs, "winner_take_all": False}
    )
    # Only the run's first numbered iteration takes the fused kernel.
    assert not _relion_firstiter_fused_bpref_enabled(**{**kwargs, "first_numbered_iteration": False})

def test_firstiter_fused_bpref_override_can_disable_but_not_expand_scope(monkeypatch):
    kwargs = dict(
        fresh_k1_guard=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        relion_exact_bpref_operands=True,
        use_relion_x_half_mstep=True,
        score_only=False,
        first_numbered_iteration=True,
    )
    monkeypatch.setenv("RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", "0")
    assert not _relion_firstiter_fused_bpref_enabled(**kwargs)

    monkeypatch.setenv("RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", "1")
    with pytest.raises(ValueError, match="requires the fresh K=1"):
        _relion_firstiter_fused_bpref_enabled(
            **{**kwargs, "fresh_k1_guard": False}
        )
