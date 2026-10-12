"""The K=1 Refine3D entry points turn on the stage glue by default."""

from relax.commands.refinement_arguments import K1_REFINE3D_ENV_DEFAULTS, apply_k1_refine3d_env_defaults
from relax.relion.image_preprocessing import jit_stage_glue_enabled


def test_k1_refine3d_defaults_turn_on_glue(monkeypatch):
    for name in K1_REFINE3D_ENV_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    # The shared library default stays off (VDAM qualifies its own).
    assert not jit_stage_glue_enabled()

    apply_k1_refine3d_env_defaults()
    assert jit_stage_glue_enabled()


def test_k1_refine3d_defaults_keep_an_explicit_value(monkeypatch):
    monkeypatch.setenv("RELAX_EM_JIT_STAGE_GLUE", "0")
    apply_k1_refine3d_env_defaults()
    assert not jit_stage_glue_enabled()
