"""The K=1 Refine3D entry points turn on the stage glue and the local ladder by default."""

from relax.helpers.preprocessing import jit_stage_glue_enabled
from relax.local.local_layout import DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER, resolve_local_image_capacity_ladder
from relax.refinement.refinement_options import K1_REFINE3D_ENV_DEFAULTS, apply_k1_refine3d_env_defaults


def test_k1_refine3d_defaults_turn_on_glue_and_ladder(monkeypatch):
    for name in K1_REFINE3D_ENV_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    # The shared library defaults stay off (VDAM qualifies its own).
    assert not jit_stage_glue_enabled()
    assert resolve_local_image_capacity_ladder() == ()

    apply_k1_refine3d_env_defaults()
    assert jit_stage_glue_enabled()
    assert resolve_local_image_capacity_ladder() == DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER


def test_k1_refine3d_defaults_keep_an_explicit_value(monkeypatch):
    monkeypatch.setenv("RELAX_EM_JIT_STAGE_GLUE", "0")
    monkeypatch.setenv("RELAX_LOCAL_IMAGE_CAPACITY_LADDER", "8,64")
    apply_k1_refine3d_env_defaults()
    assert not jit_stage_glue_enabled()
    assert resolve_local_image_capacity_ladder() == (8, 64)
