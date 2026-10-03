"""Local expectation requires complete pose state instead of global fallback."""

import pytest

from relax.refinement.half_inputs import HalfSet

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("missing_field", ["rotation_eulers", "translations"])
@pytest.mark.parametrize("index", [0, 1])
def test_local_pose_boundary_rejects_missing_half(missing_field, index):
    half = HalfSet(index=index, dataset=object(), rotation_eulers=object(), translations=object())
    setattr(half, missing_field, None)
    with pytest.raises(ValueError, match=f"half {index + 1}"):
        half.require_local_search_poses()


def test_local_pose_boundary_preserves_complete_state():
    eulers, translations = object(), object()
    half = HalfSet(index=0, dataset=object(), rotation_eulers=eulers, translations=translations)
    half.require_local_search_poses()
    assert half.rotation_eulers is eulers
    assert half.translations is translations
