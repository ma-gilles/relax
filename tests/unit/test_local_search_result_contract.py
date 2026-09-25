"""Optional local-search outputs retain their meanings."""

from types import SimpleNamespace

import numpy as np
import pytest

from relax.helpers.types import LocalEMResult
from relax.refinement import local_search_iteration

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("return_profile", [False, True])
def test_local_sample_capture_preserves_profile_visibility(monkeypatch, return_profile):
    """Sample capture enables an internal profile even if the caller hides it."""
    profile = {"reconstruction_sample_indices_by_image": (np.array([1]), np.array([2]))}
    stats = object()

    def run_local(*args, **kwargs):
        assert kwargs["return_reconstruction_sample_indices"] is True
        assert kwargs["return_profile"] == return_profile
        assert "return_significant_counts" not in kwargs
        return LocalEMResult(
            Ft_y=np.zeros(8, dtype=np.complex64),
            Ft_ctf=np.ones(8, dtype=np.float32),
            hard_assignments=np.array([0, 1], dtype=np.int32),
            stats=stats,
            profile=profile,
        )

    monkeypatch.setattr(local_search_iteration, "run_local_em_exact", run_local)
    monkeypatch.setattr(
        local_search_iteration, "_estimate_relion_em_batch_sizes",
        lambda **kwargs: SimpleNamespace(
            image_batch_size=kwargs["requested_image_batch_size"],
            rotation_block_size=kwargs["requested_rotation_block_size"],
        ),
    )
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    translations = np.zeros((2, 2), dtype=np.float32)
    result = local_search_iteration._run_local_search_iteration(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2)),
        None, None, rotations, rotations,
        healpix_order=0, sigma_rot=1.0, sigma_psi=1.0,
        translations=translations[:1], prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp", image_batch_size=2, rotation_block_size=1,
        current_size=2,
        pass2_layout=SimpleNamespace(rotation_counts=np.ones(2, dtype=np.int32), translation_grid=translations[:1]),
        return_reconstruction_sample_indices=True,
        return_profile=return_profile,
    )
    assert result.relion_stats is stats
    if return_profile:
        assert result.profile_summary is not profile
        assert result.profile_summary["reconstruction_sample_indices_by_image"] is profile["reconstruction_sample_indices_by_image"]
    else:
        assert result.profile_summary is None
    assert set(profile) == {"reconstruction_sample_indices_by_image"}
