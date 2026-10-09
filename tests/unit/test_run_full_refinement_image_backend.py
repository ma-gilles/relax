"""CLI contract for selecting RELION particle-image Fourier preprocessing."""

from __future__ import annotations

import os

import pytest
from helpers.tiny_main import controller_inputs, write_tiny_data_dir

from relax.refinement import command_options, particle_loading

BACKENDS = ("auto", "host_numpy", "jax_gpu", "relion_cuda")


def _parse(*arguments):
    return command_options.parse_refinement_args(["--data_dir", "data", "--output", "out", *arguments])


def test_image_fourier_backend_cli_defaults_by_job_type_with_typed_choices():
    # auto resolves to relion_cuda for K=1 and for Class3D on the resident pass 2, host_numpy for
    # Class3D on the compact engine (command_options.resolve_job_defaults).
    assert _parse().image_fourier_backend == "auto"
    assert [_parse("--image-fourier-backend", backend).image_fourier_backend for backend in BACKENDS] == list(BACKENDS)
    with pytest.raises(SystemExit):
        _parse("--image-fourier-backend", "fftw")


@pytest.mark.parametrize("backend", ["host_numpy", "relion_cuda"])
def test_image_fourier_backend_cli_is_forwarded_to_refinement(monkeypatch, tmp_path, backend):
    inputs = controller_inputs(monkeypatch, tmp_path, "refine", "--image-fourier-backend", backend)
    assert inputs["options"].parity.image_fourier_backend == backend


def test_relion_softmask_reduction_cli_has_sealed_diagnostic_choices():
    assert _parse().relion_softmask_reduction == "control"
    for mode in ("control", "native_lane", "native_atomic"):
        assert _parse("--relion-softmask-reduction", mode).relion_softmask_reduction == mode
    with pytest.raises(SystemExit):
        _parse("--relion-softmask-reduction", "lane")


@pytest.mark.parametrize("mode", ["native_lane", "native_atomic"])
def test_relion_softmask_reduction_routes_both_native_diagnostic_modes(monkeypatch, tmp_path, mode):
    """native_lane switches the particle backend's reduction; native_atomic sets RECOVAR's switch; either needs
    the RELION CUDA image backend."""
    from recovar.data_io import image_backends

    lanes = []
    monkeypatch.setattr(image_backends.ParticleImageDataset, "set_relion_native_lane_reduction",
                        lambda self, enabled: lanes.append(enabled))
    monkeypatch.delenv("RECOVAR_RELION_NATIVE_ATOMIC_SOFTMASK_REDUCTION", raising=False)
    data = write_tiny_data_dir(tmp_path / "data")
    arguments = ["--data_dir", str(data), "--output", str(tmp_path / "out"), "--relion-softmask-reduction", mode]
    os.makedirs(tmp_path / "out")
    with pytest.raises(ValueError, match="requires --image-fourier-backend relion_cuda"):
        particle_loading.load_particle_inputs(
            command_options.parse_refinement_args([*arguments, "--image-fourier-backend", "host_numpy"]),
            relion_half_sets_from_input=False,
        )
    particle_loading.load_particle_inputs(
        command_options.parse_refinement_args([*arguments, "--image-fourier-backend", "relion_cuda"]),
        relion_half_sets_from_input=False,
    )
    assert lanes == ([True] if mode == "native_lane" else [])
    atomic = os.environ.pop("RECOVAR_RELION_NATIVE_ATOMIC_SOFTMASK_REDUCTION", None)
    assert atomic == ("1" if mode == "native_atomic" else None)
