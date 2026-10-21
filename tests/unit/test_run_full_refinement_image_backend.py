"""CLI contract for selecting RELION particle-image Fourier preprocessing."""

from __future__ import annotations

import pytest
from helpers.tiny_main import controller_inputs

from relax.commands import refinement_parser

BACKENDS = ("auto", "host_numpy", "jax_gpu", "relion_cuda")


def _parse(*arguments):
    return refinement_parser.parse_refinement_args(["--data_dir", "data", "--output", "out", *arguments])


def test_image_fourier_backend_cli_defaults_by_job_type_with_typed_choices():
    # auto resolves to relion_cuda for K=1 and Class3D alike (refinement_arguments.resolve_job_defaults).
    assert _parse().image_fourier_backend == "auto"
    assert [_parse("--image-fourier-backend", backend).image_fourier_backend for backend in BACKENDS] == list(BACKENDS)
    with pytest.raises(SystemExit):
        _parse("--image-fourier-backend", "fftw")


@pytest.mark.parametrize("backend", ["host_numpy", "relion_cuda"])
def test_image_fourier_backend_cli_is_forwarded_to_refinement(monkeypatch, tmp_path, backend):
    inputs = controller_inputs(monkeypatch, tmp_path, "refine", "--image-fourier-backend", backend)
    assert inputs["options"].parity.image_fourier_backend == backend
