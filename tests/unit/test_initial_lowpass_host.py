"""The first-CC initial low-pass transforms on the host and puts only the filtered map on the device (relax#47)."""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import mean_helpers
from relax.relion.reference_initialization import initial_low_pass_filter_references

pytestmark = pytest.mark.unit

VOLUME_SHAPE = (16, 16, 16)


def _device_lowpass(volume_ft_flat, voxel_size, ini_high_angstrom, filter_edgewidth):
    """The earlier formulation: both transforms on the default device, the forward one in float64."""
    ftu = mean_helpers.fourier_transform_utils
    original = jnp.asarray(volume_ft_flat).reshape(VOLUME_SHAPE)
    volume_real = np.real(np.asarray(ftu.get_idft3(original))).astype(np.float64)
    filtered_real = initial_low_pass_filter_references(
        volume_real[None, ...],
        box_size=VOLUME_SHAPE[0],
        pixel_size=voxel_size,
        ini_high_ang=ini_high_angstrom,
        filter_edgewidth=filter_edgewidth,
    )[0]
    return np.asarray(ftu.get_dft3(jnp.asarray(filtered_real)).astype(original.dtype).reshape(-1))


def test_host_lowpass_matches_the_device_formulation():
    rng = np.random.default_rng(0)
    size = int(np.prod(VOLUME_SHAPE))
    volume = jnp.asarray((rng.standard_normal(size) + 1j * rng.standard_normal(size)).astype(np.complex64))
    expected = _device_lowpass(volume, 2.0, 8.0, 2.0)

    result = mean_helpers._apply_relion_initial_lowpass_filter(volume, VOLUME_SHAPE, 2.0, 8.0, filter_edgewidth=2.0)

    assert result.dtype == volume.dtype and result.shape == volume.shape
    assert result.devices() == {jax.devices()[0]}
    assert_matches(np.asarray(result), expected)
    assert not np.allclose(expected, np.asarray(volume))


_PEAK_CHILD = """
import jax, jax.numpy as jnp, numpy as np
from relax.refinement import mean_helpers
box = 256
rng = np.random.default_rng(0)
volume = jnp.asarray((rng.standard_normal(box**3) + 1j * rng.standard_normal(box**3)).astype(np.complex64))
volume.block_until_ready()
device = jax.devices()[0]
before = device.memory_stats()["bytes_in_use"]
out = mean_helpers._apply_relion_initial_lowpass_filter(volume, (box,) * 3, 1.4, 60.0, filter_edgewidth=2.0)
out.block_until_ready()
print("EXTRA_PEAK_MAPS", (device.memory_stats()["peak_bytes_in_use"] - before) / (box**3 * 8))
"""


@pytest.mark.gpu
def test_lowpass_device_high_water_is_the_filtered_map():
    """Measured on an A100 at box 256: 1.0 complex64 map (the result); 7.0 with both transforms on the device."""
    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU")
    from conftest import repo_python_command, repo_subprocess_env

    env = repo_subprocess_env()
    # A fresh process with the default BFC allocator, so peak_bytes_in_use is this call's high-water.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("TF_GPU_ALLOCATOR", None)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    proc = subprocess.run(repo_python_command("-c", _PEAK_CHILD), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    (line,) = [line for line in proc.stdout.splitlines() if line.startswith("EXTRA_PEAK_MAPS")]
    assert 0.0 < float(line.split()[1]) <= 1.5, line
