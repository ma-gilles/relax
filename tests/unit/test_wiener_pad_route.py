"""The large-grid Wiener solve and FFTW pad run on the CPU backend when the device cannot hold the pad (relax#49)."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.reconstruction import relion_functions as rf

from relax.reconstruction import volume_solver
from relax.refinement import map_postprocess, numbered_reconstruction
from relax.refinement.refinement_options import ReconstructionPrograms

pytestmark = pytest.mark.unit

GIB = 2**30


@pytest.mark.parametrize(
    ("reconstruction_shape", "limit_gib", "on_cpu"),
    [
        ((760,) * 3, 11.24, True),  # box 380 on a 16 GB card: two 1.64 GiB halves against 2.81 GiB
        ((896,) * 3, 10.23, True),  # box 448 on a 16 GB card: two 2.69 GiB halves
        ((760,) * 3, 33.0, False),  # box 380 on a 40 GB card
        ((896,) * 3, 72.0, False),  # box 448 on an 80 GB card
        ((512,) * 3, 11.24, False),  # box 256 on a 16 GB card: two 0.5 GiB halves
    ],
)
def test_pad_moves_to_the_cpu_only_when_two_halves_exceed_the_single_working_set(
    reconstruction_shape, limit_gib, on_cpu
):
    limit = int(limit_gib * GIB)
    assert volume_solver._relion_pad_exceeds_device_working_set(reconstruction_shape, allocator_limit_bytes=limit) is on_cpu


def test_cpu_route_matches_the_device_route(monkeypatch):
    """The same recovar solve and pad on the CPU backend; on a GPU this compares across backends."""
    import recovar.core.fourier_transform_utils as ftu

    monkeypatch.setenv("RECOVAR_RELION_POSTPROCESS_LARGE_GRID_SINGLE_PRECISION", "auto")
    monkeypatch.setenv("RECOVAR_RELION_POSTPROCESS_SINGLE_PRECISION_MIN_VOXELS", "200")
    volume_shape = (4, 4, 4)
    accumulator_shape = (5, 5, 5)
    half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
    rng = np.random.default_rng(20261008)
    ft_ctf = ftu.half_volume_to_full_volume(
        jnp.asarray(rng.uniform(0.5, 1.5, half_shape).astype(np.float32)), accumulator_shape
    ).reshape(-1)
    ft_y = ftu.half_volume_to_full_volume(
        jnp.asarray((rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)),
        accumulator_shape,
    ).reshape(-1)
    common = dict(
        tau=rng.uniform(0.5, 1.5, np.prod(volume_shape)).astype(np.float64),
        tau2_fudge=1.0,
        minres_map=0,
        current_size=2,
        accumulator_volume_shape=accumulator_shape,
        tau_is_1d=False,
        preserve_output_precision=True,
        relion_filter_scale=float(volume_shape[0] ** 4),
        projection_padding_factor=1,
        use_spherical_mask=True,
        grid_correct=True,
        programs=ReconstructionPrograms.from_environ(),
    )
    calls = []
    real = rf.post_process_from_filter_v2
    monkeypatch.setattr(rf, "post_process_from_filter_v2", lambda *a, **k: calls.append(1) or real(*a, **k))

    results = {}
    for route in (False, True):
        monkeypatch.setattr(volume_solver, "_relion_pad_exceeds_device_working_set", lambda *_a, route=route, **_k: route)
        results[route] = np.asarray(numbered_reconstruction._reconstruct_volume_eager(ft_ctf, ft_y, volume_shape, 2, **common))

    assert len(calls) == 2  # both routes reach the staged pad, not another path
    assert results[True].dtype == results[False].dtype == np.complex64
    assert_matches(results[True], results[False])


@pytest.mark.parametrize("negate", [False, True])
def test_sign_overlap_cpu_route_matches_the_device_route(monkeypatch, negate):
    shape = (12, 12, 12)
    rng = np.random.default_rng(31)
    reference = jnp.asarray((rng.standard_normal(1728) + 1j * rng.standard_normal(1728)).astype(np.complex64))
    volume = reference * (-0.7 if negate else 0.7) + 0.1 * jnp.asarray(rng.standard_normal(1728).astype(np.complex64))
    results = {}
    for route in (False, True):
        monkeypatch.setattr(map_postprocess, "_sign_overlap_exceeds_device_headroom", lambda *_a, route=route: route)
        results[route] = map_postprocess._align_fourier_volume_sign_to_reference(volume, reference, shape)
    assert results[True][1] is results[False][1] is negate
    assert_matches(np.asarray(results[True][0]), np.asarray(results[False][0]))


def _fake_gpu(monkeypatch, *, limit_gib, in_use_gib, largest_block_gib):
    """The running backend as a GPU whose allocator reports these readings and places blocks up to
    ``largest_block_gib`` (the probe allocation's answer)."""
    import jax

    from relax.runtime import xla_memory_reserve

    stats = {"bytes_limit": int(limit_gib * GIB), "bytes_in_use": int(in_use_gib * GIB)}

    class Device:
        platform = "gpu"

        def memory_stats(self):
            return stats

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda *a: [Device()])
    monkeypatch.setattr(xla_memory_reserve, "_backend_pool_limit_bytes", lambda: stats["bytes_limit"])
    monkeypatch.setattr(xla_memory_reserve, "device_can_place", lambda n: n <= largest_block_gib * GIB)


def test_device_fits_is_the_share_the_headroom_and_a_placement(monkeypatch):
    from relax.runtime.xla_memory_reserve import device_fits

    # A 16 GB card at box 448 after a K=1 M-step: limit 10.19 GiB, 8.4 GiB in use.
    _fake_gpu(monkeypatch, limit_gib=10.19, in_use_gib=8.4, largest_block_gib=1.5)
    assert device_fits(int(0.5 * GIB)) is True
    assert device_fits(int(1.0 * GIB)) is False  # twice it exceeds the 1.79 GiB headroom
    _fake_gpu(monkeypatch, limit_gib=10.19, in_use_gib=4.0, largest_block_gib=1.5)
    assert device_fits(int(2.01 * GIB)) is False  # the allocator cannot place it (the fragmented-pool failure)
    assert device_fits(int(2.6 * GIB)) is False  # above a quarter of the limit
    _fake_gpu(monkeypatch, limit_gib=10.19, in_use_gib=4.0, largest_block_gib=3.0)
    assert device_fits(int(2.01 * GIB)) is True


def test_sign_overlap_and_pad_on_an_h100_at_box_800_stay_on_the_device(monkeypatch):
    """EMPIAR-10202 on an 80 GB H100 (texture-reserve limit 61.5 GiB): the overlap's 11.4 GiB fits with 30 GiB free;
    box 448 on a 16 GB card that cannot place 2 GiB goes to the CPU."""

    _fake_gpu(monkeypatch, limit_gib=61.5, in_use_gib=31.5, largest_block_gib=30.0)
    assert map_postprocess._sign_overlap_exceeds_device_headroom((800,) * 3) is False
    _fake_gpu(monkeypatch, limit_gib=10.19, in_use_gib=5.2, largest_block_gib=1.5)
    assert map_postprocess._sign_overlap_exceeds_device_headroom((448,) * 3) is True
    _fake_gpu(monkeypatch, limit_gib=10.19, in_use_gib=3.0, largest_block_gib=4.0)
    assert map_postprocess._sign_overlap_exceeds_device_headroom((448,) * 3) is False


def test_device_can_place_catches_only_out_of_memory(monkeypatch):
    import jax
    from jax.errors import JaxRuntimeError

    from relax.runtime import xla_memory_reserve

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    def refuse(*_a, **_k):
        raise JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.01GiB.")

    monkeypatch.setattr(jax.numpy, "zeros", refuse)
    assert xla_memory_reserve.device_can_place(2 * GIB) is False

    def broken(*_a, **_k):
        raise JaxRuntimeError("INTERNAL: something else")

    monkeypatch.setattr(jax.numpy, "zeros", broken)
    with pytest.raises(JaxRuntimeError, match="INTERNAL"):
        xla_memory_reserve.device_can_place(2 * GIB)


def test_device_fits_skips_the_probe_well_inside_the_bounds(monkeypatch):
    from relax.runtime import xla_memory_reserve

    _fake_gpu(monkeypatch, limit_gib=72.0, in_use_gib=10.0, largest_block_gib=0.0)
    probes = []
    monkeypatch.setattr(xla_memory_reserve, "device_can_place", lambda n: probes.append(n) or False)
    assert xla_memory_reserve.device_fits(int(2 * GIB)) is True  # 2.8% of the limit, 62 GiB free: no probe
    assert probes == []
    assert xla_memory_reserve.device_fits(int(5 * GIB)) is False  # near the boundary: probed, and refused
    assert probes == [int(5 * GIB)]


_PROBE_CHILD = """
import jax, jax.numpy as jnp
from relax.runtime.xla_memory_reserve import device_can_place
x = jnp.ones((1 << 20,), jnp.float32).block_until_ready()
stats = jax.devices()[0].memory_stats()
before = stats["bytes_in_use"]
small = device_can_place(1 << 26)
huge = device_can_place(int(stats["bytes_limit"]) + (1 << 30))
after = jax.devices()[0].memory_stats()["bytes_in_use"]
print("PROBE", small, huge, before == after)
"""


@pytest.mark.gpu
@pytest.mark.parametrize("preallocate", ["true", "false"])
def test_device_can_place_answers_on_the_device_and_leaves_nothing(preallocate):
    """The probe on a real allocator, with and without preallocation: a small block places, a block larger than the
    pool limit does not, and the allocator's bytes in use are unchanged afterwards."""
    import os
    import subprocess

    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU")
    from conftest import repo_python_command, repo_subprocess_env

    env = repo_subprocess_env(dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE=preallocate))
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
    proc = subprocess.run(repo_python_command("-c", _PROBE_CHILD), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    (line,) = [line for line in proc.stdout.splitlines() if line.startswith("PROBE")]
    assert line.split()[1:] == ["True", "False", "True"], line


def test_low_resolution_join_goes_to_the_host_when_its_copies_do_not_fit(monkeypatch):
    from relax.reconstruction import regularization

    monkeypatch.delenv("RELAX_LOWRES_JOIN_HOST_FALLBACK", raising=False)
    seen = []
    monkeypatch.setattr(regularization, "device_fits", lambda nbytes: seen.append(nbytes) or False)
    assert regularization._low_resolution_join_host_fallback_enabled_for_size(1000, 10, itemsize=8) is True
    assert seen == [2 * 1000 * 8]  # both joined copies of the half pair
    monkeypatch.setattr(regularization, "device_fits", lambda nbytes: True)
    assert regularization._low_resolution_join_host_fallback_enabled_for_size(1000, 10) is False
    # A full join (every voxel) never takes the host path; physically large grids always do.
    assert regularization._low_resolution_join_host_fallback_enabled_for_size(1000, 1000) is False
    assert regularization._low_resolution_join_host_fallback_enabled_for_size(300_000_000, 10) is True
