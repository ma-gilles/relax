"""Class3D stack postprocessing in place: the first-CC low-pass and the solvent flatten (relax#45)."""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import mean_helpers

pytestmark = pytest.mark.unit

N_CLASSES = 3
VOLUME_SHAPE = (12, 12, 12)


def _class_stack(seed):
    rng = np.random.default_rng(seed)
    size = int(np.prod(VOLUME_SHAPE))
    values = rng.standard_normal((N_CLASSES, size)) + 1j * rng.standard_normal((N_CLASSES, size))
    return values.astype(np.complex64)


def test_flatten_class_stack_matches_each_class_flattened_alone():
    stack = _class_stack(0)
    solvent_mask = jnp.asarray(np.random.default_rng(1).random(VOLUME_SHAPE).astype(np.float32))
    # Reference: the K=1 flatten applied to each class on its own, then stacked (the pre-#45 shape of the code).
    expected = np.stack(
        [
            np.asarray(
                mean_helpers._apply_relion_solvent_flatten_k1(
                    jnp.asarray(stack[k]),
                    solvent_mask,
                    VOLUME_SHAPE,
                )
            )
            for k in range(N_CLASSES)
        ]
    )

    result = mean_helpers._flatten_class_stack(jnp.asarray(stack), solvent_mask, VOLUME_SHAPE, N_CLASSES)

    assert result.shape == stack.shape and result.dtype == stack.dtype
    assert_matches(np.asarray(result), expected)


def test_lowpass_class_stack_matches_each_class_filtered_alone():
    stack = _class_stack(2)
    settings = mean_helpers.ReconstructionSettings(
        box_size=VOLUME_SHAPE[0],
        voxel_size=2.0,
        volume_shape=VOLUME_SHAPE,
        padding_factor=2,
        projection_padding_factor=1,
        minres_map=0,
        width_mask_edge=5,
        fmask_edge=2,
        tau2_fudge=1,
        particle_diameter_angstrom=None,
        first_iteration_lowpass_angstrom=8.0,
    )
    expected = np.stack(
        [
            np.asarray(
                mean_helpers._apply_relion_initial_lowpass_filter(
                    jnp.asarray(stack[k]),
                    VOLUME_SHAPE,
                    2.0,
                    8.0,
                    filter_edgewidth=2,
                )
            )
            for k in range(N_CLASSES)
        ]
    )

    result = mean_helpers._lowpass_class_stack(jnp.asarray(stack), settings, N_CLASSES)

    assert result.shape == stack.shape and result.dtype == stack.dtype
    assert_matches(np.asarray(result), expected)
    # The filter is not the identity on this stack, so the comparison above checks the written rows.
    assert not np.allclose(expected, stack)


_PEAK_CHILD = """
import jax, jax.numpy as jnp, numpy as np
from relax.refinement import mean_helpers
box, n_classes = 256, 3
shape = (box,) * 3
rng = np.random.default_rng(0)
stack = jnp.asarray((rng.standard_normal((n_classes, box**3)) + 1j * rng.standard_normal((n_classes, box**3)))
                    .astype(np.complex64))
mask = jnp.asarray(rng.random(shape).astype(np.float32))
stack.block_until_ready(); mask.block_until_ready()
device = jax.devices()[0]
before = device.memory_stats()["bytes_in_use"]
out = mean_helpers._flatten_class_stack(stack, mask, shape, n_classes)
out.block_until_ready()
stats = device.memory_stats()
print("EXTRA_PEAK_CLASS_MAPS", (stats["peak_bytes_in_use"] - before) / (box**3 * 8))
"""


@pytest.mark.gpu
def test_flatten_class_stack_pool_high_water_is_about_two_class_maps():
    """On the device the flatten adds about two class maps to the pool, not copies of the stack.

    Measured on an A100 at box 256, K=3: 2.0 class maps for the per-class program; 5.0 for the same statements
    run eagerly and 11.8 for the old list-then-stack (relax#45, which ran a 16 GB card out of memory at box 380).
    The bound sits between the program and the eager statements.
    """
    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU")
    from conftest import repo_python_command, repo_subprocess_env

    env = repo_subprocess_env()
    # A fresh process with the default BFC allocator, so peak_bytes_in_use is this step's high-water.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("TF_GPU_ALLOCATOR", None)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    proc = subprocess.run(repo_python_command("-c", _PEAK_CHILD), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    (line,) = [line for line in proc.stdout.splitlines() if line.startswith("EXTRA_PEAK_CLASS_MAPS")]
    extra_maps = float(line.split()[1])
    assert 0.0 < extra_maps <= 3.0, line
