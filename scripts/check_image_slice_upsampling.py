"""ImageArm.slice at 2n x 2n (even pixels kept) against slicing the padded volume at n x n directly, which needs RECOVAR's
JAX projector to scale by the volume/image oversampling (branch claude/jax-projector-upsampling). Prints the max
difference and the time of both under jit+vmap (the arm's stats context).

usage: check_image_slice_upsampling.py [n] [rotations]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core import slicing
from scipy.spatial.transform import Rotation

n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
R = int(sys.argv[2]) if len(sys.argv) > 2 else 32 * 41
G = 2 * n
rng = np.random.default_rng(0)
vft = jnp.asarray((rng.standard_normal((2, G**3)) + 1j * rng.standard_normal((2, G**3))).astype(np.complex64))
rots = jnp.asarray(Rotation.random(R, random_state=1).as_matrix().astype(np.float32))
r2 = np.sum((np.indices((n, n)) - n // 2) ** 2, 0).reshape(-1)
disc = jnp.asarray((r2 <= (n // 2 - 1) ** 2).astype(np.float32))


def at_2n(v, rr):  # relax ImageArm.slice
    img = slicing.slice_volume(v, rr, (G, G), (G, G, G), "linear_interp", max_r=None)
    return img.reshape(-1, G, G)[:, ::2, ::2].reshape(-1, n * n) * disc


def at_n(v, rr):
    return slicing.slice_volume(v, rr, (n, n), (G, G, G), "linear_interp", max_r=None) * disc


out = {}
for name, f in (("slice_2n_keep_even", at_2n), ("slice_n_upsampled", at_n)):
    fn = jax.jit(jax.vmap(f, in_axes=(0, None)))
    res = jax.block_until_ready(fn(vft, rots))
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(vft, rots))
        ts.append(time.perf_counter() - t0)
    out[name] = (np.asarray(res), float(np.median(ts)))
a, b = out["slice_2n_keep_even"][0], out["slice_n_upsampled"][0]
extra = {}
if slicing.on_gpu():
    # Outside vmap the CUDA projector runs; compare it with RECOVAR's JAX path on the same oversampled volume.
    from recovar.core import relion_interp

    cu = np.asarray(slicing.slice_volume(vft[0], rots[:64], (n, n), (G, G, G), "linear_interp", max_r=None))
    jx = np.asarray(relion_interp.project(vft[0], rots[:64], (n, n), (G, G, G), order=1, max_r=None))
    extra = {"cuda_vs_jax_max_abs_diff": float(np.abs(cu - jx).max()), "cuda_max_abs": float(np.abs(cu).max())}
print(extra | {"n": n, "rotations": R, "devices": str(jax.devices()), "max_abs_diff": float(np.abs(a - b).max()),
       "max_abs": float(np.abs(a).max()), "ms_2n": 1e3 * out["slice_2n_keep_even"][1], "ms_n": 1e3 * out["slice_n_upsampled"][1]})
