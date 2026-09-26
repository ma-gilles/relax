"""Native firstiter contracts retained from Q 72e785f3f."""
from pathlib import Path
import numpy as np
import pytest
import jax.numpy as jnp
import recovar.cuda_backproject as cuda_backproject
from relax.cuda import kernels as em_cuda_kernels
from helpers.float_compare import assert_matches
pytestmark = pytest.mark.unit

def test_relion_firstiter_bpref_cuda_source_preserves_native_interface_and_pass_loop():
    cuda_source = (
        Path(__file__).resolve().parents[2] / "relax" / "cuda" / "relax_kernels.cu"
    )
    text = cuda_source.read_text()
    kernel_start = text.index("relion_firstiter_bpref_fused_x_half_kernel(")
    declaration_start = text.rfind("__global__", 0, kernel_start)
    kernel_end = text.index("/* ================================================================== */", kernel_start)
    kernel = text[declaration_start:kernel_end]

    assert "__launch_bounds__" not in kernel[: kernel.index("{")]
    assert "const int pixel_pass_num = (int)ceilf((float)img_xyz / 128.0f);" in kernel
    assert "for (unsigned pass = 0; pass < (unsigned)pixel_pass_num; ++pass)" in kernel
    assert "const unsigned pixel = pass * 128U + tid;" in kernel
    assert "pixel += 128" not in kernel
    assert "const float* __restrict__ image_real" in kernel
    assert "const float* __restrict__ image_imag" in kernel
    assert "const float* __restrict__ translation_x" in kernel
    assert "const float* __restrict__ translation_y" in kernel
    assert "float* __restrict__ model_real" in kernel
    assert "float* __restrict__ model_imag" in kernel
    assert "unsigned long translation_num" in kernel

    handler_start = text.index("ffi::Error RelionFirstiterBprefFusedXHalfImpl(")
    handler_end = text.index("ffi::Error RelionFusedXHalfBackprojectSignatureImpl(")
    handler = text[handler_start:handler_end]
    assert "float significant_weight" in handler
    assert "float weight_norm" in handler
    assert "cudaMemcpy" not in handler


def test_relion_firstiter_bpref_wrapper_uses_split_native_operands_and_static_scalars(
    monkeypatch,
):
    observed = {}

    def fake_ffi_call(target, result_types, **options):
        observed["target"] = target
        observed["result_types"] = result_types
        observed["options"] = options

        def call(*args, **attrs):
            observed["args"] = args
            observed["attrs"] = attrs
            return args[8], args[9], args[10]

        return call

    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(em_cuda_kernels, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)

    image_shape = (4, 4)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    data_volume = jnp.zeros(volume_size, dtype=jnp.complex64)
    weight_volume = jnp.zeros(volume_size, dtype=jnp.float32)
    image = jnp.arange(12, dtype=jnp.float32).astype(jnp.complex64) * (1.0 + 2.0j)
    ctf = jnp.ones(12, dtype=jnp.float32)
    minvsigma2 = jnp.full(12, 2.0, dtype=jnp.float32)
    posterior = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    translation_angles = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    native_eulers = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3))

    data_out, weight_out = (
        em_cuda_kernels._relion_firstiter_bpref_fused_x_half_static.__wrapped__(
            data_volume,
            weight_volume,
            image,
            ctf,
            minvsigma2,
            posterior,
            translation_angles,
            native_eulers,
            0.125,
            1.0,
            image_shape,
            volume_shape,
            2.0,
        )
    )

    assert observed["target"] == em_cuda_kernels._TARGET_RELION_FIRSTITER_BPREF_FUSED_X_HALF
    assert observed["options"]["input_output_aliases"] == {8: 0, 9: 1, 10: 2}
    assert observed["options"]["vmap_method"] == "sequential"
    assert [item.dtype for item in observed["result_types"]] == [
        jnp.float32,
        jnp.float32,
        jnp.float32,
    ]
    args = observed["args"]
    assert_matches(np.asarray(args[0]), np.asarray(jnp.real(image)))
    assert_matches(np.asarray(args[1]), np.asarray(jnp.imag(image)))
    assert_matches(np.asarray(args[5]), np.asarray(translation_angles[:, 0]))
    assert_matches(np.asarray(args[6]), np.asarray(translation_angles[:, 1]))
    assert args[8].dtype == jnp.float32 and args[9].dtype == jnp.float32
    assert observed["attrs"]["significant_weight"] == np.float32(0.125)
    assert observed["attrs"]["weight_norm"] == np.float32(1.0)
    assert data_out.dtype == jnp.complex64
    assert weight_out is weight_volume


@pytest.mark.gpu
def test_relion_firstiter_bpref_exact_native_ffi_smoke(
    monkeypatch, custom_cuda_lib, gpu_device
):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)

    image_shape = (4, 4)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    image = np.zeros(12, dtype=np.complex64)
    image[0] = np.complex64(2.0 + 3.0j)
    ctf = np.zeros(12, dtype=np.float32)
    ctf[0] = np.float32(2.0)
    minvsigma2 = np.full(12, np.float32(0.5), dtype=np.float32)

    with cuda_backproject.jax.default_device(gpu_device):
        data_out, weight_out = em_cuda_kernels.relion_firstiter_bpref_fused_x_half(
            jnp.zeros(volume_size, dtype=jnp.complex64),
            jnp.zeros(volume_size, dtype=jnp.float32),
            jnp.asarray(image),
            jnp.asarray(ctf),
            jnp.asarray(minvsigma2),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32)[None],
            0.5,
            1.0,
            image_shape,
            volume_shape,
            2.0,
        )

    data_out = np.asarray(data_out)
    weight_out = np.asarray(weight_out)
    expected_offset = 3 * (7 * 4) + 3 * 4
    expected_data = np.zeros(volume_size, dtype=np.complex64)
    expected_weight = np.zeros(volume_size, dtype=np.float32)
    expected_data[expected_offset] = np.complex64(2.0 + 3.0j)
    expected_weight[expected_offset] = np.float32(2.0)
    assert_matches(data_out, expected_data)
    assert_matches(weight_out, expected_weight)


