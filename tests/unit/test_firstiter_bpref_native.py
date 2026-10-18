"""Native firstiter contracts retained from Q 72e785f3f."""
from pathlib import Path
import pytest
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
