"""Focused tests for RELION's accelerated fine-score translation FFI."""

from pathlib import Path

import numpy as np
import pytest
from helpers.cuda_source import read_em_cuda_source
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit


def test_relion_translation_angles_match_captured_float32_values():
    from relax.fine_pass.bucket_io import relion_translation_angles_f32

    translations = np.asarray(
        [
            [-0.7461191415786743, -0.7461191415786743],
            [-0.7461191415786743, 0.2538808584213257],
        ],
        dtype=np.float32,
    )
    angles = relion_translation_angles_f32(translations, (256, 256))

    assert_matches(
        angles,
        np.asarray(
            [
                [1016464419, 1016464419],
                [1016464419, 3150720736],
            ],
            dtype=np.uint32,
        ).view(np.float32),
    )


def test_relion_translation_angle_scale_changes_only_final_angle_operand():
    """Final-Q aa0eccbfd4: the model/optics scale multiplies only the angle operand."""
    from relax.fine_pass.bucket_io import relion_translation_angles_f32

    translations = np.asarray([[0.25, -1.75]], dtype=np.float64)
    baseline_translations = translations.copy()
    angle_scale = np.float64("0.99999976470593788")
    angles = relion_translation_angles_f32(
        translations,
        (384, 384),
        angle_scale=angle_scale,
    )
    expected = np.asarray(
        -2.0 * np.pi * translations * angle_scale / 384.0,
        dtype=np.float32,
    )

    assert_matches(angles, expected)
    assert_matches(translations, baseline_translations)


def test_relion_translation_angle_scale_uses_model_over_optics_pixel_size():
    """Every class count: RELION's trial translations are Angstrom from the model pixel over the optics pixel."""
    from relax.refinement.setup_checks import _relion_translation_angle_scale

    scale = _relion_translation_angle_scale(
        model_pixel_size=544.0 / 384.0,
        optics_pixel_sizes=np.asarray([1.416667], dtype=np.float64),
    )
    assert scale == pytest.approx(0.99999976470593788, rel=0.0, abs=1e-16)
    assert _relion_translation_angle_scale(model_pixel_size=1.5, optics_pixel_sizes=None) == 1.0


def test_relion_translation_angle_scale_rejects_heterogeneous_optics():
    from relax.refinement.setup_checks import _relion_translation_angle_scale

    with pytest.raises(NotImplementedError, match="one shared optics pixel size"):
        _relion_translation_angle_scale(
            model_pixel_size=1.5,
            optics_pixel_sizes=np.asarray([1.5, 1.6], dtype=np.float64),
        )


def test_unit_translation_angle_scale_keeps_every_angle_producer_unchanged():
    """Equal pixel sizes must leave the RELION angle operands untouched."""
    from relax.fine_pass.bucket_io import (
        _relion_translation_angles_f64,
        relion_translation_angles_f32,
    )

    rng = np.random.default_rng(20260922)
    translations = rng.uniform(-5.0, 5.0, size=(17, 2)).astype(np.float32)
    reference = -2.0 * np.pi * np.asarray(translations, dtype=np.float64) / 256.0
    assert_matches(
        _relion_translation_angles_f64(translations, (256, 256)),
        reference,
    )
    assert_matches(
        relion_translation_angles_f32(translations, (256, 256), angle_scale=1.0),
        reference.astype(np.float32),
    )
    with pytest.raises(ValueError, match="positive and finite"):
        _relion_translation_angles_f64(translations, (256, 256), angle_scale=0.0)


def test_relion_translation_cuda_source_preserves_explicit_arithmetic():
    source = read_em_cuda_source()

    assert "relion_score_translate_f32" in source
    assert "__fmaf_rn(" in source
    assert "__fmul_rn(static_cast<float>(y), ty)" in source
    assert "const float translated_real = __fmaf_rn(" in source
    assert "cosine, value.x" in source
    assert "__fmul_rn(sine, value.y)" in source
    assert "const float translated_imag = __fmaf_rn(" in source
    assert "__fmul_rn(cosine, value.y)" in source
    assert "float translated_real = cosine * value.x - sine * value.y;" in source
    assert "translated_real * factor" in source
    assert "translated_imag * factor" in source


def test_relion_vdam_ordered_scatter_cuda_graph_is_opt_in_and_fail_closed():
    source = read_em_cuda_source()
    launcher = source.split(
        "cudaError_t launch_relion_vdam_mstep_fused_projector_x_half(", 1
    )[1].split("__device__ __forceinline__ float relion_fine_diff2_update_f32", 1)[0]

    assert '"RELAX_VDAM_ORDERED_SCATTER_CUDA_GRAPH"' in source
    fail_closed = launcher.split(
        "if (ordered_scatter_cuda_graph_requested &&", 1
    )[1].split("return cudaErrorInvalidValue;", 1)[0]
    for required_mode in (
        "!serial_rotation_replay",
        "persistent_serial_rotation_replay",
        "!precompute_ordered_residuals_requested",
        "!fixed_warp_order_scatter_requested",
        "parallel_worker_replay",
        "captured_rotation_replay",
        "float64_accumulator_replay",
        "reverse_rotation_replay",
        "rotation_replay_stride != 0",
        "device_trace_requested",
        "captured_particle_timing_replay",
        "quiesced_prelaunch_capture_requested",
        "exact_native_ptx_requested",
        "exact_wavg_predecessor_requested",
        "runtime_bpref_with_exact_wavg_requested",
        "wavg_bpref_host_gap_requested",
        "wavg_bpref_host_gap_trace_requested",
    ):
        assert required_mode in fail_closed

    fixed_warp_helper = launcher.split(
        "const auto launch_precomputed_fixed_warp_scatter =", 1
    )[1].split("if (ordered_scatter_cuda_graph_requested)", 1)[0]
    graph_path = launcher.split(
        "// The graph keeps every ordinary launch boundary", 1
    )[1].split("const int64_t launch_count =", 1)[0]
    before_capture, capture_and_replay = graph_path.split(
        "cudaStreamBeginCapture(", 1
    )
    capture_region = capture_and_replay.split("cudaStreamEndCapture(", 1)[0]
    assert "ordered_scatter_graph_eulers" in before_capture
    assert "cudaMemcpyDeviceToDevice" in before_capture
    assert "relion_vdam_native_project_f32_kernel<<<" in before_capture
    assert "relion_vdam_native_residual_f32_kernel<<<" in before_capture
    assert "relion_vdam_native_project_f32_kernel<<<" not in capture_region
    assert "relion_vdam_native_residual_f32_kernel<<<" not in capture_region
    assert "for (int64_t rotation_offset = 0;" in capture_region
    assert "rotation_offset < rotation_count" in capture_region
    assert capture_region.count("launch_precomputed_fixed_warp_scatter(") == 1
    assert "ordered_scatter_graph_eulers" in capture_region
    assert "static_cast<unsigned>(rotation_count)" in capture_region
    assert "relion_vdam_native_sgd_f32_kernel<" in fixed_warp_helper
    assert "true><<<" in fixed_warp_helper
    assert "1, 128, 0, particle_streams[lane]" in fixed_warp_helper
    assert "scatter_eulers + rotation_offset * 9" in fixed_warp_helper
    assert (
        "precomputed_residual_weights + ordered_operand_offset"
        in fixed_warp_helper
    )
    ordinary_launch = launcher.split(
        "const auto launch_runtime_sgd =", 1
    )[1].split("const bool use_captured_order =", 1)[0]
    assert "return launch_precomputed_fixed_warp_scatter(" in ordinary_launch
    assert "projector_eulers + particle * euler_stride" in ordinary_launch
    assert "cudaGraphGetNodes(" in graph_path
    assert "captured_node_count != static_cast<size_t>(rotation_count)" in graph_path
    assert "cudaGraphInstantiate(" in graph_path
    assert "cudaGraphLaunch(" in graph_path
    assert "cudaGraphExecDestroy(" in launcher
    assert "cudaGraphDestroy(" in launcher


def test_relion_vdam_exact_native_ptx_discriminator_is_opt_in_and_fail_closed():
    source = read_em_cuda_source()
    makefile = (
        Path(__file__).resolve().parents[2] / "relax" / "cuda" / "Makefile"
    ).read_text()

    assert '"RELAX_VDAM_EXACT_NATIVE_PTX"' in source
    assert 'exact_native_ptx_path[0] != \'\\0\'' in source
    assert "cuModuleLoad(" in source
    assert "cuModuleGetFunction(" in source
    assert "cuLaunchKernel(" in source
    assert "cuModuleUnload(" in source
    assert "-lcuda" in makefile
    assert "sizeof(RelionVdamProjectorKernel) == 64" in source
    assert "alignof(RelionVdamProjectorKernel) == 8" in source

    launcher = source.split(
        "cudaError_t launch_relion_vdam_mstep_fused_projector_x_half(", 1
    )[1].split("__device__ __forceinline__ float relion_fine_diff2_update_f32", 1)[0]
    fail_closed = launcher.split(
        "if (exact_native_ptx_requested &&", 1
    )[1].split("return cudaErrorInvalidValue;", 1)[0]
    for incompatible_mode in (
        "captured_rotation_replay",
        "serial_rotation_replay",
        "float64_accumulator_replay",
        "device_trace_requested",
        "reverse_rotation_replay",
        "rotation_replay_stride > 0",
    ):
        assert incompatible_mode in fail_closed

    native_dispatch = launcher.split("const auto launch_sgd =", 1)[1]
    exact_launch = native_dispatch.split(
        "if (exact_native_ptx_requested &&", 1
    )[1].split("relion_vdam_native_sgd_f32_kernel<", 1)[0]
    assert "!runtime_bpref_with_exact_wavg_requested" in exact_launch
    assert "if constexpr (!std::is_same_v<Accumulator, float>)" in exact_launch
    assert "return cudaErrorInvalidValue;" in exact_launch
    assert "cuLaunchKernel(" in exact_launch
    expected_arguments = (
        "&projector",
        "&image_real_arg",
        "&image_imag_arg",
        "&translation_x_arg",
        "&translation_y_arg",
        "&translation_z_arg",
        "&weights_arg",
        "&minvsigma2_arg",
        "&ctf_arg",
        "&translation_count_arg",
        "&significant_weight_arg",
        "&weight_norm_arg",
        "&eulers_arg",
        "&accumulator_real_arg",
        "&accumulator_imag_arg",
        "&accumulator_weight_arg",
        "&max_r_arg",
        "&max_r2_arg",
        "&padding_factor_arg",
        "&image_x_arg",
        "&image_y_arg",
        "&image_z_arg",
        "&image_xyz_arg",
        "&model_x_arg",
        "&model_y_arg",
        "&model_init_y_arg",
        "&model_init_z_arg",
    )
    parameter_array = exact_launch.split("void* kernel_parameters[] = {", 1)[1].split("};", 1)[0]
    actual_arguments = tuple(arg.strip() for arg in parameter_array.split(",") if arg.strip())
    assert actual_arguments == expected_arguments


def _relion_row_label(pixel_index, half_width, image_size):
    """RELION's row label for a centered packed half-spectrum index.

    ``fftw.h:99-109`` sets ``ip = (i < XSIZE) ? i : i - YSIZE`` with ``XSIZE``
    the half width, so an uncropped half image labels its Nyquist row ``+N/2``.
    RECOVAR's centered packed layout stores that same physical row at
    ``ky = -N/2``; every other row has the same label in both.
    """

    centered = pixel_index // half_width - image_size // 2
    return image_size // 2 if centered == -(image_size // 2) else centered

@pytest.mark.gpu
def test_relion_translate_score_f32_matches_float32_reference(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Match a float32 transcription of the translate kernel.

    Convention repair, not an edited expectation: this test's own expected
    values used to carry RECOVAR's centered row label, which is wrong for the
    packed Nyquist row of an uncropped half image. RELION labels that row
    ``+N/2`` (``fftw.h:99-109``: ``ip = (i < XSIZE) ? i : i - YSIZE`` with
    ``XSIZE`` the half width), and every scoring kernel in
    ``relion_scoring.cuh`` derives it that way at lines 641, 1313, 1370, 2316,
    2487 and 2658. ``pixel_indices`` here includes index 0, which is that row,
    so the expectation had to move with the kernels. Control rows are
    unaffected: the two labels agree everywhere else.
    """

    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_shape = (16, 16)
    half_width = image_shape[1] // 2 + 1
    pixel_indices = np.asarray(
        [0, 1, half_width - 1, 3 * half_width + 2, 8 * half_width, 15 * half_width + 7],
        dtype=np.int32,
    )
    images = np.asarray(
        [
            [1.0 + 0.5j, -2.0 + 0.25j, 0.125 - 4.0j, 3.0 + 2.0j, -0.75 - 0.5j, 8.0 - 3.0j],
            [-1.5 + 1.0j, 0.5 - 0.125j, 2.5 + 7.0j, -4.0 + 0.75j, 0.25 + 0.5j, -6.0 - 2.0j],
        ],
        dtype=np.complex64,
    )
    angles = np.asarray(
        [[0.0, 0.0], [0.018312519416213036, -0.006231173872947693]],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_translate_score_f32(
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
    actual = np.asarray(actual)

    expected = np.empty(
        (images.shape[0] * angles.shape[0], images.shape[1]),
        dtype=np.complex64,
    )
    for image_index, image in enumerate(images):
        for translation_index, (tx, ty) in enumerate(angles):
            output_row = image_index * angles.shape[0] + translation_index
            for pixel_row, pixel_index in enumerate(pixel_indices):
                x = int(pixel_index % half_width)
                y = _relion_row_label(int(pixel_index), half_width, image_shape[0])
                phase = np.float32(
                    np.float32(x) * tx + np.float32(y) * ty
                )
                sine = np.float32(np.sin(phase))
                cosine = np.float32(np.cos(phase))
                real = np.float32(
                    cosine * image[pixel_row].real
                    - sine * image[pixel_row].imag
                )
                imag = np.float32(
                    cosine * image[pixel_row].imag
                    + sine * image[pixel_row].real
                )
                expected[output_row, pixel_row] = np.complex64(real + 1j * imag)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)


@pytest.mark.gpu
def test_relion_translate_score_f32_per_image_angles_are_each_images_own_launch(gpu_device):
    """``[B, T, 2]`` angles (a subtomogram's tilt images, one launch) give each image's ``[T, 2]`` result."""

    from relax.cuda import kernels as em_cuda_kernels

    rng = np.random.default_rng(5)
    image_shape = (16, 16)
    pixel_indices = jnp.asarray(rng.choice(16 * 9, size=40, replace=False).astype(np.int32))
    images = (rng.normal(size=(3, 40)) + 1j * rng.normal(size=(3, 40))).astype(np.complex64)
    angles = rng.uniform(-0.3, 0.3, size=(3, 7, 2)).astype(np.float32)
    with jax.default_device(gpu_device):
        batched = np.asarray(
            em_cuda_kernels.relion_translate_score_f32(jnp.asarray(images), jnp.asarray(angles), pixel_indices, image_shape)
        )
        alone = [
            np.asarray(
                em_cuda_kernels.relion_translate_score_f32(
                    jnp.asarray(images[b : b + 1]), jnp.asarray(angles[b]), pixel_indices, image_shape
                )
            )
            for b in range(3)
        ]
    assert batched.shape == (3 * 7, 40)
    assert_matches(batched, np.concatenate(alone))


@pytest.mark.gpu
def test_relion_translate_score_f32_matches_sealed_relion_values(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Match a fixed RELION 5.0 stack-42988 translation sample within the float32 band."""

    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    # Eight spread-out pixels from the sealed RELION fine-operand capture
    # a81cf6c18e9ce47864c119ae3d827e3aeb64121bf8d071e01176e4bc350e1102.
    # The raw float32 words pin RELION's values in the repository; the
    # comparison is on the decoded floats within the default float32 band.
    pixel_indices = np.asarray(
        [16512, 17823, 19226, 20537, 12394, 13705, 15108, 16420],
        dtype=np.int32,
    )
    image_words = np.asarray(
        [
            [3171083599, 0],
            [981717925, 988820227],
            [964390543, 987107983],
            [3148702193, 3140350956],
            [3130951531, 986468636],
            [3127477872, 986896523],
            [3136130221, 991661190],
            [3116871779, 3131905678],
        ],
        dtype=np.uint32,
    )
    expected_words = np.asarray(
        [
            [3171083599, 0],
            [3104834649, 990426048],
            [3121372780, 986491646],
            [3116391690, 3150027316],
            [3120789091, 989469894],
            [3130377719, 985453807],
            [3137459980, 991064183],
            [973414023, 3131532664],
        ],
        dtype=np.uint32,
    )
    images = (
        image_words[:, 0].view(np.float32)
        + np.complex64(1j) * image_words[:, 1].view(np.float32)
    ).astype(np.complex64)[None, :]
    angles = np.asarray([[1016464419, 1016464419]], dtype=np.uint32).view(
        np.float32
    )

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_translate_score_f32(
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            (256, 256),
        )
    actual_values = np.asarray(actual)[0].view(np.float32).reshape(-1, 2)

    assert_matches(actual_values, expected_words.view(np.float32))


def test_relion_translate_score_f32_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        em_cuda_kernels.relion_translate_score_f32.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
            (8, 8),
        )


def test_relion_translate_score_f64_validates_input_dtype():
    from relax.cuda import kernels as em_cuda_kernels

    with pytest.raises(TypeError, match="images must be complex128"):
        em_cuda_kernels.relion_translate_score_f64.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.asarray([0, 1], dtype=jnp.int32),
            (8, 8),
        )


@pytest.mark.gpu
def test_relion_translate_score_f64_matches_double_sincos_reference(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Match a double-precision transcription of the translate kernel.

    Convention repair, not an edited expectation: this test's own expected
    values used to carry RECOVAR's centered row label, which is wrong for the
    packed Nyquist row of an uncropped half image. RELION labels that row
    ``+N/2`` (``fftw.h:99-109``: ``ip = (i < XSIZE) ? i : i - YSIZE`` with
    ``XSIZE`` the half width), and every scoring kernel in
    ``relion_scoring.cuh`` derives it that way at lines 641, 1313, 1370, 2316,
    2487 and 2658. ``pixel_indices`` here includes index 0, which is that row,
    so the expectation had to move with the kernels. Control rows are
    unaffected: the two labels agree everywhere else.
    """
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    image_shape = (16, 16)
    half_width = image_shape[1] // 2 + 1
    pixel_indices = np.asarray([0, 1, 3 * half_width + 2, 15 * half_width + 7], dtype=np.int32)
    images = np.asarray([[1.0 + 0.5j, -2.0 + 0.25j, 0.125 - 4.0j, -0.75 - 0.5j]], dtype=np.complex128)
    angles = np.asarray([[0.0, 0.0], [0.018312519416213036, -0.006231173872947693]], dtype=np.float64)

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_translate_score_f64(
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
    expected = np.empty((1, angles.shape[0], images.shape[1]), dtype=np.complex128)
    for translation_index, (tx, ty) in enumerate(angles):
        for row, pixel_index in enumerate(pixel_indices):
            x = int(pixel_index % half_width)
            y = _relion_row_label(int(pixel_index), half_width, image_shape[0])
            phase = x * tx + y * ty
            expected[0, translation_index, row] = images[0, row] * complex(np.cos(phase), np.sin(phase))

    np.testing.assert_allclose(
        np.asarray(actual).reshape(expected.shape),
        expected,
        rtol=2e-15,
        atol=2e-15,
    )


def test_relion_translate_bpref_f32_validates_weight_shape(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels

    with pytest.raises(ValueError, match="weighted_ctf must have shape"):
        em_cuda_kernels.relion_translate_bpref_f32.__wrapped__(
            jnp.zeros((2, 3), dtype=jnp.complex64),
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.arange(3, dtype=jnp.int32),
            (8, 8),
        )


def test_relion_translate_bpref_f64_validates_input_dtype():
    from relax.cuda import kernels as em_cuda_kernels

    with pytest.raises(TypeError, match="images must be complex128"):
        em_cuda_kernels.relion_translate_bpref_f64.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.asarray([0, 1], dtype=jnp.int32),
            (8, 8),
        )


@pytest.mark.gpu
def test_relion_translate_bpref_f32_matches_translate_then_weight(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_shape = (16, 16)
    half_width = image_shape[1] // 2 + 1
    pixel_indices = np.asarray(
        [0, 1, 3 * half_width + 2, 8 * half_width, 15 * half_width + 7],
        dtype=np.int32,
    )
    images = np.asarray(
        [[1.0 + 0.5j, -2.0 + 0.25j, 0.125 - 4.0j, 3.0 + 2.0j, -0.75 - 0.5j]],
        dtype=np.complex64,
    )
    weighted_ctf = np.asarray(
        [[2.0, -0.25, 1.5, 1000.0, -3.0]],
        dtype=np.float32,
    )
    angles = np.asarray(
        [[0.0, 0.0], [0.018312519416213036, -0.006231173872947693]],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_translate_bpref_f32(
            jnp.asarray(images),
            jnp.asarray(weighted_ctf),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
        translated = em_cuda_kernels.relion_translate_score_f32(
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
    actual = np.asarray(actual)
    translated = np.asarray(translated).reshape(1, angles.shape[0], -1)
    expected = np.empty_like(translated)
    for translation_index in range(angles.shape[0]):
        for pixel_index in range(images.shape[1]):
            expected[0, translation_index, pixel_index] = np.complex64(
                np.float32(translated[0, translation_index, pixel_index].real * weighted_ctf[0, pixel_index])
                + 1j
                * np.float32(translated[0, translation_index, pixel_index].imag * weighted_ctf[0, pixel_index])
            )

    assert_matches(actual.reshape(expected.shape), expected)


@pytest.mark.gpu
def test_relion_translate_bpref_f64_matches_translate_then_weight(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_shape = (16, 16)
    half_width = image_shape[1] // 2 + 1
    pixel_indices = np.asarray(
        [0, 1, 3 * half_width + 2, 8 * half_width, 15 * half_width + 7],
        dtype=np.int32,
    )
    images = np.asarray(
        [[1.0 + 0.5j, -2.0 + 0.25j, 0.125 - 4.0j, 3.0 + 2.0j, -0.75 - 0.5j]],
        dtype=np.complex128,
    )
    weighted_ctf = np.asarray(
        [[2.0, -0.25, 1.5, 1000.0, -3.0]],
        dtype=np.float64,
    )
    angles = np.asarray(
        [[0.0, 0.0], [0.018312519416213036, -0.006231173872947693]],
        dtype=np.float64,
    )

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_translate_bpref_f64(
            jnp.asarray(images),
            jnp.asarray(weighted_ctf),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
        translated = em_cuda_kernels.relion_translate_score_f64(
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
    actual = np.asarray(actual).reshape(1, angles.shape[0], -1)
    translated = np.asarray(translated).reshape(1, angles.shape[0], -1)
    expected = translated * weighted_ctf[:, None, :]

    assert_matches(actual, expected)


@pytest.mark.gpu
def test_relion_translate_score_labels_the_nyquist_row_as_relion_does(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """The packed Nyquist row must shift with RELION's ``+N/2`` label.

    RELION indexes the uncropped half image with ``fftw.h:99-109``, so the row
    RECOVAR stores at ``ky = -N/2`` is RELION's ``ip = +N/2``; its scoring
    kernels derive that themselves (``relion_scoring.cuh`` lines 641, 1313,
    2316, 2487, 2658). This translate kernel takes centered indices, so it has
    to convert. The two labels differ only here and only for non-integer
    shifts, where ``exp(-i*pi*dy)`` and ``exp(+i*pi*dy)`` are conjugates, which
    is what this test pins: a half-pixel shift must give the conjugate of the
    label RECOVAR's own lattice would imply.

    Measured consequence of getting it wrong (P4-B, 2026-09-20): at
    ``current_size == ori_size`` the exact local engine's peak posterior moved
    by 0.17 and two winners of eight flipped on the resident-local fixture,
    against a numpy transcription of RELION's arithmetic that the resident
    driver matched to 1.9e-7.
    """

    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_size = 16
    image_shape = (image_size, image_size)
    half_width = image_size // 2 + 1
    # Packed row 0 is the Nyquist row, in two of its columns; the controls come
    # from other rows, where both labels agree.
    pixel_indices = np.asarray(
        [0, 1, 3 * half_width + 2, 5 * half_width + 3], dtype=np.int32
    )
    images = np.asarray(
        [[1.0 + 0.5j, -2.0 + 0.25j, 0.75 - 1.5j, 3.0 + 2.0j]], dtype=np.complex64
    )
    # A half-pixel shift in y: the only regime where the two labels differ.
    angles = np.asarray(
        [[0.0, -2.0 * np.pi * 0.5 / image_size]], dtype=np.float32
    )

    with jax.default_device(gpu_device):
        actual = np.asarray(
            em_cuda_kernels.relion_translate_score_f32(
                jnp.asarray(images),
                jnp.asarray(angles),
                jnp.asarray(pixel_indices),
                image_shape,
            )
        )[0]

    def shifted(label_fn):
        out = np.empty(pixel_indices.size, dtype=np.complex64)
        for row, pixel_index in enumerate(pixel_indices):
            x = int(pixel_index % half_width)
            y = label_fn(int(pixel_index))
            phase = np.float32(x) * angles[0, 0] + np.float32(y) * angles[0, 1]
            out[row] = images[0, row] * np.complex64(
                np.cos(phase) + 1j * np.sin(phase)
            )
        return out

    relion = shifted(lambda i: _relion_row_label(i, half_width, image_size))
    packed = shifted(lambda i: i // half_width - image_size // 2)

    np.testing.assert_allclose(actual, relion, rtol=2e-6, atol=2e-6)
    # The control rows are label-independent; the two Nyquist entries are not,
    # so the old label is excluded rather than merely less accurate.
    np.testing.assert_allclose(actual[2:], packed[2:], rtol=2e-6, atol=2e-6)
    # Swapping the label changes the phase by exp(-i * N * ty): here a half
    # pixel, so the packed label returns the negated value.
    ratio = np.exp(-1j * image_size * angles[0, 1])
    for row in (0, 1):
        assert abs(actual[row] - packed[row]) > 1e-2 * abs(images[0, row])
        np.testing.assert_allclose(
            packed[row], relion[row] * ratio, rtol=2e-6, atol=2e-6
        )


@pytest.mark.gpu
def test_relion_translate_bpref_labels_the_nyquist_row_as_relion_does(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """The M-step translate must carry the same RELION label as the scorer.

    ``relion_translate_bpref_*`` shifts the raw image before applying the BPref
    weights, so a wrong row label moves reconstruction operands rather than
    scores. The label is RELION's ``+N/2`` for an uncropped half image
    (``fftw.h:99-109``); see
    ``test_relion_translate_score_labels_the_nyquist_row_as_relion_does``.
    """

    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_size = 16
    image_shape = (image_size, image_size)
    half_width = image_size // 2 + 1
    pixel_indices = np.asarray(
        [0, 2, 4 * half_width + 1, 9 * half_width + 5], dtype=np.int32
    )
    images = np.asarray(
        [[1.0 + 0.5j, -2.0 + 0.25j, 0.75 - 1.5j, 3.0 + 2.0j]], dtype=np.complex64
    )
    weighted_ctf = np.asarray([[0.5, -1.25, 2.0, 0.75]], dtype=np.float32)
    angles = np.asarray(
        [[0.0, -2.0 * np.pi * 0.5 / image_size]], dtype=np.float32
    )

    with jax.default_device(gpu_device):
        actual = np.asarray(
            em_cuda_kernels.relion_translate_bpref_f32(
                jnp.asarray(images),
                jnp.asarray(weighted_ctf),
                jnp.asarray(angles),
                jnp.asarray(pixel_indices),
                image_shape,
            )
        ).reshape(-1)

    def weighted(label_fn):
        out = np.empty(pixel_indices.size, dtype=np.complex64)
        for row, pixel_index in enumerate(pixel_indices):
            x = int(pixel_index % half_width)
            y = label_fn(int(pixel_index))
            phase = np.float32(x) * angles[0, 0] + np.float32(y) * angles[0, 1]
            shifted = images[0, row] * np.complex64(np.cos(phase) + 1j * np.sin(phase))
            out[row] = shifted * weighted_ctf[0, row]
        return out

    relion = weighted(lambda i: _relion_row_label(i, half_width, image_size))
    packed = weighted(lambda i: i // half_width - image_size // 2)
    np.testing.assert_allclose(actual, relion, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual[2:], packed[2:], rtol=2e-6, atol=2e-6)
    for row in (0, 1):
        assert abs(actual[row] - packed[row]) > 1e-2 * abs(
            images[0, row] * weighted_ctf[0, row]
        )


@pytest.mark.gpu
def test_relion_translate_bpref_f32_matches_native_captured_values(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_shape = (256, 256)
    pixel_indices = np.asarray(
        [13031, 13160, 13161, 13167],
        dtype=np.int32,
    )
    image_words = np.asarray(
        [
            [3263777648, 1133111626],
            [3272549528, 1090742966],
            [1127815781, 1125080211],
            [3252159688, 1125857107],
        ],
        dtype=np.uint32,
    )
    images = (
        image_words[:, 0].view(np.float32)
        + np.complex64(1j) * image_words[:, 1].view(np.float32)
    ).astype(np.complex64)[None, :]
    weighted_ctf = np.asarray(
        [[3106311266, 3108593774, 3108351983, 3104897938]],
        dtype=np.uint32,
    ).view(np.float32)
    angles = np.asarray([[3168013433, 3176042026]], dtype=np.uint32).view(
        np.float32
    )
    expected_words = np.asarray(
        [
            [1027138631, 3125411712],
            [1008901170, 1020422283],
            [1013229036, 3173746131],
            [1017732809, 3152071391],
        ],
        dtype=np.uint32,
    )

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_translate_bpref_f32(
            jnp.asarray(images),
            jnp.asarray(weighted_ctf),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
    actual_values = np.asarray(actual)[0].view(np.float32).reshape(-1, 2)

    assert_matches(actual_values, expected_words.view(np.float32))
