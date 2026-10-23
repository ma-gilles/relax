// EM translation unit of librelax_cuda.so, split verbatim from recovar/cuda/cuda_backproject.cu by the
// relax split (seam S4); the shared device helpers live in recovar/cuda/include/recovar_cuda_common.cuh.
/*
 * CUDA Backprojector / Projector  — v6
 *
 *   - XLA FFI handlers → JIT-compatible inside JAX
 *   - C-linkage API    → ctypes benchmarks / standalone use
 *   - Templated on T   → float (C64) or double (C128)
 *   - On-the-fly freq coords, cz=0 elimination (6-element rotation)
 *   - float2/double2 vectorized complex I/O
 *   - HALF_VOL: half-volume (Hermitian symmetry), halves volume memory
 *   - HALF_IMG: rfft image layout (H × W//2+1), halves pixel count
 *     For backproject, each non-boundary rfft pixel scatters both the
 *     primary value and its Hermitian conjugate at the negated coords.
 *
 * v6 changes:
 *   - CONJ_MODE template parameter for ~2x scatter speedup when
 *     HALF_IMG + HALF_VOL: interior kz (0 < hkz < ic2) get doubled in
 *     the primary scatter (CONJ_MODE=1) and skipped in the conjugate
 *     scatter (CONJ_MODE=2).  This works because for interior kz, the
 *     primary and conjugate scatters land at the same half-volume position
 *     after Hermitian fold, making the conjugate scatter redundant.
 *   - Nyquist fix: kz=-N/2 (Nyquist for even N) is self-conjugate and
 *     scatters directly (no fold/conj) — fixes off-by-one error
 *
 * IMPORTANT: The HALF_VOL scatter (Hermitian fold) is the correct adjoint
 * of the index-based half_volume_to_full_volume in fourier_transform_utils.py.
 * Do NOT use an FFT-based half→full expand; its VJP distributes gradients
 * differently, breaking the CUDA kernel's correctness.
 *
 * Volume: (N0, N1, N2) complex  stored as interleaved T pairs.
 * Half  : (N0, N1, N2/2+1) complex.
 * Images full : (n_images, H*W) complex, row-major (k1/col varies fastest).
 * Images rfft : (n_images, H*(W//2+1)) complex, row-major.
 * Rotations: (n_images, 6) T  — first two rows of 3×3 matrix, row-major.
 *
 * Pixel indexing: row-major — k0_idx = pix / image_w, k1_idx = pix % image_w.
 * This matches NumPy/JAX C-order flatten convention.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#include <cerrno>
#include <chrono>
#include <climits>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#include "device_scratch.cuh"

// RELION's GUI-default InitialModel jobs use --j 8.  Coarse scoring uses one
// blocking CUDA stream per worker; these helpers own stream creation, the
// parent-stream dependency, synchronization, and cleanup.
constexpr int kRelionVdamWorkerStreams = 8;

cudaError_t initialize_relion_vdam_worker_streams(
    cudaStream_t parent_stream,
    cudaStream_t worker_streams[kRelionVdamWorkerStreams],
    cudaEvent_t* inputs_ready)
{
    cudaError_t error = cudaEventCreateWithFlags(
        inputs_ready, cudaEventDisableTiming);
    if (error != cudaSuccess) return error;
    error = cudaEventRecord(*inputs_ready, parent_stream);
    if (error != cudaSuccess) return error;
    for (int worker = 0; worker < kRelionVdamWorkerStreams; ++worker)
    {
        // Ordinary blocking streams match RELION's per-worker class streams.
        error = cudaStreamCreate(&worker_streams[worker]);
        if (error != cudaSuccess) return error;
        error = cudaStreamWaitEvent(
            worker_streams[worker], *inputs_ready, 0);
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}

cudaError_t synchronize_relion_vdam_worker_streams(
    cudaStream_t worker_streams[kRelionVdamWorkerStreams])
{
    for (int worker = 0; worker < kRelionVdamWorkerStreams; ++worker)
    {
        const cudaError_t error = cudaStreamSynchronize(worker_streams[worker]);
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}

template <typename LaunchParticle>
cudaError_t dispatch_relion_vdam_round_robin_workers(
    cudaStream_t worker_streams[kRelionVdamWorkerStreams],
    int64_t particle_count,
    LaunchParticle&& launch_particle)
{
    bool worker_started[kRelionVdamWorkerStreams] = {};
    for (int64_t particle = 0; particle < particle_count; ++particle)
    {
        const int worker = static_cast<int>(
            particle % kRelionVdamWorkerStreams);
        // Match RELION's task distributor: a worker synchronizes its class
        // stream before taking its next particle, while other workers remain
        // independent.
        if (worker_started[worker])
        {
            const cudaError_t error = cudaStreamSynchronize(
                worker_streams[worker]);
            if (error != cudaSuccess) return error;
        }
        const cudaError_t error = launch_particle(particle, worker);
        if (error != cudaSuccess) return error;
        worker_started[worker] = true;
    }
    return synchronize_relion_vdam_worker_streams(worker_streams);
}

void destroy_relion_vdam_worker_streams(
    cudaStream_t worker_streams[kRelionVdamWorkerStreams],
    cudaEvent_t inputs_ready)
{
    for (int worker = 0; worker < kRelionVdamWorkerStreams; ++worker)
        if (worker_streams[worker]) cudaStreamDestroy(worker_streams[worker]);
    if (inputs_ready) cudaEventDestroy(inputs_ready);
}

#include "recovar_cuda_common.cuh"

#include "ffi_scratch.cuh"
#include "tiled_translations.cuh"


/* Strict RELION x-half diagnostic: scatter the pre-reduced complex data and
 * real weight through the same neighbor loop. RELION updates model real,
 * model imaginary, and model weight consecutively for each neighbor; keeping
 * those atomics together is the sole semantic difference from invoking the
 * generic complex and real scatter paths separately. */
template <typename T, typename ComplexT, bool CAPTURE_SIGNATURE, bool ACCUMULATE,
          bool SEPARATE_DATA = false>
static __device__ __forceinline__ void scatter_trilinear_relion_fused_x_half(
    ComplexT* __restrict__ data_volume,
    T* __restrict__ data_real_volume,
    T* __restrict__ data_imag_volume,
    T* __restrict__ weight_volume,
    T rk0, T rk1, T rk2,
    T data_re, T data_im, T Fweight,
    T c0, T c1, T c2,
    int N0, int N1, int N2_eff, int stride0, int stride1,
    int signature_base,
    int32_t* __restrict__ signature_neighbor_indices,
    float* __restrict__ signature_neighbor_coefficients,
    int32_t* __restrict__ signature_neighbor_flags)
{
    const T g0 = rk0 + c0;
    const T g1 = rk1 + c1;
    const int ic0 = (int)c0;
    const int ic1 = (int)c1;
    const int ic2 = (int)c2;
    const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
    const T g2_full = rk2 + c2;

    /* The caller's compact all-neighbor gate already proves these bounds for
     * RELION BPref shapes; retain this as a defensive array-safety check. */
    if (g0 < T(-1) || g0 >= T(N0) ||
        g1 < T(-1) || g1 >= T(N1) ||
        g2_full < T(-1) || g2_full >= T(N2_full)) return;

    /* RELION forms each interpolation fraction from the rotated coordinate
     * before applying the integer model origin.  Adding the origin first is
     * mathematically equivalent but loses float32 mantissa bits and changes
     * the trilinear coefficients by several ulp. */
    /* RELION's BP.cuh calls floorf for these integer buckets even when
     * XFLOAT is double.  Keep the original T-valued coordinate for the
     * fractional remainder, exactly as the native kernel does. */
    const int r0 = relion_floor_int(rk0);
    const int r1 = relion_floor_int(rk1);
    const int r2 = relion_floor_int(rk2);
    const int b0 = r0 + ic0;
    const int b1 = r1 + ic1;
    const int b2 = r2 + ic2;
    const T f0 = rk0 - T(r0);
    const T f1 = rk1 - T(r1);
    const T f2 = rk2 - T(r2);
    const T w0[2] = {T(1) - f0, f0};
    const T w1[2] = {T(1) - f1, f1};
    const T w2[2] = {T(1) - f2, f2};

    #pragma unroll
    for (int d0 = 0; d0 < 2; d0++) {
        int j0 = b0 + d0;
        if ((unsigned)j0 >= (unsigned)N0) continue;
        #pragma unroll
        for (int d1 = 0; d1 < 2; d1++) {
            int j1 = b1 + d1;
            if ((unsigned)j1 >= (unsigned)N1) continue;
            const T ww = w0[d0] * w1[d1];
            #pragma unroll
            for (int d2 = 0; d2 < 2; d2++) {
                const int signature_slot = d0 * 4 + d1 * 2 + d2;
                const int signature_index = signature_base + signature_slot;
                const int j2 = b2 + d2;
                if ((unsigned)j0 >= (unsigned)N0 ||
                    (unsigned)j1 >= (unsigned)N1 ||
                    (unsigned)j2 >= (unsigned)N2_full) {
                    if constexpr (CAPTURE_SIGNATURE) {
                        signature_neighbor_flags[signature_index] = 8;
                    }
                    continue;
                }
                const int kz = j2 - ic2;
                const T w = ww * w2[d2];
                int sj0 = j0;
                int sj1 = j1;
                int hkz;
                int32_t neighbor_flags = 1;
                T sre = w * data_re;
                T sim = w * data_im;
                if (kz >= 0) {
                    hkz = kz;
                } else if ((N2_full & 1) == 0 && -kz == ic2) {
                    hkz = ic2;
                    neighbor_flags |= 4;
                } else {
                    sj0 = (N0 - (N0 & 1) - j0) % N0;
                    sj1 = (N1 - (N1 & 1) - j1) % N1;
                    hkz = -kz;
                    sim = -sim;
                    neighbor_flags |= 2;
                }
                if (hkz > ic2) {
                    if constexpr (CAPTURE_SIGNATURE) {
                        signature_neighbor_flags[signature_index] = 8;
                    }
                    continue;
                }
                const int off = sj0 * stride0 + sj1 * stride1 + hkz;
                if constexpr (CAPTURE_SIGNATURE) {
                    signature_neighbor_indices[signature_index] = off;
                    signature_neighbor_coefficients[signature_index] = (float)w;
                    signature_neighbor_flags[signature_index] = neighbor_flags;
                }
                if constexpr (ACCUMULATE) {
                    if constexpr (SEPARATE_DATA) {
                        atomicAdd(&data_real_volume[off], sre);
                        atomicAdd(&data_imag_volume[off], sim);
                    } else {
                        atomicAdd(&data_volume[off].x, sre);
                        atomicAdd(&data_volume[off].y, sim);
                    }
                    atomicAdd(&weight_volume[off], w * Fweight);
                }
            }
        }
    }
}

/* One invocation corresponds to one RELION particle. blockIdx.x retains the
 * particle-local orientation-row order, while each 128-thread block walks the
 * native current-size FFTW square in serial pixel passes. */
template <typename T, typename ComplexT, bool CAPTURE_SIGNATURE, bool ACCUMULATE>
__global__ void __launch_bounds__(128)
relion_fused_x_half_backproject_kernel(
    ComplexT* __restrict__ data_volume,
    T* __restrict__ weight_volume,
    const ComplexT* __restrict__ data_rows,
    const T* __restrict__ weight_rows,
    const int32_t* __restrict__ pixel_indices,
    const T* __restrict__ rot,
    const int32_t* __restrict__ canonical_rotation_keys,
    const int32_t* __restrict__ signature_row_indices,
    int32_t* __restrict__ signature_rotation_keys,
    int32_t* __restrict__ signature_pixel_indices,
    int32_t* __restrict__ signature_row_flags,
    float* __restrict__ signature_source_values,
    int32_t* __restrict__ signature_neighbor_indices,
    float* __restrict__ signature_neighbor_coefficients,
    int32_t* __restrict__ signature_neighbor_flags,
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, T max_r2, int n_source_rows)
{
    __shared__ T R[6];
    const int output_row = (int)blockIdx.x;
    int source_row;
    if constexpr (CAPTURE_SIGNATURE) {
        source_row = (int)signature_row_indices[output_row];
        if ((unsigned)source_row >= (unsigned)n_source_rows) return;
    } else {
        source_row = output_row;
    }
    if (threadIdx.x < 6) R[threadIdx.x] = rot[source_row * 6 + threadIdx.x];
    __syncthreads();

    for (int pix = (int)threadIdx.x; pix < n_pixels; pix += 128) {
        const int orig_pix = (int)pixel_indices[pix];
        const int row_pixel = output_row * n_pixels + pix;
        const int source_row_pixel = source_row * n_pixels + pix;
        int32_t row_flags = 0;
        if constexpr (CAPTURE_SIGNATURE) {
            signature_rotation_keys[row_pixel] = canonical_rotation_keys[source_row];
            signature_pixel_indices[row_pixel] = orig_pix;
            signature_row_flags[row_pixel] = 0;
            #pragma unroll
            for (int value_index = 0; value_index < 6; value_index++) {
                signature_source_values[row_pixel * 6 + value_index] = nanf("");
            }
            #pragma unroll
            for (int slot = 0; slot < 8; slot++) {
                const int signature_index = row_pixel * 8 + slot;
                signature_neighbor_indices[signature_index] = -1;
                signature_neighbor_coefficients[signature_index] = 0.0f;
                signature_neighbor_flags[signature_index] = 8;
            }
        }
        const int k0_idx = orig_pix / image_w;
        const int k1_idx = orig_pix % image_w;

        const T k0_unscaled = (k0_idx < image_w)
            ? T(k0_idx)
            : T(k0_idx - image_h);
        const T k1_unscaled = T(k1_idx);
        const T k0 = k0_unscaled * T(upsampling);
        const T k1 = k1_unscaled * T(upsampling);

        /* RELION omits the redundant negative-y x=0 FFTW row. */
        if (k1_idx == 0 && k0_idx >= image_w) {
            if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 1;
            continue;
        }
        if (max_r2 >= 0.0f && k0 * k0 + k1 * k1 > max_r2) {
            if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 2;
            continue;
        }

        const T Fweight = weight_rows[source_row_pixel];
        const ComplexT value = data_rows[source_row_pixel];
        if constexpr (CAPTURE_SIGNATURE) {
            signature_source_values[row_pixel * 6 + 0] = value.x;
            signature_source_values[row_pixel * 6 + 1] = value.y;
            signature_source_values[row_pixel * 6 + 2] = Fweight;
        }
        /* Match cuda_kernel_backproject3D's sole outer accumulation gate. */
        if (!(Fweight > 0.0f)) {
            if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 4;
            continue;
        }

        T data_re = value.x;
        T data_im = value.y;
        /* Match RELION cuda_kernel_backproject3D: form matrix-x*source-x
         * before matrix-y*source-y, then apply padding_factor. Both the
         * addend order and delayed scaling are observable at interpolation
         * boundaries. */
        T rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * T(upsampling);
        T rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * T(upsampling);
        T rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * T(upsampling);
        if constexpr (CAPTURE_SIGNATURE) {
            signature_source_values[row_pixel * 6 + 3] = rk0;
            signature_source_values[row_pixel * 6 + 4] = rk1;
            signature_source_values[row_pixel * 6 + 5] = rk2;
        }

        if (max_r2 >= T(0)) {
            const T r2_3d = relion_radius_squared(rk0, rk1, rk2);
            if (r2_3d > max_r2) {
                if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 8;
                continue;
            }
        }
        if (rk2 < T(0)) {
            row_flags |= 16;
            rk0 = -rk0;
            rk1 = -rk1;
            rk2 = -rk2;
            data_im = -data_im;
        }
        if (max_r2 >= T(0)) {
            const int maxR = (int)floor(sqrt(max_r2) + T(0.5));
            if (relion_compact_trilinear_oob<T>(rk2, rk1, rk0, maxR)) {
                if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 32;
                continue;
            }
        }
        row_flags |= 64;
        if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags;

        const int stride1 = N2_eff;
        const int stride0 = N1 * N2_eff;
        scatter_trilinear_relion_fused_x_half<T, ComplexT, CAPTURE_SIGNATURE, ACCUMULATE>(
            data_volume, nullptr, nullptr, weight_volume,
            rk0, rk1, rk2, data_re, data_im, Fweight,
            c0, c1, c2, N0, N1, N2_eff, stride0, stride1,
            row_pixel * 8,
            signature_neighbor_indices,
            signature_neighbor_coefficients,
            signature_neighbor_flags);
    }
}

/* RELION's CUDA accelerated projector stores the Fourier reference in CUDA
 * texture objects with cudaFilterModeLinear. Hardware texture interpolation is
 * not bit-identical to the manual no_tex3D trilinear path above. This gated
 * diagnostic path mirrors RELION's texture setup for full complex64 volumes.
 *
 * Axes are transposed for the texture array: recovar stores vol[i0,i1,i2] with
 * i2 fastest, while tex3D's x coordinate addresses the fastest dimension.
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
split_complex_float_kernel(
    const float* __restrict__ vol,
    float* __restrict__ real,
    float* __restrict__ imag,
    int n_voxels)
{
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n_voxels) return;
    real[i] = vol[2 * i];
    imag[i] = vol[2 * i + 1];
}


__global__ void __launch_bounds__(BLOCK_SIZE)
fill_relion_half_texture_surface_f32_kernel(
    const float2* __restrict__ projector_half,
    cudaSurfaceObject_t surface,
    int64_t texture_voxels,
    int tex_x,
    int tex_y,
    float projector_scale)
{
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
    if (index >= texture_voxels) return;

    const int x = static_cast<int>(index % tex_x);
    const int64_t yz = index / tex_x;
    const int y = static_cast<int>(yz % tex_y);
    const int z = static_cast<int>(yz / tex_y);
    const float2 value = projector_half[index];
    const float2 scaled = make_float2(
        __fmul_rn(value.x, projector_scale),
        __fmul_rn(value.y, projector_scale));
    surf3Dwrite(scaled, surface, x * static_cast<int>(sizeof(float2)), y, z);
}

struct RelionHalfTextureF32 {
    cudaArray_t array = nullptr;
    cudaSurfaceObject_t surface = 0;
    cudaTextureObject_t texture = 0;
};

cudaError_t destroy_relion_half_texture_f32(RelionHalfTextureF32* texture)
{
    cudaError_t err = cudaSuccess;
    if (texture->texture) {
        const cudaError_t cleanup_err = cudaDestroyTextureObject(texture->texture);
        if (err == cudaSuccess) err = cleanup_err;
    }
    if (texture->surface) {
        const cudaError_t cleanup_err = cudaDestroySurfaceObject(texture->surface);
        if (err == cudaSuccess) err = cleanup_err;
    }
    if (texture->array) {
        const cudaError_t cleanup_err = cudaFreeArray(texture->array);
        if (err == cudaSuccess) err = cleanup_err;
    }
    texture->texture = 0;
    texture->surface = 0;
    texture->array = nullptr;
    return err;
}

cudaError_t synchronize_and_destroy_relion_half_texture_f32(
    cudaStream_t stream,
    RelionHalfTextureF32* texture,
    cudaError_t first_err)
{
    cudaError_t err = first_err;
    if (texture->texture || texture->surface || texture->array) {
        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (err == cudaSuccess) err = sync_err;
    }
    const cudaError_t cleanup_err = destroy_relion_half_texture_f32(texture);
    if (err == cudaSuccess) err = cleanup_err;
    return err;
}

cudaError_t create_relion_half_texture_f32(
    cudaStream_t stream,
    const float2* projector_half,
    int tex_x,
    int tex_y,
    int tex_z,
    float projector_scale,
    RelionHalfTextureF32* texture)
{
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
    const cudaExtent extent = make_cudaExtent(
        static_cast<size_t>(tex_x),
        static_cast<size_t>(tex_y),
        static_cast<size_t>(tex_z));
    cudaError_t err = cudaMalloc3DArray(
        &texture->array,
        &desc,
        extent,
        cudaArraySurfaceLoadStore);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }

    cudaResourceDesc resource;
    memset(&resource, 0, sizeof(resource));
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = texture->array;
    err = cudaCreateSurfaceObject(&texture->surface, &resource);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }

    const int64_t texture_voxels =
        static_cast<int64_t>(tex_x) * tex_y * tex_z;
    const int64_t block_count =
        (texture_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill_relion_half_texture_surface_f32_kernel<<<
        static_cast<unsigned int>(block_count), BLOCK_SIZE, 0, stream>>>(
            projector_half,
            texture->surface,
            texture_voxels,
            tex_x,
            tex_y,
            projector_scale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return synchronize_and_destroy_relion_half_texture_f32(
            stream,
            texture,
            err);
    }

    cudaTextureDesc texture_desc;
    memset(&texture_desc, 0, sizeof(texture_desc));
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = false;
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    err = cudaCreateTextureObject(
        &texture->texture, &resource, &texture_desc, nullptr);
    if (err != cudaSuccess) {
        /* The surface-fill kernel is asynchronous.  If texture creation
         * fails, wait for that queued writer before destroying its surface
         * and array.  Preserve the first error while still attempting every
         * cleanup operation. */
        return synchronize_and_destroy_relion_half_texture_f32(
            stream,
            texture,
            err);
    }
    return cudaSuccess;
}

/* A supplied host Projector::data slab should not need an equally large JAX
 * device allocation merely to populate the CUDA array required for hardware
 * interpolation.  The persistent owner below uploads that host slab once and
 * reuses its texture across sparse pass-2 buckets.  Handles are monotonic and
 * resolved through a guarded registry: a compiled FFI executable can retain a
 * stale handle after explicit destruction, but it can never dereference freed
 * storage or alias a later owner. */
struct PersistentRelionHalfTextureF32 {
    RelionHalfTextureF32 texture;
    int tex_x = 0;
    int tex_y = 0;
    int tex_z = 0;
    int device = -1;

    std::mutex state_mutex;
    std::condition_variable state_changed;
    bool closing = false;
    int active_calls = 0;

    bool acquire_call()
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (closing) return false;
        ++active_calls;
        return true;
    }

    void release_call() noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(state_mutex);
            --active_calls;
            if (active_calls == 0) state_changed.notify_all();
        } catch (...) {
            /* A mutex failure during process teardown must not cross an FFI
             * or destructor boundary. */
        }
    }

    void wait_until_idle()
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        closing = true;
        state_changed.wait(lock, [this] { return active_calls == 0; });
    }

    cudaError_t destroy_owned_texture() noexcept
    {
        int original_device = -1;
        cudaError_t err = cudaGetDevice(&original_device);
        if (err == cudaSuccess && original_device != device)
            err = cudaSetDevice(device);
        if (err == cudaSuccess &&
            (texture.texture || texture.surface || texture.array))
            err = cudaDeviceSynchronize();
        const cudaError_t cleanup_err =
            destroy_relion_half_texture_f32(&texture);
        if (err == cudaSuccess) err = cleanup_err;
        if (original_device >= 0 && original_device != device) {
            const cudaError_t restore_err = cudaSetDevice(original_device);
            if (err == cudaSuccess) err = restore_err;
        }
        return err;
    }

    ~PersistentRelionHalfTextureF32()
    {
        (void)destroy_owned_texture();
    }
};

struct PersistentRelionHalfTextureF32CallGuard {
    std::shared_ptr<PersistentRelionHalfTextureF32> owner;
    bool active = false;

    explicit PersistentRelionHalfTextureF32CallGuard(
        std::shared_ptr<PersistentRelionHalfTextureF32> value)
        noexcept
        : owner(std::move(value))
    {
        try {
            active = owner && owner->acquire_call();
        } catch (...) {
            active = false;
        }
    }

    ~PersistentRelionHalfTextureF32CallGuard()
    {
        if (active) owner->release_call();
    }
};

static std::mutex persistent_relion_half_texture_f32_mutex;
static std::unordered_map<
    uint64_t,
    std::shared_ptr<PersistentRelionHalfTextureF32>>
    persistent_relion_half_texture_f32_registry;
static std::atomic<uint64_t> persistent_relion_half_texture_f32_next_handle{1};

cudaError_t create_relion_half_texture_f32_from_host(
    const float2* projector_half_host,
    int tex_x,
    int tex_y,
    int tex_z,
    RelionHalfTextureF32* texture)
{
    if (!projector_half_host || !texture ||
        tex_x <= 0 || tex_y <= 0 || tex_z <= 0)
        return cudaErrorInvalidValue;

    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
    const cudaExtent extent = make_cudaExtent(
        static_cast<size_t>(tex_x),
        static_cast<size_t>(tex_y),
        static_cast<size_t>(tex_z));
    cudaError_t err = cudaMalloc3DArray(
        &texture->array,
        &desc,
        extent,
        cudaArrayDefault);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }

    cudaMemcpy3DParms copy_params;
    memset(&copy_params, 0, sizeof(copy_params));
    copy_params.srcPtr = make_cudaPitchedPtr(
        const_cast<float2*>(projector_half_host),
        static_cast<size_t>(tex_x) * sizeof(float2),
        static_cast<size_t>(tex_x),
        static_cast<size_t>(tex_y));
    copy_params.dstArray = texture->array;
    copy_params.extent = extent;
    copy_params.kind = cudaMemcpyHostToDevice;
    /* The blocking copy is the ownership boundary.  Python may release its
     * host view immediately after the create call returns successfully. */
    err = cudaMemcpy3D(&copy_params);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }

    cudaResourceDesc resource;
    cudaTextureDesc texture_desc;
    memset(&resource, 0, sizeof(resource));
    memset(&texture_desc, 0, sizeof(texture_desc));
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = texture->array;
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = false;
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    err = cudaCreateTextureObject(
        &texture->texture,
        &resource,
        &texture_desc,
        nullptr);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }
    return cudaSuccess;
}

extern "C" int recovar_relion_persistent_half_texture_f32_create(
    const void* projector_half_host,
    int tex_x,
    int tex_y,
    int tex_z,
    int device,
    float projector_scale,
    uint64_t* owner_handle)
{
    if (!projector_half_host || !owner_handle || projector_scale != 1.0f ||
        !std::isfinite(projector_scale) ||
        tex_x <= 0 || tex_y <= 0 || tex_z <= 0 || device < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    *owner_handle = 0;

    try {
        auto owner = std::make_shared<PersistentRelionHalfTextureF32>();
        owner->tex_x = tex_x;
        owner->tex_y = tex_y;
        owner->tex_z = tex_z;
        owner->device = device;

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) return static_cast<int>(err);
        if (device >= device_count)
            return static_cast<int>(cudaErrorInvalidDevice);
        int original_device = -1;
        err = cudaGetDevice(&original_device);
        if (err != cudaSuccess) return static_cast<int>(err);
        if (original_device != device) err = cudaSetDevice(device);
        if (err == cudaSuccess) {
            err = create_relion_half_texture_f32_from_host(
                static_cast<const float2*>(projector_half_host),
                tex_x,
                tex_y,
                tex_z,
                &owner->texture);
        }
        if (original_device != device) {
            const cudaError_t restore_err = cudaSetDevice(original_device);
            if (err == cudaSuccess) err = restore_err;
        }
        if (err != cudaSuccess) return static_cast<int>(err);

        const uint64_t handle =
            persistent_relion_half_texture_f32_next_handle.fetch_add(
                1, std::memory_order_relaxed);
        if (handle == 0 ||
            handle > static_cast<uint64_t>(
                std::numeric_limits<int64_t>::max()))
            return static_cast<int>(cudaErrorInvalidValue);
        {
            std::lock_guard<std::mutex> lock(
                persistent_relion_half_texture_f32_mutex);
            const auto inserted =
                persistent_relion_half_texture_f32_registry.emplace(
                    handle, owner);
            if (!inserted.second)
                return static_cast<int>(cudaErrorInvalidResourceHandle);
        }
        *owner_handle = handle;
        return static_cast<int>(cudaSuccess);
    } catch (const std::bad_alloc&) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    } catch (...) {
        return static_cast<int>(cudaErrorUnknown);
    }
}

extern "C" int recovar_relion_persistent_half_texture_f32_destroy(
    uint64_t owner_handle)
{
    if (owner_handle == 0)
        return static_cast<int>(cudaErrorInvalidResourceHandle);
    try {
        std::shared_ptr<PersistentRelionHalfTextureF32> owner;
        {
            std::lock_guard<std::mutex> lock(
                persistent_relion_half_texture_f32_mutex);
            const auto found =
                persistent_relion_half_texture_f32_registry.find(owner_handle);
            if (found == persistent_relion_half_texture_f32_registry.end())
                return static_cast<int>(cudaErrorInvalidResourceHandle);
            owner = found->second;
            persistent_relion_half_texture_f32_registry.erase(found);
        }
        owner->wait_until_idle();
        return static_cast<int>(owner->destroy_owned_texture());
    } catch (const std::bad_alloc&) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    } catch (...) {
        return static_cast<int>(cudaErrorUnknown);
    }
}


template <bool HALF_IMG>
__global__ void __launch_bounds__(BLOCK_SIZE)
project_relion_half_texture_f32_kernel(
    cudaTextureObject_t texture,
    float2* __restrict__ image,
    const float* __restrict__ rotations,
    int n_pixels,
    int image_h,
    int image_w,
    int tex_y_init,
    int tex_z_init,
    int padding_factor,
    int max_r2_padded)
{
    __shared__ float rotation[6];

    const int image_index = blockIdx.x;
    const int pixel = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    if (threadIdx.x < 6)
        rotation[threadIdx.x] = rotations[image_index * 6 + threadIdx.x];
    __syncthreads();
    if (pixel >= n_pixels) return;

    const int row = pixel / image_w;
    const int column = pixel % image_w;
    const float source_y = static_cast<float>(
        row == 0 ? image_h / 2 : row - image_h / 2);
    const float source_x = HALF_IMG
        ? static_cast<float>(column)
        : static_cast<float>(column - image_w / 2);

    const float model_x =
        (rotation[3] * source_x + rotation[0] * source_y) *
        static_cast<float>(padding_factor);
    const float model_y =
        (rotation[4] * source_x + rotation[1] * source_y) *
        static_cast<float>(padding_factor);
    const float model_z =
        (rotation[5] * source_x + rotation[2] * source_y) *
        static_cast<float>(padding_factor);
    const int64_t output_index =
        static_cast<int64_t>(image_index) * n_pixels + pixel;
    const float radius_squared =
        model_x * model_x + model_y * model_y + model_z * model_z;
    if (!isfinite(radius_squared) || radius_squared >= 2147483648.0f ||
        static_cast<int>(radius_squared) > max_r2_padded) {
        image[output_index] = make_float2(0.0f, 0.0f);
        return;
    }

    float texture_x = model_x;
    float texture_y = model_y;
    float texture_z = model_z;
    float imag_sign = 1.0f;
    if (texture_x < 0.0f) {
        texture_x = -texture_x;
        texture_y = -texture_y;
        texture_z = -texture_z;
        imag_sign = -1.0f;
    }
    float2 value = tex3D<float2>(
        texture,
        texture_x + 0.5f,
        texture_y - static_cast<float>(tex_y_init) + 0.5f,
        texture_z - static_cast<float>(tex_z_init) + 0.5f);
    value.y *= imag_sign;
    image[output_index] = value;
}

cudaError_t launch_relion_projector_half_texture_f32(
    cudaStream_t stream,
    const float2* projector_half,
    float2* image,
    const float* rotations,
    int64_t n_images,
    int64_t image_h,
    int64_t image_w,
    int padding_factor,
    int projector_max_r,
    float projector_scale)
{
    if (n_images == 0 || image_h == 0 || image_w == 0) return cudaSuccess;
    const int padded_max_r = projector_max_r * padding_factor;
    const int image_max_r =
        projector_max_r < image_h / 2 ? projector_max_r : image_h / 2;
    const int padded_image_max_r = image_max_r * padding_factor;
    const int tex_x = padded_max_r + 2;
    const int tex_y = 2 * padded_max_r + 3;
    const int tex_z = 2 * padded_max_r + 3;
    const int tex_y_init = -(padded_max_r + 1);
    const int tex_z_init = -(padded_max_r + 1);
    const int64_t n_pixels = image_h * image_w;
    RelionHalfTextureF32 projector_texture;
    cudaError_t err = create_relion_half_texture_f32(
        stream,
        projector_half,
        tex_x,
        tex_y,
        tex_z,
        projector_scale,
        &projector_texture);
    if (err != cudaSuccess) goto cleanup;

    {
        dim3 grid(
            static_cast<unsigned int>(n_images),
            static_cast<unsigned int>((n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE));
        dim3 block(BLOCK_SIZE);
        project_relion_half_texture_f32_kernel<true><<<grid, block, 0, stream>>>(
            projector_texture.texture,
            image,
            rotations,
            static_cast<int>(n_pixels),
            static_cast<int>(image_h),
            static_cast<int>(image_w),
            tex_y_init,
            tex_z_init,
            padding_factor,
            padded_image_max_r * padded_image_max_r);
        err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    }

cleanup:
    return synchronize_and_destroy_relion_half_texture_f32(
        stream,
        &projector_texture,
        err);
}

cudaError_t launch_relion_projector_persistent_half_texture_f32(
    cudaStream_t stream,
    const PersistentRelionHalfTextureF32& owner,
    float2* image,
    const float* rotations,
    int64_t n_images,
    int64_t image_h,
    int64_t image_w,
    int padding_factor,
    int projector_max_r)
{
    if (n_images == 0 || image_h == 0 || image_w == 0) return cudaSuccess;
    if (!owner.texture.texture) return cudaErrorInvalidResourceHandle;

    const int padded_max_r = projector_max_r * padding_factor;
    const int image_max_r =
        projector_max_r < image_h / 2 ? projector_max_r : image_h / 2;
    const int padded_image_max_r = image_max_r * padding_factor;
    const int tex_y_init = -(padded_max_r + 1);
    const int tex_z_init = -(padded_max_r + 1);
    const int64_t n_pixels = image_h * image_w;
    dim3 grid(
        static_cast<unsigned int>(n_images),
        static_cast<unsigned int>((n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE));
    dim3 block(BLOCK_SIZE);
    project_relion_half_texture_f32_kernel<true><<<grid, block, 0, stream>>>(
        owner.texture.texture,
        image,
        rotations,
        static_cast<int>(n_pixels),
        static_cast<int>(image_h),
        static_cast<int>(image_w),
        tex_y_init,
        tex_z_init,
        padding_factor,
        padded_image_max_r * padded_image_max_r);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    return err;
}

/* Diagnostic reproduction of the complete per-pixel atomic issue order in
 * RELION's Wavg kernel.  The final axis is [XA, AA, diff2].  Keeping all three
 * atomics in one thread is important: the diff2 atomic after AA delays the
 * second pixel lane (pixel + blockDim.x) exactly as the native kernel does. */


/* One body for both term layouts.  ``FLAT_ROWS`` replaces the rectangular
 * ``[batch, rotation]`` block address with a packed row whose image comes from
 * ``row_image_ids``, exactly as the flat-row fine scorer does.  In the
 * rectangular layout the order the per-cell atomic adds land in is hardware
 * scheduled, so the float32 accumulator is bitwise reproducible only for exactly
 * representable summands.
 *
 * The flat-row callers hand rows sorted by image (image-major within a class),
 * so each image's rows form one run.  The first row of a run sums the run's
 * rows in row order and adds that sum with one atomic per cell; the other rows
 * return.  With one run per image a cell receives a single add per launch and
 * the accumulator is reproducible; rows of one image split over several runs
 * stay correct and race only as the per-row atomics did. */
template <bool FLAT_ROWS = false>
__global__ void __launch_bounds__(256)
relion_wavg_rotation_atomic_runtime_triplet_f32_kernel(
    const float* __restrict__ terms,
    float* __restrict__ output,
    int n_rotations,
    int pixel_capacity,
    const int32_t* __restrict__ runtime_logical_pixel_count,
    const int32_t* __restrict__ row_image_ids = nullptr,
    int64_t row_count = 0,
    int64_t batch_size = 0,
    const int32_t* __restrict__ exact_positions = nullptr,
    int output_pixel_capacity = 0)
{
    /* ``exact_positions`` (flat rows only): ``terms`` holds only the Wavg
     * rectangle's exact-radius pixels, ``pixel_capacity`` of them, and pixel
     * ``i`` adds into rectangle position ``exact_positions[i]`` of an
     * ``output_pixel_capacity``-wide accumulator.  The other positions would
     * receive exact zeros, which change no accumulator value. */
    const int out_capacity = exact_positions == nullptr ? pixel_capacity : output_pixel_capacity;
    const int rotation = FLAT_ROWS ? 0 : static_cast<int>(blockIdx.x);
    const int64_t row = FLAT_ROWS
        ? static_cast<int64_t>(blockIdx.x)
        : static_cast<int64_t>(blockIdx.y) * n_rotations +
              static_cast<int64_t>(blockIdx.x);
    int64_t batch = FLAT_ROWS ? -1 : static_cast<int64_t>(blockIdx.y);
    const int logical_pixel_count = runtime_logical_pixel_count == nullptr
        ? out_capacity
        : runtime_logical_pixel_count[0];
    if (logical_pixel_count < 0 || logical_pixel_count > out_capacity) {
        if (row == 0 && threadIdx.x == 0)
            output[0] = nanf("");
        return;
    }
    if constexpr (FLAT_ROWS) {
        if (row >= row_count) return;
        batch = static_cast<int64_t>(row_image_ids[row]);
        /* Padding rows contribute nothing and read no term. */
        if (batch < 0) return;
        if (batch >= batch_size) {
            /* Fail closed on an out-of-range map, as the runtime kernels do. */
            if (threadIdx.x == 0) output[0] = nanf("");
            return;
        }
        if (row > 0 && static_cast<int64_t>(row_image_ids[row - 1]) == batch) return;
        /* Every return above is uniform over the block.  The whole block finds
         * the run's end: thread t reads rows row + 1 + t, + blockDim.x, ... and
         * the smallest row of another image is the end.  One thread walking an
         * early-state run of thousands of rows was a chain of dependent loads. */
        __shared__ unsigned long long run_end_shared;
        if (threadIdx.x == 0) run_end_shared = static_cast<unsigned long long>(row_count);
        __syncthreads();
        for (int64_t k = row + 1 + threadIdx.x; k < row_count; k += blockDim.x) {
            if (static_cast<int64_t>(row_image_ids[k]) != batch) {
                atomicMin(&run_end_shared, static_cast<unsigned long long>(k));
                break;
            }
        }
        __syncthreads();
        const int64_t run_end = static_cast<int64_t>(run_end_shared);
        /* A run's pixels are spread over gridDim.y blocks, one pixel per
         * thread: an early-state M-step block holds only a few images of
         * thousands of rows each, and one block per run left the GPU nearly
         * idle.  Each (image, pixel) cell still sums its run's rows in row
         * order in one thread, so the values are unchanged. */
        const int pixel = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
        if (pixel >= pixel_capacity) return;
        const int position = exact_positions == nullptr ? pixel : exact_positions[pixel];
        if (position >= logical_pixel_count) return;
        {
            const int64_t output_index = (batch * out_capacity + position) * 3;
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            /* The loads of kPrefetch rows are issued before their adds, which
             * stay one rounded add each in row order: the same sums, with the
             * loads in flight together instead of one at a time. */
            constexpr int kPrefetch = 16;
            int64_t run_row = row;
            for (; run_row + kPrefetch <= run_end; run_row += kPrefetch) {
                float term0[kPrefetch];
                float term1[kPrefetch];
                float term2[kPrefetch];
#pragma unroll
                for (int j = 0; j < kPrefetch; ++j) {
                    const int64_t input_index = ((run_row + j) * pixel_capacity + pixel) * 3;
                    term0[j] = terms[input_index];
                    term1[j] = terms[input_index + 1];
                    term2[j] = terms[input_index + 2];
                }
#pragma unroll
                for (int j = 0; j < kPrefetch; ++j) {
                    sum0 = __fadd_rn(sum0, term0[j]);
                    sum1 = __fadd_rn(sum1, term1[j]);
                    sum2 = __fadd_rn(sum2, term2[j]);
                }
            }
            for (; run_row < run_end; ++run_row) {
                const int64_t input_index = (run_row * pixel_capacity + pixel) * 3;
                sum0 = __fadd_rn(sum0, terms[input_index]);
                sum1 = __fadd_rn(sum1, terms[input_index + 1]);
                sum2 = __fadd_rn(sum2, terms[input_index + 2]);
            }
            atomicAdd(&output[output_index], sum0);
            atomicAdd(&output[output_index + 1], sum1);
            atomicAdd(&output[output_index + 2], sum2);
        }
        return;
    }
    for (int pixel = threadIdx.x;
         pixel < logical_pixel_count;
         pixel += blockDim.x) {
        // Reached only without FLAT_ROWS: the flat-row branch above returns.
        const int64_t input_index = ((batch * n_rotations + rotation) * pixel_capacity + pixel) * 3;
        const int64_t output_index =
            (batch * pixel_capacity + pixel) * 3;
        atomicAdd(&output[output_index], terms[input_index]);
        atomicAdd(&output[output_index + 1], terms[input_index + 1]);
        atomicAdd(&output[output_index + 2], terms[input_index + 2]);
    }
}

cudaError_t launch_relion_wavg_rotation_atomic_runtime_triplet_add_f32(
    cudaStream_t stream,
    const float* terms,
    float* output,
    int64_t batch_size,
    int64_t n_rotations,
    int64_t pixel_capacity,
    const int32_t* runtime_logical_pixel_count)
{
    dim3 grid(static_cast<unsigned>(n_rotations), static_cast<unsigned>(batch_size));
    dim3 block(256);
    relion_wavg_rotation_atomic_runtime_triplet_f32_kernel<false><<<grid, block, 0, stream>>>(
        terms,
        output,
        static_cast<int>(n_rotations),
        static_cast<int>(pixel_capacity),
        runtime_logical_pixel_count);
    return cudaGetLastError();
}

cudaError_t launch_relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
    cudaStream_t stream,
    const float* terms,
    const int32_t* row_image_ids,
    float* output,
    int64_t batch_size,
    int64_t row_count,
    int64_t pixel_capacity,
    const int32_t* runtime_logical_pixel_count,
    const int32_t* exact_positions = nullptr,
    int64_t output_pixel_capacity = 0)
{
    if (row_count == 0 || pixel_capacity == 0) return cudaSuccess;
    dim3 block(256);
    dim3 grid(
        static_cast<unsigned>(row_count),
        static_cast<unsigned>((pixel_capacity + block.x - 1) / block.x));
    relion_wavg_rotation_atomic_runtime_triplet_f32_kernel<true><<<grid, block, 0, stream>>>(
        terms,
        output,
        0,
        static_cast<int>(pixel_capacity),
        runtime_logical_pixel_count,
        row_image_ids,
        row_count,
        batch_size,
        exact_positions,
        static_cast<int>(output_pixel_capacity));
    return cudaGetLastError();
}

/* Keep RELION's per-rotation, per-pixel Wavg translation reduction inside one
 * CUDA thread.  The corresponding JAX reference deliberately uses a
 * translation-order fori_loop; lowering that loop emits multiple loop-body
 * kernel launches for every local-search bucket.  Explicit round-to-nearest
 * operations here retain the same non-contracted float32 arithmetic. */


/* The Wavg launch (relax#91).  A block owns one row and a span of 256-pixel
 * passes (blockIdx.y), so a launch with few rows still fills the GPU, and it
 * stages its row's posterior in shared memory, at most kWavgTranslationTile
 * translations at a time, with the list of the translations that carry
 * weight: the translation loop then reads neither the posterior from global
 * memory nor the zero-weight translations.  Each output element is computed
 * by the same statements in the same translation order as before. */
constexpr int kWavgThreads = 256;
constexpr int kWavgTranslationTile = 2048;
// Blocks a launch aims for: the pixel passes are split until the grid has them.
constexpr int64_t kWavgTargetBlocks = 8192;

__host__ __device__ inline int64_t wavg_translation_tile(int64_t translation_count)
{
    return translation_count < kWavgTranslationTile ? translation_count : kWavgTranslationTile;
}

template <bool INDEXED_RECTANGLE = false, bool FLAT_ROWS = false>
__global__ void __launch_bounds__(kWavgThreads)
relion_wavg_sequential_runtime_triplet_f32_kernel(
    const float2* __restrict__ projections,
    const float* __restrict__ raw_ctf,
    const float* __restrict__ scale,
    const float2* __restrict__ shifted_images,
    const float* __restrict__ posterior,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    const int32_t* __restrict__ runtime_logical_pixel_count,
    const double* __restrict__ full_ctf = nullptr,
    const int32_t* __restrict__ exact_positions = nullptr,
    const int32_t* __restrict__ recon_indices = nullptr,
    int64_t rectangle_capacity = 0,
    int64_t full_pixel_count = 0,
    const int32_t* __restrict__ invalid = nullptr,
    const int32_t* __restrict__ row_image_ids = nullptr,
    int64_t row_count = 0)
{
    /* ``FLAT_ROWS`` packs the rectangular ``[batch, rotation]`` grid into one
     * ``[row]`` axis whose image address is ``row_image_ids[row]``, the same
     * substitution the flat-row fine scorer makes.  Every arithmetic statement
     * below, the translation order included, is the rectangular body. */
    static_assert(!(INDEXED_RECTANGLE && FLAT_ROWS),
                  "indexed-rectangle and flat-row addressing are exclusive");
    // blockIdx.x is the row, ``batch * rotation_count + rotation`` on the
    // rectangular grids; blockIdx.y the block's span of pixel passes.
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t rotation = FLAT_ROWS ? 0 : row % rotation_count;
    int64_t batch = FLAT_ROWS ? -1 : row / rotation_count;
    if constexpr (FLAT_ROWS) {
        if (row >= row_count) return;
    } else {
        if (batch >= batch_size || rotation >= rotation_count) return;
    }
    if constexpr (INDEXED_RECTANGLE) { if (*invalid) return; }
    const int64_t logical_pixel_count = runtime_logical_pixel_count == nullptr
        ? pixel_capacity
        : static_cast<int64_t>(runtime_logical_pixel_count[0]);
    if (logical_pixel_count < 0 || logical_pixel_count > pixel_capacity) {
        if (row == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            output[0] = nanf("");
        return;
    }
    // This block's pixels: passes [pass_begin, pass_end) of kWavgThreads over
    // the physical capacity (the logical prefix is a device scalar).
    const int64_t pass_count = (pixel_capacity + kWavgThreads - 1) / kWavgThreads;
    const int64_t passes_per_block = (pass_count + gridDim.y - 1) / gridDim.y;
    const int64_t pass_begin = static_cast<int64_t>(blockIdx.y) * passes_per_block;
    const int64_t pass_end = min(pass_count, pass_begin + passes_per_block);
    if constexpr (FLAT_ROWS) {
        batch = static_cast<int64_t>(row_image_ids[row]);
        /* Padding rows read nothing and keep the zeros the launcher wrote. */
        if (batch < 0) return;
        if (batch >= batch_size) {
            /* Fail closed on an out-of-range map, as the runtime kernels do. */
            for (int64_t pixel = pass_begin * kWavgThreads + threadIdx.x;
                 pixel < min(pixel_capacity, pass_end * kWavgThreads);
                 pixel += kWavgThreads) {
                const int64_t invalid_base = (row * pixel_capacity + pixel) * 3;
                output[invalid_base] = nanf("");
                output[invalid_base + 1] = nanf("");
                output[invalid_base + 2] = nanf("");
            }
            return;
        }
    }
    const float batch_scale = scale[batch];
    const int64_t posterior_base = FLAT_ROWS
        ? row * translation_count
        : (batch * rotation_count + rotation) * translation_count;
    const int64_t shifted_stride = INDEXED_RECTANGLE ? rectangle_capacity : pixel_capacity;
    const int64_t shifted_base = batch * translation_count * shifted_stride;
    const int64_t projection_base = FLAT_ROWS
        ? row * pixel_capacity
        : (batch * rotation_count + rotation) * pixel_capacity;
    const int64_t pixel_end = INDEXED_RECTANGLE ? pixel_capacity : logical_pixel_count;
    // Uniform over the block: a span past the pixel loop's bound has no work.
    if (pass_begin * kWavgThreads >= pixel_end) return;

    // The row's posterior, a tile of translations at a time, and the tile's
    // translations with nonzero weight, in increasing order.  The fine
    // posterior is zero outside RELION's significant samples.  A zero weight
    // adds +-0 to accumulators that start at +0 and, in round-to-nearest,
    // never become -0, so skipping it leaves every sum bitwise unchanged and
    // saves the image read.
    const int64_t tile = wavg_translation_tile(translation_count);
    extern __shared__ float wavg_shared[];
    float* shared_weights = wavg_shared;                                    // [tile]
    int32_t* shared_active = reinterpret_cast<int32_t*>(wavg_shared + tile);  // [tile]
    __shared__ int32_t shared_active_count;
    auto stage_tile = [&](int64_t tile_start, int tile_count) {
        __syncthreads();
        for (int offset = threadIdx.x; offset < tile_count; offset += kWavgThreads)
            shared_weights[offset] = posterior[posterior_base + tile_start + offset];
        __syncthreads();
        if (threadIdx.x < 32) {
            int32_t count = 0;
            for (int first = 0; first < tile_count; first += 32) {
                const int offset = first + static_cast<int>(threadIdx.x);
                const bool weighted = offset < tile_count && shared_weights[offset] != 0.0f;
                const unsigned mask = __ballot_sync(0xffffffffu, weighted);
                if (weighted)
                    shared_active[count + __popc(mask & ((1u << threadIdx.x) - 1u))] = offset;
                count += __popc(mask);
            }
            if (threadIdx.x == 0) shared_active_count = count;
        }
        __syncthreads();
    };
    const bool one_tile = translation_count <= tile;
    if (one_tile) stage_tile(0, static_cast<int>(translation_count));

    // Passes, not a per-thread pixel loop: every thread takes the same number
    // of trips, so a row with more than one tile can restage inside them.
    for (int64_t pass = pass_begin; pass < pass_end; ++pass)
    {
        const int64_t pixel = pass * kWavgThreads + threadIdx.x;
        const bool pixel_in_range = pixel < pixel_end;
        const int64_t stored_pixel = !pixel_in_range
            ? 0
            : (INDEXED_RECTANGLE ? static_cast<int64_t>(exact_positions[pixel]) : pixel);
        const int64_t output_base = INDEXED_RECTANGLE
            ? ((batch * rotation_count + rotation) * rectangle_capacity + stored_pixel) * 3
            : (projection_base + pixel) * 3;
        // The indexed rectangle writes zeros past the logical prefix.
        const bool pixel_live = pixel_in_range && pixel < logical_pixel_count;
        if (INDEXED_RECTANGLE && pixel_in_range && !pixel_live) {
            output[output_base] = 0.0f;
            output[output_base + 1] = 0.0f;
            output[output_base + 2] = 0.0f;
        }
        float ref_real = 0.0f;
        float ref_imag = 0.0f;
        float ref_abs2 = 0.0f;
        if (pixel_live) {
            const float ctf_value = INDEXED_RECTANGLE
                ? __double2float_rn(full_ctf[batch * full_pixel_count + recon_indices[pixel]])
                : raw_ctf[batch * pixel_capacity + pixel];
            const float ctf_with_scale = __fmul_rn(ctf_value, batch_scale);
            const float2 projection = projections[projection_base + pixel];
            ref_real = __fmul_rn(projection.x, ctf_with_scale);
            ref_imag = __fmul_rn(projection.y, ctf_with_scale);
            ref_abs2 = __fadd_rn(
                __fmul_rn(ref_real, ref_real),
                __fmul_rn(ref_imag, ref_imag));
        }

        float xa_raw = 0.0f;
        float aa_raw = 0.0f;
        float diff2 = 0.0f;
        for (int64_t tile_start = 0; tile_start < translation_count; tile_start += tile)
        {
            if (!one_tile)
                stage_tile(tile_start, static_cast<int>(min(tile, translation_count - tile_start)));
            if (!pixel_live) continue;
            const int32_t active_count = shared_active_count;
            for (int32_t active = 0; active < active_count; ++active)
            {
                const int32_t offset = shared_active[active];
                const float weight = shared_weights[offset];
                const float2 translated = shifted_images[
                    shifted_base + (tile_start + offset) * shifted_stride + stored_pixel];
                const float diff_real = __fsub_rn(ref_real, translated.x);
                const float diff_imag = __fsub_rn(ref_imag, translated.y);
                const float diff_abs2 = __fadd_rn(
                    __fmul_rn(diff_real, diff_real),
                    __fmul_rn(diff_imag, diff_imag));
                const float cross = __fadd_rn(
                    __fmul_rn(ref_real, translated.x),
                    __fmul_rn(ref_imag, translated.y));
                xa_raw = __fadd_rn(xa_raw, __fmul_rn(weight, cross));
                aa_raw = __fadd_rn(aa_raw, __fmul_rn(weight, ref_abs2));
                diff2 = __fadd_rn(diff2, __fmul_rn(weight, diff_abs2));
            }
        }
        if (!pixel_live) continue;

        const float safe_scale = fmaxf(batch_scale, 1.0e-30f);
        output[output_base] = __fmul_rn(xa_raw, __frcp_rn(safe_scale));
        output[output_base + 1] = __fmul_rn(
            aa_raw, __frcp_rn(__fmul_rn(safe_scale, safe_scale)));
        output[output_base + 2] = diff2;
    }
}

// The launch geometry of relion_wavg_sequential_runtime_triplet_f32_kernel:
// one block per row by spans of pixel passes over the physical capacity, split
// until the grid has about kWavgTargetBlocks blocks, and its shared bytes.
struct WavgTripletLaunch {
    dim3 grid;
    size_t shared_bytes;
};

inline WavgTripletLaunch wavg_triplet_launch(
    int64_t rows, int64_t pixel_capacity, int64_t translation_count)
{
    const int64_t passes = (pixel_capacity + kWavgThreads - 1) / kWavgThreads;
    int64_t spans = (kWavgTargetBlocks + rows - 1) / rows;
    spans = spans < 1 ? 1 : (spans > passes ? passes : spans);
    return WavgTripletLaunch{
        dim3(static_cast<unsigned>(rows), static_cast<unsigned>(spans)),
        static_cast<size_t>(wavg_translation_tile(translation_count)) *
            (sizeof(float) + sizeof(int32_t))};
}


/* The flat-row Wavg triplet of a one-image launch, tiled (relax#98): a lone particle's M-step block is hundreds of
 * rows reading one image's shifted tile, which the row kernel reads once per row.  A block owns kTiledRows rows by
 * kWavgTiledPixels pixels and stages, kWavgTiledTranslations of the row tile's active translations at a time
 * (relax_tiled_translations), the rows' weights and the shifted values in shared memory.  Every output is the row
 * kernel's statements in the same translation order: the zero weights it skips add nothing. */
constexpr int kWavgTiledPixels = 64;
constexpr int kWavgTiledTranslations = 32;
constexpr int kWavgTiledPixelLanes = 16;
constexpr int kWavgTiledRowLanes = kWavgThreads / kWavgTiledPixelLanes;  // 16
constexpr int kWavgTiledRowsPerThread = relax_tiled_translations::kTiledRows / kWavgTiledRowLanes;  // 2
constexpr int kWavgTiledPixelsPerThread = kWavgTiledPixels / kWavgTiledPixelLanes;  // 4

__global__ void __launch_bounds__(kWavgThreads)
relion_wavg_tiled_one_image_triplet_f32_kernel(
    const float2* __restrict__ projections,          // [Q, P]
    const float* __restrict__ raw_ctf,               // [1, P]
    const float* __restrict__ scale,                 // [1]
    const float2* __restrict__ shifted_images,       // [1, T, P]
    const float* __restrict__ posterior,             // [Q, T]
    const int32_t* __restrict__ active_lists,        // [row tiles, T]
    const int32_t* __restrict__ active_counts,       // [row tiles]
    float* __restrict__ output,                      // [Q, P, 3]
    int64_t translation_count,
    int64_t pixel_capacity,
    const int32_t* __restrict__ runtime_logical_pixel_count,
    const int32_t* __restrict__ row_image_ids,
    int64_t row_count)
{
    constexpr int kRows = relax_tiled_translations::kTiledRows;
    const int64_t row_base = static_cast<int64_t>(blockIdx.x) * kRows;
    const int64_t pixel_base = static_cast<int64_t>(blockIdx.y) * kWavgTiledPixels;
    const int64_t logical_pixel_count = runtime_logical_pixel_count == nullptr
        ? pixel_capacity
        : static_cast<int64_t>(runtime_logical_pixel_count[0]);
    if (logical_pixel_count < 0 || logical_pixel_count > pixel_capacity) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) output[0] = nanf("");
        return;
    }
    __shared__ float shared_weight[kWavgTiledTranslations][kRows + 1];
    __shared__ float2 shared_value[kWavgTiledTranslations][kWavgTiledPixels];
    __shared__ int32_t shared_live[kRows];
    if (threadIdx.x < kRows) {
        const int64_t row = row_base + threadIdx.x;
        // 1: image 0; 0: past the rows or padding (zeros); -1: an id past the one image (fails closed).
        shared_live[threadIdx.x] = row < row_count ? (row_image_ids[row] == 0 ? 1 : (row_image_ids[row] > 0 ? -1 : 0)) : 0;
    }
    __syncthreads();
    const int pixel_lane = threadIdx.x % kWavgTiledPixelLanes;
    const int row_lane = threadIdx.x / kWavgTiledPixelLanes;
    const float batch_scale = scale[0];
    float ref_real[kWavgTiledRowsPerThread][kWavgTiledPixelsPerThread];
    float ref_imag[kWavgTiledRowsPerThread][kWavgTiledPixelsPerThread];
    float ref_abs2[kWavgTiledRowsPerThread][kWavgTiledPixelsPerThread];
    float xa_raw[kWavgTiledRowsPerThread][kWavgTiledPixelsPerThread];
    float aa_raw[kWavgTiledRowsPerThread][kWavgTiledPixelsPerThread];
    float diff2[kWavgTiledRowsPerThread][kWavgTiledPixelsPerThread];
#pragma unroll
    for (int i = 0; i < kWavgTiledRowsPerThread; ++i) {
        const int local_row = row_lane + i * kWavgTiledRowLanes;
#pragma unroll
        for (int j = 0; j < kWavgTiledPixelsPerThread; ++j) {
            const int64_t pixel = pixel_base + pixel_lane + j * kWavgTiledPixelLanes;
            ref_real[i][j] = 0.0f;
            ref_imag[i][j] = 0.0f;
            ref_abs2[i][j] = 0.0f;
            xa_raw[i][j] = 0.0f;
            aa_raw[i][j] = 0.0f;
            diff2[i][j] = 0.0f;
            if (shared_live[local_row] == 1 && pixel < logical_pixel_count) {
                const float ctf_with_scale = __fmul_rn(raw_ctf[pixel], batch_scale);
                const float2 projection = projections[(row_base + local_row) * pixel_capacity + pixel];
                ref_real[i][j] = __fmul_rn(projection.x, ctf_with_scale);
                ref_imag[i][j] = __fmul_rn(projection.y, ctf_with_scale);
                ref_abs2[i][j] = __fadd_rn(
                    __fmul_rn(ref_real[i][j], ref_real[i][j]), __fmul_rn(ref_imag[i][j], ref_imag[i][j]));
            }
        }
    }
    const int64_t active_count = active_counts[blockIdx.x];
    const int32_t* tile_active = active_lists + static_cast<int64_t>(blockIdx.x) * translation_count;
    // The next chunk's weights and values are loaded into registers while the current one is summed from shared
    // memory.
    constexpr int kLoadsW = kWavgTiledTranslations * kRows / kWavgThreads;
    constexpr int kLoadsV = kWavgTiledTranslations * kWavgTiledPixels / kWavgThreads;
    float next_weight[kLoadsW];
    float2 next_value[kLoadsV];
    auto load_chunk = [&](int64_t chunk_first) {
        const int chunk_count = active_count - chunk_first < kWavgTiledTranslations
            ? static_cast<int>(active_count - chunk_first)
            : kWavgTiledTranslations;
        // Consecutive threads read one row's consecutive active translations (the posterior is row-major).
#pragma unroll
        for (int k = 0; k < kLoadsW; ++k) {
            const int index = threadIdx.x + k * kWavgThreads;
            const int i = index / kWavgTiledTranslations;
            const int t = index - i * kWavgTiledTranslations;
            next_weight[k] = (t < chunk_count && shared_live[i] == 1)
                ? posterior[(row_base + i) * translation_count + tile_active[chunk_first + t]]
                : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < kLoadsV; ++k) {
            const int index = threadIdx.x + k * kWavgThreads;
            const int t = index / kWavgTiledPixels;
            const int j = index - t * kWavgTiledPixels;
            const int64_t pixel = pixel_base + j;
            next_value[k] = (t < chunk_count && pixel < pixel_capacity)
                ? shifted_images[static_cast<int64_t>(tile_active[chunk_first + t]) * pixel_capacity + pixel]
                : make_float2(0.0f, 0.0f);
        }
    };
    if (active_count > 0) load_chunk(0);
    for (int64_t first = 0; first < active_count; first += kWavgTiledTranslations) {
        const int count = active_count - first < kWavgTiledTranslations
            ? static_cast<int>(active_count - first)
            : kWavgTiledTranslations;
#pragma unroll
        for (int k = 0; k < kLoadsW; ++k) {
            const int index = threadIdx.x + k * kWavgThreads;
            shared_weight[index % kWavgTiledTranslations][index / kWavgTiledTranslations] = next_weight[k];
        }
#pragma unroll
        for (int k = 0; k < kLoadsV; ++k) {
            const int index = threadIdx.x + k * kWavgThreads;
            shared_value[index / kWavgTiledPixels][index % kWavgTiledPixels] = next_value[k];
        }
        __syncthreads();
        if (first + kWavgTiledTranslations < active_count) load_chunk(first + kWavgTiledTranslations);
        for (int t = 0; t < count; ++t) {
            // The translation's values, read from shared memory once for all of the thread's rows.
            float2 values[kWavgTiledPixelsPerThread];
#pragma unroll
            for (int j = 0; j < kWavgTiledPixelsPerThread; ++j) values[j] = shared_value[t][pixel_lane + j * kWavgTiledPixelLanes];
#pragma unroll
            for (int i = 0; i < kWavgTiledRowsPerThread; ++i) {
                const float weight = shared_weight[t][row_lane + i * kWavgTiledRowLanes];
                if (weight == 0.0f) continue;
#pragma unroll
                for (int j = 0; j < kWavgTiledPixelsPerThread; ++j) {
                    const float2 translated = values[j];
                    const float diff_real = __fsub_rn(ref_real[i][j], translated.x);
                    const float diff_imag = __fsub_rn(ref_imag[i][j], translated.y);
                    const float diff_abs2 = __fadd_rn(__fmul_rn(diff_real, diff_real), __fmul_rn(diff_imag, diff_imag));
                    const float cross = __fadd_rn(
                        __fmul_rn(ref_real[i][j], translated.x), __fmul_rn(ref_imag[i][j], translated.y));
                    xa_raw[i][j] = __fadd_rn(xa_raw[i][j], __fmul_rn(weight, cross));
                    aa_raw[i][j] = __fadd_rn(aa_raw[i][j], __fmul_rn(weight, ref_abs2[i][j]));
                    diff2[i][j] = __fadd_rn(diff2[i][j], __fmul_rn(weight, diff_abs2));
                }
            }
        }
        __syncthreads();
    }
    const float safe_scale = fmaxf(batch_scale, 1.0e-30f);
#pragma unroll
    for (int i = 0; i < kWavgTiledRowsPerThread; ++i) {
        const int local_row = row_lane + i * kWavgTiledRowLanes;
        const int32_t live = shared_live[local_row];
        if (live == 0) continue;
        const int64_t row = row_base + local_row;
#pragma unroll
        for (int j = 0; j < kWavgTiledPixelsPerThread; ++j) {
            const int64_t pixel = pixel_base + pixel_lane + j * kWavgTiledPixelLanes;
            const int64_t output_base = (row * pixel_capacity + pixel) * 3;
            if (live < 0) {
                if (pixel < pixel_capacity) {
                    output[output_base] = nanf("");
                    output[output_base + 1] = nanf("");
                    output[output_base + 2] = nanf("");
                }
                continue;
            }
            if (pixel >= logical_pixel_count) continue;
            output[output_base] = __fmul_rn(xa_raw[i][j], __frcp_rn(safe_scale));
            output[output_base + 1] = __fmul_rn(aa_raw[i][j], __frcp_rn(__fmul_rn(safe_scale, safe_scale)));
            output[output_base + 2] = diff2[i][j];
        }
    }
}

// A one-image launch with at least this many rows takes the tiled kernel (as the translate-sum's shifted table).
constexpr int64_t kWavgTiledMinRows = 64;

cudaError_t launch_relion_wavg_sequential_runtime_flat_rows_triplet_f32(
    cudaStream_t stream,
    ffi::ScratchAllocator& scratch,
    const float2* projections,
    const int32_t* row_image_ids,
    const float* raw_ctf,
    const float* scale,
    const float2* shifted_images,
    const float* posterior,
    float* output,
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    const int32_t* runtime_logical_pixel_count)
{
    const int64_t output_count = row_count * pixel_capacity;
    if (output_count == 0) return cudaSuccess;
    /* The zero fill owns the physical pixel tail and every padding row; the
     * kernel then writes only the rows and pixels it is allowed to read. */
    cudaError_t err = cudaMemsetAsync(
        output,
        0,
        static_cast<size_t>(output_count) * 3 * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;
    if (batch_size == 1 && row_count >= kWavgTiledMinRows) {
        int32_t* active = nullptr;
        err = relax_tiled_translations::list_row_tile_translations(
            stream, scratch, row_image_ids, posterior, nullptr, row_count, translation_count, &active);
        if (err != cudaSuccess) return err;
        const int64_t tiles = (row_count + relax_tiled_translations::kTiledRows - 1) / relax_tiled_translations::kTiledRows;
        const dim3 grid(
            static_cast<unsigned>(tiles),
            static_cast<unsigned>((pixel_capacity + kWavgTiledPixels - 1) / kWavgTiledPixels));
        relion_wavg_tiled_one_image_triplet_f32_kernel<<<grid, kWavgThreads, 0, stream>>>(
            projections, raw_ctf, scale, shifted_images, posterior, active, active + tiles * translation_count, output,
            translation_count, pixel_capacity, runtime_logical_pixel_count, row_image_ids, row_count);
        return cudaGetLastError();
    }
    const WavgTripletLaunch geometry =
        wavg_triplet_launch(row_count, pixel_capacity, translation_count);
    relion_wavg_sequential_runtime_triplet_f32_kernel<false, true><<<
        geometry.grid, kWavgThreads, geometry.shared_bytes, stream>>>(
        projections,
        raw_ctf,
        scale,
        shifted_images,
        posterior,
        output,
        batch_size,
        0,
        translation_count,
        pixel_capacity,
        runtime_logical_pixel_count,
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        nullptr,
        row_image_ids,
        row_count);
    return cudaGetLastError();
}

cudaError_t launch_relion_fused_x_half_backproject(
    cudaStream_t stream,
    float2* data_volume,
    float* weight_volume,
    const float2* data_rows,
    const float* weight_rows,
    const int32_t* pixel_indices,
    const float* rot,
    int64_t n_rows,
    int64_t n_pixels,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const float c0 = (float)(N0 / 2);
    const float c1 = (float)(N1 / 2);
    const float c2 = (float)(N2 / 2);
    const float max_r2 = (float)max_r2_x4 / 4.0f;
    dim3 grid((unsigned)n_rows, 1);
    dim3 block(128);
    relion_fused_x_half_backproject_kernel<float, float2, false, true><<<grid, block, 0, stream>>>(
        data_volume, weight_volume, data_rows, weight_rows, pixel_indices, rot,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        (int)n_pixels, (int)image_h, (int)image_w,
        (int)N0, (int)N1, N2_eff, c0, c1, c2,
        (int)upsampling, max_r2, (int)n_rows);
    return cudaGetLastError();
}

cudaError_t launch_relion_fused_x_half_backproject_f64(
    cudaStream_t stream,
    double2* data_volume,
    double* weight_volume,
    const double2* data_rows,
    const double* weight_rows,
    const int32_t* pixel_indices,
    const double* rot,
    int64_t n_rows,
    int64_t n_pixels,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const double c0 = double(N0 / 2);
    const double c1 = double(N1 / 2);
    const double c2 = double(N2 / 2);
    const double max_r2 = double(max_r2_x4) / 4.0;
    dim3 grid((unsigned)n_rows, 1);
    dim3 block(128);
    relion_fused_x_half_backproject_kernel<double, double2, false, true><<<grid, block, 0, stream>>>(
        data_volume, weight_volume, data_rows, weight_rows, pixel_indices, rot,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        (int)n_pixels, (int)image_h, (int)image_w,
        (int)N0, (int)N1, N2_eff, c0, c1, c2,
        (int)upsampling, max_r2, (int)n_rows);
    return cudaGetLastError();
}


/* ================================================================== */
/*                    XLA  FFI  handlers                               */
/* ================================================================== */

namespace {

constexpr int kRelionPreprocessBlockSize = 128;
constexpr int kRelionSoftMaskBlocks = 128;
constexpr int kRelionEulerBlockSize = 128;
constexpr int kRelionTranslateScoreBlockSize = 256;
constexpr int kRelionTranslateBprefBlockSize = 256;
constexpr int kRelionCoarseDiff2BlockSize = 128;
constexpr int kRelionCoarseEulersPerBlock = 16;
constexpr int kRelionCoarsePrefetchFraction = 4;
constexpr int kRelionFineDiff2BlockSize = 256;
// The deployed REF3D fine kernel is instantiated with a seven-translation
// shared-memory capacity. Its makeJobsForDiff2Fine call nevertheless uses
// D2F_CHUNK_DATA3D=4 when refIs3D, so native jobs contain at most four
// translations. Preserve both observable constants here.
constexpr int kRelionFineDiff2TranslationCapacity = 7;
constexpr int kRelionFineDiff2Ref3dJobChunk = 4;

template <typename T, bool DoRight>
__global__ void relion_make_scoring_rotations_kernel(
    const T* eulers_deg,
    const T* right_matrix,
    T* scorer_rotations,
    int64_t orientation_count)
{
    int64_t oid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (oid >= orientation_count) return;

    T a = eulers_deg[3 * oid] * static_cast<T>(3.14159265358979323846) /
          static_cast<T>(180.0);
    T b = eulers_deg[3 * oid + 1] * static_cast<T>(3.14159265358979323846) /
          static_cast<T>(180.0);
    T g = eulers_deg[3 * oid + 2] * static_cast<T>(3.14159265358979323846) /
          static_cast<T>(180.0);
    T ca, sa, cb, sb, cg, sg, cc, cs, sc, ss;
    T A[9], B[9];
    if constexpr (std::is_same_v<T, float>) {
        sincosf(a, &sa, &ca);
        sincosf(b, &sb, &cb);
        sincosf(g, &sg, &cg);
    } else {
        sincos(a, &sa, &ca);
        sincos(b, &sb, &cb);
        sincos(g, &sg, &cg);
    }
    cc = cb * ca;
    cs = cb * sa;
    sc = sb * ca;
    ss = sb * sa;
    A[0] = cg * cc - sg * sa;
    A[1] = cg * cs + sg * ca;
    A[2] = -cg * sb;
    A[3] = -sg * cc - cg * sa;
    A[4] = -sg * cs + cg * ca;
    A[5] = sg * sb;
    A[6] = sc;
    A[7] = ss;
    A[8] = cb;

    if constexpr (DoRight) {
        for (int i = 0; i < 9; ++i) B[i] = static_cast<T>(0);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    B[3 * i + j] += A[3 * i + k] * right_matrix[3 * k + j];
    } else {
        for (int i = 0; i < 9; ++i) B[i] = A[i];
    }

    for (int i = 0; i < 9; ++i) scorer_rotations[9 * oid + i] = B[i];
}

// make_eulers_3D<invert=true, doL=true, doR> for tilt images: one left matrix L per image
// (Aproj times the optics scale), B = L (A R) and RELION's adjugate inverse, float32
// (acc/cuda/cuda_kernels/helper.cuh:714-811). Output [n_left, N, 3, 3] in RECOVAR's
// scorer frame, the transpose of RELION's stored eulers.
template <bool DoRight>
__global__ void relion_make_scoring_rotations_left_f32_kernel(
    const float* eulers_deg,
    const float* right_matrix,
    const float* left_matrices,
    float* scorer_rotations,
    int64_t orientation_count,
    int64_t left_count)
{
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= orientation_count * left_count) return;
    const int64_t image = flat / orientation_count;
    const int64_t oid = flat - image * orientation_count;
    const float* L = left_matrices + 9 * image;

    float a = eulers_deg[3 * oid] * static_cast<float>(3.14159265358979323846) / 180.0f;
    float b = eulers_deg[3 * oid + 1] * static_cast<float>(3.14159265358979323846) / 180.0f;
    float g = eulers_deg[3 * oid + 2] * static_cast<float>(3.14159265358979323846) / 180.0f;
    float ca, sa, cb, sb, cg, sg;
    sincosf(a, &sa, &ca);
    sincosf(b, &sb, &cb);
    sincosf(g, &sg, &cg);
    const float cc = cb * ca;
    const float cs = cb * sa;
    const float sc = sb * ca;
    const float ss = sb * sa;
    float A[9], B[9];
    A[0] = (cg * cc - sg * sa);
    A[1] = (cg * cs + sg * ca);
    A[2] = (-cg * sb);
    A[3] = (-sg * cc - cg * sa);
    A[4] = (-sg * cs + cg * ca);
    A[5] = (sg * sb);
    A[6] = (sc);
    A[7] = (ss);
    A[8] = (cb);
    if constexpr (DoRight) {
        for (int i = 0; i < 9; ++i) B[i] = 0.f;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    B[i * 3 + j] += A[i * 3 + k] * right_matrix[k * 3 + j];
        for (int i = 0; i < 9; ++i) A[i] = B[i];
    }
    for (int i = 0; i < 9; ++i) B[i] = 0.f;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                B[i * 3 + j] += L[i * 3 + k] * A[k * 3 + j];
    const float det = B[0] * (B[4] * B[8] - B[7] * B[5])
        - B[1] * (B[3] * B[8] - B[6] * B[5])
        + B[2] * (B[3] * B[7] - B[6] * B[4]);
    float E[9];
    E[0] = (B[4] * B[8] - B[7] * B[5]) / det;
    E[1] = (B[7] * B[2] - B[1] * B[8]) / det;
    E[2] = (B[1] * B[5] - B[4] * B[2]) / det;
    E[3] = (B[5] * B[6] - B[8] * B[3]) / det;
    E[4] = (B[8] * B[0] - B[2] * B[6]) / det;
    E[5] = (B[2] * B[3] - B[5] * B[0]) / det;
    E[6] = (B[3] * B[7] - B[6] * B[4]) / det;
    E[7] = (B[6] * B[1] - B[0] * B[7]) / det;
    E[8] = (B[0] * B[4] - B[3] * B[1]) / det;
    float* out = scorer_rotations + 9 * flat;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[3 * i + j] = E[3 * j + i];
}

template <typename T, bool DoRight>
cudaError_t launch_relion_make_scoring_rotations(
    cudaStream_t stream,
    const T* eulers_deg,
    const T* right_matrix,
    T* scorer_rotations,
    int64_t orientation_count)
{
    if (orientation_count == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (orientation_count + kRelionEulerBlockSize - 1) / kRelionEulerBlockSize);
    relion_make_scoring_rotations_kernel<T, DoRight>
        <<<blocks, kRelionEulerBlockSize, 0, stream>>>(
            eulers_deg, right_matrix, scorer_rotations, orientation_count);
    return cudaGetLastError();
}

__device__ __forceinline__ float2 relion_score_translate_f32(
    float2 value,
    int x,
    int y,
    float tx,
    float ty)
{
    // Match the PTX emitted for the deployed RELION fine Gaussian kernel.
    // Its phase is one rounded y product followed by x FMA.  The real
    // component is contracted by the H100 driver JIT as cosine*real plus the
    // rounded negative sine*imaginary product.  The imaginary component uses
    // the complementary ordering: a rounded cosine*imaginary product is the
    // addend to sine*real.  Writing both source expressions directly lets
    // newer offline nvcc versions choose a different addend for the
    // imaginary FMA, which changes translated pixels and can move the final
    // diff2 by one binary32 ULP.
    const float phase = __fmaf_rn(
        static_cast<float>(x), tx,
        __fmul_rn(static_cast<float>(y), ty));
    float sine;
    float cosine;
    sincosf(phase, &sine, &cosine);
    const float translated_real = __fmaf_rn(
        cosine, value.x,
        -__fmul_rn(sine, value.y));
    const float translated_imag = __fmaf_rn(
        sine, value.x,
        __fmul_rn(cosine, value.y));
    return make_float2(translated_real, translated_imag);
}

__device__ __forceinline__ float2 relion_coarse_score_translate_f32(
    float2 value,
    int x,
    int y,
    float tx,
    float ty)
{
    // Match the PTX emitted for RELION's REF3D/DATA2D coarse Gaussian
    // scorer.  Its phase uses a rounded y product followed by an x FMA.
    // Unlike the fine scorer, the real component must retain RELION's source
    // expression so the deployed compiler chooses the production operation
    // sequence in this kernel context. The imaginary component is contracted
    // with the rounded cosine*imaginary product as the addend.
    const float phase = __fmaf_rn(
        static_cast<float>(x), tx,
        __fmul_rn(static_cast<float>(y), ty));
    float sine;
    float cosine;
    sincosf(phase, &sine, &cosine);
    const float translated_real = cosine * value.x - sine * value.y;
    const float translated_imag = __fmaf_rn(
        sine, value.x,
        __fmul_rn(cosine, value.y));
    return make_float2(translated_real, translated_imag);
}

// RELION's row label for a half image it is not cropping.  ``fftw.h:99-109``
// sets ``ip = (i < XSIZE) ? i : i - YSIZE`` with ``XSIZE`` the half width, so
// the Nyquist row of an uncropped half image is ``+N/2``; recovar's centered
// packed layout stores that same physical row at ``ky = -N/2``.  Every scoring
// kernel in ``relion_scoring.cuh`` already walks RELION's layout and therefore
// derives ``+N/2`` itself (``if (y > cs/2) y -= cs``); the translate kernels
// below take centered indices instead and must convert.  The two labels differ
// only for non-integer shifts, where ``exp(-i*pi*dy)`` and ``exp(+i*pi*dy)``
// are conjugates.  A cropped window never contains that row.
__device__ __forceinline__ int relion_centered_row_to_relion_label(
    int centered_row,
    int image_h)
{
    return centered_row == -(image_h / 2) ? image_h / 2 : centered_row;
}

// ``translation_angles`` is [T, 2], shared by the batch, or [B, T, 2] (``angles_per_image``), each
// image's own (a subtomogram's tilt images).
__global__ void relion_translate_score_f32_kernel(
    const float2* images,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width,
    bool angles_per_image)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * translation_count * pixel_count;
    if (flat >= total) return;

    int64_t pixel_row = flat % pixel_count;
    int64_t batch_translation = flat / pixel_count;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    int pixel_index = pixel_indices[pixel_row];
    int x = pixel_index % image_half_width;
    int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    const float* angles = translation_angles + 2 * (angles_per_image ? batch_translation : translation);
    float tx = angles[0];
    float ty = angles[1];
    float2 value = images[image * pixel_count + pixel_row];
    shifted[flat] = relion_score_translate_f32(value, x, y, tx, ty);
}

cudaError_t launch_relion_translate_score_f32(
    cudaStream_t stream,
    const float2* images,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width,
    bool angles_per_image)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (total + kRelionTranslateScoreBlockSize - 1) /
        kRelionTranslateScoreBlockSize);
    relion_translate_score_f32_kernel<<<
        blocks, kRelionTranslateScoreBlockSize, 0, stream>>>(
            images,
            translation_angles,
            pixel_indices,
            shifted,
            batch_size,
            translation_count,
            pixel_count,
            image_h,
            image_half_width,
            angles_per_image);
    return cudaGetLastError();
}

__global__ void relion_translate_score_f64_kernel(
    const double2* images,
    const double* translation_angles,
    const int32_t* pixel_indices,
    double2* shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * translation_count * pixel_count;
    if (flat >= total) return;

    int64_t pixel_row = flat % pixel_count;
    int64_t batch_translation = flat / pixel_count;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    int pixel_index = pixel_indices[pixel_row];
    int x = pixel_index % image_half_width;
    int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    double tx = translation_angles[2 * translation];
    double ty = translation_angles[2 * translation + 1];
    double sine;
    double cosine;
    sincos(x * tx + y * ty, &sine, &cosine);

    double2 value = images[image * pixel_count + pixel_row];
    shifted[flat] = make_double2(
        cosine * value.x - sine * value.y,
        cosine * value.y + sine * value.x);
}

cudaError_t launch_relion_translate_score_f64(
    cudaStream_t stream,
    const double2* images,
    const double* translation_angles,
    const int32_t* pixel_indices,
    double2* shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (total + kRelionTranslateScoreBlockSize - 1) /
        kRelionTranslateScoreBlockSize);
    relion_translate_score_f64_kernel<<<
        blocks, kRelionTranslateScoreBlockSize, 0, stream>>>(
            images,
            translation_angles,
            pixel_indices,
            shifted,
            batch_size,
            translation_count,
            pixel_count,
            image_h,
            image_half_width);
    return cudaGetLastError();
}

__global__ void relion_translate_bpref_f32_kernel(
    const float2* images,
    const float* weighted_ctf,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* weighted_shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * translation_count * pixel_count;
    if (flat >= total) return;

    int64_t pixel_row = flat % pixel_count;
    int64_t batch_translation = flat / pixel_count;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    int pixel_index = pixel_indices[pixel_row];
    int x = pixel_index % image_half_width;
    int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    float tx = translation_angles[2 * translation];
    float ty = translation_angles[2 * translation + 1];
    float sine;
    float cosine;
    sincosf(x * tx + y * ty, &sine, &cosine);

    float2 value = images[image * pixel_count + pixel_row];
    float factor = weighted_ctf[image * pixel_count + pixel_row];
    float translated_real = cosine * value.x - sine * value.y;
    // Pin the RELION BPref translatePixel FMA order (captured-bit oracle).
    float translated_imag = __fmaf_rn(
        sine, value.x, __fmul_rn(cosine, value.y));
    weighted_shifted[flat] = make_float2(
        translated_real * factor,
        translated_imag * factor);
}

cudaError_t launch_relion_translate_bpref_f32(
    cudaStream_t stream,
    const float2* images,
    const float* weighted_ctf,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* weighted_shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (total + kRelionTranslateBprefBlockSize - 1) /
        kRelionTranslateBprefBlockSize);
    relion_translate_bpref_f32_kernel<<<
        blocks, kRelionTranslateBprefBlockSize, 0, stream>>>(
            images,
            weighted_ctf,
            translation_angles,
            pixel_indices,
            weighted_shifted,
            batch_size,
            translation_count,
            pixel_count,
            image_h,
            image_half_width);
    return cudaGetLastError();
}

__global__ void relion_translate_bpref_f64_kernel(
    const double2* images,
    const double* weighted_ctf,
    const double* translation_angles,
    const int32_t* pixel_indices,
    double2* weighted_shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * translation_count * pixel_count;
    if (flat >= total) return;

    int64_t pixel_row = flat % pixel_count;
    int64_t batch_translation = flat / pixel_count;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    int pixel_index = pixel_indices[pixel_row];
    int x = pixel_index % image_half_width;
    int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    double tx = translation_angles[2 * translation];
    double ty = translation_angles[2 * translation + 1];
    double sine;
    double cosine;
    sincos(x * tx + y * ty, &sine, &cosine);

    double2 value = images[image * pixel_count + pixel_row];
    double factor = weighted_ctf[image * pixel_count + pixel_row];
    double translated_real = cosine * value.x - sine * value.y;
    double translated_imag = cosine * value.y + sine * value.x;
    weighted_shifted[flat] = make_double2(
        translated_real * factor,
        translated_imag * factor);
}

cudaError_t launch_relion_translate_bpref_f64(
    cudaStream_t stream,
    const double2* images,
    const double* weighted_ctf,
    const double* translation_angles,
    const int32_t* pixel_indices,
    double2* weighted_shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (total + kRelionTranslateBprefBlockSize - 1) /
        kRelionTranslateBprefBlockSize);
    relion_translate_bpref_f64_kernel<<<
        blocks, kRelionTranslateBprefBlockSize, 0, stream>>>(
            images, weighted_ctf, translation_angles, pixel_indices,
            weighted_shifted, batch_size, translation_count, pixel_count,
            image_h, image_half_width);
    return cudaGetLastError();
}



#include "relion_scoring.cuh"

#include "relion_preprocess.cuh"

}  // namespace

ffi::Error RelionMakeScoringRotationsLeftF32Impl(
    cudaStream_t stream,
    int64_t do_right,
    ffi::AnyBuffer eulers_deg,
    ffi::AnyBuffer right_matrix,
    ffi::AnyBuffer left_matrices,
    ffi::Result<ffi::AnyBuffer> scorer_rotations)
{
    if (eulers_deg.element_type() != ffi::DataType::F32 ||
        right_matrix.element_type() != ffi::DataType::F32 ||
        left_matrices.element_type() != ffi::DataType::F32 ||
        scorer_rotations->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsLeftF32: inputs/output must be F32");
    if (do_right != 0 && do_right != 1)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsLeftF32: do_right must be 0 or 1");
    auto euler_dims = eulers_deg.dimensions();
    auto right_dims = right_matrix.dimensions();
    auto left_dims = left_matrices.dimensions();
    auto output_dims = scorer_rotations->dimensions();
    if (euler_dims.size() != 2 || euler_dims[1] != 3 ||
        right_dims.size() != 2 || right_dims[0] != 3 || right_dims[1] != 3 ||
        left_dims.size() != 3 || left_dims[1] != 3 || left_dims[2] != 3 ||
        output_dims.size() != 4 || output_dims[0] != left_dims[0] ||
        output_dims[1] != euler_dims[0] || output_dims[2] != 3 || output_dims[3] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsLeftF32: expected eulers (N,3), right (3,3), "
            "left (B,3,3) and output (B,N,3,3)");
    const int64_t total = euler_dims[0] * left_dims[0];
    if (total == 0) return ffi::Error::Success();
    const int blocks = static_cast<int>((total + kRelionEulerBlockSize - 1) / kRelionEulerBlockSize);
    const float* eulers_ptr = static_cast<const float*>(eulers_deg.untyped_data());
    const float* right_ptr = static_cast<const float*>(right_matrix.untyped_data());
    const float* left_ptr = static_cast<const float*>(left_matrices.untyped_data());
    float* output_ptr = static_cast<float*>(scorer_rotations->untyped_data());
    if (do_right)
        relion_make_scoring_rotations_left_f32_kernel<true><<<blocks, kRelionEulerBlockSize, 0, stream>>>(
            eulers_ptr, right_ptr, left_ptr, output_ptr, euler_dims[0], left_dims[0]);
    else
        relion_make_scoring_rotations_left_f32_kernel<false><<<blocks, kRelionEulerBlockSize, 0, stream>>>(
            eulers_ptr, right_ptr, left_ptr, output_ptr, euler_dims[0], left_dims[0]);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionMakeScoringRotationsF32Impl(
    cudaStream_t stream,
    int64_t do_right,
    ffi::AnyBuffer eulers_deg,
    ffi::AnyBuffer right_matrix,
    ffi::Result<ffi::AnyBuffer> scorer_rotations)
{
    if (eulers_deg.element_type() != ffi::DataType::F32 ||
        right_matrix.element_type() != ffi::DataType::F32 ||
        scorer_rotations->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: inputs/output must be F32");
    if (do_right != 0 && do_right != 1)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: do_right must be 0 or 1");

    auto euler_dims = eulers_deg.dimensions();
    auto right_dims = right_matrix.dimensions();
    auto output_dims = scorer_rotations->dimensions();
    if (euler_dims.size() != 2 || euler_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: eulers_deg must have shape (N,3)");
    if (right_dims.size() != 2 || right_dims[0] != 3 || right_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: right_matrix must have shape (3,3)");
    if (output_dims.size() != 3 || output_dims[0] != euler_dims[0] ||
        output_dims[1] != 3 || output_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: output must have shape (N,3,3)");

    const float* eulers_ptr = static_cast<const float*>(eulers_deg.untyped_data());
    const float* right_ptr = static_cast<const float*>(right_matrix.untyped_data());
    float* output_ptr = static_cast<float*>(scorer_rotations->untyped_data());
    cudaError_t err = do_right
        ? launch_relion_make_scoring_rotations<float, true>(
              stream, eulers_ptr, right_ptr, output_ptr, euler_dims[0])
        : launch_relion_make_scoring_rotations<float, false>(
              stream, eulers_ptr, right_ptr, output_ptr, euler_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionMakeScoringRotationsF32, RelionMakeScoringRotationsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("do_right")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionMakeScoringRotationsLeftF32, RelionMakeScoringRotationsLeftF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("do_right")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionMakeScoringRotationsF64Impl(
    cudaStream_t stream,
    int64_t do_right,
    ffi::AnyBuffer eulers_deg,
    ffi::AnyBuffer right_matrix,
    ffi::Result<ffi::AnyBuffer> scorer_rotations)
{
    if (eulers_deg.element_type() != ffi::DataType::F64 ||
        right_matrix.element_type() != ffi::DataType::F64 ||
        scorer_rotations->element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: inputs/output must be F64");
    if (do_right != 0 && do_right != 1)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: do_right must be 0 or 1");

    auto euler_dims = eulers_deg.dimensions();
    auto right_dims = right_matrix.dimensions();
    auto output_dims = scorer_rotations->dimensions();
    if (euler_dims.size() != 2 || euler_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: eulers_deg must have shape (N,3)");
    if (right_dims.size() != 2 || right_dims[0] != 3 || right_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: right_matrix must have shape (3,3)");
    if (output_dims.size() != 3 || output_dims[0] != euler_dims[0] ||
        output_dims[1] != 3 || output_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: output must have shape (N,3,3)");

    const double* eulers_ptr = static_cast<const double*>(eulers_deg.untyped_data());
    const double* right_ptr = static_cast<const double*>(right_matrix.untyped_data());
    double* output_ptr = static_cast<double*>(scorer_rotations->untyped_data());
    cudaError_t err = do_right
        ? launch_relion_make_scoring_rotations<double, true>(
              stream, eulers_ptr, right_ptr, output_ptr, euler_dims[0])
        : launch_relion_make_scoring_rotations<double, false>(
              stream, eulers_ptr, right_ptr, output_ptr, euler_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionMakeScoringRotationsF64, RelionMakeScoringRotationsF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("do_right")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionTranslateScoreF32Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> shifted)
{
    if (images.element_type() != ffi::DataType::C64 ||
        shifted->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: images/output must be C64");
    if (translation_angles.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: translation angles must be F32");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto output_dims = shifted->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: images must have shape (B,P)");
    const bool angles_per_image = translation_dims.size() == 3;
    if (!((translation_dims.size() == 2 && translation_dims[1] == 2) ||
          (angles_per_image && translation_dims[0] == image_dims[0] && translation_dims[2] == 2)))
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: translation angles must have shape (T,2) or (B,T,2)");
    const int64_t translation_count = translation_dims[angles_per_image ? 1 : 0];
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: pixel indices must have shape (P,)");
    if (output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] * translation_count ||
        output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: output must have shape (B*T,P)");

    cudaError_t err = launch_relion_translate_score_f32(
        stream,
        reinterpret_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<float2*>(shifted->untyped_data()),
        image_dims[0],
        translation_count,
        image_dims[1],
        static_cast<int>(image_h),
        static_cast<int>(image_half_width),
        angles_per_image);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateScoreF32, RelionTranslateScoreF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

#if CUB_VERSION < 300000
struct RelionAmpereScanPolicy
{
    using MaxPolicy = typename cub::DeviceScanPolicy<float, cub::Sum>::Policy800;
};

cudaError_t relion_ampere_inclusive_sum_f32(
    void* temporary,
    size_t& temporary_bytes,
    const float* input,
    float* output,
    int count,
    cudaStream_t stream)
{
    using Dispatch = cub::DispatchScan<
        const float*,
        float*,
        cub::Sum,
        cub::NullType,
        int,
        float,
        RelionAmpereScanPolicy>;
    return Dispatch::Dispatch(
        temporary,
        temporary_bytes,
        input,
        output,
        cub::Sum(),
        cub::NullType(),
        count,
        stream);
}
#else
cudaError_t relion_ampere_inclusive_sum_f32(
    void* temporary,
    size_t& temporary_bytes,
    const float* input,
    float* output,
    int count,
    cudaStream_t stream)
{
    // CCCL 3 replaced the policy-dispatch API used above.  Retain a buildable
    // fallback; deployed RELION parity is qualified against CUB 2.x, where
    // the explicit Ampere policy remains available and bitwise-tested.
    return cub::DeviceScan::InclusiveSum(
        temporary, temporary_bytes, input, output, count, stream);
}
#endif
ffi::Error RelionTranslateScoreF64Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> shifted)
{
    if (images.element_type() != ffi::DataType::C128 ||
        shifted->element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: images/output must be C128");
    if (translation_angles.element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: translation angles must be F64");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto output_dims = shifted->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: images must have shape (B,P)");
    if (translation_dims.size() != 2 || translation_dims[1] != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: translation angles must have shape (T,2)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: pixel indices must have shape (P,)");
    if (output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] * translation_dims[0] ||
        output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: output must have shape (B*T,P)");

    cudaError_t err = launch_relion_translate_score_f64(
        stream,
        reinterpret_cast<const double2*>(images.untyped_data()),
        static_cast<const double*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<double2*>(shifted->untyped_data()),
        image_dims[0],
        translation_dims[0],
        image_dims[1],
        static_cast<int>(image_h),
        static_cast<int>(image_half_width));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateScoreF64, RelionTranslateScoreF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

#include "relion_posterior.cuh"

ffi::Error RelionTranslateBprefF32Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer weighted_ctf,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> weighted_shifted)
{
    if (images.element_type() != ffi::DataType::C64 ||
        weighted_shifted->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: images/output must be C64");
    if (weighted_ctf.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: weighted CTF must be F32");
    if (translation_angles.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: translation angles must be F32");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto weighted_ctf_dims = weighted_ctf.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto output_dims = weighted_shifted->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: images must have shape (B,P)");
    if (weighted_ctf_dims.size() != 2 ||
        weighted_ctf_dims[0] != image_dims[0] ||
        weighted_ctf_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: weighted CTF must have shape (B,P)");
    if (translation_dims.size() != 2 || translation_dims[1] != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: translation angles must have shape (T,2)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: pixel indices must have shape (P,)");
    if (output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] * translation_dims[0] ||
        output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: output must have shape (B*T,P)");

    cudaError_t err = launch_relion_translate_bpref_f32(
        stream,
        reinterpret_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(weighted_ctf.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<float2*>(weighted_shifted->untyped_data()),
        image_dims[0],
        translation_dims[0],
        image_dims[1],
        static_cast<int>(image_h),
        static_cast<int>(image_half_width));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateBprefF32, RelionTranslateBprefF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionTranslateBprefF64Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer weighted_ctf,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> weighted_shifted)
{
    if (images.element_type() != ffi::DataType::C128 ||
        weighted_shifted->element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: images/output must be C128");
    if (weighted_ctf.element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: weighted CTF must be F64");
    if (translation_angles.element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: translation angles must be F64");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto weighted_ctf_dims = weighted_ctf.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto output_dims = weighted_shifted->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: images must have shape (B,P)");
    if (weighted_ctf_dims.size() != 2 ||
        weighted_ctf_dims[0] != image_dims[0] ||
        weighted_ctf_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: weighted CTF must have shape (B,P)");
    if (translation_dims.size() != 2 || translation_dims[1] != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: translation angles must have shape (T,2)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: pixel indices must have shape (P,)");
    if (output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] * translation_dims[0] ||
        output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: output must have shape (B*T,P)");

    cudaError_t err = launch_relion_translate_bpref_f64(
        stream,
        reinterpret_cast<const double2*>(images.untyped_data()),
        static_cast<const double*>(weighted_ctf.untyped_data()),
        static_cast<const double*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<double2*>(weighted_shifted->untyped_data()),
        image_dims[0], translation_dims[0], image_dims[1],
        static_cast<int>(image_h), static_cast<int>(image_half_width));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateBprefF64, RelionTranslateBprefF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);


// At most 256 nonempty buckets fit the queue's 256-particle capacity.
// One column's launch arguments stay below the pre-Volta 4 KiB parameter limit.
struct BprefPackColumn {
    const uint32_t* inputs[256];
    int32_t ends[256];
};
static_assert(sizeof(BprefPackColumn) + 64 < 4096);


ffi::Error RelionCoarseDiff2RectangularF32Common(
    cudaStream_t stream,
    const ffi::AnyBuffer* runtime_full_pixel_count,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32 ||
        (runtime_full_pixel_count != nullptr &&
         (runtime_full_pixel_count->element_type() != ffi::DataType::S32 ||
          runtime_full_pixel_count->dimensions().size() != 0)))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: lookup/count must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 2 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        lookup_dims.size() != 1 ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || image_dims[0] <= 0 ||
        image_dims[1] <= 0 ||
        image_dims[1] > kRelionCoarseDiff2BlockSize ||
        image_dims[2] != reference_dims[1] ||
        weight_dims[0] != image_dims[0] ||
        weight_dims[1] != reference_dims[1] || lookup_dims[0] <= 0 ||
        initial_dims[0] != image_dims[0] ||
        output_dims[0] != image_dims[0] ||
        output_dims[1] != reference_dims[0] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: inconsistent operand shapes");

    const int64_t rotation_blocks =
        (reference_dims[0] + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t block_count = image_dims[0] * rotation_blocks;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: block count exceeds CUDA grid");
    const cudaError_t err = launch_relion_coarse_diff2_rectangular_f32(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        image_dims[0],
        reference_dims[0],
        image_dims[1],
        reference_dims[1],
        lookup_dims[0],
        runtime_full_pixel_count == nullptr
            ? nullptr
            : static_cast<const int32_t*>(
                  runtime_full_pixel_count->untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionCoarseDiff2RectangularF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionCoarseDiff2RectangularF32Common(
        stream,
        nullptr,
        reference,
        shifted_image,
        weight,
        initial_diff2,
        full_to_compact,
        output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2RectangularF32, RelionCoarseDiff2RectangularF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error ValidateRelionCoarseDiff2ProjectorF32Operands(
    int64_t current_size,
    int64_t physical_image_size,
    int64_t model_max_r,
    int64_t canonical_reduction,
    int64_t single_lane_canonical,
    int64_t prehalf_weight,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projector_full.element_type() != ffi::DataType::C64 ||
        images.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: projector/images must be C64");
    if (rotations.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: rotations/angles/weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: lookup must be S32");

    const auto projector_dims = projector_full.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto image_dims = images.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (projector_dims.size() != 3 ||
        projector_dims[0] != projector_dims[1] ||
        projector_dims[1] != projector_dims[2] ||
        rotation_dims.size() != 2 || rotation_dims[1] != 6 ||
        image_dims.size() != 2 || angle_dims.size() != 2 ||
        angle_dims[1] != 2 || weight_dims.size() != 2 ||
        initial_dims.size() != 1 || lookup_dims.size() != 1 ||
        output_dims.size() != 3 || image_dims[0] <= 0 ||
        image_dims[1] <= 0 || rotation_dims[0] <= 0 ||
        angle_dims[0] <= 0 ||
        angle_dims[0] > kRelionCoarseDiff2BlockSize ||
        weight_dims[0] != image_dims[0] ||
        weight_dims[1] != image_dims[1] ||
        initial_dims[0] != image_dims[0] ||
        lookup_dims[0] != current_size * (current_size / 2 + 1) ||
        output_dims[0] != image_dims[0] ||
        output_dims[1] != rotation_dims[0] ||
        output_dims[2] != angle_dims[0] ||
        current_size <= 0 || physical_image_size <= 0 || model_max_r <= 0 ||
        (canonical_reduction != 0 && canonical_reduction != 1) ||
        (single_lane_canonical != 0 && single_lane_canonical != 1) ||
        (prehalf_weight != 0 && prehalf_weight != 1))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: inconsistent operand shapes or attributes");
    if (single_lane_canonical &&
        (canonical_reduction != 1 ||
         angle_dims[0] <= kRelionCoarseDiff2BlockSize / 2))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: single-lane canonical reduction "
            "requires canonical reduction and 65--128 translations");
    if (prehalf_weight && (canonical_reduction || single_lane_canonical))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: prehalved weights require "
            "native atomic reduction");

    return ffi::Error::Success();
}

ffi::Error RelionCoarseDiff2ProjectorF32Impl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    int64_t current_size,
    int64_t physical_image_size,
    int64_t model_max_r,
    int64_t padding_factor,
    int64_t canonical_reduction,
    int64_t single_lane_canonical,
    int64_t prehalf_weight,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    ffi::Error validation = ValidateRelionCoarseDiff2ProjectorF32Operands(
        current_size,
        physical_image_size,
        model_max_r,
        canonical_reduction,
        single_lane_canonical,
        prehalf_weight,
        projector_full,
        rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        output);
    if (validation.failure()) return validation;
    if (padding_factor <= 0)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2Projector: padding_factor must be positive");

    const auto projector_dims = projector_full.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto image_dims = images.dimensions();
    const auto angle_dims = translation_angles.dimensions();

    cudaError_t err;
    if (prehalf_weight) {
        err = launch_relion_coarse_diff2_projector_prehalf_f32(
            stream,
            scratch,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    } else if (single_lane_canonical) {
        err = launch_relion_coarse_diff2_projector_f32<false, true, true>(
            stream,
            scratch,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    } else if (canonical_reduction) {
        err = launch_relion_coarse_diff2_projector_f32<false, true>(
            stream,
            scratch,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    } else {
        err = launch_relion_coarse_diff2_projector_f32(
            stream,
            scratch,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2ProjectorF32, RelionCoarseDiff2ProjectorF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("physical_image_size")
        .Attr<int64_t>("model_max_r")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("canonical_reduction")
        .Attr<int64_t>("single_lane_canonical")
        .Attr<int64_t>("prehalf_weight")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

// The fused coarse projector with one pose set per image: image b scores its own rotations
// rotations[b] [R, 6] and translation phases translation_angles[b] [T, 2] (a subtomogram's tilt
// images, each with its own Aproj). One texture staging and one launch for every image of the
// batch; the per-image arithmetic and the canonical lane reduction are the shared kernel's.
ffi::Error RelionCoarseDiff2ProjectorPerImageF32Impl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    int64_t current_size,
    int64_t physical_image_size,
    int64_t model_max_r,
    int64_t padding_factor,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projector_full.element_type() != ffi::DataType::C64 ||
        images.element_type() != ffi::DataType::C64 ||
        rotations.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        full_to_compact.element_type() != ffi::DataType::S32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorPerImageF32: operand dtypes");
    const auto projector_dims = projector_full.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto image_dims = images.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (projector_dims.size() != 3 ||
        projector_dims[0] != projector_dims[1] ||
        projector_dims[1] != projector_dims[2] ||
        image_dims.size() != 2 || image_dims[0] <= 0 || image_dims[1] <= 0 ||
        rotation_dims.size() != 3 || rotation_dims[0] != image_dims[0] ||
        rotation_dims[1] <= 0 || rotation_dims[2] != 6 ||
        angle_dims.size() != 3 || angle_dims[0] != image_dims[0] ||
        angle_dims[1] <= 0 || angle_dims[1] > kRelionCoarseDiff2BlockSize ||
        angle_dims[2] != 2 ||
        weight_dims.size() != 2 || weight_dims[0] != image_dims[0] ||
        weight_dims[1] != image_dims[1] ||
        initial_dims.size() != 1 || initial_dims[0] != image_dims[0] ||
        lookup_dims.size() != 1 ||
        lookup_dims[0] != current_size * (current_size / 2 + 1) ||
        output_dims.size() != 3 || output_dims[0] != image_dims[0] ||
        output_dims[1] != rotation_dims[1] || output_dims[2] != angle_dims[1] ||
        current_size <= 0 || physical_image_size <= 0 || model_max_r <= 0 ||
        padding_factor <= 0)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorPerImageF32: inconsistent operand shapes or attributes");

    const cudaError_t err = launch_relion_coarse_diff2_projector_f32_impl<false, true, false, false, true>(
        stream,
        scratch,
        static_cast<const float2*>(projector_full.untyped_data()),
        static_cast<const float*>(rotations.untyped_data()),
        static_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        nullptr,
        image_dims[0],
        rotation_dims[1],
        angle_dims[1],
        image_dims[1],
        current_size,
        projector_dims[0],
        model_max_r,
        -static_cast<float>(physical_image_size * physical_image_size),
        image_dims[0],
        0,
        static_cast<int>(padding_factor));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2ProjectorPerImageF32, RelionCoarseDiff2ProjectorPerImageF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("physical_image_size")
        .Attr<int64_t>("model_max_r")
        .Attr<int64_t>("padding_factor")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseNormalizedCcPairsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer score_weight,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer half_weights,
    ffi::AnyBuffer packed_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (shifted_image.element_type() != ffi::DataType::C64 ||
        reference.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: image/reference must be C64");
    if (score_weight.element_type() != ffi::DataType::F32 ||
        half_weights.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: weights/output must be F32");
    if (packed_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: lookup must be S32");

    const auto image_dims = shifted_image.dimensions();
    const auto score_weight_dims = score_weight.dimensions();
    const auto reference_dims = reference.dimensions();
    const auto half_weight_dims = half_weights.dimensions();
    const auto lookup_dims = packed_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (image_dims.size() != 3 || image_dims[0] <= 0 ||
        image_dims[1] <= 0 || image_dims[2] <= 0 ||
        score_weight_dims.size() != 3 || reference_dims.size() != 3 ||
        score_weight_dims[0] != image_dims[0] ||
        score_weight_dims[1] != image_dims[1] ||
        score_weight_dims[2] != image_dims[2] ||
        reference_dims[0] != image_dims[0] ||
        reference_dims[1] != image_dims[1] ||
        reference_dims[2] != image_dims[2] ||
        half_weight_dims.size() != 1 ||
        half_weight_dims[0] != image_dims[2] || lookup_dims.size() != 1 ||
        lookup_dims[0] <= 0 || output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] || output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: inconsistent operand shapes");

    const int64_t candidate_count = image_dims[0] * image_dims[1];
    if (candidate_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: candidate count exceeds CUDA grid");
    cudaError_t err = launch_relion_coarse_normalized_cc_pairs_f32(
        stream,
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(score_weight.untyped_data()),
        reinterpret_cast<const float2*>(reference.untyped_data()),
        static_cast<const float*>(half_weights.untyped_data()),
        static_cast<const int32_t*>(packed_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        candidate_count,
        image_dims[2],
        lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseNormalizedCcPairsF32,
    RelionCoarseNormalizedCcPairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseNormalizedCcNativeTexturePairsF32Impl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer eulers,
    ffi::AnyBuffer unshifted_image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer score_weight,
    ffi::AnyBuffer numerator_weight,
    ffi::AnyBuffer half_weights,
    ffi::AnyBuffer packed_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projector_full.element_type() != ffi::DataType::C64 ||
        unshifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: projector/image must be C64");
    if (eulers.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        score_weight.element_type() != ffi::DataType::F32 ||
        numerator_weight.element_type() != ffi::DataType::F32 ||
        half_weights.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: eulers/weights/output must be F32");
    if (packed_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: lookup must be S32");

    const auto projector_dims = projector_full.dimensions();
    const auto euler_dims = eulers.dimensions();
    const auto image_dims = unshifted_image.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto score_weight_dims = score_weight.dimensions();
    const auto numerator_weight_dims = numerator_weight.dimensions();
    const auto half_weight_dims = half_weights.dimensions();
    const auto lookup_dims = packed_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const int64_t expected_packed_pixels =
        current_size * (current_size / 2 + 1);
    if (current_size <= 0 || padding_factor <= 0 || projector_max_r <= 0 ||
        projector_dims.size() != 3 || projector_dims[0] <= 0 ||
        projector_dims[1] != projector_dims[0] ||
        projector_dims[2] != projector_dims[0] ||
        euler_dims.size() != 2 || euler_dims[0] <= 0 ||
        euler_dims[1] != 9 || image_dims.size() != 2 ||
        image_dims[0] != euler_dims[0] || image_dims[1] <= 0 ||
        angle_dims.size() != 2 || angle_dims[0] != image_dims[0] ||
        angle_dims[1] != 2 ||
        score_weight_dims.size() != 2 ||
        score_weight_dims[0] != image_dims[0] ||
        score_weight_dims[1] != image_dims[1] ||
        numerator_weight_dims.size() != 2 ||
        numerator_weight_dims[0] != image_dims[0] ||
        numerator_weight_dims[1] != image_dims[1] ||
        half_weight_dims.size() != 1 ||
        half_weight_dims[0] != image_dims[1] ||
        lookup_dims.size() != 1 ||
        lookup_dims[0] != expected_packed_pixels ||
        output_dims.size() != 2 || output_dims[0] != image_dims[0] ||
        output_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: inconsistent operand shapes");
    if (image_dims[0] > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: candidate count exceeds CUDA grid");

    cudaError_t err =
        launch_relion_coarse_normalized_cc_native_texture_pairs_f32(
            stream,
            scratch,
            reinterpret_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(eulers.untyped_data()),
            reinterpret_cast<const float2*>(unshifted_image.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(score_weight.untyped_data()),
            static_cast<const float*>(numerator_weight.untyped_data()),
            static_cast<const float*>(half_weights.untyped_data()),
            static_cast<const int32_t*>(packed_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            projector_dims[0],
            image_dims[0],
            image_dims[1],
            lookup_dims[0],
            static_cast<int>(current_size),
            static_cast<int>(padding_factor),
            static_cast<int>(projector_max_r));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseNormalizedCcNativeTexturePairsF32,
    RelionCoarseNormalizedCcNativeTexturePairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("projector_max_r")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2RectangularF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        lookup_dims.size() != 1 ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] <= 0 || image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        initial_dims[0] != reference_dims[0] ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: inconsistent operand shapes");

    const int64_t total_hypotheses =
        reference_dims[0] * reference_dims[1] * image_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_rectangular<float, float2, true>(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        reference_dims[0],
        reference_dims[1],
        image_dims[1],
        reference_dims[2],
        lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2RectangularF32, RelionFineDiff2RectangularF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2RectangularMaskedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer candidate_mask,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: lookup must be S32");
    if (candidate_mask.element_type() != ffi::DataType::PRED &&
        candidate_mask.element_type() != ffi::DataType::U8 &&
        candidate_mask.element_type() != ffi::DataType::S8)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: candidate_mask must be PRED/U8/S8");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto mask_dims = candidate_mask.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        lookup_dims.size() != 1 || mask_dims.size() != 3 ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] <= 0 || image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        initial_dims[0] != reference_dims[0] ||
        mask_dims[0] != reference_dims[0] ||
        mask_dims[1] != reference_dims[1] ||
        mask_dims[2] != image_dims[1] ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: inconsistent operand shapes");

    const int64_t total_hypotheses =
        reference_dims[0] * reference_dims[1] * image_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_rectangular_masked<float, float2>(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<const uint8_t*>(candidate_mask.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        reference_dims[0],
        reference_dims[1],
        image_dims[1],
        reference_dims[2],
        lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2RectangularMaskedF32, RelionFineDiff2RectangularMaskedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);


ffi::Error RelionFineDiff2FusedTranslateFlatRowsF32Common(
    cudaStream_t stream,
    ffi::ScratchAllocator& scratch,
    int64_t current_size,
    const ffi::AnyBuffer* runtime_current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output,
    const ffi::AnyBuffer* translation_chunk_live = nullptr)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslate{,Runtime,RuntimeMasked}FlatRowsF32: reference/image must be C64");
    if (row_image_ids.element_type() != ffi::DataType::S32 ||
        full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslate{,Runtime,RuntimeMasked}FlatRowsF32: row ids/lookup must be S32");
    if (translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslate{,Runtime,RuntimeMasked}FlatRowsF32: "
            "angles/weight/initial/output must be F32");
    if (runtime_current_size != nullptr &&
        (runtime_current_size->element_type() != ffi::DataType::S32 ||
         runtime_current_size->dimensions().size() != 0))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRuntime{,Masked}FlatRowsF32: "
            "logical_current_size must be an S32 scalar");
    if (runtime_current_size == nullptr &&
        (current_size <= 0 || current_size > std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslate{,Runtime,RuntimeMasked}FlatRowsF32: invalid current_size");

    const auto reference_dims = reference.dimensions();
    const auto row_image_dims = row_image_ids.dimensions();
    const auto image_dims = image.dimensions();
    const auto translation_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const bool lookup_shape_valid = lookup_dims.size() == 1 && lookup_dims[0] > 0;
    const int64_t expected_full_pixels = runtime_current_size == nullptr
        ? current_size * (current_size / 2 + 1)
        : (lookup_shape_valid ? lookup_dims[0] : 0);
    // Angles are one shared [T, 2] table, or one [T, 2] table per image, [B, T, 2] (tilt images).
    const bool per_image_angles = translation_dims.size() == 3;
    const int64_t translation_count = per_image_angles
        ? translation_dims[1]
        : (translation_dims.size() == 2 ? translation_dims[0] : 0);
    const bool angles_shape_valid = per_image_angles
        ? translation_dims[0] == image_dims[0] && translation_dims[2] == 2
        : translation_dims.size() == 2 && translation_dims[1] == 2;
    if (reference_dims.size() != 2 || row_image_dims.size() != 1 ||
        image_dims.size() != 2 || !angles_shape_valid ||
        weight_dims.size() != 2 ||
        initial_dims.size() != 1 || !lookup_shape_valid ||
        output_dims.size() != 2 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || row_image_dims[0] != reference_dims[0] ||
        image_dims[0] <= 0 || image_dims[1] != reference_dims[1] ||
        translation_count <= 0 || weight_dims[0] != image_dims[0] ||
        weight_dims[1] != reference_dims[1] ||
        initial_dims[0] != image_dims[0] ||
        lookup_dims[0] != expected_full_pixels ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != translation_count)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslate{,Runtime,RuntimeMasked}FlatRowsF32: inconsistent operand shapes");

    const int64_t translation_chunks =
        (translation_count + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    const int64_t total_blocks = reference_dims[0] * translation_chunks;
    if (total_blocks > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslate{,Runtime,RuntimeMasked}FlatRowsF32: block count exceeds CUDA grid");
    if (translation_chunk_live != nullptr) {
        const auto live_dims = translation_chunk_live->dimensions();
        if (translation_chunk_live->element_type() != ffi::DataType::U8 ||
            live_dims.size() != 2 || live_dims[0] != reference_dims[0] ||
            live_dims[1] != translation_chunks)
            return ffi::Error::InvalidArgument(
                "RelionFineDiff2FusedTranslateRuntimeMaskedFlatRowsF32: translation_chunk_live "
                "must be U8 [rows, ceil(translations / 4)]");
    }
    cudaError_t err = launch_relion_fine_diff2_fused_translate_flat_rows_f32(
        stream,
        scratch,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(image.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(row_image_ids.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        image_dims[0],
        reference_dims[0],
        translation_count,
        reference_dims[1],
        lookup_dims[0],
        static_cast<int>(current_size),
        runtime_current_size == nullptr
            ? nullptr
            : static_cast<const int32_t*>(runtime_current_size->untyped_data()),
        translation_chunk_live == nullptr
            ? nullptr
            : static_cast<const uint8_t*>(translation_chunk_live->untyped_data()),
        per_image_angles ? 2 * translation_count : 0);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}


ffi::Error RelionFineDiff2FusedTranslateRuntimeFlatRowsF32Impl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_current_size,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslateFlatRowsF32Common(
        stream, scratch, 0, &logical_current_size, reference, row_image_ids, image,
        translation_angles, weight, initial_diff2, full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateRuntimeFlatRowsF32,
    RelionFineDiff2FusedTranslateRuntimeFlatRowsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

// The runtime flat-row scorer that traverses only translation chunks holding a
// candidate cell (translation_chunk_live, U8 [rows, ceil(translations / 4)]).
ffi::Error RelionFineDiff2FusedTranslateRuntimeMaskedFlatRowsF32Impl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_current_size,
    ffi::AnyBuffer translation_chunk_live,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslateFlatRowsF32Common(
        stream, scratch, 0, &logical_current_size, reference, row_image_ids, image,
        translation_angles, weight, initial_diff2, full_to_compact, output,
        &translation_chunk_live);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateRuntimeMaskedFlatRowsF32,
    RelionFineDiff2FusedTranslateRuntimeMaskedFlatRowsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);


ffi::Error RelionFineDiff2PairsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: weight/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || lookup_dims.size() != 1 ||
        output_dims.size() != 2 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] != reference_dims[1] ||
        image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: inconsistent operand shapes");

    const int64_t total_hypotheses = reference_dims[0] * reference_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_pairs<float, float2>(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        reference_dims[0],
        reference_dims[1],
        reference_dims[2],
        lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2PairsF32, RelionFineDiff2PairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2RectangularF64Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C128 ||
        shifted_image.element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: reference/image must be C128");
    if (weight.element_type() != ffi::DataType::F64 ||
        output->element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: weight/output must be F64");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || lookup_dims.size() != 1 ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] <= 0 || image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: inconsistent operand shapes");

    const int64_t total_hypotheses =
        reference_dims[0] * reference_dims[1] * image_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_rectangular<double, double2, false>(
        stream,
        reinterpret_cast<const double2*>(reference.untyped_data()),
        reinterpret_cast<const double2*>(shifted_image.untyped_data()),
        static_cast<const double*>(weight.untyped_data()),
        nullptr,
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<double*>(output->untyped_data()),
        reference_dims[0], reference_dims[1], image_dims[1],
        reference_dims[2], lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2RectangularF64, RelionFineDiff2RectangularF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2PairsF64Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C128 ||
        shifted_image.element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: reference/image must be C128");
    if (weight.element_type() != ffi::DataType::F64 ||
        output->element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: weight/output must be F64");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || lookup_dims.size() != 1 ||
        output_dims.size() != 2 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] != reference_dims[1] ||
        image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: inconsistent operand shapes");

    const int64_t total_hypotheses = reference_dims[0] * reference_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_pairs<double, double2>(
        stream,
        reinterpret_cast<const double2*>(reference.untyped_data()),
        reinterpret_cast<const double2*>(shifted_image.untyped_data()),
        static_cast<const double*>(weight.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<double*>(output->untyped_data()),
        reference_dims[0], reference_dims[1], reference_dims[2], lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2PairsF64, RelionFineDiff2PairsF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionPreprocessRealF32ImplWithReduction(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    int64_t host_check,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    ffi::Result<ffi::AnyBuffer> workspace_out,
    ffi::Result<ffi::AnyBuffer> invalid_count_out,
    int reduction_mode)
{
    if (invalid_count_out->element_type() != ffi::DataType::S32 ||
        invalid_count_out->dimensions().size() != 1 || invalid_count_out->dimensions()[0] != 2)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: invalid_count must be S32 with shape (2,)");
    if (host_check != 0 && host_check != 1)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: host_check must be 0 or 1");
    if (images.element_type() != ffi::DataType::F32 ||
        normalization_factors.element_type() != ffi::DataType::F32 ||
        normalized_shifted_out->element_type() != ffi::DataType::F32 ||
        masked_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: images/factors/outputs must be F32");
    if (integer_shifts.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: shifts must be S32");
    auto image_dims = images.dimensions();
    auto factor_dims = normalization_factors.dimensions();
    auto shift_dims = integer_shifts.dimensions();
    auto normshift_dims = normalized_shifted_out->dimensions();
    auto masked_dims = masked_out->dimensions();
    if (image_dims.size() != 3 || image_dims[0] <= 0 || image_dims[1] <= 0 || image_dims[1] != image_dims[2])
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: images must have shape (batch,D,D)");
    if (factor_dims.size() != 1 || factor_dims[0] != image_dims[0])
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: factors must have shape (batch,)");
    if (shift_dims.size() != 2 || shift_dims[0] != image_dims[0] || shift_dims[1] != 2)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: shifts must have shape (batch,2)");
    if (normshift_dims.size() != 3 || masked_dims.size() != 3 ||
        normshift_dims[0] != image_dims[0] || normshift_dims[1] != image_dims[1] ||
        normshift_dims[2] != image_dims[2] || masked_dims[0] != image_dims[0] ||
        masked_dims[1] != image_dims[1] || masked_dims[2] != image_dims[2])
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: output shapes must match images");
    if (!(radius > 0.0f) || !(cosine_width > 0.0f) ||
        !std::isfinite(radius) || !std::isfinite(cosine_width))
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: radius/width must be finite and positive");
    if (apply_mask != 0 && apply_mask != 1)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: apply_mask must be 0 or 1");

    RelionPreprocessScratchLayout layout;
    cudaError_t err = relion_preprocess_scratch_layout(image_dims[0], reduction_mode, &layout);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    // XLA provisions the soft-mask workspace as an output buffer sized by the
    // Python wrapper from the same layout constants; fail closed if it is
    // smaller than this build's layout (for example a larger CUB temporary).
    if (workspace_out->element_type() != ffi::DataType::U8 || workspace_out->dimensions().size() != 1)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: workspace must be U8 with shape (bytes,)");
    size_t workspace_bytes = static_cast<size_t>(workspace_out->dimensions()[0]);
    if (apply_mask != 0 && workspace_bytes < layout.total_bytes)
        return ffi::Error::InvalidArgument(
            "RelionPreprocessRealF32: workspace too small: need " + std::to_string(layout.total_bytes) +
            " bytes, got " + std::to_string(workspace_bytes));
    void* scratch_ptr = apply_mask != 0 ? workspace_out->untyped_data() : nullptr;
    err = launch_relion_preprocess_real_f32(
        stream,
        static_cast<const float*>(images.untyped_data()),
        static_cast<const float*>(normalization_factors.untyped_data()),
        static_cast<const int32_t*>(integer_shifts.untyped_data()),
        static_cast<float*>(normalized_shifted_out->untyped_data()),
        static_cast<float*>(masked_out->untyped_data()),
        image_dims[0], static_cast<int>(image_dims[1]), static_cast<int>(image_dims[2]),
        radius, cosine_width, apply_mask != 0, reduction_mode, host_check != 0,
        static_cast<int32_t*>(invalid_count_out->untyped_data()), scratch_ptr, layout);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionPreprocessRealF32Impl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    int64_t host_check,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    ffi::Result<ffi::AnyBuffer> workspace_out,
    ffi::Result<ffi::AnyBuffer> invalid_count_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, host_check, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, workspace_out, invalid_count_out, 0);
}

ffi::Error RelionPreprocessRealF32NativeLaneImpl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    int64_t host_check,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    ffi::Result<ffi::AnyBuffer> workspace_out,
    ffi::Result<ffi::AnyBuffer> invalid_count_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, host_check, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, workspace_out, invalid_count_out, 1);
}

ffi::Error RelionPreprocessRealF32NativeAtomicImpl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    int64_t host_check,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    ffi::Result<ffi::AnyBuffer> workspace_out,
    ffi::Result<ffi::AnyBuffer> invalid_count_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, host_check, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, workspace_out, invalid_count_out, 2);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPreprocessRealF32, RelionPreprocessRealF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("radius")
        .Attr<float>("cosine_width")
        .Attr<int64_t>("apply_mask")
        .Attr<int64_t>("host_check")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPreprocessRealF32NativeLane, RelionPreprocessRealF32NativeLaneImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("radius")
        .Attr<float>("cosine_width")
        .Attr<int64_t>("apply_mask")
        .Attr<int64_t>("host_check")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPreprocessRealF32NativeAtomic, RelionPreprocessRealF32NativeAtomicImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("radius")
        .Attr<float>("cosine_width")
        .Attr<int64_t>("apply_mask")
        .Attr<int64_t>("host_check")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFusedXHalfBackprojectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    ffi::AnyBuffer data_rows,
    ffi::AnyBuffer weight_rows,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer data_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::Result<ffi::AnyBuffer> data_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out)
{
    const bool use_f64 = data_rows.element_type() == ffi::DataType::C128;
    const ffi::DataType complex_type = use_f64 ? ffi::DataType::C128 : ffi::DataType::C64;
    const ffi::DataType real_type = use_f64 ? ffi::DataType::F64 : ffi::DataType::F32;
    if (data_rows.element_type() != complex_type ||
        data_volume_in.element_type() != complex_type ||
        data_volume_out->element_type() != complex_type)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: data rows and volumes must share complex64 or complex128 dtype");
    if (weight_rows.element_type() != real_type ||
        weight_volume_in.element_type() != real_type ||
        weight_volume_out->element_type() != real_type)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: weight dtype must match the complex precision");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: pixel indices must be int32");
    if (rot.element_type() != real_type)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: rotation dtype must match the accumulator precision");
    if (order != 1 || half_volume != 1 || half_image != 1)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: requires order=1 and half image/volume");
    if (N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: volume must be positive, cubic, and odd-sized");
    if (image_h <= 0 || image_w != image_h / 2 + 1 || full_image_w != image_h)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: image attrs must describe a native FFTW half square");
    if (upsampling <= 0 || max_r2_x4 < 0)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: requires positive upsampling and an explicit radius");

    const auto data_row_dims = data_rows.dimensions();
    const auto weight_row_dims = weight_rows.dimensions();
    const auto pixel_dims = pixel_indices.dimensions();
    const auto rot_dims = rot.dimensions();
    const auto data_in_dims = data_volume_in.dimensions();
    const auto weight_in_dims = weight_volume_in.dimensions();
    const auto data_out_dims = data_volume_out->dimensions();
    const auto weight_out_dims = weight_volume_out->dimensions();
    if (data_row_dims.size() != 2 || weight_row_dims.size() != 2 ||
        data_row_dims[0] <= 0 || data_row_dims[1] <= 0 ||
        weight_row_dims[0] != data_row_dims[0] ||
        weight_row_dims[1] != data_row_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: data/weight rows must have matching nonempty rank-2 shapes");
    if (pixel_dims.size() != 1 || pixel_dims[0] != data_row_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: pixel index length must match row width");
    if (rot_dims.size() != 2 || rot_dims[0] != data_row_dims[0] || rot_dims[1] != 6)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: rotations must have shape (n_rows, 6)");
    if (data_row_dims[1] != image_h * image_w)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: rows must contain the full native FFTW square");

    const int64_t expected_volume_size = N0 * N1 * (N2 / 2 + 1);
    if (data_in_dims.size() != 1 || weight_in_dims.size() != 1 ||
        data_out_dims.size() != 1 || weight_out_dims.size() != 1 ||
        data_in_dims[0] != expected_volume_size ||
        weight_in_dims[0] != expected_volume_size ||
        data_out_dims[0] != expected_volume_size ||
        weight_out_dims[0] != expected_volume_size)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: accumulator sizes do not match the half volume");

    cudaError_t err;
    if (use_f64) {
        err = launch_relion_fused_x_half_backproject_f64(
            stream,
            reinterpret_cast<double2*>(data_volume_out->untyped_data()),
            static_cast<double*>(weight_volume_out->untyped_data()),
            reinterpret_cast<const double2*>(data_rows.untyped_data()),
            static_cast<const double*>(weight_rows.untyped_data()),
            static_cast<const int32_t*>(pixel_indices.untyped_data()),
            static_cast<const double*>(rot.untyped_data()),
            data_row_dims[0], data_row_dims[1], image_h, image_w,
            N0, N1, N2, upsampling, max_r2_x4);
    } else {
        err = launch_relion_fused_x_half_backproject(
            stream,
            reinterpret_cast<float2*>(data_volume_out->untyped_data()),
            static_cast<float*>(weight_volume_out->untyped_data()),
            reinterpret_cast<const float2*>(data_rows.untyped_data()),
            static_cast<const float*>(weight_rows.untyped_data()),
            static_cast<const int32_t*>(pixel_indices.untyped_data()),
            static_cast<const float*>(rot.untyped_data()),
            data_row_dims[0], data_row_dims[1], image_h, image_w,
            N0, N1, N2, upsampling, max_r2_x4);
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionFusedXHalfBackprojectParticleGridImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4, int64_t n_particles, int64_t rows_per_particle,
    ffi::AnyBuffer data_rows,
    ffi::AnyBuffer weight_rows,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer data_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::Result<ffi::AnyBuffer> data_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out)
{
    if (data_rows.element_type() != ffi::DataType::C64 ||
        data_volume_in.element_type() != ffi::DataType::C64 ||
        data_volume_out->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: data rows and volumes must be complex64");
    if (weight_rows.element_type() != ffi::DataType::F32 ||
        weight_volume_in.element_type() != ffi::DataType::F32 ||
        weight_volume_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: weight rows and volumes must be float32");
    if (pixel_indices.element_type() != ffi::DataType::S32 ||
        rot.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: indices/rotations have invalid dtypes");
    if (order != 1 || half_volume != 1 || half_image != 1 ||
        N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0 ||
        image_h <= 0 || image_w != image_h / 2 + 1 || full_image_w != image_h ||
        upsampling <= 0 || max_r2_x4 < 0 ||
        n_particles <= 0 || rows_per_particle <= 0)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: invalid strict x-half attributes");

    const auto data_dims = data_rows.dimensions();
    const auto weight_dims = weight_rows.dimensions();
    const auto pixel_dims = pixel_indices.dimensions();
    const auto rot_dims = rot.dimensions();
    const int64_t n_rows = n_particles * rows_per_particle;
    if (data_dims.size() != 2 || data_dims[0] != n_rows || data_dims[1] <= 0 ||
        weight_dims.size() != 2 || weight_dims[0] != n_rows ||
        weight_dims[1] != data_dims[1] ||
        pixel_dims.size() != 1 || pixel_dims[0] != data_dims[1] ||
        rot_dims.size() != 2 || rot_dims[0] != n_rows || rot_dims[1] != 6 ||
        data_dims[1] != image_h * image_w)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: inconsistent row topology");

    const int64_t volume_size = N0 * N1 * (N2 / 2 + 1);
    const auto data_in_dims = data_volume_in.dimensions();
    const auto weight_in_dims = weight_volume_in.dimensions();
    const auto data_out_dims = data_volume_out->dimensions();
    const auto weight_out_dims = weight_volume_out->dimensions();
    if (data_in_dims.size() != 1 || data_in_dims[0] != volume_size ||
        weight_in_dims.size() != 1 || weight_in_dims[0] != volume_size ||
        data_out_dims.size() != 1 || data_out_dims[0] != volume_size ||
        weight_out_dims.size() != 1 || weight_out_dims[0] != volume_size)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: accumulator sizes do not match");

    auto* data_out = reinterpret_cast<float2*>(data_volume_out->untyped_data());
    auto* weight_out = static_cast<float*>(weight_volume_out->untyped_data());
    const auto* data = reinterpret_cast<const float2*>(data_rows.untyped_data());
    const auto* weight = static_cast<const float*>(weight_rows.untyped_data());
    const auto* indices = static_cast<const int32_t*>(pixel_indices.untyped_data());
    const auto* rotations = static_cast<const float*>(rot.untyped_data());
    const int64_t n_pixels = data_dims[1];
    const int64_t row_stride = rows_per_particle * n_pixels;
    const int64_t rotation_stride = rows_per_particle * 6;
    for (int64_t particle = 0; particle < n_particles; ++particle) {
        cudaError_t err = launch_relion_fused_x_half_backproject(
            stream, data_out, weight_out,
            data + particle * row_stride,
            weight + particle * row_stride,
            indices,
            rotations + particle * rotation_stride,
            rows_per_particle, n_pixels, image_h, image_w,
            N0, N1, N2, upsampling, max_r2_x4);
        if (err != cudaSuccess)
            return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFusedXHalfBackproject, RelionFusedXHalfBackprojectImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("order")
        .Attr<int64_t>("half_volume")
        .Attr<int64_t>("half_image")
        .Attr<int64_t>("full_image_w")
        .Attr<int64_t>("max_r2_x4")
        .Arg<ffi::AnyBuffer>()           /* data_rows       */
        .Arg<ffi::AnyBuffer>()           /* weight_rows     */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices   */
        .Arg<ffi::AnyBuffer>()           /* rot             */
        .Arg<ffi::AnyBuffer>()           /* data_volume_in  */
        .Arg<ffi::AnyBuffer>()           /* weight_volume_in */
        .Ret<ffi::AnyBuffer>()           /* data_volume_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* weight_volume_out (aliased) */
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFusedXHalfBackprojectParticleGrid,
    RelionFusedXHalfBackprojectParticleGridImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("order")
        .Attr<int64_t>("half_volume")
        .Attr<int64_t>("half_image")
        .Attr<int64_t>("full_image_w")
        .Attr<int64_t>("max_r2_x4")
        .Attr<int64_t>("n_particles")
        .Attr<int64_t>("rows_per_particle")
        .Arg<ffi::AnyBuffer>()           /* data_rows */
        .Arg<ffi::AnyBuffer>()           /* weight_rows */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices */
        .Arg<ffi::AnyBuffer>()           /* rot */
        .Arg<ffi::AnyBuffer>()           /* data_volume_in */
        .Arg<ffi::AnyBuffer>()           /* weight_volume_in */
        .Ret<ffi::AnyBuffer>()           /* data_volume_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* weight_volume_out (aliased) */
);


struct RelionHalfTextureGeometry {
    int64_t padded_max_r;
    int64_t projector_size;
    int64_t projector_half_x;
    int64_t image_pixels;
};

bool validate_relion_half_texture_geometry(
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    RelionHalfTextureGeometry* geometry)
{
    const int64_t int_max =
        static_cast<int64_t>(std::numeric_limits<int>::max());
    /* Every launcher stores the squared padded radius in an int and forms
     * texture y/z as 2 * padded_max_r + 3.  Validate against the tighter
     * radius-square limit before evaluating either product. */
    constexpr int64_t max_padded_radius_with_int_square = 46340;
    if (current_size <= 0 || current_size > int_max ||
        padding_factor <= 0 || padding_factor > int_max ||
        projector_max_r <= 0 || projector_max_r > int_max ||
        projector_max_r >
            max_padded_radius_with_int_square / padding_factor)
        return false;

    const int64_t padded_max_r = projector_max_r * padding_factor;
    if (padded_max_r > (int_max - 3) / 2) return false;
    const int64_t image_half_x = current_size / 2 + 1;
    if (current_size > int_max / image_half_x) return false;

    geometry->padded_max_r = padded_max_r;
    geometry->projector_size = 2 * padded_max_r + 3;
    geometry->projector_half_x = padded_max_r + 2;
    geometry->image_pixels = current_size * image_half_x;
    return true;
}

ffi::Error RelionProjectorHalfTextureF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    float projector_scale,
    ffi::AnyBuffer projector_half,
    ffi::AnyBuffer rotations,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projector_half.element_type() != ffi::DataType::C64 ||
        output->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionProjectorHalfTextureF32: projector/output must be C64");
    if (rotations.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionProjectorHalfTextureF32: rotations must be F32");

    const auto projector_dims = projector_half.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto output_dims = output->dimensions();
    RelionHalfTextureGeometry geometry;
    const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (!validate_relion_half_texture_geometry(
            current_size, padding_factor, projector_max_r, &geometry) ||
        !std::isfinite(projector_scale) ||
        projector_dims.size() != 3 ||
        projector_dims[0] != geometry.projector_size ||
        projector_dims[1] != geometry.projector_size ||
        projector_dims[2] != geometry.projector_half_x ||
        rotation_dims.size() != 2 || rotation_dims[0] <= 0 ||
        rotation_dims[0] > int_max || rotation_dims[1] != 6 ||
        output_dims.size() != 2 || output_dims[0] != rotation_dims[0] ||
        output_dims[1] != geometry.image_pixels)
        return ffi::Error::InvalidArgument(
            "RelionProjectorHalfTextureF32: inconsistent operand shapes");

    cudaError_t err = launch_relion_projector_half_texture_f32(
        stream,
        reinterpret_cast<const float2*>(projector_half.untyped_data()),
        reinterpret_cast<float2*>(output->untyped_data()),
        static_cast<const float*>(rotations.untyped_data()),
        rotation_dims[0],
        current_size,
        current_size / 2 + 1,
        static_cast<int>(padding_factor),
        static_cast<int>(projector_max_r),
        projector_scale);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionProjectorHalfTextureF32,
    RelionProjectorHalfTextureF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("projector_max_r")
        .Attr<float>("projector_scale")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionProjectorPersistentHalfTextureF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    ffi::AnyBuffer owner_handle_buffer,
    ffi::AnyBuffer rotations,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (owner_handle_buffer.element_type() != ffi::DataType::U64)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: owner handle must be U64");
    if (rotations.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: rotations must be F32");
    if (output->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: output must be C64");

    const auto handle_dims = owner_handle_buffer.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto output_dims = output->dimensions();
    RelionHalfTextureGeometry geometry;
    const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (handle_dims.size() != 0 ||
        !validate_relion_half_texture_geometry(
            current_size, padding_factor, projector_max_r, &geometry) ||
        rotation_dims.size() != 2 || rotation_dims[0] <= 0 ||
        rotation_dims[0] > int_max || rotation_dims[1] != 6 ||
        output_dims.size() != 2 || output_dims[0] != rotation_dims[0] ||
        output_dims[1] != geometry.image_pixels)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: inconsistent operands");

    uint64_t owner_handle = 0;
    cudaError_t err = cudaMemcpyAsync(
        &owner_handle,
        owner_handle_buffer.untyped_data(),
        sizeof(owner_handle),
        cudaMemcpyDeviceToHost,
        stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));

    std::shared_ptr<PersistentRelionHalfTextureF32> owner;
    try {
        std::lock_guard<std::mutex> lock(
            persistent_relion_half_texture_f32_mutex);
        const auto found =
            persistent_relion_half_texture_f32_registry.find(owner_handle);
        if (found == persistent_relion_half_texture_f32_registry.end())
            return ffi::Error::InvalidArgument(
                "RelionProjectorPersistentHalfTextureF32: owner handle is not live");
        owner = found->second;
    } catch (...) {
        return ffi::Error::Internal(
            "RelionProjectorPersistentHalfTextureF32: registry lookup failed");
    }
    PersistentRelionHalfTextureF32CallGuard call_guard(owner);
    if (!call_guard.active)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: owner is closing");
    int current_device = -1;
    err = cudaGetDevice(&current_device);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    if (owner->tex_x != geometry.projector_half_x ||
        owner->tex_y != geometry.projector_size ||
        owner->tex_z != geometry.projector_size ||
        owner->device != current_device)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: texture geometry/device mismatch");

    err = launch_relion_projector_persistent_half_texture_f32(
        stream,
        *owner,
        reinterpret_cast<float2*>(output->untyped_data()),
        static_cast<const float*>(rotations.untyped_data()),
        rotation_dims[0],
        current_size,
        current_size / 2 + 1,
        static_cast<int>(padding_factor),
        static_cast<int>(projector_max_r));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionProjectorPersistentHalfTextureF32,
    RelionProjectorPersistentHalfTextureF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("projector_max_r")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error ProjectRelionHalfRuntimeCore(
    cudaStream_t stream, int64_t image_h, int64_t image_w,
    int64_t padding_factor,
    ffi::AnyBuffer half, ffi::AnyBuffer rot, ffi::AnyBuffer radius,
    ffi::Result<ffi::AnyBuffer> output, const int32_t* image_radius)
{
    const auto h = half.dimensions();
    const auto r = rot.dimensions();
    const auto o = output->dimensions();
    if (half.element_type() != ffi::DataType::C64 ||
        rot.element_type() != ffi::DataType::F32 ||
        radius.element_type() != ffi::DataType::S32 ||
        output->element_type() != ffi::DataType::C64 ||
        radius.dimensions().size() != 0)
        return ffi::Error::InvalidArgument("ProjectRelionHalfRuntime: require C64 half, F32 rotations, scalar S32 radius and C64 output");
    if (padding_factor != 1 && padding_factor != 2)
        return ffi::Error::InvalidArgument("ProjectRelionHalfRuntime: padding must be 1 or 2");
    // The staging fill kernel (launch_project_texture_float) indexes texels with int.
    if (h.size() != 3 || h[0] < 5 || h[0] != h[1] || h[0] % 2 != 1 ||
        h[2] != h[0] / 2 + 1 || (h[0] - 3) % (2 * padding_factor) != 0 ||
        h[0] * h[1] * h[2] > std::numeric_limits<int>::max() ||
        r.size() != 2 || r[1] != 6 || r[0] <= 0 || r[0] > 65535 ||
        image_h <= 0 || image_h != image_w || image_h % 2 != 0 || image_h > 4096 ||
        o.size() != 2 || o[0] != r[0] || o[1] != image_h * (image_w / 2 + 1) ||
        r[0] * o[1] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument("ProjectRelionHalfRuntime: invalid capacity, rotation or image geometry");
    cudaError_t err = launch_project_texture_float<true>(
        stream, static_cast<const float*>(half.untyped_data()),
        static_cast<float*>(output->untyped_data()), static_cast<const float*>(rot.untyped_data()),
        r[0], o[1], image_h, image_w / 2 + 1,
        h[0], h[1], h[2], padding_factor, 1, image_w, -1,
        static_cast<const int32_t*>(radius.untyped_data()), image_radius);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error ProjectRelionHalfRuntimeImpl(
    cudaStream_t stream, int64_t image_h, int64_t image_w,
    int64_t padding_factor,
    ffi::AnyBuffer half, ffi::AnyBuffer rot, ffi::AnyBuffer radius,
    ffi::Result<ffi::AnyBuffer> output)
{
    return ProjectRelionHalfRuntimeCore(
        stream, image_h, image_w, padding_factor, half, rot, radius, output, nullptr);
}

ffi::Error ProjectRelionHalfImageRadiusImpl(
    cudaStream_t stream, int64_t image_h, int64_t image_w,
    int64_t padding_factor,
    ffi::AnyBuffer half, ffi::AnyBuffer rot, ffi::AnyBuffer radius,
    ffi::AnyBuffer image_radius, ffi::Result<ffi::AnyBuffer> output)
{
    if (image_radius.element_type() != ffi::DataType::S32 ||
        image_radius.dimensions().size() != 0)
        return ffi::Error::InvalidArgument("ProjectRelionHalfImageRadius: image radius must be scalar S32");
    return ProjectRelionHalfRuntimeCore(
        stream, image_h, image_w, padding_factor, half, rot, radius, output,
        static_cast<const int32_t*>(image_radius.untyped_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProjectRelionHalfImageRadius, ProjectRelionHalfImageRadiusImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("padding_factor")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

#include "relion_capacity_texture.cuh"

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProjectRelionHalfRuntime, ProjectRelionHalfRuntimeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("padding_factor")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

/* ================================================================== */
/*       RELION BPref x=0 + point-group symmetry finalisation          */
/* ================================================================== */

template <typename T>
struct RelionSymmetryComplex {
    T real;
    T imag;
};

template <typename T>
static __device__ __forceinline__ RelionSymmetryComplex<T>
relion_symmetry_lerp(
    T fraction,
    RelionSymmetryComplex<T> low,
    RelionSymmetryComplex<T> high)
{
    return {
        low.real + (high.real - low.real) * fraction,
        low.imag + (high.imag - low.imag) * fraction,
    };
}

template <typename T>
static __device__ __forceinline__ T relion_symmetry_lerp(
    T fraction, T low, T high)
{
    return low + (high - low) * fraction;
}

template <typename T>
static __device__ __forceinline__ RelionSymmetryComplex<T>
relion_symmetry_read_data(
    const T* __restrict__ data,
    int z_index,
    int y_index,
    int x_index,
    int full_z,
    int full_y,
    int x_half)
{
    const int64_t index =
        (static_cast<int64_t>(z_index) * full_y + y_index) * x_half + x_index;
    RelionSymmetryComplex<T> value = {data[2 * index], data[2 * index + 1]};
    if (x_index != 0)
        return value;

    /* BackProjector::enforceHermitianSymmetry sums each x=0 pair and does
     * not divide by two.  BPref grids are odd, so the partner of direct
     * index i is N-1-i.  The (z=0,y=0) center is deliberately untouched. */
    const int partner_z = full_z - 1 - z_index;
    const int partner_y = full_y - 1 - y_index;
    if (partner_z == z_index && partner_y == y_index)
        return value;
    const int64_t partner_index =
        (static_cast<int64_t>(partner_z) * full_y + partner_y) * x_half;
    value.real += data[2 * partner_index];
    value.imag -= data[2 * partner_index + 1];
    return value;
}

template <typename T>
static __device__ __forceinline__ RelionSymmetryComplex<T>
relion_symmetry_read_data_split(
    const T* __restrict__ data_real,
    const T* __restrict__ data_imag,
    int z_index,
    int y_index,
    int x_index,
    int full_z,
    int full_y,
    int x_half)
{
    const int64_t index =
        (static_cast<int64_t>(z_index) * full_y + y_index) * x_half + x_index;
    RelionSymmetryComplex<T> value = {data_real[index], data_imag[index]};
    if (x_index != 0)
        return value;

    const int partner_z = full_z - 1 - z_index;
    const int partner_y = full_y - 1 - y_index;
    if (partner_z == z_index && partner_y == y_index)
        return value;
    const int64_t partner_index =
        (static_cast<int64_t>(partner_z) * full_y + partner_y) * x_half;
    value.real += data_real[partner_index];
    value.imag -= data_imag[partner_index];
    return value;
}

template <typename T, bool SPLIT_INPUT>
static __device__ __forceinline__ RelionSymmetryComplex<T>
relion_symmetry_read_data_layout(
    const T* __restrict__ data,
    const T* __restrict__ data_imag,
    int z_index,
    int y_index,
    int x_index,
    int full_z,
    int full_y,
    int x_half)
{
    if constexpr (SPLIT_INPUT)
        return relion_symmetry_read_data_split(
            data, data_imag, z_index, y_index, x_index, full_z, full_y, x_half);
    return relion_symmetry_read_data(
        data, z_index, y_index, x_index, full_z, full_y, x_half);
}

template <typename T>
static __device__ __forceinline__ T relion_symmetry_read_weight(
    const T* __restrict__ weight,
    int z_index,
    int y_index,
    int x_index,
    int full_z,
    int full_y,
    int x_half)
{
    const int64_t index =
        (static_cast<int64_t>(z_index) * full_y + y_index) * x_half + x_index;
    T value = weight[index];
    if (x_index != 0)
        return value;
    const int partner_z = full_z - 1 - z_index;
    const int partner_y = full_y - 1 - y_index;
    if (partner_z == z_index && partner_y == y_index)
        return value;
    const int64_t partner_index =
        (static_cast<int64_t>(partner_z) * full_y + partner_y) * x_half;
    return value + weight[partner_index];
}

template <typename T, bool SPLIT_INPUT, bool RANGED_OUTPUT>
__global__ void __launch_bounds__(BLOCK_SIZE)
relion_point_group_symmetrise_bpref_kernel(
    const T* __restrict__ data,
    const T* __restrict__ data_imag,
    const T* __restrict__ weight,
    const T* __restrict__ right_operators,
    const int64_t* __restrict__ range_start,
    T* __restrict__ data_out,
    T* __restrict__ weight_out,
    int full_z,
    int full_y,
    int full_x,
    int support_radius,
    int operator_count,
    int64_t output_count,
    int64_t voxel_count)
{
    const int64_t output_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count)
        return;
    int64_t direct_index = output_index;
    if constexpr (RANGED_OUTPUT)
        direct_index = range_start[0] + output_index;
    if (direct_index < 0 || direct_index >= voxel_count) {
        data_out[2 * output_index] = static_cast<T>(0);
        data_out[2 * output_index + 1] = static_cast<T>(0);
        weight_out[output_index] = static_cast<T>(0);
        return;
    }

    const int x_half = full_x / 2 + 1;
    const int x_index = static_cast<int>(direct_index % x_half);
    const int64_t yz_index = direct_index / x_half;
    const int y_index = static_cast<int>(yz_index % full_y);
    const int z_index = static_cast<int>(yz_index / full_y);

    const int x_logical = x_index;
    const int y_logical = y_index - full_y / 2;
    const int z_logical = z_index - full_z / 2;

    /* RELION initialises sum_data/sum_weight by copying the x=0-enforced
     * source.  Keep that identity contribution even outside rmax2. */
    RelionSymmetryComplex<T> data_sum = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
        data, data_imag, z_index, y_index, x_index, full_z, full_y, x_half);
    T weight_sum = relion_symmetry_read_weight(
        weight, z_index, y_index, x_index, full_z, full_y, x_half);

    const int64_t radius_squared =
        static_cast<int64_t>(x_logical) * x_logical +
        static_cast<int64_t>(y_logical) * y_logical +
        static_cast<int64_t>(z_logical) * z_logical;
    const int64_t support_squared =
        static_cast<int64_t>(support_radius) * support_radius;

    if (radius_squared <= support_squared) {
        const T x = static_cast<T>(x_logical);
        const T y = static_cast<T>(y_logical);
        const T z = static_cast<T>(z_logical);

        /* Operator zero is identity and was copied above.  Stream through
         * RELION's remaining SymList order without rotated-volume copies. */
        for (int operator_index = 1; operator_index < operator_count; ++operator_index) {
            const T* R = right_operators + static_cast<int64_t>(operator_index) * 9;
            T xp = x * R[0] + y * R[1] + z * R[2];
            T yp = x * R[3] + y * R[4] + z * R[5];
            T zp = x * R[6] + y * R[7] + z * R[8];

            bool conjugate_sample = false;
            if (xp < static_cast<T>(0)) {
                xp = -xp;
                yp = -yp;
                zp = -zp;
                conjugate_sample = true;
            }

            const int x0 = floor_int(xp);
            const int y0 = floor_int(yp) + full_y / 2;
            const int z0 = floor_int(zp) + full_z / 2;
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const int z1 = z0 + 1;
            const T fx = xp - static_cast<T>(x0);
            const T fy = yp - static_cast<T>(y0 - full_y / 2);
            const T fz = zp - static_cast<T>(z0 - full_z / 2);

            const RelionSymmetryComplex<T> d000 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z0, y0, x0, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d001 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z0, y0, x1, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d010 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z0, y1, x0, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d011 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z0, y1, x1, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d100 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z1, y0, x0, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d101 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z1, y0, x1, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d110 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z1, y1, x0, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d111 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z1, y1, x1, full_z, full_y, x_half);

            const RelionSymmetryComplex<T> dx00 = relion_symmetry_lerp(fx, d000, d001);
            const RelionSymmetryComplex<T> dx01 = relion_symmetry_lerp(fx, d100, d101);
            const RelionSymmetryComplex<T> dx10 = relion_symmetry_lerp(fx, d010, d011);
            const RelionSymmetryComplex<T> dx11 = relion_symmetry_lerp(fx, d110, d111);
            const RelionSymmetryComplex<T> dxy0 = relion_symmetry_lerp(fy, dx00, dx10);
            const RelionSymmetryComplex<T> dxy1 = relion_symmetry_lerp(fy, dx01, dx11);
            RelionSymmetryComplex<T> sample = relion_symmetry_lerp(fz, dxy0, dxy1);
            if (conjugate_sample)
                sample.imag = -sample.imag;
            data_sum.real += sample.real;
            data_sum.imag += sample.imag;

            const T w000 = relion_symmetry_read_weight(
                weight, z0, y0, x0, full_z, full_y, x_half);
            const T w001 = relion_symmetry_read_weight(
                weight, z0, y0, x1, full_z, full_y, x_half);
            const T w010 = relion_symmetry_read_weight(
                weight, z0, y1, x0, full_z, full_y, x_half);
            const T w011 = relion_symmetry_read_weight(
                weight, z0, y1, x1, full_z, full_y, x_half);
            const T w100 = relion_symmetry_read_weight(
                weight, z1, y0, x0, full_z, full_y, x_half);
            const T w101 = relion_symmetry_read_weight(
                weight, z1, y0, x1, full_z, full_y, x_half);
            const T w110 = relion_symmetry_read_weight(
                weight, z1, y1, x0, full_z, full_y, x_half);
            const T w111 = relion_symmetry_read_weight(
                weight, z1, y1, x1, full_z, full_y, x_half);
            const T wx00 = relion_symmetry_lerp(fx, w000, w001);
            const T wx01 = relion_symmetry_lerp(fx, w100, w101);
            const T wx10 = relion_symmetry_lerp(fx, w010, w011);
            const T wx11 = relion_symmetry_lerp(fx, w110, w111);
            const T wxy0 = relion_symmetry_lerp(fy, wx00, wx10);
            const T wxy1 = relion_symmetry_lerp(fy, wx01, wx11);
            weight_sum += relion_symmetry_lerp(fz, wxy0, wxy1);
        }
    }

    data_out[2 * output_index] = data_sum.real;
    data_out[2 * output_index + 1] = data_sum.imag;
    weight_out[output_index] = weight_sum;
}

template <typename T>
static cudaError_t launch_relion_point_group_symmetrise_bpref(
    cudaStream_t stream,
    const T* data,
    const T* weight,
    const T* right_operators,
    T* data_out,
    T* weight_out,
    int full_z,
    int full_y,
    int full_x,
    int support_radius,
    int operator_count)
{
    const int64_t voxel_count =
        static_cast<int64_t>(full_z) * full_y * (full_x / 2 + 1);
    const int64_t block_count = (voxel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relion_point_group_symmetrise_bpref_kernel<T, false, false>
        <<<static_cast<unsigned int>(block_count), BLOCK_SIZE, 0, stream>>>(
            data,
            nullptr,
            weight,
            right_operators,
            nullptr,
            data_out,
            weight_out,
            full_z,
            full_y,
            full_x,
            support_radius,
            operator_count,
            voxel_count,
            voxel_count);
    return cudaGetLastError();
}

template <typename T, bool SPLIT_INPUT = true>
static cudaError_t launch_relion_point_group_symmetrise_bpref_split_range(
    cudaStream_t stream,
    const T* data_real,
    const T* data_imag,
    const T* weight,
    const T* right_operators,
    const int64_t* range_start,
    T* data_out,
    T* weight_out,
    int full_z,
    int full_y,
    int full_x,
    int support_radius,
    int operator_count,
    int64_t output_count)
{
    const int64_t voxel_count =
        static_cast<int64_t>(full_z) * full_y * (full_x / 2 + 1);
    const int64_t block_count = (output_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relion_point_group_symmetrise_bpref_kernel<T, SPLIT_INPUT, true>
        <<<static_cast<unsigned int>(block_count), BLOCK_SIZE, 0, stream>>>(
            data_real,
            data_imag,
            weight,
            right_operators,
            range_start,
            data_out,
            weight_out,
            full_z,
            full_y,
            full_x,
            support_radius,
            operator_count,
            output_count,
            voxel_count);
    return cudaGetLastError();
}

ffi::Error RelionPointGroupSymmetriseBprefImpl(
    cudaStream_t stream,
    int64_t full_z,
    int64_t full_y,
    int64_t full_x,
    int64_t support_radius,
    ffi::AnyBuffer data,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer right_operators,
    ffi::Result<ffi::AnyBuffer> data_out,
    ffi::Result<ffi::AnyBuffer> weight_out)
{
    const auto data_dims = data.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto operator_dims = right_operators.dimensions();
    const auto data_out_dims = data_out->dimensions();
    const auto weight_out_dims = weight_out->dimensions();
    if (data_dims.size() != 1 || weight_dims.size() != 1 ||
        data_out_dims.size() != 1 || weight_out_dims.size() != 1 ||
        data_dims[0] != weight_dims[0] || data_dims[0] != data_out_dims[0] ||
        data_dims[0] != weight_out_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: data and weight must be matching flat buffers");
    if (operator_dims.size() != 3 || operator_dims[0] < 1 ||
        operator_dims[1] != 3 || operator_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: operators must have shape (n,3,3)");
    if (full_z <= 0 || full_y <= 0 || full_x <= 0 ||
        full_z != full_y || full_z != full_x || (full_x % 2) == 0)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: full BPref dimensions must be equal positive odd values");
    if (full_z > std::numeric_limits<int>::max() ||
        operator_dims[0] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: dimensions exceed CUDA integer bounds");
    const int64_t expected_voxels = full_z * full_y * (full_x / 2 + 1);
    if (data_dims[0] != expected_voxels)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: flat accumulator size does not match full dimensions");
    const int64_t maximum_supported_radius = full_x / 2 - 1;
    if (support_radius < 0 || support_radius > maximum_supported_radius)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: support radius must leave one interpolation voxel");

    cudaError_t err;
    if (data.element_type() == ffi::DataType::C64 &&
        data_out->element_type() == ffi::DataType::C64 &&
        weight.element_type() == ffi::DataType::F32 &&
        weight_out->element_type() == ffi::DataType::F32 &&
        right_operators.element_type() == ffi::DataType::F32) {
        err = launch_relion_point_group_symmetrise_bpref<float>(
            stream,
            static_cast<const float*>(data.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(right_operators.untyped_data()),
            static_cast<float*>(data_out->untyped_data()),
            static_cast<float*>(weight_out->untyped_data()),
            static_cast<int>(full_z),
            static_cast<int>(full_y),
            static_cast<int>(full_x),
            static_cast<int>(support_radius),
            static_cast<int>(operator_dims[0]));
    } else if (data.element_type() == ffi::DataType::C128 &&
               data_out->element_type() == ffi::DataType::C128 &&
               weight.element_type() == ffi::DataType::F64 &&
               weight_out->element_type() == ffi::DataType::F64 &&
               right_operators.element_type() == ffi::DataType::F64) {
        err = launch_relion_point_group_symmetrise_bpref<double>(
            stream,
            static_cast<const double*>(data.untyped_data()),
            static_cast<const double*>(weight.untyped_data()),
            static_cast<const double*>(right_operators.untyped_data()),
            static_cast<double*>(data_out->untyped_data()),
            static_cast<double*>(weight_out->untyped_data()),
            static_cast<int>(full_z),
            static_cast<int>(full_y),
            static_cast<int>(full_x),
            static_cast<int>(support_radius),
            static_cast<int>(operator_dims[0]));
    } else {
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: expected C64/F32/F32 or C128/F64/F64 buffers");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPointGroupSymmetriseBpref, RelionPointGroupSymmetriseBprefImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("full_z")
        .Attr<int64_t>("full_y")
        .Attr<int64_t>("full_x")
        .Attr<int64_t>("support_radius")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());


ffi::Error RelionPointGroupSymmetriseBprefComplexRangeImpl(
    cudaStream_t stream,
    int64_t full_z,
    int64_t full_y,
    int64_t full_x,
    int64_t support_radius,
    ffi::AnyBuffer data_real,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer right_operators,
    ffi::AnyBuffer range_start,
    ffi::Result<ffi::AnyBuffer> data_out,
    ffi::Result<ffi::AnyBuffer> weight_out)
{
    const auto data_real_dims = data_real.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto operator_dims = right_operators.dimensions();
    const auto range_start_dims = range_start.dimensions();
    const auto data_out_dims = data_out->dimensions();
    const auto weight_out_dims = weight_out->dimensions();
    if (data_real_dims.size() != 1 ||
        weight_dims.size() != 1 ||
        data_real_dims[0] != weight_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: complex/weight inputs must be matching flat buffers");
    if (data_out_dims.size() != 1 || weight_out_dims.size() != 1 ||
        data_out_dims[0] <= 0 || data_out_dims[0] != weight_out_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: outputs must be matching nonempty flat ranges");
    if (range_start_dims.size() != 1 || range_start_dims[0] != 1 ||
        range_start.element_type() != ffi::DataType::S64)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: range_start must be one int64 value");
    if (operator_dims.size() != 3 || operator_dims[0] < 1 ||
        operator_dims[1] != 3 || operator_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: operators must have shape (n,3,3)");
    if (full_z <= 0 || full_y <= 0 || full_x <= 0 ||
        full_z != full_y || full_z != full_x || (full_x % 2) == 0)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: full BPref dimensions must be equal positive odd values");
    if (full_z > std::numeric_limits<int>::max() ||
        operator_dims[0] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: dimensions exceed CUDA integer bounds");
    const int64_t expected_voxels = full_z * full_y * (full_x / 2 + 1);
    if (data_real_dims[0] != expected_voxels || data_out_dims[0] > expected_voxels)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: input or range size does not match full dimensions");
    const int64_t maximum_supported_radius = full_x / 2 - 1;
    if (support_radius < 0 || support_radius > maximum_supported_radius)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: support radius must leave one interpolation voxel");
    if (data_real.element_type() != ffi::DataType::C64 ||
        weight.element_type() != ffi::DataType::F32 ||
        right_operators.element_type() != ffi::DataType::F32 ||
        data_out->element_type() != ffi::DataType::C64 ||
        weight_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: expected C64/F32/F32/S64 inputs and C64/F32 outputs");

    cudaError_t err = launch_relion_point_group_symmetrise_bpref_split_range<float, false>(
        stream,
        static_cast<const float*>(data_real.untyped_data()),
        nullptr,
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(right_operators.untyped_data()),
        static_cast<const int64_t*>(range_start.untyped_data()),
        static_cast<float*>(data_out->untyped_data()),
        static_cast<float*>(weight_out->untyped_data()),
        static_cast<int>(full_z),
        static_cast<int>(full_y),
        static_cast<int>(full_x),
        static_cast<int>(support_radius),
        static_cast<int>(operator_dims[0]),
        static_cast<int64_t>(data_out_dims[0]));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPointGroupSymmetriseBprefComplexRange,
    RelionPointGroupSymmetriseBprefComplexRangeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("full_z")
        .Attr<int64_t>("full_y")
        .Attr<int64_t>("full_x")
        .Attr<int64_t>("support_radius")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());


ffi::Error RelionWavgRotationAtomicRuntimeTripletAddF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer terms,
    ffi::AnyBuffer accumulator_in,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> accumulator_out)
{
    if (terms.element_type() != ffi::DataType::F32 ||
        accumulator_in.element_type() != ffi::DataType::F32 ||
        accumulator_out->element_type() != ffi::DataType::F32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicRuntimeTripletAddF32: invalid buffers");
    const auto dims = terms.dimensions();
    const auto accumulator_dims = accumulator_in.dimensions();
    const auto output_dims = accumulator_out->dimensions();
    if (dims.size() != 4 || accumulator_dims.size() != 3 ||
        output_dims.size() != 3 || dims[3] != 3 ||
        accumulator_dims[2] != 3 || output_dims[0] != accumulator_dims[0] ||
        output_dims[1] != accumulator_dims[1] ||
        output_dims[2] != accumulator_dims[2] ||
        accumulator_dims[0] != dims[0] || accumulator_dims[1] != dims[2] ||
        dims[0] <= 0 || dims[0] > 65535 || dims[1] <= 0 ||
        dims[1] > std::numeric_limits<int>::max() || dims[2] <= 0 ||
        dims[2] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicRuntimeTripletAddF32: inconsistent topology");
    cudaError_t err = launch_relion_wavg_rotation_atomic_runtime_triplet_add_f32(
        stream,
        static_cast<const float*>(terms.untyped_data()),
        static_cast<float*>(accumulator_out->untyped_data()),
        dims[0],
        dims[1],
        dims[2],
        static_cast<const int32_t*>(logical_pixel_count.untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgRotationAtomicRuntimeTripletAddF32,
    RelionWavgRotationAtomicRuntimeTripletAddF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());


// Flat-row Wavg: projections[Q,P] and posterior[Q,T] are packed candidate rows,
// row_image_ids[Q] carries each row's image, and negative ids mark padding.
ffi::Error RelionWavgSequentialRuntimeFlatRowsTripletF32Impl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::AnyBuffer projections,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer raw_ctf,
    ffi::AnyBuffer scale,
    ffi::AnyBuffer shifted_images,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projections.element_type() != ffi::DataType::C64 ||
        raw_ctf.element_type() != ffi::DataType::F32 ||
        scale.element_type() != ffi::DataType::F32 ||
        shifted_images.element_type() != ffi::DataType::C64 ||
        posterior.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32 ||
        row_image_ids.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeFlatRowsTripletF32: invalid buffers");
    const auto projection_dims = projections.dimensions();
    const auto row_dims = row_image_ids.dimensions();
    const auto ctf_dims = raw_ctf.dimensions();
    const auto scale_dims = scale.dimensions();
    const auto shifted_dims = shifted_images.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto output_dims = output->dimensions();
    if (projection_dims.size() != 2 || row_dims.size() != 1 ||
        ctf_dims.size() != 2 || scale_dims.size() != 1 ||
        shifted_dims.size() != 3 || posterior_dims.size() != 2 ||
        output_dims.size() != 3 || output_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeFlatRowsTripletF32: invalid ranks");
    const int64_t row_count = projection_dims[0];
    const int64_t pixel_capacity = projection_dims[1];
    const int64_t batch_size = ctf_dims[0];
    const int64_t translation_count = shifted_dims[1];
    if (row_count <= 0 || row_count > std::numeric_limits<int>::max() ||
        batch_size <= 0 || translation_count <= 0 || pixel_capacity <= 0 ||
        pixel_capacity > std::numeric_limits<int>::max() ||
        row_dims[0] != row_count || ctf_dims[1] != pixel_capacity ||
        scale_dims[0] != batch_size || shifted_dims[0] != batch_size ||
        shifted_dims[2] != pixel_capacity || posterior_dims[0] != row_count ||
        posterior_dims[1] != translation_count ||
        output_dims[0] != row_count || output_dims[1] != pixel_capacity)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeFlatRowsTripletF32: inconsistent topology");
    cudaError_t err = launch_relion_wavg_sequential_runtime_flat_rows_triplet_f32(
        stream,
        scratch,
        reinterpret_cast<const float2*>(projections.untyped_data()),
        static_cast<const int32_t*>(row_image_ids.untyped_data()),
        static_cast<const float*>(raw_ctf.untyped_data()),
        static_cast<const float*>(scale.untyped_data()),
        reinterpret_cast<const float2*>(shifted_images.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        batch_size,
        row_count,
        translation_count,
        pixel_capacity,
        static_cast<const int32_t*>(logical_pixel_count.untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgSequentialRuntimeFlatRowsTripletF32,
    RelionWavgSequentialRuntimeFlatRowsTripletF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer terms,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer accumulator_in,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> accumulator_out)
{
    if (terms.element_type() != ffi::DataType::F32 ||
        accumulator_in.element_type() != ffi::DataType::F32 ||
        accumulator_out->element_type() != ffi::DataType::F32 ||
        row_image_ids.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32: invalid buffers");
    const auto dims = terms.dimensions();
    const auto row_dims = row_image_ids.dimensions();
    const auto accumulator_dims = accumulator_in.dimensions();
    const auto output_dims = accumulator_out->dimensions();
    if (dims.size() != 3 || row_dims.size() != 1 ||
        accumulator_dims.size() != 3 || output_dims.size() != 3 ||
        dims[2] != 3 || accumulator_dims[2] != 3 ||
        output_dims[0] != accumulator_dims[0] ||
        output_dims[1] != accumulator_dims[1] ||
        output_dims[2] != accumulator_dims[2] ||
        row_dims[0] != dims[0] || accumulator_dims[1] != dims[1] ||
        dims[0] <= 0 || dims[0] > std::numeric_limits<int>::max() ||
        dims[1] <= 0 || dims[1] > std::numeric_limits<int>::max() ||
        accumulator_dims[0] <= 0 ||
        accumulator_dims[0] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32: inconsistent topology");
    cudaError_t err =
        launch_relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
            stream,
            static_cast<const float*>(terms.untyped_data()),
            static_cast<const int32_t*>(row_image_ids.untyped_data()),
            static_cast<float*>(accumulator_out->untyped_data()),
            accumulator_dims[0],
            dims[0],
            dims[1],
            static_cast<const int32_t*>(logical_pixel_count.untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32,
    RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

/* The flat-row Wavg atomics on the rectangle's exact-radius pixels only:
 * ``terms`` [Q, P_exact, 3] adds into positions ``exact_positions`` of the
 * [B, P_rect, 3] accumulator.  Positions outside the exact radius keep their
 * accumulator values. */
ffi::Error RelionWavgExactAtomicFlatRowsTripletAddF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer terms,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer exact_positions,
    ffi::AnyBuffer accumulator_in,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> accumulator_out)
{
    if (terms.element_type() != ffi::DataType::F32 ||
        accumulator_in.element_type() != ffi::DataType::F32 ||
        accumulator_out->element_type() != ffi::DataType::F32 ||
        row_image_ids.element_type() != ffi::DataType::S32 ||
        exact_positions.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionWavgExactAtomicFlatRowsTripletAddF32: invalid buffers");
    const auto dims = terms.dimensions();
    const auto row_dims = row_image_ids.dimensions();
    const auto position_dims = exact_positions.dimensions();
    const auto accumulator_dims = accumulator_in.dimensions();
    const auto output_dims = accumulator_out->dimensions();
    if (dims.size() != 3 || row_dims.size() != 1 || position_dims.size() != 1 ||
        accumulator_dims.size() != 3 || output_dims.size() != 3 ||
        dims[2] != 3 || accumulator_dims[2] != 3 ||
        output_dims[0] != accumulator_dims[0] ||
        output_dims[1] != accumulator_dims[1] ||
        output_dims[2] != accumulator_dims[2] ||
        row_dims[0] != dims[0] || position_dims[0] != dims[1] ||
        dims[1] > accumulator_dims[1] ||
        dims[0] <= 0 || dims[0] > std::numeric_limits<int>::max() ||
        dims[1] <= 0 || dims[1] > std::numeric_limits<int>::max() ||
        accumulator_dims[1] > std::numeric_limits<int>::max() ||
        accumulator_dims[0] <= 0 ||
        accumulator_dims[0] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionWavgExactAtomicFlatRowsTripletAddF32: inconsistent topology");
    cudaError_t err =
        launch_relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
            stream,
            static_cast<const float*>(terms.untyped_data()),
            static_cast<const int32_t*>(row_image_ids.untyped_data()),
            static_cast<float*>(accumulator_out->untyped_data()),
            accumulator_dims[0],
            dims[0],
            dims[1],
            static_cast<const int32_t*>(logical_pixel_count.untyped_data()),
            static_cast<const int32_t*>(exact_positions.untyped_data()),
            accumulator_dims[1]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgExactAtomicFlatRowsTripletAddF32,
    RelionWavgExactAtomicFlatRowsTripletAddF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

/* Pair-sparse dual weighted sums.  For compact K-class pairs a dense
 * (B, R, T) probability table is ~1 % populated and 73 % of its rows are
 * padding, so each row here owns the contiguous pair range
 * [row_offsets[b, r], row_offsets[b, r + 1]) of pairs sorted by
 * (rotation row, translation); iterating that range in order performs the
 * fma sequence of a dense loop over ascending translations that skips zero
 * probabilities. */


// Fused sparse pass-2 posterior handlers (needs relion_ampere_inclusive_sum_f32).
#include "sparse_pass2_posterior.cuh"

// Flat-row translate-and-sum for the device-resident pass-2 M-step
// (needs relion_score_translate_f32's phase and rotation).
#include "relion_translate_sum.cuh"

// PPCA M-step moment backprojection (voxel-major accumulator, warp-wide channel atomics).
#include "ppca_moment_backproject.cuh"

// Fused elementwise stages of the streamed PPCA engine (windowed projection, latent epilogue,
// posterior weights and moment sums).
#include "ppca_stream.cuh"
