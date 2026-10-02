// PPCA M-step moment backprojection into a voxel-major accumulator.
//
// Scatters the windowed half-image moment images of one rotation block (real LHS channels and
// complex residual channels) into a half volume, with the targets and weights of recovar's
// batch_backproject_indexed_kernel<float, 1, HALF_VOL, HALF_IMG> (linear interpolation,
// RELION-centred half volume and half image, no x-fold, unit upsampling): every windowed
// half-image pixel scatters trilinearly at its rotated frequency and, off the self-conjugate
// columns, at its Hermitian partner; targets with kz < 0 fold to their Hermitian partner.
//
// The accumulator is (groups, V, 32): channel c lives in lane c % 32 of group c / 32 (the real
// channels, then the real and imaginary parts of each complex channel). One warp handles 32
// pixels of one rotation: each lane first expands its own pixel's targets, then for every pixel
// the warp adds one channel per lane, so each target is one warp-wide atomic to one 128-byte voxel
// row instead of one scattered atomic per channel. Only the float32 association order of each
// voxel's sum differs from recovar's channel-major atomics.
#pragma once

namespace ppca_moment_bp {

constexpr int kWarps = 4;
constexpr int kMaxTargets = 16;  // 8 trilinear targets of the pixel and 8 of its partner

struct Geometry {
    int n_rot, n_pix, image_h, image_w, full_image_w;
    int N0, N1, N2_eff, N2_full;
    float c0, c1, c2, max_r2;
};

// emit(i0, i1, hkz, weight, fold) for every half-volume target of one trilinear point.
// MODE (recovar CONJ_MODE): 0 normal, 1 double interior kz, 2 boundary columns only.
template <int MODE, typename Emit>
__device__ __forceinline__ void trilinear_targets(const Geometry& g, float rk0, float rk1, float rk2, Emit& emit) {
    const float g0 = rk0 + g.c0, g1 = rk1 + g.c1, g2 = rk2 + g.c2;
    const int ic2 = (int)g.c2;
    if (g0 < -1.f || g0 >= (float)g.N0 || g1 < -1.f || g1 >= (float)g.N1 || g2 < -1.f || g2 >= (float)g.N2_full)
        return;
    const int b0 = (int)floorf(g0), b1 = (int)floorf(g1), b2 = (int)floorf(g2);
    const float f0 = g0 - (float)b0, f1 = g1 - (float)b1, f2 = g2 - (float)b2;
    const float w0[2] = {1.f - f0, f0}, w1[2] = {1.f - f1, f1}, w2[2] = {1.f - f2, f2};
#pragma unroll
    for (int d0 = 0; d0 < 2; d0++) {
        const int j0 = b0 + d0;
        if ((unsigned)j0 >= (unsigned)g.N0) continue;
#pragma unroll
        for (int d1 = 0; d1 < 2; d1++) {
            const int j1 = b1 + d1;
            if ((unsigned)j1 >= (unsigned)g.N1) continue;
            const float ww = w0[d0] * w1[d1];
#pragma unroll
            for (int d2 = 0; d2 < 2; d2++) {
                const int j2 = b2 + d2;
                if ((unsigned)j2 >= (unsigned)g.N2_full) continue;
                const int kz = j2 - ic2;
                float w = ww * w2[d2];
                int sj0 = j0, sj1 = j1, hkz;
                bool fold = false;
                if (kz >= 0) {
                    hkz = kz;
                } else if ((g.N2_full & 1) == 0 && -kz == ic2) {
                    hkz = ic2;  // Nyquist: self-conjugate
                } else {
                    // Hermitian partner in the centred convention: (N - (N & 1) - j) % N.
                    sj0 = (g.N0 - (g.N0 & 1) - j0) % g.N0;
                    sj1 = (g.N1 - (g.N1 & 1) - j1) % g.N1;
                    hkz = -kz;
                    fold = true;
                }
                if (hkz > ic2) continue;
                if (MODE == 2 && hkz > 0 && hkz < ic2) continue;
                if (MODE == 1 && hkz > 0 && hkz < ic2) w *= 2.f;
                emit(sj0, sj1, hkz, w, fold);
            }
        }
    }
}

// Every target of windowed pixel `orig_pix` under rotation R (rows R[0:3], R[3:6]). The partner
// scatter is conjugated (emit.conjugate); with a fold the two conjugations cancel.
template <typename Emit>
__device__ __forceinline__ void pixel_targets(const Geometry& g, const float* R, int orig_pix, Emit& emit) {
    const int k0_idx = orig_pix / g.image_w, k1_idx = orig_pix % g.image_w;
    const float k0 = (float)(k0_idx - g.image_h / 2);
    const float k1 = (k1_idx * 2 == g.full_image_w) ? (float)(-k1_idx) : (float)k1_idx;
    if (g.max_r2 >= 0.f && k0 * k0 + k1 * k1 > g.max_r2) return;
    const float rk0 = k0 * R[0] + k1 * R[3], rk1 = k0 * R[1] + k1 * R[4], rk2 = k0 * R[2] + k1 * R[5];
    bool conj_opt = (k1_idx > 0 && k1_idx * 2 != g.full_image_w) && !(k0_idx == 0 && (g.image_h & 1) == 0);
    if (conj_opt) {
        const float pg0 = rk0 + g.c0, pg1 = rk1 + g.c1, pg2 = rk2 + g.c2;
        const float cg0 = -rk0 + g.c0, cg1 = -rk1 + g.c1, cg2 = -rk2 + g.c2;
        if (pg0 < 0.f || pg0 > (float)(g.N0 - 1) || pg1 < 0.f || pg1 > (float)(g.N1 - 1) || pg2 < 0.f ||
            pg2 > (float)(g.N2_full - 1) || cg0 < 0.f || cg0 > (float)(g.N0 - 1) || cg1 < 0.f ||
            cg1 > (float)(g.N1 - 1) || cg2 < 0.f || cg2 > (float)(g.N2_full - 1))
            conj_opt = false;
    }
    emit.conjugate = false;
    if (conj_opt) trilinear_targets<1>(g, rk0, rk1, rk2, emit);
    else trilinear_targets<0>(g, rk0, rk1, rk2, emit);
    if (k1_idx > 0 && k1_idx * 2 != g.full_image_w) {
        float crk0, crk1, crk2;
        if (k0_idx == 0 && (g.image_h & 1) == 0) {
            const float nk1 = -k1;
            crk0 = k0 * R[0] + nk1 * R[3];
            crk1 = k0 * R[1] + nk1 * R[4];
            crk2 = k0 * R[2] + nk1 * R[5];
        } else {
            crk0 = -rk0;
            crk1 = -rk1;
            crk2 = -rk2;
        }
        emit.conjugate = true;
        if (conj_opt) trilinear_targets<2>(g, crk0, crk1, crk2, emit);
        else trilinear_targets<0>(g, crk0, crk1, crk2, emit);
        emit.conjugate = false;
    }
}

// One lane's targets of its own pixel: voxel offset (high bit: negate imaginary parts) and weight.
struct Collect {
    const Geometry& g;
    int* offset;
    float* weight;
    int n = 0;
    bool conjugate = false;
    __device__ Collect(const Geometry& geometry, int* o, float* w) : g(geometry), offset(o), weight(w) {}
    __device__ __forceinline__ void operator()(int i0, int i1, int i2, float w, bool fold) {
        const int o = (i0 * g.N1 + i1) * g.N2_eff + i2;
        offset[n] = (fold != conjugate) ? (o | (1 << 31)) : o;
        weight[n] = w;
        n++;
    }
};

// grid: (ceil(n_pix / (32 kWarps)), n_rot). real (n_real, n_rot, n_pix); complex (n_cplx, n_rot, n_pix);
// out (groups, V, 32), zero on entry.
__global__ void __launch_bounds__(32 * kWarps) moment_backproject_kernel(
    Geometry g, const float* __restrict__ rot, const int* __restrict__ pix_idx, const float* __restrict__ real_images,
    const float2* __restrict__ cplx_images, int n_real, int n_cplx, float* __restrict__ out) {
    __shared__ int s_offset[kWarps][32][kMaxTargets];
    __shared__ float s_weight[kWarps][32][kMaxTargets];
    __shared__ int s_count[kWarps][32];
    __shared__ float s_value[kWarps][32][33];
    __shared__ float R[6];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.y;
    if (threadIdx.x < 6) R[threadIdx.x] = rot[6 * r + threadIdx.x];
    __syncthreads();
    const int base = (blockIdx.x * kWarps + warp) * 32;
    if (base >= g.n_pix) return;
    const int p = base + lane;
    Collect collect(g, s_offset[warp][lane], s_weight[warp][lane]);
    if (p < g.n_pix) pixel_targets(g, R, pix_idx[p], collect);
    s_count[warp][lane] = collect.n;
    const int channels = n_real + 2 * n_cplx;
    const long stride = (long)g.n_rot * g.n_pix;
    const long at = (long)r * g.n_pix + p;
    const long V = (long)g.N0 * g.N1 * g.N2_eff;
    for (int group = 0; group * 32 < channels; group++) {
        const int width = min(32, channels - group * 32);
        // Channel values of the warp's pixels, one coalesced load per channel.
        for (int k = 0; k < width; k++) {
            const int c = group * 32 + k;
            float v = 0.f;
            if (p < g.n_pix) {
                if (c < n_real) {
                    v = real_images[c * stride + at];
                } else {
                    const float2 z = cplx_images[((c - n_real) >> 1) * stride + at];
                    v = ((c - n_real) & 1) ? z.y : z.x;
                }
            }
            s_value[warp][lane][k] = v;
        }
        __syncwarp();
        const int c = group * 32 + lane;
        const bool active = lane < width;
        const bool imag = active && c >= n_real && ((c - n_real) & 1);
        float* vol = out + group * V * 32;
        for (int j = 0; j < 32; j++) {
            const float v = active ? s_value[warp][j][lane] : 0.f;
            if (__all_sync(0xffffffffu, v == 0.f)) continue;  // adds nothing
            const int n = s_count[warp][j];
            for (int t = 0; t < n; t++) {
                const int o = s_offset[warp][j][t];
                const float x = s_weight[warp][j][t] * ((imag && o < 0) ? -v : v);
                if (active) atomicAdd(&vol[(long)(o & 0x7fffffff) * 32 + lane], x);
            }
        }
        __syncwarp();
    }
}

}  // namespace ppca_moment_bp

ffi::Error PpcaMomentBackprojectF32Impl(cudaStream_t stream, int64_t image_h, int64_t image_w, int64_t full_image_w,
                                        int64_t N0, int64_t N1, int64_t N2, int64_t max_r2_x4,
                                        ffi::AnyBuffer real_images, ffi::AnyBuffer cplx_images,
                                        ffi::AnyBuffer pixel_indices, ffi::AnyBuffer rot,
                                        ffi::Result<ffi::AnyBuffer> out) {
    namespace m = ppca_moment_bp;
    if (real_images.element_type() != ffi::DataType::F32 || cplx_images.element_type() != ffi::DataType::C64 ||
        pixel_indices.element_type() != ffi::DataType::S32 || rot.element_type() != ffi::DataType::F32 ||
        out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "PpcaMomentBackprojectF32: expected F32 real images, C64 complex images, S32 indices, F32 rotations");
    const auto rd = real_images.dimensions(), cd = cplx_images.dimensions(), pd = pixel_indices.dimensions();
    const auto qd = rot.dimensions(), od = out->dimensions();
    if (rd.size() != 3 || cd.size() != 3 || pd.size() != 1 || qd.size() != 2 || od.size() != 3 || qd[1] != 6 ||
        rd[1] != qd[0] || cd[1] != qd[0] || rd[2] != pd[0] || cd[2] != pd[0] || od[2] != 32)
        return ffi::Error::InvalidArgument(
            "PpcaMomentBackprojectF32: expected real (C_r,R,F), complex (C_c,R,F), indices (F,), rotations (R,6), "
            "out (G,V,32)");
    const int64_t channels = rd[0] + 2 * cd[0];
    const int64_t V = N0 * N1 * (N2 / 2 + 1);
    if (od[0] != (channels + 31) / 32 || od[1] != V)
        return ffi::Error::InvalidArgument("PpcaMomentBackprojectF32: output groups or volume size mismatch");
    if (V * 32 >= (int64_t(1) << 31) || N0 * N1 * (N2 / 2 + 1) >= (int64_t(1) << 31))
        return ffi::Error::InvalidArgument("PpcaMomentBackprojectF32: volume too large for 31-bit voxel offsets");
    m::Geometry g;
    g.n_rot = (int)qd[0];
    g.n_pix = (int)pd[0];
    g.image_h = (int)image_h;
    g.image_w = (int)image_w;
    g.full_image_w = (int)full_image_w;
    g.N0 = (int)N0;
    g.N1 = (int)N1;
    g.N2_full = (int)N2;
    g.N2_eff = (int)(N2 / 2 + 1);
    g.c0 = (float)(N0 / 2);
    g.c1 = (float)(N1 / 2);
    g.c2 = (float)(N2 / 2);
    g.max_r2 = max_r2_x4 >= 0 ? (float)max_r2_x4 / 4.f : -1.f;
    cudaError_t err = cudaMemsetAsync(out->untyped_data(), 0, od[0] * od[1] * od[2] * sizeof(float), stream);
    if (err == cudaSuccess && g.n_rot > 0 && g.n_pix > 0 && channels > 0) {
        dim3 grid((g.n_pix + 32 * m::kWarps - 1) / (32 * m::kWarps), g.n_rot);
        m::moment_backproject_kernel<<<grid, 32 * m::kWarps, 0, stream>>>(
            g, static_cast<const float*>(rot.untyped_data()), static_cast<const int*>(pixel_indices.untyped_data()),
            static_cast<const float*>(real_images.untyped_data()),
            static_cast<const float2*>(cplx_images.untyped_data()), (int)rd[0], (int)cd[0],
            static_cast<float*>(out->untyped_data()));
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PpcaMomentBackprojectF32, PpcaMomentBackprojectF32Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("image_h")
                                  .Attr<int64_t>("image_w")
                                  .Attr<int64_t>("full_image_w")
                                  .Attr<int64_t>("N0")
                                  .Attr<int64_t>("N1")
                                  .Attr<int64_t>("N2")
                                  .Attr<int64_t>("max_r2_x4")
                                  .Arg<ffi::AnyBuffer>()   /* real images    */
                                  .Arg<ffi::AnyBuffer>()   /* complex images */
                                  .Arg<ffi::AnyBuffer>()   /* pixel indices  */
                                  .Arg<ffi::AnyBuffer>()   /* rotations      */
                                  .Ret<ffi::AnyBuffer>()); /* (G, V, 32)     */
