// PPCA M-step residuals and moment backprojection into a voxel-major accumulator.
//
// For one rotation block of the streamed PPCA engine, forms the expected residual images
// R_p - sum_q L_pq A_q and the noise-power correction from the LHS images L, RHS images R and
// projections A (residual_statistics_from_moment_images), and scatters the metric channels and the
// residual into a half volume with the targets and weights of recovar's
// batch_backproject_indexed_kernel<float, 1, HALF_VOL, HALF_IMG> (linear interpolation,
// RELION-centred half volume and half image, no x-fold, unit upsampling): every windowed
// half-image pixel scatters trilinearly at its rotated frequency and, off the self-conjugate
// columns, at its Hermitian partner; targets with kz < 0 fold to their Hermitian partner.
//
// The accumulator is (groups, V, 32): channel c lives in lane c % 32 of group c / 32 (the metric
// channels, then the real and imaginary parts of each residual channel). One warp handles 32
// pixels of one rotation: each lane first expands its own pixel's targets and channel values,
// then for every pixel the warp adds one channel per lane, so each target is one warp-wide atomic
// to one 128-byte voxel row instead of one scattered atomic per channel. Only float32 association
// orders differ from the XLA residual statistics and recovar's channel-major atomics.
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

__device__ __forceinline__ int packed_index(int i, int j, int P) {  // np.triu_indices(P) order, i <= j
    return i * P - (i * (i - 1)) / 2 + (j - i);
}

// grid: (ceil(F / (32 kWarps)), R). lhs (K, R, F) packed upper LHS images; rhs (P, R, 2F) [Re | Im] RHS
// images; proj (P, R, 2F) [Re | Im] projections on the same window. Per pixel the
// residual R_p - sum_q L_pq A_q and the noise-power correction
// sum_pq L_pq Re(conj A_p A_q) - 2 sum_p Re(R_p conj A_p) are formed in registers; the correction is
// summed over rotations into correction (F,), and the metric channels (the K packed LHS images, or their
// trace) and the residual's real/imaginary parts are scattered into out (G, V, 32). out and correction
// are zero on entry.
__global__ void __launch_bounds__(32 * kWarps) moment_scatter_kernel(
    Geometry g, const float* __restrict__ rot, const int* __restrict__ pix_idx, const float* __restrict__ lhs,
    const float* __restrict__ rhs, const float* __restrict__ proj, int P, int metric_trace,
    float* __restrict__ out, float* __restrict__ correction) {
    extern __shared__ float s_value[];  // [kWarps][32][channels]
    __shared__ int s_offset[kWarps][32][kMaxTargets];
    __shared__ float s_weight[kWarps][32][kMaxTargets];
    __shared__ int s_count[kWarps][32];
    __shared__ float R[6];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.y;
    if (threadIdx.x < 6) R[threadIdx.x] = rot[6 * r + threadIdx.x];
    __syncthreads();
    const int base = (blockIdx.x * kWarps + warp) * 32;
    if (base >= g.n_pix) return;
    const int f = base + lane;
    const int K = P * (P + 1) / 2;
    const int metric = metric_trace ? 1 : K;
    const int channels = metric + 2 * P;
    float* value = s_value + (warp * 32 + lane) * channels;
    Collect collect(g, s_offset[warp][lane], s_weight[warp][lane]);
    if (f < g.n_pix && pix_idx[f] >= 0) {  // a negative index is padding: no channels, no targets
        pixel_targets(g, R, pix_idx[f], collect);
        const long F = g.n_pix;
        const long at = (long)r * F + f;
        const long lhs_stride = (long)g.n_rot * F;
        const float* L = lhs + at;
        float trace = 0.f, power = 0.f, cross = 0.f;
        for (int i = 0; i < P; i++) {
            const long Arow_i = ((long)i * g.n_rot + r) * 2 * F;
            const float2 Ai = make_float2(proj[Arow_i + f], proj[Arow_i + F + f]);
            const float Rre = rhs[((long)i * g.n_rot + r) * 2 * F + f];
            const float Rim = rhs[((long)i * g.n_rot + r) * 2 * F + F + f];
            float pre = 0.f, pim = 0.f;
            for (int j = 0; j < P; j++) {
                const long Arow_j = ((long)j * g.n_rot + r) * 2 * F;
                const float2 Aj = make_float2(proj[Arow_j + f], proj[Arow_j + F + f]);
                const float Lij = L[packed_index(min(i, j), max(i, j), P) * lhs_stride];
                pre += Lij * Aj.x;
                pim += Lij * Aj.y;
                if (j >= i) power += (j == i ? 1.f : 2.f) * Lij * (Ai.x * Aj.x + Ai.y * Aj.y);
            }
            value[metric + 2 * i] = Rre - pre;
            value[metric + 2 * i + 1] = Rim - pim;
            cross += Rre * Ai.x + Rim * Ai.y;
            const float Lii = L[packed_index(i, i, P) * lhs_stride];
            trace += Lii;
        }
        if (metric_trace) value[0] = trace;
        else
            for (int k = 0; k < K; k++) value[k] = L[k * lhs_stride];
        atomicAdd(&correction[f], power - 2.f * cross);
    } else {
        for (int c = 0; c < channels; c++) value[c] = 0.f;
    }
    s_count[warp][lane] = collect.n;
    __syncwarp();
    const long V = (long)g.N0 * g.N1 * g.N2_eff;
    const int first_imag = metric + 1;  // residual parts alternate re, im
    for (int group = 0; group * 32 < channels; group++) {
        const int c = group * 32 + lane;
        const bool active = c < channels;
        const bool imag = active && c >= metric && ((c - first_imag) & 1) == 0;
        float* vol = out + group * V * 32;
        for (int j = 0; j < 32; j++) {
            const float v = active ? s_value[(warp * 32 + j) * channels + c] : 0.f;
            if (__all_sync(0xffffffffu, v == 0.f)) continue;  // adds nothing
            const int n = s_count[warp][j];
            for (int t = 0; t < n; t++) {
                const int o = s_offset[warp][j][t];
                const float x = s_weight[warp][j][t] * ((imag && o < 0) ? -v : v);
                if (active) atomicAdd(&vol[(long)(o & 0x7fffffff) * 32 + lane], x);
            }
        }
    }
}

}  // namespace ppca_moment_bp

ffi::Error PpcaMomentScatterF32Impl(cudaStream_t stream, int64_t image_h, int64_t image_w, int64_t full_image_w,
                                    int64_t N0, int64_t N1, int64_t N2, int64_t max_r2_x4, int64_t metric_trace,
                                    ffi::AnyBuffer lhs, ffi::AnyBuffer rhs, ffi::AnyBuffer proj,
                                    ffi::AnyBuffer pixel_indices, ffi::AnyBuffer rot,
                                    ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> correction) {
    namespace m = ppca_moment_bp;
    if (lhs.element_type() != ffi::DataType::F32 || rhs.element_type() != ffi::DataType::F32 ||
        proj.element_type() != ffi::DataType::F32 || pixel_indices.element_type() != ffi::DataType::S32 ||
        rot.element_type() != ffi::DataType::F32 || out->element_type() != ffi::DataType::F32 ||
        correction->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "PpcaMomentScatterF32: expected F32 lhs/rhs/projections, S32 indices, F32 rotations and outputs");
    const auto ld = lhs.dimensions(), rd = rhs.dimensions(), pd = proj.dimensions();
    const auto id = pixel_indices.dimensions(), qd = rot.dimensions(), od = out->dimensions();
    const auto cd = correction->dimensions();
    if (ld.size() != 3 || rd.size() != 3 || pd.size() != 3 || id.size() != 1 || qd.size() != 2 || od.size() != 3 ||
        cd.size() != 1)
        return ffi::Error::InvalidArgument("PpcaMomentScatterF32: operand rank mismatch");
    const int64_t P = rd[0], R = qd[0], F = id[0];
    if (ld[0] != P * (P + 1) / 2 || ld[1] != R || ld[2] != F || rd[1] != R || rd[2] != 2 * F || pd[0] != P ||
        pd[1] != R || pd[2] != 2 * F || qd[1] != 6 || od[2] != 32 || cd[0] != F)
        return ffi::Error::InvalidArgument(
            "PpcaMomentScatterF32: expected lhs (tri(P),R,F), rhs (P,R,2F), proj (P,R,2F), indices (F,), "
            "rotations (R,6), out (G,V,32), correction (F,)");
    const int64_t channels = (metric_trace ? 1 : P * (P + 1) / 2) + 2 * P;
    const int64_t V = N0 * N1 * (N2 / 2 + 1);
    if (od[0] != (channels + 31) / 32 || od[1] != V)
        return ffi::Error::InvalidArgument("PpcaMomentScatterF32: output groups or volume size mismatch");
    if (V >= (int64_t(1) << 31) / 32)
        return ffi::Error::InvalidArgument("PpcaMomentScatterF32: volume too large for 31-bit voxel offsets");
    m::Geometry g;
    g.n_rot = (int)R;
    g.n_pix = (int)F;
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
    const size_t smem = (size_t)m::kWarps * 32 * channels * sizeof(float);
    cudaError_t err = cudaMemsetAsync(out->untyped_data(), 0, od[0] * od[1] * od[2] * sizeof(float), stream);
    if (err == cudaSuccess) err = cudaMemsetAsync(correction->untyped_data(), 0, F * sizeof(float), stream);
    // The static target tables add to the per-pixel channel values: opt in whenever any is dynamic.
    if (err == cudaSuccess)
        err = cudaFuncSetAttribute(m::moment_scatter_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err == cudaSuccess && R > 0 && F > 0) {
        dim3 grid((g.n_pix + 32 * m::kWarps - 1) / (32 * m::kWarps), g.n_rot);
        m::moment_scatter_kernel<<<grid, 32 * m::kWarps, smem, stream>>>(
            g, static_cast<const float*>(rot.untyped_data()), static_cast<const int*>(pixel_indices.untyped_data()),
            static_cast<const float*>(lhs.untyped_data()), static_cast<const float*>(rhs.untyped_data()),
            static_cast<const float*>(proj.untyped_data()), (int)P, (int)metric_trace,
            static_cast<float*>(out->untyped_data()), static_cast<float*>(correction->untyped_data()));
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PpcaMomentScatterF32, PpcaMomentScatterF32Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("image_h")
                                  .Attr<int64_t>("image_w")
                                  .Attr<int64_t>("full_image_w")
                                  .Attr<int64_t>("N0")
                                  .Attr<int64_t>("N1")
                                  .Attr<int64_t>("N2")
                                  .Attr<int64_t>("max_r2_x4")
                                  .Attr<int64_t>("metric_trace")
                                  .Arg<ffi::AnyBuffer>()   /* lhs (tri(P), R, F)     */
                                  .Arg<ffi::AnyBuffer>()   /* rhs (P, R, 2F)         */
                                  .Arg<ffi::AnyBuffer>()   /* proj (P, R, 2F)        */
                                  .Arg<ffi::AnyBuffer>()   /* pixel indices (F,)     */
                                  .Arg<ffi::AnyBuffer>()   /* rotations (R, 6)       */
                                  .Ret<ffi::AnyBuffer>()   /* moments (G, V, 32)     */
                                  .Ret<ffi::AnyBuffer>()); /* correction (F,)        */
