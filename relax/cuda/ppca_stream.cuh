// Fused elementwise stages of the streamed PPCA engine (relax/ppca_refinement/full_row_stream.py).
//
// 1. PpcaWindowProjectF32: the augmented projections of one rotation block on the score window,
//    written in the planar [Re | Im] layout the pass-1 score GEMM reads, and the products
//    Re(conj(A_i) A_j) of every packed pair the latent Gram GEMM reads. Each pixel is recovar's
//    project_kernel<float, 1, HALF_VOL, HALF_IMG> (trilinear, RELION-centred half volume, Hermitian
//    partner reads) with the trilinear targets formed once for all P components of a voxel-major
//    (V, P) volume.
// 2. PpcaLatentEpilogueF32: from the GEMM inner products and Gram, the pose scores, latent means and
//    latent covariances of one rotation block (Cholesky of I + H_zz, section 4 of
//    docs/math/vdam_ppca_algorithm.md), written in place into the tile's kept buffers, and the
//    per-(rotation, image) maximum, first maximizing translation and sum of exp(score - maximum)
//    that the tile normalization reduces.
// 3. PpcaPosteriorPrepF32: from the kept results and the tile normalization, the posterior weights
//    gamma [1, E z] of every pose (the pass-2 RHS GEMM operand), the translation sums of
//    gamma E[[1, z][1, z]^T] (the LHS GEMM operand) and per-(rotation, image) partial diagnostics.
//
// Stages 2 and 3 run one warp per (rotation, image) with the translations across the lanes.
#pragma once

namespace ppca_stream {

constexpr int kMaxLatent = 16;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}

__device__ __forceinline__ int warp_min(int x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}

// ---------------------------------------------------------------------------------------------
// 1. Windowed projection
// ---------------------------------------------------------------------------------------------

struct ProjectGeometry {
    int n_rot, n_pix, image_h, image_w, full_image_w;
    int N0, N1, N2_eff, N2_full;
    float c0, c1, c2, max_r2;
};

// grid (ceil(F / 128), R), 128 threads; dynamic shared memory 128 * P float2.
// vol (V, P) float2 voxel-major; rot (R, 6); pix (F,) windowed half-image pixels.
// planar (P, R, 2F): [Re | Im] per component and rotation. products (K, R, F) or null.
__global__ void __launch_bounds__(128) window_project_kernel(ProjectGeometry g, const float2* __restrict__ vol,
                                                             const float* __restrict__ rot,
                                                             const int* __restrict__ pix, int P,
                                                             float* __restrict__ planar,
                                                             float* __restrict__ products) {
    extern __shared__ float2 s_value[];  // [128][P]
    __shared__ float R[6];
    const int r = blockIdx.y;
    if (threadIdx.x < 6) R[threadIdx.x] = rot[6 * r + threadIdx.x];
    __syncthreads();
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= g.n_pix) return;
    const long F = g.n_pix;
    float2* value = s_value + threadIdx.x * P;
    const int orig_pix = pix[f];
    const int k0_idx = orig_pix / g.image_w, k1_idx = orig_pix % g.image_w;
    const float k0 = (float)(k0_idx - g.image_h / 2);
    const float k1 = (k1_idx * 2 == g.full_image_w) ? (float)(-k1_idx) : (float)k1_idx;
    bool zero = g.max_r2 >= 0.f && k0 * k0 + k1 * k1 > g.max_r2;
    // Fixed slots (d0, d1, d2) keep the targets in registers; skipped neighbours carry weight 0,
    // and adding 0 * v leaves each sum exactly as recovar's skip does.
    int offset[8];
    float weight[8];
    bool conj[8];
#pragma unroll
    for (int t = 0; t < 8; t++) {
        offset[t] = 0;
        weight[t] = 0.f;
        conj[t] = false;
    }
    if (!zero) {
        const float rk0 = k0 * R[0] + k1 * R[3], rk1 = k0 * R[1] + k1 * R[4], rk2 = k0 * R[2] + k1 * R[5];
        const float g0 = rk0 + g.c0, g1 = rk1 + g.c1, g2 = rk2 + g.c2;
        const int ic2 = (int)g.c2;
        if (g0 < -1.f || g0 >= (float)g.N0 || g1 < -1.f || g1 >= (float)g.N1 || g2 < -1.f ||
            g2 >= (float)g.N2_full) {
            zero = true;
        } else {
            const int b0 = (int)floorf(g0), b1 = (int)floorf(g1), b2 = (int)floorf(g2);
            const float f0 = g0 - (float)b0, f1 = g1 - (float)b1, f2 = g2 - (float)b2;
            const float w0[2] = {1.f - f0, f0}, w1[2] = {1.f - f1, f1}, w2[2] = {1.f - f2, f2};
            const int stride1 = g.N2_eff, stride0 = g.N1 * g.N2_eff;
#pragma unroll
            for (int d0 = 0; d0 < 2; d0++) {
                const int j0 = b0 + d0;
#pragma unroll
                for (int d1 = 0; d1 < 2; d1++) {
                    const int j1 = b1 + d1;
                    const float ww = w0[d0] * w1[d1];
#pragma unroll
                    for (int d2 = 0; d2 < 2; d2++) {
                        const int j2 = b2 + d2;
                        const int slot = 4 * d0 + 2 * d1 + d2;
                        if ((unsigned)j0 >= (unsigned)g.N0 || (unsigned)j1 >= (unsigned)g.N1 ||
                            (unsigned)j2 >= (unsigned)g.N2_full)
                            continue;
                        const int kz = j2 - ic2;
                        int ri = j0, rj = j1, rk = kz;
                        if (kz < 0) {
                            // Hermitian partner: (N - (N & 1) - j) % N.
                            ri = (g.N0 - (g.N0 & 1) - j0) % g.N0;
                            rj = (g.N1 - (g.N1 & 1) - j1) % g.N1;
                            rk = -kz;
                            conj[slot] = true;
                        }
                        offset[slot] = ri * stride0 + rj * stride1 + rk;
                        weight[slot] = ww * w2[d2];
                    }
                }
            }
        }
    }
    for (int p = 0; p < P; p++) {
        float re = 0.f, im = 0.f;
        if (!zero) {
#pragma unroll
            for (int t = 0; t < 8; t++) {
                float2 v = __ldg(&vol[(long)offset[t] * P + p]);
                if (conj[t]) v.y = -v.y;
                re += weight[t] * v.x;
                im += weight[t] * v.y;
            }
        }
        const long row = ((long)p * g.n_rot + r) * 2 * F;
        planar[row + f] = re;
        planar[row + F + f] = im;
        value[p] = make_float2(re, im);
    }
    if (products == nullptr) return;
    int k = 0;
    for (int i = 0; i < P; i++) {
        const float2 a = value[i];
        for (int j = i; j < P; j++, k++) {
            const float2 b = value[j];
            products[((long)k * g.n_rot + r) * F + f] = a.x * b.x + a.y * b.y;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Latent epilogue of pass 1
// ---------------------------------------------------------------------------------------------

struct PriorTables {
    const int* rows;                // (R,) fine rows of this block (row-table slice)
    const int* rotation_parent;     // (n_rows + 1,)
    const float* rotation_log_prior;
    const int* translation_parent;  // (T,)
    const float* translation_log_prior;
    const bool* coarse_mask;        // (B, Rc + 1, Tc)
    int coarse_rotations_plus_one, coarse_translations;
};

__device__ __forceinline__ int packed(int i, int j, int n) {  // np.triu_indices(n) order, i <= j
    return i * n - (i * (i - 1)) / 2 + (j - i);
}

// grid ceil(R * B / 4), 128 threads: warp w handles pair (r, b) = divmod(blockIdx.x * 4 + w, B).
// inner (P, R, B, T); gram (K, R, B) packed upper over P; start (1,) device row offset of the block.
// score (cap, B, T), mean (Q, cap, B, T), cov (cap, B, tri(Q)), part_max / part_sum (cap, B),
// part_arg (cap, B) are written at rows start .. start + R.
template <int Q>
__global__ void __launch_bounds__(128) latent_epilogue_kernel(
    int R, int B, int T, int cap, const float* __restrict__ inner, const float* __restrict__ gram,
    PriorTables prior, const int* __restrict__ start_ptr, float* __restrict__ score, float* __restrict__ mean,
    float* __restrict__ cov, float* __restrict__ part_max, int* __restrict__ part_arg,
    float* __restrict__ part_sum) {
    constexpr int P = Q + 1;
    constexpr int TQ = Q * (Q + 1) / 2;
    const int lane = threadIdx.x & 31;
    const long pair = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (pair >= (long)R * B) return;
    const int r = (int)(pair / B), b = (int)(pair % B);
    const int start = *start_ptr;
    const long RB = (long)R * B;
    auto G = [&](int i, int j) { return gram[(long)packed(i, j, P) * RB + pair]; };
    // Cholesky of I + H (H_ij = Gram of latent components i + 1, j + 1), as _unit_shift_cholesky.
    float L[Q > 0 ? Q : 1][Q > 0 ? Q : 1];
    float logdet = 0.f;
#pragma unroll
    for (int j = 0; j < Q; j++) {
        float s = 1.f + G(j + 1, j + 1);
#pragma unroll
        for (int k = 0; k < j; k++) s = s - L[j][k] * L[j][k];
        L[j][j] = sqrtf(s);
        logdet = logdet + logf(L[j][j]);
#pragma unroll
        for (int i = j + 1; i < Q; i++) {
            float s2 = G(j + 1, i + 1);
#pragma unroll
            for (int k = 0; k < j; k++) s2 = s2 - L[i][k] * L[j][k];
            L[i][j] = s2 / L[j][j];
        }
    }
    logdet = 2.f * logdet;
    const float g00 = G(0, 0);
    float g0[Q > 0 ? Q : 1];
#pragma unroll
    for (int j = 0; j < Q; j++) g0[j] = G(0, j + 1);
    const long out_pair = (long)(start + r) * B + b;
    if (Q > 0) {
        // Covariance (L^-T L^-1) packed upper, as _lower_inverse and the covariance stack.
        float inv[Q > 0 ? Q : 1][Q > 0 ? Q : 1];
#pragma unroll
        for (int j = 0; j < Q; j++) {
            inv[j][j] = 1.f / L[j][j];
#pragma unroll
            for (int i = j + 1; i < Q; i++) {
                float s = 0.f;
#pragma unroll
                for (int k = j; k < i; k++) s = s + L[i][k] * inv[k][j];
                inv[i][j] = -s / L[i][i];
            }
        }
        int k = 0;
#pragma unroll
        for (int i = 0; i < Q; i++) {
#pragma unroll
            for (int j = i; j < Q; j++, k++) {
                if (lane == (k & 31)) {
                    float s = 0.f;
#pragma unroll
                    for (int m = j; m < Q; m++) s = s + inv[m][i] * inv[m][j];
                    cov[out_pair * TQ + k] = s;
                }
            }
        }
    }
    const int row = prior.rows[r];
    const int coarse_row = prior.rotation_parent[row];
    const float row_prior = prior.rotation_log_prior[row];
    const bool* mask = prior.coarse_mask + ((long)b * prior.coarse_rotations_plus_one + coarse_row) *
                                               prior.coarse_translations;
    const long capB = (long)cap * B;
    float best = -INFINITY;
    int best_t = T;
    for (int t = lane; t < T; t += 32) {
        const long at = pair * T + t;
        const long stride = RB * T;
        const float i0 = inner[at];
        const float rho = g00 - 2.f * i0;
        const float pose_prior =
            mask[prior.translation_parent[t]] ? row_prior + prior.translation_log_prior[t] : -INFINITY;
        float s;
        if (Q == 0) {
            s = -0.5f * rho + pose_prior;
        } else {
            float v[Q > 0 ? Q : 1];
            float norm = 0.f;
#pragma unroll
            for (int i = 0; i < Q; i++) {
                float x = inner[at + (long)(i + 1) * stride] - g0[i];
#pragma unroll
                for (int k = 0; k < i; k++) x = x - L[i][k] * v[k];
                v[i] = x / L[i][i];
                norm = norm + v[i] * v[i];
            }
            s = -0.5f * (rho - norm + logdet) + pose_prior;
            float z[Q > 0 ? Q : 1];
#pragma unroll
            for (int i = Q - 1; i >= 0; i--) {
                float x = v[i];
#pragma unroll
                for (int k = i + 1; k < Q; k++) x = x - L[k][i] * z[k];
                z[i] = x / L[i][i];
                mean[((long)i * capB + out_pair) * T + t] = z[i];
            }
        }
        score[out_pair * T + t] = s;
        if (s > best) {
            best = s;
            best_t = t;
        }
    }
    const float m = warp_max(best);
    const int arg = warp_min(best == m ? best_t : T);
    float e = 0.f;
    if (m != -INFINITY)
        for (int t = lane; t < T; t += 32) e += expf(score[out_pair * T + t] - m);
    e = warp_sum(e);
    if (lane == 0) {
        part_max[out_pair] = m;
        part_arg[out_pair] = arg == T ? 0 : arg;
        part_sum[out_pair] = e;
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Posterior weights and moment sums of pass 2
// ---------------------------------------------------------------------------------------------

// grid ceil(R * B / 4), 128 threads. Reads kept rows start .. start + R; center, logZ (B,);
// shift2 (T,). weights (P, R, B, T): gamma, gamma E z_j. sums (K, R, B): sum_t gamma E[a_i a_j]
// for a = [1, z] packed upper. partial (2, R, B): -sum gamma log-gamma terms (pose entropy) and
// sum gamma |shift|^2; count (R, B): poses with gamma > 1e-3.
template <int Q>
__global__ void __launch_bounds__(128) posterior_prep_kernel(
    int R, int B, int T, int cap, const float* __restrict__ score, const float* __restrict__ mean,
    const float* __restrict__ cov, const float* __restrict__ center, const float* __restrict__ logZ,
    const float* __restrict__ shift2, const int* __restrict__ start_ptr, float* __restrict__ weights,
    float* __restrict__ sums, float* __restrict__ partial, int* __restrict__ count) {
    constexpr int P = Q + 1;
    constexpr int TQ = Q * (Q + 1) / 2;
    constexpr int K = P * (P + 1) / 2;
    const int lane = threadIdx.x & 31;
    const long pair = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (pair >= (long)R * B) return;
    const int b = (int)(pair % B);
    const int start = *start_ptr;
    const long RB = (long)R * B;
    const long out_pair = (long)start * B + pair;  // kept row start + r, image b
    const long capB = (long)cap * B;
    float c[TQ > 0 ? TQ : 1];
#pragma unroll
    for (int k = 0; k < TQ; k++) c[k] = cov[out_pair * TQ + k];
    const float cb = center[b], lb = logZ[b];
    float acc[K];
#pragma unroll
    for (int k = 0; k < K; k++) acc[k] = 0.f;
    float entropy = 0.f, offset = 0.f;
    int significant = 0;
    for (int t = lane; t < T; t += 32) {
        const float centered = (score[out_pair * T + t] - cb) - lb;
        const float gamma = expf(centered);
        float z[Q > 0 ? Q : 1];
#pragma unroll
        for (int i = 0; i < Q; i++) z[i] = mean[((long)i * capB + out_pair) * T + t];
        const long w = pair * T + t;
        weights[w] = gamma;
#pragma unroll
        for (int i = 0; i < Q; i++) weights[(long)(i + 1) * RB * T + w] = gamma * z[i];
        int k = 0;
#pragma unroll
        for (int i = 0; i < P; i++) {
#pragma unroll
            for (int j = i; j < P; j++, k++) {
                if (i == 0) acc[k] += j == 0 ? gamma : gamma * z[j - 1];
                else acc[k] += gamma * (c[packed(i - 1, j - 1, Q)] + z[i - 1] * z[j - 1]);
            }
        }
        if (gamma > 0.f) entropy += gamma * centered;
        offset += gamma * shift2[t];
        significant += gamma > 1e-3f;
    }
#pragma unroll
    for (int k = 0; k < K; k++) {
        const float s = warp_sum(acc[k]);
        if (lane == (k & 31)) sums[(long)k * RB + pair] = s;
    }
    entropy = warp_sum(entropy);
    offset = warp_sum(offset);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) significant += __shfl_xor_sync(0xffffffffu, significant, o);
    if (lane == 0) {
        partial[pair] = -entropy;
        partial[RB + pair] = offset;
        count[pair] = significant;
    }
}

}  // namespace ppca_stream

// ---------------------------------------------------------------------------------------------
// FFI handlers
// ---------------------------------------------------------------------------------------------

ffi::Error PpcaWindowProjectF32Impl(cudaStream_t stream, int64_t image_h, int64_t image_w, int64_t full_image_w,
                                    int64_t N0, int64_t N1, int64_t N2, int64_t max_r2_x4, int64_t with_products,
                                    ffi::AnyBuffer volume, ffi::AnyBuffer rot, ffi::AnyBuffer pixel_indices,
                                    ffi::Result<ffi::AnyBuffer> planar, ffi::Result<ffi::AnyBuffer> products) {
    namespace s = ppca_stream;
    if (volume.element_type() != ffi::DataType::C64 || rot.element_type() != ffi::DataType::F32 ||
        pixel_indices.element_type() != ffi::DataType::S32 || planar->element_type() != ffi::DataType::F32 ||
        products->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument("PpcaWindowProjectF32: expected C64 volume, F32 rotations, S32 indices");
    const auto vd = volume.dimensions(), qd = rot.dimensions(), id = pixel_indices.dimensions();
    const auto pd = planar->dimensions(), kd = products->dimensions();
    if (vd.size() != 2 || qd.size() != 2 || id.size() != 1 || pd.size() != 3 || qd[1] != 6)
        return ffi::Error::InvalidArgument("PpcaWindowProjectF32: operand rank mismatch");
    const int64_t V = vd[0], P = vd[1], R = qd[0], F = id[0], K = P * (P + 1) / 2;
    if (V != N0 * N1 * (N2 / 2 + 1) || pd[0] != P || pd[1] != R || pd[2] != 2 * F)
        return ffi::Error::InvalidArgument("PpcaWindowProjectF32: expected volume (V,P), planar (P,R,2F)");
    if (with_products ? (kd.size() != 3 || kd[0] != K || kd[1] != R || kd[2] != F) : (kd.size() != 1 || kd[0] != 0))
        return ffi::Error::InvalidArgument("PpcaWindowProjectF32: products must be (K,R,F), or (0,) without");
    s::ProjectGeometry g;
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
    if (R == 0 || F == 0) return ffi::Error::Success();
    const size_t smem = (size_t)128 * P * sizeof(float2);
    cudaError_t err = cudaFuncSetAttribute(s::window_project_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int)smem);
    if (err == cudaSuccess) {
        dim3 grid((unsigned)((F + 127) / 128), (unsigned)R);
        s::window_project_kernel<<<grid, 128, smem, stream>>>(
            g, static_cast<const float2*>(volume.untyped_data()), static_cast<const float*>(rot.untyped_data()),
            static_cast<const int*>(pixel_indices.untyped_data()), (int)P,
            static_cast<float*>(planar->untyped_data()),
            with_products ? static_cast<float*>(products->untyped_data()) : nullptr);
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PpcaWindowProjectF32, PpcaWindowProjectF32Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("image_h")
                                  .Attr<int64_t>("image_w")
                                  .Attr<int64_t>("full_image_w")
                                  .Attr<int64_t>("N0")
                                  .Attr<int64_t>("N1")
                                  .Attr<int64_t>("N2")
                                  .Attr<int64_t>("max_r2_x4")
                                  .Attr<int64_t>("with_products")
                                  .Arg<ffi::AnyBuffer>()   /* volume (V, P) C64 voxel-major */
                                  .Arg<ffi::AnyBuffer>()   /* rotations (R, 6)            */
                                  .Arg<ffi::AnyBuffer>()   /* pixel indices (F,)          */
                                  .Ret<ffi::AnyBuffer>()   /* planar (P, R, 2F)           */
                                  .Ret<ffi::AnyBuffer>()); /* products (K, R, F) or (0,)  */

namespace ppca_stream {

template <template <int> class Launch, typename... Args>
cudaError_t dispatch_latent(int q, Args... args) {
    switch (q) {
#define PPCA_STREAM_CASE(N) \
    case N:                 \
        return Launch<N>::run(args...);
        PPCA_STREAM_CASE(0) PPCA_STREAM_CASE(1) PPCA_STREAM_CASE(2) PPCA_STREAM_CASE(3) PPCA_STREAM_CASE(4)
        PPCA_STREAM_CASE(5) PPCA_STREAM_CASE(6) PPCA_STREAM_CASE(7) PPCA_STREAM_CASE(8) PPCA_STREAM_CASE(9)
        PPCA_STREAM_CASE(10) PPCA_STREAM_CASE(11) PPCA_STREAM_CASE(12) PPCA_STREAM_CASE(13) PPCA_STREAM_CASE(14)
        PPCA_STREAM_CASE(15) PPCA_STREAM_CASE(16)
#undef PPCA_STREAM_CASE
        default:
            return cudaErrorInvalidValue;
    }
}

template <int Q>
struct LaunchEpilogue {
    static cudaError_t run(cudaStream_t stream, int R, int B, int T, int cap, const float* inner, const float* gram,
                           PriorTables prior, const int* start, float* score, float* mean, float* cov,
                           float* part_max, int* part_arg, float* part_sum) {
        const long pairs = (long)R * B;
        latent_epilogue_kernel<Q><<<(unsigned)((pairs + 3) / 4), 128, 0, stream>>>(
            R, B, T, cap, inner, gram, prior, start, score, mean, cov, part_max, part_arg, part_sum);
        return cudaGetLastError();
    }
};

template <int Q>
struct LaunchPrep {
    static cudaError_t run(cudaStream_t stream, int R, int B, int T, int cap, const float* score, const float* mean,
                           const float* cov, const float* center, const float* logZ, const float* shift2,
                           const int* start, float* weights, float* sums, float* partial, int* count) {
        const long pairs = (long)R * B;
        posterior_prep_kernel<Q><<<(unsigned)((pairs + 3) / 4), 128, 0, stream>>>(
            R, B, T, cap, score, mean, cov, center, logZ, shift2, start, weights, sums, partial, count);
        return cudaGetLastError();
    }
};

}  // namespace ppca_stream

// Kept buffers (score, mean, cov, part_max, part_arg, part_sum) are inputs aliased to the outputs.
ffi::Error PpcaLatentEpilogueF32Impl(cudaStream_t stream, ffi::AnyBuffer inner, ffi::AnyBuffer gram,
                                     ffi::AnyBuffer rows, ffi::AnyBuffer rotation_parent,
                                     ffi::AnyBuffer rotation_log_prior, ffi::AnyBuffer translation_parent,
                                     ffi::AnyBuffer translation_log_prior, ffi::AnyBuffer coarse_mask,
                                     ffi::AnyBuffer start, ffi::AnyBuffer, ffi::AnyBuffer, ffi::AnyBuffer,
                                     ffi::AnyBuffer, ffi::AnyBuffer, ffi::AnyBuffer,
                                     ffi::Result<ffi::AnyBuffer> score, ffi::Result<ffi::AnyBuffer> mean,
                                     ffi::Result<ffi::AnyBuffer> cov, ffi::Result<ffi::AnyBuffer> part_max,
                                     ffi::Result<ffi::AnyBuffer> part_arg, ffi::Result<ffi::AnyBuffer> part_sum) {
    namespace s = ppca_stream;
    using DT = ffi::DataType;
    if (inner.element_type() != DT::F32 || gram.element_type() != DT::F32 || rows.element_type() != DT::S32 ||
        rotation_parent.element_type() != DT::S32 || rotation_log_prior.element_type() != DT::F32 ||
        translation_parent.element_type() != DT::S32 || translation_log_prior.element_type() != DT::F32 ||
        coarse_mask.element_type() != DT::PRED || start.element_type() != DT::S32 ||
        score->element_type() != DT::F32 || mean->element_type() != DT::F32 || cov->element_type() != DT::F32 ||
        part_max->element_type() != DT::F32 || part_arg->element_type() != DT::S32 ||
        part_sum->element_type() != DT::F32)
        return ffi::Error::InvalidArgument("PpcaLatentEpilogueF32: operand dtype mismatch");
    const auto nd = inner.dimensions(), gd = gram.dimensions(), sd = score->dimensions();
    const auto md = mean->dimensions(), cd = cov->dimensions(), xd = part_max->dimensions();
    const auto cm = coarse_mask.dimensions();
    if (nd.size() != 4 || gd.size() != 3 || sd.size() != 3 || md.size() != 4 || cd.size() != 3 || xd.size() != 2 ||
        cm.size() != 3)
        return ffi::Error::InvalidArgument("PpcaLatentEpilogueF32: operand rank mismatch");
    const int64_t P = nd[0], R = nd[1], B = nd[2], T = nd[3], Q = P - 1, cap = sd[0];
    if (Q > s::kMaxLatent || gd[0] != P * (P + 1) / 2 || gd[1] != R || gd[2] != B || sd[1] != B || sd[2] != T ||
        md[0] != Q || md[1] != cap || md[2] != B || md[3] != T || cd[0] != cap || cd[1] != B ||
        cd[2] != Q * (Q + 1) / 2 || xd[0] != cap || xd[1] != B || rows.dimensions()[0] != R || cm[0] != B ||
        translation_parent.dimensions()[0] != T || translation_log_prior.dimensions()[0] != T)
        return ffi::Error::InvalidArgument("PpcaLatentEpilogueF32: shape mismatch (or latent rank above 16)");
    if (R == 0 || B == 0 || T == 0) return ffi::Error::Success();
    s::PriorTables prior{static_cast<const int*>(rows.untyped_data()),
                         static_cast<const int*>(rotation_parent.untyped_data()),
                         static_cast<const float*>(rotation_log_prior.untyped_data()),
                         static_cast<const int*>(translation_parent.untyped_data()),
                         static_cast<const float*>(translation_log_prior.untyped_data()),
                         static_cast<const bool*>(coarse_mask.untyped_data()), (int)cm[1], (int)cm[2]};
    cudaError_t err = s::dispatch_latent<s::LaunchEpilogue>(
        (int)Q, stream, (int)R, (int)B, (int)T, (int)cap, static_cast<const float*>(inner.untyped_data()),
        static_cast<const float*>(gram.untyped_data()), prior, static_cast<const int*>(start.untyped_data()),
        static_cast<float*>(score->untyped_data()), static_cast<float*>(mean->untyped_data()),
        static_cast<float*>(cov->untyped_data()), static_cast<float*>(part_max->untyped_data()),
        static_cast<int*>(part_arg->untyped_data()), static_cast<float*>(part_sum->untyped_data()));
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PpcaLatentEpilogueF32, PpcaLatentEpilogueF32Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()   /* inner (P, R, B, T)            */
                                  .Arg<ffi::AnyBuffer>()   /* gram (K, R, B)                */
                                  .Arg<ffi::AnyBuffer>()   /* rows (R,)                     */
                                  .Arg<ffi::AnyBuffer>()   /* rotation parent (n + 1,)      */
                                  .Arg<ffi::AnyBuffer>()   /* rotation log prior (n + 1,)   */
                                  .Arg<ffi::AnyBuffer>()   /* translation parent (T,)       */
                                  .Arg<ffi::AnyBuffer>()   /* translation log prior (T,)    */
                                  .Arg<ffi::AnyBuffer>()   /* coarse mask (B, Rc + 1, Tc)   */
                                  .Arg<ffi::AnyBuffer>()   /* start (1,)                    */
                                  .Arg<ffi::AnyBuffer>()   /* score (aliased)               */
                                  .Arg<ffi::AnyBuffer>()   /* mean (aliased)                */
                                  .Arg<ffi::AnyBuffer>()   /* cov (aliased)                 */
                                  .Arg<ffi::AnyBuffer>()   /* part_max (aliased)            */
                                  .Arg<ffi::AnyBuffer>()   /* part_arg (aliased)            */
                                  .Arg<ffi::AnyBuffer>()   /* part_sum (aliased)            */
                                  .Ret<ffi::AnyBuffer>()   /* score (cap, B, T)             */
                                  .Ret<ffi::AnyBuffer>()   /* mean (Q, cap, B, T)           */
                                  .Ret<ffi::AnyBuffer>()   /* cov (cap, B, tri(Q))          */
                                  .Ret<ffi::AnyBuffer>()   /* part_max (cap, B)             */
                                  .Ret<ffi::AnyBuffer>()   /* part_arg (cap, B)             */
                                  .Ret<ffi::AnyBuffer>()); /* part_sum (cap, B)             */

ffi::Error PpcaPosteriorPrepF32Impl(cudaStream_t stream, int64_t block_size, ffi::AnyBuffer score,
                                    ffi::AnyBuffer mean, ffi::AnyBuffer cov, ffi::AnyBuffer center,
                                    ffi::AnyBuffer logZ, ffi::AnyBuffer shift2, ffi::AnyBuffer start,
                                    ffi::Result<ffi::AnyBuffer> weights, ffi::Result<ffi::AnyBuffer> sums,
                                    ffi::Result<ffi::AnyBuffer> partial, ffi::Result<ffi::AnyBuffer> count) {
    namespace s = ppca_stream;
    using DT = ffi::DataType;
    if (score.element_type() != DT::F32 || mean.element_type() != DT::F32 || cov.element_type() != DT::F32 ||
        center.element_type() != DT::F32 || logZ.element_type() != DT::F32 || shift2.element_type() != DT::F32 ||
        start.element_type() != DT::S32 || weights->element_type() != DT::F32 || sums->element_type() != DT::F32 ||
        partial->element_type() != DT::F32 || count->element_type() != DT::S32)
        return ffi::Error::InvalidArgument("PpcaPosteriorPrepF32: operand dtype mismatch");
    const auto sd = score.dimensions(), md = mean.dimensions(), wd = weights->dimensions();
    const auto kd = sums->dimensions();
    if (sd.size() != 3 || md.size() != 4 || wd.size() != 4 || kd.size() != 3)
        return ffi::Error::InvalidArgument("PpcaPosteriorPrepF32: operand rank mismatch");
    const int64_t cap = sd[0], B = sd[1], T = sd[2], Q = md[0], P = Q + 1, R = block_size;
    if (Q > s::kMaxLatent || md[1] != cap || md[2] != B || md[3] != T || wd[0] != P || wd[1] != R || wd[2] != B ||
        wd[3] != T || kd[0] != P * (P + 1) / 2 || kd[1] != R || kd[2] != B || center.dimensions()[0] != B ||
        logZ.dimensions()[0] != B || shift2.dimensions()[0] != T)
        return ffi::Error::InvalidArgument("PpcaPosteriorPrepF32: shape mismatch (or latent rank above 16)");
    if (R == 0 || B == 0 || T == 0) return ffi::Error::Success();
    cudaError_t err = s::dispatch_latent<s::LaunchPrep>(
        (int)Q, stream, (int)R, (int)B, (int)T, (int)cap, static_cast<const float*>(score.untyped_data()),
        static_cast<const float*>(mean.untyped_data()), static_cast<const float*>(cov.untyped_data()),
        static_cast<const float*>(center.untyped_data()), static_cast<const float*>(logZ.untyped_data()),
        static_cast<const float*>(shift2.untyped_data()), static_cast<const int*>(start.untyped_data()),
        static_cast<float*>(weights->untyped_data()), static_cast<float*>(sums->untyped_data()),
        static_cast<float*>(partial->untyped_data()), static_cast<int*>(count->untyped_data()));
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PpcaPosteriorPrepF32, PpcaPosteriorPrepF32Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("block_size")
                                  .Arg<ffi::AnyBuffer>()   /* score (cap, B, T)        */
                                  .Arg<ffi::AnyBuffer>()   /* mean (Q, cap, B, T)      */
                                  .Arg<ffi::AnyBuffer>()   /* cov (cap, B, tri(Q))     */
                                  .Arg<ffi::AnyBuffer>()   /* center (B,)              */
                                  .Arg<ffi::AnyBuffer>()   /* centered logZ (B,)       */
                                  .Arg<ffi::AnyBuffer>()   /* shift^2 (T,)             */
                                  .Arg<ffi::AnyBuffer>()   /* start (1,)               */
                                  .Ret<ffi::AnyBuffer>()   /* weights (P, R, B, T)     */
                                  .Ret<ffi::AnyBuffer>()   /* sums (K, R, B)           */
                                  .Ret<ffi::AnyBuffer>()   /* partial (2, R, B)        */
                                  .Ret<ffi::AnyBuffer>()); /* count (R, B)             */
