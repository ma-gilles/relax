// Flat-row translate-and-sum for RECOVAR's device-resident K=1 pass-2 M-step.
//
// Replaces the XLA reduce fusion of
// relax/fine_pass/resident_block_programs.py::resident_block_weighted_sums,
// which gathers a ``[rows, translations, pixels]`` shifted tile per M-step
// block (2.3 GB at the hp3 state) and contracts it with the posterior.  This
// kernel keeps only the per-image unshifted operands and applies RELION's
// translation inside the reduction, exactly as the fused-translate fine
// scorer does, so no ``[rows, T, P]`` tile is ever materialised:
//
//   summed[r, p]        = sum_t posterior[r, t] * shift_t(recon_image[id[r], p])
//   summed_masked[r, p] = sum_t posterior[r, t] * shift_t(noise_image[id[r], p])
//   probs_sum_t[r]      = sum_t posterior[r, t]
//
// With the optional ``ctf2_over_nv`` operand it also writes the fourth output
// the M-step block needs, reproducing
// ``local_backprojection.compute_local_ctf_sums_from_probs_sum_t`` exactly:
//
//   ctf_probs[r, p] = probs_sum_t[r] != 0 ? probs_sum_t[r] * ctf2[id[r], p] : 0
//
// one rounded multiply under that helper's own ``!= 0`` mass predicate.
//
// Addressing follows the flat-row family (``relion_fine_diff2_fused_translate_
// runtime_flat_rows_f32`` and ``relion_wavg_sequential_runtime_flat_rows_
// triplet_f32``): every row carries its own image id and a negative id marks
// padding.  ``x``/``y`` come from RECOVAR's centered packed-half pixel index,
// the convention ``relion_translate_score_f32_kernel`` uses to build the very
// tile this kernel replaces; the fine scorer derives the same ``(x, y)`` from
// its own RELION full-pixel rectangle instead.
//
// Arithmetic contract.  The phase and the complex rotation are
// ``relion_score_translate_f32`` split in two, so that one ``sincosf`` per
// (pixel, translation) serves every row a block owns.  Both halves are written
// with the same rounding intrinsics as the original, which are not contractible,
// so the split emits the same operations.  The reduction over translations is
// sequential in increasing ``t`` with a rounded multiply followed by a rounded
// add, the order RELION's Wavg accumulation uses.
//
// Two operand conventions.  ``prepare_bucket_io`` builds the reconstruction
// tile with ``relion_translate_score_f32`` normally, but with
// ``relion_translate_bpref_f32`` when the exact RELION BPref operands are
// selected: that primitive pins the same complex-rotation FMA order as the
// score path, then multiplies by the weighted CTF. Applying the weight before
// versus after rotation can round differently. ``recon_weight`` selects the
// BPref convention for the first operand; the noise operand always uses the
// score convention used by ``shifted_score_half_with_dc``.
//
// Shifted table (relax#98).  When many rows read each image (a lone tilt
// particle's M-step block is hundreds of rows of one image), the launcher first
// writes every image's shifted operands at every (translation, pixel) into a
// stream-ordered scratch table with the same phase, ``sincosf`` and rotation
// functions, and the summing kernel loads them instead of forming them: the
// values, and so every sum, are bitwise those of the inline path.
//
// This header is included from cuda_backproject.cu after the anonymous
// namespace that defines relion_score_translate_f32.

namespace recovar_relion_translate_sum {

constexpr int kBlockThreads = 256;
// Dynamic shared tables, below the 48 KB a block gets without an opt-in
// (the kernel's few static shared words take the rest).
constexpr int kMaxSharedBytes = 48 * 1024 - 256;
// Blocks a launch aims for: the pixel axis is split until the grid has them.
constexpr int64_t kTargetBlocks = 8192;

// The phase of relion_score_translate_f32, verbatim: one rounded y product
// followed by an x FMA.  relion_translate_bpref_f32_kernel writes the same
// phase as ``sincosf(x * tx + y * ty)`` and its PTX contracts to this same
// sequence, so one evaluation serves both operand conventions.
__device__ __forceinline__ float translate_phase_f32(
    int x, int y, float tx, float ty)
{
    return __fmaf_rn(
        static_cast<float>(x), tx,
        __fmul_rn(static_cast<float>(y), ty));
}

// The complex rotation of relion_score_translate_f32, verbatim.
__device__ __forceinline__ float2 translate_rotate_f32(
    float2 value, float sine, float cosine)
{
    const float translated_real = __fmaf_rn(
        cosine, value.x,
        -__fmul_rn(sine, value.y));
    const float translated_imag = __fmaf_rn(
        sine, value.x,
        __fmul_rn(cosine, value.y));
    return make_float2(translated_real, translated_imag);
}

// Match the standalone BPref primitive, including the captured RELION
// imaginary-component FMA order. Keep weighting after the complex rotation.
// See docs/math/relion_refinement_algorithm.md (BPref translation arithmetic).
__device__ __forceinline__ float2 translate_rotate_bpref_f32(
    float2 value, float sine, float cosine, float factor)
{
    const float translated_real = __fmaf_rn(
        cosine, value.x,
        -__fmul_rn(sine, value.y));
    const float translated_imag = __fmaf_rn(
        sine, value.x,
        __fmul_rn(cosine, value.y));
    return make_float2(
        __fmul_rn(translated_real, factor),
        __fmul_rn(translated_imag, factor));
}

template <int ROWS_PER_BLOCK, bool BPREF_RECON, bool SHIFTED_TABLE>
__global__ void __launch_bounds__(kBlockThreads)
translate_sum_flat_rows_f32_kernel(
    const float4* __restrict__ shifted_table,        // [B, T, P] (recon, noise) when SHIFTED_TABLE
    const float2* __restrict__ recon_image,          // [B, P]
    const float2* __restrict__ noise_image,          // [B, P]
    const float* __restrict__ recon_weight,          // [B, P] or null
    const float* __restrict__ ctf2_over_nv,          // [B, P] or null
    const int32_t* __restrict__ row_image_ids,       // [Q]
    const float* __restrict__ posterior,             // [Q, T]
    const float* __restrict__ translation_angles,    // [T, 2], or [B, T, 2] with angle_image_stride
    const int32_t* __restrict__ pixel_indices,       // [P]
    const int32_t* __restrict__ runtime_valid_rows,  // scalar or null
    const int32_t* __restrict__ runtime_logical_pixels,  // scalar or null
    float2* __restrict__ summed,                     // [Q, P]
    float2* __restrict__ summed_masked,              // [Q, P]
    float* __restrict__ probs_sum_t,                 // [Q]
    float* __restrict__ ctf_probs,                   // [Q, P] or null
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    int image_h,
    int image_half_width,
    int64_t angle_image_stride)
{
    // blockIdx.x owns ROWS_PER_BLOCK rows; blockIdx.y a span of the pixels
    // (the grid splits them so a launch with few rows still fills the GPU).
    const int64_t row_base =
        static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK;
    if (row_base >= row_count) return;
    const int64_t pixels_per_span =
        ((pixel_capacity + gridDim.y - 1) / gridDim.y + kBlockThreads - 1) /
        kBlockThreads * kBlockThreads;
    const int64_t span_begin = static_cast<int64_t>(blockIdx.y) * pixels_per_span;
    const int64_t span_end = min(pixel_capacity, span_begin + pixels_per_span);
    // The per-row scalars (probs_sum_t, the fail-closed sentinels) belong to
    // the first span.
    const bool first_span = blockIdx.y == 0;

    const int64_t valid_rows = runtime_valid_rows == nullptr
        ? row_count
        : static_cast<int64_t>(runtime_valid_rows[0]);
    const int64_t logical_pixels = runtime_logical_pixels == nullptr
        ? pixel_capacity
        : static_cast<int64_t>(runtime_logical_pixels[0]);
    if (valid_rows < 0 || valid_rows > row_count ||
        logical_pixels < 0 || logical_pixels > pixel_capacity) {
        // Fail closed exactly as the runtime flat-row kernels do.
        if (blockIdx.x == 0 && first_span && threadIdx.x == 0) {
            summed[0] = make_float2(nanf(""), nanf(""));
            summed_masked[0] = make_float2(nanf(""), nanf(""));
            probs_sum_t[0] = nanf("");
        }
        return;
    }
    // Rows past the valid count keep the zeros the launcher wrote.
    if (row_base >= valid_rows) return;

    int32_t image_ids[ROWS_PER_BLOCK];
    bool live[ROWS_PER_BLOCK];
    bool any_live = false;
#pragma unroll
    for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
        const int64_t row = row_base + i;
        image_ids[i] = -1;
        live[i] = false;
        if (row < valid_rows) {
            const int32_t id = row_image_ids[row];
            if (id >= static_cast<int32_t>(batch_size)) {
                // Out-of-range map: fail closed on this row's outputs.
                for (int64_t pixel = span_begin + threadIdx.x;
                     pixel < span_end;
                     pixel += kBlockThreads) {
                    const int64_t base = row * pixel_capacity + pixel;
                    summed[base] = make_float2(nanf(""), nanf(""));
                    summed_masked[base] = make_float2(nanf(""), nanf(""));
                }
                if (first_span && threadIdx.x == 0) probs_sum_t[row] = nanf("");
                continue;
            }
            // A negative id is padding: it reads nothing and stays zero.
            if (id >= 0) {
                image_ids[i] = id;
                live[i] = true;
                any_live = true;
            }
        }
    }
    if (!any_live) return;

    // Shared copies of the block's posterior rows, its active-translation
    // list and, with one table for every image, the angles.  Without them the
    // inner translation loop re-reads them once per pixel pass.  Tilt images
    // carry one [T, 2] table each (angle_image_stride = 2T), read from global
    // memory for the image of the row being visited.
    const bool shared_table = angle_image_stride == 0;
    extern __shared__ float translate_sum_shared[];
    float* shared_posterior = translate_sum_shared;                         // [R, T]
    int32_t* shared_active = reinterpret_cast<int32_t*>(
        shared_posterior + ROWS_PER_BLOCK * translation_count);             // [T]
    float* shared_angles = reinterpret_cast<float*>(
        shared_active + translation_count);                                 // [2T] when shared_table
    __shared__ int32_t active_count;
    if (shared_table) {
        for (int64_t index = threadIdx.x;
             index < 2 * translation_count;
             index += kBlockThreads) {
            shared_angles[index] = translation_angles[index];
        }
    }
    for (int64_t index = threadIdx.x;
         index < ROWS_PER_BLOCK * translation_count;
         index += kBlockThreads) {
        const int64_t i = index / translation_count;
        const int64_t t = index - i * translation_count;
        const int64_t row = row_base + i;
        // Padding rows past the valid count are never read again, but a
        // defined zero keeps the shared tile free of stale values.
        shared_posterior[index] = row < valid_rows
            ? posterior[row * translation_count + t]
            : 0.0f;
    }
    __syncthreads();

    // The translations that carry weight in any of the block's rows, in
    // increasing order. The fine posterior is zero outside RELION's
    // significant samples (acc_ml_optimiser_impl.h:4819), so most translations
    // of a row add exact zeros: w * v is +-0 for finite v, and adding it to an
    // accumulator that starts at +0 leaves it unchanged, so skipping them keeps
    // every sum bitwise and saves their sincosf and rotations.  One warp builds
    // the list with ballots, 32 translations at a time.
    if (threadIdx.x < 32) {
        int32_t count = 0;
        for (int64_t first = 0; first < translation_count; first += 32) {
            const int64_t t = first + threadIdx.x;
            bool any = false;
            if (t < translation_count) {
#pragma unroll
                for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
                    any = any || (live[i] && shared_posterior[i * translation_count + t] != 0.0f);
                }
            }
            const unsigned mask = __ballot_sync(0xffffffffu, any);
            if (any)
                shared_active[count + __popc(mask & ((1u << threadIdx.x) - 1u))] =
                    static_cast<int32_t>(t);
            count += __popc(mask);
        }
        if (threadIdx.x == 0) active_count = count;
    }

    __syncthreads();

    // probs_sum_t is the same sequential translation sum, one row per lane,
    // over the active translations: the others add exact zeros.  ctf_probs
    // needs it on every lane, so it also lands in shared memory.
    __shared__ float shared_mass[ROWS_PER_BLOCK];
    if (threadIdx.x < ROWS_PER_BLOCK && (first_span || ctf_probs != nullptr)) {
        const int i = threadIdx.x;
        float mass = 0.0f;
        if (live[i]) {
            for (int32_t active = 0; active < active_count; ++active) {
                mass = __fadd_rn(
                    mass,
                    shared_posterior[i * translation_count + shared_active[active]]);
            }
            if (first_span) probs_sum_t[row_base + i] = mass;
        }
        shared_mass[i] = mass;
    }
    __syncthreads();

    for (int64_t pixel = span_begin + threadIdx.x;
         pixel < min(span_end, logical_pixels);
         pixel += kBlockThreads)
    {
        const int pixel_index = pixel_indices[pixel];
        const int x = pixel_index % image_half_width;
        // RELION's label for an uncropped half image puts the Nyquist row at
        // +N/2 (fftw.h:99-109); this kernel replaces a tile built by
        // relion_translate_score_f32_kernel, so it converts the same way.
        const int y = relion_centered_row_to_relion_label(
            pixel_index / image_half_width - image_h / 2, image_h);

        float2 recon_value[ROWS_PER_BLOCK];
        float2 noise_value[ROWS_PER_BLOCK];
        float2 recon_acc[ROWS_PER_BLOCK];
        float2 noise_acc[ROWS_PER_BLOCK];
        float recon_factor[ROWS_PER_BLOCK];
#pragma unroll
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            recon_acc[i] = make_float2(0.0f, 0.0f);
            noise_acc[i] = make_float2(0.0f, 0.0f);
            recon_value[i] = make_float2(0.0f, 0.0f);
            noise_value[i] = make_float2(0.0f, 0.0f);
            recon_factor[i] = 0.0f;
            if (!SHIFTED_TABLE && live[i]) {
                const int64_t base =
                    static_cast<int64_t>(image_ids[i]) * pixel_capacity + pixel;
                recon_value[i] = recon_image[base];
                noise_value[i] = noise_image[base];
                if constexpr (BPREF_RECON) recon_factor[i] = recon_weight[base];
            }
        }

        const int32_t n_active = active_count;
        for (int32_t active = 0; active < n_active; ++active)
        {
            const int64_t translation = shared_active[active];
            // The phase depends on the translation and, for tilt images, on
            // the image's table; the shifted operands on the image too.
            // Rows of one image (consecutive in a block) reuse both, so a
            // block evaluates one sincosf and one pair of rotations per
            // (image, translation, pixel) instead of per row (relax#91).
            int32_t phase_image = -2;
            float sine = 0.0f;
            float cosine = 0.0f;
            int32_t shifted_image = -1;
            float2 recon_shifted = make_float2(0.0f, 0.0f);
            float2 noise_shifted = make_float2(0.0f, 0.0f);
#pragma unroll
            for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
                if (!live[i]) continue;
                const float weight =
                    shared_posterior[i * translation_count + translation];
                if (weight == 0.0f) continue;
                if constexpr (SHIFTED_TABLE) {
                    if (image_ids[i] != shifted_image) {
                        const float4 shifted = shifted_table[
                            (static_cast<int64_t>(image_ids[i]) * translation_count + translation) *
                                pixel_capacity + pixel];
                        recon_shifted = make_float2(shifted.x, shifted.y);
                        noise_shifted = make_float2(shifted.z, shifted.w);
                        shifted_image = image_ids[i];
                    }
                } else {
                    const int32_t table_image = shared_table ? -1 : image_ids[i];
                    if (table_image != phase_image) {
                        const float* angles = shared_table
                            ? shared_angles
                            : translation_angles +
                                  static_cast<int64_t>(image_ids[i]) * angle_image_stride;
                        const float phase = translate_phase_f32(
                            x, y, angles[2 * translation], angles[2 * translation + 1]);
                        sincosf(phase, &sine, &cosine);
                        phase_image = table_image;
                        shifted_image = -1;
                    }
                    if (image_ids[i] != shifted_image) {
                        recon_shifted = BPREF_RECON
                            ? translate_rotate_bpref_f32(
                                  recon_value[i], sine, cosine, recon_factor[i])
                            : translate_rotate_f32(recon_value[i], sine, cosine);
                        noise_shifted =
                            translate_rotate_f32(noise_value[i], sine, cosine);
                        shifted_image = image_ids[i];
                    }
                }
                recon_acc[i].x = __fadd_rn(
                    recon_acc[i].x, __fmul_rn(weight, recon_shifted.x));
                recon_acc[i].y = __fadd_rn(
                    recon_acc[i].y, __fmul_rn(weight, recon_shifted.y));
                noise_acc[i].x = __fadd_rn(
                    noise_acc[i].x, __fmul_rn(weight, noise_shifted.x));
                noise_acc[i].y = __fadd_rn(
                    noise_acc[i].y, __fmul_rn(weight, noise_shifted.y));
            }
        }

#pragma unroll
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            if (!live[i]) continue;
            const int64_t base = (row_base + i) * pixel_capacity + pixel;
            summed[base] = recon_acc[i];
            summed_masked[base] = noise_acc[i];
            if (ctf_probs != nullptr) {
                // compute_local_ctf_sums_from_probs_sum_t, term for term: the
                // predicate is on the mass, not on the product, and the
                // product is one rounded float32 multiply.
                const float mass = shared_mass[i];
                ctf_probs[base] = mass != 0.0f
                    ? __fmul_rn(
                          mass,
                          ctf2_over_nv[
                              static_cast<int64_t>(image_ids[i]) *
                                  pixel_capacity +
                              pixel])
                    : 0.0f;
            }
        }
    }
}

// One image's shifted operands at every (translation, pixel), written by the
// same phase, sincosf and rotation the summing kernel evaluates inline, so a
// row reading the table adds exactly the values it would have formed itself.
template <bool BPREF_RECON>
__global__ void __launch_bounds__(kBlockThreads)
translate_shifted_table_f32_kernel(
    const float2* __restrict__ recon_image,          // [B, P]
    const float2* __restrict__ noise_image,          // [B, P]
    const float* __restrict__ recon_weight,          // [B, P] or null
    const float* __restrict__ translation_angles,    // [T, 2], or [B, T, 2] with angle_image_stride
    const int32_t* __restrict__ pixel_indices,       // [P]
    float4* __restrict__ shifted_table,              // [B, T, P]
    int64_t translation_count,
    int64_t pixel_capacity,
    int image_h,
    int image_half_width,
    int64_t angle_image_stride)
{
    const int64_t pixel = static_cast<int64_t>(blockIdx.x) * kBlockThreads + threadIdx.x;
    const int64_t translation = blockIdx.y;
    const int64_t image = blockIdx.z;
    if (pixel >= pixel_capacity) return;
    const int pixel_index = pixel_indices[pixel];
    const int x = pixel_index % image_half_width;
    const int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    const float* angles = translation_angles + image * angle_image_stride;
    const float phase = translate_phase_f32(
        x, y, angles[2 * translation], angles[2 * translation + 1]);
    float sine;
    float cosine;
    sincosf(phase, &sine, &cosine);
    const int64_t base = image * pixel_capacity + pixel;
    const float2 recon = BPREF_RECON
        ? translate_rotate_bpref_f32(recon_image[base], sine, cosine, recon_weight[base])
        : translate_rotate_f32(recon_image[base], sine, cosine);
    const float2 noise = translate_rotate_f32(noise_image[base], sine, cosine);
    shifted_table[(image * translation_count + translation) * pixel_capacity + pixel] =
        make_float4(recon.x, recon.y, noise.x, noise.y);
}

// Tiled table sum (relax#98).  When every row of a launch reads one image (a lone
// particle's M-step block), a block owns kTiledRows rows by kTiledPixels pixels
// and stages, kTiledTranslations translations at a time, the rows' posterior and
// the image's table entries in shared memory, so each table load serves
// kTiledRows rows instead of kShiftedTableRowsPerBlock.  Every output is the same
// sequence of rounded operations as the row kernel's: a rounded multiply and a
// rounded add per translation with a nonzero weight, in increasing translation
// order, from a zero accumulator; the mass is the same sequential sum.
constexpr int kTiledRows = relax_tiled_translations::kTiledRows;
constexpr int kTiledPixels = 64;
constexpr int kTiledTranslations = 32;
constexpr int kTiledPixelLanes = 16;                       // threads along the pixels
constexpr int kTiledRowLanes = kBlockThreads / kTiledPixelLanes;  // 16 along the rows
constexpr int kTiledRowsPerThread = kTiledRows / kTiledRowLanes;  // 2
constexpr int kTiledPixelsPerThread = kTiledPixels / kTiledPixelLanes;  // 4

__global__ void __launch_bounds__(kBlockThreads)
translate_sum_tiled_table_f32_kernel(
    const float4* __restrict__ shifted_table,        // [1, T, P] (recon, noise)
    const int32_t* __restrict__ active_lists,        // [row tiles, T]
    const int32_t* __restrict__ active_counts,       // [row tiles]
    const float* __restrict__ ctf2_over_nv,          // [1, P] or null
    const int32_t* __restrict__ row_image_ids,       // [Q]: 0, or negative for padding
    const float* __restrict__ posterior,             // [Q, T]
    const int32_t* __restrict__ runtime_valid_rows,  // scalar or null
    const int32_t* __restrict__ runtime_logical_pixels,  // scalar or null
    float2* __restrict__ summed,                     // [Q, P]
    float2* __restrict__ summed_masked,              // [Q, P]
    float* __restrict__ probs_sum_t,                 // [Q]
    float* __restrict__ ctf_probs,                   // [Q, P] or null
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity)
{
    const int64_t row_base = static_cast<int64_t>(blockIdx.x) * kTiledRows;
    const int64_t pixel_base = static_cast<int64_t>(blockIdx.y) * kTiledPixels;
    const int64_t valid_rows = runtime_valid_rows == nullptr
        ? row_count
        : static_cast<int64_t>(runtime_valid_rows[0]);
    const int64_t logical_pixels = runtime_logical_pixels == nullptr
        ? pixel_capacity
        : static_cast<int64_t>(runtime_logical_pixels[0]);
    if (valid_rows < 0 || valid_rows > row_count ||
        logical_pixels < 0 || logical_pixels > pixel_capacity) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            summed[0] = make_float2(nanf(""), nanf(""));
            summed_masked[0] = make_float2(nanf(""), nanf(""));
            probs_sum_t[0] = nanf("");
        }
        return;
    }
    if (row_base >= valid_rows) return;

    __shared__ float shared_weight[kTiledTranslations][kTiledRows + 1];
    __shared__ float4 shared_value[kTiledTranslations][kTiledPixels];
    __shared__ int32_t shared_live[kTiledRows];
    __shared__ float shared_mass[kTiledRows];
    // The translations with a nonzero weight in any of the block's rows, in increasing order (dynamic): the others
    // add nothing to any sum, as in the row kernel.
    extern __shared__ int32_t tiled_active[];
    __shared__ int32_t tiled_active_count;

    // A row with image 0 is live; a negative id is padding (zeros); any other id is out of range for a one-image
    // table and fails closed, as the row kernel does for an id past the batch.
    if (threadIdx.x < kTiledRows) {
        const int64_t row = row_base + threadIdx.x;
        int32_t live = 0;
        if (row < valid_rows) {
            const int32_t id = row_image_ids[row];
            live = id == 0 ? 1 : (id > 0 ? -1 : 0);
        }
        shared_live[threadIdx.x] = live;
    }
    __syncthreads();
    // The row tile's active translations (relax_tiled_translations::row_tile_active_translations_kernel), staged in shared memory.
    const int64_t tile_active_count = active_counts[blockIdx.x];
    for (int64_t index = threadIdx.x; index < tile_active_count; index += kBlockThreads)
        tiled_active[index] = active_lists[static_cast<int64_t>(blockIdx.x) * translation_count + index];
    if (threadIdx.x == 0) tiled_active_count = static_cast<int32_t>(tile_active_count);
    __syncthreads();
    const int64_t active_count = tiled_active_count;

    const int pixel_lane = threadIdx.x % kTiledPixelLanes;
    const int row_lane = threadIdx.x / kTiledPixelLanes;
    float4 acc[kTiledRowsPerThread][kTiledPixelsPerThread];
#pragma unroll
    for (int i = 0; i < kTiledRowsPerThread; ++i)
#pragma unroll
        for (int j = 0; j < kTiledPixelsPerThread; ++j) acc[i][j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // The mass of a row: lanes 0..kTiledRows-1 of the first pixel tile sum their row sequentially.
    float mass = 0.0f;

    // The next chunk's weights and values are loaded into registers while the current one is summed from shared
    // memory (each thread its share: kTiledLoadsW weights, kTiledLoadsV values).
    constexpr int kTiledLoadsW = kTiledTranslations * kTiledRows / kBlockThreads;
    constexpr int kTiledLoadsV = kTiledTranslations * kTiledPixels / kBlockThreads;
    float next_weight[kTiledLoadsW];
    float4 next_value[kTiledLoadsV];
    auto load_chunk = [&](int64_t chunk_first) {
        const int chunk_count = active_count - chunk_first < kTiledTranslations
            ? static_cast<int>(active_count - chunk_first)
            : kTiledTranslations;
#pragma unroll
        for (int k = 0; k < kTiledLoadsW; ++k) {
            const int index = threadIdx.x + k * kBlockThreads;
            const int t = index / kTiledRows;
            const int i = index - t * kTiledRows;
            next_weight[k] = (t < chunk_count && shared_live[i] == 1)
                ? posterior[(row_base + i) * translation_count + tiled_active[chunk_first + t]]
                : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < kTiledLoadsV; ++k) {
            const int index = threadIdx.x + k * kBlockThreads;
            const int t = index / kTiledPixels;
            const int j = index - t * kTiledPixels;
            const int64_t pixel = pixel_base + j;
            next_value[k] = (t < chunk_count && pixel < pixel_capacity)
                ? shifted_table[static_cast<int64_t>(tiled_active[chunk_first + t]) * pixel_capacity + pixel]
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    if (active_count > 0) load_chunk(0);
    for (int64_t first = 0; first < active_count; first += kTiledTranslations) {
        const int count = active_count - first < kTiledTranslations
            ? static_cast<int>(active_count - first)
            : kTiledTranslations;
#pragma unroll
        for (int k = 0; k < kTiledLoadsW; ++k) {
            const int index = threadIdx.x + k * kBlockThreads;
            shared_weight[index / kTiledRows][index % kTiledRows] = next_weight[k];
        }
#pragma unroll
        for (int k = 0; k < kTiledLoadsV; ++k) {
            const int index = threadIdx.x + k * kBlockThreads;
            shared_value[index / kTiledPixels][index % kTiledPixels] = next_value[k];
        }
        __syncthreads();
        if (first + kTiledTranslations < active_count) load_chunk(first + kTiledTranslations);
        if (threadIdx.x < kTiledRows && shared_live[threadIdx.x] == 1) {
            for (int t = 0; t < count; ++t) mass = __fadd_rn(mass, shared_weight[t][threadIdx.x]);
        }
        for (int t = 0; t < count; ++t) {
#pragma unroll
            for (int i = 0; i < kTiledRowsPerThread; ++i) {
                const float weight = shared_weight[t][row_lane + i * kTiledRowLanes];
                if (weight == 0.0f) continue;
#pragma unroll
                for (int j = 0; j < kTiledPixelsPerThread; ++j) {
                    const float4 value = shared_value[t][pixel_lane + j * kTiledPixelLanes];
                    acc[i][j].x = __fadd_rn(acc[i][j].x, __fmul_rn(weight, value.x));
                    acc[i][j].y = __fadd_rn(acc[i][j].y, __fmul_rn(weight, value.y));
                    acc[i][j].z = __fadd_rn(acc[i][j].z, __fmul_rn(weight, value.z));
                    acc[i][j].w = __fadd_rn(acc[i][j].w, __fmul_rn(weight, value.w));
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < kTiledRows) {
        shared_mass[threadIdx.x] = mass;
        if (blockIdx.y == 0 && shared_live[threadIdx.x] == 1) probs_sum_t[row_base + threadIdx.x] = mass;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kTiledRowsPerThread; ++i) {
        const int local_row = row_lane + i * kTiledRowLanes;
        const int64_t row = row_base + local_row;
        const int32_t live = shared_live[local_row];
        if (live == 0) continue;
#pragma unroll
        for (int j = 0; j < kTiledPixelsPerThread; ++j) {
            const int64_t pixel = pixel_base + pixel_lane + j * kTiledPixelLanes;
            if (live < 0) {
                if (pixel < pixel_capacity) {
                    summed[row * pixel_capacity + pixel] = make_float2(nanf(""), nanf(""));
                    summed_masked[row * pixel_capacity + pixel] = make_float2(nanf(""), nanf(""));
                }
                continue;
            }
            if (pixel >= logical_pixels) continue;
            const int64_t base = row * pixel_capacity + pixel;
            summed[base] = make_float2(acc[i][j].x, acc[i][j].y);
            summed_masked[base] = make_float2(acc[i][j].z, acc[i][j].w);
            if (ctf_probs != nullptr) {
                const float row_mass = shared_mass[local_row];
                ctf_probs[base] = row_mass != 0.0f ? __fmul_rn(row_mass, ctf2_over_nv[pixel]) : 0.0f;
            }
        }
    }
    if (blockIdx.y == 0 && threadIdx.x < kTiledRows && shared_live[threadIdx.x] < 0) probs_sum_t[row_base + threadIdx.x] = nanf("");
}

// The shifted table pays when many rows read each image's entries: one row
// block of a lone particle is hundreds of rows of one image.  It is bounded by
// kMaxShiftedTableBytes of stream-ordered scratch.
constexpr int64_t kMaxShiftedTableBytes = int64_t(1) << 30;
constexpr int64_t kMinRowsPerTableImage = 64;
// With the table a row's work is a load and four rounded operations, so rows
// share the load: four rows per block measured best on an H100 (relax#98:
// a flat 256-row lone launch 7.18 -> 4.02 ms; two rows 5.49, eight 5.77).
constexpr int kShiftedTableRowsPerBlock = 4;

inline bool use_shifted_table(int64_t batch_size, int64_t row_count, int64_t translation_count, int64_t pixel_capacity)
{
    return batch_size * translation_count * pixel_capacity * int64_t(sizeof(float4)) <= kMaxShiftedTableBytes &&
        row_count >= kMinRowsPerTableImage * batch_size;
}

inline int64_t shared_bytes_for(
    int rows_per_block, int64_t translation_count, bool shared_table)
{
    // The block's posterior rows, its active-translation list and, with one
    // table for every image, the angles.
    return static_cast<int64_t>(
        rows_per_block * translation_count * sizeof(float)
        + translation_count * sizeof(int32_t)
        + (shared_table ? 2 * translation_count * sizeof(float) : 0));
}

// Rows a block owns.  More rows share one sincosf and one pair of rotations
// per image (or one shifted-table load) but cost registers and visit the union
// of the rows' translations.
// The default is one: measured on an H100 (relax#91), one row per block with
// the pixel spans was faster than four rows at every SPA and sparse tilt
// shape, and 1.15x against four rows' 1.54x only for a flat posterior over
// 1,176 translations.  The launcher keeps the shared tables inside
// kMaxSharedBytes and never asks for more rows than the problem has.
inline int choose_rows_per_block(
    int requested, int64_t row_count, int64_t translation_count, bool shared_table)
{
    int rows = requested > 0 ? requested : 1;
    if (rows != 1 && rows != 2 && rows != 4 && rows != 8) rows = 1;
    while (rows > 1 && static_cast<int64_t>(rows) > row_count) rows /= 2;
    while (rows > 1 &&
           shared_bytes_for(rows, translation_count, shared_table) > kMaxSharedBytes) {
        rows /= 2;
    }
    return rows;
}

// Pixel spans per row block: enough that a launch has about kTargetBlocks
// blocks, never more spans than 256-pixel passes.
inline int64_t choose_pixel_spans(int64_t row_blocks, int64_t pixel_capacity)
{
    const int64_t passes = (pixel_capacity + kBlockThreads - 1) / kBlockThreads;
    int64_t spans = (kTargetBlocks + row_blocks - 1) / row_blocks;
    return spans < 1 ? 1 : (spans > passes ? passes : spans);
}

template <int ROWS_PER_BLOCK, bool BPREF_RECON, bool SHIFTED_TABLE>
cudaError_t launch_templated(
    cudaStream_t stream,
    const float4* shifted_table,
    const float2* recon_image,
    const float2* noise_image,
    const float* recon_weight,
    const float* ctf2_over_nv,
    const int32_t* row_image_ids,
    const float* posterior,
    const float* translation_angles,
    const int32_t* pixel_indices,
    const int32_t* runtime_valid_rows,
    const int32_t* runtime_logical_pixels,
    float2* summed,
    float2* summed_masked,
    float* probs_sum_t,
    float* ctf_probs,
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    int image_h,
    int image_half_width,
    int64_t angle_image_stride)
{
    const int64_t blocks =
        (row_count + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const size_t shared = static_cast<size_t>(
        shared_bytes_for(ROWS_PER_BLOCK, translation_count, angle_image_stride == 0));
    const dim3 grid(
        static_cast<unsigned>(blocks),
        static_cast<unsigned>(choose_pixel_spans(blocks, pixel_capacity)));
    translate_sum_flat_rows_f32_kernel<ROWS_PER_BLOCK, BPREF_RECON, SHIFTED_TABLE>
        <<<grid, kBlockThreads, shared, stream>>>(
            shifted_table,
            recon_image,
            noise_image,
            recon_weight,
            ctf2_over_nv,
            row_image_ids,
            posterior,
            translation_angles,
            pixel_indices,
            runtime_valid_rows,
            runtime_logical_pixels,
            summed,
            summed_masked,
            probs_sum_t,
            ctf_probs,
            batch_size,
            row_count,
            translation_count,
            pixel_capacity,
            image_h,
            image_half_width,
            angle_image_stride);
    return cudaGetLastError();
}

template <bool SHIFTED_TABLE>
cudaError_t dispatch_sum(
    cudaStream_t stream,
    const float4* shifted_table,
    int rows,
    const float2* recon_image,
    const float2* noise_image,
    const float* recon_weight,
    const float* ctf2_over_nv,
    const int32_t* row_image_ids,
    const float* posterior,
    const float* translation_angles,
    const int32_t* pixel_indices,
    const int32_t* runtime_valid_rows,
    const int32_t* runtime_logical_pixels,
    float2* summed,
    float2* summed_masked,
    float* probs_sum_t,
    float* ctf_probs,
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    int image_h,
    int image_half_width,
    int64_t angle_image_stride)
{
#define RELAX_TRANSLATE_SUM_DISPATCH(R, BPREF)                  \
    case R:                                                       \
        return launch_templated<R, BPREF, SHIFTED_TABLE>(         \
            stream, shifted_table, recon_image, noise_image, recon_weight, \
            ctf2_over_nv, row_image_ids, posterior,               \
            translation_angles, pixel_indices, runtime_valid_rows,\
            runtime_logical_pixels, summed, summed_masked,        \
            probs_sum_t, ctf_probs, batch_size, row_count,        \
            translation_count, pixel_capacity, image_h,           \
            image_half_width, angle_image_stride)

    if (recon_weight != nullptr) {
        switch (rows) {
            RELAX_TRANSLATE_SUM_DISPATCH(1, true);
            RELAX_TRANSLATE_SUM_DISPATCH(2, true);
            RELAX_TRANSLATE_SUM_DISPATCH(4, true);
            RELAX_TRANSLATE_SUM_DISPATCH(8, true);
            default:
                return cudaErrorInvalidValue;
        }
    }
    switch (rows) {
        RELAX_TRANSLATE_SUM_DISPATCH(1, false);
        RELAX_TRANSLATE_SUM_DISPATCH(2, false);
        RELAX_TRANSLATE_SUM_DISPATCH(4, false);
        RELAX_TRANSLATE_SUM_DISPATCH(8, false);
        default:
            return cudaErrorInvalidValue;
    }
#undef RELAX_TRANSLATE_SUM_DISPATCH
}

inline cudaError_t launch(
    cudaStream_t stream,
    const float2* recon_image,
    const float2* noise_image,
    const float* recon_weight,
    const float* ctf2_over_nv,
    const int32_t* row_image_ids,
    const float* posterior,
    const float* translation_angles,
    const int32_t* pixel_indices,
    const int32_t* runtime_valid_rows,
    const int32_t* runtime_logical_pixels,
    float2* summed,
    float2* summed_masked,
    float* probs_sum_t,
    float* ctf_probs,
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    int image_h,
    int image_half_width,
    int requested_rows_per_block,
    int64_t angle_image_stride)
{
    if (row_count == 0 || pixel_capacity == 0) return cudaSuccess;
    // The zero fill owns every padding row, every zero-mass row and the
    // physical pixel tail; the kernel writes only what it is allowed to read.
    cudaError_t err = cudaMemsetAsync(
        summed,
        0,
        static_cast<size_t>(row_count) * pixel_capacity * sizeof(float2),
        stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(
        summed_masked,
        0,
        static_cast<size_t>(row_count) * pixel_capacity * sizeof(float2),
        stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(
        probs_sum_t,
        0,
        static_cast<size_t>(row_count) * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;
    if (ctf_probs != nullptr) {
        err = cudaMemsetAsync(
            ctf_probs,
            0,
            static_cast<size_t>(row_count) * pixel_capacity * sizeof(float),
            stream);
        if (err != cudaSuccess) return err;
    }
    if (translation_count == 0) return cudaSuccess;

    const bool shared_table = angle_image_stride == 0;
    const bool table = use_shifted_table(batch_size, row_count, translation_count, pixel_capacity);
    const int rows = choose_rows_per_block(
        requested_rows_per_block > 0 ? requested_rows_per_block : (table ? kShiftedTableRowsPerBlock : 1),
        row_count, translation_count, shared_table);
    if (shared_bytes_for(rows, translation_count, shared_table) > kMaxSharedBytes)
        return cudaErrorInvalidValue;

    if (table) {
        float4* shifted_table = nullptr;
        err = cudaMallocAsync(
            reinterpret_cast<void**>(&shifted_table),
            static_cast<size_t>(batch_size * translation_count * pixel_capacity) * sizeof(float4),
            stream);
        if (err != cudaSuccess) return err;
        const dim3 table_grid(
            static_cast<unsigned>((pixel_capacity + kBlockThreads - 1) / kBlockThreads),
            static_cast<unsigned>(translation_count),
            static_cast<unsigned>(batch_size));
        if (recon_weight != nullptr)
            translate_shifted_table_f32_kernel<true><<<table_grid, kBlockThreads, 0, stream>>>(
                recon_image, noise_image, recon_weight, translation_angles, pixel_indices, shifted_table,
                translation_count, pixel_capacity, image_h, image_half_width, angle_image_stride);
        else
            translate_shifted_table_f32_kernel<false><<<table_grid, kBlockThreads, 0, stream>>>(
                recon_image, noise_image, recon_weight, translation_angles, pixel_indices, shifted_table,
                translation_count, pixel_capacity, image_h, image_half_width, angle_image_stride);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        if (batch_size == 1) {
            // One image: the tiled sum (translate_sum_tiled_table_f32_kernel) over each row tile's active
            // translations (relax_tiled_translations::row_tile_active_translations_kernel).
            const int64_t tiles = (row_count + kTiledRows - 1) / kTiledRows;
            int32_t* active = nullptr;
            err = relax_tiled_translations::list_row_tile_translations(
                stream, row_image_ids, posterior, runtime_valid_rows, row_count, translation_count, &active);
            if (err == cudaSuccess) {
                const dim3 grid(
                    static_cast<unsigned>(tiles),
                    static_cast<unsigned>((pixel_capacity + kTiledPixels - 1) / kTiledPixels));
                translate_sum_tiled_table_f32_kernel<<<
                    grid, kBlockThreads, static_cast<size_t>(translation_count) * sizeof(int32_t), stream>>>(
                    shifted_table, active, active + tiles * translation_count, ctf2_over_nv, row_image_ids,
                    posterior, runtime_valid_rows, runtime_logical_pixels, summed, summed_masked, probs_sum_t,
                    ctf_probs, row_count, translation_count, pixel_capacity);
                err = cudaGetLastError();
                const cudaError_t active_err = cudaFreeAsync(active, stream);
                if (err == cudaSuccess) err = active_err;
            }
            const cudaError_t free_err = cudaFreeAsync(shifted_table, stream);
            return err != cudaSuccess ? err : free_err;
        }
        err = dispatch_sum<true>(
            stream, shifted_table, rows, recon_image, noise_image, recon_weight, ctf2_over_nv, row_image_ids,
            posterior, translation_angles, pixel_indices, runtime_valid_rows, runtime_logical_pixels, summed,
            summed_masked, probs_sum_t, ctf_probs, batch_size, row_count, translation_count, pixel_capacity,
            image_h, image_half_width, angle_image_stride);
        const cudaError_t free_err = cudaFreeAsync(shifted_table, stream);
        return err != cudaSuccess ? err : free_err;
    }
    return dispatch_sum<false>(
        stream, nullptr, rows, recon_image, noise_image, recon_weight, ctf2_over_nv, row_image_ids,
        posterior, translation_angles, pixel_indices, runtime_valid_rows, runtime_logical_pixels, summed,
        summed_masked, probs_sum_t, ctf_probs, batch_size, row_count, translation_count, pixel_capacity,
        image_h, image_half_width, angle_image_stride);
}

ffi::Error impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    int64_t rows_per_block,
    int64_t bpref_recon_attr,
    int64_t write_ctf_probs_attr,
    ffi::AnyBuffer recon_image,
    ffi::AnyBuffer noise_image,
    ffi::AnyBuffer recon_weight,
    ffi::AnyBuffer ctf2_over_nv,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer n_valid_rows,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> summed,
    ffi::Result<ffi::AnyBuffer> summed_masked,
    ffi::Result<ffi::AnyBuffer> probs_sum_t,
    ffi::Result<ffi::AnyBuffer> ctf_probs)
{
    if (recon_image.element_type() != ffi::DataType::C64 ||
        noise_image.element_type() != ffi::DataType::C64 ||
        recon_weight.element_type() != ffi::DataType::F32 ||
        ctf2_over_nv.element_type() != ffi::DataType::F32 ||
        ctf_probs->element_type() != ffi::DataType::F32 ||
        row_image_ids.element_type() != ffi::DataType::S32 ||
        posterior.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        pixel_indices.element_type() != ffi::DataType::S32 ||
        n_valid_rows.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        summed->element_type() != ffi::DataType::C64 ||
        summed_masked->element_type() != ffi::DataType::C64 ||
        probs_sum_t->element_type() != ffi::DataType::F32 ||
        n_valid_rows.dimensions().size() != 0 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: invalid buffers");

    const auto recon_dims = recon_image.dimensions();
    const auto noise_dims = noise_image.dimensions();
    const auto weight_dims = recon_weight.dimensions();
    const auto ctf_dims = ctf2_over_nv.dimensions();
    const auto ctf_out_dims = ctf_probs->dimensions();
    const auto row_dims = row_image_ids.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto index_dims = pixel_indices.dimensions();
    const auto summed_dims = summed->dimensions();
    const auto masked_dims = summed_masked->dimensions();
    const auto mass_dims = probs_sum_t->dimensions();
    if (recon_dims.size() != 2 || noise_dims.size() != 2 ||
        weight_dims.size() != 2 || ctf_dims.size() != 2 ||
        ctf_out_dims.size() != 2 ||
        row_dims.size() != 1 || posterior_dims.size() != 2 ||
        (angle_dims.size() != 2 && angle_dims.size() != 3) ||
        angle_dims[angle_dims.size() - 1] != 2 ||
        index_dims.size() != 1 || summed_dims.size() != 2 ||
        masked_dims.size() != 2 || mass_dims.size() != 1)
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: invalid ranks");

    const int64_t batch_size = recon_dims[0];
    const int64_t pixel_capacity = recon_dims[1];
    const int64_t row_count = row_dims[0];
    // [T, 2] shared by every image, or [B, T, 2]: one table per image (tilt images).
    const bool per_image_angles = angle_dims.size() == 3;
    const int64_t translation_count = per_image_angles ? angle_dims[1] : angle_dims[0];
    if (batch_size <= 0 || pixel_capacity <= 0 || row_count <= 0 ||
        translation_count <= 0 ||
        pixel_capacity > std::numeric_limits<int>::max() ||
        noise_dims[0] != batch_size || noise_dims[1] != pixel_capacity ||
        posterior_dims[0] != row_count ||
        posterior_dims[1] != translation_count ||
        index_dims[0] != pixel_capacity ||
        summed_dims[0] != row_count || summed_dims[1] != pixel_capacity ||
        masked_dims[0] != row_count || masked_dims[1] != pixel_capacity ||
        mass_dims[0] != row_count ||
        (per_image_angles && angle_dims[0] != batch_size))
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: inconsistent topology");
    const bool bpref_recon = bpref_recon_attr != 0;
    if (bpref_recon &&
        (weight_dims[0] != batch_size || weight_dims[1] != pixel_capacity))
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: BPref weight must be [batch, pixels]");
    const bool write_ctf_probs = write_ctf_probs_attr != 0;
    if (write_ctf_probs &&
        (ctf_dims[0] != batch_size || ctf_dims[1] != pixel_capacity ||
         ctf_out_dims[0] != row_count || ctf_out_dims[1] != pixel_capacity))
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: ctf2_over_nv must be "
            "[batch, pixels] and ctf_probs [rows, pixels]");
    if (image_h <= 0 || image_half_width <= 0 ||
        image_h > std::numeric_limits<int>::max() ||
        image_half_width > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: invalid image dimensions");

    const int64_t rows_per_grid_block = rows_per_block > 0 ? rows_per_block : 1;
    const int64_t blocks_needed =
        (row_count + rows_per_grid_block - 1) / rows_per_grid_block;
    if (blocks_needed > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: row count exceeds the grid");

    cudaError_t err = launch(
        stream,
        reinterpret_cast<const float2*>(recon_image.untyped_data()),
        reinterpret_cast<const float2*>(noise_image.untyped_data()),
        bpref_recon
            ? static_cast<const float*>(recon_weight.untyped_data())
            : nullptr,
        write_ctf_probs
            ? static_cast<const float*>(ctf2_over_nv.untyped_data())
            : nullptr,
        static_cast<const int32_t*>(row_image_ids.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        static_cast<const int32_t*>(n_valid_rows.untyped_data()),
        static_cast<const int32_t*>(logical_pixel_count.untyped_data()),
        reinterpret_cast<float2*>(summed->untyped_data()),
        reinterpret_cast<float2*>(summed_masked->untyped_data()),
        static_cast<float*>(probs_sum_t->untyped_data()),
        write_ctf_probs ? static_cast<float*>(ctf_probs->untyped_data()) : nullptr,
        batch_size,
        row_count,
        translation_count,
        pixel_capacity,
        static_cast<int>(image_h),
        static_cast<int>(image_half_width),
        static_cast<int>(rows_per_block),
        per_image_angles ? 2 * translation_count : 0);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

}  // namespace recovar_relion_translate_sum

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateSumFlatRowsF32,
    recovar_relion_translate_sum::impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Attr<int64_t>("rows_per_block")
        .Attr<int64_t>("bpref_recon")
        .Attr<int64_t>("write_ctf_probs")
        .Arg<ffi::AnyBuffer>()
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
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());
