// The translations a tile of flat rows visits (relax#98).  The tiled M-step sums (the table translate-sum and the
// Wavg triplet of a one-image launch) give a block kTiledRows rows; a translation with a zero weight in all of them
// adds nothing to any of their sums, so each row tile visits only its translations with a nonzero weight, in
// increasing order, listed once per launch by row_tile_active_translations_kernel.
//
// Included from relax_kernels.cu before the kernels that use it.

namespace relax_tiled_translations {

constexpr int kTiledRows = 32;
constexpr int kActiveThreads = 256;

// One block per row tile: one bit per translation in shared memory, then the set bits in increasing order.
// A row counts when it is below the valid count and its image id is 0 (the launch's one image).
__global__ void __launch_bounds__(kActiveThreads)
row_tile_active_translations_kernel(
    const int32_t* __restrict__ row_image_ids,       // [Q]
    const float* __restrict__ posterior,             // [Q, T]
    const int32_t* __restrict__ runtime_valid_rows,  // scalar or null
    int32_t* __restrict__ active_lists,              // [row tiles, T]
    int32_t* __restrict__ active_counts,             // [row tiles]
    int64_t row_count,
    int64_t translation_count)
{
    extern __shared__ uint32_t active_bits[];        // [ceil(T / 32)]
    const int64_t words = (translation_count + 31) / 32;
    for (int64_t w = threadIdx.x; w < words; w += kActiveThreads) active_bits[w] = 0u;
    __syncthreads();
    const int64_t row_base = static_cast<int64_t>(blockIdx.x) * kTiledRows;
    const int64_t valid_rows = runtime_valid_rows == nullptr
        ? row_count
        : static_cast<int64_t>(runtime_valid_rows[0]);
    const int64_t rows = valid_rows < row_count ? valid_rows : row_count;
    for (int64_t index = threadIdx.x; index < kTiledRows * translation_count; index += kActiveThreads) {
        const int64_t i = index / translation_count;
        const int64_t t = index - i * translation_count;
        const int64_t row = row_base + i;
        if (row < rows && row_image_ids[row] == 0 && posterior[row * translation_count + t] != 0.0f)
            atomicOr(&active_bits[t / 32], 1u << (t % 32));
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        int32_t count = 0;
        for (int64_t w = 0; w < words; ++w) {
            const uint32_t bits = active_bits[w];
            if ((bits >> threadIdx.x) & 1u)
                active_lists[static_cast<int64_t>(blockIdx.x) * translation_count + count +
                             __popc(bits & ((1u << threadIdx.x) - 1u))] = static_cast<int32_t>(w * 32 + threadIdx.x);
            count += __popc(bits);
        }
        if (threadIdx.x == 0) active_counts[blockIdx.x] = count;
    }
}

// The lists of a launch's row tiles in one stream-ordered scratch allocation ([tiles, T] lists then [tiles]
// counts); the caller frees ``*scratch`` with cudaFreeAsync after the kernels that read it.
inline cudaError_t list_row_tile_translations(
    cudaStream_t stream, const int32_t* row_image_ids, const float* posterior, const int32_t* runtime_valid_rows,
    int64_t row_count, int64_t translation_count, int32_t** scratch)
{
    const int64_t tiles = (row_count + kTiledRows - 1) / kTiledRows;
    cudaError_t err = cudaMallocAsync(
        reinterpret_cast<void**>(scratch), static_cast<size_t>(tiles * (translation_count + 1)) * sizeof(int32_t),
        stream);
    if (err != cudaSuccess) return err;
    row_tile_active_translations_kernel<<<
        static_cast<unsigned>(tiles), kActiveThreads, static_cast<size_t>((translation_count + 31) / 32) * sizeof(uint32_t),
        stream>>>(row_image_ids, posterior, runtime_valid_rows, *scratch, *scratch + tiles * translation_count,
                  row_count, translation_count);
    return cudaGetLastError();
}

}  // namespace relax_tiled_translations
