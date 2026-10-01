// RELION float32 posterior primitives and FFI handlers.
// Included once at global scope after the shared Ampere scan policy.
// Keep the operation order and CUB reduction topology unchanged.

ffi::Error RelionCubSortScanF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative)
{
    if (values.element_type() != ffi::DataType::F32 ||
        sorted->element_type() != ffi::DataType::F32 ||
        cumulative->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: input and outputs must be F32");

    auto input_dims = values.dimensions();
    auto sorted_dims = sorted->dimensions();
    auto cumulative_dims = cumulative->dimensions();
    if (input_dims.size() != 1 || input_dims[0] < 1 ||
        sorted_dims.size() != 1 || sorted_dims[0] != input_dims[0] ||
        cumulative_dims.size() != 1 || cumulative_dims[0] != input_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: input and outputs must have the same nonempty 1-D shape");

    const int64_t count = input_dims[0];
    if (count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: vector is too large for CUB's item count");

    const float* input_ptr = static_cast<const float*>(values.untyped_data());
    float* sorted_ptr = static_cast<float*>(sorted->untyped_data());
    float* cumulative_ptr = static_cast<float*>(cumulative->untyped_data());
    size_t sort_bytes = 0;
    size_t scan_bytes = 0;
    cudaError_t err = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, input_ptr, sorted_ptr, static_cast<int>(count),
        0, sizeof(float) * 8, stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 sort query: ") + cudaGetErrorString(err));
    err = relion_ampere_inclusive_sum_f32(
        nullptr, scan_bytes, sorted_ptr, cumulative_ptr, static_cast<int>(count), stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 scan query: ") + cudaGetErrorString(err));

    void* temporary = nullptr;
    const size_t temporary_bytes = std::max<size_t>(1, std::max(sort_bytes, scan_bytes));
    err = recovar::scratch_alloc(&temporary, temporary_bytes, stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 cudaMalloc: ") + cudaGetErrorString(err));

    err = cub::DeviceRadixSort::SortKeys(
        temporary, sort_bytes, input_ptr, sorted_ptr, static_cast<int>(count),
        0, sizeof(float) * 8, stream);
    if (err == cudaSuccess)
        err = relion_ampere_inclusive_sum_f32(
            temporary, scan_bytes, sorted_ptr, cumulative_ptr, static_cast<int>(count), stream);
    cudaError_t free_error = recovar::scratch_free(temporary, stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 execute: ") + cudaGetErrorString(err));
    if (free_error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 cudaFree: ") + cudaGetErrorString(free_error));
    return ffi::Error::Success();
}

// Rows of the batched sort are posterior weights: zero or positive, never NaN
// (relion_exponentiate_*_f32 writes 0 below the underflow exponent). Their
// ascending sort is the row's zeros followed by its sorted positive weights, so
// only the positives are radix-sorted, as RELION's filterGreaterZeroOnDevice
// does before sortOnDevice; the zeros are written back in front. The sorted rows
// are the same values as a sort of the whole row, and the scans still run over
// whole rows.
struct RelionBatchedPositiveF32
{
    __device__ __forceinline__ bool operator()(const float& value) const
    {
        return value > 0.0f;
    }
};

// Counts are integer sums, so the per-block atomics give the same counts in any order.
constexpr int kRelionPositiveCountBlock = 256;
// gridDim.y limit: rows beyond it launch again from a row base.
constexpr int kRelionMaxGridRows = 65535;
constexpr int kRelionPositiveCountSpan = kRelionPositiveCountBlock * 16;

__global__ void relion_row_positive_counts_kernel(
    const float* values, int32_t* counts, int count, int row_base)
{
    const int row = row_base + static_cast<int>(blockIdx.y);
    const float* row_values = values + static_cast<int64_t>(row) * count;
    const int begin = static_cast<int>(blockIdx.x) * kRelionPositiveCountSpan;
    const int end = min(count, begin + kRelionPositiveCountSpan);
    int positive = 0;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
        positive += row_values[i] > 0.0f;
    using Reduce = cub::BlockReduce<int, kRelionPositiveCountBlock>;
    __shared__ typename Reduce::TempStorage temp;
    const int total = Reduce(temp).Sum(positive);
    if (threadIdx.x == 0 && total != 0) atomicAdd(counts + row, total);
}

__global__ void relion_zero_fill_right_align_kernel(
    const float* packed, const int32_t* offsets, float* rows_out, int count, int row_base)
{
    const int row = row_base + static_cast<int>(blockIdx.y);
    const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= count) return;
    const int begin = offsets[row];
    const int positive = offsets[row + 1] - begin;
    const int zeros = count - positive;
    rows_out[static_cast<int64_t>(row) * count + column] =
        column < zeros ? 0.0f : packed[begin + (column - zeros)];
}

ffi::Error RelionCubSortScanBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative)
{
    if (values.element_type() != ffi::DataType::F32 ||
        sorted->element_type() != ffi::DataType::F32 ||
        cumulative->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: input and outputs must be F32");

    const auto input_dims = values.dimensions();
    const auto sorted_dims = sorted->dimensions();
    const auto cumulative_dims = cumulative->dimensions();
    if (input_dims.size() != 2 || input_dims[0] < 1 || input_dims[1] < 1 ||
        sorted_dims.size() != 2 || sorted_dims[0] != input_dims[0] ||
        sorted_dims[1] != input_dims[1] ||
        cumulative_dims.size() != 2 || cumulative_dims[0] != input_dims[0] ||
        cumulative_dims[1] != input_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: input and outputs must have the same nonempty 2-D shape");
    if (input_dims[1] > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: row is too large for CUB's item count");

    const int64_t row_count = input_dims[0];
    const int count = static_cast<int>(input_dims[1]);
    const float* input_ptr = static_cast<const float*>(values.untyped_data());
    float* sorted_ptr = static_cast<float*>(sorted->untyped_data());
    float* cumulative_ptr = static_cast<float*>(cumulative->untyped_data());

    // Rows are sorted by one segmented radix sort per group of rows whose
    // cells fit CUB's int item count. Each group's positive weights are
    // selected in order into the cumulative buffer (free until the scans),
    // sorted by row segment into the sorted buffer, and written back right
    // aligned behind the row's zeros through the cumulative buffer. A sort is
    // an exact permutation of its keys, so each row's sorted run is bitwise
    // the per-row cub::DeviceRadixSort::SortKeys result of the whole row; only
    // the launch structure changes. The scans stay one
    // relion_ampere_inclusive_sum_f32 per whole row: its host item count fixes
    // the float32 decoupled-lookback order the significance boundary is defined by.
    const int64_t group_rows = std::max<int64_t>(
        1, std::min<int64_t>(
               row_count,
               static_cast<int64_t>(std::numeric_limits<int>::max()) / count));
    const int group_cells = static_cast<int>(group_rows * count);
    size_t sort_bytes = 0;
    size_t scan_bytes = 0;
    size_t select_bytes = 0;
    size_t offsets_scan_bytes = 0;
    cudaError_t error = cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, sort_bytes, input_ptr, sorted_ptr, group_cells,
        static_cast<int>(group_rows), static_cast<const int32_t*>(nullptr),
        static_cast<const int32_t*>(nullptr), 0, sizeof(float) * 8, stream);
    if (error == cudaSuccess)
        error = cub::DeviceSelect::If(
            nullptr, select_bytes, input_ptr, cumulative_ptr,
            static_cast<int32_t*>(nullptr), group_cells,
            RelionBatchedPositiveF32(), stream);
    if (error == cudaSuccess)
        error = cub::DeviceScan::InclusiveSum(
            nullptr, offsets_scan_bytes, static_cast<const int32_t*>(nullptr),
            static_cast<int32_t*>(nullptr), static_cast<int>(group_rows), stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 sort query: ") +
            cudaGetErrorString(error));
    error = relion_ampere_inclusive_sum_f32(
        nullptr, scan_bytes, sorted_ptr, cumulative_ptr, count, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 scan query: ") +
            cudaGetErrorString(error));

    const size_t cub_bytes = std::max<size_t>(
        1, std::max(std::max(sort_bytes, scan_bytes), std::max(select_bytes, offsets_scan_bytes)));
    const size_t aligned_cub_bytes = (cub_bytes + 255) & ~static_cast<size_t>(255);
    // offsets (group_rows + 1), counts (group_rows) and the selected count.
    const size_t index_bytes = (2 * static_cast<size_t>(group_rows) + 2) * sizeof(int32_t);
    void* temporary = nullptr;
    error = cudaMallocAsync(&temporary, aligned_cub_bytes + index_bytes, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 cudaMallocAsync: ") +
            cudaGetErrorString(error));
    int32_t* offsets = reinterpret_cast<int32_t*>(
        static_cast<char*>(temporary) + aligned_cub_bytes);
    int32_t* counts = offsets + group_rows + 1;
    int32_t* selected = counts + group_rows;

    for (int64_t first = 0; first < row_count && error == cudaSuccess; first += group_rows)
    {
        const int rows = static_cast<int>(std::min<int64_t>(group_rows, row_count - first));
        const int64_t offset = first * static_cast<int64_t>(count);
        // One block per 4096-weight span of a row: a block per row read a
        // K15 row of about 20 M weights alone (17.6 ms per batch, nsys 14693142).
        error = cudaMemsetAsync(counts, 0, static_cast<size_t>(rows) * sizeof(int32_t), stream);
        if (error == cudaSuccess)
        {
            for (int row_base = 0; row_base < rows && error == cudaSuccess; row_base += kRelionMaxGridRows)
            {
                const dim3 count_grid(
                    static_cast<unsigned>((count + kRelionPositiveCountSpan - 1) / kRelionPositiveCountSpan),
                    static_cast<unsigned>(std::min(kRelionMaxGridRows, rows - row_base)));
                relion_row_positive_counts_kernel<<<count_grid, kRelionPositiveCountBlock, 0, stream>>>(
                    input_ptr + offset, counts, count, row_base);
                error = cudaGetLastError();
            }
        }
        if (error == cudaSuccess)
            error = cudaMemsetAsync(offsets, 0, sizeof(int32_t), stream);
        if (error == cudaSuccess)
            error = cub::DeviceScan::InclusiveSum(
                temporary, offsets_scan_bytes, counts, offsets + 1, rows, stream);
        if (error == cudaSuccess)
            error = cub::DeviceSelect::If(
                temporary, select_bytes, input_ptr + offset, cumulative_ptr + offset,
                selected, rows * count, RelionBatchedPositiveF32(), stream);
        if (error == cudaSuccess)
            error = cub::DeviceSegmentedRadixSort::SortKeys(
                temporary, sort_bytes, cumulative_ptr + offset, sorted_ptr + offset,
                rows * count, rows, offsets, offsets + 1, 0, sizeof(float) * 8, stream);
        if (error == cudaSuccess)
        {
            for (int row_base = 0; row_base < rows && error == cudaSuccess; row_base += kRelionMaxGridRows)
            {
                const dim3 grid(
                    static_cast<unsigned>((count + 255) / 256),
                    static_cast<unsigned>(std::min(kRelionMaxGridRows, rows - row_base)));
                relion_zero_fill_right_align_kernel<<<grid, 256, 0, stream>>>(
                    sorted_ptr + offset, offsets, cumulative_ptr + offset, count, row_base);
                error = cudaGetLastError();
            }
        }
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(
                sorted_ptr + offset, cumulative_ptr + offset,
                static_cast<size_t>(rows) * count * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
    }
    for (int64_t row = 0; row < row_count && error == cudaSuccess; ++row)
    {
        const int64_t offset = row * static_cast<int64_t>(count);
        error = relion_ampere_inclusive_sum_f32(
            temporary, scan_bytes, sorted_ptr + offset,
            cumulative_ptr + offset, count, stream);
    }
    const cudaError_t free_error = cudaFreeAsync(temporary, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 execute: ") +
            cudaGetErrorString(error));
    if (free_error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 cudaFreeAsync: ") +
            cudaGetErrorString(free_error));
    return ffi::Error::Success();
}

// Opt-in grouped coarse score-to-support transaction. Keep the existing CUB
// sort and Ampere scan; coarse_publication.py selects this CUDA backend.
// Support capacity is B*N: threshold ties are never truncated to maxsig.
struct CoarsePosteriorFiniteMax {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

__global__ void coarse_posterior_row_max_f32(
    const float* scores, const float* raw_max, const int32_t* actual,
    float* maxima, int rows, int count)
{
    const int row = blockIdx.x;
    float best = -CUDART_INF_F;
    if (row < *actual && *actual <= rows) {
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const float value = __fadd_rn(scores[int64_t(row)*count+i], -raw_max[row]);
            if (isfinite(value)) best = fmaxf(best, value);
        }
    }
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage temp;
    const float result = Reduce(temp).Reduce(best, CoarsePosteriorFiniteMax());
    if (threadIdx.x == 0) maxima[row] = result;
}

__global__ void coarse_posterior_weights_f32(
    const float* scores, const float* raw_max, const float* maxima,
    float* weights, int count, int64_t size)
{
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index >= size) return;
    const int row = index/count;
    const float score = __fadd_rn(scores[index], -raw_max[row]);
    const float best = maxima[row];
    const float add = __fsub_rn(50.0f, isfinite(best) ? best : 0.0f);
    const float exponent = __fadd_rn(score, add);
    weights[index] = !isfinite(score) || !isfinite(best) || exponent < -88.0f
        ? 0.0f : expf(exponent);
}

struct CoarsePosteriorReduction {
    float probability, best_score;
    int winner, best, count;
};

struct CoarsePosteriorCombine {
    __device__ CoarsePosteriorReduction operator()(
        CoarsePosteriorReduction a, CoarsePosteriorReduction b) const
    {
        if (b.probability > a.probability ||
            (b.probability == a.probability && b.winner < a.winner)) {
            a.probability = b.probability; a.winner = b.winner;
        }
        // jnp.argmax picks the first NaN, otherwise the first maximum.
        if ((!isnan(a.best_score) && isnan(b.best_score)) ||
            (!isnan(a.best_score) && !isnan(b.best_score) &&
             (b.best_score > a.best_score ||
              (b.best_score == a.best_score && b.best < a.best))) ||
            (isnan(a.best_score) && isnan(b.best_score) && b.best < a.best)) {
            a.best_score = b.best_score; a.best = b.best;
        }
        a.count += b.count;
        return a;
    }
};

// ---------------------------------------------------------------------------
// RELION's coarse significance cut by radix select (no sort, no scan).
//
// RELION sorts a row's positive weights ascending, scans them in float32 and
// cuts at the first sample whose cumulative weight exceeds
// (1 - adaptive_fraction) * sum. This selects the same sample by its float bits,
// eight bits per pass, from per-bin counts and float64 per-bin masses: the cut
// has RELION's definition with float64 cumulative sums in place of the float32
// scan (whose order is not RELION's either once zeros are kept in the row).
// Positive float32 bit patterns order like the values; zeros are skipped. The
// float64 bin masses are atomic sums, so their last bits are unordered.
// ---------------------------------------------------------------------------

constexpr int kRelionCutBins = 256;
constexpr int kRelionCutBlock = 256;
constexpr int kRelionCutWarps = kRelionCutBlock / 32;
constexpr int kRelionCutSpan = 65536;

// One pass of the select: bins of bits [24 - 8 pass, 32 - 8 pass) of the weights
// whose higher bits equal the row's prefix. ``with_mass`` adds the float64 mass.
__global__ void relion_coarse_cut_histogram_kernel(
    const float* weights, int count, int pass, const uint32_t* prefix,
    const int32_t* active, bool with_mass, uint32_t* bin_count, double* bin_mass, int row_base)
{
    const int row = row_base + static_cast<int>(blockIdx.y);
    if (active != nullptr && active[row] == 0) return;
    __shared__ uint32_t counts[kRelionCutWarps][kRelionCutBins];
    __shared__ double masses[kRelionCutWarps][kRelionCutBins];
    const int warp = threadIdx.x / 32;
    for (int i = threadIdx.x; i < kRelionCutWarps * kRelionCutBins; i += blockDim.x) {
        counts[i / kRelionCutBins][i % kRelionCutBins] = 0u;
        masses[i / kRelionCutBins][i % kRelionCutBins] = 0.0;
    }
    __syncthreads();
    const float* row_weights = weights + static_cast<int64_t>(row) * count;
    const int begin = static_cast<int>(blockIdx.x) * kRelionCutSpan;
    const int end = min(count, begin + kRelionCutSpan);
    const int shift = 24 - 8 * pass;
    const uint32_t row_prefix = prefix[row];
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float weight = row_weights[i];
        const uint32_t bits = __float_as_uint(weight);
        if (!(weight > 0.0f)) continue;
        if (pass > 0 && (bits >> (shift + 8)) != row_prefix) continue;
        const uint32_t bin = (bits >> shift) & 0xffu;
        atomicAdd(&counts[warp][bin], 1u);
        if (with_mass) atomicAdd(&masses[warp][bin], static_cast<double>(weight));
    }
    __syncthreads();
    for (int bin = threadIdx.x; bin < kRelionCutBins; bin += blockDim.x) {
        uint32_t c = 0u;
        double m = 0.0;
        for (int w = 0; w < kRelionCutWarps; ++w) {
            c += counts[w][bin];
            m += masses[w][bin];
        }
        if (c == 0u) continue;
        atomicAdd(bin_count + static_cast<int64_t>(row) * kRelionCutBins + bin, c);
        if (with_mass) atomicAdd(bin_mass + static_cast<int64_t>(row) * kRelionCutBins + bin, m);
    }
}

// Per row: choose the bin holding the cut sample and narrow the prefix. On the
// first pass the row's sum and RELION's float32 tail target are formed. When the
// target is not reached (float32 target above the float64 sum) the cut is the
// largest weight, as RELION's threshold index clamps to the last sample.
__global__ void relion_coarse_cut_mass_decide_kernel(
    int rows, int pass, float fraction, const uint32_t* bin_count, const double* bin_mass,
    uint32_t* prefix, double* low, double* target, int32_t* above,
    float* sum_weight, float* threshold, int32_t* cutoff)
{
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t* counts = bin_count + static_cast<int64_t>(row) * kRelionCutBins;
    const double* masses = bin_mass + static_cast<int64_t>(row) * kRelionCutBins;
    if (pass == 0) {
        double total = 0.0;
        for (int b = 0; b < kRelionCutBins; ++b) total += masses[b];
        const float total_f32 = static_cast<float>(total);
        sum_weight[row] = total_f32;
        // _relion_cuda_f32_tail_target: (1 - textToFloat(fraction)) * sum, narrowed to XFLOAT.
        target[row] = static_cast<double>(static_cast<float>(
            (1.0 - static_cast<double>(fraction)) * static_cast<double>(total_f32)));
        low[row] = 0.0;
        above[row] = 0;
        prefix[row] = 0u;
    }
    const double tail = target[row];
    double cumulative = low[row];
    int chosen = -1, last = -1;
    for (int b = 0; b < kRelionCutBins; ++b) {
        if (counts[b] == 0u) continue;
        last = b;
        if (cumulative + masses[b] > tail) { chosen = b; break; }
        cumulative += masses[b];
    }
    if (chosen < 0) {
        if (last < 0) {  // no positive weight: has_mass is false downstream
            threshold[row] = 0.0f;
            cutoff[row] = 0;
            prefix[row] = 0xffffffffu;
            return;
        }
        chosen = last;
        cumulative -= masses[last];
    }
    int32_t greater = above[row];
    for (int b = chosen + 1; b < kRelionCutBins; ++b) greater += static_cast<int32_t>(counts[b]);
    above[row] = greater;
    low[row] = cumulative;
    const uint32_t narrowed = (prefix[row] << 8) | static_cast<uint32_t>(chosen);
    prefix[row] = narrowed;
    if (pass < 3) return;
    // The final bin holds the samples equal to the cut value. The cut is the first
    // of them, in ascending order, whose cumulative weight exceeds the target.
    const float value = __uint_as_float(narrowed);
    const int64_t ties = static_cast<int64_t>(counts[chosen]);
    const double estimate = floor((tail - cumulative) / static_cast<double>(value)) + 1.0;
    int64_t first = estimate >= static_cast<double>(ties) ? ties : (estimate <= 1.0 ? 1 : static_cast<int64_t>(estimate));
    while (first > 1 && cumulative + static_cast<double>(first - 1) * value > tail) --first;
    while (first < ties && cumulative + static_cast<double>(first) * value <= tail) ++first;
    threshold[row] = value;
    cutoff[row] = static_cast<int32_t>(greater + ties - first + 1);
}

// RELION's maximum_significants floor on the threshold index: rows whose cut keeps
// more than ``maxsig`` samples cut at the maxsig-th largest weight instead.
__global__ void relion_coarse_cut_rank_start_kernel(
    int rows, int64_t maxsig, const int32_t* cutoff, int32_t* active, uint32_t* prefix, int32_t* above)
{
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    active[row] = cutoff[row] > maxsig ? 1 : 0;
    prefix[row] = 0u;
    above[row] = 0;
}

__global__ void relion_coarse_cut_rank_decide_kernel(
    int rows, int pass, int64_t maxsig, const int32_t* active, const uint32_t* bin_count,
    uint32_t* prefix, int32_t* above, float* threshold, int32_t* cutoff)
{
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || active[row] == 0) return;
    const uint32_t* counts = bin_count + static_cast<int64_t>(row) * kRelionCutBins;
    int64_t greater = above[row];
    int chosen = 0;
    for (int b = kRelionCutBins - 1; b >= 0; --b) {
        if (greater + counts[b] >= maxsig) { chosen = b; break; }
        greater += counts[b];
    }
    above[row] = static_cast<int32_t>(greater);
    prefix[row] = (prefix[row] << 8) | static_cast<uint32_t>(chosen);
    if (pass < 3) return;
    threshold[row] = __uint_as_float(prefix[row]);
    cutoff[row] = static_cast<int32_t>(maxsig);
}

// Writes each row's float32 positive-weight sum, cut weight and pre-tie rank (RELION's
// cutoff count, with the maximum_significants floor when maxsig > 0) on ``stream``.
cudaError_t relion_coarse_cut_f32_launch(
    cudaStream_t stream, const float* input, int rows, int count, float fraction, int64_t maxsig,
    float* sums, float* cuts, int32_t* counts_out)
{
    const size_t bins = static_cast<size_t>(rows) * kRelionCutBins;
    const size_t bytes = bins * (sizeof(double) + sizeof(uint32_t)) +
                         static_cast<size_t>(rows) * (2 * sizeof(double) + 3 * sizeof(int32_t));
    void* storage = nullptr;
    cudaError_t error = cudaMallocAsync(&storage, bytes, stream);
    if (error != cudaSuccess) return error;
    double* bin_mass = static_cast<double*>(storage);
    double* low = bin_mass + bins;
    double* target = low + rows;
    uint32_t* bin_count = reinterpret_cast<uint32_t*>(target + rows);
    uint32_t* prefix = bin_count + bins;
    int32_t* above = reinterpret_cast<int32_t*>(prefix + rows);
    int32_t* active = above + rows;
    const unsigned row_blocks = static_cast<unsigned>((rows + 127) / 128);
    const unsigned spans = static_cast<unsigned>((count + kRelionCutSpan - 1) / kRelionCutSpan);

    auto histogram = [&](int pass, const int32_t* row_active, bool with_mass) {
        cudaError_t e = cudaMemsetAsync(bin_count, 0, bins * sizeof(uint32_t), stream);
        if (e == cudaSuccess && with_mass) e = cudaMemsetAsync(bin_mass, 0, bins * sizeof(double), stream);
        for (int row_base = 0; row_base < rows && e == cudaSuccess; row_base += kRelionMaxGridRows) {
            const dim3 grid(spans, static_cast<unsigned>(std::min(kRelionMaxGridRows, rows - row_base)));
            relion_coarse_cut_histogram_kernel<<<grid, kRelionCutBlock, 0, stream>>>(
                input, count, pass, prefix, row_active, with_mass, bin_count, bin_mass, row_base);
            e = cudaGetLastError();
        }
        return e;
    };
    for (int pass = 0; pass < 4 && error == cudaSuccess; ++pass) {
        // Pass 0 reads no prefix; it is written by the pass-0 decision.
        if (pass == 0) error = cudaMemsetAsync(prefix, 0, static_cast<size_t>(rows) * sizeof(uint32_t), stream);
        if (error == cudaSuccess) error = histogram(pass, nullptr, true);
        if (error == cudaSuccess) {
            relion_coarse_cut_mass_decide_kernel<<<row_blocks, 128, 0, stream>>>(
                rows, pass, fraction, bin_count, bin_mass, prefix, low, target, above, sums, cuts, counts_out);
            error = cudaGetLastError();
        }
    }
    if (maxsig > 0 && error == cudaSuccess) {
        relion_coarse_cut_rank_start_kernel<<<row_blocks, 128, 0, stream>>>(
            rows, maxsig, counts_out, active, prefix, above);
        error = cudaGetLastError();
        for (int pass = 0; pass < 4 && error == cudaSuccess; ++pass) {
            error = histogram(pass, active, false);
            if (error == cudaSuccess) {
                relion_coarse_cut_rank_decide_kernel<<<row_blocks, 128, 0, stream>>>(
                    rows, pass, maxsig, active, bin_count, prefix, above, cuts, counts_out);
                error = cudaGetLastError();
            }
        }
    }
    const cudaError_t free_error = cudaFreeAsync(storage, stream);
    return error == cudaSuccess ? free_error : error;
}

// The cut (sum, cut weight, pre-tie rank) comes from relion_coarse_cut_f32_launch.
__global__ void coarse_posterior_finish_f32(
    const float* scores, const float* weights, const float* sums,
    const float* cuts, const int32_t* cutoffs, const float* maxima, const int32_t* actual,
    uint8_t* mask, float* statistics, int32_t* indices, int rows, int count)
{
    const int row = blockIdx.x;
    const int64_t offset = int64_t(row)*count;
    __shared__ float total, threshold;
    __shared__ int has_mass, cutoff;
    if (threadIdx.x == 0) {
        total = sums[row];
        has_mass = row < *actual && *actual > 0 && *actual <= rows &&
            isfinite(maxima[row]) && isfinite(total) && total > 0.0f;
        threshold = cuts[row];
        cutoff = has_mass ? cutoffs[row] : 0;
    }
    __syncthreads();
    CoarsePosteriorReduction value{0.0f, -CUDART_INF_F, count, count, 0};
    const CoarsePosteriorCombine combine;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float weight = weights[offset+i];
        const bool significant = has_mass && weight > 0.0f && weight >= threshold;
        mask[offset+i] = significant;
        // Preserve float/float division even where rounding creates a tie.
        const float probability = has_mass ? __fdiv_rn(weight, total) : 0.0f;
        value = combine(value, {probability, scores[offset+i], i, i, int(significant)});
    }
    using Reduce = cub::BlockReduce<CoarsePosteriorReduction, 256>;
    __shared__ typename Reduce::TempStorage temp;
    const auto result = Reduce(temp).Reduce(value, combine);
    if (threadIdx.x == 0) {
        const bool valid_count = *actual > 0 && *actual <= rows;
        statistics[4*row+0] = valid_count ? result.best_score : CUDART_NAN_F;
        statistics[4*row+1] = result.probability;
        statistics[4*row+2] = total;
        statistics[4*row+3] = threshold;
        indices[4*row+0] = result.best;
        indices[4*row+1] = result.winner;
        indices[4*row+2] = result.count;
        indices[4*row+3] = cutoff;
    }
}

ffi::Error RelionCoarsePosteriorTransactionF32Impl(
    cudaStream_t stream, ffi::AnyBuffer scores, ffi::AnyBuffer raw_max,
    ffi::AnyBuffer actual, float fraction, int64_t maxsig,
    ffi::Result<ffi::AnyBuffer> statistics, ffi::Result<ffi::AnyBuffer> indices,
    ffi::Result<ffi::AnyBuffer> support, ffi::Result<ffi::AnyBuffer> support_count)
{
    const auto dims = scores.dimensions();
    if (scores.element_type() != ffi::DataType::F32 || dims.size() != 2 ||
        dims[0] < 1 || dims[1] < 1 || dims[0] > INT_MAX/dims[1] ||
        raw_max.element_type() != ffi::DataType::F32 ||
        raw_max.dimensions().size() != 1 || raw_max.dimensions()[0] != dims[0] ||
        actual.element_type() != ffi::DataType::S32 || actual.dimensions().size() != 0 ||
        !std::isfinite(fraction) || fraction <= 0 || fraction > 1 || maxsig < 0 || maxsig > INT_MAX)
        return ffi::Error::InvalidArgument("Coarse posterior: invalid scores, maxima, count or policy");
    const int rows = dims[0], count = dims[1], size = rows*count;
    if (statistics->element_type() != ffi::DataType::F32 ||
        indices->element_type() != ffi::DataType::S32 ||
        support->element_type() != ffi::DataType::S32 ||
        support_count->element_type() != ffi::DataType::S32 ||
        statistics->dimensions().size() != 2 || statistics->dimensions()[0] != rows || statistics->dimensions()[1] != 4 ||
        indices->dimensions().size() != 2 || indices->dimensions()[0] != rows || indices->dimensions()[1] != 4 ||
        support->dimensions().size() != 1 || support->dimensions()[0] != size ||
        support_count->dimensions().size() != 0)
        return ffi::Error::InvalidArgument("Coarse posterior: invalid output shapes/dtypes");
    // One score-size weight array, the per-row cut and flags; no host reads or host
    // stream synchronization. All work and temporary lifetimes use the FFI stream.
    void* storage = nullptr;
    cudaError_t error = cudaMallocAsync(&storage, size_t(size)*5+size_t(rows)*16, stream);
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    float* weights = static_cast<float*>(storage);
    float* maxima = weights+size;
    float* sums = maxima+rows;
    float* cuts = sums+rows;
    int32_t* cutoffs = reinterpret_cast<int32_t*>(cuts+rows);
    uint8_t* mask = reinterpret_cast<uint8_t*>(cutoffs+rows);
    auto* out_support = static_cast<int32_t*>(support->untyped_data());
    auto* out_count = static_cast<int32_t*>(support_count->untyped_data());
    thrust::counting_iterator<int32_t> positions(0);
    size_t select_bytes=0;
    error = cub::DeviceSelect::Flagged(nullptr, select_bytes, positions, mask, out_support, out_count, size, stream);
    void* temporary = nullptr;
    if (error == cudaSuccess) error = cudaMallocAsync(&temporary, std::max(size_t(1), select_bytes), stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(out_support, 0xff, size_t(size)*sizeof(int32_t), stream);
    if (error == cudaSuccess) {
        coarse_posterior_row_max_f32<<<rows,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), static_cast<const float*>(raw_max.untyped_data()),
            static_cast<const int32_t*>(actual.untyped_data()), maxima, rows, count);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        coarse_posterior_weights_f32<<<(int64_t(size)+255)/256,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), static_cast<const float*>(raw_max.untyped_data()), maxima, weights, count, size);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess)
        error = relion_coarse_cut_f32_launch(stream, weights, rows, count, fraction, maxsig, sums, cuts, cutoffs);
    if (error == cudaSuccess) {
        coarse_posterior_finish_f32<<<rows,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), weights, sums, cuts, cutoffs, maxima,
            static_cast<const int32_t*>(actual.untyped_data()), mask,
            static_cast<float*>(statistics->untyped_data()), static_cast<int32_t*>(indices->untyped_data()),
            rows, count);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cub::DeviceSelect::Flagged(temporary, select_bytes, positions, mask, out_support, out_count, size, stream);
    const auto free_temp = temporary ? cudaFreeAsync(temporary, stream) : cudaSuccess;
    const auto free_storage = cudaFreeAsync(storage, stream);
    if (error == cudaSuccess) error = free_temp;
    if (error == cudaSuccess) error = free_storage;
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarsePosteriorTransactionF32, RelionCoarsePosteriorTransactionF32Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Attr<float>("fraction").Attr<int64_t>("maxsig")
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
);

__global__ void relion_exponentiate_f32_kernel(
    const float* values,
    const float* add,
    float* output,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float exponent = values[index] + add[0];
    output[index] = exponent < -88.0f ? 0.0f : expf(exponent);
}

ffi::Error RelionExponentiateF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer add,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        add.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionExponentiateF32: inputs and output must be F32");

    auto value_dims = values.dimensions();
    auto add_dims = add.dimensions();
    auto output_dims = output->dimensions();
    if (value_dims.size() != 1 || value_dims[0] < 1 ||
        add_dims.size() != 0 || output_dims.size() != 1 ||
        output_dims[0] != value_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionExponentiateF32: values/output must be matching nonempty 1-D arrays and add a scalar");

    const int64_t count = value_dims[0];
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    relion_exponentiate_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(values.untyped_data()),
        static_cast<const float*>(add.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionExponentiateF32: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionExponentiateF32, RelionExponentiateF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_exponentiate_batched_f32_kernel(
    const float* values,
    const float* add,
    float* output,
    int64_t row_size,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float exponent = values[index] + add[index / row_size];
    output[index] = exponent < -88.0f ? 0.0f : expf(exponent);
}

ffi::Error RelionExponentiateBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer add,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        add.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: inputs and output must be F32");

    const auto value_dims = values.dimensions();
    const auto add_dims = add.dimensions();
    const auto output_dims = output->dimensions();
    if (value_dims.size() != 2 || value_dims[0] < 1 || value_dims[1] < 1 ||
        add_dims.size() != 1 || add_dims[0] != value_dims[0] ||
        output_dims.size() != 2 || output_dims[0] != value_dims[0] ||
        output_dims[1] != value_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: expected values/output (B,N) and add (B,)");

    const int64_t count = value_dims[0] * value_dims[1];
    constexpr int threads = 256;
    const int64_t block_count = (count + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: launch grid exceeds CUDA limit");
    relion_exponentiate_batched_f32_kernel<<<
        static_cast<int>(block_count), threads, 0, stream>>>(
            static_cast<const float*>(values.untyped_data()),
            static_cast<const float*>(add.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            value_dims[1], count);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionExponentiateBatchedF32: ") +
            cudaGetErrorString(error));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionExponentiateBatchedF32, RelionExponentiateBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_divide_f32_kernel(
    const float* values,
    const float* divisor,
    float* output,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = values[index] / divisor[0];
}

ffi::Error RelionDivideF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer divisor,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        divisor.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionDivideF32: inputs and output must be F32");

    auto value_dims = values.dimensions();
    auto divisor_dims = divisor.dimensions();
    auto output_dims = output->dimensions();
    if (value_dims.size() != 1 || value_dims[0] < 1 ||
        divisor_dims.size() != 0 || output_dims.size() != 1 ||
        output_dims[0] != value_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionDivideF32: values/output must be matching nonempty 1-D arrays and divisor a scalar");

    const int64_t count = value_dims[0];
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    relion_divide_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(values.untyped_data()),
        static_cast<const float*>(divisor.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionDivideF32: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionDivideF32, RelionDivideF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_divide_batched_f32_kernel(
    const float* values,
    const float* divisor,
    float* output,
    int64_t row_size,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = values[index] / divisor[index / row_size];
}

ffi::Error RelionDivideBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer divisor,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        divisor.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: inputs and output must be F32");

    const auto value_dims = values.dimensions();
    const auto divisor_dims = divisor.dimensions();
    const auto output_dims = output->dimensions();
    if (value_dims.size() != 2 || value_dims[0] < 1 || value_dims[1] < 1 ||
        divisor_dims.size() != 1 || divisor_dims[0] != value_dims[0] ||
        output_dims.size() != 2 || output_dims[0] != value_dims[0] ||
        output_dims[1] != value_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: expected values/output (B,N) and divisor (B,)");

    const int64_t count = value_dims[0] * value_dims[1];
    constexpr int threads = 256;
    const int64_t block_count = (count + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: launch grid exceeds CUDA limit");
    relion_divide_batched_f32_kernel<<<
        static_cast<int>(block_count), threads, 0, stream>>>(
            static_cast<const float*>(values.untyped_data()),
            static_cast<const float*>(divisor.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            value_dims[1], count);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionDivideBatchedF32: ") +
            cudaGetErrorString(error));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionDivideBatchedF32, RelionDivideBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCubSortScanF32, RelionCubSortScanF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCubSortScanBatchedF32, RelionCubSortScanBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);



ffi::Error RelionCoarseCutF32Impl(
    cudaStream_t stream, ffi::AnyBuffer weights, float fraction, int64_t maxsig,
    ffi::Result<ffi::AnyBuffer> sum_weight, ffi::Result<ffi::AnyBuffer> threshold,
    ffi::Result<ffi::AnyBuffer> cutoff)
{
    const auto dims = weights.dimensions();
    if (weights.element_type() != ffi::DataType::F32 || dims.size() != 2 || dims[0] < 1 ||
        dims[1] < 1 || dims[1] > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        !std::isfinite(fraction) || fraction <= 0.0f || fraction > 1.0f)
        return ffi::Error::InvalidArgument(
            "RelionCoarseCutF32: weights must be a nonempty F32 matrix and 0 < fraction <= 1");
    const int rows = static_cast<int>(dims[0]);
    const int count = static_cast<int>(dims[1]);
    for (auto* out : {&sum_weight, &threshold, &cutoff}) {
        const auto out_dims = (*out)->dimensions();
        if (out_dims.size() != 1 || out_dims[0] != rows)
            return ffi::Error::InvalidArgument("RelionCoarseCutF32: outputs must have one value per row");
    }
    if (sum_weight->element_type() != ffi::DataType::F32 ||
        threshold->element_type() != ffi::DataType::F32 ||
        cutoff->element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument("RelionCoarseCutF32: outputs are F32, F32, S32");
    const cudaError_t error = relion_coarse_cut_f32_launch(
        stream, static_cast<const float*>(weights.untyped_data()), rows, count, fraction, maxsig,
        static_cast<float*>(sum_weight->untyped_data()), static_cast<float*>(threshold->untyped_data()),
        static_cast<int32_t*>(cutoff->untyped_data()));
    return error == cudaSuccess ? ffi::Error::Success()
                                : ffi::Error::Internal(std::string("RelionCoarseCutF32: ") + cudaGetErrorString(error));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseCutF32, RelionCoarseCutF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Attr<float>("fraction")
        .Attr<int64_t>("maxsig")
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);
