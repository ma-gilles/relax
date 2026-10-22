// Fused sparse pass-2 posterior for RECOVAR's RELION-parity K=1 path.
//
// Runtime-shaped XLA FFI handlers for the device-resident pass-2 data model,
// in which image i owns a contiguous run (a segment) of a flat cell array:
//
//   SparsePass2SegmentedLogZF64      <-> _logsumexp_pass2_bucket_score_only per segment
//   SparsePass2SegmentedPosteriorF32 <-> _normalize_pass2_bucket_with_log_z (now in
//                                        scripts/lib/bucketed_pass2.py) followed by
//                                        relion_f32_fine_probabilities, per segment
//
// Every per-element formula is copied from the XLA path.  Order-independent
// reductions (max, argmax with smallest index, mask counts) are exact.  The
// RELION significance boundary is relion_coarse_cut_f32's definition (RELION's
// cut with float64 cumulative sums, by radix select over each segment's cells),
// with no sort, scan or host round trip.  The float64 bin
// masses and the log-sum-exp of the log-Z handler are the order-dependent sums.
// The per-row bodies below (RowState and the *_body helpers) are shared by these
// handlers; the rectangular-row handlers that used them as well were removed.
//
//   SparsePass2SegmentedReconstructionF32 <-> SparsePass2SegmentedPosteriorF32
//                                             without probs, normalized and mask
//
// runs the same kernels with the per-cell outputs only the reconstruction
// needs (relax#118: a particle with 3.5e8 cells), writing reconstruction and
// every per-segment output bit for bit as the full handler does.

namespace recovar_sparse_pass2_posterior {

constexpr int kRowThreads = 1024;

__device__ __forceinline__ bool better_candidate(
    float value, int64_t index, float best_value, int64_t best_index)
{
    // jnp.argmax picks the smallest index among equal maxima.
    return value > best_value || (value == best_value && index < best_index);
}

// Block-wide finite max with smallest-index argmax.  All threads must call it.
__device__ void block_finite_argmax(
    const float* row, int64_t n, float* shared_values, int64_t* shared_indices,
    float& best_value, int64_t& best_index)
{
    float local_best = -CUDART_INF_F;
    int64_t local_index = INT64_MAX;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = row[i];
        if (isfinite(v) && better_candidate(v, i, local_best, local_index))
        {
            local_best = v;
            local_index = i;
        }
    }
    shared_values[threadIdx.x] = local_best;
    shared_indices[threadIdx.x] = local_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            const float other = shared_values[threadIdx.x + stride];
            const int64_t other_index = shared_indices[threadIdx.x + stride];
            if (better_candidate(other, other_index, shared_values[threadIdx.x],
                                 shared_indices[threadIdx.x]))
            {
                shared_values[threadIdx.x] = other;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    best_value = shared_values[0];
    best_index = shared_indices[0];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Log-Z: the log-sum-exp of one row (segment) of float32 scores in float64.
// ---------------------------------------------------------------------------
// The per-row float64 log-sum-exp; the segmented kernel inlines it on each
// segment's slice.
__device__ __forceinline__ void log_z_body(
    const float* row, int64_t n, double* log_z_slot,
    float* shared_values, int64_t* shared_indices, double* shared_sums)
{
    float best;
    int64_t best_index;
    block_finite_argmax(row, n, shared_values, shared_indices, best, best_index);
    const bool has_finite = isfinite(best);
    // XLA: shifted = where(has_finite, scores - safe_best, -inf) in float32,
    // then exp in float64; non-finite scores were replaced by -inf first.
    const float safe_best = has_finite ? best : 0.0f;
    double local_sum = 0.0;
    if (has_finite)
    {
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        {
            const float v = row[i];
            const float shifted = isfinite(v) ? (v - safe_best) : -CUDART_INF_F;
            const double term = exp(static_cast<double>(shifted));
            local_sum += isfinite(term) ? term : 0.0;
        }
    }
    shared_sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        const double sum_exp = shared_sums[0];
        const bool has_mass = has_finite && sum_exp > 0.0 && isfinite(sum_exp);
        *log_z_slot = has_mass
            ? static_cast<double>(safe_best) + log(sum_exp)
            : -CUDART_INF;
    }
}


// ---------------------------------------------------------------------------
// Posterior: fused normalization + RELION float32 fine posterior, per row (segment).
// ---------------------------------------------------------------------------
struct RowState
{
    float best;            // finite max or -inf
    float exponent_add;    // 50 - safe_best (RELION: 50 - weights_max)
    double safe_log_z;     // external log-Z when finite together with best, else 0
    float safe_sum_weight; // sum_weight when has_mass, else 1
    float threshold;       // RELION significance cutoff
    int has_finite;        // isfinite(best)
    int has_finite_norm;   // isfinite(best) && isfinite(log_z)
    int has_mass;          // has_finite && isfinite(sum_weight) && sum_weight > 0
    int keep_all;
};

// Shared body of the row-max / normalization state.  Every thread of the block
// must call it; only thread 0 writes.
__device__ __forceinline__ void row_max_body(
    const float* row, int64_t n, double external_log_z,
    float* shared_values, int64_t* shared_indices,
    RowState* state_slot, double* log_z_slot, float* best_log_score_slot,
    int64_t* best_argmax_slot)
{
    float best;
    int64_t best_index;
    block_finite_argmax(row, n, shared_values, shared_indices, best, best_index);
    if (threadIdx.x != 0) return;
    const bool has_finite = isfinite(best);
    const bool has_finite_norm = has_finite && isfinite(external_log_z);
    RowState s;
    s.best = best;
    s.exponent_add = 50.0f - (has_finite ? best : 0.0f);
    s.safe_log_z = has_finite_norm ? external_log_z : 0.0;
    s.safe_sum_weight = 1.0f;
    s.threshold = 0.0f;
    s.has_finite = has_finite;
    s.has_finite_norm = has_finite_norm;
    s.has_mass = 0;
    s.keep_all = 0;
    *state_slot = s;
    // _normalize_pass2_bucket_with_log_z outputs.
    *log_z_slot = s.safe_log_z;
    *best_log_score_slot = has_finite_norm ? best : -CUDART_INF_F;
    *best_argmax_slot = has_finite_norm ? best_index : 0;
}


// Shared per-cell exponentiation of one candidate score.  kProb = false skips
// the float64 probability (the reconstruction-only handler has none).
template <bool kProb = true>
__device__ __forceinline__ void exponentiate_cell(
    const RowState& s, float value, float* raw_slot, double* prob_slot)
{
    const bool finite = isfinite(value);
    const float finite_score = finite ? value : -CUDART_INF_F;
    // relion_exponentiate_batched_f32_kernel: expf(value + add), zero below -88.
    const float exponent = finite_score + s.exponent_add;
    *raw_slot = exponent < -88.0f ? 0.0f : expf(exponent);
    if (!kProb) return;
    // _normalize_pass2_bucket_with_log_z: exp(scores - safe_log_z) in float64,
    // zero when the row has no finite normalizer or the value overflows.
    double p = 0.0;
    if (s.has_finite_norm)
    {
        p = exp(static_cast<double>(finite_score) - s.safe_log_z);
        if (!isfinite(p)) p = 0.0;
    }
    *prob_slot = p;
}


// RELION's significance cut of one row or segment's raw weights, by radix select over the
// positive weights' float bits (relion_coarse_cut_f32_launch's definition, one block per row):
// eight bits per pass from per-bin counts and float64 masses, then the first weight of the
// cut value's tied run whose cumulative weight exceeds the float32 tail target. Every thread
// of the block must call it; the results are valid in every thread on return.
struct BlockCut
{
    uint32_t counts[256];
    double masses[256];
    double low;
    double target;
    double total;
    uint32_t prefix;
    int empty;
    float cut;
};

__device__ __forceinline__ void radix_cut_reset(BlockCut& c)
{
    c.prefix = 0u; c.low = 0.0; c.empty = 0; c.cut = 0.0f; c.total = 0.0;
}

// One radix pass's histogram term of a weight: its bin, or -1 when the weight is not positive or lies outside
// the bins the earlier passes chose.
__device__ __forceinline__ int radix_cut_bin(float weight, int pass, uint32_t prefix)
{
    if (!(weight > 0.0f)) return -1;
    const uint32_t bits = __float_as_uint(weight);
    const int shift = 24 - 8 * pass;
    if (pass > 0 && (bits >> (shift + 8)) != prefix) return -1;
    return static_cast<int>((bits >> shift) & 0xffu);
}

// One radix pass's choice from its histogram (one thread): the bin holding the tail target, the mass below it,
// and after the fourth pass the cut value.
__device__ void radix_cut_select(BlockCut& c, int pass, float fraction)
{
    if (c.empty) return;
    if (pass == 0)
    {
        double total = 0.0;
        for (int b = 0; b < 256; ++b) total += c.masses[b];
        c.total = total;
        // relion_cuda_f32_tail_target on the float32 sum.
        c.target = static_cast<double>(static_cast<float>(
            (1.0 - static_cast<double>(fraction)) * static_cast<double>(static_cast<float>(total))));
    }
    double cumulative = c.low;
    int chosen = -1, last = -1;
    for (int b = 0; b < 256; ++b)
    {
        if (c.counts[b] == 0u) continue;
        last = b;
        if (cumulative + c.masses[b] > c.target) { chosen = b; break; }
        cumulative += c.masses[b];
    }
    if (chosen < 0 && last < 0) c.empty = 1;
    else
    {
        if (chosen < 0) { chosen = last; cumulative -= c.masses[last]; }
        c.low = cumulative;
        c.prefix = (c.prefix << 8) | static_cast<uint32_t>(chosen);
        if (pass == 3)
        {
            // The cut value is reached inside its tied run (the threshold, as a value).
            c.cut = __uint_as_float(c.prefix);
        }
    }
}

// The rest of RELION's boundary: the normalizing sum (the coarse pass's, or this pass's own)
// and the row state the normalization reads. ``n == 0`` is an empty segment.
__device__ __forceinline__ void cut_body(
    const BlockCut& c, int64_t n, float external_sum_weight, int keep_all,
    int use_external_sum_weight, RowState& s, float* sum_weight_slot, float* threshold_slot)
{
    const float fine_sum_weight = n > 0 ? static_cast<float>(c.total) : 0.0f;
    const float sum_weight = use_external_sum_weight ? external_sum_weight : fine_sum_weight;
    const bool has_mass = s.has_finite && isfinite(sum_weight) && sum_weight > 0.0f;
    const float threshold = (!keep_all && n > 0) ? c.cut : 0.0f;
    s.has_mass = has_mass;
    s.safe_sum_weight = has_mass ? sum_weight : 1.0f;
    s.threshold = threshold;
    s.keep_all = keep_all;
    *sum_weight_slot = sum_weight;
    *threshold_slot = threshold;
}

constexpr int kCutThreads = 256;


// Shared normalization / pruning body.  All pointers are already offset to the
// start of the row or segment; every thread of the block must call it.
// kCells = false writes only the reconstruction of the per-cell outputs.
template <bool kCells = true>
__device__ __forceinline__ void normalize_body(
    const float* row_scores, const float* row_raw, const RowState& s, int64_t n,
    float* row_normalized, float* row_reconstruction, bool* row_mask,
    int32_t* n_significant_slot, float* max_posterior_slot,
    int* shared_counts, float* shared_max)
{
    int local_count = 0;
    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float raw = row_raw[i];
        // relion_divide_f32_kernel: IEEE float32 division by the row sum.
        const float weight = raw / s.safe_sum_weight;
        const bool finite = isfinite(row_scores[i]);
        const bool significant = s.has_mass && finite &&
            (s.keep_all ? (raw > 0.0f) : (raw >= s.threshold));
        const float kept = significant ? weight : 0.0f;
        if (kCells)
        {
            row_normalized[i] = weight;
            row_mask[i] = significant;
        }
        row_reconstruction[i] = kept;
        local_count += significant ? 1 : 0;
        local_max = kept > local_max ? kept : local_max;
    }
    shared_counts[threadIdx.x] = local_count;
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
            const float other = shared_max[threadIdx.x + stride];
            shared_max[threadIdx.x] = other > shared_max[threadIdx.x] ? other : shared_max[threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        *n_significant_slot = shared_counts[0];
        *max_posterior_slot = shared_max[0];
    }
}


// ---------------------------------------------------------------------------
// Segmented variants.
//
// The segmented handlers take the scores as one flat array of cells in which
// image i owns the contiguous run [segment_offsets[i], segment_offsets[i + 1]).
// Cells inside a run are the CSR rows x translations of the device-resident
// pass-2 data model, so a chunk does not pad every image to a common row count.
//
// Every formula is one of the per-row bodies above, called on the segment's
// slice.  The significance boundary is a radix select on the device
// (segmented_cut_*_kernel), so the handler has no host round trip, and its launch
// geometry is capacity-only: each kernel is launched at the static segment
// count and n_valid_images is read only on the device.
//
// Empty segments and images at or beyond n_valid_images produce exactly the
// values the per-row bodies produce for an all -inf row.  Cells covered by
// no segment are written as zero / false.
// ---------------------------------------------------------------------------

struct SegmentState
{
    RowState row;
    int valid;      // segment index < n_valid_images
    int reserved;   // explicit padding, keeps begin 8-byte aligned
    int64_t begin;  // first cell of the segment, clamped into [0, n_cells]
    int64_t n;      // cell count, 0 for empty or invalid segments
    BlockCut cut;   // the significance cut's histogram and state (segmented_cut_*_kernel)
};
// The Python wrapper's _SPARSE_PASS2_SEGMENT_STATE_BYTES (relax/cuda/posterior.py).
static_assert(sizeof(SegmentState) == 3176, "SegmentState layout changed: update posterior.py");

// Last index i with offsets[i] <= cell, or -1 when no segment covers the cell.
// Empty segments are never returned: offsets[i] == offsets[i + 1] cannot
// bracket a cell.
__device__ __forceinline__ int find_segment(
    const int32_t* offsets, int n_segments, int64_t cell)
{
    int low = 0;
    int high = n_segments + 1;
    while (low < high)
    {
        const int mid = (low + high) >> 1;
        if (static_cast<int64_t>(offsets[mid]) <= cell) low = mid + 1; else high = mid;
    }
    const int index = low - 1;
    if (index < 0 || index >= n_segments) return -1;
    return index;
}

__device__ __forceinline__ void segment_extent(
    const int32_t* offsets, int segment, int valid_count, int64_t n_cells,
    int64_t& begin, int64_t& n)
{
    // Defensive clamping: a malformed offset table must not read out of bounds.
    int64_t first = static_cast<int64_t>(offsets[segment]);
    int64_t last = static_cast<int64_t>(offsets[segment + 1]);
    if (first < 0) first = 0;
    if (first > n_cells) first = n_cells;
    if (last < first) last = first;
    if (last > n_cells) last = n_cells;
    begin = first;
    n = segment < valid_count ? last - first : 0;
}

__device__ __forceinline__ int clamp_valid_count(int32_t n_valid_images, int n_segments)
{
    if (n_valid_images < 0) return 0;
    if (n_valid_images > n_segments) return n_segments;
    return static_cast<int>(n_valid_images);
}

__global__ void segmented_log_z_kernel(
    const float* scores, const int32_t* offsets, const int32_t* n_valid_images,
    int n_segments, int64_t n_cells, double* log_z)
{
    __shared__ float shared_values[kRowThreads];
    __shared__ int64_t shared_indices[kRowThreads];
    __shared__ double shared_sums[kRowThreads];
    const int segment = blockIdx.x;
    const int valid_count = clamp_valid_count(*n_valid_images, n_segments);
    int64_t begin = 0, n = 0;
    segment_extent(offsets, segment, valid_count, n_cells, begin, n);
    log_z_body(scores + begin, n, log_z + segment, shared_values, shared_indices, shared_sums);
}

__global__ void segmented_max_kernel(
    const float* scores, const int32_t* offsets, const int32_t* n_valid_images,
    const double* log_z, int n_segments, int64_t n_cells, SegmentState* state,
    double* log_z_out, float* best_log_score, int64_t* best_argmax)
{
    __shared__ float shared_values[kRowThreads];
    __shared__ int64_t shared_indices[kRowThreads];
    const int segment = blockIdx.x;
    const int valid_count = clamp_valid_count(*n_valid_images, n_segments);
    int64_t begin = 0, n = 0;
    segment_extent(offsets, segment, valid_count, n_cells, begin, n);
    row_max_body(scores + begin, n, log_z[segment], shared_values, shared_indices,
                 &state[segment].row, log_z_out + segment, best_log_score + segment,
                 best_argmax + segment);
    if (threadIdx.x == 0)
    {
        state[segment].valid = segment < valid_count ? 1 : 0;
        state[segment].reserved = 0;
        state[segment].begin = begin;
        state[segment].n = n;
    }
}

template <bool kCells>
__global__ void segmented_exponentiate_kernel(
    const float* scores, const int32_t* offsets, const SegmentState* state,
    int n_segments, int64_t n_cells, float* raw_weights, double* probs,
    float* normalized, float* reconstruction, bool* mask)
{
    __shared__ int shared_segment;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (threadIdx.x == 0)
    {
        int64_t last = first + blockDim.x - 1;
        if (last > n_cells - 1) last = n_cells - 1;
        const int head = find_segment(offsets, n_segments, first);
        const int tail = find_segment(offsets, n_segments, last);
        shared_segment = head == tail ? head : -2;
    }
    __syncthreads();
    const int64_t index = first + threadIdx.x;
    if (index >= n_cells) return;
    int segment = shared_segment;
    if (segment == -2) segment = find_segment(offsets, n_segments, index);
    bool covered = false;
    if (segment >= 0)
    {
        const int64_t begin = state[segment].begin;
        covered = state[segment].valid != 0 && index >= begin &&
                  index < begin + state[segment].n;
    }
    if (!covered)
    {
        // Padding cells, empty segments and images past n_valid_images behave
        // like an all -inf row: zero weight, zero probability, no
        // significant candidate.  normalized/reconstruction/mask are written
        // here because the normalize kernel only walks live segments.
        raw_weights[index] = 0.0f;
        reconstruction[index] = 0.0f;
        if (kCells)
        {
            probs[index] = 0.0;
            normalized[index] = 0.0f;
            mask[index] = false;
        }
        return;
    }
    exponentiate_cell<kCells>(state[segment].row, scores[index], raw_weights + index,
                              kCells ? probs + index : nullptr);
}

// The segments' significance cut is a radix select, with each segment's cells split over
// kCutBlocksPerSegment blocks (gridDim.y): every radix pass adds the blocks' shared histograms into the
// segment's (SegmentState::cut) and one thread per segment then chooses the bin (radix_cut_select). One
// block per segment ran a lone tilt particle's 188M cells alone (4.1 s on an H100). The counts are exact and
// the masses float64 sums, which the shared-memory and global atomics add in no fixed order.
__global__ void segmented_cut_reset_kernel(SegmentState* state)
{
    BlockCut& c = state[blockIdx.x].cut;
    for (int b = threadIdx.x; b < 256; b += blockDim.x) { c.counts[b] = 0u; c.masses[b] = 0.0; }
    if (threadIdx.x == 0) radix_cut_reset(c);
}

__global__ void segmented_cut_histogram_kernel(const float* raw_weights, int pass, SegmentState* state)
{
    __shared__ uint32_t counts[256];
    __shared__ double masses[256];
    const int segment = blockIdx.x;
    BlockCut& c = state[segment].cut;
    const int64_t n = state[segment].n;
    if (c.empty || n == 0) return;
    const int64_t share = (n + gridDim.y - 1) / gridDim.y;
    const int64_t first = static_cast<int64_t>(blockIdx.y) * share;
    const int64_t last = first + share < n ? first + share : n;
    if (first >= last) return;
    for (int b = threadIdx.x; b < 256; b += blockDim.x) { counts[b] = 0u; masses[b] = 0.0; }
    __syncthreads();
    const uint32_t prefix = c.prefix;
    const float* weights = raw_weights + state[segment].begin;
    for (int64_t i = first + threadIdx.x; i < last; i += blockDim.x)
    {
        const float weight = weights[i];
        const int bin = radix_cut_bin(weight, pass, prefix);
        if (bin < 0) continue;
        atomicAdd(&counts[bin], 1u);
        atomicAdd(&masses[bin], static_cast<double>(weight));
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x)
    {
        if (counts[b] == 0u) continue;
        atomicAdd(&c.counts[b], counts[b]);
        atomicAdd(&c.masses[b], masses[b]);
    }
}

__global__ void segmented_cut_select_kernel(
    int pass, const float* external_sum_weight, float adaptive_fraction, int keep_all, int use_external_sum_weight,
    SegmentState* state, float* sum_weight_out, float* threshold_out)
{
    const int segment = blockIdx.x;
    BlockCut& c = state[segment].cut;
    if (threadIdx.x == 0) radix_cut_select(c, pass, adaptive_fraction);
    __syncthreads();
    if (pass < 3)
    {
        // The next pass's histogram starts from zero.
        for (int b = threadIdx.x; b < 256; b += blockDim.x) { c.counts[b] = 0u; c.masses[b] = 0.0; }
        return;
    }
    if (threadIdx.x != 0) return;
    cut_body(c, state[segment].n, external_sum_weight[segment], keep_all, use_external_sum_weight, state[segment].row,
             sum_weight_out + segment, threshold_out + segment);
}

// Blocks per segment of the segmented cut: about kCutCellsPerBlock cells each when the cells fill one segment,
// at most kCutMaxBlocks in the grid.
constexpr int64_t kCutCellsPerBlock = 1 << 16;
constexpr int64_t kCutMaxBlocks = 8192;

inline int segmented_cut_blocks_per_segment(int64_t n_cells, int64_t n_segments)
{
    int64_t blocks = (n_cells + kCutCellsPerBlock - 1) / kCutCellsPerBlock;
    const int64_t cap = kCutMaxBlocks / (n_segments > 0 ? n_segments : 1);
    if (blocks > cap) blocks = cap;
    return static_cast<int>(blocks < 1 ? 1 : blocks);
}

template <bool kCells>
__global__ void segmented_normalize_kernel(
    const float* scores, const float* raw_weights, const SegmentState* state,
    float* normalized, float* reconstruction, bool* mask,
    int32_t* n_significant, float* max_posterior)
{
    __shared__ int shared_counts[kRowThreads];
    __shared__ float shared_max[kRowThreads];
    const int segment = blockIdx.x;
    // Read the extent as scalars and keep the RowState reference in global
    // memory.  Copying SegmentState into a
    // local and passing its address puts the record in local memory, which the
    // inner loop then reloads once per cell; on a one-block 7.3M-cell segment
    // that alone cost ~1.9 ms.
    const int64_t begin = state[segment].begin;
    const int64_t n = state[segment].n;
    normalize_body<kCells>(scores + begin, raw_weights + begin, state[segment].row, n,
                           kCells ? normalized + begin : nullptr, reconstruction + begin,
                           kCells ? mask + begin : nullptr,
                           n_significant + segment, max_posterior + segment,
                           shared_counts, shared_max);
}

inline bool flat_geometry(const ffi::Span<const int64_t> dims, int64_t& count)
{
    count = 1;
    for (size_t i = 0; i < dims.size(); ++i) count *= dims[i];
    return dims.size() >= 1 && count >= 1;
}

inline bool segmented_vector_ok(
    const ffi::AnyBuffer& buffer, ffi::DataType type, int64_t length)
{
    return buffer.element_type() == type && buffer.dimensions().size() == 1 &&
           buffer.dimensions()[0] == length;
}

inline bool segmented_flat_ok(
    const ffi::AnyBuffer& buffer, ffi::DataType type, int64_t count)
{
    int64_t total = 0;
    return buffer.element_type() == type && flat_geometry(buffer.dimensions(), total) &&
           total == count;
}

inline bool scalar_ok(const ffi::AnyBuffer& buffer, ffi::DataType type)
{
    int64_t total = 1;
    for (size_t i = 0; i < buffer.dimensions().size(); ++i) total *= buffer.dimensions()[i];
    return buffer.element_type() == type && total == 1;
}

ffi::Error segmented_log_z_impl(
    cudaStream_t stream,
    ffi::AnyBuffer scores,
    ffi::AnyBuffer segment_offsets,
    ffi::AnyBuffer n_valid_images,
    ffi::Result<ffi::AnyBuffer> log_z)
{
    int64_t n_cells = 0;
    if (scores.element_type() != ffi::DataType::F32 || !flat_geometry(scores.dimensions(), n_cells))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: scores must be a nonempty F32 array");
    if (n_cells > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: cell count exceeds the int32 offset range");
    if (segment_offsets.element_type() != ffi::DataType::S32 ||
        segment_offsets.dimensions().size() != 1 || segment_offsets.dimensions()[0] < 2)
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: segment_offsets must be S32 with shape (n_segments + 1,)");
    const int64_t n_segments = segment_offsets.dimensions()[0] - 1;
    if (n_segments > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: too many segments for one launch grid");
    if (!scalar_ok(n_valid_images, ffi::DataType::S32))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: n_valid_images must be a single S32 value");
    if (!segmented_vector_ok(*log_z, ffi::DataType::F64, n_segments))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: log_z must be F64 with shape (n_segments,)");
    segmented_log_z_kernel<<<static_cast<int>(n_segments), kRowThreads, 0, stream>>>(
        static_cast<const float*>(scores.untyped_data()),
        static_cast<const int32_t*>(segment_offsets.untyped_data()),
        static_cast<const int32_t*>(n_valid_images.untyped_data()),
        static_cast<int>(n_segments), n_cells,
        static_cast<double*>(log_z->untyped_data()));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedLogZF64: ") + cudaGetErrorString(error));
    return ffi::Error::Success();
}

// The four launches of the segmented posterior: segment max, exponentiate, cut,
// normalize.  kCells = false (SparsePass2SegmentedReconstructionF32) passes null
// probs, normalized and mask and the kernels skip them; every other value is
// computed by the same code in the same order.
template <bool kCells>
ffi::Error launch_segmented_posterior(
    cudaStream_t stream, float adaptive_fraction, int64_t keep_all, int64_t use_external_sum_weight,
    int64_t n_cells, int64_t n_segments,
    const float* scores_ptr, const int32_t* offsets_ptr, const int32_t* n_valid_images_ptr,
    const double* log_z_ptr, const float* external_sum_weight_ptr,
    double* log_z_out_ptr, float* best_log_score_ptr, int64_t* best_cell_index_ptr, float* max_posterior_ptr,
    double* probs_ptr, float* normalized_ptr, float* reconstruction_ptr, bool* mask_ptr,
    int32_t* n_significant_ptr, float* sum_weight_ptr, float* threshold_ptr, float* raw_ptr,
    SegmentState* state_ptr)
{
    segmented_max_kernel<<<static_cast<int>(n_segments), kRowThreads, 0, stream>>>(
        scores_ptr, offsets_ptr, n_valid_images_ptr, log_z_ptr,
        static_cast<int>(n_segments), n_cells, state_ptr,
        log_z_out_ptr, best_log_score_ptr, best_cell_index_ptr);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 segment max: ") + cudaGetErrorString(error));

    constexpr int threads = 256;
    const int64_t block_count = (n_cells + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: launch grid exceeds CUDA limit");
    segmented_exponentiate_kernel<kCells><<<static_cast<int>(block_count), threads, 0, stream>>>(
        scores_ptr, offsets_ptr, state_ptr, static_cast<int>(n_segments), n_cells,
        raw_ptr, probs_ptr, normalized_ptr, reconstruction_ptr, mask_ptr);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 exponentiate: ") + cudaGetErrorString(error));

    segmented_cut_reset_kernel<<<static_cast<int>(n_segments), kCutThreads, 0, stream>>>(state_ptr);
    const dim3 cut_grid(static_cast<unsigned>(n_segments),
                        static_cast<unsigned>(segmented_cut_blocks_per_segment(n_cells, n_segments)));
    for (int pass = 0; pass < 4; ++pass)
    {
        segmented_cut_histogram_kernel<<<cut_grid, kCutThreads, 0, stream>>>(raw_ptr, pass, state_ptr);
        segmented_cut_select_kernel<<<static_cast<int>(n_segments), kCutThreads, 0, stream>>>(
            pass, external_sum_weight_ptr, adaptive_fraction,
            static_cast<int>(keep_all != 0), static_cast<int>(use_external_sum_weight != 0),
            state_ptr, sum_weight_ptr, threshold_ptr);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 cut: ") + cudaGetErrorString(error));

    segmented_normalize_kernel<kCells><<<static_cast<int>(n_segments), kRowThreads, 0, stream>>>(
        scores_ptr, raw_ptr, state_ptr, normalized_ptr, reconstruction_ptr, mask_ptr,
        n_significant_ptr, max_posterior_ptr);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 normalize: ") + cudaGetErrorString(error));
    return ffi::Error::Success();
}

// Shape checks shared by the two segmented posterior handlers: inputs, the
// per-segment outputs, the reconstruction, the raw-weight and segment-state
// scratch.  Returns null when they hold, else the message; sets n_cells and
// n_segments.
inline const char* segmented_posterior_shape_error(
    const ffi::AnyBuffer& scores, const ffi::AnyBuffer& segment_offsets, const ffi::AnyBuffer& n_valid_images,
    const ffi::AnyBuffer& log_z, const ffi::AnyBuffer& external_sum_weight,
    const ffi::AnyBuffer& log_z_out, const ffi::AnyBuffer& best_log_score, const ffi::AnyBuffer& best_cell_index,
    const ffi::AnyBuffer& max_posterior, const ffi::AnyBuffer& reconstruction, const ffi::AnyBuffer& n_significant,
    const ffi::AnyBuffer& sum_weight, const ffi::AnyBuffer& threshold, const ffi::AnyBuffer& raw_weights,
    const ffi::AnyBuffer& segment_state, int64_t& n_cells, int64_t& n_segments)
{
    n_cells = 0;
    n_segments = 0;
    if (scores.element_type() != ffi::DataType::F32 || !flat_geometry(scores.dimensions(), n_cells))
        return "scores must be a nonempty F32 array";
    if (n_cells > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        return "cell count exceeds the int32 offset range";
    if (segment_offsets.element_type() != ffi::DataType::S32 ||
        segment_offsets.dimensions().size() != 1 || segment_offsets.dimensions()[0] < 2)
        return "segment_offsets must be S32 with shape (n_segments + 1,)";
    n_segments = segment_offsets.dimensions()[0] - 1;
    if (n_segments > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return "too many segments for one launch grid";
    if (!scalar_ok(n_valid_images, ffi::DataType::S32))
        return "n_valid_images must be a single S32 value";
    if (!segmented_vector_ok(log_z, ffi::DataType::F64, n_segments) ||
        !segmented_vector_ok(external_sum_weight, ffi::DataType::F32, n_segments))
        return "log_z must be F64 (n_segments,) and external_sum_weight F32 (n_segments,)";
    if (!segmented_vector_ok(log_z_out, ffi::DataType::F64, n_segments) ||
        !segmented_vector_ok(best_log_score, ffi::DataType::F32, n_segments) ||
        !segmented_vector_ok(best_cell_index, ffi::DataType::S64, n_segments) ||
        !segmented_vector_ok(max_posterior, ffi::DataType::F32, n_segments) ||
        !segmented_vector_ok(n_significant, ffi::DataType::S32, n_segments) ||
        !segmented_vector_ok(sum_weight, ffi::DataType::F32, n_segments) ||
        !segmented_vector_ok(threshold, ffi::DataType::F32, n_segments))
        return "per-segment outputs must have shape (n_segments,) with the documented dtypes";
    if (!segmented_flat_ok(reconstruction, ffi::DataType::F32, n_cells) ||
        !segmented_flat_ok(raw_weights, ffi::DataType::F32, n_cells))
        return "per-cell outputs must hold n_cells elements with the documented dtypes";
    if (segment_state.element_type() != ffi::DataType::U8 ||
        segment_state.dimensions().size() != 2 ||
        segment_state.dimensions()[0] != n_segments ||
        segment_state.dimensions()[1] != static_cast<int64_t>(sizeof(SegmentState)))
        return "segment_state scratch must be U8 (n_segments, sizeof(SegmentState))";
    return nullptr;
}

ffi::Error segmented_posterior_impl(
    cudaStream_t stream,
    float adaptive_fraction,
    int64_t keep_all,
    int64_t use_external_sum_weight,
    ffi::AnyBuffer scores,
    ffi::AnyBuffer segment_offsets,
    ffi::AnyBuffer n_valid_images,
    ffi::AnyBuffer log_z,
    ffi::AnyBuffer external_sum_weight,
    ffi::Result<ffi::AnyBuffer> log_z_out,
    ffi::Result<ffi::AnyBuffer> best_log_score,
    ffi::Result<ffi::AnyBuffer> best_cell_index,
    ffi::Result<ffi::AnyBuffer> max_posterior,
    ffi::Result<ffi::AnyBuffer> probs,
    ffi::Result<ffi::AnyBuffer> normalized,
    ffi::Result<ffi::AnyBuffer> reconstruction,
    ffi::Result<ffi::AnyBuffer> mask,
    ffi::Result<ffi::AnyBuffer> n_significant,
    ffi::Result<ffi::AnyBuffer> sum_weight,
    ffi::Result<ffi::AnyBuffer> threshold,
    ffi::Result<ffi::AnyBuffer> raw_weights,
    ffi::Result<ffi::AnyBuffer> segment_state)
{
    int64_t n_cells = 0, n_segments = 0;
    if (const char* message = segmented_posterior_shape_error(
            scores, segment_offsets, n_valid_images, log_z, external_sum_weight, *log_z_out, *best_log_score,
            *best_cell_index, *max_posterior, *reconstruction, *n_significant, *sum_weight, *threshold,
            *raw_weights, *segment_state, n_cells, n_segments))
        return ffi::Error::InvalidArgument(std::string("SparsePass2SegmentedPosteriorF32: ") + message);
    if (!segmented_flat_ok(*probs, ffi::DataType::F64, n_cells) ||
        !segmented_flat_ok(*normalized, ffi::DataType::F32, n_cells) ||
        !segmented_flat_ok(*mask, ffi::DataType::PRED, n_cells))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: per-cell outputs must hold n_cells elements with the documented dtypes");

    return launch_segmented_posterior<true>(
        stream, adaptive_fraction, keep_all, use_external_sum_weight, n_cells, n_segments,
        static_cast<const float*>(scores.untyped_data()),
        static_cast<const int32_t*>(segment_offsets.untyped_data()),
        static_cast<const int32_t*>(n_valid_images.untyped_data()),
        static_cast<const double*>(log_z.untyped_data()),
        static_cast<const float*>(external_sum_weight.untyped_data()),
        static_cast<double*>(log_z_out->untyped_data()),
        static_cast<float*>(best_log_score->untyped_data()),
        static_cast<int64_t*>(best_cell_index->untyped_data()),
        static_cast<float*>(max_posterior->untyped_data()),
        static_cast<double*>(probs->untyped_data()),
        static_cast<float*>(normalized->untyped_data()),
        static_cast<float*>(reconstruction->untyped_data()),
        static_cast<bool*>(mask->untyped_data()),
        static_cast<int32_t*>(n_significant->untyped_data()),
        static_cast<float*>(sum_weight->untyped_data()),
        static_cast<float*>(threshold->untyped_data()),
        static_cast<float*>(raw_weights->untyped_data()),
        static_cast<SegmentState*>(segment_state->untyped_data()));
}

// SparsePass2SegmentedPosteriorF32 without probs, normalized and mask: 12 bytes
// per cell (scores, reconstruction, raw weights) instead of 29.
ffi::Error segmented_reconstruction_impl(
    cudaStream_t stream,
    float adaptive_fraction,
    int64_t keep_all,
    int64_t use_external_sum_weight,
    ffi::AnyBuffer scores,
    ffi::AnyBuffer segment_offsets,
    ffi::AnyBuffer n_valid_images,
    ffi::AnyBuffer log_z,
    ffi::AnyBuffer external_sum_weight,
    ffi::Result<ffi::AnyBuffer> log_z_out,
    ffi::Result<ffi::AnyBuffer> best_log_score,
    ffi::Result<ffi::AnyBuffer> best_cell_index,
    ffi::Result<ffi::AnyBuffer> max_posterior,
    ffi::Result<ffi::AnyBuffer> reconstruction,
    ffi::Result<ffi::AnyBuffer> n_significant,
    ffi::Result<ffi::AnyBuffer> sum_weight,
    ffi::Result<ffi::AnyBuffer> threshold,
    ffi::Result<ffi::AnyBuffer> raw_weights,
    ffi::Result<ffi::AnyBuffer> segment_state)
{
    int64_t n_cells = 0, n_segments = 0;
    if (const char* message = segmented_posterior_shape_error(
            scores, segment_offsets, n_valid_images, log_z, external_sum_weight, *log_z_out, *best_log_score,
            *best_cell_index, *max_posterior, *reconstruction, *n_significant, *sum_weight, *threshold,
            *raw_weights, *segment_state, n_cells, n_segments))
        return ffi::Error::InvalidArgument(std::string("SparsePass2SegmentedReconstructionF32: ") + message);

    return launch_segmented_posterior<false>(
        stream, adaptive_fraction, keep_all, use_external_sum_weight, n_cells, n_segments,
        static_cast<const float*>(scores.untyped_data()),
        static_cast<const int32_t*>(segment_offsets.untyped_data()),
        static_cast<const int32_t*>(n_valid_images.untyped_data()),
        static_cast<const double*>(log_z.untyped_data()),
        static_cast<const float*>(external_sum_weight.untyped_data()),
        static_cast<double*>(log_z_out->untyped_data()),
        static_cast<float*>(best_log_score->untyped_data()),
        static_cast<int64_t*>(best_cell_index->untyped_data()),
        static_cast<float*>(max_posterior->untyped_data()),
        nullptr, nullptr,
        static_cast<float*>(reconstruction->untyped_data()),
        nullptr,
        static_cast<int32_t*>(n_significant->untyped_data()),
        static_cast<float*>(sum_weight->untyped_data()),
        static_cast<float*>(threshold->untyped_data()),
        static_cast<float*>(raw_weights->untyped_data()),
        static_cast<SegmentState*>(segment_state->untyped_data()));
}

}  // namespace recovar_sparse_pass2_posterior


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SparsePass2SegmentedLogZF64, recovar_sparse_pass2_posterior::segmented_log_z_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SparsePass2SegmentedPosteriorF32, recovar_sparse_pass2_posterior::segmented_posterior_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("adaptive_fraction")
        .Attr<int64_t>("keep_all")
        .Attr<int64_t>("use_external_sum_weight")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SparsePass2SegmentedReconstructionF32, recovar_sparse_pass2_posterior::segmented_reconstruction_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("adaptive_fraction")
        .Attr<int64_t>("keep_all")
        .Attr<int64_t>("use_external_sum_weight")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);
