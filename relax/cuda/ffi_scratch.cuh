// Device scratch of a relax custom call from XLA's allocator (relax#120).
//
// A handler's temporary buffers come from the FFI scratch allocator (Ctx<ffi::ScratchAllocator>), which XLA serves
// from the same device pool the planners budget, not from the CUDA default memory pool beside it. Under XLA's
// preallocated pool little device memory is left outside it (a 16 GB card: 0.24 GiB), so cudaMallocAsync scratch
// that grows with a call's size failed there while the pool had GiB free. The allocations are released when the
// handler returns; XLA reuses them only for work later on the same stream, after the kernels enqueued here.
//
// Included from relax_kernels.cu after the ffi namespace alias.

#include <optional>

namespace relax_ffi_scratch {

// XLA's DeviceMemory_Allocate refuses an alignment above 16 bytes ("Unsupported alignment": 64 and 256 failed on
// jax 0.9, 16 works); its device allocator hands out 256-byte aligned chunks in any case, and CUB aligns its own
// temporaries inside the storage it is given.
constexpr size_t kMaxAlignment = 16;

// ``*out`` = ``bytes`` of scratch aligned to ``alignment`` (at most kMaxAlignment), or cudaErrorMemoryAllocation
// when the pool has no room.
inline cudaError_t allocate(ffi::ScratchAllocator& scratch, void** out, size_t bytes, size_t alignment = kMaxAlignment)
{
    *out = nullptr;
    if (bytes == 0) return cudaSuccess;
    const std::optional<void*> allocation = scratch.Allocate(bytes, alignment);
    if (!allocation.has_value() || *allocation == nullptr) return cudaErrorMemoryAllocation;
    *out = *allocation;
    return cudaSuccess;
}

}  // namespace relax_ffi_scratch
