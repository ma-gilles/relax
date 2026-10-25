// A persistent RELION capacity-half projector texture and its projection FFI.
// Included once at global scope after the ProjectRelionHalfImageRadius handler.
//
// ProjectRelionHalfImageRadius stages the projector into a CUDA texture on
// every call: a fill kernel over the whole texture, two cudaMalloc3DArray and
// two 3-D copies, the projection, a cudaStreamSynchronize and the frees. A
// resident local pass projects one fixed projector per half from hundreds of
// chunks, so each chunk paid a full texture staging and a host-device
// synchronization. This owner stages the texture once, exactly as
// launch_project_texture_float<true> does, and each projection launches the
// same projection as project_texture_kernel<true, true> on it without a synchronization.
// The texture holds Re and Im as one float2 texel: one fetch per sample, and the
// hardware filters each channel with the weights it uses for a single float texture,
// so the values are those of the float texture pair it replaces.
// Handles are monotonic and resolved through a guarded registry, as for the
// persistent host-uploaded texture above.

struct CapacityRelionHalfTextureF32 {
    cudaArray_t array = nullptr;            // float2 (Re, Im) texels
    cudaTextureObject_t texture = 0;
    int32_t* logical_radius = nullptr;  // device scalar the fill kernel read
    float2* staging = nullptr;
    bool reusable_staging = false;
    int half_z = 0;
    int half_y = 0;
    int half_x = 0;
    int padding_factor = 0;
    int device = -1;

    std::mutex state_mutex;
    std::condition_variable state_changed;
    bool closing = false;
    bool refreshing = false;
    bool refresh_failed = false;
    int active_calls = 0;

    bool acquire_call()
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (closing || refreshing || refresh_failed) return false;
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
        }
    }

    void wait_until_idle()
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        closing = true;
        state_changed.wait(lock, [this] { return active_calls == 0 && !refreshing; });
    }

    bool begin_refresh()
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        if (closing || refreshing || refresh_failed || !reusable_staging) return false;
        refreshing = true;
        state_changed.wait(lock, [this] { return active_calls == 0; });
        return true;
    }

    void finish_refresh(bool success) noexcept
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        refresh_failed = !success;
        refreshing = false;
        state_changed.notify_all();
    }

    cudaError_t destroy_owned() noexcept
    {
        int original_device = -1;
        cudaError_t err = cudaGetDevice(&original_device);
        if (err == cudaSuccess && original_device != device)
            err = cudaSetDevice(device);
        /* Launched projections may still be reading the texture. */
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        cudaError_t cleanup = cudaSuccess;
        if (texture) cleanup = cudaDestroyTextureObject(texture);
        if (array) { const auto e = cudaFreeArray(array); if (cleanup == cudaSuccess) cleanup = e; }
        if (logical_radius) { const auto e = cudaFree(logical_radius); if (cleanup == cudaSuccess) cleanup = e; }
        if (staging) { const auto e = cudaFree(staging); if (cleanup == cudaSuccess) cleanup = e; }
        texture = 0;
        array = nullptr;
        logical_radius = nullptr;
        staging = nullptr;
        if (err == cudaSuccess) err = cleanup;
        if (original_device >= 0 && original_device != device) {
            const cudaError_t restore = cudaSetDevice(original_device);
            if (err == cudaSuccess) err = restore;
        }
        return err;
    }

    ~CapacityRelionHalfTextureF32() { (void)destroy_owned(); }
};

static std::mutex capacity_relion_half_texture_f32_mutex;
static std::unordered_map<uint64_t, std::shared_ptr<CapacityRelionHalfTextureF32>>
    capacity_relion_half_texture_f32_registry;
static std::atomic<uint64_t> capacity_relion_half_texture_f32_next_handle{1};

/* fill_relion_texture_capacity_kernel (recovar_cuda_common.cuh) for the texture
 * z-planes [z_begin, z_begin + n_planes): the same texel values, written as
 * (Re, Im) pairs into a plane-sized staging buffer and addressed with 64-bit indices. */
__global__ void __launch_bounds__(BLOCK_SIZE)
fill_relion_texture_capacity_planes_kernel(
    const float2* __restrict__ vol, float2* texels,
    const int32_t* logical_radius, int upsampling,
    int texX, int texY, int texZ, int z_begin, int n_planes)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
    if (idx >= static_cast<int64_t>(texX) * texY * n_planes) return;
    const int x = static_cast<int>(idx % texX);
    const int y = static_cast<int>((idx / texX) % texY);
    const int z = z_begin + static_cast<int>(idx / (static_cast<int64_t>(texX) * texY));
    const int radius = *logical_radius;
    float2 value = make_float2(0.0f, 0.0f);
    if (radius >= 0 && radius <= (texY / 2 - 1) / upsampling) {
        const int R = radius * upsampling;
        if (x < R + 2 && y < 2 * R + 3 && z < 2 * R + 3) {
            const int64_t iy = texY / 2 + y - (R + 1);
            const int64_t iz = texZ / 2 + z - (R + 1);
            value = vol[(iz * texY + iy) * texX + x];
        }
    }
    texels[idx] = value;
}

// The staging buffer is at most this many texels per plane group (one float2
// buffer of 1 GiB), so the transient beside the texture stays inside the
// XLA pool reserve's slack (relax/runtime/xla_memory_reserve.py).
constexpr int64_t kCapacityTextureStagingTexels = int64_t(1) << 27;

/* The staging of launch_project_texture_float<true> (recovar_cuda_common.cuh):
 * the same texel values, as one float2 texture rather than its float pair, filled
 * and copied a group of z-planes at a time so the staging transient is bounded
 * rather than a second copy of the slab. */
static cudaError_t stage_capacity_relion_half_texture_f32(
    const float2* projector_half, int logical_radius, CapacityRelionHalfTextureF32* owner)
{
    const int texX = owner->half_x;
    const int texY = owner->half_y;
    const int texZ = owner->half_z;
    const int64_t plane = static_cast<int64_t>(texX) * texY;
    const int group = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(texZ, kCapacityTextureStagingTexels / plane)));
    float2* staging = owner->staging;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err == cudaSuccess && !owner->logical_radius)
        err = cudaMalloc(reinterpret_cast<void**>(&owner->logical_radius), sizeof(int32_t));
    if (err == cudaSuccess)
        err = cudaMemcpy(owner->logical_radius, &logical_radius, sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err == cudaSuccess && !staging)
        err = cudaMalloc(reinterpret_cast<void**>(&staging), plane * group * sizeof(float2));
    if (owner->reusable_staging) owner->staging = staging;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaExtent extent = make_cudaExtent((size_t)texX, (size_t)texY, (size_t)texZ);
    if (err == cudaSuccess && !owner->array) err = cudaMalloc3DArray(&owner->array, &desc, extent);
    for (int z_begin = 0; z_begin < texZ && err == cudaSuccess; z_begin += group) {
        const int n_planes = std::min(group, texZ - z_begin);
        const int64_t texels = plane * n_planes;
        fill_relion_texture_capacity_planes_kernel<<<static_cast<unsigned>((texels + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            projector_half, staging, owner->logical_radius, owner->padding_factor, texX, texY, texZ, z_begin, n_planes);
        err = cudaGetLastError();
        cudaMemcpy3DParms copy_params = {0};
        copy_params.extent = make_cudaExtent((size_t)texX, (size_t)texY, (size_t)n_planes);
        copy_params.dstPos = make_cudaPos(0, 0, (size_t)z_begin);
        copy_params.kind = cudaMemcpyDeviceToDevice;
        copy_params.dstArray = owner->array;
        copy_params.srcPtr = make_cudaPitchedPtr(staging, (size_t)texX * sizeof(float2), (size_t)texX, (size_t)texY);
        if (err == cudaSuccess) err = cudaMemcpy3DAsync(&copy_params, stream);
    }
    if (err == cudaSuccess && !owner->texture) {
        cudaResourceDesc resource;
        cudaTextureDesc texture_desc;
        memset(&resource, 0, sizeof(resource));
        memset(&texture_desc, 0, sizeof(texture_desc));
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = owner->array;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = false;
        texture_desc.addressMode[0] = cudaAddressModeClamp;
        texture_desc.addressMode[1] = cudaAddressModeClamp;
        texture_desc.addressMode[2] = cudaAddressModeClamp;
        err = cudaCreateTextureObject(&owner->texture, &resource, &texture_desc, nullptr);
    }
    /* The texture is complete before create returns; the staging buffers go.
     * Each plane group's buffers are reused only after its copies, which are
     * ordered before the next group's fill on the one stream. */
    if (stream) {
        const cudaError_t sync = cudaStreamSynchronize(stream);
        if (err == cudaSuccess) err = sync;
        cudaStreamDestroy(stream);
    }
    if (!owner->reusable_staging && staging) cudaFree(staging);
    return err;
}

/* ``projector_half`` is a device pointer to the C64 [z, y, x>=0] capacity half
 * storage of ProjectRelionHalfImageRadius; the caller keeps it alive and
 * complete until this returns. */
static int create_capacity_relion_half_texture_f32(
    const void* projector_half, int half_z, int half_y, int half_x,
    int logical_radius, int padding_factor, int device, uint64_t* owner_handle,
    bool reusable_staging)
{
    if (!projector_half || !owner_handle || half_z < 5 || half_z != half_y ||
        half_z % 2 != 1 || half_x != half_z / 2 + 1 ||
        (padding_factor != 1 && padding_factor != 2) ||
        (half_z - 3) % (2 * padding_factor) != 0 ||
        logical_radius < 0 || logical_radius > (half_y / 2 - 1) / padding_factor || device < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    *owner_handle = 0;
    try {
        auto owner = std::make_shared<CapacityRelionHalfTextureF32>();
        owner->half_z = half_z;
        owner->half_y = half_y;
        owner->half_x = half_x;
        owner->padding_factor = padding_factor;
        owner->device = device;
        owner->reusable_staging = reusable_staging;
        int original_device = -1;
        cudaError_t err = cudaGetDevice(&original_device);
        if (err != cudaSuccess) return static_cast<int>(err);
        if (original_device != device) err = cudaSetDevice(device);
        // The slab's extent is bounded by the device's 3-D texture extents.
        cudaDeviceProp properties;
        if (err == cudaSuccess) err = cudaGetDeviceProperties(&properties, device);
        if (err == cudaSuccess &&
            (half_x > properties.maxTexture3D[0] || half_y > properties.maxTexture3D[1] ||
             half_z > properties.maxTexture3D[2]))
            err = cudaErrorInvalidValue;
        if (err == cudaSuccess)
            err = stage_capacity_relion_half_texture_f32(
                static_cast<const float2*>(projector_half), logical_radius, owner.get());
        if (original_device != device) {
            const cudaError_t restore = cudaSetDevice(original_device);
            if (err == cudaSuccess) err = restore;
        }
        if (err != cudaSuccess) return static_cast<int>(err);
        const uint64_t handle =
            capacity_relion_half_texture_f32_next_handle.fetch_add(1, std::memory_order_relaxed);
        if (handle == 0 || handle > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return static_cast<int>(cudaErrorInvalidValue);
        {
            std::lock_guard<std::mutex> lock(capacity_relion_half_texture_f32_mutex);
            if (!capacity_relion_half_texture_f32_registry.emplace(handle, owner).second)
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

extern "C" int relax_relion_capacity_half_texture_f32_create(
    const void* projector_half, int half_z, int half_y, int half_x,
    int logical_radius, int padding_factor, int device, uint64_t* owner_handle)
{
    return create_capacity_relion_half_texture_f32(
        projector_half, half_z, half_y, half_x, logical_radius,
        padding_factor, device, owner_handle, false);
}

extern "C" int relax_relion_capacity_half_texture_f32_create_reusable(
    const void* projector_half, int half_z, int half_y, int half_x,
    int logical_radius, int padding_factor, int device, uint64_t* owner_handle)
{
    return create_capacity_relion_half_texture_f32(
        projector_half, half_z, half_y, half_x, logical_radius,
        padding_factor, device, owner_handle, true);
}

extern "C" int relax_relion_capacity_half_texture_f32_refresh(
    uint64_t owner_handle, const void* projector_half, int logical_radius, int device)
{
    if (owner_handle == 0 || !projector_half || device < 0) return static_cast<int>(cudaErrorInvalidValue);
    try {
        std::shared_ptr<CapacityRelionHalfTextureF32> owner;
        {
            std::lock_guard<std::mutex> lock(capacity_relion_half_texture_f32_mutex);
            const auto found = capacity_relion_half_texture_f32_registry.find(owner_handle);
            if (found == capacity_relion_half_texture_f32_registry.end())
                return static_cast<int>(cudaErrorInvalidResourceHandle);
            owner = found->second;
        }
        if (device != owner->device || logical_radius < 0 ||
            logical_radius > (owner->half_y / 2 - 1) / owner->padding_factor ||
            !owner->begin_refresh())
            return static_cast<int>(cudaErrorInvalidValue);
        struct RefreshGuard {
            CapacityRelionHalfTextureF32* owner;
            bool success = false;
            ~RefreshGuard() { owner->finish_refresh(success); }
        } guard{owner.get()};
        int original_device = -1;
        cudaError_t err = cudaGetDevice(&original_device);
        if (err == cudaSuccess && original_device != device) err = cudaSetDevice(device);
        // Iteration boundary only: no compiled projector may still be using
        // these arrays when the refill starts. Python also blocks on the final
        // compiled completion token before entering this C API.
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err == cudaSuccess)
            err = stage_capacity_relion_half_texture_f32(
                static_cast<const float2*>(projector_half), logical_radius, owner.get());
        if (original_device >= 0 && original_device != device) {
            const cudaError_t restore = cudaSetDevice(original_device);
            if (err == cudaSuccess) err = restore;
        }
        guard.success = err == cudaSuccess;
        return static_cast<int>(err);
    } catch (const std::bad_alloc&) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    } catch (...) {
        return static_cast<int>(cudaErrorUnknown);
    }
}

extern "C" int relax_relion_capacity_half_texture_f32_destroy(uint64_t owner_handle)
{
    if (owner_handle == 0) return static_cast<int>(cudaErrorInvalidResourceHandle);
    try {
        std::shared_ptr<CapacityRelionHalfTextureF32> owner;
        {
            std::lock_guard<std::mutex> lock(capacity_relion_half_texture_f32_mutex);
            const auto found = capacity_relion_half_texture_f32_registry.find(owner_handle);
            if (found == capacity_relion_half_texture_f32_registry.end())
                return static_cast<int>(cudaErrorInvalidResourceHandle);
            owner = found->second;
            capacity_relion_half_texture_f32_registry.erase(found);
        }
        owner->wait_until_idle();
        return static_cast<int>(owner->destroy_owned());
    } catch (const std::bad_alloc&) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    } catch (...) {
        return static_cast<int>(cudaErrorUnknown);
    }
}

/* project_texture_kernel<true, true> (recovar_cuda_common.cuh) on the float2 capacity texture: the same
 * source order, radius tests and texture coordinates, one float2 fetch per pixel. */
__global__ void __launch_bounds__(BLOCK_SIZE)
project_capacity_texture_half_kernel(
    cudaTextureObject_t texture, float* __restrict__ img, const float* __restrict__ rot,
    int n_pixels, int image_h, int image_w, int upsampling,
    const int32_t* logical_radius, int capacity_radius, const int32_t* image_radius)
{
    __shared__ float R[6];

    const int img_idx = blockIdx.x;
    const int pix = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    const int k0_idx = pix / image_w;
    const int k1_idx = pix % image_w;
    // RELION keeps both even-box Nyquist axes positive in its accelerated half-spectrum projector.
    const float k0_unscaled = (float)(k0_idx == 0 ? image_h / 2 : k0_idx - image_h / 2);
    const float k1_unscaled = (float)k1_idx;

    float2* img2 = reinterpret_cast<float2*>(img);
    const int img_off = img_idx * n_pixels + pix;
    const int radius = *logical_radius;
    if (radius < 0 || radius > capacity_radius / upsampling) {
        img2[img_off] = make_float2(nanf(""), nanf(""));
        return;
    }
    const int padded_radius = radius * upsampling;
    int maxR2_padded = padded_radius * padded_radius;
    const int tex_yinit = -(padded_radius + 1);
    const int tex_zinit = -(padded_radius + 1);

    if (image_radius != nullptr) {
        const int image_r = *image_radius;
        if (image_r < 0 || image_r > image_h / 2) {
            img2[img_off] = make_float2(nanf(""), nanf(""));
            return;
        }
        const int padded_image_r = image_r * upsampling;
        // AccProjectorKernel::makeKernel clips at the smaller image/model radius.
        maxR2_padded = min(maxR2_padded, padded_image_r * padded_image_r);
    }

    // RELION's AccProjectorKernel source order (matrix-x * source-x first): reversing the addends changes the
    // contracted FMA association and can cross a texture interpolation fraction-bin boundary.
    const float rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * (float)upsampling;
    const float rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * (float)upsampling;
    const float rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * (float)upsampling;

    if ((int)(rk0 * rk0 + rk1 * rk1 + rk2 * rk2) > maxR2_padded) {
        img2[img_off] = make_float2(0.0f, 0.0f);
        return;
    }

    float xp = rk0, yp = rk1, zp = rk2, imag_sign = 1.0f;
    if (xp < 0.0f) {
        xp = -xp;
        yp = -yp;
        zp = -zp;
        imag_sign = -1.0f;
    }
    const float2 value =
        tex3D<float2>(texture, xp + 0.5f, yp - (float)tex_yinit + 0.5f, zp - (float)tex_zinit + 0.5f);
    img2[img_off] = make_float2(value.x, imag_sign * value.y);
}

/* ProjectRelionHalfImageRadius on a persistent capacity texture: the launch of
 * launch_project_texture_float<true> with its max_r2_x4 = -1 geometry. */
ffi::Error ProjectRelionHalfCapacityTextureImpl(
    cudaStream_t stream, int64_t owner_handle, int64_t image_h, int64_t image_w,
    int64_t has_image_radius,
    ffi::AnyBuffer rot, ffi::AnyBuffer image_radius, ffi::Result<ffi::AnyBuffer> output)
{
    const auto r = rot.dimensions();
    const auto o = output->dimensions();
    if (rot.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::C64 ||
        image_radius.element_type() != ffi::DataType::S32 ||
        image_radius.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "ProjectRelionHalfCapacityTexture: require F32 rotations, scalar S32 image radius and C64 output");
    if (r.size() != 2 || r[1] != 6 || r[0] <= 0 || r[0] > 65535 ||
        image_h <= 0 || image_h != image_w || image_h % 2 != 0 || image_h > 4096 ||
        o.size() != 2 || o[0] != r[0] || o[1] != image_h * (image_w / 2 + 1) ||
        r[0] * o[1] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument("ProjectRelionHalfCapacityTexture: invalid rotation or image geometry");

    std::shared_ptr<CapacityRelionHalfTextureF32> owner;
    {
        std::lock_guard<std::mutex> lock(capacity_relion_half_texture_f32_mutex);
        const auto found = capacity_relion_half_texture_f32_registry.find(static_cast<uint64_t>(owner_handle));
        if (found == capacity_relion_half_texture_f32_registry.end())
            return ffi::Error::InvalidArgument("ProjectRelionHalfCapacityTexture: owner handle is not live");
        owner = found->second;
    }
    if (!owner->acquire_call())
        return ffi::Error::InvalidArgument(
            "ProjectRelionHalfCapacityTexture: owner is closing, refreshing or failed its last refresh");
    struct Release { CapacityRelionHalfTextureF32* owner; ~Release() { owner->release_call(); } } release{owner.get()};

    const float max_r2 = (float)((owner->half_z / 2 - 1) * (owner->half_z / 2 - 1));
    const int maxR = (int)floorf(sqrtf(max_r2) + 0.5f);
    const int n_pixels = static_cast<int>(o[1]);
    dim3 grid((int)r[0], (n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    project_capacity_texture_half_kernel<<<grid, block, 0, stream>>>(
        owner->texture, static_cast<float*>(output->untyped_data()), static_cast<const float*>(rot.untyped_data()),
        n_pixels, (int)image_h, (int)(image_w / 2 + 1), owner->padding_factor, owner->logical_radius, maxR,
        has_image_radius ? static_cast<const int32_t*>(image_radius.untyped_data()) : nullptr);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProjectRelionHalfCapacityTexture, ProjectRelionHalfCapacityTextureImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("owner_handle")
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("has_image_radius")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

// ProjectRelionHalfCapacityTexture at a list of crop pixels, scaled and real-packed: row n of the
// output is [Re p_n(pixel_0..P-1) * scale | Im p_n(pixel_0..P-1) * scale], the operand layout of
// the coarse GEMM scorer. Each listed pixel is projected with project_texture_kernel<true, true>'s
// arithmetic (the image-radius test included); a pixel index of -1 is a zero row entry. This is the
// projection plus the crop gather and the dense scale of compute_relion_projector_projections_block,
// without the full crop and the real/imaginary copies. A block projects its pixels for
// kCompactPackedRotationsPerBlock consecutive rotations (each sample's arithmetic is unchanged): a
// coarse pass's rows are short (about a thousand pixels), so one rotation per block spent much of
// each block on its setup (1.34x on H100 at 924 pixels, etobspeed 2026-10-15).
constexpr int kCompactPackedRotationsPerBlock = 4;

// One pixel j of project_capacity_texture_compact_packed_kernel's rotations first..first + n_block - 1; with
// kEnergy, e[n] is its weighted energy w (Re^2 + Im^2) of the stored values for rotation n.
template <bool kEnergy>
__device__ __forceinline__ void project_compact_packed_pixel(
    cudaTextureObject_t texture, float* __restrict__ out, const float (*R)[6], const int32_t* __restrict__ crop_index,
    int first, int n_block, int j, int n_compact, int image_h, int image_w, int tex_yinit, int tex_zinit,
    int upsampling, int maxR2_padded, const int32_t* image_radius, float scale, const float* __restrict__ weight,
    int rows_per_image, float* e)
{
    const int pix = crop_index[j];
    const bool live = pix >= 0;
    const int radius = *image_radius;
    if (live && (radius < 0 || radius > image_h / 2)) {
        for (int n = 0; n < n_block; ++n) {
            float* row = out + static_cast<int64_t>(first + n) * 2 * n_compact;
            row[j] = nanf("");
            row[n_compact + j] = nanf("");
            if constexpr (kEnergy) e[n] = nanf("");
        }
        return;
    }
    const int padded_radius = radius * upsampling;
    const int max_r2 = min(maxR2_padded, padded_radius * padded_radius);
    const int k0_idx = live ? pix / image_w : 0;
    const int k1_idx = live ? pix % image_w : 0;
    const float k0_unscaled = (float)(k0_idx == 0 ? image_h / 2 : k0_idx - image_h / 2);
    const float k1_unscaled = (float)k1_idx;
#pragma unroll
    for (int n = 0; n < kCompactPackedRotationsPerBlock; ++n) {
        if (n >= n_block) break;
        float re = 0.0f, im = 0.0f;
        if (live) {
            const float rk0 = (R[n][3] * k1_unscaled + R[n][0] * k0_unscaled) * (float)upsampling;
            const float rk1 = (R[n][4] * k1_unscaled + R[n][1] * k0_unscaled) * (float)upsampling;
            const float rk2 = (R[n][5] * k1_unscaled + R[n][2] * k0_unscaled) * (float)upsampling;
            if ((int)(rk0 * rk0 + rk1 * rk1 + rk2 * rk2) <= max_r2) {
                float xp = rk0, yp = rk1, zp = rk2, imag_sign = 1.0f;
                if (xp < 0.0f) { xp = -xp; yp = -yp; zp = -zp; imag_sign = -1.0f; }
                const float2 value =
                    tex3D<float2>(texture, xp + 0.5f, yp - (float)tex_yinit + 0.5f, zp - (float)tex_zinit + 0.5f);
                re = value.x;
                im = imag_sign * value.y;
            }
        }
        // compute_relion_projector_projections_block's complex64 * float32 dense scale, per component.
        float* row = out + static_cast<int64_t>(first + n) * 2 * n_compact;
        row[j] = __fmul_rn(re, scale);
        row[n_compact + j] = __fmul_rn(im, scale);
        if constexpr (kEnergy) {
            const float w = weight[static_cast<int64_t>((first + n) / rows_per_image) * n_compact + j];
            const float re_s = __fmul_rn(re, scale), im_s = __fmul_rn(im, scale);
            e[n] = __fmul_rn(w, __fadd_rn(__fmul_rn(re_s, re_s), __fmul_rn(im_s, im_s)));
        }
    }
}

// kEnergy also writes each row's weighted energy over this block's pixels, energy[row, blockIdx.y] =
// sum_j w_j (Re_j^2 + Im_j^2) of the scaled packed values, with w the row's image's pixel weights
// (weight[(row / rows_per_image) * n_compact + j]); the caller adds the pixel blocks' partials in block order.
// The sum is a fixed shared-memory tree over the block's threads, so the result is the same on every run; it
// replaces the coarse GEMM scorer's model-energy product, whose summation order it does not keep.
template <bool kEnergy>
__global__ void __launch_bounds__(BLOCK_SIZE)
project_capacity_texture_compact_packed_kernel(
    cudaTextureObject_t texture,
    float* __restrict__ out, const float* __restrict__ rot, const int32_t* __restrict__ crop_index,
    int n_rotations, int n_compact, int image_h, int image_w, int tex_yinit, int tex_zinit, int upsampling,
    int maxR2_padded, const int32_t* image_radius, float scale,
    const float* __restrict__ weight, int rows_per_image, float* __restrict__ energy)
{
    __shared__ float R[kCompactPackedRotationsPerBlock][6];
    __shared__ float partial[kEnergy ? kCompactPackedRotationsPerBlock : 1][kEnergy ? BLOCK_SIZE : 1];
    const int first = blockIdx.x * kCompactPackedRotationsPerBlock;
    const int n_block = min(kCompactPackedRotationsPerBlock, n_rotations - first);
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    if (threadIdx.x < 6 * n_block) R[threadIdx.x / 6][threadIdx.x % 6] = rot[first * 6 + threadIdx.x];
    __syncthreads();
    if constexpr (kEnergy) {
        // Every thread takes part in the tree, a pixel past n_compact with zero energy.
        float e[kCompactPackedRotationsPerBlock];
#pragma unroll
        for (int n = 0; n < kCompactPackedRotationsPerBlock; ++n) e[n] = 0.0f;
        if (j < n_compact)
            project_compact_packed_pixel<true>(texture, out, R, crop_index, first, n_block, j, n_compact, image_h,
                                               image_w, tex_yinit, tex_zinit, upsampling, maxR2_padded, image_radius,
                                               scale, weight, rows_per_image, e);
#pragma unroll
        for (int n = 0; n < kCompactPackedRotationsPerBlock; ++n) partial[n][threadIdx.x] = e[n];
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
#pragma unroll
                for (int n = 0; n < kCompactPackedRotationsPerBlock; ++n)
                    partial[n][threadIdx.x] = __fadd_rn(partial[n][threadIdx.x], partial[n][threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x < n_block)
            energy[static_cast<int64_t>(first + threadIdx.x) * gridDim.y + blockIdx.y] = partial[threadIdx.x][0];
    } else {
        if (j >= n_compact) return;
        project_compact_packed_pixel<false>(texture, out, R, crop_index, first, n_block, j, n_compact, image_h,
                                            image_w, tex_yinit, tex_zinit, upsampling, maxR2_padded, image_radius,
                                            scale, nullptr, rows_per_image, nullptr);
    }
}

// The checks and launch of ProjectRelionHalfCapacityTextureCompactPacked; with weight and energy (kEnergy), also
// each row's per-pixel-block energy partials [n_rotations, pixel blocks] of project_capacity_texture_compact_packed_kernel.
template <bool kEnergy>
ffi::Error project_compact_packed_launch(
    const char* name, cudaStream_t stream, int64_t owner_handle, int64_t image_h, int64_t image_w, float scale,
    ffi::AnyBuffer rot, ffi::AnyBuffer image_radius, ffi::AnyBuffer crop_index, ffi::AnyBuffer* weight,
    int64_t rows_per_image, ffi::Result<ffi::AnyBuffer>& output, ffi::Result<ffi::AnyBuffer>* energy)
{
    const auto r = rot.dimensions();
    const auto c = crop_index.dimensions();
    const auto o = output->dimensions();
    if (rot.element_type() != ffi::DataType::F32 || output->element_type() != ffi::DataType::F32 ||
        image_radius.element_type() != ffi::DataType::S32 || image_radius.dimensions().size() != 0 ||
        crop_index.element_type() != ffi::DataType::S32 || c.size() != 1 || c[0] <= 0)
        return ffi::Error::InvalidArgument(
            std::string(name) + ": F32 rotations, S32 image radius and crop index, F32 output");
    // One block row per rotation on grid x (up to 2^31 - 1), so a slot block's rotations take one launch;
    // the pixel blocks on grid y.
    if (r.size() != 2 || r[1] != 6 || r[0] <= 0 || r[0] > std::numeric_limits<int>::max() ||
        image_h <= 0 || image_h != image_w || image_h % 2 != 0 || image_h > 4096 ||
        o.size() != 2 || o[0] != r[0] || o[1] != 2 * c[0] || (c[0] + BLOCK_SIZE - 1) / BLOCK_SIZE > 65535)
        return ffi::Error::InvalidArgument(std::string(name) + ": invalid geometry");
    const int64_t pixel_blocks = (c[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if constexpr (kEnergy) {
        const auto w = weight->dimensions();
        const auto e = (*energy)->dimensions();
        if (weight->element_type() != ffi::DataType::F32 || (*energy)->element_type() != ffi::DataType::F32 ||
            rows_per_image <= 0 || r[0] % rows_per_image != 0 || w.size() != 2 || w[0] != r[0] / rows_per_image ||
            w[1] != c[0] || e.size() != 2 || e[0] != r[0] || e[1] != pixel_blocks)
            return ffi::Error::InvalidArgument(
                std::string(name) + ": F32 weight [rotations / rows_per_image, pixels] and energy [rotations, pixel blocks]");
    }
    std::shared_ptr<CapacityRelionHalfTextureF32> owner;
    {
        std::lock_guard<std::mutex> lock(capacity_relion_half_texture_f32_mutex);
        const auto found = capacity_relion_half_texture_f32_registry.find(static_cast<uint64_t>(owner_handle));
        if (found == capacity_relion_half_texture_f32_registry.end())
            return ffi::Error::InvalidArgument(std::string(name) + ": owner handle is not live");
        owner = found->second;
    }
    if (!owner->acquire_call())
        return ffi::Error::InvalidArgument(
            std::string(name) + ": owner is closing, refreshing or failed its last refresh");
    struct Release { CapacityRelionHalfTextureF32* owner; ~Release() { owner->release_call(); } } release{owner.get()};

    // The geometry of ProjectRelionHalfCapacityTextureImpl.
    const float max_r2 = (float)((owner->half_z / 2 - 1) * (owner->half_z / 2 - 1));
    const int maxR = (int)floorf(sqrtf(max_r2) + 0.5f);
    const int n_compact = static_cast<int>(c[0]);
    const int n_rotations = static_cast<int>(r[0]);
    dim3 grid((n_rotations + kCompactPackedRotationsPerBlock - 1) / kCompactPackedRotationsPerBlock,
              static_cast<unsigned>(pixel_blocks));
    project_capacity_texture_compact_packed_kernel<kEnergy><<<grid, BLOCK_SIZE, 0, stream>>>(
        owner->texture, static_cast<float*>(output->untyped_data()),
        static_cast<const float*>(rot.untyped_data()), static_cast<const int32_t*>(crop_index.untyped_data()),
        n_rotations, n_compact, (int)image_h, (int)(image_w / 2 + 1), -(maxR + 1), -(maxR + 1), owner->padding_factor,
        maxR * maxR, static_cast<const int32_t*>(image_radius.untyped_data()), scale,
        kEnergy ? static_cast<const float*>(weight->untyped_data()) : nullptr, static_cast<int>(rows_per_image),
        kEnergy ? static_cast<float*>((*energy)->untyped_data()) : nullptr);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error ProjectRelionHalfCapacityTextureCompactPackedImpl(
    cudaStream_t stream, int64_t owner_handle, int64_t image_h, int64_t image_w, float scale,
    ffi::AnyBuffer rot, ffi::AnyBuffer image_radius, ffi::AnyBuffer crop_index, ffi::Result<ffi::AnyBuffer> output)
{
    return project_compact_packed_launch<false>(
        "ProjectRelionHalfCapacityTextureCompactPacked", stream, owner_handle, image_h, image_w, scale, rot,
        image_radius, crop_index, nullptr, 1, output, nullptr);
}

ffi::Error ProjectRelionHalfCapacityTextureCompactPackedEnergyImpl(
    cudaStream_t stream, int64_t owner_handle, int64_t image_h, int64_t image_w, float scale, int64_t rows_per_image,
    ffi::AnyBuffer rot, ffi::AnyBuffer image_radius, ffi::AnyBuffer crop_index, ffi::AnyBuffer weight,
    ffi::Result<ffi::AnyBuffer> output, ffi::Result<ffi::AnyBuffer> energy)
{
    return project_compact_packed_launch<true>(
        "ProjectRelionHalfCapacityTextureCompactPackedEnergy", stream, owner_handle, image_h, image_w, scale, rot,
        image_radius, crop_index, &weight, rows_per_image, output, &energy);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProjectRelionHalfCapacityTextureCompactPacked, ProjectRelionHalfCapacityTextureCompactPackedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("owner_handle")
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<float>("scale")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProjectRelionHalfCapacityTextureCompactPackedEnergy, ProjectRelionHalfCapacityTextureCompactPackedEnergyImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("owner_handle")
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<float>("scale")
        .Attr<int64_t>("rows_per_image")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

// RelionCoarseDiff2ProjectorPerImageF32 on a live capacity texture: the same per-image fused coarse kernel and
// canonical reduction, sampling the owner's float2 (Re, Im) texels instead of staging the projector's full cube
// into the coarse texture cache. The exact cut scores its undecided rotations against the class texture the coarse
// pass already holds, so no second slab lives beside it outside the XLA pool. The owner must be staged at
// model_max_r (the caller checks its logical radius): fill_relion_texture_capacity_kernel then places the texels
// fill_relion_texture_compact_kernel places from the full cube at the same origin, -(model_max_r * pf + 1), and
// the texels past them (the capacity's physical margin, zero) are never reached by a sample inside the radius.
ffi::Error RelionCoarseDiff2CapacityTexturePerImageF32Impl(
    cudaStream_t stream,
    int64_t owner_handle,
    int64_t current_size,
    int64_t physical_image_size,
    int64_t model_max_r,
    int64_t padding_factor,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    constexpr const char* name = "RelionCoarseDiff2CapacityTexturePerImageF32";
    ffi::Error validation = ValidateRelionCoarseDiff2PerImageF32Operands(
        name, current_size, physical_image_size, model_max_r, padding_factor,
        rotations, images, translation_angles, weight, initial_diff2, full_to_compact, output);
    if (validation.failure()) return validation;
    const auto rotation_dims = rotations.dimensions();
    const auto image_dims = images.dimensions();
    const auto angle_dims = translation_angles.dimensions();

    std::shared_ptr<CapacityRelionHalfTextureF32> owner;
    {
        std::lock_guard<std::mutex> lock(capacity_relion_half_texture_f32_mutex);
        const auto found = capacity_relion_half_texture_f32_registry.find(static_cast<uint64_t>(owner_handle));
        if (found == capacity_relion_half_texture_f32_registry.end())
            return ffi::Error::InvalidArgument(std::string(name) + ": owner handle is not live");
        owner = found->second;
    }
    if (!owner->acquire_call())
        return ffi::Error::InvalidArgument(
            std::string(name) + ": owner is closing, refreshing or failed its last refresh");
    struct Release { CapacityRelionHalfTextureF32* owner; ~Release() { owner->release_call(); } } release{owner.get()};

    // The texture's origin is at the staged (logical) radius, model_max_r in padded units, as the cube path's.
    const int padded_max_r = static_cast<int>(model_max_r * padding_factor);
    if (owner->padding_factor != padding_factor || padded_max_r > owner->half_z / 2 - 1)
        return ffi::Error::InvalidArgument(
            std::string(name) + ": the owner's padding factor or capacity does not hold the model radius");
    const int batch_size = static_cast<int>(image_dims[0]);
    const int rotation_count = static_cast<int>(rotation_dims[1]);
    const int translation_count = static_cast<int>(angle_dims[1]);
    float* out = static_cast<float*>(output->untyped_data());
    const int64_t output_count = static_cast<int64_t>(batch_size) * rotation_count * translation_count;
    constexpr int initialize_block_size = 256;
    relion_coarse_diff2_initialize_f32_kernel<<<
        (output_count + initialize_block_size - 1) / initialize_block_size, initialize_block_size, 0, stream>>>(
            static_cast<const float*>(initial_diff2.untyped_data()), out, rotation_count, translation_count,
            output_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    const int rotation_blocks = (rotation_count + kRelionCoarseEulersPerBlock - 1) / kRelionCoarseEulersPerBlock;
    // RelionCoarseDiff2ProjectorPerImageF32's launch: canonical reduction, per-image poses, the score radius
    // min(model radius, current size / 2) and RELION's projector scale.
    launch_relion_coarse_diff2_projector_f32_variant<
        kRelionCoarseEulersPerBlock, false, true, false, false, true, true>(
            batch_size * rotation_blocks,
            stream,
            owner->texture, 0,
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            out,
            nullptr,
            0, rotation_count, rotation_count, batch_size, translation_count,
            static_cast<int>(image_dims[1]), static_cast<int>(current_size),
            -(padded_max_r + 1), -(padded_max_r + 1),
            static_cast<int>(std::min<int64_t>(model_max_r, current_size / 2)),
            -static_cast<float>(physical_image_size * physical_image_size),
            static_cast<float>(padding_factor));
    err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2CapacityTexturePerImageF32, RelionCoarseDiff2CapacityTexturePerImageF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("owner_handle")
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
        .Ret<ffi::AnyBuffer>()
);
