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
// same project_texture_kernel<true, true> on it without a synchronization.
// Handles are monotonic and resolved through a guarded registry, as for the
// persistent host-uploaded texture above.

struct CapacityRelionHalfTextureF32 {
    cudaArray_t array_real = nullptr;
    cudaArray_t array_imag = nullptr;
    cudaTextureObject_t texture_real = 0;
    cudaTextureObject_t texture_imag = 0;
    int32_t* logical_radius = nullptr;  // device scalar the fill kernel read
    float* staging_real = nullptr;
    float* staging_imag = nullptr;
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
        if (texture_real) cleanup = cudaDestroyTextureObject(texture_real);
        if (texture_imag) { const auto e = cudaDestroyTextureObject(texture_imag); if (cleanup == cudaSuccess) cleanup = e; }
        if (array_real) { const auto e = cudaFreeArray(array_real); if (cleanup == cudaSuccess) cleanup = e; }
        if (array_imag) { const auto e = cudaFreeArray(array_imag); if (cleanup == cudaSuccess) cleanup = e; }
        if (logical_radius) { const auto e = cudaFree(logical_radius); if (cleanup == cudaSuccess) cleanup = e; }
        if (staging_real) { const auto e = cudaFree(staging_real); if (cleanup == cudaSuccess) cleanup = e; }
        if (staging_imag) { const auto e = cudaFree(staging_imag); if (cleanup == cudaSuccess) cleanup = e; }
        texture_real = texture_imag = 0;
        array_real = array_imag = nullptr;
        logical_radius = nullptr;
        staging_real = staging_imag = nullptr;
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
 * z-planes [z_begin, z_begin + n_planes): the same texel values, written into a
 * plane-sized staging buffer and addressed with 64-bit indices. */
__global__ void __launch_bounds__(BLOCK_SIZE)
fill_relion_texture_capacity_planes_kernel(
    const float2* __restrict__ vol, float* real, float* imag,
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
    real[idx] = value.x;
    imag[idx] = value.y;
}

// Staging buffers are at most this many texels per plane group (two float
// buffers of 512 MiB), so the transient beside the texture stays inside the
// XLA pool reserve's slack (relax/helpers/xla_memory_reserve.py).
constexpr int64_t kCapacityTextureStagingTexels = int64_t(1) << 27;

/* The staging of launch_project_texture_float<true> (recovar_cuda_common.cuh):
 * the same texel values into the same float texture pair, filled and copied a
 * group of z-planes at a time so the staging transient is bounded rather than
 * a second copy of the slab. */
static cudaError_t stage_capacity_relion_half_texture_f32(
    const float2* projector_half, int logical_radius, CapacityRelionHalfTextureF32* owner)
{
    const int texX = owner->half_x;
    const int texY = owner->half_y;
    const int texZ = owner->half_z;
    const int64_t plane = static_cast<int64_t>(texX) * texY;
    const int group = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(texZ, kCapacityTextureStagingTexels / plane)));
    float* real = owner->staging_real;
    float* imag = owner->staging_imag;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err == cudaSuccess && !owner->logical_radius)
        err = cudaMalloc(reinterpret_cast<void**>(&owner->logical_radius), sizeof(int32_t));
    if (err == cudaSuccess)
        err = cudaMemcpy(owner->logical_radius, &logical_radius, sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err == cudaSuccess && !real) err = cudaMalloc(reinterpret_cast<void**>(&real), plane * group * sizeof(float));
    if (err == cudaSuccess && !imag) err = cudaMalloc(reinterpret_cast<void**>(&imag), plane * group * sizeof(float));
    if (owner->reusable_staging) {
        owner->staging_real = real;
        owner->staging_imag = imag;
    }
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExtent extent = make_cudaExtent((size_t)texX, (size_t)texY, (size_t)texZ);
    if (err == cudaSuccess && !owner->array_real) err = cudaMalloc3DArray(&owner->array_real, &desc, extent);
    if (err == cudaSuccess && !owner->array_imag) err = cudaMalloc3DArray(&owner->array_imag, &desc, extent);
    for (int z_begin = 0; z_begin < texZ && err == cudaSuccess; z_begin += group) {
        const int n_planes = std::min(group, texZ - z_begin);
        const int64_t texels = plane * n_planes;
        fill_relion_texture_capacity_planes_kernel<<<static_cast<unsigned>((texels + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            projector_half, real, imag, owner->logical_radius, owner->padding_factor, texX, texY, texZ, z_begin, n_planes);
        err = cudaGetLastError();
        cudaMemcpy3DParms copy_params = {0};
        copy_params.extent = make_cudaExtent((size_t)texX, (size_t)texY, (size_t)n_planes);
        copy_params.dstPos = make_cudaPos(0, 0, (size_t)z_begin);
        copy_params.kind = cudaMemcpyDeviceToDevice;
        copy_params.dstArray = owner->array_real;
        copy_params.srcPtr = make_cudaPitchedPtr(real, (size_t)texX * sizeof(float), (size_t)texX, (size_t)texY);
        if (err == cudaSuccess) err = cudaMemcpy3DAsync(&copy_params, stream);
        copy_params.dstArray = owner->array_imag;
        copy_params.srcPtr = make_cudaPitchedPtr(imag, (size_t)texX * sizeof(float), (size_t)texX, (size_t)texY);
        if (err == cudaSuccess) err = cudaMemcpy3DAsync(&copy_params, stream);
    }
    if (err == cudaSuccess && !owner->texture_real) {
        cudaResourceDesc resource_real, resource_imag;
        cudaTextureDesc texture_desc;
        memset(&resource_real, 0, sizeof(resource_real));
        memset(&resource_imag, 0, sizeof(resource_imag));
        memset(&texture_desc, 0, sizeof(texture_desc));
        resource_real.resType = cudaResourceTypeArray;
        resource_real.res.array.array = owner->array_real;
        resource_imag.resType = cudaResourceTypeArray;
        resource_imag.res.array.array = owner->array_imag;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = false;
        texture_desc.addressMode[0] = cudaAddressModeClamp;
        texture_desc.addressMode[1] = cudaAddressModeClamp;
        texture_desc.addressMode[2] = cudaAddressModeClamp;
        err = cudaCreateTextureObject(&owner->texture_real, &resource_real, &texture_desc, nullptr);
        if (err == cudaSuccess)
            err = cudaCreateTextureObject(&owner->texture_imag, &resource_imag, &texture_desc, nullptr);
    }
    /* The texture is complete before create returns; the staging buffers go.
     * Each plane group's buffers are reused only after its copies, which are
     * ordered before the next group's fill on the one stream. */
    if (stream) {
        const cudaError_t sync = cudaStreamSynchronize(stream);
        if (err == cudaSuccess) err = sync;
        cudaStreamDestroy(stream);
    }
    if (!owner->reusable_staging) {
        if (real) cudaFree(real);
        if (imag) cudaFree(imag);
    }
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
        return ffi::Error::InvalidArgument("ProjectRelionHalfCapacityTexture: owner is closing");
    struct Release { CapacityRelionHalfTextureF32* owner; ~Release() { owner->release_call(); } } release{owner.get()};

    const float max_r2 = (float)((owner->half_z / 2 - 1) * (owner->half_z / 2 - 1));
    const int maxR = (int)floorf(sqrtf(max_r2) + 0.5f);
    const int texYInit = -(maxR + 1);
    const int texZInit = -(maxR + 1);
    const int n_pixels = static_cast<int>(o[1]);
    dim3 grid((int)r[0], (n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    project_texture_kernel<true, true><<<grid, block, 0, stream>>>(
        owner->texture_real, owner->texture_imag,
        static_cast<float*>(output->untyped_data()), static_cast<const float*>(rot.untyped_data()),
        n_pixels, (int)image_h, (int)(image_w / 2 + 1), texYInit, texZInit,
        owner->padding_factor, (int)image_w, maxR * maxR, owner->logical_radius, maxR,
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
// without the full crop and the real/imaginary copies.
__global__ void __launch_bounds__(BLOCK_SIZE)
project_capacity_texture_compact_packed_kernel(
    cudaTextureObject_t texReal, cudaTextureObject_t texImag,
    float* __restrict__ out, const float* __restrict__ rot, const int32_t* __restrict__ crop_index,
    int n_compact, int image_h, int image_w, int tex_yinit, int tex_zinit, int upsampling,
    int maxR2_padded, const int32_t* image_radius, float scale)
{
    __shared__ float R[6];
    const int img_idx = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (j >= n_compact) return;
    float* row = out + static_cast<int64_t>(img_idx) * 2 * n_compact;
    const int pix = crop_index[j];
    float re = 0.0f, im = 0.0f;
    bool live = pix >= 0;
    if (live) {
        const int radius = *image_radius;
        if (radius < 0 || radius > image_h / 2) {
            row[j] = nanf("");
            row[n_compact + j] = nanf("");
            return;
        }
        const int padded_radius = radius * upsampling;
        const int max_r2 = min(maxR2_padded, padded_radius * padded_radius);
        const int k0_idx = pix / image_w;
        const int k1_idx = pix % image_w;
        const float k0_unscaled = (float)(k0_idx == 0 ? image_h / 2 : k0_idx - image_h / 2);
        const float k1_unscaled = (float)k1_idx;
        const float rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * (float)upsampling;
        const float rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * (float)upsampling;
        const float rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * (float)upsampling;
        if ((int)(rk0 * rk0 + rk1 * rk1 + rk2 * rk2) <= max_r2) {
            float xp = rk0, yp = rk1, zp = rk2, imag_sign = 1.0f;
            if (xp < 0.0f) { xp = -xp; yp = -yp; zp = -zp; imag_sign = -1.0f; }
            re = tex3D<float>(texReal, xp + 0.5f, yp - (float)tex_yinit + 0.5f, zp - (float)tex_zinit + 0.5f);
            im = imag_sign * tex3D<float>(texImag, xp + 0.5f, yp - (float)tex_yinit + 0.5f, zp - (float)tex_zinit + 0.5f);
        }
    }
    // compute_relion_projector_projections_block's complex64 * float32 dense scale, per component.
    row[j] = __fmul_rn(re, scale);
    row[n_compact + j] = __fmul_rn(im, scale);
}

ffi::Error ProjectRelionHalfCapacityTextureCompactPackedImpl(
    cudaStream_t stream, int64_t owner_handle, int64_t image_h, int64_t image_w, float scale,
    ffi::AnyBuffer rot, ffi::AnyBuffer image_radius, ffi::AnyBuffer crop_index, ffi::Result<ffi::AnyBuffer> output)
{
    const auto r = rot.dimensions();
    const auto c = crop_index.dimensions();
    const auto o = output->dimensions();
    if (rot.element_type() != ffi::DataType::F32 || output->element_type() != ffi::DataType::F32 ||
        image_radius.element_type() != ffi::DataType::S32 || image_radius.dimensions().size() != 0 ||
        crop_index.element_type() != ffi::DataType::S32 || c.size() != 1 || c[0] <= 0)
        return ffi::Error::InvalidArgument(
            "ProjectRelionHalfCapacityTextureCompactPacked: F32 rotations, S32 image radius and crop index, F32 output");
    if (r.size() != 2 || r[1] != 6 || r[0] <= 0 || r[0] > 65535 ||
        image_h <= 0 || image_h != image_w || image_h % 2 != 0 || image_h > 4096 ||
        o.size() != 2 || o[0] != r[0] || o[1] != 2 * c[0] || r[0] * o[1] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument("ProjectRelionHalfCapacityTextureCompactPacked: invalid geometry");
    std::shared_ptr<CapacityRelionHalfTextureF32> owner;
    {
        std::lock_guard<std::mutex> lock(capacity_relion_half_texture_f32_mutex);
        const auto found = capacity_relion_half_texture_f32_registry.find(static_cast<uint64_t>(owner_handle));
        if (found == capacity_relion_half_texture_f32_registry.end())
            return ffi::Error::InvalidArgument("ProjectRelionHalfCapacityTextureCompactPacked: owner handle is not live");
        owner = found->second;
    }
    if (!owner->acquire_call())
        return ffi::Error::InvalidArgument("ProjectRelionHalfCapacityTextureCompactPacked: owner is closing");
    struct Release { CapacityRelionHalfTextureF32* owner; ~Release() { owner->release_call(); } } release{owner.get()};

    // The geometry of ProjectRelionHalfCapacityTextureImpl.
    const float max_r2 = (float)((owner->half_z / 2 - 1) * (owner->half_z / 2 - 1));
    const int maxR = (int)floorf(sqrtf(max_r2) + 0.5f);
    const int n_compact = static_cast<int>(c[0]);
    dim3 grid((int)r[0], (n_compact + BLOCK_SIZE - 1) / BLOCK_SIZE);
    project_capacity_texture_compact_packed_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        owner->texture_real, owner->texture_imag, static_cast<float*>(output->untyped_data()),
        static_cast<const float*>(rot.untyped_data()), static_cast<const int32_t*>(crop_index.untyped_data()),
        n_compact, (int)image_h, (int)(image_w / 2 + 1), -(maxR + 1), -(maxR + 1), owner->padding_factor,
        maxR * maxR, static_cast<const int32_t*>(image_radius.untyped_data()), scale);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
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
