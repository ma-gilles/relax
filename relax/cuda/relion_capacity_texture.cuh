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
    int half_z = 0;
    int half_y = 0;
    int half_x = 0;
    int padding_factor = 0;
    int device = -1;

    std::mutex state_mutex;
    std::condition_variable state_changed;
    bool closing = false;
    int active_calls = 0;

    bool acquire_call()
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (closing) return false;
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
        state_changed.wait(lock, [this] { return active_calls == 0; });
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
        texture_real = texture_imag = 0;
        array_real = array_imag = nullptr;
        logical_radius = nullptr;
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

/* The staging of launch_project_texture_float<true> (recovar_cuda_common.cuh),
 * statement for statement, into textures the owner keeps. */
static cudaError_t stage_capacity_relion_half_texture_f32(
    const float2* projector_half, int logical_radius, CapacityRelionHalfTextureF32* owner)
{
    const int texX = owner->half_x;
    const int texY = owner->half_y;
    const int texZ = owner->half_z;
    const int n_voxels = texX * texY * texZ;
    float* real = nullptr;
    float* imag = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&owner->logical_radius), sizeof(int32_t));
    if (err == cudaSuccess)
        err = cudaMemcpy(owner->logical_radius, &logical_radius, sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&real), n_voxels * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&imag), n_voxels * sizeof(float));
    if (err == cudaSuccess) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE);
        fill_relion_texture_capacity_kernel<<<grid, block, 0, stream>>>(
            projector_half, real, imag, owner->logical_radius, owner->padding_factor, texX, texY, texZ);
        err = cudaGetLastError();
    }
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExtent extent = make_cudaExtent((size_t)texX, (size_t)texY, (size_t)texZ);
    if (err == cudaSuccess) err = cudaMalloc3DArray(&owner->array_real, &desc, extent);
    if (err == cudaSuccess) err = cudaMalloc3DArray(&owner->array_imag, &desc, extent);
    if (err == cudaSuccess) {
        cudaMemcpy3DParms copy_params = {0};
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyDeviceToDevice;
        copy_params.dstArray = owner->array_real;
        copy_params.srcPtr = make_cudaPitchedPtr(real, (size_t)texX * sizeof(float), (size_t)texX, (size_t)texY);
        err = cudaMemcpy3DAsync(&copy_params, stream);
        if (err == cudaSuccess) {
            copy_params.dstArray = owner->array_imag;
            copy_params.srcPtr = make_cudaPitchedPtr(imag, (size_t)texX * sizeof(float), (size_t)texX, (size_t)texY);
            err = cudaMemcpy3DAsync(&copy_params, stream);
        }
    }
    if (err == cudaSuccess) {
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
    /* The texture is complete before create returns; the staging buffers go. */
    if (stream) {
        const cudaError_t sync = cudaStreamSynchronize(stream);
        if (err == cudaSuccess) err = sync;
        cudaStreamDestroy(stream);
    }
    if (real) cudaFree(real);
    if (imag) cudaFree(imag);
    return err;
}

/* ``projector_half`` is a device pointer to the C64 [z, y, x>=0] capacity half
 * storage of ProjectRelionHalfImageRadius; the caller keeps it alive and
 * complete until this returns. */
extern "C" int relax_relion_capacity_half_texture_f32_create(
    const void* projector_half, int half_z, int half_y, int half_x,
    int logical_radius, int padding_factor, int device, uint64_t* owner_handle)
{
    if (!projector_half || !owner_handle || half_z < 5 || half_z != half_y ||
        half_z % 2 != 1 || half_x != half_z / 2 + 1 || half_z > 1025 ||
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
        int original_device = -1;
        cudaError_t err = cudaGetDevice(&original_device);
        if (err != cudaSuccess) return static_cast<int>(err);
        if (original_device != device) err = cudaSetDevice(device);
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
