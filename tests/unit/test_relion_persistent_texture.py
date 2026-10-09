"""Persistent texture ownership contracts retained from Q."""
import logging

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from helpers.float_compare import assert_matches
pytestmark = pytest.mark.unit


# Moved from relax/sparse_pass2/dispatch.py (PLAN e1): no relax module uses it, only this test file.
def _open_persistent_relion_projector_texture(
    relion_projector_half,
    *,
    relion_projector_r_max,
    projection_padding_factor,
    relion_texture_interp=None,
    log_label="Sparse pass-2",
):
    """Upload an eligible host ``PPref`` slab once as a persistent float32 RELION texture.

    Returns ``None`` when the slab is not eligible (the caller then projects from
    ``relion_projector_half`` as before). The caller owns the texture and closes it.
    """

    from relax.helpers.projection import host_relion_projector_texture_enabled

    if not host_relion_projector_texture_enabled(
        relion_projector_half, r_max=relion_projector_r_max,
        padding_factor=projection_padding_factor, allow_float32_cast=True,
        enabled=relion_texture_interp,
    ):
        return None

    from relax.cuda.kernels import RelionPersistentHalfTextureF32

    # RELION's texture is float32 (AccProjector::setMdlData); cast before any device upload.
    relion_projector_half = np.asarray(relion_projector_half, dtype=np.complex64)

    logging.getLogger("relax.helpers.oversampling").info(
        "%s persistent RELION projector texture: shape=%s host=%.2f GiB",
        str(log_label),
        tuple(relion_projector_half.shape),
        relion_projector_half.nbytes / float(1024**3),
    )
    return RelionPersistentHalfTextureF32(
        relion_projector_half,
        padding_factor=int(projection_padding_factor),
        projector_max_r=int(relion_projector_r_max),
        projector_scale=1.0,
    )


def _projector(*, r_max=1, padding_factor=1, seed=260830):
    padded = int(r_max) * int(padding_factor)
    shape = (2 * padded + 3, 2 * padded + 3, padded + 2)
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64))


def _rotations():
    base = []
    for angle_x, angle_y, angle_z in (
        (0.37, -0.52, 0.19),
        (-0.91, 0.43, 1.17),
        (1.20, -0.73, -0.44),
        (-0.28, -1.01, 0.66),
    ):
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        rotation_x = np.asarray(
            [[1, 0, 0], [0, cx, -sx], [0, sx, cx]],
            dtype=np.float64,
        )
        rotation_y = np.asarray(
            [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
            dtype=np.float64,
        )
        rotation_z = np.asarray(
            [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        base.append((rotation_z @ rotation_y @ rotation_x).astype(np.float32))
    return np.asarray(base)[np.asarray([3, 1, 3, 0, 2, 1, 0])]


def test_capacity_texture_close_after_concrete_completion(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels

    events = []

    class Finalizer:
        def detach(self):
            events.append("detach")

    texture = object.__new__(em_cuda_kernels.RelionCapacityHalfTextureF32)
    texture._handle = 97
    texture._last_output = None
    texture._finalizer = Finalizer()
    monkeypatch.setattr(
        em_cuda_kernels,
        "_destroy_capacity_half_texture",
        lambda handle, **kwargs: events.append(("destroy", handle)),
    )
    original_ready = jax.block_until_ready

    def ready(value):
        events.append("ready")
        return original_ready(value)

    monkeypatch.setattr(em_cuda_kernels.jax, "block_until_ready", ready)
    completion = jnp.asarray([1.0], dtype=jnp.float32)
    texture.close_after(completion)
    assert events == ["ready", ("destroy", 97), "detach"]
    assert texture.closed
    texture.close_after(completion)
    texture.close()
    assert events == ["ready", ("destroy", 97), "detach"]


def test_capacity_texture_rejects_traced_completion_without_destroy(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels

    events = []
    texture = object.__new__(em_cuda_kernels.RelionCapacityHalfTextureF32)
    texture._handle = 98
    texture._last_output = None
    class Finalizer:
        def detach(self):
            pass

    texture._finalizer = Finalizer()
    monkeypatch.setattr(
        em_cuda_kernels,
        "_destroy_capacity_half_texture",
        lambda *args, **kwargs: events.append("destroy"),
    )

    def attempt(value):
        texture._last_output = value
        with pytest.raises(ValueError, match="concrete result"):
            texture.close_after(value)
        with pytest.raises(ValueError, match="concrete result"):
            texture.close()
        return value

    jax.jit(attempt)(jnp.asarray(1.0)).block_until_ready()
    assert not texture.closed
    assert events == []
    texture.close_after(jnp.asarray(1.0))
    assert texture.closed
    assert events == ["destroy"]


def test_capacity_texture_refresh_requires_concrete_boundary_and_same_geometry(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels

    calls = []

    def native_refresh(handle, pointer, radius, device):
        calls.append((int(handle.value), int(radius.value), int(device.value)))
        return 0

    monkeypatch.setattr(
        em_cuda_kernels, "_get_lib", lambda: type("Lib", (), {"relax_relion_capacity_half_texture_f32_refresh": staticmethod(native_refresh)})()
    )
    texture = object.__new__(em_cuda_kernels.RelionCapacityHalfTextureF32)
    texture._handle = 601
    texture._last_output = None
    texture.reusable_staging = True
    texture.shape = (5, 5, 3)
    texture.padding_factor = 1
    texture.logical_r_max = 1
    texture.device = jax.devices("cpu")[0]
    projector = jax.device_put(np.ones(texture.shape, dtype=np.complex64), texture.device)
    completion = jax.device_put(np.asarray([2.0], dtype=np.float32), texture.device)
    with pytest.raises(ValueError, match="capacity shape"):
        texture.refresh_after(projector[:-1], completion)
    with pytest.raises(ValueError, match="concrete result"):
        texture.refresh_after(projector, 1.0)

    def traced(value):
        with pytest.raises(ValueError, match="concrete result"):
            texture.refresh_after(projector, value)
        return value

    jax.jit(traced)(completion).block_until_ready()
    assert calls == []
    elapsed = texture.refresh_after(projector, completion)
    assert elapsed >= 0
    assert calls == [(601, 1, int(getattr(texture.device, "local_hardware_id", texture.device.id)))]
    assert not texture.closed


def test_persistent_texture_rejects_nonexact_host_inputs():
    from relax.cuda import kernels as em_cuda_kernels

    projector = _projector()
    constructor = em_cuda_kernels.RelionPersistentHalfTextureF32
    with pytest.raises(TypeError, match="NumPy host array"):
        constructor(
            jnp.asarray(projector),
            padding_factor=1,
            projector_max_r=1,
        )
    with pytest.raises(TypeError, match="complex64"):
        constructor(
            projector.astype(np.complex128),
            padding_factor=1,
            projector_max_r=1,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        constructor(
            projector[:, ::-1, :],
            padding_factor=1,
            projector_max_r=1,
        )
    with pytest.raises(ValueError, match="projector_scale=1.0"):
        constructor(
            projector,
            padding_factor=1,
            projector_max_r=1,
            projector_scale=2.0,
        )
    with pytest.raises(ValueError, match="geometry mismatch"):
        constructor(
            projector[:-1],
            padding_factor=1,
            projector_max_r=1,
        )


def _install_fake_texture_runtime(monkeypatch, *, fail_device_put=False):
    import recovar.cuda_backproject as cuda_backproject
    from relax.cuda import kernels as em_cuda_kernels

    events = []

    class FakeDevice:
        platform = "gpu"
        id = 0
        local_hardware_id = 0

    device = FakeDevice()

    def create(*args):
        events.append(("create", args[0]))
        args[-1]._obj.value = 73
        return 0

    def destroy(handle):
        events.append(("destroy", int(handle.value)))
        return 0

    def device_put(value, selected_device):
        assert selected_device is device
        events.append(("device_put", int(np.asarray(value))))
        if fail_device_put:
            raise RuntimeError("synthetic handle transfer failure")
        return jnp.asarray(value, dtype=jnp.uint64)

    def block_until_ready(value):
        events.append(("ready", value))
        return value

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        cuda_backproject.jax,
        "devices",
        lambda backend=None: [device],
    )
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(em_cuda_kernels, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(em_cuda_kernels, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(
        em_cuda_kernels,
        "_persistent_relion_half_texture_c_api",
        lambda: (create, destroy),
    )
    monkeypatch.setattr(cuda_backproject.jax, "device_put", device_put)
    monkeypatch.setattr(cuda_backproject.jax, "block_until_ready", block_until_ready)
    return em_cuda_kernels, device, events


def test_persistent_texture_reuses_one_upload_and_closes_after_readiness(monkeypatch):
    cuda_backproject, device, events = _install_fake_texture_runtime(monkeypatch)
    calls = []

    def fake_project(owner_handle, rotations, **kwargs):
        calls.append((owner_handle, tuple(rotations.shape), kwargs))
        return np.full((rotations.shape[0], 6), 2 + 3j, dtype=np.complex64)

    monkeypatch.setattr(
        cuda_backproject,
        "_relion_projector_persistent_half_texture_f32",
        fake_project,
    )
    texture = cuda_backproject.RelionPersistentHalfTextureF32(
        _projector(),
        padding_factor=1,
        projector_max_r=1,
        device=device,
    )
    rotations = jnp.eye(3, dtype=jnp.float32)[None]
    first = cuda_backproject.relion_projector_persistent_half_texture_f32(
        texture,
        rotations,
        current_size=2,
        padding_factor=1,
        projector_max_r=1,
    )
    second = cuda_backproject.relion_projector_persistent_half_texture_f32(
        texture,
        rotations,
        current_size=2,
        padding_factor=1,
        projector_max_r=1,
    )
    assert len(calls) == 2
    assert sum(event[0] == "create" for event in events) == 1
    assert_matches(first, second)

    pending = object()
    texture._last_output = pending
    texture.close()
    ready_index = next(index for index, event in enumerate(events) if event[0] == "ready" and event[1] is pending)
    destroy_index = next(index for index, event in enumerate(events) if event[0] == "destroy")
    assert ready_index < destroy_index
    texture.close()
    assert sum(event[0] == "destroy" for event in events) == 1

    with pytest.raises(RuntimeError, match="closed"):
        cuda_backproject.relion_projector_persistent_half_texture_f32(
            texture,
            rotations,
            current_size=2,
            padding_factor=1,
            projector_max_r=1,
        )


def test_persistent_texture_geometry_device_and_constructor_cleanup(monkeypatch):
    cuda_backproject, device, events = _install_fake_texture_runtime(monkeypatch)
    texture = cuda_backproject.RelionPersistentHalfTextureF32(
        _projector(),
        padding_factor=1,
        projector_max_r=1,
        device=device,
    )
    with pytest.raises(ValueError, match="geometry"):
        texture._require_live_geometry(
            padding_factor=2,
            projector_max_r=1,
        )
    texture.close()

    with pytest.raises(ValueError, match="not a local JAX GPU"):
        cuda_backproject.RelionPersistentHalfTextureF32(
            _projector(),
            padding_factor=1,
            projector_max_r=1,
            device=object(),
        )

    cuda_backproject, device, failed_events = _install_fake_texture_runtime(
        monkeypatch,
        fail_device_put=True,
    )
    with pytest.raises(RuntimeError, match="handle transfer failure"):
        cuda_backproject.RelionPersistentHalfTextureF32(
            _projector(),
            padding_factor=1,
            projector_max_r=1,
            device=device,
        )
    assert [event[0] for event in failed_events].count("create") == 1
    assert [event[0] for event in failed_events].count("destroy") == 1
    assert events


@pytest.mark.gpu
def test_persistent_host_texture_matches_transient_and_rejects_stale_token(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject
    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    projector = _projector(r_max=7, padding_factor=2)
    rotations = _rotations()

    with jax.default_device(gpu_device):
        rotations_jax = jnp.asarray(rotations)
        transient = em_cuda_kernels.relion_projector_half_texture_f32(
            jnp.asarray(projector),
            rotations_jax,
            current_size=16,
            padding_factor=2,
            projector_max_r=7,
        )
        texture = em_cuda_kernels.RelionPersistentHalfTextureF32(
            projector,
            padding_factor=2,
            projector_max_r=7,
            device=gpu_device,
        )
        handle = texture.owner_handle
        first = em_cuda_kernels.relion_projector_persistent_half_texture_f32(
            texture,
            rotations_jax[:4],
            current_size=16,
            padding_factor=2,
            projector_max_r=7,
        )
        second = em_cuda_kernels.relion_projector_persistent_half_texture_f32(
            texture,
            rotations_jax[4:],
            current_size=16,
            padding_factor=2,
            projector_max_r=7,
        )
        persistent = jnp.concatenate((first, second), axis=0)
        assert texture.owner_handle == handle
        assert_matches(
            np.asarray(persistent),
            np.asarray(transient),
        )
        transient_production, transient_abs2 = (
            compute_relion_projector_projections_block(
                jnp.asarray(projector),
                rotations_jax,
                (16, 16),
                r_max=7,
                padding_factor=2,
                centered_rows=True,
                dense_scale=True,
                projector_output_size=16,
                relion_texture_interp=True,
            )
        )
        assert_matches(
            np.asarray(persistent * np.float32(-256)),
            np.asarray(transient_production),
        )

        persistent_production, persistent_abs2 = compute_relion_projector_projections_block(
            None, rotations_jax, (16, 16), r_max=7, padding_factor=2,
            centered_rows=True, dense_scale=True, projector_output_size=16,
            persistent_texture=texture,
        )
        assert_matches(np.asarray(persistent_production), np.asarray(transient_production))
        assert_matches(np.asarray(persistent_abs2), np.asarray(transient_abs2))

        same_shape = rotations_jax[:4]
        cache_size_after_first_owner = em_cuda_kernels._relion_projector_persistent_half_texture_f32._cache_size()
        stale_token = texture._handle_array
        texture.close()
        texture.close()

        second_texture = em_cuda_kernels.RelionPersistentHalfTextureF32(
            projector,
            padding_factor=2,
            projector_max_r=7,
            device=gpu_device,
        )
        assert second_texture.owner_handle != handle
        em_cuda_kernels.relion_projector_persistent_half_texture_f32(
            second_texture,
            same_shape,
            current_size=16,
            padding_factor=2,
            projector_max_r=7,
        )
        assert (
            em_cuda_kernels._relion_projector_persistent_half_texture_f32._cache_size() == cache_size_after_first_owner
        )

        # A cached executable carrying the old dynamic token must fail while
        # a same-geometry replacement remains live; monotonic handles prevent
        # the stale token from aliasing that new texture.
        with pytest.raises(Exception, match="owner handle is not live"):
            jax.block_until_ready(
                em_cuda_kernels._relion_projector_persistent_half_texture_f32(
                    stale_token,
                    same_shape,
                    current_size=16,
                    padding_factor=2,
                    projector_max_r=7,
                )
            )
        second_texture.close()


@pytest.mark.gpu
@pytest.mark.parametrize('radius,current_size', [(8, 16), (7, 8)])
def test_persistent_texture_preserves_nyquist_and_current_radius(
    monkeypatch, custom_cuda_lib, gpu_device, radius, current_size
):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers.projection import compute_relion_projector_projections_block
    monkeypatch.setenv('RECOVAR_CUDA_LIB', str(custom_cuda_lib))
    monkeypatch.delenv('RECOVAR_DISABLE_CUDA', raising=False)
    projector = _projector(r_max=radius, padding_factor=2)
    with jax.default_device(gpu_device):
        rotations = jnp.concatenate((jnp.eye(3, dtype=jnp.float32)[None], jnp.asarray(_rotations())))
        expected, _ = compute_relion_projector_projections_block(
            jnp.asarray(projector), rotations, (current_size, current_size),
            r_max=radius, padding_factor=2, centered_rows=True, dense_scale=False,
            projector_output_size=current_size, relion_texture_interp=True,
        )
        with em_cuda_kernels.RelionPersistentHalfTextureF32(
            projector, padding_factor=2, projector_max_r=radius, device=gpu_device
        ) as texture:
            actual = em_cuda_kernels.relion_projector_persistent_half_texture_f32(
                texture, rotations, current_size=current_size, padding_factor=2,
                projector_max_r=radius,
            )
            assert_matches(np.asarray(actual), np.asarray(expected))


def test_loader_stays_on_the_ffi_bound_library(monkeypatch, tmp_path):
    """After FFI registration the loader neither re-resolves nor loads another copy."""
    import recovar.cuda_backproject as cb

    bound = tmp_path / "bound" / "libcuda_backproject.so"
    handle = object()
    monkeypatch.setattr(cb, "_ffi_registered", True)
    monkeypatch.setattr(cb, "_lib_handle", handle)
    monkeypatch.setattr(cb, "_loaded_lib_path", bound.resolve())
    monkeypatch.setattr(cb, "_existing_lib_path", lambda: pytest.fail("bound loader re-resolved the library"))
    monkeypatch.setattr(cb.ctypes, "CDLL", lambda path: pytest.fail(f"bound loader opened {path}"))

    monkeypatch.delenv("RECOVAR_CUDA_LIB", raising=False)
    assert cb._get_lib() is handle
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(bound))
    assert cb._get_lib() is handle
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(tmp_path / "other" / "libcuda_backproject.so"))
    with pytest.raises(RuntimeError, match="bound to"):
        cb._get_lib()
@pytest.mark.gpu
def test_bound_cuda_library_keeps_persistent_texture_handles_live(
    monkeypatch, tmp_path, custom_cuda_lib, gpu_device
):
    """Native calls stay on the library whose symbols XLA registered.

    XLA FFI registrations last for the process, and a second copy of the
    library has its own persistent-texture registry. Before the loader was
    bound, pointing RECOVAR_CUDA_LIB at another copy after registration
    created textures that the registered kernel rejected with "owner handle is
    not live"; test_refine_relion_mode hit this whenever an earlier test had
    loaded a different path than its custom_cuda_lib fixture.
    """
    import shutil

    import recovar.cuda_backproject as cb
    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cb, "_cuda_ok", None)
    projector = _projector(r_max=7, padding_factor=2)

    def project():
        with em_cuda_kernels.RelionPersistentHalfTextureF32(
            projector, padding_factor=2, projector_max_r=7, device=gpu_device
        ) as texture:
            return np.asarray(
                em_cuda_kernels.relion_projector_persistent_half_texture_f32(
                    texture, jnp.asarray(_rotations()), current_size=16, padding_factor=2, projector_max_r=7
                )
            )

    # The persistent textures live in the EM library (librelax_cuda.so), which binds like the pipeline one.
    with jax.default_device(gpu_device):
        em_cuda_kernels._ensure_ffi()
        bound = em_cuda_kernels._LIBRARY.loaded_path
        reference = project()
        copy = tmp_path / "second_copy" / "librelax_cuda.so"
        copy.parent.mkdir()
        shutil.copy(bound, copy)
        monkeypatch.setenv("RELAX_CUDA_LIB", str(copy))
        with pytest.raises(RuntimeError, match="bound to"):
            project()
        # Without an explicit request, native calls keep using the bound library.
        monkeypatch.delenv("RELAX_CUDA_LIB")
        assert_matches(project(), reference)
    assert em_cuda_kernels._LIBRARY.loaded_path == bound


def test_large_static_projector_uses_half_storage_without_full_cube(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers import projection
    class HostShape:
        dtype = jnp.dtype(jnp.complex64)
        shape = (1603, 1603, 802)
    slab = HostShape()
    observed = []
    # A box-800 slab (1603 x 1603 x 802 texels) takes the half-storage kernel,
    # as every size within its int32 texel indexing does.
    def project(value, rotations, radius, **kwargs):
        assert value is slab
        observed.append((int(radius), kwargs["image_shape"], kwargs["padding_factor"]))
        return jnp.zeros((rotations.shape[0], 40), dtype=jnp.complex64)
    def forbid_full(*args, **kwargs):
        raise AssertionError('large static projector expanded to a full cube')
    monkeypatch.setattr(em_cuda_kernels, 'project_relion_half_capacity', project)
    monkeypatch.setattr(em_cuda_kernels, 'relion_projector_half_texture_f32', lambda *a, **k: pytest.fail("per-call half texture"))
    monkeypatch.setattr(projection, 'relion_projector_half_to_texture_full', forbid_full)
    result = projection._project_relion_projector_texture(
        slab, jnp.eye(3, dtype=jnp.float32)[None], (8, 8),
        r_max=400, padding_factor=2, projector_output_size=8,
    )
    assert result.shape == (1, 40)
    assert observed == [(400, (8, 8), 2)]


@pytest.mark.parametrize("layout", ["valid", "strided", "double", "class_axis", "missing_radius"])
def test_host_texture_planning_and_dispatch_share_eligibility(monkeypatch, layout):
    from relax.helpers import projection
    calls = []
    monkeypatch.setattr(projection, "_relion_projector_texture_enabled", lambda *a, **kw: calls.append(kw) or True)
    slab = np.zeros((7, 7, 4), dtype=np.complex64)
    radius = 1
    if layout == "strided": slab = slab[:, :, ::2]
    if layout == "double": slab = slab.astype(np.complex128)
    if layout == "class_axis": slab = slab[None]
    if layout == "missing_radius": radius = None
    assert projection.host_relion_projector_texture_enabled(slab, r_max=radius, padding_factor=2) is (layout == "valid")
    assert len(calls) == (1 if layout == "valid" else 0)


@pytest.mark.parametrize("classes", [1, 4])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_projector_class_selection_preserves_host_view(monkeypatch, classes, dtype):
    from relax.classification import k_class_inputs
    source = np.arange(classes * 7 * 7 * 4).reshape(classes, 7, 7, 4).astype(dtype)
    def forbidden_upload(*args, **kwargs):
        raise AssertionError("class selection uploaded the host projector")
    monkeypatch.setattr(k_class_inputs.jnp, "asarray", forbidden_upload)
    result = k_class_inputs.select_projector_half_for_class(source, classes - 1, classes)
    assert isinstance(result, np.ndarray) and result.dtype == dtype
    assert result.shape == (7, 7, 4) and result.flags.c_contiguous
    assert np.shares_memory(source, result)
    assert_matches(result, source[classes - 1])


def test_sparse_pass2_opens_eligible_texture_from_original_host_slab(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers import projection

    projector = _projector()
    created = []
    sentinel = object()

    def fake_constructor(value, **kwargs):
        created.append((value, kwargs))
        return sentinel

    monkeypatch.setattr(
        projection,
        "_relion_projector_texture_enabled",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        em_cuda_kernels,
        "RelionPersistentHalfTextureF32",
        fake_constructor,
    )
    actual = _open_persistent_relion_projector_texture(
        projector,
        relion_projector_r_max=1,
        projection_padding_factor=1,
    )
    assert actual is sentinel
    assert created == [
        (
            projector,
            {
                "padding_factor": 1,
                "projector_max_r": 1,
                "projector_scale": 1.0,
            },
        )
    ]


def test_host_float32_upload_cast_preserves_double_source(monkeypatch):
    from relax.helpers import projection
    from relax.cuda import kernels as em_cuda_kernels
    source = _projector().astype(np.complex128)
    source += np.float64(2**-27)
    original = source.copy()
    captured = []
    monkeypatch.setattr(projection, "_relion_projector_texture_enabled", lambda value, **kw: value.dtype == np.complex64)
    monkeypatch.setattr(em_cuda_kernels, "RelionPersistentHalfTextureF32", lambda value, **kw: captured.append(value) or object())
    assert projection.host_relion_projector_texture_enabled(source, r_max=1, padding_factor=1, allow_float32_cast=True)
    _open_persistent_relion_projector_texture(source, relion_projector_r_max=1, projection_padding_factor=1)
    assert captured[0].dtype == np.complex64
    assert_matches(captured[0], source.astype(np.complex64))
    assert_matches(source, original)
    assert source.dtype == np.complex128 and not np.shares_memory(source, captured[0])


@pytest.mark.gpu
def test_host_cast_matches_device_cast_and_texture_projection(custom_cuda_lib, gpu_device):
    from relax.helpers.projection import compute_relion_projector_projections_block
    source = _projector(r_max=7, padding_factor=2).astype(np.complex128)
    source += np.float64(2**-26)
    source.reshape(-1)[:4] = [complex(-0.0, 0.0), complex(0.0, -0.0), 1 + 2**-24, 1 + 3 * 2**-24]
    with jax.default_device(gpu_device):
        old_double = jnp.asarray(source)
        assert old_double.dtype == jnp.complex128
        old_float = old_double.astype(jnp.complex64)
        assert_matches(np.asarray(old_float), source.astype(np.complex64))
        rotations = jnp.asarray(_rotations())
        kwargs = dict(r_max=7, padding_factor=2, projector_output_size=16, centered_rows=True, dense_scale=True)
        expected = compute_relion_projector_projections_block(old_float, rotations, (16, 16), **kwargs)
        texture = _open_persistent_relion_projector_texture(source, relion_projector_r_max=7, projection_padding_factor=2)
        assert texture is not None
        try:
            actual = compute_relion_projector_projections_block(None, rotations, (16, 16), persistent_texture=texture, **kwargs)
            for a, b in zip(actual, expected):
                assert_matches(np.asarray(a), np.asarray(b))
        finally:
            texture.close()
