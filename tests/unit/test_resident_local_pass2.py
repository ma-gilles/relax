"""The device-resident local fine pass 2 against the exact local engine (T12).

Ticket: ``em_parity_tickets_20260918/T12_resident_local_search.md``.

The two engines are compared through the production dispatch
(``local_half._run_local_search_iteration``) with the flag off and
on, so the test covers the wiring as well as the driver.

What is compared, and against what
----------------------------------
The resident stages implement the *compact* K=1 pass-2 arithmetic, which is a
different factorization of the same likelihood from the exact local engine's
(module docstring of ``resident_local_pass2`` lists the three differences).
Discrete state is therefore compared for agreement, and the continuous fields
are compared against the engine's own repeat band where one exists and against
explicit, measured bounds otherwise. Nothing here asserts bitwise equality
between the two engines; that would be false by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
import recovar.core.fourier_transform_utils as ftu
from helpers.em_arrays import _hermitian_volume
from helpers.refinement_specs import local_iteration_owners
from helpers.sparse_pass2_mock import IMAGE_SHAPE, VOLUME_SHAPE, MockDataset

from relax.fine_pass import resident_pass2 as rp
from relax.local_search import half
from relax.local_search import resident_pass2 as rlp
from relax.local_search.layout import (
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
)
from relax.relion.projector_setup import reference_to_relion_projector_half_maps
from relax.sampling import build_local_search_grid_metadata

pytestmark = pytest.mark.unit

PARENT_ORDER = 1
OVERSAMPLING = 1
N_IMAGES = 8
CURRENT_SIZE = 6


def _gpu_available():
    if jax.default_backend() != "gpu":
        return False
    from recovar import cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    return bool(
        cuda_backproject.custom_cuda_requested()
        and em_cuda_kernels.sparse_pass2_segmented_supported()
        and em_cuda_kernels.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported()
    )


requires_resident_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="every resident stage is a CUDA FFI target",
)


def _prior_eulers(n_images, seed):
    rng = np.random.default_rng(seed)
    eulers = np.zeros((n_images, 3), dtype=np.float64)
    eulers[:, 0] = rng.uniform(0.0, 360.0, size=n_images)
    eulers[:, 1] = rng.uniform(20.0, 160.0, size=n_images)
    eulers[:, 2] = rng.uniform(0.0, 360.0, size=n_images)
    return eulers


def _pass2_layout(seed=20260919, full_support=False):
    translations = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    parent = build_local_hypothesis_layout(
        _prior_eulers(N_IMAGES, seed),
        None,
        0.35,
        0.35,
        PARENT_ORDER,
        translations,
        np.zeros((N_IMAGES, 2), dtype=np.float32),
        3.0,
        None,
        1.0,
        grid_metadata=build_local_search_grid_metadata(PARENT_ORDER),
        translation_prior_reference_translations=translations,
        dtype=np.float32,
    )
    n_coarse_trans = int(translations.shape[0])
    rng = np.random.default_rng(seed + 1)
    samples = []
    for image in range(parent.n_images):
        if full_support:
            # ``None`` is how RELION's full-parent diagnostic spells "every
            # (rotation, translation) survives"; the layout then carries no
            # mask at all, and the driver must not materialize one.
            samples.append(None)
            continue
        start = int(parent.rotation_offsets[image])
        stop = int(parent.rotation_offsets[image + 1])
        parent_ids = np.asarray(parent.rotation_ids_flat[start:stop], dtype=np.int64)
        pairs = (parent_ids[:, None] * n_coarse_trans + np.arange(n_coarse_trans)).reshape(-1)
        keep = rng.choice(pairs, size=max(2, pairs.size // 3), replace=False)
        samples.append(np.sort(keep.astype(np.int64)))
    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent,
        samples,
        PARENT_ORDER,
        oversampling_order=OVERSAMPLING,
        random_perturbation=0.0,
        dtype=np.float32,
    )
    return layout, translations


def _relion_projector(volume_real, current_size):
    halves, r_max = reference_to_relion_projector_half_maps(
        np.asarray(volume_real, dtype=np.float64)[None],
        current_size=int(current_size),
        padding_factor=1,
        interpolator=1,
    )
    return jnp.asarray(halves[0], dtype=jnp.complex64), int(r_max)


def _case(seed=20260919, full_support=False):
    """Dataset, volume, RELION projector and pass-2 layout for one half."""

    dataset = MockDataset(n_images=N_IMAGES, seed=seed % 2**31)
    dataset.image_source.image_mask = jnp.linspace(
        0.2, 1.0, dataset.image_size, dtype=jnp.float32
    ).reshape(dataset.image_shape)
    volume_ft = _hermitian_volume(VOLUME_SHAPE, seed=17)
    volume_real = np.asarray(
        ftu.get_idft3(np.asarray(volume_ft).reshape(VOLUME_SHAPE)).real, dtype=np.float64
    )
    projector_half, r_max = _relion_projector(volume_real, IMAGE_SHAPE[0])
    layout, translations = _pass2_layout(seed, full_support=full_support)
    n_shells = IMAGE_SHAPE[0] // 2 + 1
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    return dict(
        dataset=dataset,
        volume=jnp.asarray(volume_ft),
        # High enough that the posterior spreads over several candidates: at a
        # low noise level every image has Pmax == 1 and the comparison would
        # only exercise the winner, not the posterior.
        noise_variance=jnp.ones(IMAGE_SHAPE[0] * IMAGE_SHAPE[1], dtype=jnp.float32) * 200.0,
        projector_half=projector_half,
        r_max=r_max,
        layout=layout,
        translations=translations,
        n_shells=n_shells,
        n_half=n_half,
    )


def _run(
    case,
    *,
    monkeypatch,
    current_size=CURRENT_SIZE,
    source_faithful_spectrum_norm=False,
    production_shapes=False,
    projector_dtype=None,
    resident_operands: bool | None = None,
    zero_oversampling: bool = False,
    **owners,
):
    """``production_shapes`` mirrors what the refinement loop actually passes:
    a projector with a singleton class axis, per-image contrast and scale
    corrections, and translation-prior centres."""

    """One fine pass 2 through the production dispatch."""

    if resident_operands is None:
        monkeypatch.delenv(rp._RESIDENT_OPERANDS_ENV, raising=False)
    else:
        monkeypatch.setenv(rp._RESIDENT_OPERANDS_ENV, "1" if resident_operands else "0")
    layout = case["layout"]
    projector = case["projector_half"]
    if projector_dtype is not None:
        projector = projector.astype(projector_dtype)
    image_corrections = scale_corrections = trans_centers = None
    if production_shapes:
        projector = projector[None]  # the loop's singleton class axis
        rng = np.random.default_rng(4242)
        image_corrections = rng.uniform(0.9, 1.1, N_IMAGES).astype(np.float32)
        scale_corrections = rng.uniform(0.9, 1.1, N_IMAGES).astype(np.float32)
        trans_centers = rng.uniform(-0.5, 0.5, (N_IMAGES, 2)).astype(np.float32)
    return half._run_local_search_iteration(*local_iteration_owners(
        case["dataset"],
        case["volume"],
        case["noise_variance"],
        _prior_eulers(N_IMAGES, 20260919),
        None,
        PARENT_ORDER + OVERSAMPLING,
        0.35,
        0.35,
        case["translations"],
        np.zeros((N_IMAGES, 2), dtype=np.float32),
        3.0,
        "linear_interp",
        current_size=current_size,
        reconstruction_current_size=current_size,
        accumulate_noise=True,
        projection_padding_factor=1,
        reconstruction_padding_factor=1,
        half_spectrum_scoring=True,
        relion_exact_score_translation=True,
        projection_relion_texture_interp=None,
        relion_projector_half=projector,
        relion_projector_r_max=case["r_max"],
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        translation_prior_centers=trans_centers,
        square_window=False,
        group_ids=np.zeros(N_IMAGES, dtype=np.int32),
        scale_correction_group_count=1,
        scale_correction_data_vs_prior=np.full(case["n_shells"], 5.0, dtype=np.float64),
        mstep_relion_x_half=True,
        # The zero-oversampling route keeps every weight for the reconstruction
        # and the statistics (dense_half.py, local_reconstruct_significant_only).
        reconstruct_significant_only=not zero_oversampling,
        stats_use_reconstruction_probs=not zero_oversampling,
        adaptive_fraction=0.999,
        max_significants=-1,
        return_best_pose_details=True,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        pass2_layout=layout,
        **owners,
    ))


@pytest.fixture
def _resident_local_env(monkeypatch):
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "64,256,1024")
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES", "2,4,8")


class _Dispatched(Exception):
    pass


def _dispatched_arguments(monkeypatch, **run):
    """The arguments the production dispatch hands the resident driver, bound to its signature."""
    import inspect

    signature = inspect.signature(rlp.compute_local_search_resident)
    bound = []

    def resident(*args, **kwargs):
        bound.append((signature.bind(*args, **kwargs), set(kwargs)))
        raise _Dispatched

    monkeypatch.setattr(half, "compute_local_search_resident", resident)
    with pytest.raises(_Dispatched):
        _run(_case(), monkeypatch=monkeypatch, **run)
    (arguments, keywords), = bound
    return arguments.arguments, keywords


@pytest.mark.parametrize("route", ["fine", "zero_oversampling", "full_box", "parent_probe"])
def test_dispatch_routes_the_fine_pass_and_the_parent_probe(monkeypatch, route):
    """The wiring: the fine pass and the pass-1 parent probe both run on the resident
    driver, which is the one local engine (local searches are K=1 only): the score-only probe, the
    zero-oversampling route (every scored sample) and RELION's final all-data full box included."""
    run = {
        "fine": {},
        "zero_oversampling": {"zero_oversampling": True},
        "full_box": {"current_size": IMAGE_SHAPE[0]},
        "parent_probe": {"score_only": True, "disable_adjoint_y": True, "disable_adjoint_ctf": True},
    }[route]
    arguments, _ = _dispatched_arguments(monkeypatch, **run)
    assert arguments["support"].score_only is (route == "parent_probe")


def test_dispatch_call_keywords_are_resident_parameters(monkeypatch):
    """Every argument the dispatcher passes must be a resident-driver parameter.

    A refactor narrowed the resident signature while the dispatcher kept
    passing ``do_gridding_correction``; the resulting TypeError only surfaced
    on GPU. The resident driver requires the RELION PPref projector, for which
    the exact local engine applies no gridding correction either. Binding the
    call to the driver's signature fails on any unknown keyword. The support
    choices travel in the support record.
    """
    arguments, keywords = _dispatched_arguments(monkeypatch)
    assert keywords == {"translation_prior_centers", "symmetry_label"}
    assert set(arguments) == {"data", "local_layout", "kernel", "support", *keywords}
    assert arguments["support"].applied_max_significants == -1
    assert arguments["support"].stats_use_reconstruction_probs is True


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"mstep_relion_x_half": False}, "x-half M-step"),
        ({"accumulate_noise": False}, "noise statistics"),
        ({"use_float64_scoring": True}, "float64"),
        ({"reconstruct_significant_only": False}, "same \\(pruned or complete\\) weights"),
        ({"relion_exact_score_translation": False}, "translation angles"),
        ({"relion_projector_half": None}, "PPref projector"),
        ({"group_ids": None}, "group scale terms"),
        ({"normalization_log_evidence": np.zeros(3)}, "externally supplied normalizer"),
        ({"return_reconstruction_sample_indices": True}, "significant-sample capture"),
        ({"use_window": False}, "radial window"),
        ({"max_significants": 500}, "maximum_significants cap"),
        ({"stats_use_reconstruction_probs": False}, "same \\(pruned or complete\\) weights"),
    ],
)
def test_gate_names_the_missing_piece(override, expected):
    kwargs = dict(
        score_only=False,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        mstep_relion_x_half=True,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        max_significants=-1,
        stats_use_reconstruction_probs=True,
        use_float64_scoring=False,
        use_float64_projections=False,
        relion_exact_score_translation=True,
        half_spectrum_scoring=True,
        relion_projector_half=object(),
        relion_projector_r_max=4,
        normalization_log_evidence=None,
        return_reconstruction_sample_indices=False,
        group_ids=np.zeros(3, dtype=np.int32),
        use_window=True,
        relion_wavg_atomic_scale_aa=True,
        relion_wavg_atomic_direct_noise=True,
        relion_wavg_atomic_direct_norm=False,
    )
    kwargs.update(override)
    with pytest.raises(NotImplementedError, match=expected):
        rlp.require_resident_local_configuration(**kwargs)


_PROBE_KWARGS = dict(
    score_only=True,
    disable_adjoint_y=True,
    disable_adjoint_ctf=True,
    mstep_relion_x_half=False,
    accumulate_noise=False,
    reconstruct_significant_only=True,
    max_significants=-1,
    stats_use_reconstruction_probs=True,
    use_float64_scoring=False,
    use_float64_projections=False,
    relion_exact_score_translation=True,
    half_spectrum_scoring=True,
    relion_projector_half=object(),
    relion_projector_r_max=4,
    normalization_log_evidence=None,
    return_reconstruction_sample_indices=True,
    group_ids=None,
    use_window=True,
    relion_wavg_atomic_scale_aa=False,
    relion_wavg_atomic_direct_noise=False,
    relion_wavg_atomic_direct_norm=False,
)


def test_parent_probe_configuration_is_accepted_without_the_mstep_pieces():
    """RELION's pass 1 has no M-step: no noise statistics, x-half accumulators or scale groups."""

    rlp.require_resident_local_configuration(**_PROBE_KWARGS)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"return_reconstruction_sample_indices": False}, "returns its significant samples"),
        ({"max_significants": 500}, "maximum_significants cap on the pass-1 support"),
        ({"use_float64_scoring": True}, "float64"),
        ({"relion_exact_score_translation": False}, "translation angles"),
        ({"half_spectrum_scoring": False}, "half-spectrum"),
        ({"relion_projector_half": None}, "PPref projector"),
        ({"normalization_log_evidence": np.zeros(3)}, "externally supplied normalizer"),
        ({"use_window": False}, "current-size window"),
    ],
)
def test_parent_probe_gate_names_the_missing_piece(override, expected):
    kwargs = dict(_PROBE_KWARGS)
    kwargs.update(override)
    with pytest.raises(NotImplementedError, match=expected):
        rlp.require_resident_local_configuration(**kwargs)


def test_projector_call_bound_covers_the_shape_that_ran_out_of_memory():
    """The end-to-end on arm died at current size 52, order 4, asking 16.12 GiB.

    The shared compact projection-block helper returns full half-spectrum rows
    and windows them afterwards, so one call holds
    ``rows x n_half x itemsize(Projector::data)`` whatever the window is. At a
    256 box with a complex128 slab that is 528 KiB per row, and the 32768-row
    chunk the window-based budget allowed asked for 16.12 GiB. Pin the bound at
    that shape and at the state-C shape beside it.
    """

    n_half = 256 * (256 // 2 + 1)
    assert n_half == 33024
    budget = rlp._projection_call_transient_max_bytes()
    for window_px, slab_bytes in ((1104, 16), (3387, 16), (1022, 16), (3387, 8)):
        rows = max(1, budget // max(n_half * slab_bytes, 1))
        peak = rows * n_half * slab_bytes
        assert peak <= budget, (window_px, slab_bytes, peak)
        # The window must not enter the bound: the helper materializes n_half.
        assert rows == max(1, budget // (n_half * slab_bytes))
    # The allocation that failed, and what the bound permits in its place.
    failed_rows, slab_bytes = 32768, 16
    assert failed_rows * n_half * slab_bytes / 1024 ** 3 > 16.0
    bounded_rows = max(1, budget // (n_half * slab_bytes))
    assert bounded_rows == 8128
    assert bounded_rows * n_half * slab_bytes / 1024 ** 3 <= 4.0


def test_plan_log_line_formats_without_a_logging_error(caplog):
    """The plan line's placeholders and arguments must agree.

    A mismatch does not fail the run, because logging swallows it, but it
    replaces every plan line in a measured arm's log with a traceback, which is
    how the projector-call bound was nearly impossible to read back.
    """

    import inspect
    import re

    src = inspect.getsource(rlp.compute_local_search_resident)
    start = src.index('"Resident local pass-2 plan:')
    call = src.rindex("logger.info(", 0, start)
    indent = call - src.rindex("\n", 0, call) - 1
    block = src[start : src.index("\n" + " " * indent + ")\n", start)]
    fmt = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', block))
    placeholders = len(re.findall(r"%[-0-9.]*[dsfgex]", fmt))
    arguments = len(
        [line for line in block.split("\n") if line.strip() and not line.strip().startswith('"')]
    )
    assert placeholders == arguments, (placeholders, arguments)


def test_row_capacity_ladder_is_capped_by_the_projection_budget():
    ladder = rlp._cap_row_capacity_ladder(
        (1024, 4096, 16384),
        n_score_pixels=3386,
        n_recon_pixels=4324,
        max_bytes=200 * 1024**2,
    )
    assert ladder and list(ladder) == sorted(ladder)
    # A budget that fits nothing still leaves the smallest class so a plan exists.
    assert rlp._cap_row_capacity_ladder(
        (1024, 4096), n_score_pixels=3386, n_recon_pixels=4324, max_bytes=1
    ) == (1024,)


def test_local_projector_texture_is_not_opened_for_manual_or_double_projection():
    slab = np.zeros((11, 11, 6), dtype=np.complex64)
    kwargs = dict(relion_projector_r_max=4, projection_padding_factor=1)
    projection_kwargs = {"projector_output_size": 8, "relion_texture_interp": None}
    assert rlp._open_capacity_texture(None, projection_kwargs=projection_kwargs, **kwargs) is None
    assert (
        rlp._open_capacity_texture(slab, projection_kwargs=dict(projection_kwargs, relion_texture_interp=False), **kwargs)
        is None
    )
    assert rlp._open_capacity_texture(slab.astype(np.complex128), projection_kwargs=projection_kwargs, **kwargs) is None


def _parent_layout(seed=20260919):
    translations = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    parent = build_local_hypothesis_layout(
        _prior_eulers(N_IMAGES, seed),
        None,
        0.35,
        0.35,
        PARENT_ORDER,
        translations,
        np.zeros((N_IMAGES, 2), dtype=np.float32),
        3.0,
        None,
        1.0,
        grid_metadata=build_local_search_grid_metadata(PARENT_ORDER),
        translation_prior_reference_translations=translations,
        dtype=np.float32,
    )
    return parent, translations


@requires_resident_gpu
def test_resident_local_capacity_texture_matches_the_per_call_texture(monkeypatch, _resident_local_env):
    """One texture staged for the pass projects the rows the per-call staging projects.

    EMPIAR-10202 (14507538): the per-call texture allocated the slab's CUDA arrays on
    every chunk and ran out of memory; every local slab now stages its texture once.
    """

    case = _case()
    opened = []
    real_open = rlp._open_capacity_texture

    def spy(*args, **kwargs):
        texture = real_open(*args, **kwargs)
        opened.append(texture is not None)
        return texture

    monkeypatch.setattr(rlp, "_open_capacity_texture", spy)
    # EMPIAR-10202 full-box pass (14575557): the device slab next to the staged
    # texture left no free block for the x-half accumulators, so the chunks of a
    # staged pass get only the slab's geometry.
    chunk_slabs = []
    real_start = rlp._start_resident_local_chunk

    def start_spy(*args, **kwargs):
        chunk_slabs.append(type(kwargs["relion_projector_half"]))
        return real_start(*args, **kwargs)

    monkeypatch.setattr(rlp, "_start_resident_local_chunk", start_spy)
    staged = _run(case, monkeypatch=monkeypatch)
    assert opened == [True]
    assert chunk_slabs and set(chunk_slabs) == {jax.ShapeDtypeStruct}
    chunk_slabs.clear()
    monkeypatch.setattr(rlp, "_open_capacity_texture", lambda *a, **k: None)
    per_call = _run(case, monkeypatch=monkeypatch)
    assert chunk_slabs and all(issubclass(kind, jax.Array) for kind in chunk_slabs)

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))

    assert_matches(np.asarray(per_call.hard_assignment), np.asarray(staged.hard_assignment))
    # Same texture kernel, same texels: the resident driver's own repeat band (1e-7).
    assert rel_l2(per_call.Ft_y, staged.Ft_y) < 1e-6
    assert rel_l2(per_call.Ft_ctf, staged.Ft_ctf) < 1e-6


@requires_resident_gpu
@pytest.mark.parametrize("unshifted", [True, False], ids=["unshifted-operands", "translated-tiles"])
def test_local_chunk_tile_count_matches_the_live_translated_arrays(monkeypatch, _resident_local_env, unshifted):
    """The capacity plan's per-stage tile counts are the translated arrays a chunk
    actually holds, for both operand families. EMPIAR-10202 it22 (14509861) ran out
    of memory when the plan counted three recon tiles and the tile preparation held
    about ten; a new translated array must update ``resident_pass2.chunk_translated_tile_pixels``."""

    from relax.fine_pass import resident_pass2 as rp_module

    planned = []
    measured = []
    baseline = set()
    real_plan = rp_module.plan_resident_chunk_memory
    real_prepare = rp_module._prepare_chunk_reconstruction_operands
    real_rows = rp_module._chunk_operand_rows
    real_unshifted = rlp._unshifted_chunk_operands
    real_gather = rp_module.gather_resident_chunk_operands

    def plan(**kwargs):
        planned.append(kwargs)
        return real_plan(**kwargs)

    def translated_bytes(capacity, n_trans):
        total = 0
        for array in jax.live_arrays():
            if id(array) in baseline:
                continue
            if (array.ndim == 2 and array.shape[0] == capacity * n_trans) or (
                array.ndim == 3 and tuple(array.shape[:2]) == (capacity, n_trans)
            ):
                total += array.nbytes
        return total

    def record(key, capacity, n_trans):
        measured.append((key, capacity * n_trans, translated_bytes(capacity, n_trans)))

    def prepare(**kwargs):
        capacity, n_trans = int(kwargs["chunk"].image_capacity), int(kwargs["n_fine_trans"])
        baseline.clear()
        baseline.update(id(a) for a in jax.live_arrays())

        def rows(arrays, *args, **row_kwargs):
            out = real_rows(arrays, *args, **row_kwargs)
            record("prepare_tile_pixels", capacity, n_trans)
            return out

        monkeypatch.setattr(rp_module, "_chunk_operand_rows", rows)
        try:
            result = real_prepare(**kwargs)
        finally:
            monkeypatch.setattr(rp_module, "_chunk_operand_rows", real_rows)
        record("held_tile_pixels", capacity, n_trans)
        return result

    def unshifted_operands(*args, **kwargs):
        capacity, n_trans = int(kwargs["image_capacity"]), int(kwargs["n_fine_trans"])
        baseline.clear()
        baseline.update(id(a) for a in jax.live_arrays())

        def gather(*gather_args, **gather_kwargs):
            out = real_gather(*gather_args, **gather_kwargs)
            record("prepare_tile_pixels", capacity, n_trans)
            return out

        monkeypatch.setattr(rp_module, "gather_resident_chunk_operands", gather)
        try:
            result = real_unshifted(*args, **kwargs)
        finally:
            monkeypatch.setattr(rp_module, "gather_resident_chunk_operands", real_gather)
        record("held_tile_pixels", capacity, n_trans)
        return result

    monkeypatch.setattr(rp_module, "plan_resident_chunk_memory", plan)
    monkeypatch.setattr(rp_module, "_prepare_chunk_reconstruction_operands", prepare)
    monkeypatch.setattr(rlp, "_unshifted_chunk_operands", unshifted_operands)
    _run(_case(), monkeypatch=monkeypatch, resident_operands=unshifted)
    assert len(planned) == 1 and len(measured) >= 2
    for key, image_translations, live in measured:
        assert live == image_translations * planned[0][key] * 8, (key, live, planned[0][key])


def test_driver_does_not_narrow_the_projector_slab():
    """Narrowing Projector::data is a change of projection arithmetic.

    It also swaps the vmapped fallback for the texture projector, so a narrowed
    arm is both a different computation and a faster one than its control. The
    exact local engine preserves the slab's dtype and does not read the compact
    engine's opt-in gate, so neither does this driver.
    """

    import inspect

    source = inspect.getsource(rlp.compute_local_search_resident)
    assert "prepare_local_projector_slab" in source
    assert "astype(jnp.complex64)" not in source
    assert "_pass2_projector_complex64_enabled" not in source


@requires_resident_gpu
def test_resident_local_repeats_itself(monkeypatch, _resident_local_env):
    """The resident driver's own repeat band, the reference for the table above."""

    case = _case()
    first = _run(case, monkeypatch=monkeypatch)
    second = _run(case, monkeypatch=monkeypatch)
    assert_matches(
        np.asarray(first.hard_assignment), np.asarray(second.hard_assignment)
    )
    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # Two unordered float32 reductions run per chunk: the x-half BPref atomics
    # and the flat-row Wavg rotation atomic that feeds the direct low-shell
    # noise residual. Measured repeat band on an A100: 1.0e-7 for the maps and
    # 2.5e-8 relative for the noise shells, so neither is asserted bitwise.
    assert rel_l2(first.Ft_y, second.Ft_y) < 1e-6
    assert rel_l2(
        first.noise_stats.wsum_sigma2_noise, second.noise_stats.wsum_sigma2_noise
    ) < 1e-6


def test_mstep_adapter_refuses_unshifted_operands_without_the_kernel_tables():
    """The local M-step entry point fails closed on T16's operand family without its tables.

    Unshifted per-image operands are translated inside the translate-and-sum
    kernel, which needs the reconstruction pixel indices and the translation
    angles; a ``recon`` of that family handed without them must be refused by
    name rather than reach the kernel with ``None`` tables.
    """

    resident_recon = {
        "shifted_recon": None,
        "shifted_noise": None,
        "recon_image": object(),
        "recon_weight": None,
        "noise_image": object(),
        "ctf2_over_nv_recon": object(),
        "direct_ctf_rfloat_recon": None,
        "raw_translated_wavg_rectangle": object(),
        "raw_translated_wavg_for_atomic": object(),
        "scale": object(),
    }
    with pytest.raises(ValueError, match="translate-sum kernel"):
        rp.run_resident_mstep_blocks(
            lambda start, stop: None,
            row_capacity=64,
            n_valid_rows=8,
            mstep_block_rows=64,
            image_capacity=2,
            row_image_local=None,
            kernel_row_image_ids=None,
            row_posterior=np.zeros((64, 1), dtype=np.float32),
            recon=resident_recon,
            n_rect=1,
            n_shells=2,
            n_recon_windowed=1,
            noise_variance_for_noise=None,
            shell_indices_noise=None,
            exact_positions_device=None,
            Ft_y_total=None,
            Ft_ctf_total=None,
            image_shape=IMAGE_SHAPE,
            recon_volume_shape=VOLUME_SHAPE,
            mstep_current_size=CURRENT_SIZE,
            relion_x_half_recon_indices=None,
            max_adjoint_block_bytes=1 << 20,
            cuda_backproject=None,
        )


@requires_resident_gpu
def test_local_chunk_runs_with_the_once_per_half_operand_flag(
    monkeypatch, _resident_local_env
):
    """One local-search chunk through the M-step adapter with T16's flag on.

    The T16 merge added ``recon_image``/``recon_weight``/``noise_image`` to
    ``_ChunkStageOperands`` and ``recon_pixel_indices`` to ``_ChunkStageTables``
    for the once-per-half kernel path. The local adapter still built the old
    field sets, so every resident local chunk raised ``TypeError``; the
    full-wave end-to-end (job 14168000) died at its first local search,
    iteration 16, while every global-order matched pair passed.

    The local pass prepares its own pre-shifted per-chunk tiles, so
    ``RELAX_SPARSE_PASS2_RESIDENT_OPERANDS`` must be inert here: both
    settings have to run the adapter and agree to the driver's own repeat band.
    """

    case = _case()
    calls = []
    real_mstep_blocks = rp.run_resident_mstep_blocks

    def counting_mstep_blocks(*args, **kwargs):
        calls.append(kwargs.get("image_capacity"))
        return real_mstep_blocks(*args, **kwargs)

    monkeypatch.setattr(rp, "run_resident_mstep_blocks", counting_mstep_blocks)

    on = _run(case, monkeypatch=monkeypatch, resident_operands=True)
    assert calls, "the local pass must reach run_resident_mstep_blocks"
    off = _run(case, monkeypatch=monkeypatch, resident_operands=False)

    assert_matches(
        np.asarray(on.hard_assignment), np.asarray(off.hard_assignment)
    )
    assert_matches(
        np.asarray(on.best_pose_rotations), np.asarray(off.best_pose_rotations)
    )
    assert_matches(
        np.asarray(on.best_pose_translations), np.asarray(off.best_pose_translations)
    )

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # The flag selects nothing on this path, so the only spread is the same
    # float32 atomics the repeat test bounds at 1e-6.
    assert rel_l2(on.Ft_y, off.Ft_y) < 1e-6
    assert rel_l2(on.Ft_ctf, off.Ft_ctf) < 1e-6
    assert rel_l2(
        on.noise_stats.wsum_sigma2_noise, off.noise_stats.wsum_sigma2_noise
    ) < 1e-6
    np.testing.assert_allclose(
        np.asarray(on.relion_stats.max_posterior_per_image, dtype=np.float64),
        np.asarray(off.relion_stats.max_posterior_per_image, dtype=np.float64),
        rtol=0,
        atol=1e-6,
    )


@requires_resident_gpu
def test_block_row_program_matches_the_slicing_callback(monkeypatch, _resident_local_env):
    """P3-G: the chunk-array form against the per-block Python callback.

    With ``RELAX_LOCAL_SEARCH_RESIDENT_BLOCK_ROW_PROGRAM=1`` the chunk's three
    row arrays go to ``run_resident_mstep_blocks`` whole and the block program
    reads its own rows; with the flag off the callback slices them per block.
    The rows the two forms hand the M-step body match, which
    ``tests/unit/test_p3g_local_glue_programs.py`` asserts directly on CPU, so
    every discrete output must agree exactly here. The accumulators go through
    the x-half BPref atomics and the flat-row Wavg rotation atomic, which are
    not bit-reproducible in either arm, so they are held to the driver's own
    repeat band (``test_resident_local_repeats_itself``) rather than bitwise.
    """

    case = _case()
    monkeypatch.setenv(rlp._BLOCK_ROW_PROGRAM_ENV, "0")
    off = _run(case, monkeypatch=monkeypatch, production_shapes=True)
    monkeypatch.setenv(rlp._BLOCK_ROW_PROGRAM_ENV, "1")
    on = _run(case, monkeypatch=monkeypatch, production_shapes=True)

    assert_matches(
        np.asarray(off.hard_assignment), np.asarray(on.hard_assignment)
    )
    assert_matches(
        np.asarray(off.best_pose_rotations), np.asarray(on.best_pose_rotations)
    )
    assert_matches(
        np.asarray(off.best_pose_translations), np.asarray(on.best_pose_translations)
    )

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    assert rel_l2(off.Ft_y, on.Ft_y) < 1e-6
    assert rel_l2(off.Ft_ctf, on.Ft_ctf) < 1e-6
    assert rel_l2(
        off.noise_stats.wsum_sigma2_noise, on.noise_stats.wsum_sigma2_noise
    ) < 1e-6


@requires_resident_gpu
def test_full_box_iteration_without_a_current_size_runs_resident(monkeypatch, _resident_local_env):
    """A regular iteration at the full box passes current_size=None (MS2 box 512 it25,
    bench 14641044, which the dispatcher refused); it is the box-size pass."""

    case = _case()
    box = int(case["dataset"].image_shape[0])
    unset = _run(case, monkeypatch=monkeypatch, current_size=None)
    explicit = _run(case, monkeypatch=monkeypatch, current_size=box)
    assert_matches(np.asarray(explicit.hard_assignment), np.asarray(unset.hard_assignment))
    a = np.asarray(explicit.Ft_y, dtype=np.float64)
    b = np.asarray(unset.Ft_y, dtype=np.float64)
    # The same pass twice: only the float32 BPref atomics' order may differ.
    assert np.linalg.norm(a - b) <= np.sqrt(N_IMAGES) * np.finfo(np.float32).eps * np.linalg.norm(a)


@requires_resident_gpu
def test_padded_posterior_bins_serve_two_bin_counts_with_one_program(monkeypatch, _resident_local_env):
    """K=1 posterior bins padded to the quantum: halves whose used-bin counts differ share the
    image-term program, the padding holds no mass, and the results are the unpadded ones."""

    cases = [_case(seed=20260919), _case(seed=20261006)]

    def runs(quantum):
        monkeypatch.setattr(rlp, "_POSTERIOR_BIN_QUANTUM", quantum)
        programs = []
        results = []
        for case in cases:
            results.append(_run(case, monkeypatch=monkeypatch))
            programs.append(rp._accumulate_chunk_image_terms._cache_size())
        return results, programs

    unpadded, unpadded_programs = runs(1)
    # The two layouts use different numbers of posterior bins: unpadded, each compiles its own program.
    assert unpadded_programs[1] == unpadded_programs[0] + 1
    padded, padded_programs = runs(4096)
    # Padded, the second half reuses the first one's program.
    assert padded_programs[1] == padded_programs[0]

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        return float(np.linalg.norm(a - b) / np.linalg.norm(a))

    for got, want in zip(padded, unpadded):
        assert_matches(np.asarray(got.hard_assignment), np.asarray(want.hard_assignment))
        assert_matches(
            np.asarray(got.relion_stats.rotation_posterior_sums),
            np.asarray(want.relion_stats.rotation_posterior_sums),
            rtol=1e-6,
        )
        # The repeat band of test_resident_local_repeats_itself (float32 BPref atomics).
        assert rel_l2(got.Ft_y, want.Ft_y) < 1e-6


def test_expanded_posterior_bins_ignore_the_padding():
    """The padded tail of the device sums is never read into the dense histogram."""

    bins = np.array([3, 7, 11])
    sums = np.array([1.0, 2.0, 3.0, 99.0, 99.0])
    dense = rlp._expand_posterior_bins(sums, bins, 12)
    assert_matches(dense[bins], sums[:3])
    assert not np.delete(dense, bins).any()
    assert rlp._posterior_bin_capacity(5785, n_classes=1) == 8192
    assert rlp._posterior_bin_capacity(4096, n_classes=1) == 4096
    assert rlp._posterior_bin_capacity(5785, n_classes=2) == 5785


def test_rotation_posterior_is_accumulated_over_the_used_bins():
    """MS2 box 512's final pass (bench 14684161): a dense device histogram of the layout's
    rotations was 9.60 GiB. The pass accumulates over the bins its rows use and expands
    them on the host; the expanded histogram is the dense one."""

    rng = np.random.default_rng(7)
    n_bins = 10_000
    row_ids = rng.choice(n_bins, size=500).astype(np.int32)
    mass = rng.random(500)
    dense = np.zeros(n_bins)
    np.add.at(dense, row_ids, mass)
    bins, row_bins = np.unique(row_ids, return_inverse=True)
    compact = np.zeros(bins.size)
    np.add.at(compact, row_bins, mass)
    np.testing.assert_allclose(rlp._expand_posterior_bins(compact, bins, n_bins), dense, rtol=1e-12, atol=0)


@requires_resident_gpu
@pytest.mark.parametrize("zero_oversampling", [True, False], ids=["all-weights", "pruned"])
def test_two_identical_classes_split_the_single_class_pass(monkeypatch, _resident_local_env, zero_oversampling):
    """Two copies of one reference, no class prior: each class carries half of the K=1 pass.

    RELION's Class3D local search scores every class at the particle's local orientations and
    normalizes over classes and poses jointly, with no pdf_class in the weights; with identical
    classes the joint evidence is the K=1 evidence plus log 2, each class's evidence and winner are
    the K=1 ones, and the two BPrefs and the noise sums add up to the K=1 pass.
    """

    case = _case()
    single = _run(case, monkeypatch=monkeypatch, zero_oversampling=zero_oversampling)
    doubled = dict(
        case,
        volume=jnp.stack([case["volume"], case["volume"]]),
        projector_half=jnp.stack([case["projector_half"], case["projector_half"]]),
    )
    joint = _run(doubled, monkeypatch=monkeypatch, zero_oversampling=zero_oversampling, n_classes=2).class_pass

    def rel_l2(a, b):
        a, b = np.asarray(a, dtype=np.complex128), np.asarray(b, dtype=np.complex128)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    k1_evidence = np.asarray(single.relion_stats.log_evidence_per_image, dtype=np.float64)
    assert np.max(np.abs(np.asarray(joint.stats.log_evidence_per_image) - (k1_evidence + np.log(2.0)))) < 1e-4
    for k in range(2):
        assert np.max(np.abs(np.asarray(joint.class_log_evidence_per_image[k]) - k1_evidence)) < 1e-4
        assert_matches(np.asarray(joint.per_class_hard_assignments[k]), np.asarray(single.hard_assignment))
    if zero_oversampling:
        # Every weight is kept, so each class's BPref is exactly half of the K=1 one up to float32 order.
        assert rel_l2(np.asarray(joint.Ft_y[0]) + np.asarray(joint.Ft_y[1]), single.Ft_y) < 1e-5
        assert rel_l2(np.asarray(joint.Ft_ctf[0]) + np.asarray(joint.Ft_ctf[1]), single.Ft_ctf) < 1e-5
        assert rel_l2(joint.noise_stats.wsum_sigma2_noise, single.noise_stats.wsum_sigma2_noise) < 1e-5
        mass = np.asarray(joint.class_reconstruction_posterior_sums)
        assert np.max(np.abs(mass - N_IMAGES / 2.0)) < 1e-4
    else:
        # The pruned support is joint over the duplicated rows, so a cutoff tie may keep one copy of a
        # pair; the totals still agree to the pruned mass.
        assert rel_l2(np.asarray(joint.Ft_y[0]) + np.asarray(joint.Ft_y[1]), single.Ft_y) < 1e-2


def test_cc_block_rows_divide_every_capacity():
    for capacity in (64, 96, 256, 1000, 1024, 65536, 7):
        block = rlp._cc_block_rows(capacity)
        assert 1 <= block <= 128 and capacity % block == 0 and block & (block - 1) == 0


@requires_resident_gpu
def test_local_firstiter_cc_class3d_takes_the_winner_over_classes(monkeypatch, _resident_local_env):
    """RELION's --firstiter_cc iteration of a local Class3D with given references scores every class
    (do_generate_seeds is off, ml_optimiser.cpp:4392-4401) and keeps each particle's best sample over
    classes and poses (relax#72: this pass was refused).

    Two different references: each class's best score and pose are those of its own K=1 CC pass, and
    the particle's whole posterior goes to the class with the larger best score.
    """

    case = _case()
    rng = np.random.default_rng(72)
    other_projector = case["projector_half"] * jnp.asarray(
        rng.uniform(0.5, 1.5, np.shape(case["projector_half"])), dtype=jnp.float32
    )
    singles = [
        _run(case, monkeypatch=monkeypatch, firstiter_cc=True),
        _run(dict(case, projector_half=other_projector), monkeypatch=monkeypatch, firstiter_cc=True),
    ]
    two = dict(
        case,
        volume=jnp.stack([case["volume"], case["volume"]]),
        projector_half=jnp.stack([case["projector_half"], other_projector]),
    )
    joint = _run(two, monkeypatch=monkeypatch, firstiter_cc=True, n_classes=2).class_pass

    best = np.stack([np.asarray(s.relion_stats.best_log_score_per_image, dtype=np.float64) for s in singles])
    for k in range(2):
        np.testing.assert_allclose(np.asarray(joint.class_best_log_score_per_image[k]), best[k], rtol=1e-5)
        assert_matches(np.asarray(joint.per_class_hard_assignments[k]), np.asarray(singles[k].hard_assignment))
    winner = np.argmax(best, axis=0)
    assert set(winner.tolist()) == {0, 1}, "the case should give both classes a particle"
    np.testing.assert_array_equal(np.asarray(joint.stats.max_posterior_per_image), 1.0)
    np.testing.assert_allclose(
        np.asarray(joint.class_reconstruction_posterior_sums), np.bincount(winner, minlength=2), atol=1e-6
    )


@requires_resident_gpu
def test_local_firstiter_cc_identical_classes_give_the_first_class_everything(monkeypatch, _resident_local_env):
    """An exact tie between classes goes to the first class: with two copies of one reference the first
    class's BPref is the K=1 CC pass's and the second class has none.

    RELION takes the CC winner as the device arg-min of the coarse weights
    (acc_ml_optimiser_impl.h:2012-2026, ``getArgMinOnDevice`` = ``cub::DeviceReduce::ArgMin``,
    cuda_utils_cub.cuh:66-84, which prefers the smaller offset on a tie), and that array is class-major
    (``mapAllWeightsToMweights`` per class, acc_ml_optimiser_impl.h:1384-1392), so the lower class wins.
    """

    case = _case()
    single = _run(case, monkeypatch=monkeypatch, firstiter_cc=True)
    doubled = dict(
        case,
        volume=jnp.stack([case["volume"], case["volume"]]),
        projector_half=jnp.stack([case["projector_half"], case["projector_half"]]),
    )
    joint = _run(doubled, monkeypatch=monkeypatch, firstiter_cc=True, n_classes=2).class_pass

    def rel_l2(a, b):
        a, b = np.asarray(a, dtype=np.complex128), np.asarray(b, dtype=np.complex128)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    np.testing.assert_allclose(np.asarray(joint.class_reconstruction_posterior_sums), [N_IMAGES, 0.0], atol=1e-6)
    assert_matches(np.asarray(joint.per_class_hard_assignments[0]), np.asarray(single.hard_assignment))
    assert rel_l2(joint.Ft_y[0], single.Ft_y) < 1e-5 and rel_l2(joint.Ft_ctf[0], single.Ft_ctf) < 1e-5
    assert not np.asarray(joint.Ft_y[1]).any() and not np.asarray(joint.Ft_ctf[1]).any()


@requires_resident_gpu
def test_local_firstiter_cc_is_winner_take_all_and_ignores_the_priors(monkeypatch, _resident_local_env):
    """RELION's --firstiter_cc iteration when the search is local from iteration 1 (--sigma_ang).

    RELION scores the normalized CC and zeroes every weight but the best
    (ml_optimiser.cpp:9266-9292): Pmax is 1 for every image, and the
    orientational and translational priors play no part (the Gaussian pass of
    the same layout spreads its posterior; noise 200 keeps Pmax below 1 there).
    relax#51: the local engine ran a Gaussian E-step in that iteration.
    """

    import dataclasses

    case = _case()
    gaussian = _run(case, monkeypatch=monkeypatch)
    assert float(np.min(np.asarray(gaussian.relion_stats.max_posterior_per_image))) < 1.0

    cc = _run(case, monkeypatch=monkeypatch, firstiter_cc=True)
    np.testing.assert_array_equal(np.asarray(cc.relion_stats.max_posterior_per_image), 1.0)

    rng = np.random.default_rng(7)
    layout = case["layout"]
    reweighted = dict(
        case,
        layout=dataclasses.replace(
            layout,
            rotation_log_priors_flat=rng.normal(0.0, 3.0, np.shape(layout.rotation_log_priors_flat)).astype(
                np.asarray(layout.rotation_log_priors_flat).dtype
            ),
            translation_log_priors=rng.normal(0.0, 3.0, np.shape(layout.translation_log_priors)).astype(
                np.asarray(layout.translation_log_priors).dtype
            ),
        ),
    )
    cc_reweighted = _run(reweighted, monkeypatch=monkeypatch, firstiter_cc=True)
    assert_matches(np.asarray(cc.hard_assignment), np.asarray(cc_reweighted.hard_assignment))


@requires_resident_gpu
def test_local_firstiter_cc_probe_keeps_the_fine_pass_winner(monkeypatch, _resident_local_env):
    """The --firstiter_cc parent probe keeps one sample per image, the one the fine pass picks.

    Scored on the same layout, the score-only probe (its own operand preparation)
    and the fine pass (the translated tiles) choose the same (rotation, translation).
    """

    parent, translations = _parent_layout()
    case = dict(_case(), layout=parent, translations=translations)
    fine = _run(case, monkeypatch=monkeypatch, firstiter_cc=True)
    probe = _run(
        case,
        monkeypatch=monkeypatch,
        firstiter_cc=True,
        score_only=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        return_reconstruction_sample_indices=True,
        return_profile=True,
    )
    samples = probe.profile_summary["reconstruction_sample_indices_by_image"]
    assert [np.asarray(s).size for s in samples] == [1] * N_IMAGES
    assert_matches(np.asarray(probe.hard_assignment), np.asarray(fine.hard_assignment))


@requires_resident_gpu
@pytest.mark.parametrize("firstiter_cc", [False, True], ids=["gaussian", "firstiter_cc"])
def test_lone_overflow_image_matches_the_one_call_chunk(monkeypatch, _resident_local_env, firstiter_cc):
    """An image past the largest local row class runs alone in row blocks (relax#49 follow-up): its rows are
    projected and scored a block at a time (the scorer the global pass's lone chunk uses), the posterior is
    formed over all of them, and each M-step block projects its own rows. Each row's projection and score are
    those of the one-call chunk, so the discrete outputs agree exactly and the accumulators within the driver's
    repeat band (the x-half BPref and Wavg atomics are not bit-reproducible in either arm). The --firstiter_cc
    iteration scores its blocks with the normalized-CC block core the one-call CC chunk uses."""

    from relax.fine_pass import resident_scoring

    case = _case()
    whole = _run(case, monkeypatch=monkeypatch, production_shapes=True, firstiter_cc=firstiter_cc)

    blocked_calls = []
    name = "score_resident_chunk_normalized_cc_in_row_blocks" if firstiter_cc else "score_resident_chunk_in_row_blocks"
    real_blocked = getattr(rlp, name)

    def spy(*args, **kwargs):
        blocked_calls.append(int(kwargs["row_capacity"]))
        return real_blocked(*args, **kwargs)

    monkeypatch.setattr(rlp, name, spy)
    assert real_blocked is getattr(resident_scoring, name)
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "8")
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES", "1,2")
    lone = _run(case, monkeypatch=monkeypatch, production_shapes=True, firstiter_cc=firstiter_cc)

    assert blocked_calls, "the 8-row ladder made no lone chunk"
    assert_matches(np.asarray(whole.hard_assignment), np.asarray(lone.hard_assignment))
    assert_matches(np.asarray(whole.best_pose_rotations), np.asarray(lone.best_pose_rotations))
    assert_matches(np.asarray(whole.best_pose_translations), np.asarray(lone.best_pose_translations))

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    assert rel_l2(whole.Ft_y, lone.Ft_y) < 1e-6
    assert rel_l2(whole.Ft_ctf, lone.Ft_ctf) < 1e-6
    assert rel_l2(whole.noise_stats.wsum_sigma2_noise, lone.noise_stats.wsum_sigma2_noise) < 1e-6


@requires_resident_gpu
def test_class3d_lone_overflow_image_matches_the_one_call_chunk(monkeypatch, _resident_local_env):
    """A Class3D local image past the largest row class runs alone in row blocks too: each block's rows project
    with their own class's reference (a block spans the class boundary of the image's class-major rows), the
    joint and per-class posteriors are formed over all rows at once, and each class's M-step blocks project with
    that class's reference. The discrete outputs agree exactly and each class's accumulators within the
    driver's repeat band, as for the single-class lone chunk.

    The block size must not divide each class's rows per image, or no block spans a class boundary and the
    per-block class split is never exercised: the rows come in multiples of the 8 oversampled children, so
    8-row blocks align with the classes (an 8-row first version of this test found no straddling block) and
    12-row blocks do not.
    The test asserts a straddling block rather than assuming one."""

    case = _case()
    other_ft = _hermitian_volume(VOLUME_SHAPE, seed=29)
    other_half, _ = _relion_projector(
        np.asarray(ftu.get_idft3(np.asarray(other_ft).reshape(VOLUME_SHAPE)).real, dtype=np.float64),
        IMAGE_SHAPE[0],
    )
    two = dict(
        case,
        volume=jnp.stack([case["volume"], jnp.asarray(other_ft)]),
        projector_half=jnp.stack([case["projector_half"], other_half]),
    )
    whole = _run(two, monkeypatch=monkeypatch, n_classes=2).class_pass

    block_classes = []
    real_project = rlp._project_class_rows

    def spy(host_chunk, *args, **kwargs):
        block_classes.append(set(np.asarray(host_chunk["row_class"])[: int(kwargs["n_valid_rows"])].tolist()))
        return real_project(host_chunk, *args, **kwargs)

    monkeypatch.setattr(rlp, "_project_class_rows", spy)
    # 12-row blocks straddle the class boundary (see the docstring).
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "12")
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES", "1,2")
    lone = _run(two, monkeypatch=monkeypatch, n_classes=2).class_pass

    assert any(len(c) == 2 for c in block_classes), "no lone row block spanned the class boundary"

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    for k in range(2):
        assert_matches(np.asarray(whole.per_class_hard_assignments[k]), np.asarray(lone.per_class_hard_assignments[k]))
        assert_matches(
            np.asarray(whole.per_class_best_pose_rotations[k]), np.asarray(lone.per_class_best_pose_rotations[k])
        )
        assert np.max(
            np.abs(np.asarray(whole.class_log_evidence_per_image[k]) - np.asarray(lone.class_log_evidence_per_image[k]))
        ) < 1e-4
        assert rel_l2(whole.Ft_y[k], lone.Ft_y[k]) < 1e-6
        assert rel_l2(whole.Ft_ctf[k], lone.Ft_ctf[k]) < 1e-6
    assert rel_l2(whole.noise_stats.wsum_sigma2_noise, lone.noise_stats.wsum_sigma2_noise) < 1e-6
    assert np.max(np.abs(np.asarray(whole.stats.log_evidence_per_image) - np.asarray(lone.stats.log_evidence_per_image))) < 1e-4


def test_lone_block_rows_divide_a_chunk_rounded_to_a_smaller_class():
    """A one-row-class re-plan rounds an overflow image to a multiple of the class it chose: the lone blocks
    are the largest class that divides those rows (EMPIAR-10073 Class3D local, 6144 rows under 1024/4096)."""

    assert rp.lone_block_rows(6144, (1024, 4096)) == 1024
    assert rp.lone_block_rows(8192, (1024, 4096)) == 4096
    assert rp.lone_block_rows(24, (8, 16)) == 8
    with pytest.raises(ValueError, match="divides"):
        rp.lone_block_rows(100, (64, 256))


@requires_resident_gpu
def test_lone_overflow_chunk_from_a_smaller_class_replan_matches_the_one_call_chunk(monkeypatch, _resident_local_env):
    """The pass's chunks planned with only the smaller row class (as plan_pass_chunks' re-plan does) give an
    overflow chunk that the largest class does not divide; it runs in blocks of the class that does."""

    case = _case()
    whole = _run(case, monkeypatch=monkeypatch, production_shapes=True)
    real_plan = rlp.plan_local_capacity_chunks

    def smaller_class_plan(tables, *, row_capacity_ladder, image_capacity_ladder):
        return real_plan(tables, row_capacity_ladder=(8,), image_capacity_ladder=image_capacity_ladder)

    monkeypatch.setattr(rlp, "plan_local_capacity_chunks", smaller_class_plan)
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "8,16")
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES", "1,2")
    sizes = []
    real_start = rlp._start_resident_local_chunk

    def spy(chunk, **kwargs):
        if kwargs.get("lone_block_rows") is not None:
            sizes.append((int(chunk.row_capacity), int(kwargs["lone_block_rows"])))
        return real_start(chunk, **kwargs)

    monkeypatch.setattr(rlp, "_start_resident_local_chunk", spy)
    lone = _run(case, monkeypatch=monkeypatch, production_shapes=True)

    assert any(rows % 16 for rows, _ in sizes), f"no overflow chunk off the 16-row class: {sizes}"
    assert all(block == 8 for rows, block in sizes if rows % 16)
    assert_matches(np.asarray(whole.hard_assignment), np.asarray(lone.hard_assignment))

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        return float(np.linalg.norm(a - b) / np.linalg.norm(a))

    assert rel_l2(whole.Ft_y, lone.Ft_y) < 1e-6
    assert rel_l2(whole.Ft_ctf, lone.Ft_ctf) < 1e-6


def test_cc_row_block_scorer_matches_the_one_call_cc_chunk():
    """The --firstiter_cc lone scorer and the one-call CC chunk scorer share one block core: scoring the same
    projections a block at a time gives the one-call chunk's scores and candidates (padding blocks included)."""

    from relax.fine_pass import resident_scoring as rs

    rng = np.random.default_rng(11)
    n_images, n_trans, n_pix, rows, n_valid = 2, 5, 24, 32, 21
    reference = jnp.asarray(rng.normal(size=(rows, n_pix)) + 1j * rng.normal(size=(rows, n_pix)), jnp.complex64)
    row_image = jnp.asarray(np.minimum(np.arange(rows) // 11, n_images - 1), jnp.int32)
    mask_bits = jnp.asarray(rng.integers(0, 256, size=(rows, 1)), jnp.uint8)
    shifted = jnp.asarray(
        rng.normal(size=(n_images, n_trans, n_pix)) + 1j * rng.normal(size=(n_images, n_trans, n_pix)), jnp.complex64
    )
    weight = jnp.asarray(rng.uniform(0.5, 1.5, size=(n_images, n_pix)), jnp.float32)
    half_norm = jnp.asarray(rng.uniform(1.0, 2.0, size=n_images), jnp.float32)
    common = dict(half_weights=jnp.ones(n_pix, jnp.float32), full_to_compact=jnp.arange(n_pix, dtype=jnp.int32))
    one_call = rs.score_resident_projected_chunk_normalized_cc(
        reference, row_image, mask_bits, jnp.int32(n_valid), shifted, weight, half_norm,
        row_capacity=rows, n_fine_trans=n_trans, block_rows=8, **common,
    )
    blocked = rs.score_resident_chunk_normalized_cc_in_row_blocks(
        lambda start: reference[start : start + 8], row_image, mask_bits, n_valid, shifted, weight, half_norm,
        block_rows=8, row_capacity=rows, n_fine_trans=n_trans, **common,
    )
    a, b = np.asarray(one_call.scores), np.asarray(blocked.scores)
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
    np.testing.assert_allclose(b[np.isfinite(b)], a[np.isfinite(a)], rtol=1e-6, atol=0)
    np.testing.assert_allclose(np.asarray(blocked.min_diff2), np.asarray(one_call.min_diff2), rtol=1e-7)


def test_class3d_lone_rows_count_the_class_posterior_copy():
    """The Class3D local lone chunk counts one class's masked posterior copy and the class ids per row."""

    plain = rp.lone_chunk_row_bytes(84)
    assert rp.lone_chunk_row_bytes(84, class_rows=True) == plain + 84 * 4 + 8


def test_projector_call_takes_at_most_half_the_chunk_budget(monkeypatch):
    """A 16 GB card's box-448 final pass had a 3.63 GiB chunk budget and a fixed 4 GiB projector call (relax#49);
    the call now takes at most half the budget. Large budgets keep the 4 GiB cap."""

    monkeypatch.delenv(rlp._PROJECTION_CALL_MAX_BYTES_ENV, raising=False)
    gib = 1024**3
    row = 1_500_000  # bytes per projected row at box 448, full radius
    assert rlp._projection_block_rows(row, None) == 4 * gib // row
    assert rlp._projection_block_rows(row, 40 * gib) == 4 * gib // row
    assert rlp._projection_block_rows(row, int(3.63 * gib)) == int(0.5 * 3.63 * gib) // row
    assert rlp._projection_block_rows(10 * gib, int(3.63 * gib)) == 1
