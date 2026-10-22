"""Segmented CUDA sparse pass-2 posterior: wrapper contracts and oracle parity.

The segmented handlers take the scores of one chunk as a flat cell array in
which image ``i`` owns ``[segment_offsets[i], segment_offsets[i + 1])`` instead
of a padded rectangular row.  The significance boundary is RELION's cut with float64 cumulative sums, by radix
select (relion_coarse_cut_f32's definition); the GPU tests apply that definition to
the handler's own raw weights and compare with ``assert_matches``.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import cuda_targets
from helpers.float_compare import assert_matches
from recovar import cuda_backproject as cb

from relax.cuda import library
from relax.cuda import posterior as cuda_posterior

pytestmark = pytest.mark.unit

_ADAPTIVE_FRACTIONS = (0.999, 1.0)


def make_scores(shape, seed, *, all_inf_row=False, nan=False):
    """Random pass-2 scores with masked (-inf) cells, as the fine scorer emits."""

    rng = np.random.default_rng(seed)
    scores = (rng.normal(size=shape) * 30 - 200).astype(np.float32)
    scores[rng.random(shape) < 0.3] = -np.inf
    if nan:
        scores[tuple(0 for _ in shape)] = np.nan
    if all_inf_row and shape[0] > 1:
        scores[1] = -np.inf
    for row in range(shape[0]):
        if all_inf_row and row == 1:
            continue
        flat = scores[row].reshape(-1)
        if not np.isfinite(flat).any():
            flat[0] = -150.0
    return scores


def _gpu_library_has(symbol):
    library.ensure_ffi()
    return hasattr(library.get_lib(), symbol)


def _require_segmented_gpu():
    assert jax.default_backend() == "gpu"
    if not _gpu_library_has("SparsePass2SegmentedPosteriorF32"):
        pytest.skip("loaded CUDA library lacks SparsePass2SegmentedPosteriorF32")
    assert cb.custom_cuda_requested()


# Handler scratch, appended by ``return_scratch``: the exponentiated weights the
# significance boundary is defined on.
_SCRATCH_NAMES = ("raw_weights",)

_OUTPUT_NAMES = (
    "log_z",
    "best_log_score",
    "best_cell_index",
    "max_posterior",
    "probs",
    "normalized_weights",
    "reconstruction_probs",
    "mask",
    "n_significant",
    "sum_weight",
    "threshold",
)


def _segmented(
    scores_flat,
    offsets,
    n_valid,
    log_z,
    external,
    *,
    adaptive_fraction,
    keep_all,
    scratch=False,
):
    segments = len(offsets) - 1
    outputs = cuda_posterior.sparse_pass2_segmented_posterior_f32(
        jnp.asarray(scores_flat, dtype=jnp.float32),
        jnp.asarray(offsets, dtype=jnp.int32),
        jnp.asarray(n_valid, dtype=jnp.int32),
        jnp.asarray(log_z, dtype=jnp.float64),
        jnp.asarray(
            np.ones(segments, np.float32) if external is None else external,
            dtype=jnp.float32,
        ),
        adaptive_fraction=float(adaptive_fraction),
        keep_all=bool(keep_all),
        use_external_sum_weight=external is not None,
        return_scratch=scratch,
    )
    names = _OUTPUT_NAMES + (_SCRATCH_NAMES if scratch else ())
    return {name: np.asarray(value) for name, value in zip(names, outputs)}


# ---------------------------------------------------------------------------
# Contracts that hold on any backend.
# ---------------------------------------------------------------------------


def test_segment_state_bytes_match_header():
    """The Python scratch size tracks sizeof(SegmentState) in the CUDA header."""

    header = open(os.path.join(os.path.dirname(cuda_posterior.__file__), "sparse_pass2_posterior.cuh")).read()
    sizes = {"RowState": cuda_posterior._SPARSE_PASS2_ROW_STATE_BYTES, "int": 4, "int64_t": 8, "uint32_t": 4,
             "double": 8, "float": 4}

    def struct_bytes(name):
        body = header.split(f"struct {name}\n", 1)[1].split("};", 1)[0]
        total = 0
        for line in body.splitlines():
            if not line.strip() or line.strip()[0] in "{/":
                continue
            kind, field = line.split()[:2]
            count = int(field.split("[", 1)[1].split("]", 1)[0]) if "[" in field else 1
            align = min(sizes[kind], 8)
            total = (total + align - 1) // align * align + sizes[kind] * count
        return (total + 7) // 8 * 8

    sizes["BlockCut"] = struct_bytes("BlockCut")
    assert struct_bytes("SegmentState") == cuda_posterior._SPARSE_PASS2_SEGMENT_STATE_BYTES


def test_segmented_targets_are_optional_abi():
    """The segmented handlers register lazily, like the other optional CUDA targets."""

    for target in (
        library.TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64,
        library.TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32,
        library.TARGET_SPARSE_PASS2_SEGMENTED_RECONSTRUCTION_F32,
    ):
        assert target in library.OPTIONAL_FFI_REGISTRATIONS
        assert target not in {name for name, _symbol in cb._FFI_REGISTRATIONS}
    assert isinstance(cuda_targets.sparse_pass2_segmented_supported(), bool)


@pytest.mark.parametrize(
    "case",
    [
        "scores_dtype",
        "scores_empty",
        "offsets_dtype",
        "offsets_rank",
        "offsets_short",
        "n_valid_dtype",
        "n_valid_size",
        "log_z_dtype",
        "log_z_shape",
        "external_dtype",
        "static_bool",
        "return_scratch",
    ],
)
def test_wrapper_rejects_bad_operands(case):
    scores = jnp.zeros((12,), jnp.float32)
    offsets = jnp.asarray([0, 6, 12], jnp.int32)
    n_valid = jnp.asarray(2, jnp.int32)
    log_z = jnp.zeros((2,), jnp.float64)
    external = jnp.ones((2,), jnp.float32)
    kwargs = dict(adaptive_fraction=0.999, keep_all=False, use_external_sum_weight=False)
    if case == "scores_dtype":
        scores = scores.astype(jnp.float64)
    if case == "scores_empty":
        scores = jnp.zeros((0,), jnp.float32)
    if case == "offsets_dtype":
        offsets = offsets.astype(jnp.int64)
    if case == "offsets_rank":
        offsets = offsets.reshape(3, 1)
    if case == "offsets_short":
        offsets = jnp.asarray([0], jnp.int32)
    if case == "n_valid_dtype":
        n_valid = n_valid.astype(jnp.float32)
    if case == "n_valid_size":
        n_valid = jnp.asarray([2, 2], jnp.int32)
    if case == "log_z_dtype":
        log_z = log_z.astype(jnp.float32)
    if case == "log_z_shape":
        log_z = jnp.zeros((3,), jnp.float64)
    if case == "external_dtype":
        external = external.astype(jnp.float64)
    if case == "static_bool":
        kwargs["keep_all"] = 1
    if case == "return_scratch":
        kwargs["return_scratch"] = 1
    with pytest.raises((TypeError, ValueError)):
        cuda_posterior.sparse_pass2_segmented_posterior_f32(
            scores, offsets, n_valid, log_z, external, **kwargs
        )


def test_wrapper_requires_gpu_backend():
    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only contract")
    scores = jnp.zeros((12,), jnp.float32)
    offsets = jnp.asarray([0, 6, 12], jnp.int32)
    n_valid = jnp.asarray(2, jnp.int32)
    with pytest.raises(RuntimeError, match="GPU backend"):
        cuda_posterior.sparse_pass2_segmented_log_z_f64(scores, offsets, n_valid)
    with pytest.raises(RuntimeError, match="GPU backend"):
        cuda_posterior.sparse_pass2_segmented_posterior_f32(
            scores,
            offsets,
            n_valid,
            jnp.zeros((2,), jnp.float64),
            jnp.ones((2,), jnp.float32),
            adaptive_fraction=0.999,
            keep_all=False,
            use_external_sum_weight=False,
        )


# ---------------------------------------------------------------------------
# GPU: the significance cut on chunks shaped like the device-resident classes.
# ---------------------------------------------------------------------------

# (image capacity, row capacity, fine translations, row occupancy) of the
# resident chunk classes the matched hp3 and early states run.  nsys records
# image capacities 32 and 128 and cell counts 1376256 / 2752512 / 5505024 /
# 11010048, which are row capacities 8192-65536 times 168 fine translations.
_PRODUCTION_CHUNKS = (
    pytest.param(128, 8192, 168, 0.76, id="128img_8192rows"),
    pytest.param(32, 32768, 168, 0.95, id="32img_32768rows"),
)

def make_production_chunk(images, rows, translations, occupancy, seed):
    """A ragged chunk: image ``i`` owns ``rows_i * translations`` cells."""

    rng = np.random.default_rng(seed)
    cells = rows * translations
    share = rng.random(images) + 0.3
    per_image = np.maximum(
        1, np.floor(share / share.sum() * round(occupancy * rows))
    ).astype(np.int64)
    while per_image.sum() > rows:
        per_image[int(np.argmax(per_image))] -= 1
    offsets = np.zeros(images + 1, np.int64)
    offsets[1:] = np.cumsum(per_image * translations)
    live = int(offsets[-1])
    scores = np.full(cells, -np.inf, np.float32)
    body = (rng.normal(size=live) * 30.0 - 200.0).astype(np.float32)
    body[rng.random(live) < 0.25] = -np.inf
    scores[:live] = body
    for index in range(images):
        segment = scores[offsets[index] : offsets[index + 1]]
        if segment.size and not np.isfinite(segment).any():
            segment[0] = -150.0
    return scores, offsets.astype(np.int32)


def _chunk_log_z(scores, offsets, images):
    return np.asarray(
        cuda_posterior.sparse_pass2_segmented_log_z_f64(
            jnp.asarray(scores, jnp.float32),
            jnp.asarray(offsets, jnp.int32),
            jnp.asarray(images, jnp.int32),
        )
    )




def _definition(raw, scores, offsets, n_valid, adaptive_fraction, keep_all):
    """Per segment: the float32 sum of positive weights and the significant count, by definition."""

    sums, counts = [], []
    for index in range(len(offsets) - 1):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        if index >= n_valid or end <= begin:
            sums.append(np.float32(0.0))
            counts.append(0)
            continue
        weights, finite = raw[begin:end], np.isfinite(scores[begin:end])
        positive = np.sort(weights[weights > 0.0])
        if positive.size == 0:
            sums.append(np.float32(0.0))
            counts.append(0)
            continue
        cumulative = np.cumsum(positive.astype(np.float64))
        total = np.float32(cumulative[-1])
        if keep_all:
            keep = weights > 0.0
        else:
            target = np.float32((1.0 - np.float64(np.float32(adaptive_fraction))) * np.float64(total))
            cut = positive[min(int(np.searchsorted(cumulative, np.float64(target), side="right")), positive.size - 1)]
            keep = weights >= cut
        sums.append(total)
        counts.append(int(np.count_nonzero(keep & finite)))
    return np.asarray(sums, np.float32), np.asarray(counts, np.int32)


@pytest.mark.gpu
@pytest.mark.parametrize("keep_all", [False, True])
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
@pytest.mark.parametrize("images, rows, translations, occupancy", _PRODUCTION_CHUNKS)
def test_cut_follows_its_definition_on_production_chunks(images, rows, translations, occupancy, adaptive_fraction, keep_all):
    _require_segmented_gpu()
    scores, offsets = make_production_chunk(images, rows, translations, occupancy, 17)
    log_z = _chunk_log_z(scores, offsets, images)
    out = _segmented(scores, offsets, images, log_z, None, adaptive_fraction=adaptive_fraction, keep_all=keep_all, scratch=True)
    sums, counts = _definition(out["raw_weights"], scores, offsets, images, adaptive_fraction, keep_all)
    assert_matches(out["sum_weight"], sums)
    assert_matches(out["n_significant"], counts)
    assert_matches(out["mask"].sum(), counts.sum())


_RECONSTRUCTION_NAMES = (
    "log_z",
    "best_log_score",
    "best_cell_index",
    "max_posterior",
    "reconstruction_probs",
    "n_significant",
    "sum_weight",
    "threshold",
)


@pytest.mark.gpu
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("keep_all", [False, True])
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
@pytest.mark.parametrize(
    "images, rows, translations, occupancy",
    [
        pytest.param(32, 4096, 21, 0.9, id="32img_4096rows"),
        # One particle owning the whole chunk, the lone tilt particle of relax#118 at a small size.
        pytest.param(1, 2048, 1176, 1.0, id="1img_2048rows_1176trans"),
    ],
)
def test_reconstruction_only_handler_returns_the_full_handlers_values(
    images, rows, translations, occupancy, adaptive_fraction, keep_all, external
):
    """SparsePass2SegmentedReconstructionF32 runs the full handler's kernels and skips probs, normalized and
    mask: its eight outputs are the full handler's, on padded ragged chunks and on one large segment."""

    _require_segmented_gpu()
    if not _gpu_library_has("SparsePass2SegmentedReconstructionF32"):
        pytest.skip("loaded CUDA library lacks SparsePass2SegmentedReconstructionF32")
    scores, offsets = make_production_chunk(images, rows, translations, occupancy, 41)
    log_z = _chunk_log_z(scores, offsets, images)
    external_sum_weight = np.linspace(2.0, 5.0, images).astype(np.float32) if external else None
    full = _segmented(
        scores, offsets, images, log_z, external_sum_weight, adaptive_fraction=adaptive_fraction, keep_all=keep_all
    )
    lean = cuda_posterior.sparse_pass2_segmented_reconstruction_f32(
        jnp.asarray(scores, dtype=jnp.float32),
        jnp.asarray(offsets, dtype=jnp.int32),
        jnp.asarray(images, dtype=jnp.int32),
        jnp.asarray(log_z, dtype=jnp.float64),
        jnp.asarray(np.ones(images, np.float32) if external_sum_weight is None else external_sum_weight),
        adaptive_fraction=float(adaptive_fraction),
        keep_all=bool(keep_all),
        use_external_sum_weight=bool(external),
    )
    assert len(lean) == len(_RECONSTRUCTION_NAMES)
    for name, value in zip(_RECONSTRUCTION_NAMES, lean):
        value = np.asarray(value)
        assert value.dtype == full[name].dtype and value.shape == full[name].shape, name
        assert np.isfinite(value[np.isfinite(full[name])]).all(), name
        assert_matches(value, full[name], err_msg=name)
    assert np.count_nonzero(full["reconstruction_probs"]) > 0


@pytest.mark.gpu
def test_external_sum_weight_normalizes_and_keeps_the_fine_cut():
    _require_segmented_gpu()
    scores, offsets = make_production_chunk(32, 4096, 21, 0.9, 29)
    log_z = _chunk_log_z(scores, offsets, 32)
    external = np.linspace(2.0, 5.0, 32).astype(np.float32)
    own = _segmented(scores, offsets, 32, log_z, None, adaptive_fraction=0.999, keep_all=False, scratch=True)
    ext = _segmented(scores, offsets, 32, log_z, external, adaptive_fraction=0.999, keep_all=False, scratch=True)
    assert_matches(ext["sum_weight"], external)
    assert_matches(ext["n_significant"], own["n_significant"])
    raw = ext["raw_weights"]
    for index in (0, 31):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        assert_matches(ext["normalized_weights"][begin:end], raw[begin:end] / external[index])


@pytest.mark.gpu
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
def test_empty_and_invalid_segments_behave_like_all_inf_rows(adaptive_fraction):
    _require_segmented_gpu()
    translations = 5
    lengths = [6 * translations, 0, 9 * translations, 4 * translations]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    cells = int(offsets[-1]) + 3 * translations  # trailing padding cells
    scores = make_scores((cells,), 131).astype(np.float32)
    offsets = np.concatenate([offsets, [offsets[-1]]]).astype(np.int32)
    n_valid = 3
    for index in range(len(offsets) - 1):
        segment = scores[offsets[index] : offsets[index + 1]]
        if segment.size and not np.isfinite(segment).any():
            segment[0] = -150.0
    out = _segmented(scores, offsets, n_valid, _chunk_log_z(scores, offsets, n_valid), None,
                     adaptive_fraction=adaptive_fraction, keep_all=False)
    empty_and_invalid = [1, 3, 4]
    for name in ("n_significant", "sum_weight", "threshold", "max_posterior"):
        assert_matches(out[name][empty_and_invalid], np.zeros(3, out[name].dtype), err_msg=name)
    tail = int(offsets[3])
    for name in ("normalized_weights", "reconstruction_probs", "mask", "probs"):
        assert_matches(out[name][tail:], np.zeros(cells - tail, out[name].dtype), err_msg=f"{name} past n_valid_images")


@pytest.mark.gpu
def test_a_malformed_offset_table_empties_the_offending_segment():
    """segment_extent clamps a decreasing entry on the device: that segment is empty."""

    _require_segmented_gpu()
    translations = 4
    lengths = [5 * translations, 7 * translations, 6 * translations]
    good = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    scores = make_scores((int(good[-1]),), 157).astype(np.float32)
    for index in range(len(lengths)):
        segment = scores[good[index] : good[index + 1]]
        if not np.isfinite(segment).any():
            segment[0] = -150.0
    malformed = good.copy()
    malformed[2] = good[1] - translations
    clamped = good.copy()
    clamped[2] = good[1]
    expected = _segmented(scores, clamped, 3, _chunk_log_z(scores, clamped, 3), None, adaptive_fraction=0.999, keep_all=False)
    actual = _segmented(scores, malformed, 3, _chunk_log_z(scores, malformed, 3), None, adaptive_fraction=0.999, keep_all=False)
    for name in ("log_z", "best_log_score", "n_significant", "sum_weight", "threshold"):
        assert_matches(actual[name][:2], expected[name][:2], err_msg=f"{name} before the malformed entry")


@pytest.mark.gpu
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
@pytest.mark.parametrize(
    "images, rows, translations, occupancy",
    [
        # One particle holding the chunk (a lone tilt particle): its cut spans ~60 blocks.
        pytest.param(1, 32768, 120, 1.0, id="1img_lone"),
        # A few large segments and many small ones in one grid.
        pytest.param(32, 32768, 168, 0.95, id="32img_32768rows"),
    ],
)
def test_a_segment_cut_split_over_blocks_is_the_rows_cut(images, rows, translations, occupancy, adaptive_fraction):
    """The segmented cut adds each segment's histogram over the blocks its cells are split into; its sum and
    threshold are those of the coarse cut (relion_coarse_cut_f32, the same radix definition) applied to each
    segment's raw weights as a row, bitwise."""

    _require_segmented_gpu()
    scores, offsets = make_production_chunk(images, rows, translations, occupancy, 23)
    log_z = _chunk_log_z(scores, offsets, images)
    out = _segmented(scores, offsets, images, log_z, None, adaptive_fraction=adaptive_fraction, keep_all=False, scratch=True)
    lengths = np.diff(offsets)
    # The handler's split (segmented_cut_blocks_per_segment): ~64K cells a block, at most 8,192 blocks in the grid.
    blocks_per_segment = min(-(-scores.size // 65536), 8192 // images)
    assert blocks_per_segment > 1 and lengths.max() > 4 * blocks_per_segment
    padded = np.zeros((images, int(lengths.max())), np.float32)
    for index in range(images):
        padded[index, : lengths[index]] = out["raw_weights"][offsets[index] : offsets[index + 1]]
    sums, cuts, _ = cuda_posterior.relion_coarse_cut_f32(
        jnp.asarray(padded), adaptive_fraction=float(adaptive_fraction), max_significants=0
    )
    np.testing.assert_array_equal(out["sum_weight"], np.asarray(sums))
    np.testing.assert_array_equal(out["threshold"], np.asarray(cuts))
