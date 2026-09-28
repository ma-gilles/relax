"""Host contracts for fingerprint-bound fixed-capacity local components."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.local import fixed_capacity_local, local_big_jit, local_bucket_stages, local_em_engine
from relax.local.fixed_capacity_local import _materialize_fixed_capacity_local_call_view
from relax.local.local_layout import (
    LocalBucketSpec,
)

pytestmark = pytest.mark.unit


class _IndexedDataset:
    def __init__(self):
        self.images = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
        self.ctf_params = np.arange(3 * 3, dtype=np.float32).reshape(3, 3) + 100

    def iter_batches(self, batch_size, *, indices, by_image):
        assert by_image is False
        indices = np.asarray(indices, dtype=np.int64)
        yield self.images[indices], None, None, self.ctf_params[indices], None, None, indices


def _bucket(image_indices, row_counts, *, radix, image_capacity, include_optional=True):
    image_indices = np.asarray(image_indices, dtype=np.int32)
    row_counts = np.asarray(row_counts, dtype=np.int32)
    n_images = int(image_indices.size)
    rotation_mask = np.arange(radix, dtype=np.int32)[None, :] < row_counts[:, None]
    rotation_ids = np.full((n_images, radix), -1, dtype=np.int32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, radix, 3, 3)).copy()
    mstep_rotations = rotations.copy()
    rotation_log_prior = np.full((n_images, radix), -1e30, dtype=np.float32)
    posterior_ids = np.full((n_images, radix), -1, dtype=np.int32) if include_optional else None
    sample_mask = np.zeros((n_images, radix, 2), dtype=bool) if include_optional else None
    translation_log_prior = np.empty((n_images, 2), dtype=np.float32)
    for row, (image_id, count) in enumerate(zip(image_indices.tolist(), row_counts.tolist(), strict=True)):
        rotation_ids[row, :count] = image_id * 10 + np.arange(count, dtype=np.int32)
        rotations[row, :count] = np.arange(count * 9, dtype=np.float32).reshape(count, 3, 3) + image_id
        mstep_rotations[row, :count] = rotations[row, :count] + 50
        rotation_log_prior[row, :count] = -np.arange(count, dtype=np.float32)
        if posterior_ids is not None:
            posterior_ids[row, :count] = image_id * 10 + np.arange(count, dtype=np.int32)
        if sample_mask is not None:
            sample_mask[row, :count] = True
        translation_log_prior[row] = np.asarray([-image_id, -image_id - 0.5], dtype=np.float32)
    return LocalBucketSpec(
        image_indices=image_indices,
        bucket_image_count=image_capacity,
        bucket_rotation_count=radix,
        actual_rotation_counts=row_counts,
        local_rotation_ids=rotation_ids,
        local_rotations=rotations,
        local_mstep_rotations=mstep_rotations,
        local_rotation_log_prior=rotation_log_prior,
        local_rotation_mask=rotation_mask,
        translation_log_prior=translation_log_prior,
        local_rotation_posterior_ids=posterior_ids,
        local_sample_mask=sample_mask,
    )


def _select_call0(bundle, mature_bucket, default_image_pre_shifts, **overrides):
    options = {
        "n_classes": 1,
        "class_log_prior": 0.0,
        "image_pre_shifts": default_image_pre_shifts,
        "image_corrections": None,
        "scale_corrections": None,
        "score_only": True,
        "disable_adjoint_y": True,
        "disable_adjoint_ctf": True,
        "accumulate_noise": False,
        "mstep_requested": False,
        "unsupported_diagnostics": (),
        "enabled": True,
    }
    options.update(overrides)
    return fixed_capacity_local._select_fixed_capacity_score_only_view(
        bundle,
        mature_bucket,
        call_index=0,
        **options,
    )


def test_fixed_capacity_call0_materialization_and_selection_are_default_off_and_inert():
    assert _materialize_fixed_capacity_local_call_view(None) is None
    assert (
        _select_call0(
            None,
            None,
            None,
            n_classes=None,
            score_only=False,
            enabled=False,
        )
        is None
    )


def _assert_bucket_arrays_equal(actual, expected):
    for field_name in (
        "image_indices",
        "actual_rotation_counts",
        "local_rotation_ids",
        "local_rotations",
        "local_mstep_rotations",
        "local_rotation_log_prior",
        "local_rotation_mask",
        "translation_log_prior",
        "local_rotation_posterior_ids",
        "local_sample_mask",
    ):
        assert_matches(getattr(actual, field_name), getattr(expected, field_name))
    assert actual.bucket_image_count == expected.bucket_image_count
    assert actual.bucket_rotation_count == expected.bucket_rotation_count


def test_local_big_jit_shared_invocation_forwards_one_call_without_numeric_changes(monkeypatch):
    positional = object()
    keyword = object()
    expected = object()
    calls = []

    def fake_big_jit(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(local_bucket_stages, "run_local_bucket_big_jit", fake_big_jit)

    result = local_em_engine._invoke_local_bucket_big_jit(positional, marker=keyword)

    assert result is expected
    assert calls == [((positional,), {"marker": keyword})]


def test_local_big_jit_donates_the_two_loop_carried_accumulators_by_signature_index():
    parameter_names = tuple(inspect.signature(local_big_jit.run_local_bucket_big_jit).parameters)
    source = inspect.getsource(local_big_jit.run_local_bucket_big_jit)

    assert parameter_names[7:9] == ("mstep", "noise")
    assert local_big_jit._LocalMstepAccumulators._fields == ("Ft_y", "Ft_ctf")
    assert "donate_argnums=(7,)" in source
    assert "donate_argnums=(4, 5)" not in source


def test_local_em_caller_allocates_and_forwards_fresh_donated_accumulators_per_run():
    source = inspect.getsource(local_em_engine.run_local_em_exact)
    allocation_y = source.index("Ft_y = jnp.zeros(")
    allocation_ctf = source.index("Ft_ctf = jnp.zeros(")
    bucket_loop = source.index("for bucket_index in range(len(bucket_specs)):")
    argument_tuple = source.index("big_jit_arguments = (")
    invocation = source.index("_invoke_local_bucket_big_jit(")

    assert allocation_y < bucket_loop < argument_tuple < invocation
    assert allocation_ctf < bucket_loop < argument_tuple < invocation
    argument_source = source[
        argument_tuple : source.index("big_jit_static_options = dict(", argument_tuple)
    ]
    positional_lines = [
        line.strip().rstrip(",") for line in argument_source.splitlines()[1:11]
    ]
    assert positional_lines[7] == "_LocalMstepAccumulators(Ft_y, Ft_ctf)"


def test_fixed_capacity_selector_is_private_default_off_and_uses_shared_mature_call():
    signature = inspect.signature(local_em_engine.run_local_em_exact)
    assert signature.parameters["_fixed_capacity_enabled"].default is False
    assert signature.parameters["_fixed_capacity_bundle"].default is None
    assert signature.parameters["_fixed_capacity_class_count"].default is None
    assert signature.parameters["_fixed_capacity_whole_boundary_enabled"].default is False
    assert signature.parameters["_flat_local_rows_enabled"].default is False
    assert signature.parameters["_stable_flat_row_capacity_enabled"].default is False
    assert signature.parameters["_packed_local_projection_enabled"].default is False
    assert signature.parameters["fused_pair_fine_score"].default is False
    source = inspect.getsource(local_em_engine.run_local_em_exact)
    assert source.count("_invoke_local_bucket_big_jit(") == 1
    assert "big_jit_result = run_local_bucket_big_jit(" not in source
    assert "fixed_capacity_enabled and not use_big_jit_buckets" in source
    assert "call_index=bucket_index" in source
    assert source.index("_fetch_and_validate_fixed_capacity_call_operands(") < source.index(
        "_invoke_local_bucket_big_jit(",
    )


def test_stable_flat_row_capacity_requires_flat_rows():
    with pytest.raises(ValueError, match="stable flat-row capacity requires flat local rows"):
        local_em_engine.run_local_em_exact(
            None,
            None,
            None,
            None,
            "nearest",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=2,
            _stable_flat_row_capacity_enabled=True,
        )


def test_fused_pair_fine_score_requires_exact_flat_rows():
    with pytest.raises(
        ValueError,
        match="fused-pair fine scoring requires exact RELION fine diff2 and flat local rows",
    ):
        local_em_engine.run_local_em_exact(
            None,
            None,
            None,
            None,
            "nearest",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=2,
            fused_pair_fine_score=True,
        )


def test_whole_local_call_preparation_removes_only_invariant_carry_positions():
    positional, _ = local_big_jit._local_bucket_big_jit_signature_parts()
    arguments = tuple(object() for _ in positional)

    prepared = local_big_jit._prepare_fixed_capacity_local_call(*arguments)

    assert prepared.leading_arguments == arguments[:7]
    assert prepared.trailing_arguments == arguments[9:]
    with pytest.raises(ValueError, match="every mature positional argument"):
        local_big_jit._prepare_fixed_capacity_local_call(*arguments[:-1])


def test_whole_local_static_options_are_complete_hashable_and_fail_closed():
    _, keyword_only = local_big_jit._local_bucket_big_jit_signature_parts()
    required = {
        parameter.name: False
        for parameter in keyword_only
        if parameter.default is inspect.Parameter.empty
    }

    canonical = local_big_jit._canonicalize_fixed_capacity_static_options(required)

    assert tuple(name for name, _ in canonical) == tuple(
        parameter.name for parameter in keyword_only
    )
    with pytest.raises(ValueError, match="unknown fixed-capacity"):
        local_big_jit._canonicalize_fixed_capacity_static_options(
            {**required, "not_a_mature_option": False}
        )
    missing = dict(required)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing fixed-capacity"):
        local_big_jit._canonicalize_fixed_capacity_static_options(missing)
    with pytest.raises(ValueError, match="recursively hashable"):
        local_big_jit._canonicalize_fixed_capacity_static_options(
            {**required, "mask_mode": []}
        )


def test_uniform_local_scan_threads_carry_with_one_stacked_call_axis():
    stacked_program = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(jnp.asarray([1, 2], dtype=jnp.int32),),
        trailing_arguments=(jnp.asarray([101, 202], dtype=jnp.int32),),
    )
    initial_carry = tuple(jnp.asarray(value, dtype=jnp.int32) for value in range(10))

    def fake_numeric_call(delta, mstep, noise, tag, *, scale):
        carry = (*mstep, *noise)
        increment = delta * scale
        next_first_eight = tuple(value + increment for value in carry[:8])
        next_last_two = tuple(value + increment for value in carry[8:])
        return local_big_jit._LocalBigJitResult(local_big_jit._LocalBigJitCore(
            *next_first_eight,
            delta * 100,
            *next_last_two,
            tag,
            delta * 1000,
            *([None] * 9),
        ))

    final_carry, stacked_outputs = (
        local_big_jit._run_fixed_capacity_uniform_local_scan_program(
            stacked_program,
            initial_carry,
            (("scale", 3),),
            numeric_call=fake_numeric_call,
        )
    )

    assert tuple(int(value) for value in final_carry) == tuple(
        value + 9 for value in range(10)
    )
    assert tuple(np.asarray(value).tolist() for value in jax.tree_util.tree_leaves(stacked_outputs)) == (
        [100, 200],
        [101, 202],
        [1000, 2000],
    )


def test_uniform_local_scan_fails_closed_on_structure_shape_or_dtype_changes():
    call = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((2,), dtype=np.float32),),
        trailing_arguments=(),
    )
    changed_structure = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((2,), dtype=np.float32), object()),
        trailing_arguments=(),
    )
    changed_shape = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((3,), dtype=np.float32),),
        trailing_arguments=(),
    )
    changed_dtype = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((2,), dtype=np.float64),),
        trailing_arguments=(),
    )

    validated = local_big_jit._validate_uniform_fixed_capacity_call_program(
        (call, call)
    )
    assert validated[0] is call and validated[1] is call
    with pytest.raises(ValueError, match="pytree structure"):
        local_big_jit._validate_uniform_fixed_capacity_call_program(
            (call, changed_structure)
        )
    for changed in (changed_shape, changed_dtype):
        with pytest.raises(ValueError, match="leaf shape or dtype"):
            local_big_jit._validate_uniform_fixed_capacity_call_program(
                (call, changed)
            )


def test_uniform_local_scan_reuses_mature_body_inside_lax_scan():
    source = inspect.getsource(local_big_jit._run_fixed_capacity_uniform_local_scan_jit)
    program_source = inspect.getsource(
        local_big_jit._run_fixed_capacity_uniform_local_scan_program
    )

    assert "run_local_bucket_big_jit.__wrapped__" in source
    assert "donate_argnums=(1, 2)" in source
    assert "jax.lax.scan" in program_source
    assert "optimization_barrier" in program_source


def test_segmented_local_scan_partitions_only_at_chronological_abi_transitions():
    call_a0 = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((2,), dtype=np.float32),),
        trailing_arguments=(),
    )
    call_a1 = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.ones((2,), dtype=np.float32),),
        trailing_arguments=(),
    )
    call_b = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((3,), dtype=np.float32),),
        trailing_arguments=(),
    )
    call_a2 = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.full((2,), 2, dtype=np.float32),),
        trailing_arguments=(),
    )

    segments = local_big_jit._partition_uniform_fixed_capacity_calls(
        (call_a0, call_a1, call_b, call_a2)
    )

    assert tuple(len(segment) for segment in segments) == (2, 1, 1)
    flattened = tuple(call for segment in segments for call in segment)
    assert all(
        actual is expected
        for actual, expected in zip(
            flattened,
            (call_a0, call_a1, call_b, call_a2),
            strict=True,
        )
    )
