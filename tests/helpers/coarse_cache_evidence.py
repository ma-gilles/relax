"""The H100 alias evidence of the coarse GEMM's projection cache, as the tests that read it describe a cache plan.

Deterministic job 13332001 observed donated-insert aliasing for exactly a ``(1, 36864, 5100)`` cache with 4608-row
chunks on one H100. Admission never relies on it: it always reserves a non-aliased copy.
"""

from relax.scoring.coarse_gaussian_gemm import _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS

ALIAS_EVIDENCE_JOB = 13_332_001


def projection_cache_stats(plan, *, enabled: bool):
    """Describe conservative admission and narrowly scoped alias evidence."""

    h100_alias_evidence_applies = bool(
        plan.cache_shape == (1, 36_864, 5_100)
        and plan.chunk_rows
        == _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
    )
    return {
        "enabled": bool(enabled),
        "scope": "one significance call",
        "cache_shape": tuple(int(value) for value in plan.cache_shape),
        "cache_dtype": plan.cache_dtype.name,
        "stores_projection_abs2": False,
        "chunk_rows": int(plan.chunk_rows),
        "chunk_count": int(plan.chunk_count_per_table),
        "retained_bytes": int(plan.retained_bytes),
        "conservative_predicted_peak_bytes": int(plan.predicted_peak_bytes),
        "budget_bytes": int(plan.budget_bytes),
        "admission_destination_alias_proven": bool(
            plan.destination_alias_proven
        ),
        # Informational only: deterministic job 13332001 observed donated
        # insert aliasing for exactly (1, 36864, 5100) with 4608-row chunks
        # on one H100. Admission always reserves a non-aliased copy.
        "h100_alias_evidence_applies_to_plan": h100_alias_evidence_applies,
        "h100_alias_evidence_cache_shape": (1, 36_864, 5_100),
        "h100_alias_evidence_chunk_rows": int(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
        ),
        "h100_observed_donated_insert_alias": (
            True if h100_alias_evidence_applies else None
        ),
        "h100_alias_evidence_job_id": int(
            ALIAS_EVIDENCE_JOB
        ),
        "h100_observed_alias_peak_bytes": (
            int(plan.predicted_peak_bytes - plan.destination_copy_bytes)
            if h100_alias_evidence_applies
            else None
        ),
        "h100_alias_evidence_used_for_admission": False,
    }
