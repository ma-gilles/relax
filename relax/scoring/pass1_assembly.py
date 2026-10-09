"""After the last batch of pass 1: the per-class significant samples and the pass's own timing log."""

import logging
import time

from relax.diagnostics.coarse_score_diagnostics import build_coarse_significance_support_audit
from relax.helpers.env_flags import parse_env_strict_flag
from relax.scoring.pass1_results import OutputPlan, Pass1Outputs, Pass1Stats
from relax.sparse_pass2.resident_significance import (
    DeviceCompactedSignificantSamples,
    build_coarse_significance_csr,
    host_support_rows,
)

logger = logging.getLogger(__name__)

_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV = "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT"
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV = "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS"


def log_batch_timing(batch_starts, loop_start, n_images):
    """Log the wall time of the pass's batches: ``batch_starts`` are the ``time.time()`` at the start of each batch.

    A batch's wall time runs from its start to the next batch's start (the last one to now), so it includes the host
    work of publishing the batch before it. Logs nothing for a pass without batches.
    """

    if not batch_starts:
        return
    loop_end = time.time()
    loop_seconds = loop_end - loop_start
    batch_walls = [b - a for a, b in zip(batch_starts, batch_starts[1:])] + [loop_end - batch_starts[-1]]
    covered = sum(batch_walls)
    ordered = sorted(batch_walls)
    logger.info(
        "K-class coarse pass-1 batch timing: batches=%d images=%d loop=%.2fs "
        "covered=%.2fs uncovered=%.2fs mean=%.3fs median=%.3fs max=%.3fs",
        len(batch_walls),
        int(n_images),
        loop_seconds,
        covered,
        loop_seconds - covered,
        covered / len(batch_walls),
        ordered[len(ordered) // 2],
        ordered[-1],
    )


def significant_samples_after_loop(outputs: Pass1Outputs, plan: OutputPlan):
    """The pass's per-class significant samples: the host rows the batches published, plus the device-compacted ones.

    Returns ``outputs.significant_sample_indices`` itself when no batch compacted its support on the device (a pass
    without ``collect_significance``, or every batch a dump batch). Otherwise returns a new ``[K]`` list whose class
    entry is a :class:`DeviceCompactedSignificantSamples` over the whole pass when every batch of the class compacted,
    and a copy of its per-image rows with the compacted batches' rows filled in when some batches kept the host mask
    (a score dump). A class compacted over the whole pass hands its per-batch device-compaction lists over to its
    CSR: they are emptied once the CSR holds the ids, so the pass never keeps every class's ids twice (relax#34).
    """

    if not any(outputs.device_significance_counts):
        return outputs.significant_sample_indices
    samples = list(outputs.significant_sample_indices)
    for class_index in range(plan.n_classes):
        covered = int(sum(int(counts.size) for counts in outputs.device_significance_counts[class_index]))
        if covered != plan.n_images:
            # Some batches kept the host mask (a score dump): publish the
            # compacted batches as host rows too.
            rows_of_class = list(samples[class_index])
            for start, counts, polarity, ids in zip(
                outputs.device_significance_starts[class_index],
                outputs.device_significance_counts[class_index],
                outputs.device_significance_polarity[class_index],
                outputs.device_significance_ids[class_index],
                strict=True,
            ):
                batch_rows = host_support_rows(
                    build_coarse_significance_csr(
                        n_images=int(counts.size),
                        n_coarse_rot=plan.n_rot,
                        n_coarse_trans=plan.n_trans,
                        n_significant_per_batch=[counts],
                        store_excluded_per_batch=[polarity],
                        ids_per_batch=[ids],
                    )
                )
                for offset, row in enumerate(batch_rows):
                    rows_of_class[start + offset] = row
            samples[class_index] = rows_of_class
            continue
        coarse_significance_csr = build_coarse_significance_csr(
            n_images=plan.n_images,
            n_coarse_rot=plan.n_rot,
            n_coarse_trans=plan.n_trans,
            n_significant_per_batch=outputs.device_significance_counts[class_index],
            store_excluded_per_batch=outputs.device_significance_polarity[class_index],
            ids_per_batch=outputs.device_significance_ids[class_index],
        )
        for per_batch in (
            outputs.device_significance_counts,
            outputs.device_significance_polarity,
            outputs.device_significance_ids,
            outputs.device_significance_starts,
        ):
            per_batch[class_index].clear()
        samples[class_index] = DeviceCompactedSignificantSamples(
            host_support_rows(coarse_significance_csr),
            csr=coarse_significance_csr,
        )
        logger.info(
            "Coarse significance compacted on the device (class %d): %d images, %d ids "
            "(%.2f MB) instead of a %.2f GB support mask",
            class_index,
            plan.n_images,
            int(coarse_significance_csr.ids.size),
            coarse_significance_csr.ids.nbytes / 1e6,
            float(plan.n_images) * float(plan.n_rot) * float(plan.n_trans) / 1e9,
        )
    return samples


def _coarse_significance_support_audit_enabled(*, default: bool = False) -> bool:
    """Resolve exact, diagnostic-only coarse-support hashing."""

    return parse_env_strict_flag(_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV, default=default)


def _coarse_significance_support_audit_ids_enabled() -> bool:
    """Whether a support audit also retains its exact selected IDs."""
    return parse_env_strict_flag(_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV)


def build_stats(
    outputs: Pass1Outputs,
    significant_sample_indices,
    plan: OutputPlan,
    *,
    executed_backend: str,
) -> Pass1Stats:
    """The pass's :class:`Pass1Stats`: its per-image statistics.

    ``executed_backend`` names the scorer that ran. The class best and runner-up statistics are present when the plan
    asked for them. The support audit (an environment diagnostic) adds the hash of the supports, and refuses a pass that
    collected none.
    """

    support_audit = None
    if _coarse_significance_support_audit_enabled():
        if significant_sample_indices is None:
            raise RuntimeError(
                f"{_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV}=1 requires "
                "collect_significance=True",
            )
        support_audit = build_coarse_significance_support_audit(
            significant_sample_indices,
            samples_per_class=plan.n_rot * plan.n_trans,
            include_ids=_coarse_significance_support_audit_ids_enabled(),
        )
    return Pass1Stats(
        normalization_log_z=outputs.normalization_log_z,
        normalization_log_evidence=outputs.normalization_log_evidence,
        log_evidence_per_image=outputs.log_evidence,
        best_log_score_per_image=outputs.best_log_score,
        max_posterior_per_image=outputs.max_posterior,
        class_log_evidence_per_image=outputs.class_log_evidence,
        # RELION serializes the cutoff rank before inclusive threshold ties
        # expand the pass-2/M-step support represented by ``n_sig_all``.
        significant_cutoff_counts=outputs.cutoff_count_all,
        executed_coarse_backend=executed_backend,
        # RELION's oversampling-zero second pass deliberately reuses this
        # coarse, maximum-shifted float32 denominator numerically.  It is not
        # interchangeable with a log-evidence value because the fine pass
        # independently shifts its own maximum to 50 before division.
        relion_f32_sum_weight=outputs.relion_f32_sum_weight,
        relion_f32_max_posterior=outputs.relion_f32_max_posterior,
        class_best_log_score_per_image=outputs.class_best_log_score,
        class_best_offset_free_log_score_per_image=outputs.class_best_offset_free_log_score,
        class_hard_assignments=outputs.class_hard_assignment,
        class_second_best_log_score_per_image=outputs.class_second_best_log_score,
        class_second_best_offset_free_log_score_per_image=outputs.class_second_best_offset_free_log_score,
        class_second_hard_assignments=outputs.class_second_hard_assignment,
        coarse_significance_support_audit=support_audit,
    )
