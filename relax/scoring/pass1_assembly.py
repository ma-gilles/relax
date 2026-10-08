"""After the last batch of pass 1: the per-class significant samples and the pass's own timing log."""

import logging

from relax.scoring.pass1_results import OutputPlan, Pass1Outputs
from relax.sparse_pass2.resident_significance import (
    DeviceCompactedSignificantSamples,
    build_coarse_significance_csr,
    host_support_rows,
)

logger = logging.getLogger(__name__)


def significant_samples_after_loop(outputs: Pass1Outputs, plan: OutputPlan):
    """The pass's per-class significant samples: the host rows the batches published, plus the device-compacted ones.

    Returns ``outputs.significant_sample_indices`` itself when no batch compacted its support on the device (a pass
    without ``collect_significance``, or every batch a dump batch). Otherwise returns a new ``[K]`` list whose class
    entry is a :class:`DeviceCompactedSignificantSamples` over the whole pass when every batch of the class compacted,
    and a copy of its per-image rows with the compacted batches' rows filled in when some batches kept the host mask
    (a score dump). ``outputs`` is not written.
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
