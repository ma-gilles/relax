#!/usr/bin/env python
"""Fingerprint what pass 1 of the adaptive E-step (coarse significance) computes, to check that a refactor moved code and nothing else.

    python scripts/dev/pass1_fingerprint.py run OUT.json [--rev REV | --source DIR] [CASE ...]
    python scripts/dev/pass1_fingerprint.py diff A.json B.json          # exit 1 unless only log rows differ
    python scripts/dev/pass1_fingerprint.py check BASE_REV [HEAD_REV]   # run both (HEAD: the worktree) and diff
    python scripts/dev/pass1_fingerprint.py cases                       # the case list, one line each
    python scripts/dev/pass1_fingerprint.py selftest [MUTATION ...]     # perturb pass 1; every mutation must show

The counterpart of ``scripts/dev/fingerprint.py`` (the refinement controller) and ``scripts/dev/vdam_fingerprint.py``
for ``relax/scoring/significance.py``: the same commands, output format and comparison rules; the pure logic
(flattening, the diff and its accepted classes, mutations) is imported from the controller's harness, and the
command line is ``scripts/dev/fingerprint_cli.py``. The refinement
fingerprint replaces ``_compute_k_class_significance_batched`` by a recorder, so it cannot see inside pass 1.

``run`` calls ``_compute_k_class_significance_batched`` itself, on the CPU exact-operand harness
(``tests/helpers/exact_pass1_harness.py``: a tiny strict-preprocess dataset, unit CTFs, no high-resolution image
power, translation as repetition and a coded projector), for a few dozen cases. Each case records the six results
(rotation support, significant-sample counts, hard assignments, class assignments, the per-class significant
samples with their CSR, and ``full_stats``), the files a dump wrote, and one ordered trace of log records and of the
calls into the stand-in kernels (projection blocks, CTF rows, translation) with a digest of their operands. The
comparison is exact because both sides run the same arithmetic on the same CPU (one Eigen thread); it is a check
for move-only commits, not a merge gate for numerical changes.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fingerprint_cli  # noqa: E402  (the shared command line; it imports no harness)
from fingerprint import (  # noqa: E402
    TMP_TOKEN,
    accepted,
    case_errors,
    diff_fingerprints,
    differing_cases,
    digest_operands,
    flatten,
    log_row,
    mutated_tree,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NOT_COVERED = (
    "the tree rescore's kernel (relax.scoring.scoring._relion_coarse_normalized_cc_rescore) and its CUDA gates: the "
    "cases replace them by a deterministic stand-in, so the selection around the kernel is covered and its arithmetic is not",
    "RELION's CUDA image preprocessing, translation kernel and texture projector (stand-ins replace them: unit CTFs, "
    "a coded projector, translation as repetition, no high-resolution image power)",
    "real noise weighting and CTFs: the cases use unit noise and unit CTFs",
    "GPU operation order, reduction shapes, peak memory and array lifetimes",
    "the overlap of the host read-back of one batch with the device scoring of the next (defer_publish)",
)
# Keys of a written JSON file that hold wall times (none today; the dump files are arrays).
TIMING_WORDS = ("time", "wall", "seconds", "elapsed")


# ----------------------------------------------------------------------------- cases


def _cases() -> dict[str, tuple[str, dict]]:
    """``{name: (description, spec)}``; the spec is what the worker builds one call from.

    Spec keys: ``n_classes`` (2), ``n_images`` (7), ``n_rot`` (5), ``rotation_codes`` (the ``[0, 1]`` entry of each
    rotation; ``n_rot`` of them, evenly spaced by default), ``box`` (4), ``n_trans`` (3), ``rotation_block_size`` (2), ``projection`` ("coded" | "phased": the harness's projector scales one phase ramp by rotation, which leaves the
    normalized CC of every rotation equal; "phased" also changes the ramp's slope with the rotation),
    ``tree`` (None | "distinct" | "tied": the CUDA gates and the rescore
    kernel of the tree rescore are replaced, scoring the two candidates distinctly or alike),
    ``rotation_prior`` ("shared" | "per_class" | None), ``translation_prior``
    ("per_image" | "shared" | None), ``noise`` ("flat" | "per_group"), ``corrections`` (a tuple of "image",
    "scale", "shift"), ``env`` (variables set for the call; ``@DUMP`` is the case's dump directory) and
    ``kwargs`` (keyword arguments of the call that replace the defaults).
    """
    cases: dict[str, tuple[str, dict]] = {}

    def add(name, description, **spec):
        assert name not in cases, name
        cases[name] = (description, spec)

    add("k2_default", "K=2, per-image translation prior, rotation prior, a 1-image tail batch")
    add("k1_default", "K=1: RELION's log-weight order and winner", n_classes=1)
    add("k1_no_priors", "K=1 without a rotation or translation prior", n_classes=1, rotation_prior=None,
        translation_prior=None)
    add("k2_class_rotation_prior", "K=2 with one rotation prior per class", rotation_prior="per_class")
    add("k2_shared_translation_prior", "K=2 with one translation prior shared by every image", translation_prior="shared")
    add("k2_no_priors", "K=2 without a rotation or translation prior", rotation_prior=None, translation_prior=None)
    add("k2_no_collect", "K=2 scoring only (collect_significance=False)", kwargs={"collect_significance": False})
    add("k2_class_second", "K=2 with the class runner-up", kwargs={"return_class_second": True})
    add("k2_no_class_best", "K=2 without the class best", kwargs={"return_class_best": False})
    add("k2_f32_normalization", "K=2 returning RELION's float32 normalization", kwargs={"return_relion_f32_normalization": True})
    add("k1_f32_normalization", "K=1 returning RELION's float32 normalization", n_classes=1,
        kwargs={"return_relion_f32_normalization": True})
    add("k2_tie_ulps", "K=2 with a two-ULP score tie tolerance", kwargs={"relion_f32_coarse_tie_ulps": 2})
    add("k1_near_ties", "K=1 with six near-tied rotations at the cutoff and a two-ULP tie tolerance", n_classes=1,
        rotation_codes=[0.3 + 1e-5 * index for index in range(6)] + [0.1, 0.2], rotation_prior=None, translation_prior=None,
        kwargs={"relion_f32_coarse_tie_ulps": 2, "adaptive_fraction": 0.9, "max_significants": 0})
    add("k2_box8_window6", "K=2 on a box of 8 scored at current size 6", box=8, kwargs={"current_size": 6})
    add("k2_window_at_box", "K=2 with RELION's radial window kept when the current size reaches the box", box=8,
        kwargs={"window_at_box": True})
    add("k2_square_window", "K=2 with the square window", box=8, kwargs={"current_size": 6, "square_window": True})
    add("k2_stable_windows", "K=2 on the quantized physical window (current size 6 in a capacity of 8)", box=16,
        kwargs={"current_size": 6, "stable_fourier_window_shapes": True})
    add("k2_box16_window6", "K=2 on a box of 16 scored at current size 6 (the control of the stable window)", box=16,
        kwargs={"current_size": 6})
    add("k2_pad_tail", "K=2 padding the tail image batch (pad_final_image_batch)", kwargs={"pad_final_image_batch": True})
    add("k2_pad_env_off", "K=2 with RELAX_COARSE_PAD_FINAL_IMAGE_BATCH=0", env={"RELAX_COARSE_PAD_FINAL_IMAGE_BATCH": "0"})
    add("k2_pad_env_on", "K=2 with RELAX_COARSE_PAD_FINAL_IMAGE_BATCH=1", env={"RELAX_COARSE_PAD_FINAL_IMAGE_BATCH": "1"})
    add("k2_one_batch", "K=2 in one image batch", kwargs={"image_batch_size": 7})
    add("k2_nyquist_once", "K=2 counting the Nyquist column once", kwargs={"nyquist_column_counting": "once"})
    add("k2_optics_noise", "K=2 with a noise table of two optics groups", noise="per_group")
    add("k2_corrections", "K=2 with image and scale corrections", corrections=("image", "scale"))
    add("k2_pre_shifts", "K=2 with integral image pre-shifts", corrections=("shift",))
    add("k2_symmetry_label", "K=2 with a symmetry label (coarse HEALPix order inferred with it)",
        kwargs={"symmetry_label": "C2", "coarse_healpix_order": 0})
    add("k2_masked_images", "K=2 scoring masked images", kwargs={"score_with_masked_images": True})
    add("k2_rot16_cached", "K=2 over 16 rotations: the cached projection program", n_rot=16, rotation_block_size=4)
    add("k1_rot16_cached", "K=1 over 16 rotations: the cached projection program", n_classes=1, n_rot=16, rotation_block_size=4)
    add("k2_rot16_cache_off", "K=2 over 16 rotations with the projection cache off", n_rot=16, rotation_block_size=4,
        env={"RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "0"})
    add("refused_cache_requested", "an explicit projection-cache request over 5 rotations is refused", env={"RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "1"})
    add("k2_generic_support", "K=2 on the generic support route (RECOVAR_K1_RELION_F32_COARSE_SUPPORT=0)",
        env={"RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "0"})
    add("k1_generic_support", "K=1 on the generic support route", n_classes=1, env={"RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "0"})
    add("k2_generic_f32_normalization", "K=2 generic route returning the float32 normalization",
        env={"RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "0"}, kwargs={"return_relion_f32_normalization": True})
    add("k1_generic_f32_normalization", "K=1 generic route returning the float32 normalization", n_classes=1,
        env={"RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "0"}, kwargs={"return_relion_f32_normalization": True})
    add("k2_rotated_radius_off", "K=2 with the legacy source-pixel disk (RELAX_K1_COARSE_ROTATED_RADIUS=0)",
        env={"RELAX_K1_COARSE_ROTATED_RADIUS": "0"})
    add("k2_support_audit", "K=2 with the coarse support audit", env={"RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "1"})
    add("k2_support_audit_ids", "K=2 with the coarse support audit and its ids",
        env={"RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "1", "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS": "1"})
    add("k2_dump", "K=2 dumping the score blocks of two images (the host-mask route)",
        env={"RELAX_SIGNIFICANCE_DUMP_DIR": "@DUMP", "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1,4"})
    add("k1_dump", "K=1 dumping the score blocks of two images", n_classes=1,
        env={"RELAX_SIGNIFICANCE_DUMP_DIR": "@DUMP", "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1,4"})
    add("cc_k1", "first-iteration normalized CC, K=1", n_classes=1, kwargs={"score_mode": "normalized_cc"})
    add("cc_k2", "first-iteration normalized CC, K=2", kwargs={"score_mode": "normalized_cc"})
    add("cc_k1_gaussian_support", "normalized CC weighted on the Gaussian support", n_classes=1,
        kwargs={"score_mode": "normalized_cc", "firstiter_cc_support": "gaussian"})
    add("cc_k1_box8", "normalized CC on a box of 8 at current size 6", n_classes=1, box=8,
        kwargs={"score_mode": "normalized_cc", "current_size": 6})
    add("cc_k1_nyquist_once", "normalized CC counting the Nyquist column once", n_classes=1,
        kwargs={"score_mode": "normalized_cc", "nyquist_column_counting": "once", "firstiter_cc_support": "gaussian"})
    add("cc_k1_class_second", "normalized CC with the class runner-up", n_classes=1,
        kwargs={"score_mode": "normalized_cc", "return_class_second": True})
    add("cc_k1_dump", "normalized CC dumping the score blocks of two images", n_classes=1,
        kwargs={"score_mode": "normalized_cc"},
        env={"RELAX_SIGNIFICANCE_DUMP_DIR": "@DUMP", "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1,4"})
    add("cc_k1_phased", "normalized CC on poses whose scores differ", n_classes=1, projection="phased",
        kwargs={"score_mode": "normalized_cc"})
    add("cc_k2_phased", "normalized CC, K=2, on poses whose scores differ", projection="phased",
        kwargs={"score_mode": "normalized_cc"})
    tree = {"score_mode": "normalized_cc", "tree_rescore_max_margin": 0.5, "collect_significance": False}
    add("cc_k1_tree_rescore", "normalized CC with the tree top-two rescore, candidates scored distinctly", n_classes=1,
        tree="distinct", kwargs=tree)
    add("cc_k1_tree_rescore_tied", "normalized CC with the tree rescore, candidates tied exactly", n_classes=1,
        tree="tied", kwargs=tree)
    add("cc_k1_tree_rescore_none_ambiguous", "the tree rescore on poses with distinct CC, 6 images and a zero margin: none is ambiguous",
        n_classes=1, n_images=6, n_trans=1, projection="phased", tree="distinct", translation_prior=None,
        kwargs=dict(tree, tree_rescore_max_margin=0.0))
    add("cc_k1_tree_rescore_tail_ambiguous", "the tree rescore on poses with distinct CC and a zero margin: only the padded tail is",
        n_classes=1, n_trans=1, projection="phased", tree="distinct", translation_prior=None,
        kwargs=dict(tree, tree_rescore_max_margin=0.0))
    add("cc_k1_tree_rescore_pad", "the tree rescore on a padded tail batch", n_classes=1, tree="distinct",
        kwargs=dict(tree, pad_final_image_batch=True))
    add("cc_k1_tree_rescore_second", "the tree rescore returning the class runner-up", n_classes=1, tree="distinct",
        kwargs=dict(tree, return_class_second=True))
    add("cc_k1_tree_rescore_dump", "the tree rescore dumping its candidates for two images (6 images: no tail batch)",
        n_classes=1, n_images=6, tree="distinct",
        kwargs=dict(tree, debug_iteration=3),
        env={"RELAX_SIGNIFICANCE_DUMP_DIR": "@DUMP", "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1,4"})
    add("refused_tree_rescore_cpu", "the tree rescore is refused without a CUDA backend", n_classes=1,
        kwargs={"score_mode": "normalized_cc", "tree_rescore_max_margin": 0.5, "return_class_second": True})
    add("refused_tree_rescore_k2", "the tree rescore is refused for K=2",
        kwargs={"score_mode": "normalized_cc", "tree_rescore_max_margin": 0.5, "return_class_second": True})
    add("refused_float64_scoring", "float64 scoring is refused", kwargs={"use_float64_scoring": True})
    add("refused_no_texture", "a projector without texture interpolation is refused", kwargs={"relion_projector_texture_interp": False})
    add("refused_no_half_spectrum", "full-spectrum scoring is refused", kwargs={"half_spectrum_scoring": False})
    add("refused_no_projector", "a call without the RELION projector is refused", kwargs={"relion_projector_half": None})
    add("refused_second_without_best", "return_class_second without return_class_best is refused",
        kwargs={"return_class_second": True, "return_class_best": False})
    add("refused_score_mode", "an unknown score mode is refused", kwargs={"score_mode": "euclidean"})
    add("refused_rotation_prior_shape", "a rotation prior of the wrong length is refused", rotation_prior="bad_shape")
    add("refused_class_prior_shape", "class priors of the wrong length are refused", kwargs={"class_log_priors": [0.0, 0.0, 0.0]})
    add("refused_batch_size", "a non-positive image batch is refused", kwargs={"image_batch_size": 0})
    add("refused_noise_table_without_groups", "a per-group noise table without optics_group_ids is refused",
        noise="per_group_without_ids")
    return cases


CASES = _cases()
# The cases whose run is a refusal (fingerprint.case_errors).
REFUSED_CASES = frozenset(name for name in CASES if name.startswith("refused_"))

# A deliberately wrong pass 1 for the self-test: (name, old lines, new lines, what breaks, detected).
# A target must occur in exactly one place of the tree: when a statement is rewritten, update its entry.
MUTATIONS = (
    ("class_prior", "jnp.asarray(class_log_priors_np[class_index], dtype=jnp.float32),",
     "jnp.asarray(class_log_priors_np[class_index] * 2.0, dtype=jnp.float32),",
     "the class prior is doubled", True),
    ("rotation_prior_class", "rotation_log_prior_padded[class_index, r0 : r0 + rotation_block_size]",
     "rotation_log_prior_padded[0, r0 : r0 + rotation_block_size]",
     "every class takes class 0's rotation prior", True),
    ("shared_translation_prior", "scores = scores + translation_log_prior[None, None, :]",
     "scores = scores - translation_log_prior[None, None, :]",
     "a shared translation prior is subtracted", True),
    ("adaptive_fraction", "n_trans=int(n_trans),\nadaptive_fraction=float(adaptive_fraction),\nmax_significants=max_significants,",
     "n_trans=int(n_trans),\nadaptive_fraction=0.5 * float(adaptive_fraction),\nmax_significants=max_significants,",
     "the float32 support takes half the adaptive fraction", True),
    ("tie_ulps", "tie_score_ulps=int(relion_f32_coarse_tie_ulps),", "tie_score_ulps=int(relion_f32_coarse_tie_ulps) + 4000,",
     "the support's tie tolerance is 4000 ULPs wider", True),
    ("exact_weight_order", "relion_exact_coarse_weight_order = bool(relion_f32_coarse_support_enabled and n_classes == 1)",
     "relion_exact_coarse_weight_order = False",
     "K=1 loses RELION's log-weight order", True),
    ("pad_target", "target_size=int(image_batch_size),", "target_size=int(image_batch_size) + 2,",
     "a padded tail batch is two images too long", True),
    ("pad_env_ignored", "pad_final_image_batch = bool(pad_final_image_batch) or _coarse_pad_final_image_batch_enabled()",
     "pad_final_image_batch = bool(pad_final_image_batch)",
     "RELAX_COARSE_PAD_FINAL_IMAGE_BATCH is ignored", True),
    ("noise_group_rows", "batch_groups = image_groups_host[np.asarray(indices, dtype=np.int64)]",
     "batch_groups = image_groups_host[np.asarray(indices, dtype=np.int64)] * 0",
     "every image scores with optics group 0's spectrum", True),
    ("sig_rot_any", "sig_rot_any |= np.asarray(", "sig_rot_any ^= np.asarray(",
     "the rotation support accumulates by exclusive or", True),
    ("n_sig_count", "n_sig_all[batch.start_idx:batch.end_idx] = np.asarray(batch.n_sig, dtype=np.int32)[:batch.actual_batch_size]",
     "n_sig_all[batch.start_idx:batch.end_idx] = 1 + np.asarray(batch.n_sig, dtype=np.int32)[:batch.actual_batch_size]",
     "every significant-sample count is one too high", True),
    ("class_assignment", "class_assignment[batch.start_idx:batch.end_idx] = np.asarray(",
     "class_assignment[batch.start_idx:batch.end_idx] = 1 + np.asarray(",
     "every class assignment is shifted by one", True),
    ("log_evidence_dtype", "normalization_log_evidence[batch.start_idx:batch.end_idx].astype(",
     "normalization_log_evidence[batch.start_idx:batch.end_idx].__add__(1e-3).astype(",
     "the published log evidence is off by 1e-3", True),
    ("class_log_evidence", "np.asarray(class_log_z, dtype=np.float64)[output_slice]",
     "np.asarray(class_log_z, dtype=np.float64)[output_slice] + 1.0",
     "the class log evidence is off by one", True),
    ("class_second_assignment", "class_second_hard_assignment[class_index, batch.start_idx:batch.end_idx] = np.asarray(",
     "class_second_hard_assignment[class_index, batch.start_idx:batch.end_idx] = 1 + np.asarray(",
     "the class runner-up poses are shifted by one", True),
    ("cc_pixel_weight", "exact_cc_pixel_weight = exact_cc_operands.windowed_corr_img * (",
     "exact_cc_pixel_weight = exact_cc_operands.windowed_corr_img * 2.0 * (",
     "the normalized-CC pixel weight is doubled", True),
    ("window_at_box", "window_at_box=bool(window_at_box) and score_mode != \"normalized_cc\",",
     "window_at_box=False,",
     "RELION's radial window at the box is ignored", True),
    ("nyquist_once", "if nyquist_column_counting != \"relion\":\n# The score weights are zero on these pixels; zeroing them here also takes them",
     "if False:\n# The score weights are zero on these pixels; zeroing them here also takes them",
     "the Gaussian pass counts the Nyquist column twice whatever was asked", True),
    ("cc_gaussian_support", "cc_gaussian_support = score_mode == \"normalized_cc\" and firstiter_cc_support == \"gaussian\"",
     "cc_gaussian_support = False",
     "the Gaussian support of the normalized CC is ignored", True),
    ("stable_windows", "stable_fourier_window_shapes=bool(stable_fourier_window_shapes),",
     "stable_fourier_window_shapes=False,",
     "the quantized physical window is ignored", True),
    ("projection_cache_disabled", "coarse_gaussian_gemm_projection_cache_requested = exact_gaussian and (",
     "coarse_gaussian_gemm_projection_cache_requested = False and (",
     "the projection cache is never requested", True),
    ("generic_route_ignored", "relion_f32_coarse_support_enabled = exact_gaussian and _k1_relion_f32_coarse_support_enabled(default=True)",
     "relion_f32_coarse_support_enabled = exact_gaussian",
     "RECOVAR_K1_RELION_F32_COARSE_SUPPORT is ignored", True),
    ("rotated_radius_disk", "mask_current_image_disk=not coarse_rotated_radius,", "mask_current_image_disk=True,",
     "the rotated-radius clipping is ignored (always the source-pixel disk)", True),
    ("support_audit_ignored", "if _coarse_significance_support_audit_enabled():", "if False:",
     "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT is ignored", True),
    ("dump_ignored", "debug_dump_enabled = collect_significance and _significance_debug_dump_matches(",
     "debug_dump_enabled = False and _significance_debug_dump_matches(",
     "the score dump is never taken", True),
    ("f32_normalization_sum_weight", "relion_f32_sum_weight[batch.start_idx:batch.end_idx] = np.asarray(\n batch.sum_weight,",
     "relion_f32_sum_weight[batch.start_idx:batch.end_idx] = 2.0 * np.asarray(\n batch.sum_weight,",
     "the float32 sum weight is doubled", True),
    ("tree_margin_bound", "np.isfinite(score_margins) & (score_margins <= tree_rescore_max_margin)",
     "np.isfinite(score_margins) & (score_margins < tree_rescore_max_margin)",
     "an image whose margin equals the bound is no longer ambiguous", True),
    ("tree_translation_ids", "candidate_translation_ids = candidate_pose_ids % n_trans",
     "candidate_translation_ids = candidate_pose_ids // n_trans",
     "the tree rescore takes each candidate's translation from its rotation index", True),
    ("tree_winner_installed", "rescored_winner_pose = candidate_pose_ids[row_ids, rescored_winner_slot]",
     "rescored_winner_pose = candidate_pose_ids[row_ids, rescored_runner_slot]",
     "the rescored runner-up is installed as the winner", True),
    ("tree_winner_changes", "np.count_nonzero(rescored_winner_pose != best_pose_np)",
     "np.count_nonzero(rescored_winner_pose == best_pose_np)",
     "the winner-change count counts the unchanged", True),
    ("tree_exact_ties", "tree_rescore_exact_ties += exact_ties", "tree_rescore_exact_ties += 0",
     "the exact-tie count is dropped", True),
    ("tree_runner_score_installed", "class_second_best_scores[0] = class_second_best_scores[0].at[\n rows_jax\n ].set(rescored_runner_score[applied_rows])",
     "class_second_best_scores[0] = class_second_best_scores[0].at[\n rows_jax\n ].set(rescored_winner_score[applied_rows])",
     "the class runner-up score takes the winner's rescored score", True),
    ("prefetch_order", "iter_indexed_batches(experiment_dataset, image_indices, image_batch_size)",
     "iter_indexed_batches(experiment_dataset, image_indices[::-1], image_batch_size)",
     "the image batches arrive in reverse order", True),
)


# ----------------------------------------------------------------------------- the worker (imports the tree)


def _worker(source: str, out_path: str, tmp_root: str, names: list[str]) -> None:
    """Run the cases against ``source`` in this process; the caller has set the CPU-only environment."""
    import inspect
    import logging
    import threading
    import traceback

    sys.dont_write_bytecode = True
    # One Eigen thread, so float32 sums do not depend on the node (set before jax is imported).
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
    sys.path[:0] = [source, os.path.join(source, "tests")]
    os.chdir(tmp_root)

    import jax.numpy as jnp
    import numpy as np
    import pytest
    from helpers.exact_pass1_harness import ExactPass1Dataset, coded_class_projectors, install_exact_pass1_mocks

    import relax
    from relax.scoring import significance

    assert relax.__file__.startswith(source), relax.__file__

    tmp_pattern = re.compile(re.escape(tmp_root) + r"(/\w+)?")
    run_paths = [(re.escape(out_path), "<OUT>"), (re.escape(str(Path(tmp_root).parent)), "<WORK>"),
                 (re.escape(source), "<SRC>")]

    def scrub(text):
        text = tmp_pattern.sub(TMP_TOKEN, text)
        for pattern, token in run_paths:
            text = re.sub(pattern, token, text)
        return text

    trace: list = []

    class Capture(logging.Handler):
        def emit(self, log_record):
            template = str(log_record.msg)
            try:
                text = log_record.getMessage()
            except Exception:  # a malformed format string is itself behaviour
                text = "<unformattable> " + template
            if threading.current_thread() is threading.main_thread() and log_record.name.startswith(("relax", "recovar")):
                trace.append(log_row(log_record.name, log_record.levelname, template, text, scrub=scrub))

    def recorded(kind, function, describe):
        """``function`` that first appends a ``["call", kind, ...]`` row describing its operands."""

        def wrapper(*args, **kwargs):
            trace.append(["call", kind, *describe(*args, **kwargs), f"seed={digest_operands(args, kwargs):016x}"])
            return function(*args, **kwargs)

        return wrapper

    def install_recorders(patch):
        from relax.cuda import kernels as em_cuda_kernels
        from relax.helpers import projection as projection_helpers
        from relax.relion import relion_ctf

        patch.setattr(
            projection_helpers, "compute_relion_projector_projections_block",
            recorded("projection", projection_helpers.compute_relion_projector_projections_block,
                     lambda projector, rotations, *_, **kw: (f"rotations={len(rotations)}",
                                                             f"rows={'all' if kw.get('pixel_indices') is None else len(kw['pixel_indices'])}")),
        )
        patch.setattr(
            em_cuda_kernels, "relion_translate_score_f32",
            recorded("translate", em_cuda_kernels.relion_translate_score_f32,
                     lambda images, angles, *_: (f"images={images.shape[0]}", f"translations={angles.shape[0]}")),
        )
        patch.setattr(
            relion_ctf, "_relion_exact_ctf_half_from_source_star",
            recorded("ctf_rows", relion_ctf._relion_exact_ctf_half_from_source_star,
                     lambda _dataset, indices, *_, **kw: (f"images={len(indices)}",)),
        )

    def install_phased_projection(patch):
        """A projector whose phase ramp depends on the rotation, so the normalized CC differs between poses."""
        from relax.helpers import projection as projection_helpers

        def phased_projection(projector_half, rotations_block, image_shape, **kwargs):
            class_value = float(np.asarray(projector_half)[0, 0, 0].real)
            offsets = np.asarray(rotations_block)[:, 0, 1].astype(np.float64)
            pixel_indices = kwargs.get("pixel_indices")
            n_pixels = (
                int(image_shape[0]) * (int(image_shape[1]) // 2 + 1) if pixel_indices is None else len(pixel_indices)
            )
            ramp = np.exp(1j * (0.3 + 2.0 * offsets)[:, None] * np.arange(n_pixels)[None, :])
            projected = jnp.asarray((class_value + 1.0 + offsets)[:, None] * ramp, dtype=jnp.complex64)
            return projected, (jnp.abs(projected) ** 2 if kwargs.get("return_abs2", True) else None)

        patch.setattr(projection_helpers, "compute_relion_projector_projections_block", phased_projection)

    def install_tree_stand_in(patch, tied):
        """The tree rescore's CUDA gates, and its rescore kernel by a score taken from the candidates' operands.

        ``tests/unit/test_refine_relion_mode.py`` replaces the same things: the kernel runs only on a GPU.
        """
        import jax
        import recovar.cuda_backproject as cuda_backproject

        from relax.cuda import kernels as em_cuda_kernels
        from relax.scoring import scoring as scoring_module

        patch.setattr(jax, "default_backend", lambda: "gpu")
        patch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
        patch.setattr(em_cuda_kernels, "custom_cuda_requested", lambda: True)
        patch.setattr(cuda_backproject, "cuda_available", lambda: True)

        def rescore(shifted_candidates, score_weight_candidates, projection_candidates, half_weights, fftw_order, **kw):
            assert kw["projector_full"] is not None and kw["rotation_matrices"].shape[-2:] == (3, 3)
            # Scores of the size of a normalized CC (below 1.25), so the posterior built from them stays finite.
            base = jnp.tanh(jnp.real(jnp.sum(shifted_candidates, axis=-1)) / 100.0).astype(jnp.float32)
            if tied:
                return jnp.broadcast_to(base[:, :1], base.shape)
            return base * jnp.asarray([1.0, 1.25], dtype=jnp.float32)[None, :] - 0.5 * kw["rotation_matrices"][..., 0, 1]

        patch.setattr(
            scoring_module, "_relion_coarse_normalized_cc_rescore",
            recorded("tree_rescore", rescore, lambda shifted, *_, **kw: (f"candidates={shifted.shape[0]}",
                                                                          f"projector={tuple(kw['projector_full'].shape)}")),
        )

    def build(spec):
        n_classes = spec.get("n_classes", 2)
        n_images = spec.get("n_images", 7)
        codes = spec.get("rotation_codes")
        n_rot = spec.get("n_rot", 5) if codes is None else len(codes)
        box = spec.get("box", 4)
        dataset = ExactPass1Dataset(np.arange(n_images), box=box)
        rotations = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
        rotations[:, 0, 1] = np.linspace(0.0, 0.4, n_rot, dtype=np.float32) if codes is None else np.asarray(codes, dtype=np.float32)
        n_half = box * (box // 2 + 1)
        n_trans = spec.get("n_trans", 3)
        noise = jnp.ones(dataset.image_size, dtype=jnp.float32)
        optics_group_ids = None
        if spec.get("noise") in ("per_group", "per_group_without_ids"):
            noise = jnp.asarray(np.stack([np.linspace(1.0, 2.0, n_half), np.linspace(2.0, 1.0, n_half)]), dtype=jnp.float32)
            if spec["noise"] == "per_group":
                optics_group_ids = np.arange(n_images) % 2
        args = (
            dataset,
            noise,
            rotations,
            jnp.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 0.0]][:n_trans], dtype=jnp.float32),
        )
        rotation_prior = spec.get("rotation_prior", "shared")
        rotation_prior = {
            "shared": np.linspace(0.0, -0.4, n_rot, dtype=np.float32),
            "per_class": -0.1 * np.arange(1, n_classes + 1, dtype=np.float32)[:, None]
            * np.linspace(0.0, 1.0, n_rot, dtype=np.float32)[None, :],
            "bad_shape": np.zeros(n_rot + 1, dtype=np.float32),
            None: None,
        }[rotation_prior]
        translation_prior = spec.get("translation_prior", "per_image")
        translation_prior = {
            "per_image": np.linspace(0.0, -0.3, n_images * n_trans, dtype=np.float32).reshape(n_images, n_trans),
            "shared": np.array([0.0, -0.1, -0.25][:n_trans], dtype=np.float32),
            None: None,
        }[translation_prior]
        kwargs = dict(
            class_log_priors=np.log(np.arange(1, n_classes + 1) / sum(range(1, n_classes + 1))),
            rotation_log_prior=rotation_prior,
            translation_log_prior=translation_prior,
            adaptive_fraction=0.9,
            max_significants=6,
            image_batch_size=3,
            rotation_block_size=spec.get("rotation_block_size", 2),
            current_size=box,
            half_spectrum_scoring=True,
            return_class_best=True,
            relion_projector_half=coded_class_projectors(n_classes),
            relion_projector_r_max=1,
            relion_projector_texture_interp=True,
        )
        if optics_group_ids is not None:
            kwargs["optics_group_ids"] = optics_group_ids
        corrections = spec.get("corrections", ())
        if "image" in corrections:
            kwargs["image_corrections"] = np.linspace(0.9, 1.1, n_images, dtype=np.float32)
        if "scale" in corrections:
            kwargs["scale_corrections"] = np.linspace(1.2, 0.8, n_images, dtype=np.float32)
        if "shift" in corrections:
            kwargs["image_pre_shifts"] = np.stack([np.arange(n_images) % 3 - 1, np.arange(n_images) % 2]).T.astype(np.float32)
        kwargs.update(spec.get("kwargs", {}))
        return args, kwargs, n_classes

    def call_pass1(args, kwargs, n_classes):
        """The call in the signature of the tree under test, so a base and a head with different signatures compare.

        A base from before the removal of two parameters nothing read takes ``means`` (read only for its class
        count) as the second positional and ``disc_type`` as the sixth.
        """
        function = significance._compute_k_class_significance_batched
        parameters = inspect.signature(function).parameters
        args = list(args)
        if "means" in parameters:
            args.insert(1, jnp.zeros((n_classes, args[0].volume_size), dtype=jnp.complex64))
        if "disc_type" in parameters:
            args.append("linear_interp")
        return function(*args, **kwargs)

    def result_fields(result):
        """The six results as named leaves (``significant_samples`` also with its device CSR)."""
        sig_rot_any, n_sig, hard, class_assignment, samples, full_stats = result
        return {
            "sig_rot_any": sig_rot_any, "n_significant": n_sig, "hard_assignment": hard, "class_assignment": class_assignment,
            "significant_samples": None if samples is None else [list(by_image) for by_image in samples],
            "significant_samples_csr": None if samples is None else [getattr(by_image, "csr", None) for by_image in samples],
            "full_stats": full_stats,
        }

    def require_finite(fields):
        """A case that is not a refusal runs a finite path: a NaN would hide every mutation downstream."""
        def walk(value, path):
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(item, f"{path}/{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")
            elif hasattr(value, "dtype") and hasattr(value, "shape") and np.asarray(value).dtype.kind in "fc":
                if not np.all(np.isfinite(np.asarray(value))):
                    raise AssertionError(f"non-finite values in {path}")

        walk({key: value for key, value in fields.items() if key != "significant_samples_csr"}, "")

    def run_case(name, spec):
        dump_dir = Path(tmp_root) / f"dump_{name}"
        env = {key: value.replace("@DUMP", str(dump_dir)) for key, value in spec.get("env", {}).items()}
        trace.clear()
        patch = pytest.MonkeyPatch()
        handler = Capture()
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        status, fields = "ok", None
        try:
            install_exact_pass1_mocks(patch)
            if spec.get("projection") == "phased":
                install_phased_projection(patch)
            install_recorders(patch)
            if spec.get("tree"):
                install_tree_stand_in(patch, spec["tree"] == "tied")
            for key, value in env.items():
                patch.setenv(key, value)
            args, kwargs, n_classes = build(spec)
            result = call_pass1(args, kwargs, n_classes)
            fields = result_fields(result)
            if name not in REFUSED_CASES:
                require_finite(fields)
        except BaseException as error:  # a refusal is behaviour; record it
            status = f"{type(error).__name__}: {scrub(str(error))}"
            if name not in REFUSED_CASES or not isinstance(error, (ValueError, RuntimeError, TypeError)):
                status += "\n" + scrub(traceback.format_exc())
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)
            patch.undo()
        files = {}
        if dump_dir.exists():
            for path in sorted(dump_dir.iterdir()):
                with np.load(path, allow_pickle=False) as archive:
                    files.update(flatten({key: archive[key] for key in archive.files}, f"/{path.name}"))
        return {"status": {"": status}, "result": {} if fields is None else flatten(fields, scrub=scrub), "files": files,
                "checkpoints": {}, "trace": [list(map(str, row)) for row in trace]}

    results = {}
    for name in names:
        description, spec = CASES[name]
        results[name] = run_case(name, spec)
        print(f"{name}: {results[name]['status']['']}".splitlines()[0], flush=True)
    fingerprint = {"schema": 1, "source": scrub(source), "cases": results}
    Path(out_path).write_text(json.dumps(fingerprint, sort_keys=True))
    print(f"{len(results)} cases from {relax.__file__} -> {out_path}")


# ----------------------------------------------------------------------------- commands


HARNESS = fingerprint_cli.Harness(
    script=Path(__file__).resolve(),
    description=__doc__.split("\n\n")[0],
    cases=CASES,
    mutations=MUTATIONS,
    not_covered=NOT_COVERED,
    worker=_worker,
    diff_fingerprints=diff_fingerprints,
    accepted=accepted,
    case_errors=lambda fingerprint: case_errors(fingerprint, REFUSED_CASES),
    differing_cases=differing_cases,
    mutated_tree=mutated_tree,
    file_prefix="p",
    temp_prefix="relax-pass1-fingerprint-",
    case_width=36,
    cases_note="",
    # The child runs with the harness's own switches only: no relax or RECOVAR variable of the caller.
    dropped_env_prefixes=("RELAX_", "RECOVAR_"),
    jax_compilation_cache=False,
)


def main(argv: list[str] | None = None) -> int:
    return fingerprint_cli.main(HARNESS, argv)


if __name__ == "__main__":
    sys.exit(main())
