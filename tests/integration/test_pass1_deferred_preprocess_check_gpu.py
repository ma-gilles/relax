"""Pass 1 defers the image-preprocess finite check to the end of its batch loop; an invalid image still stops the E-step there."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _run_adaptive_k2(monkeypatch, dataset_edit=None):
    """Run the public adaptive K=2 E-step on the native-preprocessing fixture, three images per pass-1 batch.

    Returns ``(run, pass2_calls)``; ``run()`` executes the E-step and ``pass2_calls`` counts entries into pass 2,
    the only place the E-step accumulates statistics.
    """
    from helpers.em_arrays import _hermitian_volume
    from helpers.sparse_pass2_mock import VOLUME_SHAPE
    from relax.classification import k_class
    from relax.relion import relion_ctf

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "unit"))
    from integration.test_dense_gemm_coarse_engine_gpu import _install_native_preprocessing, _variable_ctf_rows
    from unit.test_resident_pass2_driver import _driver_fixture_args
    from unit.test_resident_relion_reference import _ppref, _tie_free_noise

    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    args = _driver_fixture_args(seed=20260929)
    dataset = args.pop("experiment_dataset")
    _install_native_preprocessing(dataset)
    if dataset_edit is not None:
        dataset_edit(dataset)
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_half_from_source_star", _variable_ctf_rows)
    for name in (
        "volume", "noise_variance", "significant_sample_indices", "normalization_other_score_log_z",
        "normalization_score_mode", "relion_x_half_mstep", "return_score_log_z", "preserve_bpref_particle_order",
    ):
        args.pop(name)
    fine_rotations = args.pop("fine_rotations_override")
    rot_parent = args.pop("fine_rotation_parent_override")
    fine_translations = args.pop("fine_translations_override")
    trans_parent = args.pop("fine_translation_parent_override")
    translations = args.pop("translations")
    noise = jnp.asarray(_tie_free_noise(6, 200.0).reshape(-1), jnp.float32)
    volumes = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=17 + i) for i in range(2)])
    slabs = [_ppref(volumes[i]) for i in range(2)]
    args.update(
        score_with_masked_images=True,
        relion_projector_half=np.stack([slab for slab, _ in slabs]),
        relion_projector_r_max=slabs[0][1],
        mstep_relion_x_half=True,
    )
    disc_type = args.pop("disc_type")
    keywords = dict(
        class_log_priors=np.log(np.array([0.5, 0.5])),
        accumulate_noise=args.pop("accumulate_noise"),
        coarse_healpix_order=args.pop("nside_level"),
        oversampling_order=args.pop("oversampling_order"),
        fine_current_size=args["current_size"],
        significance_image_batch_size=3,
        **args,
    )
    pass2_calls = []
    pass2 = k_class._run_sparse_k_class_adaptive_pass2

    def counted_pass2(*positional, **named):
        pass2_calls.append(1)
        return pass2(*positional, **named)

    monkeypatch.setattr(k_class, "_run_sparse_k_class_adaptive_pass2", counted_pass2)

    def run():
        return k_class.run_dense_k_class_em_adaptive(
            dataset, volumes, None, noise,
            fine_rotations[::2], translations, fine_rotations, fine_translations,
            rot_parent, trans_parent, disc_type, **keywords,
        )

    return run, pass2_calls, dataset


@pytest.mark.gpu
def test_pass1_non_finite_image_fails_before_pass2(monkeypatch):
    """A NaN image raises from the real pass-1 batch loop, named by batch, before pass 2 accumulates anything."""
    assert jax.default_backend() == "gpu"
    from relax.cuda import kernels as em_cuda_kernels

    run, pass2_calls, dataset = _run_adaptive_k2(monkeypatch)
    result = run()
    assert pass2_calls == [1]
    assert np.all(np.isfinite(np.asarray(result.stats.log_evidence_per_image)))

    def poison(dataset):
        # The kernel's check reads the soft-mask background, so the bad pixel sits in a corner.
        dataset._images[7, 0, 0] = np.nan  # image 7 is in the third batch of three

    run, pass2_calls, dataset = _run_adaptive_k2(monkeypatch, poison)
    with pytest.raises(RuntimeError) as failure:
        run()
    message = str(failure.value)
    assert "deferred check, pass 1" in message and "1 image(s)" in message
    assert "batch 2 (images 6-8 of the pass, dataset images 6-8): 1 image(s)" in message
    assert message.count("batch ") == 1
    assert pass2_calls == []
    # Nothing is left for a later pass to report.
    assert em_cuda_kernels._RELION_PREPROCESS_DEFERRED_SCOPES == []
    assert em_cuda_kernels.pending_relion_preprocess_checks() == 0
