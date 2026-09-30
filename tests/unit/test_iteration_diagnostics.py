from types import SimpleNamespace

import numpy as np

from relax.diagnostics import iteration


def test_begin_numbered_half_bpref_diagnostics_sets_context_and_logs(monkeypatch):
    context = {}
    messages = []
    monkeypatch.setattr(
        iteration.bpref_diagnostics,
        "set_bpref_contribution_dump_context",
        lambda **values: context.update(values),
    )
    monkeypatch.setattr(
        iteration,
        "_bpref_device_signature_active_for_numbered_half",
        lambda **values: values == {"iteration": 4, "half": 2},
    )
    log = SimpleNamespace(info=lambda *values: messages.append(values))

    active = iteration._begin_numbered_half_bpref_diagnostics(
        iteration=4,
        half=2,
        log=log,
    )

    assert active is True
    assert context == {"iteration": 4, "half": 2}
    assert messages == [
        (
            "BPREF_DEVICE_SIGNATURE_ACTIVATION iteration=%d half=%d "
            "final_all_data=false active=%s",
            4,
            2,
            "true",
        )
    ]


def test_save_dense_replay_manifest_uses_scoring_objects_without_copying_schema(tmp_path):
    half = SimpleNamespace(
        k=1,
        image_corrections_k=np.array([1.5], dtype=np.float32),
        scale_corrections_k=np.array([0.75], dtype=np.float32),
        means_k=np.array([2 + 3j], dtype=np.complex64),
        mean_variance=np.array([4.0], dtype=np.float32),
        noise_variance_k=np.array([5.0], dtype=np.float32),
    )
    sampling = SimpleNamespace(
        effective_rotations=np.eye(3, dtype=np.float32)[None],
        coarse_scoring_rotations=None,
        current_translations=np.array([[1.0, -2.0]], dtype=np.float32),
        cs_for_engine=None,
        random_perturbation=0.125,
    )
    priors = SimpleNamespace(
        rotation_log_prior_k=np.array([0.25], dtype=np.float32),
        translation_log_prior=None,
        translation_search_base=np.array([[3.0, 4.0]], dtype=np.float64),
    )
    previous_translations = np.array([[5.0, 6.0]], dtype=np.float64)
    result = SimpleNamespace(
        em_stats=SimpleNamespace(
            max_posterior_per_image=np.array([0.25, 0.75], dtype=np.float32),
        ),
    )

    iteration.save_dense_replay_manifest(
        tmp_path,
        3,
        half,
        sampling,
        priors,
        previous_translations,
        result,
        perturbation_factor=0.5,
    )

    with np.load(tmp_path / "manifest_iter3_half1.npz") as manifest:
        np.testing.assert_array_equal(
            manifest["effective_rotations"], sampling.effective_rotations,
        )
        np.testing.assert_array_equal(
            manifest["image_corrections"], half.image_corrections_k,
        )
        np.testing.assert_array_equal(
            manifest["image_pre_shifts"],
            priors.translation_search_base.astype(np.float32),
        )
        np.testing.assert_array_equal(
            manifest["absolute_previous_translations"],
            previous_translations.astype(np.float32),
        )
        assert manifest["coarse_scoring_rotations"].dtype == np.float64
        assert manifest["coarse_scoring_rotations"].size == 0
        assert manifest["translation_log_prior"].dtype == np.float64
        assert manifest["translation_log_prior"].size == 0
        assert manifest["current_size"] == np.int32(-1)
        assert manifest["half_index"] == np.int32(1)
        assert manifest["iteration"] == np.int32(3)
        assert manifest["perturbation_instance"] == np.float64(0.125)
        assert manifest["perturbation_factor"] == np.float64(0.5)
        assert manifest["ave_Pmax"] == np.float64(0.5)
