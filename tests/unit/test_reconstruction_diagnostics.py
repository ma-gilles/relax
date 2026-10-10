"""Saved reconstruction captures retain their independent file contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.reconstruction_settings import reconstruction_settings

from relax.diagnostics import reconstruction as dumps
from relax.refinement.iteration_planning import ClassImageSize
from relax.refinement.priors import ClassPriorEstimate
from relax.refinement.refinement_options import ReconstructionPrograms

pytestmark = pytest.mark.unit


@pytest.fixture
def capture_inputs(tmp_path):
    spectrum = np.array([1, 2, 4], dtype=np.float64)
    complex_rows = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
    stats = {key: spectrum for key in ("avg_weight_shells", "shell_sum", "shell_count")}
    return dict(
        output_dir=tmp_path,
        iteration=2,
        class_idx=1,
        current_size=8,
        grid_size=16,
        PADDING_FACTOR=2,
        voxel_size=1.6375,
        computed_cs=10,
        prev_cs=6,
        raw_cs=9,
        res_shell=3,
        per_class_res_shell=[2, 3],
        relion_incr_size=2,
        relion_has_high_fsc_at_limit=True,
        state=SimpleNamespace(ave_Pmax=0.75, current_resolution=0.2, previous_resolution=0.1),
        data_vs_prior_prev_raw=spectrum,
        data_vs_prior_prev=spectrum,
        previous_means=[complex_rows, -complex_rows],
        Ft_y_combined=complex_rows,
        Ft_ctf_combined=complex_rows,
        Ft_ctf_0=complex_rows,
        Ft_ctf_1=-complex_rows,
        tau2_fudge=4.0,
        kclass_tau2_frame_scale=2.0,
        kclass_tau2_source="iref",
        mstep_accumulator_shape=(4, 4, 3),
        mstep_full_half_axis=0,
        tau2_shells_recovar_frame_k=spectrum,
        tau2_shells_relion_frame_k=spectrum * 2,
        shell_stats_k=stats,
        reconstruct_floor_stats_k=stats,
        data_vs_prior_k=spectrum,
        pixel_res=3.0,
        dvp_iter=spectrum,
        fsc=spectrum / 4,
        tau2_update_details={"prior_shells": spectrum, "fsc_shells": None},
        tau2_update_details_per_half=[None, {"ssnr_shells": spectrum}],
        perturb_replay_relion_dir=None,
        perturb_replay_relion_prefix="run",
        sealed_sampling_state=None,
        _replay_meta={"source": "test"},
        logger=SimpleNamespace(info=lambda *args: None),
    )


def invoke(writer, values):
    # Fixture contains inputs for all four capture boundaries.
    import inspect

    writer(**{name: values[name] for name in inspect.signature(writer).parameters})


def test_current_size_schema_and_casts(capture_inputs):
    values = capture_inputs
    plan = ClassImageSize(
        size=values["computed_cs"], resolution_shell=values["res_shell"], raw_size=values["raw_cs"],
        resolution_shells_per_class=np.asarray(values["per_class_res_shell"]),
        raw_data_vs_prior=values["data_vs_prior_prev_raw"], data_vs_prior=values["data_vs_prior_prev"],
    )
    dumps.write_class_image_size(
        plan, output_dir=values["output_dir"], previous_size=values["prev_cs"],
        box_size=values["grid_size"], iteration=values["iteration"],
        has_high_fsc_at_limit=values["relion_has_high_fsc_at_limit"],
        incr_size=values["relion_incr_size"], state=values["state"],
    )
    path = capture_inputs["output_dir"] / "recovar_kclass_current_size_it003.npz"
    with np.load(path) as saved:
        assert set(saved.files) == {
            "iteration",
            "previous_current_size",
            "grid_size",
            "resolution_shell",
            "per_class_resolution_shells",
            "ave_Pmax",
            "state_current_resolution",
            "state_previous_resolution",
            "relion_incr_size",
            "relion_has_high_fsc_at_limit",
            "data_vs_prior_prev_raw",
            "data_vs_prior_prev",
            "raw_current_size",
            "quantized_current_size",
        }
        assert saved["iteration"].item() == 3
        assert saved["quantized_current_size"].item() == 10
        assert saved["iteration"].dtype == np.int32
        assert saved["ave_Pmax"].dtype == np.float64
        assert saved["data_vs_prior_prev"].dtype == np.float32
        assert_matches(saved["per_class_resolution_shells"], [2, 3])


@pytest.mark.parametrize("token,preserve", [("", False), (" OFF ", False), ("1", True), ("unrecognized", True)])
@pytest.mark.parametrize("missing_half", [False, True])
def test_mstep_class_selection_and_dtype(capture_inputs, monkeypatch, token, preserve, missing_half):
    monkeypatch.setenv("RELAX_KCLASS_DUMP_PRESERVE_DTYPE", token)
    if missing_half:
        capture_inputs["Ft_ctf_1"] = None
    values = capture_inputs
    prior = ClassPriorEstimate(
        variance=None, shells=values["tau2_shells_recovar_frame_k"],
        relion_shells=values["tau2_shells_relion_frame_k"],
        data_vs_prior=values["data_vs_prior_k"],
        details={"sigma2_shells": 1.0 / (2**3 * values["shell_stats_k"]["avg_weight_shells"]), **values["shell_stats_k"]},
    )
    settings = reconstruction_settings(
        box_size=values["grid_size"], voxel_size=values["voxel_size"],
        volume_shape=(16, 16, 16), padding_factor=values["PADDING_FACTOR"],
        projection_padding_factor=2, minres_map=5, width_mask_edge=5, fmask_edge=2,
        tau2_fudge=values["tau2_fudge"], particle_diameter_angstrom=None,
        first_iteration_lowpass_angstrom=None, programs=ReconstructionPrograms.from_environ(),
    )

    def floor_statistics(denominator, shape, **kwargs):
        assert_matches(denominator, values["Ft_ctf_combined"][values["class_idx"]])
        assert shape == settings.volume_shape
        assert kwargs == {
            "padding_factor": 2, "r_max": 4, "shell_rounding": "floor",
            "full_half_axis": 0, "accumulator_volume_shape": (4, 4, 3),
        }
        return values["reconstruct_floor_stats_k"]

    monkeypatch.setattr(dumps.regularization_relion, "compute_relion_weight_shell_stats", floor_statistics)
    dumps.write_class_mstep(
        prior, numerators=values["Ft_y_combined"], denominators=values["Ft_ctf_combined"],
        half_denominators=(values["Ft_ctf_0"], values["Ft_ctf_1"]),
        references=values["previous_means"], settings=settings, output_dir=values["output_dir"],
        class_index=values["class_idx"], current_size=values["current_size"], iteration=values["iteration"],
        source=values["kclass_tau2_source"], accumulator_shape=values["mstep_accumulator_shape"],
        full_half_axis=values["mstep_full_half_axis"],
        frame_scale=values["kclass_tau2_frame_scale"],
    )
    with np.load(capture_inputs["output_dir"] / "recovar_kclass_mstep_it003_c02.npz") as saved:
        expected_dtype = np.complex128 if preserve else np.complex64
        assert saved["Ft_y_combined"].dtype == expected_dtype
        assert_matches(saved["Ft_y_combined"], capture_inputs["Ft_y_combined"][1])
        assert saved["previous_mean_half1"].dtype == np.complex64
        assert_matches(saved["previous_mean_half1"], capture_inputs["previous_means"][1][1])
        if missing_half:
            assert saved["Ft_ctf_1"].size == 0
            assert saved["Ft_ctf_1"].dtype == np.complex64
        assert saved["dump_preserve_dtype"].item() == preserve
        assert saved["tau2_shells"].dtype == np.float64
        assert saved["tau2_fudge"].dtype == np.float64
        assert_matches(saved["tau2_fudge"], values["tau2_fudge"])
