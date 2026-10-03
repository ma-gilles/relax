"""The ``gridding_kernel`` option: RELION's radial correction window or the separable one.

``radial`` must reach the callees main reaches with main's arguments; ``separable`` must
replace the window at every K=1 site (scoring projector, expected accuracy, M-step, final
maps), and every route that keeps the radial window must refuse it.
"""

import ast
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.reconstruction import relion_functions as rf

from relax.helpers import expected_accuracy
from relax.refinement import half_scoring, iteration_loop, mean_helpers, projector_preparation
from relax.refinement.refinement_options import KClassOptions, RefinementOptions, RelionParityOptions
from relax.relion import relion_projector_setup as setup

pytestmark = pytest.mark.unit

N = 8
PF = 2
R_MAX = 3


def _reference(seed=0):
    jax.config.update("jax_enable_x64", True)
    return np.random.default_rng(seed).normal(size=(N, N, N))


def _record(monkeypatch, owner, name):
    calls = []
    original = getattr(owner, name)

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, recorder)
    return calls


def _window(gridding_function, padding_factor=PF):
    return np.asarray(gridding_function(jnp.ones((N, N, N)), N, padding_factor, order=1)[1])


# --- scoring projector -------------------------------------------------------------------------


def test_radial_projector_setup_runs_the_unchanged_radial_program(monkeypatch):
    reference = _reference()
    radial = _record(monkeypatch, setup, "_gridding_corrected")
    separable = _record(monkeypatch, setup, "_gridding_corrected_separable")

    slab, power = setup.setup_relion_projector_on_host(reference, R_MAX, ori_size=N, padding_factor=PF)

    assert len(radial) == 1 and not separable
    assert set(radial[0][1]) == {"ori_size", "padding_factor"} and len(radial[0][0]) == 1
    # The pre-change body: the radial program, then the window build.
    checked = setup._checked_reference(reference, N, PF, jnp.float64)
    corrected = setup.gridding_correct_volume_real(checked, N, PF)
    expected_slab, expected_power = setup._build_projector_window(
        corrected, R_MAX, N, PF, R_MAX, chunk_bytes=None, to_host=True, output_radius=R_MAX
    )
    assert slab.dtype == expected_slab.dtype
    assert_matches(slab, expected_slab)
    assert_matches(power, expected_power)


def test_separable_projector_setup_applies_recovar_square_window(monkeypatch):
    reference = _reference()
    radial = _record(monkeypatch, setup, "_gridding_corrected")

    slab, power = setup.setup_relion_projector_on_host(
        reference, R_MAX, ori_size=N, padding_factor=PF, gridding_kernel="separable"
    )

    assert not radial
    corrected, _ = rf.griddingCorrect_square(jnp.asarray(reference), N, PF, order=1)
    expected_slab, expected_power = setup._build_projector_window(
        corrected, R_MAX, N, PF, R_MAX, chunk_bytes=None, to_host=True, output_radius=R_MAX
    )
    assert_matches(slab, expected_slab)
    assert_matches(power, expected_power)
    radial_slab, _ = setup.setup_relion_projector_on_host(reference, R_MAX, ori_size=N, padding_factor=PF)
    assert np.max(np.abs(slab - radial_slab)) > 1e-6 * np.max(np.abs(radial_slab))


def test_projector_setup_rejects_an_unknown_kernel():
    with pytest.raises(ValueError, match="gridding_kernel"):
        setup.setup_relion_projector_on_host(_reference(), R_MAX, ori_size=N, padding_factor=PF, gridding_kernel="x")


@pytest.mark.parametrize("kernel,expected", [("radial", {}), ("separable", {"gridding_kernel": "separable"})])
def test_reference_half_maps_forward_the_kernel_only_when_separable(monkeypatch, kernel, expected):
    calls = _record(monkeypatch, setup, "setup_relion_projector_on_host")
    kwargs = {} if kernel == "radial" else {"gridding_kernel": kernel}
    setup.reference_to_relion_projector_half_maps_and_power(
        _reference()[None], current_size=2 * R_MAX, padding_factor=PF, **kwargs
    )
    assert len(calls) == 1
    assert set(calls[0][1]) == {"ori_size", "padding_factor", "compute_dtype", *expected}
    assert {key: calls[0][1][key] for key in expected} == expected


def _scoring_slab(reference_real, **kwargs):
    from recovar.core import fourier_transform_utils as ftu

    reference_ft = np.asarray(ftu.get_dft3(jnp.asarray(reference_real))).reshape(-1)
    return projector_preparation._relion_projector_half_maps_for_scoring(
        reference_ft, volume_shape=(N, N, N), current_size=2 * R_MAX, padding_factor=PF, n_classes=1, **kwargs
    )


def test_scoring_projector_separable_matches_the_direct_window_and_radial_is_default(monkeypatch):
    monkeypatch.delenv("RELAX_RELION_PROJECTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("RELAX_RELION_PROJECTOR_DUMP_DIR", raising=False)
    reference = _reference()
    calls = _record(monkeypatch, projector_preparation, "_relion_projector_half_maps_for_scoring")
    builder = _record(monkeypatch, setup, "reference_to_relion_projector_half_maps_and_power")

    radial_slab, r_max, _ = _scoring_slab(reference)
    separable_slab, _, _ = _scoring_slab(reference, gridding_kernel="separable")

    assert len(calls) == 2
    assert "gridding_kernel" not in builder[0][1] and builder[1][1]["gridding_kernel"] == "separable"
    from recovar.utils.helpers import recovar_volume_to_relion

    relion_layout = np.asarray(recovar_volume_to_relion(reference), dtype=np.float64)
    corrected, _ = rf.griddingCorrect_square(jnp.asarray(relion_layout), N, PF, order=1)
    expected, _ = setup._build_projector_window(
        corrected, r_max, N, PF, r_max, chunk_bytes=None, to_host=True, output_radius=r_max
    )
    # The reference goes through a Fourier round trip before the setup.
    assert_matches(separable_slab[0], expected, rtol=1e-12)
    assert np.max(np.abs(separable_slab - radial_slab)) > 1e-6 * np.max(np.abs(radial_slab))


def test_projector_cache_keeps_radial_and_separable_entries_apart(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("RELAX_RELION_PROJECTOR_DUMP_DIR", raising=False)
    reference = _reference()

    radial_slab, _, _ = _scoring_slab(reference)
    radial_entries = sorted(tmp_path.glob("projector_*.npz"))
    separable_slab, _, _ = _scoring_slab(reference, gridding_kernel="separable")
    entries = sorted(tmp_path.glob("projector_*.npz"))

    assert len(radial_entries) == 1 and len(entries) == 2
    assert np.max(np.abs(separable_slab - radial_slab)) > 1e-6 * np.max(np.abs(radial_slab))
    # Each window reads its own entry back.
    builder = _record(monkeypatch, setup, "reference_to_relion_projector_half_maps_and_power")
    assert_matches(_scoring_slab(reference)[0], radial_slab)
    assert_matches(_scoring_slab(reference, gridding_kernel="separable")[0], separable_slab)
    assert not builder


# --- expected accuracy -------------------------------------------------------------------------


def test_expected_accuracy_projector_uses_the_selected_window(monkeypatch):
    references = _reference()[None]
    calls = _record(monkeypatch, setup, "setup_relion_projector_on_host")

    radial = expected_accuracy._projector_data(references, 2 * R_MAX, PF)
    separable = expected_accuracy._projector_data(references, 2 * R_MAX, PF, "separable")

    assert "gridding_kernel" not in calls[0][1] and calls[1][1]["gridding_kernel"] == "separable"
    monkeypatch.undo()
    assert_matches(
        radial[0], setup.setup_relion_projector_on_host(references[0], R_MAX, ori_size=N, padding_factor=PF)[0]
    )
    corrected, _ = rf.griddingCorrect_square(jnp.asarray(references[0]), N, PF, order=1)
    expected, _ = setup._build_projector_window(
        corrected, R_MAX, N, PF, R_MAX, chunk_bytes=None, to_host=True, output_radius=R_MAX
    )
    assert_matches(separable[0], expected)


@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_expected_accuracy_inputs_forward_the_kernel_only_when_separable(monkeypatch, kernel):
    recorded = {}
    monkeypatch.setattr(
        expected_accuracy, "estimate_relion_expected_accuracy", lambda **kw: recorded.update(kw) or "acc"
    )
    ea = SimpleNamespace(half1_particle_ids="pids", half1_ctf_params="ctf", do_ctf_correction=False)
    inputs = expected_accuracy.Half1AccuracyInputs(
        trial_order_local="order",
        dataset="half1",
        volume_shape=[8, 8, 8],
        padding_factor=2,
        sigma2_fudge=4.0,
        optimizer_random_seed=11,
        expected_accuracy=ea,
        **({} if kernel == "radial" else {"gridding_kernel": kernel}),
    )
    inputs.estimate(
        reference_fourier="ref",
        best_eulers_deg="eul",
        class_ids="ids",
        class_weights="w",
        sigma2_noise_native="noise",
        current_image_size=56,
    )
    assert recorded.get("gridding_kernel", "radial") == kernel
    assert ("gridding_kernel" in recorded) == (kernel == "separable")


def test_expected_accuracy_refuses_separable_for_subtomograms():
    from relax.refinement.tomo_half import TomoHalf

    with pytest.raises(NotImplementedError, match="subtomogram"):
        expected_accuracy.estimate_relion_expected_accuracy(
            reference_fourier=None,
            volume_shape=(N, N, N),
            best_eulers_deg=None,
            class_ids=None,
            class_weights=None,
            sigma2_noise_native=None,
            dataset=object.__new__(TomoHalf),
            trial_order_local=None,
            current_image_size=N,
            padding_factor=PF,
            sigma2_fudge=1.0,
            random_seed=0,
            gridding_kernel="separable",
        )


# --- M-step ------------------------------------------------------------------------------------


def _accumulators(seed=1):
    rng = np.random.default_rng(seed)
    padded = (PF * N,) * 3
    real = rng.normal(size=padded)
    from recovar.core import fourier_transform_utils as ftu

    ft_y = np.asarray(ftu.get_dft3(jnp.asarray(real))).reshape(-1).astype(np.complex64)
    ft_ctf = np.full(ft_y.shape, 3.0, dtype=np.float32)
    return ft_ctf, ft_y


def _reconstruct(**kwargs):
    ft_ctf, ft_y = _accumulators()
    return np.asarray(
        mean_helpers._reconstruct_volume_eager(
            ft_ctf,
            ft_y,
            (N, N, N),
            PF,
            tau=None,
            tau2_fudge=1.0,
            projection_padding_factor=PF,
            use_spherical_mask=False,
            return_real_space=True,
            **kwargs,
        )
    ).reshape(N, N, N)


@pytest.mark.parametrize("kernel,recovar_name", [("radial", "radial"), ("separable", "square")])
def test_reconstruction_hands_recovar_the_selected_window(monkeypatch, kernel, recovar_name):
    calls = _record(monkeypatch, rf, "post_process_from_filter_v2")
    _reconstruct(**({} if kernel == "radial" else {"gridding_kernel": kernel}))
    assert len(calls) == 1
    assert calls[0][1]["gridding_correct"] == recovar_name
    assert calls[0][1]["gridding_padding_factor"] == PF and calls[0][1]["kernel_width"] == 1


def test_radial_reconstruction_is_the_default_and_matches_the_radial_window():
    uncorrected = _reconstruct(grid_correct=False)
    default = _reconstruct()
    expected, _ = rf.griddingCorrect(jnp.asarray(uncorrected), N, PF, order=1)
    # Float32 reconstruction, float64 window division in the oracle.
    assert_matches(default, np.asarray(expected, dtype=default.dtype), rtol=1e-5)
    assert_matches(_reconstruct(gridding_kernel="radial"), default)


def test_separable_reconstruction_matches_recovar_square_window():
    uncorrected = _reconstruct(grid_correct=False)
    separable = _reconstruct(gridding_kernel="separable")
    expected, _ = rf.griddingCorrect_square(jnp.asarray(uncorrected), N, PF, order=1)
    assert_matches(separable, np.asarray(expected, dtype=separable.dtype), rtol=1e-5)
    assert np.max(np.abs(separable - _reconstruct())) > 1e-4 * np.max(np.abs(separable))


def test_separable_reconstruction_matches_recovar_square_window_f64():
    """Float64 companion of the window oracle: the same division at double precision."""
    jax.config.update("jax_enable_x64", True)
    volume = jnp.asarray(_reference(3))
    square, window = rf.griddingCorrect_square(volume, N, PF, order=1)
    axis = np.sinc((np.arange(N) - N // 2) / (N * PF)) ** 2
    oracle = axis[:, None, None] * axis[None, :, None] * axis[None, None, :]
    assert_matches(np.asarray(window), oracle, rtol=1e-12)
    assert_matches(np.asarray(square), np.asarray(volume) / oracle, rtol=1e-12)


def test_reconstruction_rejects_an_unknown_kernel():
    with pytest.raises(ValueError, match="gridding_kernel"):
        _reconstruct(gridding_kernel="square")


def test_every_reconstruction_branch_takes_the_selected_window():
    """The device route and both large host-staged finishes read the one resolved name."""
    source = inspect.getsource(mean_helpers._reconstruct_volume_eager)
    assert source.count("gridding_correct=gridding_correct,") == 3
    assert 'gridding_correct="' not in source


@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_forward_and_m_step_use_the_same_window(kernel):
    kwargs = {} if kernel == "radial" else {"gridding_kernel": kernel}
    ones = jnp.ones((N, N, N), dtype=jnp.float64)
    correct = setup._gridding_corrected if kernel == "radial" else setup._gridding_corrected_separable
    forward_window = 1.0 / np.asarray(correct(ones, ori_size=N, padding_factor=PF))
    uncorrected = _reconstruct(grid_correct=False)
    corrected = _reconstruct(**kwargs)
    m_step_window = uncorrected / corrected
    assert_matches(m_step_window, forward_window.astype(m_step_window.dtype), rtol=1e-4)
    recovar_window = _window(rf.griddingCorrect if kernel == "radial" else rf.griddingCorrect_square)
    assert_matches(forward_window, recovar_window, rtol=1e-12)


@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_k1_reconstruction_wrappers_forward_the_kernel_only_when_separable(monkeypatch, kernel):
    calls = []

    def fake(*_args, **kwargs):
        calls.append(kwargs)
        return jnp.ones(N**3, dtype=jnp.complex64)

    monkeypatch.setattr(mean_helpers, "_reconstruct_volume_eager", fake)
    monkeypatch.setattr(mean_helpers, "_finish_host_staged_reconstruction", lambda result, *_: result)
    extra = {} if kernel == "radial" else {"gridding_kernel": kernel}
    settings = mean_helpers.ReconstructionSettings(
        grid_size=N,
        voxel_size=1.0,
        volume_shape=(N, N, N),
        padding_factor=PF,
        projection_padding_factor=PF,
        minres_map=5,
        width_mask_edge=5,
        fmask_edge=2,
        **extra,
    )
    mean_helpers.reconstruct_k1_means(
        ("y0", "y1"),
        ("c0", "c1"),
        (np.ones(5), np.ones(5)),
        settings,
        current_size=N,
        tau2_fudge=1.0,
        accumulator_volume_shape=None,
        tau_is_1d=True,
    )
    mean_helpers.reconstruct_unregularized_k1_halfmaps(
        ("y0", "y1"),
        ("c0", "c1"),
        (N, N, N),
        tau2_fudge=1.0,
        padding_factor=PF,
        projection_padding_factor=PF,
        minres_map=5,
        **extra,
    )
    assert len(calls) == 4
    for kwargs in calls:
        assert ("gridding_kernel" in kwargs) == (kernel == "separable")
        assert kwargs.get("gridding_kernel", "radial") == kernel


# --- iteration loop wiring ---------------------------------------------------------------------


def _loop_calls(name):
    tree = ast.parse(inspect.getsource(iteration_loop.refine_single_volume))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _spreads(call):
    return {
        keyword.value.id for keyword in call.keywords if keyword.arg is None and isinstance(keyword.value, ast.Name)
    }


@pytest.mark.parametrize(
    "callee,count",
    [
        ("_relion_projector_half_maps_for_scoring", 3),
        ("ReconstructionSettings", 1),
        ("Half1AccuracyInputs", 1),
        ("reconstruct_unregularized_k1_halfmaps", 1),
        ("DenseExecutionPolicy", 2),
        ("LocalExecutionPolicy", 2),
    ],
)
def test_loop_threads_the_kernel_to_every_k1_site(callee, count):
    calls = _loop_calls(callee)
    assert len(calls) == count
    assert all("gridding_kernel_kwargs" in _spreads(call) for call in calls)


def test_loop_reconstructions_all_carry_the_kernel():
    """The solvent-FSC maps directly; the final unfiltered, merged and half maps through their shared kwargs."""
    calls = _loop_calls("_reconstruct_volume_eager")
    assert len(calls) == 5
    assert all(_spreads(call) & {"gridding_kernel_kwargs", "final_reconstruction_kwargs"} for call in calls)
    final_kwargs = next(
        node.value
        for node in ast.walk(ast.parse(inspect.getsource(iteration_loop.refine_single_volume)))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "final_reconstruction_kwargs"
    )
    assert "gridding_kernel_kwargs" in _spreads(final_kwargs)


def test_loop_reports_the_window_of_the_final_map():
    source = inspect.getsource(iteration_loop.refine_single_volume)
    assert '"final_all_data_gridding_correct": gridding_kernel,' in source
    assert 'gridding_kernel_kwargs = {} if gridding_kernel == "radial" else' in source


# --- option and refusals -----------------------------------------------------------------------


def test_option_defaults_to_radial_and_rejects_unknown_values():
    assert RelionParityOptions().gridding_kernel == "radial"
    assert RelionParityOptions(gridding_kernel="separable").gridding_kernel == "separable"
    with pytest.raises(ValueError, match="gridding_kernel"):
        RelionParityOptions(gridding_kernel="square")


def test_command_line_exposes_the_option():
    from relax.refinement import full_refinement

    base = ["--data_dir", "data", "--output", "out"]
    assert full_refinement._parse_args(base).gridding_kernel == "radial"
    assert full_refinement._parse_args([*base, "--gridding_kernel", "separable"]).gridding_kernel == "separable"
    with pytest.raises(SystemExit):
        full_refinement._parse_args([*base, "--gridding_kernel", "square"])
    source = inspect.getsource(full_refinement)
    assert "gridding_kernel=args.gridding_kernel," in source


def test_class3d_command_refuses_separable():
    from relax.refinement import full_refinement

    full_refinement._require_k1_for_gridding_kernel("radial", 4)
    full_refinement._require_k1_for_gridding_kernel("separable", 1)
    with pytest.raises(SystemExit, match="K=1"):
        full_refinement._require_k1_for_gridding_kernel("separable", 4)


def _stub_half(owner=None):
    half = SimpleNamespace() if owner is None else object.__new__(owner)
    for name, value in dict(volume_shape=(N, N, N), image_shape=(N, N), voxel_size=1.0, n_units=4).items():
        object.__setattr__(half, name, value)
    return half


def _refine(halves, **options):
    return iteration_loop.refine_single_volume(
        halves,
        None,
        None,
        None,
        None,
        options=RefinementOptions(parity=RelionParityOptions(gridding_kernel="separable"), **options),
    )


def test_class3d_refuses_separable():
    with pytest.raises(NotImplementedError, match="K=1 single-particle"):
        _refine([_stub_half(), _stub_half()], k_class=KClassOptions(n_classes=2))


def test_tomography_refuses_separable():
    from relax.refinement.tomo_half import TomoHalf

    with pytest.raises(NotImplementedError, match="K=1 single-particle"):
        _refine([_stub_half(TomoHalf), _stub_half(TomoHalf)])


def test_class_reconstruction_refuses_separable():
    settings = mean_helpers.ReconstructionSettings(
        grid_size=N,
        voxel_size=1.0,
        volume_shape=(N, N, N),
        padding_factor=PF,
        projection_padding_factor=PF,
        minres_map=5,
        width_mask_edge=5,
        fmask_edge=2,
        gridding_kernel="separable",
    )
    with pytest.raises(NotImplementedError, match="K=1 only"):
        mean_helpers.reconstruct_class_means(
            None,
            None,
            None,
            settings,
            n_classes=2,
            iteration=0,
            current_size=N,
            tau2_fudge=1.0,
            accumulator_volume_shape=None,
            tau_is_1d=True,
        )


def test_dense_engine_refuses_separable():
    """The no-slab dense engine pads and corrects its own reference with the radial window."""
    from relax.dense.em_engine import run_em

    with pytest.raises(NotImplementedError, match="dense engine"):
        run_em(None, None, None, None, None, None, "linear_interp", gridding_kernel="separable")


def test_direct_k1_dense_route_refuses_separable(monkeypatch):
    monkeypatch.setattr(half_scoring, "dataset_projection_magnification", lambda _dataset: None)
    monkeypatch.setattr(half_scoring, "warn_deprecated_engine", lambda *_args: None)
    half = SimpleNamespace(
        optics_group_ids_k=None, experiment_dataset=None, means_k=None, mean_variance=None, noise_variance_k=None
    )
    sampling = SimpleNamespace(effective_rotations=None, current_translations=None, disc_type="linear_interp")
    execution = SimpleNamespace(disable_adjoint_y=False, disable_adjoint_ctf=False)
    with pytest.raises(NotImplementedError, match="dense engine"):
        half_scoring._score_direct_k1_dense(half, sampling, execution, {"gridding_kernel": "separable"})


def test_dense_route_hands_the_engines_the_kernel_only_when_separable():
    source = inspect.getsource(half_scoring._score_half_dense_one_shape)
    assert 'if execution.gridding_kernel != "radial":\n' in source
    assert 'em_kwargs["gridding_kernel"] = execution.gridding_kernel' in source
    assert "gridding_kernel" not in half_scoring._DENSE_EM_STATIC_KWARGS


def test_adaptive_engine_without_projector_slab_refuses_separable():
    from relax.classification.k_class import run_dense_k_class_em_adaptive

    with pytest.raises(NotImplementedError, match="projector slab"):
        run_dense_k_class_em_adaptive(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            relion_projector_half=None,
            gridding_kernel="separable",
        )


def test_local_search_without_projector_slab_refuses_separable():
    execution = half_scoring.LocalExecutionPolicy(
        disable_adjoint_y=False, disable_adjoint_ctf=False, gridding_kernel="separable"
    )
    with pytest.raises(NotImplementedError, match="projector slab"):
        half_scoring._score_half_local_one_shape(None, None, None, None, execution, None, None)


def test_captured_relion_projector_refuses_separable():
    """A replayed RELION Projector::data carries RELION's radial window."""
    source = inspect.getsource(iteration_loop.refine_single_volume)
    branch = source.index('if captured_projector_state is not None:\n            if gridding_kernel != "radial":')
    assert source.index("raise NotImplementedError", branch) < source.index(
        "_validate_captured_relion_projector_for_iteration(", branch
    )


def test_initial_model_and_ppca_commands_do_not_take_the_option():
    """VDAM and PPCA keep their own gridding; they cannot be asked for the separable window."""
    from relax.commands import initial_model, ppca_initial_model

    assert "--gridding_kernel" not in initial_model.make_parser()._option_string_actions
    assert "gridding_kernel" not in inspect.getsource(initial_model)
    assert "gridding_kernel" not in inspect.getsource(ppca_initial_model)
