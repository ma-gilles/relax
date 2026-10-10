"""The ``gridding_kernel`` option: RELION's radial correction window or the separable one.

``radial`` (the default) is RELION's ``sinc^2(|x| / (pf N))``; ``separable`` is the per-axis
product, the exact transform of the trilinear kernel. The separable window must reach every K=1
site (scoring projector, expected accuracy, every reconstruction, the final maps), and every
route that keeps the radial window must refuse it.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.reconstruction_settings import reconstruction_settings
from helpers.run_options import stand_in
from helpers.tiny_refinement import record_calls, run_tiny_refinement, unconverged_accuracy

from relax.helpers import expected_accuracy
from relax.refinement import (
    command_options,
    final_reconstruction,
    finalization,
    iteration_loop,
    numbered_reconstruction,
    projector_preparation,
)
from relax.refinement.ports import InputSource, RunObserver
from relax.refinement.refinement_options import (
    KClassOptions,
    ReconstructionPrograms,
    RelionConsistencyOptions,
)
from relax.relion import relion_projector_setup as setup

pytestmark = pytest.mark.unit

PF = 2


def _reference(n, seed=0):
    jax.config.update("jax_enable_x64", True)
    return np.random.default_rng(seed).normal(size=(n, n, n))


def _radial_window(n, pf=PF):
    """RELION's Projector::griddingCorrect window (projector.cpp:595-628), in numpy."""
    c = np.arange(n) - n // 2
    r = np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2)
    return np.sinc(r / (n * pf)) ** 2


def _separable_window(n, pf=PF):
    """The trilinear kernel's transform: the product of one sinc^2 per axis, in numpy."""
    axis = np.sinc((np.arange(n) - n // 2) / (n * pf)) ** 2
    return axis[:, None, None] * axis[None, :, None] * axis[None, None, :]


WINDOWS = {"radial": _radial_window, "separable": _separable_window}


# --- scoring projector -------------------------------------------------------------------------


@pytest.mark.parametrize("n,r_max", [(8, 4), (8, 3), (6, 3), (6, 2)])
@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_projector_setup_divides_by_the_selected_window(kernel, n, r_max):
    """Full and cropped radius, two box sizes: the slab is the transform of reference / window."""
    reference = _reference(n)
    slab, power = setup.setup_relion_projector_on_host(
        reference, r_max, box_size=n, padding_factor=PF, gridding_kernel=kernel
    )
    corrected = jnp.asarray(reference / WINDOWS[kernel](n))
    expected_slab, expected_power = setup._build_projector_window(
        corrected, r_max, n, PF, r_max, chunk_bytes=None, to_host=True, output_radius=r_max
    )
    assert_matches(slab, expected_slab, rtol=1e-12)
    assert_matches(power, expected_power, rtol=1e-12)


def test_projector_setup_default_is_relions_radial_window():
    reference = _reference(8)
    default = setup.setup_relion_projector_on_host(reference, 3, box_size=8, padding_factor=PF)
    radial = setup.setup_relion_projector_on_host(reference, 3, box_size=8, padding_factor=PF, gridding_kernel="radial")
    separable = setup.setup_relion_projector_on_host(
        reference, 3, box_size=8, padding_factor=PF, gridding_kernel="separable"
    )
    assert_matches(default[0], radial[0])
    assert_matches(default[1], radial[1])
    assert np.max(np.abs(separable[0] - radial[0])) > 1e-6 * np.max(np.abs(radial[0]))


def test_separable_window_is_larger_off_axis_and_equal_on_axis():
    """prod sinc^2(x_i) >= sinc^2(|x|) with equality on the axes: the radial window over-corrects the diagonals."""
    radial, separable = _radial_window(8), _separable_window(8)
    assert np.all(separable >= radial - 1e-15)
    assert_matches(separable[4, 4, :], radial[4, 4, :], rtol=1e-12)
    assert separable[0, 0, 0] > 1.02 * radial[0, 0, 0]  # 3.1% at the corner of an 8-voxel box


def test_projector_setup_rejects_an_unknown_kernel():
    with pytest.raises(ValueError, match="gridding_kernel"):
        setup.setup_relion_projector_on_host(_reference(8), 3, box_size=8, padding_factor=PF, gridding_kernel="x")


def _scoring_projector(reference_real, r_max=3, **kwargs):
    from recovar.core import fourier_transform_utils as ftu

    n = reference_real.shape[0]
    reference_ft = np.asarray(ftu.get_dft3(jnp.asarray(reference_real))).reshape(-1)
    return projector_preparation.prepare_scoring_projector(
        reference_ft, volume_shape=(n, n, n), current_size=2 * r_max, padding_factor=PF, n_classes=1, **kwargs
    )


def test_scoring_projector_hands_the_setup_the_kernel(monkeypatch):
    monkeypatch.delenv("RELAX_RELION_PROJECTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("RELAX_RELION_PROJECTOR_DUMP_DIR", raising=False)
    reference = _reference(8)
    builder = record_calls(monkeypatch, setup, "setup_relion_projector_on_host")

    radial = _scoring_projector(reference)
    separable = _scoring_projector(reference, gridding_kernel="separable")

    assert [kwargs["gridding_kernel"] for _, kwargs in builder] == ["radial", "separable"]
    from recovar.utils.helpers import recovar_volume_to_relion

    relion_layout = np.asarray(recovar_volume_to_relion(reference), dtype=np.float64)
    expected, _ = setup._build_projector_window(
        jnp.asarray(relion_layout / _separable_window(8)), radial.r_max, 8, PF, radial.r_max,
        chunk_bytes=None, to_host=True, output_radius=radial.r_max,
    )
    # The reference goes through a Fourier round trip before the setup.
    assert_matches(separable.data[0], expected, rtol=1e-11)
    assert np.max(np.abs(separable.data - radial.data)) > 1e-6 * np.max(np.abs(radial.data))


def test_projector_cache_keeps_radial_and_separable_entries_apart(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("RELAX_RELION_PROJECTOR_DUMP_DIR", raising=False)
    reference = _reference(8)

    radial = _scoring_projector(reference)
    assert len(sorted(tmp_path.glob("projector_*.npz"))) == 1
    separable = _scoring_projector(reference, gridding_kernel="separable")
    assert len(sorted(tmp_path.glob("projector_*.npz"))) == 2
    # Each window reads its own entry back without building.
    builder = record_calls(monkeypatch, setup, "reference_to_relion_projector_half_maps_and_power")
    assert_matches(_scoring_projector(reference).data, radial.data)
    assert_matches(_scoring_projector(reference, gridding_kernel="separable").data, separable.data)
    assert not builder


def test_a_half_without_its_projector_slab_refuses_separable():
    """A scorer handed no slab pads and corrects the reference itself, with the radial window."""
    slab = projector_preparation.PreparedProjector(data=None, r_max=3)
    projector_preparation.require_projectors_for_gridding_kernel([slab, None], "radial")
    projector_preparation.require_projectors_for_gridding_kernel([slab, slab], "separable")
    with pytest.raises(NotImplementedError, match="projector slab"):
        projector_preparation.require_projectors_for_gridding_kernel([slab, None], "separable")


# --- expected accuracy -------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_expected_accuracy_projector_uses_the_selected_window(kernel):
    references = _reference(8)[None]
    built = expected_accuracy._projector_data(references, 6, PF, kernel)
    expected, _ = setup._build_projector_window(
        jnp.asarray(references[0] / WINDOWS[kernel](8)), 3, 8, PF, 3, chunk_bytes=None, to_host=True, output_radius=3
    )
    assert_matches(built[0], expected, rtol=1e-12)


@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_expected_accuracy_inputs_hand_the_estimator_the_kernel(monkeypatch, kernel):
    recorded = {}
    monkeypatch.setattr(
        expected_accuracy, "estimate_relion_expected_accuracy", lambda **kw: recorded.update(kw) or "acc"
    )
    inputs = expected_accuracy.Half1AccuracyInputs(
        trial_order_local="order",
        dataset="half1",
        volume_shape=[8, 8, 8],
        padding_factor=2,
        sigma2_fudge=4.0,
        optimizer_random_seed=11,
        expected_accuracy=SimpleNamespace(half1_particle_ids="pids", half1_ctf_params="ctf", do_ctf_correction=False),
        gridding_kernel=kernel,
    )
    inputs.estimate(
        reference_fourier="ref",
        best_eulers_deg="eul",
        class_ids="ids",
        class_weights="w",
        sigma2_noise_native="noise",
        current_image_size=56,
    )
    assert recorded["gridding_kernel"] == kernel


def test_expected_accuracy_refuses_separable_for_subtomograms():
    from relax.refinement.tomo_half import TomoHalf

    with pytest.raises(NotImplementedError, match="subtomogram"):
        expected_accuracy.estimate_relion_expected_accuracy(
            reference_fourier=None,
            volume_shape=(8, 8, 8),
            best_eulers_deg=None,
            class_ids=None,
            class_weights=None,
            sigma2_noise_native=None,
            dataset=object.__new__(TomoHalf),
            trial_order_local=None,
            current_image_size=8,
            padding_factor=PF,
            sigma2_fudge=1.0,
            random_seed=0,
            gridding_kernel="separable",
        )


# --- reconstruction ----------------------------------------------------------------------------


def _accumulators(n, seed=1):
    from recovar.core import fourier_transform_utils as ftu

    real = np.random.default_rng(seed).normal(size=(PF * n,) * 3)
    ft_y = np.asarray(ftu.get_dft3(jnp.asarray(real))).reshape(-1).astype(np.complex64)
    return np.full(ft_y.shape, 3.0, dtype=np.float32), ft_y


def _reconstruct(n=8, **kwargs):
    ft_ctf, ft_y = _accumulators(n)
    return np.asarray(
        numbered_reconstruction._reconstruct_volume_eager(
            ft_ctf,
            ft_y,
            (n, n, n),
            PF,
            tau=None,
            tau2_fudge=1.0,
            projection_padding_factor=PF,
            use_spherical_mask=False,
            return_real_space=True,
            **kwargs, programs=ReconstructionPrograms.from_environ(),
        )
    ).reshape(n, n, n)


@pytest.mark.parametrize("n", [8, 6])
@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_reconstruction_divides_by_the_selected_window(kernel, n):
    uncorrected = _reconstruct(n, grid_correct=False)
    corrected = _reconstruct(n, gridding_kernel=kernel)
    # Float32 reconstruction against a float64 window division.
    assert_matches(corrected, (uncorrected / WINDOWS[kernel](n)).astype(corrected.dtype), rtol=1e-5)


def test_reconstruction_window_oracle_f64():
    """Float64 companion: RECOVAR's two window functions against the numpy windows."""
    from recovar.reconstruction import relion_functions as rf

    jax.config.update("jax_enable_x64", True)
    volume = jnp.asarray(_reference(8, seed=3))
    for kernel, correct in (("radial", rf.griddingCorrect), ("separable", rf.griddingCorrect_square)):
        corrected, window = correct(volume, 8, PF, order=1)
        assert_matches(np.asarray(window), WINDOWS[kernel](8), rtol=1e-12)
        assert_matches(np.asarray(corrected), np.asarray(volume) / WINDOWS[kernel](8), rtol=1e-12)


def test_reconstruction_default_is_relions_radial_window():
    default = _reconstruct()
    assert_matches(_reconstruct(gridding_kernel="radial"), default)
    separable = _reconstruct(gridding_kernel="separable")
    assert np.max(np.abs(separable - default)) > 1e-4 * np.max(np.abs(separable))


def test_an_unknown_kernel_is_refused_where_it_enters():
    """The kernel is checked once, in the options (the reconstruction trusts it)."""
    with pytest.raises(ValueError, match="gridding_kernel"):
        RelionConsistencyOptions(gridding_kernel="square")


@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_forward_and_reconstruction_use_the_same_window(kernel):
    ones = jnp.ones((8, 8, 8), dtype=jnp.float64)
    correct = setup._gridding_corrected if kernel == "radial" else setup._gridding_corrected_separable
    forward_window = 1.0 / np.asarray(correct(ones, box_size=8, padding_factor=PF))
    reconstruction_window = _reconstruct(grid_correct=False) / _reconstruct(gridding_kernel=kernel)
    assert_matches(reconstruction_window, forward_window.astype(reconstruction_window.dtype), rtol=1e-4)


def _settings(**fields):
    return reconstruction_settings(
        box_size=8,
        voxel_size=1.0,
        volume_shape=(8, 8, 8),
        padding_factor=PF,
        projection_padding_factor=PF,
        minres_map=5,
        width_mask_edge=5,
        fmask_edge=2,
        tau2_fudge=1.0,
        particle_diameter_angstrom=None,
        first_iteration_lowpass_angstrom=None,
        **fields, programs=ReconstructionPrograms.from_environ(),
    )


@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_every_k1_reconstruction_hands_the_solve_the_kernel(monkeypatch, kernel):
    """Numbered regularized and unregularized half maps, final unfiltered, merged and half maps."""
    calls = []

    def solve(*_args, **kwargs):
        calls.append(kwargs)
        return jnp.ones(8**3, dtype=jnp.complex64)

    monkeypatch.setattr(numbered_reconstruction, "_reconstruct_volume_eager", solve)
    monkeypatch.setattr(numbered_reconstruction, "_finish_host_staged_reconstruction", lambda result, *_: result)
    settings = _settings(gridding_kernel=kernel)
    pair = ("a", "b")
    numbered_reconstruction.reconstruct_numbered_k1_halfmaps(
        pair, pair, (np.ones(5), np.ones(5)), settings,
        iteration=1, current_size=8, accumulator_volume_shape=None, relion_firstiter_cc_this_iter=False,
    )
    numbered_reconstruction.reconstruct_unregularized_k1_halfmaps(pair, pair, settings)
    final_reconstruction.reconstruct_unfiltered_halfmaps(
        pair, pair, settings=settings, current_size=8, accumulator_shape=None
    )
    final_reconstruction.reconstruct_final_halfmaps(
        [pair, pair], np.ones(5), settings=settings, current_size=8, accumulator_shape=None
    )
    assert [kwargs["gridding_kernel"] for kwargs in calls] == [kernel] * 9


def test_class_runs_refuse_separable_before_any_reconstruction():
    """Class3D keeps RELION's radial window (its tau2 is the power of the radially corrected reference): the
    loop's consistency route refuses the separable one before the first M-step, so the class reconstructions
    need not check it."""
    from relax.refinement.refinement_options import require_consistency_route

    separable = RelionConsistencyOptions(gridding_kernel="separable")
    with pytest.raises(NotImplementedError, match="K=1 single-particle refinement only"):
        require_consistency_route(
            stand_in.options(k_class=KClassOptions(n_classes=2), consistency=separable),
            subtomograms=False, several_image_shapes=False,
        )
    k1 = stand_in.options(consistency=separable)
    assert require_consistency_route(k1, subtomograms=False, several_image_shapes=False) is k1.consistency


# --- the controller ----------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["radial", "separable"])
def test_refinement_hands_every_k1_site_the_kernel(monkeypatch, kernel):
    """Two numbered iterations and the final all-data pass of the real controller."""
    reconstructions = record_calls(monkeypatch, numbered_reconstruction, "_reconstruct_volume_eager")
    numbered_projectors = record_calls(monkeypatch, projector_preparation, "prepare_scoring_projector")
    final_projectors = record_calls(monkeypatch, finalization, "prepare_scoring_projector")
    accuracy_kernels = []

    def estimate(self, **_kwargs):
        accuracy_kernels.append(self.gridding_kernel)
        return unconverged_accuracy()

    monkeypatch.setattr(expected_accuracy.Half1AccuracyInputs, "estimate", estimate)

    result = run_tiny_refinement(
        monkeypatch,
        parity=dict(perturb_seed=17, optimizer_random_seed=17),
        consistency=RelionConsistencyOptions(gridding_kernel=kernel),
    )

    assert result.final_all_data_ran and result.final_pass.gridding_correct == kernel
    # Per numbered iteration two regularized half maps; the final pass two unfiltered, one merged, two half maps.
    assert [kwargs["gridding_kernel"] for _, kwargs in reconstructions] == [kernel] * 9
    assert len(numbered_projectors) >= 4 and len(final_projectors) == 2
    assert {kwargs["gridding_kernel"] for _, kwargs in numbered_projectors + final_projectors} == {kernel}
    assert accuracy_kernels and set(accuracy_kernels) == {kernel}


# --- option and refusals -----------------------------------------------------------------------


BASE_ARGS = ["--data_dir", "data", "--output", "out"]


def test_class3d_command_refuses_separable():
    radial = command_options.parse_refinement_args([*BASE_ARGS, "--n_classes", "4"])
    assert command_options.resolve_consistency_options(radial).non_default() == {}
    separable = command_options.parse_refinement_args([*BASE_ARGS, "--n_classes", "4", "--gridding_kernel", "separable"])
    with pytest.raises(SystemExit, match="K=1"):
        command_options.resolve_consistency_options(separable)


def _stub_half(owner=None):
    half = SimpleNamespace() if owner is None else object.__new__(owner)
    for name, value in dict(volume_shape=(8, 8, 8), image_shape=(8, 8), voxel_size=1.0, n_units=4).items():
        object.__setattr__(half, name, value)
    return half


def _refine_stubs(halves, **options):
    return iteration_loop.refine_single_volume(
        halves,
        None,
        None,
        None,
        options=stand_in.options(consistency=RelionConsistencyOptions(gridding_kernel="separable"), **options),
        observer=RunObserver(), source=InputSource(),
    )


def test_class3d_refuses_separable():
    with pytest.raises(NotImplementedError, match="K=1 single-particle"):
        _refine_stubs([_stub_half(), _stub_half()], k_class=KClassOptions(n_classes=2))
