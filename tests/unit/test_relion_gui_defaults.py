"""relax's refinement defaults are the RELION 5 GUI's job defaults (docs/development/relion_defaults.md)."""

from types import SimpleNamespace

import pytest

from relax.refinement import command_options, particle_loading

pytestmark = pytest.mark.unit


def _parsed(*tokens):
    return command_options.parse_refinement_args(["--data_dir", "data", "--output", "out", *tokens])


def test_refine3d_and_class3d_gui_defaults():
    args = _parsed()
    # pipeline_jobs.cpp: 7.5 deg with oversampling 1, offsets 5/1 px (x2 for oversampling),
    # ini_high 60, mask diameter 200 (resolved at run time), firstiter_cc on.
    assert (args.healpix_order, args.adaptive_oversampling, args.auto_local_healpix_order) == (2, 1, 4)
    assert (args.offset_range, args.offset_step) == (5.0, 2.0)
    assert args.init_resolution == 60.0
    assert args.firstiter_cc is True
    assert args.apply_initial_lowpass is None
    assert args.particle_diameter_ang is None
    assert (args.perturb_factor, args.offset_sigma_angstrom, args.width_mask_edge_px) == (0.5, 10.0, 5.0)
    assert args.tau2_fudge is None and args.seed is None and args.max_iter is None


def test_debug_runs_can_turn_off_gui_start_options():
    args = _parsed("--no-firstiter_cc", "--no-apply-initial-lowpass")
    assert args.firstiter_cc is False and args.apply_initial_lowpass is False


@pytest.mark.parametrize("missing", ["--data_dir", "--output"])
def test_input_and_output_are_required(missing, capsys):
    tokens = {"--data_dir": "data", "--output": "out"}
    tokens.pop(missing)
    with pytest.raises(SystemExit):
        command_options.parse_refinement_args([item for pair in tokens.items() for item in pair])
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize(
    "n_classes, frozen, expected",
    [
        (1, None, (999, "relion_cuda", True)),
        # Class3D on the resident pass 2 reads RELION's CUDA preprocessing (exact BPref operands).
        (4, None, (25, "relion_cuda", True)),
        (1, "boundary", (999, "relion_cuda", False)),
    ],
)
def test_job_type_defaults(n_classes, frozen, expected):
    args = SimpleNamespace(
        n_classes=n_classes,
        max_iter=None,
        image_fourier_backend="auto",
        apply_initial_lowpass=None,
        frozen_boundary_dir=frozen,
    )
    job = command_options.resolve_job_defaults(args)
    assert (job.max_iter, job.image_fourier_backend, job.apply_initial_lowpass) == expected


def test_explicit_values_are_kept():
    args = SimpleNamespace(
        n_classes=1, max_iter=3, image_fourier_backend="host_numpy", apply_initial_lowpass=False, frozen_boundary_dir=None
    )
    job = command_options.resolve_job_defaults(args)
    assert (job.max_iter, job.image_fourier_backend, job.apply_initial_lowpass) == (3, "host_numpy", False)


def test_seed_defaults_to_the_time_like_relion(monkeypatch):
    monkeypatch.setattr(command_options.time, "time", lambda: 1234567890.7)
    args = SimpleNamespace(
        seed=None, continue_optimiser_star=None, relion_optimiser=None, relion_init_dir=None,
        perturb_replay_relion_dir=None,
    )
    assert command_options.resolve_seed(args, None, sealed=False) == command_options.RandomSeed(
        1234567890, "RELION default -1: the time"
    )
    args.seed = 7
    assert command_options.resolve_seed(args, None, sealed=False) == command_options.RandomSeed(7, "explicit CLI")


def test_mask_diameter_falls_back_to_the_gui_default(monkeypatch):
    applied = {}

    class Backend:
        def set_relion_image_mask(self, **kwargs):
            applied.update(kwargs)

    ds = SimpleNamespace(image_source=SimpleNamespace(backend=Backend()), voxel_size=2.0)
    monkeypatch.setattr(command_options, "find_relion_optimiser_star", lambda args, **_kwargs: None)
    params = particle_loading._apply_relion_image_mask(
        ds, SimpleNamespace(particle_diameter_ang=None, width_mask_edge_px=5.0), relion_half_sets_from_input=False
    )
    assert params == (200.0, 5.0)
    assert applied["particle_diameter_ang"] == 200.0


def test_initial_model_seed_default_is_relions():
    from relax.commands.initial_model import DEFAULTS, make_parser

    assert DEFAULTS.random_seed == -1
    assert make_parser().parse_args(["--i", "particles.star"]).random_seed == -1


def test_seed_used_is_recorded_in_results_and_ledger(monkeypatch, tmp_path):
    import json

    import numpy as np
    from helpers.tiny_main import run_tiny_main

    ledger = tmp_path / "ledger.json"
    output = run_tiny_main(monkeypatch, tmp_path, "refine", "--max_iter", "1", "--benchmark_ledger_json", ledger)
    with np.load(output / "refinement_results.npz", allow_pickle=True) as archive:
        assert archive["random_seed"].dtype == np.int64 and int(archive["random_seed"]) == 42
    recorded = json.loads(ledger.read_text())
    assert (recorded["random_seed"], recorded["random_seed_source"]) == (42, "explicit CLI")
