"""relax's refinement defaults are the RELION 5 GUI's job defaults (docs/development/relion_defaults.md)."""

from types import SimpleNamespace

import pytest

from scripts import run_full_refinement as driver

pytestmark = pytest.mark.unit


def _parsed(*tokens):
    return driver._parse_args(["--data_dir", "data", "--output", "out", *tokens])


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
        driver._parse_args([item for pair in tokens.items() for item in pair])
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize(
    "n_classes, frozen, expected",
    [
        (1, None, (999, "relion_cuda", True)),
        (4, None, (25, "host_numpy", True)),
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
    driver._resolve_relion_gui_defaults(args)
    assert (args.max_iter, args.image_fourier_backend, args.apply_initial_lowpass) == expected


def test_explicit_values_are_kept():
    args = SimpleNamespace(
        n_classes=1, max_iter=3, image_fourier_backend="host_numpy", apply_initial_lowpass=False, frozen_boundary_dir=None
    )
    driver._resolve_relion_gui_defaults(args)
    assert (args.max_iter, args.image_fourier_backend, args.apply_initial_lowpass) == (3, "host_numpy", False)


def test_seed_defaults_to_the_time_like_relion(monkeypatch):
    monkeypatch.setattr(driver.time, "time", lambda: 1234567890.7)
    assert driver._resolve_optimizer_random_seed(None, None) == (1234567890, "RELION default -1: the time")
    assert driver._resolve_optimizer_random_seed(7, None) == (7, "explicit CLI")


def test_mask_diameter_falls_back_to_the_gui_default(monkeypatch):
    applied = {}

    class Backend:
        def set_relion_image_mask(self, **kwargs):
            applied.update(kwargs)

    ds = SimpleNamespace(image_source=SimpleNamespace(backend=Backend()), voxel_size=2.0)
    monkeypatch.setattr(driver, "_find_relion_optimiser_star", lambda args: None)
    params = driver._maybe_apply_relion_image_mask(ds, SimpleNamespace(particle_diameter_ang=None, width_mask_edge_px=5.0))
    assert params == (200.0, 5.0)
    assert applied["particle_diameter_ang"] == 200.0


def test_initial_model_seed_default_is_relions():
    from relax.commands.initial_model import DEFAULTS, make_parser

    assert DEFAULTS.random_seed == -1
    assert make_parser().parse_args(["--i", "particles.star"]).random_seed == -1


def test_seed_used_is_recorded_in_results_and_ledger():
    source = driver.Path(driver.__file__).read_text()
    assert '"random_seed": np.int64(args.seed),' in source
    assert '"random_seed": int(args.seed),\n            "random_seed_source": str(optimizer_seed_source),' in source
