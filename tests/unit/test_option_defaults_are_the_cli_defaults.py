"""Every option record's default is the value ``relax refine`` hands the controller when no flag is given
(code rule 5: a default is defined once, and the command line is the truth)."""

from __future__ import annotations

import dataclasses

import pytest
from helpers.tiny_main import controller_inputs

pytestmark = pytest.mark.unit

# Fields whose value the command takes from the data, the seed or the start-up state, not from a flag default.
FROM_THE_RUN = {
    "schedule": {"init_current_size", "particle_diameter_ang", "ini_high_angstrom", "init_data_vs_prior"},
    "parity": {
        "perturb_seed", "optimizer_random_seed", "relion_optics_image_sizes", "relion_optics_pixel_sizes",
        "optics_group_ids_per_half", "relion_model_pixel_size", "relion_firstiter_ini_high_angstrom",
        "preserve_bpref_particle_order",
    },
    "solvent": {"split_draws"},
}
# Groups built from the data or the run files, not from flags.
NOT_FLAGS = {"start", "checkpoint", "expected_accuracy", "precision"}


def test_every_record_default_is_what_relax_refine_passes_without_flags(monkeypatch, tmp_path):
    options = controller_inputs(monkeypatch, tmp_path, "refine")["options"]
    differing = []
    for group in dataclasses.fields(options):
        if group.name in NOT_FLAGS:
            continue
        record = getattr(options, group.name)
        if not dataclasses.is_dataclass(record):
            continue
        for item in dataclasses.fields(record):
            if item.name in FROM_THE_RUN.get(group.name, ()):
                continue
            if item.default is not dataclasses.MISSING:
                default = item.default
            elif item.default_factory is not dataclasses.MISSING:
                default = item.default_factory()
            else:
                continue
            value = getattr(record, item.name)
            if not (value is default or value == default):
                differing.append(f"{group.name}.{item.name}: command {value!r}, record default {default!r}")
    assert not differing, "\n".join(differing)
