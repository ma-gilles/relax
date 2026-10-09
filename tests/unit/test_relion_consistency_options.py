"""Route rules shared by every RELION-consistency option (``RelionConsistencyOptions``).

Each option's default is RELION's rule. A non-default value is refused on the routes that keep
RELION's rules (subtomograms, optics groups on several image shapes, RELION-seeded, replayed or
frozen state), is recorded in the run files, and must be repeated by a continuation. The option's
own sites and oracles are tested in its own file.
"""

from types import SimpleNamespace

import pytest
from helpers.run_options import stand_in
from helpers.tiny_refinement import run_tiny_refinement

from relax.parity.relion_replay_source import RelionReplay
from relax.refinement import command_options, iteration_loop, iteration_snapshot
from relax.refinement.ports import RunObserver
from relax.refinement.refinement_options import (
    CheckpointOptions,
    RelionConsistencyOptions,
)

pytestmark = pytest.mark.unit

# (option, RELION's value, the corrected value)
OPTIONS = [
    ("gridding_kernel", "radial", "separable"),
    ("shell_pair_counting", "relion", "once"),
    ("noise_shell_count", "relion", "summed"),
    ("initial_noise_pair_counting", "relion", "once"),
    ("nyquist_column_counting", "relion", "once"),
    ("firstiter_cc_support", "relion", "gaussian"),
]
NON_DEFAULT = pytest.mark.parametrize("name,value", [(name, value) for name, _, value in OPTIONS])
BASE_ARGS = ["--data_dir", "data", "--output", "out"]


def test_every_default_is_relions_rule():
    options = stand_in.options().consistency
    assert {name: getattr(options, name) for name, _, _ in OPTIONS} == {name: default for name, default, _ in OPTIONS}
    assert options.non_default() == {}
    parsed = command_options.resolve_consistency_options(command_options.parse_refinement_args(BASE_ARGS))
    assert parsed == RelionConsistencyOptions()


@NON_DEFAULT
def test_option_is_named_validated_and_parsed(name, value):
    assert RelionConsistencyOptions(**{name: value}).non_default() == {name: value}
    with pytest.raises(ValueError, match=name):
        RelionConsistencyOptions(**{name: "no-such-rule"})
    args = command_options.parse_refinement_args([*BASE_ARGS, f"--{name}", value])
    assert getattr(command_options.resolve_consistency_options(args), name) == value
    with pytest.raises(SystemExit):
        command_options.parse_refinement_args([*BASE_ARGS, f"--{name}", "no-such-rule"])


@NON_DEFAULT
def test_command_line_refuses_the_option_on_relion_seeded_or_replayed_state(name, value):
    for flags in (
        ["--relion_init_dir", "relion"],
        ["--perturb_replay_relion_dir", "relion"],
        ["--frozen-boundary-dir", "frozen"],
        ["--init_noise_from_npz", "noise.npz"],
        ["--init_relion_iteration", "3"],
    ):
        args = command_options.parse_refinement_args([*BASE_ARGS, f"--{name}", value, *flags])
        with pytest.raises(SystemExit, match="RELION-seeded or replayed state"):
            command_options.resolve_consistency_options(args)
        # The default keeps every such route.
        command_options.resolve_consistency_options(command_options.parse_refinement_args([*BASE_ARGS, *flags]))


def _stub_half(owner=None):
    half = SimpleNamespace() if owner is None else object.__new__(owner)
    for field, field_value in dict(volume_shape=(8, 8, 8), image_shape=(8, 8), voxel_size=1.0, n_units=4).items():
        object.__setattr__(half, field, field_value)
    return half


def _refine_stubs(halves, name, value, relion_replay=None, **groups):
    from relax.parity.relion_replay_source import RelionReplaySource

    options = stand_in.options(consistency=RelionConsistencyOptions(**{name: value}), **groups)
    return iteration_loop.refine_single_volume(
        halves, None, None, None, options=options, source=RelionReplaySource.for_run(relion_replay, options),
        observer=RunObserver(),
    )


@NON_DEFAULT
def test_subtomograms_refuse_the_option(name, value):
    from relax.refinement.tomo_half import TomoHalf

    with pytest.raises(NotImplementedError, match="subtomogram particles"):
        _refine_stubs([_stub_half(TomoHalf), _stub_half(TomoHalf)], name, value)


@NON_DEFAULT
def test_optics_groups_on_several_image_shapes_refuse_the_option(name, value):
    from relax.refinement.optics_shapes import MultiShapeHalf

    with pytest.raises(NotImplementedError, match="several image shapes"):
        _refine_stubs([_stub_half(MultiShapeHalf), _stub_half(MultiShapeHalf)], name, value)


@NON_DEFAULT
@pytest.mark.parametrize(
    "group",
    [
        dict(relion_replay=RelionReplay(replay_iteration_overrides=[{}])),
        dict(relion_replay=RelionReplay(final_replay_override={})),
        dict(relion_replay=RelionReplay(final_replay_reference_maps=[None, None])),
        dict(relion_replay=RelionReplay(frozen_refinement_state_fields={})),
        dict(relion_replay=RelionReplay(perturb_replay_relion_dir="relion")),
        dict(relion_replay=RelionReplay(sealed_sampling_state=object())),
        dict(relion_replay=RelionReplay(state_swap_probe={"iteration": 0})),
    ],
    ids=["iteration-overrides", "final-override", "final-references", "frozen-state", "star-replay", "sealed", "swap"],
)
def test_replayed_or_frozen_relion_state_refuses_the_option(name, value, group):
    """Replayed statistics, references and frozen sampling were computed with RELION's rules."""
    with pytest.raises(NotImplementedError, match="replayed or frozen RELION state"):
        _refine_stubs([_stub_half(), _stub_half()], name, value, **group)


@NON_DEFAULT
def test_a_continuation_must_repeat_the_option(name, value):
    def validate(extra, consistency):
        snapshot = SimpleNamespace(relion_iteration=2, n_classes=1, box_size=8, extra=extra)
        iteration_snapshot.validate_resume_snapshot(
            snapshot, init_relion_iteration=2, n_classes=1, box_size=8, options=stand_in.options(consistency=consistency),
            replays_relion_trajectory=False, starts_from_frozen_boundary=False, swaps_state=False,
        )

    chosen = RelionConsistencyOptions(**{name: value})
    validate({f"consistency_{name}": value}, chosen)
    validate({}, RelionConsistencyOptions())
    with pytest.raises(ValueError, match="RELION-consistency options"):
        validate({f"consistency_{name}": value}, RelionConsistencyOptions())
    with pytest.raises(ValueError, match="RELION-consistency options"):
        validate({}, chosen)


class _Writer:
    def __init__(self):
        self.snapshots = []

    def due(self, relion_iteration):
        return True

    def wants_unfiltered_maps(self, relion_iteration, *, n_classes):
        return False

    def __call__(self, snapshot):
        self.snapshots.append(snapshot)


@NON_DEFAULT
def test_run_files_record_the_option(monkeypatch, name, value):
    writer = _Writer()
    run_tiny_refinement(
        monkeypatch,
        max_iter=1,
        # The CC support needs the CC iteration it acts on.
        parity=dict(emulate_relion_firstiter_cc=True, relion_firstiter_ini_high_angstrom=8.0),
        consistency=RelionConsistencyOptions(**{name: value}),
        checkpoint=CheckpointOptions(writer=writer),
    )
    recorded = {key: item for key, item in writer.snapshots[0].extra.items() if key.startswith("consistency_")}
    assert recorded == {f"consistency_{name}": value}


def test_default_run_files_record_no_option(monkeypatch):
    writer = _Writer()
    run_tiny_refinement(monkeypatch, max_iter=1, checkpoint=CheckpointOptions(writer=writer))
    assert not [key for key in writer.snapshots[0].extra if key.startswith("consistency_")]


def test_initial_model_command_takes_no_consistency_option():
    """VDAM keeps RELION's rules: it cannot be asked for a corrected one."""
    from relax.commands import initial_model

    actions = initial_model.make_parser()._option_string_actions
    assert not [name for name, _, _ in OPTIONS if f"--{name}" in actions]
