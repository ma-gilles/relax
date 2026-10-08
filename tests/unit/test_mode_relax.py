"""``--mode relax``: one named bundle of the RELION-consistency options, resolved once by the command.

``--mode relion`` (default) leaves every option as given. ``--mode relax`` sets the options of
``RELAX_MODE_CONSISTENCY``; an explicit option overrides it and is refused as before where its route
does not honour it, an option the mode would set is skipped there with one log line, and the mode is
refused where no option is available. See ``docs/math/relion_consistency_options.md``.
"""

import logging
from types import SimpleNamespace

import pytest
from helpers.run_options import stand_in

from relax.refinement import command_options, iteration_snapshot, refinement_options
from relax.refinement.refinement_options import (
    RELAX_MODE_CONSISTENCY,
    KClassOptions,
    RelionConsistencyOptions,
    relax_mode_consistency,
    require_consistency_route,
)

pytestmark = pytest.mark.unit

BASE_ARGS = ["--data_dir", "data", "--output", "out"]
RELAX = [*BASE_ARGS, "--mode", "relax"]
ALL = RelionConsistencyOptions(**RELAX_MODE_CONSISTENCY)


def _resolve(flags, **route):
    return command_options.resolve_consistency_options(command_options.parse_refinement_args(flags), **route)


def _without(*names):
    return RelionConsistencyOptions(**{name: value for name, value in RELAX_MODE_CONSISTENCY.items() if name not in names})


def test_the_bundle_sets_the_consistent_value_of_every_option():
    choices = refinement_options._CONSISTENCY_CHOICES
    assert set(RELAX_MODE_CONSISTENCY) == set(choices) == set(RelionConsistencyOptions.__dataclass_fields__)
    assert all(RELAX_MODE_CONSISTENCY[name] == values[1] for name, values in choices.items())


def test_mode_relion_is_the_default_and_changes_no_option(caplog):
    with caplog.at_level(logging.INFO):
        assert command_options.parse_refinement_args(BASE_ARGS).mode == "relion"
        assert _resolve(BASE_ARGS) == _resolve([*BASE_ARGS, "--mode", "relion"]) == RelionConsistencyOptions()
        explicit = _resolve([*BASE_ARGS, "--noise_shell_count", "summed"])
    assert explicit == RelionConsistencyOptions(noise_shell_count="summed")
    assert not caplog.records


def test_mode_relax_sets_every_option_on_a_k1_run(caplog):
    with caplog.at_level(logging.INFO):
        assert _resolve(RELAX) == ALL
    assert [record.getMessage() for record in caplog.records] == [
        f"--mode relax: RELION-consistency options {ALL.non_default()}"
    ]


def test_an_explicit_option_overrides_the_mode():
    assert _resolve([*RELAX, "--gridding_kernel", "radial"]) == _without("gridding_kernel")
    assert _resolve([*RELAX, "--noise_shell_count", "relion", "--shell_pair_counting", "once"]) == _without(
        "noise_shell_count"
    )


@pytest.mark.parametrize(
    "flags,route,skipped",
    [
        (["--n_classes", "2"], {}, ("gridding_kernel",)),
        (["--no-firstiter_cc"], {}, ("firstiter_cc_support",)),
        (["--coarse-engine", "gemm_dense"], {}, ("nyquist_column_counting", "firstiter_cc_support")),
        ([], {"dataset": SimpleNamespace(image_shape=(8, 8), premultiplied=True)}, ("nyquist_column_counting",)),
        (
            ["--n_classes", "3", "--no-firstiter_cc"],
            {"dataset": SimpleNamespace(image_shape=(8, 8), premultiplied=True)},
            ("gridding_kernel", "firstiter_cc_support", "nyquist_column_counting"),
        ),
    ],
    ids=["class3d", "no-cc-iteration", "gemm-dense", "ctf-premultiplied", "several"],
)
def test_mode_relax_skips_what_the_route_does_not_honour_and_says_why(monkeypatch, caplog, flags, route, skipped):
    from relax.relion import relion_ctf

    monkeypatch.setattr(
        relion_ctf, "dataset_has_premultiplied_ctf", lambda dataset, image_shape: dataset.premultiplied
    )
    route.setdefault("dataset", SimpleNamespace(image_shape=(8, 8), premultiplied=False))
    with caplog.at_level(logging.INFO):
        options = _resolve([*RELAX, *flags], **route)
    assert options == _without(*skipped)
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == f"--mode relax: RELION-consistency options {options.non_default()}"
    assert len(messages) == 1 + len(skipped)
    for name in skipped:
        assert sum(message.startswith(f"--mode relax: {name} keeps RELION's rule") for message in messages) == 1

    # The loop's guard accepts what the mode resolved for this route.
    args = command_options.parse_refinement_args([*RELAX, *flags])
    require_consistency_route(
        stand_in.options(
            consistency=options,
            k_class=KClassOptions(n_classes=int(args.n_classes)),
            adaptive=stand_in.adaptive(coarse_engine=args.coarse_engine),
            parity=stand_in.parity(emulate_relion_firstiter_cc=bool(args.firstiter_cc)),
        ),
        subtomograms=False,
        several_image_shapes=False,
    )


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--n_classes", "2", "--gridding_kernel", "separable"], "K=1 auto-refine only"),
        (["--no-firstiter_cc", "--firstiter_cc_support", "gaussian"], "needs --firstiter_cc"),
    ],
)
def test_an_explicit_option_still_fails_closed_under_mode_relax(flags, message):
    with pytest.raises(SystemExit, match=message):
        _resolve([*RELAX, *flags])
    with pytest.raises(SystemExit, match=message):
        command_options.require_consistency_arguments(command_options.parse_refinement_args([*RELAX, *flags]))


def test_an_explicit_option_the_route_refuses_is_not_skipped():
    """The explicit option reaches the loop's guard, which refuses it as it does without the mode."""
    options, skipped = relax_mode_consistency(
        {"nyquist_column_counting": "once"}, n_classes=1, has_cc_iteration=True, coarse_engine="gemm_dense",
        ctf_premultiplied=False,
    )
    assert options.nyquist_column_counting == "once" and skipped == {
        "firstiter_cc_support": "not implemented for the experimental gemm_dense coarse engine"
    }
    with pytest.raises(NotImplementedError, match="gemm_dense"):
        require_consistency_route(
            stand_in.options(consistency=options, adaptive=stand_in.adaptive(coarse_engine="gemm_dense")),
            subtomograms=False,
            several_image_shapes=False,
        )


@pytest.mark.parametrize(
    "flags",
    [
        ["--relion_init_dir", "relion"],
        ["--perturb_replay_relion_dir", "relion"],
        ["--frozen-boundary-dir", "frozen"],
        ["--init_noise_from_npz", "noise.npz"],
        ["--init_relion_iteration", "3"],
    ],
    ids=["init-dir", "star-replay", "frozen-boundary", "loaded-noise", "relion-iteration"],
)
def test_mode_relax_is_refused_on_relion_seeded_or_replayed_state(flags):
    args = command_options.parse_refinement_args([*RELAX, *flags])
    with pytest.raises(SystemExit, match="--mode relax: not available with RELION-seeded or replayed state"):
        command_options.require_consistency_arguments(args)
    with pytest.raises(SystemExit, match="--mode relax: not available with RELION-seeded or replayed state"):
        command_options.resolve_consistency_options(args)
    # The default mode keeps every such route.
    assert _resolve([*BASE_ARGS, *flags]) == RelionConsistencyOptions()


@pytest.mark.parametrize(
    "route,reason",
    [({"subtomograms": True}, "subtomogram particles"), ({"several_image_shapes": True}, "several image shapes")],
)
def test_mode_relax_is_refused_where_no_option_is_available(route, reason):
    with pytest.raises(SystemExit, match=f"--mode relax: .* not with .*{reason}"):
        _resolve(RELAX, **route)
    assert _resolve(BASE_ARGS, **route) == RelionConsistencyOptions()


def test_a_continuation_must_resolve_to_the_recorded_options():
    def validate(recorded, consistency):
        snapshot = SimpleNamespace(
            relion_iteration=2, n_classes=1, box_size=8,
            extra={f"consistency_{name}": value for name, value in recorded.non_default().items()},
        )
        iteration_snapshot.validate_resume_snapshot(
            snapshot, init_relion_iteration=2, n_classes=1, box_size=8, options=stand_in.options(consistency=consistency),
            replays_relion_trajectory=False, starts_from_frozen_boundary=False, swaps_state=False,
        )

    relax = _resolve(RELAX)
    validate(relax, relax)
    # The six options by name continue a --mode relax run: the files record the options, not the mode.
    validate(relax, _resolve([*BASE_ARGS, *(part for name, value in RELAX_MODE_CONSISTENCY.items() for part in (f"--{name}", value))]))
    with pytest.raises(ValueError, match="RELION-consistency options"):
        validate(relax, _resolve(BASE_ARGS))
    with pytest.raises(ValueError, match="RELION-consistency options"):
        validate(RelionConsistencyOptions(), relax)
    # A --mode relax continuation on a route that skips an option does not match files written with it.
    with pytest.raises(ValueError, match="RELION-consistency options"):
        validate(relax, _resolve([*RELAX, "--no-firstiter_cc"]))
