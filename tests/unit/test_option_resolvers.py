"""The command's option-group resolvers: each record's fields come from the flags they name
(relax.refinement.command_options)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.tiny_main import controller_inputs

from relax.refinement import command_options
from relax.refinement.refinement_options import (
    AdaptiveOptions,
    HalfOverlapOptions,
    LocalSearchOptions,
    RefinementBatching,
)

pytestmark = pytest.mark.unit
LOG = logging.getLogger(__name__)


def _args(*arguments):
    return command_options.parse_refinement_args(["--data_dir", "d", "--output", "o", *arguments])


def test_adaptive_options_parse_the_oracle_lists_and_keep_the_sampling_flags():
    args = _args("--relion_current_sizes", "12,16", "--relion_healpix_orders", "2,3", "--adaptive_oversampling", "0",
                 "--coarse_engine", "gemm_dense")
    args.max_significants = 7
    assert command_options.resolve_adaptive_options(args, log=LOG) == AdaptiveOptions(
        adaptive_oversampling=0, max_significants=7, coarse_engine="gemm_dense",
        relion_current_sizes=[12, 16], relion_healpix_orders=[2, 3],
    )
    defaults = command_options.resolve_adaptive_options(_args(), log=LOG)
    assert defaults.relion_current_sizes is None and defaults.relion_healpix_orders is None


def test_batching_overlap_and_local_search_come_from_their_flags():
    args = _args("--image_batch_size", "64", "--rotation_block_size", "128", "--overlap_halves",
                 "--auto_local_healpix_order", "3", "--sigma_ang", "2.5", "--local_search_profile", "on")
    assert command_options.resolve_batching(args) == RefinementBatching(image_batch_size=64, rotation_block_size=128)
    assert command_options.resolve_overlap(args) == HalfOverlapOptions(overlap_halves=True)
    assert command_options.resolve_local_search(args) == LocalSearchOptions(
        auto_local_healpix_order=3, sigma_ang_deg=2.5, local_search_profile_mode="on",
    )


def test_the_command_hands_the_resolved_groups_to_the_controller(monkeypatch, tmp_path):
    options = controller_inputs(monkeypatch, tmp_path, "refine", "--relion_current_sizes", "12,14,16",
                                "--image_batch_size", "8")["options"]
    assert options.adaptive.relion_current_sizes == [12, 14, 16]
    assert options.batching.image_batch_size == 8


def _schedule(args, frozen_boundary=None, continued_iterations=None, sigma_offset=None):
    return command_options.resolve_schedule(
        args, initial_sampling=SimpleNamespace(coarse_order=2, max_order=5), init_current_size=24,
        ini_high_angstrom=30.0, init_data_vs_prior="dvp", particle_diameter_ang=180.0,
        relion_init_sigma_offset_angstrom=sigma_offset, frozen_boundary=frozen_boundary,
        continued_iterations=continued_iterations,
    )


def test_schedule_takes_the_sampling_and_the_first_iteration_from_the_flags():
    args = _args("--max_iter", "6", "--offset_range", "3", "--offset_step", "1", "--offset_sigma_angstrom", "7",
                 "--init_relion_iteration", "2", "--skip_final_iteration")
    schedule = _schedule(args)
    assert (schedule.max_iter, schedule.init_relion_iteration, schedule.skip_final_iteration) == (6, 2, True)
    assert (schedule.init_healpix_order, schedule.max_healpix_order) == (2, 5)
    assert (schedule.init_translation_range, schedule.init_translation_step) == (3.0, 1.0)
    assert (schedule.init_translation_sigma_angstrom, schedule.init_relion_incr_size) == (7.0, 10)
    assert (schedule.init_current_size, schedule.ini_high_angstrom, schedule.init_data_vs_prior) == (24, 30.0, "dvp")
    assert schedule.particle_diameter_ang == 180.0 and schedule.init_fsc is None
    assert _schedule(args, sigma_offset=1.5).init_translation_sigma_angstrom == 1.5
    # --continue: --max_iter counts the whole run; the run files' iteration is the first one.
    continued = _schedule(args, continued_iterations=4)
    assert (continued.max_iter, continued.init_relion_iteration) == (2, 4)


def test_a_frozen_boundary_owns_its_schedule_fields():
    boundary = SimpleNamespace(fsc="fsc", ave_pmax=0.5, has_high_fsc_at_limit=True, relion_incr_size=6,
                               translation_sigma_angstrom_per_half=(1.0, 2.0))
    schedule = _schedule(_args("--max_iter", "3"), frozen_boundary=boundary, sigma_offset=1.5)
    assert (schedule.init_fsc, schedule.init_ave_Pmax, schedule.init_has_high_fsc_at_limit) == ("fsc", 0.5, True)
    assert (schedule.init_relion_incr_size, schedule.init_translation_sigma_angstrom) == (6, (1.0, 2.0))


def test_class_seeds_are_drawn_only_for_a_fresh_class3d_run_from_one_reference():
    trial_order = np.arange(12)
    seeded = command_options.resolve_k_class(_args("--n_classes", "3", "--init_volume", "ref.mrc", "--seed", "5"),
                                             trial_order=trial_order, resumed=False)
    assert seeded.n_classes == 3 and seeded.first_iteration_seed_classes.shape == (12,)
    assert set(seeded.first_iteration_seed_classes.tolist()) <= {0, 1, 2}
    for arguments, resumed in ((("--n_classes", "3", "--init_volume", "ref.mrc"), True), (("--n_classes", "3"), False),
                               (("--init_volume", "ref.mrc"), False)):
        assert command_options.resolve_k_class(_args(*arguments), trial_order=trial_order,
                                               resumed=resumed).first_iteration_seed_classes is None


def test_a_frozen_boundary_replays_its_state_and_its_sealed_sampling_on_the_fixed_arm():
    assert command_options.resolve_local_search(_args("--stop_after_local_search")).stops_after_local_search
    assert command_options.frozen_boundary_replay(None) == {}
    boundary = SimpleNamespace(fixed_diagnostic_arm=True, sampling_state="sampling", schema="s", completed_relion_iteration=3,
                               consumer_relion_iteration=4, source_sha256="h", source_roles={}, runtime_config={},
                               map_lineage=[], refinement_state_fields={"healpix_order": 2})
    sealed = command_options.frozen_boundary_replay(boundary)
    assert sealed["assert_scoring_state_unchanged"] and sealed["sealed_sampling_state"] == "sampling"
    assert sealed["sealed_scoring_context"]["consumer_relion_iteration"] == 4
    assert sealed["frozen_refinement_state_fields"] == {"healpix_order": 2}
    boundary.fixed_diagnostic_arm = False
    unsealed = command_options.frozen_boundary_replay(boundary)
    assert unsealed["assert_scoring_state_unchanged"] and unsealed["sealed_scoring_context"] is None
    assert unsealed["sealed_sampling_state"] is None


def test_the_intermediates_observer_comes_from_its_flags(tmp_path):
    from relax.diagnostics.observers import IntermediatesObserver, command_observer

    assert command_observer(_args()) is None
    observer = command_observer(_args("--save_intermediates_dir", str(tmp_path / "dump"), "--save_intermediates_skip_unregularized"))
    assert isinstance(observer, IntermediatesObserver) and (tmp_path / "dump").is_dir()
    assert observer.directory == str(tmp_path / "dump") and observer.skip_unregularized
    assert not observer.wants_unfiltered_maps(1)
    assert observer.keeps_rotation_posteriors and observer.collects_local_search_profiles
