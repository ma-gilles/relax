"""--continue from the run files of a run on several image shapes (relax#38 phase 2), on the stand-in engine.

N numbered iterations and a continuation for one more write the run files N + 1 iterations in one go write.
"""

import mrcfile
import numpy as np
import pytest
import starfile
from helpers.float_compare import assert_matches
from helpers.fake_adaptive_engine import install_fake_adaptive_engine
from helpers.tiny_main import _run_main, _stand_in_device, write_tiny_data_dir

from relax.refinement import firstiter_cc, half_scoring

pytestmark = pytest.mark.unit

# Optics group 2 has 14-pixel images at 5.44 A (16 at 4.25 A in group 1): a larger field of view on its own grid.
SECOND_SHAPE = (14, 5.44)
# The legacy fraction-based assignment counters compare an iteration's coarse pose ids with the previous
# iteration's, which no run file carries (RELION has no such field), so a continuation restarts them; the
# loop reads them only before RELION's hidden-variable trackers are populated (convergence.check_convergence).
RESTARTED_FIELDS = {
    "relax_state_nr_iter_wo_assignment_changes", "relax_state_fraction_changed", "relax_state_changes_optimal_offsets",
}


def _run(monkeypatch, tmp_path, command, data, output, *arguments):
    """``relax <command>`` on the stand-in engine, its noise sums and Fourier weights following the noise it is
    handed: the plain stand-in returns fixed sums, so a run would not notice a noise model restored wrongly from
    the run files. Here each group's noise sum is its input noise level times a fixed factor and the weights
    scale with the mean noise, so a restored model that differs changes the next iteration's noise and maps."""
    import jax.numpy as jnp

    _stand_in_device(monkeypatch)
    install_fake_adaptive_engine(monkeypatch)
    engine = half_scoring.run_dense_k_class_em_adaptive

    def run(experiment_dataset, means, mean_variance, noise_variance, *args, **kwargs):
        result = engine(experiment_dataset, means, mean_variance, noise_variance, *args, **kwargs)
        level = jnp.mean(jnp.atleast_2d(jnp.asarray(noise_variance, jnp.float32)), axis=1)

        def follow(stats):
            per_group = jnp.ndim(stats.wsum_sigma2_noise) == 2
            factor = (level * jnp.asarray(stats.sumw))[:, None] if per_group else level[0] * stats.sumw
            return stats._replace(wsum_sigma2_noise=0.9 * factor * jnp.ones_like(stats.wsum_sigma2_noise))

        return result._replace(
            Ft_ctf=result.Ft_ctf / jnp.mean(level),
            noise_stats=tuple(follow(stats) for stats in result.noise_stats),
            aggregate_noise_stats=follow(result.aggregate_noise_stats),
        )

    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", run)
    monkeypatch.setattr(firstiter_cc, "run_dense_k_class_em_adaptive", run)
    _run_main(monkeypatch, command, data, tmp_path / output, [str(argument) for argument in arguments])
    return tmp_path / output


def _assert_same_star(path_a, path_b):
    a, b = starfile.read(path_a, always_dict=True), starfile.read(path_b, always_dict=True)
    assert a.keys() == b.keys(), path_a.name
    for block, table in a.items():
        other = b[block]
        if not hasattr(table, "columns"):
            for key, value in table.items():
                if key in RESTARTED_FIELDS or key == "rlnOutputRootName":
                    continue
                if isinstance(value, float):
                    assert_matches(other[key], value, rtol=1e-6, err_msg=f"{path_a.name} {block} {key}")
                else:
                    assert other[key] == value, (path_a.name, block, key)
            continue
        assert list(table.columns) == list(other.columns), (path_a.name, block)
        for column in table.columns:
            want, got = table[column].to_numpy(), other[column].to_numpy()
            if want.dtype.kind == "f":
                assert_matches(got, want, rtol=1e-6, err_msg=f"{path_a.name} {block} {column}")
            else:
                np.testing.assert_array_equal(got, want, err_msg=f"{path_a.name} {block} {column}")


@pytest.mark.parametrize(
    ("command", "arguments", "n_classes"),
    [("refine", (), 1), ("class3d", ("--n_classes", "2"), 2)],
    ids=["refine3d", "class3d-k2"],
)
def test_continuing_a_several_shape_run_writes_what_the_uninterrupted_run_writes(
    monkeypatch, tmp_path, command, arguments, n_classes
):
    data = write_tiny_data_dir(tmp_path / "data", n_images=16, n_classes=n_classes, second_shape=SECOND_SHAPE)
    whole = _run(monkeypatch, tmp_path, command, data, "whole", *arguments, "--max_iter", "3")
    first = _run(monkeypatch, tmp_path, command, data, "first", *arguments, "--max_iter", "2")
    continued = _run(
        monkeypatch, tmp_path, command, data, "continued", *arguments, "--max_iter", "3",
        "--continue", first / "run_it002_optimiser.star",
    )

    files = sorted(path.name for path in whole.glob("run_it003_*"))
    assert files and files == sorted(path.name for path in continued.glob("run_it003_*"))
    data_star = starfile.read(whole / "run_it003_data.star", always_dict=True)
    assert sorted(data_star["optics"]["rlnImageSize"].astype(int)) == [14, 16]
    for name in files:
        if name.endswith(".mrc"):
            assert_matches(mrcfile.read(continued / name), mrcfile.read(whole / name), rtol=1e-6, err_msg=name)
        else:
            _assert_same_star(whole / name, continued / name)


def test_continuing_class3d_before_its_seed_iteration_warns(monkeypatch, tmp_path, caplog):
    """A Class3D start from one map with --firstiter_cc seeds its classes in iteration 2; continuing from iteration 1
    seeds none (RELION's --continue neither), and the run says so."""
    import logging

    data = write_tiny_data_dir(tmp_path / "data", n_images=16, n_classes=2)
    arguments = ("--n_classes", "2", "--init_volume", data / "reference_init_relion.mrc")
    first = _run(monkeypatch, tmp_path, "class3d", data, "first", *arguments, "--max_iter", "1")
    with caplog.at_level(logging.WARNING, logger="relax.refinement.full_refinement"):
        _run(
            monkeypatch, tmp_path, "class3d", data, "continued", *arguments, "--max_iter", "2",
            "--continue", first / "run_it001_optimiser.star",
        )
    assert any("does not seed random classes" in record.getMessage() for record in caplog.records)
