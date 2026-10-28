"""The PC schedule: PCs join the model at their stage as small random loadings (Config.pc_stages)."""

import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu

from relax.commands.ppca_initial_model import load_training
from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.initialization import birth_loadings, support_mask
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import Moments, empty_moments

pytestmark = pytest.mark.unit

BOX = 8
HALF = (BOX, BOX, BOX // 2 + 1)
N_FREQ = BOX * BOX * (BOX // 2 + 1)


def _write_star(root, count):
    """A RELION particle STAR of ``count`` noise images of ``BOX`` pixels with one optics group."""
    import mrcfile

    from scripts.lib.ppca_training_star import write_training_star

    root.mkdir(parents=True, exist_ok=True)
    images = np.random.default_rng(19).normal(size=(count, BOX, BOX)).astype(np.float32)
    with mrcfile.new(root / "particles.mrcs", overwrite=True) as stack:
        stack.set_data(images)
        stack.voxel_size = 2
    ctf = np.tile([15000, 15000, 0, 300, 2.7, 0.07, 0], (count, 1))
    ctf[:, 0] += 100 * np.arange(count)
    return write_training_star(root / "particles.star", "particles.mrcs", ctf, voxel_size=2, box=BOX)


def _mean_state(q, seed=3):
    """A state whose mean is a smooth blob and whose q loading columns hold stand-in values (to be zeroed or born)."""
    z, y, x = np.indices((BOX,) * 3) - BOX // 2
    blob = np.exp(-(x**2 + y**2 + z**2) / 6.0).astype(np.float32)
    mean = np.asarray(ftu.get_dft3_real(blob[None]), np.complex64).reshape(1, -1).T
    loadings = np.random.default_rng(seed).normal(size=(N_FREQ, q)).astype(np.complex64)
    theta = jnp.asarray(np.concatenate([mean, loadings], axis=1))
    moments = empty_moments(theta)
    moments = Moments(moments.first, jnp.full_like(moments.second, 0.5), moments.initialized)
    return State(theta, moments, jnp.ones(BOX // 2 + 1, jnp.float32), 0, np.arange(4), {}, 2.0, 2, {})


def _activate(state, config, iteration):
    return iteration_loop._activate_pcs(
        state, config, iteration, volume_shape=(BOX,) * 3, diameter_px=6.0, radius=3, mask=support_mask(BOX, 6.0)
    )


@pytest.mark.parametrize(
    "stages, message",
    [
        ((), "pairs"),
        (((1, 0, 5),), "pairs"),
        (((2, 0), (5, 2)), "begin at iteration 1"),
        (((1, 0), (1, 2)), "increasing first iterations"),
        (((1, 0), (5, 1)), "end at q = 2"),
        (((1, 2), (5, 1)), "must not decrease"),
        (((1, 0), (5, 3)), "end at q = 2"),
        (((1, 0), (500, 2)), "after the 200 iterations"),
        (((1, 0.0), (5, 2)), "integers"),
    ],
)
def test_pc_stages_are_validated(stages, message):
    with pytest.raises(ValueError, match=message):
        Config(q=2, pc_stages=stages)


def test_pc_stages_need_the_streamed_engine_and_a_positive_scale():
    with pytest.raises(ValueError, match="streamed engine"):
        Config(q=2, pc_stages=((1, 0), (5, 2)), stream_coarse_recompute=False)
    with pytest.raises(ValueError, match="pc_birth_scale"):
        Config(pc_birth_scale=0.0)


def test_active_pcs_follows_the_stage_of_the_iteration():
    config = Config(q=3, pc_stages=((1, 0), (10, 1), (20, 3)))
    assert [config.active_pcs(i) for i in (1, 9, 10, 19, 20, 200)] == [0, 0, 1, 1, 3, 3]
    assert Config(q=3).active_pcs(1) == 3  # without a schedule every PC is active


def test_birth_sets_only_the_asked_columns_with_the_asked_size():
    """Each new loading has the mean's standard deviation times the scale before masking and band-limiting, so the
    born column is nonzero, band-limited to the radius, and the other columns are untouched."""
    state = _mean_state(q=3)
    mask = support_mask(BOX, 6.0)
    born = birth_loadings(
        state.theta,
        [2, 3],
        volume_shape=(BOX,) * 3,
        diameter_px=6.0,
        radius=3,
        mask=mask,
        scale=0.1,
        rng=np.random.default_rng(5),
    )
    assert_matches(born[:, :2], state.theta[:, :2])
    for column in (2, 3):
        assert float(jnp.linalg.norm(born[:, column])) > 0
        assert not np.allclose(np.asarray(born[:, column]), np.asarray(state.theta[:, column]))
    radii = np.asarray(ftu.get_grid_of_radial_distances_real((BOX,) * 3, rounded=False)).reshape(-1)
    assert float(jnp.max(jnp.abs(born[radii > 3, 2:]))) == 0
    # Doubling the scale doubles the loading: the field is rescaled to the target before the (linear) masking.
    twice = birth_loadings(
        state.theta,
        [2, 3],
        volume_shape=(BOX,) * 3,
        diameter_px=6.0,
        radius=3,
        mask=mask,
        scale=0.2,
        rng=np.random.default_rng(5),
    )
    assert_matches(twice[:, 2:], 2 * born[:, 2:])


def test_a_schedule_zeroes_inactive_columns_and_births_a_pc_when_it_joins():
    config = Config(q=3, pc_stages=((1, 0), (4, 1), (7, 3)), seed=11)
    state = _mean_state(q=3)
    first = _activate(state, config, 1)
    assert_matches(first.theta[:, 0], state.theta[:, 0])
    assert float(jnp.max(jnp.abs(first.theta[:, 1:]))) == 0
    assert_matches(first.moments.second, state.moments.second)  # no PC has joined
    # Iteration 4: the first PC joins and restarts the loadings' shared second moment, the mean's is kept.
    joined = _activate(first, config, 4)
    assert float(jnp.linalg.norm(joined.theta[:, 1])) > 0
    assert float(jnp.max(jnp.abs(joined.theta[:, 2:]))) == 0
    assert_matches(joined.moments.second[:, 1], jnp.ones(N_FREQ))
    assert_matches(joined.moments.second[:, 0], first.moments.second[:, 0])
    # Iteration 7: two more join; the first PC and the second moment stay as they are.
    more = _activate(joined, config, 7)
    assert_matches(more.theta[:, 1], joined.theta[:, 1])
    assert all(float(jnp.linalg.norm(more.theta[:, c])) > 0 for c in (2, 3))
    assert_matches(more.moments.second, joined.moments.second)
    # A stage without a change leaves the state alone.
    assert _activate(more, config, 8) is more


def test_a_birth_depends_on_the_seed_and_the_iteration_only():
    config = Config(q=1, pc_stages=((1, 0), (4, 1)), seed=11)
    state = _activate(_mean_state(q=1), config, 1)
    again = _activate(state, config, 4)
    assert_matches(_activate(state, config, 4).theta, again.theta)  # resumed runs draw the same loading
    other_seed = _activate(state, dataclasses.replace(config, seed=12), 4)
    assert not np.allclose(np.asarray(other_seed.theta[:, 1]), np.asarray(again.theta[:, 1]))


def test_without_a_schedule_the_state_is_untouched():
    state = _mean_state(q=2)
    assert _activate(state, Config(q=2), 1) is state
    assert _activate(state, Config(q=2), 50) is state


def test_a_mean_only_expectation_is_the_mean_block_of_the_zero_loading_expectation(tmp_path):
    """With W = 0 the latents decouple from the pose scores, so the mean's statistics of a q = 1 model without a
    loading equal those of the mean-only model."""
    count = 6
    data, _, _ = load_training(_write_star(tmp_path, count))
    mean = np.random.default_rng(3).normal(size=(N_FREQ, 1)).astype(np.complex64) * 1e-2
    theta0 = jnp.asarray(mean)
    theta1 = jnp.asarray(np.concatenate([mean, np.zeros((N_FREQ, 1), np.complex64)], axis=1))
    options = dict(oversampling=0, image_batch_size=3, rotation_block_size=64, stages=((1, 2, 0),))

    def run(theta, config):
        state = State(
            theta,
            empty_moments(theta),
            jnp.ones(BOX // 2 + 1, jnp.float32),
            0,
            np.arange(count),
            np.random.default_rng(1).bit_generator.state,
            2.0,
            2,
            {},
        )
        return iteration_loop.expectation(data, state, config, np.arange(count), 1)

    mean_only = run(theta0, Config(q=1, pc_stages=((1, 0), (2, 1)), **options))
    zero_loading = run(theta1, Config(q=1, **options))
    assert np.asarray(mean_only.lhs_tri).shape[-1] == 1 and np.asarray(zero_loading.lhs_tri).shape[-1] == 3
    assert_matches(mean_only.lhs_tri[:, 0], zero_loading.lhs_tri[:, 0])
    assert_matches(mean_only.residual_gradient[:, 0], zero_loading.residual_gradient[:, 0])
    assert_matches(mean_only.log_likelihood, zero_loading.log_likelihood)


def _run_command(star, output, *extra):
    """``relax ppca_initial_model`` on ``star`` (written once: the checkpoint identity holds the stack's hash, and an
    MRC file carries its creation time) for 8 iterations of a fixed batch of 8 particles."""
    from relax.commands import ppca_initial_model

    argv = [
        "--i", str(star), "--particle-diameter", "12", "-o", str(output), "--q", "1", "--seed", "11",
        "--iterations", "8", "--stages", "[[1,2,1]]", "--stochastic-batch-size", "8", "--stochastic-all-iterations",
        "--image-batch-size", "4", "--rotation-block-size", "64", "--skip-final-embeddings", *extra,
    ]  # fmt: skip
    ppca_initial_model.main(argv)


def test_a_run_learns_the_mean_alone_until_its_pc_joins_and_resumes_to_the_same_model(monkeypatch, tmp_path):
    """The loading is zero through iteration 3 and nonzero from the iteration its stage starts (4); the log records
    the active count; a run stopped at iteration 4 and resumed ends at the model of the uninterrupted run."""
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    schedule = ("--pc-stages", "[[1,0],[4,1]]")
    full = tmp_path / "full"
    _run_command(star, full, *schedule)
    loadings = {}
    for iteration in range(1, 9):
        with np.load(full / f"checkpoint_{iteration:04d}.npz", allow_pickle=False) as saved:
            loadings[iteration] = np.asarray(saved["theta"])[:, 1]
    assert all(np.max(np.abs(loadings[i])) == 0 for i in (1, 2, 3))
    assert all(np.max(np.abs(loadings[i])) > 0 for i in range(4, 9))
    log = [json.loads(line) for line in open(full / "iterations.jsonl")]
    assert [row["active_pcs"] for row in log] == [0, 0, 0, 1, 1, 1, 1, 1]

    split = tmp_path / "split"
    _run_command(star, split, *schedule, "--stop-after", "4")
    assert not (split / "checkpoint_0005.npz").exists()
    _run_command(star, split, *schedule, "--resume", str(split / "checkpoint_0004.npz"))
    with (
        np.load(full / "checkpoint_0008.npz", allow_pickle=False) as expected,
        np.load(split / "checkpoint_0008.npz", allow_pickle=False) as resumed,
    ):
        assert_matches(np.asarray(resumed["theta"]), np.asarray(expected["theta"]))
