from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils
from recovar.utils.helpers import write_relion_mrc

from relax.parity.relion_replay import replay_class_relion_references, replay_k1_relion_references
from relax.refinement.refinement_options import KClassOptions, RefinementOptions


def _probe(iteration):
    return {"iteration": iteration, "replay_relion_references": True}


def test_references_stay_the_models_own_off_the_probe_target(tmp_path):
    original = [object(), object()]
    model = SimpleNamespace(maps=original)
    for probe in (None, _probe(3), {"iteration": 4}):
        assert replay_k1_relion_references(
            model, RefinementOptions(), probe=probe, iteration=4, replay_dir=tmp_path, replay_prefix="run", volume_shape=(4, 4, 4),
        ) is original


def _real_from_ft(flat, shape):
    return np.real(np.asarray(fourier_transform_utils.get_idft3(np.asarray(flat).reshape(shape))))


def test_state_swap_force_loads_shared_kclass_maps(tmp_path):
    shape = (4, 4, 4)
    class_maps = []
    for class_number in range(1, 5):
        volume = np.full(shape, np.float32(class_number) / np.float32(10.0), dtype=np.float32)
        class_maps.append(volume)
        write_relion_mrc(
            tmp_path / f"run_it002_class{class_number:03d}.mrc",
            volume,
            voxel_size=1.0,
        )

    original = [
        jnp.zeros((4, np.prod(shape)), dtype=jnp.complex64),
        jnp.ones((4, np.prod(shape)), dtype=jnp.complex64),
    ]
    replayed = replay_class_relion_references(
        SimpleNamespace(maps=original), RefinementOptions(k_class=KClassOptions(n_classes=4)), probe=_probe(2),
        iteration=2, replay_dir=tmp_path, replay_prefix="run", volume_shape=shape,
    )

    assert replayed is not original
    assert replayed[0].shape == (4, np.prod(shape))
    assert replayed[1].shape == (4, np.prod(shape))
    for half_idx in range(2):
        for class_idx, expected in enumerate(class_maps):
            np.testing.assert_allclose(
                _real_from_ft(replayed[half_idx][class_idx], shape),
                expected,
                rtol=0,
                atol=5e-6,
            )


def test_state_swap_force_replays_target_references_without_environment(tmp_path):
    shape = (4, 4, 4)
    half1 = np.full(shape, np.float32(0.25), dtype=np.float32)
    half2 = np.full(shape, np.float32(0.75), dtype=np.float32)
    write_relion_mrc(tmp_path / "run_it004_half1_class001.mrc", half1, voxel_size=1.0)
    write_relion_mrc(tmp_path / "run_it004_half2_class001.mrc", half2, voxel_size=1.0)
    original = [
        jnp.zeros(np.prod(shape), dtype=jnp.complex64),
        jnp.ones(np.prod(shape), dtype=jnp.complex64),
    ]

    replayed = replay_k1_relion_references(
        SimpleNamespace(maps=original), RefinementOptions(), probe=_probe(4), iteration=4, replay_dir=tmp_path, replay_prefix="run", volume_shape=shape,
    )

    np.testing.assert_allclose(_real_from_ft(replayed[0], shape), half1, rtol=0, atol=5e-6)
    np.testing.assert_allclose(_real_from_ft(replayed[1], shape), half2, rtol=0, atol=5e-6)
