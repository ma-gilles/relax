"""The CC iteration of a one-reference Class3D start copies class 1's model to every class, once for both halves."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from relax.refinement.maximization import ClassMaximization, copy_first_class_to_every_class
from relax.refinement.reference_state import class_mixture_from_weights
from relax.sampling.orientation_priors import DirectionPrior


def test_both_halves_share_one_copy_of_the_first_class():
    stack = jnp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4) + 1j)
    reference_model = SimpleNamespace(maps=[stack, stack], tau2=np.arange(6.0).reshape(3, 2), tau2_per_half=None)
    priors = [DirectionPrior(np.arange(6.0).reshape(3, 2), 1), DirectionPrior(None, None)]
    mstep = ClassMaximization(
        Ft_y_combined=None, Ft_ctf_combined=None, previous_means=[None, None],
        tau2_shells=np.arange(6.0).reshape(3, 2), data_vs_prior=np.arange(6.0).reshape(3, 2) + 1,
        tau2_update_details={"prior_shells": np.arange(6.0).reshape(3, 2), "absent": None},
    )

    copied, mixture = copy_first_class_to_every_class(
        reference_model, priors, mstep, class_mixture_from_weights(np.array([0.6, 0.3, 0.1])), n_classes=3,
    )

    assert reference_model.maps[0] is reference_model.maps[1]
    np.testing.assert_array_equal(np.asarray(reference_model.maps[0]), np.broadcast_to(np.asarray(stack)[:1], (3, 4)))
    np.testing.assert_array_equal(np.asarray(stack), np.arange(12, dtype=np.float32).reshape(3, 4) + 1j)
    np.testing.assert_array_equal(reference_model.tau2, [[0.0, 1.0]] * 3)
    np.testing.assert_array_equal(copied.data_vs_prior, [[1.0, 2.0]] * 3)
    np.testing.assert_array_equal(priors[0].values, [[0.0, 1.0]] * 3)
    assert priors[1].values is None
    assert copied.tau2_update_details["absent"] is None
    np.testing.assert_allclose(mixture.weights, [0.2, 0.2, 0.2])
