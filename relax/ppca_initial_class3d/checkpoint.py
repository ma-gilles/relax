"""Atomic mixture checkpoints, deliberately distinct from single-PPCA state."""

import dataclasses
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from relax.ppca_initial_class3d.auto_sampling import SamplingState
from relax.ppca_initial_class3d.membership import FIELDS as MEMBERSHIP_FIELDS
from relax.ppca_initial_class3d.membership import POSE_FIELDS as MEMBERSHIP_POSE_FIELDS
from relax.ppca_initial_class3d.membership import SCHEMA as MEMBERSHIP_SCHEMA
from relax.ppca_initial_class3d.membership import Membership
from relax.ppca_initial_class3d.state import State, validate_state
from relax.ppca_initial_model.checkpoint import RUNTIME_CONFIG_FIELDS, canonical
from relax.ppca_initial_model.update import Moments

SCHEMA = "relax-ppca-initial-class3d-v1"


# How the E-step approximates the pose integral (full grid or RELION's two passes, their significance rule) and
# how a minibatch is drawn change later updates, not the model a checkpoint holds: a run resumes across them, as
# across the single-model runtime fields.
APPROXIMATION_CONFIG_FIELDS = ("oversampling", "oversampling_start", "max_significant", "target_mass", "tomogram_batches",
                               # the search policies of local_search.py and auto_sampling.py
                               "local_search_start", "local_search_sigma_deg", "auto_sampling", "accuracy_interval",
                               "accuracy_particles", "local_order", "max_order")


# The planned length of a run is not part of its model either: a finished run resumes with a larger
# --iterations to continue (RELION's --continue with more iterations). The stage schedule stays part of the
# identity, so an extended run repeats the recorded --stages (run.json) instead of the default schedule, which
# default_stages would rescale to the new length.
EXTENSION_CONFIG_FIELDS = ("iterations",)


def _configuration(config):
    excluded = RUNTIME_CONFIG_FIELDS + APPROXIMATION_CONFIG_FIELDS + EXTENSION_CONFIG_FIELDS
    return {key: value for key, value in canonical(config).items() if key not in excluded}


def _inputs(identity):
    return {key: value for key, value in canonical(identity).items() if key != "source"}


def save(path, state, config, identity):
    validate_state(state, config)
    path = Path(path)
    metadata = {
        "schema": SCHEMA, "config": dataclasses.asdict(config), "identity": identity,
        "iteration": state.iteration, "rng_state": state.rng_state,
        "offset_variance": state.offset_variance, "radius": state.radius,
        "initialization": state.initialization, "direction_order": state.direction_order,
        "sampling": None if state.sampling is None else state.sampling.to_json(),
    }
    arrays = {
        "theta": np.asarray(state.theta), "noise": np.asarray(state.noise),
        "class_prior": np.asarray(state.class_prior), "order": state.order,
        "first": np.stack([np.asarray(m.first) for m in state.moments]),
        "second": np.stack([np.asarray(m.second) for m in state.moments]),
        "initialized": np.stack([np.asarray(m.initialized) for m in state.moments]),
        "direction_prior": np.asarray([] if state.direction_prior is None else state.direction_prior),
        "metadata": json.dumps(metadata),
    }
    if state.membership is not None:
        metadata["membership_schema"] = MEMBERSHIP_SCHEMA
        arrays["metadata"] = json.dumps(metadata)
        arrays.update({f"membership_{name}": value
                       for name, value in state.membership.arrays(state.iteration).items()})
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if state.membership is not None:
        state.membership.export(path, state.iteration)


def load(path, config, identity):
    with np.load(path, allow_pickle=False) as a:
        meta = json.loads(str(a["metadata"]))
        if (meta["schema"] != SCHEMA
                or _configuration(meta["config"]) != _configuration(dataclasses.asdict(config))
                or _inputs(meta["identity"]) != _inputs(identity)):
            raise ValueError("Mixture checkpoint input/configuration identity mismatch")
        moments = tuple(
            Moments(jnp.asarray(first), jnp.asarray(second), jnp.asarray(initialized))
            for first, second, initialized in zip(a["first"], a["second"], a["initialized"], strict=True)
        )
        state = State(
            jnp.asarray(a["theta"]), moments, jnp.asarray(a["noise"]), a["class_prior"].copy(),
            meta["iteration"], a["order"].copy(), meta["rng_state"], meta["offset_variance"],
            meta["radius"], meta["initialization"],
            a["direction_prior"].copy() if a["direction_prior"].size else None, meta["direction_order"],
        )
        if meta.get("sampling") is not None:
            state.sampling = SamplingState.from_json(meta["sampling"])
        # Additive v1 extension: old checkpoints have no historical membership.
        # Partial/corrupt new records must not silently become "unvisited".
        has_membership = any(name.startswith("membership_") for name in a.files)
        if "membership_schema" in meta or has_membership:
            if meta.get("membership_schema") != MEMBERSHIP_SCHEMA:
                raise ValueError("Unknown or missing checkpoint membership schema")
            if any(f"membership_{name}" not in a for name in MEMBERSHIP_FIELDS):
                raise ValueError("Incomplete checkpoint membership")
            # Pose memory is additive to the v1 record: older checkpoints hold no poses (unknown rows).
            poses = {name: a[f"membership_{name}"].copy() for name in MEMBERSHIP_POSE_FIELDS
                     if f"membership_{name}" in a}
            if poses and len(poses) != len(MEMBERSHIP_POSE_FIELDS):
                raise ValueError("Incomplete checkpoint membership poses")
            state.membership = Membership(*(a[f"membership_{name}"].copy() for name in MEMBERSHIP_FIELDS), **poses)
    validate_state(state, config)
    return state
