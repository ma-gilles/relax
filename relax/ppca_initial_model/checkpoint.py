"""Atomic, versioned numeric checkpoints; no pickled Python object state."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import Moments


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def save(path, state, config, identity):
    path = Path(path)
    metadata = {
        "schema": 1,
        "config": dataclasses.asdict(config),
        "identity": identity,
        "iteration": state.iteration,
        "rng_state": state.rng_state,
        "radius": state.radius,
        "offset_variance": state.offset_variance,
        "initialization": state.initialization,
        "direction_order": state.direction_order,
    }
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as stream:
        np.savez(
            stream,
            theta=np.asarray(state.theta),
            first=np.asarray(state.moments.first),
            second=np.asarray(state.moments.second),
            initialized=np.asarray(state.moments.initialized),
            noise=np.asarray(state.noise),
            order=np.asarray(state.order),
            direction_prior=np.asarray([] if state.direction_prior is None else state.direction_prior),
            metadata=json.dumps(metadata),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path, config, identity):
    with np.load(path, allow_pickle=False) as arrays:
        meta = json.loads(str(arrays["metadata"]))
        if (
            meta["schema"] != 1
            or meta["config"] != canonical(dataclasses.asdict(config))
            or meta["identity"] != canonical(identity)
        ):
            raise ValueError("Checkpoint input/configuration/source identity mismatch")
        if (
            arrays["theta"].dtype != np.complex64
            or arrays["first"].dtype != np.complex64
            or arrays["noise"].dtype != np.float32
        ):
            raise ValueError("Checkpoint is not a production float32 model")
        return State(
            jnp.asarray(arrays["theta"]),
            Moments(jnp.asarray(arrays["first"]), jnp.asarray(arrays["second"]), jnp.asarray(arrays["initialized"])),
            jnp.asarray(arrays["noise"]),
            meta["iteration"],
            arrays["order"].copy(),
            meta["rng_state"],
            meta["offset_variance"],
            meta["radius"],
            meta["initialization"],
            arrays["direction_prior"].copy() if arrays["direction_prior"].size else None,
            meta["direction_order"],
        )
