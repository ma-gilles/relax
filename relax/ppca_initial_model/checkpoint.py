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


# Runtime settings: they change how later updates are computed, not the model a checkpoint
# holds, so a run may resume under a different value (the controller logs the change).
RUNTIME_CONFIG_FIELDS = ("gemm_precision", "preread_images")


# Retired model settings and their only supported value: a checkpoint that recorded that value
# resumes; one written with the setting changed holds a different model and does not.
_RETIRED_CONFIG_DEFAULTS = {"contrast_estimate": False, "contrast_prior_sd": 0.3, "contrast_range": [0.5, 2.0]}


def _identity_config(config_dict):
    return {
        key: value
        for key, value in canonical(config_dict).items()
        if key not in RUNTIME_CONFIG_FIELDS and _RETIRED_CONFIG_DEFAULTS.get(key, ...) != value
    }


# Provenance: the code a checkpoint was written by. Stored in every checkpoint (and run.json) and logged
# on resume, but not part of the resume identity, so a run resumes after a code update. The inputs
# (manifest or optimisation-set hashes) and the model-defining configuration stay in the identity.
PROVENANCE_IDENTITY_FIELDS = ("source",)


def _resume_identity(identity):
    return {key: value for key, value in canonical(identity).items() if key not in PROVENANCE_IDENTITY_FIELDS}


def saved_source(path):
    """The source provenance a checkpoint was written under (None if it records none)."""
    with np.load(path, allow_pickle=False) as arrays:
        return json.loads(str(arrays["metadata"]))["identity"].get("source")


def saved_gemm_precision(path):
    """The stream GEMM precision setting a checkpoint was written under (fp32 before the setting existed)."""
    with np.load(path, allow_pickle=False) as arrays:
        return json.loads(str(arrays["metadata"]))["config"].get("gemm_precision", "fp32")


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
    arrays = {"theta": np.asarray(state.theta)}
    if config.optimizer == "vdam":
        if state.moments is None or state.sgd_momentum is not None:
            raise ValueError("VDAM checkpoint requires VDAM moments only")
        arrays.update(
            first=np.asarray(state.moments.first),
            second=np.asarray(state.moments.second),
            initialized=np.asarray(state.moments.initialized),
        )
    else:
        if state.sgd_momentum is None or state.moments is not None:
            raise ValueError("Momentum SGD checkpoint requires momentum only")
        arrays["sgd_momentum"] = np.asarray(state.sgd_momentum)
    arrays.update(
        noise=np.asarray(state.noise),
        order=np.asarray(state.order),
        direction_prior=np.asarray([] if state.direction_prior is None else state.direction_prior),
        metadata=json.dumps(metadata),
    )
    with open(temporary, "wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path, config, identity):
    with np.load(path, allow_pickle=False) as arrays:
        meta = json.loads(str(arrays["metadata"]))
        if (
            meta["schema"] != 1
            or _identity_config(meta["config"]) != _identity_config(canonical(dataclasses.asdict(config)))
            or _resume_identity(meta["identity"]) != _resume_identity(identity)
        ):
            raise ValueError("Checkpoint input/configuration/source identity mismatch")
        if arrays["theta"].dtype != np.complex64 or arrays["noise"].dtype != np.float32:
            raise ValueError("Checkpoint is not a production float32 model")
        if config.optimizer == "vdam":
            if "sgd_momentum" in arrays or arrays["first"].dtype != np.complex64:
                raise ValueError("VDAM checkpoint has incorrect optimizer state")
            moments = Moments(
                jnp.asarray(arrays["first"]), jnp.asarray(arrays["second"]), jnp.asarray(arrays["initialized"])
            )
            momentum = None
        else:
            if "first" in arrays or "sgd_momentum" not in arrays or arrays["sgd_momentum"].dtype != np.complex64:
                raise ValueError("Momentum SGD checkpoint has incorrect optimizer state")
            moments = None
            momentum = jnp.asarray(arrays["sgd_momentum"])
            if momentum.shape != arrays["theta"].shape:
                raise ValueError("Momentum SGD checkpoint momentum shape mismatch")
        return State(
            jnp.asarray(arrays["theta"]),
            moments,
            jnp.asarray(arrays["noise"]),
            meta["iteration"],
            arrays["order"].copy(),
            meta["rng_state"],
            meta["offset_variance"],
            meta["radius"],
            meta["initialization"],
            arrays["direction_prior"].copy() if arrays["direction_prior"].size else None,
            meta["direction_order"],
            momentum,
        )
