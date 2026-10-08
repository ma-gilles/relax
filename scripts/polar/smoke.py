"""Small float32 E-step check with a staged input file on a Slurm GPU."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import recovar

import relax

# The E-step oracle lives with the tests (tests/oracles); the script is run from the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from oracles.e_step import compute_probability_from_residual_normal_squared_one_image  # noqa: E402


def main(input_path: Path, output_path: Path) -> None:
    devices = jax.devices("gpu")
    assert len(devices) == 1, devices
    residuals = np.asarray(json.loads(input_path.read_text())["residuals"], dtype=np.float32)
    probs = np.asarray(compute_probability_from_residual_normal_squared_one_image(jnp.asarray(residuals)))
    expected = np.exp(-0.5 * (residuals - residuals.min(axis=1, keepdims=True)))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, expected, rtol=1e-6, atol=1e-7)
    result = {
        "device": str(devices[0]),
        "dtype": str(probs.dtype),
        "probabilities": probs.tolist(),
        "relax": str(Path(relax.__file__).resolve()),
        "recovar": str(Path(recovar.__file__).resolve()),
        "jax": str(Path(jax.__file__).resolve()),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
