"""Export the three training-only PPCA initial fields for a K3 VDAM override.

See docs/math/vdam_ppca_algorithm.md, section 14, for the shared-seed boundary. The maps are written through
relax.io.map_io.write_map: the model is in relax's internal frame, and the files are in RELION's convention (a
STAR-path PPCA mean is positive), as every relax map.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from recovar.core import fourier_transform_utils as ftu

from relax.io.map_io import load_relax_map, write_map
from relax.ppca_initial_model.checkpoint import file_hash
from scripts.lib.ppca_training_star import require_trained_on, training_star_geometry


def export(checkpoint, training_star, output):
    checkpoint = Path(checkpoint).resolve()
    training_star = Path(training_star).resolve()
    output = Path(output).resolve()
    geometry = training_star_geometry(training_star)
    with np.load(checkpoint, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"]))
        theta = saved["theta"].copy()
    require_trained_on(meta, training_star)
    if meta["iteration"] != 0:
        raise ValueError("Seed export requires checkpoint 0000")
    n = geometry["box"]
    if theta.shape != (n * n * (n // 2 + 1), 3) or theta.dtype != np.complex64:
        raise ValueError("Expected a q2 complex64 half-Fourier model")
    mu, w1, w2 = np.asarray(
        ftu.get_idft3_real(theta.T.reshape(3, n, n, n // 2 + 1), (n, n, n)), np.float32
    )
    a = np.float32(np.sqrt(6.0) / 2)
    b = np.float32(np.sqrt(18.0) / 6)
    fields = np.stack((mu + a * w1 + b * w2, mu - a * w1 + b * w2, mu - np.float32(2) * b * w2))
    if not np.isfinite(fields).all():
        raise ValueError("Nonfinite seed field")
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for index, field in enumerate(fields, 1):
        path = output / f"seed{index}.mrc"
        write_map(path, field, voxel_size=geometry["voxel_size"])
        restored = load_relax_map(path)
        np.testing.assert_allclose(restored, field, rtol=0, atol=0)
        paths.append(path)
    receipt = {
        "checkpoint_sha256": file_hash(checkpoint),
        "training_star_sha256": file_hash(training_star),
        "source_precision": "complex64/float32",
        "seed_maps": {path.name: file_hash(path) for path in paths},
        "vdam_override": ",".join(map(str, paths)),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("training_star", help="The particle STAR the PPCA run trained on (--i)")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.training_star, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
