"""Convert a recovar-ppca-training-v1 manifest into a RELION particle STAR for relax ppca_initial_model --i.

The retired manifest (bare particles.mrcs, cryoDRGN ctf.pkl, neutral_poses.pkl, particle_ids.npy and their
sha256) held RECOVAR-simulator images: protein dark under RECOVAR's CTF. relax reads a particle STAR as stored in
RELION's sign (protein white), so the converter writes, under a new OUTPUT directory and never beside the input:

- ``particles.mrcs``: the source images negated (float32), row r holding the particle whose id is r;
- ``particles.star``: one optics group, the CTF of ``ctf.pkl`` (recovar ``CTFParamIndex`` columns after the pickle's
  box and pixel columns), ``rlnImageName`` in particle-id order, no angles or offsets
  (scripts/lib/ppca_training_star.py);
- ``conversion.json``: provenance (the source manifest's path and sha256, the written files' sha256, and the
  manifest's particle diameter, the ``--particle-diameter`` to train with).

The loaded STAR dataset gives exactly the negated processed images and the CTF of the manifest dataset
(tests/unit/ppca_initial_model/test_prepare_vdam_ppca_star.py). Works on a test-suite dataset's
``training/manifest.json``; a 100,000-image stack is copied in blocks (run it as a Slurm CPU job).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Run as python scripts/prepare_vdam_ppca_star.py: the repository root, not scripts/, must be on the path.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relax.ppca_initial_model.checkpoint import file_hash  # noqa: E402
from scripts.lib.ppca_training_star import relion_sign_images, write_training_star  # noqa: E402

SCHEMA = "recovar-ppca-training-v1"
# Images copied per block: 4096 box-128 images are 256 MB of float32.
_BLOCK = 4096


def read_manifest(manifest_path):
    """The manifest, its CTF rows ``(n, 9)`` (cryoDRGN layout) and the stack order of particle ids ``0..n-1``.

    Checks what the retired loader checked: schema, one recorded sha256 per input, unit contrast and multiplier,
    neutral poses, the CTF's box and pixel columns, and ids that are a permutation of ``0..n-1``.
    """
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Expected a {SCHEMA} manifest")
    inputs = [manifest[key] for key in ("particles", "ctf", "neutral_poses", "particle_ids")]
    if len(set(inputs)) != 4 or set(inputs) != set(manifest["files"]):
        raise ValueError("Every training input must have exactly one recorded identity")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or file_hash(root / name) != digest:
            raise ValueError(f"Training input identity mismatch: {name}")
    if manifest["contrast"] != 1 or manifest["image_multiplier"] != 1:
        raise ValueError("Expected unit contrast and unscaled particles")
    n = int(manifest["n_images"])
    with open(root / manifest["neutral_poses"], "rb") as stream:
        rotations, translations = pickle.load(stream)
    if not np.array_equal(rotations, np.broadcast_to(np.eye(3), (n, 3, 3))) or np.any(translations):
        raise ValueError("Training poses must be neutral")
    with open(root / manifest["ctf"], "rb") as stream:
        ctf = np.asarray(pickle.load(stream))
    if ctf.ndim != 2 or ctf.shape[0] != n or ctf.shape[1] < 9:
        raise ValueError(f"CTF array shape mismatch: {ctf.shape}")
    if not np.all(ctf[:, 0] == manifest["box"]) or not np.allclose(ctf[:, 1], manifest["voxel_size"], rtol=1e-6, atol=0):
        raise ValueError("CTF geometry mismatch")
    ids = np.load(root / manifest["particle_ids"], allow_pickle=False)
    if ids.shape != (n,) or not np.array_equal(np.sort(ids), np.arange(n)):
        raise ValueError("Particle ids must be a permutation of 0..n-1")
    return manifest, ctf, np.argsort(ids, kind="stable")


def convert(manifest_path, output):
    """Write ``particles.mrcs``, ``particles.star`` and ``conversion.json`` under the new directory ``output``."""
    import mrcfile

    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    if output == manifest_path.parent or manifest_path.parent in output.parents:
        raise ValueError("Write the converted STAR outside the manifest's directory")
    manifest, ctf, order = read_manifest(manifest_path)
    n, box = int(manifest["n_images"]), int(manifest["box"])
    output.mkdir(parents=True, exist_ok=False)
    with mrcfile.mmap(manifest_path.parent / manifest["particles"], permissive=False) as source:
        if source.data.shape != (n, box, box):
            raise ValueError(f"Stack shape {source.data.shape} does not match the manifest")
        with mrcfile.new_mmap(output / "particles.mrcs", shape=(n, box, box), mrc_mode=2) as stack:
            for start in range(0, n, _BLOCK):
                rows = order[start : start + _BLOCK]
                stack.data[start : start + len(rows)] = relion_sign_images(source.data[rows])
            stack.set_image_stack()
            stack.voxel_size = float(manifest["voxel_size"])
    write_training_star(
        output / "particles.star",
        "particles.mrcs",
        ctf[order, 2:],
        voxel_size=float(manifest["voxel_size"]),
        box=box,
        header=f"relax PPCA training particles converted from {SCHEMA}",
    )
    record = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_hash(manifest_path),
        "sign": "RELION (protein white): the manifest's RECOVAR-simulator images negated",
        "n_images": n,
        "box": box,
        "voxel_size": float(manifest["voxel_size"]),
        "particle_diameter_ang": float(manifest["particle_diameter_ang"]),
        "files": {name: file_hash(output / name) for name in ("particles.star", "particles.mrcs")},
    }
    (output / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="A recovar-ppca-training-v1 manifest.json")
    parser.add_argument("-o", "--output", required=True, help="New directory for particles.star and particles.mrcs")
    args = parser.parse_args()
    print(json.dumps(convert(args.manifest, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
