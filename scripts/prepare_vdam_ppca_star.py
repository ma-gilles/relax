"""Make a training-only RELION STAR for the VDAM side of the PPCA pilot.

See docs/math/vdam_ppca_algorithm.md, section 14, for the comparison boundary.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from recovar.core.ctf import CTFParamIndex
from recovar.data_io.starfile import write_star


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(manifest_path, output_path):
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "recovar-ppca-training-v1":
        raise ValueError("Expected a training-only PPCA manifest")
    expected = {manifest[key] for key in ("particles", "ctf", "neutral_poses", "particle_ids")}
    if set(manifest["files"]) != expected:
        raise ValueError("Training input inventory mismatch")
    for name, recorded_hash in manifest["files"].items():
        if Path(name).name != name or _sha256(manifest_path.parent / name) != recorded_hash:
            raise ValueError(f"Training input identity mismatch: {name}")
    if manifest["contrast"] != 1 or manifest["image_multiplier"] != 1:
        raise ValueError("Pilot STAR requires unit contrast and unscaled images")
    n = int(manifest["n_images"])
    ids = np.load(manifest_path.parent / manifest["particle_ids"], allow_pickle=False)
    if not np.array_equal(ids, np.arange(n)):
        raise ValueError("Pilot STAR requires contiguous particle IDs")
    with open(manifest_path.parent / manifest["neutral_poses"], "rb") as stream:
        rotations, translations = pickle.load(stream)
    if not np.array_equal(rotations, np.broadcast_to(np.eye(3), (n, 3, 3))) or not np.array_equal(
        translations, np.zeros((n, 2))
    ):
        raise ValueError("Training pose file must contain only neutral poses")
    with open(manifest_path.parent / manifest["ctf"], "rb") as stream:
        ctf = np.asarray(pickle.load(stream))
    if ctf.shape[0] != n or ctf.shape[1] <= 2 + int(CTFParamIndex.PHASE_SHIFT):
        raise ValueError("CTF array shape mismatch")
    if not np.all(ctf[:, 0] == manifest["box"]) or not np.all(ctf[:, 1] == manifest["voxel_size"]):
        raise ValueError("CTF geometry mismatch")
    def col(index):
        return ctf[:, 2 + int(index)].astype(np.float64)

    optics_fields = (CTFParamIndex.VOLT, CTFParamIndex.CS, CTFParamIndex.W)
    if any(not np.all(col(index) == col(index)[0]) for index in optics_fields):
        raise ValueError("Pilot VDAM bootstrap requires one optics group")
    stack = (manifest_path.parent / manifest["particles"]).resolve()
    particles = pd.DataFrame({
        "_rlnImageName": [f"{i + 1}@{stack}" for i in range(n)],
        "_rlnOpticsGroup": np.ones(n, np.int32),
        "_rlnDefocusU": col(CTFParamIndex.DFU),
        "_rlnDefocusV": col(CTFParamIndex.DFV),
        "_rlnDefocusAngle": col(CTFParamIndex.DFANG),
        "_rlnPhaseShift": col(CTFParamIndex.PHASE_SHIFT),
        "_rlnAngleRot": np.zeros(n, np.float32),
        "_rlnAngleTilt": np.zeros(n, np.float32),
        "_rlnAnglePsi": np.zeros(n, np.float32),
        "_rlnOriginXAngst": np.zeros(n, np.float32),
        "_rlnOriginYAngst": np.zeros(n, np.float32),
    })
    optics = pd.DataFrame({
        "_rlnOpticsGroup": ["1"],
        "_rlnImageSize": [manifest["box"]],
        "_rlnImageDimensionality": [2],
        "_rlnImagePixelSize": [manifest["voxel_size"]],
        "_rlnVoltage": [col(CTFParamIndex.VOLT)[0]],
        "_rlnSphericalAberration": [col(CTFParamIndex.CS)[0]],
        "_rlnAmplitudeContrast": [col(CTFParamIndex.W)[0]],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_star(str(output_path), particles, optics, array_rows=True)
    with open(output_path, "r+") as stream:
        lines = stream.readlines()
        lines[0] = "# Training-only VDAM pilot bridge\n"
        stream.seek(0)
        stream.writelines(lines)
        stream.truncate()
    return {"input_manifest_sha256": _sha256(manifest_path), "star_sha256": _sha256(output_path), "rows": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
