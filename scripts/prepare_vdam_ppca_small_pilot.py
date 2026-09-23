"""Derive a fixed 2,000-particle training/evaluation split from the 20k fixture."""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import mrcfile
import numpy as np
from recovar.utils.helpers import write_mrc_stack

from relax.ppca_initial_model.checkpoint import file_hash


def prepare(parent, output, *, count=2000, seed=2718):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    manifest = json.loads((parent / "training/manifest.json").read_text())
    if manifest["schema"] != "recovar-ppca-training-v1" or count > manifest["n_images"]:
        raise ValueError("Invalid parent fixture or requested count")
    for name, digest in manifest["files"].items():
        if file_hash(parent / "training" / name) != digest:
            raise ValueError(f"Parent input hash mismatch: {name}")
    indices = np.random.default_rng(seed).choice(manifest["n_images"], count, replace=False).astype(np.int64)
    output.mkdir(parents=True, exist_ok=False)
    (output / "SAFE_TO_DELETE").touch()
    training, evaluation = output / "training", output / "evaluation"
    training.mkdir()
    evaluation.mkdir()
    np.save(output / "selected_parent_indices.npy", indices)
    with mrcfile.mmap(parent / "training/particles.mrcs", permissive=False) as stack:
        selected = np.asarray(stack.data[indices], np.float32)
    write_mrc_stack(str(training / "particles.mrcs"), selected, voxel_size=manifest["voxel_size"], dtype=np.float32)
    with open(parent / "training/ctf.pkl", "rb") as stream:
        ctf = np.asarray(pickle.load(stream), np.float32)
    with open(training / "ctf.pkl", "wb") as stream:
        pickle.dump(ctf[indices], stream)
    with open(parent / "training/neutral_poses.pkl", "rb") as stream:
        rotations, translations = pickle.load(stream)
    with open(training / "neutral_poses.pkl", "wb") as stream:
        pickle.dump((rotations[indices], translations[indices]), stream)
    np.save(training / "particle_ids.npy", np.arange(count, dtype=np.int64))
    small_manifest = dict(manifest)
    small_manifest["n_images"] = count
    small_manifest["files"] = {path.name: file_hash(path) for path in sorted(training.iterdir())}
    (training / "manifest.json").write_text(json.dumps(small_manifest, indent=2) + "\n")
    with np.load(parent / "evaluation/truth.npz", allow_pickle=False) as truth:
        labels = truth["labels"][indices]
        np.savez(
            evaluation / "truth.npz",
            labels=labels,
            rotations=truth["rotations"][indices],
            translations=truth["translations"][indices],
            noise_variance=truth["noise_variance"],
            volumes_fourier=truth["volumes_fourier"],
        )
    for state in range(3):
        shutil.copyfile(parent / "evaluation" / f"state{state}.mrc", evaluation / f"state{state}.mrc")
    characterization = json.loads((parent / "evaluation/manifest.json").read_text())
    characterization.update({
        "subset_parent_manifest_sha256": file_hash(parent / "training/manifest.json"),
        "subset_indices_sha256": file_hash(output / "selected_parent_indices.npy"),
        "subset_seed": seed,
        "counts": np.bincount(labels, minlength=3).tolist(),
    })
    (evaluation / "manifest.json").write_text(json.dumps(characterization, indent=2) + "\n")
    verification = {
        "identity": {"manifest_sha256": file_hash(training / "manifest.json")},
        "n_images": count,
        "box": manifest["box"],
        "parent_manifest_sha256": file_hash(parent / "training/manifest.json"),
        "selected_parent_indices_sha256": file_hash(output / "selected_parent_indices.npy"),
        "class_counts_evaluation_only": np.bincount(labels, minlength=3).tolist(),
        "files": {str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "fixture_verified.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.output), indent=2))


if __name__ == "__main__":
    main()
