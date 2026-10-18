"""Derive a fixed 2,000-particle training/evaluation split from the 20k fixture."""

import argparse
import json
import shutil
from pathlib import Path

import mrcfile
import numpy as np
from recovar.utils.helpers import write_mrc_stack

from relax.commands.ppca_initial_model import training_identity
from relax.ppca_initial_model.checkpoint import file_hash
from scripts.lib.ppca_training_star import training_star_ctf, write_training_star


def prepare(parent, output, *, count=2000, seed=2718):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    fixture = json.loads((parent / "fixture.json").read_text())
    if count > fixture["n_images"]:
        raise ValueError("Invalid parent fixture or requested count")
    for name, digest in fixture["files"].items():
        if file_hash(parent / name) != digest:
            raise ValueError(f"Parent input hash mismatch: {name}")
    parent_star = parent / fixture["particles_star"]
    indices = np.random.default_rng(seed).choice(fixture["n_images"], count, replace=False).astype(np.int64)
    output.mkdir(parents=True, exist_ok=False)
    (output / "SAFE_TO_DELETE").touch()
    training, evaluation = output / "training", output / "evaluation"
    training.mkdir()
    evaluation.mkdir()
    np.save(output / "selected_parent_indices.npy", indices)
    # The parent stack is already in RELION's sign; the subset keeps it.
    with mrcfile.mmap(parent_star.parent / "particles.mrcs", permissive=False) as stack:
        selected = np.asarray(stack.data[indices], np.float32)
    write_mrc_stack(str(training / "particles.mrcs"), selected, voxel_size=fixture["voxel_size"], dtype=np.float32)
    star = write_training_star(
        training / "particles.star",
        "particles.mrcs",
        training_star_ctf(parent_star)[indices],
        voxel_size=fixture["voxel_size"],
        box=fixture["box"],
    )
    small_fixture = dict(fixture, n_images=count)
    small_fixture["files"] = {f"training/{path.name}": file_hash(path) for path in sorted(training.iterdir())}
    (output / "fixture.json").write_text(json.dumps(small_fixture, indent=2) + "\n")
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
        "subset_parent_star_sha256": file_hash(parent_star),
        "subset_indices_sha256": file_hash(output / "selected_parent_indices.npy"),
        "subset_seed": seed,
        "counts": np.bincount(labels, minlength=3).tolist(),
    })
    (evaluation / "manifest.json").write_text(json.dumps(characterization, indent=2) + "\n")
    verification = {
        "identity": training_identity(star, [training / "particles.mrcs"]),
        "n_images": count,
        "box": fixture["box"],
        "parent_star_sha256": file_hash(parent_star),
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
