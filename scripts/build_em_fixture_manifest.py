#!/usr/bin/env python
"""Write tests/fixtures/em_fixture_manifest.json: the external EM data the test tiers read.

Each fixture set names one directory, the files a tier reads from it (size and sha256)
and its provenance (the RELION command, version and seed from the stored optimiser
header, plus the generation record where one exists). Tests resolve paths through
``tests/helpers/em_fixtures.py`` and fail when a file is missing or its hash differs.

Rebuilding the manifest is a baseline regeneration: run it only when a fixture is
added or deliberately replaced, and review the diff. ``--check`` re-hashes the
stored sets and reports differences without writing.

usage: python scripts/build_em_fixture_manifest.py [--only NAME ...] [--check] [--jobs N]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "fixtures" / "em_fixture_manifest.json"

PROJ = "/scratch/gpfs/GILLES/mg6942/em_relion_proj"
FX = "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures"
TOP = ["*"]
RUN = ["**"]
NO_BILD = ["*_angdist.bild", "**/*_angdist.bild"]

# name: (root, include globs, exclude globs, description, generation-record paths relative to root)
SETS: dict[str, tuple[str, list[str], list[str], str, list[str]]] = {
    "k1_5k128_data": (
        f"{PROJ}/data_noise1_5k_normalized",
        TOP,
        [],
        "K1 5k/128 synthetic particles, CTF, GT and initial maps",
        [],
    ),
    "k1_5k128_relion_os0": (
        f"{PROJ}/data_noise1_5k_normalized/relion_ref_os0",
        RUN,
        NO_BILD,
        "RELION auto-refine of k1_5k128_data, oversampling 0 run name (command in header)",
        [],
    ),
    "k1_5k128_relion_os1": (
        f"{FX}/data_noise1_5k_normalized_os1/relion_ref_os1",
        RUN,
        NO_BILD,
        "RELION auto-refine of k1_5k128_data with --oversampling 1",
        ["../GENERATION.json"],
    ),
    "k1_5k128_relion_os0_repeats": (f"{FX}/data_noise1_5k_normalized_os0_repeats", RUN, [], "Two same-command RELION repeats of k1_5k128_relion_os0 (all iterations; Slurm 14363910)", ["PROVENANCE.json"]),
    "k1_5k128_relion_os1_repeats": (f"{FX}/data_noise1_5k_normalized_os1/relion_repeats", RUN, [], "Two same-command RELION repeats of k1_5k128_relion_os1 (all iterations; Slurm 14363910)", ["PROVENANCE.json"]),
    "multioptics_s3b_600_data": (
        f"{FX}/multioptics_s3b_600/data",
        RUN,
        [],
        "600 particles of the S3b two-optics simulation (4.25 A/128 px and 5.44 A/112 px), greyscale-matched reference and GT",
        ["greyscale_rescale.json"],
    ),
    "multioptics_s3b_600_relion": (f"{FX}/multioptics_s3b_600/relion_ref", RUN, NO_BILD, "RELION f2c1a3 Refine3D of multioptics_s3b_600_data, 3 iterations (command in COMMAND.txt)", ["COMMAND.txt"]),
    "multioptics_s3b_600_relion_repeat": (f"{FX}/multioptics_s3b_600/relion_repeat", RUN, NO_BILD, "Same-command RELION repeat of multioptics_s3b_600_relion", ["COMMAND.txt"]),
    "k1_5k128_relion_gui60": (f"{FX}/data_noise1_5k_normalized_gui60/relion_ref", RUN, NO_BILD, "RELION f2c1a3 auto-refine of k1_5k128_data with the GUI-default --ini_high 60 --healpix_order 2 --offset_range 5 --offset_step 2 --oversampling 1 (Slurm 14406407)", ["../GENERATION.json"]),
    "k1_5k128_relion_gui60_repeat": (f"{FX}/data_noise1_5k_normalized_gui60/relion_repeat", RUN, NO_BILD, "Same-command RELION repeat of k1_5k128_relion_gui60 (Slurm 14406407)", ["../GENERATION.json"]),
    "k2_5k128_data": (
        f"{PROJ}/data_pdb_k2_5k_128",
        TOP,
        [],
        "K2 5k/128 PDB-state particles and initial maps",
        ["generation_config.json"],
    ),
    "k2_5k128_relion_os0": (
        f"{PROJ}/data_pdb_k2_5k_128/relion_pdb_k2_os0_ref",
        RUN,
        NO_BILD,
        "RELION K2 Class3D of k2_5k128_data",
        [],
    ),
    "k2_5k128_relion_repeats": (f"{FX}/data_pdb_k2_5k_128_relion_repeats", RUN, [], "Two same-command RELION repeats of k2_5k128_relion_os0 (Slurm 14363910)", ["PROVENANCE.json"]),
    "k4_5k128_data": (
        f"{PROJ}/data_pdb_k4_5k_128",
        TOP,
        [],
        "K4 5k/128 Ribosembly particles and initial maps",
        ["generation_config.json"],
    ),
    "k4_5k128_relion_os0": (
        f"{PROJ}/data_pdb_k4_5k_128/relion_pdb_k4_os0_ref",
        RUN,
        NO_BILD,
        "RELION K4 Class3D of k4_5k128_data (unit tests)",
        [],
    ),
    "k4_5k128_oracle_h2_os1": (
        f"{FX}/k4_fast_oracles/h2_os1_13630212/oracle",
        RUN,
        [],
        "RELION K4 3-iteration oracle with dispatch capture, healpix 2 os 1 (Slurm 13630212)",
        ["../PROVENANCE.json", "../attempt/invocation.json"],
    ),
    "k4_5k128_oracle_h1_os1": (
        f"{FX}/k4_fast_oracles/h1_os1_13605775/oracle",
        RUN,
        [],
        "RELION K4 3-iteration oracle with dispatch capture, healpix 1 os 1 (Slurm 13605775)",
        ["../PROVENANCE.json", "../attempt/invocation.json"],
    ),
    "k4_5k128_oracle_h2_os1_repeat": (f"{FX}/k4_fast_oracles/h2_os1_14248155", RUN, [], "Second RELION capture (Slurm 14248155) of the healpix 2 os 1 K4 oracle: RELION repeat", ["PROVENANCE.json"]),
    "k4_5k128_oracle_h1_os1_repeat": (f"{FX}/k4_fast_oracles/h1_os1_14248154", RUN, [], "Second RELION capture (Slurm 14248154) of the healpix 1 os 1 K4 oracle: RELION repeat", ["PROVENANCE.json"]),
    "k1_50k256_data": (
        f"{PROJ}/data_noise1_50k_256_normalized",
        TOP,
        [],
        "K1 50k/256 synthetic particles, CTF, GT and initial maps",
        ["prepare.log"],
    ),
    "k1_50k256_relion_os0": (
        f"{PROJ}/data_noise1_50k_256_normalized/relion_ref_os0",
        RUN,
        NO_BILD,
        "RELION auto-refine of k1_50k256_data",
        [],
    ),
    "k1_50k256_relion_initialmodel_it008": (
        f"{PROJ}/data_noise1_50k_256_normalized/relion_initialmodel_k1_it008",
        RUN,
        NO_BILD,
        "RELION InitialModel (VDAM) 8 iterations of k1_50k256_data",
        [],
    ),
    "k1_50k256_relion_repeats": (
        f"{FX}/data_noise1_50k_256_normalized/relion_repeats",
        RUN,
        [],
        "Same-command RELION repeat and a --random_seed 20260923 RELION run of k1_50k256_relion_os0 (final outputs)",
        ["PROVENANCE.json"],
    ),
    "k1_50k256_mask": (
        f"{FX}/data_noise1_50k_256_normalized/masks/noise1_k1_50k256_c1",
        RUN,
        [],
        "Frozen mask for masked FSC",
        ["MASK.json"],
    ),
    "k4_50k256_data": (
        f"{FX}/data_pdb_k4_50k_256",
        TOP,
        [],
        "K4 50k/256 Ribosembly particles and initial maps",
        ["GENERATION.json"],
    ),
    "k4_50k256_relion_os0": (
        f"{FX}/data_pdb_k4_50k_256/relion_pdb_k4_os0_ref",
        RUN,
        NO_BILD,
        "RELION K4 Class3D of k4_50k256_data (15 iterations)",
        ["../GENERATION.json"],
    ),
    "k4_50k256_relion_repeats": (
        f"{FX}/data_pdb_k4_50k_256/relion_repeats",
        RUN,
        [],
        "Same-command RELION repeat of k4_50k256_relion_os0 (final iteration)",
        ["repeat_r1_14298976/PROVENANCE.json"],
    ),
    "k4_50k256_mask": (
        f"{FX}/data_pdb_k4_50k_256/masks/pdb_k4_50k256_c1",
        RUN,
        [],
        "Frozen mask for masked FSC",
        ["MASK.json"],
    ),
    "k1_100k256_data": (
        f"{PROJ}/pdb_k1_g256_n100000_noise1_bf80_20260516",
        TOP,
        [],
        "K1 100k/256 PDB particles (noise 1, B 80) and initial maps",
        ["generation_config.json", "generation.log"],
    ),
    "k1_100k256_relion": (
        f"{PROJ}/pdb_k1_g256_n100000_noise1_bf80_20260516/relion_autorefine_k1_it015_os1",
        RUN,
        NO_BILD,
        "RELION auto-refine of k1_100k256_data (completion oracle)",
        [],
    ),
    "k1_100k256_relion_repeats": (
        f"{FX}/pdb_k1_g256_n100000_noise1_bf80/relion_repeats",
        RUN,
        [],
        "Two same-command RELION repeats of k1_100k256_relion (final outputs)",
        ["PROVENANCE.json"],
    ),
    "k1_100k256_mask": (
        f"{FX}/pdb_k1_g256_n100000_noise1_bf80/masks/pdb_k1_100k256_c1",
        RUN,
        [],
        "Frozen mask for masked FSC",
        ["MASK.json"],
    ),
    "k4_100k256_data": (
        f"{PROJ}/ribosembly_k4_g256_n100000_completion_20260512_171123",
        TOP,
        [],
        "K4 100k/256 Ribosembly particles and initial maps",
        ["generation_config.json"],
    ),
    "k4_100k256_relion": (
        f"{PROJ}/ribosembly_k4_g256_n100000_completion_20260512_171123/relion_class3d_k4_it015_clean9d9",
        RUN,
        NO_BILD,
        "RELION K4 Class3D of k4_100k256_data (stored completion reference)",
        [],
    ),
    "k4_100k256_dispatch_oracle": (
        f"{FX}/ribosembly_k4_g256_n100000/relion_dispatch_oracle_h100_13708102",
        RUN,
        [],
        "Same-command RELION K4 Class3D on H100 with dispatch schedule (Slurm 13708102): K4 completion replay oracle and second RELION run",
        ["PROVENANCE.json"],
    ),
    "k4_100k256_mask": (
        f"{FX}/ribosembly_k4_g256_n100000/masks/ribosembly_k4_100k256_c1",
        RUN,
        [],
        "Frozen mask for masked FSC",
        ["MASK.json"],
    ),
}
SETS["empiar_10097_hp3_state"] = (
    f"{FX}/empiar_10097/relion_refine_hp3_state", RUN, [],
    "EMPIAR-10097 RELION auto-refine state at iteration 13 (hp3, os1, current size 136) with its target iteration 14 and the input particles.star", ["PROVENANCE.json"],
)
SETS["empiar_10097_particle_stack"] = (
    "/projects/CRYOEM/singerlab/mg6942/10097/data/Particle-Stack", TOP, [],
    "EMPIAR-10097 T40 HA 130k equalized particle stack (256 px) read by empiar_10097_hp3_state/input/particles.star", [],
)
for _ds in ("10073", "10097", "10345"):
    SETS[f"empiar_{_ds}_relion_startup"] = (
        f"{FX}/empiar_{_ds}/relion_startup_tables",
        RUN,
        [],
        f"EMPIAR-{_ds} RELION InitialModel it200 data.star and auto-refine run_it000 stars",
        ["PROVENANCE.json"],
    )
_SYMMETRY_CASES = (
    ["25_ribo_k4_3k_g128_white_noise1_batch50_seed41001", "k1_c4"]
    + [f"{n}_ribo_k4_5k_g128_white_noise1_{g}_uniform_seed4100{s}" for n, g in ((31, "c4"), (32, "d4"), (33, "o"), (34, "i1")) for s in (1, 2, 3)]
)
for _case in _SYMMETRY_CASES:
    SETS[f"symmetry_{_case}"] = (
        f"{FX}/em_symmetry_matrix/{_case}", RUN, ["relion_ref_f2c1a3/*"] if _case == "k1_c4" else [],
        f"Symmetry-matrix case {_case}: data + RELION reference as pinned by the frozen Q 427a08bd8 run", ["PROVENANCE.json"],
    )
SETS["symmetry_k1_c4_relion_f2c1a3"] = (
    f"{FX}/em_symmetry_matrix/k1_c4/relion_ref_f2c1a3", RUN, [],
    "K1 C4 symmetry-matrix RELION auto-refine oracle recaptured with the f2c1a3 build (mt19937 order; Slurm 14367701, "
    "same command/inputs as ../relion_ref); replaces the legacy-order relion_ref for the seeded K1 C4 comparison",
    ["PROVENANCE.json"],
)

# Oracles written by a RELION build with the libc particle order (optimiser header
# "version 5.0.1" without a commit: the MOLBIO module build or the d476e6f dispatch build).
# relax implements only RELION 5.0.1 f2c1a3's mt19937 order (docs/development/relion_defaults.md).
_LIBC_ORDER_K1 = (
    "relion_ref/ was captured with the legacy-order module build; its seeded K1 comparison is invalid. "
    "Superseded by relion_ref_f2c1a3/ (set symmetry_k1_c4_relion_f2c1a3, Slurm 14367701)."
)
_LIBC_ORDER_KCLASS = (
    "Captured with a legacy-order (libc) RELION build: its Class3D expected-accuracy trial particles "
    "differ from relax's mt19937 draw. Known oracle-build difference, not a relax bug."
)
ORACLE_BUILD_NOTES = {
    "symmetry_k1_c4": _LIBC_ORDER_K1,
    "k4_5k128_oracle_h1_os1": _LIBC_ORDER_KCLASS,
    "k4_5k128_oracle_h2_os1": _LIBC_ORDER_KCLASS,
    "k4_5k128_oracle_h1_os1_repeat": _LIBC_ORDER_KCLASS,
    "k4_5k128_oracle_h2_os1_repeat": _LIBC_ORDER_KCLASS,
    "k4_100k256_dispatch_oracle": _LIBC_ORDER_KCLASS,
    **{f"symmetry_{case}": _LIBC_ORDER_KCLASS for case in _SYMMETRY_CASES if case != "k1_c4"},
}


def _select(root: Path, include: list[str], exclude: list[str]) -> list[str]:
    """TOP selects the directory's own files; RUN selects every file below it."""
    rels = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if not path.is_file() or (include == TOP and "/" in rel):
            continue
        if any(fnmatch.fnmatch(rel, g) for g in exclude):
            continue
        rels.append(rel)
    return rels


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _relion_provenance(root: Path) -> dict:
    """RELION command, version and seed from the first optimiser star in a RELION output dir."""
    stars = sorted(root.glob("run_it000_optimiser.star")) or sorted(root.glob("**/run_it000_optimiser.star"))
    out = {}
    for star in stars[:4]:
        lines = star.read_text().splitlines()
        rec = {"version": lines[0].lstrip("# ").strip(), "command": lines[1].lstrip("# ").strip()}
        for line in lines:
            if line.startswith("_rlnRandomSeed "):
                rec["random_seed"] = int(line.split()[1])
        out[star.relative_to(root).as_posix()] = rec
    return out


def build_set(name: str, jobs: int) -> dict:
    root_s, include, exclude, description, records = SETS[name]
    root = Path(root_s)
    if not root.is_dir():
        raise FileNotFoundError(f"{name}: {root} is not a directory")
    rels = _select(root, include, exclude)
    if not rels:
        raise RuntimeError(f"{name}: no files selected under {root}")
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        digests = dict(zip(rels, pool.map(lambda r: _sha256(root / r), rels)))
    files = {r: [(root / r).stat().st_size, digests[r]] for r in rels}
    provenance = {"generation_records": [str((root / r).resolve()) for r in records if (root / r).exists()]}
    relion = _relion_provenance(root)
    if relion:
        provenance["relion_optimiser_headers"] = relion
    entry = {"root": str(root), "description": description, "provenance": provenance, "files": files}
    if name in ORACLE_BUILD_NOTES:
        entry["oracle_build_note"] = ORACLE_BUILD_NOTES[name]
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="fixture sets to (re)build; default all")
    parser.add_argument(
        "--check", action="store_true", help="re-hash and compare with the stored manifest; write nothing"
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    stored = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"sets": {}}
    names = args.only or sorted(SETS)
    unknown = sorted(set(names) - set(SETS))
    if unknown:
        parser.error(f"unknown fixture sets: {unknown}")
    failed = False
    for name in names:
        entry = build_set(name, args.jobs)
        n_bytes = sum(size for size, _ in entry["files"].values())
        if args.check:
            same = stored["sets"].get(name, {}).get("files") == entry["files"]
            failed |= not same
            print(f"{name}: {'ok' if same else 'DIFFERS'} ({len(entry['files'])} files)", flush=True)
            continue
        stored["sets"][name] = entry
        print(f"{name}: {len(entry['files'])} files, {n_bytes / 1e9:.2f} GB", flush=True)
    if not args.check:
        stored["note"] = (
            "External EM fixtures read by the relax test tiers. Generated by scripts/build_em_fixture_manifest.py; "
            "tests resolve paths via tests/helpers/em_fixtures.py and fail on missing files or hash differences."
        )
        stored["sets"] = dict(sorted(stored["sets"].items()))
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(stored, indent=1, sort_keys=True)
        # One line per file entry: [size, sha256].
        text = re.sub(r"\[\n\s+(\d+),\n\s+(\"[0-9a-f]{64}\")\n\s+\]", r"[\1, \2]", text)
        MANIFEST.write_text(text + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
