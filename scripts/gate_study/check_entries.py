"""Guard for suite entry lists of the gate study: every row must point at the mean-negated copy of a STAR-trained run.

Runs of ``relax ppca_initial_model --i <STAR>`` on a converted suite dataset (scripts/prepare_vdam_ppca_star.py negates the
images) hold the negated simulator mean; the suite infers on the simulator-sign manifest, so it must score the copy with
theta[:, 0] negated (scripts/ppca_suite_score_star.write_mean_negated_copy). For each line of ENTRIES (name, checkpoint,
dataset[, grid]) this checks that

- the checkpoint is listed in <copies>/PROVENANCE.json, and its source checkpoint's sha256 still matches;
- the source run read a particle STAR (run.json input.particles ends in .star);
- the copy's mean is exactly the negated source mean and its loadings are the source's;
- the dataset of the line is the one the source run's STAR was converted from (training/<TAG>/conversion.json).

Exit status 1 and a list of the offending rows on any failure. Run before every GPU-stage submission:
python scripts/gate_study/check_entries.py ENTRIES [--copies GS/ckpt_suite]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_line(line, provenance):
    name, checkpoint, dataset = line.split("\t")[:3]
    copy = provenance["copies"].get(str(Path(checkpoint)))
    if copy is None:
        return f"{name}: {checkpoint} is not a recorded mean-negated copy"
    source = Path(copy["source"])
    if sha256(source) != copy["source_sha256"]:
        return f"{name}: source {source} changed since the copy was made"
    run = json.loads((source.parent / "run.json").read_text())
    star = Path(run["input"]["particles"] or "")
    if star.suffix != ".star":
        return f"{name}: source run did not read a STAR ({star}); a copy is not needed and would be wrong"
    conversion = json.loads((star.parent / "conversion.json").read_text())
    if Path(conversion["source_manifest"]).parent.parent != Path(dataset):
        return f"{name}: dataset {dataset} is not the one the training STAR came from ({conversion['source_manifest']})"
    with np.load(source) as a, np.load(checkpoint) as b:
        ta, tb = a["theta"], b["theta"]
    if not (np.array_equal(tb[:, 0], -ta[:, 0]) and np.array_equal(tb[:, 1:], ta[:, 1:])):
        return f"{name}: {checkpoint} is not the source with its mean negated"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entries", type=Path)
    parser.add_argument("--copies", type=Path, default=Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/gate_study_20261113/ckpt_suite"))
    args = parser.parse_args()
    provenance = json.loads((args.copies / "PROVENANCE.json").read_text())
    lines = [l for l in args.entries.read_text().splitlines() if l.strip()]
    problems = [p for p in (check_line(l, provenance) for l in lines) if p]
    for p in problems:
        print("BAD", p)
    print(f"{args.entries}: {len(lines)} rows, {len(problems)} bad")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
