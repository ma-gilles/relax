"""The suite-scoring wrapper hands the scorer a copy of the checkpoint with the mean negated, nothing else changed."""

import json
import sys

import numpy as np
import pytest

from scripts.ppca_suite_score_star import main

pytestmark = pytest.mark.unit

# A stand-in scorer: records its arguments and the checkpoint it was given.
_SCORER = """
import json, sys
from pathlib import Path
import numpy as np
args = sys.argv[1:]
checkpoint = Path(args[args.index("--checkpoint") + 1])
output = Path(args[args.index("--output") + 1])
output.mkdir(parents=True, exist_ok=True)
with np.load(checkpoint, allow_pickle=False) as saved:
    np.savez(output / "seen.npz", **{name: saved[name] for name in saved.files})
(output / "argv.json").write_text(json.dumps(args))
"""


def test_the_scorer_sees_the_mean_negated(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rng = np.random.default_rng(5)
    theta = (rng.normal(size=(10, 3)) + 1j * rng.normal(size=(10, 3))).astype(np.complex64)
    metadata = json.dumps({"iteration": 2, "config": {"stages": [[1, 4, 3]]}})
    for iteration in (1, 2):
        np.savez(run / f"checkpoint_{iteration:04d}.npz", theta=theta * iteration, noise=np.ones(3, np.float32),
                 metadata=metadata)
    scorer = tmp_path / "suite_score.py"
    scorer.write_text(_SCORER)
    assert main(["--run", str(run), "--eval", "dataset/evaluation", "--suite-score", str(scorer),
                 "--python", sys.executable, "--healpix-order", "4"]) == 0
    output = run / "suite_score_checkpoint_0002"
    argv = json.loads((output / "argv.json").read_text())
    assert argv[argv.index("--eval") + 1] == "dataset/evaluation" and argv[-2:] == ["--healpix-order", "4"]
    with np.load(output / "seen.npz") as seen:
        np.testing.assert_array_equal(seen["theta"][:, 0], -2 * theta[:, 0])
        np.testing.assert_array_equal(seen["theta"][:, 1:], 2 * theta[:, 1:])
        np.testing.assert_array_equal(seen["noise"], np.ones(3, np.float32))
        assert str(seen["metadata"]) == metadata
    record = json.loads((output / "star_sign_copy.json").read_text())
    assert record["checkpoint"] == str((run / "checkpoint_0002.npz").resolve())
