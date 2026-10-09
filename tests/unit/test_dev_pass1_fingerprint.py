"""The pass-1 fingerprint tool (``scripts/dev/pass1_fingerprint.py``): its cases, its mutations, and one short run."""

import json
from pathlib import Path

import pytest

from scripts.dev import fingerprint, fingerprint_cli, pass1_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_KEYS = {
    "n_classes", "n_images", "n_rot", "rotation_codes", "box", "n_trans", "rotation_block_size", "projection", "tree",
    "rotation_prior", "translation_prior", "noise", "corrections", "env", "kwargs",
}
# A pass-1 call on the stand-in is a second or two; these three reach both scores, the K=1 route and a refusal.
SHORT_RUN = ("k1_default", "cc_k1", "refused_float64_scoring")


def test_every_case_has_a_description_and_only_known_spec_keys():
    assert len(pass1_fingerprint.CASES) > 40
    for name, (description, spec) in pass1_fingerprint.CASES.items():
        assert description, name
        assert set(spec) <= SPEC_KEYS, (name, set(spec) - SPEC_KEYS)


def test_every_mutation_replaces_line_for_line_and_changes_its_text():
    names = [mutation[0] for mutation in pass1_fingerprint.MUTATIONS]
    assert len(names) == len(set(names))
    for name, old, new, description, expected in pass1_fingerprint.MUTATIONS:
        assert len(old.strip().splitlines()) == len(new.strip().splitlines()), name
        assert old.strip() != new.strip(), name
        assert description and isinstance(expected, bool), name


def test_refusal_cases_are_named_for_what_they_refuse():
    refusals = pass1_fingerprint.REFUSED_CASES
    assert len(refusals) >= 10 and refusals <= set(pass1_fingerprint.CASES)
    assert not refusals & {"k1_default", "cc_k1"}


def test_a_short_run_is_repeatable_and_a_changed_leaf_is_a_difference(tmp_path):
    outputs = []
    for label in ("a", "b"):
        out = tmp_path / f"pass1_{label}.json"
        assert fingerprint_cli.run_tree(
            pass1_fingerprint.HARNESS, REPO_ROOT, out, tmp_path, list(SHORT_RUN), quiet=True
        ) == 0
        outputs.append(json.loads(out.read_text()))
    first, second = outputs

    counts, _ = fingerprint.diff_fingerprints(first, second)
    assert fingerprint.accepted(counts) and not any(counts.values())
    statuses = {name: case["status"][""] for name, case in first["cases"].items()}
    assert statuses["k1_default"] == statuses["cc_k1"] == "ok"
    assert statuses["refused_float64_scoring"].startswith("ValueError: pass 1 scores RELION's exact coarse operands")
    assert first["cases"]["k1_default"]["result"] and first["cases"]["k1_default"]["trace"]
    assert pass1_fingerprint.HARNESS.case_errors(first) == []
    first["cases"]["cc_k1"]["status"][""] = "ValueError: a case that should run"
    first["cases"]["refused_float64_scoring"]["status"][""] = "ok"
    assert [line.split(":")[0] for line in pass1_fingerprint.HARNESS.case_errors(first)] == [
        "cc_k1",
        "refused_float64_scoring",
    ]
    first["cases"]["cc_k1"]["status"][""] = "ok"
    first["cases"]["refused_float64_scoring"]["status"][""] = statuses["refused_float64_scoring"]

    leaf = next(key for key in second["cases"]["cc_k1"]["result"] if key.endswith("/n_significant"))
    second["cases"]["cc_k1"]["result"][leaf] += " changed"
    counts, _ = fingerprint.diff_fingerprints(first, second)
    assert counts["outputs"] == 1 and not fingerprint.accepted(counts)


@pytest.mark.parametrize("name", ["", "no_such_case"])
def test_an_unknown_case_is_refused(name):
    with pytest.raises(SystemExit, match="unknown case"):
        fingerprint_cli._case_names(pass1_fingerprint.HARNESS, [name])
