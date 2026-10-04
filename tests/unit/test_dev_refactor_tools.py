"""Pure logic of the refactor tools ``scripts/dev/loop_kinds.py`` and ``scripts/dev/check_mutation_anchors.py``."""

from pathlib import Path

from scripts.dev import check_mutation_anchors, loop_kinds

REPO_ROOT = Path(__file__).resolve().parents[2]

_CONTROLLER = '''
def refine_single_volume():
    while True:
        logger.info("start")
        history.wall_times.append(1.0)
        if k_class_enabled:
            weights = class_step()
        else:
            halves = k1_step()
            fsc = k1_curve()
        result = wide_call(
            first,
            second,
            third,
        )
        state = result
'''


def test_loop_lines_are_counted_per_mode_and_by_kind():
    result = loop_kinds.classify_loop(_CONTROLLER)

    assert (result["body"], result["k1"], result["class3d"]) == (13, 2, 1)
    assert result["shared"] == 10
    assert result["kinds"]["logs"] == 1
    assert result["kinds"]["history writes"] == 1
    assert result["kinds"][loop_kinds.WIDE_CALL] == 5
    assert result["kinds"]["installs (assignments)"] == 1
    assert result["kinds"]["control flow"] == 2
    assert result["calls"] == [(11, 5, "wide_call")]


def test_every_selftest_mutation_finds_its_anchor_in_this_checkout():
    assert check_mutation_anchors.stale_mutations(REPO_ROOT) == []


def test_the_cpu_unit_list_names_existing_test_files():
    lines = (REPO_ROOT / "scripts/dev/refactor_cpu_unit_list.txt").read_text().splitlines()
    files = [line for line in lines if line and not line.startswith("#")]

    assert len(files) == len(set(files)) > 100
    missing = [name for name in files if not (REPO_ROOT / name).is_file()]
    assert len(missing) <= 5, missing
