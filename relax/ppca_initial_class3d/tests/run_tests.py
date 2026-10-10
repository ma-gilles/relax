"""Run the package-local suite using the unchanged repository test fixtures.

Usage from the checkout: python -m relax.ppca_initial_class3d.tests.run_tests -q
The owner's package-only constraint means the standard test tiers do not
discover this suite. This explicit entry point needs no shell PYTHONPATH or
plugin flags; it reuses tests/conftest.py, including its shared-GPU guard.
"""

import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    for path in (root, root / "tests", root / "tests/unit/ppca_initial_model"):
        sys.path.insert(0, str(path))
    import pytest

    return pytest.main(["-p", "conftest", str(Path(__file__).parent), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
