"""RELION-equivalent 3D auto-refine (Refine3D, K=1), for SPA and RELION 5 subtomogram particles.

``relax refine --data_dir DIR --output DIR ...``; ``relax refine --help`` lists the
options and ``docs/user_guide.md`` has runnable examples.
"""

from __future__ import annotations

from relax.refinement.full_refinement import run_from_command_line


def main() -> None:
    run_from_command_line("refine")


if __name__ == "__main__":
    main()
