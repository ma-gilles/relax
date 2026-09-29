"""RELION-equivalent 3D classification (Class3D, K>1).

``relax class3d --n_classes K --data_dir DIR --output DIR ...``; ``relax class3d --help``
lists the options and ``docs/user_guide.md`` has runnable examples.
"""

from __future__ import annotations

from relax.refinement.full_refinement import run_from_command_line


def main() -> None:
    run_from_command_line("class3d")


if __name__ == "__main__":
    main()
