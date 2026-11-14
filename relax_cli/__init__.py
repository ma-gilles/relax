"""The ``relax`` console script: start the ranks of a multi-GPU run, then run the command.

``relax refine --gpu 0:1`` (and ``relax class3d``, ``relax ppca_initial_model``) runs one process per listed GPU
(:mod:`relax_cli.launch`).
The ranks must be started before JAX: importing ``relax`` starts the JAX backend on every visible GPU, so this
package imports only the standard library until it hands the command to ``relax.command_line``. Importing
``relax`` itself never starts processes. ``python -m relax_cli <command> ...`` is the same entry point.
"""

from __future__ import annotations

import os
import sys

from relax_cli import launch

# The commands that take --gpu.
MULTI_GPU_COMMANDS = ("refine", "class3d", "ppca_initial_model")


def main() -> None:
    """Run ``relax <command>``: a multi-GPU run's launcher exits with its ranks' status."""

    if len(sys.argv) > 1 and sys.argv[1] in MULTI_GPU_COMMANDS:
        status = launch.launch_ranks(sys.argv, getattr(sys, "orig_argv", ()), os.environ)
        if status is not None:
            sys.exit(status)
    from relax.command_line import main_commands

    main_commands()
