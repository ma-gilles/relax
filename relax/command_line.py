"""Command-line entry point for the ``relax`` CLI.

Dispatches ``relax <command>`` to the matching module under
``relax.commands``. Each command module must define a ``main()`` function
that uses ``argparse`` for its own argument parsing. See
``[project.scripts]`` in ``pyproject.toml``.
"""

from __future__ import annotations

import importlib
import os
import sys


def _print_available_commands(available_cmds, file=None):
    file = file if file is not None else sys.stderr
    print("Available commands:", file=file)
    for cmd in available_cmds:
        print(f"  {cmd}", file=file)


def main_commands() -> None:
    """Primary entry point installed as ``relax <cmd_module_name>``."""
    cmd_dir = os.path.join(os.path.dirname(__file__), "commands")
    available_cmds = sorted(
        os.path.splitext(f)[0] for f in os.listdir(cmd_dir) if f.endswith(".py") and f != "__init__.py"
    )
    if len(sys.argv) < 2:
        print("Usage: relax <command>", file=sys.stderr)
        _print_available_commands(available_cmds)
        sys.exit(1)
    cmd_name = sys.argv[1]
    if cmd_name not in available_cmds:
        print(f"Command '{cmd_name}' not found.", file=sys.stderr)
        _print_available_commands(available_cmds)
        sys.exit(1)
    # The subcommand's own parser must not see its name.
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module_name = f"relax.commands.{cmd_name}"
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "main"):
        print(f"Module {module_name} does not define a main() function.", file=sys.stderr)
        sys.exit(1)
    mod.main()


if __name__ == "__main__":
    main_commands()
