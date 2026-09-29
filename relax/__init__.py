"""relax: EM workflows and shared numerical implementation.

Standard refinement lives in ``refinement`` and InitialModel/VDAM in ``vdam``.
Import functions and types directly from their owning modules.

The launch hooks below run before any submodule imports ``recovar`` (and
with it ``recovar.jax_config``), so every relax process reaches JAX with the
EM XLA defaults, ``relax initial_model`` and
``python -m relax.commands.initial_model`` with the requested allocator, and
``relax refine`` / ``relax class3d`` with an XLA pool sized for the RELION
projector texture.
"""

import os
import sys


def _cli_command_requested(command, argv=None, orig_argv=None) -> bool:
    """Return whether this interpreter was launched for ``relax <command>``.

    ``relax <command>`` puts the command in ``argv[1]``; ``python -m
    relax.commands.<command>`` names its module in ``orig_argv``.
    """

    argv = tuple(sys.argv if argv is None else argv)
    orig_argv = tuple(getattr(sys, "orig_argv", ()) if orig_argv is None else orig_argv)
    if len(argv) > 1 and argv[1] == command:
        return True
    return any(
        orig_argv[index] == "-m" and orig_argv[index + 1] == f"relax.commands.{command}"
        for index in range(len(orig_argv) - 1)
    )


def _configure_initial_model_cuda_allocator(*, argv=None, orig_argv=None, environ=None):
    """Apply an InitialModel allocator override before JAX initializes."""

    environ = os.environ if environ is None else environ
    if not _cli_command_requested("initial_model", argv=argv, orig_argv=orig_argv):
        return environ.get("TF_GPU_ALLOCATOR")
    requested = environ.get(
        "RELAX_INITIAL_MODEL_CUDA_ALLOCATOR",
        "default",
    ).strip()
    if requested.lower() not in {"", "default", "none", "off"}:
        environ.setdefault("TF_GPU_ALLOCATOR", requested)
    return environ.get("TF_GPU_ALLOCATOR")


def _configure_em_xla_defaults(*, environ=None):
    """Opt every relax process into the EM XLA defaults before JAX initializes.

    Every relax entry point is an EM workflow, and ``relax initial_model`` and
    ``python -m relax.commands.initial_model`` import this package before
    ``recovar``, and with it ``recovar.jax_config``, so the EM
    defaults (``--xla_gpu_autotune_level=0``) reach XLA only if they are set
    here. An explicit ``RECOVAR_EM_XLA_DEFAULTS`` in the environment still wins.
    """

    environ = os.environ if environ is None else environ
    environ.setdefault("RECOVAR_EM_XLA_DEFAULTS", "1")
    return environ.get("RECOVAR_EM_XLA_DEFAULTS")


def _reserve_refinement_projector_memory(*, argv=None, orig_argv=None) -> str | None:
    """Size the XLA pool for a refinement's RELION projector texture; return a log line.

    The pool limit is fixed when the backend starts, and ``recovar.jax_config``
    (imported below) starts it, so ``relax refine`` and ``relax class3d`` size it
    here, from the reference maps their command line names. The reserve module
    imports only the standard library.
    """

    if not any(_cli_command_requested(command, argv, orig_argv) for command in ("refine", "class3d")):
        return None
    from relax.helpers import xla_memory_reserve as reserve

    argv = sys.argv if argv is None else argv
    record = reserve.reserve_for_reference_maps(reserve.reference_maps_from_argv(argv[1:]))
    return None if record is None else reserve.format_reserve_record(record)


def _reject_renamed_environment(*, environ=None):
    """Refuse to start while a renamed relax-only ``RECOVAR_*`` variable is set.

    relax's own environment variables are named ``RELAX_*`` (relax split P5); the list of renamed names, and
    of the ``RECOVAR_*`` names that stay because recovar also reads them or sealed JSON records them, is
    ``renamed_environment.json``. A harness that still sets an old name would otherwise have its setting
    ignored without notice, so the old name is an error, not an alias. A name ending in ``_`` is a prefix.
    A retired name selected behaviour that no longer exists and is an error for the same reason.
    """

    import json
    import pathlib

    environ = os.environ if environ is None else environ
    table = json.loads((pathlib.Path(__file__).with_name("renamed_environment.json")).read_text())
    renamed = table["renamed"]
    exempt = {name for names in table["exempt"].values() for name in names}
    prefixes = [old for old in renamed if old.endswith("_")]
    retired = table["retired"]
    stale = []
    for name in sorted(environ):
        if name in retired:
            stale.append(f"{name} is retired ({retired[name]})")
            continue
        if not name.startswith("RECOVAR_") or name in exempt:
            continue
        if name in renamed:
            stale.append(f"{name} -> {renamed[name]}")
        elif any(name.startswith(prefix) for prefix in prefixes):
            stale.append(f"{name} -> RELAX_{name[len('RECOVAR_'):]}")
    if stale:
        raise RuntimeError(
            "relax reads RELAX_* names for its own settings; rename or unset these environment variables: "
            + ", ".join(stale)
        )


_reject_renamed_environment()
_configure_initial_model_cuda_allocator()
_configure_em_xla_defaults()
_XLA_RESERVE_LOG_LINE = _reserve_refinement_projector_memory()

try:
    # recovar's package import applies the XLA configuration (including the EM defaults marker set above);
    # import it here, as ``import recovar`` did for the in-tree EM package.
    import recovar.jax_config  # noqa: F401
except ModuleNotFoundError:
    pass

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"
