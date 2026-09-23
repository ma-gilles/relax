"""relax: EM workflows and shared numerical implementation.

Standard refinement lives in ``refinement`` and InitialModel/VDAM in ``vdam``.
Import functions and types directly from their owning modules.

The launch hooks below run before any submodule imports ``recovar`` (and
with it ``recovar.jax_config``), so every relax process reaches JAX with the
EM XLA defaults, and ``relax initial_model`` and
``python -m relax.commands.initial_model`` with the requested allocator.
"""

import os
import sys


def _initial_model_cli_requested(argv=None, orig_argv=None) -> bool:
    """Return whether this interpreter was launched for InitialModel."""

    argv = tuple(sys.argv if argv is None else argv)
    orig_argv = tuple(getattr(sys, "orig_argv", ()) if orig_argv is None else orig_argv)
    if len(argv) > 1 and argv[1] == "initial_model":
        return True
    return any(
        orig_argv[index] == "-m" and orig_argv[index + 1] == "relax.commands.initial_model"
        for index in range(len(orig_argv) - 1)
    )


def _configure_initial_model_cuda_allocator(*, argv=None, orig_argv=None, environ=None):
    """Apply an InitialModel allocator override before JAX initializes."""

    environ = os.environ if environ is None else environ
    if not _initial_model_cli_requested(argv=argv, orig_argv=orig_argv):
        return environ.get("TF_GPU_ALLOCATOR")
    requested = environ.get(
        "RECOVAR_INITIAL_MODEL_CUDA_ALLOCATOR",
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


_configure_initial_model_cuda_allocator()
_configure_em_xla_defaults()

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
