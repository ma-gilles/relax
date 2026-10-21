"""RELION native runtime helpers and independent parity references.

RELION-style EM uses this extension for sampling, particle ordering, CTFs
and reconstruction. Other exports provide independent numerical references
for tests and replay tools; the package is not solely a test dependency.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_external_core_if_available() -> None:
    """Load `_relion_bind_core` from an external Slurm build directory.

    Long EM jobs build the RELION binding under scratch via
    `RECOVAR_RELION_BIND_BUILD_DIR`; copying the shared object back into the
    checkout can fail when the source filesystem is quota-full. Preloading the
    extension here lets both `from relax.relion_bind import _relion_bind_core`
    and `from relax.relion_bind._relion_bind_core import ...` work without a
    package-local copy.
    """

    module_name = f"{__name__}._relion_bind_core"
    if module_name in sys.modules:
        return

    package_dir = Path(__file__).resolve().parent
    candidate_dirs = []
    build_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")
    if build_dir:
        candidate_dirs.append(Path(build_dir).expanduser())
    candidate_dirs.append(package_dir / "build")

    for candidate_dir in candidate_dirs:
        try:
            so_files = sorted(candidate_dir.glob("_relion_bind_core*.so"))
        except OSError:
            continue
        for so_path in so_files:
            try:
                if so_path.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            spec = importlib.util.spec_from_file_location(module_name, so_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            globals()["_relion_bind_core"] = module
            return


_load_external_core_if_available()
