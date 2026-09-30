"""relax imports and prepares a refinement with RELION's binding absent.

A child interpreter blocks every ``relax.relion_bind`` import, then imports the
production entry points and runs the host-side setup that used to call the
binding: symmetry operators, the HEALPix grids and RELION's random streams.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHILD = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import sys

    class BlockRelionBind(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "relax.relion_bind" or name.startswith("relax.relion_bind."):
                raise ImportError(f"{name} is blocked: production must not need RELION's binding")
            return None

    sys.meta_path.insert(0, BlockRelionBind())

    for module in (
        "relax.command_line",
        "relax.commands.refine",
        "relax.commands.class3d",
        "relax.commands.initial_model",
        "relax.commands.ppca_initial_model",
        "relax.refinement.full_refinement",
        "relax.refinement.iteration_loop",
        "relax.classification.k_class",
        "relax.vdam.driver",
    ):
        importlib.import_module(module)

    from relax import sampling, symmetry
    from relax.helpers import expected_accuracy

    for label in ("C1", "C4", "D2", "T", "O", "I"):
        symmetry.rotational_operators(label)
        symmetry.relion_point_group_code(label)
        sampling.get_relion_rotation_grid(2, symmetry=label)
    sampling.get_relion_rotation_grid_eulers(2)
    sampling._relion_rnd_unif_scaled_first_draw(43, 0.5, 1.0)
    expected_accuracy.relion_auto_refine_half_orders([1, 2, 1, 2, 2, 1], random_seed=42)
    assert "relax.relion_bind" not in sys.modules
    print("OK")
    """
)


def test_production_entry_points_run_without_the_relion_binding():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTHONNOUSERSITE="1")
    env.pop("RECOVAR_RELION_BIND_BUILD_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", CHILD], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("OK")
