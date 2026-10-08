"""Pose-marginal PPCA refinement scaffolding for EM.

Import from the owning module, not from this package: iterations and dataset
blocks from ``dense_dataset``, blocked E-steps from ``engine``, state from
``state``, and initialization from ``initialization``. The exact-local dataset
loop, the multi-iteration loops and the resolution gating, which no relax
command uses, are script libraries: ``scripts.lib.local_dataset``,
``scripts.lib.refinement_loop`` and ``scripts.lib.schedule``. See README.md for the map.
"""
