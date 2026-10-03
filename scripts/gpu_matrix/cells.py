"""Workflow cells of the cross-GPU compatibility matrix (docs/development/gpu_compatibility.md).

Each cell is one short end-to-end relax run on a small fixture. ``{fx:NAME}`` in an argument is
the fixture directory ``NAME`` (a Della path, or its staged copy on Polar), ``{out}`` the cell's
output directory and ``{cpus}`` the job's CPUs per GPU. ``box`` is the model box, which the
memory emulation needs to reproduce relax's XLA pool reserve for ``refine`` and ``class3d``.
"""

from __future__ import annotations

# Della fixture directories (the manifest's roots, or the curated benchmark cases it does not list yet).
FIXTURES = {
    "k1_5k128": "/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized",
    "k2_5k128": "/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k2_5k_128",
    "k4_5k128": "/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k4_5k_128",
    "k1_50k256": "/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_50k_256_normalized",
    "k4_50k256": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/data_pdb_k4_50k_256",
    "et_s1": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/cryoet_s1_offsets_depthfix_20260925/project",
    "et13_k2": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/cryoet_bench_20260926/cases/et13_k2conf/project",
    "et09_box64": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/cryoet_bench_20260926/cases/et09_box64/project",
    "et15_k2_box64": (
        "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/cryoet_vdam_ogtomo_20261002/cases/"
        "et15_k2conf_box64_ogtomo/project"
    ),
    "ppca_et_k3": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/cryoet_ppca_k3conf_box64_20261002/project",
}

_REFINE_K1 = [
    "--data_dir",
    "{fx:k1_5k128}",
    "--output",
    "{out}",
    "--max_iter",
    "25",
    "--healpix_order",
    "3",
    "--offset_range",
    "3.0",
    "--offset_step",
    "1.0",
    "--adaptive_oversampling",
    "1",
    "--tau2_fudge",
    "1.0",
    "--perturb_factor",
    "0.5",
    "--seed",
    "1775735620",
    "--perturb_seed",
    "1775735620",
    "--relion-half-sets-from-input",
    "--no-firstiter_cc",
    "--particle_diameter_ang",
    "544",
    "--apply-initial-lowpass",
    "--init_resolution",
    "30.0",
]


def _class3d(fx: str, k: int, max_iter: int = 25) -> list[str]:
    return [
        "--data_dir",
        "{fx:%s}" % fx,
        "--output",
        "{out}",
        "--n_classes",
        str(k),
        "--max_iter",
        str(max_iter),
        "--healpix_order",
        "2",
        "--offset_range",
        "5",
        "--offset_step",
        "2",
        "--init_resolution",
        "60.0",
        "--tau2_fudge",
        "4.0",
        "--ref_star",
        "{fx:%s}/reference_init_classes_relion.star" % fx,
        "--particle_diameter_ang",
        "200",
        "--seed",
        "29",
        "--firstiter_cc",
        "--apply-initial-lowpass",
    ]


def _vdam(fx: str, k: int, nr_iter: int = 200, diameter: int = 150) -> list[str]:
    return [
        "--i",
        "{fx:%s}/particles.star" % fx,
        "--datadir",
        "{fx:%s}" % fx,
        "--o",
        "{out}/run",
        "--nr_iter",
        str(nr_iter),
        "--padding_factor",
        "1",
        "--K",
        str(k),
        "--sym",
        "C1",
        "--particle_diameter",
        str(diameter),
        "--oversampling",
        "1",
        "--healpix_order",
        "1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--tau2_fudge",
        "4",
        "--grad_write_iter",
        "50",
        "--random_seed",
        "29",
        "--j",
        "{cpus}",
        "--gpu",
        "0",
    ]


def _tomo_vdam(fx: str, k: int, diameter: int) -> list[str]:
    return [
        "--ios",
        "{fx:%s}/optimisation_set.star" % fx,
        "--o",
        "{out}/run",
        "--nr_iter",
        "200",
        "--grad_write_iter",
        "10",
        "--K",
        str(k),
        "--sym",
        "C1",
        "--particle_diameter",
        str(diameter),
        "--oversampling",
        "1",
        "--healpix_order",
        "1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--tau2_fudge",
        "4",
        "--random_seed",
        "1",
        "--gpu",
        "0",
        "--j",
        "{cpus}",
        "--stop-file",
        "{out}/STOP",
    ]


def _ppca_tomo(optimizer: str) -> list[str]:
    return [
        "--ios",
        "{fx:ppca_et_k3}/optimisation_set.star",
        "--particle-diameter",
        "400",
        "-o",
        "{out}",
        "--oversampling",
        "0",
        "--stream-coarse-recompute",
        "--optimizer",
        optimizer,
        "--q",
        "2",
        "--seed",
        "11",
        # The default 200-iteration schedule compressed to 24 iterations over the same four stages.
        "--iterations",
        "24",
        "--stages",
        "[[1,4,1],[7,8,2],[13,16,3],[19,32,3]]",
    ]


CELLS: dict[str, dict] = {
    # SPA
    "refine_k1_5k128": {"module": "relax.commands.refine", "args": _REFINE_K1, "box": 128},
    "class3d_k2_5k128": {"module": "relax.commands.class3d", "args": _class3d("k2_5k128", 2), "box": 128},
    "class3d_k4_5k128": {"module": "relax.commands.class3d", "args": _class3d("k4_5k128", 4), "box": 128},
    "vdam_k1_5k128": {"module": "relax.commands.initial_model", "args": _vdam("k1_5k128", 1), "box": 128},
    "vdam_k4_5k128": {"module": "relax.commands.initial_model", "args": _vdam("k4_5k128", 4), "box": 128},
    # Box 256, for the small-memory emulation (short schedules: the first iterations size the caches).
    "class3d_k4_50k256_it4": {
        "module": "relax.commands.class3d",
        "args": _class3d("k4_50k256", 4, max_iter=4),
        "box": 256,
    },
    "vdam_k1_50k256_it8": {
        "module": "relax.commands.initial_model",
        "args": [a if a != "200" else "8" for a in _vdam("k1_50k256", 1, diameter=200)],
        "box": 256,
    },
    # Tomography (RELION 5 2D stacks)
    "tomo_refine_s1": {
        "module": "relax.commands.refine",
        "args": [
            "--data_dir",
            "{fx:et_s1}",
            "--output",
            "{out}",
            "--init_volume",
            "{fx:et_s1}/reference_init_relion.mrc",
            "--init_resolution",
            "40",
            "--particle_diameter_ang",
            "400",
            "--healpix_order",
            "2",
            "--auto_local_healpix_order",
            "4",
            "--offset_range",
            "5",
            "--offset_step",
            "1",
            "--seed",
            "20260926",
            "--no-firstiter_cc",
        ],
        "box": 128,
    },
    "tomo_class3d_et13_it3": {
        "module": "relax.commands.class3d",
        "args": [
            "--data_dir",
            "{fx:et13_k2}",
            "--output",
            "{out}",
            "--init_volume",
            "{fx:et13_k2}/reference_init_relion.mrc",
            "--n_classes",
            "2",
            "--init_resolution",
            "40",
            "--particle_diameter_ang",
            "400",
            "--healpix_order",
            "2",
            "--offset_range",
            "5",
            "--offset_step",
            "1",
            "--tau2_fudge",
            "4",
            "--max_iter",
            "3",
            "--seed",
            "1",
            "--no-firstiter_cc",
        ],
        "box": 128,
    },
    "tomo_vdam_k1_et09_it10": {
        "module": "relax.commands.initial_model",
        "args": _tomo_vdam("et09_box64", 1, 240),
        "box": 64,
        "touch": "{out}/STOP",
        "ok_pattern": "stopped at saved iteration 10",
    },
    "tomo_vdam_k2_et15_it10": {
        "module": "relax.commands.initial_model",
        "args": _tomo_vdam("et15_k2_box64", 2, 400),
        "box": 64,
        "touch": "{out}/STOP",
        "ok_pattern": "stopped at saved iteration 10",
    },
    "ppca_tomo_vdam_it24": {"module": "relax.commands.ppca_initial_model", "args": _ppca_tomo("vdam"), "box": 64},
    "ppca_tomo_sgd_it24": {
        "module": "relax.commands.ppca_initial_model",
        "args": _ppca_tomo("momentum_sgd"),
        "box": 64,
    },
}


def cell_fixtures(name: str) -> list[str]:
    """The fixture names a cell's arguments refer to."""

    import re

    found = []
    for arg in CELLS[name]["args"]:
        for match in re.finditer(r"\{fx:([A-Za-z0-9_]+)\}", arg):
            if match.group(1) not in found:
                found.append(match.group(1))
    return found
