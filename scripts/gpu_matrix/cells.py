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
    "w2_09_box192": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/cryoet_bench_20260926/cases/w2_09_box192/project",
    "ms2_448": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/synth_ms2_icos_448_b40_20260930",
    "multioptics_k2_10k128": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/multioptics_k2_10k128_20260930/project",
    "optics_mag_k1": "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/optics_mag_k1_10k256_20260929",
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
    "refine_k1_50k256": {
        "module": "relax.commands.refine",
        "args": [
            "--data_dir",
            "{fx:k1_50k256}",
            "--output",
            "{out}",
            "--max_iter",
            "15",
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
            "--particle_diameter_ang",
            "200",
            "--firstiter_cc",
            "--apply-initial-lowpass",
            "--init_resolution",
            "30.0",
        ],
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
        # The stop file ends the run after the iteration in progress once iteration 10 is written.
        "stop_when": ["{out}/run_it010_model.star", "{out}/STOP"],
        "ok_pattern": "VDAM stopped at saved iteration 1",
    },
    "tomo_vdam_k2_et15_it10": {
        "module": "relax.commands.initial_model",
        "args": _tomo_vdam("et15_k2_box64", 2, 400),
        "box": 64,
        # The stop file ends the run after the iteration in progress once iteration 10 is written.
        "stop_when": ["{out}/run_it010_model.star", "{out}/STOP"],
        "ok_pattern": "VDAM stopped at saved iteration 1",
    },
    "ppca_tomo_vdam_it24": {"module": "relax.commands.ppca_initial_model", "args": _ppca_tomo("vdam"), "box": 64},
    "ppca_tomo_sgd_it24": {
        "module": "relax.commands.ppca_initial_model",
        "args": _ppca_tomo("momentum_sgd"),
        "box": 64,
    },
}


def _reseeded(name: str, seed: int) -> dict:
    """``name`` with another ``--random_seed``: the same-GPU seed spread that cross-GPU differences are judged by."""

    spec = dict(CELLS[name])
    args = list(spec["args"])
    args[args.index("--random_seed") + 1] = str(seed)
    spec["args"] = args
    return spec


# PPCA with an explicit particles-per-tile request: 16 (the old default) and 150 (the dense-stream default),
# which a 16 GB card must cut down to fit (relax_ppca tile planner).
for _optimizer, _tag in (("vdam", "vdam"), ("momentum_sgd", "sgd")):
    for _batch in (16, 150):
        CELLS[f"ppca_tomo_{_tag}_it24_b{_batch}"] = {
            "module": "relax.commands.ppca_initial_model",
            "args": [*_ppca_tomo(_optimizer), "--image-batch-size", str(_batch)],
            "box": 64,
        }

for _name, _seeds in (("vdam_k1_5k128", (41, 53)), ("vdam_k4_5k128", (41, 53)), ("tomo_vdam_k1_et09_it10", (2, 3))):
    for _seed in _seeds:
        CELLS[f"{_name}_s{_seed}"] = _reseeded(_name, _seed)

# Same-seed repeats on one GPU: the run-to-run spread of a nondeterministic GPU run.
for _repeat in (2, 3):
    CELLS[f"vdam_k4_5k128_r{_repeat}"] = dict(CELLS["vdam_k4_5k128"])

# The box-256 Refine3D with a 3-iteration cap: its final all-data pass and tau2 run at the full box quickly.
CELLS["refine_k1_50k256_it3"] = dict(CELLS["refine_k1_50k256"])
CELLS["refine_k1_50k256_it3"]["args"] = [
    "3" if previous == "--max_iter" else value
    for previous, value in zip([None, *CELLS["refine_k1_50k256"]["args"]], CELLS["refine_k1_50k256"]["args"])
]

# Robustness v1 (relax_bench_k1plus_20260925/robustness_gaps_v1.md, cells 1, 2 and 8): the benchmark page's relax
# commands at full schedule, so each card's run is judged against the page row's RELION band and the H100 run of
# the same commit. Only the benchmark-only options (--benchmark_ledger_json, --scratch_dir) are left out.
CELLS["refine_k1_50k256_full"] = {  # noise1_k1_50k256_autorefine_resident_ef6234c
    "module": "relax.commands.refine",
    "args": [
        *("--data_dir", "{fx:k1_50k256}", "--output", "{out}", "--max_iter", "999", "--healpix_order", "3"),
        *("--offset_range", "3.0", "--offset_step", "1.0", "--adaptive_oversampling", "1", "--tau2_fudge", "1.0"),
        *("--perturb_factor", "0.5", "--particle_diameter_ang", "200", "--firstiter_cc", "--apply-initial-lowpass"),
        *("--init_resolution", "30.0", "--seed", "1775735620"),
    ],
    "box": 256,
}
CELLS["class3d_k4_50k256_it15"] = {  # pdb_k4_50k_class3d_15it_resident_bench
    "module": "relax.commands.class3d",
    "args": [
        *("--data_dir", "{fx:k4_50k256}", "--output", "{out}", "--n_classes", "4", "--max_iter", "15"),
        *("--healpix_order", "1", "--offset_range", "6", "--offset_step", "2", "--init_resolution", "30.0"),
        *("--tau2_fudge", "4.0", "--ref_star", "{fx:k4_50k256}/reference_init_classes_relion.star"),
        *("--particle_diameter_ang", "200", "--seed", "1775735620", "--firstiter_cc", "--apply-initial-lowpass"),
    ],
    "box": 256,
}
CELLS["refine_ms2_448_s1"] = {  # synth_ms2_icos_448_b40_autorefine, seed 1
    "module": "relax.commands.refine",
    "args": [
        *("--data_dir", "{fx:ms2_448}", "--output", "{out}"),
        *("--init_volume", "{fx:ms2_448}/reference_init_lp20_relion.mrc", "--relion-half-sets-from-input"),
        *("--n_classes", "1", "--sym", "I2", "--healpix_order", "3", "--auto_local_healpix_order", "4"),
        *("--offset_range", "5", "--offset_step", "2", "--offset_sigma_angstrom", "10", "--adaptive_oversampling", "1"),
        *("--max_significants", "-1", "--tau2_fudge", "1", "--perturb_factor", ".5", "--seed", "1"),
        *("--perturb_seed", "1", "--init_resolution", "20", "--firstiter_cc", "--apply-initial-lowpass"),
        *("--particle_diameter_ang", "320", "--width_mask_edge_px", "5", "--preread_images", "--max_iter", "50"),
    ],
    "box": 448,
}
CELLS["tomo_refine_w2_09_box192"] = {  # cryoet_w2_09_box192_2000_192_tomo_autorefine_bb9cae4, seed 1
    "module": "relax.commands.refine",
    "args": [
        *("--data_dir", "{fx:w2_09_box192}", "--output", "{out}"),
        *("--init_volume", "{fx:w2_09_box192}/reference_init_relion.mrc", "--init_resolution", "40"),
        *("--particle_diameter_ang", "240", "--sym", "C1", "--healpix_order", "2", "--auto_local_healpix_order", "4"),
        *("--offset_range", "5", "--offset_step", "1", "--seed", "1", "--no-firstiter_cc"),
    ],
    "box": 192,
}
CELLS["tomo_class3d_et13_it25"] = dict(CELLS["tomo_class3d_et13_it3"])  # cryoet_et13_k2conf_..._tomo_class3d, seed 1
CELLS["tomo_class3d_et13_it25"]["args"] = [
    "25" if previous == "--max_iter" else value
    for previous, value in zip([None, *CELLS["tomo_class3d_et13_it3"]["args"]], CELLS["tomo_class3d_et13_it3"]["args"])
]

# multioptics_k2_10k128_class3d_25it, seed 1 (two optics groups on two image shapes; robustness cell 12b)
CELLS["class3d_multioptics_k2_10k128"] = {
    "module": "relax.commands.class3d",
    "args": [
        *("--data_dir", "{fx:multioptics_k2_10k128}", "--output", "{out}"),
        *("--init_volume", "{fx:multioptics_k2_10k128}/reference_init_relion.mrc", "--n_classes", "2"),
        *("--init_resolution", "40", "--particle_diameter_ang", "200", "--healpix_order", "2", "--offset_range", "5"),
        *("--offset_step", "1", "--tau2_fudge", "4", "--max_iter", "25", "--seed", "1", "--firstiter_cc"),
    ],
    "box": 128,
}

# Robustness cell 12b: etoptics' cell-10 Refine3D K=1 with anisotropic magnification (cell10_20261006/cell10.sbatch, seed 42).
CELLS["refine_k1_mag_10k256"] = {
    "module": "relax.commands.refine",
    "args": [
        *("--data_dir", "{fx:optics_mag_k1}", "--output", "{out}", "--max_iter", "999", "--healpix_order", "2"),
        *("--auto_local_healpix_order", "4", "--offset_range", "5", "--offset_step", "2"),
        *("--adaptive_oversampling", "1"),
        *("--init_resolution", "60", "--perturb_seed", "42", "--relion-half-sets-from-input"),
        *("--particle_diameter_ang", "380", "--tau2_fudge", "1", "--firstiter_cc", "--apply-initial-lowpass"),
        *("--sym", "C1"),
        *("--init_volume", "{fx:optics_mag_k1}/reference_init_class001_relion.mrc", "--seed", "42"),
    ],
    "box": 256,
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
