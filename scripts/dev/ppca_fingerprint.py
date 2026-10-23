#!/usr/bin/env python
"""Fingerprint what relax ppca_initial_model computes, to check that a refactor moved code and nothing else.

    python scripts/dev/ppca_fingerprint.py run OUT.json [--rev REV | --source DIR] [CASE ...]
    python scripts/dev/ppca_fingerprint.py diff A.json B.json          # exit 1 unless only log rows differ
    python scripts/dev/ppca_fingerprint.py check BASE_REV [HEAD_REV]   # run both (HEAD: the worktree) and diff
    python scripts/dev/ppca_fingerprint.py cases                       # the case list, one line each
    python scripts/dev/ppca_fingerprint.py selftest [MUTATION ...]     # perturb the trainer; every mutation must show

The PPCA counterpart of ``scripts/dev/vdam_fingerprint.py``, with the same commands, output format and comparison
rules (``scripts/dev/fingerprint_cli.py``). ``run`` calls the command entry, ``relax ppca_initial_model``
(``relax.commands.ppca_initial_model.main``), on a written RELION particle STAR of 120 images of 64 pixels (RELION's
sign, protein white; the CTF in the STAR, one optics group) for three short stages, and records per case: every checkpoint array and its metadata,
every file the run wrote, and one ordered trace of log records. Nothing is replaced: the bootstrap
initialization, the start-up noise, the E-step (streamed full pose grid, host-mask dense reference or adaptive
oversampling), the VDAM and momentum-SGD updates and the final embeddings all run as in production, on the CPU.

PPCA on the GPU is not run-to-run reproducible (atomic-add backprojection in the bootstrap and the E-step
accumulation; two runs from one checkpoint differ at 1e-7 after one update); on the CPU two runs are identical,
so the comparison is exact. It is a check for move-only commits, not a merge gate for numerical changes.
Log lines are log rows (rule 2); the source identity a run records (file hashes, head) is replaced by a constant,
and the planners see 4 GiB of memory on every host.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fingerprint_cli  # noqa: E402
from fingerprint import (  # noqa: E402
    TMP_TOKEN,
    accepted,
    case_errors,
    diff_fingerprints,
    differing_cases,
    flatten,
    log_row,
    mutated_tree,
)

NOT_COVERED = (
    "the GPU kernels (relax/cuda/ppca.py, the CUDA backprojection) and GPU operation order, which is not "
    "run-to-run reproducible",
    "subtomogram particles (--ios) and several fine devices",
    "resume and stop files",
)
# The memory the harness reports to the tile and pass-2 job planners, whatever the host has.
PLAN_MEMORY_BYTES = 4 * 2**30
# Keys of a written JSON record that hold a wall time or the host's memory, by exact name (at any depth); every
# other key is compared.
HOST_KEYS = frozenset({"elapsed_seconds", "memory_gb", "memory_fraction"})

# ----------------------------------------------------------------------------- cases


def _cases() -> dict[str, tuple[str, dict]]:
    """``{name: (description, run_case keywords)}``; every case runs the command entry on the tiny particle STAR."""
    cases: dict[str, tuple[str, dict]] = {}

    def add(name, description, arguments=()):
        assert name not in cases, name
        cases[name] = (description, dict(arguments=list(arguments)))

    add("q2_vdam", "q=2, VDAM update, streamed full pose grid, three stages")
    add("q2_momentum_sgd", "q=2, momentum-SGD update", ["--optimizer", "momentum_sgd"])
    add("q2_host_mask", "q=2, host-mask dense reference engine", ["--no-stream-coarse-recompute"])
    add("q2_oversampling", "q=2, adaptive oversampling in the last stage", ["--oversampling", "1"])
    return cases


CASES = _cases()

# A deliberately wrong trainer for the self-test: (name, old lines, new lines, what breaks, detected).
# A target must occur in exactly one place of the tree: when a statement is rewritten, update its entry.
MUTATIONS = (
    ("metric_floor", "return np.float32(0.5 / int(image_size) ** 4)", "return np.float32(1.0 / int(image_size) ** 4)",
     "the coupled direction's spectral floor doubles", True),
    ("first_moment", "0.9 * moments.first + 0.1 * directions", "0.8 * moments.first + 0.2 * directions",
     "VDAM's first moment decays faster", True),
    ("second_moment", "second = jnp.where(active[:, None], 0.999 * moments.second + 0.001 * ratio, moments.second)",
     "second = jnp.where(active[:, None], 0.99 * moments.second + 0.01 * ratio, moments.second)",
     "VDAM's second moment decays faster", True),
    ("gate_fudge", "rho = jnp.where(disagreement > 0, 2 * fudge * signal / safe_disagreement, 0)",
     "rho = jnp.where(disagreement > 0, fudge * signal / safe_disagreement, 0)",
     "the half-set gate halves its signal-to-disagreement ratio", True),
    ("offset_prior", "prior = -jnp.sum(jnp.asarray(translations) ** 2, axis=-1) / (2 * state.offset_variance)",
     "prior = -jnp.sum(jnp.asarray(translations) ** 2, axis=-1) / (4 * state.offset_variance)",
     "the E-step's translation prior is twice as wide", True),
    ("initial_offset_variance", "100.0 / dataset.voxel_size**2", "50.0 / dataset.voxel_size**2",
     "the run starts with half VDAM's offset variance", True),
    ("bootstrap_images", "ids = rng.permutation(dataset.n_images)[:1000]", "ids = rng.permutation(dataset.n_images)[1:1000]",
     "the bootstrap reconstruction drops one image", True),
    ("stage_radius", "radius = min(radius, dataset.grid_size // 2 - 1)", "radius = min(radius, dataset.grid_size // 2 - 1) - 1",
     "every E-step projects one shell short", True),
)


# ----------------------------------------------------------------------------- the worker (imports the tree)


def _worker(source: str, out_path: str, tmp_root: str, names: list[str]) -> None:
    """Run the cases against ``source`` in this process; the caller has set the CPU-only environment."""
    import logging
    import threading
    import traceback

    sys.dont_write_bytecode = True
    sys.path[:0] = [source, os.path.join(source, "tests")]
    os.chdir(tmp_root)
    # The command refuses to run outside Slurm; the harness is a Slurm job or a local CPU check.
    os.environ.setdefault("SLURM_JOB_ID", "fingerprint")

    import mrcfile
    import numpy as np
    import recovar

    import relax
    from relax.commands import ppca_initial_model as command

    assert relax.__file__.startswith(source), relax.__file__

    # The tile and pass-2 job plans read the memory left to the process (on the CPU, the host's or the job's
    # cgroup limit): a fixed reading makes the plans, and so the runs, the same on every node and keeps the
    # oversampling case's pass-2 programs small (planned against the host they took 75 GB).
    from relax.ppca_refinement import full_row_tile_plan, oversampled_stream

    def fixed_memory(_stream):
        return PLAN_MEMORY_BYTES, PLAN_MEMORY_BYTES

    for module in (full_row_tile_plan, oversampled_stream):
        if hasattr(module, "available_bytes"):
            module.available_bytes = fixed_memory

    tmp_pattern = re.compile(re.escape(tmp_root) + r"(/\w+)?")
    run_paths = [(re.escape(out_path), "<OUT>"), (re.escape(str(Path(tmp_root).parent)), "<WORK>"),
                 (re.escape(source), "<SRC>")]

    def scrub(text):
        text = tmp_pattern.sub(TMP_TOKEN, text)
        for pattern, token in run_paths:
            text = re.sub(pattern, token, text)
        return text

    trace: list = []

    class Capture(logging.Handler):
        def emit(self, log_record):
            template = str(log_record.msg)
            try:
                text = log_record.getMessage()
            except Exception:  # a malformed format string is itself behaviour
                text = "<unformattable> " + template
            if threading.current_thread() is threading.main_thread():
                trace.append(log_row(log_record.name, log_record.levelname, template, text, scrub=scrub))

    def write_tiny_star(root, *, n_images=120, box=64, voxel=6.0, seed=5):
        """A RELION particle STAR of a stack of a soft blob's noisy images, in RELION's sign (protein white), one
        optics group, no angles; the CTF in the STAR."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        x = np.arange(box) - box / 2
        yy, xx = np.meshgrid(x, x, indexing="ij")
        blob = np.exp(-(xx**2 + yy**2) / 60.0)
        # The negation of the simulator-sign stack the harness wrote before PPCA read STARs (da2a70f9..01261097), so
        # the rebaselined runs are those runs' exact negations.
        images = -(
            -blob[None] * (1.0 + 0.3 * rng.standard_normal((n_images, 1, 1))) + rng.standard_normal((n_images, box, box))
        )
        with mrcfile.new(root / "particles.mrcs") as stack:
            stack.set_data(images.astype(np.float32))
            stack.voxel_size = voxel
            # mrcfile labels a new file with its creation time; without it the stack (and the run's input
            # identity, its sha256) is the same in every run.
            stack.header.nlabl = 0
            stack.header.label[:] = b""
        defocus = 12000.0 + 100.0 * np.arange(n_images)
        lines = ["data_optics", "loop_", "_rlnOpticsGroup", "_rlnOpticsGroupName", "_rlnVoltage",
                 "_rlnSphericalAberration", "_rlnAmplitudeContrast", "_rlnImagePixelSize", "_rlnImageSize",
                 "_rlnImageDimensionality", f"1 opticsGroup1 300.0 2.7 0.07 {voxel!r} {box} 2", "",
                 "data_particles", "loop_", "_rlnImageName", "_rlnOpticsGroup", "_rlnDefocusU", "_rlnDefocusV",
                 "_rlnDefocusAngle"]
        lines += [f"{i + 1}@particles.mrcs 1 {d!r} {d + 150.0!r} 20.0" for i, d in enumerate(defocus.tolist())]
        path = root / "particles.star"
        path.write_text("\n".join(lines) + "\n")
        return path

    def drop_host(value):
        if isinstance(value, dict):
            return {k: drop_host(v) for k, v in value.items() if str(k) not in HOST_KEYS}
        if isinstance(value, list):
            return [drop_host(v) for v in value]
        return value

    def hash_output_files(base, out):
        """Every file under ``base``: arrays by data, JSON records without wall times or host memory."""
        for path in sorted(Path(base).rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(base))
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as arrays:
                    for key in sorted(arrays.files):
                        value = arrays[key]
                        if key == "metadata":
                            flatten(drop_host(json.loads(str(value))), f"{rel}/{key}", out, scrub=scrub)
                        else:
                            flatten(np.asarray(value), f"{rel}/{key}", out, scrub=scrub)
            elif path.suffix == ".json":
                flatten(drop_host(json.loads(path.read_text())), rel, out, scrub=scrub)
            elif path.suffix == ".jsonl":
                rows = [drop_host(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
                flatten(rows, rel, out, scrub=scrub)
            else:
                out[rel] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return out

    def run_case(arguments):
        case_root = Path(tempfile.mkdtemp(dir=tmp_root, prefix="case"))
        particles = write_tiny_star(case_root / "data")
        outdir = case_root / "out"
        argv = ["--i", str(particles), "--particle-diameter", "240", "-o", str(outdir), "--q", "2", "--seed", "11", "--iterations", "3",
                "--stages", "[[1,6,1],[2,9,1],[3,12,2]]", "--stochastic-batch-size", "60",
                "--stochastic-all-iterations", "--image-batch-size", "20", "--rotation-block-size", "64",
                "--ppca-preread-images", "on", *arguments]
        trace.clear()
        original_identity = command.source_identity
        command.source_identity = lambda: {"fingerprint": "source identity replaced"}
        handler = Capture()
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        status = "ok"
        try:
            code = command.main(argv)
            status = f"ok rc={code}"
        except BaseException as error:  # a refusal is behaviour; record it
            status = f"{type(error).__name__}: {scrub(str(error))}"
            if not isinstance(error, (ValueError, NotImplementedError, SystemExit, RuntimeError)):
                status += "\n" + scrub(traceback.format_exc())
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)
            command.source_identity = original_identity
        files = hash_output_files(outdir, {}) if outdir.exists() else {}
        checkpoints = {key: value for key, value in files.items() if key.startswith("checkpoint_")}
        files = {key: value for key, value in files.items() if not key.startswith("checkpoint_")}
        return {"status": {"": status}, "result": {}, "files": files, "checkpoints": checkpoints,
                "trace": [list(map(str, row)) for row in trace]}

    results = {}
    for name in names:
        _, keywords = CASES[name]
        results[name] = run_case(keywords["arguments"])
        print(f"{name}: {results[name]['status']['']}", flush=True)
    fingerprint = {"schema": 1, "source": scrub(source), "cases": results}
    Path(out_path).write_text(json.dumps(fingerprint, sort_keys=True))
    print(f"{len(results)} cases from {relax.__file__} (recovar {recovar.__file__}) -> {out_path}")


# ----------------------------------------------------------------------------- commands


# The added, retired, moved and redefaulted leaves this harness accepts (fingerprint.ACCEPTED_CHANGES): (path
# pattern, class, one-line reason) per owner-accepted change; empty unless a change under review needs one.
ACCEPTED_CHANGES: tuple[tuple[str, str, str], ...] = ()

HARNESS = fingerprint_cli.Harness(
    script=Path(__file__).resolve(),
    description=__doc__.split("\n\n")[0],
    cases=CASES,
    mutations=MUTATIONS,
    not_covered=NOT_COVERED,
    worker=_worker,
    diff_fingerprints=functools.partial(
        diff_fingerprints, redefaulted_sections=("checkpoints",), accepted_changes=ACCEPTED_CHANGES
    ),
    accepted=accepted,
    case_errors=lambda fingerprint: case_errors(fingerprint, frozenset()),
    differing_cases=differing_cases,
    mutated_tree=mutated_tree,
    file_prefix="p",
    temp_prefix="relax-ppca-fingerprint-",
    case_width=20,
    cases_note="",
    # The child runs with the harness's own switches only: no relax or RECOVAR variable of the caller.
    dropped_env_prefixes=("RELAX_", "RECOVAR_"),
    jax_compilation_cache=False,
)


def main(argv: list[str] | None = None) -> int:
    return fingerprint_cli.main(HARNESS, argv)


if __name__ == "__main__":
    sys.exit(main())
