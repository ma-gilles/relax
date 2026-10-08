#!/usr/bin/env python
"""Fingerprint what the VDAM InitialModel controller does, to check that a refactor moved code and nothing else.

    python scripts/dev/vdam_fingerprint.py run OUT.json [--rev REV | --source DIR] [CASE ...]
    python scripts/dev/vdam_fingerprint.py diff A.json B.json          # exit 1 unless only log rows differ
    python scripts/dev/vdam_fingerprint.py check BASE_REV [HEAD_REV]   # run both (HEAD: the worktree) and diff
    python scripts/dev/vdam_fingerprint.py cases                       # the case list, one line each
    python scripts/dev/vdam_fingerprint.py selftest [MUTATION ...]     # perturb the controller; every mutation must show

The VDAM counterpart of ``scripts/dev/fingerprint.py`` (the refinement controller's harness), with the same
commands, output format and comparison rules; the pure logic (flattening, operand digests, the diff and its
accepted classes, mutations) is imported from it, and the command line is ``scripts/dev/fingerprint_cli.py``.
``run`` calls the command entry, ``relax initial_model`` (``relax.commands.initial_model.main``), on a written
data directory of 24 images of 24 pixels for a few iterations, and records per case: the state
``run_native_initial_model`` returns, every file the run wrote, the model state at every model-STAR write (the
checkpoints), and one ordered trace of log records, printed lines and engine calls. Everything above the
E-step engine runs as in production on the CPU: the start-up reference, the expected-accuracy estimate, the
projector, the subset schedule, the VDAM and momentum-SGD M-steps, the noise and prior updates and the output
writers.

The E-step engine (``run_dense_k_class_em_adaptive``) runs only on a GPU, so it is replaced on both sides by
one stand-in whose output is seeded by a hash of every operand it receives: a changed operand anywhere
upstream changes every later array of the run. The comparison is exact because both sides run the same
arithmetic on the same CPU; it is a check for move-only commits, not a merge gate for numerical changes.
Printed lines are log rows (rule 2): their digits are masked, so a timing report shows as its template.

NOT covered (use the GPU test tiers): the real E-step engine and its numbers; subtomograms (``--ios``);
optics groups on several image shapes; the diagnostic optimiser continuation; RELION's CUDA image
preprocessing; GPU operation order, peak memory and array lifetimes.
"""

from __future__ import annotations

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
    diff_fingerprints,
    differing_cases,
    digest_operands,
    flatten,
    log_row,
    mutated_tree,
)

NOT_COVERED = (
    "the real E-step engine and its numbers (a stand-in seeded by its operands replaces run_dense_k_class_em_adaptive)",
    "subtomogram InitialModel (--ios) and optics groups on several image shapes",
    "the diagnostic optimiser continuation (--diagnostic-continue-optimiser) and iteration reference replay",
    "RELION's CUDA image preprocessing (the cases use --image-fourier-backend host_numpy)",
    "GPU operation order, peak memory and array lifetimes",
)
# Keys of a written JSON file that hold wall times.
TIMING_WORDS = ("time", "wall", "seconds", "elapsed")


# ----------------------------------------------------------------------------- cases


def _cases() -> dict[str, tuple[str, dict]]:
    """``{name: (description, run_case keywords)}``; every case runs the command entry on the tiny data."""
    cases: dict[str, tuple[str, dict]] = {}

    def add(name, description, arguments=(), **keywords):
        assert name not in cases, name
        cases[name] = (description, dict(arguments=list(arguments), **keywords))

    add("k1_default", "K=1, VDAM, float32 M-step, solvent flattening, six iterations")
    add("k2_default", "K=2, VDAM", ["--K", "2"])
    add("k1_mstep_f64", "K=1 with the float64 diagnostic M-step", ["--mstep-compute-dtype", "float64"])
    add("k1_no_solvent", "K=1 without solvent flattening", ["--no-solvent"])
    add("k2_uniform_prior", "K=2 with the uniform class/direction prior", ["--K", "2", "--uniform-class-direction-prior"])
    add("k1_fourier_schedule", "K=1 with an explicit Fourier radius schedule", ["--fourier-radius-schedule", "3x2,4x2,5x1,6x1"])
    add("k2_momentum_sgd", "K=2 momentum SGD on a fixed grid and a radius schedule",
        ["--K", "2", "--optimizer", "momentum_sgd", "--oversampling", "0", "--fixed-healpix-order", "1",
         "--fourier-radius-schedule", "3x2,4x2,5x1,6x1", "--sgd-learning-rate", "0.5"])
    add("k1_stochastic", "K=1 stochastic batches in every iteration (pilot controls)",
        ["--stochastic-all-iterations", "--stochastic-batch-size", "10", "--grad-em-iters", "0"])
    add("k1_pilot_caps", "K=1 with pilot caps on the Fourier radius and the HEALPix order",
        ["--max-fourier-radius", "4", "--max-healpix-order", "1"])
    add("k1_em_tail", "K=1, eight iterations, the last two without gradients (EM tail)",
        ["--nr-iter", "8", "--grad-em-iters", "2"])
    add("k1_write_every_2", "K=1 writing every second iteration", ["--grad-write-iter", "2"])
    add("k1_no_artifacts", "K=1 writing only the final outputs", ["--no-write-iter-artifacts"])
    add("k1_seed_zero", "K=1 with random seed 0 (no particle shuffle)", ["--random-seed", "0"])
    add("k1_pad2", "K=1 with padding factor 2", ["--padding-factor", "2"])
    add("k1_optics_groups", "K=1 on two optics groups (per-group noise)", optics_groups=2)
    add("k1_profile", "K=1 with the stage profile (RECOVAR_INITIAL_MODEL_PROFILE)",
        env={"RECOVAR_INITIAL_MODEL_PROFILE": "1"})
    add("k1_clear_caches", "K=1 clearing the JAX caches every iteration", env={"RELAX_CLEAR_JAX_CACHES_PER_ITER": "1"})
    add("k1_skip_accuracy", "K=1 skipping the expected-accuracy estimate (diagnostic)",
        env={"RELAX_INITIALMODEL_SKIP_EXPECTED_ACCURACY": "1"})
    add("k1_adaptive_fraction", "K=1 with the adaptive fraction of the environment (RELAX_ADAPTIVE_FRACTION)",
        env={"RELAX_ADAPTIVE_FRACTION": "0.99"})
    add("k1_no_subtract_reference", "K=1 backprojecting the images, not the residuals "
        "(RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE)", env={"RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE": "1"})
    add("k1_refused_float64_scoring", "the float64 scoring switch is refused (the resident E-step is float32)",
        env={"RELAX_USE_FLOAT64_SCORING": "1"})
    add("k1_refused_nan_tau2_fudge", "four VDAM iterations are refused: RELION's tau2 fudge is NaN in iteration 1",
        ["--nr-iter", "4"])
    add("k1_refused_sgd_oversampling", "momentum SGD with oversampling 1 is refused",
        ["--optimizer", "momentum_sgd", "--fixed-healpix-order", "1", "--fourier-radius-schedule", "3x2,4x2,5x1,6x1"])
    return cases


CASES = _cases()

# A deliberately wrong controller for the self-test: (name, old lines, new lines, what breaks, detected).
# A target must occur in exactly one place of the tree: when a statement is rewritten, update its entry.
MUTATIONS = (
    ("noise_blend", "new_sigma2[g] * my_mu + (1.0 - my_mu) * np.asarray(wsum_g, dtype=np.float64) / float(shape[0] ** 4),",
     "new_sigma2[g] * my_mu + (1.0 - my_mu) * np.asarray(wsum_g, dtype=np.float64) / float(shape[0] ** 4) * 2,",
     "the noise update doubles the new spectrum", True),
    ("pdf_class_blend", "new_pdf_class += (1.0 - my_mu) * class_sums / sum_weight",
     "new_pdf_class += (1.0 - my_mu) * class_sums[::-1] / sum_weight",
     "the class prior update reads the class sums reversed", True),
    ("ave_pmax_halved", "current = replace(current, ave_Pmax=float(ave_pmax))",
     "current = replace(current, ave_Pmax=0.5 * float(ave_pmax))",
     "the loop installs half the average Pmax", True),
    ("stall_counter", "sampling_state.nr_iter_wo_resol_gain += 1", "sampling_state.nr_iter_wo_resol_gain += 2",
     "the resolution-stall counter counts twice", True),
    ("solvent_edge", "mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / width)",
     "mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / (2 * width))",
     "the solvent mask's soft edge is stretched", True),
    ("tau2_fudge_doubled", "tau2_fudge_factor=current.tau2_fudge_factor,", "tau2_fudge_factor=2 * current.tau2_fudge_factor,",
     "the VDAM M-step receives twice the tau2 fudge", True),
    ("resolution_minres", "minres_map=regularization_relion.RELION_MINRES_MAP,",
     "minres_map=regularization_relion.RELION_MINRES_MAP - 1,",
     "the current resolution starts one shell early", True),
    ("schedule_radius", "current_resolution_shell=radius,", "current_resolution_shell=radius - 1,",
     "a scheduled Fourier radius installs the resolution shell one small", True),
    ("stochastic_batch", "batch_size = min(int(pilot_controls.stochastic_batch_size), int(nr_particles))",
     "batch_size = min(int(pilot_controls.stochastic_batch_size), int(nr_particles)) - 1",
     "the stochastic batch is one particle short", True),
    ("sgd_learning_rate", "return sgd_m_step(current, accumulators, learning_rate=self.learning_rate,", "return sgd_m_step(current, accumulators, learning_rate=2 * self.learning_rate,",
     "momentum SGD steps with twice the learning rate", True),
    ("retained_fraction", "meta[\"class_retained_mass_fraction_by_class\"] = (retained_class_sums / retained_mass).tolist()",
     "meta[\"class_retained_mass_fraction_by_class\"] = (retained_class_sums / (2 * retained_mass)).tolist()",
     "the reported retained class fractions are halved", True),
    ("uniform_prior_report", "meta[\"effective_joint_direction_prior_per_class_direction\"] = 1.0 / float(current.K * n_directions)",
     "meta[\"effective_joint_direction_prior_per_class_direction\"] = 2.0 / float(current.K * n_directions)",
     "the uniform-prior report doubles the joint prior", True),
    ("subtract_switch_ignored", "engine_kwargs[\"reconstruction_subtract_projected_reference\"] = False", "pass",
     "RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE is ignored", True),
    ("adaptive_fraction_ignored", "engine_kwargs[\"adaptive_fraction\"] = adaptive_fraction", "pass",
     "RELAX_ADAPTIVE_FRACTION is ignored", True),
    ("float64_scoring_admitted", "if env.get(\"RELAX_USE_FLOAT64_SCORING\", \"\").strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}:", "if False:",
     "RELAX_USE_FLOAT64_SCORING is admitted and silently ignored", True),
    ("nan_tau2_fudge_admitted", "if not math.isfinite(compute_tau2_fudge(it, phases, True, 3, tau2_fudge_arg=self.tau2_fudge))",
     "if False", "a VDAM schedule with a NaN tau2 fudge is admitted", True),
    ("write_cadence", "return (iteration % grad_write_iter) == 0 or iteration == nr_iter",
     "return (iteration % grad_write_iter) == 1 or iteration == nr_iter",
     "the iteration files are written at the wrong cadence", True),
    ("optics_group_noise", "wsum_g = noise_relion.normalize_wsum_to_sigma2_noise(wsum_rows[g], power_rows[g], float(sumw_rows[g]), shape, apply_floors=False)",
     "wsum_g = noise_relion.normalize_wsum_to_sigma2_noise(wsum_rows[0], power_rows[0], float(sumw_rows[0]), shape, apply_floors=False)",
     "every optics group's noise is updated from group 1's sums", True),
)


# ----------------------------------------------------------------------------- the worker (imports the tree)


def _worker(source: str, out_path: str, tmp_root: str, names: list[str]) -> None:
    """Run the cases against ``source`` in this process; the caller has set the CPU-only environment."""
    import contextlib
    import io
    import logging
    import threading
    import traceback

    sys.dont_write_bytecode = True
    sys.path[:0] = [source, os.path.join(source, "tests")]
    os.chdir(tmp_root)

    import jax.numpy as jnp
    import mrcfile
    import numpy as np
    import pandas as pd
    import recovar
    import starfile
    from helpers.fake_adaptive_engine import adaptive_result

    import relax
    from relax.commands import initial_model as command
    from relax.vdam import adaptive_estep, driver

    assert relax.__file__.startswith(source), relax.__file__

    tmp_pattern = re.compile(re.escape(tmp_root) + r"(/\w+)?")
    run_paths = [(re.escape(out_path), "<OUT>"), (re.escape(str(Path(tmp_root).parent)), "<WORK>"),
                 (re.escape(source), "<SRC>")]

    def scrub(text):
        text = tmp_pattern.sub(TMP_TOKEN, text)
        for pattern, token in run_paths:
            text = re.sub(pattern, token, text)
        return text

    trace: list = []

    def record(*row):
        trace.append(list(row))

    class Capture(logging.Handler):
        def emit(self, log_record):
            template = str(log_record.msg)
            try:
                text = log_record.getMessage()
            except Exception:  # a malformed format string is itself behaviour
                text = "<unformattable> " + template
            if threading.current_thread() is threading.main_thread():
                trace.append(log_row(log_record.name, log_record.levelname, template, text, scrub=scrub))

    class PrintCapture(io.TextIOBase):
        """Printed lines become log rows with their digits masked (stage reports print wall times)."""

        def __init__(self):
            self.pending = ""

        def write(self, text):
            self.pending += text
            *lines, self.pending = self.pending.split("\n")
            for line in lines:
                if line.strip():
                    trace.append(["log", "stdout", "PRINT", re.sub(r"[0-9][0-9.e+-]*", "#", scrub(line)), ""])
            return len(text)

    def stand_in_engine(experiment_dataset, means, mean_variance, noise_variance, coarse_rotations,
                        coarse_translations, fine_rotations, fine_translations, rot_parent_map, trans_parent_map,
                        disc_type, **kwargs):
        """CPU stand-in for run_dense_k_class_em_adaptive: a result seeded by every operand it receives."""
        seed = digest_operands(means, mean_variance, noise_variance, coarse_rotations, coarse_translations,
                               fine_rotations, fine_translations, rot_parent_map, trans_parent_map, disc_type, kwargs)
        record("call", "engine", f"seed={seed:016x}", "kwargs=" + ",".join(sorted(kwargs)))
        rng = np.random.default_rng(seed)
        n_classes = int(np.asarray(means).shape[0])
        n_images = int(experiment_dataset.n_units)
        n_fine = int(np.asarray(fine_rotations).shape[0])
        n_trans = int(np.asarray(fine_translations).shape[0])
        groups = int(kwargs.get("reconstruction_group_count") or 1)
        # The pass's BackProjector cube at the current size (the layout the resident engine returns).
        padding = int(kwargs.get("reconstruction_padding_factor", 1))
        edge = 2 * (int(padding * (int(kwargs["current_size"]) // 2) + 0.5) + 1) + 1
        shape = (n_classes, groups, edge**3) if groups > 1 else (n_classes, edge**3)
        n_shells = int(experiment_dataset.image_shape[0]) // 2 + 1
        optics = kwargs.get("optics_group_ids")
        n_groups = 1 if optics is None else int(np.max(np.asarray(optics))) + 1

        def noise_fields(n):
            del n
            return {}

        result = adaptive_result(
            experiment_dataset, means, fine_rotations, kwargs,
            Ft_y=jnp.asarray((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)),
            Ft_ctf=jnp.asarray((1.0 + rng.random(shape)).astype(np.complex64)),
            max_posterior=0.2 + 0.8 * rng.random(n_images),
            pose_assignments=rng.integers(0, max(n_fine * n_trans, 1), size=n_images),
            best_pose_translations=rng.integers(-1, 2, size=(n_images, 2)).astype(np.float32),
            significant_counts=rng.integers(1, 5, size=n_images),
            sigma2_offset=float(rng.random()),
            noise_fields=noise_fields,
        )
        per_class = tuple(
            stats._replace(rotation_posterior_sums=jnp.asarray(rng.random(n_fine) * (k + 1), dtype=jnp.float32))
            for k, stats in enumerate(result.per_class_stats)
        )
        resp = rng.random((n_classes, n_images))
        resp /= resp.sum(axis=0, keepdims=True)
        full_sums = resp.sum(axis=1)
        noise = result.aggregate_noise_stats
        if n_groups > 1:
            noise = noise._replace(
                wsum_sigma2_noise=jnp.asarray(1.0 + rng.random((n_groups, n_shells)), dtype=jnp.float32),
                wsum_img_power=jnp.asarray(2.0 + rng.random((n_groups, n_shells)), dtype=jnp.float32),
                sumw=jnp.asarray(1.0 + rng.random(n_groups), dtype=jnp.float32),
            )
        else:
            noise = noise._replace(
                wsum_sigma2_noise=jnp.asarray(1.0 + rng.random(n_shells), dtype=jnp.float32),
                wsum_img_power=jnp.asarray(2.0 + rng.random(n_shells), dtype=jnp.float32),
                sumw=float(n_images),
            )
        return result._replace(
            per_class_stats=per_class, stats=per_class[0],
            class_assignments=jnp.asarray(rng.integers(0, n_classes, size=n_images), dtype=jnp.int32),
            class_responsibilities=jnp.asarray(resp, dtype=jnp.float32),
            class_posterior_sums=jnp.asarray(full_sums, dtype=jnp.float32),
            class_mstep_posterior_sums=jnp.asarray(full_sums * (0.8 + 0.2 * rng.random(n_classes)), dtype=jnp.float32),
            aggregate_noise_stats=noise,
            best_pose_rotations=jnp.asarray(np.linalg.qr(rng.standard_normal((n_images, 3, 3)))[0], dtype=jnp.float32),
        )

    def write_tiny_dataset(root, *, n_images=24, box=24, optics_groups=1, seed=5):
        """A RELION particle STAR with its stack, as a data directory."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        pixel = 2.0
        with mrcfile.new(root / "particles.mrcs") as stack:
            stack.set_data(rng.standard_normal((n_images, box, box)).astype(np.float32))
            stack.voxel_size = pixel
        groups = np.arange(1, optics_groups + 1)
        optics = pd.DataFrame({
            "rlnOpticsGroup": groups, "rlnOpticsGroupName": [f"opticsGroup{g}" for g in groups],
            "rlnAmplitudeContrast": np.full(optics_groups, 0.07), "rlnSphericalAberration": np.full(optics_groups, 2.7),
            "rlnVoltage": np.full(optics_groups, 300.0), "rlnImagePixelSize": np.full(optics_groups, pixel),
            "rlnImageSize": np.full(optics_groups, box), "rlnImageDimensionality": np.full(optics_groups, 2),
        })
        rows = np.arange(n_images)
        particles = pd.DataFrame({
            "rlnImageName": [f"{i + 1}@particles.mrcs" for i in rows],
            "rlnMicrographName": [f"mic{i % 3}" for i in rows],
            "rlnDefocusU": 15000.0 + 100.0 * rows, "rlnDefocusV": 15100.0 + 100.0 * rows,
            "rlnDefocusAngle": np.full(n_images, 10.0), "rlnPhaseShift": np.zeros(n_images),
            "rlnOpticsGroup": 1 + rows % optics_groups,
            "rlnAngleRot": rng.uniform(-180.0, 180.0, n_images), "rlnAngleTilt": rng.uniform(0.0, 180.0, n_images),
            "rlnAnglePsi": rng.uniform(-180.0, 180.0, n_images),
            "rlnOriginXAngst": np.zeros(n_images), "rlnOriginYAngst": np.zeros(n_images),
        })
        star = root / "particles.star"
        starfile.write({"optics": optics, "particles": particles}, star)
        star.write_text("".join(line for line in star.read_text().splitlines(True) if not line.startswith("# Created")))
        return star

    def hash_output_files(base, out):
        """Every file under ``base``: maps by data, JSON without wall times, text with its creation stamp dropped."""
        for path in sorted(Path(base).rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(base))
            if path.suffix in (".mrc", ".mrcs"):
                with mrcfile.open(path, permissive=True) as volume:
                    flatten(np.asarray(volume.data), rel, out, scrub=scrub)
            elif path.suffix == ".json":
                payload = json.loads(path.read_text())

                def drop_timing(value):
                    if isinstance(value, dict):
                        return {k: drop_timing(v) for k, v in value.items()
                                if not any(word in str(k) for word in TIMING_WORDS)}
                    return value

                flatten(drop_timing(payload), rel, out, scrub=scrub)
            elif path.suffix in (".star", ".txt", ".log"):
                lines = [line for line in path.read_text().splitlines() if not line.startswith("# Created")]
                out[rel] = "sha256:" + hashlib.sha256(scrub("\n".join(lines)).encode()).hexdigest()
            else:
                out[rel] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return out

    def modules_with(name):
        return [module for module_name, module in sorted(sys.modules.items())
                if module_name.startswith("relax") and callable(getattr(module, name, None))]

    def run_case(arguments, *, env=None, optics_groups=1):
        case_root = Path(tempfile.mkdtemp(dir=tmp_root, prefix="case"))
        star = write_tiny_dataset(case_root / "data", optics_groups=optics_groups)
        outdir = case_root / "out"
        # Six iterations: with four or five, RELION's tau2-fudge schedule is NaN in iteration 1.
        argv = ["--i", str(star), "--o", str(outdir / "run"), "--nr-iter", "6", "--grad-write-iter", "1",
                "--random-seed", "7", "--particle-diameter", "40", "--image-batch-size", "8",
                "--bootstrap-min-particles", "8", "--sigma2-min-particles", "8",
                "--image-fourier-backend", "host_numpy", "--no-jax-compilation-cache", "--no-require-custom-cuda", *arguments]
        trace.clear()
        checkpoints: dict = {}
        captured: dict = {}
        patches = []

        def patch(owner, name, replacement):
            patches.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

        patch(adaptive_estep, "run_dense_k_class_em_adaptive", stand_in_engine)
        original_run = driver.run_native_initial_model

        def run_and_keep(opts):
            result = original_run(opts)
            captured["result"] = result
            return result

        patch(driver, "run_native_initial_model", run_and_keep)
        for module in modules_with("_write_model_star"):
            original_write = module._write_model_star

            def write_and_record(path, state, class_mrcs, _original=original_write):
                checkpoints[f"{Path(path).name}"] = flatten(state, scrub=scrub)
                return _original(path, state, class_mrcs)

            patch(module, "_write_model_star", write_and_record)
        saved_env = {key: os.environ.get(key) for key in (env or {})}
        os.environ.update(env or {})
        handler = Capture()
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        status = "ok"
        try:
            with contextlib.redirect_stdout(PrintCapture()):
                code = command.main(argv)
            status = f"ok rc={code}"
        except BaseException as error:  # a refusal is behaviour; record it
            status = f"{type(error).__name__}: {scrub(str(error))}"
            if not isinstance(error, (ValueError, NotImplementedError, SystemExit, RuntimeError)):
                status += "\n" + scrub(traceback.format_exc())
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            for owner, name, original in reversed(patches):
                setattr(owner, name, original)
        result = captured.get("result")
        flat_result = {} if result is None else flatten(result, scrub=scrub)
        files = hash_output_files(outdir, {}) if outdir.exists() else {}
        flat_checkpoints = {f"{name}{key}": value for name, leaves in checkpoints.items() for key, value in leaves.items()}
        return {"status": {"": status}, "result": flat_result, "files": files, "checkpoints": flat_checkpoints,
                "trace": [list(map(str, row)) for row in trace]}

    results = {}
    for name in names:
        _, keywords = CASES[name]
        results[name] = run_case(keywords["arguments"], env=keywords.get("env"),
                                 optics_groups=keywords.get("optics_groups", 1))
        print(f"{name}: {results[name]['status']['']}", flush=True)
    fingerprint = {"schema": 1, "source": scrub(source), "cases": results}
    Path(out_path).write_text(json.dumps(fingerprint, sort_keys=True))
    print(f"{len(results)} cases from {relax.__file__} (recovar {recovar.__file__}) -> {out_path}")


# ----------------------------------------------------------------------------- commands


HARNESS = fingerprint_cli.Harness(
    script=Path(__file__).resolve(),
    description=__doc__.split("\n\n")[0],
    cases=CASES,
    mutations=MUTATIONS,
    not_covered=NOT_COVERED,
    worker=_worker,
    diff_fingerprints=diff_fingerprints,
    accepted=accepted,
    differing_cases=differing_cases,
    mutated_tree=mutated_tree,
    file_prefix="v",
    temp_prefix="relax-vdam-fingerprint-",
    case_width=30,
    cases_note="",
    # The child runs with the harness's own switches only: no relax or InitialModel variable of the caller.
    dropped_env_prefixes=("RELAX_", "RECOVAR_INITIAL"),
    jax_compilation_cache=False,
)


def main(argv: list[str] | None = None) -> int:
    return fingerprint_cli.main(HARNESS, argv)


if __name__ == "__main__":
    sys.exit(main())
