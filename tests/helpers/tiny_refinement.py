"""A tiny K=1 / K-class refinement on the CPU stand-in engine, for tests of what the controller hands its sites.

``run_tiny_refinement`` runs ``refine_single_volume`` on 8-pixel mock half sets with
``helpers.fake_adaptive_engine`` in place of the global E-step (its pass 2 needs a GPU), through the
numbered iterations and, for K=1, the final all-data pass. ``record_calls`` wraps a callee so a test
can read the arguments each call received; ``CallTrace`` records the order of calls to several callees and
which of them were running at each call, so a test can check a sequence or an owner without reading source.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace

import jax.numpy as jnp
import numpy as np
from helpers.em_arrays import _hermitian_volume
from helpers.fake_adaptive_engine import install_fake_adaptive_engine
from helpers.run_options import stand_in

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
N_IMAGES = 10
N_ROTATIONS = 5


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    h, w = IMAGE_SHAPE if image_shape is None else image_shape
    return jnp.ones((params.shape[0], h * (w // 2 + 1) if half_image else h * w), dtype=jnp.float32)


def _identity_process(batch, apply_image_mask=False):
    return batch


def _identity_process_half(batch, apply_image_mask=False):
    from recovar.core import fourier_transform_utils as ftu

    return ftu.full_image_to_half_image(batch, IMAGE_SHAPE)


class MockHalfSet:
    """The subset of the dataset API the refinement loop reads, over random Hermitian images."""

    particles_file = None  # built in memory: no RELION optics table

    def __init__(self, n_images, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_identity_process)
        self.process_images_half = staticmethod(_identity_process_half)
        self.premultiplied_ctf = False
        real = rng.standard_normal((n_images, *IMAGE_SHAPE)).astype(np.float32)
        self._images = np.fft.fftshift(np.fft.fft2(real), axes=(-2, -1)).reshape(n_images, -1).astype(np.complex64)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)

        class _ImageSource:
            process_images = staticmethod(_identity_process)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        indices = np.arange(self.n_images) if indices is None else np.asarray(indices)
        for start in range(0, len(indices), max(1, batch_size)):
            idx = indices[start : start + max(1, batch_size)]
            yield (
                jnp.asarray(self._images[idx]),
                self.rotation_matrices[idx],
                self.translations[idx],
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def update_poses(self, rots, trans):
        self.rotation_matrices = np.asarray(rots)
        self.translations = np.asarray(trans)

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def original_image_indices_from_local(self, indices=None):
        return np.arange(self.n_images, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)


def record_calls(monkeypatch, owner, name):
    """Replace ``owner.name`` by a recorder that forwards to it; returns the ``(args, kwargs)`` list."""

    calls = []
    original = getattr(owner, name)

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, recorder)
    return calls


@dataclass
class TracedCall:
    """One call seen by a ``CallTrace``: its label, operands, result, and the traced callees then running."""

    label: str
    args: tuple
    kwargs: dict
    inside: tuple
    result: object = None


@dataclass
class CallTrace:
    """Ordered calls to the callees ``wrap`` replaced, across threads.

    ``inside`` of each call holds the labels of the traced callees running in the same thread when it
    started, outermost first: ``calls("b")[0].inside == ("a",)`` says ``b`` ran inside ``a``.
    """

    monkeypatch: object
    calls_seen: list = field(default_factory=list)
    _running: threading.local = field(default_factory=threading.local)

    def wrap(self, owner, name, label=None, *, before=None, after=None, keep_operands=True):
        """Replace ``owner.name`` by a forwarder recorded under ``label`` (default ``name``).

        ``before(call)`` runs before the original (it may assert on the operands or the program state);
        ``after(call)`` runs once ``call.result`` is set. ``keep_operands=False`` drops the operands and
        the result from the record once ``after`` has run, for a test of their lifetime.
        """

        original = getattr(owner, name)
        label = name if label is None else label

        def traced(*args, **kwargs):
            running = self._running.__dict__.setdefault("labels", [])
            call = TracedCall(label, args, kwargs, tuple(running))
            self.calls_seen.append(call)
            if before is not None:
                before(call)
            running.append(label)
            try:
                call.result = original(*args, **kwargs)
            finally:
                running.pop()
            if after is not None:
                after(call)
            result = call.result
            if not keep_operands:
                call.args, call.kwargs, call.result = (), {}, None
            return result

        self.monkeypatch.setattr(owner, name, traced)
        return self

    def labels(self, *wanted):
        """The labels in call order, restricted to ``wanted`` when given."""

        return [call.label for call in self.calls_seen if not wanted or call.label in wanted]

    def calls(self, label):
        return [call for call in self.calls_seen if call.label == label]


def frame_holds(function, value) -> bool:
    """Whether a running call of ``function`` (unwrapped) in this thread has a local bound to ``value``.

    Compares by identity, not by name, so a test of a released buffer does not pin the local's name.
    """

    import sys

    code = getattr(function, "__wrapped__", function).__code__
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is code:
            return any(local is value for local in frame.f_locals.values())
        frame = frame.f_back
    raise AssertionError(f"{function.__qualname__} is not running")


def write_replay_dir(root, *, max_iter, n_classes=1, prior_order=2):
    """RELION run files for STAR replay under ``root``: ``run_itNNN_sampling.star`` for iterations 0 to
    ``max_iter + 1`` and, after each numbered iteration, the model STAR with a direction prior at
    ``prior_order`` (per half for K=1). Returns ``root`` as a string."""

    from pathlib import Path

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for it in range(0, max_iter + 2):
        (root / f"run_it{it:03d}_sampling.star").write_text(
            "data_sampling_general\n\n_rlnHealpixOrder 2\n_rlnPsiStep 15.0\n_rlnOffsetRange 10.0\n"
            "_rlnOffsetStep 2.0\n_rlnSamplingPerturbInstance 0.25\n_rlnSamplingPerturbFactor 0.5\n"
        )
    n_pix = 12 * 4**prior_order
    for it in range(1, max_iter + 1):
        for half in [None] if n_classes > 1 else [1, 2]:
            values = np.random.default_rng(400 + it + (half or 0)).random((n_classes, n_pix)) + 0.05
            values /= values.sum(axis=1, keepdims=True)
            text = "data_model_general\n\n_rlnCurrentImageSize 4\n_rlnCurrentResolution 0.1\n\n"
            for c in range(n_classes):
                text += f"data_model_pdf_orient_class_{c + 1}\n\nloop_\n_rlnOrientationDistribution #1\n"
                text += "".join(f"{float(v)!r}\n" for v in values[c]) + "\n"
            name = f"run_it{it:03d}_model.star" if half is None else f"run_it{it:03d}_half{half}_model.star"
            (root / name).write_text(text)
    return str(root)


def follower_scale_replay(n_iterations, n_followers=2, replay=None):
    """``run_tiny_refinement`` arguments for Class3D's strict RELION follower-scale emulation on the tiny half
    sets: two physical groups and one optics group (``StartState``), and the input source's follower topology
    with a captured dispatch schedule for RELION iterations 1 to ``n_iterations`` and the follower-scale
    ``replay`` (None: none)."""

    from relax.parity.relion_replay_source import RelionReplay
    from relax.refinement.refinement_options import StartState
    from relax.relion.worker_scale import RELION_SCALE_REDUCTION_MODES, PreparedFollowerTopology

    n_half = N_IMAGES // 2
    owners = [np.arange(n_half) % n_followers, (np.arange(n_half) + 1) % n_followers]
    return dict(
        start=StartState(
            init_group_ids=[np.arange(n_half) % 2, (np.arange(n_half) + 1) % 2],
            init_group_count=2,
            init_relion_optics_group_count=1,
        ),
        relion_replay=RelionReplay(follower_topology=PreparedFollowerTopology(
            n_followers=n_followers, replay=replay, reduction_mode=RELION_SCALE_REDUCTION_MODES[0],
            owners_by_iteration={it: owners for it in range(1, n_iterations + 1)},
        )),
    )


def run_tiny_refinement(
    monkeypatch, *, parity=None, n_classes=1, max_iter=2, engine_calls=None, schedule=None, init_volume=None,
    final_after_max_iter=True, converge_after=None, engine_noise_fields=None, observer=None, relion_replay=None,
    **option_groups,
):
    """Run the controller for ``max_iter`` numbered iterations and (K=1) the final all-data pass.

    ``parity`` and ``schedule`` are mappings of ``RelionParityOptions`` and ``RefinementSchedule``
    fields; ``option_groups`` are further ``RefinementOptions`` groups. ``engine_calls`` receives the
    fake engine's call records. ``init_volume`` replaces the default reference (a Hermitian NumPy
    volume). ``final_after_max_iter=False`` stops K=1 at the iteration cap without a final pass.
    ``converge_after=n`` marks the state converged from the n-th convergence update on, so a K-class
    run reaches its final pass. ``engine_noise_fields(n_images)`` adds ``NoiseStats`` fields to the
    stand-in engine's statistics. ``observer`` is the run's ``RunObserver`` (None: none). ``relion_replay`` is
    what the run replays (a ``RelionReplay``); the replay fields of ``parity``
    (``perturb_replay_relion_dir``, ``perturb_replay_max_iter``, ...) are moved into it.
    """

    import relax.sampling as sampling
    from relax.parity.relion_replay_source import RelionReplay, RelionReplaySource
    from relax.refinement import iteration_loop
    from relax.refinement.half_inputs import HalfPair
    from relax.refinement.ports import RunObserver
    from relax.refinement.refinement_options import (
        FinalPassOptions,
        KClassOptions,
    )
    from relax.refinement.startup_references import StartupHandoff

    calls = [] if engine_calls is None else engine_calls

    def random_half_map(k, size):
        side = round(size ** (1.0 / 3.0))
        return _hermitian_volume((side, side, side), seed=1000 + len(calls) * 7 + k)

    install_fake_adaptive_engine(monkeypatch, calls, Ft_y=random_half_map, noise_fields=engine_noise_fields)

    def identity_rotation_grid(order, dtype=None, *, symmetry="C1"):
        n_rotations = sampling.rotation_grid_size(order, symmetry=symmetry)
        return sampling.RotationGrid(
            rotations=np.repeat(np.eye(3, dtype=np.float32)[None], n_rotations, axis=0),
            rotation_eulers=np.zeros((n_rotations, 3), dtype=np.float32),
            healpix_order=order,
            symmetry=symmetry,
        )

    monkeypatch.setattr(sampling, "relion_scoring_rotation_grid", identity_rotation_grid)
    if converge_after is not None:
        import sys

        updates = []
        for module in [module for name, module in sorted(sys.modules.items()) if name.startswith("relax.")]:
            original_update = getattr(module, "update_refinement_state", None)
            if not callable(original_update):
                continue

            def converge(*args, _original=original_update, **kwargs):
                updated = _original(*args, **kwargs)
                updates.append(None)
                if len(updates) >= converge_after:
                    updated.has_converged = True
                return updated

            monkeypatch.setattr(module, "update_refinement_state", converge)
    for name in ("RELAX_PARITY_DUMP_DIR", "RELAX_PARITY_TIMING_DIR"):
        monkeypatch.delenv(name, raising=False)
    # K=1: run the final all-data pass after the last numbered iteration without waiting for convergence
    # (RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER); the rest of the environment is read as the command does.
    option_groups["final_pass"] = replace(FinalPassOptions.from_environ(), after_max_iter=final_after_max_iter)
    rng = np.random.default_rng(42)
    halves = [MockHalfSet(N_IMAGES // 2, rng), MockHalfSet(N_IMAGES // 2, rng)]
    option_groups.setdefault("adaptive", stand_in.adaptive(adaptive_oversampling=1))
    option_groups.setdefault("execution", stand_in.execution(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS))
    if n_classes > 1:
        option_groups.setdefault(
            "k_class",
            KClassOptions(n_classes=n_classes),
        )
    parity = dict(parity or {})
    replay_fields = {name: parity.pop(name) for name in list(parity) if name.startswith("perturb_replay_")}
    if replay_fields:
        relion_replay = replace(relion_replay or RelionReplay(), **replay_fields)
    options = stand_in.options(
        schedule=stand_in.schedule(
            **{"max_iter": max_iter, "init_current_size": 4, "init_healpix_order": 2, "max_healpix_order": 2,
               **(schedule or {})}
        ),
        parity=stand_in.parity(**(parity or {})),
        **option_groups,
    )
    return iteration_loop.refine_single_volume(
        halves,
        StartupHandoff(
            HalfPair.shared(_hermitian_volume(VOLUME_SHAPE, seed=42) if init_volume is None else init_volume),
            jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        ),
        HalfPair.shared(jnp.ones(IMAGE_SIZE, dtype=jnp.float32)),
        jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        options=options,
        observer=RunObserver() if observer is None else observer,
        # The input source the command chooses for these options.
        source=RelionReplaySource.for_run(relion_replay, options),
    )


class _Reached(Exception):
    """Raised by a recorder standing in for a GPU-only engine stage."""


def engine_stage_kwargs(monkeypatch, **engine_kwargs):
    """What ``run_dense_k_class_em_adaptive`` hands its GPU-only stages, without running them.

    The coarse significance pass (Gaussian and first-iteration CC routes) and the resident pass 2
    are replaced by recorders; returns ``{"pass1": kwargs, "pass1_cc": kwargs, "pass2": kwargs}``
    for a K=1 half of three mock images with ``engine_kwargs`` added to the engine's keywords.
    """

    from relax.classification import k_class
    from relax.fine_pass import resident_pass2
    from relax.scoring import significance

    recorded = []

    def recorder(*_args, **kwargs):
        recorded.append(kwargs)
        raise _Reached

    monkeypatch.setattr(significance, "_compute_k_class_significance_batched", recorder)
    monkeypatch.setattr(resident_pass2, "compute_pass2_stats_resident", recorder)
    dataset = MockHalfSet(3, np.random.default_rng(0))
    coarse = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
    fine = np.repeat(np.eye(3, dtype=np.float32)[None], 8, axis=0)
    translations, fine_translations = np.zeros((2, 2), np.float32), np.zeros((4, 2), np.float32)
    rotation_parents, translation_parents = np.repeat(np.arange(4), 2), np.repeat(np.arange(2), 2)
    means = jnp.zeros((1, VOLUME_SIZE), jnp.complex64)
    keywords = dict(half_spectrum_scoring=True, mstep_relion_x_half=True, **engine_kwargs)
    stages = {}
    for name, cc in (("pass1", False), ("pass1_cc", True)):
        try:
            k_class.run_dense_k_class_em_adaptive(
                dataset, means, None, jnp.ones(IMAGE_SIZE), coarse, translations, fine, fine_translations,
                rotation_parents, translation_parents, "linear_interp",
                oversampling_order=1, coarse_healpix_order=0, firstiter_cc_pass2_only_best_coarse=cc, **keywords,
            )
        except _Reached:
            stages[name] = recorded[-1]
    try:
        k_class._run_sparse_k_class_adaptive_pass2(
            dataset, means, jnp.ones((1, VOLUME_SIZE)), jnp.ones(IMAGE_SIZE), coarse, translations, fine, None,
            rotation_parents, fine_translations, translation_parents, [[np.array([0], np.int32)] * 3], "linear_interp",
            class_log_priors=np.zeros(1), accumulate_noise=True, return_best_pose_details=False,
            coarse_healpix_order=0, oversampling_order=1, random_perturbation=0.0,
            engine_kwargs=dict(keywords, current_size=None),
        )
    except _Reached:
        stages["pass2"] = recorded[-1]
    return stages


def unconverged_accuracy(n_classes: int = 1):
    """An ``ExpectedAccuracy`` at RELION's 999 sentinel, for stubs that must not converge on accuracy.

    A failed estimate raises, so a stub returns this instead of raising.
    """
    import numpy as np

    from relax.sampling.expected_accuracy import ExpectedAccuracy

    return ExpectedAccuracy(
        acc_rot=999.0,
        acc_trans_angstrom=999.0,
        acc_rot_per_class=np.full(n_classes, 999.0),
        acc_trans_per_class_angstrom=np.full(n_classes, 999.0),
        class_counts=np.zeros(n_classes, dtype=np.int64),
        trial_local_indices=np.zeros(0, dtype=np.int64),
        trial_particle_ids=np.zeros(0, dtype=np.int64),
        trial_rot_per_class=np.zeros((n_classes, 0)),
        trial_trans_per_class_angstrom=np.zeros((n_classes, 0)),
    )
