"""relax's expected-accuracy estimator against RELION's (the binding is the oracle).

The accuracies are sums of discrete step values, compared within the float band of
``helpers.float_compare`` (no test requires bitwise float equality), which a one-step
disagreement exceeds; the SNR arithmetic follows RELION's host double order.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement.tomo_half import TiltImageAccuracyInputs
from relax.sampling.relion_expected_accuracy import expected_angular_errors

bind = pytest.importorskip("relax.relion_bind._relion_bind_core")


def _smooth_volumes(rng, n_classes, size):
    grid = np.fft.fftfreq(size)
    radius2 = grid[:, None, None] ** 2 + grid[None, :, None] ** 2 + grid[None, None, :] ** 2
    volumes = []
    for _ in range(n_classes):
        noise = np.fft.fftn(rng.standard_normal((size,) * 3)) * np.exp(-radius2 / (2 * 0.08**2))
        volumes.append(np.real(np.fft.ifftn(noise)) * 50.0)
    return np.ascontiguousarray(volumes, dtype=np.float64)


def _case(rng, *, size=32, n_classes=2, n_particles=30, n_trials=12):
    references = _smooth_volumes(rng, n_classes, size)
    return dict(
        references=references,
        eulers=np.column_stack(
            [rng.uniform(-180, 180, n_trials), rng.uniform(0, 180, n_trials), rng.uniform(-180, 180, n_trials)]
        ),
        particles=rng.choice(n_particles, size=n_trials, replace=False).astype(np.int64),
        seed_particles=rng.integers(0, 10_000, size=n_trials).astype(np.int64),
        pdf=np.array([0.6, 0.4][:n_classes]),
        sigma2=np.linspace(2.0, 0.05, size // 2 + 1),
        defU=rng.uniform(8000, 20000, n_particles),
        defV=rng.uniform(8000, 20000, n_particles),
        defA=rng.uniform(-90, 90, n_particles),
        phase=np.zeros(n_particles),
    )


def _oracle(case, *, size, current_size, padding, do_ctf, seed=11, fudge=1.0, pixel=2.1):
    return bind.vdam_expected_angular_errors(
        case["references"],
        case["eulers"],
        case["particles"],
        np.zeros(len(case["particles"]), dtype=np.int32),
        case["pdf"],
        case["sigma2"],
        case["defU"],
        case["defV"],
        case["defA"],
        case["phase"],
        300.0,
        2.7,
        0.1,
        pixel,
        size,
        current_size,
        padding,
        1,
        fudge,
        seed,
        do_ctf,
        False,
        case["seed_particles"],
        n_threads=4,
    )


def _projector_data(case, size, current_size, padding):
    data = [
        np.asarray(bind.compute_fourier_transform_map(ref, size, padding, 1, current_size, True, 2)[0])
        for ref in case["references"]
    ]
    return np.asarray(data), current_size // 2


def _ours(case, *, size, current_size, padding, ctf_images, seed=11, fudge=1.0, pixel=2.1):
    data, r_max = _projector_data(case, size, current_size, padding)
    return expected_angular_errors(
        projector_data=data,
        projector_r_max=r_max,
        padding_factor=padding,
        eulers_deg=case["eulers"],
        particle_ids=case["particles"],
        pdf_class=case["pdf"],
        sigma2_noise=case["sigma2"],
        ctf_images=ctf_images,
        pixel_size=pixel,
        model_box_size=size,
        current_image_size=current_size,
        sigma2_fudge=fudge,
        random_seed=seed,
        random_seed_particle_ids=case["seed_particles"],
    )


def _assert_same(ours, oracle):
    assert_matches(ours.acc_rot, float(oracle["acc_rot"]))
    assert_matches(ours.acc_trans, float(oracle["acc_trans"]))
    assert_matches(ours.acc_rot_class, np.asarray(oracle["acc_rot_class"]))
    assert_matches(ours.acc_trans_class, np.asarray(oracle["acc_trans_class"]))
    np.testing.assert_array_equal(ours.class_counts, np.asarray(oracle["class_counts"]))


@pytest.mark.parametrize(("current_size", "padding"), [(32, 2), (20, 2), (24, 1)])
def test_without_ctf_matches(current_size, padding):
    case = _case(np.random.default_rng(current_size + padding))
    oracle = _oracle(case, size=32, current_size=current_size, padding=padding, do_ctf=False)
    ours = _ours(case, size=32, current_size=current_size, padding=padding, ctf_images=None)
    _assert_same(ours, oracle)


@pytest.mark.parametrize("padding", [1, 2])
def test_current_sizes_of_one_window_class_share_the_device_programs(padding):
    """Sizes 18..24 are one quantum-8 class of a 32 box: exact against RELION, one compile each."""
    from relax.sampling import relion_expected_accuracy as accuracy

    case = _case(np.random.default_rng(3))
    programs = (
        accuracy._sample_coordinates,
        accuracy._interpolate_planes,
        accuracy._finish_projection,
        accuracy._shift,
        accuracy._trial_snr,
    )
    counts = []
    for current_size in (24, 22, 20, 18):
        oracle = _oracle(case, size=32, current_size=current_size, padding=padding, do_ctf=False)
        ours = _ours(case, size=32, current_size=current_size, padding=padding, ctf_images=None)
        _assert_same(ours, oracle)
        counts.append(tuple(program._cache_size() for program in programs))
    assert len(set(counts)) == 1, counts


def _slab_bytes(size, current_size, padding):
    side = 2 * (padding * (current_size // 2) + 1) + 1
    return 2 * 8 * side * side * (side // 2 + 1), 2 * 8 * side * (side // 2 + 1)


@pytest.mark.parametrize(("planes_short", "chunk_planes"), [(0, 3), (1, 3), (7, 1), (1, 40)])
def test_slab_at_and_over_the_device_budget_matches(monkeypatch, planes_short, chunk_planes):
    """At the resident budget the slab stays whole; a plane over it streams in chunks no larger than the
    stream chunk, and the accuracies match the oracle."""
    from relax.sampling import relion_expected_accuracy as accuracy

    size, current_size, padding = 32, 32, 2
    slab_bytes, plane_bytes = _slab_bytes(size, current_size, padding)
    chunk_bytes = chunk_planes * plane_bytes
    monkeypatch.setattr(accuracy, "accuracy_slab_resident_bytes", lambda: slab_bytes - planes_short * plane_bytes)
    monkeypatch.setattr(accuracy, "ACCURACY_SLAB_STREAM_CHUNK_BYTES", chunk_bytes)
    placed = []
    device = accuracy._Projector.device

    def recording(self, resident_bytes, stream_chunk_bytes):
        slab = device(self, resident_bytes, stream_chunk_bytes)
        placed.append(slab)
        return slab

    monkeypatch.setattr(accuracy._Projector, "device", recording)
    case = _case(np.random.default_rng(17))
    oracle = _oracle(case, size=size, current_size=current_size, padding=padding, do_ctf=False)
    ours = _ours(case, size=size, current_size=current_size, padding=padding, ctf_images=None)
    _assert_same(ours, oracle)
    assert placed and all(slab.streamed == (planes_short > 0 and chunk_bytes < slab_bytes) for slab in placed)
    for slab in placed:
        if slab.streamed:
            assert all(real.nbytes + imag.nbytes <= chunk_bytes for _, (real, imag) in slab.chunks)


def test_slab_budget_streams_the_full_box_10202_slab_in_small_chunks_and_keeps_box_256_whole():
    from relax.runtime.memory_budget import ACCURACY_SLAB_STREAM_CHUNK_BYTES, accuracy_slab_resident_bytes

    h100, p100 = int(79.65 * 1024**3), int(14.30 * 1024**3)
    assert _slab_bytes(800, 800, 2)[0] > accuracy_slab_resident_bytes(h100)
    assert _slab_bytes(256, 256, 2)[0] <= accuracy_slab_resident_bytes(p100)
    # No streamed chunk is a large block: each is at most 512 MiB, at least one full-box plane (20.6 MB).
    assert _slab_bytes(800, 800, 2)[1] <= ACCURACY_SLAB_STREAM_CHUNK_BYTES <= 512 * 1024**2


def test_with_ctf_matches():
    size = 32
    case = _case(np.random.default_rng(5))
    ctf = np.asarray(
        [
            bind.get_ctf_image(
                case["defU"][p],
                case["defV"][p],
                case["defA"][p],
                300.0,
                2.7,
                0.1,
                0.0,
                2.1,
                size,
                size,
                False,
                False,
                True,
                case["phase"][p],
                1.0,
            )
            for p in case["particles"]
        ]
    )
    oracle = _oracle(case, size=size, current_size=size, padding=2, do_ctf=True)
    ours = _ours(case, size=size, current_size=size, padding=2, ctf_images=ctf)
    _assert_same(ours, oracle)


def test_empty_classes_keep_relion_sentinels():
    case = _case(np.random.default_rng(9))
    case["pdf"] = np.array([0.005, 0.995])
    oracle = _oracle(case, size=32, current_size=32, padding=2, do_ctf=False)
    ours = _ours(case, size=32, current_size=32, padding=2, ctf_images=None)
    _assert_same(ours, oracle)


def _rotation(rng):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@pytest.mark.parametrize("do_ctf", [False, True])
def test_tilt_images_match(do_ctf):
    size, n_particles = 32, 30
    rng = np.random.default_rng(31)
    case = _case(rng, n_particles=n_particles)
    counts = rng.integers(1, 5, size=n_particles)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n_images = int(offsets[-1])
    projections = np.asarray([_rotation(rng) for _ in range(n_images)])
    image_ctf = np.column_stack(
        [
            rng.uniform(8000, 20000, n_images),
            rng.uniform(8000, 20000, n_images),
            rng.uniform(-90, 90, n_images),
            rng.uniform(0, 60, n_images),
            rng.uniform(0.7, 1.0, n_images),
            np.zeros(n_images),
            -np.ones(n_images),
        ]
    )
    oracle = bind.vdam_expected_angular_errors(
        case["references"],
        case["eulers"],
        case["particles"],
        np.zeros(len(case["particles"]), dtype=np.int32),
        case["pdf"],
        case["sigma2"],
        case["defU"],
        case["defV"],
        case["defA"],
        case["phase"],
        300.0,
        2.7,
        0.1,
        2.1,
        size,
        size,
        2,
        1,
        1.0,
        11,
        do_ctf,
        False,
        case["seed_particles"],
        image_offsets=offsets,
        image_projections=projections,
        image_ctf=image_ctf,
        n_threads=4,
    )
    rows = np.concatenate([np.arange(offsets[p], offsets[p + 1]) for p in case["particles"]])
    ctf = None
    if do_ctf:
        ctf = np.asarray(
            [
                bind.get_ctf_image(
                    *image_ctf[r, :3],
                    300.0,
                    2.7,
                    0.1,
                    image_ctf[r, 3],
                    2.1,
                    size,
                    size,
                    False,
                    False,
                    True,
                    image_ctf[r, 5],
                    image_ctf[r, 4],
                )
                for r in rows
            ]
        )
    data, r_max = _projector_data(case, size, size, 2)
    ours = expected_angular_errors(
        projector_data=data,
        projector_r_max=r_max,
        padding_factor=2,
        eulers_deg=case["eulers"],
        particle_ids=case["particles"],
        pdf_class=case["pdf"],
        sigma2_noise=case["sigma2"],
        ctf_images=ctf,
        pixel_size=2.1,
        model_box_size=size,
        current_image_size=size,
        sigma2_fudge=1.0,
        random_seed=11,
        random_seed_particle_ids=case["seed_particles"],
        image_offsets=offsets,
        image_projections=projections,
    )
    _assert_same(ours, oracle)


def test_optics_group_on_another_grid_matches():
    """Trials of a group on a 36-pixel box at 1.9 A against a 32-pixel model at 2.1 A."""

    size, full = 32, 36
    case = _case(np.random.default_rng(41))
    case["sigma2"] = np.linspace(2.0, 0.05, full // 2 + 1)
    oracle = bind.vdam_expected_angular_errors(
        case["references"],
        case["eulers"],
        case["particles"],
        np.zeros(len(case["particles"]), dtype=np.int32),
        case["pdf"],
        case["sigma2"],
        case["defU"],
        case["defV"],
        case["defA"],
        case["phase"],
        300.0,
        2.7,
        0.1,
        1.9,
        size,
        24,
        2,
        1,
        1.0,
        11,
        False,
        False,
        case["seed_particles"],
        model_pixel_size=2.1,
        image_full_size=full,
        projector_current_size=20,
        n_threads=4,
    )
    data, r_max = _projector_data(case, size, 20, 2)
    ours = expected_angular_errors(
        projector_data=data,
        projector_r_max=r_max,
        padding_factor=2,
        eulers_deg=case["eulers"],
        particle_ids=case["particles"],
        pdf_class=case["pdf"],
        sigma2_noise=case["sigma2"],
        ctf_images=None,
        pixel_size=1.9,
        model_box_size=size,
        current_image_size=24,
        sigma2_fudge=1.0,
        random_seed=11,
        random_seed_particle_ids=case["seed_particles"],
        model_pixel_size=2.1,
        box_size=full,
    )
    _assert_same(ours, oracle)


@pytest.mark.parametrize("tomo", [False, True])
@pytest.mark.parametrize("current_size", [32, 20])
def test_prepared_inputs_with_relax_ctf_match(tomo, current_size):
    """The production entry point, with relax's own CTF and projector, against the binding."""

    from relax.sampling.expected_accuracy import estimate_relion_expected_accuracy_from_prepared_inputs

    size, n_particles = 32, 30
    rng = np.random.default_rng(51 + current_size)
    case = _case(rng, n_particles=n_particles)
    tilt = None
    extra = {}
    if tomo:
        counts = rng.integers(1, 4, size=n_particles)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        n_images = int(offsets[-1])
        image_ctf = np.column_stack(
            [
                rng.uniform(8000, 20000, n_images),
                rng.uniform(8000, 20000, n_images),
                rng.uniform(-90, 90, n_images),
                rng.uniform(0, 60, n_images),
                rng.uniform(0.7, 1.0, n_images),
                np.zeros(n_images),
                np.where(rng.uniform(size=n_images) < 0.5, rng.uniform(0, 80, n_images), -1.0),
            ]
        )
        extra = dict(
            image_offsets=offsets,
            image_projections=np.asarray([_rotation(rng) for _ in range(n_images)]),
            image_ctf=image_ctf,
        )
        # The optics constants come from the estimate's arguments; the tilt images' own are unused.
        tilt = TiltImageAccuracyInputs(
            **extra,
            voltage=np.full(n_images, np.nan),
            spherical_aberration=np.full(n_images, np.nan),
            amplitude_contrast=np.full(n_images, np.nan),
        )
    oracle = bind.vdam_expected_angular_errors(
        case["references"], case["eulers"], case["particles"], np.zeros(len(case["particles"]), dtype=np.int32),
        case["pdf"], case["sigma2"], case["defU"], case["defV"], case["defA"], case["phase"],
        300.0, 2.7, 0.1, 2.1, size, current_size, 2, 1, 1.0, 11, True, False, case["seed_particles"],
        n_threads=4, **extra,
    )
    ours = estimate_relion_expected_accuracy_from_prepared_inputs(
        references_relion=case["references"],
        trial_eulers_deg=case["eulers"],
        trial_local_indices=case["particles"],
        trial_class_ids=np.zeros(len(case["particles"]), dtype=np.int32),
        class_weights=case["pdf"],
        sigma2_noise_relion=case["sigma2"],
        defocus_u=case["defU"],
        defocus_v=case["defV"],
        defocus_angle=case["defA"],
        phase_shift=case["phase"],
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=2.1,
        model_box_size=size,
        current_image_size=current_size,
        padding_factor=2,
        sigma2_fudge=1.0,
        random_seed=11,
        do_ctf_correction=True,
        random_seed_particle_ids=case["seed_particles"],
        tilt_images=tilt,
    )
    assert_matches(ours.acc_rot, float(oracle["acc_rot"]))
    assert_matches(ours.acc_trans_angstrom, float(oracle["acc_trans"]))
    assert_matches(ours.acc_rot_per_class, np.asarray(oracle["acc_rot_class"]))
    assert_matches(ours.acc_trans_per_class_angstrom, np.asarray(oracle["acc_trans_class"]))


@pytest.mark.parametrize("magnified", [False, True])
def test_prepared_inputs_with_optics_rows_match(magnified):
    """The caller's CTF rows (premultiplied, Zernike, magnification) and ``inv(M3)`` against the binding."""

    from relax.sampling.expected_accuracy import estimate_relion_expected_accuracy_from_prepared_inputs

    size, current_size = 32, 24
    rng = np.random.default_rng(71)
    case = _case(rng)
    ctf = np.asarray(
        [
            bind.get_ctf_image(u, v, a, 300.0, 2.7, 0.1, 0.0, 2.1, size, size, False, False, True, p, 1.0) ** 2
            for u, v, a, p in (
                (case["defU"][q], case["defV"][q], case["defA"][q], case["phase"][q]) for q in case["particles"]
            )
        ]
    )
    optics = {"trial_ctf": ctf}
    if magnified:
        optics["projection_left"] = np.linalg.inv(np.array([[1.012, 0.004, 0.0], [0.003, 0.991, 0.0], [0.0, 0.0, 1.0]]).T)
    oracle = bind.vdam_expected_angular_errors(
        case["references"], case["eulers"], case["particles"], np.zeros(len(case["particles"]), dtype=np.int32),
        case["pdf"], case["sigma2"], case["defU"], case["defV"], case["defA"], case["phase"],
        300.0, 2.7, 0.1, 2.1, size, current_size, 2, 1, 1.0, 11, True, False, case["seed_particles"],
        n_threads=4, **optics,
    )
    ours = estimate_relion_expected_accuracy_from_prepared_inputs(
        references_relion=case["references"], trial_eulers_deg=case["eulers"], trial_local_indices=case["particles"],
        trial_class_ids=np.zeros(len(case["particles"]), dtype=np.int32), class_weights=case["pdf"],
        sigma2_noise_relion=case["sigma2"], defocus_u=case["defU"], defocus_v=case["defV"],
        defocus_angle=case["defA"], phase_shift=case["phase"], voltage=300.0, spherical_aberration=2.7,
        amplitude_contrast=0.1, pixel_size=2.1, model_box_size=size, current_image_size=current_size, padding_factor=2,
        sigma2_fudge=1.0, random_seed=11, do_ctf_correction=True, random_seed_particle_ids=case["seed_particles"],
        optics=optics,
    )
    assert_matches(ours.acc_rot, float(oracle["acc_rot"]))
    assert_matches(ours.acc_trans_angstrom, float(oracle["acc_trans"]))
    assert_matches(ours.acc_rot_per_class, np.asarray(oracle["acc_rot_class"]))
