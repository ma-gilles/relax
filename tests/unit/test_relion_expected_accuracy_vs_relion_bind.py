"""relax's expected-accuracy estimator against RELION's (the binding is the oracle).

The accuracies are sums of discrete step values, so they must agree exactly;
the SNR arithmetic follows RELION's host double order.
"""

from __future__ import annotations

import numpy as np
import pytest

from relax.helpers.relion_expected_accuracy import expected_angular_errors

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
        ori_size=size,
        current_image_size=current_size,
        sigma2_fudge=fudge,
        random_seed=seed,
        random_seed_particle_ids=case["seed_particles"],
    )


def _assert_same(ours, oracle):
    assert ours.acc_rot == float(oracle["acc_rot"])
    assert ours.acc_trans == float(oracle["acc_trans"])
    np.testing.assert_array_equal(ours.acc_rot_class, np.asarray(oracle["acc_rot_class"]))
    np.testing.assert_array_equal(ours.acc_trans_class, np.asarray(oracle["acc_trans_class"]))
    np.testing.assert_array_equal(ours.class_counts, np.asarray(oracle["class_counts"]))


@pytest.mark.parametrize(("current_size", "padding"), [(32, 2), (20, 2), (24, 1)])
def test_without_ctf_matches(current_size, padding):
    case = _case(np.random.default_rng(current_size + padding))
    oracle = _oracle(case, size=32, current_size=current_size, padding=padding, do_ctf=False)
    ours = _ours(case, size=32, current_size=current_size, padding=padding, ctf_images=None)
    _assert_same(ours, oracle)


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
        ori_size=size,
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
        ori_size=size,
        current_image_size=24,
        sigma2_fudge=1.0,
        random_seed=11,
        random_seed_particle_ids=case["seed_particles"],
        model_pixel_size=2.1,
        image_full_size=full,
    )
    _assert_same(ours, oracle)


@pytest.mark.parametrize("tomo", [False, True])
@pytest.mark.parametrize("current_size", [32, 20])
def test_prepared_inputs_with_relax_ctf_match(tomo, current_size):
    """The production entry point, with relax's own CTF and projector, against the binding."""

    from relax.helpers.expected_accuracy import estimate_relion_expected_accuracy_from_prepared_inputs

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
        tilt = dict(
            image_offsets=offsets,
            image_projections=np.asarray([_rotation(rng) for _ in range(n_images)]),
            image_ctf=image_ctf,
        )
        extra = dict(tilt)
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
        ori_size=size,
        current_image_size=current_size,
        padding_factor=2,
        sigma2_fudge=1.0,
        random_seed=11,
        do_ctf_correction=True,
        random_seed_particle_ids=case["seed_particles"],
        tilt_images=tilt,
    )
    assert ours.acc_rot == float(oracle["acc_rot"])
    assert ours.acc_trans_angstrom == float(oracle["acc_trans"])
    np.testing.assert_array_equal(ours.acc_rot_per_class, np.asarray(oracle["acc_rot_class"]))
    np.testing.assert_array_equal(ours.acc_trans_per_class_angstrom, np.asarray(oracle["acc_trans_class"]))


@pytest.mark.parametrize("magnified", [False, True])
def test_prepared_inputs_with_optics_rows_match(magnified):
    """The caller's CTF rows (premultiplied, Zernike, magnification) and ``inv(M3)`` against the binding."""

    from relax.helpers.expected_accuracy import estimate_relion_expected_accuracy_from_prepared_inputs

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
        amplitude_contrast=0.1, pixel_size=2.1, ori_size=size, current_image_size=current_size, padding_factor=2,
        sigma2_fudge=1.0, random_seed=11, do_ctf_correction=True, random_seed_particle_ids=case["seed_particles"],
        optics=optics,
    )
    assert ours.acc_rot == float(oracle["acc_rot"])
    assert ours.acc_trans_angstrom == float(oracle["acc_trans"])
    np.testing.assert_array_equal(ours.acc_rot_per_class, np.asarray(oracle["acc_rot_class"]))
