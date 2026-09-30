"""relax's InitialModel bootstrap against RELION's (the binding is the oracle)."""

from __future__ import annotations

import numpy as np
import pytest

from relax.vdam import bootstrap_reconstruction as br

bind = pytest.importorskip("relax.relion_bind._relion_bind_core")

# Float rounding only: FFT and back-projection sums in another order than RELION's.
RTOL = 1e-12  # measured up to 2e-15


def _case(size, n, seed=3):
    rng = np.random.default_rng(seed)
    grid = np.fft.fftfreq(size)
    radius2 = grid[:, None] ** 2 + grid[None, :] ** 2
    images = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, size, size))) * np.exp(-radius2 / 0.02)))
    return dict(
        images=np.ascontiguousarray(images * 10.0),
        defU=rng.uniform(9000, 20000, n),
        defV=rng.uniform(9000, 20000, n),
        defAngle=rng.uniform(-90, 90, n),
        phase_shift=np.zeros(n),
    )


def _ctf(case, size, pixel):
    return np.asarray(
        [
            bind.get_ctf_image(u, v, a, 300.0, 2.7, 0.1, 0.0, pixel, size, size, False, False, True, p, 1.0)
            for u, v, a, p in zip(case["defU"], case["defV"], case["defAngle"], case["phase_shift"], strict=True)
        ]
    )


def _relative(actual, expected):
    return float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))


@pytest.mark.parametrize(
    ("size", "padding", "do_ctf", "classes", "n", "current_size"),
    [
        (32, 1, True, 2, 40, -1),
        (32, 2, False, 1, 40, -1),
        (24, 2, True, 3, 40, -1),
        (20, 1, True, 2, 150, -1),
        (32, 1, True, 1, 40, 9),
        (32, 2, True, 2, 40, 12),
    ],
)
def test_bootstrap_and_postprocess_match(size, padding, do_ctf, classes, n, current_size):
    pixel = 3.0
    case = _case(size, n)
    ctf = _ctf(case, size, pixel) if do_ctf else None
    common = dict(
        pixel_size=pixel,
        ori_size=size,
        nr_classes=classes,
        particle_diameter_ang=0.7 * size * pixel,
        width_mask_edge_px=5.0,
        do_zero_mask=True,
        random_seed=17,
        padding_factor=padding,
    )
    seeds = np.arange(n, dtype=np.int64)[::-1].copy()
    expected = np.asarray(
        bind.vdam_bootstrap_iref(
            case["images"],
            case["defU"],
            case["defV"],
            case["defAngle"],
            case["phase_shift"],
            300.0,
            2.7,
            0.1,
            common["pixel_size"],
            size,
            classes,
            common["particle_diameter_ang"],
            5.0,
            True,
            do_ctf,
            17,
            padding,
            1,
            current_size,
            n - 10,
            seeds,
        )
    )
    post = dict(
        pixel_size=pixel,
        ini_high_ang=4.0 * pixel,
        particle_diameter_ang=common["particle_diameter_ang"],
        width_mask_edge_px=5.0,
    )
    expected_post = np.asarray(
        bind.vdam_postprocess_initial_iref(
            expected, post["pixel_size"], post["ini_high_ang"], post["particle_diameter_ang"], 5.0, True, False
        )
    )
    actual, generator = br.bootstrap_references(
        images=case["images"], ctf_images=ctf, minimum_nr_particles=n - 10, current_size=current_size, particle_seed_ids=seeds, **common
    )
    assert actual.shape == expected.shape
    assert _relative(actual, expected) < RTOL
    actual_post = br.postprocess_references(actual, generator=generator, **post)
    assert _relative(actual_post, expected_post) < RTOL


def test_production_bootstrap_with_relax_ctf_matches():
    """bootstrap_iref's entry points, with relax's own CTF, against the binding (RECOVAR frame).

    The entry point takes RELION's bootstrap size ROUND(0.07 ori_size) when none is given.
    """

    from recovar.utils.helpers import relion_volume_to_recovar

    from relax.vdam import bootstrap_iref

    size, n, pixel = 32, 60, 3.0
    case = _case(size, n, seed=8)
    args = dict(voltage=300.0, Cs=2.7, Q0=0.1, pixel_size=pixel, ori_size=size, nr_classes=2,
                particle_diameter_ang=0.7 * size * pixel, width_mask_edge_px=5.0, do_zero_mask=True,
                do_ctf_correction=True, random_seed=23, padding_factor=1, minimum_nr_particles=50)
    expected = np.asarray(
        bind.vdam_bootstrap_iref(
            case["images"], case["defU"], case["defV"], case["defAngle"], case["phase_shift"], 300.0, 2.7, 0.1,
            pixel, size, 2, args["particle_diameter_ang"], 5.0, True, True, 23, 1, 1, int(np.floor(0.07 * size + 0.5)), 50, None,
        )
    )
    expected_post = np.asarray(
        bind.vdam_postprocess_initial_iref(expected, pixel, 4.0 * pixel, args["particle_diameter_ang"], 5.0, True, False)
    )
    iref, rand_state = bootstrap_iref.compute_bootstrap_iref(**case, **args)
    post = bootstrap_iref.postprocess_bootstrap_iref(
        iref, rand_state=rand_state, pixel_size=pixel, ini_high_ang=4.0 * pixel,
        particle_diameter_ang=args["particle_diameter_ang"], width_mask_edge_px=5.0,
    )
    to_recovar = np.asarray([relion_volume_to_recovar(v) for v in expected])
    post_recovar = np.asarray([relion_volume_to_recovar(v) for v in expected_post])
    # relax's CTF agrees with RELION's to about 1e-11 relative (test_relion_ctf_formula.py).
    assert _relative(iref, to_recovar) < 1e-10
    assert _relative(post, post_recovar) < 1e-10
