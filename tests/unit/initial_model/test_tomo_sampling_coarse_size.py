"""The subtomogram VDAM coarse size on an iteration that updates the sampling."""

from types import SimpleNamespace

from relax.vdam.tomo_estep import tomo_initial_model_sampling


def _plan(order):
    return SimpleNamespace(
        healpix_order=order, oversampling=1, offset_range_angstrom=15.566, offset_step_angstrom=4.25,
        random_perturbation=0.242,
    )


def test_coarse_size_is_the_order_before_the_sampling_update():
    # RELION sets image_coarse_size (expectationSetup step A, ml_optimiser.cpp:3914) before it updates the
    # angular sampling (step D, :3945), so the iteration that moves from HEALPix 1 to 2 still scores its coarse
    # pass at the HEALPix-1 size. etob2l_plain (box 64 at 4.25 A, diameter 240): 12 at order 1, 22 at order 2;
    # stock RELION keeps 25 significant samples per particle there, the order-2 size kept 2.6 (job 14971225).
    state = SimpleNamespace(ori_size=64, pixel_size=4.25, current_size=38)
    assert tomo_initial_model_sampling(state, _plan(1), particle_diameter_ang=240.0).coarse_size == 12
    assert tomo_initial_model_sampling(state, _plan(2), particle_diameter_ang=240.0).coarse_size == 22
    updated = tomo_initial_model_sampling(state, _plan(2), particle_diameter_ang=240.0, pass1_healpix_order=1)
    assert (updated.healpix_order, updated.coarse_size, updated.fine_size) == (2, 12, 38)
