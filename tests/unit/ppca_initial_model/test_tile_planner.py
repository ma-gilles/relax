"""Tile planning on device budgets: a subtomogram stage at radius 31 / HEALPix 3 / 41 tilts on a 16 GB card."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from relax import sampling
from relax.ppca_initial_model.tomo import TiltParticles, load_tilt_tile
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_stream import (
    TILE_FRAGMENTATION_HEADROOM,
    plan_tile_images,
    prepare_full_row_stream,
    stream_tile_bytes,
    tile_bytes,
    tile_program_bytes,
)

pytestmark = pytest.mark.unit

GIB = 1 << 30
BOX = 64


def _tilt_stream(radius=31, order=3):
    """The k3conf cryo-ET last stage: box 64, radius 31, HEALPix 3, 41 tilts, 33 3D shifts, block 512, q 2."""
    n_half = BOX * (BOX // 2 + 1)
    frames = Rotation.from_euler("y", np.arange(-60, 61, 3), degrees=True).as_matrix()
    image_frames = [np.arange(41)] * 2
    particles = TiltParticles(
        image_shape=(BOX, BOX),
        volume_shape=(BOX, BOX, BOX),
        voxel_size=8.5,
        image_offsets=np.asarray([0, 41, 82], np.int64),
        image_frame=np.concatenate(image_frames).astype(np.int64),
        particle_group=np.zeros(2, np.int64),
        group_frames=(np.asarray(frames, np.float64),),
        read=lambda ids: (None, jnp.zeros((len(ids), n_half), jnp.complex64), jnp.ones((len(ids), n_half))),
    )
    translations = np.asarray(sampling.get_relion_translation_grid_3d(2, 1), np.float32)
    return _stream(particles, radius, order, translations, q=2, tile_loader=load_tilt_tile)


class _ImageShapes:
    """A single-particle dataset's shapes (EMPIAR-10076 PPCA fixture: box 64); the planner reads no images."""

    image_shape, volume_shape, grid_size, voxel_size = (BOX, BOX), (BOX, BOX, BOX), BOX, 6.55
    n_images = n_units = 300
    dtype = jnp.complex64
    CTF_params = np.zeros((300, 9), np.float32)

    def __init__(self):
        self.image_source = self

    @staticmethod
    def ctf_evaluator(params, image_shape, voxel_size, *, half_image=False):
        return jnp.ones((params.shape[0], image_shape[0] * (image_shape[1] // 2 + 1)), jnp.float32)

    def process_images(self, images, apply_image_mask=False):
        return images

    process_images_half = process_images


def _spa_stream(radius, order):
    """The EMPIAR-10076 PPCA stages: box 64, q 4, shift range 6 px in 2 px steps."""
    translations = np.asarray(sampling.get_relion_translation_grid(max_pixel=6, pixel_offset=2), np.float32)
    return _stream(_ImageShapes(), radius, order, translations, q=4, tile_loader=None)


def _stream(particles, radius, order, translations, *, q, tile_loader):
    rotations = np.asarray(sampling.get_relion_hidden_rotation_grid(order, matrices=True), np.float32)
    n_volume = BOX * BOX * (BOX // 2 + 1)
    return prepare_full_row_stream(
        particles,
        np.zeros(n_volume, np.complex64),
        np.zeros((n_volume, q), np.complex64),
        noise_variance=np.ones((BOX, BOX), np.float32),
        rotations=rotations,
        translations=translations,
        rotation_log_prior=np.full(len(rotations), -np.log(len(rotations)), np.float32),
        translation_log_prior=np.full(len(translations), -np.log(len(translations)), np.float32),
        rotation_parent=np.zeros(len(rotations), np.int32),
        translation_parent=np.zeros(len(translations), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=2 * radius, q=q, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=150, rotation_block_size=512),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        tile_loader=tile_loader,
        gemm_precision="fp32",
        pass2_mass_floor=1e-10,  # the controller's default: the planner counts the compacted pass-2 copy
    )


def _resident(stream):
    return sum(int(np.asarray(x).nbytes) for x in stream.arrays if hasattr(x, "nbytes"))


def _program_stand_in(monkeypatch):
    """On CPU the block programs are the XLA reference path, not the GPU programs a 16 GB card runs: a
    per-image stand-in for their memory (one block's projections, products and M-step images for 41
    frames, plus its GEMM outputs per image) checks the planner's arithmetic."""
    from relax.ppca_refinement import full_row_stream as frs

    block, K, P, T = 512, 41, 3, 33
    F = frs._window_pixels(_tilt_stream())
    monkeypatch.setattr(
        frs, "tile_program_bytes", lambda stream, n: 4 * (block * K * F * 4 * P + n * block * (2 * P * T + 12))
    )


def _plan_checks(stream):
    resident = _resident(stream)
    device = 16 * GIB
    available = device - resident
    budget = available - TILE_FRAGMENTATION_HEADROOM * device
    planned = plan_tile_images(stream, 300, memory_bytes=available, device_bytes=device)
    assert 1 <= planned < 300
    assert tile_bytes(stream, planned) <= budget
    # An 80 GB card takes more images per tile.
    assert plan_tile_images(stream, 300, memory_bytes=80 * GIB - resident, device_bytes=80 * GIB) > planned
    return planned


def test_radius_31_tilt_stage_fits_a_16_gb_card(monkeypatch):
    """The planned tile's counted bytes stay inside a 16 GB card's budget (the GPU test below uses the
    compiled programs)."""
    _program_stand_in(monkeypatch)
    stream = _tilt_stream()
    planned = _plan_checks(stream)
    reader_peak, operands = load_tilt_tile.operand_bytes(stream, planned)
    assert stream_tile_bytes(stream, planned) + reader_peak + operands == tile_bytes(stream, planned)


@pytest.mark.gpu
def test_radius_31_tilt_stage_fits_a_16_gb_card_with_compiled_programs():
    """On the GPU, the block programs' memory comes from XLA's compiled analysis of the CUDA programs."""
    stream = _tilt_stream()
    planned = _plan_checks(stream)
    assert stream_tile_bytes(stream, planned) > tile_program_bytes(stream, planned) > 0


def test_one_tile_per_call_is_planned_without_the_read_ahead(monkeypatch):
    """Tiles are read ahead only within one accumulate call: when every call holds one tile, the next
    tile's reader is not counted; when a call holds two tiles at that size, it is."""
    _program_stand_in(monkeypatch)
    stream = _tilt_stream()
    device = 16 * GIB
    available = device - _resident(stream)
    budget = available - TILE_FRAGMENTATION_HEADROOM * device

    def plan(particles_per_call):
        def tiles_per_call(size):
            return -(-particles_per_call // size)

        return plan_tile_images(stream, 300, memory_bytes=available, device_bytes=device, tiles_per_call=tiles_per_call)

    ahead = plan_tile_images(stream, 300, memory_bytes=available, device_bytes=device)
    single = plan(1)
    assert ahead < single < 300
    assert tile_bytes(stream, single, pipelined=False) <= budget < tile_bytes(stream, single + 1, pipelined=False)
    reader_peak, operands = load_tilt_tile.operand_bytes(stream, single)
    assert tile_bytes(stream, single, pipelined=False) == max(stream_tile_bytes(stream, single) + operands, reader_peak)
    # A call of `single` particles is one tile at the single plan.
    assert plan(single) == single
    # One more particle needs two tiles at that size, so the read-ahead is counted.
    assert plan(single + 1) == ahead
    # Calls that hold several tiles even at the requested size never plan without it.
    assert plan(301) == ahead == plan_tile_images(stream, 300, memory_bytes=available, device_bytes=device)


STAGES = ((4, 1), (8, 2), (16, 3), (31, 3), (31, 4))


@pytest.mark.gpu
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("kind", ["subtomogram", "single_particle"])
def test_an_80_gb_card_keeps_the_requested_tile_at_every_stage(kind, stage):
    """The planner changes nothing on a large card: every k3conf and EMPIAR-10076 stage keeps 150
    particles per tile on 80 GB, with the compiled block programs and the read-ahead counted."""
    stream = (_tilt_stream if kind == "subtomogram" else _spa_stream)(*stage)
    device = 80 * GIB
    assert plan_tile_images(stream, 150, memory_bytes=device - _resident(stream), device_bytes=device) == 150
