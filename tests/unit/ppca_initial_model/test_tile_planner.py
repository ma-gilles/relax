"""Tile planning on device budgets: a subtomogram stage at radius 31 / HEALPix 3 / 41 tilts on a 16 GB card."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from relax import sampling
from relax.ppca_initial_model.tomo import TiltParticles, load_tilt_tile
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_stream import (
    PLAN_REGION_MARGIN_BYTES,
    TILE_FRAGMENTATION_HEADROOM,
    plan_region_bytes,
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


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kind, stage", [("subtomogram", (31, 3)), ("single_particle", (31, 3)), ("single_particle", (16, 3))]
)
def test_an_80_gb_card_keeps_the_largest_oversampling_job_chunk(kind, stage):
    """Adaptive oversampling's job plan changes nothing on a large card either: on 80 GB a pass-2 program takes
    as many jobs as one kernel launch does (1024 for single particles, 128 at 41 tilts), with the compiled job
    programs and the job results of 150 particles x 100 samples counted."""
    from relax.ppca_refinement.oversampled_stream import (
        job_program_bytes,
        launch_job_chunk,
        plan_job_chunk,
        prepare_oversampled_stream,
    )

    stream = (_tilt_stream if kind == "subtomogram" else _spa_stream)(*stage)
    n_rotations = int(stream.arrays.rotations.shape[0]) - 1
    dimensions = int(stream.translations.shape[1])
    # The planner reads shapes only: zero child grids of RELION's sizes (8 child rotations, 2^D child shifts).
    ostream = prepare_oversampled_stream(
        stream,
        np.zeros((8 * n_rotations, 3, 3), np.float32),
        np.zeros(((1 << dimensions) * int(stream.translations.shape[0]), dimensions), np.float32),
    )
    device = 80 * GIB
    largest = launch_job_chunk(ostream)
    assert largest == (128 if kind == "subtomogram" else 1024)
    assert plan_job_chunk(ostream, 150, memory_bytes=device - _resident(stream), device_bytes=device) == largest
    # A 16 GB card plans fewer jobs per program when the largest chunk does not fit it.
    small = 16 * GIB
    planned = plan_job_chunk(ostream, 150, memory_bytes=small - _resident(stream), device_bytes=small)
    budget = small - _resident(stream) - TILE_FRAGMENTATION_HEADROOM * small
    assert planned <= largest and job_program_bytes(ostream, 150, planned) <= budget


def test_a_cpu_stream_refuses_a_rotation_block_its_host_cannot_hold():
    """On the CPU the moment program's XLA adjoint holds one half volume per block image and channel: the
    k3conf r31 stage at rotation block 512 needs far more than a 64 GiB host, and is refused before any
    allocation."""
    stream = _tilt_stream()
    if stream.static.cuda_kernels:
        pytest.skip("a CPU stream")
    with pytest.raises(ValueError, match="rotation block 512"):
        plan_tile_images(stream, 150, memory_bytes=64 * GIB, device_bytes=64 * GIB)


def test_a_preallocated_pool_has_no_region_to_take():
    """The region rule is a no-op on a pool that already holds its limit, at any device size."""
    for limit in (16 * GIB, 36 * GIB, 72 * GIB):
        assert plan_region_bytes(0.8 * limit, limit, limit, 8 * GIB) == 0
    assert plan_region_bytes(30 * GIB, None, None, None) == 0


def test_a_grown_pool_takes_the_plan_as_one_block_bounded_by_what_is_left():
    """The 40 GB A100 last stage (30.78 GiB counted, 36.4 GiB limit): the plan's bytes while the
    allocator may still take them, else everything it still may, never more than the device has free."""
    counted, limit = int(30.78 * GIB), int(36.4 * GIB)
    assert plan_region_bytes(counted, limit, 1 * GIB, 38 * GIB) == counted
    assert plan_region_bytes(counted, limit, 8 * GIB, 38 * GIB) == limit - 8 * GIB - PLAN_REGION_MARGIN_BYTES
    assert plan_region_bytes(counted, limit, 8 * GIB, 20 * GIB) == 20 * GIB - PLAN_REGION_MARGIN_BYTES
    assert plan_region_bytes(counted, limit, limit - PLAN_REGION_MARGIN_BYTES, 38 * GIB) == 0


_REGION_SCRIPT = """
import sys
import jax, jax.numpy as jnp
from relax.ppca_refinement.full_row_stream import reserve_plan_region
device = jax.devices()[0]
block = lambda megabytes: jnp.zeros((megabytes, 2**20), jnp.uint8)
limit = device.memory_stats()["bytes_limit"] // 2**20
# Two kept blocks of 30% of the limit: a grown pool puts each in a region of its own.
if sys.argv[1] == "reserve":
    taken = reserve_plan_region(device, 0.9 * limit * 2**20)
    assert taken > 0.8 * limit * 2**20, taken
    assert reserve_plan_region(device, 0.9 * limit * 2**20) == 0
kept = [block(3 * limit // 10), block(3 * limit // 10)]
try:
    block(35 * limit // 100).block_until_ready()
    print("FITS")
except Exception as error:
    print("FRAGMENTED" if "RESOURCE_EXHAUSTED" in str(error) else repr(error))
"""


@pytest.mark.parametrize("mode, outcome", [("grow", "FRAGMENTED"), ("reserve", "FITS")])
def test_a_plan_region_keeps_a_grown_pool_contiguous(mode, outcome, tmp_path):
    """Without preallocation a pool grown block by block refuses a block that is 35% of its limit
    with 40% free; after the plan's region is taken the same allocations fit."""
    import os
    import subprocess
    import sys

    import jax

    if jax.devices()[0].platform not in {"gpu", "cuda"}:
        pytest.skip("the allocator pool is a GPU allocator's")
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=".10")
    env.pop("TF_GPU_ALLOCATOR", None)
    run = subprocess.run(
        [sys.executable, "-c", _REGION_SCRIPT, mode], env=env, capture_output=True, text=True, cwd=os.getcwd()
    )
    assert run.stdout.strip().splitlines()[-1:] == [outcome], run.stdout[-2000:] + run.stderr[-2000:]
