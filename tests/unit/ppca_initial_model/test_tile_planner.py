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
)

pytestmark = pytest.mark.unit

GIB = 1 << 30
BOX = 64


def _tilt_stream():
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
    rotations = np.asarray(sampling.get_relion_hidden_rotation_grid(3, matrices=True), np.float32)
    translations = np.asarray(sampling.get_relion_translation_grid_3d(2, 1), np.float32)
    n_volume = BOX * BOX * (BOX // 2 + 1)
    return prepare_full_row_stream(
        particles,
        np.zeros(n_volume, np.complex64),
        np.zeros((n_volume, 2), np.complex64),
        noise_variance=np.ones((BOX, BOX), np.float32),
        rotations=rotations,
        translations=translations,
        rotation_log_prior=np.full(len(rotations), -np.log(len(rotations)), np.float32),
        translation_log_prior=np.full(len(translations), -np.log(len(translations)), np.float32),
        rotation_parent=np.zeros(len(rotations), np.int32),
        translation_parent=np.zeros(len(translations), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=2 * 31, q=2, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=150, rotation_block_size=512),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        tile_loader=load_tilt_tile,
        gemm_precision="fp32",
    )


def test_radius_31_tilt_stage_fits_a_16_gb_card():
    """The planned tile's counted bytes stay inside a 16 GB card's budget, and one more particle would not."""
    stream = _tilt_stream()
    resident = sum(int(np.asarray(x).nbytes) for x in stream.arrays if hasattr(x, "nbytes"))
    device = 16 * GIB
    available = device - resident
    budget = available - TILE_FRAGMENTATION_HEADROOM * device
    planned = plan_tile_images(stream, 150, memory_bytes=available, device_bytes=device)
    assert 1 <= planned < 150
    assert tile_bytes(stream, planned) <= budget < tile_bytes(stream, planned + 1)
    # Every tilt counts: the reader's operands for 41 frames outweigh the kept buffers per particle.
    reader_peak, operands = load_tilt_tile.operand_bytes(stream, planned)
    assert reader_peak + operands > stream_tile_bytes(stream, planned) - stream_tile_bytes(stream, 0)
    # An 80 GB card takes the requested batch.
    assert plan_tile_images(stream, 150, memory_bytes=80 * GIB - resident, device_bytes=80 * GIB) > planned
