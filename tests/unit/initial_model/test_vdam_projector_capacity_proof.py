"""Focused CPU/C++ proof for a stable-shape InitialModel projector ABI."""

import logging

import numpy as np
import pytest

from relax.helpers.fourier_window import stable_fourier_window_current_size
from scripts.prove_vdam_projector_capacity import (
    GF46_IMAGE_SIZE,
    GF46_LOGICAL_CURRENT_SIZES,
    analyze,
    center_pad_relion_projector,
    crop_relion_projection_capacity,
    crop_relion_projector_capacity,
    gf46_logical_physical_pairs,
)

pytestmark = pytest.mark.unit

# Float comparisons use relative bands, never bitwise equality: |a - b| <= rtol * max|a|.
# The C++ projector texels are complex128; VDAM projects complex64.
F64_RTOL = 1e-13
F32_RTOL = 1e-6
# Constructions that are not equivalent differ by at least 8e-3 relative L2 on every
# non-identity GF46 pair (measured at 6d82dfd); 1e-3 sits far above the float band.
NOT_EQUIVALENT_MIN_RELATIVE_L2 = 1e-3


def test_center_padding_preserves_relion_frequency_coordinates():
    logical = (np.arange(7 * 7 * 4, dtype=np.float32).reshape(7, 7, 4) + np.complex64(1j)).astype(np.complex64)

    padded = center_pad_relion_projector(logical, (11, 11, 6))

    assert padded.shape == (11, 11, 6)
    assert padded.dtype == np.complex64
    np.testing.assert_allclose(
        crop_relion_projector_capacity(padded, logical.shape),
        logical,
        rtol=0.0,
        atol=F32_RTOL * np.max(np.abs(logical)),
    )
    assert np.count_nonzero(padded) == logical.size


def test_gf46_fixture_tracks_every_observed_logical_to_physical_pair():
    pairs = gf46_logical_physical_pairs()

    assert tuple(logical for logical, _physical in pairs) == GF46_LOGICAL_CURRENT_SIZES
    assert all(physical == stable_fourier_window_current_size(logical, GF46_IMAGE_SIZE) for logical, physical in pairs)
    assert len(pairs) == 32
    assert len({physical for _logical, physical in pairs}) == 14


def test_projection_capacity_crop_preserves_fftw_signed_rows():
    physical_size = 8
    logical_size = 4
    physical = np.arange(
        physical_size * (physical_size // 2 + 1),
        dtype=np.float32,
    )[None, :]

    cropped = crop_relion_projection_capacity(
        physical,
        physical_size=physical_size,
        logical_size=logical_size,
    ).reshape(logical_size, logical_size // 2 + 1)

    physical_grid = physical.reshape(physical_size, physical_size // 2 + 1)
    np.testing.assert_allclose(
        cropped,
        physical_grid[[0, 1, 2, 7], : logical_size // 2 + 1],
        rtol=0.0,
        atol=F32_RTOL * np.max(np.abs(physical_grid)),
    )


@pytest.mark.slow
def test_full_gf46_projector_capacity_proof(caplog):
    pytest.importorskip("relax.relion_bind._relion_bind_core")
    caplog.set_level(logging.INFO)

    report = analyze()
    logging.info("\n%s", report["markdown"])

    summary = report["summary"]
    assert summary["pair_count"] == len(GF46_LOGICAL_CURRENT_SIZES)
    assert summary["physical_class_count"] == 14
    rows = report["pairs"]
    nonidentity = [row for row in rows if row["logical_size"] != row["physical_size"]]
    assert len(nonidentity) == summary["nonidentity_pair_count"] > 0

    def texel_l2(name):
        return max(float(row["texels"][name]["relative_l2"]) for row in rows)

    assert texel_l2("physical_rebuild_logical_sphere") <= F64_RTOL
    assert texel_l2("center_padded_logical_box") <= F64_RTOL

    def projection_l2(name, selected):
        return [float(row["projections"][name]["relative_l2"]) for row in selected]

    # Center padding with the logical r_max reproduces the logical projection.
    assert max(projection_l2("center_padded_logical_cutoff", rows)) <= F32_RTOL

    # A larger C++ rebuild fills a new Fourier annulus. Trilinear interpolation
    # can read that annulus at the logical boundary, so it is not an exact
    # substitute even if the logical radial cutoff is retained.
    assert min(projection_l2("physical_rebuild_logical_cutoff", nonidentity)) > NOT_EQUIVALENT_MIN_RELATIVE_L2

    # Capacity alone is insufficient: the logical cutoff is part of exact
    # semantics and cannot be replaced by the bucket's physical r_max.
    assert min(projection_l2("center_padded_physical_cutoff", nonidentity)) > NOT_EQUIVALENT_MIN_RELATIVE_L2
