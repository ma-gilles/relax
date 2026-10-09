"""relion_refine's --ref STAR for Class3D: reference paths and the fresh 1/K class distribution."""

import numpy as np
import pytest
import starfile
import pandas as pd

from relax.refinement.mean_helpers import _initialize_class_log_priors
from relax.relion.relion_metadata import read_relion_reference_star

pytestmark = pytest.mark.unit


def test_reference_star_paths_resolve_against_the_star_directory(tmp_path):
    table = pd.DataFrame(
        {
            "rlnReferenceImage": ["ref_a.mrc", str(tmp_path / "abs" / "ref_b.mrc")],
            "rlnClassDistribution": [0.7, 0.3],
        }
    )
    star = tmp_path / "sub" / "refs.star"
    star.parent.mkdir()
    starfile.write({"model_classes": table}, star)
    paths, distribution = read_relion_reference_star(star)
    assert paths == [star.parent / "ref_a.mrc", tmp_path / "abs" / "ref_b.mrc"]
    np.testing.assert_allclose(distribution, [0.7, 0.3])


def test_reference_star_without_reference_images_is_rejected(tmp_path):
    star = tmp_path / "refs.star"
    starfile.write({"model_classes": pd.DataFrame({"rlnClassDistribution": [1.0]})}, star)
    with pytest.raises(ValueError, match="_rlnReferenceImage"):
        read_relion_reference_star(star)


def test_fresh_class3d_class_distribution_is_uniform():
    # MlModel::initialise sets pdf_class = 1/K (ml_model.cpp:53); a --ref STAR
    # distribution is not read for a fresh run.
    log_priors, weights = _initialize_class_log_priors(4)
    np.testing.assert_allclose(weights, np.full(4, 0.25))
    np.testing.assert_allclose(log_priors, np.full(4, -np.log(4.0)))


@pytest.mark.parametrize(
    ("diameter_token", "width_token", "expected"),
    [("200.000000", "5", (200.0, 5.0)), ("2.500000e+05", "10", (250000.0, 10.0)), ("-1.00000", "5", (-1.0, 5.0))],
)
def test_mask_params_read_the_whole_token_relion_wrote(tmp_path, diameter_token, width_token, expected):
    # MetaDataTable writes doubles as %12.6f, %12.5f when negative, and in e notation outside [1e-3, 1e5]
    # (metadata_table.cpp:257-277); a digits-only pattern read 2.500000e+05 as 2.5 and missed -1.00000.
    from relax.relion.relion_metadata import load_relion_mask_params

    star = tmp_path / "run_it001_optimiser.star"
    star.write_text(
        "data_optimiser_general\n\n"
        f"_rlnParticleDiameter {diameter_token:>12}\n"
        f"_rlnWidthMaskEdge {width_token:>12}\n"
    )
    assert load_relion_mask_params(star) == expected
