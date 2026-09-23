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
