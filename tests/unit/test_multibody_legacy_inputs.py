"""A multi-body run from a RELION 2-era consensus (EMPIAR-10180): deprecated body columns, pre-RELION-4 model files."""

from __future__ import annotations

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from relax.multibody.bodies import read_bodies
from relax.multibody.checkpoint import read_relion_checkpoint
from relax.sampling.orientation_priors import make_relion_translation_log_prior

pytestmark = pytest.mark.unit


def _write_map(path, value=1.0, corner=0):
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[corner:corner + 2, 2:4, 2:4] = value
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(volume)


def _body_star(tmp_path, offset_column="rlnBodySigmaOffset", reference=("ref1.mrc", "None")):
    (tmp_path / "masks").mkdir()
    for b in range(2):
        _write_map(tmp_path / "masks" / f"m{b}.mrc", corner=4 * b)
    _write_map(tmp_path / "ref1.mrc", value=0.5)
    (tmp_path / "star").mkdir()
    star = tmp_path / "star" / "bodies.star"
    columns = {"rlnBodyMaskName": ["masks/m0.mrc", "masks/m1.mrc"], "rlnBodyRotateRelativeTo": [2, 1],
               "rlnBodySigmaAngles": [10.0, 15.0]}
    if offset_column is not None:
        columns[offset_column] = [2.0, 3.0]
    columns["rlnBodyReferenceName"] = list(reference)
    starfile.write({"bodies": pd.DataFrame(columns)}, star)
    return star


def test_deprecated_sigma_offset_is_dropped_as_relion_5_drops_it(tmp_path):
    """ml_model.cpp:1266-1269 converts rlnBodySigmaOffset and never stores it: sigma_offset_bodies stays 0."""

    bodies = read_bodies(_body_star(tmp_path), pixel_size=1.7, paths_from=tmp_path)
    assert [b.sigma_offset_angstrom for b in bodies] == [0.0, 0.0]
    # The angular priors stay, so the body is refined, not fixed (ml_model.cpp:1336-1340).
    assert not any(b.keep_fixed for b in bodies)


def test_angstrom_sigma_offset_is_read_as_is(tmp_path):
    star = _body_star(tmp_path, offset_column="rlnBodySigmaOffsetAngst")
    bodies = read_bodies(star, pixel_size=1.7, paths_from=tmp_path)
    assert [b.sigma_offset_angstrom for b in bodies] == [2.0, 3.0]


def test_missing_sigma_offset_is_refused(tmp_path):
    with pytest.raises(ValueError, match="rlnBodySigmaOffsetAngst"):
        read_bodies(_body_star(tmp_path, offset_column=None), pixel_size=1.7, paths_from=tmp_path)


def test_body_reference_names_resolve_from_relion_directory(tmp_path):
    bodies = read_bodies(_body_star(tmp_path), pixel_size=1.7, paths_from=tmp_path)
    assert bodies[0].reference_path == tmp_path / "ref1.mrc"
    assert bodies[1].reference_path is None


def test_zero_offset_width_flattens_the_offset_prior():
    """The accelerated path's pdf is 1 at every offset when sigma2 < 0.0001: tdiff2 is NaN or -inf, never > 0
    (acc_ml_optimiser_impl.h:3067-3085)."""

    translations = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, -2.0]])
    prior = make_relion_translation_log_prior(translations, 1.7, 0.0, np.zeros(2))
    assert np.array_equal(prior, np.zeros(3, dtype=np.float32))


def _write_star(path, blocks):
    with open(path, "w") as out:
        for name, block in blocks.items():
            out.write(f"data_{name}\n\n")
            if isinstance(block, dict):
                for key, value in block.items():
                    out.write(f"_{key} {value}\n")
            else:
                out.write("loop_\n" + "".join(f"_{c}\n" for c in block.columns))
                out.write(block.to_csv(sep=" ", header=False, index=False))
            out.write("\n")


def _pre4_consensus(root):
    (root / "Example").mkdir()
    shells = pd.DataFrame({"rlnSpectralIndex": [0, 1], "rlnSsnrMap": [9.0, 1.0], "rlnGoldStandardFsc": [1.0, 0.5],
                           "rlnReferenceTau2": [1.0, 0.1]})
    groups = pd.DataFrame({"rlnGroupNumber": [1, 2], "rlnGroupName": ["a", "b"], "rlnGroupNrParticles": [2, 2],
                           "rlnGroupScaleCorrection": [0.9, 1.1]})
    for h in (1, 2):
        _write_star(root / f"Example/half{h}_model.star", {
            "model_general": {"rlnPixelSize": 2.0, "rlnNrClasses": 1, "rlnNrBodies": 1, "rlnNrGroups": 2},
            "model_classes": pd.DataFrame({
                "rlnReferenceImage": [f"Example/half{h}.mrc"], "rlnAccuracyTranslations": [0.5],
            }),
            "model_class_1": shells,
            "model_groups": groups,
            "model_group_1": pd.DataFrame({"rlnSpectralIndex": [0, 1], "rlnSigma2Noise": [4.0, 3.0]}),
            "model_group_2": pd.DataFrame({"rlnSpectralIndex": [0, 1], "rlnSigma2Noise": [8.0, 7.0]}),
        })
    _write_star(root / "Example/sampling.star", {"sampling_general": {"rlnHealpixOrder": 4}})
    starfile.write({
        "optics": pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [2.0]}),
        "particles": pd.DataFrame({"rlnGroupName": ["a", "b"], "rlnRandomSubset": [1, 2]}),
    }, root / "Example/data.star")
    _write_star(root / "Example/optimiser.star", {"optimiser_general": {
        "rlnDoSplitRandomHalves": 1, "rlnModelStarFile": "Example/half1_model.star",
        "rlnModelStarFile2": "Example/half2_model.star", "rlnExperimentalDataStarFile": "Example/data.star",
        "rlnOrientSamplingStarFile": "Example/sampling.star", "rlnOverallAccuracyTranslations": 0.5,
    }})
    return root / "Example/optimiser.star"


def test_pre_relion_4_model_is_read_as_relion_5_reads_it(tmp_path):
    checkpoint = read_relion_checkpoint(_pre4_consensus(tmp_path), relion_cwd=tmp_path)
    model = checkpoint.models[0]
    assert model.general["rlnNrOpticsGroups"] == "1"
    # ml_model.cpp:396-400: the first group's noise, not the others'.
    assert model.sigma2_noise[0].tolist() == [4.0, 3.0]
    # ml_model.cpp:223-228 converts pixels to Angstroms; ml_optimiser.cpp:1497-1501 takes the pixel value as it is.
    assert float(model.classes["rlnAccuracyTranslationsAngst"][0]) == 1.0
    assert float(checkpoint.optimiser["rlnOverallAccuracyTranslationsAngst"]) == 0.5
    assert model.group_scale.tolist() == [0.9, 1.1]
