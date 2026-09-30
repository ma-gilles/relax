"""relax refuses optics-table features it does not model (beam tilt, Zernike, magnification, premultiplied CTF)."""

import pandas as pd
import pytest

from relax.relion.relion_metadata import refuse_unsupported_optics


@pytest.mark.unit
def test_unsupported_optics_features_are_refused_and_neutral_values_accepted():
    base = {"_rlnOpticsGroup": [1, 2], "_rlnVoltage": [300.0, 300.0]}
    neutral = pd.DataFrame(
        {
            **base,
            "_rlnBeamTiltX": [0.0, 0.0],
            "_rlnOddZernike": ["[0.0,0.0]", "[0,0]"],
            "_rlnMagMat00": [1.0, 1.0],
            "_rlnMagMat01": [0.0, 0.0],
            "_rlnCtfDataAreCtfPremultiplied": [0, 0],
        }
    )
    refuse_unsupported_optics(neutral, source="neutral.star")
    refuse_unsupported_optics(None, source="no optics")
    for label, value in (
        ("_rlnBeamTiltY", [0.0, 0.3]),
        ("_rlnEvenZernike", ["[0,0,0.1]", "[0,0,0]"]),
        ("_rlnMagMat11", [1.0, 1.01]),
        ("_rlnCtfDataAreCtfPremultiplied", [1, 1]),
    ):
        with pytest.raises(NotImplementedError, match=label.lstrip("_")):
            refuse_unsupported_optics(pd.DataFrame({**base, label: value}), source="bad.star")


@pytest.mark.unit
def test_ctf_corrected_images_and_several_mtfs_are_refused():
    base = {"_rlnOpticsGroup": [1, 2]}
    refuse_unsupported_optics(pd.DataFrame({**base, "_rlnCtfDataAreCtfCorrected": [0, 0]}), source="ok.star")
    refuse_unsupported_optics(pd.DataFrame({**base, "_rlnMtfFileName": ["k3.star", "k3.star"]}), source="ok.star")
    with pytest.raises(NotImplementedError, match="rlnCtfDataAreCtfCorrected"):
        refuse_unsupported_optics(pd.DataFrame({**base, "_rlnCtfDataAreCtfCorrected": [0, 1]}), source="bad.star")
    with pytest.raises(NotImplementedError, match="several different MTFs"):
        refuse_unsupported_optics(pd.DataFrame({**base, "_rlnMtfFileName": ["k2.star", "k3.star"]}), source="bad.star")
