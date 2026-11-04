"""The last iteration's run_model.star (relax.multibody.run_files.write_relion_model)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import starfile

from relax.multibody.run_files import write_relion_model


def _track(reference_sigma2):
    shells = 5
    return SimpleNamespace(
        tau2=[np.linspace(1.0, 2.0, shells) * 8.0**4], data_vs_prior=np.full(shells, 999.0), fsc=np.ones(shells),
        fourier_coverage=None, reference_sigma2=reference_sigma2, acc_rot=1.0, acc_trans=1.0, keep_fixed=False,
    )


def test_final_model_star_writes_reference_sigma2_in_relion_units(tmp_path):
    template = tmp_path / "template_model.star"
    starfile.write({
        "model_general": {"rlnPixelSize": 2.0, "rlnOriginalImageSize": 8},
        "model_groups": pd.DataFrame({"rlnGroupNumber": [1], "rlnGroupScaleCorrection": [1.0]}),
        "model_pdf_orient_class_1": pd.DataFrame({"rlnOrientationDistribution": [1.0]}),
    }, template)
    box = 8
    frame = float(box) ** 4
    sigma2 = np.array([4.0, 3.0, 2.0, 1.0, 0.5]) * frame  # relax units: RELION's times ori_size**4
    state = SimpleNamespace(
        current_resolution_inv=0.1, current_size=box, ave_pmax=0.5,
        sampling=SimpleNamespace(sigma_rot=0.01, sigma_psi=0.01),
        bodies=[_track(sigma2), _track(None)],
        noise=SimpleNamespace(radial_per_half=[np.ones(5) * frame]),
    )
    run = SimpleNamespace(
        checkpoint=SimpleNamespace(model_paths=[template]), box_size=box, pixel_size=2.0,
        bodies=[SimpleNamespace(orientation=np.eye(3)) for _ in range(2)],
    )
    write_relion_model(tmp_path, state, run)
    blocks = starfile.read(tmp_path / "run_model.star", always_dict=True)
    written = blocks["model_body_1"]["rlnReferenceSigma2"].to_numpy()
    np.testing.assert_allclose(written, sigma2 / frame, rtol=1e-7)
    # tau2 = ssnr * sigma2 in both programs, so the two columns are written in the same units
    np.testing.assert_allclose(blocks["model_body_1"]["rlnReferenceTau2"].to_numpy(), np.linspace(1.0, 2.0, 5), rtol=1e-7)
    # a body without the final iteration's sigma2 keeps RELION's unset zeros
    np.testing.assert_array_equal(blocks["model_body_2"]["rlnReferenceSigma2"].to_numpy(), np.zeros(5))
