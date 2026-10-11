# RELION-written tilt-series STAR text (test input)

Excerpts of STAR files exactly as RELION 5 wrote them, for the reader tests in
`tests/unit/test_tomo_relion_star_forms.py` (relax#79). Text only: no images. Do not reformat these files; their
whitespace, comments and block forms are what the tests are about.

Source: the precalculated RELION-5 subtomogram-averaging tutorial project, Zenodo doi 10.5281/zenodo.11068319
(Burt, Toader et al., "An image processing pipeline for electron cryo-tomography in RELION-5"; CC-BY-4.0).

| file | written by | content |
| --- | --- | --- |
| `Extract/job010/particles.star` | `relion_tomo_subtomo --stack2d --float16`, RELION 5.0-beta-2 (commit a0b145) | header and the first three particles of TS_01 and of TS_03 |
| `Extract_relion_5.0.1/job010/particles.star` | the same command re-run with RELION 5.0.1 on the shipped picks | the same six particles (column order and subtomogram angles differ: the picks were regenerated after the shipped extraction) |
| `Denoise/job008/tomograms.star` | the `starfile` Python package (RELION's denoising wrapper) | header and the rows of TS_01 and TS_03 |
| `Denoise/job008/tilt_series/TS_01.star` | RELION 5.0-beta-2 | complete; has `rlnCtfScalefactor` |
| `Denoise/job008/tilt_series/TS_03.star` | RELION 5.0-beta-2 | complete; has no `rlnCtfScalefactor` |

Paths inside the files are relative to the RELION project directory, as RELION writes them.
