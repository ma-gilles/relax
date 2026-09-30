/**
 * The subset of RELION's ObservationModel the binding needs, copied verbatim.
 *
 * ctf.cpp references ObservationModel for CTF::setValuesByGroup and
 * getFftwImage (magnification matrices, even Zernike gamma offsets,
 * premultiplied CTFs); the optics test oracles (optics_bind.cpp) also read its
 * odd-Zernike phase correction. The full obs_model.cpp pulls in most of jaz,
 * so these functions are copied unchanged from RELION 5.0.1 f2c1a38
 * src/jaz/single_particle/obs_model.cpp (lines 40-213, 586-626, 689-692,
 * 738-746, 765-768, 790-800, 1215-1307, 1309-1330) with TiltHelper::insertTilt from
 * src/jaz/single_particle/ctf/tilt_helper.cpp (lines 557-574). The Zernike
 * polynomials come from RELION's own src/jaz/math/Zernike.cpp.
 */

#include <src/jaz/single_particle/obs_model.h>
#include <src/jaz/single_particle/ctf/tilt_helper.h>
#include <src/jaz/math/Zernike.h>
#include <src/error.h>

void TiltHelper::insertTilt(
	std::vector<double>& oddZernikeCoeffs,
	double tilt_x, double tilt_y,
	double Cs, double lambda)
{
	if (oddZernikeCoeffs.size() <= 5)
		oddZernikeCoeffs.resize(5, 0);

	const double scale = Cs * 20000 * lambda * lambda * 3.141592654;
	const double Z3x = -scale * tilt_x / 3.0;
	const double Z3y = -scale * tilt_y / 3.0;

	oddZernikeCoeffs[1] += 2.0 * Z3x;
	oddZernikeCoeffs[0] += 2.0 * Z3y;

	oddZernikeCoeffs[4] += Z3x;
	oddZernikeCoeffs[3] += Z3y;
}

ObservationModel::ObservationModel()
{
}

ObservationModel::ObservationModel(const MetaDataTable &_opticsMdt, bool do_die_upon_error)
:	opticsMdt(_opticsMdt),
	angpix(_opticsMdt.numberOfObjects()),
	lambda(_opticsMdt.numberOfObjects()),
	Cs(_opticsMdt.numberOfObjects()),
	boxSizes(_opticsMdt.numberOfObjects(), 0.0),
    CtfPremultiplied(_opticsMdt.numberOfObjects(), false),
    CtfCorrected(_opticsMdt.numberOfObjects(), false)
{
	if (!(opticsMdt.containsLabel(EMDL_IMAGE_PIXEL_SIZE) ||
	      opticsMdt.containsLabel(EMDL_MICROGRAPH_PIXEL_SIZE) ||
	      opticsMdt.containsLabel(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE) ||
	      opticsMdt.containsLabel(EMDL_TOMO_TILT_SERIES_PIXEL_SIZE))
	  || !opticsMdt.containsLabel(EMDL_CTF_VOLTAGE)
	  || !opticsMdt.containsLabel(EMDL_CTF_CS))
	{
		if (do_die_upon_error)
		{
			REPORT_ERROR_STR("ERROR: not all necessary variables defined in _optics.star file: "
			              << "rlnVoltage, rlnSphericalAberration and one of rlnImagePixelSize, rlnMicrographPixelSize or rlnOriginalMicrographPixelSize. Make sure to convert older STAR files anew in version-3.1, "
			              << "with relion_convert_star.");
		}
		else
		{
			opticsMdt.clear();
			return;
		}
	}

	// symmetrical high-order aberrations:
	hasEvenZernike = opticsMdt.containsLabel(EMDL_IMAGE_EVEN_ZERNIKE_COEFFS);
	evenZernikeCoeffs = std::vector<std::vector<double> >(opticsMdt.numberOfObjects(), std::vector<double>(0));
	gammaOffset = std::vector<std::map<int,BufferedImage<RFLOAT> > >(opticsMdt.numberOfObjects());

	// antisymmetrical high-order aberrations:
	hasOddZernike = opticsMdt.containsLabel(EMDL_IMAGE_ODD_ZERNIKE_COEFFS);
	oddZernikeCoeffs = std::vector<std::vector<double> >(opticsMdt.numberOfObjects(), std::vector<double>(0));
	phaseCorr = std::vector<std::map<int,BufferedImage<Complex> > >(opticsMdt.numberOfObjects());

	const bool hasTilt = opticsMdt.containsLabel(EMDL_IMAGE_BEAMTILT_X)
	                  || opticsMdt.containsLabel(EMDL_IMAGE_BEAMTILT_Y);

	// anisotropic magnification:
	hasMagMatrices = opticsMdt.containsLabel(EMDL_IMAGE_MAG_MATRIX_00)
	              || opticsMdt.containsLabel(EMDL_IMAGE_MAG_MATRIX_01)
	              || opticsMdt.containsLabel(EMDL_IMAGE_MAG_MATRIX_10)
	              || opticsMdt.containsLabel(EMDL_IMAGE_MAG_MATRIX_11);

	magMatrices.resize(opticsMdt.numberOfObjects());

	hasBoxSizes = opticsMdt.containsLabel(EMDL_IMAGE_SIZE);

	if (opticsMdt.containsLabel(EMDL_IMAGE_OPTICS_GROUP_NAME))
	{
		groupNames.resize(opticsMdt.numberOfObjects());
	}

	if (opticsMdt.containsLabel(EMDL_IMAGE_MTF_FILENAME))
	{
		fnMtfs.resize(opticsMdt.numberOfObjects());
		mtfImage = std::vector<std::map<int,BufferedImage<RFLOAT> > >(opticsMdt.numberOfObjects());
	}
	if (opticsMdt.containsLabel(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE))
	{
		originalAngpix.resize(opticsMdt.numberOfObjects());
	}

	for (int i = 0; i < opticsMdt.numberOfObjects(); i++)
	{
		if (!opticsMdt.getValue(EMDL_IMAGE_PIXEL_SIZE, angpix[i], i))
		{
			if (!opticsMdt.getValue(EMDL_MICROGRAPH_PIXEL_SIZE, angpix[i], i))
			{
				if (!opticsMdt.getValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, angpix[i], i))
				{
					opticsMdt.getValue(EMDL_TOMO_TILT_SERIES_PIXEL_SIZE, angpix[i], i);
				}
			}
		}

		if (opticsMdt.containsLabel(EMDL_IMAGE_OPTICS_GROUP_NAME))
		{
			opticsMdt.getValue(EMDL_IMAGE_OPTICS_GROUP_NAME, groupNames[i], i);
		}

		if (opticsMdt.containsLabel(EMDL_IMAGE_MTF_FILENAME))
		{
			opticsMdt.getValue(EMDL_IMAGE_MTF_FILENAME, fnMtfs[i], i);
		}

		if (opticsMdt.containsLabel(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE))
		{
			opticsMdt.getValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE, originalAngpix[i], i);
		}

		if (opticsMdt.containsLabel(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED))
		{
			bool val;
			opticsMdt.getValue(EMDL_OPTIMISER_DATA_ARE_CTF_PREMULTIPLIED, val, i);
			CtfPremultiplied[i] = val;
		}
        if (opticsMdt.containsLabel(EMDL_OPTIMISER_DATA_ARE_CTF_CORRECTED))
        {
            bool val;
            opticsMdt.getValue(EMDL_OPTIMISER_DATA_ARE_CTF_CORRECTED, val, i);
            CtfCorrected[i] = val;
        }

		opticsMdt.getValue(EMDL_IMAGE_SIZE, boxSizes[i], i);

		double kV;
		opticsMdt.getValue(EMDL_CTF_VOLTAGE, kV, i);
		double V = kV * 1e3;
		lambda[i] = 12.2643247 / sqrt(V * (1.0 + V * 0.978466e-6));

		opticsMdt.getValue(EMDL_CTF_CS, Cs[i], i);

		if (hasEvenZernike)
		{
			opticsMdt.getValue(EMDL_IMAGE_EVEN_ZERNIKE_COEFFS, evenZernikeCoeffs[i], i);
		}

		if (hasOddZernike)
		{
			opticsMdt.getValue(EMDL_IMAGE_ODD_ZERNIKE_COEFFS, oddZernikeCoeffs[i], i);
		}

		if (hasTilt)
		{
			double tx(0), ty(0);

			opticsMdt.getValue(EMDL_IMAGE_BEAMTILT_X, tx, i);
			opticsMdt.getValue(EMDL_IMAGE_BEAMTILT_Y, ty, i);

			if (!hasOddZernike)
			{
				oddZernikeCoeffs[i] = std::vector<double>(6, 0.0);
			}

			TiltHelper::insertTilt(oddZernikeCoeffs[i], tx, ty, Cs[i], lambda[i]);
		}

		// always keep a set of mag matrices
		// if none are defined, keep a set of identity matrices

		magMatrices[i] = Matrix2D<RFLOAT>(2,2);
		magMatrices[i].initIdentity();

		// See if there is more than one MTF, for more rapid divideByMtf
		hasMultipleMtfs = false;
		for (int j = 1; j < fnMtfs.size(); j++)
		{
			if (fnMtfs[j] != fnMtfs[0])
			{
				hasMultipleMtfs = true;
				break;
			}
		}

		if (hasMagMatrices)
		{
			opticsMdt.getValue(EMDL_IMAGE_MAG_MATRIX_00, magMatrices[i](0,0), i);
			opticsMdt.getValue(EMDL_IMAGE_MAG_MATRIX_01, magMatrices[i](0,1), i);
			opticsMdt.getValue(EMDL_IMAGE_MAG_MATRIX_10, magMatrices[i](1,0), i);
			opticsMdt.getValue(EMDL_IMAGE_MAG_MATRIX_11, magMatrices[i](1,1), i);
		}
	}

	if (hasTilt) hasOddZernike = true;
}

void ObservationModel::demodulatePhase(
		const MetaDataTable& partMdt, long particle,
		MultidimArray<Complex>& obsImage,
		bool do_modulate_instead)
{
	int opticsGroup;
	partMdt.getValue(EMDL_IMAGE_OPTICS_GROUP, opticsGroup, particle);
	opticsGroup--;

	demodulatePhase(opticsGroup, obsImage, do_modulate_instead);
}

void ObservationModel::demodulatePhase(int opticsGroup, MultidimArray<Complex>& obsImage,
                                       bool do_modulate_instead)
{
	const int s = obsImage.ydim;
	const int sh = obsImage.xdim;

	if (oddZernikeCoeffs.size() > opticsGroup
			&& oddZernikeCoeffs[opticsGroup].size() > 0)
	{
		const BufferedImage<Complex>& corr = getPhaseCorrection(opticsGroup, s);

		if (do_modulate_instead)
		{
			for (int y = 0; y < s;  y++)
			for (int x = 0; x < sh; x++)
			{
				obsImage(y,x) *= corr(x,y);
			}
		}
		else
		{
			for (int y = 0; y < s;  y++)
			for (int x = 0; x < sh; x++)
			{
				obsImage(y,x) *= corr(x,y).conj();
			}
		}
	}
}

double ObservationModel::getPixelSize(int opticsGroup) const
{
	return angpix[opticsGroup];
}

int ObservationModel::getBoxSize(int opticsGroup) const
{
	if (!hasBoxSizes)
	{
		REPORT_ERROR("ObservationModel::getBoxSize: box sizes not available. Make sure particle images are available before converting/importing STAR files from earlier versions of RELION.\n");
	}

	return boxSizes[opticsGroup];
}

Matrix2D<RFLOAT> ObservationModel::getMagMatrix(int opticsGroup) const
{
	return magMatrices[opticsGroup];
}

bool ObservationModel::getCtfPremultiplied(int og) const
{
	if (og < CtfPremultiplied.size())
	{
		return CtfPremultiplied[og];
	}
	else
	{
		return false;
	}
}

const BufferedImage<Complex>& ObservationModel::getPhaseCorrection(int optGroup, int s)
{
	#pragma omp critical(ObservationModel_getPhaseCorrection)
	{
		if (phaseCorr[optGroup].find(s) == phaseCorr[optGroup].end())
		{
			if (phaseCorr[optGroup].size() > 100)
			{
				std::cerr << "Warning: " << (phaseCorr[optGroup].size()+1)
				          << " phase shift images in cache for the same ObservationModel." << std::endl;
			}

			const int sh = s/2 + 1;

			phaseCorr[optGroup][s] = BufferedImage<Complex>(sh,s);

			BufferedImage<Complex>& img = phaseCorr[optGroup][s];
			const double as = angpix[optGroup] * boxSizes[optGroup];
			const Matrix2D<RFLOAT>& M = magMatrices[optGroup];
			
			for (int y = 0; y < s;  y++)
			for (int x = 0; x < sh; x++)
			{
				double phase = 0.0;

				for (int i = 0; i < oddZernikeCoeffs[optGroup].size(); i++)
				{
					int m, n;
					Zernike::oddIndexToMN(i, m, n);

					const double xx0 = x/as;
					const double yy0 = y < sh-1? y/as : (y-s)/as;
					
					const double xx = M(0,0) * xx0 + M(0,1) * yy0;
					const double yy = M(1,0) * xx0 + M(1,1) * yy0;

					phase += oddZernikeCoeffs[optGroup][i] * Zernike::Z_cart(m,n,xx,yy);
				}

				img(x,y).real = cos(phase);
				img(x,y).imag = sin(phase);
			}
		}
	}

	return phaseCorr[optGroup][s];
}

const BufferedImage<RFLOAT>& ObservationModel::getGammaOffset(int optGroup, int s)
{
	#pragma omp critical(ObservationModel_getGammaOffset)
	{
		if (gammaOffset[optGroup].find(s) == gammaOffset[optGroup].end())
		{
			if (gammaOffset[optGroup].size() > 100)
			{
				std::cerr << "Warning: " << (gammaOffset[optGroup].size()+1)
				          << " gamma offset images in cache for the same ObservationModel." << std::endl;
			}

			const int sh = s/2 + 1;
			gammaOffset[optGroup][s] = BufferedImage<RFLOAT>(sh,s);
			BufferedImage<RFLOAT>& img = gammaOffset[optGroup][s];

			const double as = angpix[optGroup] * boxSizes[optGroup];
			const Matrix2D<RFLOAT>& M = magMatrices[optGroup];

			for (int y = 0; y < s;  y++)
			for (int x = 0; x < sh; x++)
			{
				double phase = 0.0;

				for (int i = 0; i < evenZernikeCoeffs[optGroup].size(); i++)
				{
					int m, n;
					Zernike::evenIndexToMN(i, m, n);

					const double xx0 = x/as;
					const double yy0 = y < sh-1? y/as : (y-s)/as;
					
					const double xx = M(0,0) * xx0 + M(0,1) * yy0;
					const double yy = M(1,0) * xx0 + M(1,1) * yy0;

					phase += evenZernikeCoeffs[optGroup][i] * Zernike::Z_cart(m,n,xx,yy);
				}

				img(x,y) = phase;
			}
		}
	}

	return gammaOffset[optGroup][s];
}


Matrix2D<RFLOAT> ObservationModel::applyAnisoMag(Matrix2D<RFLOAT> A3D, int opticsGroup)
{
	Matrix2D<RFLOAT> out;

	if (hasMagMatrices)
	{
		Matrix2D<RFLOAT> mag3D(3,3);
		mag3D.initIdentity();

		mag3D(0,0) = magMatrices[opticsGroup](0,0);
		mag3D(0,1) = magMatrices[opticsGroup](0,1);
		mag3D(1,0) = magMatrices[opticsGroup](1,0);
		mag3D(1,1) = magMatrices[opticsGroup](1,1);
		out = mag3D.inv() * A3D;
	}
	else
	{
		out = A3D;
	}

	return out;
}
