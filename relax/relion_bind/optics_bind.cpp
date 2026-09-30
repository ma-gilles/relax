/**
 * Optics-group CTFs and aberrations through RELION's ObservationModel: test oracles.
 *
 * relax production does not call these (nor any RELION code): its CTF rows are
 * relax.relion.relion_ctf.relion_ctf_fftw_half and its aberration phases
 * relax.relion.optics_aberrations; these functions are what the unit tests
 * compare them with.
 *
 * The optics table is read from the particle STAR with RELION's own
 * MetaDataTable, and ObservationModel (obs_model_stub.cpp, RELION's code)
 * turns it into what relion_refine uses per optics group: kV, Cs and Q0,
 * the anisotropic magnification matrix, the even Zernike gamma offset and
 * the odd Zernike phase (beam tilt included).
 *
 *   optics_ctf_images_batch  CTF::setValuesByGroup + CTF::getFftwImage per
 *                            particle, as ml_optimiser.cpp:6461-6484 and
 *                            acc_ml_optimiser_impl.h:826-837 call them.
 *   optics_phase_correction  ObservationModel::getPhaseCorrection, the factor
 *                            whose conjugate demodulatePhase multiplies into
 *                            every image.
 *   optics_gamma_offset      ObservationModel::getGammaOffset.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <src/ctf.h>
#include <src/metadata_table.h>
#include <src/jaz/single_particle/obs_model.h>

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

static ObservationModel read_optics(const std::string &star_path) {
    MetaDataTable optics;
    optics.read(star_path, "optics");
    if (optics.numberOfObjects() == 0) {
        throw std::invalid_argument("no optics table in " + star_path);
    }
    return ObservationModel(optics);
}

static int zero_based_group(const ObservationModel &obs, int optics_group) {
    const int group = optics_group - 1;
    if (group < 0 || group >= (int) obs.opticsMdt.numberOfObjects()) {
        throw std::invalid_argument("optics group " + std::to_string(optics_group) + " is not in the optics table");
    }
    return group;
}

/**
 * CTF images of N particles, shape (N, oriydim, orixdim/2+1).
 *
 * params has one row per particle: defU, defV, defAng, Bfac, scale,
 * phase_shift, optics group (1-based, as rlnOpticsGroup). kV, Cs, Q0, the
 * pixel size, the magnification and the even Zernike terms come from the
 * optics table, exactly as relion_refine takes them.
 */
static py::array_t<double> optics_ctf_images_batch(
    const std::string &star_path,
    py::array_t<double, py::array::c_style | py::array::forcecast> params,
    int orixdim,
    int oriydim,
    bool do_damping,
    int n_threads
) {
    if (params.ndim() != 2 || params.shape(1) != 7) {
        throw std::invalid_argument("params must have shape (N, 7)");
    }
    ObservationModel obs = read_optics(star_path);
    const long n = params.shape(0);
    const double *p = params.data();
    std::set<int> groups;
    for (long i = 0; i < n; ++i) {
        groups.insert(zero_based_group(obs, (int) p[7 * i + 6]));
    }
    // The gamma-offset cache is filled under an OpenMP critical section, which
    // is not compiled here: fill it for every group before the worker threads.
    if (obs.hasEvenZernike) {
        for (int group : groups) {
            obs.getGammaOffset(group, oriydim);
        }
    }
    const long ny = oriydim;
    const long nx = orixdim / 2 + 1;
    py::array_t<double> out({n, ny, nx});
    double *o = out.mutable_data();
    const int workers = std::max(1, std::min<int>(n_threads, static_cast<int>(std::max(1L, n))));
    {
        py::gil_scoped_release release;
        auto run = [&](long start, long stop) {
            for (long i = start; i < stop; ++i) {
                const double *r = p + 7 * i;
                const int group = (int) r[6] - 1;
                CTF ctf;
                ctf.setValuesByGroup(&obs, group, r[0], r[1], r[2], r[3], r[4], r[5], /*dose=*/-1.0);
                MultidimArray<RFLOAT> result(oriydim, orixdim / 2 + 1);
                ctf.getFftwImage(result, orixdim, oriydim, obs.getPixelSize(group),
                                 /*do_abs=*/false,
                                 /*do_only_flip_phases=*/false,
                                 /*do_intact_until_first_peak=*/false,
                                 do_damping,
                                 /*do_ctf_padding=*/false,
                                 /*do_intact_after_first_peak=*/false);
                std::memcpy(o + i * ny * nx, result.data, ny * nx * sizeof(double));
            }
        };
        std::vector<std::thread> threads;
        const long chunk = (n + workers - 1) / workers;
        for (int w = 0; w < workers; ++w) {
            const long start = w * chunk;
            const long stop = std::min(n, start + chunk);
            if (start < stop) {
                threads.emplace_back(run, start, stop);
            }
        }
        for (auto &t : threads) {
            t.join();
        }
    }
    return out;
}

/** getPhaseCorrection(group, s) on the FFTW half grid, shape (s, s/2+1), indexed [y][x]. */
static py::array_t<std::complex<double>> optics_phase_correction(const std::string &star_path, int optics_group, int s) {
    ObservationModel obs = read_optics(star_path);
    const int group = zero_based_group(obs, optics_group);
    const int sh = s / 2 + 1;
    py::array_t<std::complex<double>> out({(long) s, (long) sh});
    auto *o = out.mutable_data();
    if (!obs.hasOddZernike) {
        for (long i = 0; i < (long) s * sh; ++i) {
            o[i] = std::complex<double>(1.0, 0.0);
        }
        return out;
    }
    const BufferedImage<Complex> &corr = obs.getPhaseCorrection(group, s);
    for (int y = 0; y < s; ++y) {
        for (int x = 0; x < sh; ++x) {
            o[(long) y * sh + x] = std::complex<double>(corr(x, y).real, corr(x, y).imag);
        }
    }
    return out;
}

/** getGammaOffset(group, s) on the FFTW half grid, shape (s, s/2+1), indexed [y][x]. */
static py::array_t<double> optics_gamma_offset(const std::string &star_path, int optics_group, int s) {
    ObservationModel obs = read_optics(star_path);
    const int group = zero_based_group(obs, optics_group);
    const int sh = s / 2 + 1;
    py::array_t<double> out({(long) s, (long) sh});
    double *o = out.mutable_data();
    if (!obs.hasEvenZernike) {
        std::fill(o, o + (long) s * sh, 0.0);
        return out;
    }
    const BufferedImage<RFLOAT> &gamma = obs.getGammaOffset(group, s);
    for (int y = 0; y < s; ++y) {
        for (int x = 0; x < sh; ++x) {
            o[(long) y * sh + x] = gamma(x, y);
        }
    }
    return out;
}

void init_optics_bindings(py::module_ &m) {
    m.def("optics_ctf_images_batch", &optics_ctf_images_batch,
          py::arg("star_path"),
          py::arg("params"),
          py::arg("orixdim"),
          py::arg("oriydim"),
          py::arg("do_damping") = false,
          py::arg("n_threads") = 1,
          R"doc(
CTF::setValuesByGroup + getFftwImage for N rows of (defU, defV, defAng, Bfac,
scale, phase_shift, optics group); kV, Cs, Q0, pixel size, magnification and
even Zernike terms come from the STAR's optics table. Shape (N, oriydim, orixdim/2+1).
)doc");
    m.def("optics_phase_correction", &optics_phase_correction,
          py::arg("star_path"), py::arg("optics_group"), py::arg("size"),
          "ObservationModel::getPhaseCorrection (odd Zernike and beam tilt), shape (size, size/2+1).");
    m.def("optics_gamma_offset", &optics_gamma_offset,
          py::arg("star_path"), py::arg("optics_group"), py::arg("size"),
          "ObservationModel::getGammaOffset (even Zernike), shape (size, size/2+1).");
}
