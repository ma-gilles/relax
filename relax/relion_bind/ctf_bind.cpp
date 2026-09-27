/**
 * CTF bindings: getFftwImage with and without do_ctf_padding.
 *
 * Phase 2 (P1): CTF::getFftwImage — compare RELION's 2×-padded CTF
 * against recovar's direct CTF computation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <src/ctf.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

namespace py = pybind11;

/**
 * Compute a CTF image using RELION's CTF::getFftwImage.
 *
 * Returns an FFTW half-transform of shape (oriydim, orixdim/2+1).
 *
 * Parameters match CTF::setValues + getFftwImage:
 *   defU, defV     : defocus in angstroms (positive = underfocused)
 *   defAng         : defocus angle in degrees
 *   voltage        : accelerating voltage in kV
 *   Cs             : spherical aberration in mm
 *   Q0             : amplitude contrast (0.07 cryo, 0.2 stain)
 *   Bfac           : B-factor (0 for no damping)
 *   angpix         : pixel size in angstroms
 *   orixdim        : box size in x (pixels, full real-space)
 *   oriydim        : box size in y (pixels, full real-space)
 *   do_ctf_padding : if true, compute in 2× box then downsample
 *   do_abs         : return |CTF| instead of signed CTF
 *   do_damping     : apply B-factor damping
 */
static py::array_t<double> get_ctf_image(
    double defU,
    double defV,
    double defAng,
    double voltage,
    double Cs,
    double Q0,
    double Bfac,
    double angpix,
    int orixdim,
    int oriydim,
    bool do_ctf_padding,
    bool do_abs,
    bool do_damping,
    double phase_shift,
    double scale
) {
    CTF ctf;
    ctf.setValues(defU, defV, defAng, voltage, Cs, Q0, Bfac, scale,
                  phase_shift, /*dose=*/-1.0);

    MultidimArray<RFLOAT> result(oriydim, orixdim / 2 + 1);

    ctf.getFftwImage(result, orixdim, oriydim, angpix,
                     do_abs,
                     /*do_only_flip_phases=*/false,
                     /*do_intact_until_first_peak=*/false,
                     do_damping,
                     do_ctf_padding,
                     /*do_intact_after_first_peak=*/false);

    long ny = YSIZE(result);
    long nx = XSIZE(result);
    py::array_t<double> out({ny, nx});
    auto buf = out.request();
    std::memcpy(buf.ptr, result.data, ny * nx * sizeof(double));
    return out;
}

/**
 * Compute N CTF images with get_ctf_image's arithmetic on worker threads.
 *
 * params has one row per image: defU, defV, defAng, voltage, Cs, Q0, Bfac,
 * angpix, phase_shift, scale. Each row runs the same CTF::setValues and
 * getFftwImage as get_ctf_image, so every image is bitwise identical to a
 * single call; rows are independent, so they are split across n_threads
 * with the GIL released.
 */
static py::array_t<double> get_ctf_images_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> params,
    int orixdim,
    int oriydim,
    bool do_ctf_padding,
    bool do_abs,
    bool do_damping,
    int n_threads
) {
    if (params.ndim() != 2 || params.shape(1) != 10) {
        throw std::invalid_argument("params must have shape (N, 10)");
    }
    const long n = params.shape(0);
    const long ny = oriydim;
    const long nx = orixdim / 2 + 1;
    py::array_t<double> out({n, ny, nx});
    const double *p = params.data();
    double *o = out.mutable_data();
    const int workers = std::max(1, std::min<int>(n_threads, static_cast<int>(std::max(1L, n))));
    {
        py::gil_scoped_release release;
        auto run = [&](long start, long stop) {
            for (long i = start; i < stop; ++i) {
                const double *r = p + 10 * i;
                CTF ctf;
                ctf.setValues(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[9], r[8], /*dose=*/-1.0);
                MultidimArray<RFLOAT> result(oriydim, orixdim / 2 + 1);
                ctf.getFftwImage(result, orixdim, oriydim, r[7],
                                 do_abs,
                                 /*do_only_flip_phases=*/false,
                                 /*do_intact_until_first_peak=*/false,
                                 do_damping,
                                 do_ctf_padding,
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

/**
 * Compute the raw CTF value at a single frequency.
 *
 * Useful for unit tests: getCTF(x, y) where x, y are in 1/Å.
 */
static double get_ctf_value(
    double defU,
    double defV,
    double defAng,
    double voltage,
    double Cs,
    double Q0,
    double Bfac,
    double angpix,
    double freq_x,
    double freq_y,
    double phase_shift,
    double scale
) {
    CTF ctf;
    ctf.setValues(defU, defV, defAng, voltage, Cs, Q0, Bfac, scale,
                  phase_shift, /*dose=*/-1.0);
    return ctf.getCTF(freq_x, freq_y,
                      /*do_abs=*/false,
                      /*do_only_flip_phases=*/false,
                      /*do_intact_until_first_peak=*/false,
                      /*do_damping=*/(Bfac != 0.0));
}


void init_ctf_bindings(py::module_ &m) {
    m.def("get_ctf_image", &get_ctf_image,
          py::arg("defU"),
          py::arg("defV"),
          py::arg("defAng"),
          py::arg("voltage"),
          py::arg("Cs"),
          py::arg("Q0"),
          py::arg("Bfac") = 0.0,
          py::arg("angpix") = 1.0,
          py::arg("orixdim") = 128,
          py::arg("oriydim") = 128,
          py::arg("do_ctf_padding") = false,
          py::arg("do_abs") = false,
          py::arg("do_damping") = false,
          py::arg("phase_shift") = 0.0,
          py::arg("scale") = 1.0,
          R"doc(
Compute CTF image using RELION's CTF::getFftwImage.

Returns FFTW half-transform, shape (oriydim, orixdim/2+1).

Parameters
----------
defU, defV : float
    Defocus in angstroms (positive = underfocused).
defAng : float
    Defocus angle in degrees.
voltage : float
    Accelerating voltage in kV.
Cs : float
    Spherical aberration in mm.
Q0 : float
    Amplitude contrast.
Bfac : float
    B-factor (0 = no damping).
angpix : float
    Pixel size in angstroms.
orixdim, oriydim : int
    Box size in pixels.
do_ctf_padding : bool
    If True, compute in 2× box then downsample (RELION default in GUI).
do_abs : bool
    Return |CTF| instead of signed CTF.
do_damping : bool
    Apply B-factor damping.
phase_shift : float
    Phase plate shift in degrees.
scale : float
    CTF scale factor.
)doc");

    m.def("get_ctf_images_batch", &get_ctf_images_batch,
          py::arg("params"),
          py::arg("orixdim"),
          py::arg("oriydim"),
          py::arg("do_ctf_padding") = false,
          py::arg("do_abs") = false,
          py::arg("do_damping") = false,
          py::arg("n_threads") = 1,
          R"doc(
get_ctf_image for N rows of (defU, defV, defAng, voltage, Cs, Q0, Bfac, angpix,
phase_shift, scale) on n_threads worker threads; shape (N, oriydim, orixdim/2+1).
)doc");

    m.def("get_ctf_value", &get_ctf_value,
          py::arg("defU"),
          py::arg("defV"),
          py::arg("defAng"),
          py::arg("voltage"),
          py::arg("Cs"),
          py::arg("Q0"),
          py::arg("Bfac") = 0.0,
          py::arg("angpix") = 1.0,
          py::arg("freq_x") = 0.0,
          py::arg("freq_y") = 0.0,
          py::arg("phase_shift") = 0.0,
          py::arg("scale") = 1.0,
          R"doc(
Compute a single CTF value at frequency (freq_x, freq_y) in 1/Å.
)doc");
}
