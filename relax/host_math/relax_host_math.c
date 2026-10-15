/* glibc's acos and atan2 over arrays, for relax/sampling/libm.py.
 *
 * Host code only (no CUDA). Built without -ffast-math and with tree vectorization off, so every element is one call
 * of the C library's scalar acos/atan2 -- the functions CPython's math module calls -- and no vector (libmvec)
 * variant can be substituted. */
#include <math.h>
#include <stdint.h>

void relax_host_atan2_f64(const double* y, const double* x, double* out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) out[i] = atan2(y[i], x[i]);
}

void relax_host_acos_f64(const double* values, double* out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) out[i] = acos(values[i]);
}
