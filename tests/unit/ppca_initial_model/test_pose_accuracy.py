"""The expected rotation accuracy of the PPCA model: the divergence of two poses' image laws (doc section 4.1)."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.numerics.shells import shell_of_radius_sq
from relax.ppca_initial_model.pose_accuracy import PVALUE, error_angles, first_passing, kl_terms
from relax.sampling.relion_expected_accuracy import _schedule, _snr_terms, _trial_snr

pytestmark = pytest.mark.unit

N_PIXELS, Q = 40, 3


def _complex(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _real_form(array):
    """Stack real and imaginary parts: ``Re(a^* b)`` becomes the dot product of the stacked vectors."""
    return np.concatenate([array.real, array.imag], axis=0)


def _direct_kl(difference, load0, load1):
    """K(0 || 1) for N(m0, I + L0 L0^T) and N(m1, I + L1 L1^T) on the stacked real vectors, in float64."""
    d, l0, l1 = _real_form(difference), _real_form(load0), _real_form(load1)
    n = d.size
    v0, v1 = np.eye(n) + l0 @ l0.T, np.eye(n) + l1 @ l1.T
    return 0.5 * (
        np.trace(np.linalg.solve(v1, v0)) - n + np.linalg.slogdet(v1)[1] - np.linalg.slogdet(v0)[1]
        + d @ np.linalg.solve(v1, d)
    )  # fmt: skip


def _log_likelihood(image, mean, load):
    """Section 4's pose log-likelihood ``-1/2 {r^* D r - b^T M^-1 b + log det M}`` with unit noise."""
    r, b_matrix = _real_form(image - mean), _real_form(load)
    m = np.eye(load.shape[1]) + b_matrix.T @ b_matrix
    b = b_matrix.T @ r
    return -0.5 * (r @ r - b @ np.linalg.solve(m, b) + np.linalg.slogdet(m)[1])


def _case(seed, scale=0.5):
    rng = np.random.default_rng(seed)
    return (
        0.3 * _complex(rng, N_PIXELS),
        scale * _complex(rng, N_PIXELS, Q) / np.sqrt(N_PIXELS),
        scale * _complex(rng, N_PIXELS, Q) / np.sqrt(N_PIXELS),
    )


def _terms(difference, load0, load1):
    parts = kl_terms(*(jnp.asarray(x, jnp.complex64) for x in (difference, load0, load1)))
    return [float(p) for p in parts]


@pytest.mark.parametrize("scale", [0.05, 0.5, 3.0, 300.0])
def test_the_q_by_q_form_is_the_divergence_between_independent_spans(scale):
    """Loadings of the two poses that share no direction give a finite divergence, equal to the n x n formula."""
    difference, load0, load1 = _case(0, scale)
    mean, log_det, trace = _terms(difference, load0, load1)
    assert np.isfinite([mean, log_det, trace]).all()
    assert_matches(0.5 * (mean + log_det + trace), _direct_kl(difference, load0, load1), rtol=2e-4)


def test_the_mean_and_determinant_terms_are_the_engine_log_likelihood_of_the_noise_free_image():
    """log det M1 - log det M0 + delta^* V1^-1 delta = 2 [l(ybar | pose 0) - l(ybar | pose 1)] for ybar = m0."""
    difference, load0, load1 = _case(1)
    mean, log_det, _ = _terms(difference, load0, load1)
    zero_mean, mean1 = np.zeros(N_PIXELS, complex), -difference  # ybar - m1 = delta
    engine = 2 * (_log_likelihood(zero_mean, zero_mean, load0) - _log_likelihood(zero_mean, mean1, load1))
    assert_matches(mean + log_det, engine, rtol=2e-4)


def test_the_trace_term_is_the_covariance_change_and_vanishes_for_equal_loadings():
    difference, load0, load1 = _case(2)
    _, _, trace = _terms(difference, load0, load1)
    l0, l1 = _real_form(load0), _real_form(load1)
    n = l0.shape[0]
    v0, v1 = np.eye(n) + l0 @ l0.T, np.eye(n) + l1 @ l1.T
    assert_matches(trace, np.trace(np.linalg.solve(v1, v0)) - n, rtol=2e-4)
    equal = _terms(difference, load0, load0)
    assert abs(equal[1]) < 1e-3 and abs(equal[2]) < 1e-3


def test_without_loadings_the_divergence_is_half_the_squared_difference():
    difference = _case(3)[0]
    parts = kl_terms(jnp.asarray(difference, jnp.complex64), jnp.zeros((N_PIXELS, 0)), jnp.zeros((N_PIXELS, 0)))
    assert_matches(float(parts.kl), 0.5 * np.sum(np.abs(difference) ** 2), rtol=1e-5)
    assert float(parts.log_det) == 0.0 and float(parts.trace) == 0.0


def test_the_divergence_does_not_depend_on_the_basis_of_the_loadings():
    difference, load0, load1 = _case(4)
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(Q, Q)))
    assert_matches(_terms(difference, load0 @ q, load1 @ q), _terms(difference, load0, load1), rtol=2e-4)


def test_the_divergence_is_quadratic_for_small_smooth_changes():
    """A pose path A(t) = A0 + tB, m(t) = m0 + tv: K / t^2 is constant at small t."""
    rng = np.random.default_rng(6)
    load0, b, v = (
        0.3 * _complex(rng, N_PIXELS, Q) / 6,
        0.3 * _complex(rng, N_PIXELS, Q) / 6,
        0.1 * _complex(rng, N_PIXELS),
    )
    ratios = [0.5 * sum(_terms(t * v, load0, load0 + t * b)) / t**2 for t in (1e-2, 1e-1)]
    assert_matches(ratios[0], ratios[1], rtol=2e-2)


def test_the_angle_schedule_is_relions():
    angles, cap = error_angles()
    relion, _, n_scored = _schedule(0)
    assert_matches(angles, relion[:n_scored])
    assert cap == relion[n_scored]


def test_the_first_passing_angle_and_the_cap():
    angles = np.array([1.0, 2.0, 3.0])
    values = np.array([[0.0, PVALUE + 1, 9.0], [0.0, 1.0, 2.0]])
    error, index = first_passing(values, angles, cap=35.0)
    assert list(error) == [2.0, 35.0] and list(index) == [1, 2]
    with pytest.raises(ValueError, match="diverged"):
        first_passing(np.array([[0.0, np.nan, 1.0]]), angles, cap=35.0)


def test_a_higher_threshold_gives_a_larger_first_passing_angle():
    angles = np.array([1.0, 2.0, 3.0, 4.0])
    values = np.array([[1.0, 5.0, 20.0, 40.0]])
    assert first_passing(values, angles, cap=35.0)[0][0] == 2.0
    assert first_passing(values, angles, cap=35.0, threshold=15.0)[0][0] == 3.0
    assert first_passing(values, angles, cap=35.0, threshold=100.0)[0][0] == 35.0


def test_with_no_loadings_the_divergence_is_relions_snr_except_on_the_pixels_it_counts_twice():
    """RELION sums |F1-F2|^2 / (2 sigma^2) over the half plane, kx = 0 for ky >= 0 only; its sigma^2 is half the mean
    power nv = E|F|^2 of a noise pixel. With nv = 2 sigma^2 and the Hermitian weights of the real-image metric
    (2 off the self-conjugate columns kx = 0 and kx = N/2, 1 on them) the divergence 1/2 delta^* D delta equals it,
    except on the Nyquist column of the cropped image and its pixel (0, N/2), which RELION counts twice."""
    size = 12
    rng = np.random.default_rng(7)
    f1, f2 = (np.fft.rfft2(rng.normal(size=(size, size))) for _ in range(2))
    sigma2 = 0.7
    valid, denominator = _snr_terms(size, np.full(size, sigma2), 1.0, 1.0)
    iy = np.fft.fftfreq(size, 1 / size).astype(int)[:, None]
    ix = np.arange(size // 2 + 1)[None, :]
    ires = shell_of_radius_sq(iy * iy + ix * ix, rule="half_up")
    window = (ires > 0) & (ires < size // 2 + 1)
    weights = np.where((ix == 0) | (ix == size // 2), 1.0, 2.0)
    delta = np.abs(f1 - f2) ** 2

    def relion(mask):
        flat = lambda a: jnp.asarray(a.reshape(1, -1), jnp.float32)  # noqa: E731
        return float(
            _trial_snr(
                flat(f1.real), flat(f1.imag), flat(f2.real), flat(f2.imag), None,
                jnp.asarray((valid & mask).reshape(-1)), jnp.asarray(denominator.reshape(-1)),
                jnp.zeros(1, jnp.int32), n_trials=1,
            )[0]
        )  # fmt: skip

    def engine(mask):
        return 0.5 * np.sum((weights * delta)[window & mask]) / (2 * sigma2)

    doubled = (ix == size // 2) | ((ix == 0) & (np.abs(iy) == size // 2))
    assert_matches(relion(~doubled), engine(~doubled), rtol=1e-5)
    assert_matches(relion(doubled), 2 * engine(doubled), rtol=1e-5)


def _dataset_and_model(tmp_path, q, seed=3):
    """A 16-pixel noise dataset of 6 particles and a smooth random model of rank ``q`` on its half volume."""
    from test_radius_rule import _write_star

    from relax.commands.ppca_initial_model import load_training

    star = _write_star(tmp_path / "data", 6, box=16)
    data, _, _ = load_training(str(star), "off")
    rng = np.random.default_rng(seed)
    half = int(np.prod((16, 16, 9)))
    theta = (rng.normal(size=(half, q + 1)) + 1j * rng.normal(size=(half, q + 1))).astype(np.complex64)
    theta[:, 1:] *= 0.3
    return data, jnp.asarray(theta), np.full(9, 2.0, np.float32), np.array([[10.0, 40.0, 70.0]] * 6)


def _divergences(data, theta, noise, eulers, q, radius=6):
    from relax.ppca_initial_model.pose_accuracy import trial_divergences

    return trial_divergences(data, theta, noise, ids=np.arange(6), eulers_deg=eulers, radius=radius, active_pcs=q)


def test_the_divergence_scales_inversely_with_the_noise(tmp_path):
    data, theta, noise, eulers = _dataset_and_model(tmp_path, 0)
    quiet, noisy = _divergences(data, theta, noise, eulers, 0), _divergences(data, theta, 4 * noise, eulers, 0)
    assert_matches(noisy.mean_only, quiet.mean_only / 4, rtol=1e-4)
    assert_matches(noisy.kl, quiet.kl / 4, rtol=1e-4)


def test_the_divergence_is_quadratic_at_small_angles_and_the_mean_term_is_all_of_it_without_loadings(tmp_path):
    data, theta, noise, eulers = _dataset_and_model(tmp_path, 0)
    d = _divergences(data, theta, noise, eulers, 0)
    assert_matches(d.kl, d.mean_only, rtol=1e-5)
    assert np.all(d.kl[:, 1] > 0)
    ratio = d.kl[:, 0] / d.kl[:, 1]  # 0.05 against 0.10 degrees
    assert np.all((ratio > 0.2) & (ratio < 0.3))


def test_a_zero_loading_column_leaves_the_mean_quantity_and_loadings_add_the_covariance_terms(tmp_path):
    data, theta, noise, eulers = _dataset_and_model(tmp_path, 1)
    empty = jnp.asarray(np.asarray(theta).copy()).at[:, 1:].set(0)
    zero = _divergences(data, empty, noise, eulers, 1)
    assert_matches(zero.kl, zero.mean_only, rtol=1e-4)
    assert np.abs(zero.parts[..., 1:]).max() < 1e-3
    loaded = _divergences(data, theta, noise, eulers, 1)
    assert_matches(loaded.mean_only, zero.mean_only, rtol=1e-4)  # the mean images do not involve the loadings
    assert np.abs(loaded.parts[..., 1:]).max() > 1e-3
