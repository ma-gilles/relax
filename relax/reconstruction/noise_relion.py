"""EM-only symbols relocated from ``recovar.reconstruction.noise`` (relax split P1, pure move).

Function bodies are AST-identical to their originals at Q (tag q-reconcile-20260923); only the module
path changed.  See pr179_coordination/relax_split_plan_20260923/PLAN.md.
"""

import jax.numpy as jnp
import numpy as np


def apply_relion_sigma2_floors(sigma2, *, unit: float = 1.0, ctf_premultiplied: bool = False) -> np.ndarray:
    """RELION's two thresholds on an updated noise spectrum (ml_optimiser.cpp:5272-5282).

    The floor 1e-15 for CTF-premultiplied data, then for n > 0 a value below 1e-14 takes the
    previous shell's (already updated) value. ``unit`` scales the thresholds to the spectrum's
    units (``box**4`` for relax's native units, 1 for RELION's).
    """
    out = np.array(sigma2)  # a writable host copy in the caller's dtype
    if ctf_premultiplied:
        out = np.maximum(out, 1e-15 * unit)
    for i in range(1, len(out)):
        if out[i] < 1e-14 * unit:
            out[i] = out[i - 1]
    return out


def summed_noise_pixels_per_shell(image_shape, current_size, nyquist_column_counting="relion") -> np.ndarray:
    """How many pixels of each shell an expectation at ``current_size`` actually sums into the noise spectrum.

    Shells up to ``current_size / 2`` are summed on RELION's cropped image, which has no row
    ``-current_size / 2``; higher shells come from the full image (``power_img``). RELION's
    ``Npix_per_shell`` counts every shell on the full image, so below the box its count of shell
    ``current_size / 2`` includes the pixels of that missing row (``jp >= 1, ip = -cs/2``), which no
    sum contains. At the box (or ``current_size`` None) the two counts agree.
    ``nyquist_column_counting`` is the shell table's (``make_relion_noise_shell_indices_half``).
    """
    from relax.helpers.fourier_window import make_fourier_window_indices_np
    from relax.helpers.half_spectrum import (
        bin_shell_values_np,
        make_relion_noise_shell_indices_half,
        mask_relion_noise_shell_indices_to_current_window,
    )

    shell_indices = (
        make_relion_noise_shell_indices_half(image_shape)
        if nyquist_column_counting == "relion"
        else make_relion_noise_shell_indices_half(image_shape, nyquist_column_counting)
    )
    if current_size is not None and int(current_size) < int(image_shape[0]):
        window_indices, _ = make_fourier_window_indices_np(image_shape, int(current_size))
        shell_indices = mask_relion_noise_shell_indices_to_current_window(
            shell_indices, image_shape, int(current_size), window_indices
        )
    n_shells = int(image_shape[0]) // 2 + 1
    return bin_shell_values_np(np.ones(np.shape(shell_indices)), shell_indices, n_shells)


def normalize_wsum_to_sigma2_noise(
    wsum_sigma2_noise,
    wsum_img_power,
    sumw,
    image_shape,
    *,
    ctf_premultiplied=False,
    apply_floors=True,
    summed_current_size=None,
    nyquist_column_counting="relion",
):
    """Convert posterior-weighted noise accumulators to per-shell noise variance.

    Implements RELION's M-step noise update from ``maximizationOtherParameters``
    (ml_optimiser.cpp:5246-5285)::

        sigma2_noise[s] = wsum_total[s] / (2 * sumw * Npix_per_shell[s])

    where ``wsum_total = wsum_sigma2_noise + wsum_img_power`` is the total
    posterior-weighted squared residual (A2 - 2*XA + P_img), ``sumw`` is the
    total number of images processed, and ``Npix_per_shell`` counts
    half-spectrum pixels per shell.

    The factor of 2 accounts for the real and imaginary components of each
    complex Fourier coefficient (RELION convention: sigma2 is per-component
    variance).

    Parameters
    ----------
    wsum_sigma2_noise : array, shape (n_shells,)
        Accumulated A2 - 2*XA per shell from ``_compute_noise_block``.
    wsum_img_power : array, shape (n_shells,)
        Accumulated |img|^2 per shell (P_img term).
    sumw : float
        Total posterior weight (= number of images when posteriors sum to 1).
    image_shape : tuple of int
        2-D image dimensions, e.g. ``(128, 128)``.
    ctf_premultiplied : bool
        RELION floors sigma2 at 1e-15 (its units) only for CTF-premultiplied data.
    summed_current_size : int or None
        None (default) divides by RELION's ``Npix_per_shell``, counted on the full image. An
        expectation below the box sums shell ``current_size / 2`` on a crop that lacks one row, so
        that shell's sigma2 comes out low (3.6-7.1%); passing the expectation's image current size
        divides every shell by the pixels it summed (:func:`summed_noise_pixels_per_shell`).
    nyquist_column_counting : {"relion", "once"}
        ``"relion"`` counts both members of each Hermitian pair of the full-size Nyquist
        column, as the sums do; ``"once"`` counts each pair once, for sums accumulated with
        that rule.

    Returns
    -------
    sigma2_noise : jnp.ndarray, shape (n_shells,)
        Per-shell noise variance in **recovar's native FFT units**, ready to
        be fed back into the engine. Same scale as ``|process_images|^2``.
    """
    from relax.helpers.half_spectrum import bin_shell_values_jax, make_relion_noise_shell_indices_half

    # Preserve whatever real dtype the caller's accumulators already carry
    # (float64 end to end when use_float64_scoring/use_float64_projections
    # are set) instead of forcing float32 here. RELION's own
    # wsum_model.sigma2_noise stays RFLOAT (double under double-precision
    # builds) through this exact division; the unconditional float32 casts
    # this replaced discarded that precision right before the final divide,
    # on top of whatever precision compute_noise_block's callers already
    # accumulated it at.
    wsum_sigma2_noise = jnp.asarray(wsum_sigma2_noise)
    wsum_img_power = jnp.asarray(wsum_img_power)
    n_shells = image_shape[0] // 2 + 1

    shell_indices = (
        make_relion_noise_shell_indices_half(image_shape)
        if nyquist_column_counting == "relion"
        else make_relion_noise_shell_indices_half(image_shape, nyquist_column_counting)
    )
    Npix_per_shell = bin_shell_values_jax(
        jnp.ones_like(shell_indices, dtype=jnp.float32),
        shell_indices,
        n_shells,
    )

    if summed_current_size is not None:
        Npix_per_shell = jnp.asarray(
            summed_noise_pixels_per_shell(image_shape, summed_current_size, nyquist_column_counting), dtype=jnp.float32
        )

    total_wsum = wsum_sigma2_noise + wsum_img_power
    sigma2 = total_wsum / (2.0 * sumw * jnp.maximum(Npix_per_shell, 1.0))

    # NOTE: The output is in **recovar's native FFT units**, not "RELION
    # units". Both ``wsum_sigma2_noise`` and ``wsum_img_power`` are accumulated
    # from ``|process_images(batch)|^2``-scaled quantities (the engine never
    # converts), so the resulting sigma2 already lives in the same units as
    # the engine's image power. A previous version divided by ``(H*W)^2`` to
    # "convert to RELION units" — that was wrong: the engine then re-uses this
    # sigma2 in ``processed * CTF / sigma2``, and the divide blows up chi² by
    # ``(H*W)^2``, collapsing every posterior to a single rotation
    # (``Pmax → 1.0``). See ``estimate_initial_noise_spectrum_from_unaligned_images``
    # docstring and ``tmp/diagnose_pmax_gap.py``.

    # RELION's two absolute thresholds (ml_optimiser.cpp maximizationOtherParameters) act on
    # its own FFT units; relax's native sigma2 carries box**4 on top of them (the image FT
    # carries box**2), so the thresholds are scaled by box**4 rather than the values rescaled.
    # - Floor at 1e-15, only for CTF-premultiplied data ("Watch out for all-zero sigma2").
    # - For n > 0, a value below 1e-14 takes the previous shell's (already updated) value
    #   ("With unequal box sizes ... set sigma2_noise to the value in the previous pixel").
    # ``apply_floors=False`` leaves them to a caller that blends first (VDAM's running average).
    if not apply_floors:
        return sigma2
    return jnp.asarray(
        apply_relion_sigma2_floors(sigma2, unit=float(image_shape[0]) ** 4, ctf_premultiplied=ctf_premultiplied)
    )
