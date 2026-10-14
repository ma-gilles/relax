#!/usr/bin/env python3
"""Score final maps of a symmetric refinement against a reference map after a rigid fit about the symmetry axis.

A Cn refinement fixes its map's symmetry axis on the box z axis through the RELION origin (voxel n/2), but leaves the
rotation about that axis and the position along it free. Two runs of the same command can therefore differ by such a
motion without differing in structure (the RELION 5 tutorial subtomogram Refine3D: relax's map sat 41 degrees from
RELION's shipped one). This scorer removes exactly that freedom and nothing else:

- fit: rotation about z in [0, 360/n) and a shift along z, optionally with the 180-degree flip about x that maps the
  axis onto itself, maximising the real-space correlation inside a soft sphere (``fit_about_axis``);
- transform: one cubic-spline resampling of the 2x Fourier-padded map about voxel n/2, cropped back in Fourier space
  (``transform_about_axis``), so interpolation loss stays small up to the original Nyquist;
- metric: ``normalized_fsc_auc(shell_fsc(a, b))`` of ``scripts/fsc_metrics.py`` (full spectrum), with every map,
  reference included, multiplied by the same soft sphere inscribed in the box (``--window-radius-px``, default
  n/2 - 3 with a 3-pixel edge). The window is what a rotation of a cube needs (the corners leave the box); it is not
  a structure mask. A second sphere (``--sphere-radius-A``) gives a supplementary value inside the particle sphere.

Every arm is fitted onto the reference (``--reference-onto-arm``: the reference onto every arm, the second scoring
behind a row's scorer resolution); every ``--pair A:B`` fits A onto B. Reads maps only; CPU.

Usage::

    python scripts/score_axis_registered_maps.py --reference REF.mrc --symmetry-order 6 --pixel-A 8.1 \\
        --arm relax=RUN/final_merged.mrc --arm relion=RUN/run_class001.mrc --pair relax:relion --out SCORE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fsc_metrics import first_shell_below, normalized_fsc_auc, shell_fsc  # noqa: E402

UPSAMPLE = 2


def soft_sphere(n: int, radius_px: float, edge_px: float) -> np.ndarray:
    """1 inside ``radius_px`` of voxel n/2, raised-cosine fall to 0 over ``edge_px``."""
    g = np.indices((n,) * 3, dtype=np.float64) - n // 2
    r = np.sqrt((g**2).sum(0))
    t = np.clip((r - radius_px) / edge_px, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def flip_about_x(vol: np.ndarray) -> np.ndarray:
    """180-degree rotation about the x axis through voxel n/2 (z -> -z, y -> -y); array axes are (z, y, x)."""
    return np.roll(vol[::-1, ::-1, :], (1, 1), axis=(0, 1))


def _fourier_resize(vol: np.ndarray, m: int) -> np.ndarray:
    """Band-limited resampling of a cubic volume from n to m voxels per side about voxel n/2 -> m/2."""
    n = vol.shape[0]
    f = np.fft.fftshift(np.fft.fftn(vol))
    out = np.zeros((m,) * 3, dtype=np.complex128)
    k = min(n, m)
    a, b = n // 2 - k // 2, m // 2 - k // 2
    out[b : b + k, b : b + k, b : b + k] = f[a : a + k, a : a + k, a : a + k]
    return np.fft.ifftn(np.fft.ifftshift(out)).real * (m / n) ** 3


def _affine(vol: np.ndarray, angle_deg: float, dz_px: float, order: int) -> np.ndarray:
    """Rotate ``vol`` by ``angle_deg`` about z through voxel n/2 and move it by ``dz_px`` along z."""
    n = vol.shape[0]
    c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])  # output (z, y, x) -> input coordinates
    centre = np.full(3, n // 2, dtype=np.float64)
    offset = centre - matrix @ centre - np.array([dz_px, 0.0, 0.0])
    return ndimage.affine_transform(vol, matrix, offset=offset, order=order, mode="constant", cval=0.0)


def transform_about_axis(vol: np.ndarray, angle_deg: float, dz_px: float, flip: bool = False) -> np.ndarray:
    """The map after the optional flip, a rotation of ``angle_deg`` about z and a shift of ``dz_px`` along z."""
    vol = np.asarray(vol, dtype=np.float64)
    n = vol.shape[0]
    if flip:
        vol = flip_about_x(vol)
    up = _fourier_resize(vol, UPSAMPLE * n)
    return _fourier_resize(_affine(up, angle_deg, UPSAMPLE * dz_px, order=3), n)


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.corrcoef((a * mask).ravel(), (b * mask).ravel())[0, 1])


def fit_about_axis(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    symmetry_order: int,
    mask: np.ndarray,
    allow_flip: bool = True,
    max_shift_px: int = 3,
) -> dict:
    """Best (angle, z shift, flip) of ``moving`` onto ``reference`` by masked real-space correlation.

    Coarse: every whole degree of [0, 360/symmetry_order) and whole-pixel z shift, linear interpolation. Fine:
    Nelder-Mead on (angle, shift) with the scoring transform, from the best coarse point of each flip state.
    """
    moving = np.asarray(moving, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    period = 360.0 / symmetry_order
    best = None
    for flip in (False, True) if allow_flip else (False,):
        base = flip_about_x(moving) if flip else moving
        coarse = (-2.0, 0.0, 0)
        for angle in np.arange(0.0, period, 1.0):
            rotated = _affine(base, angle, 0.0, order=1)
            for dz in range(-max_shift_px, max_shift_px + 1):
                c = _corr(np.roll(rotated, dz, axis=0), reference, mask)
                if c > coarse[0]:
                    coarse = (c, float(angle), dz)
        res = optimize.minimize(
            lambda p: -_corr(transform_about_axis(moving, p[0], p[1], flip), reference, mask),
            x0=[coarse[1], float(coarse[2])],
            method="Nelder-Mead",
            options={
                "xatol": 0.01,
                "fatol": 1e-7,
                "initial_simplex": [[coarse[1], coarse[2]], [coarse[1] + 0.5, coarse[2]], [coarse[1], coarse[2] + 0.3]],
            },
        )
        cand = {
            "corr": float(-res.fun),
            "angle_deg": float(res.x[0] % period),
            "z_shift_px": float(res.x[1]),
            "flip": flip,
            "coarse_corr": coarse[0],
        }
        if best is None or cand["corr"] > best["corr"]:
            best = cand
    return best


def score_pair(
    moving: np.ndarray, reference: np.ndarray, *, symmetry_order: int, window: np.ndarray, sphere: np.ndarray | None
) -> dict:
    """Fit ``moving`` onto ``reference`` (inside ``sphere`` when given, else ``window``) and score the fitted map."""
    fit = fit_about_axis(moving, reference, symmetry_order=symmetry_order, mask=window if sphere is None else sphere)
    fitted = transform_about_axis(moving, fit["angle_deg"], fit["z_shift_px"], fit["flip"])
    curve = shell_fsc(fitted * window, reference * window)
    out = {
        "fit": fit,
        "corr_as_written": _corr(moving, reference, window if sphere is None else sphere),
        "fsc_auc_window": float(normalized_fsc_auc(curve)),
        "fsc_window": [float(v) for v in curve],
        "first_shell_below_0p5": first_shell_below(curve, 0.5),
        "first_shell_below_0p143": first_shell_below(curve, 0.143),
    }
    if sphere is not None:
        out["fsc_auc_sphere"] = float(normalized_fsc_auc(shell_fsc(fitted * sphere, reference * sphere)))
    return out


def _read(path: str) -> np.ndarray:
    import mrcfile

    return mrcfile.read(path).astype(np.float64)


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=MAP", required=True)
    ap.add_argument("--pair", action="append", default=[], metavar="A:B", help="also fit arm A onto arm B")
    ap.add_argument("--symmetry-order", type=int, required=True)
    ap.add_argument("--pixel-A", type=float, required=True)
    ap.add_argument("--window-radius-px", type=float, default=None, help="default n/2 - 3")
    ap.add_argument("--window-edge-px", type=float, default=3.0)
    ap.add_argument("--sphere-radius-A", type=float, default=None, help="supplementary sphere; also used for the fit")
    ap.add_argument("--sphere-edge-px", type=float, default=5.0)
    ap.add_argument(
        "--reference-onto-arm",
        action="store_true",
        help="fit and resample the reference onto each arm instead (the same comparison scored the other way round;"
        " the change of an arm-minus-arm gap between the two directions measures the scorer's resolution)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    reference = _read(args.reference)
    n = reference.shape[0]
    radius = args.window_radius_px if args.window_radius_px is not None else n / 2 - 3.0
    if radius + args.window_edge_px > n / 2:
        raise SystemExit("the window must end inside the box")
    window = soft_sphere(n, radius, args.window_edge_px)
    sphere = (
        None
        if args.sphere_radius_A is None
        else soft_sphere(n, args.sphere_radius_A / args.pixel_A, args.sphere_edge_px)
    )
    arms = dict(a.split("=", 1) for a in args.arm)
    maps = {label: _read(path) for label, path in arms.items()}
    for label, vol in maps.items():
        if vol.shape != reference.shape:
            raise SystemExit(f"{label}: shape {vol.shape} is not the reference's {reference.shape}")
    kw = dict(symmetry_order=args.symmetry_order, window=window, sphere=sphere)
    out = {
        "scorer": "scripts/score_axis_registered_maps.py",
        "metric": "normalized_fsc_auc(shell_fsc) of scripts/fsc_metrics.py after the rigid fit about the symmetry axis; every map times the window",
        "fit_direction": "reference onto arm" if args.reference_onto_arm else "arm onto reference",
        "symmetry_order": args.symmetry_order,
        "pixel_A": args.pixel_A,
        "box": n,
        "window": {"radius_px": radius, "edge_px": args.window_edge_px},
        "sphere": None if sphere is None else {"radius_A": args.sphere_radius_A, "edge_px": args.sphere_edge_px},
        "reference": {"path": args.reference, "sha256": _sha256(args.reference)},
        "arms": {},
        "pairs": {},
    }
    for label, vol in maps.items():
        scored = score_pair(reference, vol, **kw) if args.reference_onto_arm else score_pair(vol, reference, **kw)
        out["arms"][label] = {"path": arms[label], "sha256": _sha256(arms[label]), "vs_reference": scored}
        s = out["arms"][label]["vs_reference"]
        print(
            f"{label}: vs reference FSC-AUC {s['fsc_auc_window']:.6f} (fit {s['fit']['angle_deg']:.2f} deg, z {s['fit']['z_shift_px']:+.2f} px, flip {s['fit']['flip']}, corr {s['fit']['corr']:.4f})",
            flush=True,
        )
    for pair in args.pair:
        a, b = pair.split(":")
        out["pairs"][pair] = score_pair(maps[a], maps[b], **kw)
        s = out["pairs"][pair]
        print(
            f"{a} onto {b}: FSC-AUC {s['fsc_auc_window']:.6f} (fit {s['fit']['angle_deg']:.2f} deg, z {s['fit']['z_shift_px']:+.2f} px, flip {s['fit']['flip']})",
            flush=True,
        )
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
