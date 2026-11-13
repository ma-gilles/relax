"""Code-matched oracle tables: the task's oracle SNR times a measured interpolation-and-mask factor (gate study).

The task's oracle takes the noise variance of a voxel of the full-data reconstruction as 1 / F_all. The model's
reconstruction differs in two ways that the known-truth check (scripts/gate_study/oracle_check.py) measures: linear
interpolation averages the noise of neighbouring voxels (about (2/3)^3 of 1 / F for random positions) and the
real-space mask applied after every update removes the noise outside the particle. Both are properties of the operator,
not of the dataset. From the check's half-map FSCs, the measured full-data SNR of the masked reconstruction is
2 FSC / (1 - FSC); its ratio to the predicted snr is the factor c_m(r).

This script fits c_m on the homogeneous set's checks (H1, n 2000 and 10000; shells 13-30 where 0.1 < FSC < 0.9) as one
constant, takes the edge shell 31 from the median over every check, prints the held-out agreement of the masked
full-data weight on the other checks, and writes <oracle_dir>/<TAG>_cm.npz with ``oracle_cm`` = W(c_m snr) [shells, 2]
(mean, shared loadings) and the fit's provenance. The loading channel uses the same factor (the PPCA metric of a
standardised latent is F times the identity); that is assumed, not checked here.

Run: python scripts/gate_study/oracle_code_matched.py ORACLE_DIR [--fit H1_n3_state0_n2000 H1_n3_state0_n10000]
"""

import argparse
import json
from pathlib import Path

import numpy as np

EDGE = 31


def wiener(snr):
    return snr / (1.0 + snr)


def measured_ratios(check, masked=True):
    fsc = check["measured_fsc_half_masked" if masked else "measured_fsc_half"]
    snr = check["snr_predicted"]
    r = np.arange(len(fsc))
    ok = (fsc > 0.1) & (fsc < 0.9)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 2 * fsc / (1 - fsc) / snr
    return r, ok, ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("--fit", nargs="+", default=["H1_n3_state0_n2000", "H1_n3_state0_n10000"])
    parser.add_argument("--tags", nargs="+", default=["C_n1", "C_n3", "Cshift_n1", "T10_n1", "D10_n1", "D20_n1", "D40_n1", "H1_n3"])
    args = parser.parse_args()
    checks = {p.stem: dict(np.load(p)) for p in sorted((args.oracle_dir / "check").glob("*.npz")) if "smoke" not in p.stem}
    fitted = []
    for name in args.fit:
        r, ok, ratio = measured_ratios(checks[name])
        fitted += list(ratio[ok & (r >= 13) & (r <= EDGE - 1)])
    c_bulk = float(np.median(fitted))
    edge = [measured_ratios(c)[2][EDGE] for c in checks.values() if measured_ratios(c)[1][EDGE]]
    c_edge = float(np.median(edge))
    print(f"c_m: {c_bulk:.3f} on shells 0-{EDGE - 1} (median of {len(fitted)} shell ratios of {args.fit}); "
          f"{c_edge:.3f} at shell {EDGE} (median of {len(edge)} checks)")
    print("held-out masked full-data weight, max |W_pred - W_meas| over shells 13-30 with 0.05 < W_meas < 0.97:")
    report = {}
    for name, check in checks.items():
        snr = check["snr_predicted"]
        f = float(check["support_fraction"])
        fsc = np.clip(check["measured_fsc_half_masked"], 0, 0.999)
        w_meas = wiener(2 * fsc / (1 - fsc))
        r = np.arange(len(snr))
        c = np.where(r == EDGE, c_edge, c_bulk)
        ok = (w_meas > 0.05) & (w_meas < 0.97) & (r >= 13) & (r <= 30)
        if not ok.any():
            continue
        errs = {k: float(np.max(np.abs(w[ok] - w_meas[ok]))) for k, w in
                (("task", wiener(snr)), ("mask_analytic", wiener(snr / f)), ("code_matched", wiener(c * snr)))}
        report[name] = {"shells": [int(r[ok].min()), int(r[ok].max())], "n_shells": int(ok.sum()), **errs,
                        "held_out": name not in args.fit}
        print(f"  {name:26s} {'(fit)' if name in args.fit else '     '} shells {r[ok].min()}-{r[ok].max()}: "
              + "  ".join(f"{k} {v:.3f}" for k, v in errs.items()))
    for tag in args.tags:
        tables = dict(np.load(args.oracle_dir / f"{tag}.npz"))
        n = len(tables["snr_mean"])
        r = np.arange(n)
        c = np.where(r == EDGE, c_edge, c_bulk)
        oracle_cm = np.stack([wiener(c * tables["snr_mean"]), wiener(c * tables["snr_loading"])], axis=-1)
        np.savez(
            args.oracle_dir / f"{tag}_cm.npz",
            oracle_cm=oracle_cm,
            correction=c,
            fit=json.dumps({"c_bulk": c_bulk, "c_edge": c_edge, "fit_checks": args.fit, "validation": report,
                            "source_table": str(args.oracle_dir / f"{tag}.npz")}),
        )
        print(f"{tag}: oracle_cm loading W at shells 16/24/28/31 = "
              + " ".join(f"{oracle_cm[s, 1]:.3f}" for s in (16, 24, 28, 31))
              + f" (task: " + " ".join(f"{tables['oracle'][s, 1]:.3f}" for s in (16, 24, 28, 31)) + ")")


if __name__ == "__main__":
    main()
