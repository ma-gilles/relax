"""Which class map makes the suite v3.2 frame fit raise "Unidentifiable transformed registration norm" (gate study).

Read-only diagnostic of one suite row (CPU). Rebuilds the row's per-class calibration maps exactly as msuite_score.score
does for an unrelated-protein set (decoded route: model mean + class-mean calibration latent times the loadings; BP route:
WORK/recon/NAME/cal.npy), prints each map's statistics (std, std after the registration low-pass, correlation with its
true state), then runs the step that raised, relax's align_volume_rigid_to_reference(..., allow_mirror=False) inside
msuite_data.fit_frame, for every class, route and hand, with centered_correlation wrapped to record the trial whose
correlation is not finite (which of the two vectors has zero norm, and the trial translation).

Run with the suite's CPU environment: PYTHONPATH=$SOURCE:<metrics_suite_v3_2> $PY diag_registration_failure.py WORK NAME
CHECKPOINT DATASET [--processes N]
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


def run_one(args):
    k, route, hand, moving, target = args
    import msuite_core as mc
    import msuite_data as md
    from relax.diagnostics import gt_registration as gr
    from relax.diagnostics.gt_registration import RigidFitControls, align_volume_rigid_to_reference

    record = {}
    original = gr.centered_correlation

    def wrapped(lhs, rhs):
        value = original(lhs, rhs)
        if not np.isfinite(value):
            a, b = np.asarray(lhs, np.float64), np.asarray(rhs, np.float64)
            record.update(lhs_norm=float(np.linalg.norm(a - a.mean())), rhs_norm=float(np.linalg.norm(b - b.mean())),
                          lhs_max_abs=float(np.abs(a).max()))
        return value

    gr.centered_correlation = wrapped
    mc.set_workers(1)
    seeds = md.seed_grid()
    if hand:
        moving = mc.apply_frame(moving, np.eye(3), True, np.zeros(3), order=3)
    calls = []
    original_score = gr._continuous_score

    def score_factory(*a, **kw):
        score = original_score(*a, **kw)

        def traced(transform, translation):
            calls.append(np.asarray(translation, np.float64).copy())
            return score(transform, translation)

        return traced

    gr._continuous_score = score_factory
    try:
        fit = align_volume_rigid_to_reference(moving, target, seeds, controls=RigidFitControls(allow_mirror=False))
        return dict(k=k, route=route, hand=hand, ok=True, corr=float(fit.receipt.coarse_score) if hasattr(fit, "receipt") else None,
                    evaluations=len(calls))
    except ValueError as error:
        last = calls[-1].tolist() if calls else None
        return dict(k=k, route=route, hand=hand, ok=False, error=str(error), last_translation_voxels=last,
                    evaluations=len(calls), max_abs_translation_seen=float(np.max(np.abs(calls))) if calls else None, **record)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work", type=Path)
    parser.add_argument("name")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    import msuite_core as mc
    import msuite_data as md
    from msuite_score import calibration_maps, load_inference
    from relax.diagnostics.gt_metrics import centered_correlation, lowpass_volume_by_shell
    from relax.diagnostics.gt_registration import RigidFitControls

    truth = md.Truth(args.dataset / "evaluation")
    split = md.Split(args.work / "splits" / f"{truth.dataset}.npz")
    inf = load_inference(args.work, args.name, len(truth.labels))
    basis, meta, nb, radius = md.load_model(args.checkpoint)
    dec, bp = calibration_maps(truth.labels, split, inf["z"], basis, args.work / "recon" / args.name, truth.K)
    shell = RigidFitControls().score_max_shell
    print(f"{args.name}: K {truth.K}, unrelated {truth.discrete_unrelated}, registration low-pass shell {shell}")
    stats = []
    for k in range(truth.K):
        for route, maps in (("dec", dec), ("bp", bp)):
            v = maps[k]
            low = lowpass_volume_by_shell(v - v.mean(), shell)
            stats.append(dict(k=k, route=route, std=float(v.std()), lowpass_std=float(low.std()),
                              corr_with_truth=float(centered_correlation(v, truth.states[k])), finite=bool(np.isfinite(v).all())))
    for route in ("dec", "bp"):
        s = [x for x in stats if x["route"] == route]
        std = np.array([x["lowpass_std"] for x in s])
        print(f"  {route}: low-passed std min {std.min():.3g} (class {int(std.argmin())}) median {np.median(std):.3g}; "
              f"nonfinite maps {sum(not x['finite'] for x in s)}; corr with truth min "
              f"{min(x['corr_with_truth'] for x in s):.3f}")
    jobs = [(k, route, hand, (dec if route == "dec" else bp)[k], truth.states[k]) for k in range(truth.K)
            for route in ("dec", "bp") for hand in (False, True)]
    with mp.get_context("fork").Pool(args.processes) as pool:
        results = pool.map(run_one, jobs)
    failed = [r for r in results if not r["ok"]]
    print(f"  relax-fit calls: {len(results)}, raised: {len(failed)}")
    for r in failed:
        s = next(x for x in stats if x["k"] == r["k"] and x["route"] == r["route"])
        print(f"   class {r['k']} route {r['route']} hand {r['hand']}: {r['error']}; last trial translation (voxels) "
              f"{np.round(r['last_translation_voxels'], 1).tolist() if r['last_translation_voxels'] else None}; "
              f"norms moving {r.get('lhs_norm')} target {r.get('rhs_norm')}; map std {s['std']:.3g}, low-passed {s['lowpass_std']:.3g}, "
              f"corr with truth {s['corr_with_truth']:.3f}; evaluations {r['evaluations']}")
    if args.output:
        args.output.write_text(json.dumps(dict(stats=stats, results=results), indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
