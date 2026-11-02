"""Round 12 copy: dataset root from SUITE_DATASETS; box-128 runs get the pose fit only.
Direct pose fit for any suite dataset (round 6 copy of round 2's direct_fit.py, 2026-10-28): the dataset comes from
the run folder name (<dataset>__<arm>_s<seed>); for datasets with a distinctive state (D: structure 42) the rotation
fit and the pose numbers use that state's particles only (the other 99 structures share no frame), and the per-state
FSC is computed for that state only.

Direct pose fit (lead, 2026-10-27), the pose metric from now on; also re-scores the per-state maps in that frame.

Pose: one global rotation G plus handedness fitted from inferred vs true poses (robust trimmed Procrustes; fit()
copied from the lead's /scratch/gpfs/GILLES/mg6942/tmp/ppca_r2_posecheck_20261027/direct_fit.py; the better hand by
median), pose error = angle between inferred and true pose after G. No map registration is involved.
State maps (item 2): each state's map mu + W zbar_k (zbar_k = mean latent of that state's stratified particles, the
scorer's own per-state map) is rotated by the same G and hand (as the lead's evolve_aligned.py), shifted by the
integer cross-correlation peak to the true state map, and scored with the scorer's FSC (raw mean, shells 1-15);
reported next to the scorer's own per-state fit FSC (its registration can land in a wrong local optimum).
The map-frame numbers stay in the output as "map-frame (deprecated)".

python direct_fit.py RUN_DIR...  -> <scoring dir>/direct_fit.json for every suite_score_checkpoint_* without one
(suite cpu_env.sh, PYTHONPATH=$SOURCE)
"""
import glob
import json
import os
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, '/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_cryobench_scaling_20261002/run_tools')
from recovar.core import fourier_transform_utils as ftu  # noqa: E402
from recovar.utils.helpers import load_mrc  # noqa: E402
from relax import sampling  # noqa: E402
from relax.diagnostics.gt_registration import apply_rigid_volume_transform  # noqa: E402
from scorer_general import curve15, pose_errors  # noqa: E402

SUITE = os.environ.get('SUITE_DATASETS', '/scratch/gpfs/GILLES/mg6942/ppca_test_suite_20261006/datasets')  # round 12: env-selectable root
_cache = {}


def dataset(ds):
    if ds not in _cache:
        ev = f'{SUITE}/{ds}/evaluation'
        em = json.load(open(ev + '/manifest.json'))
        k = em.get('distinctive_state')
        states = [k] if k is not None else list(range(len(em['states'])))
        gt = {i: np.asarray(load_mrc(f"{ev}/{em['states'][i]['gt_map']}"), np.float32) for i in states}
        _cache[ds] = (np.load(ev + '/truth.npz'), gt, np.load(ev + '/eval_ids_stratified.npy'), k)
    return _cache[ds]


J, F0 = np.diag([1., 1., -1.]), np.diag([-1., 1., 1.])
N = 64
hs = ftu.volume_shape_to_half_volume_shape((N, N, N))
coords = np.arange(N, dtype=np.float32) - N // 2
rr = np.sqrt(sum(x ** 2 for x in np.meshgrid(coords, coords, coords, indexing='ij')))
MASK = ((1 + np.cos(np.pi * np.clip((rr - 24) / 5, 0, 1))) / 2).astype(np.float32)


def proj(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def ang(A, B):
    return np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', A, B) - 1) / 2, -1, 1)))


def fit(T, E):  # E ~ T G; best of candidate G's (one per 10th particle), then trimmed refit (lead's code)
    cands = np.einsum('nji,njk->nik', T[::10], E[::10])
    G = min(cands, key=lambda G: np.median(ang(E, T @ G)))
    for _ in range(10):
        e = ang(E, T @ G)
        keep = e <= np.quantile(e, .5)
        G = proj(np.einsum('nji,njk->ik', T[keep], E[keep]))
    return G, ang(E, T @ G)


def shift_fit(v, ref):
    c = np.fft.ifftn(np.fft.fftn(ref) * np.conj(np.fft.fftn(v))).real
    s = np.array(np.unravel_index(np.argmax(c), c.shape))
    return np.where(s > N // 2, s - N, s)


grids = {}
for run in sys.argv[1:]:
    for d in sorted(glob.glob(f'{run}/suite_score_checkpoint_*')):
        if not d[-4:].isdigit():  # e.g. suite_score_checkpoint_3000_fullband (box-128 full-band score): no direct fit
            continue
        if not os.path.exists(d + '/summary.json') or os.path.exists(d + '/direct_fit.json'):
            continue
        c = int(d[-4:])
        inf = np.load(d + '/inference.npz')
        r = json.load(open(d + '/report.json'))
        s = json.load(open(d + '/summary.json'))
        o = int(inf['healpix_order'])
        if o not in grids:
            grids[o] = sampling.get_relion_hidden_rotation_grid(o, matrices=True).astype(np.float64)
        ds = Path(run).name.rsplit('__', 1)[0]
        t, GT, STRAT, focus = dataset(ds)
        ids = inf['particle_ids']
        if json.load(open(f'{SUITE}/{ds}/evaluation/manifest.json')).get('composition') == 'T10':
            # T10 (lead, 2026-10-30): ten unrelated proteins share no frame, so each state gets its own rotation and
            # hand; per-state median / %<10 and the particle-weighted overall (pooled errors). Pose only.
            lab = t['labels'][ids]; per = []; pooled = []
            for k in np.unique(lab):
                m_ = lab == k
                Ek = grids[o][inf['rotation_idx']][m_]; Tk = t['rotations'][ids][m_].astype(np.float64)
                Gk, ek, mk = min(((*fit(Tk, J @ Ek @ F0 if m else Ek), m) for m in (False, True)), key=lambda x: np.median(x[1]))
                per.append({'state': int(k), 'n': int(m_.sum()), 'pose_median_deg': float(np.median(ek)), 'frac_below_10': float(np.mean(ek < 10)),
                            'mirror': bool(mk), 'rotation': Gk.tolist()})
                pooled.append(ek)
            pooled = np.concatenate(pooled)
            out = {'checkpoint': c, 'fit': 'per_state', 'pose_median_deg': float(np.median(pooled)), 'frac_below_10': float(np.mean(pooled < 10)),
                   'per_state': per, 'states': None, 'state_fsc_mean_direct': None,
                   'map_frame_deprecated': {'pose_median_deg': s['pose_median_deg'], 'frac_below_10': s['pose_frac_below_10deg'], 'mean_fsc': s['mean_fsc_to_true_mean']}}
            json.dump(out, open(d + '/direct_fit.json', 'w'), indent=1)
            continue
        sel = np.ones(len(ids), bool) if focus is None else t['labels'][ids] == focus
        E = grids[o][inf['rotation_idx']][sel]
        T = t['rotations'][ids][sel].astype(np.float64)
        best = min(((*fit(T, J @ E @ F0 if m else E), m) for m in (False, True)), key=lambda x: np.median(x[1]))
        G, e, mirror = best
        out = {'checkpoint': c, 'pose_median_deg': float(np.median(e)), 'frac_below_10': float(np.mean(e < 10)),
               'mirror': bool(mirror), 'rotation': G.tolist(), 'fit_state': focus, 'n_fit': int(sel.sum()),
               'map_frame_deprecated': {'pose_median_deg': s['pose_median_deg'], 'frac_below_10': s['pose_frac_below_10deg'],
                                        'mean_fsc': s['mean_fsc_to_true_mean']}}
        # Per-state maps in the direct-fit frame.
        th = np.load(f'{run}/checkpoint_{c:04d}.npz')['theta']
        if th.shape[0] != N * N * (N // 2 + 1):  # round 12, box 128: pose only (per-state maps here are box 64)
            out.update(states=None, state_fsc_mean_direct=None, pose_only_box=int(round((th.shape[0] * 2) ** (1 / 3))))
            json.dump(out, open(d + '/direct_fit.json', 'w'), indent=1)
            continue
        pos = {int(i): n for n, i in enumerate(ids)}
        rows_s = np.asarray([pos[int(i)] for i in STRAT])
        lab = t['labels'][STRAT]
        z = inf['z'][rows_s].astype(np.float64)
        states = []
        for k in GT:
            zb = z[lab == k].mean(axis=0).astype(np.float32)
            v = np.asarray(ftu.get_idft3_real((th[:, 0] + th[:, 1:] @ zb).reshape((1,) + hs), (N, N, N)))[0].astype(np.float32)
            v = apply_rigid_volume_transform(v, G, np.zeros(3), mirror_x=bool(mirror))
            v = np.roll(v, shift_fit(v, GT[k]), (0, 1, 2))
            states.append({'state': k, 'fsc_direct_frame': curve15(v, GT[k], MASK)['raw_mean_1_15'],
                           'fsc_map_frame_deprecated': next(x for x in r['states'] if x['gt_state'] == k).get('individual_fit_fsc', {}).get('raw_mean_1_15')})
        out['states'] = states
        out['state_fsc_mean_direct'] = float(np.mean([x['fsc_direct_frame'] for x in states]))
        out['state_fsc_mean_map_frame_deprecated'] = float(np.mean([x['fsc_map_frame_deprecated'] for x in states if x['fsc_map_frame_deprecated'] is not None] or [np.nan]))
        json.dump(out, open(d + '/direct_fit.json', 'w'), indent=1)
        print(d.split('runs_')[-1].split('/', 1)[-1], 'pose %.1f (%.2f<10, mirror %s) | map-frame %.1f | state FSC direct %.3f vs map-frame %.3f' % (
            out['pose_median_deg'], out['frac_below_10'], mirror, s['pose_median_deg'], out['state_fsc_mean_direct'],
            out['state_fsc_mean_map_frame_deprecated']), flush=True)
