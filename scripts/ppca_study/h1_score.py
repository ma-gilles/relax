"""H1 (homogeneous, true rank 0) scoring fallback (lead, 2026-10-30): suite_score.py stops after writing inference.npz
(no true PCs to stack), so for every suite_score_checkpoint_* with an inference.npz this writes
  direct_fit.json  - direct pose fit (round 12 direct_fit.py's fit: global rotation + hand), pose only;
  h1_metrics.json  - mean-map FSC to the true map: the model mean rotated by that fit (hand applied), cropped to box 64
                     as the scorer does, shifted by the integer cross-correlation peak, FSC raw/masked mean of shells
                     1-15 (the scorer's curve15, the same number as its mean_fsc_to_true_mean); plus the box-128 raw FSC
                     curve and its first shell below 0.5; and PC power / mean power of the checkpoint model;
and run-level h1_pc_power.json: loading_power / mean_power per update from iterations.jsonl (truth: 0).
python h1_score.py RUN_DIR...   (suite cpu_env.sh, PYTHONPATH=$SOURCE, SUITE_DATASETS=<recipe root>/datasets)
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
runs = sys.argv[1:]; sys.argv = ['x']
src = open('/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round17_simple_sgd_20261102/tools/direct_fit.py').read()
exec(src.split('grids = {}')[0])
sys.path.insert(0, '/scratch/gpfs/GILLES/mg6942/ppca_test_suite_20261006/tools')
from suite_score import crop_to_box64
from scripts.fsc_metrics import shell_fsc
grids = {}
for run in runs:
    ds = Path(run).name.rsplit('__', 1)[0]
    t, GT, STRAT, focus = dataset(ds)
    true_map = next(iter(GT.values())) if isinstance(GT, dict) else GT[0]
    true64 = crop_to_box64(true_map)
    rows = [json.loads(l) for l in open(f'{run}/iterations.jsonl')]
    json.dump({'update': [r['iteration'] for r in rows], 'pc_over_mean_power': [r['loading_power'] / r['mean_power'] for r in rows]},
              open(f'{run}/h1_pc_power.json', 'w'))
    for d in sorted(glob.glob(f'{run}/suite_score_checkpoint_*')):
        if not os.path.exists(d + '/inference.npz') or os.path.exists(d + '/h1_metrics.json'):
            continue
        c = int(d.split('_')[-1][:4]) if d[-4:].isdigit() else None
        if c is None: continue
        inf = np.load(d + '/inference.npz'); o = int(inf['healpix_order'])
        if o not in grids:
            grids[o] = sampling.get_relion_hidden_rotation_grid(o, matrices=True).astype(np.float64)
        ids = inf['particle_ids']
        E = grids[o][inf['rotation_idx']]; T = t['rotations'][ids].astype(np.float64)
        G, e, mirror = min(((*fit(T, J @ E @ F0 if m else E), m) for m in (False, True)), key=lambda x: np.median(x[1]))
        df = {'checkpoint': c, 'pose_median_deg': float(np.median(e)), 'frac_below_10': float(np.mean(e < 10)), 'mirror': bool(mirror),
              'rotation': G.tolist(), 'n_fit': int(len(ids)), 'pose_only': True, 'h1_fallback': True}
        json.dump(df, open(d + '/direct_fit.json', 'w'), indent=1)
        th = np.load(f'{run}/checkpoint_{c:04d}.npz')['theta']
        n = next(b for b in (64, 128, 256) if b * b * (b // 2 + 1) == th.shape[0])
        hs_n = ftu.volume_shape_to_half_volume_shape((n, n, n))
        mean = np.asarray(ftu.get_idft3_real(th[:, 0].reshape((1,) + hs_n), (n, n, n)))[0].astype(np.float32)
        mean = apply_rigid_volume_transform(mean, G, np.zeros(3), mirror_x=bool(mirror))
        m64 = crop_to_box64(mean)
        m64 = np.roll(m64, shift_fit(m64, true64), (0, 1, 2))
        c15 = curve15(m64, true64, MASK)
        cc = np.fft.ifftn(np.fft.fftn(true_map) * np.conj(np.fft.fftn(mean))).real
        sh = np.array(np.unravel_index(np.argmax(cc), cc.shape)); sh = np.where(sh > n // 2, sh - n, sh)
        full_mean = np.roll(mean, sh, (0, 1, 2))
        raw = np.asarray(shell_fsc(full_mean, np.asarray(true_map, np.float32)), np.float64)
        below = int(np.argmax(raw[1:] < 0.5)) + 1 if np.any(raw[1:] < 0.5) else None
        pc_power = float(np.sum(np.abs(th[:, 1:]) ** 2) / np.sum(np.abs(th[:, 0]) ** 2))
        out = {'checkpoint': c, 'pose_median_deg': df['pose_median_deg'], 'frac_below_10': df['frac_below_10'],
               'mean_fsc_raw_1_15_box64': c15['raw_mean_1_15'], 'mean_fsc_masked_1_15_box64': c15['masked_mean_1_15'],
               'fsc_box_full_raw': raw.tolist(), 'first_shell_below_0p5': below, 'pc_over_mean_power_model': pc_power}
        json.dump(out, open(d + '/h1_metrics.json', 'w'), indent=1)
        print(Path(run).name, c, 'pose %.1f, %.0f%% <10, mean FSC(1-15) %.3f, FSC<0.5 at shell %s, PC/mean power %.3g' % (
            out['pose_median_deg'], 100 * out['frac_below_10'], out['mean_fsc_raw_1_15_box64'], below, pc_power), flush=True)
