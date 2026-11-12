"""ribodeep Addendum 7 (lead change 2026-11-11): S3 offset seed from stage V's last stored MAP offsets.
python seed_offsets.py DS V_RUN OUT.npy -> OUT.npy (n_original, 2) px for coupled_arm_shift.py --init-offsets (indexed by original id;
the A_picked fixtures have particle_ids = arange, checked). Visited particles: V's train_shift_3000 shift_px; unvisited: mean of the
visited (batch-mean fill, as realshift's --recentre-seed-unvisited mean). Writes OUT.json: seeded fraction, fill, and agreement with
the consensus offsets (target = s * offset_A / apix for s = +-1; per-axis correlation and rms after removing the mean; DIAGNOSTIC)."""
import json, sys
from pathlib import Path
import numpy as np
RS = Path('/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/realshift_20261109')
ds, run, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
z = np.load(run / 'train_shift_3000.npz'); sh, vis = z['shift_px'].astype(np.float64), z['visit'] > 0
assert (z['original_ids'] == z['local_ids']).all(), 'particle ids are not arange'
fill = sh[vis].mean(0); seed = np.where(vis[:, None], sh, fill[None, :]).astype(np.float32)
np.save(out, seed)
info = json.load(open(RS / 'fixtures' / ds / 'evaluation/info.json')); apix = float(info['apix'])
offA = np.load(RS / 'fixtures' / ds / 'evaluation/consensus_offsets_A.npy').astype(np.float64)
agree = {}
for s in (1, -1):
    t = s * offA / apix; d = (seed - seed.mean(0)) - (t - t.mean(0))
    agree[f'sign{s:+d}'] = dict(corr_xy=[float(np.corrcoef(seed[:, k], t[:, k])[0, 1]) for k in (0, 1)],
                                rms_diff_px=float(np.sqrt((d ** 2).mean())), median_abs_diff_px=float(np.median(np.linalg.norm(d, axis=1))))
rep = dict(ds=ds, v_run=str(run), n=int(len(seed)), seeded_fraction=float(vis.mean()), unvisited=int((~vis).sum()), fill_px=fill.tolist(),
           seed_rms_px=float(np.sqrt(((seed - seed.mean(0)) ** 2).mean())), consensus_rms_px=float(np.sqrt(((offA - offA.mean(0)) ** 2).mean()) / apix),
           agreement=agree, note='DIAGNOSTIC: chosen_sign in info.json = %d' % int(info['centring']['chosen_sign']))
json.dump(rep, open(str(out).replace('.npy', '.json'), 'w'), indent=1); print(json.dumps(rep))
