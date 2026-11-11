"""d5hack: latents + poses for ALL particles of a training manifest from a PPCA checkpoint, with the suite's own inference
(suite_score.py lines: scorer_general.infer_fixed_subset at max(3, stage order), radius 31, +-6 px step 2). Truth-free.
python infer_all.py CHECKPOINT.npz TRAINING_MANIFEST OUT.npz [CHUNK=20000]"""
import sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
TOOLS = '/scratch/gpfs/GILLES/mg6942/ppca_test_suite_20261006/tools'
sys.path.insert(0, TOOLS)
from suite_common import PROTOCOL_ROOT
from suite_score import load_checkpoint
sys.path.insert(0, str(PROTOCOL_ROOT / 'run_tools'))
from scorer_general import infer_fixed_subset
from relax.commands.ppca_initial_model import load_training
ck, man, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
cp, meta, arrays = load_checkpoint(ck.parent, ck)
theta = arrays['theta']; q = theta.shape[1] - 1
stages = meta['config']['stages']
order = max(3, int(next(hp for start, _, hp in reversed(stages) if start <= max(meta['iteration'], 1))))
data, _, _ = load_training(man)
prior = arrays['direction_prior']
state = SimpleNamespace(iteration=max(1, int(meta['iteration'])), theta=theta, noise=arrays['noise'],
                        direction_prior=prior if prior.size else None, direction_order=meta['direction_order'],
                        offset_variance=float(meta['offset_variance']))
config = SimpleNamespace(q=q, gemm_precision='fp32', stage=lambda iteration: (31, order))
Z, R, T = [], [], []; t0 = time.time()
for s in range(0, data.n_images, chunk):
    ids = np.arange(s, min(s + chunk, data.n_images), dtype=np.int64)
    z, ri, ti, rgrid = infer_fixed_subset(data, state, ids, config)
    Z.append(np.asarray(z)); R.append(np.asarray(ri)); T.append(np.asarray(ti))
    print('chunk', s, f'{time.time() - t0:.0f} s', flush=True)
np.savez(out, z=np.concatenate(Z), rotation_idx=np.concatenate(R), translation_idx=np.concatenate(T), healpix_order=order,
         checkpoint=str(ck), iteration=int(meta['iteration']))
print('INFER_ALL_DONE', out, f'{time.time() - t0:.0f} s')
