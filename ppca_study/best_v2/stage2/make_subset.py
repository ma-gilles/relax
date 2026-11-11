"""d5hack: subset training folder. python make_subset.py SRC_TRAINING_DIR INDEX.npy OUT_DIR
OUT_DIR/training/{particles.mrcs, ctf.pkl, neutral_poses.pkl, particle_ids.npy (arange), manifest.json}; OUT_DIR/orig_index.npy."""
import hashlib, json, pickle, sys, shutil
from pathlib import Path
import numpy as np, mrcfile
src, idx, out = Path(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
idx = np.sort(idx.astype(np.int64)); tr = out / 'training'; tr.mkdir(parents=True, exist_ok=False)
m = json.loads((src / 'manifest.json').read_text())
stack = mrcfile.mmap(src / m['particles'], permissive=True, mode='r')
with mrcfile.new(tr / 'particles.mrcs', overwrite=False) as f:
    f.set_data(np.ascontiguousarray(stack.data[idx]).astype(np.float32)); f.voxel_size = stack.voxel_size; f.set_image_stack()
ctf = pickle.load(open(src / m['ctf'], 'rb')); pickle.dump(ctf[idx], open(tr / 'ctf.pkl', 'wb'))
rot, tr2 = pickle.load(open(src / m['neutral_poses'], 'rb')); pickle.dump((rot[idx], tr2[idx]), open(tr / 'neutral_poses.pkl', 'wb'))
np.save(tr / 'particle_ids.npy', np.arange(len(idx), dtype=np.int64)); np.save(out / 'orig_index.npy', idx)
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
m2 = dict(m); m2['n_images'] = int(len(idx))
m2['files'] = {n: h(tr / n) for n in (m['particles'], m['ctf'], m['neutral_poses'], m['particle_ids'])}
(tr / 'manifest.json').write_text(json.dumps(m2, indent=2) + '\n')
print(out, len(idx))
