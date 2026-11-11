"""d5hack zsel (truth-free): python zselect.py INFER_ALL.npz FRACTION OUT_INDEX.npy -- top FRACTION of particles by |z| after
per-dimension robust standardisation ((z - median) / std)."""
import sys, numpy as np
z = np.load(sys.argv[1])['z']; f = float(sys.argv[2])
s = np.linalg.norm((z - np.median(z, 0)) / z.std(0), axis=1)
k = int(round(f * len(s))); idx = np.sort(np.argsort(-s)[:k]).astype(np.int64)
np.save(sys.argv[3], idx); print(sys.argv[3], k, 'score cut', float(np.sort(s)[-k]))
