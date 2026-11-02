"""make_tau2.py: oracle tau2 per shell at box 64 from the crops' gt_subspace.npz: tau2(s) = (1/q) sum over the top q true PCs of lambda_k * shell mean |FT(u_k)|^2, centred FFT, backward norm, the units of volumes_fourier and theta (checked: checkpoint means match truth at box 128).
Index = rounded shell. Writes tau2_<COMP>_b64.npy."""
import json, numpy as np
T = '/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round17_simple_sgd_20261102/'; N = '/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_suite_b150_20261102/datasets/'
n = 64; f = np.arange(n) - n // 2; s = np.rint(np.sqrt(sum(x ** 2 for x in np.meshgrid(f, f, f, indexing='ij')))).astype(int).ravel(); cn = np.bincount(s).clip(1)
for comp, q in (('C', 10), ('D40', 2), ('D20', 2), ('T10', 9)):
    z = np.load(f'{N}{comp}__realistic__noise1__b128crop64/evaluation/gt_subspace.npz')
    t = np.zeros(len(cn))
    for k in range(q):
        F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(z['pcs'][k].astype(np.float64)))); t += z['eigenvalues'][k] * np.bincount(s, (np.abs(F) ** 2).ravel(), len(cn))
    t = t / cn / q; np.save(f'{T}tau2_{comp}_b64.npy', t)
    print(comp, 'tau2 shells 1 5 10 15 20 31', ['%.3g' % t[i] for i in (1, 5, 10, 15, 20, 31)], 'eig', z['eigenvalues'][:3], 'rank', int(z['rank']))
