#!/usr/bin/env python3
"""Score one relax PPCA checkpoint against one test-suite dataset.

    suite_score.py --run <relax ppca_initial_model output dir> --eval <dataset>/evaluation [--checkpoint ...] [--output ...]

Reads only arrays from the checkpoint (theta, noise, direction_prior, metadata), so any
relax version with the same checkpoint layout can be scored. Inference of latents and
poses on the fixed evaluation subsets uses the frozen protocol code (pinned relax dee8cfa,
``scorer_general.infer_fixed_subset``) at the checkpoint's HEALPix order (at least 3),
full radius 31, translations +-6 px step 2. GT enters only after inference. Datasets with
true shifts (``shifts`` in the evaluation manifest, the ``*_shift`` variants) are inferred on
a finer grid by default (radius 9 px, step 0.5 px; ``--translation-grid``), the same code with only
the translation grid changed, because the protocol grid cannot represent shifts beyond 6 px.

Box 128 (``__b128`` datasets, same field of view as box 64; added 2026-10-07): a shell is the
same physical frequency at both boxes (shell s = s / field of view), so every shell-based
quantity is kept: inference runs on the box-128 images with the box-128 model, by default only
up to shell 31 (current size 62, the band box 64 sees; ``--inference-band full`` uses current
size 124). At HEALPix 3 the full band makes the 7.5-degree rotation grid the limit: on shifted A
the true model's k-means ARI is 0.35 with the full band, 0.55 with shells <= 31 (box 64: 0.49),
and 0.80 with the full band at HEALPix 4 (``--healpix-order 4``, about 15x slower). An exact
oracle at the true poses reaches 1.0, so the data and noise model are not the cause. All volume metrics are computed
after a Fourier crop of the learned and GT maps to box 64, which keeps every shell up to 31
unchanged. The band limit of shell 15, the mask and the shell bands therefore mean the same
resolution as at box 64 and the numbers are directly comparable with the box-64 twin.
Translation grids scale with the pixel (protocol +-12 px step 4); shifted box-128 datasets use the
'finest' grid by default (radius 18 px step 0.5, 4053 offsets), the same 0.5-native-px step as box 64:
with step 1 px the true model reached k-means ARI 0.21 instead of 0.35 on A.

Metrics (helpers reused from ppca_cryobench_scaling_20261002/run_tools/scorer_general.py
and relax scripts/evaluate_vdam_ppca_pilot.py unless stated):
  frame       learned mean rigid+hand registered to the true weighted mean (shells <= 15, ``_fit``;
              protocol linear resampling);
  subspace    principal angles between the registered learned PCs and the true top PCs
              (eigenvectors of the state covariance under the dataset's state weights), all
              masked (FSC mask) and band-limited to shell 15; plus the fraction of true covariance
              energy captured;
  states      per-state map mu + W * mean(z | state) on the stratified subset, independent
              rigid+hand fit to that GT state, FSC shells 1-15 (protocol metric of record);
              datasets with more than 12 states fit only the distinctive state individually
              and report shared-frame FSC for all states;
  distinctive (D datasets) FSC to the distinctive state of the mean, of mu + W c for both
              centres c of a label-free 2-means clustering of the latents, and of the oracle
              class-mean map (best label-free map reported); purity, recall and AUROC of the
              distinctive state for the 2-means split, on the random subset;
  mean        FSC of the registered learned mean to the true weighted mean (summary headline);
  latent      (<= 12 states) k-means (k = states) ARI/NMI against labels, within-state RMS spread over
              mean between-state distance, and distance of each latent state mean from its true
              position (GT state projected on the registered learned PCs), over the true between-
              state distance, raw and after the best common scale (posterior means shrink);
  rigid       share of each PC's power in the span of the six rigid-motion derivatives of
              mu + W mean(z), per shell band (definition of the ws2 axis_metrics.py 'rigid_share');
              plus each PC's power fraction outside the true heterogeneity span;
  poses       angular error against the simulator rotations in the registered mean frame
              (pose formula validated by the protocol's pose_proof_v1);
  shifts      (added 2026-10-07) in-plane offset error against the true shifts, px and Angstrom,
              median and p90 over the inferred particles, raw and after removing the global
              translation: a learned map centred at c (3D, its own frame) moves every particle's
              best offset by the in-plane projection of c under that particle's inferred rotation.
              c is fitted by least trimmed squares (best half) to the offset differences of the
              particles with pose error < 10 degrees and removed; it is the translation the map registration absorbs. All metrics above it
              are unchanged.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite_common import PROTOCOL_ROOT, SOURCE, environment, sha, write

N = 64
RADIUS = 15  # FSC / registration band limit, as in the protocol


def load_checkpoint(run, checkpoint):
    if checkpoint is None:
        candidates = sorted(Path(run).glob('checkpoint_[0-9]*.npz'))
        if not candidates:
            raise FileNotFoundError(f'No checkpoint_*.npz in {run}')
        checkpoint = candidates[-1]
    with np.load(checkpoint, allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata']))
        arrays = dict(theta=saved['theta'].copy(), noise=saved['noise'].copy(), direction_prior=saved['direction_prior'].copy())
    if arrays['theta'].dtype != np.complex64 or arrays['theta'].shape[0] not in [n * n * (n // 2 + 1) for n in (64, 128)] or not np.isfinite(arrays['theta']).all():
        raise ValueError('Expected a finite box-64 or box-128 complex64 half-Fourier PPCA model')
    return Path(checkpoint), meta, arrays


def auroc(score, positive):
    """Mann-Whitney AUROC with average ranks for ties (score higher = more positive)."""
    from scipy.stats import rankdata
    positive = np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def two_means(z, iterations=100):
    """Deterministic 2-means: start at the point nearest the median and the point farthest from it."""
    first = z[np.argmin(np.linalg.norm(z - np.median(z, axis=0), axis=1))]
    centers = np.stack([first, z[np.argmax(np.linalg.norm(z - first, axis=1))]])
    for _ in range(iterations):
        assign = np.argmin(np.linalg.norm(z[:, None] - centers[None], axis=-1), axis=1)
        updated = np.stack([z[assign == k].mean(axis=0) if np.any(assign == k) else centers[k] for k in range(2)])
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers, assign


def weighted_subspace(learned, true_pcs, eigenvalues):
    """Principal angles between span(learned) and the true top-min(q, r) PCs; energy captured."""
    q, r = len(learned), len(true_pcs)
    a = learned.reshape(q, -1).T.astype(np.float64)
    u = true_pcs.reshape(r, -1).T.astype(np.float64)
    qa, _ = np.linalg.qr(a)
    qu, _ = np.linalg.qr(u)  # band-limited true PCs are no longer exactly orthonormal
    m = min(q, r)
    top, _ = np.linalg.qr(u[:, :m])
    cos_top = np.clip(np.linalg.svd(qa.T @ top, compute_uv=False), -1, 1)
    cos_full = np.clip(np.linalg.svd(qa.T @ qu, compute_uv=False), -1, 1)
    captured = np.sum((qa.T @ u) ** 2, axis=0) / np.sum(u ** 2, axis=0)
    return {'learned_q': q, 'true_rank': r, 'compared_top_k': m,
            'principal_angles_deg_vs_true_top_k': np.degrees(np.arccos(cos_top)).tolist(),
            'max_angle_deg_vs_true_top_k': float(np.degrees(np.arccos(cos_top.min()))),
            'principal_angles_deg_vs_full_true_span': np.degrees(np.arccos(cos_full)).tolist(),
            'true_energy_captured_fraction': float(np.sum(eigenvalues * captured) / np.sum(eigenvalues)),
            'true_energy_in_top_q_fraction': float(np.sum(eigenvalues[:q]) / np.sum(eigenvalues)),
            'per_true_pc_fraction_in_learned_span': captured[:min(r, 20)].tolist()}


# Rigid-motion absorption (axis 3), copied from the ws2 study so numbers are comparable:
# ppca_sgd_vs_vdam_20261003/improve_20261004/axis_metrics.py (rigid_derivatives, span_share, bands).
FREQ = np.fft.fftfreq(N) * N
KX, KY, KZ = np.meshgrid(FREQ, FREQ, FREQ, indexing='ij')
SHELL = np.rint(np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)).astype(int)
_C = np.arange(N) - N // 2
GX, GY, GZ = np.meshgrid(_C, _C, _C, indexing='ij')
BANDS = [(1, 5), (5, 11), (11, 16), (16, 20), (20, 24), (24, 32)]
SHELLS = list(range(1, 32))


def rigid_derivatives(m, real=False):
    """FFTs (or real volumes) of the six rigid-motion derivatives of a real volume (grid centred at N//2)."""
    F = np.fft.fftn(m)
    grads = [np.fft.ifftn(2j * np.pi * k / N * F).real for k in (KX, KY, KZ)]
    gx, gy, gz = grads
    rots = [GY * gz - GZ * gy, GZ * gx - GX * gz, GX * gy - GY * gx]
    return grads + rots if real else [np.fft.fftn(v) for v in grads + rots]


def _realvec(f, m):
    v = f[m]
    return np.concatenate([v.real, v.imag])


def span_share(basis_ffts, target_ffts):
    """Per shell and per target: power inside the least-squares span of the basis, and total power."""
    inside, total = np.zeros((len(target_ffts), 32)), np.zeros((len(target_ffts), 32))
    for sh in SHELLS:
        m = SHELL == sh
        A = np.stack([_realvec(b, m) for b in basis_ffts], axis=1)
        for j, tgt in enumerate(target_ffts):
            y = _realvec(tgt, m)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            inside[j, sh] = np.sum((A @ coef) ** 2)
            total[j, sh] = np.sum(y ** 2)
    return inside, total


def share_bands(inside, total):
    out = {f'{lo}-{hi - 1}': float(inside[..., lo:hi].sum() / max(total[..., lo:hi].sum(), 1e-300)) for lo, hi in BANDS}
    out['1-15'] = float(inside[..., 1:16].sum() / max(total[..., 1:16].sum(), 1e-300))
    return out


def clustering_metrics(z, labels, true_positions):
    """k-means ARI/NMI, spread ratio, and distance of the latent state means from their true positions."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    values = np.unique(labels)
    k = len(values)
    predicted = KMeans(k, n_init=20, random_state=0).fit_predict(z)
    means = np.stack([z[labels == v].mean(axis=0) for v in values])
    pair = lambda x: np.linalg.norm(x[:, None] - x[None], axis=-1)[np.triu_indices(len(x), 1)]
    within = float(np.sqrt(np.mean([np.sum((z[labels == v] - means[i]) ** 2, axis=1).mean() for i, v in enumerate(values)])))
    between = float(pair(means).mean())
    out = {'k': int(k), 'kmeans_ari': float(adjusted_rand_score(labels, predicted)), 'kmeans_nmi': float(normalized_mutual_info_score(labels, predicted)),
           'within_rms_spread': within, 'between_state_mean_distance': between, 'within_over_between': within / between if between > 0 else None}
    for name, t in true_positions.items():
        t = t[values]
        t_between = float(pair(t).mean())
        d = np.linalg.norm(means - t, axis=1)
        scale = float(np.sum(means * t) / max(np.sum(t * t), 1e-300))  # posterior means shrink toward the prior mean
        d_scaled = np.linalg.norm(means - scale * t, axis=1)
        out[f'state_mean_position_vs_{name}'] = {
            'true_positions': t.tolist(), 'true_between_distance': t_between,
            'distance_over_true_between': (d / t_between).tolist(), 'mean_distance_over_true_between': float(d.mean() / t_between),
            'best_scale': scale, 'mean_distance_over_true_between_after_scale': float(d_scaled.mean() / t_between)}
    return out


TRANSLATION_GRIDS = {'protocol': (6, 2), 'fine': (9, .5), 'finest': (9, .25)}  # in box-64 px; scaled by box / 64  # (max_pixel, step) of relax.sampling.get_relion_translation_grid
# Conventions fixed by validation (2026-10-07, 2k pilot of A realistic_shift; job 15175717): relax's best grid offset equals the
# simulator translation (+1; -1 gives 6 px median error), and a 3D model shift c appears in-plane as rows 0-1 of the
# inferred rotation times c (applied (1.5, -2, 0.75) voxels, fitted (1.48, -1.96, 0.63); R^T gives 1.2 px residual median).
SHIFT_SIGN, SHIFT_PROJECTION = 1, 'rows'


def protocol_translation_grid():
    """Grid of inference.npz files written before translation offsets were saved (all used the protocol grid)."""
    from relax import sampling
    return sampling.get_relion_translation_grid(*TRANSLATION_GRIDS['protocol']).astype(np.float32)


def crop_to_box64(volume):
    """Fourier crop of a real cubic volume to box 64 (same field of view, shells 0-31 kept exactly)."""
    from recovar.core import fourier_transform_utils as ftu
    n = volume.shape[0]
    if n == N:
        return volume
    lo = n // 2 - N // 2
    cropped = np.asarray(ftu.get_dft3(volume))[lo:lo + N, lo:lo + N, lo:lo + N]
    return (np.asarray(ftu.get_idft3(cropped)).real * (N / n) ** 3).astype(np.float32)


def infer_on_grid(data, state, ids, config, max_pixel, step, box=N, current_size=None, order=3):
    """scorer_general.infer_fixed_subset (frozen protocol inference) run with another translation grid.

    The frozen source is executed with textual changes only to the grid call, its size check and the image tile
    size (memory blocking only; per-image results do not depend on it): GPU memory grows as tile x translations x q,
    about 147 kB each (150 images x 1009 translations x q=2 needed 42 GiB), so the tile is sized for about 6 GiB (12 GiB failed at q=1); it also grows with the rotation count (x8 per HEALPix order)."""
    import inspect
    import scorer_general
    from relax import sampling
    grid = sampling.get_relion_translation_grid(max_pixel, step).astype(np.float32)
    code = inspect.getsource(scorer_general.infer_fixed_subset)
    tile = int(np.clip(6 * 2 ** 30 // (147e3 * (box / N) ** 2 * len(grid) * max(config.q, 1) * 8 ** max(order - 3, 0)), 1, 150))
    for old, new in (('get_relion_translation_grid(max_pixel=6, pixel_offset=2)', f'get_relion_translation_grid(max_pixel={max_pixel}, pixel_offset={step})'),
                     ('translations.shape != (29, 2)', f'translations.shape != {grid.shape}'),
                     ('image_batch_size=150', f'image_batch_size={tile}'), ('range(0, len(ids), 150)', f'range(0, len(ids), {tile})'),
                     ('ids[begin:begin + 150]', f'ids[begin:begin + {tile}]'), ('current_size=62', f'current_size={current_size or 62 * box // N}')):
        if code.count(old) != 1:
            raise RuntimeError(f'Frozen inference source changed: {old!r}')
        code = code.replace(old, new)
    namespace = dict(vars(scorer_general))
    exec(compile(code, scorer_general.__file__, 'exec'), namespace)
    return (*namespace['infer_fixed_subset'](data, state, ids, config), grid)


def shift_metrics(estimated, true, rotations, voxel, pose_ok, labels=None, distinctive=None):
    """Offset error (px, Angstrom) raw and after removing the global translation c of the learned map.

    estimated: inferred grid offsets; true: simulator translations; rotations: inferred rotations (model frame).
    Model: estimated - SHIFT_SIGN * true = P(R) c + error, with P(R) the in-plane projection (rows 0-1 of R,
    or of R^T, per SHIFT_PROJECTION); c is fitted by least trimmed squares (the best half, 10 iterations) on the
    particles whose inferred rotation is right (pose_ok: error < 10 degrees; all particles if fewer than 20), because
    particles the model cannot align (e.g. the non-distinctive D states) carry no information about c."""
    est, tru = np.asarray(estimated, np.float64), np.asarray(true, np.float64)
    rot = np.asarray(rotations, np.float64)
    proj = rot[:, :2, :] if SHIFT_PROJECTION == 'rows' else np.transpose(rot, (0, 2, 1))[:, :2, :]
    d = est - SHIFT_SIGN * tru
    fit = np.asarray(pose_ok, bool) if np.sum(pose_ok) >= 20 else np.ones(len(d), bool)
    a, y = proj[fit].reshape(-1, 3), d[fit].reshape(-1)
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    for _ in range(10):
        r = np.linalg.norm(d[fit] - proj[fit] @ c, axis=1)
        keep = np.repeat(r <= np.median(r), 2)
        c = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
    raw = np.linalg.norm(d, axis=1)
    after = np.linalg.norm(d - proj @ c, axis=1)

    def stats(err):
        return {'n': int(len(err)), 'median_px': float(np.median(err)), 'p90_px': float(np.quantile(err, .9)),
                'median_ang': float(np.median(err) * voxel), 'p90_ang': float(np.quantile(err, .9) * voxel),
                'fraction_below_1px': float(np.mean(err < 1))}
    out = {'global_translation_px': c.tolist(), 'global_translation_norm_px': float(np.linalg.norm(c)),
           'global_translation_norm_ang': float(np.linalg.norm(c) * voxel),
           'n_fit_pose_below_10deg': int(fit.sum()), 'raw': stats(raw), 'after_global': stats(after),
           'after_global_pose_below_10deg': stats(after[pose_ok]) if np.any(pose_ok) else None}
    if labels is not None and distinctive is not None:
        out['after_global_by_group'] = {str(distinctive): stats(after[labels == distinctive]), 'others': stats(after[labels != distinctive])}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--eval', type=Path, required=True, help='<dataset>/evaluation directory')
    p.add_argument('--checkpoint', type=Path, help='Default: highest checkpoint_*.npz in --run')
    p.add_argument('--training-manifest', type=Path, help='Default: <dataset>/training/manifest.json')
    p.add_argument('--output', type=Path, help='Default: <run>/suite_score_<checkpoint stem>')
    p.add_argument('--healpix-order', type=int, help='Default: max(3, the checkpoint stage order)')
    p.add_argument('--translation-grid', choices=['auto', *TRANSLATION_GRIDS], default='auto',
                   help="Inference translation grid in box-64 px (scaled at box 128): protocol (+-6 step 2), fine (radius 9 step 0.5) or finest (radius 9 step 0.25); "
                        "auto = protocol without true shifts, else a step of 0.5 native px (fine at box 64, finest at box 128)")
    p.add_argument('--inference-band', choices=['full', 'box64'], default='box64',
                   help='Box 128 only: infer only shells <= 31 (current size 62, the box-64 band; default) or up to the box-128 Nyquist (full, current size 124; '
                        'at HEALPix 3 the rotation grid then limits the score, use --healpix-order 4)')
    p.add_argument('--reuse-inference', action='store_true',
                   help='Reuse <output>/inference.npz when its IDs and HEALPix order match (rescoring without a GPU)')
    a = p.parse_args()
    started = time.time()
    runtime = environment(cpu_only=False)
    sys.path.insert(0, str(PROTOCOL_ROOT / 'run_tools'))
    from scorer_general import curve15, infer_fixed_subset, latent_metrics, pose_errors, pose_metrics, states_from_model
    from relax.commands.ppca_initial_model import load_training
    from relax.diagnostics.gt_metrics import relion_alignment_rotations
    from recovar.core import fourier_transform_utils as ftu
    from recovar.utils.helpers import load_mrc
    from scripts.evaluate_vdam_ppca_pilot import _bandlimit, _fit

    cp, meta, arrays = load_checkpoint(a.run, a.checkpoint)
    theta = arrays['theta']
    q = theta.shape[1] - 1
    stages = meta['config']['stages']
    stage_order = next(hp for start, _, hp in reversed(stages) if start <= max(meta['iteration'], 1))
    order = a.healpix_order if a.healpix_order is not None else max(3, int(stage_order))
    manifest_path = a.training_manifest or a.eval.parent / 'training' / 'manifest.json'
    evaluation = json.loads((a.eval / 'manifest.json').read_text())
    output = a.output or a.run / f'suite_score_{cp.stem}'
    n_box = int(evaluation.get('box', N))
    # auto: shifted datasets get offsets every 0.5 native px (fine at box 64, finest at box 128), radius 9 box-64 px.
    grid_name = a.translation_grid if a.translation_grid != 'auto' else (('fine' if n_box == N else 'finest') if evaluation.get('shifts') else 'protocol')
    if theta.shape[0] != n_box * n_box * (n_box // 2 + 1):
        raise ValueError(f'Checkpoint box does not match the dataset box {n_box}')
    px = n_box // N  # pixels per box-64 pixel (same field of view)
    current_size = 62 * px if a.inference_band == 'full' else 62
    max_pixel, step = (x * px for x in TRANSLATION_GRIDS[grid_name])
    from relax import sampling
    current_grid = sampling.get_relion_translation_grid(max_pixel, step).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Inference on the fixed subsets (no GT opened yet).
    stratified = np.load(a.eval / 'eval_ids_stratified.npy')
    random_ids = np.load(a.eval / 'eval_ids_random2000.npy')
    ids = np.union1d(stratified, random_ids).astype(np.int64)
    reused = False
    if a.reuse_inference and (output / 'inference.npz').is_file():
        with np.load(output / 'inference.npz', allow_pickle=False) as saved:
            if np.array_equal(saved['particle_ids'], ids) and int(saved['healpix_order']) == order and \
                    saved.get('checkpoint_sha256', np.asarray(sha(cp))).item() == sha(cp) and \
                    int(saved.get('current_size', np.asarray(62))) == current_size and \
                    np.array_equal(saved['translation_offsets'] if 'translation_offsets' in saved else protocol_translation_grid(), current_grid):
                z, rotation_idx, reused = saved['z'].copy(), saved['rotation_idx'].copy(), True
                translation_idx = saved['translation_idx'].copy() if 'translation_idx' in saved else None  # absent before 2026-10-07
    if reused:
        from relax import sampling
        rotation_grid = sampling.get_relion_hidden_rotation_grid(order, matrices=True).astype(np.float32)
        translation_grid = sampling.get_relion_translation_grid(max_pixel, step).astype(np.float32)
        inference_seconds = 0.0
    else:
        data, _, _ = load_training(manifest_path)
        prior = arrays['direction_prior']
        state = SimpleNamespace(iteration=max(1, int(meta['iteration'])), theta=theta, noise=arrays['noise'],
                                direction_prior=prior if prior.size else None, direction_order=meta['direction_order'],
                                offset_variance=float(meta['offset_variance']))
        config = SimpleNamespace(q=q, gemm_precision='fp32', stage=lambda iteration: (31, order))
        t0 = time.time()
        if grid_name == 'protocol' and n_box == N:
            from relax import sampling
            z, rotation_idx, translation_idx, rotation_grid = infer_fixed_subset(data, state, ids, config)
            translation_grid = sampling.get_relion_translation_grid(max_pixel, step).astype(np.float32)
        else:
            z, rotation_idx, translation_idx, rotation_grid, translation_grid = infer_on_grid(data, state, ids, config, max_pixel, step, n_box, current_size, order)
        inference_seconds = time.time() - t0
        np.savez(output / 'inference.npz', particle_ids=ids, z=z, rotation_idx=rotation_idx, healpix_order=order, checkpoint_sha256=sha(cp),
                 translation_idx=translation_idx, translation_grid=grid_name, translation_offsets=translation_grid,
                 current_size=current_size)
    position = {int(i): n for n, i in enumerate(ids)}
    rows_s = np.asarray([position[int(i)] for i in stratified])
    rows_r = np.asarray([position[int(i)] for i in random_ids])

    # 2. Ground truth.
    with np.load(a.eval / 'truth.npz', allow_pickle=False) as saved:
        labels_all, rotations_all, translations_all = saved['labels'].copy(), saved['rotations'].copy(), saved['translations'].copy()
    with np.load(a.eval / 'gt_subspace.npz', allow_pickle=False) as saved:
        gt_mean, gt_pcs, eigenvalues = crop_to_box64(saved['mean']), np.stack([crop_to_box64(v) for v in saved['pcs']]), saved['eigenvalues'].copy()
    states_meta = evaluation['states']
    gt = np.stack([crop_to_box64(np.asarray(load_mrc(str(a.eval / s['gt_map'])), np.float32)) for s in states_meta])
    gt_hash = [sha(a.eval / s['gt_map']) for s in states_meta]
    native_voxel = float(evaluation['voxel_size'])
    voxel = native_voxel * px  # voxel of the box-64 volumes all volume metrics use
    distinctive = evaluation.get('distinctive_state')
    coords = np.arange(N, dtype=np.float32) - N // 2
    rr = np.sqrt(sum(x ** 2 for x in np.meshgrid(coords, coords, coords, indexing='ij')))
    mask = ((1 + np.cos(np.pi * np.clip((rr - 24) / 5, 0, 1))) / 2).astype(np.float32)
    search = relion_alignment_rotations(2)

    basis = np.asarray(ftu.get_idft3_real(theta.T.reshape(q + 1, n_box, n_box, n_box // 2 + 1), (n_box,) * 3), np.float32)
    basis = np.stack([crop_to_box64(v) for v in basis])
    model_states = lambda means: np.stack([crop_to_box64(v) for v in states_from_model(theta, means, n_box)])
    mean_hash = hashlib.sha256(gt_mean.tobytes()).hexdigest()
    common_fit, common_receipt = _fit(_bandlimit(basis[0], RADIUS), _bandlimit(gt_mean, RADIUS), search, voxel, mean_hash)
    register = lambda v: common_fit.apply(v, voxel_size=voxel, gt_sha256=mean_hash)
    report = dict(schema='ppca_test_suite_score_v1', dataset_id=evaluation['dataset_id'], composition=evaluation['composition'],
                  checkpoint=str(cp), checkpoint_sha256=sha(cp), iteration=meta['iteration'], q=q, inference_healpix_order=order,
                  box=n_box, volume_metrics_box=N, inference_current_size=current_size, n_inferred=len(ids), inference_translation_grid=dict(name=grid_name, max_pixel=max_pixel, step_px=step, n=len(translation_grid)), inference_seconds=inference_seconds, inference_reused=reused, eval_manifest_sha256=sha(a.eval / 'manifest.json'),
                  scorer_sha256=sha(__file__), runtime=runtime, mean_common_fit=common_fit.to_dict(), mean_common_fit_receipt=common_receipt,
                  mean_vs_true_mean_fsc=curve15(register(basis[0]), gt_mean, mask))

    # 3. Subspace.
    if q:
        # The FSC mask (cosine edge 24-29 px) on both sides: resampling zero-fills the box faces,
        # which alone tilts the weak true PCs by ~2 degrees; masked, the floor is the fit's own error.
        learned = np.stack([_bandlimit(mask * register(v), RADIUS) for v in basis[1:]])
        true_bl = np.stack([_bandlimit(mask * v, RADIUS) for v in gt_pcs])
        report['subspace'] = weighted_subspace(learned, true_bl, eigenvalues)
        # Share of each learned PC's power outside the full true heterogeneity span (same frame, mask, band limit).
        qu, _ = np.linalg.qr(true_bl.reshape(len(true_bl), -1).T.astype(np.float64))
        flat = learned.reshape(q, -1).astype(np.float64)
        inside = np.sum((flat @ qu) ** 2, axis=1) / np.sum(flat ** 2, axis=1)
        report['subspace']['per_learned_pc_fraction_outside_true_span'] = (1 - inside).tolist()

        # 3b. PCs absorbing pose error: ws2 'rigid share' (axis_metrics.py), in the model's own frame (no
        # registration): per shell, least-squares share of each PC's power in the span of the six rigid-motion
        # derivatives of mu + W mean(z) (z over the random subset); reported in ws2's shell bands plus 1-15.
        centred = basis[0] + np.einsum('q,qxyz->xyz', z[rows_r].mean(axis=0).astype(np.float32), basis[1:])
        derivatives = rigid_derivatives(centred)
        ins, tot = span_share(derivatives, [np.fft.fftn(v) for v in basis[1:]])
        if len(gt) <= 12:
            true_components = [np.fft.fftn(x - gt_mean) for x in gt]
        else:
            true_components = [np.fft.fftn(v) for v in gt_pcs[:10]]
        true_ins, true_tot = span_share(rigid_derivatives(gt_mean), true_components)
        rng = np.random.default_rng(0)
        rand_ins, rand_tot = span_share(rigid_derivatives(gt_mean), [np.fft.fftn(rng.standard_normal((N, N, N))) for _ in range(2)])
        report['rigid_absorption'] = {
            'definition': 'ws2 axis_metrics.py rigid_share: per-shell least-squares share of PC power in span of 6 rigid derivatives of mu+W mean(z), own frame; bands are shell ranges (inclusive)',
            'pooled_all_pcs': share_bands(ins, tot), 'per_pc': [share_bands(ins[j], tot[j]) for j in range(q)],
            'per_pc_by_shell_1_31': [(ins[j, 1:32] / np.maximum(tot[j, 1:32], 1e-300)).tolist() for j in range(q)],
            'reference_true_components': share_bands(true_ins, true_tot),
            'reference_true_components_kind': 'true state minus true mean' if len(gt) <= 12 else 'top 10 true PCs',
            'reference_random_volume': share_bands(rand_ins, rand_tot)}

    # 4. Per-state maps from stratified latents.
    labels_s = labels_all[stratified]
    z_s = z[rows_s].astype(np.float64)
    latent = latent_metrics(z_s, labels_s) if q else None
    if latent is not None:
        report['latent'] = {k: v for k, v in latent.items() if k != 'means_by_gt_state'}
    present = np.unique(labels_s)
    means = np.stack([z_s[labels_s == k].mean(axis=0) for k in present]).astype(np.float32) if q else np.zeros((len(present), 0), np.float32)
    maps = model_states(means) if q else np.repeat(basis[:1], len(present), axis=0)
    individual = list(present) if len(gt) <= 12 else [distinctive]
    rows = []
    for j, k in enumerate(present):
        row = {'gt_state': int(k), 'source_index': states_meta[k]['source_index'], 'n_stratified': int(np.sum(labels_s == k)),
               'latent_mean': means[j].tolist(), 'shared_frame_fsc': curve15(register(maps[j]), gt[k], mask),
               'baseline_true_mean_fsc': curve15(gt_mean, gt[k], mask)}
        if k in individual:
            fit, receipt = _fit(_bandlimit(maps[j], RADIUS), _bandlimit(gt[k], RADIUS), search, voxel, gt_hash[k])
            row['individual_fit_fsc'] = curve15(fit.apply(maps[j], voxel_size=voxel, gt_sha256=gt_hash[k]), gt[k], mask)
            row['individual_fit'] = fit.to_dict()
        rows.append(row)
    report['states'] = rows
    fitted = [r['individual_fit_fsc']['raw_mean_1_15'] for r in rows if 'individual_fit_fsc' in r]
    report['states_summary'] = {'metric': 'individual rigid+hand fit, raw FSC mean over shells 1-15' if len(gt) <= 12 else 'shared-frame raw FSC mean over shells 1-15 (individual fit only for the distinctive state)',
                                'per_state': fitted if len(gt) <= 12 else [r['shared_frame_fsc']['raw_mean_1_15'] for r in rows],
                                'mean': float(np.mean(fitted if len(gt) <= 12 else [r['shared_frame_fsc']['raw_mean_1_15'] for r in rows])),
                                'min': float(np.min(fitted if len(gt) <= 12 else [r['shared_frame_fsc']['raw_mean_1_15'] for r in rows])),
                                'baseline_true_mean_per_state': [r['baseline_true_mean_fsc']['raw_mean_1_15'] for r in rows],
                                'baseline_note': 'FSC of the TRUE weighted mean to each state: what a model with no heterogeneity would reach'}

    # 4b. Latent clustering around the truth (datasets with at most 12 states).
    if q and len(gt) <= 12:
        def prepare(v):
            return _bandlimit(mask * v, RADIUS).reshape(-1).astype(np.float64)
        w_reg = learned.reshape(q, -1).T.astype(np.float64)  # registered, masked, band-limited learned PCs
        learned_mean_reg = prepare(register(basis[0]))
        positions = {}
        for name, origin in (('learned_mean', learned_mean_reg), ('true_mean', prepare(gt_mean))):
            positions[name] = np.stack([np.linalg.lstsq(w_reg, prepare(x) - origin, rcond=None)[0] for x in gt])
        report['latent_clustering'] = dict(clustering_metrics(z_s, labels_s, positions), subset='stratified (50 per state)',
            true_position_definition='least-squares coordinates of (GT state - origin) on the registered learned PCs (mask, shells <= 15); '
                                     'origin = registered learned mean (gauge of the model) or true weighted mean (as specified)')

    # 5. Distinctive-state recovery (D datasets).
    if distinctive is not None:
        target = gt[distinctive]
        z_r = z[rows_r].astype(np.float64)
        positive = labels_all[random_ids] == distinctive

        def fsc_to_target(volume):
            fit, _ = _fit(_bandlimit(volume, RADIUS), _bandlimit(target, RADIUS), search, voxel, gt_hash[distinctive])
            return curve15(fit.apply(volume, voxel_size=voxel, gt_sha256=gt_hash[distinctive]), target, mask)

        candidates = {'mean': basis[0]}
        block = {'distinctive_state': distinctive, 'distinctive_fraction': evaluation.get('distinctive_fraction'),
                 'n_random': len(random_ids), 'n_distinctive_in_random': int(positive.sum())}
        if q:
            # Label-free candidates: the two 2-means centres of the random-subset latents. Labels are
            # used only to score them (cluster precision/recall, FSC) and for the oracle rows.
            centers, assign = two_means(z_r)
            clusters = []
            for k in range(2):
                inside = assign == k
                candidates[f'mean_plus_W_cluster{k}'] = model_states(centers[k][None].astype(np.float32))[0]
                clusters.append({'center': centers[k].tolist(), 'size': int(inside.sum()),
                                 'precision_distinctive': float(positive[inside].mean()) if inside.any() else None,
                                 'recall_distinctive': float(positive[inside].sum() / max(1, positive.sum()))})
            best = max(range(2), key=lambda k: clusters[k]['precision_distinctive'] or 0)
            oracle = z_s[labels_s == distinctive].mean(axis=0)
            candidates['oracle_mean_plus_W_class_mean'] = model_states(oracle[None].astype(np.float32))[0]
            margin = np.linalg.norm(z_r - centers[1 - best], axis=1) - np.linalg.norm(z_r - centers[best], axis=1)
            block.update(clusters=clusters, distinctive_cluster=best,
                         purity=clusters[best]['precision_distinctive'], recall=clusters[best]['recall_distinctive'],
                         auroc_two_means_margin=auroc(margin, positive),
                         auroc_oracle_class_mean_distance=auroc(-np.linalg.norm(z_r - z_r[positive].mean(axis=0), axis=1), positive),
                         note='purity/recall: the 2-means cluster richer in the distinctive state; AUROC of the margin between the two centres; the oracle AUROC uses the true class mean')
        block['fsc'] = {name: fsc_to_target(v) for name, v in candidates.items()}
        block['best_label_free_map'] = max((n for n in block['fsc'] if not n.startswith('oracle')), key=lambda n: block['fsc'][n]['raw_mean_1_15'])
        block['best_label_free_fsc_raw_mean_1_15'] = block['fsc'][block['best_label_free_map']]['raw_mean_1_15']
        report['distinctive'] = block

    # 6. Poses in the registered mean frame.
    errors = pose_errors(np.asarray(rotation_grid)[rotation_idx].astype(np.float64), rotations_all[ids].astype(np.float64), common_fit.to_dict())
    labels_i = labels_all[ids]
    report['poses'] = {'overall': pose_metrics(errors), 'grid_spacing_deg_approx': 60 / 2 ** order,
                       'per_state': {int(k): pose_metrics(errors[labels_i == k]) for k in np.unique(labels_i)} if len(gt) <= 12 else
                       {int(distinctive): pose_metrics(errors[labels_i == distinctive]), 'others': pose_metrics(errors[labels_i != distinctive])}}

    # 7. Shifts (inferred grid offsets against the true simulator translations).
    if translation_idx is None:
        report['shifts'] = {'skipped': 'reused inference.npz predates translation indices; rescore without --reuse-inference'}
    else:
        tg = np.asarray(translation_grid, np.float64)
        true_t = translations_all[ids].astype(np.float64)
        floor = np.min(np.linalg.norm(true_t[:, None] - SHIFT_SIGN * tg[None], axis=-1), axis=1)
        block = shift_metrics(tg[translation_idx], true_t, np.asarray(rotation_grid)[rotation_idx], native_voxel, errors < 10,
                              labels_i if len(gt) > 12 else None, distinctive if len(gt) > 12 else None)
        block.update(definition='|inferred grid offset - %+d * true translation - P(R_inferred) c| over the inferred particles (stratified + random); '
                                'c = global 3D translation of the learned map (its frame), least trimmed squares on particles with pose error < 10 deg; raw = without removing c' % SHIFT_SIGN,
                     translation_grid=report['inference_translation_grid'],
                     grid_floor={'median_px': float(np.median(floor)), 'p90_px': float(np.quantile(floor, .9)),
                                 'note': 'distance of each true shift to the nearest grid offset: the best a perfect model can score'},
                     true_shift_radius_px={'median': float(np.median(np.linalg.norm(true_t, axis=1))), 'p90': float(np.quantile(np.linalg.norm(true_t, axis=1), .9))},
                     fraction_true_outside_grid=float(np.mean(np.linalg.norm(true_t, axis=1) > max_pixel)), voxel_size_ang=native_voxel)
        report['shifts'] = block
    report['elapsed_seconds'] = time.time() - started
    write(output / 'report.json', report)
    summary = {'dataset': report['dataset_id'], 'checkpoint': cp.name, 'q': q,
               'mean_fsc_to_true_mean': report['mean_vs_true_mean_fsc']['raw_mean_1_15'],
               'subspace_max_angle_deg': report.get('subspace', {}).get('max_angle_deg_vs_true_top_k'),
               'true_energy_captured': report.get('subspace', {}).get('true_energy_captured_fraction'),
               'state_fsc_mean': report['states_summary']['mean'], 'state_fsc_min': report['states_summary']['min'],
               'state_fsc_baseline_true_mean': float(np.mean(report['states_summary']['baseline_true_mean_per_state'])),
               'pose_median_deg': report['poses']['overall']['median_deg'], 'pose_frac_below_10deg': report['poses']['overall']['fraction_below_10_deg']}
    if 'raw' in report['shifts']:
        sh = report['shifts']
        summary.update(shift_err_median_px=sh['after_global']['median_px'], shift_err_p90_px=sh['after_global']['p90_px'],
                       shift_err_median_ang=sh['after_global']['median_ang'], shift_err_p90_ang=sh['after_global']['p90_ang'],
                       shift_global_translation_px=sh['global_translation_norm_px'], shift_grid_floor_median_px=sh['grid_floor']['median_px'])
    if q:
        summary.update(pc_rigid_share_1_15=report['rigid_absorption']['pooled_all_pcs']['1-15'],
                       pc_rigid_share_1_15_reference_true=report['rigid_absorption']['reference_true_components']['1-15'],
                       pc_max_fraction_outside_true_span=max(report['subspace']['per_learned_pc_fraction_outside_true_span']))
    if 'latent_clustering' in report:
        lc = report['latent_clustering']
        summary.update(kmeans_ari=lc['kmeans_ari'], kmeans_nmi=lc['kmeans_nmi'], within_over_between=lc['within_over_between'],
                       state_mean_offset_over_between=lc['state_mean_position_vs_learned_mean']['mean_distance_over_true_between'],
                       state_mean_offset_after_scale=lc['state_mean_position_vs_learned_mean']['mean_distance_over_true_between_after_scale'],
                       state_mean_shrink_scale=lc['state_mean_position_vs_learned_mean']['best_scale'])
    if distinctive is not None:
        summary.update(distinctive_best_map=report['distinctive']['best_label_free_map'], distinctive_best_fsc=report['distinctive']['best_label_free_fsc_raw_mean_1_15'],
                       distinctive_auroc=report['distinctive'].get('auroc_two_means_margin'), distinctive_purity=report['distinctive'].get('purity'),
                       distinctive_recall=report['distinctive'].get('recall'))
    write(output / 'summary.json', summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
