"""Round 17 copy (owner design, TASK.md step 1): adds --tau2-file, the per-shell PC shrink W <- W / (1 + lr (b/N) / (c tau2)) after the SGD step, and the v(s) log (v_log.jsonl, logged only).
One acceptance arm of the gated momentum SGD feature on the study protocol, with relax's own update rules.

python ws2_arm.py --source SRC --manifest M --eval-ids IDS --output OUT --seed S --stop-after 3000 -- RELAX_ARGS...

RELAX_ARGS are `relax ppca_initial_model` optimizer flags (--optimizer, --sgd-variant, --sgd-gate, --sgd-step-scale,
--noise-anneal, --sgd-learning-rate), resolved by the command's own parser and resolve_sgd_options. Nothing of the
update is patched. The harness is the study's (improve_20261004/spa_opt9.py): the recorded protocol (q=2, iterations
6000, radius 31 / HEALPix 3 from update 1, balanced 150+150 batches on every update, shift step 1 px), 600 held-out
particles never drawn for training (the batch selection is filtered, as in the study), and an evaluation E-step at
noise factor 1 on the evaluation and held-out particles every 250 updates -> inference_NNNN.npz for spa_score6.py and
axis_metrics.py.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RECOVAR_EM_XLA_DEFAULTS", "1")


GRID_BAND_DIVISOR = {"relion": 2.4, "fit": 3.0}  # resolution = D * step_rad / divisor ("fit": owner, 2026-10-28, snapped-pose sweep)


def grid_band_radius(order, diameter_ang, grid_size, voxel_ang, divisor=2.4):
    """Owner rule (PLAN.md "Band limit tied to the angular grid", 2026-10-28): RELION's coarse-pass resolution for an
    angular step, (step_deg / 360) * pi * D / 1.2 Angstrom, as a whole shell rounded down (shell k = grid * voxel / k
    Angstrom), capped at Nyquist - 1. step_deg = 360 / (6 * 2^order) (relax healpix_angular_step)."""
    import math
    step = 360.0 / (6 * 2 ** order)
    res = diameter_ang * math.radians(step) / divisor  # relion: (step/360) pi D / 1.2 = D step_rad / 2.4
    return min(int(math.floor(grid_size * voxel_ang / res)), grid_size // 2 - 1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    argv = sys.argv[1:]
    relax_argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--eval-ids", type=Path, default=None, help="omit for suite mode: no held-out set, no inference")
    p.add_argument("--shift-step", type=float, default=1.0)
    # Coupled-problem round 1 (2026-10-26): search range in px; default the manifest's shift_range_px.
    p.add_argument("--shift-range", type=float, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stop-after", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--iterations", type=int, default=6000)
    p.add_argument("--q", type=int, default=2)
    p.add_argument("--heldout", type=int, default=600)
    # Gate-free prototypes (patched here, not in relax): PC band limit while the update is <= --pc-until, and
    # the damped latent re-whitening of spa_opt7.py from the batch's posterior embeddings.
    p.add_argument("--pc-radius", type=float, default=None)
    p.add_argument("--pc-until", type=int, default=10**9)
    p.add_argument("--rewhiten", type=float, default=0.0)
    # Resolution-tied oversampling: with --oversampling 1, oversample the updates whose stage radius exceeds this
    # (instead of relax's "last stage only").
    p.add_argument("--os-from-radius", type=float, default=None)
    # Posterior-tied oversampling: with --oversampling 1, start oversampling at the update after the mean training
    # Pmax over the last --os-pmax-window updates first reaches --os-pmax, or, with --os-pmax-plateau D, after it is
    # above --os-pmax and rose less than D over the last 100 updates. The onset goes to os_onset.json.
    p.add_argument("--os-pmax", type=float, default=None)
    # Diagnosis: keep the rotation prior uniform (drop the learned direction prior after every update).
    p.add_argument("--uniform-direction-prior", action="store_true")
    # Diagnosis: oversample rotations only (each coarse translation is its own single child, no half-step shifts).
    p.add_argument("--os-rotations-only", action="store_true")
    p.add_argument("--os-pmax-window", type=int, default=50)
    p.add_argument("--os-pmax-plateau", type=float, default=None)
    # RELION's per-particle offsets: each image is pre-shifted by its stored offset (the search grid, and the os1
    # children, are centred on it), the Gaussian offset prior is evaluated on the total offset (stored + coarse
    # translation, as RELION's calculatePdfOffset), the stored offset becomes the MAP total offset after every visit,
    # and the offset variance is estimated from total offsets.
    p.add_argument("--recentre", action="store_true")
    # PC band cap: loadings band-limited to 3/4 of the shell the angular sampling supports (RELION's coarse-size
    # formula with the manifest's particle diameter), HEALPix order before os1 and order + oversampling after.
    p.add_argument("--pc-cap-auto", action="store_true")
    p.add_argument("--pc-cap-fraction", type=float, default=0.75)
    # Noise annealing with an auto-tuned start: factor F0 chosen at update 0 so the mean per-image rotation ESS is
    # this fraction of the grid, held to update HOLD, then linear to 1 over DECAY updates: "FRACTION,HOLD,DECAY".
    p.add_argument("--noise-ess", default=None)
    # RELION/relax VDAM InitialModel sampling (relax.vdam.native_sampling): HEALPix from --vdam-hp0 with oversampling 1
    # from update 1, the --auto_sampling --grad update every 10 updates on a resolution stall (HEALPix +1 up to 3, offset
    # step/range by RELION's formulas, uniform orientation prior at the ceiling), expected accuracy (acc_rot,
    # acc_trans) on 100 batch particles at update 1 and every 10. Implies --recentre. The PPCA resolution is the last
    # shell whose SSNR proxy |mu|^2 W_mu >= 1 (W_mu = the batch's mean-channel metric).
    p.add_argument("--vdam-sampling", action="store_true")
    # Black-box ab-initio schedule (job defaults + its run logs on suite A), iterations = updates (same minibatch):
    # radius at 35 A to update 200, then +0.04 shell per update to 12 A (capped at Nyquist - 1); minibatch 90 to update
    # 300, then 300; learning rate 0.4 / 0.2 / 0.1 (updates 1-100 / 101-500 / later) in its units, ours = (1 + q) x
    # theirs (our scale divides by the trace of the (1 + q) block); noise: the --noise-ess factor F0 held to update
    # 200, then blended with the estimate with weights 5000 x 0.9999^t (initial) against 50 + images seen since 200.
    p.add_argument("--bb-schedule", action="store_true")
    # Which parts of --bb-schedule are on (default all): radius (35 A start and ramp), batch (minibatch 90 then 300),
    # lr (the stepped learning rate; off = --sgd-learning-rate throughout), noise (the blend of --noise-ess; off =
    # --noise-ess is optional and, if given, follows its own HOLD,DECAY).
    p.add_argument("--bb-parts", default="radius,batch,lr,noise")
    p.add_argument("--vdam-hp0", type=int, default=1)
    # Owner's tied sampling (2026-10-07), with --vdam-sampling: the coarse translation step in Angstrom is the arc one
    # coarse angular step moves the particle edge, RELION's rotated_distance = (step_deg / 360) pi D, and it follows
    # the HEALPix order; RELION's acc_trans step rule is dropped, the range keeps RELION's 5x-movement rule.
    p.add_argument("--tied-step", action="store_true")
    # K5 (owner-approved prototype, 2026-10-26): PCs held at zero (model and momentum, every update) from update 1,
    # so only the mean is fitted (momentum SGD or, since 2026-10-29, VDAM), until the trigger: the update after the rolling (--os-pmax-window) training Pmax
    # first reaches --pc-hold-pmax, or update --pc-hold-until, whichever comes first. At the trigger the PCs are set
    # back to the run's own update-0 initial values (same seed) with zero momentum and then train normally (W = 0 is
    # a stationary point: E[z] = 0, so the W gradient is 0 and zero PCs would never grow). Trigger -> pc_hold.json,
    # per-update PC norm -> pc_hold.jsonl.
    p.add_argument("--pc-hold", action="store_true")
    # Arm V (owner, 2026-10-26): momentum coefficient B of relax's momentum_step, velocity = B previous +
    # (1 - B) lr gradient / curvature (relax fixes B = 0.9). B = 0 is plain SGD, velocity = lr gradient / curvature;
    # the curvature scaling is unchanged. Implemented by calling the stock step with previous x B/0.9 and
    # lr x (1 - B)/0.1. Omitted = the stock step, untouched.
    p.add_argument("--momentum", type=float, default=None)
    # Round 2 (C SGD, 2026-10-27). --batch: images per update (relax's stochastic_batch_size; default 300).
    # --milestones: comma list of updates to checkpoint at (replaces every --eval-every; the relax checkpoint interval
    # stays --eval-every). --init-pc-power F: rescale the initial PCs so that the summed power of all PCs is F times the
    # mean's power (mean unchanged). --init-theta FILE.npy: replace the initial model (mean and PCs, relax half-Fourier
    # layout, complex64 (N, q+1)) with FILE. Both rewrite checkpoint_0000.npz before update 1 (momentum stays zero).
    p.add_argument("--batch", type=int, default=300)
    p.add_argument("--milestones", default=None)
    p.add_argument("--init-pc-power", type=float, default=None)
    p.add_argument("--init-theta", type=Path, default=None)
    # Hybrid VDAM -> SGD (owner, 2026-10-28): --init-theta may be a checkpoint .npz (its theta); --init-raw keeps that
    # theta as it is (no rescale to this run's own start, which --init-theta otherwise applies for S3);
    # --init-carry noise,direction_prior also copies those arrays from the same .npz into this run's start.
    p.add_argument("--init-raw", action="store_true")
    p.add_argument("--init-carry", default="")
    # --resume: continue an interrupted run from its newest checkpoint (pool time ran out). Only for arms without
    # driver-side state between updates (no --os-pmax, --pc-hold, --recentre, --noise-ess, --vdam-sampling, --bb-schedule).
    p.add_argument("--resume", action="store_true")
    # S6 (owner, 2026-10-27): with --oversampling 1, oversample every update from this one on, whatever the stage (relax
    # oversamples only in the last stage; S6's radius ramp continues past the os1 onset).
    p.add_argument("--os-from-update", type=int, default=None)
    # Grid-tied band limit (owner, 2026-10-28): --grid-band relion sets every update's radius from its HEALPix order
    # (plus the oversampling order while os1 is on) by grid_band_radius(); the --stages radii are ignored. D from
    # --band-diameter: manifest (particle_diameter_ang), halfbox (box px x pixel A / 2) or a number in Angstrom. D is
    # used only for this band, never for masks. D and every stage's band go to grid_band.json and driver_log.jsonl.
    # New plain-SGD default (owner, 2026-10-28): --sgd-curv split replaces momentum_step's single trace-max scalar by
    # c_mean = max over active f of H_f[0,0] for the mean channel and c_PC = max over active f of tr(H_f[1:,1:]) / q for
    # all PC channels (one scalar keeps the step invariant to rotating the PCs), with no momentum (velocity = step):
    # theta += lr * G / c. Needs the packed metric (the stream is made to keep it, as the half-set gate does).
    # --sgd-lr-bb NIT: lr 0.4 for updates <= 100, 0.2 until NIT - 500, then 0.1 (the K-class ab initio's "cs" rule).
    # VDAM knock-outs (owner ablation, 2026-10-28), one at a time, by patching relax's update functions in this process:
    # nogate (phi = 1), gatepconly (phi = 1 for the mean), gatemeanonly (phi = 1 for the PCs), noadapt (v = 1),
    # scalardir (each half's direction = G / that half's trace-max curvature, times one factor k fixed at update 1 so
    # its norm equals the coupled direction's there; gate, moments and schedule unchanged), flatsched (step and fudge
    # held at their update-1 values). Every update logs phi per shell (mean, PCs), mean v (mean, PCs), step, fudge.
    p.add_argument("--vdam-ko", choices=["nogate", "gatepconly", "gatemeanonly", "noadapt", "scalardir", "flatsched", "noadapt_noavg"], default=None)
    p.add_argument("--sgd-curv", choices=["split"], default=None)
    # Round 14 (owner, 2026-10-30): VDAM φ / fudge knobs, used with --vdam-ko noadapt only; each goes to arm.json (argv).
    # --fudge-const F: fudge held at F for every update (RELION's tau2_fudge without its 1 -> 4 sigmoid).
    # --fudge-ramp-start U: the stock fudge sigmoid moved earlier in time so that it leaves 1 (exceeds 1.01) at update U;
    #   the step schedule is unchanged.
    # --phi-pc-floor X: φ on the PCs clamped to >= X in every shell (the mean's φ unchanged).
    # --phi-pc-one-after U: φ on the PCs fixed at 1 for updates > U (the mean's φ unchanged); used with --pc-hold.
    p.add_argument("--fudge-const", type=float, default=None)
    p.add_argument("--fudge-ramp-start", type=int, default=None)
    p.add_argument("--phi-pc-floor", type=float, default=None)
    p.add_argument("--phi-pc-one-after", type=int, default=None)
    # --fudge-split (lead 2026-10-31): φ on the mean uses fudge 1 in its ρ; φ on the PCs uses --fudge-pc-const (the
    #   schedule's own fudge if omitted). Both gates come from relax's shell_gates, called once per fudge.
    p.add_argument("--fudge-split", action="store_true")
    # Round 16 (owner, 2026-10-31): SGD step-size schedules, momentum-0 SGD only (--momentum 0). lr(t) replaces the
    # constant --sgd-learning-rate (the multiplier on G / c):
    #   const:X | step:LR,T1,T2,F (LR, times F after T1 and again after T2) | rm:LR,T0 (LR to T0, then LR*T0/t)
    #   | cos:LR,LR1,T0,T1 (LR to T0, cosine to LR1 at T1, then LR1) | warm:LR0,LR,T (linear LR0 -> LR over 1..T)
    #   | early:LR0,T,LR (LR0 to T, then LR)
    # --pc-lr-mult M / --pc-lr-ramp M0,M1,T: the PC columns' step times M (or M0 -> M1 linearly over 1..T); mean unchanged.
    # --curv-per-shell: preconditioner change: c_f = max of tr H over the active frequencies of f's shell (floored as
    #   the stock step), instead of one global c = max_f tr H_f.
    # Every update logs lr_mean_used, lr_pc_used (and curv_per_shell) in iterations.jsonl.
    p.add_argument("--lr-schedule", default=None)
    p.add_argument("--pc-lr-mult", type=float, default=None)
    p.add_argument("--pc-lr-ramp", default=None)
    p.add_argument("--curv-per-shell", action="store_true")
    # --adaptive-eta LRMAX (lead 2026-11-01): per-shell step factor eta_s = clip(rho_s, 0.05, 1) on top of --curv-per-shell,
    #   rho_s = EMA(Re<g0, g1>_s) / sqrt(EMA(|g0|^2_s) EMA(|g1|^2_s)) from the two half-batch gradients (all channels of the
    #   shell; EMA weight 1/20 per update, so about 20 updates), = SNR_s / (1 + SNR_s) when g_h = signal + independent noise.
    #   Step = LRMAX * eta_s * G / c_s. eta per shell is logged every update ("eta_shell").
    p.add_argument("--adaptive-eta", type=float, default=None)
    # --log-trh-shell U1,U2,... (lead 2026-11-01, logging only): at those updates log per-shell mean and max of tr H_f over
    #   the active frequencies ("trh_shell_mean", "trh_shell_max", shell 0 first) and the curvature c the step divides by
    #   ("trh_c": the global scalar, or per shell with --curv-per-shell). Effective step per shell = lr * trh_shell / c.
    p.add_argument("--log-trh-shell", default=None)
    p.add_argument("--fudge-pc-const", type=float, default=None)
    p.add_argument("--sgd-lr-bb", type=int, default=None)
    p.add_argument("--grid-band", choices=["relion", "fit"], default=None)
    p.add_argument("--band-diameter", default="manifest")
    p.add_argument("--pc-hold-pmax", type=float, default=0.75)
    p.add_argument("--pc-hold-until", type=int, default=1001)
    p.add_argument("--tau2-file", type=Path, default=None, help="round 17: per-shell tau2 (npy, index = rounded shell); shrink and v log")
    a = p.parse_args(argv)
    a.recentre = a.recentre or a.vdam_sampling
    bb_parts = set(a.bb_parts.split(",")) - {"none"} if a.bb_schedule else set()  # "none": no part, lr still logged
    if bb_parts - {"radius", "batch", "lr", "noise"}:
        raise ValueError(f"unknown --bb-parts {a.bb_parts}")
    if a.resume:
        if a.os_pmax is not None or a.pc_hold or a.recentre or a.noise_ess or a.vdam_sampling or a.bb_schedule:
            raise ValueError("--resume does not restore driver-side state for this arm")
        if not (a.output / "checkpoint_0000.npz").exists():
            raise FileNotFoundError(a.output / "checkpoint_0000.npz")
    elif a.output.exists():
        raise FileExistsError(a.output)
    sys.path.insert(0, str(a.source.resolve()))
    import dataclasses
    import functools

    import jax
    import jax.numpy as jnp
    import numpy as np
    import relax
    from relax import sampling
    from relax.helpers.convergence import healpix_angular_step
    from relax.commands.ppca_initial_model import add_args, load_training, resolve_sgd_options
    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config

    if not Path(relax.__file__).resolve().is_relative_to(a.source.resolve()):
        raise RuntimeError(f"relax imported from {relax.__file__}, not {a.source}")
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(["manifest.json", "-o", str(a.output), *relax_argv])
    config = Config(
        q=a.q,
        seed=a.seed,
        iterations=a.iterations,
        stages=tuple(tuple(x) for x in json.loads(args.stages)) if args.stages else ((1, 31, 3),),
        oversampling=args.oversampling,
        max_significant=args.maxsig,
        stochastic_batch_size=a.batch,
        stochastic_all_iterations=True,
        balanced_stochastic_halves=True,
        checkpoint_interval=a.eval_every,
        skip_final_embeddings=True,
        shift_step=a.shift_step,
        # The manifest's search range (6 px on the fixture and the suite, 12 px on EMPIAR-10076 at box 128).
        shift_range=a.shift_range if a.shift_range is not None else float(json.loads(a.manifest.read_text()).get("shift_range_px", 6.0)),
        optimizer=args.optimizer,
        **resolve_sgd_options(args),
    )
    data, manifest, identity = load_training(a.manifest, "on")
    identity["experiment"] = {"argv": sys.argv[1:], "script_sha256": sha256(__file__)}
    if a.resume:  # checkpoints carry the original run's experiment identity (argv, driver sha); keep it so they load
        with np.load(a.output / "checkpoint_0000.npz", allow_pickle=False) as saved:
            identity["experiment"] = json.loads(str(saved["metadata"]))["identity"]["experiment"]
    n_images = data.n_images
    suite = a.eval_ids is None
    eval_ids = np.zeros(0, np.int64) if suite else np.load(a.eval_ids, allow_pickle=False).astype(np.int64)
    pool = np.random.default_rng(20261003).permutation(np.setdiff1d(np.arange(n_images), eval_ids))
    held = (
        np.zeros(0, np.int64)
        if suite
        else np.sort(np.concatenate([pool[pool % 2 == k][: a.heldout // 2] for k in (0, 1)]))
    )
    select_halves = iteration_loop._select_halves

    def training_halves(rng, order, count, balanced):
        return select_halves(rng, order[~np.isin(order, held)], count, balanced)

    iteration_loop._select_halves = training_halves
    if a.os_rotations_only:
        stock_children = iteration_loop.relion_child_grids

        def rotation_children_only(order, translations, translation_step):
            rotations, _ = stock_children(order, translations, translation_step)
            parents = np.array(translations, np.float32)
            parents.setflags(write=False)
            return rotations, parents

        iteration_loop.relion_child_grids = rotation_children_only
    if a.uniform_direction_prior:
        stock_state = iteration_loop.State

        def state_without_prior(*fields, **named):
            fields = list(fields)
            if len(fields) > 9:
                fields[9] = None  # State.direction_prior
            named.pop("direction_prior", None)
            return stock_state(*fields, **named)

        iteration_loop.State = state_without_prior
    band = {}
    if a.grid_band:
        if a.os_from_radius is not None or a.bb_schedule or a.vdam_sampling:
            raise ValueError("--grid-band cannot be combined with --os-from-radius, --bb-schedule or --vdam-sampling")
        D = {"manifest": None, "halfbox": data.grid_size * data.voxel_size / 2}.get(a.band_diameter, None)
        if a.band_diameter == "manifest":
            D = float(manifest["particle_diameter_ang"])
        elif D is None:
            D = float(a.band_diameter)
        stock_band_stage = Config.stage

        def band_stage(self, t):
            _, hp = stock_band_stage(self, t)
            os_on = bool(iteration_loop.oversampled_update(self, t))
            return (grid_band_radius(hp + (self.oversampling if os_on else 0), D, data.grid_size, data.voxel_size,
                                     GRID_BAND_DIVISOR[a.grid_band]), hp)

        Config.stage = band_stage
        band = {"rule": a.grid_band, "divisor": GRID_BAND_DIVISOR[a.grid_band], "diameter_ang": D, "source": a.band_diameter, "stages": [
            {"start": st[0], "healpix_order": st[2], "os_on": bool(iteration_loop.oversampled_update(config, st[0])),
             "radius": config.stage(st[0])[0]} for st in config.stages]}
    if a.sgd_curv or a.sgd_lr_bb:
        if config.optimizer != "momentum_sgd":
            raise ValueError("--sgd-curv/--sgd-lr-bb need momentum_sgd")
        from relax.ppca_initial_model import sgd_update as _sgdu
        import jax.numpy as jnp
        stock_prepare = iteration_loop.prepare_full_row_stream

        def prepare_keep_metric(*args_, **kw):
            if a.sgd_curv:
                kw["metric_trace_only"] = False  # keep the packed metric: its diagonal is needed
            return stock_prepare(*args_, **kw)

        iteration_loop.prepare_full_row_stream = prepare_keep_metric
        stash = {}
        stock_trace = iteration_loop._curvature_trace

        def trace_and_stash(stats, p_, *, exact):
            if a.sgd_curv:
                stash["diag"] = _sgdu.metric_diagonal(sum(s_.lhs_tri for s_ in stats), p_)
            return stock_trace(stats, p_, exact=exact)

        iteration_loop._curvature_trace = trace_and_stash
        stock_step = iteration_loop.momentum_step

        def lr_now(lr):
            if not a.sgd_lr_bb:
                return lr
            t = clock["update"]
            return 0.4 if t <= 100 else (0.2 if t <= a.sgd_lr_bb - 500 else 0.1)

        def split_step(theta, previous, gradient, trace, active, *, learning_rate, floor):
            learning_rate = lr_now(learning_rate)
            clock["lr"] = float(learning_rate)
            if not a.sgd_curv:
                return stock_step(theta, previous, gradient, trace, active, learning_rate=learning_rate, floor=floor)
            active = jnp.asarray(active, bool)
            diag = jnp.asarray(stash["diag"])
            q_ = diag.shape[1] - 1
            c_mean = jnp.maximum(jnp.max(jnp.where(active, diag[:, 0], 0.0)), jnp.float32(floor))
            c_pc = jnp.maximum(jnp.max(jnp.where(active, jnp.sum(diag[:, 1:], axis=1) / q_, 0.0)), jnp.float32(floor))
            trace_max = jnp.max(jnp.where(active, jnp.asarray(trace), 0.0))
            scale = jnp.concatenate([c_mean[None], jnp.repeat(c_pc, q_)]).astype(jnp.float32)
            g = jnp.where(active[:, None], jnp.asarray(gradient), 0)
            velocity = jnp.where(trace_max > 0, jnp.float32(learning_rate) * g / scale[None, :], jnp.zeros_like(g)).astype(theta.dtype)
            diagnostics = {"curvature_trace_max": float(trace_max), "curvature_scalar": float(trace_max),
                           "c_mean": float(c_mean), "c_pc": float(c_pc), "trace_over_c_mean": float(trace_max / c_mean),
                           "trace_over_c_pc": float(trace_max / c_pc), "learning_rate_used": float(learning_rate),
                           "gradient_l2": float(jnp.linalg.norm(gradient)), "velocity_l2_before_projection": float(jnp.linalg.norm(velocity)),
                           "active_frequency_count": int(jnp.sum(active))}
            return theta + velocity, velocity, diagnostics

        iteration_loop.momentum_step = split_step
    if a.vdam_ko:
        if config.optimizer != "vdam":
            raise ValueError("--vdam-ko needs --optimizer vdam")
        import jax.numpy as jnp
        from relax.ppca_initial_model import update as _upd
        ko = a.vdam_ko

        @_upd.full_float32
        def stochastic_update_ko(theta, moments, directions, coverage, shells, *, step, fudge, image_size):
            # relax update.stochastic_update with the knock-out applied (same formulas otherwise).
            theta = jnp.asarray(theta)
            moments = _upd.advance_moments(moments, directions, coverage, image_size=image_size)
            first, second = moments.first, moments.second
            if ko == "noadapt_noavg":  # owner 2026-10-29: no half-set averaging, m_h = this batch's d_h (phi from per-batch disagreement)
                first = jnp.asarray(directions)
            active = jnp.any(jnp.asarray(coverage, bool), axis=0)
            mean_first = (first[0] + first[1]) * 0.5
            divisor = (jnp.ones_like(second) if ko in ("noadapt", "noadapt_noavg") else jnp.sqrt(second)) + 1e-12
            adapted = jnp.concatenate([mean_first[:, :1] / divisor[:, :1], mean_first[:, 1:] / divisor[:, 1:]], axis=-1)
            gates = _upd.shell_gates(theta, first, active, shells, fudge=fudge)
            gate = gates.per_frequency(shells, theta.shape[-1] - 1)
            if a.fudge_split:
                g_mean = _upd.shell_gates(theta, first, active, shells, fudge=1.0)
                g_pc = _upd.shell_gates(theta, first, active, shells, fudge=a.fudge_pc_const if a.fudge_pc_const is not None else fudge)
                gate = jnp.concatenate([g_mean.per_frequency(shells, theta.shape[-1] - 1)[:, :1], g_pc.per_frequency(shells, theta.shape[-1] - 1)[:, 1:]], axis=-1)
                gates = g_pc
            if ko == "nogate":
                gate = jnp.ones_like(gate)
            elif ko == "gatepconly":
                gate = gate.at[:, 0].set(1.0)
            elif ko == "gatemeanonly":
                gate = gate.at[:, 1:].set(1.0)
            if a.phi_pc_floor is not None:
                gate = gate.at[:, 1:].set(jnp.maximum(gate[:, 1:], a.phi_pc_floor))
            if a.phi_pc_one_after is not None and clock["update"] > a.phi_pc_one_after:
                gate = gate.at[:, 1:].set(1.0)
            updated = jnp.where(active[:, None], theta + step * (gate * adapted - (1 - gate) * theta), theta)
            diag = gates.diagnostics()
            act = np.asarray(active)
            sec = np.asarray(second)[act]
            g_sh = np.asarray(gates.gates).copy()  # (n_shells, 2): the gate as applied, per shell (mean, PCs)
            if a.fudge_split:
                g_sh[:, 0] = np.asarray(g_mean.gates)[:, 0]
            if ko == "nogate":
                g_sh[:] = 1.0
            elif ko == "gatepconly":
                g_sh[:, 0] = 1.0
            elif ko == "gatemeanonly":
                g_sh[:, 1] = 1.0
            if a.phi_pc_floor is not None:
                g_sh[:, 1] = np.maximum(g_sh[:, 1], a.phi_pc_floor)
            if a.phi_pc_one_after is not None and clock["update"] > a.phi_pc_one_after:
                g_sh[:, 1] = 1.0
            diag["gates_applied"] = g_sh
            diag.update(ko=ko, v_mean=[float(sec[:, 0].mean()), float(sec[:, 1].mean())] if sec.size else None,
                        step_used=float(step), fudge_used=float(fudge))
            return updated, moments, diag

        iteration_loop.stochastic_update = stochastic_update_ko
        if ko == "scalardir":
            stock_dir = iteration_loop.coupled_direction
            kfac = {}

            def scalar_direction(lhs_tri, residual_gradient, support, *, floor):
                coupled, info = stock_dir(lhs_tri, residual_gradient, support, floor=floor)
                sup = jnp.asarray(np.asarray(support, bool))
                tr = _sgd_trace(lhs_tri, residual_gradient.shape[-1])
                c = jnp.maximum(jnp.max(jnp.where(sup, tr, 0.0)), jnp.float32(floor))
                g = jnp.where(sup[:, None], jnp.asarray(residual_gradient), 0) / c
                if "k" not in kfac:  # first call (update 1, half 0): match the coupled direction's norm
                    kfac["k"] = float(jnp.linalg.norm(coupled) / jnp.maximum(jnp.linalg.norm(g), 1e-30))
                    (a.output / "scalardir_k.json").write_text(json.dumps({"k": kfac["k"], "c": float(c)}) + "\n")
                info = dict(info, scalardir_k=kfac["k"], scalardir_c=float(c))
                return (kfac["k"] * g).astype(coupled.dtype), info

            from relax.ppca_initial_model.sgd_update import metric_trace as _sgd_trace
            iteration_loop.coupled_direction = scalar_direction
        if ko == "flatsched":
            stock_sched, stock_factor = Config.schedule, Config.step_factor
            Config.schedule = lambda self, t, n: (stock_sched(self, t, n)[0], *stock_sched(self, 1, n)[1:])
            Config.step_factor = lambda self, t, n: stock_factor(self, 1, n)
    if a.fudge_const is not None or a.fudge_ramp_start is not None:
        if a.vdam_ko != "noadapt":
            raise ValueError("round-14 fudge knobs are for --vdam-ko noadapt")
        stock_sched14 = Config.schedule
        if a.fudge_const is not None:
            Config.schedule = lambda self, t, n: (*stock_sched14(self, t, n)[:2], float(a.fudge_const))
        else:
            n0 = data.n_images
            start = next(t for t in range(1, config.iterations + 1) if stock_sched14(config, t, n0)[2] > 1.01)
            fudge_dt = start - a.fudge_ramp_start
            print("FUDGE_SHIFT", json.dumps({"stock_ramp_start": start, "ramp_start": a.fudge_ramp_start, "fudge_dt": fudge_dt}), flush=True)  # every update logs its fudge in iterations.jsonl
            Config.schedule = lambda self, t, n: (*stock_sched14(self, t, n)[:2], stock_sched14(self, min(max(t + fudge_dt, 1), self.iterations), n)[2])
    if a.os_from_update is not None:
        if not config.oversampling:
            raise ValueError("--os-from-update needs --oversampling 1")
        iteration_loop.oversampled_update = lambda config_, iteration: iteration >= a.os_from_update
    if a.os_from_radius is not None:
        if not config.oversampling:
            raise ValueError("--os-from-radius needs --oversampling 1")
        iteration_loop.oversampled_update = lambda config_, iteration: config_.stage(iteration)[0] > a.os_from_radius
    clock = {"update": 0}  # the update being computed (set per update by the patched expectation_groups)
    batch = {}
    onset = {"update": None, "pmax": []}
    if a.os_pmax is not None:
        if not config.oversampling or len(config.stages) != 1:
            raise ValueError("--os-pmax needs --oversampling 1 and a single stage")
        iteration_loop.oversampled_update = lambda config_, iteration: onset["update"] is not None and iteration >= onset["update"]

    def watch_pmax(iteration, stats):
        if a.os_pmax is None or onset["update"] is not None or batch.get("infer"):
            return
        n = sum(s.n_images for s in stats)
        onset["pmax"].append(sum(s.diagnostics["pmax_mean"] * s.n_images for s in stats) / n)
        w = a.os_pmax_window
        rolling = np.convolve(onset["pmax"], np.ones(w) / w, "valid") if len(onset["pmax"]) >= w else []
        if not len(rolling) or rolling[-1] < a.os_pmax:
            return
        if a.os_pmax_plateau is not None and (len(rolling) <= 100 or rolling[-1] - rolling[-101] >= a.os_pmax_plateau):
            return
        onset["update"] = iteration + 1
        (a.output / "os_onset.json").write_text(json.dumps({"onset": iteration + 1, "rolling_pmax": float(rolling[-1])}) + "\n")
    from recovar.core import fourier_transform_utils as ftu

    radii = np.asarray(ftu.get_grid_of_radial_distances_real(data.volume_shape, rounded=False)).reshape(-1)
    groups = iteration_loop.expectation_groups

    def tracked_groups(dataset, state, config_, halves, iteration, **kwargs):
        clock["update"] = iteration
        stats = groups(dataset, state, config_, halves, iteration, **kwargs)
        batch["stats"] = stats
        watch_pmax(iteration, stats)
        return stats

    iteration_loop.expectation_groups = tracked_groups
    from relax.helpers import preprocessing
    from relax.ppca_refinement import full_row_stream as frs
    from relax.ppca_refinement import oversampled_stream as osm

    # Per-tile stored offsets (B, 2) and real-image count, keyed by the tile's y_norm (kept alive with it); every
    # tile reader call registers its tile, every E-step clears the table.
    tile_old = {}
    reading = {"old": None}
    offsets = np.zeros((n_images, 2), np.float32)
    local_ids = np.arange(n_images)
    original_ids = np.asarray(data.original_image_indices_from_local(local_ids))
    local_of = np.empty(int(original_ids.max()) + 1, np.int64)
    local_of[original_ids] = local_ids
    shift = {"var": None, "coarse": None, "fine": None}
    probe = {"on": False, "ess": []}
    stock_load = frs._load_tile

    def load_registered(stream, image_indices, significant_rows, *, collect_observation):
        ids = np.asarray(image_indices)
        reading["old"] = offsets[ids] if a.recentre else None
        try:
            tile, power, layout = stock_load(stream, image_indices, significant_rows, collect_observation=collect_observation)
        finally:
            reading["old"] = None
        B = int(tile.y_norm.shape[0])
        old = np.zeros((B, 2), np.float32)
        old[: ids.size] = offsets[ids]
        tile_old[id(tile.y_norm)] = (tile.y_norm, jnp.asarray(old), ids.size)
        return tile, power, layout

    frs._load_tile = load_registered
    stock_half_inputs = preprocessing._dense_batch_half_inputs

    def half_inputs_recentred(experiment_dataset, batch_, ctf_params, noise_variance, translations, config_, *args, **kw):
        out = stock_half_inputs(experiment_dataset, batch_, ctf_params, noise_variance, translations, config_, *args, **kw)
        if reading["old"] is None:
            return out
        processed = out[0]
        old = np.zeros((processed.shape[0], 2), np.float32)
        old[: len(reading["old"])] = reading["old"]
        # The translation phases are linear in the shift: pre-shifting by o and scoring at d is scoring at o + d.
        phases = preprocessing.half_translation_phase_table(old, config_.image_shape).reshape(processed.shape)
        return (processed * phases.astype(processed.dtype), *out[1:])

    preprocessing._dense_batch_half_inputs = half_inputs_recentred

    @functools.partial(jax.jit, donate_argnums=(0,))
    def add_tile_prior(kept, old, translations, var):
        # Prior on the total offset o + t minus the shared prior on t: -(|o|^2 + 2 o.t) / (2 var), per (image, t).
        extra = -(jnp.sum(old**2, axis=1)[:, None] + 2 * old @ translations.T) / (2 * var)
        score = kept.score + extra[None]
        part_max = jnp.max(score, axis=2)
        finite = jnp.isfinite(part_max)
        part_sum = jnp.where(finite, jnp.sum(jnp.exp(score - jnp.where(finite, part_max, 0)[..., None]), axis=2), 0)
        return kept._replace(
            score=score, part_max=part_max, part_arg=jnp.argmax(score, axis=2).astype(jnp.int32), part_sum=part_sum
        )

    @jax.jit
    def rotation_ess(kept, posterior):
        log_norm = posterior.center + posterior.centered_logZ
        mass = jnp.where(jnp.isfinite(kept.part_max), kept.part_sum * jnp.exp(kept.part_max - log_norm[None]), 0)
        return 1.0 / jnp.sum(mass**2, axis=0)

    stock_score_tile = frs._score_tile

    def score_tile_recentred(stream, tile, n_blocks, kept=None):
        kept, posterior = stock_score_tile(stream, tile, n_blocks, kept)
        entry = tile_old.get(id(tile.y_norm))
        if a.recentre and entry is not None:
            kept = add_tile_prior(
                kept, entry[1], jnp.asarray(stream.translations, jnp.float32), jnp.float32(shift["var"])
            )
            posterior = frs._tile_posterior(kept, tile.rows, n_blocks, stream.rotation_block_size, stream.static)
        if probe["on"]:
            probe["ess"].append(np.asarray(rotation_ess(kept, posterior))[: entry[2]] / shift["n_rotations"])
        return kept, posterior

    frs._score_tile = score_tile_recentred
    osm._score_tile = score_tile_recentred

    @functools.partial(jax.jit, static_argnames=("chunk", "children"), donate_argnums=(0,))
    def add_job_prior(score, old, table, order, start, coarse, var, *, chunk, children):
        # Pass 2 carries the coarse parent's prior (as RELION): the same total-offset correction per job.
        slots = jax.lax.dynamic_slice_in_dim(order, start, chunk)
        o, t = old[table.image[slots]], coarse[table.translation[slots]]
        extra = -(jnp.sum(o**2, axis=1) + 2 * jnp.sum(o * t, axis=1)) / (2 * var)
        return score.at[slots].add(extra[:, None, None])

    stock_jobs = osm._score_jobs

    def score_jobs_recentred(results, arrays, tile, phases, table, order, start, *, chunk, **kw):
        results = stock_jobs(results, arrays, tile, phases, table, order, start, chunk=chunk, **kw)
        entry = tile_old.get(id(tile.y_norm))
        if a.recentre and entry is not None:
            score = add_job_prior(
                results.score, entry[1], table, order, start, jnp.asarray(shift["coarse"]), jnp.float32(shift["var"]),
                chunk=chunk, children=kw["translation_children"],
            )
            results = results._replace(score=score)
        return results

    score_jobs_recentred.lower = stock_jobs.lower  # the job planner sizes the stock program
    osm._score_jobs = score_jobs_recentred
    stock_expectation = iteration_loop._expectation

    def expectation_recentred(dataset, state, config_, ids, iteration, **kw):
        tile_old.clear()
        if a.vdam_sampling:
            config_ = dataclasses.replace(
                config_, shift_range=vdam["sampling"].offset_range_px, shift_step=vdam["sampling"].offset_step_px
            )
            if vdam["sampling"].uniform_local_orientation_prior:
                state = dataclasses.replace(state, direction_prior=None)
        radius, hp = config_.stage(iteration)
        shift["hp"] = hp
        shift["var"] = float(state.offset_variance)
        shift["n_rotations"] = sampling.rotation_grid_size(hp)
        shift["coarse"] = sampling.get_relion_translation_grid(
            max_pixel=config_.shift_range, pixel_offset=config_.shift_step
        ).astype(np.float32)
        shift["fine_rotations"], shift["fine"] = (
            np.asarray(x) for x in iteration_loop.relion_child_grids(hp, shift["coarse"], config_.shift_step)
        )
        try:
            return stock_expectation(dataset, state, config_, ids, iteration, **kw)
        finally:
            tile_old.clear()

    iteration_loop._expectation = expectation_recentred

    vdam = {}
    poses = np.zeros((n_images, 3, 3), np.float32)
    posed = np.zeros(n_images, bool)

    def recentre(stats):
        # RELION's stored offset: the MAP total offset (stored + best translation) of every image of the batch;
        # the offset variance is estimated from total offsets (E|o + d|^2 ~ |o|^2 + 2 o.d_MAP + E|d|^2).
        new, rotated = {}, {}
        for s in stats:
            oversampled = "oversampling" in s.diagnostics
            grid = shift["fine"] if oversampled else shift["coarse"]
            ids = local_of[np.asarray(s.original_image_ids)]
            best = np.asarray(s.diagnostics["best_translation_idx"])
            ok = best >= 0
            o, d = offsets[ids[ok]], grid[best[ok]]
            s.diagnostics["offset_second_sum_px2"] += float(np.sum(o * o) + 2 * np.sum(o * d))
            new.update(zip(ids[ok].tolist(), o + d))
            if a.vdam_sampling:
                rotations = shift["fine_rotations"] if oversampled else iteration_loop._rotation_grid(shift["hp"])
                r = np.asarray(s.diagnostics["best_rotation_idx"])
                rotated.update(zip(ids[r >= 0].tolist(), rotations[r[r >= 0]]))
        if new:
            ids = np.fromiter(new, np.int64)
            moved = np.stack(list(new.values())) - offsets[ids]
            # RELION's current_changes_optimal_offsets (Angstrom): sqrt(sum |change|^2 / (2 n)).
            vdam["offset_change_angstrom"] = float(np.sqrt(np.sum(moved**2) / (2 * ids.size))) * data.voxel_size
            offsets[ids] = np.stack(list(new.values()))
        if rotated:
            ids = np.fromiter(rotated, np.int64)
            poses[ids] = np.stack(list(rotated.values()))
            posed[ids] = True

    if a.recentre:
        stock_merge = iteration_loop._merge_statistics

        def merge_keeping_best(parts):
            merged = stock_merge(parts)
            for key in ("best_translation_idx", "best_rotation_idx"):
                merged.diagnostics[key] = np.concatenate([np.asarray(s.diagnostics[key]) for s in parts])
            return merged

        iteration_loop._merge_statistics = merge_keeping_best
        stock_tracked = iteration_loop.expectation_groups

        def tracked_recentred(dataset, state, config_, halves, iteration, **kwargs):
            if a.vdam_sampling:
                vdam_before(state, halves, iteration)
            stats = stock_tracked(dataset, state, config_, halves, iteration, **kwargs)
            recentre(stats)
            if a.vdam_sampling:
                vdam_after(state, stats, iteration)
            return stats

        iteration_loop.expectation_groups = tracked_recentred
    if a.vdam_sampling:
        import types

        from recovar.utils.helpers import R_to_relion
        from relax.helpers.expected_accuracy import estimate_relion_expected_accuracy
        from relax.helpers.resolution import resolution_from_data_vs_prior
        from relax.ppca_refinement.refinement_loop import _half_volume_to_full_flat as half_to_full
        from relax.vdam import native_sampling as ns
        from relax.vdam import schedules

        if config.stochastic_batch_size < 100 or len(config.stages) != 1 or not config.oversampling:
            raise ValueError("--vdam-sampling needs one stage (its radius), --oversampling 1 and batches of 100+")
        px = float(data.voxel_size)
        vdam["sampling"] = ns.NativeSamplingState(
            healpix_order=a.vdam_hp0,
            adaptive_oversampling=int(config.oversampling),
            offset_range_angstrom=config.shift_range * px,
            offset_step_angstrom=config.shift_step * px,
            offset_range_ori_angstrom=config.shift_range * px,
            offset_step_ori_angstrom=config.shift_step * px,
            pixel_size=px,
        )
        stock_stage = Config.stage
        Config.stage = lambda self, iteration: (stock_stage(self, iteration)[0], vdam["sampling"].healpix_order)

        def tied_step(order):
            return healpix_angular_step(order) / 360 * np.pi * manifest["particle_diameter_ang"]

        if a.tied_step:
            S0 = vdam["sampling"]
            S0.offset_step_angstrom = S0.offset_step_ori_angstrom = tied_step(S0.healpix_order)
            stock_sampling_update = ns._relion_update_native_sampling_state

            def tied_sampling_update(S, **kw):
                old_range = S.offset_range_angstrom
                changed = stock_sampling_update(S, **kw)
                if S.has_fine_enough_angular_sampling:
                    return changed
                # RELION's range rule (5x the offset changes, at most 1.3x the old, at least 1.5 steps, halved above
                # 4 steps) with the tied step; no step adjustment.
                step = tied_step(S.healpix_order)
                span = max(min(1.3 * old_range, 5.0 * S.current_changes_optimal_offsets_angstrom), 1.5 * step)
                if span > 4 * step:
                    span /= 2
                S.offset_step_angstrom, S.offset_range_angstrom = float(step), float(span)
                return True

            ns._relion_update_native_sampling_state = tied_sampling_update
        trace_log = a.output / "sampling_trace.jsonl"

        def log_sampling(iteration):
            S = vdam["sampling"]
            coarse_deg = healpix_angular_step(S.healpix_order)
            fine_deg = coarse_deg / 2 ** S.adaptive_oversampling
            record = {
                "update": iteration, "healpix_order": S.healpix_order, "angular_step_deg": coarse_deg,
                "translation_step_angstrom": S.offset_step_angstrom, "translation_step_px": S.offset_step_px,
                "translation_range_px": S.offset_range_px,
                # Supported resolution (RELION's coarse_res = rotated_distance / 1.2) of the angular step, and the
                # same for a translation step taken as the edge displacement.
                "angular_supported_res_angstrom": tied_step(S.healpix_order) / 1.2,
                "translation_supported_res_angstrom": S.offset_step_angstrom / 1.2,
                "fine_angular_step_deg": fine_deg,
                "pc_cap_shell": a.pc_cap_fraction * supported_shell(S.healpix_order + S.adaptive_oversampling),
            }
            with open(trace_log, "a") as f:
                f.write(json.dumps(record) + "\n")
        iteration_loop.oversampled_update = lambda config_, iteration: True
        shells_full = np.asarray(ftu.get_grid_of_radial_distances_real(data.volume_shape), np.int64).reshape(-1)
        vdam_log = a.output / "vdam_sampling.jsonl"

        def vdam_before(state, halves, iteration):
            S = vdam["sampling"]
            record = {"update": iteration}
            if schedules._should_estimate_native_sampling_accuracy(iteration=iteration, nr_iter=config.iterations, do_grad=True):
                batch_ids = np.concatenate([np.asarray(h) for h in halves])
                trials = batch_ids[posed[batch_ids]][:100]
                if trials.size:
                    eulers = np.zeros((n_images, 3))
                    eulers[trials] = np.asarray(R_to_relion(poses[trials].astype(np.float64), degrees=True))
                    noise = np.asarray(state.noise, np.float64).reshape(-1)
                    noise = np.pad(noise, (0, max(0, data.grid_size // 2 + 1 - noise.size)), mode="edge")
                    radius = min(config.stage(iteration)[0], data.grid_size // 2 - 1)
                    acc = estimate_relion_expected_accuracy(
                        reference_fourier=np.asarray(half_to_full(state.theta[:, 0], tuple(data.volume_shape)))[None],
                        volume_shape=tuple(data.volume_shape),
                        best_eulers_deg=eulers,
                        class_ids=np.zeros(n_images, np.int32),
                        class_weights=np.ones(1),
                        sigma2_noise_native=noise,
                        dataset=data,
                        trial_order_local=np.concatenate([trials, np.setdiff1d(np.arange(n_images), trials)]),
                        current_image_size=2 * radius,
                        padding_factor=1,
                        sigma2_fudge=1.0,
                        random_seed=a.seed,
                        max_trials=int(trials.size),
                    )
                    S.acc_rot, S.acc_trans_angstrom = float(acc.acc_rot), float(acc.acc_trans_angstrom)
                    record.update(acc_rot=S.acc_rot, acc_trans_angstrom=S.acc_trans_angstrom, n_trials=int(trials.size))
            before = (S.healpix_order, S.offset_step_angstrom, S.offset_range_angstrom)
            changed = ns._prepare_native_sampling_for_iteration(
                S, types.SimpleNamespace(nr_iter=config.iterations), iteration=iteration, do_grad=True
            )
            if len(record) > 1 or changed:
                record.update(
                    changed=bool(changed), before=before, healpix_order=S.healpix_order,
                    offset_step_px=S.offset_step_px, offset_range_px=S.offset_range_px,
                    uniform_prior=S.uniform_local_orientation_prior,
                    offset_change_angstrom=S.current_changes_optimal_offsets_angstrom,
                    resolution_shell=vdam.get("shell"),
                )
                with open(vdam_log, "a") as f:
                    f.write(json.dumps(record) + "\n")
            log_sampling(iteration)

        def vdam_after(state, stats, iteration):
            S = vdam["sampling"]
            if "offset_change_angstrom" in vdam:
                S.current_changes_optimal_offsets_angstrom = vdam["offset_change_angstrom"]
            # SSNR proxy per shell: |mu|^2 times the mean-channel metric W_mu ~ trace / (1 + E|z|^2) of this batch.
            z2 = np.mean(np.concatenate([np.sum(np.asarray(s.embeddings, np.float64) ** 2, axis=1) for s in stats]))
            z2 += np.mean([s.diagnostics["latent_covariance_trace_mean"] for s in stats])
            trace = np.asarray(iteration_loop._curvature_trace(stats, config.q + 1, exact=False), np.float64)
            power = np.abs(np.asarray(state.theta[:, 0])) ** 2 * trace.reshape(-1) / (1.0 + z2)
            n = data.grid_size // 2 + 1
            counts = np.bincount(shells_full, minlength=n)[:n]
            ssnr = np.bincount(shells_full, weights=power, minlength=n)[:n] / np.maximum(counts, 1)
            shell = int(resolution_from_data_vs_prior(ssnr, ori_size=data.grid_size, allow_high_res_recovery=True))
            vdam["shell"] = shell
            ns._record_native_sampling_post_iteration(
                S, types.SimpleNamespace(current_resolution=shell / (data.grid_size * px)), meta={}
            )

    # The PC cap shells: RELION's coarse-size formula (relax.helpers.resolution.compute_coarse_image_size, unrounded).
    def supported_shell(order):
        rotated = healpix_angular_step(order) / 360 * np.pi * manifest["particle_diameter_ang"]
        return data.voxel_size * data.grid_size / (rotated / 1.2)

    if a.bb_schedule:
        if "noise" in bb_parts and not a.noise_ess:
            raise ValueError("--bb-schedule needs --noise-ess (its initial noise factor)")
        r_init = data.voxel_size * data.grid_size / 35.0
        r_max = min(data.grid_size // 2 - 1, data.voxel_size * data.grid_size / 12.0)
        bb_stage, bb_schedule = Config.stage, Config.schedule
        if "radius" in bb_parts:
            Config.stage = lambda self, t: (
                int(round(min(r_max, r_init + 0.04 * max(0, t - 200)))), bb_stage(self, t)[1]
            )

        def schedule_bb(self, t, n):
            count, step, fudge = bb_schedule(self, t, n)
            return (min(90 if t <= 300 else 300, n), step, fudge)

        if "batch" in bb_parts:
            Config.schedule = schedule_bb
        stock_momentum = iteration_loop.momentum_step

        def momentum_bb(*args, learning_rate, **kw):
            t = clock["update"]
            if "lr" in bb_parts:
                learning_rate = (config.q + 1) * (0.4 if t <= 100 else 0.2 if t <= 500 else 0.1)
            clock["lr"] = float(learning_rate)  # the rate this update used, for driver_log.jsonl
            return stock_momentum(*args, learning_rate=learning_rate, **kw)

        iteration_loop.momentum_step = momentum_bb
    if a.momentum is not None:
        if config.optimizer != "momentum_sgd" or not 0 <= a.momentum < 1:
            raise ValueError("--momentum needs momentum_sgd and 0 <= B < 1")
        inner_momentum = iteration_loop.momentum_step

        def momentum_b(theta, previous, gradient, trace, active, *, learning_rate, floor):
            return inner_momentum(theta, previous * np.float32(a.momentum / 0.9), gradient, trace, active,
                                  learning_rate=learning_rate * (1.0 - a.momentum) / 0.1, floor=floor)

        iteration_loop.momentum_step = momentum_b
    if a.adaptive_eta is not None and not a.curv_per_shell:
        raise ValueError('--adaptive-eta needs --curv-per-shell')
    if a.lr_schedule or a.pc_lr_mult is not None or a.pc_lr_ramp or a.curv_per_shell or a.log_trh_shell:
        if a.momentum != 0 or a.sgd_curv:
            raise ValueError("round-16 step knobs are for --momentum 0 SGD without --sgd-curv")
        import math
        import jax.numpy as jnp
        base_lr = float(config.sgd_learning_rate)

        def lr_of(t):
            if not a.lr_schedule:
                return base_lr
            kind, _, par = a.lr_schedule.partition(":")
            v = [float(x) for x in par.split(",")]
            if kind == "const":
                return v[0]
            if kind == "step":
                return v[0] * v[3] ** ((t > v[1]) + (t > v[2]))
            if kind == "rm":
                return v[0] if t <= v[1] else v[0] * v[1] / t
            if kind == "cos":
                if t <= v[2]:
                    return v[0]
                x = min(1.0, (t - v[2]) / (v[3] - v[2]))
                return v[1] + (v[0] - v[1]) * 0.5 * (1 + math.cos(math.pi * x))
            if kind == "warm":
                return v[1] if t >= v[2] else v[0] + (v[1] - v[0]) * (t - 1) / (v[2] - 1)
            if kind == "early":
                return v[0] if t <= v[1] else v[2]
            raise ValueError(f"unknown --lr-schedule {a.lr_schedule}")

        def pc_mult_of(t):
            if a.pc_lr_ramp:
                m0, m1, T = (float(x) for x in a.pc_lr_ramp.split(","))
                return m1 if t >= T else m0 + (m1 - m0) * (t - 1) / (T - 1)
            return 1.0 if a.pc_lr_mult is None else a.pc_lr_mult

        lr_of.pc_mult_of = pc_mult_of
        for t_ in (1, 300, 500, 501, 1000, 1001, 2000, 2001, 3000):
            print(f"LR_SCHEDULE t={t_} lr_mean={lr_of(t_):.4f} lr_pc={lr_of(t_) * pc_mult_of(t_):.4f}", flush=True)
        inner_sched = iteration_loop.momentum_step
        shell_of = jnp.asarray(np.rint(radii).astype(np.int32))
        n_shells = int(np.rint(radii).max()) + 1

        ema = {"num": None, "d0": None, "d1": None}
        trh_log = {int(x) for x in a.log_trh_shell.split(",")} if a.log_trh_shell else set()

        def eta_now():
            st = batch.get("stats")
            if not st or len(st) != 2:
                return None
            g0, g1 = (jnp.asarray(x.residual_gradient) for x in st)
            num = jnp.zeros(n_shells, jnp.float32).at[shell_of].add(jnp.real(jnp.sum(jnp.conj(g0) * g1, axis=1)))
            d0 = jnp.zeros(n_shells, jnp.float32).at[shell_of].add(jnp.sum(jnp.abs(g0) ** 2, axis=1))
            d1 = jnp.zeros(n_shells, jnp.float32).at[shell_of].add(jnp.sum(jnp.abs(g1) ** 2, axis=1))
            w = jnp.float32(1 / 20)
            for k, v in (("num", num), ("d0", d0), ("d1", d1)):
                ema[k] = v if ema[k] is None else (1 - w) * ema[k] + w * v
            rho = ema["num"] / jnp.sqrt(jnp.maximum(ema["d0"] * ema["d1"], 1e-30))
            return jnp.clip(rho, 0.05, 1.0)

        def sched_step(theta, previous, gradient, trace, active, *, learning_rate, floor):
            t = clock["update"]
            lr = lr_of(t) if a.adaptive_eta is None else float(a.adaptive_eta); pcm = pc_mult_of(t)
            eta = eta_now() if a.adaptive_eta is not None else None
            if a.curv_per_shell:
                active_ = jnp.asarray(active, bool)
                tr_ = jnp.where(active_, jnp.asarray(trace), 0.0)
                c_sh = jnp.zeros(n_shells, jnp.float32).at[shell_of].max(tr_)
                c_f = jnp.maximum(c_sh[shell_of], jnp.float32(floor))
                trace_max = jnp.max(tr_)
                g = jnp.where(active_[:, None], jnp.asarray(gradient), 0)
                scale_f = jnp.float32(lr) / c_f if eta is None else jnp.float32(lr) * eta[shell_of] / c_f
                velocity = jnp.where(trace_max > 0, g * scale_f[:, None], jnp.zeros_like(g)).astype(theta.dtype)
                diag = {"curvature_trace_max": float(trace_max), "curvature_scalar": float(trace_max),
                        "curv_per_shell_min": float(jnp.min(jnp.where(active_, c_f, jnp.inf))),
                        "gradient_l2": float(jnp.linalg.norm(gradient)), "active_frequency_count": int(jnp.sum(active_))}
            else:
                _, velocity, diag = inner_sched(theta, previous, gradient, trace, active, learning_rate=lr, floor=floor)
            if pcm != 1.0:
                velocity = jnp.asarray(velocity).at[:, 1:].multiply(jnp.float32(pcm))
            diag = dict(diag)
            if eta is not None:
                r_ = int(np.max(np.where(np.asarray(active), np.rint(radii), 0)))
                diag["eta_shell"] = [round(float(x), 4) for x in np.asarray(eta)[: r_ + 1]]
            if t in trh_log:
                act = jnp.asarray(active, bool); trv = jnp.where(act, jnp.asarray(trace), 0.0)
                r_ = int(np.max(np.where(np.asarray(act), np.rint(radii), 0)))
                cnt = jnp.zeros(n_shells, jnp.float32).at[shell_of].add(act.astype(jnp.float32))
                diag["trh_shell_mean"] = [float(x) for x in np.asarray(jnp.zeros(n_shells, jnp.float32).at[shell_of].add(trv) / jnp.maximum(cnt, 1))[: r_ + 1]]
                diag["trh_shell_max"] = [float(x) for x in np.asarray(jnp.zeros(n_shells, jnp.float32).at[shell_of].max(trv))[: r_ + 1]]
                diag["trh_c"] = ([float(x) for x in np.maximum(np.asarray(jnp.zeros(n_shells, jnp.float32).at[shell_of].max(trv))[: r_ + 1], floor)]
                                 if a.curv_per_shell else float(diag["curvature_scalar"]))
            diag.update(lr_mean_used=float(lr), lr_pc_used=float(lr * pcm), curv_per_shell=bool(a.curv_per_shell),
                                           velocity_l2_before_projection=float(jnp.linalg.norm(velocity)))
            return theta + velocity, velocity, diag

        iteration_loop.momentum_step = sched_step
    if a.tau2_file:  # round 17: the shrink and the logs; nothing else changes (momentum 0, the same G and c)
        if a.momentum != 0 or config.optimizer != "momentum_sgd":
            raise ValueError("--tau2-file needs --momentum 0 and momentum_sgd")
        import jax.numpy as jnp
        tau2 = np.load(a.tau2_file).astype(np.float64)
        shell_np = np.rint(radii).astype(np.int32)
        n_sh = int(shell_np.max()) + 1
        if len(tau2) < n_sh or np.any(tau2 < 0) or not np.isfinite(tau2).all():
            raise ValueError(f"tau2 file has {len(tau2)} shells, need {n_sh}, finite and non-negative")
        shell_j = jnp.asarray(shell_np)
        tau2_f = jnp.asarray(np.maximum(tau2, 1e-30)[shell_np], jnp.float32)
        batch_fraction = a.batch / n_images
        print("ROUND17_SHRINK", json.dumps({"tau2_file": str(a.tau2_file), "tau2_sha256": sha256(a.tau2_file), "b_over_N": batch_fraction}), flush=True)
        inner_simple = iteration_loop.momentum_step
        ema17 = {"m": None}
        sum_sh = lambda x: jnp.zeros(n_sh, jnp.float32).at[shell_j].add(x)

        def simple_step(theta, previous, gradient, trace, active, *, learning_rate, floor):
            t = clock["update"]
            proposed, velocity, diag = inner_simple(theta, previous, gradient, trace, active, learning_rate=learning_rate, floor=floor)
            c = float(diag["curvature_scalar"]); lr = float(learning_rate)
            shrink = 1 / (1 + jnp.float32(lr * batch_fraction / c) / tau2_f)
            new = jnp.asarray(proposed).at[:, 1:].multiply(shrink[:, None].astype(jnp.complex64))
            act = jnp.asarray(active, bool)
            g = jnp.where(act[:, None], jnp.asarray(gradient), 0)
            ema17["m"] = 0.1 * g if ema17["m"] is None else 0.9 * ema17["m"] + 0.1 * g  # logged only, never used in the update
            cnt = jnp.maximum(sum_sh(act.astype(jnp.float32)), 1)
            m_p, g_p = jnp.abs(ema17["m"]) ** 2, jnp.abs(g) ** 2
            A0, B0 = sum_sh(m_p[:, 0]) / cnt, sum_sh(g_p[:, 0]) / cnt
            Ap, Bp = sum_sh(jnp.sum(m_p[:, 1:], axis=1)) / cnt, sum_sh(jnp.sum(g_p[:, 1:], axis=1)) / cnt
            W0 = jnp.asarray(theta)[:, 1:]
            r_ = int(np.max(np.where(np.asarray(active), np.rint(radii), 0)))
            cut = lambda x: [float("%.5g" % v) for v in np.asarray(x)[: r_ + 1]]
            rec = dict(t=t, lr=lr, c=c, r=r_, shrink_mean=float(jnp.mean(jnp.where(act, shrink, 1))), A_mean=cut(A0), B_mean=cut(B0),
                       v_mean=cut(A0 / jnp.maximum(B0 / 19, 1e-38) - 1), A_pc=cut(Ap), B_pc=cut(Bp), v_pc=cut(Ap / jnp.maximum(Bp / 19, 1e-38) - 1),
                       W_dot_G=cut(sum_sh(jnp.real(jnp.sum(jnp.conj(W0) * g[:, 1:], axis=1)))), pc_power=cut(sum_sh(jnp.sum(jnp.abs(W0) ** 2, axis=1)) / cnt),
                       mean_power=cut(sum_sh(jnp.abs(jnp.asarray(theta)[:, 0]) ** 2) / cnt))
            with open(a.output / "v_log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            return new, new - jnp.asarray(theta), dict(diag, shrink_mean=rec["shrink_mean"])

        iteration_loop.momentum_step = simple_step
    hp0 = config.stages[0][2]
    pc_cap = {
        False: a.pc_cap_fraction * supported_shell(hp0),
        True: a.pc_cap_fraction * supported_shell(hp0 + max(config.oversampling, 0)),
    }
    if a.pc_cap_auto:
        if len({s[2] for s in config.stages}) != 1:
            raise ValueError("--pc-cap-auto assumes one HEALPix order")
        a.pc_radius = pc_cap[False]
    noise_ess = {"f0": None}
    if a.noise_ess:
        fraction, hold, decay = (float(x) for x in a.noise_ess.split(","))

        def ess_inflation(self, iteration):
            f0 = noise_ess["f0"]
            if "noise" in bb_parts:
                if iteration <= hold:
                    return float(f0)
                prior = 5000 * 0.9999**iteration
                w = prior / (prior + 50 + 300 * (iteration - hold))
                return float(w * f0 + (1 - w))
            return float(f0 + (1.0 - f0) * min(1.0, max(0.0, (iteration - hold) / decay)))

        Config.noise_inflation = ess_inflation
    if a.pc_radius is not None or a.rewhiten:
        import jax.numpy as jnp

        stock_bandlimit = iteration_loop.bandlimit_and_mask
        pc_keep = jnp.asarray(radii <= (a.pc_radius if a.pc_radius is not None else 1e9))
        pc_keep_os = jnp.asarray(radii <= pc_cap[True])
        calls = {"n": 0}

        def post_step(theta, volume_shape, radius, mask_arg):
            # Called twice per momentum-SGD update: the velocity, then the model. Both are band-limited; only
            # the model is re-whitened.
            theta = stock_bandlimit(theta, volume_shape, radius, mask_arg)
            calls["n"] += 1
            is_model = calls["n"] % 2 == 0
            if a.rewhiten and is_model and "stats" in batch:
                z = np.concatenate([np.asarray(s.embeddings, np.float64) for s in batch["stats"]])
                q = z.shape[1]
                trace = np.mean([s.diagnostics["latent_covariance_trace_mean"] for s in batch["stats"]])
                mean_z = z.mean(axis=0)
                cov = np.cov(z.T, bias=True).reshape(q, q) + (trace / q) * np.eye(q)
                root = np.linalg.cholesky(np.eye(q) + a.rewhiten * (cov - np.eye(q)))
                mu = theta[:, 0] + a.rewhiten * (theta[:, 1:] @ jnp.asarray(mean_z, jnp.float32))
                theta = jnp.concatenate([mu[:, None], theta[:, 1:] @ jnp.asarray(root, jnp.float32)], axis=-1)
                theta = theta.astype(jnp.complex64)
            if a.pc_radius is not None and clock["update"] <= a.pc_until:
                keep = pc_keep
                if a.vdam_sampling and a.pc_cap_auto:
                    S = vdam["sampling"]
                    keep = jnp.asarray(radii <= a.pc_cap_fraction * supported_shell(S.healpix_order + S.adaptive_oversampling))
                elif a.pc_cap_auto and iteration_loop.oversampled_update(config, clock["update"]):
                    keep = pc_keep_os
                theta = jnp.concatenate([theta[:, :1], jnp.where(keep[:, None], theta[:, 1:], 0)], axis=-1)
            return theta.astype(jnp.complex64)

        iteration_loop.bandlimit_and_mask = post_step
    if a.pc_hold:
        import dataclasses as _dc
        import jax.numpy as jnp

        if config.optimizer not in ("momentum_sgd", "vdam"):
            raise ValueError("--pc-hold needs momentum_sgd or vdam")
        hold = {"init": None, "trigger": None, "pmax": [], "update": None, "calls": 0}
        inner_groups, inner_post = iteration_loop.expectation_groups, iteration_loop.bandlimit_and_mask

        def hold_groups(dataset, state, config_, halves, iteration, **kwargs):
            if hold["init"] is None:  # the first E-step reads the update-0 checkpoint: the run's initial PCs
                hold["init"] = jnp.asarray(state.theta)[:, 1:]
            if hold["trigger"] is None or iteration < hold["trigger"]:
                theta = jnp.asarray(state.theta)
                state = _dc.replace(state, theta=theta.at[:, 1:].set(0))  # update 1 reads the initial PCs otherwise
            stats = inner_groups(dataset, state, config_, halves, iteration, **kwargs)
            if hold["trigger"] is None and not batch.get("infer"):
                n = sum(s_.n_images for s_ in stats)
                hold["pmax"].append(sum(s_.diagnostics["pmax_mean"] * s_.n_images for s_ in stats) / n)
                w = a.os_pmax_window
                rolling = float(np.mean(hold["pmax"][-w:])) if len(hold["pmax"]) >= w else None
                if (rolling is not None and rolling >= a.pc_hold_pmax) or iteration + 1 >= a.pc_hold_until:
                    hold["trigger"] = iteration + 1
                    (a.output / "pc_hold.json").write_text(json.dumps(
                        {"trigger": iteration + 1, "rolling_pmax": rolling,
                         "by": "pmax" if rolling is not None and rolling >= a.pc_hold_pmax else "update"}) + "\n")
            return stats

        def hold_post(theta, volume_shape, radius, mask_arg):
            theta = inner_post(theta, volume_shape, radius, mask_arg)
            t = clock["update"]
            if hold["update"] != t:
                hold["update"], hold["calls"] = t, 0
            hold["calls"] += 1
            is_model = hold["calls"] == 2  # momentum SGD: the velocity, then the model
            if hold["trigger"] is not None and t >= hold["trigger"]:
                out = theta
            elif hold["trigger"] == t + 1 and is_model:  # last held update: the model leaves with the initial PCs
                out = theta.at[:, 1:].set(hold["init"].astype(theta.dtype))
            else:
                out = theta.at[:, 1:].set(0)
            if is_model:
                with open(a.output / "pc_hold.jsonl", "a") as f:
                    f.write(json.dumps({"update": t, "pc_norm": float(jnp.linalg.norm(out[:, 1:])),
                                        "mean_norm": float(jnp.linalg.norm(out[:, 0])), "trigger": hold["trigger"]}) + "\n")
            return out.astype(jnp.complex64)

        iteration_loop.expectation_groups = hold_groups
        if config.optimizer == "momentum_sgd":
            iteration_loop.bandlimit_and_mask = hold_post
        else:
            # VDAM (owner, 2026-10-29: "PCs off" during lock-in): one stochastic_update per update, so the hold is applied
            # to its output (before the stock band-limit). While held, the PC columns of theta and of the first moments
            # are zero; at the last held update the PCs leave with the run's update-0 values (fresh first moments).
            from relax.ppca_initial_model.update import Moments as _Moments
            inner_update = iteration_loop.stochastic_update

            def hold_update(theta, moments, *args_, **kw):
                theta_, moments_, diag = inner_update(theta, moments, *args_, **kw)
                t = clock["update"]
                if hold["trigger"] is None or t < hold["trigger"]:
                    pcs = hold["init"].astype(theta_.dtype) if hold["trigger"] == t + 1 else 0
                    theta_ = jnp.asarray(theta_).at[:, 1:].set(pcs)
                    moments_ = _Moments(jnp.asarray(moments_.first).at[:, :, 1:].set(0), moments_.second, moments_.initialized)
                with open(a.output / "pc_hold.jsonl", "a") as f:
                    f.write(json.dumps({"update": t, "pc_norm": float(jnp.linalg.norm(jnp.asarray(theta_)[:, 1:])),
                                        "mean_norm": float(jnp.linalg.norm(jnp.asarray(theta_)[:, 0])), "trigger": hold["trigger"]}) + "\n")
                return theta_, moments_, diag

            iteration_loop.stochastic_update = hold_update
    a.output.mkdir(parents=True, exist_ok=a.resume)
    if band:
        (a.output / "grid_band.json").write_text(json.dumps(band, indent=1) + "\n")
    (a.output / ("arm_resume.json" if a.resume else "arm.json")).write_text(
        json.dumps(
            {
                "argv": sys.argv[1:],
                "config": dict(config.__dict__),
                "device": jax.devices()[0].device_kind,
                "manifest_sha256": identity["manifest_sha256"],
                "eval_ids_sha256": None if suite else sha256(a.eval_ids),
                "heldout_ids": held.tolist(),
                "script_sha256": sha256(__file__),
            },
            indent=1,
            default=str,
        )
        + "\n"
    )

    def expectation_of(state, ids):
        parts = []
        merge = iteration_loop._merge_statistics

        def capture(group_parts):
            parts.extend(group_parts)
            return merge(group_parts)

        iteration_loop._merge_statistics = capture
        try:
            stats = iteration_loop.expectation(data, state, config, ids, max(state.iteration, 1))
        finally:
            iteration_loop._merge_statistics = merge
        got = np.concatenate([np.asarray(part.original_image_ids) for part in parts])
        if not np.array_equal(got, ids):
            raise RuntimeError("E-step did not return the requested particles in order")
        return parts, stats

    def infer(state):
        if suite:
            return
        batch["infer"] = True
        parts, stats = expectation_of(state, eval_ids)
        parts_h, stats_h = expectation_of(state, held)
        batch["infer"] = False
        np.savez_compressed(
            a.output / f"inference_{state.iteration:04d}.npz",
            particle_ids=eval_ids,
            z=np.concatenate([np.asarray(part.embeddings) for part in parts]).astype(np.float32),
            rotation_idx=np.concatenate([part.diagnostics["best_rotation_idx"] for part in parts]),
            translation_idx=np.concatenate([part.diagnostics["best_translation_idx"] for part in parts]),
            pmax=np.concatenate([part.diagnostics["max_posterior_per_image"] for part in parts]),
            theta=np.asarray(state.theta),
            noise=np.asarray(state.noise),
            latent_covariance_trace_mean=stats.diagnostics["latent_covariance_trace_mean"],
            log_likelihood_per_image=stats.log_likelihood / len(eval_ids),
            healpix_order=config.stage(max(state.iteration, 1))[1],
            radius=min(config.stage(max(state.iteration, 1))[0], data.grid_size // 2 - 1),
            heldout_ids=held,
            heldout_pmax=np.concatenate([part.diagnostics["max_posterior_per_image"] for part in parts_h]),
            heldout_log_likelihood_per_image=stats_h.log_likelihood / len(held),
            heldout_latent_covariance_trace_mean=stats_h.diagnostics["latent_covariance_trace_mean"],
        )

    started = time.monotonic()
    diameter = manifest["particle_diameter_ang"]
    state = None if a.resume else iteration_loop.run(data, config, a.output, identity, diameter, stop_after=0, log_direction_prior=False)
    if a.noise_ess:
        # Bisection on log F0 for the mean rotation ESS share of the update-1 E-step on 300 training images.
        probe_ids = np.sort(pool[~np.isin(pool, held)][:300])
        trials = []

        def ess_share(factor):
            probe.update(on=True, ess=[])
            try:
                iteration_loop.expectation(
                    data, state, config, probe_ids, 1, embeddings_only=True, diameter_ang=diameter, noise_inflation=factor
                )
            finally:
                probe["on"] = False
            share = float(np.mean(np.concatenate(probe["ess"])))
            trials.append({"factor": factor, "ess_share": share})
            return share

        lo, hi = np.log(1 / 64), np.log(4096)
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            target = fraction if fraction < 1 else fraction / sampling.rotation_grid_size(config.stage(1)[1])
            lo, hi = (mid, hi) if ess_share(float(np.exp(mid))) < target else (lo, mid)
        noise_ess["f0"] = float(np.exp(0.5 * (lo + hi)))
        (a.output / "noise_ess.json").write_text(
            json.dumps({"f0": noise_ess["f0"], "target": fraction, "hold": hold, "decay": decay, "trials": trials}) + "\n"
        )

    def log_schedule(milestone):
        os_on = bool(iteration_loop.oversampled_update(config, max(milestone, 1)))
        radius, hp = config.stage(max(milestone, 1))
        radius = min(radius, data.grid_size // 2 - 1)
        record = {
            "update": milestone,
            "os_on": os_on,
            "os_onset": onset["update"],
            "pc_cap_shell": (pc_cap[os_on] if a.pc_cap_auto else a.pc_radius) if a.pc_radius is not None else None,
            "pass1_image_size": int(iteration_loop.pass1_image_size(hp, data, 2 * radius, diameter)) if os_on else 2 * radius,
            "pass2_image_size": 2 * radius,
            "offset_rms_px": float(np.sqrt(np.mean(np.sum(offsets**2, axis=1)))),
            "offset_mean_px": offsets.mean(axis=0).tolist(),
            "offset_variance_px2": float(state.offset_variance),
            "noise_inflation": float(config.noise_inflation(max(milestone, 1))),
            "radius": int(radius),
            "minibatch": int(config.schedule(max(milestone, 1), n_images)[0]),
            "learning_rate": clock.get("lr"),
            "band_diameter_ang": band.get("diameter_ang"),
            "healpix_order": int(hp),
        }
        if a.vdam_sampling:
            S = vdam["sampling"]
            record.update(
                healpix_order=S.healpix_order, offset_step_px=S.offset_step_px, offset_range_px=S.offset_range_px,
                acc_rot=S.acc_rot, acc_trans_angstrom=S.acc_trans_angstrom, resolution_shell=vdam.get("shell"),
                uniform_prior=S.uniform_local_orientation_prior,
                pc_cap_shell=a.pc_cap_fraction * supported_shell(S.healpix_order + S.adaptive_oversampling),
            )
        if a.recentre:
            np.save(a.output / f"offsets_{milestone:04d}.npy", offsets)
        with open(a.output / "driver_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    if (a.init_pc_power is not None or a.init_theta is not None) and not a.resume:
        first = a.output / "checkpoint_0000.npz"
        with np.load(first, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        theta0 = arrays["theta"]
        match = {}
        if a.init_theta is not None:
            if a.init_theta.suffix == ".npz":
                with np.load(a.init_theta, allow_pickle=False) as src:
                    new = src["theta"].astype(np.complex64)
                    carried = {k: src[k] for k in a.init_carry.split(",") if k}
            else:
                new, carried = np.load(a.init_theta).astype(np.complex64), {}
            if new.shape != theta0.shape:
                raise ValueError(f"--init-theta shape {new.shape} != model {theta0.shape}")
            if "noise" in carried:
                # Noise is stored per radial shell; a fresh start holds shells 0..box/2 (33 at box 64) and relax pads it to
                # the statistics' shell count (46, to the corner) at the first update, so a 46-shell spectrum (VDAM's, and
                # any trained PPCA run's) is taken as it is: same shell index, same layout as a trained run's checkpoint.
                arrays["noise"] = carried.pop("noise").astype(arrays["noise"].dtype)
            for k, v in carried.items():
                if k not in arrays or arrays[k].shape != v.shape:
                    raise ValueError(f"--init-carry {k}: shape {v.shape} does not fit this run's {arrays.get(k, np.zeros(0)).shape}")
                arrays[k] = v.astype(arrays[k].dtype)
            # One factor for mean and PCs: the file's mean gets the power of this run's own standard random-start mean
            # (lead, 2026-10-27); the sign makes the two means correlate positively. The file's PC-to-mean ratio is kept.
            if a.init_raw:
                theta0, match = new, {"init_raw": True, "init_carried": sorted(carried)}
            else:
                own, mine = theta0[:, 0], new[:, 0]
                factor = float(np.sqrt(np.sum(np.abs(own) ** 2) / np.sum(np.abs(mine) ** 2)))
                sign = 1.0 if float(np.real(np.vdot(mine, own))) >= 0 else -1.0
                theta0 = (new * np.float32(sign * factor)).astype(np.complex64)
                match = {"init_theta_scale_factor": factor, "init_theta_sign": sign}
        if a.init_pc_power is not None:
            p = np.sum(np.abs(theta0) ** 2, axis=0)
            theta0 = theta0.copy()
            theta0[:, 1:] *= np.float32(np.sqrt(a.init_pc_power * p[0] / p[1:].sum()))
        arrays["theta"] = theta0.astype(np.complex64)
        np.savez(first, **arrays)
        state = dataclasses.replace(state, theta=jnp.asarray(arrays["theta"]))
        p = np.sum(np.abs(arrays["theta"]) ** 2, axis=0)
        (a.output / "init_override.json").write_text(json.dumps({
            "init_pc_power": a.init_pc_power, "init_theta": str(a.init_theta) if a.init_theta else None,
            "pc_to_mean_power": float(p[1:].sum() / p[0]), **match}) + "\n")
    milestones = sorted({int(x) for x in a.milestones.split(",")} | {a.stop_after}) if a.milestones else \
        sorted({*range(a.eval_every, a.stop_after + 1, a.eval_every), a.stop_after})
    done_at = 0
    if a.resume:
        done_at = max([0] + [m for m in milestones if (a.output / f"checkpoint_{m:04d}.npz").exists()])
        print(json.dumps({"resume_from": done_at}), flush=True)
    else:
        log_schedule(0)
        infer(state)
    previous = a.output / f"checkpoint_{done_at:04d}.npz"
    for milestone in [m for m in milestones if done_at < m <= a.stop_after]:
        state = iteration_loop.run(
            data, config, a.output, identity, diameter, resume=previous, stop_after=milestone, log_direction_prior=False
        )
        previous = a.output / f"checkpoint_{milestone:04d}.npz"
        log_schedule(milestone)
        infer(state)
        print(json.dumps({"seed": a.seed, "update": milestone, "wall_s": time.monotonic() - started}), flush=True)
    stats = jax.devices()[0].memory_stats() or {}
    (a.output / "DONE").write_text(
        json.dumps({"wall_s": time.monotonic() - started, "peak_bytes_in_use": stats.get("peak_bytes_in_use")}) + "\n"
    )


if __name__ == "__main__":
    main()
