# PPCA ab initio study scripts

Research harness and analysis for ab initio PPCA, kept here so it isn't only on the cluster. Not production code.

- `coupled_arm.py`, `coupled_queue.py`, `queue_step.sh`, `step1_job.sbatch`: one training arm with experiment options
  (VDAM-simple, plain SGD, schedules, the PC prior shrink and per-shell gradient logging). `make_tau2.py` builds the oracle prior.
- `suite_score.py`, `direct_fit.py`, `h1_score.py`: scoring.
- `round17_simple_sgd.md`: the current design, a simpler VDAM replacement.
- `synth_check2.py`, `img_snr.py`, `real_pdb_test.py`, `protein_test.py`: SNR estimator checks and map spectra against atomic models.
- `extract.py`, `pcscale2.py`: step, fudge and PC power from run logs and checkpoints.

Paths point at the original cluster folders. Paths and options will need adapting elsewhere.
