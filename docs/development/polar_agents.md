# Polar jobs: agent quick guide

Use this for relax or RECOVAR work developed on Della and run on Polar. Keep
the agent and editable checkout on Della. `ssh polar` goes through Math; submit
compute with Polar `sbatch`, never on a login host. Polar has no internet.

| What | Polar path | Rule |
| --- | --- | --- |
| Frozen code, packed environments, toolchains | `/scratch/universal/mg6942/relax-polar` | Visible on Math and Polar; use Math for downloads. |
| Input data, outputs, caches, logs | `/scratch/network/mg6942/relax-polar` | Visible on Polar login and compute nodes. Put **all job I/O** here. |

Math's `/scratch/network` is a different filesystem; Della's `/scratch/gpfs`
is not mounted on Polar. Never leave either path inside a Polar job input.
Both Polar scratch filesystems are temporary; fetch results to Della or a
durable store when the job finishes.

## Submit from Della

1. Use a pixi environment matching the checkout's `pixi.lock`. For a new lock,
   install it on Polar once; the installer uses Polar Slurm. Download new
   dependencies on Math into `/scratch/universal`, or pack the lock-matched
   Della environment.
2. Put a `.sbatch` script inside the checkout. Start from
   [smoke.sbatch](../../scripts/polar/smoke.sbatch). Use Polar `--partition=main`,
   request the needed GPU model and realistic CPU, memory and time, and leave
   Slurm's `CUDA_VISIBLE_DEVICES` intact. Della's `cryoem` partition rule does
   not apply on Polar. Read code from `$POLAR_SOURCE_ROOT` and input data from
   `$POLAR_INPUT_ROOT`; write outputs and caches to `$POLAR_RUN_ROOT`.
3. Submit from Della:

   ```bash
   CHECKOUT=/home/mg6942/relax  # use the RECOVAR checkout for standalone RECOVAR jobs
   # Only if this pixi.lock has no ready Polar environment:
   python /home/mg6942/relax/scripts/polar/bootstrap.py \
     --checkout "$CHECKOUT" --env /della/path/to/lock-matched/.pixi/envs/default
   python /home/mg6942/relax/scripts/polar/submit.py JOB.sbatch \
     --checkout "$CHECKOUT" --input /della/path/input-file
   ```

   Replace `JOB.sbatch` and the input path with the actual workload files.
   Repeat `--input` for files. The submitter verifies an immutable code
   snapshot on `/scratch/universal`, copies each input by SHA256 to Polar
   `/scratch/network`, and prints the job ID and run directory. The job gets
   `POLAR_SOURCE_ROOT`, `POLAR_ENV_ROOT`, `POLAR_INPUT_ROOT` and
   `POLAR_RUN_ROOT`; use `--set NAME=VALUE` for other paths or settings.
   For a directory, run
   `python /home/mg6942/relax/scripts/polar/stage_dir.py /della/path/dir --name LABEL --root /scratch/network/mg6942/relax-polar/data`
   and pass its returned `polar_path` with `--set`. For the pinned EM fixtures, use
   `scripts/polar/stage_fixture_sets.py` and the [K1 replay recipe](polar.md#k1-local-replay-with-staged-real-fixtures),
   which records the relocation of STAR files containing Della paths.

## Check and retrieve

```bash
ssh polar 'squeue -u "$USER"'
ssh polar 'sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList'
python /home/mg6942/relax/scripts/polar/fetch.py RUN_ID "$FETCH_ROOT/polar_runs/RUN_ID"   # FETCH_ROOT: a Della directory you can write (global instruction file)
```

Each Polar run keeps `submission.json`, `slurm-JOB_ID.out` and
`slurm-JOB_ID.err` under its `run_root`. Record the job ID, source/lock and
input hashes, GPU model, native library hashes, result and log paths. Build
native libraries for the submitted source before GPU qualification; the
[full Polar runbook](polar.md) has the setup and verified K1 example. Compare
scientific or timing results only on matched GPU models.
