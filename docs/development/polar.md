# Running relax jobs on Polar from Della

Keep the development checkout and agent on Della. Stage a frozen checkout and
explicit data files over SSH, then submit on Polar with `sbatch`. Polar jobs do
not use Della's `/scratch/gpfs` paths or its `cryoem` partition.

## Filesystems and access

| Purpose | Path | Visible from |
| --- | --- | --- |
| Immutable source snapshots and environments | `/scratch/universal/mg6942/relax-polar` | Math login, Polar login and compute nodes |
| Input files, run outputs, caches and logs | `/scratch/network/mg6942/relax-polar` | Polar login and compute nodes |

These are different filesystems. Math's `/scratch/network` is not Polar's
`/scratch/network`, and Math cannot access the latter. `/scratch/universal` is
the NFS share advertised in Math's login banner and mounted on Polar. Polar's
compute-node visibility and writes to both paths were checked with Slurm job
`413238`. Neither scratch location is an archive; save durable results elsewhere.

SSH alias `polar` already uses the Math jump host. Math can reach GitHub, while
Polar's login and compute nodes timed out reaching it. Fetch new packages on
Math into `/scratch/universal` when needed, or transfer a lock-matched packed
environment from Della as below. Do not put installation downloads inside a
Polar computation job or run scientific computations on Math's login host.

## One-time environment setup per `pixi.lock`

Use a Della pixi environment whose `pixi.lock` hash equals this checkout's.
For the initial setup, the environment at
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relax_env_recovar_5514ac6/.pixi/envs/default`
has the same lock hash as this checkout and contains RECOVAR commit
`5514ac6e2cb63ce1e9d1d88662f80dc4a4a8e70c`. `bootstrap.py` checks the
lock hash when given `--env`, packages the environment, transfers the archive,
and submits a CPU Slurm install job:

```bash
python scripts/polar/bootstrap.py \
  --env /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relax_env_recovar_5514ac6/.pixi/envs/default
```

For an already packed archive, pass `--archive /path/to/archive.tar.gz` instead.
`conda-pack` needs `--ignore-editable-packages --ignore-missing-files` for this
pixi environment because uv overlays some conda-managed files. The install job
unpacks to `/scratch/universal`, runs `conda-unpack`, and checks that relax
imports from the staged source, JAX from the installed environment, and
RECOVAR has its pinned commit. The `.polar-ready` marker is written only after
those checks pass. The installer also handles the verified `lib/python3.1`
alias collision in this packed environment; any other tar error stops it.
Keep the archive hash and install log. On a new lock, create
or find its exact Della pixi environment before repeating this step.

## Submit a job

The smoke job stages one small JSON input and runs a float32 JAX E-step on a
Polar A100:

```bash
python scripts/polar/submit.py scripts/polar/smoke.sbatch \
  --input scripts/polar/fixtures/residuals.json
```

The submitter hashes every tracked and unignored untracked file in the Della
checkout, copies it to a new immutable source directory, and verifies all
source checksums on Polar. Each explicit input file is copied by checksum into
a content-addressed data directory on Polar's `/scratch/network`. The Slurm
script receives `POLAR_SOURCE_ROOT`, `POLAR_ENV_ROOT`, `POLAR_INPUT_ROOT`, and
`POLAR_RUN_ROOT`. Submit from Della using `ssh polar`; `sbatch` runs on Polar's
login node, and Python runs only in the assigned Slurm job. The command prints
the job ID and log paths. A `submission.json` receipt and the job's outputs
stay in its unique run directory.

For a new workload, add or adapt a `.sbatch` file in the checkout and pass its
relative path to `submit.py`, along with each required data file via `--input`.
Input basenames must be unique. The job reads them from `POLAR_INPUT_ROOT` and
writes only to `POLAR_RUN_ROOT`. Stage data by value; never refer to Della
paths from inside a Polar job. Large input transfers use resumable `rsync`, and
the submitter verifies their SHA256 hashes after transfer.

The same submitter supports a standalone RECOVAR checkout. Run
`bootstrap.py --checkout /path/to/recovar --env /path/to/recovar/.pixi/envs/default`
for its own lock, then call `submit.py --checkout /path/to/recovar JOB.sbatch`
with a Slurm script in that checkout. This route uses a separate environment
key and source snapshot; the initial end-to-end check below covers relax.

The CPU EM fast guard needs RELION's native binding. Stage the RELION `src`
directory from Della, keeping its `src` name and the checksum path printed by
the staging command:

```bash
python scripts/polar/stage_dir.py /scratch/gpfs/GILLES/mg6942/relion/src --name relion
python scripts/polar/submit.py scripts/polar/build_relion_bind.sbatch \
  --set POLAR_RELION_SRC_DIR=/scratch/universal/mg6942/relax-polar/deps/relion/SOURCE_SHA/src
python scripts/polar/submit.py scripts/polar/guard.sbatch \
  --set RECOVAR_RELION_BIND_BUILD_DIR=/scratch/network/mg6942/relax-polar/runs/BUILD_RUN_ID/relion_bind
```

Use the `polar_path` printed by `stage_dir.py` and the `run_root` printed by the
build submitter for those two example arguments. The build job records the
shared object's SHA256 in `binding.sha256`. Rebuild after changes to the
RELION or binding source. The guard creates a private copy of the submitted
source because its existing runner expects the environment under
`.pixi/envs/default`.

## K1 local replay with staged real fixtures

Stage the two checksum-pinned sets from the committed fixture manifest. This
copies their original bytes to Polar's `/scratch/network`; the command verifies
the Della files before transfer and the Polar files afterwards. Keep the
`fixture_path` from its JSON output.

```bash
python scripts/polar/stage_fixture_sets.py k1_5k128_data k1_5k128_relion_os0
python scripts/polar/stage_git_bundle.py
```

The second command places committed Git history on `/scratch/universal` and
prints its bundle path and SHA256. The parity runner checks commit ancestry, so
its job restores this history in a private copy of the submitted source. The
job also makes private STAR files with Della absolute references rewritten to
the staged network paths. `fixture-relocation.json` records each changed text
file and its old and new hashes; raw data and oracle bytes stay untouched.

Polar lacks `nvcc`. The first A100 build used Della's CUDA 12.8 toolkit,
packed as a 137 MB minimal archive and copied through Math to
`/scratch/universal/mg6942/relax-polar/toolchains/cuda-12.8-minimal.tar.gz`
(SHA256 `0be47ce0abff78240b1dc63cc0dfcd8fa45327a51eb7cb34a0eb2ead4b738d99`).
The archive contains `bin`, `nvvm`, CUDA headers, driver stubs and CUDA runtime
link libraries. The CPU Slurm build job verifies and unpacks it on universal
scratch, then builds RECOVAR and relax CUDA libraries for `sm_80`. It copies the
RELION binding from a verified prior build and records all binary and source
hashes. Do not load the build toolkit into a GPU run; the packed pixi
environment supplies its runtime libraries.

```bash
python scripts/polar/submit.py scripts/polar/build_cuda_natives.sbatch \
  --set POLAR_CUDA_ARCHIVE=/scratch/universal/mg6942/relax-polar/toolchains/cuda-12.8-minimal.tar.gz \
  --set POLAR_CUDA_ARCHIVE_SHA256=0be47ce0abff78240b1dc63cc0dfcd8fa45327a51eb7cb34a0eb2ead4b738d99 \
  --set POLAR_RELION_BIND_BUILD_DIR=/scratch/network/mg6942/relax-polar/runs/BUILD_RUN_ID/relion_bind

python scripts/polar/submit.py scripts/polar/k1_local_replay.sbatch \
  --set POLAR_FIXTURE_ROOT=FIXTURE_PATH \
  --set POLAR_GIT_BUNDLE=BUNDLE_PATH \
  --set POLAR_GIT_BUNDLE_SHA256=BUNDLE_SHA256 \
  --set POLAR_NATIVE_ROOT=/scratch/network/mg6942/relax-polar/runs/NATIVE_RUN_ID/natives
```

Replace the upper-case placeholders with the paths and hashes printed by the
staging and build commands. The replay job verifies fixture and binary hashes,
uses one Slurm-assigned A100, and runs the float32 K1 iteration 6→7 local
replay against the RELION oracle. Its log, pytest output, quality ledger, and
run-local adapted fixture manifest all live under the printed `run_root` on
Polar `/scratch/network`. This is one cross-cluster integration case; it does
not replace the full K1 and exactly K4 quality or performance qualification.

The first run on 2026-09-29 used source `675e19d`, RECOVAR `5514ac6`, the
packed lock `12fa63045fb8`, and one Polar A100. Native build `413246` passed;
K1 replay `413247` passed in 1:34, with FSC-AUC 0.99999996 or higher for both
halves (floor 0.99954) and mean |dPmax| 5.98e-05 (bound 0.0035). CPU guard
`413248` passed 102 tests. Full replay evidence is in
`/scratch/network/mg6942/relax-polar/runs/13c8c62d5954`, with a Della copy at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/polar_runs/13c8c62d5954`.
The job's 4.95 GB peak host RSS and 1:34 elapsed time support the script's
subsequent 32 GB and 10 minute requests.

Polar's partition is `main`; the initial A100 probe used
`--gres=gpu:nvidia_a100-pcie-40gb:1`. Request realistic CPU, memory and time
limits for each job. Preserve Slurm's `CUDA_VISIBLE_DEVICES` setting inside the
allocation. The existing Della `pixi run test-*` submitters hardcode Della
paths and the `cryoem` partition, so they require a Polar adapter before use.
Polar A100s have 40 GB, and Polar V100/P100 nodes have 16 GB. Treat Polar
scientific scores and timings as separate hardware evidence until matched
controls and required native libraries have been established there.

Check status and logs from Della:

```bash
ssh polar 'squeue -u "$USER"'
ssh polar 'sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList'
ssh polar 'cat /scratch/network/mg6942/relax-polar/runs/RUN_ID/slurm-JOB_ID.out'
```

To copy a finished run back to Della without removing Polar's copy:

```bash
python scripts/polar/fetch.py RUN_ID "$FETCH_ROOT/polar_runs/RUN_ID"   # FETCH_ROOT: a Della directory you can write
```

The [Slurm `sbatch` manual](https://slurm.schedmd.com/sbatch.html) describes
resource and export flags. The [Princeton Slurm guide](https://researchcomputing.princeton.edu/support/knowledge-base/slurm)
explains the login-node/compute-node boundary and job logs; Polar's local
partition and storage paths above were verified directly on its nodes.
