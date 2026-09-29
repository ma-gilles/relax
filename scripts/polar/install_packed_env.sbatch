#!/bin/bash
#SBATCH --job-name=relax-polar-env
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
: "${POLAR_ENV_ARCHIVE:?set the transferred environment archive}"
: "${POLAR_ENV_ROOT:?set the destination environment path}"
: "${POLAR_SOURCE_ROOT:?set the synced checkout path}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ -e "$POLAR_ENV_ROOT" ]]; then
    echo "Environment destination already exists: $POLAR_ENV_ROOT" >&2
    exit 1
fi
staging="${POLAR_ENV_ROOT}.unpack-${SLURM_JOB_ID}"
mkdir -p "$staging"
if ! tar -xzf "$POLAR_ENV_ARCHIVE" -C "$staging" 2>"$staging/tar.err"; then
    # This pixi environment has a python3.1 -> python3.11 alias. conda-pack
    # records some conda files under the alias, so tar first creates a directory
    # there and cannot replace it with the later symlink entry.
    if [[ $(wc -l < "$staging/tar.err") -ne 2 ]] || \
       ! sed -n '1p' "$staging/tar.err" | grep -q '^tar: lib/python3\.1: Cannot create symlink to ' || \
       ! sed -n '2p' "$staging/tar.err" | grep -qx 'tar: Exiting with failure status due to previous errors'; then
        cat "$staging/tar.err" >&2
        exit 1
    fi
    test -d "$staging/lib/python3.1" && test ! -L "$staging/lib/python3.1"
    rsync -a --ignore-existing "$staging/lib/python3.1/" "$staging/lib/python3.11/"
    mv "$staging/lib/python3.1" "$staging/lib/python3.1.packed-files"
    ln -s python3.11 "$staging/lib/python3.1"
fi
mv -T "$staging" "$POLAR_ENV_ROOT"
"$POLAR_ENV_ROOT/bin/conda-unpack"
export PATH="$POLAR_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$POLAR_SOURCE_ROOT"
python - <<'PY'
import json
import os
from pathlib import Path
import tomllib
import jax
import recovar

source = Path(os.environ["POLAR_SOURCE_ROOT"]).resolve()
env = Path(os.environ["POLAR_ENV_ROOT"]).resolve()
assert Path(jax.__file__).resolve().is_relative_to(env), jax.__file__
pixi = tomllib.loads((source / "pixi.toml").read_text())
project = pixi["workspace"]["name"]
if project == "relax":
    import relax

    assert Path(relax.__file__).resolve().is_relative_to(source), relax.__file__
    direct_url = next(Path(recovar.__file__).resolve().parents[1].glob("recovar-*.dist-info")) / "direct_url.json"
    actual_commit = json.loads(direct_url.read_text())["vcs_info"]["commit_id"]
    expected_commit = pixi["feature"]["pinned"]["pypi-dependencies"]["recovar"]["rev"]
    assert actual_commit == expected_commit, (actual_commit, expected_commit)
    print("relax", relax.__file__)
    print("recovar", recovar.__file__, actual_commit)
elif project == "recovar":
    assert Path(recovar.__file__).resolve().is_relative_to(source), recovar.__file__
    print("recovar", recovar.__file__)
else:
    raise ValueError(f"unsupported pixi project: {project}")
print("jax", jax.__version__, jax.devices())
PY
printf '%s\n' "$(date -u +%FT%TZ)" > "$POLAR_ENV_ROOT/.polar-ready"
