"""``relax build_cuda``: build the EM CUDA library ``librelax_cuda.so`` from ``relax/cuda``.

The library includes RECOVAR's public CUDA headers (``recovar.cuda_build.include_dir()``), so each shared device
helper has one implementation. ``RECOVAR_RELAX_CUDA_LIB`` selects the library at run time.
"""

import argparse
import os
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="output path of librelax_cuda.so (default: relax/cuda/librelax_cuda.so)")
    args = parser.parse_args()
    cuda_dir = pathlib.Path(__file__).resolve().parents[1] / "cuda"
    output = pathlib.Path(args.output or cuda_dir / "librelax_cuda.so").expanduser().resolve()
    subprocess.check_call(
        ["make", "-C", str(cuda_dir), f"PYTHON={sys.executable}", f"LIB={output}", "all"],
        env=dict(os.environ),
    )
    print(f"Built {output}")


if __name__ == "__main__":
    main()
