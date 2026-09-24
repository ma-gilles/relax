"""``relax build_cuda``: build the EM CUDA library ``librelax_cuda.so`` from ``relax/cuda``.

The library includes RECOVAR's public CUDA headers (``recovar.cuda_build.include_dir()``), so each shared device
helper has one implementation. ``RELAX_CUDA_LIB`` selects the library at run time. The build writes a temporary
file beside the output and renames it over the output, recording the source digest in
``<output>.sources.sha256`` (``recovar.cuda_build.NativeLibrary.build``), so a process using the old library is
not disturbed.
"""

import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="output path of librelax_cuda.so (default: relax/cuda/librelax_cuda.so)")
    args = parser.parse_args()
    from relax.cuda import kernels

    output = pathlib.Path(args.output or kernels._LIBRARY.make_dir / "librelax_cuda.so").expanduser().resolve()
    kernels._LIBRARY.build(output_path=output, force=True)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
