"""``relax build_relion_bind``: build the RELION pybind11 binding (``relax.relion_bind``).

Needs ``RELION_SRC_DIR`` (the pinned RELION 5.0.1 ``src``). ``RECOVAR_RELION_BIND_BUILD_DIR`` selects an external
build directory; see ``relax/relion_bind/build.py``.
"""


def main():
    from relax.relion_bind import build

    build.build()


if __name__ == "__main__":
    main()
