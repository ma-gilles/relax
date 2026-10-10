"""glibc ``srand``/``rand`` against the C library, and RELION's ``init_random_generator`` seed range."""

import ctypes
import ctypes.util

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.numerics import relion_random
from relax.relion.input_particle_table import relion_class3d_seed_classes

pytestmark = pytest.mark.unit


def _libc():
    name = ctypes.util.find_library("c")
    libc = ctypes.CDLL(name) if name else None
    if libc is None or not hasattr(libc, "gnu_get_libc_version"):
        pytest.skip("glibc is required as the reference")
    return libc


@pytest.mark.parametrize("seed", [0, 1, 42, 1791421219, 2**31 - 1, 2**31, 2**31 + 5, 2**32 - 1])
def test_glibc_rand_matches_libc_for_every_unsigned_seed(seed):
    # srandom_r holds the seed as int32_t and divides with C truncation: seeds of 2**31 and above are
    # negative words (glibc stdlib/random_r.c). The generator, the sequence and the vectorised first draw
    # are one port.
    libc = _libc()
    libc.srand(ctypes.c_uint(seed))
    expected = np.asarray([libc.rand() for _ in range(400)], dtype=np.int64)
    generator = relion_random.GlibcRand(seed)
    assert_matches(np.asarray([generator.rand() for _ in range(400)], dtype=np.int64), expected)
    assert_matches(relion_random.glibc_rand_sequence(seed, 400), expected)
    assert_matches(relion_random.glibc_first_rand([seed]), expected[:1])
    # rand_array continues the same stream as rand(), from any position.
    chunked = relion_random.GlibcRand(seed)
    parts = [np.asarray([chunked.rand() for _ in range(7)]), chunked.rand_array(0), chunked.rand_array(150)]
    parts += [np.asarray([chunked.rand()]), chunked.rand_array(242)]
    assert_matches(np.concatenate(parts), expected)
    assert chunked.draws == 400


@pytest.mark.parametrize("seed", [-1, 2**31, 2**32 - 1])
def test_init_random_generator_refuses_seeds_relion_takes_from_the_clock(seed):
    # init_random_generator(int seed) reseeds from time(NULL) for a negative int (funcs.cpp:570-576); a seed of
    # 2**31 or more is a negative int in RELION.
    with pytest.raises(ValueError, match="init_random_generator"):
        relion_random.init_random_generator(seed)
    with pytest.raises(ValueError, match="init_random_generator"):
        relion_random.rnd_unif_sequence(seed, 1)


def test_class3d_seed_classes_refuse_seed_sums_beyond_int():
    # init_random_generator(random_seed + part_id) (ml_optimiser.cpp:4632): the last particle's sum overflows.
    with pytest.raises(ValueError, match="init_random_generator"):
        relion_class3d_seed_classes(np.arange(3), 2**31 - 2, 2)
    relion_class3d_seed_classes(np.arange(3), 2**31 - 3, 2)
