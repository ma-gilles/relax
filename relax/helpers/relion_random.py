"""The random-number streams RELION draws from, reproduced exactly.

RELION 5.0.1 uses two generators on the refinement paths relax reproduces:

* the C library ``rand()`` behind ``init_random_generator`` / ``rnd_unif``
  (funcs.cpp:570-591), which on Linux is glibc's additive-feedback
  ``random_r`` (TYPE_3, degree 31, separation 3);
* ``std::mt19937`` fed to libstdc++'s ``std::shuffle``
  (``Experiment::randomiseParticlesOrder``, exp_model.cpp:406-456).

Both are reimplemented here without the C library or RELION, so RELION's
particle orders and perturbation draws are available with the binding absent.
``relax.relion_bind`` is the unit-test oracle
(``tests/unit/test_relion_random_vs_relion_bind.py``).
"""

from __future__ import annotations

import numpy as np

RAND_MAX = 2147483647
_MASK32 = 0xFFFFFFFF


class GlibcRand:
    """glibc ``srand(seed)`` then ``rand()`` (stdlib/random_r.c, TYPE_3)."""

    def __init__(self, seed: int):
        seed = int(seed) & _MASK32
        if seed == 0:
            seed = 1
        state = [seed]
        word = seed
        for _ in range(1, 31):
            hi, lo = divmod(word, 127773)
            word = 16807 * lo - 2836 * hi
            if word < 0:
                word += 2147483647
            state.append(word)
        self._state = [value & _MASK32 for value in state]
        self._front = 3
        self._rear = 0
        for _ in range(310):
            self._next()

    def _next(self) -> int:
        state = self._state
        value = (state[self._front] + state[self._rear]) & _MASK32
        state[self._front] = value
        self._front = (self._front + 1) % 31
        self._rear = (self._rear + 1) % 31
        return value >> 1

    def rand(self) -> int:
        return self._next()

    def rand_array(self, count: int) -> np.ndarray:
        return np.fromiter((self._next() for _ in range(int(count))), dtype=np.int64, count=int(count))


def rnd_unif(generator: GlibcRand, low: float = 0.0, high: float = 1.0) -> np.float32:
    """RELION ``rnd_unif(a, b)``: ``a + (float) rand() / (float) (RAND_MAX / (b - a))`` in float."""

    low_f = np.float32(low)
    high_f = np.float32(high)
    if low_f == high_f:
        return low_f
    denominator = np.float32(np.float32(RAND_MAX) / np.float32(high_f - low_f))
    return np.float32(low_f + np.float32(np.float32(generator.rand()) / denominator))


def rnd_unif_sequence(seed: int, count: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """``init_random_generator(seed)`` then ``count`` draws of ``rnd_unif(low, high)``."""

    if int(seed) < 0:
        raise ValueError("RELION seeds a negative random_seed from the clock; pass a nonnegative seed")
    generator = GlibcRand(seed)
    return np.asarray([rnd_unif(generator, low, high) for _ in range(int(count))], dtype=np.float32)


class MT19937:
    """``std::mt19937(seed)``: 32-bit outputs of the standard Mersenne Twister."""

    _N, _M = 624, 397

    def __init__(self, seed: int):
        state = np.empty(self._N, dtype=np.uint64)
        state[0] = int(seed) & _MASK32
        for i in range(1, self._N):
            previous = int(state[i - 1])
            state[i] = (1812433253 * (previous ^ (previous >> 30)) + i) & _MASK32
        self._state = state.astype(np.uint32)
        self._outputs: list[int] = []
        self._position = 0

    def _twist(self) -> None:
        state = self._state.astype(np.uint64)
        n, m = self._N, self._M
        lag = n - m

        def twisted(i, partner):
            y = (state[i] & 0x80000000) | (state[(i + 1) % n] & 0x7FFFFFFF)
            return state[partner] ^ (y >> 1) ^ ((y & 1) * 0x9908B0DF)

        # The in-place recurrence reads state[i + m - n] once i >= n - m: an entry
        # rewritten ``lag`` steps earlier, so blocks of ``lag`` are independent.
        for start in range(0, n, lag):
            i = np.arange(start, min(start + lag, n))
            state[i] = twisted(i, (i + m) % n)
        y = state.copy()
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        self._state = state.astype(np.uint32)
        self._outputs = (y & _MASK32).tolist()
        self._position = 0

    def __call__(self) -> int:
        if self._position >= self._N:
            self._twist()
        value = self._outputs[self._position]
        self._position += 1
        return value

    def _prime(self) -> None:
        if not self._outputs:
            self._twist()


def _uniform_int(generator: MT19937, low: int, high: int) -> int:
    """libstdc++ ``uniform_int_distribution<unsigned long>{low, high}(mt19937)``.

    The generator's range is exactly ``2^32 - 1``, so a span below it takes
    Lemire's nearly divisionless ``_S_nd`` reduction with 64-bit products.
    """

    span = high - low
    if span >= _MASK32:
        raise ValueError("std::shuffle spans beyond 2^32 - 1 are not reproduced")
    bound = span + 1
    product = generator() * bound
    low_bits = product & _MASK32
    if low_bits < bound:
        threshold = ((-bound) & _MASK32) % bound
        while low_bits < threshold:
            product = generator() * bound
            low_bits = product & _MASK32
    return (product >> 32) + low


def std_shuffle(values: np.ndarray, generator: MT19937) -> np.ndarray:
    """libstdc++ ``std::shuffle(first, last, g)`` applied in place to a 1-D array.

    Ranges whose square fits in the generator's range draw two swap positions
    per call (``__gen_two_uniform_ints``); longer ranges draw one per element.
    """

    n = int(values.shape[0])
    if n <= 1:
        return values
    generator._prime()
    items = values.tolist()
    if _MASK32 // n >= n:
        i = 1
        if n % 2 == 0:
            j = _uniform_int(generator, 0, 1)
            items[i], items[j] = items[j], items[i]
            i += 1
        while i != n:
            swap_range = i + 1
            x = _uniform_int(generator, 0, swap_range * (swap_range + 1) - 1)
            first, second = divmod(x, swap_range + 1)
            items[i], items[first] = items[first], items[i]
            i += 1
            items[i], items[second] = items[second], items[i]
            i += 1
    else:
        for i in range(1, n):
            j = _uniform_int(generator, 0, i)
            items[i], items[j] = items[j], items[i]
    values[:] = items
    return values


def shuffled_orders(sizes, seed: int) -> list[np.ndarray]:
    """``std::shuffle`` of ``0..size-1`` for each size in turn with one ``mt19937(seed)``."""

    generator = MT19937(int(seed))
    return [std_shuffle(np.arange(int(size), dtype=np.int64), generator) for size in sizes]


__all__ = ["RAND_MAX", "GlibcRand", "MT19937", "rnd_unif", "rnd_unif_sequence", "shuffled_orders", "std_shuffle"]
