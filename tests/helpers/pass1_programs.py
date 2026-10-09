"""Test isolation for the jitted pass-1 programs of ``relax.scoring.pass1_program``."""

from relax.scoring import pass1_program


def clear_pass1_programs(request) -> None:
    """Clear the compiled pass-1 programs now and when the test ends.

    A test that patches a scorer the programs trace must neither reuse a program compiled
    before the patch nor leave one compiled with the patched scorer for later tests.
    """

    def clear():
        pass1_program._coarse_pass1_blocks.clear_cache()
        pass1_program._coarse_pass1_block.clear_cache()

    clear()
    request.addfinalizer(clear)
