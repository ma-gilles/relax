"""FFI target coverage of the EM CUDA library (split from test_cuda_lib_symbol_check.py, relax split P2)."""

import pytest
from recovar import cuda_backproject as cb

from relax.cuda import library

pytestmark = pytest.mark.unit


def test_em_ffi_registrations_cover_all_em_target_constants():
    """Every EM target is eager or explicitly optional in the EM library; optional ABIs stay lazy."""
    recovar_targets = {v for k, v in vars(cb).items() if k.startswith("_TARGET_") and isinstance(v, str)}
    # _TARGET_PROJECT_INDEXED is recovar's (imported by the EM wrappers)
    em_target_constants = {
        v
        for k, v in vars(library).items()
        if k.startswith("TARGET_") and isinstance(v, str) and v not in recovar_targets
    }
    em_targets_in_table = {target for target, _symbol in library.FFI_REGISTRATIONS}
    em_optional_targets = set(library.OPTIONAL_FFI_REGISTRATIONS)
    assert em_targets_in_table.isdisjoint(em_optional_targets)
    assert em_target_constants == em_targets_in_table | em_optional_targets


def test_em_ffi_registrations_have_unique_targets_and_symbols():
    targets = [t for t, _s in library.FFI_REGISTRATIONS]
    symbols = [s for _t, s in library.FFI_REGISTRATIONS]
    assert len(set(targets)) == len(targets)
    assert len(set(symbols)) == len(symbols)
