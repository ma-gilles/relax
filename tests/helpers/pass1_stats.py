"""A :class:`~relax.scoring.pass1_results.Pass1Stats` for the tests that fake pass 1: the fields a test names, the rest ``None``."""

from relax.scoring.pass1_results import Pass1Stats


def make_pass1_stats(**fields) -> Pass1Stats:
    """``Pass1Stats`` with the given fields; every other field is ``None``."""

    unknown = set(fields) - set(Pass1Stats.__dataclass_fields__)
    if unknown:
        raise TypeError(f"Pass1Stats has no field {sorted(unknown)}")
    values = {name: None for name in Pass1Stats.__dataclass_fields__}
    values.update(fields)
    return Pass1Stats(**values)
