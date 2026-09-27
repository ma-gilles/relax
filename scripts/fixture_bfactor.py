"""One total B-factor for relax's synthetic EM fixtures.

The fixture scripts build their source maps with a known baked-in B (0 for the atomic-model and
trajectory maps they build themselves, ``ASSET_BAKED_BFACTOR`` for recovar's bundled
``assets/vol*.mrc``), and recovar's EM-development preset adds the rest on projection. The preset's
``atomic_bfactor`` (``--atomic-bfactor``) is the requested TOTAL B; ``resolve_total_bfactor``
turns it into the preset's increment and a record for the fixture metadata. recovar's own
defaults (``generate_trajectory_volumes`` Bfactor=80, ``simulate_data`` Bfactor=100, preset
``DEFAULT_ATOMIC_BFACTOR`` = 100) stack when used directly; the fixture scripts never rely on them.
"""

from __future__ import annotations

from recovar.simulation import solvent_contrast

# recovar's bundled assets/vol*.mrc are 5nrl maps already multiplied by exp(-100 |q|^2 / 4)
# (make_trajectories.ipynb; recovar docs/math/atomic_solvent_contrast.md, 3910ec587).
ASSET_BAKED_BFACTOR = 100.0


def resolve_total_bfactor(atomic_volume_kwargs: dict | None, *, baked_in: float, source: str) -> tuple[dict, dict]:
    """Return the preset kwargs that give the requested total B, and the B record for the metadata.

    ``atomic_volume_kwargs`` are the preset options (``atomic_solvent_correction``,
    ``solvent_contrast_a``, ``solvent_contrast_B``, ``atomic_bfactor``); ``atomic_bfactor`` there is
    the requested total. Omitted, the total is the source's baked-in B when it has one (so the preset adds
    only the solvent term) and recovar's ``DEFAULT_ATOMIC_BFACTOR`` for unblurred sources. A total below the
    source's baked-in B cannot be reached and is refused. With the preset off, the total is the
    baked-in B and an explicit different total is refused.
    """
    kwargs = {"atomic_solvent_correction": True} if atomic_volume_kwargs is None else dict(atomic_volume_kwargs)
    baked_in = float(baked_in)
    requested = kwargs.get("atomic_bfactor")
    if not kwargs.get("atomic_solvent_correction"):
        if requested is not None and float(requested) != baked_in:
            raise ValueError(
                f"total B {requested} A^2 needs the EM-development preset; the {source} maps carry {baked_in} A^2"
            )
        kwargs.pop("atomic_bfactor", None)
        total, added = baked_in, 0.0
    else:
        if requested is None:
            total = baked_in if baked_in > 0 else float(solvent_contrast.DEFAULT_ATOMIC_BFACTOR)
        else:
            total = float(requested)
        if total < baked_in:
            raise ValueError(
                f"requested total B {total} A^2 is below the {baked_in} A^2 already baked into the {source} maps"
            )
        added = total - baked_in
        kwargs["atomic_bfactor"] = added
    record = {
        "total_bfactor_A2": total,
        "baked_in_bfactor_A2": baked_in,
        "preset_atomic_bfactor_A2": added,
        "source": source,
        "convention": "RELION exp(-B |q|^2 / 4), q in cycles/A; total = baked-in (source maps) + preset atomic_bfactor",
    }
    return kwargs, record
