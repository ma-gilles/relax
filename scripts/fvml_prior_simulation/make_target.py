"""Write one target of the FvML-mixture prior simulation (``simulation.Target``) to an ``.npz`` file.

Three kinds, chosen by the target's entry in ``config.yaml``:
  * ``mixture``: a known FvML mixture (weights with the uniform one first, lobe directions as RELION (rot, tilt) in
    degrees, concentrations): the ground truth of a recovery test; the fits use its number of components.
  * ``refine_dir``: the learned direction prior (``pdf_direction``) of a finished relax/RELION refinement, from its
    last iteration's two half-set model STARs (their average), summed to HEALPix ``order``.
  * ``particle_star``: the histogram of a particle STAR's viewing directions at ``order``, expanded by ``symmetry``
    (a point group such as C3, which the fits then use).
  * ``kent_population``: the closed-form viewing-direction distribution of a pose-prior benchmark Kent scenario
    (``kappa, beta, mu, mu0`` as in its config) at ``order``: the smooth population a refinement's learned prior
    estimates.

Run (Snakemake calls it as rule make_target):
    python scripts/fvml_prior_simulation/make_target.py --config scripts/fvml_prior_simulation/config.yaml \
        --target gt_three_lobes --out targets/gt_three_lobes.npz
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relax.sampling.orientation_priors import infer_direction_prior_healpix_order  # noqa: E402
from relax.sampling.symmetry import rotational_operators  # noqa: E402
from scripts.fvml_prior_simulation.simulation import Target, coarsen, save_target  # noqa: E402
from relax.sampling.fvml_mixture import FvmlMixture  # noqa: E402
from scripts.pose_prior_benchmark.evaluate_run import read_direction_prior  # noqa: E402
from scripts.pose_prior_benchmark.star_nonuniformity import (  # noqa: E402
    direction_counts,
    expand_directions,
    read_viewing_directions,
)
from scripts.pose_prior_benchmark.true_prior import kent_population_prior  # noqa: E402

DEFAULT_LOBES = 5  # stored lobe count of a target without a known one; the fits take theirs from the setting


def direction_from_rot_tilt(rot_deg: float, tilt_deg: float) -> np.ndarray:
    """RELION's viewing direction of (rot, tilt) in degrees as a unit vector."""
    rot, tilt = math.radians(rot_deg), math.radians(tilt_deg)
    return np.array([math.cos(rot) * math.sin(tilt), math.sin(rot) * math.sin(tilt), math.cos(tilt)])


def last_prior_iteration(refine_dir: Path) -> int:
    """The last iteration whose two half-set model STARs exist."""
    found = [
        int(m.group(1)) for p in refine_dir.glob("run_it*_half1_model.star") if (m := re.search(r"it(\d+)", p.name))
    ]
    complete = [i for i in found if (refine_dir / f"run_it{i:03d}_half2_model.star").is_file()]
    if not complete:
        raise FileNotFoundError(f"no run_itNNN_half{{1,2}}_model.star pair in {refine_dir}")
    return max(complete)


def run_prior(refine_dir: Path, order: int) -> tuple[np.ndarray, str]:
    """The average of the two halves' learned direction priors at the last iteration, summed to ``order``."""
    iteration = last_prior_iteration(refine_dir)
    priors = []
    for half in (1, 2):
        prior = read_direction_prior(refine_dir / f"run_it{iteration:03d}_half{half}_model.star")
        if prior is None:
            raise ValueError(f"{refine_dir} iteration {iteration} half {half} has no direction prior")
        priors.append(prior)
    native = infer_direction_prior_healpix_order(priors[0])
    if native < order:
        raise ValueError(f"{refine_dir} iteration {iteration} has a prior at order {native}, below {order}")
    pmf = coarsen(0.5 * (priors[0] + priors[1]), native - order)
    return pmf / pmf.sum(), f"{refine_dir} iteration {iteration} (order {native}, halves averaged)"


def build_target(name: str, spec: dict[str, Any], config: dict[str, Any]) -> Target:
    """The target named ``name`` from its config entry ``spec``."""
    symmetry_name = spec.get("symmetry", "C1")
    symmetry = rotational_operators(symmetry_name)
    eval_order = int(spec.get("eval_order", config["eval_order"]))
    if "mixture" in spec:
        m = spec["mixture"]
        means = np.array([direction_from_rot_tilt(rot, tilt) for rot, tilt in m["directions_deg"]])
        mixture = FvmlMixture(m["weights"], means, m["kappas"], symmetry)
        return Target(name, symmetry, len(m["weights"]), eval_order, "mixture", mixture=mixture)
    order = int(spec["order"])
    n_components = int(spec.get("n_lobes", DEFAULT_LOBES)) + 1
    if "refine_dir" in spec:
        pmf, source = run_prior(Path(spec["refine_dir"].format(**config)), order)
    elif "particle_star" in spec:
        star = Path(spec["particle_star"].format(**config))
        counts = direction_counts(expand_directions(read_viewing_directions(star), symmetry_name), order)
        pmf, source = counts / counts.sum(), f"{star} expanded by {symmetry_name}"
    elif "kent_population" in spec:
        kent = spec["kent_population"]
        pmf = kent_population_prior(kent["kappa"], kent["beta"], kent["mu"], kent["mu0"], order)
        source = f"Kent population kappa {kent['kappa']} beta {kent['beta']} mu {kent['mu']} mu0 {kent['mu0']}"
    else:
        raise ValueError(f"target {name}: needs mixture, refine_dir, particle_star or kent_population")
    return Target(name, symmetry, n_components, eval_order, source, pmf=pmf, order=order)


def main() -> None:
    """Parse arguments, build the target and write it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    target = build_target(args.target, config["targets"][args.target], config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_target(target, args.out)
    print(f"{args.target}: {target.source}, {target.n_components} components, symmetry order {len(target.symmetry)}")


if __name__ == "__main__":
    main()
