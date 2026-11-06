# PPCA ab initio study: current recipe and standard

This folder holds the PPCA ab-initio study's working standard and its current best recipe, on top of this branch's
relax tree (ac8e62db, "gated momentum SGD for ppca_initial_model").

- `BEST_PRACTICE.md`: the living standard. It covers datasets, the VDAM reference, the SGD recipe, sampling and
  translations, scoring, baselines, reporting rules, open questions and a dated change log.
- `best_v1/RECIPE.md`: the current best recipe, with its exact flags, the evidence for each choice, known limits and
  validation results so far (synthetic and real EMPIAR data).
- `best_v1/coupled_arm.py`: the driver that runs the recipe on top of relax (`--source` points at this relax tree).
  `coupled_arm_sgd.py` and `coupled_arm_shifts.py` are the two drivers it was merged from, kept for provenance.
- `best_v1/vdam_step_3000.npy`: the step-size schedule. `example_arm_bestv1.json` is one complete arm.
- `best_v1/CRYOET_PROMPT.md`: a handoff for running the recipe on cryo-ET data.
- `SHARED_RULES.md`: datasets, protocol, scoring and reporting rules for comparing methods across branches.

Status, 2026-11-06: the SGD part of the recipe is settled. The translation rule is open: the exhaustive grid with a
fixed prior was rejected for SGD on synthetic data, and a learned-prior variant is being scored. See RECIPE.md.
