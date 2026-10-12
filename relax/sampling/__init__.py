"""RELION's orientation and translation sampling, one topic module each (nothing is re-exported here).

- :mod:`relax.sampling.grids`: rotation and translation grids, the M-step source rotations, RotationGrid and TrialGrid;
- :mod:`relax.sampling.perturbation`: SamplingPerturbation and the perturbed trial grids;
- :mod:`relax.sampling.device_rotations`: the device-built pass-1 rotations and their provenance check;
- :mod:`relax.sampling.oversampled_grids`: oversampled children and the local angular search grid;
- :mod:`relax.sampling.oversampling`: two-pass adaptive oversampling built on them;
- :mod:`relax.sampling.healpix`, :mod:`relax.sampling.symmetry`, :mod:`relax.sampling.orientation_priors`,
  :mod:`relax.sampling.translation_prior`, :mod:`relax.sampling.tomo_particles` and the expected-accuracy trial orders.
"""
