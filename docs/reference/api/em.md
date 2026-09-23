# relax

Expectation-Maximization algorithms for pose refinement and
heterogeneous reconstruction.

## states

Reference state containers for homogeneous and heterogeneous EM.

::: relax.reference.states
    options:
      members_order: source

## iterations

High-level EM loop orchestration and convergence tracking.

::: relax.reference.iterations
    options:
      members_order: source

## core

Core EM iteration logic: cross-correlation, residual computation.

::: relax.reference.core
    options:
      members_order: source

## e_step

E-step: posterior probability computation over poses and translations.

::: relax.reference.e_step
    options:
      members_order: source

## m_step

M-step: volume update via weighted backprojection.

::: relax.reference.m_step
    options:
      members_order: source
