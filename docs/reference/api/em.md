# relax

Expectation-Maximization algorithms for pose refinement and
heterogeneous reconstruction.

## states

Reference state containers for homogeneous and heterogeneous EM.

Source: [`tests/oracles/states.py`](../../../tests/oracles/states.py), a test oracle (imported by tests as `oracles.states`).

## iterations

High-level EM loop orchestration and convergence tracking.

Source: [`tests/oracles/iterations.py`](../../../tests/oracles/iterations.py), a test oracle (imported by tests as `oracles.iterations`).

## core

Core EM iteration logic: cross-correlation, residual computation.

Source: [`tests/oracles/core.py`](../../../tests/oracles/core.py), a test oracle (imported by tests as `oracles.core`).

## e_step

E-step: posterior probability computation over poses and translations.

Source: [`tests/oracles/e_step.py`](../../../tests/oracles/e_step.py), a test oracle (imported by tests as `oracles.e_step`).

## m_step

M-step: volume update via weighted backprojection.

Source: [`tests/oracles/m_step.py`](../../../tests/oracles/m_step.py), a test oracle (imported by tests as `oracles.m_step`).
