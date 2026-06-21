# Adjoint Solver

A Julia codebase for solving a nonlinear PDE inverse problem and exploring the
numerical methods behind it — forward solvers, gradient computation (adjoint,
automatic differentiation, finite differences), optimization, and
regularization.

## Repository layout

```
Adjoint_solver/
├── src/        clean implementations of the methods explored
├── notebooks/  explanatory derivations and method comparisons
└── analysis/   deeper analysis of individual methods and techniques
```

### `src/` — implementations

The clean, reusable implementations of every method explored in the project.

- `problem.jl` — the PDE problem definition and residual `F`.
- `solvers.jl` — Newton solver for the forward problem (preconditioned Krylov).
- `optimization.jl` — optimizers (gradient descent, Adam, L-BFGS).
- `regularization.jl` — regularizers and their gradients.
- `utils.jl` — Fourier derivatives, norms, and helpers.
- `gradient/` — gradient methods: `adjoint.jl` (adjoint gradient),
  `simple.jl` (finite differences and automatic differentiation),
  `rules.jl` (differentiation rules for AD packages).
- `applications/` — problems and solvers around them
  (`inverse_problem.jl`, `eigenvalue_problem.jl`).

### `notebooks/` — derivations and comparisons

Explanatory notebooks that derive the methods and run simple
test-comparisons between the different approaches.

- `forward_problem.ipynb`
- `inverse_problem.ipynb`
- `eigenvalue_problem.ipynb`

### `analysis/` — deeper analysis

Standalone scripts that go deeper into individual methods and techniques.

- `gradient_computation.jl`
- `optimizers.jl`
- `regularization.jl`
- `hessian.jl`