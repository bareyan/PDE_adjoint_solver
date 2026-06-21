module Adjoint_solver

# Sibling submodules. Each `using ..Sibling` reaches a previously-defined sibling,
# so they must be declared in dependency order. The top module re-exports the public API.

# ------------------------------------------------------------------ Utils
module Utils
    include("utils.jl")                       # diff_fourier, diff2_fourier, cg_ad, l2norm, ...
    export diff_fourier, diff2_fourier, cg_ad, l2norm, linspace, infnorm
end

# ------------------------------------------------------------------ Problems
module Problems
    using FFTW
    using ..Utils: diff2_fourier
    include("problem.jl")                     # Problem, F   (sibling include removed in file)
    export Problem, F
end

# ------------------------------------------------------------------ Regularization
module Regularization
    using ..Utils: diff_fourier, diff2_fourier
    using ..Problems: Problem
    include("regularization.jl")              # (two sibling includes removed in file)
    export Regularizer, R1, R2, NoReg, R, grad_R
end

# ------------------------------------------------------------------ Solvers
module Solvers
    using Krylov, LinearOperators, FFTW, LinearAlgebra
    using ..Utils: diff2_fourier, l2norm
    using ..Problems: Problem, F
    include("solvers.jl")                     # (three sibling includes removed in file)
    export Newton_solve
end

# ------------------------------------------------------------------ Gradient
module Gradient
    using Krylov, LinearOperators, FFTW, LinearAlgebra
    using ChainRulesCore, ForwardDiff, Zygote, Mooncake
    using ..Utils: diff_fourier, diff2_fourier, cg_ad
    using ..Problems: Problem
    using ..Solvers: Newton_solve
    using ..Regularization: Regularizer, NoReg, R, grad_R
    include("gradient/adjoint.jl")            # adjoint_gradient   (sibling include removed)
    include("gradient/rules.jl")              # cg_ad / Newton_solve / R rules (sibling include removed)
    include("gradient/simple.jl")             # FD, forwarddiff/zygote/mooncake (sibling include removed)
    export adjoint_gradient, FD, forwarddiff_gradient, zygote_gradient, mooncake_gradient
end

# ------------------------------------------------------------------ Optimizers
module Optimizers
    using LinearAlgebra, Optimisers
    import Optim                               # import (not using) so `optimize` is ours, not Optim's
    include("optimization.jl")
    export Optimizer, OGradientDescent, OAdam, OLBFGS, optimize
end

# ------------------------------------------------------------------ Applications
module Applications
    using ..Problems: Problem
    using ..Solvers: Newton_solve
    using ..Regularization: R, Regularizer, NoReg
    using ..Gradient: adjoint_gradient
    using ..Optimizers: Optimizer, OLBFGS, optimize
    include("applications/inverse_problem.jl")   # library only — demo script moved out
    export loss, inverse_problem
end

# ------------------------------------------------------------------ re-export
using .Utils, .Problems, .Regularization, .Solvers, .Gradient, .Optimizers, .Applications

export Problem, F
export diff_fourier, diff2_fourier, cg_ad, l2norm
export Regularizer, R1, R2, NoReg, R, grad_R
export Newton_solve
export adjoint_gradient, FD, forwarddiff_gradient, zygote_gradient, mooncake_gradient
export Optimizer, OGradientDescent, OAdam, OLBFGS, optimize
export loss, inverse_problem

end # module Adjoint_solver