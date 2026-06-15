include("../problem.jl")
include("../optimization.jl")
include("../gradient/adjoint.jl")
include("../gradient/simple.jl")
include("../solvers.jl")
include("../regularization.jl")
include("../utils.jl")

function loss(V; p::Problem, u_target, reg::Regularizer=NoReg())
    u = Newton_solve(p, V, 100)
    return (p.L/p.N) * sum((u - u_target).^2) + R(p, V, reg)
end

function inverse_problem(p::Problem, u_target; V0=ones(p.N), optimizer::Optimizer=OLBFGS(), reg::Regularizer=NoReg())
    loss_fn = V -> loss(V; p=p, u_target=u_target, reg=reg)
    grad_fn = V -> adjoint_gradient(p, V, u_target; reg=reg)[2]
    return optimize(optimizer, loss_fn, grad_fn, V0)
end


p = Problem(N=500, L=2π, f=x->cos(x), α=1.0)

u_target = sin.(2*p.x)
result, _ = inverse_problem(p, u_target; reg=NoReg(), optimizer=OLBFGS(n_iter=1000))


solution = Newton_solve(p, result, 100)
using Plots
plot(p.x, [u_target solution], label=["Target" "Result"])

V_targ = (p.f.+ diff2_fourier(u_target, p.ks) - p.α .* u_target .^3) ./ u_target

plot(p.x, [result V_targ],ylims = (-10, 10), label=["Result" "Target V"])