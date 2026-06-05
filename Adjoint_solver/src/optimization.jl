using LinearAlgebra
using Optimisers


abstract type Optimizer end

Base.@kwdef struct OGradientDescent <: Optimizer
    n_iter :: Int     = 1000
    η      :: Float64 = 1e-2
    gtol   :: Float64 = 1e-9
end

Base.@kwdef struct OAdam <: Optimizer
    n_iter :: Int     = 1000
    η      :: Float64 = 1e-2
    gtol   :: Float64 = 1e-9
end

Base.@kwdef struct OLBFGS <: Optimizer
    n_iter :: Int     = 500
    m      :: Int     = 15
    gtol   :: Float64 = 1e-9
end

function optimize end

function optimize(o::OGradientDescent, loss, grad, V0; history_step=100)
    V = copy(V0)
    hist = Float64[]

    for i in 1:o.n_iter
        g = grad(V)
        V .-= o.η .* g
        if(i % history_step==0)
            push!(hist, loss(V))
        end
        norm(g, Inf) ≤ o.gtol && break
    end
    return V, hist
end

function optimize(o::OAdam, loss, grad, V0; history_step=100)
    V = copy(V0)
    hist = Float64[]
    state = Optimisers.setup(Optimisers.Adam(o.η), V)
    for i in 1:o.n_iter
        g = grad(V)
        state, V = Optimisers.update(state, V, g)
        if(i % history_step==0)
            push!(hist, loss(V))
        end
        norm(g, Inf) ≤ o.gtol && break
    end
    return V, hist
end


function optimize(o::OLBFGS, loss, grad, V0; history_step=1)
    g!(G, V) = (G .= grad(V))
    res = Optim.optimize(loss, g!, V0, Optim.LBFGS(m = o.m),
                         Optim.Options(iterations = o.n_iter, g_abstol = o.gtol,
                                       store_trace = true, extended_trace = false))
    return res.minimizer, [t.value for t in Optim.trace(res)]
end