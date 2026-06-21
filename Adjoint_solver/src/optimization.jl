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

# `metric` is recorded into `hist` every `history_step` iterations (default: the loss),
# so callers can track e.g. (loss, potential error) per iteration without re-running.
function optimize(o::OGradientDescent, loss, grad, V0; history_step=100, metric=loss)
    V = copy(V0)
    hist = typeof(metric(V))[]
    for i in 1:o.n_iter
        g = grad(V)
        V .-= o.η .* g
        if(i % history_step==0)
            push!(hist, metric(V))
        end
        norm(g, Inf) ≤ o.gtol && break
    end
    return V, hist
end
 
function optimize(o::OAdam, loss, grad, V0; history_step=100, metric=loss)
    V = copy(V0)
    hist = typeof(metric(V))[]
    state = Optimisers.setup(Optimisers.Adam(o.η), V)
    for i in 1:o.n_iter
        g = grad(V)
        state, V = Optimisers.update(state, V, g)
        if(i % history_step==0)
            push!(hist, metric(V))
        end
        norm(g, Inf) ≤ o.gtol && break
    end
    return V, hist
end
 
function optimize(o::OLBFGS, loss, grad, V0; history_step=1, metric=loss)
    g!(G, V) = (G .= grad(V))
    res = Optim.optimize(loss, g!, V0, Optim.LBFGS(m = o.m),
                         Optim.Options(iterations = o.n_iter, g_abstol = o.gtol,
                                       store_trace = true, extended_trace = true))
    return res.minimizer, [metric(t.metadata["x"]) for t in Optim.trace(res)]
end
 